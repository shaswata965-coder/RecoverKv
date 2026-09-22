#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# RULER 32k, 13 tasks, WINDOW 64 -- otherwise the operating point of
# ruler32k_lb16_bda91b4 (window 8), so the two tables differ in one knob.
#
#   operating point  cache_budget 0.20, q 0.70 (bytes), gate 0.25,
#                    window 64, local 128 (= 2 windows), sink 5,
#                    first_eviction_step 0, capture_memory off
#   repo             THIS checkout, as it is. Nothing here pulls or checks out.
#
# Why: at window 8 niah_single_3 scored 2.40 and niah_multikey_3 21.60, with the
# model reproducing the first ~8 chars of each UUID and inventing the rest. A
# UUID is ~20+ tokens, i.e. ~3 windows at ws=8 but inside one at ws=64. If the
# 0.25 gate is opening only the first window, ws=64 should recover those tasks.
# (It is not the only difference a larger window makes -- eviction granularity
# and the int2 grid change too -- so a recovery says "window size", and
# quant_gate_ratio=1.0 at ws=8 is the run that isolates the gate.)
#
# ------------------------------------------------------------------ scheduling
# One task = one process on one GPU, all 500 examples; 8 workers pull from one
# queue, longest first (by the ws=8 wall times: 49-88 min). 13 tasks on 8 GPUs
# = two rounds, ~2.5-3 h if ws=64 runs like ws=8. Ask for >= 6 h walltime.
#
# ws=64 in the decode kernel: window_tiling() is max(1, target_keys // ws), so
# every tile rung maps to BLOCK_NW=1, BLOCK_T=64 -- the "large ws" case the tile
# search already dedups. local 128 is a multiple of 64, as required.
#
# ------------------------------------------------------------------- eviction
# STICKYKV_EVICT_CONTROL_ARM is used if this checkout has it; at bda91b4 it does
# not, so the eviction runs compiled (it held for all 13 tasks at ws=8).
#
# ------------------------------------------------------------------ resuming
# DONE = <task>.meta.json AND 500 lines. Anything else is deleted and re-run
# from scratch; RULER's own resume stays off (it keeps a 1-line crashed task as
# "done"). Re-run this script to redo only what is not done. A job refuses to
# start if HEAD moved. Do not start two copies at once.
#
# Outputs: $OUT_ROOT/<task>.{jsonl,meta.json,log}, scores.csv, summary.txt
#
# Run on the 8-GPU node:
#   cd ~/kv_cache/Cluster_QEvict/RecoverKv && conda activate sticky_env
#   nohup scripts/run_ruler32k_w64_8gpu.sh > ~/ruler32k_w64.log 2>&1 &
#   tail -f ~/ruler32k_w64.log
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
RULER_DATA="${RULER_DATA:-/home/ee/phd/eez228470/kv_cache/defensive_kv_new/DefensiveKV/defensivekv_dataset/ruler/32768}"
RULER_CONFIG="configs/ruler_niah_mk3_omega16.yaml"
# The ws=8 run to print beside this one (skipped if absent).
COMPARE_DIR="${COMPARE_DIR:-$PROJECT_ROOT/outputs/ruler32k_lb16_bda91b4/ruler}"

# Pinned -- not env-overridable.
BUDGET=0.20; QUANT_RATIO=0.70; QUANT_MODE=bytes; GATE_RATIO=0.25
WINDOW=64; LOCAL=128; SINK=5

# Longest first, by the ws=8 wall times.
TASKS=(fwe cwe niah_multiquery niah_multikey_3 qa_1 qa_2 niah_single_3
       niah_multivalue niah_multikey_2 vt niah_single_1 niah_multikey_1 niah_single_2)
EXPECT_N=500

COMMIT="$(git rev-parse --short HEAD)"; START_SHA="$(git rev-parse HEAD)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/ruler32k_w64_${COMMIT}}"

# ---- preflight -------------------------------------------------------------------
die() { echo "error: $*" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
[[ -f "$RULER_CONFIG" ]] || die "missing $RULER_CONFIG"
[[ -f "$RULER_DATA/dataset_info.json" ]] || die "RULER 32k data (save_to_disk dir) not found: $RULER_DATA"
(( LOCAL % WINDOW == 0 )) || die "local $LOCAL is not a multiple of window $WINDOW"
(( WINDOW % 4 == 0 ))     || die "window $WINDOW must be a multiple of 4 for q>0"
command -v nvidia-smi >/dev/null || die "no nvidia-smi -- login node? get a GPU node."
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"   # PBS UUIDs, used as given
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
(( ${#GPUS[@]} > 0 )) || die "no GPUs visible"
python - <<'PY' || exit 2
import sys
try:
    import torch
except ImportError:
    sys.exit("error: torch not importable -- 'conda activate sticky_env'")
if not torch.cuda.is_available():
    sys.exit("error: torch.cuda.is_available() is False -- wrong env or no GPU allocation.")
try:
    import flash_attn  # noqa: F401
except ImportError:
    sys.exit("error: flash_attn not importable; the config needs flash_attention_2.")
PY

ARM=0
grep -q '_EVICT_CONTROL_ARM_ENV = "STICKYKV_EVICT_CONTROL_ARM"' modules/windowed_cache/cache.py && ARM=1
EVICT_DESC=$([[ $ARM == 1 ]] && echo "control-arm-eager" || echo "compiled (no control arm at $COMMIT)")

mkdir -p "$OUT_ROOT"
QUEUE_POS="$OUT_ROOT/.queue_pos"; LOCK="$OUT_ROOT/.queue_lock"; FAILED="$OUT_ROOT/failed_jobs.txt"
echo 0 > "$QUEUE_POS"; : > "$FAILED"

{
    echo "commit=$START_SHA  started=$(date -Is)  host=$(hostname)  gpus=${GPUS[*]}"
    echo "evict=$EVICT_DESC  model=$MODEL_PATH  ruler_data=$RULER_DATA"
    echo "budget=$BUDGET q=$QUANT_RATIO mode=$QUANT_MODE gate=$GATE_RATIO window=$WINDOW local=$LOCAL sink=$SINK"
} > "$OUT_ROOT/run.env"

is_done() {
    [[ -f "$OUT_ROOT/$1.meta.json" && -f "$OUT_ROOT/$1.jsonl" ]] || return 1
    [[ "$(wc -l < "$OUT_ROOT/$1.jsonl")" -eq $EXPECT_N ]]
}

next_job() {
    local i
    {
        flock 9
        i=$(<"$QUEUE_POS")
        if (( i < ${#TASKS[@]} )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
    } 9>"$LOCK"
}

run_job() {    # $1 gpu, $2 task
    local gpu="$1" t="$2"
    if is_done "$t"; then echo "[$(date +%T)] gpu=$gpu SKIP   $t (complete)"; return 0; fi
    if [[ "$(git rev-parse HEAD)" != "$START_SHA" ]]; then
        echo "[$(date +%T)] gpu=$gpu REFUSE $t -- HEAD moved off $COMMIT" >&2
        echo "$t rc=HEAD-moved" >> "$FAILED"; return 1
    fi
    rm -f "$OUT_ROOT/$t.jsonl" "$OUT_ROOT/$t.meta.json" "$OUT_ROOT/$t.memory.jsonl" "$OUT_ROOT/$t.memory_summary.json"
    echo "[$(date +%T)] gpu=$gpu START  $t"
    local t0=$SECONDS rc
    (
        if [[ $ARM == 1 ]]; then export STICKYKV_EVICT_CONTROL_ARM=1; else unset STICKYKV_EVICT_CONTROL_ARM; fi
        CUDA_VISIBLE_DEVICES="$gpu" exec python main.py --config "$RULER_CONFIG" --override \
            model.name="$MODEL_PATH" \
            cache.cache_budget="$BUDGET" \
            cache.quant_ratio="$QUANT_RATIO" \
            cache.quant_budget_mode="$QUANT_MODE" \
            cache.quant_gate_ratio="$GATE_RATIO" \
            cache.window_size="$WINDOW"       window.window_size="$WINDOW" \
            cache.local_window_size="$LOCAL"  window.local_window_size="$LOCAL" \
            cache.num_sink_tokens="$SINK"     window.num_sink_tokens="$SINK" \
            cache.first_eviction_step=0 \
            ruler.data_dir="$RULER_DATA" ruler.tasks="[$t]" ruler.num_samples=max \
            ruler.resume=false ruler.capture_memory=false ruler.output_dir="$OUT_ROOT"
    ) > "$OUT_ROOT/$t.log" 2>&1
    rc=$?
    if (( rc == 0 )) && is_done "$t"; then
        echo "[$(date +%T)] gpu=$gpu DONE   $t ($(( (SECONDS - t0) / 60 )) min)"
    else
        local why
        why=$(grep -m1 -oE "ran EAGER|OutOfMemoryError|CUDA error[^.]*|[A-Za-z]+Error: [^.]{0,80}" "$OUT_ROOT/$t.log" | head -1)
        echo "[$(date +%T)] gpu=$gpu FAIL   $t rc=$rc ${why:+($why)} -- $OUT_ROOT/$t.log" >&2
        echo "$t rc=$rc ${why:-see log} $OUT_ROOT/$t.log" >> "$FAILED"
    fi
}

worker() {
    local gpu="$1" i
    while i=$(next_job) && [[ -n "$i" ]]; do run_job "$gpu" "${TASKS[$i]}"; done
}

echo "=== RULER 32k, window $WINDOW: ${#TASKS[@]} tasks on ${#GPUS[@]} GPU(s) ==="
echo "  commit $COMMIT on $(hostname) | eviction: $EVICT_DESC"
echo "  budget=$BUDGET q=$QUANT_RATIO ($QUANT_MODE) gate=$GATE_RATIO window=$WINDOW local=$LOCAL sink=$SINK"
echo "  out: $OUT_ROOT"
echo

pids=()
for g in "${GPUS[@]}"; do
    worker "$g" 2>&1 | tee -a "$OUT_ROOT/worker_${g}.log" &
    pids+=($!)
done
wait "${pids[@]}"

# ---- score + summary ---------------------------------------------------------------
echo; echo "=== scoring ==="
python -m modules.evaluation.ruler_scoring --predictions_dir "$OUT_ROOT" --out_csv "$OUT_ROOT/scores.csv" \
    > "$OUT_ROOT/scoring.log" 2>&1 || echo "scoring FAILED -- $OUT_ROOT/scoring.log" >&2

OUT_ROOT="$OUT_ROOT" COMPARE_DIR="$COMPARE_DIR" COMMIT="$COMMIT" EVICT_DESC="$EVICT_DESC" \
TASKS="${TASKS[*]}" EXPECT_N="$EXPECT_N" python - <<'PY' | tee "$OUT_ROOT/summary.txt"
import csv, json, os
E = os.environ
tasks, n_exp = E["TASKS"].split(), int(E["EXPECT_N"])

def load(d, need_meta):
    raw = {}
    if os.path.exists(f"{d}/scores.csv"):
        raw = {r["task"]: float(r["score"]) for r in csv.DictReader(open(f"{d}/scores.csv"))}
    out = {}
    for t in tasks:
        j, m = f"{d}/{t}.jsonl", f"{d}/{t}.meta.json"
        lines = sum(1 for _ in open(j)) if os.path.exists(j) else 0
        ok = os.path.exists(m) and lines == n_exp and t in raw
        gate = (json.load(open(m)).get("read_gate") or {}).get("verdict", "MISSING") if os.path.exists(m) else "-"
        out[t] = (raw[t] if ok else None, gate if ok else (f"part:{lines}" if lines else "not run"))
    return out

cur = load(E["OUT_ROOT"], True)
cmp_ = load(E["COMPARE_DIR"], True) if os.path.isdir(E["COMPARE_DIR"]) else None
print(f"RULER 32k @ {E['COMMIT']} | eviction: {E['EVICT_DESC']}")
print("budget 0.20  q 0.70 (bytes)  gate 0.25  local 128  sink 5\n")
hdr = f"{'task':18}{'ws=64':>9}" + (f"{'ws=8':>9}{'64-8':>9}" if cmp_ else "") + "   read_gate"
print(hdr)
for t in tasks:
    s, g = cur[t]
    row = f"{t:18}" + (f"{s:9.2f}" if s is not None else f"{g:>9}")
    if cmp_:
        c = cmp_[t][0]
        row += (f"{c:9.2f}" if c is not None else f"{'-':>9}")
        row += (f"{s - c:+9.2f}" if s is not None and c is not None else f"{'':>9}")
    print(row + f"   {g if s is not None else ''}")
avg = lambda r: (sum(r[t][0] for t in tasks) / len(tasks)) if all(r[t][0] is not None for t in tasks) else None
a, b = avg(cur), (avg(cmp_) if cmp_ else None)
line = f"{'average (13)':18}" + (f"{a:9.2f}" if a is not None else f"{'incompl.':>9}")
if cmp_:
    line += (f"{b:9.2f}" if b is not None else f"{'-':>9}") + (f"{a - b:+9.2f}" if None not in (a, b) else "")
print(line + "\n")
bad = [f"{t}: {cur[t][1]}" for t in tasks if cur[t][0] is None or cur[t][1] != "gated"]
if bad:
    print("!! not usable as-is (re-run the script to redo just these):")
    for x in bad: print("   ", x)
else:
    print("all 13 tasks complete, 500 examples each, gate verdict `gated` everywhere.")
PY

if [[ -s "$FAILED" ]]; then echo; echo "!! failed jobs:"; cat "$FAILED"; exit 1; fi
echo; echo "done: $OUT_ROOT/summary.txt"
