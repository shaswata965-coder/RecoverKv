#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# RULER 32k, 13 tasks, swept over WINDOW SIZE -- default 64, then 32, then 128.
# Everything else is the operating point of ruler32k_lb16_bda91b4 (window 8), so
# each column of the summary differs from that run in one knob only.
#
#   operating point  cache_budget 0.20, q 0.70 (bytes), gate 0.25, local 128,
#                    sink 5, first_eviction_step 0, capture_memory off
#   windows          WINDOWS="64 32 128" (env-overridable, run in that order)
#   repo             THIS checkout, as it is. Nothing here pulls or checks out.
#
# Why: at window 8 niah_single_3 scored 2.40 and niah_multikey_3 21.60 -- the
# model reproduces the first ~8 chars of each UUID and invents the rest. A UUID
# is ~20+ tokens: ~3 windows at ws=8, one window at ws>=32. If the 0.25 gate
# opens only part of the needle, larger windows should recover those tasks.
# A recovery points at window size (eviction granularity and the int2 grid move
# with it too); quant_gate_ratio=1.0 at ws=8 is the run that isolates the gate.
#
# ------------------------------------------------------------------ scheduling
# 39 jobs = 13 tasks x 3 windows. One task = one process on one GPU, all 500
# examples. ONE queue, window-major (every ws=64 task is claimed before any
# ws=32 one, and so on), longest task first within a window; a GPU that frees
# up takes the next job immediately, so there is no idle barrier between
# windows. At the ws=8 wall times (49-88 min/task, 14.4 GPU-h per window) that
# is ~5.5-6 h on 8 GPUs. Ask for >= 12 h walltime.
#
# ---------------------------------------------------------------- per window
#   ws=32   local 128 = 4 windows.  Tile: BLOCK_NW=2, BLOCK_T=64.
#   ws=64   local 128 = 2 windows.  Tile: BLOCK_NW=1, BLOCK_T=64.
#   ws=128  local 128 = 1 window.   Tile: BLOCK_NW=1, BLOCK_T=128 -- the first
#           geometry here with a 128-key tile. window_tiling() is
#           max(1, target_keys // ws), so nothing rounds to zero; if no rung of
#           the tile ladder fits shared memory the kernel RAISES (kernel-or-
#           error), so a misfit shows up as FAIL in the first minutes, never as
#           a quietly different kernel. NOT yet run on a GPU at this commit.
#   int2 packing (4 tokens/byte) and the gate card (per-token int8 t) accept
#   any ws % 4 == 0; the preflight checks that and local % ws == 0.
#
# ------------------------------------------------------------------- eviction
# STICKYKV_EVICT_CONTROL_ARM is used if this checkout has it; at bda91b4 it does
# not, so the eviction runs compiled (it held for all 13 tasks at ws=8).
#
# ------------------------------------------------------------------ resuming
# DONE = <task>.meta.json AND 500 lines. Anything else is deleted and re-run
# from scratch; RULER's own resume stays off (it keeps a 1-line crashed task as
# "done"). Re-run this script to redo only what is not done -- it picks up an
# existing ws=64 directory too. A job refuses to start if HEAD moved. Do not
# start two copies at once (they would share one queue counter).
#
# Outputs:
#   outputs/ruler32k_w<ws>_<commit>/<task>.{jsonl,meta.json,log}, scores.csv
#   outputs/ruler32k_windows_<commit>/summary.txt  -- ws=8 (reference) + each ws
#
# Run on the 8-GPU node:
#   cd ~/kv_cache/Cluster_QEvict/RecoverKv && conda activate sticky_env
#   nohup scripts/run_ruler32k_windows_8gpu.sh > ~/ruler32k_windows.log 2>&1 &
#   tail -f ~/ruler32k_windows.log
# A single window:   WINDOWS=64 scripts/run_ruler32k_windows_8gpu.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
RULER_DATA="${RULER_DATA:-/home/ee/phd/eez228470/kv_cache/defensive_kv_new/DefensiveKV/defensivekv_dataset/ruler/32768}"
RULER_CONFIG="configs/ruler_niah_mk3_omega16.yaml"
# The ws=8 run to print beside this one (skipped if absent).
COMPARE_DIR="${COMPARE_DIR:-$PROJECT_ROOT/outputs/ruler32k_lb16_bda91b4/ruler}"
read -r -a WINDOWS <<< "${WINDOWS:-64 32 128}"

# Pinned -- not env-overridable.
BUDGET=0.20; QUANT_RATIO=0.70; QUANT_MODE=bytes; GATE_RATIO=0.25
LOCAL=128; SINK=5

# Longest first, by the ws=8 wall times.
TASKS=(fwe cwe niah_multiquery niah_multikey_3 qa_1 qa_2 niah_single_3
       niah_multivalue niah_multikey_2 vt niah_single_1 niah_multikey_1 niah_single_2)
EXPECT_N=500

COMMIT="$(git rev-parse --short HEAD)"; START_SHA="$(git rev-parse HEAD)"
ws_dir() { printf '%s/outputs/ruler32k_w%s_%s' "$PROJECT_ROOT" "$1" "$COMMIT"; }
SWEEP_DIR="${SWEEP_DIR:-$PROJECT_ROOT/outputs/ruler32k_windows_${COMMIT}}"

# ---- preflight -------------------------------------------------------------------
die() { echo "error: $*" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
[[ -f "$RULER_CONFIG" ]] || die "missing $RULER_CONFIG"
[[ -f "$RULER_DATA/dataset_info.json" ]] || die "RULER 32k data (save_to_disk dir) not found: $RULER_DATA"
(( ${#WINDOWS[@]} > 0 )) || die "WINDOWS is empty"
for ws in "${WINDOWS[@]}"; do
    [[ "$ws" =~ ^[0-9]+$ ]] && (( ws > 0 )) || die "window '$ws' is not a positive integer"
    (( ws % 4 == 0 ))     || die "window $ws must be a multiple of 4 for q>0 (int2 packing)"
    (( LOCAL % ws == 0 )) || die "local $LOCAL is not a multiple of window $ws"
done
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

mkdir -p "$SWEEP_DIR"
for ws in "${WINDOWS[@]}"; do mkdir -p "$(ws_dir "$ws")"; done
QUEUE_POS="$SWEEP_DIR/.queue_pos"; LOCK="$SWEEP_DIR/.queue_lock"; FAILED="$SWEEP_DIR/failed_jobs.txt"
echo 0 > "$QUEUE_POS"; : > "$FAILED"

JOBS=()
for ws in "${WINDOWS[@]}"; do for t in "${TASKS[@]}"; do JOBS+=("$ws $t"); done; done

{
    echo "commit=$START_SHA  started=$(date -Is)  host=$(hostname)  gpus=${GPUS[*]}"
    echo "evict=$EVICT_DESC  model=$MODEL_PATH  ruler_data=$RULER_DATA"
    echo "windows=${WINDOWS[*]}  budget=$BUDGET q=$QUANT_RATIO mode=$QUANT_MODE gate=$GATE_RATIO local=$LOCAL sink=$SINK"
} | tee "$SWEEP_DIR/run.env" > /dev/null
for ws in "${WINDOWS[@]}"; do
    { head -2 "$SWEEP_DIR/run.env"; echo "budget=$BUDGET q=$QUANT_RATIO mode=$QUANT_MODE gate=$GATE_RATIO window=$ws local=$LOCAL sink=$SINK"; } > "$(ws_dir "$ws")/run.env"
done

is_done() {    # $1 ws, $2 task
    local d; d="$(ws_dir "$1")"
    [[ -f "$d/$2.meta.json" && -f "$d/$2.jsonl" ]] || return 1
    [[ "$(wc -l < "$d/$2.jsonl")" -eq $EXPECT_N ]]
}

next_job() {
    local i
    {
        flock 9
        i=$(<"$QUEUE_POS")
        if (( i < ${#JOBS[@]} )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
    } 9>"$LOCK"
}

run_job() {    # $1 gpu, $2 ws, $3 task
    local gpu="$1" ws="$2" t="$3" d
    d="$(ws_dir "$ws")"
    if is_done "$ws" "$t"; then echo "[$(date +%T)] gpu=$gpu SKIP   ws=$ws $t (complete)"; return 0; fi
    if [[ "$(git rev-parse HEAD)" != "$START_SHA" ]]; then
        echo "[$(date +%T)] gpu=$gpu REFUSE ws=$ws $t -- HEAD moved off $COMMIT" >&2
        echo "ws=$ws $t rc=HEAD-moved" >> "$FAILED"; return 1
    fi
    rm -f "$d/$t.jsonl" "$d/$t.meta.json" "$d/$t.memory.jsonl" "$d/$t.memory_summary.json"
    echo "[$(date +%T)] gpu=$gpu START  ws=$ws $t"
    local t0=$SECONDS rc
    (
        if [[ $ARM == 1 ]]; then export STICKYKV_EVICT_CONTROL_ARM=1; else unset STICKYKV_EVICT_CONTROL_ARM; fi
        CUDA_VISIBLE_DEVICES="$gpu" exec python main.py --config "$RULER_CONFIG" --override \
            model.name="$MODEL_PATH" \
            cache.cache_budget="$BUDGET" \
            cache.quant_ratio="$QUANT_RATIO" \
            cache.quant_budget_mode="$QUANT_MODE" \
            cache.quant_gate_ratio="$GATE_RATIO" \
            cache.window_size="$ws"           window.window_size="$ws" \
            cache.local_window_size="$LOCAL"  window.local_window_size="$LOCAL" \
            cache.num_sink_tokens="$SINK"     window.num_sink_tokens="$SINK" \
            cache.first_eviction_step=0 \
            ruler.data_dir="$RULER_DATA" ruler.tasks="[$t]" ruler.num_samples=max \
            ruler.resume=false ruler.capture_memory=false ruler.output_dir="$d"
    ) > "$d/$t.log" 2>&1
    rc=$?
    if (( rc == 0 )) && is_done "$ws" "$t"; then
        echo "[$(date +%T)] gpu=$gpu DONE   ws=$ws $t ($(( (SECONDS - t0) / 60 )) min)"
    else
        local why
        why=$(grep -m1 -oE "ran EAGER|OutOfResources|OutOfMemoryError|CUDA error[^.]*|[A-Za-z]+Error: [^.]{0,80}" "$d/$t.log" | head -1)
        echo "[$(date +%T)] gpu=$gpu FAIL   ws=$ws $t rc=$rc ${why:+($why)} -- $d/$t.log" >&2
        echo "ws=$ws $t rc=$rc ${why:-see log} $d/$t.log" >> "$FAILED"
    fi
}

worker() {
    local gpu="$1" i
    while i=$(next_job) && [[ -n "$i" ]]; do
        # shellcheck disable=SC2086
        run_job "$gpu" ${JOBS[$i]}
    done
}

echo "=== RULER 32k, windows ${WINDOWS[*]}: ${#JOBS[@]} jobs on ${#GPUS[@]} GPU(s) ==="
echo "  commit $COMMIT on $(hostname) | eviction: $EVICT_DESC"
echo "  budget=$BUDGET q=$QUANT_RATIO ($QUANT_MODE) gate=$GATE_RATIO local=$LOCAL sink=$SINK"
for ws in "${WINDOWS[@]}"; do echo "  out ws=$ws: $(ws_dir "$ws")"; done
echo

pids=()
for g in "${GPUS[@]}"; do
    worker "$g" 2>&1 | tee -a "$SWEEP_DIR/worker_${g}.log" &
    pids+=($!)
done
wait "${pids[@]}"

# ---- score + summary ---------------------------------------------------------------
echo; echo "=== scoring ==="
for ws in "${WINDOWS[@]}"; do
    d="$(ws_dir "$ws")"
    python -m modules.evaluation.ruler_scoring --predictions_dir "$d" --out_csv "$d/scores.csv" \
        > "$d/scoring.log" 2>&1 || echo "scoring FAILED for ws=$ws -- $d/scoring.log" >&2
done

WS_DIRS="$(for ws in "${WINDOWS[@]}"; do printf '%s=%s ' "$ws" "$(ws_dir "$ws")"; done)" \
COMPARE_DIR="$COMPARE_DIR" COMMIT="$COMMIT" EVICT_DESC="$EVICT_DESC" \
TASKS="${TASKS[*]}" EXPECT_N="$EXPECT_N" python - <<'PY' | tee "$SWEEP_DIR/summary.txt"
import csv, json, os
E = os.environ
tasks, n_exp = E["TASKS"].split(), int(E["EXPECT_N"])
cols = [tuple(x.split("=", 1)) for x in E["WS_DIRS"].split()]
if os.path.isdir(E["COMPARE_DIR"]):
    cols = [("8", E["COMPARE_DIR"])] + cols          # reference column first

def load(d):
    raw = {}
    if os.path.exists(f"{d}/scores.csv"):
        raw = {r["task"]: float(r["score"]) for r in csv.DictReader(open(f"{d}/scores.csv"))}
    out = {}
    for t in tasks:
        j, m = f"{d}/{t}.jsonl", f"{d}/{t}.meta.json"
        lines = sum(1 for _ in open(j)) if os.path.exists(j) else 0
        gate = (json.load(open(m)).get("read_gate") or {}).get("verdict", "MISSING") if os.path.exists(m) else None
        if os.path.exists(m) and lines == n_exp and t in raw:
            out[t] = (raw[t], gate)
        else:
            out[t] = (None, f"part:{lines}" if lines else "-")
    return out

res = {ws: load(d) for ws, d in cols}
print(f"RULER 32k @ {E['COMMIT']} | eviction: {E['EVICT_DESC']}")
print("budget 0.20  q 0.70 (bytes)  gate 0.25  local 128  sink 5\n")
print(f"{'task':18}" + "".join(f"{'ws=' + ws:>10}" for ws, _ in cols))
for t in tasks:
    print(f"{t:18}" + "".join(
        f"{res[ws][t][0]:10.2f}" if res[ws][t][0] is not None else f"{res[ws][t][1]:>10}"
        for ws, _ in cols))
def avg(ws):
    v = [res[ws][t][0] for t in tasks]
    return f"{sum(v) / len(v):10.2f}" if None not in v else f"{'incompl.':>10}"
print(f"{'average (13)':18}" + "".join(avg(ws) for ws, _ in cols) + "\n")
bad = []
for ws, _ in cols:
    for t in tasks:
        s, g = res[ws][t]
        if s is None:
            bad.append(f"ws={ws} {t}: {g if g != '-' else 'not run'}")
        elif g != "gated":
            bad.append(f"ws={ws} {t}: read_gate={g}")
if bad:
    print("!! not usable as-is (re-run the script to redo just these):")
    for x in bad: print("   ", x)
else:
    print("all tasks complete, 500 examples each, gate verdict `gated` everywhere.")
PY

if [[ -s "$FAILED" ]]; then echo; echo "!! failed jobs:"; cat "$FAILED"; exit 1; fi
echo; echo "done: $SWEEP_DIR/summary.txt"
