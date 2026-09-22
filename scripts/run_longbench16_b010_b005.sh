#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# LongBench, all 16 English datasets, at two operating points, one queue.
#
#   b010   cache_budget 0.10, local 128
#   b005   cache_budget 0.05, local 32
#   both   q 0.70 (bytes), gate 0.25, window 8, sink 5, first_eviction_step 0
#   repo   THIS checkout, as it is. Nothing here pulls or checks out.
#
# ------------------------------------------------------------------ scheduling
# 32 jobs (16 datasets x 2 configs), one process per GPU, pulled longest-first
# from one shared queue; a GPU that frees up takes the next job at once. Order
# from the measured b=0.10 / b=0.05 wall times of sweep_395af34 (the four
# datasets that sweep did not run -- repobench-p, passage_retrieval_en,
# passage_count, lcc -- are estimates). Simulated, +90 s model load per job:
#   4 GPUs  ~3.9 h  (perfect split 3.88 h)      8 GPUs  ~2.0 h
# Ask for >= 6 h walltime on the 4-GPU cluster. Uses every GPU the job has.
#
# ------------------------------------------------------------------- budgets
# When budget x prompt < sink + local the cache keeps sink + local anyway and
# exceeds the budget (modules/windowed_cache/config.py:532). Counted with the
# Llama-3.1 tokenizer over context+input (template excluded, so slightly high):
#   b010/local128 (floor 133 tok): 71/3750 examples -- multi_news 44 (22%),
#                                  lcc 21, 2wikimqa 4, multifieldqa_en 2
#   b005/local32  (floor  37 tok): 12/3750 examples -- all multi_news (6%)
# local 32 is a multiple of window 8, as the int-local rule requires.
#
# ------------------------------------------------------------------- eviction
# If this checkout has STICKYKV_EVICT_CONTROL_ARM (origin >= d2c45e3) it is
# used: same eviction body run eagerly, identical scores, immune to Dynamo's
# 64-graph recompile limit. At bda91b4 it does not exist: the eviction runs
# compiled. Watch the first half hour: `grep -l "ran EAGER" $OUT_ROOT/*/*.log`.
#
# ------------------------------------------------------------------ resuming
# DONE = .meta.json (written last) AND the full line count (200; 150 for
# multifieldqa_en; 500 for lcc and repobench-p). Anything else is deleted and
# re-run from scratch. Re-run this script to redo only what is not done. A job
# refuses to start if HEAD moved since the run began. Do not start two copies
# at once: they would share one queue counter.
#
# ------------------------------------------------------------------- outputs
#   $OUT_ROOT/q070_b010_w8_l128/<dataset>.{jsonl,meta.json,log}, scores.csv
#   $OUT_ROOT/q070_b005_w8_l32/ <dataset>.{jsonl,meta.json,log}, scores.csv
#   $OUT_ROOT/summary.txt  -- both budgets side by side
#
# Run on the GPU node:
#   cd ~/kv_cache/Cluster_QEvict/RecoverKv && conda activate sticky_env
#   nohup scripts/run_longbench16_b010_b005.sh > ~/lb16_b010_b005.log 2>&1 &
#   tail -f ~/lb16_b010_b005.log
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
export LONGBENCH_LOCAL_DIR="${LONGBENCH_LOCAL_DIR:-/home/ee/phd/eez228470/kv_cache/qwen_longbench_final_int2/RecoverKv/data/longbench}"
LB_CONFIG="configs/longbench_ours_flash_attn.yaml"

# Pinned -- not env-overridable, so a variable left exported from an earlier
# ablation cannot move this run.
QUANT_RATIO=0.70; QUANT_MODE=bytes; GATE_RATIO=0.25; WINDOW=8; SINK=5
CFGS=(b010 b005)
declare -A BUDGET=([b010]=0.10 [b005]=0.05)
declare -A LOCAL=([b010]=128 [b005]=32)

# Longest first (sweep_395af34 b=0.10/0.05 times; the 4 new ones estimated).
DATASETS=(gov_report multi_news samsum qmsum narrativeqa repobench-p trec triviaqa
          passage_retrieval_en passage_count musique hotpotqa qasper lcc 2wikimqa
          multifieldqa_en)
declare -A EXPECT_N=([multifieldqa_en]=150 [lcc]=500 [repobench-p]=500)

COMMIT="$(git rev-parse --short HEAD)"; START_SHA="$(git rev-parse HEAD)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/longbench16_b010_b005_${COMMIT}}"
cfg_dir() { printf '%s/q070_%s_w%s_l%s' "$OUT_ROOT" "$1" "$WINDOW" "${LOCAL[$1]}"; }

# ---- preflight -------------------------------------------------------------------
die() { echo "error: $*" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
[[ -f "$LB_CONFIG" ]]  || die "missing $LB_CONFIG"
command -v nvidia-smi >/dev/null || die "no nvidia-smi -- login node? get a GPU node."
for d in "${DATASETS[@]}"; do
    [[ -s "$LONGBENCH_LOCAL_DIR/$d.jsonl" ]] || die "missing $LONGBENCH_LOCAL_DIR/$d.jsonl (the loader would silently fall back to HF)"
done
for c in "${CFGS[@]}"; do
    (( ${LOCAL[$c]} % WINDOW == 0 )) || die "$c: local ${LOCAL[$c]} is not a multiple of window $WINDOW"
done
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

for c in "${CFGS[@]}"; do mkdir -p "$(cfg_dir "$c")"; done
QUEUE_POS="$OUT_ROOT/.queue_pos"; LOCK="$OUT_ROOT/.queue_lock"; FAILED="$OUT_ROOT/failed_jobs.txt"
echo 0 > "$QUEUE_POS"; : > "$FAILED"

# Both configs of a dataset sit together, longest dataset first.
JOBS=()
for d in "${DATASETS[@]}"; do for c in "${CFGS[@]}"; do JOBS+=("$c $d"); done; done

{
    echo "commit=$START_SHA  started=$(date -Is)  host=$(hostname)  gpus=${GPUS[*]}"
    echo "evict=$EVICT_DESC  model=$MODEL_PATH"
    echo "longbench_local_dir=$LONGBENCH_LOCAL_DIR"
    for c in "${CFGS[@]}"; do
        echo "$c: budget=${BUDGET[$c]} local=${LOCAL[$c]} q=$QUANT_RATIO mode=$QUANT_MODE gate=$GATE_RATIO window=$WINDOW sink=$SINK"
    done
} > "$OUT_ROOT/run.env"

is_done() {    # $1 cfg, $2 dataset
    local dir; dir="$(cfg_dir "$1")"
    [[ -f "$dir/$2.meta.json" && -f "$dir/$2.jsonl" ]] || return 1
    [[ "$(wc -l < "$dir/$2.jsonl")" -eq "${EXPECT_N[$2]:-200}" ]]
}

next_job() {
    local i
    {
        flock 9
        i=$(<"$QUEUE_POS")
        if (( i < ${#JOBS[@]} )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
    } 9>"$LOCK"
}

run_job() {    # $1 gpu, $2 cfg, $3 dataset
    local gpu="$1" c="$2" d="$3" dir
    dir="$(cfg_dir "$c")"
    if is_done "$c" "$d"; then
        echo "[$(date +%T)] gpu=$gpu SKIP   $c $d (complete)"; return 0
    fi
    if [[ "$(git rev-parse HEAD)" != "$START_SHA" ]]; then
        echo "[$(date +%T)] gpu=$gpu REFUSE $c $d -- HEAD moved off $COMMIT" >&2
        echo "$c $d rc=HEAD-moved" >> "$FAILED"; return 1
    fi
    rm -f "$dir/$d.jsonl" "$dir/$d.meta.json"     # nothing partial survives
    echo "[$(date +%T)] gpu=$gpu START  $c $d"
    local t0=$SECONDS rc
    (
        if [[ $ARM == 1 ]]; then export STICKYKV_EVICT_CONTROL_ARM=1; else unset STICKYKV_EVICT_CONTROL_ARM; fi
        CUDA_VISIBLE_DEVICES="$gpu" exec python main.py --config "$LB_CONFIG" --override \
            model.name="$MODEL_PATH" \
            cache.cache_budget="${BUDGET[$c]}" \
            cache.quant_ratio="$QUANT_RATIO" \
            cache.quant_budget_mode="$QUANT_MODE" \
            cache.quant_gate_ratio="$GATE_RATIO" \
            cache.window_size="$WINDOW"            window.window_size="$WINDOW" \
            cache.local_window_size="${LOCAL[$c]}" window.local_window_size="${LOCAL[$c]}" \
            cache.num_sink_tokens="$SINK"          window.num_sink_tokens="$SINK" \
            cache.first_eviction_step=0 \
            longbench.datasets="[$d]" \
            longbench.output_dir="$dir"
    ) > "$dir/$d.log" 2>&1
    rc=$?
    if (( rc == 0 )) && is_done "$c" "$d"; then
        echo "[$(date +%T)] gpu=$gpu DONE   $c $d ($(( (SECONDS - t0) / 60 )) min)"
    else
        local why
        why=$(grep -m1 -oE "ran EAGER|OutOfMemoryError|CUDA error[^.]*|[A-Za-z]+Error: [^.]{0,80}" "$dir/$d.log" | head -1)
        echo "[$(date +%T)] gpu=$gpu FAIL   $c $d rc=$rc ${why:+($why)} -- $dir/$d.log" >&2
        echo "$c $d rc=$rc ${why:-see log} $dir/$d.log" >> "$FAILED"
    fi
}

worker() {
    local gpu="$1" i
    while i=$(next_job) && [[ -n "$i" ]]; do
        # shellcheck disable=SC2086
        run_job "$gpu" ${JOBS[$i]}
    done
}

echo "=== LongBench 16 datasets x {b010/local128, b005/local32}: ${#JOBS[@]} jobs on ${#GPUS[@]} GPU(s) ==="
echo "  commit $COMMIT on $(hostname) | eviction: $EVICT_DESC"
echo "  q=$QUANT_RATIO ($QUANT_MODE) gate=$GATE_RATIO window=$WINDOW sink=$SINK"
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
for c in "${CFGS[@]}"; do
    dir="$(cfg_dir "$c")"
    python -m modules.evaluation.longbench_scoring --predictions_dir "$dir" --out_csv "$dir/scores.csv" \
        > "$dir/scoring.log" 2>&1 || echo "scoring FAILED for $c -- $dir/scoring.log" >&2
done

DIR_B010="$(cfg_dir b010)" DIR_B005="$(cfg_dir b005)" COMMIT="$COMMIT" EVICT_DESC="$EVICT_DESC" \
DATASETS="${DATASETS[*]}" python - <<'PY' | tee "$OUT_ROOT/summary.txt"
import csv, json, os
E = os.environ
ds = E["DATASETS"].split()
expect = {"multifieldqa_en": 150, "lcc": 500, "repobench-p": 500}
cfgs = [("10% / local 128", E["DIR_B010"]), ("5% / local 32", E["DIR_B005"])]
res, issues = {}, []
for lab, d in cfgs:
    raw = {}
    if os.path.exists(f"{d}/scores.csv"):
        raw = {r["dataset"]: float(r["score"]) for r in csv.DictReader(open(f"{d}/scores.csv"))}
    for x in ds:
        m, j = f"{d}/{x}.meta.json", f"{d}/{x}.jsonl"
        lines = sum(1 for _ in open(j)) if os.path.exists(j) else 0
        if os.path.exists(m) and lines == expect.get(x, 200) and x in raw:
            gate = (json.load(open(m)).get("read_gate") or {}).get("verdict", "MISSING")
            res[(lab, x)] = (raw[x], gate)
            if gate != "gated":
                issues.append(f"{lab} {x}: read_gate={gate}")
        else:
            res[(lab, x)] = (None, f"part:{lines}" if lines else "not run")
            issues.append(f"{lab} {x}: {res[(lab, x)][1]}")
print(f"LongBench @ {E['COMMIT']} | eviction: {E['EVICT_DESC']}")
print("q 0.70 (bytes)  gate 0.25  window 8  sink 5\n")
print(f"{'dataset':22}{'10%/l128':>10}{'5%/l32':>10}   read_gate")
for x in ds:
    cells, gates = "", []
    for lab, _ in cfgs:
        s, g = res[(lab, x)]
        cells += f"{s:10.2f}" if s is not None else f"{g:>10}"
        gates.append(g if s is not None else "-")
    print(f"{x:22}{cells}   {' / '.join(gates)}")
twelve = [x for x in ds if x not in ("passage_count", "passage_retrieval_en", "lcc", "repobench-p")]
for name, sub in (("average (16)", ds), ("average (12)", twelve)):
    row = ""
    for lab, _ in cfgs:
        v = [res[(lab, x)][0] for x in sub]
        row += f"{sum(v) / len(v):10.2f}" if None not in v else f"{'incompl.':>10}"
    print(f"{name:22}{row}")
print("\naverage (12) = the 12 non-code, non-passage datasets of sweep_395af34.")
print("Over budget (cache clamps to sink+local): 10%/l128 71 examples (multi_news 44,")
print("lcc 21, 2wikimqa 4, multifieldqa_en 2); 5%/l32 12 examples (all multi_news).\n")
if issues:
    print("!! not usable as-is (re-run the script to redo just these):")
    for i in issues: print("   ", i)
else:
    print("all 32 jobs complete, full example counts, gate verdict `gated` everywhere.")
PY

if [[ -s "$FAILED" ]]; then echo; echo "!! failed jobs:"; cat "$FAILED"; exit 1; fi
echo; echo "done: $OUT_ROOT/summary.txt"
