#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# RULER 32k (13 tasks) + LongBench (16 datasets) on one 8-GPU node, one queue.
#
#   operating point  cache_budget 0.20, q 0.70 (bytes), gate 0.25,
#                    window 8, local 128, sink 5, first_eviction_step 0
#   repo             THIS checkout, as it is. Nothing here pulls or checks out.
#
# ------------------------------------------------------------------ scheduling
# 29 jobs, each ONE process on ONE GPU: every RULER task whole (500 examples),
# every LongBench dataset whole. 8 workers pull from one queue, longest job
# first; a GPU that frees up takes the next job immediately.
#
# RULER goes first because each task is ~6 h (500 x ~43 s/example at 32k,
# from the 2026-07 cb0.20 run in Quantization_INT2_Latest -- older code and
# window 64, so treat it as rough). 13 tasks on 8 GPUs means 5 GPUs run two
# tasks back to back; the other 3 spend that second round on LongBench, which
# is ~8 GPU-h in total and fits inside it. Expected makespan ~12-13 h, set by
# the RULER tasks -- but the 43 s/example is from older code at window 64, and
# this runs window 8, so it can land well either side. Ask for 24 h of walltime:
# a RULER task cut off by walltime restarts from zero (no mid-task resume).
#
# ------------------------------------------------------------------- eviction
# If this checkout has STICKYKV_EVICT_CONTROL_ARM (origin >= d2c45e3) it is
# used: the eviction runs eagerly on purpose, identical scores, and cannot hit
# Dynamo's 64-graph recompile limit. At bda91b4 it does NOT exist, so the
# eviction runs compiled. That build has never run LongBench or RULER; the
# 2026-09-19 LongBench run on 7e68e02 died 16/16 on the recompile limit ("the
# two-tier eviction ran EAGER") and bda91b4 still forces the shapes that caused
# it. Watch the first hour: `grep -l "ran EAGER" $OUT_ROOT/*/*.log`.
#
# ------------------------------------------------------------------ resuming
# A job is DONE only if it has its .meta.json (written last) AND the full line
# count (RULER 500; LongBench 200, multifieldqa_en 150, lcc/repobench-p 500).
# Anything else is re-run from scratch. RULER's own `resume` is forced OFF: it
# skips a task if its jsonl has even ONE line, which would freeze a crashed
# task as "done" (ruler_runner.py:188). Re-run this script to redo only what
# is not done. Each job refuses to start if HEAD moved since the run began.
#
# ------------------------------------------------------------ capture_memory
# Forced OFF. The RULER config enables the per-example byte report, and at
# bda91b4 utils/cache_memory.py:288 reads `buffer_capacity`, which
# _LayerStateView no longer has: every task died after its first example
# (2026-09-22 01:36-01:40, 12/13 tasks, 0 lines kept). Scores do not need it.
#
# ------------------------------------------------------------------- outputs
#   $OUT_ROOT/ruler/<task>.{jsonl,meta.json,memory.jsonl,log}, scores.csv
#   $OUT_ROOT/longbench/<dataset>.{jsonl,meta.json,log}, scores.csv
#   $OUT_ROOT/summary.txt
#
# Run on the 8-GPU node:
#   cd ~/kv_cache/Cluster_QEvict/RecoverKv && conda activate sticky_env
#   nohup scripts/run_ruler32k_longbench16_8gpu.sh > ~/ruler_lb16.log 2>&1 &
#   tail -f ~/ruler_lb16.log
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
RULER_DATA="${RULER_DATA:-/home/ee/phd/eez228470/kv_cache/defensive_kv_new/DefensiveKV/defensivekv_dataset/ruler/32768}"
export LONGBENCH_LOCAL_DIR="${LONGBENCH_LOCAL_DIR:-/home/ee/phd/eez228470/kv_cache/qwen_longbench_final_int2/RecoverKv/data/longbench}"
RULER_CONFIG="configs/ruler_niah_mk3_omega16.yaml"
LB_CONFIG="configs/longbench_ours_flash_attn.yaml"

# Pinned -- not env-overridable, so a variable left exported from an earlier
# ablation cannot move this run.
BUDGET=0.20; QUANT_RATIO=0.70; QUANT_MODE=bytes; GATE_RATIO=0.25
WINDOW=8; LOCAL=128; SINK=5

# Longest first within each suite (RULER by the 2026-07 s/example, LongBench by
# the sweep_395af34 b=0.20 wall times; the four new ones are estimates).
RULER_TASKS=(niah_multiquery cwe niah_single_3 niah_multikey_3 vt niah_multikey_2
             fwe niah_single_1 niah_multikey_1 niah_single_2 niah_multivalue qa_1 qa_2)
LB_DATASETS=(gov_report multi_news samsum qmsum narrativeqa repobench-p trec triviaqa
             passage_retrieval_en passage_count musique hotpotqa qasper lcc 2wikimqa
             multifieldqa_en)
declare -A LB_N=([multifieldqa_en]=150 [lcc]=500 [repobench-p]=500)
RULER_N=500

COMMIT="$(git rev-parse --short HEAD)"; START_SHA="$(git rev-parse HEAD)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/ruler32k_lb16_${COMMIT}}"
R_DIR="$OUT_ROOT/ruler"; L_DIR="$OUT_ROOT/longbench"

# ---- preflight -------------------------------------------------------------------
die() { echo "error: $*" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]]          || die "model not found: $MODEL_PATH"
[[ -f "$RULER_CONFIG" && -f "$LB_CONFIG" ]] || die "missing $RULER_CONFIG or $LB_CONFIG"
[[ -f "$RULER_DATA/dataset_info.json" ]] || die "RULER 32k data (save_to_disk dir) not found: $RULER_DATA"
command -v nvidia-smi >/dev/null || die "no nvidia-smi -- login node? get a GPU node."
for d in "${LB_DATASETS[@]}"; do
    [[ -s "$LONGBENCH_LOCAL_DIR/$d.jsonl" ]] || die "missing $LONGBENCH_LOCAL_DIR/$d.jsonl (the loader would silently fall back to HF)"
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
    sys.exit("error: flash_attn not importable; both configs need flash_attention_2.")
PY

ARM=0
grep -q '_EVICT_CONTROL_ARM_ENV = "STICKYKV_EVICT_CONTROL_ARM"' modules/windowed_cache/cache.py && ARM=1
EVICT_DESC=$([[ $ARM == 1 ]] && echo "control-arm-eager" || echo "compiled (no control arm at $COMMIT)")

mkdir -p "$R_DIR" "$L_DIR"
QUEUE_POS="$OUT_ROOT/.queue_pos"; LOCK="$OUT_ROOT/.queue_lock"; FAILED="$OUT_ROOT/failed_jobs.txt"
echo 0 > "$QUEUE_POS"; : > "$FAILED"

JOBS=()
for t in "${RULER_TASKS[@]}"; do JOBS+=("ruler $t"); done
for d in "${LB_DATASETS[@]}"; do JOBS+=("lb $d"); done

{
    echo "commit=$START_SHA  started=$(date -Is)  gpus=${GPUS[*]}"
    echo "evict=$EVICT_DESC"
    echo "model=$MODEL_PATH"
    echo "ruler_data=$RULER_DATA  longbench_local_dir=$LONGBENCH_LOCAL_DIR"
    echo "budget=$BUDGET q=$QUANT_RATIO mode=$QUANT_MODE gate=$GATE_RATIO window=$WINDOW local=$LOCAL sink=$SINK"
} > "$OUT_ROOT/run.env"

expected_n() { if [[ $1 == ruler ]]; then echo $RULER_N; else echo "${LB_N[$2]:-200}"; fi; }
job_dir()    { if [[ $1 == ruler ]]; then echo "$R_DIR"; else echo "$L_DIR"; fi; }
is_done() {    # $1 kind, $2 name
    local dir; dir="$(job_dir "$1")"
    [[ -f "$dir/$2.meta.json" && -f "$dir/$2.jsonl" ]] || return 1
    [[ "$(wc -l < "$dir/$2.jsonl")" -eq "$(expected_n "$1" "$2")" ]]
}

next_job() {
    local i
    {
        flock 9
        i=$(<"$QUEUE_POS")
        if (( i < ${#JOBS[@]} )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
    } 9>"$LOCK"
}

run_job() {    # $1 gpu, $2 kind (ruler|lb), $3 name
    local gpu="$1" kind="$2" name="$3" dir
    dir="$(job_dir "$kind")"
    if is_done "$kind" "$name"; then
        echo "[$(date +%T)] gpu=$gpu SKIP   $kind $name (complete)"; return 0
    fi
    if [[ "$(git rev-parse HEAD)" != "$START_SHA" ]]; then
        echo "[$(date +%T)] gpu=$gpu REFUSE $kind $name -- HEAD moved off $COMMIT" >&2
        echo "$kind $name rc=HEAD-moved" >> "$FAILED"; return 1
    fi
    # An incomplete previous attempt: clear it so nothing partial survives.
    rm -f "$dir/$name.jsonl" "$dir/$name.meta.json" "$dir/$name.memory.jsonl" "$dir/$name.memory_summary.json"
    echo "[$(date +%T)] gpu=$gpu START  $kind $name"
    local t0=$SECONDS rc
    local common=(
        model.name="$MODEL_PATH"
        cache.cache_budget="$BUDGET"
        cache.quant_ratio="$QUANT_RATIO"
        cache.quant_budget_mode="$QUANT_MODE"
        cache.quant_gate_ratio="$GATE_RATIO"
        cache.window_size="$WINDOW"   window.window_size="$WINDOW"
        cache.local_window_size="$LOCAL" window.local_window_size="$LOCAL"
        cache.num_sink_tokens="$SINK" window.num_sink_tokens="$SINK"
        cache.first_eviction_step=0
    )
    (
        if [[ $ARM == 1 ]]; then export STICKYKV_EVICT_CONTROL_ARM=1; else unset STICKYKV_EVICT_CONTROL_ARM; fi
        if [[ $kind == ruler ]]; then
            CUDA_VISIBLE_DEVICES="$gpu" exec python main.py --config "$RULER_CONFIG" --override \
                "${common[@]}" \
                ruler.data_dir="$RULER_DATA" ruler.tasks="[$name]" ruler.num_samples=max \
                ruler.resume=false ruler.capture_memory=false ruler.output_dir="$R_DIR"
        else
            CUDA_VISIBLE_DEVICES="$gpu" exec python main.py --config "$LB_CONFIG" --override \
                "${common[@]}" \
                longbench.datasets="[$name]" longbench.output_dir="$L_DIR"
        fi
    ) > "$dir/$name.log" 2>&1
    rc=$?
    if (( rc == 0 )) && is_done "$kind" "$name"; then
        echo "[$(date +%T)] gpu=$gpu DONE   $kind $name ($(( (SECONDS - t0) / 60 )) min)"
    else
        local why
        why=$(grep -m1 -oE "ran EAGER|OutOfMemoryError|CUDA error[^.]*|[A-Za-z]+Error: [^.]{0,80}" "$dir/$name.log" | head -1)
        echo "[$(date +%T)] gpu=$gpu FAIL   $kind $name rc=$rc ${why:+($why)} -- $dir/$name.log" >&2
        echo "$kind $name rc=$rc ${why:-see log} $dir/$name.log" >> "$FAILED"
    fi
}

worker() {
    local gpu="$1" i
    while i=$(next_job) && [[ -n "$i" ]]; do
        # shellcheck disable=SC2086
        run_job "$gpu" ${JOBS[$i]}
    done
}

echo "=== RULER 32k (${#RULER_TASKS[@]} tasks) + LongBench (${#LB_DATASETS[@]} datasets): ${#JOBS[@]} jobs on ${#GPUS[@]} GPU(s) ==="
echo "  commit $COMMIT | eviction: $EVICT_DESC"
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
python -m modules.evaluation.ruler_scoring --predictions_dir "$R_DIR" --out_csv "$R_DIR/scores.csv" \
    > "$R_DIR/scoring.log" 2>&1 || echo "RULER scoring FAILED -- $R_DIR/scoring.log" >&2
python -m modules.evaluation.longbench_scoring --predictions_dir "$L_DIR" --out_csv "$L_DIR/scores.csv" \
    > "$L_DIR/scoring.log" 2>&1 || echo "LongBench scoring FAILED -- $L_DIR/scoring.log" >&2

R_DIR="$R_DIR" L_DIR="$L_DIR" COMMIT="$COMMIT" EVICT_DESC="$EVICT_DESC" RULER_N="$RULER_N" \
RULER_TASKS="${RULER_TASKS[*]}" LB_DATASETS="${LB_DATASETS[*]}" \
python - <<'PY' | tee "$OUT_ROOT/summary.txt"
import csv, json, os
E = os.environ
lb_n = {"multifieldqa_en": 150, "lcc": 500, "repobench-p": 500}

def table(dir_, names, key, expect, title):
    raw = {}
    if os.path.exists(f"{dir_}/scores.csv"):
        raw = {r[key]: float(r["score"]) for r in csv.DictReader(open(f"{dir_}/scores.csv"))}
    rows, issues = {}, []
    for n in names:
        m, j = f"{dir_}/{n}.meta.json", f"{dir_}/{n}.jsonl"
        lines = sum(1 for _ in open(j)) if os.path.exists(j) else 0
        if os.path.exists(m) and lines == expect(n) and n in raw:
            gate = (json.load(open(m)).get("read_gate") or {}).get("verdict", "MISSING")
            rows[n] = (raw[n], gate)
            if gate != "gated":
                issues.append(f"{n}: read_gate={gate}")
        else:
            rows[n] = (None, f"part:{lines}/{expect(n)}" if lines else "not run")
            issues.append(f"{n}: {rows[n][1]}")
    print(title)
    for n in names:
        s, g = rows[n]
        print(f"  {n:22}{s:8.2f}   {g}" if s is not None else f"  {n:22}{'-':>8}   {g}")
    return rows, issues

print(f"commit {E['COMMIT']} | eviction: {E['EVICT_DESC']}")
print("budget 0.20  q 0.70 (bytes)  gate 0.25  window 8  local 128  sink 5\n")
rt = E["RULER_TASKS"].split(); ld = E["LB_DATASETS"].split()
r_rows, r_iss = table(E["R_DIR"], rt, "task", lambda n: int(E["RULER_N"]), "RULER 32k")
def avg(rows, names):
    v = [rows[n][0] for n in names]
    return f"{sum(v) / len(v):8.2f}" if None not in v else "incomplete"
print(f"  {'average (13)':22}{avg(r_rows, rt)}\n")
l_rows, l_iss = table(E["L_DIR"], ld, "dataset", lambda n: lb_n.get(n, 200), "LongBench")
twelve = [d for d in ld if d not in ("passage_count", "passage_retrieval_en", "lcc", "repobench-p")]
print(f"  {'average (16)':22}{avg(l_rows, ld)}")
print(f"  {'average (12, as sweep)':22}{avg(l_rows, twelve)}")
print()
iss = [f"ruler/{i}" for i in r_iss] + [f"longbench/{i}" for i in l_iss]
if iss:
    print("!! not usable as-is (re-run the script to redo just these):")
    for i in iss: print("   ", i)
else:
    print("all 29 jobs complete, full example counts, gate verdict `gated` everywhere.")
PY

if [[ -s "$FAILED" ]]; then echo; echo "!! failed jobs:"; cat "$FAILED"; exit 1; fi
echo; echo "done: $OUT_ROOT/summary.txt"
