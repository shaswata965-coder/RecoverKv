#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# LongBench, 8 datasets x 2 code versions, one shared 8-GPU queue.
#
#   datasets   qasper multifieldqa_en hotpotqa 2wikimqa musique trec triviaqa samsum
#   operating  cache_budget 0.20, q 0.70 (bytes), gate 0.25, window 8, local 128, sink 5
#   repos      main = Cluster_QEvict/RecoverKv         (as checked out -- NOT pulled)
#              v11  = final_run_clustered/RecoverKv_v11 (as checked out -- NOT pulled)
#
# ------------------------------------------------------------------ scheduling
# 16 jobs, one dataset per process, pulled longest-first from ONE queue by 8
# workers (one per GPU). Order from the b=0.20 wall times of sweep_395af34:
#   samsum 2083s  trec 1252s  triviaqa 1037s  musique 909s
#   hotpotqa 790s  qasper 721s  2wikimqa 584s  multifieldqa_en 525s
# ~15,800 GPU-s + model loads over 8 GPUs = ~36-40 min; one samsum (~36 min)
# is the floor. Both repos' copies of a dataset sit next to each other in the
# queue, so neither repo waits on the other.
#
# ------------------------------------------------------------------- eviction
# LongBench prompts vary in length every example, and both repos force two
# per-eviction shapes (T_fp, W) to Python ints inside the compiled eviction.
# On 2026-09-19 that recompiled past Dynamo's 64-graph limit and 16/16 jobs died
# 13-48 examples in with "the two-tier eviction ran EAGER".
#   v11  has STICKYKV_EVICT_CONTROL_ARM (d2c45e3): the same body run eagerly, on
#        purpose, recorded as control-arm-eager. It never builds the compiled
#        callable, so there is no limit to hit. Used by default (EVICT_ARM=auto).
#        Eager vs compiled is numerically identical (5186353), so this changes
#        no score; it is also what every finished sweep_395af34 cell ran.
#   main has no such switch at its commit. It runs compiled -- its only mode --
#        and is EXPECTED TO RISK THE SAME CRASH. A crashed job leaves a partial
#        jsonl and no sidecar; the summary shows it as part:N and keeps it out of
#        every average. It fails in minutes, so it does not hold up v11.
# EVICT_ARM=0 forces compiled everywhere; EVICT_ARM=1 requests the arm everywhere
# (a repo without it refuses to start rather than silently run compiled).
#
# ------------------------------------------------------------------- outputs
#   $OUT_ROOT/<label>_<commit>/<dataset>.{jsonl,meta.json,log}, scores.csv
#   $OUT_ROOT/summary.txt   -- side by side, with gate verdict and eviction path
# Resumable: a job with a non-empty jsonl AND a sidecar is skipped.
# A job refuses to start if its repo's HEAD moved since the sweep began.
#
# Run on an 8-GPU node:
#   conda activate sticky_env
#   nohup scripts/run_longbench_two_repos.sh > ~/lb_two_repos.log 2>&1 &
# ============================================================================

MAIN_REPO="${MAIN_REPO:-/home/ee/phd/eez228470/kv_cache/Cluster_QEvict/RecoverKv}"
V11_REPO="${V11_REPO:-/home/ee/phd/eez228470/kv_cache/final_run_clustered/RecoverKv_v11}"
MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
export LONGBENCH_LOCAL_DIR="${LONGBENCH_LOCAL_DIR:-/home/ee/phd/eez228470/kv_cache/qwen_longbench_final_int2/RecoverKv/data/longbench}"
EVICT_ARM="${EVICT_ARM:-auto}"

# The operating point. Pinned here, not env-overridable, so a variable left
# exported from an earlier ablation cannot move this run.
BUDGET=0.20; QUANT_RATIO=0.70; QUANT_MODE=bytes; GATE_RATIO=0.25
WINDOW=8; LOCAL=128; SINK=5
CONFIG_REL="configs/longbench_ours_flash_attn.yaml"

# Longest first (b=0.20 times above).
DATASETS=(samsum trec triviaqa musique hotpotqa qasper 2wikimqa multifieldqa_en)
declare -A EXPECT_N=([multifieldqa_en]=150)

LABELS=(v11 main)
declare -A REPO=([main]="$MAIN_REPO" [v11]="$V11_REPO")
declare -A SHA SHORT ARM

# ---- preflight ----------------------------------------------------------------
die() { echo "error: $*" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
command -v nvidia-smi >/dev/null || die "no nvidia-smi -- login node? get a GPU node."
for d in "${DATASETS[@]}"; do
    [[ -s "$LONGBENCH_LOCAL_DIR/$d.jsonl" ]] || die "missing $LONGBENCH_LOCAL_DIR/$d.jsonl (the loader would silently fall back to HF)"
done
for l in "${LABELS[@]}"; do
    r="${REPO[$l]}"
    [[ -f "$r/main.py" && -f "$r/$CONFIG_REL" ]] || die "$l: $r is not a RecoverKv checkout with $CONFIG_REL"
    SHA[$l]="$(git -C "$r" rev-parse HEAD)" || die "$l: not a git repo"
    SHORT[$l]="${SHA[$l]:0:7}"
    has_arm=0
    grep -q '_EVICT_CONTROL_ARM_ENV = "STICKYKV_EVICT_CONTROL_ARM"' "$r/modules/windowed_cache/cache.py" && has_arm=1
    case "$EVICT_ARM" in
        auto) ARM[$l]=$has_arm ;;
        0)    ARM[$l]=0 ;;
        1)    (( has_arm )) || die "$l ($r @ ${SHORT[$l]}) has no STICKYKV_EVICT_CONTROL_ARM; EVICT_ARM=1 cannot be honoured"
              ARM[$l]=1 ;;
        *)    die "EVICT_ARM must be auto, 0 or 1" ;;
    esac
done
[[ "${SHA[main]}" != "${SHA[v11]}" ]] || echo "note: both repos are at the same commit ${SHORT[main]}" >&2

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"   # PBS hands out UUIDs; use them as given
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

OUT_ROOT="${OUT_ROOT:-$MAIN_REPO/outputs/longbench/two_repos_main-${SHORT[main]}_v11-${SHORT[v11]}}"
mkdir -p "$OUT_ROOT"
run_dir() { printf '%s/%s_%s' "$OUT_ROOT" "$1" "${SHORT[$1]}"; }
for l in "${LABELS[@]}"; do mkdir -p "$(run_dir "$l")"; done

# ---- the queue -------------------------------------------------------------------
JOBS=()
for d in "${DATASETS[@]}"; do for l in "${LABELS[@]}"; do JOBS+=("$l $d"); done; done
QUEUE_POS="$OUT_ROOT/.queue_pos"; LOCK="$OUT_ROOT/.queue_lock"; FAILED="$OUT_ROOT/failed_jobs.txt"
echo 0 > "$QUEUE_POS"; : > "$FAILED"

{
    echo "started=$(date -Is)  gpus=${GPUS[*]}"
    for l in "${LABELS[@]}"; do
        echo "$l: repo=${REPO[$l]} commit=${SHA[$l]} evict=$([[ ${ARM[$l]} == 1 ]] && echo control-arm-eager || echo compiled)"
    done
    echo "model=$MODEL_PATH  longbench_local_dir=$LONGBENCH_LOCAL_DIR"
    echo "budget=$BUDGET q=$QUANT_RATIO mode=$QUANT_MODE gate=$GATE_RATIO window=$WINDOW local=$LOCAL sink=$SINK"
    echo "datasets=${DATASETS[*]}"
} > "$OUT_ROOT/run.env"

next_job() {
    local i
    {
        flock 9
        i=$(<"$QUEUE_POS")
        if (( i < ${#JOBS[@]} )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
    } 9>"$LOCK"
}

run_job() {    # $1 gpu, $2 label, $3 dataset
    local gpu="$1" l="$2" d="$3" r dir
    r="${REPO[$l]}"; dir="$(run_dir "$l")"
    if [[ -s "$dir/$d.jsonl" && -f "$dir/$d.meta.json" ]]; then
        echo "[$(date +%T)] gpu=$gpu SKIP   $l $d (already done)"; return 0
    fi
    # Each job imports whatever is checked out when it starts; a pull mid-run
    # would split the table across code versions (it did, 2026-09-19).
    if [[ "$(git -C "$r" rev-parse HEAD)" != "${SHA[$l]}" ]]; then
        echo "[$(date +%T)] gpu=$gpu REFUSE $l $d -- $r moved off ${SHORT[$l]}" >&2
        echo "$l $d rc=HEAD-moved" >> "$FAILED"; return 1
    fi
    echo "[$(date +%T)] gpu=$gpu START  $l $d"
    { flock 8; echo "$l $d ${SHA[$l]} $(date -Is)" >> "$OUT_ROOT/job_commits.txt"; } 8>"$LOCK.commits"
    local t0=$SECONDS rc
    (
        cd "$r" || exit 97
        if [[ "${ARM[$l]}" == 1 ]]; then export STICKYKV_EVICT_CONTROL_ARM=1; else unset STICKYKV_EVICT_CONTROL_ARM; fi
        CUDA_VISIBLE_DEVICES="$gpu" exec python "$r/main.py" \
            --config "$r/$CONFIG_REL" \
            --override \
                model.name="$MODEL_PATH" \
                cache.cache_budget="$BUDGET" \
                cache.quant_ratio="$QUANT_RATIO" \
                cache.quant_budget_mode="$QUANT_MODE" \
                cache.quant_gate_ratio="$GATE_RATIO" \
                cache.window_size="$WINDOW" \
                cache.local_window_size="$LOCAL" \
                cache.num_sink_tokens="$SINK" \
                longbench.datasets="[$d]" \
                longbench.output_dir="$dir"
    ) > "$dir/$d.log" 2>&1
    rc=$?
    if (( rc == 0 )); then
        echo "[$(date +%T)] gpu=$gpu DONE   $l $d ($(( (SECONDS - t0) / 60 )) min)"
    else
        echo "[$(date +%T)] gpu=$gpu FAIL   $l $d rc=$rc -- $dir/$d.log" >&2
        echo "$l $d rc=$rc $dir/$d.log" >> "$FAILED"
    fi
}

worker() {
    local gpu="$1" i
    while i=$(next_job) && [[ -n "$i" ]]; do
        # shellcheck disable=SC2086
        run_job "$gpu" ${JOBS[$i]}
    done
}

echo "=== LongBench 8 datasets x 2 repos: ${#JOBS[@]} jobs on ${#GPUS[@]} GPU(s) ==="
for l in "${LABELS[@]}"; do
    echo "  $l  ${SHORT[$l]}  $([[ ${ARM[$l]} == 1 ]] && echo 'eviction: control-arm-eager' || echo 'eviction: compiled')  ${REPO[$l]}"
done
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
for l in "${LABELS[@]}"; do
    dir="$(run_dir "$l")"
    ( cd "${REPO[$l]}" && python -m modules.evaluation.longbench_scoring \
        --predictions_dir "$dir" --out_csv "$dir/scores.csv" ) > "$dir/scoring.log" 2>&1 \
        || echo "scoring FAILED for $l -- see $dir/scoring.log" >&2
done

OUT_ROOT="$OUT_ROOT" DATASETS="${DATASETS[*]}" \
DIR_MAIN="$(run_dir main)" DIR_V11="$(run_dir v11)" SHA_MAIN="${SHORT[main]}" SHA_V11="${SHORT[v11]}" \
python - <<'PY' | tee "$OUT_ROOT/summary.txt"
import csv, json, os
ds = os.environ["DATASETS"].split()
runs = [("v11", os.environ["DIR_V11"], os.environ["SHA_V11"]),
        ("main", os.environ["DIR_MAIN"], os.environ["SHA_MAIN"])]
expect = {"multifieldqa_en": 150}
res = {}
for lab, d, _ in runs:
    raw = {}
    if os.path.exists(f"{d}/scores.csv"):
        raw = {r["dataset"]: float(r["score"]) for r in csv.DictReader(open(f"{d}/scores.csv"))}
    for x in ds:
        m, j = f"{d}/{x}.meta.json", f"{d}/{x}.jsonl"
        if os.path.exists(m):
            meta = json.load(open(m))
            n = meta.get("num_examples")
            env = meta.get("stickykv_env") or {}
            arm = "eager-arm" if str(env.get("STICKYKV_EVICT_CONTROL_ARM", "")).strip() in ("1", "true") else "compiled"
            gate = (meta.get("read_gate") or {}).get("verdict", "MISSING")
            ok = n == expect.get(x, 200) and x in raw
            res[(lab, x)] = (raw.get(x) if ok else None, gate, arm, n)
        elif os.path.exists(j):
            res[(lab, x)] = (None, "absent", "-", f"part:{sum(1 for _ in open(j))}")
print("LongBench  budget 0.20  q 0.70 (bytes)  gate 0.25  window 8  local 128  sink 5")
for lab, _, sha in runs:
    print(f"  {lab:4} @ {sha}")
print()
print(f"{'dataset':17}{'v11':>9}{'main':>9}{'v11-main':>10}   read_gate (v11 / main)   eviction")
issues = []
for x in ds:
    cells, vals, gates, arms = "", [], [], []
    for lab, _, _ in runs:
        s, g, a, n = res.get((lab, x), (None, "absent", "-", "-"))
        cells += f"{s:9.2f}" if s is not None else f"{(str(n) if str(n).startswith('part') else '-'):>9}"
        vals.append(s); gates.append(g); arms.append(a)
        if s is None: issues.append(f"{lab}/{x}: {n if str(n).startswith('part') else 'not finished'}")
        elif g != "gated": issues.append(f"{lab}/{x}: read_gate={g}")
    delta = f"{vals[0] - vals[1]:+10.2f}" if None not in vals else f"{'':>10}"
    print(f"{x:17}{cells}{delta}   {gates[0]:>9} / {gates[1]:<9}   {arms[0]} / {arms[1]}")
avg = lambda lab: (sum(res[(lab, x)][0] for x in ds) / len(ds)
                   if all(res.get((lab, x), (None,))[0] is not None for x in ds) else None)
a = [avg(lab) for lab, _, _ in runs]
fmt = lambda v: f"{v:9.2f}" if v is not None else f"{'incompl.':>9}"
print(f"{'average (8)':17}{fmt(a[0])}{fmt(a[1])}" + (f"{a[0] - a[1]:+10.2f}" if None not in a else ""))
print()
if issues:
    print("!! not usable as-is:")
    for i in issues: print("   ", i)
    print("   'part:N' = crashed after N examples (no sidecar). Re-run the script to redo just those.")
else:
    print("all 16 cells complete, full example counts, gate verdict `gated`.")
PY

if [[ -s "$FAILED" ]]; then
    echo; echo "!! failed jobs:"; cat "$FAILED"; exit 1
fi
echo; echo "done: $OUT_ROOT/summary.txt"
