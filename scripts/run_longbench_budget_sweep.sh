#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# LongBench budget sweep -- cache_budget 0.20 / 0.10 / 0.05 at the pinned
# operating point otherwise (q=0.70 bytes, gate 0.25, window 8, local 128,
# sink 5), the 12 non-code English datasets minus passage_count /
# passage_retrieval_en, on every GPU this node gives you.
#
# ------------------------------------------------------------------ scheduling
# 36 jobs = 12 datasets x 3 budgets, one dataset per process, pulled from ONE
# shared queue ordered longest-first: each GPU takes the next job the moment it
# finishes its last. That is LPT scheduling done online, so it absorbs timing
# drift (the code has changed since these were measured) instead of baking a
# static packing in. Order comes from the per-dataset wall times of the
# 2026-09-15 run (outputs/longbench/ours_flash_attn_8gpu_no_code/*.meta.json):
#
#   gov_report 6852s  multi_news 6203s  samsum 2228s  qmsum 2182s
#   narrativeqa 1814s  trec 1376s  triviaqa 1118s  musique 844s
#   hotpotqa 674s  qasper 671s  2wikimqa 549s  multifieldqa_en 529s
#
# Simulated on 8 GPUs (+90 s model load per job): ~2.8 h, against a 2.7 h
# perfect split and ~5.8 h for the three budgets run back to back. The floor is
# one gov_report (~1.9 h) -- a dataset cannot be split across GPUs.
#
# ----------------------------------------------------------------- read first
# * THE 5% ROW OVER-SPENDS ON SHORT PROMPTS. When cache_budget * prompt_len <
#   sink + local (133 tokens) the cache clamps to sink + local, keeps 0
#   evictable windows, and warns (modules/windowed_cache/config.py:532). With
#   the Llama-3.1 tokenizer over context+input, that is ~228/2350 examples at
#   5% -- 132 of them multi_news (66% of it) -- 50 at 10% (44 multi_news) and
#   10 at 20% (all multi_news). multi_news at 5% therefore reads better than a
#   true 5% cache would. Token counts exclude the prompt template, so the real
#   numbers are slightly lower.
# * SCORES FROM BEFORE ac128e8 / 8a8cdc0 DO NOT COMPARE (CLAUDE.md): both moved
#   the stored format. That includes the 2026-09-15 run's 46.33. All three
#   budgets run here, at one commit, for that reason.
# * Every sidecar records read_gate.verdict; the summary prints it. Anything
#   but `gated` means the number beside it is not a gated number.
#
# ------------------------------------------------------------------ configure
#   MODEL_PATH           checkpoint                   (default: llama-3.1-8b-instruct)
#   LONGBENCH_LOCAL_DIR  the <dataset>.jsonl files    (default: the qwen_longbench copy)
#   BUDGETS              space list                   (default: "0.20 0.10 0.05")
#   (quant_ratio is pinned at 0.70 in the script body, not configurable)
#   OUT_ROOT             one sub-dir per budget        (default: outputs/longbench/sweep_<commit>)
#
# Resumable: a job whose <dataset>.jsonl is non-empty AND has a .meta.json is
# skipped, so re-running after a crash or walltime kill picks up where it left.
#
# Run on the GPU node:
#   conda activate sticky_env && scripts/run_longbench_budget_sweep.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
export LONGBENCH_LOCAL_DIR="${LONGBENCH_LOCAL_DIR:-/home/ee/phd/eez228470/kv_cache/qwen_longbench_final_int2/RecoverKv/data/longbench}"
CONFIG="configs/longbench_ours_flash_attn.yaml"
BUDGETS="${BUDGETS:-0.20 0.10 0.05}"
# Pinned, not env-overridable: a QUANT_RATIO left exported in the shell from an
# earlier ablation must not silently move this sweep off q=0.70.
QUANT_RATIO=0.70
GATE_RATIO=0.25
QUANT_MODE=bytes
WINDOW=8
LOCAL=128
SINK=5
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
START_SHA="$(git rev-parse HEAD 2>/dev/null || echo nogit)"
OUT_ROOT="${OUT_ROOT:-outputs/longbench/sweep_${COMMIT}}"

# Longest first. Keep in step with the header's measured times.
DATASETS=(gov_report multi_news samsum qmsum narrativeqa trec triviaqa
          musique hotpotqa qasper 2wikimqa multifieldqa_en)

budget_dir() { printf '%s/q%s_b%s_w%s_l%s' "$OUT_ROOT" "${QUANT_RATIO/./}" "${1/./}" "$WINDOW" "$LOCAL"; }

# ---- preflight: fail here, not 36 jobs deep ---------------------------------
[[ -f "$CONFIG" ]]     || { echo "error: config not found: $CONFIG" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]] || { echo "error: model not found: $MODEL_PATH" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "error: no nvidia-smi -- login node? get a GPU node." >&2; exit 2; }
# A missing jsonl does not error: the loader silently falls back to the
# HuggingFace script dataset, which datasets>=3 refuses to run.
for d in "${DATASETS[@]}"; do
    [[ -s "$LONGBENCH_LOCAL_DIR/$d.jsonl" ]] || { echo "error: missing $LONGBENCH_LOCAL_DIR/$d.jsonl" >&2; exit 2; }
done

# GPUs: the entries of CUDA_VISIBLE_DEVICES as given (PBS hands out UUIDs --
# indexing 0..N-1 could land on another job's card), else every device listed.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
(( ${#GPUS[@]} > 0 )) || { echo "error: no GPUs visible" >&2; exit 2; }

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

# ---- the queue ---------------------------------------------------------------
# Jobs ordered longest-first across budgets: gov_report@0.20, @0.10, @0.05,
# multi_news@..., so the three longest-running datasets start first everywhere.
JOBS=()
for d in "${DATASETS[@]}"; do
    for b in $BUDGETS; do JOBS+=("$b $d"); done
done

mkdir -p "$OUT_ROOT"
for b in $BUDGETS; do mkdir -p "$(budget_dir "$b")"; done
QUEUE_POS="$OUT_ROOT/.queue_pos"; LOCK="$OUT_ROOT/.queue_lock"; FAILED="$OUT_ROOT/failed_jobs.txt"
echo 0 > "$QUEUE_POS"; : > "$FAILED"

{
    echo "commit=$COMMIT  started=$(date -Is)"
    echo "config=$CONFIG  model=$MODEL_PATH"
    echo "longbench_local_dir=$LONGBENCH_LOCAL_DIR"
    echo "budgets=$BUDGETS  quant_ratio=$QUANT_RATIO  quant_budget_mode=$QUANT_MODE  quant_gate_ratio=$GATE_RATIO"
    echo "window_size=$WINDOW  local_window_size=$LOCAL  num_sink_tokens=$SINK"
    echo "gpus=${GPUS[*]}"
} > "$OUT_ROOT/run.env"

next_job() {   # prints the next job index, or nothing when the queue is empty
    local i
    {
        flock 9
        i=$(<"$QUEUE_POS")
        if (( i < ${#JOBS[@]} )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
    } 9>"$LOCK"
}

run_job() {    # $1 = gpu, $2 = budget, $3 = dataset
    local gpu="$1" b="$2" d="$3" dir
    dir="$(budget_dir "$b")"
    if [[ -s "$dir/$d.jsonl" && -f "$dir/$d.meta.json" ]]; then
        echo "[$(date +%T)] gpu=$gpu SKIP  b=$b $d (already done)"
        return 0
    fi
    # Each job is a fresh process that imports whatever is checked out NOW. A
    # `git pull` mid-sweep (2026-09-19, 02:32) split one sweep across two code
    # versions: every job started after it ran 4d588e2's eviction and failed.
    # Refuse rather than mix: the table is only one table at one commit.
    if [[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null)" != "$START_SHA" ]]; then
        echo "[$(date +%T)] gpu=$gpu REFUSE b=$b $d -- HEAD moved off $COMMIT since the sweep started" >&2
        echo "$b $d rc=HEAD-moved (checkout changed mid-sweep; not run)" >> "$FAILED"
        return 1
    fi
    echo "[$(date +%T)] gpu=$gpu START b=$b $d"
    # The commit a job RUNS is the one checked out when it starts (it imports
    # then). The sidecar's commit_sha is read when it finishes, so a job that
    # spans a pull is mislabelled there -- this file is the one the summary trusts.
    { flock 8; echo "$b $d $START_SHA $(date -Is)" >> "$OUT_ROOT/job_commits.txt"; } 8>"$LOCK.commits"
    local t0=$SECONDS
    CUDA_VISIBLE_DEVICES="$gpu" python "$PROJECT_ROOT/main.py" \
        --config "$CONFIG" \
        --override \
            model.name="$MODEL_PATH" \
            cache.cache_budget="$b" \
            cache.quant_ratio="$QUANT_RATIO" \
            cache.quant_budget_mode="$QUANT_MODE" \
            cache.quant_gate_ratio="$GATE_RATIO" \
            cache.window_size="$WINDOW" \
            cache.local_window_size="$LOCAL" \
            cache.num_sink_tokens="$SINK" \
            longbench.datasets="[$d]" \
            longbench.output_dir="$dir" \
        > "$dir/$d.log" 2>&1
    local rc=$?
    if (( rc == 0 )); then
        echo "[$(date +%T)] gpu=$gpu DONE  b=$b $d  ($(( (SECONDS - t0) / 60 )) min)"
    else
        echo "[$(date +%T)] gpu=$gpu FAIL  b=$b $d  rc=$rc -- see $dir/$d.log" >&2
        echo "$b $d rc=$rc $dir/$d.log" >> "$FAILED"
    fi
}

worker() {     # one per GPU: pull jobs until the queue is empty
    local gpu="$1" i
    while i=$(next_job) && [[ -n "$i" ]]; do
        # shellcheck disable=SC2086
        run_job "$gpu" ${JOBS[$i]}
    done
}

echo "=== LongBench budget sweep @ $COMMIT: ${#JOBS[@]} jobs on ${#GPUS[@]} GPU(s) ==="
echo "budgets: $BUDGETS | q=$QUANT_RATIO ($QUANT_MODE) gate=$GATE_RATIO | window=$WINDOW local=$LOCAL sink=$SINK"
echo "out: $OUT_ROOT"
echo

pids=()
for g in "${GPUS[@]}"; do
    worker "$g" 2>&1 | tee -a "$OUT_ROOT/worker_${g}.log" &
    pids+=($!)
done
wait "${pids[@]}"

# ---- score + summary ---------------------------------------------------------
echo
echo "=== scoring ==="
for b in $BUDGETS; do
    dir="$(budget_dir "$b")"
    python -m modules.evaluation.longbench_scoring \
        --predictions_dir "$dir" --out_csv "$dir/scores.csv" > "$dir/scoring.log" 2>&1 \
        || echo "scoring FAILED for b=$b -- see $dir/scoring.log" >&2
done

OUT_ROOT="$OUT_ROOT" BUDGETS="$BUDGETS" QTAG="${QUANT_RATIO/./}" WINDOW="$WINDOW" LOCAL="$LOCAL" \
DATASETS="${DATASETS[*]}" python - <<'PY' | tee "$OUT_ROOT/summary.txt"
import csv, json, os
root, budgets, ds = os.environ["OUT_ROOT"], os.environ["BUDGETS"].split(), os.environ["DATASETS"].split()
bdir = lambda b: f"{root}/q{os.environ['QTAG']}_b{b.replace('.', '')}_w{os.environ['WINDOW']}_l{os.environ['LOCAL']}"
score, verdict, commits, partial = {}, {}, {}, {}
# Commit each job ran, as recorded at its START (last entry wins: a re-run
# supersedes a crashed attempt). Not the sidecar's commit_sha -- see run_job.
started = {}
jc = f"{root}/job_commits.txt"
if os.path.exists(jc):
    for line in open(jc):
        f = line.split()
        if len(f) >= 3 and not line.startswith("#"):
            started[(f[0], f[1])] = f[2][:7]
for b in budgets:
    raw = {}
    p = f"{bdir(b)}/scores.csv"
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            raw[r["dataset"]] = float(r["score"])
    for d in ds:
        m = f"{bdir(b)}/{d}.meta.json"
        # The runner writes the jsonl AS IT GOES and the sidecar only at the end,
        # so a crashed job leaves a scoreable partial jsonl and no sidecar. The
        # scorer cannot tell; the sidecar can. No sidecar -> not a result.
        if os.path.exists(m):
            meta = json.load(open(m))
            verdict[(b, d)] = (meta.get("read_gate") or {}).get("verdict", "MISSING")
            commits[(b, d)] = started.get((b, d), "unknown")
            if d in raw:
                score[(b, d)] = raw[d]
        else:
            j = f"{bdir(b)}/{d}.jsonl"
            if os.path.exists(j):
                partial[(b, d)] = sum(1 for _ in open(j))
hdr = "".join(f"{'b=' + b:>10}" for b in budgets)
print(f"{'dataset':18}{hdr}   read_gate")
bad = []
cell = lambda b, d: (f"{score[(b, d)]:10.2f}" if (b, d) in score
                     else f"{'part:' + str(partial[(b, d)]):>10}" if (b, d) in partial else f"{'-':>10}")
for d in ds:
    cells = "".join(cell(b, d) for b in budgets)
    vs = {verdict.get((b, d), "absent") for b in budgets}
    if vs != {"gated"}:
        bad.append(d)
    print(f"{d:18}{cells}   {'gated' if vs == {'gated'} else ' / '.join(verdict.get((b, d), 'absent') for b in budgets)}")
avg = "".join(
    f"{sum(score[(b, d)] for d in ds) / len(ds):10.2f}" if all((b, d) in score for d in ds) else f"{'incomplete':>10}"
    for b in budgets)
print(f"{'macro avg':18}{avg}")
print()
print("multi_news is over budget on ~66% of examples at 5% (and ~22% at 10%): the")
print("cache clamps to sink+local when budget*len < 133 tokens. Read that row with care.")
if bad:
    print(f"!! read_gate not `gated` everywhere for: {', '.join(bad)} -- those are not gated numbers.")
if partial:
    print(f"!! {len(partial)} cell(s) crashed partway ('part:N' = N examples written, no sidecar);"
          " they are excluded from every average. Re-run the script to redo just those.")
shas = sorted(set(commits.values()) - {"unknown"})
if "unknown" in commits.values():
    print("!! some finished cells have no job_commits.txt entry: the code they ran is unrecorded.")
if len(shas) > 1:
    print(f"!! MIXED CODE: finished cells span {len(shas)} commits ({', '.join(shas)}). This is not one table.")
elif shas:
    print(f"all finished cells at commit {shas[0]}")
PY

if [[ -s "$FAILED" ]]; then
    echo
    echo "!! failed jobs (re-run this script to retry just these):"
    cat "$FAILED"
    exit 1
fi
echo
echo "done: $OUT_ROOT/summary.txt"
