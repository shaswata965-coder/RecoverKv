#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# ONE-DIRECTION ablation: is the Q->fp promote path worth its cost?
#
#   arms       twoway  (cache.one_direction=false -- shipped, the control)
#              oneway  (cache.one_direction=true  -- demotion is irreversible)
#   budgets    0.20, 0.10
#   datasets   narrativeqa 2wikimqa qmsum samsum -- one per LongBench family,
#              the longest-generation member of each where there is a winner
#              (single-doc 128, multi-doc 32 all tied, summarization 512 all
#              tied, few-shot 128).
#   fixed      q 0.70 (bytes) | gate 0.25 | window 8 | local 128 | sink 5
#   16 jobs, 200 examples each.
#
# ------------------------------------------------------------- what to expect
# Dropping promotion can only LOWER fidelity: a window whose importance rises
# after it was demoted stays int2 instead of returning to fp16. So oneway
# should tie or lose against twoway; a dataset that gains by more than noise
# means something else moved.
#
# The size of the loss is bounded by how often the path fires at all, which the
# sidecars now record. `promote.promote_mean` is windows promoted per eviction,
# measured in BOTH arms:
#   twoway promote_mean ~ 0  -> the path almost never runs, oneway should tie,
#                              and the finding is that it can be deleted (it is
#                              the widest payload in the eviction: dequantize +
#                              RoPE + splice).
#   twoway promote_mean large -> the path does real work and oneway should show
#                              a visible drop.
# Either way the ablation is informative; that is why it is worth 75 minutes.
#
# ------------------------------------------------- the check that must pass
# `promote.forced` in the oneway sidecars MUST be 0. The policy demotes fp picks
# in RANK rather than excluding them outright, because an under-filled fp tier
# would break the rectangular shapes the compiled eviction needs; if an eviction
# ever has fewer than top_k_fp never-demoted windows it falls back to promoting
# one. A oneway row with forced > 0 is partly two-way and cannot be quoted as
# this ablation. The summary prints it per row.
#
# ----------------------------------------------------- keeping 8 GPUs busy
# ONE queue, 8 workers, global longest-first (samsum 37m, qmsum 36m,
# narrativeqa 30m, 2wikimqa 9m at the 2026-09-15 Llama times). 16 jobs over 8
# GPUs is two waves: ~75 min. Ask for 3 h.
#
# At 10% with local 128 the budget floor (sink+local = 133 tokens) bites on
# ~50/2350 examples across the full 16-dataset set, 44 of them multi_news --
# none of these four. So both budgets here are true budgets.
#
# Resuming: DONE = <dataset>.jsonl non-empty AND <dataset>.meta.json present.
# Re-run to redo only what is missing; nothing is deleted. A job refuses to
# start if HEAD moved.
#
# Run:
#   cd ~/kv_cache/one-direction_clsutered/RecoverKv && conda activate sticky_env
#   nohup scripts/run_longbench4_onedirection_8gpu.sh > ~/lb4_onedir.log 2>&1 &
#   tail -f ~/lb4_onedir.log
#
#   DRY_RUN=1 ...            plan only
#   BUDGETS="0.20" ...       one budget
#   ARMS="oneway" ...        one arm (the control already exists elsewhere)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 2

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
export LONGBENCH_LOCAL_DIR="${LONGBENCH_LOCAL_DIR:-/home/ee/phd/eez228470/kv_cache/qwen_longbench_final_int2/RecoverKv/data/longbench}"
CONFIG="${CONFIG:-configs/longbench_ours_flash_attn.yaml}"

read -r -a ARMS <<< "${ARMS:-twoway oneway}"
read -r -a BUDGETS <<< "${BUDGETS:-0.20 0.10}"
read -r -a DATASETS <<< "${DATASETS:-samsum qmsum narrativeqa 2wikimqa}"   # longest first

# Pinned -- not env-overridable, so no cell can drift.
QUANT_RATIO=0.70; QUANT_MODE=bytes; GATE_RATIO=0.25; WINDOW=8; LOCAL=128; SINK=5

COMMIT="$(git rev-parse --short HEAD)"; START_SHA="$(git rev-parse HEAD)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/lb4_onedirection_${COMMIT}}"
DRY_RUN="${DRY_RUN:-0}"

arm_flag() { [[ "$1" == "oneway" ]] && echo true || echo false; }
cell_dir() { printf '%s/%s_b%s' "$OUT_ROOT" "$1" "$2"; }
die() { echo "error: $*" >&2; exit 2; }

# ---- preflight ---------------------------------------------------------------
[[ -f "$CONFIG" ]]     || die "config not found: $CONFIG"
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
for a in "${ARMS[@]}"; do [[ "$a" == twoway || "$a" == oneway ]] || die "ARMS entries are twoway|oneway, got '$a'"; done
for b in "${BUDGETS[@]}"; do
    awk -v b="$b" 'BEGIN{ if (b+0 <= 0 || b+0 > 1) { print "budget must be in (0,1], got " b > "/dev/stderr"; exit 2 } }' \
        || die "bad budget '$b'"
done
for d in "${DATASETS[@]}"; do
    [[ -s "$LONGBENCH_LOCAL_DIR/$d.jsonl" ]] || die "missing $LONGBENCH_LOCAL_DIR/$d.jsonl"
done
(( LOCAL % WINDOW == 0 )) || die "local $LOCAL is not a multiple of window $WINDOW"
(( WINDOW % 4 == 0 ))     || die "window $WINDOW is not a multiple of 4, and q=$QUANT_RATIO > 0"
# The flag must exist in THIS checkout, or `oneway` silently runs the control.
python - <<'PY' || exit 2
import sys; sys.path.insert(0, ".")
from utils.config import CacheConfig
if "one_direction" not in getattr(CacheConfig, "__dataclass_fields__", {}):
    sys.exit("error: cache.one_direction is not a field in this checkout -- the "
             "oneway arm would run the shipped two-way cache under a label that "
             "says otherwise. Wrong repo?")
print("  cache.one_direction present; default =",
      CacheConfig.__dataclass_fields__["one_direction"].default)
PY

if [[ "$DRY_RUN" != "1" ]]; then
    command -v nvidia-smi >/dev/null || die "no nvidia-smi -- login node? get a GPU node."
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
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
    GPUS=(0)
fi
(( ${#GPUS[@]} > 0 )) || die "no GPUs visible"

# ---- the queue: global longest-first -----------------------------------------
JOBS=()
for d in "${DATASETS[@]}"; do for b in "${BUDGETS[@]}"; do for a in "${ARMS[@]}"; do
    JOBS+=("$a $b $d"); done; done; done

echo "=== one-direction ablation @ $COMMIT: ${#JOBS[@]} jobs on ${#GPUS[@]} GPU(s) ==="
echo "  arms:     ${ARMS[*]}   (oneway => cache.one_direction=true)"
echo "  budgets:  ${BUDGETS[*]}"
echo "  datasets: ${DATASETS[*]}   (longest first)"
echo "  fixed:    q=$QUANT_RATIO ($QUANT_MODE) gate=$GATE_RATIO window=$WINDOW local=$LOCAL sink=$SINK"
todo=0
for a in "${ARMS[@]}"; do for b in "${BUDGETS[@]}"; do
    d="$(cell_dir "$a" "$b")"; n=0
    for x in "${DATASETS[@]}"; do
        [[ -s "$d/$x.jsonl" && -f "$d/$x.meta.json" ]] && n=$(( n + 1 ))
    done
    todo=$(( todo + ${#DATASETS[@]} - n ))
    printf "    %-7s b=%-5s %d/%d done\n" "$a" "$b" "$n" "${#DATASETS[@]}"
done; done
echo "  to run:   $todo job(s)"
NGPU=${#GPUS[@]} TODO="$(for d in "${DATASETS[@]}"; do for b in "${BUDGETS[@]}"; do for a in "${ARMS[@]}"; do
        dir="$(cell_dir "$a" "$b")"
        [[ -s "$dir/$d.jsonl" && -f "$dir/$d.meta.json" ]] || printf '%s ' "$d"; done; done; done)" \
python - <<'PY'
import os
T = {"samsum": 2228, "qmsum": 2182, "narrativeqa": 1814, "2wikimqa": 549}
LOAD, m = 90, int(os.environ["NGPU"])
jobs = [T.get(d, 1200) + LOAD for d in os.environ["TODO"].split()]
if not jobs or m < 1:
    raise SystemExit
end = [0.0]*m
for j in sorted(jobs, reverse=True):
    i = min(range(m), key=lambda k: end[k]); end[i] += j
ms, tot = max(end), sum(jobs)
print(f"  estimate: makespan {ms/60:.0f} min | {tot/3600:.1f} GPU-h | "
      f"utilisation {100*tot/(m*ms):.0f}%")
PY
echo "  out:      $OUT_ROOT"
echo
[[ "$DRY_RUN" == "1" ]] && { echo "dry run: grid is valid, nothing started."; exit 0; }

mkdir -p "$OUT_ROOT"
for a in "${ARMS[@]}"; do for b in "${BUDGETS[@]}"; do mkdir -p "$(cell_dir "$a" "$b")"; done; done
exec 200>"$OUT_ROOT/.runner_lock"
flock -n 200 || die "another copy is already running against $OUT_ROOT"
QUEUE_POS="$OUT_ROOT/.queue_pos"; LOCK="$OUT_ROOT/.queue_lock"; FAILED="$OUT_ROOT/failed_jobs.txt"
echo 0 > "$QUEUE_POS"; : > "$FAILED"
{
    echo "commit=$START_SHA  started=$(date -Is)  host=$(hostname)  gpus=${GPUS[*]}"
    echo "config=$CONFIG  model=$MODEL_PATH"
    echo "arms=${ARMS[*]}  budgets=${BUDGETS[*]}  datasets=${DATASETS[*]}"
    echo "q=$QUANT_RATIO mode=$QUANT_MODE gate=$GATE_RATIO window=$WINDOW local=$LOCAL sink=$SINK"
} > "$OUT_ROOT/run.env"

next_job() {
    local i
    { flock 9; i=$(<"$QUEUE_POS")
      if (( i < ${#JOBS[@]} )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
    } 9>"$LOCK"
}

run_job() {   # $1 gpu, $2 arm, $3 budget, $4 dataset
    local gpu="$1" a="$2" b="$3" x="$4" dir flag
    dir="$(cell_dir "$a" "$b")"; flag="$(arm_flag "$a")"
    if [[ -s "$dir/$x.jsonl" && -f "$dir/$x.meta.json" ]]; then
        echo "[$(date +%T)] gpu=$gpu SKIP   $a b=$b $x (complete)"; return 0
    fi
    if [[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" != "$START_SHA" ]]; then
        echo "[$(date +%T)] gpu=$gpu REFUSE $a b=$b $x -- HEAD moved off $COMMIT" >&2
        echo "$a b=$b $x rc=HEAD-moved" >> "$FAILED"; return 1
    fi
    echo "[$(date +%T)] gpu=$gpu START  $a b=$b $x"
    local t0=$SECONDS rc
    CUDA_VISIBLE_DEVICES="$gpu" python "$PROJECT_ROOT/main.py" --config "$CONFIG" --override \
        model.name="$MODEL_PATH" \
        cache.cache_budget="$b" \
        cache.quant_ratio="$QUANT_RATIO" \
        cache.quant_budget_mode="$QUANT_MODE" \
        cache.quant_gate_ratio="$GATE_RATIO" \
        cache.window_size="$WINDOW" \
        cache.local_window_size="$LOCAL" \
        cache.num_sink_tokens="$SINK" \
        cache.first_eviction_step=0 \
        cache.one_direction="$flag" \
        longbench.datasets="[$x]" \
        longbench.resume=false \
        longbench.output_dir="$dir" \
        > "$dir/$x.log" 2>&1
    rc=$?
    if (( rc == 0 )) && [[ -s "$dir/$x.jsonl" && -f "$dir/$x.meta.json" ]]; then
        echo "[$(date +%T)] gpu=$gpu DONE   $a b=$b $x ($(( (SECONDS - t0) / 60 )) min)"
    else
        local why
        why=$(grep -m1 -oE "ran EAGER|OutOfMemoryError|CUDA error[^.]*|[A-Za-z]+Error: [^.]{0,70}" "$dir/$x.log" | head -1)
        echo "[$(date +%T)] gpu=$gpu FAIL   $a b=$b $x rc=$rc ${why:+($why)} -- $dir/$x.log" >&2
        echo "$a b=$b $x rc=$rc ${why:-see log}" >> "$FAILED"
    fi
}

worker() { local gpu="$1" i; while i=$(next_job) && [[ -n "$i" ]]; do
    # shellcheck disable=SC2086
    run_job "$gpu" ${JOBS[$i]}; done; }

pids=()
for g in "${GPUS[@]}"; do worker "$g" 2>&1 | tee -a "$OUT_ROOT/worker_${g}.log" & pids+=($!); done
wait "${pids[@]}"

# ---- score + summary ----------------------------------------------------------
echo; echo "=== scoring ==="
for a in "${ARMS[@]}"; do for b in "${BUDGETS[@]}"; do
    d="$(cell_dir "$a" "$b")"
    python -m modules.evaluation.longbench_scoring --predictions_dir "$d" --out_csv "$d/scores.csv" \
        > "$d/scoring.log" 2>&1 || echo "scoring FAILED for $a b=$b -- $d/scoring.log" >&2
done; done

CELLS="$(for a in "${ARMS[@]}"; do for b in "${BUDGETS[@]}"; do printf '%s:%s=%s ' "$a" "$b" "$(cell_dir "$a" "$b")"; done; done)" \
COMMIT="$COMMIT" DATASETS="${DATASETS[*]}" ARMS="${ARMS[*]}" BUDGETS="${BUDGETS[*]}" \
python - <<'PY' | tee "$OUT_ROOT/summary.txt"
import csv, json, os
E = os.environ
ds = E["DATASETS"].split(); arms = E["ARMS"].split(); buds = E["BUDGETS"].split()
cells = {}
for item in E["CELLS"].split():
    key, d = item.split("=", 1); cells[key] = d

def load(d):
    raw = {}
    if os.path.exists(f"{d}/scores.csv"):
        raw = {r["dataset"]: float(r["score"]) for r in csv.DictReader(open(f"{d}/scores.csv"))}
    sc, prom = {}, {}
    for x in ds:
        j, m = f"{d}/{x}.jsonl", f"{d}/{x}.meta.json"
        n = sum(1 for _ in open(j)) if os.path.exists(j) else 0
        sc[x] = raw.get(x) if (x in raw and n) else None
        if os.path.exists(m):
            meta = json.load(open(m))
            prom[x] = (meta.get("promote") or {}, (meta.get("read_gate") or {}).get("verdict"),
                       meta.get("one_direction"))
    return sc, prom

res = {k: load(v) for k, v in cells.items()}
print(f"ONE-DIRECTION ablation @ {E['COMMIT']}")
print("q 0.70 (bytes)  gate 0.25  window 8  local 128  sink 5 | 200 examples/dataset")
print("twoway = shipped (promotion allowed) | oneway = demotion irreversible\n")

for b in buds:
    print(f"--- cache_budget {b} " + "-" * 44)
    print(f"{'dataset':16}" + "".join(f"{a:>12}" for a in arms) + f"{'delta':>10}")
    for x in ds:
        row = f"{x:16}"
        vals = []
        for a in arms:
            v = res[f"{a}:{b}"][0][x]; vals.append(v)
            row += f"{v:12.2f}" if v is not None else f"{'-':>12}"
        if len(vals) == 2 and None not in vals:
            row += f"{vals[1]-vals[0]:+10.2f}"
        print(row)
    row = f"{'average':16}"; avs = []
    for a in arms:
        v = [res[f"{a}:{b}"][0][x] for x in ds]
        av = sum(v)/len(v) if None not in v else None; avs.append(av)
        row += f"{av:12.2f}" if av is not None else f"{'incompl.':>12}"
    if len(avs) == 2 and None not in avs:
        row += f"{avs[1]-avs[0]:+10.2f}"
    print(row + "\n")

print("How often does the promote path fire? (windows promoted per eviction)")
print(f"  {'cell':>14}{'promote_mean':>14}{'promote_max':>13}{'forced':>8}   gate")
bad = []
for a in arms:
    for b in buds:
        pr = res[f"{a}:{b}"][1]
        means = [p[0].get("promote_mean") for p in pr.values() if p[0].get("promote_mean") is not None]
        mx = max([p[0].get("promote_max", 0) or 0 for p in pr.values()], default=0)
        forced = sum(p[0].get("forced", 0) or 0 for p in pr.values())
        gv = {p[1] for p in pr.values()}
        mean = sum(means)/len(means) if means else None
        print(f"  {a+' b'+b:>14}{(f'{mean:.3f}' if mean is not None else '-'):>14}"
              f"{mx:>13}{forced:>8}   {sorted(gv) if gv else '-'}")
        if a == "oneway" and forced:
            bad.append(f"{a} b={b}: forced={forced} -- the fp tier starved, this row is partly two-way")
        if a == "oneway" and mean:
            bad.append(f"{a} b={b}: promote_mean={mean:.3f} -- promotion still happened")
        if gv - {"gated"}:
            bad.append(f"{a} b={b}: gate verdict {sorted(gv)}")
if bad:
    print("\n!! these rows are not what they say they are:")
    for x in bad: print("   ", x)
else:
    print("\n  oneway promoted nothing and forced nothing; gate `gated` throughout.")
print("\n  twoway promote_mean is the work one_direction removes. Near 0 means the")
print("  promote path (dequantize + RoPE + splice, the eviction's widest payload)")
print("  is doing almost nothing and can be deleted; large means it earns its cost.")

missing = [f"{k} {x}" for k in cells for x in ds if res[k][0][x] is None]
if missing:
    print(f"\n!! {len(missing)} job(s) with no score:")
    for m in missing[:20]: print("   ", m)
PY

if [[ -s "$FAILED" ]]; then echo; echo "!! failed jobs:"; cat "$FAILED"; exit 1; fi
echo; echo "done: $OUT_ROOT/summary.txt"
