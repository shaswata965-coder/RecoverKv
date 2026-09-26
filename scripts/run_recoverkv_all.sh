#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# RecoverKV -- every observation and experiment, one command, one GPU queue.
# ============================================================================
#   nohup scripts/run_recoverkv_all.sh > ~/recoverkv_all.log 2>&1 &
#   tail -f ~/recoverkv_all.log
#
# What runs (SUITES, default all four):
#
#   obs  Observations I-V on a dataset panel (scripts/run_qevict_observations.sh):
#        one full-KV parity base per dataset, then one teacher-forced ours run
#        per OBS_ARM. I skew, II future missed mass, III revival, IV the decode
#        read ledger (per-head gate recall), V tier moves (promotion / demotion
#        / eviction and whether each landed on future attention).
#   e1   "Reading a quarter": accuracy at read-gate ratios E1_GATES (+ 0.25,
#        shared with e2) on E1_LB + E1_RULER at budget 0.20. The obs arms
#        gate0.10..gate1.0 and card4 give the per-head recall behind it.
#   e2   "Recovery at matched bytes": Bidir (shipped) vs OneWay
#        (quant_promotion=oneway) on all 16 LongBench datasets at LB_BUDGETS
#        and 13 RULER tasks at RULER_BUDGET. Same examples, decoding, scores,
#        quantizer, window, q and bytes; only promotion differs.
#   e3   "What a recovered window is made of": A = dequantized promotion
#        (shipped, = e2's Bidir), B = original fp oracle
#        (quant_promote_source=original, shadow OUTSIDE the budget), C = no
#        promotion (= e2's OneWay). Only B is new work; A and C are shared.
#
# then scores every prediction directory and writes the aggregate report:
#   $OUT_ROOT/report/report.html  (self-contained, figures embedded)
#   $OUT_ROOT/report/report.md, tables/*.csv, figures/*.png
#
# Everything else is the operating point (utils.config.OPERATING_POINT):
# q 0.70 (bytes), window 8, sink 5, local 128 (32 at budgets below 0.10, as in
# run_longbench16_b010_b005.sh), first_eviction_step 0, flash backend.
#
# ------------------------------------------------------------------- sizing
# Full grid: 8 obs jobs (each 1 base + 7 arms), 16x3x3 LongBench + 13x3 RULER
# (e2/e3), 11x3 e1 jobs. At the measured wall times of the earlier sweeps
# (LongBench ~20-40 min/job, RULER 32k 49-88 min/task) that is roughly 1.5-2
# days on 8 A100s. SMOKE=1 runs a trimmed grid with capped examples in about an
# hour -- run it first; it exercises every job type and the report.
#
# ----------------------------------------------------------------- resuming
# Re-run the same command to redo only what is not done. DONE = .meta.json and
# the full line count (LongBench 200/150/500, RULER 500, or the SMOKE cap);
# obs = a .done marker written after all its arms observed. Anything else is
# deleted and re-run. A job refuses to start if HEAD moved since the run began.
# Do not run two copies against one OUT_ROOT (they would share a queue).
#
# ------------------------------------------------------------------- knobs
#   SUITES="obs e1 e2 e3"  SMOKE=0|1  OUT_ROOT=...  MODEL_PATH=...
#   LONGBENCH_LOCAL_DIR=...  RULER_DATA=...  CUDA_VISIBLE_DEVICES=...
#   LB_DATASETS  LB_BUDGETS="0.05 0.10 0.20"  RULER_TASKS  RULER_BUDGET=0.20
#   E1_GATES="0.10 0.50 1.0"  E1_LB  E1_RULER
#   OBS_DATASETS  OBS_ARMS  OBS_PREFILL=2048  OBS_GEN=1024  OBS_SAMPLES=8
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
export LONGBENCH_LOCAL_DIR="${LONGBENCH_LOCAL_DIR:-/home/ee/phd/eez228470/kv_cache/qwen_longbench_final_int2/RecoverKv/data/longbench}"
RULER_DATA="${RULER_DATA:-/home/ee/phd/eez228470/kv_cache/defensive_kv_new/DefensiveKV/defensivekv_dataset/ruler/32768}"
LB_CONFIG="configs/longbench_ours_flash_attn.yaml"
RULER_CONFIG="configs/ruler_niah_mk3_omega16.yaml"

# Pinned: the operating point. Not env-overridable, so a variable left exported
# from an earlier ablation cannot move this run.
QUANT_RATIO=0.70; QUANT_MODE=bytes; GATE=0.25; WINDOW=8; SINK=5
LOCAL=128; LOCAL_LOW=32      # LOCAL_LOW below budget 0.10 (sink+local must fit)

SMOKE="${SMOKE:-0}"
read -r -a SUITES <<< "${SUITES:-obs e1 e2 e3}"

if [[ "$SMOKE" == 1 ]]; then
  CAP=20
  LB_DATASETS="${LB_DATASETS:-hotpotqa gov_report passage_retrieval_en}"
  LB_BUDGETS="${LB_BUDGETS:-0.10}"
  RULER_TASKS="${RULER_TASKS:-niah_single_3 cwe}"
  E1_GATES="${E1_GATES:-1.0}"
  E1_LB="${E1_LB:-hotpotqa}"; E1_RULER="${E1_RULER:-niah_single_3}"
  OBS_DATASETS="${OBS_DATASETS:-hotpotqa ruler:niah_single_3}"
  OBS_ARMS="${OBS_ARMS:-gate0.25 gate1.0 oneway original}"
  OBS_PREFILL="${OBS_PREFILL:-1024}"; OBS_GEN="${OBS_GEN:-128}"; OBS_SAMPLES="${OBS_SAMPLES:-2}"
else
  CAP=""
  # Longest first (measured wall times of sweep_395af34 / the b010_b005 run).
  LB_DATASETS="${LB_DATASETS:-gov_report multi_news samsum qmsum narrativeqa repobench-p trec triviaqa passage_retrieval_en passage_count musique hotpotqa qasper lcc 2wikimqa multifieldqa_en}"
  LB_BUDGETS="${LB_BUDGETS:-0.05 0.10 0.20}"
  RULER_TASKS="${RULER_TASKS:-fwe cwe niah_multiquery niah_multikey_3 qa_1 qa_2 niah_single_3 niah_multivalue niah_multikey_2 vt niah_single_1 niah_multikey_1 niah_single_2}"
  E1_GATES="${E1_GATES:-0.10 0.50 1.0}"
  E1_LB="${E1_LB:-narrativeqa hotpotqa musique gov_report passage_retrieval_en lcc}"
  E1_RULER="${E1_RULER:-niah_single_3 niah_multikey_2 niah_multivalue cwe vt}"
  OBS_DATASETS="${OBS_DATASETS:-hotpotqa musique narrativeqa gov_report lcc passage_retrieval_en ruler:niah_single_3 ruler:cwe}"
  OBS_ARMS="${OBS_ARMS:-gate0.10 gate0.25 gate0.50 gate1.0 card4 oneway original}"
  OBS_PREFILL="${OBS_PREFILL:-2048}"; OBS_GEN="${OBS_GEN:-1024}"; OBS_SAMPLES="${OBS_SAMPLES:-8}"
fi
RULER_BUDGET="${RULER_BUDGET:-0.20}"
E1_BUDGET=0.20

COMMIT="$(git rev-parse --short HEAD)"; START_SHA="$(git rev-parse HEAD)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/recoverkv_${COMMIT}$([[ $SMOKE == 1 ]] && echo _smoke)}"

has_suite() { local s; for s in "${SUITES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }
die() { echo "error: $*" >&2; exit 2; }

# ---- preflight -----------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"     # 1 = print the job list and exit (no GPU needed)
for s in "${SUITES[@]}"; do
  [[ " obs e1 e2 e3 " == *" $s "* ]] || die "unknown suite '$s' (obs e1 e2 e3)"
done
if [[ "$DRY_RUN" != 1 ]]; then
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
[[ -f "$LB_CONFIG" && -f "$RULER_CONFIG" ]] || die "missing $LB_CONFIG or $RULER_CONFIG"
if has_suite e1 || has_suite e2 || has_suite e3; then
  for d in $LB_DATASETS $E1_LB; do
    [[ -s "$LONGBENCH_LOCAL_DIR/$d.jsonl" ]] || die "missing $LONGBENCH_LOCAL_DIR/$d.jsonl (the loader would silently fall back to HF)"
  done
  [[ -f "$RULER_DATA/dataset_info.json" ]] || die "RULER save_to_disk dir not found: $RULER_DATA"
fi
command -v nvidia-smi >/dev/null || die "no nvidia-smi -- login node? get a GPU node."
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
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
    sys.exit("error: flash_attn not importable; every run here is on the flash backend.")
from utils.cache_factory import assert_transformers_version_supported
assert_transformers_version_supported()
PY
fi   # DRY_RUN

ARM=0
grep -q '_EVICT_CONTROL_ARM_ENV = "STICKYKV_EVICT_CONTROL_ARM"' modules/windowed_cache/cache.py && ARM=1

# ---- the job list ------------------------------------------------------------------
# Accuracy arms: bidir (the shipped method, gate 0.25), oneway, original, gate<r>.
local_for() { awk -v b="$1" -v lo="$LOCAL_LOW" -v hi="$LOCAL" 'BEGIN{print (b+0 < 0.10) ? lo : hi}'; }
acc_overrides() {
  case "$1" in
    bidir)    echo "cache.quant_gate_ratio=$GATE" ;;
    oneway)   echo "cache.quant_gate_ratio=$GATE cache.quant_promotion=oneway" ;;
    original) echo "cache.quant_gate_ratio=$GATE cache.quant_promote_source=original" ;;
    gate*)    echo "cache.quant_gate_ratio=${1#gate}" ;;
    *) die "unknown accuracy arm $1" ;;
  esac
}

declare -A SEEN=()
JOBS=()
add() { local j="$*"; [[ -n "${SEEN[$j]:-}" ]] && return; SEEN[$j]=1; JOBS+=("$j"); }

if has_suite obs; then for d in $OBS_DATASETS; do add obs "$d"; done; fi
# RULER before LongBench: the RULER 32k tasks are the longest jobs.
if has_suite e2 || has_suite e3; then
  arms=(); has_suite e2 && arms+=(bidir oneway); has_suite e3 && arms+=(bidir original oneway)
  for t in $RULER_TASKS; do for a in "${arms[@]}"; do add ruler "$a" "$RULER_BUDGET" "$t"; done; done
fi
if has_suite e1; then
  for t in $E1_RULER; do for a in bidir $(printf 'gate%s ' $E1_GATES); do add ruler "$a" "$E1_BUDGET" "$t"; done; done
fi
if has_suite e2 || has_suite e3; then
  for d in $LB_DATASETS; do for b in $LB_BUDGETS; do for a in "${arms[@]}"; do add lb "$a" "$b" "$d"; done; done; done
fi
if has_suite e1; then
  for d in $E1_LB; do for a in bidir $(printf 'gate%s ' $E1_GATES); do add lb "$a" "$E1_BUDGET" "$d"; done; done
fi
(( ${#JOBS[@]} > 0 )) || die "no jobs for SUITES=${SUITES[*]}"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY RUN -- ${#JOBS[@]} jobs, out: $OUT_ROOT"
  printf '%s\n' "${JOBS[@]}" | awk '{k[$1]++} END {for (x in k) printf "  %-6s %d\n", x, k[x]}'
  printf '  %s\n' "${JOBS[@]}"
  exit 0
fi

mkdir -p "$OUT_ROOT"/{lb,ruler,obs}
QUEUE_POS="$OUT_ROOT/.queue_pos"; LOCK="$OUT_ROOT/.queue_lock"; FAILED="$OUT_ROOT/failed_jobs.txt"
echo 0 > "$QUEUE_POS"; : > "$FAILED"
printf '%s\n' "${JOBS[@]}" > "$OUT_ROOT/jobs.txt"
{
  echo "commit=$START_SHA started=$(date -Is) host=$(hostname) gpus=${GPUS[*]} smoke=$SMOKE"
  echo "suites=${SUITES[*]} model=$MODEL_PATH"
  echo "operating point: q=$QUANT_RATIO ($QUANT_MODE) gate=$GATE window=$WINDOW sink=$SINK local=$LOCAL (<0.10: $LOCAL_LOW)"
  echo "lb_budgets=$LB_BUDGETS ruler_budget=$RULER_BUDGET e1_gates=$E1_GATES"
  echo "obs_datasets=$OBS_DATASETS obs_arms=$OBS_ARMS prefill=$OBS_PREFILL gen=$OBS_GEN samples=$OBS_SAMPLES"
} > "$OUT_ROOT/run.env"

# ---- job runners -------------------------------------------------------------------
lb_expect() { local n; case "$1" in multifieldqa_en) n=150;; lcc|repobench-p) n=500;; *) n=200;; esac
              [[ -n "$CAP" ]] && (( CAP < n )) && n=$CAP; echo "$n"; }
ruler_expect() { if [[ -n "$CAP" ]]; then echo "$CAP"; else echo 500; fi; }
lines() { [[ -f "$1" ]] && wc -l < "$1" || echo 0; }
safe() { echo "$1" | tr -c 'A-Za-z0-9._-' '_'; }

is_done() {    # job words...
  local kind="$1"
  case "$kind" in
    obs)   [[ -f "$OUT_ROOT/obs/.done_$(safe "$2")" ]] ;;
    lb)    local d="$OUT_ROOT/lb/${2}_b${3}"
           [[ -f "$d/$4.meta.json" ]] && (( $(lines "$d/$4.jsonl") == $(lb_expect "$4") )) ;;
    ruler) local d="$OUT_ROOT/ruler/${2}_b${3}"
           [[ -f "$d/$4.meta.json" ]] && (( $(lines "$d/$4.jsonl") == $(ruler_expect) )) ;;
  esac
}

run_job() {    # gpu, job words...
  local gpu="$1"; shift
  local kind="$1" label="$*"
  if is_done "$@"; then echo "[$(date +%T)] gpu=$gpu SKIP   $label (complete)"; return 0; fi
  if [[ "$(git rev-parse HEAD)" != "$START_SHA" ]]; then
    echo "[$(date +%T)] gpu=$gpu REFUSE $label -- HEAD moved off $COMMIT" >&2
    echo "$label rc=HEAD-moved" >> "$FAILED"; return 1
  fi
  echo "[$(date +%T)] gpu=$gpu START  $label"
  local t0=$SECONDS rc log
  case "$kind" in
    obs)
      local d="$2" oroot; oroot="$OUT_ROOT/obs/$(safe "$d")"; mkdir -p "$oroot"
      log="$oroot.log"
      ( CUDA_VISIBLE_DEVICES="$gpu" DATASET="$d" ARMS="$OBS_ARMS" OUT_ROOT="$oroot" \
        MODEL_PATH="$MODEL_PATH" RULER_DATA="$RULER_DATA" \
        PREFILL="$OBS_PREFILL" GEN="$OBS_GEN" SAMPLES="$OBS_SAMPLES" \
        exec bash scripts/run_qevict_observations.sh ) > "$log" 2>&1
      rc=$?
      # Done only if every arm's observations were written.
      if (( rc == 0 )); then
        local missing=0 a
        for a in $OBS_ARMS; do
          compgen -G "$oroot/*/obs_${a}/all_results.json" > /dev/null || missing=1
        done
        (( missing == 0 )) && touch "$OUT_ROOT/obs/.done_$(safe "$d")" || rc=3
      fi ;;
    lb)
      local a="$2" b="$3" d="$4" dir L; dir="$OUT_ROOT/lb/${a}_b${b}"; mkdir -p "$dir"
      L="$(local_for "$b")"; log="$dir/$d.log"
      rm -f "$dir/$d.jsonl" "$dir/$d.meta.json"
      (
        if [[ $ARM == 1 ]]; then export STICKYKV_EVICT_CONTROL_ARM=1; fi
        # shellcheck disable=SC2046
        CUDA_VISIBLE_DEVICES="$gpu" exec python main.py --config "$LB_CONFIG" --override \
          model.name="$MODEL_PATH" cache.cache_budget="$b" \
          cache.quant_ratio="$QUANT_RATIO" cache.quant_budget_mode="$QUANT_MODE" \
          $(acc_overrides "$a") \
          cache.window_size="$WINDOW" window.window_size="$WINDOW" \
          cache.local_window_size="$L" window.local_window_size="$L" \
          cache.num_sink_tokens="$SINK" window.num_sink_tokens="$SINK" \
          cache.first_eviction_step=0 \
          longbench.datasets="[$d]" longbench.output_dir="$dir" \
          ${CAP:+longbench.num_samples=$CAP}
      ) > "$log" 2>&1
      rc=$? ;;
    ruler)
      local a="$2" b="$3" t="$4" dir L; dir="$OUT_ROOT/ruler/${a}_b${b}"; mkdir -p "$dir"
      L="$(local_for "$b")"; log="$dir/$t.log"
      rm -f "$dir/$t.jsonl" "$dir/$t.meta.json" "$dir/$t.memory.jsonl" "$dir/$t.memory_summary.json"
      (
        if [[ $ARM == 1 ]]; then export STICKYKV_EVICT_CONTROL_ARM=1; fi
        # shellcheck disable=SC2046
        CUDA_VISIBLE_DEVICES="$gpu" exec python main.py --config "$RULER_CONFIG" --override \
          model.name="$MODEL_PATH" cache.cache_budget="$b" \
          cache.quant_ratio="$QUANT_RATIO" cache.quant_budget_mode="$QUANT_MODE" \
          $(acc_overrides "$a") \
          cache.window_size="$WINDOW" window.window_size="$WINDOW" \
          cache.local_window_size="$L" window.local_window_size="$L" \
          cache.num_sink_tokens="$SINK" window.num_sink_tokens="$SINK" \
          cache.first_eviction_step=0 \
          ruler.data_dir="$RULER_DATA" ruler.tasks="[$t]" \
          ruler.num_samples="${CAP:-max}" ruler.resume=false \
          ruler.capture_memory=false ruler.output_dir="$dir"
      ) > "$log" 2>&1
      rc=$? ;;
  esac
  if (( rc == 0 )) && is_done "$@"; then
    echo "[$(date +%T)] gpu=$gpu DONE   $label ($(( (SECONDS - t0) / 60 )) min)"
  else
    local why
    why=$(grep -m1 -oE "ran EAGER|OutOfResources|OutOfMemoryError|NOT-GATED[-a-z]*|CUDA error[^.]*|[A-Za-z]+Error: [^.]{0,80}" "$log" 2>/dev/null | head -1)
    echo "[$(date +%T)] gpu=$gpu FAIL   $label rc=$rc ${why:+($why)} -- $log" >&2
    echo "$label rc=$rc ${why:-see log} $log" >> "$FAILED"
  fi
}

next_job() {
  local i
  { flock 9; i=$(<"$QUEUE_POS")
    if (( i < ${#JOBS[@]} )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
  } 9>"$LOCK"
}
worker() {
  local gpu="$1" i
  while i=$(next_job) && [[ -n "$i" ]]; do
    # shellcheck disable=SC2086
    run_job "$gpu" ${JOBS[$i]}
  done
}

echo "=== RecoverKV: ${#JOBS[@]} jobs on ${#GPUS[@]} GPU(s) | suites: ${SUITES[*]} | smoke=$SMOKE ==="
echo "    commit $COMMIT | out: $OUT_ROOT"
pids=()
for g in "${GPUS[@]}"; do
  worker "$g" 2>&1 | tee -a "$OUT_ROOT/worker_${g}.log" &
  pids+=($!)
done
wait "${pids[@]}"

# ---- score + report ------------------------------------------------------------------
echo; echo "=== scoring ==="
for d in "$OUT_ROOT"/lb/*/; do
  [[ -d "$d" ]] || continue
  python -m modules.evaluation.longbench_scoring --predictions_dir "$d" \
    --out_csv "${d}scores.csv" > "${d}scoring.log" 2>&1 || echo "scoring FAILED: $d" >&2
done
for d in "$OUT_ROOT"/ruler/*/; do
  [[ -d "$d" ]] || continue
  python -m modules.evaluation.ruler_scoring --predictions_dir "$d" \
    --out_csv "${d}scores.csv" > "${d}scoring.log" 2>&1 || echo "scoring FAILED: $d" >&2
done

echo; echo "=== report ==="
python -m modules.evaluation.recoverkv_report --root "$OUT_ROOT" --out "$OUT_ROOT/report" \
  2>&1 | tail -25

if [[ -s "$FAILED" ]]; then echo; echo "!! failed jobs (re-run the same command to redo them):"; cat "$FAILED"; exit 1; fi
echo; echo "done: $OUT_ROOT/report/report.html"
