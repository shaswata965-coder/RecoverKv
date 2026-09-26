#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# QEvict observation suite, one dataset: parity base -> parity ours -> I..V
# ============================================================================
# Five observations over one matched (base, ours) parity pair
# (modules/evaluation/qevict_observations.py):
#   I    skewed importance            III  revival (oracle + policy fp)
#   II   future missed mass / churn   IV   the decode READ ledger + per-head
#                                          gate recall (needs the gated run)
#   V    tier dynamics: promotion / demotion / eviction, and whether each
#        move landed on the attention that arrived next
#
# The ours run is on the FLASH backend, because that is the only path where
# the read gate exists (flash_decode.expect_gated): an eager ours run reads the
# whole int2 tier and Observation IV would describe a method we do not ship.
# Each ARM is its own ours run against ONE shared base run:
#   gate<r>      read-gate ratio r (gate1.0 is the gate's control arm: it
#                selects every window through the same code)
#   card<spec>   gate 0.25 with card widths <spec> (4, 2, or mu4v4t2vm2 ->
#                "mu=4,v=4,t=2,vm=2"); under bytes mode a quality change
#   oneway       gate 0.25, quant_promotion=oneway (F -> Q -> E, no promotion)
#   original     gate 0.25, quant_promote_source=original (promotion oracle:
#                exact fp windows from a shadow OUTSIDE the budget)
# Default ARMS="gate0.25 gate1.0". GATES / CARD_BITS still work as before.
#
# Usage
# -----
#   DATASET=wikitext-103              scripts/run_qevict_observations.sh
#   DATASET=narrativeqa               scripts/run_qevict_observations.sh  # LongBench
#   DATASET=ruler:niah_single_3       scripts/run_qevict_observations.sh  # RULER task
#   DATASET=/path/to/corpus.jsonl TEXT_FIELD=body scripts/run_qevict_observations.sh
#   STAGE=observe DATASET=...         scripts/run_qevict_observations.sh  # re-analyse
#   ARMS="gate0.10 gate0.25 gate0.50 gate1.0 card4 oneway original" DATASET=... \
#                                     scripts/run_qevict_observations.sh
#
# DATASET resolves to a parity corpus:
#   wikitext-103 | pg19           HF corpora
#   <LongBench name>              $LONGBENCH_LOCAL_DIR/<name>.jsonl, field "context"
#   ruler:<task>                  $RULER_DATA (save_to_disk), field "context",
#                                 records with task == <task>
#   <path>                        a local .jsonl/.json/.txt/dir (TEXT_FIELD optional)
# The prompt is the first PREFILL tokens of each document (documents shorter
# than PREFILL are skipped) and the continuation is the base run's greedy
# decode, teacher-forced into ours. So this characterises attention on that
# dataset's text; it is not the task's prompt template and not a task score.
#
# Knobs (env): MODEL_PATH PREFILL GEN SAMPLES ARTICLE_INDEX BUDGET QUANT
#   QUANT_MODE ARMS (or GATES + CARD_BITS) WINDOW LOCAL SINK FMM_HORIZON PRIMARY_M
#   PRIMARY_H PRIMARY_DELTA TRACE_AXIS LAYER_STRIDE BOOTSTRAP SEED STAGE
#   OUT_ROOT FORCE
#
# Sizing (Llama-3.1-8B, 32 layers x 32 heads, fp16):
#   The base run needs output_attentions (eager): L x H x PREFILL^2 per forward,
#   ~9 GB at PREFILL=2048 -- fine on an 80 GB A100. The base npz holds two
#   [S, T, L, H, W] fp16 arrays (cumulative + per-head step mass): at ws=8,
#   PREFILL 2048, GEN 1024, SAMPLES 8 that is ~13 GB uncompressed, and the
#   observation step loads it into host RAM. Halve GEN or raise LAYER_STRIDE
#   if RAM is tight. SAMPLES is the honest bootstrap axis -- use TRACE_AXIS=sample
#   for a reported number (>= 8 samples).
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---- your cluster defaults (same as the RULER / LongBench scripts) ----------
MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
LONGBENCH_LOCAL_DIR="${LONGBENCH_LOCAL_DIR:-/home/ee/phd/eez228470/kv_cache/qwen_longbench_final_int2/RecoverKv/data/longbench}"
RULER_DATA="${RULER_DATA:-/home/ee/phd/eez228470/kv_cache/defensive_kv_new/DefensiveKV/defensivekv_dataset/ruler/32768}"

# ---- the operating point (utils.config.OPERATING_POINT) ---------------------
DATASET="${DATASET:-wikitext-103}"
BUDGET="${BUDGET:-0.20}"
QUANT="${QUANT:-0.70}"
QUANT_MODE="${QUANT_MODE:-bytes}"
CARD_BITS="${CARD_BITS:-}"         # legacy: card widths for every GATES arm
if [[ -n "${ARMS:-}" ]]; then
  read -r -a ARM_LIST <<< "$ARMS"
else                               # legacy GATES (+ CARD_BITS) spelling
  ARM_LIST=()
  for g in ${GATES:-0.25 1.0}; do
    if [[ -n "$CARD_BITS" ]]; then ARM_LIST+=("gate${g}_card${CARD_BITS}")
    else ARM_LIST+=("gate${g}"); fi
  done
fi
WINDOW="${WINDOW:-8}"
LOCAL="${LOCAL:-128}"
SINK="${SINK:-5}"

PREFILL="${PREFILL:-2048}"
GEN="${GEN:-1024}"
SAMPLES="${SAMPLES:-8}"
ARTICLE_INDEX="${ARTICLE_INDEX:-0}"
SEED="${SEED:-42}"

FMM_HORIZON="${FMM_HORIZON:-32}"
PRIMARY_M="${PRIMARY_M:-4}"
PRIMARY_H="${PRIMARY_H:-8}"
PRIMARY_DELTA="${PRIMARY_DELTA:-1}"
TRACE_AXIS="${TRACE_AXIS:-sample}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
STAGE="${STAGE:-all}"              # all | base | ours | observe
FORCE="${FORCE:-0}"                # 1 = redo stages whose outputs exist

die() { echo "error: $*" >&2; exit 2; }

# ---- resolve the dataset ----------------------------------------------------
TEXT_FIELD="${TEXT_FIELD:-}"
RECORD_FILTER="${RECORD_FILTER:-}"
LB_NAMES=" narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p "
case "$DATASET" in
  wikitext-103|pg19) CORPUS="$DATASET"; SLUG="$DATASET" ;;
  ruler:*)
    task="${DATASET#ruler:}"
    [[ -f "$RULER_DATA/dataset_info.json" ]] || die "RULER save_to_disk dir not found: $RULER_DATA"
    CORPUS="$RULER_DATA"; TEXT_FIELD="${TEXT_FIELD:-context}"
    RECORD_FILTER="task=$task"; SLUG="ruler$(basename "$RULER_DATA")-$task" ;;
  *)
    if [[ "$LB_NAMES" == *" $DATASET "* ]]; then
      CORPUS="$LONGBENCH_LOCAL_DIR/$DATASET.jsonl"; TEXT_FIELD="${TEXT_FIELD:-context}"
      [[ -s "$CORPUS" ]] || die "missing $CORPUS"
      SLUG="lb-$DATASET"
    elif [[ -e "$DATASET" ]]; then
      CORPUS="$DATASET"; SLUG="$(basename "${DATASET%.*}")"
    else
      die "DATASET=$DATASET is not wikitext-103, pg19, a LongBench name, ruler:<task>, or a path"
    fi ;;
esac

# ---- preflight ----------------------------------------------------------------
if [[ "$STAGE" != "observe" && ! -e "$MODEL_PATH" ]]; then
  echo "note: $MODEL_PATH is not a local path; treating it as a HF hub id" >&2
fi
(( WINDOW % 4 == 0 )) || die "WINDOW=$WINDOW must be a multiple of 4 when QUANT > 0 (int2 packing)"
(( LOCAL % WINDOW == 0 )) || die "LOCAL=$LOCAL is not a multiple of WINDOW=$WINDOW"
(( ${#ARM_LIST[@]} > 0 )) || die "ARMS is empty"
if [[ "$STAGE" != "observe" ]]; then
  python - <<'PY' || exit 2
import sys
try:
    import torch
except ImportError:
    sys.exit("error: torch not importable -- 'conda activate sticky_env'")
if not torch.cuda.is_available():
    sys.exit("error: no CUDA device -- the read gate only exists on the CUDA flash path.")
try:
    import flash_attn  # noqa: F401
except ImportError:
    sys.exit("error: flash_attn not importable; the ours run needs flash_attention_2.")
from utils.cache_factory import assert_transformers_version_supported
assert_transformers_version_supported()
PY
fi

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/qevict_obs_${COMMIT}}"
RUN_DIR="$OUT_ROOT/${SLUG}_p${PREFILL}_g${GEN}_w${WINDOW}_b${BUDGET}_q${QUANT}"
BASE_NPZ="$RUN_DIR/parity_base.npz"
mkdir -p "$RUN_DIR"
{
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo nogit) started=$(date -Is) host=$(hostname)"
  echo "dataset=$DATASET corpus=$CORPUS text_field=${TEXT_FIELD:-auto} record_filter=${RECORD_FILTER:-none}"
  echo "model=$MODEL_PATH prefill=$PREFILL gen=$GEN samples=$SAMPLES article_index=$ARTICLE_INDEX seed=$SEED"
  echo "budget=$BUDGET q=$QUANT mode=$QUANT_MODE arms=${ARM_LIST[*]} window=$WINDOW local=$LOCAL sink=$SINK"
} > "$RUN_DIR/run.env"

COMMON=(
  "model.name=$MODEL_PATH" "run.seed=$SEED"
  "parity.dataset=$CORPUS" "data.dataset=$CORPUS"
  "parity.article_index=$ARTICLE_INDEX"
  "parity.prefill_len=$PREFILL" "parity.gen_len=$GEN"
  "data.prefill_len=$PREFILL" "data.gen_len=$GEN" "data.num_samples=$SAMPLES"
  "window.window_size=$WINDOW" "window.num_sink_tokens=$SINK"
  "window.local_window_size=$LOCAL"
  "cache.window_size=$WINDOW" "cache.num_sink_tokens=$SINK"
  "cache.local_window_size=$LOCAL" "cache.cache_budget=$BUDGET"
  "telemetry.output_dir=$RUN_DIR"
)
if [[ -n "$TEXT_FIELD" ]]; then COMMON+=("parity.text_field=$TEXT_FIELD"); fi
if [[ -n "$RECORD_FILTER" ]]; then COMMON+=("parity.record_filter=$RECORD_FILTER"); fi

# An arm name -> its cache overrides. Every arm not naming a gate ratio runs at
# the shipped 0.25, so each differs from gate0.25 in exactly one knob.
card_spec() {   # "4" | "2" | "mu4v4t2vm2" -> quant_card_bits value
  local c="$1"
  if [[ "$c" =~ ^[0-9]+$ ]]; then echo "$c"
  else echo "$c" | sed -E 's/([a-z]+)([0-9])/\1=\2,/g; s/,$//'; fi
}
arm_overrides() {
  local a="$1" gate="0.25" out=()
  case "$a" in
    gate*_card*) gate="${a#gate}"; gate="${gate%%_card*}"
                 out+=("cache.quant_card_bits=$(card_spec "${a##*_card}")") ;;
    gate*)       gate="${a#gate}" ;;
    card*)       out+=("cache.quant_card_bits=$(card_spec "${a#card}")") ;;
    oneway)      out+=("cache.quant_promotion=oneway") ;;
    original)    out+=("cache.quant_promote_source=original") ;;
    *)           die "unknown arm '$a' (gate<r> | card<spec> | oneway | original)" ;;
  esac
  [[ "$gate" =~ ^[0-9.]+$ ]] || die "arm '$a': bad gate ratio '$gate'"
  out+=("cache.quant_gate_ratio=$gate")
  printf '%s\n' "${out[@]}"
}
for a in "${ARM_LIST[@]}"; do arm_overrides "$a" > /dev/null; done   # validate early

npz_ok() { [[ -s "$1" ]] && python -c "import numpy as np,sys; np.load(sys.argv[1], allow_pickle=True).files" "$1" 2>/dev/null; }

# ---- 1. base: full cache, eager attentions, per-head step mass ---------------
if [[ "$STAGE" == "all" || "$STAGE" == "base" ]]; then
  if [[ "$FORCE" != 1 ]] && npz_ok "$BASE_NPZ"; then
    echo "[base] $BASE_NPZ exists -- skipping (FORCE=1 to redo)"
  else
    echo "[base] full-KV parity base on $DATASET ..."
    python main.py --config configs/eval_parity_base.yaml --override \
      "${COMMON[@]}" "parity.record_head_step_mass=true" \
      "output_path=$BASE_NPZ" 2>&1 | tee "$RUN_DIR/base.log"
  fi
fi

# ---- 2. ours: flash backend (the gated read path), one run per gate ratio ----
if [[ "$STAGE" == "all" || "$STAGE" == "ours" ]]; then
  npz_ok "$BASE_NPZ" || die "no base npz at $BASE_NPZ -- run STAGE=base first"
  for a in "${ARM_LIST[@]}"; do
    OURS_NPZ="$RUN_DIR/parity_ours_${a}.npz"
    if [[ "$FORCE" != 1 ]] && npz_ok "$OURS_NPZ"; then
      echo "[ours $a] exists -- skipping"; continue
    fi
    mapfile -t EXTRA < <(arm_overrides "$a")
    echo "[ours $a] three-tier cache, q=$QUANT ($QUANT_MODE): ${EXTRA[*]} ..."
    python main.py --config configs/eval_parity_ours_flash.yaml --override \
      "${COMMON[@]}" \
      "cache.quant_ratio=$QUANT" "cache.quant_budget_mode=$QUANT_MODE" \
      "cache.first_eviction_step=0" "parity.record_gate=true" \
      "${EXTRA[@]}" \
      "base_run_npz=$BASE_NPZ" "output_path=$OURS_NPZ" 2>&1 | tee "$RUN_DIR/ours_${a}.log"
  done
fi

# ---- 3. observations I..V, per gate ratio ------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "observe" ]]; then
  for a in "${ARM_LIST[@]}"; do
    OURS_NPZ="$RUN_DIR/parity_ours_${a}.npz"
    OBS_DIR="$RUN_DIR/obs_${a}"
    npz_ok "$OURS_NPZ" || { echo "[observe $a] no $OURS_NPZ -- skipped" >&2; continue; }
    echo "[observe $a] ..."
    python -m modules.evaluation.qevict_observations \
      --base-npz "$BASE_NPZ" --ours-npz "$OURS_NPZ" \
      --output-dir "$OBS_DIR" \
      --fmm-horizon "$FMM_HORIZON" \
      --primary-inactivity "$PRIMARY_M" --primary-lir-horizon "$PRIMARY_H" \
      --primary-transition-delta "$PRIMARY_DELTA" \
      --trace-axis "$TRACE_AXIS" --layer-stride "$LAYER_STRIDE" \
      --bootstrap-samples "$BOOTSTRAP" --seed "$SEED" \
      > "$OBS_DIR.log" 2>&1 \
      || { echo "[observe $a] FAILED -- $OBS_DIR.log" >&2; continue; }
  done

  RUN_DIR="$RUN_DIR" ARMS="${ARM_LIST[*]}" python - <<'PY' | tee "$RUN_DIR/summary.txt"
import json, os
d = os.environ["RUN_DIR"]
def pct(x): return "   n/a" if x is None else f"{100 * x:6.1f}%"
def num(x): return "   n/a" if x is None else f"{x:6.2f}"
print(f"QEvict observations -- {os.path.basename(d)}")
print(f"{'arm':>14} {'path':>10} {'recall':>7} {'oracle':>7} {'worst':>7} "
      f"{'skipped':>8} {'evicted':>8} {'promote':>8} {'swap':>6} {'fidelity':>9}")
for a in os.environ["ARMS"].split():
    f = f"{d}/obs_{a}/all_results.json"
    if not os.path.exists(f):
        print(f"{a:>14}  (no results)"); continue
    r = json.load(open(f))
    o4, o5 = r.get("observation4", {}), r.get("observation5", {})
    rs = o4.get("recall_summary", {})
    led = {x["part"]: x["share_of_head_mass"] for x in o4.get("ledger_table", [])}
    sm = o5.get("summary", {})
    lift = {x["move"]: x["lift"] for x in o5.get("outcome_table", [])}
    print(f"{a:>14} {r['metadata'].get('read_path', '?'):>10} "
          f"{pct(rs.get('head_recall_mean')):>7} {pct(rs.get('oracle_recall_mean')):>7} "
          f"{pct(rs.get('worst_head_overall')):>7} {pct(led.get('q_skipped')):>8} "
          f"{pct(led.get('evicted')):>8} {num(lift.get('promote')):>8} "
          f"{num(sm.get('swap_gain_mean')):>6} {pct(sm.get('decision_fidelity_mean')):>9}")
print("\nrecall/oracle/worst: per-head int2 recall of the gate, the same-size "
      "hindsight pick, and the worst (trace, head); skipped/evicted: share of "
      "head mass read only through centroids / not at all; promote: future-mass "
      "lift of promoted windows; swap: lift(promote) - lift(demote).")
print(f"Full reports: {d}/obs_*/paper_results.md")
PY
fi
echo "done -- $RUN_DIR"
