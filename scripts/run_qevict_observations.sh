#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Run the QEvict Observation Suite (skew / window-level decisions / revival)
# ============================================================================
# Produces: $OUT_DIR/{observation*.csv, paper_results.md, all_results.json,
#                     qevict_observations.npz + .meta.json, *.pdf}
#
# Runs the full chain by default — parity base -> parity ours -> observations —
# because the observations need a *matched* pair and a base npz at schema >= 1.2
# (the fp32 per-step mass array; differencing the fp16 cumulative scores is
# noise past a few hundred decode steps).  Set STAGE=observe to reuse npzs you
# already have.
#
# Usage
# -----
#   bash scripts/run_qevict_observations.sh                  # full chain
#   STAGE=observe bash scripts/run_qevict_observations.sh    # analysis only
#   PROFILE=kaggle bash scripts/run_qevict_observations.sh   # T4/P100-sized
#   PROFILE=hpc    bash scripts/run_qevict_observations.sh   # A100-sized
#
# Every knob is an env var; the two profiles below just set defaults.
#   MODEL PREFILL GEN SAMPLES WINDOW SINK LOCAL BUDGET QUANT
#   FMM_HORIZON POOL_FACTOR PRIMARY_M PRIMARY_H PRIMARY_DELTA TRACE_AXIS
#   MAX_SAMPLES LAYER_STRIDE BOOTSTRAP SEED OUT_DIR
# ----------------------------------------------------------------------------
# Sizing notes
#   * The base run needs output_attentions=True, so prefill memory grows as
#     L x H x prefill^2.  PREFILL=2048 on 8B needs an A100; a T4/P100 wants
#     PREFILL<=1024.
#   * SAMPLES is the honest bootstrap axis.  With SAMPLES=1 the confidence
#     intervals are over layers of one prompt — fine for a sanity pass, not for
#     a paper number.  Use SAMPLES>=8 for reported results (TRACE_AXIS=sample).
#   * The observation step itself is CPU-only post-processing: seconds.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PROFILE="${PROFILE:-hpc}"
case "$PROFILE" in
  kaggle)
    : "${PREFILL:=1024}"; : "${GEN:=256}";  : "${SAMPLES:=4}"
    : "${FMM_HORIZON:=16}"; : "${LAYER_STRIDE:=1}"
    ;;
  hpc)
    : "${PREFILL:=2048}"; : "${GEN:=1024}"; : "${SAMPLES:=8}"
    : "${FMM_HORIZON:=32}"; : "${LAYER_STRIDE:=1}"
    ;;
  *) echo "Unknown PROFILE=$PROFILE (expected kaggle|hpc)" >&2; exit 2 ;;
esac

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
WINDOW="${WINDOW:-32}"          # must be divisible by 4 when QUANT > 0
SINK="${SINK:-4}"
LOCAL="${LOCAL:-256}"
BUDGET="${BUDGET:-0.25}"
QUANT="${QUANT:-0.5}"           # two-tier int2 split; 0 = fp16-only cache
SEED="${SEED:-42}"
STAGE="${STAGE:-all}"           # all | observe

POOL_FACTOR="${POOL_FACTOR:-2}"
PRIMARY_M="${PRIMARY_M:-4}"
PRIMARY_H="${PRIMARY_H:-8}"
PRIMARY_DELTA="${PRIMARY_DELTA:-1}"
TRACE_AXIS="${TRACE_AXIS:-sample_layer}"
BOOTSTRAP="${BOOTSTRAP:-2000}"

OUT_DIR="${OUT_DIR:-outputs/qevict_observations}"
BASE_NPZ="${BASE_NPZ:-outputs/qevict_parity_base.npz}"
OURS_NPZ="${OURS_NPZ:-outputs/qevict_parity_ours.npz}"

mkdir -p outputs "$OUT_DIR"
{
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo no_git)"
  echo "profile=$PROFILE model=$MODEL prefill=$PREFILL gen=$GEN samples=$SAMPLES"
  echo "window=$WINDOW sink=$SINK local=$LOCAL budget=$BUDGET quant=$QUANT seed=$SEED"
} > "$OUT_DIR/run.env"
pip freeze >> "$OUT_DIR/run.env" 2>/dev/null || true

GEOM_OVERRIDES=(
  "model.name=$MODEL"
  "run.seed=$SEED"
  "parity.prefill_len=$PREFILL" "parity.gen_len=$GEN"
  "data.prefill_len=$PREFILL"   "data.gen_len=$GEN"
  "data.num_samples=$SAMPLES"
  "window.window_size=$WINDOW"  "window.num_sink_tokens=$SINK"
  "window.local_window_size=$LOCAL"
  "cache.window_size=$WINDOW"   "cache.num_sink_tokens=$SINK"
  "cache.local_window_size=$LOCAL" "cache.cache_budget=$BUDGET"
)

if [ "$STAGE" = "all" ]; then
  echo "[1/3] Parity base (full cache, records step_window_scores)..."
  python main.py --config configs/eval_parity_base.yaml --override \
      "${GEOM_OVERRIDES[@]}" "output_path=$BASE_NPZ"

  echo "[2/3] Parity ours (two-tier cache, quant_ratio=$QUANT)..."
  python main.py --config configs/eval_parity_ours_eager.yaml --override \
      "${GEOM_OVERRIDES[@]}" "cache.quant_ratio=$QUANT" \
      "base_run_npz=$BASE_NPZ" "output_path=$OURS_NPZ"
else
  echo "[1-2/3] STAGE=observe — reusing $BASE_NPZ and $OURS_NPZ"
fi

echo "[3/3] QEvict observations..."
python -m modules.evaluation.qevict_observations \
    --base-npz "$BASE_NPZ" \
    --ours-npz "$OURS_NPZ" \
    --output-dir "$OUT_DIR" \
    --fmm-horizon "$FMM_HORIZON" \
    --pool-factor "$POOL_FACTOR" \
    --primary-inactivity "$PRIMARY_M" \
    --primary-lir-horizon "$PRIMARY_H" \
    --primary-transition-delta "$PRIMARY_DELTA" \
    --trace-axis "$TRACE_AXIS" \
    --layer-stride "$LAYER_STRIDE" \
    --bootstrap-samples "$BOOTSTRAP" \
    --seed "$SEED" \
    ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"} \
    "$@"

echo "Done — results in $OUT_DIR (start with paper_results.md)."
