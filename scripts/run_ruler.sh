#!/usr/bin/env bash
# scripts/run_ruler.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Run RULER — Ours (flash-attn backend), then score the predictions
# ============================================================================
# RULER data is NOT downloaded by this repo. Point DATA_DIR at an unpacked
# length bucket (the directory that holds the per-task .jsonl files), e.g.
#   DATA_DIR=/path/to/ruler/4096 bash scripts/run_ruler.sh
#
# Runs at the repo default first_eviction_step=0 — the prompt is compressed on
# decode step 0. RULER is the suite where that matters most: its answers are a
# handful of tokens, so a delayed first eviction would score every task at full
# cache regardless of cache_budget.
#
# Produces: outputs/ruler/4096_niah_mk3_omega16/{predictions,scores}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/ruler_niah_mk3_omega16.yaml}"
OUT_DIR="${OUT_DIR:-outputs/ruler/4096_niah_mk3_omega16}"

OVERRIDES=(ruler.output_dir="$OUT_DIR")
if [ -n "${DATA_DIR:-}" ]; then OVERRIDES+=(ruler.data_dir="$DATA_DIR"); fi

mkdir -p "$OUT_DIR"
{ git rev-parse HEAD 2>/dev/null || echo "no_git"; } > "$OUT_DIR/run.env"
pip freeze >> "$OUT_DIR/run.env"

python main.py --config "$CONFIG" --override "${OVERRIDES[@]}" "$@"

python -m modules.evaluation.ruler_scoring \
    --predictions_dir "$OUT_DIR" \
    --out_csv "$OUT_DIR/scores.csv"

echo
echo "Artifacts:"
echo "  $OUT_DIR/*.jsonl"
echo "  $OUT_DIR/*.meta.json"
echo "  $OUT_DIR/scores.csv"
