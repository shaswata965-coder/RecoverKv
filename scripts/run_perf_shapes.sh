#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Shape sweep — 3 (prefill, decode) shapes x B = 1 / 32 / 64
# ============================================================================
#   A  prefill=1024  decode=1024   generation-dominated
#   B  prefill=2048  decode= 512   balanced
#   C  prefill=4096  decode= 256   prompt-dominated
#
# Usage:
#   scripts/run_perf_shapes.sh                 # full sweep (9 cells x 4 configs x 6 passes)
#   scripts/run_perf_shapes.sh --quick         # B=1,32 only, 2 runs — a smoke pass
#   scripts/run_perf_shapes.sh --shape A       # one shape only
#   scripts/run_perf_shapes.sh -- --override perf.cooldown_s=0
#
# Outputs: outputs/perf_shapes/perf_prefill*_gen*_bs*.npz
#          outputs/perf_shapes/shapes_report.txt
#
# COST. The full sweep is ~9 x 4 x 6 forward campaigns; shape A at B=64 is the
# expensive cell (1024 decode steps per pass). Budget several hours on one
# A100/H100 and run --quick first to confirm the grid fits before committing.
#
# REQUIRES transformers 4.47.x (utils/cache_factory.py refuses newer) and
# flash-attn (every config in this sweep is flash_attention_2; the eager backend
# cannot run these shapes — its score hook needs output_attentions, i.e. a
# [B,H,L,L] tensor).
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_ROOT/configs/eval_perf_shapes.yaml"
OUT_DIR="$PROJECT_ROOT/outputs/perf_shapes"

QUICK=0; SHAPE=""; PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --shape) SHAPE="$2"; shift 2 ;;
        --out)   OUT_DIR="$2"; shift 2 ;;
        --)      shift; PASSTHROUGH=("$@"); break ;;
        -h|--help) sed -n '7,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT_DIR"
{ git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo no_git; \
  pip freeze 2>/dev/null || true; } > "$OUT_DIR/perf_shapes.env"

OVERRIDES=("telemetry.output_dir=$OUT_DIR")

# --shape selects one scenario by rewriting the grid. gen_len = decode + 1.
case "$SHAPE" in
    A) GRID='[{prefill_len: 1024, gen_len: 1025, batch_size: 1},{prefill_len: 1024, gen_len: 1025, batch_size: 32},{prefill_len: 1024, gen_len: 1025, batch_size: 64}]' ;;
    B) GRID='[{prefill_len: 2048, gen_len: 513, batch_size: 1},{prefill_len: 2048, gen_len: 513, batch_size: 32},{prefill_len: 2048, gen_len: 513, batch_size: 64}]' ;;
    C) GRID='[{prefill_len: 4096, gen_len: 257, batch_size: 1},{prefill_len: 4096, gen_len: 257, batch_size: 32},{prefill_len: 4096, gen_len: 257, batch_size: 64}]' ;;
    "") GRID="" ;;
    *) echo "--shape expects A, B or C" >&2; exit 2 ;;
esac
[[ -n "$GRID" ]] && OVERRIDES+=("perf.grid=$GRID")

if [[ "$QUICK" == "1" ]]; then
    # Drop B=64 and cut the run count. Shapes stay put so the cells are the
    # same cells, just fewer of them — a quick pass must be comparable to the
    # full one, not a different measurement.
    OVERRIDES+=("perf.grid=[]"
                "perf.prefill_lengths=[1024,2048,4096]"
                "perf.gen_lengths=[1025,513,257]"
                "perf.batch_sizes=[1,32]"
                "perf.num_measurement_runs=2"
                "perf.cooldown_s=1.0")
    echo "!! --quick: cartesian product of the shape axes, so it also measures"
    echo "   off-diagonal cells (e.g. prefill=4096 gen=1025). Read only the"
    echo "   three diagonal shapes from the report."
fi

echo "=== StickyKV shape sweep ==="
echo "    config:    $CONFIG"
echo "    out:       $OUT_DIR"
echo "    overrides: ${OVERRIDES[*]}"
echo ""

python "$PROJECT_ROOT/main.py" --config "$CONFIG" \
    --override "${OVERRIDES[@]}" \
    ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}

echo ""
echo "=== Report ==="
python "$PROJECT_ROOT/scripts/print_perf_shapes.py" \
    --npz-dir "$OUT_DIR" \
    --out "$OUT_DIR/shapes_report.txt"
