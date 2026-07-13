#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Run Batched Performance Benchmarks
# ============================================================================
# Scenario 1 — standard batch  : prefill=2048,  decode=256, batch=64
# Scenario 2 — long context    : prefill=16384, decode=512, batch=4
#
# Metrics: TPOT (ms/token), TTFT (s), throughput (token/s)
# Outputs: outputs/perf_batched/perf_prefill*_gen*_bs*.npz
#          outputs/perf_batched/summary.txt
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$PROJECT_ROOT/outputs/perf_batched"

mkdir -p "$OUT_DIR"

# Capture environment snapshot
{
  git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "no_git"
  pip freeze 2>/dev/null || true
} > "$OUT_DIR/perf_batched.env"

echo "=== StickyKV Batched Performance Benchmark ==="
echo "Scenario 1: prefill=16384, decode=512, batch=4"
echo "Scenario 2: prefill=2048,  decode=256, batch=64"
echo ""

python "$PROJECT_ROOT/main.py" \
    --config "$PROJECT_ROOT/configs/eval_perf_batched.yaml" \
    "$@"

echo ""
echo "=== Summary ==="
python "$PROJECT_ROOT/scripts/print_perf_batched.py" \
    --npz-dir "$OUT_DIR" \
    --out "$OUT_DIR/summary.txt"
