#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Run Batched Performance Benchmarks — the max-B search
# ============================================================================
# Runs configs/eval_perf_batched.yaml: every config at every rung of a batch
# ladder, with skip_if_oom: true. A config's max-B is the largest batch_size it
# did NOT OOM at, and the headline is (max-B) x (decode tok/s per row). See that
# file's header for the grid and BATCHING_PLAN.md §5 for why B=1 cannot show
# this method working.
#
# Scenarios in the shipped grid:
#   A  prefill=512   gen=1024   B = 1, 8, 32, 128, 256, 512   long generation
#   B  prefill=512   gen=4096   B = 1, 32, 128, 256           generation-dominated
#   C  prefill=4096  gen=128    B = 1, 16, 32, 64             LongBench shape
#
# Override the ladder without editing the config, e.g. one scenario only:
#   scripts/run_perf_batched.sh --override perf.num_measurement_runs=1
#
# Metrics: TPOT (ms/token), TTFT (s), throughput (token/s), and peak memory —
# torch allocated / reserved, device-level (what an OOM trips on), and split by
# prefill vs decode phase (utils/cache_memory.py::MemoryProbe).
# Outputs: outputs/perf_batched/perf_prefill*_gen*_bs*.npz
#          outputs/perf_batched/summary.txt   <- includes the MAX-B SUMMARY table
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

echo "=== StickyKV Batched Performance Benchmark (max-B search) ==="
echo "A: prefill=512  gen=1024  B=1,8,32,128,256,512"
echo "B: prefill=512  gen=4096  B=1,32,128,256"
echo "C: prefill=4096 gen=128   B=1,16,32,64"
echo ""

python "$PROJECT_ROOT/main.py" \
    --config "$PROJECT_ROOT/configs/eval_perf_batched.yaml" \
    "$@"

echo ""
echo "=== Summary ==="
python "$PROJECT_ROOT/scripts/print_perf_batched.py" \
    --npz-dir "$OUT_DIR" \
    --out "$OUT_DIR/summary.txt"
