#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8       # required for deterministic CUDA

# ============================================================================
# Reproduce All — Run every evaluation suite in order
# ============================================================================
# This script runs all suites end-to-end and produces every output.
# Expected runtime: several hours on A100; longer on T4 (eager only).
#
# Usage:
#   bash scripts/reproduce_all.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/reproduce_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "StickyKV — Full Reproduction Pipeline"
echo "============================================"
echo "Start time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

# --- Suite A: Parity ---
echo "[1/8] Running parity baseline..."
bash "$SCRIPT_DIR/run_parity_base.sh"

echo "[2/8] Running parity ours (eager)..."
bash "$SCRIPT_DIR/run_parity_ours_eager.sh"

# Uncomment if flash-attn is available:
# echo "[3/8] Running parity ours (flash)..."
# bash "$SCRIPT_DIR/run_parity_ours_flash.sh"

# --- Suite B: Faithfulness ---
echo "[4/8] Running faithfulness evaluation..."
bash "$SCRIPT_DIR/run_faithfulness.sh"

# --- Suite B2: QEvict observations (reuses the parity pair from [1]/[2]) ---
echo "[4b/8] Running QEvict observation suite..."
STAGE=observe \
  BASE_NPZ="${BASE_NPZ:-outputs/parity_base_wikitext-103.npz}" \
  OURS_NPZ="${OURS_NPZ:-outputs/parity_ours_eager_wikitext-103.npz}" \
  bash "$SCRIPT_DIR/run_qevict_observations.sh"

# --- Suite C: Performance ---
echo "[5/8] Running performance benchmarks..."
bash "$SCRIPT_DIR/run_perf.sh"

# --- Visualization ---
echo "[6/8] Generating visualizations..."
bash "$SCRIPT_DIR/run_visualize.sh"

# --- Suite D: LongBench (Prompt 04) ---
echo "[7/8] Running LongBench full-cache baseline..."
bash "$SCRIPT_DIR/run_longbench_full_cache.sh"

echo "[8/8] Running LongBench ours (eager)..."
bash "$SCRIPT_DIR/run_longbench_ours_eager.sh"

# Uncomment if flash-attn is available:
# echo "[8b/8] Running LongBench ours (flash)..."
# bash "$SCRIPT_DIR/run_longbench_ours_flash_attn.sh"

# --- Scoring ---
echo "[Score] Scoring LongBench results..."
bash "$SCRIPT_DIR/score_longbench.sh"

echo ""
echo "============================================"
echo "All suites complete!"
echo "End time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Outputs in: outputs/"
echo "============================================"
