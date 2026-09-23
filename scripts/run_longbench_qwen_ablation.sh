#!/usr/bin/env bash
# ============================================================================
# LongBench — Qwen2.5-7B: attribute the gap to the QEvict column in ONE run
# ============================================================================
# Three arms on the same datasets, each into its own directory (a shared
# OUT_DIR would let one arm overwrite another's jsonl), scored and tabulated:
#
#   ours     configs/longbench_qwen_ours_flash_attn.yaml
#            QEvict's protocol (bf16, YaRN 128K, no truncation) and split
#            (quant_ratio 0.5), with this branch's method: gate 0.25, per-head
#            union, one-byte grid + key anchor, fused kernels.
#   q070     configs/longbench_qwen_ours_q070.yaml
#            the same at the SHARED operating point, quant_ratio 0.70.
#   gate100  configs/longbench_qwen_ours_gate100.yaml
#            quant_ratio 0.5 with every int2 window read (quant_gate_ratio 1.0):
#            QEvict's read path through this branch's code.
#
# What each comparison says (QWEN_PORT_PLAN.md "Round 3"):
#   ours    vs QEvict   -> this branch's machinery at QEvict's operating point
#   q070    vs ours     -> the price of the 0.70 split on Qwen
#   gate100 vs ours     -> the price of reading 25% of the int2 tier
#
# Usage:
#   scripts/run_longbench_qwen_ablation.sh                  # default datasets
#   scripts/run_longbench_qwen_ablation.sh trec triviaqa    # choose datasets
#   ARMS="ours gate100" scripts/run_longbench_qwen_ablation.sh
#   ARMS="full ours" ...  # 'full' adds the full-cache baseline (same protocol)
#
# Check each arm's <dataset>.meta.json before quoting it: read_gate.verdict
# must be "gated" and, on Qwen, read_gate.union "per-head".
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# The three tasks that fell against QEvict, plus two broad-context canaries
# that the 0.70 split is predicted NOT to hurt.
if [ "$#" -gt 0 ]; then
    DATASETS="$*"
else
    DATASETS="trec triviaqa musique qasper multifieldqa_en"
fi
LIST="[$(echo "$DATASETS" | tr -s ' ' ',')]"
ARMS="${ARMS:-ours q070 gate100}"
ROOT_OUT="outputs/longbench/qwen_ablation"

config_for() {
    case "$1" in
        ours)    echo "configs/longbench_qwen_ours_flash_attn.yaml" ;;
        q070)    echo "configs/longbench_qwen_ours_q070.yaml" ;;
        gate100) echo "configs/longbench_qwen_ours_gate100.yaml" ;;
        full)    echo "configs/longbench_qwen_full_cache.yaml" ;;
        *) echo "unknown arm: $1 (ours|q070|gate100|full)" >&2; exit 2 ;;
    esac
}

# Validate every arm before spending GPU time on the first one.
for arm in $ARMS; do config_for "$arm" >/dev/null; done

variants=()
for arm in $ARMS; do
    cfg="$(config_for "$arm")"
    out="$ROOT_OUT/$arm"
    mkdir -p "$out"
    git rev-parse HEAD > "$out/run.env" 2>/dev/null || echo "no_git" > "$out/run.env"
    echo "=== arm=$arm config=$cfg datasets=$LIST -> $out"
    python "$PROJECT_ROOT/main.py" --config "$PROJECT_ROOT/$cfg" \
        --override "longbench.datasets=$LIST" "longbench.output_dir=$out"
    python -m modules.evaluation.longbench_scoring \
        --predictions_dir "$out" --out_csv "$out/scores.csv"
    variants+=("$out/scores.csv")
done

first="${variants[0]}"
rest=("${variants[@]:1}")
python -m modules.evaluation.longbench_scoring \
    --out "$ROOT_OUT/comparison.csv" --baseline "$first" --variants "${rest[@]}"
echo "Done. Table: $ROOT_OUT/comparison.csv (and .md)"
