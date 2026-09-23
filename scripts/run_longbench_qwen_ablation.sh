#!/usr/bin/env bash
# ============================================================================
# LongBench — Qwen2.5-7B: attribute the gap to the QEvict column
# ============================================================================
# The QEvict column and this branch's Qwen2.5 column share the operating point
# (quant_ratio 0.70, budget 0.20) and the protocol (bf16, YaRN 128K,
# untruncated), so what separates them is the machinery. Arms, same datasets,
# each into its own directory (a shared OUT_DIR would let one arm overwrite
# another's jsonl), scored and tabulated:
#
#   ours     configs/longbench_qwen_ours_flash_attn.yaml
#            the shipped method: read gate 0.25 (per-head union), one-byte grid
#            + key anchor, fused kernels. Identical to the _yarn128k runs --
#            re-running it measures run-to-run noise against those.
#   gate100  configs/longbench_qwen_ours_gate100.yaml
#            the same with every int2 window read exactly (quant_gate_ratio
#            1.0): QEvict's read path through this branch's code.
#   full     configs/longbench_qwen_full_cache.yaml (optional baseline)
#
# What each comparison says:
#   gate100 vs ours     -> what reading 25% of the int2 tier costs on Qwen
#   gate100 vs QEvict   -> what the rest of the machinery costs (one-byte
#                          grid + anchor, Triton score/decode kernels)
#
# Usage:
#   scripts/run_longbench_qwen_ablation.sh                  # default datasets
#   scripts/run_longbench_qwen_ablation.sh trec triviaqa    # choose datasets
#   ARMS="gate100" scripts/run_longbench_qwen_ablation.sh   # one arm only
#   ARMS="full ours gate100" ...                            # with the baseline
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

# The three tasks that fell against QEvict, plus two that did not (controls:
# a cause of the gap should move the first three and leave these alone).
if [ "$#" -gt 0 ]; then
    DATASETS="$*"
else
    DATASETS="trec triviaqa musique qasper multifieldqa_en"
fi
LIST="[$(echo "$DATASETS" | tr -s ' ' ',')]"
ARMS="${ARMS:-ours gate100}"
ROOT_OUT="outputs/longbench/qwen_ablation"

config_for() {
    case "$1" in
        ours)    echo "configs/longbench_qwen_ours_flash_attn.yaml" ;;
        gate100) echo "configs/longbench_qwen_ours_gate100.yaml" ;;
        full)    echo "configs/longbench_qwen_full_cache.yaml" ;;
        *) echo "unknown arm: $1 (ours|gate100|full)" >&2; exit 2 ;;
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
if [ "${#variants[@]}" -gt 1 ]; then
    python -m modules.evaluation.longbench_scoring \
        --out "$ROOT_OUT/comparison.csv" --baseline "$first" \
        --variants "${variants[@]:1}"
else
    python -m modules.evaluation.longbench_scoring \
        --out "$ROOT_OUT/comparison.csv" --baseline "$first"
fi
echo "Done. Table: $ROOT_OUT/comparison.csv (and .md)"
