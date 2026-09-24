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
#   gate50   configs/longbench_qwen_ours_gate50.yaml
#            the same reading half the tier (quant_gate_ratio 0.5).
#   q0       configs/longbench_qwen_ours_q0.yaml
#            the same eviction with NO int2 tier (quant_ratio 0.0): no card,
#            no fused two-tier kernel.
#   full     configs/longbench_qwen_full_cache.yaml (no compression)
#
# What each comparison says:
#   gate100 vs ours     -> what reading 25% of the int2 tier costs on Qwen
#   gate50 vs the two   -> whether that cost is the selection's capacity (rep 7
#                          shares one selection among seven heads) or the card
#   gate100 vs QEvict   -> what the rest of the machinery costs (one-byte
#                          grid + anchor, Triton score/decode kernels)
#   full vs ours        -> whether the gap is the method at all
#   q0 vs full / ours   -> eviction alone vs eviction + the int2 tier
#
# The gate arms have run: 0.25 / 0.50 / 0.75 average 46.98 / 46.88 / 46.71 over
# eight datasets -- flat, so the gate is not the gap (QWEN_PORT_PLAN.md
# Round 7). The next run is `ARMS="full q0"`.
#
# `ours` is not in the default ARMS: its 20% numbers on these datasets are
# already measured (the budget sweep in QWEN_PORT_PLAN.md), and its one rerun
# reproduced six of eight tasks to the digit and the other two within 0.31.
# Add it back to compare within one build.
#
# Usage:
#   scripts/run_longbench_qwen_ablation.sh                  # default datasets
#   scripts/run_longbench_qwen_ablation.sh trec triviaqa    # choose datasets
#   ARMS="gate100" scripts/run_longbench_qwen_ablation.sh   # one arm only
#   ARMS="full ours gate50 gate100" ...                     # everything
#
# Check each arm's <dataset>.meta.json before quoting it: read_gate.verdict
# must be "gated" and, on Qwen, read_gate.union "per-head" -- except `q0` and
# `full`, which have no int2 tier and must read "not-expected".
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# trec and triviaqa fell against QEvict at every budget (triviaqa is flat at
# ~81.6 from 5% to 20%); musique and narrativeqa got WORSE as the budget grew
# (20.7/23.5/25.2 and 18.3/23.7/22.7 at 20/10/5%) -- the signature of an error
# that scales with the number of unread windows. qasper is the control: it
# improved with budget and sits at parity with QEvict.
if [ "$#" -gt 0 ]; then
    DATASETS="$*"
else
    DATASETS="trec triviaqa musique narrativeqa qasper"
fi
LIST="[$(echo "$DATASETS" | tr -s ' ' ',')]"
ARMS="${ARMS:-full q0}"
ROOT_OUT="outputs/longbench/qwen_ablation"

config_for() {
    case "$1" in
        ours)    echo "configs/longbench_qwen_ours_flash_attn.yaml" ;;
        gate100) echo "configs/longbench_qwen_ours_gate100.yaml" ;;
        gate50)  echo "configs/longbench_qwen_ours_gate50.yaml" ;;
        q0)      echo "configs/longbench_qwen_ours_q0.yaml" ;;
        full)    echo "configs/longbench_qwen_full_cache.yaml" ;;
        *) echo "unknown arm: $1 (ours|gate100|gate50|q0|full)" >&2; exit 2 ;;
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
