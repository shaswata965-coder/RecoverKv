#!/bin/bash
# StreamingLLM on LongBench (Qwen2.5-7B-Instruct) at 20% / 10% / 5% cache budgets
set -euo pipefail

export KVPRESS_DATASETS=/home/mfsgb10maxpro/datasets
export MODELS_DIR=/home/mfsgb10maxpro/model

DATASET="longbench"
DATA_DIR="${KVPRESS_DATASETS}/longbench/"
MODEL="${MODELS_DIR}/Qwen_2.5_7B"
PRESS="streaming_llm"
FRACTION=1.0                     # full samples of the selected tasks
TASKS="multi_news,passage_count,passage_retrieval_en,lcc,repobench-p"
LOG_DIR="logs"

# 0.80 = 20% cache, 0.90 = 10% cache, 0.95 = 5% cache
RATIOS=(0.80 0.90 0.95)

mkdir -p "${LOG_DIR}"

for RATIO in "${RATIOS[@]}"; do
  CACHE=$(python -c "print(f'{(1-${RATIO})*100:.0f}%')")
  OUT="${LOG_DIR}/longbench_qwen7b_${PRESS}_cr${RATIO}.log"

  echo "=========================================================="
  echo ">>> ${PRESS} @ cr=${RATIO} (cache=${CACHE})"
  echo "    model: ${MODEL}"
  echo "    tasks: ${TASKS}"
  echo "    log:   ${OUT}"
  echo "=========================================================="

  CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --model "${MODEL}" \
    --press_name "${PRESS}" \
    --compression_ratio "${RATIO}" \
    --device "cuda:0" \
    --fraction "${FRACTION}" \
    --tasks "${TASKS}" \
    > "${OUT}" 2>&1

  echo "    finished cr=${RATIO}"
done

echo ""
echo "All runs done. Score files:"
ls -1 results/*Qwen_2.5_7B*Streaming*.json 2>/dev/null || true
