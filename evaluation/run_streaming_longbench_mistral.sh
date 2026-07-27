#!/bin/bash
# StreamingLLM @ 20% cache on LongBench (Mistral-7B-Instruct-v0.2)
# Tasks: multi_news, passage_count, passage_retrieval_en, lcc, repobench-p
set -euo pipefail

export KVPRESS_DATASETS=/home/mfsgb10maxpro/datasets
export MODELS_DIR=/home/mfsgb10maxpro/model

DATASET="longbench"
DATA_DIR="${KVPRESS_DATASETS}/longbench/"
MODEL="${MODELS_DIR}/Mistral_v0.2_7B/Mistral-7B-Instruct-v0.2"
PRESS="streaming_llm"
RATIO=0.80                       # 0.80 = 20% cache budget
FRACTION=1.0                     # full samples of the selected tasks
TASKS="multi_news,passage_count,passage_retrieval_en,lcc,repobench-p"
LOG_DIR="logs"

mkdir -p "${LOG_DIR}"
OUT="${LOG_DIR}/longbench_mistral7b_${PRESS}_cr${RATIO}.log"

echo "Running ${PRESS} @ cr=${RATIO} (20% cache) on ${DATASET}"
echo "  model:  ${MODEL}"
echo "  tasks:  ${TASKS}"
echo "  log:    ${OUT}"

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

echo "Done. Scores JSON:"
ls -1 results/*Mistral*Streaming* 2>/dev/null || true
