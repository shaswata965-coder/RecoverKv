#!/usr/bin/env bash
# scripts/run_longbench_deepseek_v2_lite.sh
#
# ============================================================================
# LongBench on DeepSeek-V2-Lite-Chat: latent FullKV baseline, then ours at the
# operating point, then score both and refuse ungated sidecars.
# ============================================================================
#   bash scripts/run_longbench_deepseek_v2_lite.sh                 # both arms
#   ARMS=ours bash scripts/run_longbench_deepseek_v2_lite.sh       # one arm
#   MODEL=/path/to/DeepSeek-V2-Lite-Chat bash scripts/run_longbench_deepseek_v2_lite.sh
#   SAMPLES=5 bash scripts/run_longbench_deepseek_v2_lite.sh       # smoke run
#
# One GPU, on purpose: device_map="auto" on a multi-GPU box shards the model by
# layer, and the layer-major eviction refuses a cache spread over devices. The
# prefill of a 31.5k-token prompt materialises the remote model's full-sequence
# fp32 logits (~13 GB on top of ~31 GB of bf16 weights), so use an 80 GB card.
# Needs flash-attn and Triton on CUDA (the flash backend is kernel-or-error).
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PY="${PY:-python}"
ARMS="${ARMS:-full_cache ours}"
MODEL="${MODEL:-}"
SAMPLES="${SAMPLES:-max}"
LB_DIR="outputs/longbench"
FULL_OUT="$LB_DIR/deepseek_v2_lite_full_cache"
OURS_OUT="$LB_DIR/deepseek_v2_lite_ours_flash_compression_0.8"

OVERRIDES=(longbench.num_samples="$SAMPLES")
if [ -n "$MODEL" ]; then OVERRIDES+=(model.name="$MODEL"); fi

run_arm() {  # $1 = config, $2 = output dir
  mkdir -p "$2"
  { git rev-parse HEAD 2>/dev/null || echo "no_git"; } > "$2/run.env"
  pip freeze >> "$2/run.env" 2>/dev/null || true
  "$PY" main.py --config "configs/$1" --override "${OVERRIDES[@]}" longbench.output_dir="$2"
  "$PY" -m modules.evaluation.longbench_scoring --predictions_dir "$2" --out_csv "$2/scores.csv"
}

for arm in $ARMS; do
  case "$arm" in
    full_cache) run_arm longbench_deepseek_v2_lite_full_cache.yaml "$FULL_OUT" ;;
    ours)       run_arm longbench_deepseek_v2_lite_flash.yaml "$OURS_OUT" ;;
    *) echo "unknown arm: $arm (want full_cache and/or ours)" >&2; exit 2 ;;
  esac
done

# Every sidecar must prove what ran: `gated` for ours, `not-expected` for FullKV.
DIRS=()
for arm in $ARMS; do
  if [ "$arm" = full_cache ]; then DIRS+=("$FULL_OUT"); else DIRS+=("$OURS_OUT"); fi
done
GATE_STATUS=0
"$PY" scripts/check_read_gate.py "${DIRS[@]}" || GATE_STATUS=$?

if [ -f "$FULL_OUT/scores.csv" ] && [ -f "$OURS_OUT/scores.csv" ]; then
  "$PY" -m modules.evaluation.longbench_scoring \
      --out "$LB_DIR/deepseek_v2_lite_comparison.csv" \
      --baseline "$FULL_OUT/scores.csv" --variants "$OURS_OUT/scores.csv"
fi
exit "$GATE_STATUS"
