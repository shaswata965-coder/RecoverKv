#!/usr/bin/env bash
# evaluation/gsm8k/run_gsm8k_e2e.sh
#
# ============================================================================
# GSM8K end to end: build -> baseline -> gate -> press sweep -> comparison
# ============================================================================
# Everything needed for a controlled comparison of snapkv / adakv / defensivekv,
# in one command. Every run reads the SAME dataset directory (verified by
# dataset_sha) and decodes greedily with a fixed seed, so repeated invocations
# are deterministic.
#
#   bash gsm8k/run_gsm8k_e2e.sh                       # full 1319-example sweep
#   SAMPLES=20 bash gsm8k/run_gsm8k_e2e.sh            # ~5-minute smoke run
#   MODEL=/path/to/llama bash gsm8k/run_gsm8k_e2e.sh
#   PRESSES="snapkv defensivekv" RATIOS="0.5" bash gsm8k/run_gsm8k_e2e.sh
#   STAGE=score bash gsm8k/run_gsm8k_e2e.sh           # re-score, no generation
#
# Run it from the `evaluation/` directory (or let it cd there itself), so that
# `gsm8k.*` and `kvpress` are both importable.
#
# The baseline is a HARD GATE: if the full-cache run does not reach BASELINE_MIN
# accuracy the script stops before running any press, because a compressed
# number is only meaningful relative to a baseline that reproduces.
#
# On compression ratios
# ---------------------
# The default sweep stops at 0.7 on purpose. GSM8K prompts are ~150 tokens and
# these presses force-keep a 32-token observation window, so at ratio 0.8 the
# whole retained budget IS that window and every press collapses to "keep the
# last k tokens" -- the runner's pre-flight gate will refuse such a cell. To go
# higher, shrink the window too:
#
#   RATIOS="0.8 0.9" PRESS_WINDOW=8 bash gsm8k/run_gsm8k_e2e.sh
#
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$EVAL_ROOT"

PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-data/gsm8k_cot}"
RESULTS_DIR="${RESULTS_DIR:-results/gsm8k}"
SAMPLES="${SAMPLES:-max}"
MODEL="${MODEL:-${MODELS_DIR:-/models}/Meta-Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
STAGE="${STAGE:-all}"
BASELINE_MIN="${BASELINE_MIN:-70}"
PRESSES="${PRESSES:-snapkv adakv efficient_adasnapkv defensivekv layer_defensivekv}"
RATIOS="${RATIOS:-0.3 0.5 0.7}"
COMPRESS_QUESTIONS="${COMPRESS_QUESTIONS:-True}"
PRESS_WINDOW="${PRESS_WINDOW:-}"
ALLOW_DEGENERATE="${ALLOW_DEGENERATE:-False}"

COMMON=(--data_dir "$DATA_DIR" --model "$MODEL" --device "$DEVICE"
        --num_samples "$SAMPLES" --compress_questions "$COMPRESS_QUESTIONS")
if [ -n "$PRESS_WINDOW" ]; then COMMON+=(--press_window_size "$PRESS_WINDOW"); fi

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# ---------------------------------------------------------------------------
# 1. Build the dataset once. Every run reads this exact directory.
# ---------------------------------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "data" ]; then
  banner "1/4  Building GSM8K CoT dataset -> $DATA_DIR"
  if [ -f "$DATA_DIR/manifest.json" ]; then
    echo "  already present; reusing (delete $DATA_DIR to force a rebuild)"
  else
    $PY -m gsm8k.create_huggingface_dataset --out "$DATA_DIR"
  fi
fi

# ---------------------------------------------------------------------------
# 2. Baseline -- the reference every press is measured against.
# ---------------------------------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "baseline" ]; then
  banner "2/4  Full-cache baseline (no compression)"
  $PY -m gsm8k.run_gsm8k "${COMMON[@]}" \
      --press_name none --output_dir "$RESULTS_DIR/full_cache"

  banner "Baseline gate (require accuracy >= $BASELINE_MIN)"
  $PY - "$RESULTS_DIR/full_cache" "$BASELINE_MIN" <<'PYGATE'
import sys
from pathlib import Path
from gsm8k.calculate_metrics import score_run_dir

run_dir, floor = Path(sys.argv[1]), float(sys.argv[2])
rep, meta = score_run_dir(run_dir)
print(f"  accuracy      {rep.accuracy:.2f}")
print(f"  strict        {rep.accuracy_strict:.2f}")
print(f"  marker_rate   {rep.marker_rate:.1f}%")
print(f"  truncated     {rep.n_looks_truncated}/{rep.n_total}")
for w in rep.warnings:
    print(f"  !! {w}")

fail = []
if rep.accuracy < floor:
    fail.append(
        f"accuracy {rep.accuracy:.2f} < {floor}. Llama-3.1-8B-Instruct scores ~80-85 on "
        f"GSM8K with chain-of-thought. A number near 13 means the prompt is not "
        f"eliciting reasoning -- check that {run_dir}/meta.json has "
        f"prompt_style=chain_of_thought and dataset_max_new_tokens=512."
    )
if rep.marker_rate < 90.0:
    fail.append(
        f"marker_rate {rep.marker_rate:.1f}% < 90%. Generations are not reaching "
        f"'#### <number>'. Accuracy is being carried by the last-number fallback and is "
        f"not trustworthy."
    )
if rep.n_generation_failed:
    fail.append(f"{rep.n_generation_failed} generations FAILED outright.")

if fail:
    print("\n  BASELINE GATE FAILED -- not running any press:")
    for f in fail:
        print(f"    * {f}")
    sys.exit(1)
print("\n  baseline OK")
PYGATE
fi

# ---------------------------------------------------------------------------
# 3. The press sweep.
# ---------------------------------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "presses" ]; then
  for press in $PRESSES; do
    for ratio in $RATIOS; do
      out="$RESULTS_DIR/${press}_cr${ratio}"
      banner "3/4  press=$press compression_ratio=$ratio"
      # A pre-flight refusal (exit 1) is a configuration verdict, not a crash:
      # keep sweeping the other cells and let the comparison table show the gap.
      $PY -m gsm8k.run_gsm8k "${COMMON[@]}" \
          --press_name "$press" --compression_ratio "$ratio" \
          --allow_degenerate "$ALLOW_DEGENERATE" \
          --output_dir "$out" || echo "  !! cell skipped: $press @ $ratio"
    done
  done
fi

# ---------------------------------------------------------------------------
# 4. Score everything and print the comparison table.
# ---------------------------------------------------------------------------
banner "4/4  Comparison"
$PY -m gsm8k.calculate_metrics --runs "$RESULTS_DIR"/*/ \
    --out_csv "$RESULTS_DIR/comparison.csv"

echo
echo "Artifacts:"
echo "  $RESULTS_DIR/<run>/predictions.jsonl"
echo "  $RESULTS_DIR/<run>/meta.json"
echo "  $RESULTS_DIR/comparison.csv"
