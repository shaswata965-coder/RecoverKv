#!/usr/bin/env bash
# scripts/run_gsm8k_e2e.sh
#
# ============================================================================
# GSM8K end-to-end: build -> baseline -> 80% budget -> 20% budget -> compare
# ============================================================================
# Everything needed for a controlled budget comparison, in one command. All
# three runs read the SAME dataset directory (verified by dataset_sha) and use
# greedy decoding with a fixed seed, so repeated invocations are deterministic.
#
#   bash scripts/run_gsm8k_e2e.sh                    # full 1319-example run
#   SAMPLES=20 bash scripts/run_gsm8k_e2e.sh         # 5-minute smoke run
#   MODEL=/path/to/llama bash scripts/run_gsm8k_e2e.sh
#   STAGE=score bash scripts/run_gsm8k_e2e.sh        # re-score, no generation
#
#   # DeepSeek-V2-Lite-Chat (latent FullKV baseline + the operating point):
#   CFG_PREFIX=gsm8k_deepseek_v2_lite ARMS=budget20 BASELINE_MIN=60 \
#       bash scripts/run_gsm8k_e2e.sh
#
# CFG_PREFIX picks configs/<prefix>_full_cache.yaml and configs/<prefix>_<arm>.yaml
# for each arm in ARMS, and the default results dir outputs/<prefix>. The
# DeepSeek BASELINE_MIN is a placeholder, NOT a measured number: no GSM8K run
# of DeepSeek-V2-Lite-Chat with this prompt has been made on this repo.
#
# The baseline is a HARD GATE: if it does not reach BASELINE_MIN accuracy the
# script stops before running any budget, because a budget number is only
# meaningful relative to a baseline that reproduces.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-data/gsm8k_cot}"
CFG_PREFIX="${CFG_PREFIX:-gsm8k}"
ARMS="${ARMS:-budget80 budget20}"
RESULTS_DIR="${RESULTS_DIR:-outputs/${CFG_PREFIX}}"
SAMPLES="${SAMPLES:-max}"
MODEL="${MODEL:-}"
STAGE="${STAGE:-all}"
BASELINE_MIN="${BASELINE_MIN:-70}"

OVERRIDES=(gsm8k.num_samples="$SAMPLES" gsm8k.data_dir="$DATA_DIR" gsm8k.results_dir="$RESULTS_DIR")
if [ -n "$MODEL" ]; then OVERRIDES+=(model.name="$MODEL"); fi

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# ---------------------------------------------------------------------------
# 1. Build the dataset once. Every run reads this exact directory.
# ---------------------------------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "data" ]; then
  banner "1/5  Building GSM8K CoT dataset -> $DATA_DIR"
  if [ -f "$DATA_DIR/manifest.json" ]; then
    echo "  already present; reusing (delete $DATA_DIR to force a rebuild)"
  else
    $PY -m data.gsm8k_loader --out "$DATA_DIR"
  fi
fi

run_one() {  # $1 = config basename, $2 = output subdir, $3 = human label
  local cfg="$1" out="$2" label="$3"
  banner "$label"
  mkdir -p "$RESULTS_DIR/$out"
  { git rev-parse HEAD 2>/dev/null || echo "no_git"; } > "$RESULTS_DIR/$out/run.env"
  $PY main.py --config "configs/$cfg" \
      --override "${OVERRIDES[@]}" gsm8k.output_dir="$RESULTS_DIR/$out"
}

# ---------------------------------------------------------------------------
# 2. Baseline — the reference every budget is measured against.
# ---------------------------------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "baseline" ]; then
  run_one "${CFG_PREFIX}_full_cache.yaml" full_cache "2/5  Full-cache baseline (no compression)"

  banner "3/5  Baseline gate (require accuracy >= $BASELINE_MIN)"
  $PY - "$RESULTS_DIR/full_cache" "$BASELINE_MIN" <<'PYGATE'
import sys
from pathlib import Path
from modules.evaluation.gsm8k_scoring import score_run_dir

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
        f"accuracy {rep.accuracy:.2f} < {floor} (BASELINE_MIN). Llama-3.1-8B-"
        f"Instruct scores ~80-85 on GSM8K with chain-of-thought. A number near 13 means the "
        f"prompt is not eliciting reasoning -- check that {run_dir}/meta.json "
        f"has prompt_style=chain_of_thought and dataset_max_new_tokens=512."
    )
if rep.marker_rate < 90.0:
    fail.append(
        f"marker_rate {rep.marker_rate:.1f}% < 90%. Generations are not "
        f"reaching '#### <number>'. Accuracy is being carried by the "
        f"last-number fallback and is not trustworthy."
    )
if rep.n_generation_failed:
    fail.append(f"{rep.n_generation_failed} generations FAILED outright.")

if fail:
    print("\n  BASELINE GATE FAILED -- not running the budgets:")
    for f in fail:
        print(f"    * {f}")
    sys.exit(1)
print("\n  baseline OK")
PYGATE
fi

# ---------------------------------------------------------------------------
# 3. The two budgets.
# ---------------------------------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "budgets" ]; then
  for arm in $ARMS; do
    run_one "${CFG_PREFIX}_${arm}.yaml" "$arm" "4/5  ${arm} (configs/${CFG_PREFIX}_${arm}.yaml)"
  done
fi

# A budget arm's number is a gated number only if its sidecar says so. Checked
# whenever the arms exist; reported, then turned into the exit status at the end
# so the comparison below still prints.
GATE_STATUS=0
ARM_DIRS=()
for arm in $ARMS; do ARM_DIRS+=("$RESULTS_DIR/$arm"); done
if [ "$STAGE" != "baseline" ] && [ "$STAGE" != "data" ]; then
  banner "Read-gate verdicts"
  $PY scripts/check_read_gate.py "${ARM_DIRS[@]}" || GATE_STATUS=$?
fi

# ---------------------------------------------------------------------------
# 4. Score everything and print the comparison table.
# ---------------------------------------------------------------------------
banner "5/5  Comparison"
RUN_DIRS=("$RESULTS_DIR/full_cache")
for arm in $ARMS; do RUN_DIRS+=("$RESULTS_DIR/$arm"); done
$PY -m modules.evaluation.gsm8k_scoring \
    --runs "${RUN_DIRS[@]}" \
    --out_csv "$RESULTS_DIR/comparison.csv"

echo
echo "Artifacts:"
for d in "${RUN_DIRS[@]}"; do echo "  $d/{predictions.jsonl,meta.json}"; done
echo "  $RESULTS_DIR/comparison.csv"
if [ "$GATE_STATUS" -ne 0 ]; then
  echo "!! at least one budget arm's sidecar does not prove a gated run (see above)."
fi
exit "$GATE_STATUS"
