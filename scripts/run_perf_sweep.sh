#!/usr/bin/env bash
# Sweep run_perf_table.sh over a knob grid, one OUT_DIR per config.
#
# WHY THIS EXISTS, AND WHY LOOPING BY HAND IS WRONG
# --------------------------------------------------
# perf_runner names its output `perf_prefill{P}_gen{G}_bs{B}.npz` — the SHAPE and
# the BATCH, and nothing about the config. So two configs written into one
# OUT_DIR silently overwrite each other, and `print_perf_table.py` then prints
# the survivor under whichever name it finds. A sweep done as a bare shell loop
# over `--quant-ratio ...` produces one table that looks complete and describes
# one config. This script gives every config its own directory, so the numbers
# keep their provenance.
#
# It is also RESUMABLE, which a 144-cell sweep needs: a config whose table.txt
# already exists is skipped. Delete that file to force a re-run.
#
#   CUDA_VISIBLE_DEVICES=0 OUT_DIR=outputs/sweep_v7 bash scripts/run_perf_sweep.sh \
#     --model /path/to/llama-3.1-8b-instruct \
#     --shapes "4096/256 1048/1048 2048/512" \
#     --batches "1 32" \
#     --data-source wikitext-103 \
#     --throughput all --cooldown 3 --clock-lock true
#
# Knob lists (space-separated; defaults are the grid this was written for):
#
#   QUANT_RATIOS   default "0.70 0.30"
#   GATE_RATIOS    default "0.10 0.25"
#   LOCAL_WINDOWS  default "64 128"
#   WINDOW_SIZES   default "8 16 32"
#   CACHE_BUDGET   default 0.20        (single value — the budget the claim is at)
#   NUM_SINK       default 5
#
# --dry-run validates the grid and lists the configs without running anything.
# Use it before committing a GPU to 24 configs x 6 cells.
#
# 2 x 2 x 2 x 3 = 24 configs. Every other flag is passed through untouched.
#
# CONSTRAINT THE GRID MUST SATISFY: local_window_size must be a multiple of
# window_size, and window_size must be divisible by 4 whenever quant_ratio > 0
# (int2 crumb packing). This script checks both up front rather than letting the
# 19th config fail an hour in.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/perf_sweep}"
QUANT_RATIOS="${QUANT_RATIOS:-0.70 0.30}"
GATE_RATIOS="${GATE_RATIOS:-0.10 0.25}"
LOCAL_WINDOWS="${LOCAL_WINDOWS:-64 128}"
WINDOW_SIZES="${WINDOW_SIZES:-8 16 32}"
CACHE_BUDGET="${CACHE_BUDGET:-0.20}"
NUM_SINK="${NUM_SINK:-5}"
DRY_RUN="${DRY_RUN:-0}"

PASS_THROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out|--out-dir)   OUT_DIR="$2"; shift 2;;
    --quant-ratios)    QUANT_RATIOS="$2"; shift 2;;
    --gate-ratios)     GATE_RATIOS="$2"; shift 2;;
    --local-windows)   LOCAL_WINDOWS="$2"; shift 2;;
    --window-sizes)    WINDOW_SIZES="$2"; shift 2;;
    --cache-budget|--budget) CACHE_BUDGET="$2"; shift 2;;
    --num-sink)        NUM_SINK="$2"; shift 2;;
    --dry-run)         DRY_RUN=1; shift;;
    -h|--help)         sed -n '2,40p' "$0"; exit 0;;
    # Everything else (--model, --shapes, --batches, --data-source,
    # --throughput, --cooldown, --clock-lock, ...) goes to run_perf_table.sh.
    *)                 PASS_THROUGH+=("$1"); shift;;
  esac
done

# --- validate the grid before spending a GPU-hour on it ---------------------
for ws in $WINDOW_SIZES; do
  for q in $QUANT_RATIOS; do
    if [[ "$q" != "0" && "$q" != "0.0" && $(( ws % 4 )) -ne 0 ]]; then
      echo "grid error: window_size=$ws is not divisible by 4, and quant_ratio=$q > 0" >&2
      echo "  int2 packs 4 codes per byte along the window axis (config.py)." >&2
      exit 2
    fi
  done
  for lw in $LOCAL_WINDOWS; do
    if [[ $(( lw % ws )) -ne 0 ]]; then
      echo "grid error: local_window_size=$lw is not a multiple of window_size=$ws" >&2
      exit 2
    fi
  done
done

mkdir -p "$OUT_DIR"
n_total=0
for q in $QUANT_RATIOS; do for gr in $GATE_RATIOS; do
for lw in $LOCAL_WINDOWS; do for ws in $WINDOW_SIZES; do
  n_total=$(( n_total + 1 )); done; done; done; done

echo "=== perf sweep: $n_total configs ==="
echo "budget=$CACHE_BUDGET  sink=$NUM_SINK"
echo "q=[$QUANT_RATIOS]  gate=[$GATE_RATIOS]  local=[$LOCAL_WINDOWS]  ws=[$WINDOW_SIZES]"
echo "out=$OUT_DIR"
echo ""

i=0
for q in $QUANT_RATIOS; do
 for gr in $GATE_RATIOS; do
  for lw in $LOCAL_WINDOWS; do
   for ws in $WINDOW_SIZES; do
    i=$(( i + 1 ))
    slug="b${CACHE_BUDGET}_q${q}_g${gr}_l${lw}_w${ws}"
    cell="$OUT_DIR/$slug"
    if [[ -f "$cell/table.txt" ]]; then
      echo "[$i/$n_total] $slug — already done, skipping (delete $cell/table.txt to redo)"
      continue
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[$i/$n_total] $slug -> $cell"
      continue
    fi
    echo "[$i/$n_total] $slug"
    OUT_DIR="$cell" bash "$PROJECT_ROOT/scripts/run_perf_table.sh" \
      --cache-budget "$CACHE_BUDGET" \
      --quant-ratio "$q" \
      --gate-ratio "$gr" \
      --local-window "$lw" \
      --window-size "$ws" \
      --num-sink "$NUM_SINK" \
      "${PASS_THROUGH[@]}"
   done
  done
 done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo ""
  echo "dry run: grid is valid, $n_total configs would run, nothing executed."
  exit 0
fi

echo ""
echo "=== sweep complete: $OUT_DIR ==="
echo "Per-config tables:   $OUT_DIR/<config>/table.txt"
echo "Per-config settings: $OUT_DIR/<config>/run_perf_table.env"
echo ""
echo "Combined view (TPOT_steady per config x shape x batch):"
python "$PROJECT_ROOT/scripts/print_perf_sweep.py" --sweep-dir "$OUT_DIR" \
    || echo "  (print_perf_sweep.py failed; the per-config tables above are intact)"
