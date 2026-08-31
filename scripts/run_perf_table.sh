#!/usr/bin/env bash
# ============================================================================
# run_perf_table.sh — reproduce the 7-column decode table
#   shape | batch | TTFT(s) | TPOT_steady(s) | throughput(tok/s) | peak_GB | steadyKV_GB
#
# One method, swept over the shapes and batch sizes you give it, printed in the
# exact format of that table. Everything is configurable: model path, output
# dir, quant_ratio, the (prefill/decode) shapes, and the batch sizes.
#
# It generates a self-contained perf config (no editing of configs/*.yaml),
# runs main.py in perf mode, and prints the table with scripts/print_perf_table.py.
#
# ------------------------------------------------------------------ configure
# Two ways to set anything: an environment variable, or a --flag. Flags win.
#
#   MODEL_PATH      HF id or local path to the model            (required)
#   OUT_DIR         where the .npz + table.txt are written      (default: outputs/perf_table)
#   DATA_SOURCE     prompt text: a corpus name, a path, or a    (default: wikitext-103)
#                   LongBench name. Resolved by perf_runner:
#                     wikitext-103 | pg19        built-in hub corpora
#                     /path/to/file_or_dir       your own text (any extension)
#                     local:relative/path        a repo-relative path
#                     2wikimqa, qasper, ...       a LongBench dataset
#                     longbench:NAME / corpus:NAME  force a loader
#   QUANT_RATIO     two-tier int2 split q in [0,1]              (default: 0.70)
#   QUANT_MODE      tokens | bytes  (see config.py)             (default: tokens)
#   CACHE_BUDGET    fraction of the context kept                (default: 0.50)
#   SHAPES          space list of prefill/decode pairs          (default: "4096/256 2048/512 1048/1048")
#   BATCHES         space list of batch sizes                   (default: "1 32")
#   BACKEND         flash_attn | eager                          (default: flash_attn)
#   WINDOW_SIZE     eviction window (mult. of 4 for q>0)        (default: 8)
#   NUM_SINK        sink tokens kept whole                      (default: 5)
#   LOCAL_WINDOW    local region: int (mult of window) or float (default: 64)
#   RUNS            measurement runs per cell (median reported)  (default: 3)
#   WARMUP          warmup runs per cell                        (default: 1)
#   DTYPE           float16 | bfloat16                          (default: float16)
#   STAT            median | mean  (across runs, for the table) (default: median)
#   COMPILE_EVICT   1 | 0  torch.compile the eviction step      (default: 1)
#   LSE_STRICT      1 | 0  hard-fail on an L-reuse miss         (default: 1)
#                   1 raises AT the miss, naming the layer and the cause.
#                   0 degrades to recompute: a second O(N^2) pass per layer
#                   AND the fp32 block that OOMs 4096/batch-32.
#
# ------------------------------------------------------------------- examples
#   # the shipped table (q=0.70, the two default batch sizes)
#   MODEL_PATH=/models/llama-3.1-8b-instruct scripts/run_perf_table.sh
#
#   # a different operating point, one shape, batch 1 and 16
#   scripts/run_perf_table.sh --model /models/llama --quant-ratio 0.5 \
#       --shapes "8192/512" --batches "1 16"
#
#   # your own corpus, from a file or directory of text
#   scripts/run_perf_table.sh --model /models/llama --data-source /data/my_docs
#   scripts/run_perf_table.sh --model /models/llama --data-source wikitext-103
#
#   # the historic byte-budget behaviour (cache grows with q), for comparison
#   scripts/run_perf_table.sh --model /models/llama --quant-mode bytes --quant-ratio 0.70
# ============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- defaults (env overridable) --------------------------------------------
MODEL_PATH="${MODEL_PATH:-}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/perf_table}"
DATA_SOURCE="${DATA_SOURCE:-wikitext-103}"
QUANT_RATIO="${QUANT_RATIO:-0.70}"
QUANT_MODE="${QUANT_MODE:-tokens}"
CACHE_BUDGET="${CACHE_BUDGET:-0.50}"
SHAPES="${SHAPES:-4096/256 2048/512 1048/1048}"
BATCHES="${BATCHES:-1 32}"
BACKEND="${BACKEND:-flash_attn}"
WINDOW_SIZE="${WINDOW_SIZE:-8}"
NUM_SINK="${NUM_SINK:-5}"
LOCAL_WINDOW="${LOCAL_WINDOW:-64}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-1}"
DTYPE="${DTYPE:-float16}"
STAT="${STAT:-median}"
COMPILE_EVICT="${COMPILE_EVICT:-1}"
LSE_STRICT="${LSE_STRICT:-1}"

# ---- flags (win over env) --------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model-path)   MODEL_PATH="$2"; shift 2;;
    --out|--out-dir)        OUT_DIR="$2"; shift 2;;
    --data-source|--dataset) DATA_SOURCE="$2"; shift 2;;
    --quant-ratio)          QUANT_RATIO="$2"; shift 2;;
    --quant-mode)           QUANT_MODE="$2"; shift 2;;
    --cache-budget|--budget) CACHE_BUDGET="$2"; shift 2;;
    --shapes)               SHAPES="$2"; shift 2;;
    --batches)              BATCHES="$2"; shift 2;;
    --backend)              BACKEND="$2"; shift 2;;
    --window-size)          WINDOW_SIZE="$2"; shift 2;;
    --num-sink)             NUM_SINK="$2"; shift 2;;
    --local-window)         LOCAL_WINDOW="$2"; shift 2;;
    --runs)                 RUNS="$2"; shift 2;;
    --warmup)               WARMUP="$2"; shift 2;;
    --dtype)                DTYPE="$2"; shift 2;;
    --stat)                 STAT="$2"; shift 2;;
    --compile-evict)        COMPILE_EVICT="$2"; shift 2;;
    --lse-strict)           LSE_STRICT="$2"; shift 2;;
    -h|--help)              sed -n '2,57p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; echo "run with --help" >&2; exit 2;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "error: set MODEL_PATH (env) or --model <hf-id-or-path>." >&2
  echo "       run '$0 --help' for all options." >&2
  exit 2
fi

# The eviction step is ~81% of the amortized decode launch budget; the compiled
# body is what collapses it, so default it on. It is KERNEL-OR-ERROR by design:
# if torch.compile cannot lower the eviction on your build it RAISES (that cell
# errors) rather than silently running eager under a compiled label -- because
# the flag exists to MEASURE the compiled path, and eager numbers mislabelled as
# compiled would be worse than an error. The torch<=2.6 Inductor failures
# (`((I)//ws)` guard, `aten.amin` StarDep) are fixed at the source in cache.py,
# so a supported build compiles. If yours still cannot, the error names the op
# and the fix; rerun with --compile-evict 0 for eager (launch-bound) numbers.
export STICKYKV_COMPILE_EVICT="$COMPILE_EVICT"

# L-reuse (a PREFILL optimization) hands the softmax normaliser L from the flash
# forward to the score kernel instead of recomputing it. DEFAULT IS STRICT: a
# miss raises AT the miss, naming the layer and the cause.
#
# This defaulted to 0 with a note blaming the batch>1 varlen path. That
# explanation is wrong for this harness -- perf_runner passes no attention_mask,
# so _update_causal_mask hands flash_attention_2 a None mask and
# _flash_attention_forward calls the patched flash_attn_func at every batch size.
# Nor is the miss "prefill-only cosmetic": the recompute materialises a
# [B, H_q, chunk, S] fp32 block that is 32 GB at 4096/batch-32 -- larger than the
# model weights, and the reason that cell OOMs. Pass --lse-strict 0 only when you
# knowingly want the recompute path.
export STICKYKV_LSE_STRICT="$LSE_STRICT"

case "$BACKEND" in
  flash_attn) ATTN_IMPL="flash_attention_2";;
  eager)      ATTN_IMPL="eager";;
  *) echo "error: --backend must be flash_attn or eager, got '$BACKEND'." >&2; exit 2;;
esac

mkdir -p "$OUT_DIR"
CONFIG_FILE="$OUT_DIR/_perf_table.generated.yaml"

# ---- build the grid: every shape x every batch -----------------------------
GRID_LINES=""
for shape in $SHAPES; do
  prefill="${shape%%/*}"; decode="${shape##*/}"
  if [[ "$prefill" == "$shape" || -z "$decode" ]]; then
    echo "error: --shapes items must be prefill/decode, got '$shape'." >&2; exit 2
  fi
  # gen_len = decode + 1: the runner does gen_len-1 decode steps (the prefill
  # emits the first token), so decode steps == the number you asked for.
  gen=$(( decode + 1 ))
  for b in $BATCHES; do
    GRID_LINES+="    - { prefill_len: ${prefill}, gen_len: ${gen}, batch_size: ${b} }
"
  done
done

# ---- write the self-contained config ---------------------------------------
cat > "$CONFIG_FILE" <<YAML
# GENERATED by scripts/run_perf_table.sh -- safe to delete.
run:
  mode: perf
  seed: 42

model:
  name: ${MODEL_PATH}
  revision: null
  dtype: ${DTYPE}

window:
  window_size: ${WINDOW_SIZE}
  num_sink_tokens: ${NUM_SINK}
  local_window_size: ${LOCAL_WINDOW}

cache:
  quant_ratio: ${QUANT_RATIO}
  quant_budget_mode: ${QUANT_MODE}
  first_eviction_step: 0

perf:
  # Real text, chunked to exactly prefill_len and sampled per row, so every row
  # in a batch is equal-length (the cache's Phase-2 requirement). Resolved by
  # perf_runner.iter_corpus_texts: a hub corpus name, a path, or a LongBench name.
  data_source: "${DATA_SOURCE}"
  budget_basis: context          # budget = cache_budget * prefill_len
  prefill_logits: last_only      # exclude the method-independent [B,L,V] tensor

  configs:
    - name: ours_q${QUANT_RATIO}
      cache_backend: windowed
      cache_package: ${BACKEND}
      attn_implementation: ${ATTN_IMPL}
      cache_budget: ${CACHE_BUDGET}
      quant_ratio: ${QUANT_RATIO}
      quant_budget_mode: ${QUANT_MODE}
      requires_flash_attn: $([[ "$BACKEND" == "flash_attn" ]] && echo true || echo false)

  grid:
${GRID_LINES}
  num_warmup_runs: ${WARMUP}
  num_measurement_runs: ${RUNS}
  allow_shared_gpu: true
  skip_if_oom: true              # an OOM cell prints OOM rather than aborting
  skip_if_flash_attn_unavailable: true
  enable_clock_locking: false

telemetry:
  track_scores: false            # scoring telemetry would distort the timings
  output_dir: ${OUT_DIR}
YAML

# ---- environment snapshot, for reproducibility -----------------------------
{
  echo "commit: $(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo no_git)"
  echo "model: $MODEL_PATH"
  echo "data_source: $DATA_SOURCE"
  echo "quant_ratio: $QUANT_RATIO  quant_budget_mode: $QUANT_MODE  cache_budget: $CACHE_BUDGET"
  echo "shapes: $SHAPES  batches: $BATCHES  backend: $BACKEND"
  echo "STICKYKV_COMPILE_EVICT: $STICKYKV_COMPILE_EVICT"
  echo "STICKYKV_LSE_STRICT: $STICKYKV_LSE_STRICT"
  echo "window_size: $WINDOW_SIZE  num_sink: $NUM_SINK  local_window: $LOCAL_WINDOW"
  echo "runs: $RUNS  warmup: $WARMUP  dtype: $DTYPE"
} > "$OUT_DIR/run_perf_table.env"

echo "=== run_perf_table: q=$QUANT_RATIO ($QUANT_MODE), budget=$CACHE_BUDGET, backend=$BACKEND ==="
echo "data: $DATA_SOURCE"
echo "shapes: $SHAPES   batches: $BATCHES"
echo "config: $CONFIG_FILE"
echo ""

python "$PROJECT_ROOT/main.py" --config "$CONFIG_FILE"

echo ""
echo "=== decode table ==="
python "$PROJECT_ROOT/scripts/print_perf_table.py" \
    --npz-dir "$OUT_DIR" \
    --stat "$STAT" \
    --out "$OUT_DIR/table.txt"
