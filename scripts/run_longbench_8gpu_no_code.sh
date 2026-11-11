#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# LongBench (ours, flash-attn backend) — the non-code English datasets, MINUS
# passage_retrieval_en and passage_count (excluded on request) — sharded
# across 8 GPUs, one process per GPU.
#
# Ported from 27_08/RecoverKv/scripts/run_longbench_8gpu_no_code.sh onto this
# branch (int2_clustered_rep). See "DIFFERENCES FROM THE 27_08 SCRIPT" below:
# the base config here pins quant_ratio=0.0, so the q=0.70 operating point comes
# from this script's override, not from the YAML.
#
# Base config: configs/longbench_ours_flash_attn.yaml
#   cache_budget=0.20  window_size=8  num_sink_tokens=5  local_window_size=128
#   quant_budget_mode=bytes  (the mode a memory-vs-quality claim is stated in)
#   quant_ratio=0.0 in the YAML -> overridden to $QUANT_RATIO (default 0.70)
#   window_size=8 is divisible by 4, which q>0 requires.
#
# model.name is overridden to the local checkpoint via --override (the config's
# own model.name is a different machine's path).
#
# ---------------------------------------------------------------- environment
# Needs a GPU node and the `sticky_env` conda env:
#     conda activate sticky_env
# The login node has no CUDA and no flash_attn. That does NOT fail loudly
# everywhere in this repo -- the perf runner, for one, logs "flash-attn not
# available", skips every cell and still exits 0. The preflight below refuses
# to start rather than let that happen here.
#
# ------------------------------------------------------------------ balancing
# Sharding is WALL-CLOCK BALANCED (LPT bin-packing) from the per-dataset times
# measured on the PRIOR 27_08 RUN (run_finished_utc - run_started_utc in each
# <dataset>.meta.json under outputs/longbench/ours_flash_attn_8gpu_no_code/):
#
#   gov_report 10500s  multi_news 6751s  qmsum 3313s  samsum 2831s
#   narrativeqa 1924s  trec 1448s  triviaqa 1259s  musique 843s
#   hotpotqa 716s  qasper 682s  multifieldqa_en 566s  2wikimqa 520s
#
# THOSE TIMINGS PREDATE THE READ GATE on this branch. The packing is inherited,
# not re-measured; if the gate shifts per-dataset decode cost the makespan will
# drift. Re-pack from this run's own meta.json times if it matters.
#
# LongBenchConfig has no shard/num_shards knob (unlike GSM8KConfig) -- a dataset
# cannot be split across GPUs, only assigned whole. gov_report alone (10500s)
# exceeds the 7-way mean of the other 11 (~2979s/GPU), so it is the hard floor
# on wall time no matter how the rest are packed: this assignment is
# makespan-optimal for a whole-dataset bin packing (max bin = gov_report's own
# 10500s, ~2h55m), NOT even-per-GPU like a round robin.
#
# All 8 processes write into the SAME output_dir, which is safe: each dataset
# writes its own <dataset>.jsonl / <dataset>.meta.json (longbench_runner.py), so
# there is no cross-process file collision.
#
# ------------------------------------------ DIFFERENCES FROM THE 27_08 SCRIPT
# 1. quant_ratio: the 27_08 header claimed the base config carried 0.70. This
#    repo's configs/longbench_ours_flash_attn.yaml pins 0.0 (pure fp16), so the
#    override is what selects the two-tier point. Run with QUANT_RATIO=0.0 for
#    the fp16 baseline.
# 2. quant_budget_mode is passed explicitly. It is plumbed on the LongBench path
#    here (longbench_runner.py routes it through quant_budget_mode_kwargs), but
#    it was silently inert once before (f71fec0); pinning it in the command line
#    keeps it in the sidecar.
# 3. THE READ GATE IS NOT CONFIGURABLE FROM HERE. quant_gate_ratio /
#    quant_gate_margin / quant_gate_max_windows live on WindowedCacheConfig
#    (modules/windowed_cache/config.py:281) but are NOT in utils.config's
#    CacheConfig and are NOT forwarded by longbench_runner._setup_cache, so
#    every run takes the dataclass default (ratio 0.25). Worse, _dict_to_config
#    IGNORES UNKNOWN KEYS: `--override cache.quant_gate_ratio=0.5` is accepted
#    silently and does nothing. Do not add it here expecting it to bite --
#    plumb it through the runner first.
#
# ------------------------------------------------------------------ configure
#   MODEL_PATH   local checkpoint          (default: llama-3.1-8b-instruct)
#   CONFIG       base YAML                 (default: longbench_ours_flash_attn)
#   OUT_DIR      predictions dir           (default: outputs/longbench/...)
#   QUANT_RATIO  two-tier int2 split q     (default: 0.70)
#   QUANT_MODE   bytes | tokens            (default: bytes)
#   CACHE_BUDGET fraction of context kept  (default: 0.20, from the YAML)
#   LOCAL_WINDOW local region, int tokens    (default: 128)
#
# Run manually on an 8-GPU node:
#   conda activate sticky_env && scripts/run_longbench_8gpu_no_code.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
CONFIG="${CONFIG:-configs/longbench_ours_flash_attn.yaml}"
OUT_DIR="${OUT_DIR:-outputs/longbench/ours_flash_attn_8gpu_no_code}"
QUANT_RATIO="${QUANT_RATIO:-0.70}"
QUANT_MODE="${QUANT_MODE:-bytes}"
CACHE_BUDGET="${CACHE_BUDGET:-0.20}"
# int = token count, and must be a multiple of window_size (8). A float would be
# read as a RATIO of the context instead -- 128 and 128.0 are not the same knob.
LOCAL_WINDOW="${LOCAL_WINDOW:-128}"

# ---- preflight: fail here, not 8 processes deep -----------------------------
# Each of these has already cost a wasted run: a missing checkpoint, a CPU-only
# login node, or a torch that cannot see the GPUs it was told to shard across.
[[ -f "$CONFIG" ]]      || { echo "error: config not found: $CONFIG" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]]  || { echo "error: model not found: $MODEL_PATH" >&2; exit 2; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: no nvidia-smi -- this is a login node, not a GPU node." >&2
    echo "       submit to a GPU node and 'conda activate sticky_env' first." >&2
    exit 2
fi

python - <<'PY' || exit 2
import sys
try:
    import torch
except ImportError:
    sys.exit("error: torch not importable -- wrong env? 'conda activate sticky_env'")
if not torch.cuda.is_available():
    sys.exit("error: torch.cuda.is_available() is False -- wrong env or no GPU allocation.")
n = torch.cuda.device_count()
if n < 8:
    sys.exit(f"error: this script shards across 8 GPUs, torch sees {n}. "
             "Re-pack the GPU arrays below for the node you have.")
try:
    import flash_attn  # noqa: F401
except ImportError:
    sys.exit("error: flash_attn not importable, but the config asks for "
             "flash_attention_2. Runs would be skipped or silently wrong.")
print("preflight OK: 8 GPUs, torch + flash_attn present")
PY

mkdir -p "$OUT_DIR"

# Provenance sidecar, same idiom as the other run_* scripts in this repo.
{
    git rev-parse HEAD 2>/dev/null || echo "no_git"
    echo "config=$CONFIG"
    echo "model=$MODEL_PATH"
    echo "quant_ratio=$QUANT_RATIO  quant_budget_mode=$QUANT_MODE  cache_budget=$CACHE_BUDGET  local_window_size=$LOCAL_WINDOW"
    echo "quant_gate_ratio=<WindowedCacheConfig default; not settable from here>"
} > "$OUT_DIR/run.env"

echo "=== LongBench 8-GPU (no code, no passage_*) ==="
echo "config:  $CONFIG"
echo "model:   $MODEL_PATH"
echo "out:     $OUT_DIR"
echo "q=$QUANT_RATIO ($QUANT_MODE)  budget=$CACHE_BUDGET  local=$LOCAL_WINDOW"
echo

# Wall-clock-balanced (LPT) assignment of the 12 datasets (14 minus
# passage_retrieval_en and passage_count) across 8 GPUs -- see header for the
# measured per-dataset seconds this was packed from.
GPU0=(gov_report)                         # 10500s (solo -- sets the floor)
GPU1=(multi_news)                         # 6751s  (solo)
GPU2=(qmsum)                              # 3313s  (solo)
GPU3=(samsum)                             # 2831s  (solo)
GPU4=(narrativeqa)                        # 1924s  (solo)
GPU5=(trec multifieldqa_en)               # 1448 + 566  = 2014s
GPU6=(triviaqa qasper)                    # 1259 + 682  = 1941s
GPU7=(musique hotpotqa 2wikimqa)          #  843 + 716 + 520 = 2079s

run_shard() {
    local gpu="$1"; shift
    local datasets=("$@")
    local joined
    joined=$(IFS=,; echo "${datasets[*]}")
    echo "=== GPU $gpu: ${datasets[*]} ==="
    CUDA_VISIBLE_DEVICES="$gpu" python "$PROJECT_ROOT/main.py" \
        --config "$CONFIG" \
        --override \
            model.name="$MODEL_PATH" \
            cache.cache_budget="$CACHE_BUDGET" \
            cache.quant_ratio="$QUANT_RATIO" \
            cache.quant_budget_mode="$QUANT_MODE" \
            cache.local_window_size="$LOCAL_WINDOW" \
            longbench.datasets="[${joined}]" \
            longbench.output_dir="$OUT_DIR"
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
    arr_name="GPU${gpu}[@]"
    run_shard "$gpu" "${!arr_name}" > "$OUT_DIR/gpu${gpu}.log" 2>&1 &
    pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "GPU $i OK -- $OUT_DIR/gpu${i}.log"
    else
        echo "GPU $i FAILED -- see $OUT_DIR/gpu${i}.log" >&2
        status=1
    fi
done

echo "############################################################"
echo "# LongBench predictions written to: $OUT_DIR"
echo "# Score with: scripts/score_longbench.sh"
echo "#"
echo "# Sanity-check before trusting the numbers: 12 .jsonl files, and each"
echo "# <dataset>.meta.json's operating point should read q=$QUANT_RATIO."
echo "############################################################"
exit "$status"
