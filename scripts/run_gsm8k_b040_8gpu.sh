#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# GSM8K (1319 test problems, zero-shot CoT, <=512 new tokens) on every GPU of
# the node, one contiguous shard per process, merged and scored at the end.
#
#   operating point  cache_budget 0.40, q 0.70 (bytes), gate 0.25, window 8,
#                    local 128, sink 5, first_eviction_step 0, greedy decoding
#   repo             THIS checkout, as it is. Nothing here pulls or checks out.
#
# ------------------------------------------------------------ why 40 %, not 20
# GSM8K prompts are tiny (Llama-3.1 tokens: min 124, median 157, max 287), and
# the budget is taken of prompt + 512 generated. At 20 % / local 128 the budget
# is ~133 tokens = sink 5 + local 128 exactly: 903/1319 problems (68 %) get NO
# evictable windows -- no int2 tier, no gate -- and 533 exceed the budget, so
# the method degenerates to a sink + sliding window. At 40 % / local 128 every
# problem resolves to ~5 fp16 + 42-51 int2 windows, none over budget
# (checked with WindowedCacheConfig.resolve over all 1319 prompts).
#
# ------------------------------------------------------------------ scheduling
# SHARDS contiguous shards (default: one per GPU -> 165 problems each at 8).
# The loader splits with ceil(1319 / SHARDS) per shard (gsm8k_loader.py:156).
# Time is dominated by generation: ~290 tokens/problem at ~55 ms/token (B=1)
# ~= 16 s/problem -> ~45-60 min on 8 GPUs. GSM8K has no batch>1 path (ragged
# prompts; WindowedCache supports equal-length batches only), hence shards.
#
# ------------------------------------------------------------------- eviction
# STICKYKV_EVICT_CONTROL_ARM is used if this checkout has it; at bda91b4 /
# 2f06bc4 it does not, so the eviction runs compiled.
#
# ------------------------------------------------------------------ resuming
# A shard is DONE only if meta.json exists AND predictions.jsonl has exactly
# that shard's problem count. Anything else is deleted and re-run from scratch.
# Re-run the script to redo only unfinished shards. A shard refuses to start if
# HEAD moved since the run began. Do not start two copies at once.
#
# Outputs:
#   $OUT_ROOT/shard_<k>/{predictions.jsonl,meta.json}, $OUT_ROOT/shard_<k>.log
#   $OUT_ROOT/merged/predictions.jsonl  -- all 1319, in dataset order
#   $OUT_ROOT/summary.txt               -- accuracy + per-shard gate/retention
#
# Run on the GPU node (NOT the node running the RULER window sweep):
#   cd ~/kv_cache/Cluster_QEvict/RecoverKv && conda activate sticky_env
#   nohup scripts/run_gsm8k_b040_8gpu.sh > ~/gsm8k_b040.log 2>&1 &
#   tail -f ~/gsm8k_b040.log
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
# This repo has no data/gsm8k_cot; the built dataset (manifest + jsonl) lives here.
GSM8K_DATA="${GSM8K_DATA:-/home/ee/phd/eez228470/kv_cache/Mistral_Quantization_INT2/RecoverKv/data/gsm8k_cot}"
CONFIG="configs/gsm8k_budget20.yaml"      # base only; every cache knob is overridden below

# Pinned -- not env-overridable.
BUDGET=0.40; QUANT_RATIO=0.70; QUANT_MODE=bytes; GATE_RATIO=0.25
WINDOW=8; LOCAL=128; SINK=5

COMMIT="$(git rev-parse --short HEAD)"; START_SHA="$(git rev-parse HEAD)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/gsm8k_b040_q070_w8_l128_${COMMIT}}"

# ---- preflight -------------------------------------------------------------------
die() { echo "error: $*" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
[[ -f "$CONFIG" ]]     || die "missing $CONFIG"
[[ -s "$GSM8K_DATA/gsm8k_cot.jsonl" && -f "$GSM8K_DATA/manifest.json" ]] \
    || die "GSM8K dataset (gsm8k_cot.jsonl + manifest.json) not found in $GSM8K_DATA"
(( LOCAL % WINDOW == 0 )) || die "local $LOCAL is not a multiple of window $WINDOW"
command -v nvidia-smi >/dev/null || die "no nvidia-smi -- login node? get a GPU node."
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"   # PBS UUIDs, used as given
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
(( ${#GPUS[@]} > 0 )) || die "no GPUs visible"
SHARDS="${SHARDS:-${#GPUS[@]}}"
python - <<'PY' || exit 2
import sys
try:
    import torch
except ImportError:
    sys.exit("error: torch not importable -- 'conda activate sticky_env'")
if not torch.cuda.is_available():
    sys.exit("error: torch.cuda.is_available() is False -- wrong env or no GPU allocation.")
try:
    import flash_attn  # noqa: F401
except ImportError:
    sys.exit("error: flash_attn not importable; the config needs flash_attention_2.")
PY

TOTAL=$(grep -c . "$GSM8K_DATA/gsm8k_cot.jsonl")
PER=$(( (TOTAL + SHARDS - 1) / SHARDS ))          # the loader's own split
shard_n() { local s=$(( $1 * PER )) e=$(( ($1 + 1) * PER )); (( e > TOTAL )) && e=$TOTAL; echo $(( e - s )); }

ARM=0
grep -q '_EVICT_CONTROL_ARM_ENV = "STICKYKV_EVICT_CONTROL_ARM"' modules/windowed_cache/cache.py && ARM=1
EVICT_DESC=$([[ $ARM == 1 ]] && echo "control-arm-eager" || echo "compiled (no control arm at $COMMIT)")

mkdir -p "$OUT_ROOT"
QUEUE_POS="$OUT_ROOT/.queue_pos"; LOCK="$OUT_ROOT/.queue_lock"; FAILED="$OUT_ROOT/failed_jobs.txt"
echo 0 > "$QUEUE_POS"; : > "$FAILED"
{
    echo "commit=$START_SHA  started=$(date -Is)  host=$(hostname)  gpus=${GPUS[*]}"
    echo "evict=$EVICT_DESC  model=$MODEL_PATH"
    echo "gsm8k_data=$GSM8K_DATA  problems=$TOTAL  shards=$SHARDS x <=$PER"
    echo "budget=$BUDGET q=$QUANT_RATIO mode=$QUANT_MODE gate=$GATE_RATIO window=$WINDOW local=$LOCAL sink=$SINK"
} > "$OUT_ROOT/run.env"

is_done() {
    local d="$OUT_ROOT/shard_$1"
    [[ -f "$d/meta.json" && -f "$d/predictions.jsonl" ]] || return 1
    [[ "$(wc -l < "$d/predictions.jsonl")" -eq "$(shard_n "$1")" ]]
}

next_job() {
    local i
    {
        flock 9
        i=$(<"$QUEUE_POS")
        if (( i < SHARDS )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
    } 9>"$LOCK"
}

run_job() {    # $1 gpu, $2 shard
    local gpu="$1" k="$2" d="$OUT_ROOT/shard_$2"
    if is_done "$k"; then echo "[$(date +%T)] gpu=$gpu SKIP   shard $k (complete)"; return 0; fi
    if [[ "$(git rev-parse HEAD)" != "$START_SHA" ]]; then
        echo "[$(date +%T)] gpu=$gpu REFUSE shard $k -- HEAD moved off $COMMIT" >&2
        echo "shard $k rc=HEAD-moved" >> "$FAILED"; return 1
    fi
    rm -rf "$d"
    echo "[$(date +%T)] gpu=$gpu START  shard $k/$SHARDS ($(shard_n "$k") problems)"
    local t0=$SECONDS rc
    (
        if [[ $ARM == 1 ]]; then export STICKYKV_EVICT_CONTROL_ARM=1; else unset STICKYKV_EVICT_CONTROL_ARM; fi
        CUDA_VISIBLE_DEVICES="$gpu" exec python main.py --config "$CONFIG" --override \
            model.name="$MODEL_PATH" \
            cache.cache_budget="$BUDGET" \
            cache.quant_ratio="$QUANT_RATIO" \
            cache.quant_budget_mode="$QUANT_MODE" \
            cache.quant_gate_ratio="$GATE_RATIO" \
            cache.window_size="$WINDOW"       window.window_size="$WINDOW" \
            cache.local_window_size="$LOCAL"  window.local_window_size="$LOCAL" \
            cache.num_sink_tokens="$SINK"     window.num_sink_tokens="$SINK" \
            cache.first_eviction_step=0 \
            gsm8k.data_dir="$GSM8K_DATA" gsm8k.num_samples=max \
            gsm8k.shard="$k" gsm8k.num_shards="$SHARDS" \
            gsm8k.output_dir="$d" gsm8k.results_dir="$OUT_ROOT"
    ) > "$OUT_ROOT/shard_$k.log" 2>&1
    rc=$?
    if (( rc == 0 )) && is_done "$k"; then
        echo "[$(date +%T)] gpu=$gpu DONE   shard $k ($(( (SECONDS - t0) / 60 )) min)"
    else
        local why
        why=$(grep -m1 -oE "ran EAGER|OutOfMemoryError|CUDA error[^.]*|[A-Za-z]+Error: [^.]{0,80}" "$OUT_ROOT/shard_$k.log" | head -1)
        echo "[$(date +%T)] gpu=$gpu FAIL   shard $k rc=$rc ${why:+($why)} -- $OUT_ROOT/shard_$k.log" >&2
        echo "shard $k rc=$rc ${why:-see log} $OUT_ROOT/shard_$k.log" >> "$FAILED"
    fi
}

worker() {
    local gpu="$1" i
    while i=$(next_job) && [[ -n "$i" ]]; do run_job "$gpu" "$i"; done
}

echo "=== GSM8K: $TOTAL problems in $SHARDS shards on ${#GPUS[@]} GPU(s) ==="
echo "  commit $COMMIT on $(hostname) | eviction: $EVICT_DESC"
echo "  budget=$BUDGET q=$QUANT_RATIO ($QUANT_MODE) gate=$GATE_RATIO window=$WINDOW local=$LOCAL sink=$SINK"
echo "  out: $OUT_ROOT"
echo

pids=()
for g in "${GPUS[@]}"; do
    worker "$g" 2>&1 | tee -a "$OUT_ROOT/worker_${g}.log" &
    pids+=($!)
done
wait "${pids[@]}"

# ---- merge + score -----------------------------------------------------------------
if [[ -s "$FAILED" ]]; then
    echo; echo "!! failed shards (re-run the script to redo just these) -- not merging:"; cat "$FAILED"; exit 1
fi
for (( k = 0; k < SHARDS; k++ )); do is_done "$k" || { echo "!! shard $k incomplete -- not merging" >&2; exit 1; }; done

echo; echo "=== merging $SHARDS shards ==="
OUT_ROOT="$OUT_ROOT" SHARDS="$SHARDS" PER="$PER" TOTAL="$TOTAL" python - <<'PY'
import json, os
root, S, per, total = os.environ["OUT_ROOT"], int(os.environ["SHARDS"]), int(os.environ["PER"]), int(os.environ["TOTAL"])
os.makedirs(f"{root}/merged", exist_ok=True)
n = 0
with open(f"{root}/merged/predictions.jsonl", "w") as out:
    for k in range(S):
        for line in open(f"{root}/shard_{k}/predictions.jsonl"):
            r = json.loads(line)
            r["shard"], r["shard_id"], r["id"] = k, r["id"], k * per + r["id"]   # dataset-order id
            out.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
assert n == total, f"merged {n} records, expected {total}"
print(f"merged {n} predictions -> {root}/merged/predictions.jsonl")
PY
[[ $? -eq 0 ]] || exit 1

python -m modules.evaluation.gsm8k_scoring --predictions "$OUT_ROOT/merged/predictions.jsonl" \
    --out_csv "$OUT_ROOT/merged/scores.csv" > "$OUT_ROOT/merged/scoring.log" 2>&1 \
    || echo "scoring FAILED -- $OUT_ROOT/merged/scoring.log" >&2

OUT_ROOT="$OUT_ROOT" SHARDS="$SHARDS" COMMIT="$COMMIT" EVICT_DESC="$EVICT_DESC" python - <<'PY' | tee "$OUT_ROOT/summary.txt"
import json, os
E = os.environ; root, S = E["OUT_ROOT"], int(E["SHARDS"])
print(f"GSM8K @ {E['COMMIT']} | eviction: {E['EVICT_DESC']}")
print("budget 0.40  q 0.70 (bytes)  gate 0.25  window 8  local 128  sink 5\n")
print(f"{'shard':>5} {'n':>4} {'read_gate':>10} {'retained':>9} {'seq':>6} {'no-evict%':>9} {'oom':>4}")
for k in range(S):
    m = json.load(open(f"{root}/shard_{k}/meta.json"))
    g = (m.get("read_gate") or {}).get("verdict", "MISSING")
    print(f"{k:5d} {m.get('num_examples', 0):4d} {g:>10} {m.get('mean_retained_tokens', float('nan')):9.1f} "
          f"{m.get('mean_sequence_tokens', float('nan')):6.1f} {(m.get('pct_examples_no_eviction') or 0):8.1f}% {m.get('num_oom', 0):4d}")
    # pct_examples_no_eviction is already a percentage (gsm8k_runner.py:456)
print("\n--- scorer (merged, all problems) ---")
print(open(f"{root}/merged/scoring.log").read().strip())
PY
echo; echo "done: $OUT_ROOT/summary.txt"
