#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# GSM8K (1319 test problems, zero-shot CoT, <=512 new tokens) at several cache
# budgets, one queue over every GPU of the node, merged and scored per budget.
#
#   budgets / local  CONFIGS="0.80:128 0.60:128 0.20:32 0.10:16"  (budget:local)
#   fixed            q 0.70 (bytes), gate 0.25, window 8, sink 5,
#                    first_eviction_step 0, greedy decoding
#   repo             THIS checkout, as it is. Nothing here pulls or checks out.
#
# ------------------------------------------------------------ local per budget
# GSM8K prompts are tiny (Llama-3.1 tokens: min 124, median 157, max 287) and
# the budget is taken of prompt + 512 generated. Checked with
# WindowedCacheConfig.resolve over all 1319 prompts (median prompt shown):
#   0.80 / 128   535 tok   15 fp16 + 134 int2 windows   local = 24 % of budget
#   0.60 / 128   401 tok   10 fp16 +  89 int2           local = 32 %
#   0.20 /  32   133 tok    3 fp16 +  34 int2           local = 24 %
#   0.10 /  16    66 tok    1 fp16 +  17 int2           local = 24 %
# Every problem at every point has both tiers and none exceeds its budget.
# (0.20 / 128 would leave 68 % of problems with no int2 tier -- the method
# degenerates to sink + sliding window -- which is why local shrinks.)
# 0.40 / 128 already ran: outputs/gsm8k_b040_q070_w8_l128_<commit>; if present
# it is printed in the summary as a reference row.
#
# ------------------------------------------------------------------ scheduling
# Each budget is split into SHARDS contiguous shards (default: one per GPU, so
# 165 problems each at 8); every (budget, shard) is one job on one GPU, pulled
# from one queue, budget by budget. At 40 % a shard took 47-50 min, so expect
# ~50 min per budget on 8 GPUs, ~3.3-3.5 h for four. GSM8K has no batch>1 path.
#
# ------------------------------------------------------------------- eviction
# STICKYKV_EVICT_CONTROL_ARM is used if this checkout has it; at 2f06bc4 it
# does not, so the eviction runs compiled (as in the 40 % run).
#
# ------------------------------------------------------------------ resuming
# A shard is DONE only if meta.json exists AND predictions.jsonl has exactly
# that shard's problem count; anything else is deleted and re-run. A budget is
# merged and scored only when all its shards are done. Re-run the script to
# redo only what is unfinished. A job refuses to start if HEAD moved. Do not
# start two copies at once.
#
# Outputs, per budget:
#   outputs/gsm8k_b<BB>_q070_w8_l<L>_<commit>/shard_<k>/{predictions.jsonl,meta.json}
#   .../merged/predictions.jsonl (all 1319, dataset order), merged/scoring.log
# and outputs/gsm8k_budgets_<commit>/summary.txt -- every budget side by side.
#
# Run on an 8-GPU node (not one already running another sweep):
#   cd ~/kv_cache/Cluster_QEvict/RecoverKv && conda activate sticky_env
#   nohup scripts/run_gsm8k_budgets_8gpu.sh > ~/gsm8k_budgets.log 2>&1 &
#   tail -f ~/gsm8k_budgets.log
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
GSM8K_DATA="${GSM8K_DATA:-/home/ee/phd/eez228470/kv_cache/Mistral_Quantization_INT2/RecoverKv/data/gsm8k_cot}"
CONFIG="configs/gsm8k_budget20.yaml"      # base only; every cache knob is overridden below
read -r -a CONFIGS <<< "${CONFIGS:-0.80:128 0.60:128 0.20:32 0.10:16}"

# Pinned -- not env-overridable.
QUANT_RATIO=0.70; QUANT_MODE=bytes; GATE_RATIO=0.25; WINDOW=8; SINK=5

COMMIT="$(git rev-parse --short HEAD)"; START_SHA="$(git rev-parse HEAD)"
SWEEP_DIR="${SWEEP_DIR:-$PROJECT_ROOT/outputs/gsm8k_budgets_${COMMIT}}"
tag()     { local b="${1%%:*}"; printf 'b%03d' "$(awk -v b="$b" 'BEGIN{printf "%d", b*100 + 0.5}')"; }
cfg_dir() { printf '%s/outputs/gsm8k_%s_q070_w%s_l%s_%s' "$PROJECT_ROOT" "$(tag "$1")" "$WINDOW" "${1##*:}" "$COMMIT"; }

# ---- preflight -------------------------------------------------------------------
die() { echo "error: $*" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
[[ -f "$CONFIG" ]]     || die "missing $CONFIG"
[[ -s "$GSM8K_DATA/gsm8k_cot.jsonl" && -f "$GSM8K_DATA/manifest.json" ]] \
    || die "GSM8K dataset (gsm8k_cot.jsonl + manifest.json) not found in $GSM8K_DATA"
(( ${#CONFIGS[@]} > 0 )) || die "CONFIGS is empty"
for c in "${CONFIGS[@]}"; do
    [[ "$c" =~ ^0?\.[0-9]+:[0-9]+$ || "$c" =~ ^1(\.0+)?:[0-9]+$ ]] || die "config '$c' is not budget:local (e.g. 0.20:32)"
    l="${c##*:}"
    (( l > 0 && l % WINDOW == 0 )) || die "config '$c': local $l must be a positive multiple of window $WINDOW"
done
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
PER=$(( (TOTAL + SHARDS - 1) / SHARDS ))          # the loader's own split (gsm8k_loader.py:156)
shard_n() { local s=$(( $1 * PER )) e=$(( ($1 + 1) * PER )); (( e > TOTAL )) && e=$TOTAL; echo $(( e - s )); }

ARM=0
grep -q '_EVICT_CONTROL_ARM_ENV = "STICKYKV_EVICT_CONTROL_ARM"' modules/windowed_cache/cache.py && ARM=1
EVICT_DESC=$([[ $ARM == 1 ]] && echo "control-arm-eager" || echo "compiled (no control arm at $COMMIT)")

mkdir -p "$SWEEP_DIR"
for c in "${CONFIGS[@]}"; do mkdir -p "$(cfg_dir "$c")"; done
QUEUE_POS="$SWEEP_DIR/.queue_pos"; LOCK="$SWEEP_DIR/.queue_lock"; FAILED="$SWEEP_DIR/failed_jobs.txt"
echo 0 > "$QUEUE_POS"; : > "$FAILED"

JOBS=()
for c in "${CONFIGS[@]}"; do for (( k = 0; k < SHARDS; k++ )); do JOBS+=("$c $k"); done; done

{
    echo "commit=$START_SHA  started=$(date -Is)  host=$(hostname)  gpus=${GPUS[*]}"
    echo "evict=$EVICT_DESC  model=$MODEL_PATH"
    echo "gsm8k_data=$GSM8K_DATA  problems=$TOTAL  shards=$SHARDS x <=$PER"
    echo "configs(budget:local)=${CONFIGS[*]}  q=$QUANT_RATIO mode=$QUANT_MODE gate=$GATE_RATIO window=$WINDOW sink=$SINK"
} > "$SWEEP_DIR/run.env"
for c in "${CONFIGS[@]}"; do
    { head -3 "$SWEEP_DIR/run.env"; echo "budget=${c%%:*} q=$QUANT_RATIO mode=$QUANT_MODE gate=$GATE_RATIO window=$WINDOW local=${c##*:} sink=$SINK"; } > "$(cfg_dir "$c")/run.env"
done

is_done() {    # $1 config, $2 shard
    local d; d="$(cfg_dir "$1")/shard_$2"
    [[ -f "$d/meta.json" && -f "$d/predictions.jsonl" ]] || return 1
    [[ "$(wc -l < "$d/predictions.jsonl")" -eq "$(shard_n "$2")" ]]
}

next_job() {
    local i
    {
        flock 9
        i=$(<"$QUEUE_POS")
        if (( i < ${#JOBS[@]} )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
    } 9>"$LOCK"
}

run_job() {    # $1 gpu, $2 config (budget:local), $3 shard
    local gpu="$1" c="$2" k="$3" root d
    root="$(cfg_dir "$c")"; d="$root/shard_$k"
    local b="${c%%:*}" l="${c##*:}"
    if is_done "$c" "$k"; then echo "[$(date +%T)] gpu=$gpu SKIP   b=$b l=$l shard $k (complete)"; return 0; fi
    if [[ "$(git rev-parse HEAD)" != "$START_SHA" ]]; then
        echo "[$(date +%T)] gpu=$gpu REFUSE b=$b l=$l shard $k -- HEAD moved off $COMMIT" >&2
        echo "b=$b l=$l shard $k rc=HEAD-moved" >> "$FAILED"; return 1
    fi
    rm -rf "$d"
    echo "[$(date +%T)] gpu=$gpu START  b=$b l=$l shard $k/$SHARDS ($(shard_n "$k") problems)"
    local t0=$SECONDS rc
    (
        if [[ $ARM == 1 ]]; then export STICKYKV_EVICT_CONTROL_ARM=1; else unset STICKYKV_EVICT_CONTROL_ARM; fi
        CUDA_VISIBLE_DEVICES="$gpu" exec python main.py --config "$CONFIG" --override \
            model.name="$MODEL_PATH" \
            cache.cache_budget="$b" \
            cache.quant_ratio="$QUANT_RATIO" \
            cache.quant_budget_mode="$QUANT_MODE" \
            cache.quant_gate_ratio="$GATE_RATIO" \
            cache.window_size="$WINDOW"   window.window_size="$WINDOW" \
            cache.local_window_size="$l"  window.local_window_size="$l" \
            cache.num_sink_tokens="$SINK" window.num_sink_tokens="$SINK" \
            cache.first_eviction_step=0 \
            gsm8k.data_dir="$GSM8K_DATA" gsm8k.num_samples=max \
            gsm8k.shard="$k" gsm8k.num_shards="$SHARDS" \
            gsm8k.output_dir="$d" gsm8k.results_dir="$root"
    ) > "$root/shard_$k.log" 2>&1
    rc=$?
    if (( rc == 0 )) && is_done "$c" "$k"; then
        echo "[$(date +%T)] gpu=$gpu DONE   b=$b l=$l shard $k ($(( (SECONDS - t0) / 60 )) min)"
    else
        local why
        why=$(grep -m1 -oE "ran EAGER|OutOfMemoryError|CUDA error[^.]*|[A-Za-z]+Error: [^.]{0,80}" "$root/shard_$k.log" | head -1)
        echo "[$(date +%T)] gpu=$gpu FAIL   b=$b l=$l shard $k rc=$rc ${why:+($why)} -- $root/shard_$k.log" >&2
        echo "b=$b l=$l shard $k rc=$rc ${why:-see log} $root/shard_$k.log" >> "$FAILED"
    fi
}

worker() {
    local gpu="$1" i
    while i=$(next_job) && [[ -n "$i" ]]; do
        # shellcheck disable=SC2086
        run_job "$gpu" ${JOBS[$i]}
    done
}

echo "=== GSM8K: $TOTAL problems x ${#CONFIGS[@]} budgets (${CONFIGS[*]}), $SHARDS shards each: ${#JOBS[@]} jobs on ${#GPUS[@]} GPU(s) ==="
echo "  commit $COMMIT on $(hostname) | eviction: $EVICT_DESC"
echo "  q=$QUANT_RATIO ($QUANT_MODE) gate=$GATE_RATIO window=$WINDOW sink=$SINK"
for c in "${CONFIGS[@]}"; do echo "  out ${c}: $(cfg_dir "$c")"; done
echo

pids=()
for g in "${GPUS[@]}"; do
    worker "$g" 2>&1 | tee -a "$SWEEP_DIR/worker_${g}.log" &
    pids+=($!)
done
wait "${pids[@]}"

# ---- merge + score each budget whose shards are all done ----------------------------
for c in "${CONFIGS[@]}"; do
    root="$(cfg_dir "$c")"; ok=1
    for (( k = 0; k < SHARDS; k++ )); do is_done "$c" "$k" || ok=0; done
    if (( ! ok )); then echo "!! $c: shards incomplete -- not merged (re-run the script)" >&2; continue; fi
    echo "=== merging + scoring $c ==="
    OUT_ROOT="$root" SHARDS="$SHARDS" PER="$PER" TOTAL="$TOTAL" python - <<'PY' || { echo "!! merge failed for $root" >&2; continue; }
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
print(f"  merged {n} predictions")
PY
    python -m modules.evaluation.gsm8k_scoring --predictions "$root/merged/predictions.jsonl" \
        --out_csv "$root/merged/scores.csv" > "$root/merged/scoring.log" 2>&1 \
        || echo "!! scoring FAILED -- $root/merged/scoring.log" >&2
done

# ---- summary --------------------------------------------------------------------
REF40="$PROJECT_ROOT/outputs/gsm8k_b040_q070_w8_l128_${COMMIT}"
ROWS="$( { [[ -d "$REF40/merged" ]] && echo "0.40:128=$REF40"; for c in "${CONFIGS[@]}"; do echo "$c=$(cfg_dir "$c")"; done; } | tr '\n' ' ')" \
SHARDS="$SHARDS" COMMIT="$COMMIT" EVICT_DESC="$EVICT_DESC" python - <<'PY' | tee "$SWEEP_DIR/summary.txt"
import json, os, re, glob
E = os.environ
rows = [tuple(x.split("=", 1)) for x in E["ROWS"].split()]
rows.sort(key=lambda r: -float(r[0].split(":")[0]))
def grab(txt, key):
    m = re.search(rf"^\s*{key}\s+([0-9.]+)", txt, re.M); return float(m.group(1)) if m else None
print(f"GSM8K @ {E['COMMIT']} | eviction: {E['EVICT_DESC']}")
print("q 0.70 (bytes)  gate 0.25  window 8  sink 5  |  1319 problems, greedy, <=512 new tokens\n")
print(f"{'budget':>6} {'local':>5} {'accuracy':>9} {'strict':>7} {'marker%':>8} {'retained':>9} {'seq':>6} {'kept%':>6} {'no-evict%':>9}  gate")
bad = []
for cfg, root in rows:
    b, l = cfg.split(":")
    log = f"{root}/merged/scoring.log"
    metas = [json.load(open(p)) for p in sorted(glob.glob(f"{root}/shard_*/meta.json"))]
    if not os.path.exists(log) or not metas:
        print(f"{float(b):6.2f} {l:>5} {'not done':>9}"); bad.append(cfg); continue
    t = open(log).read()
    acc, strict, mark = grab(t, "accuracy"), grab(t, "accuracy_strict"), grab(t, "marker rate")
    N = sum(m.get("num_examples", 0) for m in metas)
    w = lambda k: sum(m.get(k, 0) * m.get("num_examples", 0) for m in metas) / N
    ret, seq, noev = w("mean_retained_tokens"), w("mean_sequence_tokens"), w("pct_examples_no_eviction")
    gates = sorted({(m.get("read_gate") or {}).get("verdict", "MISSING") for m in metas})
    if gates != ["gated"]: bad.append(f"{cfg} gate={gates}")
    print(f"{float(b):6.2f} {l:>5} {acc:9.2f} {strict:7.2f} {mark:7.2f}% {ret:9.1f} {seq:6.1f} {100*ret/seq:5.0f}% {noev:8.1f}%  {','.join(gates)}")
print("\nkept% = tokens actually retained / tokens in prompt+answer. The budget is a")
print("fraction of prompt + 512, and answers average well under 512, so kept% > budget.")
if bad:
    print("\n!! not usable as-is:", "; ".join(bad))
PY
if [[ -s "$FAILED" ]]; then echo; echo "!! failed jobs (re-run the script to redo just these):"; cat "$FAILED"; exit 1; fi
echo; echo "done: $SWEEP_DIR/summary.txt"
