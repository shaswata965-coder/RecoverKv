#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# FULL-KV efficiency sweep on ONE GPU: flash_attention_2 vs eager, no
# compression, over short prompts and long generations.
#
#   arms     fullkv_flash (attn flash_attention_2) | fullkv_eager (attn eager)
#            both cache_backend: dynamic -- the stock HF cache, nothing evicted,
#            nothing quantized. This is the baseline, not the method.
#   shapes   prefill/decode, AS GIVEN: 256/1024  512/2048  256/4096
#                                      256/8192  256/16384
#   batches  1 8 16 32
#   cells    5 shapes x 4 batches x 2 arms = 40
#   data     wikitext (real text, chunked to exactly prefill_len per row)
#   model    Llama-3.1-8B-Instruct, float16
#
# Short prompt, long generation: this measures how decode slows as the KV cache
# GROWS during generation (256 -> 16640 tokens), which is the opposite regime to
# the perf tables (4096/256). TTFT here is a 256-token prefill and is nearly
# constant; the interesting column is TPOT_steady and the decode throughput.
#
# ------------------------------------------------------------------- memory
# Llama-3.1-8B fp16 KV is 128 KiB/token (2 x 32 layers x 8 kv heads x 128 x 2 B).
# Full cache at the end of generation, (prefill+decode) x batch x 128 KiB:
#            B=1     B=8     B=16    B=32
#   1280    0.2 GB  1.3 GB   2.6 GB   5.2 GB
#   2560    0.3     2.5      5.0     10.0
#   4352    0.5     4.3      8.5     17.0
#   8448    1.0     8.3     16.5     33.0
#  16640    2.0    16.3     32.5     65.0   <- +16.1 GB weights = ~81 GB
# So 256/16384 at B=32 is expected to OOM on an 80 GB card, in BOTH arms.
# skip_if_oom is on, so that cell is recorded as OOM and the sweep continues;
# every other cell fits. prefill_logits=last_only keeps the logits tensor from
# adding a batch x 256 x 128256 term to that.
#
# ------------------------------------------------------------------- runtime
# ~13,000 generated tokens per batch column per arm. At a full-KV decode step of
# roughly 15-40 ms, one measurement run over all 20 cells is ~45-60 min per arm,
# so ~1.5-2 h total at RUNS=1, WARMUP=0 (the defaults here). RUNS=3 triples it.
# Ask for >= 4 h walltime, more if you raise RUNS.
#
# WARMUP=0 by default because a warmup run repeats the WHOLE cell, and these
# cells are long. The cost: run 0 of each cell carries any first-touch CUDA/
# kernel setup. Set WARMUP=1 (and RUNS>=2) for a clean published number.
#
# ------------------------------------------------------------------ configure
#   MODEL_PATH   checkpoint            (default: llama-3.1-8b-instruct)
#   OUT_DIR      output root           (default: outputs/efficiency_fullkv_<commit>)
#   DATA_SOURCE  prompt text           (default: outputs/corpora/wikitext_test.txt,
#                                       falls back to the wikitext-103 hub name)
#   SHAPES       "prefill/decode ..."  (default: the five above)
#   BATCHES      "1 8 16 32"
#   RUNS/WARMUP  measurement/warmup runs per cell   (default: 1 / 0)
#   COOLDOWN     seconds idled between runs         (default: 3)
#   ARMS         "flash eager"         (either alone is fine)
#
# Single GPU by construction: perf_runner refuses a sharded model, and this
# script pins CUDA_VISIBLE_DEVICES to one device before anything loads.
#
# Run:
#   cd ~/kv_cache/Cluster_QEvict/RecoverKv && conda activate sticky_env
#   nohup scripts/run_efficiency_fullkv.sh > ~/eff_fullkv.log 2>&1 &
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
SHAPES="${SHAPES:-256/1024 512/2048 256/4096 256/8192 256/16384}"
BATCHES="${BATCHES:-1 8 16 32}"
RUNS="${RUNS:-1}"; WARMUP="${WARMUP:-0}"; COOLDOWN="${COOLDOWN:-3}"
DTYPE="${DTYPE:-float16}"; STAT="${STAT:-median}"
read -r -a ARMS <<< "${ARMS:-flash eager}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/efficiency_fullkv_${COMMIT}}"

# wikitext: prefer the local corpus (no network from a compute node), else the hub name.
if [[ -z "${DATA_SOURCE:-}" ]]; then
    if [[ -s "$PROJECT_ROOT/outputs/corpora/wikitext_test.txt" ]]; then
        DATA_SOURCE="$PROJECT_ROOT/outputs/corpora/wikitext_test.txt"
    else
        DATA_SOURCE="wikitext-103"
    fi
fi

die() { echo "error: $*" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
command -v nvidia-smi >/dev/null || die "no nvidia-smi -- login node? get a GPU node."
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || die "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES selects several GPUs.
       A sharded model times the interconnect, not the cache; perf_runner refuses it
       after the weights load. Re-run with CUDA_VISIBLE_DEVICES=<one id or UUID>."
for a in "${ARMS[@]}"; do [[ "$a" == flash || "$a" == eager ]] || die "ARMS entries must be flash or eager, got '$a'"; done
python - <<'PY' || exit 2
import sys
try:
    import torch
except ImportError:
    sys.exit("error: torch not importable -- 'conda activate sticky_env'")
if not torch.cuda.is_available():
    sys.exit("error: torch.cuda.is_available() is False -- wrong env or no GPU allocation.")
PY
FLASH_OK=1
python -c "import flash_attn" 2>/dev/null || FLASH_OK=0
if (( ! FLASH_OK )); then
    for a in "${ARMS[@]}"; do [[ "$a" == flash ]] && die "flash_attn not importable but ARMS includes flash.
       skip_if_flash_attn_unavailable would SKIP every flash cell and still exit 0
       (that has happened before) -- refusing instead. Fix the env or use ARMS=eager."; done
fi

mkdir -p "$OUT_DIR"
CONFIG_FILE="$OUT_DIR/_efficiency_fullkv.generated.yaml"

# ---- grid ------------------------------------------------------------------
GRID=""
for s in $SHAPES; do
    p="${s%%/*}"; d="${s##*/}"
    [[ "$p" =~ ^[0-9]+$ && "$d" =~ ^[0-9]+$ && "$p" != "$s" ]] || die "--shapes items must be prefill/decode, got '$s'"
    # gen_len = decode + 1: the prefill emits the first token, so decode steps == d.
    for b in $BATCHES; do
        [[ "$b" =~ ^[0-9]+$ ]] || die "batches must be integers, got '$b'"
        GRID+="    - { prefill_len: ${p}, gen_len: $(( d + 1 )), batch_size: ${b} }
"
    done
done

ARM_YAML=""
for a in "${ARMS[@]}"; do
    if [[ "$a" == flash ]]; then
        ARM_YAML+="    - name: fullkv_flash
      cache_backend: dynamic
      attn_implementation: flash_attention_2
      requires_flash_attn: true
"
    else
        ARM_YAML+="    - name: fullkv_eager
      cache_backend: dynamic
      attn_implementation: eager
      requires_flash_attn: false
"
    fi
done

cat > "$CONFIG_FILE" <<YAML
# GENERATED by scripts/run_efficiency_fullkv.sh -- safe to delete.
# Self-contained, like run_perf_table.sh's: NO _base_, because \`_base_\` resolves
# relative to THIS file and it lives in the output dir, not in configs/. Every
# value the run depends on is written out below; the rest are dataclass defaults.

run:
  mode: perf
  seed: 42

model:
  name: ${MODEL_PATH}
  revision: null
  dtype: ${DTYPE}

cache:
  backend: dynamic          # full KV; per-arm backend is set in perf.configs
  backend_package: null
  cache_budget: null

telemetry:
  track_scores: false
  output_dir: ${OUT_DIR}

perf:
  protocol: native
  prompt_mode: dataset
  data_source: ${DATA_SOURCE}
  prefill_logits: last_only
  budget_basis: context
  warmup_decode_steps: null
  cooldown_s: ${COOLDOWN}
  enable_clock_locking: true      # needs elevated permissions; falls back with a warning
  clock_settle_s: 5.0
  num_warmup_runs: ${WARMUP}
  num_measurement_runs: ${RUNS}
  allow_shared_gpu: true
  skip_if_oom: true               # the B=32 x 16384 cell is expected to OOM
  skip_if_flash_attn_unavailable: true

  configs:
${ARM_YAML}
  grid:
${GRID}
YAML

{
    git rev-parse HEAD 2>/dev/null || echo no_git
    echo "host=$(hostname)  started=$(date -Is)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
    echo "model=$MODEL_PATH dtype=$DTYPE  data_source=$DATA_SOURCE"
    echo "arms=${ARMS[*]}  shapes=$SHAPES  batches=$BATCHES"
    echo "runs=$RUNS warmup=$WARMUP cooldown=$COOLDOWN stat=$STAT"
} > "$OUT_DIR/run.env"

echo "=== full-KV efficiency: arms=${ARMS[*]} | $(wc -l <<< "$GRID") cells per arm ==="
cat "$OUT_DIR/run.env" | sed 's/^/  /'
echo "  config: $CONFIG_FILE"
echo "  out:    $OUT_DIR"
echo

python "$PROJECT_ROOT/main.py" --config "$CONFIG_FILE"
rc=$?
(( rc == 0 )) || { echo "!! main.py exited $rc -- see the log above" >&2; exit "$rc"; }

# ---- one table per arm, then the two side by side ---------------------------
for a in "${ARMS[@]}"; do
    name="fullkv_$a"
    python "$PROJECT_ROOT/scripts/print_perf_table.py" --npz-dir "$OUT_DIR" \
        --config "$name" --stat "$STAT" --throughput all \
        --out "$OUT_DIR/table_${name}.txt" > "$OUT_DIR/print_${name}.log" 2>&1 \
        || echo "!! printing $name failed -- $OUT_DIR/print_${name}.log" >&2
done

OUT_DIR="$OUT_DIR" ARMS="${ARMS[*]}" SHAPES="$SHAPES" BATCHES="$BATCHES" python - <<'PY' | tee "$OUT_DIR/summary.txt"
import os, re
E = os.environ
out, arms = E["OUT_DIR"], E["ARMS"].split()
shapes = [tuple(int(x) for x in s.split("/")) for s in E["SHAPES"].split()]
batches = [int(b) for b in E["BATCHES"].split()]

def rows(arm):
    """(prefill, decode, batch) -> (tpot, dec_tok_s, peak_gb) or None, from the printed table."""
    p = f"{out}/table_fullkv_{arm}.txt"
    got = {}
    if not os.path.exists(p):
        return got
    for line in open(p):
        m = re.match(r"\s*(\d+)/(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
        if not m:
            continue
        pre, gen, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        vals = m.group(5), m.group(6), m.group(9)          # TPOT, dec tok/s, peak_GB
        got[(pre, gen - 1, b)] = vals if m.group(4) != "ERROR" else None
    return got

data = {a: rows(a) for a in arms}
print("Full-KV efficiency (no compression), one GPU")
print(open(f"{out}/run.env").read().strip())
print()
hdr = f"{'prefill/decode':>16}{'batch':>6}"
for a in arms:
    hdr += f"{a + ' TPOT':>13}{a + ' tok/s':>13}{a + ' peakGB':>13}"
print(hdr)
for pre, dec in shapes:
    for b in batches:
        line = f"{f'{pre}/{dec}':>16}{b:6d}"
        for a in arms:
            v = data[a].get((pre, dec, b), "missing")
            if v is None:
                line += f"{'ERROR/OOM':>39}"
            elif v == "missing":
                line += f"{'-':>13}{'-':>13}{'-':>13}"
            else:
                line += f"{v[0]:>13}{v[1]:>13}{v[2]:>13}"
        print(line)
print("\nTPOT = seconds per generated token at steady state; tok/s = batch / TPOT.")
print("peak_GB is the whole-process peak, weights included (~16.1 GB for fp16 8B).")
missing = [(a, k) for a in arms for k in [(p, d, b) for p, d in shapes for b in batches] if k not in data[a]]
oom = [(a, k) for a in arms for k, v in data[a].items() if v is None]
if oom:
    print("\nERROR/OOM cells (expected at 256/16384 B=32: ~65 GB KV + 16 GB weights):")
    for a, k in oom: print(f"   {a}: {k[0]}/{k[1]} batch {k[2]}")
if missing:
    print(f"\n!! {len(missing)} cell(s) produced no row -- re-run, or check {out}/print_*.log")
PY
echo
echo "tables: $OUT_DIR/table_fullkv_*.txt   summary: $OUT_DIR/summary.txt"
