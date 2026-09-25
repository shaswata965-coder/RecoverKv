#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# FULL-KV efficiency sweep spread over every GPU of the node: one CELL per job.
# Same measurement as scripts/run_efficiency_fullkv.sh (which does all 40 cells
# serially on one GPU); this one runs them in parallel.
#
#   arms     fullkv_flash (flash_attention_2) | fullkv_eager (eager)
#            both cache_backend: dynamic -- nothing evicted, nothing quantized
#   shapes   prefill/decode: 256/1024  512/2048  256/4096  256/8192  256/16384
#   batches  1 8 16 32
#   jobs     5 shapes x 4 batches x 2 arms = 40, one process on one GPU each
#   data     wikitext | model Llama-3.1-8B-Instruct, float16
#
# ---------------------------------------------------------- one cell per job
# A perf npz is named perf_prefill{P}_gen{G}_bs{B}.npz -- shape and batch only,
# nothing about the arm (CLAUDE.md "Two traps in the harness"). So each ARM gets
# its own directory and two arms never collide; within an arm, cells differ by
# filename and share one directory safely.
#
# Each job measures exactly one cell, so a cell's numbers come from a process
# that did nothing else -- no other cell warmed the clocks or the allocator for
# it. The cost is one model load (~1-2 min) per job, ~1 GPU-h over 40 jobs,
# which parallelises away.
#
# Longest first, by decode steps x a batch weight, so the 16384-step cells start
# immediately and the 1024-step ones fill in behind them. Expected makespan on
# 8 GPUs: ~20-30 min, set by the longest single cell (256/16384 at B=16).
# Serial on one GPU the same work is ~1.5-2 h.
#
# ------------------------------------------------------------------- memory
# Llama-3.1-8B fp16 KV is 128 KiB/token. At the end of generation,
# (prefill+decode) x batch x 128 KiB:
#            B=1     B=8     B=16    B=32
#   1280    0.2 GB  1.3 GB   2.6 GB   5.2 GB
#   2560    0.3     2.5      5.0     10.0
#   4352    0.5     4.3      8.5     17.0
#   8448    1.0     8.3     16.5     33.0
#  16640    2.0    16.3     32.5     65.0   <- +16.1 GB weights = ~81 GB
# 256/16384 at B=32 is expected to OOM on an 80 GB card in BOTH arms; skip_if_oom
# records it and the job still exits 0. Every other cell fits.
#
# ONE ARM PER GPU, ONE CELL AT A TIME. Each GPU is pinned to a single arm for the
# whole run (round-robin over the GPUs: with 8 GPUs and 2 arms, 4 measure flash
# and 4 measure eager) and runs its cells one after another, each with a single
# CUDA_VISIBLE_DEVICES entry. So a timing is never taken beside another cell, and
# never on a device that is also running the other implementation. Both arms hold
# the same cells, so the halves carry equal work.
#
# ------------------------------------------------------------------ configure
#   MODEL_PATH DATA_SOURCE SHAPES BATCHES ARMS RUNS WARMUP COOLDOWN DTYPE STAT
#   (same meanings as run_efficiency_fullkv.sh; RUNS=1 WARMUP=0 by default)
#
# Resumable: a cell whose npz exists is skipped. Delete the npz to redo it.
#
# Run on the 8-GPU node:
#   cd ~/kv_cache/Cluster_QEvict/RecoverKv && conda activate sticky_env
#   nohup scripts/run_efficiency_fullkv_8gpu.sh > ~/eff_fullkv_8gpu.log 2>&1 &
#   tail -f ~/eff_fullkv_8gpu.log
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
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/efficiency_fullkv_${COMMIT}}"

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
for a in "${ARMS[@]}"; do [[ "$a" == flash || "$a" == eager ]] || die "ARMS entries must be flash or eager, got '$a'"; done
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"   # PBS UUIDs, used as given
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
(( ${#GPUS[@]} > 0 )) || die "no GPUs visible"
python - <<'PY' || exit 2
import sys
try:
    import torch
except ImportError:
    sys.exit("error: torch not importable -- 'conda activate sticky_env'")
if not torch.cuda.is_available():
    sys.exit("error: torch.cuda.is_available() is False -- wrong env or no GPU allocation.")
PY
if ! python -c "import flash_attn" 2>/dev/null; then
    for a in "${ARMS[@]}"; do [[ "$a" == flash ]] && die "flash_attn not importable but ARMS includes flash.
       skip_if_flash_attn_unavailable would SKIP every flash cell and still exit 0 --
       refusing instead. Fix the env or use ARMS=eager."; done
fi

mkdir -p "$OUT_ROOT/configs"
for a in "${ARMS[@]}"; do mkdir -p "$OUT_ROOT/$a"; done
LOCK="$OUT_ROOT/.queue_lock"; FAILED="$OUT_ROOT/failed_cells.txt"; : > "$FAILED"
for a in "${ARMS[@]}"; do echo 0 > "$OUT_ROOT/.queue_pos_$a"; done

# ---- jobs: one per (arm, shape, batch), longest first ------------------------
# weight ~ decode steps x (1 + batch/16): the per-step cost grows with batch, but
# far less than linearly (decode is weight-bound at small batch).
# One job list PER ARM, each longest-first. A GPU is pinned to one arm (below),
# so it only ever pulls from that arm's list.
declare -A JOBLIST
for a in "${ARMS[@]}"; do
    lines=""
    for s in $SHAPES; do
        p="${s%%/*}"; d="${s##*/}"
        [[ "$p" =~ ^[0-9]+$ && "$d" =~ ^[0-9]+$ && "$p" != "$s" ]] || die "SHAPES items must be prefill/decode, got '$s'"
        for b in $BATCHES; do
            [[ "$b" =~ ^[0-9]+$ ]] || die "BATCHES must be integers, got '$b'"
            lines+="$(( d * (16 + b) / 16 )) $p $d $b
"
        done
    done
    JOBLIST[$a]="$(printf '%s' "$lines" | sort -rn -k1,1)"
    # Trailing newline is load-bearing: next_job() counts lines with `wc -l`,
    # which would report one short for an unterminated last line and silently
    # drop the last cell of every arm (38/40 instead of 40/40).
    printf '%s\n' "${JOBLIST[$a]}" > "$OUT_ROOT/.jobs_$a"
done
N_CELLS=$(( $(wc -l < "$OUT_ROOT/.jobs_${ARMS[0]}") * ${#ARMS[@]} ))

# GPU -> arm, round-robin. ONE ARM PER GPU for the whole run, and one job at a
# time per GPU: a cell is never measured beside another cell, and never on a
# device that is also running the other implementation.
declare -A GPU_ARM
for i in "${!GPUS[@]}"; do GPU_ARM[${GPUS[$i]}]="${ARMS[$(( i % ${#ARMS[@]} ))]}"; done
if (( ${#GPUS[@]} < ${#ARMS[@]} )); then
    die "${#GPUS[@]} GPU(s) but ${#ARMS[@]} arms: with one arm per GPU, each arm needs its own GPU.
       Run the arms in separate invocations (ARMS=flash ... then ARMS=eager ...)."
fi

{
    git rev-parse HEAD 2>/dev/null || echo no_git
    echo "host=$(hostname)  started=$(date -Is)  gpus=${GPUS[*]}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
    echo "model=$MODEL_PATH dtype=$DTYPE  data_source=$DATA_SOURCE"
    echo "arms=${ARMS[*]}  shapes=$SHAPES  batches=$BATCHES  cells=$N_CELLS"
    for g in "${GPUS[@]}"; do echo "  gpu ${g} -> arm ${GPU_ARM[$g]} (only this arm, one cell at a time)"; done
    echo "runs=$RUNS warmup=$WARMUP cooldown=$COOLDOWN stat=$STAT"
} > "$OUT_ROOT/run.env"

npz_of() { printf '%s/%s/perf_prefill%s_gen%s_bs%s.npz' "$OUT_ROOT" "$1" "$2" "$(( $3 + 1 ))" "$4"; }   # arm p d b

next_job() {              # $1 = arm; prints that arm's next "weight p d b" line
    local a="$1" i n
    {
        flock 9
        i=$(<"$OUT_ROOT/.queue_pos_$a")
        n=$(wc -l < "$OUT_ROOT/.jobs_$a")
        if (( i < n )); then
            echo $((i + 1)) > "$OUT_ROOT/.queue_pos_$a"
            sed -n "$((i + 1))p" "$OUT_ROOT/.jobs_$a"
        fi
    } 9>"$LOCK"
}

run_cell() {   # $1 gpu, $2 arm, $3 prefill, $4 decode, $5 batch
    local gpu="$1" a="$2" p="$3" d="$4" b="$5" npz cfg log
    npz="$(npz_of "$a" "$p" "$d" "$b")"
    cfg="$OUT_ROOT/configs/${a}_p${p}_d${d}_b${b}.yaml"
    log="$OUT_ROOT/$a/cell_p${p}_d${d}_b${b}.log"
    if [[ -s "$npz" ]]; then echo "[$(date +%T)] gpu=${gpu:0:12} SKIP  $a $p/$d b=$b (npz exists)"; return 0; fi

    local impl requires
    if [[ "$a" == flash ]]; then impl=flash_attention_2; requires=true; else impl=eager; requires=false; fi
    cat > "$cfg" <<YAML
# GENERATED by scripts/run_efficiency_fullkv_8gpu.sh -- one cell. Safe to delete.
# Self-contained (no _base_: it would resolve next to THIS file, not configs/).
run:
  mode: perf
  seed: 42
model:
  name: ${MODEL_PATH}
  revision: null
  dtype: ${DTYPE}
cache:
  backend: dynamic
  backend_package: null
  cache_budget: null
telemetry:
  track_scores: false
  output_dir: ${OUT_ROOT}/${a}
perf:
  protocol: native
  prompt_mode: dataset
  data_source: ${DATA_SOURCE}
  prefill_logits: last_only
  budget_basis: context
  warmup_decode_steps: null
  cooldown_s: ${COOLDOWN}
  enable_clock_locking: true
  clock_settle_s: 5.0
  num_warmup_runs: ${WARMUP}
  num_measurement_runs: ${RUNS}
  allow_shared_gpu: true
  skip_if_oom: true
  skip_if_flash_attn_unavailable: true
  configs:
    - name: fullkv_${a}
      cache_backend: dynamic
      attn_implementation: ${impl}
      requires_flash_attn: ${requires}
  grid:
    - { prefill_len: ${p}, gen_len: $(( d + 1 )), batch_size: ${b} }
YAML

    echo "[$(date +%T)] gpu=${gpu:0:12} START $a $p/$d b=$b"
    local t0=$SECONDS rc
    CUDA_VISIBLE_DEVICES="$gpu" python "$PROJECT_ROOT/main.py" --config "$cfg" > "$log" 2>&1
    rc=$?
    if (( rc == 0 )) && [[ -s "$npz" ]]; then
        local note=""
        grep -q "OOM" "$log" && note=" (OOM recorded)"
        echo "[$(date +%T)] gpu=${gpu:0:12} DONE  $a $p/$d b=$b ($(( (SECONDS - t0) / 60 )) min)$note"
    else
        local why
        why=$(grep -m1 -oE "OutOfMemoryError|CUDA error[^.]*|[A-Za-z]+Error: [^.]{0,80}" "$log" | head -1)
        echo "[$(date +%T)] gpu=${gpu:0:12} FAIL  $a $p/$d b=$b rc=$rc ${why:+($why)} -- $log" >&2
        { flock 9; echo "$a $p/$d b=$b rc=$rc ${why:-see log} $log" >> "$FAILED"; } 9>"$LOCK"
    fi
}

worker() {                # one GPU, one arm, one cell at a time
    local gpu="$1" a="$2" job
    while job=$(next_job "$a") && [[ -n "$job" ]]; do
        # shellcheck disable=SC2086
        set -- $job               # weight prefill decode batch
        run_cell "$gpu" "$a" "$2" "$3" "$4"
    done
}

echo "=== full-KV efficiency, $N_CELLS cells on ${#GPUS[@]} GPU(s): one arm per GPU, one cell at a time ==="
sed 's/^/  /' "$OUT_ROOT/run.env"
echo
pids=()
for g in "${GPUS[@]}"; do
    worker "$g" "${GPU_ARM[$g]}" 2>&1 | tee -a "$OUT_ROOT/worker_${g}.log" &
    pids+=($!)
done
wait "${pids[@]}"

# ---- one table per arm, then both side by side ------------------------------
echo; echo "=== tables ==="
for a in "${ARMS[@]}"; do
    python "$PROJECT_ROOT/scripts/print_perf_table.py" --npz-dir "$OUT_ROOT/$a" \
        --config "fullkv_$a" --stat "$STAT" --throughput all \
        --out "$OUT_ROOT/table_fullkv_${a}.txt" > "$OUT_ROOT/print_${a}.log" 2>&1 \
        || echo "!! printing $a failed -- $OUT_ROOT/print_${a}.log" >&2
done

OUT_DIR="$OUT_ROOT" ARMS="${ARMS[*]}" SHAPES="$SHAPES" BATCHES="$BATCHES" python - <<'PY' | tee "$OUT_ROOT/summary.txt"
import os, re
E = os.environ
out, arms = E["OUT_DIR"], E["ARMS"].split()
shapes = [tuple(int(x) for x in s.split("/")) for s in E["SHAPES"].split()]
batches = [int(b) for b in E["BATCHES"].split()]

def rows(arm):
    p, got = f"{out}/table_fullkv_{arm}.txt", {}
    if not os.path.exists(p):
        return got
    for line in open(p):
        m = re.match(r"\s*(\d+)/(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)) - 1, int(m.group(3)))
        got[key] = (m.group(5), m.group(6), m.group(9)) if m.group(4) != "ERROR" else None
    return got

data = {a: rows(a) for a in arms}
print("Full-KV efficiency (no compression), one cell per GPU job")
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
oom = [(a, k) for a in arms for k, v in data[a].items() if v is None]
missing = [(a, k) for a in arms for k in [(p, d, b) for p, d in shapes for b in batches] if k not in data[a]]
if oom:
    print("\nERROR/OOM cells (expected at 256/16384 B=32: ~65 GB KV + 16 GB weights):")
    for a, k in oom: print(f"   {a}: {k[0]}/{k[1]} batch {k[2]}")
if missing:
    print(f"\n!! {len(missing)} cell(s) produced no row -- re-run the script (finished cells are skipped):")
    for a, k in missing[:10]: print(f"   {a}: {k[0]}/{k[1]} batch {k[2]}")
PY

if [[ -s "$FAILED" ]]; then echo; echo "!! failed cells (re-run to redo just these):"; cat "$FAILED"; exit 1; fi
echo; echo "tables: $OUT_ROOT/table_fullkv_*.txt   summary: $OUT_ROOT/summary.txt"
