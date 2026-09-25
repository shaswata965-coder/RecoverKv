#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# OURS efficiency sweep on ONE GPU: the two-tier int2 cache + rank-1 card read
# gate, over the same short-prompt / long-generation shapes the full-KV
# baseline (scripts/run_efficiency_fullkv.sh) was measured on.
#
#   method   cache_backend: windowed, quant_ratio 0.70 (bytes), gate 0.25
#            sink 5, local 128, flash_attn -- the shipped operating point
#            except for the two knobs this sweep moves.
#   grid     cache_budget  0.20 0.50 0.30        (3)
#            window_size   8 16 32 64            (4)      = 12 configs
#            shapes        256/1024 512/2048 256/4096 256/8192 256/16384
#            batches       1 8 16 32                      = 20 cells each
#            240 measured cells in total.
#   data     wikitext (real text, chunked to exactly prefill_len per row)
#   model    Llama-3.1-8B-Instruct, float16
#
# ---------------------------------------------------- one directory per config
# perf_runner names its npz `perf_prefill{P}_gen{G}_bs{B}.npz` -- SHAPE and
# BATCH, nothing about the config. Twelve configs in one OUT_DIR would overwrite
# each other and the table would print the survivor under every name. So each
# config gets its own directory and its own main.py invocation. That also makes
# the sweep RESUMABLE, which a ~12 h run needs: a config whose table.txt exists
# is skipped. Delete that file to force a redo.
#
# ------------------------------------------------------------- budget_basis
# `prefill_plus_gen`, NOT the `context` that run_perf_table.sh uses, and this is
# load-bearing here. Under `context` the budget is cache_budget x prefill_len:
# at prefill 256 and budget 0.20 that is 51 tokens, below sink(5)+local(128)=133,
# so config.py clamps to ZERO evictable windows, retains 133 tokens/row, warns,
# and measures something that is not the requested budget. Every shape in this
# sweep has a short prompt, so `context` would silently void all 240 cells.
# Under prefill_plus_gen the smallest cell (256/1024 at 0.20) budgets 256
# tokens/row, above the 133 floor. The validator below re-checks every
# (budget, shape) pair before a GPU-hour is spent.
#
# ----------------------------------------------------------- measurement setup
# RUNS=1, WARMUP=0, COOLDOWN=3, clock locking on -- identical to the full-KV
# efficiency run, so ours/full-KV is a like-for-like ratio. With WARMUP=0 the
# first cell of a config would otherwise carry Triton JIT + autotune for that
# window_size, so a throwaway PRIME cell (256/32, batch 1, a few seconds) is put
# at the FRONT of every grid and dropped from the summary. Set PRIME=0 to remove
# it; set WARMUP=1 RUNS=2+ for a published number, at 2-3x the runtime.
#
# -------------------------------------------------------------------- runtime
# ~13,000 generated tokens per batch column, 20 cells per config, so roughly
# 45-75 min per config and ~9-15 h for all twelve. Ask for >= 16 h of walltime,
# or split the grid across sessions -- it resumes:
#     BUDGETS=0.20 scripts/run_efficiency_ours.sh        # 4 configs, ~4 h
#     WINDOW_SIZES="8 16" scripts/run_efficiency_ours.sh # 6 configs, ~6 h
#
# ------------------------------------------------------------------ configure
#   MODEL_PATH    checkpoint         (default: llama-3.1-8b-instruct)
#   OUT_DIR       output root        (default: outputs/efficiency_ours_<commit>)
#   DATA_SOURCE   prompt text        (default: outputs/corpora/wikitext_test.txt)
#   BUDGETS       "0.20 0.50 0.30"
#   WINDOW_SIZES  "8 16 32 64"
#   SHAPES        "prefill/decode ..."
#   BATCHES       "1 8 16 32"
#   QUANT_RATIO / QUANT_MODE / GATE_RATIO / NUM_SINK / LOCAL_WINDOW
#   RUNS / WARMUP / COOLDOWN / CLOCK_LOCK / DTYPE / STAT / BACKEND / PRIME
#   FULLKV_DIR    a full-KV efficiency OUT_DIR; if its table_fullkv_flash.txt
#                 exists the summary adds the ours-vs-full-KV speedup.
#   DRY_RUN=1     validate the grid and list the configs, run nothing.
#
# Run:
#   cd ~/kv_cache/Cluster_QEvict/RecoverKv && conda activate sticky_env
#   nohup scripts/run_efficiency_ours.sh > ~/eff_ours.log 2>&1 &
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 2

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
BUDGETS="${BUDGETS:-0.20 0.50 0.30}"
WINDOW_SIZES="${WINDOW_SIZES:-8 16 32 64}"
SHAPES="${SHAPES:-256/1024 512/2048 256/4096 256/8192 256/16384}"
BATCHES="${BATCHES:-1 8 16 32}"
QUANT_RATIO="${QUANT_RATIO:-0.70}"
QUANT_MODE="${QUANT_MODE:-bytes}"
GATE_RATIO="${GATE_RATIO:-0.25}"
NUM_SINK="${NUM_SINK:-5}"
LOCAL_WINDOW="${LOCAL_WINDOW:-128}"
RUNS="${RUNS:-1}"; WARMUP="${WARMUP:-0}"; COOLDOWN="${COOLDOWN:-3}"
CLOCK_LOCK="${CLOCK_LOCK:-true}"
DTYPE="${DTYPE:-float16}"; STAT="${STAT:-median}"
BACKEND="${BACKEND:-flash_attn}"
PRIME="${PRIME:-1}"
DRY_RUN="${DRY_RUN:-0}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/efficiency_ours_${COMMIT}}"
FULLKV_DIR="${FULLKV_DIR:-}"

die() { echo "error: $*" >&2; exit 2; }

# ---- preflight --------------------------------------------------------------
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
case "$BACKEND" in
    flash_attn) ATTN_IMPL="flash_attention_2"; NEEDS_FLASH=true;;
    eager)      ATTN_IMPL="eager";             NEEDS_FLASH=false;;
    *) die "BACKEND must be flash_attn or eager, got '$BACKEND'";;
esac

# The GPU-only checks are skipped under DRY_RUN=1 so the grid can be validated
# from a login node before a job is queued.
if [[ "$DRY_RUN" != "1" && "${GEN_ONLY:-0}" != "1" ]]; then
command -v nvidia-smi >/dev/null || die "no nvidia-smi -- login node? get a GPU node."
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || die "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES selects several GPUs.
       device_map=auto would shard the model by layer and a sharded decode step
       times the interconnect, not the cache. perf_runner refuses it after the
       weights load; this refuses it now. Use one id or UUID."
python - <<'PY' || exit 2
import sys
try:
    import torch
except ImportError:
    sys.exit("error: torch not importable -- 'conda activate sticky_env'")
if not torch.cuda.is_available():
    sys.exit("error: torch.cuda.is_available() is False -- wrong env or no GPU allocation.")
PY

if [[ "$BACKEND" == "flash_attn" ]]; then
    python -c "import flash_attn" 2>/dev/null || die "flash_attn not importable.
       skip_if_flash_attn_unavailable would SKIP every cell and still exit 0
       (that has happened before) -- refusing instead. Fix the env, or BACKEND=eager."
fi
# The gate IS the read path: _run_fused raises without cards, and the fused
# kernel is Triton. No Triton, no run -- and better to know that now.
python -c "import triton" 2>/dev/null || die "triton not importable, but the fused
       decode kernel (and therefore the read gate) is Triton. This sweep cannot
       measure the method without it."
fi

if [[ -z "${DATA_SOURCE:-}" ]]; then
    if [[ -s "$PROJECT_ROOT/outputs/corpora/wikitext_test.txt" ]]; then
        DATA_SOURCE="$PROJECT_ROOT/outputs/corpora/wikitext_test.txt"
    else
        DATA_SOURCE="wikitext-103"
    fi
fi

# ---- validate the grid before spending a GPU-day on it ----------------------
for ws in $WINDOW_SIZES; do
    [[ "$ws" =~ ^[0-9]+$ ]] || die "WINDOW_SIZES entries must be integers, got '$ws'"
    (( ws % 4 == 0 )) || die "window_size=$ws is not divisible by 4 and quant_ratio=$QUANT_RATIO > 0.
       int2 packs 4 codes per byte along the window axis (config.py)."
    if [[ "$LOCAL_WINDOW" =~ ^[0-9]+$ ]]; then
        (( LOCAL_WINDOW % ws == 0 )) || die "local_window_size=$LOCAL_WINDOW is not a multiple of window_size=$ws"
    fi
done
for b in $BATCHES; do [[ "$b" =~ ^[0-9]+$ ]] || die "BATCHES entries must be integers, got '$b'"; done
for s in $SHAPES; do
    p="${s%%/*}"; d="${s##*/}"
    [[ "$p" =~ ^[0-9]+$ && "$d" =~ ^[0-9]+$ && "$p" != "$s" ]] || die "SHAPES items must be prefill/decode, got '$s'"
done

# The clamp check the header describes: budget = ratio x (prefill + gen_len),
# and a budget below sink+local silently becomes "retain sink+local, 0 evictable
# windows" -- a run that is not at the budget it is labelled with.
BUDGETS="$BUDGETS" SHAPES="$SHAPES" NUM_SINK="$NUM_SINK" LOCAL_WINDOW="$LOCAL_WINDOW" \
WINDOW_SIZES="$WINDOW_SIZES" \
python - <<'PY' || exit 2
import os, sys
E = os.environ
sink, local = int(E["NUM_SINK"]), E["LOCAL_WINDOW"]
if "." in local:      # fractional local is resolved against the budget itself
    sys.exit(0)
floor = sink + int(local)
bad = []
for bs in E["BUDGETS"].split():
    b = float(bs)
    if not 0 < b <= 1:
        sys.exit(f"error: BUDGETS entries must be in (0,1], got {bs}")
    for s in E["SHAPES"].split():
        p, d = (int(x) for x in s.split("/"))
        tok = int(b * (p + d + 1))          # gen_len = decode + 1
        if tok < floor:
            bad.append((bs, s, tok))
# Advisory, not an error: a cell whose evictable region holds only a couple of
# windows has a Q tier of 1-2 windows, and `gate_ratio x N_q` then rounds to
# "read everything". Those rows time the machinery, not the selection.
thin = []
for bs in E["BUDGETS"].split():
    b = float(bs)
    for ws in (int(w) for w in E["WINDOW_SIZES"].split()):
        for s in E["SHAPES"].split():
            p, d = (int(x) for x in s.split("/"))
            n = max(int(b * (p + d + 1)) - floor, 0) // ws
            if n < 4:
                thin.append((bs, ws, s, n))
if thin:
    print(f"note: {len(thin)} cell(s) have fewer than 4 evictable windows, so the "
          f"read gate\n      has almost nothing to select over and their "
          f"realised read fraction will sit\n      near 1.0. They are still "
          f"valid rows; they just do not price selection:")
    for bs, ws, s, n in thin[:12]:
        print(f"        budget {bs}  ws {ws:>2}  shape {s:>10}  -> {n} window(s)")
    if len(thin) > 12:
        print(f"        ... and {len(thin) - 12} more")

if bad:
    print("error: these (budget, shape) pairs budget fewer tokens than "
          f"num_sink({sink}) + local({local}) = {floor}.", file=sys.stderr)
    print("       config.py clamps them to 0 evictable windows and the cell then "
          "measures\n       sink+local retention, NOT the budget it is labelled "
          "with. Lower LOCAL_WINDOW\n       or drop the pair:", file=sys.stderr)
    for bs, s, tok in bad:
        print(f"         budget {bs}  shape {s}  -> {tok} tokens/row", file=sys.stderr)
    sys.exit(2)
PY

mkdir -p "$OUT_DIR"

N_SHAPES=$(wc -w <<< "$SHAPES"); N_BATCH=$(wc -w <<< "$BATCHES")
N_CELLS=$(( N_SHAPES * N_BATCH ))
N_TOTAL=0
for b in $BUDGETS; do for ws in $WINDOW_SIZES; do N_TOTAL=$(( N_TOTAL + 1 )); done; done

{
    git rev-parse HEAD 2>/dev/null || echo no_git
    echo "host=$(hostname)  started=$(date -Is)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
    echo "model=$MODEL_PATH dtype=$DTYPE  data_source=$DATA_SOURCE  backend=$BACKEND"
    echo "budgets=[$BUDGETS]  window_sizes=[$WINDOW_SIZES]"
    echo "quant_ratio=$QUANT_RATIO ($QUANT_MODE)  gate_ratio=$GATE_RATIO  sink=$NUM_SINK  local=$LOCAL_WINDOW"
    echo "shapes=$SHAPES  batches=$BATCHES  budget_basis=prefill_plus_gen"
    echo "runs=$RUNS warmup=$WARMUP cooldown=$COOLDOWN clock_lock=$CLOCK_LOCK stat=$STAT prime=$PRIME"
} > "$OUT_DIR/run.env"

echo "=== ours efficiency: $N_TOTAL configs x $N_CELLS cells = $(( N_TOTAL * N_CELLS )) ==="
sed 's/^/  /' "$OUT_DIR/run.env"
echo "  out: $OUT_DIR"
echo

# ---- the sweep --------------------------------------------------------------
i=0; fails=0; consecutive=0
for budget in $BUDGETS; do
  for ws in $WINDOW_SIZES; do
    i=$(( i + 1 ))
    slug="b${budget}_w${ws}"
    cell="$OUT_DIR/$slug"
    name="ours_${slug}"

    if [[ -f "$cell/table.txt" ]]; then
        echo "[$i/$N_TOTAL] $slug -- done already, skipping (rm $cell/table.txt to redo)"
        continue
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[$i/$N_TOTAL] $slug -> $cell   (config name: $name)"
        continue
    fi

    mkdir -p "$cell"
    GRID=""
    # Throwaway first cell: absorbs Triton JIT/autotune for this window_size so
    # it does not land inside the first measured row. Dropped by the summary.
    if [[ "$PRIME" == "1" ]]; then
        GRID+="    - { prefill_len: 256, gen_len: 33, batch_size: 1 }
"
    fi
    for s in $SHAPES; do
        p="${s%%/*}"; d="${s##*/}"
        for b in $BATCHES; do
            GRID+="    - { prefill_len: ${p}, gen_len: $(( d + 1 )), batch_size: ${b} }
"
        done
    done

    CONFIG_FILE="$cell/_efficiency_ours.generated.yaml"
    cat > "$CONFIG_FILE" <<YAML
# GENERATED by scripts/run_efficiency_ours.sh -- safe to delete.
# Self-contained: NO _base_, because \`_base_\` resolves relative to THIS file and
# it lives in the output dir, not in configs/.

run:
  mode: perf
  seed: 42

model:
  name: ${MODEL_PATH}
  revision: null
  dtype: ${DTYPE}

window:
  window_size: ${ws}
  num_sink_tokens: ${NUM_SINK}
  local_window_size: ${LOCAL_WINDOW}

cache:
  quant_ratio: ${QUANT_RATIO}
  quant_budget_mode: ${QUANT_MODE}
  # Written here AND on the arm below. A knob this script announces but does not
  # put in the config is a knob whose arms measure the same thing -- that is the
  # 6b8a188 bug (quant_gate_ratio inert in every runner) recurring in the sweep
  # that prices the gate.
  quant_gate_ratio: ${GATE_RATIO}
  first_eviction_step: 0

telemetry:
  track_scores: false          # scoring telemetry would distort the timings
  output_dir: ${cell}

perf:
  protocol: native
  prompt_mode: dataset
  data_source: "${DATA_SOURCE}"
  budget_basis: prefill_plus_gen   # see the header: 'context' would void every cell
  prefill_logits: last_only        # excludes the method-independent [B,L,V] tensor
  warmup_decode_steps: null
  num_warmup_runs: ${WARMUP}
  num_measurement_runs: ${RUNS}
  cooldown_s: ${COOLDOWN}
  enable_clock_locking: ${CLOCK_LOCK}
  clock_settle_s: 5.0
  allow_shared_gpu: true
  skip_if_oom: true
  skip_if_flash_attn_unavailable: true

  configs:
    - name: ${name}
      cache_backend: windowed
      cache_package: ${BACKEND}
      attn_implementation: ${ATTN_IMPL}
      cache_budget: ${budget}
      quant_ratio: ${QUANT_RATIO}
      quant_budget_mode: ${QUANT_MODE}
      quant_gate_ratio: ${GATE_RATIO}
      window_size: ${ws}
      num_sink_tokens: ${NUM_SINK}
      local_window_size: ${LOCAL_WINDOW}
      requires_flash_attn: ${NEEDS_FLASH}

  grid:
${GRID}
YAML

    # GEN_ONLY=1 writes every generated config and runs nothing -- so the YAML
    # can be loaded and checked (utils.config silently ignores unknown keys;
    # that is how quant_gate_ratio was inert in every runner at 6b8a188).
    if [[ "${GEN_ONLY:-0}" == "1" ]]; then
        echo "[$i/$N_TOTAL] $slug -- wrote $CONFIG_FILE (GEN_ONLY)"
        continue
    fi

    echo "[$i/$N_TOTAL] $slug  budget=$budget window=$ws  ($N_CELLS cells)  $(date +%H:%M:%S)"
    t0=$SECONDS
    python "$PROJECT_ROOT/main.py" --config "$CONFIG_FILE" > "$cell/run.log" 2>&1
    rc=$?
    dt=$(( SECONDS - t0 ))
    if (( rc != 0 )); then
        fails=$(( fails + 1 )); consecutive=$(( consecutive + 1 ))
        echo "    !! main.py exited $rc after ${dt}s -- $cell/run.log"
        tail -5 "$cell/run.log" | sed 's/^/       /'
        if (( consecutive >= 2 )); then
            echo "!! two configs in a row failed -- stopping rather than burning the"
            echo "   allocation on the same error 10 more times. Fix and re-run;"
            echo "   finished configs are skipped." >&2
            break 2
        fi
        continue
    fi
    consecutive=0

    python "$PROJECT_ROOT/scripts/print_perf_table.py" --npz-dir "$cell" \
        --config "$name" --stat "$STAT" --throughput all \
        --out "$cell/table.txt" > "$cell/print.log" 2>&1 \
        || { echo "    !! printing failed -- $cell/print.log"; fails=$(( fails + 1 )); continue; }
    # The gate verdict, in the log as well as the table: a run that did not gate
    # reads the whole int2 tier and finishes at a perfectly plausible TPOT.
    grep -E "read gate|never fired" "$cell/table.txt" | sed 's/^ */    /' || echo "    !! no gate line in table.txt"
    echo "    done in $(( dt / 60 ))m$(( dt % 60 ))s"
  done
done

# ---- summary ----------------------------------------------------------------
if [[ "$DRY_RUN" == "1" || "${GEN_ONLY:-0}" == "1" ]]; then
    echo
    echo "dry run: grid is valid, $N_TOTAL configs x $N_CELLS cells would run, nothing executed."
    exit 0
fi
echo
OUT_DIR="$OUT_DIR" BUDGETS="$BUDGETS" WINDOW_SIZES="$WINDOW_SIZES" \
SHAPES="$SHAPES" BATCHES="$BATCHES" FULLKV_DIR="$FULLKV_DIR" \
python - <<'PY' | tee "$OUT_DIR/summary.txt"
import os, re
E = os.environ
out = E["OUT_DIR"]
budgets = E["BUDGETS"].split()
wss = E["WINDOW_SIZES"].split()
shapes = [tuple(int(x) for x in s.split("/")) for s in E["SHAPES"].split()]
batches = [int(b) for b in E["BATCHES"].split()]

ROW = re.compile(r"\s*(\d+)/(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")

def read_table(path):
    """(prefill, decode, batch) -> (TPOT, dec tok/s, peak_GB) | 'OOM' | 'ERROR'."""
    got, gate = {}, None
    if not os.path.exists(path):
        return got, "!! NO TABLE (config did not run or did not finish)"
    for line in open(path):
        if "read gate" in line or "never fired" in line:
            gate = line.strip()
        m = ROW.match(line)
        if not m:
            continue
        pre, gen, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        key = (pre, gen - 1, b)
        marker = m.group(4)
        if marker in ("OOM", "ERROR"):
            got[key] = marker
        else:
            got[key] = (m.group(4), m.group(5), m.group(6), m.group(9))
    return got, gate

# marker columns: TTFT TPOT dec e2e legacy peak steadyKV -> with --throughput all
# group(4)=TTFT group(5)=TPOT group(6)=dec tok/s group(9)=peak_GB
data, gates = {}, {}
for bud in budgets:
    for ws in wss:
        slug = f"b{bud}_w{ws}"
        data[slug], gates[slug] = read_table(f"{out}/{slug}/table.txt")

print("Ours (two-tier int2 + rank-1 card read gate) efficiency, one GPU")
print(open(f"{out}/run.env").read().strip())
print()

for bud in budgets:
    print(f"--- cache_budget {bud} " + "-" * 52)
    hdr = f"{'prefill/decode':>15}{'batch':>6}"
    for ws in wss:
        hdr += f"{'ws'+ws+' TPOT':>12}{'ws'+ws+' tok/s':>13}"
    print(hdr)
    for pre, dec in shapes:
        for b in batches:
            line = f"{f'{pre}/{dec}':>15}{b:6d}"
            for ws in wss:
                v = data[f"b{bud}_w{ws}"].get((pre, dec, b), "missing")
                if v == "missing":
                    line += f"{'-':>12}{'-':>13}"
                elif isinstance(v, str):
                    line += f"{v:>12}{'':>13}"
                else:
                    line += f"{v[1]:>12}{v[2]:>13}"
            print(line)
    print()

print("TPOT = seconds per generated token at steady state; tok/s = batch / TPOT.")
print("Peak memory per cell is in each config's table.txt (peak_GB, steadyKV_GB).")

print()
print("Read-gate verdict per config -- a run that did NOT gate reads the whole")
print("int2 tier, is correct, and finishes at a plausible TPOT. No verdict here,")
print("or a '!!' one, means the numbers above it are not gated numbers.")
for bud in budgets:
    for ws in wss:
        slug = f"b{bud}_w{ws}"
        g = gates[slug]
        if g is None:
            g = "!! NO GATE LINE (config did not finish, or no npz)"
        print(f"  {slug:>14}: {g.replace('read gate: ', '')}")

# --- optional: ours vs the full-KV baseline ---------------------------------
fk = E.get("FULLKV_DIR", "")
if fk:
    fk_tbl = f"{fk}/table_fullkv_flash.txt"
    base, _ = read_table(fk_tbl)
    if not base:
        print(f"\n(no full-KV rows read from {fk_tbl})")
    else:
        print("\nSpeedup vs full-KV flash (TPOT_fullkv / TPOT_ours), cells both ran:")
        for bud in budgets:
            for ws in wss:
                slug = f"b{bud}_w{ws}"
                rs = []
                for k, v in data[slug].items():
                    bv = base.get(k)
                    if isinstance(v, tuple) and isinstance(bv, tuple):
                        try:
                            rs.append(float(bv[1]) / float(v[1]))
                        except (ValueError, ZeroDivisionError):
                            pass
                if rs:
                    print(f"  {slug:>14}: {sum(rs)/len(rs):5.2f}x  (over {len(rs)} cells)")

missing = [(s, k) for s in data for k in
           [(p, d, b) for p, d in shapes for b in batches] if k not in data[s]]
if missing:
    print(f"\n!! {len(missing)} cell(s) produced no row. Configs with none at all "
          f"did not run; re-running the script resumes them.")
PY

echo
echo "per-config tables: $OUT_DIR/<b*_w*>/table.txt"
echo "summary:           $OUT_DIR/summary.txt"
(( fails == 0 )) || { echo "!! $fails config(s) failed -- see their run.log" >&2; exit 1; }
