#!/usr/bin/env bash
set -uo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# LongBench knob sweep on 4 datasets -- qmsum, narrativeqa, triviaqa, musique.
# Llama-3.1-8B-Instruct, cache_budget 0.20 throughout, sink 5, local 128.
#
#   combo 1  window_size  4 / 16 / 32                 (q 0.70, gate 0.25)
#   combo 2  quant_ratio  0.10 / 0.30 / 0.50 / 0.90   (window 8, gate 0.25)
#   combo 3  gate_ratio   0.10 / 0.50 / 0.75          (window 8, q 0.70)
#
#   10 configs x 4 datasets = 40 jobs, 200 examples each.
#
# Each combo moves ONE knob off the pinned operating point (q 0.70, gate 0.25,
# window 8) and holds the rest there, so every column reads against the same
# default -- which is NOT re-run here, because it exists at a comparable commit
# and the summary prints it as the `default` column:
#   outputs/ruler32k_lb16_bda91b4/longbench   (q0.70 g0.25 w8 l128 b0.20)
#     qmsum 25.78  narrativeqa 29.92  triviaqa 91.04  musique 32.64
# bda91b4 -> HEAD (2f06bc4) adds scripts only, no module changed, so those rows
# compare. BASELINE_DIR= drops the column.
#
# ----------------------------------------------------- keeping 8 GPUs busy
# ONE queue, 8 workers, jobs ordered GLOBALLY LONGEST-FIRST: all ten qmsum jobs,
# then narrativeqa, then triviaqa, then musique. That is online LPT, and the
# ordering is worth real time here. Simulated against the measured per-dataset
# wall times (2026-09-15 sidecars: qmsum 2182s, narrativeqa 1814s, triviaqa
# 1118s, musique 844s) plus 90 s of model load per job, on 8 GPUs:
#
#   total work 17.55 GPU-h, perfect 8-way split 2.19 h
#   config-major (finish whole configs first)  2.41 h, 1.75 GPU-h idle, 90.9%
#   global LPT (this script)                   2.28 h, 0.72 GPU-h idle, 96.1%
#
# The cost of LPT is that no config is complete until near the end, so a run
# killed early leaves every column partial rather than some finished. The
# script resumes per job, so that costs re-queueing, not re-running.
# Ask for 5 h: window 4 and q 0.90 carry more windows than the times above.
#
# ------------------------------------------------------------------- read this
# * A knob that does not move looks exactly like a knob that changed nothing,
#   and quant_gate_ratio was inert once (6b8a188). The summary prints, per
#   config, the REALISED read fraction against the gate asked for, and the
#   resolved tier split (fp vs int2 windows) which is what quant_ratio moves.
#   Read that block before the scores.
# * At budget 0.20 with local 128 a prompt below ~665 tokens clamps to
#   sink+local with 0 evictable windows (config.py:532). Across the 16-dataset
#   set that was ~10/2350 examples, all multi_news; none of these four is short
#   enough for it to bite.
# * window 4 is the smallest legal value at q>0 (int2 packs 4 codes per byte
#   along the window axis) and local 128 is a multiple of every window here.
#   Both checked up front, not 30 jobs in.
#
# ------------------------------------------------------------------- resuming
# DONE = <dataset>.jsonl non-empty AND <dataset>.meta.json present. Re-run to
# redo only what is missing; nothing is deleted, because the runner truncates
# its own output file and this passes longbench.resume=false. A job refuses to
# start if HEAD moved: ten columns must come from one decoder.
#
# Outputs:
#   $OUT_ROOT/q<q>_g<gate>_w<ws>/<dataset>.{jsonl,meta.json,log}, scores.csv
#   $OUT_ROOT/summary.txt
#
# Run:
#   cd ~/kv_cache/Cluster_QEvict/RecoverKv && conda activate sticky_env
#   nohup scripts/run_longbench4_combos_8gpu.sh > ~/lb4_combos.log 2>&1 &
#   tail -f ~/lb4_combos.log
#
#   DRY_RUN=1 scripts/run_longbench4_combos_8gpu.sh    # plan + makespan, runs nothing
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 2

MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez228470/llama-3.1-8b-instruct}"
export LONGBENCH_LOCAL_DIR="${LONGBENCH_LOCAL_DIR:-/home/ee/phd/eez228470/kv_cache/qwen_longbench_final_int2/RecoverKv/data/longbench}"
CONFIG="${CONFIG:-configs/longbench_ours_flash_attn.yaml}"

# q:gate:window
read -r -a CONFIGS <<< "${CONFIGS:-0.70:0.25:4 0.70:0.25:16 0.70:0.25:32 \
0.10:0.25:8 0.30:0.25:8 0.50:0.25:8 0.90:0.25:8 \
0.70:0.10:8 0.70:0.50:8 0.70:0.75:8}"
# Longest first -- this order IS the scheduling decision (see the header).
read -r -a DATASETS <<< "${DATASETS:-qmsum narrativeqa triviaqa musique}"

# Pinned by the request -- not env-overridable, so no column can drift.
BUDGET=0.20; QUANT_MODE=bytes; LOCAL=128; SINK=5
DEF_Q=0.70; DEF_G=0.25; DEF_W=8        # the point each combo varies one knob from

COMMIT="$(git rev-parse --short HEAD)"; START_SHA="$(git rev-parse HEAD)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/lb4_combos_${COMMIT}}"
BASELINE_DIR="${BASELINE_DIR-$PROJECT_ROOT/outputs/ruler32k_lb16_bda91b4/longbench}"
DRY_RUN="${DRY_RUN:-0}"

cfg_dir() { printf '%s/q%s_g%s_w%s' "$OUT_ROOT" "$1" "$2" "$3"; }
die() { echo "error: $*" >&2; exit 2; }

# ---- preflight ---------------------------------------------------------------
[[ -f "$CONFIG" ]]     || die "config not found: $CONFIG"
[[ -e "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
(( ${#CONFIGS[@]} > 0 ))  || die "CONFIGS is empty"
(( ${#DATASETS[@]} > 0 )) || die "DATASETS is empty"
# A missing jsonl does not error: the loader silently falls back to the
# HuggingFace script dataset, which datasets>=3 refuses to run.
for d in "${DATASETS[@]}"; do
    [[ -s "$LONGBENCH_LOCAL_DIR/$d.jsonl" ]] || die "missing $LONGBENCH_LOCAL_DIR/$d.jsonl"
done
for c in "${CONFIGS[@]}"; do
    IFS=':' read -r q g w <<< "$c"
    [[ -n "$q" && -n "$g" && -n "$w" ]] || die "CONFIGS items are q:gate:window, got '$c'"
    [[ "$w" =~ ^[0-9]+$ ]] || die "window must be an integer, got '$w' in '$c'"
    awk -v q="$q" -v g="$g" 'BEGIN{
        if (q+0 <= 0 || q+0 > 1) { print "quant_ratio must be in (0,1], got " q > "/dev/stderr"; exit 2 }
        if (g+0 <= 0 || g+0 > 1) { print "quant_gate_ratio must be in (0,1], got " g > "/dev/stderr"; exit 2 }
    }' || die "bad config '$c'"
    (( w % 4 == 0 ))     || die "window $w is not a multiple of 4, and q=$q > 0 (int2 crumb packing)"
    (( LOCAL % w == 0 )) || die "local $LOCAL is not a multiple of window $w (config '$c')"
done
dupes=$(printf '%s\n' "${CONFIGS[@]}" | while IFS=':' read -r q g w; do
            awk -v q="$q" -v g="$g" -v w="$w" 'BEGIN{printf "%.4f:%.4f:%d\n", q, g, w}'; done | sort | uniq -d)
[[ -z "$dupes" ]] || die "CONFIGS contains the same (q,gate,window) twice: $dupes"

if [[ "$DRY_RUN" != "1" ]]; then
    command -v nvidia-smi >/dev/null || die "no nvidia-smi -- login node? get a GPU node."
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
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"   # PBS UUIDs, used as given
elif command -v nvidia-smi >/dev/null; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
    GPUS=(0)
fi
(( ${#GPUS[@]} > 0 )) || die "no GPUs visible"

# ---- the queue: global longest-first (dataset-major) --------------------------
JOBS=()
for d in "${DATASETS[@]}"; do
    for c in "${CONFIGS[@]}"; do
        IFS=':' read -r q g w <<< "$c"
        JOBS+=("$q $g $w $d")
    done
done

echo "=== LongBench x${#DATASETS[@]} knob sweep @ $COMMIT: ${#JOBS[@]} jobs on ${#GPUS[@]} GPU(s) ==="
echo "  fixed:    budget=$BUDGET  mode=$QUANT_MODE  local=$LOCAL  sink=$SINK"
echo "  default:  q=$DEF_Q gate=$DEF_G window=$DEF_W  (each combo moves one knob off this)"
echo "  order:    global longest-first -- ${DATASETS[*]}"
todo=0
for c in "${CONFIGS[@]}"; do
    IFS=':' read -r q g w <<< "$c"; d="$(cfg_dir "$q" "$g" "$w")"; n=0
    for x in "${DATASETS[@]}"; do
        [[ -s "$d/$x.jsonl" && -f "$d/$x.meta.json" ]] && n=$(( n + 1 ))
    done
    todo=$(( todo + ${#DATASETS[@]} - n ))
    printf "    q=%-4s gate=%-4s window=%-2s  %d/%d done\n" "$q" "$g" "$w" "$n" "${#DATASETS[@]}"
done
echo "  to run:   $todo job(s)"
# Schedule what is ACTUALLY left, in queue order, so the estimate reflects a
# resumed run rather than a fresh one.
NGPU=${#GPUS[@]} TODO_LIST="$(for d in "${DATASETS[@]}"; do for c in "${CONFIGS[@]}"; do
        IFS=':' read -r q g w <<< "$c"; dir="$(cfg_dir "$q" "$g" "$w")"
        [[ -s "$dir/$d.jsonl" && -f "$dir/$d.meta.json" ]] || printf '%s ' "$d"; done; done)" \
python - <<'PY'
import os
T = {"qmsum": 2182, "narrativeqa": 1814, "triviaqa": 1118, "musique": 844}
LOAD, m = 90, int(os.environ["NGPU"])
jobs = [T.get(d, 1500) + LOAD for d in os.environ["TODO_LIST"].split()]
if not jobs or m < 1:
    raise SystemExit
end = [0.0] * m
for j in jobs:                      # the queue's own order, greedily assigned
    i = min(range(m), key=lambda k: end[k]); end[i] += j
ms, tot = max(end), sum(jobs)
print(f"  estimate: makespan {ms/3600:.2f} h | {tot/3600:.2f} GPU-h of work | "
      f"utilisation {100*tot/(m*ms):.1f}% | floor {tot/m/3600:.2f} h")
print( "            (measured 2026-09-15 dataset times + 90 s load; window 4 and")
print( "             q 0.90 are slower than that, so treat it as a lower bound)")
PY
echo "  out:      $OUT_ROOT"
[[ -d "$BASELINE_DIR" ]] && echo "  default column: $BASELINE_DIR"
echo

if [[ "$DRY_RUN" == "1" ]]; then echo "dry run: grid is valid, nothing started."; exit 0; fi

mkdir -p "$OUT_ROOT"
for c in "${CONFIGS[@]}"; do IFS=':' read -r q g w <<< "$c"; mkdir -p "$(cfg_dir "$q" "$g" "$w")"; done
QUEUE_POS="$OUT_ROOT/.queue_pos"; LOCK="$OUT_ROOT/.queue_lock"; FAILED="$OUT_ROOT/failed_jobs.txt"
exec 200>"$OUT_ROOT/.runner_lock"
flock -n 200 || die "another copy of this script is already running against $OUT_ROOT"
echo 0 > "$QUEUE_POS"; : > "$FAILED"

{
    echo "commit=$START_SHA  started=$(date -Is)  host=$(hostname)  gpus=${GPUS[*]}"
    echo "config=$CONFIG  model=$MODEL_PATH  longbench_local_dir=$LONGBENCH_LOCAL_DIR"
    echo "budget=$BUDGET mode=$QUANT_MODE local=$LOCAL sink=$SINK"
    echo "configs(q:gate:window)=${CONFIGS[*]}"
    echo "datasets=${DATASETS[*]}"
} > "$OUT_ROOT/run.env"

next_job() {
    local i
    {
        flock 9
        i=$(<"$QUEUE_POS")
        if (( i < ${#JOBS[@]} )); then echo $((i + 1)) > "$QUEUE_POS"; echo "$i"; fi
    } 9>"$LOCK"
}

run_job() {    # $1 gpu, $2 q, $3 gate, $4 window, $5 dataset
    local gpu="$1" q="$2" g="$3" w="$4" x="$5" dir
    dir="$(cfg_dir "$q" "$g" "$w")"
    if [[ -s "$dir/$x.jsonl" && -f "$dir/$x.meta.json" ]]; then
        echo "[$(date +%T)] gpu=$gpu SKIP   q=$q g=$g w=$w $x (complete)"; return 0
    fi
    if [[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" != "$START_SHA" ]]; then
        echo "[$(date +%T)] gpu=$gpu REFUSE q=$q g=$g w=$w $x -- HEAD moved off $COMMIT" >&2
        echo "q=$q g=$g w=$w $x rc=HEAD-moved" >> "$FAILED"; return 1
    fi
    echo "[$(date +%T)] gpu=$gpu START  q=$q g=$g w=$w $x"
    local t0=$SECONDS rc
    # LongBench builds its cache from cfg.cache.* only (longbench_runner.py:488);
    # window.* is never consulted there. resume=false is passed explicitly so a
    # partial .jsonl from a killed job is overwritten (the runner opens "w"),
    # never treated as finished.
    CUDA_VISIBLE_DEVICES="$gpu" python "$PROJECT_ROOT/main.py" --config "$CONFIG" --override \
        model.name="$MODEL_PATH" \
        cache.cache_budget="$BUDGET" \
        cache.quant_ratio="$q" \
        cache.quant_budget_mode="$QUANT_MODE" \
        cache.quant_gate_ratio="$g" \
        cache.window_size="$w" \
        cache.local_window_size="$LOCAL" \
        cache.num_sink_tokens="$SINK" \
        cache.first_eviction_step=0 \
        longbench.datasets="[$x]" \
        longbench.resume=false \
        longbench.output_dir="$dir" \
        > "$dir/$x.log" 2>&1
    rc=$?
    if (( rc == 0 )) && [[ -s "$dir/$x.jsonl" && -f "$dir/$x.meta.json" ]]; then
        echo "[$(date +%T)] gpu=$gpu DONE   q=$q g=$g w=$w $x ($(( (SECONDS - t0) / 60 )) min)"
    else
        local why
        why=$(grep -m1 -oE "ran EAGER|OutOfMemoryError|CUDA error[^.]*|[A-Za-z]+Error: [^.]{0,80}" "$dir/$x.log" | head -1)
        echo "[$(date +%T)] gpu=$gpu FAIL   q=$q g=$g w=$w $x rc=$rc ${why:+($why)} -- $dir/$x.log" >&2
        echo "q=$q g=$g w=$w $x rc=$rc ${why:-see log} $dir/$x.log" >> "$FAILED"
    fi
}

worker() {
    local gpu="$1" i
    while i=$(next_job) && [[ -n "$i" ]]; do
        # shellcheck disable=SC2086
        run_job "$gpu" ${JOBS[$i]}
    done
}

pids=()
for g in "${GPUS[@]}"; do
    worker "$g" 2>&1 | tee -a "$OUT_ROOT/worker_${g}.log" &
    pids+=($!)
done
wait "${pids[@]}"

# ---- score + summary ----------------------------------------------------------
echo; echo "=== scoring ==="
for c in "${CONFIGS[@]}"; do
    IFS=':' read -r q g w <<< "$c"; d="$(cfg_dir "$q" "$g" "$w")"
    python -m modules.evaluation.longbench_scoring --predictions_dir "$d" --out_csv "$d/scores.csv" \
        > "$d/scoring.log" 2>&1 || echo "scoring FAILED for $c -- $d/scoring.log" >&2
done

CFG_DIRS="$(for c in "${CONFIGS[@]}"; do IFS=':' read -r q g w <<< "$c"; printf '%s=%s ' "$c" "$(cfg_dir "$q" "$g" "$w")"; done)" \
COMMIT="$COMMIT" DATASETS="${DATASETS[*]}" BASELINE_DIR="$BASELINE_DIR" \
BUDGET="$BUDGET" LOCAL="$LOCAL" SINK="$SINK" DEF_Q="$DEF_Q" DEF_G="$DEF_G" DEF_W="$DEF_W" \
python - <<'PY' | tee "$OUT_ROOT/summary.txt"
import csv, json, os
E = os.environ
ds = E["DATASETS"].split()
cols = [tuple(x.split("=", 1)) for x in E["CFG_DIRS"].split()]
DQ, DG, DW = float(E["DEF_Q"]), float(E["DEF_G"]), int(E["DEF_W"])

def load(d):
    raw = {}
    if os.path.exists(f"{d}/scores.csv"):
        raw = {r["dataset"]: float(r["score"]) for r in csv.DictReader(open(f"{d}/scores.csv"))}
    out, fr, verd, geo = {}, [], set(), None
    for x in ds:
        j, m = f"{d}/{x}.jsonl", f"{d}/{x}.meta.json"
        n = sum(1 for _ in open(j)) if os.path.exists(j) else 0
        if os.path.exists(m):
            meta = json.load(open(m))
            g = meta.get("read_gate") or {}
            if g.get("read_fraction") is not None:
                fr.append(float(g["read_fraction"]))
            verd.add(g.get("verdict", "MISSING"))
            if geo is None:
                geo = meta.get("resolved_geometry_first_example") or {}
        out[x] = raw.get(x) if (x in raw and n) else None
    return out, (sum(fr) / len(fr) if fr else None), verd, (geo or {})

res = {c: load(d) for c, d in cols}
base = {}
if E["BASELINE_DIR"] and os.path.exists(f"{E['BASELINE_DIR']}/scores.csv"):
    base = {r["dataset"]: float(r["score"]) for r in csv.DictReader(open(f"{E['BASELINE_DIR']}/scores.csv"))}

print(f"LongBench knob sweep @ {E['COMMIT']}")
print(f"budget {E['BUDGET']}  local {E['LOCAL']}  sink {E['SINK']}  mode bytes  "
      f"| {len(ds)} datasets x 200 examples")
print(f"default point = q{DQ:.2f} g{DG:.2f} w{DW}, from {E['BASELINE_DIR'] or '(none)'}\n")

# Which knob a config varies decides its heading -- derived, not hardcoded, so
# editing CONFIGS cannot file a column under the wrong combo.
def which(c):
    q, g, w = float(c.split(":")[0]), float(c.split(":")[1]), int(c.split(":")[2])
    diff = [n for n, a, b in (("window_size", w, DW), ("quant_ratio", q, DQ),
                              ("gate_ratio", g, DG)) if a != b]
    return diff[0] if len(diff) == 1 else ("default point" if not diff else " + ".join(diff))

def label(c, knob):
    q, g, w = c.split(":")
    return {"window_size": f"w{w}", "quant_ratio": f"q{q}", "gate_ratio": f"g{g}"}.get(knob, c)

order, groups = [], {}
for c, _ in cols:
    k = which(c); groups.setdefault(k, []).append(c)
    if k not in order: order.append(k)

for k in order:
    members = groups[k]
    print(f"--- varying {k} " + "-" * max(0, 56 - len(k)))
    print(f"{'dataset':16}{'default':>10}" + "".join(f"{label(c,k):>10}" for c in members))
    for x in ds:
        row = f"{x:16}" + (f"{base[x]:10.2f}" if x in base else f"{'-':>10}")
        for c in members:
            v = res[c][0][x]
            row += f"{v:10.2f}" if v is not None else f"{'-':>10}"
        print(row)
    bv = [base.get(x) for x in ds]
    row = f"{'average':16}" + (f"{sum(bv)/len(bv):10.2f}" if None not in bv else f"{'-':>10}")
    for c in members:
        v = [res[c][0][x] for x in ds]
        row += f"{sum(v)/len(v):10.2f}" if None not in v else f"{'incompl.':>10}"
    print(row + "\n")

print("Did the knobs move? (from each config's own sidecars)")
print(f"  {'config':>18}{'gate asked':>11}{'realised':>10}{'fp win':>8}{'int2 win':>9}   verdict")
bad = []
for c, _ in cols:
    q, g, w = c.split(":")
    _, frac, verd, geo = res[c]
    vd = ",".join(sorted(verd)) if verd else "-"
    fpw, nq = geo.get("top_k_fp"), geo.get("N_q")
    fs = f"{frac:10.3f}" if frac is not None else f"{'-':>10}"
    flag = ""
    if frac is not None and abs(frac - float(g)) > 0.05:
        flag = "   <-- GATE DID NOT MOVE"; bad.append(f"{c}: gate asked {g}, realised {frac:.3f}")
    if vd not in ("gated", "-"):
        bad.append(f"{c}: verdict {vd}")
    print(f"  {'q'+q+' g'+g+' w'+w:>18}{g:>11}{fs}"
          f"{(fpw if fpw is not None else '-'):>8}{(nq if nq is not None else '-'):>9}   {vd}{flag}")
if bad:
    print("\n!! these columns are not what they say they are:")
    for b in bad: print("   ", b)
else:
    print("\n  every config realised its gate ratio, verdict `gated` throughout.")
print("  fp win / int2 win: the first example's resolved tier split -- what")
print("  quant_ratio actually moves. Equal across the quant_ratio group = q inert.")

missing = [f"{c} {x}" for c, _ in cols for x in ds if res[c][0][x] is None]
if missing:
    print(f"\n!! {len(missing)} job(s) with no score (re-run to redo just these):")
    for m in missing[:20]: print("   ", m)
PY

if [[ -s "$FAILED" ]]; then echo; echo "!! failed jobs:"; cat "$FAILED"; exit 1; fi
echo; echo "done: $OUT_ROOT/summary.txt"
