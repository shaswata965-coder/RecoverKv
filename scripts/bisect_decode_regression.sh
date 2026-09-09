#!/usr/bin/env bash
# scripts/bisect_decode_regression.sh
#
# ============================================================================
# One command that settles the ~5 ms/step decode regression.
# ============================================================================
# Runs the perf table three times into THREE FRESH DIRECTORIES -- once as
# shipped, once with each control arm off -- then profiles the shipped build,
# then prints a side-by-side verdict.
#
#   bash scripts/bisect_decode_regression.sh --model /path/to/llama-3.1-8b
#
# Why fresh directories: print_perf_table globs its --npz-dir for
# `perf_prefill*_gen*_bs*.npz` and each cell is its OWN FILE keyed by shape. A
# re-run only overwrites the shapes it ran, so a result for a shape no longer in
# the grid survives forever and lands in the table as a stale row. One directory
# per arm makes that impossible rather than merely unlikely.
#
# What is being bisected. Two changes landed together and the table went from
# 0.0626 to ~0.0669 at 4096/B=32 -- a flat +4.3..5.3 ms across a 32x batch range
# and a 4x context range, i.e. a fixed per-step cost that tracks neither traffic
# nor n_active. Neither change has a mechanism that predicts a constant of that
# size, so it is measured rather than argued:
#
#   baseline    as shipped
#   noexp2      STICKYKV_DECODE_EXP2=0        base-e softmax (the old kernel)
#   nordtype    STICKYKV_ROPE_STORE_DTYPE=0   fp32 RoPE tables (the old dtype)
#
# The decision is pre-committed, so the result is not argued about afterwards:
#
#   * exp2 owns it      -> DROP exp2. Worth ~0.42 ms in theory, buying nothing.
#   * RoPE dtype owns it -> KEEP IT ANYWAY. It is the accuracy fix that closed a
#     9.5-point qasper regression, and scores outrank 5 ms. Record the price.
#   * neither            -> the cause is elsewhere; the profile printed at the
#     end (GPU busy %, chosen tile rung) is the next instrument.
#
# Default grid is the two cells that matter -- the regression is flat, so one
# context length settles it and the full sweep is a waste of GPU hours. Override
# with SHAPES/BATCHES for the complete table.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-}"
SHAPES="${SHAPES:-4096/256}"
BATCHES="${BATCHES:-1 32}"
ROOT="${ROOT:-outputs/bisect_$(date +%m%d_%H%M%S)}"
SKIP_PROFILE="${SKIP_PROFILE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model-path) MODEL_PATH="$2"; shift 2;;
    --shapes)             SHAPES="$2"; shift 2;;
    --batches)            BATCHES="$2"; shift 2;;
    --out|--root)         ROOT="$2"; shift 2;;
    --skip-profile)       SKIP_PROFILE=1; shift;;
    -h|--help)            sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "error: set MODEL_PATH or pass --model /path/to/llama-3.1-8b-instruct" >&2
  exit 2
fi

# One GPU. device_map="auto" shards by layer, and a pipeline-sharded decode step
# times the interconnect rather than the cache; run_perf_table refuses it anyway,
# but failing here costs a second instead of a model load.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "error: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES selects multiple GPUs." >&2
  echo "       Re-run with CUDA_VISIBLE_DEVICES=0 -- this box's earlier sweeps" >&2
  echo "       were lost to exactly this." >&2
  exit 2
fi

PY="${PY:-python}"
mkdir -p "$ROOT"
echo "bisect root: $ROOT"
echo "grid:        SHAPES='$SHAPES'  BATCHES='$BATCHES'"
echo

run_arm() {   # $1 = arm name, $2.. = env assignments
  local arm="$1"; shift
  local dir="$ROOT/$arm"
  echo "=============================================================="
  echo "  arm: $arm    ${*:-<as shipped>}"
  echo "=============================================================="
  mkdir -p "$dir"
  # `env` scopes the control arm to this arm only -- no leakage between runs.
  if env "$@" MODEL_PATH="$MODEL_PATH" OUT_DIR="$PROJECT_ROOT/$dir" \
         SHAPES="$SHAPES" BATCHES="$BATCHES" \
         bash scripts/run_perf_table.sh > "$dir/run.log" 2>&1; then
    cp "$dir/table.txt" "$ROOT/table_$arm.txt" 2>/dev/null || true
    echo "  ok -> $dir/table.txt"
  else
    echo "  FAILED -- tail of $dir/run.log:" >&2
    tail -20 "$dir/run.log" >&2
  fi
  echo
}

run_arm baseline
run_arm noexp2   STICKYKV_DECODE_EXP2=0
run_arm nordtype STICKYKV_ROPE_STORE_DTYPE=0

# ---------------------------------------------------------------------------
# The profile: GPU busy % and the chosen tile rung, on the shipped build.
# ---------------------------------------------------------------------------
if [[ "$SKIP_PROFILE" == "0" ]]; then
  CFG="$ROOT/baseline/_perf_table.generated.yaml"
  if [[ -f "$CFG" ]]; then
    echo "=============================================================="
    echo "  Stage 0 profile (shipped build)"
    echo "=============================================================="
    for B in 32 1; do
      $PY scripts/profile_decode.py --config "$CFG" \
          --prefill 4096 --batch "$B" --steps 24 --warmup 12 \
          > "$ROOT/profile_bs$B.log" 2>&1 || true
      echo "--- batch=$B ---"
      grep -E "GPU busy|wall |CUDA kernels|kernel launches|HOST-BOUND|KERNEL-BOUND|MIXED" \
           "$ROOT/profile_bs$B.log" || echo "  (see $ROOT/profile_bs$B.log)"
    done
    echo
    echo "tile rung chosen by the shared-memory ladder:"
    grep -h "fused decode tiling" "$ROOT"/*/run.log "$ROOT"/profile_bs*.log 2>/dev/null \
      | sort -u | sed 's/^/  /' || echo "  (not announced -- kernel may not have run)"
    echo
  fi
fi

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
$PY - "$ROOT" <<'PYEOF'
import re, sys
from pathlib import Path

root = Path(sys.argv[1])
arms = ["baseline", "noexp2", "nordtype"]
data = {}
for arm in arms:
    f = root / f"table_{arm}.txt"
    if not f.exists():
        continue
    for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*(\d+/\d+)\s+(\d+)\s+\S+\s+([\d.]+)\s", ln)
        if m:
            data.setdefault((m.group(1), int(m.group(2))), {})[arm] = float(m.group(3))

if not data:
    print("no cells parsed -- check the run.log files under", root)
    raise SystemExit(0)

print("=" * 74)
print("  VERDICT   TPOT_steady (s), by arm")
print("=" * 74)
hdr = f"{'cell':>14}" + "".join(f"{a:>12}" for a in arms) + f"{'owns it?':>22}"
print(hdr); print("-" * len(hdr))
votes = []
for cell in sorted(data):
    row = data[cell]
    line = f"{cell[0]+' b'+str(cell[1]):>14}" + "".join(
        f"{row.get(a, float('nan')):>12.4f}" for a in arms)
    base = row.get("baseline")
    verdict = ""
    if base:
        gains = {a: base - row[a] for a in ("noexp2", "nordtype") if a in row}
        # "owns it" = turning that change OFF recovers >2 ms
        best = max(gains, key=gains.get) if gains else None
        if best and gains[best] > 0.002:
            verdict = f"{best} (-{gains[best]*1000:.1f} ms)"
            votes.append(best)
        elif gains:
            verdict = "neither (<2 ms)"
            votes.append("neither")
    print(line + f"{verdict:>22}")

print()
if votes and all(v == votes[0] for v in votes):
    v = votes[0]
    if v == "noexp2":
        print("-> exp2 owns the regression. DROP IT: worth ~0.42 ms in theory and")
        print("   evidently negative in practice. Revert the base-2 softmax.")
    elif v == "nordtype":
        print("-> the RoPE dtype owns the regression. KEEP IT ANYWAY -- it is the")
        print("   accuracy fix that closed the qasper regression (42.43 vs 32.78),")
        print("   and scores outrank 5 ms/step. Record the cost in the plan and")
        print("   note the table still clears both targets.")
    else:
        print("-> NEITHER change owns it. The cause is outside the two suspects;")
        print("   read the GPU busy % above and go to Stage 0 rather than guessing.")
else:
    print("-> arms disagree across cells. Trust the largest-batch cell, and if it")
    print("   is still ambiguous the effect is below this grid's noise -- raise")
    print("   RUNS before drawing a conclusion.")
print()
print(f"artefacts: {root}")
PYEOF
