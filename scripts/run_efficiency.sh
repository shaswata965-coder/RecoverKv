#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Efficiency benchmark driver (Suite C)
# ============================================================================
# Runs Suite C under the comparison protocol and prints the result in the
# published-table format. Shapes are sweepable from here, so exploring a grid
# does not mean editing YAML.
#
# Usage:
#   scripts/run_efficiency.sh [options] [-- extra args for main.py]
#
#   --prefill LIST   prefill (context) lengths      e.g. 4096,8192,16384
#   --decode  LIST   DECODE STEPS, not gen_len      e.g. 100,1024
#   --batch   LIST   batch sizes                    e.g. 1,4,16
#   --window  LIST   window sizes                   e.g. 8,32
#   --dataset SRC    prefill from real text instead of the synthetic prompt.
#                    SRC is usually A PATH -- a file or a directory, any suffix
#                    (.txt .md .jsonl .json, or none). Also accepts wikitext-103,
#                    pg19, or a LongBench name.
#   --config  PATH   config to run   (default configs/eval_efficiency.yaml)
#   --out     DIR    output root     (default outputs/efficiency)
#   --                everything after this is passed through to main.py
#
# Examples:
#   scripts/run_efficiency.sh                        # the protocol's own grid
#   scripts/run_efficiency.sh --prefill 4096,8192 --decode 128,1024 --batch 1,4
#   scripts/run_efficiency.sh --window 8,32 --prefill 8192 --decode 100
#   scripts/run_efficiency.sh --dataset ./mycorpus/ --prefill 4096 --decode 256
#   scripts/run_efficiency.sh --dataset ./notes.md  --prefill 8192 --batch 1,4
#   scripts/run_efficiency.sh --dataset wikitext-103 --prefill 4096 --batch 1,4
#   scripts/run_efficiency.sh -- --override perf.cooldown_s=5
#
# Each batch row gets exactly --prefill tokens cut from that text, and generation
# is exactly --decode steps regardless of how long the corpus is: the corpus
# supplies the prompt, the flags fix the shape.
#
# NOTE ON --dataset. The comparison protocol prefills a tiled synthetic sentence,
# so a real corpus is a DEVIATION from it: use it to characterise your own method
# on realistic text, not to produce a number you place beside a published table.
# The npz records prompt_mode and data_source either way, so a run always says
# which it was.
#
# NOTE ON --decode. It is DECODE STEPS, which is what the protocol and every
# published table count. The runner's `gen_len` is decode steps + 1 (the prefill
# emits the first token), and this script does that conversion for you -- passing
# gen_len here by hand is the classic off-by-one in this comparison.
#
# NOTE ON --window. window_size is config-level, not per-cell, so each value is a
# separate invocation writing to its own subdirectory (outputs/.../ws8, ws32).
# Without that they would overwrite each other: npz filenames key on
# prefill/gen/batch, not on window size.
#
# Anything not swept keeps whatever the config says.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="$PROJECT_ROOT/configs/eval_efficiency.yaml"
OUT_ROOT="outputs/efficiency"
PREFILL=""; DECODE=""; BATCH=""; WINDOW=""; DATASET=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefill) PREFILL="$2"; shift 2 ;;
        --decode)  DECODE="$2";  shift 2 ;;
        --batch)   BATCH="$2";   shift 2 ;;
        --window)  WINDOW="$2";  shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --config)  CONFIG="$2";  shift 2 ;;
        --out)     OUT_ROOT="$2"; shift 2 ;;
        --)        shift; PASSTHROUGH=("$@"); break ;;
        -h|--help) sed -n '9,50p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

# --- shape overrides ---------------------------------------------------------
# Any shape flag switches the runner off its explicit `grid` and onto the
# cartesian product of the three sweep axes.
SHAPE_OVERRIDES=()
if [[ -n "$PREFILL$DECODE$BATCH" ]]; then
    SHAPE_OVERRIDES+=("perf.grid=[]")
    [[ -n "$PREFILL" ]] && SHAPE_OVERRIDES+=("perf.prefill_lengths=[$PREFILL]")
    [[ -n "$BATCH"   ]] && SHAPE_OVERRIDES+=("perf.batch_sizes=[$BATCH]")
    if [[ -n "$DECODE" ]]; then
        # decode steps -> gen_len (= steps + 1)
        GENS=""
        IFS=',' read -ra STEPS <<< "$DECODE"
        for s in "${STEPS[@]}"; do
            s="$(echo "$s" | tr -d '[:space:]')"
            [[ "$s" =~ ^[0-9]+$ ]] || { echo "--decode expects integers, got '$s'" >&2; exit 2; }
            GENS+="$(( s + 1 )),"
        done
        SHAPE_OVERRIDES+=("perf.gen_lengths=[${GENS%,}]")
    fi
fi

# A real corpus replaces the synthetic prompt for every row in the run.
if [[ -n "$DATASET" ]]; then
    SHAPE_OVERRIDES+=("perf.prompt_mode=dataset" "perf.data_source=$DATASET")
fi

# --- window sweep: one invocation each, separate output dirs -----------------
if [[ -n "$WINDOW" ]]; then
    IFS=',' read -ra WINDOWS <<< "$WINDOW"
else
    WINDOWS=("")
fi

run_one() {
    local out_dir="$1"; shift
    mkdir -p "$out_dir"
    # Provenance the npz cannot carry: exact commit and full env.
    git rev-parse HEAD > "$out_dir/perf.env" 2>/dev/null || echo "no_git" > "$out_dir/perf.env"
    pip freeze >> "$out_dir/perf.env" 2>/dev/null || true

    local overrides=("$@")
    echo "=== $out_dir ==="
    echo "    overrides: ${overrides[*]:-<none>}"
    local args=(--config "$CONFIG")
    if [[ ${#overrides[@]} -gt 0 ]]; then
        args+=(--override "${overrides[@]}")
    fi
    python "$PROJECT_ROOT/main.py" "${args[@]}" "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

    python "$PROJECT_ROOT/scripts/print_efficiency.py" \
        --npz-dir "$out_dir" \
        --out "$out_dir/efficiency_report.txt"
}

for w in "${WINDOWS[@]}"; do
    overrides=("${SHAPE_OVERRIDES[@]+"${SHAPE_OVERRIDES[@]}"}")
    if [[ -n "$w" ]]; then
        w="$(echo "$w" | tr -d '[:space:]')"
        [[ "$w" =~ ^[0-9]+$ ]] || { echo "--window expects integers, got '$w'" >&2; exit 2; }
        out_dir="$OUT_ROOT/ws$w"
        overrides+=("window.window_size=$w")
    else
        out_dir="$OUT_ROOT"
    fi
    overrides+=("telemetry.output_dir=$out_dir")
    run_one "$out_dir" "${overrides[@]}"
done
