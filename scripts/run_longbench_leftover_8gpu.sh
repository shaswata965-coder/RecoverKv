#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Finish the 2026-09-19 LongBench budget sweep (outputs/longbench/sweep_395af34):
# run ONLY the jobs that crashed, on the SAME code as the 20 that finished.
#
# Why pinned: the sweep ran at 395af34. A `git pull` at 02:32 moved the checkout
# onto 4d588e2, whose compiled eviction recompiles per prompt length, hits
# Dynamo's 64-entry limit and raises "the two-tier eviction ran EAGER". All 16
# jobs started after the pull died that way, 13-48 examples in. Re-running them
# at HEAD would crash again, and would mix two code versions into one table.
# So they run from a worktree detached at 395af34, which no pull can move.
#
# How: this wrapper hands off to run_longbench_budget_sweep.sh (the tested
# shared queue, longest job first, one job per free GPU) with OUT_ROOT pointed
# at the existing sweep. That script skips every job whose <dataset>.jsonl AND
# .meta.json exist, so the 20 finished cells are left alone and only the crashed
# ones run. A crashed job has a partial jsonl but no .meta.json; the runner
# reopens the jsonl with "w", so the partial file is replaced, not appended to.
#
# Expected: 16 jobs, ~12,400 GPU-s, ~30-35 min on 8 GPUs.
#
# Run on an 8-GPU node:
#   conda activate sticky_env
#   nohup scripts/run_longbench_leftover_8gpu.sh > ~/lb_leftover.log 2>&1 &
#   tail -f ~/lb_leftover.log
# ============================================================================

REPO=/home/ee/phd/eez228470/kv_cache/Cluster_QEvict/RecoverKv
PIN=395af34
WT="${WT:-/home/ee/phd/eez228470/kv_cache/Cluster_QEvict/RecoverKv_${PIN}}"
OUT_ROOT="$REPO/outputs/longbench/sweep_${PIN}"
DATASETS=(gov_report multi_news samsum qmsum narrativeqa trec triviaqa
          musique hotpotqa qasper 2wikimqa multifieldqa_en)
BUDGETS="0.20 0.10 0.05"

[[ -d "$OUT_ROOT" ]] || { echo "error: no sweep at $OUT_ROOT" >&2; exit 2; }

# ---- the pinned checkout ------------------------------------------------------
if [[ ! -d "$WT" ]]; then
    echo "creating worktree $WT at $PIN"
    git -C "$REPO" worktree add --detach "$WT" "$PIN"
fi
have="$(git -C "$WT" rev-parse --short HEAD)"
if [[ "$have" != "$PIN" ]]; then
    echo "error: $WT is at $have, not $PIN. Run: git -C $WT checkout --detach $PIN" >&2
    exit 2
fi
# The sweep script is not in git at 395af34; use the current one (it carries the
# HEAD-moved guard, start-time commit record and partial-aware summary).
cp "$REPO/scripts/run_longbench_budget_sweep.sh" "$WT/scripts/"

# ---- what will run --------------------------------------------------------------
todo=()
for d in "${DATASETS[@]}"; do
    for b in $BUDGETS; do
        dir="$OUT_ROOT/q070_b${b/./}_w8_l128"
        [[ -s "$dir/$d.jsonl" && -f "$dir/$d.meta.json" ]] || todo+=("b=$b $d")
    done
done
echo "=== leftover LongBench jobs at $PIN: ${#todo[@]} ==="
printf '  %s\n' "${todo[@]}"
if (( ${#todo[@]} == 0 )); then
    echo "nothing left to run; summary: $OUT_ROOT/summary.txt"
    exit 0
fi
echo

# ---- hand off --------------------------------------------------------------------
cd "$WT"
OUT_ROOT="$OUT_ROOT" BUDGETS="$BUDGETS" exec "$WT/scripts/run_longbench_budget_sweep.sh"
