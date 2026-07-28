"""M7 — Global LIR, episode revival rate (*why promote*).

Reads ``R0`` (ground truth), ``R3`` (three-tier ws=8) and ``R4`` (three-tier
ws=32) — the *same* npzs ``configs/eval_tier_study.yaml`` already declares for
``combined.py``, restricted to the three conditions this question needs.  No new
recorded run is involved.

This is Observation III of ``qevict_observations`` (``episode_lir`` /
``binary_transition``) ported onto the run-matrix framework.  The question is
**why promote**: once a window has gone cold, does it ever regain importance,
and how much of that does the real fp-tier promotion mechanism actually catch?

What it reads
-------------
* ``matrix.mass_cum(cid)`` — R0's ground-truth head-mean mass, re-bucketed onto
  ``cid``'s window axis.  This is the ranking signal behind the *oracle*
  selection: the headroom a promotion path could capture.
* ``view.acc_fp`` ``[M, R, W]`` — the run's measured fp tier, already scattered
  from the survivor axis onto the merged window axis.  This is the *policy_fp*
  selection: what promotion recovers today.
* ``view.band_mask()`` ``[R, W]`` — the evictable band.  The unconditionally
  kept local tail is excluded, exactly as M2 excludes it from churn and for the
  same reason: a window that no policy may drop would make "importance" look
  trivially persistent.
* ``view.ew_act`` / ``view.event_steps`` — the band geometry, from which the
  per-window *creation event* is derived and passed explicitly rather than
  inferred from the ``-1`` pattern (see below).

Both selections are trinary ``[M, R, W]`` ``int8`` matrices over
``(trace, event, window)``: ``-1`` outside the band at that event, ``0`` in the
band and not selected, ``1`` selected.  ``int8`` is simply the narrowest integer
NumPy offers — it is bookkeeping, unrelated to this branch's int2 K/V packing.

No horizon
----------
Every rate here is ``episode_lir(..., horizon=None)``: a promotion counts no
matter how far out it lands, scored against however much trace remains.  There
is no ``H`` knob, and nothing is right-censored — the denominator is every
episode that ever went cold for ``m`` events.

That is deliberate, not a default.  The old estimator fixed a horizon and asked
"was the episode rescued within ``H`` events", which biases the rate two ways at
once:

1. **down** — a rescue landing at ``H+1`` or later scores as a *failure*;
2. **up** — any episode whose eligibility event sits within ``H`` of the trace
   end leaves numerator **and** denominator, and those late-starting episodes
   are exactly the ones with the least trace left in which to recover, so
   censoring preferentially deletes failures.

No choice of ``H`` escapes both: raising it trades the first bias for the
second, and once ``H >= R - m + 1`` every episode is censored and ``eligible``
is zero.  There is no limit in which the finite estimator converges on the
uncapped one, which is why this module carries no finite-``H`` comparison — the
uncapped rate is the estimate, not a reference point for a fixed one.  The
finite path still exists in :func:`utils.qevict_metrics.episode_lir` for
``qevict_observations``' own Observation III sweep.

The secondary table sweeps the *inactivity* threshold ``m`` instead, at the same
uncapped horizon: a larger ``m`` demands a longer silence before an episode
counts, which shrinks the denominator honestly rather than by censoring.

``binary_transition`` is a fixed-lag flip probability, not a lookback, so none
of the above touches it; it is computed exactly as ``analyse_revival`` does.

Creation events
---------------
``ConditionView.creation`` is ``[W]`` in *step* space and marks when a window
becomes valid at all (``w_act``).  The selection matrices here live on the
*event* axis and are restricted to the *evictable* band, which a window enters
``local_windows`` later.  Passing the former would mark windows as born while
they are still ``-1``, and ``episode_lir`` drops any window that is ``-1`` after
its creation — i.e. it would silently discard the whole trace.  So the creation
event is derived from the same geometry on the right axis
(:func:`band_creation_events`): the first routing event at which the window is
an evictable candidate.

Per head
--------
LIR is a set-membership statistic over the *shared per-layer* retained set (the
policy ranks head-mean scores), so a per-``(layer, head)`` array would be
layer-invariant by construction — the same situation M2 documents for its raw
churn.  No ``[L, H]`` array is fabricated; the per-layer resolution that is real
is emitted instead, and ``--no-per-head`` therefore changes nothing here.

Caveat
------
**Do not compare this LIR to Suite B's ``global_lir``.**  Suite B counts every
``(event, window)`` lookback pair (so one long inactive run contributes many
eligible pairs), accepts a rescue arbitrarily far in the future, applies no
censoring, and simulates Sticky-K on ground truth with ``m = 3`` hard-wired.
They are different estimators of related quantities (``EVALUATION_GUIDE.md``,
Observation III).  Dropping the horizon closes *one* of those differences and
none of the others: this is still an **episode** count (one long cold run
contributes one eligible unit, not many pairs), still measured over the real
run's selections rather than a Sticky-K simulation, and still at a configurable
``m``.  Uncapped does not make the two numbers comparable.

CLI::

    python -m modules.evaluation.tier_study.m7_global_lir \\
        --r0-npz ... --r3-npz ... --r4-npz ... \\
        --output-dir outputs/tier_study

``combined.py`` runs this as the seventh module of the tier study, so
``python main.py --config configs/eval_tier_study.yaml`` writes
``m7_global_lir.{csv,json,npz}`` alongside ``all_metrics.*`` and the viz bundle
without a separate invocation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.evaluation.qevict_observations import DEFAULTS as QEVICT_DEFAULTS
from modules.evaluation.tier_study import (
    _ci_pct,
    _num,
    _pct,
    add_matrix_args,
    bootstrap_row,
    matrix_from_args,
    matrix_header_lines,
    per_trace_to_layer,
    warn_vacuous,
    write_metric_outputs,
)
from utils import qevict_metrics as QM
from utils.logger import get_logger

log = get_logger(__name__)

METRIC_ID = "m7"
METRIC_NAME = "global_lir"
METRIC_TITLE = "M7 — Global LIR (episode revival rate)"
METRIC_QUESTION = "why promote"
STEM = f"{METRIC_ID}_{METRIC_NAME}"
RUNS: Tuple[str, ...] = ("R0", "R3", "R4")
#: R1/R2 have no Q tier, so "cold but still resident" is not a state they can be
#: in — there is nothing for a promotion path to promote *from*.
POLICY_RUNS: Tuple[str, ...] = ("R3", "R4")

#: The two selections of Observation III's table, same names and same meaning.
SELECTIONS: Tuple[str, ...] = ("oracle", "policy_fp")

#: Sweep knobs come from the QEvict suite so the two cannot drift.  Neither
#: ``lir_horizon_values`` nor ``primary_lir_horizon`` is pulled in: this metric
#: has no horizon knob at all (see the module docstring).
DEFAULTS: Dict[str, Any] = {
    "lir_inactivity_values": QEVICT_DEFAULTS["lir_inactivity_values"],
    "transition_deltas": QEVICT_DEFAULTS["transition_deltas"],
    "primary_inactivity": QEVICT_DEFAULTS["primary_inactivity"],
    "primary_transition_delta": QEVICT_DEFAULTS["primary_transition_delta"],
}


def band_creation_events(view) -> np.ndarray:
    """``[W]`` first *routing event* at which each window is an evictable candidate.

    ``ew_act`` is non-decreasing and ``event_steps`` is increasing, so the band
    width per event is non-decreasing too and the answer is one ``searchsorted``
    — the same closed form ``sticky_metrics.flush_geometry`` uses for its
    step-space ``creation``, on the event axis and against the *evictable* count
    rather than the valid count.  ``R`` for windows that never enter the band.
    """
    band_width = view.band_mask().sum(axis=1)                  # [R], sorted
    idx = np.searchsorted(band_width, np.arange(view.num_windows) + 1,
                          side="left")
    return np.minimum(idx, view.num_events).astype(int)


def selection_matrices(matrix, cid: str) -> List[Tuple[str, np.ndarray]]:
    """``(name, [M, R, W] int8)`` trinary selections for one condition.

    Mirrors ``qevict_observations``' construction exactly (same per-event
    argsort-top-k over the band), reading ``matrix.mass_cum(cid)`` and
    ``view.acc_fp`` instead of the arrays that module builds for itself.
    """
    view = matrix.conditions[cid]
    M, R, W = matrix.num_traces, view.num_events, view.num_windows
    band_width = view.band_mask().sum(axis=1)                  # [R]
    rank = matrix.mass_cum(cid)[:, view.event_steps, :]        # [M, R, W]
    k_fp = int(view.top_k_fp)

    oracle = np.full((M, R, W), -1, dtype=np.int8)
    policy = np.full((M, R, W), -1, dtype=np.int8)
    for r in range(R):
        ew = int(band_width[r])
        if ew <= 0:
            continue
        oracle[:, r, :ew] = 0
        policy[:, r, :ew] = 0
        policy[:, r, :ew][view.acc_fp[:, r, :ew]] = 1
        k = min(k_fp, ew)
        if k > 0:
            sel = np.argsort(-rank[:, r, :ew], axis=-1, kind="stable")[:, :k]
            np.put_along_axis(oracle[:, r, :ew], sel, 1, axis=-1)
    return [("oracle", oracle), ("policy_fp", policy)]


def _rate_ci(res: Dict[str, Any], matrix, *, confidence: float, n_boot: int,
             seed: int) -> Tuple[float, float, float]:
    """Bootstrap the group ratio-of-sums rescued/eligible.

    Not ``bootstrap_row``: that group-*reduces* (mean of per-trace values), and
    averaging per-trace ratios is the wrong aggregation for a count-based rate —
    ``sticky_metrics`` and ``analyse_revival`` both sum the counts first.
    """
    return QM.bootstrap_mean_ci(
        QM.group_ratio(res["rescued_by_trace"], res["eligible_by_trace"],
                       matrix.trace_group),
        confidence, n_boot, seed)


def _ttr_quantiles(res: Dict[str, Any]) -> Tuple[float, float, float]:
    ttr = np.asarray(res["time_to_revival"], dtype=float)
    if not ttr.size:
        return np.nan, np.nan, np.nan
    q1, q3 = (float(v) for v in np.quantile(ttr, [0.25, 0.75]))
    return float(np.median(ttr)), q1, q3


def compute(
    matrix,
    *,
    inactivity_values: Sequence[int] = DEFAULTS["lir_inactivity_values"],
    transition_deltas: Sequence[int] = DEFAULTS["transition_deltas"],
    primary_inactivity: int = DEFAULTS["primary_inactivity"],
    primary_transition_delta: int = DEFAULTS["primary_transition_delta"],
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    per_head: bool = True,
) -> Dict[str, Any]:
    """Compute M7 over a loaded run matrix."""
    present = [c for c in POLICY_RUNS if c in matrix.conditions]
    if not present:
        raise KeyError(f"{METRIC_TITLE} needs at least one of {POLICY_RUNS}")
    matrix.require(*present)

    ms = tuple(int(m) for m in inactivity_values)
    m0 = int(primary_inactivity)

    rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []
    trans_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    n = 0

    for cid in present:
        view = matrix.conditions[cid]
        R = view.num_events
        creation = band_creation_events(view)
        ds = tuple(int(d) for d in transition_deltas if 0 < int(d) < R)
        arrays[f"event_steps__{cid}"] = view.event_steps
        arrays[f"creation_events__{cid}"] = creation
        if view.top_k_fp <= 0:
            warn_vacuous(f"{cid}/oracle", METRIC_TITLE,
                         "the run declares top_k_fp=0, so the oracle selection "
                         "is empty at every event")

        for sel_name, sel in selection_matrices(matrix, cid):
            key = f"{cid}__{sel_name}"
            n += 1
            base = seed + 977 * n

            # ── headline: uncapped, "was it EVER rescued" ────────────────
            unc = QM.episode_lir(sel, m0, None, creation)
            if unc["eligible"] == 0:
                warn_vacuous(
                    f"{cid}/{sel_name}", METRIC_TITLE,
                    f"no window stays cold for m={m0} consecutive events over "
                    f"R={R} routing events — there is no episode to revive")
            mu, lo, hi = _rate_ci(unc, matrix, confidence=confidence,
                                  n_boot=n_boot, seed=base)
            med, q1, q3 = _ttr_quantiles(unc)

            # ── secondary: the same uncapped rate across inactivity m ────
            # What survives of the old [m x H] grid once H is gone: how the
            # revival rate moves with how cold "cold" has to be.
            sweep = np.full(len(ms), np.nan)
            sweep_lo = np.full(len(ms), np.nan)
            sweep_hi = np.full(len(ms), np.nan)
            for i, m in enumerate(ms):
                res = (unc if m == m0
                       else QM.episode_lir(sel, m, None, creation))
                sweep[i], sweep_lo[i], sweep_hi[i] = _rate_ci(
                    res, matrix, confidence=confidence, n_boot=n_boot,
                    seed=base + 1 + i)
                sweep_rows.append({
                    "condition": cid, "selection": sel_name, "inactivity_m": m,
                    "global_lir_uncapped": sweep[i],
                    "ci_lower": sweep_lo[i], "ci_upper": sweep_hi[i],
                    "global_lir_uncapped_pooled": float(res["global_rate"]),
                    "eligible_episodes": int(res["eligible"]),
                    "rescued_episodes": int(res["rescued"]),
                })

            # ── transitions ──────────────────────────────────────────────
            p01_by_delta: List[float] = []
            p10_by_delta: List[float] = []
            prim_t: Dict[str, Any] = {}
            for i, d in enumerate(ds):
                tr = QM.binary_transition(sel, d, creation)
                p = tr["probabilities_by_trace"]
                p01 = bootstrap_row(p[:, 0, 1], matrix, confidence=confidence,
                                    n_boot=n_boot, seed=base + 500 + i)
                p10 = bootstrap_row(p[:, 1, 0], matrix, confidence=confidence,
                                    n_boot=n_boot, seed=base + 600 + i)
                row = {
                    "condition": cid, "selection": sel_name, "delta": d,
                    "P01_mean": p01[0], "P01_ci_lower": p01[1],
                    "P01_ci_upper": p01[2],
                    "P10_mean": p10[0], "P10_ci_lower": p10[1],
                    "P10_ci_upper": p10[2],
                    "pooled_P01": float(tr["pooled_probabilities"][0, 1]),
                    "pooled_P10": float(tr["pooled_probabilities"][1, 0]),
                }
                trans_rows.append(row)
                p01_by_delta.append(p01[0])
                p10_by_delta.append(p10[0])
                if d == int(primary_transition_delta):
                    prim_t = row
            # Indexed by ``ds`` — the requested deltas filtered to 0 < d < R,
            # which is not necessarily ``transition_deltas`` in full.
            arrays[f"transition_deltas__{key}"] = np.asarray(ds, dtype=int)
            arrays[f"transition_p01__{key}"] = np.asarray(p01_by_delta,
                                                          dtype=float)
            arrays[f"transition_p10__{key}"] = np.asarray(p10_by_delta,
                                                          dtype=float)

            # ── arrays ───────────────────────────────────────────────────
            arrays[f"lir_uncapped__{key}"] = np.asarray(mu)
            arrays[f"lir_uncapped_ci_lower__{key}"] = np.asarray(lo)
            arrays[f"lir_uncapped_ci_upper__{key}"] = np.asarray(hi)
            arrays[f"time_to_revival_uncapped__{key}"] = np.asarray(
                unc["time_to_revival"], dtype=np.int64)
            arrays[f"lir_uncapped_per_trace__{key}"] = unc["rate_by_trace"]
            arrays[f"lir_uncapped_per_layer__{key}"] = per_trace_to_layer(
                unc["rate_by_trace"], matrix)
            arrays[f"lir_by_inactivity__{key}"] = sweep
            arrays[f"lir_by_inactivity_ci_lower__{key}"] = sweep_lo
            arrays[f"lir_by_inactivity_ci_upper__{key}"] = sweep_hi

            rows.append({
                "condition": cid,
                "label": view.label,
                "window_size": view.window_size,
                "selection": sel_name,
                "inactivity_m": m0,
                "global_lir_uncapped": mu,
                "ci_lower": lo,
                "ci_upper": hi,
                "global_lir_uncapped_pooled": float(unc["global_rate"]),
                "eligible_episodes": int(unc["eligible"]),
                "rescued_episodes": int(unc["rescued"]),
                "time_to_revival_median": med,
                "time_to_revival_q1": q1,
                "time_to_revival_q3": q3,
                "time_to_revival_max": (int(unc["time_to_revival"].max())
                                        if unc["time_to_revival"].size else np.nan),
                "transition_delta": int(primary_transition_delta),
                "P01_mean": prim_t.get("P01_mean", np.nan),
                "P01_ci_lower": prim_t.get("P01_ci_lower", np.nan),
                "P01_ci_upper": prim_t.get("P01_ci_upper", np.nan),
                "P10_mean": prim_t.get("P10_mean", np.nan),
                "P10_ci_lower": prim_t.get("P10_ci_lower", np.nan),
                "P10_ci_upper": prim_t.get("P10_ci_upper", np.nan),
                "routing_events": R,
                "is_primary": sel_name == "policy_fp",
            })

    arrays["lir_inactivity_values"] = np.asarray(ms, dtype=int)
    arrays["transition_deltas"] = np.asarray(
        [int(d) for d in transition_deltas], dtype=int)
    arrays["layer_ids"] = matrix.layer_ids
    arrays["head_ids"] = matrix.head_ids
    arrays["trace_group"] = matrix.trace_group

    knobs = {
        "inactivity_values": list(ms),
        "transition_deltas": [int(d) for d in transition_deltas],
        "primary_inactivity": m0,
        "horizon": None,
        "primary_transition_delta": int(primary_transition_delta),
        "confidence": confidence, "bootstrap_samples": int(n_boot),
        "seed": int(seed), "per_head": bool(per_head),
    }
    result = {
        "metric_id": METRIC_ID, "metric_name": METRIC_NAME,
        "title": METRIC_TITLE, "question": METRIC_QUESTION,
        "runs_used": [c for c in RUNS if c in matrix.conditions],
        "knobs": knobs, "summary_table": rows,
        "inactivity_sweep_table": sweep_rows,
        "transition_table": trans_rows,
        "diagnostics": {
            "horizon": None,
            "horizon_note": (
                "this metric has no horizon knob: every rate is "
                "episode_lir(horizon=None) — 'was the episode EVER rescued', "
                "scored against however much trace remains, with no "
                "right-censoring and an unbounded time_to_revival"),
            "per_head_lir_is_head_invariant": True,
            "per_head_note": (
                "LIR is set membership over the shared per-layer retained set "
                "(the policy ranks head-mean scores), so a [L, H] array would "
                "be the layer value broadcast — none is emitted, and "
                "--no-per-head changes nothing for this metric; the real "
                "resolution is lir_uncapped_per_layer__*"),
            "creation_events": (
                "passed explicitly as the first ROUTING EVENT at which a window "
                "is an evictable candidate (derived from band_mask), not "
                "ConditionView.creation — that one is in step space and marks "
                "window validity, which would mark windows born while still -1"),
            "not_comparable_to_suite_b": (
                "Suite B's global_lir counts every (event, window) lookback "
                "pair, applies no censoring and hard-wires m=3 on a Sticky-K "
                "simulation over ground truth — a different estimator"),
        },
        "arrays": arrays,
    }
    result["report"] = render(matrix, result)
    return result


def render(matrix, result: Dict[str, Any]) -> str:
    knobs = result["knobs"]
    lines = [
        f"## {METRIC_TITLE} ({METRIC_QUESTION})",
        "",
        "Each maximal cold run counts once, becomes eligible at its `m`-th "
        "consecutive miss outside the important set, and is **rescued if a hit "
        "lands anywhere later in the trace** — uncapped, and with no "
        "right-censoring. `oracle` is R0's top-`top_k_fp` by ground-truth mass "
        "(the headroom a promotion path could capture); `policy_fp` is the "
        "run's real fp tier (what promotion recovers today).",
        "",
    ]
    for r in result["summary_table"]:
        lines.append(
            f"- `{r['condition']}` / `{r['selection']}`: after "
            f"{r['inactivity_m']} consecutive events outside the "
            f"{r['selection']} important set, "
            f"{_pct(r['global_lir_uncapped_pooled'])} of cold episodes were "
            f"EVER rescued ({r['rescued_episodes']}/{r['eligible_episodes']}), "
            f"with a median of {_num(r['time_to_revival_median'], 1)} events "
            f"before it came back and a longest revival at "
            f"{_num(r['time_to_revival_max'], 0)}.")
    lines += [
        "",
        "| run | ws | selection | LIR (ever) | episodes | time to revival (q1/med/q3/max) | "
        f"P01 (lag {knobs['primary_transition_delta']}) | P10 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["summary_table"]:
        lines.append(
            f"| `{r['condition']}` {r['label']} | {r['window_size']} | "
            f"`{r['selection']}` | "
            f"{_ci_pct(r['global_lir_uncapped'], r['ci_lower'], r['ci_upper'])} | "
            f"{r['rescued_episodes']}/{r['eligible_episodes']} | "
            f"{_num(r['time_to_revival_q1'], 1)} / "
            f"{_num(r['time_to_revival_median'], 1)} / "
            f"{_num(r['time_to_revival_q3'], 1)} / "
            f"{_num(r['time_to_revival_max'], 0)} | "
            f"{_ci_pct(r['P01_mean'], r['P01_ci_lower'], r['P01_ci_upper'])} | "
            f"{_ci_pct(r['P10_mean'], r['P10_ci_lower'], r['P10_ci_upper'])} |")

    ms = knobs["inactivity_values"]
    if len(ms) > 1:
        lines += [
            "",
            "**Secondary — how cold is cold enough** (the same uncapped rate "
            "across the inactivity threshold `m`; a larger `m` demands a longer "
            "silence before an episode counts, so it shrinks the denominator "
            "without ever censoring one):",
            "",
            "| run | selection | " + " | ".join(f"m={m}" for m in ms) + " |",
            "| --- | --- | " + " | ".join("---" for _ in ms) + " |",
        ]
        by_key: Dict[Tuple[str, str], Dict[int, Any]] = {}
        for g in result["inactivity_sweep_table"]:
            by_key.setdefault((g["condition"], g["selection"]), {})[
                int(g["inactivity_m"])] = g
        for (cid, sel_name), per_m in by_key.items():
            cells = []
            for m in ms:
                g = per_m.get(int(m))
                cells.append(
                    "NA" if g is None else
                    f"{_pct(g['global_lir_uncapped_pooled'])} "
                    f"({g['rescued_episodes']}/{g['eligible_episodes']})")
            lines.append(f"| `{cid}` | `{sel_name}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "> There is **no horizon knob**. An episode is rescued if a hit lands "
        "anywhere later in the trace, so nothing is right-censored and the "
        "denominator is every episode that ever went cold. A fixed `H` would "
        "bias this two ways at once — *down* by scoring a rescue at `H+1` as a "
        "failure, *up* by censoring away the episodes nearest the trace end, "
        "which are exactly the ones with the least room left to recover — and "
        "no `H` escapes it: raising `H` trades the first bias for the second "
        "until `H >= R - m + 1` censors every episode away.",
        "> `P01` (cold -> hot at lag `delta`) is the promotion argument and "
        "`P10` the demotion argument; both are fixed-lag flip probabilities, "
        "not lookbacks, so neither is affected by the above.",
        "> **Do not compare this to Suite B's `global_lir`.** Suite B counts "
        "every `(event, window)` lookback pair, applies no censoring, and "
        "hard-wires `m=3` on a Sticky-K simulation over ground truth — a "
        "different estimator of a related quantity.",
        "> LIR is set membership over the retained set the policy shares across "
        "heads, so there is no per-head resolution to report; "
        "`lir_uncapped_per_layer__*` is the finest axis that carries signal.",
    ]
    return "\n".join(lines)


def write(matrix, result: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    meta = {**matrix.metadata_block(result["runs_used"]),
            "metric": METRIC_ID, "metric_name": METRIC_NAME,
            "question": METRIC_QUESTION, "knobs": result["knobs"]}
    return write_metric_outputs(
        out_dir, STEM,
        tables=(("summary", result["summary_table"]),
                ("inactivity_sweep", result["inactivity_sweep_table"]),
                ("transitions", result["transition_table"])),
        payload={k: result[k] for k in
                 ("metric_id", "metric_name", "title", "question", "runs_used",
                  "knobs", "summary_table", "inactivity_sweep_table",
                  "transition_table", "diagnostics")},
        arrays=result["arrays"], meta=meta, report=result["report"])


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_matrix_args(ap, POLICY_RUNS)
    args = ap.parse_args(argv)
    matrix = matrix_from_args(args, POLICY_RUNS)
    result = compute(matrix, confidence=args.confidence,
                     n_boot=args.bootstrap_samples, seed=args.seed,
                     per_head=not args.no_per_head)
    write(matrix, result, args.output_dir)
    print("\n".join(matrix_header_lines(matrix, result["runs_used"])))
    print()
    print(result["report"])


if __name__ == "__main__":
    main()
