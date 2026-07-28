"""M6 — Mass the Q tier recovers instead of evicting (*why three tiers*).

Reads ``R0`` (ground truth), ``R2`` (evict-only ws=8) and ``R3`` (three-tier
ws=8).  Two paired quantities, one simulated and one measured:

**(a) Recovered Mass Q** — ``missed_mass_fp - missed_mass_kept`` from
``sticky_metrics.simulate_policy`` over R0's ground-truth masses:

* ``missed_mass_fp``   — mass on evictable windows *below the fp tier*: what a
  cache with no middle tier throws away.
* ``missed_mass_kept`` — mass on windows in **neither** tier: what the three-tier
  cache actually loses.
* their difference is the mass int4 rescues.

The same simulation is also run at ``R2``'s capacity (``n_q = 0``, all the byte
budget in fp16), which gives the honest **cross-run** recovery
``missed(R2 capacity) - missed_kept(R3 capacity)``: not "Q vs nothing" but "Q vs
spending those same bytes on more fp16 windows", which is the design decision.

The primary simulation is **Fresh-K** (re-rank the whole band at every flush) —
that is what ``EvictionPolicy.compute_two_tier_retain`` does.  Sticky-K is
reported alongside it because ``compute_sticky_metrics`` treats it as the default
elsewhere in the repo, and the gap between them is itself informative.

**(b) Paired FMM reduction** — the same question against *measured* accessible
sets and future attention (``qevict_metrics.future_missed_mass``, horizon ``H``):

* ``R2.alive`` vs ``R3.alive`` — **byte-matched across runs** (the loader checks
  both runs declare the same ``cache_budget``; the resolver splits that one
  budget between fp16 and int4);
* ``R3.fp_only`` vs ``R3.alive`` — within one run, showing what crediting the
  int4 tier adds at a fixed fp capacity.  This one is *not* byte-matched and is
  labelled as such: a three-tier run's fp tier alone is a smaller cache.

Per head
--------
Both quantities are re-run per head against the layer's shared retained set /
capacities.

CLI::

    python -m modules.evaluation.tier_study.m6_recovered_mass_q \\
        --r0-npz ... --r2-npz ... --r3-npz ... \\
        --output-dir outputs/tier_study --fmm-horizon 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.evaluation.tier_study import (
    _ci_num,
    _ci_pct,
    _pct,
    add_matrix_args,
    bootstrap_row,
    matrix_from_args,
    matrix_header_lines,
    paired_row,
    per_trace_to_layer,
    per_trace_to_layer_series,
    warn_vacuous,
    write_metric_outputs,
)
from utils import qevict_metrics as QM
from utils import sticky_metrics as SM
from utils.logger import get_logger

log = get_logger(__name__)

METRIC_ID = "m6"
METRIC_NAME = "recovered_mass_q"
METRIC_TITLE = "M6 — Recovered mass (Q tier)"
METRIC_QUESTION = "why three tiers"
STEM = f"{METRIC_ID}_{METRIC_NAME}"
RUNS: Tuple[str, ...] = ("R0", "R2", "R3")
POLICY_RUNS: Tuple[str, ...] = ("R2", "R3")

DEFAULTS: Dict[str, Any] = {"horizon": 32, "sticky": False}


def simulate_missed(matrix, cid: str, *, budget_k: int, n_q: int,
                    is_sticky: bool, head: Optional[int] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """``([M, T] missed_fp, [M, T] missed_kept)`` over R0's ground-truth masses.

    Drives ``sticky_metrics.simulate_policy`` one trace at a time.  The head-mean
    call reproduces ``compute_sticky_metrics``' ``missed_mass_per_layer`` exactly
    (same policy, same inputs) without materialising the ``[S, T, L, H, W]``
    float64 array that driver needs — which at ``ws = 8`` and a long run is
    several GB — and without its per-head LIR loop, which this metric does not use.
    """
    view = matrix.conditions[cid]
    mass = matrix.mass_cum(cid, head=head)                    # [M, T, W]
    M, T, _ = mass.shape
    missed_fp = np.zeros((M, T))
    missed_kept = np.zeros((M, T))
    for m in range(M):
        _, fp, kept = SM.simulate_policy(
            mass[m], budget_k, view.w_act, view.ew_act,
            is_sticky=is_sticky, n_q=n_q)
        missed_fp[m] = fp
        missed_kept[m] = kept
    return missed_fp, missed_kept


def _series_stats(matrix, cid: str, values: np.ndarray, *, confidence: float,
                  n_boot: int, seed: int) -> Dict[str, Any]:
    """Per-trace mean over all steps and over flush points only, with CIs."""
    view = matrix.conditions[cid]
    all_steps = QM.nanmean(values, axis=1)                    # [M]
    at_flush = QM.nanmean(values[:, view.event_steps], axis=1)
    mu, lo, hi = bootstrap_row(all_steps, matrix, confidence=confidence,
                               n_boot=n_boot, seed=seed)
    f_mu, f_lo, f_hi = bootstrap_row(at_flush, matrix, confidence=confidence,
                                     n_boot=n_boot, seed=seed + 1)
    return {"mean": mu, "ci_lower": lo, "ci_upper": hi,
            "mean_at_flushes": f_mu, "flush_ci_lower": f_lo,
            "flush_ci_upper": f_hi, "per_trace": all_steps}


def compute(
    matrix,
    *,
    horizon: int = DEFAULTS["horizon"],
    sticky: bool = DEFAULTS["sticky"],
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    per_head: bool = True,
) -> Dict[str, Any]:
    """Compute M6 over a loaded run matrix."""
    matrix.require("R2", "R3")
    r2, r3 = matrix.conditions["R2"], matrix.conditions["R3"]
    if r2.window_size != r3.window_size:
        raise ValueError(
            f"M6 compares R2 (ws={r2.window_size}) with R3 (ws={r3.window_size}) "
            "at a matched budget; different window sizes make the comparison a "
            "granularity contrast instead. Fix the run matrix.")
    if r3.n_q <= 0:
        warn_vacuous("R3", METRIC_TITLE,
                     "the three-tier run records N_q=0, so there is no Q tier "
                     "to recover anything")
    byte_matched = (r2.cache_budget is not None and r3.cache_budget is not None
                    and abs(float(r2.cache_budget) - float(r3.cache_budget)) < 1e-9)
    if not byte_matched:
        log.warning(
            "R2 cache_budget=%s vs R3 cache_budget=%s — the cross-run "
            "comparison below is NOT byte-matched; report it as such.",
            r2.cache_budget, r3.cache_budget)

    arrays: Dict[str, np.ndarray] = {}
    mass_rows: List[Dict[str, Any]] = []
    fmm_rows: List[Dict[str, Any]] = []
    paired: List[Dict[str, Any]] = []

    # ── (a) simulated recovery over R0's ground-truth masses ─────────────
    policies = ([("sticky", True), ("fresh", False)] if sticky
                else [("fresh", False), ("sticky", True)])
    per_trace_recovery: Dict[str, np.ndarray] = {}
    for pi, (pol_name, is_sticky) in enumerate(policies):
        log.info("M6(a): simulating %s policy over %d trace(s) x 2 capacities",
                 pol_name, matrix.num_traces)
        fp3, kept3 = simulate_missed(matrix, "R3", budget_k=r3.top_k_fp,
                                     n_q=r3.n_q, is_sticky=is_sticky)
        fp2, _ = simulate_missed(matrix, "R3", budget_k=r2.top_k_fp, n_q=0,
                                 is_sticky=is_sticky)
        recovered = fp3 - kept3                       # within R3's capacity
        cross = fp2 - kept3                           # vs spending it all on fp16

        for name, series, note in (
            ("missed_mass_fp_R3", fp3,
             "mass below R3's fp tier (what a cache with no middle tier drops)"),
            ("missed_mass_kept_R3", kept3,
             "mass in neither tier — R3's honest miss"),
            ("missed_mass_R2_capacity", fp2,
             "evict-only at R2's fp capacity: the same bytes, all in fp16"),
            ("recovered_mass_q", recovered,
             "missed_mass_fp - missed_mass_kept, within R3's capacity"),
            ("recovered_vs_evict_only", cross,
             "missed(R2 capacity) - missed_kept(R3) — "
             + ("byte-matched" if byte_matched else "NOT byte-matched, budgets differ")),
        ):
            st = _series_stats(matrix, "R3", series, confidence=confidence,
                               n_boot=n_boot, seed=seed + 61 * pi + len(mass_rows))
            key = f"{name}__{pol_name}"
            arrays[f"{key}__series"] = QM.nanmean(series, axis=0).astype(np.float32)
            arrays[f"{key}__per_layer"] = per_trace_to_layer(st["per_trace"], matrix)
            arrays[f"{key}__per_trace"] = st["per_trace"]
            if name == "recovered_vs_evict_only":
                per_trace_recovery[pol_name] = st["per_trace"]
            mass_rows.append({
                "quantity": name,
                "policy": pol_name,
                "is_primary": pol_name == "fresh",
                "mean": st["mean"],
                "ci_lower": st["ci_lower"],
                "ci_upper": st["ci_upper"],
                "mean_at_flushes": st["mean_at_flushes"],
                "flush_ci_lower": st["flush_ci_lower"],
                "flush_ci_upper": st["flush_ci_upper"],
                "byte_matched": byte_matched if "R2" in name or "evict_only" in name
                                else "",
                "note": note,
            })

    if per_head:
        pol_name, is_sticky = policies[0]
        log.info("M6(a): per-head recovery over %d head(s) — %d simulations",
                 matrix.num_heads, matrix.num_heads * matrix.num_traces * 2)
        rec = np.full((matrix.num_layers, matrix.num_heads), np.nan)
        cross_ph = np.full((matrix.num_layers, matrix.num_heads), np.nan)
        rec_s = cross_s = None
        for j, h in enumerate(matrix.head_ids):
            fp3, kept3 = simulate_missed(matrix, "R3", budget_k=r3.top_k_fp,
                                         n_q=r3.n_q, is_sticky=is_sticky,
                                         head=int(h))
            fp2, _ = simulate_missed(matrix, "R3", budget_k=r2.top_k_fp, n_q=0,
                                     is_sticky=is_sticky, head=int(h))
            rec[:, j] = per_trace_to_layer(QM.nanmean(fp3 - kept3, axis=1), matrix)
            cross_ph[:, j] = per_trace_to_layer(
                QM.nanmean(fp2 - kept3, axis=1), matrix)
            # simulate_missed returns a per-STEP series (length T), not a
            # per-flush one — this axis is therefore decode steps, which is why
            # the bundle pairs it with `step_index` rather than `event_steps`.
            if rec_s is None:
                n = fp3.shape[1]
                rec_s = np.full((matrix.num_layers, matrix.num_heads, n),
                                np.nan, dtype=np.float32)
                cross_s = np.full_like(rec_s, np.nan)
            rec_s[:, j, :] = per_trace_to_layer_series(fp3 - kept3, matrix)
            cross_s[:, j, :] = per_trace_to_layer_series(fp2 - kept3, matrix)
        arrays[f"recovered_mass_q_per_layer_head__{pol_name}"] = rec
        arrays[f"recovered_vs_evict_only_per_layer_head__{pol_name}"] = cross_ph
        if rec_s is not None:
            arrays[f"recovered_mass_q_per_layer_head_step__{pol_name}"] = rec_s
            arrays[f"recovered_vs_evict_only_per_layer_head_step__{pol_name}"] = cross_s

    # ── (b) measured FMM, fp-only vs fp+Q ────────────────────────────────
    variants = [
        ("R2", "alive", r2.acc_alive),
        ("R3", "fp_only", r3.acc_fp),
        ("R3", "alive", r3.acc_alive),
    ]
    fmm_trace: Dict[Tuple[str, str], np.ndarray] = {}
    for vi, (cid, set_name, mask) in enumerate(variants):
        view = matrix.conditions[cid]
        fmm = QM.future_missed_mass(
            matrix.mass_step(cid), mask, view.event_steps, horizon,
            creation_steps=view.creation, require_full_horizon=True)
        scored = int(np.isfinite(fmm).any(axis=0).sum())
        if scored == 0:
            warn_vacuous(f"{cid}/{set_name}", f"{METRIC_TITLE} (b)",
                         f"no flush has a full {horizon}-step future horizon")
        trace_vals = QM.nanmean(fmm, axis=1)
        fmm_trace[(cid, set_name)] = trace_vals
        mu, lo, hi = bootstrap_row(trace_vals, matrix, confidence=confidence,
                                   n_boot=n_boot, seed=seed + 800 + vi)
        key = f"{cid}__{set_name}"
        arrays[f"fmm_event__{key}"] = fmm.astype(np.float32)
        arrays[f"fmm_per_trace__{key}"] = trace_vals
        arrays[f"fmm_per_layer__{key}"] = per_trace_to_layer(trace_vals, matrix)
        fmm_rows.append({
            "condition": cid, "accessible_set": set_name,
            "window_size": view.window_size, "horizon": int(horizon),
            "future_missed_mass_mean": mu, "ci_lower": lo, "ci_upper": hi,
            "flushes_scored": scored, "flushes_total": view.num_events,
            "mean_alive_units": float(mask.sum(axis=-1).mean()),
        })

    for i, (ref, cmp, matched, note) in enumerate((
        (("R2", "alive"), ("R3", "alive"), byte_matched,
         "byte-matched across runs: one cache_budget, spent all-fp16 vs split "
         "fp16+int4"),
        (("R3", "fp_only"), ("R3", "alive"), False,
         "within-run at a fixed fp capacity — NOT byte-matched (a three-tier "
         "run's fp tier alone is a smaller cache)"),
    )):
        if ref not in fmm_trace or cmp not in fmm_trace:
            continue
        res = paired_row(fmm_trace[ref], fmm_trace[cmp], matrix,
                         confidence=confidence, n_boot=n_boot, seed=seed + 900 + i)
        paired.append({
            "metric": "future_missed_mass",
            "reference_condition": f"{ref[0]}:{ref[1]}",
            "compared_condition": f"{cmp[0]}:{cmp[1]}",
            "byte_matched": bool(matched),
            "horizon": int(horizon),
            "note": note,
            **res,
        })

    if per_head:
        log.info("M6(b): per-head FMM over %d head(s)", matrix.num_heads)
        for cid, set_name, mask in variants:
            view = matrix.conditions[cid]
            ph = np.full((matrix.num_layers, matrix.num_heads), np.nan)
            phs = np.full((matrix.num_layers, matrix.num_heads,
                           view.num_events), np.nan, dtype=np.float32)
            for j, h in enumerate(matrix.head_ids):
                fmm_h = QM.future_missed_mass(
                    matrix.mass_step(cid, head=int(h)), mask, view.event_steps,
                    horizon, creation_steps=view.creation,
                    require_full_horizon=True)
                ph[:, j] = per_trace_to_layer(QM.nanmean(fmm_h, axis=1), matrix)
                phs[:, j, :] = per_trace_to_layer_series(fmm_h, matrix)
            arrays[f"fmm_per_layer_head__{cid}__{set_name}"] = ph
            arrays[f"fmm_per_layer_head_step__{cid}__{set_name}"] = phs
        if ("R2", "alive") in fmm_trace and ("R3", "alive") in fmm_trace:
            arrays["fmm_reduction_per_layer_head__R2_alive_vs_R3_alive"] = (
                arrays["fmm_per_layer_head__R2__alive"]
                - arrays["fmm_per_layer_head__R3__alive"])

    arrays["event_steps__R3"] = r3.event_steps
    arrays["event_steps__R2"] = r2.event_steps
    arrays["layer_ids"] = matrix.layer_ids
    arrays["head_ids"] = matrix.head_ids
    arrays["trace_group"] = matrix.trace_group

    knobs = {"horizon": int(horizon), "primary_policy": policies[0][0],
             "confidence": confidence, "bootstrap_samples": int(n_boot),
             "seed": int(seed), "per_head": bool(per_head)}
    result = {
        "metric_id": METRIC_ID, "metric_name": METRIC_NAME,
        "title": METRIC_TITLE, "question": METRIC_QUESTION,
        "runs_used": [c for c in RUNS if c in matrix.conditions],
        "knobs": knobs, "mass_table": mass_rows, "fmm_table": fmm_rows,
        "paired_table": paired,
        "diagnostics": {
            "byte_matched": byte_matched,
            "R2_cache_budget": r2.cache_budget,
            "R3_cache_budget": r3.cache_budget,
            "R2_top_k_fp": r2.top_k_fp, "R3_top_k_fp": r3.top_k_fp,
            "R3_N_q": r3.n_q,
        },
        "arrays": arrays,
    }
    result["report"] = render(matrix, result)
    return result


def render(matrix, result: Dict[str, Any]) -> str:
    d = result["diagnostics"]
    primary = result["knobs"]["primary_policy"]
    lines = [
        f"## {METRIC_TITLE} ({METRIC_QUESTION})",
        "",
        f"Capacities: `R2` keeps {d['R2_top_k_fp']} fp16 windows; `R3` keeps "
        f"{d['R3_top_k_fp']} fp16 + {d['R3_N_q']} int4 "
        f"(cache_budget {d['R2_cache_budget']} vs {d['R3_cache_budget']} — "
        f"{'byte-matched' if d['byte_matched'] else '**NOT** byte-matched'}).",
        "",
        f"**(a) Simulated over R0's ground-truth masses** (primary policy: "
        f"`{primary}`, matching the production per-flush re-rank):",
        "",
        "| quantity | policy | mean | at flush points | reading |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in result["mass_table"]:
        lines.append(
            f"| `{r['quantity']}` | {r['policy']} | "
            f"{_ci_num(r['mean'], r['ci_lower'], r['ci_upper'], 4)} | "
            f"{_ci_num(r['mean_at_flushes'], r['flush_ci_lower'], r['flush_ci_upper'], 4)} | "
            f"{r['note']} |")
    lines += [
        "",
        f"**(b) Measured Future Missed Mass** (H={result['knobs']['horizon']}):",
        "",
        "| run | set | FMM | flushes scored | units kept |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in result["fmm_table"]:
        lines.append(
            f"| `{r['condition']}` | `{r['accessible_set']}` | "
            f"{_ci_pct(r['future_missed_mass_mean'], r['ci_lower'], r['ci_upper'])} | "
            f"{r['flushes_scored']}/{r['flushes_total']} | "
            f"{r['mean_alive_units']:.1f} |")
    lines += ["", "**Paired** (positive = the compared set misses less):", ""]
    for p in result["paired_table"]:
        lines.append(
            f"- `{p['reference_condition']}` -> `{p['compared_condition']}`: "
            f"{_pct(p['reference_mean'])} -> {_pct(p['compared_mean'])}, "
            f"reduction {_ci_pct(p['absolute_reduction'], p['ci_lower'], p['ci_upper'])} "
            f"({_pct(p['relative_reduction'])} relative) over "
            f"{p['paired_traces']} traces  _({p['note']})_.")
    return "\n".join(lines)


def write(matrix, result: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    meta = {**matrix.metadata_block(result["runs_used"]),
            "metric": METRIC_ID, "metric_name": METRIC_NAME,
            "question": METRIC_QUESTION, "knobs": result["knobs"]}
    return write_metric_outputs(
        out_dir, STEM,
        tables=(("mass_recovery", result["mass_table"]),
                ("future_missed_mass", result["fmm_table"]),
                ("paired", result["paired_table"])),
        payload={k: result[k] for k in
                 ("metric_id", "metric_name", "title", "question", "runs_used",
                  "knobs", "mass_table", "fmm_table", "paired_table",
                  "diagnostics")},
        arrays=result["arrays"], meta=meta, report=result["report"])


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_matrix_args(ap, POLICY_RUNS)
    ap.add_argument("--fmm-horizon", type=int, default=DEFAULTS["horizon"])
    ap.add_argument("--sticky-primary", action="store_true",
                    help="Report Sticky-K as the primary simulated policy "
                         "(default: Fresh-K, which is what the production "
                         "two-tier eviction does).")
    args = ap.parse_args(argv)
    matrix = matrix_from_args(args, POLICY_RUNS)
    result = compute(matrix, horizon=args.fmm_horizon, sticky=args.sticky_primary,
                     confidence=args.confidence, n_boot=args.bootstrap_samples,
                     seed=args.seed, per_head=not args.no_per_head)
    write(matrix, result, args.output_dir)
    print("\n".join(matrix_header_lines(matrix, result["runs_used"])))
    print()
    print(result["report"])


if __name__ == "__main__":
    main()
