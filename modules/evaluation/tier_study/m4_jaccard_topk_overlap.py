"""M4 — Alive-set overlap with the oracle, over the run (*why three tiers*).

Reads ``R0`` (ground truth), ``R2`` (evict-only ws=8) and ``R3`` (three-tier
ws=8).

At every flush each run has an **alive set** — the evictable windows it can still
answer from: ``fp`` for the evict-only run, ``fp ∪ Q`` for the three-tier run.
The oracle set is R0's genuinely most important windows of the same band, taken
at the *same size* as the alive set being scored (``faithfulness_runner._base_top_ids``),
so a run is never rewarded or punished for holding more windows than it does.
Overlap is ``_jaccard_sets`` — the tier sets vary in size from flush to flush, and
``utils.metrics.jaccard_topk`` is a fixed-``K`` kernel, so the variable-size set
op is the right primitive here.

The claim this metric tests is *sustained* overlap, not a favourable average:
``fp`` alone decays as history accumulates and the fp budget stops covering the
important band, while ``fp ∪ Q`` should hold. So the trajectory is the result —
the per-flush series, its early/late halves, and the fraction of flushes on which
the three-tier run leads — and the mean is only a summary of it.

The local recency tail is excluded from both sides: every policy keeps it
unconditionally, so including it would drive every overlap toward 1.

Per head
--------
The alive set is one per layer; the oracle is recomputed from **each head's own**
ground-truth scores, so ``…_per_layer_head`` asks whether the layer's shared
choice also matches what that head would have wanted.

CLI::

    python -m modules.evaluation.tier_study.m4_jaccard_topk_overlap \\
        --r0-npz ... --r2-npz ... --r3-npz ... --output-dir outputs/tier_study
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.evaluation.faithfulness_runner import _base_top_ids, _jaccard_sets
from modules.evaluation.tier_study import (
    _ci_num,
    _num,
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
from utils.logger import get_logger

log = get_logger(__name__)

METRIC_ID = "m4"
METRIC_NAME = "jaccard_topk_overlap"
METRIC_TITLE = "M4 — Alive-set / oracle Jaccard overlap"
METRIC_QUESTION = "why three tiers"
STEM = f"{METRIC_ID}_{METRIC_NAME}"
RUNS: Tuple[str, ...] = ("R0", "R2", "R3")
POLICY_RUNS: Tuple[str, ...] = ("R2", "R3")

DEFAULTS: Dict[str, Any] = {}


def _alive_sets(view) -> List[Tuple[str, np.ndarray]]:
    """``(name, [M, R, W] bool)`` alive-set variants, restricted to the band."""
    band = np.broadcast_to(view.band_mask()[None], view.acc_alive.shape)
    if view.n_q > 0:
        return [("fp_plus_q", view.acc_alive & band),
                ("fp_only", view.acc_fp & band)]
    return [("fp_only", view.acc_fp & band)]


def overlap_trajectory(alive: np.ndarray, scores: np.ndarray, view
                       ) -> Tuple[np.ndarray, int]:
    """``([M, R] Jaccard, n_undefined)`` for one alive set against the oracle.

    ``scores`` is ``[M, R, W]`` ground-truth importance at each flush.  A flush
    where the run holds no evictable window at all is scored NaN rather than the
    both-empty ``1.0`` convention — an empty set trivially "matching" an empty
    oracle would read as perfect fidelity.
    """
    M, R, _ = alive.shape
    out = np.full((M, R), np.nan)
    undefined = 0
    for r, t in enumerate(view.event_steps):
        ew = int(view.ew_act[int(t)])
        if ew <= 0:
            undefined += M
            continue
        for m in range(M):
            ours = set(np.flatnonzero(alive[m, r, :ew]).tolist())
            if not ours:
                undefined += 1
                continue
            out[m, r] = _jaccard_sets(
                ours, _base_top_ids(scores[m, r], ew, len(ours)))
    return out, undefined


def _halves(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a ``[M, R]`` trajectory into early / late per-trace means."""
    R = series.shape[1]
    half = R // 2
    if half == 0:
        nan = np.full(series.shape[0], np.nan)
        return nan, nan.copy()
    return (QM.nanmean(series[:, :half], axis=1),
            QM.nanmean(series[:, half:], axis=1))


def compute(
    matrix,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    per_head: bool = True,
) -> Dict[str, Any]:
    """Compute M4 over a loaded run matrix."""
    present = [c for c in POLICY_RUNS if c in matrix.conditions]
    if not present:
        raise KeyError(f"{METRIC_TITLE} needs at least one of {POLICY_RUNS}")
    matrix.require(*present)

    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    per_trace: Dict[Tuple[str, str], np.ndarray] = {}
    trajectories: Dict[Tuple[str, str], np.ndarray] = {}

    for ci, cid in enumerate(present):
        view = matrix.conditions[cid]
        scores = matrix.mass_cum(cid)                       # [M, T, W]
        scores_ev = scores[:, view.event_steps, :]          # [M, R, W]
        for si, (set_name, alive) in enumerate(_alive_sets(view)):
            jac, undefined = overlap_trajectory(alive, scores_ev, view)
            if not np.isfinite(jac).any():
                warn_vacuous(
                    f"{cid}/{set_name}", METRIC_TITLE,
                    "no flush has both a non-empty evictable band and a "
                    "non-empty alive set")
            trace_vals = QM.nanmean(jac, axis=1)
            per_trace[(cid, set_name)] = trace_vals
            trajectories[(cid, set_name)] = jac
            mu, lo, hi = bootstrap_row(trace_vals, matrix, confidence=confidence,
                                       n_boot=n_boot, seed=seed + 31 * ci + si)
            early, late = _halves(jac)
            e_mu, e_lo, e_hi = bootstrap_row(
                early, matrix, confidence=confidence, n_boot=n_boot,
                seed=seed + 300 + 31 * ci + si)
            l_mu, l_lo, l_hi = bootstrap_row(
                late, matrix, confidence=confidence, n_boot=n_boot,
                seed=seed + 400 + 31 * ci + si)
            c_mu, c_lo, c_hi = QM.bootstrap_curve_ci(
                jac, confidence, n_boot, seed + 1000 + 31 * ci + si)

            key = f"{cid}__{set_name}"
            arrays[f"jaccard_event__{key}"] = jac.astype(np.float32)
            arrays[f"jaccard_per_trace__{key}"] = trace_vals
            arrays[f"jaccard_per_layer__{key}"] = per_trace_to_layer(
                trace_vals, matrix)
            arrays[f"jaccard_curve_mean__{key}"] = c_mu
            arrays[f"jaccard_curve_ci_lower__{key}"] = c_lo
            arrays[f"jaccard_curve_ci_upper__{key}"] = c_hi

            rows.append({
                "condition": cid,
                "label": view.label,
                "tiers": view.spec.tiers,
                "window_size": view.window_size,
                "alive_set": set_name,
                "jaccard_mean": mu,
                "ci_lower": lo,
                "ci_upper": hi,
                "jaccard_first_half": e_mu,
                "first_half_ci_lower": e_lo,
                "first_half_ci_upper": e_hi,
                "jaccard_second_half": l_mu,
                "second_half_ci_lower": l_lo,
                "second_half_ci_upper": l_hi,
                "decay_second_minus_first": (l_mu - e_mu
                                             if np.isfinite(l_mu) and
                                             np.isfinite(e_mu) else np.nan),
                "mean_alive_units": float(alive.sum(axis=-1).mean()),
                "flushes": view.num_events,
                "undefined_flush_cells": int(undefined),
                "is_primary": si == 0,
            })

        arrays[f"event_steps__{cid}"] = view.event_steps

        # ── per-(layer, head): the layer's alive set vs each head's oracle ──
        if per_head:
            log.info("%s: per-head oracle overlap over %d head(s)", cid,
                     matrix.num_heads)
            name, alive = _alive_sets(view)[0]
            ph = np.full((matrix.num_layers, matrix.num_heads), np.nan)
            phs = np.full((matrix.num_layers, matrix.num_heads,
                           view.num_events), np.nan, dtype=np.float32)
            for j, h in enumerate(matrix.head_ids):
                s_h = matrix.mass_cum(cid, head=int(h))[:, view.event_steps, :]
                jac_h, _ = overlap_trajectory(alive, s_h, view)
                ph[:, j] = per_trace_to_layer(QM.nanmean(jac_h, axis=1), matrix)
                phs[:, j, :] = per_trace_to_layer_series(jac_h, matrix)
            arrays[f"jaccard_per_layer_head__{cid}"] = ph
            arrays[f"jaccard_per_layer_head_step__{cid}"] = phs
            arrays[f"jaccard_per_layer_from_heads__{cid}"] = QM.nanmean(ph, axis=1)

    # ── the claim: fp+Q sustains overlap that fp alone does not ──────────
    paired: List[Dict[str, Any]] = []
    comparisons = [
        ("R3", "fp_plus_q", "R2", "fp_only",
         "byte-matched across runs: same cache_budget, one run spends it all on "
         "fp16, the other splits it fp16 + int4"),
        ("R3", "fp_plus_q", "R3", "fp_only",
         "within-run: what crediting the int4 tier adds at the same fp capacity "
         "(NOT byte-matched — the fp-only set of a three-tier run is a smaller "
         "budget)"),
    ]
    for i, (a_cid, a_set, b_cid, b_set, note) in enumerate(comparisons):
        if (a_cid, a_set) not in per_trace or (b_cid, b_set) not in per_trace:
            continue
        res = paired_row(per_trace[(a_cid, a_set)], per_trace[(b_cid, b_set)],
                         matrix, confidence=confidence, n_boot=n_boot,
                         seed=seed + 700 + i)
        ja, jb = trajectories[(a_cid, a_set)], trajectories[(b_cid, b_set)]
        lead = np.nan
        if ja.shape == jb.shape:
            ok = np.isfinite(ja) & np.isfinite(jb)
            lead = float((ja[ok] > jb[ok]).mean()) if ok.any() else np.nan
        paired.append({
            "metric": "jaccard_overlap",
            "reference_condition": f"{a_cid}:{a_set}",
            "compared_condition": f"{b_cid}:{b_set}",
            "absolute_lift": res["absolute_reduction"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"],
            "reference_mean": res["reference_mean"],
            "compared_mean": res["compared_mean"],
            "relative_lift": res["relative_reduction"],
            "paired_traces": res["paired_traces"],
            "flush_cells_where_reference_leads": lead,
            "note": note,
        })

    arrays["layer_ids"] = matrix.layer_ids
    arrays["head_ids"] = matrix.head_ids
    arrays["trace_group"] = matrix.trace_group

    knobs = {"confidence": confidence, "bootstrap_samples": int(n_boot),
             "seed": int(seed), "per_head": bool(per_head)}
    result = {
        "metric_id": METRIC_ID, "metric_name": METRIC_NAME,
        "title": METRIC_TITLE, "question": METRIC_QUESTION,
        "runs_used": [c for c in RUNS if c in matrix.conditions],
        "knobs": knobs, "summary_table": rows, "paired_table": paired,
        "arrays": arrays,
    }
    result["report"] = render(matrix, result)
    return result


def render(matrix, result: Dict[str, Any]) -> str:
    lines = [
        f"## {METRIC_TITLE} ({METRIC_QUESTION})",
        "",
        "Jaccard between each run's alive set (evictable band only) and R0's "
        "top-|alive| windows of the same band, at every flush. Halves of the "
        "run are reported because the claim is *sustained* overlap.",
        "",
        "| run | alive set | Jaccard | first half | second half | late - early | units | flushes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["summary_table"]:
        lines.append(
            f"| `{r['condition']}` {r['label']} | `{r['alive_set']}` | "
            f"{_ci_num(r['jaccard_mean'], r['ci_lower'], r['ci_upper'])} | "
            f"{_num(r['jaccard_first_half'])} | {_num(r['jaccard_second_half'])} | "
            f"{_num(r['decay_second_minus_first'], 4)} | "
            f"{r['mean_alive_units']:.1f} | {r['flushes']} |")
    lines += ["", "**Paired comparisons** (positive = the reference overlaps "
              "the oracle more):", ""]
    for p in result["paired_table"]:
        lines.append(
            f"- `{p['reference_condition']}` vs `{p['compared_condition']}`: "
            f"{_num(p['compared_mean'])} -> {_num(p['reference_mean'])}, lift "
            f"{_ci_num(p['absolute_lift'], p['ci_lower'], p['ci_upper'], 4)}; "
            f"the reference leads on {_pct(p['flush_cells_where_reference_leads'])} "
            f"of scored flush cells.  _{p['note']}_")
    return "\n".join(lines)


def write(matrix, result: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    meta = {**matrix.metadata_block(result["runs_used"]),
            "metric": METRIC_ID, "metric_name": METRIC_NAME,
            "question": METRIC_QUESTION, "knobs": result["knobs"]}
    return write_metric_outputs(
        out_dir, STEM,
        tables=(("summary", result["summary_table"]),
                ("paired", result["paired_table"])),
        payload={k: result[k] for k in
                 ("metric_id", "metric_name", "title", "question", "runs_used",
                  "knobs", "summary_table", "paired_table")},
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
