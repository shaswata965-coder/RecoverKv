"""M3 — Tier mass distribution and boundary sharpness (*why three tiers*).

Reads ``R0`` (ground truth), ``R2`` (evict-only ws=8) and ``R3`` (three-tier
ws=8).

At every flush the windows of that flush are ranked by ground-truth importance
and cut into the three groups the run's own capacities define::

    rank 0 … k_fp-1              -> fp        (ultra-important, kept in fp16)
    rank k_fp … k_fp+N_q-1       -> Q         (semi-important, kept in int2)
    rank k_fp+N_q … ew-1         -> evicted   (throwaway)

``k_fp`` / ``N_q`` come from the run's resolved config exactly as
``EvictionPolicy.tier_counts`` computes them (clamped to the evictable band), so
``R2`` — which has no Q tier — cuts the same ranking in *two*.  That contrast is
the metric: the mass ``R3`` routes to int2 is precisely the mass ``R2`` throws
away.

Shares alone cannot make the argument, though: any monotone score has a "top
slice" and a "rest", so three shares would look the same whether importance
falls off a cliff at the tier boundaries or decays smoothly.  So each boundary
also gets a **sharpness**:

Two boundaries are scored: ``fp -> Q`` (internal to a three-tier run) and
``alive -> evicted`` (the last window kept vs the first dropped, which exists in
both runs — at rank ``k_fp`` for the evict-only run and at ``k_fp + N_q`` for the
three-tier one, so it is the cut the two designs actually place differently).

``gap_ratio``
    ``s[b-1] / s[b]`` — how much stronger the last window inside the tier is
    than the first one outside.
``sharpness_ratio``
    the drop at the boundary divided by the *median* adjacent-rank drop across
    the band.  ``~1`` means the boundary is an arbitrary cut through a smooth
    curve; ``>> 1`` means the ranking really does have regimes.
``mean_score_ratio_<tier>``
    each bucket's mean per-window score over the band mean — the three-regime
    picture in one number per tier.

Per head
--------
The bucket *sizes* come from the run; the *ranking* is evaluated per head
against the same capacities, so ``…_per_layer_head`` answers "does this head's
mass split three ways at the layer's tier boundaries too?".

CLI::

    python -m modules.evaluation.tier_study.m3_tier_mass_distribution \\
        --r0-npz ... --r2-npz ... --r3-npz ... --output-dir outputs/tier_study
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.evaluation.tier_study import (
    _ci_pct,
    _num,
    _pct,
    add_matrix_args,
    bootstrap_row,
    matrix_from_args,
    matrix_header_lines,
    per_trace_to_layer,
    per_trace_to_layer_series,
    warn_vacuous,
    write_metric_outputs,
)
from utils import qevict_metrics as QM
from utils.logger import get_logger

log = get_logger(__name__)

METRIC_ID = "m3"
METRIC_NAME = "tier_mass_distribution"
METRIC_TITLE = "M3 — Tier mass distribution"
METRIC_QUESTION = "why three tiers"
STEM = f"{METRIC_ID}_{METRIC_NAME}"
RUNS: Tuple[str, ...] = ("R0", "R2", "R3")
POLICY_RUNS: Tuple[str, ...] = ("R2", "R3")

DEFAULTS: Dict[str, Any] = {}

#: Per-event quantities produced by :func:`bucket_event_masses`.
SERIES = (
    "share_fp", "share_q", "share_evicted", "share_local",
    "band_share_fp", "band_share_q", "band_share_evicted",
    "mean_score_ratio_fp", "mean_score_ratio_q", "mean_score_ratio_evicted",
    "gap_ratio_fp_q", "gap_ratio_alive_evicted",
    "sharpness_ratio_fp_q", "sharpness_ratio_alive_evicted",
    "k_fp", "n_q", "band_windows",
)


def bucket_event_masses(scores: np.ndarray, view) -> Dict[str, np.ndarray]:
    """Rank-and-bucket every flush of one condition → ``{series: [M, R]}``.

    ``scores`` is ``[M, R, W]`` ground-truth importance at each flush (the
    cumulative head-mean signal the policy ranks on, or one head's own).
    """
    M, R, _ = scores.shape
    out = {k: np.full((M, R), np.nan) for k in SERIES}
    for r, t in enumerate(view.event_steps):
        ew = int(view.ew_act[int(t)])
        wv = int(view.w_act[int(t)])
        if ew <= 0:
            continue
        band = scores[:, r, :ew]
        s = -np.sort(-band, axis=-1)                       # descending [M, ew]
        # EvictionPolicy.tier_counts: both counts are clamped to the band.
        k_fp = min(view.top_k_fp, ew)
        n_q = min(view.n_q, ew - k_fp)
        local = scores[:, r, ew:wv].sum(axis=-1) if wv > ew else np.zeros(M)

        fp = s[:, :k_fp].sum(axis=-1)
        q = s[:, k_fp:k_fp + n_q].sum(axis=-1)
        ev = s[:, k_fp + n_q:].sum(axis=-1)
        band_total = s.sum(axis=-1)
        total = band_total + local
        ok_t, ok_b = total > 0, band_total > 0

        def _frac(num, den, ok):
            return np.divide(num, den, out=np.full(M, np.nan), where=ok)

        out["share_fp"][:, r] = _frac(fp, total, ok_t)
        out["share_q"][:, r] = _frac(q, total, ok_t)
        out["share_evicted"][:, r] = _frac(ev, total, ok_t)
        out["share_local"][:, r] = _frac(local, total, ok_t)
        out["band_share_fp"][:, r] = _frac(fp, band_total, ok_b)
        out["band_share_q"][:, r] = _frac(q, band_total, ok_b)
        out["band_share_evicted"][:, r] = _frac(ev, band_total, ok_b)

        # Per-window mean score of each bucket, over the band's mean window.
        band_mean = _frac(band_total, np.full(M, float(ew)), ok_b)
        for name, mass, n in (("fp", fp, k_fp), ("q", q, n_q),
                              ("evicted", ev, ew - k_fp - n_q)):
            if n > 0:
                out[f"mean_score_ratio_{name}"][:, r] = _frac(
                    mass / n, band_mean, ok_b & (band_mean > 0))

        # Boundary sharpness: the drop at the cut vs the typical adjacent drop.
        drops = -np.diff(s, axis=-1)                        # >= 0, [M, ew-1]
        median_drop = (np.median(drops, axis=-1) if drops.shape[-1] else
                       np.zeros(M))
        # "fp -> Q" only exists with a middle tier; "alive -> evicted" is the
        # last-kept/first-dropped cut, which both designs have (at a different
        # rank) and is therefore the comparable one.
        for name, b, live in (("fp_q", k_fp, n_q > 0),
                              ("alive_evicted", k_fp + n_q,
                               ew - k_fp - n_q > 0)):
            if not live or b <= 0 or b >= ew:
                continue
            inside, outside = s[:, b - 1], s[:, b]
            out[f"gap_ratio_{name}"][:, r] = np.divide(
                inside, outside, out=np.full(M, np.nan), where=outside > 0)
            out[f"sharpness_ratio_{name}"][:, r] = np.divide(
                inside - outside, median_drop,
                out=np.full(M, np.nan), where=median_drop > 0)

        out["k_fp"][:, r] = k_fp
        out["n_q"][:, r] = n_q
        out["band_windows"][:, r] = ew
    return out


def compute(
    matrix,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    per_head: bool = True,
) -> Dict[str, Any]:
    """Compute M3 over a loaded run matrix."""
    present = [c for c in POLICY_RUNS if c in matrix.conditions]
    if not present:
        raise KeyError(f"{METRIC_TITLE} needs at least one of {POLICY_RUNS}")
    matrix.require(*present)

    rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}

    for ci, cid in enumerate(present):
        view = matrix.conditions[cid]
        # Slice to the flush axis BEFORE bucketing: bucket_event_masses indexes
        # its input by the event index r while taking the window geometry from
        # the decode step t, so handing it the full [M, T, W] array ranks the
        # mass at step r against the geometry at step t (for ws=8 that is steps
        # 0..5 scored against flushes at 1, 9, 17, ...).
        series = bucket_event_masses(
            matrix.mass_cum(cid)[:, view.event_steps, :], view)
        if not np.isfinite(series["share_fp"]).any():
            warn_vacuous(cid, METRIC_TITLE,
                         "no flush has a non-empty evictable band (the whole "
                         "history fits inside the local tail)")
        if view.n_q == 0:
            log.warning(
                "%s has no Q tier (N_q=0) — its Q share is a structural zero, "
                "which is the ablation, not a measurement of the tier.", cid)

        stats: Dict[str, Dict[str, Any]] = {}
        for si, name in enumerate(SERIES):
            trace_vals = QM.nanmean(series[name], axis=1)          # [M]
            mu, lo, hi = bootstrap_row(
                trace_vals, matrix, confidence=confidence, n_boot=n_boot,
                seed=seed + 101 * ci + si)
            stats[name] = {"mean": mu, "ci_lower": lo, "ci_upper": hi}
            arrays[f"{name}__{cid}"] = series[name].astype(np.float32)
            arrays[f"{name}_per_layer__{cid}"] = per_trace_to_layer(
                trace_vals, matrix)

        arrays[f"event_steps__{cid}"] = view.event_steps
        rows.append({
            "condition": cid,
            "label": view.label,
            "tiers": view.spec.tiers,
            "window_size": view.window_size,
            "has_q_tier": view.n_q > 0,
            "top_k_fp": view.top_k_fp,
            "N_q": view.n_q,
            "mean_band_windows": stats["band_windows"]["mean"],
            "share_fp": stats["share_fp"]["mean"],
            "share_fp_ci_lower": stats["share_fp"]["ci_lower"],
            "share_fp_ci_upper": stats["share_fp"]["ci_upper"],
            "share_q": stats["share_q"]["mean"],
            "share_q_ci_lower": stats["share_q"]["ci_lower"],
            "share_q_ci_upper": stats["share_q"]["ci_upper"],
            "share_evicted": stats["share_evicted"]["mean"],
            "share_evicted_ci_lower": stats["share_evicted"]["ci_lower"],
            "share_evicted_ci_upper": stats["share_evicted"]["ci_upper"],
            "share_local": stats["share_local"]["mean"],
            "band_share_fp": stats["band_share_fp"]["mean"],
            "band_share_q": stats["band_share_q"]["mean"],
            "band_share_evicted": stats["band_share_evicted"]["mean"],
            "mean_score_ratio_fp": stats["mean_score_ratio_fp"]["mean"],
            "mean_score_ratio_q": stats["mean_score_ratio_q"]["mean"],
            "mean_score_ratio_evicted": stats["mean_score_ratio_evicted"]["mean"],
            "flushes": view.num_events,
        })
        for bname, label in (("fp_q", "fp -> Q"),
                             ("alive_evicted", "alive -> evicted")):
            boundary_rows.append({
                "condition": cid,
                "boundary": label,
                "boundary_rank": (view.top_k_fp if bname == "fp_q"
                                  else view.top_k_fp + view.n_q),
                "defined": bool(np.isfinite(stats[f"gap_ratio_{bname}"]["mean"])),
                "gap_ratio": stats[f"gap_ratio_{bname}"]["mean"],
                "gap_ratio_ci_lower": stats[f"gap_ratio_{bname}"]["ci_lower"],
                "gap_ratio_ci_upper": stats[f"gap_ratio_{bname}"]["ci_upper"],
                "sharpness_ratio": stats[f"sharpness_ratio_{bname}"]["mean"],
                "sharpness_ratio_ci_lower":
                    stats[f"sharpness_ratio_{bname}"]["ci_lower"],
                "sharpness_ratio_ci_upper":
                    stats[f"sharpness_ratio_{bname}"]["ci_upper"],
                "note": ("" if view.n_q > 0 else
                         "no Q tier in this run: 'fp -> Q' does not exist, and "
                         "'alive -> evicted' sits directly below the fp tier"),
            })

        # ── per-(layer, head) ────────────────────────────────────────────
        if per_head:
            log.info("%s: per-head tier shares over %d head(s)", cid,
                     matrix.num_heads)
            ph = {k: np.full((matrix.num_layers, matrix.num_heads), np.nan)
                  for k in SERIES}
            phs = {k: np.full((matrix.num_layers, matrix.num_heads,
                               view.num_events), np.nan, dtype=np.float32)
                   for k in SERIES}
            for j, h in enumerate(matrix.head_ids):
                s_h = bucket_event_masses(
                    matrix.mass_cum(cid, head=int(h))[:, view.event_steps, :],
                    view)
                for name in SERIES:
                    ph[name][:, j] = per_trace_to_layer(
                        QM.nanmean(s_h[name], axis=1), matrix)
                    phs[name][:, j, :] = per_trace_to_layer_series(
                        s_h[name], matrix)
            for name in SERIES:
                arrays[f"{name}_per_layer_head__{cid}"] = ph[name]
                arrays[f"{name}_per_layer_head_step__{cid}"] = phs[name]

    arrays["layer_ids"] = matrix.layer_ids
    arrays["head_ids"] = matrix.head_ids
    arrays["trace_group"] = matrix.trace_group

    knobs = {"confidence": confidence, "bootstrap_samples": int(n_boot),
             "seed": int(seed), "per_head": bool(per_head)}
    result = {
        "metric_id": METRIC_ID, "metric_name": METRIC_NAME,
        "title": METRIC_TITLE, "question": METRIC_QUESTION,
        "runs_used": [c for c in RUNS if c in matrix.conditions],
        "knobs": knobs, "summary_table": rows, "boundary_table": boundary_rows,
        "arrays": arrays,
    }
    result["report"] = render(matrix, result)
    return result


def render(matrix, result: Dict[str, Any]) -> str:
    lines = [
        f"## {METRIC_TITLE} ({METRIC_QUESTION})",
        "",
        "Ground-truth importance at each flush, cut by the run's own tier "
        "capacities. Shares are of the total mass over all valid windows "
        "(`fp + Q + evicted + local = 1`).",
        "",
        "| run | k_fp | N_q | fp share | Q share | evicted share | local share | band windows |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["summary_table"]:
        lines.append(
            f"| `{r['condition']}` {r['label']} | {r['top_k_fp']} | {r['N_q']} | "
            f"{_ci_pct(r['share_fp'], r['share_fp_ci_lower'], r['share_fp_ci_upper'])} | "
            f"{_ci_pct(r['share_q'], r['share_q_ci_lower'], r['share_q_ci_upper'])} | "
            f"{_ci_pct(r['share_evicted'], r['share_evicted_ci_lower'], r['share_evicted_ci_upper'])} | "
            f"{_pct(r['share_local'])} | {_num(r['mean_band_windows'], 1)} |")
    lines += [
        "",
        "Per-window mean score, relative to the band's mean window "
        "(the three-regime picture):",
        "",
        "| run | fp | Q | evicted |",
        "| --- | --- | --- | --- |",
    ]
    def _ratio_cell(v: Any) -> str:
        return "NA" if not np.isfinite(float(v)) else f"{float(v):.2f}x"

    for r in result["summary_table"]:
        lines.append(
            f"| `{r['condition']}` | {_ratio_cell(r['mean_score_ratio_fp'])} | "
            f"{_ratio_cell(r['mean_score_ratio_q'])} | "
            f"{_ratio_cell(r['mean_score_ratio_evicted'])} |")
    lines += [
        "",
        "**Boundary sharpness** — `gap_ratio` is the score of the last window "
        "inside over the first outside; `sharpness_ratio` is that drop over the "
        "median adjacent-rank drop (`~1` = an arbitrary cut through a smooth "
        "curve, `>> 1` = a real regime change).",
        "",
        "| run | boundary | gap ratio | sharpness ratio |",
        "| --- | --- | --- | --- |",
    ]
    for b in result["boundary_table"]:
        if not b["defined"]:
            lines.append(
                f"| `{b['condition']}` | {b['boundary']} | NA | NA |")
            continue
        lines.append(
            f"| `{b['condition']}` | {b['boundary']} | "
            f"{_num(b['gap_ratio'], 3)} | "
            f"{_num(b['sharpness_ratio'], 2)} |")
    if any(not r["has_q_tier"] for r in result["summary_table"]):
        lines += [
            "",
            "> The evict-only run's Q share is a **structural zero** — it has no "
            "middle tier by construction. That is the ablation being made, not a "
            "measurement of an int2 tier that underperformed.",
        ]
    return "\n".join(lines)


def write(matrix, result: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    meta = {**matrix.metadata_block(result["runs_used"]),
            "metric": METRIC_ID, "metric_name": METRIC_NAME,
            "question": METRIC_QUESTION, "knobs": result["knobs"]}
    return write_metric_outputs(
        out_dir, STEM,
        tables=(("summary", result["summary_table"]),
                ("boundary", result["boundary_table"])),
        payload={k: result[k] for k in
                 ("metric_id", "metric_name", "title", "question", "runs_used",
                  "knobs", "summary_table", "boundary_table")},
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
