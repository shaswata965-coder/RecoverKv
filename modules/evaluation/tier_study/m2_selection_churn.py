"""M2 — Selection Churn across decision granularities (*why a window*).

Reads ``R1`` (evict-only ws=1), ``R3`` (three-tier ws=8) and ``R4`` (three-tier
ws=32); ``R0`` supplies the ground-truth mass used by the mass-weighted variant.

Churn is the Jaccard **distance** between the sets retained at consecutive flush
points::

    churn(r) = 1 - |S_r ∩ S_{r+1}| / |S_r ∪ S_{r+1}|

``0`` = the flush changed nothing, ``1`` = the two sets are disjoint.  It is
``qevict_metrics.selection_churn`` applied to the run's measured accessible
sets, so it measures decision *stability*: a policy that keeps re-admitting what
it just dropped is thrashing, and thrashing is what a coarser decision unit is
supposed to damp.

Two framings, both reported
---------------------------
* **Per flush** — the headline number, directly comparable to
  ``qevict_observations``.
* **Per decode step** — ``churn x (R-1) / T``.  Flush cadence *is* the
  independent variable here: ws=1 flushes every step and ws=32 every 32nd, so a
  per-flush number alone lets a run look stable purely by deciding less often.

Which set
---------
Primary is the **evictable band** — the windows the policy actually chose among.
The local recency tail is retained unconditionally and slides forward one window
per flush, so including it adds a turnover term that grows as ``window_size``
grows (``local_windows = local_tokens / ws``), which would bias exactly the
comparison this metric exists to make.  The tail-inclusive series is emitted too,
as ``alive_with_local``.

Per head
--------
``compute_retain_window_indices`` emits **one retained set per layer**, so a pure
set metric evaluated per head against that shared set is head-invariant by
construction: ``churn_per_layer_head`` is the layer value broadcast, and the
diagnostics say so rather than presenting H copies as H measurements.  The
per-head signal that *is* real is ``churn_mass_weighted_per_layer_head`` — the
same set transition weighted by each head's own ground-truth mass::

    churn_mass(r) = 1 - Σ_{w ∈ S_r ∩ S_{r+1}} mass_h[w] / Σ_{w ∈ S_r ∪ S_{r+1}} mass_h[w]

Per-head *decisions* would be a policy change and are not implemented here.

CLI::

    python -m modules.evaluation.tier_study.m2_selection_churn \\
        --r0-npz ... --r1-npz ... --r3-npz ... --r4-npz ... \\
        --output-dir outputs/tier_study
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.evaluation.tier_study import (
    _ci_num,
    _num,
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

METRIC_ID = "m2"
METRIC_NAME = "selection_churn"
METRIC_TITLE = "M2 — Selection Churn"
METRIC_QUESTION = "why a window"
STEM = f"{METRIC_ID}_{METRIC_NAME}"
RUNS: Tuple[str, ...] = ("R0", "R1", "R3", "R4")
POLICY_RUNS: Tuple[str, ...] = ("R1", "R3", "R4")

DEFAULTS: Dict[str, Any] = {}

#: (reference, compared, granularity_only, note) — mirrors M1's pairing.
PAIRS: Tuple[Tuple[str, str, bool, str], ...] = (
    ("R4", "R3", True, "granularity only: both three-tier, ws 32 -> 8"),
    ("R3", "R1", False,
     "token-level reference: R1 is evict-only, so tiering co-varies with "
     "granularity — do NOT read this as a granularity effect"),
    ("R4", "R1", False,
     "token-level reference: R1 is evict-only, so tiering co-varies with "
     "granularity — do NOT read this as a granularity effect"),
)


def _selection_sets(view) -> List[Tuple[str, np.ndarray]]:
    """``(name, [M, R, W] bool)`` selection variants for one condition."""
    band = np.broadcast_to(view.band_mask()[None], view.acc_alive.shape)
    return [
        ("alive_band", view.acc_alive & band),        # primary: the decision
        ("alive_with_local", view.acc_alive),         # + unconditional recency tail
    ]


def _mass_weighted_churn(sel: np.ndarray, mass_events: np.ndarray) -> np.ndarray:
    """``[M, R-1]`` mass-weighted Jaccard distance between consecutive sets.

    ``mass_events`` is ``[M, R, W]`` ground-truth mass at each flush; the pair
    ``(r, r+1)`` is weighted by the mass at ``r+1`` (what the kept set is about
    to be asked for).
    """
    if sel.shape[1] < 2:
        return np.zeros((sel.shape[0], 0), dtype=float)
    a, b = sel[:, :-1], sel[:, 1:]
    w = mass_events[:, 1:]
    inter = np.einsum("mrw,mrw->mr", w, (a & b).astype(w.dtype))
    union = np.einsum("mrw,mrw->mr", w, (a | b).astype(w.dtype))
    sim = np.divide(inter, union, out=np.ones_like(inter), where=union > 0)
    return 1.0 - sim


def compute(
    matrix,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    per_head: bool = True,
) -> Dict[str, Any]:
    """Compute M2 over a loaded run matrix."""
    present = [c for c in POLICY_RUNS if c in matrix.conditions]
    if not present:
        raise KeyError(f"{METRIC_TITLE} needs at least one of {POLICY_RUNS}")
    matrix.require(*present)

    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    per_trace: Dict[Tuple[str, str], np.ndarray] = {}

    for cid in present:
        view = matrix.conditions[cid]
        T = matrix.num_steps
        for set_name, sel in _selection_sets(view):
            churn = QM.selection_churn(sel)                       # [M, R-1]
            if churn.shape[1] == 0:
                warn_vacuous(
                    f"{cid}/{set_name}", METRIC_TITLE,
                    f"only {view.num_events} flush point(s) recorded — churn "
                    "needs at least two consecutive flushes")
            elif float(sel.sum(axis=-1).mean()) <= 0:
                warn_vacuous(f"{cid}/{set_name}", METRIC_TITLE,
                             "the retained set is empty at every flush")

            trace_vals = QM.nanmean(churn, axis=1)                 # [M]
            per_trace[(cid, set_name)] = trace_vals
            mu, lo, hi = bootstrap_row(trace_vals, matrix, confidence=confidence,
                                       n_boot=n_boot, seed=seed + 17 * len(rows))
            c_mu, c_lo, c_hi = (
                QM.bootstrap_curve_ci(churn, confidence, n_boot,
                                      seed + 1000 + 17 * len(rows))
                if churn.shape[1] else (np.zeros(0),) * 3)
            transitions = max(view.num_events - 1, 0)
            per_step = (mu * transitions / T if np.isfinite(mu) and T > 0
                        else np.nan)

            key = f"{cid}__{set_name}"
            arrays[f"churn_event__{key}"] = churn.astype(np.float32)
            arrays[f"churn_per_trace__{key}"] = trace_vals
            arrays[f"churn_per_layer__{key}"] = per_trace_to_layer(trace_vals, matrix)
            arrays[f"churn_curve_mean__{key}"] = c_mu
            arrays[f"churn_curve_ci_lower__{key}"] = c_lo
            arrays[f"churn_curve_ci_upper__{key}"] = c_hi

            rows.append({
                "condition": cid,
                "label": view.label,
                "tiers": view.spec.tiers,
                "window_size": view.window_size,
                "selection_set": set_name,
                "granularity_only_comparable": view.spec.granularity_only_comparable,
                "selection_churn_mean": mu,
                "ci_lower": lo,
                "ci_upper": hi,
                "churn_per_decode_step": per_step,
                "flushes": view.num_events,
                "flush_transitions": transitions,
                "mean_retained_units": float(sel.sum(axis=-1).mean()),
                "is_primary": set_name == "alive_band",
            })

        arrays[f"event_steps__{cid}"] = view.event_steps

        # ── per-(layer, head) ────────────────────────────────────────────
        if per_head:
            band = np.broadcast_to(view.band_mask()[None], view.acc_alive.shape)
            sel = view.acc_alive & band
            layer_churn = arrays[f"churn_per_layer__{cid}__alive_band"]
            # Head-invariant by construction (one shared set per layer) — kept
            # so every metric emits the required [L, H] array, and flagged.
            arrays[f"churn_per_layer_head__{cid}"] = np.repeat(
                layer_churn[:, None], matrix.num_heads, axis=1)
            mw = np.full((matrix.num_layers, matrix.num_heads), np.nan)
            # Churn is defined between CONSECUTIVE flushes, so its event axis is
            # R-1 long and indexed by event_steps[1:], not event_steps.
            mws = np.full((matrix.num_layers, matrix.num_heads,
                           max(view.num_events - 1, 0)), np.nan, dtype=np.float32)
            for j, h in enumerate(matrix.head_ids):
                mass_ev = matrix.mass_cum(cid, head=int(h))[:, view.event_steps, :]
                mwc = _mass_weighted_churn(sel, mass_ev)          # [M, R-1]
                mw[:, j] = per_trace_to_layer(QM.nanmean(mwc, axis=1), matrix)
                if mwc.shape[1]:
                    mws[:, j, :] = per_trace_to_layer_series(mwc, matrix)
            arrays[f"churn_mass_weighted_per_layer_head__{cid}"] = mw
            # The raw churn per (layer, head, flush) would be the layer value
            # broadcast over heads (one shared retained set per layer), so the
            # mass-weighted variant is the one carrying real per-head signal —
            # it is what this axis stores.
            arrays[f"churn_mass_weighted_per_layer_head_step__{cid}"] = mws
            arrays[f"churn_mass_weighted_per_layer__{cid}"] = QM.nanmean(mw, axis=1)

    # ── paired comparisons ───────────────────────────────────────────────
    paired: List[Dict[str, Any]] = []
    for i, (ref, cmp, gran_only, note) in enumerate(PAIRS):
        if (ref, "alive_band") not in per_trace or (cmp, "alive_band") not in per_trace:
            continue
        res = paired_row(per_trace[(ref, "alive_band")],
                         per_trace[(cmp, "alive_band")], matrix,
                         confidence=confidence, n_boot=n_boot, seed=seed + 500 + i)
        paired.append({
            "metric": "selection_churn",
            "reference_condition": ref,
            "compared_condition": cmp,
            "reference_window_size": matrix.conditions[ref].window_size,
            "compared_window_size": matrix.conditions[cmp].window_size,
            "granularity_only": gran_only,
            "note": note + "; churn is per FLUSH — the flush cadence differs "
                           "between these runs, see churn_per_decode_step",
            **res,
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
        "diagnostics": {
            "per_head_churn_is_head_invariant": True,
            "per_head_note": (
                "the retained set is shared across heads (the policy ranks "
                "head-mean scores), so churn_per_layer_head is the layer value "
                "broadcast; churn_mass_weighted_per_layer_head is the per-head "
                "signal"),
        },
        "arrays": arrays,
    }
    result["report"] = render(matrix, result)
    return result


def render(matrix, result: Dict[str, Any]) -> str:
    lines = [
        f"## {METRIC_TITLE} ({METRIC_QUESTION})",
        "",
        "`1 - Jaccard` between the sets retained at consecutive flush points, "
        "over the evictable band (the local tail is retained unconditionally "
        "and would add a `window_size`-dependent turnover term).",
        "",
        "| run | window_size | set | churn / flush | churn / decode step | flushes | units kept |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["summary_table"]:
        lines.append(
            f"| `{r['condition']}` {r['label']} | {r['window_size']} | "
            f"`{r['selection_set']}` | "
            f"{_ci_num(r['selection_churn_mean'], r['ci_lower'], r['ci_upper'])} | "
            f"{_num(r['churn_per_decode_step'], 4)} | {r['flushes']} | "
            f"{r['mean_retained_units']:.1f} |")
    lines += ["", "**Paired comparisons** (positive = the compared run's "
              "selection is more stable per flush):", ""]
    for p in result["paired_table"]:
        tag = ("granularity only" if p["granularity_only"]
               else "CONFOUNDED — tiering co-varies")
        lines.append(
            f"- `{p['reference_condition']}` (ws={p['reference_window_size']}) -> "
            f"`{p['compared_condition']}` (ws={p['compared_window_size']}): "
            f"{_num(p['reference_mean'], 4)} -> {_num(p['compared_mean'], 4)}, "
            f"absolute reduction "
            f"{_ci_num(p['absolute_reduction'], p['ci_lower'], p['ci_upper'], 4)} "
            f"over {p['paired_traces']} traces  _({tag})_.")
    lines += [
        "",
        "> Churn is measured **per flush**, and the flush cadence is the "
        "independent variable (every step at ws=1, every 32nd at ws=32). Read "
        "`churn_per_decode_step` alongside it before concluding that a coarser "
        "run is more stable.",
        "> The retained set is shared across heads, so `churn_per_layer_head` is "
        "the layer value broadcast; the per-head signal is "
        "`churn_mass_weighted_per_layer_head`.",
    ]
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
                  "knobs", "summary_table", "paired_table", "diagnostics")},
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
