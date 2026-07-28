"""M1 — Future Missed Mass across decision granularities (*why a window*).

Reads ``R0`` (ground truth), ``R1`` (evict-only ws=1), ``R3`` (three-tier ws=8)
and ``R4`` (three-tier ws=32).

At every flush point the run has just decided what to keep.  FMM asks what that
decision costs against the attention that *actually arrives* next::

    FMM = Σ_{t' = t+1}^{t+H} Σ_w mass[t', w] · [w existed at t ∧ w was dropped]
          ───────────────────────────────────────────────────────────────────
          Σ_{t' = t+1}^{t+H} Σ_w mass[t', w] · [w existed at t]

``mass`` is R0's real per-step window mass, re-bucketed to the condition's
``window_size``.  Two properties are load-bearing and come straight from
``qevict_metrics.future_missed_mass``:

* the denominator counts **only windows that existed at the decision**, so
  tokens created after the flush — which no policy could have kept — cannot
  inflate the score; and
* events without a full ``H``-step future are **right-censored** (NaN), not
  scored over a shorter and therefore easier horizon.

Granularity, honestly labelled
------------------------------
``R1`` is evict-only (int4 packing needs an even ``window_size``), so the
R1-vs-R3 gap mixes granularity with tiering.  It is reported as the
*token-level reference* and flagged ``granularity_only = False``; the
granularity-only contrast is ``R4 → R3`` (ws 32 → 8, tiering fixed).

Per head
--------
The retained set is one per layer (the policy ranks head-mean scores), so the
per-head figures score that same shared set against **each head's own** future
mass.  The per-layer figure uses the head-mean per-step mass recorded in fp32 by
``BaseParityRunner``; the per-head signal has to be differenced out of the fp16
cumulative array, so ``…_per_layer`` (exact source) and
``…_per_layer_from_heads`` (differenced) are both emitted rather than one being
quietly presented as the other.

CLI::

    python -m modules.evaluation.tier_study.m1_future_missed_mass \\
        --r0-npz outputs/parity_base_ws1.npz \\
        --r1-npz outputs/parity_ours_ws1_evictonly.npz \\
        --r3-npz outputs/parity_ours_ws8_threetier.npz \\
        --r4-npz outputs/parity_ours_ws32_threetier.npz \\
        --output-dir outputs/tier_study --fmm-horizon 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.evaluation.tier_study import (
    _ci_pct,
    _pct,
    add_matrix_args,
    bootstrap_row,
    matrix_from_args,
    matrix_header_lines,
    paired_row,
    per_trace_to_layer,
    warn_vacuous,
    write_metric_outputs,
)
from utils import qevict_metrics as QM
from utils.logger import get_logger

log = get_logger(__name__)

METRIC_ID = "m1"
METRIC_NAME = "future_missed_mass"
METRIC_TITLE = "M1 — Future Missed Mass"
METRIC_QUESTION = "why a window"
STEM = f"{METRIC_ID}_{METRIC_NAME}"
RUNS: Tuple[str, ...] = ("R0", "R1", "R3", "R4")
POLICY_RUNS: Tuple[str, ...] = ("R1", "R3", "R4")

DEFAULTS: Dict[str, Any] = {"horizon": 32}

#: (reference, compared, granularity_only, note) — reduction = FMM(ref) - FMM(cmp).
PAIRS: Tuple[Tuple[str, str, bool, str], ...] = (
    ("R4", "R3", True,
     "granularity only: both three-tier, ws 32 -> 8"),
    ("R3", "R1", False,
     "token-level reference: R1 is evict-only, so tiering co-varies with "
     "granularity — do NOT read this as a granularity effect"),
    ("R4", "R1", False,
     "token-level reference: R1 is evict-only, so tiering co-varies with "
     "granularity — do NOT read this as a granularity effect"),
)


def _accessible_sets(view) -> List[Tuple[str, np.ndarray]]:
    """The accessible-set variants worth scoring for one condition."""
    sets = [("alive", view.acc_alive)]
    if view.n_q > 0:
        # fp-only is what the same run would have kept without an int4 tier at
        # the SAME fp capacity — a within-run contrast, not a byte-matched one
        # (that is M6's job, against the real evict-only run).
        sets.append(("fp_only", view.acc_fp))
    return sets


def compute(
    matrix,
    *,
    horizon: int = DEFAULTS["horizon"],
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    per_head: bool = True,
) -> Dict[str, Any]:
    """Compute M1 over a loaded run matrix."""
    present = [c for c in POLICY_RUNS if c in matrix.conditions]
    if not present:
        raise KeyError(f"{METRIC_TITLE} needs at least one of {POLICY_RUNS}")
    matrix.require(*present)

    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    per_trace: Dict[Tuple[str, str], np.ndarray] = {}

    for cid in present:
        view = matrix.conditions[cid]
        mass = matrix.mass_step(cid)                      # [M, T, W] head-mean
        exists = np.broadcast_to(view.event_valid()[None], view.acc_alive.shape)
        for set_name, mask in _accessible_sets(view):
            fmm = QM.future_missed_mass(
                mass, mask, view.event_steps, horizon,
                creation_steps=view.creation, require_full_horizon=True)
            scored = int(np.isfinite(fmm).any(axis=0).sum())
            dropped = float(np.sum(exists & ~mask, axis=-1).mean())
            if scored == 0:
                warn_vacuous(
                    f"{cid}/{set_name}", METRIC_TITLE,
                    f"no flush point has a full {horizon}-step future horizon "
                    f"(T={matrix.num_steps}, last flush at trace index "
                    f"{int(view.event_steps[-1])}) — lower --fmm-horizon or "
                    "record more decode steps")
            elif dropped <= 0:
                warn_vacuous(
                    f"{cid}/{set_name}", METRIC_TITLE,
                    "the policy dropped no existing window at any flush, so FMM "
                    "is trivially 0 (check top_k_fp/N_q against W="
                    f"{view.num_windows})")

            trace_vals = QM.nanmean(fmm, axis=1)                     # [M]
            per_trace[(cid, set_name)] = trace_vals
            mu, lo, hi = bootstrap_row(trace_vals, matrix, confidence=confidence,
                                       n_boot=n_boot, seed=seed + 17 * len(rows))
            c_mu, c_lo, c_hi = QM.bootstrap_curve_ci(
                fmm, confidence, n_boot, seed + 1000 + 17 * len(rows))

            key = f"{cid}__{set_name}"
            arrays[f"fmm_event__{key}"] = fmm.astype(np.float32)
            arrays[f"fmm_per_trace__{key}"] = trace_vals
            arrays[f"fmm_per_layer__{key}"] = per_trace_to_layer(trace_vals, matrix)
            arrays[f"fmm_curve_mean__{key}"] = c_mu
            arrays[f"fmm_curve_ci_lower__{key}"] = c_lo
            arrays[f"fmm_curve_ci_upper__{key}"] = c_hi

            rows.append({
                "condition": cid,
                "label": view.label,
                "tiers": view.spec.tiers,
                "window_size": view.window_size,
                "accessible_set": set_name,
                "granularity_only_comparable": view.spec.granularity_only_comparable,
                "horizon": int(horizon),
                "future_missed_mass_mean": mu,
                "ci_lower": lo,
                "ci_upper": hi,
                "flushes_total": view.num_events,
                "flushes_scored": scored,
                "mean_alive_units": float(mask.sum(axis=-1).mean()),
                "mean_dropped_units": dropped,
                "is_primary": set_name == "alive",
            })

        arrays[f"event_steps__{cid}"] = view.event_steps

        # ── per-(layer, head): the shared retained set vs each head's future ──
        if per_head:
            log.info("%s: per-head FMM over %d head(s) x %d flush point(s)",
                     cid, matrix.num_heads, view.num_events)
            ph = np.full((matrix.num_layers, matrix.num_heads), np.nan)
            for j, h in enumerate(matrix.head_ids):
                fmm_h = QM.future_missed_mass(
                    matrix.mass_step(cid, head=int(h)), view.acc_alive,
                    view.event_steps, horizon, creation_steps=view.creation,
                    require_full_horizon=True)
                ph[:, j] = per_trace_to_layer(QM.nanmean(fmm_h, axis=1), matrix)
            arrays[f"fmm_per_layer_head__{cid}"] = ph
            arrays[f"fmm_per_layer_from_heads__{cid}"] = QM.nanmean(ph, axis=1)

    # ── paired comparisons over the shared trace axis ────────────────────
    paired: List[Dict[str, Any]] = []
    for i, (ref, cmp, gran_only, note) in enumerate(PAIRS):
        if (ref, "alive") not in per_trace or (cmp, "alive") not in per_trace:
            continue
        res = paired_row(per_trace[(ref, "alive")], per_trace[(cmp, "alive")],
                         matrix, confidence=confidence, n_boot=n_boot,
                         seed=seed + 500 + i)
        paired.append({
            "metric": "future_missed_mass",
            "reference_condition": ref,
            "compared_condition": cmp,
            "reference_window_size": matrix.conditions[ref].window_size,
            "compared_window_size": matrix.conditions[cmp].window_size,
            "granularity_only": gran_only,
            "note": note,
            **res,
        })

    arrays["layer_ids"] = matrix.layer_ids
    arrays["head_ids"] = matrix.head_ids
    arrays["trace_group"] = matrix.trace_group

    knobs = {"horizon": int(horizon), "confidence": confidence,
             "bootstrap_samples": int(n_boot), "seed": int(seed),
             "per_head": bool(per_head)}
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
    """Markdown section for this metric."""
    h = result["knobs"]["horizon"]
    lines = [
        f"## {METRIC_TITLE} ({METRIC_QUESTION})",
        "",
        f"Fraction of the next `H={h}` decode steps' real attention mass that "
        "lands on windows the run had already dropped. Events without a full "
        "horizon are right-censored; the denominator counts only windows that "
        "existed at the decision.",
        "",
        "| run | window_size | set | FMM | flushes scored | units kept | units dropped |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["summary_table"]:
        lines.append(
            f"| `{r['condition']}` {r['label']} | {r['window_size']} | "
            f"`{r['accessible_set']}` | "
            f"{_ci_pct(r['future_missed_mass_mean'], r['ci_lower'], r['ci_upper'])} | "
            f"{r['flushes_scored']}/{r['flushes_total']} | "
            f"{r['mean_alive_units']:.1f} | {r['mean_dropped_units']:.1f} |")
    lines += ["", "**Paired comparisons** (positive = the compared run misses "
              "less mass):", ""]
    for p in result["paired_table"]:
        tag = ("granularity only" if p["granularity_only"]
               else "CONFOUNDED — tiering co-varies")
        lines.append(
            f"- `{p['reference_condition']}` (ws={p['reference_window_size']}) -> "
            f"`{p['compared_condition']}` (ws={p['compared_window_size']}): "
            f"{_pct(p['reference_mean'])} -> {_pct(p['compared_mean'])}, "
            f"absolute reduction "
            f"{_ci_pct(p['absolute_reduction'], p['ci_lower'], p['ci_upper'])} "
            f"over {p['paired_traces']} traces  _({tag})_.")
    if any(not p["granularity_only"] for p in result["paired_table"]):
        lines += [
            "",
            "> `R1` (ws=1) is the **token-level reference**, not a granularity "
            "arm: it is evict-only, so any R1-vs-R3/R4 gap contains both the "
            "granularity change and the loss of the Q tier. The granularity-only "
            "statement is `R4 -> R3`.",
        ]
    return "\n".join(lines)


def write(matrix, result: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    """Write ``m1_future_missed_mass.{csv,json,npz}``."""
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
    ap.add_argument("--fmm-horizon", type=int, default=DEFAULTS["horizon"],
                    help="H — decode steps of future attention each routing "
                         "decision is scored against.")
    args = ap.parse_args(argv)
    matrix = matrix_from_args(args, POLICY_RUNS)
    result = compute(matrix, horizon=args.fmm_horizon,
                     confidence=args.confidence, n_boot=args.bootstrap_samples,
                     seed=args.seed, per_head=not args.no_per_head)
    write(matrix, result, args.output_dir)
    print("\n".join(matrix_header_lines(matrix, result["runs_used"])))
    print()
    print(result["report"])


if __name__ == "__main__":
    main()
