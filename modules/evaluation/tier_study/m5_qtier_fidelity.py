"""M5 — Q-tier numerical fidelity (*why three tiers*).

Reads ``R0`` (ground truth) and ``R3`` (three-tier ws=8).

M4 credits the three-tier run for the windows its int4 tier still *holds*.  That
credit is only worth anything if what int4 holds is numerically trustworthy, so
this metric takes the same windows and asks how faithful they are: per flush,
per layer, per head, the cosine similarity between

* the **Q tier's own** attention scores, as the compressed run measured them
  through dequantized int4 K/V (``window_scores`` of the ours npz at the columns
  tagged ``tier == 1``), and
* R0's true scores over that same window set (``all_window_ids`` maps survivor
  columns back onto the baseline's window axis).

``_cosine`` from ``faithfulness_runner`` is the kernel — the same one Suite B
uses for its head-mean ``q_tier_fidelity``, here resolved per head and per flush
instead of collapsed to one scalar.  Cosine is scale-invariant, so the magnitude
question is answered separately by ``q_mass_ratio`` (base mass over the Q
windows ÷ the run's own mass over them; ``~1`` = well matched).

What a cosine below 1 contains
------------------------------
Two things, and they cannot be separated from this pair of runs alone:
quantization error in the dequantized K/V, **and** the trajectory divergence that
eviction itself causes — the compressed run's queries attend over a smaller
cache, so even an exact fp16 tier would not reproduce R0's scores bit for bit.
Read it as an upper bound on Q-tier damage, not as a pure quantization error.

CLI::

    python -m modules.evaluation.tier_study.m5_qtier_fidelity \\
        --r0-npz ... --r3-npz ... --output-dir outputs/tier_study
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from modules.evaluation.faithfulness_runner import _cosine
from modules.evaluation.tier_study import (
    TIER_Q,
    _ci_num,
    _num,
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

METRIC_ID = "m5"
METRIC_NAME = "qtier_fidelity"
METRIC_TITLE = "M5 — Q-tier fidelity"
METRIC_QUESTION = "why three tiers"
STEM = f"{METRIC_ID}_{METRIC_NAME}"
RUNS: Tuple[str, ...] = ("R0", "R3")
POLICY_RUNS: Tuple[str, ...] = ("R3",)

DEFAULTS: Dict[str, Any] = {}


def _q_columns(ids: np.ndarray, tier: np.ndarray, num_windows: int
               ) -> Tuple[np.ndarray, np.ndarray]:
    """``(survivor columns, baseline window ids)`` of one flush's Q tier."""
    cols = np.flatnonzero((tier == TIER_Q) & (ids >= 0) & (ids < num_windows))
    return cols, ids[cols]


def fidelity_trajectory(matrix, cid: str, head: Optional[int] = None
                        ) -> Tuple[np.ndarray, np.ndarray, int]:
    """``([M, R] cosine, [M, R] mass ratio, n_empty_q)`` for one head (or mean).

    ``head=None`` uses the head-mean scores on both sides — the same reduction
    ``faithfulness_runner`` applies before its ``q_tier_fidelity``.
    """
    view = matrix.conditions[cid]
    ours_ws, all_ids, all_tier = matrix.ours_window_scores(cid)
    if head is not None and head >= ours_ws.shape[3]:
        log.warning(
            "%s: ours npz records %d head(s) but head %d was requested — the "
            "two runs disagree on the head axis; skipping this head.",
            cid, ours_ws.shape[3], head)
        nan = np.full((matrix.num_traces, view.num_events), np.nan)
        return nan, nan.copy(), 0
    base = matrix.mass_cum(cid, head=head)                  # [M, T, W]
    layer_pos = {int(li): i for i, li in enumerate(matrix.layer_ids)}
    M, R = matrix.num_traces, view.num_events
    cos = np.full((M, R), np.nan)
    ratio = np.full((M, R), np.nan)
    empty = 0
    for r, t in enumerate(view.event_steps):
        t = int(t)
        for m, (s, li) in enumerate(matrix.trace_labels):
            j = layer_pos[int(li)]
            cols, ids = _q_columns(all_ids[s, t, j], all_tier[s, t, j],
                                   view.num_windows)
            if cols.size == 0:
                empty += 1
                continue
            block = np.asarray(ours_ws[s, t, j], dtype=np.float32)   # [H, Wo]
            o = (block[:, cols].mean(axis=0) if head is None
                 else block[head, cols])
            b = base[m, t, ids].astype(np.float32)
            cos[m, r] = float(_cosine(torch.from_numpy(np.ascontiguousarray(o)),
                                      torch.from_numpy(b)))
            o_sum = float(o.sum())
            if o_sum > 1e-8:
                ratio[m, r] = float(b.sum()) / o_sum
    return cos, ratio, empty


def compute(
    matrix,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    per_head: bool = True,
) -> Dict[str, Any]:
    """Compute M5 over a loaded run matrix."""
    present = [c for c in POLICY_RUNS if c in matrix.conditions]
    if not present:
        raise KeyError(f"{METRIC_TITLE} needs {POLICY_RUNS[0]}")
    matrix.require(*present)

    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}

    for ci, cid in enumerate(present):
        view = matrix.conditions[cid]
        if view.n_q <= 0:
            warn_vacuous(cid, METRIC_TITLE,
                         "the run has no Q tier (N_q=0), so there is no int4 "
                         "signal to score")
            rows.append({
                "condition": cid, "label": view.label,
                "window_size": view.window_size, "N_q": view.n_q,
                "cosine_mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "q_mass_ratio": np.nan, "flushes": view.num_events,
                "flush_cells_without_q": view.num_events * matrix.num_traces,
                "mean_q_units": 0.0,
            })
            continue

        cos, ratio, empty = fidelity_trajectory(matrix, cid)
        if not np.isfinite(cos).any():
            warn_vacuous(cid, METRIC_TITLE,
                         "no flush recorded a non-empty Q tier")
        trace_vals = QM.nanmean(cos, axis=1)
        mu, lo, hi = bootstrap_row(trace_vals, matrix, confidence=confidence,
                                   n_boot=n_boot, seed=seed + 41 * ci)
        r_mu, r_lo, r_hi = bootstrap_row(
            QM.nanmean(ratio, axis=1), matrix, confidence=confidence,
            n_boot=n_boot, seed=seed + 200 + 41 * ci)
        c_mu, c_lo, c_hi = QM.bootstrap_curve_ci(
            cos, confidence, n_boot, seed + 1000 + 41 * ci)

        arrays[f"cosine_event__{cid}"] = cos.astype(np.float32)
        arrays[f"cosine_per_trace__{cid}"] = trace_vals
        arrays[f"cosine_per_layer__{cid}"] = per_trace_to_layer(trace_vals, matrix)
        arrays[f"cosine_curve_mean__{cid}"] = c_mu
        arrays[f"cosine_curve_ci_lower__{cid}"] = c_lo
        arrays[f"cosine_curve_ci_upper__{cid}"] = c_hi
        arrays[f"q_mass_ratio_event__{cid}"] = ratio.astype(np.float32)
        arrays[f"q_mass_ratio_per_layer__{cid}"] = per_trace_to_layer(
            QM.nanmean(ratio, axis=1), matrix)
        arrays[f"event_steps__{cid}"] = view.event_steps

        rows.append({
            "condition": cid,
            "label": view.label,
            "window_size": view.window_size,
            "N_q": view.n_q,
            "quant_ratio": view.quant_ratio,
            "cosine_mean": mu,
            "ci_lower": lo,
            "ci_upper": hi,
            "q_mass_ratio": r_mu,
            "q_mass_ratio_ci_lower": r_lo,
            "q_mass_ratio_ci_upper": r_hi,
            "flushes": view.num_events,
            "flush_cells_without_q": int(empty),
            "mean_q_units": float(view.acc_q.sum(axis=-1).mean()),
        })

        if per_head:
            log.info("%s: per-head Q-tier fidelity over %d head(s) x %d flushes",
                     cid, matrix.num_heads, view.num_events)
            ph = np.full((matrix.num_layers, matrix.num_heads), np.nan)
            for jj, h in enumerate(matrix.head_ids):
                cos_h, _, _ = fidelity_trajectory(matrix, cid, head=int(h))
                ph[:, jj] = per_trace_to_layer(QM.nanmean(cos_h, axis=1), matrix)
            arrays[f"cosine_per_layer_head__{cid}"] = ph
            arrays[f"cosine_per_layer_from_heads__{cid}"] = QM.nanmean(ph, axis=1)

    arrays["layer_ids"] = matrix.layer_ids
    arrays["head_ids"] = matrix.head_ids
    arrays["trace_group"] = matrix.trace_group

    knobs = {"confidence": confidence, "bootstrap_samples": int(n_boot),
             "seed": int(seed), "per_head": bool(per_head)}
    result = {
        "metric_id": METRIC_ID, "metric_name": METRIC_NAME,
        "title": METRIC_TITLE, "question": METRIC_QUESTION,
        "runs_used": [c for c in RUNS if c in matrix.conditions],
        "knobs": knobs, "summary_table": rows, "arrays": arrays,
        "diagnostics": {
            "cosine_contains": ("int4 quantization error AND the trajectory "
                                "divergence caused by eviction — an upper bound "
                                "on Q-tier damage, not pure quantization error"),
        },
    }
    result["report"] = render(matrix, result)
    return result


def render(matrix, result: Dict[str, Any]) -> str:
    lines = [
        f"## {METRIC_TITLE} ({METRIC_QUESTION})",
        "",
        "Cosine similarity between the int4 tier's own window scores and R0's "
        "true scores over the same windows — the numerical backing for the "
        "windows M4 credits to the Q tier.",
        "",
        "| run | N_q | cosine | mass ratio (base/ours) | Q units | flushes | cells without Q |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["summary_table"]:
        lines.append(
            f"| `{r['condition']}` {r['label']} | {r['N_q']} | "
            f"{_ci_num(r['cosine_mean'], r['ci_lower'], r['ci_upper'], 4)} | "
            f"{_num(r.get('q_mass_ratio'), 3)} | {_num(r['mean_q_units'], 1)} | "
            f"{r['flushes']} | {r['flush_cells_without_q']} |")
    lines += [
        "",
        "> A cosine below 1 mixes int4 quantization error with the trajectory "
        "divergence eviction itself causes (the compressed run's queries attend "
        "over a smaller cache). Read it as an **upper bound** on Q-tier damage.",
    ]
    return "\n".join(lines)


def write(matrix, result: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    meta = {**matrix.metadata_block(result["runs_used"]),
            "metric": METRIC_ID, "metric_name": METRIC_NAME,
            "question": METRIC_QUESTION, "knobs": result["knobs"]}
    return write_metric_outputs(
        out_dir, STEM,
        tables=(("summary", result["summary_table"]),),
        payload={k: result[k] for k in
                 ("metric_id", "metric_name", "title", "question", "runs_used",
                  "knobs", "summary_table", "diagnostics")},
        arrays=result["arrays"], meta=meta, report=result["report"])


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_matrix_args(ap, POLICY_RUNS)
    args = ap.parse_args(argv)
    matrix = matrix_from_args(args, POLICY_RUNS, keep_ours_scores=True)
    result = compute(matrix, confidence=args.confidence,
                     n_boot=args.bootstrap_samples, seed=args.seed,
                     per_head=not args.no_per_head)
    write(matrix, result, args.output_dir)
    print("\n".join(matrix_header_lines(matrix, result["runs_used"])))
    print()
    print(result["report"])


if __name__ == "__main__":
    main()
