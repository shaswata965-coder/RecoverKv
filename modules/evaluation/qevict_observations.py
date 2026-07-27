"""QEvict observation suite — the three motivating observations, from real runs.

Where the other suites ask "how faithful is our cache?", this one asks "is the
*design premise* true?" and answers it from the parity npzs we already produce:

  I.   **Skewed importance** — how concentrated is window importance, and where
       do the fp / fp+Q capacity boundaries actually fall on that curve?
  II.  **Window-level decisions** — how much of the attention mass arriving in
       the next ``H`` steps does a routing decision throw away (Future Missed
       Mass), and how unstable is the accessible set (Selection Churn)?  Scored
       for the *measured* fp-only and fp∪Q sets, plus a matched-byte
       fine-vs-coarse decision-granularity pair.
  III. **Historical importance revives** — after a window has been cold for
       ``m`` events, how often does it become important again within ``H``?
       Computed both for the ground-truth oracle set (the upside a promotion
       path could capture) and for our cache's real fp tier (what its Q→fp
       promotion actually recovers), plus the lag-``delta`` flip matrix.

Inputs are the existing ``parity_base`` + ``parity_ours`` npzs — the same pair
:mod:`modules.evaluation.faithfulness_runner` consumes.  Two things to know:

  * **Observation II needs the per-step mass, not the cumulative score.**  A
    base npz at schema >= 1.2 records ``step_window_scores`` (fp32) and the
    suite uses it directly.  Older npzs only have the fp16 *cumulative*
    ``window_scores``; the suite then differences them, warns, and reports
    ``step_mass_negative_fraction`` — but that path is only trustworthy for
    short runs.  At ``gen_len ~ 1000`` one fp16 ulp of the accumulated score
    is comparable to a whole per-step delta, so re-run the base parity rather
    than reporting a differenced Observation II.
  * **Decision granularity is the base run's ``window_size``.**  Observation II
    compares "one decision per unit" against "one decision per block of
    ``pool_factor`` units" at a matched retained-unit count.  With the usual
    ``window_size=8`` base run that is 8-token vs 16-token decisions; for the
    true *token*-vs-window comparison, record a base run with
    ``window.window_size=1`` and pass ``--pool-factor 8``.

CLI::

    python -m modules.evaluation.qevict_observations \\
        --base-npz outputs/parity_base_wikitext-103.npz \\
        --ours-npz outputs/parity_ours_eager_wikitext-103.npz \\
        --output-dir outputs/qevict_observations \\
        --fmm-horizon 32 --primary-inactivity 4 --primary-lir-horizon 8 \\
        --primary-transition-delta 1

Or via ``main.py --config configs/eval_qevict.yaml`` (mode
``qevict_observations``), which reads the npz paths from the ``faithfulness:``
section and uses the CLI defaults for every knob.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import qevict_metrics as QM
from utils import sticky_metrics as SM
from utils.config import ParityValidationError
from utils.hashing import sha256_file
from utils.logger import get_logger

log = get_logger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:                                        # pragma: no cover
    HAS_MPL = False

# Tier codes in ours npz ``all_window_tier`` (ours_parity_runner._extract_row_retained).
TIER_FP, TIER_Q, TIER_LOCAL = 0, 1, 2

DEFAULTS: Dict[str, Any] = {
    "report_fractions": (0.05, 0.10, 0.20, 0.40),
    "target_masses": (0.50, 0.80, 0.90),
    "fmm_horizon": 32,
    "pool_factor": 2,
    "lir_inactivity_values": (1, 2, 4, 8),
    "lir_horizon_values": (1, 2, 4, 8, 16),
    "transition_deltas": (1, 2, 4, 8),
    "primary_inactivity": 4,
    "primary_lir_horizon": 8,
    "primary_transition_delta": 1,
    "primary_top_fraction": 0.10,
    "primary_target_mass": 0.80,
    "confidence": 0.95,
    "bootstrap_samples": 2000,
    "seed": 0,
    "trace_axis": "sample_layer",
}


# ---------------------------------------------------------------------------
# npz loading & observation-input construction
# ---------------------------------------------------------------------------


def resolve_npz(path: str | Path, pattern: str) -> Path:
    """Return ``path``, or fall back to an unambiguous ``pattern`` match.

    The parity runners name their outputs ``parity_<side>_<dataset>_<sha8>.npz``,
    so the sha suffix is not knowable when writing a config.  If the configured
    path is missing, glob its directory and accept a single match — that keeps
    the Kaggle flow (run base, run ours, run observations) working without
    hand-editing paths between cells.
    """
    p = Path(path)
    if p.exists():
        return p
    matches = sorted(p.parent.glob(pattern)) if p.parent.exists() else []
    if len(matches) == 1:
        log.info("%s not found; using the only %s match: %s",
                 p.name, pattern, matches[0].name)
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"{p} not found and {pattern} is ambiguous in {p.parent}: "
            f"{[m.name for m in matches]}. Pass the path explicitly.")
    raise FileNotFoundError(
        f"Parity npz not found: {p} (no {pattern} in {p.parent} either). "
        "Run the parity base + ours suites first.")


def load_parity_npz(path: str | Path) -> dict:
    """Load a parity npz into ``{"arrays": ..., "metadata": ..., "path": ...}``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parity npz not found: {p}")
    data = np.load(str(p), allow_pickle=True)
    meta = json.loads(str(data["metadata_json"][0]))
    arrays = {k: data[k] for k in data.files if k != "metadata_json"}
    return {"arrays": arrays, "metadata": meta, "path": str(p)}


@dataclass
class ObservationInputs:
    """Everything the three observations need, on a shared ``[M, ...]`` axis.

    ``M`` traces are always ``(sample, layer)`` pairs; ``trace_group`` then says
    which traces share a bootstrap group (``trace_axis="sample_layer"`` → one
    group each, ``"sample"`` → grouped by article).  With the usual
    single-article parity run the layer axis is the only axis wide enough to
    bootstrap over, and layers of one prompt are *not* independent samples — so
    read those intervals as within-run variability, not population CIs.
    """

    mass_cum: np.ndarray          # [M, T, W] cumulative head-mean window mass
    mass_step: np.ndarray         # [M, T, W] per-step mass (differenced, clamped)
    valid: np.ndarray             # [M, T, W] bool — window exists at that step
    rank_scores: np.ndarray       # [M, R, W] ranking signal at each event
    event_valid: np.ndarray       # [M, R, W] bool — valid, restricted to events
    event_steps: np.ndarray       # [R] decode step of each routing event
    accessible: Dict[str, np.ndarray]   # name -> [M, R, W] bool
    oracle_selected: np.ndarray   # [M, R, W] trinary, ground-truth important set
    policy_selected: np.ndarray   # [M, R, W] trinary, our real fp tier
    w_act: np.ndarray             # [T] valid windows per step
    ew_act: np.ndarray            # [T] evictable windows per step
    creation: np.ndarray          # [W] first step at which a window exists
    trace_labels: List[Tuple[int, int]]
    trace_group: np.ndarray       # [M] bootstrap group id per trace
    geometry: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _as5d(x: np.ndarray) -> np.ndarray:
    """Promote a legacy ``[T, L, H, W]`` array to ``[1, T, L, H, W]``."""
    return x[None] if x.ndim == 4 else x


def _as4d(x: np.ndarray) -> np.ndarray:
    """Promote a legacy ``[T, L, W]`` array to ``[1, T, L, W]``."""
    return x[None] if x.ndim == 3 else x


def build_observation_inputs(
    base: dict,
    ours: dict,
    *,
    trace_axis: str = "sample_layer",
    oracle_budget: str = "fp",
    pool_factor: int = 2,
    max_samples: Optional[int] = None,
    layer_stride: int = 1,
) -> ObservationInputs:
    """Derive the observation arrays from a base/ours parity npz pair.

    Parameters
    ----------
    base, ours : dict
        Loaded by :func:`load_parity_npz`.  ``base`` supplies the ground-truth
        (never-evicted) window masses; ``ours`` supplies the real accessible
        sets, tier tags and eviction cadence.
    trace_axis : {"sample_layer", "sample", "layer"}
        Which axis becomes the bootstrap axis ``M``.  ``"sample"`` averages over
        layers first, ``"layer"`` averages over samples first.
    oracle_budget : {"fp", "kept"}
        Size of the ground-truth "important set" for Observation III: the fp
        tier alone (``top_k_fp``) or fp+Q (``top_k_fp + N_q``).
    pool_factor : int
        Block size for the coarse policy in Observation II.
    max_samples, layer_stride : int, optional
        Sub-sample the trace axis on memory- or time-constrained machines
        (Kaggle).  ``layer_stride=4`` keeps every 4th layer; the retained layer
        ids stay in ``trace_labels`` so a subsampled run is still labelled
        honestly.  Both default to "use everything".
    """
    ba, oa = base["arrays"], ours["arrays"]
    bm, om = base["metadata"], ours["metadata"]
    for key in ("window_scores",):
        if key not in ba:
            raise KeyError(f"base npz is missing {key!r}")
    for key in ("all_window_ids", "all_window_tier"):
        if key not in oa:
            raise KeyError(
                f"ours npz is missing {key!r} — re-run OursParityRunner "
                "(schema >= 1.2) so the full-survivor tier arrays are recorded."
            )

    ws_sz = int(om.get("window_size", bm.get("window_size", 8)))
    ns = int(om.get("num_sink_tokens", bm.get("num_sink_tokens", 0)))
    prefill_len = int(om.get("prefill_len", bm.get("prefill_len", 0)))
    lr = int(om.get("local_window_size_resolved", 0))
    local_windows = lr // ws_sz if ws_sz > 0 else 0
    top_k_windows = int(om.get("top_k_windows", 0))
    top_k_fp = int(om.get("top_k_fp", top_k_windows))
    n_q = int(om.get("N_q", 0))
    oracle_k = top_k_fp + (n_q if oracle_budget == "kept" else 0)

    base_ws = _as5d(np.asarray(ba["window_scores"]))          # [S, T, L, H, W]
    all_ids = _as4d(np.asarray(oa["all_window_ids"]))         # [S, T, L, Wo]
    all_tier = _as4d(np.asarray(oa["all_window_tier"]))       # [S, T, L, Wo]
    S = min(base_ws.shape[0], all_ids.shape[0])
    T = min(base_ws.shape[1], all_ids.shape[1])
    L_all = min(base_ws.shape[2], all_ids.shape[2])
    W = int(base_ws.shape[4])
    if max_samples is not None:
        S = max(1, min(S, int(max_samples)))
    layer_idx = np.arange(0, L_all, max(1, int(layer_stride)))
    L = int(layer_idx.size)

    # ── ground-truth mass: head-mean, then fold (sample, layer) into traces ──
    # Traces are always (sample, layer) pairs — a coarser bootstrap axis is a
    # *grouping* applied to the per-trace statistics afterwards, because
    # collapsing layers earlier would mean averaging accessible sets.
    cum = np.mean(base_ws[:S, :T][:, :, layer_idx], axis=3, dtype=np.float64)
    traces = cum.transpose(0, 2, 1, 3).reshape(S * L, T, W)        # [S*L, T, W]
    labels = [(s, int(li)) for s in range(S) for li in layer_idx]
    M = traces.shape[0]
    if trace_axis == "sample_layer":
        trace_group = np.arange(M)
    elif trace_axis == "sample":
        trace_group = np.asarray([s for s, _ in labels], dtype=int)
    elif trace_axis == "layer":
        trace_group = np.asarray([li for _, li in labels], dtype=int)
    else:
        raise ValueError(
            f"unknown trace_axis {trace_axis!r}; expected sample_layer|sample|layer")

    # ── per-step mass: recorded if available, else differenced ───────────
    if "step_window_scores" in ba:
        sws = np.asarray(ba["step_window_scores"], dtype=np.float64)  # [S,T,L,W]
        if sws.ndim == 3:
            sws = sws[None]
        sws = sws[:S, :T][:, :, layer_idx]
        if sws.shape[-1] < W:
            sws = np.pad(sws, [(0, 0), (0, 0), (0, 0), (0, W - sws.shape[-1])])
        mass_step = sws[..., :W].transpose(0, 2, 1, 3).reshape(M, T, W)
        mass_source = "recorded_step_mass"
        neg_frac = float((mass_step < 0).mean()) if mass_step.size else 0.0
        neg_mag = 0.0
        mass_step = np.clip(mass_step, 0.0, None)
    else:
        # Fallback for base npzs written before schema 1.2.  Differencing an
        # fp16 cumulative array is lossy in exactly the regime we care about:
        # by T~1000 one ulp of the accumulated score is comparable to a whole
        # per-step delta, so Observation II degrades to noise.
        step = np.diff(traces, axis=1, prepend=np.zeros((M, 1, W)))
        neg = step < 0
        neg_frac = float(neg.sum()) / float(neg.size) if neg.size else 0.0
        neg_mag = (float(-step[neg].sum()) / float(step[step > 0].sum())
                   if np.any(step > 0) else 0.0)
        mass_step = np.clip(step, 0.0, None)
        mass_source = "differenced_cumulative"
        log.warning(
            "base npz has no 'step_window_scores' (schema %s) — falling back to "
            "differencing the fp16 cumulative scores; %.1f%% of entries came out "
            "negative and were clamped. Re-run BaseParityRunner to record the "
            "per-step mass before trusting Observation II.",
            bm.get("schema_version", "?"), 100 * neg_frac)

    # ── geometry (shared with the Sticky-K suite) ────────────────────────
    w_act, ew_act, creation = SM.flush_geometry(
        T, W, prefill_len, ns, ws_sz, local_windows)
    cols = np.arange(W)[None, :]
    valid_t = cols < w_act[:, None]                           # [T, W]
    valid = np.broadcast_to(valid_t[None], (M, T, W)).copy()

    # ── routing events: the steps at which eviction actually fired ───────
    if "eviction_step_mask" in oa and np.asarray(oa["eviction_step_mask"]).size:
        ev_mask = np.asarray(oa["eviction_step_mask"])
        if ev_mask.ndim == 1:
            ev_mask = ev_mask[None]
        ev_mask = ev_mask[:S, :T].astype(bool)
        event_steps = np.flatnonzero(ev_mask.any(axis=0))
        if ev_mask.any() and not np.array_equal(ev_mask.any(0), ev_mask.all(0)):
            log.warning("eviction_step_mask differs across samples; using the union")
    else:
        event_steps = np.zeros(0, dtype=int)
    if event_steps.size == 0:
        log.warning(
            "No eviction steps in ours npz — treating every decode step as a "
            "routing event (Observation II/III then measure the ranking signal, "
            "not real evictions).")
        event_steps = np.arange(T)
    R = event_steps.size

    # ── real accessible sets + our fp-tier selection, from ours' tier tags ──
    acc_fp = np.zeros((M, R, W), dtype=bool)
    acc_kept = np.zeros((M, R, W), dtype=bool)
    out_of_range = 0
    for r, t in enumerate(event_steps):
        for m, (s, li) in enumerate(labels):
            ids = all_ids[s, t, li]
            tier = all_tier[s, t, li]
            ok = (ids >= 0) & (tier >= 0)
            out_of_range += int(np.sum(ok & (ids >= W)))
            ok &= ids < W
            idx, tr = ids[ok], tier[ok]
            acc_kept[m, r, idx] = True
            acc_fp[m, r, idx[tr != TIER_Q]] = True
    if out_of_range:
        log.warning("%d survivor ids exceeded the base window axis (W=%d); ignored",
                    out_of_range, W)

    # ── trinary selection matrices over the evictable band ───────────────
    ev_w = ew_act[event_steps]                                # [R]
    oracle = np.full((M, R, W), -1, dtype=np.int8)
    policy = np.full((M, R, W), -1, dtype=np.int8)
    rank_scores = traces[:, event_steps, :]                   # [M, R, W]
    for r in range(R):
        ew = int(ev_w[r])
        if ew <= 0:
            continue
        oracle[:, r, :ew] = 0
        policy[:, r, :ew] = 0
        policy[:, r, :ew][acc_fp[:, r, :ew]] = 1
        k = min(oracle_k, ew)
        if k > 0:
            sel = np.argsort(-rank_scores[:, r, :ew], axis=-1, kind="stable")[:, :k]
            np.put_along_axis(oracle[:, r, :ew], sel, 1, axis=-1)

    # ── matched-byte fine vs coarse decision granularity (simulated) ─────
    accessible: Dict[str, np.ndarray] = {
        "measured_fp_only": acc_fp,
        "measured_fp_plus_q": acc_kept,
    }
    pool = max(1, int(pool_factor))
    if top_k_fp >= pool:
        # Round the budget down to whole blocks and give *both* policies that
        # same unit count — otherwise the fine policy silently keeps the
        # remainder units too and the comparison is not byte-matched.
        matched_units = (top_k_fp // pool) * pool
        fine = QM.granularity_accessible(
            rank_scores, event_steps, w_act, ew_act, matched_units, pool=1)
        coarse = QM.granularity_accessible(
            rank_scores, event_steps, w_act, ew_act, matched_units, pool=pool)
        accessible[f"simulated_unit_g{ws_sz}"] = fine
        accessible[f"simulated_block_g{ws_sz * pool}"] = coarse
    else:
        log.warning(
            "top_k_fp=%d < pool_factor=%d — skipping the granularity comparison "
            "(a coarse policy cannot keep even one whole block).", top_k_fp, pool)

    return ObservationInputs(
        mass_cum=traces,
        mass_step=mass_step,
        valid=valid,
        rank_scores=rank_scores,
        event_valid=valid[:, event_steps, :],
        event_steps=event_steps,
        accessible=accessible,
        oracle_selected=oracle,
        policy_selected=policy,
        w_act=w_act,
        ew_act=ew_act,
        creation=creation,
        trace_labels=labels,
        trace_group=trace_group,
        geometry={
            "num_traces": M, "num_groups": int(np.unique(trace_group).size),
            "num_samples": S, "num_layers": L, "num_layers_total": L_all,
            "layer_stride": int(max(1, layer_stride)), "num_steps": T,
            "num_windows": W, "num_events": int(R), "window_size": ws_sz,
            "num_sink_tokens": ns, "prefill_len": prefill_len,
            "local_windows": local_windows, "top_k_windows": top_k_windows,
            "top_k_fp": top_k_fp, "N_q": n_q, "oracle_k": oracle_k,
            "quant_ratio": float(om.get("quant_ratio", 0.0)),
            "pool_factor": pool, "trace_axis": trace_axis,
        },
        diagnostics={
            "mass_source": mass_source,
            "step_mass_negative_fraction": neg_frac,
            "step_mass_negative_magnitude_ratio": neg_mag,
            "survivor_ids_out_of_range": int(out_of_range),
            "base_schema_version": str(bm.get("schema_version", "?")),
            "ours_schema_version": str(om.get("schema_version", "?")),
        },
    )


# ---------------------------------------------------------------------------
# Observation I
# ---------------------------------------------------------------------------


def analyse_skewed_importance(
    inp: ObservationInputs,
    report_fractions: Sequence[float] = DEFAULTS["report_fractions"],
    target_masses: Sequence[float] = DEFAULTS["target_masses"],
    fp_boundary: Optional[float] = None,
    fp_plus_q_boundary: Optional[float] = None,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Concentration curve + captured-mass / required-fraction tables."""
    groups = inp.trace_group
    conc = QM.concentration_curve(
        inp.rank_scores, inp.event_valid, report_fractions, target_masses)
    per_trace_curve = QM.group_reduce(
        QM.nanmean(conc["curve"], axis=1), groups)                   # [G, X]
    mean, lo, hi = QM.bootstrap_curve_ci(per_trace_curve, confidence, n_boot, seed)

    mass_rows = []
    for i, p in enumerate(report_fractions):
        vals = QM.group_reduce(
            QM.nanmean(conc["mass_at"][float(p)], axis=1), groups)
        mu, l, h = QM.bootstrap_mean_ci(vals, confidence, n_boot, seed + i + 1)
        mass_rows.append({
            "top_fraction": float(p), "mean_cumulative_mass": mu,
            "ci_lower": l, "ci_upper": h, "concentration_gap": mu - float(p),
        })
    coverage_rows = []
    for i, q in enumerate(target_masses):
        vals = QM.group_reduce(
            QM.nanmean(conc["coverage_at"][float(q)], axis=1), groups)
        mu, l, h = QM.bootstrap_mean_ci(vals, confidence, n_boot, seed + 100 + i)
        coverage_rows.append({
            "target_mass": float(q), "mean_required_fraction": mu,
            "ci_lower": l, "ci_upper": h,
        })

    # Where the real capacity boundaries fall on this x-axis (fraction of valid
    # windows that stay accessible): fp tier + local tail, then fp+Q + local.
    g = inp.geometry
    n_valid = np.maximum(QM.nanmean(conc["num_valid"].astype(float), axis=0), 1.0)
    ev = np.maximum(inp.ew_act[inp.event_steps].astype(float), 0.0)
    fp_units = np.minimum(g["top_k_fp"], ev) + g["local_windows"]
    q_units = np.minimum(g["top_k_fp"] + g["N_q"], ev) + g["local_windows"]
    auto_fp = float(np.mean(fp_units / n_valid))
    auto_q = float(np.mean(q_units / n_valid))

    curve_rows = [
        {"fraction": float(f), "mean": float(m),
         "ci_lower": float(a), "ci_upper": float(b)}
        for f, m, a, b in zip(conc["fraction_grid"], mean, lo, hi)
    ]
    return {
        "mass_table": mass_rows,
        "coverage_table": coverage_rows,
        "curve_table": curve_rows,
        "curve": {"fraction": conc["fraction_grid"], "mean": mean,
                  "ci_lower": lo, "ci_upper": hi},
        "fp_boundary": auto_fp if fp_boundary is None else float(fp_boundary),
        "fp_plus_q_boundary": auto_q if fp_plus_q_boundary is None
        else float(fp_plus_q_boundary),
        "fp_boundary_auto": auto_fp,
        "fp_plus_q_boundary_auto": auto_q,
    }


# ---------------------------------------------------------------------------
# Observation II
# ---------------------------------------------------------------------------


def analyse_window_level_decisions(
    inp: ObservationInputs,
    horizon: int = DEFAULTS["fmm_horizon"],
    pairs: Sequence[Tuple[str, str]] = (),
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Future Missed Mass + Selection Churn for every accessible-set variant."""
    groups = inp.trace_group
    # How many *existing* windows each policy actually threw away.  A policy
    # that drops nothing (tiny geometry, or N_q wide enough to hold the whole
    # band) makes FMM trivially zero — surfacing the count keeps that visible
    # in the artifact instead of looking like a win.
    exists = inp.event_valid                                   # [M, R, W]

    def dropped_per_event(mask: np.ndarray) -> np.ndarray:
        return np.sum(exists & ~mask, axis=-1)                 # [M, R]
    fmm: Dict[str, np.ndarray] = {}
    churn: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for i, (name, mask) in enumerate(inp.accessible.items()):
        fmm[name] = QM.future_missed_mass(
            inp.mass_step, mask, inp.event_steps, horizon,
            creation_steps=inp.creation, require_full_horizon=True)
        churn[name] = QM.selection_churn(mask)
        scored = int(np.isfinite(fmm[name]).any(axis=0).sum())
        if scored == 0:
            log.warning(
                "policy %s: no routing event has a full %d-step future horizon "
                "(T=%d, last event=%d) — FMM is undefined; lower --fmm-horizon.",
                name, horizon, inp.mass_step.shape[1], int(inp.event_steps[-1]))
        fm = QM.group_reduce(QM.nanmean(fmm[name], axis=1), groups)
        ch = QM.group_reduce(QM.nanmean(churn[name], axis=1), groups)
        fm_mu, fm_lo, fm_hi = QM.bootstrap_mean_ci(fm, confidence, n_boot, seed + 10 * i)
        ch_mu, ch_lo, ch_hi = QM.bootstrap_mean_ci(ch, confidence, n_boot, seed + 10 * i + 1)
        rows.append({
            "policy": name,
            "future_missed_mass_mean": fm_mu,
            "future_missed_mass_ci_lower": fm_lo,
            "future_missed_mass_ci_upper": fm_hi,
            "selection_churn_mean": ch_mu,
            "selection_churn_ci_lower": ch_lo,
            "selection_churn_ci_upper": ch_hi,
            "mean_accessible_units": float(mask.sum(axis=-1).mean()),
            "mean_dropped_units": float(dropped_per_event(mask).mean()),
            "events_scored": scored,
        })

    paired: List[Dict[str, Any]] = []
    for j, (ref_name, cmp_name) in enumerate(pairs):
        if ref_name not in fmm or cmp_name not in fmm:
            continue
        for metric, store in (("future_missed_mass", fmm),
                              ("selection_churn", churn)):
            ref = QM.group_reduce(QM.nanmean(store[ref_name], axis=1), groups)
            cmp = QM.group_reduce(QM.nanmean(store[cmp_name], axis=1), groups)
            ok = np.isfinite(ref) & np.isfinite(cmp)
            if not np.any(ok):
                continue
            mu, lo, hi = QM.bootstrap_mean_ci(
                ref[ok] - cmp[ok], confidence, n_boot, seed + 500 + j)
            ref_mean = float(ref[ok].mean())
            paired.append({
                "metric": metric,
                "reference_policy": ref_name,
                "compared_policy": cmp_name,
                "absolute_reduction": mu,
                "ci_lower": lo, "ci_upper": hi,
                "reference_mean": ref_mean,
                "relative_reduction": mu / ref_mean if ref_mean > 0 else np.nan,
            })

    total_dropped = max((r["mean_dropped_units"] for r in rows), default=0.0)
    if total_dropped <= 0:
        log.warning(
            "No policy dropped a single existing window — Observation II is "
            "vacuous for this run (check top_k_fp/N_q against W=%d; a short "
            "prefill can leave the whole band inside the budget).",
            inp.geometry["num_windows"])

    curves = {}
    for i, name in enumerate(inp.accessible):
        f_mu, f_lo, f_hi = QM.bootstrap_curve_ci(
            fmm[name], confidence, n_boot, seed + 1000 + i)
        c_mu, c_lo, c_hi = QM.bootstrap_curve_ci(
            churn[name], confidence, n_boot, seed + 2000 + i) \
            if churn[name].shape[1] else (np.zeros(0),) * 3
        curves[name] = {"fmm": (f_mu, f_lo, f_hi), "churn": (c_mu, c_lo, c_hi)}

    return {
        "summary_table": rows,
        "paired_comparison_table": paired,
        "future_missed_mass": fmm,
        "selection_churn": churn,
        "curves": curves,
        "horizon": int(horizon),
    }


# ---------------------------------------------------------------------------
# Observation III
# ---------------------------------------------------------------------------


def analyse_revival(
    selected: np.ndarray,
    label: str,
    groups: Optional[np.ndarray] = None,
    inactivity_values: Sequence[int] = DEFAULTS["lir_inactivity_values"],
    horizon_values: Sequence[int] = DEFAULTS["lir_horizon_values"],
    transition_deltas: Sequence[int] = DEFAULTS["transition_deltas"],
    primary_inactivity: int = DEFAULTS["primary_inactivity"],
    primary_horizon: int = DEFAULTS["primary_lir_horizon"],
    primary_delta: int = DEFAULTS["primary_transition_delta"],
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Episode-LIR sweep + transition sweep for one trinary selection matrix."""
    R = selected.shape[1]
    ms = tuple(int(m) for m in inactivity_values)
    Hs = tuple(int(h) for h in horizon_values)
    ds = tuple(int(d) for d in transition_deltas if 0 < int(d) < R)

    grid = np.full((len(ms), len(Hs)), np.nan)
    grid_lo = np.full_like(grid, np.nan)
    grid_hi = np.full_like(grid, np.nan)
    details: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for i, m in enumerate(ms):
        for j, H in enumerate(Hs):
            res = QM.episode_lir(selected, m, H)
            details[(m, H)] = res
            mu, lo, hi = QM.bootstrap_mean_ci(
                QM.group_ratio(res["rescued_by_trace"],
                               res["eligible_by_trace"], groups),
                confidence, n_boot, seed + i * 100 + j)
            grid[i, j], grid_lo[i, j], grid_hi[i, j] = mu, lo, hi

    key = (int(primary_inactivity), int(primary_horizon))
    primary = details.get(key) or QM.episode_lir(selected, *key)
    lir_mu, lir_lo, lir_hi = QM.bootstrap_mean_ci(
        QM.group_ratio(primary["rescued_by_trace"],
                       primary["eligible_by_trace"], groups),
        confidence, n_boot, seed + 9000)
    if primary["eligible"] == 0:
        log.warning(
            "%s: no eligible episode survives right-censoring at m=%d, H=%d "
            "with R=%d routing events (need m-1+H < R). Global LIR is NA — "
            "lower --primary-inactivity/--primary-lir-horizon or run more "
            "decode steps.", label, primary_inactivity, primary_horizon, R)
    ttr = primary["time_to_revival"].astype(float)
    if ttr.size:
        median = float(np.median(ttr))
        q1, q3 = (float(v) for v in np.quantile(ttr, [0.25, 0.75]))
    else:
        median = q1 = q3 = np.nan

    trans_rows: List[Dict[str, Any]] = []
    trans_details: Dict[int, Dict[str, Any]] = {}
    for i, d in enumerate(ds):
        res = QM.binary_transition(selected, d)
        trans_details[d] = res
        p = res["probabilities_by_trace"]
        p01 = QM.bootstrap_mean_ci(
            QM.group_reduce(p[:, 0, 1], groups), confidence, n_boot, seed + 10000 + i)
        p10 = QM.bootstrap_mean_ci(
            QM.group_reduce(p[:, 1, 0], groups), confidence, n_boot, seed + 11000 + i)
        trans_rows.append({
            "delta": d,
            "P01_mean": p01[0], "P01_ci_lower": p01[1], "P01_ci_upper": p01[2],
            "P10_mean": p10[0], "P10_ci_lower": p10[1], "P10_ci_upper": p10[2],
            "pooled_P01": float(res["pooled_probabilities"][0, 1]),
            "pooled_P10": float(res["pooled_probabilities"][1, 0]),
        })
    pd_ = int(primary_delta)
    primary_trans = trans_details.get(pd_)
    if primary_trans is None and 0 < pd_ < R:
        primary_trans = QM.binary_transition(selected, pd_)
    if primary_trans is not None:
        pp = primary_trans["probabilities_by_trace"]
        p01 = QM.bootstrap_mean_ci(
            QM.group_reduce(pp[:, 0, 1], groups), confidence, n_boot, seed + 12000)
        p10 = QM.bootstrap_mean_ci(
            QM.group_reduce(pp[:, 1, 0], groups), confidence, n_boot, seed + 13000)
    else:
        p01 = p10 = (np.nan, np.nan, np.nan)

    summary = {
        "selection": label,
        "inactivity_m": int(primary_inactivity),
        "horizon_H": int(primary_horizon),
        "global_lir_mean": lir_mu,
        "global_lir_ci_lower": lir_lo,
        "global_lir_ci_upper": lir_hi,
        "global_lir_pooled": primary["global_rate"],
        "eligible_episodes": primary["eligible"],
        "rescued_episodes": primary["rescued"],
        "time_to_revival_median": median,
        "time_to_revival_q1": q1,
        "time_to_revival_q3": q3,
        "transition_delta": pd_,
        "P01_mean": p01[0], "P01_ci_lower": p01[1], "P01_ci_upper": p01[2],
        "P10_mean": p10[0], "P10_ci_lower": p10[1], "P10_ci_upper": p10[2],
    }
    grid_rows = [
        {"inactivity_m": m,
         **{f"H{H}": grid[i, j] for j, H in enumerate(Hs)}}
        for i, m in enumerate(ms)
    ]
    episode_rows = [
        {"trace": int(primary["episode_trace"][e]),
         "window": int(primary["episode_window"][e]),
         "episode_start": int(primary["episode_start"][e]),
         "episode_end": int(primary["episode_end"][e]),
         "episode_length": int(primary["episode_length"][e]),
         "eligibility_event": int(primary["episode_eligibility_event"][e]),
         "rescued": bool(primary["episode_rescued"][e]),
         "time_to_revival": int(primary["episode_time_to_revival"][e])}
        for e in range(primary["episode_trace"].size)
    ]
    return {
        "label": label,
        "lir_grid": grid, "lir_ci_lower": grid_lo, "lir_ci_upper": grid_hi,
        "lir_grid_rows": grid_rows,
        "lir_inactivity_values": ms, "lir_horizon_values": Hs,
        "transition_table": trans_rows,
        "quantifiable_result": summary,
        "primary_episode_rows": episode_rows,
        "primary_pooled_transition": (
            primary_trans["pooled_probabilities"] if primary_trans is not None
            else np.full((2, 2), np.nan)),
        "time_to_revival": primary["time_to_revival"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(x: Any, digits: int = 1) -> str:
    x = float(x) if x is not None else np.nan
    return "NA" if not np.isfinite(x) else f"{100.0 * x:.{digits}f}%"


def _num(x: Any, digits: int = 3) -> str:
    x = float(x) if x is not None else np.nan
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def _ci_pct(mean: Any, lo: Any, hi: Any) -> str:
    if not np.isfinite(float(mean)):
        return "NA"
    if np.isfinite(float(lo)) and np.isfinite(float(hi)):
        return f"{_pct(mean)} [{_pct(lo)}, {_pct(hi)}]"
    return _pct(mean)


def _ci_num(mean: Any, lo: Any, hi: Any, digits: int = 3) -> str:
    if not np.isfinite(float(mean)):
        return "NA"
    if np.isfinite(float(lo)) and np.isfinite(float(hi)):
        return (f"{_num(mean, digits)} "
                f"[{_num(lo, digits)}, {_num(hi, digits)}]")
    return _num(mean, digits)


def _nearest(rows: List[Dict[str, Any]], key: str, target: float) -> Dict[str, Any]:
    return min(rows, key=lambda r: abs(float(r[key]) - float(target)))


def build_paper_summary(
    inp: ObservationInputs,
    obs1: Dict[str, Any],
    obs2: Dict[str, Any],
    obs3: Dict[str, Dict[str, Any]],
    primary_top_fraction: float = DEFAULTS["primary_top_fraction"],
    primary_target_mass: float = DEFAULTS["primary_target_mass"],
) -> str:
    """Render the observation results as a markdown report."""
    g = inp.geometry
    mass_row = _nearest(obs1["mass_table"], "top_fraction", primary_top_fraction)
    cov_row = _nearest(obs1["coverage_table"], "target_mass", primary_target_mass)
    by_policy = {r["policy"]: r for r in obs2["summary_table"]}

    lines: List[str] = [
        "# QEvict observation results",
        "",
        f"Traces `M={g['num_traces']}` = {g['num_samples']} sample(s) x "
        f"{g['num_layers']} layer(s), bootstrapped over {g['num_groups']} "
        f"`{g['trace_axis']}` group(s), `T={g['num_steps']}` decode "
        f"steps, `R={g['num_events']}` routing events, `W={g['num_windows']}` "
        f"windows of {g['window_size']} tokens.  Cache: "
        f"top_k_fp={g['top_k_fp']}, N_q={g['N_q']}, local={g['local_windows']} "
        f"windows, quant_ratio={g['quant_ratio']:.2f}.",
        "",
        "Intervals are percentile bootstrap CIs over the trace axis; with a "
        "single-article run those traces are layers of the same prompt, so read "
        "them as within-run variability rather than population CIs.",
        "",
        "## Observation I: skewed importance",
        "",
        f"- Top {_pct(mass_row['top_fraction'], 0)} of windows hold "
        f"{_ci_pct(mass_row['mean_cumulative_mass'], mass_row['ci_lower'], mass_row['ci_upper'])}"
        " of the importance mass "
        f"(concentration gap {_pct(mass_row['concentration_gap'])} over uniform).",
        f"- Reaching {_pct(cov_row['target_mass'], 0)} of the mass needs "
        f"{_ci_pct(cov_row['mean_required_fraction'], cov_row['ci_lower'], cov_row['ci_upper'])}"
        " of windows.",
        f"- Measured capacity boundaries on this axis: fp keeps "
        f"{_pct(obs1['fp_boundary'])} of valid windows, fp+Q keeps "
        f"{_pct(obs1['fp_plus_q_boundary'])}.",
        "",
        f"## Observation II: window-level decisions (H={obs2['horizon']})",
        "",
        "| policy | Future Missed Mass | Selection Churn | units kept | units dropped | events scored |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in obs2["summary_table"]:
        lines.append(
            f"| `{r['policy']}` | "
            f"{_ci_pct(r['future_missed_mass_mean'], r['future_missed_mass_ci_lower'], r['future_missed_mass_ci_upper'])} | "
            f"{_ci_num(r['selection_churn_mean'], r['selection_churn_ci_lower'], r['selection_churn_ci_upper'])} | "
            f"{r['mean_accessible_units']:.1f} | {r['mean_dropped_units']:.1f} | "
            f"{r['events_scored']} |")
    lines.append("")
    for p in obs2["paired_comparison_table"]:
        lines.append(
            f"- Paired {p['metric'].replace('_', ' ')}: `{p['compared_policy']}` "
            f"vs `{p['reference_policy']}` -> absolute reduction "
            f"{_ci_num(p['absolute_reduction'], p['ci_lower'], p['ci_upper'], 4)}"
            f" ({_pct(p['relative_reduction'])} relative).")
    if inp.diagnostics["mass_source"] == "differenced_cumulative":
        lines += ["", "> **Caveat.** The base npz predates schema 1.2, so the "
                  "per-step mass was differenced from the fp16 cumulative "
                  "scores "
                  f"({_pct(inp.diagnostics['step_mass_negative_fraction'])} of "
                  "entries came out negative and were clamped). Re-run "
                  "BaseParityRunner to record `step_window_scores` before "
                  "reporting Observation II."]

    lines += ["", "## Observation III: historical importance revives", ""]
    for label, res in obs3.items():
        s = res["quantifiable_result"]
        lines += [
            f"### {label}",
            "",
            f"- Global LIR (m={s['inactivity_m']}, H={s['horizon_H']}): "
            f"{_ci_pct(s['global_lir_mean'], s['global_lir_ci_lower'], s['global_lir_ci_upper'])}"
            f" per trace; {_pct(s['global_lir_pooled'])} pooled "
            f"({s['rescued_episodes']}/{s['eligible_episodes']} episodes).",
            f"- Median time to revival: {_num(s['time_to_revival_median'], 1)} "
            f"events (IQR {_num(s['time_to_revival_q1'], 1)}-"
            f"{_num(s['time_to_revival_q3'], 1)}).",
            f"- At lag delta={s['transition_delta']}: "
            f"P01={_ci_pct(s['P01_mean'], s['P01_ci_lower'], s['P01_ci_upper'])} "
            f"(promotion), "
            f"P10={_ci_pct(s['P10_mean'], s['P10_ci_lower'], s['P10_ci_upper'])} "
            f"(demotion).",
            "",
        ]

    # Manuscript sentences are only emitted for quantities that actually
    # resolved — a sentence reading "NA of episodes" is worse than no sentence.
    fp_only = by_policy.get("measured_fp_only")
    fp_q = by_policy.get("measured_fp_plus_q")
    if fp_only and fp_q and fp_only["mean_dropped_units"] > 0:
        lines += [
            "## Suggested manuscript sentences",
            "",
            f"> Window importance is strongly skewed: the top "
            f"{_pct(mass_row['top_fraction'], 0)} of windows carry "
            f"{_pct(mass_row['mean_cumulative_mass'])} of the attention mass, so "
            f"{_pct(cov_row['mean_required_fraction'])} of windows suffice for "
            f"{_pct(cov_row['target_mass'], 0)} of it.",
            "",
            f"> Holding the low-rank band in int4 instead of dropping it lowers "
            f"Future Missed Mass over the next {obs2['horizon']} decode steps "
            f"from {_pct(fp_only['future_missed_mass_mean'])} to "
            f"{_pct(fp_q['future_missed_mass_mean'])} at a matched measured "
            f"budget.",
            "",
        ]
    oracle = obs3.get("oracle")
    if oracle is not None and np.isfinite(
            float(oracle["quantifiable_result"]["global_lir_mean"])):
        s = oracle["quantifiable_result"]
        lines += [
            f"> Importance is not monotone: after {s['inactivity_m']} consecutive "
            f"events outside the oracle important set, "
            f"{_pct(s['global_lir_mean'])} of fully observed inactive episodes "
            f"regain importance within {s['horizon_H']} events, and the lag-"
            f"{s['transition_delta']} cold-to-hot transition probability is "
            f"{_pct(s['P01_mean'])} - the headroom a promotion path can recover.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        log.warning("No rows for %s — skipped", path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if (isinstance(v, float) and not np.isfinite(v))
                                 else v) for k, v in r.items()})


def make_figures(
    inp: ObservationInputs,
    obs1: Dict[str, Any],
    obs2: Dict[str, Any],
    obs3: Dict[str, Dict[str, Any]],
    out_dir: Path,
    dpi: int = 300,
) -> List[Path]:
    """Write the three paper figures; returns [] when matplotlib is absent."""
    if not HAS_MPL:
        log.warning("matplotlib not installed — skipping figures "
                    "(all tables/JSON are still written)")
        return []
    written: List[Path] = []

    c = obs1["curve"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(c["fraction"], c["mean"], label="Observed cumulative mass")
    if np.isfinite(c["ci_lower"]).any():
        ax.fill_between(c["fraction"], c["ci_lower"], c["ci_upper"], alpha=0.2)
    ax.plot(c["fraction"], c["fraction"], "--", label="Uniform importance")
    ax.axvline(obs1["fp_boundary"], linestyle=":", color="k", label="fp boundary")
    ax.axvline(obs1["fp_plus_q_boundary"], linestyle="-.", color="k",
               label="fp + Q boundary")
    ax.set(xlabel="Fraction of highest-ranked windows",
           ylabel="Cumulative importance mass", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    p = out_dir / "observation1_cumulative_mass.pdf"
    fig.tight_layout(); fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    written.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    steps = inp.event_steps
    for name, cur in obs2["curves"].items():
        f_mu, f_lo, f_hi = cur["fmm"]
        axes[0].plot(steps, f_mu, label=name)
        if np.isfinite(f_lo).any():
            axes[0].fill_between(steps, f_lo, f_hi, alpha=0.18)
        c_mu, c_lo, c_hi = cur["churn"]
        if c_mu.size:
            axes[1].plot(steps[1:], c_mu, label=name)
            if np.isfinite(c_lo).any():
                axes[1].fill_between(steps[1:], c_lo, c_hi, alpha=0.18)
    axes[0].set(xlabel="Routing step (decode step)",
                ylabel=f"Future Missed Mass (H={obs2['horizon']})")
    axes[1].set(xlabel="Routing step (decode step)", ylabel="Selection Churn")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    p = out_dir / "observation2_window_decisions.pdf"
    fig.tight_layout(); fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    written.append(p)

    for label, res in obs3.items():
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1))
        im = axes[0].imshow(res["lir_grid"], aspect="auto", origin="lower",
                            vmin=0, vmax=1)
        axes[0].set_xticks(range(len(res["lir_horizon_values"])),
                           res["lir_horizon_values"])
        axes[0].set_yticks(range(len(res["lir_inactivity_values"])),
                           res["lir_inactivity_values"])
        axes[0].set(xlabel="Revival horizon H", ylabel="Inactive events m",
                    title=f"Episode Global LIR ({label})")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        ttr = res["time_to_revival"]
        H = res["quantifiable_result"]["horizon_H"]
        if ttr.size:
            axes[1].hist(ttr, bins=np.arange(0.5, H + 1.5, 1))
            axes[1].axvline(float(np.median(ttr)), linestyle="--", label="Median")
            axes[1].legend(fontsize=8)
        axes[1].set(xlabel="Routing events until revival",
                    ylabel="Rescued episodes", title="Time to revival")
        pooled = res["primary_pooled_transition"]
        im2 = axes[2].imshow(pooled, vmin=0, vmax=1)
        for a in range(2):
            for b in range(2):
                v = pooled[a, b]
                axes[2].text(b, a, f"{v:.3f}" if np.isfinite(v) else "nan",
                             ha="center", va="center")
        axes[2].set_xticks([0, 1], ["Future 0", "Future 1"])
        axes[2].set_yticks([0, 1], ["Current 0", "Current 1"])
        axes[2].set_title(
            f"Transitions (delta={res['quantifiable_result']['transition_delta']})")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        p = out_dir / f"observation3_revival_{label}.pdf"
        fig.tight_layout(); fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        written.append(p)
    return written


def write_outputs(
    inp: ObservationInputs,
    obs1: Dict[str, Any],
    obs2: Dict[str, Any],
    obs3: Dict[str, Dict[str, Any]],
    out_dir: Path,
    meta: Dict[str, Any],
    figures: bool = True,
) -> Path:
    """Write CSVs, JSON, markdown, raw npz and (optionally) figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "observation1_mass_table.csv", obs1["mass_table"])
    _write_csv(out_dir / "observation1_coverage_table.csv", obs1["coverage_table"])
    _write_csv(out_dir / "observation1_curve.csv", obs1["curve_table"])
    _write_csv(out_dir / "observation2_summary_table.csv", obs2["summary_table"])
    _write_csv(out_dir / "observation2_paired_comparison.csv",
               obs2["paired_comparison_table"])
    for label, res in obs3.items():
        _write_csv(out_dir / f"observation3_lir_grid_{label}.csv",
                   res["lir_grid_rows"])
        _write_csv(out_dir / f"observation3_transition_table_{label}.csv",
                   res["transition_table"])
        _write_csv(out_dir / f"observation3_quantifiable_result_{label}.csv",
                   [res["quantifiable_result"]])
        _write_csv(out_dir / f"observation3_primary_episodes_{label}.csv",
                   res["primary_episode_rows"])

    summary_md = build_paper_summary(inp, obs1, obs2, obs3)
    (out_dir / "paper_results.md").write_text(summary_md, encoding="utf-8")

    results = {
        "metadata": meta,
        "geometry": inp.geometry,
        "diagnostics": inp.diagnostics,
        "observation1": {
            "mass_table": obs1["mass_table"],
            "coverage_table": obs1["coverage_table"],
            "fp_boundary": obs1["fp_boundary"],
            "fp_plus_q_boundary": obs1["fp_plus_q_boundary"],
            "fp_boundary_auto": obs1["fp_boundary_auto"],
            "fp_plus_q_boundary_auto": obs1["fp_plus_q_boundary_auto"],
        },
        "observation2": {
            "horizon": obs2["horizon"],
            "summary_table": obs2["summary_table"],
            "paired_comparison_table": obs2["paired_comparison_table"],
        },
        "observation3": {
            label: {
                "quantifiable_result": res["quantifiable_result"],
                "transition_table": res["transition_table"],
                "lir_grid": res["lir_grid"],
                "lir_ci_lower": res["lir_ci_lower"],
                "lir_ci_upper": res["lir_ci_upper"],
                "lir_inactivity_values": list(res["lir_inactivity_values"]),
                "lir_horizon_values": list(res["lir_horizon_values"]),
            }
            for label, res in obs3.items()
        },
    }
    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(results), f, indent=2)

    arrays: Dict[str, np.ndarray] = {
        "fraction_grid": obs1["curve"]["fraction"],
        "concentration_mean": obs1["curve"]["mean"],
        "concentration_ci_lower": obs1["curve"]["ci_lower"],
        "concentration_ci_upper": obs1["curve"]["ci_upper"],
        "event_steps": inp.event_steps,
        "w_act": inp.w_act,
        "ew_act": inp.ew_act,
        "metadata_json": np.array([json.dumps(_json_safe(meta))], dtype=object),
    }
    for name, arr in obs2["future_missed_mass"].items():
        arrays[f"fmm__{name}"] = arr
    for name, arr in obs2["selection_churn"].items():
        arrays[f"churn__{name}"] = arr
    for label, res in obs3.items():
        arrays[f"lir_grid__{label}"] = res["lir_grid"]
        arrays[f"lir_ci_lower__{label}"] = res["lir_ci_lower"]
        arrays[f"lir_ci_upper__{label}"] = res["lir_ci_upper"]
        arrays[f"time_to_revival__{label}"] = res["time_to_revival"]
    npz_path = out_dir / "qevict_observations.npz"
    np.savez_compressed(str(npz_path), **arrays)
    # Sidecar, matching every other suite's output convention.
    with open(npz_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe({**meta, "geometry": inp.geometry,
                              "diagnostics": inp.diagnostics}), f, indent=2,
                  default=str)

    if figures:
        make_figures(inp, obs1, obs2, obs3, out_dir)
    log.info("Saved QEvict observations to %s", out_dir.resolve())
    return npz_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_observations(
    base_npz: str | Path,
    ours_npz: str | Path,
    out_dir: str | Path,
    *,
    report_fractions: Sequence[float] = DEFAULTS["report_fractions"],
    target_masses: Sequence[float] = DEFAULTS["target_masses"],
    fp_boundary: Optional[float] = None,
    fp_plus_q_boundary: Optional[float] = None,
    fmm_horizon: int = DEFAULTS["fmm_horizon"],
    pool_factor: int = DEFAULTS["pool_factor"],
    lir_inactivity_values: Sequence[int] = DEFAULTS["lir_inactivity_values"],
    lir_horizon_values: Sequence[int] = DEFAULTS["lir_horizon_values"],
    transition_deltas: Sequence[int] = DEFAULTS["transition_deltas"],
    primary_inactivity: int = DEFAULTS["primary_inactivity"],
    primary_lir_horizon: int = DEFAULTS["primary_lir_horizon"],
    primary_transition_delta: int = DEFAULTS["primary_transition_delta"],
    trace_axis: str = DEFAULTS["trace_axis"],
    oracle_budget: str = "fp",
    max_samples: Optional[int] = None,
    layer_stride: int = 1,
    confidence: float = DEFAULTS["confidence"],
    n_boot: int = DEFAULTS["bootstrap_samples"],
    seed: int = DEFAULTS["seed"],
    figures: bool = True,
) -> Dict[str, Any]:
    """End-to-end: load the parity pair, compute all three observations, write."""
    t0 = time.time()
    base = load_parity_npz(base_npz)
    ours = load_parity_npz(ours_npz)
    mismatches = []
    for f in ("article_sha", "seed", "prefill_len", "gen_len", "window_size",
              "num_sink_tokens", "local_window_size_resolved", "model_name"):
        bv, ov = base["metadata"].get(f), ours["metadata"].get(f)
        if bv is not None and ov is not None and bv != ov:
            mismatches.append(f"  {f}: base={bv!r}, ours={ov!r}")
    if mismatches:
        raise ParityValidationError(
            "QEvict observation alignment failed — the two npzs are not the "
            "same run:\n" + "\n".join(mismatches))
    if base["metadata"].get("mode") != "parity_base":
        log.warning("base npz mode is %r, expected 'parity_base'",
                    base["metadata"].get("mode"))
    if ours["metadata"].get("mode") != "parity_ours":
        log.warning("ours npz mode is %r, expected 'parity_ours'",
                    ours["metadata"].get("mode"))

    inp = build_observation_inputs(
        base, ours, trace_axis=trace_axis, oracle_budget=oracle_budget,
        pool_factor=pool_factor, max_samples=max_samples,
        layer_stride=layer_stride)
    g = inp.geometry
    log.info("Traces M=%d (%d sample(s) x %d/%d layer(s)) | T=%d steps | "
             "R=%d routing events | W=%d windows | mass=%s",
             g["num_traces"], g["num_samples"], g["num_layers"],
             g["num_layers_total"], g["num_steps"], g["num_events"],
             g["num_windows"], inp.diagnostics["mass_source"])

    obs1 = analyse_skewed_importance(
        inp, report_fractions, target_masses, fp_boundary, fp_plus_q_boundary,
        confidence, n_boot, seed)

    names = list(inp.accessible)
    pairs: List[Tuple[str, str]] = []
    if "measured_fp_only" in names and "measured_fp_plus_q" in names:
        pairs.append(("measured_fp_only", "measured_fp_plus_q"))
    sim = [n for n in names if n.startswith("simulated_")]
    if len(sim) == 2:
        pairs.append((sim[0], sim[1]))
    obs2 = analyse_window_level_decisions(
        inp, fmm_horizon, pairs, confidence, n_boot, seed + 100)

    obs3 = {
        label: analyse_revival(
            sel, label, groups=inp.trace_group,
            inactivity_values=lir_inactivity_values,
            horizon_values=lir_horizon_values,
            transition_deltas=transition_deltas,
            primary_inactivity=primary_inactivity,
            primary_horizon=primary_lir_horizon,
            primary_delta=primary_transition_delta,
            confidence=confidence, n_boot=n_boot, seed=seed + 200 + off)
        for off, (label, sel) in enumerate(
            (("oracle", inp.oracle_selected), ("policy_fp", inp.policy_selected)))
    }

    meta = {
        "schema_version": "1.0",
        "mode": "qevict_observations",
        "base_npz_path": base["path"],
        "base_npz_sha256": sha256_file(base["path"]),
        "ours_npz_path": ours["path"],
        "ours_npz_sha256": sha256_file(ours["path"]),
        "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        # ── run identity, propagated from the ours npz (for auditing) ──
        "model_name": ours["metadata"].get("model_name"),
        "dataset": ours["metadata"].get("dataset"),
        "article_index_start": ours["metadata"].get("article_index_start"),
        "prefill_len": ours["metadata"].get("prefill_len"),
        "gen_len": ours["metadata"].get("gen_len"),
        "window_size": ours["metadata"].get("window_size"),
        "num_sink_tokens": ours["metadata"].get("num_sink_tokens"),
        "local_window_size_resolved": ours["metadata"].get(
            "local_window_size_resolved"),
        "cache_budget": ours["metadata"].get("cache_budget"),
        "quant_ratio": ours["metadata"].get("quant_ratio", 0.0),
        "top_k_fp": ours["metadata"].get("top_k_fp"),
        "N_q": ours["metadata"].get("N_q", 0),
        # ── analysis knobs ──
        "seed": seed,
        "confidence": confidence,
        "bootstrap_samples": n_boot,
        "fmm_horizon": int(fmm_horizon),
        "pool_factor": int(pool_factor),
        "oracle_budget": oracle_budget,
        "trace_axis": trace_axis,
        "max_samples": max_samples,
        "layer_stride": int(layer_stride),
        "primary_inactivity": int(primary_inactivity),
        "primary_lir_horizon": int(primary_lir_horizon),
        "primary_transition_delta": int(primary_transition_delta),
        "mass_source": inp.diagnostics["mass_source"],
    }
    npz_path = write_outputs(inp, obs1, obs2, obs3, Path(out_dir), meta, figures)
    return {"inputs": inp, "observation1": obs1, "observation2": obs2,
            "observation3": obs3, "npz_path": npz_path, "metadata": meta}


class QEvictObservationRunner:
    """``main.py --config ... --override run.mode=qevict_observations``.

    Reads the parity pair from the ``faithfulness:`` config section and writes
    under ``telemetry.output_dir/qevict_observations``.  Every analysis knob
    uses its CLI default — tune them through the module CLI (a ``qevict:`` YAML
    section would need a matching dataclass in ``utils/config.py``).
    """

    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        fc = cfg.faithfulness
        if not fc.base_npz_path or not fc.ours_npz_path:
            raise ValueError(
                "qevict_observations needs faithfulness.base_npz_path and "
                "faithfulness.ours_npz_path in the config.")
        out_dir = Path(cfg.output_path) if cfg.output_path else (
            Path(cfg.telemetry.output_dir) / "qevict_observations")
        res = run_observations(
            resolve_npz(fc.base_npz_path, "parity_base_*.npz"),
            resolve_npz(fc.ours_npz_path, "parity_ours_*.npz"),
            out_dir,
            seed=int(getattr(cfg.run, "seed", 0)))
        return res["npz_path"]


def _cli_main() -> None:
    ap = argparse.ArgumentParser(
        description="QEvict observation suite over a parity npz pair.")
    ap.add_argument("--base-npz", type=Path, required=True,
                    help="parity_base npz (ground-truth window masses).")
    ap.add_argument("--ours-npz", type=Path, required=True,
                    help="parity_ours npz (tier tags + eviction cadence).")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fp-boundary", type=float, default=None,
                    help="Override the fp capacity line (default: derived).")
    ap.add_argument("--fp-plus-q-boundary", type=float, default=None,
                    help="Override the fp+Q capacity line (default: derived).")
    ap.add_argument("--report-fractions", type=float, nargs="+",
                    default=list(DEFAULTS["report_fractions"]))
    ap.add_argument("--target-masses", type=float, nargs="+",
                    default=list(DEFAULTS["target_masses"]))
    ap.add_argument("--fmm-horizon", type=int, default=DEFAULTS["fmm_horizon"])
    ap.add_argument("--pool-factor", type=int, default=DEFAULTS["pool_factor"],
                    help="Coarse-policy block size in base windows "
                         "(use 8 with a window_size=1 base run for the true "
                         "token-vs-window comparison).")
    ap.add_argument("--lir-inactivity-values", type=int, nargs="+",
                    default=list(DEFAULTS["lir_inactivity_values"]))
    ap.add_argument("--lir-horizon-values", type=int, nargs="+",
                    default=list(DEFAULTS["lir_horizon_values"]))
    ap.add_argument("--transition-deltas", type=int, nargs="+",
                    default=list(DEFAULTS["transition_deltas"]))
    ap.add_argument("--primary-inactivity", type=int,
                    default=DEFAULTS["primary_inactivity"])
    ap.add_argument("--primary-lir-horizon", type=int,
                    default=DEFAULTS["primary_lir_horizon"])
    ap.add_argument("--primary-transition-delta", type=int,
                    default=DEFAULTS["primary_transition_delta"])
    ap.add_argument("--trace-axis", choices=("sample_layer", "sample", "layer"),
                    default=DEFAULTS["trace_axis"])
    ap.add_argument("--oracle-budget", choices=("fp", "kept"), default="fp",
                    help="Size of the ground-truth important set: top_k_fp, "
                         "or top_k_fp + N_q.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Use at most this many articles (memory relief).")
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="Keep every Nth layer (memory/time relief).")
    ap.add_argument("--confidence", type=float, default=DEFAULTS["confidence"])
    ap.add_argument("--bootstrap-samples", type=int,
                    default=DEFAULTS["bootstrap_samples"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    res = run_observations(
        args.base_npz, args.ours_npz, args.output_dir,
        report_fractions=args.report_fractions,
        target_masses=args.target_masses,
        fp_boundary=args.fp_boundary,
        fp_plus_q_boundary=args.fp_plus_q_boundary,
        fmm_horizon=args.fmm_horizon,
        pool_factor=args.pool_factor,
        lir_inactivity_values=args.lir_inactivity_values,
        lir_horizon_values=args.lir_horizon_values,
        transition_deltas=args.transition_deltas,
        primary_inactivity=args.primary_inactivity,
        primary_lir_horizon=args.primary_lir_horizon,
        primary_transition_delta=args.primary_transition_delta,
        trace_axis=args.trace_axis,
        oracle_budget=args.oracle_budget,
        max_samples=args.max_samples,
        layer_stride=args.layer_stride,
        confidence=args.confidence,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
        figures=not args.no_figures,
    )
    print(build_paper_summary(
        res["inputs"], res["observation1"], res["observation2"],
        res["observation3"]))
    print(f"\nSaved all outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    _cli_main()
