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
       path could capture) and for our cache's real fp tier, plus the
       lag-``delta`` flip matrix.  For the policy that matrix is fp-membership
       flips, NOT promotions: its ``0`` pools the int2 tier with the evicted
       windows (Observation V separates them).
  IV.  **The decode read ledger** — at every decode step, where each query
       head's ground-truth attention landed: fp / local / fresh (read exactly),
       int2 windows the read gate OPENED, int2 windows it SKIPPED (reached only
       through the centroid fill), evicted, sink.  Per query head, so the gate's
       recall is the §12 per-head quantity on real attention, beside a
       hindsight-optimal pick of the same size.  Needs an ours npz at schema
       >= 1.3 (``gate_read``) and, for exact per-head mass, a base npz recorded
       with ``parity.record_head_step_mass``.
  V.   **Tier dynamics** — the three-state F / Q / E chain: promotion (Q→F),
       demotion (F→Q), eviction from either tier, with rates, and whether each
       move landed on the windows the next ``H`` steps attend (lift over the
       candidates, hindsight hit rates, swap gain, eviction regret), plus how
       closely the cache's own ranking — gate-filled scores for skipped int2
       windows — matches the ground truth on the windows it holds.

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
    # Observation IV: a head-step counts toward gate recall only when at least
    # this share of the head's mass sits in the int2 tier (a head with ~0 int2
    # mass has an undefined recall, not a perfect or a zero one).
    "min_q_share": 0.01,
    # Observation V: F/Q/E transition lags, in routing events.
    "tier_deltas": (1, 2, 4),
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
    acc_q: Optional[np.ndarray] = None    # [M, R, W] bool — int2 survivors
    band: Optional[np.ndarray] = None     # [R, W] bool — evictable band per event
    states: Optional[np.ndarray] = None   # [M, R, W] int8 F/Q/E (qevict_metrics)


def _as5d(x: np.ndarray) -> np.ndarray:
    """Promote a legacy ``[T, L, H, W]`` array to ``[1, T, L, H, W]``."""
    return x[None] if x.ndim == 4 else x


def _as4d(x: np.ndarray) -> np.ndarray:
    """Promote a legacy ``[T, L, W]`` array to ``[1, T, L, W]``."""
    return x[None] if x.ndim == 3 else x


def read_path_info(ours_meta: Dict[str, Any]) -> Dict[str, Any]:
    """How the ours run READ its int2 tier — the gate verdict and knobs.

    ``read_path`` is one of ``gated`` (the read gate ran and its pick is
    recorded), ``gated-unrecorded`` (it ran, schema < 1.3 or recording off, so
    Observation IV cannot see which windows it opened), ``ungated`` (eager
    backend or ``quant_ratio = 0``: no gate exists, the whole tier is read) or
    ``NOT-GATED`` (it should have gated and did not — the run is not the
    shipped method; ``flash_decode.gate_report``).
    """
    rg = ours_meta.get("read_gate") or {}
    verdict = rg.get("verdict")
    recorded = bool(ours_meta.get("gate_recorded", False))
    if verdict is None:
        path = "unknown"
    elif verdict == "gated":
        path = "gated" if recorded else "gated-unrecorded"
    elif verdict == "not-expected":
        path = "ungated"
    else:
        path = "NOT-GATED"
    return {
        "read_path": path,
        "read_gate_verdict": verdict,
        "quant_gate_ratio": ours_meta.get("quant_gate_ratio"),
        "quant_budget_mode": ours_meta.get("quant_budget_mode"),
        "quant_card_bits": ours_meta.get("quant_card_bits"),
        "realised_read_fraction": rg.get("read_fraction"),
        "gate_recorded": recorded,
    }


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
    acc_q = np.zeros((M, R, W), dtype=bool)
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
            acc_q[m, r, idx[tr == TIER_Q]] = True
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

    # ── F / Q / E states over the evictable band (Observation V) ─────────
    band = np.zeros((R, W), dtype=bool)
    for r in range(R):
        band[r, :max(int(ev_w[r]), 0)] = True
    states = QM.tier_states(acc_fp, acc_q, band)

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
            **read_path_info(om),
        },
        diagnostics={
            "mass_source": mass_source,
            "step_mass_negative_fraction": neg_frac,
            "step_mass_negative_magnitude_ratio": neg_mag,
            "survivor_ids_out_of_range": int(out_of_range),
            "base_schema_version": str(bm.get("schema_version", "?")),
            "ours_schema_version": str(om.get("schema_version", "?")),
            # fp32 since base schema 1.3; before it the oracle's running sum was
            # fp16 and dropped most decode steps (base_parity_runner docstring).
            "base_score_accum_dtype": str(bm.get("score_accum_dtype", "float16")),
            "tier_resurrections": int(
                QM.tier_transitions(states, 1)["resurrections"]) if R > 1 else 0,
        },
        acc_q=acc_q,
        band=band,
        states=states,
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
# Observation IV — the decode read ledger
# ---------------------------------------------------------------------------


def _head_step_mass(base_arrays: Dict[str, np.ndarray], s: int, li: int, T: int,
                    W: int) -> Tuple[np.ndarray, str]:
    """``[T, H, W]`` float32 per-step per-query-head window mass for one trace."""
    if "step_window_scores_heads" in base_arrays:
        hm = base_arrays["step_window_scores_heads"]
        hm = hm if hm.ndim == 5 else hm[None]
        x = np.asarray(hm[s, :T, li], dtype=np.float32)
        src = "recorded_head_step_mass"
    else:
        cum = _as5d(base_arrays["window_scores"])[s, :T, li].astype(np.float64)
        x = np.clip(np.diff(cum, axis=0, prepend=np.zeros((1,) + cum.shape[1:])),
                    0.0, None).astype(np.float32)
        src = "differenced_cumulative"
    if x.shape[-1] < W:
        x = np.pad(x, [(0, 0), (0, 0), (0, W - x.shape[-1])])
    return x[..., :W], src


def analyse_decode_reads(
    base: dict,
    ours: dict,
    inp: ObservationInputs,
    min_q_share: float = 0.01,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Observation IV: split every head's per-step mass by how the decode read it.

    Per ``(sample, layer)`` trace and query head, over decode steps ``t >= 1``,
    the ground-truth attention (the base run's, one query row per step, so each
    head's mass sums to 1) is assigned to :data:`utils.qevict_metrics.LEDGER_PARTS`
    by the ours run's tier tags and, where the gate ran, its recorded pick for
    that head's KV group. Every statistic is per head first and then averaged;
    nothing sums attention across query heads (CLAUDE.md).

    Returns the ledger table, the per-head gate-recall table (gate vs the
    hindsight-optimal pick of the same size), a per-layer recall table, and the
    per-trace arrays behind them.
    """
    ba, oa = base["arrays"], ours["arrays"]
    bm, om = base["metadata"], ours["metadata"]
    g = inp.geometry
    T, W = int(g["num_steps"]), int(g["num_windows"])
    groups = inp.trace_group
    all_ids = _as4d(np.asarray(oa["all_window_ids"]))
    all_tier = _as4d(np.asarray(oa["all_window_tier"]))
    gate_read = oa.get("gate_read")
    gate_fired = oa.get("gate_fired")
    if gate_read is not None:
        gate_read = np.asarray(gate_read)
        gate_read = gate_read if gate_read.ndim == 5 else gate_read[None]
        gate_fired = np.asarray(gate_fired, dtype=bool)
        gate_fired = gate_fired if gate_fired.ndim == 3 else gate_fired[None]

    H = int(_as5d(np.asarray(ba["window_scores"])).shape[3])
    if gate_read is not None:
        H_kv = int(gate_read.shape[3])
    else:
        H_kv = int(om.get("num_key_value_heads") or bm.get("num_key_value_heads")
                   or H)
    rep = max(1, H // max(H_kv, 1))
    P = len(QM.LEDGER_PARTS)
    M = len(inp.trace_labels)

    ledger = np.full((M, P), np.nan)          # mean over steps and heads
    ledger_win = np.full((M, P), np.nan)      # share of post-sink window mass
    recall = np.full((M, H), np.nan)
    oracle = np.full((M, H), np.nan)
    read_frac = np.full(M, np.nan)
    gated_frac = np.zeros(M)
    step_recalls: List[np.ndarray] = []
    src = "none"
    oob = 0
    for m, (s, li) in enumerate(inp.trace_labels):
        mass, src = _head_step_mass(ba, s, li, T, W)
        cls, rd, o = QM.survivor_to_base(
            all_ids[s, :T, li], all_tier[s, :T, li], W,
            None if gate_read is None else gate_read[s, :T, li])
        oob += o
        fired = None if gate_fired is None else gate_fired[s, :T, li]
        led = QM.decode_read_ledger(mass, cls, rd, fired, rep)
        parts = led["parts"]
        if parts.shape[0] == 0:
            continue
        ledger[m] = parts.mean(axis=(0, 1))
        win = parts[..., :QM.LEDGER_PARTS.index("sink")].sum(-1)      # [T', H]
        tot = win.sum()
        if tot > 0:
            ledger_win[m] = parts.sum(axis=(0, 1)) / tot
            ledger_win[m, QM.LEDGER_PARTS.index("sink")] = np.nan
        hr = QM.ledger_head_recall(parts, led["gated"], led["oracle_q_read"],
                                   min_q_share=min_q_share)
        recall[m] = hr["recall"]
        oracle[m] = hr["oracle_recall"]
        sr = hr["step_recall"]
        step_recalls.append(sr[np.isfinite(sr)])
        read_frac[m] = QM.nanmean(led["read_frac"])
        gated_frac[m] = float(led["gated"].mean())
    if oob:
        log.warning("Observation IV: %d survivor ids exceeded the base window axis",
                    oob)
    if src == "differenced_cumulative":
        log.warning(
            "Observation IV: the base npz has no per-head step mass "
            "(step_window_scores_heads) — differencing the fp16 cumulative "
            "per-head scores instead. That is noise past a few hundred steps; "
            "record the base run with parity.record_head_step_mass=true.")

    rows = []
    for i, name in enumerate(QM.LEDGER_PARTS):
        mu, lo, hi = QM.bootstrap_mean_ci(
            QM.group_reduce(ledger[:, i], groups), confidence, n_boot, seed + i)
        wmu = float(QM.nanmean(ledger_win[:, i]))
        rows.append({"part": name, "share_of_head_mass": mu,
                     "ci_lower": lo, "ci_upper": hi,
                     "share_of_window_mass": wmu})
    derived = {
        "decode_missed_mass": ledger[:, QM.LEDGER_PARTS.index("evicted")]
        + ledger[:, QM.LEDGER_PARTS.index("q_skipped")],
        "hard_missed_mass": ledger[:, QM.LEDGER_PARTS.index("evicted")],
    }
    for j, (name, vals) in enumerate(derived.items()):
        mu, lo, hi = QM.bootstrap_mean_ci(
            QM.group_reduce(vals, groups), confidence, n_boot, seed + 50 + j)
        rows.append({"part": name, "share_of_head_mass": mu,
                     "ci_lower": lo, "ci_upper": hi,
                     "share_of_window_mass": np.nan})

    any_gated = bool(np.any(gated_frac > 0))
    head_mean = QM.nanmean(recall, axis=1)                       # [M]
    orc_mean = QM.nanmean(oracle, axis=1)
    worst = np.array([np.nanmin(r) if np.isfinite(r).any() else np.nan
                      for r in recall])
    flat = recall[np.isfinite(recall)]
    steps_flat = (np.concatenate(step_recalls) if step_recalls
                  else np.zeros(0))
    rf = float(QM.nanmean(read_frac))

    def ci(vals: np.ndarray, off: int) -> Tuple[float, float, float]:
        return QM.bootstrap_mean_ci(QM.group_reduce(vals, groups),
                                    confidence, n_boot, seed + 100 + off)

    r_mu, r_lo, r_hi = ci(head_mean, 0)
    o_mu, o_lo, o_hi = ci(orc_mean, 1)
    w_mu, w_lo, w_hi = ci(worst, 2)
    eff = np.divide(head_mean, orc_mean, out=np.full(M, np.nan),
                    where=np.isfinite(orc_mean) & (orc_mean > 0))
    e_mu, e_lo, e_hi = ci(eff, 3)
    recall_summary = {
        "gated": any_gated,
        "gated_step_fraction": float(gated_frac.mean()) if M else np.nan,
        "realised_read_fraction": rf,
        "head_recall_mean": r_mu, "head_recall_ci_lower": r_lo,
        "head_recall_ci_upper": r_hi,
        "oracle_recall_mean": o_mu, "oracle_recall_ci_lower": o_lo,
        "oracle_recall_ci_upper": o_hi,
        "gate_efficiency_mean": e_mu, "gate_efficiency_ci_lower": e_lo,
        "gate_efficiency_ci_upper": e_hi,
        "worst_head_per_trace_mean": w_mu, "worst_head_ci_lower": w_lo,
        "worst_head_ci_upper": w_hi,
        "worst_head_overall": float(flat.min()) if flat.size else np.nan,
        "heads_below_0.99": float(np.mean(flat < 0.99)) if flat.size else np.nan,
        "heads_below_0.90": float(np.mean(flat < 0.90)) if flat.size else np.nan,
        "heads_below_0.50": float(np.mean(flat < 0.50)) if flat.size else np.nan,
        "step_recall_p01": float(np.quantile(steps_flat, 0.01)) if steps_flat.size else np.nan,
        "step_recall_p05": float(np.quantile(steps_flat, 0.05)) if steps_flat.size else np.nan,
        "step_recall_median": float(np.median(steps_flat)) if steps_flat.size else np.nan,
        "head_steps_below_0.50": float(np.mean(steps_flat < 0.5)) if steps_flat.size else np.nan,
        "mass_lift_per_opened_window": (r_mu / rf if np.isfinite(rf) and rf > 0
                                        else np.nan),
        "min_q_share": float(min_q_share),
        "head_mass_source": src,
        "num_query_heads": H, "num_kv_heads": H_kv, "gqa_rep": rep,
    }
    layers = sorted({li for _, li in inp.trace_labels})
    layer_rows = []
    for li in layers:
        sel = np.array([lab[1] == li for lab in inp.trace_labels])
        layer_rows.append({
            "layer": int(li),
            "head_recall_mean": float(QM.nanmean(recall[sel])),
            "oracle_recall_mean": float(QM.nanmean(oracle[sel])),
            "worst_head": float(np.nanmin(recall[sel]))
            if np.isfinite(recall[sel]).any() else np.nan,
            "q_skipped_share": float(QM.nanmean(
                ledger[sel, QM.LEDGER_PARTS.index("q_skipped")])),
            "evicted_share": float(QM.nanmean(
                ledger[sel, QM.LEDGER_PARTS.index("evicted")])),
        })
    return {
        "ledger_table": rows,
        "recall_summary": recall_summary,
        "layer_table": layer_rows,
        "ledger_by_trace": ledger,
        "head_recall": recall,
        "oracle_recall": oracle,
        "step_recall": steps_flat,
    }


# ---------------------------------------------------------------------------
# Observation V — tier dynamics (promotion / demotion / eviction)
# ---------------------------------------------------------------------------


def analyse_tier_dynamics(
    inp: ObservationInputs,
    horizon: int = DEFAULTS["fmm_horizon"],
    deltas: Sequence[int] = (1, 2, 4),
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Observation V: the F / Q / E chain and whether each move paid off."""
    states = inp.states
    groups = inp.trace_group
    R = states.shape[1]
    rate_rows: List[Dict[str, Any]] = []
    first: Optional[Dict[str, Any]] = None
    for i, d in enumerate(int(x) for x in deltas):
        if not 0 < d < R:
            continue
        tr = QM.tier_transitions(states, d)
        first = first or tr
        counts = tr["counts_by_trace"]
        for j, (move, (a, b)) in enumerate(QM.MOVES.items()):
            num = counts[:, a, b]
            den = counts[:, a, :].sum(-1)
            mu, lo, hi = QM.bootstrap_mean_ci(
                QM.group_ratio(num, den, groups), confidence, n_boot,
                seed + 100 * i + j)
            rate_rows.append({
                "delta": d, "move": move,
                "from": QM.STATE_NAMES[a], "to": QM.STATE_NAMES[b],
                "rate_mean": mu, "ci_lower": lo, "ci_upper": hi,
                "pooled_rate": float(tr["pooled_probabilities"][a, b]),
                "count": int(num.sum()),
            })
    entry_rows = []
    if first is not None:
        ent = first["entry_counts"].sum(axis=0).astype(float)
        tot = ent.sum()
        for b in range(3):
            entry_rows.append({"enters_as": QM.STATE_NAMES[b],
                               "fraction": ent[b] / tot if tot > 0 else np.nan,
                               "count": int(ent[b])})
        if first["resurrections"]:
            log.warning(
                "Observation V: %d E -> F/Q pairs. Eviction is permanent in the "
                "cache, so the survivor ids are being mapped onto the wrong "
                "windows — check all_window_ids against the base window axis.",
                first["resurrections"])

    out = QM.transition_outcomes(states, inp.mass_step, inp.event_steps, horizon)
    outcome_rows = []
    for j, move in enumerate(QM.MOVES):
        s_mu, s_lo, s_hi = QM.bootstrap_mean_ci(
            QM.group_reduce(out[f"share_{move}"], groups), confidence, n_boot,
            seed + 700 + j)
        l_mu, l_lo, l_hi = QM.bootstrap_mean_ci(
            QM.group_reduce(out[f"lift_{move}"], groups), confidence, n_boot,
            seed + 800 + j)
        outcome_rows.append({
            "move": move, "count": int(np.nansum(out[f"count_{move}"])),
            "future_mass_share": s_mu, "share_ci_lower": s_lo,
            "share_ci_upper": s_hi,
            "lift": l_mu, "lift_ci_lower": l_lo, "lift_ci_upper": l_hi,
        })
    summary: Dict[str, Any] = {"horizon": int(horizon),
                               "events_scored": int(out["events_scored"])}
    for j, key in enumerate(("swap_gain", "eviction_regret")):
        mu, lo, hi = QM.bootstrap_mean_ci(
            QM.group_reduce(out[key], groups), confidence, n_boot, seed + 900 + j)
        summary.update({f"{key}_mean": mu, f"{key}_ci_lower": lo,
                        f"{key}_ci_upper": hi})
    for j, key in enumerate(("hit_promote", "hit_demote", "hit_evict",
                             "fp_precision")):
        mu, lo, hi = QM.bootstrap_mean_ci(
            QM.group_ratio(out[f"{key}_num"], out[f"{key}_den"], groups),
            confidence, n_boot, seed + 950 + j)
        summary.update({f"{key}_mean": mu, f"{key}_ci_lower": lo,
                        f"{key}_ci_upper": hi,
                        f"{key}_count": int(np.nansum(out[f"{key}_den"]))})
    fid = QM.tier_decision_fidelity(states, inp.rank_scores)
    f_mu, f_lo, f_hi = QM.bootstrap_mean_ci(
        QM.group_reduce(QM.nanmean(fid, axis=1), groups), confidence, n_boot,
        seed + 990)
    summary.update({"decision_fidelity_mean": f_mu,
                    "decision_fidelity_ci_lower": f_lo,
                    "decision_fidelity_ci_upper": f_hi})
    if out["events_scored"] == 0:
        log.warning("Observation V: no routing event has a full %d-step future "
                    "— move outcomes are undefined; lower --fmm-horizon.", horizon)
    return {
        "rate_table": rate_rows,
        "entry_table": entry_rows,
        "outcome_table": outcome_rows,
        "summary": summary,
        "pooled_transition": (first["pooled_probabilities"] if first is not None
                              else np.full((3, 3), np.nan)),
        "transition_counts": (first["counts_by_trace"] if first is not None
                              else np.zeros((0, 3, 3), np.int64)),
        "decision_fidelity": fid,
        "outcomes_by_trace": {k: v for k, v in out.items()
                              if isinstance(v, np.ndarray)},
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


def _read_path_line(g: Dict[str, Any]) -> str:
    path = g.get("read_path", "unknown")
    ratio = g.get("quant_gate_ratio")
    rf = g.get("realised_read_fraction")
    bits = g.get("quant_card_bits")
    knobs = (f"gate ratio {ratio}, realised read fraction "
             f"{'n/a' if rf is None else f'{float(rf):.3f}'}, card bits {bits}")
    return {
        "gated": f"Read path: **gated** — verdict `{g.get('read_gate_verdict')}`, "
                 f"{knobs}; the gate's pick is recorded.",
        "gated-unrecorded": f"Read path: **gated, pick NOT recorded** — {knobs}. "
                            "Observation IV counts the whole int2 tier as read "
                            "and so overstates what the decode read.",
        "ungated": "Read path: **ungated** (eager backend or quant_ratio=0; "
                   "verdict `not-expected`) — the whole int2 tier is read.",
        "NOT-GATED": f"Read path: **NOT GATED** — verdict "
                     f"`{g.get('read_gate_verdict')}`. The run should have gated "
                     "and did not; it is not the shipped method.",
    }.get(path, "Read path: **unknown** — the ours npz predates schema 1.3 "
                "(no `read_gate` verdict). Observation IV counts the whole int2 "
                "tier as read.")


def build_paper_summary(
    inp: ObservationInputs,
    obs1: Dict[str, Any],
    obs2: Dict[str, Any],
    obs3: Dict[str, Dict[str, Any]],
    primary_top_fraction: float = DEFAULTS["primary_top_fraction"],
    primary_target_mass: float = DEFAULTS["primary_target_mass"],
    obs4: Optional[Dict[str, Any]] = None,
    obs5: Optional[Dict[str, Any]] = None,
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
        _read_path_line(g),
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
    if inp.diagnostics.get("base_score_accum_dtype", "float16") != "float32":
        lines += ["", "> **Caveat.** The base npz predates schema 1.3: its "
                  "cumulative scores were accumulated in fp16, which drops most "
                  "decode steps once a window's total passes ~8. The oracle "
                  "ranking behind Observations I, III and V's decision fidelity "
                  "is then close to the prompt's ranking. Re-run BaseParityRunner."]

    lines += ["", "## Observation III: historical importance revives", ""]
    for label, res in obs3.items():
        s = res["quantifiable_result"]
        flips = (("cold→hot", "hot→cold") if label == "oracle" else
                 ("not-fp→fp: pools Q with evicted, see Observation V",
                  "fp→not-fp: pools demotion with eviction"))
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
            f"({flips[0]}), "
            f"P10={_ci_pct(s['P10_mean'], s['P10_ci_lower'], s['P10_ci_upper'])} "
            f"({flips[1]}).",
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
            f"> Holding the low-rank band in int2 instead of dropping it lowers "
            f"Future Missed Mass over the next {obs2['horizon']} decode steps "
            f"from {_pct(fp_only['future_missed_mass_mean'])} to "
            f"{_pct(fp_q['future_missed_mass_mean'])} at the same fp capacity "
            f"(within one run, so not byte-matched: the byte-matched contrast "
            f"is an evict-only run at the same cache_budget, tier study M6).",
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
    if obs4 is not None:
        lines += _obs4_lines(obs4, g)
    if obs5 is not None:
        lines += _obs5_lines(obs5)
    return "\n".join(lines)


def _obs4_lines(obs4: Dict[str, Any], g: Dict[str, Any]) -> List[str]:
    rs = obs4["recall_summary"]
    lines = ["", "## Observation IV: the decode read ledger", "",
             "Each query head's ground-truth attention at every decode step, by "
             "how the decode read it (per head, then averaged).", "",
             "| part | share of head mass | share of window mass |",
             "| --- | --- | --- |"]
    for r in obs4["ledger_table"]:
        lines.append(f"| `{r['part']}` | "
                     f"{_ci_pct(r['share_of_head_mass'], r['ci_lower'], r['ci_upper'])} | "
                     f"{_pct(r['share_of_window_mass'])} |")
    lines.append("")
    if rs["gated"]:
        lines += [
            f"- Per-head gate recall (int2 mass in opened windows / int2 mass): "
            f"{_ci_pct(rs['head_recall_mean'], rs['head_recall_ci_lower'], rs['head_recall_ci_upper'])}"
            f" at a realised read fraction of {_pct(rs['realised_read_fraction'])}"
            f" — {_num(rs['mass_lift_per_opened_window'], 2)}x the mass per "
            "opened window of a uniform pick.",
            f"- Hindsight pick of the same size (best per-step group share): "
            f"{_ci_pct(rs['oracle_recall_mean'], rs['oracle_recall_ci_lower'], rs['oracle_recall_ci_upper'])}"
            f"; gate efficiency {_ci_pct(rs['gate_efficiency_mean'], rs['gate_efficiency_ci_lower'], rs['gate_efficiency_ci_upper'])}.",
            f"- Worst head: {_pct(rs['worst_head_overall'])} overall; "
            f"{_pct(rs['heads_below_0.99'])} of (trace, head) pairs under 99%, "
            f"{_pct(rs['heads_below_0.50'])} under 50%. Head-steps holding "
            f">= {_pct(rs['min_q_share'])} of their mass in int2: median recall "
            f"{_pct(rs['step_recall_median'])}, 5th percentile "
            f"{_pct(rs['step_recall_p05'])}, {_pct(rs['head_steps_below_0.50'])} "
            "under 50%.",
            "",
            f"> At a {_pct(rs['realised_read_fraction'], 0)} read ratio the gate "
            f"opens the int2 windows holding {_pct(rs['head_recall_mean'])} of "
            f"each query head's int2-tier attention on real decoding, "
            f"{_pct(rs['gate_efficiency_mean'])} of what a hindsight pick of "
            "the same size reaches.",
            ""]
    else:
        lines += ["- The gate did not run (or was not recorded) in this ours run, "
                  "so every int2 window counts as read and recall is 1 by "
                  "construction.", ""]
    if rs["head_mass_source"] != "recorded_head_step_mass":
        lines += ["> **Caveat.** Per-head step mass was differenced from the fp16 "
                  "cumulative scores; record the base run with "
                  "`parity.record_head_step_mass=true`.", ""]
    return lines


def _obs5_lines(obs5: Dict[str, Any]) -> List[str]:
    sm = obs5["summary"]
    lines = ["", "## Observation V: tier dynamics (F / Q / E)", "",
             "| lag | move | rate | pooled | count |", "| --- | --- | --- | --- | --- |"]
    for r in obs5["rate_table"]:
        lines.append(f"| {r['delta']} | `{r['from']}→{r['to']}` {r['move']} | "
                     f"{_ci_pct(r['rate_mean'], r['ci_lower'], r['ci_upper'])} | "
                     f"{_pct(r['pooled_rate'])} | {r['count']} |")
    if obs5["entry_table"]:
        lines += ["", "Windows leaving the local tail enter as: " + ", ".join(
            f"{e['enters_as']} {_pct(e['fraction'])}" for e in obs5["entry_table"])
            + "."]
    lines += ["", f"Did the move land on future attention? (next {sm['horizon']} "
              f"steps, {sm['events_scored']} events)", "",
              "| move | count | share of candidates' future mass | lift |",
              "| --- | --- | --- | --- |"]
    for r in obs5["outcome_table"]:
        lines.append(f"| `{r['move']}` | {r['count']} | "
                     f"{_ci_pct(r['future_mass_share'], r['share_ci_lower'], r['share_ci_upper'])} | "
                     f"{_ci_num(r['lift'], r['lift_ci_lower'], r['lift_ci_upper'], 2)} |")
    lines += [
        "",
        f"- Swap gain (lift of promoted − lift of demoted): "
        f"{_ci_num(sm['swap_gain_mean'], sm['swap_gain_ci_lower'], sm['swap_gain_ci_upper'], 2)}.",
        f"- Hindsight hit rates: promotions into the best fp set "
        f"{_ci_pct(sm['hit_promote_mean'], sm['hit_promote_ci_lower'], sm['hit_promote_ci_upper'])}"
        f" (n={sm['hit_promote_count']}); demotions out of it "
        f"{_ci_pct(sm['hit_demote_mean'], sm['hit_demote_ci_lower'], sm['hit_demote_ci_upper'])}"
        f"; evictions outside the best retained set "
        f"{_ci_pct(sm['hit_evict_mean'], sm['hit_evict_ci_lower'], sm['hit_evict_ci_upper'])}"
        f"; fp precision {_ci_pct(sm['fp_precision_mean'], sm['fp_precision_ci_lower'], sm['fp_precision_ci_upper'])}.",
        f"- Eviction regret (candidates' future mass on windows evicted at the "
        f"event): {_ci_pct(sm['eviction_regret_mean'], sm['eviction_regret_ci_lower'], sm['eviction_regret_ci_upper'])}.",
        f"- Decision fidelity (Jaccard of the fp set vs the ground-truth ranking "
        f"of the same survivors): "
        f"{_ci_pct(sm['decision_fidelity_mean'], sm['decision_fidelity_ci_lower'], sm['decision_fidelity_ci_upper'])}.",
        ""]
    promote = next((r for r in obs5["outcome_table"] if r["move"] == "promote"), None)
    if promote and promote["count"] > 0 and np.isfinite(float(promote["lift"])):
        lines += [f"> Promotion is not bookkeeping noise: a window promoted from "
                  f"int2 back to full precision receives "
                  f"{_num(promote['lift'], 2)}x the average future attention of "
                  f"the windows the eviction decided on, and "
                  f"{_pct(sm['hit_promote_mean'])} of promotions land in the "
                  "hindsight-best fp set.", ""]
    return lines


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
        if value.ndim == 0:
            return _json_safe(value.item())
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
    obs4: Optional[Dict[str, Any]] = None,
    obs5: Optional[Dict[str, Any]] = None,
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

    if obs4 is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        rows = [r for r in obs4["ledger_table"] if r["part"] in QM.LEDGER_PARTS]
        left = 0.0
        for r in rows:
            v = float(r["share_of_head_mass"]) if np.isfinite(
                float(r["share_of_head_mass"])) else 0.0
            axes[0].barh([0], [v], left=left, label=r["part"])
            left += v
        axes[0].set(xlim=(0, 1), yticks=[], xlabel="Share of each head's mass",
                    title="Decode read ledger")
        axes[0].legend(fontsize=7, ncol=4, loc="upper center",
                       bbox_to_anchor=(0.5, -0.25))
        lt = obs4["layer_table"]
        if lt:
            axes[1].plot([r["layer"] for r in lt],
                         [r["head_recall_mean"] for r in lt], "o-", label="gate")
            axes[1].plot([r["layer"] for r in lt],
                         [r["oracle_recall_mean"] for r in lt], "s--",
                         label="hindsight pick, same size")
            axes[1].plot([r["layer"] for r in lt],
                         [r["worst_head"] for r in lt], "v:", label="worst head")
        axes[1].set(xlabel="Layer", ylabel="Per-head int2 recall", ylim=(0, 1.02))
        axes[1].legend(fontsize=8)
        hr = obs4["head_recall"][np.isfinite(obs4["head_recall"])]
        if hr.size:
            axes[2].hist(hr, bins=np.linspace(0, 1, 41))
        axes[2].set(xlabel="Per-(trace, head) recall", ylabel="count",
                    title="Recall distribution")
        p = out_dir / "observation4_read_ledger.pdf"
        fig.tight_layout(); fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        written.append(p)

    if obs5 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        pooled = obs5["pooled_transition"]
        im = axes[0].imshow(pooled, vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                v = pooled[a, b]
                axes[0].text(b, a, f"{v:.3f}" if np.isfinite(v) else "nan",
                             ha="center", va="center")
        axes[0].set_xticks(range(3), [f"to {n}" for n in QM.STATE_NAMES])
        axes[0].set_yticks(range(3), [f"from {n}" for n in QM.STATE_NAMES])
        axes[0].set_title("Tier transitions (lag 1)")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        rows = obs5["outcome_table"]
        x = np.arange(len(rows))
        lift = np.array([float(r["lift"]) for r in rows])
        lo = np.array([float(r["lift_ci_lower"]) for r in rows])
        hi = np.array([float(r["lift_ci_upper"]) for r in rows])
        err = np.vstack([np.nan_to_num(lift - lo), np.nan_to_num(hi - lift)])
        axes[1].bar(x, np.nan_to_num(lift), yerr=err, capsize=3)
        axes[1].axhline(1.0, color="k", linestyle=":")
        axes[1].set_xticks(x, [r["move"] for r in rows], rotation=30, ha="right")
        axes[1].set(ylabel="Future-mass lift over candidates",
                    title=f"Move outcomes (H={obs5['summary']['horizon']})")
        p = out_dir / "observation5_tier_dynamics.pdf"
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
    obs4: Optional[Dict[str, Any]] = None,
    obs5: Optional[Dict[str, Any]] = None,
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

    if obs4 is not None:
        _write_csv(out_dir / "observation4_ledger_table.csv", obs4["ledger_table"])
        _write_csv(out_dir / "observation4_recall_summary.csv",
                   [obs4["recall_summary"]])
        _write_csv(out_dir / "observation4_layer_table.csv", obs4["layer_table"])
    if obs5 is not None:
        _write_csv(out_dir / "observation5_rate_table.csv", obs5["rate_table"])
        _write_csv(out_dir / "observation5_entry_table.csv", obs5["entry_table"])
        _write_csv(out_dir / "observation5_outcome_table.csv", obs5["outcome_table"])
        _write_csv(out_dir / "observation5_summary.csv", [obs5["summary"]])

    summary_md = build_paper_summary(inp, obs1, obs2, obs3, obs4=obs4, obs5=obs5)
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
    if obs4 is not None:
        results["observation4"] = {k: obs4[k] for k in
                                   ("ledger_table", "recall_summary", "layer_table")}
    if obs5 is not None:
        results["observation5"] = {
            "rate_table": obs5["rate_table"], "entry_table": obs5["entry_table"],
            "outcome_table": obs5["outcome_table"], "summary": obs5["summary"],
            "pooled_transition": obs5["pooled_transition"]}
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
    if obs4 is not None:
        arrays["ledger_by_trace"] = obs4["ledger_by_trace"]
        arrays["ledger_parts"] = np.array(QM.LEDGER_PARTS)
        arrays["head_recall"] = obs4["head_recall"]
        arrays["oracle_recall"] = obs4["oracle_recall"]
    if obs5 is not None:
        arrays["tier_states"] = inp.states
        arrays["transition_counts"] = obs5["transition_counts"]
        arrays["decision_fidelity"] = obs5["decision_fidelity"]
        for k, v in obs5["outcomes_by_trace"].items():
            arrays[f"outcome__{k}"] = v
    npz_path = out_dir / "qevict_observations.npz"
    np.savez_compressed(str(npz_path), **arrays)
    # Sidecar, matching every other suite's output convention.
    with open(npz_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe({**meta, "geometry": inp.geometry,
                              "diagnostics": inp.diagnostics}), f, indent=2,
                  default=str)

    if figures:
        make_figures(inp, obs1, obs2, obs3, out_dir, obs4=obs4, obs5=obs5)
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
    min_q_share: float = DEFAULTS["min_q_share"],
    tier_deltas: Sequence[int] = DEFAULTS["tier_deltas"],
) -> Dict[str, Any]:
    """End-to-end: load the parity pair, compute all five observations, write."""
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

    rp = read_path_info(ours["metadata"])
    if rp["read_path"] == "NOT-GATED":
        log.warning(
            "the ours run should have gated and did not (read_gate verdict %r). "
            "It is not the shipped method; Observation IV describes a different "
            "read path than the one the paper reports.", rp["read_gate_verdict"])
    elif rp["read_path"] in ("gated-unrecorded", "unknown"):
        log.warning(
            "the ours npz does not record the read gate's pick (read path %r) — "
            "Observation IV counts the whole int2 tier as read. Re-run "
            "OursParityRunner (schema >= 1.3) on the flash backend.",
            rp["read_path"])

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
    obs4 = analyse_decode_reads(base, ours, inp, min_q_share=min_q_share,
                                confidence=confidence, n_boot=n_boot,
                                seed=seed + 300)
    obs5 = analyse_tier_dynamics(inp, horizon=fmm_horizon, deltas=tier_deltas,
                                 confidence=confidence, n_boot=n_boot,
                                 seed=seed + 400)

    meta = {
        # 1.1: Observations IV (read ledger) and V (tier dynamics), read path.
        "schema_version": "1.1",
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
        "head_mass_source": obs4["recall_summary"]["head_mass_source"],
        "min_q_share": float(min_q_share),
        "tier_deltas": [int(d) for d in tier_deltas],
        **rp,
        "base_score_accum_dtype": inp.diagnostics.get("base_score_accum_dtype"),
    }
    npz_path = write_outputs(inp, obs1, obs2, obs3, Path(out_dir), meta, figures,
                             obs4=obs4, obs5=obs5)
    return {"inputs": inp, "observation1": obs1, "observation2": obs2,
            "observation3": obs3, "observation4": obs4, "observation5": obs5,
            "npz_path": npz_path, "metadata": meta}


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
    ap.add_argument("--min-q-share", type=float, default=DEFAULTS["min_q_share"],
                    help="Observation IV: minimum int2 share of a head's mass for "
                         "a head-step to count toward gate recall.")
    ap.add_argument("--tier-deltas", type=int, nargs="+",
                    default=list(DEFAULTS["tier_deltas"]),
                    help="Observation V: F/Q/E transition lags (events).")
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
        min_q_share=args.min_q_share,
        tier_deltas=args.tier_deltas,
    )
    print(build_paper_summary(
        res["inputs"], res["observation1"], res["observation2"],
        res["observation3"], obs4=res["observation4"],
        obs5=res["observation5"]))
    print(f"\nSaved all outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    _cli_main()
