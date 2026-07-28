"""QEvict observation metrics — concentration, future-missed-mass, episode LIR.

Third sibling of :mod:`utils.metrics` (pure vectorised Jaccard) and
:mod:`utils.sticky_metrics` (sequential Sticky-K policy simulation).  This
module holds the three *observation* statistics that motivate a two-tier,
window-level, promotion-capable cache:

  **Observation I — skewed importance.**
      :func:`concentration_curve`.  Sort a step's window scores descending and
      ask what fraction of the total mass the top ``p`` fraction of windows
      holds.  ``C(p) - p`` is the concentration gap over a uniform allocation;
      the fp and fp+Q capacity fractions can be read off the same curve.

  **Observation II — window-level decisions.**
      :func:`future_missed_mass` and :func:`selection_churn`.  FMM scores a
      routing decision against the attention that *actually arrives* in the
      next ``H`` decode steps, so it is a forward-looking complement to
      ``sticky_metrics``' instantaneous ``missed_mass``.  Churn is the Jaccard
      distance between consecutive accessible sets — decision stability.

  **Observation III — historical importance revives.**
      :func:`episode_lir` and :func:`binary_transition`.  ``episode_lir`` is
      the *episode*-based Global LIR: each maximal inactive run counts once,
      becomes eligible after ``m`` consecutive misses, and is rescued only if a
      hit lands within the next ``H`` events — or anywhere later in the trace,
      uncapped and uncensored, with ``H=None``.  This differs deliberately from
      ``sticky_metrics.lir_counts``, which counts every ``(event, window)``
      lookback pair and accepts a rescue arbitrarily far in the future — see
      the comparison table in ``modules/evaluation/qevict_observations.py``.
      :func:`binary_transition` adds the lag-``delta`` flip probabilities:
      ``P01`` (cold → hot, the promotion argument) and ``P10`` (hot → cold, the
      demotion argument).

Conventions shared by every function here:

  ``M``  independent traces — the bootstrap axis (see ``trace_axis`` in the
         adapter; ``(sample, layer)`` pairs by default).
  ``R``  routing events (the decode steps at which eviction actually fired).
  ``T``  decode steps.
  ``W``  windows (the "units" a decision is made over).

Selection matrices are trinary ``{-1, 0, 1}``: ``-1`` = the window is not an
observable candidate at that event (padded, not yet created, or still in the
protected local tail), ``0`` = candidate but not selected, ``1`` = selected.

Loops are fine in this module (as in ``sticky_metrics``); only
``utils/metrics.py`` is required to stay loop-free.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# NaN-safe helpers & bootstrap
# ---------------------------------------------------------------------------


def nanmean(x: Array, axis=None) -> Array:
    """``np.nanmean`` without the all-NaN RuntimeWarning (empty slice → NaN)."""
    x = np.asarray(x, dtype=float)
    count = np.sum(np.isfinite(x), axis=axis)
    total = np.nansum(x, axis=axis)
    out = np.full(np.shape(total), np.nan, dtype=float)
    np.divide(total, count, out=out, where=count > 0)
    return out


def bootstrap_mean_ci(
    values: Array,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap of a mean over independent traces.

    Returns ``(mean, ci_lower, ci_upper)``; the interval is NaN when fewer than
    two finite values are supplied (one trace carries no resampling
    information), which is the common case for a single-sample parity run
    bootstrapped over the sample axis.
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(x.mean())
    if x.size == 1 or n_boot <= 0:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = x[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return mean, float(lo), float(hi)


def bootstrap_curve_ci(
    curves: Array,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[Array, Array, Array]:
    """Pointwise percentile bootstrap of a ``[M, X]`` curve bundle over ``M``."""
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError(f"curves must be [M, X]; got shape {curves.shape}")
    width = curves.shape[1]
    curves = curves[np.any(np.isfinite(curves), axis=1)]
    if curves.size == 0:
        # Every row censored (e.g. no event has a full FMM horizon) — an
        # all-NaN curve is the honest answer, not an exception.
        nan = np.full(width, np.nan)
        return nan, nan.copy(), nan.copy()
    mean = nanmean(curves, axis=0)
    if curves.shape[0] <= 1 or n_boot <= 0:
        nan = np.full_like(mean, np.nan)
        return mean, nan, nan
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, curves.shape[1]), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, curves.shape[0], size=curves.shape[0])
        boot[b] = nanmean(curves[idx], axis=0)
    alpha = (1.0 - confidence) / 2.0
    with warnings.catch_warnings():
        # A fully censored column (e.g. an event with no future horizon) is an
        # expected NaN here, not a numerical problem.
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        lo = np.nanquantile(boot, alpha, axis=0)
        hi = np.nanquantile(boot, 1.0 - alpha, axis=0)
    return mean, lo, hi


def group_reduce(values: Array, groups: Optional[Array]) -> Array:
    """Average per-trace values within each bootstrap group.

    ``values`` is ``[M]`` or ``[M, X]``; ``groups`` is ``[M]`` of group ids
    (``None`` = every trace is its own group).  Grouping *after* the per-trace
    statistic is what lets the same run be bootstrapped over samples or over
    ``(sample, layer)`` traces without recomputing anything: collapsing the
    layer axis earlier would require averaging accessible *sets*, which is not
    defined.
    """
    values = np.asarray(values, dtype=float)
    if groups is None:
        return values
    g = np.asarray(groups, dtype=int).reshape(-1)
    if g.size != values.shape[0]:
        raise ValueError(
            f"groups length {g.size} != leading axis {values.shape[0]}")
    uniq = np.unique(g)
    out = np.stack([nanmean(values[g == u], axis=0) for u in uniq], axis=0)
    return out


def group_ratio(
    numerator: Array,
    denominator: Array,
    groups: Optional[Array] = None,
) -> Array:
    """Per-group ratio-of-sums (the correct aggregation for count-based rates).

    Mirrors ``sticky_metrics``' convention of summing eligible/rescued counts
    before dividing, rather than averaging per-trace ratios.
    """
    num = np.asarray(numerator, dtype=float).reshape(-1)
    den = np.asarray(denominator, dtype=float).reshape(-1)
    if num.size != den.size:
        raise ValueError("numerator and denominator must have equal length")
    g = np.arange(num.size) if groups is None else np.asarray(groups, int).reshape(-1)
    uniq = np.unique(g)
    out = np.full(uniq.size, np.nan)
    for i, u in enumerate(uniq):
        sel = g == u
        d = den[sel].sum()
        if d > 0:
            out[i] = num[sel].sum() / d
    return out


# ---------------------------------------------------------------------------
# Observation I — skewed importance
# ---------------------------------------------------------------------------


def concentration_curve(
    scores: Array,
    valid: Array,
    report_fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.40),
    target_masses: Sequence[float] = (0.50, 0.80, 0.90),
    curve_points: int = 101,
) -> Dict[str, object]:
    """Cumulative-importance-mass concentration per ``(trace, event)``.

    Parameters
    ----------
    scores : np.ndarray ``[M, R, W]``
        Non-negative importance scores (the signal the policy ranks on).
    valid : np.ndarray[bool] ``[M, R, W]``
        Which windows exist at that event.  Invalid columns are excluded from
        both the ranking and the total, so padding cannot dilute the curve.
    report_fractions : sequence of float
        Window fractions ``p`` at which to report captured mass ``C(p)``.
    target_masses : sequence of float
        Mass targets ``q`` for which to report the required window fraction.
    curve_points : int
        Grid resolution of the returned curve on ``[0, 1]``.

    Returns
    -------
    dict
      ``fraction_grid`` ``[X]``, ``curve`` ``[M, R, X]`` (NaN at skipped
      events), ``mass_at`` ``{p: [M, R]}``, ``coverage_at`` ``{q: [M, R]}``,
      ``num_valid`` ``[M, R]``.
    """
    scores = np.asarray(scores, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if scores.ndim != 3:
        raise ValueError(f"scores must be [M, R, W]; got {scores.shape}")
    if valid.shape != scores.shape:
        raise ValueError("valid must have the same shape as scores")
    if np.any(scores[valid] < 0):
        raise ValueError("scores must be non-negative")

    grid = np.linspace(0.0, 1.0, int(curve_points))

    # Fully vectorised over (trace, event): invalid columns are zeroed, so they
    # sort to the tail and contribute nothing to the total.  Past the n-th
    # entry the cumulative fraction is pinned at 1.0, which makes the coverage
    # count `(cum < q).sum()` automatically ignore them.
    masked = np.where(valid & np.isfinite(scores), scores, 0.0)
    num_valid = np.count_nonzero(valid & np.isfinite(scores), axis=-1)   # [M, R]
    total = masked.sum(axis=-1, keepdims=True)                           # [M, R, 1]
    ordered = -np.sort(-masked, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cum = np.cumsum(ordered, axis=-1) / total
    ok = (num_valid > 0) & (total[..., 0] > 0)                           # [M, R]
    n = np.maximum(num_valid, 1)[..., None]                              # [M, R, 1]

    def _at(fracs: Array) -> Array:
        k = np.clip(np.ceil(fracs * n).astype(np.int64), 1, n) - 1
        out = np.take_along_axis(cum, k, axis=-1)
        return np.where(ok[..., None], out, np.nan)

    curve = _at(grid[None, None, :])
    curve[..., 0] = np.where(ok, 0.0, np.nan)
    mass_vals = _at(np.asarray([float(p) for p in report_fractions])[None, None, :])
    mass_at = {float(p): mass_vals[..., i].copy()
               for i, p in enumerate(report_fractions)}
    coverage_at = {}
    for q in target_masses:
        k = np.count_nonzero(cum < float(q), axis=-1) + 1
        cov = np.minimum(k, num_valid) / np.maximum(num_valid, 1)
        coverage_at[float(q)] = np.where(ok, cov, np.nan)

    return {
        "fraction_grid": grid,
        "curve": curve,
        "mass_at": mass_at,
        "coverage_at": coverage_at,
        "num_valid": num_valid,
    }


# ---------------------------------------------------------------------------
# Observation II — window-level decisions
# ---------------------------------------------------------------------------


def future_missed_mass(
    step_mass: Array,
    accessible: Array,
    event_steps: Sequence[int],
    horizon: int,
    creation_steps: Optional[Array] = None,
    require_full_horizon: bool = True,
) -> Array:
    """Fraction of the *next* ``H`` steps' attention mass sitting on drops.

    For each trace and routing event at decode step ``t``::

        FMM = Σ_{t' = t+1}^{t+H} Σ_w mass[t', w] · [w existed ∧ w dropped]
              ─────────────────────────────────────────────────────────────
              Σ_{t' = t+1}^{t+H} Σ_w mass[t', w] · [w existed]

    The denominator counts only windows that already existed at the decision,
    so future tokens (which no policy could have kept) never inflate the score.

    Parameters
    ----------
    step_mass : np.ndarray ``[M, T, W]``
        Per-step (**not** cumulative) attention mass per window.
    accessible : np.ndarray[bool] ``[M, R, W]``
        Accessible set immediately after each routing event.
    event_steps : sequence of int ``[R]``
        Decode step of each routing event.
    horizon : int
        ``H`` — how many future decode steps to score against.
    creation_steps : np.ndarray, optional
        ``[W]`` or ``[M, W]`` first step at which each window exists.  Defaults
        to ``creation[w] = w`` (position ``i`` created at step ``i``).
    require_full_horizon : bool
        ``True`` (default) returns NaN for events with fewer than ``H`` future
        steps recorded, so late events are *excluded* rather than silently
        scored over a shorter — and therefore easier — horizon.  ``False``
        truncates instead.

    Returns
    -------
    np.ndarray ``[M, R]`` — NaN where the event is censored or carries no mass.
    """
    mass = np.asarray(step_mass, dtype=float)
    acc = np.asarray(accessible, dtype=bool)
    if mass.ndim != 3 or acc.ndim != 3:
        raise ValueError("step_mass and accessible must both be 3-D")
    M, T, W = mass.shape
    if acc.shape[0] != M or acc.shape[2] != W:
        raise ValueError(
            f"accessible must be [M, R, W] = [{M}, R, {W}]; got {acc.shape}")
    steps = np.asarray(event_steps, dtype=int)
    if steps.size != acc.shape[1]:
        raise ValueError("event_steps length must equal accessible.shape[1]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if creation_steps is None:
        created = np.broadcast_to(np.arange(W)[None, :], (M, W))
    else:
        created = np.asarray(creation_steps, dtype=int)
        if created.ndim == 1:
            created = np.broadcast_to(created[None, :], (M, W))
        if created.shape != (M, W):
            raise ValueError("creation_steps must be [W] or [M, W]")

    out = np.full((M, steps.size), np.nan)
    for r, t in enumerate(steps):
        end = min(int(t) + horizon + 1, T)
        if end <= t + 1:
            continue
        if require_full_horizon and end < t + horizon + 1:
            continue
        # Sum the horizon first, then contract with the masks — avoids
        # materialising an [M, h, W] product at every event.
        future = mass[:, int(t) + 1:end, :].sum(axis=1)  # [M, W]
        eligible = created <= int(t)                     # [M, W]
        dropped = eligible & ~acc[:, r]
        num = np.einsum("mw,mw->m", future, dropped.astype(future.dtype))
        den = np.einsum("mw,mw->m", future, eligible.astype(future.dtype))
        out[:, r] = np.divide(num, den, out=np.full(M, np.nan), where=den > 0)
    return out


def selection_churn(accessible: Array) -> Array:
    """Jaccard *distance* between consecutive accessible sets → ``[M, R-1]``.

    ``0`` = the routing event changed nothing; ``1`` = disjoint sets.  Empty on
    both sides counts as no change (similarity 1), matching
    ``utils.metrics.jaccard_topk``'s empty-set convention.
    """
    acc = np.asarray(accessible, dtype=bool)
    if acc.ndim != 3:
        raise ValueError(f"accessible must be [M, R, W]; got {acc.shape}")
    if acc.shape[1] < 2:
        return np.zeros((acc.shape[0], 0), dtype=float)
    inter = np.sum(acc[:, :-1] & acc[:, 1:], axis=-1)
    union = np.sum(acc[:, :-1] | acc[:, 1:], axis=-1)
    sim = np.divide(inter, union, out=np.ones_like(inter, dtype=float),
                    where=union > 0)
    return 1.0 - sim


# ---------------------------------------------------------------------------
# Granularity policies (the fine-vs-coarse decision comparison)
# ---------------------------------------------------------------------------


def pool_scores(scores: Array, pool: int) -> Array:
    """Sum adjacent unit scores into blocks of ``pool`` → ``[..., ceil(W/pool)]``."""
    scores = np.asarray(scores, dtype=float)
    if pool <= 0:
        raise ValueError("pool must be positive")
    if pool == 1:
        return scores.copy()
    W = scores.shape[-1]
    pad = (-W) % pool
    if pad:
        scores = np.pad(scores, [(0, 0)] * (scores.ndim - 1) + [(0, pad)])
    new_shape = scores.shape[:-1] + (scores.shape[-1] // pool, pool)
    return scores.reshape(new_shape).sum(-1)


def granularity_accessible(
    rank_scores: Array,
    event_steps: Sequence[int],
    w_act: Array,
    ew_act: Array,
    budget_units: int,
    pool: int = 1,
) -> Array:
    """Top-``N`` retention at a chosen decision granularity → ``[M, R, W]``.

    Ranks the evictable band by ``rank_scores`` and keeps the strongest units
    (``pool == 1``) or the strongest *blocks* of ``pool`` adjacent units, then
    expands the choice back to unit granularity.  The protected local tail
    ``[ew_act, w_act)`` is always accessible, exactly as in the production
    policy.

    Both granularities retain ``(budget_units // pool) * pool`` evictable units,
    so a fine-vs-coarse comparison built from two calls is byte-matched by
    construction — the only difference is the granularity of the decision.

    Parameters
    ----------
    rank_scores : np.ndarray ``[M, R, W]``
        Ranking signal at each event (e.g. cumulative head-mean window scores).
    event_steps, w_act, ew_act
        Routing-event steps ``[R]`` and the per-*step* window / evictable counts
        ``[T]`` from :func:`utils.sticky_metrics.flush_geometry`.
    budget_units : int
        Evictable unit budget before rounding to whole blocks.
    pool : int
        Decision granularity in units (``1`` = per-unit decisions).
    """
    rank = np.asarray(rank_scores, dtype=float)
    if rank.ndim != 3:
        raise ValueError(f"rank_scores must be [M, R, W]; got {rank.shape}")
    steps = np.asarray(event_steps, dtype=int)
    if steps.size != rank.shape[1]:
        raise ValueError("event_steps length must equal rank_scores.shape[1]")
    if pool <= 0:
        raise ValueError("pool must be positive")
    M, R, W = rank.shape
    keep_units = (int(budget_units) // pool) * pool

    out = np.zeros((M, R, W), dtype=bool)
    for r, t in enumerate(steps):
        ew = int(ew_act[int(t)])
        wv = int(w_act[int(t)])
        if wv > ew:
            out[:, r, ew:min(wv, W)] = True              # protected local tail
        if ew <= 0 or keep_units <= 0:
            continue
        band = rank[:, r, :ew]                            # [M, ew]
        if pool == 1:
            k = min(keep_units, ew)
            sel = np.argsort(-band, axis=-1, kind="stable")[:, :k]
            np.put_along_axis(out[:, r, :ew], sel, True, axis=-1)
            continue
        blocks = pool_scores(band, pool)                  # [M, ceil(ew/pool)]
        nb = blocks.shape[-1]
        kb = min(keep_units // pool, nb)
        sel_b = np.argsort(-blocks, axis=-1, kind="stable")[:, :kb]
        chosen = np.zeros((M, nb), dtype=bool)
        np.put_along_axis(chosen, sel_b, True, axis=-1)
        out[:, r, :ew] |= np.repeat(chosen, pool, axis=-1)[:, :ew]
    return out


# ---------------------------------------------------------------------------
# Observation III — historical importance revives
# ---------------------------------------------------------------------------


def infer_creation_events(selected: Array) -> Array:
    """First event at which each window is an observable candidate → ``[M, W]``.

    ``R`` (one past the last event) for windows that never become candidates.
    """
    sel = np.asarray(selected, dtype=int)
    M, R, W = sel.shape
    observable = sel >= 0
    first = np.argmax(observable, axis=1)                 # 0 when never observable
    never = ~observable.any(axis=1)
    creation = np.where(never, R, first).astype(int)
    return creation


def episode_lir(
    selected: Array,
    inactivity: int,
    horizon: Optional[int],
    creation_events: Optional[Array] = None,
) -> Dict[str, object]:
    """Episode-based Global LIR: do *cold* windows regain importance?

    Each maximal run of ``0``s counts **once**.  A run becomes *eligible* at its
    ``inactivity``-th consecutive miss and is *rescued* if a ``1`` lands within
    the following ``horizon`` events.  Episodes whose horizon would run past the
    end of the trace are right-censored — dropped from numerator *and*
    denominator rather than counted as failures.

    With ``horizon=None`` there is no cap: an episode is rescued iff a hit lands
    *anywhere* later in the trace, and nothing is censored (there is no fixed
    horizon left to run out of).  That is a different question — "was it **ever**
    rescued" rather than "was it rescued in time" — and it is the honest
    denominator when the point is how much a fixed ``H`` was undercounting.

    Contrast with :func:`utils.sticky_metrics.lir_counts`, which counts every
    ``(event, window)`` lookback pair (so one long inactive run contributes many
    eligible pairs) and accepts a rescue at *any* later event (so it has no
    horizon and no censoring).

    Parameters
    ----------
    selected : np.ndarray ``[M, R, W]``
        Trinary selection matrix (``-1`` unobservable / ``0`` miss / ``1`` hit).
    inactivity : int
        ``m`` above; must be positive.
    horizon : int or None
        ``H`` above; must be positive when given.  ``None`` means *uncapped* —
        score every eligible episode against however much trace remains, with no
        right-censoring and an unbounded ``time_to_revival``.  It is **not** the
        same as passing a very large ``H``: a large finite ``H`` still censors
        every episode whose window would overrun the trace end, so it shrinks the
        denominator as ``H`` grows, while ``None`` keeps all of it.
    creation_events : np.ndarray, optional
        ``[W]`` or ``[M, W]``; defaults to :func:`infer_creation_events`.

    Returns
    -------
    dict
      ``eligible_by_trace``/``rescued_by_trace``/``rate_by_trace`` ``[M]``,
      ``eligible``/``rescued`` totals, ``global_rate`` (ratio of sums),
      ``time_to_revival`` ``[n_rescued]``, and per-episode columns
      ``episode_trace``/``_window``/``_start``/``_end``/``_length``/
      ``_eligibility_event``/``_rescued``/``_time_to_revival``.
    """
    x = np.asarray(selected, dtype=int)
    if x.ndim != 3:
        raise ValueError(f"selected must be [M, R, W]; got {x.shape}")
    if not np.all(np.isin(x, (-1, 0, 1))):
        raise ValueError("selected must contain only -1, 0, 1")
    if inactivity <= 0 or (horizon is not None and horizon <= 0):
        raise ValueError("inactivity and horizon must be positive")
    M, R, W = x.shape

    if creation_events is None:
        creation = infer_creation_events(x)
    else:
        creation = np.asarray(creation_events, dtype=int)
        if creation.ndim == 1:
            creation = np.broadcast_to(creation[None, :], (M, W))
        if creation.shape != (M, W):
            raise ValueError("creation_events must be [W] or [M, W]")

    # ── vectorised episode enumeration ───────────────────────────────────
    # A window is usable only if it never falls back to -1 after creation, so
    # inside its observable span every event is a miss (0) or a hit (1).  A
    # maximal miss-run therefore ends exactly at the next hit, which turns the
    # whole episode calculus into closed form on `next_hit`:
    #     run_len  = next_hit - run_start          (R - run_start if no hit)
    #     eligible = run_len >= m  and  run_start + m - 1 + H < R
    #     rescued  = eligible and run_len <= m - 1 + H
    #     ttr      = run_len - m + 1
    # H = None drops both horizon terms — eligibility keeps the run-length test
    # only, and rescue becomes "a hit exists at all" (next_hit < R).  `ttr` is
    # the same expression either way, just unbounded above.
    ev = np.arange(R, dtype=np.int32)
    window_ok = ~np.any((x < 0) & (ev[None, :, None] >= creation[:, None, :]),
                        axis=1)                                        # [M, W]
    born = ev[None, :, None] >= creation[:, None, :]
    is_miss = (x == 0) & window_ok[:, None, :] & born
    prev_miss = np.concatenate(
        [np.zeros((M, 1, W), dtype=bool), is_miss[:, :-1, :]], axis=1)
    starts = is_miss & ~prev_miss                                      # [M, R, W]

    hit_idx = np.where((x == 1) & window_ok[:, None, :] & born,
                       ev[None, :, None], np.int32(R))
    next_hit = np.minimum.accumulate(hit_idx[:, ::-1, :], axis=1)[:, ::-1, :]

    m_idx, r_idx, w_idx = np.nonzero(starts)
    run_start = r_idx.astype(np.int64)
    nh = next_hit[m_idx, r_idx, w_idx].astype(np.int64)
    run_len = nh - run_start
    if horizon is None:
        is_eligible = run_len >= inactivity
        is_rescued = is_eligible & (nh < R)
    else:
        elig_event = run_start + inactivity - 1
        is_eligible = (run_len >= inactivity) & (elig_event + horizon < R)
        is_rescued = is_eligible & (run_len <= inactivity - 1 + horizon)

    keep = is_eligible
    order = np.lexsort((run_start[keep], w_idx[keep], m_idx[keep]))
    ep_trace = m_idx[keep][order].astype(int)
    ep_window = w_idx[keep][order].astype(int)
    ep_start = run_start[keep][order]
    ep_len = run_len[keep][order]
    ep_rescued = is_rescued[keep][order]
    ep_ttr = np.where(ep_rescued, ep_len - inactivity + 1, -1)

    eligible = np.bincount(ep_trace, minlength=M).astype(np.int64)
    rescued = np.bincount(ep_trace, weights=ep_rescued,
                          minlength=M).astype(np.int64)
    rate = np.divide(rescued, eligible, out=np.full(M, np.nan, dtype=float),
                     where=eligible > 0)
    total_e, total_r = int(eligible.sum()), int(rescued.sum())
    return {
        "eligible_by_trace": eligible,
        "rescued_by_trace": rescued,
        "rate_by_trace": rate,
        "eligible": total_e,
        "rescued": total_r,
        "global_rate": total_r / total_e if total_e else np.nan,
        "time_to_revival": ep_ttr[ep_ttr > 0],
        "episode_trace": ep_trace,
        "episode_window": ep_window,
        "episode_start": ep_start,
        "episode_end": ep_start + ep_len - 1,
        "episode_length": ep_len,
        "episode_eligibility_event": ep_start + inactivity - 1,
        "episode_rescued": ep_rescued,
        "episode_time_to_revival": ep_ttr,
    }


def binary_transition(
    selected: Array,
    delta: int = 1,
    creation_events: Optional[Array] = None,
) -> Dict[str, object]:
    """Lag-``delta`` importance transition matrix ``[[P00, P01], [P10, P11]]``.

    ``P01`` — an unimportant window becoming important ``delta`` events later
    (what a promotion path would have to catch); ``P10`` — the reverse (what a
    demotion path frees).  Rows are normalised per trace and pooled over
    traces; both are returned since the two weightings differ when traces
    contribute unequal numbers of pairs.

    Only pairs where the window is an observable candidate at *both* ends are
    counted.
    """
    x = np.asarray(selected, dtype=int)
    if x.ndim != 3:
        raise ValueError(f"selected must be [M, R, W]; got {x.shape}")
    if not np.all(np.isin(x, (-1, 0, 1))):
        raise ValueError("selected must contain only -1, 0, 1")
    M, R, W = x.shape
    if not 0 < int(delta) < R:
        raise ValueError(f"delta must satisfy 0 < delta < R = {R}; got {delta}")
    delta = int(delta)

    if creation_events is None:
        creation = infer_creation_events(x)
    else:
        creation = np.asarray(creation_events, dtype=int)
        if creation.ndim == 1:
            creation = np.broadcast_to(creation[None, :], (M, W))
        if creation.shape != (M, W):
            raise ValueError("creation_events must be [W] or [M, W]")

    counts = np.zeros((M, 2, 2), dtype=np.int64)
    born = creation[:, None, :] <= np.arange(R)[None, :, None]   # [M, R, W]
    cur, fut = x[:, :R - delta, :], x[:, delta:, :]
    ok = born[:, :R - delta, :] & (cur >= 0) & (fut >= 0)
    for a in (0, 1):
        for b in (0, 1):
            counts[:, a, b] = np.sum(ok & (cur == a) & (fut == b), axis=(1, 2))

    probs = np.full((M, 2, 2), np.nan)
    den = counts.sum(axis=2, keepdims=True)
    np.divide(counts, den, out=probs, where=den > 0)
    pooled_counts = counts.sum(axis=0)
    pooled = np.full((2, 2), np.nan)
    pden = pooled_counts.sum(axis=1, keepdims=True)
    np.divide(pooled_counts, pden, out=pooled, where=pden > 0)
    return {
        "delta": delta,
        "counts_by_trace": counts,
        "probabilities_by_trace": probs,
        "pooled_counts": pooled_counts,
        "pooled_probabilities": pooled,
    }
