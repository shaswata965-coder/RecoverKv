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


# ---------------------------------------------------------------------------
# Tier dynamics — the F / Q / E chain (promotion, demotion, eviction)
# ---------------------------------------------------------------------------
#
# Observation III's ``binary_transition`` over the policy's fp tier asks "was
# the window in fp?", so its ``0`` pools two different states: held in int2
# (Q, promotable) and evicted (E, gone for good). A window evicted once
# contributes a ``0 -> 0`` pair at every later event, so ``P01`` read as "the
# promotion rate" is diluted by windows that can never be promoted, and ``P10``
# mixes demotion with outright eviction. In a three-tier cache those are the
# design's three separate moves, so they are counted separately here.

STATE_F, STATE_Q, STATE_E = 0, 1, 2
STATE_NAMES: Tuple[str, str, str] = ("F", "Q", "E")

#: Named lag-``delta`` moves, ``(from, to)``.
MOVES: Dict[str, Tuple[int, int]] = {
    "stay_fp": (STATE_F, STATE_F),
    "promote": (STATE_Q, STATE_F),
    "demote": (STATE_F, STATE_Q),
    "stay_q": (STATE_Q, STATE_Q),
    "evict_from_fp": (STATE_F, STATE_E),
    "evict_from_q": (STATE_Q, STATE_E),
}


def tier_states(acc_fp: Array, acc_q: Array, band: Array) -> Array:
    """``[M, R, W]`` int8 tier state over the evictable band.

    ``-1`` outside the band at that event (not yet created, still in the
    protected local tail, or padding); ``0`` F (full precision); ``1`` Q (int2);
    ``2`` E — in the band but in neither tier, i.e. evicted at this event or an
    earlier one. ``band`` is ``[R, W]`` or ``[M, R, W]``.
    """
    fp = np.asarray(acc_fp, dtype=bool)
    q = np.asarray(acc_q, dtype=bool)
    b = np.asarray(band, dtype=bool)
    if b.ndim == 2:
        b = np.broadcast_to(b[None], fp.shape)
    if fp.shape != q.shape or fp.shape != b.shape:
        raise ValueError(f"shape mismatch: fp {fp.shape}, q {q.shape}, band {b.shape}")
    if np.any(fp & q & b):
        raise ValueError("a window is tagged both fp and Q at one event")
    out = np.full(fp.shape, -1, dtype=np.int8)
    out[b] = STATE_E
    out[b & fp] = STATE_F
    out[b & q] = STATE_Q
    return out


def tier_transitions(states: Array, delta: int = 1) -> Dict[str, object]:
    """Lag-``delta`` F/Q/E transition counts and rates.

    Only pairs where the window is in the band at *both* ends are counted, so
    a window leaving the local tail enters through ``entry_counts`` (the state
    it is first given) rather than as a transition. ``E`` is absorbing in the
    cache; ``resurrections`` counts ``E -> F|Q`` pairs, which must be zero —
    anything else means the survivor ids were mapped onto the wrong windows.

    Returns counts ``[M, 3, 3]`` (``from, to``), per-trace row probabilities,
    pooled counts/probabilities, ``entry_counts`` ``[M, 3]`` and one
    ``rate_<move>`` ``[M]`` per :data:`MOVES` entry (``P(to | from)``).
    """
    x = np.asarray(states, dtype=int)
    if x.ndim != 3:
        raise ValueError(f"states must be [M, R, W]; got {x.shape}")
    M, R, _ = x.shape
    delta = int(delta)
    if not 0 < delta < R:
        raise ValueError(f"delta must satisfy 0 < delta < R = {R}; got {delta}")
    cur, fut = x[:, :R - delta], x[:, delta:]
    ok = (cur >= 0) & (fut >= 0)
    counts = np.zeros((M, 3, 3), dtype=np.int64)
    for a in range(3):
        for b in range(3):
            counts[:, a, b] = np.sum(ok & (cur == a) & (fut == b), axis=(1, 2))
    entry = np.zeros((M, 3), dtype=np.int64)
    for b in range(3):
        entry[:, b] = np.sum((cur < 0) & (fut == b), axis=(1, 2))
    probs = np.full((M, 3, 3), np.nan)
    den = counts.sum(axis=2, keepdims=True)
    np.divide(counts, den, out=probs, where=den > 0)
    pooled_counts = counts.sum(axis=0)
    pooled = np.full((3, 3), np.nan)
    pden = pooled_counts.sum(axis=1, keepdims=True)
    np.divide(pooled_counts, pden, out=pooled, where=pden > 0)
    out: Dict[str, object] = {
        "delta": delta,
        "counts_by_trace": counts,
        "probabilities_by_trace": probs,
        "pooled_counts": pooled_counts,
        "pooled_probabilities": pooled,
        "entry_counts": entry,
        "resurrections": int(counts[:, STATE_E, STATE_F].sum()
                             + counts[:, STATE_E, STATE_Q].sum()),
    }
    for name, (a, b) in MOVES.items():
        out[f"rate_{name}"] = probs[:, a, b]
    return out


def _future_mass(step_mass: Array, t: int, horizon: int) -> Optional[Array]:
    """``[M, W]`` mass over steps ``t+1 .. t+H``; ``None`` if not fully observed."""
    T = step_mass.shape[1]
    if t + horizon + 1 > T:
        return None
    return step_mass[:, t + 1:t + horizon + 1, :].sum(axis=1)


def _top_k_mask(scores: Array, cand: Array, k: Array) -> Array:
    """``[M, W]`` bool — each row's top ``k[m]`` candidates by ``scores``."""
    s = np.where(cand, scores, -np.inf)
    order = np.argsort(-s, axis=-1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(s.shape[-1])[None, :], axis=-1)
    return cand & (rank < np.asarray(k)[:, None])


def transition_outcomes(
    states: Array,
    step_mass: Array,
    event_steps: Sequence[int],
    horizon: int,
) -> Dict[str, object]:
    """Did each tier move land on the windows the future actually attends?

    For every routing event ``r >= 1`` (decode step ``t``), the lag-1 move of
    each window from event ``r-1`` to ``r`` is scored against its ground-truth
    mass over the next ``H`` steps, ``F_w = Σ_{t'=t+1}^{t+H} mass[t', w]``.
    The **candidates** are the windows that event actually decided: in the
    band at ``r`` and not already evicted at ``r-1``.

    Per trace, summed over events (ratio-of-sums, so an event with many moves
    weighs more than one with few):

    * ``share_<move>`` — fraction of the candidates' future mass sitting on the
      windows that made that move. ``share_promote`` is the mass promotions
      brought back to full precision; ``share_evict_from_fp`` +
      ``share_evict_from_q`` is the eviction regret.
    * ``lift_<move>`` — a moved window's mean future mass over the candidate
      mean. ``> 1`` = the move picked windows the future attends more than
      average. ``lift_promote - lift_demote`` is the swap gain: positive when
      what came up to fp outweighs what went down.
    * ``hit_promote`` — fraction of promotions into the hindsight-best fp set
      (the top ``|F|`` candidates by future mass, ``|F|`` the policy's own fp
      count, so the comparison is size-matched); ``hit_demote`` — fraction of
      demotions out of it; ``hit_evict`` — fraction of evictions outside the
      hindsight-best retained set (top ``|F| + |Q|``); ``fp_precision`` —
      ``|F ∩ best F| / |F|`` over all candidates.

    Events without a full ``H``-step future are skipped (right-censored), as in
    :func:`future_missed_mass`.
    """
    x = np.asarray(states, dtype=int)
    mass = np.asarray(step_mass, dtype=float)
    steps = np.asarray(event_steps, dtype=int)
    if x.ndim != 3 or mass.ndim != 3:
        raise ValueError("states and step_mass must both be 3-D")
    M, R, W = x.shape
    if mass.shape[0] != M or mass.shape[2] != W:
        raise ValueError(f"step_mass must be [{M}, T, {W}]; got {mass.shape}")
    if steps.size != R:
        raise ValueError("event_steps length must equal states.shape[1]")

    names = list(MOVES)
    s_mass = {n: np.zeros(M) for n in names}
    s_cnt = {n: np.zeros(M) for n in names}
    cand_mass = np.zeros(M)
    cand_cnt = np.zeros(M)
    hits = {k: np.zeros(M) for k in ("promote", "demote", "evict", "fp")}
    tots = {k: np.zeros(M) for k in ("promote", "demote", "evict", "fp")}
    scored = 0
    for r in range(1, R):
        fut = _future_mass(mass, int(steps[r]), int(horizon))
        if fut is None:
            continue
        scored += 1
        prev, now = x[:, r - 1], x[:, r]
        cand = (now >= 0) & (prev != STATE_E)
        cand_mass += np.sum(np.where(cand, fut, 0.0), axis=-1)
        cand_cnt += cand.sum(axis=-1)
        for n in names:
            a, b = MOVES[n]
            sel = cand & (prev == a) & (now == b)
            s_mass[n] += np.sum(np.where(sel, fut, 0.0), axis=-1)
            s_cnt[n] += sel.sum(axis=-1)
        k_fp = np.sum(cand & (now == STATE_F), axis=-1)
        k_alive = k_fp + np.sum(cand & (now == STATE_Q), axis=-1)
        best_fp = _top_k_mask(fut, cand, k_fp)
        best_alive = _top_k_mask(fut, cand, k_alive)
        prom = cand & (prev == STATE_Q) & (now == STATE_F)
        dem = cand & (prev == STATE_F) & (now == STATE_Q)
        ev = cand & (prev != STATE_E) & (now == STATE_E)
        fpn = cand & (now == STATE_F)
        for key, sel, good in (("promote", prom, best_fp),
                               ("demote", dem, ~best_fp),
                               ("evict", ev, ~best_alive),
                               ("fp", fpn, best_fp)):
            hits[key] += np.sum(sel & good, axis=-1)
            tots[key] += sel.sum(axis=-1)

    def ratio(a: Array, b: Array) -> Array:
        return np.divide(a, b, out=np.full(M, np.nan), where=b > 0)

    cand_mean = ratio(cand_mass, cand_cnt)
    out: Dict[str, object] = {"events_scored": scored, "horizon": int(horizon),
                              "candidate_count": cand_cnt}
    for n in names:
        out[f"count_{n}"] = s_cnt[n]
        out[f"share_{n}"] = ratio(s_mass[n], cand_mass)
        out[f"lift_{n}"] = ratio(ratio(s_mass[n], s_cnt[n]), cand_mean)
    out["swap_gain"] = out["lift_promote"] - out["lift_demote"]
    out["eviction_regret"] = out["share_evict_from_fp"] + out["share_evict_from_q"]
    for key in ("promote", "demote", "evict"):
        out[f"hit_{key}"] = ratio(hits[key], tots[key])
        out[f"hit_{key}_num"] = hits[key]
        out[f"hit_{key}_den"] = tots[key]
    out["fp_precision"] = ratio(hits["fp"], tots["fp"])
    out["fp_precision_num"] = hits["fp"]
    out["fp_precision_den"] = tots["fp"]
    return out


def tier_decision_fidelity(states: Array, rank_scores: Array) -> Array:
    """``[M, R]`` Jaccard of the policy's fp set vs the oracle's, same survivors.

    Among the windows the cache still holds at an event (F or Q), the oracle
    puts in fp the ``|F|`` with the highest ground-truth cumulative score.
    This isolates the *ranking* the eviction used — its own running scores,
    which under the read gate credit a skipped int2 window with its card
    estimate — from what the cache happened to retain. NaN where the event
    holds no fp window.
    """
    x = np.asarray(states, dtype=int)
    rank = np.asarray(rank_scores, dtype=float)
    if x.shape != rank.shape:
        raise ValueError(f"states {x.shape} and rank_scores {rank.shape} differ")
    M, R, _ = x.shape
    out = np.full((M, R), np.nan)
    for r in range(R):
        alive = (x[:, r] == STATE_F) | (x[:, r] == STATE_Q)
        pol = x[:, r] == STATE_F
        k = pol.sum(axis=-1)
        best = _top_k_mask(rank[:, r], alive, k)
        inter = np.sum(pol & best, axis=-1)
        union = np.sum(pol | best, axis=-1)
        np.divide(inter, union, out=out[:, r], where=union > 0)
    return out


# ---------------------------------------------------------------------------
# Observation IV — the decode read ledger (per query head, gate-aware)
# ---------------------------------------------------------------------------
#
# FMM scores what the cache RETAINS. On the flash path that is not what a decode
# step READS: the read gate opens a fraction of the int2 tier per KV head, and a
# skipped int2 window reaches the output only through its value centroid, at
# the card's calibrated weight. So at every decode step, each query head's
# ground-truth attention is split into where it actually landed.

#: Window classes on the baseline window axis, per decode step.
CLS_FP, CLS_Q, CLS_LOCAL, CLS_EVICTED, CLS_FRESH = 0, 1, 2, 3, 4

#: Ledger columns. ``fp`` and ``local`` are read exactly; ``q_read`` at int2;
#: ``q_skipped`` only through the centroid fill; ``evicted`` not at all.
#: ``fresh`` is the newest tokens, appended to the fp store since the last score
#: column was created (read exactly). ``sink`` is the rest of the head's mass.
LEDGER_PARTS: Tuple[str, ...] = (
    "fp", "local", "fresh", "q_read", "q_skipped", "evicted", "sink")


def survivor_to_base(ids: Array, tier: Array, W: int,
                     gate_read: Optional[Array] = None
                     ) -> Tuple[Array, Optional[Array], int]:
    """Scatter one trace's survivor axis onto the baseline window axis.

    Parameters
    ----------
    ids, tier : ``[T, Wo]`` — ``all_window_ids`` / ``all_window_tier`` for one
        ``(sample, layer)``; ``-1`` pads.
    W : baseline window count.
    gate_read : ``[T, H_kv, Wo]`` bool, optional — the gate's pick.

    Returns ``(cls [T, W] int8, read [T, H_kv, W] bool or None, n_out_of_range)``.
    A window absent from the survivors is ``CLS_EVICTED`` when some newer window
    survives (ids are chronological and eviction is the only way off the axis),
    and ``CLS_FRESH`` when it is newer than every survivor: its tokens are
    already in the fp store and its score column arrives with the next update.
    """
    ids = np.asarray(ids)
    tier = np.asarray(tier)
    T = ids.shape[0]
    valid = (ids >= 0) & (tier >= 0)
    oob = int(np.sum(valid & (ids >= W)))
    valid &= ids < W
    max_id = np.where(valid, ids, -1).max(axis=1) if ids.size else np.full(T, -1)
    cols = np.arange(W)[None, :]
    cls = np.where(cols <= max_id[:, None], CLS_EVICTED, CLS_FRESH).astype(np.int8)
    t_idx, o_idx = np.nonzero(valid)
    w_idx = ids[t_idx, o_idx]
    tmap = np.array([CLS_FP, CLS_Q, CLS_LOCAL], dtype=np.int8)
    cls[t_idx, w_idx] = tmap[np.clip(tier[t_idx, o_idx], 0, 2)]
    read = None
    if gate_read is not None:
        g = np.asarray(gate_read, dtype=bool)
        read = np.zeros((T, g.shape[1], W), dtype=bool)
        if t_idx.size:
            read[t_idx, :, w_idx] = g[t_idx, :, o_idx]
    return cls, read, oob


def decode_read_ledger(
    head_mass: Array,
    cls: Array,
    read: Optional[Array],
    fired: Optional[Array],
    rep: int,
    first_step: int = 1,
) -> Dict[str, Array]:
    """Split each query head's per-step attention by where the decode read it.

    Parameters
    ----------
    head_mass : ``[T, H, W]`` ground-truth per-step, per-query-head window mass
        (post-sink; one query row per decode step, so each head's total over
        sink + windows is 1).
    cls : ``[T, W]`` window class (:func:`survivor_to_base`).
    read : ``[T, H_kv, W]`` bool — the gate's pick; ``None`` = never gated.
    fired : ``[T]`` bool — the gate ran at that step. Where it did not, the
        whole Q tier counts as read (the ungated materialize / eager read).
    rep : query heads per KV head (GQA); query head ``h`` reads KV head
        ``h // rep``, the layout ``repeat_kv`` and the gate both use.
    first_step : steps before it are skipped (step 0 is the prefill forward,
        which has many query rows and no gate).

    Returns
    -------
    ``parts`` ``[T', H, P]`` — fraction of each head's mass per
    :data:`LEDGER_PARTS` column (rows sum to 1 up to the sink clamp);
    ``gated`` ``[T']`` bool; ``q_mass`` ``[T', H]``; ``read_frac`` ``[T', H_kv]``
    (opened / active int2 windows, NaN where ungated or no Q tier);
    ``oracle_q_read`` ``[T', H]`` — the int2 mass a hindsight pick of the SAME
    number of windows per KV head would have opened, choosing at each step the
    windows with the largest summed per-head share over the group (each head's
    Q mass normalised to 1 first, never raw mass: CLAUDE.md, the GQA union).
    It is optimal for that per-step objective, which is the gate's own; recall
    aggregated as a ratio of sums over steps can still land a hair above it.
    """
    m = np.asarray(head_mass, dtype=np.float32)[first_step:]
    c = np.asarray(cls)[first_step:]
    Tn, H, W = m.shape
    if c.shape != (Tn, W):
        raise ValueError(f"cls must be [{Tn}, {W}] after first_step; got {c.shape}")
    rep = max(1, int(rep))
    kv_of = np.arange(H) // rep
    H_kv = int(kv_of.max()) + 1 if H else 0
    if fired is None:
        gated = np.zeros(Tn, dtype=bool)
    else:
        gated = np.asarray(fired, dtype=bool)[first_step:]
    if read is None:
        rd = np.zeros((Tn, H_kv, W), dtype=bool)
        gated = np.zeros(Tn, dtype=bool)
    else:
        rd = np.asarray(read, dtype=bool)[first_step:]
        if rd.shape[1] != H_kv:
            raise ValueError(
                f"gate_read has {rd.shape[1]} KV heads but H={H} query heads at "
                f"rep={rep} imply {H_kv}")

    is_q = c == CLS_Q                                          # [Tn, W]
    read_h = rd[:, kv_of, :]                                   # [Tn, H, W]
    g = gated[:, None, None]
    q_read = is_q[:, None, :] & (read_h | ~g)
    q_skip = is_q[:, None, :] & ~read_h & g

    parts = np.zeros((Tn, H, len(LEDGER_PARTS)), dtype=np.float64)
    for i, (name, klass) in enumerate((("fp", CLS_FP), ("local", CLS_LOCAL),
                                       ("fresh", CLS_FRESH),
                                       ("evicted", CLS_EVICTED))):
        parts[..., LEDGER_PARTS.index(name)] = np.einsum(
            "thw,tw->th", m, (c == klass).astype(np.float32))
    parts[..., LEDGER_PARTS.index("q_read")] = np.einsum(
        "thw,thw->th", m, q_read.astype(np.float32))
    parts[..., LEDGER_PARTS.index("q_skipped")] = np.einsum(
        "thw,thw->th", m, q_skip.astype(np.float32))
    win_total = parts[..., :LEDGER_PARTS.index("sink")].sum(-1)
    parts[..., LEDGER_PARTS.index("sink")] = np.clip(1.0 - win_total, 0.0, None)
    q_mass = (parts[..., LEDGER_PARTS.index("q_read")]
              + parts[..., LEDGER_PARTS.index("q_skipped")])

    n_q = is_q.sum(-1).astype(float)                           # [Tn]
    opened = np.sum(rd & is_q[:, None, :], axis=-1).astype(float)   # [Tn, H_kv]
    read_frac = np.full((Tn, H_kv), np.nan)
    ok = gated & (n_q > 0)
    np.divide(opened, n_q[:, None], out=read_frac, where=ok[:, None])

    # Hindsight ceiling: same count per KV head, best by summed per-head share.
    oracle = np.full((Tn, H), np.nan)
    if np.any(ok):
        share = m / np.maximum(q_mass, 1e-30)[..., None]       # [Tn, H, W]
        share = np.where(is_q[:, None, :], share, 0.0)
        grp = np.zeros((Tn, H_kv, W), dtype=np.float64)
        np.add.at(grp, (slice(None), kv_of, slice(None)), share)
        grp = np.where(is_q[:, None, :], grp, -np.inf)
        order = np.argsort(-grp, axis=-1, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(W)[None, None, :], axis=-1)
        pick = rank < opened[..., None]                        # [Tn, H_kv, W]
        o_read = np.einsum("thw,thw->th", m,
                           (pick[:, kv_of, :] & is_q[:, None, :]).astype(np.float32))
        oracle[ok] = o_read[ok]
    return {"parts": parts, "gated": gated, "q_mass": q_mass,
            "read_frac": read_frac, "oracle_q_read": oracle}


def ledger_head_recall(parts: Array, gated: Array, oracle: Optional[Array] = None,
                       min_q_share: float = 0.0) -> Dict[str, Array]:
    """Per-head gate recall over one trace's gated steps (ratio of sums).

    ``recall[h] = Σ_t q_read / Σ_t (q_read + q_skipped)`` over steps where the
    gate ran and head ``h`` put at least ``min_q_share`` of its mass on the int2
    tier. ``oracle_recall`` is the same with the hindsight pick's mass. Per head,
    never pooled across heads first: a group sum is dominated by its loudest
    head, which is how the old recall test read 1.000 while one head got 0.05%
    of its mass read (CLAUDE.md, §12).
    """
    p = np.asarray(parts, dtype=float)
    g = np.asarray(gated, dtype=bool)
    qr = p[..., LEDGER_PARTS.index("q_read")]
    qs = p[..., LEDGER_PARTS.index("q_skipped")]
    qm = qr + qs
    use = g[:, None] & (qm > max(float(min_q_share), 0.0)) & (qm > 0)
    num = np.sum(np.where(use, qr, 0.0), axis=0)
    den = np.sum(np.where(use, qm, 0.0), axis=0)
    H = p.shape[1]
    recall = np.divide(num, den, out=np.full(H, np.nan), where=den > 0)
    out = {"recall": recall, "q_mass_total": den,
           "step_recall": np.divide(qr, qm, out=np.full(qr.shape, np.nan),
                                    where=use)}
    if oracle is not None:
        o = np.asarray(oracle, dtype=float)
        onum = np.sum(np.where(use & np.isfinite(o), o, 0.0), axis=0)
        out["oracle_recall"] = np.divide(onum, den, out=np.full(H, np.nan),
                                         where=den > 0)
    return out
