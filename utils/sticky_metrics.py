"""Sticky-K policy-simulation analytics — Global LIR & absolute missed mass.

These metrics characterise how a **Sticky-K** history-eviction policy behaves
against the *ground-truth* attention masses recorded by the base (no-eviction)
parity run.  Unlike the Suite-A Jaccard metrics in :mod:`utils.metrics` — which
are pure, fully-vectorised set ops — these are inherently *sequential* policy
simulations (each flush's retain decision depends on the previous flush's
retained set), so they live in their own module and use explicit loops.  Do not
fold them back into ``utils/metrics.py``: ``test_faithfulness.py`` asserts that
module is loop-free.

Two quantities, both derived from one Sticky-K simulation over the truth masses:

  **Global LIR (Lazy Insertion Rescue).**
      Of all ``(window, flush)`` pairs where a window had been *ignored*
      (not retained) for the last ``m`` consecutive flushes, what fraction are
      later *rescued* (retained again at some future flush)?  High LIR ⇒ the
      policy thrashes — it repeatedly drops then re-admits the same windows;
      low LIR ⇒ once a window is dropped it stays dropped (stable selection).

  **Absolute missed mass.**
      Per flush, the raw ground-truth attention mass sitting on evictable
      history windows the policy did *not* retain.  Low ⇒ the retained set
      captures the attention that actually mattered.

The policy mirrors the production cache (``modules/windowed_cache/policy.py``):
the last ``local_windows`` history windows form an always-retained recency tail,
and the remaining (evictable) windows compete for ``history_budget_K`` slots via
a sticky set that swaps at most one window per flush (strongest-outsider for
weakest-insider, only on strict improvement).

Granularities returned: a single scalar, per-layer, and per-(layer, head).
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# Window geometry
# ---------------------------------------------------------------------------


def flush_geometry(
    num_steps: int,
    num_windows: int,
    prefill_len: int,
    num_sink: int,
    window_size: int,
    local_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-flush window counts and per-window creation flush.

    A *flush* is one recorded trace index ``t`` (0-indexed).  **Index 0 is the
    prefill forward**, not the first decode step: the parity runners append one
    entry per ``model(...)`` call and the first of those is the prompt.  So at
    index ``t`` the cache holds ``prefill_len + t`` tokens, the post-sink
    sequence holds ``Sp_t = prefill_len + t - num_sink``, and
    ``w_act = ceil(Sp_t / window_size)`` — capped at the padded array width
    ``num_windows``.  The last ``local_windows`` of those are the
    always-retained recency tail; the rest are evictable candidates.

    This used to carry a ``+ 1``, which modelled index 0 as the first decode
    step and so ran one flush ahead of the data.  Checked against a recorded
    base run (prefill 192, sink 4, ws 8) the true 24 -> 25 window transition is
    at ``t = 5``; the ``+1`` form put it at ``t = 4``.  Every window's
    ``creation`` was therefore one flush early, which shifts every revival
    episode's start, and one step in every ``window_size`` admitted a
    not-yet-existing window into the evictable band.
    ``faithfulness_runner._compute_metrics`` carried the same ``+1`` and is
    corrected in lockstep, so all of Suite B/E still agrees step for step.

    Returns
    -------
    w_act : np.ndarray[int]  ``[T]``  total valid windows at each flush.
    ew_act : np.ndarray[int] ``[T]``  evictable (candidate) count = ``w_act - lnw``.
    creation : np.ndarray[int] ``[W]``  first flush at which window ``k`` is valid
        (``num_steps`` if it never appears within the recorded steps).
    """
    t = np.arange(num_steps)
    Sp = np.maximum(1, prefill_len + t - num_sink)
    w_act = np.minimum(np.ceil(Sp / window_size).astype(int), num_windows)
    lnw = np.minimum(local_windows, w_act)
    ew_act = np.maximum(w_act - lnw, 0)

    # creation[k] = first flush whose w_act exceeds k.  w_act is non-decreasing,
    # so this is a single searchsorted: the leftmost t with w_act[t] >= k + 1.
    creation = np.searchsorted(w_act, np.arange(num_windows) + 1, side="left")
    creation = np.minimum(creation, num_steps).astype(int)
    return w_act, ew_act, creation


# ---------------------------------------------------------------------------
# Policy simulation
# ---------------------------------------------------------------------------


def simulate_policy(
    truth_masses: np.ndarray,
    history_budget_K: int,
    w_act: np.ndarray,
    ew_act: np.ndarray,
    is_sticky: bool = True,
    n_q: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate two-tier Sticky-K (or Fresh-K) retention over history windows.

    The **fp tier** keeps ``history_budget_K`` evictable windows (the fp16
    survivors).  On top of it a **Q tier** keeps the next ``n_q`` strongest
    evictable windows *not already in fp* — the int2 survivors that a two-tier
    cache holds instead of dropping (design.md §5, mirroring
    ``policy.compute_two_tier_retain``: rank the band, top ``k_fp`` → fp, next
    ``n_q`` → Q).  ``n_q == 0`` reduces to the single-tier drop policy exactly.

    Parameters
    ----------
    truth_masses : np.ndarray ``[T, W]``
        Non-negative ground-truth window masses (e.g. base-run window scores,
        already reduced to a single layer or head).
    history_budget_K : int
        Number of evictable windows the **fp tier** may retain.  ``<= 0`` means
        no fp window is kept (only the recency tail + Q tier).
    w_act, ew_act : np.ndarray[int] ``[T]``
        Total / evictable window counts per flush, from :func:`flush_geometry`.
    is_sticky : bool
        ``True`` = Sticky-K fp tier (persistent set, single best-swap per flush);
        ``False`` = Fresh-K (re-pick top-K every flush, no memory).  The Q tier
        is always the fresh next-``n_q`` residual, matching the production
        eviction which re-ranks the whole band every flush.
    n_q : int
        Number of extra evictable windows held in the int2 Q tier.

    Returns
    -------
    selection : np.ndarray[bool] ``[T, W]``
        ``True`` where window ``w`` is retained in the **fp tier** at flush ``t``
        (plus the always-kept local tail).  Q-tier windows are *not* marked here
        so the LIR thrashing analysis stays a pure fp-tier quantity.
    missed_fp : np.ndarray[float] ``[T]``
        Fraction of total valid-window mass on evictable windows *outside the fp
        tier* — the single-tier "drop everything below fp" baseline (identical to
        the legacy ``missed`` at ``n_q == 0``).
    missed_kept : np.ndarray[float] ``[T]``
        Fraction of total valid-window mass on evictable windows in **neither**
        tier (truly dropped).  ``missed_fp - missed_kept`` is the mass the Q tier
        rescues.  Both series are normalised by the sum of scores over all valid
        windows (evictable + local tail), so they are in ``[0, 1]`` and
        independent of the raw H2O score magnitude.
    """
    T, W = truth_masses.shape
    selection = np.zeros((T, W), dtype=bool)
    missed_fp = np.zeros(T, dtype=float)
    missed_kept = np.zeros(T, dtype=float)
    sticky: set[int] = set()

    for t in range(T):
        scores = truth_masses[t]
        ew = int(ew_act[t])   # evictable candidate count
        wv = int(w_act[t])    # total valid windows

        # Local recency tail [ew, wv) is always retained (never evicted).
        if wv > ew:
            selection[t, ew:wv] = True

        if ew == 0:
            continue

        candidates = range(ew)

        if history_budget_K <= 0:
            selected: set[int] = set()
        elif is_sticky:
            # Drop windows that aged out of the evictable region (ew only grows,
            # so this is rarely a no-op only at the very start).
            sticky = {w for w in sticky if w < ew}
            if len(sticky) < history_budget_K:
                # FILLING PHASE — top up empty slots with the strongest outsiders.
                slots = history_budget_K - len(sticky)
                outsiders = sorted(
                    (w for w in candidates if w not in sticky),
                    key=lambda w: scores[w],
                    reverse=True,
                )
                sticky.update(outsiders[:slots])
            else:
                # SWAPPING PHASE — at most one swap, only on strict improvement.
                weakest_in = min(sticky, key=lambda w: scores[w])
                outsiders = [w for w in candidates if w not in sticky]
                if outsiders:
                    strongest_out = max(outsiders, key=lambda w: scores[w])
                    if scores[strongest_out] > scores[weakest_in]:
                        sticky.discard(weakest_in)
                        sticky.add(strongest_out)
            selected = sticky
        else:
            # FRESH-K — pick the top-K candidates from scratch every flush.
            ordered = sorted(candidates, key=lambda w: scores[w], reverse=True)
            selected = set(ordered[:history_budget_K])

        for w in selected:
            selection[t, w] = True

        # ── Q tier: next-n_q strongest evictable windows not already in fp ──
        q_set: set[int] = set()
        if n_q > 0:
            residual = sorted(
                (w for w in candidates if w not in selected),
                key=lambda w: scores[w],
                reverse=True,
            )
            q_set = set(residual[:n_q])
        kept = selected | q_set

        # Missed mass = fraction of total valid-window mass on evictable windows
        # we did not keep.  Normalise by the sum over ALL valid windows (not
        # just evictable) so the value is in [0, 1] and independent of how
        # large the raw H2O cumulative scores happen to be at this flush.
        total_mass = float(scores[:wv].sum())
        if total_mass > 1e-12:
            missed_fp[t] = float(
                sum(scores[w] for w in candidates if w not in selected)
            ) / total_mass
            missed_kept[t] = float(
                sum(scores[w] for w in candidates if w not in kept)
            ) / total_mass
        # else: all scores are zero — leave both at 0.0

    return selection, missed_fp, missed_kept


# ---------------------------------------------------------------------------
# Global LIR from a selection matrix
# ---------------------------------------------------------------------------


def lir_counts(
    selection: np.ndarray,
    creation: np.ndarray,
    w_act: np.ndarray,
    m: int = 3,
) -> tuple[int, int]:
    """Count *eligible* (ignored-for-``m``-flushes) and *rescued* window pairs.

    A ``(flush r, window k)`` pair is **eligible** when

      * the lookback ``[r-m+1, r]`` fits in range (``r >= m-1``),
      * window ``k`` already existed at the lookback start
        (``creation[k] <= r-m+1``), and
      * window ``k`` was **not** retained in any flush of that lookback window.

    It is **rescued** if window ``k`` is retained at some later flush ``> r``.

    Counts are returned raw (not divided) so callers can aggregate to any
    granularity by summing eligible/rescued before taking the final ratio —
    the statistically correct ratio-of-sums.

    Returns
    -------
    (eligible, rescued) : tuple[int, int]
    """
    T, W = selection.shape
    eligible = 0
    rescued = 0
    for r in range(T):
        start = r - m + 1
        if start < 0:
            continue
        for k in range(W):
            # Window must have existed since the lookback start; creation[k] <=
            # start also guarantees k < w_act[r] (w_act is non-decreasing), so
            # padded / not-yet-created windows are skipped automatically.
            if creation[k] > start:
                continue
            if selection[start : r + 1, k].any():
                continue
            eligible += 1
            if selection[r + 1 :, k].any():
                rescued += 1
    return eligible, rescued


def _ratio(rescued: float, eligible: float) -> float:
    return float(rescued) / float(eligible) if eligible > 0 else 0.0


# ---------------------------------------------------------------------------
# Driver — full analytics over base (ground-truth) window scores
# ---------------------------------------------------------------------------


def compute_sticky_metrics(
    base_ws: np.ndarray,
    *,
    prefill_len: int,
    num_sink: int,
    window_size: int,
    local_windows: int,
    history_budget_K: int,
    m: int = 3,
    n_q: int = 0,
) -> Dict[str, np.ndarray]:
    """Full two-tier Sticky-K analytics over base-run window scores.

    The layer-level and global figures simulate the policy on **head-mean**
    masses — exactly what the production cache ranks when it evicts
    (``policy.compute_retain_window_indices`` averages over heads first).  The
    per-head figures simulate each head independently, exposing how much an
    individual head *would* thrash if it owned the budget.

    Parameters
    ----------
    base_ws : np.ndarray ``[S, T, L, H, W]``
        Base-run per-head window scores (the ground-truth attention masses).
    prefill_len, num_sink, window_size : int
        Token geometry shared by both parity runs.
    local_windows : int
        Number of always-retained recency windows (``local_tokens // window_size``).
    history_budget_K : int
        **fp-tier** evictable window budget.  For a single-tier run this is
        ``top_k_windows``; for a two-tier run pass ``top_k_fp``.
    m : int
        "Ignored" duration threshold for LIR.
    n_q : int
        int2 Q-tier width (``N_q``).  ``0`` (default) reduces every series below
        to the legacy single-tier numbers.

    Returns
    -------
    dict with (all missed/recovered series normalised to ``[0, 1]``):
      ``global_lir``               float — fp-tier rescue rate (all layers/heads).
      ``lir_per_layer`` / ``lir_per_head`` — fp-tier rescue rate per granularity.
      ``missed_mass`` (== ``missed_mass_fp``) ``[T]`` — fp-only-drop missed mass
                                   (mass below the fp tier; the single-tier baseline).
      ``missed_mass_per_layer`` ``[T, L]`` — fp-only missed mass per layer.
      ``missed_mass_kept`` ``[T]`` / ``missed_mass_kept_per_layer`` ``[T, L]`` —
                                   mass dropped by **neither** tier (the honest
                                   two-tier missed mass; ≤ ``missed_mass``).
      ``recovered_mass_q`` ``[T]`` / ``recovered_mass_q_per_layer`` ``[T, L]`` —
                                   ``missed_mass − missed_mass_kept``: mass the Q
                                   tier rescues (the visible benefit).
      ``missed_mass_fresh`` / ``missed_mass_fresh_kept`` / ``recovered_mass_q_fresh``
                                   ``[T]`` — Fresh-K counterparts (mirrors the
                                   production two-tier eviction's per-flush re-rank).
      ``missed_mass_total`` / ``missed_mass_kept_total`` / ``recovered_mass_q_total``
                                   float scalars — mean over flushes.
    """
    base_ws = np.asarray(base_ws, dtype=np.float64)
    if base_ws.ndim != 5:
        raise ValueError(
            f"base_ws must be [S, T, L, H, W]; got shape {base_ws.shape}"
        )
    S, T, L, H, W = base_ws.shape

    w_act, ew_act, creation = flush_geometry(
        T, W, prefill_len, num_sink, window_size, local_windows
    )

    # Accumulators (ratio-of-sums for LIR, mean for missed mass).
    glob_elig = glob_resc = 0
    lir_layer_elig = np.zeros(L)
    lir_layer_resc = np.zeros(L)
    lir_head_elig = np.zeros((L, H))
    lir_head_resc = np.zeros((L, H))
    mm_layer = np.zeros((T, L))        # fp-only missed, summed over samples
    mm_layer_kept = np.zeros((T, L))   # two-tier (kept) missed, summed over samples
    mm_fresh = np.zeros(T)             # fp-only Fresh-K, summed over samples & layers
    mm_fresh_kept = np.zeros(T)        # two-tier Fresh-K, summed over samples & layers

    for s in range(S):
        for li in range(L):
            # ── layer / global: head-mean masses (matches the real cache) ──
            masses_hm = base_ws[s, :, li, :, :].mean(axis=1)   # [T, W]
            sel, miss_fp, miss_kept = simulate_policy(
                masses_hm, history_budget_K, w_act, ew_act,
                is_sticky=True, n_q=n_q,
            )
            e, r = lir_counts(sel, creation, w_act, m)
            lir_layer_elig[li] += e
            lir_layer_resc[li] += r
            glob_elig += e
            glob_resc += r
            mm_layer[:, li] += miss_fp
            mm_layer_kept[:, li] += miss_kept

            _, miss_fresh_fp, miss_fresh_kept = simulate_policy(
                masses_hm, history_budget_K, w_act, ew_act,
                is_sticky=False, n_q=n_q,
            )
            mm_fresh += miss_fresh_fp
            mm_fresh_kept += miss_fresh_kept

            # ── per-head: each head simulated independently (LIR only) ──
            for h in range(H):
                sel_h, _, _ = simulate_policy(
                    base_ws[s, :, li, h, :], history_budget_K,
                    w_act, ew_act, is_sticky=True,
                )
                eh, rh = lir_counts(sel_h, creation, w_act, m)
                lir_head_elig[li, h] += eh
                lir_head_resc[li, h] += rh

    lir_per_layer = np.divide(
        lir_layer_resc, lir_layer_elig,
        out=np.zeros(L), where=lir_layer_elig > 0,
    )
    lir_per_head = np.divide(
        lir_head_resc, lir_head_elig,
        out=np.zeros((L, H)), where=lir_head_elig > 0,
    )
    missed_mass_per_layer = mm_layer / max(S, 1)
    missed_mass = missed_mass_per_layer.mean(axis=1)
    missed_mass_kept_per_layer = mm_layer_kept / max(S, 1)
    missed_mass_kept = missed_mass_kept_per_layer.mean(axis=1)
    recovered_mass_q_per_layer = missed_mass_per_layer - missed_mass_kept_per_layer
    recovered_mass_q = missed_mass - missed_mass_kept
    missed_mass_fresh = mm_fresh / max(S * L, 1)
    missed_mass_fresh_kept = mm_fresh_kept / max(S * L, 1)
    recovered_mass_q_fresh = missed_mass_fresh - missed_mass_fresh_kept

    return {
        "global_lir": np.array(_ratio(glob_resc, glob_elig), dtype=np.float64),
        "lir_per_layer": lir_per_layer,
        "lir_per_head": lir_per_head,
        "missed_mass": missed_mass,
        "missed_mass_fp": missed_mass,                       # explicit alias
        "missed_mass_per_layer": missed_mass_per_layer,
        "missed_mass_kept": missed_mass_kept,
        "missed_mass_kept_per_layer": missed_mass_kept_per_layer,
        "recovered_mass_q": recovered_mass_q,
        "recovered_mass_q_per_layer": recovered_mass_q_per_layer,
        "missed_mass_fresh": missed_mass_fresh,
        "missed_mass_fresh_kept": missed_mass_fresh_kept,
        "recovered_mass_q_fresh": recovered_mass_q_fresh,
        "missed_mass_total": np.array(float(missed_mass.mean()), dtype=np.float64),
        "missed_mass_kept_total": np.array(
            float(missed_mass_kept.mean()), dtype=np.float64),
        "recovered_mass_q_total": np.array(
            float(recovered_mass_q.mean()), dtype=np.float64),
    }
