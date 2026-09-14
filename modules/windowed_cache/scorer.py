"""Pure scoring functions for windowed KV cache.

Two functions:
- ``compute_window_scores`` — reduces ``[B, H_q, T, S]`` attention to
  ``[B, H_q, W]`` per-window scores.
- ``accumulate`` — in-place ``+=`` wrapper for unit testability.
"""

from __future__ import annotations

import torch
from torch import Tensor

from einops import reduce


def compute_window_scores(
    attn: Tensor,
    num_sink: int,
    window_size: int,
) -> Tensor:
    """Reduce full attention weights to per-window scores.

    Algorithm:
    1. Sum over T query rows → per-token received attention ``[B, H_q, S]``.
    2. Strip sink prefix (never scored).
    3. Right-pad trailing partial window with zeros.
    4. ``einops.reduce('b h (w s) -> b h w', 'sum')``.

    Parameters
    ----------
    attn : Tensor
        Shape ``[B, H_q, T, S]``, post-softmax attention weights.
    num_sink : int
        Number of sink tokens to strip from the key dimension.
    window_size : int
        Window size for aggregation.

    Returns
    -------
    Tensor
        Shape ``[B, H_q, W]``.  Sink tokens are **not** represented.
    """
    # 1. Sum over T query rows → per-token scores [B, H_q, S]
    token_scores = attn.sum(dim=-2)

    # 2-4. Strip sink, pad, window-reduce.
    return reduce_token_scores_to_windows(token_scores, num_sink, window_size)


def reduce_token_scores_to_windows(
    token_scores: Tensor,
    num_sink: int,
    window_size: int,
) -> Tensor:
    """Reduce per-token received-attention to per-window scores.

    This is steps 2-4 of :func:`compute_window_scores`, split out so callers
    that build ``token_scores`` incrementally (e.g. the flash hook accumulating
    over query-row chunks to bound peak memory at full context) can reuse the
    exact same sink-strip + pad + window-sum without materializing the full
    ``[B, H, T, S]`` attention matrix.

    Parameters
    ----------
    token_scores : Tensor
        Shape ``[B, H_q, S]`` — per-key total received attention (already
        summed over the query dimension).
    num_sink, window_size : int

    Returns
    -------
    Tensor
        Shape ``[B, H_q, W]``.  Sink tokens are **not** represented.
    """
    # 2. Strip sink prefix
    post_sink = token_scores[..., num_sink:]  # [B, H_q, S_post]

    # 3. Right-pad to make divisible by window_size
    s_post = post_sink.shape[-1]
    remainder = s_post % window_size
    if remainder != 0:
        pad_size = window_size - remainder
        post_sink = torch.nn.functional.pad(post_sink, (0, pad_size), value=0.0)

    # 4. einops.reduce to window scores
    window_scores = reduce(
        post_sink, "b h (w s) -> b h w", "sum", s=window_size
    )
    return window_scores


def reduce_two_tier_scores(
    token_scores: Tensor,
    num_sink: int,
    window_size: int,
    q_token_len: int,
    order: Tensor,
) -> Tensor:
    """Per-window scores for the **unsorted** two-tier effective-K layout.

    At ``q > 0`` the effective K is laid out ``[sink ‖ fp body ‖ Q]`` and is
    *not* globally id-sorted (``materialize_effective_kv`` skips the reorder
    because attention is order-free). This reduces that layout to the same
    ``[B, H_q, W]`` per-window scores the sorted layout produced, **bit-for-bit**:

    - Each tier is internally contiguous, ascending-id, window-size-aligned, and
      no window straddles the tier boundary — so the ordinary contiguous
      window-sum is correct within each tier. The fp body may have a trailing
      partial (local) window, handled by the same right-pad as the single-tier
      path; the Q tier is always whole windows.
    - Reducing each tier independently never reorders any window's ``ws``
      summands (they are position-ordered within their tier in both layouts), so
      the sums are identical to the sorted path's contiguous reduce.
    - ``order`` (from :func:`materialize_effective_kv`) is ``argsort`` of the
      tier-concatenated per-window ids, i.e. it scatters the physical
      ``[body ‖ Q]`` per-window sums back into **ascending-merged-id order** —
      the order ``state.window_scores`` / ``original_window_ids`` are kept in, so
      the running accumulation stays aligned.

    Parameters
    ----------
    token_scores : Tensor
        ``[B, H_q, S]`` per-key received attention, already summed over queries.
    num_sink, window_size : int
    q_token_len : int
        Number of Q-tier tokens at the tail of ``token_scores`` (``n_q * ws``).
    order : Tensor
        ``[B, W]`` int64 scatter permutation, physical → ascending-merged-id.

    Returns
    -------
    Tensor
        ``[B, H_q, W]``. Sink tokens are **not** represented.
    """
    s = token_scores.shape[-1]
    body_scores = token_scores[..., : s - q_token_len]
    q_scores = token_scores[..., s - q_token_len :]

    # fp body: strip sink, right-pad the trailing partial window, window-sum.
    body_w = reduce_token_scores_to_windows(body_scores, num_sink, window_size)
    # Q tier: whole windows, no sink — a direct contiguous window-sum.
    q_w = reduce(q_scores, "b h (w s) -> b h w", "sum", s=window_size)

    phys = torch.cat([body_w, q_w], dim=-1)                        # [B, H_q, W]
    idx = order.unsqueeze(1).expand(phys.shape[0], phys.shape[1], order.shape[1])
    return torch.gather(phys, -1, idx)


def accumulate(state_scores: Tensor, new_scores: Tensor) -> Tensor:
    """Accumulate new window scores into existing state scores (in-place +=).

    Parameters
    ----------
    state_scores : Tensor
        Shape ``[B, H_q, W]``, running cumulative scores.
    new_scores : Tensor
        Shape ``[B, H_q, W]``, scores from the latest step.

    Returns
    -------
    Tensor
        The mutated *state_scores* tensor (same storage).
    """
    state_scores += new_scores
    return state_scores


def fill_skipped_window_scores(
    exact: Tensor,
    keep: Tensor,
    logmass: Tensor,
) -> Tensor:
    """Complete the Q-tier window scores when only some windows were read.

    The gated read path dequantizes a fraction of the int2 tier, so only those
    windows receive real attention and only they produce a real score. Leaving
    the rest at zero is not an option: ``window_scores`` is what eviction ranks
    on, so a skipped window would score zero, rank last, and be dropped — the
    gate would silently destroy the tier it exists to read less often, and at
    ``quant_ratio = 0.7`` that is 70% of the evictable cache.

    So every skipped window is credited with the estimate its own card produced,
    rescaled onto the same footing as the real scores:

        c = sum(exact over selected) / sum(estimated over selected)

    ``c`` costs one extra reduction and needs nothing the step did not already
    compute — the selected windows have *both* a real and an estimated score, so
    the correction factor is free and self-calibrating. Without it the two
    populations sit on different scales and the ranking between them is
    arbitrary.

    Parameters
    ----------
    exact : ``[B, H_q, W]`` real per-window scores. Only entries where ``keep``
        are read; the rest may hold anything.
    keep : ``[B, H_q, W]`` bool — which windows were actually dequantized. Pass
        the KV-head mask expanded over its query-head group; a window read for a
        KV head was read for every query head sharing it.
    logmass : ``[B, H_q, W]`` the card's log-domain mass estimate for **every**
        window, from :func:`modules.quant.sketch.gate_and_score`.

    Returns
    -------
    ``[B, H_q, W]`` — ``exact`` where selected, ``c * estimate`` where not.
    """
    # Exponentiate relative to each head's own maximum. The offset cancels in the
    # ratio below, so this is a pure overflow guard, not an approximation.
    est = (logmass - logmass.amax(dim=-1, keepdim=True)).exp()
    sel = keep.to(est.dtype)
    num = (exact * sel).sum(dim=-1, keepdim=True)
    den = (est * sel).sum(dim=-1, keepdim=True)
    c = num / den.clamp_min(torch.finfo(est.dtype).tiny)
    return torch.where(keep, exact, c * est)


def expand_keep_to_query_heads(keep: Tensor, num_query_heads: int) -> Tensor:
    """``[B, H_kv, W]`` -> ``[B, H_q, W]``. A window read for a KV head was read
    for every query head sharing it, so the mask repeats along the group."""
    B, hkv, W = keep.shape
    return keep.repeat_interleave(num_query_heads // hkv, dim=1)
