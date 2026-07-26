"""EvictionPolicy — pure index / state-machine for windowed cache eviction.

Tracks region boundaries (sink, evictable, local) and the generation step
counter.  **Does not touch tensors** except via the ``window_scores`` input
to ``compute_retain_window_indices``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .config import ResolvedConfig


#: Decode step of the **first** eviction, INDEPENDENT of ``window_size``.
#:
#: **0 = compress the prompt at the earliest moment the design allows.** Prefill
#: itself cannot evict — the score hook is a ``forward_hook`` that runs *after*
#: the attention module, so during the prefill ``update()`` no scores exist yet.
#: The prefill scores first become readable on decode step 0's ``update()``, and
#: that is where the first compaction fires: the prompt is cut to budget before
#: the step-0 query attends, so every token from the first decode step onward is
#: generated against the compressed cache. That is the operating point the
#: prompt-compression baselines (SnapKV / AdaKV / DefensiveKV) are measured at,
#: which is what makes the comparison a comparison.
#:
#: A positive value delays the first compaction by that many decode steps and is
#: kept as a knob for ablations (and for reproducing the earlier step-8 result
#: set). It is *not* free: with ``first_eviction_step = N`` the prompt stays whole
#: through decode steps ``0 … N-1``, so any answer that finishes inside that
#: window is scored at FULL cache no matter what ``cache_budget`` says — which
#: silently turns short-answer datasets budget-invariant.
#:
#: See :meth:`EvictionPolicy.should_evict`.
FIRST_EVICTION_STEP = 0


class EvictionPolicy:
    """Stateful eviction controller.

    Parameters
    ----------
    resolved : ResolvedConfig
        Resolved configuration with concrete integer counts.
    """

    def __init__(self, resolved: ResolvedConfig) -> None:
        self.window_size: int = resolved.window_size
        self.num_sink_tokens: int = resolved.num_sink_tokens
        self.local_tokens: int = resolved.local_tokens
        self.local_windows: int = resolved.local_tokens // resolved.window_size
        self.top_k_windows: int = resolved.top_k_windows
        # Two-tier split (design.md §5, §7). Unused when quant_ratio == 0.
        self.quant_ratio: float = resolved.quant_ratio
        self.top_k_fp: int = resolved.top_k_fp
        self.N_q: int = resolved.N_q
        # total_tokens tracks the MERGED length (T_fp + T_q) so window/region
        # counts stay aligned with the merged axis and get_seq_length (design §5).
        # At q=0, T_q == 0, so this is exactly the fp-only token count.
        self.total_tokens: int = 0
        # Decode step of the first eviction — the config knob (default
        # FIRST_EVICTION_STEP = 0, i.e. compress the prompt before the step-0
        # query attends), carried through ResolvedConfig. An instance attribute
        # so ablations (and tests isolating eviction *mechanics* from this
        # *timing* policy) can delay it without rebuilding the config.
        self.first_eviction_step: int = resolved.first_eviction_step

    # -----------------------------------------------------------------
    # State bookkeeping
    # -----------------------------------------------------------------

    def initialize_after_prefill(self, prefill_len: int) -> None:
        """Set state after the initial prefill pass."""
        self.total_tokens = prefill_len

    def extend_total_after_append(self, n_new: int) -> None:
        """Update total token count after appending *n_new* tokens."""
        self.total_tokens += n_new

    def set_total_after_compaction(self, new_total: int) -> None:
        """Update state after eviction compaction."""
        self.total_tokens = new_total

    # -----------------------------------------------------------------
    # Eviction trigger
    # -----------------------------------------------------------------

    def should_evict(self, step: int) -> bool:
        """Return ``True`` if eviction should run at decode *step*.

        The **first** eviction fires at a fixed ``first_eviction_step``,
        **independent of ``window_size``**, so a window-size sweep shares one
        first-compaction point instead of first evicting at a ws-dependent step
        (which would confound the comparison). After it, the natural window
        cadence resumes — every step where ``step % window_size == 0``, aligned to
        absolute step 0.

        **At the default** :data:`FIRST_EVICTION_STEP` **= 0** the two rules
        coincide and the schedule is simply ``0, ws, 2*ws, …`` for every
        ``window_size``: the prompt is compressed on the first decode step, before
        that step's query attends, and every generated token from then on is
        produced against the budgeted cache. Prefill itself cannot evict — the
        score hook runs *after* the attention module, so no scores exist during
        the prefill ``update()``; step 0 is the earliest point at which the
        prefill scores are readable, and is where they are consumed.

        With a **positive** ``first_eviction_step = N`` nothing evicts before step
        ``N`` (the prompt stays whole through steps ``0 … N-1``), then:

        * ``ws == 8``, ``N == 8`` → a uniform ``8, 16, 24`` …
        * ``ws == 12``, ``N == 8`` → ``8`` (forced), ``12``, ``24`` … — a short
          ``8 → 12`` gap, then full windows.
        * ``ws < N`` suppresses the natural boundaries below ``N`` and resumes at
          the next multiple: ``ws == 3``, ``N == 8`` → ``8, 9, 12`` … .

        Ablations aside, a positive value means every answer that terminates
        within ``N`` decode steps is scored at full cache regardless of
        ``cache_budget`` — see :data:`FIRST_EVICTION_STEP`.
        """
        if step < self.first_eviction_step:
            return False
        if step == self.first_eviction_step:
            return True
        return step % self.window_size == 0

    # -----------------------------------------------------------------
    # Region helpers (computed from current state)
    # -----------------------------------------------------------------

    @property
    def post_sink_tokens(self) -> int:
        return max(self.total_tokens - self.num_sink_tokens, 0)

    @property
    def num_total_windows(self) -> int:
        ps = self.post_sink_tokens
        return (ps + self.window_size - 1) // self.window_size if ps > 0 else 0

    @property
    def num_evictable_windows(self) -> int:
        return max(self.num_total_windows - self.local_windows, 0)

    # -----------------------------------------------------------------
    # Retain indices — window granularity
    # -----------------------------------------------------------------

    def compute_retain_window_indices(
        self, window_scores: Tensor
    ) -> Tensor:
        """Primary retain-decision method at **window** granularity.

        Algorithm (all single-call tensor ops):
        1. ``mean_scores = window_scores.mean(dim=1)`` → ``[B, W_total]``.
        2. Slice to evictable window range.
        3. ``torch.topk`` on the slice.
        4. Sort indices chronologically (never by score).
        5. ``cat([sorted_topk_idx, local_window_idx], dim=-1)``

        Parameters
        ----------
        window_scores : Tensor
            Shape ``[B, H_q, W_total]``.

        Returns
        -------
        Tensor
            Shape ``[B, W_retained]``, window indices to keep.
        """
        B = window_scores.shape[0]
        W_total = window_scores.shape[2]
        device = window_scores.device

        # Number of local and evictable windows
        local_w = min(self.local_windows, W_total)
        evictable_w = W_total - local_w

        # 1. Mean across heads
        mean_scores = window_scores.mean(dim=1)  # [B, W_total]

        # 2. Slice to evictable window range [0, evictable_w)
        evictable_scores = mean_scores[:, :evictable_w]  # [B, evictable_w]

        # Edge case: if num_evictable ≤ top_k, retain all evictable
        k = min(self.top_k_windows, evictable_w)

        if k == 0 or evictable_w == 0:
            # No evictable windows to select — just keep local
            local_idx = torch.arange(
                W_total - local_w, W_total, device=device, dtype=torch.long
            ).unsqueeze(0).expand(B, -1)
            return local_idx

        if k >= evictable_w:
            # Keep all evictable + all local
            all_idx = torch.arange(
                W_total, device=device, dtype=torch.long
            ).unsqueeze(0).expand(B, -1)
            return all_idx

        # 3. Top-K on evictable slice
        _, topk_idx = torch.topk(evictable_scores, k, dim=-1)  # [B, k]

        # 4. Sort indices chronologically
        topk_sorted, _ = torch.sort(topk_idx, dim=-1)

        # 5. Concatenate with local window indices
        local_idx = torch.arange(
            W_total - local_w, W_total, device=device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)

        retained = torch.cat([topk_sorted, local_idx], dim=-1)  # [B, k + local_w]
        return retained

    # -----------------------------------------------------------------
    # Two-tier retain + tier assignment (design.md §5) — q > 0 only, B = 1
    # -----------------------------------------------------------------

    def tier_counts(self, W: int) -> "tuple[int, int, int]":
        """``(k_fp, n_q, local_w)`` for a merged axis of ``W`` windows.

        **Host ints, and identical for every row** — that is the load-bearing
        fact behind batching (BATCHING_PLAN.md §3). Both counts derive only from
        the resolved config and ``W`` (the scores tensor's width, shared across
        the batch), never from the scores themselves. So rows evict *divergently*
        — different windows — but always retain the **same number** of fp and Q
        windows. ``T_fp`` and ``T_q`` are therefore equal across rows and the
        effective K/V stays a dense ``[B, H_kv, T_total, D]``: no padding, no
        keep-mask, no raggedness (as long as prompts are equal-length).

        The cache also uses these as loop-free tensor widths and to size the fp
        store, so it never has to sync to learn its own shapes.
        """
        local_w = min(self.local_windows, W)
        evictable_w = W - local_w
        if evictable_w <= 0:
            return 0, 0, local_w
        k_fp = min(self.top_k_fp, evictable_w)
        n_q = min(self.N_q, evictable_w - k_fp)
        return k_fp, n_q, local_w

    def compute_two_tier_retain(
        self, window_scores: Tensor
    ) -> "tuple[Tensor, Tensor]":
        """Rank the evictable band and assign each survivor a tier.

        Sink + local are force-fp (never eligible for Q). The evictable band is
        partitioned by rank: top ``top_k_fp`` → fp, next ``N_q`` → Q, rest
        dropped (design §5 steps 1–2). Returns the retained windows in
        **chronological** (ascending merged-index) order plus their assigned
        tier, so the caller can drive demote/promote/keep against the store.

        **Per row.** Rows rank independently and therefore retain *different*
        windows — but always the same *number* of them, since ``k_fp`` / ``n_q``
        come from :meth:`tier_counts` (config + ``W``, both shared across the
        batch). That is what keeps the result rectangular (BATCHING_PLAN.md §3).

        Parameters
        ----------
        window_scores : Tensor
            Shape ``[B, H_q, W]`` — merged-axis cumulative scores.

        Returns
        -------
        retained_idx : Tensor
            Shape ``[B, W_retained]`` — merged-axis indices, ascending per row.
        new_tier : Tensor
            Shape ``[B, W_retained]`` — 0 = fp, 1 = Q, aligned with
            ``retained_idx``.
        """
        B, _, W = window_scores.shape
        device = window_scores.device

        k_fp, n_q, local_w = self.tier_counts(W)
        evictable_w = W - local_w
        local_idx = (
            torch.arange(W - local_w, W, device=device, dtype=torch.long)
            .unsqueeze(0).expand(B, -1)
        )
        local_tier = torch.zeros(B, local_w, device=device, dtype=torch.long)

        if evictable_w <= 0:
            return local_idx.contiguous(), local_tier

        mean = window_scores.mean(dim=1)            # [B, W]
        ev_scores = mean[:, :evictable_w]

        # Rank the evictable band by score, descending: top k_fp → fp, next n_q → Q.
        order = torch.argsort(ev_scores, dim=-1, descending=True)
        fp_sel = order[:, :k_fp]
        q_sel = order[:, k_fp:k_fp + n_q]

        ev_idx = torch.cat([fp_sel, q_sel], dim=-1)
        ev_tier = torch.cat([
            torch.zeros(B, k_fp, device=device, dtype=torch.long),
            torch.ones(B, n_q, device=device, dtype=torch.long),
        ], dim=-1)
        # Sort the retained evictable windows chronologically (by merged index).
        perm = torch.argsort(ev_idx, dim=-1)
        ev_idx = torch.gather(ev_idx, 1, perm)
        ev_tier = torch.gather(ev_tier, 1, perm)

        retained = torch.cat([ev_idx, local_idx], dim=-1)
        tier = torch.cat([ev_tier, local_tier], dim=-1)
        return retained, tier

    # -----------------------------------------------------------------
    # Retain indices — token granularity
    # -----------------------------------------------------------------

    def expand_to_token_indices(
        self, retained_window_idx: Tensor, num_windows: int
    ) -> Tensor:
        """Expand window indices to absolute token indices.

        Prepends sink prefix.  Trims trailing partial window via geometric
        cap computed without touching tensor data (Python int arithmetic).

        Parameters
        ----------
        retained_window_idx : Tensor
            Shape ``[B, W_retained]``.
        num_windows : int
            The total scored-window count (``window_scores.shape[2]``, a host
            int). Needed to locate the one window that can be partial — the
            newest scored window, index ``num_windows - 1`` — so the valid-token
            count can be computed without a device sync (see below).

        Returns
        -------
        Tensor
            Shape ``[B, T_retained]``.
        """
        B, W_retained = retained_window_idx.shape
        device = retained_window_idx.device

        # Sink prefix [0, 1, ..., num_sink-1]
        sink_idx = torch.arange(
            self.num_sink_tokens, device=device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)  # [B, num_sink]

        # Expand windows to tokens:
        # For window w: tokens = num_sink + w * window_size + offset
        offsets = torch.arange(
            self.window_size, device=device, dtype=torch.long
        )  # [window_size]

        # [B, W_retained, 1] * window_size + [window_size] → [B, W_retained, window_size]
        token_idx = (
            self.num_sink_tokens
            + retained_window_idx.unsqueeze(-1) * self.window_size
            + offsets
        )
        token_idx = token_idx.reshape(B, -1)  # [B, W_retained * window_size]

        # Concatenate sink + window tokens
        all_idx = torch.cat([sink_idx, token_idx], dim=-1)  # [B, total]

        # Mask out indices that exceed the actual sequence length
        # (partial last window produces OOB token positions).
        valid_mask = all_idx < self.total_tokens  # [B, total]

        # Rectangular gather needs one valid-token count. It is the SAME for every
        # row, so we compute it as a host int rather than reducing the mask and
        # stalling the pipeline on `.item()` every eviction. Three facts make this
        # exact (they are the same ones batching rests on, BATCHING_PLAN.md §3):
        #   * `total_tokens` is a scalar shared across the batch, and rows retain
        #     different windows but always the same COUNT (W_retained);
        #   * the only window that can be partial is the newest SCORED window,
        #     index num_windows-1: its token indices are num_sink+(num_windows-1)*
        #     ws + [0..ws-1], and every earlier retained window sits fully below
        #     total_tokens; and
        #   * that newest window is ALWAYS retained, because it is local and
        #     local_windows >= 1 by config (an int local is a window_size multiple
        #     >= window_size; a float local ceils then snaps up to one), and it is
        #     the max retained index.
        # So the valid count is sink + all retained windows minus that newest
        # window's out-of-range tail. NB the tail must be measured off the newest
        # window's START, not off total_tokens % ws: `total_tokens` can run ahead
        # of the scored-window span (then the newest window is fully in range and
        # there is NO tail), which a total_tokens%ws form gets wrong.
        if W_retained == 0:
            min_valid = min(self.num_sink_tokens, self.total_tokens)
        else:
            last_start = self.num_sink_tokens + (num_windows - 1) * self.window_size
            last_valid = min(
                max(self.total_tokens - last_start, 0), self.window_size
            )
            oob = self.window_size - last_valid   # newest window's OOB tail
            min_valid = self.num_sink_tokens + W_retained * self.window_size - oob

        # Gather only valid indices: sort valid-first via the mask, take prefix
        # argsort of ~mask (False=0 sorts before True=1) gives valid-idx-first order
        order = torch.argsort(~valid_mask, dim=1, stable=True)  # valid first
        all_idx = torch.gather(all_idx, 1, order)[:, :min_valid]

        return all_idx
