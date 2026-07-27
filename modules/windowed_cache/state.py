"""CacheState — tensor storage for a single attention layer's KV cache.

Layout follows original sequence order: ``[sink | evictable | local]``.
Region boundaries are tracked by :class:`EvictionPolicy`, not here.

The fp store is **preallocated** and appended into in place. It used to
``torch.cat`` every decode step, per layer — reallocating and copying the whole
store to add one token. The fp tier is ~57% of per-row KV bytes, so at max batch
that copy is ~4x the weight traffic and the single largest term in the step
(BATCHING_PLAN.md §5); it also doubles peak VRAM at the moment of copy, which
directly caps the batch size the whole method is trying to raise. At B = 1 it is
only ~0.2% of runtime — real but invisible, which is why it survived this long.

**Two capacities, not one.** ``StaticCache`` preallocates to
``prefill + max_new_tokens`` because it never evicts, so that *is* its steady
state. This cache evicts: the store compacts back to the budget every
``window_size`` steps and its steady-state length is ~``budget``, not
~``prefill``. Sizing the buffer at ``prefill + max_new`` would therefore pin the
fp store at its **un-evicted** size for the whole decode — measured on
Llama-3.1-8B at the qasper steady state, 708 MB/row instead of 74 MB/row, which
caps B at ~73 against the ~458 that §5's thesis rests on. That is worse than the
``cat`` it replaces, and it defeats the point of evicting at all.

So: ``capacity`` is the **eviction budget** (the steady state, where every decode
step lands), while ``prefill_capacity`` covers the one-off prompt, which is
genuinely larger and unavoidable — the whole prompt's KV must exist before the
first eviction can score it. The prefill-sized buffer is released at that first
eviction, and every step after it runs allocation-free at budget size.

``key_states`` / ``value_states`` / ``position_ids`` are **views** into those
buffers, so they are not contiguous when the length is below capacity. Reads
are unaffected; writes must go through :meth:`append`, :meth:`slice_and_keep`,
or :meth:`replace`, which own the buffers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


class CacheState:
    """Mutable tensor state for one layer's KV cache.

    Parameters
    ----------
    capacity : int
        **Steady-state** size, in tokens: the post-eviction budget plus a window
        of growth. This is what the store settles at after the first eviction and
        holds for the rest of the decode, so it is what per-row memory — and
        therefore the max batch — is actually charged.
    prefill_capacity : int
        Size for the **initial** allocation, covering the prompt (which has to be
        resident before the first eviction can score it). Defaults to
        ``capacity``. Released the first time :meth:`replace` compacts back to
        the budget.

    Both are hints, not caps: :meth:`append` grows the buffer if a caller exceeds
    them, so an under-sized hint costs a realloc, never correctness.

    Attributes
    ----------
    key_states : Tensor
        Shape ``[B, H_kv, T, D]`` — a view into the preallocated buffer.
    value_states : Tensor
        Shape ``[B, H_kv, T, D]`` — a view into the preallocated buffer.
    position_ids : Tensor
        Shape ``[B, T]``, int64.  After eviction, gathered **per row** to the
        surviving tokens' **original** positions so keys keep their original
        RoPE phase.  Each batch row may evict different windows, so rows can
        carry different surviving positions.  Only rebased to ``arange(T)``
        (per row) when ``rerotate_keys`` runs (the opt-in StreamingLLM path).
    window_scores : Tensor
        Shape ``[B, H_q, W]``.  Running cumulative per-window scores.
    original_window_ids : Tensor
        Shape ``[B, W]``, int64.  Maps each surviving compact window index to
        its original sequence window index (0-based after sinks), **per row**.
        Stays identity before the first eviction; gathered per row alongside
        ``window_scores`` at every subsequent eviction so that compact
        top-K indices can be translated back to original positions for
        faithful Jaccard comparison.
    """

    __slots__ = ("key_states", "value_states", "position_ids",
                 "window_scores", "original_window_ids",
                 "_key_buf", "_val_buf", "_pos_buf", "_capacity", "_prefill_capacity")

    def __init__(self, capacity: int = 0, prefill_capacity: int = 0) -> None:
        self.key_states: Optional[Tensor] = None
        self.value_states: Optional[Tensor] = None
        self.position_ids: Optional[Tensor] = None
        self.window_scores: Optional[Tensor] = None
        self.original_window_ids: Optional[Tensor] = None
        # Backing storage; the public tensors above are views of [:, :, :T].
        self._key_buf: Optional[Tensor] = None
        self._val_buf: Optional[Tensor] = None
        self._pos_buf: Optional[Tensor] = None
        self._capacity: int = max(int(capacity), 0)
        self._prefill_capacity: int = max(int(prefill_capacity), self._capacity)

    # -----------------------------------------------------------------
    # seq_length property
    # -----------------------------------------------------------------

    @property
    def seq_length(self) -> int:
        """Current sequence length (number of cached tokens)."""
        if self.key_states is None:
            return 0
        return self.key_states.shape[2]

    @property
    def buffer_capacity(self) -> int:
        """Tokens the fp buffers can hold before a realloc. Diagnostics/tests."""
        return 0 if self._key_buf is None else self._key_buf.shape[2]

    # -----------------------------------------------------------------
    # buffer management
    # -----------------------------------------------------------------

    def _reslice(self, length: int) -> None:
        """Re-point the public views at ``[:, :, :length]`` of the buffers."""
        self.key_states = self._key_buf[:, :, :length]
        self.value_states = self._val_buf[:, :, :length]
        self.position_ids = self._pos_buf[:, :length]

    def _allocate(self, key: Tensor, capacity: int) -> None:
        B, H, _, D = key.shape
        self._key_buf = torch.empty(
            (B, H, capacity, D), dtype=key.dtype, device=key.device
        )
        self._val_buf = torch.empty(
            (B, H, capacity, D), dtype=key.dtype, device=key.device
        )
        self._pos_buf = torch.empty(
            (B, capacity), dtype=torch.long, device=key.device
        )

    def _grow(self, needed: int) -> None:
        """Reallocate to at least ``needed`` tokens, preserving the live prefix.

        A safety net, not the steady state: with the buffer sized to the eviction
        budget plus a window of slack, and eviction compacting every
        ``window_size`` steps, this should not fire during decode. Growth is
        exact rather than doubling — the length here is bounded by the budget, so
        there is no amortization argument to make, and over-allocating costs the
        batch capacity this is all in service of.
        """
        cur = self.seq_length
        new_cap = max(needed, 1)
        old_k, old_v, old_p = self._key_buf, self._val_buf, self._pos_buf
        self._allocate(old_k, new_cap)
        if cur > 0:
            self._key_buf[:, :, :cur] = old_k[:, :, :cur]
            self._val_buf[:, :, :cur] = old_v[:, :, :cur]
            self._pos_buf[:, :cur] = old_p[:, :cur]
        self._reslice(cur)

    # -----------------------------------------------------------------
    # append
    # -----------------------------------------------------------------

    def append(
        self,
        key: Tensor,
        value: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> None:
        """Append new key/value states along the sequence dimension.

        Writes into the preallocated buffer in place — no reallocation, no copy
        of the existing store.

        Parameters
        ----------
        key : Tensor
            Shape ``[B, H_kv, N_new, D]``.
        value : Tensor
            Shape ``[B, H_kv, N_new, D]``.
        position_ids : Tensor, optional
            Shape ``[N_new]`` (shared across the batch, e.g. HF's
            ``cache_position``) or ``[B, N_new]`` (per row).  If ``None``,
            auto-increments from the current length.  Stored canonically as
            ``[B, N_new]``.
        """
        n_new = key.shape[2]
        B = key.shape[0]
        device = key.device
        cur = self.seq_length

        if self._key_buf is None:
            # First append is the prompt: size for it, not for the budget.
            self._allocate(key, max(self._prefill_capacity, n_new))
        elif cur + n_new > self.buffer_capacity:
            self._grow(cur + n_new)

        # position_ids is canonical as [B, N_new]. A 1-D input is shared across
        # the batch and broadcast; a [B, N_new] input is used as-is.
        if position_ids is not None:
            new_pos = position_ids
            if new_pos.dim() == 1:
                new_pos = new_pos.unsqueeze(0).expand(B, -1)
        else:
            new_pos = (
                torch.arange(cur, cur + n_new, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(B, -1)
            )

        self._key_buf[:, :, cur:cur + n_new] = key
        self._val_buf[:, :, cur:cur + n_new] = value
        self._pos_buf[:, cur:cur + n_new] = new_pos.to(self._pos_buf.device)
        self._reslice(cur + n_new)

    # -----------------------------------------------------------------
    # replace
    # -----------------------------------------------------------------

    def replace(self, key: Tensor, value: Tensor, position_ids: Tensor) -> None:
        """Adopt a freshly-built store, copying it into the preallocated buffers.

        The two-tier eviction rebuilds ``[sink ‖ fp windows by id]`` from scratch
        (``slice_and_keep`` alone cannot express a promotion's *insertion*), so
        it needs a way to hand the state a whole new tensor without giving up
        preallocation.

        ``key`` / ``value`` / ``position_ids`` must not alias the buffers; if
        they do, they are cloned first (the alternative is a self-overlapping
        copy, which is silently wrong).

        This is also where the prefill-sized buffer is **released**. Compaction
        is the moment the store drops from ~prefill to ~budget, so the buffer
        resizes down to :attr:`_capacity` here and stays there — that steady-state
        size is what per-row memory, and therefore max batch, is charged for.
        Every subsequent eviction lands on the same target, so this reallocates
        exactly once and never thrashes.
        """
        if self._key_buf is not None:
            if key.untyped_storage().data_ptr() == self._key_buf.untyped_storage().data_ptr():
                key = key.clone()
            if value.untyped_storage().data_ptr() == self._val_buf.untyped_storage().data_ptr():
                value = value.clone()
            if position_ids.untyped_storage().data_ptr() == self._pos_buf.untyped_storage().data_ptr():
                position_ids = position_ids.clone()

        n = key.shape[2]
        target = max(self._capacity, n)
        if self._key_buf is None or self.buffer_capacity != target:
            self._allocate(key, target)

        self._key_buf[:, :, :n] = key
        self._val_buf[:, :, :n] = value
        self._pos_buf[:, :n] = position_ids.to(self._pos_buf.device)
        self._reslice(n)

    # -----------------------------------------------------------------
    # slice_and_keep
    # -----------------------------------------------------------------

    def slice_and_keep(self, retain_token_indices: Tensor) -> None:
        """Compact the cache by keeping only the tokens at *retain_token_indices*.

        Uses ``torch.gather`` with ``.expand()`` — never ``.repeat()``.

        Parameters
        ----------
        retain_token_indices : Tensor
            Shape ``[B, T_retained]``, int64 indices into the seq dimension.
        """
        B, T_retained = retain_token_indices.shape
        H_kv = self.key_states.shape[1]
        D = self.key_states.shape[3]

        # Expand indices for gather: [B, H_kv, T_retained, D]
        idx_k = (
            retain_token_indices
            .unsqueeze(1)   # [B, 1, T_retained]
            .unsqueeze(3)   # [B, 1, T_retained, 1]
            .expand(B, H_kv, T_retained, D)
        )

        # Gather allocates, so the result never aliases the buffer we copy it
        # back into.
        new_k = torch.gather(self.key_states, dim=2, index=idx_k)
        new_v = torch.gather(self.value_states, dim=2, index=idx_k)

        # Gather position_ids to the surviving tokens' ORIGINAL positions.
        # The keys retain the RoPE rotation they were stored with, so their
        # positions must stay original for the query<->key relative phase to be
        # correct. (Rebasing to arange is only valid when the keys are *also*
        # re-rotated and the query position is rebased -- see rerotate_keys.)
        # position_ids is [B, T]; gather each row independently because rows
        # may evict different windows and thus keep different positions.
        new_p = torch.gather(
            self.position_ids, 1, retain_token_indices.to(self.position_ids.device)
        )

        self.replace(new_k, new_v, new_p)

    # -----------------------------------------------------------------
    # rerotate_keys
    # -----------------------------------------------------------------

    def rerotate_keys(
        self,
        rope_module: torch.nn.Module,
        old_position_ids: Tensor,
    ) -> None:
        """Strip old RoPE rotation and re-apply with contiguous positions.

        Uses the model's own ``apply_rotary_pos_emb`` to preserve NTK / YaRN
        scaling.  Values are **not** rotated (RoPE applies to keys only in
        LLaMA / Qwen).

        Parameters
        ----------
        rope_module : nn.Module
            The model's rotary embedding module (e.g. ``model.model.rotary_emb``).
        old_position_ids : Tensor
            Shape ``[B, T_retained]`` (per row), or ``[T_retained]`` (shared
            across the batch, broadcast).  The original positions before
            compaction.
        """
        # Lazy import to avoid circular deps at module level
        try:
            from transformers.models.llama.modeling_llama import (
                apply_rotary_pos_emb,
            )
        except ImportError:
            from transformers.models.qwen2.modeling_qwen2 import (
                apply_rotary_pos_emb,
            )

        B = self.key_states.shape[0]
        T_retained = self.key_states.shape[2]
        device = self.key_states.device

        # Old positions → cos/sin.  Accept a shared 1-D vector or per-row [B, T].
        old_pos = old_position_ids
        if old_pos.dim() == 1:
            old_pos = old_pos.unsqueeze(0).expand(B, -1)
        cos_old, sin_old = rope_module(self.value_states, old_pos)

        # Undo old rotation: cos(-θ)=cos(θ), sin(-θ)=-sin(θ).
        #
        # Only an inverse when the rotary module folds no scaling into cos/sin.
        # YaRN sets attention_scaling a = 0.1·ln(factor)+1 (1.1386 at factor=4),
        # making the forward map a·R(θ); the negated-sin pass then lands on a²·k
        # and the re-rotation below would emit a³·R(θ_new)·k against a true
        # a·R(θ_new)·k. Divide the a² out here — the single place both the
        # unrotate and the rotate see. a == 1 on llama3 / default / linear /
        # dynamic rope, so Llama-3.1 and Mistral skip the branch and stay
        # byte-identical. See modules.quant.effective.rope_attention_scaling.
        _, k_unrotated = apply_rotary_pos_emb(
            self.key_states, self.key_states, cos_old, -sin_old
        )
        from modules.quant.effective import rope_attention_scaling

        a = rope_attention_scaling(rope_module)
        if a != 1.0:
            k_unrotated = k_unrotated / (a * a)

        # New contiguous positions (same for every row)
        new_pos = (
            torch.arange(T_retained, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
        )
        cos_new, sin_new = rope_module(self.value_states, new_pos)

        # Apply new rotation
        _, k_rerotated = apply_rotary_pos_emb(
            k_unrotated, k_unrotated, cos_new, sin_new
        )

        # Keys now live at contiguous positions [0..T_retained-1]; keep the
        # bookkeeping in sync (slice_and_keep left the *original* positions) so
        # a subsequent eviction snapshots correct "old" positions.
        self.replace(k_rerotated, self.value_states, new_pos.contiguous())
