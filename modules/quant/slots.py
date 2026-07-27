"""Dense slot table for the Q tier (design.md §4, §6; BATCHING_PLAN.md §4).

The batch-carrying replacement for the per-window ``Dict[int, LedgerEntry]``
ledger. Same contract — §6's frozen record per Q window — but as tensors with a
leading **row** axis, because rows evict divergently: row 0 may hold windows
``{1, 5}`` in int2 while row 1 holds ``{4, 11}``. A dict keyed by window id
cannot express that; a slot table can.

Layout (one layer, ``B`` rows, ``N`` slots)::

    key_codes   [B, N, H_kv, D, ws//4]  uint8   channel-major, 4 tokens/byte
    key_scale   [B, N, H_kv, D]         fp16    pinned grid
    key_zero    [B, N, H_kv, D]         fp16
    val_codes   [B, N, H_kv, ws, D//4]  uint8   token-major, 4 channels/byte
    val_scale   [B, N, H_kv, ws]        fp16
    val_zero    [B, N, H_kv, ws]        fp16
    slot_wid    [B, N]                  int64   original_window_id; -1 = free
    slot_active [B, N]                  bool    active vs dormant (§10)
    slot_pos    [B, N, ws]              int64   frozen original positions

**Slots are addressed by rank, not by window id.** A window id is unbounded
(generation keeps minting them) so it cannot index a fixed table; instead every
lookup resolves a window id to a slot through a ``[B, W, N]`` equality match.
The table's invariant — **at most one slot per row carries a given window id** —
is what makes that match resolve to a single slot, so ``argmax`` over it is
unambiguous rather than first-match-wins.

**Dormant entries stay resident.** design §10 forbids re-quantizing on
re-demotion (it would pick up fp16 rounding from the intermediate promote
dequant and could flip boundary codes), so a promoted window keeps its slot with
``slot_active = False``. That is why allocation goes through a free list rather
than rebuilding a dense ``[B, n_q]`` table each eviction: the table holds active
**and** dormant entries, and only :meth:`retain_only` frees anything.

Sizing: see :func:`n_slots_for`.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

FREE = -1
"""``slot_wid`` sentinel for an unoccupied slot. Real window ids are >= 0."""


def n_slots_for(top_k_fp: int, n_q: int) -> int:
    """Slots needed per row, with margin. Bounded — no growth policy (§4).

    ``retain_only`` frees every entry whose window was dropped outright, so live
    entries at any instant are (active Q) + (dormant), and the eviction cycle
    bounds both:

    - **active** <= ``N_q``: exactly the ``n_q`` windows the rank assigns to the
      Q tier.
    - **dormant** <= ``top_k_fp``: a dormant entry's window has been demoted at
      least once, so it is *evictable* (never sink/local — those are force-fp and
      never enter the Q band), and it survived ``retain_only``, so it is one of
      the ``k_fp`` retained evictable-fp windows. A window's trajectory is
      one-way — born local, ages into evictable, never returns — so a local
      window can never carry a dormant entry.

    Peak is measured **after** ``retain_only`` and before the fresh demotions are
    written, which is why the cache runs ``retain_only`` first: that ordering
    makes ``top_k_fp + N_q`` exact rather than merely likely. The margin below is
    slack against an off-by-one, not against the bound being wrong;
    :meth:`QuantSlotTable.validate` asserts the real thing in tests.
    """
    return top_k_fp + n_q + 8


class QuantSlotTable:
    """Per-row Q-tier slot storage for one layer.

    Every method is loop-free and sync-free: the table is addressed with
    tensor indices only, so nothing here drags the Q tier's identity
    bookkeeping back onto the host (BATCHING_PLAN.md §2).
    """

    def __init__(
        self,
        batch_size: int,
        n_slots: int,
        window_size: int,
        head_dim: int,
        num_kv_heads: int,
        device: torch.device,
        grid_dtype: torch.dtype = torch.float16,
    ) -> None:
        B, N, H, D, S = batch_size, n_slots, num_kv_heads, head_dim, window_size
        self.batch_size = B
        self.n_slots = N
        self.window_size = S
        self.head_dim = D
        self.num_kv_heads = H

        self.key_codes = torch.zeros((B, N, H, D, S // 4), dtype=torch.uint8, device=device)
        # The (scale, zero) grid. `grid_dtype` MUST be a 2-byte float: the budget
        # resolver charges it at 2 bytes/element and at int2 it is the dominant
        # cost of a Q window, so a wider one would move every reported
        # compression ratio. See modules.quant.quantizer.grid_dtype_for for why
        # it follows the KV dtype (fp16 KV keeps fp16 and stays bit-identical;
        # anything else takes bf16, which cannot overflow).
        self.grid_dtype = grid_dtype
        self.key_scale = torch.zeros((B, N, H, D), dtype=grid_dtype, device=device)
        self.key_zero = torch.zeros((B, N, H, D), dtype=grid_dtype, device=device)
        self.val_codes = torch.zeros((B, N, H, S, D // 4), dtype=torch.uint8, device=device)
        self.val_scale = torch.zeros((B, N, H, S), dtype=grid_dtype, device=device)
        self.val_zero = torch.zeros((B, N, H, S), dtype=grid_dtype, device=device)
        self.slot_wid = torch.full((B, N), FREE, dtype=torch.long, device=device)
        self.slot_active = torch.zeros((B, N), dtype=torch.bool, device=device)
        self.slot_pos = torch.zeros((B, N, S), dtype=torch.long, device=device)

        # Row offsets for flat indexing. Scattering with a broadcast [B, n, H, D,
        # ws//2] index tensor would allocate an int64 index the size of the codes
        # themselves (GBs at max B); flattening (B, N) -> B*N and indexing dim 0
        # with a [B*n] vector costs kilobytes instead.
        self._row_base = (torch.arange(B, device=device) * N).unsqueeze(1)  # [B, 1]

    # -- flat addressing -----------------------------------------------------

    def _flat(self, slot_idx: Tensor) -> Tensor:
        """``[B, n]`` slot indices -> ``[B*n]`` indices into the flattened axis."""
        return (slot_idx + self._row_base).reshape(-1)

    # -- lookup --------------------------------------------------------------

    def lookup(self, wids: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Resolve window ids to slots, per row.

        Parameters
        ----------
        wids : ``[B, W]`` int64 — window ids to look up (>= 0, unique per row).

        Returns
        -------
        has_entry : ``[B, W]`` bool — an entry exists (active **or** dormant).
        is_active : ``[B, W]`` bool — the entry exists and is in the Q tier.
        slot_of : ``[B, W]`` int64 — the owning slot; **only valid where
            ``has_entry``** (0 elsewhere, from argmax over an all-False row).
        match : ``[B, W, N]`` bool — the raw match, reused by
            :meth:`retain_only` (its ``any(1)`` is exactly the keep mask).

        Free slots hold ``-1`` and real ids are >= 0, so a free slot never
        matches. At most one slot per row can carry a given id (the table's
        invariant), so ``argmax`` picks that slot outright.
        """
        match = self.slot_wid.unsqueeze(1) == wids.unsqueeze(2)      # [B, W, N]
        has_entry = match.any(-1)
        is_active = (match & self.slot_active.unsqueeze(1)).any(-1)
        slot_of = match.to(torch.uint8).argmax(-1)
        return has_entry, is_active, slot_of, match

    # -- allocation ----------------------------------------------------------

    def free_slots(self, n: int) -> Tensor:
        """The ``n`` lowest-indexed free slots per row: ``[B, n]`` int64.

        Callers pair this with a *validity* mask and must write through
        :meth:`write`, which no-ops on invalid lanes. Only lanes
        ``j < n_free[row]`` are guaranteed free; the bound in
        :func:`n_slots_for` guarantees every **valid** lane is.
        """
        is_free = self.slot_wid == FREE
        order = torch.argsort(~is_free, dim=1, stable=True)   # free slots first
        return order[:, :n]

    def write(
        self,
        slot_idx: Tensor,
        valid: Tensor,
        wid: Tensor,
        k_codes: Tensor,
        k_scale: Tensor,
        k_zero: Tensor,
        v_codes: Tensor,
        v_scale: Tensor,
        v_zero: Tensor,
        pos: Tensor,
    ) -> None:
        """Write ``n`` fresh entries per row, masked by ``valid``.

        Reads each target's current contents and writes them back unchanged on
        invalid lanes (read-modify-write) rather than masking the *index*. That
        matters: the caller sizes ``slot_idx`` by the worst case (``n_q``), so
        trailing lanes can point at an occupied slot when a row has few fresh
        demotions. Masking the value leaves those slots untouched; masking the
        index would need somewhere safe to dump the write, and there isn't one.

        ``slot_idx`` must be duplicate-free per row (it is — it comes from
        :meth:`free_slots`, a permutation slice), so the flat index put below is
        deterministic.

        Parameters
        ----------
        slot_idx : ``[B, n]`` target slots.
        valid : ``[B, n]`` bool — which lanes carry a real fresh demotion.
        wid : ``[B, n]`` int64 — window ids (``-1`` on invalid lanes).
        k_codes .. v_zero : ``[B, n, ...]`` quantized fields.
        pos : ``[B, n, ws]`` int64 frozen positions.
        """
        fi = self._flat(slot_idx)
        v = valid.reshape(-1)

        def put(store: Tensor, src: Tensor) -> None:
            flat = store.view(store.shape[0] * store.shape[1], *store.shape[2:])
            cur = flat[fi]
            m = v.reshape(-1, *([1] * (cur.dim() - 1)))
            flat[fi] = torch.where(m, src.reshape(cur.shape).to(cur.dtype), cur)

        put(self.key_codes, k_codes)
        put(self.key_scale, k_scale)
        put(self.key_zero, k_zero)
        put(self.val_codes, v_codes)
        put(self.val_scale, v_scale)
        put(self.val_zero, v_zero)
        put(self.slot_pos, pos)
        put(self.slot_wid, wid)
        put(self.slot_active, torch.ones_like(valid))

    def set_active(self, slot_idx: Tensor, valid: Tensor, value: bool) -> None:
        """Flip ``slot_active`` on the given slots where ``valid`` (§10).

        This is the whole of re-demotion (``value=True``) and promotion
        (``value=False``): codes, grid, and ``position_range`` are frozen, so a
        tier transition is a single bit. Invalid lanes keep their current bit.
        """
        fi = self._flat(slot_idx)
        flat = self.slot_active.view(-1)
        cur = flat[fi]
        flat[fi] = torch.where(
            valid.reshape(-1), torch.full_like(cur, value), cur
        )

    # -- read ----------------------------------------------------------------

    def gather(self, slot_idx: Tensor) -> Tuple[Tensor, ...]:
        """Gather ``[B, n]`` slots, flattened to a ``[B*n]`` leading axis.

        The flattened leading axis is what the batched quantizers already
        consume (they treat leading ``N`` as opaque and reduce only over the
        quant-group axis), so the (row, slot) pair rides through them untouched
        and the numerics are identical to the per-window singular form.
        """
        fi = self._flat(slot_idx)

        def take(store: Tensor) -> Tensor:
            flat = store.view(store.shape[0] * store.shape[1], *store.shape[2:])
            return flat[fi]

        return (
            take(self.key_codes), take(self.key_scale), take(self.key_zero),
            take(self.val_codes), take(self.val_scale), take(self.val_zero),
            take(self.slot_pos),
        )

    def active_order(self, n_active: int) -> Tensor:
        """Slot indices of each row's active windows, **ascending by window id**.

        ``[B, n_active]``. That order is the chronological interleave order the
        read path needs (design §5). Active window ids are unique per row, so the
        sort has no ties among the entries that survive the ``[:, :n_active]``
        slice — only the ``INT64_MAX`` padding ties, and it is discarded.
        """
        sentinel = torch.iinfo(torch.long).max
        key = torch.where(
            self.slot_active, self.slot_wid, torch.full_like(self.slot_wid, sentinel)
        )
        return torch.argsort(key, dim=1)[:, :n_active]

    def active_wids(self, n_active: int) -> Tensor:
        """``[B, n_active]`` active window ids, ascending. Diagnostics/tests."""
        return self.slot_wid.gather(1, self.active_order(n_active))

    # -- eviction bookkeeping ------------------------------------------------

    def retain_only(self, match: Tensor) -> None:
        """Free every slot whose window is not in the retained set (§6).

        ``match`` is :meth:`lookup`'s ``[B, W, N]`` output for the retained
        window ids, so ``match.any(1)`` — "does this slot's id appear anywhere in
        this row's retained list?" — is exactly the keep mask, already computed.

        Freed slots keep their stale codes; ``slot_wid = -1`` is what makes a
        slot free, and :meth:`write` overwrites every field on reuse.
        """
        keep = match.any(1)                                          # [B, N]
        self.slot_wid = torch.where(
            keep, self.slot_wid, torch.full_like(self.slot_wid, FREE)
        )
        self.slot_active = self.slot_active & keep

    # -- invariants (tests; each syncs) --------------------------------------

    def validate(self) -> None:
        """Assert the table's invariants. Test-only — every check syncs."""
        live = self.slot_wid != FREE
        assert bool((self.slot_active & ~live).sum() == 0), \
            "a free slot is marked active"
        # At most one slot per row per window id.
        wid = self.slot_wid
        same = (wid.unsqueeze(2) == wid.unsqueeze(1)) & live.unsqueeze(2) & live.unsqueeze(1)
        eye = torch.eye(wid.shape[1], dtype=torch.bool, device=wid.device)
        assert not bool((same & ~eye).any()), \
            "duplicate window id in one row's slot table"

    @property
    def n_live(self) -> Tensor:
        """``[B]`` live (active + dormant) entries per row. Diagnostics/tests."""
        return (self.slot_wid != FREE).sum(1)
