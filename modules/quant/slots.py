"""Dense slot table for the Q tier (design.md §4, §6; BATCHING_PLAN.md §4).

The batch-carrying replacement for the per-window ``Dict[int, LedgerEntry]``
ledger. Same contract — §6's frozen record per Q window — but as tensors with a
leading **row** axis, because rows evict divergently: row 0 may hold windows
``{1, 5}`` in int2 while row 1 holds ``{4, 11}``. A dict keyed by window id
cannot express that; a slot table can.

Layout (one layer, ``B`` rows, ``N`` slots)::

    key_codes   [B, N, H_kv, D, ws//4]  uint8   channel-major, 4 tokens/byte
    key_scale_q [B, N, H_kv, D]         uint8   pinned grid, one byte per entry
    key_scale_s [B, N, H_kv, D//g]      fp16    the scale each group shares
    key_zero_q  [B, N, H_kv, D]         int8
    key_zero_s  [B, N, H_kv, D//g]      fp16
    val_codes   [B, N, H_kv, ws, D//4]  uint8   token-major, 4 channels/byte
    val_scale_q [B, N, H_kv, ws]        uint8
    val_scale_s [B, N, H_kv, ws//g]     fp16
    val_zero_q  [B, N, H_kv, ws]        int8
    val_zero_s  [B, N, H_kv, ws//g]     fp16
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

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from .compact import stable_partition
from .quantizer import QGrid, grid_group

FREE = -1
"""``slot_wid`` sentinel for an unoccupied slot. Real window ids are >= 0."""

GRID_FIELDS = (
    "key_scale_q", "key_scale_s", "key_zero_q", "key_zero_s",
    "val_scale_q", "val_scale_s", "val_zero_q", "val_zero_s",
)
"""The two :class:`~modules.quant.quantizer.QGrid` pairs, as flat columns.

A ``QGrid`` is codes plus the fp16 scale a group of them shares, so each of the
four grid fields is two tensors here. They are named rather than nested so every
name-driven path in this file — ``join_layers``, ``write``, ``gather`` — keeps
working on a list of strings."""

SKETCH_FIELDS = (
    "sk_mu_q", "sk_mu_s", "sk_v_q", "sk_v_s",
    "sk_t_q", "sk_t_s", "sk_vm_q", "sk_vm_s",
)
"""The rank-1 gate card's columns, in :class:`modules.quant.sketch.Sketch` order.

They ride the slot table rather than living beside it so they inherit its whole
lifecycle for free: ``write`` freezes a card at first demotion, ``retain_only``
drops it with its window, ``set_active`` reactivates it on re-demotion without
recomputation, and ``join_layers`` folds it layer-major. None of those needed a
code change -- only this name list."""


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
        sketch: bool = False,
    ) -> None:
        B, N, H, D, S = batch_size, n_slots, num_kv_heads, head_dim, window_size
        self.batch_size = B
        self.n_slots = N
        self.window_size = S
        self.head_dim = D
        self.num_kv_heads = H

        self.key_codes = torch.zeros((B, N, H, D, S // 4), dtype=torch.uint8, device=device)
        # The grid is one byte per entry against an fp16 scale shared by
        # `grid_group` of them (QGrid). uint8 for `scale` (non-negative, and a
        # zero would divide by zero at fit time), int8 for `zero` (an offset).
        gk, gv = D // grid_group(D), S // grid_group(S)
        self.key_scale_q = torch.zeros((B, N, H, D), dtype=torch.uint8, device=device)
        self.key_scale_s = torch.zeros((B, N, H, gk), dtype=torch.float16, device=device)
        self.key_zero_q = torch.zeros((B, N, H, D), dtype=torch.int8, device=device)
        self.key_zero_s = torch.zeros((B, N, H, gk), dtype=torch.float16, device=device)
        self.val_codes = torch.zeros((B, N, H, S, D // 4), dtype=torch.uint8, device=device)
        self.val_scale_q = torch.zeros((B, N, H, S), dtype=torch.uint8, device=device)
        self.val_scale_s = torch.zeros((B, N, H, gv), dtype=torch.float16, device=device)
        self.val_zero_q = torch.zeros((B, N, H, S), dtype=torch.int8, device=device)
        self.val_zero_s = torch.zeros((B, N, H, gv), dtype=torch.float16, device=device)
        self.slot_wid = torch.full((B, N), FREE, dtype=torch.long, device=device)
        self.slot_active = torch.zeros((B, N), dtype=torch.bool, device=device)
        self.slot_pos = torch.zeros((B, N, S), dtype=torch.long, device=device)

        # Rank-1 gate cards (modules/quant/sketch.py). Allocated only when the
        # gate is on, so a q>0 run with the gate off is byte-identical to before.
        #
        # Slot-major with D innermost, mirroring `key_scale_q [B, N, H, D]`: that
        # layout already gives 256 contiguous bytes per (slot, head) at D=128, so
        # a gathered window is a contiguous run rather than a scalar gather, and
        # every existing index path (`_flat`, `write`, `gather`, `retain_only`)
        # works on it unchanged. A head-major layout would coalesce no better --
        # the inner dim is what matters -- and would need a `slots.py` refactor.
        self.sketch = sketch
        if sketch:
            self.sk_mu_q = torch.zeros((B, N, H, D), dtype=torch.int8, device=device)
            self.sk_mu_s = torch.zeros((B, N, H), dtype=torch.float16, device=device)
            self.sk_v_q = torch.zeros((B, N, H, D), dtype=torch.int8, device=device)
            self.sk_v_s = torch.zeros((B, N, H), dtype=torch.float16, device=device)
            self.sk_t_q = torch.zeros((B, N, H, S), dtype=torch.int8, device=device)
            self.sk_t_s = torch.zeros((B, N, H), dtype=torch.float16, device=device)
            # Value-side centroid. This is what makes a SKIPPED window contribute
            # to the attention output instead of vanishing from it; its weight is
            # the card's mass estimate, recalibrated every step against the
            # windows that were actually read. It replaces the `eps` residual
            # field, which served a bound the fused gate never implemented.
            self.sk_vm_q = torch.zeros((B, N, H, D), dtype=torch.int8, device=device)
            self.sk_vm_s = torch.zeros((B, N, H), dtype=torch.float16, device=device)

        # Row offsets for flat indexing. Scattering with a broadcast [B, n, H, D,
        # ws//2] index tensor would allocate an int64 index the size of the codes
        # themselves (GBs at max B); flattening (B, N) -> B*N and indexing dim 0
        # with a [B*n] vector costs kilobytes instead.
        self._row_base = (torch.arange(B, device=device) * N).unsqueeze(1)  # [B, 1]

    # -- layer-major join (DECODE_SPEED_PLAN.md §4.1) -------------------------

    @classmethod
    def join_layers(cls, tables: "list[QuantSlotTable]") -> "QuantSlotTable":
        """Fold ``L`` per-layer tables into one whose row axis is ``L*B``.

        The row axis is already opaque to every method here — ``lookup`` matches
        ``[B, W, N]``, ``free_slots`` argsorts along the slot axis, ``write`` and
        ``gather`` flat-index through :attr:`_row_base` — so widening it from
        ``B`` to ``L*B`` needs no code change at all, only the concatenated
        tensors and a rebuilt ``_row_base``. Layer ``i`` owns rows
        ``[i*B, (i+1)*B)``, matching :meth:`CacheState.join_layers`.

        Every table must share ``n_slots``, ``window_size``, ``head_dim`` and
        ``num_kv_heads`` — all config-derived, hence identical across layers, and
        checked here so a mismatch raises instead of concatenating into a table
        whose rows mean different things.
        """
        if not tables:
            raise ValueError("join_layers needs at least one table")
        ref = tables[0]
        for i, t in enumerate(tables):
            if (t.n_slots, t.window_size, t.head_dim, t.num_kv_heads,
                    t.batch_size) != (ref.n_slots, ref.window_size, ref.head_dim,
                                      ref.num_kv_heads, ref.batch_size):
                raise RuntimeError(
                    f"join_layers: layer {i}'s slot table geometry differs from "
                    "layer 0's; every layer resolves the same config, so this "
                    "means the tables were built against different settings"
                )
        joint = cls.__new__(cls)
        joint.batch_size = ref.batch_size * len(tables)
        joint.n_slots = ref.n_slots
        joint.window_size = ref.window_size
        joint.head_dim = ref.head_dim
        joint.num_kv_heads = ref.num_kv_heads
        joint.sketch = ref.sketch
        fields = ["key_codes", *GRID_FIELDS, "val_codes",
                  "slot_wid", "slot_active", "slot_pos"]
        if ref.sketch:
            fields += list(SKETCH_FIELDS)
        for field in fields:
            setattr(joint, field, torch.cat(
                [getattr(t, field) for t in tables], dim=0
            ).contiguous())
        joint._row_base = (
            torch.arange(joint.batch_size, device=ref.slot_wid.device)
            * joint.n_slots
        ).unsqueeze(1)
        return joint

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
        keep : ``[B, N]`` bool — "some looked-up id lives in this slot", which is
            exactly what :meth:`retain_only` needs.

        Free slots hold ``-1`` and real ids are >= 0, so a free slot never
        matches. At most one slot per row can carry a given id (the table's
        invariant), so ``argmax`` picks that slot outright.

        **Three passes over the [B, W, N] match, not six.** That tensor is the
        largest transient in the eviction (167 MB at B·L=1024, W=512, N=318), and
        this used to walk it once to build it, once for ``any(-1)``, once for an
        ``&`` against ``slot_active`` (materialising a second one), once more to
        reduce that, once for the ``uint8`` copy and once for ``argmax`` — then
        hand it out so ``retain_only`` could walk it a seventh time. Every one of
        those but the build and the ``max`` answers a question that the [B, W]
        results already answer:

        * ``max(-1)`` returns the maximum AND its index, so ``has_entry`` and
          ``slot_of`` come out of one pass instead of two plus a separate
          ``argmax``.
        * a matched id lives in exactly one slot, so "is that entry active" is
          ``slot_active`` read at ``slot_of`` — a ``[B, W]`` gather, not a
          ``[B, W, N]`` conjunction.
        * "does some retained id live in slot j" is that same map inverted, so it
          is a ``[B, W]`` scatter. Lanes with no entry are routed to a dump
          column rather than writing ``False`` at slot 0, which would erase a
          real hit there (they alias, because ``argmax`` returns 0 for a row that
          matched nothing).
        """
        match = self.slot_wid.unsqueeze(1) == wids.unsqueeze(2)      # [B, W, N]
        hit, slot_of = match.to(torch.uint8).max(-1)
        has_entry = hit.bool()
        is_active = has_entry & self.slot_active.gather(1, slot_of)
        N = self.slot_wid.shape[1]
        ext = torch.zeros((wids.shape[0], N + 1), dtype=torch.bool,
                          device=wids.device)
        # `N` and `True` as SCALARS: the tensor forms of both (`full_like`,
        # `ones_like`) are allocations the overloads do not need.
        ext.scatter_(1, torch.where(has_entry, slot_of, N), True)
        return has_entry, is_active, slot_of, ext[:, :N]

    # -- allocation ----------------------------------------------------------

    def free_slots(self, n: int) -> Tensor:
        """The ``n`` lowest-indexed free slots per row: ``[B, n]`` int64.

        Callers pair this with a *validity* mask and must write through
        :meth:`write`, which no-ops on invalid lanes. Only lanes
        ``j < n_free[row]`` are guaranteed free; the bound in
        :func:`n_slots_for` guarantees every **valid** lane is.
        """
        # Free slots first, in slot order -- a partition, not a sort
        # (`compact.stable_partition` says why that distinction is worth code).
        return stable_partition(self.slot_wid == FREE)[:, :n]

    def write(
        self,
        slot_idx: Tensor,
        valid: Tensor,
        wid: Tensor,
        k_codes: Tensor,
        k_scale: QGrid,
        k_zero: QGrid,
        v_codes: Tensor,
        v_scale: QGrid,
        v_zero: QGrid,
        pos: Tensor,
        sketch: Optional[Sequence[Tensor]] = None,
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
        k_codes, v_codes : ``[B, n, ...]`` packed int2 codes.
        k_scale .. v_zero : the four :class:`QGrid` fields, ``[B, n, ...]``.
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
        put(self.val_codes, v_codes)
        for name, src_t in zip(GRID_FIELDS, (*k_scale, *k_zero, *v_scale, *v_zero)):
            put(getattr(self, name), src_t)
        put(self.slot_pos, pos)
        if sketch is not None:
            if not self.sketch:
                raise RuntimeError(
                    "write() was given sketch fields but the table was built "
                    "without them; pass sketch=True to QuantSlotTable."
                )
            for name, src in zip(SKETCH_FIELDS, sketch):
                put(getattr(self, name), src)
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

    def gather(self, slot_idx: Tensor) -> Tuple:
        """Gather ``[B, n]`` slots, flattened to a ``[B*n]`` leading axis.

        Returns ``(k_codes, k_scale, k_zero, v_codes, v_scale, v_zero, pos)``,
        the grids as :class:`QGrid` pairs — the same seven values the quantizers
        take, so the caller never reassembles a grid by hand.

        The flattened leading axis is what the batched quantizers already
        consume (they treat leading ``N`` as opaque and reduce only over the
        quant-group axis), so the (row, slot) pair rides through them untouched
        and the numerics are identical to the per-window singular form.
        """
        fi = self._flat(slot_idx)

        def take(store: Tensor) -> Tensor:
            flat = store.view(store.shape[0] * store.shape[1], *store.shape[2:])
            return flat[fi]

        g = [take(getattr(self, name)) for name in GRID_FIELDS]
        return (
            take(self.key_codes), QGrid(g[0], g[1]), QGrid(g[2], g[3]),
            take(self.val_codes), QGrid(g[4], g[5]), QGrid(g[6], g[7]),
            take(self.slot_pos),
        )

    def gather_sketch(self, slot_idx: Tensor) -> Tuple[Tensor, ...]:
        """The eight card fields for ``[B, n]`` slots, keeping the ``[B, n]``
        leading pair (unlike :meth:`gather`, which flattens it): the gate is a
        per-(row, window) reduction, not a per-window quantizer op."""
        if not self.sketch:
            raise RuntimeError("this slot table carries no sketch fields")
        fi = self._flat(slot_idx)
        B, n = slot_idx.shape
        out = []
        for name in SKETCH_FIELDS:
            st = getattr(self, name)
            flat = st.view(st.shape[0] * st.shape[1], *st.shape[2:])
            out.append(flat[fi].reshape(B, n, *st.shape[2:]))
        return tuple(out)

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

    def retain_only(self, keep: Tensor) -> None:
        """Free every slot whose window is not in the retained set (§6).

        ``keep`` is :meth:`lookup`'s fourth output for the retained window ids:
        ``[B, N]``, "does this slot's id appear anywhere in this row's retained
        list?". It is built there because that is where the map from id to slot
        already exists, and building it there costs a ``[B, W]`` scatter instead
        of a reduction over the ``[B, W, N]`` match.

        Freed slots keep their stale codes; ``slot_wid = -1`` is what makes a
        slot free, and :meth:`write` overwrites every field on reuse.
        """
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
