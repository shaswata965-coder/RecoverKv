"""QuantizedStore — the Q-tier facade over the dense slot table (design.md §4–§6).

One store per layer. It is the object the cache talks to for all Q-tier
operations, and every one of them carries a **row** axis: rows evict
divergently, so the Q tier's identity bookkeeping cannot live on the host
(BATCHING_PLAN.md §2, §4).

- :meth:`demote_many` — first-time demotion (quantize once) into free slots.
- :meth:`reactivate_many` — re-demotion of a dormant entry: one bit, no
  recompute, exactly zero added error (design §10).
- :meth:`promote_many` — hand back a window's frozen record, mark it dormant.
- :meth:`effective_q_tier` — dequantize + RoPE every **active** window in
  chronological order for the read path (design §5, §8).

Every method takes and returns tensors indexed by *slot*, never by window id:
the cache resolves ids to slots once per eviction through :meth:`lookup` and
drives the rest with masks. Nothing here loops or syncs.

Keys are stored **pre-RoPE**; RoPE is applied at read using each window's frozen
``position_range``. Values carry no RoPE (asymmetric store).

**Worst-case widths.** Callers size the promote / demote batches by the *bound*
(``N_q``, ``min(N_q, n_fp)``) rather than the actual per-row count, because the
counts are **ragged across rows** even though the retained counts are not
(BATCHING_PLAN.md §3 covers retained only): row 0 may re-demote 4 windows while
row 1 demotes none. Invalid lanes carry garbage through the quantizer and are
discarded by mask. This is safe *and* bit-identical because every quantizer op
is elementwise or reduces only over a window's own quant-group axis — lane ``j``
cannot influence lane ``k``. It costs work proportional to the bound, which is
the price of rectangularity; the alternative is a host sync per layer per
eviction to learn the true max.
"""

from __future__ import annotations

import math

from typing import Optional, Tuple

import torch
from torch import Tensor

from .compact import stable_partition
from .quantizer import (
    QGrid,
    dequantize_key_windows,
    dequantize_value_windows,
    quantize_key_windows,
    quantize_value_windows,
)
from .slots import QuantSlotTable


class QuantizedStore:
    """Q-tier storage + operations for one layer, over ``B`` rows.

    Parameters
    ----------
    window_size, head_dim, num_kv_heads : int
    n_slots : int
        Slots per row. Bounded by config — see :func:`modules.quant.slots.n_slots_for`.
    memoize_read : bool
        Cache :meth:`effective_q_tier`'s dequantized + RoPE'd result between
        evictions. See that method for the memory/bandwidth trade.
    """

    def __init__(
        self,
        window_size: int,
        head_dim: int,
        num_kv_heads: int,
        n_slots: int,
        memoize_read: bool = True,
        sketch_enabled: bool = False,
    ) -> None:
        self.window_size = window_size
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.n_slots = n_slots
        self.memoize_read = memoize_read

        # Rank-1 gate cards (modules/quant/sketch.py). Off by default: with it
        # off nothing is allocated and every path below is the pre-gate one.
        self.sketch_enabled = sketch_enabled
        # [B, H_kv, D] common mode, FROZEN at the first demotion batch. It must
        # be frozen, not running: a card is written once and never revisited
        # (§10), so a later anchor change would silently reinterpret every card
        # already on disk. Freezing makes the encoding as immutable as the codes.
        self._anchor: Optional[Tensor] = None
        # [B, H_kv, D] value-side common mode, frozen on the same batch and for
        # the same reason as `_anchor`.
        self._v_anchor: Optional[Tensor] = None

        # Allocated on first use: the row count and device are not known until
        # the first forward pass reaches the cache.
        self.table: Optional[QuantSlotTable] = None

        # Active windows per row, as a HOST int. It is the same for every row —
        # `compute_two_tier_retain` keeps exactly n_q = min(N_q, evictable_w -
        # k_fp) windows and both terms derive from config + W, which are shared
        # across rows (BATCHING_PLAN.md §3). Tracking it as an int (rather than
        # slot_active.sum(1)) keeps get_seq_length() — which HF calls per layer
        # per step and which must return an int — free of a device sync.
        # `validate()` asserts it against the table.
        self._n_active = 0

        # Bumped by every mutation; read paths memoize against it. Entries are
        # write-once (§10) and the active set only moves at eviction, so between
        # evictions a read is bit-identical.
        self._version = 0
        self._read_cache: Optional[Tuple] = None

    # -- lifecycle -----------------------------------------------------------

    def ensure(self, batch_size: int, device: torch.device) -> QuantSlotTable:
        """Allocate the slot table on first use; return it."""
        if self.table is None:
            self.table = QuantSlotTable(
                batch_size=batch_size,
                n_slots=self.n_slots,
                window_size=self.window_size,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                device=device,
                sketch=self.sketch_enabled,
            )
        elif self.table.batch_size != batch_size:
            raise ValueError(
                f"Q store was allocated for batch size {self.table.batch_size}, "
                f"got {batch_size}; a cache instance serves one batch shape."
            )
        return self.table

    @classmethod
    def join_layers(cls, stores: "list[QuantizedStore]") -> "QuantizedStore":
        """Fold ``L`` per-layer stores into one whose row axis is ``L*B``.

        The companion to :meth:`QuantSlotTable.join_layers` and
        :meth:`CacheState.join_layers`: layer ``i`` owns rows
        ``[i*B, (i+1)*B)`` of the joint table, so one batched eviction covers
        every layer (DECODE_SPEED_PLAN.md §4.1).

        ``_n_active`` is a host int that is the same for every row **and** every
        layer (it is ``n_q`` from the budget resolver, which every layer resolves
        identically), so the joint store inherits it rather than re-deriving it;
        a disagreement between layers means the per-layer evictions diverged and
        raises here rather than producing a store whose count lies about its own
        table.
        """
        if not stores:
            raise ValueError("join_layers needs at least one store")
        ref = stores[0]
        for i, s in enumerate(stores):
            if s._n_active != ref._n_active:
                raise RuntimeError(
                    f"join_layers: layer {i} has {s._n_active} active Q windows, "
                    f"layer 0 has {ref._n_active} — layers must stay in lockstep"
                )
            if (s.window_size, s.head_dim, s.num_kv_heads, s.n_slots) != (
                    ref.window_size, ref.head_dim, ref.num_kv_heads, ref.n_slots):
                raise RuntimeError(
                    f"join_layers: layer {i}'s store geometry differs from layer 0's"
                )
            if (s.table is None) != (ref.table is None):
                raise RuntimeError(
                    f"join_layers: layer {i} "
                    f"{'has' if s.table is not None else 'has no'} slot table but "
                    f"layer 0 {'has' if ref.table is not None else 'has none'} — "
                    "the table is allocated by the first eviction, which every "
                    "layer runs on the same step"
                )
        joint = cls(
            window_size=ref.window_size,
            head_dim=ref.head_dim,
            num_kv_heads=ref.num_kv_heads,
            n_slots=ref.n_slots,
            memoize_read=ref.memoize_read,
            sketch_enabled=ref.sketch_enabled,
        )
        # Anchors are per-row already, so joining is the same row-axis concat the
        # tables use: layer i owns rows [i*B, (i+1)*B).
        if ref.sketch_enabled and all(s._anchor is not None for s in stores):
            joint._anchor = torch.cat([s._anchor for s in stores], dim=0)
            joint._v_anchor = torch.cat([s._v_anchor for s in stores], dim=0)
        if ref.table is not None:
            joint.table = QuantSlotTable.join_layers([s.table for s in stores])
        joint._n_active = ref._n_active
        # A fresh store starts at version 0 with an empty read memo, which is
        # correct: nothing has read this object yet, so no memo can be stale.
        return joint

    @property
    def version(self) -> int:
        """Monotonic counter — changes iff the Q tier's contents changed."""
        return self._version

    def _invalidate(self) -> None:
        self._version += 1
        self._read_cache = None

    # -- counts --------------------------------------------------------------

    @property
    def num_active_windows(self) -> int:
        return self._n_active

    @property
    def num_active_tokens(self) -> int:
        return self._n_active * self.window_size

    def commit_active_count(self, n_active: int) -> None:
        """Record the post-eviction active-window count (same for every row).

        The cache knows this as a plain int before it touches a tensor — it is
        ``n_q`` from the budget resolver — so the store never has to reduce
        ``slot_active`` and sync to learn its own size.
        """
        self._n_active = n_active

    def active_ids(self) -> Optional[Tensor]:
        """``[B, n_active]`` active window ids, ascending per row. Diagnostics."""
        if self.table is None:
            return None
        return self.table.active_wids(self._n_active)

    # -- lookup / bookkeeping ------------------------------------------------

    def lookup(self, wids: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Resolve ``[B, W]`` window ids to slots. See :meth:`QuantSlotTable.lookup`."""
        return self.table.lookup(wids)

    def retain_only(self, match: Tensor) -> None:
        """Free slots for windows dropped outright (§6). ``match`` from :meth:`lookup`."""
        self._invalidate()
        self.table.retain_only(match)

    def reactivate_many(self, slot_idx: Tensor, valid: Tensor) -> None:
        """Re-demote dormant entries: dormant → active, no recompute (§10)."""
        self._invalidate()
        self.table.set_active(slot_idx, valid, True)

    # -- demotion ------------------------------------------------------------

    def demote_many(
        self,
        slot_idx: Tensor,
        valid: Tensor,
        wid: Tensor,
        keys_pre_rope: Tensor,
        values: Tensor,
        position_ranges: Tensor,
        keys_post_rope: Optional[Tensor] = None,
    ) -> None:
        """First-time demotion of up to ``n`` windows per row, in one quantize.

        Every ``valid`` lane must be a window with **no** existing entry —
        callers route re-demotions through :meth:`reactivate_many`, which never
        re-quantizes (§10). Invalid lanes are quantized too (they ride the same
        dense tensor) and then discarded by :meth:`QuantSlotTable.write`.

        Parameters
        ----------
        slot_idx : ``[B, n]`` target slots, from :meth:`QuantSlotTable.free_slots`.
        valid : ``[B, n]`` bool — real demotions.
        wid : ``[B, n]`` int64 window ids (``-1`` on invalid lanes).
        keys_pre_rope, values : ``[B, n, H_kv, window, D]``.
        position_ranges : ``[B, n, window]`` int64 original absolute positions.
        keys_post_rope : ``[B, n, H_kv, window, D]``, required when
            :attr:`sketch_enabled`. The gate card is built from the keys **as
            they are here**, i.e. still rotated, which is why it costs no extra
            RoPE: the caller already holds them (it is about to un-rotate them to
            get ``keys_pre_rope``). The card is then frozen for life, because
            eviction never rebases positions.
        """
        self._invalidate()
        B, n = slot_idx.shape
        H, S, D = self.num_kv_heads, self.window_size, self.head_dim

        # Flatten (row, lane) into the quantizers' leading window axis. They
        # treat it as opaque and reduce only over the quant group, so this is the
        # singular op applied slice-wise — bit-identical, one launch per tier.
        k_flat = keys_pre_rope.reshape(B * n, H, S, D)
        v_flat = values.reshape(B * n, H, S, D)
        k_codes, k_scale, k_zero = quantize_key_windows(k_flat)
        v_codes, v_scale, v_zero = quantize_value_windows(v_flat)

        sketch = None
        if self.sketch_enabled:
            if keys_post_rope is None:
                raise ValueError(
                    "sketch_enabled but demote_many got no keys_post_rope; the "
                    "card must be built from the rotated keys (see §5, §10)."
                )
            from .sketch import build_sketch
            kp = keys_post_rope.reshape(B * n, H, S, D).to(torch.float32)
            vp = values.reshape(B * n, H, S, D).to(torch.float32)
            if self._anchor is None:
                # Frozen here, from the first batch of demoted windows: the mean
                # over (window, token) of this layer's keys, per row and head.
                # Frozen and not running, because a card is written once and never
                # revisited (§10) -- a later anchor change would silently
                # reinterpret every card already stored.
                self._anchor = kp.reshape(B, n, H, S, D).mean(dim=(1, 3))
                self._v_anchor = vp.reshape(B, n, H, S, D).mean(dim=(1, 3))
            anc = self._anchor.repeat_interleave(n, dim=0)          # [B*n, H, D]
            vanc = self._v_anchor.repeat_interleave(n, dim=0)
            sketch = tuple(build_sketch(kp, anc, vp, vanc))

        self.table.write(
            slot_idx, valid, wid,
            k_codes, k_scale, k_zero,
            v_codes, v_scale, v_zero,
            position_ranges.to(torch.long),
            sketch=sketch,
        )

    # -- promotion -----------------------------------------------------------

    def promote_many(
        self, slot_idx: Tensor, valid: Tensor, out_dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Promote up to ``n`` windows per row out of the Q tier, in one dequantize.

        Marks every ``valid`` entry dormant — **retained**, not freed, so a later
        re-demotion is a pure reactivation (§10) — and returns the frozen record.
        The caller applies RoPE at the returned positions and splices each window
        into the fp store at its chronological slot (§5).

        Returns
        -------
        keys_pre_rope : ``[B, n, H_kv, window, D]``
        values : ``[B, n, H_kv, window, D]``
        position_ranges : ``[B, n, window]`` int64
        """
        self._invalidate()
        B, n = slot_idx.shape
        H, S, D = self.num_kv_heads, self.window_size, self.head_dim

        k_codes, k_scale, k_zero, v_codes, v_scale, v_zero, pos = self.table.gather(slot_idx)
        keys = dequantize_key_windows(
            k_codes, k_scale, k_zero, self.window_size, out_dtype=out_dtype
        )
        values = dequantize_value_windows(
            v_codes, v_scale, v_zero, self.head_dim, out_dtype=out_dtype
        )
        self.table.set_active(slot_idx, valid, False)
        return (
            keys.reshape(B, n, H, S, D),
            values.reshape(B, n, H, S, D),
            pos.reshape(B, n, S),
        )

    # -- read-path gather ----------------------------------------------------

    def gate_and_select(
        self,
        query: Tensor,
        scaling: float,
        ratio: float,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Which active Q windows this step dequantizes, and what the rest score.

        Parameters
        ----------
        query : ``[B, H_q, D]`` post-RoPE decode query.
        scaling : attention ``1 / sqrt(head_dim)``.
        ratio : fraction of this step's ACTIVE windows to dequantize. Resolved
            against the live ``n_active`` rather than a fixed count, because
            ``N_q`` moves with the shape and a pinned integer would not be the
            same fraction anywhere. At least one window always survives.

        Returns
        -------
        ``(keep, logmass, slots)`` or ``None`` for an empty tier.
        keep : ``[B, H_kv, n_active]`` bool -- the union over each GQA group, which
            is what the kernel needs because one program owns a whole group.
        logmass : ``[B, H_q, n_active]`` -- the estimated log mass of EVERY window,
            selected or not. The skipped ones are what the caller credits back,
            both to ``window_scores`` and to the attention output; without that a
            skipped window scores zero, ranks last, and gets evicted -- which
            would silently destroy the tier this gate exists to read less often.
        slots : ``[B, n_active]`` the slot index behind each column.
        """
        if self.table is None or self._n_active == 0:
            return None
        if not self.sketch_enabled or self._anchor is None:
            raise RuntimeError(
                "gate_and_select needs sketch cards; construct the store with "
                "sketch_enabled=True and demote at least once."
            )
        from .sketch import Sketch, gate_and_score, group_max, select_windows

        slots = self.table.active_order(self._n_active)
        card = Sketch(*self.table.gather_sketch(slots))
        logmass, est = gate_and_score(query, card, self._anchor, scaling)
        # Union the GQA group FIRST, then select. The KV head is the unit of
        # work -- one program loads a window once for every query head sharing
        # it -- so the cap has to bind there. Capping per query head and unioning
        # afterwards would let ratio r read up to r*rep windows (see group_max).
        keep = select_windows(
            group_max(est, self.num_kv_heads), self.cap_for_ratio(ratio))
        return keep, logmass, slots

    def cap_for_ratio(self, ratio: float) -> int:
        """Windows read per head per step at ``ratio``, against the live count."""
        return max(1, min(self._n_active, math.ceil(ratio * self._n_active)))

    def window_centroids(self, slots: Tensor) -> Tensor:
        """``[B, n_active, H_kv, D]`` — each active window's representative value.

        What a skipped window contributes to the attention output. Gathered from
        the same card the scan already pulled, so it rides the cache line the
        scan warmed rather than costing a second pass over the tier.
        """
        from .sketch import Sketch, value_centroid

        card = Sketch(*self.table.gather_sketch(slots))
        return value_centroid(card, self._v_anchor)

    def gated_q_tier(
        self,
        keep: Tensor,
        slots: Tensor,
        rope_module: torch.nn.Module,
        out_dtype: torch.dtype,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Read-ready Q tier for the SELECTED windows only — the gated read path.

        The counterpart of :meth:`effective_q_tier`, which dequantizes the whole
        tier. This one dequantizes ``n_sel`` windows per ``(row, KV head)``, which
        is the point of the gate.

        Selection is per **KV head**, not per row, and that is load-bearing.
        Measured on a shape-C tier (271 windows, 8 KV heads, 32 query heads) at a
        0.25 ratio:

            per (row, KV head)      68/271 windows read   100.00% mass recall
            per row, union of heads 240/271               100.00%
            per row, top-k row-max   68/271                97.54%

        The union reads 89% of the tier — the gate buys nothing — and a row-level
        top-k pays 2.5% of the worst head's mass for the same traffic. So each KV
        head picks its own windows, and the cost is that positions become
        per-head: ``[B, H_kv, n_sel*ws]`` rather than the shared ``[B, n*ws]``.

        Counts are equal across rows and heads (the cap is a fixed
        ``ceil(ratio * n_active)``), so the result is still dense and needs no
        padding or keep-mask — the same rectangularity argument the tier split
        already relies on (BATCHING_PLAN.md §3).

        Parameters
        ----------
        keep : ``[B, H_kv, n_active]`` bool, equal row sums, from
            :meth:`gate_and_select`.
        slots : ``[B, n_active]`` the slot behind each column, same call.

        Returns
        -------
        ``(keys, values, positions)`` with keys/values ``[B, H_kv, n_sel*ws, D]``
        and positions ``[B, H_kv, n_sel*ws]``, or ``None`` for an empty tier.
        """
        if self.table is None or self._n_active == 0:
            return None
        from .effective import dequant_rotate_q_keys
        from .quantizer import dequantize_value_windows

        B, H, S, D = keep.shape[0], self.num_kv_heads, self.window_size, self.head_dim
        n_sel = int(keep[0, 0].sum())
        # Rank within each head: the selected columns come first, so a
        # fixed-width slice takes exactly them. The partition is stable, so
        # column order stays ascending-by-id within the selection — which is
        # what compute_score_meta_gated relies on.
        pick = stable_partition(keep)[..., :n_sel]                      # [B,H,n_sel]
        sel = torch.gather(slots.unsqueeze(1).expand(B, H, slots.shape[1]), 2, pick)

        t = self.table
        rows = torch.arange(B, device=sel.device)[:, None, None]
        heads = torch.arange(H, device=sel.device)[None, :, None]

        def take(store_t, head_axis=True):
            return (store_t[rows, sel, heads] if head_axis
                    else store_t[rows, sel])

        def take_grid(q_name, s_name):
            return QGrid(take(getattr(t, q_name)), take(getattr(t, s_name)))

        k_codes = take(t.key_codes)                    # [B,H,n_sel,D,S//4]
        k_scale = take_grid("key_scale_q", "key_scale_s")
        k_zero = take_grid("key_zero_q", "key_zero_s")
        v_codes = take(t.val_codes)                    # [B,H,n_sel,S,D//4]
        v_scale = take_grid("val_scale_q", "val_scale_s")
        v_zero = take_grid("val_zero_q", "val_zero_s")
        pos = take(t.slot_pos, head_axis=False)        # [B,H,n_sel,S]

        # Fold the head axis into the batch axis: the read kernel treats its
        # leading axis as opaque, so a per-head selection costs no new code path,
        # only a reshape. cos/sin then come from this row's own positions.
        BH = B * H
        pos_flat = pos.reshape(BH, n_sel * S)
        keys = dequant_rotate_q_keys(
            k_codes.reshape(BH * n_sel, 1, D, S // 4),
            k_scale.reshape(BH * n_sel, 1, D), k_zero.reshape(BH * n_sel, 1, D),
            S, pos_flat, rope_module, out_dtype, BH, n_sel, 1, D,
        ).reshape(B, H, n_sel * S, D)

        values = dequantize_value_windows(
            v_codes.reshape(BH * n_sel, 1, S, D // 4),
            v_scale.reshape(BH * n_sel, 1, S), v_zero.reshape(BH * n_sel, 1, S),
            D, out_dtype=out_dtype,
        ).reshape(B, H, n_sel * S, D).to(out_dtype)

        return keys, values, pos.reshape(B, H, n_sel * S)

    def effective_q_tier(
        self, rope_module: torch.nn.Module, out_dtype: torch.dtype
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Read-ready Q tier: post-RoPE keys, values, and their positions.

        Returns ``None`` for an empty tier, else a triple flattened window-major
        over each row's active windows (ascending id):

        - ``keys``     : ``[B, H_kv, N_q * window, D]`` — dequantized, RoPE'd at
          each window's frozen ``position_range``.
        - ``values``   : ``[B, H_kv, N_q * window, D]``.
        - ``positions``: ``[B, N_q * window]`` int64 original absolute positions.

        **Memoized on** :attr:`version` when :attr:`memoize_read` is set. Codes
        and grids are written once at first demotion and ``position_range`` is
        never rebased (§10), so the dequant + RoPE are bit-identical until the
        ledger next changes — and it only changes at eviction. At
        ``window_size = 8`` that makes 7 of every 8 decode steps a cache hit on
        the *large* tier (int2 holds the majority of retained windows).

        The memo is not free: it holds the whole tier **dequantized to fp16**,
        which is ~8x the int2 codes it was computed from, per layer, for the
        whole decode. At B = 1 that is invisible against the weights and the
        method is weight-bound anyway (BATCHING_PLAN.md §5), so it is pure win.
        At B > 1 it is charged per row and competes directly with batch capacity
        — which is the entire thesis — so the cache defaults it **off** there.
        See ``WindowedCacheConfig.quant_memoize_read``.
        """
        if self.table is None or self._n_active == 0:
            return None

        key = (self._version, out_dtype)
        if (
            self.memoize_read
            and self._read_cache is not None
            and self._read_cache[0] == key
        ):
            return self._read_cache[1]

        # Local import: `effective` owns the RoPE primitives and imports nothing
        # from this module, so this cannot cycle.
        from .effective import dequant_rotate_q_keys

        idx = self.table.active_order(self._n_active)          # [B, n_q] slots
        B, n = idx.shape
        H, S, D = self.num_kv_heads, self.window_size, self.head_dim

        k_codes, k_scale, k_zero, v_codes, v_scale, v_zero, pos = self.table.gather(idx)

        # Keys: dequant + RoPE fused into one (optionally compiled) kernel — the
        # ~20-launch elementwise chain that dominates the per-step read path
        # (design §8). Window-major -> token-major and the per-token RoPE happen
        # inside; bit-identical to the old dequant-then-rotate_key_window pair.
        pos_flat = pos.reshape(B, n * S)
        keys = dequant_rotate_q_keys(
            k_codes, k_scale, k_zero, self.window_size, pos_flat,
            rope_module, out_dtype, B, n, H, D,
        )

        # Values carry no RoPE (asymmetric store) — just dequant, then
        # window-major -> token-major: [B, n, H, S, D] -> [B, H, n*S, D].
        values = dequantize_value_windows(
            v_codes, v_scale, v_zero, self.head_dim, out_dtype=out_dtype
        )
        v_flat = values.reshape(B, n, H, S, D).permute(0, 2, 1, 3, 4).reshape(B, H, n * S, D)
        vals = v_flat.to(out_dtype)

        result = (keys, vals, pos_flat)
        if self.memoize_read:
            self._read_cache = (key, result)
        return result

    # -- invariants (tests) --------------------------------------------------

    def validate(self) -> None:
        """Assert the store's invariants. Test-only — syncs."""
        if self.table is None:
            assert self._n_active == 0
            return
        self.table.validate()
        counts = self.table.slot_active.sum(1)
        assert bool((counts == self._n_active).all()), (
            f"commit_active_count({self._n_active}) disagrees with the table "
            f"({counts.tolist()}) — the retained Q count must be identical "
            f"across rows (BATCHING_PLAN.md §3)"
        )
