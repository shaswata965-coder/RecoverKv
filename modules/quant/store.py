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

from typing import Optional, Tuple

import torch
from torch import Tensor

from .quantizer import (
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
    ) -> None:
        self.window_size = window_size
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.n_slots = n_slots
        self.memoize_read = memoize_read

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
            )
        elif self.table.batch_size != batch_size:
            raise ValueError(
                f"Q store was allocated for batch size {self.table.batch_size}, "
                f"got {batch_size}; a cache instance serves one batch shape."
            )
        return self.table

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

        self.table.write(
            slot_idx, valid, wid,
            k_codes, k_scale, k_zero,
            v_codes, v_scale, v_zero,
            position_ranges.to(torch.long),
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
