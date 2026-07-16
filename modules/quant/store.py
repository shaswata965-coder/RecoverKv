"""QuantizedStore — the Q-tier facade over the per-window ledger (design.md §4–§6).

One store per layer (B = 1 in v1). It is the object the cache talks to for all
Q-tier operations:

- :meth:`demote` — first-time demotion (quantize once) **or** reactivation of a
  dormant entry (re-demotion; no recompute — design §10).
- :meth:`promote` — hand back a window's frozen record and mark it dormant.
- :meth:`gather_active` — dequantize every **active** window (pre-RoPE keys +
  values) in chronological order for the read path (design §5, §8). The dense
  gap-free "Q store" of §4 is materialized here on demand from the active
  ledger entries; the physical byte ``offset`` is a Phase-2 layout concern.

Keys are stored **pre-RoPE**; RoPE is applied at read by
:func:`modules.quant.effective.materialize_effective_kv` using each window's
frozen ``position_range``. Values carry no RoPE (asymmetric store).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from .ledger import LedgerEntry, QuantLedger
from .quantizer import (
    dequantize_key_window,
    dequantize_key_windows,
    dequantize_value_window,
    dequantize_value_windows,
    quantize_key_window,
    quantize_key_windows,
    quantize_value_window,
    quantize_value_windows,
)


class QuantizedStore:
    """Q-tier storage + operations for one layer (B = 1)."""

    def __init__(self, window_size: int, head_dim: int, num_kv_heads: int) -> None:
        self.window_size = window_size
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.ledger = QuantLedger()
        # Bumped by every mutation; read paths memoize against it (see
        # `version`). Entries are write-once (§10) and the active set only moves
        # at eviction, so between evictions a read is bit-identical.
        self._version = 0
        self._read_cache: Optional[Tuple[Any, Any]] = None

    @property
    def version(self) -> int:
        """Monotonic counter — changes iff the Q tier's contents changed.

        Every mutator (:meth:`demote`, :meth:`reactivate`, :meth:`promote`,
        :meth:`retain_only`) bumps it, and all four are only reachable from the
        cache's eviction pass. A read path can therefore cache against this and
        trust the result until the next eviction.
        """
        return self._version

    def _invalidate(self) -> None:
        self._version += 1
        self._read_cache = None

    # -- counts --------------------------------------------------------------

    @property
    def num_active_windows(self) -> int:
        return self.ledger.num_active

    @property
    def num_active_tokens(self) -> int:
        return self.ledger.num_active * self.window_size

    def active_ids(self) -> List[int]:
        return self.ledger.active_ids()

    # -- demotion ------------------------------------------------------------

    def demote(
        self,
        window_id: int,
        key_pre_rope: Tensor,
        value: Tensor,
        position_range: Tensor,
    ) -> None:
        """Demote one window into the Q tier.

        First time for this ``window_id``: quantize keys (pre-RoPE) and values
        once against a freshly-pinned grid and register an active ledger entry.
        If a **dormant** entry already exists (a prior promotion), this is a
        re-demotion — reactivate it and **ignore the passed tensors** (the codes
        and grid are frozen; re-quantizing could flip boundary codes — §10).

        Parameters
        ----------
        window_id : int
            The window's ``original_window_id``.
        key_pre_rope : Tensor
            ``[H_kv, window, D]`` — pre-RoPE keys (already un-rotated by the
            caller for a first demotion; unused on reactivation).
        value : Tensor
            ``[H_kv, window, D]`` values (unused on reactivation).
        position_range : Tensor
            ``[window]`` int64 original absolute positions (unused on
            reactivation — the frozen entry already carries them).
        """
        self._invalidate()
        if self.ledger.contains(window_id):
            # Dormant entry exists → pure reactivation, no recompute.
            self.ledger.reactivate(window_id)
            return

        self._quantize_and_register(window_id, key_pre_rope, value, position_range)

    def reactivate(self, window_id: int) -> None:
        """Re-demote a window that already has a (dormant) ledger entry (§10).

        Pure reactivation — no dequant, no re-quantize, zero added error. Fails
        if the window has no entry (a first demotion must go through
        :meth:`demote`).
        """
        self._invalidate()
        self.ledger.reactivate(window_id)

    def has_entry(self, window_id: int) -> bool:
        """True if the window has a ledger entry (active or dormant)."""
        return self.ledger.contains(window_id)

    def _quantize_and_register(
        self, window_id: int, key_pre_rope: Tensor, value: Tensor, position_range: Tensor
    ) -> None:
        k_codes, k_scale, k_zero = quantize_key_window(key_pre_rope)
        v_codes, v_scale, v_zero = quantize_value_window(value)
        entry = LedgerEntry(
            original_window_id=window_id,
            key_codes=k_codes,
            key_scale=k_scale,
            key_zero=k_zero,
            val_codes=v_codes,
            val_scale=v_scale,
            val_zero=v_zero,
            position_range=position_range.to(torch.long).clone(),
            active=True,
        )
        self.ledger.demote_new(entry)

    def demote_many(
        self,
        window_ids: List[int],
        keys_pre_rope: Tensor,
        values: Tensor,
        position_ranges: Tensor,
    ) -> None:
        """First-time demotion of N windows in one batched quantize (§5).

        Equivalent to :meth:`demote` per window — the batched quantizers are the
        singular ones applied slice-wise — but pays one launch per tier instead
        of one per window. Every id must be **new** (no ledger entry); callers
        route reactivations through :meth:`reactivate`, which needs no quantize.

        Parameters
        ----------
        window_ids : list of int
            ``original_window_id`` per leading slice, in order.
        keys_pre_rope, values : Tensor
            ``[N, H_kv, window, D]``.
        position_ranges : Tensor
            ``[N, window]`` int64 original absolute positions.
        """
        if not window_ids:
            return
        self._invalidate()
        k_codes, k_scale, k_zero = quantize_key_windows(keys_pre_rope)
        v_codes, v_scale, v_zero = quantize_value_windows(values)
        pos = position_ranges.to(torch.long)
        for i, wid in enumerate(window_ids):
            self.ledger.demote_new(LedgerEntry(
                original_window_id=wid,
                key_codes=k_codes[i].contiguous(),
                key_scale=k_scale[i].contiguous(),
                key_zero=k_zero[i].contiguous(),
                val_codes=v_codes[i].contiguous(),
                val_scale=v_scale[i].contiguous(),
                val_zero=v_zero[i].contiguous(),
                position_range=pos[i].clone(),
                active=True,
            ))

    # -- promotion -----------------------------------------------------------

    def promote_many(
        self, window_ids: List[int], out_dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Promote N windows out of the Q tier in one batched dequantize (§5).

        Equivalent to :meth:`promote` per window, at one launch per tier instead
        of one per window. Marks every entry dormant and returns the tier
        stacked on a leading window axis, aligned with ``window_ids``.

        Returns
        -------
        keys_pre_rope : ``[N, H_kv, window, D]``
        values : ``[N, H_kv, window, D]``
        position_ranges : ``[N, window]`` int64
        """
        self._invalidate()
        entries = [self.ledger.get(w) for w in window_ids]
        k_codes = torch.stack([e.key_codes for e in entries], dim=0)
        k_scale = torch.stack([e.key_scale for e in entries], dim=0)
        k_zero = torch.stack([e.key_zero for e in entries], dim=0)
        v_codes = torch.stack([e.val_codes for e in entries], dim=0)
        v_scale = torch.stack([e.val_scale for e in entries], dim=0)
        v_zero = torch.stack([e.val_zero for e in entries], dim=0)
        positions = torch.stack([e.position_range for e in entries], dim=0).to(torch.long)

        keys = dequantize_key_windows(
            k_codes, k_scale, k_zero, self.window_size, out_dtype=out_dtype
        )
        values = dequantize_value_windows(
            v_codes, v_scale, v_zero, self.head_dim, out_dtype=out_dtype
        )
        for w in window_ids:
            self.ledger.deactivate(w)
        return keys, values, positions

    def promote(self, window_id: int, out_dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor]:
        """Promote one window out of the Q tier.

        Marks the entry dormant (retained for a possible re-demotion) and
        returns the dequantized **pre-RoPE** keys, values, and the frozen
        ``position_range``. The caller applies RoPE at those positions and
        splices the window into the fp store at its chronological slot (§5).

        Returns
        -------
        key_pre_rope : ``[H_kv, window, D]``
        value : ``[H_kv, window, D]``
        position_range : ``[window]`` int64
        """
        e = self.ledger.get(window_id)
        key_pre_rope = dequantize_key_window(
            e.key_codes, e.key_scale, e.key_zero, self.window_size, out_dtype=out_dtype
        )
        value = dequantize_value_window(
            e.val_codes, e.val_scale, e.val_zero, self.head_dim, out_dtype=out_dtype
        )
        position_range = e.position_range.clone()
        self._invalidate()
        self.ledger.deactivate(window_id)
        return key_pre_rope, value, position_range

    # -- read-path gather ----------------------------------------------------

    def gather_active(
        self, out_dtype: torch.dtype
    ) -> Optional[Tuple[List[int], Tensor, Tensor, Tensor]]:
        """Dequantize all **active** windows in chronological order.

        Returns ``None`` when the Q tier is empty. Otherwise:

        - ``window_ids`` : list of active ``original_window_id`` (ascending).
        - ``keys_pre_rope`` : ``[N_q, H_kv, window, D]`` — pre-RoPE.
        - ``values`` : ``[N_q, H_kv, window, D]``.
        - ``position_ranges`` : ``[N_q, window]`` int64 original positions.
        """
        entries = self.ledger.active_entries()
        if not entries:
            return None

        # Stack each frozen field into one dense tensor (a handful of constant
        # launches, regardless of window count), then dequantize the whole tier
        # in ONE batched call per tier — no per-window Python loop. On GPU the
        # old singular-per-window loop was launch-bound (~70 dispatches/window);
        # this path is flat at ~a dozen for any N_q. Numerics are unchanged:
        # the batched dequant is the singular op applied slice-wise.
        ids = [e.original_window_id for e in entries]
        k_codes = torch.stack([e.key_codes for e in entries], dim=0)   # [N,H,D,S/2]
        k_scale = torch.stack([e.key_scale for e in entries], dim=0)   # [N,H,D]
        k_zero = torch.stack([e.key_zero for e in entries], dim=0)     # [N,H,D]
        v_codes = torch.stack([e.val_codes for e in entries], dim=0)   # [N,H,S,D/2]
        v_scale = torch.stack([e.val_scale for e in entries], dim=0)   # [N,H,S]
        v_zero = torch.stack([e.val_zero for e in entries], dim=0)     # [N,H,S]
        positions = torch.stack(
            [e.position_range for e in entries], dim=0
        ).to(torch.long)                                              # [N,window]

        keys = dequantize_key_windows(
            k_codes, k_scale, k_zero, self.window_size, out_dtype=out_dtype
        )                                                             # [N,H,window,D]
        values = dequantize_value_windows(
            v_codes, v_scale, v_zero, self.head_dim, out_dtype=out_dtype
        )                                                             # [N,H,window,D]

        return ids, keys, values, positions

    def effective_q_tier(
        self, rope_module: torch.nn.Module, out_dtype: torch.dtype
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Read-ready Q tier: post-RoPE keys, values, and their positions.

        Returns ``None`` for an empty tier, else a triple flattened
        window-major over the active windows (ascending id):

        - ``keys``     : ``[H_kv, N_q * window, D]`` — dequantized, RoPE'd at
          each window's frozen ``position_range``.
        - ``values``   : ``[H_kv, N_q * window, D]``.
        - ``positions``: ``[N_q * window]`` int64 original absolute positions.

        **Memoized on** :attr:`version`. Codes and grids are written once at
        first demotion and ``position_range`` is never rebased (§10), so the
        dequant + RoPE are bit-identical until the ledger next changes — and the
        ledger only changes at eviction. Recomputing per decode step re-ran the
        whole tier's dequant and RoPE for a result that could not have moved:
        at ``window_size = 8`` that is wasted on 7 of every 8 steps, and the
        tier is the large one (int4 holds the *majority* of retained windows).
        """
        key = (self._version, out_dtype)
        if self._read_cache is not None and self._read_cache[0] == key:
            return self._read_cache[1]

        gathered = self.gather_active(out_dtype=out_dtype)
        if gathered is None:
            self._read_cache = (key, None)
            return None

        # Local import: `effective` owns the RoPE primitives and imports nothing
        # from this module, so this cannot cycle.
        from .effective import rotate_key_window

        _, keys_pre, values, positions = gathered
        n_q, h_kv, s, d = keys_pre.shape
        k_flat = keys_pre.permute(1, 0, 2, 3).reshape(h_kv, n_q * s, d)
        pos_flat = positions.reshape(n_q * s)
        keys = rotate_key_window(k_flat, pos_flat, rope_module).to(out_dtype)
        vals = values.to(out_dtype).permute(1, 0, 2, 3).reshape(h_kv, n_q * s, d)

        result = (keys, vals, pos_flat)
        self._read_cache = (key, result)
        return result

    # -- eviction bookkeeping ------------------------------------------------

    def retain_only(self, keep_ids) -> None:
        """Free ledger entries whose window was dropped outright (§6)."""
        self._invalidate()
        self.ledger.retain_only(keep_ids)
