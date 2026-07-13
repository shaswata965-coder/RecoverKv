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

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .ledger import LedgerEntry, QuantLedger
from .quantizer import (
    dequantize_key_window,
    dequantize_value_window,
    quantize_key_window,
    quantize_value_window,
)


class QuantizedStore:
    """Q-tier storage + operations for one layer (B = 1)."""

    def __init__(self, window_size: int, head_dim: int, num_kv_heads: int) -> None:
        self.window_size = window_size
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.ledger = QuantLedger()

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

    # -- promotion -----------------------------------------------------------

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

        keys: List[Tensor] = []
        values: List[Tensor] = []
        positions: List[Tensor] = []
        ids: List[int] = []
        for e in entries:
            keys.append(
                dequantize_key_window(
                    e.key_codes, e.key_scale, e.key_zero,
                    self.window_size, out_dtype=out_dtype,
                )
            )
            values.append(
                dequantize_value_window(
                    e.val_codes, e.val_scale, e.val_zero,
                    self.head_dim, out_dtype=out_dtype,
                )
            )
            positions.append(e.position_range.to(torch.long))
            ids.append(e.original_window_id)

        return (
            ids,
            torch.stack(keys, dim=0),       # [N_q, H_kv, window, D]
            torch.stack(values, dim=0),     # [N_q, H_kv, window, D]
            torch.stack(positions, dim=0),  # [N_q, window]
        )

    # -- eviction bookkeeping ------------------------------------------------

    def retain_only(self, keep_ids) -> None:
        """Free ledger entries whose window was dropped outright (§6)."""
        self.ledger.retain_only(keep_ids)
