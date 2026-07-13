"""Per-window ledger for the Q tier (design.md §6).

One :class:`QuantLedger` per layer (B = 1 in v1). Each :class:`LedgerEntry`
is keyed by ``original_window_id`` and holds everything frozen about a Q
window across evictions:

| field | mutable? | purpose |
|---|---|---|
| ``original_window_id`` | no | chronological identity; used for the interleaved sort |
| ``key_codes`` / ``val_codes`` | no | packed int4 bits; never change after demotion |
| ``key_scale`` / ``key_zero`` / ``val_scale`` / ``val_zero`` | no | pinned fp16 grid; set once at demotion |
| ``position_range`` | no | the window's **original absolute positions**; what the read path feeds to RoPE |
| ``active`` | yes | in the Q tier (True) vs promoted-and-dormant (False) |

Codes are written **exactly once** (first demotion). A promotion **deactivates**
the entry (retained dormant, not freed) so a re-demotion is a pure reactivation
— never a re-quant (§3, §10). An entry is freed only when its window is dropped
outright.

The design's physical ``offset`` (byte offset into a compacting dense store) is
a Phase-2 layout concern; Phase 1 keeps the codes in the entry and materializes
the dense gap-free Q store on demand from the active entries (see
:mod:`modules.quant.store`). No arithmetic path ever runs ``quant(dequant(·))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from torch import Tensor


@dataclass
class LedgerEntry:
    """Frozen quant record for one Q window (one row; B = 1 in v1).

    Every tensor here is set once at first demotion and never rewritten. Only
    :attr:`active` changes (promotion → dormant, re-demotion → active).
    """

    original_window_id: int
    key_codes: Tensor      # [H_kv, D, window // 2] uint8, channel-major
    key_scale: Tensor      # [H_kv, D] fp16
    key_zero: Tensor       # [H_kv, D] fp16
    val_codes: Tensor      # [H_kv, window, D // 2] uint8, token-major
    val_scale: Tensor      # [H_kv, window] fp16
    val_zero: Tensor       # [H_kv, window] fp16
    position_range: Tensor  # [window] int64 — original absolute positions
    active: bool = True


class QuantLedger:
    """Per-layer collection of Q-window records keyed by ``original_window_id``."""

    def __init__(self) -> None:
        self._entries: Dict[int, LedgerEntry] = {}

    # -- membership / queries ------------------------------------------------

    def contains(self, window_id: int) -> bool:
        return window_id in self._entries

    def is_active(self, window_id: int) -> bool:
        e = self._entries.get(window_id)
        return e is not None and e.active

    def get(self, window_id: int) -> LedgerEntry:
        return self._entries[window_id]

    @property
    def num_active(self) -> int:
        return sum(1 for e in self._entries.values() if e.active)

    @property
    def num_dormant(self) -> int:
        return sum(1 for e in self._entries.values() if not e.active)

    def active_ids(self) -> List[int]:
        """Active window ids in chronological (window-id) order."""
        return sorted(wid for wid, e in self._entries.items() if e.active)

    def active_entries(self) -> List[LedgerEntry]:
        """Active entries sorted by ``original_window_id`` (the interleave order)."""
        return [self._entries[wid] for wid in self.active_ids()]

    # -- mutations -----------------------------------------------------------

    def demote_new(self, entry: LedgerEntry) -> None:
        """First-time demotion: register a fresh **active** entry.

        Raises if the window already has an entry (a re-demotion must go through
        :meth:`reactivate`, never re-quantize).
        """
        wid = entry.original_window_id
        if wid in self._entries:
            raise KeyError(
                f"window {wid} already in ledger; re-demotion must reactivate, "
                f"not re-quantize (design §10)"
            )
        entry.active = True
        self._entries[wid] = entry

    def reactivate(self, window_id: int) -> LedgerEntry:
        """Re-demotion: flip a dormant entry back to active (no recompute)."""
        e = self._entries[window_id]
        if e.active:
            raise ValueError(f"window {window_id} is already active")
        e.active = True
        return e

    def deactivate(self, window_id: int) -> LedgerEntry:
        """Promotion: retain the entry dormant (codes/grid/positions kept)."""
        e = self._entries[window_id]
        e.active = False
        return e

    def drop(self, window_id: int) -> None:
        """Free an entry outright (its window was dropped, not promoted)."""
        self._entries.pop(window_id, None)

    def retain_only(self, keep_ids) -> None:
        """Free every entry whose ``window_id`` is **not** in ``keep_ids``.

        Used at eviction to release entries for windows dropped entirely
        (neither an fp survivor nor a Q survivor nor a live-dormant promotion).
        """
        keep = set(int(w) for w in keep_ids)
        for wid in [w for w in self._entries if w not in keep]:
            del self._entries[wid]
