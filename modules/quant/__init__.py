"""modules.quant — shared two-tier quantization package (design.md §2–§8).

A single copy imported by BOTH cache backends (``windowed_cache`` and
``windowed_eager_cache``); the backends differ only in ``hooks.py``. Everything
here is pure-tensor and CPU-testable — the only ``transformers`` dependency is
the rope module passed in at read time (for the Q-tier RoPE apply and the
one-time demotion un-rotate).

Design realities baked in (design.md):

- **Keep-original-positions.** Eviction never renumbers. A Q window's
  ``position_range`` (its original absolute positions) is frozen at first
  demotion; the read path applies RoPE at those fixed positions every step. No
  rerotation, no position override (§5).
- **Pinned grid by identity.** A window's int4 codes + fp16 scale/zero are
  written exactly once (first demotion) and never recomputed. A re-demotion
  reactivates the dormant slot; it never re-quantizes (§3, §10).
- **The Q tier carries a batch axis.** Rows evict divergently — row 0 may hold
  windows ``{1, 5}`` in int4 while row 1 holds ``{4, 11}`` — so the per-window
  record lives in a dense ``[B, N_slots, ...]`` slot table
  (:mod:`modules.quant.slots`), not a host-side dict keyed by window id. The
  retained counts stay equal across rows, so the effective K/V is dense and
  needs no padding (BATCHING_PLAN.md §3).
"""

from __future__ import annotations

from .quantizer import (
    dequantize_key_window,
    dequantize_key_windows,
    dequantize_value_window,
    dequantize_value_windows,
    pack_nibbles_last,
    quantize_key_window,
    quantize_value_window,
    unpack_nibbles_last,
)

__all__ = [
    "quantize_key_window",
    "dequantize_key_window",
    "dequantize_key_windows",
    "quantize_value_window",
    "dequantize_value_window",
    "dequantize_value_windows",
    "pack_nibbles_last",
    "unpack_nibbles_last",
]

# Slots / store / effective are re-exported once written (see build order).
try:  # pragma: no cover - populated incrementally during bring-up
    from .slots import QuantSlotTable, n_slots_for  # noqa: F401
    from .store import QuantizedStore  # noqa: F401
    from .effective import (  # noqa: F401
        materialize_effective_kv,
        rotate_key_window,
        unrotate_key_window,
        window_id_of,
    )

    __all__ += [
        "QuantSlotTable",
        "n_slots_for",
        "QuantizedStore",
        "materialize_effective_kv",
        "rotate_key_window",
        "unrotate_key_window",
        "window_id_of",
    ]
except ImportError:
    pass
