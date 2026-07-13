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
  reactivates the dormant ledger entry; it never re-quantizes (§3, §10).
- **v1 is batch-size 1.** ``materialize_effective_kv`` runs per row and the
  ledger/store carry no batch axis; callers guard ``q > 0`` to ``B == 1``
  (design.md §10).
"""

from __future__ import annotations

from .quantizer import (
    dequantize_key_window,
    dequantize_value_window,
    pack_nibbles_last,
    quantize_key_window,
    quantize_value_window,
    unpack_nibbles_last,
)

__all__ = [
    "quantize_key_window",
    "dequantize_key_window",
    "quantize_value_window",
    "dequantize_value_window",
    "pack_nibbles_last",
    "unpack_nibbles_last",
]

# Ledger / store / effective are re-exported once written (see build order).
try:  # pragma: no cover - populated incrementally during bring-up
    from .ledger import LedgerEntry, QuantLedger  # noqa: F401
    from .store import QuantizedStore  # noqa: F401
    from .effective import (  # noqa: F401
        materialize_effective_kv,
        unrotate_key_window,
        window_id_of,
    )

    __all__ += [
        "LedgerEntry",
        "QuantLedger",
        "QuantizedStore",
        "materialize_effective_kv",
        "unrotate_key_window",
        "window_id_of",
    ]
except ImportError:
    pass
