"""MiKV: importance-aware mixed-precision KV cache (Yang et al., 2024).

"No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed
Precision Quantization", arXiv:2402.18096. H2O decides which tokens are
important; important tokens stay at high precision, and every other token —
the ones H2O would evict — is kept at low precision instead of dropped, with
an outlier-aware channel balancer on the keys.

Usage::

    from modules.mikv import MiKVCache, MiKVConfig, install_mikv_hooks

    cache = MiKVCache(MiKVConfig(importance_ratio=0.2, lo_bits=2),
                      num_layers=model.config.num_hidden_layers,
                      dtype=model.dtype, max_new_tokens=256)
    hooks = install_mikv_hooks(model, cache)
    try:
        out = model.generate(ids, past_key_values=cache, max_new_tokens=256)
    finally:
        hooks.remove()
    print(cache.memory_report())

See ``MIKV.md`` at the repository root for the design and its status.
"""

from .cache import MiKVCache, MiKVHooksMissing, apply_rope, attention_colsum
from .config import (
    PAPER_IMPORTANCE_RATIO,
    MiKVBudget,
    MiKVConfig,
    MiKVConfigError,
    paper_cache_size,
)
from .hooks import MiKVHookHandles, install_mikv_hooks
from .method import MiKVMethod, make_mikv_method

__all__ = [
    "MiKVBudget",
    "MiKVCache",
    "MiKVConfig",
    "MiKVConfigError",
    "MiKVHookHandles",
    "MiKVHooksMissing",
    "MiKVMethod",
    "PAPER_IMPORTANCE_RATIO",
    "apply_rope",
    "attention_colsum",
    "install_mikv_hooks",
    "make_mikv_method",
    "paper_cache_size",
]
