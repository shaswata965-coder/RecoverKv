"""MiKV as a plug-in method for the evaluation runners.

``make_mikv_method`` follows the ``method_factory`` protocol of
:func:`modules.evaluation.perf_runner.resolve_method_factory`: it is called
once per config with the model and the operating point, and the handle it
returns builds one fresh cache per run and installs its hooks. The quality
runners (GSM8K, LongBench, RULER) use the same handle through
``cache.backend: external`` (see :func:`utils.cache_factory.build_external_method`).

How the operating point is chosen, in order:

1. ``importance_ratio`` or ``cache_budget`` in ``method_kwargs`` — MiKV's own
   knobs (:class:`MiKVConfig`), taken as given.
2. otherwise ``budget_tokens`` from the runner — the retained-token budget an
   evicting method is held to. MiKV evicts nothing, so it is held to the same
   **bytes** instead: ``budget_tokens`` fp16 tokens' worth at the end of the
   cell, i.e. ``cache_budget = budget_tokens / (prefill_len + gen_len)``.
3. otherwise the paper's headline, ``importance_ratio = 0.2``.

``describe()`` says which, so a row can never be read under the wrong one.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .cache import MiKVCache
from .config import MiKVConfig
from .hooks import MiKVHookHandles, install_mikv_hooks

_CONFIG_KEYS = set(MiKVConfig.__dataclass_fields__)


class MiKVMethod:
    """Method handle: ``new_cache()`` + ``install_hooks(model, cache)``."""

    #: MiKV forms its tiers inside the prefill forward (``update`` at prefill),
    #: so its one-off work is already inside TTFT.
    compaction_phase = "prefill"
    #: Scores come from the captured query, not ``attn_weights``.
    requires_output_attentions = False

    def __init__(self, config: MiKVConfig, num_layers: int, dtype: torch.dtype,
                 max_new_tokens: int = 0, sizing: str = "explicit",
                 head_dim: Optional[int] = None) -> None:
        self.config = config
        self.num_layers = num_layers
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens or 0)
        self.sizing = sizing
        self.head_dim = head_dim

    def new_cache(self, max_new_tokens: Optional[int] = None) -> MiKVCache:
        horizon = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        return MiKVCache(self.config, self.num_layers, self.dtype, horizon)

    def install_hooks(self, model, cache: MiKVCache) -> MiKVHookHandles:
        return install_mikv_hooks(model, cache)

    def describe(self) -> str:
        c = self.config
        if c.importance_ratio is not None:
            size = f"importance_ratio={c.importance_ratio:g}"
        else:
            size = f"cache_budget={c.cache_budget:.4f}"
        if self.head_dim:
            eff = c.resolve(self.head_dim, self._elem_bytes()).importance_ratio
            size += f" (r={eff:.4f})"
        g = c.group_size if c.group_size is not None else "D/2"
        return (f"MiKV {size} [{self.sizing}] hi=INT{c.hi_bits}"
                f"{'(fp)' if c.hi_bits == 16 else ''} lo=INT{c.lo_bits} "
                f"g={g} balancer={'on' if c.channel_balancer else 'off'} "
                f"recent_ratio={c.recent_ratio:g}")

    def metadata(self) -> Dict[str, Any]:
        """What a run's sidecar should record about the method."""
        meta: Dict[str, Any] = {"method": "mikv", "sizing": self.sizing,
                                "config": self.config.to_dict()}
        if self.head_dim:
            meta["resolved"] = dict(vars(self.config.resolve(
                self.head_dim, self._elem_bytes())))
        return meta

    def _elem_bytes(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()


def _head_dim(model_config) -> Optional[int]:
    if model_config is None:
        return None
    hd = getattr(model_config, "head_dim", None)
    if hd:
        return int(hd)
    return int(model_config.hidden_size // model_config.num_attention_heads)


def make_mikv_method(model=None, tokenizer=None, prefill_len: Optional[int] = None,
                     gen_len: Optional[int] = None, batch_size: Optional[int] = None,
                     budget_tokens: Optional[int] = None,
                     dtype: Optional[torch.dtype] = None,
                     **method_kwargs: Any) -> MiKVMethod:
    """``method_factory`` entry point: ``modules.mikv:make_mikv_method``."""
    unknown = set(method_kwargs) - _CONFIG_KEYS
    if unknown:
        raise TypeError(f"unknown MiKV method_kwargs {sorted(unknown)}; "
                        f"expected a subset of {sorted(_CONFIG_KEYS)}")
    kwargs = dict(method_kwargs)
    if "importance_ratio" in kwargs or "cache_budget" in kwargs:
        sizing = "explicit"
    elif budget_tokens is not None and prefill_len:
        horizon = int(prefill_len) + int(gen_len or 0)
        kwargs["cache_budget"] = min(1.0, float(budget_tokens) / horizon)
        sizing = "bytes-matched"
    else:
        sizing = "paper-default"
    config = MiKVConfig(**kwargs)

    model_config = getattr(model, "config", None)
    num_layers = int(model_config.num_hidden_layers) if model_config is not None else 0
    if num_layers <= 0:
        raise ValueError("make_mikv_method needs a model with config.num_hidden_layers")
    if dtype is None:
        dtype = getattr(model, "dtype", torch.float16)
    head_dim = _head_dim(model_config)
    if head_dim:
        # A budget below MiKV's floor fails here, not at the first prefill.
        config.resolve(head_dim, torch.empty((), dtype=dtype).element_size())
    return MiKVMethod(config, num_layers, dtype, max_new_tokens=gen_len or 0,
                      sizing=sizing, head_dim=head_dim)
