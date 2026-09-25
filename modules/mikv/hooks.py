"""The two hooks MiKV needs, and nothing else.

``Cache.update`` is handed keys and values only, but H2O's importance is a
function of the query. Rather than recompute the query (a second ``q_proj``
matmul per layer per step, and a second RoPE convention to keep in sync), the
hooks hand ``update`` the model's own:

* a **forward hook on** ``self_attn.q_proj`` stashes its output. In every
  Llama-family attention (eager, SDPA, flash-attention-2) ``q_proj`` runs
  before ``past_key_value.update`` in the same forward, so the stash is always
  there when ``update`` looks for it — and ``update`` raises if it is not.
* a **forward pre-hook on** ``self_attn`` stashes the ``attention_mask`` it is
  called with, which is the only place padding is visible.

Neither hook changes an activation. Removing them restores the model exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import torch.nn as nn

from .cache import MiKVCache


@dataclass
class MiKVHookHandles:
    handles: List[Any] = field(default_factory=list)
    cache: Any = None
    removed: bool = False

    def remove(self) -> None:
        """Remove every hook. Idempotent."""
        if self.removed:
            return
        for h in self.handles:
            h.remove()
        self.handles.clear()
        if self.cache is not None:
            self.cache.hooks_installed = False
            self.cache._pending_q.clear()
            self.cache._pending_mask.clear()
        self.removed = True


def attention_modules(model: nn.Module) -> List[nn.Module]:
    """Every attention module with a separate ``q_proj`` and a ``layer_idx``."""
    found = []
    for mod in model.modules():
        if (hasattr(mod, "q_proj") and hasattr(mod, "k_proj")
                and isinstance(getattr(mod, "layer_idx", None), int)):
            found.append(mod)
    return found


def install_mikv_hooks(model: nn.Module, cache: MiKVCache) -> MiKVHookHandles:
    """Wire ``cache`` to ``model``'s queries. Call once per cache, before generate."""
    attns = attention_modules(model)
    if not attns:
        raise TypeError(
            f"{type(model).__name__} has no attention module with a separate "
            "q_proj and a layer_idx; MiKV's hooks support the Llama family")
    layer_ids = sorted(m.layer_idx for m in attns)
    if layer_ids != list(range(cache.num_layers)):
        raise ValueError(
            f"model has attention layers {layer_ids[:3]}..{layer_ids[-1:]} but the "
            f"cache was built for {cache.num_layers} layers")

    out = MiKVHookHandles(cache=cache)

    for attn in attns:
        idx = attn.layer_idx

        def q_hook(_mod, _inp, output, _idx=idx):
            cache._stash_query(_idx, output)

        def mask_hook(_mod, args, kwargs, _idx=idx):
            mask = kwargs.get("attention_mask")
            if mask is None and len(args) > 1:
                mask = args[1]
            cache._stash_mask(_idx, mask)

        out.handles.append(attn.q_proj.register_forward_hook(q_hook))
        out.handles.append(attn.register_forward_pre_hook(mask_hook, with_kwargs=True))
    cache.hooks_installed = True
    return out
