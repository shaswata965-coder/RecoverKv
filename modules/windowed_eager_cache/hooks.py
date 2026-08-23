"""Score hooks for the eager-attention backend — plain ``forward_hook``.

When ``attn_implementation="eager"``, HF's eager attention materializes the
full softmax-attention tensor and returns it via the module output tuple
(gated by ``output_attentions=True``).  A plain ``register_forward_hook``
reads it directly — no monkey-patch, no captured q/k, no auxiliary pass.

Runner contract: the runner **must** pass ``output_attentions=True`` to
``model.generate(...)`` / ``model.forward(...)``.  Without it, HF returns
``None`` for attn_weights and the hook **raises** — reading those weights IS the
eager scoring path, so their absence is an unrecoverable failure, not a
warn-and-skip (it would otherwise time eviction degraded to sink+local under the
config's name). This mirrors the flash backend's Triton-or-raise contract.

Scoring policy: H2O-style cumulative.  Every query row in the current
forward pass contributes to the per-key score; the cache's ``update()``
accumulates the per-step scores into ``state.window_scores`` across
steps.  There is no observation window.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .scorer import reduce_token_scores_to_windows, reduce_two_tier_scores

try:
    from transformers.models.llama.modeling_llama import LlamaAttention
except ImportError:
    LlamaAttention = None  # type: ignore[assignment,misc]

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
except ImportError:
    Qwen2Attention = None  # type: ignore[assignment,misc]


def _get_attn_classes() -> Tuple:
    """Return a tuple of attention module classes to target."""
    classes = []
    if LlamaAttention is not None:
        classes.append(LlamaAttention)
    if Qwen2Attention is not None:
        classes.append(Qwen2Attention)
    return tuple(classes)


def score_from_attn_weights(output, cache, layer_idx, num_sink, window_size):
    """Reduce an eager attention module's output to per-window scores, or RAISE.

    Reading HF's materialized ``attn_weights`` **is** the eager backend's scoring
    path. If they are absent — the module did not run eager attention, or
    ``output_attentions=True`` was not passed — there is nothing to score, and a
    silent skip would time a DIFFERENT method (eviction degraded to sink+local)
    under this config's name. So this RAISES instead of degrading, mirroring the
    flash backend's Triton-or-raise contract (score_kernel.compute_token_scores).

    Lives at module scope (not inside the hook closure) so the fail-hard contract
    is unit-testable without a real attention module.
    """
    if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
        raise RuntimeError(
            "Eager score hook: attn_weights are absent from the attention "
            "output. The eager backend scores by reading HF's materialized "
            "attention weights, which requires attn_implementation='eager' AND "
            "output_attentions=True on every forward. Without them scoring is "
            "impossible and eviction would silently degrade to sink+local, so "
            "this fails rather than mis-measure the method. Pair the eager cache "
            "package with eager attention (validate_backend_attn_pairing) and "
            "pass output_attentions=True."
        )

    # attn_weights: [B, H_q, T, S] over the effective-K axis. Sum across the T
    # (query) axis — every query row contributes (H2O cumulative, no obs_window)
    # — to per-key received attention.
    token_scores = output[1].sum(dim=-2)                 # [B, H_q, S]

    # At q > 0 the effective K is the unsorted [sink ‖ body ‖ Q] layout, so the
    # key axis of attn_weights is unsorted too; undo it on the score axis with
    # the scatter map update() stashed this pass (bit-identical to the old sorted
    # reduce). Otherwise it is a single ascending-id run and the plain contiguous
    # reduce applies.
    meta_stash = getattr(cache, "_last_score_meta", None)
    score_meta = meta_stash[layer_idx] if meta_stash is not None else None
    if score_meta is not None:
        order, q_token_len = score_meta
        scores = reduce_two_tier_scores(
            token_scores, num_sink, window_size, q_token_len, order
        )
        meta_stash[layer_idx] = None
    else:
        scores = reduce_token_scores_to_windows(
            token_scores, num_sink, window_size
        )

    # Push into cache_kwargs; cache.update() accumulates across steps.
    cache.cache_kwargs[layer_idx]["window_scores"] = scores


# Guards the one-per-process "which scoring path" banner (install runs per
# sample in the runners, so an unguarded print would spam thousands of lines).
_PATH_ANNOUNCED = [False]


# ---------------------------------------------------------------------------
# HookHandles — idempotent removal
# ---------------------------------------------------------------------------


@dataclass
class HookHandles:
    """Manages installed hooks with idempotent ``remove()``."""

    _hook_handles: List[Any] = field(default_factory=list)
    _removed: bool = False

    def remove(self) -> None:
        """Remove all hooks.  Idempotent."""
        if self._removed:
            return
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._removed = True


# ---------------------------------------------------------------------------
# install_score_hooks
# ---------------------------------------------------------------------------


def install_score_hooks(
    model: nn.Module,
    cache: Any,
    config: Any,
) -> HookHandles:
    """Install score-extraction hooks on all attention modules.

    For each ``LlamaAttention`` / ``Qwen2Attention`` module, registers a
    ``forward_hook`` that reads ``attn_weights`` from the module output tuple
    (requires ``output_attentions=True``) and reduces it to per-window scores.

    Scoring uses every query row in the current forward pass (H2O-style
    cumulative); the cache accumulates the per-step scores across steps.

    Parameters
    ----------
    model : nn.Module
        The HuggingFace language model.
    cache : WindowedCache
        The cache instance — scores are written to ``cache.cache_kwargs``.
    config : WindowedCacheConfig or ResolvedConfig
        Configuration.

    Returns
    -------
    HookHandles
        Call ``.remove()`` to uninstall all hooks.
    """
    handles = HookHandles()
    attn_classes = _get_attn_classes()
    if not attn_classes:
        warnings.warn(
            "No LlamaAttention or Qwen2Attention found — no hooks installed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return handles

    window_size = getattr(config, "window_size", 8)
    num_sink = getattr(config, "num_sink_tokens", 4)

    # Explicit, once-per-process banner: the eager path reads HF's attention
    # weights directly, so there is no Triton kernel involved here.
    if not _PATH_ANNOUNCED[0]:
        _PATH_ANNOUNCED[0] = True
        print(
            "[StickyKV] score path: EAGER (attention weights read directly; "
            "no Triton kernel)",
            flush=True,
        )

    # Discover attention modules
    layer_idx_map: Dict[int, int] = {}
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, attn_classes):
            layer_idx_map[id(module)] = layer_idx
            layer_idx += 1

    for name, module in model.named_modules():
        if not isinstance(module, attn_classes):
            continue

        this_layer_idx = layer_idx_map[id(module)]

        def make_hook(lidx):
            def score_hook(module, input, output):
                # output = (hidden_states, attn_weights, past_key_value) when
                # output_attentions=True. Reducing to per-window scores — and
                # RAISING if the weights are absent (eager-or-fail, mirroring the
                # flash Triton-or-raise contract) — lives in the module-level
                # helper so it is unit-testable without a real attention module.
                score_from_attn_weights(
                    output, cache, lidx, num_sink, window_size
                )

            return score_hook

        handle = module.register_forward_hook(make_hook(this_layer_idx))
        handles._hook_handles.append(handle)

    return handles
