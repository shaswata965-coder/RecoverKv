"""Route q>0 decode ``flash_attn_func`` calls through the fused two-tier kernel.

Same interception mechanism as :mod:`flash_lse` (transformers 4.47.1 has no
attention-function registry, so the one module-global ``flash_attn_func`` symbol
is the only seam), but where ``flash_lse`` *observes* the call, this one
*replaces* it: on a fused decode step the wrapper computes the attention output
directly from the fp tier (passed by the module) and the int2 Q tier (handed over
by the cache in a pending context), and writes the eviction score the same kernel
emits — so the model never materializes the Q tier to fp16, never concatenates,
and never runs a second ``q·kᵀ`` for scoring.

Routing is driven entirely by the cache: :meth:`WindowedCache.update` sets a
pending context (:func:`set_pending`) immediately before the module's flash call
on a real decode step whose Q tier is non-empty. Any call with no pending context
— prefill, ``q==0``, an empty Q tier, or a non-fused run — passes straight
through to the original ``flash_attn_func`` (which may itself be the
:mod:`flash_lse` wrapper). A forward-pre-hook clears the slot at the start of
every layer so a context can never leak between layers.

Layout: ``flash_attn_func`` is called with ``[B, S, H, D]`` (seqlen-major), so the
wrapper transposes q/k/v to the heads-major ``[B, H, S, D]`` the kernel and cache
use, and transposes the output back.

GPU-verify points (this file is exercised only on a CUDA+flash-attn box; the CPU
dev box never installs the patch): (1) ``flash_attn_func``'s positional arg order
``(q, k, v, ...)`` and the ``[B, S, H, D]`` layout; (2) that the module passes the
fp tier we returned from ``update`` as ``k``/``v``; (3) the softmax scale (we use
the cache-provided ``head_dim ** -0.5``, correct for Llama/Qwen).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .decode_kernel import fused_two_tier_decode
from .scorer import reduce_two_tier_scores


# Single-slot pending context, set by the cache in update() right before the
# module's flash call and consumed by the wrapper. Single-threaded forward, so
# one slot is enough; a forward-pre-hook clears it per layer against leaks.
_PENDING: dict = {"ctx": None}


def set_pending(ctx: dict) -> None:
    """Arm the next ``flash_attn_func`` call to run the fused decode kernel."""
    _PENDING["ctx"] = ctx


def clear() -> None:
    """Empty the pending slot. Call in a forward-pre-hook per layer."""
    _PENDING["ctx"] = None


def _flash_utils_module():
    try:  # lazy: keeps this file importable without transformers/flash-attn
        import transformers.modeling_flash_attention_utils as m  # type: ignore
        return m
    except Exception:
        return None


def _run_fused(ctx: dict, q_flash: torch.Tensor,
               k_flash: torch.Tensor, v_flash: torch.Tensor) -> torch.Tensor:
    """Compute the fused decode output + score for one layer. Returns flash layout.

    ``q_flash`` is ``[B, 1, H_q, D]``; ``k_flash``/``v_flash`` are the fp tier
    ``[B, S_fp, H_kv, D]`` (seqlen-major, as flash receives them). The Q tier and
    the score-scatter map come from ``ctx`` (heads-major, built by the cache).
    """
    q_hd = q_flash.transpose(1, 2)[:, :, 0, :]        # [B, H_q, D]
    k_fp = k_flash.transpose(1, 2)                    # [B, H_kv, S_fp, D]
    v_fp = v_flash.transpose(1, 2)

    out, token_scores = fused_two_tier_decode(
        q_hd, k_fp, v_fp, ctx["qtier"], ctx["scaling"]
    )                                                  # out [B,H_q,D], scores [B,H_q,S]

    order, q_token_len = ctx["score_meta"]
    scores = reduce_two_tier_scores(
        token_scores, ctx["num_sink"], ctx["window_size"], q_token_len, order
    )
    ctx["cache"].cache_kwargs[ctx["layer_idx"]]["window_scores"] = scores

    return out.unsqueeze(1)                            # [B, 1, H_q, D] (flash layout)


def _make_wrapper(orig):
    def wrapper(*args, **kwargs):
        ctx = _PENDING["ctx"]
        if ctx is None:
            return orig(*args, **kwargs)               # prefill / empty-Q / non-fused
        _PENDING["ctx"] = None
        # flash_attn_func(q, k, v, ...) — positional (see GPU-verify note above).
        q, k, v = args[0], args[1], args[2]
        return _run_fused(ctx, q, k, v)

    wrapper.__wrapped__ = orig
    wrapper._sticky_decode_wrapper = True
    return wrapper


@dataclass
class DecodeHandle:
    """Restores the original ``flash_attn_func`` on :meth:`restore`. Idempotent."""

    module: Any
    original: Any
    _restored: bool = False

    def restore(self) -> None:
        if self._restored:
            return
        try:
            if getattr(self.module, "flash_attn_func", None) is not None:
                self.module.flash_attn_func = self.original
        finally:
            self._restored = True
            _PENDING["ctx"] = None


def enable(module: Any = None) -> Optional[DecodeHandle]:
    """Install the fused-decode routing patch. Returns a handle, or None.

    Layered over :mod:`flash_lse` when both are on: this wrapper is outermost, so a
    non-fused call falls through to ``orig`` (the flash_lse wrapper, which still
    captures ``L`` for prefill scoring) and a fused call never reaches it.
    """
    mod = module if module is not None else _flash_utils_module()
    if mod is None:
        return None
    orig = getattr(mod, "flash_attn_func", None)
    if orig is None:
        return None
    if getattr(orig, "_sticky_decode_wrapper", False):
        return DecodeHandle(module=mod, original=getattr(orig, "__wrapped__", orig))
    _PENDING["ctx"] = None
    mod.flash_attn_func = _make_wrapper(orig)
    return DecodeHandle(module=mod, original=orig)
