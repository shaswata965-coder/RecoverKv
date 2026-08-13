"""Capture FlashAttention's softmax normaliser ``L`` from the real forward pass.

Why this file exists
--------------------
The Design-B score kernel needs ``L_i = m_i + log(ℓ_i)`` (the log-sum-exp softmax
normaliser) per query row. FlashAttention's forward *computes* ``L`` to finish
its own softmax, then throws it away: transformers 4.47.1's
``_flash_attention_forward`` calls ``flash_attn_func(...)`` **without**
``return_attn_probs=True`` and returns only ``attn_output`` (verified against the
v4.47.1 source). So today the score path recomputes ``L`` from scratch — a whole
extra ``q·kᵀ`` pass (:func:`score_kernel.compute_lse`).

This module recovers ``L`` for free by intercepting the flash call.

How (and why this is the only option on 4.47.1)
-----------------------------------------------
transformers 4.47.1 has **no** pluggable attention-function registry
(``ALL_ATTENTION_FUNCTIONS`` arrives in 4.48). Attention is a concrete
``LlamaFlashAttention2`` / ``Qwen2FlashAttention2`` chosen at init, and both route
through the *same* module-global ``flash_attn_func`` imported into
``transformers.modeling_flash_attention_utils`` via
``from flash_attn import flash_attn_func``. So a single monkeypatch of that one
symbol intercepts every flash call for every supported model.

The wrapper:

* calls the **real** ``flash_attn_func`` with ``return_attn_probs=True``,
* stashes ``softmax_lse`` (shape ``[B, H_q, T]`` for the non-varlen path),
* returns ``attn_output`` **unchanged** — the attention output is byte-identical
  to the unpatched call, so nothing downstream of attention is affected.

Safety / correctness guarantees
-------------------------------
* **Never corrupts attention.** The returned output is exactly flash's output.
  If ``return_attn_probs`` is unsupported by the installed flash-attn, the
  wrapper detects it once, marks itself broken, and thereafter calls flash
  plainly (no capture, no double compute).
* **Freshness by construction.** :func:`clear` is meant to run in a
  *forward-pre-hook* on each attention module, so the stash is emptied at the
  start of every layer's forward and only re-filled if that layer actually ran
  the non-varlen ``flash_attn_func``. A layer that took the padded
  (``flash_attn_varlen_func``) path leaves the stash ``None`` → the caller falls
  back to :func:`score_kernel.compute_lse`. This prevents one layer's ``L`` from
  leaking into another (all layers in one forward share the same ``[B,H_q,T]``
  shape, so a shape check alone would not catch staleness).

Everything here imports transformers lazily, so importing this module is safe
even where transformers/flash-attn are absent (e.g. CPU-only dev boxes): every
entry point degrades to a no-op and the caller keeps recomputing ``L``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


# Single-slot stash. ``lse`` holds the most recent non-varlen softmax_lse (or
# None). ``broken`` latches True if the installed flash-attn rejects
# ``return_attn_probs`` so we stop retrying it.
_STATE: dict = {"lse": None, "broken": False}


def clear() -> None:
    """Empty the stash. Call in a forward-pre-hook so each layer starts clean."""
    _STATE["lse"] = None


def pop() -> Optional[torch.Tensor]:
    """Return the stashed ``softmax_lse`` (``[B, H_q, T]``) and clear it."""
    lse = _STATE["lse"]
    _STATE["lse"] = None
    return lse


def _flash_utils_module():
    """The ``modeling_flash_attention_utils`` module, or None if unavailable."""
    try:  # lazy: keeps this file importable without transformers/flash-attn
        import transformers.modeling_flash_attention_utils as m  # type: ignore
        return m
    except Exception:
        return None


def _make_wrapper(orig):
    """Wrap ``flash_attn_func`` so it captures ``softmax_lse`` and returns output.

    The wrapper is transparent: callers still receive exactly ``attn_output``.
    """

    def wrapper(*args, **kwargs):
        # Once we've learned return_attn_probs isn't supported, don't retry it.
        if _STATE["broken"]:
            _STATE["lse"] = None
            return orig(*args, **kwargs)

        try:
            k2 = dict(kwargs)
            k2["return_attn_probs"] = True
            out = orig(*args, **k2)
        except TypeError:
            # This flash-attn build doesn't accept return_attn_probs. Latch off
            # and fall back to a clean call — capture disabled, output correct.
            _STATE["broken"] = True
            _STATE["lse"] = None
            return orig(*args, **kwargs)

        # With return_attn_probs the return is (out, softmax_lse, S_dmask).
        if isinstance(out, tuple) and len(out) >= 2:
            lse = out[1]
            # Only the non-varlen path yields a batched [B, H_q, T] lse we can
            # use directly; anything else (e.g. varlen (nheads, total_q)) is left
            # for the caller to detect via a shape check and fall back.
            _STATE["lse"] = lse if isinstance(lse, torch.Tensor) else None
            return out[0]

        # Unexpected shape of return value — don't guess, just don't capture.
        _STATE["lse"] = None
        return out

    wrapper.__wrapped__ = orig
    wrapper._sticky_lse_wrapper = True
    return wrapper


@dataclass
class CaptureHandle:
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
            _STATE["lse"] = None
            _STATE["broken"] = False


def enable(module: Any = None) -> Optional[CaptureHandle]:
    """Install the capture patch. Returns a handle, or None if unavailable.

    Parameters
    ----------
    module : optional
        The module whose ``flash_attn_func`` attribute to patch. Defaults to
        ``transformers.modeling_flash_attention_utils``. Injectable so the
        plumbing can be unit-tested with a fake module (no real flash-attn).

    Returns
    -------
    CaptureHandle or None
        None when the target module or ``flash_attn_func`` symbol is missing
        (flash-attn not installed) — in which case capture is simply disabled
        and the caller keeps recomputing ``L``.
    """
    mod = module if module is not None else _flash_utils_module()
    if mod is None:
        return None
    orig = getattr(mod, "flash_attn_func", None)
    if orig is None:
        return None
    # Already patched: hand back a handle that restores the underlying original.
    if getattr(orig, "_sticky_lse_wrapper", False):
        return CaptureHandle(module=mod, original=getattr(orig, "__wrapped__", orig))
    _STATE["lse"] = None
    _STATE["broken"] = False
    mod.flash_attn_func = _make_wrapper(orig)
    return CaptureHandle(module=mod, original=orig)
