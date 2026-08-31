"""FlashInfer-backed attention + **reliable** softmax-LSE for the score path.

Why this exists (vs :mod:`flash_lse`)
-------------------------------------
:mod:`flash_lse` recovers the softmax normaliser ``L`` by calling the installed
``flash_attn_func`` with ``return_attn_probs=True`` and stashing ``softmax_lse``.
That capture is **fragile**: many flash-attn builds reject or reshape that
argument, in which case the wrapper latches ``broken`` and *every* prefill layer
silently falls back to :func:`score_kernel.compute_lse` — a whole extra
``O(N²)`` PyTorch ``matmul + logsumexp`` pass per layer. That fallback is the
prime suspect for the pathological prefill TTFT (our method ~4.5× FullKV-flash
at the same shape, while SnapKV / KIVI sit ~1×).

FlashInfer exposes ``L`` as a **first-class return** of its prefill kernel
(``return_lse=True``), so there is no probability-mask hack to reject and no
silent latch-off. As a bonus, FlashInfer's prefill kernel *replaces* the model's
attention in the same call — the model gets a (numerically-equivalent) attention
output AND we get ``L`` from **one** pass. So this module delivers both asks at
once: reliable L-reuse *and* the faster kernel for the prefill attention itself.

Interface parity with :mod:`flash_lse`
--------------------------------------
Exposes the same ``enable() -> handle`` / ``clear()`` / ``pop() -> Optional[Tensor]``
surface, so :func:`hooks.install_score_hooks` selects between the two L-sources by
an env knob without any other change. The decode path is untouched: FlashInfer
does not do the two-tier int2-in-register + score-emitting decode kernel, so
:mod:`flash_decode` still owns decode; this wrapper only handles ``T > 1`` prefill
calls and passes ``T == 1`` (empty-Q decode) straight through to the real
``flash_attn_func``.

Scope handled
-------------
The **equal-length, non-padded batched prefill** that the perf/eviction path runs
(``[B, S, H, D]`` heads-second-to-last, contiguous KV). Anything else — a padded
(varlen) call, a layout it does not recognise, or any FlashInfer error — falls
through to the original ``flash_attn_func`` with the stash left ``None``, so the
score path degrades to :func:`score_kernel.compute_lse` exactly as before. Never
wrong, only (in the fallback) slower.

============================ GPU-VALIDATION CHECKLIST ============================
This file is authored on a CPU-only box and ships **UNVALIDATED**, like the
Triton kernels (design convention). Before trusting a run, confirm on the GPU:

  1. **LSE base.**  The score kernel needs ``L = ln Σ exp(scale·q·kᵀ)`` (natural
     log, over the *scaled* logits). Recent FlashInfer (>= 0.2) returns natural-
     log LSE; some builds historically returned **log2**. ``test_flashinfer_lse``
     asserts FlashInfer's LSE == :func:`score_kernel.compute_lse` to fp rounding.
     If it is off by exactly a factor ``ln 2 ≈ 0.6931``, set
     ``STICKYKV_FLASHINFER_LSE_LOG2=1`` to convert.
  2. **API names.**  Targets the ``plan`` / ``run`` wrapper API (FlashInfer
     >= 0.2). Older 0.1.x used ``begin_forward`` / ``forward``.
  3. **Output parity.**  The attention output returned to the model must match
     flash-attn to fp rounding (else the model degrades). The parity test checks
     this against ``F.scaled_dot_product_attention``.
  4. **Scale.**  We pass ``sm_scale = softmax_scale`` from the model's own flash
     call (``head_dim**-0.5`` for Llama/Qwen); do not double-apply it.
=================================================================================
"""

from __future__ import annotations

import math
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import torch


# Single-slot stash + latches, mirroring flash_lse._STATE. ``lse`` holds the most
# recent prefill softmax_lse as [B, H_q, T] (or None); ``broken`` latches True on
# the first FlashInfer failure so we stop retrying and run plain flash.
#
# ``reason`` / ``trace`` record WHY it latched. Without them the latch is
# indistinguishable from "the patched seam was never reached": both present as
# a successful install whose L never arrives, so the perf runner's L-reuse-miss
# warning had to *guess* at a cause (it blames the batch>1 varlen path, which
# the perf harness cannot take — it passes no attention_mask, so
# _update_causal_mask hands flash_attention_2 a None mask and
# _flash_attention_forward calls the patched flash_attn_func). Same lesson the
# eviction's compile failure learned in d44baba: a swallowed exception on a
# fallback path costs more than the fallback saves.
_STATE: dict = {"lse": None, "broken": False, "reason": None, "trace": None}

# Reusable device workspace + a plan cache keyed on shape. FlashInfer's wrapper
# wants a scratch buffer and a per-shape plan(); prefill calls every layer at the
# same shape, so we plan once per (shape, causal, dtype) and reuse across layers.
_WORKSPACE: dict = {"buf": None, "device": None}
_PLAN_CACHE: dict = {}


def _workspace_bytes() -> int:
    """Size of the FlashInfer scratch buffer (override with env; default 128 MiB)."""
    try:
        mb = int(os.environ.get("STICKYKV_FLASHINFER_WORKSPACE_MB", "128"))
        return max(mb, 16) * 1024 * 1024
    except (TypeError, ValueError):
        return 128 * 1024 * 1024


def _lse_is_log2() -> bool:
    """Whether the installed FlashInfer returns **log2** LSE (needs ×ln2 → ln)."""
    v = os.environ.get("STICKYKV_FLASHINFER_LSE_LOG2", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def clear() -> None:
    """Empty the stash. Call in a forward-pre-hook so each layer starts clean."""
    _STATE["lse"] = None


def pop() -> Optional[torch.Tensor]:
    """Return the stashed ``softmax_lse`` (``[B, H_q, T]``) and clear it."""
    lse = _STATE["lse"]
    _STATE["lse"] = None
    return lse


def strict() -> bool:
    """Whether an L-capture failure RAISES instead of degrading (default ON).

    The prefill score kernel is already Triton-or-error (score_kernel.py: no
    PyTorch prefill fallback, because the reference OOMs the shapes this method
    targets). The L-capture was the one piece of the prefill path that still
    degraded silently, which is why "installed source: flashinfer" and
    "compute_lse ran 32x" could coexist for a whole benchmark campaign.

    Degrading here is not a correctness fallback -- the recompute produces the
    same L -- but it silently costs a second O(N^2) pass and a fp32 block that
    is 32 GB at 4096/32, i.e. the OOM. A cost that large must not be paid
    quietly. ``STICKYKV_LSE_STRICT=0`` restores the old degrade for runs where
    only the output matters (quality suites).
    """
    return os.environ.get("STICKYKV_LSE_STRICT", "1").strip().lower() in (
        "1", "true", "yes", "on")


def broken_reason() -> Optional[str]:
    """Why the FlashInfer L-capture latched off (``"Type: msg"``), or ``None``.

    ``None`` with ``compute_lse`` still running every layer means the capture
    never *failed* — it was never *reached*, which is a different bug in a
    different place (the patched ``flash_attn_func`` symbol is not the one the
    model calls). Distinguishing those two is the whole point of recording it.
    """
    return _STATE["reason"]


def broken_traceback() -> Optional[str]:
    """Full traceback of the latching failure, or ``None``."""
    return _STATE["trace"]


def _flashinfer():
    """The flashinfer module, or None (lazy — keeps this importable on CPU)."""
    try:  # pragma: no cover - only importable on a CUDA box with flashinfer
        import flashinfer  # type: ignore
        return flashinfer
    except Exception:
        return None


def _flash_utils_module():
    """``transformers.modeling_flash_attention_utils`` or None (lazy)."""
    try:  # pragma: no cover
        import transformers.modeling_flash_attention_utils as m  # type: ignore
        return m
    except Exception:
        return None


def _get_workspace(device: torch.device) -> torch.Tensor:
    """Allocate (once per device) and return the FlashInfer scratch buffer."""
    if _WORKSPACE["buf"] is None or _WORKSPACE["device"] != device:
        _WORKSPACE["buf"] = torch.empty(
            _workspace_bytes(), dtype=torch.uint8, device=device
        )
        _WORKSPACE["device"] = device
    return _WORKSPACE["buf"]


def _planned_wrapper(fi, B, S_q, S_kv, H_q, H_kv, D, causal, scale, q_dtype, device):
    """A FlashInfer ragged-prefill wrapper, planned for this shape (cached).

    Equal-length batch → the ragged indptrs are just multiples of the per-row
    lengths, so no padding metadata is needed. Cached across layers because every
    prefill layer in one forward shares the shape.
    """
    key = (B, S_q, S_kv, H_q, H_kv, D, bool(causal), q_dtype, str(device))
    entry = _PLAN_CACHE.get(key)
    if entry is not None:
        return entry
    workspace = _get_workspace(device)
    wrapper = fi.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    qo_indptr = torch.arange(0, (B + 1) * S_q, S_q, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(0, (B + 1) * S_kv, S_kv, dtype=torch.int32, device=device)
    wrapper.plan(
        qo_indptr, kv_indptr,
        num_qo_heads=H_q, num_kv_heads=H_kv, head_dim_qk=D,
        causal=bool(causal), sm_scale=float(scale),
        q_data_type=q_dtype,
    )
    _PLAN_CACHE[key] = wrapper
    return wrapper


def _make_wrapper(orig):
    """Wrap ``flash_attn_func`` to run FlashInfer prefill and capture ``L``.

    Transparent: returns exactly an attention output in the ``[B, S, H, D]`` layout
    the caller expects. Only ``T > 1`` (prefill) is served by FlashInfer; ``T == 1``
    and any unhandled/failed case fall through to ``orig`` with ``L`` left ``None``.
    """
    def wrapper(*args, **kwargs):
        if _STATE["broken"]:
            _STATE["lse"] = None
            return orig(*args, **kwargs)

        q, k, v = args[0], args[1], args[2]
        # Only the clean batched prefill is handled here.
        if not (isinstance(q, torch.Tensor) and q.dim() == 4 and q.shape[1] > 1):
            _STATE["lse"] = None
            return orig(*args, **kwargs)

        fi = _flashinfer()
        if fi is None:
            _STATE["broken"] = True
            _STATE["lse"] = None
            return orig(*args, **kwargs)

        B, S_q, H_q, D = q.shape
        S_kv, H_kv = k.shape[1], k.shape[2]
        causal = kwargs.get("causal", True)
        scale = kwargs.get("softmax_scale", None)
        if scale is None:
            scale = 1.0 / math.sqrt(D)

        try:
            wrapper_fi = _planned_wrapper(
                fi, B, S_q, S_kv, H_q, H_kv, D, causal, scale, q.dtype, q.device
            )
            q_flat = q.reshape(B * S_q, H_q, D)
            k_flat = k.reshape(B * S_kv, H_kv, D)
            v_flat = v.reshape(B * S_kv, H_kv, D)
            out_flat, lse_flat = wrapper_fi.run(
                q_flat, k_flat, v_flat, return_lse=True
            )
        except Exception as exc:
            # Any API / shape / build mismatch. RECORD the reason first — this
            # latch is the difference between prefill paying one O(N^2) pass and
            # two, and until it is reported the only visible symptom is "L-reuse
            # installed but never fired".
            _STATE["broken"] = True
            _STATE["reason"] = f"{type(exc).__name__}: {exc}"
            _STATE["trace"] = traceback.format_exc()
            _STATE["lse"] = None
            if strict():
                # Raise FROM the original, at the point of failure, with the
                # FlashInfer traceback intact. A post-hoc per-cell warning
                # cannot show which call in which layer broke or why.
                raise RuntimeError(
                    "FlashInfer L-capture failed and STICKYKV_LSE_STRICT is on, "
                    "so this is a hard error rather than a silent second "
                    f"O(N^2) prefill pass. Shape was B={B} S_q={S_q} S_kv={S_kv} "
                    f"H_q={H_q} H_kv={H_kv} D={D} causal={bool(causal)} "
                    f"dtype={q.dtype}. Remedies, cheapest first: "
                    "STICKYKV_LSE_BACKEND=flash (same kernel, asks it for the "
                    "softmax_lse it already computed — attention output stays "
                    "bit-identical); a larger "
                    "STICKYKV_FLASHINFER_WORKSPACE_MB; or "
                    "STICKYKV_LSE_STRICT=0 to accept the recompute. See "
                    "PREFILL_PLAN.md Stage 1."
                ) from exc
            return orig(*args, **kwargs)

        # LSE: [B*S_q, H_q] -> [B, H_q, S_q] (natural log; convert if log2 build).
        lse = lse_flat.reshape(B, S_q, H_q).permute(0, 2, 1).contiguous().float()
        if _lse_is_log2():
            lse = lse * math.log(2.0)
        _STATE["lse"] = lse

        # Output back to the model's expected [B, S_q, H_q, D] layout, orig dtype.
        return out_flat.reshape(B, S_q, H_q, D).to(q.dtype)

    wrapper.__wrapped__ = orig
    wrapper._sticky_flashinfer_lse_wrapper = True
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
            _PLAN_CACHE.clear()


def available() -> bool:
    """True iff both transformers' flash utils and flashinfer import."""
    return _flash_utils_module() is not None and _flashinfer() is not None


def enable(module: Any = None) -> Optional[CaptureHandle]:
    """Install the FlashInfer capture patch. Returns a handle, or None.

    None when transformers' flash utils or flashinfer is unavailable — the caller
    then either selects :mod:`flash_lse` or recomputes ``L``. Layered under
    :mod:`flash_decode` exactly like :mod:`flash_lse`: install this BEFORE
    flash_decode so a decode call reaches flash_decode's wrapper first and a
    prefill call falls through to this one.
    """
    mod = module if module is not None else _flash_utils_module()
    if mod is None or _flashinfer() is None:
        return None
    orig = getattr(mod, "flash_attn_func", None)
    if orig is None:
        return None
    if getattr(orig, "_sticky_flashinfer_lse_wrapper", False):
        return CaptureHandle(module=mod, original=getattr(orig, "__wrapped__", orig))
    _STATE["lse"] = None
    _STATE["broken"] = False
    _PLAN_CACHE.clear()
    mod.flash_attn_func = _make_wrapper(orig)
    return CaptureHandle(module=mod, original=orig)
