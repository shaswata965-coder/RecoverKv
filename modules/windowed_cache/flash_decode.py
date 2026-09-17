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
every layer, and an armed-but-unconsumed slot at that point means the kernel did
not run for the previous layer — a hard :class:`FusedDecodeNotReached`, never a
silent skip (see its docstring for why silence would be worse than failing).

Layout: ``flash_attn_func`` is called with ``[B, S, H, D]`` (seqlen-major), so the
wrapper transposes q/k/v to the heads-major ``[B, H, S, D]`` the kernel and cache
use, and transposes the output back.

The read gate runs here too, for a reason that is structural rather than
incidental: picking which int2 windows to dequantize needs the **query**, and
``Cache.update()`` is handed keys and values only. So ``update()`` hands over the
frozen tier plus its sketch cards, and :func:`_run_fused` — the one place that
holds ``q`` — scores the cards, picks the top ``n_sel``, and passes the pick to
the kernel as an indirection (``sel``), together with the card estimates the
kernel needs to score the windows it skipped.

GPU-verify points (this file is exercised only on a CUDA+flash-attn box; the CPU
dev box never installs the patch): (1) ``flash_attn_func``'s positional arg order
``(q, k, v, ...)`` and the ``[B, S, H, D]`` layout; (2) that the module passes the
fp tier we returned from ``update`` as ``k``/``v``; (3) the softmax scale (we use
the cache-provided ``head_dim ** -0.5``, correct for Llama/Qwen); (4) that
``_gate_kernel`` matches :func:`~modules.windowed_cache.gate_kernel.gate_reference`
— it has never been executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .decode_kernel import fused_two_tier_decode
from .gate_kernel import fused_gate


# Single-slot pending context, set by the cache in update() right before the
# module's flash call and consumed by the wrapper. Single-threaded forward, so
# one slot is enough; a forward-pre-hook clears it per layer against leaks.
_PENDING: dict = {"ctx": None}

#: Proof-of-execution counters. ``armed`` counts cache.update() hand-offs,
#: ``fired`` counts wrapper invocations that actually ran the kernel. They must
#: stay equal; :func:`clear` raises the moment they diverge.
#:
#: ``gated`` and the two window counters are the same idea one level down. The
#: fused kernel running does NOT mean the gate ran: a store with no sketch cards
#: hands over ``gate=None`` and reads the whole tier, correctly and slowly, with
#: nothing in the output to say so. Counting is Python integer arithmetic on
#: values already in hand (``n_sel`` from the context, ``n_active`` from a tensor
#: SHAPE), so it costs no kernel launch and no device sync — the read fraction is
#: readable without a ``.item()`` anywhere on the decode path.
_STATS: dict = {"armed": 0, "fired": 0, "gated": 0,
                "windows_read": 0, "windows_active": 0}


def stats() -> dict:
    """Hand-offs vs. kernel runs vs. gate runs, plus the realised read fraction.

    ``armed`` == ``fired`` says the fused decode kernel served every layer it was
    handed. ``gated`` == ``fired`` says the read gate ran on every one of those;
    ``gated`` of 0 against a large ``fired`` is the silent no-cards case, which
    looks exactly like success in every other measurement.

    ``read_fraction`` is the windows actually dequantized over the windows
    available — ``quant_gate_ratio`` as realised, not as configured. It is
    ``None`` until the gate has run at least once.
    """
    s = dict(_STATS)
    s["read_fraction"] = (s["windows_read"] / s["windows_active"]
                          if s["windows_active"] else None)
    return s


def reset_stats() -> None:
    """Zero the counters (per-run harnesses; tests)."""
    for k in _STATS:
        _STATS[k] = 0


class FusedDecodeNotReached(RuntimeError):
    """The kernel was armed for a layer and the wrapper never ran it.

    Being able to *launch* the Triton kernel (what ``assert_decode_kernel_available``
    checks at hook install) is not the same as the model actually *routing*
    through it. The known way to arm-without-firing is transformers taking the
    **varlen** flash path (``flash_attn_varlen_func``, chosen when a padding
    ``attention_mask`` survives ``_update_causal_mask``): the module-global
    ``flash_attn_func`` this patch owns is then never called, the model silently
    attends over the FP TIER ALONE — the Q tier dropped — and the score hook
    still skips its own pass, so no ``window_scores`` are written either.

    That is silently wrong output *and* a decode timed against a method it is not
    running, which is precisely what the Triton-or-error contract exists to
    prevent. So it is a hard error, raised on the next layer's pre-hook — i.e.
    within one layer of the first offending step.
    """


def set_pending(ctx: dict) -> None:
    """Arm the next ``flash_attn_func`` call to run the fused decode kernel."""
    if _PENDING["ctx"] is not None:
        stale = _PENDING["ctx"]
        _PENDING["ctx"] = None
        raise FusedDecodeNotReached(
            "fused decode was armed for layer "
            f"{stale.get('layer_idx')} and never ran (armed="
            f"{_STATS['armed']}, fired={_STATS['fired']}) before layer "
            f"{ctx.get('layer_idx')} armed again. "
            + _DIAGNOSIS
        )
    _PENDING["ctx"] = ctx
    _STATS["armed"] += 1


def clear() -> None:
    """Empty the pending slot. Call in a forward-pre-hook per layer.

    Raises :class:`FusedDecodeNotReached` if the slot still holds a context: the
    previous layer handed the Q tier over and ``flash_attn_func`` was never
    called, so the fused kernel did not run for that layer.
    """
    ctx = _PENDING["ctx"]
    if ctx is None:
        return
    _PENDING["ctx"] = None
    raise FusedDecodeNotReached(
        f"fused decode was armed for layer {ctx.get('layer_idx')} and the "
        f"kernel never ran (armed={_STATS['armed']}, fired={_STATS['fired']}). "
        + _DIAGNOSIS
    )


_DIAGNOSIS = (
    "transformers called something other than the module-global "
    "`flash_attn_func` this patch wraps — almost certainly "
    "`flash_attn_varlen_func`, which `_flash_attention_forward` selects when a "
    "padding attention_mask reaches it. The fused path cannot serve that call, "
    "and letting it through would attend over the fp tier alone (Q tier "
    "dropped) with no eviction scores written. Use equal-length prompts (no "
    "padding), or set STICKYKV_FUSED_DECODE=0 to run the Phase-1 materialize "
    "path, which is correct under varlen."
)


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

    **This is where the read gate lives**, and it has to be here rather than in
    ``Cache.update()``: selecting windows needs the **query**, and transformers
    hands ``update()`` keys and values only. That is the shape of the API, not a
    choice we can revisit — so ``update()`` hands over the frozen tier and its
    cards, and the selection happens one layer later, with ``q`` in hand. The
    pure-PyTorch statement of the same step is
    :func:`~modules.windowed_cache.gated_decode.gated_decode_step`.
    """
    q_hd = q_flash.transpose(1, 2)[:, :, 0, :]        # [B, H_q, D]
    k_fp = k_flash.transpose(1, 2)                    # [B, H_kv, S_fp, D]
    v_fp = v_flash.transpose(1, 2)

    order, q_token_len = ctx["score_meta"]
    num_sink, ws = ctx["num_sink"], ctx["window_size"]
    # The window axis the kernel emits must be the axis `order` permutes, so the
    # body window count is derived the same way `compute_score_meta` derives it:
    # from the fp body's length, ceil-divided by ws (the newest body window is
    # usually partial and still owns a column).
    n_body_win = -(-max(k_fp.shape[2] - num_sink, 0) // ws)

    # 1. Gate: score every window from its card, keep the top `n_sel`. One extra
    #    launch reading 280 B/head/window, against the 8448 B/window it decides
    #    not to read. `sel` is an INDIRECTION into the tier the cache already
    #    gathered, not a re-gather — see `_two_tier_decode_kernel`'s GATED block.
    #    Indexed, not `.get`: a missing key would run ungated and correct, i.e. a
    #    decode silently timed against a method it is not running — the same
    #    failure FusedDecodeNotReached exists to refuse. `None` is the explicit
    #    way to say "this store has no cards".
    gate, sel, logmass = ctx["gate"], None, None
    if gate is not None:
        sel, logmass = fused_gate(
            q_hd, gate["card"], gate["anchor"], ctx["scaling"], gate["n_sel"])
        # Shapes only — no device sync, no launch. See _STATS.
        _STATS["gated"] += 1
        _STATS["windows_read"] += int(sel.shape[-1])
        _STATS["windows_active"] += int(logmass.shape[-1])

    # 2. Attend over [sink | fp body | selected Q], scoring as it goes. The
    #    kernel also scores the windows it skipped, from `logmass`, AND attends
    #    over them through their value centroids at that same calibrated weight —
    #    both in an epilogue pass that already runs. Doing either on the host cost
    #    22 torch ops here, 704 launches per token at L=32, on a path bound by
    #    launch count.
    out, wsum = fused_two_tier_decode(
        q_hd, k_fp, v_fp, ctx["qtier"], ctx["scaling"], num_sink, n_body_win,
        sel=sel, logmass=logmass,
        centroids=None if gate is None else gate["centroid"],
    )                                     # out [B,H_q,D], wsum [B,H_q,W_phys]

    # §5.1: the kernel already reduced S -> W in registers, so all that is left is
    # the physical -> merged-id permutation. This replaces
    # `reduce_two_tier_scores`'s pad + reshape + sum + einops-reduce + cat (and
    # the [B,H_q,S] fp32 round trip that fed them) with one gather.
    if order.shape[1] != wsum.shape[-1]:
        raise RuntimeError(
            f"score_meta permutes {order.shape[1]} windows but the kernel emitted "
            f"{wsum.shape[-1]} (n_body_win={n_body_win}, S_fp={k_fp.shape[2]}, "
            f"num_sink={num_sink}, ws={ws}). These are derived from the same store "
            "and must agree; a mismatch would scatter scores onto the wrong windows."
        )
    idx = order.unsqueeze(1).expand(wsum.shape[0], wsum.shape[1], order.shape[1])
    ctx["cache"].cache_kwargs[ctx["layer_idx"]]["window_scores"] = torch.gather(
        wsum, -1, idx)

    return out.unsqueeze(1)                            # [B, 1, H_q, D] (flash layout)


def _make_wrapper(orig):
    def wrapper(*args, **kwargs):
        ctx = _PENDING["ctx"]
        if ctx is None:
            return orig(*args, **kwargs)               # prefill / empty-Q / non-fused
        _PENDING["ctx"] = None
        _STATS["fired"] += 1
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
