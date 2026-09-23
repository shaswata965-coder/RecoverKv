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
#: ``gated`` and the two window counters are the same idea one level down.
#: ``gated`` must equal ``fired``: there is no ungated arm on the flash path, so
#: a fused layer that did not gate is a hard error (``_run_fused``), not a slow
#: success. The counters are how a finished run proves it. Counting is Python
#: integer arithmetic on
#: values already in hand (``n_sel`` from the context, ``n_active`` from a tensor
#: SHAPE), so it costs no kernel launch and no device sync — the read fraction is
#: readable without a ``.item()`` anywhere on the decode path.
_STATS: dict = {"armed": 0, "fired": 0, "gated": 0,
                "windows_read": 0, "windows_active": 0}


def stats() -> dict:
    """Hand-offs vs. kernel runs vs. gate runs, plus the realised read fraction.

    ``armed`` == ``fired`` says the fused decode kernel served every layer it was
    handed, and ``gated`` == ``fired`` says the read gate ran on every one of
    those. Both are invariants the code enforces rather than hopes for — a
    missing gate raises in ``_run_fused`` — so these read as proof, not as a
    diagnostic to go checking when a number looks wrong.

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


#: Verdicts :func:`gate_report` can return, as stable strings a runner can store
#: in its output and a reader can grep for.
GATE_OK = "gated"
GATE_NOT_EXPECTED = "not-expected"
GATE_NEVER_FIRED = "NOT-GATED-kernel-never-fired"
GATE_PARTIAL = "NOT-GATED-partial"


def expect_gated(backend_package, quant_ratio, cuda: bool) -> bool:
    """Should this configuration have gated? One definition, for every runner.

    Takes primitives rather than a config object so it can live beside the
    counters it explains without importing the evaluation layer. The three
    conditions are the same ones ``hooks.install_score_hooks`` uses to choose the
    path, restated where a runner can check them AFTER the fact:

    * the flash backend — the fused decode patch replaces the module-global
      ``flash_attn_func``, so the eager backend never reaches it (that is the
      supported way to ask for an ungated decode);
    * ``quant_ratio > 0`` — with no Q tier there is nothing to select over;
    * CUDA — the fused kernel is Triton-or-raise, and CPU runs the materialize
      path, which exists as the kernel's oracle and not as an alternative.
    """
    try:
        q = float(quant_ratio or 0.0)
    except (TypeError, ValueError):
        q = 0.0
    return bool(cuda) and q > 0.0 and str(backend_package or "") == "flash_attn"


def gate_report(expect_gated: bool) -> dict:
    """:func:`stats` plus a verdict — the one check every runner should record.

    ``expect_gated`` is the caller's claim about its own configuration: True
    when the run is on the flash backend with a Q tier (``quant_ratio > 0``), so
    the read gate is the only path the decode step may take. False for the eager
    backend or ``quant_ratio == 0``, where there is no Q tier to select over and
    an ungated read is the correct and only behaviour.

    Why every runner and not just the perf suite: a run that did NOT gate reads
    the whole int2 tier, produces correct output, and scores normally. It is
    invisible in an accuracy number in exactly the way it was invisible in a
    latency number — and this branch shipped eight commits of the latter. A
    LongBench score quoted for "the gated method" when the gate never ran is the
    same error wearing different clothes, and it is harder to catch, because
    quality moves for a hundred reasons and nobody re-derives them.

    Returns the counters plus ``verdict`` (one of the ``GATE_*`` constants),
    ``expected``, and ``ok``. It never raises: a quality run that has finished
    generating should record what happened rather than lose the work.
    """
    s = stats()
    s["expected"] = bool(expect_gated)
    if not expect_gated:
        s["verdict"] = GATE_NOT_EXPECTED
    elif not s["fired"]:
        s["verdict"] = GATE_NEVER_FIRED
    elif s["gated"] != s["fired"]:
        s["verdict"] = GATE_PARTIAL
    else:
        s["verdict"] = GATE_OK
    s["ok"] = s["verdict"] in (GATE_OK, GATE_NOT_EXPECTED)
    return s


def log_gate_report(log, label: str, expect_gated: bool) -> dict:
    """:func:`gate_report`, announced. Returns the report for the caller to store."""
    r = gate_report(expect_gated)
    rf = r.get("read_fraction")
    rf_s = "n/a" if rf is None else f"{rf:.3f}"
    if r["verdict"] == GATE_OK:
        log.info("%s: read gate ran on all %d fused layers, realised read "
                 "fraction %s", label, r["fired"], rf_s)
    elif r["verdict"] == GATE_NOT_EXPECTED:
        log.info("%s: no read gate expected (eager backend, or quant_ratio=0 so "
                 "there is no Q tier to select over)", label)
    elif r["verdict"] == GATE_NEVER_FIRED:
        log.warning(
            "%s: THE READ GATE NEVER RAN. The fused decode kernel fired 0 times, "
            "so this run used the materialize path over the WHOLE int2 tier. The "
            "output is correct and the score is real, but it is NOT the gated "
            "method and must not be quoted as one.", label)
    else:
        log.warning(
            "%s: the read gate ran on %d of %d fused layers. The flash path has "
            "no ungated arm, so a gap means some layers took a different route "
            "entirely; this run is not the shipped method.",
            label, r["gated"], r["fired"])
    return r


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

    # 1. Gate: score every window from its card, keep the top `n_sel` by the
    #    largest share of its own int2 mass any query head of the GQA group puts
    #    on it (NOT by raw logit: a raw max lets the head with the largest
    #    baseline choose for the whole group -- sketch.group_share). Two launches:
    #    the card scan, reading 272 B/head/window against the window it decides
    #    not to read, and the share union over its output. `sel` is an
    #    INDIRECTION into the tier the cache already gathered, not a re-gather —
    #    see `_two_tier_decode_kernel`'s GATED block.
    #    Indexed, not `.get`: a missing key would run ungated and correct, i.e. a
    #    decode silently timed against a method it is not running — the same
    #    failure FusedDecodeNotReached exists to refuse. `None` is the explicit
    #    way to say "this store has no cards".
    gate = ctx["gate"]
    if gate is None:
        # No ungated arm on this path, by design. The caller only arms the fused
        # decode when `store.num_active_windows > 0`, so arriving here without
        # cards means a store holding a Q tier it cannot select over: the gate
        # would be skipped, the whole tier read, and the run would look exactly
        # like success while timing a method it is not running. That is the
        # failure FusedDecodeNotReached exists to refuse, one level down, and it
        # is how the gate sat unreachable for eight commits.
        raise FusedDecodeNotReached(
            f"layer {ctx.get('layer_idx')} armed the fused decode over "
            f"{int(ctx['qtier']['k_codes'].shape[1])} active int2 windows with "
            "no gate cards. The read gate is the only Q-tier read path on the "
            "flash backend — there is no ungated fallback. A store with a Q "
            "tier must carry cards: check `quant_sketch_enabled` (derived from "
            "quant_ratio > 0) and that the store has demoted at least once. "
            "To read every window, set quant_gate_ratio=1.0, which selects all "
            "of them through this same path."
        )
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
        sel=sel, logmass=logmass, centroids=gate["centroid"],
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
