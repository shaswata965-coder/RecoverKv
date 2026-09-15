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

from .digest_kernel import digest_select

from .decode_kernel import fused_two_tier_decode


# Single-slot pending context, set by the cache in update() right before the
# module's flash call and consumed by the wrapper. Single-threaded forward, so
# one slot is enough; a forward-pre-hook clears it per layer against leaks.
_PENDING: dict = {"ctx": None}

#: Proof-of-execution counters. ``armed`` counts cache.update() hand-offs,
#: ``fired`` counts wrapper invocations that actually ran the kernel. They must
#: stay equal; :func:`clear` raises the moment they diverge.
_STATS: dict = {"armed": 0, "fired": 0}


def stats() -> dict:
    """``{"armed": n, "fired": m}`` — hand-offs vs. actual kernel runs."""
    return dict(_STATS)


def reset_stats() -> None:
    """Zero the counters (per-run harnesses; tests)."""
    _STATS["armed"] = 0
    _STATS["fired"] = 0


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



# ---------------------------------------------------------------------------
# Digest gate (DIGEST_GATED_DECODE_PLAN.md §3.2, §3.3)
# ---------------------------------------------------------------------------


def _gate_qtier(qtier: dict, q_hd: torch.Tensor):
    """Pick the top-``k`` Q windows by digest bound. Returns ``(qtier, sel)``.

    ``sel`` is ``[B, k]`` int32, ascending, indexing the **resident** window axis
    — or ``None`` when no gating happened (gate off, or ``k`` covers everything,
    in which case the untouched ``qtier`` is handed on so the step stays
    bit-identical to an ungated one rather than merely equal).

    **This selects; it does not gather.** ``sel`` rides into the kernel, which
    reads the chosen windows in place. The obvious alternative — compacting the
    six code fields plus cos/sin host-side so "the kernel needs no changes" — was
    implemented first and measured: eight ``index_select`` copies per layer per
    step, +2723 kernel launches and +26.7 ms of CUDA at B=8, against the ~16 ms
    the shorter kernel loop saves. Selection has to be an indirection, not a copy,
    or the gate costs more than it recovers.
    """
    lo_t = qtier.get("digest_lo_t")
    if lo_t is None:
        return qtier, None
    k = qtier.get("digest_top_k")
    B, n = lo_t.shape[0], lo_t.shape[-1]
    if k is None or k >= n:
        return qtier, None

    # One Triton launch for the bound AND the max-over-heads reduction, instead
    # of ~20 host ops. See modules/windowed_cache/digest_kernel.py for why the
    # selection lives here rather than inside the decode kernel.
    #
    # NOT sorted. Sorting `sel` ascending kept the selected axis chronological —
    # cosmetic, and expensive: CUDA `sort` is a multi-kernel radix pass, and at 32
    # layers x every decode step it was a measurable slice of the launches this
    # path costs. Nothing needs the order: the kernel resolves each column through
    # `sel` (pinned by tests/test_digest_gate_kernel.py, which includes an
    # UNSORTED case), and `_run_fused`'s scatter map is built from `sel` itself.
    sel = digest_select(q_hd, lo_t, qtier["digest_hi_t"], k)

    gated = dict(qtier)
    gated["sel"] = sel
    return gated, sel



# ---------------------------------------------------------------------------
# Selection tracing — answers "how many DISTINCT windows matter over a run?"
# ---------------------------------------------------------------------------
#
# The gate proves ~75% of the Q tier is unread on any GIVEN step. That is not the
# same as 75% being unnecessary: if each step wants a different 110 of 439, every
# window is still needed and nothing can be dropped. If the UNION over a
# generation is small, the tier is genuinely oversized and `N_q` could shrink —
# a real memory saving that needs no recall path (plan §1.3 forbids one).
#
# Keyed by (eviction epoch, layer), because eviction changes which windows are
# resident: a union taken across epochs would grow for a trivial reason. Within an
# epoch the resident set is frozen (design §10), so union / resident is meaningful.
#
# Off unless STICKYKV_DIGEST_TRACE is set, so production pays nothing.

_TRACE: dict = {"on": False, "epochs": {}}


def trace_enabled() -> bool:
    import os
    return bool(os.environ.get("STICKYKV_DIGEST_TRACE"))


def trace_reset() -> None:
    _TRACE["on"] = trace_enabled()
    _TRACE["epochs"] = {}


def _trace_selection(epoch, layer_idx, sel, q_win_ids) -> None:
    """Record which WINDOW IDS this step admitted. Window ids are stable; the
    positions in `sel` are not (they index the active axis, which is rebuilt at
    every eviction), so recording raw `sel` across epochs would be meaningless."""
    if not _TRACE["on"]:
        return
    # One sync per layer per step. Tracing runs are never timed — the whole point
    # is to read the selection, which lives on device.
    wids = torch.gather(q_win_ids, 1, sel.long()).flatten().tolist()
    key = (int(epoch), int(layer_idx))
    rec = _TRACE["epochs"].get(key)
    if rec is None:
        rec = {"selected": set(), "resident": int(q_win_ids.shape[1]),
               "rows": int(q_win_ids.shape[0]), "steps": 0, "k": int(sel.shape[1])}
        _TRACE["epochs"][key] = rec
    rec["selected"].update(wids)
    rec["steps"] += 1


def trace_summary() -> dict:
    """Per-epoch union sizes plus an aggregate. `union_frac` is the number to read.

    ``union_frac = 1.0`` means every resident window was wanted by some step, so
    none could have been dropped. A small value means the tier is oversized.
    """
    rows = []
    for (epoch, layer), rec in sorted(_TRACE["epochs"].items()):
        if rec["steps"] < 2:
            continue          # a 1-step epoch cannot show reuse; union == k by construction
        resident_total = rec["resident"] * rec["rows"]
        rows.append({
            "epoch": epoch, "layer": layer, "steps": rec["steps"],
            "k_per_step": rec["k"], "resident": rec["resident"],
            "union": len(rec["selected"]),
            "union_frac": len(rec["selected"]) / max(rec["resident"], 1),
            "k_frac": rec["k"] / max(rec["resident"], 1),
        })
    agg = {}
    if rows:
        agg = {
            "n_epoch_layer": len(rows),
            "mean_steps_per_epoch": sum(r["steps"] for r in rows) / len(rows),
            "mean_k_frac": sum(r["k_frac"] for r in rows) / len(rows),
            "mean_union_frac": sum(r["union_frac"] for r in rows) / len(rows),
            "max_union_frac": max(r["union_frac"] for r in rows),
            "min_union_frac": min(r["union_frac"] for r in rows),
        }
    return {"rows": rows, "aggregate": agg}


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

    order, q_token_len = ctx["score_meta"]
    num_sink, ws = ctx["num_sink"], ctx["window_size"]
    # The window axis the kernel emits must be the axis `order` permutes, so the
    # body window count is derived the same way `compute_score_meta` derives it:
    # from the fp body's length, ceil-divided by ws (the newest body window is
    # usually partial and still owns a column).
    n_body_win = -(-max(k_fp.shape[2] - num_sink, 0) // ws)

    qtier = ctx["qtier"]
    n_active = qtier["k_codes"].shape[1] if qtier is not None else 0
    # §3.3: only the Q half of the window axis is gated. The `n_body_win` fp
    # columns carry the sinks and the local window and are never touched — the
    # kernel-level statement of §0.1.
    if qtier is not None:
        qtier, sel = _gate_qtier(qtier, q_hd)
    else:
        sel = None

    # §4.1 decomposition. "both" is production: one gated run, its scores feed
    # eviction, and the two effects are entangled — which is precisely why a
    # quality delta measured under it cannot be attributed to either.
    mode = ctx["qtier"].get("digest_gate_mode", "both") if sel is not None else "both"
    if mode == "attention" and sel is not None:
        # Gated OUTPUT, ungated SCORES: eviction sees exactly what it would have
        # seen with the gate off, so any quality delta is the approximation in
        # attention alone. Two kernel runs per layer per step — diagnostic only.
        out, _ = fused_two_tier_decode(
            q_hd, k_fp, v_fp, qtier, ctx["scaling"], num_sink, n_body_win)
        _, wsum = fused_two_tier_decode(
            q_hd, k_fp, v_fp, ctx["qtier"], ctx["scaling"], num_sink, n_body_win)
        sel = None                        # scores are full-width; scatter normally
    elif mode == "eviction" and sel is not None:
        # Ungated OUTPUT (bit-identical to gate-off attention), gated SCORES: the
        # quality delta is then entirely the changed eviction ranking.
        out, wsum = fused_two_tier_decode(
            q_hd, k_fp, v_fp, ctx["qtier"], ctx["scaling"], num_sink, n_body_win)
    else:
        out, wsum = fused_two_tier_decode(
            q_hd, k_fp, v_fp, qtier, ctx["scaling"], num_sink, n_body_win
        )                                 # out [B,H_q,D], wsum [B,H_q,W_emit]
    if _TRACE["on"] and sel is not None:
        _trace_selection(ctx.get("epoch", 0), ctx["layer_idx"], sel,
                         ctx["q_win_ids"])

    # §5.1: the kernel already reduced S -> W in registers, so all that is left is
    # the physical -> merged-id permutation. This replaces
    # `reduce_two_tier_scores`'s pad + reshape + sum + einops-reduce + cat (and
    # the [B,H_q,S] fp32 round trip that fed them) with one gather.
    #
    # `order` always spans the FULL physical axis (every body window plus every
    # active Q window), because `window_scores` must stay full-width: a gated-out
    # window still owns a column in the running score, it just contributes
    # nothing to it this step. Gating shortens what the KERNEL emits, not what the
    # scorer is indexed by, so the two widths are checked separately.
    n_phys = n_body_win + n_active
    if order.shape[1] != n_phys:
        raise RuntimeError(
            f"score_meta permutes {order.shape[1]} windows but the physical axis "
            f"is {n_phys} (n_body_win={n_body_win}, n_active={n_active}, "
            f"S_fp={k_fp.shape[2]}, num_sink={num_sink}, ws={ws}). These are "
            "derived from the same store and must agree; a mismatch would scatter "
            "scores onto the wrong windows."
        )
    ungated_emit = sel is None or mode == "eviction"
    n_emit = n_phys if ungated_emit else n_body_win + sel.shape[1]
    if wsum.shape[-1] != n_emit:
        raise RuntimeError(
            f"the kernel emitted {wsum.shape[-1]} windows but the gate handed it "
            f"{n_emit} (n_body_win={n_body_win}, "
            f"n_selected={sel.shape[1] if sel is not None else n_active})."
        )

    B, H_q = wsum.shape[0], wsum.shape[1]
    if sel is None:
        idx = order.unsqueeze(1).expand(B, H_q, n_phys)
        scores = torch.gather(wsum, -1, idx)
    elif mode == "eviction":
        # The kernel ran ungated, so every physical column exists; the gate's
        # effect is applied here instead, by zeroing what it would have skipped.
        idx = order.unsqueeze(1).expand(B, H_q, n_phys)
        gathered = torch.gather(wsum, -1, idx)
        kept_phys = torch.zeros(B, n_phys, dtype=torch.bool, device=wsum.device)
        kept_phys[:, :n_body_win] = True
        kept_phys.scatter_(1, n_body_win + sel.long(), True)
        kept = torch.gather(kept_phys, 1, order)
        scores = gathered * kept.unsqueeze(1).to(wsum.dtype)
    else:
        # Scatter the emitted columns straight into merged-id space, rather than
        # building a physical->emitted map and gathering through it. `inv_order`
        # (merged position of each physical column) depends only on `order`,
        # which is memoized per window epoch, so the only per-step work is
        # indexing it by `sel`. That took this branch from ~28 launches to ~4 —
        # the scatter map was half the gate's host cost (§12.3).
        inv_order = ctx["inv_order"]                          # [B, n_phys]
        merged_q = torch.gather(inv_order, 1, n_body_win + sel.long())
        merged = torch.cat([inv_order[:, :n_body_win], merged_q], dim=1)
        scores = torch.zeros(B, H_q, n_phys, dtype=wsum.dtype, device=wsum.device)
        # §4.1: a skipped window keeps its column and receives nothing. Under a
        # decay-free `+=` accumulator that IS freezing it, and §12.1.1 measured
        # the whole effect as null at this operating point anyway.
        scores.scatter_(-1, merged.unsqueeze(1).expand(B, H_q, merged.shape[1]), wsum)
        ctx["cache"].cache_kwargs[ctx["layer_idx"]]["window_scores"] = scores
        return out.unsqueeze(1)
    ctx["cache"].cache_kwargs[ctx["layer_idx"]]["window_scores"] = scores

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
