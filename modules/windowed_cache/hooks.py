"""Score hooks for the flash-attn-2 backend — auxiliary-SDPA forward hook.

Flash-attention-2 never materializes the attention matrix, so per-key
importance scores cannot be read from the real forward pass.  Instead, a
``forward_hook`` on each attention module:

1. Recomputes the post-RoPE query states from the layer's own inputs
   (``hidden_states`` + ``position_embeddings``) — one extra ``q_proj``
   matmul, cheap relative to attention itself.
2. Reads the post-RoPE keys straight from the cache — they were appended by
   ``WindowedCache.update`` earlier in the same forward pass.
3. Runs an auxiliary SDPA pass over (q, k) to produce explicit attention
   weights.  Multi-row (prefill) passes are causally masked so a query row
   never attends to keys ahead of it.
4. Scores the weights via :func:`scorer.compute_window_scores` and writes
   the result to ``cache.cache_kwargs[layer_idx]["window_scores"]``.

Scoring policy: H2O-style cumulative.  Every query row in the current
forward pass contributes to the per-key score; the cache's ``update()``
then accumulates the per-step scores into ``state.window_scores``.  There
is no observation window.

Cost: the prefill auxiliary SDPA is ``O(N²)`` — the same order as the real
attention — and each generation step is ``O(S)``.  Neither is a bottleneck.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import os

from . import flash_lse, flash_decode
from .decode_kernel import (
    assert_decode_kernel_available,
    describe_decode_backend,
    fused_decode_enabled,
)
from .score_kernel import (
    assert_prefill_kernel_available,
    compute_token_scores,
    describe_prefill_backend,
)
from .scorer import (
    compute_window_scores,
    reduce_token_scores_to_windows,
    reduce_two_tier_scores,
)


def _prefill_score_chunk() -> int:
    """Query-row block size for the prefill score pass.

    The flash hook reconstructs ``softmax(q·kᵀ).sum(over queries)`` to score
    keys. Doing it in one shot materializes the full ``[B, H_q, T, S]`` matrix —
    tens of GiB per layer at full LongBench context (T up to ~18k). Because the
    score is a sum over query rows, we accumulate it in blocks of this many rows
    and never hold more than ``[B, H_q, chunk, S]``. Override with the env var
    ``STICKYKV_PREFILL_SCORE_CHUNK`` (smaller = less memory, more iterations).
    """
    try:
        v = int(os.environ.get("STICKYKV_PREFILL_SCORE_CHUNK", "1024"))
        return v if v > 0 else 1024
    except (TypeError, ValueError):
        return 1024


def _lse_from_forward() -> bool:
    """Whether to reuse flash's ``softmax_lse`` instead of recomputing ``L``.

    **On by default.** The flash forward is monkeypatched (:mod:`flash_lse`) to
    hand out the softmax normaliser it already computes, so the Triton score path
    skips its ``L`` recompute pass (:func:`score_kernel.compute_lse`) — and that
    recompute is the ``[B, H, chunk, S]`` fp32 transient (~17 GB at prefill=4096,
    batch=32) that OOMs the very shapes this method targets. Reusing L eliminates
    it: the fused key-outer kernel never materialises the score matrix. This is
    THE prefill-memory fix, so it is the default rather than an opt-in.

    Degrades safely: :func:`flash_lse.enable` returns None if flash-attn is
    absent, and the wrapper latches off if the installed build rejects
    ``return_attn_probs`` — in either case the score path falls back to
    recomputing L (and its transient) exactly as before. Set
    ``STICKYKV_SCORE_LSE_FROM_FORWARD=0`` to force that recompute (e.g. for parity
    against the pre-reuse numbers).
    """
    v = os.environ.get("STICKYKV_SCORE_LSE_FROM_FORWARD", "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def _score_softmax_dtype() -> torch.dtype:
    """Dtype for the auxiliary-score softmax intermediate.

    The softmax runs over the full ``[.., blk, S]`` logit block — the single
    largest transient in the prefill score pass. ``bfloat16`` halves that tensor
    versus ``float32`` while keeping fp32's exponent range (unlike ``float16``,
    whose 5-bit exponent can overflow on large logits), so it is the memory-lean
    default. Set ``STICKYKV_SCORE_SOFTMAX_DTYPE=float32`` to restore the exact
    fp32 reduction (byte-identical scores) for parity checks.
    """
    name = os.environ.get("STICKYKV_SCORE_SOFTMAX_DTYPE", "bfloat16").lower()
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }.get(name, torch.bfloat16)

try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        apply_rotary_pos_emb,
        repeat_kv,
    )
except ImportError:
    LlamaAttention = None  # type: ignore[assignment,misc]
    apply_rotary_pos_emb = None  # type: ignore[assignment]
    repeat_kv = None  # type: ignore[assignment]

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


def _extract_arg(
    args: Tuple, kwargs: Dict[str, Any], name: str, pos: int
) -> Optional[Any]:
    """Pull a forward argument by keyword name, falling back to position."""
    if name in kwargs:
        return kwargs[name]
    if len(args) > pos:
        return args[pos]
    return None


# ---------------------------------------------------------------------------
# HookHandles — idempotent removal
# ---------------------------------------------------------------------------


@dataclass
class HookHandles:
    """Manages installed forward hooks with idempotent ``remove()``.

    ``_cleanups`` holds zero-arg callables run at removal — used to restore the
    optional flash ``softmax_lse`` monkeypatch (:mod:`flash_lse`) so the process
    is left exactly as we found it.
    """

    _hook_handles: List[Any] = field(default_factory=list)
    _cleanups: List[Any] = field(default_factory=list)
    _removed: bool = False

    def remove(self) -> None:
        """Remove all hooks and run cleanups.  Idempotent."""
        if self._removed:
            return
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for cleanup in self._cleanups:
            try:
                cleanup()
            except Exception:
                pass
        self._cleanups.clear()
        self._removed = True


# ---------------------------------------------------------------------------
# install_score_hooks
# ---------------------------------------------------------------------------

# Guards the one-per-process "which scoring path" banner (install runs per
# sample in the runners, so an unguarded print would spam thousands of lines).
_PATH_ANNOUNCED = [False]


def install_score_hooks(
    model: nn.Module,
    cache: Any,
    config: Any,
) -> HookHandles:
    """Install score-extraction hooks on all attention modules.

    For each ``LlamaAttention`` / ``Qwen2Attention`` module, registers a
    ``forward_hook`` (with kwargs) that recomputes the post-RoPE query from
    the layer inputs, runs a causally-masked auxiliary SDPA against the
    cached keys, and reduces the result to per-window scores.

    Scoring uses every query row in the current forward pass (H2O-style
    cumulative); the cache accumulates the per-step scores across steps.

    Parameters
    ----------
    model : nn.Module
        The HuggingFace language model.
    cache : WindowedCache
        The cache instance — scores are written to ``cache.cache_kwargs``.
    config : WindowedCacheConfig or ResolvedConfig
        Configuration (``window_size``, ``num_sink_tokens``).

    Returns
    -------
    HookHandles
        Call ``.remove()`` to uninstall all hooks.
    """
    handles = HookHandles()

    # Committed to the Triton path? Choosing the flash backend on CUDA commits to
    # the Triton PREFILL score kernel unconditionally — compute_token_scores has
    # no PyTorch fallback at T > 1 — so every degrade-to-silence branch below
    # becomes a hard error instead: a run that quietly scores nothing times
    # sink+local eviction under this method's name, which is the exact failure
    # the Triton-or-error contract exists to make impossible.
    #
    # Note this is deliberately NOT gated on fused_decode_enabled(): turning the
    # decode kernel off (STICKYKV_FUSED_DECODE=0) opts out of the *decode* kernel
    # only, and prefill scoring still has to run on Triton.
    #
    # Off CUDA (CPU hook-install unit tests) the warn-and-degrade behaviour is
    # kept — nothing there could have launched a kernel in the first place.
    _cuda = torch.cuda.is_available()
    _committed = _cuda

    attn_classes = _get_attn_classes()
    if not attn_classes:
        msg = "No LlamaAttention or Qwen2Attention found — no hooks installed."
        if _committed:
            raise RuntimeError(
                msg + " The flash backend is committed to the Triton score/decode "
                "kernels, which only the hooked attention modules can reach, so "
                "this would score nothing at all. Check the model class."
            )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return handles
    if apply_rotary_pos_emb is None or repeat_kv is None:
        msg = ("transformers RoPE/GQA helpers unavailable — flash score hooks "
               "not installed; eviction would degrade to sink+local only.")
        if _committed:
            raise RuntimeError(msg + " Refusing to run: see the Triton-or-error "
                               "contract in score_kernel/decode_kernel.")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return handles

    window_size = getattr(config, "window_size", 8)
    num_sink = getattr(config, "num_sink_tokens", 4)

    # Optionally reuse flash's own softmax normaliser L instead of recomputing
    # it. This monkeypatches the process-global flash_attn_func (the only
    # interception point in transformers 4.47.1 — no attention-function registry
    # exists there) to hand out `softmax_lse`, without changing the attention
    # output. enable() returns None if flash-attn isn't installed, in which case
    # capture is silently disabled and the score path recomputes L as before.
    lse_capture = _lse_from_forward()
    if lse_capture:
        _lse_handle = flash_lse.enable()
        if _lse_handle is not None:
            handles._cleanups.append(_lse_handle.restore)
        else:
            lse_capture = False  # flash-attn unavailable; fall back to recompute

    # Fused two-tier decode (decode_kernel + flash_decode). Choosing the flash
    # backend commits to the Triton path for BOTH the prefill score kernel and the
    # decode kernel, so gate at install: a CUDA box that cannot launch them fails
    # HERE, not mid-run. Only activated on CUDA — the eager/CPU path uses the
    # materialize read path (and a CPU flash run already raises at the first
    # prefill), so hook-install unit tests on CPU are unaffected.
    fused_active = fused_decode_enabled() and _cuda
    if fused_active:
        assert_prefill_kernel_available(True)
        assert_decode_kernel_available(True)
        _dec_handle = flash_decode.enable()
        if _dec_handle is not None:
            handles._cleanups.append(_dec_handle.restore)
        setattr(cache, "_fused_decode_active", True)

    # Explicit, once-per-process banner: which scoring/attention path is active.
    if not _PATH_ANNOUNCED[0]:
        _PATH_ANNOUNCED[0] = True
        print(
            "[StickyKV] score path: FLASH (flash_attention_2) | prefill scoring "
            f"-> {describe_prefill_backend(_cuda)} | "
            f"decode -> {describe_decode_backend(_cuda)} | "
            f"L-reuse: {'ON' if lse_capture else 'off'}",
            flush=True,
        )

    # Discover attention modules and assign layer indices in module order.
    layer_idx_map: Dict[int, int] = {}
    layer_idx = 0
    for _name, module in model.named_modules():
        if isinstance(module, attn_classes):
            layer_idx_map[id(module)] = layer_idx
            layer_idx += 1

    warned_once = [False]

    # Fix: reuse the q_proj the real attention forward already computed this pass
    # instead of redoing the projection in the score hook. A forward hook on each
    # module.q_proj stashes its output here (keyed by layer); the score hook, which
    # fires just after the attention forward completes, consumes it. Recomputing it
    # was a re-read of ~6.7% of the layer's weights every step, at every batch size.
    q_proj_stash: Dict[int, Any] = {}

    def make_qproj_stash_hook(lidx: int):
        def qproj_hook(_module, _inp, output):
            # q_proj is called exactly once per attention forward, so this
            # overwrites cleanly each step and never accumulates across layers.
            q_proj_stash[lidx] = output
        return qproj_hook

    for _name, module in model.named_modules():
        if not isinstance(module, attn_classes):
            continue

        this_layer_idx = layer_idx_map[id(module)]

        # Capture this layer's q_proj output (pre-RoPE query). Skipped if the
        # module has no q_proj submodule (the score hook then recomputes it).
        if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Module):
            q_handle = module.q_proj.register_forward_hook(
                make_qproj_stash_hook(this_layer_idx)
            )
            handles._hook_handles.append(q_handle)

        # Clear the flash-lse stash at the START of each attention forward, so a
        # captured L can never leak from one layer to the next: the wrapper only
        # re-fills it if THIS layer runs the non-varlen flash_attn_func. A layer
        # that took the padded (varlen) path leaves it None → score hook falls
        # back to recomputing L. (All layers share the same [B,H_q,T] shape, so a
        # shape check alone would not catch staleness — this pre-hook does.)
        if lse_capture:
            pre_handle = module.register_forward_pre_hook(
                lambda _m, _a: flash_lse.clear()
            )
            handles._hook_handles.append(pre_handle)

        # Clear any pending fused-decode context at the START of each layer's
        # forward. update() re-arms it for this layer right before the flash call,
        # so this only guards against a context leaking from a layer that (against
        # expectation) never reached flash_attn_func.
        if fused_active:
            dec_pre = module.register_forward_pre_hook(
                lambda _m, _a: flash_decode.clear()
            )
            handles._hook_handles.append(dec_pre)

        def make_hook(lidx: int):
            def score_hook(module, args, kwargs, output):
                hidden_states = _extract_arg(args, kwargs, "hidden_states", 0)
                position_embeddings = _extract_arg(
                    args, kwargs, "position_embeddings", 1
                )
                if hidden_states is None or position_embeddings is None:
                    msg = ("Flash hook: hidden_states / position_embeddings "
                           "not found in the attention call — scoring "
                           "disabled, eviction degrades to sink+local only.")
                    if _committed:
                        raise RuntimeError(
                            msg + " Refusing to continue: the Triton score kernel "
                            "cannot run without them, and a silently unscored run "
                            "times sink+local under this method's name."
                        )
                    if not warned_once[0]:
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)
                        warned_once[0] = True
                    return

                # Fused decode already produced this step's scores. On a q>0 decode
                # step with a non-empty Q tier (exactly update()'s fused condition),
                # the flash_decode kernel wrote window_scores, so the auxiliary
                # score pass must NOT run — it would need the effective K, which the
                # fused path deliberately never builds. Prefill (T>1) and empty-Q
                # decode steps fall through and score as before.
                if (fused_active and getattr(cache, "_fused_decode_active", False)
                        and hidden_states.shape[1] == 1
                        and getattr(cache, "_q", 0.0) > 0.0):
                    st = cache._stores[lidx]
                    if st is not None and st.num_active_windows > 0:
                        return

                # Keys: already RoPE-applied and appended by cache.update()
                # earlier in this same forward pass. At q > 0 the raw fp store
                # misses the Q tier, so source the effective K (fp + dequantized,
                # RoPE'd Q windows, concatenated as [sink ‖ body ‖ Q]) instead —
                # the same tensor attention saw (design §8, §9).
                score_meta = None
                if getattr(cache, "_q", 0.0) > 0.0:
                    if cache._states[lidx].key_states is None:
                        if _committed:
                            raise RuntimeError(
                                f"Flash hook (layer {lidx}): the fp store is empty "
                                "at a scoring pass, so the Triton score kernel has "
                                "no keys to score. update() should have appended "
                                "them earlier in this same forward."
                            )
                        return
                    # Reuse the effective K + score-scatter map that
                    # cache.update() already built for this layer this pass (fix
                    # #2) instead of rebuilding them (a second full Q-tier dequant
                    # + RoPE, per layer, per step). Consume them so a later stray
                    # hook call can't read stale state; fall back to a fresh
                    # materialize if update() didn't run for this layer.
                    stash = getattr(cache, "_last_effective_k", None)
                    meta_stash = getattr(cache, "_last_score_meta", None)
                    if stash is not None and stash[lidx] is not None:
                        k_current = stash[lidx]  # [1, H_kv, S, D]
                        stash[lidx] = None
                        if meta_stash is not None:
                            score_meta = meta_stash[lidx]
                            meta_stash[lidx] = None
                    else:
                        k_current, _, score_meta = cache._materialize(lidx)
                else:
                    k_current = cache._states[lidx].key_states  # [B, H_kv, S, D]
                if k_current is None:
                    if _committed:
                        raise RuntimeError(
                            f"Flash hook (layer {lidx}): no effective K to score. "
                            "The Triton prefill score kernel cannot run, and "
                            "skipping would leave this layer unscored."
                        )
                    return

                # 1. Post-RoPE query from the layer's own inputs. Reuse the
                #    q_proj output the attention forward just computed (stashed by
                #    the q_proj forward hook above) rather than redoing the matmul;
                #    fall back to a recompute if the stash is empty (e.g. the
                #    module had no q_proj to hook). Same [B, T, H_q*D] tensor either
                #    way, so the view/transpose/RoPE below are byte-identical.
                head_dim = module.head_dim
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, head_dim)
                q_raw = q_proj_stash.pop(lidx, None)
                if q_raw is None:
                    q_raw = module.q_proj(hidden_states)
                q = q_raw.view(hidden_shape).transpose(1, 2)  # [B, H_q, T, D]
                cos, sin = position_embeddings
                q, _ = apply_rotary_pos_emb(q, q, cos, sin)
                q = q.to(k_current.dtype)

                # 2-3. Per-key received attention:
                #      softmax(scale·q·kᵀ).sum(over query rows) → token_scores
                #      [B, H_q, S]. Delegated to score_kernel.compute_token_scores,
                #      which routes by pass, not by an env knob:
                #        • prefill (T > 1) — the fused FlashAttention-2-*backward*-
                #          style Triton kernel: exact softmax exp(scale·q·kᵀ − L)
                #          per key, looped key-outer so each key block writes a
                #          disjoint slice (no atomics), fully-masked causal tiles
                #          skipped. Triton-ONLY — it RAISES if triton/CUDA is
                #          unavailable (the PyTorch prefill fallback OOMs at these
                #          shapes and has been removed).
                #        • decode (T == 1) — the chunked bf16 softmax loop
                #          (token_scores_torch); no grid to tile, so the kernel
                #          would only add launch overhead (design §6).
                #      Both return the same [B, H_q, S]. The window size never
                #      enters the score pass — it stays a downstream reshape in
                #      scorer.reduce_* below, so it can change freely per run.
                scaling = getattr(module, "scaling", head_dim ** -0.5)

                # Reuse flash's softmax normaliser L when it was captured this
                # forward (STICKYKV_SCORE_LSE_FROM_FORWARD + Triton backend). It
                # is only valid when the keys the score pass sees equal the keys
                # flash saw: pure prefill (T > 1) with no post-eviction reorder
                # (score_meta is None ⟹ k_current is the plain fp store). The
                # padded (varlen) flash path leaves the stash empty; the shape
                # check then rejects it and compute_token_scores recomputes L.
                lse = None
                if lse_capture and q.shape[2] > 1 and score_meta is None:
                    cand = flash_lse.pop()
                    if (
                        isinstance(cand, torch.Tensor)
                        and tuple(cand.shape) == (q.shape[0], q.shape[1], q.shape[2])
                    ):
                        lse = cand.to(device=q.device)

                token_scores = compute_token_scores(
                    q,
                    k_current,
                    scaling,
                    lse=lse,
                    chunk=_prefill_score_chunk(),
                    softmax_dtype=_score_softmax_dtype(),
                    out_dtype=q.dtype,
                )                                                    # [B, H_q, S]

                # 4. Reduce to per-window scores and hand off to the cache. At
                #    q > 0 the effective K is the unsorted [sink ‖ body ‖ Q]
                #    layout, so undo it on the score axis (bit-identical to the
                #    old sorted-layout reduce); otherwise it is a single
                #    ascending-id run and the plain contiguous reduce applies.
                if score_meta is not None:
                    order, q_token_len = score_meta
                    scores = reduce_two_tier_scores(
                        token_scores, num_sink, window_size, q_token_len, order
                    )
                else:
                    scores = reduce_token_scores_to_windows(
                        token_scores, num_sink, window_size
                    )
                cache.cache_kwargs[lidx]["window_scores"] = scores

            return score_hook

        handle = module.register_forward_hook(
            make_hook(this_layer_idx), with_kwargs=True
        )
        handles._hook_handles.append(handle)

    return handles
