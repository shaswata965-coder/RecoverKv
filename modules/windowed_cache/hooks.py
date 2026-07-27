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
import torch.nn.functional as F

import os

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

try:
    from transformers.models.mistral.modeling_mistral import MistralAttention
except ImportError:
    MistralAttention = None  # type: ignore[assignment,misc]


def _get_attn_classes() -> Tuple:
    """Return a tuple of attention module classes to target."""
    classes = []
    if LlamaAttention is not None:
        classes.append(LlamaAttention)
    if Qwen2Attention is not None:
        classes.append(Qwen2Attention)
    if MistralAttention is not None:
        classes.append(MistralAttention)
    return tuple(classes)


def _extract_arg(
    args: Tuple, kwargs: Dict[str, Any], name: str, pos: Optional[int] = None
) -> Optional[Any]:
    """Pull a forward argument by keyword name, falling back to position.

    ``pos=None`` means name-only: no positional fallback. Use it whenever a wrong
    guess would return a DIFFERENT tensor rather than nothing, because every
    caller here treats ``None`` as "not available" and degrades loudly, while a
    wrong tensor is consumed silently.
    """
    if name in kwargs:
        return kwargs[name]
    if pos is not None and len(args) > pos:
        return args[pos]
    return None


# ---------------------------------------------------------------------------
# HookHandles — idempotent removal
# ---------------------------------------------------------------------------


@dataclass
class HookHandles:
    """Manages installed forward hooks with idempotent ``remove()``."""

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
    attn_classes = _get_attn_classes()
    if not attn_classes:
        warnings.warn(
            "No LlamaAttention or Qwen2Attention found — no hooks installed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return handles
    if apply_rotary_pos_emb is None or repeat_kv is None:
        warnings.warn(
            "transformers RoPE/GQA helpers unavailable — flash score hooks "
            "not installed; eviction would degrade to sink+local only.",
            RuntimeWarning,
            stacklevel=2,
        )
        return handles

    window_size = getattr(config, "window_size", 8)
    num_sink = getattr(config, "num_sink_tokens", 4)

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

        def make_hook(lidx: int):
            def score_hook(module, args, kwargs, output):
                hidden_states = _extract_arg(args, kwargs, "hidden_states", 0)
                # Name-only, deliberately. position_embeddings is the LAST
                # parameter of the attention forward (index 7 on Llama and Qwen2,
                # absent entirely on Mistral) — not index 1, which is
                # attention_mask. Every decoder layer in 4.47.1 passes it by
                # keyword, so the lookup always succeeds; a positional fallback
                # could only ever fire when it is wrong, and would hand a mask to
                # the RoPE path where nothing would raise. None instead falls
                # through to the rotary_emb recompute below, then to a warning.
                position_embeddings = _extract_arg(
                    args, kwargs, "position_embeddings"
                )
                if position_embeddings is None and hasattr(module, "rotary_emb"):
                    # Older-style attention modules (e.g. MistralAttention in
                    # transformers<=4.45) never receive position_embeddings as a
                    # forward argument -- the top-level model doesn't compute it
                    # at all (unlike Llama's refactored interface). Each layer
                    # instead computes (cos, sin) itself from its own
                    # self.rotary_emb(x, position_ids). Recompute the same way;
                    # x is only used for its .dtype/.device, so hidden_states
                    # stands in fine for the value-states shape rotary_emb wants.
                    position_ids = _extract_arg(args, kwargs, "position_ids", 2)
                    if hidden_states is not None and position_ids is not None:
                        position_embeddings = module.rotary_emb(
                            hidden_states, position_ids
                        )
                if hidden_states is None or position_embeddings is None:
                    if not warned_once[0]:
                        warnings.warn(
                            "Flash hook: hidden_states / position_embeddings "
                            "not found in the attention call — scoring "
                            "disabled, eviction degrades to sink+local only.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        warned_once[0] = True
                    return

                # Keys: already RoPE-applied and appended by cache.update()
                # earlier in this same forward pass. At q > 0 the raw fp store
                # misses the Q tier, so source the effective K (fp + dequantized,
                # RoPE'd Q windows, concatenated as [sink ‖ body ‖ Q]) instead —
                # the same tensor attention saw (design §8, §9).
                score_meta = None
                if getattr(cache, "_q", 0.0) > 0.0:
                    if cache._states[lidx].key_states is None:
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

                T = q.shape[2]
                S = k_current.shape[2]

                # 2. Score at KV-head granularity WITHOUT expanding K.
                #    repeat_kv materializes a num_groups×-larger [B, H_q, S, D]
                #    key copy (its expand+reshape forces a real copy) that is
                #    only averaged back to a per-window mean downstream. Instead,
                #    view the query heads as (H_kv, n_rep) and let the S-dim
                #    matmul broadcast over n_rep against the un-expanded keys —
                #    identical per-head scores, no num_groups× key copy.
                B = q.shape[0]
                H_q = q.shape[1]
                num_groups = getattr(module, "num_key_value_groups", 1)
                H_kv = H_q // num_groups
                q5 = q.reshape(B, H_kv, num_groups, T, head_dim)  # [B,H_kv,rep,T,D]
                k_t = k_current.transpose(-2, -1).unsqueeze(2)    # [B,H_kv,1,D,S]

                # 3. Auxiliary attention scoring, CHUNKED over the query rows.
                #    The score we need is softmax(q·kᵀ).sum(over query rows) — a
                #    sum, so we accumulate it in query-row blocks and never
                #    materialize the full [B, H_q, T, S] matrix. Peak memory is
                #    O(chunk · S) instead of O(T · S).
                #
                #    Numerics: the softmax intermediate runs in
                #    _score_softmax_dtype() (bf16 by default — half the fp32
                #    transient, same exponent range). Set
                #    STICKYKV_SCORE_SOFTMAX_DTYPE=float32 for the exact fp32
                #    reduction earlier baselines used.
                scaling = getattr(module, "scaling", head_dim ** -0.5)
                sm_dtype = _score_softmax_dtype()
                token_scores = torch.zeros(
                    B, H_kv, num_groups, S, device=q.device, dtype=q.dtype
                )
                chunk = _prefill_score_chunk()
                for start in range(0, T, chunk):
                    end = min(start + chunk, T)
                    q_blk = q5[:, :, :, start:end, :]                # [B,H_kv,rep,blk,D]
                    aw = torch.matmul(q_blk, k_t) * scaling          # [B,H_kv,rep,blk,S]

                    # Causal mask for this block: the global query row (start+r)
                    # sits at absolute position S-T+start+r and may attend to
                    # keys 0..S-T+start+r. Generation (T==1) needs no mask.
                    # The [blk, S] mask broadcasts over the leading head dims.
                    if T > 1:
                        blk = end - start
                        causal = torch.triu(
                            torch.ones(
                                blk, S, device=aw.device, dtype=torch.bool
                            ),
                            diagonal=S - T + start + 1,
                        )
                        aw = aw.masked_fill(causal, float("-inf"))

                    aw = F.softmax(aw, dim=-1, dtype=sm_dtype).to(q.dtype)
                    token_scores += aw.sum(dim=-2)                   # [B,H_kv,rep,S]
                    del aw

                # Merge (H_kv, n_rep) back to H_q in the original head order.
                token_scores = token_scores.reshape(B, H_q, S)

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
