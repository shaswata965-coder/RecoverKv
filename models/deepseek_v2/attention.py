"""The owned MLA attention forward — RecoverKV's seam into DeepSeek-V2.

Why the forward is owned instead of patched
-------------------------------------------
On Llama the cache reaches attention through a monkeypatch of the module-global
``flash_attn_func`` in ``transformers.modeling_flash_attention_utils``. DeepSeek's
remote code imports ``flash_attn_func`` **into its own namespace**
(``modeling_deepseek.py`` l.62), so that patch would be a silent no-op: the
fused path would never fire and every run would look like success. It also
slices its RoPE table by the cache's held length, which an evicting cache makes
shorter than the absolute position (``rope.py``). So this module replaces the
attention ``forward`` outright, on every layer, at load
(``loader.install_recoverkv_mla``).

What it computes
----------------
* **Prefill** (``q_len > 1``) runs the non-absorbed form upstream runs: per-head
  K = ``[W_UK c ‖ k_pe]`` (192), V = ``W_UV c`` (128), causal. The eviction
  score is each key's received softmax mass summed over the prompt's query rows,
  per query head -- the H2O cumulative score the method ranks on -- reduced to
  windows and handed to the cache exactly where the Llama score hooks hand it
  (``cache.cache_kwargs[layer]["window_scores"]``, read on the next
  ``update()``).
* **Decode** (``q_len == 1``) runs the absorbed form over the cache's latent:
  ``q~ = [W_UK^T q_nope ‖ q_pe]`` against packed keys ``[c ‖ r]``, values ``c``,
  output ``W_UV`` applied per head. Mathematically the same logits as upstream.
  When the cache armed the fused path (CUDA, a non-empty int2 tier), the step is
  served by ``flash_decode._run_fused`` -- the same function, the same read
  gate, the same proof counters as on Llama -- through
  :func:`modules.windowed_cache.flash_decode.run_pending`. Otherwise the cache
  returned the materialized ``[sink ‖ body ‖ Q]`` latent and this computes the
  step in PyTorch: the CPU oracle, and the eager backend's ungated decode.

The cache is handed ``(key, key)``: the value aliases the key's latent dims
(``geometry.MLACacheConfig.kv_value_alias``), so the store keeps one tensor.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.windowed_cache import flash_decode
from modules.windowed_cache.hooks import SCORE_ACCUM_DTYPE
from modules.windowed_cache.scorer import (
    reduce_token_scores_to_windows,
    reduce_two_tier_scores,
)

from .geometry import MLAGeometry
from .rope import apply_rotary, cos_sin, ensure_table


#: Per-step evidence of which attention path served each call, keyed by path.
#: Python ints only (no device sync). Read by tests and by the runners'
#: sidecars, next to ``flash_decode.stats()``.
_PATH_STATS = {"prefill_eager": 0, "prefill_flash": 0, "decode_materialize": 0,
               "decode_fused": 0, "decode_latent_fullkv": 0, "no_cache": 0}


def path_stats() -> dict:
    return dict(_PATH_STATS)


def reset_path_stats() -> None:
    for k in _PATH_STATS:
        _PATH_STATS[k] = 0


def _is_windowed(cache: Any) -> bool:
    from modules.windowed_cache.cache import WindowedCache
    return isinstance(cache, WindowedCache)


def _absorb_weights(module) -> Tuple[Tensor, Tensor]:
    """``(W_UK, W_UV)`` ``[H, d, Dc]`` views of ``kv_b_proj.weight``.

    ``kv_b_proj`` is ``Linear(Dc, H * (nope + v))``, so ``weight`` is
    ``[H*(nope+v), Dc]`` and ``k_nope,h = W_UK,h c``, ``v_h = W_UV,h c``.
    """
    g: MLAGeometry = module._recoverkv_geometry
    w = module.kv_b_proj.weight.view(g.num_heads, g.qk_nope_head_dim + g.v_head_dim,
                                     g.kv_lora_rank)
    return w[:, :g.qk_nope_head_dim], w[:, g.qk_nope_head_dim:]


def _reject_padding_decode(attention_mask: Optional[Tensor], flash: bool) -> None:
    """Equal-length prompts only, exactly as on Llama.

    On the flash attention implementation the upstream model already nulls the
    mask when it has no zeros, so a non-``None`` mask IS padding. On eager the
    mask is the 4-D additive one, and any negative entry is a masked key.
    """
    if attention_mask is None:
        return
    if flash or bool((attention_mask < 0).any()):
        raise RuntimeError(
            "a padding attention mask reached the RecoverKV MLA decode. The cache "
            "models equal-length prompts only (BATCHING_PLAN.md §4) -- a padded "
            "row would attend over keys it must not see. Use equal-length prompts.")


def _write_scores(cache, layer_idx: int, token_scores: Tensor, score_meta) -> None:
    """Per-key mass -> per-window scores -> the cache, as the score hooks do."""
    ws = cache.resolved.window_size
    num_sink = cache.resolved.num_sink_tokens
    if score_meta is not None:
        order, q_token_len = score_meta
        scores = reduce_two_tier_scores(token_scores, num_sink, ws, q_token_len, order)
    else:
        scores = reduce_token_scores_to_windows(token_scores, num_sink, ws)
    cache.cache_kwargs[layer_idx]["window_scores"] = scores.to(SCORE_ACCUM_DTYPE)


def _prefill_eager(module, q_nope, q_pe, k_pe, c, attention_mask):
    """Upstream's eager attention, op for op, plus the per-key score."""
    g: MLAGeometry = module._recoverkv_geometry
    bsz, H, q_len, _ = q_nope.shape
    kv = (module.kv_b_proj(c)
          .view(bsz, q_len, H, g.qk_nope_head_dim + g.v_head_dim)
          .transpose(1, 2))
    k_nope, value_states = torch.split(kv, [g.qk_nope_head_dim, g.v_head_dim], dim=-1)

    query_states = k_pe.new_empty(bsz, H, q_len, g.q_head_dim)
    query_states[:, :, :, :g.qk_nope_head_dim] = q_nope
    query_states[:, :, :, g.qk_nope_head_dim:] = q_pe
    key_states = k_pe.new_empty(bsz, H, q_len, g.q_head_dim)
    key_states[:, :, :, :g.qk_nope_head_dim] = k_nope
    key_states[:, :, :, g.qk_nope_head_dim:] = k_pe

    attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3))
                    * module.softmax_scale)
    if attention_mask is None:
        mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device, dtype=attn_weights.dtype)
        attention_mask = torch.triu(mask, diagonal=1)[None, None]
    elif attention_mask.shape[-1] != q_len or attention_mask.shape[-2] != q_len:
        raise RuntimeError(
            f"prefill attention mask {tuple(attention_mask.shape)} does not cover "
            f"the {q_len}-token prompt; the RecoverKV MLA prefill expects the "
            "cache to be empty before it (one prompt, then single-token decode).")
    attn_weights = attn_weights + attention_mask
    probs32 = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights = probs32.to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)            # [B,H,T,v]
    token_scores = probs32.sum(dim=2)                                  # [B,H,S]
    return attn_output, token_scores, attn_weights


def _flash_prefill(module, hidden_states: Tensor) -> bool:
    """Whether this prefill runs FA2 (flash attention implementation, on CUDA)."""
    return bool(getattr(module, "_recoverkv_flash", False)) and hidden_states.is_cuda


def _prefill_flash(module, q_nope, q_pe, k_pe, c, score: bool = True):
    """FA2 over the decompressed heads; the score kernel reuses its LSE.

    ``flash_attn_func`` is called HERE, by name, with ``return_attn_probs=True``
    for ``softmax_lse`` -- the L-reuse contract (README §2): the prefill score is
    never a second O(N^2) pass to recompute L.

    ``score=False`` skips the score kernel and returns ``None`` scores. Only a
    RecoverKV cache consumes prefill scores; the latent FullKV baseline does not,
    and running the score kernel there anyway costs a second O(N^2) pass per
    layer that inflates the baseline's prefill time (TTFT) and changes nothing
    it outputs.
    """
    from flash_attn import flash_attn_func
    from modules.windowed_cache.score_kernel import compute_token_scores
    from modules.windowed_cache.hooks import PREFILL_SCORE_CHUNK, SCORE_SOFTMAX_DTYPE

    g: MLAGeometry = module._recoverkv_geometry
    bsz, H, q_len, _ = q_nope.shape
    kv = (module.kv_b_proj(c)
          .view(bsz, q_len, H, g.qk_nope_head_dim + g.v_head_dim)
          .transpose(1, 2))
    k_nope, value_states = torch.split(kv, [g.qk_nope_head_dim, g.v_head_dim], dim=-1)
    query_states = torch.cat([q_nope, q_pe], dim=-1)                   # [B,H,T,192]
    key_states = torch.cat([k_nope, k_pe.expand(-1, H, -1, -1)], dim=-1)
    v_pad = F.pad(value_states, [0, g.q_head_dim - g.v_head_dim])
    out, lse, _ = flash_attn_func(
        query_states.transpose(1, 2), key_states.transpose(1, 2),
        v_pad.transpose(1, 2), 0.0, softmax_scale=module.softmax_scale,
        causal=True, return_attn_probs=True)
    attn_output = out[..., :g.v_head_dim].transpose(1, 2)              # [B,H,T,v]
    if not score:
        return attn_output, None
    # The score kernel tiles the head with `tl.arange(0, HEAD_DIM)`, which wants
    # a power of two; the decompressed head is 192. Zero lanes change no dot
    # product, so padding to 256 scores exactly the logits flash normalised with
    # `lse` -- the kernel itself is the shipped one, unmodified.
    d_pad = 1 << (g.q_head_dim - 1).bit_length()
    if d_pad != g.q_head_dim:
        query_states = F.pad(query_states, [0, d_pad - g.q_head_dim])
        key_states = F.pad(key_states, [0, d_pad - g.q_head_dim])
    token_scores = compute_token_scores(
        query_states, key_states, module.softmax_scale, lse=lse,
        chunk=PREFILL_SCORE_CHUNK, softmax_dtype=SCORE_SOFTMAX_DTYPE,
        out_dtype=SCORE_ACCUM_DTYPE)
    return attn_output, token_scores


def _decode_absorbed(module, q_nope, q_pe, k_eff):
    """Absorbed decode over a latent ``k_eff`` ``[B, 1, S, Dc+Dr]``.

    Returns the per-head output ``[B, H, 1, v]`` and the fp32 probabilities
    ``[B, H, 1, S]``.
    """
    g: MLAGeometry = module._recoverkv_geometry
    w_uk, w_uv = _absorb_weights(module)
    q_c = torch.einsum("bhtd,hdc->bhtc", q_nope, w_uk)                # [B,H,1,Dc]
    q_t = g.pack(q_c, q_pe)                                            # [B,H,1,576]
    logits = torch.matmul(q_t, k_eff.transpose(-1, -2)) * module.softmax_scale
    probs32 = F.softmax(logits, dim=-1, dtype=torch.float32)
    o_packed = torch.matmul(probs32.to(k_eff.dtype), k_eff)            # [B,H,1,576]
    o_c = g.latent_of(o_packed)                                        # [B,H,1,Dc]
    o = torch.einsum("bhtc,hvc->bhtv", o_c, w_uv)                      # [B,H,1,v]
    return o, probs32


def _decode_fused(module, q_nope, q_pe, k_fp):
    """The armed fused step: ``flash_decode._run_fused`` with the absorbed query."""
    g: MLAGeometry = module._recoverkv_geometry
    w_uk, w_uv = _absorb_weights(module)
    q_c = torch.einsum("bhtd,hdc->bhtc", q_nope, w_uk)
    q_t = g.pack(q_c, q_pe)                                            # [B,H,1,576]
    k_flash = k_fp.transpose(1, 2)                                     # [B,S,1,576]
    # The value aliases the key; the kernel reads V from the same tile.
    out = flash_decode.run_pending(q_t.transpose(1, 2), k_flash, k_flash)
    o_packed = out.transpose(1, 2)                                     # [B,H,1,576]
    o_c = g.latent_of(o_packed)
    return torch.einsum("bhtc,hvc->bhtv", o_c.to(w_uv.dtype), w_uv)


def mla_attention_forward(
    self,
    hidden_states: Tensor,
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """Drop-in for ``DeepseekV2Attention.forward`` / ``DeepseekV2FlashAttention2.forward``.

    Same signature and return shape as upstream: ``(attn_output, attn_weights,
    past_key_value)``. ``attn_weights`` are the probabilities at the attention
    dtype, returned only with ``output_attentions`` on the eager paths (prefill
    eager, materialized/latent-FullKV decode) -- upstream's flash class returns
    none either.
    """
    g: MLAGeometry = self._recoverkv_geometry
    flash = bool(getattr(self, "_recoverkv_flash", False))
    bsz, q_len, _ = hidden_states.shape
    H = g.num_heads
    if position_ids is None:
        raise RuntimeError(
            "the RecoverKV MLA forward needs explicit ABSOLUTE position_ids. "
            "generate() supplies them (attention_mask.cumsum - 1); a hand-written "
            "loop must pass them, because an evicting cache's length is not the "
            "next token's position.")
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.shape[0] != bsz:
        position_ids = position_ids.expand(bsz, -1)

    # --- projections: upstream, unchanged -----------------------------------
    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, H, g.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [g.qk_nope_head_dim, g.qk_rope_head_dim], dim=-1)
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [g.kv_lora_rank, g.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, g.qk_rope_head_dim).transpose(1, 2)
    c = self.kv_a_layernorm(compressed_kv)                             # [B,T,Dc]

    # --- RoPE at absolute positions ------------------------------------------
    cache = past_key_value
    seen = 0
    if _is_windowed(cache):
        seen = int(cache._tokens_seen)
    elif cache is not None:
        seen = int(cache.get_seq_length(self.layer_idx))
    ensure_table(self.rotary_emb, seen + q_len, hidden_states.device,
                 hidden_states.dtype)
    cos, sin = cos_sin(self.rotary_emb, position_ids, q_pe.dtype)
    q_pe, k_pe = apply_rotary(q_pe, k_pe, cos, sin)                   # de-interleaved

    key_packed = g.pack(c.unsqueeze(1), k_pe)                          # [B,1,T,576]

    # --- no cache: upstream math, nothing stored -----------------------------
    if cache is None:
        _PATH_STATS["no_cache"] += 1
        if q_len == 1:
            raise RuntimeError("a single-token call without a cache has nothing to attend")
        attn_output, _, w = _prefill_eager(self, q_nope, q_pe, k_pe, c, attention_mask)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, H * g.v_head_dim)
        return self.o_proj(attn_output), (w if output_attentions else None), cache

    windowed = _is_windowed(cache)
    if windowed:
        k_eff, _ = cache.update(key_packed, key_packed, self.layer_idx,
                                {"cache_position": position_ids})
    else:
        # Latent FullKV baseline: any HF Cache (DynamicCache) stores the packed
        # latent as one head and a zero-width value, so the baseline holds the
        # same 1,152 B/token the method is priced against -- not upstream's
        # decompressed 12,288 B -- and decodes with the same absorbed math.
        k_eff, _ = cache.update(key_packed, key_packed[..., :0], self.layer_idx,
                                {"cache_position": position_ids})

    weights = None              # returned only on the eager paths, as upstream does
    if q_len > 1:
        if k_eff.shape[2] != q_len:
            raise RuntimeError(
                f"layer {self.layer_idx}: a {q_len}-token prefill found "
                f"{k_eff.shape[2] - q_len} keys already cached. The RecoverKV MLA "
                "path models one prompt followed by single-token decode.")
        if _flash_prefill(self, hidden_states):
            _PATH_STATS["prefill_flash"] += 1
            attn_output, token_scores = _prefill_flash(self, q_nope, q_pe, k_pe, c,
                                                       score=windowed)
        else:
            _PATH_STATS["prefill_eager"] += 1
            attn_output, token_scores, weights = _prefill_eager(
                self, q_nope, q_pe, k_pe, c, attention_mask)
        if windowed:
            _write_scores(cache, self.layer_idx, token_scores, None)
    else:
        _reject_padding_decode(attention_mask, flash)
        if windowed and flash_decode.pending() is not None:
            _PATH_STATS["decode_fused"] += 1
            attn_output = _decode_fused(self, q_nope, q_pe, k_eff)
        else:
            attn_output, probs32 = _decode_absorbed(self, q_nope, q_pe, k_eff)
            weights = probs32.to(hidden_states.dtype)
            if windowed:
                _PATH_STATS["decode_materialize"] += 1
                meta = cache._last_score_meta[self.layer_idx]
                cache._last_score_meta[self.layer_idx] = None
                cache._last_effective_k[self.layer_idx] = None
                _write_scores(cache, self.layer_idx, probs32[:, :, 0, :], meta)
            else:
                _PATH_STATS["decode_latent_fullkv"] += 1

    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, H * g.v_head_dim)
    attn_output = self.o_proj(attn_output.to(hidden_states.dtype))
    # The parity / observation runners read `out.attentions` (eager only). On a
    # windowed decode the key axis is the cache's effective [sink|body|Q] layout,
    # exactly what the Llama eager path hands them.
    return attn_output, (weights if output_attentions else None), cache
