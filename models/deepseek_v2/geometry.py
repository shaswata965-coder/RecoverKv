"""MLA geometry: how DeepSeek-V2's latent maps onto the RecoverKV cache.

What the cache stores, per token per layer
------------------------------------------
DeepSeek-V2 attention is Multi-head Latent Attention. Per token it produces one
``kv_lora_rank``-wide latent ``c`` (512 on V2-Lite, after ``kv_a_layernorm``)
and one ``qk_rope_head_dim``-wide RoPE key ``r`` (64), and **every** query head
reads both. In the absorbed form a decode step uses::

    q~_h      = [ W_UK,h^T q_nope,h  ||  q_pe,h ]          (Dc + Dr wide)
    key_t     = [ c_t || r_t ]                              (Dc + Dr wide)
    value_t   = c_t                                         (Dc wide)
    logit_h,t = softmax_scale * q~_h . key_t
    o_h       = W_UV,h  sum_t p_h,t c_t

So, as far as the cache is concerned, an MLA layer is **one KV head**
(``H_kv = 1``) of width ``Dc + Dr = 576``, shared by all ``H_q = 16`` query
heads (``rep = 16``), whose **value is the key's latent dims**. That is the whole
mapping. The cache's eviction, tiers, promotion, positions, card and gate are
then the shipped ones, unchanged; the only things the cache learns are that
there is one head of width 576 and that the value aliases the key
(``kv_value_alias``), so it stores ``c`` once.

The packed key layout
---------------------
The cache's RoPE machinery (``modules/quant/effective.py``, and the decode
kernel's in-register RoPE) rotates a head with HF's **half-split** convention:
element ``i`` pairs with ``i + D/2``. DeepSeek rotates only the last ``Dr``
dims, after de-interleaving them (``rope.deinterleave``). Laying the key out as::

    [ c[:Dc/2] | r[:Dr/2] | c[Dc/2:] | r[Dr/2:] ]          halves of (Dc+Dr)/2

makes the half-split pairing exactly right: latent element ``j`` pairs with
latent element ``j + Dc/2`` and gets ``cos = 1, sin = 0`` (an exact identity in
IEEE arithmetic, ``x*1 + y*0 == x``), and RoPE element ``j`` pairs with RoPE
element ``j + Dr/2`` and gets DeepSeek's own ``cos``/``sin``. Every existing
rotate/unrotate primitive therefore rotates **only** ``r``, bit-for-bit as the
upstream ``apply_rotary_pos_emb`` does (``tests/test_deepseek_mla.py`` pins it).
A dot product is invariant under a permutation applied to both operands, so the
query is packed the same way and nothing else notices.

At V2-Lite's dims both halves are ``256 + 32``: whole 32-entry grid groups
(``quantizer.GRID_GROUP``), so no int2 grid group straddles latent and RoPE dims.

The budget
----------
``bytes_per_token = (Dc + Dr) * elem = 1152`` B at bf16, because the value
aliases the key. **Never** price the cache from ``config.num_key_value_heads``
(16 on V2-Lite): that field describes the decompressed per-head path and would
price 8,192 B/token, 7.1x the latent. :meth:`MLAGeometry.cache_config` is the
only config object the cache is handed, and the resolver refuses a raw
DeepSeek config (``modules/windowed_cache/config.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Upstream's ``yarn_get_mscale`` (modeling_deepseek.py), restated exactly."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def softmax_scale_from_config(config: Any) -> float:
    """``q_head_dim ** -0.5 * mscale**2``, exactly as upstream's ``__init__``.

    0.11472 on V2-Lite. The module's own ``softmax_scale`` attribute is the
    authority at runtime; this exists so a test can pin that the two agree and
    so a geometry can be built from a config alone.
    """
    q_head_dim = int(config.qk_nope_head_dim) + int(config.qk_rope_head_dim)
    s = q_head_dim ** (-0.5)
    rs = getattr(config, "rope_scaling", None)
    if rs is not None:
        mscale_all_dim = rs.get("mscale_all_dim", 0)
        factor = rs["factor"]
        if mscale_all_dim:
            m = yarn_get_mscale(factor, mscale_all_dim)
            s = s * m * m
    return s


@dataclass(frozen=True)
class MLACacheConfig:
    """The model-config view the RecoverKV cache is constructed with.

    ``WindowedCacheConfig.resolve`` and ``WindowedCache.__init__`` read
    ``num_key_value_heads`` / ``head_dim`` off the model config. For MLA those
    must describe the latent (one head of width ``Dc + Dr``), not the
    decompressed heads, so the cache never sees the raw DeepSeek config.

    ``kv_value_alias`` tells the cache the value is the key's latent dims, so the
    fp tier stores one tensor and the int2 tier one set of codes.
    ``attn_softmax_scale`` replaces the cache's ``head_dim ** -0.5`` default,
    which is wrong here by 2.75x at 576 (and 1.59x at the per-head 192).
    """

    num_key_value_heads: int
    num_attention_heads: int
    head_dim: int
    hidden_size: int
    num_hidden_layers: int
    kv_value_alias: bool
    attn_softmax_scale: float
    latent_dim: int
    rope_dim: int
    model_type: str = "deepseek_v2"
    mla_latent: bool = True


@dataclass(frozen=True)
class MLAGeometry:
    """Every MLA dimension the port needs, plus the packed key layout."""

    num_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    softmax_scale: float
    num_layers: int
    hidden_size: int

    def __post_init__(self) -> None:
        if self.kv_lora_rank % 2 or self.qk_rope_head_dim % 2:
            raise ValueError(
                "the packed key layout needs an even latent and an even RoPE "
                f"width, got kv_lora_rank={self.kv_lora_rank}, "
                f"qk_rope_head_dim={self.qk_rope_head_dim}")
        if self.key_dim % 4:
            raise ValueError(
                f"int2 crumb packing needs the key width {self.key_dim} to be a "
                "multiple of 4")

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "MLAGeometry":
        if getattr(config, "model_type", None) != "deepseek_v2":
            raise ValueError(
                f"MLAGeometry.from_config wants a deepseek_v2 config, got "
                f"model_type={getattr(config, 'model_type', None)!r}")
        if getattr(config, "q_lora_rank", None) not in (None, 0):
            # V2-Lite has no query compression. The owned forward handles the
            # q_lora path too, but nothing here has been checked against it.
            pass
        return cls(
            num_heads=int(config.num_attention_heads),
            kv_lora_rank=int(config.kv_lora_rank),
            qk_nope_head_dim=int(config.qk_nope_head_dim),
            qk_rope_head_dim=int(config.qk_rope_head_dim),
            v_head_dim=int(config.v_head_dim),
            softmax_scale=softmax_scale_from_config(config),
            num_layers=int(config.num_hidden_layers),
            hidden_size=int(config.hidden_size),
        )

    # -- derived sizes --------------------------------------------------------

    @property
    def q_head_dim(self) -> int:
        """Per-head query/key width on the decompressed path (192)."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def key_dim(self) -> int:
        """Latent key width, ``Dc + Dr`` (576). The cache's ``head_dim``."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def half(self) -> int:
        return self.key_dim // 2

    @property
    def half_c(self) -> int:
        return self.kv_lora_rank // 2

    @property
    def half_r(self) -> int:
        return self.qk_rope_head_dim // 2

    def bytes_per_token(self, elem: int) -> int:
        """Latent fp bytes per token per layer: ``(Dc + Dr) * elem``."""
        return self.key_dim * elem

    def cache_config(self, softmax_scale: float = None) -> MLACacheConfig:
        """The config view to hand ``WindowedCacheConfig.resolve`` / the cache."""
        return MLACacheConfig(
            num_key_value_heads=1,
            num_attention_heads=self.num_heads,
            head_dim=self.key_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            kv_value_alias=True,
            attn_softmax_scale=float(self.softmax_scale if softmax_scale is None
                                     else softmax_scale),
            latent_dim=self.kv_lora_rank,
            rope_dim=self.qk_rope_head_dim,
        )

    # -- packed layout --------------------------------------------------------

    def pack(self, latent: Tensor, rope: Tensor) -> Tensor:
        """``[..., Dc]`` and ``[..., Dr]`` -> ``[..., Dc + Dr]`` packed.

        ``rope`` must already be in DeepSeek's de-interleaved order (what
        upstream's ``apply_rotary_pos_emb`` produces and consumes). Broadcasts
        leading axes, so a ``[B, 1, T, Dr]`` RoPE key packs against a
        ``[B, H, T, Dc]`` latent query term.
        """
        hc, hr = self.half_c, self.half_r
        lead = torch.broadcast_shapes(latent.shape[:-1], rope.shape[:-1])
        latent = latent.expand(*lead, latent.shape[-1])
        rope = rope.expand(*lead, rope.shape[-1])
        return torch.cat(
            [latent[..., :hc], rope[..., :hr], latent[..., hc:], rope[..., hr:]],
            dim=-1)

    def latent_of(self, packed: Tensor) -> Tensor:
        """``[..., Dc + Dr]`` -> ``[..., Dc]``: the latent dims, in order."""
        hc, h = self.half_c, self.half
        return torch.cat([packed[..., :hc], packed[..., h:h + hc]], dim=-1)

    def rope_of(self, packed: Tensor) -> Tensor:
        """``[..., Dc + Dr]`` -> ``[..., Dr]``: the RoPE dims, de-interleaved order."""
        hc, h = self.half_c, self.half
        return torch.cat([packed[..., hc:h], packed[..., h + hc:]], dim=-1)

    def rope_mask(self, device=None) -> Tensor:
        """``[Dc + Dr]`` bool, True on packed RoPE dims."""
        m = torch.zeros(self.key_dim, dtype=torch.bool, device=device)
        hc, h = self.half_c, self.half
        m[hc:h] = True
        m[h + hc:] = True
        return m

    def rope_freq_index(self, device=None) -> Tensor:
        """``[Dc + Dr]`` long: the RoPE table column each packed dim reads.

        Packed RoPE dim ``hc + j`` and ``h + hc + j`` both read column ``j`` of a
        ``cat(freqs, freqs)`` table (its first half is the distinct part).
        Latent dims read column 0; :func:`rope.packed_cos_sin` masks them to
        ``cos = 1, sin = 0``.
        """
        idx = torch.zeros(self.key_dim, dtype=torch.long, device=device)
        hc, hr, h = self.half_c, self.half_r, self.half
        idx[hc:h] = torch.arange(hr, device=device)
        idx[h + hc:] = torch.arange(hr, device=device)
        return idx
