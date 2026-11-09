"""DeepSeek-V2 RoPE, restated for an evicting cache.

Two things upstream does that an evicting cache cannot live with, and what
replaces them here:

* **The table is sliced by the held length.** Upstream computes
  ``kv_seq_len = q_len + past_key_value.get_usable_length(...)`` and then
  ``cos_cached[:kv_seq_len][position_ids]``. After the first eviction the cache
  holds fewer keys than the absolute position of the next token, so the slice is
  shorter than the index: a device-side assert on CUDA, which poisons the
  context (README §3). Here the table is indexed by **absolute position** and
  never sliced; :func:`ensure_table` grows it from a host int, never a sync.
* **The rotation is interleaved.** Upstream de-interleaves the RoPE dims
  (``view(d/2, 2).transpose``) before HF's half-split ``rotate_half``, so the
  post-RoPE key is in de-interleaved order. That order is what the cache stores
  and what the packed layout (``geometry.py``) is built on.

Everything here is bit-identical to upstream at the same dtype:
``tests/test_deepseek_mla.py`` compares :func:`apply_rotary` against the vendored
``apply_rotary_pos_emb`` and :class:`PackedRope` against it through the cache's
own ``rotate_key_window`` / ``unrotate_key_window``.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from .geometry import MLAGeometry


def deinterleave(x: Tensor) -> Tensor:
    """Upstream's ``x.view(..., d // 2, 2).transpose(-1, -2).reshape(..., d)``."""
    *lead, d = x.shape
    return x.reshape(*lead, d // 2, 2).transpose(-1, -2).reshape(*lead, d)


def rotate_half(x: Tensor) -> Tensor:
    """Upstream's (and HF's) ``rotate_half``: ``cat(-x2, x1)``."""
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def ensure_table(rotary_emb: nn.Module, needed_len: int, device, dtype) -> None:
    """Grow upstream's cached tables to cover ``needed_len`` positions.

    ``needed_len`` is a HOST int (absolute positions seen + this step's tokens),
    so this costs no device sync. Upstream builds the tables to
    ``max_position_embeddings`` (163,840 on V2-Lite) at construction, so in
    practice this never fires; it exists so a longer run regrows exactly as
    upstream's own ``forward`` would, rather than indexing past the end.
    """
    cached = getattr(rotary_emb, "max_seq_len_cached", None)
    if cached is None or needed_len > cached:
        # Upstream's own forward regrows with `dtype=x.dtype`; so does this.
        rotary_emb._set_cos_sin_cache(seq_len=needed_len, device=device,
                                      dtype=dtype)


def cos_sin(rotary_emb: nn.Module, position_ids: Tensor, dtype) -> Tuple[Tensor, Tensor]:
    """``cos, sin`` ``[B, T, Dr]`` at ABSOLUTE ``position_ids`` ``[B, T]``.

    Upstream returns ``cos_cached[:seq_len].to(x.dtype)`` and then indexes it by
    ``position_ids``; casting then indexing equals indexing then casting element
    for element, so this is the same tensor without the slice.
    """
    pos = position_ids.to(torch.long)
    cos = rotary_emb.cos_cached[pos].to(dtype)
    sin = rotary_emb.sin_cached[pos].to(dtype)
    return cos, sin


def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Upstream ``apply_rotary_pos_emb`` given already-indexed ``cos``/``sin``.

    ``q`` ``[B, H, T, Dr]``, ``k`` ``[B, 1, T, Dr]``, ``cos``/``sin``
    ``[B, T, Dr]``. Returns both post-RoPE, in de-interleaved order. Same ops in
    the same order as upstream, so the result is bit-identical.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = deinterleave(q)
    k = deinterleave(k)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PackedRope(nn.Module):
    """The ``rope_module`` the RecoverKV cache is constructed with, for MLA.

    The cache calls ``rope_module(ref, position_ids)`` and expects HF-shaped
    ``(cos, sin)`` ``[B, T, D]`` for its full-head half-split rotation
    (``effective._apply_rotary_one``, ``decode_kernel.rope_cos_sin_halves``).
    Over the packed layout this returns ``cos = 1, sin = 0`` on every latent dim
    and DeepSeek's ``cos``/``sin`` on every RoPE dim, so those primitives rotate
    ``r`` and leave ``c`` bit-exact (``x*1 + y*0 == x``).

    No ``torch.cat``: this runs inside the compiled eviction (``unrotate`` at
    demotion, ``rotate`` at promotion), where a cat is lowered through Inductor's
    ``pointwise_cat`` on CUDA -- the source of the ``Identity`` node that once
    made the whole eviction unlowerable (``effective._rotate_half_no_cat``). The
    packed tables are a column gather plus a ``where`` instead.

    The upstream rotary module is shared by reference, so its tables are the
    model's own; nothing is recomputed.
    """

    def __init__(self, rotary_emb: nn.Module, geometry: MLAGeometry) -> None:
        super().__init__()
        self.rotary_emb = rotary_emb
        self.geometry = geometry
        self.register_buffer("freq_index", geometry.rope_freq_index(), persistent=False)
        self.register_buffer("rope_mask", geometry.rope_mask(), persistent=False)

    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        cos_r, sin_r = cos_sin(self.rotary_emb, position_ids, x.dtype)   # [B, T, Dr]
        idx = self.freq_index.to(cos_r.device)
        mask = self.rope_mask.to(cos_r.device)
        cos = torch.where(mask, cos_r.index_select(-1, idx), 1.0)
        sin = torch.where(mask, sin_r.index_select(-1, idx), 0.0)
        return cos.to(x.dtype), sin.to(x.dtype)
