"""MiKV's quantizer and channel balancer (Yang et al., 2024, §3.1-3.2).

Everything here is the paper's arithmetic, stated once:

* **Eq. 1 — per-token asymmetric N-bit round-to-nearest.** For a group ``x``,
  ``alpha = (max(x) - min(x)) / (2^N - 1)``, ``beta = min(x)``, and
  ``x_hat = alpha * round((x - beta) / alpha) + beta``. A *group* is a
  contiguous run of ``group_size`` channels of one token of one head: the paper
  quantizes per token, and "impose[s] a group size of half of the attention
  head dimension, which mitigates the artifact of RoPE" — Llama's
  ``rotate_half`` pairs channel ``c`` with ``c + D/2``, so a group of ``D/2``
  never mixes the two halves a rotation couples.
* **Eq. 2 — the channel balancer.** Per (layer, head, channel),
  ``b = sqrt(max|q| / max|k|)`` over the prompt, computed once at prefill.
* **Eqs. 3-4 — applying it.** ``k_hat = I(k * b)`` and ``q_hat = q / b``, so
  ``q_hat . k_hat`` is the logit. This module applies the division to the
  *dequantized key* instead of the query — ``sum_c (q_c / b_c) k_hat_c ==
  sum_c q_c (k_hat_c / b_c)`` exactly — which keeps the model's own attention
  untouched. Values are never balanced; the balancer is a key/query device.

Storage is real, not simulated: codes are bit-packed (8 codes -> ``bits``
bytes), and ``alpha`` / ``beta`` are stored in fp16, one pair per group. That is
the paper's accounting — at ``D = 128`` and ``group_size = 64`` it is 0.5 extra
bits per element, which is what makes its Table 3 sizes (33 / 23 / 18 / 16 %)
come out exactly (see :func:`bits_per_element`).

The codes are fit against the **fp16-stored** grid, not the fp32 intermediate,
so the grid a token was quantized against is bit-identical to the grid every
later read decodes with.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

#: Bit widths a tier may be stored at. 16 means "not quantized" (the model's own
#: dtype); everything else is packed codes plus an fp16 (alpha, beta) grid.
SUPPORTED_BITS = (2, 3, 4, 8, 16)

#: Bytes of one stored grid entry (alpha or beta). fp16, as in the paper.
GRID_BYTES = 2

#: Smallest positive fp16 subnormal. A range that rounds to zero in fp16 would
#: decode every code in its group to beta; clamping alpha up to this keeps the
#: division in the fit finite.
_FP16_TINY = 2.0 ** -24

#: Bounds on a balancer entry. A channel whose key or query never leaves zero
#: over the prompt carries no information about the ratio, so it gets 1.0; the
#: clamp only guards the division against denormal maxima.
BALANCER_MIN = 1e-4
BALANCER_MAX = 1e4


# ---------------------------------------------------------------------------
# bit packing — 8 codes into ``bits`` bytes
# ---------------------------------------------------------------------------


def packed_bytes(n_codes: int, bits: int) -> int:
    """Bytes that ``n_codes`` codes of ``bits`` bits occupy once packed."""
    if n_codes % 8:
        raise ValueError(f"n_codes must be a multiple of 8, got {n_codes}")
    return n_codes * bits // 8


def pack_codes(codes: Tensor, bits: int) -> Tensor:
    """Pack uint8 codes ``[..., n]`` (each ``< 2**bits``) into ``[..., n*bits/8]``.

    Eight consecutive codes form one ``8*bits``-bit little-endian word, emitted
    as ``bits`` bytes; code ``j`` of the eight occupies bits
    ``[bits*j, bits*(j+1))``. Any ``bits`` in 1..8 packs with no padding, which
    is what lets int3 cost 3 bits and not 4.
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in 1..8, got {bits}")
    n = codes.shape[-1]
    if n % 8:
        raise ValueError(f"last dim must be a multiple of 8, got {n}")
    if bits == 8:
        return codes.to(torch.uint8)
    if 8 % bits == 0:
        return _pack_whole_bytes(codes, bits)
    return _pack_words(codes, bits)


def _pack_whole_bytes(codes: Tensor, bits: int) -> Tensor:
    """1/2/4 bits: every byte holds ``8/bits`` whole codes, lowest code lowest.

    Byte-for-byte the layout :func:`_pack_words` produces (a 64-bit word's
    bytes, little-endian, are exactly these bytes), in uint8 throughout.
    """
    per = 8 // bits
    lead, n = codes.shape[:-1], codes.shape[-1]
    c = codes.to(torch.uint8).reshape(*lead, n // per, per)
    out = c[..., 0].clone()
    for j in range(1, per):
        out |= c[..., j] << (bits * j)
    return out


def _pack_words(codes: Tensor, bits: int) -> Tensor:
    """Any width: eight codes form one ``8*bits``-bit word, emitted as bytes."""
    n = codes.shape[-1]
    lead = codes.shape[:-1]
    c = codes.to(torch.int64).reshape(*lead, n // 8, 8)
    shifts = torch.arange(8, device=codes.device, dtype=torch.int64) * bits
    word = (c << shifts).sum(-1, keepdim=True)  # [..., n/8, 1], < 2**(8*bits)
    byte_shifts = torch.arange(bits, device=codes.device, dtype=torch.int64) * 8
    out = (word >> byte_shifts) & 0xFF  # [..., n/8, bits]
    return out.to(torch.uint8).reshape(*lead, n * bits // 8)


def unpack_codes(packed: Tensor, bits: int, n: int) -> Tensor:
    """Inverse of :func:`pack_codes`: ``[..., n*bits/8]`` -> uint8 ``[..., n]``."""
    if bits == 8:
        return packed
    lead = packed.shape[:-1]
    if 8 % bits == 0:
        # uint8 throughout: the int64 word path would make an 8-bytes-per-code
        # transient of the whole tier on every read.
        per = 8 // bits
        shifts = torch.arange(per, device=packed.device, dtype=torch.uint8) * bits
        codes = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
        return codes.reshape(*lead, n)
    p = packed.to(torch.int64).reshape(*lead, n // 8, bits)
    byte_shifts = torch.arange(bits, device=packed.device, dtype=torch.int64) * 8
    word = (p << byte_shifts).sum(-1, keepdim=True)  # [..., n/8, 1]
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * bits
    codes = (word >> shifts) & ((1 << bits) - 1)  # [..., n/8, 8]
    return codes.to(torch.uint8).reshape(*lead, n)


# ---------------------------------------------------------------------------
# Eq. 1 — per-token, group-wise asymmetric round-to-nearest
# ---------------------------------------------------------------------------


@dataclass
class QTensor:
    """A quantized ``[..., D]`` tensor: packed codes plus its fp16 grid.

    ``codes`` is ``[..., D*bits/8]`` uint8; ``alpha`` and ``beta`` are
    ``[..., D/group_size]`` fp16.
    """

    codes: Tensor
    alpha: Tensor
    beta: Tensor
    bits: int
    group_size: int
    dim: int

    def nbytes(self) -> int:
        return (self.codes.numel() * self.codes.element_size()
                + self.alpha.numel() * self.alpha.element_size()
                + self.beta.numel() * self.beta.element_size())


def quantize(x: Tensor, bits: int, group_size: int) -> QTensor:
    """Quantize the last axis of ``x`` per Eq. 1, one grid per ``group_size`` run."""
    if bits not in SUPPORTED_BITS or bits == 16:
        raise ValueError(f"quantize() takes bits in {SUPPORTED_BITS[:-1]}, got {bits}")
    d = x.shape[-1]
    if d % group_size:
        raise ValueError(f"head_dim {d} is not a multiple of group_size {group_size}")
    if group_size % 8:
        raise ValueError(f"group_size must be a multiple of 8, got {group_size}")
    lead = x.shape[:-1]
    g = x.float().reshape(*lead, d // group_size, group_size)
    mn = g.amin(-1)
    mx = g.amax(-1)
    levels = float((1 << bits) - 1)
    # Round the grid to its stored precision FIRST, then fit the codes to it.
    alpha16 = ((mx - mn) / levels).clamp_min(_FP16_TINY).to(torch.float16)
    beta16 = mn.to(torch.float16)
    alpha = alpha16.float().unsqueeze(-1)
    beta = beta16.float().unsqueeze(-1)
    codes = torch.round((g - beta) / alpha).clamp_(0.0, levels).to(torch.uint8)
    return QTensor(
        codes=pack_codes(codes.reshape(*lead, d), bits),
        alpha=alpha16,
        beta=beta16,
        bits=bits,
        group_size=group_size,
        dim=d,
    )


def dequantize(q: QTensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """``alpha * code + beta``, shaped ``[..., D]``."""
    lead = q.codes.shape[:-1]
    codes = unpack_codes(q.codes, q.bits, q.dim).float()
    codes = codes.reshape(*lead, q.dim // q.group_size, q.group_size)
    x = codes * q.alpha.float().unsqueeze(-1) + q.beta.float().unsqueeze(-1)
    return x.reshape(*lead, q.dim).to(dtype)


def fake_quantize(x: Tensor, bits: int, group_size: int) -> Tensor:
    """``dequantize(quantize(x))`` in ``x``'s dtype, or ``x`` itself at 16 bits."""
    if bits == 16:
        return x
    return dequantize(quantize(x, bits, group_size), x.dtype)


# ---------------------------------------------------------------------------
# Eq. 2 — the channel balancer
# ---------------------------------------------------------------------------


def channel_balancer(q_absmax: Tensor, k_absmax: Tensor) -> Tensor:
    """``b = sqrt(max|q| / max|k|)`` per channel (Eq. 2), in fp32.

    ``q_absmax`` / ``k_absmax`` are the per-channel maxima over the prompt, both
    ``[..., D]`` on the same (KV-head) grid. A channel where either maximum is
    zero is left at 1.0: the ratio is undefined there, and 1.0 is the identity.

    With ``b`` applied, ``max|q/b| == max|k*b| == sqrt(max|q| * max|k|)`` — the
    SmoothQuant alpha = 0.5 point: an outlier channel's magnitude is split
    evenly between the fp16 query and the quantized key, so the key's per-token
    range stops being set by one channel.
    """
    q = q_absmax.float()
    k = k_absmax.float()
    ok = (q > 0) & (k > 0)
    ratio = torch.where(ok, q / torch.where(ok, k, torch.ones_like(k)),
                        torch.ones_like(q))
    return ratio.sqrt().clamp_(BALANCER_MIN, BALANCER_MAX)


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------


def bits_per_element(bits: int, group_size: int, head_dim: int = 128) -> float:
    """Stored bits per K/V element at ``bits``, including the fp16 grid.

    16 means unquantized: exactly 16. Otherwise ``bits + 2 * 16 / group_size``
    (one fp16 alpha and one fp16 beta per group). ``head_dim`` only validates.
    """
    if bits == 16:
        return 16.0
    if head_dim % group_size:
        raise ValueError(f"head_dim {head_dim} not a multiple of group_size {group_size}")
    return bits + 8.0 * 2 * GRID_BYTES / group_size


def token_bytes(bits: int, group_size: int, head_dim: int,
                elem_bytes: int = 2) -> int:
    """Bytes one token occupies for ONE of K or V, one head, at ``bits``.

    ``elem_bytes`` is the model dtype's width, used at 16 bits only.
    """
    if bits == 16:
        return head_dim * elem_bytes
    return packed_bytes(head_dim, bits) + 2 * GRID_BYTES * (head_dim // group_size)
