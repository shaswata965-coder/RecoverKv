"""Hand-rolled KIVI-style affine int4 quantizer (design.md §2).

Numerics are pinned by the design and must not drift:

- ``scale = (mx − mn) / 15``, ``zero = mn`` (float offset, **not** an integer
  zero-point), computed in fp32 over each quant group.
- ``q = clamp(round((x − zero) / scale), 0, 15)`` — round-half-even (torch's
  default), clamp **before** the uint cast.
- ``x̂ = q · scale + zero``.
- Degenerate group (``mx == mn``): ``scale = 1`` ⇒ all codes 0 and ``x̂ = mn``.
- **Scales and zeros are stored fp16 (pinned).** Both quantization and every
  later dequant run against the *fp16-stored* scale/zero (not the fp32
  intermediates), so the grid the codes were fit to is bit-identical to the
  grid used at read. This is what makes a re-demotion an exact reactivation.

Granularity (design.md §2):

- **Keys** — per-channel at the window level: one ``(scale, zero)`` per
  ``(head, channel)`` for a window, i.e. mx/mn reduced over the **token** axis.
  Stored **channel-major** ``[H_kv, D, window]`` and packed 2 tokens per byte.
- **Values** — per-token: one ``(scale, zero)`` per ``(head, token)``, i.e.
  mx/mn reduced over the **head_dim (channel)** axis. Stored **token-major**
  ``[H_kv, window, D]`` and packed 2 channels per byte.

Nibble packing (design.md §2): two int4 codes that share a scale go in one
byte; the **even-index** code occupies the low 4 bits. ``window_size`` must be
even (head_dim always is), so there is never a tail nibble to pad.

All functions here operate on a **single window** for **one row** (B = 1 in
v1) — shapes carry no batch axis. Keys/values come in token-major
``[H_kv, window, D]`` (the natural slice out of the fp store's row).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

_LEVELS = 15.0  # int4 asymmetric: codes in [0, 15]


# ---------------------------------------------------------------------------
# Core affine quant / dequant (grouped along one axis)
# ---------------------------------------------------------------------------


def _affine_quantize(x: Tensor, group_dim: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Affine asymmetric int4 quantize ``x`` grouped along ``group_dim``.

    The quant group is the slice along ``group_dim``: mx/mn are reduced over
    that axis so every group shares one ``(scale, zero)``.

    Returns
    -------
    codes : uint8 Tensor
        Same shape as ``x``; values in ``[0, 15]`` (unpacked, one code per
        element).
    scale, zero : fp16 Tensor
        Shape of ``x`` with ``group_dim`` reduced away (kept, not squeezed —
        callers squeeze as needed). Pinned fp16 grid.
    """
    x32 = x.to(torch.float32)
    mx = x32.amax(dim=group_dim, keepdim=True)
    mn = x32.amin(dim=group_dim, keepdim=True)

    scale = (mx - mn) / _LEVELS
    # Degenerate group: mx == mn ⇒ scale = 1 so every code rounds to 0 and the
    # dequant returns exactly mn (= zero).
    degenerate = mx == mn
    scale = torch.where(degenerate, torch.ones_like(scale), scale)
    zero = mn

    # Pin the grid in fp16, then quantize against the fp16 values (upcast for
    # the arithmetic) so the fit grid == the read grid, bit for bit.
    scale16 = scale.to(torch.float16)
    zero16 = zero.to(torch.float16)
    scale_grid = scale16.to(torch.float32)
    zero_grid = zero16.to(torch.float32)

    q = torch.round((x32 - zero_grid) / scale_grid)  # round-half-even
    q = torch.clamp(q, 0.0, _LEVELS)                 # clamp BEFORE uint cast
    codes = q.to(torch.uint8)

    return codes, scale16, zero16


def _affine_dequantize(
    codes: Tensor, scale16: Tensor, zero16: Tensor, out_dtype: torch.dtype
) -> Tensor:
    """Inverse of :func:`_affine_quantize`.

    ``scale16`` / ``zero16`` broadcast against ``codes`` along the (already
    reduced) group axis. Arithmetic upcasts the fp16 grid to fp32, then casts
    the result to ``out_dtype``.
    """
    scale = scale16.to(torch.float32)
    zero = zero16.to(torch.float32)
    x_hat = codes.to(torch.float32) * scale + zero
    return x_hat.to(out_dtype)


# ---------------------------------------------------------------------------
# Nibble packing (design.md §2) — two codes per byte, even index in low bits
# ---------------------------------------------------------------------------


def pack_nibbles_last(codes: Tensor) -> Tensor:
    """Pack unsigned int4 codes (0–15) two-per-byte along the **last** axis.

    ``codes`` last dim must be even. The even-index code goes in the low nibble.
    Returns a uint8 tensor with the last dim halved.
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError(
            f"pack_nibbles_last needs an even last dim, got {codes.shape[-1]}"
        )
    codes = codes.to(torch.uint8)
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)


def unpack_nibbles_last(packed: Tensor, n: int) -> Tensor:
    """Inverse of :func:`pack_nibbles_last`; ``n`` = original last-dim length.

    Returns a uint8 tensor whose last dim is ``n`` (== ``2 * packed.shape[-1]``
    for even ``n``), values in ``[0, 15]``.
    """
    packed = packed.to(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    return out[..., :n]


# ---------------------------------------------------------------------------
# Key quantization — per (head, channel), channel-major store, pack over tokens
# ---------------------------------------------------------------------------


def quantize_key_window(k_win: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Quantize one window's keys.

    Parameters
    ----------
    k_win : Tensor
        Shape ``[H_kv, window, D]`` (token-major, pre-RoPE keys for one window).
        ``window`` must be even.

    Returns
    -------
    packed : uint8 Tensor
        Shape ``[H_kv, D, window // 2]`` — channel-major, 2 tokens per byte.
    scale, zero : fp16 Tensor
        Shape ``[H_kv, D]`` — one grid per ``(head, channel)``.
    """
    if k_win.dim() != 3:
        raise ValueError(f"k_win must be [H_kv, window, D], got {tuple(k_win.shape)}")
    window = k_win.shape[1]
    if window % 2 != 0:
        raise ValueError(f"key window must be even, got {window}")

    # Quant group = token axis (dim 1) ⇒ scale/zero per (head, channel).
    codes, scale16, zero16 = _affine_quantize(k_win, group_dim=1)
    scale = scale16.squeeze(1)  # [H_kv, D]
    zero = zero16.squeeze(1)    # [H_kv, D]

    # Channel-major, then pack along the token axis (now last).
    codes_cm = codes.transpose(1, 2).contiguous()  # [H_kv, D, window]
    packed = pack_nibbles_last(codes_cm)            # [H_kv, D, window // 2]
    return packed, scale, zero


def dequantize_key_window(
    packed: Tensor,
    scale: Tensor,
    zero: Tensor,
    window: int,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Inverse of :func:`quantize_key_window`.

    Returns token-major ``[H_kv, window, D]`` in ``out_dtype`` — the same
    layout the fp store uses, ready for RoPE at the window's positions.
    """
    codes_cm = unpack_nibbles_last(packed, window)          # [H_kv, D, window]
    codes = codes_cm.transpose(1, 2).contiguous()           # [H_kv, window, D]
    scale16 = scale.unsqueeze(1)                            # [H_kv, 1, D]
    zero16 = zero.unsqueeze(1)                              # [H_kv, 1, D]
    return _affine_dequantize(codes, scale16, zero16, out_dtype)


# ---------------------------------------------------------------------------
# Value quantization — per (head, token), token-major store, pack over channels
# ---------------------------------------------------------------------------


def quantize_value_window(v_win: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Quantize one window's values.

    Parameters
    ----------
    v_win : Tensor
        Shape ``[H_kv, window, D]`` (token-major). ``D`` must be even.

    Returns
    -------
    packed : uint8 Tensor
        Shape ``[H_kv, window, D // 2]`` — token-major, 2 channels per byte.
    scale, zero : fp16 Tensor
        Shape ``[H_kv, window]`` — one grid per ``(head, token)``.
    """
    if v_win.dim() != 3:
        raise ValueError(f"v_win must be [H_kv, window, D], got {tuple(v_win.shape)}")
    D = v_win.shape[2]
    if D % 2 != 0:
        raise ValueError(f"value head_dim must be even, got {D}")

    # Quant group = channel axis (dim 2) ⇒ scale/zero per (head, token).
    codes, scale16, zero16 = _affine_quantize(v_win, group_dim=2)
    scale = scale16.squeeze(2)  # [H_kv, window]
    zero = zero16.squeeze(2)    # [H_kv, window]

    # Token-major already; pack along the channel axis (last).
    packed = pack_nibbles_last(codes.contiguous())  # [H_kv, window, D // 2]
    return packed, scale, zero


def dequantize_value_window(
    packed: Tensor,
    scale: Tensor,
    zero: Tensor,
    head_dim: int,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Inverse of :func:`quantize_value_window`.

    Returns token-major ``[H_kv, window, D]`` in ``out_dtype``.
    """
    codes = unpack_nibbles_last(packed, head_dim)  # [H_kv, window, D]
    scale16 = scale.unsqueeze(2)                   # [H_kv, window, 1]
    zero16 = zero.unsqueeze(2)                     # [H_kv, window, 1]
    return _affine_dequantize(codes, scale16, zero16, out_dtype)
