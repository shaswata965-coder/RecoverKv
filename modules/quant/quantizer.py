"""Hand-rolled KIVI-style affine int2 quantizer (design.md §2).

Numerics are pinned by the design and must not drift:

- ``scale = (mx − mn) / 3``, ``zero = mn`` (float offset, **not** an integer
  zero-point), computed in fp32 over each quant group.
- ``q = clamp(round((x − zero) / scale), 0, 3)`` — round-half-even (torch's
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
  Stored **channel-major** ``[H_kv, D, window]`` and packed 4 tokens per byte.
- **Values** — per-token: one ``(scale, zero)`` per ``(head, token)``, i.e.
  mx/mn reduced over the **head_dim (channel)** axis. Stored **token-major**
  ``[H_kv, window, D]`` and packed 4 channels per byte.

Crumb packing (design.md §2): four int2 codes that share a scale go in one
byte; code index ``j`` occupies bits ``[2·(j mod 4), +2)`` (so index 0 is the
lowest pair). ``window_size`` must be a multiple of 4 (head_dim always is), so
there is never a tail to pad. At int2 the fp16 scale/zero grid is fixed
overhead independent of the bit-width, so it becomes the *dominant* Q-window
cost — see the budget resolver (config.py) for the byte accounting.

All functions here operate on a **single window** for **one row** (B = 1 in
v1) — shapes carry no batch axis. Keys/values come in token-major
``[H_kv, window, D]`` (the natural slice out of the fp store's row).
"""

from __future__ import annotations

import os
import warnings
from typing import Tuple

import torch
from torch import Tensor

_LEVELS = 3.0  # int2 asymmetric: codes in [0, 3]


# ---------------------------------------------------------------------------
# Grid storage dtype (design.md §2 — the scale/zero pair, NOT the int2 codes)
# ---------------------------------------------------------------------------


def grid_dtype_for(kv_dtype: torch.dtype) -> torch.dtype:
    """The dtype the int2 ``(scale, zero)`` grid is stored in, for a given KV dtype.

    **Must stay 2 bytes.** The budget resolver charges the grid at 2 bytes per
    element (``modules/windowed_cache/config.py``, ``b_q``), and at int2 that
    grid is the *dominant* cost of a Q window — so a wider grid would silently
    change every compression ratio the method reports.

    - ``float16`` KV → ``float16``. Byte-identical to the original pinned grid,
      so the Llama-3.1 and Mistral-v0.2 fp16 columns produce bit-for-bit the
      results they always did. Overflow is structurally impossible there anyway:
      the inputs are fp16, so ``|zero| <= 65504`` and
      ``scale = (mx - mn)/3 <= 43669``.
    - **anything else → ``bfloat16``.** bf16 has fp32's exponent range in the
      same 2 bytes, so *no* value the cache can hold — bf16 or fp32 — can
      overflow it. That is the property being bought, and it is why this is a
      structural fix rather than a wider guard: a bf16 KV cache CAN hold a key
      past fp16's 65504, and casting the grid there produced ``inf`` and a
      window of ``nan`` with nothing raising.

    The trade is 3 mantissa bits (bf16's 8 vs fp16's 11) on the grid, against
    int2's 4 quantization levels. The code error dominates the grid error by
    orders of magnitude, and the exactness invariant that matters — fit grid ==
    read grid, bit for bit, which is what makes a re-demotion an exact
    reactivation (design §10) — is dtype-agnostic and preserved.
    """
    return torch.float16 if kv_dtype == torch.float16 else torch.bfloat16


# ---------------------------------------------------------------------------
# Grid representability guard — now a should-never-fire assertion
# ---------------------------------------------------------------------------
#
# With grid_dtype_for() above, neither overflow nor a zero-underflowed scale can
# reach the divide: the grid dtype covers the cache dtype's whole range, and
# _affine_quantize re-applies the degenerate rule after the cast. This check
# stays as the thing that would notice if that reasoning ever stopped holding —
# a new cache dtype, a caller passing grid_dtype explicitly, a change to the
# fallback. It reports; it does not repair (the `where` above already did).
#
# Bounded rather than always-on because it is a device sync: unbounded it would
# serialize the decode loop on GPU. It runs over the first evictions, which
# quantize the whole prompt across every layer and carry the largest magnitudes
# a run will ever see. STICKYKV_GRID_CHECKS=0 disables it; a larger value widens
# the window.
_GRID_CHECK_BUDGET = [None]
_grid_warned = [False]


def _grid_checks_remaining() -> int:
    if _GRID_CHECK_BUDGET[0] is None:
        try:
            n = int(os.environ.get("STICKYKV_GRID_CHECKS", "64"))
        except (TypeError, ValueError):
            n = 64
        _GRID_CHECK_BUDGET[0] = max(n, 0)
    return _GRID_CHECK_BUDGET[0]


def reset_grid_guard() -> None:
    """Re-arm the guard (tests; a new process starts armed)."""
    _GRID_CHECK_BUDGET[0] = None
    _grid_warned[0] = False


def _check_grid_representable(x, scale, zero, g_scale, g_zero, usable) -> None:
    """Warn once if the grid dtype could not hold a value the cache can.

    Skipped when the grid dtype is at least as wide as the input's exponent
    range, which is the case :func:`grid_dtype_for` guarantees — so on every
    shipped configuration this costs nothing and never fires. It exists for the
    configurations that reasoning does not cover.
    """
    if _grid_warned[0]:
        return
    # fp16 grid + fp16 input: |zero| <= 65504 and scale <= 43669 by construction.
    if g_scale.dtype == x.dtype:
        return
    # bf16 grid: same exponent range as fp32, so nothing the cache holds can
    # overflow it and nothing narrower than its subnormals can reach the divide.
    if g_scale.dtype == torch.bfloat16 and x.dtype in (
        torch.bfloat16, torch.float32, torch.float16
    ):
        return
    remaining = _grid_checks_remaining()
    if remaining <= 0:
        return
    _GRID_CHECK_BUDGET[0] = remaining - 1

    if not bool((~usable).any()):            # the sync, bounded to N calls
        return
    _grid_warned[0] = True
    finite = torch.isfinite(scale) & torch.isfinite(zero)
    biggest = float(torch.where(finite, zero.abs(), torch.zeros_like(zero)).amax())
    warnings.warn(
        f"int2 grid was not representable in {g_scale.dtype} while quantizing a "
        f"{x.dtype} tensor (largest finite |zero| seen: {biggest:.1f}). Those "
        f"groups fell back to the degenerate grid — they dequantize to their own "
        f"minimum rather than to inf/nan, so the run is not corrupt, but those "
        f"windows carry no information. Use modules.quant.quantizer.grid_dtype_for "
        f"to pick the grid dtype; it returns bfloat16 for any non-fp16 cache "
        f"dtype, which cannot overflow and costs the same 2 bytes.",
        RuntimeWarning,
        stacklevel=2,
    )
_CODES_PER_BYTE = 4  # int2: 4 crumbs per uint8


# ---------------------------------------------------------------------------
# Core affine quant / dequant (grouped along one axis)
# ---------------------------------------------------------------------------


def _affine_quantize(
    x: Tensor, group_dim: int, grid_dtype: torch.dtype = torch.float16
) -> Tuple[Tensor, Tensor, Tensor]:
    """Affine asymmetric int2 quantize ``x`` grouped along ``group_dim``.

    The quant group is the slice along ``group_dim``: mx/mn are reduced over
    that axis so every group shares one ``(scale, zero)``.

    ``grid_dtype`` is the storage dtype of that ``(scale, zero)`` pair — see
    :func:`grid_dtype_for`. It must be a 2-byte float, because the budget
    resolver charges the grid at 2 bytes per element and every reported
    compression ratio follows from that.

    Total by construction: no input can drive this to ``nan`` or to an undefined
    ``uint8`` cast. Three ways it used to be possible, all closed below —
    a grid value that overflows the storage dtype, a scale that underflows it to
    zero (``0/0`` in the division), and a non-finite input.

    Returns
    -------
    codes : uint8 Tensor
        Same shape as ``x``; values in ``[0, 3]`` (unpacked, one code per
        element).
    scale, zero : ``grid_dtype`` Tensor
        Shape of ``x`` with ``group_dim`` reduced away (kept, not squeezed —
        callers squeeze as needed).
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

    # Pin the grid in `grid_dtype`, then quantize against the STORED values
    # (upcast for the arithmetic) so the fit grid == the read grid, bit for bit.
    # That identity is what makes a re-demotion an exact reactivation (design
    # §10), and it holds for any grid dtype.
    g_scale = scale.to(grid_dtype)
    g_zero = zero.to(grid_dtype)

    # Re-apply the degenerate rule AFTER the cast. `mx == mn` catches a group
    # that is exactly constant, but the cast can also produce an unusable scale:
    # it underflows to 0 for a group whose spread is below grid_dtype's smallest
    # subnormal, and to inf for one whose spread overflows it. Dividing by either
    # yields inf or 0/0 = nan, and `nan.to(uint8)` is undefined — garbage codes
    # that dequantize to plausible values with nothing raising. Falling back to
    # the degenerate grid (scale 1, all codes 0, x̂ = zero) is the same answer
    # the exactly-constant case already gets, and is exact for a group that
    # narrow. Written as `where`, not a branch, so there is no host sync.
    usable = torch.isfinite(g_scale) & (g_scale != 0)
    g_scale = torch.where(usable, g_scale, torch.ones_like(g_scale))
    g_zero = torch.where(torch.isfinite(g_zero), g_zero, torch.zeros_like(g_zero))
    _check_grid_representable(x, scale, zero, g_scale, g_zero, usable)

    scale_grid = g_scale.to(torch.float32)
    zero_grid = g_zero.to(torch.float32)

    q = torch.round((x32 - zero_grid) / scale_grid)  # round-half-even
    # A non-finite INPUT (the model itself diverged) would otherwise reach the
    # uint8 cast as nan. Map it to a defined code instead of undefined behaviour:
    # the fp tier still carries the nan, so the divergence stays visible where it
    # actually happened. Identity on finite q, and ±inf already clamped to the
    # same place below, so this does not move a single healthy value.
    q = torch.nan_to_num(q, nan=0.0, posinf=_LEVELS, neginf=0.0)
    q = torch.clamp(q, 0.0, _LEVELS)                 # clamp BEFORE uint cast
    codes = q.to(torch.uint8)

    return codes, g_scale, g_zero


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
# Crumb packing (design.md §2) — four int2 codes per byte, index 0 in low bits
# ---------------------------------------------------------------------------


def pack_crumbs_last(codes: Tensor) -> Tensor:
    """Pack unsigned int2 codes (0–3) four-per-byte along the **last** axis.

    ``codes`` last dim must be a multiple of 4. Code index ``j`` goes into bits
    ``[2·(j mod 4), +2)`` of its byte (index 0 in the two lowest bits). Returns
    a uint8 tensor with the last dim quartered.
    """
    if codes.shape[-1] % _CODES_PER_BYTE != 0:
        raise ValueError(
            f"pack_crumbs_last needs a last dim divisible by {_CODES_PER_BYTE}, "
            f"got {codes.shape[-1]}"
        )
    codes = codes.to(torch.uint8)
    c0 = codes[..., 0::4]
    c1 = codes[..., 1::4]
    c2 = codes[..., 2::4]
    c3 = codes[..., 3::4]
    return (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).to(torch.uint8)


def unpack_crumbs_last(packed: Tensor, n: int) -> Tensor:
    """Inverse of :func:`pack_crumbs_last`; ``n`` = original last-dim length.

    Returns a uint8 tensor whose last dim is ``n`` (== ``4 * packed.shape[-1]``
    for ``n`` divisible by 4), values in ``[0, 3]``.
    """
    packed = packed.to(torch.uint8)
    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03
    out = torch.stack([c0, c1, c2, c3], dim=-1).reshape(*packed.shape[:-1], -1)
    return out[..., :n]


# ---------------------------------------------------------------------------
# Key quantization — per (head, channel), channel-major store, pack over tokens
# ---------------------------------------------------------------------------


def quantize_key_window(k_win: Tensor, grid_dtype: torch.dtype = torch.float16) -> Tuple[Tensor, Tensor, Tensor]:
    """Quantize one window's keys.

    Parameters
    ----------
    k_win : Tensor
        Shape ``[H_kv, window, D]`` (token-major, pre-RoPE keys for one window).
        ``window`` must be a multiple of 4.

    Returns
    -------
    packed : uint8 Tensor
        Shape ``[H_kv, D, window // 4]`` — channel-major, 4 tokens per byte.
    scale, zero : fp16 Tensor
        Shape ``[H_kv, D]`` — one grid per ``(head, channel)``.
    """
    if k_win.dim() != 3:
        raise ValueError(f"k_win must be [H_kv, window, D], got {tuple(k_win.shape)}")
    window = k_win.shape[1]
    if window % _CODES_PER_BYTE != 0:
        raise ValueError(
            f"key window must be a multiple of {_CODES_PER_BYTE}, got {window}"
        )

    # Quant group = token axis (dim 1) ⇒ scale/zero per (head, channel).
    codes, scale16, zero16 = _affine_quantize(k_win, group_dim=1, grid_dtype=grid_dtype)
    scale = scale16.squeeze(1)  # [H_kv, D]
    zero = zero16.squeeze(1)    # [H_kv, D]

    # Channel-major, then pack along the token axis (now last).
    codes_cm = codes.transpose(1, 2).contiguous()  # [H_kv, D, window]
    packed = pack_crumbs_last(codes_cm)             # [H_kv, D, window // 4]
    return packed, scale, zero


def quantize_key_windows(k_wins: Tensor, grid_dtype: torch.dtype = torch.float16) -> Tuple[Tensor, Tensor, Tensor]:
    """Batched :func:`quantize_key_window` over a leading window axis.

    Bit-identical to calling the singular form per window: the quant group is
    still the token axis, and every op below is either elementwise or a
    reduction over a non-batch axis, so a leading ``N`` rides through untouched.
    Batching matters at eviction, where demoting N windows one at a time is a
    launch storm on GPU.

    Parameters
    ----------
    k_wins : Tensor
        ``[N, H_kv, window, D]`` token-major pre-RoPE keys. ``window`` a
        multiple of 4.

    Returns
    -------
    packed : uint8 ``[N, H_kv, D, window // 4]`` — channel-major.
    scale, zero : fp16 ``[N, H_kv, D]``.
    """
    if k_wins.dim() != 4:
        raise ValueError(
            f"k_wins must be [N, H_kv, window, D], got {tuple(k_wins.shape)}"
        )
    window = k_wins.shape[2]
    if window % _CODES_PER_BYTE != 0:
        raise ValueError(
            f"key window must be a multiple of {_CODES_PER_BYTE}, got {window}"
        )

    # Quant group = token axis (dim 2 with the batch axis) ⇒ grid per (N, head, channel).
    codes, scale16, zero16 = _affine_quantize(k_wins, group_dim=2, grid_dtype=grid_dtype)
    scale = scale16.squeeze(2)  # [N, H_kv, D]
    zero = zero16.squeeze(2)    # [N, H_kv, D]

    codes_cm = codes.transpose(2, 3).contiguous()  # [N, H_kv, D, window]
    packed = pack_crumbs_last(codes_cm)            # [N, H_kv, D, window // 4]
    return packed, scale, zero


def quantize_value_windows(v_wins: Tensor, grid_dtype: torch.dtype = torch.float16) -> Tuple[Tensor, Tensor, Tensor]:
    """Batched :func:`quantize_value_window` over a leading window axis.

    Bit-identical to the singular form applied per window — see
    :func:`quantize_key_windows`.

    Parameters
    ----------
    v_wins : Tensor
        ``[N, H_kv, window, D]`` token-major values. ``D`` a multiple of 4.

    Returns
    -------
    packed : uint8 ``[N, H_kv, window, D // 4]`` — token-major.
    scale, zero : fp16 ``[N, H_kv, window]``.
    """
    if v_wins.dim() != 4:
        raise ValueError(
            f"v_wins must be [N, H_kv, window, D], got {tuple(v_wins.shape)}"
        )
    D = v_wins.shape[3]
    if D % _CODES_PER_BYTE != 0:
        raise ValueError(
            f"value head_dim must be a multiple of {_CODES_PER_BYTE}, got {D}"
        )

    # Quant group = channel axis ⇒ grid per (N, head, token).
    codes, scale16, zero16 = _affine_quantize(v_wins, group_dim=3, grid_dtype=grid_dtype)
    scale = scale16.squeeze(3)  # [N, H_kv, window]
    zero = zero16.squeeze(3)    # [N, H_kv, window]

    packed = pack_crumbs_last(codes.contiguous())  # [N, H_kv, window, D // 4]
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
    codes_cm = unpack_crumbs_last(packed, window)           # [H_kv, D, window]
    codes = codes_cm.transpose(1, 2).contiguous()           # [H_kv, window, D]
    scale16 = scale.unsqueeze(1)                            # [H_kv, 1, D]
    zero16 = zero.unsqueeze(1)                              # [H_kv, 1, D]
    return _affine_dequantize(codes, scale16, zero16, out_dtype)


def dequantize_key_windows(
    packed: Tensor,
    scale: Tensor,
    zero: Tensor,
    window: int,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Batched :func:`dequantize_key_window` over a leading window axis.

    Same numerics as the singular form applied per slice — every op here is
    elementwise on the last axis or a broadcast, so the ``N`` windows are
    dequantized in a handful of tensor ops instead of ``N`` Python-loop calls
    (read-path launch-count fix; the per-window loop was launch-bound on GPU).

    Parameters
    ----------
    packed : uint8 ``[N, H_kv, D, window // 4]`` — channel-major, 4 tokens/byte.
    scale, zero : fp16 ``[N, H_kv, D]`` — one grid per ``(window, head, channel)``.
    window : int — original (unpacked) token count.

    Returns
    -------
    ``[N, H_kv, window, D]`` token-major, in ``out_dtype``.
    """
    codes_cm = unpack_crumbs_last(packed, window)           # [N, H_kv, D, window]
    codes = codes_cm.transpose(2, 3).contiguous()           # [N, H_kv, window, D]
    scale16 = scale.unsqueeze(2)                            # [N, H_kv, 1, D]
    zero16 = zero.unsqueeze(2)                              # [N, H_kv, 1, D]
    return _affine_dequantize(codes, scale16, zero16, out_dtype)


# ---------------------------------------------------------------------------
# Value quantization — per (head, token), token-major store, pack over channels
# ---------------------------------------------------------------------------


def quantize_value_window(v_win: Tensor, grid_dtype: torch.dtype = torch.float16) -> Tuple[Tensor, Tensor, Tensor]:
    """Quantize one window's values.

    Parameters
    ----------
    v_win : Tensor
        Shape ``[H_kv, window, D]`` (token-major). ``D`` must be a multiple of 4.

    Returns
    -------
    packed : uint8 Tensor
        Shape ``[H_kv, window, D // 4]`` — token-major, 4 channels per byte.
    scale, zero : fp16 Tensor
        Shape ``[H_kv, window]`` — one grid per ``(head, token)``.
    """
    if v_win.dim() != 3:
        raise ValueError(f"v_win must be [H_kv, window, D], got {tuple(v_win.shape)}")
    D = v_win.shape[2]
    if D % _CODES_PER_BYTE != 0:
        raise ValueError(
            f"value head_dim must be a multiple of {_CODES_PER_BYTE}, got {D}"
        )

    # Quant group = channel axis (dim 2) ⇒ scale/zero per (head, token).
    codes, scale16, zero16 = _affine_quantize(v_win, group_dim=2, grid_dtype=grid_dtype)
    scale = scale16.squeeze(2)  # [H_kv, window]
    zero = zero16.squeeze(2)    # [H_kv, window]

    # Token-major already; pack along the channel axis (last).
    packed = pack_crumbs_last(codes.contiguous())  # [H_kv, window, D // 4]
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
    codes = unpack_crumbs_last(packed, head_dim)   # [H_kv, window, D]
    scale16 = scale.unsqueeze(2)                   # [H_kv, window, 1]
    zero16 = zero.unsqueeze(2)                     # [H_kv, window, 1]
    return _affine_dequantize(codes, scale16, zero16, out_dtype)


def dequantize_value_windows(
    packed: Tensor,
    scale: Tensor,
    zero: Tensor,
    head_dim: int,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Batched :func:`dequantize_value_window` over a leading window axis.

    Parameters
    ----------
    packed : uint8 ``[N, H_kv, window, D // 4]`` — token-major, 4 channels/byte.
    scale, zero : fp16 ``[N, H_kv, window]`` — one grid per ``(window, head, token)``.
    head_dim : int — original (unpacked) channel count.

    Returns
    -------
    ``[N, H_kv, window, D]`` token-major, in ``out_dtype``.
    """
    codes = unpack_crumbs_last(packed, head_dim)   # [N, H_kv, window, D]
    scale16 = scale.unsqueeze(3)                   # [N, H_kv, window, 1]
    zero16 = zero.unsqueeze(3)                     # [N, H_kv, window, 1]
    return _affine_dequantize(codes, scale16, zero16, out_dtype)
