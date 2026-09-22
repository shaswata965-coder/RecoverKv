"""Hand-rolled KIVI-style affine int2 quantizer (design.md §2).

Numerics are pinned by the design and must not drift:

- ``scale = (mx − mn) / 3``, ``zero = mn`` (float offset, **not** an integer
  zero-point), computed in fp32 over each quant group.
- ``q = clamp(round((x − zero) / scale), 0, 3)`` — round-half-even (torch's
  default), clamp **before** the uint cast.
- ``x̂ = q · scale + zero``.
- Degenerate group (``mx == mn``): ``scale = 1`` ⇒ all codes 0 and ``x̂ = mn``.
- **Scales and zeros are stored one byte each** (:class:`QGrid`), against an
  fp16 scale shared by 32 of them. Both quantization and every later dequant run
  against the *stored, decoded* grid (not the fp32 intermediates), so the grid
  the codes were fit to is bit-identical to the grid used at read. This is what
  makes a re-demotion an exact reactivation.

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
there is never a tail to pad.

**The grid is one byte per entry, not two.** At int2 the scale/zero grid is
fixed overhead independent of the bit-width, so an fp16 grid was 48.5% of a
window's codes-plus-grid — two bytes of precision wrapping a two-bit code. It is
now int8 against an fp16 scale shared by :data:`GRID_GROUP` entries, which is
:func:`grid_bytes_per_head` bytes instead of ``4·axis``: 272 B/head against
512 at ``D=128``, and a Q window 6432 B instead of 8448 before its card. See
:class:`QGrid` for what that cost in accuracy and why the sharing is grouped.
The budget resolver (config.py) prices a window through
:func:`bytes_per_q_window`, so the format and its byte count cannot drift apart.

All functions here operate on a **single window** for **one row** (B = 1 in
v1) — shapes carry no batch axis. Keys/values come in token-major
``[H_kv, window, D]`` (the natural slice out of the fp store's row).
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import torch
from torch import Tensor

_LEVELS = 3.0  # int2 asymmetric: codes in [0, 3]
_CODES_PER_BYTE = 4  # int2: 4 crumbs per uint8
_TINY = torch.finfo(torch.float32).tiny
#: Smallest positive fp16 (a subnormal, 2**-24). A group scale is clamped up to
#: it before the cast: dividing a group's amax by 255 can underflow fp16 where
#: the amax itself is perfectly representable, and a shared scale of zero decodes
#: the whole group's grid to zero, which the fit then divides by.
_FP16_MIN_POS = 2.0 ** -24

#: Grid entries sharing one fp16 scale. Measured, not chosen for roundness --
#: see :class:`QGrid`. An axis shorter than this shares a single scale.
GRID_GROUP = 32


# Whether the quant range-reductions may be traced into a torch.compile graph.
#
# They used to be unconditionally excluded. In the demote path this function's
# input is a DATA-DEPENDENT gather (searchsorted-derived indices into the fp
# body), and Inductor <= 2.6 could not lower a reduction whose read index is a
# StarDep: it fused gather->amin and then failed to schedule it ("StarDep does
# not have an index on aten.amin.default"). A graph break sidestepped that.
#
# The break was never free, and the cost was mis-sized when it was taken. It was
# scored as "~10 ops per demoted window, the launch win is essentially intact",
# but a decode profile at 4096/batch-32 says otherwise: the eviction step runs
# ~195 ms against a ~72 ms steady step, and ~15 ms/step amortized sits in exactly
# this chain -- round, clamp, div, sub, where, scatter -- as sixteen separate
# multi-millisecond kernels, because the break stops Inductor fusing the one part
# of the eviction worth fusing.
#
# The lowering bug is fixed upstream (>= 2.7, the boundary the original
# workaround named). But tracing this function is only SAFE under a second
# condition, and it is the more important one:
#
#   Inductor, by default, does not emulate intermediate precision casts. It
#   drops an fp32 -> fp16 -> fp32 round trip and keeps the fp32 value in a
#   register, because for most code that is a free accuracy win. Here it is not:
#   the round trip below is the whole point. Codes are fit to the *fp16-stored*
#   grid so the grid they were fit to is bit-identical to the grid every later
#   dequant reads (see the module docstring). Elide it and the codes are fit to
#   an fp32 grid and read back on an fp16 one.
#
# Measured on 2.14, CPU Inductor, 24 seeds: undecorated and without the flag,
# the packed codes differ from eager on 11/24 seeds for keys and 14/24 for
# values -- always a single int2 level, always on an element whose pre-round
# quantity straddles .5 (the two grids differ by an fp16 ulp, ~1.7e-03 here).
# With torch._inductor.config.emulate_precision_casts on: 0/24, both.
#
# So the compile site is responsible for turning that flag on, and this gate
# refuses to trace on a build that has no such flag to turn on -- an older build
# keeps the graph break and keeps working, rather than silently writing a
# different cache under torch.compile than without it. The eviction's compile
# site is modules/windowed_cache/cache.py:_emulating_precision_casts, which is
# also where the same hazard in the sketch cards is handled.
def _quant_may_be_traced() -> bool:
    """True when Inductor can lower, AND be made to keep, this function."""
    try:
        major, minor = (int(p) for p in torch.__version__.split(".")[:2])
    except Exception:  # pragma: no cover - unparseable version string
        return False
    if (major, minor) < (2, 7):  # pragma: no cover - torch-version dependent
        return False  # gather -> amin StarDep lowering failure
    try:
        from torch._inductor import config as _inductor_config
    except Exception:  # pragma: no cover - no Inductor on this build
        return False
    # No knob to preserve the fp16 grid round trip => do not trace.
    return hasattr(_inductor_config, "emulate_precision_casts")


if _quant_may_be_traced():
    def _compile_disable(fn):
        """Identity: the quantiser is traced and fused with the eviction body."""
        return fn
else:  # pragma: no cover - torch-version dependent
    # torch.compiler.disable is the stable API (torch 2.1+); fall back to the
    # private one on older builds.
    try:
        _compile_disable = torch.compiler.disable
    except AttributeError:
        _compile_disable = torch._dynamo.disable


# ---------------------------------------------------------------------------
# The stored grid: one byte per entry, an fp16 scale per GRID_GROUP of them
# ---------------------------------------------------------------------------


def grid_group(axis_len: int) -> int:
    """Entries sharing one fp16 scale along a grid axis of ``axis_len``.

    ``GRID_GROUP``, or the whole axis when it is shorter (the value grid's axis
    is ``window_size``, which is 8 at the shipped operating point). A longer axis
    must be a multiple of ``GRID_GROUP``: a ragged tail would need a mask on
    every read, including inside the decode kernel's inner loop.
    """
    if axis_len <= GRID_GROUP:
        return axis_len
    if axis_len % GRID_GROUP:
        raise ValueError(
            f"a grid axis of {axis_len} must be a multiple of GRID_GROUP="
            f"{GRID_GROUP} (or shorter than it); a ragged tail would need a "
            "masked load in the decode kernel's inner loop."
        )
    return GRID_GROUP


def grid_bytes_per_head(axis_len: int) -> int:
    """Bytes one head's ``(scale, zero)`` grid costs over ``axis_len`` entries.

    Two one-byte codes per entry plus the two fp16 scales each group shares.
    LOAD-BEARING: :func:`bytes_per_q_window` prices a window with it and the
    budget resolver prices the tier with that, so the format and the memory
    claim cannot drift apart.
    """
    return 2 * axis_len + 4 * (axis_len // grid_group(axis_len))


def bytes_per_q_window(num_kv_heads: int, head_dim: int, window_size: int) -> int:
    """Bytes one int2 window costs: codes + both grids, **excluding its card**.

    The card is priced by :func:`modules.quant.sketch.sketch_bytes_per_head`;
    the caller adds it when the gate is live. At ``H=8, D=128, ws=8`` this is
    6432 B against the fp16 grid's 8448.
    """
    codes = (head_dim * window_size) // 2          # K and V, 2 bits per element
    return num_kv_heads * (codes + grid_bytes_per_head(head_dim)
                           + grid_bytes_per_head(window_size))


class QGrid(NamedTuple):
    """An affine grid field (``scale`` or ``zero``) as stored: codes + scales.

    ``q`` holds one byte per entry -- uint8 in ``[1, 255]`` for ``scale``, which
    is non-negative and must never decode to zero, int8 for ``zero``, which is an
    absolute offset and signed. ``s`` holds the fp16 scale that
    :func:`grid_group` consecutive codes share, so ``s`` has the same shape as
    ``q`` with its last axis divided by the group.

    Why one byte, and why grouped
    -----------------------------
    An fp16 grid was 48.5% of a window's codes-plus-grid. Narrowing it is nearly
    free because design §2 pins the codes to the *stored* grid: a rounded scale
    just repositions the four int2 levels and the codes re-fit against them, so
    the int2 error (~13% on keys) swamps the grid's rounding.

    Measured over 12 seeds on keys with massive-activation channels and one hot
    token per window (``tests/test_quant_grid.py`` pins these):

    ======================  ===========  ===========  ==========
    grid                    K rel err    q.k rel err  B/head (D=128)
    ======================  ===========  ===========  ==========
    fp16 (was)              0.13348      0.13541      512
    int8, one scale/head    0.13370      0.13568      260
    **int8, groups of 32**  **0.13368**  **0.13562**  **272**
    ======================  ===========  ===========  ==========

    The grouping is what makes it safe rather than merely cheap. A single scale
    per head is an aggregate +0.2% and hides a per-channel failure: on keys whose
    channel gains span 1000x -- which is what a massive-activation channel *is*
    -- the worst channel's own reconstruction error goes 0.305 -> 1.082, i.e.
    that channel decodes to noise while the Frobenius norm, dominated by the big
    channels, moves by +0.14% and says nothing. Groups of 32 put it back at
    0.306, fp16's own figure, for 12 B/head. Aggregate error is the wrong
    instrument for a shared exponent, so the choice is made on the worst channel.

    fp8 was measured too (GATE_REGRESSION §5: e4m3 +1.6%, e5m2 +4.5%) and is
    worse on both counts, and sm80 has no hardware convert for it.
    """

    q: Tensor      # [..., axis]            uint8 (scale) / int8 (zero)
    s: Tensor      # [..., axis // group]   fp16

    @property
    def group(self) -> int:
        """Entries per shared scale."""
        return self.q.shape[-1] // self.s.shape[-1]

    def decode(self) -> Tensor:
        """The fp32 grid a reader sees. **This** is what codes are fit to."""
        lead = self.q.shape[:-1]
        wide = self.q.to(torch.float32).reshape(*lead, self.s.shape[-1], -1)
        return (wide * self.s.to(torch.float32).unsqueeze(-1)).reshape(self.q.shape)

    def reshape(self, *shape: int) -> "QGrid":
        """Reshape with the grid axis last; ``s`` follows with its own last axis."""
        return QGrid(self.q.reshape(*shape), self.s.reshape(*shape[:-1], -1))

    def map(self, fn) -> "QGrid":
        """Apply one tensor op to both halves — a view, a slice, a layer split.

        Use this rather than indexing: a ``QGrid`` is a ``NamedTuple``, so
        ``grid[0]`` silently returns ``q`` instead of the first row.
        """
        return QGrid(fn(self.q), fn(self.s))


def grid_dtype_for(kv_dtype: torch.dtype) -> torch.dtype:
    """The dtype the one-byte grid's group scale (``QGrid.s``) is stored in.

    **Must stay 2 bytes.** The budget resolver charges the grid at 2 bytes per
    group scale (:func:`grid_bytes_per_head`), and at int2 the grid is a large
    part of a Q window, so a wider scale would silently change every compression
    ratio the method reports.

    - ``float16`` KV → ``float16``. **Byte-identical** to the original pinned
      grid, so the Llama-3.1 and Mistral-v0.2 fp16 columns produce bit-for-bit
      the results they always did. Overflow is structurally impossible there: the
      grid scale is ``amax / levels`` of an fp16-derived value, well inside fp16.
    - **anything else (bfloat16, float32) → ``bfloat16``.** bf16 has fp32's
      exponent range in the same 2 bytes, so no value the cache can hold can
      overflow it. This is the property being bought: a **bf16** KV cache (Qwen2.5
      native) CAN hold a key past fp16's 65504, and casting the group scale to
      fp16 there produced ``inf`` and a window of ``nan`` — with nothing raising —
      for the rest of the run. The overflow is on the grid's *group scale*
      (``amax / 127`` for the zero field can exceed 65504 once ``|zero| > 8.3e6``,
      which a bf16 massive-activation channel reaches), not on the int2 codes.

    The trade is 3 mantissa bits (bf16's 8 vs fp16's 11) on the group scale,
    against int2's 4 levels — the code error dominates by orders of magnitude, and
    the exactness invariant (fit grid == read grid, bit for bit; design §10) is
    dtype-agnostic and preserved because the codes are always fit to the *stored*
    grid.
    """
    return torch.float16 if kv_dtype == torch.float16 else torch.bfloat16


def _quantize_grid(
    x: Tensor, *, signed: bool, grid_dtype: torch.dtype = torch.float16
) -> QGrid:
    """Encode one grid field. Codes are fit to the **stored** group scale.

    ``signed`` picks the field: ``zero`` is an offset and takes int8 symmetric;
    ``scale`` is non-negative and takes the full uint8 range, clamped at 1 so a
    decoded scale is never zero (the fit divides by it).

    ``grid_dtype`` is the storage dtype of the group scale ``QGrid.s`` — see
    :func:`grid_dtype_for`. It defaults to fp16 (the original, byte-identical
    behaviour); a bf16 KV cache passes bfloat16 so the scale cannot overflow.
    """
    g = grid_group(x.shape[-1])
    xg = x.reshape(*x.shape[:-1], -1, g)
    levels = 127.0 if signed else 255.0
    amax = (xg.abs() if signed else xg).amax(dim=-1, keepdim=True)
    s = torch.where(amax > 0, amax / levels, torch.ones_like(amax))
    # clamp_min BEFORE the cast keeps a decoded scale non-zero (the fit divides by
    # it); the floor is fp16's smallest subnormal, which is representable in bf16
    # too, so it never underflows either grid dtype to zero.
    sg = s.clamp_min(_FP16_MIN_POS).to(grid_dtype)
    s32 = sg.to(torch.float32).clamp_min(_TINY)
    q = torch.round(xg / s32)
    q = q.clamp_(-levels, levels) if signed else q.clamp_(1.0, levels)
    codes = q.to(torch.int8 if signed else torch.uint8).reshape(x.shape)
    return QGrid(codes, sg.squeeze(-1))


# ---------------------------------------------------------------------------
# Core affine quant / dequant (grouped along one axis)
# ---------------------------------------------------------------------------


@_compile_disable
def _affine_quantize(x: Tensor, group_dim: int) -> Tuple[Tensor, "QGrid", "QGrid"]:
    """Affine asymmetric int2 quantize ``x`` grouped along ``group_dim``.

    The quant group is the slice along ``group_dim``: mx/mn are reduced over
    that axis so every group shares one ``(scale, zero)``.

    Returns
    -------
    codes : uint8 Tensor
        Same shape as ``x``; values in ``[0, 3]`` (unpacked, one code per
        element).
    scale, zero : :class:`QGrid`
        Shape of ``x`` with ``group_dim`` reduced **away** (squeezed, so the
        grid axis is last and :class:`QGrid`'s sharing applies to it).
    """
    # The group scale is stored in a dtype that follows the KV dtype, so a
    # bf16 cache (Qwen2.5) cannot overflow the fp16 grid the way it silently did.
    # `x` is the KV window in the model dtype, so its dtype IS the KV dtype — no
    # threading through the store is needed. fp16 keeps the fp16 grid, byte for
    # byte, so Llama/Mistral are unchanged. See `grid_dtype_for`.
    grid_dtype = grid_dtype_for(x.dtype)

    x32 = x.to(torch.float32)
    mx = x32.amax(dim=group_dim, keepdim=True)
    mn = x32.amin(dim=group_dim, keepdim=True)

    scale = (mx - mn) / _LEVELS
    # Degenerate group: mx == mn ⇒ scale = 1 so every code rounds to 0 and the
    # dequant returns exactly mn (= zero).
    degenerate = mx == mn
    scale = torch.where(degenerate, torch.ones_like(scale), scale)
    zero = mn

    # Encode the grid, then quantize against the DECODED grid, so the grid the
    # codes were fit to is the grid every later dequant reads, bit for bit.
    # `_quantize_grid` takes the grid axis last, which is what squeezing
    # `group_dim` leaves: keys reduce over tokens and share over channels,
    # values reduce over channels and share over tokens.
    scale_g = _quantize_grid(scale.squeeze(group_dim), signed=False, grid_dtype=grid_dtype)
    zero_g = _quantize_grid(zero.squeeze(group_dim), signed=True, grid_dtype=grid_dtype)
    scale_grid = scale_g.decode().unsqueeze(group_dim)
    zero_grid = zero_g.decode().unsqueeze(group_dim)

    # In place, deliberately. The out-of-place form
    #     q = torch.round((x32 - zero_grid) / scale_grid)
    #     q = torch.clamp(q, 0.0, _LEVELS)
    # holds three fp32 buffers of x's size live at its peak (the subtract's
    # output, the divide's output, and x32 itself), which is 12 bytes per
    # element to quantize a 2-byte one. At 2048/batch-32 the first eviction --
    # unavoidably full width, because on the first eviction everything genuinely
    # is fresh -- asked the allocator for 7.25 GiB here and did not get it.
    # Chaining in place keeps exactly one, and every op and its order is
    # unchanged, so the codes are bit-for-bit what the expression above
    # produced (test_affine_quantize_inplace_matches_the_out_of_place_form).
    if x32 is x:
        # `.to()` is a no-op when x is already fp32, and mutating the caller's
        # tensor would be a silent corruption rather than a slow path.
        x32 = x32.clone()
    q = x32.sub_(zero_grid).div_(scale_grid).round_()  # round-half-even
    q = q.clamp_(0.0, _LEVELS)                         # clamp BEFORE uint cast
    codes = q.to(torch.uint8)

    return codes, scale_g, zero_g


def _affine_dequantize(
    codes: Tensor, scale: Tensor, zero: Tensor, out_dtype: torch.dtype
) -> Tensor:
    """Inverse of :func:`_affine_quantize`.

    ``scale`` / ``zero`` are **decoded** fp32 grids (``QGrid.decode()``), already
    broadcast against ``codes`` along the reduced group axis by the caller —
    which is the one place that knows where that axis sits.
    """
    # Same reasoning as the quantiser: `codes.to(float32) * scale + zero` holds
    # three fp32 buffers at its peak. The upcast allocates a fresh tensor
    # (codes is uint8), so chaining in place onto it is safe and mutates nothing
    # the caller owns. `scale` and `zero` broadcast into the already-larger
    # codes shape, so the in-place output shape is unchanged.
    x_hat = codes.to(torch.float32).mul_(scale).add_(zero)
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


def quantize_key_window(k_win: Tensor) -> Tuple[Tensor, "QGrid", "QGrid"]:
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
    scale, zero : :class:`QGrid`
        Codes ``[H_kv, D]`` — one grid entry per ``(head, channel)`` — over
        ``[H_kv, D // group]`` shared fp16 scales.
    """
    if k_win.dim() != 3:
        raise ValueError(f"k_win must be [H_kv, window, D], got {tuple(k_win.shape)}")
    window = k_win.shape[1]
    if window % _CODES_PER_BYTE != 0:
        raise ValueError(
            f"key window must be a multiple of {_CODES_PER_BYTE}, got {window}"
        )

    # Quant group = token axis (dim 1) ⇒ scale/zero per (head, channel).
    codes, scale, zero = _affine_quantize(k_win, group_dim=1)

    # Channel-major, then pack along the token axis (now last).
    codes_cm = codes.transpose(1, 2).contiguous()  # [H_kv, D, window]
    packed = pack_crumbs_last(codes_cm)             # [H_kv, D, window // 4]
    return packed, scale, zero


def quantize_key_windows(k_wins: Tensor) -> Tuple[Tensor, "QGrid", "QGrid"]:
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
    scale, zero : :class:`QGrid`, codes ``[N, H_kv, D]``.
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
    codes, scale, zero = _affine_quantize(k_wins, group_dim=2)

    codes_cm = codes.transpose(2, 3).contiguous()  # [N, H_kv, D, window]
    packed = pack_crumbs_last(codes_cm)            # [N, H_kv, D, window // 4]
    return packed, scale, zero


def quantize_value_windows(v_wins: Tensor) -> Tuple[Tensor, "QGrid", "QGrid"]:
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
    scale, zero : :class:`QGrid`, codes ``[N, H_kv, window]``.
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
    codes, scale, zero = _affine_quantize(v_wins, group_dim=3)

    packed = pack_crumbs_last(codes.contiguous())  # [N, H_kv, window, D // 4]
    return packed, scale, zero


def dequantize_key_window(
    packed: Tensor,
    scale: "QGrid",
    zero: "QGrid",
    window: int,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Inverse of :func:`quantize_key_window`.

    Returns token-major ``[H_kv, window, D]`` in ``out_dtype`` — the same
    layout the fp store uses, ready for RoPE at the window's positions.
    """
    codes_cm = unpack_crumbs_last(packed, window)           # [H_kv, D, window]
    codes = codes_cm.transpose(1, 2).contiguous()           # [H_kv, window, D]
    return _affine_dequantize(
        codes, scale.decode().unsqueeze(1), zero.decode().unsqueeze(1), out_dtype
    )


def dequantize_key_windows(
    packed: Tensor,
    scale: "QGrid",
    zero: "QGrid",
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
    scale, zero : :class:`QGrid` — one grid entry per ``(window, head, channel)``.
    window : int — original (unpacked) token count.

    Returns
    -------
    ``[N, H_kv, window, D]`` token-major, in ``out_dtype``.
    """
    codes_cm = unpack_crumbs_last(packed, window)           # [N, H_kv, D, window]
    codes = codes_cm.transpose(2, 3).contiguous()           # [N, H_kv, window, D]
    return _affine_dequantize(
        codes, scale.decode().unsqueeze(2), zero.decode().unsqueeze(2), out_dtype
    )


# ---------------------------------------------------------------------------
# Value quantization — per (head, token), token-major store, pack over channels
# ---------------------------------------------------------------------------


def quantize_value_window(v_win: Tensor) -> Tuple[Tensor, "QGrid", "QGrid"]:
    """Quantize one window's values.

    Parameters
    ----------
    v_win : Tensor
        Shape ``[H_kv, window, D]`` (token-major). ``D`` must be a multiple of 4.

    Returns
    -------
    packed : uint8 Tensor
        Shape ``[H_kv, window, D // 4]`` — token-major, 4 channels per byte.
    scale, zero : :class:`QGrid`
        Codes ``[H_kv, window]`` — one grid entry per ``(head, token)``.
    """
    if v_win.dim() != 3:
        raise ValueError(f"v_win must be [H_kv, window, D], got {tuple(v_win.shape)}")
    D = v_win.shape[2]
    if D % _CODES_PER_BYTE != 0:
        raise ValueError(
            f"value head_dim must be a multiple of {_CODES_PER_BYTE}, got {D}"
        )

    # Quant group = channel axis (dim 2) ⇒ scale/zero per (head, token).
    codes, scale, zero = _affine_quantize(v_win, group_dim=2)

    # Token-major already; pack along the channel axis (last).
    packed = pack_crumbs_last(codes.contiguous())  # [H_kv, window, D // 4]
    return packed, scale, zero


def dequantize_value_window(
    packed: Tensor,
    scale: "QGrid",
    zero: "QGrid",
    head_dim: int,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Inverse of :func:`quantize_value_window`.

    Returns token-major ``[H_kv, window, D]`` in ``out_dtype``.
    """
    codes = unpack_crumbs_last(packed, head_dim)   # [H_kv, window, D]
    return _affine_dequantize(
        codes, scale.decode().unsqueeze(2), zero.decode().unsqueeze(2), out_dtype
    )


def dequantize_value_windows(
    packed: Tensor,
    scale: "QGrid",
    zero: "QGrid",
    head_dim: int,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Batched :func:`dequantize_value_window` over a leading window axis.

    Parameters
    ----------
    packed : uint8 ``[N, H_kv, window, D // 4]`` — token-major, 4 channels/byte.
    scale, zero : :class:`QGrid` — one grid entry per ``(window, head, token)``.
    head_dim : int — original (unpacked) channel count.

    Returns
    -------
    ``[N, H_kv, window, D]`` token-major, in ``out_dtype``.
    """
    codes = unpack_crumbs_last(packed, head_dim)   # [N, H_kv, window, D]
    return _affine_dequantize(
        codes, scale.decode().unsqueeze(3), zero.decode().unsqueeze(3), out_dtype
    )
