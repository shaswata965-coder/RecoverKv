"""Per-window rank-1 sketches — the gate that lets decode skip int2 windows.

The problem
-----------
Every decode step dequantises **every** active int2 window so it can attend over
it and accrue ``window_scores`` (design §8). Most of those windows contribute
almost nothing: the Q tier is by construction the band ranked *below* the fp
tier. We pay full price to discover they were cheap.

The fix is a card written once per window, when the window is sealed. Cards are
small enough to scan cheaply; the scan says which windows are worth
dequantising, and only those are read. The card is frozen for life and
reactivated on re-demotion, exactly like the codes and the pinned grid (§10) —
it can be, because eviction never rebases positions, so a post-RoPE statistic
computed at demotion stays valid forever.

Why a mean is not enough
------------------------
The obvious card is the window's mean key. It fails, and it fails on exactly the
case that matters: a window holding one hot token and seven cold ones has a mean
that looks cold, so the gate skips the window the query needed. ``exp`` is
convex, so a centroid *always* understates a window's mass, and understates it
most when the window is internally mixed. (Tactic, arXiv 2502.12216 §3.7, hits
this wall and patches it with a runtime curve fit.)

The card here is a **rank-1 model** of the window instead of a point::

    k_i  ~=  mu  +  t_i * v          v = direction of the farthest deviation

which gives a per-token estimate from two dot products. ``v`` points at the
token furthest from the mean — in a sparse window that is the hot one — so the
model spends its one direction on the token the gate exists to find.

Construction is one pass (mean, farthest point, projection) with a fixed
instruction count and no convergence loop. This is greedy k-center, not Lloyd's:
k-center minimises exactly the radius that would loosen the estimate, without
iterating.

Encoding, and what measurement decided
--------------------------------------
``mu`` and ``vbar`` are **int4**, packed two per byte; ``v`` and ``t`` are int8.
All four carry a per-(window, head) fp16 scale. **272 B per head, 2176 B per
window** at ``H_kv=8, D=128, ws=8``.

The width split is the whole economics of the gate, so it is worth stating why.
A card must be read for *every* window on *every* step — that is what selecting
means — while a window is only opened if selected. So the gate costs
``card + ratio * window`` against ``window``, and break-even sits at
``1 - card/window``. At int8 the card was 400 B against a 804 B window of token
data: **78% of the thing it summarises**, break-even at a 0.50 read ratio, and
measurably worth nothing at the shipped 0.25. At int4 it is 272 B, and
break-even moves to **0.66**. The card was written at four times the precision
of the 2-bit tokens it stands in for; that, not the idea, is what made skimming
expensive.

Four encoding choices were measured on synthetic keys carrying massive-
activation channels and one outlier token per window (``tests/test_sketch.py``
pins the conclusions):

* **``v`` must be int8.** A 1-bit sign vector leaves the hot token with the
  *largest* residual in its window (38.0 vs a 15.0 mean), worse than useless for
  the case the card exists to serve; int4 is nearly as bad (7.0). int8 brings it
  to 0.39 against a 9.2 mean — the hot token becomes the best-fit token, which is
  the property the whole design turns on. This is why the int4 change took
  ``mu`` and ``vbar`` and left ``v`` alone.
* **``mu`` and ``vbar`` do not need int8.** Both are means, both stored as a
  residual from a frozen anchor, and coarsening them to int4 leaves the gate's
  selection and the skipped windows' contribution unchanged at the shipped read
  ratio. They are 64% of the card, which is where the 400 -> 272 B comes from.
* **``mu`` is stored as a residual from a frozen per-(layer, head) anchor.**
  Massive-activation channels are near-identical across windows, so they carry no
  information that separates one window from another while eating the entire
  int8 range. Subtracting the anchor first cuts ``mu`` reconstruction error 17x
  (0.078 -> 0.0046).
* **No Hadamard rotation.** An earlier draft rotated both vectors into a
  Hadamard domain to densify them before coding. Measured against the anchored
  int8 encoding it changes ``mu`` error from 0.0046 to 0.0042 and leaves the mass
  estimate identical to three decimals. It was load-bearing only for the 1-bit
  variant, which is gone, so it is gone too — one fewer matmul on the launch-
  bound eviction path, one fewer transform in the kernel prologue, and no
  resident transform matrix.

The width knob: ``quant_card_bits``
-----------------------------------
Every field's width is a setting, :class:`CardBits` ``(mu, v, t, vm)``, each one
of **8, 4 or 2** bits. The default is the measured split above,
``CardBits(mu=4, v=8, t=8, vm=4)``, and at the default every path -- the
encoder, the slot table, the budget, both Triton kernels -- is the code it was
before the knob existed: the width reaches the kernels as a ``constexpr``, so a
default launch compiles to the same unpack it always did. The findings above
are why the default is what it is; the knob exists to re-measure them, not to
overrule them.

Every width uses one rule, :func:`_q_symb`: symmetric, per-(window, head) fp16
scale fit to ``amax / qmax`` with ``qmax = 2**(bits-1) - 1``. 8 bits is stored
as plain ``int8``, unpacked (unchanged). 4 and 2 bits are stored as ``code +
2**(bits-1)`` so every lane is unsigned, packed ``8 // bits`` per byte along the
last axis, element ``j`` at byte ``j // per``, shift ``bits * (j % per)`` --
low lanes first, the int2 crumb packer's convention. **int2 is therefore
ternary**, ``{-1, 0, +1} * scale``: symmetric with an exact zero, the same rule
as int4 one width down. A window's card bytes follow from the widths through
:func:`sketch_bytes_per_head`, which is what the budget prices, so a narrower
card buys int2 windows under ``quant_budget_mode='bytes'`` exactly as the
int8 -> int4 change did.

Shapes
------
The leading axis is opaque, matching :class:`~modules.quant.slots.QuantSlotTable`
(which flattens ``(row, slot)`` into one axis before handing tensors to the
quantisers). Everything here takes ``[N, H, ws, D]`` keys and returns
``[N, H, ...]`` fields.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "CardBits",
    "CARD_BITS_DEFAULT",
    "CARD_BITS_ALLOWED",
    "parse_card_bits",
    "card_field_layout",
    "Sketch",
    "sketch_bytes_per_head",
    "build_sketch",
    "decode_sketch",
    "value_centroid",
    "gate_and_score",
    "select_windows",
    "group_max",
    "head_log_share",
    "group_share",
]


class CardBits(NamedTuple):
    """Bits per element of each card field: ``(mu, v, t, vm)``.

    The default is the measured split (module docstring): ``mu`` and ``vm`` are
    means stored as residuals from a frozen anchor and take int4; ``v`` is the
    deviation direction and ``t`` the per-token projection, and both stay int8.
    Each field accepts 8, 4 or 2 -- see the module docstring for the encoding.
    """

    mu: int = 4
    v: int = 8
    t: int = 8
    vm: int = 4


#: The shipped widths. Every default in this module, the slot table, the store
#: and both config classes is this object, so "no knob set" is one definition.
CARD_BITS_DEFAULT = CardBits()
#: Widths a card field can take. 8 is unpacked int8; 4 and 2 are packed.
CARD_BITS_ALLOWED = (8, 4, 2)


def _one_width(name: str, val: Any) -> int:
    """One field's width from an int or an ``"int4"``-style string."""
    if isinstance(val, bool):
        raise ValueError(f"quant_card_bits.{name} must be 8, 4 or 2, got bool")
    if isinstance(val, str):
        txt = val.strip().lower()
        txt = txt[3:] if txt.startswith("int") else txt
        try:
            val = int(txt)
        except ValueError:
            raise ValueError(
                f"quant_card_bits.{name} must be 8, 4 or 2, got {val!r}") from None
    if not isinstance(val, int) or val not in CARD_BITS_ALLOWED:
        raise ValueError(
            f"quant_card_bits.{name} must be one of {CARD_BITS_ALLOWED}, got "
            f"{val!r}")
    return val


def parse_card_bits(spec: Any) -> CardBits:
    """Normalise the ``quant_card_bits`` knob to a validated :class:`CardBits`.

    Accepted forms, so a YAML value and a CLI string both work:

    * ``None`` -- the default, :data:`CARD_BITS_DEFAULT`.
    * one width (``4``, ``"int2"``) -- **every** field at that width.
    * a mapping over any subset of ``mu, v, t, vm`` (``{"v": 4}``) -- those
      fields overridden, the rest at their default.
    * ``"mu=4,v=4,t=2,vm=2"`` -- the same, as a string.
    * a 4-sequence in ``(mu, v, t, vm)`` order, including a :class:`CardBits`.

    Anything else raises: a typo'd field name silently falling back to the
    default would run the default card under a config that says otherwise.
    """
    if spec is None:
        return CARD_BITS_DEFAULT
    if isinstance(spec, str) and "=" in spec:
        pairs = {}
        for part in spec.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(
                    f"quant_card_bits string must be 'field=bits,...', got {spec!r}")
            k, v = part.split("=", 1)
            pairs[k.strip()] = v
        spec = pairs
    if isinstance(spec, Mapping):
        unknown = set(spec) - set(CardBits._fields)
        if unknown:
            raise ValueError(
                f"quant_card_bits has no field(s) {sorted(unknown)}; the card's "
                f"fields are {list(CardBits._fields)}")
        return CARD_BITS_DEFAULT._replace(
            **{k: _one_width(k, v) for k, v in spec.items()})
    if isinstance(spec, (int, str)):
        w = _one_width("*", spec)
        return CardBits(w, w, w, w)
    if isinstance(spec, (tuple, list)):
        if len(spec) != len(CardBits._fields):
            raise ValueError(
                f"quant_card_bits as a sequence must be (mu, v, t, vm), got "
                f"{len(spec)} entries: {spec!r}")
        return CardBits(*(_one_width(n, v) for n, v in zip(CardBits._fields, spec)))
    raise ValueError(
        f"quant_card_bits must be None, a width, a mapping, a 'field=bits' "
        f"string or a 4-sequence; got {type(spec).__name__} {spec!r}")


def _as_bits(bits: Any) -> CardBits:
    """``bits`` if it is already a :class:`CardBits`, else parsed. The store and
    the cache always hold a parsed one, so on their paths this is one isinstance
    check -- which matters inside the compiled eviction."""
    return bits if isinstance(bits, CardBits) else parse_card_bits(bits)


def card_field_layout(n: int, bits: int) -> Tuple[int, torch.dtype]:
    """``(stored last-axis width, dtype)`` of an ``n``-element field at ``bits``.

    8 bits is ``(n, int8)``; 4 and 2 are ``(n * bits // 8, uint8)``. ``n`` must
    fill whole bytes -- a ragged tail would need a masked lane map in both
    kernels' inner loops.
    """
    if bits == 8:
        return n, torch.int8
    per = 8 // bits
    if n % per:
        raise ValueError(
            f"a {n}-element card field cannot be packed {per} per byte at "
            f"{bits} bits; it must be a multiple of {per}.")
    return n // per, torch.uint8


def sketch_bytes_per_head(head_dim: int, window_size: int,
                          bits: Any = CARD_BITS_DEFAULT) -> int:
    """Bytes one head's card costs — for the budget arithmetic.

    Each field is its packed codes plus one fp16 scale: ``mu``, ``v`` and
    ``vbar`` over ``head_dim`` elements, ``t`` over ``window_size``, each at its
    :class:`CardBits` width. At the default that is ``mu`` **int4**(D, packed
    2/byte) + fp16 scale, ``v`` int8(D) + fp16 scale, ``t`` int8(ws) + fp16
    scale, ``vbar`` **int4**(D, packed) + fp16 scale.

    ``mu`` and ``vbar`` are int4 and ``v`` is not, and that split is measured,
    not stylistic. ``v`` is the deviation DIRECTION -- the thing that lets the
    card find a window's one hot token -- and at int4 the hot token stops being
    the best-fit token, which is the single property the card exists for
    (``tests/test_sketch.py::test_int8_direction_is_required_not_a_preference``).
    ``mu`` and ``vbar`` are both means, both stored as residuals from a frozen
    anchor, and neither needs the range: coarsening them is accuracy-neutral at
    the shipped read ratio. ``t`` is 10 B of the card and cannot shrink either --
    without it the card is a plain average again.

    This number is LOAD-BEARING, not documentation: ``config.resolve`` adds it to
    ``bytes_per_q_window``, so a card field that is not counted here is a window
    the cache holds and the budget does not know about. At ``D=128, ws=8, H=8``
    the default card is 2176 B against the codes-plus-grid's 6432 — down from
    3200 B when ``mu`` and ``vbar`` were int8. Unbudgeted, that would be a 34%
    overrun on the tier the memory claim is made about.
    """
    b = _as_bits(bits)
    return sum(card_field_layout(n, w)[0] + 2 for n, w in (
        (head_dim, b.mu), (head_dim, b.v), (window_size, b.t), (head_dim, b.vm)))


class Sketch(NamedTuple):
    """One window's card, per head. Leading axis opaque (``N``).

    Shapes are at the default :class:`CardBits`; each ``*_q`` field's last axis
    and dtype follow :func:`card_field_layout` for its width.
    """

    mu_q: Tensor      # [N, H, D//2] uint8  — mu - anchor, int4 packed 2/byte
    mu_s: Tensor      # [N, H]       fp16
    v_q: Tensor       # [N, H, D]    int8   — unit deviation direction
    v_s: Tensor       # [N, H]       fp16
    t_q: Tensor       # [N, H, ws]   int8   — projection onto v
    t_s: Tensor       # [N, H]       fp16
    vm_q: Tensor      # [N, H, D//2] uint8  — value centroid - v_anchor, int4
    vm_s: Tensor      # [N, H]       fp16


# ---------------------------------------------------------------------------
# quantisation helpers
# ---------------------------------------------------------------------------


def _q_sym(x: Tensor, bits_max: int = 127) -> Tuple[Tensor, Tensor]:
    """Symmetric per-(N, H) int8. Fits codes to the **fp16-stored** scale, so the
    grid the codes were fit to is the grid every dequant uses (design §2)."""
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / bits_max, torch.ones_like(amax))
    s16 = scale.to(torch.float16)
    s32 = s16.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    codes = torch.round(x / s32).clamp_(-bits_max, bits_max).to(torch.int8)
    return codes, s16.squeeze(-1)


def _dq_sym(codes: Tensor, scale: Tensor) -> Tensor:
    return codes.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)


#: int4 nibble bias. Codes are symmetric in ``[-7, 7]``; storing ``code + 8``
#: makes every nibble unsigned in ``[1, 15]``, so unpacking is a shift-and-mask
#: with no sign extension — the same shape as the int2 crumb unpack the decode
#: kernel already does, which is why the Triton side is three extra lines.
#: :func:`_q_symb` is the same rule at any packed width: bias ``2**(bits-1)``,
#: codes in ``[-(2**(bits-1) - 1), 2**(bits-1) - 1]``.
_NIB_BIAS = 8
_NIB_MAX = 7


def _q_symb(x: Tensor, bits: int) -> Tuple[Tensor, Tensor]:
    """Symmetric ``bits``-wide codes along the last axis, + a per-row fp16 scale.

    ``bits == 8`` is :func:`_q_sym` -- plain int8, unpacked. ``bits`` of 4 or 2
    returns ``(packed uint8 [..., n * bits // 8], scale fp16 [...])``.

    The packing order matches the int2 crumb packer in
    :mod:`modules.quant.quantizer`: element ``j`` lives in byte ``j // per`` at
    shift ``bits * (j % per)``, low lanes first. Keeping one convention across
    every width means the kernels' unpack is the same idiom at every width.

    This is a REAL narrow encoder, not a coarsened int8 one: the scale is re-fit
    to ``amax / qmax`` rather than inherited from the wider grid, so the codes
    use the whole range. At 2 bits ``qmax`` is 1, so the codes are ternary.
    """
    if bits == 8:
        return _q_sym(x)
    per = 8 // bits
    qmax = (1 << (bits - 1)) - 1
    if x.shape[-1] % per:
        raise ValueError(
            f"cannot pack a last axis of {x.shape[-1]} {per} per byte")
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    s16 = scale.to(torch.float16)
    s32 = s16.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    codes = torch.round(x / s32).clamp_(-qmax, qmax)
    lanes = (codes + (1 << (bits - 1))).to(torch.uint8)       # [..., n] unsigned
    packed = lanes[..., 0::per]
    for j in range(1, per):
        packed = packed | (lanes[..., j::per] << (bits * j))
    return packed, s16.squeeze(-1)


def _dq_symb(packed: Tensor, scale: Tensor, bits: int) -> Tensor:
    """Unpack + dequantize :func:`_q_symb`. ``[..., n * bits // 8]`` -> ``[..., n]``."""
    if bits == 8:
        return _dq_sym(packed, scale)
    per = 8 // bits
    mask = (1 << bits) - 1
    bias = 1 << (bits - 1)
    lanes = [((packed if j == 0 else packed >> (bits * j)) & mask).to(torch.int16)
             - bias for j in range(per)]
    codes = torch.stack(lanes, dim=-1).flatten(-2)             # interleave back
    return codes.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)


def _q_sym4(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Symmetric int4, **packed two per byte** along the last axis.

    Returns ``(packed uint8 [..., D//2], scale fp16 [...])``. :func:`_q_symb` at
    4 bits: element ``j`` in byte ``j // 2`` at shift ``4 * (j % 2)``, low nibble
    first, scale re-fit to ``amax / 7``.
    """
    return _q_symb(x, 4)


def _dq_sym4(packed: Tensor, scale: Tensor) -> Tensor:
    """Unpack + dequantize :func:`_q_sym4`. ``[..., D//2]`` -> ``[..., D]``."""
    return _dq_symb(packed, scale, 4)


# ---------------------------------------------------------------------------
# build — one pass, at demotion
# ---------------------------------------------------------------------------


def build_sketch(
    keys: Tensor,
    anchor: Tensor,
    values: Tensor,
    v_anchor: Tensor,
    bits: Any = CARD_BITS_DEFAULT,
) -> Sketch:
    """One pass over a window's **post-RoPE** keys and values -> its card.

    Parameters
    ----------
    keys : ``[N, H, ws, D]`` float — post-RoPE keys, as they are at demotion
        *before* ``unrotate_key_window`` strips RoPE. Taking them at that point
        is what makes the card cost zero extra rotation, and it is frozen for
        life because eviction never rebases positions (§5, §10).
    anchor : ``[H, D]`` float — the per-(layer, head) key common mode, frozen at
        the first eviction.
    values : ``[N, H, ws, D]`` float — the window's values, for the centroid a
        skipped window contributes to the OUTPUT through.
    v_anchor : ``[H, D]`` float — the value-side common mode, same role as
        ``anchor``: values carry a strong per-head common mode that would
        otherwise eat the int8 range without separating one window from another.
    bits : :class:`CardBits` (or anything :func:`parse_card_bits` takes) -- each
        field's width. The default is the shipped card.

    Why the value side is a plain mean while the key side is rank-1
    ---------------------------------------------------------------
    The two sides answer different questions. The key side decides *whether* a
    window can matter, so it needs an outlier-seeking direction — a centroid
    there would hide the one hot token (see the module docstring). The value side
    only has to answer *what* the window contributes once we have decided it does
    not matter much, and that contribution is a softmax-weighted mean of values.
    Inside a window the gate declined to read, the weights are by construction
    near-uniform, so the unweighted mean is the correct first-order term and a
    rank-1 value model would buy a second order we have no budget for.

    ``eps`` is gone
    ---------------
    The card used to carry a per-token residual norm, which turned the estimate
    into a Cauchy-Schwarz upper bound. Nothing on the production path ever read
    it: the bound exists to serve a finite ``quant_gate_margin``, and the fused
    gate is a top-k on the point estimate with no margin term. So the field was
    built, stored, budgeted and discarded. It is removed rather than zero-filled,
    because a zero eps is not a weaker bound — it is not a bound at all, and
    keeping the column invited a reader to think one was available.
    """
    k = keys.to(torch.float32)
    # ``N`` is the flattened row axis (``B * n`` from ``demote_many``) and stays
    # symbolic with ``B``; ``H``/``D`` are model constants and are forced, because
    # this is inside the compiled eviction and they are ``expand`` extents below.
    # ``ws`` was unpacked and never read.
    N = k.shape[0]
    H, D = int(k.shape[1]), int(k.shape[3])
    anc = _align_anchor(anchor, k[..., 0, :])

    mu = k.mean(dim=-2)                                        # sweep 1
    dev = k - mu.unsqueeze(-2)
    nrm = dev.norm(dim=-1)                                     # [N, H, ws]
    far = nrm.argmax(dim=-1)                                   # farthest point
    fdev = torch.gather(dev, -2, far[..., None, None].expand(N, H, 1, D))
    fnrm = torch.gather(nrm, -1, far[..., None]).clamp_min(1e-12)
    v = fdev.squeeze(-2) / fnrm                                # sweep 2, unit
    t = (dev * v.unsqueeze(-2)).sum(-1)                        # sweep 3

    b = _as_bits(bits)
    mu_q, mu_s = _q_symb(mu - anc, b.mu)
    v_q, v_s = _q_symb(v, b.v)
    t_q, t_s = _q_symb(t, b.t)

    vbar = values.to(torch.float32).mean(dim=-2)               # [N, H, D]
    vm_q, vm_s = _q_symb(vbar - _align_anchor(v_anchor, vbar), b.vm)

    return Sketch(mu_q, mu_s, v_q, v_s, t_q, t_s, vm_q, vm_s)


def _align_anchor(anchor: Tensor, like: Tensor) -> Tensor:
    """Broadcast an ``[H, D]`` or ``[B, H, D]`` anchor onto ``like``'s layout.

    The card's leading axes differ by caller and always have: ``build_sketch``
    sees a flattened ``[N, H, D]`` while ``QuantSlotTable.gather_sketch`` keeps
    the ``[B, n, H, D]`` pair. Both end in ``(H, D)``, so aligning on the trailing
    two dims and inserting singletons after the batch axis is the one rule that
    is right for every caller. Plain broadcasting is not: a ``[B, H, D]`` anchor
    against a ``[B, n, H, D]`` card lines ``H`` up with ``n`` and either raises
    or, at ``n == H``, silently mixes heads.
    """
    while anchor.dim() < like.dim():
        anchor = anchor.unsqueeze(0) if anchor.dim() == 2 else anchor.unsqueeze(1)
    return anchor


def value_centroid(s: Sketch, v_anchor: Tensor,
                   bits: Any = CARD_BITS_DEFAULT) -> Tensor:
    """``[..., H, D]`` — the window's representative value vector.

    What a window contributes to the attention output when the gate did not read
    it. Its weight is the card's mass estimate rescaled by the deviation the read
    windows measured this step, so the term is calibrated against real attention
    on every step rather than trusted open-loop.
    """
    vm = _dq_symb(s.vm_q, s.vm_s, _as_bits(bits).vm)
    return vm + _align_anchor(v_anchor, vm)


def decode_sketch(s: Sketch, anchor: Tensor, bits: Any = CARD_BITS_DEFAULT):
    """``(mu_hat, v_hat, t_hat)`` — the decoded card."""
    # Align against the DECODED mu, not the stored codes: a packed mu_q's last
    # axis is narrower than D, so aligning on it would broadcast the anchor onto
    # part of a head.
    b = _as_bits(bits)
    mu = _dq_symb(s.mu_q, s.mu_s, b.mu)
    return (
        _align_anchor(anchor, mu) + mu,
        _dq_symb(s.v_q, s.v_s, b.v),
        _dq_symb(s.t_q, s.t_s, b.t),
    )


# ---------------------------------------------------------------------------
# read — every decode step
# ---------------------------------------------------------------------------


def gate_and_score(
    q: Tensor,
    s: Sketch,
    anchor: Tensor,
    scaling: float,
    bits: Any = CARD_BITS_DEFAULT,
) -> Tuple[Tensor, Tensor]:
    """Per-window ``(logmass, est)`` from two dot products.

    Parameters
    ----------
    q : ``[B, Hq, D]`` post-RoPE decode query.
    s : card fields with leading axes ``[B, Nw, Hkv, ...]``.
    anchor : ``[Hkv, D]`` or ``[B, Hkv, D]``.
    scaling : the attention ``1 / sqrt(head_dim)``.
    bits : the card's :class:`CardBits` -- how ``s`` was encoded.

    Returns
    -------
    logmass : ``[B, Hq, Nw]`` ``logsumexp_i(scaling * q.k_i_hat)`` -- what a
        skipped window contributes, both to ``window_scores`` and (weighted by
        the step's measured deviation) to the attention output. The selection
        ranks on this too, normalised per query head -- see :func:`group_share`.
    est : ``[B, Hq, Nw]`` point estimate ``max_i scaling * q.k_i_hat``.
        Diagnostic only: it is a raw logit, and raw logits are not comparable
        across query heads, which is why the selection no longer ranks on it.

    Both live in the log domain, so they cannot overflow. Two ``D``-length dots
    (``q.mu``, ``q.v``) yield both; ``t`` is 10 bytes and rides in the cache line
    the vectors already pulled in.

    **Neither is comparable across query heads.** Each carries its head's own
    baseline (``q.anchor``) and scale (``|q|``), and a head's softmax is
    invariant to both. Comparing them across the heads of a GQA group is the
    defect :func:`group_share` exists to fix.

    **There is no bound any more.** The card used to return a Cauchy-Schwarz
    upper bound alongside the estimate, for a ``quant_gate_margin`` rule that the
    fused gate never implemented. What is gone is a guarantee nothing consumed,
    and the ``eps`` field that paid for it.
    """
    B, Hq, D = q.shape
    Nw, Hkv = s.mu_q.shape[1], s.mu_q.shape[2]
    rep = Hq // Hkv
    qf = q.to(torch.float32)

    b = _as_bits(bits)
    mu_r = _dq_symb(s.mu_q, s.mu_s, b.mu)                            # [B,Nw,Hkv,D]
    v_h = _dq_symb(s.v_q, s.v_s, b.v)
    t_h = _dq_symb(s.t_q, s.t_s, b.t)                                # [B,Nw,Hkv,ws]

    qg = qf.reshape(B, Hkv, rep, D)
    a_base = (torch.einsum("bhrd,hd->bhr", qg, anchor) if anchor.dim() == 2
              else torch.einsum("bhrd,bhd->bhr", qg, anchor))        # [B,Hkv,rep]
    m = a_base.unsqueeze(1) + torch.einsum("bhrd,bnhd->bnhr", qg, mu_r)
    g = torch.einsum("bhrd,bnhd->bnhr", qg, v_h)

    x = scaling * (m.unsqueeze(-1) + t_h.unsqueeze(-2) * g.unsqueeze(-1))

    def flat(z):                                                     # -> [B,Hq,Nw]
        return z.permute(0, 2, 3, 1).reshape(B, Hq, Nw)

    # logsumexp written out rather than called. Two reasons, both deliberate:
    # `scripts/audit_e2e.py` keys its compute_lse attribution on `torch.logsumexp(`
    # being unique across `modules/` (tests/test_audit_e2e.py guards it), and this
    # max-subtract-exp-sum form is exactly what the kernel's online softmax does,
    # so the reference and the kernel share a shape as well as a result.
    xmax = x.amax(dim=-1, keepdim=True)
    logmass = (xmax + (x - xmax).exp().sum(dim=-1, keepdim=True).log()).squeeze(-1)
    return flat(logmass), flat(x.amax(-1))


def group_max(est: Tensor, num_kv_heads: int) -> Tensor:
    """``[B, Hq, Nw]`` -> ``[B, Hkv, Nw]``: the GQA group's union.

    One kernel program owns a KV head and every query head sharing it, so a
    window is loaded once for the whole group and the group's cost is one window
    either way. Taking the **max** over the group before selecting is what makes
    that true: whatever any member of the group needs, the group keeps.

    This reduction MUST come before any cap. Capping per query head and unioning
    afterwards lets a ratio of ``r`` turn into as much as ``r * rep`` windows
    actually read -- at ``rep = 4`` (Llama-3.1-8B) a 25% cap becomes 100% loaded
    and the gate buys nothing. ``tests/test_sketch_store.py`` pins this.

    **The input must already be on one footing across query heads.** A max over
    raw logits is not a union: see :func:`group_share`, which is what the
    selection passes in.
    """
    B, Hq, Nw = est.shape
    return est.reshape(B, num_kv_heads, Hq // num_kv_heads, Nw).amax(2)


def head_log_share(logmass: Tensor) -> Tensor:
    """``[B, Hq, Nw]`` -> each query head's log share of ITS OWN int2-tier mass.

    ``logmass - logsumexp_w(logmass)``: the card's estimate of what fraction of
    this head's attention over the int2 tier each window holds, in the log
    domain. It is invariant to anything a softmax is invariant to -- a per-head
    constant on every logit (the head's ``q.anchor`` baseline) cancels exactly --
    and it puts every head of a GQA group on the same footing: ``0`` means "this
    window is all of my int2 mass", ``-log(Nw)`` means "I am spread evenly".

    The logsumexp is written out for the same two reasons as in
    :func:`gate_and_score`, and the max is guarded so a head whose every entry is
    ``-inf`` yields ``-inf`` shares rather than NaN.
    """
    m = logmass.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    lse = m + (logmass - m).exp().sum(dim=-1, keepdim=True).log()
    return logmass - lse


def group_share(logmass: Tensor, num_kv_heads: int) -> Tensor:
    """``[B, Hq, Nw]`` -> ``[B, Hkv, Nw]``: what the gate ranks windows on.

    For each window, the largest share of its own int2-tier mass that ANY query
    head of the GQA group puts on it::

        score(w) = max_h [ logmass_h(w) - logsumexp_w' logmass_h(w') ]

    **Why not the max of the raw estimates, which is what shipped.** A head's
    attention is a softmax over its own logits, so what a window is worth to a
    head is its share of THAT head's mass. The raw estimate also carries the
    head's baseline (``q.anchor``, which differs by several logit units between
    the heads of a group) and its scale (``|q|``). The max over raw estimates was
    therefore decided by whichever head had the largest baseline or scale, and
    every other head in the group was read only where that head happened to
    agree. On the repo's own recall fixture the group-summed metric said 1.000
    while the worst query head had **0.0005** of its mass read. At RULER geometry
    (840 windows x ws=32) a retrieval head's needle window was read on about
    **half** of the steps even when it held 95% of that head's int2 mass. A copy
    step can only reproduce a token its head actually reads, and a UUID is about
    25 copy steps long. ``tests/test_gate_union_is_per_head.py`` pins both.

    **What the share guarantees.** A head's shares sum to one, so at most
    ``rep / tau`` windows of a group can hold more than ``tau`` of any member's
    int2 mass. With ``n_sel`` windows read per KV head, every window that holds
    more than about ``rep / n_sel`` of ANY member's int2 mass is read. At the
    32k operating point that is about 2%. The raw max gave no such guarantee to
    any head but the dominant one.

    **Why ``logmass`` and not ``est``.** ``est`` is one token's rank-1 estimate.
    At ``ws = 32`` the token a query wants is rarely the card's farthest point,
    so that estimate is mostly noise from ``t_far * q.v``. ``logmass`` sums the
    whole window, which is the quantity a softmax consumes. Measured on the
    kernel's oracle at ``ws=32``, it reads the needle window on every seed
    (``est``-based: 60-90%), and gives lower output error on diffuse heads too.
    """
    return group_max(head_log_share(logmass), num_kv_heads)


def select_windows(est: Tensor, max_windows: int) -> Tensor:
    """Keep-mask over the window axis: the top ``max_windows`` by score.

    Call it on **KV-head** scores (i.e. after :func:`group_share`), not on
    query-head ones: one kernel program owns a KV head and every query head
    sharing it, so a window is loaded once for the whole group and the KV head is
    the unit the cap has to bind on. Capping per query head and unioning
    afterwards lets a ratio ``r`` read up to ``r * rep`` windows — at ``rep = 4``
    a 25% cap becomes 100% loaded and the gate buys nothing.

    The margin rule is gone with ``eps``. It thresholded on a Cauchy-Schwarz
    bound the fused gate never implemented. What is left is a deterministic
    top-k: one criterion, no calibration, no host sync, no data-dependent shape.
    """
    rank = torch.argsort(torch.argsort(est, dim=-1, descending=True), dim=-1)
    return rank < max_windows


