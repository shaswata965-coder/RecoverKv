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

Shapes
------
The leading axis is opaque, matching :class:`~modules.quant.slots.QuantSlotTable`
(which flattens ``(row, slot)`` into one axis before handing tensors to the
quantisers). Everything here takes ``[N, H, ws, D]`` keys and returns
``[N, H, ...]`` fields.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import torch
from torch import Tensor

__all__ = [
    "Sketch",
    "sketch_bytes_per_head",
    "build_sketch",
    "decode_sketch",
    "value_centroid",
    "gate_and_score",
    "select_windows",
    "group_max",
    "head_log_partition",
    "in_head_units",
]


def sketch_bytes_per_head(head_dim: int, window_size: int) -> int:
    """Bytes one head's card costs — for the budget arithmetic.

    ``mu`` **int4**(D, packed 2/byte) + fp16 scale, ``v`` int8(D) + fp16 scale,
    ``t`` int8(ws) + fp16 scale, ``vbar`` **int4**(D, packed) + fp16 scale.

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
    the card is 2176 B against the codes-plus-grid's 6432 — down from 3200 B
    when ``mu`` and ``vbar`` were int8. Unbudgeted, that would be a 34% overrun
    on the tier the memory claim is made about.
    """
    return 2 * (head_dim // 2 + 2) + (head_dim + 2) + (window_size + 2)


class Sketch(NamedTuple):
    """One window's card, per head. Leading axis opaque (``N``)."""

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
_NIB_BIAS = 8
_NIB_MAX = 7


def _q_sym4(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Symmetric int4, **packed two per byte** along the last axis.

    Returns ``(packed uint8 [..., D//2], scale fp16 [...])``.

    The packing order matches the int2 crumb packer in
    :mod:`modules.quant.quantizer`: element ``j`` lives in byte ``j // 2`` at
    shift ``4 * (j % 2)``, low nibble first. Keeping one convention across both
    widths means the kernel's unpack is the same idiom twice, not two idioms.

    This is a REAL int4 encoder, not a coarsened int8 one: the scale is re-fit to
    ``amax / 7`` rather than inherited from the wider grid, so the codes use the
    whole 4-bit range. The ablation that cleared this change simulated int4 by
    rounding int8 codes against the int8 scale, which is strictly worse — so the
    measured accuracy is a floor, not an estimate.
    """
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / _NIB_MAX, torch.ones_like(amax))
    s16 = scale.to(torch.float16)
    s32 = s16.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    codes = torch.round(x / s32).clamp_(-_NIB_MAX, _NIB_MAX)
    nib = (codes + _NIB_BIAS).to(torch.uint8)                   # [..., D] in [1,15]
    lo, hi = nib[..., 0::2], nib[..., 1::2]
    return (lo | (hi << 4)), s16.squeeze(-1)


def _dq_sym4(packed: Tensor, scale: Tensor) -> Tensor:
    """Unpack + dequantize :func:`_q_sym4`. ``[..., D//2]`` -> ``[..., D]``."""
    lo = (packed & 0x0F).to(torch.int16) - _NIB_BIAS
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - _NIB_BIAS
    codes = torch.stack([lo, hi], dim=-1).flatten(-2)           # interleave back
    return codes.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)


# ---------------------------------------------------------------------------
# build — one pass, at demotion
# ---------------------------------------------------------------------------


def build_sketch(
    keys: Tensor,
    anchor: Tensor,
    values: Tensor,
    v_anchor: Tensor,
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

    mu_q, mu_s = _q_sym4(mu - anc)
    v_q, v_s = _q_sym(v)
    t_q, t_s = _q_sym(t)

    vbar = values.to(torch.float32).mean(dim=-2)               # [N, H, D]
    vm_q, vm_s = _q_sym4(vbar - _align_anchor(v_anchor, vbar))

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


def value_centroid(s: Sketch, v_anchor: Tensor) -> Tensor:
    """``[..., H, D]`` — the window's representative value vector.

    What a window contributes to the attention output when the gate did not read
    it. Its weight is the card's mass estimate rescaled by the deviation the read
    windows measured this step, so the term is calibrated against real attention
    on every step rather than trusted open-loop.
    """
    vm = _dq_sym4(s.vm_q, s.vm_s)
    return vm + _align_anchor(v_anchor, vm)


def decode_sketch(s: Sketch, anchor: Tensor):
    """``(mu_hat, v_hat, t_hat)`` — the decoded card."""
    # Align against the DECODED mu, not the stored codes: mu_q's last axis is
    # D//2 now, so aligning on it would broadcast the anchor onto half a head.
    mu = _dq_sym4(s.mu_q, s.mu_s)
    return (
        _align_anchor(anchor, mu) + mu,
        _dq_sym(s.v_q, s.v_s),
        _dq_sym(s.t_q, s.t_s),
    )


# ---------------------------------------------------------------------------
# read — every decode step
# ---------------------------------------------------------------------------


def gate_and_score(
    q: Tensor,
    s: Sketch,
    anchor: Tensor,
    scaling: float,
) -> Tuple[Tensor, Tensor]:
    """Per-window ``(logmass, est)`` from two dot products.

    Parameters
    ----------
    q : ``[B, Hq, D]`` post-RoPE decode query.
    s : card fields with leading axes ``[B, Nw, Hkv, ...]``.
    anchor : ``[Hkv, D]`` or ``[B, Hkv, D]``.
    scaling : the attention ``1 / sqrt(head_dim)``.

    Returns
    -------
    logmass : ``[B, Hq, Nw]`` ``logsumexp_i(scaling * q.k_i_hat)`` -- what a
        skipped window contributes, both to ``window_scores`` and (weighted by
        the step's measured deviation) to the attention output.
    est : ``[B, Hq, Nw]`` point estimate ``max_i scaling * q.k_i_hat``. The
        selection ranks on this.

    Both live in the log domain, so they are comparable across windows and cannot
    overflow. Two ``D``-length dots (``q.mu``, ``q.v``) yield both; ``t`` is 10
    bytes and rides in the cache line the vectors already pulled in.

    **There is no bound any more.** The card used to return a Cauchy-Schwarz
    upper bound alongside the estimate, for a ``quant_gate_margin`` rule that the
    fused gate never implemented. Ranking on the estimate is not a weakening of
    that: the estimate was already the better ranking quantity (the bound's slack
    term lets a loosely-bounded window outrank a tighter one carrying a higher
    true logit, measured at 100.0% worst-head mass recall against the bound's
    99.7%). What is gone is a guarantee nothing consumed, and the ``eps`` field
    that paid for it.
    """
    B, Hq, D = q.shape
    Nw, Hkv = s.mu_q.shape[1], s.mu_q.shape[2]
    rep = Hq // Hkv
    qf = q.to(torch.float32)

    mu_r = _dq_sym4(s.mu_q, s.mu_s)                                  # [B,Nw,Hkv,D]
    v_h = _dq_sym(s.v_q, s.v_s)
    t_h = _dq_sym(s.t_q, s.t_s)                                      # [B,Nw,Hkv,ws]

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

    **A max across heads is only a union if the heads are in the same units.**
    Raw logits are not: each query head has its own softmax, and a constant
    added to every one of its logits leaves that head's attention unchanged
    while moving it up or down against its siblings here. Feed this
    :func:`in_head_units` when the heads' offsets differ, or the head with the
    largest offset picks for the whole group (see that function).
    """
    B, Hq, Nw = est.shape
    return est.reshape(B, num_kv_heads, Hq // num_kv_heads, Nw).amax(2)


def head_log_partition(logmass: Tensor) -> Tensor:
    """``[B, Hq, Nw]`` -> ``[B, Hq, 1]``: each query head's log-partition over
    the Q tier, ``log sum_w exp(logmass_w)``, from the cards.

    Written out rather than calling ``torch.logsumexp``: ``scripts/audit_e2e.py``
    keys its attribution on that call being unique across ``modules/``, and the
    kernel merges per-tile (max, sum) partials in exactly this shape.
    """
    m = logmass.amax(dim=-1, keepdim=True)
    return m + (logmass - m).exp().sum(dim=-1, keepdim=True).log()


def in_head_units(est: Tensor, logmass: Tensor) -> Tensor:
    """``est - head_log_partition(logmass)``: the hot token's estimated share of
    its own head's Q-tier attention, in log units. What :func:`group_max` should
    compare across a GQA group.

    Why the raw estimate is the wrong thing to take a max over
    ------------------------------------------------------------
    ``est`` is a logit. A query head's softmax discards any constant added to all
    of its logits, and the model never had a reason to keep that constant
    similar across heads, so it is not: every head carries ``q_h . k_common``,
    its query's projection onto the keys' common mode, and that is a different
    number for each head. A cross-head max over raw logits is therefore decided
    by those offsets, not by what any head attends to. Once one head's offset
    beats its siblings' by more than their windows' spread, that head picks the
    whole group's ``n_sel`` and every other head's hot windows are skipped --
    read only through a value centroid, which for a head attending to one token
    is the wrong vector.

    A q/k bias (Qwen2) makes the offsets larger by construction: the bias is a
    token-independent key component, and each head's query projects onto it
    differently. Through the mid-frequency RoPE pairs the same bias also puts a
    smooth positional swing on some heads' logits -- real attention, but on a
    larger logit scale than a sibling's content match, so it outvotes that head
    too. Subtracting the anchor term ``q . anchor`` removes the first and not the
    second (in a synthetic check with +-20-nat positional heads it did no better
    than the raw union for the retrieval heads beside them); dividing each head
    by its own partition function removes both, because it compares heads in
    probability, which is the only unit they share.

    Measured on ``tests/test_sketch.py``'s own recall fixture (keys with a
    massive-activation common mode), per QUERY head at ratio 0.25: the raw union
    leaves the worst head 0.0000 of its Q-tier mass at both rep=4 (Llama-3.1
    GQA) and rep=7 (Qwen2.5-7B), mean 0.947 / 0.857; in head units the worst head
    keeps 0.998, mean 0.9999 / 0.9997. The pooled per-KV-head recall that test
    asserts on sums raw ``exp(logit)`` across the group first, so it is
    dominated by the same head the raw union serves and could not see this.

    Within one head this is a constant shift, so a head's own ranking and every
    ``rep == 1`` selection are unchanged; only the cross-head comparison moves.
    ``est`` rather than ``logmass`` stays the ranked quantity for the reason the
    module docstring gives: the centroid a skipped window attends through is
    exact for a window whose weight is spread evenly and wrong for one with a
    single hot token, and the hot token is what ``est`` measures.
    """
    return est - head_log_partition(logmass)


def select_windows(est: Tensor, max_windows: int) -> Tensor:
    """Keep-mask over the window axis: the top ``max_windows`` by estimate.

    Call it on **KV-head** estimates (i.e. after :func:`group_max`), not on
    query-head ones: one kernel program owns a KV head and every query head
    sharing it, so a window is loaded once for the whole group and the KV head is
    the unit the cap has to bind on. Capping per query head and unioning
    afterwards lets a ratio ``r`` read up to ``r * rep`` windows — at ``rep = 4``
    a 25% cap becomes 100% loaded and the gate buys nothing.

    The margin rule is gone with ``eps``. It thresholded on a Cauchy-Schwarz
    bound the fused gate never implemented, and the estimate was already the
    better ranking quantity — 100.0% worst-head mass recall at a 0.15 ratio
    against the bound's 99.7%. What is left is a deterministic top-k: one
    criterion, no calibration, no host sync, no data-dependent shape.

    Those recall figures pool raw ``exp(logit)`` over each KV head's query heads,
    which is blind to a query head the union never serves -- see
    :func:`in_head_units` for that failure and its per-query-head numbers.
    """
    rank = torch.argsort(torch.argsort(est, dim=-1, descending=True), dim=-1)
    return rank < max_windows


