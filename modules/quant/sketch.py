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

which gives a per-token estimate from two dot products, plus a per-token
residual ``eps_i`` that makes the gate a genuine **upper bound** rather than a
guess:

    |q.k_i - (q.mu + t_i * q.v)|  <=  ||q|| * eps_i        (Cauchy-Schwarz)

so a window whose bound falls below the threshold provably carries no logit above
it. The gate can over-select; it cannot miss. ``v`` points at the token furthest
from the mean — in a sparse window that is the hot one — so the model spends its
one direction on the token the gate exists to find.

Construction is one pass (mean, farthest point, projection, residual) with a
fixed instruction count and no convergence loop. This is greedy k-center, not
Lloyd's: the gate is a max-bound, and k-center minimises exactly the radius that
loosens it, without iterating.

Encoding, and what measurement decided
--------------------------------------
``mu`` and ``v`` are int8 with a per-(window, head) scale; ``t`` int8; ``eps``
uint8. 280 B per head, 2240 B per window at ``H_kv=8, D=128, ws=8`` — 26.5% of
``b_q`` (8448 B).

Three encoding choices were measured on synthetic keys carrying massive-
activation channels and one outlier token per window (``tests/test_sketch.py``
pins the conclusions):

* **``v`` must be int8.** A 1-bit sign vector leaves the hot token with the
  *largest* residual in its window (38.0 vs a 15.0 mean), worse than useless for
  the case the card exists to serve; int4 is nearly as bad (7.0). int8 brings it
  to 0.39 against a 9.2 mean — the hot token becomes the best-fit token, which is
  the property the whole design turns on.
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

The bound survives quantisation
-------------------------------
``eps`` is computed against the **decoded** ``mu``/``v``/``t``, not against the
ideal rank-1 fit, and is quantised by rounding **up**. So Cauchy-Schwarz holds
for the values actually stored, not for the values we wished we had stored.
:func:`build_sketch` therefore encodes, decodes, and only then measures the
residual — that order is the guarantee, not an implementation detail.

Shapes
------
The leading axis is opaque, matching :class:`~modules.quant.slots.QuantSlotTable`
(which flattens ``(row, slot)`` into one axis before handing tensors to the
quantisers). Everything here takes ``[N, H, ws, D]`` keys and returns
``[N, H, ...]`` fields.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "Sketch",
    "sketch_bytes_per_head",
    "build_sketch",
    "decode_sketch",
    "gate_and_score",
    "select_windows",
    "group_max",
]


def sketch_bytes_per_head(head_dim: int, window_size: int) -> int:
    """Bytes one head's card costs — for the budget arithmetic.

    ``mu`` int8(D) + fp16 scale, ``v`` int8(D) + fp16 scale, ``t`` int8(ws) +
    fp16 scale, ``eps`` uint8(ws) + fp16 scale.
    """
    return 2 * (head_dim + 2) + 2 * (window_size + 2)


class Sketch(NamedTuple):
    """One window's card, per head. Leading axis opaque (``N``)."""

    mu_q: Tensor      # [N, H, D]    int8   — mu - anchor
    mu_s: Tensor      # [N, H]       fp16
    v_q: Tensor       # [N, H, D]    int8   — unit deviation direction
    v_s: Tensor       # [N, H]       fp16
    t_q: Tensor       # [N, H, ws]   int8   — projection onto v
    t_s: Tensor       # [N, H]       fp16
    e_q: Tensor       # [N, H, ws]   uint8  — residual norm, rounded UP
    e_s: Tensor       # [N, H]       fp16


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


def _q_up(x: Tensor, bits_max: int = 255) -> Tuple[Tensor, Tensor]:
    """Unsigned uint8 rounding **UP**: guarantees ``dequant >= x``.

    Rounding to nearest would let the stored residual fall below the true one on
    half the tokens, which is precisely the guarantee the gate is built on. The
    scale is nudged before the fp16 cast so a downward fp16 rounding cannot push
    ``x / scale`` past ``bits_max`` and get clamped back under ``x``.
    """
    amax = x.amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / bits_max, torch.ones_like(amax))
    s16 = (scale * (1.0 + 2.0 ** -9)).to(torch.float16)
    s32 = s16.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    codes = torch.ceil(x / s32).clamp_(0, bits_max).to(torch.uint8)
    return codes, s16.squeeze(-1)


def _dq_up(codes: Tensor, scale: Tensor) -> Tensor:
    return codes.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)


# ---------------------------------------------------------------------------
# build — one pass, at demotion
# ---------------------------------------------------------------------------


def build_sketch(keys: Tensor, anchor: Tensor) -> Sketch:
    """One pass over a window's **post-RoPE** keys -> its card.

    Parameters
    ----------
    keys : ``[N, H, ws, D]`` float — post-RoPE keys, as they are at demotion
        *before* ``unrotate_key_window`` strips RoPE. Taking them at that point
        is what makes the card cost zero extra rotation, and it is frozen for
        life because eviction never rebases positions (§5, §10).
    anchor : ``[H, D]`` float — the per-(layer, head) common mode, frozen at the
        first eviction.
    """
    k = keys.to(torch.float32)
    N, H, ws, D = k.shape
    anc = anchor.unsqueeze(0) if anchor.dim() == 2 else anchor   # [1|N, H, D]

    mu = k.mean(dim=-2)                                        # sweep 1
    dev = k - mu.unsqueeze(-2)
    nrm = dev.norm(dim=-1)                                     # [N, H, ws]
    far = nrm.argmax(dim=-1)                                   # farthest point
    fdev = torch.gather(dev, -2, far[..., None, None].expand(N, H, 1, D))
    fnrm = torch.gather(nrm, -1, far[..., None]).clamp_min(1e-12)
    v = fdev.squeeze(-2) / fnrm                                # sweep 2, unit
    t = (dev * v.unsqueeze(-2)).sum(-1)                        # sweep 3

    mu_q, mu_s = _q_sym(mu - anc)
    v_q, v_s = _q_sym(v)
    t_q, t_s = _q_sym(t)

    # eps against what is ACTUALLY STORED, not against the ideal fit.
    mu_h = anc + _dq_sym(mu_q, mu_s)
    v_h = _dq_sym(v_q, v_s)
    t_h = _dq_sym(t_q, t_s)
    recon = mu_h.unsqueeze(-2) + t_h.unsqueeze(-1) * v_h.unsqueeze(-2)
    e_q, e_s = _q_up((k - recon).norm(dim=-1))

    return Sketch(mu_q, mu_s, v_q, v_s, t_q, t_s, e_q, e_s)


def decode_sketch(s: Sketch, anchor: Tensor):
    """``(mu_hat, v_hat, t_hat, eps_hat)`` — the decoded card."""
    anc = anchor.unsqueeze(0) if anchor.dim() == 2 else anchor
    return (
        anc + _dq_sym(s.mu_q, s.mu_s),
        _dq_sym(s.v_q, s.v_s),
        _dq_sym(s.t_q, s.t_s),
        _dq_up(s.e_q, s.e_s),
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
    """Per-window ``(bound, logmass)`` from two dot products.

    Parameters
    ----------
    q : ``[B, Hq, D]`` post-RoPE decode query.
    s : card fields with leading axes ``[B, Nw, Hkv, ...]``.
    anchor : ``[Hkv, D]``.
    scaling : the attention ``1 / sqrt(head_dim)``.

    Returns
    -------
    bound : ``[B, Hq, Nw]`` upper bound on ``max_i scaling * q.k_i``.
    logmass : ``[B, Hq, Nw]`` ``logsumexp_i(scaling * q.k_i_hat)`` -- what a
        skipped window contributes to ``window_scores``.
    est : ``[B, Hq, Nw]`` point estimate ``max_i scaling * q.k_i_hat``, i.e. the
        bound without its slack term. The cap ranks on this; see
        :func:`select_windows` for why the two are not the same quantity.

    Both live in the log domain, so they are comparable across windows and cannot
    overflow. Two ``D``-length dots (``q.mu``, ``q.v``) yield **both**: ``t`` and
    ``eps`` are 10 bytes each and ride in the cache line the vectors already
    pulled in.
    """
    B, Hq, D = q.shape
    Nw, Hkv = s.mu_q.shape[1], s.mu_q.shape[2]
    rep = Hq // Hkv
    qf = q.to(torch.float32)
    qn = qf.norm(dim=-1)                                             # [B, Hq]

    mu_r = _dq_sym(s.mu_q, s.mu_s)                                   # [B,Nw,Hkv,D]
    v_h = _dq_sym(s.v_q, s.v_s)
    t_h = _dq_sym(s.t_q, s.t_s)                                      # [B,Nw,Hkv,ws]
    e_h = _dq_up(s.e_q, s.e_s)

    qg = qf.reshape(B, Hkv, rep, D)
    a_base = (torch.einsum("bhrd,hd->bhr", qg, anchor) if anchor.dim() == 2
              else torch.einsum("bhrd,bhd->bhr", qg, anchor))        # [B,Hkv,rep]
    m = a_base.unsqueeze(1) + torch.einsum("bhrd,bnhd->bnhr", qg, mu_r)
    g = torch.einsum("bhrd,bnhd->bnhr", qg, v_h)

    x = scaling * (m.unsqueeze(-1) + t_h.unsqueeze(-2) * g.unsqueeze(-1))
    slack = scaling * qn.reshape(B, Hkv, rep)[:, None, :, :, None] \
        * e_h.unsqueeze(-2)                                          # [B,Nw,Hkv,rep,ws]

    def flat(z):                                                     # -> [B,Hq,Nw]
        return z.permute(0, 2, 3, 1).reshape(B, Hq, Nw)

    # logsumexp written out rather than called. Two reasons, both deliberate:
    # `scripts/audit_e2e.py` keys its compute_lse attribution on `torch.logsumexp(`
    # being unique across `modules/` (tests/test_audit_e2e.py guards it), and this
    # max-subtract-exp-sum form is exactly what the kernel's online softmax does,
    # so the reference and the kernel share a shape as well as a result.
    xmax = x.amax(dim=-1, keepdim=True)
    logmass = (xmax + (x - xmax).exp().sum(dim=-1, keepdim=True).log()).squeeze(-1)
    return flat((x + slack).amax(-1)), flat(logmass), flat(x.amax(-1))


def group_max(bound: Tensor, num_kv_heads: int) -> Tensor:
    """``[B, Hq, Nw]`` -> ``[B, Hkv, Nw]``: the GQA group's union, as a bound.

    One kernel program owns a KV head and every query head sharing it, so a
    window is loaded once for the whole group and the group's cost is one window
    either way. Taking the **max** bound over the group before selecting is what
    makes that true: whatever any member of the group needs, the group keeps.

    This reduction MUST come before any cap. Capping per query head and unioning
    afterwards lets a ratio of ``r`` turn into as much as ``r * rep`` windows
    actually read -- at ``rep = 4`` (Llama-3.1-8B) a 25% cap becomes 100% loaded
    and the gate buys nothing. ``tests/test_sketch_store.py`` pins this.
    """
    B, Hq, Nw = bound.shape
    return bound.reshape(B, num_kv_heads, Hq // num_kv_heads, Nw).amax(2)


def select_windows(
    bound: Tensor,
    margin: float,
    max_windows: Optional[int] = None,
    rank_by: Optional[Tensor] = None,
) -> Tensor:
    """Keep-mask with the same shape as ``bound``.

    Call it on **KV-head** bounds (i.e. after :func:`group_max`), not on query-
    head bounds: the KV head is the unit of work, so it is the unit the cap has
    to bind on.

    The threshold is relative to **each head's own maximum**, so it is a top-p in
    log space: a peaked head admits few windows, a flat head admits many, with no
    calibration, no sort, no host sync and no data-dependent shape. ``margin =
    inf`` selects everything, which is how the feature ships dark.

    ``bound`` and ``rank_by`` play different roles and that split is deliberate.
    The margin thresholds on the **bound**, where "cannot miss" is a real
    guarantee: every window is kept whose logit could possibly exceed
    ``best - margin``. The cap top-k's on ``rank_by`` -- the point **estimate**,
    with no slack term -- because a cap has no safety guarantee from either
    quantity (it keeps ``k`` windows whatever their bounds say), so the right
    criterion is simply the more accurate one. Measured on a shape-C tier
    (271 windows, Llama-3.1-8B GQA), ranking by estimate holds 100.0% of the
    attention mass at a 0.15 ratio against the bound's 99.7% worst-head, and
    costs less: the ranking path never touches ``eps``.

    ``rank_by`` defaults to ``bound`` so a caller that has only the bound still
    gets sane behaviour.
    """
    top = bound.amax(dim=-1, keepdim=True)
    keep = torch.ones_like(bound, dtype=torch.bool) if margin == float("inf") \
        else (bound >= top - margin)
    keep = keep | (bound == top)                            # always the best one
    if max_windows is not None and max_windows < bound.shape[-1]:
        crit = bound if rank_by is None else rank_by
        rank = torch.argsort(torch.argsort(crit, dim=-1, descending=True), dim=-1)
        keep = keep & (rank < max_windows)
    return keep


