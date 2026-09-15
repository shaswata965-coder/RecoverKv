"""Bounding-volume digests over groups of KV-cache keys.

A mechanism-independent primitive. A *digest* is a geometric summary of a group
of keys, small enough to stay resident while the keys themselves are not read.
Given a query it answers exactly one question — "could anything in this group
have scored high?" — as an **upper bound** on the largest dot product the group
could produce:

    estimate(q, digest(K))  >=  max_{k in K} q . k

The bound is what makes it usable for ranking: a group whose bound is low
cannot contain a high-scoring key, so it can be skipped without being read; a
group whose bound is high might, so it is worth reading. Bounds are one-sided —
they never under-estimate, so a genuinely important group is never wrongly
skipped, and an over-estimate costs unnecessary work rather than correctness.

A digest is a **selector, not a substitute**: `lo`/`hi` are box corners, not
real keys, and must never enter an attention computation. See
DIGEST_GATED_DECODE_PLAN.md for the decode-time gate built on this.

Two shapes are provided. They differ in cost, not in contract:

============  =======================  ===========================
              storage per group/head   tightness
============  =======================  ===========================
AABB          ``2 * D``                tighter (per-channel bounds)
sphere        ``D + 1``                looser (one radius for all
                                       directions), half the memory
============  =======================  ===========================

**Keys must be POST-RoPE.** The Q tier stores keys pre-RoPE and rotates them
at read time against a frozen ``position_range`` (``modules/quant/store.py``),
but a bound built on pre-RoPE keys says nothing about post-RoPE dot products —
RoPE rotates each key by its own absolute position, so the two live in
different frames. Build digests from the keys as attention actually sees them.
That is available at the same point the Q tier quantizes: ``_evict_two_tier_impl``
gathers ``k_post`` from the fp body and only then calls ``unrotate_key_window``.
Because a stored key's absolute position is frozen, its post-RoPE vector is
fixed for the rest of the decode, so a bound built once stays valid against
every future query.

**The group axis is the caller's choice.** Nothing here assumes a group is one
eviction window. Grouping ``P`` consecutive windows under one digest divides
the resident cost by ``P`` at the price of a looser bound and a coarser
selection unit — which matters at ``window_size = 8``, where a per-window fp16
AABB is roughly half the size of the int2 record it summarizes.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _check_keys(keys: Tensor) -> None:
    if keys.dim() != 5:
        raise ValueError(
            f"keys must be [B, G, H_kv, S, D] (post-RoPE), got {tuple(keys.shape)}"
        )


def build_aabb(keys: Tensor) -> Tuple[Tensor, Tensor]:
    """Axis-aligned bounding box over each group's keys.

    Parameters
    ----------
    keys : ``[B, G, H_kv, S, D]`` post-RoPE keys, ``S`` tokens per group.

    Returns
    -------
    lo, hi : ``[B, G, H_kv, D]`` per-channel min and max.
    """
    _check_keys(keys)
    return keys.amin(dim=3), keys.amax(dim=3)


def build_sphere(keys: Tensor) -> Tuple[Tensor, Tensor]:
    """Enclosing sphere over each group's keys: centroid + covering radius.

    Not the *minimal* enclosing sphere (which needs an iterative solve); the
    centroid with the max distance to it is a valid enclosing sphere in one
    pass, which is what the bound requires. It is slightly looser than minimal,
    and that costs unnecessary reads, never correctness.

    Parameters
    ----------
    keys : ``[B, G, H_kv, S, D]`` post-RoPE keys.

    Returns
    -------
    center : ``[B, G, H_kv, D]``
    radius : ``[B, G, H_kv]``
    """
    _check_keys(keys)
    center = keys.mean(dim=3)
    radius = (keys - center.unsqueeze(3)).norm(dim=-1).amax(dim=3)
    return center, radius


def _grouped_query(q: Tensor, h_kv: int) -> Tuple[Tensor, int, int, int, int]:
    """``[B, H_q, T, D]`` -> ``[B, H_kv, rep, T, D]`` in float32.

    Matches the GQA convention the score kernel already uses
    (``score_kernel._as_grouped``): query head ``h`` belongs to KV head
    ``h // rep``, expressed as a reshape so no key copy is ever materialized.
    Float32 because these are ranking scores accumulated over ``D`` terms and
    the digests are stored in fp16.
    """
    if q.dim() != 4:
        raise ValueError(f"q must be [B, H_q, T, D], got {tuple(q.shape)}")
    B, H_q, T, D = q.shape
    if H_q % h_kv != 0:
        raise ValueError(f"H_q={H_q} not divisible by H_kv={h_kv}")
    rep = H_q // h_kv
    return q.reshape(B, h_kv, rep, T, D).float(), B, H_q, T, rep


def estimate_aabb(q: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """Upper bound on ``max_k q . k`` for every group, from an AABB digest.

    Exact over the box: per channel the extreme is at ``hi`` when ``q_d > 0``
    and at ``lo`` when ``q_d < 0``, so

        max_d-sum = relu(q) . hi  +  (q - relu(q)) . lo

    which is two matmuls rather than a ``[..., D, G]`` elementwise max.

    Parameters
    ----------
    q : ``[B, H_q, T, D]`` post-RoPE queries.
    lo, hi : ``[B, G, H_kv, D]`` from :func:`build_aabb`.

    Returns
    -------
    ``[B, H_q, T, G]`` float32 upper bounds.
    """
    # float32 here: this is the reference form, used by tests and by any caller
    # that is not in the decode hot path, so tightness beats bytes.
    lo_t, hi_t = prepare_aabb(lo.float(), hi.float())    # [B, H_kv, 1, D, G]
    return estimate_aabb_prepared(q.float(), lo_t, hi_t)


def prepare_aabb(lo: Tensor, hi: Tensor) -> Tuple[Tensor, Tensor]:
    """Pre-transpose an AABB into the layout :func:`estimate_aabb_prepared` wants.

    ``[B, G, H_kv, D] -> [B, H_kv, 1, D, G]``, **dtype preserved**. Split out because the
    digests are **write-once and the active set only moves at eviction**, so this
    transpose-and-cast is constant between evictions — doing it inside the
    per-step estimate re-materializes a ``[B, H_kv, D, G]`` float32 tensor every
    layer of every decode step, which at ``G=439, H_kv=8, D=128`` is ~1.8 MB per
    layer per step of pure repeat work. Hoist it into whatever memoizes the Q
    tier and pass the result to :func:`estimate_aabb_prepared`.

    **Does not widen to float32.** The prepared digest is re-read in full by every
    layer on every decode step, so its dtype is the gate's dominant running cost,
    not a storage detail. At ``G=439, H_kv=8, D=128, B=8`` the pair is ~14.4 MB
    per layer in fp16 and ~28.8 MB in fp32 — against the ~28 MB of int2 codes and
    scale/zero grid that gating to 25% saves reading. Widening therefore turns a
    ~13 MB net saving into a net loss, which is exactly what it did when measured
    (CUDA 91.94 -> 98.95 ms/step at B=8, `aten::copy_` becoming the #2 kernel).
    Keep the digest in the dtype it is stored in.
    """
    return (lo.permute(0, 2, 3, 1).unsqueeze(2).contiguous(),
            hi.permute(0, 2, 3, 1).unsqueeze(2).contiguous())


def estimate_aabb_prepared(q: Tensor, lo_t: Tensor, hi_t: Tensor) -> Tensor:
    """:func:`estimate_aabb` against an already-transposed digest.

    Same arithmetic as :func:`estimate_aabb`; only the layout work is lifted out.
    ``lo_t``/``hi_t`` come from :func:`prepare_aabb`.

    The matmul runs in the **digest's** dtype rather than promoting to float32,
    because the digest is the large operand and promoting it doubles the bytes
    read (see :func:`prepare_aabb`). fp16 matmul accumulates in fp32 on tensor
    cores, and these are ranking scores feeding a top-k — a tie broken differently
    costs one window swap at the margin, never correctness, because the bound
    stays one-sided either way.
    """
    if q.dim() != 4:
        raise ValueError(f"q must be [B, H_q, T, D], got {tuple(q.shape)}")
    B, H_q, T, D = q.shape
    h_kv = lo_t.shape[1]
    if H_q % h_kv != 0:
        raise ValueError(f"H_q={H_q} not divisible by H_kv={h_kv}")
    rep = H_q // h_kv
    q5 = q.reshape(B, h_kv, rep, T, D).to(lo_t.dtype)
    qp = q5.clamp_min(0)
    qn = q5 - qp
    est = qp @ hi_t + qn @ lo_t
    return est.reshape(B, H_q, T, -1).float()


def estimate_sphere(q: Tensor, center: Tensor, radius: Tensor) -> Tensor:
    """Upper bound on ``max_k q . k`` for every group, from a sphere digest.

    Exact over the ball: ``max_{|k-c| <= r} q . k == q . c + r * |q|``.

    Parameters
    ----------
    q : ``[B, H_q, T, D]`` post-RoPE queries.
    center : ``[B, G, H_kv, D]``, radius : ``[B, G, H_kv]`` from
        :func:`build_sphere`.

    Returns
    -------
    ``[B, H_q, T, G]`` float32 upper bounds.
    """
    q5, B, H_q, T, rep = _grouped_query(q, center.shape[2])
    c_t = center.permute(0, 2, 3, 1).unsqueeze(2).float()    # [B, H_kv, 1, D, G]
    r_t = radius.permute(0, 2, 1).unsqueeze(2).unsqueeze(3).float()  # [B,H_kv,1,1,G]
    est = q5 @ c_t + q5.norm(dim=-1, keepdim=True) * r_t
    return est.reshape(B, H_q, T, -1)
