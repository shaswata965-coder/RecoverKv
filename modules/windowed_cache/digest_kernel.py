"""Fused digest scoring for the decode gate (DIGEST_GATED_DECODE_PLAN.md §12.3).

One Triton kernel replacing the ~20 host launches that the AABB bound plus its
max-over-query-heads reduction used to cost, **per layer, per decode step**.

**Why this exists.** The gate's kernel-side win is real — handing
``_two_tier_decode_kernel`` a ``sel`` index vector cut it from 22.4 to 6.8 ms/step
at B=1 — but the host-side selection that produces ``sel`` cost ~51 kernel
launches per layer, ~1620 per step across 32 layers. Against an ungated baseline
running at 94.2% occupancy with a permanently full queue, that per-layer
dependency stall drove exposed host time from 5.65 to 48.7 ms and turned a 15 ms
kernel saving into a 46 ms regression. The arithmetic was never the problem; the
launches were.

**Why the selection is not fused into the decode kernel itself.** Two reasons,
both structural:

1. **Top-k has no Triton primitive.** Picking the k largest of ``n_phys`` needs a
   selection algorithm; hand-rolling one per program per layer costs more than it
   saves. ``torch.topk`` over the ``[B, n_phys]`` this kernel writes is a single
   launch and is hard to beat.
2. **Tile granularity.** The decode kernel's Q loop tiles ``BLOCK_NW`` windows at
   a time, so skipping requires *every* window in a tile to be unselected. At 75%
   skipped and a scattered selection that is ``0.75 ** BLOCK_NW`` of tiles — ~10%
   at ``BLOCK_NW=8`` — so in-kernel skipping would recover a tenth of the
   available win. Ordering the windows by relevance first would fix it, and
   ordering is exactly what cannot be done in-kernel.

So the split is: **score here (one launch), select with ``topk`` (one launch),
read in place in the decode kernel (no copy).**

Parallelism is over ``(row, window block)`` and each program loops the ``H_kv``
heads internally, so the max-over-query-heads reduction stays inside one program.
Reducing over the head axis in the grid instead would need ``atomic_max`` into a
shared ``[B, G]`` buffer — correct, but an atomic per block per head, and
needless when ``H_kv`` is 8.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

try:  # pragma: no cover - import-time capability probe
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _digest_score_kernel(
        Q,                      # [B, H_q, D]      post-RoPE decode query
        LO, HI,                 # [B, H_kv, 1, D, G] prepared AABB corners
        RANK,                   # [B, G]           out: max over query heads
        H_q, H_kv, G, rep,
        sqb, sqh, sqd,
        HEAD_DIM: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """``rank[b, g] = max_h ( relu(q_h) . hi_g + (q_h - relu(q_h)) . lo_g )``.

        That expression is the exact maximum of ``q . k`` over the box: per
        channel the extreme is at ``hi`` when ``q_d > 0`` and at ``lo`` when
        ``q_d < 0``. It is an **upper bound** on the real ``max_k q . k``, which
        is the direction the gate depends on — over-estimating costs a window
        that did not need reading, under-estimating silently drops one the model
        wanted.

        **Max over heads, not mean** (plan §3.2): a window one head wants badly
        is worth reading even when the others are indifferent, and that is the
        failure the gate exists to catch.
        """
        b = tl.program_id(0)
        g0 = tl.program_id(1) * BLOCK_G
        offs_g = g0 + tl.arange(0, BLOCK_G)
        gmask = offs_g < G
        offs_d = tl.arange(0, BLOCK_D)
        dmask = offs_d < HEAD_DIM

        best = tl.full((BLOCK_G,), float("-inf"), tl.float32)

        for kv in range(0, H_kv):
            # Digest corners for this KV head: [BLOCK_D, BLOCK_G].
            base = b * (H_kv * HEAD_DIM * G) + kv * (HEAD_DIM * G)
            lo = tl.load(LO + base + offs_d[:, None] * G + offs_g[None, :],
                         mask=dmask[:, None] & gmask[None, :], other=0.0).to(tl.float32)
            hi = tl.load(HI + base + offs_d[:, None] * G + offs_g[None, :],
                         mask=dmask[:, None] & gmask[None, :], other=0.0).to(tl.float32)
            for r in range(0, rep):
                h = kv * rep + r
                q = tl.load(Q + b * sqb + h * sqh + offs_d * sqd,
                            mask=dmask, other=0.0).to(tl.float32)
                qp = tl.maximum(q, 0.0)
                qn = q - qp
                est = tl.sum(qp[:, None] * hi + qn[:, None] * lo, axis=0)
                best = tl.maximum(best, est)

        tl.store(RANK + b * G + offs_g, best, mask=gmask)


def _pow2_at_least(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def digest_rank(q: Tensor, lo_t: Tensor, hi_t: Tensor) -> Tensor:
    """``[B, G]`` float32 per-window bound, reduced over query heads with max.

    Parameters
    ----------
    q : ``[B, H_q, D]`` post-RoPE decode query.
    lo_t, hi_t : ``[B, H_kv, 1, D, G]`` from
        :func:`modules.quant.digest.prepare_aabb`.

    Triton-or-raise, like the decode kernel it feeds: a PyTorch fallback here
    would silently reintroduce the ~20 launches per layer this exists to remove,
    and a gate that is quietly slow is worse than one that fails loudly.
    """
    if not (_HAS_TRITON and q.is_cuda):
        reason = "triton not installed" if not _HAS_TRITON else "not on CUDA"
        raise RuntimeError(
            f"digest_rank requires the Triton kernel on CUDA ({reason}). The "
            "PyTorch form is modules.quant.digest.estimate_aabb_prepared, which "
            "is the reference used by tests; it is not wired into decode because "
            "its launch count is the entire reason this kernel exists."
        )
    B, H_q, D = q.shape
    H_kv, G = lo_t.shape[1], lo_t.shape[-1]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} not divisible by H_kv={H_kv}")
    if not (lo_t.is_contiguous() and hi_t.is_contiguous()):
        raise RuntimeError(
            "digest_rank needs contiguous prepared digests; strides are derived "
            "from shape inside the kernel (see prepare_aabb, which returns "
            "contiguous tensors)."
        )
    rep = H_q // H_kv
    rank = torch.empty((B, G), device=q.device, dtype=torch.float32)
    BLOCK_G = 64 if G >= 64 else _pow2_at_least(max(G, 1))
    BLOCK_D = _pow2_at_least(D)
    grid = (B, -(-G // BLOCK_G))
    _digest_score_kernel[grid](
        q, lo_t, hi_t, rank,
        H_q, H_kv, G, rep,
        q.stride(0), q.stride(1), q.stride(2),
        HEAD_DIM=D, BLOCK_G=BLOCK_G, BLOCK_D=BLOCK_D,
    )
    return rank


def digest_select(q: Tensor, lo_t: Tensor, hi_t: Tensor, k: int) -> Tensor:
    """``[B, k]`` int32 window indices — the gate's whole host cost, 3 launches.

    Unsorted: the decode kernel resolves every emitted column through this
    vector, so order is free, and a CUDA ``sort`` is a multi-kernel radix pass
    that measurably was not.
    """
    rank = digest_rank(q, lo_t, hi_t)
    sel = rank.topk(k, dim=-1, sorted=False).indices
    return sel.to(torch.int32).contiguous()
