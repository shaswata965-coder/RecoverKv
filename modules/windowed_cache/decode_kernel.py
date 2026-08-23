"""Fused two-tier decode attention + eviction score (design.md §11 Phase 2).

What this replaces
------------------
Phase 1 decode reads the Q tier by dequantizing every active int2 window to
fp16, RoPE-ing it, and concatenating it with the fp tier into one effective K/V
that the model then attends over (``materialize_effective_kv``); a separate score
pass (the flash hook) recomputes ``softmax(q·kᵀ)`` to get eviction scores. That is
four per-step costs, every layer (design §8): the fp16 write-back of the Q tier,
the concat allocation, a full-budget fp16 attention read, and a **duplicate**
``q·kᵀ`` for the score.

This kernel does all of it in one launch, reading the Q tier **as int2** and
dequantizing + RoPE-ing **in registers** — no fp16 tier is ever written to HBM,
no concat, and the eviction score falls out of the same ``q·kᵀ`` the attention
already computes (for a single decode query the per-key softmax weight *is* the
H2O score). So the fp16 write-back and the duplicate score pass both disappear,
and because nothing is held in fp16 the read-memo can stay off at ``B>1`` with no
penalty — keeping max-B *and* the speed.

Layout at the flash boundary
----------------------------
The kernel is invoked from the ``flash_attn_func`` monkeypatch
(:mod:`flash_decode`), so it consumes the same layout flash does — ``[B, S, H, D]``
(seqlen-major, per design of ``_flash_attention_forward``) — and returns
``attn_output`` in that layout, plus ``token_scores`` ``[B, H_q, S_eff]`` in
``[sink ‖ fp body ‖ Q]`` order (the order the window scorer's ``score_meta``
undoes, exactly as the materialize path produced).

Backend contract (mirrors :mod:`score_kernel`)
----------------------------------------------
* :func:`two_tier_decode_reference` — pure-PyTorch oracle over an *already
  effective* K/V. Defines correctness (attention output **and** per-key score),
  runs on CPU, and is the Triton kernel's test oracle.
* :func:`_two_tier_decode_kernel` + :func:`_decode_triton` — the fused GPU kernel.
  **GPU-only and ships unvalidated by construction** (this repo's dev box is
  CPU-only), exactly like ``score_kernel._token_scores_triton``.
* :func:`fused_two_tier_decode` — the dispatcher: **Triton-or-raise**. There is no
  PyTorch decode fallback in production; the reference exists for tests, and the
  flash backend is gated to refuse to run at all without the kernel
  (:func:`assert_decode_kernel_available`).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch import Tensor

try:  # pragma: no cover - import guard, exercised only where triton is present
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # ImportError, or a broken triton build
    _HAS_TRITON = False


def fused_decode_enabled() -> bool:
    """Whether the fused two-tier decode path is active (default ON).

    Off only when ``STICKYKV_FUSED_DECODE`` is explicitly falsey. When off, the
    flash backend falls back to the Phase-1 materialize path. It is ON by default
    because the fused kernel is now the intended decode path for the flash
    backend; the materialize path is kept for the eager backend and for CPU tests
    (which never install the flash monkeypatch, so they never reach this kernel).
    """
    v = os.environ.get("STICKYKV_FUSED_DECODE", "1").strip().lower()
    return v in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Availability / gating
# ---------------------------------------------------------------------------


def decode_kernel_available(cuda: bool) -> bool:
    """True iff the fused decode Triton kernel can actually launch."""
    return bool(_HAS_TRITON and cuda)


def describe_decode_backend(cuda: bool) -> str:
    """Human string for the decode attention backend, given CUDA availability."""
    if not fused_decode_enabled():
        return "materialize (fused decode disabled via STICKYKV_FUSED_DECODE=0)"
    if decode_kernel_available(cuda):
        return "TRITON fused two-tier decode kernel (required)"
    reason = "triton not installed" if not _HAS_TRITON else "not running on CUDA"
    return f"WILL ERROR: fused decode needs the Triton kernel on CUDA ({reason})"


def assert_decode_kernel_available(cuda: bool) -> None:
    """Raise unless the fused decode kernel can launch (flash-backend gate).

    Called once at flash-hook install: choosing the flash backend commits to the
    Triton path for the decode kernel just as it does for the prefill score
    kernel, so a box that cannot launch it must fail loudly at setup rather than
    silently on the first quantized decode step (or, worse, run a degraded path).
    """
    if not fused_decode_enabled():
        return
    if not decode_kernel_available(cuda):
        reason = "triton not installed" if not _HAS_TRITON else "not running on CUDA"
        raise RuntimeError(
            "The flash backend requires the fused two-tier decode Triton kernel "
            f"on CUDA, and it cannot launch ({reason}). This is the decode analog "
            "of the prefill score kernel's Triton-or-error contract: there is no "
            "PyTorch decode fallback. Install triton and run on GPU, use the eager "
            "backend, or set STICKYKV_FUSED_DECODE=0 to fall back to the Phase-1 "
            "materialize path."
        )


# ---------------------------------------------------------------------------
# PyTorch reference / oracle — over an ALREADY-EFFECTIVE K/V
# ---------------------------------------------------------------------------


def two_tier_decode_reference(
    q: Tensor,
    k_eff: Tensor,
    v_eff: Tensor,
    scaling: float,
) -> Tuple[Tensor, Tensor]:
    """Single-query attention output **and** per-key eviction score.

    This is the exact math the fused kernel implements, in PyTorch, so it is the
    kernel's CPU oracle and the definition of "correct". For a single decode
    query the per-key softmax weight is the H2O score, so both outputs come from
    one softmax — which is the whole reason the kernel can emit the score for
    free.

    Parameters
    ----------
    q : ``[B, H_q, D]`` or ``[B, H_q, 1, D]`` — the decode query (post-RoPE).
    k_eff, v_eff : ``[B, H_kv, S, D]`` — the effective K/V (fp tier ‖ dequantized
        Q tier), in whatever order the caller wants the scores emitted in.
    scaling : softmax logit scale.

    Returns
    -------
    out : ``[B, H_q, D]`` attention output.
    token_scores : ``[B, H_q, S]`` per-key received attention (== the softmax
        weights, since there is exactly one query row).
    """
    if q.dim() == 3:
        q = q.unsqueeze(2)                                  # [B, H_q, 1, D]
    B, H_q, _, D = q.shape
    H_kv, S = k_eff.shape[1], k_eff.shape[2]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} not divisible by H_kv={H_kv}")
    rep = H_q // H_kv

    q5 = q.reshape(B, H_kv, rep, 1, D)                      # [B,H_kv,rep,1,D]
    k = k_eff.unsqueeze(2)                                  # [B,H_kv,1,S,D]
    v = v_eff.unsqueeze(2)                                  # [B,H_kv,1,S,D]

    logits = torch.matmul(q5, k.transpose(-2, -1)).float() * scaling  # [B,H_kv,rep,1,S]
    p = torch.softmax(logits, dim=-1)                       # over S, fp32
    out = torch.matmul(p.to(v.dtype), v)                   # [B,H_kv,rep,1,D]

    out = out.reshape(B, H_q, D)
    token_scores = p.reshape(B, H_q, S)
    return out.to(v_eff.dtype), token_scores.to(v_eff.dtype)


# ---------------------------------------------------------------------------
# Triton fused kernel — GPU-only, ships unvalidated (validate on a GPU vs the
# reference above, exactly as score_kernel's Triton path is validated).
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _two_tier_decode_kernel(
        Q,                       # [B, H_q, D]           fp query (post-RoPE)
        KFP, VFP,                # [B, H_kv, Sfp, D]     fp tier (post-RoPE keys)
        KDQ, VDQ,                # [B, H_kv, Sq, D]      Q tier, dequantized+RoPE'd
        OUT,                     # [B, H_q, D]
        SCORES,                  # [B, H_q, Sfp + Sq]    per-key scores, [fp ‖ Q]
        scale,
        stride_qb, stride_qh, stride_qd,
        stride_kfb, stride_kfh, stride_kfs, stride_kfd,
        stride_vfb, stride_vfh, stride_vfs, stride_vfd,
        stride_kqb, stride_kqh, stride_kqs, stride_kqd,
        stride_vqb, stride_vqh, stride_vqs, stride_vqd,
        stride_ob, stride_oh, stride_od,
        stride_sb, stride_sh, stride_ss,
        Sfp, Sq, H_q, num_groups,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program == one (batch, query-head): a decode GEMV over both tiers.

        Reads the fp tier and the *dequantized* Q tier (the dequant+RoPE is fused
        into the read that produces KDQ/VDQ — see :func:`_decode_triton`), runs a
        single-query online softmax over ``[fp ‖ Q]``, writes the attention output
        and every key's normalized weight (the eviction score). Two light passes
        share one logit buffer (``SCORES``): pass 1 stores logits + tracks
        (m, l) + accumulates the output online; a final elementwise turns the
        stored logits into ``exp(logit − LSE)``.
        """
        pid = tl.program_id(0)                  # b * H_q + h
        b = pid // H_q
        h = pid % H_q
        kv = h // num_groups                    # GQA: which KV head this reads

        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_d * stride_qd)  # [D]

        m = -float("inf")
        l = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        S = Sfp + Sq
        # ---- Pass 1: online softmax over both tiers, storing logits ----------
        for start in range(0, S, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            in_fp = offs_n < Sfp
            in_q = (offs_n >= Sfp) & (offs_n < S)
            # fp key rows for this block
            kfp_ptr = (KFP + b * stride_kfb + kv * stride_kfh
                       + offs_n[:, None] * stride_kfs + offs_d[None, :] * stride_kfd)
            kfp = tl.load(kfp_ptr, mask=in_fp[:, None], other=0.0)
            # Q key rows for this block (index into the dequantized Q tier)
            offs_q = offs_n - Sfp
            kq_ptr = (KDQ + b * stride_kqb + kv * stride_kqh
                      + offs_q[:, None] * stride_kqs + offs_d[None, :] * stride_kqd)
            kq = tl.load(kq_ptr, mask=in_q[:, None], other=0.0)
            k_tile = tl.where(in_q[:, None], kq, kfp)                 # [BLOCK_N, D]

            logit = tl.sum(k_tile * q[None, :], axis=1) * scale       # [BLOCK_N]
            valid = offs_n < S
            logit = tl.where(valid, logit, -float("inf"))
            tl.store(SCORES + b * stride_sb + h * stride_sh + offs_n * stride_ss,
                     logit, mask=valid)

            m_new = tl.maximum(m, tl.max(logit, axis=0))
            corr = tl.exp(m - m_new)
            p = tl.exp(logit - m_new)                                 # [BLOCK_N]
            p = tl.where(valid, p, 0.0)

            vfp_ptr = (VFP + b * stride_vfb + kv * stride_vfh
                       + offs_n[:, None] * stride_vfs + offs_d[None, :] * stride_vfd)
            vfp = tl.load(vfp_ptr, mask=in_fp[:, None], other=0.0)
            vq_ptr = (VDQ + b * stride_vqb + kv * stride_vqh
                      + offs_q[:, None] * stride_vqs + offs_d[None, :] * stride_vqd)
            vq = tl.load(vq_ptr, mask=in_q[:, None], other=0.0)
            v_tile = tl.where(in_q[:, None], vq, vfp)                 # [BLOCK_N, D]

            acc = acc * corr + tl.sum(p[:, None] * v_tile, axis=0)    # [D]
            l = l * corr + tl.sum(p, axis=0)
            m = m_new

        lse = m + tl.log(l)
        out = acc / l
        tl.store(OUT + b * stride_ob + h * stride_oh + offs_d * stride_od, out)

        # ---- Pass 2: normalize the stored logits into scores -----------------
        for start in range(0, S, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            valid = offs_n < S
            sp = SCORES + b * stride_sb + h * stride_sh + offs_n * stride_ss
            logit = tl.load(sp, mask=valid, other=-float("inf"))
            tl.store(sp, tl.exp(logit - lse), mask=valid)


def _decode_triton(
    q: Tensor,
    k_fp: Tensor,
    v_fp: Tensor,
    k_q: Tensor,
    v_q: Tensor,
    scaling: float,
    *,
    block_n: int = 64,
) -> Tuple[Tensor, Tensor]:
    """Launch the fused decode kernel. Returns ``(out [B,H_q,D], scores [B,H_q,S])``.

    ``k_q``/``v_q`` are the Q tier **already dequantized and RoPE'd** into fp
    ``[B, H_kv, Sq, D]``. NOTE: fusing the int2 unpack + RoPE *into the kernel's
    key load* is the final register-level step (the point of Phase 2); this first
    cut keeps that dequant in the compiled read path (:func:`effective`) and hands
    the kernel dense fp Q keys, so the attention + score fusion is exercised and
    validated first, then the dequant is pulled inside. Both are Triton-or-error
    — no PyTorch attention runs in production either way.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton not available; fused decode requires CUDA+triton.")
    if not q.is_cuda:
        raise RuntimeError("Fused decode kernel requires CUDA tensors.")
    B, H_q, D = q.shape
    H_kv, Sfp = k_fp.shape[1], k_fp.shape[2]
    Sq = k_q.shape[2] if k_q is not None else 0
    num_groups = H_q // H_kv
    S = Sfp + Sq

    out = torch.empty((B, H_q, D), device=q.device, dtype=q.dtype)
    scores = torch.empty((B, H_q, S), device=q.device, dtype=torch.float32)

    # Empty Q tier: give the kernel a zero-width Q store it will never index.
    if k_q is None or Sq == 0:
        k_q = q.new_zeros((B, H_kv, 1, D))
        v_q = q.new_zeros((B, H_kv, 1, D))

    grid = (B * H_q,)
    _two_tier_decode_kernel[grid](
        q, k_fp, v_fp, k_q, v_q, out, scores,
        scaling,
        q.stride(0), q.stride(1), q.stride(2),
        k_fp.stride(0), k_fp.stride(1), k_fp.stride(2), k_fp.stride(3),
        v_fp.stride(0), v_fp.stride(1), v_fp.stride(2), v_fp.stride(3),
        k_q.stride(0), k_q.stride(1), k_q.stride(2), k_q.stride(3),
        v_q.stride(0), v_q.stride(1), v_q.stride(2), v_q.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        scores.stride(0), scores.stride(1), scores.stride(2),
        Sfp, Sq, H_q, num_groups,
        HEAD_DIM=D,
        BLOCK_N=block_n,
    )
    return out, scores


# ---------------------------------------------------------------------------
# Public dispatcher — Triton-or-raise
# ---------------------------------------------------------------------------


def fused_two_tier_decode(
    q: Tensor,
    k_fp: Tensor,
    v_fp: Tensor,
    k_q: Optional[Tensor],
    v_q: Optional[Tensor],
    scaling: float,
) -> Tuple[Tensor, Tensor]:
    """Fused decode attention + score. **Triton-or-raise** — no PyTorch fallback.

    Parameters
    ----------
    q : ``[B, H_q, D]`` post-RoPE decode query.
    k_fp, v_fp : ``[B, H_kv, S_fp, D]`` fp tier (post-RoPE keys), in
        ``[sink ‖ body]`` order.
    k_q, v_q : ``[B, H_kv, S_q, D]`` dequantized+RoPE'd Q tier, or ``None`` for an
        empty Q tier.
    scaling : logit scale.

    Returns
    -------
    out : ``[B, H_q, D]``
    token_scores : ``[B, H_q, S_fp + S_q]`` in ``[sink ‖ body ‖ Q]`` order.
    """
    if not (_HAS_TRITON and q.is_cuda):
        reason = "triton not installed" if not _HAS_TRITON else "not on CUDA"
        raise RuntimeError(
            "fused_two_tier_decode requires the Triton kernel on CUDA "
            f"({reason}); there is no PyTorch decode fallback in production "
            "(the reference is for tests only). See assert_decode_kernel_available."
        )
    return _decode_triton(q, k_fp, v_fp, k_q, v_q, scaling)
