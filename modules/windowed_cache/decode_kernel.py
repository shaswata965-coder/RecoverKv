"""Fused two-tier decode attention + eviction score (design.md §11 Phase 2).

What this replaces
------------------
Phase 1 decode reads the Q tier by dequantizing every active int2 window to
fp16, RoPE-ing it, and concatenating it with the fp tier into one effective K/V
that the model attends over (``materialize_effective_kv``); a separate score pass
recomputes ``softmax(q·kᵀ)`` for the eviction scores. Four per-step costs, every
layer (design §8): the fp16 write-back of the Q tier, the concat, a full-budget
fp16 attention read, and a **duplicate** ``q·kᵀ`` for the score.

This kernel does all of it in one launch, reading the Q tier **as int2** and
dequantizing + RoPE-ing **in registers** — no fp16 tier is written to HBM, no
concat, and the eviction score falls out of the same ``q·kᵀ`` (for a single
decode query the per-key softmax weight IS the H2O score). Nothing is held in
fp16, so the read-memo can stay off at ``B>1`` with no penalty — max-B AND speed.

Fast-decode mechanics folded in (flash-decoding)
------------------------------------------------
* **GQA load-once.** One program owns one ``(batch, KV head)`` and handles all
  ``rep = num_groups`` query heads sharing it, so each key/value tile — and each
  int2 dequant — is read/done **once per KV head**, not once per query head
  (``num_groups``× less traffic; 4× on Llama-3.1-8B). Hence grid ``B·H_kv``.
* **Online softmax** with a running max — the ``[S]`` score row is never held;
  a second pass turns the stored logits into normalized per-key scores.
* **Tiled ``tl.dot``** over the fp tier (``BLOCK_N`` keys) and the Q tier (one
  window, padded to ``BLOCK_WS``); query heads padded to ``BLOCK_R``.
* **RoPE without a rotate gather.** Load the two head-dim halves separately and
  apply ``k_lo·c − k_hi·s`` / ``k_hi·c + k_lo·s`` from the frozen per-position
  ``cos``/``sin`` halves — pure register math (HF stores ``cos=cat(h,h)``, so the
  first half is the distinct part).
* (Not yet: split-K over the sequence for occupancy when ``B·H_kv`` is small — a
  tuning follow-up: partial ``(out, lse)`` per KV-split + a combine.)

Backend contract (mirrors :mod:`score_kernel`)
----------------------------------------------
* :func:`two_tier_decode_reference` — pure-PyTorch oracle over an *already
  effective* K/V. Defines correctness (output **and** score); runs on CPU; is the
  Triton kernel's test oracle (build ``k_eff`` with ``materialize_effective_kv``
  on the same store, then compare).
* ``_two_tier_decode_kernel`` + :func:`_decode_triton` — the fused GPU kernel.
  **GPU-only, ships unvalidated by construction** (CPU-only dev box), like
  ``score_kernel._token_scores_triton``.
* :func:`fused_two_tier_decode` — dispatcher: **Triton-or-raise**. No PyTorch
  decode fallback in production; the flash backend is gated to refuse to run
  without the kernel (:func:`assert_decode_kernel_available`).
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

    Off only when ``STICKYKV_FUSED_DECODE`` is explicitly falsey. ON by default
    because the fused kernel is the intended decode path for the flash backend;
    the materialize path is kept for the eager backend and CPU tests (which never
    install the flash monkeypatch, so they never reach this kernel).
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

    Choosing the flash backend commits to the Triton path for the decode kernel
    just as for the prefill score kernel, so a box that cannot launch it must fail
    at setup rather than silently on the first quantized decode step.
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
# RoPE halves for the Q positions (built on the host; passed to the kernel)
# ---------------------------------------------------------------------------


def rope_cos_sin_halves(
    rope_module: torch.nn.Module, pos_flat: Tensor
) -> Tuple[Tensor, Tensor]:
    """``cos``/``sin`` **first halves** ``[B, T, D//2]`` for the Q positions.

    HF builds ``cos``/``sin`` as ``cat(h, h)`` along head_dim, so the first half
    carries the distinct per-frequency values; the kernel applies RoPE from those
    halves directly (no rotate-half gather). Position-only, so this is cheap and
    holds no key data — the point of the fused path.
    """
    ref = torch.empty(1, 1, 1, device=pos_flat.device)
    pos = pos_flat.to(torch.long)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    cos, sin = rope_module(ref, pos)                       # [B, T, D]
    half = cos.shape[-1] // 2
    return cos[..., :half].contiguous(), sin[..., :half].contiguous()


# ---------------------------------------------------------------------------
# PyTorch reference / oracle — over an ALREADY-EFFECTIVE K/V
# ---------------------------------------------------------------------------


def two_tier_decode_reference(
    q: Tensor,
    k_eff: Tensor,
    v_eff: Tensor,
    scaling: float,
) -> Tuple[Tensor, Tensor]:
    """Single-query attention output **and** per-key eviction score (the oracle).

    For a single decode query the per-key softmax weight is the H2O score, so both
    outputs come from one softmax.

    q : ``[B, H_q, D]`` or ``[B, H_q, 1, D]`` post-RoPE query.
    k_eff, v_eff : ``[B, H_kv, S, D]`` effective K/V (``[sink ‖ body ‖ Q]``).
    Returns ``(out [B,H_q,D], token_scores [B,H_q,S])``.
    """
    if q.dim() == 3:
        q = q.unsqueeze(2)
    B, H_q, _, D = q.shape
    H_kv, S = k_eff.shape[1], k_eff.shape[2]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} not divisible by H_kv={H_kv}")
    rep = H_q // H_kv

    q5 = q.reshape(B, H_kv, rep, 1, D)
    k = k_eff.unsqueeze(2)
    v = v_eff.unsqueeze(2)
    logits = torch.matmul(q5, k.transpose(-2, -1)).float() * scaling
    p = torch.softmax(logits, dim=-1)
    out = torch.matmul(p.to(v.dtype), v)

    out = out.reshape(B, H_q, D)
    token_scores = p.reshape(B, H_q, S)
    return out.to(v_eff.dtype), token_scores.to(v_eff.dtype)


# ---------------------------------------------------------------------------
# Triton fused kernel — GQA-grouped, int2-in-register, RoPE-in-register.
# GPU-only, ships UNVALIDATED (validate on a GPU vs two_tier_decode_reference).
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _two_tier_decode_kernel(
        Q, KFP, VFP,
        KC, KS, KZ,                # Q keys: codes u8 [B,n,H_kv,D,ws//4]; scale/zero fp16 [B,n,H_kv,D]
        VC, VS, VZ,                # Q vals: codes u8 [B,n,H_kv,ws,D//4]; scale/zero fp16 [B,n,H_kv,ws]
        COS, SIN,                  # RoPE halves [B, n*ws, D//2]
        OUT, SCORES,
        scale,
        H_kv, n_active, Sfp, rep,
        sqb, sqh, sqd,
        kfb, kfh, kfs, kfd,
        vfb, vfh, vfs, vfd,
        kcb, kcn, kch, kcd, kcp,
        ksb, ksn, ksh, ksd,
        vcb, vcn, vch, vct, vcp,
        vsb, vsn, vsh, vst,
        cob, cot, cod,
        ob, oh, od,
        sb, sh, ss,
        HEAD_DIM: tl.constexpr, HALF: tl.constexpr, WS: tl.constexpr,
        BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_WS: tl.constexpr,
    ):
        """One program == one (batch, KV head); all ``rep`` query heads at once."""
        pid = tl.program_id(0)
        b = pid // H_kv
        kv = pid % H_kv

        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < rep
        hq = kv * rep + offs_r                              # [BLOCK_R] query heads
        offs_hl = tl.arange(0, HALF)
        offs_d = tl.arange(0, HEAD_DIM)

        qg = tl.load(Q + b * sqb + hq[:, None] * sqh + offs_d[None, :] * sqd,
                     mask=r_mask[:, None], other=0.0).to(tl.float32)   # [BLOCK_R, D]
        q_lo = tl.load(Q + b * sqb + hq[:, None] * sqh + offs_hl[None, :] * sqd,
                       mask=r_mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_R, HALF]
        q_hi = tl.load(Q + b * sqb + hq[:, None] * sqh + (offs_hl + HALF)[None, :] * sqd,
                       mask=r_mask[:, None], other=0.0).to(tl.float32)

        m = tl.full([BLOCK_R], -float("inf"), tl.float32)
        l = tl.zeros([BLOCK_R], tl.float32)
        acc = tl.zeros([BLOCK_R, HEAD_DIM], tl.float32)

        # -------- fp tier: post-RoPE fp16 keys, load once per KV head --------
        for start in range(0, Sfp, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            nmask = offs_n < Sfp
            kf = tl.load(KFP + b * kfb + kv * kfh + offs_n[:, None] * kfs + offs_d[None, :] * kfd,
                         mask=nmask[:, None], other=0.0).to(tl.float32)   # [BLOCK_N, D]
            logit = tl.dot(qg, tl.trans(kf)) * scale                      # [BLOCK_R, BLOCK_N]
            logit = tl.where(nmask[None, :], logit, -float("inf"))
            tl.store(SCORES + b * sb + hq[:, None] * sh + offs_n[None, :] * ss, logit,
                     mask=r_mask[:, None] & nmask[None, :])
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp(m - m_new)
            p = tl.where(nmask[None, :], tl.exp(logit - m_new[:, None]), 0.0)
            vf = tl.load(VFP + b * vfb + kv * vfh + offs_n[:, None] * vfs + offs_d[None, :] * vfd,
                         mask=nmask[:, None], other=0.0).to(tl.float32)
            acc = acc * corr[:, None] + tl.dot(p, vf)
            l = l * corr + tl.sum(p, axis=1)
            m = m_new

        # -------- Q tier: int2 dequant + RoPE in registers, one window/iter --
        offs_t = tl.arange(0, BLOCK_WS)
        tmask = offs_t < WS
        byte_t = (offs_t // 4)
        shift_t = (2 * (offs_t % 4)).to(tl.uint8)
        cbyte = (offs_d // 4)
        cshift = (2 * (offs_d % 4)).to(tl.uint8)
        for w in range(0, n_active):
            ks_lo = tl.load(KS + b * ksb + w * ksn + kv * ksh + offs_hl * ksd).to(tl.float32)
            kz_lo = tl.load(KZ + b * ksb + w * ksn + kv * ksh + offs_hl * ksd).to(tl.float32)
            ks_hi = tl.load(KS + b * ksb + w * ksn + kv * ksh + (offs_hl + HALF) * ksd).to(tl.float32)
            kz_hi = tl.load(KZ + b * ksb + w * ksn + kv * ksh + (offs_hl + HALF) * ksd).to(tl.float32)
            kb_lo = tl.load(KC + b * kcb + w * kcn + kv * kch
                            + offs_hl[:, None] * kcd + byte_t[None, :] * kcp,
                            mask=tmask[None, :], other=0)                 # [HALF, BLOCK_WS] u8
            kb_hi = tl.load(KC + b * kcb + w * kcn + kv * kch
                            + (offs_hl + HALF)[:, None] * kcd + byte_t[None, :] * kcp,
                            mask=tmask[None, :], other=0)
            k_lo = ((kb_lo >> shift_t[None, :]) & 3).to(tl.float32) * ks_lo[:, None] + kz_lo[:, None]
            k_hi = ((kb_hi >> shift_t[None, :]) & 3).to(tl.float32) * ks_hi[:, None] + kz_hi[:, None]
            crow = w * WS + offs_t
            c = tl.load(COS + b * cob + crow[None, :] * cot + offs_hl[:, None] * cod,
                        mask=tmask[None, :], other=0.0)                   # [HALF, BLOCK_WS]
            s = tl.load(SIN + b * cob + crow[None, :] * cot + offs_hl[:, None] * cod,
                        mask=tmask[None, :], other=0.0)
            k_rlo = k_lo * c - k_hi * s
            k_rhi = k_hi * c + k_lo * s
            logit = (tl.dot(q_lo, k_rlo) + tl.dot(q_hi, k_rhi)) * scale   # [BLOCK_R, BLOCK_WS]
            logit = tl.where(tmask[None, :], logit, -float("inf"))
            key_col = Sfp + w * WS + offs_t
            tl.store(SCORES + b * sb + hq[:, None] * sh + key_col[None, :] * ss, logit,
                     mask=r_mask[:, None] & tmask[None, :])
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp(m - m_new)
            p = tl.where(tmask[None, :], tl.exp(logit - m_new[:, None]), 0.0)
            vs = tl.load(VS + b * vsb + w * vsn + kv * vsh + offs_t * vst, mask=tmask, other=0.0).to(tl.float32)
            vz = tl.load(VZ + b * vsb + w * vsn + kv * vsh + offs_t * vst, mask=tmask, other=0.0).to(tl.float32)
            vb = tl.load(VC + b * vcb + w * vcn + kv * vch + offs_t[:, None] * vct + cbyte[None, :] * vcp,
                         mask=tmask[:, None], other=0)                    # [BLOCK_WS, D] u8
            vv = ((vb >> cshift[None, :]) & 3).to(tl.float32) * vs[:, None] + vz[:, None]
            acc = acc * corr[:, None] + tl.dot(p, vv)
            l = l * corr + tl.sum(p, axis=1)
            m = m_new

        out = acc / l[:, None]
        lse = m + tl.log(l)
        tl.store(OUT + b * ob + hq[:, None] * oh + offs_d[None, :] * od,
                 out.to(OUT.dtype.element_ty), mask=r_mask[:, None])

        # pass 2: normalize stored logits into per-key scores
        S = Sfp + n_active * WS
        for start in range(0, S, BLOCK_N):
            offs_s = start + tl.arange(0, BLOCK_N)
            smask = offs_s < S
            sp = SCORES + b * sb + hq[:, None] * sh + offs_s[None, :] * ss
            lg = tl.load(sp, mask=r_mask[:, None] & smask[None, :], other=-float("inf"))
            tl.store(sp, tl.exp(lg - lse[:, None]), mask=r_mask[:, None] & smask[None, :])


def _pow2_at_least(x: int, floor: int = 16) -> int:
    v = floor
    while v < x:
        v *= 2
    return v


def _decode_triton(
    q: Tensor,
    k_fp: Tensor,
    v_fp: Tensor,
    qtier: Optional[dict],
    scaling: float,
    *,
    block_n: int = 64,
) -> Tuple[Tensor, Tensor]:
    """Launch the fused decode kernel. Returns ``(out [B,H_q,D], scores [B,H_q,S])``.

    ``qtier`` (or None for an empty Q tier) carries the gathered active-window int2
    fields shaped ``[B, n, H_kv, ...]`` plus RoPE halves ``cos``/``sin``
    ``[B, n*ws, D//2]`` and ``window_size`` — the int2 unpack + affine dequant +
    RoPE all happen inside the kernel, so no fp16 Q tensor is built.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton not available; fused decode requires CUDA+triton.")
    if not q.is_cuda:
        raise RuntimeError("Fused decode kernel requires CUDA tensors.")

    B, H_q, D = q.shape
    H_kv, Sfp = k_fp.shape[1], k_fp.shape[2]
    rep = H_q // H_kv
    half = D // 2

    if qtier is not None:
        n_active = int(qtier["k_codes"].shape[1])
        ws = int(qtier["window_size"])
    else:
        n_active, ws = 0, 4  # WS is a constexpr; the Q loop runs 0 times

    S = Sfp + n_active * ws
    out = torch.empty((B, H_q, D), device=q.device, dtype=q.dtype)
    scores = torch.empty((B, H_q, S), device=q.device, dtype=torch.float32)

    # Dummies for the empty-Q case: valid tensors so strides exist; never indexed
    # (the Q loop and cos/sin loads run only for w < n_active == 0).
    dev = q.device
    if qtier is None:
        kc = torch.zeros((B, 1, H_kv, D, max(ws // 4, 1)), dtype=torch.uint8, device=dev)
        ksz = torch.zeros((B, 1, H_kv, D), dtype=torch.float16, device=dev)
        vc = torch.zeros((B, 1, H_kv, ws, max(D // 4, 1)), dtype=torch.uint8, device=dev)
        vsz = torch.zeros((B, 1, H_kv, ws), dtype=torch.float16, device=dev)
        cs = torch.zeros((B, 1, half), dtype=torch.float32, device=dev)
        KC, KS, KZ = kc, ksz, ksz
        VC, VS, VZ = vc, vsz, vsz
        COS, SIN = cs, cs
    else:
        KC, KS, KZ = qtier["k_codes"], qtier["k_scale"], qtier["k_zero"]
        VC, VS, VZ = qtier["v_codes"], qtier["v_scale"], qtier["v_zero"]
        COS, SIN = qtier["cos"], qtier["sin"]

    BLOCK_R = _pow2_at_least(rep)
    BLOCK_WS = _pow2_at_least(ws)
    grid = (B * H_kv,)
    _two_tier_decode_kernel[grid](
        q, k_fp, v_fp, KC, KS, KZ, VC, VS, VZ, COS, SIN, out, scores,
        scaling,
        H_kv, n_active, Sfp, rep,
        q.stride(0), q.stride(1), q.stride(2),
        k_fp.stride(0), k_fp.stride(1), k_fp.stride(2), k_fp.stride(3),
        v_fp.stride(0), v_fp.stride(1), v_fp.stride(2), v_fp.stride(3),
        KC.stride(0), KC.stride(1), KC.stride(2), KC.stride(3), KC.stride(4),
        KS.stride(0), KS.stride(1), KS.stride(2), KS.stride(3),
        VC.stride(0), VC.stride(1), VC.stride(2), VC.stride(3), VC.stride(4),
        VS.stride(0), VS.stride(1), VS.stride(2), VS.stride(3),
        COS.stride(0), COS.stride(1), COS.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        scores.stride(0), scores.stride(1), scores.stride(2),
        HEAD_DIM=D, HALF=half, WS=ws,
        BLOCK_R=BLOCK_R, BLOCK_N=block_n, BLOCK_WS=BLOCK_WS,
    )
    return out, scores


# ---------------------------------------------------------------------------
# Public dispatcher — Triton-or-raise
# ---------------------------------------------------------------------------


def fused_two_tier_decode(
    q: Tensor,
    k_fp: Tensor,
    v_fp: Tensor,
    qtier: Optional[dict],
    scaling: float,
) -> Tuple[Tensor, Tensor]:
    """Fused decode attention + score. **Triton-or-raise** — no PyTorch fallback.

    q : ``[B, H_q, D]`` post-RoPE decode query.
    k_fp, v_fp : ``[B, H_kv, S_fp, D]`` fp tier (post-RoPE keys), ``[sink ‖ body]``.
    qtier : int2 Q-tier context (see :func:`_decode_triton`) or ``None`` (empty).
    Returns ``(out [B,H_q,D], token_scores [B,H_q,S_fp + n*ws])``.
    """
    if not (_HAS_TRITON and q.is_cuda):
        reason = "triton not installed" if not _HAS_TRITON else "not on CUDA"
        raise RuntimeError(
            "fused_two_tier_decode requires the Triton kernel on CUDA "
            f"({reason}); there is no PyTorch decode fallback in production "
            "(the reference is for tests only). See assert_decode_kernel_available."
        )
    return _decode_triton(q, k_fp, v_fp, qtier, scaling)
