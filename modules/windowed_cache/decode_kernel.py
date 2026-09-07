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
# Window-score emission (DECODE_SPEED_PLAN §5.1) — tiling + CPU oracle
# ---------------------------------------------------------------------------


def window_tiling(window_size: int, target_keys: int = 64) -> Tuple[int, int]:
    """``(BLOCK_NW, BLOCK_T)`` — how many windows per tile, padded to a power of 2.

    The kernel tiles the fp **body** and the Q tier in whole windows rather than
    in keys, which is what makes "every window lives entirely inside one tile"
    true for **any** ``window_size``, not just powers of two. ``tl.arange``
    requires a power-of-2 length, so a tile of ``BLOCK_NW * ws`` keys is padded up
    to ``BLOCK_T`` and the tail lanes are masked off.

    That padding is the entire cost of supporting a non-power-of-2 ``ws``:

    =====  ========  =========  =========
    ``ws``  BLOCK_NW   BLOCK_T   utilisation
    =====  ========  =========  =========
    1            64         64      100%
    8             8         64      100%
    12            5         64       94%
    128           1        128      100%
    =====  ========  =========  =========

    Tiling by windows also makes the ``num_sink`` offset a non-issue: the body
    loop starts at key ``num_sink`` (window 0's first token) and steps in window
    units, so tile boundaries land on window boundaries by construction. The sink
    prefix is handled by its own prologue tile, which contributes to the softmax
    but emits no window score — matching ``reduce_token_scores_to_windows``,
    which strips the sink before windowing.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    block_nw = max(1, target_keys // window_size)
    return block_nw, _pow2_at_least(block_nw * window_size, floor=1)


def two_tier_window_reference(
    q: Tensor,
    k_eff: Tensor,
    v_eff: Tensor,
    scaling: float,
    num_sink: int,
    window_size: int,
    n_body_win: int,
    Sfp: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """CPU oracle for the kernel's **windowed** output, tile for tile.

    This is deliberately not a shortcut. It reproduces the kernel's actual
    algorithm — sink prologue, window-tiled body, window-tiled Q tier, the online
    softmax's running max, per-tile window sums taken against that running max,
    and the epilogue that rescales each window by ``exp(m_tile - lse)`` — so that
    the parts of §5.1 that cannot be executed on a CPU-only box (the Triton
    lowering) are the *only* parts left unverified. The arithmetic, the tiling,
    the masking and the rescale are all pinned by
    ``tests/test_window_scores.py`` against the plain
    ``softmax -> strip sink -> window-sum`` path.

    ``k_eff``/``v_eff`` are the effective ``[sink ‖ body ‖ Q]`` store, exactly
    what ``two_tier_decode_reference`` takes. ``Sfp`` is where the fp tier ends
    and the Q tier begins; it defaults to "all of it" (no Q tier).

    Returns ``(out [B,H_q,D], wsum [B,H_q,W_phys])`` where ``W_phys`` is
    ``n_body_win + n_q_win`` in **physical** order — body windows then Q windows,
    which is the order ``scorer.reduce_two_tier_scores`` scatters from.
    """
    if q.dim() == 3:
        q = q.unsqueeze(2)
    B, H_q, _, D = q.shape
    H_kv, S = k_eff.shape[1], k_eff.shape[2]
    rep = H_q // H_kv
    ws = window_size
    block_nw, _ = window_tiling(ws)

    # The fp store's REAL length. Deriving it as num_sink + n_body_win*ws would
    # overrun into the Q tier whenever the newest body window is partial, which
    # is the common case. The kernel masks the body at Sfp for the same reason.
    body_end = S if Sfp is None else min(Sfp, S)
    n_q_win = -(-(S - body_end) // ws)
    W_phys = n_body_win + n_q_win

    q5 = q.reshape(B, H_kv, rep, 1, D).float()
    k = k_eff.unsqueeze(2).float()
    logits = torch.matmul(q5, k.transpose(-2, -1)) * scaling      # [B,H_kv,rep,1,S]
    logits = logits.reshape(B, H_q, S)
    v_flat = (v_eff.reshape(B, H_kv, 1, S, D)
              .expand(B, H_kv, rep, S, D).reshape(B, H_q, S, D).float())

    NEG = float("-inf")
    m = torch.full((B, H_q), NEG, dtype=torch.float32)
    l = torch.zeros((B, H_q), dtype=torch.float32)
    acc = torch.zeros((B, H_q, D), dtype=torch.float32)
    wsum = torch.zeros((B, H_q, W_phys), dtype=torch.float32)
    wmax = torch.full((B, H_q, W_phys), NEG, dtype=torch.float32)

    def tile(lo: int, hi: int, win_base: int) -> None:
        """One tile. ``win_base < 0`` means "softmax only" (the sink prologue).

        ``lo`` is on a window boundary whenever ``win_base >= 0``, so window ``j``
        of the tile is exactly keys ``[lo + j*ws, lo + (j+1)*ws)`` clipped to
        ``hi``. That is the property the window tiling buys, and it is why no
        window is ever split across two tiles for any ``ws``.
        """
        nonlocal m, l, acc
        if hi <= lo:
            return
        lg = logits[:, :, lo:hi]
        m_new = torch.maximum(m, lg.max(dim=-1).values)
        corr = torch.exp(m - m_new)
        corr = torch.nan_to_num(corr, nan=0.0)          # first tile: -inf - -inf
        p = torch.exp(lg - m_new.unsqueeze(-1))
        acc = acc * corr.unsqueeze(-1) + torch.matmul(
            p.unsqueeze(-2), v_flat[:, :, lo:hi]).squeeze(-2)
        l = l * corr + p.sum(dim=-1)
        m = m_new
        if win_base < 0:
            return
        for j in range(-(-(hi - lo) // ws)):
            w = win_base + j
            if w >= W_phys:
                break
            a, b = j * ws, min((j + 1) * ws, hi - lo)
            wsum[:, :, w] = p[:, :, a:b].sum(dim=-1)
            wmax[:, :, w] = m_new

    tile(0, num_sink, -1)                                  # sink prologue
    pos, wb = num_sink, 0                                  # fp body
    while pos < body_end:
        hi = min(pos + block_nw * ws, body_end)
        tile(pos, hi, wb)
        wb += -(-(hi - pos) // ws)
        pos = hi
    pos, wb = body_end, n_body_win                         # Q tier
    while pos < S:
        hi = min(pos + block_nw * ws, S)
        tile(pos, hi, wb)
        wb += -(-(hi - pos) // ws)
        pos = hi

    out = acc / l.unsqueeze(-1)
    lse = m + torch.log(l)
    live = wmax > NEG
    wsum = torch.where(live, wsum * torch.exp(wmax - lse.unsqueeze(-1)),
                       torch.zeros_like(wsum))
    return out.to(v_eff.dtype), wsum


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
        OUT, WSUM, WMAX,
        scale,
        H_q, H_kv, n_active, Sfp, rep, num_sink, n_body_win, W_phys,
        sqb, sqh, sqd,
        kfb, kfh, kfs, kfd,
        vfb, vfh, vfs, vfd,
        HEAD_DIM: tl.constexpr, HALF: tl.constexpr, WS: tl.constexpr,
        BLOCK_R: tl.constexpr, BLOCK_NW: tl.constexpr, BLOCK_T: tl.constexpr,
        BLOCK_W: tl.constexpr, PACK_K: tl.constexpr, PACK_V: tl.constexpr,
    ):
        """One program == one (batch, KV head); all ``rep`` query heads at once.

        Emits **per-window** softmax mass (DECODE_SPEED_PLAN §5.1) instead of a
        ``[B,H_q,S]`` per-token buffer that then has to be read back twice and
        reduced on the host. Both tiers are tiled in **whole windows** (§5.3), so
        no window is ever split across a tile and the scheme holds for any
        ``window_size`` -- see :func:`window_tiling`.

        The exact algorithm, tile for tile, is mirrored on CPU by
        :func:`two_tier_window_reference` and pinned by
        ``tests/test_window_scores.py``. **Keep the two in step**: that oracle is
        the only thing standing between this kernel and an unverified rewrite,
        because Triton cannot run on the CPU box this repo is developed on.

        Strides for the Q-tier tensors, ``COS``/``SIN``, ``OUT``, ``WSUM`` and
        ``WMAX`` are DERIVED from shapes rather than passed (§5.2); the dispatcher
        asserts those tensors are contiguous, so it is a checked contract rather
        than an assumption.
        """
        pid = tl.program_id(0)
        b = pid // H_kv
        kv = pid % H_kv

        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < rep
        hq = kv * rep + offs_r                              # [BLOCK_R] query heads
        offs_hl = tl.arange(0, HALF)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_t = tl.arange(0, BLOCK_T)
        t_win = offs_t // WS                                # window index within tile
        t_tok = offs_t % WS                                 # token index within window
        in_tile = offs_t < (BLOCK_NW * WS)                  # masks the pow2 padding

        qg = tl.load(Q + b * sqb + hq[:, None] * sqh + offs_d[None, :] * sqd,
                     mask=r_mask[:, None], other=0.0).to(tl.float32)   # [BLOCK_R, D]
        q_lo = tl.load(Q + b * sqb + hq[:, None] * sqh + offs_hl[None, :] * sqd,
                       mask=r_mask[:, None], other=0.0).to(tl.float32)
        q_hi = tl.load(Q + b * sqb + hq[:, None] * sqh + (offs_hl + HALF)[None, :] * sqd,
                       mask=r_mask[:, None], other=0.0).to(tl.float32)

        m = tl.full([BLOCK_R], -float("inf"), tl.float32)
        l = tl.zeros([BLOCK_R], tl.float32)
        acc = tl.zeros([BLOCK_R, HEAD_DIM], tl.float32)

        # Derived strides -- contiguous by contract (see the dispatcher's check).
        wsb = H_q * W_phys
        wsh = W_phys
        ob = H_q * HEAD_DIM
        kcb = n_active * H_kv * HEAD_DIM * PACK_K
        kcn = H_kv * HEAD_DIM * PACK_K
        kch = HEAD_DIM * PACK_K
        ksb = n_active * H_kv * HEAD_DIM
        ksn = H_kv * HEAD_DIM
        ksh = HEAD_DIM
        vcb = n_active * H_kv * WS * PACK_V
        vcn = H_kv * WS * PACK_V
        vch = WS * PACK_V
        vsb = n_active * H_kv * WS
        vsn = H_kv * WS
        vsh = WS
        cob = n_active * WS * HALF

        # ---- 1. sink prologue: softmax only, emits no window score -----------
        # Sinks are not represented in window scores (the scorer strips them
        # before windowing), so they contribute to `out` and the LSE, nothing else.
        for s0 in range(0, num_sink, BLOCK_T):
            offs_n = s0 + offs_t
            nmask = offs_n < num_sink
            kf = tl.load(KFP + b * kfb + kv * kfh + offs_n[:, None] * kfs
                         + offs_d[None, :] * kfd,
                         mask=nmask[:, None], other=0.0).to(tl.float32)
            logit = tl.dot(qg, tl.trans(kf)) * scale
            logit = tl.where(nmask[None, :], logit, -float("inf"))
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp(m - m_new)
            p = tl.where(nmask[None, :], tl.exp(logit - m_new[:, None]), 0.0)
            vf = tl.load(VFP + b * vfb + kv * vfh + offs_n[:, None] * vfs
                         + offs_d[None, :] * vfd,
                         mask=nmask[:, None], other=0.0).to(tl.float32)
            acc = acc * corr[:, None] + tl.dot(p, vf)
            l = l * corr + tl.sum(p, axis=1)
            m = m_new

        # ---- 2. fp body: whole-window tiles, starting at num_sink -------------
        for w0 in range(0, n_body_win, BLOCK_NW):
            key0 = num_sink + w0 * WS
            offs_n = key0 + offs_t
            nmask = in_tile & (offs_n < Sfp)
            kf = tl.load(KFP + b * kfb + kv * kfh + offs_n[:, None] * kfs
                         + offs_d[None, :] * kfd,
                         mask=nmask[:, None], other=0.0).to(tl.float32)
            logit = tl.dot(qg, tl.trans(kf)) * scale
            logit = tl.where(nmask[None, :], logit, -float("inf"))
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp(m - m_new)
            p = tl.where(nmask[None, :], tl.exp(logit - m_new[:, None]), 0.0)
            vf = tl.load(VFP + b * vfb + kv * vfh + offs_n[:, None] * vfs
                         + offs_d[None, :] * vfd,
                         mask=nmask[:, None], other=0.0).to(tl.float32)
            acc = acc * corr[:, None] + tl.dot(p, vf)
            l = l * corr + tl.sum(p, axis=1)
            m = m_new
            # Per-window sums against THIS tile's running max. One WMAX per
            # window is what lets the epilogue rescale to the global LSE.
            for j in tl.static_range(BLOCK_NW):
                w = w0 + j
                sel = (t_win == j) & nmask
                pj = tl.sum(tl.where(sel[None, :], p, 0.0), axis=1)
                keep = r_mask & (w < n_body_win)
                tl.store(WSUM + b * wsb + hq * wsh + w, pj, mask=keep)
                tl.store(WMAX + b * wsb + hq * wsh + w, m_new, mask=keep)

        # ---- 3. Q tier: int2 dequant + RoPE in registers, whole-window tiles ---
        cbyte = (offs_d // 4)
        cshift = (2 * (offs_d % 4)).to(tl.uint8)
        byte_t = (t_tok // 4)
        shift_t = (2 * (t_tok % 4)).to(tl.uint8)
        for w0 in range(0, n_active, BLOCK_NW):
            widx = w0 + t_win                                # [BLOCK_T] window ids
            qmask = in_tile & (widx < n_active)
            ks_lo = tl.load(KS + b * ksb + widx[None, :] * ksn + kv * ksh
                            + offs_hl[:, None],
                            mask=qmask[None, :], other=0.0).to(tl.float32)
            kz_lo = tl.load(KZ + b * ksb + widx[None, :] * ksn + kv * ksh
                            + offs_hl[:, None],
                            mask=qmask[None, :], other=0.0).to(tl.float32)
            ks_hi = tl.load(KS + b * ksb + widx[None, :] * ksn + kv * ksh
                            + (offs_hl + HALF)[:, None],
                            mask=qmask[None, :], other=0.0).to(tl.float32)
            kz_hi = tl.load(KZ + b * ksb + widx[None, :] * ksn + kv * ksh
                            + (offs_hl + HALF)[:, None],
                            mask=qmask[None, :], other=0.0).to(tl.float32)
            kb_lo = tl.load(KC + b * kcb + widx[None, :] * kcn + kv * kch
                            + offs_hl[:, None] * PACK_K + byte_t[None, :],
                            mask=qmask[None, :], other=0)
            kb_hi = tl.load(KC + b * kcb + widx[None, :] * kcn + kv * kch
                            + (offs_hl + HALF)[:, None] * PACK_K + byte_t[None, :],
                            mask=qmask[None, :], other=0)
            k_lo = ((kb_lo >> shift_t[None, :]) & 3).to(tl.float32) * ks_lo + kz_lo
            k_hi = ((kb_hi >> shift_t[None, :]) & 3).to(tl.float32) * ks_hi + kz_hi
            crow = widx * WS + t_tok
            c = tl.load(COS + b * cob + crow[None, :] * HALF + offs_hl[:, None],
                        mask=qmask[None, :], other=0.0)
            s = tl.load(SIN + b * cob + crow[None, :] * HALF + offs_hl[:, None],
                        mask=qmask[None, :], other=0.0)
            k_rlo = k_lo * c - k_hi * s
            k_rhi = k_hi * c + k_lo * s
            logit = (tl.dot(q_lo, k_rlo) + tl.dot(q_hi, k_rhi)) * scale
            logit = tl.where(qmask[None, :], logit, -float("inf"))
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp(m - m_new)
            p = tl.where(qmask[None, :], tl.exp(logit - m_new[:, None]), 0.0)
            vs = tl.load(VS + b * vsb + widx * vsn + kv * vsh + t_tok,
                         mask=qmask, other=0.0).to(tl.float32)
            vz = tl.load(VZ + b * vsb + widx * vsn + kv * vsh + t_tok,
                         mask=qmask, other=0.0).to(tl.float32)
            vb = tl.load(VC + b * vcb + widx[:, None] * vcn + kv * vch
                         + t_tok[:, None] * PACK_V + cbyte[None, :],
                         mask=qmask[:, None], other=0)
            vv = ((vb >> cshift[None, :]) & 3).to(tl.float32) * vs[:, None] + vz[:, None]
            acc = acc * corr[:, None] + tl.dot(p, vv)
            l = l * corr + tl.sum(p, axis=1)
            m = m_new
            for j in tl.static_range(BLOCK_NW):
                w = w0 + j
                sel = (t_win == j) & qmask
                pj = tl.sum(tl.where(sel[None, :], p, 0.0), axis=1)
                keep = r_mask & (w < n_active)
                col = n_body_win + w
                tl.store(WSUM + b * wsb + hq * wsh + col, pj, mask=keep)
                tl.store(WMAX + b * wsb + hq * wsh + col, m_new, mask=keep)

        out = acc / l[:, None]
        lse = m + tl.log(l)
        tl.store(OUT + b * ob + hq[:, None] * HEAD_DIM + offs_d[None, :],
                 out.to(OUT.dtype.element_ty), mask=r_mask[:, None])

        # ---- 4. epilogue: rescale each window from its tile max to the LSE ----
        # Runs over W_phys values, not S. That is the whole of §5.1's traffic cut.
        offs_w = tl.arange(0, BLOCK_W)
        for w0 in range(0, W_phys, BLOCK_W):
            cols = w0 + offs_w
            cmask = cols < W_phys
            ptr = WSUM + b * wsb + hq[:, None] * wsh + cols[None, :]
            mptr = WMAX + b * wsb + hq[:, None] * wsh + cols[None, :]
            sm = r_mask[:, None] & cmask[None, :]
            ssum = tl.load(ptr, mask=sm, other=0.0)
            smax = tl.load(mptr, mask=sm, other=-float("inf"))
            scaled = tl.where(smax > -float("inf"),
                              ssum * tl.exp(smax - lse[:, None]), 0.0)
            tl.store(ptr, scaled, mask=sm)


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
    num_sink: int = 0,
    n_body_win: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Launch the fused decode kernel.

    Returns ``(out [B,H_q,D], wsum [B,H_q,W_phys])`` — per-**window** softmax mass
    in physical order (body windows, then Q windows), which is the order
    :func:`scorer.reduce_two_tier_scores` scatters from. §5.1 replaced a
    ``[B,H_q,S]`` token-score buffer that was written, read back, rewritten and
    then reduced on the host; the reduction now happens in registers, and the
    caller is left with one gather.

    ``qtier`` (or None for an empty Q tier) carries the gathered active-window
    int2 fields shaped ``[B, n, H_kv, ...]`` plus RoPE halves ``cos``/``sin``
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

    body = max(Sfp - num_sink, 0)
    if n_body_win is None:
        n_body_win = -(-body // ws)
    if n_body_win * ws < body:
        raise ValueError(
            f"n_body_win={n_body_win} covers {n_body_win * ws} tokens but the fp "
            f"body holds {body} (Sfp={Sfp}, num_sink={num_sink}). Every body token "
            "must fall inside a scored window or its attention mass would be "
            "dropped from the eviction scores."
        )
    W_phys = n_body_win + n_active

    out = torch.empty((B, H_q, D), device=q.device, dtype=q.dtype)
    wsum = torch.empty((B, H_q, W_phys), device=q.device, dtype=torch.float32)
    wmax = torch.empty((B, H_q, W_phys), device=q.device, dtype=torch.float32)

    # Dummies for the empty-Q case: valid tensors so the pointers exist; never
    # indexed (the Q loop runs only while w0 < n_active == 0).
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

    # §5.2 passes shapes instead of strides for these, so contiguity stops being
    # an assumption and becomes a checked contract. It holds by construction --
    # they come from `QuantSlotTable.gather` (a fresh index_select) reshaped, and
    # from `rope_cos_sin_halves`, which calls `.contiguous()` -- so this never
    # fires in practice and costs one flag read per launch. It is here because a
    # silently non-contiguous tensor would read garbage rather than fail.
    for name, t in (("k_codes", KC), ("k_scale", KS), ("k_zero", KZ),
                    ("v_codes", VC), ("v_scale", VS), ("v_zero", VZ),
                    ("cos", COS), ("sin", SIN)):
        if not t.is_contiguous():
            raise RuntimeError(
                f"fused decode requires a contiguous {name}; its strides are "
                f"derived from its shape inside the kernel (DECODE_SPEED_PLAN "
                f"§5.2). Got shape {tuple(t.shape)} strides {tuple(t.stride())}."
            )

    BLOCK_R = _pow2_at_least(rep)
    BLOCK_NW, BLOCK_T = window_tiling(ws)
    BLOCK_W = _pow2_at_least(min(W_phys, 128), floor=16)
    grid = (B * H_kv,)
    _two_tier_decode_kernel[grid](
        q, k_fp, v_fp, KC, KS, KZ, VC, VS, VZ, COS, SIN, out, wsum, wmax,
        scaling,
        H_q, H_kv, n_active, Sfp, rep, num_sink, n_body_win, W_phys,
        q.stride(0), q.stride(1), q.stride(2),
        k_fp.stride(0), k_fp.stride(1), k_fp.stride(2), k_fp.stride(3),
        v_fp.stride(0), v_fp.stride(1), v_fp.stride(2), v_fp.stride(3),
        HEAD_DIM=D, HALF=half, WS=ws,
        BLOCK_R=BLOCK_R, BLOCK_NW=BLOCK_NW, BLOCK_T=BLOCK_T, BLOCK_W=BLOCK_W,
        PACK_K=max(ws // 4, 1), PACK_V=max(D // 4, 1),
    )
    return out, wsum


# ---------------------------------------------------------------------------
# Public dispatcher — Triton-or-raise
# ---------------------------------------------------------------------------


def fused_two_tier_decode(
    q: Tensor,
    k_fp: Tensor,
    v_fp: Tensor,
    qtier: Optional[dict],
    scaling: float,
    num_sink: int = 0,
    n_body_win: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Fused decode attention + score. **Triton-or-raise** — no PyTorch fallback.

    q : ``[B, H_q, D]`` post-RoPE decode query.
    k_fp, v_fp : ``[B, H_kv, S_fp, D]`` fp tier (post-RoPE keys), ``[sink ‖ body]``.
    qtier : int2 Q-tier context (see :func:`_decode_triton`) or ``None`` (empty).
    num_sink : sink tokens at the head of ``k_fp``; they carry no window score.
    n_body_win : scored windows the fp body spans. Defaults to
        ``ceil((S_fp - num_sink) / ws)``; pass it when the caller already knows
        it, so the kernel's window axis matches the caller's score axis exactly.

    Returns ``(out [B,H_q,D], wsum [B,H_q,W_phys])`` — per-window softmax mass in
    physical order (body windows, then Q windows).
    """
    if not (_HAS_TRITON and q.is_cuda):
        reason = "triton not installed" if not _HAS_TRITON else "not on CUDA"
        raise RuntimeError(
            "fused_two_tier_decode requires the Triton kernel on CUDA "
            f"({reason}); there is no PyTorch decode fallback in production "
            "(the reference is for tests only). See assert_decode_kernel_available."
        )
    return _decode_triton(q, k_fp, v_fp, qtier, scaling, num_sink, n_body_win)
