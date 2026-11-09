"""The fused two-tier decode kernel, for an MLA latent (DeepSeek-V2).

This is ``decode_kernel._two_tier_decode_kernel`` with exactly three things
changed, each forced by the latent and none of them by the method:

1. **The head is not a power of two.** The latent key is ``Dc + Dr = 576``
   wide, packed (``models/deepseek_v2/geometry.py``) so each half is
   ``[latent HA | RoPE HB]`` -- ``256 + 32`` on V2-Lite, both powers of two.
   ``tl.arange`` admits only powers of two, so where the shipped kernel loads a
   half as one ``[HALF]`` tile this loads it as an ``[HA]`` and an ``[HB]`` tile.
   The logit is the sum of the four partial dots; nothing else about the tiling
   moves.
2. **RoPE touches only the HB lanes.** The packed tables carry ``cos = 1,
   sin = 0`` on every latent lane, where the shipped kernel's
   ``k_lo*c - k_hi*s`` is an exact identity (``x*1 - y*0 == x``). So the latent
   lanes skip the multiply and the RoPE lanes rotate as before. Bit-identical
   for finite keys.
3. **The value is the key's latent lanes.** One dequantized ``[HA, T]`` tile
   per half serves QK *and* PV, so the int2 tier is unpacked once, not twice,
   and there are no value codes to load. The output's RoPE lanes are written as
   zero; the attention module reads only the latent lanes.

Everything else is the shipped kernel's, line for line: one program per (row,
KV head) with every query head at once (``rep = H_q = 16``, which fills
``BLOCK_R = 16`` exactly -- on Llama it padded 4 heads to 16), the sink
prologue, whole-window tiles of the fp body and the int2 tier, the GATED loop
over ``n_sel`` slots dereferenced through ``SEL``, the per-window score epilogue,
the mean-log-ratio fill of the skipped windows' scores **and** their centroid
contribution to the output (the centroid is ``mu``, see ``sketch.value_centroid``),
and the renormalisation by ``1 + sum(restored mass)``.

What it does NOT have yet, and why that matters: **split-K**. With one KV head
the grid is ``B`` programs, so at small ``B`` the GPU is mostly idle and one
program walks the whole cache. The shipped kernel has the same gap for small
``B * H_kv``, but MLA makes it the common case. A split-K variant needs the fill
moved into a combine pass (the deviation is a mean over the read set across all
splits) and is the next kernel change, to be measured on a GPU.

**Unvalidated on a GPU, by contract** (``CLAUDE.md``). The algorithm is pinned:
``tests/interp_mla_kernels.py`` runs this kernel under ``TRITON_INTERPRET=1``
against ``decode_kernel.two_tier_window_reference`` over a real int2 latent tier
with real cards, gated and ungated.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from modules.quant.quantizer import grid_group
from modules.quant.sketch import card_field_layout

from .decode_kernel import (
    _EXP2,
    _FIT_ANNOUNCED,
    _FIT_CHOICE,
    _FIT_LADDER,
    _FIT_SEEN,
    _FIT_TIMINGS,
    _LOG2E,
    _HAS_TRITON,
    _pow2_at_least,
    _scratch,
    _search_rungs,
    check_gate_selection,
    window_tiling,
)

if _HAS_TRITON:  # pragma: no cover - exercised only where triton is present
    import triton
    import triton.language as tl

    from .gate_kernel import _card_codes

    @triton.jit
    def _dq_rows(KC, KS, KSS, KZ, KZS, cbase, sbase, gbase, widx, ch, qmask,
                 kcn, ksn, ksgn, byte_t, shift_t,
                 PACK_K: tl.constexpr, GROUP_K: tl.constexpr):
        """Dequantize channels ``ch`` of the int2 key tile: ``[len(ch), BLOCK_T]``.

        The shipped kernel's channel-major unpack, one channel range at a time:
        the one-byte grid times the fp16 scale its group shares, codes fit to
        that stored grid, ``x = code * scale + zero``.
        """
        kg = (ch // GROUP_K)[:, None]
        sc = (tl.load(KS + sbase + widx[None, :] * ksn + ch[:, None],
                      mask=qmask[None, :], other=0).to(tl.float32)
              * tl.load(KSS + gbase + widx[None, :] * ksgn + kg,
                        mask=qmask[None, :], other=0.0).to(tl.float32))
        zr = (tl.load(KZ + sbase + widx[None, :] * ksn + ch[:, None],
                      mask=qmask[None, :], other=0).to(tl.float32)
              * tl.load(KZS + gbase + widx[None, :] * ksgn + kg,
                        mask=qmask[None, :], other=0.0).to(tl.float32))
        kb = tl.load(KC + cbase + widx[None, :] * kcn + ch[:, None] * PACK_K
                     + byte_t[None, :], mask=qmask[None, :], other=0)
        return ((kb >> shift_t[None, :]) & 3).to(tl.float32) * sc + zr

    @triton.jit
    def _mla_two_tier_decode_kernel(
        Q, KFP,
        KC, KS, KSS, KZ, KZS,
        COS, SIN,
        SEL, LOGM,
        VM, VMS, VANC,
        OUT, WSUM, WMAX,
        scale,
        H_q, n_active, n_sel, Sfp, rep, num_sink, n_body_win, W_phys,
        sqb, sqh, sqd,
        kfb, kfs, kfd,
        selb,
        HEAD_DIM: tl.constexpr, HALF: tl.constexpr,
        HA: tl.constexpr, HB: tl.constexpr, WS: tl.constexpr,
        BLOCK_R: tl.constexpr, BLOCK_NW: tl.constexpr, BLOCK_T: tl.constexpr,
        BLOCK_W: tl.constexpr, PACK_K: tl.constexpr, GROUP_K: tl.constexpr,
        GATED: tl.constexpr, LOG2E: tl.constexpr, VM_BITS: tl.constexpr,
    ):
        """One program == one row (``H_kv == 1``); all ``rep`` query heads at once.

        Section numbers are the shipped kernel's; see its docstring for the
        reasoning behind each, which is unchanged here.
        """
        b = tl.program_id(0)

        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < rep
        hq = offs_r                                         # every head of the row
        offs_a = tl.arange(0, HA)                           # latent lanes of a half
        offs_b = tl.arange(0, HB)                           # RoPE lanes of a half
        offs_t = tl.arange(0, BLOCK_T)
        t_win = offs_t // WS
        t_tok = offs_t % WS
        in_tile = offs_t < (BLOCK_NW * WS)

        qrow = Q + b * sqb + hq[:, None] * sqh
        q_la = tl.load(qrow + offs_a[None, :] * sqd,
                       mask=r_mask[:, None], other=0.0).to(tl.float32)
        q_lb = tl.load(qrow + (HA + offs_b)[None, :] * sqd,
                       mask=r_mask[:, None], other=0.0).to(tl.float32)
        q_ha = tl.load(qrow + (HALF + offs_a)[None, :] * sqd,
                       mask=r_mask[:, None], other=0.0).to(tl.float32)
        q_hb = tl.load(qrow + (HALF + HA + offs_b)[None, :] * sqd,
                       mask=r_mask[:, None], other=0.0).to(tl.float32)

        m = tl.full([BLOCK_R], -float("inf"), tl.float32)
        l = tl.zeros([BLOCK_R], tl.float32)
        acc_a = tl.zeros([BLOCK_R, HA], tl.float32)         # latent, low half
        acc_c = tl.zeros([BLOCK_R, HA], tl.float32)         # latent, high half

        wsb = H_q * W_phys
        wsh = W_phys
        lgb = H_q * n_active
        lgh = n_active
        ob = H_q * HEAD_DIM
        kcb = n_active * HEAD_DIM * PACK_K
        kcn = HEAD_DIM * PACK_K
        ksb = n_active * HEAD_DIM
        ksn = HEAD_DIM
        GK: tl.constexpr = HEAD_DIM // GROUP_K
        ksgb = n_active * GK
        ksgn = GK
        cob = n_active * WS * HALF
        VMW: tl.constexpr = HEAD_DIM * VM_BITS // 8
        vmb = n_active * VMW
        vmn = VMW
        vmsb = n_active
        vab = HEAD_DIM

        # ---- 1. sink prologue ------------------------------------------------
        for s0 in range(0, num_sink, BLOCK_T):
            offs_n = s0 + offs_t
            nmask = offs_n < num_sink
            krow = KFP + b * kfb + offs_n[:, None] * kfs
            k_la = tl.load(krow + offs_a[None, :] * kfd, mask=nmask[:, None],
                           other=0.0).to(tl.float32)
            k_lb = tl.load(krow + (HA + offs_b)[None, :] * kfd, mask=nmask[:, None],
                           other=0.0).to(tl.float32)
            k_ha = tl.load(krow + (HALF + offs_a)[None, :] * kfd,
                           mask=nmask[:, None], other=0.0).to(tl.float32)
            k_hb = tl.load(krow + (HALF + HA + offs_b)[None, :] * kfd,
                           mask=nmask[:, None], other=0.0).to(tl.float32)
            logit = (tl.dot(q_la, tl.trans(k_la)) + tl.dot(q_lb, tl.trans(k_lb))
                     + tl.dot(q_ha, tl.trans(k_ha))
                     + tl.dot(q_hb, tl.trans(k_hb))) * scale
            logit = tl.where(nmask[None, :], logit, -float("inf"))
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp2(m - m_new)
            p = tl.where(nmask[None, :], tl.exp2(logit - m_new[:, None]), 0.0)
            acc_a = acc_a * corr[:, None] + tl.dot(p, k_la)
            acc_c = acc_c * corr[:, None] + tl.dot(p, k_ha)
            l = l * corr + tl.sum(p, axis=1)
            m = m_new

        # ---- 2. fp body: whole-window tiles ----------------------------------
        for w0 in range(0, n_body_win, BLOCK_NW):
            key0 = num_sink + w0 * WS
            offs_n = key0 + offs_t
            nmask = in_tile & (offs_n < Sfp)
            krow = KFP + b * kfb + offs_n[:, None] * kfs
            k_la = tl.load(krow + offs_a[None, :] * kfd, mask=nmask[:, None],
                           other=0.0).to(tl.float32)
            k_lb = tl.load(krow + (HA + offs_b)[None, :] * kfd, mask=nmask[:, None],
                           other=0.0).to(tl.float32)
            k_ha = tl.load(krow + (HALF + offs_a)[None, :] * kfd,
                           mask=nmask[:, None], other=0.0).to(tl.float32)
            k_hb = tl.load(krow + (HALF + HA + offs_b)[None, :] * kfd,
                           mask=nmask[:, None], other=0.0).to(tl.float32)
            logit = (tl.dot(q_la, tl.trans(k_la)) + tl.dot(q_lb, tl.trans(k_lb))
                     + tl.dot(q_ha, tl.trans(k_ha))
                     + tl.dot(q_hb, tl.trans(k_hb))) * scale
            logit = tl.where(nmask[None, :], logit, -float("inf"))
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp2(m - m_new)
            p = tl.where(nmask[None, :], tl.exp2(logit - m_new[:, None]), 0.0)
            acc_a = acc_a * corr[:, None] + tl.dot(p, k_la)
            acc_c = acc_c * corr[:, None] + tl.dot(p, k_ha)
            l = l * corr + tl.sum(p, axis=1)
            m = m_new
            for j in tl.static_range(BLOCK_NW):
                w = w0 + j
                sel = (t_win == j) & nmask
                pj = tl.sum(tl.where(sel[None, :], p, 0.0), axis=1)
                # `offs_r * 0 + w`: a tensor in both the compiler (where `w` is
                # already one) and the interpreter (where it is a Python int and
                # `bool_tensor & python_bool` is not a valid mask). Folded away
                # when compiled.
                keep = r_mask & ((offs_r * 0 + w) < n_body_win)
                tl.store(WSUM + b * wsb + hq * wsh + w, pj, mask=keep)
                tl.store(WMAX + b * wsb + hq * wsh + w, m_new, mask=keep)

        # ---- 3. int2 tier ----------------------------------------------------
        if GATED:
            offs_wi = tl.arange(0, BLOCK_W)
            for wi0 in range(n_body_win, W_phys, BLOCK_W):
                icols = wi0 + offs_wi
                imask = r_mask[:, None] & (icols < W_phys)[None, :]
                iptr = b * wsb + hq[:, None] * wsh + icols[None, :]
                tl.store(WSUM + iptr, tl.zeros([BLOCK_R, BLOCK_W], tl.float32),
                         mask=imask)
                tl.store(WMAX + iptr,
                         tl.full([BLOCK_R, BLOCK_W], -float("inf"), tl.float32),
                         mask=imask)

        byte_t = (t_tok // 4)
        shift_t = (2 * (t_tok % 4)).to(tl.uint8)
        if GATED:
            n_q_iter = n_sel
        else:
            n_q_iter = n_active
        for w0 in range(0, n_q_iter, BLOCK_NW):
            if GATED:
                slot = w0 + t_win
                qmask = in_tile & (slot < n_sel)
                widx = tl.load(SEL + b * selb
                               + tl.minimum(slot, n_sel - 1)).to(tl.int32)
            else:
                widx = w0 + t_win
                qmask = in_tile & (widx < n_active)
            cbase = b * kcb
            sbase = b * ksb
            gbase = b * ksgb
            k_la = _dq_rows(KC, KS, KSS, KZ, KZS, cbase, sbase, gbase, widx,
                            offs_a, qmask, kcn, ksn, ksgn, byte_t, shift_t,
                            PACK_K, GROUP_K)                    # [HA, T]
            k_lb = _dq_rows(KC, KS, KSS, KZ, KZS, cbase, sbase, gbase, widx,
                            HA + offs_b, qmask, kcn, ksn, ksgn, byte_t, shift_t,
                            PACK_K, GROUP_K)                    # [HB, T]
            k_ha = _dq_rows(KC, KS, KSS, KZ, KZS, cbase, sbase, gbase, widx,
                            HALF + offs_a, qmask, kcn, ksn, ksgn, byte_t, shift_t,
                            PACK_K, GROUP_K)
            k_hb = _dq_rows(KC, KS, KSS, KZ, KZS, cbase, sbase, gbase, widx,
                            HALF + HA + offs_b, qmask, kcn, ksn, ksgn, byte_t,
                            shift_t, PACK_K, GROUP_K)
            # RoPE on the HB lanes only: the latent lanes' cos/sin are 1/0.
            crow = widx * WS + t_tok
            c = tl.load(COS + b * cob + crow[None, :] * HALF + (HA + offs_b)[:, None],
                        mask=qmask[None, :], other=0.0).to(tl.float32)
            s = tl.load(SIN + b * cob + crow[None, :] * HALF + (HA + offs_b)[:, None],
                        mask=qmask[None, :], other=0.0).to(tl.float32)
            k_rlb = k_lb * c - k_hb * s
            k_rhb = k_hb * c + k_lb * s
            logit = (tl.dot(q_la, k_la) + tl.dot(q_lb, k_rlb)
                     + tl.dot(q_ha, k_ha) + tl.dot(q_hb, k_rhb)) * scale
            logit = tl.where(qmask[None, :], logit, -float("inf"))
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp2(m - m_new)
            p = tl.where(qmask[None, :], tl.exp2(logit - m_new[:, None]), 0.0)
            # The value is the latent: the SAME dequantized tiles, transposed.
            acc_a = acc_a * corr[:, None] + tl.dot(p, tl.trans(k_la))
            acc_c = acc_c * corr[:, None] + tl.dot(p, tl.trans(k_ha))
            l = l * corr + tl.sum(p, axis=1)
            m = m_new
            for j in tl.static_range(BLOCK_NW):
                lane = (t_win == j) & qmask
                pj = tl.sum(tl.where(lane[None, :], p, 0.0), axis=1)
                if GATED:
                    sj = w0 + j
                    col = n_body_win + tl.load(
                        SEL + b * selb + tl.minimum(sj, n_sel - 1)).to(tl.int32)
                    keep = r_mask & ((offs_r * 0 + sj) < n_sel)
                else:
                    w = w0 + j
                    col = n_body_win + w
                    keep = r_mask & ((offs_r * 0 + w) < n_active)
                tl.store(WSUM + b * wsb + hq * wsh + col, pj, mask=keep)
                tl.store(WMAX + b * wsb + hq * wsh + col, m_new, mask=keep)

        out_a = acc_a / l[:, None]
        out_c = acc_c / l[:, None]
        lse = m + tl.log2(l)

        # ---- 4. epilogue: rescale every window to the LSE; measure the cards --
        offs_w = tl.arange(0, BLOCK_W)
        dsum = tl.zeros([BLOCK_R], tl.float32)
        dcnt = tl.zeros([BLOCK_R], tl.float32)
        for w0 in range(0, W_phys, BLOCK_W):
            cols = w0 + offs_w
            cmask = cols < W_phys
            ptr = WSUM + b * wsb + hq[:, None] * wsh + cols[None, :]
            mptr = WMAX + b * wsb + hq[:, None] * wsh + cols[None, :]
            sm = r_mask[:, None] & cmask[None, :]
            ssum = tl.load(ptr, mask=sm, other=0.0)
            smax = tl.load(mptr, mask=sm, other=-float("inf"))
            scaled = tl.where(smax > -float("inf"),
                              ssum * tl.exp2(smax - lse[:, None]), 0.0)
            tl.store(ptr, scaled, mask=sm)
            if GATED:
                live = (sm & (smax > -float("inf")) & (ssum > 0.0)
                        & (cols[None, :] >= n_body_win))
                qc = tl.maximum(cols - n_body_win, 0)
                lg = tl.load(LOGM + b * lgb + hq[:, None] * lgh + qc[None, :],
                             mask=live, other=0.0)
                lsc = (tl.log2(tl.where(live, ssum, 1.0)) + smax
                       - lse[:, None]) / LOG2E
                dsum += tl.sum(tl.where(live, lsc - lg, 0.0), axis=1)
                dcnt += tl.sum(tl.where(live, 1.0, 0.0), axis=1)

        # ---- 5. GATED: credit and attend the skipped windows -----------------
        if GATED:
            dmean = tl.where(dcnt > 0.0, dsum / tl.maximum(dcnt, 1.0),
                             -float("inf"))
            vacc_a = tl.zeros([BLOCK_R, HA], tl.float32)
            vacc_c = tl.zeros([BLOCK_R, HA], tl.float32)
            msum = tl.zeros([BLOCK_R], tl.float32)
            vanc_a = tl.load(VANC + b * vab + offs_a).to(tl.float32)
            vanc_c = tl.load(VANC + b * vab + HALF + offs_a).to(tl.float32)
            for w0 in range(n_body_win, W_phys, BLOCK_W):
                cols = w0 + offs_w
                cmask = cols < W_phys
                ptr = WSUM + b * wsb + hq[:, None] * wsh + cols[None, :]
                mptr = WMAX + b * wsb + hq[:, None] * wsh + cols[None, :]
                sm = r_mask[:, None] & cmask[None, :]
                smax = tl.load(mptr, mask=sm, other=0.0)
                skipped = sm & (smax == -float("inf"))
                qc = tl.maximum(cols - n_body_win, 0)
                lg = tl.load(LOGM + b * lgb + hq[:, None] * lgh + qc[None, :],
                             mask=skipped, other=-float("inf"))
                fill = tl.exp2((lg + dmean[:, None]) * LOG2E)
                tl.store(ptr, fill, mask=skipped)

                f = tl.where(skipped, fill, 0.0)
                msum += tl.sum(f, axis=1)
                # The centroid is `mu` (value aliases key): its latent lanes only.
                qmask = cmask & (cols >= n_body_win)
                vrow = VM + b * vmb + qc[:, None] * vmn
                vsc = tl.load(VMS + b * vmsb + qc, mask=qmask, other=0.0).to(tl.float32)
                vq_a = _card_codes(vrow, offs_a[None, :], qmask[:, None], VM_BITS)
                vq_c = _card_codes(vrow, (HALF + offs_a)[None, :], qmask[:, None],
                                   VM_BITS)
                vacc_a = vacc_a + tl.dot(f, vq_a * vsc[:, None] + vanc_a[None, :])
                vacc_c = vacc_c + tl.dot(f, vq_c * vsc[:, None] + vanc_c[None, :])

            den = 1.0 + msum
            out_a = (out_a + vacc_a) / den[:, None]
            out_c = (out_c + vacc_c) / den[:, None]
            for w0 in range(0, W_phys, BLOCK_W):
                cols = w0 + offs_w
                sm = r_mask[:, None] & (cols < W_phys)[None, :]
                ptr = WSUM + b * wsb + hq[:, None] * wsh + cols[None, :]
                tl.store(ptr, tl.load(ptr, mask=sm, other=0.0) / den[:, None],
                         mask=sm)

        orow = OUT + b * ob + hq[:, None] * HEAD_DIM
        tl.store(orow + offs_a[None, :], out_a.to(OUT.dtype.element_ty),
                 mask=r_mask[:, None])
        tl.store(orow + (HALF + offs_a)[None, :], out_c.to(OUT.dtype.element_ty),
                 mask=r_mask[:, None])
        zb = tl.zeros([BLOCK_R, HB], tl.float32).to(OUT.dtype.element_ty)
        tl.store(orow + (HA + offs_b)[None, :], zb, mask=r_mask[:, None])
        tl.store(orow + (HALF + HA + offs_b)[None, :], zb, mask=r_mask[:, None])


def _require_cuda(q: Tensor) -> None:
    """The production guard. ``tests/interp_mla_kernels.py`` replaces it."""
    if not q.is_cuda:
        raise RuntimeError("Fused decode kernel requires CUDA tensors.")


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def mla_decode_triton(
    q: Tensor,
    k_fp: Tensor,
    qtier: dict,
    scaling: float,
    num_sink: int,
    n_body_win: Optional[int],
    sel: Optional[Tensor],
    logmass: Optional[Tensor],
    centroids: Optional[Tuple[Tensor, Tensor, Tensor]],
    vm_bits: int,
) -> Tuple[Tensor, Tensor]:
    """Launch :func:`_mla_two_tier_decode_kernel`. Same contract as ``_decode_triton``.

    ``qtier["mla_split"] = (HA, HB)`` is the packed layout's half split (latent,
    RoPE lanes per half); ``qtier["value_alias"]`` must be True.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton not available; fused decode requires CUDA+triton.")
    _require_cuda(q)
    if not qtier.get("value_alias", False):
        raise RuntimeError("the MLA decode kernel serves an aliased-value tier only")
    if qtier.get("mla_split") is None:
        raise RuntimeError(
            "an aliased-value tier reached the MLA decode kernel without its "
            "half split (latent, RoPE lanes); build the cache from "
            "models.deepseek_v2.geometry.MLACacheConfig, which carries both dims.")
    HA, HB = qtier["mla_split"]
    B, H_q, D = q.shape
    H_kv, Sfp = k_fp.shape[1], k_fp.shape[2]
    if H_kv != 1:
        raise RuntimeError(f"an MLA latent is one KV head; got H_kv={H_kv}")
    half = D // 2
    if HA + HB != half or not (_is_pow2(HA) and _is_pow2(HB)) or HB < 16:
        raise RuntimeError(
            f"MLA half split ({HA}, {HB}) must sum to D/2={half} with both parts "
            "powers of two and the RoPE part >= 16 (tl.dot's minimum)")
    rep = H_q
    n_active = int(qtier["k_codes"].shape[1])
    ws = int(qtier["window_size"])
    body = max(Sfp - num_sink, 0)
    if n_body_win is None:
        n_body_win = -(-body // ws)
    if n_body_win * ws < body:
        raise ValueError(
            f"n_body_win={n_body_win} covers {n_body_win * ws} tokens but the fp "
            f"body holds {body}; every body token must fall inside a scored window.")
    W_phys = n_body_win + n_active

    KC = qtier["k_codes"]
    KS, KSS = qtier["k_scale"]
    KZ, KZS = qtier["k_zero"]
    COS, SIN = qtier["cos"], qtier["sin"]
    for name, t in (("k_codes", KC), ("k_scale", KS), ("k_scale_s", KSS),
                    ("k_zero", KZ), ("k_zero_s", KZS), ("cos", COS), ("sin", SIN)):
        if not t.is_contiguous():
            raise RuntimeError(f"fused MLA decode requires a contiguous {name}")

    out = torch.empty((B, H_q, D), device=q.device, dtype=q.dtype)
    wsum = _scratch("wsum", (B, H_q, W_phys), torch.float32, q.device)
    wmax = _scratch("wmax", (B, H_q, W_phys), torch.float32, q.device)
    dev = q.device

    n_sel = check_gate_selection(sel, B, H_kv, n_active)
    gated = n_sel > 0
    if gated:
        SEL, selb = sel, sel.stride(0)
        if logmass is None or logmass.shape != (B, H_q, n_active) \
                or logmass.dtype != torch.float32 or not logmass.is_contiguous():
            raise RuntimeError(
                "a gated fused MLA decode needs the gate's contiguous fp32 "
                f"logmass [B, H_q, n_active] = [{B}, {H_q}, {n_active}]")
        LOGM = logmass
        if centroids is None:
            raise RuntimeError(
                "a gated fused MLA decode needs the centroids (the card's mu): the "
                "windows it skips attend through them.")
        VM, VMS, VANC = centroids
        vm_width, vm_dtype = card_field_layout(D, vm_bits)
        if VM.shape != (B, n_active, 1, vm_width) or VM.dtype != vm_dtype:
            raise RuntimeError(
                f"centroid codes at {vm_bits} bits must be {vm_dtype} "
                f"[B, n_active, 1, {vm_width}]; got {VM.dtype} {tuple(VM.shape)}")
        if VMS.shape != (B, n_active, 1) or VANC.shape != (B, 1, D):
            raise RuntimeError("centroid scale / anchor shapes do not match the tier")
        for name, t in (("vm", VM), ("vms", VMS), ("vanc", VANC)):
            if not t.is_contiguous():
                raise RuntimeError(f"fused MLA decode requires a contiguous {name}")
    else:
        SEL = torch.zeros((1,), dtype=torch.int32, device=dev)
        LOGM = torch.zeros((1,), dtype=torch.float32, device=dev)
        VM = torch.zeros((1,), dtype=torch.uint8, device=dev)
        VMS = torch.zeros((1,), dtype=torch.float16, device=dev)
        VANC = torch.zeros((1,), dtype=torch.float32, device=dev)
        selb = 0

    BLOCK_R = _pow2_at_least(rep)
    if _EXP2[0]:
        scaling = scaling * _LOG2E
    gk = grid_group(D)
    sig = ("mla", int(ws), int(D), int(HA), int(BLOCK_R), bool(gated), int(B),
           int(Sfp).bit_length(), int(n_sel).bit_length(), int(vm_bits))

    def _launch(rung):
        target_keys, num_stages, num_warps = rung
        BLOCK_NW, BLOCK_T = window_tiling(ws, target_keys)
        BLOCK_W = _pow2_at_least(min(W_phys, target_keys), floor=16)
        _mla_two_tier_decode_kernel[(B,)](
            q, k_fp, KC, KS, KSS, KZ, KZS, COS, SIN, SEL, LOGM, VM, VMS, VANC,
            out, wsum, wmax,
            scaling,
            H_q, n_active, n_sel, Sfp, rep, num_sink, n_body_win, W_phys,
            q.stride(0), q.stride(1), q.stride(2),
            k_fp.stride(0), k_fp.stride(2), k_fp.stride(3),
            selb,
            HEAD_DIM=D, HALF=half, HA=HA, HB=HB, WS=ws,
            BLOCK_R=BLOCK_R, BLOCK_NW=BLOCK_NW, BLOCK_T=BLOCK_T, BLOCK_W=BLOCK_W,
            PACK_K=max(ws // 4, 1), GROUP_K=gk,
            GATED=gated, LOG2E=_LOG2E, VM_BITS=vm_bits,
            num_stages=num_stages, num_warps=num_warps,
        )

    _search_rungs(
        _FIT_CHOICE, _FIT_TIMINGS, _FIT_ANNOUNCED, _FIT_SEEN, sig, _FIT_LADDER,
        _launch, "fused MLA decode tiling",
        key=lambda r: (window_tiling(ws, r[0]),
                       _pow2_at_least(min(W_phys, r[0]), floor=16), r[1], r[2]),
    )
    return out, wsum
