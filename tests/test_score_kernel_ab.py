"""A/B + per-step localization for the Design-B Triton score kernel.

Purpose
-------
Phase-1 inspection proved the Triton kernel (``score_kernel._score_kernel``) is
*logically* identical to its PyTorch oracle (``token_scores_from_lse``): every
stage — logit/scale, GQA head map, LSE index, exp, causal mask, colsum, block
skip, store — maps 1:1. So any real divergence (the ~21-pt LongBench collapse)
must be empirical: a Triton codegen / ``tl.dot`` precision effect at production
shapes, or an integration mismatch (wrong ``L``). This file finds *which* by
running the kernel and comparing it to the oracle **one stage at a time**.

Layout
------
1. Section A (CPU-ok) — re-pins the PyTorch oracle chain so the A/B baseline is
   trustworthy: full-matrix == ``token_scores_torch`` == ``token_scores_from_lse``,
   and ``compute_lse`` == ``logsumexp``. Runs on the dev box (no CUDA/triton).
2. Section B (GPU) — **the A/B headline**: production-shaped
   ``_token_scores_triton`` vs ``token_scores_from_lse``. Pass/fail gate.
3. Section C (GPU) — **step localization** via an *instrumented mirror* of the
   production kernel that additionally dumps its internal ``S`` (logits) and
   ``P`` (softmax) tiles. We then assert, in order:
     C0  mirror OUT == production OUT      (the mirror is faithful)
     C1  mirror S    == torch scale·q·kᵀ   (dot / scale / GQA map / strides)
     C2  mirror P    == torch exp(S−L)     (exp / LSE index)
     C3  mirror P    == 0 on future keys   (causal geometry)
     C4  mirror OUT  == P.sum(queries)     (reduction + per-block store offset)
     C5  per-key-block error profile       (localizes a bug to a ``pid_n``)
   The first assertion that fails names the broken stage.
4. Section D (GPU) — sweeps that make single-tile bugs impossible to hide:
   tiny block sizes, offset>0 (past keys), a GQA-map stress shape, dtype.
5. Section E (CPU-ok) — documents the LSE-reuse failure mode: the score is only
   correct when ``L``'s scale matches the score's scale (the flash_lse risk).

Run it
------
    pytest tests/test_score_kernel_ab.py -v
    python  tests/test_score_kernel_ab.py      # prints a localization report

On a CPU box, Sections B–D skip and A/E still run.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Allow `python tests/test_score_kernel_ab.py` (bare script) to find `modules/`;
# under pytest the rootdir is already importable so this is a no-op.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.windowed_cache import flash_lse
from modules.windowed_cache.score_kernel import (
    _HAS_TRITON,
    _token_scores_triton,
    compute_lse,
    token_scores_from_lse,
    token_scores_torch,
)

CUDA = torch.cuda.is_available()
GPU = _HAS_TRITON and CUDA
gpu_only = pytest.mark.skipif(not GPU, reason="needs CUDA + triton")

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _HAS_FLASH = True
except Exception:
    _HAS_FLASH = False

flash_only = pytest.mark.skipif(
    not (GPU and _HAS_FLASH), reason="needs CUDA + triton + flash_attn"
)

if _HAS_TRITON:
    import triton
    import triton.language as tl


# ===========================================================================
# Shared PyTorch stage references (fp32) — the ground truth every kernel tile
# is measured against. Kept dead-simple and dense (no tiling) on purpose.
# ===========================================================================


def _future_mask(T: int, S: int, device) -> torch.Tensor:
    """[T, S] bool, True where key n is in query m's future (n > offset + m)."""
    offset = S - T
    m = torch.arange(T, device=device)[:, None]
    n = torch.arange(S, device=device)[None, :]
    return n > (offset + m)


def torch_stages(q, k, scaling, lse):
    """Dense fp32 references for every internal quantity of the kernel.

    Returns ``(S_full, P_full, future, out)`` where
      * ``S_full`` = scaling · q·kᵀ                       [B, H_q, T, S]
      * ``P_full`` = exp(S − L), zeroed on future keys    [B, H_q, T, S]
      * ``future`` = the causal mask                       [T, S]
      * ``out``    = Σ_queries P_full                       [B, H_q, S]
    ``lse`` is the *same* [B, H_q, T] fp32 tensor handed to the kernel, so the
    only source of C2/C4 disagreement is the kernel's own arithmetic.
    """
    B, H_q, T, D = q.shape
    H_kv, S = k.shape[1], k.shape[2]
    rep = H_q // H_kv
    k_exp = k.repeat_interleave(rep, dim=1).float()          # [B, H_q, S, D]
    S_full = torch.matmul(q.float(), k_exp.transpose(-2, -1)) * scaling
    future = _future_mask(T, S, q.device)
    P_full = torch.exp(S_full - lse.float()[..., None])
    P_full = P_full.masked_fill(future[None, None], 0.0)
    out = P_full.sum(dim=-2)                                  # [B, H_q, S]
    return S_full, P_full, future, out


def _make(B, H_q, H_kv, T, S, D, *, seed=0, dtype=torch.float32, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H_q, T, D, generator=g, dtype=torch.float32)
    k = torch.randn(B, H_kv, S, D, generator=g, dtype=torch.float32)
    return q.to(device=device, dtype=dtype), k.to(device=device, dtype=dtype)


# ===========================================================================
# Instrumented mirror of the production kernel (GPU only).
#
# This is a *faithful copy* of score_kernel._score_kernel with two extra stores
# (S and P tiles). C0 asserts its OUT equals the production kernel's OUT, which
# proves the mirror did not drift — so its S/P dumps are exactly what production
# computes internally, and C1–C4 localize any production bug.
# ===========================================================================

if _HAS_TRITON:

    @triton.jit
    def _score_kernel_debug(
        Q, K, LSE, OUT, SDUMP, PDUMP,
        scale,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_lb, stride_lh, stride_lt,
        stride_ob, stride_oh, stride_os,
        stride_db, stride_dh, stride_dt, stride_ds,   # SDUMP & PDUMP: [B,H_q,T,S]
        T, S, OFFSET, H_q, num_groups,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H_q
        h = pid_bh % H_q
        kv = h // num_groups

        ks = pid_n * BLOCK_N
        offs_n = ks + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        n_mask = offs_n < S

        k_ptrs = (K + b * stride_kb + kv * stride_kh
                  + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        if IS_CAUSAL:
            m_lo = ks - OFFSET
            m_lo = tl.maximum(m_lo, 0)
            m_start = (m_lo // BLOCK_M) * BLOCK_M
        else:
            m_start = 0

        for m in range(m_start, T, BLOCK_M):
            offs_m = m + tl.arange(0, BLOCK_M)
            m_mask = offs_m < T
            q_ptrs = (Q + b * stride_qb + h * stride_qh
                      + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd)
            q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

            s = tl.dot(q_tile, tl.trans(k_tile)) * scale

            lse_i = tl.load(LSE + b * stride_lb + h * stride_lh + offs_m * stride_lt,
                            mask=m_mask, other=0.0)
            p = tl.exp(s - lse_i[:, None])

            valid = m_mask[:, None] & n_mask[None, :]
            if IS_CAUSAL:
                valid = valid & ((offs_n[None, :] - offs_m[:, None]) <= OFFSET)
            p = tl.where(valid, p, 0.0)

            acc += tl.sum(p, axis=0)

            # --- instrumentation: dump this tile's S and P into the full grids.
            d_ptrs = (b * stride_db + h * stride_dh
                      + offs_m[:, None] * stride_dt + offs_n[None, :] * stride_ds)
            dump_mask = m_mask[:, None] & n_mask[None, :]
            tl.store(SDUMP + d_ptrs, s, mask=dump_mask)
            tl.store(PDUMP + d_ptrs, p, mask=dump_mask)

        out_ptrs = OUT + b * stride_ob + h * stride_oh + offs_n * stride_os
        tl.store(out_ptrs, acc, mask=n_mask)

    def run_debug_kernel(q, k, scaling, lse, *, block_m=64, block_n=64):
        """Launch the mirror; return (out, S_full, P_full) as fp32 tensors.

        S_full is NaN-initialised: positions never visited (query blocks skipped
        by the causal fast-path) stay NaN and are excluded from comparison. P_full
        is zero-initialised: a skipped (m,n) is all-future, whose true P is 0.
        """
        B, H_q, T, D = q.shape
        H_kv, S = k.shape[1], k.shape[2]
        num_groups = H_q // H_kv
        lse = lse.contiguous().float()

        out = torch.empty(B, H_q, S, device=q.device, dtype=torch.float32)
        sdump = torch.full((B, H_q, T, S), float("nan"), device=q.device, dtype=torch.float32)
        pdump = torch.zeros((B, H_q, T, S), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(S, block_n), B * H_q)
        _score_kernel_debug[grid](
            q, k, lse, out, sdump, pdump,
            scaling,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            sdump.stride(0), sdump.stride(1), sdump.stride(2), sdump.stride(3),
            T, S, S - T, H_q, num_groups,
            HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n, IS_CAUSAL=(T > 1),
        )
        return out, sdump, pdump


# ===========================================================================
# Section A — PyTorch oracle chain (CPU-ok). Trust the baseline before A/B.
# ===========================================================================

# (name, B, H_q, H_kv, T, S, D). Includes offset>0 and a GQA case.
REF_SHAPES = [
    ("mha_prefill", 2, 4, 4, 48, 48, 16),
    ("gqa_prefill", 2, 8, 2, 40, 40, 16),
    ("gqa_past",    1, 8, 2, 24, 40, 16),   # offset = 16
]


@pytest.mark.parametrize("name,B,H_q,H_kv,T,S,D", REF_SHAPES)
def test_A_oracle_chain(name, B, H_q, H_kv, T, S, D):
    q, k = _make(B, H_q, H_kv, T, S, D, seed=1)
    scaling = D ** -0.5
    lse = compute_lse(q, k, scaling)

    _, _, _, dense = torch_stages(q, k, scaling, lse)
    from_lse = token_scores_from_lse(q, k, scaling, lse)
    softmax_way = token_scores_torch(q, k, scaling, softmax_dtype=torch.float32)

    torch.testing.assert_close(dense, from_lse, atol=1e-4, rtol=1e-4, msg=f"{name}: dense vs from_lse")
    torch.testing.assert_close(dense, softmax_way, atol=1e-4, rtol=1e-4, msg=f"{name}: dense vs softmax")


@pytest.mark.parametrize("name,B,H_q,H_kv,T,S,D", REF_SHAPES)
def test_A_compute_lse(name, B, H_q, H_kv, T, S, D):
    q, k = _make(B, H_q, H_kv, T, S, D, seed=2)
    scaling = D ** -0.5
    got = compute_lse(q, k, scaling)
    rep = H_q // H_kv
    k_exp = k.repeat_interleave(rep, dim=1)
    logits = torch.matmul(q, k_exp.transpose(-2, -1)).float() * scaling
    logits = logits.masked_fill(_future_mask(T, S, q.device)[None, None], float("-inf"))
    ref = torch.logsumexp(logits, dim=-1)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


# ===========================================================================
# Section B — A/B HEADLINE (GPU). Production kernel vs exact oracle.
# ===========================================================================

# Production-representative: GQA 32/8, head_dim 128, multi-tile in both axes.
PROD_SHAPES = [
    ("prefill_1024_fp32", 1, 32, 8, 1024, 1024, 128, torch.float32),
    ("prefill_1024_bf16", 1, 32, 8, 1024, 1024, 128, torch.bfloat16),
    ("past_512_1024_fp32", 1, 32, 8, 512, 1024, 128, torch.float32),
]


@gpu_only
@pytest.mark.parametrize("name,B,H_q,H_kv,T,S,D,dt", PROD_SHAPES)
def test_B_kernel_vs_oracle(name, B, H_q, H_kv, T, S, D, dt):
    q, k = _make(B, H_q, H_kv, T, S, D, seed=7, dtype=dt, device="cuda")
    scaling = D ** -0.5
    lse = compute_lse(q, k, scaling)                       # exact fp32 L
    ref = token_scores_from_lse(q, k, scaling, lse)        # oracle [B,H_q,S] fp32
    got = _token_scores_triton(q, k, scaling, lse).float()
    # bf16 dot is coarser; tolerance scales with dtype. A logic bug is O(1)+.
    atol, rtol = (5e-2, 5e-2) if dt == torch.float32 else (2e-1, 1e-1)
    torch.testing.assert_close(got, ref, atol=atol, rtol=rtol, msg=name)


# ===========================================================================
# Section C — STEP LOCALIZATION (GPU). Runs the mirror and checks each stage.
# ===========================================================================


def _per_block_maxerr(err_by_key: torch.Tensor, block_n: int):
    """Max |error| per key-block → localizes a bug to a specific ``pid_n``."""
    S = err_by_key.shape[-1]
    flat = err_by_key.reshape(-1, S).max(dim=0).values  # worst over B,H_q
    blocks = []
    for i in range(0, S, block_n):
        blocks.append(float(flat[i:i + block_n].max()))
    return blocks


@gpu_only
def test_C_step_localization():
    # Small enough to dump full [B,H_q,T,S] P/S, large enough for a 3x3 tile grid.
    B, H_q, H_kv, T, S, D = 2, 8, 2, 192, 192, 128
    block_m = block_n = 64
    q, k = _make(B, H_q, H_kv, T, S, D, seed=11, dtype=torch.float32, device="cuda")
    scaling = D ** -0.5
    lse = compute_lse(q, k, scaling)

    S_ref, P_ref, future, out_ref = torch_stages(q, k, scaling, lse)
    out_dbg, S_dbg, P_dbg = run_debug_kernel(q, k, scaling, lse, block_m=block_m, block_n=block_n)
    out_prod = _token_scores_triton(q, k, scaling, lse, block_m=block_m, block_n=block_n).float()

    inrange = ~future  # [T,S]; compare S only on non-future (always-visited) cells
    inrange4 = inrange[None, None].expand_as(S_ref)

    # C0 — mirror is faithful to production (its dumps are production's internals).
    torch.testing.assert_close(out_dbg, out_prod, atol=1e-3, rtol=1e-3,
                               msg="C0 mirror OUT != production OUT")

    # C1 — logits: dot, scale, GQA head map, strides.
    torch.testing.assert_close(S_dbg[inrange4], S_ref[inrange4], atol=5e-2, rtol=5e-2,
                               msg="C1 kernel S (scale/dot/GQA) != torch")

    # C2 — softmax exp(S-L): exp + LSE indexing (compare on non-future cells).
    torch.testing.assert_close(P_dbg[inrange4], P_ref[inrange4], atol=5e-2, rtol=5e-2,
                               msg="C2 kernel P = exp(S-L) != torch")

    # C3 — causal geometry: P must be exactly 0 on future keys.
    assert torch.count_nonzero(P_dbg[future[None, None].expand_as(P_dbg)]) == 0, \
        "C3 kernel leaked attention onto future keys"

    # C4 — reduction + per-block store offset.
    torch.testing.assert_close(out_dbg, out_ref, atol=5e-2, rtol=5e-2,
                               msg="C4 colsum/store != torch")
    torch.testing.assert_close(out_prod, out_ref, atol=5e-2, rtol=5e-2,
                               msg="C4 production OUT != torch")

    # C5 — per-key-block error profile. A bug in pid_n handling shows as ~0
    # error in block 0 and large error for blocks >= 1.
    prof = _per_block_maxerr((out_prod - out_ref).abs(), block_n)
    assert max(prof) < 5e-2, f"C5 per-block maxerr {prof}"


# ===========================================================================
# Section D — sweeps that defeat single-tile blind spots (GPU).
# ===========================================================================


@gpu_only
@pytest.mark.parametrize("block_m,block_n", [(16, 16), (32, 64), (64, 32), (128, 64)])
def test_D_tile_size_invariance(block_m, block_n):
    """Result must not depend on tile size (it is a pure sum)."""
    B, H_q, H_kv, T, S, D = 1, 8, 2, 200, 200, 128
    q, k = _make(B, H_q, H_kv, T, S, D, seed=13, dtype=torch.float32, device="cuda")
    scaling = D ** -0.5
    lse = compute_lse(q, k, scaling)
    ref = token_scores_from_lse(q, k, scaling, lse)
    got = _token_scores_triton(q, k, scaling, lse, block_m=block_m, block_n=block_n).float()
    torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2, msg=f"blk=({block_m},{block_n})")


@gpu_only
def test_D_gqa_head_map_stress():
    """Each KV head gets a distinct constant K; a wrong h->kv map blows up.

    With K[:, kv] == kv (constant across S,D) and unit q, head h's logits are
    proportional to (h // rep). Any off-by-group mapping mis-scores whole heads.
    """
    B, H_q, H_kv, T, S, D = 1, 8, 4, 64, 64, 128   # rep = 2
    rep = H_q // H_kv
    q = torch.ones(B, H_q, T, D, device="cuda")
    k = torch.empty(B, H_kv, S, D, device="cuda")
    for kv in range(H_kv):
        k[:, kv] = float(kv + 1)
    scaling = D ** -0.5
    lse = compute_lse(q, k, scaling)
    ref = token_scores_from_lse(q, k, scaling, lse)
    got = _token_scores_triton(q, k, scaling, lse).float()
    torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2)
    # sanity: heads sharing a kv head must produce identical scores
    for kv in range(H_kv):
        h0 = kv * rep
        for r in range(1, rep):
            torch.testing.assert_close(got[:, h0], got[:, h0 + r], atol=1e-4, rtol=1e-4)


# ===========================================================================
# Section E — LSE-reuse failure mode (CPU-ok). The flash_lse scale risk.
# ===========================================================================


def test_E_lse_scale_must_match():
    """The score is correct ONLY if L's scale equals the score's scale.

    This is precisely the flash-lse reuse hazard: if flash's softmax_lse was
    produced with a softmax_scale != module.scaling, exp(scale·qk − L) is biased.
    We assert (a) matched scale reproduces the softmax score, and (b) a mismatched
    L visibly diverges — so a regression here is caught as a real error, not noise.
    """
    q, k = _make(1, 8, 2, 40, 40, 16, seed=17)
    scaling = 16 ** -0.5

    lse_ok = compute_lse(q, k, scaling)
    good = token_scores_from_lse(q, k, scaling, lse_ok)
    ref = token_scores_torch(q, k, scaling, softmax_dtype=torch.float32)
    torch.testing.assert_close(good, ref, atol=1e-4, rtol=1e-4)

    lse_bad = compute_lse(q, k, scaling * 2.0)   # L from a different scale
    bad = token_scores_from_lse(q, k, scaling, lse_bad)
    assert not torch.allclose(bad, ref, atol=1e-2), \
        "scale-mismatched L did NOT diverge — the guard test is ineffective"


# ===========================================================================
# Section F — flash softmax_lse parity (GPU + flash_attn). THE remaining link.
#
# The kernel is proven correct GIVEN a correct L. The one thing the suite has
# not exercised end-to-end is L sourced from flash's own forward, which is what
# STICKYKV_SCORE_LSE_FROM_FORWARD=1 turns on. If flash's softmax_lse disagrees
# with compute_lse — different scale, log base, causal alignment, GQA order, or
# a padded seqlen — the kernel is fed a bad L and scores corrupt while every
# test above stays green. Given the kernel is now exonerated, this is the prime
# suspect for the 2wikimqa collapse, so it gets its own direct test.
# ===========================================================================

# flash pads softmax_lse's seqlen up to a multiple of 128 on some builds. T is
# chosen a multiple of 128 so the [B,H,T] shape check passes and reuse actually
# engages — the exact case where a value mismatch would corrupt (a padded shape
# is instead rejected by the hook and safely recomputed).
_FLASH_SHAPE = dict(B=1, T=512, H_q=32, H_kv=8, D=128)


def _flash_qkv():
    p = _FLASH_SHAPE
    dt = torch.bfloat16
    q = torch.randn(p["B"], p["T"], p["H_q"], p["D"], device="cuda", dtype=dt)
    k = torch.randn(p["B"], p["T"], p["H_kv"], p["D"], device="cuda", dtype=dt)
    v = torch.randn(p["B"], p["T"], p["H_kv"], p["D"], device="cuda", dtype=dt)
    return q, k, v


@flash_only
def test_F_raw_flash_lse_convention():
    """flash's own softmax_lse == compute_lse (scale / log-base / causal / GQA).

    Calls flash_attn_func directly (bypassing the monkeypatch) to isolate the
    CONVENTION question from the capture plumbing. flash layout is [B,T,H,D];
    ours is [B,H,T,D].
    """
    p = _FLASH_SHAPE
    scaling = p["D"] ** -0.5
    q_f, k_f, v_f = _flash_qkv()
    ours = compute_lse(q_f.transpose(1, 2), k_f.transpose(1, 2), scaling)  # [B,H_q,T]

    out = _flash_attn_func(q_f, k_f, v_f, causal=True, softmax_scale=scaling,
                           return_attn_probs=True)
    assert isinstance(out, tuple) and len(out) >= 2, \
        "this flash build did not return softmax_lse with return_attn_probs"
    lse = out[1]
    # A padded/other shape would be REJECTED by the hook (safe recompute), so
    # treat that as 'reuse never engages', not corruption.
    assert tuple(lse.shape) == (p["B"], p["H_q"], p["T"]), (
        f"flash softmax_lse shape {tuple(lse.shape)} != {(p['B'], p['H_q'], p['T'])}: "
        "the hook would reject this and recompute L (safe, but LSE-reuse is inert)."
    )
    torch.testing.assert_close(lse.float(), ours, atol=1e-2, rtol=1e-2,
                               msg="flash softmax_lse disagrees with compute_lse")


@flash_only
def test_F_captured_lse_matches_recompute():
    """End-to-end: wrapper-captured L == compute_lse for the same q,k.

    Exercises the exact path STICKYKV_SCORE_LSE_FROM_FORWARD turns on: the
    patched transformers symbol -> flash_lse.pop().
    """
    import transformers.modeling_flash_attention_utils as fa_utils

    p = _FLASH_SHAPE
    scaling = p["D"] ** -0.5
    q_f, k_f, v_f = _flash_qkv()
    ours = compute_lse(q_f.transpose(1, 2), k_f.transpose(1, 2), scaling)

    handle = flash_lse.enable()
    assert handle is not None, "flash_lse.enable() found no flash_attn_func to patch"
    try:
        flash_lse.clear()
        _ = fa_utils.flash_attn_func(q_f, k_f, v_f, causal=True, softmax_scale=scaling)
        cap = flash_lse.pop()
    finally:
        handle.restore()

    if cap is None:
        pytest.skip("capture returned None (flash rejected return_attn_probs or "
                    "padded the shape) — the hook safely recomputes L, no corruption")
    assert tuple(cap.shape) == (p["B"], p["H_q"], p["T"])
    torch.testing.assert_close(cap.float(), ours, atol=1e-2, rtol=1e-2,
                               msg="captured flash L disagrees with compute_lse")


# ===========================================================================
# Script entry — prints an ordered localization report on a GPU box.
# ===========================================================================


def _report():
    print(f"[env] torch={torch.__version__} cuda={CUDA} triton={_HAS_TRITON}")
    if not GPU:
        print("[skip] no CUDA+triton -- running CPU oracle self-check only.")
        q, k = _make(2, 8, 2, 40, 40, 16, seed=1)
        scaling = 16 ** -0.5
        lse = compute_lse(q, k, scaling)
        _, _, _, dense = torch_stages(q, k, scaling, lse)
        ref = token_scores_torch(q, k, scaling, softmax_dtype=torch.float32)
        print(f"[cpu] oracle chain max|d| = {float((dense - ref).abs().max()):.2e} (want ~0)")
        return

    B, H_q, H_kv, T, S, D = 2, 8, 2, 192, 192, 128
    bn = 64
    q, k = _make(B, H_q, H_kv, T, S, D, seed=11, dtype=torch.float32, device="cuda")
    scaling = D ** -0.5
    lse = compute_lse(q, k, scaling)

    S_ref, P_ref, future, out_ref = torch_stages(q, k, scaling, lse)
    out_dbg, S_dbg, P_dbg = run_debug_kernel(q, k, scaling, lse, block_m=bn, block_n=bn)
    out_prod = _token_scores_triton(q, k, scaling, lse, block_m=bn, block_n=bn).float()
    inrange4 = (~future)[None, None].expand_as(S_ref)

    stages = [
        ("C0 mirror==prod   ", float((out_dbg - out_prod).abs().max())),
        ("C1 S logits/GQA    ", float((S_dbg[inrange4] - S_ref[inrange4]).abs().max())),
        ("C2 P=exp(S-L)      ", float((P_dbg[inrange4] - P_ref[inrange4]).abs().max())),
        ("C3 causal leak     ", float(P_dbg[future[None, None].expand_as(P_dbg)].abs().max())),
        ("C4 colsum/store    ", float((out_prod - out_ref).abs().max())),
    ]
    print("\n stage                   max|d|      verdict")
    print(" " + "-" * 46)
    first_bad = None
    for name, err in stages:
        ok = err < 5e-2
        if not ok and first_bad is None:
            first_bad = name.strip()
        print(f" {name}    {err:.3e}   {'ok' if ok else 'FAIL <-- first broken stage' if first_bad == name.strip() else 'fail'}")
    prof = _per_block_maxerr((out_prod - out_ref).abs(), bn)
    print(f"\n per-key-block max|d|: {[f'{x:.2e}' for x in prof]}")
    print(" (block0 ok + later blocks large  =>  pid_n / store-offset bug)")
    print("\n VERDICT:", "all stages within tol" if first_bad is None
          else f"first failing stage = {first_bad}")


if __name__ == "__main__":
    _report()
