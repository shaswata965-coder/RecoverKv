"""CPU parity tests for the Design-B score path (modules.windowed_cache.score_kernel).

These pin the *math*, which is what must hold before the Triton kernel (the same
math on GPU) can be trusted. Everything here runs on CPU — no CUDA, no triton, no
transformers — so it is the one layer of this change that is validated on the
dev box. The Triton kernel itself must be validated on a GPU against
``token_scores_from_lse`` (its exact oracle); see the note at the bottom.

What is asserted:
  1. ``token_scores_torch`` reproduces the historic inline hook loop exactly
     (independent triu-mask reference), for MHA and GQA, and is invariant to the
     query-chunk size.
  2. ``exp(S − L)`` (Design B) equals ``softmax(S)`` — i.e. ``token_scores_from_lse``
     == ``token_scores_torch`` — for prefill (offset 0), past+current (offset > 0),
     and decode (T == 1).
  3. ``compute_lse`` equals a full-matrix ``logsumexp``.
  4. The dispatcher routes by pass: prefill (T > 1) is Triton-only and RAISES on
     CPU (no silent PyTorch fallback); decode (T == 1) matches ``token_scores_torch``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from modules.windowed_cache.score_kernel import (
    compute_lse,
    compute_token_scores,
    token_scores_from_lse,
    token_scores_torch,
)


# ---------------------------------------------------------------------------
# Independent ground truth: the ORIGINAL inline hook loop, re-implemented here
# with the original triu mask (score_kernel uses a different mask expression, so
# agreement pins that the two masks are equivalent).
# ---------------------------------------------------------------------------


def _oldhook_reference(q, k, scaling, softmax_dtype, chunk):
    """Verbatim re-implementation of the pre-refactor hooks.py score loop."""
    B, H_q, T, D = q.shape
    H_kv, S = k.shape[1], k.shape[2]
    rep = H_q // H_kv
    q5 = q.reshape(B, H_kv, rep, T, D)
    k_t = k.transpose(-2, -1).unsqueeze(2)  # [B,H_kv,1,D,S]
    token_scores = torch.zeros(B, H_kv, rep, S, dtype=q.dtype)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        aw = torch.matmul(q5[:, :, :, start:end, :], k_t) * scaling
        if T > 1:
            blk = end - start
            causal = torch.triu(
                torch.ones(blk, S, dtype=torch.bool), diagonal=S - T + start + 1
            )
            aw = aw.masked_fill(causal, float("-inf"))
        aw = F.softmax(aw, dim=-1, dtype=softmax_dtype).to(q.dtype)
        token_scores += aw.sum(dim=-2)
    return token_scores.reshape(B, H_q, S)


def _full_matrix_reference(q, k, scaling):
    """Dead-simple full-materialisation truth (fp32) for the offset/causal logic."""
    B, H_q, T, D = q.shape
    H_kv, S = k.shape[1], k.shape[2]
    rep = H_q // H_kv
    k_exp = k.repeat_interleave(rep, dim=1)          # [B, H_q, S, D]
    logits = torch.matmul(q, k_exp.transpose(-2, -1)).float() * scaling  # [B,H_q,T,S]
    if T > 1:
        offset = S - T
        rows = torch.arange(T)[:, None]
        cols = torch.arange(S)[None, :]
        mask = cols > (offset + rows)                # future keys
        logits = logits.masked_fill(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)            # [B, H_q, T, S]
    return probs.sum(dim=-2)                          # [B, H_q, S]


# ---------------------------------------------------------------------------
# Shapes: (name, B, H_q, H_kv, T, S). S == T is pure prefill (offset 0);
# S > T mimics scoring current queries against an effective-K with past keys.
# ---------------------------------------------------------------------------

SHAPES = [
    ("mha_prefill",    2, 4, 4, 16, 16),
    ("gqa_prefill",    2, 8, 2, 24, 24),
    ("gqa_past",       1, 8, 2, 12, 20),   # offset = 8
    ("mha_decode",     2, 4, 4, 1, 30),    # T == 1, no mask
]


def _make(B, H_q, H_kv, T, S, D=8, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H_q, T, D, generator=g, dtype=dtype)
    k = torch.randn(B, H_kv, S, D, generator=g, dtype=dtype)
    return q, k


@pytest.mark.parametrize("name,B,H_q,H_kv,T,S", SHAPES)
def test_torch_matches_old_hook_loop(name, B, H_q, H_kv, T, S):
    """token_scores_torch == the original inline loop (mask equivalence)."""
    q, k = _make(B, H_q, H_kv, T, S)
    scaling = q.shape[-1] ** -0.5
    got = token_scores_torch(q, k, scaling, chunk=1024, softmax_dtype=torch.float32)
    ref = _oldhook_reference(q, k, scaling, torch.float32, chunk=1024)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), name


@pytest.mark.parametrize("name,B,H_q,H_kv,T,S", SHAPES)
def test_chunk_invariance(name, B, H_q, H_kv, T, S):
    """The query-chunk size must not change the result (it is a pure sum)."""
    q, k = _make(B, H_q, H_kv, T, S, seed=1)
    scaling = q.shape[-1] ** -0.5
    whole = token_scores_torch(q, k, scaling, chunk=10_000, softmax_dtype=torch.float32)
    tiny = token_scores_torch(q, k, scaling, chunk=3, softmax_dtype=torch.float32)
    assert torch.allclose(whole, tiny, atol=1e-5, rtol=1e-4), name


@pytest.mark.parametrize("name,B,H_q,H_kv,T,S", SHAPES)
def test_design_b_equals_softmax(name, B, H_q, H_kv, T, S):
    """exp(S − L) (Design B) == softmax(S). This is the kernel's math."""
    q, k = _make(B, H_q, H_kv, T, S, seed=2)
    scaling = q.shape[-1] ** -0.5
    lse = compute_lse(q, k, scaling, chunk=1024)
    from_lse = token_scores_from_lse(q, k, scaling, lse, chunk=1024)
    softmax_way = token_scores_torch(q, k, scaling, chunk=1024, softmax_dtype=torch.float32)
    assert torch.allclose(from_lse, softmax_way, atol=1e-5, rtol=1e-4), name


@pytest.mark.parametrize("name,B,H_q,H_kv,T,S", SHAPES)
def test_full_matrix_truth(name, B, H_q, H_kv, T, S):
    """Both paths match a dead-simple full-materialisation reference."""
    q, k = _make(B, H_q, H_kv, T, S, seed=3)
    scaling = q.shape[-1] ** -0.5
    truth = _full_matrix_reference(q, k, scaling)
    lse = compute_lse(q, k, scaling)
    assert torch.allclose(token_scores_torch(q, k, scaling, softmax_dtype=torch.float32),
                          truth, atol=1e-5, rtol=1e-4), f"{name} torch"
    assert torch.allclose(token_scores_from_lse(q, k, scaling, lse),
                          truth, atol=1e-5, rtol=1e-4), f"{name} lse"


@pytest.mark.parametrize("name,B,H_q,H_kv,T,S", SHAPES)
def test_compute_lse_matches_logsumexp(name, B, H_q, H_kv, T, S):
    """compute_lse == logsumexp over the (masked) full logit rows."""
    q, k = _make(B, H_q, H_kv, T, S, seed=4)
    scaling = q.shape[-1] ** -0.5
    got = compute_lse(q, k, scaling)                 # [B, H_q, T]
    rep = H_q // H_kv
    k_exp = k.repeat_interleave(rep, dim=1)
    logits = torch.matmul(q, k_exp.transpose(-2, -1)).float() * scaling
    if T > 1:
        offset = S - T
        rows = torch.arange(T)[:, None]
        cols = torch.arange(S)[None, :]
        logits = logits.masked_fill(cols > (offset + rows), float("-inf"))
    ref = torch.logsumexp(logits, dim=-1)            # [B, H_q, T]
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), name


def test_dispatcher_prefill_requires_triton_on_cpu():
    """Prefill (T > 1) is Triton-only: on CPU it RAISES, never silently falls back.

    This is the contract that stops a GPU box with triton missing from silently
    running the [B, H_q, chunk, S] PyTorch reconstruction that OOMs batched
    long-context prefills. CPU has no CUDA, so the dispatcher must refuse.
    """
    q, k = _make(2, 8, 2, 20, 20, seed=5)          # T = 20 > 1 → prefill
    scaling = q.shape[-1] ** -0.5
    with pytest.raises(RuntimeError, match="requires the Triton kernel"):
        compute_token_scores(
            q, k, scaling, softmax_dtype=torch.float32, out_dtype=torch.float32
        )


def test_dispatcher_decode_uses_torch_path():
    """Decode (T == 1) still uses the exact PyTorch reference (no kernel grid)."""
    q, k = _make(2, 8, 2, 1, 20, seed=5)           # T = 1 → decode
    scaling = q.shape[-1] ** -0.5
    got = compute_token_scores(
        q, k, scaling, softmax_dtype=torch.float32, out_dtype=torch.float32
    )
    ref = token_scores_torch(q, k, scaling, softmax_dtype=torch.float32)
    assert torch.allclose(got, ref, atol=1e-6)


def test_token_scores_decode_matches_torch_at_T1():
    """The T==1 fast path is byte-identical to the general reference at T==1.

    Also for GQA (rep>1) and MHA (rep==1), so the size-1-query fold matches the
    accumulator path's head layout.
    """
    from modules.windowed_cache.score_kernel import token_scores_decode

    for B, H_q, H_kv in [(2, 8, 2), (1, 4, 4), (3, 16, 4)]:
        q, k = _make(B, H_q, H_kv, 1, 24, seed=7)      # T = 1 → decode
        scaling = q.shape[-1] ** -0.5
        fast = token_scores_decode(q, k, scaling, softmax_dtype=torch.float32)
        ref = token_scores_torch(q, k, scaling, softmax_dtype=torch.float32)
        assert torch.allclose(fast, ref, atol=1e-6), (B, H_q, H_kv)

    # It is a T==1 path only — a prefill shape must be rejected, not silently
    # mis-scored (the general path handles T>1).
    q2, k2 = _make(1, 4, 2, 5, 20, seed=7)
    with pytest.raises(ValueError, match="T==1"):
        token_scores_decode(q2, k2, q2.shape[-1] ** -0.5)


# NOTE: the Triton kernel (_token_scores_triton) has no CPU test by construction —
# it requires CUDA. On a GPU box, assert it against token_scores_from_lse:
#
#     lse = compute_lse(q, k, scaling)                       # or from the forward
#     ref = token_scores_from_lse(q, k, scaling, lse)
#     got = _token_scores_triton(q, k, scaling, lse)
#     assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3)  # fp32, matmul order
