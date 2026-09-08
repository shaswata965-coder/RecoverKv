"""CPU tests for the fused two-tier decode path (modules.windowed_cache.decode_kernel).

These pin the *math* of the PyTorch reference (the Triton kernel's oracle) and the
Triton-or-error contract. The Triton kernel itself has no CPU test by construction
— it requires CUDA — and must be validated on a GPU against
``two_tier_decode_reference`` (see the note at the bottom), exactly as the prefill
score kernel is validated against ``token_scores_from_lse``.
"""

from __future__ import annotations

import torch
import pytest

from modules.windowed_cache.decode_kernel import (
    assert_decode_kernel_available,
    describe_decode_backend,
    fused_decode_enabled,
    fused_two_tier_decode,
    two_tier_decode_reference,
)
from modules.windowed_cache.score_kernel import token_scores_decode


def _naive_decode(q, k, v, scaling, rep):
    """Independent ground truth: repeat_kv + explicit softmax attention."""
    kk = k.repeat_interleave(rep, dim=1)                 # [B, H_q, S, D]
    vv = v.repeat_interleave(rep, dim=1)
    logits = torch.einsum("bhd,bhsd->bhs", q, kk).float() * scaling
    p = torch.softmax(logits, dim=-1)                    # [B, H_q, S]
    out = torch.einsum("bhs,bhsd->bhd", p.to(vv.dtype), vv)
    return out, p


@pytest.mark.parametrize("B,H_kv,rep,S,D", [(2, 2, 2, 10, 8), (1, 4, 1, 7, 16), (3, 2, 4, 12, 8)])
def test_reference_matches_naive_sdpa(B, H_kv, rep, S, D):
    """The reference's attention output AND per-key score equal naive SDPA.

    MHA (rep==1) and GQA (rep>1). The score is the softmax weight vector, which is
    exactly the H2O per-key received attention for a single decode query.
    """
    torch.manual_seed(0)
    H_q = H_kv * rep
    q = torch.randn(B, H_q, D)
    k = torch.randn(B, H_kv, S, D)
    v = torch.randn(B, H_kv, S, D)
    scaling = D ** -0.5

    out, scores = two_tier_decode_reference(q, k, v, scaling)
    out_ref, p_ref = _naive_decode(q, k, v, scaling, rep)

    assert torch.allclose(out, out_ref, atol=1e-5), "attention output"
    assert torch.allclose(scores, p_ref.to(scores.dtype), atol=1e-5), "per-key score"


def test_reference_score_matches_score_path():
    """The fused per-key score == the decode score path (token_scores_decode).

    Both are ``softmax(scale · q·kᵀ)`` for the single decode query, so the fused
    kernel emits exactly what the standalone score path would have computed — this
    is what lets it replace the separate score pass with no change to eviction.
    """
    torch.manual_seed(1)
    B, H_kv, rep, S, D = 2, 2, 3, 9, 8
    H_q = H_kv * rep
    q = torch.randn(B, H_q, D)
    k = torch.randn(B, H_kv, S, D)
    v = torch.randn(B, H_kv, S, D)
    scaling = D ** -0.5

    _, scores = two_tier_decode_reference(q, k, v, scaling)
    # token_scores_decode wants a query axis: [B, H_q, 1, D].
    ts = token_scores_decode(q.unsqueeze(2), k, scaling, softmax_dtype=torch.float32)
    assert torch.allclose(scores.float(), ts.float(), atol=1e-5)


def test_dispatcher_is_triton_or_error_on_cpu():
    """fused_two_tier_decode RAISES on CPU — no PyTorch decode fallback in prod."""
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    with pytest.raises(RuntimeError, match="requires the Triton kernel"):
        fused_two_tier_decode(q, k, v, None, 8 ** -0.5)   # qtier=None (empty Q)


def test_fused_decode_enabled_default_and_off(monkeypatch):
    """Fused decode is ON by default and honours STICKYKV_FUSED_DECODE=0."""
    monkeypatch.delenv("STICKYKV_FUSED_DECODE", raising=False)
    assert fused_decode_enabled() is True
    monkeypatch.setenv("STICKYKV_FUSED_DECODE", "0")
    assert fused_decode_enabled() is False


def test_gate_raises_without_triton():
    """The flash-backend gate refuses to run without a launchable decode kernel."""
    with pytest.raises(RuntimeError, match="fused two-tier decode Triton kernel"):
        assert_decode_kernel_available(True)   # CPU box: triton absent
    assert "WILL ERROR" in describe_decode_backend(True)


def test_gate_is_noop_when_fused_disabled(monkeypatch):
    """With fused decode disabled, the gate does not raise (materialize fallback)."""
    monkeypatch.setenv("STICKYKV_FUSED_DECODE", "0")
    assert_decode_kernel_available(True)        # must not raise
    assert "disabled" in describe_decode_backend(True)


# NOTE: the Triton kernel (_two_tier_decode_kernel) has no CPU test by construction
# — it requires CUDA. On a GPU box, validate it against the reference by building
# the effective K/V the *materialize* path would (so the int2 dequant+RoPE inside
# the kernel is checked against the already-tested primitives):
#
#     from modules.quant.effective import materialize_effective_kv
#     k_eff, v_eff, _ = materialize_effective_kv(k_fp, v_fp, fp_pos, store,
#                                                num_sink, ws, rope)
#     out_ref, sc_ref = two_tier_decode_reference(q, k_eff, v_eff, scaling)
#     # qtier = the gathered int2 fields + rope halves the cache builds in update()
#     out, sc = fused_two_tier_decode(q, k_fp, v_fp, qtier, scaling)
#     assert torch.allclose(out, out_ref, atol=1e-2, rtol=1e-2)
#     assert torch.allclose(sc,  sc_ref,  atol=1e-2, rtol=1e-2)  # scores are [sink‖body‖Q] order


# ---------------------------------------------------------------------------
# Shared-memory fit ladder (§5.3 widened the Q tile; A100 smem is 163 KB)
# ---------------------------------------------------------------------------


def test_out_of_resources_is_detected_by_name_and_by_message():
    """Triton has moved this exception between modules across versions, so the
    detector matches the type name OR the message, and unwraps causes."""
    from modules.windowed_cache.decode_kernel import _is_out_of_resources

    class OutOfResources(Exception):
        pass

    assert _is_out_of_resources(OutOfResources("anything"))
    assert _is_out_of_resources(RuntimeError(
        "out of resource: shared memory, Required: 176128, Hardware limit: 166912"))
    assert _is_out_of_resources(RuntimeError("out of resource: registers"))
    # wrapped
    inner = OutOfResources("x")
    outer = RuntimeError("launch failed")
    outer.__cause__ = inner
    assert _is_out_of_resources(outer)


def test_unrelated_errors_are_not_swallowed_by_the_ladder():
    """A real bug must propagate, not be retried at a smaller tile until the
    ladder is exhausted and the true cause is buried."""
    from modules.windowed_cache.decode_kernel import _is_out_of_resources

    assert not _is_out_of_resources(RuntimeError("illegal memory access"))
    assert not _is_out_of_resources(ValueError("shape mismatch"))
    assert not _is_out_of_resources(RuntimeError("CUDA error: out of memory"))


def test_fit_ladder_is_ordered_fastest_first_and_terminates():
    from modules.windowed_cache.decode_kernel import _FIT_LADDER, window_tiling

    assert len(_FIT_LADDER) >= 2
    keys = [k for k, _ in _FIT_LADDER]
    assert keys == sorted(keys, reverse=True), "biggest tile must be tried first"
    # every rung yields a legal, shrinking tile at the shipped window size
    seen = []
    for target_keys, num_stages in _FIT_LADDER:
        nw, t = window_tiling(8, target_keys)
        assert nw >= 1 and t & (t - 1) == 0
        assert num_stages >= 1
        seen.append(nw)
    assert seen[0] >= seen[-1], "tiles must not grow as the ladder steps down"
    assert seen[-1] >= 1, "the last rung must still be launchable"


@pytest.mark.parametrize("ws", [1, 4, 8, 12, 32, 128])
def test_every_ladder_rung_is_a_legal_tiling(ws):
    """A rung that produced a zero-width tile would raise instead of stepping to
    the next one, turning a fit problem into a crash."""
    from modules.windowed_cache.decode_kernel import _FIT_LADDER, window_tiling

    for target_keys, _ in _FIT_LADDER:
        nw, t = window_tiling(ws, target_keys)
        assert nw >= 1, f"ws={ws} target={target_keys} produced {nw} windows/tile"
        assert nw * ws <= t
        assert t & (t - 1) == 0
