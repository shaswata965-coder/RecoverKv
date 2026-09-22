"""Qwen2.5 port — the numerics fixes that need torch.

Two fixes, both no-ops on the fp16 Llama/Mistral columns and both essential for
Qwen2.5:

* **int2 grid dtype follows the KV dtype** (``grid_dtype_for``). fp16 KV keeps the
  fp16 group scale, byte for byte; a bf16 cache stores it bf16 so a key past
  fp16's 65504 cannot cast the grid to inf and a whole window to nan.
* **YaRN a² RoPE inverse** (``rope_attention_scaling`` +
  ``unrotate_key_window``). The negated-sin inverse lands on a²·k when the module
  folds a scalar a into cos/sin; dividing a² out restores the pre-RoPE key. At
  a == 1 the branch is skipped, so Llama/Mistral are byte-identical.

CPU-only; skipped where torch is absent (e.g. the bare web container). Run in the
stickykv env.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Stage 5 — grid dtype
# ---------------------------------------------------------------------------


def test_grid_dtype_for():
    from modules.quant.quantizer import grid_dtype_for

    assert grid_dtype_for(torch.float16) == torch.float16
    # anything else takes bf16 (fp32's exponent range in 2 bytes)
    assert grid_dtype_for(torch.bfloat16) == torch.bfloat16
    assert grid_dtype_for(torch.float32) == torch.bfloat16


def test_fp16_grid_is_still_fp16():
    """fp16 KV keeps an fp16 group scale — the Llama/Mistral bit-identity path."""
    from modules.quant.quantizer import quantize_key_window

    torch.manual_seed(0)
    k = torch.randn(4, 8, 128, dtype=torch.float16)  # [H_kv, window, D]
    _, scale, zero = quantize_key_window(k)
    assert scale.s.dtype == torch.float16
    assert zero.s.dtype == torch.float16


def test_bf16_grid_survives_a_value_past_fp16_max():
    """A bf16 key past fp16's range must not cast the grid to inf/nan.

    |zero| here is ~1e7, so the zero-grid group scale (amax/127 ≈ 7.9e4) exceeds
    fp16's 65504. Under the old fp16 grid that became inf and the window
    dequantized to nan; the bf16 grid holds it.
    """
    from modules.quant.quantizer import (
        dequantize_key_window,
        quantize_key_window,
    )

    k = torch.randn(4, 8, 128, dtype=torch.bfloat16)
    # One massive-activation channel, varying across tokens so the group is NOT
    # degenerate — |zero| ~1e7 drives the zero-grid group scale (amax/127) past
    # fp16's 65504.
    k[0, :, 0] = torch.linspace(-1.0e7, 1.0e7, 8, dtype=torch.bfloat16)
    packed, scale, zero = quantize_key_window(k)

    assert scale.s.dtype == torch.bfloat16
    assert zero.s.dtype == torch.bfloat16
    assert torch.isfinite(scale.decode()).all()
    assert torch.isfinite(zero.decode()).all()

    k_hat = dequantize_key_window(packed, scale, zero, window=8, out_dtype=torch.float32)
    assert torch.isfinite(k_hat).all()


# ---------------------------------------------------------------------------
# Stage 4 — YaRN a² RoPE inverse
# ---------------------------------------------------------------------------


class _FakeRope(torch.nn.Module):
    """A rotary module that folds ``attention_scaling`` into cos/sin, like YaRN.

    Returns ``cos = a·cat(cosθ, cosθ)``, ``sin = a·cat(sinθ, sinθ)`` in HF's
    layout, so applying ``(cos, sin)`` is ``a·R(θ)`` and applying ``(cos, −sin)``
    is ``a·R(−θ)`` — the exact geometry ``unrotate_key_window`` inverts.
    """

    def __init__(self, a: float, head_dim: int = 8, actual_scale: float = None):
        super().__init__()
        self.attention_scaling = a
        self.head_dim = head_dim
        # What the forward ACTUALLY folds into cos/sin. Defaults to the reported
        # scalar (a faithful module); a test can set it different to model a
        # module that scales without exposing attention_scaling.
        self._actual_scale = a if actual_scale is None else actual_scale

    def forward(self, ref, pos):            # ref: [B,H,T,D]; pos: [B,T]
        half = self.head_dim // 2
        inv = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / half))
        theta = pos.float().unsqueeze(-1) * inv         # [B, T, half]
        c, s = torch.cos(theta), torch.sin(theta)
        cos = torch.cat([c, c], dim=-1) * self._actual_scale  # [B, T, D]
        sin = torch.cat([s, s], dim=-1) * self._actual_scale
        return cos.to(ref.dtype), sin.to(ref.dtype)


def test_rope_attention_scaling_reads_the_scalar():
    from modules.quant.effective import rope_attention_scaling

    assert rope_attention_scaling(_FakeRope(1.0)) == 1.0
    assert rope_attention_scaling(_FakeRope(1.13862)) == pytest.approx(1.13862)

    # Missing / non-finite / non-positive all fall back to 1.0 (legacy behaviour).
    class _Bare(torch.nn.Module):
        pass
    assert rope_attention_scaling(_Bare()) == 1.0
    assert rope_attention_scaling(_FakeRope(float("nan"))) == 1.0
    assert rope_attention_scaling(_FakeRope(float("inf"))) == 1.0
    assert rope_attention_scaling(_FakeRope(0.0)) == 1.0


def test_unrotate_is_byte_identical_at_a_equals_one():
    """a == 1 is a SKIPPED branch, not a divide-by-one — assert torch.equal."""
    from modules.quant import effective

    rope = _FakeRope(1.0)
    torch.manual_seed(1)
    k = torch.randn(2, 4, 8, 8, dtype=torch.float32)          # [B,H,T,D]
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)

    got = effective.unrotate_key_window(k, pos, rope)
    # Reference: exactly the pre-change expression (no division).
    cos, sin = effective._rope_cos_sin(rope, k, pos)
    ref = effective._apply_rotary_one(k, cos, -sin)
    assert torch.equal(got, ref)


def test_yarn_round_trip_recovers_the_pre_rope_key():
    """rotate then unrotate is identity ONLY because a² is divided out.

    Build a pre-RoPE key, apply a·R(θ) (the forward the model runs), then
    unrotate: without the a² division the result is a²·k₀ (1.296× at factor 4).
    """
    from modules.quant import effective

    a = 1.13862                                   # YaRN factor 4.0
    rope = _FakeRope(a)
    torch.manual_seed(2)
    k0 = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    pos = torch.arange(8).unsqueeze(0)

    cos, sin = effective._rope_cos_sin(rope, k0, pos)
    k_rope = effective._apply_rotary_one(k0, cos, sin)         # a·R(θ)·k0

    recovered = effective.unrotate_key_window(k_rope, pos, rope)
    assert torch.allclose(recovered, k0, atol=1e-4, rtol=1e-4)

    # And prove the correction is what did it: the uncorrected inverse is a²·k0.
    uncorrected = effective._apply_rotary_one(k_rope, cos, -sin)
    assert torch.allclose(uncorrected, (a * a) * k0, atol=1e-3, rtol=1e-3)


def test_rope_consistency_check_warns_on_hidden_scaling():
    """A module that scales cos/sin without exposing attention_scaling is loud."""
    from modules.quant import effective

    effective._warned_rope_scaling[0] = False   # re-arm the once-per-process guard

    # Reports attention_scaling=1.0 but the forward actually scales cos/sin by
    # 2.0 → cos²+sin² = 4 ≠ a² = 1, which is exactly the silent-corruption case.
    liar = _FakeRope(1.0, actual_scale=2.0)
    k = torch.randn(1, 2, 8, 8, dtype=torch.float32)
    pos = torch.arange(8).unsqueeze(0)

    with pytest.warns(RuntimeWarning):
        effective._rope_cos_sin(liar, k, pos)


# ---------------------------------------------------------------------------
# Parity with the clustered architecture — cache_position resolution
# ---------------------------------------------------------------------------


class _FakeModelConfig:
    num_attention_heads = 32
    num_key_value_heads = 8
    hidden_size = 4096
    head_dim = 128
    num_hidden_layers = 1


def _make_cache():
    from modules.windowed_cache.config import WindowedCacheConfig
    from modules.windowed_cache.cache import WindowedCache

    cfg = WindowedCacheConfig(
        window_size=1, num_sink_tokens=0, local_window_size=1, cache_budget=0.375,
    )
    return WindowedCache(
        config=cfg, prefill_len=8, model_config=_FakeModelConfig(),
        kv_dtype=torch.float32, rope_module=torch.nn.Identity(),
        num_layers=1, max_tokens=0,
    )


def test_resolve_cache_position_returns_passed_value():
    """When the attention module supplies cache_position, it is used verbatim.

    This is Qwen2's path (its three attention variants all pass cache_position),
    so the derive-branch never fires and positions are byte-identical to before.
    """
    cache = _make_cache()
    want = torch.arange(100, 104, dtype=torch.long)
    got = cache._resolve_cache_position({"cache_position": want}, n_new=4, device="cpu")
    assert torch.equal(got, want)


def test_resolve_cache_position_derives_from_token_count_when_absent():
    """When cache_position is absent, positions come from the monotonic count."""
    cache = _make_cache()
    cache._tokens_seen = 40          # absolute index after this step's advance
    with pytest.warns(RuntimeWarning):
        got = cache._resolve_cache_position({"sin": 1, "cos": 1}, n_new=8, device="cpu")
    assert torch.equal(got, torch.arange(32, 40, dtype=torch.long))


# ---------------------------------------------------------------------------
# GQA un-repeat — the Qwen2 fused-decode fix (rep = H_q / H_kv = 7)
# ---------------------------------------------------------------------------


def test_gqa_unrepeat_recovers_true_kv_heads():
    """repeat_kv then the fp-tier un-repeat (::rep) recovers the KV heads exactly.

    Qwen2FlashAttention2 calls repeat_kv (4 KV heads -> 28) BEFORE the flash call,
    so _run_fused receives a 28-head fp tier while the gate cards are keyed by the
    true 4 KV heads. The fix slices k_fp[:, ::rep]. This pins that slice against
    the REAL transformers repeat_kv, at Qwen2.5-7B's geometry (rep = 7).
    """
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    B, H_kv, S, D, rep = 1, 4, 6, 8, 7            # Qwen2.5-7B: 28 q / 4 kv
    kv = torch.randn(B, H_kv, S, D)
    repeated = repeat_kv(kv, rep)                  # [B, 28, S, D] — what flash gets
    assert repeated.shape[1] == H_kv * rep == 28

    # The un-repeat _run_fused applies (head h = kv*rep + r, so ::rep picks r=0).
    recovered = repeated[:, ::rep].contiguous()
    assert recovered.shape[1] == H_kv
    assert torch.equal(recovered, kv)              # bit-exact (expand copies)

    # rep == 1 (Llama's fp tier already H_kv) is a no-op: nothing to slice.
    assert torch.equal(repeat_kv(kv, 1)[:, ::1], kv)
