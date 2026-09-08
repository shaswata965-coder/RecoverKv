"""§5.1's window-score emission is exactly the old token-score path, reduced.

**Why this file carries the weight.** The Triton kernel is GPU-only and this repo
develops on CPU, so the *lowering* of §5.1 ships unvalidated (the same contract
the score kernel and the fused decode kernel already ship under). What must NOT
ship unvalidated is the algorithm: the window tiling, the sink prologue, the
per-tile window sums taken against a *running* max, and the epilogue that
rescales each window from its own tile max to the global LSE.

``two_tier_window_reference`` implements that algorithm tile for tile, and these
tests pin it against the path it replaces — ``softmax`` over all keys, strip the
sink, window-sum — which is what ``scorer.reduce_token_scores_to_windows``
computes from the kernel's old ``[B, H_q, S]`` output. If the two agree for every
geometry, the only thing left to be wrong on a GPU is Triton codegen.
"""

from __future__ import annotations

import pytest
import torch

from modules.windowed_cache.decode_kernel import (
    two_tier_decode_reference,
    two_tier_window_reference,
    window_tiling,
)
from modules.windowed_cache.scorer import reduce_token_scores_to_windows


def _expected(q, k, v, scaling, num_sink, ws, n_body_win):
    """The path §5.1 replaces: full softmax -> strip sink -> window-sum."""
    out, token_scores = two_tier_decode_reference(q, k, v, scaling)
    S = k.shape[2]
    body_end = min(num_sink + n_body_win * ws, S)
    body = reduce_token_scores_to_windows(
        token_scores[..., :body_end].float(), num_sink, ws)
    tail = token_scores[..., body_end:].float()
    if tail.shape[-1]:
        pad = (-tail.shape[-1]) % ws
        if pad:
            tail = torch.nn.functional.pad(tail, (0, pad))
        q_w = tail.reshape(*tail.shape[:-1], -1, ws).sum(-1)
        expected = torch.cat([body, q_w], dim=-1)
    else:
        expected = body
    return out, expected


@pytest.mark.parametrize("ws", [1, 2, 3, 4, 5, 8, 12, 16, 32, 64, 128])
@pytest.mark.parametrize("num_sink", [0, 1, 5])
def test_window_reference_matches_the_token_score_path(ws, num_sink):
    """Every ws from 1 to 128, power-of-2 or not, with and without sinks."""
    torch.manual_seed(ws * 100 + num_sink)
    B, H_kv, rep, D = 2, 2, 2, 16
    H_q = H_kv * rep
    n_body_win, n_q_win = 3, 2
    S = num_sink + (n_body_win + n_q_win) * ws
    q = torch.randn(B, H_q, D)
    k = torch.randn(B, H_kv, S, D)
    v = torch.randn(B, H_kv, S, D)
    scaling = D ** -0.5

    out_e, exp = _expected(q, k, v, scaling, num_sink, ws, n_body_win)
    out_g, got = two_tier_window_reference(q, k, v, scaling, num_sink, ws,
                                           n_body_win, num_sink + n_body_win * ws)

    assert got.shape == exp.shape, f"{tuple(got.shape)} != {tuple(exp.shape)}"
    torch.testing.assert_close(out_g.float(), out_e.float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got, exp, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("ws", [4, 8, 12])
def test_partial_trailing_body_window(ws):
    """The newest body window is usually partial — it still gets its own column,
    summing only its live tokens, exactly as the right-pad path does."""
    torch.manual_seed(7)
    B, H_kv, rep, D = 1, 2, 2, 16
    H_q, num_sink = H_kv * rep, 2
    n_body_win = 3
    S = num_sink + (n_body_win - 1) * ws + 1          # last body window has 1 token
    q = torch.randn(B, H_q, D); k = torch.randn(B, H_kv, S, D)
    v = torch.randn(B, H_kv, S, D); scaling = D ** -0.5
    _, exp = _expected(q, k, v, scaling, num_sink, ws, n_body_win)
    _, got = two_tier_window_reference(q, k, v, scaling, num_sink, ws, n_body_win, S)
    torch.testing.assert_close(got, exp, rtol=1e-5, atol=1e-6)


def test_scores_are_a_normalized_softmax():
    """Window scores must sum to 1 over all non-sink keys — they are softmax mass.

    Catches a whole class of epilogue bug (wrong m_tile, missed rescale) that a
    shape check would not.
    """
    torch.manual_seed(3)
    B, H_kv, rep, D, ws, num_sink = 2, 2, 2, 16, 8, 0
    H_q, n_body_win = H_kv * rep, 4
    S = (n_body_win + 3) * ws
    q = torch.randn(B, H_q, D); k = torch.randn(B, H_kv, S, D)
    v = torch.randn(B, H_kv, S, D)
    _, got = two_tier_window_reference(q, k, v, D ** -0.5, num_sink, ws,
                                      n_body_win, num_sink + n_body_win * ws)
    torch.testing.assert_close(got.sum(-1), torch.ones(B, H_q), rtol=1e-5, atol=1e-6)


def test_extreme_logits_do_not_overflow():
    """The running-max rescale is the only thing keeping this finite; a naive
    `exp(logit)` would inf out here."""
    torch.manual_seed(1)
    B, H_kv, rep, D, ws = 1, 1, 2, 16, 4
    H_q, n_body_win, num_sink = H_kv * rep, 3, 0
    S = (n_body_win + 2) * ws
    q = torch.randn(B, H_q, D) * 60.0
    k = torch.randn(B, H_kv, S, D) * 60.0
    v = torch.randn(B, H_kv, S, D)
    _, exp = _expected(q, k, v, D ** -0.5, num_sink, ws, n_body_win)
    out, got = two_tier_window_reference(q, k, v, D ** -0.5, num_sink, ws,
                                        n_body_win, num_sink + n_body_win * ws)
    assert torch.isfinite(got).all() and torch.isfinite(out).all()
    torch.testing.assert_close(got, exp, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("ws,expect_nw,expect_t", [
    (1, 64, 64), (2, 32, 64), (4, 16, 64), (8, 8, 64),
    (12, 5, 64), (16, 4, 64), (32, 2, 64), (64, 1, 64), (128, 1, 128),
])
def test_window_tiling_geometry(ws, expect_nw, expect_t):
    """A tile is whole windows, padded to a power of 2 for ``tl.arange``."""
    nw, t = window_tiling(ws)
    assert (nw, t) == (expect_nw, expect_t)
    assert nw * ws <= t, "tile must fit inside its padded block"
    assert t & (t - 1) == 0, "tl.arange needs a power-of-2 block"


# ---------------------------------------------------------------------------
# End-to-end against a REAL two-tier cache: does §5.1 reproduce the old scores?
# ---------------------------------------------------------------------------


def _seeded_two_tier_cache(ws, num_sink, prefill, steps, quant_ratio=0.5):
    """A cache driven past its first eviction, so it has a live Q tier."""
    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig

    class _Cfg:
        num_key_value_heads = 2
        num_attention_heads = 4
        hidden_size = 4 * 16
        head_dim = 16

    class _Rope(torch.nn.Module):
        def forward(self, x, position_ids):
            pos = position_ids.to(torch.float32)
            f = pos.unsqueeze(-1) * torch.arange(1, 9, dtype=torch.float32) * 0.01
            e = torch.cat([f, f], dim=-1)
            return e.cos(), e.sin()

    cache = WindowedCache(
        config=WindowedCacheConfig(
            window_size=ws, num_sink_tokens=num_sink, local_window_size=ws,
            cache_budget=0.5, quant_ratio=quant_ratio, first_eviction_step=0),
        prefill_len=prefill, model_config=_Cfg(), kv_dtype=torch.float32,
        rope_module=_Rope(), num_layers=1, max_tokens=steps)
    g = torch.Generator().manual_seed(19)
    k = torch.randn(1, 2, prefill, 16, generator=g)
    cache.update(k, k.clone(), 0, cache_kwargs={"cache_position": torch.arange(prefill)})
    for t in range(steps):
        st, store = cache._states[0], cache._stores[0]
        nq = store.num_active_windows if store is not None else 0
        W = -(-max(st.seq_length - num_sink, 0) // ws) + nq
        cache.cache_kwargs[0]["window_scores"] = torch.rand(1, 4, W, generator=g)
        k1 = torch.randn(1, 2, 1, 16, generator=g)
        cache.update(k1, k1.clone(), 0,
                     cache_kwargs={"cache_position": torch.tensor([prefill + t])})
    return cache


@pytest.mark.parametrize("ws,num_sink", [(4, 0), (4, 2), (8, 0), (8, 5), (16, 3)])
def test_end_to_end_scores_match_the_old_reduce_path(ws, num_sink):
    """The whole §5.1 caller change, against a real cache's own geometry.

    Old path: kernel emits [B,H_q,S] token scores -> ``reduce_two_tier_scores``
    (strip sink, right-pad, window-sum, einops-reduce the Q tier, cat, gather).
    New path: kernel emits [B,H_q,W_phys] window mass -> one gather by ``order``.

    Both are driven off the SAME cache state, the same effective K/V and the same
    ``score_meta``, so any disagreement is §5.1's fault and nothing else. This is
    what makes the physical window order, the ``n_body_win`` derivation and the
    sink handling verified rather than argued.
    """
    from modules.windowed_cache.scorer import reduce_two_tier_scores

    cache = _seeded_two_tier_cache(ws, num_sink, prefill=max(64, ws * 8), steps=ws * 3)
    store = cache._stores[0]
    if store is None or store.num_active_windows == 0:
        pytest.skip("fixture produced no Q tier; nothing two-tier to compare")

    eff_k, eff_v, score_meta = cache._materialize(0)
    assert score_meta is not None
    order, q_token_len = score_meta
    S_fp = cache._states[0].seq_length
    q = torch.randn(1, 4, 16, generator=torch.Generator().manual_seed(5))
    scaling = 16 ** -0.5

    # --- old path -------------------------------------------------------
    _, token_scores = two_tier_decode_reference(q, eff_k, eff_v, scaling)
    old = reduce_two_tier_scores(
        token_scores.float(), num_sink, ws, q_token_len, order)

    # --- new path (§5.1) ------------------------------------------------
    n_body_win = -(-max(S_fp - num_sink, 0) // ws)
    _, wsum = two_tier_window_reference(
        q, eff_k, eff_v, scaling, num_sink, ws, n_body_win, S_fp)
    assert wsum.shape[-1] == order.shape[1], (
        f"kernel emitted {wsum.shape[-1]} windows, score_meta permutes "
        f"{order.shape[1]} — these must agree")
    idx = order.unsqueeze(1).expand(wsum.shape[0], wsum.shape[1], order.shape[1])
    new = torch.gather(wsum, -1, idx)

    torch.testing.assert_close(new, old.float(), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Padding cannot leak into a window sum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ws", list(range(1, 129)))
def test_padding_lanes_are_unreachable_by_any_window(ws):
    """Exhaustive over every ``ws`` from 1 to 128, not a sample.

    A Triton block must be a power of two, so a tile of ``BLOCK_NW * ws`` keys is
    materialised as ``BLOCK_T`` lanes with the surplus masked. The masking is one
    defence; this is the other, and it is structural: a padding lane sits at
    ``offs_t >= BLOCK_NW*ws``, so its ``t_win = offs_t // ws`` is ``>= BLOCK_NW``,
    while the per-window store loop only runs ``j`` over ``0 … BLOCK_NW-1``. No
    window's selector can name it, so padding cannot be folded into a window sum
    even if the mask were dropped.

    Also pins the converse: every REAL lane belongs to exactly one window, and
    each window owns exactly ``ws`` lanes — i.e. the padding does not shrink or
    stretch what a window means.
    """
    block_nw, block_t = window_tiling(ws)
    offs_t = torch.arange(block_t)
    t_win = offs_t // ws
    is_pad = offs_t >= block_nw * ws

    # 1. no padding lane is reachable by any j in [0, BLOCK_NW)
    for j in range(block_nw):
        assert not bool(((t_win == j) & is_pad).any()), (
            f"ws={ws}: window {j} would select a padding lane")

    # 2. every real lane lands in exactly one window, and windows are full width
    real = ~is_pad
    assert int(real.sum()) == block_nw * ws
    for j in range(block_nw):
        owned = int(((t_win == j) & real).sum())
        assert owned == ws, (
            f"ws={ws}: window {j} owns {owned} lanes, expected exactly {ws} — "
            "padding must not change what a window sums")


@pytest.mark.parametrize("ws", [3, 5, 7, 12, 24, 48])
def test_non_power_of_two_windows_sum_the_right_token_count(ws):
    """The cases where padding actually exists (``BLOCK_NW*ws < BLOCK_T``).

    A one-hot value tensor turns each window's score into a literal count of the
    tokens it summed, so a padding leak or a dropped token shows up as an integer
    that is off by exactly that many — far easier to read than a float mismatch.
    """
    block_nw, block_t = window_tiling(ws)
    assert block_nw * ws < block_t, f"ws={ws} has no padding; wrong fixture"

    torch.manual_seed(ws)
    B, H_kv, rep, D, num_sink = 1, 1, 2, 16, 0
    H_q, n_body_win = H_kv * rep, block_nw + 2      # spans >1 tile
    S = n_body_win * ws
    q = torch.zeros(B, H_q, D)                      # all logits equal -> uniform
    k = torch.zeros(B, H_kv, S, D)
    v = torch.randn(B, H_kv, S, D)
    _, got = two_tier_window_reference(q, k, v, 1.0, num_sink, ws, n_body_win, S)

    # Uniform softmax over S keys => each window's mass is (its token count)/S.
    expected = torch.full((B, H_q, n_body_win), ws / S)
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(got.sum(-1), torch.ones(B, H_q),
                               rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# §4.3(a) base-2 softmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ws,num_sink", [(4, 0), (8, 5), (12, 3), (32, 1)])
def test_exp2_fold_matches_base_e(ws, num_sink):
    """Folding ``log2(e)`` into ``scale`` is exact in real arithmetic.

    Every logit becomes a base-2 exponent, so ``m``, ``wmax`` and ``lse`` are all
    in base-2 units while ``p``, ``l``, ``acc`` and ``out`` are unchanged
    quantities — including through the window epilogue's ``exp2(wmax - lse)``,
    which is the part a naive base swap would get wrong.
    """
    torch.manual_seed(ws * 17 + num_sink)
    B, H_kv, rep, D = 2, 2, 2, 16
    H_q, n_body_win, n_q_win = H_kv * rep, 3, 2
    S = num_sink + (n_body_win + n_q_win) * ws
    q = torch.randn(B, H_q, D)
    k = torch.randn(B, H_kv, S, D)
    v = torch.randn(B, H_kv, S, D)
    body_end = num_sink + n_body_win * ws

    oe, we = two_tier_window_reference(q, k, v, D ** -0.5, num_sink, ws,
                                       n_body_win, body_end, exp2=False)
    o2, w2 = two_tier_window_reference(q, k, v, D ** -0.5, num_sink, ws,
                                       n_body_win, body_end, exp2=True)
    torch.testing.assert_close(o2.float(), oe.float(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(w2, we, rtol=1e-4, atol=1e-7)


def test_exp2_is_default_and_has_a_control_arm(monkeypatch):
    from modules.windowed_cache.decode_kernel import _LOG2E, decode_exp2_enabled
    import math

    monkeypatch.delenv("STICKYKV_DECODE_EXP2", raising=False)
    assert decode_exp2_enabled() is True, "base-2 softmax must be the default"
    monkeypatch.setenv("STICKYKV_DECODE_EXP2", "0")
    assert decode_exp2_enabled() is False, "the A/B control arm must work"
    assert _LOG2E == pytest.approx(math.log2(math.e), rel=1e-15)


def test_no_base_e_exp_survives_in_the_kernel():
    """The fold is per-launch on a scalar; a stray ``tl.exp`` would mean some
    logits are base-2 and some are not, which is silently wrong rather than
    slow."""
    import pathlib
    src = pathlib.Path("modules/windowed_cache/decode_kernel.py").read_text(
        encoding="utf-8")
    body = src[src.index("if _HAS_TRITON:"):src.index("def _pow2_at_least(")]
    assert "tl.exp(" not in body, "base-e tl.exp left in the kernel"
    assert "tl.log(" not in body, "base-e tl.log left in the kernel"
    assert body.count("tl.exp2(") >= 7 and "tl.log2(" in body


# ---------------------------------------------------------------------------
# §4.3(b) the RoPE tables must match the store's dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rope_halves_follow_the_store_dtype(dtype):
    from modules.windowed_cache.decode_kernel import rope_cos_sin_halves

    class _Rope(torch.nn.Module):
        def forward(self, x, position_ids):
            pos = position_ids.to(torch.float32)
            f = pos.unsqueeze(-1) * torch.arange(1, 9, dtype=torch.float32) * 0.01
            e = torch.cat([f, f], dim=-1)
            return e.cos().to(x.dtype), e.sin().to(x.dtype)

    cos, sin = rope_cos_sin_halves(_Rope(), torch.arange(16).unsqueeze(0), dtype)
    assert cos.dtype == dtype and sin.dtype == dtype
    assert cos.is_contiguous() and sin.is_contiguous()


def test_matching_the_rope_dtype_makes_the_q_tier_round_trip():
    """The Q tier is un-rotated at the store dtype, so it must be re-rotated
    there too.

    Every other RoPE in this cache hands HF's rotary the fp16 *key* as its
    reference and so runs in fp16: the model's own rotation of the fp tier, and
    ``unrotate_key_window`` / ``rotate_key_window``. The decode kernel used to
    pass ``torch.empty(1,1,1)`` and get fp32, so the key it reconstructed was not
    the key that had been demoted — a token's value depended on which tier it was
    in. This pins the fix by measuring the round trip both ways.
    """
    torch.manual_seed(0)
    T, half = 512, 64
    pos = torch.arange(1000, 1000 + T)
    inv = 1.0 / (500000.0 ** (torch.arange(0, half).float() / half))
    ang = pos.unsqueeze(-1).float() * inv
    c32, s32 = ang.cos(), ang.sin()
    c16, s16 = c32.half().float(), s32.half().float()
    k_lo = torch.randn(T, half).half().float()
    k_hi = torch.randn(T, half).half().float()

    # demote always un-rotates at the store dtype (fp16)
    p_lo, p_hi = k_lo * c16 + k_hi * s16, k_hi * c16 - k_lo * s16

    def rel(c, s):
        r_lo, r_hi = p_lo * c - p_hi * s, p_hi * c + p_lo * s
        err = torch.cat([(r_lo - k_lo).abs(), (r_hi - k_hi).abs()])
        ref = torch.cat([k_lo.abs(), k_hi.abs()]).clamp_min(1e-3)
        return float((err / ref).max())

    matched, mismatched = rel(c16, s16), rel(c32, s32)
    assert matched < mismatched / 100, (
        f"matching the dtype should cut max round-trip error by orders of "
        f"magnitude; got matched={matched:.2e} mismatched={mismatched:.2e}")
