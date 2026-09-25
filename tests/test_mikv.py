"""MiKV (arXiv:2402.18096) — quantizer, balancer, budget, cache, hooks.

CPU-only, on a tiny random Llama. What is pinned, and why each matters:

* the quantizer is the paper's Eq. 1 at every width, and packs to exactly
  ``bits`` bits — int3 costs 3, not 4;
* the balancer is Eq. 2, preserves every logit (Eqs. 3-4), and does what it is
  for: less logit error when the query has outlier channels;
* the budget reproduces the paper's own Table 2/3 cache sizes;
* ``importance_ratio = 1.0`` is token-for-token the full cache — the control arm;
* the H2O scores are the attention the model computed, not an estimate of it;
* the tiers partition the sequence, the recent window is never demoted, and the
  dense read is ``I(k * b) / b`` where quantized and exact where not;
* nothing runs without the hooks: an unscored MiKV would still look like MiKV.
"""

from __future__ import annotations

import math

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM  # noqa: E402

from modules.mikv import (  # noqa: E402
    MiKVCache,
    MiKVConfig,
    MiKVConfigError,
    MiKVHooksMissing,
    install_mikv_hooks,
    make_mikv_method,
    paper_cache_size,
)
from modules.mikv.quant import (  # noqa: E402
    channel_balancer,
    dequantize,
    pack_codes,
    quantize,
    token_bytes,
    unpack_codes,
)

N_LAYERS = 2
HQ, HKV, D = 4, 2, 32


def _tiny_llama(impl: str = "eager", seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=128, hidden_size=HQ * D, intermediate_size=256,
                      num_hidden_layers=N_LAYERS, num_attention_heads=HQ,
                      num_key_value_heads=HKV, max_position_embeddings=512)
    return LlamaForCausalLM._from_config(cfg, attn_implementation=impl).eval()


def _generate(model, ids, cache, n=12, mask=None):
    hooks = install_mikv_hooks(model, cache) if isinstance(cache, MiKVCache) else None
    try:
        with torch.no_grad():
            return model.generate(ids, attention_mask=mask if mask is not None
                                  else torch.ones_like(ids), max_new_tokens=n,
                                  do_sample=False, past_key_values=cache,
                                  pad_token_id=0)
    finally:
        if hooks is not None:
            hooks.remove()


# ---------------------------------------------------------------------------
# quantizer (Eq. 1) and packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", range(1, 9))
def test_pack_unpack_roundtrip(bits):
    g = torch.Generator().manual_seed(bits)
    codes = torch.randint(0, 1 << bits, (3, 5, 64), generator=g, dtype=torch.int64)
    packed = pack_codes(codes.to(torch.uint8), bits)
    assert packed.shape[-1] == 64 * bits // 8
    assert torch.equal(unpack_codes(packed, bits, 64), codes.to(torch.uint8))


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_byte_aligned_fast_path_has_the_word_layout(bits):
    from modules.mikv.quant import _pack_words

    codes = torch.randint(0, 1 << bits, (2, 3, 128),
                          generator=torch.Generator().manual_seed(bits)).to(torch.uint8)
    assert torch.equal(pack_codes(codes, bits), _pack_words(codes, bits))


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_quantize_is_eq1_on_the_stored_grid(bits):
    torch.manual_seed(1)
    x = torch.randn(4, 3, 128) * 3
    q = quantize(x, bits, 64)
    levels = (1 << bits) - 1
    g = x.view(4, 3, 2, 64)
    alpha = ((g.amax(-1) - g.amin(-1)) / levels).to(torch.float16)
    beta = g.amin(-1).to(torch.float16)
    assert torch.equal(q.alpha, alpha) and torch.equal(q.beta, beta)
    a, b = alpha.float().unsqueeze(-1), beta.float().unsqueeze(-1)
    want = (torch.round((g - b) / a).clamp(0, levels) * a + b).view(4, 3, 128)
    assert torch.allclose(dequantize(q), want, atol=0, rtol=0)
    # Round-to-nearest: every element within half a step (plus the fp16 grid).
    err = (dequantize(q) - x).abs().view(4, 3, 2, 64)
    assert bool((err <= a * 0.5 + 1e-2 * a + 1e-3).all())


def test_packed_sizes_are_exact_bit_widths():
    x = torch.randn(1, 1, 1, 128)
    assert quantize(x, 2, 64).codes.shape[-1] == 32
    assert quantize(x, 3, 64).codes.shape[-1] == 48
    assert quantize(x, 4, 64).codes.shape[-1] == 64
    # codes + one fp16 (alpha, beta) pair per group of 64
    assert token_bytes(2, 64, 128) == 32 + 8
    assert token_bytes(16, 64, 128) == 256


def test_constant_group_is_exact():
    x = torch.full((2, 128), 0.375)
    assert torch.equal(dequantize(quantize(x, 2, 64)), x)


# ---------------------------------------------------------------------------
# channel balancer (Eqs. 2-4)
# ---------------------------------------------------------------------------


def test_balancer_is_eq2_and_one_where_undefined():
    qmax = torch.tensor([4.0, 1.0, 0.0, 9.0])
    kmax = torch.tensor([1.0, 4.0, 3.0, 0.0])
    b = channel_balancer(qmax, kmax)
    assert torch.allclose(b, torch.tensor([2.0, 0.5, 1.0, 1.0]))
    # Balanced maxima meet at the geometric mean.
    assert torch.allclose((qmax / b)[:2], (kmax * b)[:2])


def test_balancer_preserves_logits_and_reduces_error_under_query_outliers():
    torch.manual_seed(0)
    n, d = 512, 128
    q = torch.randn(n, d)
    k = torch.randn(n, d)
    q[:, 5] *= 30.0     # a query outlier channel: key error there is amplified
    k[:, 70] *= 12.0    # a key outlier channel: it sets every token's range
    b = channel_balancer(q.abs().amax(0), k.abs().amax(0))
    assert torch.allclose((q / b) @ (k * b).T, q @ k.T, rtol=1e-4, atol=1e-3)

    exact = q @ k.T
    plain = q @ dequantize(quantize(k, 2, 64)).T
    balanced = q @ (dequantize(quantize(k * b, 2, 64)) / b).T
    err_plain = (plain - exact).pow(2).mean().sqrt()
    err_bal = (balanced - exact).pow(2).mean().sqrt()
    assert err_bal < 0.7 * err_plain, (err_bal, err_plain)


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_paper_cache_sizes_table2_and_table3():
    # Table 3 (importance ratio 20%, outlier-aware INT2 retained cache).
    assert paper_cache_size(0.2, 16, 2) == pytest.approx(0.325)    # 33%
    assert paper_cache_size(0.2, 8, 2) == pytest.approx(0.23125)   # 23%
    assert paper_cache_size(0.2, 4, 2) == pytest.approx(0.18125)   # 18%
    assert paper_cache_size(0.2, 2, 2) == pytest.approx(0.15625)   # 16%
    # Table 2, outlier-aware INT3: 38%.
    assert paper_cache_size(0.2, 16, 3) == pytest.approx(0.375)


def test_cache_budget_resolves_to_the_importance_ratio_it_prices():
    b = MiKVConfig(cache_budget=0.325).resolve(head_dim=128)
    assert b.mode == "cache_budget"
    assert b.importance_ratio == pytest.approx(0.2)
    with pytest.raises(MiKVConfigError, match="floor"):
        MiKVConfig(cache_budget=0.15).resolve(head_dim=128)


def test_config_rejects_contradictions():
    with pytest.raises(MiKVConfigError):
        MiKVConfig(importance_ratio=0.2, cache_budget=0.3)
    with pytest.raises(MiKVConfigError):
        MiKVConfig(hi_bits=4, lo_bits=8)
    with pytest.raises(MiKVConfigError):
        MiKVConfig(lo_bits=16)
    assert MiKVConfig().importance_ratio == 0.2


def test_importance_size_is_monotone_and_grows_by_at_most_one():
    b = MiKVConfig(importance_ratio=0.37, recent_ratio=0.5).resolve(head_dim=64)
    prev_hi = prev_rec = 0
    for s in range(1, 400):
        hi, rec = b.n_hi(s), b.n_recent(s)
        assert 0 <= hi - prev_hi <= 1 and 0 <= rec - prev_rec <= 1
        assert rec <= hi <= s
        prev_hi, prev_rec = hi, rec


# ---------------------------------------------------------------------------
# the cache on a real (tiny) Llama
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_importance_ratio_one_is_the_full_cache(impl):
    model = _tiny_llama(impl)
    ids = torch.randint(0, 128, (2, 37), generator=torch.Generator().manual_seed(3))
    ref = _generate(model, ids, DynamicCache(), n=16)
    cache = MiKVCache(MiKVConfig(importance_ratio=1.0), N_LAYERS, torch.float32, 16)
    out = _generate(model, ids, cache, n=16)
    assert torch.equal(out, ref)
    assert cache.stats["demoted"] == 0
    assert cache.memory_report()["lo_tokens"] == 0


def test_scores_are_the_attention_the_model_computed():
    model = _tiny_llama("eager")
    ids = torch.randint(0, 128, (2, 30), generator=torch.Generator().manual_seed(4))
    cache = MiKVCache(MiKVConfig(importance_ratio=1.0), N_LAYERS, torch.float32, 2)
    hooks = install_mikv_hooks(model, cache)
    try:
        with torch.no_grad():
            out = model(ids, past_key_values=cache, output_attentions=True, use_cache=True)
            nxt = out.logits[:, -1:].argmax(-1)
            out2 = model(nxt, past_key_values=cache, output_attentions=True, use_cache=True)
    finally:
        hooks.remove()
    g = HQ // HKV
    for layer in range(N_LAYERS):
        pre = out.attentions[layer].float().sum(2).view(2, HKV, g, 30).sum(2)
        step = out2.attentions[layer].float().sum(2).view(2, HKV, g, 31).sum(2)
        want = torch.cat([pre, torch.zeros(2, HKV, 1)], -1) + step
        st = cache._layers[layer]
        got = torch.zeros(2, HKV, 31).scatter(2, st.hi.live_pos(), st.hi_score)
        assert torch.allclose(got, want, atol=1e-5)


def _check_partition(cache: MiKVCache, seq_len: int) -> None:
    budget = cache.budget
    for st in cache._layers:
        assert st.length == seq_len
        assert st.hi.length == budget.n_hi(seq_len)
        assert st.hi.length + st.lo.length == seq_len
        allpos = torch.cat([st.hi.live_pos(), st.lo.live_pos().long()], dim=-1)
        want = torch.arange(seq_len).expand_as(allpos)
        assert torch.equal(allpos.sort(dim=-1).values, want)
        n_rec = budget.n_recent(seq_len)
        if n_rec:
            recent = torch.arange(seq_len - n_rec, seq_len)
            for row in st.hi.live_pos().reshape(-1, st.hi.length):
                assert bool(torch.isin(recent, row).all())


@pytest.mark.parametrize("hi_bits,lo_bits", [(16, 2), (8, 2), (4, 3), (16, 4)])
def test_tiers_partition_the_sequence(hi_bits, lo_bits):
    model = _tiny_llama("sdpa")
    ids = torch.randint(0, 128, (2, 41), generator=torch.Generator().manual_seed(5))
    cache = MiKVCache(MiKVConfig(importance_ratio=0.25, hi_bits=hi_bits,
                                 lo_bits=lo_bits), N_LAYERS, torch.float32, 20)
    out = _generate(model, ids, cache, n=20)
    _check_partition(cache, out.shape[-1] - 1)  # the last token is never fed back


def test_h2o_arm_drops_what_mikv_keeps():
    model = _tiny_llama("eager")
    ids = torch.randint(0, 128, (1, 40), generator=torch.Generator().manual_seed(13))
    full = _generate(model, ids, DynamicCache(), n=10)
    same = MiKVCache(MiKVConfig(importance_ratio=1.0, lo_bits=0), N_LAYERS,
                     torch.float32, 10)
    assert torch.equal(_generate(model, ids, same, n=10), full)

    h2o = MiKVCache(MiKVConfig(importance_ratio=0.25, lo_bits=0), N_LAYERS,
                    torch.float32, 10)
    out = _generate(model, ids, h2o, n=10)
    seq = out.shape[-1] - 1
    rep = h2o.memory_report()
    assert rep["lo_tokens"] == 0 and rep["lo_kv_bytes"] == 0
    assert rep["hi_tokens"] == h2o.budget.n_hi(seq)
    assert rep["hi_tokens"] + rep["dropped_tokens"] == seq
    # Same budget code: MiKV at the same ratio retains as many as H2O drops.
    mikv = MiKVCache(MiKVConfig(importance_ratio=0.25, lo_bits=2), N_LAYERS,
                     torch.float32, 10)
    _generate(model, ids, mikv, n=10)
    assert mikv.memory_report()["lo_tokens"] == rep["dropped_tokens"]


def test_h2o_arm_refuses_padding():
    model = _tiny_llama("eager")
    ids = torch.randint(3, 128, (1, 12))
    mask = torch.ones_like(ids)
    mask[:, :3] = 0
    cache = MiKVCache(MiKVConfig(importance_ratio=0.5, lo_bits=0), N_LAYERS, torch.float32)
    with pytest.raises(NotImplementedError, match="padding"):
        _generate(model, ids, cache, n=2, mask=mask)


def test_rtn_arm_quantizes_every_token():
    model = _tiny_llama("sdpa")
    ids = torch.randint(0, 128, (1, 33), generator=torch.Generator().manual_seed(6))
    cache = MiKVCache(MiKVConfig(importance_ratio=0.0), N_LAYERS, torch.float32, 8)
    out = _generate(model, ids, cache, n=8)
    rep = cache.memory_report()
    assert rep["hi_tokens"] == 0 and rep["lo_tokens"] == out.shape[-1] - 1


# ---------------------------------------------------------------------------
# the cache driven directly: the read is the paper's equations
# ---------------------------------------------------------------------------


def _drive(cache: MiKVCache, k, v, q_rows):
    """One ``update`` on layer 0 with a hand-made query (identity RoPE)."""
    b, _, n, d = k.shape
    cache._stash_query(0, q_rows.transpose(1, 2).reshape(b, n, -1))
    cos = torch.ones(b, n, d, dtype=k.dtype)
    sin = torch.zeros(b, n, d, dtype=k.dtype)
    return cache.update(k, v, 0, {"cos": cos, "sin": sin})


def test_prefill_is_exact_and_decode_reads_eq3_eq4():
    torch.manual_seed(7)
    b, n, d = 1, 64, 32
    cfg = MiKVConfig(importance_ratio=0.25, lo_bits=2, group_size=16)
    cache = MiKVCache(cfg, num_layers=1, dtype=torch.float16, max_new_tokens=4)
    k = torch.randn(b, HKV, n, d).half()
    k[..., 3] *= 20
    v = torch.randn(b, HKV, n, d).half()
    q = torch.randn(b, HQ, n, d).half()
    q[..., 9] *= 25
    rk, rv = _drive(cache, k, v, q)
    assert rk is k and rv is v  # the prompt attends at full precision

    st = cache._layers[0]
    g = HQ // HKV
    want_b = channel_balancer(q.abs().view(b, HKV, g, n, d).amax(dim=(2, 3)),
                              k.abs().amax(dim=2))
    assert torch.equal(st.balancer, want_b)

    k1 = torch.randn(b, HKV, 1, d).half()
    v1 = torch.randn(b, HKV, 1, d).half()
    q1 = torch.randn(b, HQ, 1, d).half()
    lo_pos = st.lo.live_pos().long()  # before the step demotes anything
    hi_pos = st.hi.live_pos()
    dk, dv = _drive(cache, k1, v1, q1)
    assert torch.equal(dk[:, :, n:], k1) and torch.equal(dv[:, :, n:], v1)
    gather = lambda x, p: x.gather(2, p.unsqueeze(-1).expand(*p.shape, d))  # noqa: E731
    assert torch.equal(gather(dk, hi_pos), gather(k, hi_pos))  # exact at fp16
    lk = gather(k, lo_pos).float()
    want_k = (dequantize(quantize(lk * want_b.unsqueeze(2), 2, 16))
              / want_b.unsqueeze(2)).half()
    want_v = dequantize(quantize(gather(v, lo_pos).float(), 2, 16)).half()
    assert torch.equal(gather(dk, lo_pos), want_k)
    assert torch.equal(gather(dv, lo_pos), want_v)


def test_memory_report_counts_every_byte():
    torch.manual_seed(8)
    b, n, d = 2, 80, 32
    cfg = MiKVConfig(importance_ratio=0.2, lo_bits=2, group_size=16)
    cache = MiKVCache(cfg, num_layers=1, dtype=torch.float16, max_new_tokens=0)
    _drive(cache, torch.randn(b, HKV, n, d).half(), torch.randn(b, HKV, n, d).half(),
           torch.randn(b, HQ, n, d).half())
    rep = cache.memory_report()
    n_hi, n_lo = 16, 64
    assert (rep["hi_tokens"], rep["lo_tokens"]) == (n_hi, n_lo)
    per_head = 2 * (n_hi * token_bytes(16, 16, d) + n_lo * token_bytes(2, 16, d))
    assert rep["kv_bytes"] == b * HKV * per_head
    assert rep["dense_fp16_bytes"] == b * HKV * 2 * n * d * 2
    # 20% exact + 80% at 2 + 32/16 bits = (0.2*16 + 0.8*4)/16
    assert rep["kv_fraction"] == pytest.approx((0.2 * 16 + 0.8 * 4) / 16)
    meta = b * HKV * (n_hi * 8 + n_lo * 4 + n_hi * 4 + d * 4)
    assert rep["meta_bytes"] == meta


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------


def test_update_without_hooks_raises():
    model = _tiny_llama("sdpa")
    cache = MiKVCache(MiKVConfig(), N_LAYERS, torch.float32)
    with pytest.raises(MiKVHooksMissing):
        with torch.no_grad():
            model(torch.randint(0, 128, (1, 8)), past_key_values=cache, use_cache=True)


def test_hooks_remove_cleanly():
    model = _tiny_llama("sdpa")
    before = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
    hooks = install_mikv_hooks(model, MiKVCache(MiKVConfig(), N_LAYERS, torch.float32))
    during = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
    hooks.remove()
    hooks.remove()
    after = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
    assert during == before + 2 * N_LAYERS and after == before


def test_left_padding_is_invisible_to_scores_and_balancer():
    model = _tiny_llama("eager")
    real = torch.randint(3, 128, (1, 20), generator=torch.Generator().manual_seed(9))
    pad = 6
    padded = torch.cat([torch.zeros(1, pad, dtype=torch.long), real], dim=1)
    mask = torch.cat([torch.zeros(1, pad, dtype=torch.long),
                      torch.ones(1, 20, dtype=torch.long)], dim=1)
    solo = MiKVCache(MiKVConfig(importance_ratio=1.0), N_LAYERS, torch.float32, 2)
    _generate(model, real, solo, n=2)
    both = MiKVCache(MiKVConfig(importance_ratio=1.0), N_LAYERS, torch.float32, 2)
    _generate(model, padded, both, n=2, mask=mask)
    for a, c in zip(solo._layers, both._layers):
        assert torch.allclose(a.balancer, c.balancer, atol=1e-5)
        sa = torch.zeros(1, HKV, 21).scatter(2, a.hi.live_pos(), a.hi_score)
        sc = torch.zeros(1, HKV, 27).scatter(2, c.hi.live_pos(), c.hi_score)
        assert torch.allclose(sc[..., :pad], torch.zeros(1, HKV, pad))
        assert torch.allclose(sc[..., pad:], sa, atol=1e-4)


# ---------------------------------------------------------------------------
# fidelity moves the right way
# ---------------------------------------------------------------------------


def _teacher_forced_logits(model, ids, cache, n_prompt):
    hooks = install_mikv_hooks(model, cache) if isinstance(cache, MiKVCache) else None
    outs = []
    try:
        with torch.no_grad():
            model(ids[:, :n_prompt], past_key_values=cache, use_cache=True)
            for t in range(n_prompt, ids.shape[1]):
                outs.append(model(ids[:, t:t + 1], past_key_values=cache,
                                  use_cache=True).logits[:, -1])
    finally:
        if hooks is not None:
            hooks.remove()
    return torch.stack(outs, 1)


def test_error_shrinks_with_more_bits_and_more_importance():
    model = _tiny_llama("sdpa", seed=11)
    ids = torch.randint(0, 128, (2, 72), generator=torch.Generator().manual_seed(12))
    ref = _teacher_forced_logits(model, ids, DynamicCache(), 56)

    def err(**kw):
        cache = MiKVCache(MiKVConfig(**kw), N_LAYERS, torch.float32, 16)
        got = _teacher_forced_logits(model, ids, cache, 56)
        return ((got - ref).norm() / ref.norm()).item()

    e2, e3, e4 = (err(importance_ratio=0.2, lo_bits=b) for b in (2, 3, 4))
    assert e2 > e3 > e4 > 0
    assert err(importance_ratio=0.6, lo_bits=2) < e2
    assert err(importance_ratio=1.0, lo_bits=2) < 1e-6


# ---------------------------------------------------------------------------
# method factory (perf / quality runners)
# ---------------------------------------------------------------------------


def test_method_factory_sizes_by_bytes_unless_told_otherwise():
    model = _tiny_llama("sdpa")
    m = make_mikv_method(model=model, prefill_len=96, gen_len=32, budget_tokens=48,
                         dtype=torch.float32)
    assert m.sizing == "bytes-matched"
    assert m.config.cache_budget == pytest.approx(48 / 128)
    assert "bytes-matched" in m.describe()
    explicit = make_mikv_method(model=model, prefill_len=96, gen_len=32,
                                budget_tokens=48, importance_ratio=0.2)
    assert explicit.sizing == "explicit" and explicit.config.cache_budget is None
    assert make_mikv_method(model=model).sizing == "paper-default"
    with pytest.raises(TypeError, match="unknown MiKV"):
        make_mikv_method(model=model, importance_ration=0.2)
    with pytest.raises(MiKVConfigError, match="floor"):
        make_mikv_method(model=model, prefill_len=96, gen_len=32, budget_tokens=8)

    cache = m.new_cache()
    hooks = m.install_hooks(model, cache)
    try:
        with torch.no_grad():
            model.generate(torch.randint(0, 128, (1, 96)), max_new_tokens=4,
                           do_sample=False, past_key_values=cache, pad_token_id=0)
    finally:
        hooks.remove()
    rep = cache.memory_report()
    assert rep["tokens"] == 99
    assert rep["kv_fraction"] <= 48 / 128 + 1e-9
    assert math.isfinite(rep["total_fraction"])
