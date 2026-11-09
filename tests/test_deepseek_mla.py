"""DeepSeek-V2 (MLA) port: the owned forward, the latent cache, and eviction fidelity.

What this file pins, and why each matters
------------------------------------------
* **The reference did not move.** The vendored upstream files hash to the
  pinned revision's.
* **The forward is upstream's.** The owned MLA forward (absorbed decode over a
  latent cache) produces upstream's logits and greedy tokens, on the vendored
  model, at full cache -- both with an HF ``DynamicCache`` holding the latent and
  with the RecoverKV cache at budget 1.0.
* **RoPE is upstream's, bit for bit**, including through the cache's own
  rotate/unrotate primitives over the packed layout, which must leave the latent
  dims untouched (an exact identity) and rotate only the RoPE dims.
* **The eviction evidence is the model's own attention.** The prefill window
  scores the cache receives equal upstream's eager attention probabilities summed
  over query rows, per head, reduced to windows -- the H2O cumulative score.
* **The eviction mechanism is the shipped one, unchanged by aliasing.** Driven by
  identical scores, a cache whose value aliases its key keeps exactly the windows,
  tiers, fp keys, int2 key codes, grids, cards and positions that the same cache
  storing an explicit copy of the value keeps. Aliasing removes the duplicate;
  it changes no decision.
* **The bytes are the latent's**: 1,152 B/token at V2-Lite's real config, the
  int2 window and card at the PLAN.md arithmetic, and the raw DeepSeek config is
  refused rather than priced at 8,192 B/token.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v2 import attention as mla_attention
from models.deepseek_v2.geometry import MLAGeometry, softmax_scale_from_config
from models.deepseek_v2.loader import install_recoverkv_mla, make_cache
from models.deepseek_v2.rope import PackedRope, apply_rotary, cos_sin, deinterleave
from models.deepseek_v2.upstream import modeling_deepseek as up
from models.deepseek_v2.upstream.configuration_deepseek import DeepseekV2Config
from modules.quant.effective import rotate_key_window, unrotate_key_window
from modules.windowed_cache.cache import WindowedCache
from modules.windowed_cache.config import WindowedCacheConfig

UPSTREAM = os.path.join(os.path.dirname(__file__), "..", "models", "deepseek_v2",
                        "upstream")
YARN = {"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096,
        "beta_fast": 32, "beta_slow": 1, "mscale": 0.707, "mscale_all_dim": 0.707}


def tiny_config(layers: int = 2) -> DeepseekV2Config:
    """V2-Lite's structure at toy width: MLA, YaRN, MoE after a dense layer 0.

    ``kv_lora_rank + qk_rope_head_dim = 64`` keeps the packed key a whole
    number of 32-entry grid groups, as V2-Lite's 576 is.
    """
    cfg = DeepseekV2Config(
        vocab_size=128, hidden_size=64, intermediate_size=96,
        moe_intermediate_size=32, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=4, kv_lora_rank=48,
        q_lora_rank=None, qk_rope_head_dim=16, qk_nope_head_dim=16,
        v_head_dim=16, n_routed_experts=4, num_experts_per_tok=2,
        n_shared_experts=1, first_k_dense_replace=1, moe_layer_freq=1,
        topk_method="greedy", n_group=1, topk_group=1,
        max_position_embeddings=4096, rope_scaling=dict(YARN),
        attn_implementation="eager")
    return cfg


def tiny_model(dtype=torch.float32, layers: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    return up.DeepseekV2ForCausalLM(tiny_config(layers)).to(dtype).eval()


def greedy(model, ids, steps, cache):
    """A decode loop that does what generate() does: masks and ABSOLUTE positions."""
    B, P = ids.shape
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                    attention_mask=torch.ones(B, P, dtype=torch.long),
                    position_ids=torch.arange(P).unsqueeze(0).expand(B, -1))
        logits = [out.logits[:, -1].float()]
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        toks = [tok]
        for t in range(steps):
            out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                        attention_mask=torch.ones(B, P + t + 1, dtype=torch.long),
                        position_ids=torch.full((B, 1), P + t))
            logits.append(out.logits[:, -1].float())
            tok = out.logits[:, -1].argmax(-1, keepdim=True)
            toks.append(tok)
    return torch.stack(logits, 1), torch.cat(toks, 1)


# ---------------------------------------------------------------------------
# the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,sha", [
    ("modeling_deepseek.py",
     "7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c"),
    ("configuration_deepseek.py",
     "6de5e7445ab490a16482027b131d5bd6a003da9b58a182a11432115fac84ce59"),
    ("config.json",
     "f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"),
])
def test_vendored_upstream_is_unmodified(name, sha):
    with open(os.path.join(UPSTREAM, name), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == sha


def _real_config() -> DeepseekV2Config:
    with open(os.path.join(UPSTREAM, "config.json")) as f:
        d = json.load(f)
    for k in ("architectures", "auto_map", "model_type", "transformers_version",
              "torch_dtype"):
        d.pop(k, None)
    return DeepseekV2Config(**d)


# ---------------------------------------------------------------------------
# geometry and bytes, at V2-Lite's real config
# ---------------------------------------------------------------------------


def test_softmax_scale_is_upstreams():
    """0.11472 on V2-Lite: q_head_dim**-0.5 * mscale**2, not any head dim's."""
    real = _real_config()
    s = softmax_scale_from_config(real)
    m = 0.1 * 0.707 * math.log(40) + 1
    assert s == pytest.approx(192 ** -0.5 * m * m, rel=1e-12)
    assert s == pytest.approx(0.11472, abs=5e-6)
    attn = up.DeepseekV2Attention(tiny_config(), layer_idx=0)
    assert attn.softmax_scale == pytest.approx(
        softmax_scale_from_config(tiny_config()), rel=1e-15)


def test_latent_bytes_at_the_real_config():
    """PLAN.md §3.1 arithmetic, through the resolver the cache actually uses."""
    geo = MLAGeometry.from_config(_real_config())
    assert (geo.key_dim, geo.half, geo.half_c, geo.half_r) == (576, 288, 256, 32)
    cfg = WindowedCacheConfig(window_size=8, num_sink_tokens=5,
                              local_window_size=128, cache_budget=0.20,
                              quant_ratio=0.70, quant_budget_mode="bytes",
                              quant_gate_ratio=0.25, first_eviction_step=0)
    r = cfg.resolve(4096, geo.cache_config(), torch.bfloat16, 256)
    assert r.bytes_per_token == 1152
    assert r.bytes_per_fp_window == 1152 * 8
    assert r.bytes_per_gate_card == 878          # mu int4 + v int8 + t int8, no vm
    assert r.bytes_per_q_window == 2376 + 878    # 1152 codes + 1224 grid + card
    # PLAN.md §3.2 said 182 from two floor divisions; resolve() also spends the
    # remainder (two more int2 windows), which is the number the cache holds.
    assert (r.top_k_fp, r.N_q) == (27, 184)
    assert r.budget_utilisation >= 0.99


def test_raw_deepseek_config_is_refused():
    """Its num_key_value_heads would price 8,192 B/token -- 7.1x the latent."""
    cfg = WindowedCacheConfig(window_size=8, num_sink_tokens=5,
                              local_window_size=128, cache_budget=0.2,
                              quant_ratio=0.7)
    with pytest.raises(ValueError, match="raw deepseek_v2 config"):
        cfg.resolve(4096, _real_config(), torch.bfloat16, 256)


def test_aliased_card_refuses_a_separate_centroid_width():
    geo = MLAGeometry.from_config(_real_config())
    cfg = WindowedCacheConfig(window_size=8, num_sink_tokens=5,
                              local_window_size=128, cache_budget=0.2,
                              quant_ratio=0.7, quant_card_bits={"vm": 8})
    with pytest.raises(ValueError, match="aliases its key"):
        cfg.resolve(4096, geo.cache_config(), torch.bfloat16, 256)


def test_pack_and_unpack_are_inverse():
    geo = MLAGeometry.from_config(_real_config())
    c, r = torch.randn(3, 512), torch.randn(3, 64)
    p = geo.pack(c, r)
    assert p.shape == (3, 576)
    assert torch.equal(geo.latent_of(p), c) and torch.equal(geo.rope_of(p), r)
    assert int(geo.rope_mask().sum()) == 64


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_is_upstreams_bit_for_bit(dtype):
    cfg = tiny_config()
    attn = up.DeepseekV2Attention(cfg, layer_idx=0).to(dtype)
    geo = MLAGeometry.from_config(cfg)
    torch.manual_seed(1)
    B, H, T, Dr = 2, 4, 10, 16
    pos = torch.stack([torch.arange(3, 13), torch.arange(1000, 1010)])
    q = torch.randn(B, H, T, Dr, dtype=dtype)
    k = torch.randn(B, 1, T, Dr, dtype=dtype)
    cos, sin = attn.rotary_emb(k, seq_len=2000)
    q_up, k_up = up.apply_rotary_pos_emb(q, k, cos, sin, pos)
    c2, s2 = cos_sin(attn.rotary_emb, pos, dtype)
    q_us, k_us = apply_rotary(q, k, c2, s2)
    assert torch.equal(q_up, q_us) and torch.equal(k_up, k_us)

    # Through the CACHE's primitives over the packed layout: RoPE dims rotate
    # exactly as upstream, latent dims do not move at all.
    lat = torch.randn(B, 1, T, cfg.kv_lora_rank, dtype=dtype)
    pre = geo.pack(lat, deinterleave(k))
    rope = PackedRope(attn.rotary_emb, geo)
    post = rotate_key_window(pre, pos, rope)
    assert torch.equal(geo.rope_of(post), k_up)
    assert torch.equal(geo.latent_of(post), lat)
    back = unrotate_key_window(post, pos, rope)
    assert torch.equal(geo.latent_of(back), lat)


def test_packed_rope_has_no_cat_in_the_compiled_region():
    """It runs inside the compiled eviction, where a cat once broke lowering."""
    import inspect
    src = inspect.getsource(PackedRope.forward)
    assert "torch.cat" not in src


# ---------------------------------------------------------------------------
# the forward against upstream
# ---------------------------------------------------------------------------


def test_owned_forward_matches_upstream_at_full_cache():
    """Latent cache + absorbed decode == upstream's decompressed cache, fp32."""
    from transformers import DynamicCache

    m = tiny_model()
    ids = torch.randint(0, 128, (2, 29), generator=torch.Generator().manual_seed(2))
    ref_logits, ref_toks = greedy(m, ids, 10, DynamicCache())
    h = install_recoverkv_mla(m)
    try:
        mla_attention.reset_path_stats()
        lat = DynamicCache()
        our_logits, our_toks = greedy(m, ids, 10, lat)
        assert torch.equal(our_toks, ref_toks)
        assert (our_logits - ref_logits).abs().max().item() < 1e-5
        # The baseline holds ONE latent head and no value bytes.
        assert tuple(lat.key_cache[0].shape) == (2, 1, 29 + 10, 64)
        assert lat.value_cache[0].shape[-1] == 0
        assert mla_attention.path_stats()["decode_latent_fullkv"] == 2 * 10

        cfg = WindowedCacheConfig(window_size=4, num_sink_tokens=2,
                                  local_window_size=8, cache_budget=1.0,
                                  quant_ratio=0.0, first_eviction_step=0)
        cache = make_cache(m, cfg, prefill_len=29, max_tokens=11)
        w_logits, w_toks = greedy(m, ids, 10, cache)
        assert torch.equal(w_toks, ref_toks)
        assert (w_logits - ref_logits).abs().max().item() < 1e-5
    finally:
        h.remove()


def test_install_refuses_a_non_deepseek_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    m = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                                     intermediate_size=32, num_hidden_layers=1,
                                     num_attention_heads=2))
    with pytest.raises(ValueError, match="not a deepseek_v2 model"):
        install_recoverkv_mla(m)


def test_prefill_scores_are_the_models_own_attention():
    """The H2O score handed to the cache == upstream's eager probabilities.

    Upstream's eager attention, run on the vendored module, returns its
    probabilities; summed over query rows per head and reduced to windows they
    must equal what the owned forward wrote to ``cache.cache_kwargs``.
    """
    from modules.windowed_cache.scorer import reduce_token_scores_to_windows

    m = tiny_model(layers=1)
    attn = m.model.layers[0].self_attn
    ids = torch.randint(0, 128, (2, 21), generator=torch.Generator().manual_seed(3))
    B, P = ids.shape
    hs = m.model.embed_tokens(ids)
    hs = m.model.layers[0].input_layernorm(hs)
    pos = torch.arange(P).unsqueeze(0).expand(B, -1)
    mask = torch.triu(torch.full((P, P), torch.finfo(torch.float32).min), 1)[None, None]
    with torch.no_grad():
        _, w_up, _ = attn(hs, attention_mask=mask.expand(B, 1, P, P),
                          position_ids=pos, output_attentions=True)
    h = install_recoverkv_mla(m)
    try:
        cfg = WindowedCacheConfig(window_size=4, num_sink_tokens=2,
                                  local_window_size=8, cache_budget=0.5,
                                  quant_ratio=0.7, first_eviction_step=0)
        cache = make_cache(m, cfg, prefill_len=P, max_tokens=8)
        with torch.no_grad():
            attn(hs, attention_mask=mask.expand(B, 1, P, P), position_ids=pos,
                 past_key_value=cache)
        got = cache.cache_kwargs[0]["window_scores"]
        want = reduce_token_scores_to_windows(w_up.float().sum(2), 2, 4)
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    finally:
        h.remove()


# ---------------------------------------------------------------------------
# RecoverKV on the latent
# ---------------------------------------------------------------------------


def test_recoverkv_runs_through_evictions_on_the_latent():
    """q=0.7 at budget 0.5: many evictions, layer-major, one latent stored once."""
    m = tiny_model()
    ids = torch.randint(0, 128, (2, 37), generator=torch.Generator().manual_seed(4))
    h = install_recoverkv_mla(m)
    try:
        cfg = WindowedCacheConfig(window_size=4, num_sink_tokens=2,
                                  local_window_size=8, cache_budget=0.5,
                                  quant_ratio=0.7, first_eviction_step=0)
        cache = make_cache(m, cfg, prefill_len=37, max_tokens=33)
        mla_attention.reset_path_stats()
        greedy(m, ids, 32, cache)
        st = cache._joint_store
        assert cache._joint is not None, "never reached the layer-major store"
        assert st.num_active_windows > 0
        assert st.value_alias and st.table.val_codes is None
        assert st.table.sk_vm_q is None                      # centroid is mu
        assert cache._joint._val_buf is None
        assert cache._joint.value_states is cache._joint.key_states
        st.validate()
        pos = cache._joint.position_ids
        assert bool((pos[:, 1:] > pos[:, :-1]).all()), "positions must stay monotonic"
        ps = mla_attention.path_stats()
        assert ps["prefill_eager"] == 2 and ps["decode_materialize"] == 2 * 32
        assert ps["decode_fused"] == 0          # CPU: the fused path is CUDA-only
        assert cache._attn_scaling == pytest.approx(
            m.model.layers[0].self_attn.softmax_scale)
    finally:
        h.remove()


class _MLACfg(SimpleNamespace):
    pass


def _eviction_twin(alias: bool, num_layers: int, D: int, H_q: int):
    """Two caches with the MLA geometry: aliased value vs an explicit copy."""
    # `tokens` mode: the twin also pays for its value bytes, so under `bytes`
    # the two would buy different tier sizes and not be a like-for-like pair.
    # Pinning the window counts makes every decision comparable one to one.
    cfg = WindowedCacheConfig(window_size=4, num_sink_tokens=2,
                              local_window_size=4, cache_budget=0.5,
                              quant_ratio=0.6, quant_budget_mode="tokens",
                              first_eviction_step=0)
    mc = _MLACfg(num_key_value_heads=1, num_attention_heads=H_q, head_dim=D,
                 hidden_size=H_q * D, kv_value_alias=alias,
                 attn_softmax_scale=0.1)

    class Rope(torch.nn.Module):
        def forward(self, x, position_ids):
            pos = position_ids.to(torch.float32)
            f = pos.unsqueeze(-1) * torch.arange(1, D // 2 + 1) * 0.01
            # latent-style identity on the first half of each half, RoPE on the rest
            f[..., : D // 4] = 0.0
            emb = torch.cat([f, f], -1)
            return emb.cos(), emb.sin()

    return WindowedCache(config=cfg, prefill_len=32, model_config=mc,
                         kv_dtype=torch.float32, rope_module=Rope(),
                         num_layers=num_layers, max_tokens=64)


def test_aliasing_changes_no_eviction_decision():
    """Same scores, same keys -> same windows, tiers, keys, codes, cards, positions.

    The aliased cache stores the latent once; the twin stores an explicit copy
    as its value. Every key-side tensor and every decision must agree exactly
    at every step, across promotions and demotions: aliasing removes the
    duplicate and nothing else.
    """
    L, D, H_q, ws, sink, P, steps = 2, 64, 4, 4, 2, 32, 40
    a = _eviction_twin(True, L, D, H_q)
    b = _eviction_twin(False, L, D, H_q)
    assert (a.resolved.top_k_fp, a.resolved.N_q) == (b.resolved.top_k_fp,
                                                     b.resolved.N_q)
    assert a.resolved.N_q > 0
    assert a.resolved.bytes_per_token * 2 == b.resolved.bytes_per_token
    kv = torch.Generator().manual_seed(11)
    sg = torch.Generator().manual_seed(12)

    def merged(cache, i):
        st, store = cache._states[i], cache._stores[i]
        body = max(st.seq_length - sink, 0)
        return -(-body // ws) + (store.num_active_windows if store is not None else 0)

    for i in range(L):
        k = torch.randn(1, 1, P, D, generator=kv)
        for c in (a, b):
            c.update(k, k.clone(), i, {"cache_position": torch.arange(P)})
    for t in range(steps):
        for i in range(L):
            W = merged(a, i)
            assert W == merged(b, i)
            s = torch.rand(1, H_q, W, generator=sg) ** 4      # skewed, like attention
            for c in (a, b):
                c.cache_kwargs[i]["window_scores"] = s.clone()
        for i in range(L):
            k = torch.randn(1, 1, 1, D, generator=kv)
            for c in (a, b):
                c.update(k, k.clone(), i, {"cache_position": torch.tensor([P + t])})
        for i in range(L):
            sa, sb = a._states[i], b._states[i]
            assert torch.equal(sa.key_states, sb.key_states), f"step {t} layer {i}"
            # (The twin's value is NOT its key after a promotion: it comes back
            # from its own per-token int2 codes. Only key-side state decides.)
            assert torch.equal(sa.position_ids, sb.position_ids)
            assert torch.equal(sa.original_window_ids, sb.original_window_ids)
            assert torch.equal(sa.window_scores, sb.window_scores)
            qa, qb = a._stores[i], b._stores[i]
            assert qa.num_active_windows == qb.num_active_windows
            if qa.num_active_windows:
                assert torch.equal(qa.active_ids(), qb.active_ids())
    # Not vacuous: the run must have crossed the tier boundary both ways.
    from modules.windowed_cache.cache import evict_width_stats
    wa, wb = evict_width_stats(a), evict_width_stats(b)
    assert wa == wb
    assert wa["evictions"] >= 5
    assert wa["promote_max"] > 0, "no int2 -> fp promotion was exercised"
    assert wa["fresh_max"] > 0, "no fresh demotion was exercised"
    ta, tb = a._joint_store.table, b._joint_store.table
    for f in ("key_codes", "key_scale_q", "key_scale_s", "key_zero_q", "key_zero_s",
              "slot_wid", "slot_active", "slot_pos",
              "sk_mu_q", "sk_mu_s", "sk_v_q", "sk_v_s", "sk_t_q", "sk_t_s"):
        assert torch.equal(getattr(ta, f), getattr(tb, f)), f
    assert ta.val_codes is None and tb.val_codes is not None
    assert torch.equal(a._joint_store._anchor, b._joint_store._anchor)
    # The aliased centroid is the key mean; the twin's separate vbar is the same
    # mean of the same tokens, encoded at its own scale.
    from modules.quant.sketch import Sketch, value_centroid
    idx = ta.active_order(a._joint_store.num_active_windows)
    ca = value_centroid(Sketch(*ta.gather_sketch(idx)), a._joint_store._v_anchor)
    cb = value_centroid(Sketch(*tb.gather_sketch(idx)), b._joint_store._v_anchor)
    assert (ca - cb).abs().max().item() < 0.25 * cb.abs().max().item()


def test_aliased_q_tier_reads_return_the_key_as_the_value():
    m = tiny_model()
    ids = torch.randint(0, 128, (1, 33), generator=torch.Generator().manual_seed(5))
    h = install_recoverkv_mla(m)
    try:
        cfg = WindowedCacheConfig(window_size=4, num_sink_tokens=2,
                                  local_window_size=8, cache_budget=0.5,
                                  quant_ratio=0.7, first_eviction_step=0)
        cache = make_cache(m, cfg, prefill_len=33, max_tokens=12)
        greedy(m, ids, 10, cache)
        k, v, pos = cache._joint_store.effective_q_tier(cache.rope_module,
                                                        torch.float32)
        assert v is k
        geo = m.model.layers[0].self_attn._recoverkv_geometry
        # promote returns pre-RoPE keys; their latent dims equal the rotated
        # read's latent dims exactly, because RoPE is an identity there.
        n = cache._joint_store.num_active_windows
        kc, ks, kz, vc, vs, vz, p = cache._joint_store.table.gather(
            cache._joint_store.table.active_order(n))
        assert vc is None and vs is None and vz is None
        from modules.quant.quantizer import dequantize_key_windows
        pre = dequantize_key_windows(kc, ks, kz, 4, out_dtype=torch.float32)
        R = k.shape[0]
        pre = pre.reshape(R, n, 1, 4, 64).permute(0, 2, 1, 3, 4).reshape(R, 1, n * 4, 64)
        assert torch.equal(geo.latent_of(pre), geo.latent_of(k))
    finally:
        h.remove()


def test_gate_selects_per_latent_head_over_all_query_heads():
    """rep = H_q: one KV head, the share union over every query head, one cap."""
    m = tiny_model()
    ids = torch.randint(0, 128, (2, 41), generator=torch.Generator().manual_seed(6))
    h = install_recoverkv_mla(m)
    try:
        cfg = WindowedCacheConfig(window_size=4, num_sink_tokens=2,
                                  local_window_size=8, cache_budget=0.5,
                                  quant_ratio=0.7, first_eviction_step=0,
                                  quant_gate_ratio=0.25)
        cache = make_cache(m, cfg, prefill_len=41, max_tokens=12)
        greedy(m, ids, 10, cache)
        st = cache._joint_store
        n = st.num_active_windows
        q = torch.randn(st.table.batch_size, 4, 64)
        keep, logmass, slots = st.gate_and_select(q, 0.1, 0.25)
        assert keep.shape == (st.table.batch_size, 1, n)
        assert logmass.shape == (st.table.batch_size, 4, n)
        assert bool((keep.sum(-1) == math.ceil(0.25 * n)).all())
        # The rule is the per-head share union (sketch.group_share), unchanged.
        from modules.quant.sketch import group_share, select_windows
        want = select_windows(group_share(logmass, 1), math.ceil(0.25 * n))
        assert torch.equal(keep, want)
        # centroid of a window = its decoded mu + anchor (the aliased vbar)
        cen = st.window_centroids(slots)
        assert cen.shape == (st.table.batch_size, n, 1, 64)
    finally:
        h.remove()


def test_runner_path_generate_matches_the_explicit_loop():
    """What every quality runner does: ``model.config`` + the model's rotary
    module straight into ``WindowedCache``, ``install_score_hooks``, then
    ``generate()``. The cache converts the raw config to the latent view, the
    hooks delegate to the owned forward, and the remote
    ``prepare_inputs_for_generation`` slices the prompt by ``seen_tokens`` --
    the monotonic count -- so generate() decodes exactly what a loop passing
    absolute positions decodes, through every eviction."""
    from modules.windowed_cache import install_score_hooks

    m = tiny_model()
    h_mla = install_recoverkv_mla(m)
    try:
        ids = torch.randint(0, 128, (2, 37), generator=torch.Generator().manual_seed(4))
        cfg = WindowedCacheConfig(window_size=4, num_sink_tokens=2,
                                  local_window_size=8, cache_budget=0.5,
                                  quant_ratio=0.7, first_eviction_step=0)

        def cache():
            return WindowedCache(config=cfg, prefill_len=37, model_config=m.config,
                                 kv_dtype=torch.float32,
                                 rope_module=m.model.layers[0].self_attn.rotary_emb,
                                 num_layers=2, max_tokens=21)

        c = cache()
        assert c._value_alias and c.resolved.bytes_per_token == 64 * 4
        assert isinstance(c.rope_module, PackedRope)
        hooks = install_score_hooks(m, c, cfg)
        try:
            out = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             past_key_values=c, max_new_tokens=20, do_sample=False,
                             pad_token_id=0)
        finally:
            hooks.remove()
        assert c._joint is not None and c._tokens_seen == 37 + 19
        _, toks = greedy(m, ids, 19, cache())
        assert torch.equal(out[:, 37:], toks)
    finally:
        h_mla.remove()


def test_hooks_refuse_a_deepseek_model_without_the_owned_forward():
    from modules.windowed_cache import install_score_hooks
    m = tiny_model(layers=1)
    cfg = WindowedCacheConfig(window_size=4, num_sink_tokens=2, local_window_size=8,
                              cache_budget=0.5, quant_ratio=0.7)
    c = WindowedCache(config=cfg, prefill_len=21, model_config=m.config,
                      kv_dtype=torch.float32,
                      rope_module=m.model.layers[0].self_attn.rotary_emb,
                      num_layers=1, max_tokens=4)
    with pytest.raises(RuntimeError, match="without the owned MLA attention forward"):
        install_score_hooks(m, c, cfg)


def test_output_attentions_match_upstream_at_full_cache():
    """The parity/observation runners read `out.attentions`; at full cache the
    owned forward's probabilities (absorbed, over the latent) are upstream's."""
    from transformers import DynamicCache

    m = tiny_model()
    ids = torch.randint(0, 128, (1, 17), generator=torch.Generator().manual_seed(8))

    def run():
        c = DynamicCache()
        with torch.no_grad():
            o1 = m(input_ids=ids, past_key_values=c, use_cache=True,
                   attention_mask=torch.ones_like(ids), output_attentions=True,
                   position_ids=torch.arange(17)[None])
            o2 = m(input_ids=ids[:, -1:], past_key_values=c, use_cache=True,
                   attention_mask=torch.ones(1, 18, dtype=torch.long),
                   output_attentions=True, position_ids=torch.tensor([[17]]))
        return o1.attentions, o2.attentions

    ref = run()
    h = install_recoverkv_mla(m)
    try:
        got = run()
    finally:
        h.remove()
    for a_ref, a_got in zip(ref, got):
        for x, y in zip(a_ref, a_got):
            torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-6)
