"""The single-run observation collector (modules/evaluation/observation_collector.py).

Three things are pinned, because each would be silently wrong otherwise:

1. **The taps rebuild the model's own query.** With a full cache, the
   collector's ground truth (this query over the shadow KV) must reproduce the
   attention output the model actually computed, and the prompt's attention
   the model reports -- or every "missed mass" is mass of a different query.
2. **The recorder reads the cache correctly.** On the real cache and the real
   read gate (CPU reference; decode kernel stubbed), the recorded tiers are the
   store's, the gate's picks are int2 windows, the ranking signal is exactly
   what each eviction ranked on, and the output-error ladder isolates each
   cause (a gate-free read gives zero gate error; the promotion oracle gives
   zero payload error, dequantized promotion does not).
3. **The zip round-trips and feeds the observation suite.**

CPU-only. The real-cache runs pay the CPU Inductor compile of the eviction.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch

from modules.evaluation import observation_collector as OC


# ---------------------------------------------------------------------------
# 1. the taps, on a real (tiny) model with a full cache
# ---------------------------------------------------------------------------


def test_taps_rebuild_the_models_own_query_and_output():
    pytest.importorskip("transformers")
    from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(cfg).eval()
    H, Hk, D, P, G = 4, 2, 16, 40, 6
    cache = DynamicCache()
    shadow = OC.ShadowKV(2, Hk, P + G, D, torch.float32, torch.device("cpu"))
    OC.attach_shadow(cache, shadow)
    errs, prefill_q = [], {}

    def on_layer(layer, q, o):
        K, V = shadow.get(layer)
        if q.shape[1] > 1:
            prefill_q[layer] = q
            return
        o_full, p = OC.attend(q[:, 0], K, V, D ** -0.5, H // Hk)
        errs.append(float((o_full - o[:, 0]).abs().max()))
        assert torch.allclose(p.sum(-1), torch.ones(H), atol=1e-5)

    taps = OC.ModelTaps(model, H, D, on_layer)
    ids = torch.randint(0, 256, (1, P))
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                    output_attentions=True)
        for layer in range(2):
            K, _ = shadow.get(layer)
            mass = OC.prefill_token_mass(prefill_q[layer], K, D ** -0.5, H // Hk)
            ref = out.attentions[layer][0].sum(dim=1)            # [H, P]
            assert torch.allclose(mass, ref, atol=1e-4)
        for _ in range(G):
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
    taps.remove()
    assert len(errs) == 2 * G
    assert max(errs) < 1e-5, max(errs)
    assert shadow.n == [P + G, P + G]


def test_window_mass_and_merged_ids():
    tok = torch.arange(1, 12, dtype=torch.float32).repeat(2, 1)   # [2, 11]
    wm = OC.window_mass(tok, ns=3, ws=4, W=3)
    assert wm[0].tolist() == [4 + 5 + 6 + 7, 8 + 9 + 10 + 11, 0.0]
    ids = OC._merged_ids(torch.tensor([0, 3, 5]), 5)
    assert ids.tolist() == [0, 3, 5, 6, 7]
    assert OC._merged_ids(torch.tensor([0, 3, 5]), 2).tolist() == [0, 3]


# ---------------------------------------------------------------------------
# 2. the recorder, on the real cache and the real gate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1b. the decode loop hands the cache an absolute position
# ---------------------------------------------------------------------------


def _evicting_llama():
    """A tiny eager Llama over a real WindowedCache that evicts every step.

    Window scores come from a forward hook (random, correctly shaped) in place
    of the flash score kernel, which cannot run on CPU.
    """
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig
    import warnings

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(cfg).eval()
    ws, ns, prefill, gen = 4, 2, 48, 24
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cache = WindowedCache(
            config=WindowedCacheConfig(
                window_size=ws, num_sink_tokens=ns, local_window_size=ws,
                cache_budget=0.35, quant_ratio=0.5, first_eviction_step=0),
            prefill_len=prefill, model_config=cfg, kv_dtype=torch.float32,
            rope_module=model.model.rotary_emb, num_layers=2, max_tokens=gen)
    g = torch.Generator().manual_seed(1)

    def scores(layer):
        def hook(*_):
            st, store = cache._states[layer], cache._stores[layer]
            n_q = store.num_active_windows if store is not None else 0
            w = -(-(st.seq_length - ns) // ws) + int(n_q)
            cache.cache_kwargs.setdefault(layer, {})["window_scores"] = (
                torch.rand(1, cfg.num_attention_heads, w, generator=g))
        return hook

    for i, blk in enumerate(model.model.layers):
        blk.self_attn.register_forward_hook(scores(i))
    return model, cache, torch.randint(0, 256, (1, prefill)), gen


def test_decode_loop_passes_an_absolute_cache_position():
    """Regression: the collector's loop omitted ``cache_position``, so
    transformers derived it from the RETAINED key count, which runs backward at
    the first eviction -- on the GPU, "cache_position starts at 353 but 513
    tokens have been appended". ``forward_at`` passes it explicitly."""
    from modules.evaluation.ours_parity_runner import forward_at

    model, cache, ids, gen = _evicting_llama()
    with torch.no_grad():
        inp, pos = ids, 0
        for _ in range(gen):
            out, pos = forward_at(model, inp, cache, pos)
            inp = out.logits[:, -1].argmax(-1, keepdim=True)
    assert pos == ids.shape[1] + gen - 1                 # step 0 is the prefill
    assert cache._tokens_seen == pos
    assert cache.get_seq_length() < pos                    # it did evict

    model, cache, ids, gen = _evicting_llama()
    with torch.no_grad(), pytest.raises(RuntimeError,
                                        match="cache_position starts at"):
        inp = ids
        for _ in range(gen):
            out = model(input_ids=inp, past_key_values=cache, use_cache=True)
            inp = out.logits[:, -1].argmax(-1, keepdim=True)


class _Rope(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x, position_ids):
        f = position_ids.to(torch.float32).unsqueeze(-1) * torch.arange(
            1, self.d // 2 + 1, dtype=torch.float32) * 0.01
        e = torch.cat([f, f], dim=-1)
        return e.cos(), e.sin()


L, HK, HQ, D, WS, NS, P, STEPS = 2, 2, 4, 16, 4, 2, 48, 40
RATIO = 0.5


def _drive(source: str, monkeypatch):
    from modules.evaluation.ours_parity_runner import GateRecorder
    from modules.windowed_cache import flash_decode
    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig
    import warnings

    mcfg = types.SimpleNamespace(num_key_value_heads=HK, num_attention_heads=HQ,
                                 hidden_size=HQ * D, head_dim=D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cache = WindowedCache(
            config=WindowedCacheConfig(
                window_size=WS, num_sink_tokens=NS, local_window_size=WS,
                cache_budget=0.35, quant_ratio=0.5, first_eviction_step=0,
                quant_gate_ratio=RATIO, quant_promote_source=source),
            prefill_len=P, model_config=mcfg, kv_dtype=torch.float32,
            rope_module=_Rope(D), num_layers=L, max_tokens=STEPS)
    cache._fused_decode_active = True
    S_max = P + STEPS
    W = -(-(S_max - NS) // WS)
    T = STEPS + 1
    shadow = OC.ShadowKV(L, HK, S_max, D, torch.float32, torch.device("cpu"))
    OC.attach_shadow(cache, shadow)
    buf = OC.SampleBuffers(T, L, HQ, HK, W, S_max, False, torch.device("cpu"))
    gate = GateRecorder()
    gate.bind(cache)
    armed = []
    step = {"t": 0}
    g = torch.Generator().manual_seed(3)

    def stub_decode(q_hd, k_fp, v_fp, qtier, scaling, num_sink, n_body_win,
                    *, sel, logmass, centroids, vm_bits: int = 4):
        w = n_body_win + qtier["k_codes"].shape[1]
        return (torch.zeros_like(q_hd),
                torch.rand(q_hd.shape[0], q_hd.shape[1], w, generator=g)
                * (2.0 ** step["t"]))

    monkeypatch.setattr(flash_decode, "set_pending", armed.append)
    monkeypatch.setattr(flash_decode, "fused_two_tier_decode", stub_decode)
    flash_decode.set_gate_observer(gate)
    scale = D ** -0.5

    def merged(i):
        st, store = cache._states[i], cache._stores[i]
        return -(-st.seq_length // WS) + store.num_active_windows

    def held_output(i, q):
        st = cache._states[i]
        qt = OC.held_int2_tier(cache._stores[i], cache.rope_module, WS, D,
                               torch.float32)
        K, V = st.key_states[0].float(), st.value_states[0].float()
        if qt is not None:
            K, V = torch.cat([K, qt[0]], 1), torch.cat([V, qt[1]], 1)
        return OC.attend(q, K, V, scale, HQ // HK)[0]

    try:
        for i in range(L):
            k = torch.randn(1, HK, P, D, generator=g)
            cache.update(k, torch.randn_like(k), i,
                         cache_kwargs={"cache_position": torch.arange(P)})
            cache.cache_kwargs[i]["window_scores"] = torch.rand(
                1, HQ, merged(i), generator=g)
            OC.record_prefill_layer(buf, i, torch.randn(HQ, P, D, generator=g),
                                    cache, shadow, NS, WS, scale)
        for t in range(1, T):
            step["t"] = t
            gate.clear()
            if any(cache._policies[i].should_evict(cache._generation_step[i])
                   for i in range(L)):
                buf.evict_step[t] = True
                OC.snapshot_rank_signal(buf, t, cache, L)
            for i in range(L):
                k = torch.randn(1, HK, 1, D, generator=g)
                k_fp, v_fp = cache.update(
                    k, torch.randn_like(k), i,
                    cache_kwargs={"cache_position": torch.tensor([P + t - 1])})
                q = torch.randn(HQ, D, generator=g)
                if armed:
                    ctx = armed.pop()
                    flash_decode._run_fused(ctx, q.view(1, 1, HQ, D),
                                            k_fp.transpose(1, 2),
                                            v_fp.transpose(1, 2))
                else:
                    cache.cache_kwargs[i]["window_scores"] = torch.rand(
                        1, HQ, merged(i), generator=g) * (2.0 ** t)
                OC.record_decode_layer(buf, t, i, q, held_output(i, q), cache,
                                       shadow, gate, NS, WS, scale)
                for d in (gate.picks, gate.logmass, gate.active_ids):
                    d.pop(i, None)
    finally:
        flash_decode.clear_gate_observer()
    return cache, buf.to_numpy(), W


@pytest.fixture(scope="module")
def runs():
    mp = pytest.MonkeyPatch()
    try:
        return {src: _drive(src, mp) for src in ("dequant", "original")}
    finally:
        mp.undo()


class TestRecorder:
    def test_full_mass_is_each_heads_whole_attention(self, runs):
        _, a, _ = runs["dequant"]
        tot = a["full_mass"][1:].astype(np.float64).sum(-1) + a["sink_mass"][1:]
        np.testing.assert_allclose(tot, 1.0, atol=5e-3)

    def test_tiers_gate_and_cards_agree_with_the_store(self, runs):
        _, a, W = runs["dequant"]
        gated = a["gate_fired"]
        assert gated[1:].any(), "the gate never fired -- nothing checked"
        for t, li in zip(*np.nonzero(gated)):
            is_q = a["tier"][t, li] == OC.TIER_Q
            n = int(is_q.sum())
            assert n == a["n_q_windows"][t, li]
            for kv in range(HK):
                opened = a["gate_open"][t, li, kv]
                assert not (opened & ~is_q).any()
                assert opened.sum() == max(1, math.ceil(RATIO * n))
            card = np.isfinite(a["card_logmass"][t, li]).any(0)
            assert (card == is_q).all()

    def test_the_read_gate_rung_is_zero_when_nothing_is_gated_away(self, runs):
        """o_actual was the held full read, so the read-gate rung must vanish;
        eviction must cost something once windows are gone."""
        _, a, _ = runs["dequant"]
        dec = a["out_step_err"][1:]
        assert np.nanmax(dec[..., 3]) < 1e-5
        assert np.nanmax(dec[..., 0]) > 1e-3
        assert np.isfinite(a["out_err"][1:]).all()

    def test_promotion_payload_isolates_the_oracle(self, runs):
        _, deq, _ = runs["dequant"]
        _, orig, _ = runs["original"]
        assert deq["fp_dequant"].any(), "no promotion happened -- nothing checked"
        steps = np.nonzero(deq["fp_dequant"].any(axis=(1, 2)))[0]
        assert np.nanmax(deq["out_step_err"][steps][..., 2]) > 1e-4
        assert not orig["fp_dequant"].any()
        assert np.nanmax(orig["out_step_err"][1:][..., 2]) == 0.0

    def test_int2_reconstruction_error_is_recorded_once_per_window(self, runs):
        _, a, _ = runs["dequant"]
        seen = a["q_first_step"] >= 0
        assert seen.any()
        err = a["q_recon_err"]
        assert np.isfinite(err[:, :, :, 0].transpose(0, 2, 1)[seen]).all()
        assert (err[:, :, :, 0].transpose(0, 2, 1)[seen] > 0).all()
        assert np.isnan(err[:, :, :, 0].transpose(0, 2, 1)[~seen]).all()

    def test_rank_signal_is_what_the_eviction_ranked_on(self, runs):
        """After each eviction, every fp window outranks every int2 window,
        which outranks every window evicted at that event -- on the recorded
        signal's head mean, the quantity the policy ranks."""
        _, a, _ = runs["dequant"]
        checked = 0
        for r, t in enumerate(a["rank_signal_steps"]):
            for li in range(L):
                sig = a["rank_signal"][r, li].mean(0)
                now, before = a["tier"][t, li], a["tier"][t - 1, li]
                fp = sig[now == OC.TIER_FP]
                q = sig[now == OC.TIER_Q]
                gone = sig[(now == OC.TIER_EVICTED) & (before != OC.TIER_EVICTED)
                           & (before != OC.TIER_NONE)]
                if fp.size and q.size:
                    assert fp.min() >= q.max() - 1e-4 * abs(q.max())
                    checked += 1
                if q.size and gone.size:
                    assert q.min() >= gone.max() - 1e-4 * abs(gone.max())
        assert checked > 0


# ---------------------------------------------------------------------------
# 3. zip -> load -> export -> the observation suite
# ---------------------------------------------------------------------------


def test_zip_roundtrip_export_and_observations(runs, tmp_path):
    from modules.evaluation import qevict_observations as QO
    cache, arrays, W = runs["dequant"]
    arrays = dict(arrays, prompt_ids=np.zeros(P, np.int64))
    meta = {
        "schema_version": OC.SCHEMA_VERSION,
        "config": {"model_path": "synthetic", "seed": 0, "article_index": 0,
                   "quant_ratio": 0.5, "gate_ratio": RATIO,
                   "quant_budget_mode": "bytes", "quant_promotion": "bidir",
                   "quant_promote_source": "dequant", "cache_budget": 0.35},
        "dataset_resolved": {"corpus": "synthetic"},
        "model": {"num_layers": L, "num_heads": HQ, "num_kv_heads": HK,
                  "head_dim": D},
        "axes": {"T": STEPS + 1, "S_max": P + STEPS, "W": W, "window_size": WS,
                 "num_sink_tokens": NS, "prefill": P, "gen": STEPS},
        "resolved_geometry": {"top_k_fp": int(cache.resolved.top_k_fp),
                              "N_q": int(cache.resolved.N_q),
                              "top_k_windows": int(cache.resolved.top_k_windows),
                              "local_windows": 1},
        "read_gate": {"verdict": "gated", "read_fraction": RATIO},
        "samples": [{"file": "sample_000.npz", "article_index": 0,
                     "article_sha": "x"}],
    }
    w = OC.ZipWriter(tmp_path / "run.zip")
    w.add_sample("sample_000.npz", arrays)
    path = w.close(meta)
    assert path.exists() and not (tmp_path / "run.zip.partial").exists()
    meta2, samples = OC.load_zip(path)
    assert meta2["axes"]["W"] == W
    for k in ("full_mass", "tier", "gate_open", "out_step_err", "rank_signal"):
        np.testing.assert_array_equal(samples[0][k], arrays[k])

    base, ours = OC.export_parity(path, tmp_path / "parity")
    res = QO.run_observations(base, ours, tmp_path / "obs", fmm_horizon=4,
                              n_boot=20, figures=False, primary_inactivity=2,
                              primary_lir_horizon=2)
    assert res["inputs"].diagnostics["tier_resurrections"] == 0
    rs = res["observation4"]["recall_summary"]
    assert rs["gated"] and rs["head_mass_source"] == "recorded_head_step_mass"
    assert res["observation5"]["summary"]["events_scored"] > 0
