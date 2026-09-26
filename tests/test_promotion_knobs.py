"""The two promotion ablations, pinned on the real eviction.

* ``quant_promotion="oneway"`` — ``F -> Q -> E``: an int2 window may be kept or
  evicted, never promoted back to fp. Bidir vs OneWay is only a clean ablation
  if everything else is equal, so the tier SIZES must match exactly (they come
  from the budget resolver, which does not know about promotion).
* ``quant_promote_source="original"`` — the evaluation-only oracle: a promoted
  window comes back bit-identical to what was demoted, from an fp shadow held
  outside the byte budget. ``"dequant"`` (shipped) returns the int2
  reconstruction, which must differ — otherwise the oracle arm proves nothing.

The cache runs its real compiled eviction on CPU (one Inductor compile per
configuration, ~1-2 min each), fed synthetic scores whose increments grow
geometrically so every eviction re-ranks the band from scratch — that is what
makes promotions happen under ``bidir`` and makes the test non-vacuous.
"""

from __future__ import annotations

import types

import pytest
import torch

from modules.windowed_cache.config import WindowedCacheConfig
from modules.windowed_cache.policy import EvictionPolicy


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_are_the_shipped_method(self):
        c = WindowedCacheConfig(window_size=4, num_sink_tokens=0,
                                local_window_size=4, cache_budget=0.35,
                                quant_ratio=0.5)
        assert (c.quant_promotion, c.quant_promote_source) == ("bidir", "dequant")

    @pytest.mark.parametrize("kw", [{"quant_promotion": "never"},
                                    {"quant_promote_source": "fp32"}])
    def test_bad_values_raise(self, kw):
        with pytest.raises(ValueError):
            WindowedCacheConfig(window_size=4, num_sink_tokens=0,
                                local_window_size=4, cache_budget=0.35,
                                quant_ratio=0.5, **kw)

    def test_the_oracle_warns_it_is_an_oracle(self):
        with pytest.warns(UserWarning, match="OUTSIDE the byte budget"):
            WindowedCacheConfig(window_size=4, num_sink_tokens=0,
                                local_window_size=4, cache_budget=0.35,
                                quant_ratio=0.5, quant_promote_source="original")

    def test_experiment_config_validates_and_the_factory_threads_it(self):
        from utils.cache_factory import (quant_tier_policy_kwargs,
                                         quant_tier_policy_record)
        from utils.config import CacheConfig, ConfigValidationError
        with pytest.raises(ConfigValidationError):
            CacheConfig(quant_promotion="sideways")
        assert quant_tier_policy_kwargs(CacheConfig()) == {}
        c = CacheConfig(quant_promotion="oneway", quant_promote_source="original")
        assert quant_tier_policy_kwargs(c) == {
            "quant_promotion": "oneway", "quant_promote_source": "original"}
        assert quant_tier_policy_record(CacheConfig()) == {
            "quant_promotion": "bidir", "quant_promote_source": "dequant"}

    def test_every_cache_building_runner_threads_the_knobs(self):
        import pathlib
        for f in ("longbench_runner", "ruler_runner", "gsm8k_runner",
                  "ours_parity_runner"):
            src = pathlib.Path(f"modules/evaluation/{f}.py").read_text()
            assert "quant_tier_policy_kwargs(cfg.cache)" in src, f
            assert "quant_tier_policy_record(cfg.cache)" in src, f


# ---------------------------------------------------------------------------
# the policy, in isolation
# ---------------------------------------------------------------------------


def _policy():
    r = types.SimpleNamespace(
        window_size=4, num_sink_tokens=0, local_tokens=8, top_k_windows=6,
        quant_ratio=0.5, top_k_fp=3, N_q=4, first_eviction_step=0)
    return EvictionPolicy(r)


class TestPolicyNoPromote:
    def test_blocked_windows_never_take_an_fp_slot(self):
        pol = _policy()
        g = torch.Generator().manual_seed(0)
        for _ in range(20):
            ws = torch.rand(2, 3, 14, generator=g)
            blocked = torch.rand(2, 14, generator=g) < 0.4
            idx, tier = pol.compute_two_tier_retain(ws, blocked)
            fp = torch.gather(blocked, 1, idx) & (tier == 0)
            local = idx >= 14 - 2
            assert not bool((fp & ~local).any())
            ref_idx, ref_tier = pol.compute_two_tier_retain(ws)
            assert idx.shape == ref_idx.shape
            assert int((tier == 1).sum()) == int((ref_tier == 1).sum())

    def test_nothing_blocked_is_the_shipped_ranking(self):
        pol = _policy()
        ws = torch.rand(2, 3, 14, generator=torch.Generator().manual_seed(1))
        a = pol.compute_two_tier_retain(ws)
        b = pol.compute_two_tier_retain(ws, torch.zeros(2, 14, dtype=torch.bool))
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ---------------------------------------------------------------------------
# the real cache
# ---------------------------------------------------------------------------


class _Rope(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x, position_ids):
        f = position_ids.to(torch.float32).unsqueeze(-1) * torch.arange(
            1, self.d // 2 + 1, dtype=torch.float32) * 0.01
        e = torch.cat([f, f], dim=-1)
        return e.cos(), e.sin()


L, B, H_KV, H_Q, D, WS, P, STEPS = 2, 1, 2, 4, 16, 4, 48, 30


def _drive(promotion: str, source: str):
    """Run STEPS decode steps; return per-eviction (fp ids, Q ids) per layer,
    the promoted windows, and everything needed to check their payload."""
    from modules.windowed_cache.cache import WindowedCache
    cfg = types.SimpleNamespace(num_key_value_heads=H_KV, num_attention_heads=H_Q,
                                hidden_size=H_Q * D, head_dim=D)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cache = WindowedCache(
            config=WindowedCacheConfig(
                window_size=WS, num_sink_tokens=0, local_window_size=WS,
                cache_budget=0.35, quant_ratio=0.5, first_eviction_step=0,
                quant_promotion=promotion, quant_promote_source=source),
            prefill_len=P, model_config=cfg, kv_dtype=torch.float32,
            rope_module=_Rope(D), num_layers=L, max_tokens=P + STEPS + 8)
    g = torch.Generator().manual_seed(7)
    fed_k, fed_v = {}, {}                  # (layer) -> [B, H, T, D] by position

    def merged(i):
        st, store = cache._states[i], cache._stores[i]
        return -(-st.seq_length // WS) + store.num_active_windows

    def scores(i, t):
        return torch.rand(B, H_Q, merged(i), generator=g) * (2.0 ** t)

    for i in range(L):
        k = torch.randn(B, H_KV, P, D, generator=g)
        v = torch.randn(B, H_KV, P, D, generator=g)
        fed_k[i], fed_v[i] = [k], [v]
        cache.update(k, v, i, cache_kwargs={"cache_position": torch.arange(P)})
    for i in range(L):
        cache.cache_kwargs[i]["window_scores"] = scores(i, 0)

    history = []                           # per step: {layer: (fp_ids, q_ids)}
    for t in range(STEPS):
        snap = {}
        for i in range(L):
            k = torch.randn(B, H_KV, 1, D, generator=g)
            v = torch.randn(B, H_KV, 1, D, generator=g)
            fed_k[i].append(k)
            fed_v[i].append(v)
            cache.update(k, v, i,
                         cache_kwargs={"cache_position": torch.tensor([P + t])})
            cache.cache_kwargs[i]["window_scores"] = scores(i, t + 1)
            st, store = cache._states[i], cache._stores[i]
            q_ids = set(store.active_ids()[0].tolist()) if store.active_ids() is not None else set()
            fp_ids = set(st.original_window_ids[0].tolist()) - q_ids
            snap[i] = (fp_ids, q_ids)
        history.append(snap)
    keys = {i: torch.cat(fed_k[i], dim=2) for i in range(L)}
    vals = {i: torch.cat(fed_v[i], dim=2) for i in range(L)}
    return cache, history, keys, vals


def _promotions(history):
    """``[(step, layer, window)]`` — windows that were Q and are fp one step later."""
    out = []
    for t in range(1, len(history)):
        for i, (fp_now, _) in history[t].items():
            _, q_before = history[t - 1][i]
            out += [(t, i, w) for w in fp_now & q_before]
    return out


@pytest.fixture(scope="module")
def runs():
    return {arm: _drive(*arm) for arm in
            (("bidir", "dequant"), ("oneway", "dequant"), ("bidir", "original"))}


class TestOneWay:
    def test_bidir_promotes_so_the_test_is_not_vacuous(self, runs):
        assert len(_promotions(runs[("bidir", "dequant")][1])) > 0

    def test_oneway_never_promotes(self, runs):
        assert _promotions(runs[("oneway", "dequant")][1]) == []

    def test_oneway_still_demotes_and_keeps_an_int2_tier(self, runs):
        hist = runs[("oneway", "dequant")][1]
        assert any(q for snap in hist for _, q in snap.values())

    def test_tier_sizes_match_exactly(self, runs):
        """Matched bytes: same fp and int2 window counts at every step."""
        a = runs[("bidir", "dequant")][1]
        b = runs[("oneway", "dequant")][1]
        for sa, sb in zip(a, b):
            for i in sa:
                assert len(sa[i][1]) == len(sb[i][1])
                assert len(sa[i][0]) == len(sb[i][0])
        ra, rb = runs[("bidir", "dequant")][0].resolved, runs[("oneway", "dequant")][0].resolved
        assert (ra.top_k_fp, ra.N_q, ra.bytes_per_q_window) == (
            rb.top_k_fp, rb.N_q, rb.bytes_per_q_window)


def _promoted_payload_error(run):
    """Max relative error of promoted windows' fp K and V against what was fed."""
    cache, hist, keys, vals = run
    errs = []
    for t, i, w in _promotions(hist):
        if t != len(hist) - 1 and w not in hist[-1][i][0]:
            continue                              # only windows still fp at the end
        st = cache._states[i]
        pos = st.position_ids[0]
        sel = (pos >= w * WS) & (pos < (w + 1) * WS)
        if int(sel.sum()) != WS:
            continue
        p = pos[sel]
        for got, fed in ((st.key_states[0][:, sel], keys[i][0][:, p]),
                         (st.value_states[0][:, sel], vals[i][0][:, p])):
            errs.append(float((got - fed).norm() / fed.norm()))
    return errs


class TestOriginalPayload:
    def test_the_oracle_returns_the_window_bit_identical(self, runs):
        errs = _promoted_payload_error(runs[("bidir", "original")])
        assert errs, "no promoted window survived to the end -- nothing checked"
        assert max(errs) == 0.0

    def test_dequant_returns_the_int2_reconstruction(self, runs):
        errs = _promoted_payload_error(runs[("bidir", "dequant")])
        assert errs, "no promoted window survived to the end -- nothing checked"
        assert min(errs) > 1e-3

    def test_the_shadow_exists_only_on_the_oracle_arm(self, runs):
        for arm, run in runs.items():
            store = run[0]._joint_store or run[0]._stores[0]
            assert store.shadow_enabled == (arm[1] == "original")
            assert bool(getattr(store.table, "shadow", False)) == (arm[1] == "original")
