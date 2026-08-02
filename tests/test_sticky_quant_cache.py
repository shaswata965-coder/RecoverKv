
from __future__ import annotations

import numpy as np
import pytest
import torch

import utils.qevict_metrics as QM
from modules.windowed_cache.cache import WindowedCache
from modules.windowed_cache.config import WindowedCacheConfig
from modules.quant.effective import rotate_key_window

from tests.test_quant_cache import (
    _FakeModelConfig,
    _RealRoPE,
    _active,
    _merged_W,
    _seed_prefill_state,
)


def _make_cache(promote, quant_ratio=0.5, ws=4, num_sink=0, prefill_len=16, B=1):
    cfg = WindowedCacheConfig(
        window_size=ws, num_sink_tokens=num_sink, local_window_size=ws,
        cache_budget=0.5, quant_ratio=quant_ratio, quant_promotion=promote,
    )
    return WindowedCache(
        config=cfg, prefill_len=prefill_len, model_config=_FakeModelConfig(),
        kv_dtype=torch.float32, rope_module=_RealRoPE(4),
        num_layers=1, max_tokens=16,
    )


def _seed_two_tier(cache, ws=4):
    """Run one eviction that parks window 1 in the Q tier."""
    _seed_prefill_state(cache, n_win=4, ws=ws)
    st = cache._states[0]
    st.window_scores = torch.zeros(1, 2, 4)
    st.window_scores[0, :, [0, 1, 2]] = torch.tensor([100.0, 50.0, 10.0])
    st.original_window_ids = torch.tensor([[0, 1, 2, 3]])
    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1
    cache._evict_two_tier(0, step=2)
    assert _active(cache._stores[0]) == [1]
    return st, pol


def _rescore_favouring_the_q_window(st):
    """Window 1 (in Q) now outscores window 0 (in fp)."""
    st.window_scores = torch.zeros(1, 2, 3)
    st.window_scores[0, :, [0, 1]] = torch.tensor([10.0, 100.0])
    st.original_window_ids = torch.tensor([[0, 1, 3]])


class TestConfig:

    def test_promotion_is_on_by_default(self):
        cfg = WindowedCacheConfig(
            window_size=4, num_sink_tokens=0, local_window_size=4,
            cache_budget=0.5, quant_ratio=0.5,
        )
        assert cfg.quant_promotion is True
        assert _make_cache(promote=True)._promote is True

    def test_flag_reaches_policy_and_cache(self):
        cache = _make_cache(promote=False)
        assert cache._promote is False
        assert cache.resolved.quant_promotion is False
        assert cache._policies[0].quant_promotion is False

    def test_non_bool_is_rejected(self):
        with pytest.raises(ValueError, match="quant_promotion must be a bool"):
            WindowedCacheConfig(
                window_size=4, num_sink_tokens=0, local_window_size=4,
                cache_budget=0.5, quant_promotion="no",
            )


class TestNoPromotion:

    def test_top_scoring_q_window_stays_quantized(self):
        cache = _make_cache(promote=False)
        st, pol = _seed_two_tier(cache)
        store = cache._stores[0]

        _rescore_favouring_the_q_window(st)
        pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1
        cache._evict_two_tier(0, step=4)

        # window 1 outscored window 0 but is already int2, so it stays there and
        # window 0 keeps the single fp seat.
        assert _active(store) == [1]
        assert st.position_ids[0].tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
        store.validate()

    def test_promotion_path_does_the_opposite(self):
        cache = _make_cache(promote=True)
        st, pol = _seed_two_tier(cache)
        store = cache._stores[0]

        _rescore_favouring_the_q_window(st)
        pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1
        cache._evict_two_tier(0, step=4)

        assert _active(store) == [0]
        assert st.position_ids[0].tolist() == [4, 5, 6, 7, 12, 13, 14, 15]

    def test_low_scoring_q_window_is_evicted_not_promoted(self):
        cache = _make_cache(promote=False)
        st, pol = _seed_two_tier(cache)
        store = cache._stores[0]

        # window 1 is now worthless; with N_q = 0 its only exit is eviction.
        st.window_scores = torch.zeros(1, 2, 3)
        st.window_scores[0, :, [0, 1]] = torch.tensor([100.0, 0.0])
        st.original_window_ids = torch.tensor([[0, 1, 3]])
        pol.top_k_fp, pol.N_q, pol.local_windows = 1, 0, 1
        cache._evict_two_tier(0, step=4)

        assert _active(store) == []
        assert int(store.table.n_live[0]) == 0, "an evicted window kept its slot"
        assert st.position_ids[0].tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
        store.validate()

    def test_no_dormant_entries_are_ever_left_behind(self):
        cache = _make_cache(promote=False)
        st, pol = _seed_two_tier(cache)
        store = cache._stores[0]

        _rescore_favouring_the_q_window(st)
        pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1
        cache._evict_two_tier(0, step=4)

        # sticky Q means live == active: nothing is parked for a re-demotion.
        assert int(store.table.n_live[0]) == int(store.table.slot_active[0].sum())

    def test_fp_seats_are_capped_by_the_non_q_pool(self):
        cache = _make_cache(promote=False)
        pol = cache._policies[0]
        pol.top_k_fp, pol.N_q, pol.local_windows = 3, 2, 1

        # 5 evictable windows, 2 of them already int2 -> at most 3 fp seats.
        assert pol.tier_counts(6, n_q_resident=0) == (3, 2, 1)
        assert pol.tier_counts(6, n_q_resident=2) == (3, 2, 1)
        # only 2 windows are still in fp, so the third seat cannot be filled by
        # promotion; it converts into a Q seat instead.
        assert pol.tier_counts(6, n_q_resident=3) == (2, 2, 1)


class TestStickySelection:

    def _split(self, order, q_resident, k_fp, n_q):
        from modules.windowed_cache.policy import EvictionPolicy

        return EvictionPolicy._sticky_tier_split(
            torch.tensor([order]), torch.tensor([q_resident]), k_fp, n_q
        )

    def test_fp_seats_skip_q_residents_in_score_order(self):
        # score order 3 > 1 > 0 > 2; windows 0 and 3 are already int2.
        fp, q = self._split([3, 1, 0, 2], [True, False, False, True], 1, 2)
        assert fp[0].tolist() == [1]
        assert q[0].tolist() == [3, 0]

    def test_selection_is_exact_when_every_candidate_is_resident(self):
        fp, q = self._split([2, 0, 1], [True, True, True], 0, 2)
        assert fp.shape == (1, 0)
        assert q[0].tolist() == [2, 0]

    def test_matches_plain_topk_when_nothing_is_resident(self):
        fp, q = self._split([3, 1, 0, 2], [False] * 4, 2, 1)
        assert fp[0].tolist() == [3, 1]
        assert q[0].tolist() == [0]


def _drive_sticky(cache, prefill=16, steps=16, ws=4, num_sink=0, H=2, D=4, B=1):
    """Full update() loop; returns per batch row a per-step {window id: tier}.

    Rows route independently — each carries its own scores — so tier histories
    are only meaningful per row.
    """
    torch.manual_seed(3)
    trace = [[] for _ in range(B)]
    W0 = _merged_W(cache, ws, num_sink, prefill)
    cache.update(
        torch.randn(B, H, prefill, D), torch.randn(B, H, prefill, D), 0,
        cache_kwargs={"cache_position": torch.arange(prefill),
                      "window_scores": torch.rand(B, H, W0)},
    )
    for t in range(prefill, prefill + steps):
        W = _merged_W(cache, ws, num_sink, cache._states[0].seq_length + 1)
        cache.update(
            torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), 0,
            cache_kwargs={"cache_position": torch.arange(t, t + 1),
                          "window_scores": torch.rand(B, H, W)},
        )
        store = cache._stores[0]
        store.validate()
        assert store.num_active_windows <= cache._policies[0].N_q
        ids = cache._states[0].original_window_ids
        q_ids = store.active_ids()
        for b in range(B):
            live = ids[b].tolist()
            q = set() if q_ids is None else set(q_ids[b].tolist())
            trace[b].append({w: (1 if w in q else 0) for w in live})
    return trace


class TestEndToEnd:

    @pytest.mark.parametrize("B", [1, 3])
    def test_a_window_never_goes_back_to_fp(self, B):
        cache = _make_cache(promote=False, B=B)
        cache._policies[0].top_k_fp = 2
        cache._policies[0].N_q = 2
        cache._policies[0].local_windows = 1
        trace = _drive_sticky(cache, B=B)

        assert any(1 in tags.values() for row in trace for tags in row), \
            "Q tier never used"
        for b, row in enumerate(trace):
            was_q: set = set()
            for tags in row:
                for wid, tier in tags.items():
                    assert not (wid in was_q and tier == 0), (
                        f"row {b}: window {wid} was promoted back to fp")
                was_q |= {w for w, t in tags.items() if t == 1}

    def test_promotion_run_does_revive_windows(self):
        cache = _make_cache(promote=True)
        cache._policies[0].top_k_fp = 2
        cache._policies[0].N_q = 2
        cache._policies[0].local_windows = 1
        trace = _drive_sticky(cache)[0]

        was_q: set = set()
        revived = False
        for tags in trace:
            revived |= any(w in was_q and t == 0 for w, t in tags.items())
            was_q |= {w for w, t in tags.items() if t == 1}
        assert revived, "control run never promoted — the contrast is untested"

    def test_budget_and_length_bookkeeping_are_unchanged(self):
        ws, num_sink, prefill = 4, 0, 16
        cache = _make_cache(promote=False, ws=ws, num_sink=num_sink,
                            prefill_len=prefill)
        pol = cache._policies[0]
        pol.top_k_fp, pol.N_q, pol.local_windows = 2, 2, 1
        _drive_sticky(cache, prefill=prefill, ws=ws, num_sink=num_sink)

        tfp = cache._states[0].seq_length
        tq = cache._stores[0].num_active_tokens
        cap = num_sink + (pol.top_k_fp + pol.N_q + pol.local_windows + 2) * ws
        assert cache.get_seq_length(0) == tfp + tq <= cap
        p = cache._states[0].position_ids[0]
        assert torch.all(p[1:] > p[:-1])


class TestPromotionPathIsUnchanged:
    """The promoting default must compute exactly what it did before the flag.

    Everything else in `_evict_two_tier` is behind an `if self._promote` that
    takes the original branch; the one substantive edit is that the slot-table
    lookup moved from the retained subset to all windows, then gathers down.
    That is only safe if lookup-then-gather == gather-then-lookup.
    """

    @pytest.mark.parametrize("seed", range(8))
    def test_lookup_then_gather_equals_gather_then_lookup(self, seed):
        from modules.quant.store import QuantizedStore

        g = torch.Generator().manual_seed(seed)
        B, W, n_ret = 3, 11, 7
        store = QuantizedStore(window_size=4, head_dim=4, num_kv_heads=2,
                               n_slots=9)
        store.ensure(B, torch.device("cpu"))

        # scatter some live/active/dormant slots around
        table = store.table
        for b in range(B):
            for s in range(table.n_slots):
                r = torch.rand(1, generator=g).item()
                if r < 0.45:
                    table.slot_wid[b, s] = int(
                        torch.randint(0, W, (1,), generator=g))
                    table.slot_active[b, s] = bool(
                        torch.randint(0, 2, (1,), generator=g))
        # de-duplicate window ids per row: the table forbids duplicates
        for b in range(B):
            seen = set()
            for s in range(table.n_slots):
                w = int(table.slot_wid[b, s])
                if w != -1 and w in seen:
                    table.slot_wid[b, s] = -1
                    table.slot_active[b, s] = False
                elif w != -1:
                    seen.add(w)
        table.validate()

        all_wids = torch.stack([
            torch.randperm(W, generator=g)[:W] for _ in range(B)])
        retained = torch.stack([
            torch.sort(torch.randperm(W, generator=g)[:n_ret]).values
            for _ in range(B)])

        # old: look up only the retained windows
        old_has, old_q, old_slot, old_match = store.lookup(
            torch.gather(all_wids, 1, retained))

        # new: look up every window, then gather down to the retained ones
        a_has, a_q, a_slot, a_match = store.lookup(all_wids)
        n = a_match.shape[-1]
        new_has = torch.gather(a_has, 1, retained)
        new_q = torch.gather(a_q, 1, retained)
        new_slot = torch.gather(a_slot, 1, retained)
        new_match = torch.gather(
            a_match, 1, retained.unsqueeze(-1).expand(B, -1, n))

        assert torch.equal(old_has, new_has)
        assert torch.equal(old_q, new_q)
        assert torch.equal(old_slot, new_slot)
        assert torch.equal(old_match, new_match)
        # retain_only consumes match.any(1) — the only use of the mask
        assert torch.equal(old_match.any(1), new_match.any(1))


class TestEagerTwinParity:

    def _build(self, module):
        cfg_cls = module.config.WindowedCacheConfig
        cfg = cfg_cls(window_size=4, num_sink_tokens=0, local_window_size=4,
                      cache_budget=0.5, quant_ratio=0.5, quant_promotion=False)
        c = module.cache.WindowedCache(
            config=cfg, prefill_len=16, model_config=_FakeModelConfig(),
            kv_dtype=torch.float32, rope_module=_RealRoPE(4),
            num_layers=1, max_tokens=16,
        )
        c._policies[0].top_k_fp, c._policies[0].N_q, c._policies[0].local_windows = 2, 2, 1
        return c

    def test_both_backends_agree_under_sticky_q(self):
        import modules.windowed_cache as flash_mod
        import modules.windowed_eager_cache as eager_mod
        from modules.windowed_cache import cache as _f_cache  # noqa: F401
        from modules.windowed_eager_cache import cache as _e_cache  # noqa: F401

        flash, eager = self._build(flash_mod), self._build(eager_mod)
        assert flash._promote is False and eager._promote is False

        torch.manual_seed(7)
        B, H, D, prefill, ws = 2, 2, 4, 16, 4
        kp, vp = torch.randn(B, H, prefill, D), torch.randn(B, H, prefill, D)
        sc = torch.rand(B, H, _merged_W(flash, ws, 0, prefill))
        fk, _ = flash.update(kp.clone(), vp.clone(), 0, cache_kwargs={
            "cache_position": torch.arange(prefill), "window_scores": sc.clone()})
        ek, _ = eager.update(kp.clone(), vp.clone(), 0, cache_kwargs={
            "cache_position": torch.arange(prefill), "window_scores": sc.clone()})
        assert torch.equal(fk, ek)

        for t in range(prefill, prefill + 12):
            k1, v1 = torch.randn(B, H, 1, D), torch.randn(B, H, 1, D)
            sc = torch.rand(B, H, _merged_W(
                flash, ws, 0, flash._states[0].seq_length + 1))
            fk, fv = flash.update(k1.clone(), v1.clone(), 0, cache_kwargs={
                "cache_position": torch.arange(t, t + 1),
                "window_scores": sc.clone()})
            ek, ev = eager.update(k1.clone(), v1.clone(), 0, cache_kwargs={
                "cache_position": torch.arange(t, t + 1),
                "window_scores": sc.clone()})
            assert torch.equal(fk, ek) and torch.equal(fv, ev)
            assert (flash._stores[0].active_ids().tolist()
                    == eager._stores[0].active_ids().tolist())


class TestMetricsOnAStickyTrace:
    """The offline metrics read tier tags; sticky Q must not break them."""

    def _selected(self, trace, num_windows):
        # same encoding the runners emit: 1 = fp, 0 = dropped-or-Q, -1 = unborn
        sel = np.full((1, len(trace), num_windows), -1, dtype=np.int8)
        for r, tags in enumerate(trace):
            for wid, tier in tags.items():
                if wid < num_windows:
                    sel[0, r, wid] = 1 if tier == 0 else 0
        return sel

    def _sticky_trace(self):
        cache = _make_cache(promote=False)
        cache._policies[0].top_k_fp = 2
        cache._policies[0].N_q = 2
        cache._policies[0].local_windows = 1
        return _drive_sticky(cache)[0]

    def test_episode_lir_reports_zero_revivals_without_erroring(self):
        trace = self._sticky_trace()
        sel = self._selected(trace, num_windows=12)
        res = QM.episode_lir(sel, inactivity=1, horizon=None)
        assert res["eligible"] > 0, "no episode to score — trace too short"
        assert res["rescued"] == 0
        assert res["global_rate"] == 0.0
        # the empty ragged array is what m7 writes to the npz
        assert res["time_to_revival"].size == 0

    def test_ttr_quantiles_degrade_to_nan_not_a_crash(self):
        from modules.evaluation.tier_study.m7_global_lir import _ttr_quantiles

        trace = self._sticky_trace()
        res = QM.episode_lir(self._selected(trace, 12), inactivity=1, horizon=None)
        med, q1, q3 = _ttr_quantiles(res)
        assert all(np.isnan(v) for v in (med, q1, q3))

    def test_binary_transition_still_normalises(self):
        trace = self._sticky_trace()
        sel = self._selected(trace, num_windows=12)
        res = QM.binary_transition(sel, delta=1)
        pooled = res["pooled_probabilities"]
        assert res["pooled_counts"][0, 1] == 0, "a cold window went hot"
        for row in range(2):
            if res["pooled_counts"][row].sum() > 0:
                assert pooled[row].sum() == pytest.approx(1.0)

    def test_selection_churn_and_missed_mass_still_compute(self):
        trace = self._sticky_trace()
        sel = self._selected(trace, num_windows=12)
        acc = sel == 1
        churn = QM.selection_churn(acc)
        assert churn.shape == (1, len(trace) - 1)
        assert np.all(np.isfinite(churn))

        rng = np.random.default_rng(0)
        mass = rng.random((1, len(trace), 12))
        missed = QM.future_missed_mass(
            mass, acc, event_steps=list(range(len(trace))), horizon=1,
            require_full_horizon=False,
        )
        assert missed.shape == (1, len(trace))
        assert np.all((missed[np.isfinite(missed)] >= 0.0)
                      & (missed[np.isfinite(missed)] <= 1.0))
