"""Tests for utils.sticky_metrics — Global LIR & absolute missed mass."""
from __future__ import annotations

import numpy as np
import pytest

from utils import sticky_metrics as SM


# ---------------------------------------------------------------------------
# flush_geometry
# ---------------------------------------------------------------------------


class TestFlushGeometry:
    def test_counts_grow_and_match_creation(self):
        # Trace index 0 is the PREFILL forward, so the cache holds prefill_len
        # tokens there: prefill 16, sink 0, window 8 → ceil(16/8) = 2 windows.
        w_act, ew_act, creation = SM.flush_geometry(
            num_steps=10, num_windows=8, prefill_len=16,
            num_sink=0, window_size=8, local_windows=1,
        )
        assert w_act[0] == 2                  # ceil((16+0)/8) = 2
        assert (np.diff(w_act) >= 0).all()    # non-decreasing
        assert (ew_act == np.maximum(w_act - 1, 0)).all()
        # creation[k] is the first flush where the window is valid.
        for k in range(int(w_act.max())):
            first = int(creation[k])
            assert w_act[first] > k
            if first > 0:
                assert w_act[first - 1] <= k

    def test_index_zero_is_the_prefill_forward(self):
        """The whole off-by-one, pinned.

        The parity runners record one entry per ``model(...)`` call and the
        first is the prompt, so index t holds prefill_len + t tokens. The old
        ``+1`` form modelled index 0 as the first decode step and put every
        window boundary one flush early — verified wrong against a recorded
        base run, where the 24 -> 25 transition sits at t=5, not t=4.
        """
        w_act, _, _ = SM.flush_geometry(
            num_steps=10, num_windows=64, prefill_len=192,
            num_sink=4, window_size=8, local_windows=2,
        )
        # post-sink at t: 188 + t; 24 windows until it exceeds 192, i.e. t=5.
        assert w_act[:6].tolist() == [24, 24, 24, 24, 24, 25]

    def test_cap_at_num_windows(self):
        w_act, _, _ = SM.flush_geometry(
            num_steps=5, num_windows=2, prefill_len=100,
            num_sink=0, window_size=8, local_windows=0,
        )
        assert (w_act <= 2).all()


# ---------------------------------------------------------------------------
# lir_counts — hand-checked example
# ---------------------------------------------------------------------------


class TestLirCounts:
    def test_known_selection_matrix(self):
        # 5 flushes, 3 windows, all created at t=0, m=2.
        sel = np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 0, 0],
            ],
            dtype=bool,
        )
        creation = np.zeros(3, dtype=int)
        w_act = np.full(5, 3, dtype=int)
        eligible, rescued = SM.lir_counts(sel, creation, w_act, m=2)
        # Worked out by hand: eligible pairs = 3, rescued = 2.
        assert eligible == 3
        assert rescued == 2

    def test_never_ignored_means_zero(self):
        sel = np.ones((6, 4), dtype=bool)
        creation = np.zeros(4, dtype=int)
        w_act = np.full(6, 4, dtype=int)
        eligible, rescued = SM.lir_counts(sel, creation, w_act, m=3)
        assert eligible == 0
        assert rescued == 0


# ---------------------------------------------------------------------------
# simulate_policy
# ---------------------------------------------------------------------------


class TestSimulatePolicy:
    def test_budget_covers_all_means_zero_missed(self):
        T, W = 4, 5
        masses = np.random.RandomState(0).rand(T, W)
        w_act = np.full(T, W, dtype=int)
        ew_act = np.full(T, W, dtype=int)   # no local tail
        sel, missed, _ = SM.simulate_policy(masses, history_budget_K=W, w_act=w_act,
                                            ew_act=ew_act, is_sticky=True)
        assert np.allclose(missed, 0.0)
        assert sel.all()

    def test_local_tail_always_retained(self):
        T, W = 3, 5
        masses = np.zeros((T, W))
        w_act = np.full(T, W, dtype=int)
        ew_act = np.full(T, 3, dtype=int)   # last 2 windows are local
        sel, missed, _ = SM.simulate_policy(masses, history_budget_K=0, w_act=w_act,
                                            ew_act=ew_act, is_sticky=True)
        # Budget 0 → no evictable kept, but local tail [3,5) always kept.
        assert sel[:, 3:].all()
        assert not sel[:, :3].any()

    def test_missed_mass_is_unselected_evictable_mass(self):
        # One flush, 4 evictable windows, keep top-2 by mass.
        masses = np.array([[0.4, 0.1, 0.3, 0.2]])
        w_act = np.array([4]); ew_act = np.array([4])
        sel, missed, _ = SM.simulate_policy(masses, history_budget_K=2, w_act=w_act,
                                            ew_act=ew_act, is_sticky=False)
        # Top-2 are windows 0 (0.4) and 2 (0.3); missed = 0.1 + 0.2 = 0.3.
        assert sel[0].tolist() == [True, False, True, False]
        assert missed[0] == pytest.approx(0.3)

    def test_sticky_caps_at_one_swap_per_flush(self):
        # Sticky-K swaps at most ONE window per flush; Fresh-K re-picks the
        # whole top-K.  When two cached windows both fall below two outsiders in
        # a single flush, Fresh replaces both but Sticky only replaces the
        # weakest, so the two policies diverge.
        masses = np.array([
            [1.0, 0.9, 0.0, 0.0],   # fill sticky -> {0, 1}
            [0.0, 0.1, 1.0, 0.9],   # fresh -> {2, 3}; sticky swaps only 0->2 -> {1, 2}
        ])
        w_act = np.array([4, 4]); ew_act = np.array([4, 4])
        sel_s, _, _ = SM.simulate_policy(masses, 2, w_act, ew_act, is_sticky=True)
        sel_f, _, _ = SM.simulate_policy(masses, 2, w_act, ew_act, is_sticky=False)
        assert sel_s[1].tolist() == [False, True, True, False]   # {1, 2}
        assert sel_f[1].tolist() == [False, False, True, True]   # {2, 3}


# ---------------------------------------------------------------------------
# compute_sticky_metrics — shapes, ranges, granularity consistency
# ---------------------------------------------------------------------------


class TestComputeStickyMetrics:
    def _run(self):
        rng = np.random.RandomState(7)
        S, T, L, H, W = 2, 12, 3, 4, 6
        base_ws = rng.rand(S, T, L, H, W).astype(np.float32)
        return SM.compute_sticky_metrics(
            base_ws, prefill_len=16, num_sink=0, window_size=8,
            local_windows=1, history_budget_K=2, m=3,
        ), (S, T, L, H, W)

    def test_shapes(self):
        out, (S, T, L, H, W) = self._run()
        assert out["global_lir"].shape == ()
        assert out["lir_per_layer"].shape == (L,)
        assert out["lir_per_head"].shape == (L, H)
        assert out["missed_mass"].shape == (T,)
        assert out["missed_mass_per_layer"].shape == (T, L)
        assert out["missed_mass_fresh"].shape == (T,)
        assert out["missed_mass_total"].shape == ()

    def test_ranges(self):
        out, _ = self._run()
        assert 0.0 <= float(out["global_lir"]) <= 1.0
        assert ((out["lir_per_layer"] >= 0) & (out["lir_per_layer"] <= 1)).all()
        assert ((out["lir_per_head"] >= 0) & (out["lir_per_head"] <= 1)).all()
        assert (out["missed_mass"] >= 0).all()
        assert (out["missed_mass_fresh"] >= 0).all()

    def test_fresh_never_misses_more_than_sticky(self):
        # Fresh-K is greedy-optimal per flush, so it can never leave more mass
        # behind than Sticky-K on the same masses.
        out, _ = self._run()
        assert (out["missed_mass_fresh"] <= out["missed_mass"] + 1e-9).all()

    def test_missed_mass_total_matches_trajectory(self):
        out, _ = self._run()
        assert float(out["missed_mass_total"]) == pytest.approx(
            float(out["missed_mass"].mean())
        )

    def test_n_q_zero_collapses_to_single_tier(self):
        # The Q tier off (default) must leave every legacy series untouched and
        # recover nothing — the q == 0 back-compat guarantee.
        out, _ = self._run()   # n_q defaults to 0
        assert np.allclose(out["missed_mass_kept"], out["missed_mass"])
        assert np.allclose(out["recovered_mass_q"], 0.0)
        assert float(out["recovered_mass_q_total"]) == pytest.approx(0.0)

    def test_q_tier_recovers_nonnegative_mass(self):
        rng = np.random.RandomState(7)
        base_ws = rng.rand(2, 12, 3, 4, 6).astype(np.float32)
        common = dict(prefill_len=16, num_sink=0, window_size=8,
                      local_windows=1, m=3)
        # fp budget 1 + Q budget 2 vs fp-only budget 1.
        two = SM.compute_sticky_metrics(base_ws, history_budget_K=1, n_q=2, **common)
        # kept never misses more than fp; recovered is exactly the difference.
        assert (two["missed_mass_kept"] <= two["missed_mass"] + 1e-9).all()
        assert np.allclose(
            two["recovered_mass_q"], two["missed_mass"] - two["missed_mass_kept"])
        assert float(two["recovered_mass_q_total"]) > 0   # real mass gets rescued
