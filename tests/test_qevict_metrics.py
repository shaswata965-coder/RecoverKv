"""Tests for utils.qevict_metrics — concentration, FMM, churn, episode LIR."""
from __future__ import annotations

import numpy as np
import pytest

from utils import qevict_metrics as QM


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_single_trace_has_no_interval(self):
        mu, lo, hi = QM.bootstrap_mean_ci([0.4])
        assert mu == pytest.approx(0.4)
        assert np.isnan(lo) and np.isnan(hi)

    def test_ci_brackets_mean_and_is_deterministic(self):
        x = np.linspace(0.0, 1.0, 40)
        mu, lo, hi = QM.bootstrap_mean_ci(x, n_boot=500, seed=7)
        assert lo <= mu <= hi
        assert (mu, lo, hi) == QM.bootstrap_mean_ci(x, n_boot=500, seed=7)

    def test_all_nan_curve_returns_nan_not_raise(self):
        mean, lo, hi = QM.bootstrap_curve_ci(np.full((3, 5), np.nan))
        assert mean.shape == (5,) and np.isnan(mean).all()
        assert np.isnan(lo).all() and np.isnan(hi).all()

    def test_group_reduce_averages_within_group(self):
        vals = np.array([0.0, 1.0, 10.0, 20.0])
        out = QM.group_reduce(vals, np.array([0, 0, 1, 1]))
        assert out.tolist() == [0.5, 15.0]
        # None = identity (every trace its own group).
        assert QM.group_reduce(vals, None).tolist() == vals.tolist()

    def test_group_ratio_is_ratio_of_sums(self):
        # Group 0: 1+3 rescued of 2+10 eligible = 4/12, NOT mean(1/2, 3/10).
        out = QM.group_ratio(np.array([1, 3]), np.array([2, 10]),
                             np.array([0, 0]))
        assert out.tolist() == [pytest.approx(4 / 12)]

    def test_group_ratio_empty_denominator_is_nan(self):
        out = QM.group_ratio(np.array([0]), np.array([0]), np.array([0]))
        assert np.isnan(out).all()


# ---------------------------------------------------------------------------
# Observation I — concentration
# ---------------------------------------------------------------------------


class TestConcentrationCurve:
    def test_uniform_scores_lie_on_the_diagonal(self):
        scores = np.ones((1, 1, 10))
        valid = np.ones_like(scores, dtype=bool)
        res = QM.concentration_curve(scores, valid, report_fractions=(0.2, 0.5))
        assert res["mass_at"][0.2][0, 0] == pytest.approx(0.2)
        assert res["mass_at"][0.5][0, 0] == pytest.approx(0.5)

    def test_single_hot_window_captures_everything(self):
        scores = np.zeros((1, 1, 10))
        scores[0, 0, 3] = 1.0
        valid = np.ones_like(scores, dtype=bool)
        res = QM.concentration_curve(
            scores, valid, report_fractions=(0.1,), target_masses=(0.9,))
        assert res["mass_at"][0.1][0, 0] == pytest.approx(1.0)
        # One window out of ten suffices for 90% of the mass.
        assert res["coverage_at"][0.9][0, 0] == pytest.approx(0.1)

    def test_invalid_windows_are_excluded_from_the_total(self):
        # Two hot windows + eight invalid columns → top 50% of the *valid* two
        # holds half the mass, and padding cannot dilute it.
        scores = np.zeros((1, 1, 10))
        scores[0, 0, :2] = 1.0
        valid = np.zeros_like(scores, dtype=bool)
        valid[0, 0, :2] = True
        res = QM.concentration_curve(scores, valid, report_fractions=(0.5,))
        assert res["num_valid"][0, 0] == 2
        assert res["mass_at"][0.5][0, 0] == pytest.approx(0.5)

    def test_zero_mass_event_is_nan(self):
        res = QM.concentration_curve(
            np.zeros((1, 1, 4)), np.ones((1, 1, 4), bool),
            report_fractions=(0.5,))
        assert np.isnan(res["mass_at"][0.5][0, 0])

    def test_rejects_negative_scores(self):
        with pytest.raises(ValueError, match="non-negative"):
            QM.concentration_curve(-np.ones((1, 1, 2)), np.ones((1, 1, 2), bool))


# ---------------------------------------------------------------------------
# Observation II — FMM, churn, granularity
# ---------------------------------------------------------------------------


class TestFutureMissedMass:
    def _setup(self):
        # 4 windows, 6 steps.  All mass in the future lands on windows 0 and 1.
        mass = np.zeros((1, 6, 4))
        mass[0, 2:, 0] = 3.0
        mass[0, 2:, 1] = 1.0
        return mass

    def test_dropping_the_hot_window_costs_its_share(self):
        mass = self._setup()
        acc = np.zeros((1, 1, 4), dtype=bool)
        acc[0, 0, 1] = True                       # keep only window 1 (mass 1)
        fmm = QM.future_missed_mass(
            mass, acc, [1], horizon=4, creation_steps=np.zeros(4, int))
        assert fmm[0, 0] == pytest.approx(3 / 4)

    def test_keeping_everything_misses_nothing(self):
        mass = self._setup()
        acc = np.ones((1, 1, 4), dtype=bool)
        fmm = QM.future_missed_mass(
            mass, acc, [1], horizon=4, creation_steps=np.zeros(4, int))
        assert fmm[0, 0] == pytest.approx(0.0)

    def test_not_yet_created_windows_are_excluded_from_both_terms(self):
        # Window 0 exists; windows 1..3 are created later, so the mass on
        # window 1 must not count against a decision that could not keep it.
        mass = self._setup()
        creation = np.array([0, 5, 5, 5])
        acc = np.zeros((1, 1, 4), dtype=bool)
        fmm = QM.future_missed_mass(mass, acc, [1], horizon=4,
                                    creation_steps=creation)
        assert fmm[0, 0] == pytest.approx(1.0)    # only window 0 is eligible

    def test_censors_events_without_a_full_horizon(self):
        mass = self._setup()
        acc = np.ones((1, 2, 4), dtype=bool)
        fmm = QM.future_missed_mass(mass, acc, [1, 4], horizon=4,
                                    creation_steps=np.zeros(4, int))
        assert np.isfinite(fmm[0, 0]) and np.isnan(fmm[0, 1])
        # Truncation instead of censoring scores the late event anyway.
        trunc = QM.future_missed_mass(mass, acc, [1, 4], horizon=4,
                                      creation_steps=np.zeros(4, int),
                                      require_full_horizon=False)
        assert np.isfinite(trunc[0, 1])


class TestSelectionChurn:
    def test_stable_selection_has_zero_churn(self):
        acc = np.ones((1, 3, 5), dtype=bool)
        assert QM.selection_churn(acc).tolist() == [[0.0, 0.0]]

    def test_disjoint_selection_has_unit_churn(self):
        acc = np.zeros((1, 2, 4), dtype=bool)
        acc[0, 0, :2] = True
        acc[0, 1, 2:] = True
        assert QM.selection_churn(acc)[0, 0] == pytest.approx(1.0)

    def test_one_of_two_swapped_is_jaccard_distance_two_thirds(self):
        acc = np.zeros((1, 2, 4), dtype=bool)
        acc[0, 0, [0, 1]] = True
        acc[0, 1, [1, 2]] = True
        assert QM.selection_churn(acc)[0, 0] == pytest.approx(1 - 1 / 3)


class TestGranularityPolicies:
    def test_pool_scores_sums_blocks_and_pads(self):
        x = np.array([[1.0, 2.0, 3.0]])
        assert QM.pool_scores(x, 2).tolist() == [[3.0, 3.0]]
        assert QM.pool_scores(x, 1).tolist() == x.tolist()

    def test_fine_and_coarse_keep_the_same_unit_count(self):
        # 8 evictable windows, no local tail; budget 4 with pool 2 → both
        # policies retain exactly 4 units, so the comparison is byte-matched.
        rank = np.array([[[8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0]]])
        w_act = np.array([8]); ew_act = np.array([8])
        fine = QM.granularity_accessible(rank, [0], w_act, ew_act, 4, pool=1)
        coarse = QM.granularity_accessible(rank, [0], w_act, ew_act, 4, pool=2)
        assert fine.sum() == coarse.sum() == 4
        # Fine takes the four strongest units; coarse is forced to take their
        # weak neighbours too.
        assert fine[0, 0].tolist() == [True, False, True, False,
                                       True, False, True, False]
        assert coarse[0, 0].tolist() == [True, True, True, True,
                                         False, False, False, False]

    def test_local_tail_is_always_accessible(self):
        rank = np.zeros((1, 1, 6))
        w_act = np.array([6]); ew_act = np.array([4])
        acc = QM.granularity_accessible(rank, [0], w_act, ew_act, 0, pool=1)
        assert acc[0, 0].tolist() == [False] * 4 + [True, True]


# ---------------------------------------------------------------------------
# Observation III — episode LIR & transitions
# ---------------------------------------------------------------------------


class TestEpisodeLIR:
    def test_one_episode_rescued_within_horizon(self):
        # window 0: hit, miss, miss, hit  → one episode of length 2, eligible at
        # event 2 (m=2), rescued at event 3.
        sel = np.array([[[1], [0], [0], [1]]])          # [M=1, R=4, W=1]
        res = QM.episode_lir(sel, inactivity=2, horizon=1)
        assert res["eligible"] == 1 and res["rescued"] == 1
        assert res["global_rate"] == pytest.approx(1.0)
        assert res["time_to_revival"].tolist() == [1]

    def test_long_run_counts_once_not_per_pair(self):
        # Six consecutive misses then a hit: the *episode* view counts one
        # eligible episode; a pairwise view (sticky_metrics.lir_counts) would
        # count several.
        sel = np.array([[[1], [0], [0], [0], [0], [0], [0], [1]]])
        res = QM.episode_lir(sel, inactivity=2, horizon=5)
        assert res["eligible"] == 1 and res["rescued"] == 1
        assert res["time_to_revival"].tolist() == [5]   # eligible at 2, hit at 7

    def test_rescue_beyond_horizon_does_not_count(self):
        sel = np.array([[[1], [0], [0], [0], [0], [1], [1]]])
        near = QM.episode_lir(sel, inactivity=2, horizon=1)
        assert near["eligible"] == 1 and near["rescued"] == 0
        far = QM.episode_lir(sel, inactivity=2, horizon=3)
        assert far["eligible"] == 1 and far["rescued"] == 1

    def test_right_censored_episode_is_dropped_entirely(self):
        # Eligible at the very end: the horizon runs past the trace, so the
        # episode is excluded rather than scored as a failure.
        sel = np.array([[[1], [0], [0]]])
        res = QM.episode_lir(sel, inactivity=2, horizon=4)
        assert res["eligible"] == 0 and np.isnan(res["global_rate"])

    def test_unobservable_prefix_sets_creation(self):
        # -1 = not a candidate yet; the episode clock starts at the first 0/1.
        sel = np.array([[[-1], [-1], [0], [0], [1]]])
        res = QM.episode_lir(sel, inactivity=2, horizon=1)
        assert res["eligible"] == 1 and res["rescued"] == 1
        assert QM.infer_creation_events(sel).tolist() == [[2]]

    def test_per_trace_rates_are_independent(self):
        rescued = np.array([[[1], [0], [0], [1]]])
        never = np.array([[[1], [0], [0], [0]]])
        sel = np.concatenate([rescued, never], axis=0)
        res = QM.episode_lir(sel, inactivity=2, horizon=1)
        assert res["rate_by_trace"].tolist() == [1.0, 0.0]

    def test_rejects_non_trinary_input(self):
        with pytest.raises(ValueError, match="-1, 0, 1"):
            QM.episode_lir(np.array([[[2]]]), 1, 1)

    def test_matches_the_reference_loop_implementation(self):
        """The closed-form enumeration must equal the naive scan, exactly.

        ``episode_lir`` is vectorised (14x faster at parity scale, and the
        difference between seconds and minutes in the every-step fallback);
        this pins it to the obvious-but-slow definition it replaced.
        """
        def reference(x, inactivity, horizon):
            x = np.asarray(x, dtype=int)
            M, R, W = x.shape
            creation = QM.infer_creation_events(x)
            eligible = np.zeros(M, int)
            rescued = np.zeros(M, int)
            episodes = []
            for m in range(M):
                for w in range(W):
                    start = int(creation[m, w])
                    if start >= R or np.any(x[m, start:, w] < 0):
                        continue
                    r = start
                    while r < R:
                        if x[m, r, w] == 1:
                            r += 1
                            continue
                        run_start = r
                        while r + 1 < R and x[m, r + 1, w] == 0:
                            r += 1
                        if r - run_start + 1 >= inactivity:
                            elig = run_start + inactivity - 1
                            if elig + horizon < R:
                                hits = np.flatnonzero(
                                    x[m, elig + 1:elig + horizon + 1, w] == 1)
                                eligible[m] += 1
                                rescued[m] += int(hits.size > 0)
                                episodes.append(
                                    (m, w, run_start, r, r - run_start + 1, elig,
                                     hits.size > 0,
                                     int(hits[0]) + 1 if hits.size else -1))
                        r += 1
            return eligible, rescued, sorted(episodes)

        rng = np.random.default_rng(1)
        for _ in range(15):
            M, R, W = 2, int(rng.integers(4, 18)), 3
            x = rng.choice([0, 1], size=(M, R, W), p=[0.7, 0.3]).astype(np.int8)
            for m in range(M):                       # ragged unobservable prefix
                for w in range(W):
                    x[m, :rng.integers(0, R // 2), w] = -1
            for inactivity in (1, 2, 3):
                for horizon in (1, 2, 4):
                    e_ref, r_ref, eps = reference(x, inactivity, horizon)
                    got = QM.episode_lir(x, inactivity, horizon)
                    assert np.array_equal(got["eligible_by_trace"], e_ref)
                    assert np.array_equal(got["rescued_by_trace"], r_ref)
                    assert got["episode_trace"].size == len(eps)
                    for i, ref in enumerate(eps):
                        assert (int(got["episode_trace"][i]),
                                int(got["episode_window"][i]),
                                int(got["episode_start"][i]),
                                int(got["episode_end"][i]),
                                int(got["episode_length"][i]),
                                int(got["episode_eligibility_event"][i]),
                                bool(got["episode_rescued"][i]),
                                int(got["episode_time_to_revival"][i])) == ref


class TestBinaryTransition:
    def test_alternating_selection_flips_every_event(self):
        sel = np.array([[[0], [1], [0], [1]]])
        res = QM.binary_transition(sel, delta=1)
        pooled = res["pooled_probabilities"]
        assert pooled[0, 1] == pytest.approx(1.0)      # P01
        assert pooled[1, 0] == pytest.approx(1.0)      # P10
        # At lag 2 the state never changes.
        pooled2 = QM.binary_transition(sel, delta=2)["pooled_probabilities"]
        assert pooled2[0, 0] == pytest.approx(1.0)
        assert pooled2[1, 1] == pytest.approx(1.0)

    def test_unobservable_pairs_are_not_counted(self):
        sel = np.array([[[-1], [-1], [0], [1]]])
        res = QM.binary_transition(sel, delta=1)
        assert res["pooled_counts"].sum() == 1          # only (0 -> 1)
        assert res["pooled_probabilities"][0, 1] == pytest.approx(1.0)

    def test_rows_sum_to_one_where_defined(self):
        rng = np.random.default_rng(0)
        sel = rng.integers(0, 2, size=(3, 12, 5))
        pooled = QM.binary_transition(sel, delta=1)["pooled_probabilities"]
        assert np.allclose(pooled.sum(axis=1), 1.0)

    def test_rejects_out_of_range_delta(self):
        with pytest.raises(ValueError, match="delta"):
            QM.binary_transition(np.zeros((1, 3, 1), int), delta=3)
