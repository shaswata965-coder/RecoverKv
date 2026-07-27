"""CPU tests for batch planning and padding-induced budget drift."""

from __future__ import annotations

import pytest

from gsm8k.batching import (
    BatchPlan,
    assert_drift_within,
    budget_drift,
    n_kept_for,
    plan_batches,
)


class TestPlanBatches:
    def test_every_example_appears_exactly_once(self):
        lengths = [124 + (i * 7) % 90 for i in range(500)]
        plan = plan_batches(lengths, max_batch_size=64, pad_to_multiple=16)
        seen = [i for g in plan.groups for i in g]
        assert sorted(seen) == list(range(500))

    def test_exact_length_mode_pads_nothing(self):
        lengths = [130, 131, 130, 145, 131, 130]
        plan = plan_batches(lengths, max_batch_size=64, pad_to_multiple=None)
        assert plan.max_spread == 0
        for group, padded in zip(plan.groups, plan.group_lengths):
            assert all(lengths[i] == padded for i in group)

    def test_pad_to_multiple_bounds_the_spread(self):
        lengths = [124 + (i * 13) % 160 for i in range(800)]
        for w in (8, 16, 32):
            plan = plan_batches(lengths, max_batch_size=128, pad_to_multiple=w)
            assert plan.max_spread <= w - 1

    def test_cap_is_respected(self):
        lengths = [160] * 300
        plan = plan_batches(lengths, max_batch_size=64, pad_to_multiple=16)
        assert all(len(g) <= 64 for g in plan.groups)
        assert plan.n_batches == 5  # 300 -> 64,64,64,64,44

    def test_group_length_is_the_group_max(self):
        lengths = [129, 130, 136, 140]
        plan = plan_batches(lengths, max_batch_size=64, pad_to_multiple=16)
        for group, padded in zip(plan.groups, plan.group_lengths):
            assert padded == max(lengths[i] for i in group)

    def test_plan_is_deterministic(self):
        lengths = [124 + (i * 31) % 120 for i in range(400)]
        a = plan_batches(lengths, 128, 16)
        b = plan_batches(lengths, 128, 16)
        assert a.groups == b.groups

    def test_bigger_cap_never_makes_more_batches(self):
        lengths = [124 + (i * 17) % 150 for i in range(1000)]
        prev = None
        for cap in (8, 16, 32, 64, 128):
            n = plan_batches(lengths, cap, 16).n_batches
            if prev is not None:
                assert n <= prev
            prev = n

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            plan_batches([160], max_batch_size=0)
        with pytest.raises(ValueError, match="pad_to_multiple"):
            plan_batches([160], pad_to_multiple=0)


class TestBudgetDrift:
    def test_n_kept_reproduces_the_upstream_float_artefact(self):
        """int(150 * (1 - 0.8)) is 29, not 30 -- and that is what the press computes."""
        assert n_kept_for(150, 0.8) == 29
        assert n_kept_for(150, 0.5) == 75

    def test_exact_length_groups_have_zero_drift(self):
        lengths = [124 + (i * 7) % 90 for i in range(400)]
        plan = plan_batches(lengths, 128, pad_to_multiple=None)
        drift = budget_drift(plan, lengths, 0.5)
        assert drift["worst_pct"] == 0.0
        assert drift["worst_tokens"] == 0.0

    def test_drift_grows_with_pad_multiple(self):
        lengths = [124 + (i * 13) % 160 for i in range(800)]
        worst = [
            budget_drift(plan_batches(lengths, 128, w), lengths, 0.5)["worst_pct"]
            for w in (8, 16, 32)
        ]
        assert worst[0] <= worst[1] <= worst[2]

    def test_drift_is_zero_for_the_full_cache_control(self):
        lengths = [124 + (i * 11) % 100 for i in range(200)]
        plan = plan_batches(lengths, 128, 16)
        assert budget_drift(plan, lengths, 0.0)["worst_pct"] == 0.0

    def test_a_row_padded_up_retains_more_than_it_should(self):
        """The concrete failure: same prompt, different batch, different compression."""
        lengths = [100, 128]  # one short row grouped with a long one
        plan = plan_batches(lengths, max_batch_size=64, pad_to_multiple=32)
        assert plan.groups == [[0, 1]]
        # padded to 128 -> budget 64, but row 0 alone would get 50
        assert n_kept_for(128, 0.5) == 64
        assert n_kept_for(100, 0.5) == 50
        assert budget_drift(plan, lengths, 0.5)["worst_tokens"] == 14

    def test_assert_drift_raises_and_names_the_fix(self):
        lengths = [100, 128]
        plan = plan_batches(lengths, 64, pad_to_multiple=32)
        with pytest.raises(ValueError, match="pad_to_multiple"):
            assert_drift_within(plan, lengths, 0.5, limit_pct=10.0)

    def test_assert_drift_passes_a_tight_plan(self):
        lengths = [124 + (i * 7) % 90 for i in range(400)]
        plan = plan_batches(lengths, 128, pad_to_multiple=8)
        drift = assert_drift_within(plan, lengths, 0.5, limit_pct=10.0)
        assert drift["worst_pct"] <= 10.0


class TestPlanSummary:
    def test_summary_mentions_the_knobs_that_matter(self):
        lengths = [124 + (i * 7) % 90 for i in range(300)]
        s = plan_batches(lengths, 128, 16).summary()
        assert "pad_to_multiple=16" in s and "cap 128" in s

    def test_mean_batch_size(self):
        plan = plan_batches([160] * 200, max_batch_size=64, pad_to_multiple=16)
        assert plan.n_examples == 200
        assert plan.mean_batch_size == pytest.approx(200 / plan.n_batches)
