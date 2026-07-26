"""Tests for the QEvict observation adapter — real npz schema in, tables out."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from modules.evaluation import qevict_observations as QO


# ---------------------------------------------------------------------------
# Synthetic parity pair matching the real npz schema (see base/ours runners)
# ---------------------------------------------------------------------------


def _synth_pair(
    *,
    S: int = 2,
    T: int = 12,
    L: int = 3,
    H: int = 2,
    W: int = 8,
    window_size: int = 1,
    prefill_len: int = 4,
    local_windows: int = 2,
    top_k_fp: int = 2,
    n_q: int = 2,
    evict_every: int = 2,
    record_step_mass: bool = True,
):
    """A base/ours pair with skewed, *drifting* importance.

    Window ``w`` is hot in phases so the oracle set genuinely changes over
    decoding — that is what makes the revival and churn numbers non-trivial.
    Geometry uses ``window_size=1`` so one window == one token and the
    granularity comparison is the real token-vs-block one.
    """
    rng = np.random.default_rng(0)
    # Per-step mass: a slowly rotating hot window plus a small floor.
    step = np.full((S, T, L, H, W), 0.01)
    for t in range(T):
        hot = (t // 2) % max(W - local_windows, 1)
        step[:, t, :, :, hot] += 1.0
        step[:, t, :, :, -1] += 0.5              # recency always attended
    step *= (1.0 + 0.05 * rng.standard_normal(step.shape))
    step = np.clip(step, 0.0, None)
    # Windows only exist once the sequence reaches them (mass before that = 0).
    exists = (np.arange(W)[None, :] < np.minimum(prefill_len + np.arange(T) + 1, W)[:, None])
    step *= exists[None, :, None, None, :]
    cum = np.cumsum(step, axis=1).astype(np.float16)      # what the runners save

    base_arrays = {
        "window_scores": cum,
        "top_window_indices": np.zeros((S, T, L, top_k_fp), np.int64),
        "eviction_step_mask": np.zeros((S, T), bool),
        "generated_tokens": np.zeros((S, T), np.int64),
    }
    if record_step_mass:                       # schema >= 1.2 (fp32, head-mean)
        base_arrays["step_window_scores"] = step.mean(axis=3).astype(np.float32)
    base_meta = {
        "mode": "parity_base",
        "schema_version": "1.2" if record_step_mass else "1.1", "seed": 42,
        "window_size": window_size, "num_sink_tokens": 0,
        "prefill_len": prefill_len, "gen_len": T,
        "local_window_size_resolved": local_windows * window_size,
        "top_k_windows": top_k_fp + n_q, "model_name": "synthetic",
        "num_samples": S,
    }

    # Ours: keep the local tail (tier 2), the top-k_fp evictable in fp (tier 0)
    # and the next n_q in Q (tier 1); everything else is absent = evicted.
    ev_mask = np.zeros((S, T), bool)
    ev_mask[:, ::evict_every] = True
    ids = np.full((S, T, L, W), -1, np.int64)
    tier = np.full((S, T, L, W), -1, np.int64)
    cum64 = cum.astype(np.float64).mean(axis=3)            # [S, T, L, W] head-mean
    for s in range(S):
        for t in range(T):
            w_act = min(prefill_len + t + 1, W)
            ew = max(w_act - local_windows, 0)
            for li in range(L):
                order = np.argsort(-cum64[s, t, li, :ew])
                fp = order[:top_k_fp]
                q = order[top_k_fp:top_k_fp + n_q]
                keep = np.concatenate([np.sort(np.concatenate([fp, q])),
                                       np.arange(ew, w_act)]).astype(np.int64)
                ids[s, t, li, :keep.size] = keep
                tags = np.where(np.isin(keep, q), 1, 0)
                tags[keep >= ew] = 2
                tier[s, t, li, :keep.size] = tags
    ours_arrays = {
        "window_scores": cum,
        "top_window_indices": np.zeros((S, T, L, top_k_fp), np.int64),
        "eviction_step_mask": ev_mask,
        "all_window_ids": ids,
        "all_window_tier": tier,
        "retained_window_ids": ids,
        "retained_window_scores": np.zeros((S, T, L, H, W), np.float16),
    }
    ours_meta = dict(base_meta, mode="parity_ours", top_k_fp=top_k_fp, N_q=n_q,
                     quant_ratio=0.5, schema_version="1.2", cache_budget=0.5)
    return ({"arrays": base_arrays, "metadata": base_meta, "path": "synthetic-base"},
            {"arrays": ours_arrays, "metadata": ours_meta, "path": "synthetic-ours"})


def _write_pair(tmp_path: Path, base: dict, ours: dict):
    paths = []
    for name, d in (("base", base), ("ours", ours)):
        p = tmp_path / f"parity_{name}.npz"
        np.savez_compressed(
            str(p), **d["arrays"],
            metadata_json=np.array([json.dumps(d["metadata"])], dtype=object))
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


class TestObservationInputs:
    def test_shapes_and_geometry(self):
        base, ours = _synth_pair()
        inp = QO.build_observation_inputs(base, ours)
        g = inp.geometry
        assert g["num_traces"] == g["num_samples"] * g["num_layers"] == 6
        assert inp.mass_cum.shape == inp.mass_step.shape == (6, 12, 8)
        assert inp.rank_scores.shape[:2] == (6, g["num_events"])
        for mask in inp.accessible.values():
            assert mask.shape == (6, g["num_events"], 8)

    def test_event_steps_come_from_the_eviction_mask(self):
        base, ours = _synth_pair(evict_every=3)
        inp = QO.build_observation_inputs(base, ours)
        assert inp.event_steps.tolist() == [0, 3, 6, 9]

    def test_no_evictions_falls_back_to_every_step(self):
        base, ours = _synth_pair()
        ours["arrays"]["eviction_step_mask"] = np.zeros_like(
            ours["arrays"]["eviction_step_mask"])
        inp = QO.build_observation_inputs(base, ours)
        assert inp.event_steps.tolist() == list(range(12))

    def test_recorded_step_mass_is_preferred_over_differencing(self):
        base, ours = _synth_pair()
        inp = QO.build_observation_inputs(base, ours)
        assert inp.diagnostics["mass_source"] == "recorded_step_mass"
        assert (inp.mass_step >= 0).all()
        # The fp32 per-step record must agree with the fp16 cumulative totals.
        assert np.allclose(inp.mass_step.sum(axis=1), inp.mass_cum[:, -1],
                           rtol=0.02, atol=1e-3)

    def test_legacy_npz_falls_back_to_differencing_and_says_so(self):
        base, ours = _synth_pair(record_step_mass=False)
        inp = QO.build_observation_inputs(base, ours)
        assert inp.diagnostics["mass_source"] == "differenced_cumulative"
        assert (inp.mass_step >= 0).all()
        assert np.allclose(inp.mass_step.sum(axis=1), inp.mass_cum[:, -1],
                           rtol=0.05, atol=1e-2)

    def test_both_mass_sources_agree_on_a_short_run(self):
        # Short run → fp16 differencing is still usable, so the two paths must
        # land in the same place; this is what licenses the fallback at all.
        rec = QO.build_observation_inputs(*_synth_pair(T=8))
        dif = QO.build_observation_inputs(*_synth_pair(T=8, record_step_mass=False))
        assert np.allclose(rec.mass_step, dif.mass_step, rtol=0.05, atol=5e-3)

    def test_fp_only_is_a_subset_of_fp_plus_q(self):
        base, ours = _synth_pair()
        inp = QO.build_observation_inputs(base, ours)
        fp, kept = inp.accessible["measured_fp_only"], inp.accessible["measured_fp_plus_q"]
        assert (kept | fp == kept).all()
        assert kept.sum() > fp.sum()             # the Q tier really holds windows

    def test_q_windows_are_excluded_from_the_fp_tier(self):
        base, ours = _synth_pair()
        inp = QO.build_observation_inputs(base, ours)
        # policy_selected marks the fp tier only, over the evictable band.
        pol = inp.policy_selected
        assert set(np.unique(pol).tolist()) <= {-1, 0, 1}
        n_fp = (pol == 1).sum()
        assert n_fp > 0
        # Each event keeps at most top_k_fp evictable windows per trace.
        assert (pol == 1).sum(axis=-1).max() <= inp.geometry["top_k_fp"]

    def test_oracle_selection_is_trinary_and_budget_sized(self):
        base, ours = _synth_pair()
        inp = QO.build_observation_inputs(base, ours)
        orc = inp.oracle_selected
        assert set(np.unique(orc).tolist()) <= {-1, 0, 1}
        assert (orc == 1).sum(axis=-1).max() <= inp.geometry["oracle_k"]
        # Local tail / uncreated windows are never candidates.
        for r, t in enumerate(inp.event_steps):
            ew = int(inp.ew_act[t])
            assert (orc[:, r, ew:] == -1).all()

    def test_trace_axis_sample_groups_layers_together(self):
        base, ours = _synth_pair()
        inp = QO.build_observation_inputs(base, ours, trace_axis="sample")
        assert inp.geometry["num_traces"] == 6
        assert inp.geometry["num_groups"] == 2
        assert sorted(np.unique(inp.trace_group).tolist()) == [0, 1]

    def test_missing_tier_arrays_fail_loudly(self):
        base, ours = _synth_pair()
        del ours["arrays"]["all_window_tier"]
        with pytest.raises(KeyError, match="all_window_tier"):
            QO.build_observation_inputs(base, ours)

    def test_unknown_trace_axis_rejected(self):
        base, ours = _synth_pair()
        with pytest.raises(ValueError, match="trace_axis"):
            QO.build_observation_inputs(base, ours, trace_axis="head")

    def test_layer_stride_and_max_samples_subset_the_trace_axis(self):
        base, ours = _synth_pair(S=2, L=4)
        inp = QO.build_observation_inputs(base, ours, max_samples=1,
                                          layer_stride=2)
        assert inp.geometry["num_samples"] == 1
        assert inp.geometry["num_layers"] == 2
        assert inp.geometry["num_layers_total"] == 4
        assert inp.geometry["num_traces"] == 2
        # Retained layer ids stay honest (0 and 2, not renumbered 0 and 1).
        assert [li for _, li in inp.trace_labels] == [0, 2]

    def test_mismatched_runs_are_rejected(self, tmp_path):
        base, ours = _synth_pair()
        ours["metadata"]["prefill_len"] = base["metadata"]["prefill_len"] + 1
        base_p, ours_p = _write_pair(tmp_path, base, ours)
        from utils.config import ParityValidationError
        with pytest.raises(ParityValidationError, match="prefill_len"):
            QO.run_observations(base_p, ours_p, tmp_path / "out", figures=False)


# ---------------------------------------------------------------------------
# The three observations
# ---------------------------------------------------------------------------


class TestObservations:
    def test_skew_is_above_uniform_and_boundaries_are_derived(self):
        base, ours = _synth_pair()
        inp = QO.build_observation_inputs(base, ours)
        obs1 = QO.analyse_skewed_importance(inp, n_boot=200)
        for row in obs1["mass_table"]:
            assert row["mean_cumulative_mass"] >= row["top_fraction"] - 1e-9
            assert row["concentration_gap"] >= -1e-9
        assert 0.0 < obs1["fp_boundary"] <= obs1["fp_plus_q_boundary"] <= 1.0
        # Curve is monotone non-decreasing and ends at full mass.
        mean = obs1["curve"]["mean"]
        assert np.all(np.diff(mean) >= -1e-9)
        assert mean[-1] == pytest.approx(1.0, abs=1e-6)

    def test_q_tier_lowers_future_missed_mass(self):
        base, ours = _synth_pair()
        inp = QO.build_observation_inputs(base, ours)
        obs2 = QO.analyse_window_level_decisions(
            inp, horizon=2, pairs=[("measured_fp_only", "measured_fp_plus_q")],
            n_boot=200)
        by = {r["policy"]: r for r in obs2["summary_table"]}
        assert (by["measured_fp_plus_q"]["future_missed_mass_mean"]
                < by["measured_fp_only"]["future_missed_mass_mean"])
        paired = {p["metric"]: p for p in obs2["paired_comparison_table"]}
        assert paired["future_missed_mass"]["absolute_reduction"] > 0
        assert 0.0 < paired["future_missed_mass"]["relative_reduction"] <= 1.0

    @pytest.mark.parametrize("top_k_fp,pool", [(4, 2), (5, 2), (5, 3), (3, 3)])
    def test_granularity_policies_are_byte_matched(self, top_k_fp, pool):
        # Includes budgets that are NOT divisible by the pool factor — that is
        # exactly where an unmatched comparison hides (the fine policy would
        # otherwise keep the remainder units for free).
        base, ours = _synth_pair(W=12, top_k_fp=top_k_fp, n_q=1)
        inp = QO.build_observation_inputs(base, ours, pool_factor=pool)
        sim = [n for n in inp.accessible if n.startswith("simulated_")]
        assert len(sim) == 2
        fine, coarse = inp.accessible[sim[0]], inp.accessible[sim[1]]
        assert fine.sum(axis=-1).tolist() == coarse.sum(axis=-1).tolist()
        obs2 = QO.analyse_window_level_decisions(inp, horizon=2, n_boot=100)
        by = {r["policy"]: r for r in obs2["summary_table"]}
        assert by[sim[0]]["mean_accessible_units"] == pytest.approx(
            by[sim[1]]["mean_accessible_units"])
        assert by[sim[0]]["mean_dropped_units"] == pytest.approx(
            by[sim[1]]["mean_dropped_units"])

    def test_oracle_revival_shows_promotion_and_demotion_pressure(self):
        base, ours = _synth_pair(T=16)
        inp = QO.build_observation_inputs(base, ours)
        kw = dict(inactivity_values=(1, 2), horizon_values=(1, 2),
                  transition_deltas=(1,), primary_inactivity=1,
                  primary_horizon=2, primary_delta=1, n_boot=100)
        orc = QO.analyse_revival(inp.oracle_selected, "oracle",
                                 groups=inp.trace_group, **kw)
        s = orc["quantifiable_result"]
        assert s["eligible_episodes"] > 0
        assert 0.0 < s["global_lir_pooled"] <= 1.0
        # A drifting oracle must show both promotion and demotion pressure.
        assert s["P01_mean"] > 0 and s["P10_mean"] > 0
        assert orc["lir_grid"].shape == (2, 2)
        assert np.isfinite(orc["lir_ci_lower"]).any()

    def test_lir_grid_ci_arrays_exist(self):
        # The original script wrote lir_ci_lower/upper that were never returned;
        # this is the regression guard.
        base, ours = _synth_pair()
        inp = QO.build_observation_inputs(base, ours)
        res = QO.analyse_revival(
            inp.oracle_selected, "oracle", groups=inp.trace_group,
            inactivity_values=(1,), horizon_values=(1, 2), transition_deltas=(1,),
            primary_inactivity=1, primary_horizon=1, primary_delta=1, n_boot=50)
        for key in ("lir_grid", "lir_ci_lower", "lir_ci_upper"):
            assert res[key].shape == (1, 2)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_run_observations_writes_every_artifact(self, tmp_path):
        base_p, ours_p = _write_pair(tmp_path, *_synth_pair(T=16))
        out = tmp_path / "obs"
        res = QO.run_observations(
            base_p, ours_p, out, fmm_horizon=4, primary_inactivity=1,
            primary_lir_horizon=2, primary_transition_delta=1,
            lir_inactivity_values=(1, 2), lir_horizon_values=(1, 2),
            transition_deltas=(1, 2), n_boot=100, figures=False)
        expected = [
            "observation1_mass_table.csv", "observation1_coverage_table.csv",
            "observation1_curve.csv", "observation2_summary_table.csv",
            "observation2_paired_comparison.csv",
            "observation3_lir_grid_oracle.csv",
            "observation3_transition_table_oracle.csv",
            "observation3_quantifiable_result_oracle.csv",
            "observation3_primary_episodes_oracle.csv",
            "observation3_lir_grid_policy_fp.csv",
            "paper_results.md", "all_results.json", "qevict_observations.npz",
        ]
        for name in expected:
            assert (out / name).exists(), f"missing {name}"
        # JSON round-trips (no NaN / numpy leakage) and keeps both selections.
        payload = json.loads((out / "all_results.json").read_text())
        assert set(payload["observation3"]) == {"oracle", "policy_fp"}
        assert payload["observation2"]["horizon"] == 4
        assert res["npz_path"].exists()
        md = (out / "paper_results.md").read_text(encoding="utf-8")
        assert "Observation I" in md and "Observation III" in md

    def test_runner_class_uses_faithfulness_paths(self, tmp_path):
        base_p, ours_p = _write_pair(tmp_path, *_synth_pair(T=16))
        cfg = type("Cfg", (), {})()
        cfg.faithfulness = type("F", (), {"base_npz_path": str(base_p),
                                          "ours_npz_path": str(ours_p)})()
        cfg.telemetry = type("T", (), {"output_dir": str(tmp_path / "outputs")})()
        cfg.run = type("R", (), {"seed": 1})()
        cfg.output_path = None
        npz = QO.QEvictObservationRunner(cfg).run()
        assert npz.exists()
        assert npz.parent.name == "qevict_observations"

    def test_runner_requires_both_paths(self):
        cfg = type("Cfg", (), {})()
        cfg.faithfulness = type("F", (), {"base_npz_path": "",
                                          "ours_npz_path": ""})()
        cfg.telemetry = type("T", (), {"output_dir": "outputs"})()
        cfg.run = type("R", (), {"seed": 0})()
        cfg.output_path = None
        with pytest.raises(ValueError, match="base_npz_path"):
            QO.QEvictObservationRunner(cfg).run()

    def test_mode_is_registered_in_main(self):
        import main
        assert "qevict_observations" in main._RUNNER_REGISTRY
