"""Tests for the faithfulness runner and metrics module."""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import pytest


def _synth_pair(*, tier: bool):
    """A tiny (S=1,T=2,L=1,H=1) base/ours pair for the tier-aware metrics.

    5 windows, ids 0..4; id4 is the local tail.  Base holds all; ours keeps
    fp={0}, Q={1,2}, drops {3} (when ``tier``).  The Q windows carry real mass,
    so a Q-crediting metric must show recovered_mass_q > 0.
    """
    S, T, L, H = 1, 2, 1, 1
    base_hm = np.array([9., 7., 5., 3., 6.], dtype=np.float32)
    base_ws = np.tile(base_hm, (S, T, L, H, 1))
    base_tk = np.tile(np.array([0, 1]), (S, T, L, 1)).astype(np.int64)
    ours_hm = base_hm.copy(); ours_hm[1] *= 0.9; ours_hm[2] *= 1.1   # Q noise
    ours_tk = np.tile(np.array([0, 1]), (S, T, L, 1)).astype(np.int64)
    ours_rid = np.tile(np.array([0, 1, 4]), (S, T, L, 1)).astype(np.int64)
    ours_rsc = np.tile(ours_hm[[0, 1, 4]], (S, T, L, H, 1))
    base = {"arrays": {"window_scores": base_ws, "top_window_indices": base_tk,
                       "generated_tokens": np.zeros((S, T), np.int64)},
            "metadata": {"mode": "parity_base", "window_size": 1,
                         "num_sink_tokens": 0, "prefill_len": 3, "gen_len": T,
                         "top_k_windows": 2, "local_window_size_resolved": 1},
            "path": "/tmp/base"}
    o_meta = dict(base["metadata"], mode="parity_ours")
    o_arr = {"top_window_indices": ours_tk, "retained_window_ids": ours_rid,
             "retained_window_scores": ours_rsc}
    if tier:
        o_meta.update(quant_ratio=0.5, top_k_fp=1, N_q=2)
        o_arr["all_window_ids"] = np.tile(
            np.array([0, 1, 2, 4]), (S, T, L, 1)).astype(np.int64)
        o_arr["all_window_tier"] = np.tile(
            np.array([0, 1, 1, 2]), (S, T, L, 1)).astype(np.int64)
        o_arr["window_scores"] = np.tile(
            ours_hm[[0, 1, 2, 4]], (S, T, L, H, 1)).astype(np.float32)
    else:
        o_meta.update(quant_ratio=0.0, top_k_fp=2, N_q=0)
    return base, {"arrays": o_arr, "metadata": o_meta, "path": "/tmp/ours"}


def _run(base, ours):
    from modules.evaluation.faithfulness_runner import FaithfulnessRunner
    return FaithfulnessRunner.__new__(FaithfulnessRunner)._compute_metrics(base, ours)


class TestTierAwareMetrics:
    def test_q_tier_recovers_mass_and_is_credited(self):
        res = _run(*_synth_pair(tier=True))
        # The Q tier holds real mass, so it rescues some and is credited.
        assert (res["recovered_mass_q"] > 0).all()
        assert (res["missed_mass_kept"] <= res["missed_mass"] + 1e-9).all()
        assert 0.0 < float(res["q_tier_fidelity"]) <= 1.0
        # Discounted rescued mass never exceeds the raw rescued mass.
        assert (res["recovered_mass_q_discounted"]
                <= res["recovered_mass_q"] + 1e-9).all()
        # fp overlap is a real Jaccard in [0, 1]; kept ⊇ fp so it is defined too.
        assert (0.0 <= res["jaccard_fp"]).all() and (res["jaccard_fp"] <= 1.0).all()

    def test_no_tier_collapses_to_legacy(self):
        # Legacy ours npz (no tier arrays) → fp == kept == legacy jaccard,
        # nothing recovered.  This is the q == 0 back-compat guarantee.
        res = _run(*_synth_pair(tier=False))
        assert np.allclose(res["jaccard_fp"], res["jaccard_kept"])
        assert np.allclose(res["jaccard_lift"], 0.0)
        assert np.allclose(res["recovered_mass_q"], 0.0)
        assert np.allclose(res["missed_mass_kept"], res["missed_mass"])
        assert float(res["q_tier_fidelity"]) == 0.0
        # fp/kept equal the legacy per-layer jaccard exactly.
        assert np.allclose(res["jaccard_fp"], res["jaccard_per_layer"])


class TestFaithfulness:
    def test_faithfulness_rejects_unaligned_npz(self):
        from utils.config import ParityValidationError
        from modules.evaluation.faithfulness_runner import FaithfulnessRunner
        runner = FaithfulnessRunner.__new__(FaithfulnessRunner)
        bm = {"article_sha": "abc", "seed": 42, "prefill_len": 100,
              "gen_len": 10, "window_size": 8, "num_sink_tokens": 4, "model_name": "t"}
        om = dict(bm, article_sha="xyz")
        with pytest.raises(ParityValidationError):
            runner._validate_alignment(bm, om)

    def test_metrics_vectorized(self):
        src = Path("utils/metrics.py").read_text()
        tree = ast.parse(src)
        # Check function bodies for for-loops (allow module-level)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.For):
                        pytest.fail(f"for-loop in {node.name}")
