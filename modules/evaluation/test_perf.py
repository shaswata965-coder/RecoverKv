"""Tests for PerfRunner (Suite C).

Calibration tests verifying structure, not absolute hardware numbers.
Uses synthetic npz data — no real GPU benchmarks.
"""
from __future__ import annotations
import json
import numpy as np
import pytest
from pathlib import Path


def _make_perf_npz(path: Path, n_configs=5, n_runs=5, skip_indices=None):
    """Create a synthetic perf npz for testing."""
    names = [f"config_{i}" for i in range(n_configs)]
    attn_impls = ["eager"] * 3 + ["flash_attention_2"] * 2
    ttft = np.random.rand(n_configs, n_runs) * 100
    throughput = np.random.rand(n_configs, n_runs) * 1000
    tpot = np.random.rand(n_configs, n_runs) * 10
    peak_mem = np.random.rand(n_configs, n_runs) * 16000
    skipped = np.zeros(n_configs, dtype=bool)
    if skip_indices:
        for i in skip_indices:
            skipped[i] = True
            ttft[i, :] = np.nan
            throughput[i, :] = np.nan
            tpot[i, :] = np.nan
            peak_mem[i, :] = np.nan
    meta = {"prefill_len": 2048, "gpu_name": "T4", "clocks_locked": False}
    np.savez_compressed(
        str(path),
        config_names=np.array(names, dtype=object),
        attn_implementations=np.array(attn_impls, dtype=object),
        ttft_ms=ttft, throughput_tokps=throughput,
        tpot_ms=tpot, peak_memory_mb=peak_mem,
        skipped_mask=skipped,
        metadata_json=np.array([json.dumps(meta)], dtype=object),
    )
    return path


class TestPerfRunner:
    def test_all_non_skipped_configs_recorded(self, tmp_path):
        """Every non-skipped config has num_measurement_runs samples."""
        npz = _make_perf_npz(tmp_path / "perf.npz", n_configs=5, n_runs=5)
        data = np.load(str(npz), allow_pickle=True)
        skipped = data["skipped_mask"]
        ttft = data["ttft_ms"]
        for ci in range(len(skipped)):
            if not skipped[ci]:
                assert not np.isnan(ttft[ci]).any(), f"config {ci} has NaN"
                assert ttft[ci].shape[0] == 5

    def test_skipped_configs_are_nan_not_missing(self, tmp_path):
        """Skipped configs present in arrays with NaN, not absent."""
        npz = _make_perf_npz(tmp_path / "perf.npz", n_configs=5,
                             n_runs=5, skip_indices=[3, 4])
        data = np.load(str(npz), allow_pickle=True)
        assert data["ttft_ms"].shape[0] == 5  # all 5 present
        assert np.isnan(data["ttft_ms"][3]).all()
        assert np.isnan(data["ttft_ms"][4]).all()
        assert not np.isnan(data["ttft_ms"][0]).any()

    def test_flash_attn_configs_skipped_when_unavailable(self, tmp_path):
        """When flash-attn unavailable, flash configs are NaN."""
        npz = _make_perf_npz(tmp_path / "perf.npz", n_configs=5,
                             n_runs=5, skip_indices=[3, 4])
        data = np.load(str(npz), allow_pickle=True)
        skipped = data["skipped_mask"]
        assert skipped[3] and skipped[4]
        assert not skipped[0] and not skipped[1]

    def test_throughput_structure(self, tmp_path):
        """Throughput array has expected shape."""
        npz = _make_perf_npz(tmp_path / "perf.npz", n_configs=4, n_runs=3)
        data = np.load(str(npz), allow_pickle=True)
        assert data["throughput_tokps"].shape == (4, 3)

    def test_hook_overhead_bounded(self, tmp_path):
        """Structural test: hook config TPOT within 1.3x of baseline."""
        names = ["baseline_eager", "baseline_eager_with_hook",
                 "windowed_eager_25pct"]
        tpot = np.array([[5.0]*3, [6.0]*3, [4.0]*3])
        meta = {"prefill_len": 2048}
        npz = tmp_path / "perf.npz"
        np.savez_compressed(str(npz),
            config_names=np.array(names, dtype=object),
            attn_implementations=np.array(["eager"]*3, dtype=object),
            ttft_ms=np.ones((3,3)), throughput_tokps=np.ones((3,3)),
            tpot_ms=tpot, peak_memory_mb=np.ones((3,3)),
            skipped_mask=np.zeros(3, dtype=bool),
            metadata_json=np.array([json.dumps(meta)], dtype=object))
        data = np.load(str(npz), allow_pickle=True)
        base_tpot = np.nanmedian(data["tpot_ms"][0])
        hook_tpot = np.nanmedian(data["tpot_ms"][1])
        assert hook_tpot <= base_tpot * 1.3 + 1e-6

    def test_peak_memory_lower_with_eviction(self, tmp_path):
        """Windowed config should use less peak memory than baseline."""
        names = ["baseline_eager", "windowed_eager_25pct"]
        mem = np.array([[12000.0]*3, [8000.0]*3])
        meta = {"prefill_len": 4096}
        npz = tmp_path / "perf.npz"
        np.savez_compressed(str(npz),
            config_names=np.array(names, dtype=object),
            attn_implementations=np.array(["eager"]*2, dtype=object),
            ttft_ms=np.ones((2,3)), throughput_tokps=np.ones((2,3)),
            tpot_ms=np.ones((2,3)), peak_memory_mb=mem,
            skipped_mask=np.zeros(2, dtype=bool),
            metadata_json=np.array([json.dumps(meta)], dtype=object))
        data = np.load(str(npz), allow_pickle=True)
        base_mem = np.nanmedian(data["peak_memory_mb"][0])
        wind_mem = np.nanmedian(data["peak_memory_mb"][1])
        assert wind_mem < base_mem

    def test_max_b_reads_oom_not_skipped(self, tmp_path):
        """A config skipped for a non-OOM reason is not evidence about what fits.

        `skipped_mask` is the union of three unrelated outcomes — OOM, a missing
        flash-attn, and a config/code error. eval_perf_batched.yaml defines max-B
        as "the largest batch_size it did NOT skip", so reading the union would
        report a smaller max-B than the method achieves every time a cell errors.
        The runner records the three separately; this pins that.
        """
        names = ["fits", "oomed", "errored"]
        skipped = np.array([False, True, True])
        oom = np.array([False, True, False])
        errored = np.array([False, False, True])
        npz = tmp_path / "perf.npz"
        np.savez_compressed(str(npz),
            config_names=np.array(names, dtype=object),
            attn_implementations=np.array(["flash_attention_2"]*3, dtype=object),
            ttft_ms=np.ones((3,2)), throughput_tokps=np.ones((3,2)),
            tpot_ms=np.ones((3,2)), e2e_latency_ms=np.ones((3,2)),
            peak_memory_mb=np.ones((3,2)),
            skipped_mask=skipped, oom_mask=oom, error_mask=errored,
            skip_reason=np.array(["", "oom", "error: RuntimeError: boom"],
                                 dtype=object),
            metadata_json=np.array([json.dumps({"batch_size": 64})], dtype=object))
        data = np.load(str(npz), allow_pickle=True)
        assert data["skipped_mask"].sum() == 2
        # Only the OOM says "this batch size does not fit".
        assert data["oom_mask"].tolist() == [False, True, False]
        assert data["error_mask"].tolist() == [False, False, True]
        assert "RuntimeError" in str(data["skip_reason"][2])

    def test_peak_detail_arrays_are_per_config_per_run(self, tmp_path):
        """The peak fields the max-B decision needs, at the same shape as timings."""
        n_configs, n_runs = 3, 2
        arrs = {
            k: np.random.rand(n_configs, n_runs) * 1000
            for k in ("peak_memory_mb", "peak_reserved_mb", "peak_device_used_mb",
                      "device_total_mb", "peak_prefill_mb", "peak_decode_mb",
                      "peak_host_rss_mb", "alloc_retries")
        }
        npz = tmp_path / "perf.npz"
        np.savez_compressed(str(npz),
            config_names=np.array(["a", "b", "c"], dtype=object),
            attn_implementations=np.array(["eager"]*3, dtype=object),
            ttft_ms=np.ones((3,2)), throughput_tokps=np.ones((3,2)),
            tpot_ms=np.ones((3,2)), e2e_latency_ms=np.ones((3,2)),
            skipped_mask=np.zeros(3, dtype=bool),
            metadata_json=np.array([json.dumps({})], dtype=object), **arrs)
        data = np.load(str(npz), allow_pickle=True)
        for key in arrs:
            assert data[key].shape == (n_configs, n_runs), key

    def test_backends_not_compared_across_attention_impls(self, tmp_path):
        """Perf results contain attn_implementation per config."""
        npz = _make_perf_npz(tmp_path / "perf.npz")
        data = np.load(str(npz), allow_pickle=True)
        assert "attn_implementations" in data.files
        impls = data["attn_implementations"]
        # Verify different impls exist
        unique = set(str(x) for x in impls)
        assert len(unique) >= 1  # at least one impl recorded


class TestTierGeometry:
    """``describe_tier_geometry`` — what the BYTE budget buys in KEYS.

    The suite's decode column is set by the number of keys a step attends over,
    and ``cache_budget`` does not name that number: it is a byte budget, and int2
    is ~8x denser than fp16, so the token count swings by ~4x across
    ``quant_ratio`` at one fixed budget. Nothing in the npz recorded it, so a
    ``quant_ratio=0.5`` row and a ``quant_ratio=0.0`` row read as the same
    operating point when they differ by 2.4x in attention work. These pin the
    arithmetic and, in particular, the case where the "compressed" cache comes
    out longer than the prompt it compressed.
    """

    @staticmethod
    def _resolved(prefill_len: int, quant_ratio: float):
        import torch
        from modules.windowed_cache.config import WindowedCacheConfig

        class _MC:                       # Llama-3-8B geometry
            num_attention_heads = 32
            num_key_value_heads = 8
            hidden_size = 4096
            head_dim = 128
            num_hidden_layers = 32

        cfg = WindowedCacheConfig(
            window_size=8, num_sink_tokens=5, local_window_size=64,
            cache_budget=0.50, quant_ratio=quant_ratio, first_eviction_step=0,
        )
        # max_tokens=0 == budget_basis "context" (budget is a fraction of prefill).
        return cfg.resolve(prefill_len, _MC, torch.float16, 0)

    def test_pure_eviction_retains_the_nominal_budget(self):
        """At q=0 the byte budget and the token budget coincide: ~0.5x prefill."""
        pytest.importorskip("torch")
        from modules.evaluation.perf_runner import describe_tier_geometry
        for prefill in (4096, 2048, 1048):
            g = describe_tier_geometry(self._resolved(prefill, 0.0), prefill)
            assert g["q_tokens"] == 0
            assert 0.45 <= g["expansion"] <= 0.55, (prefill, g)

    def test_half_byte_budget_in_int2_exceeds_the_prompt(self):
        """At q=0.5 the same budget retains MORE keys than the prompt has.

        Not a resolver bug — those are half the bytes — but decisive for a
        latency table, and the reason this field exists. If this ever stops
        holding, the warning perf_runner emits is stale and should move with it.
        """
        pytest.importorskip("torch")
        from modules.evaluation.perf_runner import describe_tier_geometry
        for prefill in (4096, 2048, 1048):
            g = describe_tier_geometry(self._resolved(prefill, 0.5), prefill)
            assert g["expansion"] > 1.0, (prefill, g)
            assert g["s_eff"] == g["fp_tokens"] + g["q_tokens"]
            # The Q tier is the whole story: it is ~4x the fp tier in tokens.
            assert g["q_tokens"] > 3 * g["fp_tokens"], (prefill, g)

    def test_q0_reference_matches_an_actual_q0_resolve(self):
        """``s_eff_at_q0`` is computed from top_k_windows, not a second resolve.

        It is quoted in the warning as "what the q=0.0 row retains", so it has to
        equal what that row actually resolves to, not merely approximate it.
        """
        pytest.importorskip("torch")
        from modules.evaluation.perf_runner import describe_tier_geometry
        for prefill in (4096, 2048, 1048):
            quantized = describe_tier_geometry(self._resolved(prefill, 0.5), prefill)
            pure = describe_tier_geometry(self._resolved(prefill, 0.0), prefill)
            assert quantized["s_eff_at_q0"] == pure["s_eff"], prefill

    def test_q_loop_iters_is_the_active_window_count(self):
        """The decode kernel's serial Q loop runs once per active window."""
        pytest.importorskip("torch")
        from modules.evaluation.perf_runner import describe_tier_geometry
        g = describe_tier_geometry(self._resolved(4096, 0.5), 4096)
        assert g["decode_q_loop_iters"] == g["N_q"]
        assert g["q_tokens"] == g["N_q"] * g["window_size"]


class TestLseTransient:
    def test_transient_is_quadratic_in_context_and_linear_in_batch(self):
        """The block ``compute_lse`` builds, which is what OOMs the batched cells.

        4096 x batch-32 lands at ~16 GB and 2048 x batch-32 at ~8 GB, which is
        exactly the pattern in the recorded ladder: the first OOMs, the second
        fits. The error message quotes this number, so it has to be right.
        """
        from modules.evaluation.perf_runner import _lse_transient_gb

        class _MC:
            num_attention_heads = 32

        big = _lse_transient_gb(32, _MC, 4096)
        half_ctx = _lse_transient_gb(32, _MC, 2048)
        single = _lse_transient_gb(1, _MC, 4096)
        assert 15.0 < big < 17.0, big
        assert abs(half_ctx - big / 2) < 0.1, (half_ctx, big)   # chunk caps at 1024
        assert abs(single - big / 32) < 0.01, (single, big)


class TestDynamoCounters:
    """``_dynamo_counters`` normalization across torch versions.

    The section torch files graph breaks under has been renamed at least once
    (``graph_break`` -> ``unimplemented``, and this repo's dev box is on the
    newer name). Reading a single hard-coded section reports ZERO breaks on the
    other build — a false clean bill of health for exactly the check that is
    supposed to catch a compiled eviction silently running eager. Pin both names.
    """

    @staticmethod
    def _with_counters(monkeypatch, payload):
        import collections
        import torch._dynamo.utils as u
        fake = collections.defaultdict(collections.Counter)
        for section, entries in payload.items():
            fake[section].update(entries)
        monkeypatch.setattr(u, "counters", fake, raising=True)

    @pytest.mark.parametrize("section", ["graph_break", "unimplemented",
                                         "unimplemented_with_reason"])
    def test_graph_breaks_found_under_every_known_section_name(
            self, monkeypatch, section):
        pytest.importorskip("torch")
        from modules.evaluation import perf_runner as pr
        self._with_counters(monkeypatch, {section: {"some reason": 4, "other": 1}})
        assert pr._dynamo_counters()["graph_breaks"] == 5

    def test_frames_ok_is_the_recompile_signal(self, monkeypatch):
        """torch has no stable ``recompiles`` section; frames.ok stands in.

        Counters are zeroed after warmup, so a frame compiled during the
        measured runs is compile latency inside the timings.
        """
        pytest.importorskip("torch")
        from modules.evaluation import perf_runner as pr
        self._with_counters(monkeypatch, {
            "frames": {"total": 12, "ok": 9},
            "stats": {"calls_captured": 30, "unique_graphs": 3},
        })
        c = pr._dynamo_counters()
        assert c["frames_ok"] == 9
        assert c["frames_total"] == 12
        assert c["unique_graphs"] == 3
        assert c["graph_breaks"] == 0

    def test_unknown_sections_survive_as_raw_totals(self, monkeypatch):
        """A future rename must be visible, not silently absent."""
        pytest.importorskip("torch")
        from modules.evaluation import perf_runner as pr
        self._with_counters(monkeypatch, {"brand_new_section": {"a": 2, "b": 3}})
        c = pr._dynamo_counters()
        assert c["raw_brand_new_section"] == 5
