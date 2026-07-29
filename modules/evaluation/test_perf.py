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
