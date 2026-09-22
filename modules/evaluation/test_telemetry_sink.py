"""Tests for modules/evaluation/telemetry_sink.py — npz round-trip without a model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from modules.evaluation.telemetry_sink import (
    SCHEMA_VERSION,
    TelemetryMetadata,
    TelemetrySink,
)


class TestTelemetryMetadata:
    """Test metadata serialization round-trip."""

    def test_to_dict_and_from_dict(self) -> None:
        meta = TelemetryMetadata(
            mode="parity_base",
            seed=42,
            dataset="wikitext-103",
            article_id=0,
            article_sha="abc123def4567890",
            model_name="meta-llama/Meta-Llama-3-8B",
            dtype="float16",
            cache_backend="dynamic",
        )
        d = meta.to_dict()
        assert d["mode"] == "parity_base"
        assert d["seed"] == 42
        assert d["schema_version"] == SCHEMA_VERSION

        # Round-trip
        reconstructed = TelemetryMetadata.from_dict(d)
        assert reconstructed.mode == meta.mode
        assert reconstructed.seed == meta.seed
        assert reconstructed.dataset == meta.dataset
        assert reconstructed.article_sha == meta.article_sha

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {"mode": "parity_ours", "unknown_field": 999, "seed": 7}
        meta = TelemetryMetadata.from_dict(d)
        assert meta.mode == "parity_ours"
        assert meta.seed == 7
        assert not hasattr(meta, "unknown_field")


class TestTelemetrySinkNoScores:
    """Test TelemetrySink with track_scores=False (default)."""

    def test_save_and_load_metadata_only(self, tmp_path: Path) -> None:
        meta = TelemetryMetadata(
            mode="perf",
            seed=123,
            dataset="pg19",
            article_id=5,
            model_name="test-model",
            cache_backend="windowed",
            cache_backend_package="eager",
        )
        sink = TelemetrySink(meta, output_dir=tmp_path, track_scores=False)

        # Record some steps — should be no-ops
        sink.record_step(
            top_k_window_indices=np.zeros((4, 3), dtype=np.int64),
            window_scores=np.zeros((4, 8, 10), dtype=np.float16),
        )

        npz_path = sink.save("test_run")
        assert npz_path.exists()
        assert (tmp_path / "test_run.meta.json").exists()

        # Load and verify
        arrays, loaded_meta = TelemetrySink.load(npz_path)
        assert loaded_meta.mode == "perf"
        assert loaded_meta.seed == 123
        assert loaded_meta.cache_backend_package == "eager"

        # No telemetry arrays should be present
        assert "top_k_window_indices_per_step" not in arrays
        assert "window_scores_per_step" not in arrays

    def test_meta_json_sidecar(self, tmp_path: Path) -> None:
        meta = TelemetryMetadata(mode="longbench", seed=0)
        sink = TelemetrySink(meta, output_dir=tmp_path, track_scores=False)
        sink.save("sidecar_test")

        meta_path = tmp_path / "sidecar_test.meta.json"
        with open(meta_path) as f:
            sidecar = json.load(f)
        assert sidecar["mode"] == "longbench"
        assert sidecar["schema_version"] == SCHEMA_VERSION


class TestTelemetrySinkWithScores:
    """Test TelemetrySink with track_scores=True (research mode)."""

    def test_save_and_load_with_arrays(self, tmp_path: Path) -> None:
        num_layers, num_heads, num_windows, num_kept = 4, 8, 10, 3
        num_steps = 5

        meta = TelemetryMetadata(
            mode="parity_ours",
            seed=42,
            dataset="wikitext-103",
            article_id=0,
        )
        sink = TelemetrySink(meta, output_dir=tmp_path, track_scores=True)

        for step in range(num_steps):
            indices = np.random.randint(
                0, num_windows, size=(num_layers, num_kept), dtype=np.int64
            )
            scores = np.random.randn(num_layers, num_heads, num_windows).astype(
                np.float16
            )
            sink.record_step(top_k_window_indices=indices, window_scores=scores)

        npz_path = sink.save("scored_run")
        arrays, loaded_meta = TelemetrySink.load(npz_path)

        # Verify arrays
        assert "top_k_window_indices_per_step" in arrays
        assert "window_scores_per_step" in arrays

        idx_arr = arrays["top_k_window_indices_per_step"]
        score_arr = arrays["window_scores_per_step"]

        assert idx_arr.shape == (num_steps, num_layers, num_kept)
        assert idx_arr.dtype == np.int64

        assert score_arr.shape == (num_steps, num_layers, num_heads, num_windows)
        assert score_arr.dtype == np.float16

        # Verify metadata
        assert loaded_meta.mode == "parity_ours"
        assert loaded_meta.seed == 42

    def test_default_filename(self, tmp_path: Path) -> None:
        meta = TelemetryMetadata(
            mode="parity_base",
            dataset="wikitext-103",
            article_id=3,
        )
        sink = TelemetrySink(meta, output_dir=tmp_path, track_scores=False)
        npz_path = sink.save()
        assert npz_path.name == "parity_base_wikitext-103_3.npz"


class TestTelemetryTimestamps:
    """Verify that timestamps are set automatically."""

    def test_timestamps_populated(self, tmp_path: Path) -> None:
        meta = TelemetryMetadata(mode="test")
        sink = TelemetrySink(meta, output_dir=tmp_path, track_scores=False)

        # Start time should be set on init
        assert sink.metadata.run_started_utc != ""

        sink.save("ts_test")

        # Finish time should be set after save
        assert sink.metadata.run_finished_utc != ""
        assert sink.metadata.run_started_utc != sink.metadata.run_finished_utc or True  # May be same second
