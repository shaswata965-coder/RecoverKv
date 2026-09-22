"""Telemetry sink — writes one npz per run with a stable schema.

The ``TelemetrySink`` class is the shared serialization layer that every
runner writes through. It owns the npz schema (v1.0) and guarantees
round-trip fidelity.

Schema
------
Two arrays (optional — absent in perf/LongBench runs with ``track_scores=False``):
- ``top_k_window_indices_per_step``: ``[num_steps, num_layers, num_kept_windows]`` int64
- ``window_scores_per_step``: ``[num_steps, num_layers, num_heads, num_windows]`` fp16

Plus a metadata dict with full reproducibility fields (see module docstring).

Independently testable (npz round-trip without a model).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = "1.0"


@dataclass
class TelemetryMetadata:
    """Full reproducibility metadata for a single run.

    All fields map 1:1 to the spec in §3 and §9.
    """

    # Schema
    schema_version: str = SCHEMA_VERSION

    # Run identity
    mode: str = ""  # parity_base | parity_ours | longbench | perf | faithfulness
    seed: int = 0

    # Article identity
    dataset: str = ""
    article_id: int = 0
    article_sha: str = ""
    tokenizer_sha: str = ""
    prefill_len: int = 0
    gen_len: int = 0

    # Window config
    window_size: int = 0
    num_sink_tokens: int = 0
    local_window_size_resolved: int = 0
    top_k_windows: int = 0

    # Model
    model_name: str = ""
    model_revision: Optional[str] = None
    dtype: str = ""
    attn_implementation: str = ""

    # Cache
    cache_backend: str = ""  # "dynamic" or "windowed"
    cache_backend_package: Optional[str] = None  # "flash_attn", "eager", None
    cache_budget: Optional[float] = None

    # Environment
    transformers_version: str = ""
    torch_version: str = ""
    flash_attn_version: Optional[str] = None
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None
    commit_sha: Optional[str] = None

    # Timestamps
    run_started_utc: str = ""
    run_finished_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            # np/torch types → Python scalars
            if hasattr(v, "item"):
                v = v.item()  # type: ignore[union-attr]
            d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TelemetryMetadata":
        """Construct from a plain dict (e.g. loaded from JSON)."""
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


class TelemetrySink:
    """Accumulates telemetry data and writes it to npz files.

    Parameters
    ----------
    metadata : TelemetryMetadata
        Pre-filled metadata for this run (timestamps are set automatically).
    output_dir : str or Path
        Directory to write output files.
    track_scores : bool
        If False, ``record_step`` calls are no-ops and the npz contains
        only metadata.
    """

    def __init__(
        self,
        metadata: TelemetryMetadata,
        output_dir: str | Path = "outputs",
        track_scores: bool = False,
    ) -> None:
        self.metadata = metadata
        self.output_dir = Path(output_dir)
        self.track_scores = track_scores

        # Accumulation buffers (only used if track_scores=True)
        self._top_k_indices: List[np.ndarray] = []
        self._window_scores: List[np.ndarray] = []

        # Set start time
        self.metadata.run_started_utc = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )

    def record_step(
        self,
        top_k_window_indices: Optional[np.ndarray] = None,
        window_scores: Optional[np.ndarray] = None,
    ) -> None:
        """Record one generation step's telemetry.

        Parameters
        ----------
        top_k_window_indices : np.ndarray, optional
            Shape ``[num_layers, num_kept_windows]``, int64.
        window_scores : np.ndarray, optional
            Shape ``[num_layers, num_heads, num_windows]``, fp16.
            Only recorded in research mode (``track_scores=True``).
        """
        if not self.track_scores:
            return

        if top_k_window_indices is not None:
            self._top_k_indices.append(
                np.asarray(top_k_window_indices, dtype=np.int64)
            )

        if window_scores is not None:
            self._window_scores.append(
                np.asarray(window_scores, dtype=np.float16)
            )

    def save(self, filename: Optional[str] = None) -> Path:
        """Write the accumulated telemetry to an npz file + metadata sidecar.

        Parameters
        ----------
        filename : str, optional
            Base filename (without extension). Defaults to
            ``{mode}_{dataset}_{article_id}``.

        Returns
        -------
        Path
            Path to the written ``.npz`` file.
        """
        # Set finish time
        self.metadata.run_finished_utc = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = (
                f"{self.metadata.mode}_{self.metadata.dataset}"
                f"_{self.metadata.article_id}"
            )

        npz_path = self.output_dir / f"{filename}.npz"
        meta_path = self.output_dir / f"{filename}.meta.json"

        # Build arrays dict
        arrays: Dict[str, np.ndarray] = {}

        if self.track_scores and self._top_k_indices:
            arrays["top_k_window_indices_per_step"] = np.stack(
                self._top_k_indices, axis=0
            )

        if self.track_scores and self._window_scores:
            arrays["window_scores_per_step"] = np.stack(
                self._window_scores, axis=0
            )

        # Save npz with metadata embedded as JSON string
        meta_dict = self.metadata.to_dict()
        arrays["metadata_json"] = np.array(
            [json.dumps(meta_dict)], dtype=object
        )
        np.savez(npz_path, **arrays)

        # Write sidecar .meta.json
        with open(meta_path, "w") as f:
            json.dump(meta_dict, f, indent=2, default=str)

        log.info("Telemetry saved to %s", npz_path)
        return npz_path

    @staticmethod
    def load(npz_path: str | Path) -> tuple[Dict[str, np.ndarray], TelemetryMetadata]:
        """Load a telemetry npz file and return ``(arrays, metadata)``.

        Parameters
        ----------
        npz_path : str or Path
            Path to the ``.npz`` file.

        Returns
        -------
        tuple[dict, TelemetryMetadata]
            The arrays dict and reconstructed metadata.
        """
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)

        # Extract metadata
        meta_json_arr = data["metadata_json"]
        meta_str = str(meta_json_arr[0])
        meta_dict = json.loads(meta_str)
        metadata = TelemetryMetadata.from_dict(meta_dict)

        # Extract telemetry arrays
        arrays: Dict[str, np.ndarray] = {}
        for key in data.files:
            if key != "metadata_json":
                arrays[key] = data[key]

        return arrays, metadata
