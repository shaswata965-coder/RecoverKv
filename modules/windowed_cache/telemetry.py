"""Telemetry and NullTelemetry for windowed KV cache.

``Telemetry`` records ``.detach().cpu()`` clones of per-layer, per-step state.
OFF by default — used only by parity / faithfulness suites.

``NullTelemetry`` overrides every method with no-op.  Subclass dispatch avoids
``if self.enabled`` branches in the hot path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


class Telemetry:
    """Records per-layer, per-step snapshots of cache state.

    Memory grows linearly in ``num_layers × H_q × num_windows × num_steps``.
    Enable only when needed (e.g., parity / faithfulness evaluation).

    Parameters
    ----------
    num_layers : int
        Number of transformer layers to track.
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self._records: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(num_layers)
        }

    def record_scores(
        self,
        layer_idx: int,
        step: int,
        window_scores: Tensor,
        retain_indices: Optional[Tensor] = None,
    ) -> None:
        """Record a snapshot of window scores for a layer at a step.

        Parameters
        ----------
        layer_idx : int
        step : int
        window_scores : Tensor
            Shape ``[B, H_q, W]``.
        retain_indices : Tensor, optional
            Shape ``[B, T_retained]``, retained token indices (if eviction ran).
        """
        record: Dict[str, Any] = {
            "step": step,
            "window_scores": window_scores.detach().cpu().clone(),
        }
        if retain_indices is not None:
            record["retain_indices"] = retain_indices.detach().cpu().clone()
        self._records[layer_idx].append(record)

    def record_cache_state(
        self,
        layer_idx: int,
        step: int,
        key_states: Tensor,
        value_states: Tensor,
        position_ids: Tensor,
    ) -> None:
        """Record a snapshot of K/V/position state."""
        self._records[layer_idx].append({
            "step": step,
            "key_states": key_states.detach().cpu().clone(),
            "value_states": value_states.detach().cpu().clone(),
            "position_ids": position_ids.detach().cpu().clone(),
        })

    def get_records(self, layer_idx: int) -> List[Dict[str, Any]]:
        """Return all recorded snapshots for a layer."""
        return self._records.get(layer_idx, [])

    def clear(self) -> None:
        """Clear all recorded data."""
        for layer_idx in self._records:
            self._records[layer_idx].clear()


class NullTelemetry(Telemetry):
    """No-op telemetry.  Subclass dispatch avoids if-branches in the hot path."""

    def __init__(self, num_layers: int = 0) -> None:
        # Don't allocate storage
        self.num_layers = num_layers
        self._records: Dict[int, List[Dict[str, Any]]] = {}

    def record_scores(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""

    def record_cache_state(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""

    def get_records(self, layer_idx: int) -> List[Dict[str, Any]]:
        return []

    def clear(self) -> None:
        """No-op."""
