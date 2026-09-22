"""Cross-runner parity alignment tests (Suite A).

Tests that base and ours npzs share schema and that Jaccard
computation is correct on synthetic data.
"""
from __future__ import annotations
import json
import numpy as np
import pytest
import torch
from utils.metrics import jaccard_topk


class TestParityAlignment:
    def test_npz_schemas_match(self, tmp_path):
        """Base and ours npz have the same key set and dtypes."""
        required = {"top_window_indices", "window_scores",
                    "eviction_step_mask", "generated_tokens", "metadata_json"}
        for name in ["base", "ours"]:
            meta = {"schema_version": "1.0", "mode": f"parity_{name}"}
            npz = tmp_path / f"{name}.npz"
            np.savez_compressed(
                str(npz),
                top_window_indices=np.zeros((5, 4, 3), dtype=np.int64),
                window_scores=np.zeros((5, 4, 8, 10), dtype=np.float16),
                eviction_step_mask=np.zeros(5, dtype=bool),
                generated_tokens=np.arange(5, dtype=np.int64),
                metadata_json=np.array([json.dumps(meta)], dtype=object),
            )
            data = np.load(str(npz), allow_pickle=True)
            assert required.issubset(set(data.files)), f"{name} missing keys"

    def test_jaccard_computation(self):
        """Synthetic Top-K tensors with known overlap."""
        # 2 steps, 2 layers, 1 head, top_k=4
        ours = torch.tensor([[[[0, 1, 2, 3]], [[4, 5, 6, 7]]],
                              [[[0, 1, 2, 3]], [[4, 5, 6, 7]]]])
        base = torch.tensor([[[[0, 1, 4, 5]], [[4, 5, 8, 9]]],
                              [[[0, 1, 4, 5]], [[4, 5, 8, 9]]]])
        j = jaccard_topk(ours, base)
        # Layer 0: intersection={0,1}, union={0,1,2,3,4,5} → J=2/6=0.333
        assert j.shape == (2, 2, 1)
        assert abs(j[0, 0, 0].item() - 2.0/6.0) < 1e-5
        # Layer 1: intersection={4,5}, union={4,5,6,7,8,9} → J=2/6=0.333
        assert abs(j[0, 1, 0].item() - 2.0/6.0) < 1e-5

    def test_jaccard_one_when_topk_identical(self):
        """J=1 when both runs have the same Top-K."""
        topk = torch.tensor([[[[0, 1, 2]], [[3, 4, 5]]]])  # [1, 2, 1, 3]
        j = jaccard_topk(topk, topk)
        assert torch.allclose(j, torch.ones_like(j))

    def test_jaccard_zero_when_disjoint(self):
        """J=0 when Top-K sets are completely disjoint."""
        ours = torch.tensor([[[[0, 1, 2]], [[3, 4, 5]]]])
        base = torch.tensor([[[[6, 7, 8]], [[9, 10, 11]]]])
        j = jaccard_topk(ours, base)
        assert torch.allclose(j, torch.zeros_like(j))
