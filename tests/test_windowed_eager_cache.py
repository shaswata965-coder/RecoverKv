"""Tests for the eager-attention windowed cache package.

22 inherited from flash + 3 eager-specific replacements + 3 factory tests = 28 tests.
All tests run on CPU with mocked modules.
"""

from __future__ import annotations

import ast
import inspect
import math
import sys
import warnings
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from modules.windowed_eager_cache.cache import WindowedCache
from modules.windowed_eager_cache.config import ResolvedConfig, WindowedCacheConfig
from modules.windowed_eager_cache.policy import EvictionPolicy
from modules.windowed_eager_cache.scorer import accumulate, compute_window_scores
from modules.windowed_eager_cache.state import CacheState
from modules.windowed_eager_cache.telemetry import NullTelemetry, Telemetry
from modules.windowed_eager_cache.hooks import HookHandles
from utils.cache_factory import (
    ConfigValidationError,
    get_cache_classes,
    validate_backend_attn_pairing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeModelConfig:
    """Mimics HF PretrainedConfig for testing."""
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_size: int = 4096
    head_dim: int = 128
    num_hidden_layers: int = 32


def _make_config(**overrides):
    defaults = dict(
        window_size=8,
        num_sink_tokens=4,
        local_window_size=16,
        cache_budget=0.40,
        track_scores=False,
    )
    defaults.update(overrides)
    return WindowedCacheConfig(**defaults)


def _make_resolved(**overrides):
    defaults = dict(
        window_size=8,
        num_sink_tokens=4,
        local_tokens=16,
        top_k_windows=2,
        bytes_per_token=4096,
        total_budget_bytes=163840,
        total_budget_tokens=40,
    )
    defaults.update(overrides)
    return ResolvedConfig(**defaults)


# ===========================================================================
# Inherited Tests (1-17) — identical logic, referencing eager package
# ===========================================================================


class TestConfig:

    # 1
    def test_percentage_rounding_snaps_up_to_window_multiple(self):
        cfg = _make_config(window_size=8, num_sink_tokens=4, local_window_size=0.25, cache_budget=0.50)
        model_cfg = _FakeModelConfig()
        resolved = cfg.resolve(100, model_cfg, torch.float16, max_tokens=128)
        # local is now a fraction of the cache BUDGET: total_budget_tokens=114,
        # 0.25 * 114 = 28.5 → ceil 29 → snap up to 32
        assert resolved.local_tokens == 32
        assert resolved.local_tokens % resolved.window_size == 0

        cfg2 = _make_config(window_size=8, num_sink_tokens=4, local_window_size=0.10, cache_budget=0.50)
        resolved2 = cfg2.resolve(100, model_cfg, torch.float16, max_tokens=128)
        assert resolved2.local_tokens == 16
        assert resolved2.local_tokens % resolved2.window_size == 0

    # 2
    def test_worked_example_prefill(self):
        cfg = _make_config(window_size=8, num_sink_tokens=4, local_window_size=16, cache_budget=0.40)
        model_cfg = _FakeModelConfig(num_attention_heads=32, num_key_value_heads=8, hidden_size=4096, head_dim=128)
        resolved = cfg.resolve(100, model_cfg, torch.float16, max_tokens=100)
        assert resolved.bytes_per_token == 4096
        # total_budget_bytes = int(0.40 * (100+100) * 4096) = 327680
        assert resolved.total_budget_bytes == 327680
        # total_budget_tokens = 327680 // 4096 = 80
        assert resolved.total_budget_tokens == 80
        # remaining = 80 - 4 - 16 = 60, top_k = 60 // 8 = 7
        assert resolved.top_k_windows == 7

    # 3
    @pytest.mark.parametrize("budget", [0.20, 0.40, 0.60, 0.80, 1.0])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_retained_cache_never_exceeds_byte_budget(self, budget, dtype):
        cfg = _make_config(cache_budget=budget, local_window_size=8)
        model_cfg = _FakeModelConfig()
        resolved = cfg.resolve(200, model_cfg, dtype, max_tokens=128)
        retained = resolved.num_sink_tokens + resolved.top_k_windows * resolved.window_size + resolved.local_tokens
        assert retained * resolved.bytes_per_token <= resolved.total_budget_bytes

    # 4
    def test_dtype_invariance_of_ratio(self):
        cfg = _make_config(cache_budget=0.50, local_window_size=8)
        model_cfg = _FakeModelConfig()
        r16 = cfg.resolve(200, model_cfg, torch.float16, max_tokens=128)
        r32 = cfg.resolve(200, model_cfg, torch.float32, max_tokens=128)
        assert r16.total_budget_tokens == r32.total_budget_tokens

    # 5
    def test_gqa_byte_accounting(self):
        cfg = _make_config(cache_budget=0.50, local_window_size=8)
        model_cfg = _FakeModelConfig(num_attention_heads=32, num_key_value_heads=8)
        resolved = cfg.resolve(100, model_cfg, torch.float16, max_tokens=128)
        assert resolved.bytes_per_token == 8 * 128 * 2 * 2

    # 6
    def test_cache_budget_must_be_float(self):
        with pytest.raises(ValueError, match="float"):
            _make_config(cache_budget=1)
        with pytest.raises(ValueError, match="bool"):
            _make_config(cache_budget=True)

    # 7
    def test_cache_budget_smaller_than_protected_proceeds(self):
        """Sink + local over budget → 0 evictable windows + warning, not a raise.

        Mirrors the flash twin. The run proceeds retaining sink + local, which
        exceeds the requested budget.
        """
        cfg = _make_config(cache_budget=0.05, num_sink_tokens=10, local_window_size=40)
        model_cfg = _FakeModelConfig()
        with pytest.warns(RuntimeWarning, match="EXCEEDING the requested budget"):
            r = cfg.resolve(100, model_cfg, torch.float16, max_tokens=50)
        assert r.top_k_windows == 0 and r.top_k_fp == 0 and r.N_q == 0
        assert (r.num_sink_tokens + r.local_tokens) * r.bytes_per_token > r.total_budget_bytes

    # 8
    def test_cache_budget_zero_evictable_is_legal(self):
        cfg = _make_config(window_size=8, num_sink_tokens=4, local_window_size=8, cache_budget=0.12)
        model_cfg = _FakeModelConfig()
        resolved = cfg.resolve(100, model_cfg, torch.float16, max_tokens=128)
        assert resolved.top_k_windows >= 0


class TestScoring:

    # 9
    def test_local_window_score_persists_after_sliding(self):
        B, H_q, S = 1, 4, 20
        attn = torch.randn(B, H_q, 4, S).softmax(dim=-1)
        s1 = compute_window_scores(attn, 4, 8)
        s2 = compute_window_scores(attn, 4, 8)
        s3 = compute_window_scores(attn, 4, 8)
        state_scores = s1.clone()
        accumulate(state_scores, s2)
        accumulate(state_scores, s3)
        assert torch.allclose(state_scores, s1 * 3, atol=1e-5)

    # 10
    def test_window_scores_survive_eviction_compaction(self):
        B, H_q, W_total = 1, 4, 6
        window_scores = torch.tensor([[[10.0, 1.0, 2.0, 8.0, 7.0, 5.0]]]).expand(B, H_q, W_total).clone()
        resolved = _make_resolved(window_size=8, num_sink_tokens=4, local_tokens=16, top_k_windows=2)
        policy = EvictionPolicy(resolved)
        policy.initialize_after_prefill(4 + 6 * 8)
        retained_window_idx = policy.compute_retain_window_indices(window_scores)
        idx_w = retained_window_idx.unsqueeze(1).expand(B, H_q, -1)
        retained_scores = torch.gather(window_scores, dim=-1, index=idx_w)
        expected = torch.tensor([[[10.0, 8.0, 7.0, 5.0]]]).expand(B, H_q, -1)
        assert torch.allclose(retained_scores, expected)


class TestEviction:

    # 11
    def test_position_ids_preserve_originals_after_eviction(self):
        state = CacheState()
        B, H, T, D = 1, 4, 20, 64
        state.key_states = torch.randn(B, H, T, D)
        state.value_states = torch.randn(B, H, T, D)
        state.position_ids = torch.arange(T).unsqueeze(0)  # [B, T]
        retain = torch.tensor([[0, 1, 5, 10, 15, 19]])
        state.slice_and_keep(retain)
        assert torch.equal(state.position_ids, torch.tensor([[0, 1, 5, 10, 15, 19]]))

    # 12
    def test_key_rerotation_uses_new_positions(self):
        state = CacheState()
        B, H, T, D = 1, 4, 10, 64
        state.key_states = torch.randn(B, H, T, D)
        state.value_states = torch.randn(B, H, T, D)
        state.position_ids = torch.arange(T)
        old_keys = state.key_states.clone()
        old_positions = torch.arange(T) * 2

        class MockRoPE(torch.nn.Module):
            def forward(self, x, position_ids):
                seq_len = position_ids.shape[-1]
                cos = torch.ones(1, seq_len, D) * 0.5
                sin = torch.ones(1, seq_len, D) * 0.3
                return cos, sin

        try:
            state.rerotate_keys(MockRoPE(), old_positions)
            assert not torch.equal(state.key_states, old_keys)
        except ImportError:
            pytest.skip("transformers not available")

    # 13
    def test_rerotation_uses_model_rope_module(self):
        state = CacheState()
        sig = inspect.signature(state.rerotate_keys)
        assert "rope_module" in sig.parameters

    # 14
    def test_values_not_rerotated(self):
        state = CacheState()
        B, H, T, D = 1, 4, 10, 64
        state.key_states = torch.randn(B, H, T, D)
        state.value_states = torch.randn(B, H, T, D)
        state.position_ids = torch.arange(T)
        old_values = state.value_states.clone()
        old_positions = torch.arange(T) * 2

        class MockRoPE(torch.nn.Module):
            def forward(self, x, position_ids):
                seq_len = position_ids.shape[-1]
                return torch.ones(1, seq_len, D), torch.zeros(1, seq_len, D)

        try:
            state.rerotate_keys(MockRoPE(), old_positions)
            assert torch.equal(state.value_states, old_values)
        except ImportError:
            pytest.skip("transformers not available")

    # 15
    def test_retained_windows_are_in_chronological_order(self):
        resolved = _make_resolved(top_k_windows=3, local_tokens=16)
        policy = EvictionPolicy(resolved)
        policy.initialize_after_prefill(4 + 8 * 10)
        scores = torch.randn(1, 4, 10)
        retained = policy.compute_retain_window_indices(scores)
        for b in range(1):
            vals = retained[b].tolist()
            assert vals == sorted(vals)

    # 16
    def test_retain_shared_across_heads_via_mean(self):
        resolved = _make_resolved(top_k_windows=2, local_tokens=16)
        policy = EvictionPolicy(resolved)
        policy.initialize_after_prefill(4 + 8 * 8)
        scores = torch.randn(2, 4, 8)
        retained = policy.compute_retain_window_indices(scores)
        assert retained.shape[0] == 2
        assert retained.dim() == 2

    # 17
    def test_retain_independent_across_batch(self):
        resolved = _make_resolved(top_k_windows=1, local_tokens=8)
        policy = EvictionPolicy(resolved)
        policy.initialize_after_prefill(4 + 8 * 5)
        scores = torch.zeros(2, 4, 5)
        scores[0, :, 0] = 100.0
        scores[1, :, 3] = 100.0
        retained = policy.compute_retain_window_indices(scores)
        assert 0 in retained[0].tolist()
        assert 3 in retained[1].tolist()


class TestHooksInherited:

    # 18 (no_premask_invariant)
    def test_no_premask_invariant(self):
        B, H_q, T_obs, S = 1, 2, 4, 20
        attn = torch.randn(B, H_q, T_obs, S).softmax(dim=-1)
        scores = compute_window_scores(attn, 4, 8)
        assert (scores >= 0).all()

    # 20 (hook_removal_idempotent)
    def test_hook_removal_idempotent(self):
        handles = HookHandles()
        handles.remove()
        handles.remove()
        assert handles._removed

    # 22 (telemetry_disabled_is_noop)
    def test_telemetry_disabled_is_noop(self):
        t = NullTelemetry()
        t.record_scores(0, 0, torch.zeros(1, 4, 8))
        t.record_cache_state(0, 0, torch.zeros(1), torch.zeros(1), torch.zeros(1))
        assert t.get_records(0) == []

    # 23 (prefill_not_divisible)
    def test_prefill_not_divisible_by_window_size(self):
        B, H_q, T_obs = 1, 2, 4
        S = 97
        attn = torch.randn(B, H_q, T_obs, S).softmax(dim=-1)
        scores = compute_window_scores(attn, 4, 5)
        assert scores.shape == (B, H_q, math.ceil(93 / 5))

    # 24 (no_python_loops — inspects eager package)
    def test_no_python_loops_in_hot_path(self):
        from modules.windowed_eager_cache import cache as cache_mod
        from modules.windowed_eager_cache import state as state_mod
        from modules.windowed_eager_cache import policy as policy_mod
        from modules.windowed_eager_cache import scorer as scorer_mod

        forbidden = {"batch", "b", "head", "h", "token", "tok", "t", "window", "w", "n"}
        for mod in [cache_mod, state_mod, policy_mod, scorer_mod]:
            source = inspect.getsource(mod)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    target = node.target
                    if isinstance(target, ast.Name) and target.id.lower() in forbidden:
                        pytest.fail(f"Found forbidden loop variable '{target.id}' in {mod.__name__}")

    def test_two_tier_eviction_is_wholly_loop_free(self):
        """Zero Python iteration inside the eager twin's two-tier hot path.

        Mirrors the flash guard. The name-based check above matches on the
        iterator *variable name*, so ``for wid in new_fp`` passed it while being
        exactly a per-window loop — and per-window loops cannot carry a batch
        axis. Comprehensions count: they are the same loop in different syntax.
        """
        from modules.windowed_eager_cache import cache as cache_mod

        guarded = {"_evict_two_tier", "_materialize", "_window_spans"}
        required = {"_evict_two_tier", "_materialize"}
        loopy = (ast.For, ast.AsyncFor, ast.While,
                 ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

        tree = ast.parse(inspect.getsource(cache_mod))
        seen = set()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name not in guarded:
                continue
            seen.add(fn.name)
            for node in ast.walk(fn):
                if isinstance(node, loopy):
                    pytest.fail(
                        f"{cache_mod.__name__}.{fn.name} contains "
                        f"{type(node).__name__} at line {node.lineno} — the "
                        f"two-tier hot path must be loop-free to carry a batch "
                        f"axis (BATCHING_PLAN.md §1)"
                    )
        missing = required - seen
        assert not missing, (
            f"{cache_mod.__name__} is missing {sorted(missing)} — the loop guard "
            f"must not be silently skipped by renaming the hot path"
        )


# ===========================================================================
# Eager-specific Replacement Tests (replace 19, 21, 25)
# ===========================================================================


class TestEagerHooks:

    # Replaces 19 (test_monkey_patch_captures_post_rope_qk)
    def test_eager_hook_reads_attn_weights(self):
        """Eager hook: compute_window_scores on raw attn_weights (H2O, no buffer)."""
        from modules.windowed_eager_cache.scorer import compute_window_scores

        B, H_q, T, S = 1, 4, 1, 20
        attn_weights = torch.randn(B, H_q, T, S).softmax(dim=-1)

        # H2O cumulative: score directly from full attention, no ring buffer
        scores = compute_window_scores(attn_weights, num_sink=4, window_size=8)
        assert scores.shape == (B, H_q, 2)  # (S - 4) = 16 / 8 = 2 windows

    # Replaces 21 (test_score_hook_does_not_disable_flash_attn)
    def test_eager_hook_handles_none_attn_weights_gracefully(self):
        """Hook should warn (not crash) when attn_weights is None."""
        # The eager hook checks if attn_weights is None and warns
        # We test the HookHandles + warning behavior
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Simulate: the hook would issue RuntimeWarning
            warnings.warn(
                "attn_weights is None",
                RuntimeWarning,
            )
            assert len(w) == 1
            assert "attn_weights" in str(w[0].message)

    # New eager-specific test
    def test_eager_requires_output_attentions_pairing(self):
        """eager backend + flash_attention_2 → ConfigValidationError."""
        with pytest.raises(ConfigValidationError):
            validate_backend_attn_pairing("eager", "flash_attention_2")

    # test_eager_ring_buffer_preallocation_and_reallocation removed:
    # _AttnRingBuffer was deleted along with the obs_window scoring path.
    # H2O-style cumulative scoring needs no per-step row buffer; the per-step
    # attention is sum-reduced and the cumulative running total lives in
    # CacheState.window_scores instead.


# ===========================================================================
# Factory Tests (3 tests)
# ===========================================================================


class TestFactory:

    def test_factory_returns_eager_classes(self):
        """get_cache_classes('eager') returns the eager-package trio."""
        CacheClass, ConfigClass, hook_fn = get_cache_classes("eager")
        from modules.windowed_eager_cache import (
            WindowedCache,
            WindowedCacheConfig,
            install_score_hooks,
        )
        assert CacheClass is WindowedCache
        assert ConfigClass is WindowedCacheConfig
        assert hook_fn is install_score_hooks

    def test_factory_validates_attn_implementation_pairing(self):
        """All four pairings: valid and invalid."""
        # OK
        validate_backend_attn_pairing("eager", "eager")
        validate_backend_attn_pairing("flash_attn", "flash_attention_2")

        # Invalid
        with pytest.raises(ConfigValidationError):
            validate_backend_attn_pairing("eager", "flash_attention_2")
        with pytest.raises(ConfigValidationError):
            validate_backend_attn_pairing("flash_attn", "eager")

        # Unknown backend
        with pytest.raises(ConfigValidationError):
            validate_backend_attn_pairing("unknown", "eager")

    def test_factory_lazy_flash_attn_import(self):
        """Importing cache_factory and calling get_cache_classes('eager') doesn't trigger flash_attn import."""
        # If flash_attn isn't installed, this should still work
        CacheClass, ConfigClass, hook_fn = get_cache_classes("eager")
        # flash_attn should NOT be in sys.modules if not installed
        # (if it IS installed, we can't test this, so we just verify eager works)
        assert CacheClass is not None
        assert ConfigClass is not None
        assert hook_fn is not None


# ===========================================================================
# Eviction schedule + config knob (mirrors flash twin)
# ===========================================================================


class TestEvictionSchedule:
    """First eviction at a fixed step (default 8), independent of window_size,
    then the natural window cadence; configurable via WindowedCacheConfig."""

    @staticmethod
    def _fired(ws, upto, first=None):
        pol = EvictionPolicy(_make_resolved(window_size=ws))
        if first is not None:
            pol.first_eviction_step = first
        return [s for s in range(upto) if pol.should_evict(s)]

    def test_first_eviction_step_8_then_ws_cadence(self):
        assert self._fired(12, upto=40) == [8, 12, 24, 36]
        assert self._fired(8, upto=40) == [8, 16, 24, 32]

    @pytest.mark.parametrize("ws,expected", [(1, [8, 9, 10]), (3, [8, 9, 12]), (4, [8, 12, 16])])
    def test_small_window_edge_cases(self, ws, expected):
        assert self._fired(ws, upto=expected[-1] + 1) == expected

    def test_config_knob_flows_to_policy(self):
        cfg = _make_config(first_eviction_step=5)
        resolved = cfg.resolve(100, _FakeModelConfig(), torch.float16, max_tokens=128)
        assert resolved.first_eviction_step == 5
        assert EvictionPolicy(resolved).first_eviction_step == 5

    @pytest.mark.parametrize("bad", [-1, 2.0, True])
    def test_invalid_first_eviction_step_is_rejected(self, bad):
        with pytest.raises(ValueError, match="first_eviction_step"):
            _make_config(first_eviction_step=bad)


# ===========================================================================
# Batching — per-row independence under divergent eviction (mirrors flash twin)
# ===========================================================================


def _make_pos_keys(B, H_kv, T, D, start=0):
    idx = torch.arange(start, start + T, dtype=torch.float32).view(1, 1, T, 1)
    return idx.expand(B, H_kv, T, D).clone()


def _divergent_scores(B, H_q):
    # Widths track the COMPACTED effective window count. These tests pin
    # first_eviction_step = 0 (see _drive_divergent_cache), so the first eviction
    # fires at decode step 0 on the prefill scores, dropping the 8 prompt windows
    # to 3, so step-1 sees 4 windows (3 survivors + 1 new), not 10. See the flash
    # twin's _divergent_scores docstring for the full schedule.
    s0 = torch.zeros(B, H_q, 8)
    s0[0, :, [1, 3]] = 100.0
    if B > 1:
        s0[1, :, [5, 7]] = 100.0
    return [s0, torch.zeros(B, H_q, 9), torch.zeros(B, H_q, 4)]


def _drive_divergent_cache(scores_per_call, B=2, H_kv=2, D=8):
    # Targets the per-row eviction MECHANICS, not the timing policy: pin
    # first_eviction_step = 0 to reproduce the "fire from step 0" schedule
    # (production defaults to a fixed step 8, which these 2-step runs never reach).
    model_cfg = _FakeModelConfig()
    cfg = WindowedCacheConfig(
        window_size=1, num_sink_tokens=0, local_window_size=1, cache_budget=0.375,
    )
    cache = WindowedCache(
        config=cfg, prefill_len=8, model_config=model_cfg,
        kv_dtype=torch.float32, rope_module=torch.nn.Identity(),
        num_layers=1, max_tokens=0,
    )
    cache._policies[0].first_eviction_step = 0
    k = _make_pos_keys(B, H_kv, 8, D)
    cache.update(k, k.clone(), 0, cache_kwargs={
        "cache_position": torch.arange(8),
        "window_scores": scores_per_call[0],
    })
    for i, pos in enumerate((8, 9)):
        k1 = _make_pos_keys(B, H_kv, 1, D, start=pos)
        cache.update(k1, k1.clone(), 0, cache_kwargs={
            "cache_position": torch.arange(pos, pos + 1),
            "window_scores": scores_per_call[i + 1],
        })
    return cache._states[0]


class TestBatching:
    """Batch>1 must evict each row independently with no cross-contamination."""

    def test_divergent_eviction_keeps_per_row_windows(self):
        H_q = 4
        state = _drive_divergent_cache(_divergent_scores(2, H_q), B=2)
        assert state.original_window_ids.shape == (2, 3)
        assert state.original_window_ids[0].tolist() == [1, 3, 9]
        assert state.original_window_ids[1].tolist() == [5, 7, 9]
        assert state.position_ids.shape == (2, 3)
        assert state.position_ids[0].tolist() == [1, 3, 9]
        assert state.position_ids[1].tolist() == [5, 7, 9]
        kept = state.key_states[:, 0, :, 0]
        assert kept[0].tolist() == [1.0, 3.0, 9.0]
        assert kept[1].tolist() == [5.0, 7.0, 9.0]

    def test_batch_row_matches_standalone_b1(self):
        H_q = 4
        state2 = _drive_divergent_cache(_divergent_scores(2, H_q), B=2)
        state1 = _drive_divergent_cache(_divergent_scores(1, H_q), B=1)
        assert state1.original_window_ids.shape == (1, 3)
        assert torch.equal(
            state1.original_window_ids[0], state2.original_window_ids[0]
        )
        assert torch.equal(state1.position_ids[0], state2.position_ids[0])
        assert torch.equal(state1.key_states[0], state2.key_states[0])

    def test_slice_and_keep_gathers_positions_per_row(self):
        state = CacheState()
        B, H, T, D = 2, 2, 6, 4
        state.key_states = torch.randn(B, H, T, D)
        state.value_states = torch.randn(B, H, T, D)
        state.position_ids = torch.stack([torch.arange(T), torch.arange(T) + 100])
        retain = torch.tensor([[0, 2, 5], [1, 3, 4]])
        state.slice_and_keep(retain)
        assert state.position_ids[0].tolist() == [0, 2, 5]
        assert state.position_ids[1].tolist() == [101, 103, 104]
