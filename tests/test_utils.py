"""Tests for utils/ — seed, hashing, config, env_capture, cache_factory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.cache_factory import ConfigValidationError, validate_backend_attn_pairing
from utils.config import (
    CacheConfig,
    ConfigValidationError as CfgValidationError,
    ExperimentConfig,
    ParityValidationError,
    load_config,
    validate_parity_pair,
)
from utils.env_capture import capture_environment
from utils.hashing import sha256_file, sha256_string, sha256_tokenizer
from utils.seed import SeedContext, seed_everything


# -----------------------------------------------------------------------
# seed.py
# -----------------------------------------------------------------------


class TestSeedEverything:
    def test_sets_python_hash_seed(self) -> None:
        seed_everything(42)
        assert os.environ["PYTHONHASHSEED"] == "42"

    def test_torch_deterministic_enabled(self) -> None:
        seed_everything(0)
        assert torch.are_deterministic_algorithms_enabled()

    def test_reproducible_torch_rand(self) -> None:
        seed_everything(123)
        a = torch.rand(5)
        seed_everything(123)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_reproducible_numpy_rand(self) -> None:
        seed_everything(77)
        a = np.random.rand(5)
        seed_everything(77)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_rejects_negative_seed(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            seed_everything(-1)


class TestSeedContext:
    def test_restores_torch_state(self) -> None:
        seed_everything(10)
        before = torch.rand(3)
        seed_everything(10)
        # Now consume the same 3 values
        _ = torch.rand(3)

        with SeedContext(99):
            inside = torch.rand(3)

        after = torch.rand(3)
        # After context exit, state should continue from where it was
        # (we consumed 3 values before entering context)
        # The inside values should differ from before/after
        assert not torch.equal(inside, before)


# -----------------------------------------------------------------------
# hashing.py
# -----------------------------------------------------------------------


class TestSha256File:
    def test_file_hash_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!")
        h1 = sha256_file(f)
        h2 = sha256_file(f)
        assert h1 == h2
        assert len(h1) == 64  # Full hex digest

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            sha256_file("/nonexistent/path.txt")


class TestSha256String:
    def test_truncated_to_16(self) -> None:
        result = sha256_string("test")
        assert len(result) == 16

    def test_deterministic(self) -> None:
        assert sha256_string("abc") == sha256_string("abc")

    def test_different_inputs_different_hashes(self) -> None:
        assert sha256_string("a") != sha256_string("b")


class TestSha256Tokenizer:
    def test_with_mock_tokenizer(self) -> None:
        class MockTokenizer:
            def get_vocab(self):
                return {"hello": 0, "world": 1, "!": 2}

        h1 = sha256_tokenizer(MockTokenizer())
        h2 = sha256_tokenizer(MockTokenizer())
        assert h1 == h2
        assert len(h1) == 64

    def test_different_vocab_different_hash(self) -> None:
        class TokA:
            def get_vocab(self):
                return {"a": 0}

        class TokB:
            def get_vocab(self):
                return {"b": 0}

        assert sha256_tokenizer(TokA()) != sha256_tokenizer(TokB())


# -----------------------------------------------------------------------
# env_capture.py
# -----------------------------------------------------------------------


class TestEnvCapture:
    def test_returns_dict_with_required_keys(self) -> None:
        env = capture_environment()
        assert "transformers_version" in env
        assert "torch_version" in env
        assert "flash_attn_version" in env  # May be None
        assert "cuda_version" in env
        assert "gpu_name" in env
        assert "commit_sha" in env

    def test_flash_attn_version_is_string_or_none(self) -> None:
        env = capture_environment()
        v = env["flash_attn_version"]
        assert v is None or isinstance(v, str)


# -----------------------------------------------------------------------
# config.py
# -----------------------------------------------------------------------


class TestCacheConfig:
    def test_rejects_int_budget(self) -> None:
        with pytest.raises(CfgValidationError, match="float ratio"):
            CacheConfig(cache_budget=40)

    def test_rejects_budget_out_of_range(self) -> None:
        with pytest.raises(CfgValidationError, match="in \\(0, 1\\]"):
            CacheConfig(cache_budget=1.5)

    def test_accepts_valid_budget(self) -> None:
        cfg = CacheConfig(cache_budget=0.4)
        assert cfg.cache_budget == 0.4

    def test_rejects_non_multiple_local_window(self) -> None:
        with pytest.raises(CfgValidationError, match="multiple of window_size"):
            CacheConfig(local_window_size=7, window_size=8)

    def test_resolve_local_window_int(self) -> None:
        cfg = CacheConfig(local_window_size=16, window_size=8)
        # int local is taken verbatim, independent of the budget argument
        assert cfg.resolve_local_window_size(100) == 16

    def test_resolve_local_window_percentage(self) -> None:
        """Float local is a fraction of the cache BUDGET: 95 budget tokens,
        0.25 → ceil(23.75)=24 → snap to 25 (window_size 5)."""
        cfg = CacheConfig(local_window_size=0.25, window_size=5)
        result = cfg.resolve_local_window_size(95)
        assert result == 25

    def test_resolve_local_window_snaps_up(self) -> None:
        cfg = CacheConfig(local_window_size=0.10, window_size=8)
        # budget_tokens=100: 0.10 * 100 = 10 → ceil=10 → snap to 16
        result = cfg.resolve_local_window_size(100)
        assert result == 16
        assert result % 8 == 0


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(
            """
run:
  mode: parity_base
  seed: 42
model:
  name: test-model
  dtype: float16
"""
        )
        config = load_config(cfg_file)
        assert config.run.mode == "parity_base"
        assert config.model.name == "test-model"

    def test_load_with_inheritance(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(
            """
run:
  seed: 42
model:
  name: base-model
"""
        )
        child = tmp_path / "child.yaml"
        child.write_text(
            """
_base_: base.yaml
model:
  name: child-model
"""
        )
        config = load_config(child)
        assert config.run.seed == 42  # Inherited
        assert config.model.name == "child-model"  # Overridden

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent.yaml")

    def test_window_geometry_under_window_block_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """`cache:` is the single source of truth for window geometry.

        Regression: window_size/num_sink_tokens/local_window_size were declared on
        BOTH CacheConfig and WindowConfig with different defaults (8/4/0.25 vs
        32/4/256). LongBench and perf read cache.*, the parity suites read
        window.*, so a config that set only one block silently ran the other's
        values — a whole sweep could be labelled ws=32 while running ws=64.
        This must fail at load, not warn.
        """
        cfg_file = tmp_path / "drifted.yaml"
        cfg_file.write_text(
            """
run:
  mode: longbench
cache:
  window_size: 64
  num_sink_tokens: 5
window:
  window_size: 32
  num_sink_tokens: 4
"""
        )
        with pytest.raises(CfgValidationError) as exc:
            load_config(cfg_file)
        msg = str(exc.value)
        assert "window_size" in msg and "num_sink_tokens" in msg
        assert "cache:" in msg  # tells the user where to move them

    def test_window_block_with_only_top_k_is_accepted(self, tmp_path: Path) -> None:
        """top_k_windows has no CacheConfig equivalent and stays on `window:`."""
        cfg_file = tmp_path / "ok.yaml"
        cfg_file.write_text(
            """
run:
  mode: parity_ours
cache:
  window_size: 64
  num_sink_tokens: 5
window:
  top_k_windows: 7
"""
        )
        config = load_config(cfg_file)
        assert config.window.top_k_windows == 7
        assert config.cache.window_size == 64
        assert config.cache.num_sink_tokens == 5

    def test_empty_window_block_loads(self, tmp_path: Path) -> None:
        """A bare `window:` key parses as None — must not crash the loader."""
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("run:\n  mode: longbench\nwindow:\n")
        assert load_config(cfg_file).window.top_k_windows is None

    def test_resolved_top_k_reads_cache_geometry(self) -> None:
        """WindowConfig.resolved_top_k derives K from cache.*, not its own fields."""
        from utils.config import WindowConfig

        cache = CacheConfig(cache_budget=0.25, window_size=8,
                            num_sink_tokens=4, local_window_size=32)
        w = WindowConfig()
        # budget_tokens = 0.25 * (1000 + 24) = 256; 256 - 4 sink - 32 local = 220
        assert w.resolved_top_k(cache, prefill_len=1000, max_tokens=24) == 220 // 8
        # An explicit override still short-circuits the derivation.
        assert WindowConfig(top_k_windows=3).resolved_top_k(
            cache, prefill_len=1000, max_tokens=24
        ) == 3


class TestValidateParityPair:
    def test_matching_configs_pass(self) -> None:
        base_meta = {
            "seed": 42,
            "dataset": "wikitext-103",
            "article_id": 0,
            "prefill_len": 100,
            "gen_len": 50,
            "model_name": "test-model",
            "model_revision": None,
            "dtype": "float16",
        }
        ours = ExperimentConfig()
        ours.run.seed = 42
        ours.parity.dataset = "wikitext-103"
        ours.parity.article_index = 0
        ours.parity.prefill_len = 100
        ours.parity.gen_len = 50
        ours.model.name = "test-model"
        ours.model.revision = None
        ours.model.dtype = "float16"

        # Should not raise
        validate_parity_pair(base_meta, ours)

    def test_mismatched_seed_raises(self) -> None:
        base_meta = {"seed": 42, "dataset": "wikitext-103", "article_id": 0,
                      "prefill_len": 100, "gen_len": 50,
                      "model_name": "m", "model_revision": None, "dtype": "fp16"}
        ours = ExperimentConfig()
        ours.run.seed = 99

        with pytest.raises(ParityValidationError, match="seed"):
            validate_parity_pair(base_meta, ours)


# -----------------------------------------------------------------------
# cache_factory.py
# -----------------------------------------------------------------------


class TestValidateBackendAttnPairing:
    def test_flash_attn_requires_flash_attention_2(self) -> None:
        # Valid
        validate_backend_attn_pairing("flash_attn", "flash_attention_2")

    def test_flash_attn_rejects_eager(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_backend_attn_pairing("flash_attn", "eager")

    def test_eager_requires_eager(self) -> None:
        validate_backend_attn_pairing("eager", "eager")

    def test_eager_rejects_flash(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_backend_attn_pairing("eager", "flash_attention_2")

    def test_unknown_backend(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_backend_attn_pairing("sdpa", "sdpa")


# -----------------------------------------------------------------------
# first_eviction_step — one default, no literals
# -----------------------------------------------------------------------


class TestFirstEvictionStepDefault:
    """The YAML-side default and the cache-side default must be one number.

    They are declared in two places on purpose (``utils.config`` must not import
    the cache packages — that pulls in the flash-attn hooks), so nothing but a
    test stops them drifting. When they drifted before, a LongBench run silently
    sat at a different operating point than its config claimed.
    """

    def test_utils_default_matches_the_policy_constant(self) -> None:
        from utils.config import FIRST_EVICTION_STEP_DEFAULT
        from modules.windowed_cache.policy import FIRST_EVICTION_STEP
        from modules.windowed_eager_cache.policy import (
            FIRST_EVICTION_STEP as EAGER_FIRST_EVICTION_STEP,
        )

        assert FIRST_EVICTION_STEP_DEFAULT == FIRST_EVICTION_STEP
        assert FIRST_EVICTION_STEP_DEFAULT == EAGER_FIRST_EVICTION_STEP

    def test_default_is_step_zero(self) -> None:
        """Pinned, not incidental: step 0 is the comparison operating point.

        Anything above 0 leaves every answer that finishes inside that window
        measured at full cache whatever ``cache_budget`` says.
        """
        from utils.config import CacheConfig, FIRST_EVICTION_STEP_DEFAULT

        assert FIRST_EVICTION_STEP_DEFAULT == 0
        assert CacheConfig().first_eviction_step == 0

    def test_no_runner_hardcodes_a_fallback_literal(self) -> None:
        """No runner may fall back to a bare number.

        A numeric literal is what silently decoupled the fallback from the
        config default; the fallback must name the shared constant so the two
        cannot disagree. (Nested getattr chains are fine as long as the
        innermost default is the constant, which the second assert covers.)
        """
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        pattern = re.compile(
            r'getattr\(\s*[\w.]+\s*,\s*"first_eviction_step"\s*,\s*([^,)]+)'
        )
        bad, seen = [], []
        for path in sorted((root / "modules" / "evaluation").glob("*_runner.py")):
            for m in pattern.finditer(path.read_text(encoding="utf-8")):
                fallback = m.group(1).strip()
                seen.append(f"{path.name}: {fallback}")
                if re.fullmatch(r"-?\d+", fallback):
                    bad.append(f"{path.name}: {m.group(0)}")
        assert not bad, "numeric first_eviction_step fallback(s): " + "; ".join(bad)
        assert any("FIRST_EVICTION_STEP_DEFAULT" in s for s in seen), (
            "no runner references FIRST_EVICTION_STEP_DEFAULT — did the "
            "constant get inlined away? saw: " + "; ".join(seen)
        )


# -----------------------------------------------------------------------
# Shipped eval configs — the operating point they actually declare
# -----------------------------------------------------------------------


class TestShippedEvalConfigsOperatingPoint:
    """What operating point the shipped eval configs actually declare.

    ``quant_ratio: 0.0`` is a legitimate run — the single-tier fp16 cache, and
    the ablation arm of the paper — so a config sitting there is reported, not
    failed. What it must never be is *silent*: every LongBench / RULER / GSM8K
    config on this branch once shipped at 0.0, so running the shipped set
    measured a different method from the one the branch exists for and nothing in
    the predictions said so. ``utils.config.log_operating_point`` now warns at
    run time; this reports it at test time.

    The remaining checks stay hard, because they are not choices of arm: a
    quant_ratio on a full-cache config is a label claiming something the run does
    not do, and an invalid Q-tier window size cannot produce the numbers it is
    labelled with.
    """

    @staticmethod
    def _eval_configs():
        from pathlib import Path
        from utils.config import load_config

        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "configs").glob("*.yaml")):
            cfg = load_config(str(path))
            if cfg.run.mode in ("longbench", "ruler", "gsm8k"):
                yield path.name, cfg

    def test_windowed_configs_at_q_zero_are_reported(self) -> None:
        """Reports, does not fail: q=0 is the single-tier ablation arm, and
        pinning the shipped set to one arm would make running the other one a
        test failure. Surfaces in the pytest warnings summary so the shipped
        operating point is visible without going and reading eleven YAMLs."""
        import warnings

        single = [
            name for name, cfg in self._eval_configs()
            if cfg.cache.backend == "windowed" and cfg.cache.quant_ratio == 0.0
        ]
        if single:
            warnings.warn(
                f"{len(single)} windowed eval config(s) at quant_ratio=0.0 — these "
                f"measure the SINGLE-TIER fp16 cache, not the two-tier int4 "
                f"method: {', '.join(single)}. Intended for the ablation arm; if "
                f"these are meant to be the method, set quant_ratio > 0.",
                UserWarning,
                stacklevel=2,
            )

    def test_full_cache_configs_do_not_pretend_to_quantize(self) -> None:
        """The baselines are the other half of the comparison: no eviction, no
        Q tier. A quant_ratio > 0 on a dynamic backend is inert, so it would be a
        label claiming something the run does not do."""
        for name, cfg in self._eval_configs():
            if cfg.cache.backend != "windowed":
                assert cfg.cache.quant_ratio == 0.0, name
                assert cfg.cache.cache_budget is None, name

    def test_the_q_tier_geometry_is_valid_wherever_it_is_enabled(self) -> None:
        """int4 packs 2 nibbles to a byte, so window_size must be even.
        A config that enables the Q tier at an invalid window size is a run that
        cannot produce the numbers it is labelled with."""
        for name, cfg in self._eval_configs():
            if cfg.cache.backend == "windowed" and cfg.cache.quant_ratio > 0:
                assert cfg.cache.window_size % 2 == 0, (
                    f"{name}: quant_ratio={cfg.cache.quant_ratio} needs "
                    f"window_size % 2 == 0, got {cfg.cache.window_size}"
                )

    def test_configs_off_the_comparison_operating_point_are_reported(self) -> None:
        """Reports, does not fail — same reasoning as the q=0 check above.

        A delayed first eviction is a legitimate labelled ablation, so the shipped
        set must not be pinned to step 0. But it is the more consequential of the
        two knobs to get wrong by accident: at step N the prompt stays whole
        through decode steps 0..N-1, so every answer that finishes inside that
        window is measured at FULL cache whatever cache_budget says — which on
        LongBench silently pins the six max_gen=32 datasets plus trec and
        narrativeqa to budget-invariant scores.
        """
        import warnings

        delayed = [
            f"{name} (step {cfg.cache.first_eviction_step})"
            for name, cfg in self._eval_configs()
            if cfg.cache.first_eviction_step != 0
        ]
        if delayed:
            warnings.warn(
                f"{len(delayed)} eval config(s) not at first_eviction_step=0: "
                f"{', '.join(delayed)}. Every answer finishing inside that window "
                f"is measured at full cache whatever cache_budget says, so these "
                f"cannot share a table with step-0 runs. Intended as a labelled "
                f"ablation?",
                UserWarning,
                stacklevel=2,
            )
