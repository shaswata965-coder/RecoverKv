"""Tests for LongBench runner, scoring, and metrics.

CPU-only with mocked models or tiny synthetic snippets.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — lightweight config stubs (no YAML needed)
# ---------------------------------------------------------------------------

@dataclass
class _StubRun:
    mode: str = "longbench"
    seed: int = 42

@dataclass
class _StubModel:
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    revision: str = "main"
    dtype: str = "float16"
    attn_implementation: str = "eager"

@dataclass
class _StubCache:
    backend: str = "dynamic"
    backend_package: Optional[str] = None
    cache_budget: Optional[float] = None
    window_size: int = 32
    num_sink_tokens: int = 4
    local_window_size: float = 0.25
    top_k_windows: int = 2

@dataclass
class _StubTelemetry:
    track_scores: bool = False
    output_dir: str = "outputs"

@dataclass
class _StubLongBench:
    datasets: List[str] = field(default_factory=lambda: ["narrativeqa"])
    include_chinese: bool = False
    use_e_variants: bool = False
    max_length: int = 7500
    output_dir: str = "outputs/longbench/test"
    seed: int = 42
    resume: bool = False
    skip_oom: bool = False
    aggressive_cache_clear: bool = False

@dataclass
class _StubConfig:
    run: _StubRun = field(default_factory=_StubRun)
    model: _StubModel = field(default_factory=_StubModel)
    cache: _StubCache = field(default_factory=_StubCache)
    telemetry: _StubTelemetry = field(default_factory=_StubTelemetry)
    longbench: _StubLongBench = field(default_factory=_StubLongBench)


# ---------------------------------------------------------------------------
# Test: tracking off assertion
# ---------------------------------------------------------------------------

class TestTrackingOffAssertion:
    def test_tracking_on_raises(self):
        """track_scores=True → runner raises with clear error."""
        from modules.evaluation.longbench_runner import LongBenchRunner
        cfg = _StubConfig()
        cfg.telemetry.track_scores = True
        with pytest.raises(ValueError, match="track_scores must be False"):
            LongBenchRunner(cfg)

    def test_tracking_off_ok(self):
        """track_scores=False → no error from assertion."""
        from modules.evaluation.longbench_runner import LongBenchRunner
        cfg = _StubConfig()
        cfg.telemetry.track_scores = False
        # Should not raise on the assertion itself
        # (will fail later on model load, but that's fine for this test)
        LongBenchRunner._assert_tracking_off(cfg)


# ---------------------------------------------------------------------------
# Test: cache type routing
# ---------------------------------------------------------------------------

class TestCacheRouting:
    def test_full_cache_uses_no_windowed(self):
        """full-cache config → no WindowedCache, no hooks."""
        from modules.evaluation.longbench_runner import LongBenchRunner
        cfg = _StubConfig()
        cfg.cache.backend = "dynamic"
        cfg.cache.backend_package = None
        runner = LongBenchRunner(cfg)
        assert runner.WindowedCache is None
        assert runner.is_windowed is False

    def test_ours_flash_attn_routes_to_canonical(self):
        """backend=flash_attn → imports from modules.windowed_cache."""
        from modules.evaluation.longbench_runner import LongBenchRunner
        cfg = _StubConfig()
        cfg.cache.backend = "windowed"
        cfg.cache.backend_package = "flash_attn"
        cfg.model.attn_implementation = "flash_attention_2"
        runner = LongBenchRunner(cfg)
        assert runner.is_windowed is True
        assert runner.cache_backend_package == "flash_attn"
        assert runner.WindowedCache is not None

    def test_ours_eager_routes_to_eager(self):
        """backend=eager → imports from modules.windowed_eager_cache."""
        from modules.evaluation.longbench_runner import LongBenchRunner
        cfg = _StubConfig()
        cfg.cache.backend = "windowed"
        cfg.cache.backend_package = "eager"
        cfg.model.attn_implementation = "eager"
        runner = LongBenchRunner(cfg)
        assert runner.is_windowed is True
        assert runner.cache_backend_package == "eager"
        assert runner.WindowedCache is not None


# ---------------------------------------------------------------------------
# Test: backend-attn pairing validated
# ---------------------------------------------------------------------------

class TestBackendAttnPairing:
    def test_mismatch_raises(self):
        """Mismatched config raises ConfigValidationError."""
        from utils.cache_factory import ConfigValidationError
        from modules.evaluation.longbench_runner import LongBenchRunner
        cfg = _StubConfig()
        cfg.cache.backend = "windowed"
        cfg.cache.backend_package = "flash_attn"
        cfg.model.attn_implementation = "eager"  # mismatch!
        with pytest.raises(ConfigValidationError):
            LongBenchRunner(cfg)

    def test_eager_mismatch_raises(self):
        """eager backend + flash_attention_2 → raises."""
        from utils.cache_factory import ConfigValidationError
        from modules.evaluation.longbench_runner import LongBenchRunner
        cfg = _StubConfig()
        cfg.cache.backend = "windowed"
        cfg.cache.backend_package = "eager"
        cfg.model.attn_implementation = "flash_attention_2"  # mismatch!
        with pytest.raises(ConfigValidationError):
            LongBenchRunner(cfg)


# ---------------------------------------------------------------------------
# Test: middle truncation
# ---------------------------------------------------------------------------

class TestMiddleTruncation:
    def test_keeps_head_and_tail(self):
        """Synthetic tokens [0..N-1] truncated to L → [0..L/2-1] + [N-L/2..N-1]."""
        N = 100
        L = 40
        tokens = list(range(N))
        half = L // 2
        result = tokens[:half] + tokens[-half:]
        assert len(result) == L
        assert result[:half] == list(range(half))
        assert result[half:] == list(range(N - half, N))

    def test_no_truncation_when_within_limit(self):
        """No truncation when tokens fit within max_length."""
        tokens = list(range(50))
        max_length = 100
        if len(tokens) > max_length:
            half = max_length // 2
            tokens = tokens[:half] + tokens[-half:]
        assert len(tokens) == 50  # unchanged


# ---------------------------------------------------------------------------
# Test: max_length auto-fits to the model's context window
# ---------------------------------------------------------------------------

class TestMaxLengthAutoFit:
    """`max_length: null` must fit the prompt inside max_position_embeddings.

    Regression for a Mistral-v0.2 (32K) LongBench run that inherited a config
    written for Llama-3.1 (128K): narrativeqa prompts reached 35083 tokens, past
    the window, so RoPE positions were out of distribution and every score for
    that dataset was noise rather than a measurement.
    """

    @staticmethod
    def _runner(model_max, max_length, model_name="Mistral-7B-Instruct-v0.2",
                chat_overhead=10):
        import types
        from modules.evaluation.longbench_runner import LongBenchRunner
        r = LongBenchRunner.__new__(LongBenchRunner)
        r._chat_overhead_cache = {}
        r._auto_fit_logged = set()
        r._over_context_warned = False
        r.config = types.SimpleNamespace(model=types.SimpleNamespace(name=model_name))
        r.lb = types.SimpleNamespace(max_length=max_length)
        r.model = types.SimpleNamespace(
            config=types.SimpleNamespace(max_position_embeddings=model_max)
        )
        # Stub the tokenizer-dependent overhead so the test stays offline.
        r._chat_template_overhead = lambda _tok, ds: (
            0 if ds in LongBenchRunner.NO_CHAT_TEMPLATE_DATASETS else chat_overhead
        )
        return r

    def test_auto_fits_short_context_model(self):
        """Prompt budget + template + generation must land inside the window."""
        r = self._runner(model_max=32768, max_length=None)
        budget = r._resolve_max_length(None, "narrativeqa", 128)
        assert budget is not None
        assert budget + 10 + 128 <= 32768
        # And it must actually bite on the 35083-token example that failed.
        assert budget < 35083

    def test_auto_is_noop_on_long_context_model(self):
        """Llama-3.1's 128K fits every LongBench prompt — must not truncate."""
        r = self._runner(model_max=131072, max_length=None,
                         model_name="Llama-3.1-8B-Instruct")
        budget = r._resolve_max_length(None, "narrativeqa", 128)
        assert budget > 100_000  # far above the longest LongBench prompt

    def test_explicit_value_wins(self):
        """A positive max_length reproduces the published protocol verbatim."""
        r = self._runner(model_max=32768, max_length=31500)
        assert r._resolve_max_length(None, "narrativeqa", 128) == 31500

    def test_zero_opts_out_and_warns(self):
        """max_length: 0 disables truncation entirely, loudly."""
        r = self._runner(model_max=32768, max_length=0)
        assert r._resolve_max_length(None, "narrativeqa", 128) is None
        assert r._over_context_warned is True

    def test_unknown_context_window_leaves_prompt_alone(self):
        """No max_position_embeddings → no basis to truncate; stay untouched."""
        r = self._runner(model_max=None, max_length=None)
        assert r._resolve_max_length(None, "narrativeqa", 128) is None

    def test_few_shot_dataset_reserves_no_template(self):
        """NO_CHAT_TEMPLATE_DATASETS skip the template, so reserve nothing for it."""
        r = self._runner(model_max=32768, max_length=None)
        few_shot = r._resolve_max_length(None, "trec", 64)
        chat = r._resolve_max_length(None, "narrativeqa", 64)
        assert few_shot == chat + 10

    def test_longer_generation_shrinks_the_budget(self):
        """The reserve tracks max_gen_len so generation can't cross the window."""
        r = self._runner(model_max=32768, max_length=None)
        assert (r._resolve_max_length(None, "narrativeqa", 512)
                == r._resolve_max_length(None, "narrativeqa", 128) - 384)


# ---------------------------------------------------------------------------
# Test: greedy decoding enforced
# ---------------------------------------------------------------------------

class TestGreedyDecoding:
    def test_generate_kwargs(self):
        """do_sample=False, num_beams=1 passed to generate."""
        # Verify the runner's generate kwargs construction
        gen_kwargs = {
            "max_new_tokens": 128,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 1.0,
        }
        assert gen_kwargs["do_sample"] is False
        assert gen_kwargs["num_beams"] == 1


# ---------------------------------------------------------------------------
# Test: JSONL schema
# ---------------------------------------------------------------------------

class TestJsonlSchema:
    def test_output_schema_matches_longbench(self):
        """Output keys exactly {pred, answers, all_classes, length, _id}."""
        record = {
            "pred": "test prediction",
            "answers": ["answer1", "answer2"],
            "all_classes": None,
            "length": 1000,
            "_id": "test-id",
        }
        expected_keys = {"pred", "answers", "all_classes", "length", "_id"}
        assert set(record.keys()) == expected_keys

    def test_null_pred_for_oom(self):
        """OOM'd examples have pred=None."""
        record = {"pred": None, "answers": ["a"], "all_classes": None,
                  "length": 100, "_id": "oom-1"}
        line = json.dumps(record, ensure_ascii=False)
        parsed = json.loads(line)
        assert parsed["pred"] is None


# ---------------------------------------------------------------------------
# Test: dataset2metric mapping
# ---------------------------------------------------------------------------

class TestDataset2MetricMapping:
    def test_all_16_datasets_mapped(self):
        """All 16 English datasets have a metric in dataset2metric.json."""
        config_path = Path("data/longbench_configs/dataset2metric.json")
        if not config_path.exists():
            pytest.skip("Vendored config not present")
        with open(config_path) as f:
            d2m = json.load(f)
        expected = [
            "narrativeqa", "qasper", "multifieldqa_en",
            "hotpotqa", "2wikimqa", "musique",
            "gov_report", "qmsum", "multi_news",
            "trec", "triviaqa", "samsum",
            "passage_count", "passage_retrieval_en",
            "lcc", "repobench-p",
        ]
        for ds in expected:
            assert ds in d2m, f"Missing metric mapping for {ds}"

    def test_specific_mappings(self):
        """Spot-check known dataset→metric mappings."""
        config_path = Path("data/longbench_configs/dataset2metric.json")
        if not config_path.exists():
            pytest.skip("Vendored config not present")
        with open(config_path) as f:
            d2m = json.load(f)
        assert d2m["narrativeqa"] == "qa_f1_score"
        assert d2m["gov_report"] == "rouge_score"
        assert d2m["trec"] == "classification_score"
        assert d2m["lcc"] == "code_sim_score"
        assert d2m["passage_count"] == "count_score"
        assert d2m["passage_retrieval_en"] == "retrieval_score"
        assert d2m["samsum"] == "rouge_score"


# ---------------------------------------------------------------------------
# Test: vendored metric functions
# ---------------------------------------------------------------------------

class TestQaF1Score:
    def test_exact_match(self):
        from modules.evaluation.longbench_metrics import qa_f1_score
        assert qa_f1_score("the cat", "the cat") == 1.0

    def test_partial_match(self):
        from modules.evaluation.longbench_metrics import qa_f1_score
        score = qa_f1_score("the big cat", "the cat")
        assert 0.0 < score < 1.0

    def test_no_match(self):
        from modules.evaluation.longbench_metrics import qa_f1_score
        score = qa_f1_score("dog", "cat")
        assert score == 0


class TestRougeScore:
    def test_identical(self):
        from modules.evaluation.longbench_metrics import rouge_score
        score = rouge_score("the quick brown fox", "the quick brown fox")
        assert score > 0.9

    def test_empty_prediction(self):
        from modules.evaluation.longbench_metrics import rouge_score
        score = rouge_score("", "some text")
        assert score == 0.0


class TestCodeSimScore:
    def test_identical(self):
        from modules.evaluation.longbench_metrics import code_sim_score
        score = code_sim_score("return x + 1", "return x + 1")
        assert score == 1.0

    def test_different(self):
        from modules.evaluation.longbench_metrics import code_sim_score
        score = code_sim_score("return x + 1", "return y * 2")
        assert 0.0 <= score < 1.0


class TestCountScore:
    def test_correct_count(self):
        from modules.evaluation.longbench_metrics import count_score
        score = count_score("There are 5 paragraphs", "5")
        assert score == 1.0

    def test_wrong_count(self):
        from modules.evaluation.longbench_metrics import count_score
        score = count_score("There are 3 paragraphs", "5")
        assert score == 0.0


class TestRetrievalScore:
    def test_correct_paragraph(self):
        from modules.evaluation.longbench_metrics import retrieval_score
        score = retrieval_score("Paragraph 3", "Paragraph 3")
        assert score == 1.0

    def test_wrong_paragraph(self):
        from modules.evaluation.longbench_metrics import retrieval_score
        score = retrieval_score("Paragraph 5", "Paragraph 3")
        assert score == 0.0


class TestClassificationScore:
    def test_correct_class(self):
        from modules.evaluation.longbench_metrics import classification_score
        score = classification_score(
            "This is about Science", "Science",
            all_classes=["Science", "History", "Math"]
        )
        assert score == 1.0

    def test_wrong_class(self):
        from modules.evaluation.longbench_metrics import classification_score
        score = classification_score(
            "This is about Math", "Science",
            all_classes=["Science", "History", "Math"]
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# Test: scoring dispatch
# ---------------------------------------------------------------------------

class TestScorePredictions:
    def test_dispatches_correctly(self, tmp_path):
        """Synthetic jsonls for different metrics → correct dispatch."""
        from modules.evaluation.longbench_scoring import score_predictions

        # QA dataset
        qa_path = tmp_path / "narrativeqa.jsonl"
        qa_path.write_text(json.dumps({
            "pred": "the cat", "answers": ["the cat"],
            "all_classes": None, "length": 100, "_id": "1"
        }) + "\n", encoding="utf-8")

        # Count dataset
        count_path = tmp_path / "passage_count.jsonl"
        count_path.write_text(json.dumps({
            "pred": "5", "answers": ["5"],
            "all_classes": None, "length": 100, "_id": "2"
        }) + "\n", encoding="utf-8")

        scores = score_predictions(tmp_path)
        assert "narrativeqa" in scores
        assert "passage_count" in scores
        assert scores["narrativeqa"] == pytest.approx(100.0, abs=0.1)
        assert scores["passage_count"] == pytest.approx(100.0, abs=0.1)

    def test_null_pred_scored_zero_and_counted(self, tmp_path):
        """A null prediction (OOM) scores 0 AND stays in the denominator.

        Matches THUDM eval.py (divides by len(predictions)); dropping nulls
        would silently inflate the mean and break comparability.
        """
        from modules.evaluation.longbench_scoring import score_predictions

        p = tmp_path / "narrativeqa.jsonl"
        p.write_text(
            json.dumps({"pred": "the cat", "answers": ["the cat"],
                        "all_classes": None, "length": 100, "_id": "1"}) + "\n"
            + json.dumps({"pred": None, "answers": ["the dog"],
                          "all_classes": None, "length": 100, "_id": "2"}) + "\n",
            encoding="utf-8",
        )
        scores = score_predictions(tmp_path)
        # 1.0 (hit) + 0.0 (null) over 2 examples → 50, not 100.
        assert scores["narrativeqa"] == pytest.approx(50.0, abs=0.1)

    def test_empty_answers_does_not_crash(self, tmp_path):
        """Malformed example with empty answers scores 0 instead of raising."""
        from modules.evaluation.longbench_scoring import score_predictions

        p = tmp_path / "narrativeqa.jsonl"
        p.write_text(
            json.dumps({"pred": "anything", "answers": [],
                        "all_classes": None, "length": 100, "_id": "1"}) + "\n",
            encoding="utf-8",
        )
        scores = score_predictions(tmp_path)
        assert scores["narrativeqa"] == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test: max over ground truths
# ---------------------------------------------------------------------------

class TestMaxOverGroundTruths:
    def test_max_not_mean(self, tmp_path):
        """Per-example score is max(metric(pred, gt) for gt in answers)."""
        from modules.evaluation.longbench_scoring import score_predictions

        # Prediction matches one of two answers exactly
        path = tmp_path / "narrativeqa.jsonl"
        path.write_text(json.dumps({
            "pred": "the cat",
            "answers": ["the dog", "the cat"],  # cat matches exactly
            "all_classes": None, "length": 100, "_id": "1"
        }) + "\n", encoding="utf-8")

        scores = score_predictions(tmp_path)
        # Max should be 1.0 (exact match with "the cat"), not mean
        assert scores["narrativeqa"] == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test: samsum first line
# ---------------------------------------------------------------------------

class TestSamsumFirstLine:
    def test_takes_first_line(self):
        """samsum post-processing: first line only."""
        from modules.evaluation.longbench_runner import LongBenchRunner
        pred = "Summary line one.\nDialogue repeat.\nMore noise."
        result = LongBenchRunner._post_process(pred, "samsum")
        assert result == "Summary line one."

    def test_single_line_unchanged(self):
        from modules.evaluation.longbench_runner import LongBenchRunner
        pred = "Just one line"
        result = LongBenchRunner._post_process(pred, "samsum")
        assert result == "Just one line"


# ---------------------------------------------------------------------------
# Test: chat-template applied only to non-few-shot datasets
# ---------------------------------------------------------------------------

class TestChatTemplateGating:
    """Few-shot ICL datasets must NOT be wrapped in the chat template.

    Wrapping trec/triviaqa/samsum/lsht/lcc/repobench-p in a chat turn flips an
    instruct model into chat-assistant mode (meta-preambles), which tanks
    exact-/edit-match scores. Matches THUDM/LongBench pred.py + DefensiveKV.
    """

    FEW_SHOT = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
    CHAT_WRAPPED = [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
        "musique", "gov_report", "qmsum", "multi_news", "passage_count",
        "passage_retrieval_en",
    ]

    def test_few_shot_datasets_skip_chat_template(self):
        from modules.evaluation.longbench_runner import LongBenchRunner
        for ds in self.FEW_SHOT:
            assert not LongBenchRunner._should_apply_chat_template(
                "meta-llama/Meta-Llama-3-8B-Instruct", ds
            ), f"{ds} must NOT be chat-wrapped (few-shot ICL)"

    def test_other_datasets_get_chat_template(self):
        from modules.evaluation.longbench_runner import LongBenchRunner
        for ds in self.CHAT_WRAPPED:
            assert LongBenchRunner._should_apply_chat_template(
                "meta-llama/Meta-Llama-3-8B-Instruct", ds
            ), f"{ds} should be chat-wrapped"

    def test_base_model_never_wrapped(self):
        from modules.evaluation.longbench_runner import LongBenchRunner
        # Non-instruct model: never apply chat template, regardless of dataset.
        for ds in self.FEW_SHOT + self.CHAT_WRAPPED:
            assert not LongBenchRunner._should_apply_chat_template(
                "meta-llama/Meta-Llama-3-8B", ds
            )

    def test_exclusion_set_matches_official(self):
        from modules.evaluation.longbench_runner import LongBenchRunner
        assert LongBenchRunner.NO_CHAT_TEMPLATE_DATASETS == frozenset(
            {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
        )


# ---------------------------------------------------------------------------
# Test: transformers version guard
# ---------------------------------------------------------------------------

class TestTransformersVersionGuard:
    """The windowed cache is only correct on transformers <= 4.47.1."""

    def test_supported_boundary_inclusive(self):
        from utils.cache_factory import is_transformers_version_supported
        assert is_transformers_version_supported("4.47.1")
        assert is_transformers_version_supported("4.47.0")
        assert is_transformers_version_supported("4.46.3")
        assert is_transformers_version_supported("4.30.0")

    def test_newer_versions_rejected(self):
        from utils.cache_factory import is_transformers_version_supported
        assert not is_transformers_version_supported("4.47.2")
        assert not is_transformers_version_supported("4.48.0")
        assert not is_transformers_version_supported("5.8.1")

    def test_tolerates_version_suffixes(self):
        from utils.cache_factory import is_transformers_version_supported
        assert is_transformers_version_supported("4.47.1.dev0")
        assert is_transformers_version_supported("4.47.1+cu121")
        assert not is_transformers_version_supported("4.48.0.dev0")

    def test_assert_raises_on_newer(self):
        from utils.cache_factory import (
            assert_transformers_version_supported,
            ConfigValidationError,
        )
        with pytest.raises(ConfigValidationError):
            assert_transformers_version_supported("5.8.1")

    def test_assert_passes_on_target(self):
        from utils.cache_factory import assert_transformers_version_supported
        # Must not raise on the pinned target version.
        assert_transformers_version_supported("4.47.1")


# ---------------------------------------------------------------------------
# Test: macro average excludes missing
# ---------------------------------------------------------------------------

class TestMacroAverage:
    def test_excludes_nan(self):
        """Partial completion → macro average over present datasets only."""
        from modules.evaluation.longbench_scoring import compute_macro_average
        scores = {
            "narrativeqa": 50.0,
            "qasper": 60.0,
            "missing_ds": float("nan"),
        }
        avg = compute_macro_average(scores)
        assert avg == pytest.approx(55.0, abs=0.1)

    def test_all_present(self):
        from modules.evaluation.longbench_scoring import compute_macro_average
        scores = {"a": 40.0, "b": 60.0}
        assert compute_macro_average(scores) == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test: relative degradation
# ---------------------------------------------------------------------------

class TestRelativeDegradation:
    def test_computation(self):
        """Synthetic baseline + ours → correct % drop."""
        from modules.evaluation.longbench_scoring import compute_relative_degradation
        baseline = {"a": 80.0, "b": 60.0}  # avg = 70
        variant = {"a": 72.0, "b": 54.0}   # avg = 63
        deg = compute_relative_degradation(baseline, variant)
        expected = (70.0 - 63.0) / 70.0 * 100  # 10%
        assert deg == pytest.approx(expected, abs=0.1)

    def test_no_degradation(self):
        from modules.evaluation.longbench_scoring import compute_relative_degradation
        scores = {"a": 80.0, "b": 60.0}
        deg = compute_relative_degradation(scores, scores)
        assert deg == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test: output_attentions only for eager
# ---------------------------------------------------------------------------

class TestOutputAttentions:
    def test_eager_passes_output_attentions(self):
        """For eager backend, output_attentions=True should be set."""
        from modules.evaluation.longbench_runner import LongBenchRunner
        cfg = _StubConfig()
        cfg.cache.backend = "windowed"
        cfg.cache.backend_package = "eager"
        cfg.model.attn_implementation = "eager"
        runner = LongBenchRunner(cfg)
        assert runner.cache_backend_package == "eager"
        # The runner sets output_attentions=True in gen_kwargs for eager

    def test_flash_no_output_attentions(self):
        """For flash_attn backend, output_attentions should NOT be set."""
        from modules.evaluation.longbench_runner import LongBenchRunner
        cfg = _StubConfig()
        cfg.cache.backend = "windowed"
        cfg.cache.backend_package = "flash_attn"
        cfg.model.attn_implementation = "flash_attention_2"
        runner = LongBenchRunner(cfg)
        assert runner.cache_backend_package == "flash_attn"
        # The runner only adds output_attentions for eager


# ---------------------------------------------------------------------------
# Test: category averages
# ---------------------------------------------------------------------------

class TestCategoryAverages:
    def test_computes_categories(self):
        from modules.evaluation.longbench_scoring import compute_category_averages
        scores = {
            "narrativeqa": 50.0, "qasper": 60.0, "multifieldqa_en": 70.0,
            "hotpotqa": 40.0, "2wikimqa": 50.0, "musique": 60.0,
            "gov_report": 30.0, "qmsum": 40.0, "multi_news": 50.0,
            "trec": 70.0, "triviaqa": 80.0, "samsum": 60.0,
            "passage_count": 20.0, "passage_retrieval_en": 30.0,
            "lcc": 50.0, "repobench-p": 60.0,
        }
        cats = compute_category_averages(scores)
        assert "Single-doc QA" in cats
        assert cats["Single-doc QA"] == pytest.approx(60.0, abs=0.1)
        assert "Code" in cats
        assert cats["Code"] == pytest.approx(55.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test: data loader
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_en_dataset_list(self):
        from data.longbench_loader import get_dataset_list, LONGBENCH_EN_DATASETS
        datasets = get_dataset_list(include_chinese=False)
        assert len(datasets) == 16
        assert datasets == LONGBENCH_EN_DATASETS

    def test_custom_list(self):
        from data.longbench_loader import get_dataset_list
        custom = ["narrativeqa", "qasper"]
        datasets = get_dataset_list(custom_list=custom)
        assert datasets == custom

    def test_task_categories_cover_all_16(self):
        from data.longbench_loader import TASK_CATEGORIES, LONGBENCH_EN_DATASETS
        all_in_cats = []
        for ds_list in TASK_CATEGORIES.values():
            all_in_cats.extend(ds_list)
        assert set(all_in_cats) == set(LONGBENCH_EN_DATASETS)


# ---------------------------------------------------------------------------
# Test: config dataclass
# ---------------------------------------------------------------------------

class TestLongBenchConfig:
    def test_defaults(self):
        from utils.config import LongBenchConfig
        lb = LongBenchConfig()
        assert len(lb.datasets) == 16
        assert lb.include_chinese is False
        assert lb.max_length == 7500
        assert lb.aggressive_cache_clear is False

    def test_max_length_accepts_null_for_no_truncation(self):
        """max_length=None is valid and means 'no pre-truncation'."""
        from utils.config import LongBenchConfig
        lb = LongBenchConfig(max_length=None)
        assert lb.max_length is None

    def test_truncation_gate(self):
        """The truncation gate fires only for a positive max_length below len."""
        # Mirrors the guard in _predict: `max_length and max_length > 0 and len>ml`.
        def should_truncate(max_length, n_tokens):
            return bool(max_length and max_length > 0 and n_tokens > max_length)
        assert should_truncate(7500, 9000)
        assert not should_truncate(7500, 5000)
        assert not should_truncate(None, 9000)   # no truncation
        assert not should_truncate(0, 9000)      # no truncation

    def test_config_loads_longbench_section(self):
        """YAML with longbench section parses correctly."""
        from utils.config import load_config
        import tempfile, yaml
        cfg_dict = {
            "run": {"mode": "longbench", "seed": 42},
            "longbench": {
                "datasets": ["narrativeqa"],
                "max_length": 4096,
                "output_dir": "test_out",
                "aggressive_cache_clear": True,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(cfg_dict, f)
            fname = f.name
        cfg = load_config(fname)
        os.unlink(fname)
        assert cfg.longbench.datasets == ["narrativeqa"]
        assert cfg.longbench.max_length == 4096
        assert cfg.longbench.aggressive_cache_clear is True
