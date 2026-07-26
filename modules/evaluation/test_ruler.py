"""Tests for the RULER pipeline — scoring metrics, sidecar handling, config.

The RULER path shipped with no tests at all, which is how the memory-sidecar
glob below survived: the scorer's `*.jsonl` sweep picked up the runner's own
`<task>.memory.jsonl` probe output and reported every row of it as a null
prediction.
"""
from __future__ import annotations

import json

import pytest

from data.ruler_loader import RULER_TASKS
from modules.evaluation.ruler_scoring import (
    _task_metric_fn,
    compute_macro_average,
    score_predictions,
    string_match_all,
    string_match_part,
)


# ---------------------------------------------------------------------------
# Metric functions (ported verbatim from DefensiveKV calculate_metrics.py)
# ---------------------------------------------------------------------------


class TestStringMatchMetrics:
    def test_part_scores_one_if_any_reference_matches(self):
        assert string_match_part(["the answer is 42"], [["42", "99"]]) == 100.0

    def test_part_is_case_insensitive(self):
        assert string_match_part(["ANSWER: Foo"], [["foo"]]) == 100.0

    def test_part_scores_zero_when_nothing_matches(self):
        assert string_match_part(["nothing here"], [["42"]]) == 0.0

    def test_all_scores_the_fraction_of_references_found(self):
        # one of two references present -> 50
        assert string_match_all(["contains 42"], [["42", "99"]]) == 50.0
        assert string_match_all(["contains 42 and 99"], [["42", "99"]]) == 100.0

    def test_all_averages_over_examples(self):
        score = string_match_all(["42", "nope"], [["42"], ["99"]])
        assert score == 50.0

    @pytest.mark.parametrize("task", RULER_TASKS)
    def test_every_task_dispatches_to_a_metric(self, task):
        fn = _task_metric_fn(task)
        assert fn in (string_match_part, string_match_all)

    def test_qa_tasks_use_part_and_niah_uses_all(self):
        assert _task_metric_fn("qa_1") is string_match_part
        assert _task_metric_fn("qa_2") is string_match_part
        assert _task_metric_fn("niah_multikey_3") is string_match_all


# ---------------------------------------------------------------------------
# score_predictions over a directory
# ---------------------------------------------------------------------------


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


class TestScorePredictions:
    def test_scores_one_task_file(self, tmp_path):
        _write_jsonl(tmp_path / "niah_multikey_3.jsonl", [
            {"pred": "the number is 8371", "answer": ["8371"]},
            {"pred": "no idea", "answer": ["8371"]},
        ])
        scores = score_predictions(tmp_path)
        assert scores == {"niah_multikey_3": 50.0}
        assert compute_macro_average(scores) == 50.0

    def test_null_predictions_are_dropped_not_zeroed(self, tmp_path):
        """Matches DefensiveKV's calculate_metrics.py, and differs from
        LongBench scoring, which counts nulls as 0 in the denominator."""
        _write_jsonl(tmp_path / "niah_single_1.jsonl", [
            {"pred": "8371", "answer": ["8371"]},
            {"pred": None, "answer": ["8371"]},
        ])
        assert score_predictions(tmp_path) == {"niah_single_1": 100.0}

    def test_memory_sidecar_is_not_scored_as_a_task(self, tmp_path):
        """The whole reason this file exists.

        `ruler.capture_memory: true` writes `<task>.memory.jsonl` beside the
        predictions. It has no `pred` field, so the scorer's `*.jsonl` glob
        used to report every row as a null prediction — a wall of
        "N/N examples had pred=null (OOM/budget-too-small)" warnings that look
        like the run failed.
        """
        _write_jsonl(tmp_path / "niah_multikey_3.jsonl", [
            {"pred": "8371", "answer": ["8371"]},
        ])
        _write_jsonl(tmp_path / "niah_multikey_3.memory.jsonl", [
            {"task": "niah_multikey_3", "total_live": 123, "memo_bytes": 45},
        ])
        scores = score_predictions(tmp_path)
        assert scores == {"niah_multikey_3": 100.0}
        assert "niah_multikey_3.memory" not in scores

    def test_control_characters_are_stripped_before_matching(self, tmp_path):
        _write_jsonl(tmp_path / "niah_single_2.jsonl", [
            {"pred": "\n\t8371\x00", "answer": ["8371"]},
        ])
        assert score_predictions(tmp_path) == {"niah_single_2": 100.0}

    def test_csv_is_written_with_the_expected_columns(self, tmp_path):
        _write_jsonl(tmp_path / "niah_single_1.jsonl", [
            {"pred": "8371", "answer": ["8371"]},
            {"pred": None, "answer": ["8371"]},
        ])
        out = tmp_path / "scores.csv"
        score_predictions(tmp_path, out_csv=out)
        header = out.read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",") == ["task", "num_examples", "score", "dropped"]


# ---------------------------------------------------------------------------
# Config plumbing — the eviction schedule must reach the cache
# ---------------------------------------------------------------------------


class TestRulerConfigPlumbing:
    def test_ruler_config_defaults_to_step_zero_eviction(self):
        from utils.config import CacheConfig, FIRST_EVICTION_STEP_DEFAULT

        assert CacheConfig().first_eviction_step == FIRST_EVICTION_STEP_DEFAULT == 0

    def test_shipped_ruler_config_is_at_step_zero(self):
        """RULER answers are a handful of tokens: a delayed first eviction
        would score every task at full cache whatever cache_budget says."""
        from pathlib import Path
        from utils.config import load_config

        root = Path(__file__).resolve().parents[2]
        cfg = load_config(root / "configs" / "ruler_niah_mk3_omega16.yaml")
        assert cfg.run.mode == "ruler"
        assert cfg.cache.first_eviction_step == 0


# ---------------------------------------------------------------------------
# Prompt format — no silent LLaMA-3 substitution on a non-Llama tokenizer
# ---------------------------------------------------------------------------


class TestPromptFormatHasNoLlama3Fallback:
    """`apply_chat_template` used to sit inside `except Exception:` that fell back
    to a hand-written LLaMA-3 wrapper. Its <|begin_of_text|> / <|start_header_id|>
    markers are ordinary text to any other tokenizer, so on Mistral the prompt was
    silently mis-formatted and still produced plausible-looking scores. Removed
    from LongBench by the mistral fix; these pin the same for RULER and GSM8K.
    """

    class _Stop(Exception):
        """Raised by the stub tokenizer to halt _predict once the prompt is built."""

    def _prompt_reaching_the_tokenizer(self, runner_cls, ex):
        """The exact string *runner_cls* hands to the tokenizer for *ex*.

        The stub ships NO chat template — the case that used to hit the LLaMA-3
        fallback — and aborts tokenization so nothing downstream of the prompt
        needs to be real.
        """
        from unittest.mock import MagicMock

        seen = {}

        def record(prompt, **kwargs):
            seen["prompt"] = prompt
            raise self._Stop

        tok = MagicMock(side_effect=record)
        tok.chat_template = None
        tok.bos_token = "<s>"
        tok.bos_token_id = 1

        runner = runner_cls.__new__(runner_cls)   # no __init__: prompt path only
        runner.model = MagicMock()
        runner.tokenizer = tok
        runner.config = MagicMock()
        runner.gs = MagicMock(max_new_tokens=None)   # GSM8K's config shortcut
        with pytest.raises(self._Stop):
            runner._predict(ex)
        return seen["prompt"]

    def test_ruler_sends_a_raw_prompt_when_there_is_no_template(self):
        from modules.evaluation.ruler_runner import RulerRunner

        prompt = self._prompt_reaching_the_tokenizer(
            RulerRunner,
            {"context": "CTX ", "question": "Q?", "answer_prefix": " A:",
             "max_new_tokens": 4},
        )
        assert prompt == "CTX Q? A:"
        assert "<|" not in prompt

    def test_gsm8k_sends_a_raw_prompt_when_there_is_no_template(self):
        from modules.evaluation.gsm8k_runner import GSM8KRunner

        prompt = self._prompt_reaching_the_tokenizer(
            GSM8KRunner,
            {"context": "CTX ", "question": "Q?", "answer_prefix": " A:",
             "max_new_tokens": 4},
        )
        assert prompt == "CTX Q? A:"
        assert "<|" not in prompt

    def test_a_template_error_is_not_swallowed(self):
        """A tokenizer that HAS a template but fails to render must raise, not
        quietly produce a differently-formatted prompt."""
        from unittest.mock import MagicMock
        from modules.evaluation.ruler_runner import RulerRunner

        tok = MagicMock()
        tok.chat_template = "{{ boom }}"
        tok.apply_chat_template.side_effect = ValueError("template blew up")

        runner = RulerRunner.__new__(RulerRunner)   # no __init__: prompt path only
        runner.model = MagicMock()
        runner.tokenizer = tok
        runner.config = MagicMock()
        ex = {"context": "c", "question": "q", "answer_prefix": "a",
              "max_new_tokens": 4}
        with pytest.raises(ValueError, match="template blew up"):
            runner._predict(ex)


# ---------------------------------------------------------------------------
# Memory summary — both compression framings, not just the memo-inclusive one
# ---------------------------------------------------------------------------


class TestMemorySummary:
    """`<task>.memory_summary.json` is what anyone reads; the jsonl is for
    re-analysis. At B=1 the read memo defaults ON and is the largest line item,
    so a summary carrying only `compression_vs_fp16` reports the two-tier cache
    as *bigger* than fp16."""

    def _summary(self, tmp_path):
        from modules.evaluation.ruler_runner import RulerRunner

        runner = RulerRunner.__new__(RulerRunner)     # no config needed
        runner._memory_reports = [
            {
                "total_live": 20 * 1024, "total_alloc": 22 * 1024,
                "fp_content_live": 3 * 1024, "q_content_live": 1 * 1024,
                "bookkeeping_live": 0, "memo_bytes": 16 * 1024,
                "total_live_excl_memo": 4 * 1024,
                "compression_vs_fp16": 0.8, "reduction_vs_full": 1.6,
                "compression_vs_fp16_excl_memo": 4.0,
                "reduction_vs_full_excl_memo": 8.0,
                "retained_tokens": 128, "observed_context_len": 512,
            }
        ]
        runner._write_memory("niah_single_1", tmp_path)
        return json.loads(
            (tmp_path / "niah_single_1.memory_summary.json").read_text(
                encoding="utf-8")
        )

    def test_summary_carries_both_compression_framings(self, tmp_path):
        s = self._summary(tmp_path)
        for key in ("compression_vs_fp16", "compression_vs_fp16_excl_memo",
                    "reduction_vs_full", "reduction_vs_full_excl_memo",
                    "memo_bytes", "total_live_excl_memo"):
            assert key in s, f"{key} missing from the memory summary"

    def test_the_two_framings_actually_differ_here(self, tmp_path):
        """Pins the failure this guards: memo-inclusive says <1x, excl says 4x."""
        s = self._summary(tmp_path)
        assert s["compression_vs_fp16"]["mean"] < 1.0
        assert s["compression_vs_fp16_excl_memo"]["mean"] == 4.0

    def test_per_example_jsonl_is_still_written(self, tmp_path):
        self._summary(tmp_path)
        lines = (tmp_path / "niah_single_1.memory.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["memo_bytes"] == 16 * 1024
