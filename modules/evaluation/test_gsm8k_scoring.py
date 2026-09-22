"""Tests for modules/evaluation/gsm8k_scoring.py.

The regression tests at the bottom (`TestDefensiveKVRegressions`) are the point of
this file: each one is a case where the imported DefensiveKV scorer reports the
wrong number, pinned so it cannot come back.
"""

from __future__ import annotations

import math

import pytest

from modules.evaluation.gsm8k_scoring import (
    Extraction,
    answers_match,
    calculate_metrics,
    extract_answer,
    extract_predicted_number,
    gsm8k_score,
    score_predictions,
    _coerce_answer_cell,
    _normalize_ground_truth,
)


RUNON = (
    "\n\nQuestion:\nJohn buys 5 apples and 7 oranges.\n\n"
    "Reasoning:\n5 + 7 = 12\n\nAnswer: 12"
)


# ---------------------------------------------------------------------------
# Extraction: the formats models actually emit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Answer: 18",
        "Answer: $18",
        "Answer: **18**",
        "**Answer:** 18",
        "Answer:\n18",
        "Answer: 18 dollars",
        "Answer: 18.",
        "Answer: 18.00",
        "The answer is 18.",
        "The final answer is $18",
        "#### 18",
        "\\boxed{18}",
        "Final answer: 18",
        "answer = 18",
    ],
)
def test_marker_formats_extract_18(text):
    ex = extract_answer(text)
    assert ex.value == 18.0
    assert ex.method == "marker", f"{text!r} should be marker-backed, got {ex.method}"


def test_thousands_separator():
    assert extract_answer("Answer: 1,234").value == 1234.0


def test_negative_answer():
    assert extract_answer("Answer: -42").value == -42.0


def test_decimal_answer():
    assert extract_answer("Answer: 3.5").value == 3.5


def test_bare_comma_is_not_a_number():
    """DefensiveKV's `[\\d,]+` matched a lone ',' and produced '' -> score 0."""
    ex = extract_answer("The eggs sell for 18, so she makes 18,")
    assert ex.value == 18.0


def test_no_marker_falls_back_and_is_labelled():
    ex = extract_answer("16 - 3 - 4 = 9\n9 * 2 = 18")
    assert ex.value == 18.0
    assert ex.method == "fallback"
    assert not ex.is_confident


def test_no_number_at_all():
    ex = extract_answer("I am not sure how to solve this.")
    assert ex.value is None
    assert ex.method == "none"


# ---------------------------------------------------------------------------
# Run-on: the core defect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer_line",
    ["Answer: 18", "Answer: $18", "Answer: **18**", "\\boxed{18}", "#### 18"],
)
def test_runon_does_not_change_the_score(answer_line):
    """Same correct answer, with and without a hallucinated follow-up problem."""
    base = f"Reasoning:\n16 - 3 - 4 = 9\n9 * 2 = 18\n\n{answer_line}"
    assert extract_answer(base).value == 18.0
    assert extract_answer(base + RUNON).value == 18.0


def test_runon_is_reported():
    ex = extract_answer("Answer: 18" + RUNON)
    assert ex.runon_trimmed is True


def test_accuracy_is_invariant_to_runon_rate():
    """The headline regression: reported accuracy must not track termination.

    Under the DefensiveKV scorer this same input yields 100/90/75/50/25/0.
    """
    correct = "Reasoning:\n16 - 3 - 4 = 9\n9 * 2 = 18\n\nAnswer: $18"
    for n_runon in (0, 10, 25, 50, 75, 100):
        preds = [correct + RUNON] * n_runon + [correct] * (100 - n_runon)
        rep = score_predictions(preds, [["18"]] * 100)
        assert rep.accuracy == 100.0, f"run-on rate {n_runon}% changed the score"


def test_self_correction_takes_the_last_marker():
    text = "Answer: 12 — wait, that double counts the eggs. Answer: 18"
    assert extract_answer(text).value == 18.0


def test_marker_does_not_reach_across_a_paragraph():
    text = "Answer: see the work above.\n\n" + "x" * 200 + "\n\n99"
    ex = extract_answer(text)
    assert ex.method == "fallback"


# ---------------------------------------------------------------------------
# Generation failures must never look like wrong answers
# ---------------------------------------------------------------------------


def test_failure_string_is_not_scored():
    ex = extract_answer("Failure: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert ex.method == "failed"
    assert ex.value is None


def test_failures_are_excluded_from_accuracy_and_reported():
    preds = ["Answer: 18"] * 8 + ["Failure: CUDA out of memory"] * 2
    rep = score_predictions(preds, [["18"]] * 10)

    assert rep.n_generation_failed == 2
    assert rep.accuracy == 100.0                    # 8/8 scoreable
    assert rep.accuracy_including_failures == 80.0  # old denominator, for comparison
    assert any("FAILED" in w for w in rep.warnings)


def test_all_failures_does_not_silently_report_zero():
    """DefensiveKV published 0.0 for a run that had crashed. Make that visible."""
    rep = score_predictions(["Failure: boom"] * 50, [["18"]] * 50)
    assert rep.n_generation_failed == 50
    assert rep.warnings, "an all-failed run must warn, not report a clean 0.0"


@pytest.mark.parametrize("empty", [None, "", "   ", float("nan")])
def test_empty_predictions_are_bucketed(empty):
    rep = score_predictions([empty, "Answer: 18"], [["18"], ["18"]])
    assert rep.n_empty == 1
    assert rep.accuracy == 100.0


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_fallback_correctness_is_tracked_separately():
    preds = ["Answer: 18"] * 5 + ["9 * 2 = 18"] * 5
    rep = score_predictions(preds, [["18"]] * 10)

    assert rep.n_marker == 5
    assert rep.n_fallback == 5
    assert rep.n_correct_lenient == 10
    assert rep.n_correct_strict == 5
    assert rep.n_correct_from_fallback == 5
    assert rep.accuracy == 100.0
    assert rep.accuracy_strict == 50.0
    assert any("fallback rung" in w for w in rep.warnings)


def test_clean_run_has_no_warnings():
    rep = score_predictions(["Answer: 18"] * 20, [["18"]] * 20)
    assert rep.warnings == []
    assert rep.accuracy == 100.0
    assert rep.accuracy_strict == 100.0


def test_truncation_is_flagged():
    truncated = "Reasoning:\nFirst she collects 16 eggs, then she eats 3, then she bakes with 4, leaving 9, and 9 times"
    rep = score_predictions([truncated] * 10, [["18"]] * 10)
    assert rep.n_looks_truncated == 10
    assert any("truncated" in w for w in rep.warnings)


# ---------------------------------------------------------------------------
# Ground truth / comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gt,expected",
    [
        ("18", 18.0),
        ("1,234", 1234.0),
        ("She has 9 eggs left.\n#### 18", 18.0),
        (18, 18.0),
    ],
)
def test_ground_truth_normalization(gt, expected):
    assert _normalize_ground_truth(gt) == expected


def test_numeric_tolerance():
    assert answers_match(18.0, 18.0)
    assert answers_match(18.00000001, 18.0)
    assert not answers_match(18.0, 19.0)
    assert not answers_match(None, 18.0)


def test_scalar_and_list_answers_both_work():
    assert score_predictions(["Answer: 18"], ["18"]).accuracy == 100.0
    assert score_predictions(["Answer: 18"], [["18"]]).accuracy == 100.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_predictions(["a", "b"], [["1"]])


# ---------------------------------------------------------------------------
# Drop-in compatibility surface
# ---------------------------------------------------------------------------


def test_extract_predicted_number_returns_str():
    assert extract_predicted_number("Answer: 18") == "18"
    assert extract_predicted_number("Answer: 18.5") == "18.5"
    assert extract_predicted_number("no numbers here") == ""


def test_gsm8k_score_signature():
    assert gsm8k_score("Answer: 18", "18") == 1.0
    assert gsm8k_score("Answer: 19", "18") == 0.0


def test_calculate_metrics_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "task": ["gsm8k"] * 4,
            "predicted_answer": ["Answer: 18", "Answer: $18" + RUNON, "Answer: 5", "Failure: oom"],
            "answer": [["18"], ["18"], ["18"], ["18"]],
        }
    )
    metrics = calculate_metrics(df, verbose=False)
    # 3 scoreable, 2 correct -> 66.67; the failure is excluded, not counted wrong.
    assert metrics["gsm8k"] == pytest.approx(66.67, abs=0.01)
    assert metrics["_diagnostics"]["gsm8k"]["n_generation_failed"] == 1


@pytest.mark.parametrize(
    "cell,expected",
    [
        ("['18']", ["18"]),
        ('["18"]', ["18"]),
        ("['18', '19']", ["18", "19"]),
        ("['18' '19']", ["18", "19"]),   # numpy repr — bare eval() raised here
        ("[]", []),
        ("18", "18"),
    ],
)
def test_answer_cell_parsing(cell, expected):
    assert _coerce_answer_cell(cell) == expected


# ---------------------------------------------------------------------------
# Pinned regressions vs. the DefensiveKV implementation
# ---------------------------------------------------------------------------


class TestTruncationFix:
    """The `####` prompt format from gsm8k_dataset.py must score cleanly."""

    def test_dataset_constants(self):
        from modules.evaluation import gsm8k_dataset as ds

        assert ds.MAX_NEW_TOKENS == 512, "256 clips long reasoning before the answer"
        assert "####" in ds.SYSTEM_PROMPT
        assert ds.STOP_STRINGS, "a stop string is what keeps 512 tokens cheap"

    def test_canonical_output_is_marker_backed(self):
        text = "Reasoning:\n16 - 3 - 4 = 9\n9 * 2 = 18\n\n#### 18"
        ex = extract_answer(text)
        assert ex.value == 18.0
        assert ex.method == "marker"
        assert not ex.looks_truncated

    def test_stop_strings_cut_at_the_runon_boundary(self):
        from modules.evaluation import gsm8k_dataset as ds

        for stop in ds.STOP_STRINGS:
            ex = extract_answer("#### 18" + stop + "\nJohn buys 5 apples.\n\n#### 12")
            assert ex.value == 18.0

    def test_clean_run_reports_no_truncation(self):
        rep = score_predictions(["Reasoning:\n9 * 2 = 18\n\n#### 18"] * 20, [["18"]] * 20)
        assert rep.n_looks_truncated == 0
        assert rep.marker_rate == 100.0
        assert rep.warnings == []


class TestDefensiveKVRegressions:
    """Each case: what the old scorer returned, and what it must return now."""

    def test_dollar_answer_with_runon(self):
        # old: 12 (last number in the hallucinated follow-up problem)
        assert extract_answer("Answer: $18" + RUNON).value == 18.0

    def test_bold_answer_with_runon(self):
        # old: 12
        assert extract_answer("Answer: **18**" + RUNON).value == 18.0

    def test_boxed_answer_with_runon(self):
        # old: 12
        assert extract_answer("\\boxed{18}" + RUNON).value == 18.0

    def test_trailing_comma_answer(self):
        # old: "" -> scored 0 despite being right
        assert gsm8k_score("she makes 18,", "18") == 1.0

    def test_oom_message_yields_no_score(self):
        # old: extracted "2.00" from the OOM text and scored it wrong
        assert extract_answer("Failure: CUDA OOM, tried to allocate 2.00 GiB").value is None

    def test_first_vs_last_marker(self):
        # old: 12 (re.search takes the FIRST match)
        assert extract_answer("Answer: 12. Correction. Answer: 18").value == 18.0
