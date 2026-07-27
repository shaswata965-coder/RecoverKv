"""CPU tests for the GSM8K harness: loader, scorer, budget math, comparison gates.

No model and no network — the HF download is stubbed. What these protect is the
*validity machinery*: the dataset_sha gate, the degenerate-budget detector, the
extraction ladder's ordering. Those are what stop an invalid run from being reported
as a result, so they are the parts that must not silently regress.

    cd evaluation && pytest gsm8k/test_gsm8k.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gsm8k import create_huggingface_dataset as ds
from gsm8k.calculate_metrics import (
    build_comparison_table,
    extract_answer,
    score_predictions,
    score_run_dir,
)
from gsm8k.press_budget import (
    effective_retention,
    preflight,
    resolve_budget,
)

_RAW = [
    {"question": "Janet has 3 eggs and buys 2 more. How many?", "answer": "3+2=5\n#### 5"},
    {"question": "Tom runs 4 miles twice. How far?", "answer": "4*2=8\n#### 8"},
    {"question": "A pie costs $6. Two pies?", "answer": "6*2=12\n#### 12"},
    {"question": "Sam had 10, gave 4 away. Left?", "answer": "10-4=6\n#### 6"},
    {"question": "5 boxes of 7. Total?", "answer": "5*7=35\n#### 35"},
]


@pytest.fixture()
def built_dataset(tmp_path, monkeypatch):
    import datasets as _ds

    monkeypatch.setattr(_ds, "load_dataset", lambda n, c, split: list(_RAW))
    out = tmp_path / "gsm8k_cot"
    ds.build_gsm8k_cot(out)
    return out


def _write_run(run_dir: Path, preds, answers, meta: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, (p, a) in enumerate(zip(preds, answers)):
            f.write(json.dumps({"id": i, "pred": p, "answer": a, "task": "gsm8k"}) + "\n")
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class TestDataset:
    def test_build_writes_all_three_artifacts(self, built_dataset):
        assert (built_dataset / ds.JSONL_NAME).exists()
        assert (built_dataset / ds.MANIFEST_NAME).exists()
        assert (built_dataset / ds.ARROW_SUBDIR).exists()

    def test_manifest_records_cot_prompt_and_512_tokens(self, built_dataset):
        m = ds.read_manifest(built_dataset)
        assert m["prompt_style"] == "chain_of_thought"
        assert m["max_new_tokens"] == 512
        assert "step-by-step" in m["system_prompt"]

    def test_answers_are_bare_numbers(self, built_dataset):
        rows = ds.load_gsm8k_dataset(built_dataset)
        assert [r["answer"] for r in rows] == [["5"], ["8"], ["12"], ["6"], ["35"]]

    def test_sha_is_stable_across_rebuilds(self, tmp_path, monkeypatch):
        import datasets as _ds

        monkeypatch.setattr(_ds, "load_dataset", lambda n, c, split: list(_RAW))
        a = ds.build_gsm8k_cot(tmp_path / "a")
        b = ds.build_gsm8k_cot(tmp_path / "b")
        assert ds.read_manifest(a)["dataset_sha"] == ds.read_manifest(b)["dataset_sha"]

    def test_sha_changes_when_content_changes(self, tmp_path, monkeypatch):
        import datasets as _ds

        monkeypatch.setattr(_ds, "load_dataset", lambda n, c, split: list(_RAW))
        full = ds.build_gsm8k_cot(tmp_path / "full")
        part = ds.build_gsm8k_cot(tmp_path / "part", limit=3)
        assert (
            ds.read_manifest(full)["dataset_sha"]
            != ds.read_manifest(part)["dataset_sha"]
        )

    def test_num_samples_cap_applies_before_sharding(self, built_dataset):
        """A capped sweep must be the SAME problems at every compression ratio."""
        s0 = ds.load_gsm8k_dataset(built_dataset, num_samples=4, shard=0, num_shards=2)
        s1 = ds.load_gsm8k_dataset(built_dataset, num_samples=4, shard=1, num_shards=2)
        assert len(s0) + len(s1) == 4

    def test_shards_partition_without_overlap(self, built_dataset):
        whole = ds.load_gsm8k_dataset(built_dataset)
        parts = [
            ds.load_gsm8k_dataset(built_dataset, shard=i, num_shards=3) for i in range(3)
        ]
        flat = [r for p in parts for r in p]
        assert [r["question"] for r in flat] == [r["question"] for r in whole]

    def test_missing_manifest_names_the_build_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="create_huggingface_dataset"):
            ds.read_manifest(tmp_path / "nope")

    def test_compress_questions_moves_the_problem_into_the_context(self, built_dataset):
        """This is the switch that decides what the compression ratio acts on."""
        ex = ds.load_gsm8k_dataset(built_dataset)[0]
        ctx_on, q_on = ds.split_context_question(ex, compress_questions=True)
        ctx_off, q_off = ds.split_context_question(ex, compress_questions=False)
        assert "Janet" in ctx_on and q_on == ""
        assert "Janet" not in ctx_off and "Janet" in q_off
        assert len(ctx_on) > len(ctx_off)

    def test_stop_strings_do_not_swallow_the_answer_marker(self):
        """`####` introduces the answer, so stopping on it would cut the number off."""
        assert all("####" not in s for s in ds.STOP_STRINGS)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


class TestExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("reasoning\n#### 18", 18.0),
            ("Answer: $18", 18.0),
            ("Answer: **18**", 18.0),
            (r"The final answer is \boxed{18}", 18.0),
            ("Answer: 1,800", 1800.0),
            ("Answer: 18.", 18.0),
            ("#### -5", -5.0),
            ("#### 18.50", 18.5),
        ],
    )
    def test_marker_formats_models_actually_emit(self, text, expected):
        ex = extract_answer(text)
        assert ex.value == expected
        assert ex.method == "marker"

    def test_last_marker_wins_because_models_self_correct(self):
        ex = extract_answer("Answer: 12 -- wait, that double counts. Answer: 18")
        assert ex.value == 18.0

    def test_runon_is_trimmed_before_extraction(self):
        """The defect that makes accuracy track termination instead of correctness."""
        ex = extract_answer("reasoning\n#### 18\n\nQuestion: another one\n#### 12")
        assert ex.value == 18.0
        assert ex.runon_trimmed is True

    def test_bare_comma_is_not_a_number(self):
        ex = extract_answer("the values were 1, 2,")
        assert ex.value == 2.0  # not "" -> not scored 0 by accident

    def test_failures_are_not_scored_as_wrong(self):
        ex = extract_answer("Failure: CUDA out of memory (tried to allocate 2 GiB)")
        assert ex.method == "failed"
        assert ex.value is None

    def test_empty_and_none_are_separated_from_failures(self):
        assert extract_answer(None).method == "empty"
        assert extract_answer("   ").method == "empty"

    def test_fallback_is_labelled_not_hidden(self):
        ex = extract_answer("so the total comes to 42")
        assert ex.value == 42.0
        assert ex.method == "fallback"
        assert ex.is_confident is False

    def test_format_echo_before_the_answer_still_scores(self):
        """A model that restates the format first must not lose its real answer."""
        ex = extract_answer("Format: #### <number>\n\nReasoning: 3+2=5\n#### 5")
        assert ex.value == 5.0


class TestScoring:
    def test_failures_leave_the_denominator(self):
        rep = score_predictions(
            ["#### 5", "Failure: OOM", "#### 8"], [["5"], ["5"], ["8"]]
        )
        assert rep.n_generation_failed == 1
        assert rep.accuracy == 100.0  # 2/2 scoreable
        assert rep.accuracy_including_failures == pytest.approx(66.67, abs=0.01)
        assert any("FAILED" in w for w in rep.warnings)

    def test_strict_excludes_the_guess_rung(self):
        rep = score_predictions(["#### 5", "the total is 8"], [["5"], ["8"]])
        assert rep.accuracy == 100.0
        assert rep.accuracy_strict == 50.0
        assert rep.n_correct_from_fallback == 1

    def test_accuracy_is_immune_to_run_on_rate(self):
        """The property the whole scorer exists to guarantee.

        Same arithmetic, all correct; only the fraction that keeps generating varies.
        A first-match/last-number ladder reports 100/50/0 here.
        """
        clean = "reasoning\n#### 7"
        runon = "reasoning\n#### 7\n\nQuestion: unrelated\nAnswer: 12"
        for n_runon in (0, 5, 10):
            preds = [runon] * n_runon + [clean] * (10 - n_runon)
            rep = score_predictions(preds, [["7"]] * 10)
            assert rep.accuracy == 100.0

    def test_numeric_tolerance(self):
        assert score_predictions(["#### 18.00"], [["18"]]).accuracy == 100.0


# ---------------------------------------------------------------------------
# press budget
# ---------------------------------------------------------------------------


class TestBudget:
    def test_n_kept_matches_scorer_press(self):
        """ScorerPress.compress: n_kept = int(q_len * (1 - compression_ratio))."""
        b = resolve_budget(context_tokens=150, compression_ratio=0.5, window_size=32)
        assert b.n_kept == 75
        assert b.n_pruned == 75
        assert b.binds and not b.degenerate and b.applicable

    def test_degenerate_when_budget_is_the_observation_window(self):
        """cr=0.8 on a 150-token GSM8K prompt: n_kept < window=32.

        29, not 30: ``1 - 0.8`` is 0.19999999999999996 in binary floating point, so
        ``int(150 * (1 - 0.8))`` truncates to 29. That is what ``ScorerPress.compress``
        computes, so it is what this must compute — a "cleaner" formula here would
        silently disagree with the press it is predicting.
        """
        b = resolve_budget(context_tokens=150, compression_ratio=0.8, window_size=32)
        assert b.n_kept == 29
        assert b.degenerate is True
        assert b.free_slots < 0  # DefensiveKV's stage-1 clamp goes negative here

    def test_shrinking_the_window_rescues_a_high_ratio(self):
        b = resolve_budget(context_tokens=150, compression_ratio=0.8, window_size=8)
        assert b.degenerate is False and b.free_slots > 0

    def test_inapplicable_when_context_is_shorter_than_the_window(self):
        """The press asserts q_len > window_size and would raise mid-prefill."""
        assert resolve_budget(20, 0.5, 32).applicable is False

    def test_zero_ratio_is_the_full_cache_control(self):
        b = resolve_budget(150, 0.0, 32)
        assert b.n_kept == 150 and not b.binds and not b.degenerate

    def test_preflight_blocks_a_degenerate_sweep(self):
        rep = preflight([150] * 10, compression_ratio=0.8, window_size=32)
        assert rep.pct_degenerate == 100.0
        reasons = rep.blocking_reasons()
        assert reasons and "observation window" in reasons[0]
        assert any("clamp" in r for r in reasons)

    def test_preflight_passes_a_workable_sweep(self):
        rep = preflight([150] * 10, compression_ratio=0.5, window_size=32)
        assert rep.blocking_reasons() == []

    def test_effective_retention_counts_the_uncompressed_tail(self):
        """The number that stops '0.8 compression' from being quoted end to end."""
        # 150-token context compressed to 30, empty question, 200 generated tokens.
        eff = effective_retention(
            n_kept=30, question_tokens=0, gen_tokens=200, context_tokens=150
        )
        assert eff == pytest.approx(230 / 350, abs=1e-6)
        assert eff > 0.6  # nominal ratio 0.8 -> real saving is ~34%


# ---------------------------------------------------------------------------
# comparison gates
# ---------------------------------------------------------------------------


class TestComparison:
    def test_score_run_dir_reads_jsonl_and_meta(self, tmp_path):
        run = _write_run(
            tmp_path / "r",
            ["reasoning\n#### 5", "reasoning\n#### 9"],
            [["5"], ["8"]],
            {"compression_ratio": None, "dataset_sha": "abc123"},
        )
        rep, meta = score_run_dir(run)
        assert rep.n_total == 2 and rep.accuracy == 50.0
        assert meta["dataset_sha"] == "abc123"

    def test_dataset_sha_mismatch_is_fatal(self, tmp_path):
        _write_run(tmp_path / "a", ["#### 5"], [["5"]],
                   {"press_name": "none", "dataset_sha": "aaaaaaaaaaaa"})
        _write_run(tmp_path / "b", ["#### 5"], [["5"]],
                   {"press_name": "snapkv", "compression_ratio": 0.5,
                    "dataset_sha": "bbbbbbbbbbbb"})
        table = build_comparison_table([tmp_path / "a", tmp_path / "b"])
        assert "FATAL" in table and "DIFFERENT datasets" in table

    def test_compress_questions_mismatch_is_fatal(self, tmp_path):
        """Two runs where the ratio acts on different text are not comparable."""
        _write_run(tmp_path / "a", ["#### 5"], [["5"]],
                   {"press_name": "snapkv", "compression_ratio": 0.5,
                    "dataset_sha": "s", "compress_questions": True})
        _write_run(tmp_path / "b", ["#### 5"], [["5"]],
                   {"press_name": "snapkv", "compression_ratio": 0.5,
                    "dataset_sha": "s", "compress_questions": False})
        table = build_comparison_table([tmp_path / "a", tmp_path / "b"])
        assert "FATAL" in table and "compress_questions" in table

    def test_matching_sha_is_accepted(self, tmp_path):
        for name, cr in (("full_cache", None), ("snapkv_cr0.5", 0.5)):
            _write_run(tmp_path / name, ["#### 5"], [["5"]],
                       {"press_name": name, "compression_ratio": cr,
                        "dataset_sha": "same00000000", "compress_questions": True})
        table = build_comparison_table([tmp_path / "full_cache", tmp_path / "snapkv_cr0.5"])
        assert "FATAL" not in table and "identical across runs" in table

    def test_degenerate_run_is_flagged(self, tmp_path):
        _write_run(tmp_path / "snapkv_cr0.8", ["#### 5"], [["5"]],
                   {"press_name": "snapkv", "compression_ratio": 0.8,
                    "dataset_sha": "s", "pct_examples_degenerate": 100.0})
        table = build_comparison_table([tmp_path / "snapkv_cr0.8"])
        assert "collapsed into the observation window" in table

    def test_skipped_press_is_flagged(self, tmp_path):
        _write_run(tmp_path / "snapkv_cr0.5", ["#### 5"], [["5"]],
                   {"press_name": "snapkv", "compression_ratio": 0.5,
                    "dataset_sha": "s", "pct_examples_press_skipped": 4.0})
        table = build_comparison_table([tmp_path / "snapkv_cr0.5"])
        assert "SKIPPED" in table

    def test_low_marker_rate_is_flagged(self, tmp_path):
        _write_run(tmp_path / "bad", ["the total comes to 5"] * 4, [["5"]] * 4,
                   {"press_name": "snapkv", "compression_ratio": 0.5, "dataset_sha": "s"})
        table = build_comparison_table([tmp_path / "bad"])
        assert "measuring output format" in table

    def test_mixed_batch_sizes_are_flagged(self, tmp_path):
        """Batched output is batch-size dependent, so mixed cells aren't comparable."""
        _write_run(tmp_path / "a", ["#### 5"], [["5"]],
                   {"press_name": "snapkv", "compression_ratio": 0.5,
                    "dataset_sha": "s", "batching": {"batch_size": 32}})
        _write_run(tmp_path / "b", ["#### 5"], [["5"]],
                   {"press_name": "snapkv", "compression_ratio": 0.8,
                    "dataset_sha": "s", "batching": {"batch_size": 16}})
        table = build_comparison_table([tmp_path / "a", tmp_path / "b"])
        assert "DIFFERENT batch sizes" in table
        assert "batching artifact" in table

    def test_uniform_batch_size_is_not_flagged(self, tmp_path):
        for name, cr in (("a", 0.5), ("b", 0.8)):
            _write_run(tmp_path / name, ["#### 5"], [["5"]],
                       {"press_name": "snapkv", "compression_ratio": cr,
                        "dataset_sha": "s", "batching": {"batch_size": 32}})
        table = build_comparison_table([tmp_path / "a", tmp_path / "b"])
        assert "DIFFERENT batch sizes" not in table

    def test_oom_backoff_is_flagged(self, tmp_path):
        """Recovering rows at a smaller batch is worth it, but must not be silent."""
        _write_run(tmp_path / "cr0.20", ["#### 5"], [["5"]],
                   {"press_name": "layer_defensivekv", "compression_ratio": 0.2,
                    "dataset_sha": "s", "batching": {"batch_size": 32},
                    "n_batch_size_backoffs": 3})
        table = build_comparison_table([tmp_path / "cr0.20"])
        assert "backoff occurred" in table
        assert "cr0.20 (3x)" in table

    def test_no_backoff_is_not_flagged(self, tmp_path):
        _write_run(tmp_path / "clean", ["#### 5"], [["5"]],
                   {"press_name": "snapkv", "compression_ratio": 0.5,
                    "dataset_sha": "s", "batching": {"batch_size": 32},
                    "n_batch_size_backoffs": 0})
        assert "backoff occurred" not in build_comparison_table([tmp_path / "clean"])

    def test_noise_floor_is_reported(self, tmp_path):
        _write_run(tmp_path / "a", ["#### 5"] * 50, [["5"]] * 50,
                   {"press_name": "none", "dataset_sha": "s"})
        assert "Noise floor" in build_comparison_table([tmp_path / "a"])

    def test_comparison_writes_csv(self, tmp_path):
        _write_run(tmp_path / "a", ["#### 5"], [["5"]],
                   {"press_name": "none", "dataset_sha": "s"})
        out = tmp_path / "cmp.csv"
        build_comparison_table([tmp_path / "a"], out)
        assert "compression_ratio" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# runner wiring (no weights)
# ---------------------------------------------------------------------------


class TestRunnerWiring:
    def test_press_registry_builds_all_three_families(self):
        from gsm8k.run_gsm8k import build_press, needs_var_flash_attn, press_window_size_of

        for name in ("snapkv", "adakv", "defensivekv", "layer_defensivekv"):
            press = build_press(name, 0.5)
            assert press is not None
            assert press.compression_ratio == 0.5
            assert press_window_size_of(press) > 0

        assert build_press("none", 0.0) is None
        assert needs_var_flash_attn(build_press("defensivekv", 0.5)) is True
        assert needs_var_flash_attn(build_press("snapkv", 0.5)) is False

    def test_window_size_override_reaches_the_inner_press(self):
        from gsm8k.run_gsm8k import build_press, press_window_size_of

        assert press_window_size_of(build_press("snapkv", 0.5, window_size=8)) == 8
        # AdaKVPress wraps a ScorerPress; the override has to reach the inner one.
        assert build_press("adakv", 0.5, window_size=8).press.window_size == 8

    def test_aliases_resolve(self):
        from gsm8k.run_gsm8k import build_press

        assert build_press("full_cache", 0.0) is None
        assert build_press("efficient_defensivekv", 0.5) is not None

    def test_unknown_press_names_the_alternatives(self):
        from gsm8k.run_gsm8k import build_press

        with pytest.raises(ValueError, match="snapkv"):
            build_press("nope", 0.5)

    def test_baseline_rejects_a_compression_ratio(self):
        from gsm8k.run_gsm8k import GSM8KRunner

        with pytest.raises(ValueError, match="full-cache control"):
            GSM8KRunner(model="m", press_name="none", compression_ratio=0.5)

    def test_press_rejects_an_out_of_range_ratio(self):
        from gsm8k.run_gsm8k import GSM8KRunner

        with pytest.raises(ValueError, match="fraction REMOVED"):
            GSM8KRunner(model="m", press_name="snapkv", compression_ratio=1.0)
