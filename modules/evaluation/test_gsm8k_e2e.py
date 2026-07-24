"""Tests for the GSM8K end-to-end harness: loader, config, runner wiring, scoring.

No model and no network: the HF download is stubbed, and the runner is exercised
through the parts that do not need weights (config wiring, the compression
diagnostic, meta assembly). What these tests protect is the *validity
machinery* — the dataset_sha gate, the no-eviction detector, the baseline
gate's inputs — because those are what stop an invalid run from being reported
as a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data import gsm8k_loader
from modules.evaluation.gsm8k_scoring import (
    build_comparison_table,
    score_run_dir,
)
from utils.config import ConfigValidationError, GSM8KConfig, load_config


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

_RAW = [
    {"question": "Janet has 3 eggs and buys 2 more. How many?", "answer": "3+2=5\n#### 5"},
    {"question": "Tom runs 4 miles twice. How far?", "answer": "4*2=8\n#### 8"},
    {"question": "A pie costs $6. Two pies?", "answer": "6*2=12\n#### 12"},
    {"question": "Sam had 10, gave 4 away. Left?", "answer": "10-4=6\n#### 6"},
    {"question": "5 boxes of 7. Total?", "answer": "5*7=35\n#### 35"},
]


@pytest.fixture()
def built_dataset(tmp_path, monkeypatch):
    """Build a GSM8K CoT dataset from stubbed HF rows."""
    monkeypatch.setattr(gsm8k_loader, "load_dataset", None, raising=False)

    def fake_load_dataset(name, config, split):  # noqa: ARG001
        return list(_RAW)

    import datasets as _ds

    monkeypatch.setattr(_ds, "load_dataset", fake_load_dataset)
    out = tmp_path / "gsm8k_cot"
    gsm8k_loader.build_gsm8k_cot(out)
    return out


def _write_run(run_dir: Path, preds, answers, meta: dict) -> Path:
    """Fabricate a GSM8KRunner output directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, (p, a) in enumerate(zip(preds, answers)):
            f.write(json.dumps({"id": i, "pred": p, "answer": a, "task": "gsm8k"}) + "\n")
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_build_writes_all_three_artifacts(self, built_dataset):
        assert (built_dataset / gsm8k_loader.JSONL_NAME).exists()
        assert (built_dataset / gsm8k_loader.MANIFEST_NAME).exists()
        assert (built_dataset / "hf").exists()

    def test_manifest_records_cot_prompt_and_512_tokens(self, built_dataset):
        m = gsm8k_loader.read_manifest(built_dataset)
        assert m["prompt_style"] == "chain_of_thought"
        assert m["max_new_tokens"] == 512
        # The defect that produced 12.66%: a prompt that forbids reasoning.
        assert "step-by-step" in m["system_prompt"]
        assert "do not show any reasoning" not in m["system_prompt"].lower()

    def test_answers_are_bare_numbers(self, built_dataset):
        rows = gsm8k_loader.load_gsm8k_dataset(built_dataset)
        assert [r["answer"] for r in rows] == [["5"], ["8"], ["12"], ["6"], ["35"]]

    def test_sha_is_stable_across_rebuilds(self, tmp_path, monkeypatch):
        import datasets as _ds

        monkeypatch.setattr(_ds, "load_dataset", lambda n, c, split: list(_RAW))
        a = gsm8k_loader.build_gsm8k_cot(tmp_path / "a")
        b = gsm8k_loader.build_gsm8k_cot(tmp_path / "b")
        assert (
            gsm8k_loader.read_manifest(a)["dataset_sha"]
            == gsm8k_loader.read_manifest(b)["dataset_sha"]
        )

    def test_sha_changes_when_content_changes(self, tmp_path, monkeypatch):
        import datasets as _ds

        monkeypatch.setattr(_ds, "load_dataset", lambda n, c, split: list(_RAW))
        full = gsm8k_loader.build_gsm8k_cot(tmp_path / "full")
        part = gsm8k_loader.build_gsm8k_cot(tmp_path / "part", limit=3)
        assert (
            gsm8k_loader.read_manifest(full)["dataset_sha"]
            != gsm8k_loader.read_manifest(part)["dataset_sha"]
        )

    def test_order_is_never_shuffled(self, built_dataset):
        first = gsm8k_loader.load_gsm8k_dataset(built_dataset)
        second = gsm8k_loader.load_gsm8k_dataset(built_dataset)
        assert [r["question"] for r in first] == [r["question"] for r in second]

    def test_num_samples_cap_applies_before_sharding(self, built_dataset):
        """A capped sweep must be the SAME problems at every budget."""
        s0 = gsm8k_loader.load_gsm8k_dataset(built_dataset, num_samples=4, shard=0, num_shards=2)
        s1 = gsm8k_loader.load_gsm8k_dataset(built_dataset, num_samples=4, shard=1, num_shards=2)
        assert len(s0) + len(s1) == 4

    def test_shards_partition_without_overlap(self, built_dataset):
        whole = gsm8k_loader.load_gsm8k_dataset(built_dataset)
        parts = [
            gsm8k_loader.load_gsm8k_dataset(built_dataset, shard=i, num_shards=3)
            for i in range(3)
        ]
        flat = [r for p in parts for r in p]
        assert len(flat) == len(whole)
        assert [r["question"] for r in flat] == [r["question"] for r in whole]

    def test_missing_manifest_names_the_build_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="data.gsm8k_loader"):
            gsm8k_loader.read_manifest(tmp_path / "nope")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfig:
    @pytest.mark.parametrize(
        "name", ["gsm8k_full_cache", "gsm8k_budget80", "gsm8k_budget20"]
    )
    def test_shipped_configs_load(self, name):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg.run.mode == "gsm8k"
        assert cfg.gsm8k.data_dir == "data/gsm8k_cot"

    def test_budget_configs_carry_the_intended_budgets(self):
        assert load_config("configs/gsm8k_full_cache.yaml").cache.cache_budget is None
        assert load_config("configs/gsm8k_budget80.yaml").cache.cache_budget == 0.80
        assert load_config("configs/gsm8k_budget20.yaml").cache.cache_budget == 0.20

    def test_backend_and_attn_impl_agree(self):
        for name in ("gsm8k_budget80", "gsm8k_budget20"):
            cfg = load_config(f"configs/{name}.yaml")
            assert cfg.cache.backend_package == "flash_attn"
            assert cfg.model.attn_implementation == "flash_attention_2"

    def test_shard_out_of_range_rejected(self):
        with pytest.raises(ConfigValidationError, match="shard"):
            GSM8KConfig(shard=2, num_shards=2)

    def test_num_shards_must_be_positive(self):
        with pytest.raises(ConfigValidationError, match="num_shards"):
            GSM8KConfig(num_shards=0)

    def test_registry_resolves_both_modes(self):
        from main import _RUNNER_REGISTRY, _import_runner

        assert "gsm8k" in _RUNNER_REGISTRY and "gsm8k_score" in _RUNNER_REGISTRY
        assert _import_runner("gsm8k").__name__ == "GSM8KRunner"
        assert _import_runner("gsm8k_score").__name__ == "GSM8KScorer"


# ---------------------------------------------------------------------------
# runner (weight-free surface)
# ---------------------------------------------------------------------------


class TestRunnerDiagnostic:
    def _runner(self):
        from modules.evaluation.gsm8k_runner import GSM8KRunner

        return GSM8KRunner(load_config("configs/gsm8k_full_cache.yaml"))

    def test_full_cache_config_is_not_windowed(self):
        assert self._runner().is_windowed is False

    def test_windowed_config_selects_the_backend(self):
        from modules.evaluation.gsm8k_runner import GSM8KRunner

        r = GSM8KRunner(load_config("configs/gsm8k_budget20.yaml"))
        assert r.is_windowed is True
        assert r.cache_backend_package == "flash_attn"

    def test_no_eviction_is_detected(self):
        """The budget-never-binds case must be visible, not silent."""
        r = self._runner()
        r.is_windowed = True
        r._sequence_tokens = [280] * 10
        r._retained_tokens = [280] * 10   # budget exceeded the sequence
        r._n_no_eviction = 10
        s = r._compression_summary(10)
        assert s["compression_active"] is False
        assert s["pct_examples_no_eviction"] == 100.0
        assert s["effective_retention"] == 1.0

    def test_real_eviction_is_reported_as_active(self):
        r = self._runner()
        r.is_windowed = True
        r._sequence_tokens = [280] * 10
        r._retained_tokens = [128] * 10
        r._n_no_eviction = 0
        s = r._compression_summary(10)
        assert s["compression_active"] is True
        assert s["pct_examples_no_eviction"] == 0.0
        assert s["effective_retention"] == pytest.approx(128 / 280, abs=1e-3)

    def test_track_scores_is_refused(self):
        from modules.evaluation.gsm8k_runner import GSM8KRunner

        cfg = load_config("configs/gsm8k_full_cache.yaml")
        cfg.telemetry.track_scores = True
        with pytest.raises(ValueError, match="track_scores"):
            GSM8KRunner(cfg)


# ---------------------------------------------------------------------------
# scoring + comparison
# ---------------------------------------------------------------------------


class TestRunScoring:
    def test_score_run_dir_reads_jsonl(self, tmp_path):
        run = _write_run(
            tmp_path / "r",
            ["reasoning...\n#### 5", "reasoning...\n#### 9"],
            [["5"], ["8"]],
            {"cache_budget": None, "dataset_sha": "abc123"},
        )
        rep, meta = score_run_dir(run)
        assert rep.n_total == 2
        assert rep.accuracy == 50.0
        assert meta["dataset_sha"] == "abc123"

    def test_comparison_flags_dataset_sha_mismatch(self, tmp_path):
        _write_run(tmp_path / "a", ["#### 5"], [["5"]],
                   {"cache_budget": None, "dataset_sha": "aaaaaaaaaaaa"})
        _write_run(tmp_path / "b", ["#### 5"], [["5"]],
                   {"cache_budget": 0.2, "dataset_sha": "bbbbbbbbbbbb"})
        table = build_comparison_table([tmp_path / "a", tmp_path / "b"])
        assert "FATAL" in table and "DIFFERENT datasets" in table

    def test_comparison_accepts_matching_sha(self, tmp_path):
        for name, budget in (("a", None), ("b", 0.2)):
            _write_run(tmp_path / name, ["#### 5"], [["5"]],
                       {"cache_budget": budget, "dataset_sha": "same00000000"})
        table = build_comparison_table([tmp_path / "a", tmp_path / "b"])
        assert "FATAL" not in table
        assert "identical across runs" in table

    def test_comparison_flags_a_budget_that_never_bound(self, tmp_path):
        _write_run(tmp_path / "budget80", ["#### 5"], [["5"]],
                   {"cache_budget": 0.8, "dataset_sha": "s", "pct_examples_no_eviction": 100.0})
        table = build_comparison_table([tmp_path / "budget80"])
        assert "full-cache baseline in disguise" in table

    def test_comparison_flags_low_marker_rate(self, tmp_path):
        # No marker anywhere -> every extraction is the fallback rung.
        _write_run(tmp_path / "bad", ["the total comes to 5"] * 4, [["5"]] * 4,
                   {"cache_budget": 0.2, "dataset_sha": "s"})
        table = build_comparison_table([tmp_path / "bad"])
        assert "measuring output format" in table

    def test_comparison_writes_csv(self, tmp_path):
        _write_run(tmp_path / "a", ["#### 5"], [["5"]],
                   {"cache_budget": None, "dataset_sha": "s"})
        out = tmp_path / "cmp.csv"
        build_comparison_table([tmp_path / "a"], out)
        assert out.exists()
        assert "cache_budget" in out.read_text(encoding="utf-8")

    def test_noise_floor_is_reported(self, tmp_path):
        _write_run(tmp_path / "a", ["#### 5"] * 50, [["5"]] * 50,
                   {"cache_budget": None, "dataset_sha": "s"})
        assert "Noise floor" in build_comparison_table([tmp_path / "a"])
