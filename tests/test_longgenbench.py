"""LongGenBench: the prompt, the metric and the runner's record, on CPU.

The two claims a LongGenBench number rests on are checked here:

* the metric IS SCOPE's ``eval_longgenbench.py`` (regexes, positional pairing,
  rounding), so a score is comparable with its table; and
* the prompt IS the string ``run_longgenbench.py`` tokenizes.

Both are also checked against SCOPE's own files when they are on disk
(``python -m data.longgenbench_loader --out data/longgenbench``, or point
``LONGGENBENCH_DATA`` at a copy): its full-cache predictions must re-score to its
published numbers, and every prompt it recorded must be rebuilt byte for byte.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

from data import longgenbench_loader as L
from modules.evaluation import longgenbench_scoring as S

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("LONGGENBENCH_DATA", REPO / "data" / "longgenbench"))


def _have(setting, task):
    return ((DATA / L.data_relpath(setting, task)).is_file()
            and (DATA / L.reference_relpath(setting, task)).is_file())


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------


def test_prompt_is_build_chat_llama3_modify():
    raw = "Examples: \nQuestion_1: \nx\n"
    p = L.build_prompt("gsm8k", raw)
    assert p == f"<<SYS>>\n {L.SYSTEM_PROMPTS['gsm8k']} \n<</SYS>>\n\n{raw}"
    # No chat-template header and no BOS in the string: the tokenizer adds the one BOS.
    assert "<|start_header_id|>" not in p and not p.startswith("<|begin_of_text|>")


def test_the_mmlu_system_prompt_keeps_the_reference_quirk():
    """run_longgenbench.py formats the system prompt BEFORE appending the subject."""
    assert L.SYSTEM_PROMPTS["mmlu"].endswith("(with answers) about ")


def test_only_the_llama3_branch_is_accepted():
    L.assert_supported_model("meta-llama/Llama-3.1-8B-Instruct")
    L.assert_supported_model("/content/models/Llama-3.1-8B-Instruct")
    with pytest.raises(ValueError):
        L.assert_supported_model("mistralai/Mistral-7B-Instruct-v0.2")


def test_settings_match_the_reference_scripts():
    assert L.MAX_NEW_TOKENS == {"4K": 4096, "8K": 7950}
    assert L.K_PER_TASK["8K"] == {"gsm8k": 60, "mmlu": 60, "csqa": 80}
    assert L.K_PER_TASK["4K"] == {"gsm8k": 30, "mmlu": 30, "csqa": 40}
    assert L.data_relpath("8K", "csqa") == "data/LongGenBench/csqa/csqa_80.jsonl"
    assert set(L.DATA_FILES) == {L.data_relpath(s, t) for s in L.SETTINGS for t in L.TASKS}


def test_a_file_with_the_wrong_bytes_is_refused(tmp_path):
    rel = L.data_relpath("8K", "gsm8k")
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text('{"prompt": "x", "questions": [], "answers": []}\n')
    with pytest.raises(ValueError, match="not the pinned"):
        L.load_longgenbench(tmp_path, "8K", "gsm8k")


def test_jsonl_reader_does_not_split_on_unicode_line_separators(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_bytes(json.dumps({"a": "one two\x85three"}, ensure_ascii=False)
                  .encode("utf-8") + b"\n")
    assert L._read_jsonl(p) == [{"a": "one two\x85three"}]


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------

GOLD = ["... #### 6", "... #### 5", "... #### 39"]


def test_gsm8k_reference_metric_is_positional():
    pred = ("Answer_9:\nso 6.\nThe answer is 6.\n\n"
            "Answer_10:\n3 + 2 = 5.\nThe answer is 5.\n\n"
            "Answer_11:\nThe answer is 39.\n")
    assert S.scope_example_accuracy("gsm8k", pred, GOLD) == 1.0
    # Skip one answer: everything after it shifts onto the wrong gold.
    skipped = "Answer_9:\nThe answer is 6.\n\nAnswer_11:\nThe answer is 39.\n"
    assert S.scope_example_accuracy("gsm8k", skipped, GOLD) == pytest.approx(1 / 3)
    al = S.aligned_example("gsm8k", skipped, GOLD)
    assert al["acc_aligned"] == pytest.approx(2 / 3) and al["coverage"] == pytest.approx(2 / 3)


def test_gsm8k_dollar_sign_lets_the_reference_steal_the_next_answer():
    """``answer is $6`` does not match ``(\\d+)``; the lazy DOTALL match runs on
    into the next block. Reproduced, because the published numbers include it."""
    pred = "Answer_9:\nThe answer is $6.\n\nAnswer_10:\nThe answer is 5.\n\nAnswer_11:\nThe answer is 39."
    assert S.scope_predictions("gsm8k", pred) == ["5", "39"]
    assert S.scope_example_accuracy("gsm8k", pred, GOLD) == 0.0
    # The aligned diagnostic applies the same pattern inside each block only.
    assert S.aligned_example("gsm8k", pred, GOLD)["acc_aligned"] == pytest.approx(2 / 3)


def test_choice_metric_needs_the_answer_on_the_first_line():
    """No DOTALL: ``.*?`` cannot cross a newline, so a block whose 'answer is'
    sits on a later line yields no match and later answers shift."""
    gold = ["A", "B"]
    one_line = "Answer_6:\nThe answer is (A).\n\nAnswer_7:\nThe answer is (B)."
    assert S.scope_example_accuracy("mmlu", one_line, gold) == 1.0
    two_line = "Answer_6:\nBecause x.\nThe answer is (A).\n\nAnswer_7:\nThe answer is (B)."
    assert S.scope_predictions("mmlu", two_line) == ["B"]
    assert S.scope_example_accuracy("mmlu", two_line, gold) == 0.0


def test_task_score_rounds_like_the_reference():
    preds = ["Answer_9:\nThe answer is 6.", ""]
    assert S.scope_task_score("gsm8k", preds, [GOLD, GOLD]) == round(100 * (1 / 3) / 2, 4)


def test_a_failed_generation_scores_zero_and_is_counted():
    recs = [{"pred": None, "answers": GOLD},
            {"pred": "Answer_9:\nThe answer is 6.", "answers": GOLD}]
    s = S.summarize("gsm8k", recs)
    assert s["n_failed"] == 1 and s["acc"] == round(100 * (1 / 3) / 2, 4)


def test_first_block_per_number_wins():
    pred = "Answer_9:\nThe answer is 6.\n\nAnswer_9:\nThe answer is 7."
    assert S.answer_blocks(pred)[9].strip() == "The answer is 6."


# ---------------------------------------------------------------------------
# against SCOPE's own files (skipped when they are not on disk)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("setting", L.SETTINGS)
@pytest.mark.parametrize("task", L.TASKS)
def test_rescoring_scope_full_cache_reproduces_its_published_number(setting, task):
    if not _have(setting, task):
        pytest.skip("SCOPE files not fetched (python -m data.longgenbench_loader)")
    assert S.rescore_reference(DATA, setting, task) == L.published_reference_score(setting, task)


@pytest.mark.parametrize("setting", L.SETTINGS)
@pytest.mark.parametrize("task", L.TASKS)
def test_every_scope_prompt_is_rebuilt_byte_for_byte(setting, task):
    if not _have(setting, task):
        pytest.skip("SCOPE files not fetched (python -m data.longgenbench_loader)")
    ids = L.example_ids(DATA, setting, task)
    ref = L.reference_by_id(DATA, setting, task, ids)   # raises on any prompt mismatch
    assert sorted(ref) == ids, "SCOPE ran a different example set than the one built here"


# ---------------------------------------------------------------------------
# config and wiring
# ---------------------------------------------------------------------------


def test_config_validates():
    from utils.config import ConfigValidationError, LongGenBenchConfig

    assert LongGenBenchConfig(tasks="csqa").tasks == ["csqa"]
    for bad in (dict(setting="16K"), dict(tasks=["qasper"]), dict(start=-1),
                dict(start=4, end=4), dict(max_new_tokens=0)):
        with pytest.raises(ConfigValidationError):
            LongGenBenchConfig(**bad)


def test_the_mode_is_registered():
    src = (REPO / "main.py").read_text()
    assert '"longgenbench": "modules.evaluation.longgenbench_runner.LongGenBenchRunner"' in src


def test_both_configs_load():
    from utils.config import load_config

    ours = load_config(REPO / "configs" / "longgenbench_ours_flash.yaml")
    full = load_config(REPO / "configs" / "longgenbench_full_cache.yaml")
    assert ours.run.mode == full.run.mode == "longgenbench"
    assert ours.cache.backend == "windowed" and full.cache.backend == "dynamic"
    assert ours.longgenbench.setting == full.longgenbench.setting == "8K"


# ---------------------------------------------------------------------------
# the runner, end to end on a stub model (full-cache path, CPU)
# ---------------------------------------------------------------------------


def test_runner_writes_records_and_an_attributable_sidecar(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from modules.evaluation import longgenbench_runner as R
    from utils.config import ExperimentConfig, LongGenBenchConfig, ModelConfig

    exs = [{"id": i, "task": "gsm8k", "subject": None, "prompt": f"<<SYS>>\n p{i}",
            "answers": GOLD, "n_questions": 3} for i in range(3)]
    monkeypatch.setattr(R, "load_longgenbench",
                        lambda d, s, t, drop_scope_oom, start, end: exs[start:end])
    monkeypatch.setattr(R, "dataset_sha", lambda d, s, t: "f" * 64)

    answer = "Answer_9:\nThe answer is 6.\n\nAnswer_10:\nThe answer is 5."

    class Tok:
        bos_token, bos_token_id, eos_token_id, pad_token_id = "<s>", 1, 2, None
        eos_token = "</s>"

        def __call__(self, prompt, add_special_tokens=True, **kw):
            ids = ([1] if add_special_tokens else []) + [5, 6, 7]
            return _Enc(torch.tensor([ids]))

        def decode(self, ids, skip_special_tokens=True):
            return answer

    class _Enc(dict):
        def __init__(self, t):
            super().__init__(input_ids=t, attention_mask=torch.ones_like(t))
            self.input_ids, self.attention_mask = t, torch.ones_like(t)

    seen = {}

    class Model:
        device = torch.device("cpu")
        config = types.SimpleNamespace(max_position_embeddings=131072)

        def generate(self, input_ids, **kw):
            seen.update(kw)
            return torch.cat([input_ids, torch.tensor([[9, 9, 2]])], dim=-1)

    cfg = ExperimentConfig()
    cfg.model = ModelConfig(name="meta-llama/Llama-3.1-8B-Instruct", dtype="float16",
                            attn_implementation="flash_attention_2")
    cfg.longgenbench = LongGenBenchConfig(data_dir=str(tmp_path), setting="8K",
                                          tasks=["gsm8k"], output_dir=str(tmp_path / "out"),
                                          start=1, end=3)
    runner = R.LongGenBenchRunner(cfg)
    monkeypatch.setattr(runner, "_load_model_and_tokenizer", lambda: (Model(), Tok()))
    runner.run()

    # The reference's generation kwargs, exactly.
    assert seen["max_new_tokens"] == 7950 and seen["num_beams"] == 1
    assert seen["do_sample"] is False and seen["temperature"] == 1.0
    assert seen["eos_token_id"] == [2] and seen["min_length"] == 4 + 1

    recs = S.read_records(tmp_path / "out" / "gsm8k.jsonl")
    assert [r["id"] for r in recs] == [1, 2]
    assert recs[0]["gen_tokens"] == 3 and recs[0]["stopped_on_eos"] is True
    assert recs[0]["prompt_tokens"] == 4                     # one BOS added, not two

    meta = json.loads((tmp_path / "out" / "gsm8k.meta.json").read_text())
    assert meta["ids"] == [1, 2] and meta["setting"] == "8K"
    assert meta["cache_type"] == "full_cache"
    assert meta["read_gate"]["verdict"] == "not-expected"
    assert meta["reference_commit"] == L.SCOPE_COMMIT and meta["prompt_format"] == L.PROMPT_FORMAT
    assert S.summarize("gsm8k", recs)["acc"] == round(100 * 2 / 3, 4)
