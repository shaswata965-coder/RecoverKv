"""LongGenBench (Wu et al., ICLR'25): the protocol, the metric and the runner, on CPU.

What a LongGenBench number here rests on, checked against the reference code
(``mozhu621/LongGenBench`` @ 45c320e: ``Evalution/inference.py``, ``eval.py``,
``results/static.ipynb``):

* the prompt is the dataset prompt verbatim, generation stops at
  ``*** finished`` (excluded) and the scored text is ``prefix + generation``;
* blocks are keyed by the first ``"<type> N"`` in them, first block per N wins;
* the judge prompt is eval.py's few-shot prompt, character for character, and
  the verdict rule is ``'yes' in response``;
* CR, STIC-1, STIC-2, wAvg and the per-category accuracies are the reference's
  definitions (micro over checks; CR a mean over documents).

The data-file checks run when the pinned files are on disk
(``python -m data.longgenbench_loader --out data/longgenbench``, or point
``LONGGENBENCH_DATA`` at a copy).
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


def _have(setting):
    return (DATA / L.FILES[setting][0]).is_file()


# ---------------------------------------------------------------------------
# data and protocol
# ---------------------------------------------------------------------------


def test_settings_match_the_reference_scripts():
    assert L.MAX_NEW_TOKENS == {"16K": 16000, "32K": 32000}
    assert L.FILES["16K"][0] == "Dataset_short.json" and L.FILES["32K"][0] == "Dataset_long.json"
    assert L.STOP_STRING == "*** finished" and L.BLOCK_SEP == "#*#"
    assert L.TYPES["16K"] == {"Week": 52, "Floor": 100, "Menu Week": 52, "Block": 100}
    assert L.REPO_SAMPLING["temperature"] == 0.95 and L.REPO_SAMPLING["top_p"] == 0.95


def test_a_file_with_the_wrong_bytes_is_refused(tmp_path):
    (tmp_path / "Dataset_short.json").write_text("[]")
    with pytest.raises(ValueError, match="not the pinned"):
        L.load_all(tmp_path, "16K")


@pytest.mark.parametrize("setting", L.SETTINGS)
def test_the_pinned_file_is_the_benchmark(setting):
    if not _have(setting):
        pytest.skip("data not fetched (python -m data.longgenbench_loader)")
    rows = L.load_all(DATA, setting)
    assert len(rows) == 400
    assert {t: sum(r["type"] == t for r in rows) for t in L.TYPES[setting]} == \
        {t: 100 for t in L.TYPES[setting]}
    for r in rows:
        assert r["number"] == L.TYPES[setting][r["type"]]
        # The prompt ends with the first entry's header: the model continues it.
        assert r["prompt"].rstrip().endswith(r["prefix"])
        assert "<|start_header_id|>" not in r["prompt"]
    # static.csv "Total Checks": 6205 at 16K, 8215 at 32K.
    total = sum(len(r["checks_once"]) + len(r["checks_range"]) + len(r["checks_periodic"])
                for r in rows)
    assert total == {"16K": 6205, "32K": 8215}[setting]


def test_load_type_slices_within_a_type():
    if not _have("16K"):
        pytest.skip("data not fetched")
    week = L.load_type(DATA, "16K", "Week", 0, 3)
    floor = L.load_type(DATA, "16K", "Floor", 0, 3)
    assert [d["type"] for d in week + floor] == ["Week"] * 3 + ["Floor"] * 3
    assert [d["id"] for d in L.load_type(DATA, "16K", "Floor", 1, 2)] == [floor[1]["id"]]
    with pytest.raises(ValueError):
        L.load_type(DATA, "16K", "Day")          # a 32K type


# ---------------------------------------------------------------------------
# the metric, against eval.py / static.ipynb
# ---------------------------------------------------------------------------

DOC = {
    "id": 7, "type": "Floor", "number": 4,
    "checks_once": {"2": "music school", "9": "rooftop bar"},   # Floor 9 is never written
    "checks_range": {"3": "gym", "4": "gym"},
    "checks_periodic": {"2": "restaurant"},
    "output": ("#*# Floor 1:\nLobby.\n#*# Floor 2:\nA music school and a restaurant.\n"
               "#*# Floor 3:\nA gym. See Floor 2 for food.\n#*# Floor 3:\nA second Floor 3.\n"),
}


def test_blocks_are_keyed_by_the_first_type_number_and_the_first_block_wins():
    blocks = S.parse_blocks(S.split_blocks(DOC["output"]), "Floor")
    assert sorted(blocks) == [1, 2, 3]
    assert "A gym." in blocks[3]                   # not the second "Floor 3" block
    assert S.completion_rate(blocks, 4) == 75.0    # Floor 4 missing


def test_the_type_pattern_is_case_and_space_exact():
    blocks = S.parse_blocks(["#*# Menu Week 3 (Jan):", "#*# Week 5:", "#*# week 6:"], "Menu Week")
    assert sorted(blocks) == [3]
    assert sorted(S.parse_blocks(["#*# Menu Week 3:", "#*# Week 5:"], "Week")) == [3, 5]


def test_only_checks_whose_entry_exists_are_judged():
    checks = S.document_checks(DOC)
    assert [(c["category"], c["identifier"], c["block"] is not None) for c in checks] == [
        ("once", 2, True), ("once", 9, False), ("range", 3, True), ("range", 4, False),
        ("periodic", 2, True)]


def test_judge_prompt_is_eval_py_character_for_character():
    p = S.judge_prompt("CTX", "gym")
    assert p.startswith(S.JUDGE_EXAMPLES[0] + "\n" + S.JUDGE_EXAMPLES[1] + "\n")
    assert p.endswith(" \n ### Refer to the examples above for how to answer. \nContext: CTX"
                      "\n\n### Instruction: Now, for the following context, does it include the "
                      "gym? Please answer with 'yes' or 'no' only. Answer: ")
    assert len(S.JUDGE_EXAMPLES) == 3 and S.JUDGE_EXAMPLES[2].endswith("Answer: no")


def test_the_verdict_rule_and_its_quirk():
    assert S.verdict(" Yes.") == "yes" and S.verdict("no") == "no"
    # A judge that says no and then runs on into an invented example is a yes
    # under the reference rule; the strict diagnostic sees the first word.
    run_on = "no\nExample 4: Context: ... Answer: yes"
    assert S.verdict(run_on) == "yes" and S.strict_verdict(run_on) == "no"
    assert S.strict_verdict("I think so") is None
    assert S.strict_verdict(" **Yes**.") == "yes"


def test_aggregates_are_the_reference_definitions():
    judg = {7: [{"category": "once", "identifier": 2, "verdict": "yes", "strict": "yes"},
                {"category": "range", "identifier": 3, "verdict": "no", "strict": "no"},
                {"category": "periodic", "identifier": 2, "verdict": "yes", "strict": "no"}]}
    s = S.summarize([DOC], judg)
    assert s["n_checks"] == 5 and s["n_judged"] == 3 and s["n_yes"] == 2
    assert s["CR"] == 75.0
    assert s["STIC-1"] == round(100 * 2 / 3, 2)        # yes / judged
    assert s["STIC-2"] == 40.0                         # yes / all checks
    assert s["wAvg"] == 30.0                           # CR * STIC-2
    assert s["STIC-2_strict"] == 20.0
    assert (s["acc_once"], s["acc_range"], s["acc_periodic"]) == (100.0, 0.0, 100.0)
    assert s["Len"] is None                            # <= 40 blocks: not in the length mean


def test_a_failed_document_can_only_lower_the_score():
    s = S.summarize([dict(DOC, output=None)], {})
    assert s["CR"] == 0.0 and s["STIC-2"] == 0.0 and s["n_judged"] == 0


def test_bootstrap_is_paired_and_keeps_ratio_of_sums():
    a = [dict(DOC, id=i) for i in range(4)]
    b = [dict(DOC, id=i) for i in range(4)]
    ja = {i: [{"category": "once", "identifier": 2, "verdict": "no", "strict": "no"}] for i in range(4)}
    jb = {i: [{"category": "once", "identifier": 2, "verdict": "yes", "strict": "yes"}] for i in range(4)}
    d = S.bootstrap_delta(a, ja, b, jb, iters=200)
    assert d["STIC-2"] == {"delta": 20.0, "lo": 20.0, "hi": 20.0}
    assert d["CR"]["delta"] == 0.0


# ---------------------------------------------------------------------------
# config and wiring
# ---------------------------------------------------------------------------


def test_config_validates():
    from utils.config import ConfigValidationError, LongGenBenchConfig

    assert LongGenBenchConfig(types="Week").types == ["Week"]
    for bad in (dict(setting="8K"), dict(decoding="beam"), dict(start=-1),
                dict(start=4, end=4), dict(max_new_tokens=0), dict(types=[])):
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
    assert ours.longgenbench.setting == full.longgenbench.setting == "16K"
    assert ours.longgenbench.decoding == full.longgenbench.decoding == "greedy"


# ---------------------------------------------------------------------------
# the runner, end to end on a stub model (full-cache path, CPU)
# ---------------------------------------------------------------------------


def _stub_run(tmp_path, monkeypatch, decoding, text):
    torch = pytest.importorskip("torch")
    from modules.evaluation import longgenbench_runner as R
    from utils.config import ExperimentConfig, LongGenBenchConfig, ModelConfig

    docs = [dict(DOC, id=100 + i, prefix="#*# Floor 1:", prompt=f"Build. *** started ***\n#*# Floor 1:{i}")
            for i in range(3)]
    for d in docs:
        d.pop("output")
    monkeypatch.setattr(R, "load_type", lambda dd, s, t, start, end: docs[start:end])
    monkeypatch.setattr(R, "dataset_sha", lambda dd, s: "f" * 64)

    class Enc(dict):
        def __init__(self, t):
            super().__init__(input_ids=t, attention_mask=torch.ones_like(t))
            self.input_ids, self.attention_mask = t, torch.ones_like(t)

    class Tok:
        bos_token, bos_token_id, eos_token_id, pad_token_id = "<s>", 1, 9, None
        eos_token = "</s>"

        def __call__(self, prompt, add_special_tokens=True, **kw):
            return Enc(torch.tensor([([1] if add_special_tokens else []) + [5, 6, 7]]))

        def decode(self, ids, skip_special_tokens=True):
            return text

    seen = {}

    class Model:
        device = torch.device("cpu")
        config = types.SimpleNamespace(max_position_embeddings=131072)
        generation_config = types.SimpleNamespace(eos_token_id=[2, 3, 9])

        def generate(self, input_ids, **kw):
            seen.update(kw)
            return torch.cat([input_ids, torch.tensor([[8, 8, 9]])], dim=-1)

    cfg = ExperimentConfig()
    cfg.model = ModelConfig(name="meta-llama/Llama-3.1-8B-Instruct", dtype="float16",
                            attn_implementation="flash_attention_2")
    cfg.longgenbench = LongGenBenchConfig(data_dir=str(tmp_path), setting="16K", types=["Floor"],
                                          output_dir=str(tmp_path / "out"), start=1, end=3,
                                          decoding=decoding)
    runner = R.LongGenBenchRunner(cfg)
    monkeypatch.setattr(runner, "_load_model_and_tokenizer", lambda: (Model(), Tok()))
    runner.run()
    return seen, S.read_jsonl(tmp_path / "out" / "floor.jsonl"), \
        json.loads((tmp_path / "out" / "floor.meta.json").read_text())


def test_runner_reproduces_the_generation_protocol(tmp_path, monkeypatch):
    text = "\nLobby.\n#*# Floor 2:\nStudio.\n*** finished ***\n#*# Floor 3: never"
    seen, recs, meta = _stub_run(tmp_path, monkeypatch, "greedy", text)
    assert seen["max_new_tokens"] == 16000 and seen["do_sample"] is False
    assert seen["stop_strings"] == ["*** finished"] and seen["eos_token_id"] == [2, 3, 9]
    assert [r["id"] for r in recs] == [101, 102]
    r = recs[0]
    # prefix + generation, cut BEFORE the stop string, as vLLM does
    assert r["output"] == "#*# Floor 1:\nLobby.\n#*# Floor 2:\nStudio.\n"
    assert r["stopped_on"] == "stop_string" and r["prompt_tokens"] == 4   # one BOS
    assert r["completion_rate"] == 50.0                                  # Floors 1-2 of 4
    assert meta["ids"] == [101, 102] and meta["setting"] == "16K" and meta["type"] == "Floor"
    assert meta["cache_type"] == "full_cache" and meta["read_gate"]["verdict"] == "not-expected"
    assert meta["dataset_revision"] == L.HF_REVISION and meta["prompt_format"] == "raw_completion"
    assert meta["generation"]["decoding"] == "greedy" and meta["generation"]["sampling"] is None


def test_runner_repo_sampling_is_seeded_per_document(tmp_path, monkeypatch):
    seen, recs, meta = _stub_run(tmp_path, monkeypatch, "repo_sampling", "\nx")
    assert seen["do_sample"] is True and seen["temperature"] == 0.95 and seen["top_p"] == 0.95
    assert seen["top_k"] == 0
    assert meta["generation"]["sampling"]["seed"] == 6211027
    assert recs[0]["stopped_on"] == "eos"


def test_judge_asks_only_existing_entries_with_the_reference_prompt(tmp_path, monkeypatch):
    from modules.evaluation import longgenbench_judge as J

    text = "\nLobby.\n#*# Floor 2:\nA music school and a restaurant.\n#*# Floor 3:\nA gym.\n"
    _stub_run(tmp_path, monkeypatch, "greedy", text)
    asked = []

    class StubJudge:
        def ask(self, prompt, seed=0):
            asked.append(prompt)
            return " no\nExample 4: ... Answer: yes" if "gym?" in prompt else " Yes"

        def record(self):
            return {"judge_model": "stub", "judge_decoding": "greedy"}

    done = J.judge_run_dir(StubJudge(), tmp_path / "out")
    assert done == [("floor", {"n_judged": 6, "n_yes": 6})]
    # Floors 2 and 3 exist in both documents: once/2, range/3, periodic/2 each.
    assert len(asked) == 6 and all(p.startswith(S.JUDGE_EXAMPLES[0]) for p in asked)
    # split('#*#') drops the separator, so the judged block starts " Floor 2:"
    assert asked[0] == S.judge_prompt(" Floor 2:\nA music school and a restaurant.\n", "music school")
    rows = S.read_jsonl(tmp_path / "out" / "floor.judge.jsonl")
    gym = [r for r in rows if r["desc"] == "gym"]
    assert gym and all(r["verdict"] == "yes" and r["strict"] == "no" for r in gym)
    jm = json.loads((tmp_path / "out" / "floor.judge.meta.json").read_text())
    assert jm["ids"] == [101, 102] and jm["n_strict_disagree"] == 2
    recs = S.read_jsonl(tmp_path / "out" / "floor.jsonl")
    s = S.summarize(recs, S.judgments_by_doc(rows))
    assert (s["n_checks"], s["n_judged"], s["n_yes"]) == (10, 6, 6) and s["CR"] == 75.0
