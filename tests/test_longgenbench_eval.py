"""longgenbench_eval: the portable judge-and-score harness, checked against the paper.

What a score from the harness rests on:

* the aggregate formulas reproduce the paper's Table 2 from the authors' own
  counts (``Evalution/results/static.csv``), digit for digit;
* block parsing, completion, the judge prompt and the verdict rule are the
  reference code's (``eval.py`` @ 951769c, the version in force when Table 2's
  numbers were produced), character for character;
* the judge driver asks exactly the instructions whose entries exist, ties every
  verdict file to the generations it judged, and refuses a broken judge;
* the scorer refuses stale verdicts and compares runs on the documents they share.

The harness must import nothing from this repository, so it can be copied next to
any baseline's generation code.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from longgenbench_eval import data as D
from longgenbench_eval import judge as J
from longgenbench_eval import metrics as M
from longgenbench_eval import reference as R
from longgenbench_eval import score as SC

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("LONGGENBENCH_DATA", REPO / "data" / "longgenbench"))


def _have(setting="16K"):
    return (DATA / R.FILES[setting][0]).is_file()


# ---------------------------------------------------------------------------
# the formulas against the paper's Table 2
# ---------------------------------------------------------------------------

#: static.csv @ mozhu621/LongGenBench (setting, model, Total Checks, Total Results,
#: Total Yes, CR, Average Word Count).
STATIC_CSV = [
    ("16K", "FILM-7B", 6205, 2487, 245, 36.028846153846196, 6279.821428571428),
    ("32K", "FILM-7B", 8215, 2911, 899, 37.445409757269815, 13775.222222222223),
    ("16K", "LongWriter-llama3.1-8b", 6205, 2700, 610, 45.99307692307697, 11036.0773480663),
    ("32K", "LongWriter-llama3.1-8b", 8215, 2447, 823, 34.526729151646265, 19924.6),
    ("16K", "Llama-3.1-70B-Instruct", 6205, 5257, 1310, 79.27346153846158, 8055.321428571428),
    ("32K", "Llama-3.1-70B-Instruct", 8215, 4984, 1783, 63.09285365992478, 15196.61797752809),
    ("16K", "Llama-3.1-8B-Instruct", 6205, 5820, 1364, 93.51326923076932, 8804.327176781002),
    ("32K", "Llama-3.1-8B-Instruct", 8215, 5850, 1551, 77.59392842687107, 17353.786301369862),
    ("16K", "Mistral-7B-Instruct-v0.2", 6205, 4924, 1082, 81.81288461538462, 7295.738675958188),
    ("32K", "Mistral-7B-Instruct-v0.2", 8215, 3631, 1342, 48.217743457417875, 16145.679324894514),
    ("16K", "Mixtral-8x7B-Instruct-v0.1", 6205, 4908, 1739, 83.03365384615392, 8112.675496688742),
    ("32K", "Mixtral-8x7B-Instruct-v0.1", 8215, 4599, 1836, 60.45164646023857, 15838.68345323741),
    ("16K", "Qwen2-72B-Instruct", 6205, 5842, 1490, 94.33173076923075, 8013.498680738786),
    ("32K", "Qwen2-72B-Instruct", 8215, 5189, 1428, 66.20188953819304, 19845.15172413793),
    ("16K", "Qwen2-7B-Instruct", 6205, 3587, 1001, 59.9596153846154, 5138.333333333333),
    ("32K", "Qwen2-7B-Instruct", 8215, 3267, 1035, 39.97315052682178, 9617.09090909091),
]


@pytest.mark.parametrize("setting,model,checks,judged,yes,cr,words", STATIC_CSV)
def test_the_formulas_reproduce_the_papers_table_2(setting, model, checks, judged, yes, cr, words):
    s = M.rounded(M.from_counts(cr, checks, judged, yes, [words]))
    assert tuple(s[k] for k in R.TABLE2_COLUMNS) == R.PAPER_TABLE2[setting][model]
    assert checks == R.TOTAL_CHECKS[setting]


def test_every_table_2_row_is_covered():
    assert {(s, m) for s, m, *_ in STATIC_CSV} == {(s, m) for s in R.PAPER_TABLE2 for m in R.PAPER_TABLE2[s]}


# ---------------------------------------------------------------------------
# the reference code, character for character
# ---------------------------------------------------------------------------


def test_the_judge_prompt_is_eval_py_at_the_time_of_the_paper():
    # eval.py @ 951769c, line 46:
    # prompt = f"Context: " + type_to_block[identifier] + f"### Instruction: Does this context
    #          include the {event_desc}? Please answer with 'yes' or 'no' only."
    assert R.judge_prompt(" Floor 7:\nA dojo.\n", "martial arts dojo") == (
        "Context:  Floor 7:\nA dojo.\n### Instruction: Does this context include the martial arts dojo? "
        "Please answer with 'yes' or 'no' only.")
    assert R.JUDGE_MODEL == "meta-llama/Llama-3.1-8B-Instruct"
    assert R.JUDGE_SAMPLING == {"temperature": 0.95, "top_p": 0.95, "max_tokens": 50, "seed": 42}


def test_the_later_few_shot_prompt_is_kept_apart():
    p = R.judge_prompt_2025("CTX", "gym")
    assert p.startswith(R.JUDGE_EXAMPLES_2025[0]) and p.endswith(
        "Context: CTX\n\n### Instruction: Now, for the following context, does it include the gym? "
        "Please answer with 'yes' or 'no' only. Answer: ")
    assert J.prompt_sha("paper") != J.prompt_sha("2025")


def test_the_generation_reference_is_inference_py():
    assert R.GENERATION_SAMPLING == {"temperature": 0.95, "top_p": 0.95, "top_k": 0, "seed": 6211027}
    assert R.STOP_STRING == "*** finished" and R.BLOCK_SEP == "#*#"
    assert R.MAX_NEW_TOKENS == {"16K": 16000, "32K": 32000}


def test_the_verdict_rule():
    assert R.verdict(" Yes.") == "yes" and R.verdict("No, it does not.") == "no"
    # The reference rule counts any "yes" in the response, a run-on included.
    assert R.verdict("no\n(yes or no)") == "yes"


DOC = {"id": 7, "type": "Floor", "number": 4, "prompt": "P", "prefix": "#*# Floor 1:",
       "checks_once": {"2": "music school", "9": "rooftop bar"},
       "checks_range": {"3": "gym", "4": "gym"}, "checks_periodic": {"2": "restaurant"}}
OUT = ("#*# Floor 1:\nLobby.\n#*# Floor 2:\nA music school and a restaurant.\n"
       "#*# Floor 3:\nA gym. See Floor 2 for food.\n#*# Floor 3:\nA second Floor 3.\n")


def test_blocks_completion_and_judged_instructions():
    blocks = M.parse_blocks(M.split_blocks(OUT), "Floor")
    assert sorted(blocks) == [1, 2, 3] and "A gym." in blocks[3]      # first block wins
    assert M.completion_rate(blocks, 4) == 75.0
    assert [(c["category"], c["identifier"], c["block"] is not None) for c in M.document_checks(DOC, OUT)] == [
        ("once", 2, True), ("once", 9, False), ("range", 3, True), ("range", 4, False), ("periodic", 2, True)]
    assert sorted(M.parse_blocks(["#*# Menu Week 3:", "#*# Week 5:"], "Week")) == [3, 5]
    assert sorted(M.parse_blocks(["#*# Menu Week 3:", "#*# Week 5:"], "Menu Week")) == [3]


def test_aggregates_on_a_document():
    st = M.doc_stats(DOC, OUT, ["yes", "no", "yes"])
    s = M.aggregate([st])
    assert (s["checks"], s["judged"], s["yes"]) == (5, 3, 2)
    assert s["CR"] == 75.0 and s["STIC-2"] == 40.0 and s["wAvg"] == 30.0
    assert abs(s["STIC-1"] - 200 / 3) < 1e-9 and s["Len"] is None    # <= 40 blocks
    with pytest.raises(ValueError):
        M.doc_stats(DOC, OUT, ["yes"])                                # verdicts must match
    failed = M.aggregate([M.doc_stats(DOC, None, [])])
    assert failed["CR"] == 0.0 and failed["STIC-2"] == 0.0


def test_paired_bootstrap_keeps_ratio_of_sums():
    a = [M.doc_stats(DOC, OUT, ["no", "no", "no"])] * 4
    b = [M.doc_stats(DOC, OUT, ["yes", "no", "no"])] * 4
    d = M.paired_bootstrap(a, b, iters=200)
    assert d["STIC-2"]["delta"] == 20.0 and d["STIC-2"]["lo"] == d["STIC-2"]["hi"] == 20.0


# ---------------------------------------------------------------------------
# data, input formats, judge driver, scorer
# ---------------------------------------------------------------------------


def test_the_pinned_file_is_the_benchmark():
    if not _have():
        pytest.skip("data not fetched (python -m longgenbench_eval.data)")
    rows = D.load_all(DATA, "16K")
    assert len(rows) == 400
    assert sum(len(r["checks_once"]) + len(r["checks_range"]) + len(r["checks_periodic"]) for r in rows) == 6205
    for r in rows:
        assert r["prompt"].rstrip().endswith(r["prefix"]) and r["number"] == R.TYPES["16K"][r["type"]]
    assert D.first_n_ids(DATA, "16K", 2) == [0, 1, 100, 101, 200, 201, 300, 301]


def test_a_file_with_the_wrong_bytes_is_refused(tmp_path):
    (tmp_path / "Dataset_short.json").write_text("[]")
    with pytest.raises(ValueError, match="not the pinned"):
        D.load_all(tmp_path, "16K")


def test_both_generation_formats_load(tmp_path):
    docs = [DOC, dict(DOC, id=8, prompt="Q")]
    jl = tmp_path / "g.jsonl"
    jl.write_text(json.dumps({"id": 7, "output": OUT, "extra": 1}) + "\n" + json.dumps({"id": 8, "output": None}) + "\n")
    assert M.load_generations(jl, docs) == {7: OUT, 8: None}
    ref = tmp_path / "ref.json"      # the reference inference.py's own output
    ref.write_text(json.dumps([{"input": "P", "output_blocks": M.split_blocks(OUT), "word_count": 9}]))
    assert M.load_generations(ref, docs) == {7: OUT}
    jl.write_text(json.dumps({"id": 99, "output": "x"}) + "\n")
    with pytest.raises(ValueError, match="not a document"):
        M.load_generations(jl, docs)


def _fake_dataset(tmp_path, monkeypatch):
    docs = [dict(DOC, id=i, prompt=f"P{i}") for i in range(3)]
    monkeypatch.setattr(J, "load_all", lambda d, s: docs)
    monkeypatch.setattr(SC, "load_all", lambda d, s: docs)
    monkeypatch.setattr(J, "dataset_sha", lambda d, s: "f" * 64)
    return docs


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class StubEngine:
    """Answers yes when the asked-for description is in the context."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompts):
        self.calls.append(len(prompts))
        out = []
        for p in prompts:
            ctx, q = re.match(r"Context: (.*)### Instruction: Does this context include the (.*)\? Please",
                              p, re.S).groups()
            out.append(" yes" if q in ctx else " No.")
        return out


def test_judge_asks_exactly_the_existing_entries_in_one_batch(tmp_path, monkeypatch):
    _fake_dataset(tmp_path, monkeypatch)
    ga = _write(tmp_path / "a.jsonl", [{"id": 0, "output": OUT}, {"id": 1, "output": OUT}])
    gb = _write(tmp_path / "b.jsonl", [{"id": 0, "output": OUT.replace("music school", "bakery")},
                                       {"id": 1, "output": None}])
    eng = StubEngine()
    paths = J.judge_files("16K", tmp_path, [ga, gb], lambda prompts: (eng, {"vllm_version": "stub"}))
    # selftest (4) first, then ONE batched call for both files: 3 + 3 + 3 + 0 instructions
    assert eng.calls == [4, 9]
    va = M.load_verdicts(paths[0])
    assert [v["verdict"] for v in va[0]] == ["yes", "yes", "yes"]
    vb = M.load_verdicts(paths[1])
    assert [v["verdict"] for v in vb[0]] == ["no", "yes", "yes"] and 1 not in vb
    meta = json.loads(paths[0].with_suffix(".meta.json").read_text())
    assert meta["generations_sha"] == J.sha16(ga) and meta["judge_prompt"] == "paper"
    assert meta["judge_sampling"] == R.JUDGE_SAMPLING and meta["selftest"]["correct"] == 4
    # up to date: no engine is built again
    J.judge_files("16K", tmp_path, [ga, gb], lambda prompts: pytest.fail("judged again"))


def test_a_judge_that_fails_the_selftest_scores_nothing(tmp_path, monkeypatch):
    _fake_dataset(tmp_path, monkeypatch)
    ga = _write(tmp_path / "a.jsonl", [{"id": 0, "output": OUT}])
    with pytest.raises(SystemExit, match="selftest"):
        J.judge_files("16K", tmp_path, [ga], lambda prompts: (lambda ps: ["yes"] * len(ps), {}))
    assert not (tmp_path / "a.verdicts.jsonl").exists()


def test_scorer_pairs_runs_and_refuses_stale_verdicts(tmp_path, monkeypatch):
    _fake_dataset(tmp_path, monkeypatch)
    ga = _write(tmp_path / "a.jsonl", [{"id": i, "output": OUT} for i in range(3)])
    gb = _write(tmp_path / "b.jsonl", [{"id": i, "output": OUT.replace("music school", "bakery")} for i in range(2)])
    J.judge_files("16K", tmp_path, [ga, gb], lambda prompts: (StubEngine(), {}))
    res = SC.score("16K", tmp_path, [("a", str(ga), None), ("b", str(gb), None)], iters=100)
    assert res["ids"] == [0, 1]                                        # common documents only
    a, b = res["runs"]["a"]["ALL"], res["runs"]["b"]["ALL"]
    assert (a["CR"], a["STIC-2"], a["STIC-1"]) == (75.0, 60.0, 100.0)
    assert (b["STIC-2"], b["STIC-1"]) == (40.0, 66.7)
    assert res["deltas"]["b"]["ALL"]["STIC-2"]["delta"] == -20.0
    assert "paper Llama3.1-8B" in SC.format_table(res)
    _write(ga, [{"id": i, "output": OUT + " edited"} for i in range(3)])
    with pytest.raises(ValueError, match="different version"):
        SC.score("16K", tmp_path, [("a", str(ga), None)])


def test_the_harness_imports_nothing_from_this_repository():
    for f in (REPO / "longgenbench_eval").glob("*.py"):
        src = f.read_text()
        assert not re.search(r"^\s*(from|import)\s+(data|modules|utils|scripts)\b", src, re.M), f.name
