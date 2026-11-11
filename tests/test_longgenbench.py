"""LongGenBench generation (mode ``longgenbench``): protocol, batching, config, on CPU.

The runner must generate the way the runs behind the paper's Table 2 did
(``inference.py`` @ 7b59ec6, see ``longgenbench_eval/reference.py``): the
dataset prompt verbatim with one BOS and no chat template, stop at
``*** finished`` (excluded), the model's EOS ids, ``prefix + generation``, and
temperature 0.95 / top_p 0.95 / no top-k. It batches only documents whose
prompts have the same token length (the windowed cache takes no padding), falls
back down 16 -> 8 -> 6 when a batch runs out of memory, and writes records that
``longgenbench_eval`` scores as they are. Judging and scoring are tested in
``tests/test_longgenbench_eval.py``.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from longgenbench_eval import metrics as M
from longgenbench_eval import reference as R
from modules.evaluation import longgenbench_batching as B

REPO = Path(__file__).resolve().parents[1]

DOC = {"type": "Floor", "number": 4, "prefix": "#*# Floor 1:",
       "checks_once": {"2": "music school"}, "checks_range": {"3": "gym", "4": "gym"},
       "checks_periodic": {"2": "restaurant"}}

#: what the stub model writes for each document id (decoded text, before the cut)
TEXT = {
    101: "\nLobby.\n#*# Floor 2:\nStudio.\n*** finished ***\n#*# Floor 3: never",
    102: "\nLobby.\n#*# Floor 2:\nGym.",                          # ends on EOS
    103: "\nLobby.",                                             # a different prompt length
    104: "\nA.\n#*# Floor 2:\nB.",
}


def _stub_run(tmp_path, monkeypatch, decoding="reference", ids=(101, 102, 103), lengths=None,
              batch_size=16, fallback=(8, 6), oom_above=None, skip_oom=False, batches=None):
    torch = pytest.importorskip("torch")
    from modules.evaluation import longgenbench_runner as RUN
    from utils.config import ExperimentConfig, LongGenBenchConfig, ModelConfig

    lengths = lengths or {101: 4, 102: 4, 103: 5, 104: 4}
    docs = {i: dict(DOC, id=i, prompt=f"P{i}") for i in lengths}
    monkeypatch.setattr(RUN, "load_all", lambda dd, s: [docs[i] for i in sorted(docs)])
    monkeypatch.setattr(RUN, "dataset_sha", lambda dd, s: "f" * 64)

    class Enc(dict):
        def __init__(self, t):
            super().__init__(input_ids=t, attention_mask=torch.ones_like(t))
            self.input_ids, self.attention_mask = t, torch.ones_like(t)

    class Tok:
        bos_token, bos_token_id, eos_token_id, pad_token_id = "<s>", 1, 9, None
        eos_token = "</s>"

        def __call__(self, prompt, add_special_tokens=True, **kw):
            i = int(prompt[1:])
            body = [5] * (lengths[i] - 1)
            return Enc(torch.tensor([([1] if add_special_tokens else []) + body]))

        def decode(self, ids, skip_special_tokens=True):
            return TEXT[ids[0] - 1000] if ids else ""

    seen = {"seeds": [], "shapes": []}

    class Model:
        device = torch.device("cpu")
        config = types.SimpleNamespace(max_position_embeddings=131072)
        generation_config = types.SimpleNamespace(eos_token_id=[2, 3, 9])

        def generate(self, input_ids, **kw):
            seen.update(kw)
            seen["seeds"].append(torch.initial_seed())
            seen["shapes"].append(tuple(input_ids.shape))
            if oom_above is not None and input_ids.shape[0] > oom_above:
                raise torch.cuda.OutOfMemoryError("stub: too many rows")
            rows = []
            for r in range(input_ids.shape[0]):
                i = seen["batch_ids"][r]
                # token 1000+id carries the row's text; then EOS for 102, pad (= EOS 9) otherwise
                rows.append([1000 + i, 7, 2 if i == 102 else 9])
            out = torch.cat([input_ids, torch.tensor(rows)], dim=-1)
            for k in range(3):                          # generate calls the criteria once per step
                for crit in kw["stopping_criteria"]:
                    assert not crit(out[:, :input_ids.shape[1] + k + 1], None).any()
            return out

    cfg = ExperimentConfig()
    cfg.model = ModelConfig(name="meta-llama/Llama-3.1-8B-Instruct", dtype="float16",
                            attn_implementation="flash_attention_2")
    cfg.longgenbench = LongGenBenchConfig(data_dir=str(tmp_path), setting="16K", ids=list(ids),
                                          batches=batches, output_dir=str(tmp_path / "out"),
                                          decoding=decoding, batch_size=batch_size,
                                          batch_fallback=list(fallback), skip_oom=skip_oom)
    runner = RUN.LongGenBenchRunner(cfg)
    monkeypatch.setattr(runner, "_load_model_and_tokenizer", lambda: (Model(), Tok()))
    real = runner._generate_batch

    def spy(bi, ds, max_new):
        seen["batch_ids"] = [d["id"] for d in ds]
        return real(bi, ds, max_new)

    monkeypatch.setattr(runner, "_generate_batch", spy)
    runner.run()
    recs = [json.loads(l) for l in (tmp_path / "out" / "generations.jsonl").read_text().splitlines()]
    meta = json.loads((tmp_path / "out" / "generations.meta.json").read_text())
    return seen, recs, meta, docs


def test_runner_generates_as_the_reference_runs_did(tmp_path, monkeypatch):
    seen, recs, meta, docs = _stub_run(tmp_path, monkeypatch)
    assert seen["max_new_tokens"] == 16000 and seen["stop_strings"] == ["*** finished"]
    assert seen["eos_token_id"] == [2, 3, 9]                        # the generation config's, as vLLM
    assert seen["do_sample"] is True and seen["temperature"] == 0.95 and seen["top_p"] == 0.95
    assert seen["top_k"] == 0                                        # vLLM's top_k=-1: no top-k
    assert seen["seeds"] == [6211027, 6211027]                       # the seed, before every batch
    # equal-length batches: 101 and 102 share a length, 103 does not
    assert seen["shapes"] == [(2, 4), (1, 5)]
    assert meta["batches_planned"] == [[101, 102], [103]]
    by = {r["id"]: r for r in recs}
    assert by[101]["output"] == "#*# Floor 1:\nLobby.\n#*# Floor 2:\nStudio.\n"   # cut at the stop string
    assert by[101]["stopped_on"] == "stop_string" and by[101]["gen_tokens"] == 2
    assert by[102]["output"] == "#*# Floor 1:\nLobby.\n#*# Floor 2:\nGym."
    assert by[102]["stopped_on"] == "eos" and by[102]["gen_tokens"] == 3
    assert by[101]["batch_ids"] == [101, 102] and by[101]["batch_size"] == 2
    assert by[103]["prompt_tokens"] == 5 and by[101]["prompt_tokens"] == 4          # one BOS
    assert by[101]["completion_rate"] == 50.0
    assert by[101]["first_token_s"] is not None and by[101]["decode_ms_per_step"] is not None
    g = meta["generation"]
    assert g["decoding"] == "reference" and g["batching"] == "equal-length prompts, no padding"
    assert g["batch_size"] == 16 and g["batch_fallback"] == [8, 6]
    assert g["sampling"]["seed_rule"] == "the same seed set before every batch"
    assert meta["ids"] == [101, 102, 103] and meta["oom_events"] == []
    assert meta["prompt_tokens"] == {"101": 4, "102": 4, "103": 5}
    assert meta["cache_type"] == "full_cache" and meta["read_gate"]["verdict"] == "not-expected"
    assert meta["dataset_revision"] == R.HF_REVISION and meta["prompt_format"] == "raw_completion"
    # The records are what the harness scores, as they are.
    out = M.load_generations(tmp_path / "out" / "generations.jsonl", list(docs.values()))
    assert out == {i: by[i]["output"] for i in (101, 102, 103)}


def test_out_of_memory_falls_back_to_smaller_batches(tmp_path, monkeypatch):
    seen, recs, meta, _ = _stub_run(tmp_path, monkeypatch, ids=(101, 102, 104), batch_size=4,
                                    fallback=(2, 1), oom_above=2)
    assert seen["shapes"] == [(3, 4), (2, 4), (1, 4)]              # 3 fails, re-run as 2 + 1
    assert meta["oom_events"] == [{"batch": 0, "ids": [101, 102, 104], "size": 3, "fell_back_to": 2}]
    assert meta["final_batch_cap"] == 2
    assert sorted(r["id"] for r in recs) == [101, 102, 104] and all(r["output"] for r in recs)
    assert [b["size"] for b in meta["batches_realised"]] == [2, 1]


def test_out_of_memory_at_the_smallest_size(tmp_path, monkeypatch):
    with pytest.raises(Exception, match="too many rows"):
        _stub_run(tmp_path, monkeypatch, ids=(101, 102), batch_size=2, fallback=(), oom_above=1)
    _, recs, meta, _ = _stub_run(tmp_path, monkeypatch, ids=(101, 102), batch_size=2, fallback=(),
                                 oom_above=1, skip_oom=True)
    assert [r["output"] for r in recs] == [None, None] and meta["num_oom"] == 2
    assert meta["oom_events"][0]["fell_back_to"] is None


def test_a_batch_never_mixes_prompt_lengths(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="mixes prompt lengths"):
        _stub_run(tmp_path, monkeypatch, batches=[[101, 103], [102]])


def test_greedy_is_available(tmp_path, monkeypatch):
    seen, recs, meta, _ = _stub_run(tmp_path, monkeypatch, decoding="greedy")
    assert seen["do_sample"] is False and meta["generation"]["sampling"] is None


# ---------------------------------------------------------------------------
# batch planning
# ---------------------------------------------------------------------------


def test_plan_batches_groups_equal_lengths_evenly():
    lengths = {i: (10 if i < 18 else 11) for i in range(20)}
    plan = B.plan_batches(lengths, 16)
    assert plan == [list(range(0, 9)), list(range(9, 18)), [18, 19]]   # 18 -> 9 + 9, not 16 + 2
    assert all(len({lengths[i] for i in b}) == 1 for b in plan)
    assert B.split_evenly(list(range(7)), 6) == [[0, 1, 2, 3], [4, 5, 6]]
    assert B.next_smaller(16) == 8 and B.next_smaller(8) == 6 and B.next_smaller(7) == 6
    assert B.next_smaller(6) is None


def test_pick_batch_size_by_memory():
    # 16K: ~770-token prompts, 16,000 new tokens
    assert B.pick_batch_size("ours", 39.4, 770, 16000, cache_budget=0.10) == 16
    assert B.pick_batch_size("fullkv", 79.2, 770, 16000) == 16
    assert B.pick_batch_size("fullkv", 39.4, 770, 16000) == 6           # 16 and 8 do not fit 40 GB


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_validates():
    from utils.config import ConfigValidationError, LongGenBenchConfig

    c = LongGenBenchConfig()
    assert (c.setting, c.decoding, c.sample_seed) == ("16K", "reference", 6211027)
    assert (c.batch_size, c.batch_fallback) == (16, [8, 6])
    assert LongGenBenchConfig(types="Week").types == ["Week"]
    for bad in (dict(setting="8K"), dict(decoding="repo_sampling"), dict(start=-1),
                dict(start=4, end=4), dict(max_new_tokens=0), dict(types=[]), dict(batch_size=0),
                dict(batch_fallback=[6, 8]), dict(batch_fallback=[16]), dict(ids=[1, 1]),
                dict(batches=[[1]])):
        with pytest.raises(ConfigValidationError):
            LongGenBenchConfig(**bad)


def test_the_mode_is_registered():
    src = (REPO / "main.py").read_text()
    assert '"longgenbench": "modules.evaluation.longgenbench_runner.LongGenBenchRunner"' in src


def test_both_configs_load_at_the_reference_protocol():
    from utils.config import load_config

    ours = load_config(REPO / "configs" / "longgenbench_ours_flash.yaml")
    full = load_config(REPO / "configs" / "longgenbench_full_cache.yaml")
    assert ours.run.mode == full.run.mode == "longgenbench"
    assert ours.cache.backend == "windowed" and full.cache.backend == "dynamic"
    assert ours.longgenbench.setting == full.longgenbench.setting == "16K"
    assert ours.longgenbench.decoding == full.longgenbench.decoding == "reference"
    assert ours.longgenbench.batch_size == 16 and ours.longgenbench.batch_fallback == [8, 6]
