"""LongGenBench generation (mode ``longgenbench``): the runner's protocol and config, on CPU.

The runner must generate the way the runs behind the paper's Table 2 did
(``inference.py`` @ 7b59ec6, see ``longgenbench_eval/reference.py``): the
dataset prompt verbatim with one BOS and no chat template, stop at
``*** finished`` (excluded), the model's EOS ids, ``prefix + generation``, and
temperature 0.95 / top_p 0.95 / no top-k with the same seed for every document.
Its records must be what ``longgenbench_eval`` scores. Judging and scoring are
tested in ``tests/test_longgenbench_eval.py``.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from longgenbench_eval import metrics as M
from longgenbench_eval import reference as R

REPO = Path(__file__).resolve().parents[1]

DOC = {"type": "Floor", "number": 4, "prefix": "#*# Floor 1:",
       "checks_once": {"2": "music school"}, "checks_range": {"3": "gym", "4": "gym"},
       "checks_periodic": {"2": "restaurant"}}


def _stub_run(tmp_path, monkeypatch, decoding, text):
    torch = pytest.importorskip("torch")
    from modules.evaluation import longgenbench_runner as RUN
    from utils.config import ExperimentConfig, LongGenBenchConfig, ModelConfig

    docs = [dict(DOC, id=100 + i, prompt=f"Build. *** started ***\n#*# Floor 1:{i}") for i in range(3)]
    monkeypatch.setattr(RUN, "load_type", lambda dd, s, t, start, end: docs[start:end])
    monkeypatch.setattr(RUN, "dataset_sha", lambda dd, s: "f" * 64)

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

    seen = {"seeds": []}

    class Model:
        device = torch.device("cpu")
        config = types.SimpleNamespace(max_position_embeddings=131072)
        generation_config = types.SimpleNamespace(eos_token_id=[2, 3, 9])

        def generate(self, input_ids, **kw):
            seen.update(kw)
            seen["seeds"].append(torch.initial_seed())
            out = torch.cat([input_ids, torch.tensor([[8, 8, 9]])], dim=-1)
            for k in range(3):                       # generate calls the criteria once per step
                for crit in kw["stopping_criteria"]:
                    assert not crit(out[:, :input_ids.shape[1] + k + 1], None).any()
            return out

    cfg = ExperimentConfig()
    cfg.model = ModelConfig(name="meta-llama/Llama-3.1-8B-Instruct", dtype="float16",
                            attn_implementation="flash_attention_2")
    cfg.longgenbench = LongGenBenchConfig(data_dir=str(tmp_path), setting="16K", types=["Floor"],
                                          output_dir=str(tmp_path / "out"), start=1, end=3,
                                          decoding=decoding)
    runner = RUN.LongGenBenchRunner(cfg)
    monkeypatch.setattr(runner, "_load_model_and_tokenizer", lambda: (Model(), Tok()))
    runner.run()
    recs = [json.loads(l) for l in (tmp_path / "out" / "floor.jsonl").read_text().splitlines()]
    return seen, recs, json.loads((tmp_path / "out" / "floor.meta.json").read_text()), docs


def test_runner_generates_as_the_reference_runs_did(tmp_path, monkeypatch):
    text = "\nLobby.\n#*# Floor 2:\nStudio.\n*** finished ***\n#*# Floor 3: never"
    seen, recs, meta, docs = _stub_run(tmp_path, monkeypatch, "reference", text)
    assert seen["max_new_tokens"] == 16000 and seen["stop_strings"] == ["*** finished"]
    assert seen["eos_token_id"] == [2, 3, 9]                        # the generation config's, as vLLM
    assert seen["do_sample"] is True and seen["temperature"] == 0.95 and seen["top_p"] == 0.95
    assert seen["top_k"] == 0                                        # vLLM's top_k=-1: no top-k
    assert seen["seeds"] == [6211027, 6211027]                       # the same seed per document
    assert [r["id"] for r in recs] == [101, 102]
    r = recs[0]
    assert r["output"] == "#*# Floor 1:\nLobby.\n#*# Floor 2:\nStudio.\n"   # prefix + text, cut at the stop
    assert r["stopped_on"] == "stop_string" and r["prompt_tokens"] == 4     # one BOS
    assert r["completion_rate"] == 50.0
    # the step clock: first token vs steady decode, kept apart
    assert r["first_token_s"] is not None and r["decode_ms_per_token"] is not None
    assert r["decode_ms_per_token_last_half_median"] is not None
    assert meta["decode_ms_per_token_by_document"] == [x["decode_ms_per_token"] for x in recs]
    assert meta["generation"]["decoding"] == "reference"
    assert meta["generation"]["sampling"]["seed"] == 6211027
    assert meta["dataset_revision"] == R.HF_REVISION and meta["prompt_format"] == "raw_completion"
    assert meta["cache_type"] == "full_cache" and meta["read_gate"]["verdict"] == "not-expected"
    # The records are what the harness scores, as they are.
    p = tmp_path / "out" / "floor.jsonl"
    assert M.load_generations(p, [dict(d) for d in docs]) == {101: r["output"], 102: recs[1]["output"]}


def test_runner_greedy_is_available(tmp_path, monkeypatch):
    seen, recs, meta, _ = _stub_run(tmp_path, monkeypatch, "greedy", "\nx")
    assert seen["do_sample"] is False and meta["generation"]["sampling"] is None
    assert recs[0]["stopped_on"] == "eos"


def test_config_validates():
    from utils.config import ConfigValidationError, LongGenBenchConfig

    c = LongGenBenchConfig()
    assert (c.setting, c.decoding, c.sample_seed) == ("16K", "reference", 6211027)
    assert LongGenBenchConfig(types="Week").types == ["Week"]
    for bad in (dict(setting="8K"), dict(decoding="repo_sampling"), dict(start=-1),
                dict(start=4, end=4), dict(max_new_tokens=0), dict(types=[])):
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
