"""The LongBench runner's batched path (LongBenchRunner._run_dataset_batched).

On a tiny float64 Llama with the full cache, on CPU:

1. With no fixed prompt length and no truncation, the batched path produces
   exactly the B=1 path's predictions (same encoding, same decode, cut at EOS).
2. With ``prompt_tokens``, rows are middle-truncated to exactly that many tokens,
   shorter examples are skipped, and batching them changes no prediction: a batch
   of 3 gives what batches of 1 give.
3. A row that finishes early is cut at its first EOS, and an out-of-memory batch
   is halved and recorded.
"""

from __future__ import annotations

import json
import types

import pytest
import torch

pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from modules.evaluation import longbench_runner as LR  # noqa: E402
from utils.config import LongBenchConfig  # noqa: E402

WORDS = [f"w{i}" for i in range(60)]
VOCAB = {"[UNK]": 0, "<s>": 1, "</s>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
DS = "gov_report"
TEMPLATE = "Doc: {context} Q: {input} A:"


@pytest.fixture(scope="module")
def tok():
    t = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    t.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    return PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<s>", eos_token="</s>",
                                   unk_token="[UNK]", pad_token="</s>")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(VOCAB), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      bos_token_id=1, eos_token_id=2, pad_token_id=2)
    cfg._attn_implementation = "eager"
    m = LlamaForCausalLM(cfg).double().eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(30.0)          # peaked logits: no near-ties for greedy to flip on
    m.generation_config.eos_token_id = 2
    m.generation_config.pad_token_id = 2
    return m


def _examples(lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, n in enumerate(lengths):
        ctx = " ".join(WORDS[int(j)] for j in torch.randint(0, len(WORDS), (n,), generator=g))
        out.append({"context": ctx, "input": f"w{i} w{i + 1}", "answers": [f"w{i}"], "length": n,
                    "all_classes": None, "_id": f"id{i:02d}"})
    return out


def _run(tok, model, examples, monkeypatch, tmp_path, **lb):
    r = object.__new__(LR.LongBenchRunner)
    r.config = types.SimpleNamespace(model=types.SimpleNamespace(name="tiny-base"), cache=types.SimpleNamespace())
    r.lb = LongBenchConfig(datasets=[DS], max_length=None, output_dir=str(tmp_path), **lb)
    r.model, r.tokenizer = model, tok
    r.is_windowed, r.cache_backend_package = False, None
    r.dataset2prompt, r.dataset2maxlen = {DS: TEMPLATE}, {DS: 8}
    r._resolved_sample, r._batch_meta = None, None
    metas = []
    monkeypatch.setattr(r, "_write_meta", lambda name, n, mg, a, b, eps, out: metas.append(
        (name, n, json.loads(json.dumps(r._batch_meta)) if r._batch_meta else None)))
    monkeypatch.setattr(LR, "load_longbench_dataset", lambda name, use_e_variant=False: examples)
    tmp_path.mkdir(parents=True, exist_ok=True)
    r._run_dataset(DS, tmp_path)
    rows = [json.loads(l) for l in (tmp_path / f"{DS}.jsonl").read_text().splitlines()]
    return rows, metas[-1]


def test_batched_path_reproduces_the_b1_path_without_truncation(tok, model, monkeypatch, tmp_path):
    ex = _examples([20, 31, 20, 44, 31, 20])
    one, m1 = _run(tok, model, ex, monkeypatch, tmp_path / "a")
    assert m1[2] is None, "batch_size 1 without prompt_tokens must take the original path"
    bat, mb = _run(tok, model, ex, monkeypatch, tmp_path / "b", batch_size=3)
    assert [r["_id"] for r in bat] == [e["_id"] for e in ex]
    assert [r["pred"] for r in bat] == [r["pred"] for r in one]
    assert [r["answers"] for r in bat] == [r["answers"] for r in one]
    # only identical lengths share a batch: 3 x 20-word prompts, 2 x 31, 1 x 44
    sizes = sorted(len(b) for b in mb[2]["batches_planned"])
    assert sizes == [1, 2, 3]


def test_fixed_length_truncates_skips_and_batching_changes_nothing(tok, model, monkeypatch, tmp_path):
    ex = _examples([40, 12, 55, 60, 13, 48, 70, 41])          # 12 and 13 words are too short
    L = 30
    kw = dict(prompt_tokens={DS: L}, num_samples=5)
    bat, mb = _run(tok, model, ex, monkeypatch, tmp_path / "b", batch_size=3, **kw)
    one, m1 = _run(tok, model, ex, monkeypatch, tmp_path / "a", batch_size=1, **kw)
    meta = mb[2]
    assert meta["prompt_tokens"] == L and meta["skipped_short"] == ["id01", "id04"]
    want_ids = ["id00", "id02", "id03", "id05", "id06"]       # the first 5 that are long enough
    assert [r["_id"] for r in bat] == want_ids == [r["_id"] for r in one]
    assert [r["pred"] for r in bat] == [r["pred"] for r in one]
    assert [len(b) for b in meta["batches_planned"]] == [3, 2]
    assert all(b["prompt_tokens"] == L for b in meta["batches_realised"])
    assert all(n > L for n in meta["prompt_tokens_original"].values())
    # and the truncation is the documented one: head L//2, tail L - L//2 of the full ids
    r = object.__new__(LR.LongBenchRunner)
    full = list(range(100))
    assert r.middle_truncate_ids(full, L) == full[:15] + full[85:]


def test_rows_are_cut_at_their_own_eos_and_oom_halves_the_batch(tok, model, monkeypatch, tmp_path):
    ex = _examples([40, 41, 42, 43])
    calls = []
    eos, w = 2, {w_: VOCAB[w_] for w_ in WORDS}

    def fake_generate(input_ids, **kw):
        calls.append(int(input_ids.shape[0]))
        if input_ids.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        new = []
        for row in input_ids:
            k = int(row[-3]) % 4                             # a row-specific stopping point
            toks = [w["w5"], w["w6"], w["w7"]][:k] + [eos]
            new.append(toks + [eos] * (6 - len(toks)))       # padded like generate pads a finished row
        return torch.cat([input_ids, torch.tensor(new)], 1)

    monkeypatch.setattr(model, "generate", fake_generate)
    rows, meta = _run(tok, model, ex, monkeypatch, tmp_path, batch_size=4, prompt_tokens={DS: 30})
    assert calls[0] == 4 and all(c <= 2 for c in calls[1:])
    assert meta[2]["oom_events"] and meta[2]["oom_events"][0]["fell_back_to"] == 2
    assert meta[2]["final_cap"] == 2
    for r in rows:
        assert "w7" not in r["pred"] or r["pred"].split()[-1] == "w7"
        assert "</s>" not in r["pred"]
    gens = [g for b in meta[2]["batches_realised"] for g in b["gen_tokens"]]
    assert all(0 <= g <= 3 for g in gens)


def test_config_refuses_bad_batching_knobs():
    from utils.config import ConfigValidationError
    for bad in (dict(batch_size=0), dict(batch_tokens=0), dict(prompt_tokens={DS: 4}),
                dict(prompt_tokens=[1, 2]), dict(completion_rungs_stride=0), dict(completion_rungs="yes")):
        with pytest.raises(ConfigValidationError):
            LongBenchConfig(**bad)
