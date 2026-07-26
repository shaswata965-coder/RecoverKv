"""Tests for utils.prompting — one BOS, not two.

`apply_chat_template(tokenize=False)` returns a string that already carries the
BOS; `tokenizer(prompt)` then adds another, because `add_special_tokens`
defaults to True. Measured on the real tokenizers before the fix:

    Llama-3.1-8B-Instruct  [128000, 128000, 128006, 882]
    Mistral-7B-Instruct    [1, 1, 733, 16289]

These use a stub tokenizer so the check runs without a model download; the
`encode_prompt` contract they pin is exactly what the three generation runners
depend on.
"""
from __future__ import annotations

import pytest

from utils.prompting import encode_prompt, prompt_carries_bos


class _StubTokenizer:
    """Minimal stand-in: prepends BOS iff add_special_tokens, splits on spaces."""

    bos_token = "<s>"
    bos_token_id = 1

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, add_special_tokens=True, **kwargs):
        self.calls.append({"add_special_tokens": add_special_tokens, **kwargs})
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for tok in prompt.split():
            ids.append(self.bos_token_id if tok == self.bos_token else 100 + len(tok))
        return {"input_ids": [ids]}


class _NoBosTokenizer(_StubTokenizer):
    bos_token = None
    bos_token_id = None


class TestPromptCarriesBos:
    def test_true_for_a_templated_string(self):
        assert prompt_carries_bos(_StubTokenizer(), "<s> [INST] hi [/INST]")

    def test_false_for_a_raw_prompt(self):
        assert not prompt_carries_bos(_StubTokenizer(), "Answer the question:")

    def test_false_when_the_tokenizer_has_no_bos(self):
        assert not prompt_carries_bos(_NoBosTokenizer(), "<s> anything")


class TestEncodePrompt:
    def test_templated_prompt_does_not_get_a_second_bos(self):
        """The defect: [1, 1, ...] instead of [1, ...]."""
        tok = _StubTokenizer()
        ids = encode_prompt(tok, "<s> [INST] hi [/INST]")["input_ids"][0]
        assert ids[0] == tok.bos_token_id
        assert ids[1] != tok.bos_token_id, f"duplicate BOS: {ids[:4]}"
        assert tok.calls[0]["add_special_tokens"] is False

    def test_raw_prompt_still_gets_its_bos(self):
        """LongBench's few-shot datasets skip the chat template on purpose —
        they need the tokenizer to supply the BOS, so this must not be a blanket
        add_special_tokens=False."""
        tok = _StubTokenizer()
        ids = encode_prompt(tok, "Passage: foo Answer: bar")["input_ids"][0]
        assert ids[0] == tok.bos_token_id
        assert tok.calls[0]["add_special_tokens"] is True

    def test_explicit_override_wins(self):
        tok = _StubTokenizer()
        encode_prompt(tok, "<s> hi", add_special_tokens=True)
        assert tok.calls[0]["add_special_tokens"] is True

    def test_kwargs_are_forwarded(self):
        tok = _StubTokenizer()
        encode_prompt(tok, "hi", return_tensors="pt", truncation=False)
        assert tok.calls[0]["return_tensors"] == "pt"
        assert tok.calls[0]["truncation"] is False

    def test_tokenizer_without_bos_is_left_alone(self):
        tok = _NoBosTokenizer()
        encode_prompt(tok, "hi")
        assert tok.calls[0]["add_special_tokens"] is True

    def test_surviving_duplicate_bos_warns(self, caplog):
        """A template that embeds the BOS in a form the prefix test misses must
        not reintroduce the bug silently — that is how it lasted this long."""
        import utils.prompting as P
        P._warned_duplicate[0] = False

        class _Sneaky(_StubTokenizer):
            def __call__(self, prompt, add_special_tokens=True, **kwargs):
                # BOS present in the ids but NOT as a literal string prefix.
                return {"input_ids": [[1, 1, 42]]}

        with caplog.at_level("WARNING", logger="utils.prompting"):
            encode_prompt(_Sneaky(), "no literal bos here")
        assert any(
            "DUPLICATE leading BOS" in r.getMessage() for r in caplog.records
        ), [r.getMessage() for r in caplog.records]
        P._warned_duplicate[0] = False


@pytest.mark.parametrize("runner", ["longbench_runner", "gsm8k_runner", "ruler_runner"])
def test_every_chat_templated_runner_uses_encode_prompt(runner):
    """No runner may go back to a bare tokenizer() on a templated prompt."""
    import importlib
    from pathlib import Path

    mod = importlib.import_module(f"modules.evaluation.{runner}")
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "encode_prompt(" in src, f"{runner} does not use encode_prompt"
    # The one bare tokenizer(prompt, ...) still allowed is LongBench's
    # truncation-length probe, which measures the RAW prompt and must match
    # THUDM/pred.py byte for byte.
    bare = src.count("tokenizer(prompt, truncation=False, return_tensors=")
    allowed = 1 if runner == "longbench_runner" else 0
    assert bare == allowed, (
        f"{runner} has {bare} bare tokenizer(prompt, ...) call(s), expected {allowed}"
    )
