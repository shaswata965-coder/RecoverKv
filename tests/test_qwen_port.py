"""Qwen2.5 port — the parts that need no torch (config, prompting, generation).

Pins the three model-agnostic guarantees the port rests on:

* ``ModelConfig`` hardens dtype and validates the new context/RoPE overrides;
* ``encode_prompt`` decides ``add_special_tokens`` from *has-BOS AND
  not-carries-BOS*, which is False on Qwen (no BOS) and unchanged on Llama;
* ``unmappable_token_ids`` returns the padded-vocab gap on Qwen and ``None`` on
  a model with no gap (so Llama/Mistral add no logits processor).

The numerics fixes (grid dtype, YaRN a²) need torch and live in
``tests/test_qwen_numerics.py``.
"""

from __future__ import annotations

import pytest

from utils.config import ConfigValidationError, ModelConfig
from utils.generation import unmappable_token_ids
from utils.prompting import encode_prompt, tokenizer_has_bos


# ---------------------------------------------------------------------------
# Fakes — no torch, no transformers, no weights.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer: records the add_special_tokens it was called with."""

    def __init__(self, *, bos_token, bos_token_id, n_ids):
        self.bos_token = bos_token
        self.bos_token_id = bos_token_id
        self._n = n_ids
        self.last_add_special = None

    def __len__(self):
        return self._n

    def __call__(self, prompt, **kwargs):
        self.last_add_special = kwargs.get("add_special_tokens")
        # Emit a leading BOS only when add_special_tokens is True and we have one.
        ids = []
        if self.last_add_special and self.bos_token_id is not None:
            ids.append(self.bos_token_id)
        ids.append(999)
        return {"input_ids": ids}


class _FakeConfig:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


class _FakeModel:
    def __init__(self, vocab_size):
        self.config = _FakeConfig(vocab_size)


# Representative tokenizers.
def _llama_tok():
    return _FakeTokenizer(bos_token="<|begin_of_text|>", bos_token_id=128000, n_ids=128256)


def _qwen_tok():
    # Qwen2.5: no BOS token; config bos_token_id is <|endoftext|> but the
    # tokenizer's bos_token is None, and vocab is padded past len(tokenizer).
    return _FakeTokenizer(bos_token=None, bos_token_id=None, n_ids=151665)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


def test_model_config_rejects_unknown_dtype():
    with pytest.raises(ConfigValidationError):
        ModelConfig(dtype="bloat16")


def test_model_config_rejects_unknown_attn():
    with pytest.raises(ConfigValidationError):
        ModelConfig(attn_implementation="flashy")


def test_model_config_accepts_yarn_overrides():
    m = ModelConfig(
        name="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
        max_position_embeddings=131072,
        rope_scaling={"rope_type": "yarn", "factor": 4.0,
                      "original_max_position_embeddings": 32768},
    )
    assert m.max_position_embeddings == 131072
    assert m.rope_scaling["rope_type"] == "yarn"


def test_model_config_defaults_leave_overrides_none():
    # Every existing (Llama/Mistral) config sets none of the four → all None,
    # so utils.model_loading applies no overrides. This is what keeps them inert.
    m = ModelConfig()
    assert m.max_position_embeddings is None
    assert m.rope_scaling is None
    assert m.sliding_window is None
    assert m.use_sliding_window is None


@pytest.mark.parametrize("bad", [0, -5, True, 3.5])
def test_model_config_rejects_bad_max_position_embeddings(bad):
    with pytest.raises(ConfigValidationError):
        ModelConfig(max_position_embeddings=bad)


def test_model_config_rejects_non_bool_use_sliding_window():
    with pytest.raises(ConfigValidationError):
        ModelConfig(use_sliding_window="yes")


# ---------------------------------------------------------------------------
# Prompting — the BOS rule truth table
# ---------------------------------------------------------------------------


def test_tokenizer_has_bos():
    assert tokenizer_has_bos(_llama_tok()) is True
    assert tokenizer_has_bos(_qwen_tok()) is False


def test_llama_raw_prompt_gets_bos():
    # has BOS, prompt does not carry it → add_special_tokens True (unchanged).
    tok = _llama_tok()
    encode_prompt(tok, "Solve: 2+2=")
    assert tok.last_add_special is True


def test_llama_templated_prompt_no_second_bos():
    # has BOS, prompt already opens with it → add_special_tokens False (unchanged).
    tok = _llama_tok()
    encode_prompt(tok, "<|begin_of_text|>hello")
    assert tok.last_add_special is False


def test_qwen_prompt_never_adds_special_tokens():
    # No BOS at all → the `has BOS` conjunct forces False on BOTH rows, matching
    # kvpress. Raw and templated alike.
    tok = _qwen_tok()
    encode_prompt(tok, "<|im_start|>system\nyou are helpful")
    assert tok.last_add_special is False
    tok2 = _qwen_tok()
    encode_prompt(tok2, "raw few-shot prompt")
    assert tok2.last_add_special is False


def test_explicit_add_special_tokens_still_wins():
    tok = _qwen_tok()
    encode_prompt(tok, "x", add_special_tokens=True)
    assert tok.last_add_special is True


# ---------------------------------------------------------------------------
# Generation safety — the padded-vocab gap
# ---------------------------------------------------------------------------


def test_unmappable_none_when_no_gap():
    # Llama/Mistral: vocab_size == len(tokenizer) → None → no logits processor.
    assert unmappable_token_ids(_FakeModel(128256), _llama_tok()) is None


def test_unmappable_returns_gap_on_qwen():
    # Qwen2.5: vocab_size 152064 > len(tokenizer) 151665 → the 399 padded ids.
    ids = unmappable_token_ids(_FakeModel(152064), _qwen_tok())
    assert ids == list(range(151665, 152064))
    assert len(ids) == 399


def test_unmappable_none_on_missing_config():
    class _NoConfig:
        config = None
    assert unmappable_token_ids(_NoConfig(), _qwen_tok()) is None
