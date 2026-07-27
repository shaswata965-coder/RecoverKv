"""Two Qwen-shaped generation hazards, both found by running Qwen2 end-to-end.

1. A model whose ``config.vocab_size`` exceeds ``len(tokenizer)``. Qwen2.5 pads
   its vocabulary to 152064 for tensor-core alignment against a 151665-id
   tokenizer, leaving 399 ids that decode to ``''``. Emitting one crashes
   ``generate(stop_strings=...)``, which GSM8K passes.

2. The flash hook's forward-argument extraction. ``position_embeddings`` is the
   last parameter of the attention forward, so a positional fallback aimed at an
   early index returns ``attention_mask`` — a wrong tensor, consumed silently.
"""
from __future__ import annotations

import pytest

from utils.generation import unmappable_token_ids


class _Cfg:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


class _Model:
    def __init__(self, vocab_size):
        self.config = _Cfg(vocab_size)


class _Tok:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class TestUnmappableTokenIds:
    def test_qwen_style_gap_is_reported(self):
        """The real numbers: Qwen2.5-7B-Instruct is 152064 vs 151665."""
        ids = unmappable_token_ids(_Model(152064), _Tok(151665))
        assert ids == list(range(151665, 152064))
        assert len(ids) == 399

    def test_no_gap_returns_none_so_generation_is_untouched(self):
        """Llama-3.1 and Mistral-v0.2 both have vocab_size == len(tokenizer).

        None, not [] — the runner adds no `suppress_tokens` kwarg at all, so
        their generation config is byte-identical to before this existed.
        """
        assert unmappable_token_ids(_Model(128256), _Tok(128256)) is None
        assert unmappable_token_ids(_Model(32000), _Tok(32000)) is None

    def test_a_tokenizer_larger_than_the_model_is_not_a_gap(self):
        """Added tokens beyond the embedding table are a different problem, and
        suppressing nothing is the right answer here."""
        assert unmappable_token_ids(_Model(1000), _Tok(1200)) is None

    def test_missing_pieces_are_tolerated(self):
        assert unmappable_token_ids(_Model(100), None) is None
        assert unmappable_token_ids(object(), _Tok(10)) is None
        assert unmappable_token_ids(_Model(100), object()) is None


class TestStopStringsHazardIsReal:
    """Pins the mechanism, so the fix is not guarding a hypothetical.

    Uses a tiny real tokenizer rather than a stub because the failure lives in
    transformers' own ``StopStringCriteria``: it sizes ``embedding_vec`` by
    ``len(tokenizer)`` and indexes it with generated ids.
    """

    def test_an_id_past_the_tokenizer_raises_in_stop_string_criteria(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer
        from transformers.generation.stopping_criteria import StopStringCriteria

        try:
            tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        except Exception:                                    # pragma: no cover
            pytest.skip("no tokenizer available offline")

        crit = StopStringCriteria(tokenizer=tok, stop_strings=["\n\nQuestion:"])
        n = crit.embedding_vec.shape[0]
        assert n == len(tok), "criterion is sized by the tokenizer, not the model"

        crit(torch.tensor([[1, 2, 3]]), None)                # in range: fine
        with pytest.raises(IndexError):
            crit(torch.tensor([[1, 2, n + 5]]), None)        # past it: the crash


class TestExtractArg:
    """``position_embeddings`` must never be taken positionally."""

    def test_name_lookup_wins(self):
        from modules.windowed_cache.hooks import _extract_arg

        assert _extract_arg((1, 2, 3), {"position_embeddings": "pe"},
                            "position_embeddings") == "pe"

    def test_no_positional_fallback_when_pos_is_omitted(self):
        """The bug this prevents: with pos=1 the call returned args[1], which is
        `attention_mask` on every architecture — a real tensor, so nothing
        downstream would raise, and RoPE would be applied from a mask."""
        from modules.windowed_cache.hooks import _extract_arg

        args = ("hidden", "ATTENTION_MASK", "position_ids", None, False, True,
                "cache_position", "position_embeddings")
        assert _extract_arg(args, {}, "position_embeddings") is None

    def test_positional_fallback_still_works_where_it_is_correct(self):
        from modules.windowed_cache.hooks import _extract_arg

        args = ("hidden", "mask", "pids")
        assert _extract_arg(args, {}, "hidden_states", 0) == "hidden"
        assert _extract_arg(args, {}, "position_ids", 2) == "pids"

    def test_the_indices_the_hook_uses_match_the_real_signatures(self):
        """The two positional indices the hook still uses must be right, and
        position_embeddings must be nowhere near the index it used to guess.

        Variadics are excluded: LlamaAttention.forward ends in ``**kwargs``, so
        position_embeddings is the last NAMED parameter without being the last
        entry in the signature.
        """
        import inspect

        pytest.importorskip("transformers")
        from transformers.models.llama import modeling_llama as L
        from transformers.models.qwen2 import modeling_qwen2 as Q
        from transformers.models.mistral import modeling_mistral as M

        VARIADIC = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        for cls in (L.LlamaAttention, Q.Qwen2Attention, M.MistralAttention):
            named = [n for n, p in inspect.signature(cls.forward).parameters.items()
                     if n != "self" and p.kind not in VARIADIC]
            assert named.index("hidden_states") == 0, cls.__name__
            assert named.index("position_ids") == 2, cls.__name__
            if "position_embeddings" in named:
                # The point: index 1 is attention_mask, so the old pos=1 fallback
                # could only ever have returned the wrong tensor.
                assert named.index("position_embeddings") != 1, cls.__name__
                assert named.index("position_embeddings") == len(named) - 1, (
                    f"{cls.__name__}: position_embeddings is no longer the last "
                    f"named parameter. The hook's name-only lookup is still safe; "
                    f"this assumption is stale."
                )
            assert named[1] == "attention_mask", cls.__name__
