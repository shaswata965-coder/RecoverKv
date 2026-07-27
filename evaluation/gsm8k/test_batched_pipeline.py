"""Tests for the batched GSM8K pipeline — padding, positions, pad-safe scoring, stops.

The batched path exists to make B=128 possible, but every one of its mechanisms is a
chance to silently change the *numbers*: left padding moves the scoring window, pad
tokens can win retention budget, per-row positions can drift, and finished rows can
keep emitting. Each of those is checked here on CPU with a stub tokenizer — no weights,
no flash-attn, no GPU.

The one thing CPU tests cannot check is that a real batched run equals a real ``B=1``
run. That is the contract :func:`compare_batched_vs_single` encodes; it is a plain
function so it can be called from a notebook or a GPU box, and the ``gpu``-marked test
at the bottom runs it when a model is available.

    cd evaluation && pytest gsm8k/test_batched_pipeline.py -q
    cd evaluation && GSM8K_TEST_MODEL=/models/Llama-3.1-8B-Instruct \\
        pytest gsm8k/test_batched_pipeline.py -q -m gpu
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import pytest
import torch

from gsm8k.batched_pipeline import (
    BatchedGSM8KPipeline,
    _cut_at_stop,
    pad_masked_scores,
)


# ---------------------------------------------------------------------------
# Stub tokenizer: word-level, deterministic, no downloads
# ---------------------------------------------------------------------------


class StubTokenizer:
    """Whitespace tokenizer with a stable vocab, enough for ``preprocess``.

    ``chat_template = None`` takes ``preprocess``'s no-template branch, which keeps the
    context/question split explicit instead of routing through a real chat template —
    the point here is padding arithmetic, not template fidelity.
    """

    chat_template = None
    bos_token = "<s>"
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self._vocab = {"<pad>": 0, "<s>": 1, "</s>": 2}
        self._inv = {0: "<pad>", 1: "<s>", 2: "</s>"}

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            idx = len(self._vocab)
            self._vocab[word] = idx
            self._inv[idx] = word
        return self._vocab[word]

    def encode(self, text, return_tensors=None, add_special_tokens=False):
        ids = [self._id(w) for w in text.split()]
        if return_tensors == "pt":
            return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        words = [self._inv.get(int(i), "?") for i in ids]
        if skip_special_tokens:
            words = [w for w in words if w not in ("<pad>", "<s>", "</s>")]
        return " ".join(words)


def _bare_pipeline() -> BatchedGSM8KPipeline:
    """A pipeline instance with only a tokenizer — no model, no weights loaded.

    ``encode_batch`` and ``_tail_has_stop`` touch nothing else, so bypassing
    ``Pipeline.__init__`` is what keeps these tests weight-free.
    """
    pipe = BatchedGSM8KPipeline.__new__(BatchedGSM8KPipeline)
    pipe.tokenizer = StubTokenizer()
    return pipe


def _contexts(lengths: Sequence[int]) -> List[str]:
    """Contexts that tokenize to exactly the requested lengths.

    ``preprocess``'s no-template branch prepends the BOS *without* a separator
    (``bos_token + context``), so the stub's whitespace split fuses it onto the first
    word — ``n`` words in, ``n`` tokens out.
    """
    return [" ".join(f"w{i}_{j}" for j in range(n)) for i, n in enumerate(lengths)]


def _is_unpatched(press: "_FakePress") -> bool:
    """Whether ``press.score`` is still the class's own method.

    ``press.score is original`` cannot be used: attribute access on a method descriptor
    builds a fresh bound-method object every time, so the identity check fails even when
    nothing was patched. The patched version is a plain closure with no ``__func__``,
    which is the actual discriminator.
    """
    return getattr(press.score, "__func__", None) is _FakePress.score


# ---------------------------------------------------------------------------
# encode_batch: padding side, mask, positions
# ---------------------------------------------------------------------------


class TestEncodeBatch:
    def test_padding_is_on_the_left(self):
        """Right padding would put the press's scoring window on pad tokens.

        SnapKV and both DefensiveKV variants score with ``hidden_states[:, -window:]``.
        If padding were on the right, that window would be pure pad for every short row
        and the scoring signal would be destroyed — silently, since nothing asserts it.
        """
        pipe = _bare_pipeline()
        enc = pipe.encode_batch(_contexts([5, 9]), question="q")

        mask = enc["attention_mask"]
        assert mask.shape[0] == 2
        # Real tokens are flush against the RIGHT edge in every row.
        assert bool(mask[:, -1].all())
        # The short row has its pad at the front.
        assert not bool(mask[0, 0])
        assert bool(mask[1].all())

    def test_true_lengths_are_recorded_not_inferred(self):
        pipe = _bare_pipeline()
        enc = pipe.encode_batch(_contexts([5, 9]), question="q")
        assert enc["true_lengths"].tolist() == [5, 9]
        # The batch runs at the widest row, not at some rounded-up constant.
        assert enc["context_ids"].shape[1] == 9

    def test_every_row_starts_at_position_zero(self):
        """Left padding shifts tokens right; positions must not follow them.

        Without this, an identical prompt would get different RoPE phase depending on
        how much padding its batch-mates forced, so a row's output would depend on its
        batch composition.
        """
        pipe = _bare_pipeline()
        enc = pipe.encode_batch(_contexts([5, 9]), question="q")
        pos, mask = enc["position_ids"], enc["attention_mask"]

        for row in range(pos.shape[0]):
            real = pos[row][mask[row]]
            assert real.tolist() == list(range(real.numel()))

    def test_pad_positions_are_clamped_not_negative(self):
        """``cumsum(mask) - 1`` is -1 on leading pads; RoPE cannot index that."""
        pipe = _bare_pipeline()
        enc = pipe.encode_batch(_contexts([3, 8]), question="q")
        assert int(enc["position_ids"].min()) >= 0

    def test_equal_lengths_need_no_padding_at_all(self):
        """The configuration the B=1 equivalence contract is validated in."""
        pipe = _bare_pipeline()
        enc = pipe.encode_batch(_contexts([7, 7, 7]), question="q")
        assert bool(enc["attention_mask"].all())
        assert enc["true_lengths"].tolist() == [7, 7, 7]

    def test_divergent_question_lengths_are_rejected(self):
        """A ragged question would need padding *after* the compressed prefix."""
        pipe = _bare_pipeline()

        real_preprocess = pipe.preprocess
        calls = {"n": 0}

        def preprocess_with_drifting_question(context, questions, **kw):
            out = real_preprocess(context, questions=questions, **kw)
            # Second row gets a longer question than the first.
            if calls["n"] == 1:
                out["questions_ids"] = [torch.zeros((1, 5), dtype=torch.long)]
            calls["n"] += 1
            return out

        pipe.preprocess = preprocess_with_drifting_question
        with pytest.raises(ValueError, match="one shared question"):
            pipe.encode_batch(_contexts([6, 6]), question="q")

    def test_single_row_batch_is_a_no_op_encoding(self):
        pipe = _bare_pipeline()
        enc = pipe.encode_batch(_contexts([6]), question="q")
        assert bool(enc["attention_mask"].all())
        assert enc["position_ids"].tolist() == [list(range(6))]


# ---------------------------------------------------------------------------
# pad_masked_scores
# ---------------------------------------------------------------------------


class _FakePress:
    """Minimal stand-in exposing the one method the wrapper patches."""

    def __init__(self, out: torch.Tensor) -> None:
        self._out = out
        self.calls = 0

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        self.calls += 1
        return self._out.clone()

    def _invoke(self):
        return self.score(None, None, None, None, None, {})


class TestPadMaskedScores:
    def test_dense_layout_pads_become_unselectable(self):
        """``[B, H, T]`` — the ScorerPress family (snapkv, streaming_llm)."""
        press = _FakePress(torch.ones(2, 3, 4))
        valid = torch.tensor([[False, False, True, True], [True, True, True, True]])

        with pad_masked_scores(press, valid):
            out = press._invoke()

        assert torch.isinf(out[0, :, :2]).all() and (out[0, :, :2] < 0).all()
        assert torch.isfinite(out[0, :, 2:]).all()
        assert torch.isfinite(out[1]).all()

    def test_flattened_layout_pads_become_unselectable(self):
        """``[B, H*T]`` — the Ada family (defensivekv, layer_defensivekv).

        The flattened layout is where a naive mask would corrupt the head striding, so
        the head count has to be recovered from the width.
        """
        b, h, t = 2, 3, 4
        press = _FakePress(torch.arange(b * h * t, dtype=torch.float).view(b, h * t))
        valid = torch.tensor([[False, True, True, True], [True, True, True, True]])

        with pad_masked_scores(press, valid):
            out = press._invoke()

        view = out.view(b, h, t)
        # Exactly the pad column of row 0, in every head.
        assert torch.isinf(view[0, :, 0]).all()
        assert torch.isfinite(view[0, :, 1:]).all()
        assert torch.isfinite(view[1]).all()

    def test_unpadded_batch_takes_the_original_code_path(self):
        """An all-true mask must not wrap at all — that is the B=1 equivalence hinge."""
        press = _FakePress(torch.ones(2, 3, 4))
        with pad_masked_scores(press, torch.ones(2, 4, dtype=torch.bool)):
            assert _is_unpatched(press)

    def test_none_mask_and_none_press_are_no_ops(self):
        press = _FakePress(torch.ones(1, 2, 3))
        with pad_masked_scores(press, None):
            assert _is_unpatched(press)
        with pad_masked_scores(None, torch.zeros(1, 3, dtype=torch.bool)):
            pass  # must not raise

    def test_score_is_restored_even_when_the_prefill_raises(self):
        """A press outlives one prefill; a leaked patch would poison later batches."""
        press = _FakePress(torch.ones(1, 2, 3))
        valid = torch.tensor([[False, True, True]])

        with pytest.raises(RuntimeError):
            with pad_masked_scores(press, valid):
                assert not _is_unpatched(press)
                raise RuntimeError("prefill blew up")

        assert _is_unpatched(press)


# ---------------------------------------------------------------------------
# stop handling
# ---------------------------------------------------------------------------


class TestStopHandling:
    def test_cut_at_stop_takes_the_earliest_boundary(self):
        text = "answer\n#### 18\n\nQuestion: next\n\nProblem: another"
        assert _cut_at_stop(text, ["\n\nQuestion:", "\n\nProblem:"]) == "answer\n#### 18"

    def test_cut_at_stop_leaves_clean_text_alone(self):
        text = "reasoning\n#### 18"
        assert _cut_at_stop(text, ["\n\nQuestion:"]) == text

    def test_tail_check_only_decodes_the_recent_window(self):
        """O(1) per step: the check must not re-decode the whole answer each token."""
        pipe = _bare_pipeline()
        tok = pipe.tokenizer

        stop_ids = tok.encode("STOP")
        long_prefix = tok.encode(" ".join(f"t{i}" for i in range(200)))

        # Stop token far in the past -> outside the lookback window -> not a hit.
        assert pipe._tail_has_stop(stop_ids + long_prefix, ["STOP"]) is False
        # Stop token at the tail -> hit.
        assert pipe._tail_has_stop(long_prefix + stop_ids, ["STOP"]) is True

    def test_no_stop_strings_never_fires(self):
        pipe = _bare_pipeline()
        assert pipe._tail_has_stop(pipe.tokenizer.encode("anything"), []) is False


# ---------------------------------------------------------------------------
# integration with the batch planner
# ---------------------------------------------------------------------------


class TestPlanIntegration:
    def test_planned_groups_encode_without_padding_when_exact(self):
        """``pad_to_multiple=None`` is the validation configuration: zero pad tokens."""
        from gsm8k.batching import plan_batches

        lengths = [10, 10, 10, 10, 25, 25]
        plan = plan_batches(lengths, max_batch_size=4, pad_to_multiple=None)
        pipe = _bare_pipeline()

        for group, width in zip(plan.groups, plan.group_lengths):
            group_lengths = [lengths[i] for i in group]
            if len(set(group_lengths)) != 1:
                continue  # mixed group: padding is expected, checked elsewhere
            enc = pipe.encode_batch(_contexts(group_lengths), question="q")
            assert bool(enc["attention_mask"].all()), (
                f"group of equal lengths {group_lengths} should need no padding"
            )
            assert width > 0

    def test_every_example_is_planned_exactly_once(self):
        from gsm8k.batching import plan_batches

        lengths = [7, 19, 12, 31, 8, 22, 15, 9]
        plan = plan_batches(lengths, max_batch_size=3, pad_to_multiple=None)
        flat = sorted(i for g in plan.groups for i in g)
        assert flat == list(range(len(lengths)))


# ---------------------------------------------------------------------------
# The B=1 equivalence contract (needs weights)
# ---------------------------------------------------------------------------


def compare_batched_vs_single(
    model: str,
    contexts: Sequence[str],
    question: str = "",
    press=None,
    max_new_tokens: int = 64,
    stop_strings: Optional[List[str]] = None,
    device: str = "cuda:0",
) -> dict:
    """Run *contexts* batched and one-at-a-time, and report where they diverge.

    This is the contract the batched path is only trustworthy under: with an
    **unpadded** group (all contexts the same token length) the batched run does the
    same arithmetic as ``B=1``, so the generated text must match exactly. Any
    difference is a bug in padding, positions, or the pad-safe scoring — not noise,
    because both paths decode greedily.

    With a *padded* group the texts may legitimately differ slightly (the press's
    ``avg_pool1d`` smoothing smears across the pad boundary), so the useful signal
    there is the *size* of the divergence, which is what ``n_mismatched`` reports.

    Returns ``{"single": [...], "batched": [...], "n_mismatched": int, "padded": bool}``.
    """
    from transformers import pipeline as hf_pipeline

    from gsm8k.batched_pipeline import BATCHED_TASK_NAME
    from gsm8k.pipeline import TASK_NAME

    single_pipe = hf_pipeline(TASK_NAME, model=model, device=device)
    single = [
        single_pipe(
            c, questions=[question], press=press,
            max_new_tokens=max_new_tokens, stop_strings=stop_strings,
        )["generations"][0]["text"]
        for c in contexts
    ]

    batched_pipe = hf_pipeline(BATCHED_TASK_NAME, model=model, device=device)
    batched = batched_pipe.generate_batch(
        contexts, question=question, press=press,
        max_new_tokens=max_new_tokens, stop_strings=stop_strings,
    )

    enc = batched_pipe.encode_batch(contexts, question)
    padded = not bool(enc["attention_mask"].all())

    return {
        "single": single,
        "batched": batched.texts,
        "n_mismatched": sum(a != b for a, b in zip(single, batched.texts)),
        "padded": padded,
    }


@pytest.mark.gpu
def test_batched_is_deterministic_and_permutation_invariant():
    """The two properties that ARE exact, and whose failure would kill batching.

    Deliberately *not* a B=1 equivalence test. Measured on Llama-3.1-8B, ``B=6`` differs
    from ``B=1`` on 3/6 rows with no press and 4/6 under SnapKV, because cuBLAS picks
    different GEMM kernels per batch size and bf16 rounding then flips near-tied greedy
    argmaxes (a press amplifies this, since its top-k is discrete). Asserting equality
    there would fail forever for a reason no code change can fix.

    What must hold, and what this checks, both at a FIXED batch size so kernel selection
    is held constant:

    * same rows, same order, twice -> identical (determinism, so runs reproduce)
    * same rows, reversed order -> identical after un-reversing (no cross-row
      contamination; a row must not depend on its batch-mates)

    Set ``GSM8K_TEST_MODEL`` to run. Skipped by default (needs weights + a GPU).
    """
    model = os.environ.get("GSM8K_TEST_MODEL")
    if not model or not torch.cuda.is_available():
        pytest.skip("set GSM8K_TEST_MODEL and run on a GPU box")

    from transformers import pipeline as hf_pipeline

    from gsm8k.batched_pipeline import BATCHED_TASK_NAME
    from gsm8k.create_huggingface_dataset import STOP_STRINGS
    from gsm8k.run_gsm8k import build_press

    contexts = [
        "Solve. " + " ".join(["token"] * 40) + f" case {i}" for i in range(4)
    ]
    press = build_press("snapkv", 0.5, window_size=8)
    pipe = hf_pipeline(BATCHED_TASK_NAME, model=model, device="cuda:0")

    def run(ctxs):
        return pipe.generate_batch(
            ctxs, question="", press=press, max_new_tokens=32,
            stop_strings=STOP_STRINGS,
        ).texts

    base = run(contexts)
    assert run(contexts) == base, "same batch run twice must give the same output"

    perm = list(range(len(contexts)))[::-1]
    out = run([contexts[i] for i in perm])
    restored = [None] * len(contexts)
    for slot, orig in enumerate(perm):
        restored[orig] = out[slot]
    assert restored == base, (
        "reversing the batch changed a row's output -- rows are contaminating each "
        "other, which invalidates every batched number"
    )
