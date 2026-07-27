"""Batched kvpress generation for GSM8K — the path that makes B=128 possible.

At ``B = 1`` a GSM8K cell costs ~18 s/example on Llama-3.1-8B (measured), so the full
1319-example split is ~6.7 h per press/ratio pair. Decode is one forward per token and
is weight-bound at that batch size, so batching is the lever.

Three things this has to get right, none of which the stock pipeline does.

1. Padding side and positions
-----------------------------
Prompts are ragged, so a batch needs padding — and it must be **left** padding.
SnapKV and both DefensiveKV variants score using "the last ``window_size`` queries"
(``hidden_states[:, -window_size:]``); right padding would make that window pure pad
tokens and destroy the scoring signal. Left padding keeps the observation window on
real text.

Left padding then requires per-row positions: ``position_ids = cumsum(mask) - 1`` so
every row's first *real* token sits at position 0 regardless of how much pad precedes
it. Without that, RoPE phase differs per row and rows are no longer comparable to a
``B=1`` run.

2. Pads must never win retention
--------------------------------
``compute_window_attention`` builds a purely causal mask with no padding term
(``snapkv_press.py:68`` and both DefensiveKV variants), so pad keys are visible to
every query. Two distinct consequences, and only one of them matters:

*   Pads absorb softmax mass, scaling real tokens' scores down. This is **harmless for
    the ranking**: the softmax denominator is shared by every key in a query row, so
    the relative order of real tokens is unchanged, and top-k is order-only.
*   Pads are themselves scored and can be *retained*, spending budget on nothing. This
    one is real, so :func:`pad_masked_scores` forces pad columns to ``-inf`` after
    scoring, which is a single patch point per press family rather than three
    reimplementations of the window attention.

Residual effect: the ``avg_pool1d(kernel_size=5)`` smoothing smears scores across the
pad/real boundary, perturbing ~2 tokens per row, and DefensiveKV's cumulative-mass
threshold normalises over a sum that includes pads. Both are bounded by the pad
fraction, which is why :mod:`gsm8k.batching` caps intra-group spread rather than using
free-form sort-and-chunk.

3. Rows finish at different times
---------------------------------
GSM8K generations run 120-400 tokens. A batch must keep stepping until its slowest row
is done, masking finished rows so they neither emit tokens nor stop early. At B=128
this costs real utilisation (~30-35%); dropping finished rows from the batch entirely
would recover it and is the obvious next optimisation, deliberately left out of this
first version so the correctness story stays simple.

Validation contract
-------------------
An earlier version of this note claimed that with ``pad_to_multiple=None`` (exact-length
groups, no padding at all) the output would be **identical to ``B=1``**. Measured on
Llama-3.1-8B, it is not, and the claim was wrong in principle: cuBLAS selects different
GEMM kernels for different batch sizes, so reduction order — and therefore bf16 rounding
— changes with ``B``. Over 32 layers that is enough to flip a greedy ``argmax`` wherever
two logits are near-tied. On 6 zero-padding GSM8K rows, ``B=6`` vs ``B=1`` differed on
3/6 with no press and 4/6 under SnapKV.

SnapKV diverges *earlier* in the text (char 18 vs char 234) because a press amplifies
this: top-k over attention scores is discrete, so a rounding difference on two near-tied
scores changes *which KV is retained*, which changes the next token. Compression turns a
rare late coin-flip into an early structural one. That is inherent to batching a
selection-based method, not a bug in this file.

What *is* checkable exactly, and what the tests assert:

* **Determinism** — the same rows at the same ``B`` produce the same output. Measured
  0/6 mismatches for both presses.
* **Permutation invariance** — reversing the row order and un-reversing the results
  changes nothing. Measured 0/6 for both presses. This is the property that matters for
  correctness: a row's output must not depend on who it is batched with. A failure here
  would mean cross-row contamination and would invalidate batching entirely.

Batch-size equivalence must therefore be judged **statistically** — accuracy over a
run, within its binomial CI — not by string equality. ``compare_batched_vs_single`` in
``test_batched_pipeline.py`` remains useful as a *diagnostic* (where and how much the
paths drift), but a non-zero mismatch count there is expected, not a failure.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY

from kvpress.presses.base_press import BasePress

from gsm8k.pipeline import GSM8KKVPressPipeline, _logits_kwargs, build_cache
from gsm8k.press_budget import measure_cache

#: Trailing tokens re-decoded each step when checking for a stop string. Every stop
#: string in use is <= 4 tokens; 16 is slack. Kept small so the check stays O(1) per
#: step instead of re-decoding the whole answer 512 times per row.
_STOP_LOOKBACK_TOKENS = 16


# ---------------------------------------------------------------------------
# Pad-aware scoring
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def pad_masked_scores(press: Optional[BasePress], valid: Optional[Tensor]):
    """Force padded positions to ``-inf`` score for the duration of a prefill.

    Wraps ``press.score`` rather than the three separate ``compute_window_attention``
    implementations: whatever the press computes, pads come out unselectable.

    Handles both score layouts kvpress uses:

    * ``[B, H_kv, T]`` — the dense ``ScorerPress`` family (snapkv, streaming_llm, ...)
    * ``[B, H_kv * T]`` — the flattened Ada family (defensivekv, layer_defensivekv)

    Parameters
    ----------
    valid : Tensor
        ``[B, T]`` bool, ``True`` for real tokens. ``None`` (or an all-true mask)
        disables the wrapper entirely, so an unpadded run takes the original code path
        and stays bit-identical to ``B=1``.
    """
    if press is None or valid is None or bool(valid.all()):
        yield
        return

    original = press.score
    neg_inf = float("-inf")

    def scored(module, hidden_states, keys, values, attentions, kwargs):
        out = original(module, hidden_states, keys, values, attentions, kwargs)
        bsz, seq_len = valid.shape
        mask = valid.to(out.device)
        if out.dim() == 3:                       # [B, H, T]
            out = out.masked_fill(~mask[:, None, :], neg_inf)
        elif out.dim() == 2:                     # [B, H*T]
            heads = out.shape[1] // seq_len
            out = out.view(bsz, heads, seq_len).masked_fill(
                ~mask[:, None, :], neg_inf
            ).view(bsz, heads * seq_len)
        return out

    press.score = scored  # type: ignore[method-assign]
    try:
        yield
    finally:
        press.score = original  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Batched generation result
# ---------------------------------------------------------------------------


@dataclass
class BatchedGeneration:
    """Per-row outputs plus the batch-level cache measurement."""

    texts: List[str]
    n_generated: List[int]
    stop_reasons: List[str]
    context_lengths: List[int]      # true (unpadded) context length per row
    padded_length: int
    question_tokens: int
    cache_measurement: Any


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class BatchedGSM8KPipeline(GSM8KKVPressPipeline):
    """kvpress generation over a batch of equal-or-similar-length prompts."""

    # -- tokenisation ------------------------------------------------------

    def encode_batch(
        self,
        contexts: Sequence[str],
        question: str,
        answer_prefix: str = "",
    ) -> Dict[str, Tensor]:
        """Left-pad a group of contexts into one batch.

        Returns ``context_ids``/``attention_mask``/``position_ids`` for the compressed
        part, plus the (shared) ``question_ids``. The question is required to be
        identical across the group — true for GSM8K, where ``compress_questions=True``
        leaves only the chat template's assistant header — so it needs no padding and
        contributes no raggedness after the prefill.
        """
        per_row = [
            self.preprocess(
                c, questions=[question], answer_prefix=answer_prefix,
                max_context_length=int(1e10),
            )
            for c in contexts
        ]
        ids = [p["context_ids"][0] for p in per_row]
        q_ids = [p["questions_ids"][0] for p in per_row]

        q_len = {int(q.shape[-1]) for q in q_ids}
        if len(q_len) != 1:
            raise ValueError(
                f"batched GSM8K requires one shared question per group, got "
                f"{len(q_len)} distinct question lengths. Use compress_questions=True."
            )

        true_lengths = [int(t.shape[-1]) for t in ids]
        width = max(true_lengths)
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        bsz = len(ids)
        context_ids = torch.full((bsz, width), pad_id, dtype=torch.long)
        mask = torch.zeros((bsz, width), dtype=torch.bool)
        for i, t in enumerate(ids):
            n = t.shape[-1]
            context_ids[i, width - n :] = t     # LEFT pad
            mask[i, width - n :] = True

        # Real tokens start at position 0 in every row, whatever precedes them.
        position_ids = (mask.long().cumsum(-1) - 1).clamp_min(0)

        return {
            "context_ids": context_ids,
            "attention_mask": mask,
            "position_ids": position_ids,
            "question_ids": q_ids[0].unsqueeze(0) if q_ids[0].dim() == 1 else q_ids[0],
            "true_lengths": torch.tensor(true_lengths),
        }

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate_batch(
        self,
        contexts: Sequence[str],
        question: str,
        answer_prefix: str = "",
        press: Optional[BasePress] = None,
        max_new_tokens: int = 512,
        stop_strings: Optional[List[str]] = None,
    ) -> BatchedGeneration:
        """Prefill a batch under *press*, then greedily decode every row."""
        stop_strings = stop_strings or []
        enc = self.encode_batch(contexts, question, answer_prefix)

        device = self.model.device
        context_ids = enc["context_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        position_ids = enc["position_ids"].to(device)
        question_ids = enc["question_ids"].to(device)
        true_lengths = enc["true_lengths"].tolist()

        bsz, width = context_ids.shape
        cache = build_cache(press)
        logits_kw = _logits_kwargs(self.model)

        # --- prefill under the press -------------------------------------
        with pad_masked_scores(press, enc["attention_mask"]):
            with press(self.model) if press is not None else contextlib.nullcontext():
                self.model(
                    input_ids=context_ids,
                    attention_mask=attn_mask.long(),
                    position_ids=position_ids,
                    past_key_values=cache,
                    output_attentions=self.output_attentions(press),
                    **logits_kw,
                )

        measurement = measure_cache(cache, width)

        # --- the shared question turn ------------------------------------
        q_len = question_ids.shape[-1]
        q_ids = question_ids.expand(bsz, q_len).contiguous()
        # Positions continue from each row's TRUE context length, so a padded row is
        # not pushed forward by its padding.
        base = torch.tensor(true_lengths, device=device).unsqueeze(1)
        q_pos = base + torch.arange(q_len, device=device).unsqueeze(0)
        q_attn = torch.cat(
            [attn_mask, torch.ones((bsz, q_len), dtype=torch.bool, device=device)], dim=1
        )

        out = self.model(
            input_ids=q_ids,
            attention_mask=q_attn.long(),
            position_ids=q_pos,
            past_key_values=cache,
            **logits_kw,
        )

        next_pos = q_pos[:, -1:] + 1
        tokens: List[List[int]] = [[] for _ in range(bsz)]
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        stop_reasons = ["max_new_tokens"] * bsz

        eos = self.model.generation_config.eos_token_id
        eos_ids = (
            [] if eos is None else (eos if isinstance(eos, list) else [eos])
        )
        eos_t = torch.tensor(eos_ids, device=device) if eos_ids else None

        next_tok = out.logits[:, -1].argmax(dim=-1)                 # [B]

        for step in range(max_new_tokens):
            for i in range(bsz):
                if not finished[i]:
                    tokens[i].append(int(next_tok[i]))

            # EOS
            if eos_t is not None:
                hit = (next_tok.unsqueeze(1) == eos_t.unsqueeze(0)).any(1) & ~finished
                for i in hit.nonzero(as_tuple=True)[0].tolist():
                    stop_reasons[i] = "eos"
                    tokens[i].pop()          # do not keep the EOS token itself
                finished |= hit

            # stop strings
            if stop_strings:
                for i in range(bsz):
                    if finished[i] or not tokens[i]:
                        continue
                    if self._tail_has_stop(tokens[i], stop_strings):
                        stop_reasons[i] = "stop_string"
                        finished[i] = True

            if bool(finished.all()) or step == max_new_tokens - 1:
                break

            q_attn = torch.cat(
                [q_attn, torch.ones((bsz, 1), dtype=torch.bool, device=device)], dim=1
            )
            out = self.model(
                input_ids=next_tok.unsqueeze(1),
                attention_mask=q_attn.long(),
                position_ids=next_pos,
                past_key_values=cache,
            )
            next_pos = next_pos + 1
            next_tok = out.logits[:, -1].argmax(dim=-1)

        texts = []
        for i in range(bsz):
            text = self.tokenizer.decode(tokens[i], skip_special_tokens=True)
            if stop_reasons[i] == "stop_string":
                text = _cut_at_stop(text, stop_strings)
            texts.append(text)

        return BatchedGeneration(
            texts=texts,
            n_generated=[len(t) for t in tokens],
            stop_reasons=stop_reasons,
            context_lengths=true_lengths,
            padded_length=width,
            question_tokens=int(q_len),
            cache_measurement=measurement,
        )

    def _tail_has_stop(self, token_ids: List[int], stop_strings: List[str]) -> bool:
        tail = token_ids[-_STOP_LOOKBACK_TOKENS:]
        text = self.tokenizer.decode(tail, skip_special_tokens=True)
        return any(s in text for s in stop_strings)


def _cut_at_stop(text: str, stop_strings: Sequence[str]) -> str:
    cut = len(text)
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


BATCHED_TASK_NAME = "gsm8k-batched-kv-press-text-generation"

PIPELINE_REGISTRY.register_pipeline(
    BATCHED_TASK_NAME,
    pipeline_class=BatchedGSM8KPipeline,
    pt_model=AutoModelForCausalLM,
)
