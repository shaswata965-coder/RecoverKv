"""kvpress pipeline for GSM8K: stop strings + per-example cache telemetry.

``KVPressTextGenerationPipeline.generate_answer`` decodes greedily and breaks **only**
on EOS. That is fine for LongBench (32-128 token answers) but wrong for GSM8K in two
ways, both of which distort the accuracy-vs-compression curve rather than merely
costing time:

*   **Wall clock.** GSM8K needs ``max_new_tokens=512`` so the answer line is always
    reachable (see ``create_huggingface_dataset.py``). Without a stop condition every
    example that fails to emit EOS runs the full 512 steps. Typical solutions finish in
    ~150.
*   **Scoring noise.** A model that answers correctly and then hallucinates a *new*
    problem ("\\n\\nQuestion: ... #### 12") leaves two answers in the string. The scorer
    trims that (``_strip_runon``), but trimming after the fact is a repair; stopping at
    the boundary removes the failure mode at the source. It matters because run-on rate
    co-varies with the KV budget, so it is a *confound*, not just noise.

This subclass therefore adds ``stop_strings`` to the decode loop, and — since it is
already overriding ``_forward`` — returns the compressed-cache measurement alongside
the answer, so the runner never has to guess what the press did.

Everything else (chat-template handling, the press context manager, the
uncompressed-position bookkeeping, restoring the cache after generation) is inherited
unchanged, so a run here is the same computation ``evaluate.py`` would do.
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, Cache, DynamicCache
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

from kvpress.ada_cache import DynamicCacheSplitHeadFlatten
from kvpress.presses.base_press import BasePress
from kvpress.presses.efficient_ada_global_scorer_press import EfficientAdaGlobalScorerPress
from kvpress.presses.efficient_ada_scorer_press import EfficientAdaScorerPress
from kvpress.pipeline import KVPressTextGenerationPipeline

from gsm8k.press_budget import measure_cache

TASK_NAME = "gsm8k-kv-press-text-generation"

#: How many trailing tokens to re-decode each step when checking for a stop string.
#: Every stop string in use is <= 4 tokens; 16 is slack. Keeping this small is what
#: stops the check from being O(n^2) over the generation.
_STOP_LOOKBACK_TOKENS = 16


def _logits_kwargs(model) -> Dict[str, Any]:
    """``{"logits_to_keep": 1}`` or the pre-4.54 spelling, or ``{}``.

    Only the last position's logits are ever used, and the vocab is ~128k wide, so
    asking for one row instead of ``q_len`` rows is worth the introspection.
    """
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    return {}


def build_cache(press: Optional[BasePress]) -> Cache:
    """Pick the cache layout the press needs (mirrors ``KVPressTextGenerationPipeline``)."""
    if isinstance(press, (EfficientAdaScorerPress, EfficientAdaGlobalScorerPress)):
        return DynamicCacheSplitHeadFlatten()
    return DynamicCache()


class GSM8KKVPressPipeline(KVPressTextGenerationPipeline):
    """kvpress text generation with stop strings and compression telemetry."""

    # -----------------------------------------------------------------
    # parameter plumbing
    # -----------------------------------------------------------------

    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[List[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        cache: Optional[Cache] = None,
        enable_thinking: bool = False,
        stop_strings: Optional[List[str]] = None,
        **kwargs,
    ):
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = super()._sanitize_parameters(
            question=question,
            questions=questions,
            answer_prefix=answer_prefix,
            press=press,
            max_new_tokens=max_new_tokens,
            max_context_length=max_context_length,
            cache=cache,
            enable_thinking=enable_thinking,
            **kwargs,
        )
        forward_kwargs["stop_strings"] = stop_strings
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def postprocess(self, model_outputs, single_question):  # noqa: ARG002
        """Pass the telemetry dict from :meth:`_forward` straight through."""
        return model_outputs

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def _forward(
        self,
        input_tensors: Dict[str, GenericTensor],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]

        if cache is None:
            cache = build_cache(press)

        logits_kw = _logits_kwargs(self.model)

        # Prefill the context under the press. This is the only moment compression
        # happens: kvpress presses are prefill-time prompt compressors, and the cache
        # is never pruned again during decode.
        with press(self.model) if press is not None else contextlib.nullcontext():
            self.model(
                input_ids=context_ids,
                past_key_values=cache,
                output_attentions=self.output_attentions(press),
                **logits_kw,
            )

        # Measure what the press actually built, before generation appends to it.
        measurement = measure_cache(cache, context_length)

        results = []
        for question_ids in input_tensors["questions_ids"]:
            results.append(
                self.generate_answer(
                    question_ids=question_ids.to(self.model.device),
                    cache=cache,
                    context_length=context_length,
                    max_new_tokens=max_new_tokens,
                    stop_strings=stop_strings,
                )
            )

        return {
            "answers": [r["text"] for r in results],
            "generations": results,
            "context_length": context_length,
            "cache_measurement": measurement,
        }

    # -----------------------------------------------------------------
    # generation
    # -----------------------------------------------------------------

    def generate_answer(  # type: ignore[override]
        self,
        question_ids: torch.Tensor,
        cache: Cache,
        context_length: int,
        max_new_tokens: int,
        stop_strings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Greedy decode with EOS **and** stop-string termination.

        Returns ``{"text", "n_generated", "stop_reason", "question_tokens"}``.
        ``stop_reason`` is one of ``eos`` / ``stop_string`` / ``max_new_tokens`` and is
        recorded per example, because "how the generation ended" is precisely the
        variable that corrupts GSM8K accuracy when it is left unobserved.
        """
        stop_strings = stop_strings or []

        # Snapshot the pre-generation cache size so the answer can be stripped
        # afterwards and the compressed context reused (inherited behaviour).
        if isinstance(cache, DynamicCacheSplitHeadFlatten):
            cache_seq_lengths = cache.metadata_list[0].head_lens[0].cpu().item()
        else:
            cache_seq_lengths = [
                cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))
            ]

        # Positions continue from the UNCOMPRESSED context length: surviving keys keep
        # the RoPE phase they were prefilled with, so the query must stay on the same
        # absolute axis. (This is why ``context_length`` is passed in rather than read
        # off the compressed cache.)
        position_ids = torch.arange(
            context_length,
            context_length + question_ids.shape[1],
            device=self.model.device,
        ).unsqueeze(0)

        logits_kw = _logits_kwargs(self.model)
        outputs = self.model(
            input_ids=question_ids.to(self.model.device),
            past_key_values=cache,
            position_ids=position_ids,
            **logits_kw,
        )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if should_stop_token_ids is None:
            should_stop_token_ids = []
        elif not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        stop_reason = "max_new_tokens"
        if generated_ids[-1].item() in should_stop_token_ids:
            stop_reason = "eos"
        elif self._hit_stop_string(generated_ids, stop_strings):
            stop_reason = "stop_string"

        if stop_reason == "max_new_tokens":
            for i in range(max_new_tokens - 1):
                outputs = self.model(
                    input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                    past_key_values=cache,
                    position_ids=position_ids + i,
                )
                new_id = outputs.logits[0, -1].argmax()
                generated_ids.append(new_id)
                if new_id.item() in should_stop_token_ids:
                    stop_reason = "eos"
                    break
                if self._hit_stop_string(generated_ids, stop_strings):
                    stop_reason = "stop_string"
                    break

        answer = self.tokenizer.decode(
            torch.stack(generated_ids), skip_special_tokens=True
        )
        if stop_reason == "stop_string":
            answer = _truncate_at_stop_string(answer, stop_strings)

        # Restore the cache to its post-compression state so it stays reusable.
        if isinstance(cache, DynamicCacheSplitHeadFlatten):
            n = cache.metadata_list[0].head_lens[0].cpu().item() - cache_seq_lengths
            cache.remove_tokens(n)
        else:
            self._remove_answer_from_cache(cache, cache_seq_lengths)

        return {
            "text": answer,
            "n_generated": len(generated_ids),
            "stop_reason": stop_reason,
            "question_tokens": int(question_ids.shape[1]),
        }

    def _hit_stop_string(
        self, generated_ids: List[torch.Tensor], stop_strings: List[str]
    ) -> bool:
        """Whether the tail of the generation contains a stop string.

        Only the last :data:`_STOP_LOOKBACK_TOKENS` tokens are decoded, so the check
        is O(1) per step rather than O(len) — the difference between a negligible cost
        and re-decoding the whole answer 512 times per example.
        """
        if not stop_strings:
            return False
        tail = torch.stack(generated_ids[-_STOP_LOOKBACK_TOKENS:])
        text = self.tokenizer.decode(tail, skip_special_tokens=True)
        return any(s in text for s in stop_strings)


def _truncate_at_stop_string(text: str, stop_strings: List[str]) -> str:
    """Cut *text* at the earliest stop string, so the run-on never reaches the scorer."""
    cut = len(text)
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


PIPELINE_REGISTRY.register_pipeline(
    TASK_NAME,
    pipeline_class=GSM8KKVPressPipeline,
    pt_model=AutoModelForCausalLM,
)
