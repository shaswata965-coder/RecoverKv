"""Prompt tokenization — one BOS, not two.

``tokenizer.apply_chat_template(..., tokenize=False)`` returns a **string that
already contains the BOS token**: ``<|begin_of_text|>`` for Llama-3, ``<s>`` for
Mistral. Feeding that string back through ``tokenizer(prompt)`` — whose
``add_special_tokens`` defaults to ``True`` — prepends a *second* one:

    >>> s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    >>> tok(s).input_ids[:4]
    [128000, 128000, 128006, 882]     # Llama-3.1-8B-Instruct
    [1, 1, 733, 16289]                # Mistral-7B-Instruct-v0.2

Both models, and every chat-templated prompt in the repo. It is quiet — nothing
errors, and the model still answers — but it is not free:

* every position shifts by one, so the prompt this cache scores is not the
  prompt the reference implementations score; and
* ``num_sink_tokens`` protects the first N tokens, so one of the protected sink
  slots is spent on a duplicate BOS.

That second point matters most for the comparison: THUDM/LongBench, kvpress and
DefensiveKV all tokenize the templated string **without** re-adding specials, so
published baselines are measured on a prompt this repo was not reproducing.

:func:`encode_prompt` decides from the string itself rather than from a flag at
each call site, so it stays right for a raw (non-templated) prompt — LongBench's
few-shot datasets deliberately skip the chat template, and those DO need the
tokenizer to supply the BOS.
"""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

_warned_duplicate = [False]


def _first_row(input_ids: Any) -> list:
    """The first sequence of token ids, whatever container it arrived in.

    ``input_ids`` is a flat list of ints for a single un-tensorized string, a
    ``[1, T]`` tensor under ``return_tensors="pt"``, and a list of lists for a
    batch — so unwrap until the elements are scalars rather than guessing from
    the type.
    """
    seq = input_ids
    for _ in range(2):                      # at most batch -> sequence
        if len(seq) == 0:
            return []
        head = seq[0]
        if isinstance(head, (list, tuple)) or getattr(head, "ndim", 0) > 0:
            seq = head
        else:
            break
    return [int(x) for x in seq[:2]]


def tokenizer_has_bos(tokenizer: Any) -> bool:
    """Whether this tokenizer has a BOS token at all.

    Qwen2/Qwen2.5 do not: the prompt opens with ``<|im_start|>``, an ordinary
    token, and ``bos_token`` is ``None``. Llama-3 and Mistral both do.
    """
    return bool(getattr(tokenizer, "bos_token", None))


def prompt_carries_bos(tokenizer: Any, prompt: str) -> bool:
    """Whether *prompt* already opens with the tokenizer's BOS token."""
    bos = getattr(tokenizer, "bos_token", None)
    return bool(bos) and prompt.startswith(bos)


def encode_prompt(tokenizer: Any, prompt: str, **kwargs: Any):
    """Tokenize *prompt*, adding a BOS only if it does not already carry one.

    A drop-in replacement for ``tokenizer(prompt, **kwargs)`` at any call site
    whose input may or may not have come out of ``apply_chat_template``.

    The rule is ``add_special_tokens = the tokenizer has a BOS AND the prompt
    does not already start with it``, which covers all four cases:

    ========================  ================  ==================================
    tokenizer / prompt        add_special       why
    ========================  ================  ==================================
    Llama/Mistral, templated  False             the template emitted the BOS
    Llama/Mistral, raw         True             few-shot prompt; tokenizer supplies it
    Qwen2, templated          False             no BOS exists to add or duplicate
    Qwen2, raw                False             ditto
    ========================  ================  ==================================

    The ``has BOS`` conjunct is what makes the Qwen rows False, and that is not
    cosmetic: it is what kvpress (and therefore DefensiveKV) does. It encodes
    with ``add_special_tokens=False`` unconditionally and prepends ``bos_token``
    by hand only when there is no chat template — which for a tokenizer with no
    BOS prepends nothing. Deciding False here rather than relying on "a Qwen
    tokenizer probably adds nothing anyway" removes the assumption instead of
    betting on it, which is the same mistake that produced the doubled BOS.

    Llama and Mistral are unaffected — for them ``has BOS`` is True, so the rule
    reduces to the previous one exactly.

    ``kwargs`` are forwarded verbatim (``return_tensors``, ``truncation``, …).
    Passing ``add_special_tokens`` explicitly overrides the decision, so a caller
    that genuinely knows better still can.

    The result is checked for a leading duplicate BOS and warns once if one
    survives — a tokenizer whose template embeds the BOS in some form this
    prefix test does not recognise would otherwise reintroduce the bug silently,
    which is exactly how it lasted this long.
    """
    if "add_special_tokens" not in kwargs:
        kwargs["add_special_tokens"] = (
            tokenizer_has_bos(tokenizer) and not prompt_carries_bos(tokenizer, prompt)
        )

    encoded = tokenizer(prompt, **kwargs)

    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and not _warned_duplicate[0]:
        row = _first_row(encoded["input_ids"])
        if len(row) >= 2 and row[0] == bos_id and row[1] == bos_id:
            _warned_duplicate[0] = True
            log.warning(
                "Prompt tokenized to a DUPLICATE leading BOS (id=%s). The chat "
                "template embeds a BOS that utils.prompting.prompt_carries_bos "
                "did not recognise, so the tokenizer added another. Every "
                "position is shifted by one and a sink slot is wasted on it; "
                "results will not match the reference implementations.",
                bos_id,
            )
    return encoded
