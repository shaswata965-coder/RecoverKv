"""Generation-safety helpers shared by the three generation runners.

Currently one concern: models whose ``config.vocab_size`` exceeds the number of
ids their tokenizer can map.
"""

from __future__ import annotations

from typing import Any, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

_warned: set = set()


def unmappable_token_ids(model: Any, tokenizer: Any) -> Optional[List[int]]:
    """Ids the model can emit but the tokenizer cannot map, for ``suppress_tokens``.

    Qwen2.5 pads its vocabulary for tensor-core alignment: ``config.vocab_size``
    is 152064 while ``len(tokenizer)`` is 151665, leaving **399 ids that decode to
    the empty string**. Llama-3.1 and Mistral-v0.2 have no such gap, so this
    returns ``None`` for them and generation is untouched.

    Two reasons to suppress them rather than leave it:

    * They are not valid outputs. ``tokenizer.decode([151700]) == ""`` — an id in
      the gap contributes nothing to the prediction being scored.
    * ``generate(stop_strings=...)`` **crashes** on one.
      ``StopStringCriteria`` sizes its internal ``embedding_vec`` by
      ``len(tokenizer)`` and indexes it with the generated ids, so a padded id
      raises ``IndexError: index out of range in self`` from inside
      ``F.embedding`` — a hard failure mid-run, far from its cause. GSM8K passes
      ``stop_strings``, so this is reachable there today.

    Scope, stated honestly: under greedy decoding with TRAINED weights this is
    effectively unreachable, because the padded rows are never updated during
    training and their logits stay near zero while real tokens reach double
    digits. It shows up immediately with randomly-initialised weights, which is
    what the CPU smoke path uses. So this is about keeping that path usable and
    removing a crash class — not about changing any real result. It cannot change
    one either: for a trained model these ids never win, and for Llama/Mistral the
    list is empty and no logits processor is added at all.
    """
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is None or tokenizer is None:
        return None
    try:
        n_mappable = len(tokenizer)
    except TypeError:
        return None
    if vocab_size <= n_mappable:
        return None

    key = (id(model), vocab_size, n_mappable)
    if key not in _warned:
        _warned.add(key)
        log.info(
            "model.config.vocab_size=%d exceeds len(tokenizer)=%d: suppressing the "
            "%d padded ids, which decode to '' and would crash "
            "generate(stop_strings=...) via StopStringCriteria. Expected on "
            "Qwen2.5 (vocab padded for tensor-core alignment); no effect on a "
            "model without a gap.",
            vocab_size, n_mappable, vocab_size - n_mappable,
        )
    return list(range(n_mappable, vocab_size))
