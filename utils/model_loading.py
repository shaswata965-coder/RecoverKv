"""One model/tokenizer load path, shared by every generation runner.

LongBench, RULER and GSM8K each grew their own ``_load_model_and_tokenizer``.
They agreed on the obvious parts and diverged on the parts that decide a score,
which is the worst possible split: the Qwen2.5 column was running a different
decoding protocol from the Llama and Mistral columns with nothing in the logs
saying so. This module is the single place all three now go through, so "the
three columns differ only by model" is a property of the code rather than a
comment in three YAML files.

Three things it owns.

**1. Greedy really means greedy** (:func:`pin_greedy_generation_defaults`).
Passing ``do_sample=False`` to ``generate()`` does not disable a *penalty*: any
knob the runner does not pass is taken from the checkpoint's own
``generation_config.json``. Qwen2.5-7B-Instruct ships ``repetition_penalty:
1.05``; Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.2 ship none.
``RepetitionPenaltyLogitsProcessor`` is built in ``_get_logits_processor``, not
the sampling warper, so it applies under greedy decoding — the Qwen runs were
penalised and the others were not. THUDM/LongBench, kvpress and DefensiveKV all
decode with a bare greedy search, so the checkpoint's sampling preferences are
reset here, once, loudly.

**2. Context/RoPE overrides** (:func:`apply_model_config_overrides`).
Declared in the config and applied to the ``AutoConfig`` before the weights
load, instead of hand-edited into a cached ``config.json`` where they are
invisible to the result files and unshareable.

**3. dtype is resolved, not guessed** (:func:`resolve_dtype`). The old
``dtypes.get(name, torch.float16)`` in each runner turned an unknown name into
an fp16 run silently. It now raises, and a dtype that disagrees with the
checkpoint's own ``torch_dtype`` is reported — the Qwen2.5 fp16 trap, where a
bf16-native model overflows at long context and the damage looks like a method
result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from utils.config import DTYPE_NAMES, ConfigValidationError
from utils.logger import get_logger

log = get_logger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

#: ``generate()`` knobs a checkpoint may set that would change a greedy run, and
#: the value that means "off". ``do_sample``/``temperature``/``top_p``/``top_k``
#: are inert under an explicit ``do_sample=False`` (the warper is not built), but
#: they are reset anyway so ``model.generation_config`` reads as the protocol the
#: runners actually execute — and so a future runner that forgets to pass
#: ``do_sample=False`` does not silently start sampling.
GREEDY_GENERATION_DEFAULTS: Dict[str, Any] = {
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "typical_p": 1.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "length_penalty": 1.0,
    "num_beams": 1,
    "penalty_alpha": None,
    "diversity_penalty": 0.0,
}


def resolve_dtype(name: str) -> torch.dtype:
    """``model.dtype`` name → ``torch.dtype``. Raises on anything unknown."""
    try:
        return _DTYPES[name]
    except KeyError:
        raise ConfigValidationError(
            f"model.dtype={name!r} is not one of {list(DTYPE_NAMES)}."
        ) from None


def apply_model_config_overrides(hf_config: Any, model_cfg: Any) -> List[str]:
    """Apply the declared context/RoPE overrides to a loaded HF config.

    Returns one human-readable line per applied change (empty when the config
    declares none), for the run log and the meta sidecar.

    Order matters for the sliding-window pair: ``Qwen2Config.__init__`` resolves
    ``sliding_window = sliding_window if use_sliding_window else None`` *at
    construction*, so setting the attributes afterwards is what actually lands on
    the object the attention layers read. Both are set, then the same
    disable-rule is re-applied, so a config that declares
    ``sliding_window: 131072`` with ``use_sliding_window: false`` ends up with
    ``sliding_window=None`` — matching what Qwen2.5 does with its own config.json
    rather than quietly switching banded attention on.

    ``rope_scaling`` is applied *before* anything reads it, but note what
    transformers 4.47.1 does with it: ``_compute_yarn_parameters`` derives the
    YaRN correction range from ``config.max_position_embeddings`` and **ignores
    ``original_max_position_embeddings`` entirely** (it is not in that version's
    accepted key set, so it also logs an "Unrecognized keys" warning). The key is
    still worth declaring — it is what every other YaRN consumer reads, and it
    documents the intent — but on this pinned version the scaled window is what
    shapes the frequencies. Setting ``max_position_embeddings`` alongside it is
    therefore mandatory, not cosmetic.
    """
    changes: List[str] = []

    def _set(attr: str, value: Any) -> None:
        old = getattr(hf_config, attr, None)
        setattr(hf_config, attr, value)
        changes.append(f"{attr}: {old!r} -> {value!r}")

    mpe = getattr(model_cfg, "max_position_embeddings", None)
    if mpe is not None:
        _set("max_position_embeddings", int(mpe))

    rope_scaling = getattr(model_cfg, "rope_scaling", None)
    if rope_scaling is not None:
        rs = dict(rope_scaling)
        # PretrainedConfig mirrors the legacy "type" onto "rope_type" in its
        # __init__; we are past that, so mirror it here or a legacy-spelled block
        # validates and computes as 'default' — i.e. no scaling at all.
        if "rope_type" not in rs and "type" in rs:
            rs["rope_type"] = rs["type"]
        _set("rope_scaling", rs)

    sw = getattr(model_cfg, "sliding_window", None)
    usw = getattr(model_cfg, "use_sliding_window", None)
    if sw is not None:
        _set("sliding_window", int(sw))
    if usw is not None:
        _set("use_sliding_window", bool(usw))
    if (sw is not None or usw is not None) and hasattr(hf_config, "use_sliding_window"):
        # Re-apply Qwen2Config's own disable rule to the post-override pair.
        if not getattr(hf_config, "use_sliding_window", False):
            if getattr(hf_config, "sliding_window", None) is not None:
                _set("sliding_window", None)

    if changes:
        _revalidate_rope(hf_config)

    return changes


def _revalidate_rope(hf_config: Any) -> None:
    """Re-run transformers' own RoPE validator against the patched config.

    A malformed ``rope_scaling`` should fail here, at load, rather than from
    inside the first forward pass. The entry point moved between releases
    (``rope_config_validation`` on the pinned 4.47.1; ``config.validate_rope()``
    on 5.x, where the old name is a deprecated shim that needs methods only a
    real ``PretrainedConfig`` has), so try both and treat *the validator being
    unavailable* as different from *the config being wrong*: the first is logged
    and shrugged off, the second is raised.
    """
    real_errors = (KeyError, ValueError)

    validate = getattr(hf_config, "validate_rope", None)
    if callable(validate):
        try:
            validate()
            return
        except real_errors as exc:
            raise ConfigValidationError(
                f"model.rope_scaling was rejected by transformers: {exc}"
            ) from exc
        except Exception as exc:                 # pragma: no cover - version drift
            log.debug("validate_rope() unusable (%s); trying the legacy entry point", exc)

    try:
        from transformers.modeling_rope_utils import rope_config_validation

        rope_config_validation(hf_config)
    except real_errors as exc:
        raise ConfigValidationError(
            f"model.rope_scaling was rejected by transformers: {exc}"
        ) from exc
    except Exception as exc:                     # pragma: no cover - version drift
        log.debug(
            "Could not re-validate rope_scaling on this transformers (%s). The "
            "override was still applied; the model load will surface a genuinely "
            "malformed block.", exc,
        )


def pin_greedy_generation_defaults(model: Any) -> Dict[str, Tuple[Any, Any]]:
    """Reset a checkpoint's sampling knobs so greedy decoding is bare greedy.

    Returns ``{field: (was, now)}`` for every field actually changed, so the
    caller can log it and the meta sidecar can record it. Empty dict for a
    checkpoint that ships no sampling preferences (Llama-3.1, Mistral-v0.2),
    which is what makes those two runs byte-identical to before this existed.

    Does **not** touch ``eos_token_id`` / ``pad_token_id`` / ``bos_token_id``:
    those are the model's real stopping contract (Qwen2.5 stops on both
    ``<|im_end|>`` and ``<|endoftext|>`) and overriding them is how a run starts
    generating past its own turn boundary.
    """
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return {}
    changed: Dict[str, Tuple[Any, Any]] = {}
    for field_name, target in GREEDY_GENERATION_DEFAULTS.items():
        if not hasattr(gen_cfg, field_name):
            continue
        current = getattr(gen_cfg, field_name)
        if current == target or (current is None and target is None):
            continue
        setattr(gen_cfg, field_name, target)
        changed[field_name] = (current, target)
    return changed


def load_model_and_tokenizer(
    cfg: Any,
    *,
    raw_prompt_hint: str = "",
    is_windowed: bool = False,
) -> Tuple[Any, Any]:
    """Load the model + tokenizer for a generation run. One path for all runners.

    Parameters
    ----------
    cfg
        The full ``ExperimentConfig`` (reads ``cfg.model``).
    raw_prompt_hint
        Suite-specific tail for the "no chat template" warning — what it costs
        *this* suite to prompt an instruct model raw.
    is_windowed
        Whether a compacting cache is in play, which is what makes a banded
        attention mask a correctness problem rather than a curiosity.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    m = cfg.model
    revision = getattr(m, "revision", None)
    model_dtype = resolve_dtype(m.dtype)

    log.info("Loading tokenizer: %s", m.name)
    tokenizer = AutoTokenizer.from_pretrained(m.name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_config = AutoConfig.from_pretrained(m.name, revision=revision)
    overrides = apply_model_config_overrides(hf_config, m)
    for line in overrides:
        log.warning("model config override -> %s", line)

    log.info(
        "Loading model: %s (dtype=%s, attn=%s, max_position_embeddings=%s, "
        "rope_scaling=%s)",
        m.name, m.dtype, m.attn_implementation,
        getattr(hf_config, "max_position_embeddings", None),
        getattr(hf_config, "rope_scaling", None),
    )
    model = AutoModelForCausalLM.from_pretrained(
        m.name,
        revision=revision,
        config=hf_config,
        torch_dtype=model_dtype,
        attn_implementation=m.attn_implementation,
        device_map="auto",
    )
    model.eval()

    _warn_on_dtype_mismatch(model, m)

    changed = pin_greedy_generation_defaults(model)
    if changed:
        log.warning(
            "Reset %d generation_config field(s) this checkpoint ships, so greedy "
            "decoding matches the THUDM/LongBench + kvpress protocol and the other "
            "model columns: %s. (Qwen2.5 ships repetition_penalty=1.05, which "
            "applies under greedy decoding — it is a logits PROCESSOR, not a "
            "sampling warper — while Llama-3.1 and Mistral-v0.2 ship none.)",
            len(changed),
            ", ".join(f"{k}: {was!r} -> {now!r}" for k, (was, now) in changed.items()),
        )

    if getattr(tokenizer, "chat_template", None) is None:
        log.warning(
            "Tokenizer for %s ships no chat template — prompts will be sent RAW. "
            "Correct for a base model; on an instruct model it costs most of the "
            "score.%s",
            m.name, f" {raw_prompt_hint}" if raw_prompt_hint else "",
        )

    # Mistral-7B-Instruct-v0.1 sets sliding_window=4096 (v0.2/v0.3 set null).
    # Both the FA2 kernel and the sdpa mask builder band attention by *cache-slot*
    # distance, which stops matching real token distance once the cache is
    # compacted. Qwen2.5 declares sliding_window but ships use_sliding_window=false,
    # and Qwen2Config nulls it out in that case — so this correctly stays quiet
    # there, including after the overrides above.
    sw = getattr(model.config, "sliding_window", None)
    if sw and is_windowed:
        log.warning(
            "%s has an ACTIVE sliding_window=%s. The windowed cache keeps survivors "
            "at their original positions but stores them compacted, so a banded "
            "attention mask no longer corresponds to real token distance. These "
            "numbers are not comparable with an unbanded model.",
            m.name, sw,
        )

    return model, tokenizer


def _warn_on_dtype_mismatch(model: Any, model_cfg: Any) -> None:
    """Report a run dtype that differs from the checkpoint's released dtype.

    Not an error — fp16 is the only option on a T4/P100, and it is fine on
    Llama-3.1 and Mistral-v0.2. It is a warning because on a bf16-native model
    with large activations (Qwen2.5) fp16 overflows at long context, and the
    damage shows up as empty or repeated generations that read like a bad method
    result rather than a numerics failure.
    """
    native = getattr(getattr(model, "config", None), "torch_dtype", None)
    native_name = getattr(native, "__str__", lambda: "")().replace("torch.", "")
    if not native_name or native_name == model_cfg.dtype:
        return
    log.warning(
        "model.dtype=%s but %s was released as %s. On a bf16-native model, fp16 "
        "can overflow at long context and produce empty/repeated generations. "
        "Prefer --override model.dtype=%s, applied to EVERY model column so the "
        "columns still differ only by model.",
        model_cfg.dtype, model_cfg.name, native_name, native_name,
    )


def describe_model_load(cfg: Any) -> Dict[str, Any]:
    """The load-time knobs a meta sidecar needs to be self-describing."""
    m = cfg.model
    return {
        "model_dtype": m.dtype,
        "model_max_position_embeddings_override": getattr(
            m, "max_position_embeddings", None
        ),
        "model_rope_scaling_override": getattr(m, "rope_scaling", None),
        "model_sliding_window_override": getattr(m, "sliding_window", None),
        "model_use_sliding_window_override": getattr(m, "use_sliding_window", None),
        "greedy_generation_defaults_pinned": sorted(GREEDY_GENERATION_DEFAULTS),
    }
