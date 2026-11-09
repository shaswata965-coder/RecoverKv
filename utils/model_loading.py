"""One loader for every runner: a plain HF causal LM, or DeepSeek-V2 for RecoverKV.

``load_causal_lm`` is ``AutoModelForCausalLM.from_pretrained`` for every model
the repo already ran (Llama, Qwen2), byte for byte the same call. For a
``deepseek_v2`` checkpoint it additionally:

* loads with ``trust_remote_code`` at the PINNED revision
  (``models.deepseek_v2.loader.PINNED_REVISIONS``) unless one is given;
* installs the owned MLA attention forward on every layer, which is what the
  cache and the fused decode reach (``models/deepseek_v2/attention.py``).

The model type is read from ``config.json`` WITHOUT executing remote code
(``PretrainedConfig.get_config_dict``), so deciding whether to trust it never
requires trusting it first. Every runner goes through here: a knob threaded into
some runners and not others is how ``quant_budget_mode`` was once inert in three
of them (``CLAUDE.md``).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def model_type_of(name_or_path: str, revision: Optional[str] = None) -> Optional[str]:
    from transformers import PretrainedConfig
    try:
        d, _ = PretrainedConfig.get_config_dict(name_or_path, revision=revision)
    except Exception:
        return None
    return d.get("model_type")


def load_causal_lm(name_or_path: str, revision: Optional[str] = None,
                   torch_dtype=None, attn_implementation: str = "eager",
                   device_map=None, **kwargs) -> Any:
    from transformers import AutoModelForCausalLM

    if model_type_of(name_or_path, revision) == "deepseek_v2":
        from models.deepseek_v2.loader import PINNED_REVISIONS, install_recoverkv_mla
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            revision=revision or PINNED_REVISIONS.get(name_or_path),
            trust_remote_code=True, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation, device_map=device_map,
            **kwargs)
        model.eval()
        model._recoverkv_mla_handle = install_recoverkv_mla(model)
        return model
    return AutoModelForCausalLM.from_pretrained(
        name_or_path, revision=revision, torch_dtype=torch_dtype,
        attn_implementation=attn_implementation, device_map=device_map, **kwargs)


def load_tokenizer(name_or_path: str, revision: Optional[str] = None, **kwargs):
    from transformers import AutoTokenizer

    if model_type_of(name_or_path, revision) == "deepseek_v2":
        from models.deepseek_v2.loader import PINNED_REVISIONS
        revision = revision or PINNED_REVISIONS.get(name_or_path)
    return AutoTokenizer.from_pretrained(name_or_path, revision=revision, **kwargs)
