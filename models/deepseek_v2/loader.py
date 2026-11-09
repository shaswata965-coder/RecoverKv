"""Load DeepSeek-V2-Lite for RecoverKV: pin, install the owned forward, build the cache.

One helper for every runner, so no runner threads the MLA specifics by hand
(``CLAUDE.md``: "Read the invocation, not the config default" -- a knob threaded
into some runners and not others is how ``quant_budget_mode`` was once inert in
three of them).

* :func:`install_recoverkv_mla` replaces ``forward`` on every attention module.
  It recognises the remote classes **by name and** ``config.model_type ==
  "deepseek_v2"`` and raises on anything else: a CPU run that installed nothing
  would score nothing and look exactly like success.
* :func:`make_cache` builds the shipped ``WindowedCache`` against the latent view
  of the config (``geometry.MLACacheConfig``) and the packed RoPE adapter, so the
  cache is priced at 1,152 B/token and rotates only the 64 RoPE dims.
* :func:`load_model` is ``from_pretrained`` with ``trust_remote_code`` at a pinned
  revision, bf16, plus the install.

The pinned revisions are the ones ``docs/deepseek_v2_lite/PLAN.md`` was read
against. Moving them is a change to the reference, not a convenience.
"""

from __future__ import annotations

import types
from typing import Any, List, Optional

import torch
from torch import nn

from .attention import mla_attention_forward
from .geometry import MLAGeometry
from .rope import PackedRope

PINNED_REVISIONS = {
    "deepseek-ai/DeepSeek-V2-Lite-Chat": "85864749cd611b4353ce1decdb286193298f64c7",
    "deepseek-ai/DeepSeek-V2-Lite": "604d5664dddd88a0433dbae533b7fe9472482de0",
}

#: The remote attention classes the owned forward may replace. By name, because
#: a ``trust_remote_code`` class lives in a dynamically named module and an
#: ``isinstance`` against an import of our vendored copy would never match it.
ATTENTION_CLASS_NAMES = ("DeepseekV2Attention", "DeepseekV2FlashAttention2")


def is_deepseek_v2(model_or_config: Any) -> bool:
    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "model_type", None) == "deepseek_v2"


def attention_modules(model: nn.Module) -> List[nn.Module]:
    """Every MLA attention module, in layer order. Raises if the model is not one."""
    if not is_deepseek_v2(model):
        raise ValueError(
            "not a deepseek_v2 model (config.model_type="
            f"{getattr(getattr(model, 'config', None), 'model_type', None)!r})")
    layers = model.model.layers
    mods = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        name = type(attn).__name__
        if name not in ATTENTION_CLASS_NAMES:
            raise RuntimeError(
                f"layer {i}: self_attn is {name!r}, not one of "
                f"{ATTENTION_CLASS_NAMES}. The owned MLA forward is only defined "
                "for the pinned DeepSeek-V2 remote code; refusing to install over "
                "an attention it was not written against.")
        mods.append(attn)
    return mods


class MLAHandle:
    """Restores every replaced ``forward``. Idempotent."""

    def __init__(self, restores: list) -> None:
        self._restores = restores
        self.removed = False

    def remove(self) -> None:
        if self.removed:
            return
        for fn in self._restores:
            fn()
        self._restores.clear()
        self.removed = True


def install_recoverkv_mla(model: nn.Module, flash: Optional[bool] = None) -> MLAHandle:
    """Replace ``forward`` on every MLA attention module with the owned one.

    ``flash``: route prefill through FA2 + the Triton score kernel. Defaults to
    the model's own attention implementation (``flash_attention_2``) on CUDA.

    The per-sample half -- arming the fused decode on a cache and the
    forward-pre-hook that raises ``FusedDecodeNotReached`` within one layer if a
    layer armed the kernel and never ran it -- is ``hooks.install_score_hooks``,
    which every runner already calls per sample and which delegates here for a
    deepseek_v2 model.
    """
    mods = attention_modules(model)
    geo = MLAGeometry.from_config(model.config)
    if flash is None:
        flash = (getattr(model.config, "_attn_implementation", "eager")
                 == "flash_attention_2" and torch.cuda.is_available())
    restores = []
    for mod in mods:
        if abs(float(mod.softmax_scale) - geo.softmax_scale) > 1e-12:
            raise RuntimeError(
                f"layer {mod.layer_idx}: module softmax_scale {mod.softmax_scale} "
                f"!= the geometry's {geo.softmax_scale}; the scale is taken from "
                "the module, never from a head dim, and the two must agree.")
        orig = mod.forward
        mod._recoverkv_geometry = geo
        mod._recoverkv_flash = bool(flash)
        mod.forward = types.MethodType(mla_attention_forward, mod)

        def _restore(m=mod, o=orig):
            m.forward = o
            for a in ("_recoverkv_geometry", "_recoverkv_flash"):
                if hasattr(m, a):
                    delattr(m, a)
        restores.append(_restore)
    return MLAHandle(restores)


def make_cache(model: nn.Module, cache_config, prefill_len: int, max_tokens: int,
               telemetry=None, fused_decode: Optional[bool] = None):
    """The shipped ``WindowedCache``, built against the MLA latent.

    ``fused_decode``: arm the fused gated decode (CUDA + Triton). Defaults to on
    exactly when the model was installed with ``flash=True`` on CUDA, mirroring
    ``hooks.install_score_hooks``'s ``fused_decode_enabled() and cuda``.
    """
    from modules.windowed_cache.cache import WindowedCache

    mods = attention_modules(model)
    geo = MLAGeometry.from_config(model.config)
    rope = PackedRope(mods[0].rotary_emb, geo)
    kv_dtype = next(model.parameters()).dtype
    cache = WindowedCache(
        config=cache_config,
        prefill_len=prefill_len,
        model_config=geo.cache_config(float(mods[0].softmax_scale)),
        kv_dtype=kv_dtype,
        rope_module=rope,
        num_layers=geo.num_layers,
        max_tokens=max_tokens,
        telemetry=telemetry,
    )
    if fused_decode is None:
        fused_decode = bool(getattr(mods[0], "_recoverkv_flash", False)
                            and torch.cuda.is_available())
    if fused_decode:
        from modules.windowed_cache.decode_kernel import (
            assert_decode_kernel_available)
        from modules.windowed_cache.score_kernel import (
            assert_prefill_kernel_available)
        assert_prefill_kernel_available(True)
        assert_decode_kernel_available(True)
        cache._fused_decode_active = True
    return cache


def load_model(name_or_path: str, revision: Optional[str] = None,
               attn_implementation: str = "eager", device_map=None,
               torch_dtype=torch.bfloat16):
    """``from_pretrained`` at a pinned revision, then :func:`install_recoverkv_mla`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if revision is None:
        revision = PINNED_REVISIONS.get(name_or_path)
    tok = AutoTokenizer.from_pretrained(name_or_path, revision=revision,
                                        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, revision=revision, trust_remote_code=True,
        torch_dtype=torch_dtype, attn_implementation=attn_implementation,
        device_map=device_map)
    model.eval()
    handle = install_recoverkv_mla(model)
    return model, tok, handle
