"""MiKV through the runners: ``cache.backend: external`` end to end, on CPU.

What this pins is the failure mode this repository keeps meeting — a knob or a
method that is configured, loads, runs, and never reaches the cache. So:

* the config refuses an external backend without a factory, and refuses a
  ``cache_budget`` beside it (it could reach nothing);
* the runners build the method, hand each example a fresh cache WITH hooks, and
  the sidecar says ``cache_type: external`` with what was measured;
* the operating-point log line names the method instead of "FULL CACHE".
"""

from __future__ import annotations

import json
import logging

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM  # noqa: E402

from modules.mikv import MiKVCache  # noqa: E402
from utils.cache_factory import ExternalCacheMethod  # noqa: E402
from utils.config import CacheConfig, ConfigValidationError, load_config  # noqa: E402

MIKV = "modules.mikv:make_mikv_method"


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    # head_dim 64: a realistic grid overhead (groups of 32), so the shipped
    # configs' budgets are above MiKV's floor here as they are on Llama.
    cfg = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=256,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=1024)
    return LlamaForCausalLM._from_config(cfg, attn_implementation="sdpa").eval()


class _ByteTokenizer:
    """Just enough tokenizer for a runner's ``_predict``: bytes in, bytes out."""

    bos_token = None
    bos_token_id = None
    eos_token_id = 1
    pad_token_id = 0
    chat_template = None

    def apply_chat_template(self, *a, **k):
        raise ValueError("no chat template")  # the runners fall back to a string

    def __call__(self, text, add_special_tokens=True, return_tensors="pt", **_):
        ids = torch.tensor([[2 + b % 120 for b in text.encode()][-96:]])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def encode(self, text, add_special_tokens=False):
        return [2 + b % 120 for b in text.encode()]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(32 + int(i) % 90) for i in ids)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_external_backend_needs_a_factory_and_refuses_cache_budget():
    with pytest.raises(ConfigValidationError, match="method_factory"):
        CacheConfig(backend="external")
    with pytest.raises(ConfigValidationError, match="does not reach"):
        CacheConfig(backend="external", method_factory=MIKV, cache_budget=0.2)
    with pytest.raises(ConfigValidationError, match="only apply to 'external'"):
        CacheConfig(backend="dynamic", method_kwargs={"importance_ratio": 0.2})
    with pytest.raises(ConfigValidationError, match="cache.backend must be"):
        CacheConfig(backend="mikv")
    ok = CacheConfig(backend="external", method_factory=MIKV,
                     method_kwargs={"importance_ratio": 0.2})
    assert ok.method_kwargs == {"importance_ratio": 0.2}


@pytest.mark.parametrize("name", ["gsm8k_mikv", "longbench_mikv", "eval_perf_mikv"])
def test_shipped_mikv_configs_load_and_build(name):
    cfg = load_config(f"configs/{name}.yaml")
    model = _tiny_llama()
    if cfg.run.mode == "perf":
        rows = [c for c in cfg.perf.configs if c.get("cache_backend") == "external"]
        assert rows and all(r["method_factory"] == MIKV for r in rows)
        return
    assert cfg.cache.backend == "external" and cfg.cache.method_factory == MIKV
    method = ExternalCacheMethod(cfg.cache, model, None, torch.float32)
    cache, hooks = method.new_cache(max_new_tokens=8)
    try:
        assert isinstance(cache, MiKVCache) and cache.hooks_installed
        assert cache.max_new_tokens == 8
    finally:
        hooks.remove()


def test_operating_point_log_names_the_external_method(caplog):
    from utils.config import log_operating_point

    cfg = load_config("configs/gsm8k_mikv.yaml")
    with caplog.at_level(logging.INFO):
        log_operating_point(cfg, is_windowed=False)
    text = caplog.text
    assert "EXTERNAL method modules.mikv:make_mikv_method" in text
    assert "FULL CACHE" not in text


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runner_path,config", [
    ("modules.evaluation.gsm8k_runner.GSM8KRunner", "gsm8k_mikv"),
    ("modules.evaluation.longbench_runner.LongBenchRunner", "longbench_mikv"),
])
def test_runners_select_the_external_path(runner_path, config):
    import importlib

    mod, cls = runner_path.rsplit(".", 1)
    runner = getattr(importlib.import_module(mod), cls)(load_config(f"configs/{config}.yaml"))
    assert runner.is_external is True
    assert runner.is_windowed is False


def test_ruler_runner_selects_the_external_path():
    from modules.evaluation.ruler_runner import RulerRunner

    cfg = load_config("configs/ruler_niah_mk3_omega16.yaml")
    cfg.cache = CacheConfig(backend="external", method_factory=MIKV,
                            method_kwargs={"importance_ratio": 0.2})
    runner = RulerRunner(cfg)
    assert runner.is_external is True and runner.is_windowed is False


def test_gsm8k_predict_runs_mikv_and_the_sidecar_says_so(tmp_path):
    from modules.evaluation.gsm8k_runner import GSM8KRunner

    cfg = load_config("configs/gsm8k_mikv.yaml")
    cfg.gsm8k.max_new_tokens = 6
    runner = GSM8KRunner(cfg)
    runner.model = _tiny_llama()
    runner.tokenizer = _ByteTokenizer()
    runner._stop_strings_supported = False
    runner.external = ExternalCacheMethod(cfg.cache, runner.model, runner.tokenizer,
                                          runner.model.dtype)
    ex = {"context": "Q: Tom has 3 apples and buys 4 more. ", "question": "How many?",
          "answer_prefix": "", "max_new_tokens": 512, "answer": "7", "task": "gsm8k"}
    pred, stats = runner._predict(ex)
    assert isinstance(pred, str) and stats["gen_tokens"] == 6
    # 20% exact + 80% at INT2 (+ grid), measured against this fp32 model's dense cache.
    assert 0.0 < stats["kv_fraction"] < 0.4
    # Hooks are gone after the example: a second example gets a fresh cache.
    n_hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
                  for m in runner.model.modules())
    assert n_hooks == 0
    runner._predict(ex)

    runner._write_meta({"dataset_sha": "x", "prompt_style": "cot",
                        "max_new_tokens": 512}, 2, 0, 0.0, 1.0, 2.0, tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["cache_type"] == "external"
    ext = meta["external_method"]
    assert ext["method_factory"] == MIKV
    assert ext["n_memory_reports"] == 2
    assert ext["metadata"]["resolved"]["importance_ratio"] == pytest.approx(0.2)
    assert meta["read_gate"]["verdict"] == "not-expected"
    assert meta["compression_active"] is True
