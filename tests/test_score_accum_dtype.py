"""Window scores accumulate in fp32 on a bf16 cache (Qwen2.5), and only there.

``state.window_scores`` is a running sum -- every prefill query row, then every
decode step -- and it takes the dtype of the first scores handed to it, which the
score hook produced in the KV dtype. ``int2_qwen`` (``446a7eb``) measured what
that costs: ~76% of per-step contributions round away in bf16 (8-bit mantissa)
against ~52% in fp16, and the Qwen column -- the only bf16 one -- ranked its
windows on a sum the others did not. The Qwen port deferred that fix because
widening every model moves the Llama/Mistral numbers. This keeps fp16 exactly as
it was and widens only the dtype that is actually lossy.

The hook-level test drives the real ``install_score_hooks`` on CPU. Two things
stand in for a GPU: the flash L-capture (it patches ``flash_attn``, which is not
installed here) is replaced by "no L", and every forward is ONE token, because
multi-row scoring is Triton-only while the ``T == 1`` path is plain PyTorch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from modules.windowed_cache import hooks  # noqa: E402
from modules.windowed_cache.hooks import score_accum_dtype  # noqa: E402


def test_bf16_accumulates_in_fp32_and_fp16_is_unchanged():
    assert score_accum_dtype(torch.bfloat16) == torch.float32
    assert score_accum_dtype(torch.float16) == torch.float16      # Llama, Mistral
    assert score_accum_dtype(torch.float32) == torch.float32


def test_why_bf16_is_not_good_enough_for_a_running_sum():
    """The arithmetic behind the choice, on a realistic magnitude: a window that
    collected ~40 units of prefill attention stops hearing decode steps."""
    total = torch.tensor(40.0)
    step = torch.tensor(0.05)
    assert (total.bfloat16() + step.bfloat16()).float() == total   # swallowed
    assert (total.half() + step.half()).float() != total           # fp16 keeps it
    assert (total + step) != total


def _run_hooked(model_cls, cfg_cls, dtype, monkeypatch, n_tokens=24):
    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig

    # No flash_attn here, so no L-capture to install; the T == 1 score path
    # never reads L anyway.
    monkeypatch.setattr(hooks, "_install_lse_source", lambda handles: (False, "off"))
    torch.manual_seed(0)
    mcfg = cfg_cls(hidden_size=256, num_attention_heads=2, num_key_value_heads=1,
                   num_hidden_layers=1, intermediate_size=64, vocab_size=64,
                   max_position_embeddings=512)
    mcfg._attn_implementation = "eager"
    model = model_cls(mcfg).to(dtype).eval()
    ccfg = WindowedCacheConfig(window_size=8, num_sink_tokens=4,
                               local_window_size=16, cache_budget=0.5,
                               first_eviction_step=10_000)
    cache = WindowedCache(config=ccfg, prefill_len=n_tokens, model_config=mcfg,
                          kv_dtype=dtype, rope_module=model.model.rotary_emb,
                          num_layers=1, max_tokens=8)
    handles = hooks.install_score_hooks(model, cache, ccfg)
    try:
        with torch.no_grad():
            for t in range(n_tokens):
                model(torch.tensor([[t % 64]]), past_key_values=cache,
                      use_cache=True, cache_position=torch.tensor([t]))
    finally:
        handles.remove()
    return cache._states[0].window_scores


def test_the_score_hook_accumulates_a_bf16_qwen_cache_in_fp32(monkeypatch):
    ws = _run_hooked(transformers.Qwen2ForCausalLM, transformers.Qwen2Config,
                     torch.bfloat16, monkeypatch)
    assert ws is not None and ws.shape[-1] > 0
    assert ws.dtype == torch.float32


def test_the_score_hook_leaves_an_fp16_cache_in_fp16(monkeypatch):
    ws = _run_hooked(transformers.LlamaForCausalLM, transformers.LlamaConfig,
                     torch.float16, monkeypatch)
    assert ws is not None and ws.shape[-1] > 0
    assert ws.dtype == torch.float16
