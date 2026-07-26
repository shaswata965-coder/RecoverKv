"""End-to-end: a real `model.generate()` through both cache backends.

Every other test drives `WindowedCache.update()` directly with synthetic scores.
That leaves the actual integration — HuggingFace `generate()` → attention module
→ score hook → `cache.update()` → eviction — covered only by manual runs, and
the failure mode there is **silent**: if a hook stops receiving what it needs,
it warns once through `warnings.warn` (easily swallowed) and eviction quietly
degrades to sink+local. Nothing raises. Scores just get worse.

Two specific things this pins that nothing else does:

* **The eager backend needs `output_attentions=True` to reach the forward.**
  `generate()` prints "`output_attentions` is ignored" when
  `return_dict_in_generate` is False, which reads like the flag is dropped. It
  is not — it is dropped from the *returned* object, but still forwarded to the
  attention call. If that ever changes, the eager backend scores nothing.
* **The flash backend needs `hidden_states` + `position_embeddings` in the
  attention call signature.** It recomputes its own query and runs an auxiliary
  SDPA rather than reading the kernel's weights, so it is exercisable on CPU
  without `flash_attn` installed — and a transformers signature change would
  otherwise only surface on GPU.

The model is built from a config, not downloaded: ~1M params, no network, ~2s.
"""
from __future__ import annotations

import warnings

import pytest
import torch

transformers = pytest.importorskip("transformers")

PREFILL, GEN = 96, 24
WS, SINK, LOCAL = 8, 4, 16

#: Tokens the cache actually sees if nothing ever evicts.
#:
#: NOT ``PREFILL + GEN``. `generate()` runs one prefill forward (which yields the
#: 1st new token) and then ``GEN - 1`` decode forwards; the last token it emits
#: is never fed back, so it never reaches the cache.
UNEVICTED = PREFILL + GEN - 1


#: Architectures every case below runs on.
#:
#: Mistral is not redundant with Llama here. The two model files diverge in the
#: cache contract: Mistral's attention builds its own ``cache_kwargs`` and the
#: FA2 variant omits ``cache_position`` (transformers 4.47.1), which is what
#: `WindowedCache.update`'s position fallback exists for — window identity is
#: ``(position - num_sink) // window_size``, so a wrong position scrambles the
#: grouping eviction ranks. Running the same assertions on both is what stops a
#: per-model-file convention from silently becoming a Llama-only assumption.
ARCHITECTURES = ["llama", "mistral"]


def _tiny_model(arch="llama"):
    torch.manual_seed(0)
    # head_dim = 64/4 = 16, divisible by 4 as the int2 Q tier requires.
    kw = dict(vocab_size=256, hidden_size=64, intermediate_size=128,
              num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=512)
    if arch == "mistral":
        from transformers import MistralConfig, MistralForCausalLM

        # sliding_window=None matches Mistral-v0.2/v0.3. v0.1's 4096 bands
        # attention by CACHE SLOT, which stops matching real token distance over
        # a compacted cache — the runners warn about it at load.
        cfg = MistralConfig(sliding_window=None, **kw)
        return MistralForCausalLM(cfg).eval(), cfg

    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(**kw)
    return LlamaForCausalLM(cfg).eval(), cfg


def _find_rope(model):
    """The model's rotary embedding, found the way the runners find it.

    Not ``model.model.rotary_emb``: transformers 4.47.1 hoists that to the model
    on Llama but leaves it per-attention-layer on Mistral, where the attribute
    does not exist at all. Every runner scans ``named_modules()`` for this
    reason, so the test does too — hard-coding the Llama location here would make
    this file pass on the one architecture whose layout it assumed.
    """
    for name, mod in model.named_modules():
        if "rotary" in name.lower() or "rope" in name.lower():
            return mod
    raise AssertionError("no rotary embedding module found")


def _backend(name):
    if name == "eager":
        from modules.windowed_eager_cache.cache import WindowedCache
        from modules.windowed_eager_cache.config import WindowedCacheConfig
        from modules.windowed_eager_cache.hooks import install_score_hooks
    else:
        from modules.windowed_cache.cache import WindowedCache
        from modules.windowed_cache.config import WindowedCacheConfig
        from modules.windowed_cache.hooks import install_score_hooks
    return WindowedCache, WindowedCacheConfig, install_score_hooks


def _generate(backend, quant_ratio, first_eviction_step=None, arch="llama"):
    """Run one full generate() and report what the cache did."""
    Cache, Config, install = _backend(backend)
    model, cfg = _tiny_model(arch)

    kw = dict(window_size=WS, num_sink_tokens=SINK, local_window_size=LOCAL,
              cache_budget=0.4, quant_ratio=quant_ratio)
    if first_eviction_step is not None:
        kw["first_eviction_step"] = first_eviction_step
    cache_cfg = Config(**kw)

    cache = Cache(config=cache_cfg, prefill_len=PREFILL, model_config=cfg,
                  kv_dtype=torch.float32, rope_module=_find_rope(model),
                  num_layers=cfg.num_hidden_layers, max_tokens=GEN)
    hooks = install(model, cache, cache_cfg)

    gen_kwargs = dict(max_new_tokens=GEN, do_sample=False, num_beams=1,
                      past_key_values=cache, pad_token_id=0)
    if backend == "eager":
        # The runners set this for the eager backend only; the flash hook
        # reconstructs its own scores and does not need it.
        gen_kwargs["output_attentions"] = True

    ids = torch.randint(0, 256, (1, PREFILL))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            out = model.generate(ids, **gen_kwargs)
    hooks.remove()

    return {
        "hook_warnings": [
            str(w.message) for w in caught
            if "hook" in str(w.message).lower()
        ],
        "scores_set": [
            cache._states[i].window_scores is not None
            for i in range(cfg.num_hidden_layers)
        ],
        "n_generated": int(out.shape[-1] - PREFILL),
        # The fp store length is the signal that eviction ran. `get_seq_length`
        # is the MERGED count (T_fp + T_q) and barely moves at q > 0 — a demoted
        # window still counts as its tokens, it just costs ~4x fewer bytes — so
        # asserting on it would pass whether or not anything evicted.
        "fp_len": cache._states[0].seq_length,
        "merged_len": cache.get_seq_length(0),
        "q_windows": (
            cache._stores[0].num_active_windows
            if cache._stores[0] is not None else 0
        ),
    }


@pytest.mark.parametrize("arch", ARCHITECTURES)
@pytest.mark.parametrize("backend", ["eager", "flash"])
@pytest.mark.parametrize("quant_ratio", [0.0, 0.5])
class TestGenerateEndToEnd:
    def test_hooks_fire_and_scores_reach_the_cache(self, backend, quant_ratio, arch):
        """The silent failure: hooks installed, but nothing scored."""
        r = _generate(backend, quant_ratio, arch=arch)
        assert not r["hook_warnings"], r["hook_warnings"]
        assert all(r["scores_set"]), (
            f"{arch}/{backend} q={quant_ratio}: window_scores never populated — "
            f"eviction would silently degrade to sink+local"
        )

    def test_eviction_actually_happened(self, backend, quant_ratio, arch):
        """The fp store must end below the un-evicted length."""
        r = _generate(backend, quant_ratio, arch=arch)
        assert r["n_generated"] == GEN
        assert r["fp_len"] < UNEVICTED, (
            f"{arch}/{backend} q={quant_ratio}: fp store reached {r['fp_len']} "
            f"tokens, the un-evicted length ({UNEVICTED}) — nothing was evicted"
        )

    def test_q_tier_is_populated_iff_enabled(self, backend, quant_ratio, arch):
        r = _generate(backend, quant_ratio, arch=arch)
        if quant_ratio > 0.0:
            # Windows only enter the Q store by being DEMOTED at an eviction, so
            # this is independent evidence that the two-tier path ran.
            assert r["q_windows"] > 0, "Q tier enabled but no window landed in it"
        else:
            assert r["q_windows"] == 0, "Q tier populated at quant_ratio=0"

    def test_the_q_tier_is_why_merged_length_barely_moves(
        self, backend, quant_ratio, arch
    ):
        """Guards the reading of the numbers, not just the numbers.

        At q > 0 the merged count stays near the un-evicted length while the fp
        store collapses: that gap IS the Q tier holding windows a single-tier
        cache would have dropped. Anyone asserting compression on
        `get_seq_length` would conclude nothing happened.
        """
        r = _generate(backend, quant_ratio, arch=arch)
        assert r["merged_len"] - r["fp_len"] == (
            r["q_windows"] * WS
        ), "merged length must be exactly T_fp + T_q"
        if quant_ratio == 0.0:
            assert r["merged_len"] == r["fp_len"]


@pytest.mark.parametrize("arch", ARCHITECTURES)
@pytest.mark.parametrize("quant_ratio", [0.0, 0.5])
def test_both_backends_agree(quant_ratio, arch):
    """They share cache/policy/state/scorer byte for byte and differ only in how
    they obtain attention weights, so the cache trajectory must match."""
    eager = _generate("eager", quant_ratio, arch=arch)
    flash = _generate("flash", quant_ratio, arch=arch)
    assert eager["fp_len"] == flash["fp_len"]
    assert eager["merged_len"] == flash["merged_len"]
    assert eager["q_windows"] == flash["q_windows"]
    assert eager["n_generated"] == flash["n_generated"]


@pytest.mark.parametrize("arch", ARCHITECTURES)
@pytest.mark.parametrize("quant_ratio", [0.0, 0.5])
def test_both_architectures_take_the_same_cache_trajectory(quant_ratio, arch):
    """Llama and Mistral must compact identically.

    Same geometry, same budget, same schedule — the cache is model-agnostic, so
    any divergence means a model file's cache contract leaked into the policy.
    That is exactly what the missing-`cache_position` bug was: positions taken
    from the compacted length instead of the real ones, so Mistral's windows
    stopped matching Llama's.
    """
    llama = _generate("eager", quant_ratio, arch="llama")
    other = _generate("eager", quant_ratio, arch=arch)
    assert llama["fp_len"] == other["fp_len"]
    assert llama["merged_len"] == other["merged_len"]
    assert llama["q_windows"] == other["q_windows"]


@pytest.mark.parametrize("arch", ARCHITECTURES)
@pytest.mark.parametrize("backend", ["eager", "flash"])
@pytest.mark.parametrize("quant_ratio", [0.0, 0.5])
def test_step_zero_compacts_and_a_delayed_step_does_not(backend, quant_ratio, arch):
    """The default schedule, observed through generate() rather than asserted on
    should_evict — and with a control that proves the comparison has teeth."""
    step0 = _generate(backend, quant_ratio, first_eviction_step=0, arch=arch)
    never = _generate(backend, quant_ratio, first_eviction_step=GEN + 1, arch=arch)

    assert never["fp_len"] == UNEVICTED, (
        f"control should never evict, but fp store is {never['fp_len']}"
    )
    assert never["q_windows"] == 0, "control demoted windows without evicting"
    assert step0["fp_len"] < never["fp_len"]
