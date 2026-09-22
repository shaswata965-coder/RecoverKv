"""Mistral-7B-Instruct-v0.2 port onto the clustered-rep architecture.

Two things this branch has to get right, and one it must not disturb:

* the shipped Mistral configs sit at exactly the same operating point as the
  Llama ones — same budget, window, two-tier split and read gate — so the two
  model columns differ only by the model (``TestMistralConfigsAtOperatingPoint``);
* the score hooks target ``MistralAttention``, which is not a subclass of
  ``LlamaAttention`` on transformers 4.47.x, so it has to be listed explicitly
  or Mistral runs with no window scores (``TestMistralHooksWiring``);
* the ``cache_position`` derivation that makes Mistral's flash path correct is a
  no-op wherever the caller passes ``cache_position`` — the Llama columns stay
  byte-identical (``TestResolveCachePosition``).

The config tests are torch-free and run everywhere. The hook and cache tests
need torch/transformers and skip when they are absent (they run in CI).
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from utils.config import OPERATING_POINT, OPERATING_POINT_EXEMPT, load_config

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
MISTRAL_MAX_LENGTH = 31500

OURS_CONFIGS = [
    "longbench_mistral_ours_flash_attn.yaml",
    "longbench_mistral_ours_eager.yaml",
]
ALL_CONFIGS = ["longbench_mistral_full_cache.yaml", *OURS_CONFIGS]


# ---------------------------------------------------------------------------
# Configs — torch-free
# ---------------------------------------------------------------------------


class TestMistralConfigsExist:
    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_file_exists(self, name):
        assert (CONFIGS / name).is_file(), f"missing config {name}"

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_loads_through_the_real_loader(self, name):
        # Resolves _base_ inheritance and runs every dataclass validator — the
        # same path main.py takes — so a bad field fails here, not at model load.
        cfg = load_config(str(CONFIGS / name))
        assert cfg.model.name == MISTRAL_MODEL
        assert cfg.model.dtype == "float16"

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_max_length_pinned_for_32k_context(self, name):
        # Mistral-v0.2 is a 32K model; inheriting Llama-3.1's null max_length
        # would feed prompts past max_position_embeddings, where scores are noise.
        cfg = load_config(str(CONFIGS / name))
        assert cfg.longbench.max_length == MISTRAL_MAX_LENGTH

    def test_v0_2_not_v0_1(self):
        # v0.1 ships sliding_window=4096, which bands attention by cache-slot
        # distance once the windowed cache compacts. Every shipped config must
        # name v0.2.
        for name in ALL_CONFIGS:
            text = (CONFIGS / name).read_text()
            assert "v0.2" in text
            assert "v0.1" not in yaml.safe_load(text)["model"]["name"]


class TestMistralConfigsAtOperatingPoint:
    """Gating and cards are the default: the ours configs resolve the shipped
    two-tier gated cache, identical to the Llama columns."""

    @pytest.mark.parametrize("name", OURS_CONFIGS)
    def test_every_operating_point_field_matches(self, name):
        block = yaml.safe_load((CONFIGS / name).read_text())["cache"]
        for field, want in OPERATING_POINT.items():
            if want is None:  # a "derive it" tri-state; not a concrete value
                assert block.get(field) is None or block.get(field) == want
                continue
            assert block.get(field) is not None, f"{name} does not set {field}"
            assert block[field] == want, (
                f"{name}: {field}={block[field]!r}, operating point is {want!r}")

    @pytest.mark.parametrize("name", OURS_CONFIGS)
    def test_gate_and_two_tier_are_on(self, name):
        # The gate IS the read path wherever there is a Q tier (quant_ratio>0),
        # and quant_gate_ratio is its selectivity. Both being set to the shipped
        # values is what "gating and cards are the default" means for these runs.
        block = yaml.safe_load((CONFIGS / name).read_text())["cache"]
        assert block["quant_ratio"] > 0.0            # two-tier: fp16 + int2 (cards)
        assert block["quant_ratio"] == 0.70
        assert block["quant_gate_ratio"] == 0.25     # the rank-1 read gate
        assert block["quant_budget_mode"] == "bytes"

    @pytest.mark.parametrize("name", OURS_CONFIGS)
    def test_gate_and_two_tier_survive_the_loader(self, name):
        # Written into the YAML AND surviving dataclass resolution are two
        # different things; check the resolved CacheConfig, not just the text.
        cfg = load_config(str(CONFIGS / name))
        assert cfg.cache.quant_ratio == 0.70
        assert cfg.cache.quant_gate_ratio == 0.25

    def test_backends_pair_with_attn_implementation(self):
        flash = load_config(str(CONFIGS / "longbench_mistral_ours_flash_attn.yaml"))
        assert flash.cache.backend_package == "flash_attn"
        assert flash.model.attn_implementation == "flash_attention_2"
        eager = load_config(str(CONFIGS / "longbench_mistral_ours_eager.yaml"))
        assert eager.cache.backend_package == "eager"
        assert eager.model.attn_implementation == "eager"


class TestMistralBaselineExempt:
    def test_full_cache_is_exempt(self):
        # The FullKV baseline has no eviction, so it cannot be "at the operating
        # point"; it must be recorded as an intentional exemption, not flagged.
        assert "longbench_mistral_full_cache.yaml" in OPERATING_POINT_EXEMPT

    @pytest.mark.parametrize("name", OURS_CONFIGS)
    def test_ours_configs_are_not_exempt(self, name):
        # The method configs are held TO the operating point, never exempted.
        assert name not in OPERATING_POINT_EXEMPT

    def test_full_cache_does_not_evict(self):
        cfg = load_config(str(CONFIGS / "longbench_mistral_full_cache.yaml"))
        assert cfg.cache.backend == "dynamic"
        assert cfg.cache.cache_budget is None


# ---------------------------------------------------------------------------
# Score hooks — needs torch + transformers
# ---------------------------------------------------------------------------


class TestMistralHooksWiring:
    def test_mistral_attention_is_targeted(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("transformers.models.mistral.modeling_mistral")
        from transformers.models.mistral.modeling_mistral import MistralAttention
        from modules.windowed_cache.hooks import _get_attn_classes

        assert MistralAttention in _get_attn_classes(), (
            "MistralAttention is not in the score-hook target classes; Mistral "
            "layers would get no hooks and eviction would run unscored.")


# ---------------------------------------------------------------------------
# cache_position derivation — needs torch
# ---------------------------------------------------------------------------


class TestResolveCachePosition:
    """The unit-level contract of the fix; the integration is covered by
    tests/test_windowed_cache.py::TestMissingCachePosition."""

    def _make_cache(self):
        torch = pytest.importorskip("torch")
        from modules.windowed_cache.cache import WindowedCache
        from modules.windowed_cache.config import WindowedCacheConfig
        from tests.test_windowed_cache import _FakeModelConfig

        cfg = WindowedCacheConfig(
            window_size=1, num_sink_tokens=0, local_window_size=1,
            cache_budget=0.375,
        )
        return WindowedCache(
            config=cfg, prefill_len=8, model_config=_FakeModelConfig(),
            kv_dtype=torch.float32, rope_module=torch.nn.Identity(),
            num_layers=1, max_tokens=0,
        )

    def test_supplied_cache_position_is_returned_unchanged(self):
        torch = pytest.importorskip("torch")
        cache = self._make_cache()
        want = torch.arange(3, 7)
        got = cache._resolve_cache_position(
            {"cache_position": want}, n_new=4, device=torch.device("cpu"))
        assert got is want  # identity: a Llama run is untouched

    def test_absent_cache_position_is_derived_from_token_count(self):
        torch = pytest.importorskip("torch")
        cache = self._make_cache()
        cache._tokens_seen = 12  # as update() sets it at layer 0, before dispatch
        got = cache._resolve_cache_position({"sin": 0, "cos": 0}, n_new=1,
                                            device=torch.device("cpu"))
        assert got.tolist() == [11]  # absolute, not the compacted store length

    def test_none_cache_kwargs_is_derived(self):
        torch = pytest.importorskip("torch")
        cache = self._make_cache()
        cache._tokens_seen = 5
        got = cache._resolve_cache_position(None, n_new=2,
                                            device=torch.device("cpu"))
        assert got.tolist() == [3, 4]
