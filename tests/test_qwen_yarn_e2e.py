"""End-to-end: a YaRN-scaled Qwen2 through the two-tier windowed cache.

``tests/test_qwen_protocol.py`` pins the RoPE inversion against a hand-built
rotary module. This file runs the real thing — a ``Qwen2ForCausalLM`` carrying
the eval configs' own ``rope_scaling`` block — all the way through
``generate()`` with eviction and the int4 tier live, because the failure mode
being guarded against is silent: a mis-inverted RoPE does not raise, it just
makes every dequantized key the wrong magnitude.

Skipped rather than failed on an unsupported transformers: the windowed cache
requires monotonic ``cache_position`` (<= 4.47.1, see ``utils.cache_factory``),
and a red test on a dev box with a newer transformers is noise, not signal.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import transformers  # noqa: E402

from utils.cache_factory import is_transformers_version_supported  # noqa: E402

pytestmark = pytest.mark.skipif(
    not is_transformers_version_supported(transformers.__version__),
    reason=(
        f"windowed cache requires transformers <= 4.47.1 (monotonic "
        f"cache_position); installed {transformers.__version__}"
    ),
)

#: The eval configs' block, verbatim.
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 512}
EXPECTED_SCALING = 0.1 * math.log(4.0) + 1.0        # 1.13862...

PREFILL, GEN = 96, 12


def _tiny_qwen(rope_scaling=None, max_pos=512, attn="eager"):
    """A Qwen2 with the real architecture's shape ratios, small enough for CPU.

    ``attn`` is set explicitly and defaults to ``eager`` — what the eager configs
    declare. Leaving it to ``Qwen2Config``'s default picks **sdpa**, which under
    ``output_attentions=True`` takes a degraded fallback path that changes what
    the cache receives (see
    :func:`test_qwen2_sdpa_drops_cache_position_under_output_attentions`).
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=max_pos, rope_theta=1000000.0,
        use_sliding_window=False, sliding_window=131072,
        rope_scaling=rope_scaling,
    )
    cfg._attn_implementation = attn
    return Qwen2ForCausalLM(cfg).eval(), cfg


def _find_rope(model):
    """The rotary module the way every runner finds it."""
    for name, mod in model.named_modules():
        if "rotary" in name.lower() or "rope" in name.lower():
            return mod
    raise AssertionError("no rotary embedding module found")


def _run(model, cfg, quant_ratio):
    from modules.windowed_eager_cache.cache import WindowedCache
    from modules.windowed_eager_cache.config import WindowedCacheConfig
    from modules.windowed_eager_cache.hooks import install_score_hooks

    cache_cfg = WindowedCacheConfig(
        window_size=8, num_sink_tokens=4, local_window_size=0.25,
        cache_budget=0.30, quant_ratio=quant_ratio, first_eviction_step=0,
    )
    ids = torch.arange(PREFILL, dtype=torch.long).remainder(200).unsqueeze(0)
    cache = WindowedCache(
        config=cache_cfg, prefill_len=PREFILL, model_config=cfg,
        kv_dtype=torch.float32, rope_module=_find_rope(model),
        num_layers=cfg.num_hidden_layers, max_tokens=GEN,
    )
    hooks = install_score_hooks(model, cache, cache_cfg)
    try:
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=GEN, do_sample=False, num_beams=1,
                past_key_values=cache, output_attentions=True,
                attention_mask=torch.ones_like(ids),
                pad_token_id=cfg.eos_token_id,
            )
    finally:
        hooks.remove()
    return out, cache


def test_yarn_config_actually_reaches_the_rotary_module():
    """The whole correction hinges on this scalar being non-unity under YaRN."""
    model, _ = _tiny_qwen(rope_scaling=YARN, max_pos=2048)
    rope = _find_rope(model)
    assert rope.attention_scaling == pytest.approx(EXPECTED_SCALING, rel=1e-6)

    plain, _ = _tiny_qwen(rope_scaling=None)
    assert _find_rope(plain).attention_scaling == pytest.approx(1.0)


def test_config_declared_sliding_window_stays_disabled():
    """Qwen2Config nulls the declared window out; the cache needs that to hold."""
    _, cfg = _tiny_qwen(rope_scaling=YARN, max_pos=2048)
    assert cfg.sliding_window is None
    assert cfg.use_sliding_window is False


@pytest.mark.parametrize("quant_ratio", [0.0, 0.5], ids=["fp16-only", "two-tier-int4"])
def test_generation_runs_and_evicts_under_yarn(quant_ratio):
    model, cfg = _tiny_qwen(rope_scaling=YARN, max_pos=2048)
    out, cache = _run(model, cfg, quant_ratio)

    assert out.shape[-1] == PREFILL + GEN
    assert torch.isfinite(model.get_input_embeddings().weight).all()
    # The budget must actually have bound — otherwise this asserts nothing.
    assert cache.get_seq_length(0) < PREFILL + GEN


def test_the_rope_round_trip_is_exact_on_the_real_qwen2_module():
    """The load-bearing assertion, measured numerically rather than by tokens.

    Token agreement on a randomly-initialised tiny model is a weak signal (near-
    tied logits), so assert the invariant directly: a post-RoPE key stripped and
    re-applied at its own positions, through the REAL YaRN-scaled
    ``Qwen2RotaryEmbedding``, must come back unchanged.
    """
    from modules.quant.effective import rotate_key_window, unrotate_key_window

    model, _ = _tiny_qwen(rope_scaling=YARN, max_pos=2048)
    rope = _find_rope(model)
    assert rope.attention_scaling == pytest.approx(EXPECTED_SCALING, rel=1e-6)

    torch.manual_seed(1)
    k_post = torch.randn(1, 2, 8, 16)
    pos = torch.arange(40, 48, dtype=torch.long).unsqueeze(0)

    k_back = rotate_key_window(unrotate_key_window(k_post, pos, rope), pos, rope)
    torch.testing.assert_close(k_back, k_post, rtol=1e-5, atol=1e-5)


def test_dropping_the_correction_inflates_the_key_by_a_squared():
    """Pins the size and sign of the regression, so a revert fails here.

    Without the correction the round trip returns ``a²·k`` — 1.2965x at YaRN
    factor 4.0, on every int4 key and every Q->fp promotion.
    """
    import modules.quant.effective as eff

    model, _ = _tiny_qwen(rope_scaling=YARN, max_pos=2048)
    rope = _find_rope(model)

    torch.manual_seed(1)
    k_post = torch.randn(1, 2, 8, 16)
    pos = torch.arange(40, 48, dtype=torch.long).unsqueeze(0)

    original = eff.rope_attention_scaling
    eff._warned_rope_scaling[0] = True            # the guard is tested elsewhere
    try:
        eff.rope_attention_scaling = lambda _m: 1.0        # the pre-fix behaviour
        k_bad = eff.rotate_key_window(
            eff.unrotate_key_window(k_post, pos, rope), pos, rope
        )
    finally:
        eff.rope_attention_scaling = original
        eff._warned_rope_scaling[0] = False

    ratio = (k_bad / k_post).mean().item()
    assert ratio == pytest.approx(EXPECTED_SCALING ** 2, rel=1e-3)
    assert ratio == pytest.approx(1.2965, rel=1e-3)


def test_unscaled_qwen_is_untouched_by_the_correction():
    """Without rope_scaling, a == 1 and the branch is skipped — same tokens.

    End-to-end through ``generate()``, because the claim being made about the
    Llama and Mistral columns is that nothing in their pipeline moved.
    """
    import modules.quant.effective as eff

    model, cfg = _tiny_qwen(rope_scaling=None)
    good, _ = _run(model, cfg, 0.5)

    original = eff.rope_attention_scaling
    try:
        eff.rope_attention_scaling = lambda _m: 1.0
        same, _ = _run(model, cfg, 0.5)
    finally:
        eff.rope_attention_scaling = original

    assert torch.equal(good, same)


def test_the_config_overrides_reach_a_real_model_the_way_the_loader_applies_them():
    """The production path, end to end, on real transformers objects.

    ``utils.model_loading`` patches the ``AutoConfig`` and then constructs the
    model from it. Everything downstream — the rotary module's frequencies and
    ``attention_scaling``, the runners' ``_model_context_window`` (which reads
    ``max_position_embeddings`` and decides LongBench truncation), the banded-
    attention guard — depends on that patch actually landing, so build the model
    the same way and read the results off the object.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from utils.config import ModelConfig
    from utils.model_loading import (
        apply_model_config_overrides,
        pin_greedy_generation_defaults,
    )

    # The shipped Qwen2.5-7B-Instruct config, in miniature: 32K window, no
    # rope_scaling, a declared-but-disabled sliding window.
    hf = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32768, rope_theta=1000000.0,
        sliding_window=131072, use_sliding_window=False,
    )
    assert hf.rope_scaling is None and hf.sliding_window is None

    # …and the eval configs' model block.
    m = ModelConfig(
        name="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16",
        attn_implementation="eager",
        max_position_embeddings=131072,
        rope_scaling={"rope_type": "yarn", "factor": 4.0,
                      "original_max_position_embeddings": 32768},
        sliding_window=131072, use_sliding_window=False,
    )
    changes = apply_model_config_overrides(hf, m)
    assert changes, "overrides must be reported for the log and the sidecar"

    hf._attn_implementation = "eager"
    model = Qwen2ForCausalLM(hf).eval()

    # 1. The context window the runners read for truncation / the OOD guard.
    assert model.config.max_position_embeddings == 131072

    # 2. YaRN reached the rotary module — the whole point.
    assert _find_rope(model).attention_scaling == pytest.approx(
        EXPECTED_SCALING, rel=1e-6
    )

    # 3. The declared sliding window is still OFF, so the banded-attention guard
    #    stays correctly quiet and attention is comparable over a compacted cache.
    assert model.config.sliding_window is None

    # 4. Greedy is bare greedy. Qwen2.5-7B-Instruct's real generation_config.
    model.generation_config.repetition_penalty = 1.05
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.8
    model.generation_config.top_k = 20
    model.generation_config.do_sample = True
    model.generation_config.eos_token_id = [151645, 151643]

    changed = pin_greedy_generation_defaults(model)
    assert changed["repetition_penalty"] == (1.05, 1.0)
    assert model.generation_config.repetition_penalty == 1.0
    assert model.generation_config.do_sample is False
    # The stopping contract is untouched — that is the model's, not a preference.
    assert model.generation_config.eos_token_id == [151645, 151643]


def test_qwen2_sdpa_drops_cache_position_under_output_attentions():
    """Why the eager configs must declare ``eager`` and not ``sdpa``.

    ``Qwen2SdpaAttention.forward`` cannot serve ``output_attentions=True`` (SDPA
    returns no weights), so it delegates to ``Qwen2Attention.forward`` — and the
    4.47.1 delegation passes neither ``cache_position`` nor
    ``position_embeddings``. The eager score backend REQUIRES
    ``output_attentions=True``, so pairing it with ``sdpa`` silently lands here:
    the cache stops being handed absolute positions (it falls back to its own
    token count) and the flash hook's ``position_embeddings`` lookup misses.

    ``utils.cache_factory.validate_backend_attn_pairing`` already refuses that
    pairing; this pins the reason, so nobody relaxes the check.
    """
    from utils.cache_factory import (
        ConfigValidationError,
        validate_backend_attn_pairing,
    )

    seen = []
    model, cfg = _tiny_qwen(rope_scaling=YARN, max_pos=2048, attn="sdpa")

    from transformers import DynamicCache

    class _Probe(DynamicCache):
        def update(self, k, v, layer_idx, cache_kwargs=None):
            if layer_idx == 0:
                seen.append((cache_kwargs or {}).get("cache_position"))
            return super().update(k, v, layer_idx, cache_kwargs)

    ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        model(ids, past_key_values=_Probe(), use_cache=True, output_attentions=True)
    assert seen and seen[0] is None, (
        "Qwen2Sdpa no longer drops cache_position under output_attentions — the "
        "pairing rule below may be relaxable, but check position_embeddings too"
    )

    # …and this is why the pairing is rejected outright.
    with pytest.raises(ConfigValidationError):
        validate_backend_attn_pairing("eager", "sdpa")


def test_the_declared_eager_pairing_passes_cache_position():
    """The configuration the eager configs actually declare does not have the bug."""
    from transformers import DynamicCache

    seen = []
    model, _ = _tiny_qwen(rope_scaling=YARN, max_pos=2048, attn="eager")

    class _Probe(DynamicCache):
        def update(self, k, v, layer_idx, cache_kwargs=None):
            if layer_idx == 0:
                seen.append((cache_kwargs or {}).get("cache_position"))
            return super().update(k, v, layer_idx, cache_kwargs)

    ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        model(ids, past_key_values=_Probe(), use_cache=True, output_attentions=True)
    assert seen and seen[0] is not None
    assert seen[0].tolist() == list(range(8))


def test_load_model_and_tokenizer_against_a_real_checkpoint_on_disk(tmp_path, monkeypatch):
    """The production load path, on real files, through ``device_map="auto"``.

    Everything above builds models in memory. This saves a checkpoint and goes
    through ``utils.model_loading.load_model_and_tokenizer`` exactly as a run
    does, because the parts that only exist on disk are the ones a CPU unit test
    otherwise never touches: ``AutoConfig.from_pretrained`` reading the shipped
    config, ``from_pretrained(config=<patched>, torch_dtype=…,
    attn_implementation=…, device_map="auto")`` accepting a config object
    alongside those kwargs, and the checkpoint's ``generation_config.json`` being
    the thing that carries the sampling defaults.

    The tokenizer is stubbed — it is standard HF plumbing this change does not
    touch, and pulling a real one would make the test need the network.
    """
    pytest.importorskip("accelerate")          # what device_map="auto" needs
    from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

    from utils.config import ExperimentConfig, ModelConfig
    import utils.model_loading as ml

    ckpt = tmp_path / "qwen-tiny"
    saved = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32768, rope_theta=1000000.0,
        sliding_window=131072, use_sliding_window=False,
        torch_dtype="bfloat16",
    ))
    # Qwen2.5-7B-Instruct's real generation_config.json.
    saved.generation_config = GenerationConfig(
        do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
        repetition_penalty=1.05, eos_token_id=[151645, 151643],
        pad_token_id=151643,
    )
    saved.save_pretrained(ckpt)

    class _Tok:
        pad_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"
        chat_template = "{{ messages }}"
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        classmethod(lambda cls, *a, **k: _Tok()),
    )

    cfg = ExperimentConfig()
    cfg.model = ModelConfig(
        name=str(ckpt), revision=None, dtype="bfloat16",
        attn_implementation="eager",
        max_position_embeddings=131072,
        rope_scaling={"rope_type": "yarn", "factor": 4.0,
                      "original_max_position_embeddings": 32768},
        sliding_window=131072, use_sliding_window=False,
    )

    model, tok = ml.load_model_and_tokenizer(cfg, is_windowed=True)

    # 1. The overrides survived from_pretrained.
    assert model.config.max_position_embeddings == 131072
    assert model.config.rope_scaling["rope_type"] == "yarn"
    assert model.config.sliding_window is None          # still disabled
    # 2. YaRN reached the rotary module the runners will hand to the cache.
    assert _find_rope(model).attention_scaling == pytest.approx(
        EXPECTED_SCALING, rel=1e-6
    )
    # 3. Weights actually loaded in the requested dtype.
    assert next(model.parameters()).dtype is torch.bfloat16
    # 4. The checkpoint's sampling defaults were neutralised…
    assert model.generation_config.repetition_penalty == 1.0
    assert model.generation_config.do_sample is False
    # 5. …but its stopping contract was not.
    assert model.generation_config.eos_token_id == [151645, 151643]
    assert tok is not None


@pytest.mark.parametrize("kv_dtype,expect_grid", [
    (torch.float16, torch.float16),
    (torch.bfloat16, torch.bfloat16),
])
def test_the_live_cache_gives_the_q_tier_a_grid_matching_its_kv_dtype(
    kv_dtype, expect_grid
):
    """End-to-end: the dtype the cache is built with reaches the slot buffers.

    The bug was a bf16 cache writing its grid into fp16 storage. Assert on the
    live table rather than on the helper, because the failure was in the wiring.
    """
    from modules.windowed_eager_cache.cache import WindowedCache
    from modules.windowed_eager_cache.config import WindowedCacheConfig

    model, cfg = _tiny_qwen(rope_scaling=YARN, max_pos=2048)
    cache = WindowedCache(
        config=WindowedCacheConfig(
            window_size=8, num_sink_tokens=4, local_window_size=0.25,
            cache_budget=0.30, quant_ratio=0.5,
        ),
        prefill_len=PREFILL, model_config=cfg, kv_dtype=kv_dtype,
        rope_module=_find_rope(model), num_layers=cfg.num_hidden_layers,
        max_tokens=GEN,
    )
    table = cache._stores[0].ensure(1, torch.device("cpu"))
    assert cache._stores[0].grid_dtype is expect_grid
    for t in (table.key_scale, table.key_zero, table.val_scale, table.val_zero):
        assert t.dtype is expect_grid


def test_the_grid_widening_does_not_move_a_single_reported_byte():
    """The claim that makes this fix safe to take: compression ratios are unchanged.

    fp16 and bf16 are both 2 bytes, so the Q tier must allocate byte-for-byte the
    same amount either way. Measured off the real slot table, not recomputed from
    the same formula the resolver uses — a formula can agree with itself.
    """
    from modules.windowed_eager_cache.cache import WindowedCache
    from modules.windowed_eager_cache.config import WindowedCacheConfig

    model, cfg = _tiny_qwen(rope_scaling=YARN, max_pos=2048)

    def q_tier_bytes(kv_dtype):
        cache = WindowedCache(
            config=WindowedCacheConfig(
                window_size=8, num_sink_tokens=4, local_window_size=0.25,
                cache_budget=0.30, quant_ratio=0.5,
            ),
            prefill_len=PREFILL, model_config=cfg, kv_dtype=kv_dtype,
            rope_module=_find_rope(model), num_layers=cfg.num_hidden_layers,
            max_tokens=GEN,
        )
        t = cache._stores[0].ensure(1, torch.device("cpu"))
        tensors = (t.key_codes, t.key_scale, t.key_zero,
                   t.val_codes, t.val_scale, t.val_zero)
        return sum(x.numel() * x.element_size() for x in tensors)

    assert q_tier_bytes(torch.float16) == q_tier_bytes(torch.bfloat16)
