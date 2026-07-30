"""The window-score accumulator must not run in the KV dtype.

`state.window_scores` is a running sum with two nested accumulations: over the T
query rows of a forward pass (up to ~18k at full LongBench context), and then
over every decode step of the generation. Its dtype is decided by whatever the
score hook allocates.

In a reduced-precision accumulator an addend below half an ULP of the running
total rounds to no change and is **discarded outright**. Measured on a realistic
LongBench trajectory that silently drops ~76% of the per-decode-step
contributions in bfloat16 and ~52% in float16, and the retained top-k set
diverges from the exact answer by ~9% on a 512-step generation (gov_report).

It surfaced as a Qwen2.5 problem because Qwen2.5 is the only bf16 model column
(8 mantissa bits against float16's 11), so its scores were not measured the same
way the Llama-3.1 and Mistral-v0.2 columns' were — but both hooks had it, and the
fix moves all three onto an exact sum rather than onto the same rounding.

The window ranking IS the method, so this is pinned rather than left to review.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from modules.windowed_cache.scorer import accumulate  # noqa: E402
from utils.cache_factory import (  # noqa: E402
    MAX_SUPPORTED_TRANSFORMERS,
    is_transformers_version_supported,
)

WS, SINK, LOCAL = 8, 4, 16
PREFILL = 96

#: The arithmetic tests below run anywhere. The model-driven ones need a
#: transformers the windowed cache actually works on — past 4.47.1 the Cache ABI
#: changed (`Cache.layers`, `get_mask_sizes`) and even a bare `model.forward`
#: with this cache raises from inside the mask builder, which is why the suite
#: already carries a version gate rather than pretending otherwise. Skipped, not
#: xfailed: on an unsupported version the result carries no information either way.
needs_pinned_transformers = pytest.mark.skipif(
    not is_transformers_version_supported(transformers.__version__),
    reason=(
        f"needs transformers <= "
        f"{'.'.join(str(p) for p in MAX_SUPPORTED_TRANSFORMERS)}; "
        f"installed {transformers.__version__}"
    ),
)


# ---------------------------------------------------------------------------
# The defect itself, stated as arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expect_loss",
    [(torch.bfloat16, True), (torch.float16, True), (torch.float32, False)],
)
def test_sub_ulp_increments_vanish_in_reduced_precision(dtype, expect_loss):
    """A decode step's contribution is below the running total's ULP.

    Realistic magnitudes: a window carries ~8.0 of accumulated prefill mass, and
    one decode step adds ~2.2e-3 to it (the step's softmax mass spread over a
    budget-sized cache). `expect_loss` is not a claim about *how much* is lost —
    it is the claim that the addition does nothing at all.
    """
    total = torch.full((64,), 8.0, dtype=dtype)
    step = torch.full((64,), 2.2e-3, dtype=dtype)

    before = total.clone()
    for _ in range(32):
        accumulate(total, step)

    unchanged = bool((total == before).all())
    assert unchanged is expect_loss, (
        f"{dtype}: 32 increments of 2.2e-3 on a running 8.0 "
        f"{'were' if unchanged else 'were not'} entirely discarded"
    )
    if not expect_loss:
        assert total[0].item() == pytest.approx(8.0 + 32 * 2.2e-3, rel=1e-6)


def test_accumulate_preserves_the_dtype_it_is_given():
    """`accumulate` is in-place, so it cannot widen a narrow accumulator.

    This is why the fix belongs in the hooks (which allocate) and not here.
    """
    acc = torch.zeros(4, dtype=torch.bfloat16)
    accumulate(acc, torch.ones(4, dtype=torch.float32))
    assert acc.dtype is torch.bfloat16


# ---------------------------------------------------------------------------
# Both hooks, on a real bf16 attention module
# ---------------------------------------------------------------------------


def _tiny_model(arch, dtype):
    """~1M params from a config — no network. head_dim = 16, divisible by 4."""
    torch.manual_seed(0)
    kw = dict(vocab_size=256, hidden_size=64, intermediate_size=128,
              num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=512)
    if arch == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        cfg = Qwen2Config(use_sliding_window=False, sliding_window=4096, **kw)
        return Qwen2ForCausalLM(cfg).to(dtype).eval(), cfg

    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(**kw)
    return LlamaForCausalLM(cfg).to(dtype).eval(), cfg


def _rope_of(model):
    for name, mod in model.named_modules():
        if "rotary" in name.lower():
            return mod
    raise AssertionError("no rotary module found")


def _run_one_forward(backend, arch, dtype, quant_ratio):
    """Prefill once through `model.forward` and return the layer-0 scores.

    Deliberately `forward`, not `generate`: the hook fires from the attention
    module either way, and `generate` drags in version-specific cache plumbing
    that has nothing to do with what is being asserted here.
    """
    from utils.cache_factory import get_cache_classes

    WindowedCache, WindowedCacheConfig, install_score_hooks = get_cache_classes(
        backend
    )
    model, cfg = _tiny_model(arch, dtype)

    cache_config = WindowedCacheConfig(
        window_size=WS, num_sink_tokens=SINK, local_window_size=LOCAL,
        cache_budget=0.5, quant_ratio=quant_ratio,
    )
    cache = WindowedCache(
        config=cache_config, prefill_len=PREFILL, model_config=cfg,
        kv_dtype=dtype, rope_module=_rope_of(model),
        num_layers=cfg.num_hidden_layers, max_tokens=8,
    )
    hooks = install_score_hooks(model, cache, cache_config)
    try:
        ids = torch.randint(0, cfg.vocab_size, (1, PREFILL))
        with torch.no_grad():
            model(
                ids,
                past_key_values=cache,
                use_cache=True,
                output_attentions=(backend == "eager"),
            )
    finally:
        hooks.remove()

    scores = cache.cache_kwargs[0].get("window_scores")
    assert scores is not None, (
        f"{backend}/{arch}: the hook produced no scores — it degraded to "
        f"sink+local and this test would pass vacuously"
    )
    return scores


@needs_pinned_transformers
@pytest.mark.parametrize("backend", ["flash_attn", "eager"])
@pytest.mark.parametrize("arch", ["llama", "qwen2"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hook_scores_are_fp32_whatever_the_kv_dtype(backend, arch, dtype):
    """The accumulator is fp32 even when the cache — and so the model — is not.

    Parametrized over both KV dtypes on purpose: the fp32 accumulator is not a
    "Qwen fix", it is the same fix on every column, and a future change that
    reintroduces `dtype=q.dtype` would pass a bf16-only test if the model
    happened to be fp16.
    """
    scores = _run_one_forward(backend, arch, dtype, quant_ratio=0.0)
    assert scores.dtype is torch.float32, (
        f"{backend}/{arch} at kv_dtype={dtype} accumulates window scores in "
        f"{scores.dtype}; sub-ULP per-step contributions are discarded there"
    )


@needs_pinned_transformers
@pytest.mark.parametrize("backend", ["flash_attn", "eager"])
def test_two_tier_scores_are_fp32_too(backend):
    """`quant_ratio > 0` takes the `reduce_two_tier_scores` path — same rule."""
    scores = _run_one_forward(backend, "qwen2", torch.bfloat16, quant_ratio=0.5)
    assert scores.dtype is torch.float32


@needs_pinned_transformers
@pytest.mark.parametrize("backend", ["flash_attn", "eager"])
def test_state_window_scores_inherit_fp32_through_update(backend):
    """The dtype has to survive into `state.window_scores`, which is the one
    that is accumulated across steps — `cache.update` clones the hook's tensor,
    so a narrow hook would have silently pinned the running total narrow.
    """
    from utils.cache_factory import get_cache_classes

    WindowedCache, WindowedCacheConfig, install_score_hooks = get_cache_classes(
        backend
    )
    model, cfg = _tiny_model("qwen2", torch.bfloat16)
    cache_config = WindowedCacheConfig(
        window_size=WS, num_sink_tokens=SINK, local_window_size=LOCAL,
        cache_budget=0.5,
    )
    cache = WindowedCache(
        config=cache_config, prefill_len=PREFILL, model_config=cfg,
        kv_dtype=torch.bfloat16, rope_module=_rope_of(model),
        num_layers=cfg.num_hidden_layers, max_tokens=8,
    )
    hooks = install_score_hooks(model, cache, cache_config)
    try:
        ids = torch.randint(0, cfg.vocab_size, (1, PREFILL))
        with torch.no_grad():
            model(ids, past_key_values=cache, use_cache=True,
                  output_attentions=(backend == "eager"))
    finally:
        hooks.remove()

    ws = cache._states[0].window_scores
    assert ws is not None and ws.dtype is torch.float32


# ---------------------------------------------------------------------------
# The eager-attention ceiling has a num_layers factor
# ---------------------------------------------------------------------------
#
# The eager backend needs `output_attentions=True` to score at all, and
# transformers does not hand that back one layer at a time: `<Model>Model.forward`
# accumulates every layer's `attn_weights` into `all_self_attns` and returns the
# whole tuple, so all `num_hidden_layers` [B, H_q, T, T] matrices are resident
# simultaneously. The guard used to report ONE layer's worth, which understated
# the requirement by 28x on Qwen2.5-7B and put its warning threshold at ~11.9k
# tokens when the real 8 GB point is ~2.3k — i.e. it stayed quiet through the
# entire range where LongBench actually lives.


class _FakeRunner:
    """Just enough surface for `_warn_if_eager_attention_cannot_fit`.

    Bound as an unbound method so the guard is exercised exactly as written,
    without constructing a LongBenchRunner (which reads vendored configs off disk
    and builds SHAs).
    """

    def __init__(self, heads, layers, backend="eager"):
        # Read the threshold off the real class rather than restating 8.0, so
        # these assertions track it if it is ever retuned.
        from modules.evaluation.longbench_runner import LongBenchRunner

        self._EAGER_ATTN_WARN_GB = LongBenchRunner._EAGER_ATTN_WARN_GB
        self.cache_backend_package = backend
        self.model = type("M", (), {"config": type("C", (), {
            "num_attention_heads": heads, "num_hidden_layers": layers,
        })()})()
        self.config = type("Cfg", (), {"model": type("MC", (), {
            "attn_implementation": backend,
        })()})()


def _warn_calls(heads, layers, prompt_tokens, backend="eager"):
    from modules.evaluation.longbench_runner import LongBenchRunner

    captured = []
    runner = _FakeRunner(heads, layers, backend)

    import modules.evaluation.longbench_runner as mod

    real = mod.log.warning
    mod.log.warning = lambda msg, *a: captured.append(msg % a)
    try:
        LongBenchRunner._warn_if_eager_attention_cannot_fit(runner, prompt_tokens)
    finally:
        mod.log.warning = real
    return captured


#: Qwen2.5-7B-Instruct.
QWEN_HEADS, QWEN_LAYERS = 28, 28


def test_guard_fires_where_one_layer_alone_would_not():
    """4000 tokens: 0.9 GB for one layer, 25 GB across 28. It must warn."""
    one_layer_gb = QWEN_HEADS * 4000 ** 2 * 2 / 1e9
    assert one_layer_gb < 8.0, "premise: one layer alone is under the threshold"
    assert _warn_calls(QWEN_HEADS, QWEN_LAYERS, 4000), (
        "the guard stayed quiet at a prompt length that needs ~25 GB of "
        "attention weights"
    )


def test_reported_size_scales_with_layer_count():
    """Same prompt, 1 layer vs 28 — the reported GB differs by 28x.

    14000 tokens, chosen so even a single layer (11 GB) is over the threshold and
    both cases produce a message to compare. It is the ratio being asserted, not
    either absolute figure.
    """
    import re

    def gb(layers):
        msgs = _warn_calls(QWEN_HEADS, layers, 14000)
        assert msgs, f"no warning at 14000 tokens with {layers} layers"
        return float(re.search(r"materializes (\d+) GB", msgs[0]).group(1))

    assert gb(QWEN_LAYERS) == pytest.approx(gb(1) * QWEN_LAYERS, rel=0.02)


def test_suggested_max_length_accounts_for_every_layer():
    """The suggested budget must itself fit, not be sqrt(num_layers) too long."""
    import re

    msgs = _warn_calls(QWEN_HEADS, QWEN_LAYERS, 9000)
    fits = int(re.search(r"is ~(\d+) tokens", msgs[0]).group(1))
    actual_gb = QWEN_LAYERS * QWEN_HEADS * fits ** 2 * 2 / 1e9
    assert actual_gb <= 8.0 + 0.05, (
        f"suggested max_length={fits} would still need {actual_gb:.1f} GB"
    )
    # And it is not absurdly conservative either.
    assert actual_gb > 7.0


def test_guard_is_silent_on_the_flash_backend():
    """Flash-attention-2 never materializes the matrix — no ceiling to warn about."""
    assert not _warn_calls(QWEN_HEADS, QWEN_LAYERS, 20000, backend="flash_attn")


def test_guard_survives_an_unreadable_layer_count():
    """Falls back to one layer rather than skipping the check entirely."""
    from modules.evaluation.longbench_runner import LongBenchRunner

    runner = _FakeRunner(QWEN_HEADS, QWEN_LAYERS)
    del type(runner.model.config).num_hidden_layers
    captured = []
    import modules.evaluation.longbench_runner as mod

    real = mod.log.warning
    mod.log.warning = lambda msg, *a: captured.append(msg % a)
    try:
        LongBenchRunner._warn_if_eager_attention_cannot_fit(runner, 20000)
    finally:
        mod.log.warning = real
    assert captured, "a missing layer count must not silence the guard"
