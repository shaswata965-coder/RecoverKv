"""What ``quant_ratio`` divides, and that the knob actually reaches the cache.

Two separate failures produced the LongBench regression in
``ACCURACY_RECOVERY_PLAN.md`` §2, and each needs its own guard:

1. The default flipped from ``bytes`` to ``tokens`` in f71fec0, so the retained
   cache stopped costing what ``cache_budget`` granted — 57% of it at q=0.7 —
   while still being reported at the granted budget.
2. ``quant_budget_mode`` was never forwarded from the YAML into
   ``WindowedCacheConfig`` by the LongBench/GSM8K/RULER runners, so the knob was
   inert and nothing in the run record said which mode had run. A test that only
   checks the resolver would have passed throughout.
"""

from __future__ import annotations

import types

import pytest
import torch

from modules.windowed_cache.config import WindowedCacheConfig
from utils.cache_factory import ConfigValidationError, quant_budget_mode_kwargs
from utils.config import CacheConfig


# Llama-3.1-8B geometry, the shape every LongBench row runs at.
MODEL_CONFIG = types.SimpleNamespace(
    num_key_value_heads=8, head_dim=128, num_hidden_layers=32,
    hidden_size=4096, num_attention_heads=32,
)


def _resolve(mode: str, q: float, prefill: int = 3600):
    cfg = WindowedCacheConfig(
        window_size=8, num_sink_tokens=5, local_window_size=128,
        cache_budget=0.20, quant_ratio=q, quant_budget_mode=mode,
    )
    return cfg.resolve(prefill, MODEL_CONFIG, torch.float16, 128)


# ---------------------------------------------------------------------------
# The default
# ---------------------------------------------------------------------------


def test_default_is_bytes_on_both_config_classes():
    """The YAML-facing dataclass and the cache dataclass must not disagree.

    They did between f71fec0 and now on the *value*; a split default is how a
    quality suite ends up on a latency suite's operating point.
    """
    assert CacheConfig().quant_budget_mode == "bytes"
    assert WindowedCacheConfig(
        window_size=8, num_sink_tokens=5, local_window_size=128,
        cache_budget=0.20).quant_budget_mode == "bytes"


# ---------------------------------------------------------------------------
# What each mode actually buys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.0, 0.3, 0.5, 0.7])
def test_bytes_mode_spends_the_budget_it_was_granted(q):
    """The property that makes 'bytes' the mode a quality claim is stated in.

    ``cache_budget=0.20`` must mean 20% of the full cache's BYTES at every
    quant_ratio. Floor division at each tier costs at most one window, hence the
    tolerance rather than an equality.
    """
    r = _resolve("bytes", q)
    assert r.budget_utilisation == pytest.approx(1.0, abs=0.02), (
        f"bytes mode at q={q} spends {r.budget_utilisation:.1%} of its budget"
    )


@pytest.mark.parametrize("q,ceiling", [(0.5, 0.75), (0.7, 0.65)])
def test_tokens_mode_underspends_and_that_is_the_regression(q, ceiling):
    """Documents the cost of the mode a latency table needs.

    Pinning the key count while the keys get cheaper means declining to spend
    the granted budget. This is not a bug in 'tokens' — it is why 'tokens' must
    never be the default for a quality suite.
    """
    r = _resolve("tokens", q)
    assert r.budget_utilisation < ceiling
    assert r.retained_windows == _resolve("tokens", 0.0).retained_windows


@pytest.mark.parametrize("field", ["retained_windows", "retained_tokens",
                                   "retained_bytes", "top_k_fp", "N_q"])
def test_q0_is_identical_under_both_modes(field):
    """At q=0 there is no Q tier to split, so the modes cannot diverge.

    Any divergence here would mean the pure-fp16 path is no longer
    byte-identical between the two, which the resolver's `if q == 0.0` guard
    exists to prevent.
    """
    assert getattr(_resolve("bytes", 0.0), field) == \
           getattr(_resolve("tokens", 0.0), field)


def test_bytes_mode_retains_more_keys_than_tokens_mode():
    """The accuracy the regression cost, stated as a ratio.

    Same config, same budget: 'bytes' holds ~2.2x the keys at q=0.5. That factor
    is the whole of the LongBench delta this plan is chasing.
    """
    ratio = _resolve("bytes", 0.5).retained_tokens / _resolve("tokens", 0.5).retained_tokens
    assert ratio > 2.0


# ---------------------------------------------------------------------------
# The knob has to reach the cache
# ---------------------------------------------------------------------------


class _FakeEagerConfig:
    """Stands in for the eager package's config, which has no such field."""
    __dataclass_fields__ = {"window_size": None, "quant_ratio": None}


def test_kwargs_are_passed_to_a_backend_that_has_the_field():
    assert quant_budget_mode_kwargs(WindowedCacheConfig, "tokens") == \
        {"quant_budget_mode": "tokens"}


def test_bytes_is_dropped_for_eager_because_eager_already_computes_bytes():
    """Omitting it is a no-op only because the two agree on the value."""
    assert quant_budget_mode_kwargs(_FakeEagerConfig, "bytes") == {}


def test_tokens_on_eager_raises_rather_than_being_dropped():
    """Silently dropping it would run a different operating point.

    The eager resolver has no mode branch, so a config asking for 'tokens' would
    get 'bytes' — 2.2x the retained keys — under the config's own name.
    """
    with pytest.raises(ConfigValidationError, match="quant_budget_mode"):
        quant_budget_mode_kwargs(_FakeEagerConfig, "tokens")


def test_longbench_runner_forwards_the_mode():
    """The inert-knob regression itself: YAML -> WindowedCacheConfig.

    Constructed with ``object.__new__`` so the test needs no model download —
    ``_setup_windowed_cache`` only reads ``config``, ``model.named_modules`` and
    the three injected cache classes.
    """
    from modules.evaluation.longbench_runner import LongBenchRunner

    seen = {}

    def fake_config_cls(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(**kwargs)

    fake_config_cls.__dataclass_fields__ = \
        WindowedCacheConfig.__dataclass_fields__

    rope = types.SimpleNamespace()
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(num_hidden_layers=2),
        named_modules=lambda: iter([("model.rotary_emb", rope)]),
    )
    cfg = types.SimpleNamespace(
        cache=CacheConfig(cache_budget=0.20, window_size=8, num_sink_tokens=5,
                          local_window_size=128, quant_ratio=0.5,
                          quant_budget_mode="bytes"),
        model=types.SimpleNamespace(dtype="float16"),
    )

    runner = object.__new__(LongBenchRunner)
    runner.config = cfg
    runner.model = model
    runner._resolved_sample = None
    runner.WindowedCacheConfig = fake_config_cls
    runner.WindowedCache = lambda **kw: types.SimpleNamespace(
        resolved=_resolve("bytes", 0.5))
    runner.install_score_hooks = lambda *a, **k: None

    runner._setup_windowed_cache(torch.zeros(1, 3600, dtype=torch.long), 128)

    assert seen["quant_budget_mode"] == "bytes"
    # And the sidecar can now say what actually ran, not just what was asked.
    assert runner._resolved_sample["quant_budget_mode"] == "bytes"
    assert runner._resolved_sample["budget_utilisation"] == pytest.approx(
        1.0, abs=0.02)
