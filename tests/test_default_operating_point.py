"""The shipped operating point, pinned so it cannot drift without a test failing.

Running the perf table or the profiler with no flags must land on exactly one
configuration — budget 0.20, window 8, gate 0.25, quant_ratio 0.70, local window
128, on the fused gated Triton kernel — and both tools must land on the SAME one.

That has not been true. Three separate splits were live in one week: the YAML
loader rejected ``quant_gate_ratio`` outright, five runners dropped it silently,
and ``STICKYKV_COMPILE_EVICT`` defaulted to 0 in the library while the benchmark
script exported 1, so the profiler attributed a method the table never ran.

Each of those is invisible in the output. A number produced under the wrong one
still looks like a number, which is why this file reads the defaults out of
``run_perf_table.sh`` itself rather than restating them: change the script and
the test moves with it; change one tool's default alone and it breaks.
"""

from __future__ import annotations

import pathlib
import re

import pytest

SCRIPT = pathlib.Path("scripts/run_perf_table.sh")

#: What a no-flag run must resolve to. Values are read from the script below;
#: this is the intent those values have to keep matching.
EXPECTED = {
    "CACHE_BUDGET": "0.20",
    "WINDOW_SIZE": "8",
    "GATE_RATIO": "0.25",
    "QUANT_RATIO": "0.70",
    "LOCAL_WINDOW": "128",
    "BACKEND": "flash_attn",
}


def _script_defaults() -> dict:
    """``VAR="${VAR:-default}"`` pairs from the perf script."""
    src = SCRIPT.read_text(encoding="utf-8")
    return dict(re.findall(r'^([A-Z_]+)="\$\{\1:-([^}]*)\}"', src, re.M))


@pytest.mark.parametrize("name,want", sorted(EXPECTED.items()))
def test_the_perf_script_ships_the_intended_operating_point(name, want):
    got = _script_defaults().get(name)
    assert got == want, (
        f"run_perf_table.sh defaults {name}={got!r}, expected {want!r}. A "
        "no-flag run is what every quoted number comes from; changing it "
        "silently re-bases the whole table.")


def test_the_generated_config_carries_that_point_through_the_loader():
    """The values have to survive YAML load, not merely be written into it.

    ``quant_gate_ratio`` was written into the generated config and then rejected
    by ``utils.config.CacheConfig``, which has no such field — every run died
    before the model loaded. Writing a value and loading it are two different
    things and both have to work.
    """
    import os
    import tempfile

    import yaml

    from utils.config import load_config

    d = _script_defaults()
    raw = {
        "run": {"mode": "perf", "seed": 42},
        "model": {"name": "x", "revision": None, "dtype": "float16"},
        "window": {"window_size": int(d["WINDOW_SIZE"]),
                   "num_sink_tokens": int(d["NUM_SINK"]),
                   "local_window_size": int(d["LOCAL_WINDOW"])},
        "cache": {"quant_ratio": float(d["QUANT_RATIO"]),
                  "quant_budget_mode": d["QUANT_MODE"],
                  "quant_gate_ratio": float(d["GATE_RATIO"]),
                  "first_eviction_step": 0},
        "perf": {"configs": [{"name": "ours", "cache_backend": "windowed",
                              "cache_package": d["BACKEND"],
                              "cache_budget": float(d["CACHE_BUDGET"]),
                              "quant_ratio": float(d["QUANT_RATIO"])}]},
    }
    p = os.path.join(tempfile.mkdtemp(), "c.yaml")
    pathlib.Path(p).write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(p)

    assert cfg.cache.quant_gate_ratio == 0.25
    assert cfg.cache.quant_ratio == 0.70
    assert cfg.window.window_size == 8
    assert cfg.window.local_window_size == 128


def test_the_profiler_resolves_the_same_cache_as_the_table():
    """Both tools must build the same cache, or the profile attributes a method
    the table never ran — which is exactly what compile_evict=0 did."""
    import os
    import sys
    import tempfile

    import yaml

    sys.path.insert(0, "scripts")
    from audit_e2e import resolve_cache_kwargs
    from utils.config import load_config

    d = _script_defaults()
    raw = {
        "run": {"mode": "perf", "seed": 42},
        "model": {"name": "x", "revision": None, "dtype": "float16"},
        "window": {"window_size": int(d["WINDOW_SIZE"]),
                   "num_sink_tokens": int(d["NUM_SINK"]),
                   "local_window_size": int(d["LOCAL_WINDOW"])},
        "cache": {"quant_ratio": float(d["QUANT_RATIO"]),
                  "quant_budget_mode": d["QUANT_MODE"],
                  "quant_gate_ratio": float(d["GATE_RATIO"]),
                  "first_eviction_step": 0},
        "perf": {"configs": [{"name": "ours", "cache_backend": "windowed",
                              "cache_package": d["BACKEND"],
                              "cache_budget": float(d["CACHE_BUDGET"]),
                              "quant_ratio": float(d["QUANT_RATIO"])}]},
    }
    p = os.path.join(tempfile.mkdtemp(), "c.yaml")
    pathlib.Path(p).write_text(yaml.safe_dump(raw), encoding="utf-8")
    kw = resolve_cache_kwargs(load_config(p), 0.70, "flash_attn")

    assert kw["cache_budget"] == 0.20
    assert kw["window_size"] == 8
    assert kw["local_window_size"] == 128
    assert kw["quant_ratio"] == 0.70
    assert kw["quant_gate_ratio"] == 0.25


def test_the_fused_gated_kernel_is_the_only_decode_on_cuda():
    """No setting routes a production run to a different implementation."""
    import inspect
    import os

    from modules.windowed_cache.decode_kernel import (
        describe_decode_backend, fused_decode_enabled)

    prev = os.environ.get("STICKYKV_FUSED_DECODE")
    try:
        for v in ("0", "false", "off", "1"):
            os.environ["STICKYKV_FUSED_DECODE"] = v
            assert fused_decode_enabled() is True, (
                f"STICKYKV_FUSED_DECODE={v} still selects a decode path")
    finally:
        if prev is None:
            os.environ.pop("STICKYKV_FUSED_DECODE", None)
        else:
            os.environ["STICKYKV_FUSED_DECODE"] = prev

    assert "TRITON fused two-tier decode kernel" in describe_decode_backend(True)
    assert "materialize" not in describe_decode_backend(True)
    # And the reason is structural, not a default that could be flipped back.
    from modules.windowed_cache import decode_kernel as dk
    assert "os.environ" not in inspect.getsource(dk.fused_decode_enabled)


def test_eviction_is_compiled_on_cuda_and_eager_on_cpu_with_nothing_set():
    """The device decides, not an env var.

    The library defaulted to eager while run_perf_table.sh exported compiled, so
    a profile and a table of the same cell measured different methods — worth
    ~16 ms of TPOT, landing in the elementwise kernels a profile is read to
    attribute.
    """
    import os
    from unittest import mock

    from modules.windowed_cache.cache import _compile_evict_enabled

    prev = os.environ.pop("STICKYKV_COMPILE_EVICT", None)
    try:
        with mock.patch("torch.cuda.is_available", return_value=True):
            assert _compile_evict_enabled() is True, "CUDA must compile"
        with mock.patch("torch.cuda.is_available", return_value=False):
            assert _compile_evict_enabled() is False, "CPU must not"
        # An explicit value still wins, for the compile-failure bisection.
        os.environ["STICKYKV_COMPILE_EVICT"] = "0"
        with mock.patch("torch.cuda.is_available", return_value=True):
            assert _compile_evict_enabled() is False
    finally:
        os.environ.pop("STICKYKV_COMPILE_EVICT", None)
        if prev is not None:
            os.environ["STICKYKV_COMPILE_EVICT"] = prev
