"""Speed and quality must be measured at the same operating point.

This is the guard for the one defect that a paper cannot survive: the
throughput table and the accuracy table describing different caches.

Before ``utils.config.OPERATING_POINT`` existed they did. ``run_perf_table.sh``
timed the method at ``quant_ratio 0.70`` in ``quant_budget_mode tokens``;
``configs/longbench_ours_flash_attn.yaml`` scored it at ``quant_ratio 0.0`` --
no int2 tier, no read gate, pure fp16 eviction -- and ``longbench_ours_eager``
additionally used ``window_size 32``, ``num_sink_tokens 4`` and a fractional
local window. Nothing compared them and nothing printed the difference.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_every_reported_config_is_at_the_operating_point():
    """``scripts/check_operating_point.py`` must exit clean.

    Its output names each divergence, so a failure here is self-describing.
    An intentional ablation is recorded in ``OPERATING_POINT_EXEMPT`` with a
    reason -- which is the difference between a choice and an accident.
    """
    r = subprocess.run([sys.executable, "scripts/check_operating_point.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_decode_graph_is_not_on_by_default(monkeypatch):
    """One method, one code path.

    The CUDA-graph runner is wired into ``perf_runner``'s decode loop and NOT
    into ``model.generate``, which is what LongBench and GSM8K call. On by
    default would mean the speed table came from a path no accuracy number
    describes -- the same class of error as profiling an eager eviction against
    a compiled table, except it reaches the paper.
    """
    monkeypatch.delenv("STICKYKV_DECODE_GRAPH", raising=False)
    from modules.windowed_cache.graph_decode import graph_mode
    assert graph_mode() == "off"


@pytest.mark.parametrize("name", [
    "longbench_ours_flash_attn.yaml",
    "longbench_ours_eager.yaml",
    "gsm8k_budget20.yaml",
])
def test_the_quality_configs_resolve_the_same_cache(name):
    """The flash and eager arms are meant to be the same method by two routes,
    and GSM8K the same method on another dataset. If they differ, the three
    numbers are not comparable to each other either."""
    import yaml
    from utils.config import OPERATING_POINT

    block = yaml.safe_load((ROOT / "configs" / name).read_text())["cache"]
    for field, want in OPERATING_POINT.items():
        got = block.get(field)
        assert got is not None, f"{name} does not set {field}"
        assert got == want, f"{name}: {field}={got!r}, operating point is {want!r}"
