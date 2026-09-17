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
        if want is None:
            # A `None` reference means "derive it" -- the knob is a tri-state ARM
            # whose shipped value is the derivation, not a number. Requiring
            # every config to spell `null` would not make anything safer: a
            # config that sets it to anything OTHER than the derivation is still
            # caught, both by the `got == want` line below and by
            # check_operating_point.py. What must not happen is the reverse --
            # the reference moving off None while these configs keep deriving --
            # and `test_a_non_derived_reference_must_be_spelled_out` covers that.
            assert got is None or got == want, (
                f"{name}: {field}={got!r}, operating point is {want!r} (derive)")
            continue
        assert got is not None, f"{name} does not set {field}"
        assert got == want, f"{name}: {field}={got!r}, operating point is {want!r}"


def test_a_non_derived_reference_must_be_spelled_out():
    """The escape hatch above is only safe while the reference IS the derivation.

    If someone changes ``OPERATING_POINT["quant_read_gate"]`` to ``True`` or
    ``False``, "unset means the shipped value" stops being true and every
    quality config silently keeps deriving. This fails at that moment, naming
    the field, so the change comes with the config edits it requires.
    """
    from utils.config import OPERATING_POINT

    derived = {k for k, v in OPERATING_POINT.items() if v is None}
    assert derived == {"quant_read_gate"}, (
        f"{derived} are treated as 'derive it' by the loop above. If a field "
        "left or joined that set, the quality configs need to be re-checked: a "
        "field with a CONCRETE reference must be written into each of them.")
