"""The flash decode path has no ungated arm, and that is enforced, not hoped.

The gate sat unreachable for eight commits (``3658f72``..``9d131e4``) while every
other signal said the method was running: the cards were built on every eviction,
the tests passed, the design docs described it, and the step read 100% of the
tier. What was missing was a caller — and nothing in the output said so.

So the invariant is enforced in two places and pinned here:

* ``flash_decode._run_fused`` raises when a fused layer arrives without cards,
  rather than reading the whole tier "correctly and slowly";
* ``flash_decode.stats()`` reports ``gated`` against ``fired``, so a finished run
  carries its own proof.

``quant_gate_ratio = 1.0`` is how you ask for an ungated read. It selects every
window through the same code, which makes it a *provable* no-op rather than a
second implementation that has to be kept in step.

CPU-only.
"""

from __future__ import annotations

import pytest
import torch

from modules.windowed_cache import flash_decode


def _ctx_without_cards():
    """A fused hand-off over a non-empty Q tier that carries no gate cards."""
    return {
        "layer_idx": 3,
        "gate": None,
        "qtier": {"k_codes": torch.zeros(1, 17, 2, 8, 2, dtype=torch.uint8)},
        "score_meta": (torch.zeros(1, 4, dtype=torch.long), 8),
        "num_sink": 5,
        "window_size": 8,
        "scaling": 0.1,
        "cache": None,
    }


def test_a_fused_layer_without_cards_raises_rather_than_reading_everything():
    """No silent ungated arm. Reading the whole tier here would be *correct* and
    would look exactly like success — a decode timed against a method it is not
    running, which is the one failure this path must refuse."""
    q = torch.randn(1, 8, 16)
    with pytest.raises(flash_decode.FusedDecodeNotReached) as e:
        flash_decode._run_fused(
            _ctx_without_cards(), q.unsqueeze(1), q.unsqueeze(1), q.unsqueeze(1))
    msg = str(e.value)
    assert "no gate cards" in msg
    assert "quant_gate_ratio=1.0" in msg, "the error must name the supported way"


def test_the_error_names_the_layer_and_the_tier_it_would_have_read():
    """A bare refusal is not actionable; the count is what says how much of the
    cache was about to be read ungated."""
    q = torch.randn(1, 8, 16)
    with pytest.raises(flash_decode.FusedDecodeNotReached, match="17 active"):
        flash_decode._run_fused(
            _ctx_without_cards(), q.unsqueeze(1), q.unsqueeze(1), q.unsqueeze(1))


def test_stats_expose_gated_against_fired():
    """``gated == fired`` is the runtime form of the invariant, and it must be
    readable without a device sync — the counters are Python ints taken off
    tensor SHAPES, never off values."""
    flash_decode.reset_stats()
    s = flash_decode.stats()
    assert {"armed", "fired", "gated", "windows_read", "windows_active"} <= set(s)
    assert s["read_fraction"] is None, "undefined until the gate has run"
    assert all(s[k] == 0 for k in ("armed", "fired", "gated"))


def test_read_fraction_is_the_realised_ratio_not_the_configured_one():
    """What ``quant_gate_ratio`` actually came out as. A configured 0.25 that
    realises as 1.0 is the bug this number exists to make visible."""
    flash_decode.reset_stats()
    flash_decode._STATS["windows_read"] = 46
    flash_decode._STATS["windows_active"] = 184
    assert flash_decode.stats()["read_fraction"] == pytest.approx(0.25)
    flash_decode.reset_stats()


# ---------------------------------------------------------------------------
# every runner, not just the perf suite
# ---------------------------------------------------------------------------


def test_expect_gated_is_one_definition_for_every_runner():
    """The three conditions that decide the path, restated where a runner can
    check them after the fact. A perf row and a LongBench sidecar must not be
    able to disagree about what the run was."""
    e = flash_decode.expect_gated
    assert e("flash_attn", 0.70, True) is True
    assert e("flash_attn", 0.0, True) is False, "no Q tier: nothing to gate"
    assert e("eager", 0.70, True) is False, "eager is the supported ungated ask"
    assert e("flash_attn", 0.70, False) is False, "CPU runs the materialize oracle"
    # Tolerant of what a YAML actually yields.
    assert e("flash_attn", None, True) is False
    assert e(None, 0.70, True) is False


@pytest.mark.parametrize("fired,gated,expect,verdict,ok", [
    (192, 192, True, flash_decode.GATE_OK, True),
    (0, 0, False, flash_decode.GATE_NOT_EXPECTED, True),
    (0, 0, True, flash_decode.GATE_NEVER_FIRED, False),
    (192, 5, True, flash_decode.GATE_PARTIAL, False),
])
def test_gate_report_verdicts(fired, gated, expect, verdict, ok):
    """The verdict a runner stores. Stable strings, so a reader can grep a
    results directory for the runs that were not gated."""
    flash_decode.reset_stats()
    flash_decode._STATS.update(fired=fired, gated=gated,
                               windows_read=48 if fired else 0,
                               windows_active=190 if fired else 0)
    r = flash_decode.gate_report(expect)
    assert r["verdict"] == verdict and r["ok"] is ok and r["expected"] is expect
    flash_decode.reset_stats()


def test_gate_report_never_raises_and_always_records():
    """A quality run that has finished generating must record what happened
    rather than lose the work to an assertion."""
    flash_decode.reset_stats()
    for expect in (True, False):
        r = flash_decode.gate_report(expect)
        assert set(r) >= {"verdict", "ok", "expected", "fired", "gated",
                          "read_fraction"}
