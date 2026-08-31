"""The bisection ladder has to be trustworthy before its deltas mean anything.

Two failure modes would silently invalidate a whole GPU session and look like a
result rather than a bug: an env var leaking from one rung into the next (so a
later rung measures a path it did not select), and a rung that raises instead of
recording (so the run dies and localises nothing). Both are tested here on CPU.
"""

from __future__ import annotations

import os

import pytest
import torch

from scripts.audit_e2e import RUNGS, _report, _time_rung


class TestRungDefinitions:
    def test_rungs_are_unique_and_ordered_by_what_they_add(self):
        ids = [r["id"] for r in RUNGS]
        assert len(set(ids)) == len(ids)
        assert ids == sorted(ids), "ids are numbered so the ladder reads in order"

    def test_the_floor_runs_none_of_our_code(self):
        floor = RUNGS[0]
        assert floor["windowed"] is False
        assert floor["q"] == 0.0
        assert floor["env"] == {}

    def test_each_rung_changes_exactly_one_thing_from_the_previous(self):
        """The ladder only attributes cost if consecutive rungs differ by one
        layer. Two changes in a gap makes that gap uninterpretable."""
        for a, b in zip(RUNGS, RUNGS[1:]):
            changed = sum([
                a["windowed"] != b["windowed"],
                a["q"] != b["q"],
                a["env"] != b["env"] and a["q"] == b["q"],
            ])
            assert changed >= 1, f"{a['id']} -> {b['id']} adds nothing"
            assert changed <= 1, (
                f"{a['id']} -> {b['id']} changes {changed} things; the gap "
                "cannot be attributed to a single layer")

    def test_the_top_rung_is_the_shipped_configuration(self):
        top = RUNGS[-1]
        assert top["windowed"] is True
        assert top["q"] > 0
        assert top["env"].get("STICKYKV_FUSED_DECODE") == "1"

    def test_every_rung_documents_what_it_adds(self):
        for r in RUNGS:
            assert r.get("what"), f"{r['id']} has no description"


class _Exploding:
    """A model that fails on the prefill call."""

    def __init__(self, exc):
        self._exc = exc

    def __call__(self, *a, **k):
        raise self._exc


@pytest.fixture
def cfg():
    import types
    return types.SimpleNamespace(
        window=types.SimpleNamespace(window_size=8, num_sink_tokens=5,
                                     local_window_size=64),
        cache=types.SimpleNamespace(cache_budget=0.20, rerotate_on_evict=False),
    )


def _run_floor(cfg, exc, env):
    rung = dict(RUNGS[0], env=env)
    ids = torch.zeros((1, 8), dtype=torch.long)
    return _time_rung(torch, _Exploding(exc), ids, rung, cfg, 8, 4,
                      torch.float16, "flash_attn", 1)


class TestFailureIsRecordedNotRaised:
    def test_a_failing_rung_records_instead_of_killing_the_ladder(self, cfg):
        rec = _run_floor(cfg, RuntimeError("boom"), {})
        assert "failed" in rec and "boom" in rec["failed"]
        assert "traceback" in rec and "RuntimeError" in rec["traceback"]

    def test_the_record_still_identifies_the_rung(self, cfg):
        rec = _run_floor(cfg, RuntimeError("boom"), {})
        assert rec["id"] == RUNGS[0]["id"]
        assert rec["what"]

    def test_an_oom_is_recorded_like_any_other_failure(self, cfg):
        rec = _run_floor(cfg, RuntimeError("CUDA out of memory"), {})
        assert "out of memory" in rec["failed"]

    def test_report_survives_a_ladder_with_failed_rungs(self, cfg, capsys):
        good = {"id": "0_baseline", "what": "floor", "ttft_s": 0.1,
                "tpot_steady_s": 0.02, "peak_alloc_gb": 16.0}
        bad = {"id": "3_q70_fused", "what": "shipped", "failed": "OOM"}
        _report([good, bad], batch=32)
        out = capsys.readouterr().out
        assert "FAILED" in out and "0_baseline" in out


class TestEnvIsolationBetweenRungs:
    """A leaked env var makes a later rung measure a path it did not select —
    and the number still looks like a measurement."""

    def test_env_is_restored_after_a_rung_that_set_it(self, cfg, monkeypatch):
        monkeypatch.delenv("STICKYKV_FUSED_DECODE", raising=False)
        _run_floor(cfg, RuntimeError("x"), {"STICKYKV_FUSED_DECODE": "0"})
        assert "STICKYKV_FUSED_DECODE" not in os.environ

    def test_a_preexisting_value_is_restored_not_cleared(self, cfg, monkeypatch):
        monkeypatch.setenv("STICKYKV_FUSED_DECODE", "1")
        _run_floor(cfg, RuntimeError("x"), {"STICKYKV_FUSED_DECODE": "0"})
        assert os.environ["STICKYKV_FUSED_DECODE"] == "1"

    def test_env_is_restored_even_when_the_rung_fails(self, cfg, monkeypatch):
        """The failing rung is exactly when a leak would go unnoticed."""
        monkeypatch.setenv("STICKYKV_SCORE_LSE_FROM_FORWARD", "1")
        _run_floor(cfg, MemoryError("oom"),
                   {"STICKYKV_SCORE_LSE_FROM_FORWARD": "0"})
        assert os.environ["STICKYKV_SCORE_LSE_FROM_FORWARD"] == "1"

    def test_rung_env_is_applied_during_the_rung(self, cfg, monkeypatch):
        """Sanity: the var must actually be set while the rung runs, or the
        isolation tests above would pass on a no-op."""
        seen = {}

        class _Watcher:
            def __call__(self, *a, **k):
                seen["v"] = os.environ.get("STICKYKV_FUSED_DECODE")
                raise RuntimeError("stop")

        rung = dict(RUNGS[0], env={"STICKYKV_FUSED_DECODE": "0"})
        ids = torch.zeros((1, 8), dtype=torch.long)
        _time_rung(torch, _Watcher(), ids, rung, cfg, 8, 4, torch.float16,
                   "flash_attn", 1)
        assert seen["v"] == "0"
