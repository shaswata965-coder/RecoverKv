"""Tests for utils.cache_memory — byte accounting and the memo-exclusive ratios.

The read memo is a derived fp16 copy of the Q tier. It is real resident memory,
so `total_live` counts it — but it is not cache state, and at B = 1 (where the
auto rule turns it ON) it is the largest line item by far. Measured on a RULER
example it was 16.3 MB of 21.4 MB, which pulls `compression_vs_fp16` to 0.91x:
the report claimed the two-tier cache was *bigger* than fp16, for a method whose
claim is ~3.9x. These pin the pair of ratios that keeps that legible.
"""
from __future__ import annotations

import time

import pytest
import torch

from utils.cache_memory import (
    CacheMemoryReport,
    MemoryProbe,
    format_peak_report,
    format_report,
    host_peak_rss_bytes,
    host_rss_bytes,
)


def _report(memo_bytes: int) -> CacheMemoryReport:
    """A report whose cache state is 4 MB and whose fp16 equivalent is 16 MB."""
    mb = 1024 * 1024
    return CacheMemoryReport(
        kind="windowed",
        num_layers=2,
        batch_size=1,
        device="cpu",
        kv_dtype="torch.float16",
        window_size=8,
        observed_context_len=512,
        retained_tokens=128,
        fp_live_tokens=64,
        fp_alloc_tokens=64,
        q_active_tokens=64,
        q_active_windows=8,
        q_slots=10,
        fp_content_live=3 * mb,
        fp_content_alloc=3 * mb,
        q_content_live=1 * mb,
        q_content_alloc=1 * mb,
        bookkeeping_live=0,
        bookkeeping_alloc=0,
        memo_bytes=memo_bytes,
        total_live=4 * mb + memo_bytes,
        total_alloc=4 * mb + memo_bytes,
        fp16_equiv_retained=16 * mb,
        dense_full_context=32 * mb,
        resolved={
            "quant_ratio": 0.5, "window_size": 8, "num_sink_tokens": 4,
            "local_tokens": 16, "top_k_fp": 2, "N_q": 8,
            "quant_memoize_read": bool(memo_bytes),
        },
    )


class TestMemoExclusiveRatios:
    def test_memo_inverts_the_headline_ratio(self):
        """The failure this exists to make visible, not a hypothetical."""
        r = _report(memo_bytes=16 * 1024 * 1024)
        assert r.compression_vs_fp16 == pytest.approx(0.8, rel=1e-6)
        assert r.compression_vs_fp16 < 1.0          # "bigger than fp16"
        assert r.compression_vs_fp16_excl_memo == pytest.approx(4.0, rel=1e-6)

    def test_ratios_agree_when_the_memo_is_off(self):
        """With memoization off there is one number, not two."""
        r = _report(memo_bytes=0)
        assert r.total_live_excl_memo == r.total_live
        assert r.compression_vs_fp16 == r.compression_vs_fp16_excl_memo
        assert r.reduction_vs_full == r.reduction_vs_full_excl_memo

    def test_to_dict_carries_both_framings(self):
        """The jsonl artifact must let a reader recompute either number."""
        d = _report(memo_bytes=16 * 1024 * 1024).to_dict()
        for key in (
            "compression_vs_fp16",
            "compression_vs_fp16_excl_memo",
            "reduction_vs_full",
            "reduction_vs_full_excl_memo",
            "total_live_excl_memo",
            "memo_bytes",
        ):
            assert key in d, f"{key} missing from the memory report dict"

    def test_formatted_report_flags_a_dominant_memo(self):
        text = format_report(_report(memo_bytes=16 * 1024 * 1024))
        assert "read memo is 80% of TOTAL live" in text
        assert "3.99x vs fp16 at same retention" in text or "4.00x" in text
        assert "quant_memoize_read: false" in text

    def test_formatted_report_is_quiet_without_a_memo(self):
        text = format_report(_report(memo_bytes=0))
        assert "read memo is" not in text


# ---------------------------------------------------------------------------
# MemoryProbe — the attachable high-water recorder
# ---------------------------------------------------------------------------


class _Tiny(torch.nn.Module):
    """A module whose forward allocates, so a peak exists to be recorded."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(32, 32)

    def forward(self, x):
        return self.lin(x).relu()


class TestHostMemory:
    def test_rss_is_available_without_psutil(self):
        """psutil is NOT in environment.yml. A memory tool that reports nothing
        on the default env is not a memory tool, so both fall back to the OS."""
        assert host_rss_bytes() > 0
        assert host_peak_rss_bytes() > 0

    def test_peak_rss_is_at_least_current(self):
        assert host_peak_rss_bytes() >= host_rss_bytes() * 0.5


class TestMemoryProbe:
    def test_context_manager_produces_a_report(self):
        with MemoryProbe(label="unit", poll_interval_s=None) as probe:
            torch.randn(256, 256)
        r = probe.report()
        assert r.label == "unit"
        assert r.duration_s >= 0.0
        assert r.samples >= 2                      # start + stop, at minimum
        assert r.host_rss_peak > 0

    def test_phases_are_recorded_in_order(self):
        with MemoryProbe(poll_interval_s=None) as probe:
            with probe.phase("prefill"):
                torch.randn(512, 512)
            with probe.phase("decode"):
                torch.randn(64, 64)
        r = probe.report()
        assert [p.name for p in r.phases] == ["prefill", "decode"]
        assert all(p.seconds >= 0.0 for p in r.phases)
        assert all(p.samples >= 2 for p in r.phases)

    def test_entering_a_phase_closes_the_previous_one(self):
        """Phases do not nest — a forgotten exit must not swallow the next span."""
        probe = MemoryProbe(poll_interval_s=None).start()
        probe._begin_phase("a")
        probe._begin_phase("b")
        r = probe.stop().report()
        assert [p.name for p in r.phases] == ["a", "b"]

    def test_report_is_idempotent(self):
        """The perf runner reports once per run and then again at save time."""
        with MemoryProbe(poll_interval_s=None) as probe:
            with probe.phase("only"):
                pass
        first = probe.report()
        second = probe.report()
        assert len(first.phases) == len(second.phases) == 1
        assert first.duration_s == second.duration_s

    def test_attach_samples_once_per_forward(self):
        model = _Tiny()
        probe = MemoryProbe(poll_interval_s=None).start()
        probe.attach(model)
        before = probe.report().samples
        for _ in range(5):
            model(torch.randn(4, 32))
        after = probe.report().samples
        assert after - before >= 5
        probe.stop()
        # Detached on stop: further forwards must not keep sampling.
        settled = probe.report().samples
        model(torch.randn(4, 32))
        assert probe.report().samples == settled

    def test_model_kwarg_attaches_and_detaches(self):
        model = _Tiny()
        with MemoryProbe(model=model, poll_interval_s=None) as probe:
            model(torch.randn(4, 32))
        assert probe.report().hooked is True
        assert probe._hook is None, "hook outlived the probe's context"

    def test_poller_samples_between_forwards(self):
        """The poller is what catches a peak that no forward boundary brackets."""
        with MemoryProbe(poll_interval_s=0.01) as probe:
            time.sleep(0.15)
        assert probe.report().samples > 2

    def test_stop_is_idempotent(self):
        probe = MemoryProbe(poll_interval_s=None).start()
        probe.stop()
        d1 = probe.report().duration_s
        probe.stop()
        assert probe.report().duration_s == d1

    def test_to_dict_carries_the_peak_fields(self):
        with MemoryProbe(poll_interval_s=None) as probe:
            with probe.phase("prefill"):
                pass
        d = probe.to_dict()
        for key in (
            "torch_alloc_peak", "torch_reserved_peak", "device_used_peak",
            "device_total", "host_rss_peak", "num_ooms", "num_alloc_retries",
            "fragmentation_peak", "device_headroom", "device_utilization",
            "peak_phase", "phases",
        ):
            assert key in d, f"{key} missing from the peak report dict"
        assert d["phases"][0]["name"] == "prefill"

    def test_cpu_report_has_no_device_figures(self):
        """On CPU the GPU columns are absent rather than zero-and-misleading."""
        with MemoryProbe(device=torch.device("cpu"), poll_interval_s=None) as probe:
            pass
        r = probe.report()
        assert r.device_total == 0
        assert r.device_utilization is None
        text = format_peak_report(r)
        assert "GPU device used" not in text
        assert "host RSS" in text

    def test_peak_phase_picks_the_larger_phase(self):
        """Which phase holds the peak decides whether compression can raise
        max-B at all, so it must not be guesswork."""
        with MemoryProbe(poll_interval_s=None) as probe:
            with probe.phase("prefill"):
                pass
            with probe.phase("decode"):
                pass
        r = probe.report()
        r.phases[0].torch_alloc_peak = 900
        r.phases[0].host_rss_peak = 900
        r.phases[1].torch_alloc_peak = 100
        r.phases[1].host_rss_peak = 100
        assert r.peak_phase == "prefill"
        # With a device measured, the ranking carries a max-B verdict...
        r.device_total = 80 * 1024**3
        r.phases[0].device_used_peak = 900
        r.phases[1].device_used_peak = 100
        assert "UN-EVICTED prompt" in format_peak_report(r)

    def test_cpu_phase_ranking_does_not_claim_a_max_b_verdict(self):
        """...but on CPU the only phase signal is RSS, which is dominated by the
        weights and never falls. It ranks the phases honestly and means nothing
        for batch capacity, so the verdict text must not appear."""
        with MemoryProbe(device=torch.device("cpu"), poll_interval_s=None) as probe:
            with probe.phase("prefill"):
                pass
            with probe.phase("decode"):
                pass
        text = format_peak_report(probe.report())
        assert "NOT a max-B signal" in text
        assert "UN-EVICTED prompt" not in text
        assert "cache_budget maps ~1:1" not in text

    def test_mid_run_report_does_not_close_an_open_phase(self):
        """A progress report must not truncate the span it is reporting on."""
        probe = MemoryProbe(poll_interval_s=None).start()
        with probe.phase("decode"):
            assert probe.report().phases == []      # not closed by reporting
        assert [p.name for p in probe.report().phases] == ["decode"]
        probe.stop()

    def test_stop_closes_a_dangling_phase(self):
        """A phase entered without its context manager still lands in the report."""
        probe = MemoryProbe(poll_interval_s=None).start()
        probe._begin_phase("prefill")
        probe.stop()
        assert [p.name for p in probe.report().phases] == ["prefill"]
