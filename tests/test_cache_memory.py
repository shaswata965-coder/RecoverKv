"""Tests for utils.cache_memory — byte accounting and the memo-exclusive ratios.

The read memo is a derived fp16 copy of the Q tier. It is real resident memory,
so `total_live` counts it — but it is not cache state, and at B = 1 (where the
auto rule turns it ON) it is the largest line item by far. Measured on a RULER
example it was 16.3 MB of 21.4 MB, which pulls `compression_vs_fp16` to 0.91x:
the report claimed the two-tier cache was *bigger* than fp16, for a method whose
claim is ~3.9x. These pin the pair of ratios that keeps that legible.
"""
from __future__ import annotations

import pytest

from utils.cache_memory import CacheMemoryReport, format_report


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
