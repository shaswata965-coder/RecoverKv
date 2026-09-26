"""The decode table must not present stale cells and silent errors as results.

Both failures happened on real runs and both produced a table that looked
complete:

* a cell that ERRORED printed five dashes -- indistinguishable from "no data" --
  so a deliberate hard error (the strict L-reuse miss) read as an empty row
  while the reason sat unread in the npz;
* the printer globs the WHOLE npz directory, so re-measuring one shape left
  every other row showing its previous run's number with nothing on screen to
  say so. Six August cells and one fresh cell rendered as one table, and the
  August TTFTs were read as the current ones;
* its one throughput column was the npz's legacy ``throughput_tokps``, which is
  neither the decode rate nor the e2e rate. A reader took a prompt-dominated
  B=32 cell for a decode figure and it was understated 1.5x, because 11.7 s of
  TTFT was sitting in the denominator. The corrected columns are asserted here.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from scripts.print_perf_table import STALE_S, build_table


def _write_cell(d, prefill, gen, bs, *, name="ours_q0.70", ttft=0.5,
                oom=False, err=False, reason="", age_s=0.0,
                tpot_steady=85.0, throughput=11.7,
                decode_tps=None, e2e_tps=None, e2e_ms=None):
    """One perf npz, shaped like the runner's, with a controllable mtime.

    ``decode_tps`` / ``e2e_tps`` / ``e2e_ms`` default to absent, which is what an
    npz written before fbcf368 looks like: the corrected rates have to be derived
    from ``tpot_steady_ms`` / ``e2e_latency_ms`` or reported as unavailable.
    """
    path = d / f"perf_prefill{prefill}_gen{gen}_bs{bs}.npz"
    one = np.array([[ttft * 1000.0]])
    optional = {}
    if tpot_steady is not None:
        optional["tpot_steady_ms"] = np.array([[tpot_steady]])
    if decode_tps is not None:
        optional["throughput_decode_tokps"] = np.array([[decode_tps]])
    if e2e_tps is not None:
        optional["throughput_e2e_tokps"] = np.array([[e2e_tps]])
    if e2e_ms is not None:
        optional["e2e_latency_ms"] = np.array([[e2e_ms]])
    np.savez_compressed(
        path,
        config_names=np.array([name], dtype=object),
        ttft_ms=one,
        throughput_tokps=np.array([[throughput]]),
        peak_memory_mb=np.array([[16000.0]]),
        peak_decode_steady_mb=np.array([[17000.0]]),
        oom_mask=np.array([oom]),
        error_mask=np.array([err]),
        skipped_mask=np.array([oom or err]),
        skip_reason=np.array([reason], dtype=object),
        **optional,
    )
    if age_s:
        old = time.time() - age_s
        os.utime(path, (old, old))
    return path


class TestErroredCellsAreNotDashes:
    def test_an_errored_cell_says_ERROR(self, tmp_path):
        _write_cell(tmp_path, 1048, 9, 1, err=True, reason="RuntimeError: boom")
        out = build_table(tmp_path, None, "median")
        assert "ERROR" in out

    def test_the_reason_is_printed(self, tmp_path):
        _write_cell(tmp_path, 1048, 9, 1, err=True,
                    reason="RuntimeError: L-reuse MISS at layer 0")
        out = build_table(tmp_path, None, "median")
        assert "L-reuse MISS at layer 0" in out

    def test_an_oom_cell_also_carries_its_reason(self, tmp_path):
        _write_cell(tmp_path, 4096, 257, 32, oom=True, reason="oom")
        out = build_table(tmp_path, None, "median")
        assert "OOM" in out and "4096/257" in out

    def test_a_missing_reason_says_so_rather_than_printing_blank(self, tmp_path):
        _write_cell(tmp_path, 1048, 9, 1, err=True, reason="")
        out = build_table(tmp_path, None, "median")
        assert "no reason recorded" in out

    def test_a_good_cell_is_untouched(self, tmp_path):
        _write_cell(tmp_path, 1048, 1049, 1, ttft=0.212)
        out = build_table(tmp_path, None, "median")
        assert "0.212" in out
        assert "ERROR" not in out


class TestStaleCellsAreCalledOut:
    """Reproduces the reported table exactly: six cells from a previous run
    plus one fresh cell, presented as a single result."""

    @pytest.fixture
    def mixed(self, tmp_path):
        old = STALE_S * 24 * 30  # ~a month, as in the real case
        _write_cell(tmp_path, 4096, 257, 1, ttft=1.132, age_s=old)
        _write_cell(tmp_path, 4096, 257, 32, oom=True, reason="oom", age_s=old)
        _write_cell(tmp_path, 2048, 513, 1, ttft=0.419, age_s=old)
        _write_cell(tmp_path, 2048, 513, 32, ttft=11.267, age_s=old)
        _write_cell(tmp_path, 1048, 1049, 1, ttft=0.212, age_s=old)
        _write_cell(tmp_path, 1048, 1049, 32, ttft=4.343, age_s=old)
        _write_cell(tmp_path, 1048, 9, 1, err=True,
                    reason="RuntimeError: FlashInfer L-capture failed")
        return tmp_path

    def test_the_table_warns_that_it_mixes_runs(self, mixed):
        out = build_table(mixed, None, "median")
        assert "MIXED RUNS" in out

    def test_it_counts_the_stale_cells(self, mixed):
        out = build_table(mixed, None, "median")
        assert "6 of 7" in out

    def test_it_names_which_rows_are_stale(self, mixed):
        out = build_table(mixed, None, "median")
        for shape in ("4096/257", "2048/513", "1048/1049"):
            assert shape in out
        stale_block = out[out.index("MIXED RUNS"):]
        assert "1048/9" not in stale_block, "the fresh cell is not stale"

    def test_the_fresh_error_is_still_reported(self, mixed):
        out = build_table(mixed, None, "median")
        assert "FlashInfer L-capture failed" in out

    def test_a_single_run_produces_no_warning(self, tmp_path):
        """The warning must not fire on an ordinary run, or it gets ignored."""
        _write_cell(tmp_path, 1048, 1049, 1, ttft=0.212)
        _write_cell(tmp_path, 1048, 1049, 32, ttft=4.343)
        out = build_table(tmp_path, None, "median")
        assert "MIXED RUNS" not in out

    def test_no_warning_for_a_single_cell(self, tmp_path):
        _write_cell(tmp_path, 1048, 9, 1, ttft=0.2)
        assert "MIXED RUNS" not in build_table(tmp_path, None, "median")


class TestThroughputColumns:
    """The headline column has to be a quantity you can name.

    ``throughput_tokps`` is prefill-inclusive AND omits the prefill->decode gap,
    so it is neither claim; perf_runner records ``throughput_decode_tokps`` and
    ``throughput_e2e_tokps`` and documents the legacy field as not-to-be-quoted.
    """

    #: the exact header the table shipped with, which `legacy` must reproduce.
    LEGACY_HEADER = ("shape        batch    TTFT (s)   TPOT_steady (s)   "
                     "throughput (tok/s)    peak_GB   steadyKV_GB")

    def test_default_prints_both_corrected_columns(self, tmp_path):
        _write_cell(tmp_path, 4096, 257, 32)
        out = build_table(tmp_path, None, "median")
        assert "dec tok/s" in out and "e2e tok/s" in out
        assert "throughput (tok/s)" not in out

    def test_default_does_not_quote_the_legacy_number(self, tmp_path):
        """The regression itself: the headline column was the legacy field."""
        _write_cell(tmp_path, 4096, 257, 32, throughput=230.1, tpot_steady=92.5)
        out = build_table(tmp_path, None, "median")
        assert "230.1" not in out
        assert "345.9" in out, "batch 32 / 92.5 ms = 345.9 tok/s"

    def test_decode_prefers_the_runners_own_field(self, tmp_path):
        """A recorded rate is quoted, not recomputed behind the runner's back."""
        _write_cell(tmp_path, 4096, 257, 32, tpot_steady=92.5, decode_tps=333.3)
        out = build_table(tmp_path, None, "median", "decode")
        assert "333.3" in out and "345.9" not in out
        assert "derived" not in out

    def test_decode_is_derived_from_steady_tpot_when_absent(self, tmp_path):
        _write_cell(tmp_path, 4096, 257, 32, tpot_steady=92.5)
        out = build_table(tmp_path, None, "median", "decode")
        assert "345.9" in out
        assert "dec tok/s derived as batch / TPOT_steady" in out

    def test_e2e_prefers_the_runners_own_field(self, tmp_path):
        _write_cell(tmp_path, 4096, 257, 32, e2e_tps=229.3, e2e_ms=1.0)
        out = build_table(tmp_path, None, "median", "e2e")
        assert "229.3" in out

    def test_e2e_is_derived_from_e2e_latency_when_absent(self, tmp_path):
        _write_cell(tmp_path, 4096, 257, 32, e2e_ms=35_860.0)
        out = build_table(tmp_path, None, "median", "e2e")
        assert "229.3" in out, "32*257 tokens / 35.86 s = 229.3 tok/s"
        assert "e2e tok/s derived as batch*gen_len / e2e_latency" in out

    def test_e2e_never_substitutes_the_legacy_field(self, tmp_path):
        """The substitution that would re-introduce the bug, with a new label."""
        _write_cell(tmp_path, 4096, 257, 32, throughput=230.1, e2e_ms=None)
        out = build_table(tmp_path, None, "median", "e2e")
        assert "230.1" not in out
        assert "e2e tok/s unavailable" in out
        assert "NOT substituted" in out

    def test_a_missing_denominator_prints_a_dash_not_an_infinity(self, tmp_path):
        """A zero duration is a missing measurement, not an infinite rate."""
        _write_cell(tmp_path, 4096, 257, 32, tpot_steady=0.0)
        out = build_table(tmp_path, None, "median", "decode")
        assert "inf" not in out.lower()
        assert "dec tok/s unavailable" in out

    def test_legacy_reproduces_the_original_header(self, tmp_path):
        """So a table printed before this change can be identified exactly."""
        _write_cell(tmp_path, 4096, 257, 32, throughput=230.1)
        out = build_table(tmp_path, None, "median", "legacy")
        assert out.splitlines()[0] == self.LEGACY_HEADER

    def test_legacy_adds_no_footnote_of_its_own(self, tmp_path):
        """Byte-identity with the pre-change printer is the mode's whole point,
        so a cell with no stored rate prints "-" and nothing else."""
        _write_cell(tmp_path, 4096, 257, 32, throughput=float("nan"))
        out = build_table(tmp_path, None, "median", "legacy")
        assert "note:" not in out

    def test_legacy_prints_the_stored_field(self, tmp_path):
        _write_cell(tmp_path, 4096, 257, 32, throughput=230.1, tpot_steady=92.5)
        out = build_table(tmp_path, None, "median", "legacy")
        assert "230.1" in out and "345.9" not in out

    def test_all_puts_the_three_rates_side_by_side(self, tmp_path):
        _write_cell(tmp_path, 4096, 257, 32, throughput=230.1, tpot_steady=92.5,
                    e2e_ms=35_860.0)
        out = build_table(tmp_path, None, "median", "all")
        for value in ("345.9", "229.3", "230.1"):
            assert value in out

    def test_an_oom_row_still_dashes_every_added_column(self, tmp_path):
        """One dash per column -- a short row would silently shift the header."""
        _write_cell(tmp_path, 4096, 257, 32, oom=True, reason="oom")
        row = [ln for ln in build_table(tmp_path, None, "median", "all").splitlines()
               if ln.startswith("4096/257")][0]
        assert row.split()[2] == "OOM"
        # 9 columns in `all`, minus shape, batch and the OOM marker itself.
        assert row.split().count("-") == 6, row
