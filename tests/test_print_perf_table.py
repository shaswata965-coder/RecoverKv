"""The decode table must not present stale cells and silent errors as results.

Both failures happened on real runs and both produced a table that looked
complete:

* a cell that ERRORED printed five dashes -- indistinguishable from "no data" --
  so a deliberate hard error (the strict L-reuse miss) read as an empty row
  while the reason sat unread in the npz;
* the printer globs the WHOLE npz directory, so re-measuring one shape left
  every other row showing its previous run's number with nothing on screen to
  say so. Six August cells and one fresh cell rendered as one table, and the
  August TTFTs were read as the current ones.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from scripts.print_perf_table import STALE_S, build_table


def _write_cell(d, prefill, gen, bs, *, name="ours_q0.70", ttft=0.5,
                oom=False, err=False, reason="", age_s=0.0):
    """One perf npz, shaped like the runner's, with a controllable mtime."""
    path = d / f"perf_prefill{prefill}_gen{gen}_bs{bs}.npz"
    one = np.array([[ttft * 1000.0]])
    np.savez_compressed(
        path,
        config_names=np.array([name], dtype=object),
        ttft_ms=one,
        tpot_steady_ms=np.array([[85.0]]),
        throughput_tokps=np.array([[11.7]]),
        peak_memory_mb=np.array([[16000.0]]),
        peak_decode_steady_mb=np.array([[17000.0]]),
        oom_mask=np.array([oom]),
        error_mask=np.array([err]),
        skipped_mask=np.array([oom or err]),
        skip_reason=np.array([reason], dtype=object),
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
