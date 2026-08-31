"""The prefill memory model is only worth trusting where it reproduces reality.

These tests pin it against the one A100 run we actually have numbers for
(Meta-Llama-3-8B-Instruct, fp16, batch 32) and against the pass/fail pattern of
the whole decode table. They run on CPU: the point is to catch a wrong
prediction here rather than by burning another GPU hour on a shape that was
never going to fit.
"""

from __future__ import annotations

import types

import pytest

from modules.evaluation.memory_model import (
    CALIBRATION,
    lse_recompute_gb,
    predict_prefill_peak,
    prefill_kv_gb,
    weights_gb,
)


@pytest.fixture
def llama3_8b():
    """Meta-Llama-3-8B-Instruct geometry (configs/eval_perf.yaml)."""
    return types.SimpleNamespace(
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
        hidden_size=4096, intermediate_size=14336, vocab_size=128256,
        head_dim=128,
    )


#: 8.03B params — the published Llama-3-8B count.
N_PARAMS = 8_030_261_248


def _predict(cfg, batch, prefill, **kw):
    kw.setdefault("device_total_gb", 79.25)
    kw.setdefault("n_params", N_PARAMS)
    return predict_prefill_peak(cfg, batch, prefill, **kw)


class TestAgainstTheLoggedRun:
    """Terms the 2026-08-30 run reported directly."""

    def test_lse_transient_matches_the_logged_figure(self, llama3_8b):
        """The run logged '~4.1 GB' for the compute_lse transient at 1048/32."""
        got = lse_recompute_gb(llama3_8b, 32, 1048, chunk=1024,
                               noninplace_factor=1.0)
        assert got == pytest.approx(4.1, abs=0.1)

    def test_derived_weights_sit_just_below_the_end_of_run_allocation(
            self, llama3_8b):
        """End-of-run alloc is weights PLUS the compacted KV, so it must exceed
        the derived weights by a small, positive margin — not equal it."""
        w = weights_gb(llama3_8b, n_params=N_PARAMS)
        end = CALIBRATION["measured_end_alloc_gb"]
        assert w < end
        assert end - w < 2.0, "residual is meant to be the compacted KV, not a bug"

    def test_reproduces_BOTH_measured_points_at_the_calibration_shape(
            self, llama3_8b):
        """1048/batch-32 was measured twice — L recomputed and L reused — and
        the two differ only by the compute_lse transient. That makes the system
        exactly determined, so the model must hit BOTH, not just the one it was
        fitted on."""
        off = _predict(llama3_8b, 32, 1048, lse_recomputes=True)
        on = _predict(llama3_8b, 32, 1048, lse_recomputes=False)
        assert off.alloc_total == pytest.approx(
            CALIBRATION["measured_alloc_peak_gb"], abs=0.05)
        assert on.alloc_total == pytest.approx(
            CALIBRATION["measured_alloc_peak_lse_reused_gb"], abs=0.05)

    def test_the_transient_is_measured_not_assumed(self, llama3_8b):
        """NONINPLACE_FACTOR was a guessed 2.0; the two points give 1.10. Pin
        that it is derived from the measurements, not re-guessed."""
        from modules.evaluation.memory_model import NONINPLACE_FACTOR
        measured = (CALIBRATION["measured_alloc_peak_gb"]
                    - CALIBRATION["measured_alloc_peak_lse_reused_gb"])
        one_block = lse_recompute_gb(llama3_8b, 32, 1048, noninplace_factor=1.0)
        assert NONINPLACE_FACTOR == pytest.approx(measured / one_block, rel=1e-3)
        assert 1.0 <= NONINPLACE_FACTOR < 1.5, "a guess crept back in"

    def test_reproduces_its_own_calibration_point(self, llama3_8b):
        """The model must return the measured peak at the shape it was fitted
        on. It did not, before the fit and the prediction were made to share a
        noninplace_factor and a weights figure."""
        b = _predict(llama3_8b, CALIBRATION["batch_size"],
                     CALIBRATION["prefill_len"])
        assert b.alloc_total == pytest.approx(
            CALIBRATION["measured_alloc_peak_gb"], abs=0.05)
        assert b.device_total_used == pytest.approx(
            CALIBRATION["measured_device_peak_gb"], abs=0.05)


class TestReproducesTheDecodeTable:
    """Every cell's observed outcome, from the run that produced the table."""

    @pytest.mark.parametrize("prefill,batch,observed", [
        (1048, 1, "ran"),
        (1048, 32, "ran"),
        (2048, 1, "ran"),
        (2048, 32, "ran"),
        (4096, 1, "ran"),
        (4096, 32, "oom"),
    ])
    def test_cell_outcome(self, llama3_8b, prefill, batch, observed):
        v = _predict(llama3_8b, batch, prefill).verdict
        if observed == "oom":
            assert v in ("oom", "marginal"), (
                f"{prefill}/{batch} OOMed on the A100; model says {v!r}")
        else:
            assert v in ("fits", "marginal"), (
                f"{prefill}/{batch} ran on the A100; model says {v!r}")

    def test_only_the_batch32_4096_cell_is_predicted_to_fail(self, llama3_8b):
        """The failure is specific. A model that predicted trouble everywhere
        would pass the parametrized test above by accident."""
        over = _predict(llama3_8b, 32, 4096)
        assert over.verdict != "fits"
        assert over.device_total_used > over.device_total
        assert _predict(llama3_8b, 1, 4096).verdict == "fits"


class TestTheLeverIsTheLseTransient:
    """The actionable claim: the wasted transient is what breaks 4096/32."""

    def test_lse_transient_exceeds_the_weights_at_the_ooming_shape(
            self, llama3_8b):
        b = _predict(llama3_8b, 32, 4096)
        assert b.lse_recompute > b.weights

    def test_working_l_reuse_turns_the_overrun_into_a_fit(self, llama3_8b):
        """Removing the term the L-reuse miss creates — changing nothing else.

        Deliberately NOT asserting the as-run cell predicts "oom": the model
        puts it 7 GB over an 80 GB card, which is a genuine overrun but only a
        9% one. Over the limit is what the model supports; certainty is not.
        """
        from modules.evaluation.memory_model import OVERHEAD_FRACTION
        broken = _predict(llama3_8b, 32, 4096)
        assert broken.device_total_used > broken.device_total
        fixed = _predict(llama3_8b, 32, 4096, lse_recomputes=False)
        assert fixed.lse_recompute == 0.0
        assert fixed.verdict == "fits"
        # The saving is exactly the transient plus its share of allocator
        # overhead — an identity, not a threshold. The previous version asserted
        # "> 30 GB", which silently encoded the 2.0 transient factor that the
        # two measured points later disproved; a magic number in an assertion
        # outlives the reasoning that produced it.
        saved = broken.device_total_used - fixed.device_total_used
        assert saved == pytest.approx(
            broken.lse_recompute * (1 + OVERHEAD_FRACTION), rel=1e-6)
        assert fixed.device_total < broken.device_total_used, (
            "the cell must be over the limit before and under it after")

    def test_smaller_chunk_also_clears_it_without_touching_logic(self, llama3_8b):
        """STICKYKV_PREFILL_SCORE_CHUNK is a pure env knob: same FLOPs, smaller
        blocks. It must clear the overrun on its own, L-reuse still broken."""
        broken = _predict(llama3_8b, 32, 4096, chunk=1024)
        assert broken.device_total_used > broken.device_total
        chunked = _predict(llama3_8b, 32, 4096, chunk=128)
        assert chunked.verdict == "fits"
        assert chunked.device_total_used < chunked.device_total

    def test_transient_scales_linearly_in_chunk_below_saturation(self, llama3_8b):
        a = lse_recompute_gb(llama3_8b, 32, 4096, chunk=1024)
        b = lse_recompute_gb(llama3_8b, 32, 4096, chunk=128)
        assert a / b == pytest.approx(8.0, rel=1e-6)

    def test_transient_saturates_once_chunk_exceeds_context(self, llama3_8b):
        """min(chunk, S): past S the block stops growing."""
        a = lse_recompute_gb(llama3_8b, 32, 512, chunk=1024)
        b = lse_recompute_gb(llama3_8b, 32, 512, chunk=4096)
        assert a == pytest.approx(b)


class TestGeometryTerms:
    def test_prefill_kv_is_the_full_uncompressed_cache(self, llama3_8b):
        """Eviction runs on decode step 0, so prefill holds every prompt token:
        2 x 32 layers x B x 8 kv-heads x S x 128 x fp16."""
        got = prefill_kv_gb(llama3_8b, 32, 1048)
        want = 2 * 32 * 32 * 8 * 1048 * 128 * 2 / (1024 ** 3)
        assert got == pytest.approx(want)

    def test_prefill_kv_is_linear_in_both_batch_and_context(self, llama3_8b):
        base = prefill_kv_gb(llama3_8b, 1, 1024)
        assert prefill_kv_gb(llama3_8b, 8, 1024) == pytest.approx(8 * base)
        assert prefill_kv_gb(llama3_8b, 1, 8192) == pytest.approx(8 * base)

    def test_derived_weights_match_the_published_param_count(self, llama3_8b):
        derived = weights_gb(llama3_8b)
        exact = weights_gb(llama3_8b, n_params=N_PARAMS)
        assert derived == pytest.approx(exact, rel=0.05)


class TestOomAutopsyCapturesTheCrucialFields:
    """The autopsy exists so the FIRST OOM is diagnostic. If it silently wrote
    nothing, or omitted the term that overflowed, the GPU hour is spent again.
    """

    def _run_autopsy(self, tmp_path, llama3_8b, monkeypatch):
        import types as _t
        from modules.evaluation.perf_runner import PerfRunner

        cfg = _t.SimpleNamespace(
            telemetry=_t.SimpleNamespace(output_dir=str(tmp_path)))
        runner = PerfRunner.__new__(PerfRunner)
        monkeypatch.setenv("STICKYKV_PREFILL_SCORE_CHUNK", "1024")
        runner._dump_oom_autopsy(
            cfg, 4096, 32, "ours_q0.70",
            RuntimeError("CUDA out of memory. Tried to allocate 16.00 GiB"),
            llama3_8b)
        written = list(tmp_path.glob("oom_autopsy_*.txt"))
        assert written, "the autopsy wrote no file"
        return written[0].read_text(encoding="utf-8")

    def test_writes_a_file_naming_the_cell(self, tmp_path, llama3_8b, monkeypatch):
        body = self._run_autopsy(tmp_path, llama3_8b, monkeypatch)
        assert "prefill=4096" in body and "batch=32" in body
        assert "ours_q0.70" in body

    def test_records_the_per_term_breakdown(self, tmp_path, llama3_8b, monkeypatch):
        """Naming the overflowing term is the whole point — 'OOM' alone is what
        we already had."""
        body = self._run_autopsy(tmp_path, llama3_8b, monkeypatch)
        for term in ("weights", "prefill KV", "compute_lse transient", "other"):
            assert term in body, f"breakdown omitted {term!r}"

    def test_records_the_l_reuse_state_and_the_chunk_knob(
            self, tmp_path, llama3_8b, monkeypatch):
        body = self._run_autopsy(tmp_path, llama3_8b, monkeypatch)
        assert "compute_lse calls this process" in body
        assert "STICKYKV_PREFILL_SCORE_CHUNK" in body

    def test_survives_a_missing_model_config(self, tmp_path, monkeypatch):
        """A failing autopsy must never mask the OOM it is explaining."""
        import types as _t
        from modules.evaluation.perf_runner import PerfRunner

        cfg = _t.SimpleNamespace(
            telemetry=_t.SimpleNamespace(output_dir=str(tmp_path)))
        runner = PerfRunner.__new__(PerfRunner)
        runner._dump_oom_autopsy(cfg, 4096, 32, "x", RuntimeError("oom"), None)
        assert list(tmp_path.glob("oom_autopsy_*.txt"))
