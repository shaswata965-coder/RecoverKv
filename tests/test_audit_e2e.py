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


class TestComputeLseMarkerIsUnambiguous:
    """`aten::logsumexp` is the only thing that isolates compute_lse — no env
    switch does, because L-reuse currently misses in BOTH arms of its A/B. If a
    second logsumexp ever lands in the hot path, the attribution silently starts
    over-counting and the trace still looks authoritative."""

    def test_torch_logsumexp_occurs_exactly_once_in_modules(self):
        import pathlib
        import re

        hits = []
        for p in pathlib.Path("modules").rglob("*.py"):
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                if re.search(r"(torch\.logsumexp|\.logsumexp)\s*\(", code):
                    hits.append(f"{p}:{i}")
        assert len(hits) == 1, (
            f"the compute_lse marker is no longer unique: {hits}. Update "
            "_LSE_MARKER in scripts/audit_e2e.py or the attribution over-counts.")
        assert "score_kernel" in hits[0]

    def test_the_marker_constant_matches_that_op(self):
        from scripts.audit_e2e import _LSE_CHAIN, _LSE_MARKER
        assert _LSE_MARKER == "aten::logsumexp"
        assert _LSE_MARKER in _LSE_CHAIN


class TestProfileRungEnvIsolation:
    """_profile_rung sets the same env as _time_rung and must restore it the
    same way — a leak here mislabels the attribution, not just the timing."""

    def test_env_restored_after_a_failing_profile_rung(self, cfg, monkeypatch):
        import torch as _torch
        from scripts.audit_e2e import RUNGS, _profile_rung

        monkeypatch.setenv("STICKYKV_FUSED_DECODE", "1")
        rung = dict(RUNGS[0], env={"STICKYKV_FUSED_DECODE": "0"})
        ids = _torch.zeros((1, 8), dtype=_torch.long)
        with pytest.raises(Exception):
            _profile_rung(_torch, _Exploding(RuntimeError("boom")), ids, rung,
                          cfg, 8, 4, _torch.float16, "flash_attn", 1, 2, 5)
        assert os.environ["STICKYKV_FUSED_DECODE"] == "1"


class TestLadderMatchesTheBenchmarkedMethod:
    """The ladder is only meaningful if its rungs run the SAME method the decode
    table ran. A divergence here does not raise — it produces deltas that
    quietly fail to add up, which is the most expensive kind of wrong.
    """

    @pytest.fixture
    def generated_cfg(self):
        """Mirrors scripts/run_perf_table.sh's generated YAML: window geometry
        under `window:`, budget and quant settings under `perf.configs[0]`."""
        import types
        return types.SimpleNamespace(
            window=types.SimpleNamespace(window_size=8, num_sink_tokens=5,
                                         local_window_size=64),
            cache=types.SimpleNamespace(quant_ratio=0.70,
                                        quant_budget_mode="tokens",
                                        first_eviction_step=0,
                                        rerotate_on_evict=False),
            perf=types.SimpleNamespace(configs=[{
                "name": "ours_q0.70", "cache_backend": "windowed",
                "cache_package": "flash_attn", "cache_budget": 0.20,
                "quant_ratio": 0.70, "quant_budget_mode": "tokens",
            }]),
        )

    def test_budget_comes_from_perf_configs_not_cfg_cache(self, generated_cfg):
        """cache_budget lives in perf.configs[0]; reading cfg.cache would give
        the 0.5 default and silently change the tier geometry."""
        from scripts.audit_e2e import resolve_cache_kwargs
        assert resolve_cache_kwargs(generated_cfg, 0.70)["cache_budget"] == 0.20

    def test_first_eviction_step_is_carried(self, generated_cfg):
        """first_eviction_step=0 puts the compaction on decode step 0. Dropping
        it changes which step carries the O(prefill) cost."""
        from scripts.audit_e2e import resolve_cache_kwargs
        assert resolve_cache_kwargs(generated_cfg, 0.70)["first_eviction_step"] == 0

    def test_quant_budget_mode_is_carried(self, generated_cfg):
        from scripts.audit_e2e import resolve_cache_kwargs
        assert resolve_cache_kwargs(generated_cfg, 0.70)["quant_budget_mode"] == "tokens"

    def test_window_geometry_comes_from_the_window_section(self, generated_cfg):
        from scripts.audit_e2e import resolve_cache_kwargs
        k = resolve_cache_kwargs(generated_cfg, 0.70)
        assert (k["window_size"], k["num_sink_tokens"], k["local_window_size"]) \
            == (8, 5, 64)

    def test_q_is_the_ladders_axis_and_overrides_the_config(self, generated_cfg):
        """Rung 1 must actually run q=0 even though the config says 0.70."""
        from scripts.audit_e2e import resolve_cache_kwargs
        assert resolve_cache_kwargs(generated_cfg, 0.0)["quant_ratio"] == 0.0
        assert resolve_cache_kwargs(generated_cfg, 0.70)["quant_ratio"] == 0.70

    def test_every_key_is_a_real_WindowedCacheConfig_field(self, generated_cfg):
        """A stale kwarg name would TypeError on the first GPU invocation,
        after the model load — the most annoying place to find out."""
        from scripts.audit_e2e import resolve_cache_kwargs
        from utils.cache_factory import get_cache_classes
        _, WCC, _ = get_cache_classes("flash_attn")
        fields = set(WCC.__dataclass_fields__)
        assert set(resolve_cache_kwargs(generated_cfg, 0.70)) <= fields

    def test_resolution_matches_perf_runners_own_field_list(self, generated_cfg):
        """Guards against perf_runner gaining a field the ladder does not set."""
        import re
        from scripts.audit_e2e import resolve_cache_kwargs
        src = open("modules/evaluation/perf_runner.py", encoding="utf-8").read()
        block = src[src.index("cc = WCC("):]
        block = block[:block.index(")\n")]
        used = set(re.findall(r"(\w+)\s*=", block))
        ours = set(resolve_cache_kwargs(generated_cfg, 0.70))
        missing = used - ours - {"cc", "WCC"}
        assert not missing, (
            f"perf_runner sets {sorted(missing)} and the ladder does not; the "
            "rungs would run a different method than the table.")


class TestEquivalenceReporting:
    """Rungs 2 and 3 differ only by STICKYKV_FUSED_DECODE, so they are one
    method computed two ways. The fused kernel is the CUDA default and has never
    been checked against its reference on a GPU (test_decode_kernel.py:109-120
    is a comment, not a test), so this report is the first evidence either way —
    it must not stay silent on a divergence."""

    def _rungs(self, mat, fus):
        return [
            {"id": "2_q70_materialize", "what": "m", "ttft_s": 1.0,
             "tpot_steady_s": 0.1, "first_tokens": mat},
            {"id": "3_q70_fused", "what": "f", "ttft_s": 1.0,
             "tpot_steady_s": 0.1, "first_tokens": fus},
        ]

    def test_identical_sequences_report_ok(self, capsys):
        from scripts.audit_e2e import _report_equivalence
        _report_equivalence(self._rungs([1, 2, 3], [1, 2, 3]))
        assert "OK" in capsys.readouterr().out

    def test_a_divergence_is_reported_loudly_with_the_index(self, capsys):
        from scripts.audit_e2e import _report_equivalence
        _report_equivalence(self._rungs([1, 2, 3, 4], [1, 2, 9, 4]))
        out = capsys.readouterr().out
        assert "DIVERGED" in out
        assert "decode token 2" in out
        assert "materialize=3" in out and "fused=9" in out

    def test_silent_when_a_rung_did_not_run(self, capsys):
        """A ladder run with --rungs 3_q70_fused has nothing to compare, and
        must not imply agreement it never checked."""
        from scripts.audit_e2e import _report_equivalence
        _report_equivalence([{"id": "3_q70_fused", "what": "f",
                              "first_tokens": [1, 2]}])
        assert capsys.readouterr().out == ""

    def test_compares_only_the_common_prefix(self, capsys):
        from scripts.audit_e2e import _report_equivalence
        _report_equivalence(self._rungs([1, 2, 3], [1, 2]))
        assert "OK" in capsys.readouterr().out


class TestBackendResolutionMatchesTheGeneratedConfig:
    """The bug this class exists for: the generated perf config's `model:` block
    has ONLY name/revision/dtype. Reading cfg.model.attn_implementation gets
    ModelConfig's dataclass default "eager" (utils/config.py:100), which selects
    the EAGER cache package -- a different implementation whose
    WindowedCacheConfig has no quant_budget_mode (TypeError at construction) --
    and loads the model with eager attention, so rung 0 is not the flash
    baseline the ladder claims to compare against. One wrong lookup, two
    symptoms.

    The earlier config-parity tests missed it because their fixture had no
    `model` section at all and never exercised package selection. This fixture
    mirrors run_perf_table.sh's YAML exactly, absent fields included.
    """

    @pytest.fixture
    def generated_cfg(self):
        import types
        from utils.config import ModelConfig
        return types.SimpleNamespace(
            # EXACTLY what the generated YAML sets: no attn_implementation.
            model=ModelConfig(name="meta-llama/Meta-Llama-3-8B-Instruct",
                              revision=None, dtype="float16"),
            window=types.SimpleNamespace(window_size=8, num_sink_tokens=5,
                                         local_window_size=64),
            cache=types.SimpleNamespace(quant_ratio=0.70,
                                        quant_budget_mode="tokens",
                                        first_eviction_step=0,
                                        rerotate_on_evict=False),
            perf=types.SimpleNamespace(configs=[{
                "name": "ours_q0.70", "cache_backend": "windowed",
                "cache_package": "flash_attn",
                "attn_implementation": "flash_attention_2",
                "cache_budget": 0.20, "quant_ratio": 0.70,
                "quant_budget_mode": "tokens",
            }]),
        )

    def test_the_model_section_really_does_lack_attn_implementation(
            self, generated_cfg):
        """Pins the premise: if this ever changes, the bug class disappears and
        this whole test group can be revisited."""
        from utils.config import ModelConfig
        assert ModelConfig().attn_implementation == "eager"
        assert generated_cfg.model.attn_implementation == "eager"

    def test_backend_comes_from_the_perf_cell_not_the_model_section(
            self, generated_cfg):
        from scripts.audit_e2e import resolve_backend
        pkg, attn = resolve_backend(generated_cfg)
        assert pkg == "flash_attn", "resolved the eager package for a flash run"
        assert attn == "flash_attention_2", "would load the model with eager attn"

    def test_cache_kwargs_fit_the_resolved_package(self, generated_cfg):
        """The exact TypeError from the first GPU invocation:
        WindowedCacheConfig.__init__() got an unexpected keyword argument
        'quant_budget_mode'."""
        from scripts.audit_e2e import resolve_backend, resolve_cache_kwargs
        from utils.cache_factory import get_cache_classes
        pkg, _ = resolve_backend(generated_cfg)
        _, WCC, _ = get_cache_classes(pkg)
        WCC(**resolve_cache_kwargs(generated_cfg, 0.70, pkg))  # must not raise

    def test_kwargs_are_filtered_to_the_eager_packages_smaller_field_list(
            self, generated_cfg):
        """The eager package genuinely has no quant_budget_mode. Filtering the
        DEFAULT is correct; see the next test for a non-default."""
        from scripts.audit_e2e import resolve_cache_kwargs
        from utils.cache_factory import get_cache_classes
        _, WCC, _ = get_cache_classes("eager")
        kw = resolve_cache_kwargs(generated_cfg, 0.0, "eager")
        assert "quant_budget_mode" not in kw
        WCC(**kw)

    def test_dropping_a_NON_default_value_raises_instead(self, generated_cfg):
        """Silently discarding a setting the config asked for would run a
        different method under the benchmarked name."""
        from scripts.audit_e2e import resolve_cache_kwargs
        generated_cfg.perf.configs[0]["quant_budget_mode"] = "bytes"
        with pytest.raises(ValueError, match="quant_budget_mode"):
            resolve_cache_kwargs(generated_cfg, 0.0, "eager")

    def test_a_mismatched_pairing_fails_with_a_named_cause(self, generated_cfg):
        """perf_runner validates this before its model load; the ladder now does
        too, so a mismatch is not a TypeError several frames down."""
        from utils.config import ConfigValidationError
        from scripts.audit_e2e import resolve_backend
        generated_cfg.perf.configs[0]["attn_implementation"] = "eager"
        with pytest.raises(ConfigValidationError):
            resolve_backend(generated_cfg)

    def test_falls_back_to_the_model_section_when_the_cell_is_silent(self):
        """A hand-written config that sets it the old way must still work."""
        import types
        from scripts.audit_e2e import resolve_backend
        from utils.config import ModelConfig
        cfg = types.SimpleNamespace(
            model=ModelConfig(attn_implementation="flash_attention_2"),
            cache=types.SimpleNamespace(backend_package="flash_attn"),
            perf=types.SimpleNamespace(configs=[]),
        )
        assert resolve_backend(cfg) == ("flash_attn", "flash_attention_2")
