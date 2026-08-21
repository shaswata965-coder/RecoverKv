"""Tests for the standardized (kvpress-family) efficiency protocol.

Suite C can be run two ways: `native` (this project's own shape) and `kvpress`
(the protocol SnapKV / AdaKV / CriticalKV / DefensiveKV all publish through, via
kvpress' `evaluation/efficiency_evaluate.py`). These tests pin the four things
that make the second one *comparable*, none of which are visible in a timing
number once it is wrong:

  * the prompt is exactly `prefill_len` tokens of tiled English, not random ids;
  * the budget denominator is the prompt, not prompt + generation;
  * the protocol knobs are validated, so a typo fails loudly instead of silently
    reverting to native behaviour;
  * compression cost is reported in the column that matches where their press
    charges it.

All CPU-only -- no model, no GPU, no flash-attn.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from modules.evaluation.perf_runner import build_repeat_sentence_ids
from utils.config import ConfigValidationError, PerfConfig


class _CharTokenizer:
    """One token per `group` characters, so copies MERGE across the seam.

    This is the property that makes the tiling non-trivial:
    ``len(encode(s * n)) != n * len(encode(s))`` whenever ``len(s) % group != 0``.
    Real BPE tokenizers do the same thing for the same reason, which is why
    `build_repeat_sentence_ids` has to encode-check-grow rather than multiply.
    """

    def __init__(self, group: int = 4) -> None:
        self.group = group

    def encode(self, text, add_special_tokens=True, return_tensors=None):
        return [1] * math.ceil(len(text) / self.group)


class _EmptyTokenizer:
    def encode(self, text, add_special_tokens=True, return_tensors=None):
        return []


class TestRepeatSentencePrompt:
    @pytest.mark.parametrize("prefill_len", [1, 7, 100, 1024, 4096])
    def test_length_is_exact(self, prefill_len):
        """Exactly prefill_len tokens -- the context length IS the x-axis.

        A prompt one token off is a different point on every published curve.
        """
        ids = build_repeat_sentence_ids(_CharTokenizer(), "abcde", prefill_len, 1)
        assert ids.shape == (1, prefill_len)

    def test_grow_loop_runs_when_seam_merging_shortens_the_tiling(self):
        """The first tiling estimate is short; the result is still exact.

        "abcde" is 5 chars -> 2 tokens alone, but 5*reps/4 tokens when tiled, so
        the ceil(prefill/2)+1 first guess undershoots and the loop must grow.
        """
        tok = _CharTokenizer(group=4)
        assert len(tok.encode("abcde")) == 2          # 2 tokens alone
        first_guess = math.ceil(100 / 2) + 1
        assert len(tok.encode("abcde" * first_guess)) < 100   # ... but short tiled
        ids = build_repeat_sentence_ids(tok, "abcde", 100, 1)
        assert ids.shape == (1, 100)

    def test_rows_are_identical(self):
        """Every batch row is the same prompt: equal-length, no padding.

        Ragged rows would make the batch a padding benchmark, and would give each
        row a different score distribution to evict against.
        """
        ids = build_repeat_sentence_ids(_CharTokenizer(), "abcde", 64, 4)
        assert ids.shape == (4, 64)
        for r in range(1, 4):
            assert torch.equal(ids[0], ids[r])

    def test_is_deterministic(self):
        a = build_repeat_sentence_ids(_CharTokenizer(), "abcde", 128, 2)
        b = build_repeat_sentence_ids(_CharTokenizer(), "abcde", 128, 2)
        assert torch.equal(a, b)

    def test_batch_size_floor_is_one(self):
        ids = build_repeat_sentence_ids(_CharTokenizer(), "abcde", 16, 0)
        assert ids.shape == (1, 16)

    def test_untokenizable_sentence_raises(self):
        """A sentence that encodes to nothing would loop forever, not fail."""
        with pytest.raises(ValueError, match="tokenizes to nothing"):
            build_repeat_sentence_ids(_EmptyTokenizer(), "", 16, 1)

    def test_dtype_is_long(self):
        ids = build_repeat_sentence_ids(_CharTokenizer(), "abcde", 16, 1)
        assert ids.dtype == torch.long


class TestBudgetBasis:
    """`context` must drop the generation term from the budget denominator.

    kvpress defines compression_ratio = 1 - budget / context_length, so the
    budget is a fraction of the PROMPT. design.md §7 sizes ours against
    prefill + generation. At their gen_len=100 over a 32k prompt the two differ
    by 0.3%; at gen_len >> prefill they are different caches entirely.
    """

    class _ModelConfig:
        num_key_value_heads = 8
        num_attention_heads = 32
        hidden_size = 4096
        head_dim = 128

    def _resolve(self, prefill_len, max_tokens, ratio=0.5):
        from modules.windowed_eager_cache.config import WindowedCacheConfig
        cc = WindowedCacheConfig(window_size=8, num_sink_tokens=4,
                                 local_window_size=64, cache_budget=ratio)
        return cc.resolve(prefill_len, self._ModelConfig(), torch.float16, max_tokens)

    def test_context_basis_is_ratio_times_prefill(self):
        # max_tokens=0 is what budget_basis="context" passes.
        assert self._resolve(1000, 0).total_budget_tokens == 500

    def test_prefill_plus_gen_basis_includes_generation(self):
        assert self._resolve(1000, 500).total_budget_tokens == 750

    def test_the_two_bases_disagree_and_that_is_the_point(self):
        ctx_only = self._resolve(32768, 0).total_budget_tokens
        with_gen = self._resolve(32768, 100).total_budget_tokens
        assert ctx_only == 16384                 # == int(0.5 * 32768), their value
        assert with_gen > ctx_only               # ours is strictly larger
        # Small at the published shape, which is why it is easy to miss.
        assert (with_gen - ctx_only) / ctx_only < 0.01


class TestPerfConfigValidation:
    """A typo must fail loudly, not silently fall back to native behaviour."""

    @pytest.mark.parametrize("field,bad", [
        ("prompt_mode", "repeated_sentence"),   # plausible near-miss
        ("budget_basis", "prefill"),
        ("prefill_logits", "last"),
    ])
    def test_unknown_enum_value_rejected(self, field, bad):
        with pytest.raises(ConfigValidationError, match=field):
            PerfConfig(**{field: bad})

    def test_dataset_mode_without_data_source_rejected(self):
        with pytest.raises(ConfigValidationError, match="data_source"):
            PerfConfig(prompt_mode="dataset")

    def test_dataset_mode_with_data_source_accepted(self):
        assert PerfConfig(prompt_mode="dataset", data_source="2wikimqa").data_source

    def test_empty_repeat_sentence_rejected(self):
        with pytest.raises(ConfigValidationError, match="repeat_sentence"):
            PerfConfig(repeat_sentence="")

    def test_defaults_are_the_historical_behaviour(self):
        """Existing configs must not change meaning by adding these fields."""
        pc = PerfConfig()
        assert pc.protocol == "native"
        assert pc.prompt_mode == "auto"
        assert pc.budget_basis == "prefill_plus_gen"
        assert pc.prefill_logits == "full"


class TestShippedEfficiencyConfig:
    """The shipped config must actually pin the published protocol."""

    @pytest.fixture(scope="class")
    def cfg(self):
        from utils.config import load_config
        return load_config("configs/eval_efficiency.yaml")

    def test_protocol_knobs(self, cfg):
        pc = cfg.perf
        assert pc.protocol == "standard"
        assert pc.prompt_mode == "repeat_sentence"
        assert pc.budget_basis == "context"
        assert pc.num_warmup_runs == 5      # upstream warmup_rounds
        assert pc.num_measurement_runs == 10  # upstream measurement_rounds
        assert cfg.model.dtype == "bfloat16"

    def test_grid_is_exactly_100_decode_steps(self, cfg):
        """Upstream runs 100 decode forwards; this runner runs gen_len - 1."""
        for cell in cfg.perf.grid:
            assert cell["gen_len"] - 1 == 100, cell
            assert cell["batch_size"] == 1

    def test_context_ladder_matches_upstream(self, cfg):
        assert [c["prefill_len"] for c in cfg.perf.grid] == [24576, 32768, 40960]

    def test_budget_is_half(self, cfg):
        budgets = [c["cache_budget"] for c in cfg.perf.configs if "cache_budget" in c]
        assert budgets and all(b == 0.50 for b in budgets)


def _write_npz(path, *, protocol="standard", ttft=100.0, step0=250.0,
               steady=10.0, n_runs=4, compaction=None):
    """A synthetic Suite C npz in the schema `PerfRunner._save` writes."""
    names = ["fullkv_flash", "ours_flash_b50"]
    n = len(names)
    gen_len = 101
    n_decode = gen_len - 1
    ttft_a = np.full((n, n_runs), ttft)
    step0_a = np.array([[0.0] * n_runs, [step0] * n_runs])   # baseline never compacts
    steady_a = np.full((n, n_runs), steady)
    # tpot over ALL decode steps, exactly as the runner computes it.
    tpot_a = (step0_a + steady_a * (n_decode - 1)) / n_decode
    meta = {"prefill_len": 32768, "gen_len": gen_len, "batch_size": 1,
            "protocol": protocol, "prompt_mode": "repeat_sentence",
            "budget_basis": "context", "prefill_logits": "last_only",
            "num_warmup_runs": 5, "num_measurement_runs": n_runs,
            "compaction": compaction or {n: "decode_step0" for n in names},
            "model_name": "m", "dtype": "bfloat16", "gpu_name": "A100"}
    np.savez_compressed(
        str(path),
        config_names=np.array(names, dtype=object),
        attn_implementations=np.array(["flash_attention_2"] * n, dtype=object),
        ttft_ms=ttft_a, throughput_tokps=np.full((n, n_runs), 50.0),
        tpot_ms=tpot_a, e2e_latency_ms=np.full((n, n_runs), 1000.0),
        decode_step0_ms=step0_a, tpot_steady_ms=steady_a,
        prefill_plus_compress_ms=ttft_a + step0_a,
        peak_memory_mb=np.array([[8192.0] * n_runs, [5120.0] * n_runs]),
        peak_decode_step0_mb=np.array([[8192.0] * n_runs, [8192.0] * n_runs]),
        peak_decode_steady_mb=np.array([[8192.0] * n_runs, [2048.0] * n_runs]),
        peak_reserved_mb=np.array([[9216.0] * n_runs, [6144.0] * n_runs]),
        skipped_mask=np.zeros(n, bool), oom_mask=np.zeros(n, bool),
        error_mask=np.zeros(n, bool),
        skip_reason=np.array([""] * n, dtype=object),
        metadata_json=np.array([json.dumps(meta)], dtype=object),
    )
    return path


class TestEfficiencyPrinter:
    def _report(self, tmp_path, **kw):
        from scripts.print_efficiency import build_report
        p = _write_npz(tmp_path / "perf_prefill32768_gen101_bs1.npz", **kw)
        return build_report([p])

    def test_prefill_latency_column_includes_compression(self, tmp_path):
        """The published prefill column must be ttft + step 0, not ttft.

        Their press compacts inside prefill. Printing our bare ttft against it
        would credit us with a prefill that never paid for the compaction.
        """
        rep = self._report(tmp_path, ttft=100.0, step0=250.0)
        # ours: (100 + 250) ms = 0.350 s ; baseline: (100 + 0) ms = 0.100 s
        assert "prefill_latency(s)    0.350" in rep
        assert "prefill_latency(s)    0.100" in rep

    def test_decoding_latency_column_is_steady_state(self, tmp_path):
        """Their decoding_latency is a clean steady state; ours must be too."""
        rep = self._report(tmp_path, steady=10.0)
        assert "decoding_latency(s)   0.01000" in rep

    def test_raw_tpot_would_have_overcharged_us(self, tmp_path):
        """The note quantifies the error the naive mapping would have made."""
        rep = self._report(tmp_path, step0=250.0, steady=10.0)
        # tpot = (250 + 10*99)/100 = 12.4 ms vs 10 ms steady -> +24%
        assert "raw tpot_ms is 24% above steady state" in rep
        assert "ours_flash_b50" in rep

    def test_baseline_gets_no_inflation_note(self, tmp_path):
        """A full-KV config never compacts, so its tpot IS its steady state."""
        rep = self._report(tmp_path)
        notes = [ln for ln in rep.splitlines() if "above steady state" in ln]
        assert notes and all("fullkv_flash" not in ln for ln in notes)

    def test_native_protocol_npz_is_flagged_as_not_comparable(self, tmp_path):
        rep = self._report(tmp_path, protocol="native")
        assert "NOT comparable" in rep

    def test_kvpress_protocol_npz_is_not_flagged(self, tmp_path):
        assert "NOT comparable" not in self._report(tmp_path, protocol="standard")

    def test_old_npz_without_new_fields_still_renders(self, tmp_path):
        """Files written before this change must not crash the printer."""
        from scripts.print_efficiency import build_report
        p = _write_npz(tmp_path / "perf_prefill32768_gen101_bs1.npz")
        data = dict(np.load(str(p), allow_pickle=True))
        for f in ("decode_step0_ms", "tpot_steady_ms", "prefill_plus_compress_ms"):
            data.pop(f)
        np.savez_compressed(str(p), **data)
        rep = build_report([p])
        assert "prefill_latency(s)        -" in rep   # absent, not a fake zero


class TestEagerScoringGate:
    """Regression: windowed+eager cells must pass output_attentions=True.

    The eager score hook reads `attn_weights` off the attention module output,
    which transformers only populates under output_attentions=True. Without it
    the hook silently no-ops: scoring is disabled, eviction degrades to
    sink+local, and the cell times a DIFFERENT method than its name -- optimist-
    ically, because the scoring work never ran. Suite C gated this on
    `install_hooks_for_measurement` (an unrelated baseline-overhead knob), so
    every windowed+eager cell measured the degraded path.
    """

    def test_windowed_eager_requires_output_attentions(self):
        from modules.evaluation.perf_runner import needs_output_attentions
        assert needs_output_attentions("windowed", "eager", "eager", {}) is True

    def test_the_old_gate_alone_is_not_what_decides_it(self):
        """Without the flag it must STILL be on -- that was the actual bug."""
        from modules.evaluation.perf_runner import needs_output_attentions
        cfg_without_flag = {}
        assert needs_output_attentions("windowed", "eager", "eager",
                                       cfg_without_flag) is True

    def test_flash_package_does_not_need_it(self):
        """The flash path reconstructs scores from the LSE, not attn_weights."""
        from modules.evaluation.perf_runner import needs_output_attentions
        assert needs_output_attentions(
            "windowed", "flash_attn", "flash_attention_2", {}) is False

    def test_dynamic_baseline_only_with_the_hook_flag(self):
        from modules.evaluation.perf_runner import needs_output_attentions
        assert needs_output_attentions("dynamic", None, "eager", {}) is False
        assert needs_output_attentions(
            "dynamic", None, "eager", {"install_hooks_for_measurement": True}) is True

    def test_never_enabled_for_non_eager_attention(self):
        """output_attentions on a flash forward would not produce weights anyway."""
        from modules.evaluation.perf_runner import needs_output_attentions
        assert needs_output_attentions(
            "windowed", "eager", "flash_attention_2", {}) is False

    def test_shipped_efficiency_config_has_no_eager_rows(self):
        """output_attentions is [B, H, L, L]; at L=24576 that cannot run.

        The comparison protocol is 24k-40k context, so the eager backend is
        structurally excluded from it -- not merely slower.
        """
        from utils.config import load_config
        cfg = load_config("configs/eval_efficiency.yaml")
        impls = {c.get("attn_implementation") for c in cfg.perf.configs}
        assert impls == {"flash_attention_2"}


class TestWarmupAndCooldown:
    """GPU-state hygiene: keep the machine from writing itself into the score."""

    class _Cfg:
        class cache:
            first_eviction_step = 0

    def _steps(self, pc_kwargs, c, gen_len=101, fes=0):
        from modules.evaluation.perf_runner import PerfRunner
        from utils.config import PerfConfig
        cfg = self._Cfg()
        cfg.cache.first_eviction_step = fes
        return PerfRunner._warmup_decode_steps(PerfConfig(**pc_kwargs), c, cfg, gen_len)

    def test_auto_crosses_the_first_compaction(self):
        """The Triton eviction kernel compiles on its first call -- step 0.

        A prefill-only warmup never reaches it, so the JIT + autotune cost lands
        in measurement run 0 and reads as a slow compaction.
        """
        assert self._steps({}, {"cache_backend": "windowed"}, fes=0) == 2

    def test_auto_follows_a_delayed_first_eviction(self):
        assert self._steps({}, {"cache_backend": "windowed"}, fes=8) == 10

    def test_per_config_first_eviction_step_wins(self):
        assert self._steps({}, {"cache_backend": "windowed",
                                "first_eviction_step": 4}, fes=0) == 6

    def test_baseline_still_warms_the_decode_kernels(self):
        assert self._steps({}, {"cache_backend": "dynamic"}) == 2

    def test_explicit_value_overrides_auto(self):
        assert self._steps({"warmup_decode_steps": 7},
                           {"cache_backend": "windowed"}) == 7

    def test_zero_restores_prefill_only_warmup(self):
        assert self._steps({"warmup_decode_steps": 0},
                           {"cache_backend": "windowed"}) == 0

    def test_never_exceeds_what_the_cell_generates(self):
        """A 2-token cell cannot warm 10 decode steps."""
        assert self._steps({"warmup_decode_steps": 10},
                           {"cache_backend": "windowed"}, gen_len=3) == 2
        assert self._steps({}, {"cache_backend": "windowed"},
                           gen_len=2, fes=8) == 1

    def test_cooldown_idles_for_the_configured_time(self, monkeypatch):
        from modules.evaluation import perf_runner as pr
        from utils.config import PerfConfig
        slept = []
        monkeypatch.setattr(pr.time, "sleep", lambda s: slept.append(s))
        pr.PerfRunner._cooldown(PerfConfig(cooldown_s=2.5), "run 0")
        assert slept == [2.5]

    def test_zero_cooldown_does_not_sleep(self, monkeypatch):
        from modules.evaluation import perf_runner as pr
        from utils.config import PerfConfig
        slept = []
        monkeypatch.setattr(pr.time, "sleep", lambda s: slept.append(s))
        pr.PerfRunner._cooldown(PerfConfig(), "run 0")
        assert slept == []

    def test_shipped_config_idles_between_runs(self):
        from utils.config import load_config
        cfg = load_config("configs/eval_efficiency.yaml")
        assert cfg.perf.cooldown_s > 0
        assert cfg.perf.warmup_decode_steps is None   # auto

    @pytest.mark.parametrize("bad", [-1.0, "2s", True])
    def test_negative_or_non_numeric_cooldown_rejected(self, bad):
        from utils.config import ConfigValidationError, PerfConfig
        with pytest.raises(ConfigValidationError, match="cooldown_s"):
            PerfConfig(cooldown_s=bad)

    def test_negative_warmup_decode_steps_rejected(self):
        from utils.config import ConfigValidationError, PerfConfig
        with pytest.raises(ConfigValidationError, match="warmup_decode_steps"):
            PerfConfig(warmup_decode_steps=-1)


class TestLogitsKwargProbe:
    """transformers renamed the kwarg in 4.49; this project pins 4.47.1."""

    def test_picks_the_4_47_spelling(self):
        from modules.evaluation.perf_runner import _logits_kwarg_name

        class Pinned:            # transformers 4.47.1 signature
            def forward(self, input_ids=None, num_logits_to_keep: int = 0): ...
        assert _logits_kwarg_name(Pinned()) == "num_logits_to_keep"

    def test_picks_the_modern_spelling(self):
        from modules.evaluation.perf_runner import _logits_kwarg_name

        class Modern:            # transformers >= 4.49
            def forward(self, input_ids=None, logits_to_keep: int = 0): ...
        assert _logits_kwarg_name(Modern()) == "logits_to_keep"

    def test_returns_none_when_the_model_has_neither(self):
        """Must fall back to full logits, not crash the benchmark."""
        from modules.evaluation.perf_runner import _logits_kwarg_name

        class Old:
            def forward(self, input_ids=None): ...
        assert _logits_kwarg_name(Old()) is None


class TestDecodePhaseSplit:
    def test_nanmax2_keeps_the_present_value(self):
        from modules.evaluation.perf_runner import _nanmax2
        nan = float("nan")
        assert _nanmax2(1.0, 2.0) == 2.0
        assert _nanmax2(nan, 2.0) == 2.0
        assert _nanmax2(1.0, nan) == 1.0
        assert _nanmax2(nan, nan) != _nanmax2(nan, nan)   # nan

    def test_printer_reports_the_compressed_footprint(self, tmp_path):
        """Whole-run peak cannot show compression; the steady phase can.

        Peak is max(prefill+1, budget) because prefill cannot compact, so at
        these shapes it is the prompt for every method. The steady column is the
        post-compaction cache -- 2 GB vs the baseline's 8 GB in this fixture.
        """
        from scripts.print_efficiency import build_report
        p = _write_npz(tmp_path / "perf_prefill32768_gen101_bs1.npz")
        rep = build_report([p])
        assert "steadyKV_GB" in rep
        lines = {ln.split()[0]: ln for ln in rep.splitlines()
                 if ln.strip().startswith(("fullkv", "ours_"))}
        assert lines["ours_flash_b50"].split()[-1] == "2.00"
        assert lines["fullkv_flash"].split()[-1] == "8.00"

    def test_b1_caveat_is_stated(self, tmp_path):
        """A tie on decode latency at B=1 is physics, and must not read as a loss."""
        from scripts.print_efficiency import build_report
        p = _write_npz(tmp_path / "perf_prefill32768_gen101_bs1.npz")
        assert "cannot produce a decode speedup" in build_report([p])


# ---------------------------------------------------------------------------
# The external-method seam: can this harness benchmark a method it does not own?
# ---------------------------------------------------------------------------
# This matters because the published protocols are under-specified. Papers report
# "peak memory and decoding latency" without stating context length, batch size,
# warmup rounds, dtype, or prompt construction, so quoting a printed number
# beside ours is not a controlled comparison. Running the baseline in-process
# under one protocol is -- which requires a working plug-in seam.


def _factory_stub(**kw):
    class M:
        def new_cache(self): return object()
        def describe(self): return "stub"
    return M()


def _factory_no_new_cache(**kw):
    return object()


def _factory_wants_attn(**kw):
    class M:
        requires_output_attentions = True
        def new_cache(self): return object()
    return M()


class TestMethodFactoryResolution:
    def test_resolves_a_dotted_spec(self):
        from modules.evaluation.perf_runner import resolve_method_factory
        f = resolve_method_factory(f"{__name__}:_factory_stub")
        assert f is _factory_stub

    @pytest.mark.parametrize("spec,match", [
        ("no_colon_here", "package.module:callable"),
        ("", "package.module:callable"),
        ("nonexistent_module_xyz:build", "not importable"),
        (f"{__name__}:does_not_exist", "has no"),
    ])
    def test_bad_specs_fail_loudly(self, spec, match):
        from modules.evaluation.perf_runner import resolve_method_factory
        with pytest.raises(ValueError, match=match):
            resolve_method_factory(spec)

    def test_non_callable_target_rejected(self):
        from modules.evaluation.perf_runner import resolve_method_factory
        with pytest.raises(ValueError, match="non-callable"):
            resolve_method_factory(f"{__name__}:MB_PER_GB_NOT_CALLABLE")


MB_PER_GB_NOT_CALLABLE = 1024.0


class TestExternalBudgetParity:
    """A plugged-in baseline must be held to OUR budget, not to its own default.

    Otherwise the comparison silently comes from two different operating points
    and the table means nothing.
    """

    def test_context_basis_matches_the_windowed_resolver(self):
        from modules.evaluation.perf_runner import resolve_budget_tokens
        # budget_horizon=0 is what budget_basis="context" passes, and
        # TestBudgetBasis pins the windowed resolver to the same 16384.
        assert resolve_budget_tokens(0.5, 32768, 0) == 16384

    def test_prefill_plus_gen_basis_includes_generation(self):
        from modules.evaluation.perf_runner import resolve_budget_tokens
        assert resolve_budget_tokens(0.5, 1000, 500) == 750

    def test_full_cache_config_has_no_budget(self):
        from modules.evaluation.perf_runner import resolve_budget_tokens
        assert resolve_budget_tokens(None, 32768, 0) is None

    def test_external_method_can_demand_attention_weights(self):
        """An outside method that scores off attn_weights must get them."""
        from modules.evaluation.perf_runner import needs_output_attentions
        assert needs_output_attentions(
            "external", None, "eager", {"_external_needs_attn": True}) is True
        assert needs_output_attentions("external", None, "eager", {}) is False


class TestCoreIsMethodAgnostic:
    """The measurement core must fit any cache: pressed, quantized, evicted, none.

    Nothing about a method reaches the timing/memory code. The core builds a
    cache, runs a prefill and N decode steps, and records phases -- identical for
    every method. The ONE thing it needs to know is WHERE a method does its
    one-off compaction, because the same physical work lands in a different
    column depending on the design. These tests pin that, with stand-in method
    handles rather than any shipped method implementation.
    """

    def test_press_style_compacts_in_prefill(self):
        """SnapKV/AdaKV-shaped: compaction inside the prefill forward."""
        from modules.evaluation.perf_runner import resolve_compaction_phase

        class Press:
            compaction_phase = "prefill"
        assert resolve_compaction_phase(
            {"cache_backend": "external"}, Press()) == "prefill"

    def test_deferred_eviction_compacts_on_decode_step0(self):
        """Our shape, and any method whose scores need the prefill to finish."""
        from modules.evaluation.perf_runner import resolve_compaction_phase

        class Deferred:
            compaction_phase = "decode_step0"
        assert resolve_compaction_phase(
            {"cache_backend": "external"}, Deferred()) == "decode_step0"
        assert resolve_compaction_phase({"cache_backend": "windowed"}) == "decode_step0"

    def test_quantize_only_method_compacts_nowhere(self):
        """A method that shrinks bytes without ever evicting is still measurable."""
        from modules.evaluation.perf_runner import resolve_compaction_phase

        class QuantOnly:
            compaction_phase = "none"
        assert resolve_compaction_phase(
            {"cache_backend": "external"}, QuantOnly()) == "none"

    def test_full_cache_compacts_nowhere(self):
        from modules.evaluation.perf_runner import resolve_compaction_phase
        assert resolve_compaction_phase({"cache_backend": "dynamic"}) == "none"

    def test_an_explicit_config_key_overrides_the_inference(self):
        from modules.evaluation.perf_runner import resolve_compaction_phase
        assert resolve_compaction_phase(
            {"cache_backend": "windowed", "compaction": "prefill"}) == "prefill"

    def test_a_bogus_compaction_value_fails_loudly(self):
        from modules.evaluation.perf_runner import resolve_compaction_phase
        with pytest.raises(ValueError, match="compaction must be one of"):
            resolve_compaction_phase({"name": "x", "compaction": "at_the_end"})

    def test_an_unbudgeted_method_is_supported(self):
        """Quantize-only / full-cache methods get budget_tokens=None, not a crash."""
        from modules.evaluation.perf_runner import resolve_budget_tokens
        assert resolve_budget_tokens(None, 32768, 0) is None

    def test_a_method_needs_no_hooks(self):
        """install_hooks is optional -- a plain Cache subclass needs none."""
        from modules.evaluation.perf_runner import resolve_method_factory
        f = resolve_method_factory(f"{__name__}:_factory_stub")
        m = f()
        assert m.new_cache() is not None
        assert not hasattr(m, "install_hooks")


class TestPrefillColumnPerMethodShape:
    """Charging compaction to the wrong column is a real measurement error.

    ``prefill_latency`` means "input to first token, including compression paid
    on the way". A decode_step0 method must add step 0; a press-style or
    full-cache row must not, because it either already paid inside TTFT or never
    paid at all. Assuming one shape for every row overstates the others by a
    whole decode step.
    """

    def _report(self, tmp_path, compaction):
        from scripts.print_efficiency import build_report
        p = _write_npz(tmp_path / "perf_prefill32768_gen101_bs1.npz",
                       ttft=100.0, step0=250.0, compaction=compaction)
        return build_report([p])

    def test_decode_step0_row_adds_step0(self, tmp_path):
        rep = self._report(tmp_path, {"fullkv_flash": "none",
                                      "ours_flash_b50": "decode_step0"})
        assert "prefill_latency(s)    0.350" in rep      # 100 + 250 ms

    def test_press_style_row_does_not_add_step0(self, tmp_path):
        """Its compaction is already inside TTFT; adding step 0 double-counts."""
        rep = self._report(tmp_path, {"fullkv_flash": "none",
                                      "ours_flash_b50": "prefill"})
        assert "prefill_latency(s)    0.350" not in rep
        assert "prefill_latency(s)    0.100" in rep

    def test_full_cache_row_reports_bare_ttft(self, tmp_path):
        rep = self._report(tmp_path, {"fullkv_flash": "none",
                                      "ours_flash_b50": "decode_step0"})
        fullkv = [ln for ln in rep.splitlines() if "fullkv_flash" in ln
                  and "prefill_latency" in ln][0]
        assert "0.100" in fullkv          # its step0 is a normal decode step

    def test_the_tpot_note_is_only_for_decode_step0_rows(self, tmp_path):
        """A press-style row has nothing one-off in step 0 to explain away."""
        rep = self._report(tmp_path, {"fullkv_flash": "none",
                                      "ours_flash_b50": "prefill"})
        assert "above steady state" not in rep

    def test_the_phase_is_shown_per_row(self, tmp_path):
        rep = self._report(tmp_path, {"fullkv_flash": "none",
                                      "ours_flash_b50": "decode_step0"})
        assert "compaction" in rep and "decode_step0" in rep


class TestShapeSweep:
    """prefill / decode / batch / window must be drivable from the shell.

    scripts/run_efficiency.sh turns --prefill/--decode/--batch into these
    overrides, so what is pinned here is the contract that script depends on.
    """

    def _cells(self, **kw):
        from modules.evaluation.perf_runner import build_cells
        from utils.config import PerfConfig
        return build_cells(PerfConfig(**kw))

    def test_cartesian_product_of_the_three_axes(self):
        cells = self._cells(prefill_lengths=[128, 256], gen_lengths=[4, 6],
                            batch_sizes=[1, 2])
        assert len(cells) == 8
        assert (128, 4, 1) in cells and (256, 6, 2) in cells

    def test_axes_fall_back_to_their_scalars(self):
        """Empty list = "not swept", so old configs keep their exact cells."""
        assert self._cells(prefill_lengths=[512], gen_len=256, batch_size=4) \
            == [(512, 256, 4)]

    def test_only_one_axis_swept(self):
        assert self._cells(prefill_lengths=[128, 256], gen_len=32, batch_size=1) \
            == [(128, 32, 1), (256, 32, 1)]

    def test_batch_only_sweep(self):
        assert self._cells(prefill_lengths=[512], gen_len=32, batch_sizes=[1, 4, 16]) \
            == [(512, 32, 1), (512, 32, 4), (512, 32, 16)]

    def test_an_explicit_grid_still_wins(self):
        """A hand-written grid is used verbatim -- sweeps never override it."""
        cells = self._cells(
            grid=[{"prefill_len": 4096, "gen_len": 101, "batch_size": 2}],
            prefill_lengths=[128, 256], gen_lengths=[4], batch_sizes=[1])
        assert cells == [(4096, 101, 2)]

    def test_grid_batch_size_defaults_to_the_scalar(self):
        cells = self._cells(grid=[{"prefill_len": 4096, "gen_len": 101}],
                            batch_size=8)
        assert cells == [(4096, 101, 8)]

    @pytest.mark.parametrize("axis", ["prefill_lengths", "gen_lengths", "batch_sizes"])
    def test_non_list_axis_rejected_with_the_cli_syntax_in_the_message(self, axis):
        from utils.config import ConfigValidationError, PerfConfig
        with pytest.raises(ConfigValidationError, match=r"bracket syntax"):
            PerfConfig(**{axis: 4096})

    @pytest.mark.parametrize("bad", [0, -1, True, 2.5])
    def test_nonsense_axis_entries_rejected(self, bad):
        from utils.config import ConfigValidationError, PerfConfig
        with pytest.raises(ConfigValidationError, match="ints >= 1"):
            PerfConfig(batch_sizes=[bad])


class TestCliListOverrides:
    """--override must carry lists, or none of the sweep is reachable from sh."""

    def test_bracket_syntax_becomes_a_list(self):
        from main import _parse_value
        assert _parse_value("[4096,8192]") == [4096, 8192]

    def test_empty_brackets_clear_the_grid(self):
        """perf.grid=[] is how the shell switches on the cartesian path."""
        from main import _parse_value
        assert _parse_value("[]") == []

    def test_scalars_are_unchanged(self):
        from main import _parse_value
        assert _parse_value("1024") == 1024
        assert _parse_value("0.5") == 0.5
        assert _parse_value("true") is True

    def test_a_string_containing_commas_is_not_split(self):
        """Bare-comma parsing would have shredded perf.repeat_sentence."""
        from main import _parse_value
        assert _parse_value("The quick, brown fox.") == "The quick, brown fox."

    def test_whitespace_in_a_list_is_tolerated(self):
        from main import _parse_value
        assert _parse_value("[1, 4, 16]") == [1, 4, 16]


class TestCorpusRouting:
    """perf.data_source must reach any corpus the project can load, not only LongBench."""

    def test_a_longbench_name_routes_to_the_longbench_loader(self, monkeypatch):
        import data.longbench_loader as lb
        monkeypatch.setattr(lb, "load_longbench_dataset",
                            lambda name: [{"context": f"ctx:{name}", "input": "q"}])
        from modules.evaluation.perf_runner import iter_corpus_texts
        assert "ctx:2wikimqa" in next(iter_corpus_texts("2wikimqa"))

    def test_wikitext_routes_to_the_corpus_loader(self, monkeypatch):
        """The case that motivated this: wikitext-103 is not a LongBench name."""
        import data.corpus_loader as cl
        seen = {}

        class FakeLoader:
            def __init__(self, name, *a, **k): seen["name"] = name
            def load(self): return ["article one", "article two"]
        monkeypatch.setattr(cl, "CorpusLoader", FakeLoader)
        from modules.evaluation.perf_runner import iter_corpus_texts
        assert list(iter_corpus_texts("wikitext-103")) == ["article one", "article two"]
        assert seen["name"] == "wikitext-103"

    def test_a_local_path_routes_to_the_corpus_loader(self, monkeypatch):
        import data.corpus_loader as cl
        seen = {}

        class FakeLoader:
            def __init__(self, name, *a, **k): seen["name"] = name
            def load(self): return ["doc"]
        monkeypatch.setattr(cl, "CorpusLoader", FakeLoader)
        from modules.evaluation.perf_runner import iter_corpus_texts
        list(iter_corpus_texts("./data/mine.jsonl"))
        assert seen["name"] == "./data/mine.jsonl"

    def test_explicit_prefixes_win_over_the_name_lookup(self, monkeypatch):
        """`corpus:2wikimqa` must not be hijacked by the LongBench name set."""
        import data.corpus_loader as cl
        seen = {}

        class FakeLoader:
            def __init__(self, name, *a, **k): seen["name"] = name
            def load(self): return ["forced"]
        monkeypatch.setattr(cl, "CorpusLoader", FakeLoader)
        from modules.evaluation.perf_runner import iter_corpus_texts
        assert list(iter_corpus_texts("corpus:2wikimqa")) == ["forced"]
        assert seen["name"] == "2wikimqa"

    def test_longbench_prefix_strips_the_prefix(self, monkeypatch):
        import data.longbench_loader as lb
        seen = {}

        def fake(name):
            seen["name"] = name
            return [{"context": "c", "input": "i"}]
        monkeypatch.setattr(lb, "load_longbench_dataset", fake)
        from modules.evaluation.perf_runner import iter_corpus_texts
        list(iter_corpus_texts("longbench:qasper"))
        assert seen["name"] == "qasper"

    def test_the_longbench_name_set_is_read_from_the_shipped_config(self):
        """Routing must not drift from the loader's own dataset list."""
        from modules.evaluation.perf_runner import _longbench_names
        names = _longbench_names()
        assert "2wikimqa" in names and "qasper" in names
        assert "wikitext-103" not in names and "pg19" not in names


class TestCorpusChunking:
    """Fixed-length chunks, built without tokenizing the whole corpus."""

    class _Tok:
        """One token per character, so token counts are readable in the test."""
        def encode(self, text, add_special_tokens=True, return_tensors=None):
            return [ord(ch) for ch in text]

    def _chunks(self, texts, length, pool):
        from modules.evaluation.perf_runner import chunk_texts_to_length
        return chunk_texts_to_length(iter(texts), self._Tok(), length, pool)

    def test_every_chunk_is_exactly_the_requested_length(self):
        chunks = self._chunks(["a" * 100], length=32, pool=3)
        assert [c.shape[0] for c in chunks] == [32, 32, 32]

    def test_stops_as_soon_as_the_pool_is_full(self):
        """The whole point: wikitext-103 must not be tokenized to fill 2 rows."""
        consumed = []

        def gen():
            for i in range(1000):
                consumed.append(i)
                yield "b" * 64
        from modules.evaluation.perf_runner import chunk_texts_to_length
        chunks = chunk_texts_to_length(gen(), self._Tok(), 32, pool=2)
        assert len(chunks) == 2
        assert len(consumed) <= 2      # not 1000

    def test_documents_are_joined_and_cut_on_token_boundaries(self):
        """A chunk may span documents; for wall-clock that is irrelevant."""
        chunks = self._chunks(["aaaa", "bbbb", "cccc"], length=6, pool=4)
        assert chunks and all(c.shape[0] == 6 for c in chunks)

    def test_a_short_corpus_yields_fewer_chunks_than_the_pool(self):
        assert len(self._chunks(["a" * 40], length=32, pool=8)) == 1

    def test_a_corpus_shorter_than_one_chunk_yields_nothing(self):
        assert self._chunks(["short"], length=1024, pool=4) == []

    def test_chunks_are_long_dtype_for_input_ids(self):
        assert self._chunks(["a" * 100], 32, 1)[0].dtype == torch.long


class TestCorpusBatchAssembly:
    """batch_size rows of real text, equal-length, deterministic."""

    class _Runner:
        pass

    def _build(self, monkeypatch, batch_size, n_articles=8, seed=42):
        import data.corpus_loader as cl

        class FakeLoader:
            def __init__(self, name, *a, **k): pass
            def load(self):
                return [chr(ord("a") + i) * 200 for i in range(n_articles)]
        monkeypatch.setattr(cl, "CorpusLoader", FakeLoader)
        from modules.evaluation.perf_runner import PerfRunner
        return PerfRunner.__new__(PerfRunner)._build_input_ids(
            "fake-corpus", TestCorpusChunking._Tok(), 32, batch_size, seed)

    def test_shape_is_batch_by_prefill(self, monkeypatch):
        assert self._build(monkeypatch, 4).shape == (4, 32)

    def test_rows_are_not_all_identical(self, monkeypatch):
        """Real-text rows should differ, unlike the synthetic tiled prompt."""
        ids = self._build(monkeypatch, 4)
        assert not all(torch.equal(ids[0], ids[i]) for i in range(1, 4))

    def test_same_seed_gives_the_same_batch(self, monkeypatch):
        a = self._build(monkeypatch, 3, seed=7)
        b = self._build(monkeypatch, 3, seed=7)
        assert torch.equal(a, b)

    def test_too_small_a_corpus_fails_with_an_actionable_message(self, monkeypatch):
        with pytest.raises(ValueError, match="Reduce batch_size or prefill_len"):
            self._build(monkeypatch, batch_size=64, n_articles=1)


class TestArbitraryCorpusPaths:
    """Point the benchmark at your own text: any file, any directory, any suffix.

    A path is the common case (wikitext-103 was only ever an example), so it has
    to work for a corpus that happens to be .md, .text or extensionless, and it
    has to fail legibly when the path is simply wrong.
    """

    @pytest.fixture
    def corpus(self, tmp_path):
        (tmp_path / "plain.txt").write_text("alpha " * 50, encoding="utf-8")
        (tmp_path / "notes.md").write_text("beta " * 50, encoding="utf-8")
        (tmp_path / "noext").write_text("gamma " * 50, encoding="utf-8")
        sub = tmp_path / "nested"
        sub.mkdir()
        (sub / "deep.txt").write_text("delta " * 50, encoding="utf-8")
        return tmp_path

    def _texts(self, src):
        from modules.evaluation.perf_runner import iter_corpus_texts
        return list(iter_corpus_texts(str(src)))

    def test_a_plain_text_file(self, corpus):
        assert "alpha" in self._texts(corpus / "plain.txt")[0]

    def test_a_markdown_file(self, corpus):
        """CorpusLoader only knows .txt/.json*; a benchmark corpus may be .md."""
        assert "beta" in self._texts(corpus / "notes.md")[0]

    def test_an_extensionless_file(self, corpus):
        assert "gamma" in self._texts(corpus / "noext")[0]

    def test_a_directory_is_walked_recursively(self, corpus):
        joined = " ".join(self._texts(corpus))
        for token in ("alpha", "beta", "gamma", "delta"):
            assert token in joined

    def test_directory_order_is_deterministic(self, corpus):
        """Sorted walk, so the same corpus always presents in the same order."""
        assert self._texts(corpus) == self._texts(corpus)

    def test_the_local_prefix_is_accepted(self, corpus):
        assert "alpha" in self._texts(f"local:{corpus / 'plain.txt'}")[0]

    def test_a_jsonl_file_is_parsed_as_records(self, tmp_path):
        """Structured formats still go through CorpusLoader, not raw read."""
        f = tmp_path / "docs.jsonl"
        f.write_text('{"text": "one"}\n{"text": "two"}\n', encoding="utf-8")
        assert self._texts(f) == ["one", "two"]

    def test_an_empty_file_yields_nothing(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("   \n", encoding="utf-8")
        assert self._texts(f) == []

    def test_a_path_wins_over_a_dataset_name(self, tmp_path):
        """A local dir named like a dataset must not be shadowed by the name."""
        d = tmp_path / "2wikimqa"
        d.mkdir()
        (d / "a.txt").write_text("local content here", encoding="utf-8")
        assert "local content here" in self._texts(d)[0]

    def test_a_missing_path_says_so_instead_of_unsupported_dataset(self, tmp_path):
        """The usual cause is a mistyped PATH, not an unknown dataset NAME."""
        from modules.evaluation.perf_runner import iter_corpus_texts
        with pytest.raises(ValueError, match="not an existing file or directory"):
            list(iter_corpus_texts(str(tmp_path / "nope.txt")))

    def test_the_error_lists_what_was_actually_tried(self, tmp_path):
        from modules.evaluation.perf_runner import iter_corpus_texts
        with pytest.raises(ValueError, match="LongBench dataset names"):
            list(iter_corpus_texts(str(tmp_path / "nope.txt")))


class TestExactPrefillAndGenerationFromAPath:
    """Given a path: exactly prefill_len tokens in, exactly gen_len-1 steps out."""

    class _Tok:
        def encode(self, text, add_special_tokens=True, return_tensors=None):
            return [ord(c) for c in text]

    def test_prefill_is_exactly_the_requested_token_count(self, tmp_path):
        (tmp_path / "c.txt").write_text("x" * 10000, encoding="utf-8")
        from modules.evaluation.perf_runner import PerfRunner
        ids = PerfRunner.__new__(PerfRunner)._build_input_ids(
            str(tmp_path), self._Tok(), prefill_len=512, batch_size=2, seed=42)
        assert ids.shape == (2, 512)

    def test_a_corpus_too_small_for_the_batch_is_actionable(self, tmp_path):
        (tmp_path / "c.txt").write_text("x" * 600, encoding="utf-8")
        from modules.evaluation.perf_runner import PerfRunner
        with pytest.raises(ValueError, match="Reduce batch_size or prefill_len"):
            PerfRunner.__new__(PerfRunner)._build_input_ids(
                str(tmp_path), self._Tok(), prefill_len=512, batch_size=8, seed=42)

    def test_decode_length_is_fixed_by_gen_len_not_by_the_corpus(self):
        """Generation length is a protocol knob; the corpus never changes it."""
        from modules.evaluation.perf_runner import build_cells
        from utils.config import PerfConfig
        cells = build_cells(PerfConfig(prompt_mode="dataset", data_source="/some/path",
                                       prefill_lengths=[512], gen_lengths=[9],
                                       batch_sizes=[2]))
        assert cells == [(512, 9, 2)]     # 9 -> 8 decode steps, corpus-independent

    def test_the_corpus_identity_is_recorded(self, tmp_path):
        """A dataset run must say WHICH corpus, or it is not reproducible."""
        from utils.config import PerfConfig
        pc = PerfConfig(prompt_mode="dataset", data_source=str(tmp_path))
        assert pc.data_source == str(tmp_path)
