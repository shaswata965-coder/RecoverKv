"""The decode profile's rollup, pinned against real CUDA kernel names.

``profile_decode.py`` can only run on a GPU, so the part that can be wrong
everywhere else — the name → bucket classification — is tested here instead.
It matters because the rollup is what replaces subtraction: reading a flat
top-N and calling the remainder a "tail" is how a 10.4 ms unattributed block
got quoted in this repo that was never a block, just every kernel below the cut.

A mis-filed kernel is worse than no rollup, because it looks authoritative. So
the names below are the real ones an A100 emits for Llama-3.1-8B fp16 with
flash-attn, including the three that match more than one pattern and are
therefore decided by bucket ORDER rather than by the pattern itself.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "scripts")
from profile_decode import _BUCKETS, _bucket_of, _print_buckets  # noqa: E402


CASES = [
    # -- ours ---------------------------------------------------------------
    ("_two_tier_decode_kernel", "ours: two-tier decode"),
    ("_gate_kernel", "ours: read gate"),
    ("_score_kernel", "ours: prefill score"),
    ("triton_poi_fused_add_clamp_div_round_sub_0", "ours: compiled evict"),
    ("triton_red_fused_amax_amin_2", "ours: compiled evict"),
    # -- model --------------------------------------------------------------
    ("sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x64x32_stage3"
     "_warpsize2x2x1_tensor16x8x16_kernel", "model: GEMM"),
    ("void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_f16_128x128"
     "_32x4_tn_align8>(T1::Params)", "model: GEMM"),
    ("void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4,"
     " false, false, cutlass::half_t>>(Flash_fwd_params)", "model: attention"),
    ("void at::native::(anonymous namespace)::vectorized_layer_norm_kernel"
     "<c10::Half, float>(int, float, ...)", "model: norm/softmax"),
    ("void at::native::(anonymous namespace)::softmax_warp_forward"
     "<c10::Half, c10::Half, float, 7, false, false>(...)", "model: norm/softmax"),
    # -- shared -------------------------------------------------------------
    ("void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl"
     "<at::native::copy_device_to_device(...)>>", "memory: copy/cat"),
    ("void at::native::(anonymous namespace)::CatArrayBatchedCopy"
     "<c10::Half, unsigned int, 1, 128, 1>(...)", "memory: copy/cat"),
    ("Memcpy DtoD (Device -> Device)", "memory: copy/cat"),
    ("void at::native::index_elementwise_kernel<128, 4, ...>",
     "memory: index/gather"),
    ("void at_cuda_detail::cub::DeviceRadixSortUpsweepKernel"
     "<cub::DeviceRadixSortPolicy<...>>(...)", "sort/topk"),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::"
     "CUDAFunctorOnSelf_add<c10::Half>, ...>", "elementwise"),
    ("void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, ...>>",
     "reduction"),
]


@pytest.mark.parametrize("key,want", CASES, ids=[c[1] + "|" + c[0][:22]
                                                 for c in CASES])
def test_kernel_names_land_in_the_right_bucket(key, want):
    got = _bucket_of(key)
    assert got == want, f"{key[:60]!r}\n  -> {got!r}, expected {want!r}"


# ---------------------------------------------------------------------------
# the three that two patterns both claim — order is what decides them
# ---------------------------------------------------------------------------


def test_ambiguous_names_are_decided_by_bucket_order():
    """Each of these matches a LATER bucket too; the earlier one must win."""
    # contains "copy_device_to_device" AND "elementwise_kernel"
    assert _bucket_of("elementwise_kernel<copy_device_to_device>") == "memory: copy/cat"
    # contains "index_elementwise" AND "elementwise_kernel"
    assert _bucket_of("index_elementwise_kernel<128>") == "memory: index/gather"
    # contains "splitKreduce" AND "reduce_kernel"
    assert _bucket_of("splitKreduce_kernel<32, 16, int, __half>") == "model: GEMM"
    # contains "radix" (sort) AND "cub::" (reduction)
    assert _bucket_of("at_cuda_detail::cub::DeviceRadixSortScanKernel") == "sort/topk"


def test_an_unknown_kernel_falls_to_other_rather_than_being_guessed():
    assert _bucket_of("some_kernel_nobody_has_seen_before") == "other"


def test_every_bucket_name_is_reachable():
    """A pattern list that can never match is a bucket that lies by omission."""
    for name, pats in _BUCKETS:
        assert pats, f"{name} has no patterns"
        probe = f"prefix_{pats[0]}_suffix"
        assert _bucket_of(probe) == name, (
            f"{name}'s own first pattern lands in {_bucket_of(probe)!r} — an "
            "earlier bucket shadows it entirely")


# ---------------------------------------------------------------------------
# the rollup itself
# ---------------------------------------------------------------------------


class _Ev:
    """The two attributes _print_buckets reads off a profiler event."""

    def __init__(self, key, us, count):
        self.key, self.self_device_time_total, self.count = key, us, count


def test_the_rollup_reconciles_and_never_leaves_other_anonymous(capsys):
    evs = [_Ev(k, 1000.0, 2) for k, _ in CASES]
    evs.append(_Ev("mystery_kernel_v2", 5000.0, 1))   # must be named in full
    evs.append(_Ev("a_zero_time_event", 0.0, 9))      # must be skipped
    gpu_us = sum(e.self_device_time_total for e in evs)

    _print_buckets(evs, n=10, gpu_us=gpu_us)
    out = capsys.readouterr().out

    assert "the rollup is dropping work" not in out, (
        "buckets did not sum to the CUDA self time")
    assert "mystery_kernel_v2" in out, (
        "an unmatched kernel was counted but not named — that is exactly the "
        "anonymous tail this rollup exists to remove")
    assert "ours (cache)" in out and "model" in out and "shared/other" in out


# ---------------------------------------------------------------------------
# kernel events vs the operator view that launched them
# ---------------------------------------------------------------------------


def test_aten_ops_are_not_counted_as_kernels():
    """``key_averages()`` returns both views and both carry self device time.

    A leaf op's self device time IS its kernels' time, reported again under the
    op's name. Summing the lot double-counts every kernel in the step — which is
    what every number this script printed had been doing: a real profile showed
    ``aten::mm`` at 225.0 launches/step against four ``ampere_*gemm*`` kernels
    summing to exactly 225.0.
    """
    from profile_decode import _is_device_event

    for op in ("aten::mm", "aten::copy_", "aten::mul", "aten::topk",
               "autograd::engine::evaluate_function", "ProfilerStep#42",
               "cudaLaunchKernel", "cudaMemcpyAsync"):
        assert not _is_device_event(_Ev(op, 1.0, 1)), f"{op} counted as a kernel"

    for kern in ("ampere_fp16_s16816gemm_fp16_256x64_ldg8_f2f_stages_64x3_tn",
                 "_two_tier_decode_kernel", "_gate_kernel",
                 "void at::native::vectorized_elementwise_kernel<4, ...>",
                 "Memcpy DtoD (Device -> Device)", "Memset (Device)",
                 "triton_poi_fused_add_0"):
        assert _is_device_event(_Ev(kern, 1.0, 1)), f"{kern} dropped"


def test_device_type_wins_over_the_name_when_torch_exposes_it():
    """The name check is a fallback. A device event keeps its name whatever it is."""
    # SKIP, not fail, where torch is absent. This box has no torch and no GPU by
    # contract (CLAUDE.md: "Triton cannot run on this dev box"), and a red bar
    # that only means "wrong machine" is what made the suite unusable as a
    # regression net. Where torch exists this runs exactly as before.
    pytest.importorskip("torch")
    from torch.autograd import DeviceType

    from profile_decode import _is_device_event

    ev = _Ev("aten::mm", 1.0, 1)          # named like an op ...
    ev.device_type = DeviceType.CUDA      # ... but tagged as a device event
    assert _is_device_event(ev) is True

    ev2 = _Ev("some_kernel_looking_name", 1.0, 1)
    ev2.device_type = DeviceType.CPU
    assert _is_device_event(ev2) is False


def test_the_double_count_from_the_real_profile_is_removed():
    """The exact rows from the 4096/batch-32 profile that exposed this."""
    from profile_decode import _is_device_event

    evs = [
        _Ev("ampere_fp16_s16816gemm_fp16_256x64_ldg8_f2f_stages_64x3_tn", 5115.0, 64),
        _Ev("ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x3_tn", 2657.0, 32),
        _Ev("ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stage", 2429.0, 65),
        _Ev("ampere_s16816gemm_fp16_128x64_ldg8_stages_32x6_tn", 630.0, 64),
        _Ev("aten::mm", 11052.0, 225),     # the same 225 launches, again
    ]
    kernels = [e for e in evs if _is_device_event(e)]
    assert sum(e.count for e in kernels) == 225, (
        "the op view is still being counted alongside its own kernels")
    assert abs(sum(e.self_device_time_total for e in kernels) - 10831.0) < 1.0


# ---------------------------------------------------------------------------
# "Compiled" and "fused" are different claims (_print_evict_fusion_verdict)
# ---------------------------------------------------------------------------


def _verdict(agg, n, compiled, eager, monkeypatch, control_arm=0):
    """Run the verdict printer against a synthetic rollup + path counters."""
    import profile_decode as pd
    pytest.importorskip("torch")   # cache_mod imports torch at module scope
    from modules.windowed_cache import cache as cache_mod
    monkeypatch.setattr(cache_mod, "_EVICT_STATS",
                        {"compiled": compiled, "eager": eager,
                         "control_arm": control_arm})
    pd._print_evict_fusion_verdict(agg, n)


def test_compiled_with_no_inductor_kernels_is_called_out(capsys, monkeypatch):
    """The 2026-09-16 GPU profile, reproduced: compiled runs, zero fused work.

    `_run_compiled_evict` is kernel-or-error, so finishing the run proves the
    compiled callable RAN. It does not prove Inductor generated anything, and
    the eviction that traces, graph-breaks and lowers back to ATen satisfies
    every other check in this repo while delivering none of the saving.
    """
    _verdict({"ours: two-tier decode": {"us": 7104.0, "n": 32.0}}, 1,
             compiled=3, eager=0, monkeypatch=monkeypatch)
    out = capsys.readouterr().out
    assert "COMPILED BUT NOT FUSED" in out
    assert "test_evict_graph_breaks.py" in out
    assert "--compile-evict 0" in out


def test_a_fused_eviction_reports_its_kernels_and_does_not_warn(capsys,
                                                               monkeypatch):
    _verdict({"ours: compiled evict": {"us": 9000.0, "n": 40.0}}, 1,
             compiled=3, eager=0, monkeypatch=monkeypatch)
    out = capsys.readouterr().out
    assert "NOT FUSED" not in out
    assert "9.000 ms/step" in out and "40 launches/step" in out


def test_no_eviction_in_the_window_is_not_reported_as_fused_or_unfused(
        capsys, monkeypatch):
    """A window shorter than `window_size` measures the steady step only.

    Reporting that as "not fused" would be a false alarm; reporting it as fused
    would be worse. It is a profile that did not cover the eviction at all.
    """
    _verdict({}, 1, compiled=0, eager=0, monkeypatch=monkeypatch)
    out = capsys.readouterr().out
    assert "NO EVICTION RAN" in out
    assert "NOT FUSED" not in out


def test_a_window_that_mixed_both_paths_says_so(capsys, monkeypatch):
    """An average over two methods is not a measurement of either."""
    _verdict({"ours: compiled evict": {"us": 100.0, "n": 4.0}}, 1,
             compiled=2, eager=1, monkeypatch=monkeypatch)
    assert "MIXED eviction paths" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# compilation ranges are not kernels (the 2026-09-19 "203.9% busy" profile)
# ---------------------------------------------------------------------------

#: The five `other`-bucket rows from the 4096/B=32 profile, verbatim, with the
#: per-step microseconds they reported over its 24 steps. Three have n < 1
#: launch in the WHOLE window: a kernel with no launches cannot have time, which
#: is the structural tell that these are ranges.
_REAL_COMPILE_ROWS = [
    ("CachingAutotuner.benchmark_all_configs (dynamo_timed)", 343_295.0 * 24, 58),
    ("compile_fx_inner (dynamo_timed)", 268_588.0 * 24, 1),
    ("_compile.compile_inner (dynamo_timed)", 58_345.0 * 24, 1),
    ("create_aot_dispatcher_function (dynamo_timed)", 56_318.0 * 24, 1),
    ("_recursive_joint_graph_passes (dynamo_timed)", 279.0 * 24, 1),
]


def test_compile_ranges_are_not_counted_as_kernels():
    """The rows that made `GPU busy` read 203.9%, which is impossible."""
    from profile_decode import _is_compile_range, _is_device_event

    for key, us, cnt in _REAL_COMPILE_ROWS:
        ev = _Ev(key, us, cnt)
        assert _is_compile_range(ev), f"{key} not recognised as a compile range"
        assert not _is_device_event(ev), f"{key} still counted as a kernel"


def test_a_compile_range_is_rejected_even_when_tagged_as_a_device_event():
    """Hard reject, ahead of ``device_type``.

    On the build that produced the profile these ranges carried CUDA self time
    and passed the device-type arm. A range's self device time is the CUDA time
    of everything nested inside it, so trusting the tag double-counts the very
    kernels it encloses.
    """
    torch = pytest.importorskip("torch")
    from torch.autograd import DeviceType

    from profile_decode import _is_device_event

    ev = _Ev("compile_fx_inner (dynamo_timed)", 268_588.0, 1)
    ev.device_type = DeviceType.CUDA
    assert _is_device_event(ev) is False


def test_real_kernels_are_not_swept_up_by_the_markers():
    """The markers must not eat the kernels the compiler PRODUCED.

    `triton_*_fused_*` are Inductor's output — the eviction's actual work — and
    they are the thing the `ours: compiled evict` bucket is counting. A marker
    list that caught them would hide exactly what it was added to reveal.
    """
    from profile_decode import _is_compile_range, _is_device_event

    for kern in ("triton_poi_fused_add_gather_mul_21",
                 "triton_red_fused__to_copy_mul_sub_sum_11",
                 "_two_tier_decode_kernel", "_gate_kernel",
                 "ampere_fp16_s16816gemm_fp16_256x64_ldg8_f2f_stages_64x3_tn",
                 "void at::native::vectorized_elementwise_kernel<4, ...>",
                 "Memcpy DtoD (Device -> Device)"):
        ev = _Ev(kern, 1000.0, 32)
        assert not _is_compile_range(ev), f"{kern} wrongly marked a compile range"
        assert _is_device_event(ev), f"{kern} dropped from the kernels"


def test_the_impossible_busy_number_reconciles_once_ranges_are_dropped():
    """1071.94 - 726.82 = 345.1 ms/step, i.e. 65.6% busy, not 203.9%."""
    from profile_decode import _is_device_event

    n = 24
    evs = [_Ev(k, us, c) for k, us, c in _REAL_COMPILE_ROWS]
    # the real kernels from the same profile, ms/step -> us over 24 steps
    for key, ms in (("_two_tier_decode_kernel", 14.480),
                    ("_gate_kernel", 2.441),
                    ("triton_poi_fused_add_gather_mul_21", 105.541),
                    ("ampere_fp16_s16816gemm_fp16_256x64_ldg8_f2f_stages", 10.952),
                    ("void at::native::vectorized_elementwise_kernel<4, ...>", 171.580),
                    ("Memcpy DtoD (Device -> Device)", 35.699),
                    ("void at::native::index_elementwise_kernel<128, 4, ...>", 2.536),
                    ("void at::native::radixSortKVInPlace", 0.832),
                    ("void at::native::reduce_kernel<512, 1, ...>", 0.934),
                    ("void at::native::silu_kernel", 0.117)):
        evs.append(_Ev(key, ms * 1000.0 * n, 32 * n))

    gpu_us = sum(e.self_device_time_total for e in evs if _is_device_event(e))
    steady_us = 525.81 * 1000.0 * n            # the unprofiled wall

    assert abs(gpu_us / n / 1000.0 - 345.11) < 0.5, (
        "the compile ranges are still in the kernel total")
    busy = gpu_us / steady_us
    assert 0.60 < busy < 0.70, f"busy={busy:.3f}, expected ~0.656"
    assert busy < 1.0, "a single stream cannot be more than 100% busy"


def test_compile_inside_the_window_is_reported_not_silently_dropped(capsys):
    """Excluding the ranges must not make a 10x-slow profile look normal.

    Before this, the ONLY reason a compile-contaminated window was noticeable
    was that the ranges had leaked into the kernel table. Fixing that leak
    without this printer would have removed the symptom and kept the disease.
    """
    from profile_decode import _print_compile_contamination

    n = 24
    evs = [_Ev(k, us, c) for k, us, c in _REAL_COMPILE_ROWS]
    _print_compile_contamination(evs, steady_us=525.81 * 1000.0 * n, n=n)
    out = capsys.readouterr().out

    assert "NOT A STEADY-STATE STEP" in out
    assert "CachingAutotuner" in out
    assert "726" in out or "727" in out, "the ms/step total is not reported"
    # It must NOT send the reader straight at --warmup; that is the reflex that
    # hides a per-eviction recompile, which no warmup length can absorb.
    assert "10.10" in out and "dynamic=True" in out


def test_a_clean_window_prints_no_contamination_block(capsys):
    from profile_decode import _print_compile_contamination

    evs = [_Ev("_two_tier_decode_kernel", 14_480.0, 32),
           _Ev("_gate_kernel", 2_441.0, 32)]
    _print_compile_contamination(evs, steady_us=45_690.0, n=1)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# the control arm is reported as a reference, never as a fault
# ---------------------------------------------------------------------------


def test_the_control_arm_is_not_reported_as_a_broken_compiled_run(capsys,
                                                                  monkeypatch):
    """An eager run BY REQUEST is the reference arm, not a fusion failure."""
    _verdict({}, 1, compiled=0, eager=0, monkeypatch=monkeypatch, control_arm=3)
    out = capsys.readouterr().out

    assert "CONTROL ARM" in out and "3 eager run(s)" in out
    assert "COMPILED BUT NOT FUSED" not in out
    assert "NO EVICTION RAN" not in out, (
        "the arm ran three evictions; reporting none is the opposite of true")


def test_a_control_arm_window_with_compiled_runs_in_it_is_called_out(capsys,
                                                                     monkeypatch):
    _verdict({}, 1, compiled=2, eager=0, monkeypatch=monkeypatch, control_arm=3)
    out = capsys.readouterr().out
    assert "MIXED" in out and "Not a clean arm" in out
