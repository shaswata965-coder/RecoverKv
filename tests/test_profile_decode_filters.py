"""``profile_decode.py``'s event classification, pinned against real names.

The script can only RUN on a GPU, so the part that can be wrong everywhere else
-- which profiler events count as kernels -- is tested here instead. It matters
because `gpu_us` feeds `GPU busy %`, and that one number is what the script
exists to print: it decides host-bound vs kernel-bound, and therefore what the
next optimisation is.

Two distinct over-counts are covered, both of which were live:

1. the ATen operator view. `key_averages()` returns the op AND the kernels it
   launched, and both carry self device time -- so summing every event counts
   each kernel twice.
2. compilation ranges. `dynamo_timed` wraps compiles in a `record_function`
   whose self device time is the CUDA time of everything nested inside it.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "scripts")
from profile_decode import (  # noqa: E402
    _is_compile_range, _is_device_event, _print_compile_contamination,
)


class _Ev:
    """The attributes the script reads off a profiler event."""

    def __init__(self, key, us, count):
        self.key, self.self_device_time_total, self.count = key, us, count


#: The five `dynamo_timed` rows from a 2026-09-19 A100 profile at 4096/B=32,
#: with the ms/step they reported over its 24 steps. Three have n<1 launch in
#: the whole window: a kernel with no launches cannot have time.
_REAL_COMPILE_ROWS = [
    ("CachingAutotuner.benchmark_all_configs (dynamo_timed)", 343.295, 58),
    ("compile_fx_inner (dynamo_timed)", 268.588, 1),
    ("_compile.compile_inner (dynamo_timed)", 58.345, 1),
    ("create_aot_dispatcher_function (dynamo_timed)", 56.318, 1),
    ("_recursive_joint_graph_passes (dynamo_timed)", 0.279, 1),
]

#: Real kernel names an A100 emits for Llama-3.1-8B fp16 with flash-attn.
_REAL_KERNELS = [
    "_two_tier_decode_kernel",
    "_gate_kernel",
    "triton_poi_fused_add_gather_mul_21",
    "triton_red_fused__to_copy_mul_sub_sum_11",
    "ampere_fp16_s16816gemm_fp16_256x64_ldg8_f2f_stages_64x3_tn",
    "void at::native::vectorized_elementwise_kernel<4, ...>",
    "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4>>",
    "Memcpy DtoD (Device -> Device)",
    "Memset (Device)",
]


@pytest.mark.parametrize("key,us,cnt", _REAL_COMPILE_ROWS)
def test_compile_ranges_are_not_kernels(key, us, cnt):
    ev = _Ev(key, us * 1000.0, cnt)
    assert _is_compile_range(ev), f"{key} not recognised as a compile range"
    assert not _is_device_event(ev), f"{key} still counted as a kernel"


@pytest.mark.parametrize("key", _REAL_KERNELS)
def test_real_kernels_survive(key):
    """The markers must not eat the kernels the compiler PRODUCED.

    `triton_*_fused_*` are Inductor's OUTPUT -- the eviction's actual work, and
    the thing a compile-contaminated profile needs to show. A marker list that
    caught them would hide exactly what it was added to reveal.
    """
    ev = _Ev(key, 1000.0, 32)
    assert not _is_compile_range(ev), f"{key} wrongly marked a compile range"
    assert _is_device_event(ev), f"{key} dropped from the kernels"


@pytest.mark.parametrize("op", [
    "aten::mm", "aten::copy_", "aten::topk", "aten::index_put_",
    "autograd::engine::evaluate_function", "ProfilerStep#42",
    "cudaLaunchKernel", "cudaMemcpyAsync", "cudaDeviceSynchronize",
])
def test_the_operator_view_is_not_a_kernel(op):
    assert not _is_device_event(_Ev(op, 1.0, 1))


def test_a_compile_range_is_rejected_even_when_tagged_as_a_device_event():
    """Hard reject, ahead of ``device_type``.

    On the build that produced that profile these ranges carried CUDA self time
    and passed the device-type arm.
    """
    pytest.importorskip("torch")
    from torch.autograd import DeviceType

    ev = _Ev("compile_fx_inner (dynamo_timed)", 268_588.0, 1)
    ev.device_type = DeviceType.CUDA
    assert _is_device_event(ev) is False


def test_the_double_count_that_inflated_every_number():
    """aten::mm at 225 launches against four gemm kernels summing to 225."""
    evs = [
        _Ev("ampere_fp16_s16816gemm_fp16_256x64_ldg8_f2f_stages_64x3_tn", 5115.0, 64),
        _Ev("ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x3_tn", 2657.0, 32),
        _Ev("ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stage", 2429.0, 65),
        _Ev("ampere_s16816gemm_fp16_128x64_ldg8_stages_32x6_tn", 630.0, 64),
        _Ev("aten::mm", 11052.0, 225),     # the same 225 launches, again
    ]
    kernels = [e for e in evs if _is_device_event(e)]
    assert sum(e.count for e in kernels) == 225
    assert abs(sum(e.self_device_time_total for e in kernels) - 10831.0) < 1.0


def test_busy_becomes_possible_once_the_ranges_are_dropped():
    """1071.94 - 726.82 = 345.1 ms/step. Over a 525.81 ms wall that is 65.6%,
    not the 203.9% the contaminated sum produced."""
    n = 24
    evs = [_Ev(k, ms * 1000.0 * n, c) for k, ms, c in _REAL_COMPILE_ROWS]
    for key, ms in (("_two_tier_decode_kernel", 14.480),
                    ("_gate_kernel", 2.441),
                    ("triton_poi_fused_add_gather_mul_21", 105.541),
                    ("ampere_fp16_s16816gemm_fp16_256x64_ldg8", 10.952),
                    ("void at::native::vectorized_elementwise_kernel<4, ...>", 171.580),
                    ("Memcpy DtoD (Device -> Device)", 35.699),
                    ("void at::native::index_elementwise_kernel<128, 4, ...>", 2.536),
                    ("void at::native::radixSortKVInPlace", 0.832),
                    ("void at::native::reduce_kernel<512, 1, ...>", 0.934),
                    ("void at::native::silu_kernel", 0.117)):
        evs.append(_Ev(key, ms * 1000.0 * n, 32 * n))

    gpu_us = sum(e.self_device_time_total for e in evs if _is_device_event(e))
    assert abs(gpu_us / n / 1000.0 - 345.11) < 0.5
    busy = gpu_us / (525.81 * 1000.0 * n)
    assert busy < 1.0, "a single stream cannot be more than 100% busy"
    assert 0.60 < busy < 0.70, f"busy={busy:.3f}, expected ~0.656"


def test_compile_inside_the_window_is_reported_not_silently_dropped(capsys):
    """Excluding the ranges must not make a 10x-slow profile look normal."""
    n = 24
    evs = [_Ev(k, ms * 1000.0 * n, c) for k, ms, c in _REAL_COMPILE_ROWS]
    _print_compile_contamination(evs, wall_us=525.81 * 1000.0 * n, n=n)
    out = capsys.readouterr().out

    assert "NOT A STEADY-STATE STEP" in out
    assert "CachingAutotuner" in out
    # It must not send the reader straight at --warmup; that reflex hides a
    # per-eviction recompile, which no warmup length absorbs.
    assert "recompile axis is the bug" in out


def test_a_clean_window_prints_nothing(capsys):
    _print_compile_contamination(
        [_Ev("_two_tier_decode_kernel", 14_480.0, 32)], wall_us=45_690.0, n=1)
    assert capsys.readouterr().out == ""
