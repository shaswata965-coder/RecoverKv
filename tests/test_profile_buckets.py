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
