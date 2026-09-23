"""Compile the Triton kernels for a CUDA target on a box with no CUDA.

The kernels here have always shipped "unvalidated by construction" because the
dev box is CPU-only. That is true of *launching* them. It is not true of
**compiling** them: ``triton.compile`` takes an explicit ``GPUTarget``, so the
whole front end — AST, type inference, broadcasting, constexpr folding, every
``tl.*`` signature — runs without a device.

That is worth having, because the class of bug it catches is exactly the class
that a CPU oracle cannot: a lane mask that does not broadcast, a scalar where a
tensor is required, an ``arange`` whose length is not a power of two. Those never
show up in ``two_tier_window_reference``; they show up on the first GPU run, at
the end of a long feedback loop.

What this still does not prove: that the compiled kernel computes the right
numbers. The oracle in ``tests/test_gated_fused_path.py`` is what says that.
Compile here, compare there, and the only thing left for a GPU is codegen quality
and speed.
"""

from __future__ import annotations

import pytest

triton = pytest.importorskip("triton", reason="Triton front end not installed")

from triton.backends.compiler import GPUTarget            # noqa: E402
from triton.compiler import ASTSource, compile as tcompile  # noqa: E402

import triton.language as tl                              # noqa: E402

from modules.windowed_cache import gate_kernel as gk      # noqa: E402
from modules.windowed_cache import decode_kernel as dk    # noqa: E402
from modules.windowed_cache.decode_kernel import window_tiling  # noqa: E402
from modules.quant.quantizer import grid_group                # noqa: E402

#: An A100. Any real target exercises the same front end; the arch only matters
#: for codegen, which is not what is being checked.
TARGET = GPUTarget("cuda", 80, 32)


def _compile(fn, ptrs, consts):
    sig = {n: "i32" for n in fn.arg_names}
    sig.update(ptrs)
    for k in consts:
        sig[k] = "constexpr"
    tcompile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=TARGET)


#: The int2 grid is one byte per entry (u8 scale, i8 zero) against an fp16
#: scale per group -- `*fp16` here would type-check a kernel nobody runs.
_DECODE_PTRS = {
    **{n: "*fp16" for n in ("Q", "KFP", "VFP", "COS", "SIN", "OUT",
                            "KSS", "KZS", "VSS", "VZS", "VMS")},
    **{n: "*u8" for n in ("KC", "VC", "KS", "VS")},
    **{n: "*i8" for n in ("KZ", "VZ", "VM")},
    "SEL": "*i32",
    "WSUM": "*fp32", "WMAX": "*fp32", "LOGM": "*fp32", "scale": "fp32",
    "VANC": "*fp32", "KANC": "*fp32",
}

#: ``MU`` is ``*u8``: the card packs ``mu`` as int4 nibbles in a uint8
#: (``sketch._q_sym4``), and the kernel shifts it -- an ``*i8`` here type-checked
#: an arithmetic shift nobody runs. ``PMX``/``PSM`` are the HEAD_NORM partials.
_GATE_PTRS = {
    **{n: "*fp16" for n in ("Q", "MUS", "VS", "TS", "ANCH")},
    "MU": "*u8",
    **{n: "*i8" for n in ("V", "T")},
    "EST": "*fp32", "LOGM": "*fp32", "PMX": "*fp32", "PSM": "*fp32",
    "SCALE": "fp32",
}

_UNION_PTRS = {n: "*fp32" for n in ("ESTH", "PMX", "PSM", "EST")}

#: One ``ws`` per structural case in :func:`window_tiling`, not a sweep — each
#: entry is a real compile, and they are slow. ``1`` is the degenerate
#: ``BLOCK_NW = 64``; ``8`` the typical whole-tile case; ``12`` the non-power-of-2
#: that leaves padding lanes; ``128`` the window that fills a tile on its own.
WINDOW_SIZES = [1, 8, 12, 128]


def test_the_harness_reports_a_compile_error_rather_than_swallowing_it():
    """Guards the rest of this file against passing vacuously.

    If ``tcompile`` ever stopped raising — a changed API, a target that silently
    no-ops — every test below would go green while checking nothing.
    """
    @triton.jit
    def _broken(X, N: tl.constexpr):
        # A pointer is not a float; type inference must reject this.
        tl.store(X + tl.arange(0, N), X + 1.0)

    with pytest.raises(Exception):
        _compile(_broken, {"X": "*fp32"}, {"N": 16})


@pytest.mark.parametrize("key_anchor", [False, True])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("ws", WINDOW_SIZES)
def test_the_fused_decode_kernel_compiles(gated, ws, key_anchor):
    """Every GATED x KEY_ANCHOR specialization, at every window size
    ``window_tiling`` allows.

    Both are constexprs, so these are genuinely different kernels: the gated one
    carries the ``SEL`` dereference and the skipped-column prologue, the ungated
    one folds both away; the anchored one (Qwen2's biased keys) loads the key
    anchor and adds it to the decoded zero, the un-anchored one (Llama, Mistral)
    compiles exactly the kernel it always did. Compiling only some would leave
    the others' front end unchecked.
    """
    block_nw, block_t = window_tiling(ws)
    _compile(dk._two_tier_decode_kernel, _DECODE_PTRS, dict(
        HEAD_DIM=128, HALF=64, WS=ws, BLOCK_R=4, BLOCK_NW=block_nw,
        BLOCK_T=block_t, BLOCK_W=16, PACK_K=max(ws // 4, 1), PACK_V=32,
        GROUP_K=grid_group(128), GROUP_V=grid_group(ws),
        GATED=gated, KEY_ANCHOR=key_anchor))


@pytest.mark.parametrize("head_norm", [False, True])
@pytest.mark.parametrize("ws", WINDOW_SIZES)
def test_the_gate_kernel_compiles(ws, head_norm):
    """``_gate_kernel`` at every window size — the one that found a real bug.

    It indexed its token axis with ``tl.arange(0, WS)``, which admits only powers
    of two, so a ``window_size`` of 12 (which ``window_tiling`` documents and
    ``tests/test_window_scores.py`` exercises) could not compile at all. Now the
    axis is padded to ``BLOCK_WS`` and masked.

    Both ``HEAD_NORM`` specializations: per-query-head ``est`` plus per-tile
    partials (the GQA union in head units), or the in-kernel group max.
    """
    _compile(gk._gate_kernel, _GATE_PTRS, dict(
        HEAD_DIM=128, WS=ws, BLOCK_WS=triton.next_power_of_2(ws),
        BLOCK_D=128, BLOCK_W=64, HEAD_NORM=head_norm))


@pytest.mark.parametrize("block_r", [2, 4, 8])
def test_the_gate_union_kernel_compiles(block_r):
    """``_gate_union_kernel`` at every ``BLOCK_R`` a GQA model reaches with
    ``rep > 1`` (``_gate_triton`` skips it at ``rep == 1``): 2, 4 (Llama-3.1),
    8 (Qwen2.5-7B's rep=7, one padded head lane)."""
    _compile(gk._gate_union_kernel, _UNION_PTRS, dict(
        BLOCK_R=block_r, BLOCK_T=gk._UNION_BLOCK_T, BLOCK_U=gk._UNION_BLOCK_U))


def test_the_qwen25_gate_specialization_compiles():
    """The gate Qwen2.5 launches: a bf16 query, ``HEAD_NORM`` on (its
    projections carry a bias, so ``quant_gate_head_norm`` resolves on), and the
    union kernel at rep = 28 / 4 = 7."""
    ptrs = dict(_GATE_PTRS)
    ptrs["Q"] = "*bf16"
    _compile(gk._gate_kernel, ptrs, dict(
        HEAD_DIM=128, WS=8, BLOCK_WS=8, BLOCK_D=128, BLOCK_W=64, HEAD_NORM=True))
    _compile(gk._gate_union_kernel, _UNION_PTRS, dict(
        BLOCK_R=triton.next_power_of_2(7), BLOCK_T=gk._UNION_BLOCK_T,
        BLOCK_U=gk._UNION_BLOCK_U))


@pytest.mark.parametrize("ws", [3, 5, 7, 12, 24, 48])
def test_padding_lanes_cannot_reach_the_gate_kernels_reductions(ws):
    """The masking half of the ``ws`` fix, checked as arithmetic.

    Padding a ``ws``-token axis up to a power of two is only safe if the surplus
    lanes are removed from ``max`` and ``logsumexp``. Zeroing their ``t`` is not
    enough — ``x`` would still be ``SCALE * m``, a perfectly plausible logit for a
    token sitting exactly at the window's mean, and every window would be scored
    as if it held one more token than it does. They must be ``-inf``.
    """

    import torch

    block_ws = triton.next_power_of_2(ws)
    w = torch.arange(block_ws)
    wm = w < ws
    x = torch.randn(4, block_ws)
    padded = torch.where(wm, x, torch.full_like(x, -float("inf")))
    zeroed = torch.where(wm, x, torch.zeros_like(x))

    torch.testing.assert_close(padded.amax(-1), x[:, :ws].amax(-1))
    torch.testing.assert_close(padded.logsumexp(-1), x[:, :ws].logsumexp(-1))
    if block_ws != ws:
        assert not torch.allclose(zeroed.logsumexp(-1), x[:, :ws].logsumexp(-1)), (
            "this test cannot distinguish the two treatments; fixture is wrong")


def test_the_qwen25_production_specialization_compiles():
    """The kernel Qwen2.5 actually launches, which no case above is.

    bf16 everywhere a Qwen2.5 cache is bf16 -- q/k/v, the RoPE tables, the output
    and, since the slot table stores group scales in ``grid_dtype_for(kv)``, the
    grid scales too (they were fp16 whatever the cache). ``BLOCK_R`` is what the
    dispatcher picks for rep = 28 / 4 = 7, gated as shipped, and ``KEY_ANCHOR``
    on, because Qwen2's ``k_proj`` has a bias. ``*bf16`` grid pointers are the
    part the fp16 harness above never type-checks.
    """
    ptrs = dict(_DECODE_PTRS)
    ptrs.update({n: "*bf16" for n in ("Q", "KFP", "VFP", "COS", "SIN", "OUT",
                                      "KSS", "KZS", "VSS", "VZS")})
    ws = 8
    block_nw, block_t = window_tiling(ws)
    _compile(dk._two_tier_decode_kernel, ptrs, dict(
        HEAD_DIM=128, HALF=64, WS=ws, BLOCK_R=dk._pow2_at_least(7),
        BLOCK_NW=block_nw, BLOCK_T=block_t, BLOCK_W=16, PACK_K=ws // 4,
        PACK_V=32, GROUP_K=grid_group(128), GROUP_V=grid_group(ws),
        GATED=True, KEY_ANCHOR=True))
