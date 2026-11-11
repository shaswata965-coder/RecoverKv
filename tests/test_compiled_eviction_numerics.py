"""The compiled eviction must write the same cache as the eager one.

``_evict_two_tier_impl`` runs under ``torch.compile`` on CUDA (the device
decides — see ``_compile_evict_enabled``) and eager on CPU. Those two are meant
to be the *same method*: the eager body is the reference every correctness test
in this suite validates, and the compiled one is a launch-coalescing rewrite of
it. If they disagree, every number in the perf table was produced by a method no
accuracy number describes.

They did disagree. Inductor does not emulate intermediate precision casts by
default: it drops an ``fp32 -> fp16 -> fp32`` round trip and keeps the wider
value in a register. Three quantisers inside the eviction graph take that round
trip **deliberately** — they fit codes to the fp16-stored grid so the grid the
codes were fit to is bit-identical to the grid every later dequant reads
(design §2). Eliding it fits the codes to one grid and reads them back on
another.

The damage is not uniform. For the int2 tier it is a code off by one level on
elements that straddle a rounding boundary. The same hazard runs through the
gate cards, whose encoder takes the same round trip, and through the int2
grid's own group scales, which are fp16 for the same reason the
property the read gate uses to decide a window cannot matter. Against an
codes are fit to them.

These tests pin the fix (``cache._emulating_precision_casts``) at three levels:
the flag is actually applied and actually scoped, the three quantisers are
byte-identical under it.
"""

from __future__ import annotations

import pytest
import torch

from modules.quant import quantizer as Q
from modules.quant import sketch as S
from modules.windowed_cache.cache import _emulating_precision_casts

inductor_config = pytest.importorskip("torch._inductor.config")

#: Enough seeds to catch it. The divergence is data-dependent — it only shows on
#: elements whose pre-round quantity lands within an fp16 ulp of a rounding
#: boundary — so a single tensor proves nothing. Measured without the fix on
#: torch 2.14: keys diverge on 11 of these, values on 14, the cards on all 24.
SEEDS = range(24)


def _cases():
    """A [N, H, window, D] fp16 window batch per seed, at varied magnitudes."""
    for seed in SEEDS:
        torch.manual_seed(seed)
        yield seed, (torch.randn(3, 2, 8, 64) * (0.5 + seed % 4)).to(torch.float16)


@pytest.fixture(scope="module")
def compiled():
    """The quantisers the eviction traces, compiled the way it compiles them."""
    return {
        name: _emulating_precision_casts(torch.compile(fn, dynamic=True))
        for name, fn in (("k", Q.quantize_key_windows),
                         ("v", Q.quantize_value_windows),
                         ("sym", S._q_sym), ("grid", Q._quantize_grid))
    }


# ---------------------------------------------------------------------------
# the wrapper itself
# ---------------------------------------------------------------------------


def test_the_wrapper_turns_the_flag_on_for_the_call():
    """Applied per call, not once at import — so it holds for a recompile too."""
    seen = []
    wrapped = _emulating_precision_casts(
        lambda: seen.append(inductor_config.emulate_precision_casts))
    wrapped()
    wrapped()
    assert seen == [True, True]


def test_the_wrapper_restores_the_process_default():
    """Scoped, not global: the host's own torch.compile calls are untouched.

    A library that flips a global Inductor config at import is a library that
    silently changes the numerics of everything else in the process.
    """
    before = inductor_config.emulate_precision_casts
    _emulating_precision_casts(lambda: None)()
    assert inductor_config.emulate_precision_casts is before

    # And it restores on the way out of a failure, not just a clean return.
    with pytest.raises(RuntimeError):
        _emulating_precision_casts(
            lambda: (_ for _ in ()).throw(RuntimeError("boom")))()
    assert inductor_config.emulate_precision_casts is before


# ---------------------------------------------------------------------------
# the int2 tier
# ---------------------------------------------------------------------------


def test_int2_codes_are_byte_identical_under_inductor(compiled):
    """Packed codes, scales and zeros, eager vs compiled, over every seed.

    ``_affine_quantize`` carried a ``@torch.compiler.disable`` for years, so this
    is newly reachable: it is traced now that the Inductor lowering bug it worked
    around (``aten.amin`` behind a data-dependent gather) is fixed upstream.
    """
    for seed, x in _cases():
        for tier, eager in (("key", Q.quantize_key_windows),
                            ("val", Q.quantize_value_windows)):
            pe, se, ze = eager(x)
            pc, sc, zc = compiled[tier[0]](x)
            assert torch.equal(se.q, sc.q), f"{tier} scale codes, seed {seed}"
            assert torch.equal(se.s, sc.s), f"{tier} scale group, seed {seed}"
            assert torch.equal(ze.q, zc.q), f"{tier} zero codes, seed {seed}"
            assert torch.equal(ze.s, zc.s), f"{tier} zero group, seed {seed}"
            n = int((pe != pc).sum())
            assert n == 0, (
                f"{tier} codes differ on {n}/{pe.numel()} packed bytes at seed "
                f"{seed}: the compiled eviction would store a different cache "
                f"than the eager reference every accuracy test validates.")


# ---------------------------------------------------------------------------
# the gate cards
# ---------------------------------------------------------------------------


def test_gate_card_codes_are_byte_identical_under_inductor(compiled):
    """``_q_sym`` was never behind a graph break.

    So unlike the int2 tier this is not a new exposure — every compiled eviction
    since the cards landed built them on the elided grid.
    """
    for seed, x in _cases():
        f = x.to(torch.float32).reshape(6, 8, 64)
        ce, se = S._q_sym(f)
        cc, sc = compiled["sym"](f)
        assert torch.equal(se, sc) and torch.equal(ce, cc), f"_q_sym seed {seed}"


def test_the_grid_group_scale_survives_inductor(compiled):
    """The int2 grid's own fp16 round trip, isolated.

    ``_quantize_grid`` fits a byte to an fp16 group scale, so it takes the same
    ``fp32 -> fp16 -> fp32`` trip the tier quantisers do, one level down. A grid
    that decodes differently compiled than eager is a cache whose int2 codes were
    fit against a grid the reader will not reproduce — the same defect as the
    tier's, on the values every one of its codes is multiplied by.
    """
    for seed, x in _cases():
        v = x.to(torch.float32).reshape(6, 8, 64)
        for signed, val in ((False, v.abs()), (True, v)):
            ge = Q._quantize_grid(val, signed=signed)
            gc = compiled["grid"](val, signed=signed)
            assert torch.equal(ge.q, gc.q), f"grid codes, signed={signed}, seed {seed}"
            assert torch.equal(ge.s, gc.s), f"group scale, signed={signed}, seed {seed}"
            assert torch.equal(ge.decode(), gc.decode())
