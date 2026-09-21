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
elements that straddle a rounding boundary. For the gate cards it is worse:
``sketch._q_up`` rounds **up** so that ``dequant >= x``, and that inequality is
the only reason the gate's Cauchy–Schwarz score is an *upper* bound — the
property the read gate uses to decide a window cannot matter. Against an
un-narrowed scale the bound silently stops holding.

These tests pin the fix (``cache._emulating_precision_casts``) at three levels:
the flag is actually applied and actually scoped, the three quantisers are
byte-identical under it, and ``_q_up``'s inequality survives.
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
    """The four quantisers, compiled exactly the way the eviction compiles."""
    return {
        name: _emulating_precision_casts(torch.compile(fn, dynamic=True))
        for name, fn in (("k", Q.quantize_key_windows),
                         ("v", Q.quantize_value_windows),
                         ("sym", S._q_sym), ("up", S._q_up))
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
            assert torch.equal(se, sc), f"{tier} scale, seed {seed}"
            assert torch.equal(ze, zc), f"{tier} zero, seed {seed}"
            n = int((pe != pc).sum())
            assert n == 0, (
                f"{tier} codes differ on {n}/{pe.numel()} packed bytes at seed "
                f"{seed}: the compiled eviction would store a different cache "
                f"than the eager reference every accuracy test validates.")


# ---------------------------------------------------------------------------
# the gate cards
# ---------------------------------------------------------------------------


def test_gate_card_codes_are_byte_identical_under_inductor(compiled):
    """``_q_sym`` / ``_q_up`` were never behind a graph break.

    So unlike the int2 tier this is not a new exposure — every compiled eviction
    since the cards landed built them on the elided grid.
    """
    for seed, x in _cases():
        f = x.to(torch.float32).reshape(6, 8, 64)
        ce, se = S._q_sym(f)
        cc, sc = compiled["sym"](f)
        assert torch.equal(se, sc) and torch.equal(ce, cc), f"_q_sym seed {seed}"

        pos = f.abs()
        ce, se = S._q_up(pos)
        cc, sc = compiled["up"](pos)
        assert torch.equal(se, sc) and torch.equal(ce, cc), f"_q_up seed {seed}"


def test_the_gate_bound_still_bounds_under_inductor(compiled):
    """``_q_up``'s guarantee, asserted directly rather than via byte identity.

    ``dequant >= x`` is what makes the card's Cauchy–Schwarz score an upper
    bound, and therefore what makes it safe for the read gate to skip a window
    on a low score. The ``(1 + 2**-9)`` scale nudge in ``_q_up`` is sized for an
    fp16 grid; fit against an un-narrowed one it is sized for the wrong grid and
    the inequality fails — 115 elements in this sample, measured without the fix.
    """
    for seed, x in _cases():
        pos = x.to(torch.float32).reshape(6, 8, 64).abs()
        codes, scale = compiled["up"](pos)
        under = int((S._dq_up(codes, scale) < pos).sum())
        assert under == 0, (
            f"seed {seed}: {under} elements dequantise BELOW the value they "
            "encode, so the gate's upper bound is not an upper bound.")


def test_the_int4_packing_is_byte_identical_under_inductor():
    """``t``'s nibble packing, compiled exactly the way the eviction compiles it.

    Unlike everything above, this one does not depend on
    ``emulate_precision_casts``: packing is integer work, with no fp16 round trip
    to elide. It is here because it is *new* integer work inside the eviction
    graph — uint8 masks, shifts and an or — and a miscompiled pack would write
    cards whose ``t`` the kernel then reads as different numbers entirely, with
    no assertion anywhere else to notice. Compiled separately from the fixture
    above because it takes int8 codes, not an fp16 window batch.
    """
    packed = _emulating_precision_casts(
        torch.compile(S.pack_nibbles_last, dynamic=True))
    for seed in SEEDS:
        torch.manual_seed(seed)
        codes = torch.randint(-7, 8, (3, 2, 8), dtype=torch.int8)
        assert torch.equal(S.pack_nibbles_last(codes), packed(codes)), (
            f"seed {seed}: the compiled eviction would pack `t` differently "
            "than the eager reference")
        assert torch.equal(
            S.unpack_nibbles_last(S.pack_nibbles_last(codes), 8), codes)
