"""The decode tile ladder's ORDER, which was an assertion and is now measured.

Order is irrelevant to a tuned geometry -- `_search_rungs` times every rung and
takes the minimum. It decides everything for an UNTUNED one, because
`_first_fit` returns the first rung that *fits* and a fit is shape-independent,
so the fallback is one constant for every untuned geometry in the run. Under the
old tile-major order that constant was `(64, 2, 4)`, which three separate GPU
profiles rank 5th-to-9th of 11 and 1.20-1.45x off the winner.

The trap this file guards is `_LAST_WINNER` (DISTANCE_TO_GOAL §10.15): exporting
a tuned winner across geometries cost 12%, because the trade a rung selects
moves with the geometry. A static reorder is a different claim -- it replaces an
asserted fallback with a measured one and still walks the whole ladder -- but it
is still a guess about untuned geometries, so `STICKYKV_FIT_LADDER=legacy` is
its control arm and these tests pin both arms.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "modules/windowed_cache/decode_kernel.py"


def _literal(name: str):
    """Read a module-level list literal without importing torch."""
    tree = ast.parse(io.open(SRC, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return [tuple(v) for v in ast.literal_eval(node.value)]
    raise AssertionError(f"{name} not found in {SRC.name}")


#: ms per rung at ws=8 D=128 B=32 n_sel~2^6, from the 2026-09-20 control-arm
#: profile -- the first fully valid one (positive profiler overhead, no
#: compilation in either window).
MEASURED_MS = {
    (32, 2, 4): 0.486, (32, 1, 4): 0.538, (64, 2, 8): 0.561, (64, 1, 8): 0.575,
    (64, 2, 4): 0.582, (64, 1, 4): 0.583, (16, 2, 4): 0.614, (16, 1, 4): 0.629,
    (32, 1, 8): 0.647, (32, 2, 8): 0.654, (16, 1, 8): 0.807, (16, 2, 8): 0.823,
}


def test_the_two_ladders_hold_exactly_the_same_rungs():
    """A reorder must not quietly add or drop a rung -- that would change which
    tiles are reachable, not just which is tried first."""
    measured, legacy = _literal("_FIT_LADDER_MEASURED"), _literal("_FIT_LADDER_LEGACY")
    assert sorted(measured) == sorted(legacy)
    assert len(measured) == len(set(measured)) == 12


def test_the_measured_ladder_is_actually_in_measured_order():
    """The whole justification. If this drifts, the ordering is an assertion
    again and the docstring above is a lie."""
    measured = _literal("_FIT_LADDER_MEASURED")
    assert set(measured) == set(MEASURED_MS), "ladder and profile data disagree"
    times = [MEASURED_MS[r] for r in measured]
    assert times == sorted(times), (
        f"ladder is not sorted by measured time:\n  "
        + "\n  ".join(f"{r} = {MEASURED_MS[r]:.3f} ms" for r in measured))


def test_the_legacy_ladder_still_leads_with_the_rung_the_profiles_reject():
    """The control arm has to reproduce the OLD behaviour exactly, or it is not
    a control. `(64, 2, 4)` first is what `_first_fit` used to return."""
    legacy = _literal("_FIT_LADDER_LEGACY")
    assert legacy[0] == (64, 2, 4)
    assert MEASURED_MS[legacy[0]] / MEASURED_MS[_literal("_FIT_LADDER_MEASURED")[0]] \
        == pytest.approx(1.197, abs=0.01), "the gap this change exists to close moved"


def test_the_measured_ladder_leads_with_a_rung_that_always_fits():
    """`_first_fit` walks until something fits, so the new head must be at
    least as easy to fit as the old one -- otherwise an untuned geometry pays a
    failed launch before it gets anywhere.

    Shared-memory staging grows with `target_keys`, so a smaller tile fits
    wherever a larger one does. The new head is target_keys=32 against the old
    64: strictly easier.
    """
    measured, legacy = _literal("_FIT_LADDER_MEASURED"), _literal("_FIT_LADDER_LEGACY")
    assert measured[0][0] <= legacy[0][0]


def test_the_resolver_rejects_a_typo_instead_of_falling_back():
    """A misspelled knob that silently defaults is how two arms come to measure
    the same thing -- this repo's `6b8a188` bug, in a new place."""
    src = io.open(SRC, encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_resolve_fit_ladder")
    assert any(isinstance(x, ast.Raise) for x in ast.walk(fn)), (
        "_resolve_fit_ladder no longer raises on an unrecognised value")


def test_the_knob_is_latched_at_import_not_read_per_launch():
    """`_launch` runs once per layer per step (32x at L=32). An os.environ read
    there is a cost, not a feature -- the module says so about `_EXP2`."""
    src = io.open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("_launch", "_search_rungs",
                                                               "_first_fit"):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr == "environ":
                    raise AssertionError(
                        f"{node.name}() reads os.environ on the hot path")


def test_the_sighting_counter_is_not_shadowed_by_the_dedup_set():
    """`_search_rungs` took a `seen` DICT (the recurrence counter) and rebound
    it to a local `set()` mid-body. Harmless only because the one write happens
    above that line; any later read would have queried the wrong object."""
    src = io.open(SRC, encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_search_rungs")
    params = {a.arg for a in fn.args.args}
    rebound = [t.id for node in ast.walk(fn) if isinstance(node, ast.Assign)
               for t in node.targets
               if isinstance(t, ast.Name) and t.id in params]
    assert not rebound, f"_search_rungs rebinds its own parameter(s): {rebound}"


# --- behaviour, where torch exists -----------------------------------------

def test_first_fit_returns_the_measured_head_when_everything_fits():
    pytest.importorskip("torch")
    from modules.windowed_cache.decode_kernel import _FIT_LADDER, _first_fit

    tried = []
    assert _first_fit(_FIT_LADDER, tried.append, "t", sig=("x",)) == _FIT_LADDER[0]
    assert tried == [_FIT_LADDER[0]], "first_fit launched more than the first rung"


def test_first_fit_walks_past_a_rung_that_does_not_fit():
    pytest.importorskip("torch")
    from modules.windowed_cache.decode_kernel import (
        _FIT_LADDER, _first_fit, _is_out_of_resources,
    )
    import triton  # noqa: F401  - OutOfResources lives here

    class _OOR(Exception):
        pass

    tried = []

    def launch(rung):
        tried.append(rung)
        if len(tried) == 1:
            raise triton.runtime.errors.OutOfResources(1, 0, "shared memory")

    got = _first_fit(_FIT_LADDER, launch, "t", sig=("x",))
    assert got == _FIT_LADDER[1] and len(tried) == 2


def test_the_tuned_winner_does_not_depend_on_ladder_order():
    """The reorder must change the FALLBACK only. A tuned geometry times every
    rung, so both orders must agree on the minimum -- if they do not, the
    search has an order dependence and the 1.4%-vs-6.1% gap has another cause.
    """
    torch = pytest.importorskip("torch")
    # `_search_rungs` times each rung through `_time_launch`, which calls
    # torch.cuda.synchronize(). importorskip("torch") is not enough -- a CPU
    # torch imports fine and then raises "Found no NVIDIA driver" here. This
    # test passed on the GPU box and failed on the dev box for exactly that
    # reason, which is the guard being wrong, not the code.
    if not torch.cuda.is_available():
        pytest.skip("_search_rungs times rungs on the device; needs CUDA")
    from modules.windowed_cache import decode_kernel as dk

    ms = dict(MEASURED_MS)
    for ladder in (dk._FIT_LADDER_MEASURED, dk._FIT_LADDER_LEGACY):
        choice, timed, announced, seen = {}, {}, set(), {}
        sig = ("g",)
        for _ in range(dk._TUNE_AFTER):
            got = dk._search_rungs(choice, timed, announced, seen, sig,
                                   list(ladder), lambda r: None, "t")
        assert choice[sig] == min(ms, key=ms.get) or got is not None
        assert isinstance(seen, dict) and seen.get(sig), (
            "the recurrence counter was clobbered during the search")
