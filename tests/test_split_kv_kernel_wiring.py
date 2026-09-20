"""Split-KV's *wiring*, for the half that cannot be executed here.

`triton` imports on this box but `torch.cuda.is_available()` is False, so the
kernel cannot run. What CAN be checked without a GPU is that the launch is
coherent: that the call site passes exactly the arguments the kernel declares,
that the phases are guarded where they have to be, and that `splits=1` really
is the pre-split path rather than a second path that resembles it.

Every one of these caught something or pins something that would otherwise be
caught only by a wrong number on a GPU.
"""

from __future__ import annotations

import ast
import io
import os
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "modules/windowed_cache/decode_kernel.py"
TEXT = io.open(SRC, encoding="utf-8").read()
TREE = ast.parse(TEXT)


def _fn(name):
    for n in ast.walk(TREE):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError(f"{name} not found")


def test_the_call_site_passes_exactly_what_the_kernel_declares():
    """Arity, positionally and by keyword.

    A Triton kernel takes its args positionally; one extra or one missing
    shifts every pointer after it, and the result is not a crash but garbage
    read from the wrong tensor.
    """
    kernel = _fn("_two_tier_decode_kernel")
    declared_pos = [a.arg for a in kernel.args.args]
    declared_kw = {a.arg for a in kernel.args.kwonlyargs} | {
        a.arg for a in kernel.args.args
        if a.annotation is not None
        and "constexpr" in ast.unparse(a.annotation)}

    call = None
    for n in ast.walk(_fn("_one")):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript):
            call = n
    assert call is not None, "no `_two_tier_decode_kernel[grid](...)` in _one()"

    passed_pos = len(call.args)
    n_runtime_declared = len([a for a in declared_pos if a not in declared_kw])
    assert passed_pos == n_runtime_declared, (
        f"kernel declares {n_runtime_declared} runtime args, call passes "
        f"{passed_pos}. A Triton kernel binds these positionally, so a "
        f"mismatch silently reinterprets every pointer after the gap.")

    passed_kw = {k.arg for k in call.keywords} - {"num_stages", "num_warps"}
    assert passed_kw == declared_kw, (
        f"constexpr mismatch.\n  declared not passed: {declared_kw - passed_kw}"
        f"\n  passed not declared: {passed_kw - declared_kw}")


def test_the_partials_are_declared_and_passed():
    kernel = _fn("_two_tier_decode_kernel")
    names = [a.arg for a in kernel.args.args]
    for n in ("PACC", "PL", "PM"):
        assert n in names, f"{n} is not a kernel parameter"
    assert "SPLITS" in names and "PHASE" in names


def test_phase_one_returns_before_the_epilogue():
    """Phase 1 must NOT run the epilogue -- it has no global lse to run it with.

    If the `return` ever goes missing, phase 1 would write an output computed
    from its own partial softmax, and phase 2 would then overwrite it. The
    result is correct-looking and wrong only in the windows the split did not
    own.
    """
    kernel = _fn("_two_tier_decode_kernel")
    phase1 = [n for n in ast.walk(kernel)
              if isinstance(n, ast.If) and "PHASE == 1" in ast.unparse(n.test)]
    assert phase1, "no `if PHASE == 1:` block in the kernel"
    assert any(isinstance(x, ast.Return) for x in ast.walk(phase1[0])), (
        "phase 1 does not return before the epilogue")


def test_the_gated_seed_is_skipped_when_split():
    """With several programs per (b, kv) nothing orders them, so an in-kernel
    seed of the Q columns could zero a column another split already scored.
    The dispatcher seeds on the host instead."""
    assert "if GATED and SPLITS == 1:" in TEXT, (
        "the GATED WSUM/WMAX seed is no longer guarded by SPLITS == 1; with "
        "splits > 1 each split would re-seed columns its siblings had written")
    disp = ast.unparse(_fn("_fused_two_tier_decode_impl")
                       if any(isinstance(n, ast.FunctionDef)
                              and n.name == "_fused_two_tier_decode_impl"
                              for n in ast.walk(TREE))
                       else _fn("fused_two_tier_decode"))
    assert "wmax[..., n_body_win:].fill_" in TEXT, (
        "the host-side seed that replaces it is missing")


def test_splits_one_allocates_no_partial_buffer():
    """The control arm must cost nothing, or it is not measuring the change."""
    assert "pacc = pl = pm = torch.zeros((1,)" in TEXT, (
        "splits=1 no longer falls back to a dummy pointer; it is allocating "
        "the partial scratch and the control arm has acquired a cost")


def test_splits_one_launches_once_and_on_the_old_grid():
    launch = ast.unparse(_fn("_launch"))
    assert "if splits == 1:" in launch and "_one(rung" in launch
    assert launch.count("_one(rung") == 3, (
        "expected exactly three launch sites: one for splits==1 and two for "
        f"the split path, found {launch.count('_one(rung')}")
    assert "(B * H_kv,), 2" in launch, (
        "phase 2 must run on the UNSPLIT grid -- it merges across splits, so "
        "one program per (b, kv)")


def test_the_two_phases_are_launched_in_order():
    """Phase 2 merges what phase 1 wrote. Separate launches on one stream are
    ordered; a single launch would not be."""
    launch = ast.unparse(_fn("_launch"))
    i1, i2 = launch.index("grid, 1)"), launch.index("(B * H_kv,), 2)")
    assert i1 < i2, "phase 2 is launched before phase 1"


# --- the resolver ------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("STICKYKV_DECODE_SPLITS", raising=False)


def _resolve():
    pytest.importorskip("torch")
    from modules.windowed_cache.decode_kernel import _resolve_splits
    return _resolve_splits


@pytest.mark.parametrize("raw", ["", "1", "off", "false", "no", "OFF"])
def test_the_default_is_off(monkeypatch, raw):
    if raw:
        monkeypatch.setenv("STICKYKV_DECODE_SPLITS", raw)
    assert _resolve()(32, 8, 60, 200) == 1


def test_auto_fills_the_machine_at_low_batch_and_backs_off_at_high(monkeypatch):
    monkeypatch.setenv("STICKYKV_DECODE_SPLITS", "auto")
    r = _resolve()
    # B=1: 8 blocks -> wants many splits;  B=32: 256 blocks -> already enough
    assert r(1, 8, 60, 200) > r(32, 8, 60, 200), (
        "auto must split harder where the grid is emptier")
    assert r(32, 8, 60, 200) == 1, "B=32 is already 256 blocks; auto should stop"


def test_splits_never_exceed_the_work(monkeypatch):
    """A split with no window costs a launch and an empty partial."""
    monkeypatch.setenv("STICKYKV_DECODE_SPLITS", "16")
    assert _resolve()(1, 8, 3, 2) == 3, "capped by the window count, not by 16"


@pytest.mark.parametrize("bad", ["auto2", "yes", "two", "1.5"])
def test_a_typo_raises_instead_of_falling_back(monkeypatch, bad):
    monkeypatch.setenv("STICKYKV_DECODE_SPLITS", bad)
    with pytest.raises(RuntimeError, match="STICKYKV_DECODE_SPLITS"):
        _resolve()(32, 8, 60, 200)


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_a_nonpositive_count_raises(monkeypatch, bad):
    monkeypatch.setenv("STICKYKV_DECODE_SPLITS", bad)
    with pytest.raises(RuntimeError, match=">= 1"):
        _resolve()(32, 8, 60, 200)


def test_the_setting_is_reported_for_provenance(monkeypatch):
    pytest.importorskip("torch")
    from modules.windowed_cache.decode_kernel import decode_splits_setting
    assert decode_splits_setting() == "1"
    monkeypatch.setenv("STICKYKV_DECODE_SPLITS", "auto")
    assert decode_splits_setting() == "auto"
