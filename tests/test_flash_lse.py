"""CPU tests for the flash softmax_lse capture plumbing (flash_lse.py).

Real flash-attn / CUDA are absent on the dev box, so these drive the monkeypatch
against a FAKE ``flash_attn_func`` — validating the parts that are pure control
flow and thus GPU-independent: that capture is transparent to the attention
output, that ``L`` is stashed and popped, that an unsupported ``return_attn_probs``
latches off safely, that freshness holds, and that restore is clean.

The one thing these cannot check is that the REAL flash ``softmax_lse`` has the
value/shape we assume — that requires a GPU (see test_score_kernel.py's note).
"""

from __future__ import annotations

import types

import torch
import pytest

from modules.windowed_cache import flash_lse


def _fake_module(broken=False, out=None, lse=None):
    """A stand-in for transformers.modeling_flash_attention_utils."""
    ns = types.SimpleNamespace()
    seen = {"return_attn_probs": []}

    def faf(*args, **kwargs):
        seen["return_attn_probs"].append(kwargs.get("return_attn_probs", False))
        if kwargs.get("return_attn_probs"):
            if broken:
                raise TypeError("this build doesn't support return_attn_probs")
            return (out, lse, None)          # (attn_output, softmax_lse, S_dmask)
        return out                            # plain call

    ns.flash_attn_func = faf
    ns._seen = seen
    return ns


@pytest.fixture(autouse=True)
def _reset():
    """Keep the module-global stash from leaking between tests."""
    flash_lse.clear()
    yield
    flash_lse.clear()


def test_capture_is_transparent_and_stashes_lse():
    out = torch.ones(2, 4, 8)                 # pretend attn_output
    lse = torch.arange(2 * 4 * 5).float().reshape(2, 4, 5)  # [B, H_q, T]
    mod = _fake_module(out=out, lse=lse)

    handle = flash_lse.enable(mod)
    try:
        returned = mod.flash_attn_func(object(), object(), object(), 0.0,
                                       softmax_scale=0.1, causal=True)
        # Output the caller sees is byte-identical to flash's real output.
        assert torch.equal(returned, out)
        # ...and L was captured.
        got = flash_lse.pop()
        assert got is not None and torch.equal(got, lse)
        # pop clears it.
        assert flash_lse.pop() is None
        # The wrapper did ask flash for the probs.
        assert mod._seen["return_attn_probs"][-1] is True
    finally:
        handle.restore()


def test_broken_build_latches_off_and_stays_correct(monkeypatch):
    """The degrade path — now reachable only with STICKYKV_LSE_STRICT=0.

    The default became strict (see the companion test below and
    tests/test_lse_strict.py): a build that cannot hand back ``L`` costs a
    second O(N^2) prefill pass per layer plus the fp32 block that OOMs
    4096/batch-32, and paying that quietly is what let a whole benchmark
    campaign be recorded on the recompute path. The degrade still has to WORK
    when asked for, which is what this pins.
    """
    monkeypatch.setenv("STICKYKV_LSE_STRICT", "0")
    out = torch.ones(1, 2, 3)
    mod = _fake_module(broken=True, out=out, lse=None)

    handle = flash_lse.enable(mod)
    try:
        r1 = mod.flash_attn_func(1, 2, 3, causal=True)
        assert torch.equal(r1, out)           # output still correct
        assert flash_lse.pop() is None         # nothing captured
        # First call tried return_attn_probs (raised); it latched broken, so the
        # NEXT call must not even attempt it.
        n_calls_before = len(mod._seen["return_attn_probs"])
        r2 = mod.flash_attn_func(1, 2, 3, causal=True)
        assert torch.equal(r2, out)
        # The retry after the TypeError plus this call: none should pass True now.
        assert mod._seen["return_attn_probs"][-1] is False
        assert len(mod._seen["return_attn_probs"]) > n_calls_before
        # And the cause is retrievable either way — the OOM autopsy reads it.
        assert "return_attn_probs" in (flash_lse.broken_reason() or "")
    finally:
        handle.restore()


def test_broken_build_raises_by_default(monkeypatch):
    """The new default: no silent fallback, the failure is visible."""
    monkeypatch.delenv("STICKYKV_LSE_STRICT", raising=False)
    mod = _fake_module(broken=True, out=torch.ones(1, 2, 3), lse=None)

    handle = flash_lse.enable(mod)
    try:
        with pytest.raises(RuntimeError, match="return_attn_probs"):
            mod.flash_attn_func(1, 2, 3, causal=True)
    finally:
        handle.restore()


def test_freshness_varlen_layer_leaves_stash_empty():
    """clear() at forward start + no non-varlen call ⇒ pop() is None (fallback)."""
    out = torch.ones(2, 4, 8)
    lse = torch.zeros(2, 4, 5)
    mod = _fake_module(out=out, lse=lse)
    handle = flash_lse.enable(mod)
    try:
        # Layer A: normal non-varlen call fills the stash.
        mod.flash_attn_func(1, 2, 3, causal=True)
        assert flash_lse.pop() is not None
        # Layer B (varlen): pre-hook clears, and NO flash_attn_func runs.
        flash_lse.clear()
        assert flash_lse.pop() is None         # nothing stale leaks through
    finally:
        handle.restore()


def test_enable_none_when_symbol_absent():
    empty = types.SimpleNamespace()            # no flash_attn_func
    assert flash_lse.enable(empty) is None


def test_enable_is_idempotent_and_restore_is_clean():
    out = torch.ones(1, 1, 1)
    mod = _fake_module(out=out, lse=torch.zeros(1, 1, 1))
    original = mod.flash_attn_func

    h1 = flash_lse.enable(mod)
    wrapped = mod.flash_attn_func
    assert wrapped is not original
    assert getattr(wrapped, "_sticky_lse_wrapper", False) is True

    # Enabling again must not double-wrap.
    h2 = flash_lse.enable(mod)
    assert mod.flash_attn_func is wrapped

    h2.restore()
    h1.restore()
    assert mod.flash_attn_func is original      # exactly as we found it
