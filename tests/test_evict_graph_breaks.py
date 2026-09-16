"""The compiled eviction must be ONE graph, not seven.

``STICKYKV_COMPILE_EVICT`` exists for exactly one reason: the eviction step is
~493 launching ops against a steady step's 80, and compiling it fuses that
pointwise/gather/scatter chain into a handful of kernels. ``1434b2c`` prices the
difference at TPOT 101.6 -> 85 ms.

**Fusion cannot cross a graph break.** So "the eviction is compiled" and "the
eviction is fused" are different claims, and until this file existed only the
first one was checked — ``tests/test_quant_cache.py`` asserts the compiled body
writes the same cache as the eager one (semantics), and
``tests/test_compiled_eviction_numerics.py`` asserts the quantisers agree under
``emulate_precision_casts`` (numerics). Neither notices a body that traces,
breaks eight times, and lowers to the same sixteen ATen elementwise kernels the
flag was set to remove.

That is what was happening. ``CacheState.replace`` and
``CacheState.replace_body`` guard against a source aliasing the buffer it is
about to be written into, and the only way to ask that is
``untyped_storage().data_ptr()``, which Dynamo cannot trace:

    Unsupported method call
    Explanation: Dynamo does not know how to trace method `data_ptr`
    of class `UntypedStorage`

Measured on torch 2.14 CPU Inductor over one eviction cycle: **8 graph breaks,
7 Inductor graphs.** With ``state._tracing()`` gating the guards: **0 breaks.**
A GPU profile of the same build showed the rollup's ``ours: compiled evict``
bucket — which matches ``triton_{poi,red,tem,for,unk}_fused`` — contributing
exactly 0.000 ms while ``ours (cache)`` came to 7.917 ms, the two hand-written
Triton kernels to the milligram.

This file is the regression guard. It asserts on the break **count**, because a
break's cost is not a number this suite can measure — only a GPU can — and zero
is the only value that needs no measurement to interpret.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
dynamo = pytest.importorskip("torch._dynamo")
pytest.importorskip("torch._inductor")

from tests.test_quant_cache import _run_decode_across_eviction  # noqa: E402


def _evict_with_counters(backend: str):
    """One eviction cycle through ``torch.compile``; return Dynamo's counters."""
    dynamo.reset()
    dynamo.utils.counters.clear()
    _run_decode_across_eviction(compile_backend=backend)
    breaks = dict(dynamo.utils.counters["graph_break"])
    graphs = dynamo.utils.counters["stats"].get("unique_graphs", 0)
    return breaks, graphs


def test_compiled_eviction_has_no_graph_breaks():
    """The eviction body traces end to end.

    A break here is not a style point. Each one splits the body into fragments
    that Inductor compiles independently, so the quantiser's round/clamp/div
    chain cannot fuse with the gather that feeds it or the scatter that consumes
    it — which is the entire saving the flag is set to collect.
    """
    breaks, _ = _evict_with_counters("inductor")
    assert not breaks, (
        "the compiled eviction graph-breaks, so it cannot fuse:\n"
        + "\n".join(f"  {n}x  {reason.splitlines()[0]}"
                    for reason, n in breaks.items())
        + "\n\nEach break splits _evict_two_tier_impl into a separate Inductor "
          "graph. If this is a new `data_ptr` / `untyped_storage` call, gate it "
          "on `modules.windowed_cache.state._tracing()` the way `replace` and "
          "`replace_body` do."
    )


def test_compiled_eviction_is_a_small_number_of_graphs():
    """One graph per distinct shape, not one per guard.

    Two is the expected count for this fixture: the first eviction takes the
    per-layer ``replace`` path (it is what releases the prompt-sized buffer) and
    later ones take ``replace_body``. The bound is deliberately loose — what it
    refuses is the 7 that the `data_ptr` breaks produced.
    """
    _, graphs = _evict_with_counters("inductor")
    assert graphs <= 3, (
        f"the eviction lowered to {graphs} Inductor graphs; at most 3 are "
        "expected (one per shape/path). More means the body is being split — "
        "check for graph breaks with test_compiled_eviction_has_no_graph_breaks."
    )


def test_tracing_guard_is_off_in_eager():
    """The aliasing guards are skipped **only** while Dynamo traces.

    Eager is where the whole CPU suite runs, so this is what says the checks
    still have coverage rather than having been deleted.
    """
    from modules.windowed_cache.state import _tracing
    assert _tracing() is False


def test_replace_still_clones_an_aliasing_source_in_eager():
    """``replace`` repairs a self-overlapping copy — in eager, as before."""
    from modules.windowed_cache.state import CacheState

    st = CacheState(capacity=8)
    k = torch.randn(1, 2, 4, 3)
    st.replace(k, k.clone(), torch.arange(4).unsqueeze(0))

    # A view of the live buffer aliases it; replace must not copy it onto itself.
    aliased_k = st.key_states
    aliased_v = st.value_states
    aliased_p = st.position_ids
    expected = aliased_k.clone()
    st.replace(aliased_k, aliased_v, aliased_p)
    assert torch.equal(st.key_states, expected)


def test_replace_body_still_refuses_an_aliasing_source_in_eager():
    """``replace_body`` asserts rather than repairing — in eager, as before."""
    from modules.windowed_cache.state import CacheState

    st = CacheState(capacity=8)
    st.replace(torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3),
               torch.arange(4).unsqueeze(0))
    body_k = st.key_states[:, :, 1:, :]          # a view of the key buffer
    with pytest.raises(RuntimeError, match="aliases the key buffer"):
        st.replace_body(1, body_k, torch.randn(1, 2, 3, 3),
                        torch.arange(1, 4).unsqueeze(0))
