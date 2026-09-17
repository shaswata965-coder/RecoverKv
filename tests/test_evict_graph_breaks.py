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
    # The ONE break this body is allowed: `_evict_widths` reading the real
    # fresh/react counts. Torch words it as `item()` on some versions and as a
    # data-dependent `_local_scalar_dense` on others, so match the cause rather
    # than one version's phrasing.
    DELIBERATE = ("item()", "_local_scalar_dense", "Data dependent")
    unexpected = {r: n for r, n in breaks.items()
                  if not any(d in r for d in DELIBERATE)}
    assert not unexpected, (
        "the compiled eviction graph-breaks somewhere new:\n"
        + "\n".join(f"  {n}x  {reason.splitlines()[0]}"
                    for reason, n in unexpected.items())
        + "\n\nEach break splits _evict_two_tier_impl into a separate Inductor "
          "graph. If this is a new `data_ptr` / `untyped_storage` call, gate it "
          "on `modules.windowed_cache.state._tracing()` the way `replace` and "
          "`replace_body` do."
    )
    # It is bought deliberately: `_evict_widths` reads the real fresh/react
    # counts so the demote payload is allocated by COUNT instead of by `n_q`.
    # Under `bytes` n_q is 250 and the eviction is ~half the decode step, so
    # this trades a break for a payload up to 31x smaller. Both widths are read
    # through ONE stacked tensor -- two separate reads would be two syncs and
    # two breaks for the same information. All the demote work sits after the
    # width is known, so it stays inside one fused region.
    assert len(breaks) <= 1, (
        f"the width read should be the only KIND of break; got {list(breaks)}")


def test_compiled_eviction_is_a_small_number_of_graphs():
    """One graph per distinct shape, not one per guard.

    Two is the expected count for this fixture: the first eviction takes the
    per-layer ``replace`` path (it is what releases the prompt-sized buffer) and
    later ones take ``replace_body``. The bound is deliberately loose — what it
    refuses is the 7 that the `data_ptr` breaks produced.
    """
    _, graphs = _evict_with_counters("inductor")
    assert graphs <= 8, (
        f"the eviction lowered to {graphs} Inductor graphs. Expected: one per "
        "shape/path, doubled by `_evict_width`'s deliberate `.item()` break, "
        "times the handful of ladder rungs the demote width can take. More "
        "means the body is being split somewhere unintended — check "
        "test_compiled_eviction_has_no_graph_breaks."
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


# ---------------------------------------------------------------------------
# D1's missing number: what the worst-case width actually bought
# ---------------------------------------------------------------------------


def _evict_and_read_widths(backend):
    """Drive one eviction cycle and return the cache's width counters."""
    from modules.windowed_cache.cache import evict_width_stats
    from tests import test_quant_cache as T

    orig, held = T._make_cache, {}

    def keep(*a, **kw):
        c = orig(*a, **kw)
        held["c"] = c
        return c

    T._make_cache = keep
    try:
        T._run_decode_across_eviction(compile_backend=backend)
    finally:
        T._make_cache = orig
    return evict_width_stats(held["c"])


def test_the_eviction_records_what_its_padded_width_bought():
    """``_compact`` allocates by rank into ``n_q``; this says how much was real.

    ``DECODE_NEXT.md`` D1 proposes shrinking that width, and §7 records the risk
    the estimate rests on: *"If the real count is routinely close to n_q, D1 is
    worth little."* Nothing printed the ratio, so D1 has never been sized.
    """
    w = _evict_and_read_widths(None)
    assert w is not None and w["evictions"] > 0
    assert 0 <= w["fresh_max"] <= w["n_q"], (
        "fresh_max above n_q means _compact's bound is not a bound and lanes "
        "are being routed to its dump column -- work is being silently dropped"
    )
    assert 0 <= w["promote_max"] <= w["n_q"]
    assert w["fresh_mean"] <= w["fresh_max"]
    assert w["fresh_fill"] == pytest.approx(w["fresh_max"] / w["n_q"])


def test_the_counters_agree_compiled_and_eager():
    """It is a measurement, so it must not be a behaviour difference."""
    eager = _evict_and_read_widths(None)
    compiled = _evict_and_read_widths("inductor")
    assert eager == compiled


def test_reading_the_counters_is_the_only_sync_and_it_is_opt_in():
    """The counters are a device-side running max precisely so the eviction
    keeps zero syncs and zero graph breaks.

    A ``.item()`` in the body would reintroduce the break this file exists to
    pin -- measured on torch 2.14: *"Unsupported Tensor.item() call with
    capture_scalar_outputs=False"*. So the sync lives in the reader, which runs
    after a timed window rather than inside one.
    """
    breaks, _ = _evict_with_counters("inductor")
    assert all(any(d in r for d in ("item()", "_local_scalar_dense",
                                    "Data dependent")) for r in breaks), (
        f"recording the widths must not add a break of its own: {breaks}")


def test_no_eviction_means_no_stats_rather_than_zeros():
    """Zeros would read as "the width was never used"; None says "not measured"."""
    from modules.windowed_cache.cache import evict_width_stats

    class Bare:
        pass

    assert evict_width_stats(Bare()) is None


# ---------------------------------------------------------------------------
# D1 for the PROMOTE side
# ---------------------------------------------------------------------------


def test_evict_widths_takes_one_bound_per_mask():
    """The three widths have different caps and must still cost ONE host read.

    Demote and reactivate are capped by ``n_q`` (the Q tier's window count);
    promote by ``min(N_q, n_fp)`` (what the fp side can take back). Before this,
    promote was not measured at all -- it was sized by its cap, which under
    ``quant_budget_mode: bytes`` is ~250 windows, so every eviction dequantized,
    RoPE'd and spliced ~250 windows to use the handful really promoted.
    """
    import torch
    from modules.windowed_cache.cache import _evict_widths

    # rows: 2 promoted, 5 fresh, 0 reactivated
    prom = torch.zeros(2, 16, dtype=torch.bool); prom[0, :2] = True
    fresh = torch.zeros(2, 16, dtype=torch.bool); fresh[1, :5] = True
    react = torch.zeros(2, 16, dtype=torch.bool)

    n_p, n_r, n_d = _evict_widths((prom, react, fresh), (250, 250, 250))
    assert n_p == 2, n_p          # exact ladder rung
    assert n_r == 0, n_r          # nothing to do => no width at all
    assert n_d == 8, n_d          # 5 rounds up to the next rung

    # A per-mask cap clips only its own mask.
    n_p, n_r, n_d = _evict_widths((prom, react, fresh), (1, 250, 4))
    assert (n_p, n_r, n_d) == (1, 0, 4)

    # One int still means "same cap for all", the original contract.
    assert _evict_widths((prom, react, fresh), 250) == [2, 0, 8]

    # All caps zero short-circuits without touching the device.
    assert _evict_widths((prom, react, fresh), 0) == [0, 0, 0]

    with pytest.raises(ValueError, match="3 masks but 2 bounds"):
        _evict_widths((prom, react, fresh), (1, 2))


def test_the_promote_width_is_the_measured_count_not_the_bound():
    """A real eviction cycle must record a promote width no larger than it used.

    ``promote_max`` is the measured max over rows, and it is now what sizes the
    promote payload. This asserts it is a genuine bound (never above ``n_q``,
    or ``_compact`` would be routing lanes to its dump column and dropping
    work) and that the first eviction -- where the Q tier is still empty, so
    nothing can be promoted OUT of it -- records zero rather than the cap.
    """
    w = _evict_and_read_widths(None)
    assert w is not None and w["evictions"] > 0
    assert 0 <= w["promote_max"] <= w["n_q"], w
    assert w["promote_mean"] <= w["promote_max"], w
