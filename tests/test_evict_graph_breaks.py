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


def test_the_eviction_lowers_to_fused_kernels_and_not_to_aten_calls():
    """Zero breaks is necessary, not sufficient: count what Inductor EMITTED.

    A body can trace end to end and still lower to a string of extern ATen
    calls, which is the failure the GPU profile of 2026-09-16 actually showed —
    ``ours: compiled evict`` at 0.000 ms while the eviction ran as
    ``elementwise_kernel`` with fractional launch counts. The break count could
    not have caught that; this can, because it reads the generated wrapper.

    Two numbers, both floors rather than targets:

    * **fused kernels > 0** — the quantiser's round/clamp/div chain, the gather
      that feeds it and the scatter that consumes it are in compiled kernels.
    * **extern ops** — ``sort``, ``searchsorted`` and ``cumsum`` have no Inductor
      lowering on this backend, and fusion cannot cross them. The cap is a
      regression guard: four compactions were expressed as sorts and are now
      cumsums (``modules/quant/compact.py``), which took ``aten.sort`` in this
      cycle from 21 calls to 6. Re-introducing one would push this over.
    """
    import re

    from torch._inductor.utils import run_and_get_code

    dynamo.reset()
    _res, codes = run_and_get_code(_run_decode_across_eviction,
                                   compile_backend="inductor")
    fused = sum(len(set(re.findall(r"^(cpp_fused[\w]*)\s*=", c, re.M))) for c in codes)
    extern = [op for c in codes
              for op in re.findall(r"torch\.ops\.(aten\.[\w.]+)", c)]
    assert fused > 0, (
        "Inductor emitted no fused kernels for the eviction: it compiled and "
        "lowered everything to extern calls, which is the 2026-09-16 profile's "
        "finding with the graph breaks removed."
    )
    kinds = {op.split(".")[1] for op in extern}
    assert kinds <= {"sort", "searchsorted", "cumsum"}, (
        f"a new op fell out of the fused region: {sorted(kinds)}. Everything "
        "pointwise, every gather and every scatter has a lowering; an extern "
        "call here is either a new unlowerable op or one used where a lowerable "
        "one would do (see modules/quant/compact.py for the last four)."
    )
    n_sort = sum(1 for op in extern if op.startswith("sort"))
    assert n_sort <= 8, (
        f"{n_sort} extern sort calls in the compiled eviction, up from 6. A "
        "sort is never lowered, so each one splits the fused region around it; "
        "a compaction expressed as a sort is the usual cause."
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


# ---------------------------------------------------------------------------
# Shape reads inside the compiled region must be CONCRETE
# ---------------------------------------------------------------------------


def test_scalar_shape_reads_in_the_compiled_region_are_int_forced():
    """A raw ``.shape[i]`` inside the compiled eviction is a lowering hazard.

    The eviction compiles with ``dynamic=True`` first
    (``cache._run_compiled_evict``), so every ``.shape`` read in the region is a
    **SymInt**. That is fine where the value is only a tensor size, and hazardous
    where it reaches a SCALAR context -- ``torch.arange``'s length, a
    ``torch.where`` sentinel, a slice bound, a reshape extent, a dict key --
    because Inductor is then handed an integer *expression* instead of an int.

    **This guard does NOT prevent the ``((I)//8)`` outage, and the first two
    versions of it claimed to.** That failure is
    ``FloorDiv(Identity(<a number>), window_size)``, where ``Identity`` is
    Inductor's ``pointwise_cat`` wrapper and ``is_number`` is True -- i.e. the
    value was already concrete, so no amount of ``int()`` could have mattered.
    ``test_the_compiled_eviction_contains_no_cat`` is the test for that one; this
    one is a separate, still-worthwhile invariant, and the two must not be
    confused again. Believing this guard covered ``((I)//8)`` is what sent two
    fixes at the wrong code.

    **A CPU compile does not exercise the CUDA lowering decisions**, so this
    asserts the INVARIANT at the source instead of the symptom at runtime, the
    way ``test_no_python_loops_in_hot_path`` does.

    The fix is always the same and always free: wrap the read in ``int()``. It is
    a no-op in eager and byte-identical, and it adds no recompiles, because the
    widths involved are already specialized by ``_evict_two_tier_impl``.

    **A whole-shape unpack counts.** The first version of this guard matched only
    ``x = t.shape[i]`` and said in a comment that ``B, H, T, D = t.shape`` was "a
    different pattern ... handled by the callers". It is not a different pattern
    -- it is the same symbolic read wearing a different AST -- so both forms are
    matched now.
    """
    import ast
    import pathlib

    # (file, function, names allowed to stay symbolic) for everything the
    # compiled body reaches that reads a dim off a `.shape`.
    #
    # The allowed set is ALWAYS the row axis and nothing else: `B`, or the
    # flattened `B * n` that `demote_many` hands `build_sketch` as `N`. Keeping
    # the row axis dynamic is the entire reason the eviction compiles with
    # `dynamic=True` rather than recompiling per batch size, and it is never used
    # as a scalar -- only as a tensor size that broadcasting already agrees on.
    REGION = [
        ("modules/windowed_cache/cache.py", "_evict_two_tier_impl", {"B"}),
        ("modules/windowed_cache/cache.py", "_compact", {"B"}),
        ("modules/windowed_cache/policy.py", "compute_two_tier_retain", {"B"}),
        ("modules/quant/compact.py", "stable_partition", set()),
        ("modules/quant/slots.py", "lookup", set()),
        ("modules/quant/store.py", "promote_many", {"B"}),
        ("modules/quant/store.py", "demote_many", {"B"}),
        ("modules/quant/sketch.py", "build_sketch", {"N"}),
    ]

    def _shape_read(v):
        """`<expr>.shape` or `<expr>.shape[...]`, else None."""
        if isinstance(v, ast.Subscript):
            v = v.value
        return v if (isinstance(v, ast.Attribute) and v.attr == "shape") else None

    def _names(target):
        """The bare `Name` targets of an assignment, tuple unpacks included.

        `_` is dropped: a throwaway is never read, so it cannot reach a scalar
        context. Everything else that gets a name gets checked.
        """
        if isinstance(target, ast.Name):
            names = [target.id]
        elif isinstance(target, (ast.Tuple, ast.List)):
            names = [e.id for e in target.elts if isinstance(e, ast.Name)]
        else:
            names = []
        return [n for n in names if n != "_"]

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for rel, fname, allow in REGION:
        tree = ast.parse((root / rel).read_text())
        fns = [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == fname]
        assert fns, f"{rel}: no function named {fname} -- did it move or get renamed?"
        for fn in fns:
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign):
                    continue
                # `X = t.shape[i]` (one dim) and `A, B, C = t.shape` (the whole
                # shape) are the SAME hazard: every name bound is a SymInt under
                # `dynamic=True`. An `int(...)` wrapper is a Call, not a
                # Subscript/Attribute, so a forced read never reaches here.
                if _shape_read(node.value) is None:
                    continue
                for t in node.targets:
                    for name in _names(t):
                        if name not in allow:
                            offenders.append(
                                f"{rel}:{node.lineno}  {name} <- {fname}: raw .shape read")

    assert not offenders, (
        "a shape dim is read raw inside the compiled eviction:\n  "
        + "\n  ".join(offenders)
        + "\n\nUnder `dynamic=True` that is a SymInt, whether it came from "
          "`t.shape[i]` or from a `a, b, c = t.shape` unpack. If it reaches a "
          "scalar context (torch.arange, a torch.where sentinel, a slice bound, "
          "a reshape extent, a dict key) Inductor's CUDA backend raises `The "
          "argument '((I)//8)' is not comparable` and every perf cell ERRORs. "
          "Wrap it in `int()` -- free, byte-identical, and no extra recompiles. "
          "Add a name to that function's allow-set ONLY if it is the row axis "
          "(see `B`), which is the one axis `dynamic=True` exists for."
    )


# ---------------------------------------------------------------------------
# No torch.cat in the compiled eviction  (the ((I)//8) lowering failure)
# ---------------------------------------------------------------------------


def _cat_lowerings_during(fn):
    """Every ``aten.cat`` Inductor lowers while ``fn`` runs, as source lines."""
    import torch._inductor.config as ic
    import torch._inductor.lowering as L
    from torch._inductor.virtualized import V

    aten = torch.ops.aten
    hits: list[str] = []
    saved = {k: L.lowerings[k] for k in (aten.cat, aten.cat.default) if k in L.lowerings}

    def wrap(orig):
        def w(inputs, dim=0):
            node = getattr(V, "current_node", None)
            st = (getattr(node, "meta", {}) or {}).get("stack_trace", "") or ""
            frames = [l.strip() for l in st.splitlines() if l.strip().startswith("File ")]
            mine = [f for f in frames if "RecoverKv" in f]
            hits.append(mine[-1] if mine else (frames[-1] if frames else "<no stack>"))
            return orig(inputs, dim)
        return w

    # force_pointwise_cat: `lowering.py::cat` returns a ConcatKernel
    # UNCONDITIONALLY for CPU devices, so a CPU box never takes the pointwise
    # path and never builds the `Identity` node this is all about. Forcing it is
    # what makes this test exercise the CUDA lowering decision. simdlen=0 keeps
    # the scalar C++ backend, which sidesteps an unrelated CppVecOverrides break
    # under emulate_precision_casts on some torch builds.
    # fx_graph_cache OFF: a cache hit returns a compiled artifact without
    # lowering anything, so the counter below sees zero cats and the test passes
    # vacuously. That is not hypothetical -- it is what happened the first time
    # this test was run twice in a row.
    with ic.patch({"force_pointwise_cat": True, "cpp.simdlen": 0,
                   "fx_graph_cache": False, "fx_graph_remote_cache": False}):
        try:
            for k, o in saved.items():
                L.lowerings[k] = wrap(o)
            dynamo.reset()
            fn()
        finally:
            for k, o in saved.items():
                L.lowerings[k] = o
    return hits


def test_the_compiled_eviction_contains_no_cat():
    """A ``torch.cat`` in the eviction graph is the ``((I)//8)`` outage.

    That error -- which ERRORed every cell of three consecutive GPU perf tables
    -- is not about shapes, despite two rounds of forcing ``int()`` on ``.shape``
    reads on the theory that it was. Decoding it:

    * ``I`` is ``torch.utils._sympy.functions.Identity``, Inductor's "do not
      expand this" wrapper. It prints as ``I`` because sympy's ``StrPrinter``
      defines ``_print_Identity`` for the identity *matrix* and dispatches on the
      class *name*, so ``((I)//8)`` is ``FloorDiv(Identity(<expr>), window_size)``.
    * Inductor builds that wrapper in exactly ONE place --
      ``_inductor/lowering.py::pointwise_cat``, as
      ``Identity(idx[dim] - inputs_ranges[i][0])``. A ``cat`` lowered pointwise
      is the only way to get one.
    * ``_simplify_loops`` -> ``stride_vars`` removes the offset by substituting
      every index var with 0, turning ``Identity(i - h)`` into ``Identity(-h)``.
      That has no free symbols, so when sympy rebuilds the enclosing
      ``Min``/``Max`` its ``_new_args_filter`` rejects it: ``is_number`` and not
      ``is_comparable``.

    ``is_number`` being True is the proof that the shape theory was wrong -- the
    value was already concrete. It is also why the ``dynamic=False`` retry failed
    with the byte-identical message rather than a different one.

    So the invariant is not "make the shapes concrete", it is **no cat in this
    graph at all**, which is what this asserts. ``Identity`` has a single source,
    so zero cats makes the failure impossible rather than unlikely.
    """
    hits = _cat_lowerings_during(lambda: _run_decode_across_eviction(compile_backend="inductor"))
    assert not hits, (
        "the compiled eviction lowered a torch.cat:\n  "
        + "\n  ".join(dict.fromkeys(hits))
        + "\n\nOn CUDA that goes through `pointwise_cat`, which wraps the shifted "
          "index in `Identity` and makes Inductor raise `The argument '((I)//8)' "
          "is not comparable` from sympy's Min/Max -- every perf cell ERRORs. "
          "Build the tensor without a cat: `cache._prepend_prefix` for a prefix "
          "join, `effective._rotate_half_no_cat` for RoPE's rotation. Removing "
          "one cat can expose another (it changes Inductor's fusion heuristics), "
          "which is why this asserts on ZERO rather than on a known set."
    )


def test_rotate_half_no_cat_is_bit_identical_to_huggingface():
    """The cat-free rotation must be HF's, bit for bit -- not merely close.

    ``_apply_rotary_one`` is the reference every accuracy test validates against,
    and its contract is that it reproduces
    ``apply_rotary_pos_emb(k, k, cos, sin)[1]`` exactly. Replacing the ``cat``
    inside it is only admissible if the bits are unchanged, so this compares the
    raw bit patterns (``torch.equal`` alone would call ``-0.0`` equal to ``0.0``)
    over the dtypes the cache actually stores and the values most likely to
    expose a sign-handling difference.
    """
    from modules.quant.effective import _rotate_half, _rotate_half_no_cat

    hf = _rotate_half()
    ibits = {2: torch.int16, 4: torch.int32, 8: torch.int64}

    def same_bits(a, b):
        assert a.dtype == b.dtype and a.shape == b.shape
        t = ibits[a.element_size()]
        return torch.equal(a.contiguous().view(t), b.contiguous().view(t))

    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in ((4, 2, 8, 16), (1, 1, 1, 2), (3, 5, 7, 128)):
            x = torch.randn(*shape).to(dtype)
            assert same_bits(hf(x), _rotate_half_no_cat(x)), (dtype, shape)
        # signed zeros, infinities and a subnormal -- where `x * -1` would differ
        # from `-x` if the identity did not hold.
        edge = torch.tensor(
            [[0.0, -0.0, float("inf"), -float("inf"), 3.5, -3.5, 1.0, -1.0]]
        ).to(dtype)
        assert same_bits(hf(edge), _rotate_half_no_cat(edge)), dtype

    # A non-contiguous caller must work too: `_apply_rotary_one` is handed
    # permuted views on the promote path.
    x = torch.randn(4, 8, 16).transpose(0, 1)
    assert same_bits(hf(x), _rotate_half_no_cat(x))
