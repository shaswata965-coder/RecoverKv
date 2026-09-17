"""The CUDA-graph runner's policy layer, which is the only part a CPU box can run.

The capture/replay mechanism needs CUDA and ships unvalidated by construction. It
is the default there anyway, which is what makes this file load-bearing rather
than nice to have: ``STICKYKV_DECODE_GRAPH=off`` is the escape hatch, and these
are the checks that stand between a wrong replay and a wrong answer. What *is*
checkable here
is everything that decides **whether a replay is legal**, and that is where the
danger lives: a graph replayed against the wrong geometry does not crash, it
indexes the wrong windows and emits scores that still look like probabilities.

Two things are pinned:

* :class:`EpochSchedule` agrees with :meth:`EvictionPolicy.should_evict` on every
  step of every cadence. It is a restatement, and a restatement that drifts would
  replay a steady graph over an eviction.
* :class:`StepSignature` refuses every field it is supposed to refuse — including
  ``buffer_ptrs``, which is the one that catches a reallocated ``_key_buf`` at
  identical shapes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modules.windowed_cache.graph_decode import (  # noqa: E402
    DecodeGraphRunner, EpochSchedule, GraphSignatureMismatch, HostDelta,
    StepSignature, graph_mode, replay_is_reachable,
)
from modules.windowed_cache.policy import EvictionPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# The schedule must be the policy, not an approximation of it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ws", [1, 2, 3, 4, 8, 12, 16])
@pytest.mark.parametrize("first", [0, 1, 3, 8, 13])
def test_the_schedule_agrees_with_the_real_policy_on_every_step(ws, first):
    """Restating ``should_evict`` is the risk; this is the check that it holds.

    The cases in the policy's own docstring are the hard ones — ``ws=12, N=8``
    gives a short ``8 -> 12`` gap, and ``ws=3, N=8`` suppresses the natural
    boundaries below 8 and resumes at 9.
    """
    sched = EpochSchedule(window_size=ws, first_eviction_step=first)
    pol = EvictionPolicy.__new__(EvictionPolicy)
    pol.window_size, pol.first_eviction_step = ws, first
    for step in range(0, 200):
        assert sched.evicts(step) == pol.should_evict(step), (
            f"step {step}, ws={ws}, first={first}")


@pytest.mark.parametrize("ws,first", [(8, 0), (8, 8), (4, 0), (12, 8)])
def test_an_eviction_step_is_never_given_a_slot(ws, first):
    """A slot on an eviction step is the one bug here that produces output."""
    sched = EpochSchedule(window_size=ws, first_eviction_step=first)
    for step in range(200):
        if sched.evicts(step):
            assert sched.slot(step) is None


def test_nothing_before_the_first_eviction_is_capturable():
    """Until the store has been compacted its geometry is the prompt's.

    ``CacheState`` releases the prefill-sized buffer at that first compaction, so
    a slot captured before it holds a pointer into a buffer about to be freed.
    """
    sched = EpochSchedule(window_size=8, first_eviction_step=16)
    assert all(sched.slot(s) is None for s in range(17))
    assert sched.slot(17) == 1


def test_slots_repeat_with_the_epoch():
    """Slot s of epoch n must be slot s of epoch n+1, or the ws-graph scheme
    has no basis: each slot is captured once and replayed for the rest of the
    generation."""
    sched = EpochSchedule(window_size=8, first_eviction_step=0)
    for s in range(1, 8):
        assert sched.slot(s) == sched.slot(s + 8) == sched.slot(s + 800) == s
    assert sched.slots == 8


@pytest.mark.parametrize("bad", [dict(window_size=0), dict(window_size=-2),
                                 dict(first_eviction_step=-1)])
def test_a_nonsense_schedule_raises(bad):
    kw = dict(window_size=8, first_eviction_step=0); kw.update(bad)
    with pytest.raises(ValueError):
        EpochSchedule(**kw)


# ---------------------------------------------------------------------------
# The signature is the replay contract
# ---------------------------------------------------------------------------


def _sig(**over):
    base = dict(slot=1, batch=32, seq_lengths=(870,), store_versions=(4,),
                n_active=(64,), buffer_ptrs=(140_000,), dtype="torch.float16",
                device="cuda:0")
    base.update(over)
    return StepSignature(**base)


def test_an_identical_signature_matches():
    assert _sig().mismatch(_sig()) is None


@pytest.mark.parametrize("field,value", [
    ("slot", 2),
    ("batch", 16),
    ("seq_lengths", (871,)),
    ("store_versions", (5,)),
    ("n_active", (63,)),
    ("buffer_ptrs", (999,)),
    ("dtype", "torch.bfloat16"),
    ("device", "cuda:1"),
])
def test_every_field_is_load_bearing(field, value):
    """None of these would raise on a GPU. All of them change what the captured
    kernels address."""
    why = _sig().mismatch(_sig(**{field: value}))
    assert why is not None and why.startswith(field)


def test_a_reallocated_buffer_is_caught_at_identical_shapes():
    """``replace`` reallocates whenever the target size moves, and a graph holds
    ADDRESSES. Same length, new buffer, must not replay."""
    assert _sig().mismatch(_sig(buffer_ptrs=(140_064,))) is not None


# ---------------------------------------------------------------------------
# The host delta — the part a replay has to do by hand
# ---------------------------------------------------------------------------


def test_the_delta_is_the_difference_between_the_two_signatures():
    pre, post = _sig(seq_lengths=(870,)), _sig(seq_lengths=(871,))
    d = HostDelta.between(pre, post)
    assert d.seq_len_delta == (1,) and d.version_delta == (0,)


def test_a_store_that_ticked_its_version_voids_the_capture():
    """The Q tier is frozen between evictions (design §10). If a steady step
    moved a version, something mutated the tier mid-epoch and no slot in that
    epoch describes the store any more."""
    d = HostDelta(seq_len_delta=(1,), version_delta=(1,))
    with pytest.raises(GraphSignatureMismatch, match="frozen"):
        d.apply(_FakeCache())


def test_the_delta_advances_the_state_through_its_own_reslice():
    """The views after a replay must be what ``append`` would have produced."""
    from modules.windowed_cache.state import CacheState
    st = CacheState(capacity=16)
    st.replace(torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3),
               torch.arange(4).unsqueeze(0))
    cache = _FakeCache(states=[st])
    assert st.key_states.shape[2] == 4
    HostDelta(seq_len_delta=(1,), version_delta=(0,)).apply(cache)
    assert st.key_states.shape[2] == 5
    assert st.value_states.shape[2] == 5 and st.position_ids.shape[1] == 5


class _FakeCache:
    def __init__(self, states=None, stores=None):
        self._joint = None
        self._joint_store = None
        self._states = states or []
        self._stores = stores or []


# ---------------------------------------------------------------------------
# The runner's control flow, with the mechanism stubbed out
# ---------------------------------------------------------------------------


def _runner(mode="on", ws=8, warmup=1):
    # require_replayable=False: these tests are about the capture/invalidate
    # POLICY, which is pure Python and testable here. The guard they bypass is
    # about whether capturing is WORTH it on a GPU, which is a different
    # question and has its own tests below.
    return DecodeGraphRunner(_FakeCache(), EpochSchedule(window_size=ws),
                             mode=mode, warmup_epochs=warmup,
                             require_replayable=False)


def test_off_never_captures_anything():
    r = _runner(mode="off")
    for step in range(40):
        r.step(step, lambda: "eager")
    assert r.stats["captured"] == 0 and r.stats["replayed"] == 0


def test_an_eviction_step_always_runs_eager_and_invalidates_first():
    """Invalidation happens BEFORE the eviction runs, so a raise inside it
    cannot leave a live graph pointing at a half-written tier."""
    r = _runner()
    r._graphs[3] = object(); r._sigs[3] = _sig()
    r._deltas[3] = None; r._posts[3] = _sig()
    r.step(8, lambda: "eager")                # ws=8 -> step 8 evicts
    assert r._graphs == {} and r.stats["invalidated"] == 1


def test_warmup_epochs_are_run_eagerly():
    """A capture before the first eviction holds a pointer into the
    prefill-sized buffer that compaction is about to release."""
    r = _runner(warmup=2)
    for step in range(1, 8):
        r.step(step, lambda: "eager")
    assert r.stats["captured"] == 0
    assert r.stats["eager"] == 7


def test_a_moved_signature_falls_back_to_eager_rather_than_replaying():
    """A legal geometry change is not an error — it just means the capture is
    stale. Errors are reserved for a replay already judged legal."""
    r = _runner()
    r._evictions_seen = 5
    r._sigs[1] = _sig(batch=1)                # will not match the fake cache
    r._graphs[1] = object()
    out = r.step(1, lambda: "eager")
    assert out == "eager" and r._graphs == {} and r.stats["replayed"] == 0


def test_an_explicit_mode_always_wins_over_the_device(monkeypatch):
    """`verify` is an alias: the post-replay check it used to select is now
    unconditional, so there is nothing left for a separate mode to turn on."""
    for raw, want in [("0", "off"), ("off", "off"), ("no", "off"),
                      ("1", "on"), ("on", "on"), ("verify", "on"),
                      ("check", "on")]:
        monkeypatch.setenv("STICKYKV_DECODE_GRAPH", raw)
        assert graph_mode() == want


def test_the_default_is_off_on_every_device(monkeypatch):
    """One method, one path.

    The runner is wired into perf_runner's decode loop, NOT into
    `model.generate`, which is what LongBench and GSM8K call. On by default
    would mean the throughput table came from a path no accuracy number
    describes. It is an ablation you ask for, never the shipped default.
    """
    monkeypatch.delenv("STICKYKV_DECODE_GRAPH", raising=False)
    for available in (False, True):
        monkeypatch.setattr(torch.cuda, "is_available", lambda a=available: a)
        assert graph_mode() == "off"


def test_a_typo_raises_rather_than_silently_disabling_the_graph(monkeypatch):
    monkeypatch.setenv("STICKYKV_DECODE_GRAPH", "yes-please")
    with pytest.raises(ValueError, match="on / off / verify"):
        graph_mode()


# ---------------------------------------------------------------------------
# Is a capture ever USED?  (the reason the graph is refused, not just off)
# ---------------------------------------------------------------------------


def _census(ws=8, first=0, n_steps=1024, warmup=1, monkeypatch=None):
    """Count captures vs replays over a generation, with the CUDA half stubbed.

    The stubs do exactly what the real ones do to the runner's bookkeeping --
    capture records the signature, replay reads it back -- so what is being
    counted is the POLICY, which is the thing in question. `signature_of` is
    replaced with one whose only moving part is `store_versions`, ticking once
    per eviction, because that is what the real store does.
    """
    import modules.windowed_cache.graph_decode as gd

    sched = EpochSchedule(window_size=ws, first_eviction_step=first)
    r = DecodeGraphRunner(_FakeCache(), sched, mode="on", warmup_epochs=warmup,
                          require_replayable=False)

    def fake_capture(slot, pre, runfn):
        r._sigs[slot] = pre
        r._graphs[slot] = object()
        r._posts[slot] = pre
        r._static_out[slot] = "out"
        r.stats["captured"] += 1
        return runfn()

    def fake_replay(slot):
        r.stats["replayed"] += 1
        return r._static_out[slot]

    r._capture = fake_capture
    r._replay = fake_replay

    seen = {"evictions": 0}

    def sig_of(cache, slot):
        return _sig(slot=slot, seq_lengths=(1024 + slot,),
                    store_versions=(seen["evictions"],))

    monkeypatch.setattr(gd, "signature_of", sig_of)
    for step in range(n_steps):
        if sched.evicts(step):
            seen["evictions"] += 1
        r.step(step, lambda: "eager")
    return r.stats


@pytest.mark.parametrize("ws,first", [(8, 0), (16, 0), (8, 32)])
def test_the_capture_schedule_never_replays_anything(ws, first, monkeypatch):
    """The headline fact: captures are never used, so capturing is pure cost.

    A slot is destroyed at the next eviction and each slot value occurs exactly
    once per epoch, so nothing captured survives to a step that could replay it.
    This is what `replay_is_reachable` reports and what the construction guard
    refuses on; if someone makes the graph actually replay, THIS test is the one
    that should start failing first.
    """
    stats = _census(ws=ws, first=first, monkeypatch=monkeypatch)
    assert stats["captured"] > 0, "the census did not exercise the capture path"
    assert stats["replayed"] == 0, (
        f"replay became reachable ({stats}); replay_is_reachable() and the "
        "construction guard are now wrong and must be revisited together"
    )


def test_turning_the_graph_on_is_refused_with_the_reason():
    """Refused at construction, not after minutes of a sweep.

    The previous behaviour captured every steady step, replayed none, and died
    in the CUDA allocator partway through -- which is how `table_v5_graph` came
    back with six ERROR rows and no usable number.
    """
    with pytest.raises(RuntimeError) as ei:
        DecodeGraphRunner(_FakeCache(), EpochSchedule(window_size=8), mode="on")
    msg = str(ei.value)
    assert "never replayed" in msg or "no capture is ever replayed" in msg
    assert "STICKYKV_DECODE_GRAPH" in msg


def test_mode_off_is_never_refused():
    """The default path must not be reachable by this guard at all."""
    r = DecodeGraphRunner(_FakeCache(), EpochSchedule(window_size=8), mode="off")
    for step in range(40):
        r.step(step, lambda: "eager")
    assert r.stats["captured"] == 0 and r.stats["replayed"] == 0


def test_invalidate_drops_pool_owned_tensors_before_the_pool():
    """The order in `invalidate` is the `use_count > 0` allocator crash.

    A captured graph's outputs are allocated inside the graph's memory pool.
    Destroying the CUDAGraph objects releases the pool, and PyTorch asserts the
    pool has no live blocks when it does. `_static_out` holds exactly those
    blocks, so it has to go first -- and the pool handle has to be dropped after,
    because once its last graph is gone the handle is dead and capturing into it
    again is the other half of the same crash.
    """
    r = _runner()
    order = []

    class _Tracking(dict):
        def __init__(self, label):
            super().__init__()
            self._label = label

        def clear(self):
            order.append(self._label)
            super().clear()

    r._static_out = _Tracking("static_out")
    r._graphs = _Tracking("graphs")
    r._static_out[1] = "pool-owned tensor"
    r._graphs[1] = object()
    r._pool = "pool-handle"

    r.invalidate("test")

    assert order.index("static_out") < order.index("graphs"), (
        "pool-owned output tensors were still live when the graphs (and so the "
        "pool) were destroyed")
    assert r._pool is None, (
        "the pool handle survived its last graph; the next capture would reuse "
        "a torn-down pool")


def test_no_caller_disables_the_replayability_guard():
    """`require_replayable=False` is a test affordance, not a shipping flag."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in list(root.glob("modules/**/*.py")) + list(root.glob("scripts/**/*.py")):
        if "require_replayable" in path.read_text(encoding="utf-8"):
            if path.name != "graph_decode.py":
                offenders.append(str(path.relative_to(root)))
    assert not offenders, (
        f"{offenders} set require_replayable outside tests/; that re-enables a "
        "capture path that replays nothing")
