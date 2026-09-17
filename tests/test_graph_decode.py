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
    StepSignature, graph_mode,
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
    return DecodeGraphRunner(_FakeCache(), EpochSchedule(window_size=ws),
                             mode=mode, warmup_epochs=warmup)


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
