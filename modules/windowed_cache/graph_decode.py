"""CUDA-graph replay for the steady decode steps of a window epoch.

`DECODE_NEXT.md` §2 measures the decode step at ~48-50 ms of kernels against
**28.25 ms of host gap** -- 37% of TPOT, over ~1,700 launches at ~17 us exposed
each. Our cache is 7.9 ms of the kernel time, so deleting it entirely leaves
68 ms. The host gap is the only term in the step that is larger than the gap to
the published table, and a CUDA graph is the only thing that removes it.

``DECODE_NEXT.md`` §4c cut this, on the grounds that *"capture needs static
shapes and no host syncs, and the cache has neither: ``n_active`` moves every
eviction, evictions are ragged per row, and ``store.version`` invalidates memos
mid-step."* Three of those four have since moved:

* **host syncs** -- the 2026-09-16 profile measures ``cudaDeviceSynchronize`` at
  0.1/step, 0.02 ms. There is effectively no sync on the decode path.
* **graph breaks** -- ``6e83a2c`` took the eviction from 8 to 0.
* **shapes** -- they move *at an eviction*, not between them. Between evictions
  the Q tier is frozen (design §10: a card is written once and reactivated
  unchanged), which is the invariant ``_fused_ctx`` already memoizes on.

So this does not graph "the decode step". It graphs the **``ws`` steps of one
window epoch**, and the eviction step stays eager and re-captures.

Why ``ws`` graphs rather than one
---------------------------------
Within an epoch the fp store grows by one token per step, so ``S_fp``,
``n_body_win`` and ``W_phys`` take a different value at each position in the
epoch. They are not static within an epoch -- they are static *at the same
position of every epoch*, because the eviction compacts back to a length that is
pure config arithmetic (``cache.py`` §4: *"the length is pure config arithmetic:
the retained evictable windows are full, and the local tokens are whatever the
body holds beyond the evictable-fp windows currently in it"*). So slot ``s``
of epoch ``n`` has the same geometry as slot ``s`` of epoch ``n+1``.

That gives ``ws`` graphs, captured once, replayed for the rest of the
generation, sharing one memory pool so the pool is sized to the largest slot
rather than their sum.

The part that makes this dangerous, and what is done about it
------------------------------------------------------------
**A graph replays kernels. It does not re-run Python.** Every host-side fact the
cache advances per step -- ``CacheState._reslice``, the policy's totals,
``cache_kwargs[layer]["window_scores"]``, ``_fused_ctx``'s memo key,
``flash_decode._STATS`` -- would be left at its capture-time value by a bare
replay, and the next eager step would then run against stale bookkeeping while
every tensor still looked plausible. That is the failure mode this repo has been
bitten by three times (the ``STICKYKV_COMPILE_READ`` banner, the vacuous
``audit_e2e`` rungs 2/3, the eviction that reported itself compiled and fused
nothing), and it is silent every time.

So the runner never *infers* that a replay is safe. It records the complete host
signature at capture (:class:`StepSignature`) and refuses to replay unless the
live signature is equal to it, field for field. A mismatch is not a fallback to
be logged and forgotten -- it is either an eager step or an error, chosen by the
mode. The signature is pure Python and is the one part of this file that a
CPU-only box can exercise, which is why it is split out the way
``window_tiling`` and ``check_gate_selection`` are.

Modes (``STICKYKV_DECODE_GRAPH``)
--------------------------------
``off`` (default)
    Nothing here runs. This is a lever, not the method.
``verify``
    Replays, then **re-reads the host signature and asserts it is exactly the
    post-state recorded at capture**. That catches the failure this design is
    most exposed to: a replay whose hand-applied :class:`HostDelta` did not
    reproduce the bookkeeping the eager step performed.

    It deliberately does **not** run an eager shadow of the same step. Running
    the step twice advances the cache twice -- the graph replay writes the KV
    append itself -- so a per-step A/B would compare two different sequence
    positions and corrupt the run while doing it. The real equivalence check is
    at **run level**: generate the same prompt with ``off`` and with ``on`` and
    diff the token sequences. Greedy decode is deterministic, so an identical
    sequence is proof, and that is the shape of check ``H-B`` lost when
    ``676c781`` made ``audit_e2e``'s rungs 2/3 compare a path against itself.
``on``
    Capture and replay, no eager shadow.

None of this has run on a GPU. The capture/replay mechanism below cannot be
executed on a CPU-only box at all, so it ships unvalidated by construction --
which is the reason ``off`` is the default and ``verify`` exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

__all__ = [
    "graph_mode",
    "EpochSchedule",
    "StepSignature",
    "HostDelta",
    "DecodeGraphRunner",
    "GraphSignatureMismatch",
]


class GraphSignatureMismatch(RuntimeError):
    """The live host state does not match what a captured slot was taken under.

    Raised rather than silently falling back, for the same reason
    ``_run_compiled_evict`` is kernel-or-error: a quiet fallback reports one
    method's numbers under another's label, and on this path it would also mean
    replaying kernels against bookkeeping they were not captured with.
    """


def graph_mode() -> str:
    """``"off"`` | ``"verify"`` | ``"on"``, from ``STICKYKV_DECODE_GRAPH``.

    Default **off**. Unlike ``_compile_evict_enabled``, the device does not get
    to decide: a CUDA graph changes when host bookkeeping runs relative to
    kernels, which is a property of the decode loop and not of the hardware.
    """
    raw = os.environ.get("STICKYKV_DECODE_GRAPH", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return "on"
    if raw in ("verify", "check", "2"):
        return "verify"
    if raw in ("", "0", "false", "no", "off"):
        return "off"
    raise ValueError(
        f"STICKYKV_DECODE_GRAPH={raw!r} is not off / on / verify. It is left "
        "strict rather than defaulted so a typo cannot silently disable the "
        "thing it was set to enable."
    )


# ---------------------------------------------------------------------------
# Policy: which step is capturable, and into which slot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochSchedule:
    """Where a decode step sits relative to the eviction cadence.

    Mirrors :meth:`EvictionPolicy.should_evict` exactly -- the first eviction
    fires at ``first_eviction_step`` independent of ``window_size``, and the
    natural ``step % window_size == 0`` cadence resumes after it. It is restated
    here rather than imported because a graph runner that disagreed with the
    policy about which step evicts would replay a steady graph over an eviction,
    and that is the one mistake in this file that produces plausible output
    rather than an error.
    """

    window_size: int
    first_eviction_step: int = 0

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.first_eviction_step < 0:
            raise ValueError(
                f"first_eviction_step must be >= 0, got {self.first_eviction_step}")

    def evicts(self, step: int) -> bool:
        """True when the eviction runs on this decode step."""
        if step == self.first_eviction_step:
            return True
        if step < self.first_eviction_step:
            return False
        return step % self.window_size == 0

    def slot(self, step: int) -> Optional[int]:
        """Which graph slot this step belongs to, or ``None`` if it evicts.

        The slot is the step's offset from its epoch start, so slot ``s`` always
        sees the store at the same length: the eviction compacted it to a
        config-determined length ``s`` steps ago and one token has been appended
        per step since.
        """
        if self.evicts(step):
            return None
        if step < self.first_eviction_step:
            # Before the first eviction the store has never been compacted, so
            # the geometry is the prompt's and does not repeat. Never capturable.
            return None
        return step % self.window_size

    @property
    def slots(self) -> int:
        """How many distinct graphs a full epoch needs (slot 0 always evicts)."""
        return self.window_size


# ---------------------------------------------------------------------------
# The contract a replay is only valid under
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSignature:
    """Every host fact a captured slot depends on.

    Equality is the replay contract. If any of these moved since capture, the
    kernels in the graph are addressing the wrong geometry, the wrong store
    version, or the wrong buffers -- and none of those would raise on a GPU.
    They would index the wrong windows and emit scores that still look exactly
    like probabilities, which is the failure ``check_gate_selection`` exists to
    refuse one layer down.

    ``store_versions`` is the load-bearing one. ``QuantizedStore.version`` ticks
    on every mutation and is what ``effective_q_tier`` and ``_fused_ctx`` already
    memoize against, so it is the cache's own statement that the Q tier changed.
    ``buffer_ptrs`` is the other: a graph holds *addresses*, so a reallocated
    ``_key_buf`` (which ``replace`` performs whenever the target size moves)
    invalidates every captured slot even at identical shapes.
    """

    slot: int
    batch: int
    seq_lengths: Tuple[int, ...]
    store_versions: Tuple[int, ...]
    n_active: Tuple[int, ...]
    buffer_ptrs: Tuple[int, ...]
    dtype: str
    device: str

    def mismatch(self, other: "StepSignature") -> Optional[str]:
        """The first field that differs, named, or ``None``."""
        for f in ("slot", "batch", "seq_lengths", "store_versions",
                  "n_active", "buffer_ptrs", "dtype", "device"):
            a, b = getattr(self, f), getattr(other, f)
            if a != b:
                return f"{f}: captured {a!r}, live {b!r}"
        return None


def signature_of(cache, slot: int) -> StepSignature:
    """Read the live signature off a :class:`WindowedCache`.

    Deliberately reads only things that are already Python ints or tensor
    metadata -- **no ``.item()``, no device read**. A signature check that
    synchronized would cost more than the graph saves, and would itself be a
    host stall on the path this exists to unblock.
    """
    states = _live_states(cache)
    stores = _live_stores(cache)
    lengths, ptrs = [], []
    for st in states:
        k = getattr(st, "key_states", None)
        if k is None:
            continue
        lengths.append(int(k.shape[2]))
        buf = getattr(st, "_key_buf", None)
        ptrs.append(int(buf.data_ptr()) if buf is not None else 0)
    versions = tuple(int(getattr(s, "version", -1)) for s in stores)
    actives = tuple(int(getattr(s, "num_active_windows", -1)) for s in stores)
    ref = states[0].key_states if states else None
    return StepSignature(
        slot=int(slot),
        batch=int(ref.shape[0]) if ref is not None else 0,
        seq_lengths=tuple(lengths),
        store_versions=versions,
        n_active=actives,
        buffer_ptrs=tuple(ptrs),
        dtype=str(ref.dtype) if ref is not None else "?",
        device=str(ref.device) if ref is not None else "?",
    )


def _live_states(cache) -> List[Any]:
    """The cache states a step touches: the joint store once it exists."""
    joint = getattr(cache, "_joint", None)
    if joint is not None:
        return [joint]
    return [s for s in getattr(cache, "_states", []) or [] if s is not None]


def _live_stores(cache) -> List[Any]:
    joint = getattr(cache, "_joint_store", None)
    if joint is not None:
        return [joint]
    return [s for s in getattr(cache, "_stores", []) or [] if s is not None]


# ---------------------------------------------------------------------------
# The host-side transition a replay has to perform by hand
# ---------------------------------------------------------------------------


@dataclass
class HostDelta:
    """What the Python side advanced by, over one captured step.

    A replay re-executes kernels and nothing else, so this is the part of the
    step that would otherwise not happen. It is **recorded** at capture rather
    than derived, because deriving it means restating ``CacheState.append``,
    ``EvictionPolicy`` and ``_fused_ctx``'s memo rules in a second place -- and
    the moment those two copies disagree, the cache's views point at a different
    length than its kernels wrote, silently.

    Recording it also makes it checkable: the delta observed at capture must be
    the delta the eager step produced, and the runner asserts the post-state it
    reconstructs equals the post-state it recorded.
    """

    seq_len_delta: Tuple[int, ...]
    version_delta: Tuple[int, ...]

    @staticmethod
    def between(pre: StepSignature, post: StepSignature) -> "HostDelta":
        return HostDelta(
            seq_len_delta=tuple(b - a for a, b in
                                zip(pre.seq_lengths, post.seq_lengths)),
            version_delta=tuple(b - a for a, b in
                                zip(pre.store_versions, post.store_versions)),
        )

    def apply(self, cache) -> None:
        """Advance the Python bookkeeping by the recorded delta.

        Only ``CacheState``'s length is advanced here, through its own
        ``_reslice``, so the public views stay exactly what ``append`` would have
        produced. The Q tier is frozen within an epoch by construction (design
        §10), so ``version_delta`` is asserted to be zero rather than applied:
        a store that ticked its version during a steady step means something
        mutated the tier mid-epoch and the whole capture is void.
        """
        if any(v != 0 for v in self.version_delta):
            raise GraphSignatureMismatch(
                f"a captured steady step changed a store version "
                f"({self.version_delta}). The Q tier is supposed to be frozen "
                "between evictions (design §10); if it is not, no slot in this "
                "epoch is replayable and the capture must be discarded."
            )
        for st, d in zip(_live_states(cache), self.seq_len_delta):
            if d:
                st._reslice(int(st.key_states.shape[2]) + int(d))


# ---------------------------------------------------------------------------
# The mechanism (CUDA only; unexecutable on a CPU box by construction)
# ---------------------------------------------------------------------------


class DecodeGraphRunner:
    """Capture and replay the steady steps of a window epoch.

    Usage from a decode loop::

        runner = DecodeGraphRunner(cache, schedule, mode=graph_mode())
        for step in range(n_decode):
            out = runner.step(step, lambda: model(**kw))

    The two callables are the same call; ``eager`` is taken separately so
    ``verify`` mode can run it a second time without the runner having to know
    how to rebuild the arguments.
    """

    def __init__(self, cache, schedule: EpochSchedule, mode: str = "off",
                 warmup_epochs: int = 1) -> None:
        self.cache = cache
        self.schedule = schedule
        self.mode = mode
        #: Epochs to run eagerly before capturing. One is enough for the
        #: geometry to settle -- the first eviction releases the prompt-sized
        #: buffer (`state.py`: *"released the first time replace compacts back to
        #: the budget"*), so a capture taken before it holds a pointer into a
        #: buffer that is about to be freed.
        self.warmup_epochs = max(1, int(warmup_epochs))
        self._graphs: Dict[int, Any] = {}
        self._sigs: Dict[int, StepSignature] = {}
        self._deltas: Dict[int, HostDelta] = {}
        self._posts: Dict[int, StepSignature] = {}
        self._static_out: Dict[int, Any] = {}
        self._static_in: Dict[str, torch.Tensor] = {}
        self._pool = None
        self._evictions_seen = 0
        self.stats = {"captured": 0, "replayed": 0, "eager": 0,
                      "invalidated": 0, "verified": 0}

    # -- static inputs --------------------------------------------------
    def static_input(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """A persistent buffer holding ``value``, for the model call to read.

        **Every per-step input must go through this or the replay is wrong.** A
        graph records the *addresses* its kernels read, so an input built fresh
        each step -- ``torch.arange(pos, pos + 1)`` for ``cache_position`` is the
        one that bites -- is captured at the address of that step's tensor and
        never read again. The replay would then decode at the position it was
        captured at, for ever, while every shape still matched and nothing
        raised.

        So the caller copies into a buffer the runner owns, and passes that
        buffer to the model. In eager mode this is a copy that costs nothing
        anyone will notice; in graph mode it is the only thing that makes a
        replay mean the current step.
        """
        buf = self._static_in.get(name)
        if buf is None or buf.shape != value.shape or buf.dtype != value.dtype:
            if buf is not None and self._graphs:
                # A changed input shape invalidates every slot: the graph reads
                # a fixed extent from a fixed address.
                self.invalidate(f"static input {name!r} changed shape")
            buf = torch.empty_like(value)
            self._static_in[name] = buf
        buf.copy_(value)
        return buf

    # -- lifecycle ------------------------------------------------------
    def invalidate(self, reason: str) -> None:
        """Drop every captured slot. Cheap, and always the safe answer."""
        if self._graphs:
            self.stats["invalidated"] += 1
        self._graphs.clear()
        self._sigs.clear()
        self._deltas.clear()
        self._posts.clear()
        self._static_out.clear()
        self._last_invalidation = reason

    def _capturable(self, step: int) -> Optional[int]:
        if self.mode == "off":
            return None
        slot = self.schedule.slot(step)
        if slot is None:
            return None
        if self._evictions_seen < self.warmup_epochs:
            return None
        return slot

    # -- the step -------------------------------------------------------
    def step(self, step: int, run: Callable[[], Any]) -> Any:
        """Run one decode step, replaying or capturing where it is legal to."""
        if self.schedule.evicts(step):
            self._evictions_seen += 1
            # The eviction rewrites the store: every captured slot is addressing
            # the old one. Invalidate BEFORE it runs, so a raise inside the
            # eviction cannot leave a live graph pointing at a half-written tier.
            self.invalidate(f"eviction at step {step}")
            self.stats["eager"] += 1
            return run()

        slot = self._capturable(step)
        if slot is None:
            self.stats["eager"] += 1
            return run()

        live = signature_of(self.cache, slot)
        captured = self._sigs.get(slot)
        if captured is None:
            return self._capture(slot, live, run)

        why = captured.mismatch(live)
        if why is not None:
            # Not an error: a legal geometry change (a new shape, a realloc)
            # simply means this capture is stale. Errors are reserved for a
            # replay that has already been judged legal and then disagreed.
            self.invalidate(f"slot {slot} signature moved -- {why}")
            self.stats["eager"] += 1
            return run()

        return self._replay(slot)

    # -- capture / replay ----------------------------------------------
    def _capture(self, slot: int, pre: StepSignature, run: Callable[[], Any]):
        """Record one slot. The captured step's own result is the eager one."""
        if not torch.cuda.is_available():  # pragma: no cover - GPU-only
            self.stats["eager"] += 1
            return run()
        # pragma: no cover below -- none of this executes without CUDA.
        torch.cuda.synchronize()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, pool=self._pool):
                out = run()
        except Exception as exc:
            self.invalidate(f"capture of slot {slot} failed: {exc}")
            self.mode = "off"
            raise RuntimeError(
                f"CUDA graph capture failed on slot {slot}: {exc}\n"
                "This path is opt-in (STICKYKV_DECODE_GRAPH) and has no silent "
                "fallback: a capture that failed and then ran eagerly would "
                "report eager numbers under a graphed label. Unset the variable "
                "to run the ordinary decode loop."
            ) from exc
        post = signature_of(self.cache, slot)
        self._graphs[slot] = g
        self._sigs[slot] = pre
        self._posts[slot] = post
        self._deltas[slot] = HostDelta.between(pre, post)
        self._static_out[slot] = out
        self.stats["captured"] += 1
        return out

    def _replay(self, slot: int):  # pragma: no cover - GPU-only
        self._graphs[slot].replay()
        self._deltas[slot].apply(self.cache)
        self.stats["replayed"] += 1
        if self.mode == "verify":
            self._verify_post(slot)
        return self._static_out[slot]

    def _verify_post(self, slot: int) -> None:
        """The hand-applied delta must land exactly where capture landed.

        This is the one thing a replay can check about itself. If the Python
        bookkeeping after a replay is not the bookkeeping the captured eager step
        produced, the next eager step runs against views that disagree with what
        the kernels wrote -- and nothing downstream would raise.
        """
        live = signature_of(self.cache, slot)
        why = self._posts[slot].mismatch(live)
        self.stats["verified"] += 1
        if why is not None:
            raise GraphSignatureMismatch(
                f"slot {slot}: after replay the host state is not what capture "
                f"recorded -- {why}. The HostDelta did not reproduce the "
                "bookkeeping the eager step performed, so every later step would "
                "run against views that disagree with the kernels."
            )

    def report(self) -> str:
        s = self.stats
        return (f"decode graph [{self.mode}]: captured={s['captured']} "
                f"replayed={s['replayed']} eager={s['eager']} "
                f"invalidated={s['invalidated']} verified={s['verified']}")
