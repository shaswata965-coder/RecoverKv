"""WindowedCache — HuggingFace Cache integration for windowed KV cache.

Orchestration only.  No scoring math, no Top-K math, no attention computation,
no RoPE math — only calls into :mod:`state` and :mod:`policy`.

NOTE: This module used to be byte-identical to
``modules/windowed_eager_cache/cache.py`` (the backends differed only in their
``hooks.py``), and everything except the layer-major decode store still is. Any
change here MUST be mirrored to the eager twin until the duplication is
refactored away.

**The one divergence: layer-major decode (DECODE_SPEED_PLAN.md §4.1).** This twin
folds ``L`` into the row axis after the first eviction so all 32 layers evict in
one pass; the eager twin still evicts per layer. That is a launch-count change on
the flash decode path, which is the only path the decode benchmark measures — the
eager backend is the reference/quality path and never reaches the fused decode
kernel. **The two still produce identical caches**, which is not an assumption:
``tests/test_quant_cache.py::test_flash_eager_two_tier_parity`` runs both and
compares K/V, active window ids and effective length every step, and
``tests/test_layer_major_evict.py`` pins the same equivalence within this twin
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import os
import traceback
import warnings

import torch
from torch import Tensor

try:
    from transformers import Cache as _HFCacheBase
except ImportError:
    _HFCacheBase = object  # type: ignore[assignment,misc]

from .config import ResolvedConfig, WindowedCacheConfig
from .policy import EvictionPolicy
from .scorer import accumulate
from .state import CacheState
from .telemetry import NullTelemetry, Telemetry

from modules.quant import (
    QuantizedStore,
    materialize_effective_kv,
    unrotate_key_window,
)
from modules.quant.effective import rotate_key_window
from modules.quant.compact import join, stable_partition
from modules.quant.quantizer import QGrid
from modules.quant.slots import GRID_FIELDS, QuantSlotTable, n_slots_for


# ---------------------------------------------------------------------------
# torch.compile of the eviction step. ALWAYS -- there is no knob and no device
# test: `_evict_two_tier` calls the compiled body on every backend and raises if
# it cannot be lowered. `_evict_two_tier_impl` is still the reference the
# compiled body must equal (tests call it directly), and
# _emulating_precision_casts is what keeps them equal. The compiled callable is
# process-global and built lazily on first use so a run that never evicts never
# pays for it.
# ---------------------------------------------------------------------------

_COMPILED_EVICT_FN = None
_EVICT_ANNOUNCED = {"done": False}
#: True once the dynamic=True compile has failed and we have retried static.
_EVICT_COMPILE_TRIED_STATIC = False

#: Proof-of-execution counters for the eviction path, same contract as
#: :data:`modules.windowed_cache.flash_decode._STATS`: being *able* to compile is
#: not the same as the compiled body actually running. ``compiled`` counts
#: evictions that went through ``_COMPILED_EVICT_FN``. ``eager`` is kept at zero
#: and is now a TRIPWIRE rather than a measurement: since the compiled path
#: became unconditional nothing increments it, so a nonzero value means an eager
#: fallback was reintroduced somewhere and the run's numbers are not the
#: compiled path's.
_EVICT_STATS = {"eager": 0, "compiled": 0}

#: One-shot flag for the reset-and-retry on a Dynamo bail. Not sticky like
#: :data:`_EVICT_COMPILE_FAILED`: a bail is a run property, so it gets exactly
#: one rescue per process and then becomes a real error rather than an infinite
#: reset loop that would recompile the world on every eviction.
_EVICT_BAIL_RETRIED = {"done": False}

#: Sticky record of a torch.compile failure on the eviction, process-global and
#: NOT cleared by :func:`reset_evict_path_stats` — a build that cannot lower the
#: eviction cannot lower it on the next cell either, so we record it ONCE and
#: route every later eviction straight to eager rather than re-attempting (and
#: re-failing) per cell / per layer. ``None`` = no failure seen. Set to a short
#: ``"Type: message"`` string on the first failure. Read via
#: :func:`evict_compile_failed`; the perf runner folds it into each row's
#: provenance so a fallback is visible, never silent.
_EVICT_COMPILE_FAILED: Optional[str] = None

#: The FULL, untruncated traceback of that failure — the Inductor stack that
#: names the exact FX node / aten op / source line that could not be lowered.
#: ``_EVICT_COMPILE_FAILED`` is the one-line summary (and is what gets truncated
#: into the npz); this is the whole thing, so a diagnosis never depends on the
#: summary. Read via :func:`evict_compile_traceback`; the perf runner writes it
#: to a file next to the outputs so it survives the npz's field-length cap.
_EVICT_COMPILE_TRACEBACK: Optional[str] = None


def evict_path_stats() -> dict:
    """``{"eager": n, "compiled": m}`` — evictions per path since the last reset."""
    return dict(_EVICT_STATS)


#: Widths the demote / reactivate compacts may take. A count that varies per
#: eviction gives ``torch.compile`` a new shape every time, so the measured
#: count is rounded UP to one of these rungs. The first eviction, where
#: everything genuinely is fresh, still lands on the top one.
_EVICT_WIDTH_LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def _evict_widths(masks, bounds):
    """Smallest ladder rung covering each mask's widest row. **One sync, total.**

    ``bounds`` is one cap for every mask, or one per mask. They differ: the
    demote and reactivate widths are capped by ``n_q`` (the Q tier's window
    count) and the promote width by ``min(N_q, n_fp)`` (what the fp side can
    take back). Passing them together is what keeps this to a single host read.

    ``_compact`` allocates by RANK into a worst-case width, because the counts
    are ragged per row. That width is ``n_q`` -- every window the Q tier can
    hold -- and at steady state only a handful of lanes are really fresh: one
    window is sealed per ``window_size`` steps, plus whatever crosses the tier
    boundary on re-ranking. Everything else is gathered, un-rotated, quantized
    and sketched, then masked away.

    It did not matter much under ``quant_budget_mode: tokens``, where ``n_q`` is
    64. Under ``bytes`` -- the mode the budget is honestly spent in -- ``n_q`` is
    **250**. Fitting DECODE_SPEED_PLAN.md §3's measured eviction cost against
    the 2026-09-17 table reproduces every cell to within 2-13% only if the
    eviction scales with ``n_q``, and it puts the eviction at **~60 ms of a
    124 ms step** at 4096/B=32. About half. This width is now the largest single
    item in the decode step, not a tidy-up.

    **The bound has to come from the data.** ``_compact`` routes overflow lanes
    to a dump column, so a width that is ever too small silently drops work. A
    max over rows is a true upper bound by construction; config arithmetic is
    not, which is why this pays a ``.item()`` rather than estimating one.

    That ``.item()`` costs one graph break in the compiled eviction -- two
    Inductor graphs instead of one. The trade is explicit and it is not close:
    one break against a payload up to 31x smaller. All the demote work sits
    AFTER the width is known, so it stays inside one fused region either way.
    """
    masks = tuple(masks)
    if isinstance(bounds, int):
        bounds = (bounds,) * len(masks)
    else:
        bounds = tuple(int(b) for b in bounds)
        if len(bounds) != len(masks):
            raise ValueError(
                f"{len(masks)} masks but {len(bounds)} bounds")
    if all(b <= 0 for b in bounds):
        return [0] * len(masks)
    # ONE `.item()` for every width this eviction needs. Two separate reads
    # would be two host syncs and two graph breaks per eviction, for the same
    # information; stacking them first makes it one of each.
    #
    # Per mask, NOT one reduction over a stack of them: the masks are different
    # widths -- fresh/react span the merged window axis, promote spans the fp
    # picks -- so stacking raises. (Tried; the compiled eviction refused it on
    # the first call, which is the kernel-or-error contract working.)
    needs = torch.stack([m.sum(1).max() for m in masks]).tolist()
    out = []
    for need, bound in zip(needs, bounds):
        need = int(need)
        if need <= 0 or bound <= 0:
            out.append(0)
            continue
        for rung in _EVICT_WIDTH_LADDER:
            if rung >= need:
                out.append(min(rung, bound))
                break
        else:
            out.append(bound)
    return out


def evict_width_stats(cache) -> Optional[dict]:
    """What ``_compact``'s worst-case width bought, per eviction. Syncs once.

    ``None`` until an eviction has run. Otherwise::

        {"evictions", "n_q", "W_retained",
         "fresh_max", "fresh_mean", "promote_max", "promote_mean", "react_max",
         "fresh_fill", "promote_fill"}

    ``fresh_fill`` is ``fresh_max / n_q`` and ``promote_fill`` is
    ``promote_max / n_q`` -- **the numbers that size D1.** Sized by their caps,
    the demote path un-rotates, quantizes and sketches
    ``[L*B, n_q, H_kv, ws, D]`` every eviction and the promote path dequantizes,
    RoPEs and splices ``[L*B, min(N_q, n_fp), H_kv, ws, D]``, with everything
    beyond a row's real count masked away. A fill of 0.05 says 95% of that
    payload is discarded; a fill near 1.0 says the cap is about right and
    measuring the width buys nothing.

    Both paths are now sized by the measured max, so these read as *what the cap
    would have cost*, not as waste still being paid.

    the design notes records this as the risk the D1 estimate rests on --
    *"If the real count is routinely close to n_q, D1 is worth little"* -- and it
    has never been printed. Read it AFTER a timed window, never inside one: the
    single ``.item()`` here is the host sync the counters exist to avoid on the
    eviction path itself.
    """
    buf = getattr(cache, "_evict_widths", None)
    bound = getattr(cache, "_evict_width_bound", None)
    if buf is None or bound is None:
        return None
    ev, fmax, pmax, rmax, fsum, psum = (int(x) for x in buf.tolist())
    if ev == 0:
        return None
    n_q, W = bound
    return {
        "evictions": ev, "n_q": n_q, "W_retained": W,
        "fresh_max": fmax, "fresh_mean": fsum / ev,
        "promote_max": pmax, "promote_mean": psum / ev,
        "react_max": rmax,
        "fresh_fill": (fmax / n_q) if n_q else None,
        "promote_fill": (pmax / n_q) if n_q else None,
    }


def evict_compile_failed() -> Optional[str]:
    """The eviction's torch.compile failure (``"Type: msg"``), or ``None``.

    Sticky for the life of the process. A non-None value means the compiled
    eviction was requested and could not be lowered on THIS torch/build (the run
    then raised — kernel-or-error). Pair with :func:`evict_compile_traceback` for
    the op that failed.
    """
    return _EVICT_COMPILE_FAILED


def evict_compile_traceback() -> Optional[str]:
    """The FULL Inductor traceback of the eviction compile failure, or ``None``.

    Untruncated, unlike the npz's ``skip_reason``: this is the stack that names
    the exact aten op / FX node / source line that did not lower, which is what a
    fix actually needs. The perf runner also writes it to a file.
    """
    return _EVICT_COMPILE_TRACEBACK


def reset_evict_path_stats() -> None:
    """Zero the per-path counters. Does NOT clear the sticky compile-failure flag
    (that is a property of the build, not of one measured run).

    It DOES clear the one-shot bail rescue, and that is the right granularity:
    the recompile budget is process-wide, so a cell can arrive with it already
    spent by the cell before. One rescue per measured run means each cell can
    recover once from someone else's spending, while a cell that bails twice on
    its own still fails loudly instead of resetting the world every eviction.
    """
    _EVICT_STATS["eager"] = 0
    _EVICT_STATS["compiled"] = 0
    _EVICT_BAIL_RETRIED["done"] = False


class _LayerStateView:
    """Read-only per-layer window onto the layer-major :class:`CacheState`.

    ``WindowedCache._states[i]`` keeps working after the join — for the score
    hooks, the parity runner and the tests — by becoming one of these. Slices are
    built lazily on attribute access (a view, no kernel) rather than refreshed
    eagerly every step, so the fused decode path, which reads none of them, pays
    nothing.

    Writes go through the joint state, never through a view: a view has no
    buffers of its own and silently mutating one would desynchronise the layer
    from the store the eviction actually rewrites.
    """

    __slots__ = ("_joint", "_r0", "_rows")

    def __init__(self, joint: CacheState, r0: int, rows: int) -> None:
        self._joint = joint
        self._r0 = r0
        self._rows = rows

    def _rowslice(self, t: Optional[Tensor]) -> Optional[Tensor]:
        return None if t is None else t[self._r0:self._r0 + self._rows]

    @property
    def key_states(self) -> Optional[Tensor]:
        return self._rowslice(self._joint.key_states)

    @property
    def value_states(self) -> Optional[Tensor]:
        return self._rowslice(self._joint.value_states)

    @property
    def position_ids(self) -> Optional[Tensor]:
        return self._rowslice(self._joint.position_ids)

    @property
    def window_scores(self) -> Optional[Tensor]:
        return self._rowslice(self._joint.window_scores)

    @property
    def original_window_ids(self) -> Optional[Tensor]:
        return self._rowslice(self._joint.original_window_ids)

    @property
    def seq_length(self) -> int:
        return self._joint.seq_length


class _ReadOnlySlotTableView(QuantSlotTable):
    """One layer's rows of a layer-major slot table, for reads only.

    Every read on :class:`QuantSlotTable` goes through the row axis generically —
    ``lookup`` matches ``[B, W, N]``, ``free_slots`` and ``active_order`` argsort
    the slot axis, ``gather`` flat-indexes through ``_row_base`` — so a view built
    from row slices plus a rebuilt ``_row_base`` answers all of them exactly for
    the layer it covers.

    The mutators do NOT work on a slice: ``retain_only`` rebinds ``slot_wid``
    rather than writing through it, so a mutation here would land on the view and
    vanish. They raise instead of silently doing nothing — the eviction mutates
    the joint table directly and has no reason to come through here.
    """

    def __init__(self, parent: QuantSlotTable, r0: int, rows: int) -> None:
        self.batch_size = rows
        self.n_slots = parent.n_slots
        self.window_size = parent.window_size
        self.head_dim = parent.head_dim
        self.num_kv_heads = parent.num_kv_heads
        for field in ("key_codes", "val_codes", *GRID_FIELDS,
                      "slot_wid", "slot_active", "slot_pos"):
            setattr(self, field, getattr(parent, field)[r0:r0 + rows])
        self._row_base = (
            torch.arange(rows, device=parent.slot_wid.device) * parent.n_slots
        ).unsqueeze(1)

    def _readonly(self, *_a, **_k):
        raise RuntimeError(
            "this slot table is a per-layer VIEW of the layer-major table and is "
            "read-only; mutations must go through the joint table the batched "
            "eviction owns (WindowedCache._joint_store.table)"
        )

    write = set_active = retain_only = _readonly


class _LayerStoreView:
    """Read-only per-layer window onto the layer-major :class:`QuantizedStore`.

    The counts (``num_active_windows``, ``num_active_tokens``, ``version``) are
    layer-independent by construction — they come from the budget resolver, which
    every layer resolves identically — so they pass straight through.
    :meth:`active_ids` and :attr:`table` are row-shaped and get sliced.
    """

    __slots__ = ("_joint", "_r0", "_rows")

    def __init__(self, joint: QuantizedStore, r0: int, rows: int) -> None:
        self._joint = joint
        self._r0 = r0
        self._rows = rows

    @property
    def num_active_windows(self) -> int:
        return self._joint.num_active_windows

    @property
    def num_active_tokens(self) -> int:
        return self._joint.num_active_tokens

    @property
    def version(self) -> int:
        return self._joint.version

    @property
    def memoize_read(self) -> bool:
        return self._joint.memoize_read

    @property
    def table(self):
        """This layer's rows of the joint slot table (read-only).

        At ``L == 1`` the slice is the whole table, so the joint object is handed
        back unwrapped — no view, no copy, and mutators keep working for the
        single-layer callers (the quant unit tests) that drive the store directly.
        """
        t = self._joint.table
        if t is None or (self._r0 == 0 and self._rows == t.batch_size):
            return t
        return _ReadOnlySlotTableView(t, self._r0, self._rows)

    def active_ids(self) -> Optional[Tensor]:
        ids = self._joint.active_ids()
        return None if ids is None else ids[self._r0:self._r0 + self._rows]

    def validate(self) -> None:
        """Assert the joint store's invariants (test-only; syncs).

        Deliberately validates the WHOLE table rather than this layer's slice:
        the invariants — one slot per window id per row, no active-but-free slot,
        ``_n_active`` agreeing with ``slot_active`` — are properties of the table
        the eviction actually writes, and checking a read-only slice of it would
        be a weaker claim that could pass while the joint table was broken.
        """
        self._joint.validate()


def _clamp_index(x: Tensor, hi) -> Tensor:
    """Clamp an index tensor to ``[0, hi]`` WITHOUT ever emitting ``aten.amin``.

    torch <= 2.6's Inductor cannot schedule the scalar ``StarDep`` that a clamp's
    UPPER bound lowers to ("StarDep does not have an index on aten.amin.default"):
    ``tensor.clamp(max=s)`` / ``clamp_max`` / a two-sided ``clamp(x, lo, hi)`` /
    ``torch.minimum(x, scalar)`` all decompose through ``aten.amin`` on the upper
    side. The LOWER bound (``clamp_min`` → ``aten.amax``) lowers fine — that is why
    the bare ``clamp_min(0)`` calls in this file never errored.

    So build the upper bound out of ``clamp_min`` only, via the identity
    ``min(x, hi) == hi - relu(hi - x)``: ``hi - x`` is a scalar-minus-tensor
    (pointwise), ``.clamp_min(0)`` is the working ``relu`` path, and the outer
    ``hi -`` is pointwise. No ``amin`` node is ever created. ``hi`` is a concrete
    Python int (the body makes its shape-derived ints concrete), so nothing here
    is symbolic either.
    """
    x = x.clamp_min(0)
    return hi - (hi - x).clamp_min(0)


def _emulating_precision_casts(fn):
    """Wrap ``fn`` so Inductor keeps every narrowing cast in the eviction graph.

    Inductor does not emulate intermediate precision casts by default: it drops
    an ``fp32 -> fp16 -> fp32`` round trip and keeps the wider value in a
    register. For most code that is a free accuracy win. In the eviction it is a
    correctness bug, because three quantisers in that graph use the round trip
    deliberately — they fit codes to the **fp16-stored** grid so the grid the
    codes were fit to is bit-identical to the grid every later dequant reads
    (``quant/quantizer.py`` module docstring, design §2):

    - ``_affine_quantize`` (the int2 Q tier). Measured on 2.14 CPU Inductor over
      24 seeds: without this flag the packed codes differ from eager on 11/24
      seeds for keys and 14/24 for values — one int2 level, on elements whose
      pre-round quantity straddles .5 because the two grids differ by an fp16
      ulp. With it: 0/24.
    - ``sketch._q_sym`` and ``sketch._q_up`` (the gate cards). Worse: 24/24 seeds
      diverge, and ``_q_up``'s ``dequant >= x`` guarantee — which is what makes
      the gate's Cauchy–Schwarz bound an upper bound at all — is violated 115
      times in that sample. Its ``(1 + 2**-9)`` scale nudge is sized for the fp16
      grid; against an un-narrowed one it is sized for the wrong grid. With this
      flag: 0 divergences, 0 violations.

    The sketch half is not new — those helpers were never behind a graph break,
    so every compiled eviction since the cards landed has been building them on
    the wrong grid. Tracing the int2 quantiser (removing its ``@_compile_disable``)
    is what makes the flag load-bearing for the Q tier too.

    Scoped, not global: ``config.patch`` is entered per call and restores the
    process default on exit, so nothing else the host compiles is affected.
    Inductor keys its code cache on config, so the wrapped callable compiles once
    and the patch is a dict swap on every later call. A no-op when the backend is
    not Inductor (``aot_eager`` runs eager kernels, which narrow by definition).
    """
    try:
        from torch._inductor import config as inductor_config
    except Exception:  # pragma: no cover - no Inductor on this build
        return fn
    if not hasattr(inductor_config, "emulate_precision_casts"):
        # Then _quant_may_be_traced() is False and the int2 quantiser keeps its
        # graph break, so the Q tier is safe. The sketch cards are NOT — they
        # were never behind one — and on such a build there is no knob here that
        # would protect them. That is the pre-existing behaviour, unchanged.
        return fn  # pragma: no cover - torch-version dependent
    return inductor_config.patch("emulate_precision_casts", True)(fn)


class EvictionRanEager(RuntimeError):
    """The compiled eviction body executed in Python instead of being traced.

    A distinct type because it is a **run** property, not a **build** property,
    and the two must not share a fate. A lowering failure says "this build
    cannot lower this graph" -- true for every later cell, which is why
    :data:`_EVICT_COMPILE_FAILED` is sticky. A Dynamo bail says "Dynamo declined
    this frame right now", and the commonest reason is that the process-wide
    recompile budget is already spent by an EARLIER cell. Filing that as a build
    failure is what turned one failing cell into five untried ones on the
    2026-09-19 sweep.
    """


def _scalar_outputs_off(fn):
    """Wrap ``fn`` so ``.item()`` / ``.tolist()`` GRAPH-BREAK instead of going symbolic.

    **This is the fix for the 2026-09-19 total outage, and it is a torch default
    that flipped under code written for the other one.**

    :func:`_evict_widths` reads three widths with one ``.tolist()`` and then
    branches on them::

        needs = torch.stack([m.sum(1).max() for m in masks]).tolist()
        ...
        if need <= 0 or bound <= 0:

    Its docstring states the intended cost out loud -- *"that .item() costs one
    graph break in the compiled eviction -- two Inductor graphs instead of one"*
    -- and ``tests/test_evict_graph_breaks.py`` was written against exactly that,
    down to the recorded message ``Unsupported Tensor.item() call with
    capture_scalar_outputs=False``.

    Under ``capture_scalar_outputs=True`` -- **the default since torch 2.6** --
    ``.tolist()`` stops breaking and returns an *unbacked* symint instead. The
    branch above then becomes a guard on ``u0 <= 0`` that Dynamo cannot resolve,
    and its response is not a graph break: it abandons the frame and runs the
    whole eviction EAGER. Measured on the 2026-09-19 run::

        frames: {'total': 10, 'ok': 9}
        9x  Could not guard on data-dependent expression u0 <= 0
        Caused by: if need <= 0 or bound <= 0:   # cache.py:168 in _evict_widths

    **Capturing this scalar symbolically is not merely unhelpful, it is
    incompatible with the design.** The width is what ``_compact`` allocates
    with, and ``_evict_widths`` exists precisely to pay one host sync for a real
    integer, because a width that is ever too small routes overflow to a dump
    column and silently drops work. A symbolic width cannot size that allocation
    without a guard, and the guard is the thing that cannot be resolved. So the
    sync is the point, and the break that comes with it is the designed
    structure -- not a wart to be optimised away by a config default.

    Scoped, not global, mirroring :func:`_emulating_precision_casts`:
    ``config.patch`` is entered per call and restores the process default on
    exit, so nothing else the host compiles is affected.
    """
    try:
        from torch._dynamo import config as dynamo_config
    except Exception:  # pragma: no cover - no Dynamo on this build
        return fn
    if not hasattr(dynamo_config, "capture_scalar_outputs"):
        return fn  # pragma: no cover - torch-version dependent
    return dynamo_config.patch("capture_scalar_outputs", False)(fn)


#: Dynamo graphs this one code object may hold before the recompile limit fires.
#:
#: The default is 8, and the eviction blew it: the body used to take ``layer_idx``
#: and index a Python list with it, so the 32 per-layer calls of the FIRST
#: eviction minted 32 specialisations and the joint evictions -- the 39 that
#: matter -- never got a slot. That is fixed at the source (see
#: :meth:`WindowedCache._evict_two_tier`), so the specialisations left are the
#: shape ones the body asks for ON PURPOSE: it makes its control ints concrete
#: with ``int()`` so Inductor gets real integers instead of symbols, which costs a
#: graph per distinct geometry during the fill phase and one at steady state.
#:
#: A limit is not a budget to spend -- it is the point at which Dynamo silently
#: STOPS compiling. So it is set well clear of the handful expected, and the body
#: raises outright if it ever runs eager anyway. The eviction is the only
#: ``torch.compile`` in this repo, so nothing else competes for this.
_EVICT_CACHE_SIZE_LIMIT = 64


#: The recompile-limit config names Dynamo has used across versions. Both the
#: per-code-object limit and the process-wide accumulated one matter, and both
#: have been renamed, so every spelling is tried and the ones that exist are read
#: back into the diagnosis.
_DYNAMO_LIMIT_NAMES = (
    "cache_size_limit", "accumulated_cache_size_limit",
    "recompile_limit", "accumulated_recompile_limit",
)


def _raise_dynamo_cache_limit() -> None:
    """Give the eviction's code object room for its shape specialisations.

    Idempotent and cheap; called before every compiled eviction rather than once
    at build, because a `torch._dynamo.reset()` or another library's
    `config.patch` can put a limit back between evictions and the cost of
    re-checking is four attribute reads.

    **Only ever raised, never lowered** -- a caller that deliberately set a
    higher limit keeps it, and Dynamo's `accumulated_*` limits default to 256,
    well above ours.
    """
    import torch._dynamo as _td

    # `hasattr` first: `torch._dynamo.config` is a ConfigModule that rejects
    # unknown names.
    for name in _DYNAMO_LIMIT_NAMES:
        if not hasattr(_td.config, name):
            continue
        if getattr(_td.config, name) < _EVICT_CACHE_SIZE_LIMIT:
            setattr(_td.config, name, _EVICT_CACHE_SIZE_LIMIT)


def _dynamo_diagnosis() -> str:
    """Dynamo's own state, read at the moment an eviction ran eager.

    This exists because the first version of the eager check named three
    possible causes and gave the reader no way to tell which. That cost a whole
    GPU run: the failure said "one of these three" and the evidence needed to
    choose was sitting in `torch._dynamo` the entire time. A diagnostic that
    names a suspect list is not a diagnostic.

    Everything here is read defensively -- this runs on a path that is already
    failing, and a diagnosis that raises while diagnosing is worse than none.
    """
    lines = []
    try:
        import torch._dynamo as _td

        limits = {n: getattr(_td.config, n, None) for n in _DYNAMO_LIMIT_NAMES}
        lines.append(
            "  limits now: "
            + ", ".join(f"{n}={v}" for n, v in limits.items() if v is not None)
            + f"   (this module raises them to {_EVICT_CACHE_SIZE_LIMIT})"
        )
        # If a limit reads BELOW ours, `_raise_dynamo_cache_limit` did not take
        # -- a different config object, a ConfigModule that ignored the write, or
        # something lowered it afterwards. That is cause 1 and it is now provable
        # rather than inferred.
        low = [n for n, v in limits.items()
               if v is not None and v < _EVICT_CACHE_SIZE_LIMIT]
        if low:
            lines.append(
                f"  !! {low} read BELOW {_EVICT_CACHE_SIZE_LIMIT}: the raise did "
                "not take. THIS IS CAUSE 1 and it is a bug in "
                "_raise_dynamo_cache_limit, not in your run."
            )
        for flag in ("disable", "suppress_errors"):
            v = getattr(_td.config, flag, None)
            if v:
                lines.append(f"  !! torch._dynamo.config.{flag}={v}  <- CAUSE 2")
    except Exception as exc:  # pragma: no cover - diagnosis must not raise
        lines.append(f"  (dynamo config unreadable: {type(exc).__name__}: {exc})")

    try:
        import os as _os
        for env in ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
            if _os.environ.get(env):
                lines.append(f"  !! {env}={_os.environ[env]}  <- CAUSE 2")
    except Exception:  # pragma: no cover
        pass

    try:
        from torch._dynamo.utils import counters as _counters

        frames = dict(_counters.get("frames", {}))
        if frames:
            lines.append(f"  frames: {frames}   (ok << total means Dynamo bailed)")
        # THE NUMBER THAT SETTLES THE TPOT REGRESSION, and it is free: Dynamo
        # publishes it. `_EVICT_STATS["compiled"]` counts CALLS through the
        # compiled callable and says nothing about how many graphs were built.
        # If this reads ~2x the eviction count, the body is minting a graph per
        # eviction -- `T_fp` and `W` are int()-forced and both move while the Q
        # tier fills -- and the compile time IS the regression. A wrapper around
        # the backend was tried to get this and broke the sweep; see
        # `_build_compiled_evict`.
        stats = dict(_counters.get("stats", {}))
        if stats:
            lines.append(f"  GRAPHS BUILT: {stats}   <- ~2x the eviction count "
                         "means a graph per eviction")
        breaks = _counters.get("graph_break", {})
        if breaks:
            top = sorted(breaks.items(), key=lambda kv: -kv[1])[:5]
            lines.append("  top graph breaks (CAUSE 3 if one of these is new):")
            lines.extend(f"      {n}x  {r}" for r, n in top)
            # The one we have already diagnosed and fixed. If it is back, the
            # scoped patch in `_scalar_outputs_off` is not reaching the trace.
            if any("data-dependent" in r or "u0" in r for r, _ in top):
                lines.append(
                    "  !! a data-dependent guard is back. That is the KNOWN "
                    "2026-09-19 outage: `_evict_widths` branches on a "
                    "`.tolist()` width, which must GRAPH-BREAK. Check that "
                    "`_scalar_outputs_off` is still wrapping the compiled "
                    "callable and that capture_scalar_outputs reads False "
                    "during the trace."
                )
    except Exception as exc:  # pragma: no cover
        lines.append(f"  (dynamo counters unreadable: {type(exc).__name__}: {exc})")

    lines.append(
        f"  evictions compiled before this one: {_EVICT_STATS['compiled']}. "
        "If that is large and this is the first failure, the graph count grew "
        "with the SHAPES -- the body forces int() on W and T_fp, which move at "
        "every eviction until the Q tier stops filling, so a run that never "
        "reaches steady state mints a graph per eviction."
    )
    return "\n".join(lines)


def _build_compiled_evict(dynamic: bool):
    """``torch.compile`` the eviction body. Lazy — never raises here; a lowering
    failure surfaces on the first CALL, not at construction."""
    _raise_dynamo_cache_limit()
    # NO custom backend -- but NOT for the reason first recorded here. A counting
    # wrapper around `compile_fx` was tried on 2026-09-19 and the sweep failed
    # with `IndexError: list assignment index out of range`; that was blamed on
    # the wrapper. **It was not the wrapper.** The next run, with the wrapper
    # removed, raised the identical error under stock `backend='inductor'`. The
    # cause is `merge_view_inputs` (see `_EVICT_COMPILE_HELP`).
    #
    # The wrapper is still gone, on its own merits: the graph count it existed
    # to collect is already published read-only in `torch._dynamo.utils.counters`
    # (see `_dynamo_diagnosis`), so it bought a free number and added a moving
    # part to the one path in this repo that cannot afford one.
    kw: Dict[str, Any] = {"dynamic": dynamic}
    # Both wrappers are scoped `config.patch`es entered per call. `_scalar_
    # outputs_off` is outermost because it governs DYNAMO (whether `.tolist()`
    # breaks the graph at all) while `_emulating_precision_casts` governs
    # INDUCTOR (how the resulting graphs lower), so the former has to be in
    # effect while the latter's graphs are being traced.
    return _scalar_outputs_off(
        _emulating_precision_casts(
            torch.compile(WindowedCache._evict_two_tier_impl, **kw)))


_EVICT_COMPILE_HELP = """The compiled eviction is KERNEL-OR-ERROR: it never silently runs the eager
body under a compiled label -- the point is to MEASURE the compiled path, and a
quiet eager run would report eager numbers as if compiled. So this raises rather
than falling back.
  THERE IS NO WAY TO TURN IT OFF. Compilation is unconditional (see the banner
at the top of this module): `_evict_two_tier` always calls the compiled body, on
every backend and every device. `STICKYKV_COMPILE_EVICT`, `--compile-evict` and
`STICKYKV_COMPILE_EVICT_BACKEND` were deleted in 0974687 ("one production path,
no fallbacks") and this text used to still advertise all three -- so a failure
here sent you after knobs that no longer exist. This module reads NO environment
variable. `perf_runner` still SETS `STICKYKV_COMPILE_EVICT=1`, which nothing
reads; it records intent, it does not control anything. A lowering failure is
therefore a total outage until it is fixed, which is why the diagnosis below is
the only path forward.
  What this is: torch.compile / Inductor could not lower
WindowedCache._evict_two_tier_impl on this build. Both attempts are reported --
`dynamic=True` first, then `dynamic=False` after a `torch._dynamo.reset()` (the
reset is load-bearing: without it Dynamo reuses the code object's frame state and
the "static" retry gets automatic-dynamic promoted straight back, so it re-raises
the identical symbolic error). The message recorded below is the SECOND attempt's.
  The known causes are addressed at the source, and each has a named home:
    * `The argument '((I)//8)' is not comparable` -- a `torch.cat` in this graph.
      READ THIS BEFORE TOUCHING A `.shape` READ: it is NOT a symbolic-shape bug,
      and two rounds of forcing `int()` on shape reads (`d44baba`, `014c40b`)
      changed nothing because of that. `I` is not the imaginary unit and not a
      shape symbol -- it is `torch.utils._sympy.functions.Identity`, Inductor's
      "do not expand this" wrapper, which prints as `I` only because sympy's
      StrPrinter defines `_print_Identity` for the identity MATRIX and dispatches
      on the class name. So `((I)//8)` is `FloorDiv(Identity(<expr>), ws)`.
      Inductor builds that wrapper in exactly one place --
      `_inductor/lowering.py::pointwise_cat` -- so a cat lowered pointwise is the
      only way to get one. It then blows up in `_simplify_loops` -> `stride_vars`,
      which substitutes every index var with 0: `Identity(i - h)` becomes
      `Identity(-h)`, which has no free symbols, and sympy's `Min`/`Max` rejects
      any arg that `is_number` but is not `is_comparable`. That `is_number` is
      the proof the value was already CONCRETE -- and why the `dynamic=False`
      retry failed with the byte-identical message instead of a different one.
      The fix is never `int()`; it is to build the tensor without a cat:
      `compact.join` for a plain join, `effective._rotate_half_no_cat` for RoPE's
      rotation. `test_the_compiled_eviction_contains_no_cat` asserts zero cats in
      this graph, which is what makes the failure impossible rather than
      unlikely; it asserts ZERO because removing one cat changes the fusion
      heuristics and can expose another (it did, twice).
    * `IndexError: list assignment index out of range` in
      `_functorch/_aot_autograd/runtime_wrappers.py::merge_view_inputs` --
      **FIXED 2026-09-19 by moving the store write out of the graph; read this
      before putting one back.** It is an UPSTREAM bug, not a lowering failure
      in the usual sense. AOTAutograd builds "synthetic
      bases" when graph inputs are VIEWS ALIASING A COMMON BASE and at least one
      is mutated, and it mis-sizes `post_processed_calling_convention_meta`.
      This body is made of exactly that shape: `_LayerStateView` ->
      `joint.key_states` -> `_key_buf`, plus the `body_k` / `body_v` /
      `body_pos` slices of the same buffer.
      It fires in the RESUMED region (`torch_dynamo_resume_in_..._at_2701`, at
      `store.demote_many`), i.e. the half created by `_evict_widths`' `.tolist()`
      break -- so it is reachable only when that break exists, which is to say
      only when `capture_scalar_outputs` is False. With it True (the torch 2.6
      default) there is no break and Dynamo abandons the whole frame to eager
      instead. Both roads ended at "not compiled", by two independent
      mechanisms.
      **The fix was (B), and the aliasing pair is named exactly:**
      `replace_body` does `_key_buf[:, :, num_sink:n] = body_key`, mutating the
      BASE that `state.key_states` is a VIEW of -- and the region reads that
      view, as `body_k`. Three such pairs (key / value / positions); the
      `replace` branch is worse, reading `state.key_states` and reallocating in
      one expression. Both now run in `_evict_two_tier`, outside the graph, and
      the compiled body returns the compacted body instead of installing it.
      **The invariant to keep: the compiled eviction may mutate the slot table
      (separate allocations) but must never write a buffer it reads a view of.**
      Putting such a write back re-breaks the build.
    * an `aten.amin` StarDep from a single-sided clamp -- use `_clamp_index`.
    * a symbolic scalar reaching `torch.arange`, a `torch.where` sentinel, a
      slice bound or a reshape extent. Still a real hazard class, just NOT the
      one above: every shape-derived scalar in the region is forced with `int()`
      and `test_scalar_shape_reads_in_the_compiled_region_are_int_forced` keeps
      it that way. `B` (and the flattened `B * n`) is the one axis deliberately
      left symbolic.
  If it still fails, the traceback names the op that did not lower.
  To diagnose: the FULL Inductor traceback (which names the exact aten op /
source line) is in evict_compile_traceback() and is written by the perf runner to
<output_dir>/evict_compile_error_*.txt -- read that first. Then
TORCH_LOGS='+inductor,graph_breaks' TORCHDYNAMO_VERBOSE=1 on the failing shape.
  To reproduce on a CPU box, with no GPU: `pytest
tests/test_evict_graph_breaks.py -k no_cat`. It needs
`force_pointwise_cat=True`, and that is not a detail -- `lowering.py::cat`
returns a `ConcatKernel` UNCONDITIONALLY for CPU devices ("negative performance
impact of pointwise_cat optimization on CPU"), so a CPU box never takes the
pointwise path and never builds an `Identity` at all. That branch, not anything
about the C++ backend tolerating symbolic ints, is why this class of failure is
invisible on CPU by default. The test also disables `fx_graph_cache`, because a
cache hit lowers nothing and passes vacuously."""


def _cuda_oom_cause(exc: BaseException) -> Optional[BaseException]:
    """The out-of-memory exception inside ``exc``'s cause chain, or ``None``.

    Kept separate from the lowering-failure path because the two mean opposite
    things. A lowering failure is a property of the BUILD -- it will fail the
    same way on every later cell, so recording it once and failing fast is
    right. An OOM is a property of THIS CELL's shape and the allocator's
    fragmentation right now; the next cell may be half the size and fine.

    Returns the exception rather than a bool so the caller can re-raise the OOM
    *itself*. That matters: ``perf_runner`` catches ``torch.cuda.OutOfMemoryError``
    by type to mark a cell ``oom`` (skipped, and legitimate max-B evidence) and
    everything else to mark it ``errored`` (a code/config bug, explicitly NOT
    max-B evidence). torch.compile may deliver the OOM wrapped in a Dynamo
    ``BackendCompilerFailed``, and re-raising the wrapper would file a genuine
    OOM under "errored". Unwrapping keeps the two categories meaning what the
    runner's own log message says they mean.
    """
    chain: List[BaseException] = []
    seen: set = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__

    # Typed first, and over the WHOLE chain before falling back to the message.
    # Order matters: Dynamo's `BackendCompilerFailed` repeats the inner error's
    # text, so a message test applied depth-first returns the wrapper -- which
    # is precisely the type the perf runner does not catch.
    for c in chain:
        if isinstance(c, torch.cuda.OutOfMemoryError):
            return c
    for c in chain:
        if type(c).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
            return c
    # Last resort: the phrase, but normalised to the type the runner catches.
    # A wrapper that only *says* "out of memory" is still an OOM as far as the
    # sweep is concerned, and filing it as a build failure is the bug this
    # function exists to stop. The original is attached, so the autopsy and the
    # traceback still show what actually happened.
    for c in chain:
        if "out of memory" in str(c).lower():
            normalised = torch.cuda.OutOfMemoryError(str(c))
            normalised.__cause__ = exc
            return normalised
    return None


def _run_compiled_evict(cache, state, store, policy, step: int):
    """Run one eviction through ``torch.compile``, or RAISE. Kernel-or-error: no
    eager fallback (a silent eager run under a compiled label defeats the whole
    measurement -- the design intent). On success ``cache`` is mutated in place
    and the body's ``(pre-eviction scores, retained indices)`` pair is returned
    for the caller to record as telemetry.

    ``step`` is taken for the failure message only and is NOT forwarded: an int
    that increments every eviction is an int Dynamo specialises on, and the graph
    this function exists to build must not carry one.

    Tries ``dynamic=True`` first (one graph across the varying window count),
    then once with ``dynamic=False`` (concrete shapes are the belt to the body's
    ``int()`` suspenders). If neither lowers, records the reason in
    :data:`_EVICT_COMPILE_FAILED` (for provenance) and raises ``RuntimeError``
    with actionable steps. Safe to raise on the same step: a lowering/guard error
    fires before any kernel executes, so ``cache`` is untouched and the perf
    runner catches it as an ``errored`` cell, never a corrupt one.
    """
    global _COMPILED_EVICT_FN, _EVICT_COMPILE_FAILED, _EVICT_COMPILE_TRIED_STATIC
    global _EVICT_COMPILE_TRACEBACK

    if _EVICT_COMPILE_FAILED is not None:
        # A prior eviction already proved this build cannot lower it; fail fast
        # with the recorded reason rather than re-attempting the compile per cell.
        raise RuntimeError(
            "torch.compile of the two-tier eviction already failed on this "
            f"build ({_EVICT_COMPILE_FAILED}); refusing the eviction at decode "
            f"step {step}.\n" + _EVICT_COMPILE_HELP
        )
    # Before EVERY call, not only at build: a `torch._dynamo.reset()` (this
    # function performs one on the static retry) or another library's
    # `config.patch` can put a limit back between evictions, and re-checking is
    # four attribute reads against a body that runs once per `ws` steps.
    _raise_dynamo_cache_limit()
    if _COMPILED_EVICT_FN is None:
        _COMPILED_EVICT_FN = _build_compiled_evict(dynamic=True)
    try:
        out = _COMPILED_EVICT_FN(cache, state, store, policy)
        _announce_evict_path_once(compiled=True)
        _EVICT_STATS["compiled"] += 1
        return out
    except EvictionRanEager:
        # A Dynamo BAIL, not a lowering failure -- see EvictionRanEager. The
        # dominant cause is a spent recompile budget, and the budget is
        # process-wide: the 2026-09-19 sweep spent it in its FIRST cell (~32
        # evictions x 2 regions while the Q tier filled) and every later cell
        # bailed on its first eviction.
        #
        # So clear the caches and try once more. `reset()` is the only thing
        # that gives this code object a fresh budget, it is process-global (safe
        # here -- the eviction is the only torch.compile in this repo), and the
        # retry is cheap relative to the sweep it rescues. Safe to retry on the
        # same step: the check that raised sits at the TOP of the body, before
        # any mutation, so `cache` is untouched.
        if _EVICT_BAIL_RETRIED["done"]:
            raise
        _EVICT_BAIL_RETRIED["done"] = True
        import torch._dynamo as _td
        _td.reset()
        _COMPILED_EVICT_FN = _build_compiled_evict(dynamic=True)
        out = _COMPILED_EVICT_FN(cache, state, store, policy)
        _announce_evict_path_once(compiled=True)
        _EVICT_STATS["compiled"] += 1
        return out
    except Exception as e_dyn:  # pragma: no cover - GPU/Inductor-build dependent
        # An OOM is a RESOURCE condition, not a lowering failure, and the two
        # must not be confused. `_EVICT_COMPILE_FAILED` is sticky by design --
        # it means "this build cannot lower the body", so no later cell wastes
        # a compile on it. Recording an OOM there poisons every remaining cell
        # in the sweep: one 4096/batch-32 row that ran out of memory turned all
        # six rows of a table into "already failed on this build", which is not
        # what happened and is not max-B evidence either. Re-raise it as itself
        # so the perf runner's own OOM handling (skip_if_oom, the max-B ladder)
        # sees the exception it is written to catch.
        oom = _cuda_oom_cause(e_dyn)
        if oom is not None:
            raise oom
        terminal = e_dyn
        if not _EVICT_COMPILE_TRIED_STATIC:
            _EVICT_COMPILE_TRIED_STATIC = True
            try:
                # RESET FIRST, or the retry is not a retry. `torch.compile` keys
                # its caches on the code object, and `_evict_two_tier_impl` is
                # the same one we just failed on: Dynamo's per-code-object frame
                # state still records those dims as having varied, so
                # `dynamic=False` gets automatic-dynamic promoted right back and
                # the "static" attempt re-raises the identical symbolic error.
                # That is why a `((I)//8)` lowering failure was reported twice
                # and read as one — the message the runner prints is the STATIC
                # attempt's, recorded only after this branch has run.
                #
                # `reset()` is process-global, and that is acceptable here only
                # because the eviction is the ONLY `torch.compile` in this repo
                # (it also clears Dynamo's counters, which `perf_runner` reads —
                # but this path has already failed, and the failure's own
                # traceback is what gets recorded). Imported locally: `torch`
                # alone does not bring `torch._dynamo` in on every build.
                import torch._dynamo as _td
                _td.reset()
                _COMPILED_EVICT_FN = _build_compiled_evict(dynamic=False)
                out = _COMPILED_EVICT_FN(cache, state, store, policy)
                _announce_evict_path_once(compiled=True)
                _EVICT_STATS["compiled"] += 1
                return out
            except Exception as e_static:
                oom = _cuda_oom_cause(e_static)
                if oom is not None:
                    raise oom
                terminal = e_static
        _EVICT_COMPILE_FAILED = f"{type(terminal).__name__}: {terminal}"[:400]
        # Capture the WHOLE Inductor stack before we lose it — it names the aten
        # op / FX node / source line that did not lower. Try to include the
        # static-retry failure too if we have both.
        _EVICT_COMPILE_TRACEBACK = "".join(
            traceback.format_exception(type(terminal), terminal,
                                       terminal.__traceback__)
        )
        _COMPILED_EVICT_FN = None
        raise RuntimeError(
            "torch.compile of the two-tier eviction failed at decode step "
            f"{step} ({_EVICT_COMPILE_FAILED}).\n" + _EVICT_COMPILE_HELP
        ) from terminal


def _announce_evict_path_once(compiled: bool) -> None:
    """Print, once per process, which eviction path is live."""
    if _EVICT_ANNOUNCED["done"]:
        return
    _EVICT_ANNOUNCED["done"] = True
    if compiled:
        print(
            "[StickyKV] eviction path: COMPILED two-tier eviction ACTIVE [OK] "
            "(unconditional; no env var selects this)",
            flush=True,
        )
    else:
        print(
            "[StickyKV] eviction path: eager two-tier eviction "
            "(STICKYKV_COMPILE_EVICT off)",
            flush=True,
        )


class WindowedCache(_HFCacheBase):
    """Windowed KV cache with H2O-style cumulative eviction.

    Supports ``B > 1`` at any ``quant_ratio``, for **equal-length** prompts.
    Rows evict divergently — each ranks its own windows — but the tier split
    keeps the same *count* per row, so both tiers stay dense and rectangular
    (BATCHING_PLAN.md §3). Ragged / left-padded batches are not implemented, at
    either tier (BATCHING_PLAN.md §4 Phase 3).

    Parameters
    ----------
    config : WindowedCacheConfig
        User-facing configuration.
    prefill_len : int
        Number of tokens in the prompt (used for budget resolution).
    model_config
        HuggingFace ``PretrainedConfig`` or compatible.
    kv_dtype : torch.dtype
        Data type of the KV cache tensors.
    rope_module : nn.Module
        The model's rotary embedding module, used for key re-rotation only
        when ``config.rerotate_on_evict`` is enabled.
    num_layers : int
        Number of transformer layers.
    telemetry : Telemetry, optional
        Telemetry recorder.  Defaults to :class:`NullTelemetry`.
    """

    def __init__(
        self,
        config: WindowedCacheConfig,
        prefill_len: int,
        model_config: Any,
        kv_dtype: torch.dtype,
        rope_module: torch.nn.Module,
        num_layers: int,
        max_tokens: int,
        telemetry: Optional[Telemetry] = None,
    ) -> None:
        if isinstance(_HFCacheBase, type) and _HFCacheBase is not object:
            try:
                super().__init__()
            except (TypeError, ValueError):
                # transformers >= 4.50 changed Cache.__init__ to require
                # `layers` or `layer_class_to_replicate`. We manage our own
                # per-layer state (self._states) and override the full Cache
                # interface, so skipping the base init is safe.
                pass

        self.config = config
        self.resolved = config.resolve(prefill_len, model_config, kv_dtype, max_tokens)
        # Kept for the fused path's RoPE tables, which must be built at the STORE's
        # dtype so the Q tier is re-rotated with the same convention it was
        # un-rotated with (decode_kernel.rope_cos_sin_halves explains why).
        self._kv_dtype: torch.dtype = kv_dtype
        # NOTE: the fused hand-off dict is built fresh per layer per step, on
        # purpose. It carries ``"cache": self``, so parking it on the cache
        # closes a reference cycle (cache -> list -> dict -> cache). Every
        # quality runner builds one cache PER SAMPLE and frees it with a bare
        # ``del`` (``longbench_runner._cleanup_memory``), which is refcounting
        # only -- a cycle keeps each sample's whole KV store alive until the
        # generational collector happens to run. That cost an OOM on narrativeqa
        # and ~6% TPOT from the resulting allocator pressure, to save 5.7 us/step
        # (0.009%). Do not re-add it; test_cache_is_freed_by_refcounting guards it.
        self.rope_module = rope_module
        self.num_layers = num_layers
        self.telemetry = telemetry if telemetry is not None else NullTelemetry()

        # Two-tier quantization (design.md §2–§8). q == 0 disables the Q tier and
        # every path below is byte-identical to the single-tier fp16 cache.
        self._q: float = self.resolved.quant_ratio
        self._stores: List[Optional[QuantizedStore]] = [None] * num_layers
        if self._q > 0.0:
            num_kv_heads = getattr(
                model_config, "num_key_value_heads",
                getattr(model_config, "num_attention_heads", None),
            )
            head_dim = getattr(model_config, "head_dim", None)
            if head_dim is None:
                nh = getattr(model_config, "num_attention_heads", None)
                hidden = getattr(model_config, "hidden_size", None)
                head_dim = hidden // nh
            # Slots are bounded by config — retain_only drops every non-retained
            # window, so live entries never exceed top_k_fp + N_q and no growth
            # policy is needed (design §6; see n_slots_for).
            n_slots = n_slots_for(self.resolved.top_k_fp, self.resolved.N_q)
            self._stores = [
                QuantizedStore(
                    self.resolved.window_size, head_dim, num_kv_heads,
                    n_slots=n_slots,
                    # Provisional: `None` means auto, resolved from the real batch
                    # size at the first update() (see _resolve_memoization).
                    memoize_read=self.resolved.quant_memoize_read is not False,
                    sketch_enabled=self.resolved.quant_sketch_enabled,
                )
                for _ in range(num_layers)
            ]
            # Softmax scale for the fused decode kernel (Llama/Qwen: head_dim**-0.5).
            self._attn_scaling: float = head_dim ** -0.5
        else:
            self._attn_scaling = 1.0
        self._memoization_resolved = False

        # Fused two-tier decode (decode_kernel + flash_decode). Off until the flash
        # score hook activates it (CUDA + triton + STICKYKV_FUSED_DECODE on); the
        # eager backend and CPU tests never activate it, so they keep the
        # materialize read path. See update()'s q>0 return.
        self._fused_decode_active: bool = False

        # Per-layer memo for the fused hand-off (Q-tier gather + RoPE halves +
        # score-scatter map). Keyed on `store.version` and the fp body's window
        # count, so it can only ever hand back what a recompute would produce;
        # _evict_two_tier drops it outright as a second line of defence.
        self._fused_ctx: List[Optional[Dict[str, Any]]] = [None] * num_layers

        # Per-layer state and policy. The fp store is preallocated so append()
        # never re-cats the whole store (BATCHING_PLAN.md §5: at max batch that
        # copy is ~4x the weight traffic and the largest single term in the step).
        #
        # Sized to the EVICTION BUDGET, not to prefill + max_tokens. The latter is
        # StaticCache's rule and is right for a cache that only grows; this one
        # compacts back to the budget every window_size steps, so prefill-sizing
        # would pin the fp store at its un-evicted size for the whole decode —
        # ~708 MB/row vs ~74 MB/row on Llama-3.1-8B at the qasper steady state,
        # capping B at ~73 against the ~458 the method is supposed to reach. The
        # prompt still needs a full-size buffer for exactly one pass, before the
        # first eviction can score it; `replace` releases it at that eviction.
        r = self.resolved
        steady = (
            r.num_sink_tokens
            + r.top_k_fp * r.window_size     # == top_k_windows at q = 0
            + r.local_tokens
            + 2 * r.window_size              # a window of growth + slack
        )
        # The prompt is resident until the FIRST eviction (EvictionPolicy.
        # should_evict fires it at r.first_eviction_step, not at a window
        # boundary), so the prefill buffer must cover the prompt plus every decode
        # token appended up to and including that step: prefill_len +
        # first_eviction_step + 1. The `window_size` term keeps the historical
        # slack for large windows; the max() guarantees no safety-net realloc when
        # window_size <= that step.
        prefill_cap = prefill_len + max(
            r.window_size, r.first_eviction_step + 1
        )
        self._states: List[CacheState] = [
            CacheState(capacity=steady, prefill_capacity=prefill_cap)
            for _ in range(num_layers)
        ]
        self._policies: List[EvictionPolicy] = [
            EvictionPolicy(self.resolved) for _ in range(num_layers)
        ]
        self._generation_step: List[int] = [0] * num_layers
        self._prefill_done: List[bool] = [False] * num_layers
        # Running counter of the next original-sequence window ID to assign
        # when new windows appear (post-eviction or as generation extends the cache).
        # Without this, the W_new > W_old branch would emit compact-space indices
        # that collide with surviving original IDs.
        self._next_original_window_id: List[int] = [0] * num_layers

        # Shared scratch for cache_kwargs communication with hooks
        self.cache_kwargs: Dict[int, Dict[str, Any]] = {
            i: {} for i in range(num_layers)
        }

        # Fix #2: the flash score hook needs the SAME effective K that update()
        # built for this layer this pass. Stash it here so the hook reuses it
        # instead of calling _materialize a second time (which would
        # re-dequantize the entire Q tier again, per layer, per decode step).
        # Stays None at q == 0 (the hook reads state.key_states directly there).
        self._last_effective_k: List[Optional[Tensor]] = [None] * num_layers

        # Paired with _last_effective_k: the score-scatter map (order, q_token_len)
        # that undoes the unsorted [sink ‖ body ‖ Q] effective-K layout on the
        # score axis. None when the Q tier is empty (no reorder needed). Both
        # backends' score hooks consume it; see modules.quant.effective.
        self._last_score_meta: List[Optional[Any]] = [None] * num_layers

        # -----------------------------------------------------------------
        # Layer-major decode (DECODE_SPEED_PLAN.md §4.1)
        # -----------------------------------------------------------------
        # Every layer evicts on the SAME step, at shapes that come from config
        # and W rather than from data, so all 32 evictions are one call over a
        # wider row axis. Folding L into the row axis (R = L*B, layer i owning
        # rows [i*B, (i+1)*B)) is enough: every op the eviction performs is
        # per-row, so the body is unchanged and only the row count grows. That
        # takes the eviction from 273 launching ops x 32 layers = 8,736 per
        # eviction to 273 — the single largest launch cut available, and the
        # eviction is ~64% of the amortized per-token launch budget.
        #
        # PHASES. Prefill and the first eviction run PER LAYER, exactly as
        # before: joining before the first compaction would require the prompt
        # to be resident in one L*B-row tensor, which raises the prefill peak by
        # a whole extra copy of the steady-state cache — and peak memory is the
        # number this method exists to lower. So the join happens immediately
        # AFTER the first eviction, when every per-layer prompt buffer has
        # already been released and the copy peaks at two steady-state stores
        # instead. Layer-major prefill is a separate, gated change (see
        # _PREFILL_LAYER_MAJOR).
        #
        # ORDERING. From the join onward, eviction moves from "inside layer i's
        # update, after that layer appended this step's token" to "once, before
        # any layer appends". It has to: when layer 0's update runs at step t,
        # layers 1..L-1 have not produced token t yet, so the only state all L
        # layers share is the state as of the end of step t-1. The retained set
        # is unchanged — compute_two_tier_retain reads only window_scores, whose
        # width and values are the same either way, and this step's token has no
        # score column and sits in the never-evictable local region — so the
        # cache at the END of each step is identical; only "append then compact"
        # becomes "compact then append". tests/test_layer_major_evict.py pins
        # that byte-for-byte against the per-layer path.
        self._layer_major: bool = True
        self._joint: Optional[CacheState] = None
        self._joint_store: Optional[QuantizedStore] = None
        self._joint_step: int = 0
        self._joint_write_at: int = 0
        self._layers_opened: int = 0
        self._evicted_this_step: int = 0
        self._steady_capacity: int = steady
        # The joint Q-tier read, memoized on the joint store's version exactly as
        # QuantizedStore.effective_q_tier memoizes the per-layer one. Only the
        # materialize path uses it; the fused path never dequantizes.
        self._joint_qtier_memo: Optional[Any] = None
        # [L*B, n_q*ws] frozen Q-tier positions, rebuilt with the fused context.
        self._joint_qpos: Optional[Tensor] = None

        # Absolute tokens appended so far (host int), and a one-shot flag for the
        # cache_position contract check below. See _check_position_contract.
        self._tokens_seen: int = 0
        self._pos_contract_checked: bool = False
        # One-shot flag for the derived-position warning. See
        # _resolve_cache_position: some attention modules (Mistral's FA2 path)
        # omit cache_position from cache_kwargs, so it is derived from
        # _tokens_seen rather than left to state.append's compacted-length
        # fallback.
        self._derived_pos_warned: bool = False

    # -----------------------------------------------------------------
    # HF Cache interface
    # -----------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the **effective** sequence length for *layer_idx*.

        At ``q > 0`` this is ``T_fp + T_q`` (design §5): HF uses it to size the
        causal mask over the returned effective K/V, and the flash hook uses it
        as the key count. It is a key-count report only — query positioning
        still follows HF's monotonic absolute positions (no override). At
        ``q == 0`` the Q tier is empty, so this is exactly ``state.seq_length``.
        """
        t_fp = self._states[layer_idx].seq_length
        store = self._stores[layer_idx]
        if store is None:
            return t_fp
        return t_fp + store.num_active_tokens

    def get_max_length(self) -> Optional[int]:
        """Return ``None`` — windowed cache doesn't have a static max."""
        return None

    def _resolve_memoization(self, batch_size: int) -> None:
        """Settle the Q-tier read memo once the real batch size is known.

        ``quant_memoize_read=None`` (the default) means: **on at B=1, off above.**
        The memo holds the whole Q tier dequantized to fp16 per layer for the
        whole decode — ~149 MB/row at the qasper steady state against ~125 MB/row
        of actual two-tier KV. At B=1 that is invisible (decode is weight-bound —
        BATCHING_PLAN.md §5) and it saves 7 of every 8 steps' dequant. At B>1 it
        is charged per row and halves the batch that fits, which is the one thing
        the whole method exists to raise. An explicit True/False always wins.
        """
        if self._memoization_resolved:
            return
        self._memoization_resolved = True
        pref = self.resolved.quant_memoize_read
        memo = (batch_size == 1) if pref is None else pref
        # A LIVE gate means no memo, full stop; config rejects an explicit True.
        # The memo is keyed on store.version -- which only moves at eviction --
        # while the gate's selected set moves every step, so it would serve a
        # set the gate did not choose.
        #
        # `quant_read_gate=False` makes the memo legitimate again -- ungated
        # there is no per-step selected set -- so the auto rule above turns it
        # back on at B=1, saving 7 of every 8 steps' whole-tier dequant.
        #
        # Scope, because this is easy to over-read: the memo covers
        # `effective_q_tier`, which is reached only through `_joint_qtier_read`
        # -> `_materialize_joint`. On CUDA the fused two-tier kernel takes raw
        # int2 and dequantizes inside, so that path is not taken and this line
        # changes nothing about a GPU decode step. It is the eager/materialize
        # path -- CPU, and the eager backend -- that gets the 8x back.
        if self.resolved.quant_sketch_enabled:
            memo = False
        for store in self._stores:
            if store is not None:
                store.memoize_read = memo

    def _check_position_contract(
        self, cache_kwargs: Optional[Dict[str, Any]]
    ) -> None:
        """Verify once that ``cache_position`` is the ABSOLUTE token index.

        An evicting cache breaks an assumption transformers makes. When
        ``cache_position`` is not supplied, ``LlamaModel.forward`` derives it as
        ``arange(past_key_values.get_seq_length(), ...)`` — i.e. it treats "how
        many keys do you hold" as "how many tokens have been processed". For a
        cache that never evicts those are the same number. For this one they
        diverge the moment the budget binds, and ``get_seq_length()`` becomes a
        sawtooth: it drops by ``window_size - 1`` at every eviction. The step
        after an eviction is then told it sits ~``ws`` tokens *earlier* than the
        step before it, so the query's RoPE phase goes backward and the appended
        K/V is stored under a position that already exists in the store.

        ``generate()`` is immune — it advances ``cache_position`` itself
        (``_update_model_kwargs_for_generation``: ``cache_position[-1:] +
        num_new_tokens``) and never re-derives it — which is why the quality
        runners were never affected. A hand-written decode loop that omits it is
        not immune, which is what this check exists to catch.

        **Cost: nothing until it matters.** The outer test is a host-int
        comparison of two values the cache already tracks, so it is free on every
        step where the retained key count still equals the absolute index — i.e.
        before the first eviction. Only once they diverge does it read one
        element of ``cache_position`` (a single sync, once per cache lifetime),
        and then it never runs again.

        The contract is absolute-from-zero: this cache serves one prompt followed
        by single-token decode (``_update_joint`` refuses anything else), so a
        base other than ``_tokens_seen`` is a bug, not a calling convention.
        """
        if self._pos_contract_checked:
            return
        # Host ints only. Before the first eviction these are equal and there is
        # nothing a derived cache_position could get wrong.
        if self.get_seq_length(0) >= self._tokens_seen:
            return
        self._pos_contract_checked = True
        pos = (cache_kwargs or {}).get("cache_position")
        if pos is None or not torch.is_tensor(pos) or pos.numel() == 0:
            return
        first = int(pos.reshape(-1)[0])          # one sync, once per cache
        if first != self._tokens_seen:
            raise RuntimeError(
                f"cache_position starts at {first} but {self._tokens_seen} tokens "
                f"have been appended; the cache currently retains "
                f"{self.get_seq_length(0)} keys. "
                "This is transformers deriving cache_position from "
                "get_seq_length() because the caller did not pass it. For an "
                "evicting cache that is the RETAINED KEY COUNT, not the absolute "
                "token index, so positions go backward after every eviction and "
                "the store ends up with duplicated, non-monotonic positions. "
                "Fix: pass cache_position explicitly and advance it monotonically "
                "(model(..., cache_position=torch.arange(pos, pos + n_new))), "
                "which is exactly what generate() does. Every quality runner here "
                "goes through generate() and is unaffected; a hand-written decode "
                "loop must do it itself."
            )

    def _resolve_cache_position(
        self,
        cache_kwargs: Optional[Dict[str, Any]],
        n_new: int,
        device: Any,
    ) -> Optional[Tensor]:
        """Absolute positions for the ``n_new`` tokens appended this step.

        The caller's ``cache_position`` when it supplies one; otherwise positions
        derived from this cache's own monotonic token count.

        Why derive rather than let :meth:`CacheState.append` auto-increment:
        ``append(pos=None)`` counts up from the store's CURRENT length, which is
        the *compacted* length after an eviction. Most attention modules pass
        ``cache_position`` through ``cache_kwargs`` (every Llama path, and
        Mistral's eager/sdpa paths), so this branch never runs for them and their
        positions are byte-identical to before. ``MistralFlashAttention2.forward``
        on transformers 4.47.x builds ``cache_kwargs = {"sin": sin, "cos": cos}``
        and omits ``cache_position`` — so without this the first post-eviction
        token would be filed ~``window_size`` positions early, its surviving-key
        RoPE would go backward, and its window id ``(pos - num_sink) //
        window_size`` would collide with a survivor's, scrambling the eviction's
        grouping. ``_tokens_seen`` is the absolute index and is advanced once per
        step (at ``layer_idx == 0``, before this runs), so ``_tokens_seen -
        n_new`` is the correct start for every layer of the step, identical to the
        ``cache_position`` HF passes whenever it does.
        """
        if cache_kwargs is not None:
            pos = cache_kwargs.get("cache_position")
            if pos is not None:
                return pos
        start = self._tokens_seen - n_new
        if not self._derived_pos_warned:
            self._derived_pos_warned = True
            warnings.warn(
                "cache_position was not supplied through cache_kwargs; deriving "
                f"absolute positions from the cache's own token count "
                f"(start={start}, n={n_new}). This is expected for "
                "MistralFlashAttention2, which omits cache_position — the derived "
                "positions match what generate() advances, so surviving keys keep "
                "their original RoPE after an eviction. If you see this on a "
                "hand-written decode loop, pass cache_position explicitly instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.arange(
            start, start + n_new, device=device, dtype=torch.long
        )

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Append new KV states and optionally evict.

        Two phases, switched once and never switched back (§4.1, and the
        ``_joint`` block in :meth:`__init__`):

        * **before the first eviction** — :meth:`_update_per_layer`, this cache's
          original body: prefill and step 0, per layer, each layer owning its own
          prompt-sized buffer.
        * **after it** — :meth:`_update_joint`: one layer-major store, one
          batched eviction for all ``L`` layers, and each layer's ``update``
          reduced to writing its own rows.

        The switch happens in :meth:`_migrate_to_joint`, on the first ``update``
        of the step **after** every layer has evicted for the first time — not at
        the end of that eviction step, because the tensors this method already
        returned for that step (and the fused hand-off built from them) must stay
        valid until the model has finished with them.
        """
        if layer_idx == 0:
            self._check_position_contract(cache_kwargs)
            self._tokens_seen += key_states.shape[2]
        if (self._joint is None and self._layer_major
                and self._evicted_this_step > 0 and layer_idx == 0):
            if self._evicted_this_step != self.num_layers:
                raise RuntimeError(
                    f"{self._evicted_this_step} of {self.num_layers} layers "
                    "evicted on the first eviction step. Every layer shares one "
                    "schedule and one budget, so a partial eviction means some "
                    "layer reached update() unscored — the batched decode path "
                    "cannot be built on layers that are not in lockstep, and "
                    "silently staying per-layer would hide the unscored layer."
                )
            self._migrate_to_joint()
        if self._joint is not None:
            return self._update_joint(
                key_states, value_states, layer_idx, cache_kwargs
            )
        return self._update_per_layer(
            key_states, value_states, layer_idx, cache_kwargs
        )

    def _update_per_layer(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Per-layer append + evict — prefill and the first eviction.

        Steps:
        1. ``state.append(k, v, pos)``
        2. Pull pre-computed window scores from *cache_kwargs*.
        3. Accumulate into ``state.window_scores``.
        4. If ``policy.should_evict(step)``:
           a. Two-step retain: window indices → token indices.
           b. ``state.slice_and_keep`` (surviving keys keep their original RoPE).
           c. Optionally ``state.rerotate_keys`` when ``rerotate_on_evict``.
           d. Gather ``state.window_scores`` by retained window indices.
        5. Return ``(state.key_states, state.value_states)``.
        """
        state = self._states[layer_idx]
        policy = self._policies[layer_idx]

        self._resolve_memoization(key_states.shape[0])

        # Positions for the appended tokens. cache_position when the caller
        # supplies it (every Llama path; Mistral eager/sdpa), otherwise derived
        # from the monotonic token count so a caller that omits it (Mistral FA2)
        # does not fall through to append()'s compacted-length auto-increment.
        n_new = key_states.shape[2]
        pos = self._resolve_cache_position(cache_kwargs, n_new, key_states.device)

        # 1. Append
        state.append(key_states, value_states, pos)
        policy.extend_total_after_append(n_new)

        # Detect prefill vs generation
        is_prefill = not self._prefill_done[layer_idx]
        if is_prefill:
            policy.initialize_after_prefill(state.seq_length)
            self._prefill_done[layer_idx] = True

        # 2. Pull pre-computed window scores
        merged_kwargs = {}
        if cache_kwargs is not None:
            merged_kwargs.update(cache_kwargs)
        merged_kwargs.update(self.cache_kwargs.get(layer_idx, {}))
        new_window_scores = merged_kwargs.get("window_scores")

        # Clear consumed scores so a hook that silently returns on the next step
        # does not cause stale values to be re-accumulated.
        layer_kwargs = self.cache_kwargs.get(layer_idx)
        if layer_kwargs is not None and "window_scores" in layer_kwargs:
            del layer_kwargs["window_scores"]

        # 3. Initialize or accumulate window_scores
        if new_window_scores is not None:
            if state.window_scores is None:
                state.window_scores = new_window_scores.clone()
                # Initialize identity mapping: compact index i == original window i.
                # Stored per row ([B, W]) so divergent per-row eviction keeps each
                # row's surviving window identities independently.
                W = new_window_scores.shape[-1]
                B_w = new_window_scores.shape[0]
                state.original_window_ids = (
                    torch.arange(W, device=new_window_scores.device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(B_w, -1)
                    .contiguous()
                )
                self._next_original_window_id[layer_idx] = W
            else:
                # Handle size mismatch: new scores may cover more windows
                W_old = state.window_scores.shape[-1]
                W_new = new_window_scores.shape[-1]
                if W_new > W_old:
                    pad = torch.zeros(
                        state.window_scores.shape[0],
                        state.window_scores.shape[1],
                        W_new - W_old,
                        device=state.window_scores.device,
                        dtype=state.window_scores.dtype,
                    )
                    state.window_scores = torch.cat(
                        [state.window_scores, pad], dim=-1
                    )
                    # Extend original_window_ids for new windows using the
                    # running original-sequence counter, not compact-space
                    # indices (which would collide with surviving IDs).
                    if state.original_window_ids is not None:
                        n_extra = W_new - W_old
                        start_id = self._next_original_window_id[layer_idx]
                        B_w = state.original_window_ids.shape[0]
                        extra = (
                            torch.arange(
                                start_id, start_id + n_extra,
                                device=state.original_window_ids.device,
                                dtype=torch.long,
                            )
                            .unsqueeze(0)
                            .expand(B_w, -1)
                        )
                        state.original_window_ids = torch.cat(
                            [state.original_window_ids, extra], dim=1
                        )
                        self._next_original_window_id[layer_idx] = start_id + n_extra
                elif W_new < W_old:
                    # Symmetric pad on the incoming scores so in-place += works
                    # without changing accumulate's contract. No
                    # original_window_ids change: no new windows appeared.
                    pad = torch.zeros(
                        new_window_scores.shape[0],
                        new_window_scores.shape[1],
                        W_old - W_new,
                        device=new_window_scores.device,
                        dtype=new_window_scores.dtype,
                    )
                    new_window_scores = torch.cat(
                        [new_window_scores, pad], dim=-1
                    )
                accumulate(state.window_scores, new_window_scores)

        # 4. Eviction
        step = self._generation_step[layer_idx]
        should_evict = not is_prefill and policy.should_evict(step)

        if should_evict and state.window_scores is not None and self._q > 0.0:
            # Two-tier eviction (design §5). B == 1 in v1 (guarded at append).
            self._evict_two_tier(layer_idx, step)
            self._evicted_this_step += 1
        elif should_evict and state.window_scores is not None:
            B = state.key_states.shape[0]
            H_q = state.window_scores.shape[1]

            # a. Two-step retain
            retained_window_idx = policy.compute_retain_window_indices(
                state.window_scores
            )
            retain_token_idx = policy.expand_to_token_indices(
                retained_window_idx, state.window_scores.shape[2]
            )

            # Telemetry
            self.telemetry.record_scores(
                layer_idx, step, state.window_scores, retain_token_idx
            )

            # b. Snapshot old positions before compaction (only needed when
            #    re-rotating; scoped to the flag to avoid confusion).
            #    position_ids is [B, T]; gather per row so each row's snapshot
            #    matches the tokens it actually keeps.
            old_positions = (
                torch.gather(
                    state.position_ids, 1,
                    retain_token_idx.to(state.position_ids.device),
                ).clone()
                if self.resolved.rerotate_on_evict
                else None
            )

            # c. Compact K/V. Surviving keys keep their original RoPE rotation
            #    and position_ids are gathered to their original values.
            state.slice_and_keep(retain_token_idx)

            # d. Optionally re-rotate keys to contiguous positions
            #    (StreamingLLM-style). OFF by default: HF generate advances the
            #    query's cache_position monotonically (it does not re-derive it
            #    from get_seq_length each step on transformers <= 4.47), so
            #    re-rotating keys to contiguous positions while the query stays
            #    at its original absolute position corrupts the RoPE relative
            #    phase after the first eviction. Keeping original positions
            #    matches KVPress / H2O and is correct on any version.
            if self.resolved.rerotate_on_evict:
                state.rerotate_keys(self.rope_module, old_positions)

            # e. Gather window_scores by retained_window_idx
            idx_w = retained_window_idx.unsqueeze(1).expand(B, H_q, -1)
            state.window_scores = torch.gather(
                state.window_scores, dim=-1, index=idx_w
            ).contiguous()

            # f. Keep original_window_ids in sync with the surviving windows.
            #    Gather per row ([B, W]) because rows may retain different windows.
            if state.original_window_ids is not None:
                state.original_window_ids = torch.gather(
                    state.original_window_ids, 1,
                    retained_window_idx.to(state.original_window_ids.device),
                ).contiguous()

            # Update policy
            policy.set_total_after_compaction(state.seq_length)
            self._evicted_this_step += 1

        # Advance generation step (only after prefill is done)
        if not is_prefill:
            self._generation_step[layer_idx] = step + 1

        # 5. Return. At q > 0 the return is the interleaved effective K/V so
        #    attention (and the eager scorer) see both tiers; at q == 0 this is
        #    the live fp store, byte-identical to the single-tier cache.
        if self._q > 0.0:
            store = self._stores[layer_idx]
            # Fused decode path: on a real decode step with a non-empty Q tier and
            # the flash kernel active, hand the Q tier + scatter map to the
            # flash_attn_func patch and return the FP TIER for it to attend over —
            # no fp16 concat, and the kernel emits the eviction score. Everything
            # else (prefill, empty Q tier, eager backend, fused disabled) falls
            # through to the materialize path below, unchanged.
            if (self._fused_decode_active and not is_prefill
                    and store is not None and store.num_active_windows > 0):
                from . import flash_decode
                n = store.num_active_windows
                B = state.key_states.shape[0]
                ws = self.resolved.window_size
                # Both hand-offs are MEMOIZED (see _fused_qtier / _fused_meta):
                # everything the kernel needs except this step's query is frozen
                # between evictions, so rebuilding it per layer per step was pure
                # launch overhead. Identical tensors either way — same values, not
                # merely equivalent ones.
                qtier = self._fused_qtier(layer_idx, store, B, n, ws)
                score_meta = self._fused_meta(layer_idx, state, ws)
                flash_decode.set_pending({
                    "layer_idx": layer_idx,
                    "qtier": qtier,
                    "score_meta": score_meta,
                    # The gate needs `q`, which update() never sees; the patch
                    # selects with it in hand. See flash_decode._run_fused.
                    "gate": self._fused_ctx[layer_idx]["gate"],
                    "num_sink": self.resolved.num_sink_tokens,
                    "window_size": ws,
                    "scaling": self._attn_scaling,
                    "cache": self,
                })
                # The patch writes window_scores; the score forward-hook skips this
                # step. Clear the materialize stashes so no stale effective-K leaks.
                self._last_effective_k[layer_idx] = None
                self._last_score_meta[layer_idx] = None
                return state.key_states, state.value_states

            eff_k, eff_v, score_meta = self._materialize(layer_idx)
            # Hand the effective K + its score-scatter map to the score hook
            # (fix #2) so it need not rebuild them. Overwritten each step; the
            # hook consumes (clears) them.
            self._last_effective_k[layer_idx] = eff_k
            self._last_score_meta[layer_idx] = score_meta
            return eff_k, eff_v
        return state.key_states, state.value_states

    # -----------------------------------------------------------------
    # Layer-major decode (DECODE_SPEED_PLAN.md §4.1)
    # -----------------------------------------------------------------

    def _migrate_to_joint(self) -> None:
        """Fold the ``L`` per-layer stores into one whose row axis is ``L*B``.

        Runs once, on the first ``update`` after every layer has evicted. At that
        point each layer's prompt-sized buffer has already been released by
        ``CacheState.replace`` (which reallocates down to the steady capacity at
        the first compaction), so the copy peaks at two steady-state stores
        rather than at the prompt — which is why prefill is left per-layer.

        After this, ``_states[i]`` / ``_stores[i]`` become read-only views onto
        the joint objects so the score hooks, the parity runner and the tests
        keep reading a per-layer shape.
        """
        # Every layer must live on ONE device: the join concatenates all L layers
        # into a single row axis, and `device_map="auto"` shards a model across
        # GPUs by LAYER, so on a multi-GPU box layers 0..k sit on cuda:0 and the
        # rest on cuda:1. Concatenating across that is not merely slow, it is
        # impossible -- it raised
        # "Expected all tensors to be on the same device, cuda:0 and cuda:1"
        # on the first multi-GPU perf run, and under torch.compile the same
        # mismatch surfaced instead as a device-side assert, which poisons the
        # CUDA context and makes every later cell in the sweep fail too.
        #
        # Refuse, rather than fall back silently. This is a PERF path, so quietly
        # running the per-layer eviction would report a regression under the
        # layer-major label and there would be nothing in the output to say why.
        # Both escapes are named because both are legitimate: one GPU is what
        # run_perf_table.sh already defaults to, and the env flag restores the
        # byte-identical per-layer path (tests/test_layer_major_evict.py pins
        # that equivalence, so nothing is lost but speed).
        devices = {st.key_states.device for st in self._states}
        if len(devices) > 1:
            raise RuntimeError(
                "layer-major decode needs every layer's KV on one device, but "
                f"this model is sharded across {sorted(str(d) for d in devices)} "
                "(device_map=\"auto\" splits by layer). Either pin the model to a "
                "single GPU (CUDA_VISIBLE_DEVICES=0, which scripts/run_perf_table.sh "
                "already defaults to) or set STICKYKV_LAYER_MAJOR_DECODE=0 to run "
                "the per-layer eviction, which produces byte-identical results at "
                "the pre-optimisation launch count. Per-device grouping is the "
                "proper fix and is not implemented yet -- see DECODE_SPEED_PLAN.md."
            )
        B = self._states[0].key_states.shape[0]
        self._joint = CacheState.join_layers(self._states, self._steady_capacity)
        if self._q > 0.0:
            self._joint_store = QuantizedStore.join_layers(self._stores)
        # Per-layer handles become views. The originals are dropped here, which
        # is what releases the L separate steady buffers the join copied from.
        self._states = [
            _LayerStateView(self._joint, i * B, B) for i in range(self.num_layers)
        ]
        if self._joint_store is not None:
            self._stores = [
                _LayerStoreView(self._joint_store, i * B, B)
                for i in range(self.num_layers)
            ]
        # The memoized fused hand-offs were built against the per-layer stores.
        self._fused_ctx = [None] * self.num_layers
        self._joint_qtier_memo = None
        # Every layer advanced its own counter identically; adopt it as the one
        # global step, and keep the per-layer list in sync for the readers that
        # still index it (ours_parity_runner, demo_generate).
        self._joint_step = self._generation_step[0]
        self._layers_opened = 0
        self._evicted_this_step = 0

    def _begin_step(self, n_new: int, position_ids: Optional[Tensor]) -> None:
        """Open one decode step for ALL layers at once — the §4.1 batched pass.

        Runs exactly once per step, from the first layer's ``update``, and does
        in three calls what the per-layer path did in ``3 * L``:

        1. accumulate every layer's pending window scores (one ``cat``, one add),
        2. evict every layer in ONE batched pass when the schedule says so,
        3. reserve this step's token slot and write its positions.

        Step 2 is the reorder §4.1 rests on: it runs **before** any layer has
        appended this step's token, because that token does not exist yet for
        layers 1..L-1. The retained set is unaffected — the tier assignment reads
        only ``window_scores``, and this step's token has no score column — so the
        cache at the end of the step is identical to appending first and
        compacting after.
        """
        step = self._joint_step
        joint = self._joint

        # A dequantized Q tier is only carried between steps when the store's own
        # read memo is on; at its B>1 default (off) it is dropped here so the
        # materialize path never holds one longer than the step that built it.
        if self._joint_store is not None and not self._joint_store.memoize_read:
            self._joint_qtier_memo = None

        # 1. Scores. The hooks wrote one [B, H_q, W] per layer during the
        #    previous step's forward; concatenating in layer order lands each
        #    layer on exactly the rows join_layers gave it.
        pending = []
        missing = []
        for i in range(self.num_layers):
            lk = self.cache_kwargs.get(i)
            ws_i = lk.pop("window_scores", None) if lk is not None else None
            pending.append(ws_i)
            if ws_i is None:
                missing.append(i)
        if len(missing) < self.num_layers:
            if missing:
                raise RuntimeError(
                    f"layers {missing} produced no window scores for step "
                    f"{step} while the others did. Scoring is per layer per "
                    "step and kernel-or-error; a partial step means one layer's "
                    "score hook silently returned, and accumulating the rest "
                    "would time an unscored layer under this method's name."
                )
            self._check_score_width(pending[0].shape[-1])
            self._accumulate_joint_scores(torch.cat(pending, dim=0))

        # 2. Eviction — one pass over L*B rows.
        if (self._policies[0].should_evict(step)
                and joint.window_scores is not None):
            self._evict_joint(step)

        # 3. This step's token slot, for every layer at once. Each layer's K/V
        #    lands later, in its own update(), through write_rows.
        self._joint_write_at = joint.reserve(n_new, position_ids)
        for policy in self._policies:
            policy.extend_total_after_append(n_new)

        self._joint_step = step + 1
        for i in range(self.num_layers):
            self._generation_step[i] = self._joint_step

    def _merged_window_count(self) -> int:
        """Merged (fp body + Q) window count of the joint store as it stands."""
        ws = self.resolved.window_size
        body = max(self._joint.seq_length - self.resolved.num_sink_tokens, 0)
        n_q = (
            self._joint_store.num_active_windows
            if self._joint_store is not None else 0
        )
        return -(-body // ws) + n_q

    def _check_score_width(self, width: int) -> None:
        """The incoming score axis must describe the store the eviction will see.

        Under the batched ordering the eviction runs **before** this step's token
        is appended, so the scores it consumes and the store it compacts describe
        the same key set — which is exactly what a score pass over the previous
        step's attention produces, since that attention ran after that step's
        append. The two agreeing is not an accident, it is the contract.

        Violating it is silent and catastrophic rather than merely wrong: a score
        axis one window wider than the store makes ``n_ev_fp_cur`` over-count, so
        the rebuild's ``searchsorted`` resolves a window id that is not in the
        body, clamps to the end, and lands the same token in both tiers. Cheaper
        to refuse than to debug.
        """
        want = self._merged_window_count()
        if width != want:
            raise RuntimeError(
                f"window scores span {width} merged windows but the store spans "
                f"{want} (fp body {self._joint.seq_length} tokens, "
                f"{self._merged_window_count() - width} window(s) apart). The "
                "batched eviction compacts BEFORE this step's token is appended, "
                "so scores must describe the store as it stands now — i.e. the "
                "key set the previous step's attention saw. A width sized for "
                "the post-append store belongs to the per-layer ordering "
                "(STICKYKV_LAYER_MAJOR_DECODE=0)."
            )

    def _accumulate_joint_scores(self, new_window_scores: Tensor) -> None:
        """Accumulate ``[L*B, H_q, W_new]`` into the joint running scores.

        The layer-major counterpart of step 3 of :meth:`_update_per_layer`, with
        the same width-reconciliation: a wider incoming tensor extends the
        running scores and mints new original window ids, a narrower one is
        right-padded so the in-place add keeps ``accumulate``'s contract. All
        ``L`` layers share one width (they append the same tokens on the same
        steps), so this runs once instead of ``L`` times.
        """
        joint = self._joint
        if joint.window_scores is None:
            joint.window_scores = new_window_scores.clone()
            W = new_window_scores.shape[-1]
            R = new_window_scores.shape[0]
            joint.original_window_ids = (
                torch.arange(W, device=new_window_scores.device, dtype=torch.long)
                .unsqueeze(0).expand(R, -1).contiguous()
            )
            for i in range(self.num_layers):
                self._next_original_window_id[i] = W
            return

        W_old = joint.window_scores.shape[-1]
        W_new = new_window_scores.shape[-1]
        if W_new > W_old:
            pad = torch.zeros(
                joint.window_scores.shape[0], joint.window_scores.shape[1],
                W_new - W_old,
                device=joint.window_scores.device,
                dtype=joint.window_scores.dtype,
            )
            joint.window_scores = torch.cat([joint.window_scores, pad], dim=-1)
            if joint.original_window_ids is not None:
                n_extra = W_new - W_old
                start_id = self._next_original_window_id[0]
                R = joint.original_window_ids.shape[0]
                extra = (
                    torch.arange(
                        start_id, start_id + n_extra,
                        device=joint.original_window_ids.device, dtype=torch.long,
                    ).unsqueeze(0).expand(R, -1)
                )
                joint.original_window_ids = torch.cat(
                    [joint.original_window_ids, extra], dim=1
                )
                for i in range(self.num_layers):
                    self._next_original_window_id[i] = start_id + n_extra
        elif W_new < W_old:
            pad = torch.zeros(
                new_window_scores.shape[0], new_window_scores.shape[1],
                W_old - W_new,
                device=new_window_scores.device, dtype=new_window_scores.dtype,
            )
            new_window_scores = torch.cat([new_window_scores, pad], dim=-1)
        accumulate(joint.window_scores, new_window_scores)

    def _evict_joint(self, step: int) -> None:
        """One eviction for every layer. ``layer_idx=None`` selects the joint state.

        At ``q > 0`` this is :meth:`_evict_two_tier` over ``L*B`` rows — the same
        body, the same compiled path, ``L`` times fewer launches. At ``q == 0`` it
        is the single-tier retain, likewise widened.
        """
        if self._q > 0.0:
            self._evict_two_tier(None, step)
            self._joint_qtier_memo = None
            for policy in self._policies:
                policy.set_total_after_compaction(
                    self._joint.seq_length + self._joint_store.num_active_tokens
                )
            return

        joint = self._joint
        policy = self._policies[0]
        R = joint.key_states.shape[0]
        H_q = joint.window_scores.shape[1]
        retained_window_idx = policy.compute_retain_window_indices(
            joint.window_scores
        )
        retain_token_idx = policy.expand_to_token_indices(
            retained_window_idx, joint.window_scores.shape[2]
        )
        self._record_joint_scores(step, joint.window_scores, retain_token_idx)
        old_positions = (
            torch.gather(
                joint.position_ids, 1,
                retain_token_idx.to(joint.position_ids.device),
            ).clone()
            if self.resolved.rerotate_on_evict
            else None
        )
        joint.slice_and_keep(retain_token_idx)
        if self.resolved.rerotate_on_evict:
            joint.rerotate_keys(self.rope_module, old_positions)
        idx_w = retained_window_idx.unsqueeze(1).expand(R, H_q, -1)
        joint.window_scores = torch.gather(
            joint.window_scores, dim=-1, index=idx_w
        ).contiguous()
        if joint.original_window_ids is not None:
            joint.original_window_ids = torch.gather(
                joint.original_window_ids, 1,
                retained_window_idx.to(joint.original_window_ids.device),
            ).contiguous()
        for p in self._policies:
            p.set_total_after_compaction(joint.seq_length)

    def _record_joint_scores(
        self, step: int, window_scores: Tensor, retained: Tensor
    ) -> None:
        """Split the joint row axis back into layers for the telemetry recorder.

        Telemetry is per layer by contract and the joint tensors are ``L*B``-row,
        so the split has to happen somewhere. It happens here, and only when a
        real recorder is installed: the perf suite runs ``NullTelemetry`` and must
        not pay ``L`` slices per eviction for a no-op.
        """
        if isinstance(self.telemetry, NullTelemetry):
            return
        B = window_scores.shape[0] // self.num_layers
        for i in range(self.num_layers):
            r0 = i * B
            self.telemetry.record_scores(
                i, step, window_scores[r0:r0 + B], retained[r0:r0 + B]
            )

    def _update_joint(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """One layer's part of a layer-major decode step.

        Everything shared — scores, eviction, the length advance, the position
        write — happened once in :meth:`_begin_step`. What is left per layer is
        writing this layer's K/V into its own rows and building its return value.
        """
        if key_states.shape[2] != 1 and self._joint.seq_length > 0:
            # A multi-token update after the join means a second prompt, which
            # this cache does not model: the joint store's capacity, the window
            # schedule and the score width are all sized for one prompt followed
            # by single-token decode.
            raise RuntimeError(
                f"layer {layer_idx} appended {key_states.shape[2]} tokens after "
                "the layer-major decode store was built; only single-token "
                "decode steps are defined past the first eviction."
            )
        B = key_states.shape[0]
        if self._layers_opened >= self.num_layers:
            self._layers_opened = 0

        # Scores handed in with the call, rather than pre-written to
        # cache.cache_kwargs[layer_idx] by a score hook. The batched step reads
        # every layer's pending scores in one pass at the START of the step, so
        # an inline score can only be consumed in the same step it arrived if the
        # step has not been opened yet — i.e. on the first update() of the step.
        # It is stashed where _begin_step looks, which is exactly what the hooks
        # do; past that point it would be picked up a step late, and a silent
        # one-step shift in the eviction's evidence is precisely the kind of
        # change this path must not make quietly.
        inline_scores = (
            cache_kwargs.get("window_scores") if cache_kwargs is not None else None
        )
        if inline_scores is not None:
            if self._layers_opened != 0:
                raise RuntimeError(
                    f"layer {layer_idx} passed window_scores through update()'s "
                    "cache_kwargs after this step was already opened by an "
                    "earlier layer. The batched decode path collects every "
                    "layer's scores once per step, before any layer appends, so "
                    "these would be accumulated one step late. Write them to "
                    "cache.cache_kwargs[layer_idx]['window_scores'] during the "
                    "previous step instead — that is what both score hooks do."
                )
            self.cache_kwargs.setdefault(layer_idx, {})["window_scores"] = (
                inline_scores
            )

        if self._layers_opened == 0:
            # Same position resolution as the per-layer path: derive from the
            # monotonic token count when the caller omits cache_position (Mistral
            # FA2), so reserve() does not auto-increment from the compacted length.
            pos = self._resolve_cache_position(
                cache_kwargs, key_states.shape[2], key_states.device
            )
            self._begin_step(key_states.shape[2], pos)
        self._layers_opened += 1

        joint = self._joint
        r0 = layer_idx * B
        joint.write_rows(r0, B, key_states, value_states, self._joint_write_at)

        k_layer = joint.key_states[r0:r0 + B]
        v_layer = joint.value_states[r0:r0 + B]
        if self._q <= 0.0:
            return k_layer, v_layer

        store = self._joint_store
        ws = self.resolved.window_size
        if self._fused_decode_active and store.num_active_windows > 0:
            from . import flash_decode
            n = store.num_active_windows
            qtier = self._fused_qtier_joint(layer_idx, store, B, n, ws)
            score_meta = self._fused_meta_joint(layer_idx, B, ws)
            flash_decode.set_pending({
                "layer_idx": layer_idx,
                "qtier": qtier,
                "score_meta": score_meta,
                "gate": self._fused_ctx[layer_idx]["gate"],
                "num_sink": self.resolved.num_sink_tokens,
                "window_size": ws,
                "scaling": self._attn_scaling,
                "cache": self,
            })
            self._last_effective_k[layer_idx] = None
            self._last_score_meta[layer_idx] = None
            return k_layer, v_layer

        eff_k, eff_v, score_meta = self._materialize_joint(layer_idx, B)
        self._last_effective_k[layer_idx] = eff_k
        self._last_score_meta[layer_idx] = score_meta
        return eff_k, eff_v

    def _build_joint_fused_ctx(self, store, B: int, n: int, ws: int) -> None:
        """Rebuild EVERY layer's fused hand-off in one pass.

        The per-layer :meth:`_fused_qtier` costs ~21 launches per layer to gather
        the active slots and derive the RoPE halves; layer-major it is one gather
        and one rotary call for all ``L``, then ``L`` free views. The memo
        contract is unchanged — keyed on the store's version, which only moves at
        eviction — so this fires once per eviction cycle, not once per step.

        The split back to per-layer dicts is exact: ``QuantSlotTable.gather``
        flattens ``[R, n]`` row-major and ``R = L*B`` is laid out layer-major, so
        the flat axis is ``(layer, row, slot)`` and reshaping to ``[L, B, n, …]``
        puts each layer's slots where :meth:`_migrate_to_joint` put its rows.
        """
        from .decode_kernel import rope_cos_sin_halves

        L = self.num_layers
        idx = store.table.active_order(n)                       # [R, n] slots
        kc, ks, kz, vc, vs, vz, qpos = store.table.gather(idx)  # [R*n, …]

        def by_layer(t: Tensor) -> Tensor:
            return t.reshape(L, B, n, *t.shape[1:])

        kc, vc = by_layer(kc), by_layer(vc)
        ks, kz, vs, vz = (g.map(by_layer) for g in (ks, kz, vs, vz))
        qpos_flat = qpos.reshape(L * B, n * ws)                 # [R, n*ws]
        # The KV store's dtype, NOT fp32: every other RoPE in this cache runs at
        # the store dtype (HF's own, and unrotate/rotate_key_window), so an fp32
        # table here would re-rotate the Q tier with a different convention than
        # the one it was un-rotated with. See rope_cos_sin_halves.
        cos_h, sin_h = rope_cos_sin_halves(
            self.rope_module, qpos_flat, self._kv_dtype)
        self._joint_qpos = qpos_flat

        # The gate's cards ride the same one-gather-for-all-L trick as the codes:
        # `gather_sketch` keeps the [R, n] leading pair, and R is layer-major, so
        # layer i's rows are the same [r0, r0+B) slice everything else here uses.
        joint_gate = self._gate_ctx(store, idx, n)

        key = (store.version, n, B)
        for i in range(L):
            r0 = i * B
            # `.map` and not `[i]`: a QGrid is a NamedTuple, so indexing it with
            # an int takes its first FIELD, not its first layer.
            self._fused_ctx[i] = {
                "qkey": key,
                "qtier": {
                    "k_codes": kc[i], "k_scale": ks.map(lambda t: t[i]),
                    "k_zero": kz.map(lambda t: t[i]),
                    "v_codes": vc[i], "v_scale": vs.map(lambda t: t[i]),
                    "v_zero": vz.map(lambda t: t[i]),
                    "cos": cos_h[r0:r0 + B], "sin": sin_h[r0:r0 + B],
                    "window_size": ws,
                },
                "qpos": qpos_flat[r0:r0 + B],
                "mkey": None, "score_meta": None,
                "gate": None if joint_gate is None else {
                    "card": tuple(t[r0:r0 + B] for t in joint_gate["card"]),
                    "anchor": joint_gate["anchor"][r0:r0 + B],
                    "centroid": tuple(t[r0:r0 + B]
                                      for t in joint_gate["centroid"]),
                    "n_sel": joint_gate["n_sel"],
                },
            }

    def _fused_qtier_joint(self, layer_idx: int, store, B: int, n: int, ws: int):
        """Layer ``layer_idx``'s Q-tier context, off the layer-major rebuild."""
        slot = self._fused_ctx[layer_idx]
        key = (store.version, n, B)
        if slot is None or slot["qkey"] != key:
            self._build_joint_fused_ctx(store, B, n, ws)
            slot = self._fused_ctx[layer_idx]
        return slot["qtier"]

    def _fused_meta_joint(self, layer_idx: int, B: int, ws: int):
        """Layer ``layer_idx``'s ``(order, q_token_len)`` scatter map.

        Same window-epoch memo as :meth:`_fused_meta` — ``compute_score_meta``
        reads the fp body's positions at a ``ws`` stride, so its result depends on
        the body only through how many whole windows it spans — and likewise
        rebuilt for all ``L`` layers at once, since the joint positions are one
        tensor.
        """
        slot = self._fused_ctx[layer_idx]
        num_sink = self.resolved.num_sink_tokens
        n_body = max(self._joint.position_ids.shape[1] - num_sink, 0)
        n_body_win = -(-n_body // ws)            # ceil
        key = (slot["qkey"], n_body_win)
        if slot["mkey"] == key:
            return slot["score_meta"]

        from modules.quant.effective import compute_score_meta

        order, q_token_len = compute_score_meta(
            self._joint.position_ids, self._joint_qpos, num_sink, ws,
        )
        for i in range(self.num_layers):
            s = self._fused_ctx[i]
            s["mkey"] = (s["qkey"], n_body_win)
            r0 = i * B
            s["score_meta"] = (order[r0:r0 + B], q_token_len)
        return slot["score_meta"]

    def _joint_qtier_read(self):
        """The joint dequantized + RoPE'd Q tier, for the materialize path.

        One call for all ``L`` layers instead of ``L``. Held across steps only
        when the store's own read memo is on (``quant_memoize_read``, default:
        on at B=1, off above) — :meth:`_begin_step` drops it otherwise, so the
        B>1 default still never carries a dequantized tier between steps.
        """
        store = self._joint_store
        memo = self._joint_qtier_memo
        if memo is not None and memo[0] == store.version:
            return memo[1]
        res = store.effective_q_tier(
            self.rope_module, self._joint.key_states.dtype
        )
        self._joint_qtier_memo = (store.version, res)
        return res

    def _materialize_joint(self, layer_idx: int, B: int):
        """``[sink ‖ body ‖ Q]`` effective K/V for one layer of the joint store."""
        joint = self._joint
        store = self._joint_store
        r0 = layer_idx * B
        k_layer = joint.key_states[r0:r0 + B]
        v_layer = joint.value_states[r0:r0 + B]
        if store is None or store.num_active_windows == 0:
            return k_layer, v_layer, None
        q_k, q_v, q_pos = self._joint_qtier_read()
        return materialize_effective_kv(
            k_layer, v_layer, joint.position_ids[r0:r0 + B], store,
            num_sink=self.resolved.num_sink_tokens,
            window_size=self.resolved.window_size,
            rope_module=self.rope_module,
            out_dtype=joint.key_states.dtype,
            q_tier=(q_k[r0:r0 + B], q_v[r0:r0 + B], q_pos[r0:r0 + B]),
        )

    # -----------------------------------------------------------------
    # Fused-decode hand-off, memoized (design §10 — entries are write-once)
    # -----------------------------------------------------------------

    def _fused_qtier(self, layer_idx: int, store, B: int, n: int, ws: int) -> Dict:
        """The kernel's Q-tier context — gathered int2 fields + RoPE halves.

        **Rebuilt only when the Q tier changes.** design §10 makes every entry
        write-once and moves the active set only at eviction, so between
        evictions the gathered codes/scales/zeros, the frozen positions, and the
        ``cos``/``sin`` those positions produce are all constant — the same
        memoization contract ``QuantizedStore.effective_q_tier`` already uses for
        the materialize path, keyed on the same ``store.version`` (bumped by
        every mutator: retain_only / reactivate_many / demote_many / promote_many).

        Rebuilding it per layer per step was ~21 kernel launches of pure repeat
        work — 8 for the gather, 10 for the rotary module, 3 for the ``argsort``
        that orders the slots. Measured over one eviction cycle, memoizing takes a
        non-eviction ``update()`` from 30 launches to 5; at 32 layers that is ~800
        fewer launches per generated token, and at these shapes decode is bound by
        launch count, not by the kernel those launches feed.

        The returned tensors are read-only to the kernel and to
        :mod:`flash_decode`, so handing out the same objects across steps is safe.
        """
        slot = self._fused_ctx[layer_idx]
        key = (store.version, n, B)
        if slot is not None and slot["qkey"] == key:
            return slot["qtier"]

        from .decode_kernel import rope_cos_sin_halves

        # Gather the active Q windows' RAW int2 fields (no dequant — the kernel
        # unpacks + dequants + RoPEs in registers), plus the RoPE halves
        # (position-only) the kernel applies in registers.
        idx = store.table.active_order(n)                     # [B, n] slots
        kc, ks, kz, vc, vs, vz, qpos = store.table.gather(idx)

        def by_row(t: Tensor) -> Tensor:
            return t.reshape(B, n, *t.shape[1:])

        kc, vc = by_row(kc), by_row(vc)
        ks, kz, vs, vz = (g.map(by_row) for g in (ks, kz, vs, vz))
        qpos_flat = qpos.reshape(B, n * ws)
        # The KV store's dtype, NOT fp32: every other RoPE in this cache runs at
        # the store dtype (HF's own, and unrotate/rotate_key_window), so an fp32
        # table here would re-rotate the Q tier with a different convention than
        # the one it was un-rotated with. See rope_cos_sin_halves.
        cos_h, sin_h = rope_cos_sin_halves(
            self.rope_module, qpos_flat, self._kv_dtype)
        qtier = {
            "k_codes": kc, "k_scale": ks, "k_zero": kz,
            "v_codes": vc, "v_scale": vs, "v_zero": vz,
            "cos": cos_h, "sin": sin_h, "window_size": ws,
        }
        # A new Q tier invalidates the scatter map built against the old one.
        self._fused_ctx[layer_idx] = {
            "qkey": key, "qtier": qtier, "qpos": qpos_flat,
            "mkey": None, "score_meta": None,
            "gate": self._gate_ctx(store, idx, n),
        }
        return qtier

    def _gate_ctx(self, store, idx: Tensor, n: int) -> Optional[Dict]:
        """The read gate's hand-off — the cards, the anchor, and how many to keep.

        ``None`` only when the store has **no cards** — ``quant_sketch_enabled``
        off, or nothing demoted yet. That is a capability fact, not a setting:
        there is nothing to select on.

        The ratio does **not** get its own path. ``1.0`` used to return ``None``
        and skip the gate entirely, which meant the shipped config and the
        control arm ran different code and a reader had to know which. Now every
        ratio takes the same route and ``1.0`` simply selects every window — the
        no-op proof, on the one implementation, rather than a second one.

        Memoized with the tier for the same reason the tier is (design §10): cards
        are written once when a window is sealed and reactivated unchanged on
        re-demotion, so between evictions this gather is pure repeat work.

        ``n_sel`` is resolved against the **live** ``n_active`` rather than pinned,
        because ``N_q`` moves with the shape and a fixed count would not be the
        same fraction anywhere.
        """
        ratio = self.resolved.quant_gate_ratio
        if not (store.sketch_enabled and store._anchor is not None
                and getattr(store.table, "sketch", False)):
            return None
        card = store.table.gather_sketch(idx)
        # The centroid columns ride the same gather, so the windows this step
        # skips cost no second pass over the tier to attend through. `vm_q` /
        # `vm_s` are SKETCH_FIELDS[-2:]; the kernel wants them contiguous and
        # separate from the scan fields, which the gather already gives.
        return {
            "card": card,
            "anchor": store._anchor,
            "centroid": (card[-2], card[-1], store._v_anchor),
            "n_sel": max(1, math.ceil(ratio * n)),
        }

    def _fused_meta(self, layer_idx: int, state, ws: int):
        """The ``(order, q_token_len)`` scatter map — memoized per window epoch.

        :func:`~modules.quant.effective.compute_score_meta` reads the fp body's
        positions at a ``ws`` stride, so its result depends on the fp store only
        through **how many whole windows the body spans**, not through its exact
        token count. Between evictions ``position_ids`` is append-only (it is a
        ``_pos_buf[:, :length]`` view), so a body of the same window count is the
        same values at the same strided indices — and the ``argsort`` over
        per-row-unique window ids has no ties, so the permutation is unique and
        the cached one is the one a recompute would produce.

        Net: rebuilt once per ``ws`` decode steps instead of every step, per layer.
        """
        slot = self._fused_ctx[layer_idx]
        num_sink = self.resolved.num_sink_tokens
        n_body = max(state.position_ids.shape[1] - num_sink, 0)
        n_body_win = -(-n_body // ws)            # ceil — windows the body spans
        key = (slot["qkey"], n_body_win)
        if slot["mkey"] == key:
            return slot["score_meta"]

        from modules.quant.effective import compute_score_meta

        meta = compute_score_meta(
            state.position_ids, slot["qpos"], num_sink, ws,
        )
        slot["mkey"] = key
        slot["score_meta"] = meta
        return meta

    # -----------------------------------------------------------------
    # Two-tier read path + eviction (design §5, §8) — q > 0, B = 1
    # -----------------------------------------------------------------

    def _materialize(self, layer_idx: int) -> Tuple[Tensor, Tensor, Any]:
        """Build the ``[sink ‖ body ‖ Q]`` effective K/V for one layer.

        Returns ``(eff_k, eff_v, score_meta)`` — freshly-built tensors (callers
        must not assume they alias the stored fp cache, design §9) plus the
        score-scatter map the hook needs, or ``score_meta=None`` at an empty Q
        tier (the fp store, byte-identical).

        Routes to :meth:`_materialize_joint` once the layer-major decode store
        exists: past that point ``_stores[i]`` is a read-only view with no slot
        table of its own, so the per-layer body below cannot run. The score
        hook's "update() didn't run for this layer" fallback calls this, so it
        has to stay correct in both phases.
        """
        if self._joint is not None:
            B = self._joint.key_states.shape[0] // self.num_layers
            return self._materialize_joint(layer_idx, B)
        state = self._states[layer_idx]
        store = self._stores[layer_idx]
        if store is None or store.num_active_windows == 0:
            return state.key_states, state.value_states, None

        return materialize_effective_kv(
            state.key_states,
            state.value_states,
            state.position_ids,
            store,
            num_sink=self.resolved.num_sink_tokens,
            window_size=self.resolved.window_size,
            rope_module=self.rope_module,
            out_dtype=state.key_states.dtype,
        )

    def _record_evict_widths(self, fresh: Tensor, promote: Tensor,
                             react: Tensor, n_q: int, W: int) -> None:
        """How many lanes each eviction actually used, against the width it got.

        ``_compact`` allocates by **rank into a worst-case width**, not by count:
        the counts are ragged per row (*"row 0 may demote 4 windows while row 1
        demotes none"*), so allocating by count would need the max, and the max
        needs a host sync. the design notesD1 proposes paying that sync -- one
        per eviction -- to stop un-rotating, quantizing and sketching a
        ``[L*B, n_q, H_kv, ws, D]`` tensor of which most lanes are masked away.

        **D1's size is entirely this ratio, and nothing has ever printed it.**
        DECODE_NEXT.md §7 lists it as the risk the estimate rests on: *"If the
        real count is routinely close to n_q, D1 is worth little."* So measure
        before rewriting, especially now that the eviction actually fuses (the
        waste is pointwise work inside a fused kernel, not sixteen launches of
        it, which is a different and un-sized quantity).

        **This costs no sync and no graph break**, which is the whole reason it
        is a running max in device memory rather than the ``.item()`` D1 wants.
        A ``.item()`` here reintroduces exactly the break
        ``tests/test_evict_graph_breaks.py`` pins at zero -- measured:
        ``Unsupported Tensor.item() call with capture_scalar_outputs=False``.
        The buffer hangs off ``self``, which is argument 0 of the compiled body,
        so Inductor functionalizes the mutation like any other input mutation.

        Read it with :func:`evict_width_stats`, which syncs once, on demand,
        outside whatever window was being timed.
        """
        buf = getattr(self, "_evict_widths", None)
        if buf is None or buf.device != fresh.device:
            buf = torch.zeros(6, dtype=torch.long, device=fresh.device)
            self._evict_widths = buf
        f = fresh.sum(1).max()
        p = promote.sum(1).max()
        r = react.sum(1).max()
        buf[0] += 1
        buf[1] = torch.maximum(buf[1], f)
        buf[2] = torch.maximum(buf[2], p)
        buf[3] = torch.maximum(buf[3], r)
        buf[4] += f                      # running sums -> a mean, not just a max
        buf[5] += p
        # n_q and W are Python ints (tier_counts is config arithmetic), so they
        # are trace-time constants -- recorded on the host, never in the graph.
        self._evict_width_bound = (int(n_q), int(W))

    @staticmethod
    def _compact(src: Tensor, mask: Tensor, width: int) -> Tuple[Tensor, Tensor]:
        """Gather ``src``'s masked lanes to the front of a ``[B, width]`` tensor.

        The counterpart to ``BATCHING_PLAN.md §3``'s rectangularity result. The
        *retained* counts are equal across rows, but the **transition** counts
        are not: ``demote = is_q_new & ~is_q_cur`` depends on where each row's Q
        tier already is, so row 0 may demote 4 windows while row 1 demotes none.
        Allocating by *count* would therefore need a host sync to learn the max;
        allocating by **rank** into a worst-case-width tensor does not.

        ``width`` must be a proven upper bound on ``mask.sum(1)`` — the caller
        takes it from the budget resolver. Lanes beyond a row's count are
        ``valid=False`` and carry ``-1``; overflow lanes (if the bound were ever
        wrong) are routed to a dump column rather than scattering out of bounds,
        so a bad bound drops work loudly in tests instead of corrupting memory.

        Returns ``(compacted [B, width], valid [B, width])``.
        """
        B = mask.shape[0]
        rank = mask.cumsum(1) - 1                                   # [B, W]
        # `width` as a SCALAR other, not a full_like tensor of it: the tensor was
        # a [B, W] int64 allocation and fill per call, three times an eviction,
        # to be read by one `where`. And no `clamp_min(0)` on the true branch --
        # a lane the condition selects has `mask` true, so its `rank` is already
        # >= 0; the -1 lanes it was guarding take `width` regardless.
        idx = torch.where(mask & (rank < width), rank, width)
        out = torch.full((B, width + 1), -1, dtype=src.dtype, device=src.device)
        out.scatter_(1, idx, src)
        valid = torch.zeros((B, width + 1), dtype=torch.bool, device=src.device)
        valid.scatter_(1, idx, mask)
        return out[:, :width], valid[:, :width]

    def _evict_targets(self, layer_idx: Optional[int]):
        """``(state, store, policy)`` for an eviction.

        ``layer_idx=None`` selects the layer-major store — one call covering every
        layer's rows (DECODE_SPEED_PLAN §4.1), which is the decode default from
        the first eviction onward. An int selects that layer's own objects, which
        is prefill and the first eviction.
        """
        if layer_idx is None:
            return self._joint, self._joint_store, self._policies[0]
        return (
            self._states[layer_idx],
            self._stores[layer_idx],
            self._policies[layer_idx],
        )

    def _evict_two_tier(self, layer_idx: Optional[int], step: int) -> None:
        """Dispatch one two-tier eviction to the eager or torch.compile'd body.

        ``layer_idx=None`` is the layer-major eviction: identical body, ``L*B``
        rows instead of ``B``, so all ``L`` layers' 273 launching ops collapse
        into one set of 273 (§4.1).

        The body always runs through a lazily-built, process-global
        ``torch.compile`` (``dynamic=True`` for the varying window count), which
        fuses the eviction step's ~360 pointwise / gather / scatter / quantize
        launches into a handful of kernels. That step is ~89% of the fused-decode
        path's per-token launch budget (the other ``ws-1`` steps are just the KV
        append), so it is the one place left where launches can be cut without
        touching the eviction math — and compilation is semantics-preserving by
        construction, so the retained windows, tier moves, and rebuilt fp store
        are identical to the eager path. It fires once per ``ws`` steps, so a
        compile / recompile amortizes.

        **Kernel-or-error — it never falls back to eager.** A silent (or even
        loud) eager run under a "compiled" label would report eager numbers, and
        the ~81%-of-budget eviction step would look cut when it was not. So if
        ``torch.compile`` cannot lower the body,
        :func:`_run_compiled_evict` raises (:data:`_EVICT_COMPILE_FAILED` records
        the reason for provenance) rather than running eager — the perf runner
        then marks that cell ``errored``, loudly, instead of quietly mislabelling
        eager timings.

        The torch <= 2.6 Inductor failures this used to hit — a symbolic
        ``((I)//ws)`` guard "not comparable" and an ``aten.amin`` StarDep from a
        single-sided clamp — are fixed at the source in :meth:`_evict_two_tier_impl`
        (the shape-derived control ints are made concrete, and the clamp is
        two-sided), so on a supported build it now compiles rather than raising.
        If a build still cannot lower it, the raised error names the failing op

        **``layer_idx`` never crosses into the compiled body, and that is
        load-bearing** (2026-09-19). It used to: the body opened with
        ``self._evict_targets(layer_idx)``, which indexes ``self._states`` --  a
        PYTHON LIST -- so Dynamo specialised on the integer's value and built one
        graph per layer. The first eviction runs per layer, 32 times, so it blew
        ``cache_size_limit`` at layer 8, and Dynamo's response to that limit is to
        run the frame EAGER. Every eviction after it, including all 39 joint ones
        that actually matter, ran eager under a "compiled" label:

            torch._dynamo hit config.cache_size_limit (8)
              function: '_evict_two_tier_impl'
              last reason: 0/0: L['layer_idx'] == 0   # self._states[layer_idx]

        ``_EVICT_STATS`` could not see it -- it counts calls to the compiled
        CALLABLE, and the callable was called; it was Dynamo underneath that
        declined. A profile caught it as "3 compiled / 0 eager runs, Inductor
        kernels 0.000 ms/step", i.e. the eviction's ~360 pointwise / gather /
        scatter / quantize launches were never fused at all.

        So everything that needs the layer's identity happens HERE, outside the
        graph: resolving the targets, dropping the stale fused hand-offs,
        installing the compacted body, and recording telemetry. What the body
        gets is the three objects and nothing else.

        **The ``joint`` bool went the same way, on 2026-09-19.** It selected
        ``replace_body`` versus ``replace``, and both of those are now the
        caller's -- so the body no longer branches on it at all, and dropping it
        removes a specialisation axis: one graph per shape instead of two.
        """
        state, store, policy = self._evict_targets(layer_idx)

        # Eviction rewrites the Q tier AND compacts the fp store's positions, so
        # every memoized fused hand-off for this layer is stale. `store.version`
        # already covers the Q half (`retain_only` inside the body bumps it); this
        # covers the fp half without relying on that ordering. A layer-major
        # eviction rewrites every layer, so it drops every entry. Hoisted out of
        # the body because indexing `self._fused_ctx` by `layer_idx` specialises
        # the graph exactly as `_evict_targets` did.
        if layer_idx is None:
            self._fused_ctx = [None] * self.num_layers
        else:
            self._fused_ctx[layer_idx] = None

        # Compiled-or-raise. No eager fallback by design.
        joint = layer_idx is None
        (scores_before, retained_idx,
         new_k, new_v, new_pos, n_q) = _run_compiled_evict(
            self, state, store, policy, step)

        # --- install the compacted body, OUTSIDE the graph -------------------
        # This is deliberately not in the compiled body: writing a buffer that
        # the same graph reads a view of is what made it uncompilable (the
        # `merge_view_inputs` IndexError -- see `_evict_two_tier_impl` step 4b
        # and `_EVICT_COMPILE_HELP`). Three slice-assignments gain nothing from
        # being traced.
        num_sink = self.resolved.num_sink_tokens
        if joint:
            # Layer-major: the buffer is already at its steady capacity (the
            # per-layer first eviction is what sized it), so the compacted body
            # goes straight in behind the sink prefix, which the rebuild carried
            # through byte-identically at the same offsets. That skips the three
            # full-store `cat`s below — ~2.9 GB of transient at 4096/batch-32
            # once the row axis carries all 32 layers — and the reallocation
            # `replace` performs whenever the target size moves.
            state.replace_body(num_sink, new_k, new_v, new_pos)
        else:
            # Per-layer: this IS the first eviction, where `replace`
            # reallocating from the prompt capacity down to the budget is the
            # point — it is what releases the prompt-sized buffer.
            # `join`, not `torch.cat`: same tensor, but a cat is lowered through
            # Inductor's `pointwise_cat` on CUDA, the single source of the
            # `Identity` node behind the `((I)//8)` failure. Kept here even out
            # of the graph so the two branches stay the same operation.
            state.replace(
                join(state.key_states[:, :, :num_sink, :], new_k, 2),
                join(state.value_states[:, :, :num_sink, :], new_v, 2),
                join(state.position_ids[:, :num_sink], new_pos, 1),
            )

        # Only now is `state.seq_length` the compacted length.
        store.commit_active_count(n_q)
        policy.set_total_after_compaction(
            state.seq_length + store.num_active_tokens
        )

        # Telemetry is per layer by contract and lives outside the graph for the
        # same reason as the rest of this method. It is also the one consumer of
        # the PRE-eviction score axis, which step 5 of the body overwrites -- so
        # the body hands the tensor back rather than recording it.
        if not isinstance(self.telemetry, NullTelemetry):
            if joint:
                self._record_joint_scores(step, scores_before, retained_idx)
            else:
                self.telemetry.record_scores(
                    layer_idx, step, scores_before, retained_idx
                )
        return None

    def _evict_two_tier_impl(
        self, state, store, policy
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """One two-tier eviction (design §5), per row, entirely on device.

        **This is the compiled body, and its whole parameter list is chosen so
        Dynamo builds ONE graph for it.** It takes the resolved
        ``(state, store, policy)`` and nothing else: no ``layer_idx`` (it would
        index a Python list), no ``step`` (an integer that increments every
        eviction is one Dynamo specialises on), and since 2026-09-19 no ``joint``
        bool either, because the branch it selected now runs in the caller.

        **It is also pure with respect to the fp buffers**, which is what makes
        it compilable at all -- see step 4b. It may mutate the slot table, whose
        fields are separate allocations, but it never writes a buffer it has read
        a view of.

        Returns ``(scores_before, retained_idx, new_k, new_v, new_pos, n_q)``.
        The first two are the caller's telemetry input, returned rather than
        recorded here because step 5 overwrites ``state.window_scores``. The next
        three are the compacted body for the caller to install. ``n_q`` is the
        active-window count to commit once it has.

        **The row axis is opaque here**, which is what makes the layer-major
        decode path (§4.1) a change of caller rather than of body: every
        operation below is per row — ``gather``/``scatter`` on dim 1, ``argsort``
        and ``cumsum`` along dim 1, ``searchsorted`` row-wise — and every width is
        a host int from :meth:`EvictionPolicy.tier_counts`. So passing
        ``layer_idx=None`` (rows = ``L*B``, layer ``i`` owning
        ``[i*B, (i+1)*B)``) evicts all ``L`` layers with exactly the arithmetic
        one layer used, at ``1/L`` the launches.

        Ranks the merged window axis, assigns tiers, moves boundary-crossers
        (demote K→Q / promote Q→K), rebuilds the fp store as
        ``[sink ‖ fp windows in chronological id order]`` keeping every survivor's
        original absolute positions, and updates the Q store. Positions are never
        renumbered.

        **No Python loops and no host syncs.** The tier bookkeeping used to be
        host-side Python sets over ``.tolist()``ed window ids — which is both a
        pipeline stall per layer per eviction and, more fundamentally, unable to
        carry a batch axis at all: rows evict divergently, so "the set of Q
        windows" is per row. Every decision below is ``[B, W]`` mask algebra
        against the slot table instead, and every tensor width comes from
        :meth:`EvictionPolicy.tier_counts` — host ints that are identical across
        rows (BATCHING_PLAN.md §3), so nothing has to be measured off a tensor.
        """
        # KERNEL-OR-ERROR, enforced where it can actually be seen. The eviction
        # is "compiled unconditionally", but being CALLED through a compiled
        # callable is not the same as being COMPILED: when Dynamo gives up on a
        # frame -- recompile limit, an unsupported construct, a disabled cache --
        # it runs the Python eagerly and says so only in a warning log. That is
        # precisely what happened between 2026-09-18 and 2026-09-19 (see
        # `_evict_two_tier`), and `_EVICT_STATS` reported "compiled" throughout.
        #
        # `torch.compiler.is_compiling()` is the one check that cannot be fooled,
        # because it is answered by whoever is executing this line. Under Dynamo
        # it is a trace-time constant, so the branch folds away and NOTHING of
        # this reaches the graph; under an eager fallback it is False and the run
        # stops. One bool test per eviction, i.e. one per `ws` steps.
        if not torch.compiler.is_compiling():
            raise EvictionRanEager(
                "the two-tier eviction ran EAGER. It is compiled-or-error by "
                "design: the compiled body exists to fuse ~360 pointwise / "
                "gather / scatter / quantize launches into a handful of kernels, "
                "and an eager run reports eager numbers under a compiled label.\n"
                "\nWHAT DYNAMO SAYS, RIGHT NOW:\n" + _dynamo_diagnosis() + "\n"
                "\nDynamo fell back without raising, which it does for exactly "
                "three reasons:\n"
                "  1. the recompile limit -- look for 'torch._dynamo hit "
                "config.cache_size_limit' in the log, then find what is "
                "specialising the graph (a Python-list index or a changing int "
                "argument in this signature is the usual cause) and raise "
                "_EVICT_CACHE_SIZE_LIMIT only after that;\n"
                "  2. compilation disabled process-wide -- TORCHDYNAMO_DISABLE, "
                "torch._dynamo.config.disable, or a torch._dynamo.disable() "
                "decorator somewhere up the stack;\n"
                "  3. an unsupported construct Dynamo declined to trace -- look "
                "for a graph-break warning naming it.\n"
                "There is no knob that makes an eager eviction acceptable: it "
                "reports eager numbers under a compiled label, which is the one "
                "failure this path exists to refuse."
            )

        rope = self.rope_module

        ws = self.resolved.window_size
        num_sink = self.resolved.num_sink_tokens
        dtype = state.key_states.dtype
        device = state.key_states.device

        # CONCRETE ints for the integer index arithmetic below — arange sizes,
        # window ranks, clamp bounds, and the ``//ws`` token→window map. Off a
        # tensor ``.shape`` these are Python ints in eager but SymInts under
        # torch.compile, and the eviction's gather/scatter index math then emits
        # symbolic integer expressions Inductor cannot lower on torch <= 2.6:
        # ``((I)//8) is not comparable`` (the ``i // ws`` guard) and the
        # ``aten.amin`` StarDep from a symbolic single-sided clamp. ``int()``
        # forces specialization here — a no-op in eager, byte-identical — so every
        # count below is a real int and the index math is concrete. The cache
        # compacts to the budget every ``ws`` steps, so this specializes to a
        # handful of shapes during the fill phase and is stable at steady state
        # (which is why the compiled body is worth its recompiles — design §5).
        #
        # Read one dim at a time, NOT as ``B, H_kv, T_fp, D = ...shape``. An
        # unpack of the whole shape is the same symbolic read wearing a different
        # AST, and it is where the guard used to have a blind spot: the four
        # ``int()``s ``014c40b`` added all went on ``x = t.shape[i]`` forms while
        # the live offender — ``B, _, W = window_scores.shape`` in
        # ``policy.compute_two_tier_retain`` — was an unpack and sailed through.
        B = state.key_states.shape[0]
        H_kv = int(state.key_states.shape[1])
        T_fp = int(state.key_states.shape[2])
        D = int(state.key_states.shape[3])
        T_body = T_fp - num_sink
        W = int(state.window_scores.shape[2])
        # The memoized fused hand-offs this eviction invalidates are dropped by
        # `_evict_two_tier` before it calls here, because indexing `_fused_ctx`
        # by layer is exactly the kind of Python-list access that specialises the
        # graph.
        n_q_prev = store.num_active_windows

        # --- 1–2. Rank + tier assignment on the merged axis -----------------
        retained_idx, new_tier = policy.compute_two_tier_retain(state.window_scores)
        # W is a concrete int (above), so these are Python ints even under
        # torch.compile; the int() is belt-and-suspenders against a future
        # tier_counts that returns a tensor scalar.
        k_fp, n_q, local_w = policy.tier_counts(W)
        k_fp, n_q, local_w = int(k_fp), int(n_q), int(local_w)
        n_fp = k_fp + local_w

        # Telemetry's input, captured before step 5 rebinds `state.window_scores`
        # to the gathered axis. The recording itself is the caller's, which is
        # also what keeps `layer_idx` and `step` out of this graph. For the
        # two-tier path the retained indices are merged-WINDOW indices (not token
        # indices — the fp and Q survivors live in different stores).
        scores_before = state.window_scores

        wids = torch.gather(state.original_window_ids, 1, retained_idx)   # [B, W_ret]
        is_q_new = new_tier == 1

        # --- Resolve window ids → slots ONCE, then decide with masks --------
        store.ensure(B, device)
        has_entry, is_q_cur, slot_of, keep_slots = store.lookup(wids)

        demote = is_q_new & ~is_q_cur      # currently fp, wants Q
        fresh = demote & ~has_entry        # never quantized → quantize now
        react = demote & has_entry         # dormant → reactivate, NO requant (§10)
        promote = ~is_q_new & is_q_cur     # currently Q, wants fp

        self._record_evict_widths(fresh, promote, react, n_q, W)

        # --- 3a. Free dropped entries FIRST (§6) ----------------------------
        # Before allocating, not after: it is what makes n_slots_for's
        # top_k_fp + N_q bound exact rather than merely likely. Safe in either
        # order — retain_only only touches windows that are NOT retained, and
        # every demote/promote/reactivate target is retained by construction.
        store.retain_only(keep_slots)

        # --- fp window order: retained-fp windows, ascending id -------------
        # `wids` is already ascending (the merged axis is chronological), so
        # compacting the non-Q picks to the front keeps them ascending, and
        # everything the fp rebuild needs is gathered onto that same axis. This
        # used to push the Q picks to a sentinel and sort; the sentinel existed
        # only to express "these go last", which is what a partition says
        # directly, and the sort was the eviction's widest unfusable op.
        take = stable_partition(~is_q_new)[:, :n_fp]                   # [B, n_fp]
        fp_wids = torch.gather(wids, 1, take)
        fp_prom = torch.gather(promote, 1, take)
        fp_slot = torch.gather(slot_of, 1, take)
        prom_rank = fp_prom.cumsum(1) - 1                              # [B, n_fp]

        # --- D1, all three widths, ONE host read -----------------------------
        # `_compact` allocates by RANK into a worst-case width because the counts
        # are ragged per row, and everything past a row's real count is computed
        # and then masked away. Promote was still sized by its BOUND
        # (`min(N_q, n_fp)`), which under `quant_budget_mode: bytes` is ~250
        # windows -- so `promote_many` dequantized, RoPE'd and spliced ~250
        # windows per layer per eviction to use the handful that were really
        # promoted. That is the same waste D1 removed from demote, on a payload
        # that is larger, and it was the 7.25 GiB / 3.62 GiB the allocator
        # refused at 2048/batch-32.
        #
        # Read here rather than in two places: the three masks are stacked into
        # one `.item()`, so this is ONE host sync and ONE graph break for the
        # whole eviction, exactly as before -- and now the promote block sits
        # after the break with the demote block, in the same fused region.
        # The measured max over rows is a true upper bound by construction,
        # which is what `_compact` requires: a width that is ever too small
        # routes overflow to a dump column and silently drops work.
        n_p, n_r, n_d = _evict_widths(
            (fp_prom, react, fresh),
            (min(self.resolved.N_q, n_fp), n_q, n_q),
        )

        body_k = state.key_states[:, :, num_sink:, :]
        body_v = state.value_states[:, :, num_sink:, :]
        body_pos = state.position_ids[:, num_sink:]
        # Token → window id. Non-decreasing per row (the fp store is
        # [sink ‖ windows by ascending id] and generation appends the newest
        # tokens at the end), which is what lets searchsorted below resolve a
        # window to its run in one shot — no host span map, no sync.
        body_wid = ((body_pos - num_sink) // ws).contiguous()          # [B, T_body]

        # --- 3b. Promote (Q→fp): ONE batched dequant + ONE RoPE -------------
        # Width is the measured MAX over rows (n_p above), not the bound:
        # promotions are ragged per row, so lanes past a row's own count still
        # dequantize garbage from free slots and are dropped by the mask below —
        # bit-identical for the valid ones, since every op here is per-window or
        # per-token pointwise. What changed is how many such lanes there are.
        p_max = n_p              # D1: the MEASURED count, not min(N_q, n_fp)
        prom_slot, prom_valid = self._compact(fp_slot, fp_prom, p_max)
        prom_k = prom_v = prom_pos = None
        if p_max > 0:
            k_pre, v_pre, pos_pre = store.promote_many(prom_slot, prom_valid, dtype)
            prom_k = rotate_key_window(
                k_pre.permute(0, 2, 1, 3, 4).reshape(B, H_kv, p_max * ws, D),
                pos_pre.reshape(B, p_max * ws),
                rope,
            ).to(dtype)                                                # [B,H,p*ws,D]
            prom_v = v_pre.to(dtype).permute(0, 2, 1, 3, 4).reshape(
                B, H_kv, p_max * ws, D)
            prom_pos = pos_pre.reshape(B, p_max * ws).to(body_pos.dtype)

        # --- 3c. Demote (fp→Q): un-rotate + quantize the FRESH windows ------
        # Must read the OLD fp store, so it runs before the rebuild below.
        # Reactivations are free — codes/grid/positions are frozen (§10) — so
        # they are a bit flip and never touch this path.
        # Widths (n_r, n_d) were measured with n_p above -- one read for all
        # three, so the demote path costs no extra sync for its own bound.
        react_slot, react_valid = self._compact(slot_of, react, n_r)
        store.reactivate_many(react_slot, react_valid)

        fresh_wid, fresh_valid = self._compact(wids, fresh, n_d)
        if n_d > 0 and T_body > 0:
            start_f = torch.searchsorted(body_wid, fresh_wid.clamp_min(0))  # [B, n_d]
            tok_f = _clamp_index(
                (start_f.unsqueeze(-1) + torch.arange(ws, device=device))
                .reshape(B, n_d * ws),
                T_body - 1,
            )
            idx_d = tok_f.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, n_d * ws, D)
            k_post = torch.gather(body_k, 2, idx_d)
            v_tok = torch.gather(body_v, 2, idx_d)
            prange = torch.gather(body_pos, 1, tok_f)
            k_pre_d = unrotate_key_window(k_post, prange, rope)
            store.demote_many(
                store.table.free_slots(n_d), fresh_valid, fresh_wid,
                k_pre_d.reshape(B, H_kv, n_d, ws, D).permute(0, 2, 1, 3, 4),
                v_tok.reshape(B, H_kv, n_d, ws, D).permute(0, 2, 1, 3, 4),
                prange.reshape(B, n_d, ws),
                # The gate card is built from the ROTATED keys -- the tensor we
                # already hold, one line above un-rotating it -- so it costs no
                # extra RoPE, and positions are never rebased so it stays valid
                # for the window's whole life (§5, §10).
                keys_post_rope=(
                    k_post.reshape(B, H_kv, n_d, ws, D).permute(0, 2, 1, 3, 4)
                    if store.sketch_enabled else None
                ),
            )

        # --- 4. Rebuild the fp store: [sink ‖ fp windows by id] -------------
        # Every retained fp window contributes ws tokens except the newest, which
        # may be partial — and it is always last, because a window's id fixes its
        # position range and only the newest can straddle the end of the
        # sequence. So new-body token i belongs to fp window rank i // ws at
        # offset i % ws, and the length is pure config arithmetic: the retained
        # evictable windows are full, and the local tokens are whatever the body
        # holds beyond the evictable-fp windows currently in it.
        n_ev_fp_cur = W - local_w - n_q_prev            # evictable windows in fp now
        new_body_len = k_fp * ws + (T_body - n_ev_fp_cur * ws)
        i = torch.arange(new_body_len, device=device)
        # The newest kept window can run LONGER than ws. When T_body is not a whole
        # number of windows, the trailing partial window has no score column of its
        # own yet, so the retain step (which works on the W scored windows) never
        # counts it — physically it sits right after the newest retained window in
        # the chronological body. Addressing it as a phantom rank n_fp would gather
        # out of bounds (start_of_rank has only n_fp entries), so clamp the rank to
        # the last kept window and let its offset run past ws: those <= ws-1 tail
        # tokens fold into the newest local window, which we keep to the end anyway
        # (negligible vs budget), and get re-split into their own window on the next
        # scoring pass. When the body IS a whole number of kept windows this never
        # triggers — i // ws already tops out at n_fp - 1 — so it is a no-op on every
        # config that worked before.
        # Cap the token→window rank at the last kept window (phantom-tail
        # handling). Via _clamp_index so the upper bound never becomes an
        # ``aten.amin`` StarDep Inductor <= 2.6 cannot schedule.
        # ``torch.div(…, 'floor')`` is the explicit integer floor (a tensor op,
        # so no symbolic ``//`` scalar reaches the guard system); with ``n_fp``
        # now concrete the bound is a real int.
        rank = _clamp_index(torch.div(i, ws, rounding_mode="floor"), n_fp - 1)
        jj = rank.unsqueeze(0).expand(B, -1)                          # [B, T_new]
        off = (i - rank * ws).unsqueeze(0)                            # [1, T_new]

        start_of_rank = torch.searchsorted(body_wid, fp_wids)          # [B, n_fp]
        src_fp = _clamp_index(
            torch.gather(start_of_rank, 1, jj) + off, max(T_body - 1, 0))
        idx_fp = src_fp.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, new_body_len, D)
        new_k = torch.gather(body_k, 2, idx_fp)
        new_v = torch.gather(body_v, 2, idx_fp)
        new_pos = torch.gather(body_pos, 1, src_fp)

        if p_max > 0:
            # A promoted window has no tokens in the fp store — splice its
            # freshly-rotated copy in at its chronological slot (design §5 step
            # 3): same layout, different source.
            tok_is_prom = torch.gather(fp_prom, 1, jj)                 # [B, T_new]
            src_pr = _clamp_index(
                torch.gather(prom_rank, 1, jj) * ws + off, p_max * ws - 1)
            idx_pr = src_pr.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, new_body_len, D)
            sel = tok_is_prom.unsqueeze(1).unsqueeze(-1)
            new_k = torch.where(sel, torch.gather(prom_k, 2, idx_pr), new_k)
            new_v = torch.where(sel, torch.gather(prom_v, 2, idx_pr), new_v)
            new_pos = torch.where(
                tok_is_prom, torch.gather(prom_pos, 1, src_pr), new_pos
            )

        # --- 4b. The store write is the CALLER's, and that is the 2026-09-19
        # fix. It used to happen right here, and it is what made this graph
        # uncompilable:
        #
        #   `replace_body` does `self._key_buf[:, :, num_sink:n] = body_key`,
        #   mutating the BASE that `state.key_states` is a VIEW of -- and this
        #   region reads that view, as `body_k`. So Inductor lifted both the
        #   base and the view as graph inputs with the base mutated, which is
        #   exactly the shape AOTAutograd builds "synthetic bases" for, and
        #   `merge_view_inputs` mis-sizes its metadata list on it:
        #
        #       _aot_autograd/runtime_wrappers.py:1442 in merge_view_inputs
        #           post_processed_calling_convention_meta[k] = v
        #       IndexError: list assignment index out of range
        #
        # Three aliasing pairs, all mutated: key/value/positions. The `replace`
        # branch is worse -- it reads `state.key_states` AND reallocates the
        # buffer in the same expression.
        #
        # So the compiled region now ends as pure compute: it RETURNS the
        # compacted body and touches no buffer it has read a view of. Three
        # slice-assignments have nothing to gain from being in a graph anyway,
        # and `replace_body`'s own aliasing assertion is a host check that only
        # really runs out here.
        #
        # **The invariant this establishes, and the one to keep:** the compiled
        # eviction may mutate the slot table (separate allocations, no aliasing)
        # but must never write a buffer it reads a view of. Adding such a write
        # back into this region re-breaks the build.
        # --- 5. Gather scores + ids to the retained merged axis -------------
        # Only an `expand` size below, which tolerates a SymInt -- but `H_q` is
        # the model's query-head count and never varies within a run, so making
        # it concrete costs zero recompiles and keeps the "no raw scalar shape
        # read in this region" rule absolute rather than case-by-case. `B` is the
        # sole exception, and it is the one that would actually cost recompiles.
        H_q = int(state.window_scores.shape[1])
        idx_w = retained_idx.unsqueeze(1).expand(B, H_q, -1)
        state.window_scores = torch.gather(
            state.window_scores, dim=-1, index=idx_w
        ).contiguous()
        state.original_window_ids = torch.gather(
            state.original_window_ids, 1, retained_idx
        ).contiguous()

        # --- 6. Commit counts; policy total tracks T_fp + T_q ---------------
        # `commit_active_count` and `set_total_after_compaction` move out with
        # the write: the latter reads `state.seq_length`, which only becomes
        # correct once the caller has installed the compacted body.
        return scores_before, retained_idx, new_k, new_v, new_pos, n_q

    def reorder_cache(self, beam_idx: Tensor) -> None:
        """Beam search is out of scope (v1)."""
        raise NotImplementedError(
            "WindowedCache does not support beam search (reorder_cache). "
            "Use greedy or sampling decoding."
        )
