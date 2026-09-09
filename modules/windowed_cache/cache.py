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
against ``STICKYKV_LAYER_MAJOR_DECODE=0``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
from modules.quant.slots import QuantSlotTable, n_slots_for


# ---------------------------------------------------------------------------
# Optional torch.compile of the eviction step (STICKYKV_COMPILE_EVICT).
# Off by default: the eager _evict_two_tier_impl is the reference path and ships
# GPU-unvalidated compilation OFF, mirroring the decode read path's convention
# (modules/quant/effective.py). The compiled callable is process-global and
# built lazily on first use so a run that never evicts never pays for it.
# ---------------------------------------------------------------------------

_COMPILED_EVICT_FN = None
_EVICT_ANNOUNCED = {"done": False}
#: True once the dynamic=True compile has failed and we have retried static.
_EVICT_COMPILE_TRIED_STATIC = False

#: Proof-of-execution counters for the eviction path, same contract as
#: :data:`modules.windowed_cache.flash_decode._STATS`: being *able* to compile is
#: not the same as the compiled body actually running. ``compiled`` counts
#: evictions that went through ``_COMPILED_EVICT_FN``, ``eager`` the ones that
#: took the reference body. A benchmark that believes it enabled the compiled
#: eviction and reports ``eager > 0`` measured the path it was trying to replace.
_EVICT_STATS = {"eager": 0, "compiled": 0}

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
    (that is a property of the build, not of one measured run)."""
    _EVICT_STATS["eager"] = 0
    _EVICT_STATS["compiled"] = 0


def _compile_evict_enabled() -> bool:
    """Whether to run the eviction through ``torch.compile`` (default OFF)."""
    v = os.environ.get("STICKYKV_COMPILE_EVICT", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _compile_evict_backend() -> Optional[str]:
    """Optional ``torch.compile`` backend override for the eviction.

    ``None`` (unset) uses the default (Inductor), which is what fuses the
    launches on GPU. Set ``STICKYKV_COMPILE_EVICT_BACKEND`` to pick another —
    ``aot_eager`` traces + runs eager (no C++/Triton codegen, so it validates the
    tracing and semantics on a box without a compiler), ``cudagraphs`` for the
    graph backend, etc.
    """
    v = os.environ.get("STICKYKV_COMPILE_EVICT_BACKEND", "").strip()
    return v or None


def _layer_major_decode() -> bool:
    """Whether decode uses the layer-major store (default **ON**).

    This is DECODE_SPEED_PLAN §4.1 and it is the shipped path, not an experiment:
    one eviction for all ``L`` layers instead of ``L`` of them, which is a 32x cut
    on ~64% of the per-token launch budget.

    ``STICKYKV_LAYER_MAJOR_DECODE=0`` restores the per-layer eviction. It exists
    for two reasons and no others: it is the control arm of
    ``tests/test_layer_major_evict.py``, which asserts the two paths produce
    byte-identical caches, and it is how the same A/B is run on a GPU. It is not
    a fallback — nothing selects it automatically, and no failure degrades into
    it; a layer-major path that cannot run raises.
    """
    v = os.environ.get("STICKYKV_LAYER_MAJOR_DECODE", "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def _prefill_layer_major() -> bool:
    """Whether PREFILL also allocates layer-major (default OFF).

    Decode is layer-major unconditionally — that is DECODE_SPEED_PLAN §4.1 and
    it is the default path, not a flag. Prefill is not, and the reason is memory
    rather than taste: a layer-major prompt buffer means the first compaction has
    to hold the whole ``L*B``-row prompt AND the whole ``L*B``-row steady store at
    once, where per-layer buffers compact one at a time and only ever hold a
    single layer's steady store alongside the prompt. At 4096/batch-32 that is
    about +4 GB of peak — roughly 8% of the cell, taken straight out of max
    batch, which is the number the whole method exists to raise.

    So prefill stays per-layer and this knob exists to MEASURE the other side on
    a box where the peak has headroom. It changes allocation only; the eviction,
    the retained set and the cache contents are identical either way.
    """
    v = os.environ.get("STICKYKV_PREFILL_LAYER_MAJOR", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


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
        for field in ("key_codes", "key_scale", "key_zero",
                      "val_codes", "val_scale", "val_zero",
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


def _build_compiled_evict(dynamic: bool):
    """``torch.compile`` the eviction body. Lazy — never raises here; a lowering
    failure surfaces on the first CALL, not at construction."""
    kw: Dict[str, Any] = {"dynamic": dynamic}
    backend = _compile_evict_backend()
    if backend is not None:
        kw["backend"] = backend
    return torch.compile(WindowedCache._evict_two_tier_impl, **kw)


_EVICT_COMPILE_HELP = """The compiled eviction is KERNEL-OR-ERROR: it never silently runs the eager
body under a compiled label -- the whole point of the flag is to MEASURE the
compiled path, and a quiet eager run would report eager numbers as if compiled.
So this raises rather than falling back.
  What this is: torch.compile / Inductor could not lower
WindowedCache._evict_two_tier_impl on this build. The known torch<=2.6 causes --
a symbolic `((I)//ws)` guard and an `aten.amin` StarDep from a single-sided
clamp -- are addressed in the body (concrete shape ints + a two-sided clamp). If
it still fails, the traceback names the op that did not lower.
  To get numbers now: rerun with STICKYKV_COMPILE_EVICT=0 (or --compile-evict 0).
That runs the EAGER eviction -- correct, just launch-bound -- so the ~81%
eviction launch budget is not cut and TPOT stays high.
  To diagnose the lowering: the FULL Inductor traceback (which names the exact
aten op / source line) is in evict_compile_traceback() and is written by the perf
runner to <output_dir>/evict_compile_error_*.txt -- read that first. Also
TORCH_LOGS='+inductor,graph_breaks' TORCHDYNAMO_VERBOSE=1 on the failing shape,
and try STICKYKV_COMPILE_EVICT_BACKEND=aot_eager (traces + runs without codegen --
if THAT works the failure is Inductor codegen, not the traced graph)."""


def _run_compiled_evict(cache, layer_idx: int, step: int) -> None:
    """Run one eviction through ``torch.compile``, or RAISE. Kernel-or-error: no
    eager fallback (a silent eager run under a compiled label defeats the whole
    measurement -- the design intent). On success ``cache`` is mutated in place.

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
            "STICKYKV_COMPILE_EVICT is set but torch.compile of the two-tier "
            "eviction already failed on this build "
            f"({_EVICT_COMPILE_FAILED}).\n" + _EVICT_COMPILE_HELP
        )
    if _COMPILED_EVICT_FN is None:
        _COMPILED_EVICT_FN = _build_compiled_evict(dynamic=True)
    try:
        _COMPILED_EVICT_FN(cache, layer_idx, step)
        _announce_evict_path_once(compiled=True)
        _EVICT_STATS["compiled"] += 1
        return
    except Exception as e_dyn:  # pragma: no cover - GPU/Inductor-build dependent
        terminal = e_dyn
        if not _EVICT_COMPILE_TRIED_STATIC:
            _EVICT_COMPILE_TRIED_STATIC = True
            try:
                _COMPILED_EVICT_FN = _build_compiled_evict(dynamic=False)
                _COMPILED_EVICT_FN(cache, layer_idx, step)
                _announce_evict_path_once(compiled=True)
                _EVICT_STATS["compiled"] += 1
                return
            except Exception as e_static:
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
            "STICKYKV_COMPILE_EVICT is set but torch.compile of the two-tier "
            f"eviction failed ({_EVICT_COMPILE_FAILED}).\n" + _EVICT_COMPILE_HELP
        ) from terminal


def _announce_evict_path_once(compiled: bool) -> None:
    """Print, once per process, which eviction path is live."""
    if _EVICT_ANNOUNCED["done"]:
        return
    _EVICT_ANNOUNCED["done"] = True
    if compiled:
        print(
            "[StickyKV] eviction path: COMPILED two-tier eviction ACTIVE [OK] "
            "(STICKYKV_COMPILE_EVICT=1)",
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
        self._layer_major: bool = _layer_major_decode()
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

        # Extract position_ids from cache_kwargs if provided
        pos = None
        if cache_kwargs is not None and "cache_position" in cache_kwargs:
            pos = cache_kwargs["cache_position"]

        # 1. Append
        state.append(key_states, value_states, pos)
        n_new = key_states.shape[2]
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
            pos = (
                cache_kwargs.get("cache_position")
                if cache_kwargs is not None else None
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
        fields = [t.reshape(L, B, n, *t.shape[1:]) for t in (kc, ks, kz, vc, vs, vz)]
        qpos_flat = qpos.reshape(L * B, n * ws)                 # [R, n*ws]
        # The KV store's dtype, NOT fp32: every other RoPE in this cache runs at
        # the store dtype (HF's own, and unrotate/rotate_key_window), so an fp32
        # table here would re-rotate the Q tier with a different convention than
        # the one it was un-rotated with. See rope_cos_sin_halves.
        cos_h, sin_h = rope_cos_sin_halves(
            self.rope_module, qpos_flat, self._kv_dtype)
        self._joint_qpos = qpos_flat

        key = (store.version, n, B)
        kc, ks, kz, vc, vs, vz = fields
        for i in range(L):
            r0 = i * B
            self._fused_ctx[i] = {
                "qkey": key,
                "qtier": {
                    "k_codes": kc[i], "k_scale": ks[i], "k_zero": kz[i],
                    "v_codes": vc[i], "v_scale": vs[i], "v_zero": vz[i],
                    "cos": cos_h[r0:r0 + B], "sin": sin_h[r0:r0 + B],
                    "window_size": ws,
                },
                "qpos": qpos_flat[r0:r0 + B],
                "mkey": None, "score_meta": None,
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
        kc = kc.reshape(B, n, *kc.shape[1:])
        ks = ks.reshape(B, n, *ks.shape[1:])
        kz = kz.reshape(B, n, *kz.shape[1:])
        vc = vc.reshape(B, n, *vc.shape[1:])
        vs = vs.reshape(B, n, *vs.shape[1:])
        vz = vz.reshape(B, n, *vz.shape[1:])
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
        }
        return qtier

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
        dump = torch.full_like(rank, width)
        idx = torch.where(mask & (rank < width), rank.clamp_min(0), dump)
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

        Default (``STICKYKV_COMPILE_EVICT`` off) calls the eager implementation
        below — byte-for-byte the historic path, one extra function-pointer hop.
        With it ON, the body is run through a lazily-built, process-global
        ``torch.compile`` (``dynamic=True`` for the varying window count), which
        fuses the eviction step's ~360 pointwise / gather / scatter / quantize
        launches into a handful of kernels. That step is ~89% of the fused-decode
        path's per-token launch budget (the other ``ws-1`` steps are just the KV
        append), so it is the one place left where launches can be cut without
        touching the eviction math — and compilation is semantics-preserving by
        construction, so the retained windows, tier moves, and rebuilt fp store
        are identical to the eager path. It fires once per ``ws`` steps, so a
        compile / recompile amortizes.

        **Kernel-or-error when enabled — it never falls back to eager.** A silent
        (or even loud) eager run under a "compiled" label defeats the flag's
        entire purpose, which is to MEASURE the compiled path; an eager run would
        report eager numbers, and the ~81%-of-budget eviction step would look
        cut when it was not. So if ``torch.compile`` cannot lower the body,
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
        and the steps to get numbers (``STICKYKV_COMPILE_EVICT=0`` runs eager).
        """
        if not _compile_evict_enabled():
            _announce_evict_path_once(compiled=False)
            _EVICT_STATS["eager"] += 1
            return WindowedCache._evict_two_tier_impl(self, layer_idx, step)
        # Compile requested: compiled-or-raise. No eager fallback by design.
        _run_compiled_evict(self, layer_idx, step)
        return None

    def _evict_two_tier_impl(self, layer_idx: Optional[int], step: int) -> None:
        """One two-tier eviction (design §5), per row, entirely on device.

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
        state, store, policy = self._evict_targets(layer_idx)
        rope = self.rope_module

        ws = self.resolved.window_size
        num_sink = self.resolved.num_sink_tokens
        dtype = state.key_states.dtype
        device = state.key_states.device

        B, H_kv, T_fp, D = state.key_states.shape
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
        T_body = int(T_fp) - num_sink
        W = int(state.window_scores.shape[2])
        # Eviction rewrites the Q tier AND compacts the fp store's positions, so
        # every memoized fused hand-off for this layer is stale. store.version
        # already covers the Q half (retain_only below bumps it); this also covers
        # the fp half without relying on that ordering. A layer-major eviction
        # (layer_idx is None) rewrites every layer, so it drops every entry.
        if layer_idx is None:
            self._fused_ctx = [None] * self.num_layers
        else:
            self._fused_ctx[layer_idx] = None

        n_q_prev = store.num_active_windows

        # --- 1–2. Rank + tier assignment on the merged axis -----------------
        retained_idx, new_tier = policy.compute_two_tier_retain(state.window_scores)
        # W is a concrete int (above), so these are Python ints even under
        # torch.compile; the int() is belt-and-suspenders against a future
        # tier_counts that returns a tensor scalar.
        k_fp, n_q, local_w = policy.tier_counts(W)
        k_fp, n_q, local_w = int(k_fp), int(n_q), int(local_w)
        n_fp = k_fp + local_w

        # Telemetry: snapshot the merged-axis scores. For the two-tier path the
        # retained indices are merged-WINDOW indices (not token indices — the fp
        # and Q survivors live in different stores).
        if layer_idx is None:
            self._record_joint_scores(step, state.window_scores, retained_idx)
        else:
            self.telemetry.record_scores(
                layer_idx, step, state.window_scores, retained_idx
            )

        wids = torch.gather(state.original_window_ids, 1, retained_idx)   # [B, W_ret]
        is_q_new = new_tier == 1

        # --- Resolve window ids → slots ONCE, then decide with masks --------
        store.ensure(B, device)
        has_entry, is_q_cur, slot_of, match = store.lookup(wids)

        demote = is_q_new & ~is_q_cur      # currently fp, wants Q
        fresh = demote & ~has_entry        # never quantized → quantize now
        react = demote & has_entry         # dormant → reactivate, NO requant (§10)
        promote = ~is_q_new & is_q_cur     # currently Q, wants fp

        # --- 3a. Free dropped entries FIRST (§6) ----------------------------
        # Before allocating, not after: it is what makes n_slots_for's
        # top_k_fp + N_q bound exact rather than merely likely. Safe in either
        # order — retain_only only touches windows that are NOT retained, and
        # every demote/promote/reactivate target is retained by construction.
        store.retain_only(match)

        # --- fp window order: retained-fp windows, ascending id -------------
        # `wids` is already ascending (the merged axis is chronological), so
        # pushing the Q-tier picks to a sentinel and sorting compacts the fp
        # picks to the front, still ascending. Everything the fp rebuild needs
        # is then gathered onto that same axis.
        sentinel = torch.iinfo(torch.long).max
        fp_key = torch.where(is_q_new, torch.full_like(wids, sentinel), wids)
        take = torch.argsort(fp_key, dim=1)[:, :n_fp]                  # [B, n_fp]
        fp_wids = torch.gather(wids, 1, take)
        fp_prom = torch.gather(promote, 1, take)
        fp_slot = torch.gather(slot_of, 1, take)
        prom_rank = fp_prom.cumsum(1) - 1                              # [B, n_fp]

        body_k = state.key_states[:, :, num_sink:, :]
        body_v = state.value_states[:, :, num_sink:, :]
        body_pos = state.position_ids[:, num_sink:]
        # Token → window id. Non-decreasing per row (the fp store is
        # [sink ‖ windows by ascending id] and generation appends the newest
        # tokens at the end), which is what lets searchsorted below resolve a
        # window to its run in one shot — no host span map, no sync.
        body_wid = ((body_pos - num_sink) // ws).contiguous()          # [B, T_body]

        # --- 3b. Promote (Q→fp): ONE batched dequant + ONE RoPE -------------
        # Width is the BOUND, not the count: promotions are ragged per row. The
        # invalid lanes dequantize garbage from free slots and are dropped by the
        # mask below — bit-identical for the valid ones, since every op here is
        # per-window or per-token pointwise.
        p_max = min(self.resolved.N_q, n_fp)
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
        react_slot, react_valid = self._compact(slot_of, react, n_q)
        store.reactivate_many(react_slot, react_valid)

        fresh_wid, fresh_valid = self._compact(wids, fresh, n_q)
        if n_q > 0 and T_body > 0:
            start_f = torch.searchsorted(body_wid, fresh_wid.clamp_min(0))  # [B, n_q]
            tok_f = _clamp_index(
                (start_f.unsqueeze(-1) + torch.arange(ws, device=device))
                .reshape(B, n_q * ws),
                T_body - 1,
            )
            idx_d = tok_f.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, n_q * ws, D)
            k_post = torch.gather(body_k, 2, idx_d)
            v_tok = torch.gather(body_v, 2, idx_d)
            prange = torch.gather(body_pos, 1, tok_f)
            k_pre_d = unrotate_key_window(k_post, prange, rope)
            store.demote_many(
                store.table.free_slots(n_q), fresh_valid, fresh_wid,
                k_pre_d.reshape(B, H_kv, n_q, ws, D).permute(0, 2, 1, 3, 4),
                v_tok.reshape(B, H_kv, n_q, ws, D).permute(0, 2, 1, 3, 4),
                prange.reshape(B, n_q, ws),
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

        if layer_idx is None:
            # Layer-major: the buffer is already at its steady capacity (the
            # per-layer first eviction is what sized it), so the compacted body
            # goes straight in behind the sink prefix, which this rebuild carries
            # through byte-identically at the same offsets. That skips the three
            # full-store `cat`s below — ~2.9 GB of transient at 4096/batch-32 once
            # the row axis carries all 32 layers — and the reallocation `replace`
            # performs whenever the target size moves.
            state.replace_body(num_sink, new_k, new_v, new_pos)
        else:
            # Per-layer: this IS the first eviction, where `replace` reallocating
            # from the prompt capacity down to the budget is the point — it is
            # what releases the prompt-sized buffer.
            state.replace(
                torch.cat([state.key_states[:, :, :num_sink, :], new_k], dim=2),
                torch.cat([state.value_states[:, :, :num_sink, :], new_v], dim=2),
                torch.cat([state.position_ids[:, :num_sink], new_pos], dim=1),
            )

        # --- 5. Gather scores + ids to the retained merged axis -------------
        H_q = state.window_scores.shape[1]
        idx_w = retained_idx.unsqueeze(1).expand(B, H_q, -1)
        state.window_scores = torch.gather(
            state.window_scores, dim=-1, index=idx_w
        ).contiguous()
        state.original_window_ids = torch.gather(
            state.original_window_ids, 1, retained_idx
        ).contiguous()

        # --- 6. Commit counts; policy total tracks T_fp + T_q ---------------
        store.commit_active_count(n_q)
        policy.set_total_after_compaction(
            state.seq_length + store.num_active_tokens
        )

    def reorder_cache(self, beam_idx: Tensor) -> None:
        """Beam search is out of scope (v1)."""
        raise NotImplementedError(
            "WindowedCache does not support beam search (reorder_cache). "
            "Use greedy or sampling decoding."
        )
