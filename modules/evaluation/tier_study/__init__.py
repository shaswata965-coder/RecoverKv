"""Tier-study run matrix — five recorded runs behind seven observation metrics.

The QEvict suite (``modules/evaluation/qevict_observations.py``) asks whether the
*design premise* holds inside **one** cache configuration.  This package asks the
next question: **why this cache and not a simpler one?**  It answers with seven
metrics computed across a matrix of five *separately recorded* runs:

===== ============================ ================================================
 id    run                          what it contributes
===== ============================ ================================================
 R0    full-KV baseline             ground truth — never evicts, so its window
                                    scores are the real attention masses
 R1    evict-only, ``ws = 1``       token-level routing decisions, **no Q tier**
 R2    evict-only, ``ws = 8``       window-level routing decisions, **no Q tier**
 R3    three-tier, ``ws = 8``       fp / Q(int2) / evict
 R4    three-tier, ``ws = 32``      fp / Q(int2) / evict, coarser decisions
===== ============================ ================================================

Three groups of metrics read it:

  **Why a window (M1, M2)** — R0, R1, R3, R4.  Future Missed Mass and Selection
  Churn as the decision granularity coarsens.

  **Why three tiers (M3–M6)** — R0, R2, R3.  Tier mass distribution, alive-set
  Jaccard against the oracle, Q-tier numerical fidelity, and the mass the Q tier
  recovers relative to evicting it.

  **Why promote (M7)** — R0, R3, R4.  Episode-level Global LIR: how often a
  window that has gone cold regains importance at all, and how much of that
  the real fp tier catches.

Reading the ``ws = 1`` point
---------------------------
``R1`` is **evict-only** while ``R3``/``R4`` are three-tier, so tiering
co-varies with granularity at that point — the int2 crumb packer needs a
``window_size`` divisible by 4 (``WindowedCacheConfig.resolve``), so a
three-tier ``ws = 1`` run cannot exist.  ``R1`` is therefore labelled the
**token-level reference**, and the granularity-only contrast is ``R3`` vs
``R4`` (same tiering, ws 8 vs 32).
Every table this package writes carries that distinction in a
``granularity_only`` column; no summary sentence attributes the R1-vs-R3 gap to
granularity alone.

Why the runs must be real
-------------------------
Flush cadence changes cache state divergently: what a ``ws = 32`` run evicts at
its first flush determines what its *second* flush can even see, so a ``ws = 32``
trace cannot be recovered from a ``ws = 8`` trace by pooling scores.  This
package therefore refuses to synthesise a missing condition — no
``granularity_accessible`` stand-in — and :func:`load_run_matrix` stops with a
precise message when a condition is absent, duplicated, or configured other than
declared.

The **baseline** is the one exception, and it is not a policy substitution: R0
never evicts, so its per-window masses at a coarse ``window_size`` are the sum of
its masses at a fine one (both runners bucket post-sink tokens the same way —
``reshape(..., W, ws).sum(-1)``), up to the fp16 storage of the recorded scores.
R0 is recorded once at the finest granularity in the matrix and re-bucketed per
condition with ``qevict_metrics.pool_scores``; :func:`load_run_matrix` requires
exact divisibility and records ``pool_factor`` in every artifact's metadata.

Note that *producing* the matrix still needs a base run per window size:
``utils.config.validate_parity_pair`` requires the ours run's ``window_size`` to
match the base npz it is teacher-forced from.  Those base runs are the same
greedy decode (a base run's ``window_size`` changes only how its scores are
bucketed, never what the model computes), and the loader verifies it — the
forced-token streams of all five npzs must agree.  R0 is the ``ws = 1`` one.

Shared execution contract (verified, not assumed)
-------------------------------------------------
* Prefill is processed identically in every condition and never evicts — the
  score hook is a ``forward_hook``, so no scores exist during the prefill
  ``update()`` (``EvictionPolicy.should_evict`` is gated on ``not is_prefill``).
* Eviction begins at decode step 0 and repeats every ``window_size`` decode
  steps (``first_eviction_step = 0``; see ``policy.should_evict``).
* Trace index 0 is the **prefill forward**, so decode step ``d`` is trace index
  ``d + 1`` and the recorded ``eviction_step_mask`` is expected at trace indices
  ``1, ws+1, 2*ws+1, …``.  :func:`check_flush_cadence` compares the recorded mask
  against that schedule and warns on any drift.
* All five runs must share one prompt, prefill length and forced-token stream;
  the loader checks the identity metadata *and* the recorded
  ``generated_tokens`` arrays.

Per-head convention
-------------------
``policy.compute_retain_window_indices`` ranks **head-mean** scores and emits one
retained set per layer, and the ours-side npz reflects that
(``all_window_ids`` / ``all_window_tier`` are ``[S, T, L, Wo]`` — no head axis).
So "per head" here means each metric is *evaluated* per head against the layer's
shared retained set, using the base run's per-head ``window_scores``
``[S, T, L, H, W]``.  Per-head eviction *decisions* would be a policy change and
are not implemented here.

Layout
------
This module is the shared run-matrix layer (loading, validation, geometry,
mass re-bucketing, output plumbing).  The metrics themselves live one per file
and are runnable standalone::

    python -m modules.evaluation.tier_study.m1_future_missed_mass --help
    python -m modules.evaluation.tier_study.combined --help
    python main.py --config configs/eval_tier_study.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Output plumbing and npz loading are shared with the QEvict suite by design —
# one artifact convention across every observation suite in the repo.
from modules.evaluation.qevict_observations import (
    TIER_FP,
    TIER_LOCAL,
    TIER_Q,
    _ci_num,
    _ci_pct,
    _json_safe,
    _num,
    _pct,
    _write_csv,
    load_parity_npz,
    resolve_npz,
)
from utils import qevict_metrics as QM
from utils import sticky_metrics as SM
from utils.hashing import sha256_file
from utils.logger import get_logger

log = get_logger(__name__)

__all__ = [
    "SCHEMA_VERSION", "CONDITIONS", "ConditionSpec", "ConditionView", "RunMatrix",
    "load_run_matrix", "check_flush_cadence", "add_matrix_args", "matrix_from_args",
    "write_metric_outputs", "stack_tables", "matrix_header_lines",
    "per_trace_to_layer", "per_trace_to_layer_series", "bootstrap_row",
    "paired_row", "warn_vacuous",
    "TIER_FP", "TIER_Q", "TIER_LOCAL",
    "_pct", "_num", "_ci_pct", "_ci_num", "_write_csv", "_json_safe",
]

SCHEMA_VERSION = "1.0"

#: Default output root; every module writes ``<root>/mN_<name>.{csv,json,npz}``.
DEFAULT_OUTPUT_DIR = Path("outputs") / "tier_study"

#: Shared bootstrap / sub-sampling defaults (mirrors ``qevict_observations``).
COMMON_DEFAULTS: Dict[str, Any] = {
    "confidence": 0.95,
    "bootstrap_samples": 2000,
    "seed": 0,
    "trace_axis": "sample_layer",
    "layer_stride": 1,
    "head_stride": 1,
    "max_samples": None,
}


# ---------------------------------------------------------------------------
# The run matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionSpec:
    """Static declaration of one cell of the run matrix."""

    cid: str
    role: str            # "base" (parity_base) | "ours" (parity_ours)
    tiers: str           # "full_kv" | "evict_only" | "three_tier"
    expected_window_size: Optional[int]
    label: str
    note: str

    @property
    def is_baseline(self) -> bool:
        return self.role == "base"

    @property
    def granularity_only_comparable(self) -> bool:
        """Whether this condition may be contrasted for *granularity alone*.

        Only the three-tier points may: R1 is evict-only, so a R1-vs-R3 gap
        mixes granularity with tiering (see the module docstring).
        """
        return self.tiers == "three_tier"


CONDITIONS: Dict[str, ConditionSpec] = {
    "R0": ConditionSpec(
        "R0", "base", "full_kv", None, "full-KV baseline",
        "never evicts; supplies the ground-truth attention masses"),
    "R1": ConditionSpec(
        "R1", "ours", "evict_only", 1, "evict-only ws=1 (token-level reference)",
        "token-level decisions, no Q tier — int2 crumb packing needs a "
        "window_size divisible by 4, so tiering co-varies with granularity "
        "here"),
    "R2": ConditionSpec(
        "R2", "ours", "evict_only", 8, "evict-only ws=8",
        "window-level decisions, no Q tier — the two-tier ablation baseline"),
    "R3": ConditionSpec(
        "R3", "ours", "three_tier", 8, "three-tier ws=8",
        "fp / Q(int2) / evict at the same granularity as R2"),
    "R4": ConditionSpec(
        "R4", "ours", "three_tier", 32, "three-tier ws=32",
        "fp / Q(int2) / evict at coarser granularity than R3"),
}

CONDITION_IDS: Tuple[str, ...] = tuple(CONDITIONS)

#: Identity fields every run in the matrix must agree on (same prompt, same
#: prefill, same forced decode).  ``window_size`` deliberately is *not* here —
#: it is the independent variable.
_IDENTITY_FIELDS = (
    "article_sha", "prefill_len", "gen_len", "num_sink_tokens",
    "model_name", "model_revision", "seed", "tokenizer_sha",
)


@dataclass
class ConditionView:
    """One loaded condition, resolved onto the shared trace axis.

    Window axis
    -----------
    Every array here lives on the **baseline's window axis re-bucketed to this
    condition's ``window_size``** — width ``W``, index ``k`` covering post-sink
    tokens ``[num_sink + k*ws, num_sink + (k+1)*ws)``.  The ours npz's own score
    columns are the *survivor* axis (evicted windows are gone), so survivor ids
    from ``all_window_ids`` are scattered onto this axis; that is what makes the
    accessible masks comparable to the baseline masses.
    """

    spec: ConditionSpec
    path: str
    sha256: str
    metadata: Dict[str, Any]
    window_size: int
    pool_factor: int                    # base window_size -> this one
    num_windows: int                    # W on the re-bucketed axis
    local_windows: int
    top_k_windows: int
    top_k_fp: int
    n_q: int
    quant_ratio: float
    cache_budget: Optional[float]
    w_act: np.ndarray                   # [T] valid windows per trace index
    ew_act: np.ndarray                  # [T] evictable windows per trace index
    creation: np.ndarray                # [W] first trace index at which w exists
    valid_tw: np.ndarray                # [T, W] bool
    event_steps: np.ndarray             # [R] trace indices where eviction fired
    acc_fp: Optional[np.ndarray] = None      # [M, R, W] fp survivors + local tail
    acc_q: Optional[np.ndarray] = None       # [M, R, W] int2 survivors
    acc_alive: Optional[np.ndarray] = None   # [M, R, W] fp | Q | local
    cadence_warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    # (scores, ids, tier) on the ours-side survivor axis — only when the caller
    # asked for them (M5).  Deliberately NOT in ``diagnostics``: that dict is
    # serialised into every artifact's metadata, and these are whole arrays.
    survivor_arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default=None, repr=False)

    # ---- derived views -------------------------------------------------

    @property
    def cid(self) -> str:
        return self.spec.cid

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def num_events(self) -> int:
        return int(self.event_steps.size)

    def event_valid(self) -> np.ndarray:
        """``[R, W]`` bool — which windows exist at each routing event."""
        return self.valid_tw[self.event_steps]

    def band_mask(self) -> np.ndarray:
        """``[R, W]`` bool — the *evictable* band at each routing event.

        The local tail is retained by every policy, so including it would push
        every set-overlap metric toward 1.  Metrics that compare *decisions*
        restrict to this band, exactly as ``faithfulness_runner`` does.
        """
        cols = np.arange(self.num_windows)[None, :]
        return cols < self.ew_act[self.event_steps][:, None]

    def geometry(self) -> Dict[str, Any]:
        return {
            "condition": self.cid,
            "label": self.label,
            "tiers": self.spec.tiers,
            "npz_path": self.path,
            "npz_sha256": self.sha256,
            "window_size": self.window_size,
            "pool_factor_from_baseline": self.pool_factor,
            "num_windows": self.num_windows,
            "num_events": self.num_events,
            "local_windows": self.local_windows,
            "top_k_windows": self.top_k_windows,
            "top_k_fp": self.top_k_fp,
            "N_q": self.n_q,
            "quant_ratio": self.quant_ratio,
            "cache_budget": self.cache_budget,
            "first_eviction_step": self.metadata.get("first_eviction_step"),
            "cadence_warnings": list(self.cadence_warnings),
            **{k: v for k, v in self.diagnostics.items()},
        }


@dataclass
class RunMatrix:
    """The loaded matrix plus the shared trace axis and the baseline masses.

    Traces are ``(sample, layer)`` pairs, as in ``qevict_observations``: an
    accessible *set* cannot be averaged, so the layer axis is folded into the
    bootstrap axis and any coarser grouping is applied to the per-trace
    statistics afterwards via ``trace_group``.
    """

    conditions: Dict[str, ConditionView]
    base_meta: Dict[str, Any]
    base_path: str
    base_sha256: str
    base_window_size: int
    num_samples: int
    num_steps: int
    layer_ids: np.ndarray
    head_ids: np.ndarray
    num_layers_total: int
    num_heads_total: int
    trace_labels: List[Tuple[int, int]]
    trace_group: np.ndarray
    identity: Dict[str, Any]
    diagnostics: Dict[str, Any]
    prefill_len: int
    num_sink: int
    # Raw baseline arrays, kept at their recorded dtype and re-bucketed on demand.
    _base_ws: np.ndarray = field(repr=False, default=None)        # [S,T,L,H,W0] fp16
    _base_step: Optional[np.ndarray] = field(repr=False, default=None)  # [S,T,L,W0]
    _cache: Dict[Tuple[str, str, Optional[int]], np.ndarray] = field(
        repr=False, default_factory=dict)

    # ---- shapes --------------------------------------------------------

    @property
    def num_traces(self) -> int:
        return len(self.trace_labels)

    @property
    def num_layers(self) -> int:
        return int(self.layer_ids.size)

    @property
    def num_heads(self) -> int:
        return int(self.head_ids.size)

    def require(self, *cids: str) -> Tuple[ConditionView, ...]:
        """Fetch conditions, raising a precise error when one was not loaded."""
        missing = [c for c in cids if c not in self.conditions]
        if missing:
            raise KeyError(
                f"run matrix is missing {missing} — this metric reads "
                f"{list(cids)}. Pass the npz for every listed condition; a "
                "missing condition is a missing RUN and cannot be simulated "
                "from another window size (see the package docstring)."
            )
        return tuple(self.conditions[c] for c in cids)

    # ---- baseline masses, re-bucketed per condition --------------------

    def _traces(self, pooled: np.ndarray) -> np.ndarray:
        """``[S, T, L, W] -> [S*L, T, W]`` in ``trace_labels`` order."""
        S, T, L, W = pooled.shape
        return pooled.transpose(0, 2, 1, 3).reshape(S * L, T, W)

    def mass_cum(self, cid: str, head: Optional[int] = None) -> np.ndarray:
        """Cumulative ground-truth window mass ``[M, T, W]`` for ``cid``.

        ``head=None`` is the head-mean signal the eviction policy actually ranks
        on; an int selects one head of ``head_ids`` (evaluated against the
        layer's shared retained set — see the package docstring).
        """
        return self._mass("cum", cid, head)

    def mass_step(self, cid: str, head: Optional[int] = None) -> np.ndarray:
        """Per-step ground-truth window mass ``[M, T, W]`` for ``cid``.

        Head-mean uses the base npz's recorded fp32 ``step_window_scores``.  The
        per-head signal has no recorded counterpart (that array is head-mean by
        construction), so it is differenced from the fp16 cumulative array and
        clamped — noisy for long runs, which is why the diagnostics carry
        ``differenced_step_mass_negative_fraction`` and the metrics report the
        per-layer figure from the exact source.
        """
        return self._mass("step", cid, head)

    def _mass(self, kind: str, cid: str, head: Optional[int]) -> np.ndarray:
        key = (kind, cid, head)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        view = self.conditions[cid]
        f = view.pool_factor
        if kind == "cum" or head is not None or self._base_step is None:
            src = (self._base_ws[:, :, :, head, :] if head is not None
                   else self._base_ws.mean(axis=3))          # [S, T, L, W0]
            pooled = QM.pool_scores(src, f)                  # [S, T, L, W]
            cum = self._traces(pooled)
            if kind == "cum":
                out = cum
            else:
                step = np.diff(cum, axis=1, prepend=np.zeros((cum.shape[0], 1,
                                                              cum.shape[2])))
                neg = step < 0
                self.diagnostics.setdefault(
                    "differenced_step_mass_negative_fraction",
                    float(neg.mean()) if neg.size else 0.0)
                out = np.clip(step, 0.0, None)
        else:
            pooled = QM.pool_scores(self._base_step, f)      # [S, T, L, W]
            out = np.clip(self._traces(pooled), 0.0, None)
        W = view.num_windows
        if out.shape[-1] < W:
            out = np.pad(out, [(0, 0), (0, 0), (0, W - out.shape[-1])])
        out = out[..., :W]
        # Only head-mean arrays are cached: a per-head cache would grow with H.
        if head is None:
            self._cache[key] = out
        return out

    def base_window_scores(self, cid: str) -> np.ndarray:
        """Baseline per-head scores ``[S, T, L, H, W]`` at ``cid``'s bucketing.

        The shape ``utils.sticky_metrics.compute_sticky_metrics`` expects.  Held
        in float64 by ``pool_scores``, so it is the one accessor that materialises
        the whole head axis — fine at ``ws >= 8``, avoid at ``ws = 1``.
        """
        view = self.conditions[cid]
        pooled = QM.pool_scores(self._base_ws, view.pool_factor)
        W = view.num_windows
        if pooled.shape[-1] < W:
            pooled = np.pad(pooled, [(0, 0)] * 4 + [(0, W - pooled.shape[-1])])
        return pooled[..., :W]

    def ours_window_scores(self, cid: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(scores [S,T,L,H,Wo], ids [S,T,L,Wo], tier [S,T,L,Wo])`` for ``cid``.

        The ours-side *survivor* axis, un-scattered: what the compressed run
        itself measured, which is what M5 compares against the baseline.
        """
        raw = self.conditions[cid].survivor_arrays
        if raw is None:
            raise KeyError(
                f"{cid}: survivor score arrays were not retained; load the "
                "matrix with keep_ours_scores=True (M5 needs them).")
        return raw

    # ---- metadata ------------------------------------------------------

    def metadata_block(self, cids: Sequence[str]) -> Dict[str, Any]:
        """The provenance block every artifact of this study carries."""
        return {
            "schema_version": SCHEMA_VERSION,
            "suite": "tier_study",
            "runs_used": list(cids),
            "inputs": {
                cid: {
                    "npz_path": self.conditions[cid].path,
                    "npz_sha256": self.conditions[cid].sha256,
                    "window_size": self.conditions[cid].window_size,
                    "tiers": self.conditions[cid].spec.tiers,
                }
                for cid in cids if cid in self.conditions
            },
            "geometry": {
                "prefill_len": self.prefill_len,
                "num_sink": self.num_sink,
                "baseline_window_size": self.base_window_size,
                "num_samples": self.num_samples,
                "num_steps": self.num_steps,
                "num_layers": self.num_layers,
                "num_layers_total": self.num_layers_total,
                "num_heads": self.num_heads,
                "num_heads_total": self.num_heads_total,
                "num_traces": self.num_traces,
                "trace_axis_groups": int(np.unique(self.trace_group).size),
                "layer_ids": self.layer_ids.tolist(),
                "head_ids": self.head_ids.tolist(),
                "per_condition": {
                    cid: self.conditions[cid].geometry()
                    for cid in cids if cid in self.conditions
                },
            },
            "identity": self.identity,
            "diagnostics": self.diagnostics,
            "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------


def expected_flush_indices(num_steps: int, window_size: int,
                           first_eviction_step: int = 0) -> np.ndarray:
    """Trace indices at which eviction is contracted to fire.

    Trace index 0 is the prefill forward, so decode step ``d`` is index ``d+1``
    and ``EvictionPolicy.should_evict`` maps to indices
    ``first_eviction_step + 1`` then every ``window_size`` decode steps.
    """
    d = np.arange(max(num_steps - 1, 0))
    fires = (d == first_eviction_step) | ((d > first_eviction_step)
                                          & (d % max(window_size, 1) == 0))
    return np.flatnonzero(fires) + 1


def check_flush_cadence(view_steps: np.ndarray, num_steps: int, window_size: int,
                        first_eviction_step: int, cid: str) -> List[str]:
    """Compare a recorded eviction mask against the contracted schedule."""
    want = expected_flush_indices(num_steps, window_size, first_eviction_step)
    got = np.asarray(view_steps, dtype=int)
    problems: List[str] = []
    if first_eviction_step != 0:
        problems.append(
            f"first_eviction_step={first_eviction_step} (contract says 0: "
            "eviction begins at decode step 0)")
    if got.size and int(got[0]) == 0:
        problems.append(
            "eviction recorded at trace index 0 — that index is the PREFILL "
            "forward, which must not evict")
    missing = np.setdiff1d(want, got)
    extra = np.setdiff1d(got, want)
    if missing.size:
        problems.append(
            f"{missing.size} contracted flush point(s) absent from the mask "
            f"(first: {missing[:5].tolist()})")
    if extra.size:
        problems.append(
            f"{extra.size} unexpected flush point(s) in the mask "
            f"(first: {extra[:5].tolist()})")
    for p in problems:
        log.warning("%s (ws=%d): %s", cid, window_size, p)
    return problems


def _as5d(x: np.ndarray) -> np.ndarray:
    return x[None] if x.ndim == 4 else x


def _as4d(x: np.ndarray) -> np.ndarray:
    return x[None] if x.ndim == 3 else x


def _condition_tier_kind(meta: Mapping[str, Any]) -> str:
    n_q = int(meta.get("N_q", 0) or 0)
    return "three_tier" if n_q > 0 else "evict_only"


def load_run_matrix(
    paths: Mapping[str, Any],
    *,
    required: Sequence[str] = CONDITION_IDS,
    trace_axis: str = "sample_layer",
    max_samples: Optional[int] = None,
    layer_stride: int = 1,
    head_stride: int = 1,
    keep_ours_scores: bool = False,
    strict: bool = True,
) -> RunMatrix:
    """Load and validate the five-run matrix.

    Parameters
    ----------
    paths
        ``{"R0": path, "R1": path, ...}``.  A path that does not exist is
        retried as a glob in its own directory (``resolve_npz``), so the
        sha-suffixed parity filenames stay usable from a config.
    required
        Which conditions this caller needs.  ``"R0"`` is always required — it is
        the ground truth every metric scores against.
    trace_axis
        Bootstrap grouping over the ``(sample, layer)`` traces:
        ``sample_layer`` (each trace its own group), ``sample``, or ``layer``.
    max_samples, layer_stride, head_stride
        Sub-sampling for memory/time relief.  The retained ids are recorded in
        the metadata so a sub-sampled study is still labelled honestly.
    keep_ours_scores
        Retain each ours run's own ``window_scores``/id/tier arrays (M5 needs
        them; they roughly double peak memory).
    strict
        Raise on a condition whose recorded geometry contradicts its
        declaration.  ``False`` downgrades those to warnings — for inspecting a
        half-finished matrix, never for reporting results.

    Raises
    ------
    FileNotFoundError, KeyError, ValueError
        With a message naming the offending condition.  A missing or mis-shaped
        condition stops the study rather than being simulated: see the package
        docstring on why a flush cadence cannot be pooled after the fact.
    """
    required = list(dict.fromkeys(["R0", *required]))
    unknown = [c for c in list(paths) + required if c not in CONDITIONS]
    if unknown:
        raise KeyError(f"unknown condition id(s) {sorted(set(unknown))}; "
                       f"expected a subset of {list(CONDITION_IDS)}")
    missing = [c for c in required if not paths.get(c)]
    if missing:
        raise FileNotFoundError(
            "tier study cannot start — no npz given for condition(s) "
            f"{missing}. Each condition is a SEPARATE recorded run "
            f"({', '.join(f'{c}: {CONDITIONS[c].label}' for c in missing)}). "
            "Record it with the parity runners; it cannot be derived from "
            "another window size."
        )

    # ── load ─────────────────────────────────────────────────────────────
    loaded: Dict[str, Dict[str, Any]] = {}
    shas: Dict[str, str] = {}
    for cid in required:
        spec = CONDITIONS[cid]
        pattern = ("parity_base_*.npz" if spec.is_baseline else "parity_ours_*.npz")
        p = resolve_npz(paths[cid], pattern)
        loaded[cid] = load_parity_npz(p)
        shas[cid] = sha256_file(p)
        log.info("%s: %s (%s, sha %s…)", cid, Path(p).name, spec.label, shas[cid][:8])

    dupes = {s for s in shas.values() if list(shas.values()).count(s) > 1}
    if dupes:
        same = {c: s[:8] for c, s in shas.items() if s in dupes}
        raise ValueError(
            f"the same npz was passed for several conditions ({same}). Each "
            "condition must be its own recorded run — reusing one file makes "
            "the comparison vacuous.")

    base, base_meta = loaded["R0"], loaded["R0"]["metadata"]
    if base_meta.get("mode") != "parity_base":
        log.warning("R0 npz mode is %r, expected 'parity_base'",
                    base_meta.get("mode"))
    if "window_scores" not in base["arrays"]:
        raise KeyError("R0 npz is missing 'window_scores' — it is the ground "
                       "truth every metric scores against.")

    # ── identity: one prompt, one prefill, one forced decode stream ──────
    identity: Dict[str, Any] = {f: base_meta.get(f) for f in _IDENTITY_FIELDS}
    problems: List[str] = []
    base_tokens = np.asarray(base["arrays"].get("generated_tokens", np.zeros(0)))
    for cid in required:
        if cid == "R0":
            continue
        meta = loaded[cid]["metadata"]
        if meta.get("mode") != "parity_ours":
            log.warning("%s npz mode is %r, expected 'parity_ours'", cid,
                        meta.get("mode"))
        for f in _IDENTITY_FIELDS:
            bv, ov = base_meta.get(f), meta.get(f)
            if bv is not None and ov is not None and bv != ov:
                problems.append(f"  {cid}.{f}: R0={bv!r}, {cid}={ov!r}")
        tok = np.asarray(loaded[cid]["arrays"].get("generated_tokens", np.zeros(0)))
        if base_tokens.size and tok.size:
            n = min(base_tokens.shape[-1], tok.shape[-1])
            k = min(base_tokens.shape[0] if base_tokens.ndim > 1 else 1,
                    tok.shape[0] if tok.ndim > 1 else 1)
            b2 = base_tokens.reshape(-1, base_tokens.shape[-1])[:k, :n]
            o2 = tok.reshape(-1, tok.shape[-1])[:k, :n]
            if not np.array_equal(b2, o2):
                diff = int(np.argmax((b2 != o2).any(axis=0)))
                problems.append(
                    f"  {cid}: forced-token stream diverges from R0 at step "
                    f"{diff} — the runs did not decode the same sequence")
    if problems:
        raise ValueError(
            "tier study alignment failed — the runs are not the same "
            "prompt/prefill/decode:\n" + "\n".join(problems))

    # ── shared axes ──────────────────────────────────────────────────────
    base_ws_raw = _as5d(np.asarray(base["arrays"]["window_scores"]))   # [S,T,L,H,W0]
    S = base_ws_raw.shape[0]
    T = base_ws_raw.shape[1]
    L_all = base_ws_raw.shape[2]
    H_all = base_ws_raw.shape[3]
    for cid in required:
        if cid == "R0":
            continue
        ids = _as4d(np.asarray(loaded[cid]["arrays"]["all_window_ids"]))
        S = min(S, ids.shape[0])
        T = min(T, ids.shape[1])
        L_all = min(L_all, ids.shape[2])
    if max_samples is not None:
        S = max(1, min(S, int(max_samples)))
    layer_ids = np.arange(0, L_all, max(1, int(layer_stride)))
    head_ids = np.arange(0, H_all, max(1, int(head_stride)))
    L = int(layer_ids.size)

    labels = [(s, int(li)) for s in range(S) for li in layer_ids]
    M = len(labels)
    if trace_axis == "sample_layer":
        trace_group = np.arange(M)
    elif trace_axis == "sample":
        trace_group = np.asarray([s for s, _ in labels], dtype=int)
    elif trace_axis == "layer":
        trace_group = np.asarray([li for _, li in labels], dtype=int)
    else:
        raise ValueError(f"unknown trace_axis {trace_axis!r}; expected "
                         "sample_layer|sample|layer")

    base_ws = base_ws_raw[:S, :T][:, :, layer_ids]                    # [S,T,L,H,W0]
    base_step = None
    if "step_window_scores" in base["arrays"]:
        sws = np.asarray(base["arrays"]["step_window_scores"])
        if sws.ndim == 3:
            sws = sws[None]
        base_step = sws[:S, :T][:, :, layer_ids]                      # [S,T,L,W0]
    else:
        log.warning(
            "R0 npz has no 'step_window_scores' (schema %s) — the per-step mass "
            "will be differenced from the fp16 cumulative scores for EVERY "
            "granularity, which is noise at long gen_len. Re-run "
            "BaseParityRunner before reporting M1/M6(b).",
            base_meta.get("schema_version", "?"))

    ws_base = int(base_meta.get("window_size", 1))
    ns = int(base_meta.get("num_sink_tokens", 0))
    prefill_len = int(base_meta.get("prefill_len", 0))

    diagnostics: Dict[str, Any] = {
        "baseline_mass_source": ("recorded_step_mass" if base_step is not None
                                 else "differenced_cumulative"),
        "baseline_schema_version": str(base_meta.get("schema_version", "?")),
        "layer_stride": int(max(1, layer_stride)),
        "head_stride": int(max(1, head_stride)),
        "trace_axis": trace_axis,
    }

    # ── per-condition views ──────────────────────────────────────────────
    views: Dict[str, ConditionView] = {}
    for cid in required:
        if cid == "R0":
            continue
        views[cid] = _build_view(
            CONDITIONS[cid], loaded[cid], shas[cid], base_ws.shape[-1], ws_base,
            S=S, T=T, layer_ids=layer_ids, labels=labels, prefill_len=prefill_len,
            num_sink=ns, keep_ours_scores=keep_ours_scores, strict=strict)

    # The baseline gets a view too, so geometry tables list it like any other
    # run; it has no accessible sets because it drops nothing.
    w_act0, ew_act0, creation0 = SM.flush_geometry(
        T, base_ws.shape[-1], prefill_len, ns, ws_base,
        int(base_meta.get("local_window_size_resolved", 0)) // max(ws_base, 1))
    cols0 = np.arange(base_ws.shape[-1])[None, :]
    views["R0"] = ConditionView(
        spec=CONDITIONS["R0"], path=base["path"], sha256=shas["R0"],
        metadata=base_meta, window_size=ws_base, pool_factor=1,
        num_windows=int(base_ws.shape[-1]),
        local_windows=int(base_meta.get("local_window_size_resolved", 0)) // max(ws_base, 1),
        top_k_windows=int(base_meta.get("top_k_windows", 0)),
        top_k_fp=int(base_meta.get("top_k_windows", 0)), n_q=0, quant_ratio=0.0,
        cache_budget=base_meta.get("cache_budget"),
        w_act=w_act0, ew_act=ew_act0, creation=creation0,
        valid_tw=cols0 < w_act0[:, None], event_steps=np.zeros(0, dtype=int),
        diagnostics={"role": "ground truth (never evicts)"})

    _check_matrix_shape(views, strict)

    matrix = RunMatrix(
        conditions=views, base_meta=base_meta, base_path=base["path"],
        base_sha256=shas["R0"], base_window_size=ws_base, num_samples=S,
        num_steps=T, layer_ids=layer_ids, head_ids=head_ids,
        num_layers_total=int(L_all), num_heads_total=int(H_all),
        trace_labels=labels, trace_group=trace_group, identity=identity,
        diagnostics=diagnostics, prefill_len=prefill_len, num_sink=ns,
        _base_ws=base_ws, _base_step=base_step,
    )
    log.info(
        "Run matrix: %d condition(s) | M=%d traces (%d sample(s) x %d/%d layers) "
        "| T=%d | heads %d/%d | baseline ws=%d, mass=%s",
        len(views) - 1, M, S, L, L_all, T, head_ids.size, H_all, ws_base,
        diagnostics["baseline_mass_source"])
    return matrix


def _build_view(spec, loaded, sha, base_W, ws_base, *, S, T, layer_ids, labels,
                prefill_len, num_sink, keep_ours_scores, strict) -> ConditionView:
    """Resolve one ours run onto the shared trace axis."""
    arrays, meta = loaded["arrays"], loaded["metadata"]
    for key in ("all_window_ids", "all_window_tier"):
        if key not in arrays:
            raise KeyError(
                f"{spec.cid} npz is missing {key!r} — re-run OursParityRunner "
                "(schema >= 1.2) so the full survivor/tier arrays are recorded.")

    ws = int(meta.get("window_size", 0))
    if ws <= 0:
        raise ValueError(f"{spec.cid}: npz metadata has no usable window_size")
    if ws % ws_base:
        raise ValueError(
            f"{spec.cid}: window_size={ws} is not a multiple of the baseline's "
            f"window_size={ws_base}, so the baseline masses cannot be re-bucketed "
            f"onto this condition's window axis. Record R0 at the FINEST "
            f"window_size in the matrix (ws=1 for the R1 point).")
    problems: List[str] = []
    if spec.expected_window_size is not None and ws != spec.expected_window_size:
        problems.append(
            f"{spec.cid} is declared at window_size={spec.expected_window_size} "
            f"but the npz records window_size={ws}")
    kind = _condition_tier_kind(meta)
    if kind != spec.tiers:
        problems.append(
            f"{spec.cid} is declared {spec.tiers} but the npz records "
            f"N_q={meta.get('N_q', 0)}, quant_ratio={meta.get('quant_ratio', 0)} "
            f"({kind})")
    if problems:
        msg = ("run matrix declaration does not match the recorded runs:\n  "
               + "\n  ".join(problems))
        if strict:
            raise ValueError(
                msg + "\nFix the paths (or re-record the run); do not relabel a "
                "run that was not the condition it is being reported as.")
        for p in problems:
            log.warning("%s", p)

    pool_factor = ws // ws_base
    W = -(-base_W // pool_factor)          # ceil, matching pool_scores' padding
    local_windows = int(meta.get("local_window_size_resolved", 0)) // ws
    w_act, ew_act, creation = SM.flush_geometry(
        T, W, prefill_len, num_sink, ws, local_windows)
    cols = np.arange(W)[None, :]
    valid_tw = cols < w_act[:, None]

    # ── routing events ───────────────────────────────────────────────────
    ev_mask = np.asarray(arrays.get("eviction_step_mask", np.zeros(0)))
    if ev_mask.ndim == 1:
        ev_mask = ev_mask[None]
    if ev_mask.size:
        ev_mask = ev_mask[:S, :T].astype(bool)
        event_steps = np.flatnonzero(ev_mask.any(axis=0))
        if ev_mask.any() and not np.array_equal(ev_mask.any(0), ev_mask.all(0)):
            log.warning("%s: eviction_step_mask differs across samples; "
                        "using the union", spec.cid)
    else:
        event_steps = np.zeros(0, dtype=int)
    if event_steps.size == 0:
        raise ValueError(
            f"{spec.cid}: eviction_step_mask is empty — this npz records a run "
            "that never evicted, so it is not the condition it is declared as.")
    cadence = check_flush_cadence(
        event_steps, T, ws, int(meta.get("first_eviction_step", 0)), spec.cid)

    # ── accessible sets, scattered onto the baseline window axis ─────────
    all_ids = _as4d(np.asarray(arrays["all_window_ids"]))[:S, :T][:, :, layer_ids]
    all_tier = _as4d(np.asarray(arrays["all_window_tier"]))[:S, :T][:, :, layer_ids]
    M, R = len(labels), int(event_steps.size)
    acc_fp = np.zeros((M, R, W), dtype=bool)
    acc_q = np.zeros((M, R, W), dtype=bool)
    out_of_range = 0
    layer_pos = {int(li): i for i, li in enumerate(layer_ids)}
    for r, t in enumerate(event_steps):
        for m, (s, li) in enumerate(labels):
            j = layer_pos[int(li)]
            ids, tier = all_ids[s, t, j], all_tier[s, t, j]
            ok = (ids >= 0) & (tier >= 0)
            out_of_range += int(np.sum(ok & (ids >= W)))
            ok &= ids < W
            idx, tr = ids[ok], tier[ok]
            acc_q[m, r, idx[tr == TIER_Q]] = True
            acc_fp[m, r, idx[tr != TIER_Q]] = True
    if out_of_range:
        log.warning("%s: %d survivor id(s) exceeded the baseline window axis "
                    "(W=%d); ignored", spec.cid, out_of_range, W)

    diagnostics: Dict[str, Any] = {
        "survivor_ids_out_of_range": int(out_of_range),
        "schema_version": str(meta.get("schema_version", "?")),
        "mean_alive_units": float((acc_fp | acc_q).sum(axis=-1).mean()),
        "mean_q_units": float(acc_q.sum(axis=-1).mean()),
    }
    survivor = None
    if keep_ours_scores and "window_scores" in arrays:
        ows = _as5d(np.asarray(arrays["window_scores"]))[:S, :T][:, :, layer_ids]
        survivor = (ows, all_ids, all_tier)

    return ConditionView(
        spec=spec, path=loaded["path"], sha256=sha, metadata=meta, window_size=ws,
        pool_factor=pool_factor, num_windows=W, local_windows=local_windows,
        top_k_windows=int(meta.get("top_k_windows", 0)),
        top_k_fp=int(meta.get("top_k_fp", meta.get("top_k_windows", 0))),
        n_q=int(meta.get("N_q", 0) or 0),
        quant_ratio=float(meta.get("quant_ratio", 0.0) or 0.0),
        cache_budget=meta.get("cache_budget"),
        w_act=w_act, ew_act=ew_act, creation=creation, valid_tw=valid_tw,
        event_steps=event_steps, acc_fp=acc_fp, acc_q=acc_q,
        acc_alive=acc_fp | acc_q, cadence_warnings=cadence,
        diagnostics=diagnostics, survivor_arrays=survivor)


def _check_matrix_shape(views: Mapping[str, ConditionView], strict: bool) -> None:
    """Cross-condition sanity: the contrasts must actually be contrasts."""
    msgs: List[str] = []
    if "R2" in views and "R3" in views:
        if views["R2"].window_size != views["R3"].window_size:
            msgs.append(
                f"R2 (ws={views['R2'].window_size}) and R3 "
                f"(ws={views['R3'].window_size}) differ in window_size — the "
                "three-tier contrast is then confounded with granularity")
        b2, b3 = views["R2"].cache_budget, views["R3"].cache_budget
        if b2 is not None and b3 is not None and abs(float(b2) - float(b3)) > 1e-9:
            msgs.append(
                f"R2 cache_budget={b2} != R3 cache_budget={b3} — M6's "
                "'matched retained-byte budget' claim does not hold")
    if "R3" in views and "R4" in views:
        if views["R3"].window_size == views["R4"].window_size:
            msgs.append(
                f"R3 and R4 share window_size={views['R3'].window_size} — there "
                "is no granularity contrast left in the matrix")
    for m in msgs:
        log.warning("%s", m)
    if msgs and strict:
        raise ValueError("run matrix is not a valid comparison:\n  "
                         + "\n  ".join(msgs))


# ---------------------------------------------------------------------------
# Aggregation helpers shared by the metric modules
# ---------------------------------------------------------------------------


def per_trace_to_layer(values: np.ndarray, matrix: RunMatrix) -> np.ndarray:
    """``[M]`` per-trace values → ``[L]`` per-layer means over samples."""
    v = np.asarray(values, dtype=float).reshape(matrix.num_samples, matrix.num_layers)
    return QM.nanmean(v, axis=0)


def per_trace_to_layer_series(values: np.ndarray, matrix: RunMatrix) -> np.ndarray:
    """``[M, R]`` per-trace series → ``[L, R]`` per-layer means over samples.

    The event axis is *kept*, which is what separates this from
    :func:`per_trace_to_layer`.  Stacking one of these per head is how the
    metric modules build their ``[L, H, R]`` (layer x head x flush) arrays —
    the axis a visualisation needs to show one head's trajectory over decode,
    and the one every ``…_per_layer_head`` array throws away by averaging over
    events first.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or v.shape[0] != matrix.num_traces:
        raise ValueError(
            f"expected [M, R] with M={matrix.num_traces}; got {v.shape}")
    v = v.reshape(matrix.num_samples, matrix.num_layers, v.shape[1])
    return QM.nanmean(v, axis=0)


def bootstrap_row(values: np.ndarray, matrix: RunMatrix, *, confidence: float,
                  n_boot: int, seed: int) -> Tuple[float, float, float]:
    """Group-reduce per-trace values then percentile-bootstrap their mean."""
    grouped = QM.group_reduce(np.asarray(values, dtype=float), matrix.trace_group)
    return QM.bootstrap_mean_ci(grouped, confidence, n_boot, seed)


def paired_row(ref: np.ndarray, cmp: np.ndarray, matrix: RunMatrix, *,
               confidence: float, n_boot: int, seed: int) -> Dict[str, Any]:
    """Paired ``ref - cmp`` over the trace axis, with a percentile CI.

    Traces are ``(sample, layer)`` pairs shared by every condition, so two
    conditions with different flush cadences (and therefore different event
    counts) still pair trace-for-trace after each is reduced over its own
    events.
    """
    r = QM.group_reduce(np.asarray(ref, dtype=float), matrix.trace_group)
    c = QM.group_reduce(np.asarray(cmp, dtype=float), matrix.trace_group)
    ok = np.isfinite(r) & np.isfinite(c)
    if not np.any(ok):
        return {"absolute_reduction": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "reference_mean": np.nan,
                "compared_mean": np.nan, "relative_reduction": np.nan,
                "paired_traces": 0}
    mu, lo, hi = QM.bootstrap_mean_ci(r[ok] - c[ok], confidence, n_boot, seed)
    ref_mean = float(QM.nanmean(r[ok]))
    return {
        "absolute_reduction": mu, "ci_lower": lo, "ci_upper": hi,
        "reference_mean": ref_mean, "compared_mean": float(QM.nanmean(c[ok])),
        "relative_reduction": (mu / ref_mean if np.isfinite(ref_mean)
                               and ref_mean > 0 else np.nan),
        "paired_traces": int(np.count_nonzero(ok)),
    }


def warn_vacuous(condition: str, metric: str, reason: str) -> None:
    """Loud, uniform warning for an undefined metric (never a silent zero)."""
    log.warning(
        "%s / %s is UNDEFINED: %s. Reporting NaN — a zero here would read as a "
        "win.", condition, metric, reason)


# ---------------------------------------------------------------------------
# Output plumbing
# ---------------------------------------------------------------------------


def stack_tables(tables: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]]
                 ) -> List[Dict[str, Any]]:
    """Merge several tables into one CSV-able long table.

    ``_write_csv`` takes its field names from the first row, so rows are
    normalised to the union of keys (missing → ``""``) and tagged with the table
    they came from.  One CSV per metric module keeps the artifact contract
    (``mN_<name>.{csv,json,npz}``) exact.
    """
    fields: List[str] = ["table"]
    for _, rows in tables:
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
    out: List[Dict[str, Any]] = []
    for name, rows in tables:
        for row in rows:
            merged = {f: "" for f in fields}
            merged["table"] = name
            merged.update({k: v for k, v in row.items()})
            out.append(merged)
    return out


def write_metric_outputs(
    out_dir: Path,
    stem: str,
    *,
    tables: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]],
    payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
    report: str = "",
) -> Dict[str, Path]:
    """Write ``<stem>.csv`` / ``.json`` / ``.npz`` for one metric module."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    _write_csv(csv_path, stack_tables(tables))

    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe({"metadata": meta, **payload,
                              "report_markdown": report}), f, indent=2)

    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        str(npz_path),
        **{k: np.asarray(v) for k, v in arrays.items()},
        metadata_json=np.array([json.dumps(_json_safe(meta))], dtype=object))
    log.info("Wrote %s.{csv,json,npz} to %s", stem, out_dir.resolve())
    return {"csv": csv_path, "json": json_path, "npz": npz_path}


def matrix_header_lines(matrix: RunMatrix, cids: Sequence[str]) -> List[str]:
    """The shared provenance / caveat header every report section sits under."""
    rows = [
        "| run | condition | window_size | tiers | top_k_fp | N_q | flushes | npz |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cid in cids:
        v = matrix.conditions.get(cid)
        if v is None:
            continue
        rows.append(
            f"| `{cid}` | {v.label} | {v.window_size} | {v.spec.tiers} | "
            f"{v.top_k_fp} | {v.n_q} | {v.num_events or '—'} | "
            f"`{Path(v.path).name}` (sha `{v.sha256[:8]}`) |")
    return [
        f"Traces `M={matrix.num_traces}` = {matrix.num_samples} sample(s) x "
        f"{matrix.num_layers} layer(s) (of {matrix.num_layers_total}); "
        f"`T={matrix.num_steps}` recorded forwards (index 0 = prefill); "
        f"heads evaluated: {matrix.num_heads}/{matrix.num_heads_total}; "
        f"prefill_len={matrix.prefill_len}, num_sink={matrix.num_sink}.",
        "",
        *rows,
        "",
        "Intervals are percentile bootstrap CIs over the trace axis. With a "
        "single-article run those traces are **layers of one prompt**, not "
        "independent samples — read them as within-run variability, not "
        "population CIs.",
        "",
        "`R1` (ws=1) is evict-only because int2 crumb packing requires a "
        "`window_size` divisible by 4, so tiering co-varies with granularity "
        "at that point: it is the **token-level reference**, and the "
        "granularity-only contrast is `R3` vs `R4`.",
    ]


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def add_matrix_args(ap: argparse.ArgumentParser, needed: Sequence[str]) -> None:
    """Add ``--rN-npz`` for each needed condition plus the shared knobs."""
    needed = list(dict.fromkeys(["R0", *needed]))
    for cid in CONDITION_IDS:
        if cid not in needed:
            continue
        spec = CONDITIONS[cid]
        ap.add_argument(f"--{cid.lower()}-npz", type=Path, required=True,
                        help=f"{cid} — {spec.label} ({spec.note})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--trace-axis", choices=("sample_layer", "sample", "layer"),
                    default=COMMON_DEFAULTS["trace_axis"])
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Use at most this many articles (memory relief).")
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="Keep every Nth layer (memory/time relief).")
    ap.add_argument("--head-stride", type=int, default=1,
                    help="Evaluate every Nth head; the per-head arrays then "
                         "carry the retained head ids.")
    ap.add_argument("--no-per-head", action="store_true",
                    help="Skip the per-(layer, head) arrays (per-layer only).")
    ap.add_argument("--confidence", type=float,
                    default=COMMON_DEFAULTS["confidence"])
    ap.add_argument("--bootstrap-samples", type=int,
                    default=COMMON_DEFAULTS["bootstrap_samples"])
    ap.add_argument("--seed", type=int, default=COMMON_DEFAULTS["seed"])
    ap.add_argument("--non-strict", action="store_true",
                    help="Downgrade run-matrix declaration mismatches to "
                         "warnings (inspection only — never for reporting).")


def paths_from_args(args: argparse.Namespace, needed: Sequence[str]
                    ) -> Dict[str, Any]:
    needed = list(dict.fromkeys(["R0", *needed]))
    return {cid: getattr(args, f"{cid.lower()}_npz") for cid in needed
            if getattr(args, f"{cid.lower()}_npz", None) is not None}


def matrix_from_args(args: argparse.Namespace, needed: Sequence[str],
                     *, keep_ours_scores: bool = False) -> RunMatrix:
    """Load the run matrix from parsed CLI args."""
    return load_run_matrix(
        paths_from_args(args, needed), required=needed,
        trace_axis=args.trace_axis, max_samples=args.max_samples,
        layer_stride=args.layer_stride, head_stride=args.head_stride,
        keep_ours_scores=keep_ours_scores, strict=not args.non_strict)
