"""Grouping GSM8K prompts into batches, and accounting for what padding costs.

Measured on the real 1319-example test split with the Llama-3.1 tokenizer:

    context tokens:   min 124   p50 157   p90 191   max 287   mean 161
    distinct lengths: 112

Two consequences that decide the batching strategy:

*   **Exact-length bucketing saturates at ~12.** The largest exact-length bucket holds
    38 examples and none holds 64, so no cap above ~38 changes anything. Zero padding,
    zero distortion, but no path to a large batch.
*   **Reaching B=64/128 requires padding**, and the cost that matters is *not* wasted
    tokens.

The cost that matters
---------------------
``ScorerPress.compress`` sizes the budget off ``hidden_states.shape[1]`` — the
**padded** length::

    n_kept = int(q_len * (1 - compression_ratio))

So every row in a group gets a budget sized to the *longest* row in that group. A row
6 tokens short of the group max retains ~3 extra KV entries at ratio 0.5, i.e. it is
compressed slightly less than requested, by an amount that depends on which other
prompts it was grouped with.

Pure sort-and-chunk maximises batch size and is the worst offender: at cap 128 it
reaches a mean batch of 119.9 for only 2.4% wasted tokens, but its unbounded
intra-group spread produces a **34.9% worst-case budget drift** on the tail chunk.
That is the same class of error as the cross-row budget theft this package fixes in
``layer_defensivekv`` — a measurement that changes with batch composition — so it is
not worth the extra rows.

Constraining the group to a length multiple bounds the spread, and therefore the
drift, at a modest cost in batch size (measured, cap 128):

    strategy              mean batch   worst drift
    exact length               11.8          0.0%
    pad to multiple of 8       57.3          6.2%
    pad to multiple of 16      69.4          8.3%
    pad to multiple of 32      94.2         15.6%
    sort and chunk            119.9         34.9%

:func:`plan_batches` implements the bounded variants; :func:`budget_drift` computes
the drift so a run can assert on it rather than discover it later. Passing
``pad_to_multiple=None`` gives exact-length groups, which make a batched run
bit-identical to ``B=1`` — the configuration to validate against before trusting a
padded one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class BatchPlan:
    """A grouping of example indices into batches, plus what it will cost."""

    groups: List[List[int]]
    #: Padded length each group runs at (max true length in the group).
    group_lengths: List[int]
    pad_to_multiple: Optional[int]
    max_batch_size: int

    @property
    def n_batches(self) -> int:
        return len(self.groups)

    @property
    def n_examples(self) -> int:
        return sum(len(g) for g in self.groups)

    @property
    def mean_batch_size(self) -> float:
        return self.n_examples / self.n_batches if self.groups else 0.0

    @property
    def max_spread(self) -> int:
        """Largest (max - min) true length inside any group. 0 means no padding."""
        return max(self._spreads, default=0)

    _spreads: List[int] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        return (
            f"{self.n_batches} batches over {self.n_examples} examples  "
            f"(mean {self.mean_batch_size:.1f}, cap {self.max_batch_size}, "
            f"pad_to_multiple={self.pad_to_multiple}, max spread {self.max_spread} tok)"
        )


def plan_batches(
    lengths: Sequence[int],
    max_batch_size: int = 128,
    pad_to_multiple: Optional[int] = 16,
) -> BatchPlan:
    """Group example indices into batches of similar prompt length.

    Parameters
    ----------
    lengths
        Tokenized context length per example, in dataset order.
    max_batch_size
        Cap on rows per batch.
    pad_to_multiple
        ``None`` groups only exactly-equal lengths (no padding at all — the
        validation configuration). An int rounds each length up to that multiple and
        groups by the rounded value, bounding intra-group spread to
        ``pad_to_multiple - 1`` tokens.

    Returns
    -------
    BatchPlan
        Groups are emitted in ascending key order, and each group's members are
        sorted by index, so a run is reproducible.
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
    if pad_to_multiple is not None and pad_to_multiple < 1:
        raise ValueError(
            f"pad_to_multiple must be >= 1 or None, got {pad_to_multiple}"
        )

    def key(length: int) -> int:
        if pad_to_multiple is None:
            return length
        return ((length + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

    buckets: Dict[int, List[int]] = {}
    for i, length in enumerate(lengths):
        buckets.setdefault(key(int(length)), []).append(i)

    groups: List[List[int]] = []
    group_lengths: List[int] = []
    spreads: List[int] = []
    for _, members in sorted(buckets.items()):
        # Sort within a bucket by true length so that when a bucket has to be split,
        # the pieces are as internally uniform as possible — this is what keeps the
        # realised spread below the pad_to_multiple bound rather than at it.
        members.sort(key=lambda i: (lengths[i], i))
        for start in range(0, len(members), max_batch_size):
            chunk = sorted(members[start : start + max_batch_size])
            chunk_lengths = [int(lengths[i]) for i in chunk]
            groups.append(chunk)
            group_lengths.append(max(chunk_lengths))
            spreads.append(max(chunk_lengths) - min(chunk_lengths))

    return BatchPlan(
        groups=groups,
        group_lengths=group_lengths,
        pad_to_multiple=pad_to_multiple,
        max_batch_size=max_batch_size,
        _spreads=spreads,
    )


def n_kept_for(length: int, compression_ratio: float) -> int:
    """``ScorerPress.compress``'s budget, reproduced exactly.

    Uses the same float expression as upstream — ``int(q_len * (1 - ratio))`` — so
    the drift numbers below match what the press will really do, float artefacts
    included (``int(150 * (1 - 0.8))`` is 29, not 30).
    """
    return int(length * (1.0 - compression_ratio))


def budget_drift(
    plan: BatchPlan, lengths: Sequence[int], compression_ratio: float
) -> Dict[str, float]:
    """How much extra KV each row retains because it was padded up to its group.

    Returns ``mean_pct`` / ``worst_pct`` (relative to the row's own correct budget)
    and ``worst_tokens`` (absolute). A run should assert on ``worst_pct``: it is the
    amount by which the *reported* compression ratio can differ from the one actually
    applied to an individual example.
    """
    if compression_ratio <= 0.0:
        return {"mean_pct": 0.0, "worst_pct": 0.0, "worst_tokens": 0.0, "n_rows": 0.0}

    rel: List[float] = []
    absolute: List[int] = []
    for group, padded in zip(plan.groups, plan.group_lengths):
        padded_budget = n_kept_for(padded, compression_ratio)
        for i in group:
            true_budget = n_kept_for(int(lengths[i]), compression_ratio)
            extra = padded_budget - true_budget
            absolute.append(extra)
            rel.append(extra / max(true_budget, 1))

    return {
        "mean_pct": round(100.0 * sum(rel) / len(rel), 3),
        "worst_pct": round(100.0 * max(rel), 3),
        "worst_tokens": float(max(absolute)),
        "n_rows": float(len(rel)),
    }


def assert_drift_within(
    plan: BatchPlan,
    lengths: Sequence[int],
    compression_ratio: float,
    limit_pct: float = 10.0,
) -> Dict[str, float]:
    """Raise unless padding-induced budget drift stays under *limit_pct*.

    The point of failing here is that the alternative is not failing at all: an
    over-padded batch plan produces plausible accuracy numbers that silently
    correspond to a weaker compression ratio than the one in the run's name.
    """
    drift = budget_drift(plan, lengths, compression_ratio)
    if drift["worst_pct"] > limit_pct:
        raise ValueError(
            f"padding-induced budget drift {drift['worst_pct']:.1f}% exceeds the "
            f"{limit_pct}% limit (worst row retains {drift['worst_tokens']:.0f} extra "
            f"KV entries). Batches are padded to their group max and "
            f"n_kept = int(padded_len * (1 - ratio)), so an over-wide group compresses "
            f"its short rows less than requested. Fix: lower pad_to_multiple "
            f"(currently {plan.pad_to_multiple}) or pass pad_to_multiple=None for "
            f"exact-length groups."
        )
    return drift
