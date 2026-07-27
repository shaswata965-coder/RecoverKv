"""What a ``compression_ratio`` actually buys on a GSM8K-length prompt.

Every press in this repo was designed for 4k-32k contexts. GSM8K prompts are ~120-200
tokens. Three things break at that scale, none of them loudly, and all three would
show up as "the method scores badly on GSM8K" rather than as "this configuration never
measured the method":

1.  **The press cannot run at all.** ``SnapKVPress.score`` (and every press derived
    from it, including DefensiveKV) opens with ``assert q_len > self.window_size``.
    With ``window_size=32`` a short context raises mid-prefill and takes the run with
    it. :class:`ContextBudget` predicts this before any GPU work happens.

2.  **The budget collapses into the observation window.** The retained budget is
    ``n_kept = int(q_len * (1 - compression_ratio))``. The last ``window_size`` tokens
    are force-kept at max score ("Add back the observation window", snapkv_press.py:106),
    so when ``n_kept <= window_size`` the *entire* budget is spent on the window and
    top-k has no free slots left to express a preference. Every scoring method — SnapKV,
    AdaKV, DefensiveKV — degenerates to the same thing: keep the last ``n_kept`` tokens.
    At ``q_len=130, cr=0.8, window_size=32`` this is exactly what happens: ``n_kept=26``,
    which is *below* the window. Comparing presses in that regime compares nothing.

3.  **DefensiveKV silently mis-selects.** ``efficient_defensivekv_press.py`` clamps its
    CriticalKV stage-1 count with
    ``count.clamp_(max=int(q_len * (1 - cr) - window_size))``. When that bound is
    negative — the same ``n_kept <= window_size`` regime — the clamp makes ``count``
    negative, and ``sorted_indices[b, h, :k]`` with negative ``k`` selects *everything
    except the last |k|* instead of the top ``k``. The defensive mechanism inverts. No
    exception, no warning, just a different algorithm than the one named in the paper.

So: compute the budget host-side, refuse to start a run that lands in regime 2 or 3
unless explicitly allowed, and afterwards report what was *measured* off the cache
rather than what was requested.

A note on the denominator
-------------------------
kvpress compresses the context **once, at prefill**, and never touches the cache again.
The question tokens are appended uncompressed and every generated token is appended
uncompressed. So a nominal ``compression_ratio=0.8`` on a 130-token context with a
200-token answer removes 104 of 330 KV entries — an end-to-end saving of 31%, not 80%.
:func:`effective_retention` is that honest number, and it is what the comparison table
prints next to the nominal ratio.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Per-example budget prediction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBudget:
    """What the press will do to one context, computed before the forward pass."""

    context_tokens: int
    compression_ratio: float
    window_size: int

    #: Retained KV entries per head after prefill (``ScorerPress.compress``). For the
    #: head-wise presses (AdaKV / DefensiveKV) this is the per-head *mean* — the layer
    #: budget is ``n_kept * num_kv_heads`` and is redistributed across heads, so the
    #: total is the same and only the allocation differs.
    n_kept: int
    n_pruned: int

    #: The press asserts ``q_len > window_size``; below that it raises mid-prefill.
    applicable: bool
    #: ``n_kept <= window_size``: the budget is entirely the force-kept observation
    #: window, so the scorer's ranking cannot express a preference (see module docstring).
    degenerate: bool
    #: ``n_kept < context_tokens``: anything is actually pruned at all.
    binds: bool

    @property
    def requested_compression(self) -> float:
        return self.compression_ratio

    @property
    def achieved_context_compression(self) -> float:
        """Fraction of the context pruned, after integer flooring."""
        if self.context_tokens <= 0:
            return 0.0
        return self.n_pruned / self.context_tokens

    @property
    def free_slots(self) -> int:
        """Budget slots the scorer can actually allocate by rank (``n_kept - window``).

        This is the number DefensiveKV's stage-1 clamp uses. Negative means regime 3.
        """
        return self.n_kept - min(self.window_size, self.context_tokens)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["achieved_context_compression"] = round(self.achieved_context_compression, 4)
        d["free_slots"] = self.free_slots
        return d


def resolve_budget(
    context_tokens: int, compression_ratio: float, window_size: int = 32
) -> ContextBudget:
    """Predict what the press will retain for a context of *context_tokens* tokens.

    Mirrors ``ScorerPress.compress``: ``n_kept = int(q_len * (1 - compression_ratio))``
    (the ``keys.shape[2] - q_len`` term is zero during prefill, which is the only time
    a press runs).
    """
    if compression_ratio <= 0.0:
        return ContextBudget(
            context_tokens=context_tokens,
            compression_ratio=0.0,
            window_size=window_size,
            n_kept=context_tokens,
            n_pruned=0,
            applicable=True,
            degenerate=False,
            binds=False,
        )

    n_kept = int(context_tokens * (1.0 - compression_ratio))
    return ContextBudget(
        context_tokens=context_tokens,
        compression_ratio=compression_ratio,
        window_size=window_size,
        n_kept=n_kept,
        n_pruned=max(context_tokens - n_kept, 0),
        applicable=context_tokens > window_size,
        degenerate=n_kept <= min(window_size, context_tokens),
        binds=n_kept < context_tokens,
    )


def effective_retention(
    n_kept: int, question_tokens: int, gen_tokens: int, context_tokens: int
) -> float:
    """Retained KV / full-sequence KV, end to end.

    The context is compressed once; the question and every generated token are
    appended uncompressed and never pruned. This is the fraction of the full-cache
    footprint the run actually paid for.
    """
    full = context_tokens + question_tokens + gen_tokens
    if full <= 0:
        return 1.0
    return (n_kept + question_tokens + gen_tokens) / full


# ---------------------------------------------------------------------------
# Pre-flight gate over the whole dataset
# ---------------------------------------------------------------------------


@dataclass
class PreflightReport:
    """Aggregate budget prediction over every example, before any GPU work."""

    n_examples: int
    compression_ratio: float
    window_size: int
    n_inapplicable: int
    n_degenerate: int
    n_negative_free_slots: int
    mean_context_tokens: float
    min_context_tokens: int
    max_context_tokens: int
    mean_n_kept: float
    mean_free_slots: float

    @property
    def pct_degenerate(self) -> float:
        return 100.0 * self.n_degenerate / self.n_examples if self.n_examples else 0.0

    @property
    def pct_inapplicable(self) -> float:
        return (
            100.0 * self.n_inapplicable / self.n_examples if self.n_examples else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pct_degenerate"] = round(self.pct_degenerate, 2)
        d["pct_inapplicable"] = round(self.pct_inapplicable, 2)
        return d

    def format(self) -> str:
        return (
            f"  context tokens   mean {self.mean_context_tokens:.1f}  "
            f"min {self.min_context_tokens}  max {self.max_context_tokens}\n"
            f"  compression_ratio {self.compression_ratio}  window_size {self.window_size}\n"
            f"  n_kept           mean {self.mean_n_kept:.1f}  "
            f"(free slots after the observation window: mean {self.mean_free_slots:.1f})\n"
            f"  press skipped    {self.n_inapplicable}/{self.n_examples} "
            f"({self.pct_inapplicable:.1f}%)  [context <= window_size]\n"
            f"  degenerate       {self.n_degenerate}/{self.n_examples} "
            f"({self.pct_degenerate:.1f}%)  [budget <= observation window]"
        )

    def blocking_reasons(self, degenerate_pct_limit: float = 10.0) -> List[str]:
        """Reasons this configuration should not be run as a measurement of the press."""
        reasons: List[str] = []
        if self.pct_degenerate > degenerate_pct_limit:
            reasons.append(
                f"budget collapses into the observation window on "
                f"{self.pct_degenerate:.1f}% of examples (n_kept <= window_size={self.window_size}). "
                f"Every press degenerates to 'keep the last n_kept tokens' there, so the run "
                f"would not distinguish snapkv from adakv from defensivekv. "
                f"Fix: lower --compression_ratio (need n_kept > {self.window_size}, i.e. "
                f"ratio < 1 - {self.window_size}/context_len ~ "
                f"{max(0.0, 1.0 - self.window_size / max(self.mean_context_tokens, 1.0)):.2f}), "
                f"or lower the press window_size via --press_window_size."
            )
        if self.n_negative_free_slots:
            reasons.append(
                f"{self.n_negative_free_slots}/{self.n_examples} examples have "
                f"n_kept < window_size, which makes DefensiveKV's stage-1 clamp "
                f"(count.clamp_(max=int(q_len*(1-cr) - window_size))) NEGATIVE. Its "
                f"selection then inverts silently -- the run would report a different "
                f"algorithm under the DefensiveKV name."
            )
        if self.pct_inapplicable > 0.0:
            reasons.append(
                f"{self.n_inapplicable}/{self.n_examples} examples have a context no longer "
                f"than window_size={self.window_size}; the press asserts q_len > window_size "
                f"and would raise mid-prefill."
            )
        return reasons


def preflight(
    context_token_counts: Sequence[int],
    compression_ratio: float,
    window_size: int = 32,
) -> PreflightReport:
    """Run :func:`resolve_budget` over every example and aggregate."""
    counts = list(context_token_counts)
    if not counts:
        raise ValueError("preflight() needs at least one context length")

    budgets = [resolve_budget(c, compression_ratio, window_size) for c in counts]
    return PreflightReport(
        n_examples=len(budgets),
        compression_ratio=compression_ratio,
        window_size=window_size,
        n_inapplicable=sum(1 for b in budgets if not b.applicable),
        n_degenerate=sum(1 for b in budgets if b.degenerate),
        n_negative_free_slots=sum(1 for b in budgets if b.free_slots < 0),
        mean_context_tokens=sum(counts) / len(counts),
        min_context_tokens=min(counts),
        max_context_tokens=max(counts),
        mean_n_kept=sum(b.n_kept for b in budgets) / len(budgets),
        mean_free_slots=sum(b.free_slots for b in budgets) / len(budgets),
    )


# ---------------------------------------------------------------------------
# Measuring the cache that was actually built
# ---------------------------------------------------------------------------


@dataclass
class CacheMeasurement:
    """What the compressed cache really holds, read off the cache object itself.

    Predicting the budget is not the same as measuring it: integer flooring, per-head
    redistribution, and any press-specific safeguard all move the real number. This is
    read after the press has run, so it is the number to report.
    """

    #: Retained KV entries summed over layers and heads (and batch rows).
    total_kv_entries: int
    #: Same, but for an uncompressed cache of the same context.
    full_kv_entries: int
    #: Per-layer retained entries (summed over heads).
    per_layer_entries: List[int]
    #: True when the cache allocates per-head budgets independently (AdaKV family).
    head_wise: bool
    #: Per-head lengths for the head-wise caches; empty otherwise. Reported because
    #: unequal allocation is the entire point of AdaKV/DefensiveKV, and a run where the
    #: spread is zero has not exercised it.
    head_len_min: Optional[int] = None
    head_len_max: Optional[int] = None

    @property
    def measured_compression(self) -> float:
        """Fraction of the context KV actually pruned."""
        if self.full_kv_entries <= 0:
            return 0.0
        return 1.0 - self.total_kv_entries / self.full_kv_entries

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["measured_compression"] = round(self.measured_compression, 4)
        return d


def measure_cache(cache: Any, context_tokens: int) -> CacheMeasurement:
    """Read the retained KV size out of a kvpress cache.

    Handles both cache layouts the pipeline can produce:

    * ``DynamicCache`` — dense ``[B, H_kv, T, D]`` per layer; every head keeps the same
      ``T``, so retained entries are ``T * H_kv``.
    * ``DynamicCacheSplitHeadFlatten`` (AdaKV / DefensiveKV) — flattened ``[N, D]`` per
      layer with per-head lengths in ``metadata_list[i].head_lens``. ``get_seq_length``
      is *not* meaningful there, which is why this function exists.
    """
    per_layer: List[int] = []
    head_wise = False
    head_min = head_max = None
    n_heads = 0

    metadata_list = getattr(cache, "metadata_list", None)
    if metadata_list:
        # AdaKV family: flattened per-head storage.
        head_wise = True
        for meta in metadata_list:
            head_lens = meta.head_lens
            total = int(head_lens.sum().item())
            per_layer.append(total)
            lo, hi = int(head_lens.min().item()), int(head_lens.max().item())
            head_min = lo if head_min is None else min(head_min, lo)
            head_max = hi if head_max is None else max(head_max, hi)
            n_heads = max(n_heads, int(head_lens.numel()))
    else:
        layers = getattr(cache, "layers", None)
        if layers is None:
            # transformers < 4.54 kept the tensors in flat lists.
            keys_list = getattr(cache, "key_cache", [])
            layers = [type("L", (), {"keys": k})() for k in keys_list]
        for layer in layers:
            keys = getattr(layer, "keys", None)
            if keys is None or keys.numel() == 0:
                continue
            bsz, h_kv, seq_len, _ = keys.shape
            per_layer.append(bsz * h_kv * seq_len)
            n_heads = max(n_heads, bsz * h_kv)
            head_min = seq_len if head_min is None else min(head_min, seq_len)
            head_max = seq_len if head_max is None else max(head_max, seq_len)

    total = sum(per_layer)
    full = context_tokens * n_heads * len(per_layer)
    return CacheMeasurement(
        total_kv_entries=total,
        full_kv_entries=full,
        per_layer_entries=per_layer,
        head_wise=head_wise,
        head_len_min=head_min,
        head_len_max=head_max,
    )
