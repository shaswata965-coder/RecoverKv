"""Batch-enable ``EfficientLayerDefensiveKVPress`` (and the varlen attention it needs).

Nothing here edits ``kvpress``. The two fixes are a subclass and a reversible
monkey-patch, so the upstream repo stays clean and a run that does not import this
module behaves exactly as before.

Two independent problems block ``layer_defensivekv`` at ``B > 1``.

1. A stale assert in the attention (mechanical)
----------------------------------------------
``kvpress/ada_attn.py:219,373,696`` carry ``assert bsz == 1`` under a
``# TODO: support batch size > 1``. The varlen machinery underneath is already
batched — ``_init_metadata`` builds ``head_lens`` / ``cu_seqlens_k`` with
``bsz * num_key_value_heads`` entries (``ada_cache.py:44-47``), and the query is
flattened as::

    q.view(bsz, H_kv, G, T, D).transpose(2, 3).reshape(-1, G, D)   # (b, h_kv, t) major

which matches the ``(b, h_kv)``-major key order that ``cu_seqlens_k`` describes. The
post-attention reshape::

    out.reshape(bsz, H_kv, q_len, G, D).transpose(1, 2).reshape(bsz, q_len, -1)

exactly inverts that construction for any ``bsz``, and head index ``h_kv * G + g``
recovers the standard GQA ordering. So the assert is caution, not a constraint;
:func:`enable_batched_ada_attention` removes it.

2. A batch-flattened global top-k (semantic)
--------------------------------------------
This one is a real bug, and it is silent. ``EfficientAdaGlobalScorerPress.compress``
allocates the cross-layer budget with::

    gloabl_flatten_scores = torch.stack(cache.layer_scores).view(-1)   # batch flattened
    cache_topk_idx = gloabl_flatten_scores.topk(n_kept, dim=-1).indices

The whole premise of LayerDefensiveKV is that *layers* compete for one sequence's
budget. Flattening the batch into the same top-k makes *sequences* compete too: an
easy prompt donates budget to a hard one, and a row's retention depends on which
other rows happened to share its batch. Deleting the assert without fixing this would
produce numbers that quietly change with batch composition — worse than a crash.

:class:`BatchedLayerDefensiveKVPress` replaces both the cross-layer allocation and the
intra-layer selection with per-row equivalents. At ``B == 1`` it reduces to the
original computation (see ``test_batched_presses.py``), which is what makes the
rewrite checkable.

The index math is factored into free functions taking plain tensors so it can be
tested on CPU without flash-attn, a GPU, or model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from kvpress.presses.efficient_layer_defensivekv_press import (
    EfficientLayerDefensiveKVPress,
)


# ---------------------------------------------------------------------------
# Pure index math (CPU-testable)
# ---------------------------------------------------------------------------


def per_row_layer_budgets(
    layer_scores: Tensor,
    n_kept_per_row: int,
    num_kv_heads: int,
    head_len: int,
) -> Tensor:
    """Split a per-row global budget across layers by score rank.

    This is the cross-layer allocation, done **independently per batch row**.

    Parameters
    ----------
    layer_scores : Tensor
        ``[L, B, H*T]`` — one flattened score vector per (layer, row), in the order
        ``cache.layer_scores`` accumulates them.
    n_kept_per_row : int
        Total KV entries to retain for **one** sequence across all layers and heads.
    num_kv_heads, head_len
        ``H`` and ``T``; their product is the per-layer stride inside a row.

    Returns
    -------
    Tensor
        ``[B, L]`` int64 — entries to retain in each layer, per row. Each row sums to
        ``n_kept_per_row``.
    """
    n_layers, bsz, hidden = layer_scores.shape
    stride = num_kv_heads * head_len
    if stride != hidden:
        raise ValueError(
            f"layer_scores last dim ({hidden}) != num_kv_heads*head_len ({stride})"
        )
    total = n_layers * hidden
    if not (0 < n_kept_per_row <= total):
        raise ValueError(
            f"n_kept_per_row must be in (0, {total}], got {n_kept_per_row}"
        )

    # [L, B, H*T] -> [B, L*H*T]. The permute is what keeps rows independent: the
    # original code's `.view(-1)` folded B into the ranked axis, which is the bug.
    per_row = layer_scores.permute(1, 0, 2).reshape(bsz, total)

    idx = per_row.topk(n_kept_per_row, dim=-1).indices          # [B, n_kept]
    layer_of = idx // stride                                    # [B, n_kept]

    budgets = torch.zeros(bsz, n_layers, dtype=torch.int64, device=layer_scores.device)
    budgets.scatter_add_(1, layer_of, torch.ones_like(layer_of))
    return budgets


def per_row_retain_mask(scores: Tensor, budgets: Tensor) -> Tensor:
    """Top-``budgets[b]`` mask over ``scores[b]``, with a **different k per row**.

    ``torch.topk`` takes a scalar ``k``, so a ragged per-row budget cannot be
    expressed directly. Ranking once and thresholding the rank by each row's own
    budget gives the same selection with no Python loop over the batch and no host
    sync — which matters because this runs per layer, per prefill.

    Parameters
    ----------
    scores : Tensor
        ``[B, H*T]`` flattened per-row scores.
    budgets : Tensor
        ``[B]`` int — how many entries row ``b`` retains.

    Returns
    -------
    Tensor
        ``[B, H*T]`` bool, with ``mask[b].sum() == budgets[b]``.
    """
    bsz, width = scores.shape
    order = scores.argsort(dim=-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(
        1, order, torch.arange(width, device=scores.device).expand(bsz, width)
    )
    return rank < budgets.to(rank.device).unsqueeze(1).clamp(0, width)


def flat_gather_indices(
    mask: Tensor, num_kv_heads: int, head_len: int
) -> Tuple[Tensor, Tensor]:
    """Turn a ``[B, H*T]`` retain mask into flat-cache gather indices + new head lengths.

    ``DynamicCacheSplitHeadFlatten`` stores a layer as ``[B*H*T, D]`` in ``(b, h, t)``
    order (``ada_cache.py:_init_metadata`` builds ``head_lens`` as
    ``bsz * num_key_value_heads * [k_len]``). A boolean mask viewed as ``[B, H, T]``
    flattens in exactly that order, so ``nonzero()`` already yields ascending indices
    grouped by ``(b, h)`` — no sort needed, and the per-segment counts are just the
    row sums.

    Returns
    -------
    (indices, head_lens)
        ``indices`` ``[n_retained]`` int64 into the flat cache;
        ``head_lens`` ``[B*H]`` int32, the new per-(row, head) segment lengths.
    """
    bsz = mask.shape[0]
    m3 = mask.view(bsz, num_kv_heads, head_len)
    indices = m3.reshape(-1).nonzero(as_tuple=True)[0]
    head_lens = m3.sum(-1).reshape(-1).to(torch.int32)
    return indices, head_lens


def rebuild_cu_seqlens(head_lens: Tensor) -> Tuple[Tensor, int]:
    """``(cu_seqlens_k, max_seqlen_k)`` for the compacted flat cache.

    Mirrors the layout ``_update_metadata_while_compressing`` expects: a leading zero
    followed by the cumulative sum over every ``(row, head)`` segment.
    """
    cu = torch.cumsum(head_lens, dim=0, dtype=torch.int32)
    cu = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=head_lens.device), cu], dim=0
    )
    return cu, int(head_lens.max().item())


# ---------------------------------------------------------------------------
# The batched press
# ---------------------------------------------------------------------------


@dataclass
class BatchedLayerDefensiveKVPress(EfficientLayerDefensiveKVPress):
    """``EfficientLayerDefensiveKVPress`` with per-row budget allocation.

    Scoring is inherited untouched — only the two allocation steps are replaced, so
    any difference from upstream at ``B == 1`` would be a bug in this class, not a
    change of method.
    """

    window_size: int = 32
    kernel_size: int = 5

    def __str__(self) -> str:
        return (
            f"BatchedLayerDefensiveKV={self.compression_ratio}"
            f"_win={self.window_size}_kerl={self.kernel_size}"
        )

    # -- intra-layer -------------------------------------------------------

    def layer_compress(  # type: ignore[override]
        self,
        module,
        hidden_states: Tensor,
        keys: Tensor,
        values: Tensor,
        budget,
        layer_idx: int,
        kwargs: dict,
    ) -> Tuple[Tensor, Tensor]:
        """Compact one layer's flat cache to a **per-row** budget.

        ``budget`` is a ``[B]`` tensor here, not the scalar the base class takes: rows
        legitimately receive different layer budgets, because the cross-layer split is
        computed per row.
        """
        cache = kwargs.get("past_key_value", kwargs.get("past_key_values", None))
        assert cache is not None, "Cache is required for AdaScorerPress"

        keys = cache.key_cache[layer_idx]
        values = cache.value_cache[layer_idx]
        if self.compression_ratio == 0:
            return keys, values

        meta = cache.metadata_list[layer_idx]
        flatten_scores = cache.layer_scores[layer_idx]          # [B, H*T]
        bsz = flatten_scores.shape[0]
        num_kv_heads = meta.num_key_value_heads
        head_len = int(meta.head_lens[0].item())

        budgets = (
            budget
            if isinstance(budget, Tensor)
            else torch.full((bsz,), int(budget), dtype=torch.int64)
        )
        budgets = budgets.to(flatten_scores.device).to(torch.int64)

        mask = per_row_retain_mask(flatten_scores, budgets)
        indices, new_head_lens = flat_gather_indices(mask, num_kv_heads, head_len)
        cu_seqlens_k, max_seqlen_k = rebuild_cu_seqlens(new_head_lens)
        meta._update_metadata_while_compressing(new_head_lens, cu_seqlens_k, max_seqlen_k)

        idx = indices.unsqueeze(-1).expand(-1, keys.shape[-1])
        keys = keys.gather(0, idx).contiguous()
        values = values.gather(0, idx).contiguous()

        cache.key_cache[layer_idx] = keys
        cache.value_cache[layer_idx] = values
        return keys, values

    # -- cross-layer -------------------------------------------------------

    def compress(  # type: ignore[override]
        self,
        module,
        hidden_states: Tensor,
        keys: Tensor,
        values: Tensor,
        attentions: Tensor,
        kwargs: dict,
    ) -> Tuple[Tensor, Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        cache = kwargs.get("past_key_value", kwargs.get("past_key_values", None))
        assert cache is not None, "Cache is required for AdaScorerPress"
        meta = cache.metadata_list[module.layer_idx]

        with torch.no_grad():
            kwargs["metadata"] = meta
            flatten_scores = self.score(
                module, hidden_states, keys, values, attentions, kwargs
            )

        # Compression only fires once every layer has been scored (the global scorer
        # needs the whole stack before it can allocate across layers).
        cache.layer_scores.append(flatten_scores)
        n_layers = module.config.num_hidden_layers
        if module.layer_idx < n_layers - 1:
            return keys, values

        q_len = hidden_states.shape[1]
        num_kv_heads = meta.num_key_value_heads
        head_len = int(meta.head_lens[0].item())

        # Budget for ONE sequence across all layers and heads — unchanged from
        # upstream; what changes is that every row now gets its own.
        n_kept_per_row = int(
            q_len * (1 - self.compression_ratio) * num_kv_heads * n_layers
        )

        budgets = per_row_layer_budgets(
            torch.stack(cache.layer_scores, dim=0),
            n_kept_per_row=n_kept_per_row,
            num_kv_heads=num_kv_heads,
            head_len=head_len,
        )                                                        # [B, L]

        for layer_idx in range(n_layers):
            self.layer_compress(
                module, hidden_states, keys, values,
                budgets[:, layer_idx], layer_idx, kwargs,
            )

        return (
            cache.key_cache[module.layer_idx],
            cache.value_cache[module.layer_idx],
        )


# ---------------------------------------------------------------------------
# Reversible patch for the stale attention assert
# ---------------------------------------------------------------------------

_ADA_ATTN_PATCHED: List[Tuple[object, str, object]] = []


def enable_batched_ada_attention() -> int:
    """Drop the stale ``assert bsz == 1`` from the varlen attention forwards.

    Implemented by recompiling each ``forward`` with the assert line removed, rather
    than by rewriting the file, so ``kvpress`` on disk is untouched and
    :func:`disable_batched_ada_attention` restores the originals exactly.

    Returns the number of patched classes.
    """
    import inspect
    import textwrap

    from kvpress import ada_attn

    names = [
        "AdaLlamaFlashAttention",
        "AdaMistralFlashAttention",
        "AdaQwen2FlashAttention",
        "AdaQwen3FlashAttention",
        "AdaQwen3MoeFlashAttention",
    ]

    n = 0
    for name in names:
        cls = getattr(ada_attn, name, None)
        if cls is None:
            continue
        try:
            src = textwrap.dedent(inspect.getsource(cls.forward))
        except (OSError, TypeError):
            continue
        if "assert bsz == 1" not in src:
            continue

        patched_src = "\n".join(
            line for line in src.splitlines()
            if line.strip() != "assert bsz == 1"
        )
        namespace = dict(vars(ada_attn))
        exec(compile(patched_src, f"<batched:{name}>", "exec"), namespace)
        _ADA_ATTN_PATCHED.append((cls, "forward", cls.forward))
        cls.forward = namespace["forward"]
        n += 1
    return n


def disable_batched_ada_attention() -> None:
    """Restore the original ``forward`` methods."""
    while _ADA_ATTN_PATCHED:
        cls, attr, original = _ADA_ATTN_PATCHED.pop()
        setattr(cls, attr, original)
