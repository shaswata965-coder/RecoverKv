"""Batch planning for LongGenBench generation. Standard library only.

**A batch holds documents whose prompts have exactly the same token length.**
``WindowedCache`` supports ``B > 1`` only for equal-length rows: it keeps every
row's tiers dense and rectangular, and ragged / left-padded batches are not
implemented. A padding mask that reached flash-attention would also route the
model through ``flash_attn_varlen_func``, which the fused decode path refuses
(``flash_decode.FusedDecodeNotReached``) rather than silently attending over the
fp tier alone. So documents are never padded: they are grouped by prompt length,
and each group is cut into batches of at most the batch size, as evenly as
possible (a batch decodes until its longest row ends, so an even split wastes
less on stragglers than 16 + 2).

At 16K the 400 prompts take 54 distinct lengths, so a cap of 16 gives 58
batches of 6.9 documents on average (6 full); 8 gives 79, 6 gives 92.

The full-KV arm uses the same planner, so both arms see the same kind of batch.

Batch size: the largest of ``BATCH_SIZES`` (16, 8, 6) whose estimated peak fits
the GPU, chosen per arm before anything runs (:func:`pick_batch_size`), and the
runner falls back down the same list if a batch still runs out of memory.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

BATCH_SIZES = (16, 8, 6)

#: Llama-3.1-8B-Instruct: KV bytes per token over all layers (32 layers x 8 KV
#: heads x 128 dims x K and V x 2 bytes) and fp16 weight bytes.
KV_BYTES_PER_TOKEN = 32 * 8 * 128 * 2 * 2
WEIGHT_GIB = 8.03e9 * 2 / 2 ** 30


def plan_batches(lengths: Dict[int, int], max_batch: int,
                 order: Optional[Sequence[int]] = None) -> List[List[int]]:
    """Equal-length batches of at most ``max_batch`` documents.

    ``lengths``: ``{document id: prompt token length}`` (BOS included).
    ``order``: the ids in the order groups should appear (default: by id); the
    first appearance of a length places its group. Within a group ids are sorted.
    """
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    ids = list(order) if order is not None else sorted(lengths)
    groups: Dict[int, List[int]] = defaultdict(list)
    seq: List[int] = []
    for i in ids:
        n = lengths[i]
        if n not in groups:
            seq.append(n)
        groups[n].append(i)
    out: List[List[int]] = []
    for n in seq:
        g = sorted(groups[n])
        k = -(-len(g) // max_batch)                       # batches for this length
        base, extra = divmod(len(g), k)
        start = 0
        for j in range(k):
            size = base + (1 if j < extra else 0)
            out.append(g[start:start + size])
            start += size
    return out


def split_evenly(ids: Sequence[int], max_batch: int) -> List[List[int]]:
    """``ids`` cut into ``ceil(n / max_batch)`` near-equal batches (OOM fallback)."""
    ids = list(ids)
    k = -(-len(ids) // max_batch)
    base, extra = divmod(len(ids), k)
    out, start = [], 0
    for j in range(k):
        size = base + (1 if j < extra else 0)
        out.append(ids[start:start + size])
        start += size
    return out


def next_smaller(size: int, sizes: Iterable[int] = BATCH_SIZES) -> Optional[int]:
    """The largest allowed batch size strictly below ``size``, or None."""
    smaller = [s for s in sizes if s < size]
    return max(smaller) if smaller else None


def estimate_peak_gib(arm: str, batch: int, prompt_len: int, max_new: int,
                      cache_budget: Optional[float] = None) -> float:
    """Arithmetic, deliberately generous, estimate of a batch's peak GPU memory.

    Full KV holds every token of every row at the end. Ours holds
    ``cache_budget`` of the (prompt + max_new_tokens) bytes per row (bytes mode),
    plus the uncompressed prompt before the first eviction; the factor 2 covers
    the eviction's working copies. 3 GiB covers prefill activations, logits and
    allocator slack. It picks where to START; the runner still falls back if a
    batch runs out of memory.
    """
    seq = prompt_len + max_new
    if arm == "fullkv":
        kv = batch * seq * KV_BYTES_PER_TOKEN * 1.1      # + DynamicCache's per-step cat
    else:
        if cache_budget is None:
            raise ValueError("ours needs cache_budget")
        kv = batch * (2 * cache_budget * seq + prompt_len) * KV_BYTES_PER_TOKEN
    return WEIGHT_GIB + kv / 2 ** 30 + 3.0


def pick_batch_size(arm: str, gpu_gib: float, prompt_len: int, max_new: int,
                    cache_budget: Optional[float] = None, sizes: Sequence[int] = BATCH_SIZES,
                    headroom: float = 0.9) -> Optional[int]:
    """The largest size in ``sizes`` whose estimate fits ``headroom`` of the GPU."""
    for s in sorted(sizes, reverse=True):
        if estimate_peak_gib(arm, s, prompt_len, max_new, cache_budget) <= headroom * gpu_gib:
            return s
    return None
