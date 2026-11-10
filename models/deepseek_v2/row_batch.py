"""Left-padded batches over per-row caches, for the owned MLA forward.

A batch of prompts with different lengths is left-padded, as in DefensiveKV's
batched SnapKV script: row ``i`` carries ``pad_i`` pad tokens, then its own
tokens at positions ``0 .. L_i - 1``. Every non-attention layer runs on the
padded batch as one tensor. Attention is where padding matters: a pad key must
be invisible to every query, and each row's cache must hold only that row's
tokens, with that row's own budget (QEvict sizes the budget from the row's
prompt length, so rows of one batch have different budgets).

:class:`RowBatchCache` is the ``past_key_values`` for such a batch. It holds one
cache per row -- a :class:`~modules.windowed_cache.cache.WindowedCache` for the
method, or any HF ``Cache`` for the latent FullKV reference -- and, for the
current forward call, the index of each row's first real token. The owned
attention forward (:func:`attend_rows`) splits the call by row: row ``i``'s
queries, keys and values are its own tokens ``[start_i:]`` only, and they go
through exactly the single-row path a batch of one takes (prefill, score,
eviction, gated fused decode). Pad positions get a zero attention output, which
is what flash attention's unpad/pad path hands back for them.

So a padding mask is never needed and never read: a row attends over its own
cache, which never received a pad key. That is the same set of keys the
additive padding mask leaves visible in the batched SnapKV script, reached by
construction instead of by masking.

A row whose call is all padding (``start_i == T``; a short question's leading
pad columns) is skipped: its cache is not touched and its step count does not
advance, so each row sees its own tokens in order, at its own positions.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch
from torch import Tensor

try:  # the HF base class, so the remote model keeps the object as is
    from transformers.cache_utils import Cache as _HFCache
except ImportError:  # pragma: no cover - transformers is a hard dependency
    _HFCache = object


class RowBatchCache(_HFCache):
    """One cache per row of a left-padded batch (see the module docstring)."""

    def __init__(self, rows: Sequence[Any]) -> None:
        try:
            super().__init__()
        except (TypeError, ValueError):
            pass
        self.rows: List[Any] = list(rows)
        if not self.rows:
            raise ValueError("RowBatchCache needs at least one row")
        self._starts: Optional[List[int]] = None

    # -- the per-call row map ----------------------------------------------
    def begin_call(self, starts: Sequence[int]) -> None:
        """Set, for the next forward call, each row's first real column."""
        starts = [int(s) for s in starts]
        if len(starts) != len(self.rows):
            raise ValueError(f"{len(starts)} row starts for {len(self.rows)} rows")
        if any(s < 0 for s in starts):
            raise ValueError(f"negative row start in {starts}")
        self._starts = starts

    def end_call(self) -> None:
        self._starts = None

    @property
    def starts(self) -> List[int]:
        if self._starts is None:
            raise RuntimeError(
                "RowBatchCache reached attention without begin_call(): each forward "
                "call must say where every row's real tokens start.")
        return self._starts

    # -- what the remote model asks of a Cache -------------------------------
    # Positions are always passed explicitly (per row), so the remote model's
    # past-length bookkeeping is never used to place a token.
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return 0

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return 0

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def update(self, *args, **kwargs):
        raise RuntimeError(
            "RowBatchCache.update() was called directly: it is consumed only by the "
            "owned MLA forward, which updates each row's own cache.")


def attend_rows(module, hidden_states: Tensor, position_ids: Tensor,
                cache: RowBatchCache, forward) -> tuple:
    """Run ``forward`` (the owned single-row MLA forward) once per row.

    ``forward(module, hidden_states, position_ids=..., past_key_value=row_cache)``
    is :func:`models.deepseek_v2.attention.mla_attention_forward`.
    """
    from modules.windowed_cache import flash_decode

    bsz, q_len, _ = hidden_states.shape
    starts = cache.starts
    if len(starts) != bsz:
        raise RuntimeError(f"{bsz}-row call against a {len(starts)}-row RowBatchCache")
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.shape[0] != bsz:
        position_ids = position_ids.expand(bsz, -1)
    out = None
    for i, (s, row_cache) in enumerate(zip(starts, cache.rows)):
        if s > q_len:
            raise RuntimeError(f"row {i} starts at column {s} of a {q_len}-column call")
        if s == q_len:
            continue
        # The per-layer pre-hook's job, per row: a fused context armed by one row
        # must have been consumed before the next row's cache can arm its own.
        flash_decode.clear()
        o, _, _ = forward(module, hidden_states[i:i + 1, s:],
                          position_ids=position_ids[i:i + 1, s:],
                          past_key_value=row_cache)
        if out is None:
            out = hidden_states.new_zeros(bsz, q_len, o.shape[-1])
        out[i, s:] = o[0].to(out.dtype)
    if out is None:
        out = hidden_states.new_zeros(bsz, q_len, module.o_proj.out_features)
    return out, None, cache
