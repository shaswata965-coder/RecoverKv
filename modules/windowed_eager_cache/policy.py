
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .config import ResolvedConfig


FIRST_EVICTION_STEP = 0


def _compact_by_rank(src: Tensor, mask: Tensor, width: int) -> Tensor:
    B = mask.shape[0]
    rank = mask.cumsum(1) - 1
    dump = torch.full_like(rank, width)
    idx = torch.where(mask & (rank < width), rank.clamp_min(0), dump)
    out = torch.full((B, width + 1), -1, dtype=src.dtype, device=src.device)
    out.scatter_(1, idx, src)
    return out[:, :width]


class EvictionPolicy:

    def __init__(self, resolved: ResolvedConfig) -> None:
        self.window_size: int = resolved.window_size
        self.num_sink_tokens: int = resolved.num_sink_tokens
        self.local_tokens: int = resolved.local_tokens
        self.local_windows: int = resolved.local_tokens // resolved.window_size
        self.top_k_windows: int = resolved.top_k_windows
        self.quant_ratio: float = resolved.quant_ratio
        self.top_k_fp: int = resolved.top_k_fp
        self.N_q: int = resolved.N_q
        self.quant_promotion: bool = resolved.quant_promotion
        self.total_tokens: int = 0
        self.first_eviction_step: int = resolved.first_eviction_step


    def initialize_after_prefill(self, prefill_len: int) -> None:
        self.total_tokens = prefill_len

    def extend_total_after_append(self, n_new: int) -> None:
        self.total_tokens += n_new

    def set_total_after_compaction(self, new_total: int) -> None:
        self.total_tokens = new_total


    def should_evict(self, step: int) -> bool:
        if step < self.first_eviction_step:
            return False
        if step == self.first_eviction_step:
            return True
        return step % self.window_size == 0


    @property
    def post_sink_tokens(self) -> int:
        return max(self.total_tokens - self.num_sink_tokens, 0)

    @property
    def num_total_windows(self) -> int:
        ps = self.post_sink_tokens
        return (ps + self.window_size - 1) // self.window_size if ps > 0 else 0

    @property
    def num_evictable_windows(self) -> int:
        return max(self.num_total_windows - self.local_windows, 0)


    def compute_retain_window_indices(
        self, window_scores: Tensor
    ) -> Tensor:
        B = window_scores.shape[0]
        W_total = window_scores.shape[2]
        device = window_scores.device

        local_w = min(self.local_windows, W_total)
        evictable_w = W_total - local_w

        mean_scores = window_scores.mean(dim=1)

        evictable_scores = mean_scores[:, :evictable_w]

        k = min(self.top_k_windows, evictable_w)

        if k == 0 or evictable_w == 0:
            local_idx = torch.arange(
                W_total - local_w, W_total, device=device, dtype=torch.long
            ).unsqueeze(0).expand(B, -1)
            return local_idx

        if k >= evictable_w:
            all_idx = torch.arange(
                W_total, device=device, dtype=torch.long
            ).unsqueeze(0).expand(B, -1)
            return all_idx

        _, topk_idx = torch.topk(evictable_scores, k, dim=-1)

        topk_sorted, _ = torch.sort(topk_idx, dim=-1)

        local_idx = torch.arange(
            W_total - local_w, W_total, device=device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)

        retained = torch.cat([topk_sorted, local_idx], dim=-1)
        return retained


    def tier_counts(self, W: int, n_q_resident: int = 0) -> "tuple[int, int, int]":
        local_w = min(self.local_windows, W)
        evictable_w = W - local_w
        if evictable_w <= 0:
            return 0, 0, local_w
        k_fp = min(self.top_k_fp, evictable_w)
        if not self.quant_promotion:
            k_fp = min(k_fp, max(evictable_w - n_q_resident, 0))
        n_q = min(self.N_q, evictable_w - k_fp)
        return k_fp, n_q, local_w

    @staticmethod
    def _sticky_tier_split(
        order: Tensor, q_resident: Tensor, k_fp: int, n_q: int
    ) -> "tuple[Tensor, Tensor]":
        is_q = torch.gather(q_resident, 1, order)
        fp_rank = (~is_q).cumsum(1) - 1
        gets_fp = ~is_q & (fp_rank < k_fp)
        q_rank = (~gets_fp).cumsum(1) - 1
        gets_q = ~gets_fp & (q_rank < n_q)
        return (
            _compact_by_rank(order, gets_fp, k_fp),
            _compact_by_rank(order, gets_q, n_q),
        )

    def compute_two_tier_retain(
        self,
        window_scores: Tensor,
        q_resident: "Optional[Tensor]" = None,
        n_q_resident: int = 0,
    ) -> "tuple[Tensor, Tensor]":
        B, _, W = window_scores.shape
        device = window_scores.device

        sticky_q = not self.quant_promotion and q_resident is not None
        k_fp, n_q, local_w = self.tier_counts(
            W, n_q_resident if sticky_q else 0
        )
        evictable_w = W - local_w
        local_idx = (
            torch.arange(W - local_w, W, device=device, dtype=torch.long)
            .unsqueeze(0).expand(B, -1)
        )
        local_tier = torch.zeros(B, local_w, device=device, dtype=torch.long)

        if evictable_w <= 0:
            return local_idx.contiguous(), local_tier

        mean = window_scores.mean(dim=1)
        ev_scores = mean[:, :evictable_w]

        order = torch.argsort(ev_scores, dim=-1, descending=True)
        if sticky_q:
            fp_sel, q_sel = self._sticky_tier_split(
                order, q_resident[:, :evictable_w], k_fp, n_q
            )
        else:
            fp_sel = order[:, :k_fp]
            q_sel = order[:, k_fp:k_fp + n_q]

        ev_idx = torch.cat([fp_sel, q_sel], dim=-1)
        ev_tier = torch.cat([
            torch.zeros(B, k_fp, device=device, dtype=torch.long),
            torch.ones(B, n_q, device=device, dtype=torch.long),
        ], dim=-1)
        perm = torch.argsort(ev_idx, dim=-1)
        ev_idx = torch.gather(ev_idx, 1, perm)
        ev_tier = torch.gather(ev_tier, 1, perm)

        retained = torch.cat([ev_idx, local_idx], dim=-1)
        tier = torch.cat([ev_tier, local_tier], dim=-1)
        return retained, tier


    def expand_to_token_indices(
        self, retained_window_idx: Tensor, num_windows: int
    ) -> Tensor:
        B, W_retained = retained_window_idx.shape
        device = retained_window_idx.device

        sink_idx = torch.arange(
            self.num_sink_tokens, device=device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)

        offsets = torch.arange(
            self.window_size, device=device, dtype=torch.long
        )

        token_idx = (
            self.num_sink_tokens
            + retained_window_idx.unsqueeze(-1) * self.window_size
            + offsets
        )
        token_idx = token_idx.reshape(B, -1)

        scored_end = self.num_sink_tokens + num_windows * self.window_size
        tail = max(self.total_tokens - scored_end, 0)
        pieces = [sink_idx, token_idx]
        if tail > 0:
            pieces.append(
                torch.arange(
                    scored_end, scored_end + tail, device=device, dtype=torch.long
                ).unsqueeze(0).expand(B, -1)
            )

        all_idx = torch.cat(pieces, dim=-1)

        valid_mask = all_idx < self.total_tokens

        if W_retained == 0:
            min_valid = min(self.num_sink_tokens, self.total_tokens) + tail
        else:
            last_start = self.num_sink_tokens + (num_windows - 1) * self.window_size
            last_valid = min(
                max(self.total_tokens - last_start, 0), self.window_size
            )
            oob = self.window_size - last_valid
            min_valid = (
                self.num_sink_tokens + W_retained * self.window_size - oob + tail
            )

        order = torch.argsort(~valid_mask, dim=1, stable=True)
        all_idx = torch.gather(all_idx, 1, order)[:, :min_valid]

        return all_idx
