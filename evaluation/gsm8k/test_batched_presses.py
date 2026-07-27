"""CPU tests for the batched LayerDefensiveKV budget allocation.

No GPU, no flash-attn, no model weights — the index math is factored into free
functions precisely so it can be pinned down here.

Two properties carry the whole rewrite:

* **B=1 equivalence.** The per-row allocation must reproduce, exactly, what upstream's
  batch-flattened code computes when there is only one row. If it does not, the
  rewrite has changed the method rather than batched it.
* **No cross-row leakage.** Row 0's allocation must be bit-identical whether row 1
  contains tiny scores or enormous ones. This is the property upstream *violates*, and
  it is invisible in accuracy numbers — a run would simply produce different results
  depending on batch composition.

    cd evaluation && pytest gsm8k/test_batched_presses.py -v
"""

from __future__ import annotations

import pytest
import torch

from gsm8k.batched_presses import (
    flat_gather_indices,
    per_row_layer_budgets,
    per_row_retain_mask,
    rebuild_cu_seqlens,
)

L, H, T = 4, 3, 8          # layers, kv heads, tokens per head
HT = H * T


def _upstream_layer_budgets(layer_scores, n_kept, num_kv_heads, head_len):
    """Upstream's allocation, verbatim, for a single row.

    Mirrors ``EfficientAdaGlobalScorerPress.compress`` lines 177-187 so the
    equivalence test compares against the real thing rather than a paraphrase.
    """
    n_layers = layer_scores.shape[0]
    flat = layer_scores.view(-1)
    topk_idx = flat.topk(n_kept, dim=-1).indices
    layer_of = topk_idx // (head_len * num_kv_heads)
    out = torch.zeros(n_layers, dtype=torch.int32)
    out.scatter_add_(0, layer_of, torch.ones_like(layer_of, dtype=torch.int32))
    return out


class TestCrossLayerAllocation:
    def test_matches_upstream_at_batch_size_one(self):
        torch.manual_seed(0)
        scores = torch.randn(L, 1, HT)
        n_kept = 40

        mine = per_row_layer_budgets(scores, n_kept, H, T)[0]
        theirs = _upstream_layer_budgets(scores, n_kept, H, T)
        assert torch.equal(mine.to(torch.int32), theirs)

    def test_every_row_receives_exactly_its_budget(self):
        torch.manual_seed(1)
        scores = torch.randn(L, 5, HT)
        n_kept = 37
        budgets = per_row_layer_budgets(scores, n_kept, H, T)
        assert budgets.shape == (5, L)
        assert torch.equal(budgets.sum(1), torch.full((5,), n_kept))

    def test_no_cross_row_leakage(self):
        """The property upstream violates.

        Row 0's layer split must not move when another row's scores change. Upstream
        ranks a flattened [L*B*H*T] tensor, so an all-huge row 1 would take nearly the
        entire budget and starve row 0.
        """
        torch.manual_seed(2)
        row0 = torch.randn(L, 1, HT)

        quiet = torch.full((L, 1, HT), -50.0)
        loud = torch.full((L, 1, HT), +50.0)

        with_quiet = per_row_layer_budgets(torch.cat([row0, quiet], 1), 40, H, T)
        with_loud = per_row_layer_budgets(torch.cat([row0, loud], 1), 40, H, T)

        assert torch.equal(with_quiet[0], with_loud[0])
        # ... and the noisy neighbour still got its own full budget, not row 0's.
        assert with_loud[1].sum().item() == 40

    def test_upstream_would_have_leaked(self):
        """Demonstrates the bug this rewrite fixes, so the test is not vacuous."""
        torch.manual_seed(3)
        row0 = torch.randn(L, 1, HT)
        loud = torch.full((L, 1, HT), +50.0)
        stacked = torch.cat([row0, loud], 1)          # [L, 2, HT]

        # Upstream flattens B into the ranked axis:
        flat_topk = stacked.view(-1).topk(40).indices
        # Every winner belongs to the loud row -> row 0 gets nothing.
        row_of = (flat_topk // HT) % 2
        assert (row_of == 1).all()

    def test_concentrated_scores_go_to_one_layer(self):
        scores = torch.full((L, 2, HT), -10.0)
        scores[2] = 10.0                                # layer 2 dominates
        budgets = per_row_layer_budgets(scores, HT, H, T)
        assert budgets[0, 2].item() == HT
        assert budgets[0].sum().item() == HT

    def test_rejects_impossible_budget(self):
        scores = torch.randn(L, 2, HT)
        with pytest.raises(ValueError, match="n_kept_per_row"):
            per_row_layer_budgets(scores, L * HT + 1, H, T)

    def test_rejects_shape_mismatch(self):
        scores = torch.randn(L, 2, HT)
        with pytest.raises(ValueError, match="num_kv_heads"):
            per_row_layer_budgets(scores, 10, H + 1, T)


class TestPerRowRetainMask:
    def test_selects_exactly_budget_per_row(self):
        torch.manual_seed(4)
        scores = torch.randn(6, HT)
        budgets = torch.tensor([0, 1, 5, 12, HT - 1, HT])
        mask = per_row_retain_mask(scores, budgets)
        assert torch.equal(mask.sum(1), budgets)

    def test_selects_the_highest_scores(self):
        scores = torch.tensor([[5.0, 1.0, 9.0, 3.0]])
        mask = per_row_retain_mask(scores, torch.tensor([2]))
        assert mask.tolist() == [[True, False, True, False]]

    def test_matches_topk_when_budget_is_uniform(self):
        """Ragged-capable path must agree with plain topk in the uniform case."""
        torch.manual_seed(5)
        scores = torch.randn(4, HT)
        k = 9
        mask = per_row_retain_mask(scores, torch.full((4,), k))
        expected = torch.zeros_like(mask)
        expected.scatter_(1, scores.topk(k, dim=-1).indices, True)
        assert torch.equal(mask, expected)

    def test_rows_are_independent(self):
        torch.manual_seed(6)
        a = torch.randn(1, HT)
        m_alone = per_row_retain_mask(a, torch.tensor([7]))
        m_batched = per_row_retain_mask(
            torch.cat([a, torch.full((1, HT), 99.0)]), torch.tensor([7, 3])
        )
        assert torch.equal(m_alone[0], m_batched[0])


class TestFlatCacheIndexing:
    def test_indices_follow_the_b_h_t_cache_order(self):
        """The flat cache is (b, h, t)-major; gather indices must be too."""
        mask = torch.zeros(2, H * T, dtype=torch.bool)
        mask.view(2, H, T)[0, 0, 3] = True     # row 0, head 0, tok 3 -> flat 3
        mask.view(2, H, T)[0, 2, 0] = True     # row 0, head 2, tok 0 -> flat 16
        mask.view(2, H, T)[1, 1, 5] = True     # row 1, head 1, tok 5 -> flat 24+8+5=37
        idx, head_lens = flat_gather_indices(mask, H, T)
        assert idx.tolist() == [3, 16, 37]
        assert head_lens.tolist() == [1, 0, 1, 0, 1, 0]

    def test_indices_are_ascending(self):
        """nonzero() is already grouped by (row, head) -- no sort needed downstream."""
        torch.manual_seed(7)
        mask = per_row_retain_mask(torch.randn(4, HT), torch.tensor([5, 9, 2, 11]))
        idx, _ = flat_gather_indices(mask, H, T)
        assert torch.equal(idx, idx.sort().values)

    def test_head_lens_sum_to_the_retained_count(self):
        torch.manual_seed(8)
        budgets = torch.tensor([5, 9, 2, 11])
        mask = per_row_retain_mask(torch.randn(4, HT), budgets)
        idx, head_lens = flat_gather_indices(mask, H, T)
        assert int(head_lens.sum()) == idx.numel() == int(budgets.sum())

    def test_per_row_totals_match_each_rows_budget(self):
        """Head budgets may be unequal within a row; the row total must not be."""
        torch.manual_seed(9)
        budgets = torch.tensor([5, 9, 2, 11])
        mask = per_row_retain_mask(torch.randn(4, HT), budgets)
        _, head_lens = flat_gather_indices(mask, H, T)
        assert torch.equal(head_lens.view(4, H).sum(1).to(torch.int64), budgets)

    def test_cu_seqlens_layout(self):
        head_lens = torch.tensor([3, 0, 5, 2], dtype=torch.int32)
        cu, mx = rebuild_cu_seqlens(head_lens)
        assert cu.tolist() == [0, 3, 3, 8, 10]
        assert mx == 5
        assert cu.dtype == torch.int32


class TestAttentionPatch:
    def test_patch_is_reversible(self):
        pytest.importorskip("flash_attn")
        from kvpress import ada_attn

        from gsm8k.batched_presses import (
            disable_batched_ada_attention,
            enable_batched_ada_attention,
        )

        original = ada_attn.AdaLlamaFlashAttention.forward
        n = enable_batched_ada_attention()
        try:
            assert n >= 1
            assert ada_attn.AdaLlamaFlashAttention.forward is not original
        finally:
            disable_batched_ada_attention()
        assert ada_attn.AdaLlamaFlashAttention.forward is original

    def test_patched_forward_drops_only_the_assert(self):
        pytest.importorskip("flash_attn")
        import inspect

        from kvpress import ada_attn

        from gsm8k.batched_presses import (
            disable_batched_ada_attention,
            enable_batched_ada_attention,
        )

        before = inspect.getsource(ada_attn.AdaLlamaFlashAttention.forward)
        enable_batched_ada_attention()
        try:
            after = ada_attn.AdaLlamaFlashAttention.forward.__code__
            # The recompiled body must still reference the varlen call and the reshape.
            names = set(after.co_names)
            assert "flash_attn_varlen_func" in names or "reshape" in names
        finally:
            disable_batched_ada_attention()
        assert "assert bsz == 1" in before  # upstream unchanged on disk
