"""The gated read path: dequantize the selected windows, score all of them.

Three things have to hold for the gate to be the *only* read path rather than a
variant of one:

* the gated Q tier is bit-comparable to the selected columns of the ungated one
  (gating changes WHICH windows are read, never what a read produces);
* every skipped window still gets a score, rescaled onto the same footing as the
  real ones -- otherwise it ranks last and eviction drops the tier the gate
  exists to read less often;
* selection is per KV head, because the alternatives measurably do not work.

CPU-only; no GPU, no Triton, no model forward.
"""

from __future__ import annotations

import math

import pytest
import torch

from modules.quant.sketch import (
    Sketch, build_sketch, gate_and_score, group_max, select_windows,
)
from modules.quant.store import QuantizedStore
from modules.windowed_cache.scorer import (
    expand_keep_to_query_heads, fill_skipped_window_scores,
)

B, N, H, S, D = 2, 8, 4, 8, 16
SCALING = 1.0 / math.sqrt(D)


class _RoPE(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.inv = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))

    def forward(self, x, pos):
        f = (pos.float().unsqueeze(-1) * self.inv).to(x.dtype)
        e = torch.cat([f, f], -1)
        return e.cos(), e.sin()


def _loaded(n=N, seed=0):
    st = QuantizedStore(S, D, H, n_slots=32, memoize_read=False, sketch_enabled=True)
    st.ensure(B, torch.device("cpu"))
    g = torch.Generator().manual_seed(seed)
    k_post = torch.randn(B, n, H, S, D, generator=g)
    k_post[:, :, :, 0, :] += torch.randn(B, n, H, D, generator=g) * 6
    st.demote_many(
        st.table.free_slots(n), torch.ones(B, n, dtype=torch.bool),
        torch.arange(n).reshape(1, n).expand(B, n).contiguous(),
        torch.randn(B, n, H, S, D, generator=g),
        torch.randn(B, n, H, S, D, generator=g),
        torch.arange(n * S).reshape(1, n, S).expand(B, n, S).contiguous(),
        keys_post_rope=k_post,
    )
    st.commit_active_count(n)
    return st


# ---------------------------------------------------------------------------
# the gated read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.25, 0.5, 1.0])
def test_gated_tier_equals_the_selected_columns_of_the_full_tier(ratio):
    """Gating decides WHICH windows are read; it never changes what a read gives."""
    st = _loaded()
    rope = _RoPE(D)
    keep, _, slots = st.gate_and_select(
        torch.randn(B, H * 2, D, generator=torch.Generator().manual_seed(1)),
        SCALING, ratio=ratio)
    k_g, v_g, p_g = st.gated_q_tier(keep, slots, rope, torch.float32)
    k_f, v_f, p_f = st.effective_q_tier(rope, torch.float32)

    n_sel = int(keep[0, 0].sum())
    assert k_g.shape == (B, H, n_sel * S, D)
    pick = torch.argsort(~keep, dim=-1, stable=True)[..., :n_sel]
    for b in range(B):
        for h in range(H):
            for j, w in enumerate(pick[b, h].tolist()):
                sl, fl = slice(j * S, (j + 1) * S), slice(w * S, (w + 1) * S)
                assert torch.allclose(k_g[b, h, sl], k_f[b, h, fl], atol=1e-5)
                assert torch.allclose(v_g[b, h, sl], v_f[b, h, fl], atol=1e-5)
            assert torch.equal(p_g[b, h],
                               p_f[b].reshape(-1, S)[pick[b, h]].reshape(-1))


def test_gated_tier_reads_exactly_the_ratio():
    st = _loaded(n=16)
    keep, _, slots = st.gate_and_select(torch.randn(B, H * 2, D), SCALING, ratio=0.25)
    k_g, _, _ = st.gated_q_tier(keep, slots, _RoPE(D), torch.float32)
    assert k_g.shape[2] == math.ceil(0.25 * 16) * S


def test_counts_are_equal_across_rows_and_heads():
    """What keeps the effective K/V dense with no padding or keep-mask."""
    st = _loaded(n=16)
    keep, _, _ = st.gate_and_select(torch.randn(B, H * 2, D), SCALING, ratio=0.3)
    assert keep.sum(-1).unique().numel() == 1


# ---------------------------------------------------------------------------
# scoring every window, including the ones never read
# ---------------------------------------------------------------------------


def test_selected_windows_keep_their_exact_scores():
    g = torch.Generator().manual_seed(2)
    exact = torch.rand(B, H * 2, N, generator=g) * 3
    keep = torch.rand(B, H * 2, N, generator=g) < 0.4
    out = fill_skipped_window_scores(exact, keep, exact.clamp_min(1e-6).log())
    assert torch.equal(out[keep], exact[keep])


def test_skipped_windows_are_credited_not_zeroed():
    """The starvation bug, and the fix, as one comparison.

    Leaving a skipped window at zero collapses the correlation between the score
    eviction ranks on and the truth; crediting the card's estimate keeps it.
    """
    g = torch.Generator().manual_seed(3)
    true = torch.rand(B, H * 2, 64, generator=g) * 3
    logmass = true.clamp_min(1e-6).log() + torch.randn(B, H * 2, 64, generator=g) * 0.05
    keep = torch.rand(B, H * 2, 64, generator=g) < 0.25
    keep |= torch.nn.functional.one_hot(true.argmax(-1), 64).bool()

    filled = fill_skipped_window_scores(true, keep, logmass)
    zeroed = torch.where(keep, true, torch.zeros_like(true))
    corr = lambda x: torch.corrcoef(
        torch.stack([x.flatten(), true.flatten()]))[0, 1].item()
    assert corr(filled) > 0.99
    assert corr(zeroed) < 0.5
    assert 0.95 < filled.sum(-1).mean() / true.sum(-1).mean() < 1.05


def test_calibration_puts_both_populations_on_one_scale():
    """A constant offset in the card's estimate must not shift the ranking."""
    g = torch.Generator().manual_seed(4)
    true = torch.rand(B, H, 32, generator=g) * 3
    keep = torch.rand(B, H, 32, generator=g) < 0.3
    keep |= torch.nn.functional.one_hot(true.argmax(-1), 32).bool()
    base = true.clamp_min(1e-6).log()
    a = fill_skipped_window_scores(true, keep, base)
    b = fill_skipped_window_scores(true, keep, base + 7.5)   # same shape, offset
    assert torch.allclose(a, b, atol=1e-4)


def test_expand_keep_covers_the_query_head_group():
    keep = torch.zeros(1, 4, 5, dtype=torch.bool)
    keep[0, 1, 2] = True
    e = expand_keep_to_query_heads(keep, 8)
    assert e.shape == (1, 8, 5) and e[0, 2, 2] and e[0, 3, 2] and e.sum() == 2


# ---------------------------------------------------------------------------
# why selection is per KV head
# ---------------------------------------------------------------------------


def test_per_head_selection_beats_both_row_level_alternatives():
    """Pins the measurement that fixed the granularity.

    At a 0.25 ratio on a shape-C tier: per-KV-head reads 68/271 windows at 100%
    worst-head mass recall; a row-level union of the same picks reads 240/271
    (the gate buys nothing); a row-level top-k reads 68 but drops to ~97.5%.
    """
    hkv, hq, nw = 8, 32, 271
    g = torch.Generator().manual_seed(5)
    a = torch.randn(hkv, 128, generator=g) * 0.4
    a[:, 3] += 40.0
    k = a[None, :, None, :] + torch.randn(nw, hkv, S, 128, generator=g)
    k[:, :, 0, :] += torch.randn(nw, hkv, 128, generator=g) * 6
    q = torch.randn(1, hq, 128, generator=g) * 1.5
    sc = 1.0 / math.sqrt(128)
    card = Sketch(*[f.unsqueeze(0) for f in build_sketch(k, a)])
    bound, _, est = gate_and_score(q, card, a, sc)
    true = sc * torch.einsum("nhwd,bhrd->bhrnw", k,
                             q.reshape(1, hkv, hq // hkv, 128)).reshape(1, hq, nw, S)
    mass = true.logsumexp(-1).reshape(1, hkv, hq // hkv, nw).logsumexp(2).exp()
    cap = math.ceil(0.25 * nw)

    def recall(keep):
        return ((mass * keep).sum(-1) / mass.sum(-1)).min().item()

    per_head = select_windows(group_max(bound, hkv), float("inf"), cap,
                              rank_by=group_max(est, hkv))
    union = per_head.any(1, keepdim=True).expand_as(per_head)
    row_topk = select_windows(group_max(bound, hkv).amax(1, keepdim=True),
                              float("inf"), cap,
                              rank_by=group_max(est, hkv).amax(1, keepdim=True))

    assert per_head.sum(-1).max() == cap
    assert recall(per_head) > 0.999
    assert union.sum(-1).float().mean() > 3 * cap        # union defeats the gate
    assert recall(row_topk) < recall(per_head)           # same cost, worse recall
