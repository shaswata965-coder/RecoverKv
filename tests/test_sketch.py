"""Rank-1 window sketches — the properties the gate's correctness rests on.

The load-bearing test is :func:`test_bound_never_misses`: the gate is only safe
because ``bound`` is a genuine upper bound on the window's largest logit, for the
values ACTUALLY STORED (quantised mu / v / t / eps), not for the ideal rank-1
fit. Everything else supports it or pins an encoding decision that was made by
measurement rather than by argument.

CPU-only; no GPU, no Triton.
"""

from __future__ import annotations

import math

import pytest
import torch

from modules.quant.sketch import (
    _dq_sym,
    _q_sym,
    build_sketch,
    decode_sketch,
    gate_and_score,
    group_max,
    select_windows,
    sketch_bytes_per_head,
)

D, WS, HKV = 128, 8, 4


def _fixture(n=64, h=HKV, ws=WS, d=D, hot=6.0, spread=1.0, seed=0):
    """Keys shaped like real ones: a per-head common mode with massive-activation
    channels, plus (optionally) one outlier token per window."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(h, d, generator=g) * 0.4
    a[:, 3] += 40.0
    a[:, 17] -= 25.0
    k = a[None, :, None, :] + torch.randn(n, h, ws, d, generator=g) * spread
    if hot:
        k[:, :, 0, :] += torch.randn(n, h, d, generator=g) * hot
    return k, a


def _batched(s, b):
    return type(s)(*[f.unsqueeze(0).expand(b, *f.shape).contiguous() for f in s])


def _true_logits(k, q, hkv=HKV):
    """``[B, Hq, N, ws]`` exact ``q.k`` for every token of every window."""
    b, hq, _ = q.shape
    qg = q.reshape(b, hkv, hq // hkv, D)
    return torch.einsum("nhwd,bhrd->bhrnw", k, qg).reshape(b, hq, k.shape[0], k.shape[2])


# ---------------------------------------------------------------------------
# the guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hot", [0.0, 6.0, 20.0])
@pytest.mark.parametrize("spread", [0.25, 1.0, 4.0])
def test_bound_never_misses(hot, spread):
    """``bound >= max_i scaling * q.k_i`` for every window, head and query.

    Checked against the true keys using the DECODED card. This is what makes a
    skipped window safe to skip.
    """
    k, a = _fixture(hot=hot, spread=spread)
    s = _batched(build_sketch(k, a), 3)
    q = torch.randn(3, HKV * 2, D, generator=torch.Generator().manual_seed(1)) * 1.5
    scaling = 1.0 / math.sqrt(D)
    bound, _, _ = gate_and_score(q, s, a, scaling)
    true = (scaling * _true_logits(k, q)).amax(-1)
    assert (true - bound).amax() <= 1e-4, f"violated by {(true - bound).amax():.6f}"


def test_bound_holds_when_every_token_is_identical():
    """Degenerate window: zero deviation, zero residual, no NaN, bound intact."""
    _, a = _fixture()
    k = a[None, :, None, :].expand(4, HKV, WS, D).contiguous()
    s = build_sketch(k, a)
    for f in decode_sketch(s, a):
        assert torch.isfinite(f).all()
    q = torch.randn(1, HKV, D, generator=torch.Generator().manual_seed(2))
    bound, logmass, _ = gate_and_score(q, _batched(s, 1), a, 1.0 / math.sqrt(D))
    assert torch.isfinite(bound).all() and torch.isfinite(logmass).all()
    true = (_true_logits(k, q) / math.sqrt(D)).amax(-1)
    assert (true - bound).amax() <= 1e-4


def test_hot_token_is_the_best_fit_token():
    """``v`` points at the farthest token, so that token fits best.

    This is the whole reason a rank-1 card beats a mean on a 1-hot-in-8 window:
    the model spends its one direction on the token the gate exists to find.
    """
    k, a = _fixture(n=128)
    _, _, _, eps = decode_sketch(build_sketch(k, a), a)
    far = (k - k.mean(-2, keepdim=True)).norm(dim=-1).argmax(-1)
    e_far = eps.gather(-1, far.unsqueeze(-1)).squeeze(-1)
    assert (e_far < eps.median(dim=-1).values).float().mean() > 0.99


def test_rank1_beats_centroid_on_reconstruction():
    k, a = _fixture(n=128)
    mu_h, v_h, t_h, _ = decode_sketch(build_sketch(k, a), a)
    rank1 = (k - (mu_h.unsqueeze(-2) + t_h.unsqueeze(-1) * v_h.unsqueeze(-2))).norm(-1)
    centroid = (k - k.mean(-2, keepdim=True)).norm(dim=-1)
    assert rank1.mean() < 0.6 * centroid.mean()


def test_int8_direction_is_required_not_a_preference():
    """Pins the measurement that killed the 1-bit and int4 variants.

    A coarser ``v`` leaves the hot token with a residual no better than the
    window mean — i.e. the card stops doing the one thing it exists to do. Guard
    it so nobody 'optimises' those bytes back out.
    """
    k, a = _fixture(n=128)
    mu = k.mean(-2)
    dev = k - mu.unsqueeze(-2)
    nrm = dev.norm(dim=-1)
    far = nrm.argmax(-1)
    v = torch.gather(dev, -2, far[..., None, None].expand(*k.shape[:2], 1, D)
                     ).squeeze(-2) / torch.gather(nrm, -1, far[..., None]).clamp_min(1e-12)
    t = (dev * v.unsqueeze(-2)).sum(-1)
    t_h = _dq_sym(*_q_sym(t))
    mu_h = a.unsqueeze(0) + _dq_sym(*_q_sym(mu - a.unsqueeze(0)))

    def hot_residual(v_h):
        rec = mu_h.unsqueeze(-2) + t_h.unsqueeze(-1) * v_h.unsqueeze(-2)
        e = (k - rec).norm(dim=-1)
        return e.gather(-1, far[..., None]).mean().item(), e.mean().item()

    hot8, mean8 = hot_residual(_dq_sym(*_q_sym(v, 127)))
    hot4, _ = hot_residual(_dq_sym(*_q_sym(v, 7)))
    sign = torch.sign(v)
    sign[sign == 0] = 1.0
    hot1, _ = hot_residual(sign / math.sqrt(D))

    assert hot8 < 0.1 * mean8, f"int8 v should nail the hot token: {hot8} vs {mean8}"
    assert hot4 > 5 * hot8, "int4 v was expected to be much worse; re-measure"
    assert hot1 > 5 * hot8, "1-bit v was expected to be much worse; re-measure"


def test_anchor_removes_the_common_mode():
    """Massive-activation channels must not eat the int8 range."""
    k, a = _fixture(n=128)
    mu = k.mean(-2)
    with_a = decode_sketch(build_sketch(k, a), a)[0]
    zero = torch.zeros_like(a)
    without = decode_sketch(build_sketch(k, zero), zero)[0]
    assert (mu - with_a).abs().mean() < 0.2 * (mu - without).abs().mean()


def test_logmass_tracks_true_mass():
    """The score written back for a skipped window has to be usable for ranking."""
    k, a = _fixture(n=128)
    q = torch.randn(4, HKV, D, generator=torch.Generator().manual_seed(3))
    scaling = 1.0 / math.sqrt(D)
    _, logmass, _ = gate_and_score(q, _batched(build_sketch(k, a), 4), a, scaling)
    true = torch.logsumexp(scaling * _true_logits(k, q), dim=-1)
    err = (logmass - true).abs()
    assert err.mean() < 0.35 and err.max() < 3.0
    # Ranking is what eviction consumes, so check order, not just magnitude.
    rho = torch.corrcoef(torch.stack([logmass.flatten(), true.flatten()]))[0, 1]
    assert rho > 0.99


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_infinite_margin_is_a_no_op():
    """How the feature ships dark."""
    assert select_windows(torch.randn(2, 8, 40), float("inf")).all()


def test_selection_always_keeps_the_best_window():
    b = torch.randn(2, 8, 40)
    assert select_windows(b, margin=0.0).gather(-1, b.argmax(-1, keepdim=True)).all()


def test_margin_is_monotone_in_selectivity():
    b = torch.randn(4, 8, 60)
    counts = [select_windows(b, m).sum().item() for m in (0.5, 2.0, 8.0, 32.0)]
    assert counts == sorted(counts)


def test_cap_bounds_the_work():
    assert select_windows(torch.randn(2, 8, 40), float("inf"),
                          max_windows=5).sum(-1).max() <= 5


def test_peaked_head_selects_fewer_than_flat_head():
    """Adaptivity with no calibration: the threshold is per-head relative."""
    g = torch.Generator().manual_seed(4)
    flat = torch.randn(1, 1, 64, generator=g) * 0.05
    peaked = torch.randn(1, 1, 64, generator=g) * 0.05
    peaked[..., 0] += 10.0
    assert select_windows(peaked, 2.0).sum() < select_windows(flat, 2.0).sum()


def test_group_max_is_the_gqa_contract():
    """A window any query head in the group wants survives for the whole group."""
    b = torch.full((1, 8, 5), -10.0)
    b[0, 3, 2] = 5.0                          # one query head of group 1 wants w2
    g = group_max(b, num_kv_heads=4)
    assert g.shape == (1, 4, 5) and g[0, 1, 2] == 5.0
    assert select_windows(g, margin=1.0)[0, 1, 2]


def test_cap_must_bind_after_the_group_union_not_before():
    """Capping per query head then unioning inflates the ratio by up to `rep`.

    At rep=4 a 25% cap would become 100% of windows actually loaded, and the gate
    would buy nothing. Pins the ordering that prevents it.
    """
    g = torch.Generator().manual_seed(21)
    bound = torch.randn(1, 8, 40, generator=g)          # Hq=8, Hkv=2 -> rep=4
    wrong = select_windows(bound, float("inf"), max_windows=10)
    wrong = wrong.reshape(1, 2, 4, 40).any(2)           # cap-then-union
    right = select_windows(group_max(bound, 2), float("inf"), max_windows=10)
    assert right.sum(-1).max() == 10
    assert wrong.sum(-1).max() > 10


def test_gate_selects_the_window_holding_the_true_argmax():
    """End to end: the window containing the globally hottest key is never cut."""
    k, a = _fixture(n=96)
    q = torch.randn(4, HKV, D, generator=torch.Generator().manual_seed(5)) * 1.5
    bound, _, _ = gate_and_score(q, _batched(build_sketch(k, a), 4), a, 1.0 / math.sqrt(D))
    true_win = _true_logits(k, q).amax(-1).argmax(-1)          # [B, Hq]
    for margin in (0.5, 2.0, 8.0):
        keep = select_windows(bound, margin)
        assert keep.gather(-1, true_win.unsqueeze(-1)).all(), f"missed at {margin}"


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_card_size_is_what_the_plan_is_costed_on():
    assert sketch_bytes_per_head(128, 8) == 400               # 3200 B/window at H=8
    b_q = (8 * 128 * 8) // 2 + 4 * 8 * 128 + 4 * 8 * 8        # config.py resolve()
    assert abs(280 * 8 / b_q - 0.265) < 0.002


# ---------------------------------------------------------------------------
# the default operating point
# ---------------------------------------------------------------------------


def test_recall_at_the_shipped_25_percent_ratio():
    """What justifies ``quant_gate_ratio = 0.25`` as a default.

    Built at the real shape-C geometry -- 271 active Q windows, Llama-3.1-8B GQA
    (32 query heads over 8 KV heads, rep=4) -- and measured as **attention mass
    recall per KV head**, which is the quantity a quality loss would come out of.
    Deliberately the worst head, not the mean: a mean hides the head that breaks.
    """
    hkv, hq, nw = 8, 32, 271
    g = torch.Generator().manual_seed(31)
    a = torch.randn(hkv, D, generator=g) * 0.4
    a[:, 3] += 40.0
    a[:, 17] -= 25.0
    k = a[None, :, None, :] + torch.randn(nw, hkv, WS, D, generator=g)
    k[:, :, 0, :] += torch.randn(nw, hkv, D, generator=g) * 6.0
    q = torch.randn(1, hq, D, generator=g) * 1.5
    scaling = 1.0 / math.sqrt(D)

    card = _batched(build_sketch(k, a), 1)
    bound, _, est = gate_and_score(q, card, a, scaling)
    true = scaling * torch.einsum(
        "nhwd,bhrd->bhrnw", k, q.reshape(1, hkv, hq // hkv, D)).reshape(1, hq, nw, WS)
    mass_kv = true.logsumexp(-1).reshape(1, hkv, hq // hkv, nw).logsumexp(2).exp()

    for ratio, floor in ((0.15, 0.99), (0.25, 0.995), (0.50, 0.999)):
        keep = select_windows(group_max(bound, hkv), float("inf"),
                              max_windows=math.ceil(ratio * nw),
                              rank_by=group_max(est, hkv))
        assert keep.sum(-1).max() <= math.ceil(ratio * nw)
        recall = (mass_kv * keep).sum(-1) / mass_kv.sum(-1)
        assert recall.min() >= floor, f"ratio {ratio}: worst head {recall.min():.4f}"


def test_ranking_by_estimate_beats_ranking_by_bound():
    """Pins why the cap ranks on the estimate and not on the bound.

    The bound carries a per-window slack term, so a loosely-bounded window can
    outrank a tighter one with a higher true logit. The estimate has no slack
    term and tracks truth more closely -- and costs less, since the ranking path
    never loads ``eps``.
    """
    hkv, hq, nw = 8, 32, 271
    g = torch.Generator().manual_seed(32)
    a = torch.randn(hkv, D, generator=g) * 0.4
    a[:, 3] += 40.0
    k = a[None, :, None, :] + torch.randn(nw, hkv, WS, D, generator=g)
    k[:, :, 0, :] += torch.randn(nw, hkv, D, generator=g) * 6.0
    q = torch.randn(1, hq, D, generator=g) * 1.5
    scaling = 1.0 / math.sqrt(D)
    bound, _, est = gate_and_score(q, _batched(build_sketch(k, a), 1), a, scaling)
    true = scaling * torch.einsum(
        "nhwd,bhrd->bhrnw", k, q.reshape(1, hkv, hq // hkv, D)).reshape(1, hq, nw, WS)
    mass_kv = true.logsumexp(-1).reshape(1, hkv, hq // hkv, nw).logsumexp(2).exp()

    cap = math.ceil(0.15 * nw)
    bm = group_max(bound, hkv)
    def recall(rank):
        keep = select_windows(bm, float("inf"), max_windows=cap, rank_by=rank)
        return ((mass_kv * keep).sum(-1) / mass_kv.sum(-1)).min().item()
    assert recall(group_max(est, hkv)) >= recall(bm)
