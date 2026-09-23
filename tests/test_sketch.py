"""Rank-1 window sketches — the properties the gate's correctness rests on.

The load-bearing tests are the two that pin the **encoding widths**:
:func:`test_int8_direction_is_required_not_a_preference` (why ``v`` may not
shrink) and :func:`test_int4_mu_and_vbar_are_accuracy_neutral` (why ``mu`` and
``vbar`` may). The card is read for every window on every step while a window is
only opened if selected, so the gate costs ``card + ratio * window`` against
``window`` — which makes the card's width the whole economics of the feature.
Everything else here supports those or pins a decision made by measurement.

CPU-only; no GPU, no Triton.
"""

from __future__ import annotations

import math

import pytest
import torch

import modules.quant.sketch as sketch_mod
from modules.quant.sketch import (
    _dq_sym,
    _dq_sym4,
    _q_sym,
    _q_sym4,
    build_sketch,
    decode_sketch,
    gate_and_score,
    group_max,
    group_share,
    select_windows,
    sketch_bytes_per_head,
    value_centroid,
)

D, WS, HKV = 128, 8, 4


def _fixture(n=64, h=HKV, ws=WS, d=D, hot=6.0, spread=1.0, seed=0):
    """Keys shaped like real ones: a per-head common mode with massive-activation
    channels, plus (optionally) one outlier token per window. Values and their
    anchor come along, because the card carries a value centroid now."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(h, d, generator=g) * 0.4
    a[:, 3] += 40.0
    a[:, 17] -= 25.0
    k = a[None, :, None, :] + torch.randn(n, h, ws, d, generator=g) * spread
    if hot:
        k[:, :, 0, :] += torch.randn(n, h, d, generator=g) * hot
    v = torch.randn(n, h, ws, d, generator=g)
    return k, a, v, v.mean(dim=(0, 2))


def _card(k, a, v, va, b=1):
    """A built card with a leading batch axis, as the read path sees it."""
    s = build_sketch(k, a, v, va)
    return type(s)(*[f.unsqueeze(0).expand(b, *f.shape).contiguous() for f in s])


def _true_logits(k, q, hkv=HKV):
    """``[B, Hq, N, ws]`` exact ``q.k`` for every token of every window."""
    b, hq, _ = q.shape
    qg = q.reshape(b, hkv, hq // hkv, D)
    return torch.einsum("nhwd,bhrd->bhrnw", k, qg).reshape(b, hq, k.shape[0], k.shape[2])


# ---------------------------------------------------------------------------
# the int4 codec
# ---------------------------------------------------------------------------


def test_int4_packing_round_trips_on_the_grid():
    """Two codes per byte, low nibble first — the same lane map as the int2
    crumb packer, one width up. The kernels unpack by that rule, so if the
    interleave ever flips, every mu and every centroid silently transposes."""
    x = torch.zeros(1, 1, 4)
    x[0, 0, 0], x[0, 0, 3] = 1.0, -1.0
    packed, scale = _q_sym4(x)
    assert packed.shape == (1, 1, 2) and packed.dtype == torch.uint8
    got = _dq_sym4(packed, scale)
    assert got.shape == x.shape
    assert got[0, 0, 0] > 0 and got[0, 0, 3] < 0
    assert got[0, 0, 1].abs() < 1e-6 and got[0, 0, 2].abs() < 1e-6


def test_int4_codes_use_the_whole_four_bit_range():
    """A real int4 encoder re-fits the scale to amax/7. Coarsening int8 codes
    against the int8 scale instead would waste half the range and is what the
    original ablation simulated — so the measured accuracy here is a floor."""
    x = torch.randn(8, 4, D) * 3.0
    packed, scale = _q_sym4(x)
    codes = torch.round(_dq_sym4(packed, scale) / scale.unsqueeze(-1).float())
    assert codes.min() >= -7 and codes.max() <= 7
    assert codes.min() <= -6 and codes.max() >= 6, "the range is barely used"


@pytest.mark.parametrize("ratio", [0.10, 0.25])
def test_int4_mu_and_vbar_are_accuracy_neutral(ratio):
    """The measurement that authorises the width change.

    ``mu`` and ``vbar`` are 64% of the card and both are means stored as a
    residual from a frozen anchor. Coarsening them must not move what the gate
    SELECTS, which is the only thing that changes which keys attention sees.
    Compared against the previous int8 encoding on the same fixture.
    """
    int4 = (sketch_mod._q_sym4, sketch_mod._dq_sym4)
    int8 = (sketch_mod._q_sym, sketch_mod._dq_sym)

    def picks(enc, k, a, v, va, q, n_sel):
        sketch_mod._q_sym4, sketch_mod._dq_sym4 = enc
        try:
            logmass, _ = gate_and_score(q, _card(k, a, v, va), a, 1.0 / math.sqrt(D))
        finally:
            sketch_mod._q_sym4, sketch_mod._dq_sym4 = int4
        # The shipped selection: the group union of per-head shares.
        return group_share(logmass, HKV).topk(n_sel, dim=-1).indices

    overlaps = []
    for seed in range(6):
        k, a, v, va = _fixture(n=120, seed=seed)
        q = torch.randn(1, HKV * 2, D, generator=torch.Generator().manual_seed(seed))
        n_sel = max(1, math.ceil(ratio * k.shape[0]))
        s4, s8 = picks(int4, k, a, v, va, q, n_sel), picks(int8, k, a, v, va, q, n_sel)
        overlaps += [len(set(s4[0, h].tolist()) & set(s8[0, h].tolist())) / n_sel
                     for h in range(HKV)]
    mean_overlap = sum(overlaps) / len(overlaps)
    assert mean_overlap > 0.95, f"int4 changed the selection: {mean_overlap:.3f}"


def test_int8_direction_is_required_not_a_preference():
    """Pins the measurement that keeps ``v`` at int8 while mu and vbar went int4.

    A coarser ``v`` leaves the hot token with a residual no better than the
    window mean — i.e. the card stops doing the one thing it exists to do. This
    is the guard that makes "shrink the card" a targeted change rather than a
    sweep over every field.
    """
    k, a, _, _ = _fixture(n=128)
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


# ---------------------------------------------------------------------------
# the card as a model of its window
# ---------------------------------------------------------------------------


def test_degenerate_window_is_finite():
    """Zero deviation everywhere: no NaN out of the normalise-by-farthest step."""
    _, a, _, va = _fixture()
    k = a[None, :, None, :].expand(4, HKV, WS, D).contiguous()
    v = torch.zeros(4, HKV, WS, D)
    s = build_sketch(k, a, v, va)
    for f in decode_sketch(s, a):
        assert torch.isfinite(f).all()
    q = torch.randn(1, HKV, D, generator=torch.Generator().manual_seed(2))
    logmass, est = gate_and_score(q, _card(k, a, v, va), a, 1.0 / math.sqrt(D))
    assert torch.isfinite(logmass).all() and torch.isfinite(est).all()


def test_hot_token_is_the_best_fit_token():
    """``v`` points at the farthest token, so that token fits best.

    This is the whole reason a rank-1 card beats a mean on a 1-hot-in-8 window:
    the model spends its one direction on the token the gate exists to find.
    """
    k, a, v, va = _fixture(n=128)
    mu_h, v_h, t_h = decode_sketch(build_sketch(k, a, v, va), a)
    # dim=-1, not p=-1: `.norm(-1)` is a p-norm over EVERY axis and returns a
    # scalar, which made the pre-existing version of this assertion vacuous.
    resid = (k - (mu_h.unsqueeze(-2) + t_h.unsqueeze(-1) * v_h.unsqueeze(-2))
             ).norm(dim=-1)
    far = (k - k.mean(-2, keepdim=True)).norm(dim=-1).argmax(-1)
    e_far = resid.gather(-1, far.unsqueeze(-1)).squeeze(-1)
    assert (e_far < resid.median(dim=-1).values).float().mean() > 0.99


def test_rank1_beats_centroid_on_reconstruction():
    k, a, v, va = _fixture(n=128)
    mu_h, v_h, t_h = decode_sketch(build_sketch(k, a, v, va), a)
    rank1 = (k - (mu_h.unsqueeze(-2) + t_h.unsqueeze(-1) * v_h.unsqueeze(-2))
             ).norm(dim=-1)
    centroid = (k - k.mean(-2, keepdim=True)).norm(dim=-1)
    assert rank1.mean() < 0.6 * centroid.mean()


def test_anchor_removes_the_common_mode():
    """Massive-activation channels must not eat the code range — and at int4
    there are four fewer bits for them to eat, so this matters more, not less."""
    k, a, v, va = _fixture(n=128)
    mu = k.mean(-2)
    with_a = decode_sketch(build_sketch(k, a, v, va), a)[0]
    zero = torch.zeros_like(a)
    without = decode_sketch(build_sketch(k, zero, v, va), zero)[0]
    assert (mu - with_a).abs().mean() < 0.2 * (mu - without).abs().mean()


def test_value_centroid_decodes_against_its_anchor():
    """The centroid is what a SKIPPED window contributes to the output, so a
    centroid decoded against the wrong anchor is a silent output error on 75% of
    the tier."""
    k, a, v, va = _fixture(n=32)
    got = value_centroid(_card(k, a, v, va), va)
    assert got.shape == (1, 32, HKV, D)
    true = v.mean(dim=-2)
    rel = (got[0] - true).norm() / true.norm()
    assert rel < 0.25, f"centroid off by {rel:.3f}"


def test_logmass_tracks_true_mass():
    """The score written back for a skipped window has to be usable for ranking."""
    k, a, v, va = _fixture(n=128)
    q = torch.randn(4, HKV, D, generator=torch.Generator().manual_seed(3))
    scaling = 1.0 / math.sqrt(D)
    logmass, _ = gate_and_score(q, _card(k, a, v, va, b=4), a, scaling)
    true = torch.logsumexp(scaling * _true_logits(k, q), dim=-1)
    err = (logmass - true).abs()
    assert err.mean() < 0.35 and err.max() < 3.0
    # Ranking is what eviction consumes, so check order, not just magnitude.
    rho = torch.corrcoef(torch.stack([logmass.flatten(), true.flatten()]))[0, 1]
    assert rho > 0.99


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_cap_bounds_the_work():
    assert select_windows(torch.randn(2, 8, 40), 5).sum(-1).max() == 5


def test_selection_always_keeps_the_best_window():
    e = torch.randn(2, 8, 40)
    assert select_windows(e, 5).gather(-1, e.argmax(-1, keepdim=True)).all()


def test_group_max_is_the_gqa_contract():
    """A window any query head in the group wants survives for the whole group."""
    e = torch.full((1, 8, 5), -10.0)
    e[0, 3, 2] = 5.0                          # one query head of group 1 wants w2
    g = group_max(e, num_kv_heads=4)
    assert g.shape == (1, 4, 5) and g[0, 1, 2] == 5.0
    assert select_windows(g, 1)[0, 1, 2]


def test_cap_must_bind_after_the_group_union_not_before():
    """Capping per query head then unioning inflates the ratio by up to `rep`.

    At rep=4 a 25% cap would become 100% of windows actually loaded, and the gate
    would buy nothing. Pins the ordering that prevents it.
    """
    g = torch.Generator().manual_seed(21)
    est = torch.randn(1, 8, 40, generator=g)            # Hq=8, Hkv=2 -> rep=4
    wrong = select_windows(est, 10).reshape(1, 2, 4, 40).any(2)   # cap-then-union
    right = select_windows(group_max(est, 2), 10)
    assert right.sum(-1).max() == 10
    assert wrong.sum(-1).max() > 10


def test_gate_selects_the_window_holding_the_true_argmax():
    """End to end: the window containing the globally hottest key is never cut."""
    k, a, v, va = _fixture(n=96)
    q = torch.randn(4, HKV, D, generator=torch.Generator().manual_seed(5)) * 1.5
    _, est = gate_and_score(q, _card(k, a, v, va, b=4), a, 1.0 / math.sqrt(D))
    true_win = _true_logits(k, q).amax(-1).argmax(-1)          # [B, Hq]
    keep = select_windows(est, math.ceil(0.25 * 96))
    assert keep.gather(-1, true_win.unsqueeze(-1)).all()


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_card_size_is_what_the_plan_is_costed_on():
    """272 B/head, and the ratio the gate's break-even is derived from.

    ``config.resolve`` adds this to ``bytes_per_q_window``, so it is the number
    the memory claim is made against — not documentation.
    """
    assert sketch_bytes_per_head(128, 8) == 272               # 2176 B/window at H=8
    # mu(64+2) + v(128+2) + t(8+2) + vbar(64+2), the int4 split spelled out.
    assert sketch_bytes_per_head(128, 8) == 66 + 130 + 10 + 66
    # Break-even read ratio = 1 - card/window, on the token data a card stands in
    # for: 512 B of int2 K+V plus its 292 B grid, per head.
    window = 512 + 292
    assert abs((1 - 272 / window) - 0.662) < 0.005


def test_int4_moved_break_even_the_right_way():
    """The point of the change, as arithmetic rather than as a claim.

    At int8 the card was 400 B against an 804 B window — 78% of the data it
    replaces, break-even at a 0.50 read ratio, which is why the gate measured as
    worth nothing at the shipped 0.25.
    """
    int8_card = 3 * (128 + 2) + (8 + 2)
    assert int8_card == 400
    window = 512 + 292
    assert abs((1 - int8_card / window) - 0.502) < 0.005
    assert sketch_bytes_per_head(128, 8) < int8_card


# ---------------------------------------------------------------------------
# the default operating point
# ---------------------------------------------------------------------------


def test_recall_at_the_shipped_25_percent_ratio():
    """What justifies ``quant_gate_ratio = 0.25`` as a default.

    Built at the real shape-C geometry -- 271 active Q windows, Llama-3.1-8B GQA
    (32 query heads over 8 KV heads, rep=4) -- and measured as **attention mass
    recall per QUERY head**: each head's share of its OWN softmax that the gate
    reads, which is the quantity a quality loss comes out of. Deliberately the
    worst head, not the mean: a mean hides the head that breaks.

    This used to measure recall of the group's SUMMED raw mass, ``exp(logit)``
    added across the group's query heads. That sum is dominated by the head with
    the largest baseline logit, which is the same head the old raw-max union let
    choose for the group, so metric and selection shared one blind spot. It read
    1.000 while the worst query head had 0.0005 of its mass read
    (``tests/test_gate_union_is_per_head.py``).
    """
    hkv, hq, nw = 8, 32, 271
    g = torch.Generator().manual_seed(31)
    a = torch.randn(hkv, D, generator=g) * 0.4
    a[:, 3] += 40.0
    a[:, 17] -= 25.0
    k = a[None, :, None, :] + torch.randn(nw, hkv, WS, D, generator=g)
    k[:, :, 0, :] += torch.randn(nw, hkv, D, generator=g) * 6.0
    v = torch.randn(nw, hkv, WS, D, generator=g)
    q = torch.randn(1, hq, D, generator=g) * 1.5
    scaling = 1.0 / math.sqrt(D)

    s = build_sketch(k, a, v, v.mean(dim=(0, 2)))
    card = type(s)(*[f.unsqueeze(0) for f in s])
    logmass, _ = gate_and_score(q, card, a, scaling)
    true = scaling * torch.einsum(
        "nhwd,bhrd->bhrnw", k, q.reshape(1, hkv, hq // hkv, D)).reshape(1, hq, nw, WS)
    tw = true.logsumexp(-1)
    p = (tw - tw.logsumexp(-1, keepdim=True)).exp()          # each head's own softmax

    for ratio, floor in ((0.10, 0.95), (0.15, 0.985), (0.25, 0.995), (0.50, 0.999)):
        keep = select_windows(group_share(logmass, hkv), math.ceil(ratio * nw))
        assert keep.sum(-1).max() <= math.ceil(ratio * nw)
        recall = (p * keep.repeat_interleave(hq // hkv, dim=1)).sum(-1)
        assert recall.min() >= floor, f"ratio {ratio}: worst head {recall.min():.4f}"
