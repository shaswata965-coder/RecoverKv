"""The gate must read what EACH query head needs, not what the loudest one wants.

The defect behind the RULER 32k regression that survived ``0b21c9f``. The gate
reads ``n_sel`` windows per KV head, chosen for the whole GQA group, and it chose
them by the group MAX of the raw per-head estimate ``max_i scale * q.k_i``.
A raw logit carries its head's baseline (``q.anchor``) and scale (``|q|``), and a
head's softmax is invariant to both. So the max was decided by whichever head of
the group had the largest baseline or scale, and every other head was read only
where that head happened to agree. A retrieval head in a group with a louder head
lost its needle window on about half of the steps, and a UUID is about 25 copy
steps long.

The fix unions each head's SHARE of its own int2 mass,
``logmass_h(w) - logsumexp_w' logmass_h(w')`` (``sketch.group_share``), which
is what the head's own softmax computes and is invariant to both.

Everything here runs the repo's CPU oracles, with real cards from
``build_sketch``: ``gate_reference`` for the selection the Triton gate makes,
``gate_share_tiled_reference`` for the share kernel's tiling, and
``two_tier_window_reference`` for the decode output. Each defect test has a
guard that runs the OLD rule on the same fixture and requires it to fail, so the
fixture cannot quietly stop discriminating.
"""

from __future__ import annotations

import math

import pytest
import torch

from modules.quant.sketch import (
    Sketch, build_sketch, gate_and_score, group_max, group_share,
    head_log_share, value_centroid,
)
from modules.windowed_cache.decode_kernel import two_tier_window_reference
from modules.windowed_cache.gate_kernel import (
    gate_reference, gate_share_tiled_reference,
)

D = 128
SCALING = D ** -0.5


def _old_rule(est: torch.Tensor, hkv: int) -> torch.Tensor:
    """The shipped union before this fix: the group max of the RAW estimate."""
    return group_max(est, hkv)


def _pick(score: torch.Tensor, n_sel: int) -> torch.Tensor:
    """``[B, Hkv, n_sel]`` ascending, exactly as ``gate_reference`` returns it."""
    return score.topk(n_sel, dim=-1).indices.sort(-1).values


# ---------------------------------------------------------------------------
# 1. The property: invariant to what a softmax is invariant to
# ---------------------------------------------------------------------------


def _constant_channel_fixture(seed, nw=96, hkv=4, rep=4, ws=8):
    """Keys whose channel 0 is the same constant on every token of every window.

    Moving a query head along channel 0 then adds one exact constant to every
    logit that head produces -- its baseline, which its softmax ignores -- and
    to its card estimates, because the card's anchor carries the constant and
    ``mu - anchor``, ``v`` and ``t`` are all exactly zero there.
    """
    g = torch.Generator().manual_seed(seed)
    anc = torch.randn(hkv, D, generator=g) * 0.4
    anc[:, 0] = 8.0
    k = anc[None, :, None, :] + torch.randn(nw, hkv, ws, D, generator=g)
    k[..., 0] = 8.0
    v = torch.randn(nw, hkv, ws, D, generator=g)
    q = torch.randn(1, hkv * rep, D, generator=g)
    q[..., 0] = 0.0
    card = Sketch(*(t.unsqueeze(0) for t in build_sketch(k, anc, v, v.mean((0, 2)))))
    return q, card, anc, hkv


@pytest.mark.parametrize("seed", range(4))
def test_selection_ignores_a_per_head_baseline(seed):
    """Give each query head its own baseline. The softmax does not see it, so the
    selection must not either."""
    q, card, anc, hkv = _constant_channel_fixture(seed)
    n_sel = 24
    base, _ = gate_reference(q, *card, anc, SCALING, n_sel)
    g = torch.Generator().manual_seed(100 + seed)
    shifted = q.clone()
    shifted[..., 0] = 3.0 * torch.randn(q.shape[:2], generator=g)
    moved, _ = gate_reference(shifted, *card, anc, SCALING, n_sel)
    assert torch.equal(base, moved)


def test_the_old_rule_is_not_invariant_to_a_per_head_baseline():
    """Guard: on the same fixture, the raw-estimate union moves with the
    baselines. If this ever passes, the fixture no longer shows the defect."""
    q, card, anc, hkv = _constant_channel_fixture(0)
    n_sel = 24
    _, est = gate_and_score(q, card, anc, SCALING)
    shifted = q.clone()
    shifted[..., 0] = 3.0 * torch.randn(
        q.shape[:2], generator=torch.Generator().manual_seed(100))
    _, est2 = gate_and_score(shifted, card, anc, SCALING)
    assert not torch.equal(_pick(_old_rule(est, hkv), n_sel),
                           _pick(_old_rule(est2, hkv), n_sel))


def test_head_share_is_a_distribution_per_head():
    """``exp(share)`` sums to one per query head: 0 means the whole int2 mass."""
    lm = torch.randn(2, 8, 50) * 4.0 + torch.randn(2, 8, 1) * 30.0
    share = head_log_share(lm)
    torch.testing.assert_close(share.exp().sum(-1), torch.ones(2, 8))
    torch.testing.assert_close(head_log_share(lm + 17.0), share)


# ---------------------------------------------------------------------------
# 2. Recall, per QUERY head, on the repo's own recall fixture
# ---------------------------------------------------------------------------


def _recall_fixture(seed=31, nw=271, hkv=8, hq=32, ws=8):
    """``tests/test_sketch.py``'s shipped-ratio fixture, verbatim."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(hkv, D, generator=g) * 0.4
    a[:, 3] += 40.0
    a[:, 17] -= 25.0
    k = a[None, :, None, :] + torch.randn(nw, hkv, ws, D, generator=g)
    k[:, :, 0, :] += torch.randn(nw, hkv, D, generator=g) * 6.0
    v = torch.randn(nw, hkv, ws, D, generator=g)
    q = torch.randn(1, hq, D, generator=g) * 1.5
    s = build_sketch(k, a, v, v.mean(dim=(0, 2)))
    card = type(s)(*[f.unsqueeze(0) for f in s])
    rep = hq // hkv
    true = SCALING * torch.einsum(
        "nhwd,bhrd->bhrnw", k, q.reshape(1, hkv, rep, D)).reshape(1, hq, nw, ws)
    tw = true.logsumexp(-1)
    p = (tw - tw.logsumexp(-1, keepdim=True)).exp()      # each head's OWN softmax
    return q, card, a, p, hkv, rep, nw


def _per_head_recall(p, sel, rep):
    keep = torch.zeros(p.shape[0], sel.shape[1], p.shape[-1], dtype=torch.bool)
    keep.scatter_(-1, sel, True)
    return (p * keep.repeat_interleave(rep, dim=1)).sum(-1)


def test_every_query_head_keeps_its_own_mass_at_the_shipped_ratio():
    """The metric a quality loss comes out of: each head's share of ITS OWN
    softmax that the gate reads, through the shipped selection. The
    group-summed metric this fixture was built with is dominated by the loudest
    head of each group and reported 1.000 under the old rule."""
    q, card, a, p, hkv, rep, nw = _recall_fixture()
    sel, _ = gate_reference(q, *card, a, SCALING, math.ceil(0.25 * nw))
    assert _per_head_recall(p, sel, rep).min() > 0.995


def test_the_old_rule_starves_a_query_head_on_the_same_fixture():
    """Guard: the raw-estimate union read 0.05% of one head's mass here."""
    q, card, a, p, hkv, rep, nw = _recall_fixture()
    _, est = gate_and_score(q, card, a, SCALING)
    sel = _pick(_old_rule(est, hkv), math.ceil(0.25 * nw))
    assert _per_head_recall(p, sel, rep).min() < 0.05


# ---------------------------------------------------------------------------
# 3. RULER-shaped: a retrieval head sharing its KV head with louder ones
# ---------------------------------------------------------------------------


WS32, HKV, REP, SINK = 32, 4, 4, 5


def _retrieval_scenario(seed, n_q=160, n_body=4, sharp=14.0):
    """``ws = 32``, the 32k operating point's window. In every GQA group one
    query head retrieves: its query points at one random token of one random
    int2 window -- not the card's farthest point -- with logit advantage
    ``sharp``. The other three are ordinary heads with their own directions,
    scales and baselines, which is what makes one of them louder than the
    retrieval head on the raw scale. The retrieval head keeps its own baseline.
    """
    g = torch.Generator().manual_seed(seed)
    hq = HKV * REP
    anc = torch.randn(HKV, D, generator=g) * 0.4
    anc[:, 3] += 30.0
    anc[:, 17] -= 20.0
    anc[:, 64] += 12.0

    def keys(nw):
        return (anc[None, :, None, :] + 0.5 * torch.randn(nw, HKV, 1, D, generator=g)
                + torch.randn(nw, HKV, WS32, D, generator=g))

    kq, kb = keys(n_q), keys(n_body)
    ks = anc[None, :, None, :] + torch.randn(1, HKV, SINK, D, generator=g)
    vq = torch.randn(n_q, HKV, WS32, D, generator=g)
    vb = torch.randn(n_body, HKV, WS32, D, generator=g)
    vs = torch.randn(1, HKV, SINK, D, generator=g)
    u = anc / anc.norm(dim=-1, keepdim=True)
    q = torch.randn(1, hq, D, generator=g) * (0.5 + 1.5 * torch.rand(1, hq, 1, generator=g))
    for h in range(hq):
        q[0, h] += (6.0 * float(torch.randn((), generator=g)) * u[h // REP]
                    / (anc[h // REP].norm() * SCALING))
    targets = []
    for kv in range(HKV):
        h = kv * REP + int(torch.randint(0, REP, (1,), generator=g))
        w = int(torch.randint(0, n_q, (1,), generator=g))
        t = int(torch.randint(0, WS32, (1,), generator=g))
        kt = kq[w, kv, t] - anc[kv]
        base = float(SCALING * (q[0, h] @ anc[kv]))
        q[0, h] = (0.3 * torch.randn(D, generator=g)
                   + (sharp / SCALING) * kt / kt.norm() ** 2
                   + (base / SCALING) * u[kv] / anc[kv].norm())
        targets.append((kv, h, w))

    def flat(x):
        return x.permute(1, 0, 2, 3).reshape(1, HKV, -1, D)

    v_anc = vq.mean(dim=(0, 2))
    return dict(
        q=q, n_q=n_q, n_body=n_body, targets=targets, anc=anc, v_anc=v_anc,
        k=torch.cat([ks, flat(kb), flat(kq)], dim=2),
        v=torch.cat([vs, flat(vb), flat(vq)], dim=2),
        card=Sketch(*(t.unsqueeze(0) for t in build_sketch(kq, anc, vq, v_anc))),
    )


def _decode(d, sel, logmass):
    """One gated decode step on the kernel's oracle, and the ungated reference."""
    cent = (value_centroid(d["card"], d["v_anc"]).permute(0, 2, 1, 3)
            .repeat_interleave(REP, dim=1))
    sfp = SINK + d["n_body"] * WS32
    common = (d["q"], d["k"], d["v"], SCALING, SINK, WS32, d["n_body"])
    exact, _ = two_tier_window_reference(*common, Sfp=sfp)
    out, _ = two_tier_window_reference(
        *common, Sfp=sfp, sel=sel, logmass=logmass, centroids=cent)
    return ((out.float() - exact.float()).norm(dim=-1)
            / exact.float().norm(dim=-1))[0]                     # [H_q]


def _needle_read(d, sel):
    return [bool((sel[0, kv] == w).any()) for kv, _, w in d["targets"]]


@pytest.mark.parametrize("seed", range(4))
def test_a_retrieval_head_reads_its_needle_window(seed):
    """The step RULER is made of. Every group's retrieval head must get its
    needle window read, whatever its group-mates' baselines."""
    d = _retrieval_scenario(seed)
    n_sel = math.ceil(0.25 * d["n_q"])
    sel, _ = gate_reference(d["q"], *d["card"], d["anc"], SCALING, n_sel)
    assert all(_needle_read(d, sel))


@pytest.mark.parametrize("seed", range(2))
def test_a_retrieval_heads_output_is_its_needle(seed):
    """Through the decode oracle: reading the needle is what makes the output the
    needle's value, and not a blend of centroids."""
    d = _retrieval_scenario(seed)
    n_sel = math.ceil(0.25 * d["n_q"])
    sel, logmass = gate_reference(d["q"], *d["card"], d["anc"], SCALING, n_sel)
    err = _decode(d, sel, logmass)
    assert max(err[h].item() for _, h, _ in d["targets"]) < 0.05


def test_the_old_rule_loses_needles_on_the_same_scenarios():
    """Guard: the raw-estimate union reads well under all of the needles across
    the scenarios above, and a missed needle leaves the head's output far from
    its value."""
    hits, worst = [], 0.0
    for seed in range(4):
        d = _retrieval_scenario(seed)
        n_sel = math.ceil(0.25 * d["n_q"])
        logmass, est = gate_and_score(d["q"], d["card"], d["anc"], SCALING)
        sel = _pick(_old_rule(est, HKV), n_sel)
        hits += _needle_read(d, sel)
        if seed < 2:
            err = _decode(d, sel, logmass)
            worst = max(worst, max(err[h].item() for _, h, _ in d["targets"]))
    assert sum(hits) / len(hits) < 0.8
    assert worst > 0.5


# ---------------------------------------------------------------------------
# 4. The share kernel's tiling, mirrored on CPU, agrees with the spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nw", [1, 7, 128, 129, 840])
@pytest.mark.parametrize("rep", [1, 3, 4])
@pytest.mark.parametrize("block_w", [16, 128])
def test_the_share_kernel_mirror_matches_the_spec(nw, rep, block_w):
    """``gate_share_tiled_reference`` walks the window axis in the kernel's tiles
    with its online logsumexp and padding lanes. Tails (``nw % block_w``), a
    single window, and a non-power-of-two group are the cases the masking has to
    get right."""
    g = torch.Generator().manual_seed(nw * 31 + rep)
    hkv = 2
    lm = (torch.randn(3, hkv * rep, nw, generator=g) * 5.0
          + torch.randn(3, hkv * rep, 1, generator=g) * 40.0)
    torch.testing.assert_close(
        gate_share_tiled_reference(lm, hkv, block_w), group_share(lm, hkv),
        rtol=1e-5, atol=1e-4)


def test_the_share_ranks_like_each_heads_own_logmass():
    """Within one head the share is ``logmass`` minus a constant, so a group of
    one ranks exactly as that head's own estimate does."""
    lm = torch.randn(2, 4, 60, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(group_share(lm, num_kv_heads=4).argsort(-1),
                               lm.argsort(-1), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 5. What must not move
# ---------------------------------------------------------------------------


def test_ratio_one_still_reads_every_window():
    """The control arm: 1.0 selects everything through the same code, so the
    union rule cannot change it."""
    d = _retrieval_scenario(0, n_q=40)
    sel, _ = gate_reference(d["q"], *d["card"], d["anc"], SCALING, d["n_q"])
    assert torch.equal(sel[0], torch.arange(d["n_q"]).expand(HKV, -1))


def test_the_cap_still_binds_on_the_kv_head():
    """The union is still taken BEFORE the cap: ``n_sel`` windows per KV head,
    not ``n_sel`` per query head."""
    d = _retrieval_scenario(1, n_q=64)
    sel, _ = gate_reference(d["q"], *d["card"], d["anc"], SCALING, 16)
    assert sel.shape == (1, HKV, 16)
    assert all(len(set(sel[0, kv].tolist())) == 16 for kv in range(HKV))
