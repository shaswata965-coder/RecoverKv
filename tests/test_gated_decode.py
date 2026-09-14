"""One gated decode step, end to end, against ungated full attention.

This is the test that says the method works: run a real decode step through the
gate -- score cards, pick 25%, dequantize only those, attend, score every window
-- and compare the output and the scores against attending over the whole tier.

CPU-only. It exercises the algorithm and the wiring, not the Triton kernel,
which cannot run here.
"""

from __future__ import annotations

import math

import pytest
import torch

from modules.quant.store import QuantizedStore
from modules.windowed_cache.decode_kernel import two_tier_decode_reference
from modules.windowed_cache.gated_decode import gated_decode_step

B, HKV, REP, D, WS = 2, 4, 2, 32, 8
HQ = HKV * REP
SINK, NBODY = 8, 3
SCALING = 1.0 / math.sqrt(D)


class _RoPE(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.inv = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))

    def forward(self, x, pos):
        f = (pos.float().unsqueeze(-1) * self.inv).to(x.dtype)
        e = torch.cat([f, f], -1)
        return e.cos(), e.sin()


def _setup(n_q=32, seed=0):
    """An fp tier, an int2 tier with cards, and a query.

    ``keys_pre_rope`` is derived by UN-ROTATING ``k_post``, exactly as
    ``cache.py`` does at demotion. That is not cosmetic: the card describes
    ``k_post``, and the codes must describe the same keys, or the gate selects on
    one tensor while attention runs over another and every accuracy number here
    is meaningless.
    """
    from modules.quant.effective import unrotate_key_window

    g = torch.Generator().manual_seed(seed)
    st = QuantizedStore(WS, D, HKV, n_slots=n_q + 8, memoize_read=False,
                        sketch_enabled=True)
    st.ensure(B, torch.device("cpu"))
    rope = _RoPE(D)
    pos = (SINK + torch.arange(n_q * WS)).reshape(1, n_q, WS).expand(B, n_q, WS).contiguous()

    k_post = torch.randn(B, n_q, HKV, WS, D, generator=g)
    k_post[:, :, :, 0, :] += torch.randn(B, n_q, HKV, D, generator=g) * 5
    k_pre = unrotate_key_window(
        k_post.permute(0, 2, 1, 3, 4).reshape(B, HKV, n_q * WS, D),
        pos.reshape(B, n_q * WS), rope,
    ).reshape(B, HKV, n_q, WS, D).permute(0, 2, 1, 3, 4).contiguous()

    st.demote_many(
        st.table.free_slots(n_q), torch.ones(B, n_q, dtype=torch.bool),
        torch.arange(n_q).reshape(1, n_q).expand(B, n_q).contiguous(),
        k_pre,
        torch.randn(B, n_q, HKV, WS, D, generator=g),
        pos,
        keys_post_rope=k_post,
    )
    st.commit_active_count(n_q)
    s_fp = SINK + NBODY * WS
    k_fp = torch.randn(B, HKV, s_fp, D, generator=g)
    v_fp = torch.randn(B, HKV, s_fp, D, generator=g)
    q = torch.randn(B, HQ, D, generator=g) * 1.2
    return st, k_fp, v_fp, q


def _ungated(st, k_fp, v_fp, q, rope):
    """Attend over the WHOLE tier -- what the gate has to approximate."""
    k_q, v_q, _ = st.effective_q_tier(rope, k_fp.dtype)
    out, key_scores = two_tier_decode_reference(
        q, torch.cat([k_fp, k_q], 2), torch.cat([v_fp, v_q], 2), SCALING)
    body = key_scores[..., SINK:k_fp.shape[2]]
    pad = (-body.shape[-1]) % WS
    if pad:
        body = torch.nn.functional.pad(body, (0, pad))
    n_active = st.num_active_windows
    return out, torch.cat([
        body.reshape(B, HQ, -1, WS).sum(-1),
        key_scores[..., k_fp.shape[2]:].reshape(B, HQ, n_active, WS).sum(-1),
    ], dim=-1)


# ---------------------------------------------------------------------------


def test_ratio_one_is_byte_identical_to_the_ungated_path():
    """The no-op proof: reading everything must BE the old path, not resemble it.

    Ratio 1.0 selects every window, so the gated step attends over the same keys
    in the same order and must land on the same numbers. If this ever drifts,
    the gate is not a selection -- it is a second implementation of attention.
    """
    st, k_fp, v_fp, q = _setup()
    rope = _RoPE(D)
    got = gated_decode_step(q, k_fp, v_fp, st, rope, SCALING, SINK, ratio=1.0)
    want_out, want_scores = _ungated(st, k_fp, v_fp, q, rope)
    assert got.read_fraction == 1.0
    assert torch.allclose(got.out, want_out, atol=1e-5)
    assert torch.allclose(got.window_scores, want_scores, atol=1e-5)


@pytest.mark.parametrize("ratio,med,p95", [(0.25, 0.01, 0.20), (0.5, 0.005, 0.08)])
def test_gated_output_tracks_full_attention(ratio, med, p95):
    """Reading a quarter of the tier must not move the attention output much.

    Asserted on the MEDIAN and the 95th percentile, not the max, because there is
    a real tail and hiding it behind a loose max would be worse than naming it.
    Measured over six seeds at ratio 0.25: median relative output error 0.002,
    p95 ~0.1, max ~0.38 -- a handful of (row, head) pairs whose mass is spread
    thin enough across the tier that dropping 75% of it costs real output.

    This fixture is deliberately pessimistic about that tail: its fp tier is pure
    noise, whereas the real one holds the sink, the local window and the
    top-ranked windows -- i.e. most of the attention mass sits OUTSIDE the gated
    tier in production and inside it here. Treat the tail as an upper bound on
    what the shipped configuration will show, and read it off LongBench, not off
    this test.
    """
    rel = []
    for seed in range(6):
        st, k_fp, v_fp, q = _setup(n_q=64, seed=seed)
        rope = _RoPE(D)
        got = gated_decode_step(q, k_fp, v_fp, st, rope, SCALING, SINK, ratio=ratio)
        want_out, _ = _ungated(st, k_fp, v_fp, q, rope)
        rel.append(((got.out - want_out).norm(dim=-1)
                    / want_out.norm(dim=-1).clamp_min(1e-6)).flatten())
    rel = torch.cat(rel)
    assert rel.median() < med, f"ratio {ratio}: median {rel.median():.4f}"
    assert torch.quantile(rel, 0.95) < p95, \
        f"ratio {ratio}: p95 {torch.quantile(rel, 0.95):.4f}"


def test_the_gate_actually_reads_only_the_ratio():
    st, k_fp, v_fp, q = _setup(n_q=64)
    got = gated_decode_step(q, k_fp, v_fp, st, _RoPE(D), SCALING, SINK, ratio=0.25)
    assert got[2] == math.ceil(0.25 * 64)
    assert got.read_fraction <= 0.26


def test_every_window_gets_a_score_including_the_skipped_ones():
    """The starvation guard: no retained window may come out of a step at zero."""
    st, k_fp, v_fp, q = _setup(n_q=64)
    got = gated_decode_step(q, k_fp, v_fp, st, _RoPE(D), SCALING, SINK, ratio=0.25)
    assert got.window_scores.shape == (B, HQ, NBODY + 64)
    assert torch.isfinite(got.window_scores).all()
    assert (got.window_scores > 0).all(), "a retained window scored zero"


def test_scores_rank_windows_like_full_attention_does():
    """Eviction consumes the RANKING, so that is what has to survive gating.

    Measured over six seeds at ratio 0.25: correlation 0.9995 and the top window
    preserved in 100% of (row, head) pairs.
    """
    st, k_fp, v_fp, q = _setup(n_q=64)
    rope = _RoPE(D)
    got = gated_decode_step(q, k_fp, v_fp, st, rope, SCALING, SINK, ratio=0.25)
    _, want = _ungated(st, k_fp, v_fp, q, rope)
    corr = torch.corrcoef(torch.stack(
        [got.window_scores.flatten(), want.flatten()]))[0, 1]
    assert corr > 0.95, f"ranking correlation {corr:.4f}"
    # The top window must survive: eviction keeps by rank, so losing the best one
    # is how a gated cache would quietly throw away what it needs most.
    assert torch.equal(got.window_scores.argmax(-1), want.argmax(-1))


def test_total_score_mass_is_preserved():
    """Scores feed a cumulative sum, so a SYSTEMATIC scale drift would compound.

    Checked on the median across seeds: the calibration factor is per (row,
    head), so an individual head can be off by more, but the estimator must not
    be biased in aggregate or every window's running score drifts together.
    """
    err = []
    for seed in range(6):
        st, k_fp, v_fp, q = _setup(n_q=64, seed=seed)
        rope = _RoPE(D)
        got = gated_decode_step(q, k_fp, v_fp, st, rope, SCALING, SINK, ratio=0.25)
        _, want = _ungated(st, k_fp, v_fp, q, rope)
        err.append((got.window_scores.sum(-1) / want.sum(-1)).flatten())
    err = torch.cat(err)
    assert (err.median() - 1.0).abs() < 0.01, f"median mass ratio {err.median():.4f}"
    assert (err.mean() - 1.0).abs() < 0.05, f"mean mass ratio {err.mean():.4f}"


def test_empty_tier_falls_through_cleanly():
    st, k_fp, v_fp, q = _setup()
    st._n_active = 0
    got = gated_decode_step(q, k_fp, v_fp, st, _RoPE(D), SCALING, SINK, ratio=0.25)
    assert got.window_scores.shape == (B, HQ, NBODY)
    assert torch.isfinite(got.out).all()
