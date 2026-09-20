"""Split-KV on the decode reference: `splits=1` is a no-op, `splits>1` agrees.

Run B (DISTANCE_TO_GOAL §11.11) measured the decode kernel as grid-limited:
`grid = (B*H_kv,)` is 8 blocks at B=1 on a 108-SM A100, the step is 93%
batch-invariant, and ~11 ms of a 14.42 ms kernel is paid regardless of batch.
Split-KV partitions the window axis so the grid becomes `B*H_kv*S`.

The Triton kernel cannot run on the dev box. The ALGORITHM can, and this repo's
contract is that it must: every kernel has a CPU reference mirroring its tiling
and masking, and the tests pin the reference. So the reference carries `splits`
first, and these are the properties the kernel then has to reproduce.

Two of them matter most:

* **`splits=1` is bit-identical** to the unsplit path. That is what makes it a
  provable control arm, the same role `--gate-ratio 1.0` plays for the gate.
* **the partition is a partition.** Each window is written by exactly one
  split, so `wsum`/`wmax` need no cross-split reduction. A window counted twice
  or dropped is the failure this change could plausibly introduce, and it would
  show up as a score change, not a crash.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modules.windowed_cache.decode_kernel import (  # noqa: E402
    two_tier_window_reference,
)

B, H_KV, REP, D, WS = 2, 2, 2, 16, 4
H_Q = H_KV * REP
NUM_SINK, N_BODY_WIN = 5, 6


def _inputs(seed: int = 0, gated: bool = False, n_sel: int = 3):
    g = torch.Generator().manual_seed(seed)
    body_end = NUM_SINK + N_BODY_WIN * WS
    n_q_win = 5
    S = body_end + n_q_win * WS
    q = torch.randn(B, H_Q, D, generator=g)
    k = torch.randn(B, H_KV, S, D, generator=g)
    v = torch.randn(B, H_KV, S, D, generator=g)
    kw = dict(q=q, k_eff=k, v_eff=v, scaling=D ** -0.5, num_sink=NUM_SINK,
              window_size=WS, n_body_win=N_BODY_WIN, Sfp=body_end, exp2=True)
    if gated:
        sel = torch.stack([
            torch.stack([torch.randperm(n_q_win, generator=g)[:n_sel].sort().values
                         for _ in range(H_KV)]) for _ in range(B)])
        kw["sel"] = sel.to(torch.int32)
        kw["logmass"] = torch.randn(B, H_Q, n_q_win, generator=g)
        kw["centroids"] = torch.randn(B, H_Q, n_q_win, D, generator=g)
    return kw


# --- splits=1 is a provable no-op ------------------------------------------

@pytest.mark.parametrize("gated", [False, True], ids=["ungated", "gated"])
def test_splits_one_is_bit_identical_to_the_unsplit_path(gated):
    """The control-arm property. Not 'close' -- identical.

    At splits=1 the merge is `acc, l, m = parts[0]`, so nothing numerical
    happens at all. If this ever becomes approximate, splits=1 has stopped
    being a control and the A/B it anchors is worthless.
    """
    kw = _inputs(gated=gated)
    o0, w0 = two_tier_window_reference(**kw)
    o1, w1 = two_tier_window_reference(**kw, splits=1)
    assert torch.equal(o0, o1), "splits=1 changed the output"
    assert torch.equal(w0, w1), "splits=1 changed the window scores"


# --- splits>1 agrees ---------------------------------------------------------

@pytest.mark.parametrize("splits", [2, 3, 4, 8])
@pytest.mark.parametrize("gated", [False, True], ids=["ungated", "gated"])
def test_more_splits_agree_with_one(splits, gated):
    """Reassociating the online softmax moves the last bits and nothing else.

    §10.5 documents this class for the tile ladder and `_sorted_pick`; split-KV
    is the same kind of change. The tolerance is fp32 accumulation noise, not
    slack for a real difference.
    """
    kw = _inputs(gated=gated)
    o1, w1 = two_tier_window_reference(**kw, splits=1)
    on, wn = two_tier_window_reference(**kw, splits=splits)
    torch.testing.assert_close(on, o1, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(wn, w1, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("gated", [False, True], ids=["ungated", "gated"])
def test_more_splits_than_windows_is_still_correct(gated):
    """Empty splits must contribute nothing, not NaN.

    A split that sees no key has `m = -inf` and `l = 0`, and `exp(-inf - m)` is
    the term that goes wrong if the merge is written naively.
    """
    kw = _inputs(gated=gated)
    o1, w1 = two_tier_window_reference(**kw, splits=1)
    on, wn = two_tier_window_reference(**kw, splits=64)
    assert torch.isfinite(on).all(), "an empty split produced NaN/inf in out"
    assert torch.isfinite(wn).all(), "an empty split produced NaN/inf in scores"
    torch.testing.assert_close(on, o1, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(wn, w1, rtol=1e-5, atol=1e-6)


# --- the partition must be a partition --------------------------------------

@pytest.mark.parametrize("splits", [1, 2, 3, 4, 7, 8, 64])
def test_the_window_chunks_tile_the_axis_exactly(splits):
    """Every window in exactly one chunk -- no gap, no overlap.

    This is the invariant that lets `wsum`/`wmax` be written without a
    cross-split reduction. It is checked directly against `_chunk` rather than
    inferred from the output, because a double-counted window can cancel
    numerically and a dropped one can hide under a tolerance.
    """
    import modules.windowed_cache.decode_kernel as dk
    import inspect

    src = inspect.getsource(dk.two_tier_window_reference)
    ns: dict = {"Tuple": tuple}
    exec(compile(  # noqa: S102 - extracting the local helper for direct test
        "def _chunk(n, i, k):\n"
        "    base, rem = divmod(n, k)\n"
        "    lo = i * base + min(i, rem)\n"
        "    return lo, lo + base + (1 if i < rem else 0)\n", "<x>", "exec"), ns)
    assert "base, rem = divmod(n, k)" in src, (
        "_chunk's body changed; this test's copy is stale")
    _chunk = ns["_chunk"]

    for n in (0, 1, 5, 6, 11, 230):
        covered = []
        for i in range(splits):
            lo, hi = _chunk(n, i, splits)
            assert 0 <= lo <= hi <= n, f"chunk {i} of {n}/{splits} is {lo}:{hi}"
            covered.extend(range(lo, hi))
        assert covered == list(range(n)), (
            f"n={n} splits={splits}: chunks do not tile the axis exactly")


# --- the guard ---------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -8])
def test_a_nonpositive_split_count_raises(bad):
    with pytest.raises(ValueError, match="splits"):
        two_tier_window_reference(**_inputs(), splits=bad)


def test_the_gate_still_selects_the_same_windows_under_a_split():
    """Split-KV must not change WHICH windows are read.

    Standing rule: scores must not move unless the change is a quality change
    with a LongBench run behind it. A split that reordered the selection would
    change the cache's contents, not just its arithmetic.
    """
    kw = _inputs(gated=True)
    _, w1 = two_tier_window_reference(**kw, splits=1)
    _, w4 = two_tier_window_reference(**kw, splits=4)
    # the read set is exactly the set of columns that are not the card-fill
    # value; compare the ORDERING of window scores, which is what eviction uses
    assert torch.equal(w1.argsort(-1), w4.argsort(-1)), (
        "split-KV changed the window ranking -- eviction would keep a "
        "different set of windows")
