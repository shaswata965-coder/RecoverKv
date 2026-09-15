"""The gate on the fused decode path: the SEL indirection and the `_run_fused` wiring.

**Why this file carries the weight.** The two things landed here — the kernel's
``widx = tl.load(SEL + …)`` indirection and ``flash_decode._run_fused`` calling the
gate — are the parts that only run on a GPU, and this repo develops on CPU. So the
*Triton lowering* ships unvalidated, under the same contract as every other kernel
here. What must not ship unvalidated is everything around it:

* the **algorithm** — ``two_tier_window_reference(sel=…)`` models the gated Q-tier
  loop tile for tile, lane for lane, including which column a window's score lands
  on and what a skipped column holds. Pinned below against a plain softmax over
  ``[sink | body | selected]``, which shares no code with it.
* the **contract** — ``check_gate_selection`` is the launch path's validation,
  lifted out so it can be called at all (``_decode_triton`` cannot run here).
* the **wiring** — ``_run_fused`` is driven end to end with the kernel replaced by
  that same oracle, and compared against ``gated_decode_step``, the independently
  verified pure-PyTorch statement of the method.

What is left for the GPU is codegen: that Triton lowers this loop the way the
oracle says. ``docs`` for that live in the kernel's ``GATED`` docstring.
"""

from __future__ import annotations

import math

import pytest
import torch

from modules.windowed_cache import flash_decode
from modules.windowed_cache.decode_kernel import (
    check_gate_selection,
    two_tier_decode_reference,
    two_tier_window_reference,
)
from modules.windowed_cache.gated_decode import gated_decode_step

from tests.test_gated_decode import B, D, HKV, HQ, NBODY, REP, SCALING, SINK, WS
from tests.test_gated_decode import _RoPE, _setup


# ---------------------------------------------------------------------------
# A. The SEL indirection, as an algorithm
# ---------------------------------------------------------------------------


def _geom(ws, num_sink, n_body_win=3, n_act=5, seed=0):
    torch.manual_seed(seed)
    b, h_kv, rep, d = 2, 2, 2, 16
    h_q = h_kv * rep
    s = num_sink + (n_body_win + n_act) * ws
    return (torch.randn(b, h_q, d), torch.randn(b, h_kv, s, d),
            torch.randn(b, h_kv, s, d), d ** -0.5, num_sink + n_body_win * ws)


@pytest.mark.parametrize("ws", [1, 2, 3, 4, 5, 8, 12, 16, 32, 64])
@pytest.mark.parametrize("num_sink", [0, 1, 5])
def test_selecting_everything_is_byte_identical_to_the_ungated_path(ws, num_sink):
    """The no-op proof, at the level the kernel works at.

    ``sel = [0, 1, … n-1]`` is the identity indirection, so the gated loop visits
    the same windows in the same order as the ungated one and must land on the
    same bits — not merely close. If this ever drifts, the indirection is not a
    selection, it is a second implementation of the Q-tier loop.
    """
    n_act = 5
    q, k, v, sc, body_end = _geom(ws, num_sink, n_act=n_act, seed=ws * 100 + num_sink)
    o0, w0 = two_tier_window_reference(q, k, v, sc, num_sink, ws, NBODY, body_end)
    sel = (torch.arange(n_act, dtype=torch.int32)
           .view(1, 1, -1).expand(q.shape[0], k.shape[1], n_act).contiguous())
    o1, w1 = two_tier_window_reference(q, k, v, sc, num_sink, ws, NBODY, body_end,
                                       sel=sel)
    assert torch.equal(o0, o1), "identity SEL changed the attention output"
    assert torch.equal(w0, w1), "identity SEL changed the window scores"


@pytest.mark.parametrize("ws", [1, 3, 4, 8, 12, 16])
@pytest.mark.parametrize("num_sink", [0, 2, 5])
@pytest.mark.parametrize("n_sel", [1, 2, 4])
def test_gated_read_equals_plain_softmax_over_the_selected_subset(ws, num_sink, n_sel):
    """A genuine subset, against an expectation that shares no code with it.

    The oracle tiles, carries a running max and rescales in an epilogue; the
    expectation here concatenates ``[sink | body | selected]`` and takes one
    softmax. Agreement means the tiling and the rescale are right, and — because
    the selection differs per ``(row, KV head)`` — that each program resolves its
    own lanes rather than sharing one row's pick.
    """
    n_act = 5
    q, k, v, sc, body_end = _geom(ws, num_sink, n_act=n_act,
                                  seed=ws * 31 + num_sink * 7 + n_sel)
    b, h_kv = q.shape[0], k.shape[1]
    rep = q.shape[1] // h_kv
    sel = torch.stack([
        torch.stack([torch.randperm(n_act)[:n_sel].sort().values
                     for _ in range(h_kv)]) for _ in range(b)]).to(torch.int32)

    got_out, got_w = two_tier_window_reference(
        q, k, v, sc, num_sink, ws, NBODY, body_end, sel=sel)

    k_e, v_e = _subset_kv(k, v, sel, body_end, ws)
    exp_out, ts = two_tier_decode_reference(q, k_e, v_e, sc)
    exp_w = _subset_scores(ts, sel, num_sink, body_end, ws, n_act, rep)

    torch.testing.assert_close(got_out.float(), exp_out.float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got_w, exp_w, rtol=1e-5, atol=1e-6)


def _subset_kv(k, v, sel, body_end, ws):
    """``[sink | body | the windows sel names]`` per ``(row, KV head)``."""
    ks, vs = [], []
    for b in range(k.shape[0]):
        kr, vr = [], []
        for h in range(k.shape[1]):
            cols = sel[b, h].tolist()
            kr.append(torch.cat([k[b, h, :body_end]] + [
                k[b, h, body_end + c * ws: body_end + (c + 1) * ws] for c in cols]))
            vr.append(torch.cat([v[b, h, :body_end]] + [
                v[b, h, body_end + c * ws: body_end + (c + 1) * ws] for c in cols]))
        ks.append(torch.stack(kr))
        vs.append(torch.stack(vr))
    return torch.stack(ks), torch.stack(vs)


def _subset_scores(ts, sel, num_sink, body_end, ws, n_act, rep):
    """Window scores for that subset, each on the column of the window it came from."""
    b, h_q = ts.shape[0], ts.shape[1]
    n_sel = sel.shape[-1]
    out = torch.zeros(b, h_q, NBODY + n_act)
    body = ts[..., num_sink:body_end].float()
    pad = (-body.shape[-1]) % ws
    if pad:
        body = torch.nn.functional.pad(body, (0, pad))
    out[..., :NBODY] = body.reshape(b, h_q, -1, ws).sum(-1)
    qs = ts[..., body_end:].float().reshape(b, h_q, n_sel, ws).sum(-1)
    for r in range(b):
        for hq in range(h_q):
            for j, c in enumerate(sel[r, hq // rep].tolist()):
                out[r, hq, NBODY + c] = qs[r, hq, j]
    return out


def test_a_windows_score_lands_on_its_own_column_not_its_slot():
    """The failure this is built to catch: scores credited to the wrong window.

    Reading windows ``[4, 1]`` must score column 4 and column 1 — not columns 0
    and 1. Getting it wrong produces perfectly plausible probabilities on the
    wrong windows, so eviction quietly keeps the wrong ones and no shape check,
    no sum-to-one check and no output check would notice. Here the selection is
    permuted against slot order so the two orderings cannot coincide.
    """
    ws, num_sink, n_act = 4, 2, 6
    q, k, v, sc, body_end = _geom(ws, num_sink, n_act=n_act, seed=99)
    b, h_kv = q.shape[0], k.shape[1]

    for cols in ([4, 1], [5, 0], [3, 2]):
        sel = (torch.tensor(cols, dtype=torch.int32)
               .view(1, 1, -1).expand(b, h_kv, len(cols)).contiguous())
        _, w = two_tier_window_reference(q, k, v, sc, num_sink, ws, NBODY,
                                         body_end, sel=sel)
        qw = w[..., NBODY:]
        read = set(cols)
        for c in range(n_act):
            if c in read:
                assert (qw[..., c] > 0).all(), f"read window {c} scored zero"
            else:
                assert torch.equal(qw[..., c], torch.zeros_like(qw[..., c])), (
                    f"window {c} was never read but carries a score")


def test_selection_order_does_not_change_the_result():
    """Attention is order-free, so a shuffled ``sel`` must land on the same numbers.

    Pins that nothing in the gated loop depends on ``sel`` being ascending — the
    scores are placed by ``widx``, never by position in the tile.
    """
    ws, num_sink, n_act = 4, 0, 6
    q, k, v, sc, body_end = _geom(ws, num_sink, n_act=n_act, seed=5)
    b, h_kv = q.shape[0], k.shape[1]
    asc = torch.tensor([1, 2, 4], dtype=torch.int32)
    shuf = torch.tensor([4, 1, 2], dtype=torch.int32)

    outs = []
    for order in (asc, shuf):
        sel = order.view(1, 1, -1).expand(b, h_kv, 3).contiguous()
        outs.append(two_tier_window_reference(
            q, k, v, sc, num_sink, ws, NBODY, body_end, sel=sel))
    torch.testing.assert_close(outs[0][0].float(), outs[1][0].float(),
                               rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(outs[0][1], outs[1][1], rtol=1e-6, atol=1e-7)


def test_scores_over_read_windows_and_the_body_still_sum_to_one():
    """The gated softmax is normalized over what it actually attended to.

    A dropped rescale or a stale running max would break this while leaving every
    individual score looking reasonable.
    """
    ws, num_sink, n_act = 8, 0, 6
    q, k, v, sc, body_end = _geom(ws, num_sink, n_act=n_act, seed=11)
    b, h_kv = q.shape[0], k.shape[1]
    sel = (torch.tensor([0, 3, 5], dtype=torch.int32)
           .view(1, 1, -1).expand(b, h_kv, 3).contiguous())
    _, w = two_tier_window_reference(q, k, v, sc, num_sink, ws, NBODY, body_end,
                                     sel=sel)
    torch.testing.assert_close(w.sum(-1), torch.ones(b, q.shape[1]),
                               rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# B. The launch-path contract
# ---------------------------------------------------------------------------


def test_a_well_formed_selection_is_accepted():
    sel = torch.zeros((2, 4, 3), dtype=torch.int32)
    assert check_gate_selection(sel, 2, 4, 10) == 3
    assert check_gate_selection(None, 2, 4, 10) == 0
    assert check_gate_selection(sel, 2, 4, 0) == 0, "an empty tier is never gated"


@pytest.mark.parametrize("bad,msg", [
    (torch.zeros((2, 3), dtype=torch.int32), "sel \\[B, H_kv, n_sel\\]"),
    (torch.zeros((1, 4, 3), dtype=torch.int32), "sel \\[B, H_kv, n_sel\\]"),
    (torch.zeros((2, 2, 3), dtype=torch.int32), "sel \\[B, H_kv, n_sel\\]"),
    (torch.zeros((2, 4, 11), dtype=torch.int32), "selected 11 of 10"),
    (torch.zeros((2, 4, 0), dtype=torch.int32), "selected 0 of 10"),
    (torch.zeros((2, 4, 3), dtype=torch.int64), "int32"),
])
def test_a_malformed_selection_is_an_error_not_a_wrong_answer(bad, msg):
    """Every one of these would run happily on a GPU and index the wrong windows.

    That is the whole argument for checking: the failure mode of a bad ``sel`` is
    not a crash, it is plausible output attended over keys nobody chose.
    """
    with pytest.raises(RuntimeError, match=msg):
        check_gate_selection(bad, 2, 4, 10)


def test_a_non_contiguous_selection_is_rejected():
    """The kernel derives SEL's innermost stride as 1; a view with a gap would
    read whatever sits between the rows."""
    sel = torch.zeros((2, 4, 6), dtype=torch.int32)[:, :, ::2]
    assert not sel.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous sel"):
        check_gate_selection(sel, 2, 4, 10)


# ---------------------------------------------------------------------------
# C. `_run_fused` end to end, kernel replaced by the oracle
# ---------------------------------------------------------------------------


class _StubCache:
    """Just the slot ``_run_fused`` writes ``window_scores`` into."""

    def __init__(self):
        self.cache_kwargs = {0: {}}


def _oracle_kernel(store, rope):
    """Stand in for ``fused_two_tier_decode``: same signature, run on CPU.

    Builds the **full** effective tier — the gate is an indirection into a tier
    that is gathered whole, so the stand-in must be handed the whole thing too or
    the test would not exercise what the kernel does.
    """
    seen = []

    def fake(q, k_fp, v_fp, qtier, scaling, num_sink=0, n_body_win=None,
             sel=None, logmass=None):
        seen.append(sel)
        k_q, v_q, _ = store.effective_q_tier(rope, k_fp.dtype)
        return two_tier_window_reference(
            q, torch.cat([k_fp, k_q], 2), torch.cat([v_fp, v_q], 2),
            scaling, num_sink, WS, n_body_win, k_fp.shape[2], sel=sel,
            logmass=logmass)

    return fake, seen


def _ctx(store, n_sel, order, gated=True):
    slots = store.table.active_order(store.num_active_windows)
    return {
        "layer_idx": 0,
        "qtier": {"window_size": WS},
        "score_meta": (order, store.num_active_windows * WS),
        "gate": ({"card": store.table.gather_sketch(slots),
                  "anchor": store._anchor, "n_sel": n_sel} if gated else None),
        "num_sink": SINK,
        "window_size": WS,
        "scaling": SCALING,
        "cache": _StubCache(),
    }


def _drive(ctx, q, k_fp, v_fp, store, rope, monkeypatch):
    """Run ``_run_fused`` with the kernel replaced; return (out, window_scores)."""
    fake, seen = _oracle_kernel(store, rope)
    monkeypatch.setattr(flash_decode, "fused_two_tier_decode", fake)
    # flash layout: q [B, 1, H_q, D], k/v [B, S_fp, H_kv, D] — seqlen-major, as
    # the wrapper receives them from the module.
    out = flash_decode._run_fused(
        ctx, q[:, None, :, :], k_fp.transpose(1, 2), v_fp.transpose(1, 2))
    return out, ctx["cache"].cache_kwargs[0]["window_scores"], seen


@pytest.mark.parametrize("ratio", [0.25, 0.5, 1.0])
def test_run_fused_reproduces_the_verified_gated_step(ratio, monkeypatch):
    """The wiring test: does ``_run_fused`` compute the method, or something near it?

    ``gated_decode_step`` is the independently verified statement of one gated
    decode step (``tests/test_gated_decode.py``). ``_run_fused`` reaches the same
    place by a different route — the fused gate rather than ``gate_and_select``,
    the kernel's window axis rather than a token-score reduce, and a final
    permutation by ``order``. They must agree, and a non-identity ``order`` is
    used so a mixed-up permutation cannot pass by coincidence.
    """
    store, k_fp, v_fp, q = _setup()
    rope = _RoPE(D)
    n_act = store.num_active_windows
    n_sel = max(1, math.ceil(ratio * n_act))
    w = NBODY + n_act
    order = torch.arange(w - 1, -1, -1).unsqueeze(0).expand(B, w).contiguous()

    ctx = _ctx(store, n_sel, order)
    got_out, got_scores, seen = _drive(ctx, q, k_fp, v_fp, store, rope, monkeypatch)

    want = gated_decode_step(q, k_fp, v_fp, store, rope, SCALING, SINK, ratio=ratio)
    idx = order.unsqueeze(1).expand(B, HQ, w)
    want_scores = torch.gather(want.window_scores.float(), -1, idx)

    assert seen[0] is not None and seen[0].shape == (B, HKV, n_sel)
    torch.testing.assert_close(got_out[:, 0].float(), want.out.float(),
                               rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got_scores.float(), want_scores, rtol=1e-4, atol=1e-5)


def test_the_fused_gate_picks_the_same_windows_as_gate_and_select():
    """Two selectors, one selection — otherwise the agreement above is luck.

    ``gate_and_select`` thresholds on the bound then caps by rank on the estimate;
    the fused gate is a plain top-k on the estimate. At the default ``inf`` margin
    the threshold keeps everything, so the cap decides alone and the two are the
    same operation. This pins that, because a finite margin would make them differ
    silently — which is why ``_gate_ctx`` refuses one.

    Compared as **sets**: the gate no longer sorts its pick, because the decode
    kernel places each window's score by its own column id and so cannot care
    what order the selection arrives in. Which windows are chosen is the claim;
    their order is not.
    """
    from modules.windowed_cache.gate_kernel import fused_gate

    store, _, _, q = _setup()
    n_act = store.num_active_windows
    for ratio in (0.25, 0.5, 1.0):
        n_sel = max(1, math.ceil(ratio * n_act))
        slots = store.table.active_order(n_act)
        sel, _ = fused_gate(q, store.table.gather_sketch(slots), store._anchor,
                            SCALING, n_sel)
        keep, _, _ = store.gate_and_select(q, SCALING, float("inf"), ratio=ratio)
        ref = torch.argsort(~keep, dim=-1, stable=True)[..., :n_sel].sort(-1).values
        assert torch.equal(sel.long().sort(-1).values, ref), (
            f"selections diverge at ratio={ratio}")


def test_every_window_gets_a_score_so_the_gate_cannot_evict_its_own_tier(monkeypatch):
    """The safety property the whole feature turns on.

    The kernel leaves a skipped window's column at zero. Shipping that would make
    every unread window rank last and be evicted — the gate would destroy the tier
    it exists to read less often, and at a 0.25 ratio that is 75% of it per step.
    So ``_run_fused`` must fill those columns from the cards before the scores
    leave. Here: strictly positive everywhere, and the skipped ones no longer zero.
    """
    store, k_fp, v_fp, q = _setup()
    rope = _RoPE(D)
    n_act = store.num_active_windows
    n_sel = max(1, math.ceil(0.25 * n_act))
    w = NBODY + n_act
    order = torch.arange(w).unsqueeze(0).expand(B, w).contiguous()

    ctx = _ctx(store, n_sel, order)
    _, scores, seen = _drive(ctx, q, k_fp, v_fp, store, rope, monkeypatch)

    assert seen[0].shape[-1] == n_sel < n_act, "fixture did not actually gate"
    qw = scores[..., NBODY:]
    assert (qw > 0).all(), (
        f"{int((qw <= 0).sum())} of {qw.numel()} Q-window scores are <= 0; a "
        "skipped window scoring zero ranks last and gets evicted")
    assert torch.isfinite(scores).all()


def test_without_a_gate_context_nothing_is_selected_and_nothing_is_filled(monkeypatch):
    """A store with no cards reads the whole tier — the gate ships off, not broken."""
    store, k_fp, v_fp, q = _setup()
    rope = _RoPE(D)
    n_act = store.num_active_windows
    w = NBODY + n_act
    order = torch.arange(w).unsqueeze(0).expand(B, w).contiguous()

    ctx = _ctx(store, 0, order, gated=False)
    _, scores, seen = _drive(ctx, q, k_fp, v_fp, store, rope, monkeypatch)
    assert seen == [None], "no gate context must mean no selection"

    # Scores cover every non-sink key and nothing else, so they sum to the mass
    # left after the sink takes its share (the scorer never represents sinks).
    k_q, v_q, _ = store.effective_q_tier(rope, k_fp.dtype)
    _, ts = two_tier_decode_reference(
        q, torch.cat([k_fp, k_q], 2), torch.cat([v_fp, v_q], 2), SCALING)
    torch.testing.assert_close(scores.sum(-1).float(),
                               ts[..., SINK:].float().sum(-1),
                               rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("margin,ratio,expect", [
    (float("inf"), 0.25, "gate"),
    (float("inf"), 1.0, "gate"),    # one path: 1.0 gates, selecting everything
    (2.0, 0.25, "raise"),
])
def test_gate_context_is_built_only_when_it_would_mean_something(margin, ratio, expect):
    """``_gate_ctx``'s outcomes. The ratio never selects a different code path.

    A finite ``quant_gate_margin`` raises rather than being accepted: the fused
    gate is a top-k on the card estimate with no margin term, so honouring the
    config is impossible and ignoring it would leave a configured selectivity
    doing nothing at all.
    """
    from types import SimpleNamespace

    from modules.windowed_cache.cache import WindowedCache

    store, _, _, _ = _setup()
    n = store.num_active_windows
    idx = store.table.active_order(n)
    stub = SimpleNamespace(resolved=SimpleNamespace(
        quant_gate_ratio=ratio, quant_gate_margin=margin))

    if expect == "raise":
        with pytest.raises(NotImplementedError, match="quant_gate_margin"):
            WindowedCache._gate_ctx(stub, store, idx, n)
        return
    got = WindowedCache._gate_ctx(stub, store, idx, n)
    assert got["n_sel"] == max(1, math.ceil(ratio * n))
    assert len(got["card"]) == 8 and got["anchor"] is store._anchor
    if ratio >= 1.0:
        assert got["n_sel"] == n, (
            "ratio 1.0 must select every window on the SAME path, not skip the "
            "gate — one implementation, no second route for the control arm")


# ---------------------------------------------------------------------------
# D. Kernel-source invariants (the part no CPU can execute)
# ---------------------------------------------------------------------------


def _kernel_src():
    import pathlib
    src = pathlib.Path("modules/windowed_cache/decode_kernel.py").read_text(
        encoding="utf-8")
    return src[src.index("if _HAS_TRITON:"):src.index("def _pow2_at_least(")]


def test_the_q_loop_dereferences_sel_and_bounds_it():
    """``widx`` must come from SEL under GATED, and be clamped so a dead lane
    cannot derive an out-of-tier offset for the codes or the RoPE row."""
    body = _kernel_src()
    assert "tl.load(SEL + b * selb + kv * selh" in body
    assert "tl.minimum(slot, n_sel - 1)" in body
    assert "qmask = in_tile & (slot < n_sel)" in body


def test_the_gated_store_uses_the_selected_column():
    """Under GATED the score column is ``n_body_win + SEL[...]``, never
    ``n_body_win + slot`` — the bug that puts real scores on skipped windows."""
    body = _kernel_src()
    gated_store = body[body.index("if GATED:\n                    # Slot w0+j"):]
    assert "col = n_body_win + tl.load(" in gated_store


def test_skipped_columns_are_seeded_before_the_q_loop():
    """``wsum``/``wmax`` are ``torch.empty``; the epilogue rescales every column
    it reads. Without this prologue a skipped window's score is whatever the
    allocator handed back."""
    body = _kernel_src()
    assert "for wi0 in range(n_body_win, W_phys, BLOCK_W)" in body
    i_seed = body.index("for wi0 in range(n_body_win, W_phys, BLOCK_W)")
    i_loop = body.index("for w0 in range(0, n_q_iter, BLOCK_NW)")
    assert i_seed < i_loop, "the seed must run before the Q loop, not after"
    assert '-float("inf"), tl.float32' in body[i_seed:i_loop]


def test_the_ungated_path_compiles_the_indirection_away():
    """GATED is a constexpr so the ungated kernel keeps its straight-line
    ``widx = w0 + t_win``; the gate costs it nothing, not even a predicated load."""
    body = _kernel_src()
    assert "GATED: tl.constexpr" in body
    assert "widx = w0 + t_win" in body


# ---------------------------------------------------------------------------
# E. The skipped-window fill, moved into the kernel
# ---------------------------------------------------------------------------


def test_the_kernels_fill_equals_the_host_side_one_it_replaced():
    """Scoring skipped windows moved into the kernel; the numbers must not move.

    It was 22 torch ops per layer per step — 704 launches per token at 32 layers
    — to reshape a small tensor, on a decode path bound by launch count. The
    kernel now does it inside an epilogue pass that already runs, using a max and
    a sum instead of ``fill_skipped_window_scores``'s explicit normalisation
    (the offset cancels in the ratio, so no logarithm is needed). Different
    arithmetic, same answer — which is what this pins, against the original.
    """
    from modules.windowed_cache.scorer import (
        expand_keep_to_query_heads, fill_skipped_window_scores)

    ws, num_sink, n_act = 4, 2, 6
    q, k, v, sc, body_end = _geom(ws, num_sink, n_act=n_act, seed=21)
    b, h_kv, h_q = q.shape[0], k.shape[1], q.shape[1]
    sel = (torch.tensor([4, 1, 5], dtype=torch.int32)
           .view(1, 1, -1).expand(b, h_kv, 3).contiguous())
    logmass = torch.randn(b, h_q, n_act) * 2.0

    _, sparse = two_tier_window_reference(
        q, k, v, sc, num_sink, ws, NBODY, body_end, sel=sel)
    _, filled = two_tier_window_reference(
        q, k, v, sc, num_sink, ws, NBODY, body_end, sel=sel, logmass=logmass)

    keep = torch.zeros((b, h_kv, n_act), dtype=torch.bool)
    keep.scatter_(-1, sel.to(torch.long), True)
    want = fill_skipped_window_scores(
        sparse[..., NBODY:], expand_keep_to_query_heads(keep, h_q), logmass)

    torch.testing.assert_close(filled[..., NBODY:], want, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(filled[..., :NBODY], sparse[..., :NBODY]), (
        "the fp body is never gated and must be untouched")
    assert (filled[..., NBODY:] > 0).all(), "a skipped window still scored zero"


def test_the_fill_leaves_read_windows_exactly_as_they_were():
    """Only skipped columns are written. A read window's score is measured, not
    estimated, and overwriting it with its card would silently degrade eviction
    for the windows the gate was most confident about."""
    ws, num_sink, n_act = 8, 0, 6
    q, k, v, sc, body_end = _geom(ws, num_sink, n_act=n_act, seed=33)
    b, h_kv = q.shape[0], k.shape[1]
    sel = (torch.tensor([0, 3], dtype=torch.int32)
           .view(1, 1, -1).expand(b, h_kv, 2).contiguous())
    logmass = torch.randn(b, q.shape[1], n_act)

    _, sparse = two_tier_window_reference(
        q, k, v, sc, num_sink, ws, NBODY, body_end, sel=sel)
    _, filled = two_tier_window_reference(
        q, k, v, sc, num_sink, ws, NBODY, body_end, sel=sel, logmass=logmass)

    for c in (0, 3):
        torch.testing.assert_close(filled[..., NBODY + c], sparse[..., NBODY + c])


# ---------------------------------------------------------------------------
# F. Gate occupancy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows,nw,sm", [(8, 271, 108), (8, 64, 108),
                                        (256, 271, 108), (16, 1000, 80)])
def test_gate_tiles_are_a_legal_power_of_two(rows, nw, sm):
    from modules.windowed_cache.gate_kernel import gate_window_tiles
    bw = gate_window_tiles(rows, nw, sm)
    assert bw in (16, 32, 64), bw
    assert bw & (bw - 1) == 0


def test_gate_splits_finely_at_small_batch_and_coarsely_at_large():
    """The point of splitting the window axis: fill the machine at ``B=1``.

    One program per ``(row, KV head)`` is 8 programs on Llama-3.1-8B at batch 1,
    which leaves 93% of a 108-SM A100 idle while one serial loop walks the whole
    card set. At batch 32 the rows already fill it, so the tile stays coarse and
    the per-program prologue is not paid 17 times over.
    """
    from modules.windowed_cache.gate_kernel import gate_window_tiles

    nw, sm = 271, 108
    small = 1 * 8                                   # B=1, H_kv=8
    large = 32 * 8                                  # B=32

    bw_s = gate_window_tiles(small, nw, sm)
    blocks_s = small * -(-nw // bw_s)
    assert blocks_s > sm, (
        f"batch-1 gate would use {blocks_s} blocks on {sm} SMs; the split has to "
        "fill the machine or it has not fixed anything")
    assert small < sm, "fixture is wrong: un-split batch 1 should underfill"

    bw_l = gate_window_tiles(large, nw, sm)
    assert bw_l >= bw_s, "large batch should not split more finely than small"
    assert bw_l == 64, "rows already fill the machine; keep the coarse tile"


# ---------------------------------------------------------------------------
# G. Proof the gate ran
# ---------------------------------------------------------------------------


def test_stats_distinguish_a_gated_run_from_an_ungated_one(monkeypatch):
    """``fired`` counts the kernel; ``gated`` counts the gate. They differ.

    A store with no sketch cards hands over ``gate=None``: the fused kernel still
    runs, still produces correct output, and reads the whole tier — a silent
    return to the unoptimised path that no other measurement distinguishes from
    success. This is the signal that does.
    """
    store, k_fp, v_fp, q = _setup()
    rope = _RoPE(D)
    n_act = store.num_active_windows
    n_sel = max(1, math.ceil(0.25 * n_act))
    w = NBODY + n_act
    order = torch.arange(w).unsqueeze(0).expand(B, w).contiguous()

    flash_decode.reset_stats()
    assert flash_decode.stats()["read_fraction"] is None, "nothing has run yet"

    _drive(_ctx(store, n_sel, order), q, k_fp, v_fp, store, rope, monkeypatch)
    s = flash_decode.stats()
    assert s["gated"] == 1
    assert s["read_fraction"] == pytest.approx(n_sel / n_act)
    assert 0.2 < s["read_fraction"] < 0.3, (
        f"a 0.25 ratio realised as {s['read_fraction']:.3f}")

    _drive(_ctx(store, 0, order, gated=False), q, k_fp, v_fp, store, rope,
           monkeypatch)
    s2 = flash_decode.stats()
    assert s2["gated"] == 1, "an ungated run must not count as gated"
    flash_decode.reset_stats()
    assert flash_decode.stats()["gated"] == 0


# ---------------------------------------------------------------------------
# H. The YAML knob must not be inert
# ---------------------------------------------------------------------------


def test_quant_gate_ratio_survives_the_yaml_schema():
    """It has to exist on the LOADER's CacheConfig, not just the cache's own.

    These are two different dataclasses. `quant_gate_ratio` was added to
    `modules.windowed_cache.config.WindowedCacheConfig` and written into the
    generated perf config, but not to `utils.config.CacheConfig` — so every run
    died at load with "unexpected keyword argument". Verifying against the wrong
    schema is what let that through.
    """
    from utils.config import CacheConfig

    assert "quant_gate_ratio" in CacheConfig.__dataclass_fields__
    assert CacheConfig(cache_budget=0.2, quant_gate_ratio=1.0).quant_gate_ratio == 1.0


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, True])
def test_a_bad_gate_ratio_is_rejected_at_load_not_at_decode(bad):
    """Rejected while reading the config, not on the first decode step of a
    benchmarked run that has already spent minutes on prefill."""
    from utils.config import CacheConfig, ConfigValidationError

    with pytest.raises(ConfigValidationError, match="quant_gate_ratio"):
        CacheConfig(cache_budget=0.2, quant_gate_ratio=bad)


def test_the_gate_ratio_reaches_the_cache_config_and_is_not_dropped():
    """The other half: present in the schema but never threaded through.

    That is precisely how `quant_budget_mode` came to be silently inert in three
    runners at once while the YAML said otherwise (ACCURACY_RECOVERY_PLAN.md §2).
    The flash config must receive it; a backend with no gate must not be handed
    a kwarg it cannot take.
    """
    import dataclasses

    from modules.windowed_cache.config import WindowedCacheConfig
    from utils.cache_factory import quant_gate_ratio_kwargs

    assert quant_gate_ratio_kwargs(WindowedCacheConfig, 1.0) == {
        "quant_gate_ratio": 1.0}, "the flash backend must receive the ratio"

    @dataclasses.dataclass
    class _EagerLike:
        window_size: int = 8

    assert quant_gate_ratio_kwargs(_EagerLike, 0.25) == {}, (
        "a backend without a gate must not be handed the kwarg")


def test_every_runner_threads_the_gate_ratio_through():
    """Source check, because the failure is an OMISSION — there is nothing to
    assert against at runtime when a kwarg simply is not passed."""
    import pathlib

    for name in ("perf_runner", "longbench_runner", "ruler_runner",
                 "gsm8k_runner", "ours_parity_runner"):
        src = pathlib.Path(f"modules/evaluation/{name}.py").read_text(
            encoding="utf-8")
        assert "quant_gate_ratio_kwargs" in src, (
            f"{name} builds a cache config without threading quant_gate_ratio; "
            "the YAML knob would be inert there")
