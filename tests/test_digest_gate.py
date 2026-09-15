"""The decode-time digest gate (DIGEST_GATED_DECODE_PLAN.md §3.2, §3.3, §8.3).

Collection needs a GPU box: importing anything under ``modules.windowed_cache``
pulls in ``score_kernel``, which builds its Triton autotuner at import time (§9).
Nothing here *runs* on the GPU — the kernel is stubbed so the tests pin the
selection and the score-scatter, which is where the plumbing bugs live.

What these pin:

- **§8.3 gate correctness.** With ``k >= n_active`` the gate is a no-op and
  hands the kernel the very same object, so "the plumbing is right" is separable
  from "the selection is good".
- **The score map is subset correctly.** ``window_scores`` stays full-width —
  a gated-out window still owns its column, it just contributes nothing to it —
  while the kernel emits only the selected columns. Getting that mapping wrong
  scatters attention mass onto the wrong windows, which would then be evicted on
  the wrong evidence.
- **Only the Q half is gated** (§0.1 / §3.3): the fp body columns, which carry
  the sinks and the local window, come through untouched.
"""

import pytest
import torch

import modules.windowed_cache.flash_decode as fd
from modules.quant.digest import (estimate_aabb, estimate_aabb_prepared,
                                  prepare_aabb)

B, H_Q, H_KV, D, WS = 1, 4, 2, 8, 4
NUM_SINK, N_BODY_WIN, N_ACTIVE = 2, 3, 5


def _qtier(top_k=None, centers=(0.1, 0.9, 0.2, 0.8, 0.3), n=N_ACTIVE):
    """A Q-tier hand-off whose window ``i`` is tagged ``i`` in ``k_scale``.

    The digest of window ``i`` is the degenerate box ``lo = hi = centers[i]``, so
    against an all-positive query the bound is monotone in ``centers`` and the
    expected selection is known by construction rather than by re-running the
    ranking the code under test computes.
    """
    c = torch.tensor(centers).reshape(1, n, 1, 1)
    k_scale = torch.zeros(B, n, H_KV, D)
    k_scale[0, :, 0, 0] = torch.arange(n, dtype=torch.float32)   # identity tag
    q = {
        "k_codes": torch.arange(n).reshape(1, n, 1, 1, 1)
                   .expand(B, n, H_KV, D, WS // 4).to(torch.uint8).contiguous(),
        "k_scale": k_scale,
        "k_zero": torch.zeros(B, n, H_KV, D),
        "v_codes": torch.zeros(B, n, H_KV, WS, D // 4, dtype=torch.uint8),
        "v_scale": torch.zeros(B, n, H_KV, WS),
        "v_zero": torch.zeros(B, n, H_KV, WS),
        # token-major, tagged by window so the re-windowing can be checked
        "cos": torch.arange(n).reshape(1, n, 1, 1).expand(B, n, WS, D // 2)
               .reshape(B, n * WS, D // 2).float().contiguous(),
        "sin": torch.arange(n).reshape(1, n, 1, 1).expand(B, n, WS, D // 2)
               .reshape(B, n * WS, D // 2).float().contiguous(),
        "window_size": WS,
    }
    lo = c.expand(B, n, H_KV, D).contiguous()
    q["digest_lo_t"], q["digest_hi_t"] = prepare_aabb(lo, lo)
    if top_k is not None:
        q["digest_top_k"] = top_k
    return q


def _q_hd():
    """An all-positive query, so the AABB bound is monotone in the box centre."""
    return torch.ones(B, H_Q, D)


@pytest.fixture(autouse=True)
def _cpu_digest_select(monkeypatch):
    """Run the gate's selection through the PyTorch reference on CPU.

    Production scores digests with a Triton kernel (``digest_kernel.digest_select``)
    because the ~20 host launches it replaces were half the reason the gate cost
    more than it saved. It is Triton-or-raise by design, so these CPU tests
    substitute :func:`estimate_aabb_prepared` — the same arithmetic, and the
    reference the GPU kernel is checked against in
    ``tests/test_digest_gate_kernel.py``.
    """
    def cpu_select(q_hd, lo_t, hi_t, k):
        est = estimate_aabb_prepared(q_hd.unsqueeze(2), lo_t, hi_t)
        rank = est.amax(dim=1).squeeze(1)
        return rank.topk(k, dim=-1, sorted=False).indices.to(torch.int32).contiguous()
    monkeypatch.setattr(fd, "digest_select", cpu_select)


# -- selection ---------------------------------------------------------------


def test_no_digests_is_a_pure_passthrough():
    q = _qtier(top_k=None)
    del q["digest_lo_t"], q["digest_hi_t"]
    gated, sel = fd._gate_qtier(q, _q_hd())
    assert sel is None and gated is q


def test_k_covering_every_window_is_a_no_op(sel_k=N_ACTIVE):
    """§8.3: gate everything in and nothing may change, object included."""
    q = _qtier(top_k=sel_k)
    gated, sel = fd._gate_qtier(q, _q_hd())
    assert sel is None, "k >= n_active must not select"
    assert gated is q, "a no-op gate must hand on the same object, not a copy"


def test_k_above_n_active_is_also_a_no_op():
    gated, sel = fd._gate_qtier(_qtier(top_k=N_ACTIVE + 3), _q_hd())
    assert sel is None


def test_selection_takes_the_highest_bounds():
    """The SET matters; the order does not, so do not assert one.

    `sel` is deliberately unsorted — sorting it was a multi-kernel CUDA radix
    pass per layer per step, bought nothing, and cost measurable launches. The
    kernel resolves every column through `sel`, so any order is correct.
    """
    gated, sel = fd._gate_qtier(_qtier(top_k=2), _q_hd())
    assert sorted(sel[0].tolist()) == [1, 3], "expected the two largest boxes"
    assert sel.dtype == torch.int32 and sel.is_contiguous(), \
        "the kernel offsets a raw pointer by sel; dtype/layout are a contract"


def test_the_gate_selects_without_copying_anything():
    """The whole point of the redesign: selection is an indirection, not a gather.

    Compacting the Q tier host-side cost +2723 launches and +26.7 ms of CUDA at
    B=8 to save ~16 ms. If a future change starts materializing again, the code
    fields will stop being the same objects and this fails.
    """
    q = _qtier(top_k=2)
    gated, sel = fd._gate_qtier(q, _q_hd())
    for f in ("k_codes", "k_scale", "k_zero", "v_codes", "v_scale", "v_zero",
              "cos", "sin"):
        assert gated[f] is q[f], f"{f} was copied; the gate must only pass `sel`"
    assert gated["sel"] is sel


def test_the_resident_axis_is_left_at_full_width():
    """The kernel needs every resident window addressable; `sel` picks among them."""
    gated, sel = fd._gate_qtier(_qtier(top_k=3), _q_hd())
    assert gated["k_codes"].shape[1] == N_ACTIVE
    assert gated["cos"].shape == (B, N_ACTIVE * WS, D // 2)
    assert sel.shape == (B, 3)
    assert gated["window_size"] == WS


def test_the_head_reduction_is_max_not_mean():
    """One head wanting a window badly must be enough to keep it (§3.2).

    Window 0 is the clear winner for a single head and near-worthless to the
    other three; under a mean it loses, under a max it wins. The gate exists to
    catch exactly this, so the test is written so the two policies disagree.
    """
    n = 3
    q = _qtier(top_k=1, centers=(0.0,) * n, n=n)
    lo = torch.zeros(B, n, H_KV, D)
    hi = torch.zeros(B, n, H_KV, D)
    # KV head 0 serves query heads 0-1, KV head 1 serves 2-3, so a window that
    # only KV head 0 wants scores on half the query heads and zero on the rest —
    # its mean is halved while its max is not. Calibrated so window 0 wins the
    # max (80 > 48) and loses the mean (40 < 48): the policies really disagree.
    hi[0, 0, 0, :] = 10.0     # window 0: huge, but only for KV head 0
    hi[0, 1, :, :] = 6.0      # window 1: moderate for BOTH heads -> wins a mean
    hi[0, 2, :, :] = 0.5
    q["digest_lo_t"], q["digest_hi_t"] = prepare_aabb(lo, hi)

    _, sel = fd._gate_qtier(q, _q_hd())
    assert sel.tolist() == [[0]], "max-over-heads must keep the window one head wants"

    est = estimate_aabb(_q_hd().unsqueeze(2), lo, hi)
    assert est.mean(dim=1).squeeze(1).argmax(-1).item() == 1, \
        "the scenario no longer distinguishes max from mean"


# -- the score scatter -------------------------------------------------------


BODY_BASE = 100.0


def _stub_kernel(monkeypatch, seen):
    """Replace the Triton kernel with one that reports which windows it got.

    Body column ``i`` emits ``BODY_BASE + i``; Q column ``j`` emits the identity
    tag of the window it was handed. The scatter can then be checked by value.
    """
    def stub(q_hd, k_fp, v_fp, qtier, scaling, num_sink, n_body_win):
        sel = qtier.get("sel")
        # Mirrors the real kernel: it reads the FULL resident arrays and walks
        # only the windows `sel` names, emitting one column per selected window.
        idx = (torch.arange(qtier["k_codes"].shape[1]) if sel is None
               else sel[0].long())
        seen["n_sel"] = len(idx)
        w = torch.empty(B, H_Q, n_body_win + len(idx))
        w[..., :n_body_win] = BODY_BASE + torch.arange(n_body_win, dtype=torch.float32)
        w[..., n_body_win:] = qtier["k_scale"][0, idx, 0, 0]
        return torch.zeros(B, H_Q, D), w
    monkeypatch.setattr(fd, "fused_two_tier_decode", stub)


class _Cache:
    def __init__(self):
        self.cache_kwargs = {0: {}}


# Physical axis is [body ‖ Q]; body windows are the NEWEST (largest ids) since
# the local region trails the sequence, so the merged order puts the Q windows
# first. That is the real layout, not a convenient one.
BODY_IDS = [10, 11, 12]
Q_IDS = [1, 3, 5, 7, 9]
ORDER = torch.tensor([[3, 4, 5, 6, 7, 0, 1, 2]])       # argsort(BODY_IDS + Q_IDS)


def _ctx(qtier):
    from modules.quant.effective import invert_order
    return {
        "layer_idx": 0,
        "qtier": qtier,
        "score_meta": (ORDER, N_ACTIVE * WS),
        "inv_order": invert_order(ORDER),
        "num_sink": NUM_SINK,
        "window_size": WS,
        "scaling": 1.0,
        "cache": _Cache(),
    }


def _flash_inputs():
    s_fp = NUM_SINK + N_BODY_WIN * WS
    return (torch.ones(B, 1, H_Q, D),
            torch.zeros(B, s_fp, H_KV, D),
            torch.zeros(B, s_fp, H_KV, D))


def test_ungated_scatter_is_unchanged(monkeypatch):
    seen = {}
    _stub_kernel(monkeypatch, seen)
    ctx = _ctx(_qtier(top_k=N_ACTIVE))
    fd._run_fused(ctx, *_flash_inputs())
    scores = ctx["cache"].cache_kwargs[0]["window_scores"]

    assert seen["n_sel"] == N_ACTIVE, "nothing should have been gated out"
    # merged order: the five Q windows (tags 0-4), then the three body windows.
    expected = [0., 1., 2., 3., 4., BODY_BASE, BODY_BASE + 1, BODY_BASE + 2]
    assert scores.shape == (B, H_Q, len(expected))
    assert scores[0, 0].tolist() == expected


def test_gated_scatter_keeps_full_width_and_zeroes_the_skipped(monkeypatch):
    seen = {}
    _stub_kernel(monkeypatch, seen)
    ctx = _ctx(_qtier(top_k=2))                 # selects windows 1 and 3
    fd._run_fused(ctx, *_flash_inputs())
    scores = ctx["cache"].cache_kwargs[0]["window_scores"]

    assert seen["n_sel"] == 2, "the kernel must receive only the selected windows"
    assert scores.shape == (B, H_Q, N_BODY_WIN + N_ACTIVE), \
        "window_scores must stay full-width; a gated window still owns its column"
    expected = [0., 1., 0., 3., 0., BODY_BASE, BODY_BASE + 1, BODY_BASE + 2]
    assert scores[0, 0].tolist() == expected
    for h in range(H_Q):
        assert scores[0, h].tolist() == expected


def test_the_fp_body_columns_are_never_gated(monkeypatch):
    """§0.1 at the kernel level: sinks and locals live in the fp body."""
    seen = {}
    _stub_kernel(monkeypatch, seen)
    ctx = _ctx(_qtier(top_k=1))
    fd._run_fused(ctx, *_flash_inputs())
    scores = ctx["cache"].cache_kwargs[0]["window_scores"]
    body = scores[0, 0, -N_BODY_WIN:]
    assert body.tolist() == [BODY_BASE, BODY_BASE + 1, BODY_BASE + 2]


def test_a_wrong_score_map_raises_rather_than_misscattering(monkeypatch):
    seen = {}
    _stub_kernel(monkeypatch, seen)
    ctx = _ctx(_qtier(top_k=2))
    ctx["score_meta"] = (ORDER[:, :-1], N_ACTIVE * WS)      # one column short
    with pytest.raises(RuntimeError, match="score_meta permutes"):
        fd._run_fused(ctx, *_flash_inputs())


# -- §4.1 decomposition modes -----------------------------------------------


def _ctx_mode(mode, top_k=2):
    q = _qtier(top_k=top_k)
    q["digest_gate_mode"] = mode
    return _ctx(q)


def test_eviction_mode_attends_ungated_but_zeroes_skipped_scores(monkeypatch):
    """Isolates the gate's EVICTION effect: exact attention, gated ranking.

    The kernel runs over every window (so the output is what the gate-off run
    would produce), and the gate is applied only to the score columns. Any
    quality delta under this arm is the changed eviction ranking and nothing
    else.
    """
    seen = {}
    _stub_kernel(monkeypatch, seen)
    ctx = _ctx_mode("eviction")
    fd._run_fused(ctx, *_flash_inputs())
    scores = ctx["cache"].cache_kwargs[0]["window_scores"]

    assert seen["n_sel"] == N_ACTIVE, "eviction mode must attend over ALL windows"
    # Windows 1 and 3 are selected; 0, 2, 4 are zeroed. Body untouched.
    expected = [0., 1., 0., 3., 0., BODY_BASE, BODY_BASE + 1, BODY_BASE + 2]
    assert scores[0, 0].tolist() == expected


def test_attention_mode_gates_output_but_scores_ungated(monkeypatch):
    """Isolates the gate's ATTENTION effect: approximate output, honest ranking.

    Two kernel runs — one gated for the output, one ungated for the scores — so
    eviction sees exactly what it would have seen with the gate off.
    """
    calls = []

    def stub(q_hd, k_fp, v_fp, qtier, scaling, num_sink, n_body_win):
        sel = qtier.get("sel")
        idx = (torch.arange(qtier["k_codes"].shape[1]) if sel is None
               else sel[0].long())
        calls.append(len(idx))
        w = torch.empty(B, H_Q, n_body_win + len(idx))
        w[..., :n_body_win] = BODY_BASE + torch.arange(n_body_win, dtype=torch.float32)
        w[..., n_body_win:] = qtier["k_scale"][0, idx, 0, 0]
        return torch.zeros(B, H_Q, D), w

    monkeypatch.setattr(fd, "fused_two_tier_decode", stub)
    ctx = _ctx_mode("attention")
    fd._run_fused(ctx, *_flash_inputs())
    scores = ctx["cache"].cache_kwargs[0]["window_scores"]

    assert calls == [2, N_ACTIVE], \
        "expected a gated run for the output then an ungated run for the scores"
    # No window is zeroed: eviction sees the full, ungated picture.
    expected = [0., 1., 2., 3., 4., BODY_BASE, BODY_BASE + 1, BODY_BASE + 2]
    assert scores[0, 0].tolist() == expected


def test_modes_are_inert_when_the_gate_does_not_engage(monkeypatch):
    """A mode must do nothing when k covers everything — no second kernel run."""
    seen = {}
    _stub_kernel(monkeypatch, seen)
    for mode in ("both", "attention", "eviction"):
        ctx = _ctx_mode(mode, top_k=N_ACTIVE)
        fd._run_fused(ctx, *_flash_inputs())
        scores = ctx["cache"].cache_kwargs[0]["window_scores"]
        expected = [0., 1., 2., 3., 4., BODY_BASE, BODY_BASE + 1, BODY_BASE + 2]
        assert scores[0, 0].tolist() == expected, mode

