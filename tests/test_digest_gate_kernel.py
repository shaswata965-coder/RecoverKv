"""The gate's kernel indirection, on GPU (DIGEST_GATED_DECODE_PLAN.md §11.2).

**Why this file exists.** The digest gate first shipped by compacting the Q tier
host-side, on plan §3.3's assumption that "the kernel itself needs no changes".
Measured, that cost +2723 kernel launches and +26.7 ms of CUDA at B=8 to save
~16 ms — a net TPOT regression of ~80%. The fix passes the kernel a ``sel``
index vector so it reads the selected windows *in place*. That is a change to a
Triton kernel which, per ``tests/test_decode_kernel.py``, has **no active GPU
correctness test** — so the change needs its own, and this is it.

The check that carries the weight is **equivalence against the path the gate used
to take**: gather the windows host-side and run the ungated kernel, versus leave
them resident and run the kernel with ``sel``. Those must agree bit-for-bit. That
compares the new fast path against an already-exercised one rather than against a
fresh oracle, which is the strongest thing available here.

CUDA only — ``fused_two_tier_decode`` is Triton-or-raise by design.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused decode kernel requires CUDA"
)

from modules.windowed_cache.decode_kernel import fused_two_tier_decode

B, H_Q, H_KV, D, WS = 2, 8, 2, 128, 8
N_PHYS, NUM_SINK, N_BODY_WIN = 24, 5, 3


def _qtier(device, seed=0):
    """A Q-tier hand-off with all ``N_PHYS`` windows resident."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    half = D // 2
    r = lambda *sh: torch.rand(*sh, generator=g)
    return {
        "k_codes": torch.randint(0, 256, (B, N_PHYS, H_KV, D, WS // 4),
                                 dtype=torch.uint8, generator=g).to(device),
        "k_scale": (r(B, N_PHYS, H_KV, D) * 0.1).half().to(device),
        "k_zero": (r(B, N_PHYS, H_KV, D) - 0.5).half().to(device),
        "v_codes": torch.randint(0, 256, (B, N_PHYS, H_KV, WS, D // 4),
                                 dtype=torch.uint8, generator=g).to(device),
        "v_scale": (r(B, N_PHYS, H_KV, WS) * 0.1).half().to(device),
        "v_zero": (r(B, N_PHYS, H_KV, WS) - 0.5).half().to(device),
        "cos": r(B, N_PHYS * WS, half).float().to(device).contiguous(),
        "sin": r(B, N_PHYS * WS, half).float().to(device).contiguous(),
        "window_size": WS,
    }


def _inputs(device, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s_fp = NUM_SINK + N_BODY_WIN * WS
    q = torch.randn(B, H_Q, D, generator=g).half().to(device)
    k = torch.randn(B, H_KV, s_fp, D, generator=g).half().to(device)
    v = torch.randn(B, H_KV, s_fp, D, generator=g).half().to(device)
    return q, k, v


def _gather_host(qtier, sel):
    """What the gate used to do: compact the resident axis down to ``sel``."""
    k = sel.shape[1]
    base = torch.arange(B, device=sel.device).unsqueeze(1) * N_PHYS
    fi = (sel.long() + base).reshape(-1)
    out = dict(qtier)
    for f in ("k_codes", "k_scale", "k_zero", "v_codes", "v_scale", "v_zero"):
        t = qtier[f]
        flat = t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])
        out[f] = flat[fi].reshape(B, k, *t.shape[2:]).contiguous()
    for f in ("cos", "sin"):
        t = qtier[f].reshape(B, N_PHYS, WS, -1)
        flat = t.reshape(B * N_PHYS, *t.shape[2:])
        out[f] = flat[fi].reshape(B, k * WS, -1).contiguous()
    return out


def test_identity_sel_matches_no_sel_bitwise():
    """``sel = [0..n-1]`` must reproduce the ungated kernel exactly.

    This isolates the indirection from the selection: if it fails, the pointer
    arithmetic is wrong, independent of which windows the digest picked.
    """
    dev = torch.device("cuda")
    q, k, v = _inputs(dev)
    qt = _qtier(dev)
    out_ref, ws_ref = fused_two_tier_decode(q, k, v, qt, D ** -0.5, NUM_SINK, N_BODY_WIN)

    ident = (torch.arange(N_PHYS, dtype=torch.int32, device=dev)
             .unsqueeze(0).expand(B, N_PHYS).contiguous())
    out_sel, ws_sel = fused_two_tier_decode(
        q, k, v, {**qt, "sel": ident}, D ** -0.5, NUM_SINK, N_BODY_WIN)

    assert torch.equal(out_ref, out_sel), "identity sel changed the output"
    assert torch.equal(ws_ref, ws_sel), "identity sel changed the window scores"


@pytest.mark.parametrize("k", [1, 5, 12, N_PHYS - 1])
def test_sel_indirection_equals_a_host_gather(k):
    """The load-bearing check: indirection == the compaction it replaced.

    Same windows, same order, two different ways of getting them to the kernel.
    Any disagreement means the kernel is reading the wrong windows — a silent
    wrong-output bug, which is why this is an exact comparison.
    """
    dev = torch.device("cuda")
    q, kf, v = _inputs(dev)
    qt = _qtier(dev)

    # UNSORTED on purpose: production sends `topk(sorted=False)` output, so the
    # kernel must be correct for an arbitrary permutation, not just ascending.
    g = torch.Generator(device="cpu").manual_seed(k)
    sel = torch.stack([torch.randperm(N_PHYS, generator=g)[:k]
                       for _ in range(B)]).to(torch.int32).to(dev).contiguous()

    out_gather, ws_gather = fused_two_tier_decode(
        q, kf, v, _gather_host(qt, sel), D ** -0.5, NUM_SINK, N_BODY_WIN)
    out_sel, ws_sel = fused_two_tier_decode(
        q, kf, v, {**qt, "sel": sel}, D ** -0.5, NUM_SINK, N_BODY_WIN)

    assert out_sel.shape == out_gather.shape
    assert ws_sel.shape == ws_gather.shape == (B, H_Q, N_BODY_WIN + k)
    assert torch.equal(out_gather, out_sel), f"k={k}: output differs from the gather"
    assert torch.equal(ws_gather, ws_sel), f"k={k}: scores differ from the gather"


def test_emitted_column_order_follows_sel_order():
    """Emitted Q column ``j`` belongs to the window ``sel[:, j]`` names.

    Tested by permutation rather than by magnitude: run the SAME three windows in
    ascending and in reversed order. Attention is order-free over a set of keys,
    so ``out`` must agree; and if column ``j`` really tracks ``sel[j]``, the three
    emitted Q columns must come back reversed.

    An earlier version of this test tried to make one window "hot" by scaling its
    ``k_scale`` up. That does not work: scaling a key amplifies it in whichever
    direction it already points, which is *away* from q for about half the query
    heads, so the loud window wins on some heads and scores near zero on others.
    Permutation needs no such assumption.

    ``allclose`` rather than ``equal``: reordering the windows changes which of
    them share a ``BLOCK_NW`` tile, so the online-softmax running max is reached
    by a different accumulation order and the last bits move. That is rounding,
    not disagreement.
    """
    dev = torch.device("cuda")
    q, kf, v = _inputs(dev)
    qt = _qtier(dev)

    asc = torch.tensor([[3, 17, 20]] * B, dtype=torch.int32, device=dev).contiguous()
    dsc = torch.tensor([[20, 17, 3]] * B, dtype=torch.int32, device=dev).contiguous()

    out_a, ws_a = fused_two_tier_decode(
        q, kf, v, {**qt, "sel": asc}, D ** -0.5, NUM_SINK, N_BODY_WIN)
    out_d, ws_d = fused_two_tier_decode(
        q, kf, v, {**qt, "sel": dsc}, D ** -0.5, NUM_SINK, N_BODY_WIN)

    assert torch.allclose(out_a, out_d, atol=1e-2, rtol=1e-2), \
        "the same window SET gave different attention output under reordering"
    # fp body columns are untouched by the gate (§0.1 / §3.3).
    assert torch.allclose(ws_a[..., :N_BODY_WIN], ws_d[..., :N_BODY_WIN],
                          atol=1e-3, rtol=1e-3), "a body column moved with sel"
    # Q columns must be the exact reversal.
    assert torch.allclose(ws_a[..., N_BODY_WIN:],
                          ws_d[..., N_BODY_WIN:].flip(-1), atol=1e-3, rtol=1e-3), \
        "emitted Q column j does not track sel[:, j]"


def test_non_int32_sel_is_rejected():
    """The kernel offsets a raw pointer by `sel`; a wrong dtype reads garbage."""
    dev = torch.device("cuda")
    q, kf, v = _inputs(dev)
    qt = _qtier(dev)
    bad = torch.zeros(B, 4, dtype=torch.int64, device=dev)
    with pytest.raises(RuntimeError, match="contiguous int32"):
        fused_two_tier_decode(q, kf, v, {**qt, "sel": bad},
                              D ** -0.5, NUM_SINK, N_BODY_WIN)


# ---------------------------------------------------------------------------
# Fused digest scoring (modules/windowed_cache/digest_kernel.py)
# ---------------------------------------------------------------------------

from modules.quant.digest import build_aabb, estimate_aabb_prepared, prepare_aabb
from modules.windowed_cache.digest_kernel import digest_rank, digest_select

G_WIN = 137          # deliberately not a multiple of BLOCK_G, to exercise masking


def _digests(device, dtype=torch.float16, seed=3):
    """Prepared AABB corners over keys with per-channel scale spread."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    scale = torch.logspace(-1, 1, D).reshape(1, 1, 1, 1, D)
    keys = torch.randn(B, G_WIN, H_KV, WS, D, generator=g) * scale
    lo, hi = build_aabb(keys)
    lo_t, hi_t = prepare_aabb(lo.to(dtype), hi.to(dtype))
    return lo_t.to(device), hi_t.to(device)


def test_fused_rank_matches_the_pytorch_reference():
    """The kernel must agree with estimate_aabb_prepared + amax.

    That reference is what every CPU test and the plan's bound proofs are written
    against, so a divergence here means the gate is ranking on something other
    than the documented bound.
    """
    dev = torch.device("cuda")
    lo_t, hi_t = _digests(dev)
    q = torch.randn(B, H_Q, D, generator=torch.Generator(device="cpu").manual_seed(5)
                    ).half().to(dev)

    got = digest_rank(q, lo_t, hi_t)
    ref = estimate_aabb_prepared(q.unsqueeze(2), lo_t, hi_t).amax(dim=1).squeeze(1)

    assert got.shape == ref.shape == (B, G_WIN)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_fused_rank_reduces_with_max_not_mean():
    """§3.2: one head wanting a window badly must carry it."""
    dev = torch.device("cuda")
    n = 3
    lo = torch.zeros(B, n, H_KV, D)
    hi = torch.zeros(B, n, H_KV, D)
    hi[:, 0, 0, :] = 10.0      # window 0: huge, KV head 0 only
    hi[:, 1, :, :] = 6.0       # window 1: moderate on both -> wins a mean
    hi[:, 2, :, :] = 0.5
    lo_t, hi_t = prepare_aabb(lo.half(), hi.half())
    q = torch.ones(B, H_Q, D).half().to(dev)

    rank = digest_rank(q, lo_t.to(dev), hi_t.to(dev))
    assert rank.argmax(-1).tolist() == [0] * B, "max-over-heads was not applied"

    mean = estimate_aabb_prepared(q.unsqueeze(2), lo_t.to(dev), hi_t.to(dev)
                                  ).mean(dim=1).squeeze(1)
    assert mean.argmax(-1).tolist() == [1] * B, \
        "the scenario no longer distinguishes max from mean"


def test_the_bound_still_upper_bounds_real_dot_products():
    """The one-sided contract, through the fused path.

    An over-estimate costs a window that did not need reading; an under-estimate
    silently drops one the model wanted. Only the second is a correctness bug,
    so the assert is one-sided.
    """
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(9)
    scale = torch.logspace(-1, 1, D).reshape(1, 1, 1, 1, D)
    keys = (torch.randn(B, G_WIN, H_KV, WS, D, generator=g) * scale)
    lo, hi = build_aabb(keys)
    lo_t, hi_t = prepare_aabb(lo.half(), hi.half())
    q = torch.randn(B, H_Q, D, generator=g).half()

    rank = digest_rank(q.to(dev), lo_t.to(dev), hi_t.to(dev)).cpu().float()

    rep = H_Q // H_KV
    q_g = q.float().reshape(B, H_KV, rep, D)
    kk = keys.permute(0, 2, 1, 3, 4)                     # [B, H_kv, G, WS, D]
    real = torch.einsum("bhrd,bhgsd->bhrgs", q_g, kk).amax(-1)   # [B,H_kv,rep,G]
    real = real.reshape(B, H_Q, G_WIN).amax(1)                   # max over heads

    assert (rank >= real - 5e-2).all(), "the fused bound under-estimated a real q.k"


def test_digest_select_returns_the_top_k_as_int32():
    dev = torch.device("cuda")
    lo_t, hi_t = _digests(dev)
    q = torch.randn(B, H_Q, D, generator=torch.Generator(device="cpu").manual_seed(11)
                    ).half().to(dev)
    k = 20
    sel = digest_select(q, lo_t, hi_t, k)

    assert sel.shape == (B, k)
    assert sel.dtype == torch.int32 and sel.is_contiguous()
    rank = digest_rank(q, lo_t, hi_t)
    # the selected set must be exactly the k largest, order irrelevant
    want = rank.topk(k, dim=-1).indices.sort(dim=-1).values
    assert torch.equal(sel.long().sort(dim=-1).values, want)
