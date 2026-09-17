"""The int2 scale/zero grid is one byte per entry — what that cost, and why.

The grid is fixed overhead: it does not shrink with the bit-width, so at int2 an
fp16 grid was 48.5% of a window's codes-plus-grid — two bytes of precision
wrapping a two-bit code. Narrowing it to one byte against a shared fp16 scale is
the largest memory lever in the repo that costs no accuracy worth having, and
this file is the evidence for both halves of that claim.

The measurements here are reconstruction error on synthetic keys shaped like
real ones (a per-head common mode with massive-activation channels, one hot
token per window). They are not LongBench. What they can settle is the
*encoding* question — which of several 1-byte layouts to store — and that is
what they are used for; the quality claim is a separate run.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from modules.quant.quantizer import (  # noqa: E402
    GRID_GROUP,
    QGrid,
    _quantize_grid,
    bytes_per_q_window,
    dequantize_key_windows,
    dequantize_value_windows,
    grid_bytes_per_head,
    grid_group,
    quantize_key_windows,
    quantize_value_windows,
)

D, WS, H = 128, 8, 8


def _keys(n=64, hot=6.0, spread=1.0, seed=0):
    """Keys shaped like real ones — the fixture ``tests/test_sketch.py`` uses."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(H, D, generator=g) * 0.4
    a[:, 3] += 40.0
    a[:, 17] -= 25.0
    k = a[None, :, None, :] + torch.randn(n, H, WS, D, generator=g) * spread
    if hot:
        k[:, :, 0, :] += torch.randn(n, H, D, generator=g) * hot
    return k


def _wide_gain_keys(n=64, seed=0, ratio=1000.0):
    """Keys whose per-channel gain spans ``ratio``.

    This is what a massive-activation channel *is*, taken to the point where a
    shared exponent has to cope with it. The aggregate error cannot see the
    damage — the big channels carry the norm — so the test below reads the worst
    channel instead.
    """
    g = torch.Generator().manual_seed(seed)
    gain = torch.exp(torch.linspace(0, math.log(ratio), D)).repeat(H, 1)
    dc = torch.randn(H, D, generator=g) * 20.0
    dc[:, 3] += 60.0
    k = dc[None, :, None, :] + torch.randn(n, H, WS, D, generator=g) * gain[None, :, None, :]
    k[:, :, 0, :] += torch.randn(n, H, D, generator=g) * gain[None] * 6.0
    return k


def _fp16_grid_roundtrip(x, group_dim):
    """What the dequant produced when the grid was stored fp16 — the baseline."""
    x32 = x.float()
    mx = x32.amax(group_dim, keepdim=True)
    mn = x32.amin(group_dim, keepdim=True)
    scale = torch.where(mx == mn, torch.ones_like(mx), (mx - mn) / 3.0)
    sg = scale.to(torch.float16).float()
    zg = mn.to(torch.float16).float()
    return torch.round((x32 - zg) / sg).clamp_(0, 3) * sg + zg


def _grouped_roundtrip(x, group_dim, group):
    """The shipped path, but with ``group`` entries per shared scale."""
    x32 = x.float()
    mx = x32.amax(group_dim, keepdim=True)
    mn = x32.amin(group_dim, keepdim=True)
    scale = torch.where(mx == mn, torch.ones_like(mx), (mx - mn) / 3.0)

    def enc(v, signed):
        v = v.squeeze(group_dim)
        lead, n = v.shape[:-1], v.shape[-1]
        vg = v.reshape(*lead, n // group, group)
        lev = 127.0 if signed else 255.0
        amax = (vg.abs() if signed else vg).amax(-1, keepdim=True)
        s = torch.where(amax > 0, amax / lev, torch.ones_like(amax)).to(torch.float16)
        s32 = s.float().clamp_min(torch.finfo(torch.float32).tiny)
        q = torch.round(vg / s32)
        q = q.clamp_(-lev, lev) if signed else q.clamp_(1.0, lev)
        return (q * s32).reshape(*lead, n).unsqueeze(group_dim)

    sg, zg = enc(scale, False), enc(mn, True)
    return torch.round((x32 - zg) / sg).clamp_(0, 3) * sg + zg


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _worst_channel(a, b):
    """Per-channel relative error, worst channel. The metric a norm hides."""
    num = (a - b).pow(2).sum(dim=(0, 2))
    den = b.pow(2).sum(dim=(0, 2)).clamp_min(1e-12)
    return (num / den).sqrt().amax().item()


# ---------------------------------------------------------------------------
# the byte accounting — this is what the budget resolver spends
# ---------------------------------------------------------------------------


def test_the_grid_is_one_byte_per_entry_plus_its_group_scales():
    """272 B/head at D=128, not 512. The 12 over 260 are the four fp16 scales."""
    assert grid_bytes_per_head(D) == 2 * D + 4 * (D // GRID_GROUP) == 272
    # ws=8 is shorter than a group, so a value grid shares one scale per head.
    assert grid_group(WS) == WS
    assert grid_bytes_per_head(WS) == 2 * WS + 4 == 20


def test_a_q_window_costs_6432_bytes_before_its_card():
    """The number the memory claim rests on, at the shipped geometry."""
    assert bytes_per_q_window(H, D, WS) == 6432
    fp16_grid_era = H * ((D * WS) // 2 + 4 * D + 4 * WS)
    assert fp16_grid_era == 8448
    assert bytes_per_q_window(H, D, WS) / fp16_grid_era == pytest.approx(0.761, abs=1e-3)


def test_the_budget_buys_21_percent_more_int2_windows():
    """What the narrowing is FOR, read off the resolver rather than asserted.

    Same config, same bytes: only the price of an int2 window moved.
    """
    from modules.quant.sketch import sketch_bytes_per_head

    b_fp = (2 * 2 * H * D) * WS
    b_card = H * sketch_bytes_per_head(D, WS)
    m_evict = 10_000_000                       # a fixed byte allowance
    q = 0.7

    def n_q(b_q):
        return int((q * m_evict) // b_q)

    now = n_q(bytes_per_q_window(H, D, WS) + b_card)
    before = n_q(H * ((D * WS) // 2 + 4 * D + 4 * WS) + b_card)
    assert now > before
    assert now / before == pytest.approx(1.21, abs=0.02)
    # And an int2 window is still much cheaper than an fp16 one, card included.
    assert b_fp / (bytes_per_q_window(H, D, WS) + b_card) == pytest.approx(3.4, abs=0.1)


# ---------------------------------------------------------------------------
# what it cost
# ---------------------------------------------------------------------------


def test_the_one_byte_grid_costs_under_one_percent_of_reconstruction():
    """Keys and values, against the fp16 grid, on the realistic fixture.

    Near-free because design §2 pins the codes to the STORED grid: a rounded
    scale repositions the four int2 levels and the codes re-fit against them, so
    the int2 error (~13% on keys) swamps the grid's own rounding.
    """
    k_err, f_err = [], []
    for seed in range(6):
        k = _keys(seed=seed)
        codes, scale, zero = quantize_key_windows(k)
        got = dequantize_key_windows(codes, scale, zero, WS, out_dtype=torch.float32)
        k_err.append(_rel(got, k))
        f_err.append(_rel(_fp16_grid_roundtrip(k, 2), k))
    assert sum(k_err) / len(k_err) <= sum(f_err) / len(f_err) * 1.01

    v_err, vf_err = [], []
    for seed in range(6):
        v = torch.randn(64, H, WS, D, generator=torch.Generator().manual_seed(seed))
        codes, scale, zero = quantize_value_windows(v)
        got = dequantize_value_windows(codes, scale, zero, D, out_dtype=torch.float32)
        v_err.append(_rel(got, v))
        vf_err.append(_rel(_fp16_grid_roundtrip(v, 3), v))
    assert sum(v_err) / len(v_err) <= sum(vf_err) / len(vf_err) * 1.01


def test_the_logits_move_by_under_one_percent():
    """``q·k`` is what attention actually reads, so it gets its own check."""
    ours, fp16 = [], []
    for seed in range(6):
        k = _keys(seed=seed)
        codes, scale, zero = quantize_key_windows(k)
        got = dequantize_key_windows(codes, scale, zero, WS, out_dtype=torch.float32)
        ref = _fp16_grid_roundtrip(k, 2)
        q = torch.randn(4, H, D, generator=torch.Generator().manual_seed(100 + seed))

        def logits(x):
            return torch.einsum("nhwd,bhd->bhnw", x, q) / math.sqrt(D)

        true = logits(k)
        ours.append((logits(got) - true).norm().item() / true.norm().item())
        fp16.append((logits(ref) - true).norm().item() / true.norm().item())
    assert sum(ours) / len(ours) <= sum(fp16) / len(fp16) * 1.01


def test_grouping_is_what_makes_the_narrowing_safe():
    """One scale per head passes every aggregate check and ruins a channel.

    This is the measurement that chose ``GRID_GROUP``. On keys whose channel
    gains span 1000x, sharing one fp16 scale across all 128 channels leaves the
    worst channel decoding to noise (rel err > 1.0, i.e. worse than returning
    zeros) while the Frobenius norm moves by ~0.1% and reports nothing. Groups of
    32 put the worst channel back at the fp16 grid's own figure.
    """
    whole, grouped, fp16 = [], [], []
    for seed in range(4):
        k = _wide_gain_keys(seed=seed)
        whole.append(_worst_channel(_grouped_roundtrip(k, 2, D), k))
        grouped.append(_worst_channel(_grouped_roundtrip(k, 2, GRID_GROUP), k))
        fp16.append(_worst_channel(_fp16_grid_roundtrip(k, 2), k))
    whole_m = sum(whole) / len(whole)
    grouped_m = sum(grouped) / len(grouped)
    fp16_m = sum(fp16) / len(fp16)

    assert whole_m > 3 * fp16_m, (
        "the un-grouped grid stopped being the bad case this test exists to "
        f"refuse (worst channel {whole_m:.3f} vs fp16 {fp16_m:.3f})"
    )
    assert grouped_m <= fp16_m * 1.05, (
        f"grouped worst-channel error {grouped_m:.4f} against fp16's {fp16_m:.4f}"
    )
    # ...and the aggregate is blind to all of it, which is why it is not the test.
    assert _rel(_grouped_roundtrip(_wide_gain_keys(), 2, D), _wide_gain_keys()) == \
        pytest.approx(_rel(_fp16_grid_roundtrip(_wide_gain_keys(), 2), _wide_gain_keys()),
                      rel=0.01)


# ---------------------------------------------------------------------------
# the invariants a stored format has to keep
# ---------------------------------------------------------------------------


def test_a_scale_code_is_never_zero():
    """The fit divides by the decoded scale; a zero there is a NaN cache.

    A window whose channel is nearly constant beside one that is not rounds to
    zero on the group's scale, so the clamp is load-bearing, not defensive.
    """
    x = torch.full((4, H, D), 1e-6)
    x[..., 0] = 100.0                      # forces a huge group scale
    g = _quantize_grid(x, signed=False)
    assert int(g.q.min()) >= 1
    assert bool((g.decode() > 0).all())

    # The other half of the same hazard, and the reason for the fp16 clamp: a
    # group whose entries are all tiny divides its amax by 255 and underflows
    # fp16 -- a SHARED scale of zero decodes the whole group to zero, so the fit
    # divides by zero. The entries here are representable; their 255th is not.
    tiny = torch.full((2, H, D), 3e-6)
    g = _quantize_grid(tiny, signed=False)
    assert bool((g.s > 0).all()) and bool((g.decode() > 0).all())
    assert torch.allclose(g.decode(), tiny, rtol=0.01)


def test_the_codes_are_fit_to_the_grid_that_will_be_read():
    """Dequantize twice: bit-identical. Re-quantize the dequant: same codes.

    The pinned-grid property, which is what lets a dormant window be reactivated
    rather than re-quantized (design §10). Note what is NOT claimed: the grid
    itself is not reproduced bit-for-bit by a re-quantization, because a group's
    shared scale is a reduction over its entries and an fp16 dequant moves them
    by an ulp. That costs nothing, because the design never re-quantizes — a
    re-demotion flips ``slot_active`` and touches neither codes nor grid.
    """
    k = _keys(n=16, seed=3)
    codes, scale, zero = quantize_key_windows(k)
    once = dequantize_key_windows(codes, scale, zero, WS, out_dtype=torch.float16)
    again = dequantize_key_windows(codes, scale, zero, WS, out_dtype=torch.float16)
    assert torch.equal(once, again)

    codes2, scale2, zero2 = quantize_key_windows(once.float())
    assert torch.equal(codes2, codes)
    assert torch.equal(zero2.q, zero.q)
    drift = ((scale2.decode() - scale.decode()).abs() / scale.decode()).max()
    assert float(drift) < 0.02


def test_a_qgrid_is_never_indexed_like_a_tuple():
    """``grid[0]`` is ``q``, not the first row — so the code uses ``.map``."""
    k = _keys(n=4, seed=1)
    _, scale, _ = quantize_key_windows(k)
    assert scale[0] is scale.q                      # the trap, stated
    assert torch.equal(scale.map(lambda t: t[0]).q, scale.q[0])
    assert torch.equal(scale.map(lambda t: t[0]).s, scale.s[0])

    r = scale.reshape(2, 2, H, D)
    assert r.q.shape == (2, 2, H, D)
    assert r.s.shape == (2, 2, H, D // grid_group(D))
    assert torch.equal(r.decode().reshape(scale.q.shape), scale.decode())


def test_a_grid_axis_that_cannot_be_grouped_is_refused():
    """A ragged tail would need a masked load in the kernel's inner loop."""
    assert grid_group(64) == GRID_GROUP
    with pytest.raises(ValueError, match="multiple of GRID_GROUP"):
        grid_group(48)


def test_the_decoded_grid_is_exactly_what_the_kernel_recomputes():
    """The kernel multiplies the byte by its group's fp16 scale, in fp32.

    It cannot call ``decode``; it re-derives it from two loads. If the order or
    the dtypes here ever drift from that, the GPU reads a different grid than
    the codes were fit to and nothing on this box would notice.
    """
    k = _keys(n=8, seed=5)
    _, scale, _ = quantize_key_windows(k)
    g = grid_group(D)
    kernel_side = (scale.q.to(torch.float32)
                   * scale.s.to(torch.float32).repeat_interleave(g, dim=-1))
    assert torch.equal(kernel_side, scale.decode())


def test_the_grid_survives_a_store_round_trip():
    """Through :class:`QuantSlotTable` — the grid is eight columns, not four."""
    from modules.quant.slots import GRID_FIELDS, QuantSlotTable

    t = QuantSlotTable(batch_size=2, n_slots=4, window_size=WS, head_dim=D,
                       num_kv_heads=H, device=torch.device("cpu"))
    k = _keys(n=2 * 3, seed=7).reshape(2, 3, H, WS, D)
    v = torch.randn_like(k)
    kc, ks, kz = quantize_key_windows(k.reshape(6, H, WS, D))
    vc, vs, vz = quantize_value_windows(v.reshape(6, H, WS, D))
    slot_idx = torch.arange(3).unsqueeze(0).expand(2, 3).contiguous()
    t.write(slot_idx, torch.ones(2, 3, dtype=torch.bool),
            torch.arange(3).unsqueeze(0).expand(2, 3).contiguous(),
            kc.reshape(2, 3, *kc.shape[1:]),
            ks.reshape(2 * 3, *ks.q.shape[1:]).map(lambda x: x.reshape(2, 3, *x.shape[1:])),
            kz.reshape(2 * 3, *kz.q.shape[1:]).map(lambda x: x.reshape(2, 3, *x.shape[1:])),
            vc.reshape(2, 3, *vc.shape[1:]),
            vs.reshape(2 * 3, *vs.q.shape[1:]).map(lambda x: x.reshape(2, 3, *x.shape[1:])),
            vz.reshape(2 * 3, *vz.q.shape[1:]).map(lambda x: x.reshape(2, 3, *x.shape[1:])),
            torch.zeros(2, 3, WS, dtype=torch.long))
    assert len(GRID_FIELDS) == 8
    got = t.gather(slot_idx)
    assert isinstance(got[1], QGrid) and isinstance(got[4], QGrid)
    assert torch.equal(got[1].q, ks.q.reshape(6, H, D))
    assert torch.equal(got[1].s, ks.s.reshape(6, H, D // grid_group(D)))
    back = dequantize_key_windows(got[0], got[1], got[2], WS, out_dtype=torch.float32)
    assert torch.equal(
        back, dequantize_key_windows(kc, ks, kz, WS, out_dtype=torch.float32))
