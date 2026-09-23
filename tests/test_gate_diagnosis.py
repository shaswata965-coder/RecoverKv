"""The real-model gate diagnostic's arithmetic and plumbing, on CPU.

``scripts/diagnose_gate_error.py`` only runs where the fused kernels do, so its
numbers are checked here against the decode kernel's own CPU oracle
(``two_tier_window_reference``), and its wrapper against fakes of the two
kernels it calls.
"""

from __future__ import annotations

import math

import torch

from modules.quant.sketch import Sketch, build_sketch, value_centroid
from modules.windowed_cache import gate_diagnosis
from modules.windowed_cache.decode_kernel import two_tier_window_reference
from modules.windowed_cache.gate_kernel import gate_reference

D, WS, SINK = 64, 8, 4


def _step(ratio, seed=0, hkv=2, rep=3, n_q=40, n_fp=10):
    g = torch.Generator().manual_seed(seed)
    s_fp = SINK + n_fp * WS
    S = s_fp + n_q * WS
    q = torch.randn(1, hkv * rep, D, generator=g) * 2.0
    k = torch.randn(1, hkv, S, D, generator=g)
    k[0, :, s_fp::13] += 3.0                     # some hot int2 tokens
    v = torch.randn(1, hkv, S, D, generator=g)
    kq = k[:, :, s_fp:].reshape(1, hkv, n_q, WS, D).permute(0, 2, 1, 3, 4)[0]
    vq = v[:, :, s_fp:].reshape(1, hkv, n_q, WS, D).permute(0, 2, 1, 3, 4)[0]
    a, va = kq.mean(dim=(0, 2)), vq.mean(dim=(0, 2))
    card = Sketch(*[f.unsqueeze(0) for f in build_sketch(kq, a, vq, va)])
    cent = value_centroid(card, va)[0].permute(1, 0, 2).repeat_interleave(rep, 0)[None]
    nbw = -(-(s_fp - SINK) // WS)
    sel, lm = gate_reference(q, *card, a, D ** -0.5, max(1, math.ceil(ratio * n_q)))
    out_g, _ = two_tier_window_reference(q, k, v, D ** -0.5, SINK, WS, nbw, Sfp=s_fp,
                                         sel=sel, logmass=lm, centroids=cent)
    out_f, wsum_f = two_tier_window_reference(q, k, v, D ** -0.5, SINK, WS, nbw, Sfp=s_fp)
    return out_g, out_f, wsum_f, nbw, sel, lm


def test_the_full_read_measures_as_exact():
    """ratio 1.0 is the no-op: zero output error, recall 1, whatever the card."""
    out_g, out_f, wsum_f, nbw, sel, lm = _step(1.0)
    m = gate_diagnosis.gate_error_metrics(out_g, out_f, wsum_f, nbw, sel, lm)
    assert m["out_err"].max() < 1e-5
    assert torch.allclose(m["recall"], torch.ones_like(m["recall"]), atol=1e-5)


def test_a_partial_read_reports_its_cost():
    out_g, out_f, wsum_f, nbw, sel, lm = _step(0.25)
    m = gate_diagnosis.gate_error_metrics(out_g, out_f, wsum_f, nbw, sel, lm)
    assert set(m) == {"out_err", "card_tv", "recall", "q_share"}
    assert all(t.shape == (1, 6) for t in m.values())
    assert (m["out_err"] > 0).all()
    assert ((m["recall"] > 0) & (m["recall"] <= 1 + 1e-6)).all()
    assert ((m["card_tv"] >= 0) & (m["card_tv"] <= 1 + 1e-6)).all()
    assert ((m["q_share"] > 0) & (m["q_share"] < 1)).all()


def test_card_tv_is_zero_for_an_exact_card_and_one_for_a_disjoint_one():
    out_g, out_f, wsum_f, nbw, sel, _ = _step(0.25)
    wq = wsum_f[..., nbw:]
    exact = (wq / wq.sum(-1, keepdim=True)).log()
    m = gate_diagnosis.gate_error_metrics(out_g, out_f, wsum_f, nbw, sel, exact)
    assert m["card_tv"].max() < 1e-5
    worst = torch.full_like(exact, -1e4)
    worst[..., 0] = 0.0                       # all card mass on one window...
    wsum_f = wsum_f.clone()
    wsum_f[..., nbw] = 0.0                    # ...that truly has none
    m = gate_diagnosis.gate_error_metrics(out_g, out_f, wsum_f, nbw, sel, worst)
    assert (m["card_tv"] > 0.999).all()


def test_the_wrapper_measures_only_the_first_steps_and_passes_the_step_through(monkeypatch):
    """The shipped output is returned untouched; the full read is extra."""
    from modules.windowed_cache import decode_kernel, flash_decode, gate_kernel

    out_g, out_f, wsum_f, nbw, sel, lm = _step(0.25)
    hq, hkv = out_g.shape[1], sel.shape[1]
    shipped = out_g.unsqueeze(1)                              # flash layout

    monkeypatch.setattr(flash_decode, "_run_fused", lambda *a: shipped)
    monkeypatch.setattr(gate_kernel, "fused_gate", lambda *a, **k: (sel, lm))
    monkeypatch.setattr(decode_kernel, "fused_two_tier_decode",
                        lambda *a, **k: (out_f, wsum_f))
    stats, uninstall = gate_diagnosis.install(max_steps=2)
    try:
        cache = object()
        q = torch.zeros(1, 1, hq, D)
        kv = torch.zeros(1, SINK + nbw * WS, hkv, D)
        ctx = {"cache": cache, "n_kv_heads": hkv, "num_sink": SINK,
               "window_size": WS, "scaling": D ** -0.5, "qtier": {},
               "gate": {"card": None, "anchor": None, "n_sel": sel.shape[-1],
                        "head_norm": True}}
        for _step_i in range(4):
            for layer in (0, 1):
                got = flash_decode._run_fused(dict(ctx, layer_idx=layer), q, kv, kv)
                assert got is shipped
    finally:
        uninstall()
    assert flash_decode._run_fused is not None
    assert sorted(stats) == [0, 1]
    assert len(stats[0]["out_err"]) == 2 * hq                 # 2 of 4 steps measured
    s = gate_diagnosis.summarize(stats)
    assert set(s["layers"]) == {0, 1}
    assert s["overall"]["recall_median"] <= 1.0
