"""Run the fused decode kernel on CPU against a dequantized reference, at CODE_BITS.

Invoked by ``tests/test_code_bits.py`` in a subprocess with ``TRITON_INTERPRET=1``
(the variable must be set before Triton is imported, hence the subprocess).
Exits non-zero on the first mismatch.

``tests/interp_card_kernels.py`` isolates the CARD unpack by comparing packed
cards against their int8 twins. Nothing there checks the K/V CODE unpack -- the
``>> CODE_BITS * (j % P) & mask`` the Q loop does in registers -- against an
independent reference, and that is the part :data:`CODE_BITS` changes. So, for
a real tier quantized at this build's width:

* the ungated kernel's ``out`` and ``wsum`` must match
  ``two_tier_window_reference`` over the tier dequantized by
  :mod:`modules.quant.quantizer` and rotated with the same fp16 RoPE halves --
  to fp16 output rounding and fp32 accumulation tolerance;
* the gated kernel with EVERY window selected must equal the ungated one
  bitwise (nothing skipped, so nothing filled and nothing renormalised);
* the gated kernel at a real selection must run and stay finite;
* a tier packed at another width must be refused, not read.

This is the interpreter, not a GPU compile: it checks the kernel's arithmetic,
masking and lane maps, not its lowering.
"""

from __future__ import annotations

import os
import sys

import torch

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"

import modules.windowed_cache.decode_kernel as dk  # noqa: E402
import modules.windowed_cache.gate_kernel as gk  # noqa: E402
from modules.quant import sketch as S  # noqa: E402
from modules.quant.quantizer import (CODE_BITS, CODES_PER_BYTE,  # noqa: E402
                                     dequantize_key_windows,
                                     dequantize_value_windows,
                                     quantize_key_windows,
                                     quantize_value_windows)

# Same trick as interp_card_kernels.py: drop only the wrapper's CUDA guard.
_GUARD = ('    if not q.is_cuda:\n'
          '        raise RuntimeError("Fused decode kernel requires CUDA tensors.")\n')
_src = open(dk.__file__).read()
assert _src.count(_GUARD) == 1, "decode wrapper's CUDA guard moved; update this"
exec(compile(_src.replace(_GUARD, ""), dk.__file__, "exec"), dk.__dict__)
dk._TUNE_AFTER = 10 ** 9          # first fit only: tuning times on a GPU

#: (B, H_kv, rep, D, ws, n_windows, num_sink, fp body, n_sel)
SHAPES = ((2, 2, 4, 64, 8, 12, 4, 29, 4),
          (1, 2, 2, 128, 8, 9, 5, 40, 3),
          (1, 1, 4, 64, 16, 5, 3, 37, 2))


def _case(shape, seed):
    B, HKV, REP, D, WS, N, NUM_SINK, BODY, N_SEL = shape
    bits = S.CARD_BITS_DEFAULT
    g = torch.Generator().manual_seed(seed)
    sfp = NUM_SINK + BODY
    anchor = torch.randn(B, HKV, D, generator=g)
    k = anchor[:, None, :, None] + torch.randn(B, N, HKV, WS, D, generator=g)
    k[:, :, :, 3] += 4 * torch.randn(B, N, HKV, D, generator=g)
    v = torch.randn(B, N, HKV, WS, D, generator=g)
    v_anchor = v.mean(dim=(1, 3))
    card = S.build_sketch(
        k.reshape(B * N, HKV, WS, D), anchor.repeat_interleave(N, 0),
        v.reshape(B * N, HKV, WS, D), v_anchor.repeat_interleave(N, 0), bits)
    card = S.Sketch(*(f.reshape(B, N, *f.shape[1:]).contiguous() for f in card))
    kc, ks, kz = quantize_key_windows(k.reshape(B * N, HKV, WS, D))
    vc, vs, vz = quantize_value_windows(v.reshape(B * N, HKV, WS, D))

    def by_row(t):
        return t.reshape(B, N, *t.shape[1:]).contiguous()

    pos = (torch.arange(N * WS) + sfp + 100).reshape(1, -1).expand(B, -1)
    ang = pos[..., None].float() / (10000 ** (torch.arange(D // 2) / (D // 2)))
    cos = ang.cos().to(torch.float16).contiguous()
    sin = ang.sin().to(torch.float16).contiguous()
    qtier = {"k_codes": by_row(kc), "k_scale": ks.map(by_row),
             "k_zero": kz.map(by_row), "v_codes": by_row(vc),
             "v_scale": vs.map(by_row), "v_zero": vz.map(by_row),
             "cos": cos, "sin": sin, "window_size": WS}
    q = torch.randn(B, HKV * REP, D, generator=g).to(torch.float16)
    k_fp = torch.randn(B, HKV, sfp, D, generator=g).to(torch.float16)
    v_fp = torch.randn(B, HKV, sfp, D, generator=g).to(torch.float16)

    # The reference tier: the quantizer's own dequant, the kernel's two-half
    # RoPE with the same fp16 halves, window-major like the kernel's columns.
    kd = dequantize_key_windows(kc, ks, kz, WS, out_dtype=torch.float32)
    kd = kd.reshape(B, N, HKV, WS, D).permute(0, 2, 1, 3, 4).reshape(B, HKV, N * WS, D)
    c, s = cos.float()[:, None], sin.float()[:, None]
    lo, hi = kd[..., :D // 2], kd[..., D // 2:]
    kr = torch.cat([lo * c - hi * s, hi * c + lo * s], dim=-1)
    vd = dequantize_value_windows(vc, vs, vz, D, out_dtype=torch.float32)
    vd = vd.reshape(B, N, HKV, WS, D).permute(0, 2, 1, 3, 4).reshape(B, HKV, N * WS, D)
    return dict(q=q, k_fp=k_fp, v_fp=v_fp, qtier=qtier, card=card, anchor=anchor,
                v_anchor=v_anchor, bits=bits, num_sink=NUM_SINK, ws=WS, sfp=sfp,
                n_body=-(-BODY // WS), n=N, n_sel=N_SEL, B=B, HKV=HKV,
                k_eff=torch.cat([k_fp.float(), kr], dim=2),
                v_eff=torch.cat([v_fp.float(), vd], dim=2))


def main() -> int:
    worst = 0.0
    for shape in SHAPES:
        for seed in range(2):
            d = _case(shape, seed)
            tag = f"CODE_BITS={CODE_BITS} shape={shape} seed={seed}"
            scaling = d["q"].shape[-1] ** -0.5
            q, qt, ns = d["q"], d["qtier"], d["num_sink"]
            cents = (d["card"].vm_q, d["card"].vm_s, d["v_anchor"].contiguous())

            out_u, wsum_u = dk._decode_triton(q, d["k_fp"], d["v_fp"], qt, scaling, ns, None)
            out_u, wsum_u = out_u.clone(), wsum_u.clone()
            ref_out, ref_wsum = dk.two_tier_window_reference(
                q.float(), d["k_eff"], d["v_eff"], scaling, ns, d["ws"], d["n_body"],
                Sfp=d["sfp"], exp2=True)
            e_out = (out_u.float() - ref_out.float()).abs().max().item()
            e_ws = (wsum_u - ref_wsum).abs().max().item()
            worst = max(worst, e_out, e_ws)
            # `out` is fp16 (~1e-3 ulp at these magnitudes); `wsum` is fp32.
            if not (e_out < 2e-3 and e_ws < 1e-4):
                print(f"FAIL {tag}: kernel vs dequantized reference: out {e_out:.2e}, "
                      f"wsum {e_ws:.2e}")
                return 1

            sel_all = torch.arange(d["n"]).expand(d["B"], d["HKV"], d["n"]).contiguous()
            _, logm_all = gk._gate_triton(q.float(), d["card"], d["anchor"], scaling,
                                          d["n"], d["bits"])
            out_a, wsum_a = dk._decode_triton(q, d["k_fp"], d["v_fp"], qt, scaling, ns,
                                              None, sel_all.to(torch.int32),
                                              logm_all.clone(), cents, d["bits"].vm)
            if not (torch.equal(out_a, out_u) and torch.equal(wsum_a, wsum_u)):
                print(f"FAIL {tag}: gated with every window selected differs from "
                      "the ungated read")
                return 1

            sel, logm = gk._gate_triton(q.float(), d["card"], d["anchor"], scaling,
                                        d["n_sel"], d["bits"])
            out_g, wsum_g = dk._decode_triton(q, d["k_fp"], d["v_fp"], qt, scaling, ns,
                                              None, sel.clone().contiguous(),
                                              logm.clone(), cents, d["bits"].vm)
            if not (torch.isfinite(out_g).all() and torch.isfinite(wsum_g).all()):
                print(f"FAIL {tag}: gated output not finite")
                return 1
            print(f"ok {tag}: |kernel - reference| out {e_out:.1e} wsum {e_ws:.1e}; "
                  "gated(all) == ungated bitwise")

    # A tier packed at another width reads the right addresses and the wrong
    # values. It must be refused.
    d = _case(SHAPES[0], 0)
    other = dict(d["qtier"])
    # ws // P is right; twice that is a tier packed at half the codes per byte.
    other["k_codes"] = torch.zeros(*d["qtier"]["k_codes"].shape[:-1],
                                   2 * (d["ws"] // CODES_PER_BYTE), dtype=torch.uint8)
    try:
        dk._decode_triton(d["q"], d["k_fp"], d["v_fp"], other,
                          d["q"].shape[-1] ** -0.5, d["num_sink"], None)
    except RuntimeError as e:
        if "CODE_BITS" not in str(e):
            print(f"FAIL: wrong-width codes raised the wrong error: {e}")
            return 1
    else:
        print("FAIL: a tier packed at another width was read, not refused")
        return 1
    print(f"ok CODE_BITS={CODE_BITS}: a tier at another width is refused; "
          f"worst |kernel - reference| {worst:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
