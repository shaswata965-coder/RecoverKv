"""Run the gate and fused-decode Triton kernels on CPU at every card width.

Invoked by ``tests/test_card_bits.py`` in a subprocess with ``TRITON_INTERPRET=1``
(the variable must be set before Triton is imported, hence the subprocess).
Exits non-zero on the first mismatch.

For each ``quant_card_bits`` setting and seed:

* the card is built at that width, the gate kernel scores it and the decode
  kernel attends over a real int2 tier with the gate's selection, reading the
  value centroids at ``CardBits.vm``;
* the SAME integer codes are then stored as plain int8 and run again at
  ``CardBits(8, 8, 8, 8)``. The two runs must agree **bitwise** in ``logmass``,
  ``sel``, ``out`` and ``wsum``: a packed field decodes to the same integers
  its int8 twin loads, so any difference is the unpack;
* the gate's ``logmass`` must match :func:`modules.quant.sketch.gate_and_score`
  to fp32 accumulation tolerance.

This is the interpreter, not a GPU compile: it checks the kernels' arithmetic,
masking and lane maps, not their lowering.
"""

from __future__ import annotations

import os
import sys

import torch

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"

import modules.windowed_cache.decode_kernel as dk  # noqa: E402
import modules.windowed_cache.gate_kernel as gk  # noqa: E402
from modules.quant import sketch as S  # noqa: E402
from modules.quant.quantizer import (quantize_key_windows,  # noqa: E402
                                     quantize_value_windows)

# The decode wrapper refuses non-CUDA tensors. Re-execute its module with only
# that guard removed, so everything else -- argument marshalling, the shape
# checks, the tile ladder's first fit -- is the production code.
_GUARD = ('    if not q.is_cuda:\n'
          '        raise RuntimeError("Fused decode kernel requires CUDA tensors.")\n')
_src = open(dk.__file__).read()
assert _src.count(_GUARD) == 1, "decode wrapper's CUDA guard moved; update this"
exec(compile(_src.replace(_GUARD, ""), dk.__file__, "exec"), dk.__dict__)
dk._TUNE_AFTER = 10 ** 9          # first fit only: tuning times on a GPU

B, HKV, REP, D, WS, N = 2, 2, 4, 64, 8, 12
NUM_SINK, BODY = 4, 29
N_SEL = 4


def _inputs(seed: int, bits: S.CardBits) -> dict:
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
    qtier = {"k_codes": by_row(kc), "k_scale": ks.map(by_row),
             "k_zero": kz.map(by_row), "v_codes": by_row(vc),
             "v_scale": vs.map(by_row), "v_zero": vz.map(by_row),
             "cos": ang.cos().to(torch.float16).contiguous(),
             "sin": ang.sin().to(torch.float16).contiguous(),
             "window_size": WS}
    return dict(
        q=torch.randn(B, HKV * REP, D, generator=g).to(torch.float16),
        card=card, anchor=anchor, v_anchor=v_anchor, qtier=qtier,
        k_fp=torch.randn(B, HKV, sfp, D, generator=g).to(torch.float16),
        v_fp=torch.randn(B, HKV, sfp, D, generator=g).to(torch.float16))


def _run(d: dict, bits: S.CardBits) -> dict:
    scaling = D ** -0.5
    sel, logm = gk._gate_triton(d["q"].float(), d["card"], d["anchor"], scaling,
                                N_SEL, bits)
    logm = logm.clone()                   # scratch, reused by the next launch
    c = d["card"]
    out, wsum = dk._decode_triton(
        d["q"], d["k_fp"], d["v_fp"], d["qtier"], scaling, NUM_SINK, None,
        sel.contiguous(), logm, (c.vm_q, c.vm_s, d["v_anchor"].contiguous()),
        bits.vm)
    return {"logm": logm, "sel": sel.clone(), "out": out.clone(),
            "wsum": wsum.clone()}


def _as_int8(card: S.Sketch, bits: S.CardBits) -> S.Sketch:
    """The card's integer codes, unpacked and stored as plain int8."""
    fields = []
    for q, s, w in ((card.mu_q, card.mu_s, bits.mu), (card.v_q, card.v_s, bits.v),
                    (card.t_q, card.t_s, bits.t), (card.vm_q, card.vm_s, bits.vm)):
        ints = S._dq_symb(q, torch.ones_like(s), w).round().to(torch.int8)
        fields += [ints.contiguous(), s]
    return S.Sketch(*fields)


def main() -> int:
    specs = [None, 4, 2, "mu=2,v=4,t=2,vm=8", "mu=8,v=2,t=4,vm=2"]
    for spec in specs:
        bits = S.parse_card_bits(spec)
        for seed in range(2):
            d = _inputs(seed, bits)
            got = _run(d, bits)
            twin = _run(dict(d, card=_as_int8(d["card"], bits)), S.CardBits(8, 8, 8, 8))
            for key in ("logm", "sel", "out", "wsum"):
                if not torch.equal(got[key], twin[key]):
                    print(f"FAIL {tuple(bits)} seed {seed}: {key} differs from "
                          "the same codes stored as int8")
                    return 1
            ref, _ = S.gate_and_score(d["q"].float(), d["card"], d["anchor"],
                                      D ** -0.5, bits)
            err = (got["logm"] - ref).abs().max().item()
            if err > 1e-4:
                print(f"FAIL {tuple(bits)} seed {seed}: gate kernel vs reference "
                      f"logmass max |diff| {err:.2e}")
                return 1
            if not torch.isfinite(got["out"]).all():
                print(f"FAIL {tuple(bits)} seed {seed}: non-finite output")
                return 1
            print(f"ok {tuple(bits)} seed {seed}: bitwise == int8 twin, "
                  f"gate vs reference {err:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
