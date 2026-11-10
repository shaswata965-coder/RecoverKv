"""Run the MLA gate + fused-decode Triton kernels on CPU, against the CPU oracle.

Invoked by ``tests/test_mla_kernels.py`` in a subprocess with
``TRITON_INTERPRET=1`` (it must be set before Triton is imported). Exits non-zero
on the first mismatch.

The geometry is DeepSeek-V2's at toy width: ONE latent KV head shared by every
query head (``rep = H_q = 16``), a packed key whose halves are
``[latent HA | RoPE HB]``, the value aliased to the latent lanes, and the card's
``mu`` serving as the value centroid. The int2 tier is real (codes from
``quantize_key_windows`` on pre-RoPE keys, cards from ``build_sketch`` on
post-RoPE keys) and the oracle is the shipped kernel's own CPU statement,
``decode_kernel.two_tier_window_reference``, fed the dequantized, rotated tier.

Checked, gated (the production path) and ungated (the ``ratio = 1.0`` shape):

* the gate kernel's ``logmass`` == ``sketch.gate_and_score``, and its pick ==
  the reference's share-union pick;
* the decode kernel's output, on the LATENT lanes the attention module reads,
  == the oracle's, and its RoPE lanes are zero;
* every per-window score == the oracle's, skipped windows' fills included.

This is the interpreter, not a GPU compile: it checks arithmetic, masking and
lane maps, not lowering or speed.
"""

from __future__ import annotations

import os
import sys

import torch

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"

import modules.windowed_cache.decode_kernel as dk  # noqa: E402
import modules.windowed_cache.gate_kernel as gk  # noqa: E402
import modules.windowed_cache.mla_decode_kernel as mk  # noqa: E402
from modules.quant import sketch as S  # noqa: E402
from modules.quant.quantizer import (dequantize_key_windows,  # noqa: E402
                                     quantize_key_windows)

mk._require_cuda = lambda q: None      # the only production line skipped
dk._TUNE_AFTER = 10 ** 9               # first fit only: tuning times on a GPU

B, HQ, WS, N = 2, 16, 8, 12
HA, HB = 32, 16                         # latent / RoPE lanes per half
HALF = HA + HB
D = 2 * HALF                            # 96: three 32-entry grid groups
NUM_SINK, BODY = 4, 29
N_SEL = 4


def _tables(pos: torch.Tensor):
    """Packed half-tables: cos=1, sin=0 on latent lanes, angles on RoPE lanes."""
    ang = pos[..., None].float() / (10000 ** (torch.arange(HB) / HB))
    cos = torch.cat([torch.ones(*pos.shape, HA), ang.cos()], -1)
    sin = torch.cat([torch.zeros(*pos.shape, HA), ang.sin()], -1)
    return cos, sin                                       # [.., HALF]


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF half-split RoPE over the full packed head, as the cache applies it."""
    c, s = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
    rh = torch.cat([-x[..., HALF:], x[..., :HALF]], -1)
    return x * c + rh * s


def _inputs(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    sfp = NUM_SINK + BODY
    pos = (torch.arange(N * WS) + sfp + 100).reshape(1, -1).expand(B, -1)
    cos, sin = _tables(pos)                               # [B, N*WS, HALF]
    cos16, sin16 = cos.to(torch.float16), sin.to(torch.float16)
    anchor_pre = torch.randn(B, 1, D, generator=g)
    k_pre = anchor_pre[:, None, :, None] + torch.randn(B, N, 1, WS, D, generator=g)
    k_pre[:, :, :, 3] += 4 * torch.randn(B, N, 1, D, generator=g)   # a hot token
    # post-RoPE keys, at the tables' stored (fp16) precision, for the card
    cw = cos16.float().reshape(B, N, 1, WS, HALF)
    sw = sin16.float().reshape(B, N, 1, WS, HALF)
    k_post = _rotate(k_pre, cw, sw)
    anchor = k_post.mean(dim=(1, 3))                      # [B, 1, D]
    card = S.build_sketch(
        k_post.reshape(B * N, 1, WS, D), anchor.repeat_interleave(N, 0),
        k_post.reshape(B * N, 1, WS, D), anchor.repeat_interleave(N, 0),
        S.CARD_BITS_DEFAULT, value_alias=True)
    card = S.Sketch(*(None if f is None else f.reshape(B, N, *f.shape[1:]).contiguous()
                      for f in card))
    kc, ks, kz = quantize_key_windows(k_pre.reshape(B * N, 1, WS, D))

    def by_row(t):
        return t.reshape(B, N, *t.shape[1:]).contiguous()

    qtier = {"k_codes": by_row(kc), "k_scale": ks.map(by_row),
             "k_zero": kz.map(by_row), "v_codes": None, "v_scale": None,
             "v_zero": None, "cos": cos16.contiguous(), "sin": sin16.contiguous(),
             "window_size": WS, "value_alias": True, "mla_split": (HA, HB)}
    # the oracle's view of the same tier: dequantized, rotated at fp16 tables
    deq = dequantize_key_windows(kc, ks, kz, WS, out_dtype=torch.float32)
    deq = deq.reshape(B, N, 1, WS, D)
    k_q = _rotate(deq, cw, sw).permute(0, 2, 1, 3, 4).reshape(B, 1, N * WS, D)
    k_fp = torch.randn(B, 1, sfp, D, generator=g).to(torch.float16)
    q = torch.randn(B, HQ, D, generator=g).to(torch.float16)
    return dict(q=q, k_fp=k_fp, qtier=qtier, card=card, anchor=anchor.contiguous(),
                k_q=k_q.to(torch.float16), sfp=sfp)


def _check(seed: int, gated: bool) -> str:
    d = _inputs(seed)
    scaling = 0.11
    q, card, anchor = d["q"], d["card"], d["anchor"]
    n_sel = N_SEL if gated else N
    sel, logm = gk._gate_triton(q.float(), card, anchor, scaling, n_sel,
                                S.CARD_BITS_DEFAULT)
    sel, logm = sel.clone().contiguous(), logm.clone()
    ref_logm, _ = S.gate_and_score(q.float(), card, anchor, scaling)
    err = (logm - ref_logm).abs().max().item()
    if err > 1e-4:
        raise AssertionError(f"gate logmass vs gate_and_score {err:.2e}")
    share = S.group_share(ref_logm, 1)
    picked = torch.zeros(B, 1, N, dtype=torch.bool)
    picked.scatter_(-1, sel.long(), True)
    # Exactly the reference's pick, except where the n_sel-th and (n_sel+1)-th
    # shares tie (to 1e-5, well above the 4e-6 the kernel's logmass is off by):
    # there either window is the share-union's pick, and two top-k calls may
    # break the tie differently. At D = 576 random keys saturate some heads, so
    # several windows tie at a share of exactly 0.
    kth = share.sort(-1, descending=True).values[..., n_sel - 1:n_sel]
    must, may = share > kth + 1e-5, share >= kth - 1e-5
    if not (bool((picked | ~must).all()) and bool((~picked | may).all())):
        raise AssertionError("gate pick differs from the share-union reference")

    cen = (card.mu_q, card.mu_s, anchor)              # the aliased centroid
    out, wsum = mk.mla_decode_triton(
        q, d["k_fp"], d["qtier"], scaling, NUM_SINK, None,
        sel if gated else None, logm if gated else None,
        cen if gated else None, S.CARD_BITS_DEFAULT.mu)
    out, wsum = out.clone().float(), wsum.clone()

    k_eff = torch.cat([d["k_fp"], d["k_q"]], dim=2)
    n_body_win = -(-BODY // WS)
    centroids = None
    if gated:
        vc = S.value_centroid(card, anchor)               # [B, N, 1, D]
        centroids = vc.permute(0, 2, 1, 3).expand(B, HQ, N, D)
    r_out, r_wsum = dk.two_tier_window_reference(
        q, k_eff, k_eff, scaling, NUM_SINK, WS, n_body_win, Sfp=d["sfp"],
        exp2=True, sel=sel if gated else None, logmass=logm if gated else None,
        centroids=centroids)
    lat = torch.cat([torch.arange(HA), HALF + torch.arange(HA)])
    rope = torch.cat([HA + torch.arange(HB), HALF + HA + torch.arange(HB)])
    e_out = (out[..., lat] - r_out.float()[..., lat]).abs().max().item()
    scale_out = r_out.float()[..., lat].abs().max().item()
    if e_out > 2e-2 * max(scale_out, 1.0):
        raise AssertionError(f"latent output vs oracle {e_out:.3e} (scale {scale_out:.2f})")
    if out[..., rope].abs().max().item() != 0.0:
        raise AssertionError("RoPE lanes of the output must be zero")
    e_ws = (wsum - r_wsum).abs().max().item()
    if e_ws > 2e-3:
        raise AssertionError(f"window scores vs oracle {e_ws:.3e}")
    return f"out {e_out:.1e} wsum {e_ws:.1e} gate {err:.1e}"


def _e2e() -> str:
    """The production route, end to end, on a tiny DeepSeek-V2.

    owned MLA forward -> cache.update arms the fused path -> run_pending ->
    flash_decode._run_fused -> the gate kernel -> the MLA decode kernel -> scores
    back into the cache. At ``quant_gate_ratio = 1.0`` it must reproduce the
    materialize path (the CPU oracle) step for step; at 0.25 the proof counters
    must say every fused layer gated, at the configured read fraction.
    """
    import math
    import modules.windowed_cache.flash_decode as fd
    from models.deepseek_v2 import attention as A
    from models.deepseek_v2.loader import install_recoverkv_mla, make_cache
    from models.deepseek_v2.upstream.configuration_deepseek import DeepseekV2Config
    from models.deepseek_v2.upstream.modeling_deepseek import DeepseekV2ForCausalLM
    from modules.windowed_cache.config import WindowedCacheConfig

    # The production dispatcher, minus only its CUDA guard (as interp_card_kernels
    # does for the shipped kernel); the gate goes through its Triton kernel.
    guard = ('    if not (_HAS_TRITON and q.is_cuda):\n'
             '        reason = "triton not installed" if not _HAS_TRITON else "not on CUDA"\n')
    src = open(dk.__file__).read()
    assert src.count(guard) == 1, "fused_two_tier_decode's guard moved; update this"
    exec(compile(src.replace(guard, '    if not _HAS_TRITON:\n        reason = ""\n'),
                 dk.__file__, "exec"), dk.__dict__)
    fd.fused_two_tier_decode = dk.fused_two_tier_decode
    dk._TUNE_AFTER = 10 ** 9          # the exec reset it; first fit only on CPU
    fd.fused_gate = lambda q, card, anchor, scaling, n_sel, bits=None: \
        gk._gate_triton(q.float(), card, anchor, scaling, n_sel, S._as_bits(bits))

    cfg = DeepseekV2Config(
        vocab_size=96, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=16, num_key_value_heads=16,
        kv_lora_rank=64, q_lora_rank=None, qk_rope_head_dim=32, qk_nope_head_dim=16,
        v_head_dim=16, n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        first_k_dense_replace=1, topk_method="greedy", n_group=1, topk_group=1,
        max_position_embeddings=2048, attn_implementation="eager",
        rope_scaling={"type": "yarn", "factor": 40, "original_max_position_embeddings":
                      2048, "beta_fast": 32, "beta_slow": 1, "mscale": 0.707,
                      "mscale_all_dim": 0.707})
    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(cfg).eval()
    install_recoverkv_mla(model, flash=False)
    ids = torch.randint(0, 96, (2, 45), generator=torch.Generator().manual_seed(1))
    P, steps = ids.shape[1], 14

    def run(gate, fused):
        wc = WindowedCacheConfig(window_size=8, num_sink_tokens=5, local_window_size=8,
                                 cache_budget=0.6, quant_ratio=0.7,
                                 quant_gate_ratio=gate, first_eviction_step=0)
        cache = make_cache(model, wc, prefill_len=P, max_tokens=steps + 1,
                           fused_decode=fused)
        fd.reset_stats()
        A.reset_path_stats()
        outs = []
        with torch.no_grad():
            o = model(input_ids=ids, past_key_values=cache, use_cache=True,
                      attention_mask=torch.ones_like(ids),
                      position_ids=torch.arange(P).unsqueeze(0).expand(2, -1))
            tok = o.logits[:, -1].argmax(-1, keepdim=True)
            for t in range(steps):
                o = model(input_ids=tok, past_key_values=cache, use_cache=True,
                          attention_mask=torch.ones(2, P + t + 1, dtype=torch.long),
                          position_ids=torch.full((2, 1), P + t))
                outs.append(o.logits[:, -1])
                tok = outs[-1].argmax(-1, keepdim=True)
        return torch.stack(outs, 1), cache, fd.stats(), A.path_stats()

    ref, _, s_ref, p_ref = run(1.0, False)
    assert p_ref["decode_fused"] == 0 and s_ref["fired"] == 0
    got, cache, s1, p1 = run(1.0, True)
    assert p1["decode_fused"] > 0, f"the fused path never ran: {p1}"
    assert s1["armed"] == s1["fired"] == s1["gated"] == p1["decode_fused"], s1
    assert s1["read_fraction"] == 1.0, s1
    err = (got - ref).abs().max().item()
    assert err < 2e-3, f"gate 1.0 fused vs materialize: max |dlogit| {err:.3e}"

    got25, cache25, s25, p25 = run(0.25, True)
    assert s25["gated"] == s25["fired"] == p25["decode_fused"] > 0, s25
    n = cache25._joint_store.num_active_windows
    assert s25["read_fraction"] <= math.ceil(0.25 * n) / n + 1e-9, s25
    assert torch.isfinite(got25).all()
    rep = fd.gate_report(fd.expect_gated("flash_attn", 0.7, True))
    assert rep["verdict"] == fd.GATE_OK, rep
    return (f"gate1.0 fused==materialize |dlogit| {err:.1e}; gate0.25 fired "
            f"{s25['fired']} gated {s25['gated']} read {s25['read_fraction']:.3f}; "
            f"verdict {rep['verdict']}")


def _at_real_dims() -> str:
    """``_check`` at DeepSeek-V2-Lite's own packed geometry: D = 576, halves of
    [256 latent | 32 RoPE] lanes, rep = 16 -- the shapes production launches."""
    global HA, HB, HALF, D, N_SEL
    saved = (HA, HB, HALF, D, N_SEL)
    HA, HB = 256, 32
    HALF, D, N_SEL = HA + HB, 2 * (HA + HB), 3
    try:
        return "; ".join(f"gated={g}: {_check(0, g)}" for g in (True, False))
    finally:
        HA, HB, HALF, D, N_SEL = saved


def main() -> int:
    for seed in range(2):
        for gated in (True, False):
            try:
                msg = _check(seed, gated)
            except AssertionError as exc:
                print(f"FAIL seed {seed} gated={gated}: {exc}")
                return 1
            print(f"ok seed {seed} gated={gated}: {msg}")
    try:
        print("ok real dims (D=576, 256+32 per half, rep=16):", _at_real_dims())
    except AssertionError as exc:
        print(f"FAIL real dims: {exc}")
        return 1
    try:
        print("ok e2e:", _e2e())
    except AssertionError as exc:
        print(f"FAIL e2e: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
