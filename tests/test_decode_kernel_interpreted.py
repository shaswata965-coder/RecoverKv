"""The fused two-tier decode kernel, EXECUTED on CPU, against its oracle.

Every Triton kernel here "ships unvalidated by construction" because the dev box
has no GPU -- true of launching, and it had also been taken to be true of
running. It is not: Triton's interpreter (``TRITON_INTERPRET=1``) executes the
real ``@triton.jit`` body on CPU tensors, lane masks and all. So the kernel the
Qwen2.5 run launches -- ``KEY_ANCHOR`` (Qwen2's biased keys), gated through a
head-units selection, window-score epilogue, centroid fill -- is checked here
number for number against ``two_tier_window_reference`` over the
materialize-path read of the same store.

fp32 so the comparison is about the algorithm: the fp16 run of the same case
differs by ~2% of the output, which is rounding at the fixture's ~400-nat
logits (both paths round the huge bias-channel keys differently), and it is
the same size anchored and un-anchored.

Runs in a subprocess because the interpreter is chosen when ``@triton.jit``
decorates, i.e. at import. ``_decode_triton``'s CUDA guard is lifted in that
process only; nothing else about the dispatcher changes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_SCRIPT = textwrap.dedent("""
    import inspect, os, sys
    assert os.environ.get("TRITON_INTERPRET") == "1"
    import torch
    import transformers
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    from modules.windowed_cache import decode_kernel as dk
    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig
    from modules.windowed_cache.gate_kernel import gate_reference
    from modules.quant.effective import rotate_key_window
    from modules.quant.sketch import Sketch, value_centroid

    src = inspect.getsource(dk._decode_triton)
    assert src.count("if not q.is_cuda:") == 1, "the CUDA guard moved; update this test"
    ns = dict(dk.__dict__)
    exec(src.replace("if not q.is_cuda:", "if False:"), ns)
    decode = ns["_decode_triton"]

    KV, HQ, HK, D, WS, SINK, N_WIN = torch.float32, 8, 2, 128, 8, 4, 32
    bad = []
    for key_anchor in (True, False):
        mcfg = transformers.Qwen2Config(
            hidden_size=1024, num_attention_heads=HQ, num_key_value_heads=HK,
            num_hidden_layers=1, intermediate_size=64, vocab_size=64,
            max_position_embeddings=4096, rope_theta=1.0e6)
        rope = Qwen2RotaryEmbedding(config=mcfg)
        cache = WindowedCache(
            config=WindowedCacheConfig(
                window_size=WS, num_sink_tokens=SINK, local_window_size=16,
                cache_budget=0.5, quant_ratio=0.5, quant_key_anchor=key_anchor),
            prefill_len=260, model_config=mcfg, kv_dtype=KV, rope_module=rope,
            num_layers=1, max_tokens=8)
        T = SINK + N_WIN * WS
        g = torch.Generator().manual_seed(0)
        k_pre = torch.randn(HK, T, D, generator=g) * 1.5
        bias_cols = torch.tensor([60, 61, 62, 63, 124, 125, 126, 127])
        k_pre[:, :, bias_cols] = torch.tensor(
            [180.0, -130.0, 95.0, -240.0, 210.0, -75.0, 150.0, -60.0])
        pos = torch.arange(T)
        st = cache._states[0]
        st.replace(rotate_key_window(k_pre, pos, rope).unsqueeze(0).clone(),
                   torch.randn(1, HK, T, D, generator=g), pos.unsqueeze(0).clone())
        st.window_scores = torch.linspace(100.0, 1.0, N_WIN).expand(1, HQ, N_WIN).clone()
        st.original_window_ids = torch.arange(N_WIN).unsqueeze(0)
        cache._policies[0].initialize_after_prefill(T)
        cache._prefill_done[0] = True
        cache._evict_two_tier(0, step=0)
        store = cache._stores[0]
        store.commit_active_count(store.num_active_windows)
        n = store.num_active_windows

        qtier = cache._fused_qtier(0, store, 1, n, WS)
        assert (qtier["k_anchor"] is not None) == key_anchor
        gate = cache._fused_ctx[0]["gate"]
        k_fp, v_fp = st.key_states.contiguous(), st.value_states.contiguous()
        s_fp = k_fp.shape[2]
        nbw = -(-(s_fp - SINK) // WS)
        kq, vq, _ = store.effective_q_tier(rope, KV)
        k_eff, v_eff = torch.cat([k_fp, kq], 2), torch.cat([v_fp, vq], 2)
        q = torch.randn(1, HQ, D, generator=torch.Generator().manual_seed(1)) * 0.5
        q[..., bias_cols] += 3.0
        scaling = D ** -0.5

        # `wsum` is the kernel's reused scratch buffer (`_scratch("wsum")`) --
        # the next launch overwrites it -- so it is copied out at once, exactly
        # as `_run_fused` consumes it before the next layer runs.
        out, wsum = decode(q, k_fp, v_fp, qtier, scaling, SINK, nbw)
        wsum = wsum.clone()
        ref, ref_w = dk.two_tier_window_reference(
            q, k_eff, v_eff, scaling, SINK, WS, nbw, Sfp=s_fp)
        cases = [("ungated", out, wsum, ref, ref_w)]

        sel, lm = gate_reference(q, *gate["card"], gate["anchor"], scaling,
                                 gate["n_sel"], head_norm=gate["head_norm"])
        out, wsum = decode(q, k_fp, v_fp, qtier, scaling, SINK, nbw,
                           sel=sel.contiguous(), logmass=lm.contiguous(),
                           centroids=gate["centroid"])
        wsum = wsum.clone()
        cent = value_centroid(Sketch(*gate["card"]), store._v_anchor)[0]
        cent = cent.permute(1, 0, 2).repeat_interleave(HQ // HK, dim=0).unsqueeze(0)
        ref, ref_w = dk.two_tier_window_reference(
            q, k_eff, v_eff, scaling, SINK, WS, nbw, Sfp=s_fp,
            sel=sel, logmass=lm, centroids=cent)
        cases.append(("gated", out, wsum, ref, ref_w))

        for name, o, w, r, rw in cases:
            rel = float((o - r).norm() / r.norm())
            dw = float((w[..., :rw.shape[-1]] - rw).abs().max())
            print(f"key_anchor={key_anchor} {name}: out rel {rel:.2e} wsum {dw:.2e}")
            if not (rel < 1e-4 and dw < 1e-5):
                bad.append((key_anchor, name, rel, dw))
    print("BAD", bad)
    sys.exit(1 if bad else 0)
""")


def test_the_fused_decode_kernel_matches_its_oracle_when_executed():
    """Anchored (Qwen) and un-anchored (Llama/Mistral), whole-tier read and gated
    read: output to 1e-4 relative, every window score to 1e-5 absolute."""
    pytest.importorskip("triton")
    pytest.importorskip("transformers")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-4000:]
