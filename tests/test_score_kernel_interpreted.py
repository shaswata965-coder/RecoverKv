"""The prefill score kernel, EXECUTED on CPU, against its oracle.

``_score_kernel`` decides which windows survive the first eviction -- nearly
everything a LongBench answer is read from -- and had never been executed off a
GPU. Triton's interpreter runs its real ``@triton.jit`` body on CPU tensors, so
it is checked here against ``token_scores_from_lse`` at Qwen2.5's GQA shape
(rep 7, head_dim 128) and at MHA, with a massive-channel key common mode.

fp32 only, deliberately: the interpreter has no bfloat16 (numpy has none), and
a five-line bf16 dot-then-exp2 kernel returns ``inf`` under it, so a bf16 case
here would test the interpreter. On a GPU, ``tl.dot`` of bf16 inputs is exact
products with fp32 accumulation -- the arithmetic the fp32 case already covers.

Subprocess, because the interpreter is chosen when ``@triton.jit`` decorates.
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
    import os, sys
    assert os.environ.get("TRITON_INTERPRET") == "1"
    import torch, triton
    from modules.windowed_cache import score_kernel as sk

    bad = []
    for (B, Hq, Hkv, T, S, D) in ((1, 14, 2, 130, 130, 128),   # Qwen2.5 GQA, rep 7
                                  (2, 8, 2, 70, 150, 64),      # prefix keys (S > T)
                                  (1, 4, 4, 50, 50, 64)):      # MHA
        g = torch.Generator().manual_seed(Hq + T)
        q = torch.randn(B, Hq, T, D, generator=g)
        k = torch.randn(B, Hkv, S, D, generator=g)
        k[..., D // 2 - 4:D // 2] += 40.0                 # massive-channel common mode
        q[..., D // 2 - 4:D // 2] += torch.randn(B, Hq, 1, 4, generator=g) * 2.0
        scaling = D ** -0.5
        lse = sk.compute_lse(q, k, scaling)
        ref = sk.token_scores_from_lse(q, k, scaling, lse)
        out = torch.empty(B, Hq, S, dtype=torch.float32)
        sk._score_kernel[(triton.cdiv(S, 64), B * Hq)](
            q, k, lse, out, scaling * sk.LOG2E,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            T, S, S - T, Hq, Hq // Hkv,
            HEAD_DIM=D, BLOCK_M=64, BLOCK_N=64, IS_CAUSAL=(T > 1), USE_EXP2=True)
        rel = float((out - ref).abs().max() / ref.abs().max())
        # every query row's softmax sums to 1, so each head's column sums total T
        tot = out.sum(-1)
        print(f"Hq={Hq} Hkv={Hkv} T={T} S={S}: rel {rel:.2e} totals "
              f"[{float(tot.min()):.3f}, {float(tot.max()):.3f}]")
        if not (rel < 1e-4 and torch.allclose(tot, torch.full_like(tot, T), rtol=1e-4)):
            bad.append((Hq, Hkv, T, S, rel))
    print("BAD", bad)
    sys.exit(1 if bad else 0)
""")


def test_the_prefill_score_kernel_matches_its_oracle_when_executed():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
