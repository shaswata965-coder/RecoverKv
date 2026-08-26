"""GPU parity for the FlashInfer L-source (:mod:`flashinfer_lse`).

Skipped unless BOTH CUDA and flashinfer are present — this is the validation the
CPU dev box cannot do. It pins the two things most likely to be wrong in a blind
integration:

  * **LSE base/scale** — FlashInfer's ``return_lse`` must equal
    :func:`score_kernel.compute_lse` (natural-log logsumexp of the *scaled*
    logits). If it is off by ``ln 2``, the installed build returns log2 and the
    run needs ``STICKYKV_FLASHINFER_LSE_LOG2=1`` (this test says so explicitly).
  * **Attention-output parity** — the output handed back to the model must match
    ``scaled_dot_product_attention`` to fp rounding, else the model degrades.

Run on the GPU box:  ``pytest tests/test_flashinfer_lse.py -q``
"""

from __future__ import annotations

import math
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlashInfer parity needs CUDA"
)


def _sdpa_reference(q, k, v, scale, causal):
    """Reference attention over the flash layout [B, S, H, D] (GQA-aware)."""
    import torch.nn.functional as F

    B, Sq, Hq, D = q.shape
    Hkv = k.shape[2]
    qh = q.permute(0, 2, 1, 3)                       # [B, Hq, Sq, D]
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    if Hkv != Hq:                                    # expand GQA groups
        rep = Hq // Hkv
        kh = kh.repeat_interleave(rep, dim=1)
        vh = vh.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(
        qh, kh, vh, is_causal=causal, scale=scale
    )
    return out.permute(0, 2, 1, 3).contiguous()      # back to [B, Sq, Hq, D]


def test_flashinfer_lse_matches_compute_lse_and_output():
    flashinfer = pytest.importorskip("flashinfer")
    from modules.windowed_cache import flashinfer_lse
    from modules.windowed_cache.score_kernel import compute_lse

    torch.manual_seed(0)
    B, S, Hq, Hkv, D = 2, 128, 8, 8, 128
    dev = torch.device("cuda")
    dt = torch.float16
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, S, Hq, D, device=dev, dtype=dt)
    k = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)
    v = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)

    # Install the patch on a stub module whose flash_attn_func is our SDPA ref, so
    # a FALLBACK (broken latch) would return the reference, not crash.
    stub = types.SimpleNamespace(
        flash_attn_func=lambda *a, **kw: _sdpa_reference(
            a[0], a[1], a[2], kw.get("softmax_scale", scale), kw.get("causal", True)
        )
    )
    handle = flashinfer_lse.enable(stub)
    assert handle is not None, "flashinfer_lse.enable returned None despite flashinfer present"
    try:
        out = stub.flash_attn_func(q, k, v, 0.0, softmax_scale=scale, causal=True)
        lse = flashinfer_lse.pop()
    finally:
        handle.restore()

    assert lse is not None, (
        "FlashInfer path did not capture LSE (it fell back). Check the plan/run "
        "API version and the [B,S,H,D] layout."
    )
    # Output parity vs SDPA.
    ref = _sdpa_reference(q, k, v, scale, causal=True)
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2), \
        "FlashInfer attention output diverged from SDPA reference"

    # LSE parity vs compute_lse ([B, Hq, T]).
    q_hd = q.permute(0, 2, 1, 3).contiguous()        # [B, Hq, S, D]
    k_hd = k.permute(0, 2, 1, 3).contiguous()        # [B, Hkv, S, D]
    want = compute_lse(q_hd, k_hd, scale)            # [B, Hq, S] fp32
    if not torch.allclose(lse.float(), want.float(), atol=1e-2, rtol=1e-2):
        ratio = (want.float() / lse.float().clamp_min(1e-6)).median().item()
        pytest.fail(
            f"FlashInfer LSE != compute_lse. median(want/got)={ratio:.4f}. "
            f"If ~{1/math.log(2):.4f} (1/ln2), the build returns log2 LSE — set "
            "STICKYKV_FLASHINFER_LSE_LOG2=1."
        )
