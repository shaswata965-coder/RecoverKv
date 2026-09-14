"""One gated decode step, end to end — the reference the kernel path must match.

This is the whole method in one function: score every int2 window from its card,
dequantize only the ``ratio`` that clear the bar, attend over ``[sink | fp body |
selected Q]``, and emit a window score for **every** window -- exact where it was
read, the card's rescaled estimate where it was not.

It exists for two reasons. It is the CPU oracle for the fused path (which cannot
run on a CPU-only box), and it is the readable statement of the algorithm: the
Triton path is this, tiled.

Why the gate lives here and not in ``update()``
-----------------------------------------------
Selecting windows needs the **query**, and ``Cache.update()`` never sees one --
transformers hands it keys and values only. That is not a design choice we can
revisit; it is the shape of the API. So the gate runs where the query is: inside
the attention step, which is exactly where ``flash_decode`` already patches in
for the fused path. ``update()`` hands over the fp tier and a handle to the
store; the selection happens here, one layer later, with ``q`` in hand.

The score every window gets
---------------------------
A skipped window that scored zero would rank last and be evicted -- the gate
would destroy the tier it exists to read less often, and at ``quant_ratio = 0.7``
that is 70% of the evictable cache. So :func:`fill_skipped_window_scores`
credits it with its card's estimate, rescaled by the ratio between the real and
estimated masses of the windows that *were* read. That factor costs one reduction
and needs nothing the step did not already compute.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from .scorer import expand_keep_to_query_heads, fill_skipped_window_scores

__all__ = ["gated_decode_step", "GatedStepOut"]


class GatedStepOut(tuple):
    """``(out, window_scores, n_read, n_active)``."""

    @property
    def out(self) -> Tensor:
        return self[0]

    @property
    def window_scores(self) -> Tensor:
        return self[1]

    @property
    def read_fraction(self) -> float:
        return self[2] / max(self[3], 1)


def gated_decode_step(
    q: Tensor,
    k_fp: Tensor,
    v_fp: Tensor,
    store,
    rope_module,
    scaling: float,
    num_sink: int,
    ratio: float,
    margin: float = float("inf"),
) -> GatedStepOut:
    """Attention output and per-window scores for one gated decode step.

    Parameters
    ----------
    q : ``[B, H_q, D]`` post-RoPE decode query.
    k_fp, v_fp : ``[B, H_kv, S_fp, D]`` fp tier, ``[sink | body]``.
    store : :class:`~modules.quant.store.QuantizedStore` holding the int2 tier.
    ratio : fraction of the active tier to dequantize. 1.0 reads everything and
        makes this identical to the ungated path -- the no-op proof.

    Returns
    -------
    :class:`GatedStepOut` — ``out`` is ``[B, H_q, D]``; ``window_scores`` is
    ``[B, H_q, n_body_win + n_active]`` in PHYSICAL order (body windows, then Q
    windows in ascending active-column order), which is the axis
    ``compute_score_meta``'s permutation expects.
    """
    from .decode_kernel import two_tier_decode_reference

    B, hq, D = q.shape
    hkv, S_fp = k_fp.shape[1], k_fp.shape[2]
    ws = store.window_size
    n_active = store.num_active_windows
    n_body_win = -(-max(S_fp - num_sink, 0) // ws)

    if n_active == 0:                                   # nothing to gate
        out, key_scores = two_tier_decode_reference(q, k_fp, v_fp, scaling)
        body = key_scores[..., num_sink:]
        pad = (-body.shape[-1]) % ws
        if pad:
            body = torch.nn.functional.pad(body, (0, pad))
        return GatedStepOut((out, body.reshape(B, hq, -1, ws).sum(-1), 0, 0))

    # --- 1. score every window from its card, keep the top `ratio` ----------
    keep, logmass, slots = store.gate_and_select(q, scaling, margin, ratio=ratio)
    n_sel = int(keep[0, 0].sum())

    # --- 2. dequantize ONLY those ------------------------------------------
    k_q, v_q, _ = store.gated_q_tier(keep, slots, rope_module, k_fp.dtype)

    # --- 3. attend over [sink | body | selected Q] -------------------------
    out, key_scores = two_tier_decode_reference(
        q, torch.cat([k_fp, k_q], dim=2), torch.cat([v_fp, v_q], dim=2), scaling)

    # --- 4. body windows: the usual contiguous reduce -----------------------
    body = key_scores[..., num_sink:S_fp]
    pad = (-body.shape[-1]) % ws
    if pad:
        body = torch.nn.functional.pad(body, (0, pad))
    body_win = body.reshape(B, hq, -1, ws).sum(-1)               # [B,Hq,n_body]

    # --- 5. Q windows: exact where read, estimated where not ---------------
    # `pick` is the same stable argsort gated_q_tier used, so physical column j
    # of the Q block is active column pick[..., j] -- the two MUST agree or the
    # scores land on the wrong windows.
    pick = torch.argsort(~keep, dim=-1, stable=True)[..., :n_sel]   # [B,Hkv,n_sel]
    q_read = key_scores[..., S_fp:].reshape(B, hq, n_sel, ws).sum(-1)
    pick_q = pick.repeat_interleave(hq // hkv, dim=1)                # [B,Hq,n_sel]

    exact = torch.zeros(B, hq, n_active, dtype=q_read.dtype, device=q.device)
    exact.scatter_(-1, pick_q, q_read)
    keep_q = expand_keep_to_query_heads(keep, hq)
    q_win = fill_skipped_window_scores(exact, keep_q, logmass)

    return GatedStepOut(
        (out, torch.cat([body_win, q_win], dim=-1), n_sel, n_active))
