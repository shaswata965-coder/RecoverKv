"""Triton gate: score every int2 window from its card, pick the top ``n_sel``.

What it replaces
----------------
Without the gate the decode step dequantizes every active int2 window to find
out which ones mattered (design §8). This kernel reads each window's card --
``mu``/``v`` int8 plus two tiny scalars, 280 B per head against a 8448 B window
-- and emits the compacted slot list the Q-tier loop should actually visit, plus
the log-mass estimate that every **skipped** window contributes to
``window_scores``.

Shape of the work
-----------------
One program owns one ``(batch, KV head)``, matching ``decode_kernel``'s grid, so
the GQA group's union is free: the program reduces over its own ``rep`` query
heads before selecting, which is what makes the cap bind on the KV head -- the
real unit of work, since one program loads a window once for the whole group.
Capping per query head and unioning afterwards would let a 0.25 ratio read up to
``0.25 * rep`` of the tier (100% at Llama-3.1-8B's rep=4).

Two dot products per window give both outputs::

    m = q.anchor + q.mu        g = q.v
    x_i   = scale * (m + t_i * g)                  per-token estimate
    bound = max_i (x_i + scale * ||q|| * eps_i)    upper bound, cannot miss
    est   = max_i x_i                              slack-free, what the cap ranks on
    logmass = logsumexp_i x_i                      what a skipped window scores

``bound`` and ``est`` are different quantities with different jobs; see
:func:`modules.quant.sketch.select_windows` for why the cap ranks on the second.

A separate kernel, for now
--------------------------
This could be a prologue inside ``_two_tier_decode_kernel`` -- one program
already owns the right ``(batch, KV head)`` -- and that is where it belongs
eventually, because the decode path is launch-bound and this costs one extra
launch per layer per step. It is separate first because a prologue competes for
the main kernel's register budget: its Q-tier loop already stages ten
``[D/2, BLOCK_T]`` fp32 tiles and the autotuner falls back silently when that
budget is exceeded, so a rung dropped by the prologue would cost more than the
gate saves and would produce no error. Land it standalone, measure the launch
cost against the traffic saved, then fold it in with the rung pinned.

Backend contract (mirrors ``score_kernel`` / ``decode_kernel``)
--------------------------------------------------------------
* :func:`gate_reference` -- pure-PyTorch oracle. Defines correctness, runs on
  CPU, and is the Triton kernel's test oracle.
* ``_gate_kernel`` + :func:`_gate_triton` -- the GPU kernel. **GPU-only, ships
  unvalidated by construction** (this repo's dev box is CPU-only), like every
  other Triton kernel here.
* :func:`fused_gate` -- dispatcher: Triton on CUDA, reference otherwise.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

try:  # pragma: no cover - import guard, exercised only where triton is present
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # ImportError, or a broken triton build
    _HAS_TRITON = False


def gate_reference(
    q: Tensor,
    mu_q: Tensor, mu_s: Tensor,
    v_q: Tensor, v_s: Tensor,
    t_q: Tensor, t_s: Tensor,
    e_q: Tensor, e_s: Tensor,
    anchor: Tensor,
    scaling: float,
    n_sel: int,
) -> Tuple[Tensor, Tensor]:
    """``(sel, logmass)`` — the oracle.

    Parameters
    ----------
    q : ``[B, H_q, D]`` post-RoPE decode query.
    mu_q .. e_s : card fields, ``[B, Nw, H_kv, ...]``.
    anchor : ``[B, H_kv, D]`` or ``[H_kv, D]``.
    n_sel : windows to keep per ``(row, KV head)``.

    Returns
    -------
    sel : ``[B, H_kv, n_sel]`` int32 column indices, ascending.
    logmass : ``[B, H_q, Nw]`` the estimate for every window.
    """
    from modules.quant.sketch import Sketch, gate_and_score, group_max

    card = Sketch(mu_q, mu_s, v_q, v_s, t_q, t_s, e_q, e_s)
    _, logmass, est = gate_and_score(q, card, anchor, scaling)
    hkv = mu_q.shape[2]
    top = group_max(est, hkv).topk(n_sel, dim=-1).indices          # [B,Hkv,n_sel]
    return top.sort(dim=-1).values.to(torch.int32), logmass


if _HAS_TRITON:  # pragma: no cover - GPU-only

    @triton.jit
    def _gate_kernel(
        Q, MU, MUS, V, VS, T, TS, E, ES, ANCH,
        BOUND, EST, LOGM,
        qb, qh, mub, mun, muh, sb, sn, sh, tb, tn, th, ab, ah, ob, oh,
        NW, REP, SCALE,
        HEAD_DIM: tl.constexpr, WS: tl.constexpr, BLOCK_WS: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        """One program per ``(batch, KV head)``; streams windows in BLOCK_W tiles.

        The card vectors are streamed through registers and never staged: only
        three scalars per window (bound, est, logmass) leave the loop, so the
        program's shared footprint is O(BLOCK_W), not O(NW * HEAD_DIM).

        ``BLOCK_WS`` is ``WS`` rounded up to a power of two, because
        ``tl.arange`` admits nothing else. The surplus lanes are masked out of
        the ``t``/``eps`` loads **and** forced to ``-inf`` in ``x``: a padding
        lane that merely loaded zero would still carry the real ``SCALE * m``
        into the max and the logsumexp, so a ``ws`` of 12 would score every
        window as if it held a 13th token sitting exactly at its mean.
        """
        b = tl.program_id(0)
        kv = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        dm = d < HEAD_DIM
        w = tl.arange(0, BLOCK_WS)
        wm = w < WS

        # This KV head's query group, and the anchor term that is constant in w.
        # Accumulated per query head into registers; REP is small (4 on Llama-3.1-8B).
        for r in range(0, REP):
            hq = kv * REP + r
            q_v = tl.load(Q + b * qb + hq * qh + d, mask=dm, other=0.0).to(tl.float32)
            qn = tl.sqrt(tl.sum(q_v * q_v, axis=0))
            anc = tl.load(ANCH + b * ab + kv * ah + d, mask=dm, other=0.0).to(tl.float32)
            base = tl.sum(q_v * anc, axis=0)

            for w0 in range(0, NW, BLOCK_W):
                cols = w0 + tl.arange(0, BLOCK_W)
                cm = cols < NW

                # mu, v: int8 [BLOCK_W, D] with a per-(window, head) fp16 scale.
                mu = tl.load(MU + b * mub + cols[:, None] * mun + kv * muh + d[None, :],
                             mask=cm[:, None] & dm[None, :], other=0).to(tl.float32)
                mus = tl.load(MUS + b * sb + cols * sn + kv * sh,
                              mask=cm, other=0.0).to(tl.float32)
                vv = tl.load(V + b * mub + cols[:, None] * mun + kv * muh + d[None, :],
                             mask=cm[:, None] & dm[None, :], other=0).to(tl.float32)
                vs = tl.load(VS + b * sb + cols * sn + kv * sh,
                             mask=cm, other=0.0).to(tl.float32)

                m = base + mus * tl.sum(mu * q_v[None, :], axis=1)      # [BLOCK_W]
                g = vs * tl.sum(vv * q_v[None, :], axis=1)              # [BLOCK_W]

                tt = tl.load(T + b * tb + cols[:, None] * tn + kv * th + w[None, :],
                             mask=cm[:, None] & wm[None, :], other=0).to(tl.float32)
                ts = tl.load(TS + b * sb + cols * sn + kv * sh,
                             mask=cm, other=0.0).to(tl.float32)
                ee = tl.load(E + b * tb + cols[:, None] * tn + kv * th + w[None, :],
                             mask=cm[:, None] & wm[None, :], other=0).to(tl.float32)
                es = tl.load(ES + b * sb + cols * sn + kv * sh,
                             mask=cm, other=0.0).to(tl.float32)

                x = SCALE * (m[:, None] + (ts[:, None] * tt) * g[:, None])
                # A padding lane must not exist for the reductions below. Zeroing
                # its `t` is not enough: x would still be SCALE * m, a plausible
                # logit for a token that is not there.
                x = tl.where(wm[None, :], x, -float("inf"))
                bd = tl.max(x + SCALE * qn * (es[:, None] * ee), axis=1)
                es_max = tl.max(x, axis=1)
                # logsumexp, written out: the online-softmax form the decode
                # kernel uses, and `torch.logsumexp(` must stay unique in modules/
                # for audit_e2e.py's attribution (tests/test_audit_e2e.py).
                lm = es_max + tl.log(tl.sum(tl.exp(x - es_max[:, None]), axis=1))

                o = b * ob + hq * oh + cols
                tl.store(BOUND + o, tl.where(cm, bd, -float("inf")), mask=cm)
                tl.store(EST + o, tl.where(cm, es_max, -float("inf")), mask=cm)
                tl.store(LOGM + o, tl.where(cm, lm, -float("inf")), mask=cm)


def _gate_triton(q, card, anchor, scaling, n_sel):  # pragma: no cover - GPU-only
    if not _HAS_TRITON:
        raise RuntimeError("fused_gate requires triton")
    mu_q, mu_s, v_q, v_s, t_q, t_s, e_q, e_s = card
    B, NW, HKV, D = mu_q.shape
    HQ = q.shape[1]
    ws = t_q.shape[-1]
    if anchor.dim() == 2:
        anchor = anchor.unsqueeze(0).expand(B, HKV, D).contiguous()
    out = [torch.empty((B, HQ, NW), dtype=torch.float32, device=q.device)
           for _ in range(3)]
    bound, est, logm = out
    _gate_kernel[(B, HKV)](
        q, mu_q, mu_s, v_q, v_s, t_q, t_s, e_q, e_s, anchor,
        bound, est, logm,
        q.stride(0), q.stride(1),
        mu_q.stride(0), mu_q.stride(1), mu_q.stride(2),
        mu_s.stride(0), mu_s.stride(1), mu_s.stride(2),
        t_q.stride(0), t_q.stride(1), t_q.stride(2),
        anchor.stride(0), anchor.stride(1),
        bound.stride(0), bound.stride(1),
        NW, HQ // HKV, scaling,
        HEAD_DIM=D, WS=ws, BLOCK_WS=triton.next_power_of_2(ws),
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_W=min(64, triton.next_power_of_2(NW)),
        num_warps=4,
    )
    from modules.quant.sketch import group_max
    top = group_max(est, HKV).topk(n_sel, dim=-1).indices
    return top.sort(dim=-1).values.to(torch.int32), logm


def fused_gate(
    q: Tensor,
    card,
    anchor: Tensor,
    scaling: float,
    n_sel: int,
    force_reference: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Dispatch: Triton on CUDA, the reference elsewhere.

    Unlike ``decode_kernel``'s Triton-or-raise contract, this one falls back,
    because the reference is not a *degraded* path -- it is the same arithmetic
    at the same selectivity, just without the fused launch. Nothing silently
    changes what gets read.
    """
    if force_reference or not (_HAS_TRITON and q.is_cuda):
        return gate_reference(*( (q,) + tuple(card) + (anchor, scaling, n_sel) ))
    return _gate_triton(q, card, anchor, scaling, n_sel)
