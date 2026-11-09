"""Evaluation epilogue for the completion-term ablation (``quant_completion``).

**Evaluation only.** The shipped decode path never enters this module: with
``quant_completion = null`` :func:`flash_decode._run_fused` does exactly what it
did before the knob existed. Every value of the knob, including
``"calibrated"``, comes through here. See :mod:`utils.completion_spec` for what
each variant means.

Why an epilogue around the kernel, not a kernel change
------------------------------------------------------
The fused kernel's GATED epilogue (``decode_kernel._two_tier_decode_kernel``,
§5) returns

    out  = (o_bar + sum_u fill(u) * nu_u) / den,        den = 1 + sum_u fill(u)
    wsum = [body mass | read mass | fill] / den

over the skipped windows ``u``, where ``o_bar`` is exact attention over the read
set A and ``fill(u) = exp(logmass(u) + delta)``. Every quantity a variant needs
can be recovered from that exactly:

* the skipped columns of ``wsum`` sum to ``S_U = sum fill / den``, so
  ``den = 1 / (1 - S_U)`` and ``fill(u) = wsum_u * den``;
* ``o_bar = out * den - sum_u fill(u) * nu_u``;
* the read windows' true masses are ``m_bar(w) = wsum_w * den`` (relative to
  ``Z_A``), and with ``Z_fp`` the sink + fp body's softmax normaliser
  (``logsumexp(scaling * q . k_fp)``, one matmul), ``log Z_A =
  log Z_fp - log(1 - sum_R m_bar)``.

So the kernel stays byte-for-byte the shipped kernel. The only change on this
path is that ``out`` is written to an fp32 buffer (its store casts to the
buffer's dtype), so the inversion loses nothing to fp16 rounding. Everything
here is plain torch: it is checked on CPU against
``two_tier_window_reference(..., completion=...)``, which computes each variant
DIRECTLY from the reference's own ``o_bar`` / ``lse``, not by this inversion
(``tests/test_completion_ablation.py``).

The cost is a few dozen small torch ops per fused layer per step, on ablation
arms only. Their decode time is not a number this repo quotes.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from modules.quant.sketch import _align_anchor, _dq_symb
from utils.completion_spec import CompletionSpec

_M31 = (1 << 31) - 1


def _mix(x: Tensor) -> Tensor:
    """A 31-bit integer hash (xorshift-multiply). int64 throughout, no overflow:
    every product is < 2^31 * 2^27."""
    x = x & _M31
    x = x ^ (x >> 16)
    x = (x * 0x45D9F3B) & _M31
    x = x ^ (x >> 16)
    x = (x * 0x45D9F3B) & _M31
    return x ^ (x >> 16)


def window_noise(seed: int, layer_idx: int, wids: Tensor, n_q_heads: int) -> Tensor:
    """``[B, n] window ids -> [B, H_q, n]`` standard normals, a pure function of
    ``(seed, layer, query head, window id)``.

    A window keeps its draw for as long as it stays in the int2 tier, on every
    step and in every column it occupies, because a card's error is a property
    of the window it summarises, not of the step. Integer hashing, then
    Box-Muller in fp64, so the draw does not depend on device, batch row or
    slot.
    """
    dev = wids.device
    s = _mix(torch.tensor(int(seed) & _M31, dtype=torch.int64, device=dev))
    s = _mix(s ^ (int(layer_idx) & _M31))
    h = torch.arange(n_q_heads, dtype=torch.int64, device=dev).view(1, -1, 1)
    k = _mix(_mix(s ^ h) ^ wids.to(torch.int64).unsqueeze(1))      # [B,H_q,n]
    u1 = (_mix(k ^ 0x2545F491).to(torch.float64) + 0.5) / float(1 << 31)
    u2 = (_mix(k ^ 0x1B873593).to(torch.float64) + 0.5) / float(1 << 31)
    z = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    return z.to(torch.float32)


def variant_mass(spec: CompletionSpec, fill: Tensor, skip: Tensor,
                 logmass: Tensor, log_za: Optional[Tensor],
                 eps: Optional[Tensor]) -> Tensor:
    """The mass each skipped window attends with under ``spec``. ``[B,H_q,n]``.

    ``fill`` is the shipped calibrated term (0 on read windows), ``log_za`` the
    read set's natural-log normaliser ``[B,H_q,1]`` (uncalibrated only), ``eps``
    the standard normals of :func:`window_noise` (``sigma > 0`` only).
    """
    zero = torch.zeros_like(fill)
    if spec.mode == "off":
        return zero
    if spec.mode == "uncalibrated":
        return torch.where(skip, (logmass.float() - log_za).exp(), zero)
    m = fill * spec.scale
    if spec.sigma > 0.0:
        mp = torch.where(skip, m * (spec.sigma * eps).exp(), zero)
        tot, totp = m.sum(-1, keepdim=True), mp.sum(-1, keepdim=True)
        # Total preserved exactly, so this arm is a pure redistribution and
        # stays orthogonal to `scale`.
        m = torch.where(totp > 0, mp * (tot / totp.clamp_min(1e-38)), m)
    return m


def apply_completion(
    spec: CompletionSpec,
    *,
    out: Tensor,
    wsum: Tensor,
    n_body_win: int,
    sel: Tensor,
    logmass: Tensor,
    centroids: Tuple[Tensor, Tensor, Tensor],
    vm_bits: int,
    q: Tensor,
    k_fp: Tensor,
    scaling: float,
    wids: Optional[Tensor] = None,
    layer_idx: int = 0,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Replace the gated step's completion term with ``spec``'s variant.

    ``out [B,H_q,D]`` and ``wsum [B,H_q,W_phys]`` are what the fused kernel
    returned (its calibrated completion already applied); ``sel [B,H_kv,n_sel]``,
    ``logmass [B,H_q,n]`` and ``centroids`` are what the gate handed it;
    ``q [B,H_q,D]`` and ``k_fp [B,H_kv,S_fp,D]`` are the step's query and fp
    tier. Returns ``(out', wsum')``: ``wsum`` is returned unchanged unless
    ``spec.scores == "coupled"``.
    """
    B, Hq, D = out.shape
    Hkv = sel.shape[1]
    rep = Hq // Hkv
    n = logmass.shape[-1]
    if wsum.shape[-1] != n_body_win + n:
        raise RuntimeError(
            f"wsum has {wsum.shape[-1]} columns, expected n_body_win + n_active = "
            f"{n_body_win} + {n}: the epilogue would read the wrong windows.")

    keep = torch.zeros(B, Hkv, n, dtype=torch.bool, device=out.device)
    keep.scatter_(-1, sel.long(), True)
    read = keep.repeat_interleave(rep, dim=1)                         # [B,H_q,n]
    skip = ~read

    w = wsum.float()
    wq = w[..., n_body_win:]
    s_u = torch.where(skip, wq, torch.zeros_like(wq)).sum(-1, keepdim=True)
    den = 1.0 / (1.0 - s_u).clamp_min(1e-30)                          # 1 + sum fill
    fill = torch.where(skip, wq * den, torch.zeros_like(wq))
    mbar = torch.where(read, wq * den, torch.zeros_like(wq))

    vm_q, vm_s, v_anc = centroids
    nu = _dq_symb(vm_q, vm_s, vm_bits)                                # [B,n,H_kv,D]
    nu = nu + _align_anchor(v_anc.to(torch.float32), nu)
    nu = nu.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)         # [B,H_q,n,D]

    o_bar = out.float() * den - torch.einsum("bhn,bhnd->bhd", fill, nu)

    log_za = eps = None
    if spec.mode == "uncalibrated":
        qg = q.float().reshape(B, Hkv, rep, D)
        z = torch.einsum("bgrd,bgsd->bgrs", qg, k_fp.float()).reshape(B, Hq, -1)
        lse_fp = torch.logsumexp(z * scaling, dim=-1, keepdim=True)
        m_r = mbar.sum(-1, keepdim=True)
        log_za = lse_fp - torch.log((1.0 - m_r).clamp_min(1e-30))
    if spec.mode == "calibrated" and spec.sigma > 0.0:
        if wids is None or tuple(wids.shape) != (B, n):
            raise RuntimeError(
                "quant_completion_sigma > 0 needs the active window ids [B, n_active] "
                f"to key each window's draw; got {None if wids is None else tuple(wids.shape)}.")
        eps = window_noise(spec.seed, layer_idx, wids, Hq)

    m = variant_mass(spec, fill, skip, logmass, log_za, eps)
    den_v = 1.0 + m.sum(-1, keepdim=True)
    out_v = (o_bar + torch.einsum("bhn,bhnd->bhd", m, nu)) / den_v

    if spec.scores == "coupled":
        w_v = torch.cat([w[..., :n_body_win] * den / den_v,
                         (mbar + m) / den_v], dim=-1)
    else:
        w_v = wsum
    return out_v.to(out_dtype or out.dtype), w_v


# ---------------------------------------------------------------------------
# Experiment 1b: every variant on ONE decode state (observation collector)
# ---------------------------------------------------------------------------

#: The outputs :func:`completion_rungs` computes, in order. ``cal`` is the shipped
#: term; ``mass_oracle`` uses each skipped window's exact int2 mass with the
#: stored centroid (the best any mass calibrator could do); ``value_oracle`` the
#: calibrated mass with the window's exact attention-weighted value;
#: ``full_oracle`` both, which is exactly attention over every held window.
COMPLETION_RUNGS = ("off", "uncal", "cal",
                    "scale0.5", "scale0.8", "scale1.25", "scale2", "scale4",
                    "noise0.25", "noise0.5", "noise1", "noise2",
                    "mass_oracle", "value_oracle", "full_oracle")
_RUNG_SCALE = {"scale0.5": 0.5, "scale0.8": 0.8, "scale1.25": 1.25,
               "scale2": 2.0, "scale4": 4.0}
_RUNG_SIGMA = {"noise0.25": 0.25, "noise0.5": 0.5, "noise1": 1.0, "noise2": 2.0}


def completion_rungs(q: Tensor, fpK: Tensor, fpV: Tensor, qK: Tensor, qV: Tensor,
                     ws: int, read_kv: Tensor, logmass: Tensor, nu: Tensor,
                     scale: float, wids: Tensor, layer_idx: int,
                     noise_seed: int = 0) -> dict:
    """Every completion variant, and the exact masses, at one decode state.

    One sample row, heads-major: ``q [H, D]``; the fp tier ``fpK, fpV
    [Hk, S_fp, D]`` (sinks + fp body, as the cache holds it); the int2 tier
    dequantized ``qK, qV [Hk, n*ws, D]`` in the gate's active order;
    ``read_kv [Hk, n]`` the gate's pick; ``logmass [H, n]`` the cards'
    estimates it ranked on; ``nu [Hk, n, D]`` the cards' value centroids;
    ``wids [n]`` the active window ids (the noise rungs key on them exactly as
    the ablation arms do, at ``noise_seed``).

    Returns ``outs [K, H, D]`` (one row per :data:`COMPLETION_RUNGS`), and per
    window ``log_mstar`` (the exact int2 mass over ``Z_A``, natural log) and
    ``logerr_cal`` (``log m~_cal - log m*``; on read windows it is the in-sample
    residual, whose mean is 0 by construction of ``delta``), per head ``delta``,
    ``log_za`` and ``uncal_shift = log m~_uncal - log m~_cal = -(delta +
    log Z_A)`` -- the common factor Experiment 3 removes -- and ``read [H, n]``.
    All fp32.
    """
    H, D = q.shape
    Hk = fpK.shape[0]
    rep = H // Hk
    n = read_kv.shape[-1]

    def gqa(x):
        return x.repeat_interleave(rep, dim=0) if rep > 1 else x

    qf = q.float()
    z_fp = torch.einsum("hd,hsd->hs", qf, gqa(fpK).float()) * scale
    zq = (torch.einsum("hd,hsd->hs", qf, gqa(qK).float()) * scale).view(H, n, ws)
    lse_w = zq.logsumexp(-1)                                          # [H, n]
    pw = (zq - lse_w.unsqueeze(-1)).exp()
    nu_star = torch.einsum("hnt,hntd->hnd", pw, gqa(qV).float().view(H, n, ws, D))
    read = gqa(read_kv)                                               # [H, n]
    skip = ~read
    ninf = torch.full_like(lse_w, float("-inf"))
    log_za = torch.logaddexp(z_fp.logsumexp(-1),
                             torch.where(read, lse_w, ninf).logsumexp(-1))   # [H]
    log_mstar = lse_w - log_za.unsqueeze(-1)
    mstar = log_mstar.exp()
    o_bar = (torch.einsum("hs,hsd->hd", (z_fp - log_za.unsqueeze(-1)).exp(),
                          gqa(fpV).float())
             + torch.einsum("hn,hnd->hd", torch.where(read, mstar, 0.0), nu_star))

    lm = logmass.float()
    ok = read & (mstar > 0)
    cnt = ok.sum(-1)
    delta = torch.where(cnt > 0,
                        torch.where(ok, log_mstar - lm, 0.0).sum(-1) / cnt.clamp_min(1),
                        torch.full_like(log_za, float("-inf")))
    m_cal = torch.where(skip, (lm + delta.unsqueeze(-1)).exp(), 0.0)
    m_unc = torch.where(skip, (lm - log_za.unsqueeze(-1)).exp(), 0.0)
    m_star = torch.where(skip, mstar, 0.0)
    nu_c = gqa(nu).float()

    def comp(m, v):
        return (o_bar + torch.einsum("hn,hnd->hd", m, v)) / (1.0 + m.sum(-1, keepdim=True))

    eps = window_noise(noise_seed, layer_idx, wids.view(1, -1), H)[0]
    outs = {"off": o_bar, "uncal": comp(m_unc, nu_c), "cal": comp(m_cal, nu_c),
            "mass_oracle": comp(m_star, nu_c), "value_oracle": comp(m_cal, nu_star),
            "full_oracle": comp(m_star, nu_star)}
    for k, c in _RUNG_SCALE.items():
        outs[k] = comp(m_cal * c, nu_c)
    for k, s in _RUNG_SIGMA.items():
        mp = torch.where(skip, m_cal * (s * eps).exp(), 0.0)
        mp = mp * (m_cal.sum(-1, keepdim=True) / mp.sum(-1, keepdim=True).clamp_min(1e-38))
        outs[k] = comp(mp, nu_c)
    return dict(outs=torch.stack([outs[r] for r in COMPLETION_RUNGS]),
                log_mstar=log_mstar, logerr_cal=(lm + delta.unsqueeze(-1)) - log_mstar,
                delta=delta, log_za=log_za, uncal_shift=-(delta + log_za), read=read)
