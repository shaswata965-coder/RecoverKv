"""What the read gate's approximation costs, measured on the real model.

The gate reads ``quant_gate_ratio`` of the int2 tier exactly and represents
every other window by its card: an estimated log-mass (its attention weight)
and a value centroid (what it contributes). On synthetic keys the card is
accurate; whether it is accurate on a given model is a property of that
model's queries and keys, which a CPU box does not have. This module measures
it where the model is.

:func:`install` wraps :func:`flash_decode._run_fused`. Every fused decode layer
still runs exactly as shipped -- the gated output is what the model continues
with, so the run's predictions are the shipped method's -- and each measured
layer additionally runs the full read (``sel=None``: every int2 window read
exactly, QEvict's read path) through the same kernel. :func:`gate_error_metrics`
compares the two, per query head:

``out_err``
    ``||out_gated - out_full|| / ||out_full||`` -- what the approximation does
    to the layer's attention output. The end effect.
``card_tv``
    total variation between the card's per-window mass distribution
    (``softmax(logmass)``) and the true one (the full read's window scores),
    over the int2 tier. 0 = the card weighs the skipped windows exactly as
    attention does; 1 = disjoint. Measures the ESTIMATE.
``recall``
    share of the head's true int2-tier mass inside the windows the gate read.
    Measures the SELECTION.
``q_share``
    the int2 tier's share of the head's whole attention -- how much any of the
    above can matter for this head.

Diagnostic only: one extra kernel launch per measured layer, no effect on what
is generated. ``scripts/diagnose_gate_error.py`` is the entry point.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch
from torch import Tensor

__all__ = ["gate_error_metrics", "install", "summarize"]


def gate_error_metrics(
    out_gated: Tensor,
    out_full: Tensor,
    wsum_full: Tensor,
    n_body_win: int,
    sel: Tensor,
    logmass: Tensor,
) -> Dict[str, Tensor]:
    """Per-(row, query head) error of one gated decode layer against its full read.

    Parameters
    ----------
    out_gated, out_full : ``[B, H_q, D]`` attention outputs of the gated step
        and of the full read.
    wsum_full : ``[B, H_q, W_phys]`` the full read's per-window softmax mass in
        physical order -- fp body windows, then the int2 tier's active columns.
    n_body_win : fp body windows at the head of ``wsum_full``'s window axis.
    sel : ``[B, H_kv, n_sel]`` the gate's pick (active-column indices).
    logmass : ``[B, H_q, n_active]`` the card's log-mass for every active window,
        columns in the same order as ``wsum_full[..., n_body_win:]``.

    Returns ``{"out_err", "card_tv", "recall", "q_share"}``, each ``[B, H_q]``.
    """
    og, of = out_gated.float(), out_full.float()
    out_err = (og - of).norm(dim=-1) / of.norm(dim=-1).clamp_min(1e-12)

    wq = wsum_full[..., n_body_win:].float()                     # [B, H_q, n]
    q_share = wq.sum(-1)
    p_true = wq / q_share.clamp_min(1e-30).unsqueeze(-1)
    p_card = torch.softmax(logmass.float(), dim=-1)
    card_tv = 0.5 * (p_true - p_card).abs().sum(-1)

    B, H_q, n = p_true.shape
    H_kv = sel.shape[1]
    keep = torch.zeros(B, H_kv, n, dtype=torch.bool, device=p_true.device)
    keep.scatter_(-1, sel.long(), True)
    recall = (p_true * keep.repeat_interleave(H_q // H_kv, dim=1)).sum(-1)
    return {"out_err": out_err, "card_tv": card_tv, "recall": recall,
            "q_share": q_share}


def install(max_steps: int = 16):
    """Wrap ``flash_decode._run_fused`` to measure the first ``max_steps`` decode
    steps of every generation. Returns ``(stats, uninstall)``: ``stats[layer]``
    maps each metric name to a list of per-head values."""
    from . import flash_decode
    from .decode_kernel import fused_two_tier_decode
    from .gate_kernel import fused_gate

    orig = flash_decode._run_fused
    stats = defaultdict(lambda: defaultdict(list))
    seen = {"cache": None, "step": 0}

    def wrapped(ctx, q_flash, k_flash, v_flash):
        out = orig(ctx, q_flash, k_flash, v_flash)       # the shipped step
        cache, layer = ctx.get("cache"), ctx.get("layer_idx")
        if cache is not seen["cache"]:                   # a new generation
            seen["cache"], seen["step"] = cache, 0
        if layer == 0:
            seen["step"] += 1
        if seen["step"] > max_steps:
            return out
        with torch.no_grad():
            # The same inputs `_run_fused` built, restated: flash layout ->
            # heads-major, GQA un-repeat, the body's window count.
            q_hd = q_flash.transpose(1, 2)[:, :, 0, :]
            k_fp = k_flash.transpose(1, 2)
            v_fp = v_flash.transpose(1, 2)
            n_kv = ctx.get("n_kv_heads")
            if n_kv is not None and k_fp.shape[1] != n_kv:
                rep = k_fp.shape[1] // n_kv
                k_fp = k_fp[:, ::rep].contiguous()
                v_fp = v_fp[:, ::rep].contiguous()
            num_sink, ws = ctx["num_sink"], ctx["window_size"]
            n_body_win = -(-max(k_fp.shape[2] - num_sink, 0) // ws)
            gate = ctx["gate"]
            sel, logmass = fused_gate(q_hd, gate["card"], gate["anchor"],
                                      ctx["scaling"], gate["n_sel"],
                                      head_norm=gate["head_norm"])
            # Both come from reused scratch buffers; the full read below reuses
            # the decode kernel's, so everything is copied out first.
            sel, logmass = sel.clone(), logmass.clone()
            out_full, wsum_full = fused_two_tier_decode(
                q_hd, k_fp, v_fp, ctx["qtier"], ctx["scaling"], num_sink,
                n_body_win)                              # sel=None: every window
            m = gate_error_metrics(out[:, 0], out_full.clone(), wsum_full.clone(),
                                   n_body_win, sel, logmass)
            for k, v in m.items():
                stats[layer][k].extend(v.flatten().tolist())
        return out

    flash_decode._run_fused = wrapped

    def uninstall():
        flash_decode._run_fused = orig

    return stats, uninstall


def summarize(stats) -> Dict:
    """Per-layer medians and upper deciles, plus the all-layer summary."""
    def q(xs, p):
        t = torch.tensor(xs, dtype=torch.float64)
        return float(t.quantile(p)) if t.numel() else float("nan")

    layers = {}
    pooled = defaultdict(list)
    for layer in sorted(stats):
        row = {}
        for k, xs in stats[layer].items():
            row[f"{k}_median"] = q(xs, 0.5)
            row[f"{k}_p90"] = q(xs, 0.9)
            row[f"{k}_p10"] = q(xs, 0.1)
            pooled[k].extend(xs)
        row["n"] = len(stats[layer].get("out_err", []))
        layers[int(layer)] = row
    overall = {f"{k}_{name}": q(xs, p) for k, xs in pooled.items()
               for name, p in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))}
    return {"layers": layers, "overall": overall}
