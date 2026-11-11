"""What Q -> F promotion buys, in attention mass: the paper's Table 18 on one run.

Table 18 compares the three-tier cache with and without Q -> F promotion on two
pilot runs over *different* source articles, and says so -- directional
evidence, not a paired estimate. One observation-collector run of the shipped
policy (``quant_promotion="bidir"``) pairs it exactly, in three columns:

* **with gated promotion** is the run as measured: its tiers, its read gate's
  picks (the card gate reads ``quant_gate_ratio`` of the int2 tier);
* **with promotion** is the same run with the gate off -- every int2 window
  read, as in the paper's pilot. Same tiers and scores; only what the decode
  step reads changes. Its R_Q needs the 2-bit read of the windows the gate
  skipped, which a gated run never records, so it is left empty (a
  ``--gate-ratio 1.0`` collector run measures it);
* **no promotion** is the one-way policy (``quant_promotion="oneway"``,
  ``F -> Q -> E``: the arm ``policy.compute_two_tier_retain`` implements)
  REPLAYED on the same trajectory. At every eviction it ranks the same signal
  the cache ranked on (``rank_signal``, head mean -- what the policy ranks) at
  the same tier sizes, with the windows it currently holds in int2 barred from
  the fp slots.

Both arms are scored against the same ground truth -- the same queries over
the never-evicted KV (``full_mass``) -- by the observation suite's own code
(``qevict_observations`` / ``utils.qevict_metrics``), so a difference between
them is the promotion rule and nothing else: same prompts, same queries, same
geometry, same bytes. Deltas are paired per ``(prompt, layer)`` trace.

Table 18's rows keep the paper's labels. R_Q, "FullKV attention mass
preserved", is int2 fidelity: of each query head's FullKV attention on the int2
tier, the share the decode step itself gives those windows (their 2-bit read,
or the card fill for windows the gate skipped), window by window
(:func:`int2_fidelity`). Beside them, where the mass went: Future Missed Mass
of the tiers read every step, the decode read ledger (fp, int2 read, int2
reached only through the card fill, evicted), and the mass a promoted window
carries against the fp window demoted for it.

What the replay assumes, each checked or bounded on the run itself
------------------------------------------------------------------
1. **The replay is the policy.** Replaying the *bidirectional* rule on
   ``rank_signal`` must reproduce the recorded tiers; the check
   ``bidir_replay_mismatch`` counts the layer-evictions where it does not.
   (Wikitext q0.70 and q0.20: none of 16,384.)
2. **Scores do not depend on the arm.** A window keeps the score the recorded
   run gave it. Under one-way a window the run promoted would have stayed int2
   and accumulated its gated read (or fill) instead of an fp read; M5 of the
   tier study puts the two at cosine 0.9989 on the int2 tier. A window the
   replay keeps but the run had already evicted has no score; it is counted
   (``unscored``) and ranked last.
3. **The gate.** Where the one-way int2 tier differs from the recorded one, the
   gate is re-run by its own rule (``sketch.group_share`` + ``select_windows``:
   each window's largest per-head share of that head's int2 mass over the GQA
   group, top ``n_sel`` per KV head) on the card estimates the run recorded.
   ``gate_check`` re-runs the rule on the recorded tier and counts the
   ``(step, layer, KV head)`` picks it reproduces. A window the run had
   promoted has no card estimate at those steps (an fp window's card is not
   read), so the gate's treatment of it is bracketed:

   * ``open`` -- the gate opens it for every KV head, i.e. never misses the
     window promotion would have made permanent. **This favours no promotion,
     and is the primary arm**: every promotion benefit reported against it is
     a lower bound on this count.
   * ``closed`` -- the gate never opens it. Favours promotion; an upper bound.

   If the one-way tier holds more such windows than the gate reads, ``open``
   reads the ones with the most FullKV mass (``gate_over_budget_rows``).

A collector run of the real one-way arm (``--promotion oneway``) removes
assumptions 2 and 3: pass it as ``--oneway-zip`` and the no-promotion column is
measured, paired per ``(prompt, layer)`` but on its own decode trajectory. A
bidirectional ``--gate-ratio 1.0`` run passed as ``--gate-off-zip`` measures the
gate-off column the same way, R_Q included.

CLI::

    python -m modules.evaluation.promotion_ablation \\
        --zip outputs/obs_data/<run>.zip --out-dir outputs/promotion_ablation \\
        [--oneway-zip outputs/obs_data/<run>_oneway.zip] \\
        [--gate-off-zip outputs/obs_data/<run at --gate-ratio 1.0>.zip] [--horizon 32]
"""

from __future__ import annotations

import argparse
import itertools
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from modules.evaluation import qevict_observations as QO
from modules.evaluation.observation_collector import (
    BASE_KEYS, OURS_KEYS, TIER_EVICTED, TIER_FP, TIER_LOCAL, TIER_Q, iter_zip_samples,
    parity_base, parity_ours)
from utils import qevict_metrics as QM
from utils.logger import get_logger

log = get_logger(__name__)

#: What one pass over a collector zip reads per sample.
REPLAY_KEYS = ("full_mass", "tier", "gate_open", "gate_fired", "evict_step",
               "tokens", "rank_signal", "rank_signal_steps", "card_logmass",
               "cache_step_mass")
GATE_BRACKET = ("open", "closed")


# ---------------------------------------------------------------------------
# the one-way replay
# ---------------------------------------------------------------------------


def tier_counts(merged: int, top_k_fp: int, n_q: int, local_w: int
                ) -> Tuple[int, int, int]:
    """``(k_fp, n_q, local_w)`` for a merged axis of ``merged`` windows --
    ``WindowPolicy.tier_counts``, which is what the cache sizes its tiers by."""
    local_w = min(local_w, merged)
    ev = merged - local_w
    if ev <= 0:
        return 0, 0, local_w
    k_fp = min(top_k_fp, ev)
    return k_fp, min(n_q, ev - k_fp), local_w


def _desc(scores: np.ndarray) -> np.ndarray:
    """Indices by descending score; ties keep ascending window id (``-inf`` last)."""
    return np.argsort(-scores, kind="stable")


def replay_tiers(tier: np.ndarray, evict_step: np.ndarray, rank_signal: np.ndarray,
                 rank_steps: np.ndarray, top_k_fp: int, n_q: int, local_w: int,
                 *, oneway: bool) -> Tuple[np.ndarray, Dict[str, int]]:
    """Replay the tier policy on one sample's recorded ranking signal.

    ``tier`` ``[T, L, W]`` is the recorded run's; the result is the same array
    with every window the replay decided on relabelled -- fp, int2, or evicted
    -- from each eviction to the next. Everything the replay does not decide
    (sinks, the local band, windows that left the local band since the last
    eviction and wait for the next one, windows not yet created) is the
    recorded run's, because it is geometry, not policy.

    At each eviction the evictable band is the replay's own survivors plus the
    windows that left the local band since the previous eviction; the local
    band is the newest windows of the recorded merged axis (the finite columns
    of ``rank_signal``), which no policy evicts. ``oneway=False`` must
    reproduce ``tier``; ``oneway=True`` bars the windows the replay currently
    holds in int2 from the fp slots (``policy.compute_two_tier_retain`` with
    ``no_promote``): fp slots go to the best of the rest, int2 slots to the best
    of everything the fp pick left.
    """
    T, L, W = tier.shape
    ev = np.flatnonzero(evict_step)
    if not np.array_equal(ev, np.asarray(rank_steps)):
        raise ValueError(f"rank_signal_steps {list(rank_steps)[:4]}... do not match "
                         f"the eviction steps {list(ev)[:4]}...")
    out = tier.copy()
    diag = {"events": 0, "unscored": 0, "promotions": 0, "demotions": 0,
            "kept_differs": 0, "fp_differs": 0}
    for li in range(L):
        fp = q = np.zeros(0, np.int64)
        gone = np.zeros(W, bool)
        prev_local: Optional[int] = None
        for r, e in enumerate(ev):
            sig = rank_signal[r, li].astype(np.float64).mean(0)       # [W]
            merged = np.flatnonzero(np.isfinite(sig))
            _, _, lw = tier_counts(merged.size, top_k_fp, n_q, local_w)
            # The local band is the newest `lw` windows, never evicted, so it is
            # contiguous: everything from its first id up is local.
            local0 = int(merged[merged.size - lw]) if lw else int(merged[-1]) + 1
            if prev_local is None:
                band = merged[merged < local0]
            else:
                band = np.union1d(np.union1d(fp, q), np.arange(prev_local, local0))
            k_fp, k_q, _ = tier_counts(band.size + lw, top_k_fp, n_q, local_w)
            s = sig[band]
            bad = ~np.isfinite(s)
            diag["unscored"] += int(bad.sum())
            s = np.where(bad, -np.inf, s)
            if oneway:
                barred = np.isin(band, q)
                fp_new = band[_desc(np.where(barred, -np.inf, s))[:k_fp]]
                taken = np.isin(band, fp_new)
                q_new = band[_desc(np.where(taken, -np.inf, s))[:k_q]]
            else:
                order = band[_desc(s)]
                fp_new, q_new = order[:k_fp], order[k_fp:k_fp + k_q]
            diag["promotions"] += int(np.isin(fp_new, q).sum())
            diag["demotions"] += int(np.isin(q_new, fp).sum())
            gone[np.setdiff1d(band, np.union1d(fp_new, q_new))] = True
            fp, q, prev_local = np.sort(fp_new), np.sort(q_new), local0
            e1 = ev[r + 1] if r + 1 < ev.size else T
            blk = out[e:e1, li]
            blk[:, gone] = TIER_EVICTED
            blk[:, fp] = TIER_FP
            blk[:, q] = TIER_Q
            rec = tier[e, li]
            kept_rec = np.flatnonzero((rec == TIER_FP) | (rec == TIER_Q))
            diag["events"] += 1
            diag["kept_differs"] += int(not np.array_equal(kept_rec, np.union1d(fp, q)))
            diag["fp_differs"] += int(not np.array_equal(np.flatnonzero(rec == TIER_FP), fp))
    return out, diag


# ---------------------------------------------------------------------------
# the read gate, re-run by its own rule
# ---------------------------------------------------------------------------


def gate_pick(card: np.ndarray, known: np.ndarray, forced: np.ndarray,
              n_sel: np.ndarray, n_kv: int, norm: Optional[np.ndarray] = None
              ) -> np.ndarray:
    """The read gate's selection rule, batched over ``N`` (step) rows.

    ``card`` ``[N, H, W]`` the card log-mass estimates; ``known`` ``[N, W]`` the
    int2 windows eligible to be picked; ``forced`` ``[N, W]`` (every KV head) or
    ``[N, n_kv, W]`` windows opened regardless -- they take their slots first;
    ``n_sel`` ``[N]`` windows read per KV head. Returns ``[N, n_kv, W]`` bool.
    ``norm`` ``[N, W]`` is the int2 tier each head's share is taken over
    (default ``known``); every window in it needs a card estimate.

    ``sketch.group_share`` then ``select_windows``: each query head's log share
    of its own int2 mass (``logmass - logsumexp`` over the tier), the max over
    the query heads sharing a KV head, the top ``n_sel`` per KV head. Never raw
    logits across heads (CLAUDE.md).
    """
    N, H, W = card.shape
    rep = H // n_kv
    norm = known if norm is None else norm
    x = np.where(norm[:, None, :], card.astype(np.float64), -np.inf)
    m = x.max(-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = x - (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))
    g = np.where(known[:, None, :], share, -np.inf).reshape(N, n_kv, rep, W).max(2)
    order = np.argsort(-g, axis=-1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(W), order.shape), -1)
    f = forced if forced.ndim == 3 else np.broadcast_to(forced[:, None, :], (N, n_kv, W))
    k = np.asarray(n_sel)[:, None, None] - f.sum(-1, keepdims=True)
    return ((rank < k) & known[:, None, :]) | f


def _n_sel(gate_open: np.ndarray) -> np.ndarray:
    """``[T, L, Hk, W] -> [T, L]`` windows the recorded gate read per KV head."""
    return gate_open.sum(-1).max(-1)


def gate_check(tier: np.ndarray, gate_open: np.ndarray, gate_fired: np.ndarray,
               card_logmass: np.ndarray) -> Tuple[int, int]:
    """``(reproduced, total)`` ``(step, layer, KV head)`` picks: the rule re-run
    on the recorded int2 tier and card estimates, against the recorded pick."""
    T, L, Hk, W = gate_open.shape
    n_sel = _n_sel(gate_open)
    same = total = 0
    for li in range(L):
        known = tier[:, li] == TIER_Q
        rows = gate_fired[:, li] & known.any(-1)
        if not rows.any():
            continue
        pick = gate_pick(card_logmass[rows, li], known[rows], np.zeros_like(known[rows]),
                         n_sel[rows, li], Hk)
        rec = gate_open[rows, li] & known[rows][:, None, :]
        same += int((pick == rec).all(-1).sum())
        total += int(pick.shape[0] * Hk)
    return same, total


def _top_by_truth(full_mass: np.ndarray, q_arm: np.ndarray, extra: np.ndarray,
                  n_sel: np.ndarray, n_kv: int) -> np.ndarray:
    """``[N, n_kv, W]``: per KV head, the ``n_sel`` windows of ``extra`` with the
    largest FullKV share of any member head's int2-tier mass -- the gate's own
    criterion, scored on the truth instead of the card."""
    N, H, W = full_mass.shape
    m = np.where(q_arm[:, None, :], full_mass.astype(np.float64), 0.0)
    share = m / np.maximum(m.sum(-1, keepdims=True), 1e-30)
    g = np.where(extra[:, None, :], share, -np.inf).reshape(N, n_kv, H // n_kv, W).max(2)
    order = np.argsort(-g, axis=-1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(W), order.shape), -1)
    return (rank < np.asarray(n_sel)[:, None, None]) & extra[:, None, :]


def replay_gate(tier_rec: np.ndarray, tier_arm: np.ndarray, gate_open: np.ndarray,
                gate_fired: np.ndarray, card_logmass: np.ndarray, blocked: str,
                full_mass: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, Dict[str, int]]:
    """The gate's picks over the replay's int2 tier.

    Where the replay holds the same int2 windows as the run, the recorded pick
    stands. Elsewhere the rule is re-run on the recorded card estimates of the
    int2 windows both hold, at the recorded ``n_sel``, ranked exactly as the
    run's gate ranked them (each head's share taken over the run's int2 tier),
    so under ``open`` the replay reads a subset of what the run read. Windows
    only the replay holds in int2 (the run had promoted them, so no card was
    read) are opened for every KV head (``blocked="open"``) or never
    (``"closed"``).

    When the replay holds more of them than the gate reads (a small ``n_sel``
    against several blocked promotions), ``open`` fills every slot from them,
    per KV head the ones with the most FullKV mass (``full_mass`` ``[T, L, H, W]``
    is then required) -- still the choice most favourable to no promotion.
    Such rows are counted (``over_budget_rows``).
    """
    if blocked not in GATE_BRACKET:
        raise ValueError(f"blocked must be one of {GATE_BRACKET}, got {blocked!r}")
    T, L, Hk, W = gate_open.shape
    n_sel = _n_sel(gate_open)
    out = gate_open.copy()
    diag = {"rerun_rows": 0, "blocked_windows": 0, "over_budget_rows": 0}
    for li in range(L):
        q_rec = tier_rec[:, li] == TIER_Q
        q_arm = tier_arm[:, li] == TIER_Q
        rows = gate_fired[:, li] & (q_rec != q_arm).any(-1)
        if not rows.any():
            continue
        known = q_arm[rows] & q_rec[rows]
        extra = q_arm[rows] & ~q_rec[rows]
        ns = n_sel[rows, li]
        forced: np.ndarray = extra if blocked == "open" else np.zeros_like(extra)
        over = extra.sum(-1) > ns
        if blocked == "open" and over.any():
            if full_mass is None:
                raise ValueError("more int2 windows without a card than the gate "
                                 "reads: pass full_mass to rank them")
            forced = np.broadcast_to(extra[:, None, :], (extra.shape[0], Hk, W)).copy()
            idx = np.flatnonzero(rows)[over]
            forced[over] = _top_by_truth(full_mass[idx, li], q_arm[idx], extra[over],
                                         ns[over], Hk)
            diag["over_budget_rows"] += int(over.sum())
        out[rows, li] = gate_pick(card_logmass[rows, li], known, forced, ns, Hk,
                                  norm=q_rec[rows])
        diag["rerun_rows"] += int(rows.sum())
        diag["blocked_windows"] += int(extra.sum())
    return out, diag


# ---------------------------------------------------------------------------
# per-sample mass metrics that need the zip (the parity export drops them)
# ---------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return np.nan
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def score_agreement(tier: np.ndarray, evict_step: np.ndarray, rank_signal: np.ndarray,
                    full_mass: np.ndarray) -> np.ndarray:
    """``[L]`` quantized-score agreement: at each eviction, Spearman between the
    cache's ranking signal (head mean) and the FullKV cumulative mass through
    the previous step (head mean), over the windows that eviction puts in int2;
    mean over evictions."""
    ev = np.flatnonzero(evict_step)
    T, L, W = tier.shape
    out = np.full(L, np.nan)
    for li in range(L):
        cum = np.cumsum(full_mass[:, li].astype(np.float32).mean(1), 0)   # [T, W]
        vals = []
        for r, e in enumerate(ev):
            sig = rank_signal[r, li].astype(np.float64).mean(0)
            qw = np.flatnonzero((tier[e, li] == TIER_Q) & np.isfinite(sig))
            vals.append(_spearman(sig[qw], cum[e - 1][qw]))
        out[li] = QM.nanmean(np.asarray(vals))
    return out


def swap_mass(tier: np.ndarray, evict_step: np.ndarray, full_mass: np.ndarray,
              horizon: int) -> Dict[str, np.ndarray]:
    """Per layer, the FullKV mass of each promoted window and of each fp window
    demoted to int2 at the same eviction (the slots the promotions took), over
    the next ``horizon`` steps. Under one-way those demoted windows would still
    hold the fp slots, so the pair is the per-decision counterfactual.

    Mass is a share of each query head's attention per step, averaged over the
    steps and heads (per-head shares, so the average never mixes raw mass
    across heads). Returns ``[L]`` sums and counts: ``promoted_sum``,
    ``promoted_n``, ``demoted_sum``, ``demoted_n``. Only evictions with a full
    horizon count, the same censoring FMM uses; steps ``e+1 .. e+horizon``.
    """
    ev = np.flatnonzero(evict_step)
    T, L, W = tier.shape
    out = {k: np.zeros(L) for k in ("promoted_sum", "promoted_n",
                                    "demoted_sum", "demoted_n")}
    for li in range(L):
        for r in range(1, ev.size):
            e = ev[r]
            if e + horizon > T - 1:
                break
            prev, cur = tier[ev[r - 1], li], tier[e, li]
            pro = (prev == TIER_Q) & (cur == TIER_FP)
            dem = (prev == TIER_FP) & (cur == TIER_Q)
            if not pro.any():
                continue
            fut = full_mass[e + 1:e + horizon + 1, li].astype(np.float32).mean((0, 1))
            out["promoted_sum"][li] += fut[pro].sum()
            out["promoted_n"][li] += pro.sum()
            out["demoted_sum"][li] += fut[dem].sum()
            out["demoted_n"][li] += dem.sum()
    return out


# ---------------------------------------------------------------------------
# R_Q: how much of the int2 tier's FullKV attention the decode step preserves
# ---------------------------------------------------------------------------


def card_fill(credit: np.ndarray, logmass: np.ndarray, read: np.ndarray) -> np.ndarray:
    """The kernel's credit for a skipped int2 window, per query head.

    ``exp(logmass_i + mean over read windows j of (ln credit_j - logmass_j))``
    -- ``decode_kernel``'s gated epilogue and ``scorer.fill_skipped_window_scores``.
    ``credit`` ``[..., W]`` is the cache's own window mass (``cache_step_mass``);
    the kernel's final renormalisation divides read and skipped windows alike, so
    it cancels out of the mean and the fill comes out on ``credit``'s own scale.
    ``read`` marks the read windows that have a card (``logmass`` finite). NaN
    where a row has none.
    """
    ok = read & (credit > 0) & np.isfinite(logmass)
    d = np.where(ok, np.log(np.where(ok, credit, 1.0)) - np.where(ok, logmass, 0.0), 0.0)
    cnt = ok.sum(-1, keepdims=True)
    dmean = np.where(cnt > 0, d.sum(-1, keepdims=True) / np.maximum(cnt, 1), np.nan)
    return np.exp(logmass.astype(np.float64) + dmean)


def _per_head(gate: np.ndarray, rep: int) -> np.ndarray:
    """``[..., Hk, W] -> [..., H, W]``: query head ``h`` reads KV head ``h // rep``."""
    return np.repeat(gate, rep, axis=-2)


def fill_check(tier: np.ndarray, gate_open: np.ndarray, gate_fired: np.ndarray,
               card_logmass: np.ndarray, cache_step_mass: np.ndarray,
               floor: float = 1e-4) -> Dict[str, int]:
    """:func:`card_fill` against the fills the run recorded, on every skipped
    int2 window whose recorded credit is at least ``floor`` (below it fp16 has
    no relative precision). Counts within 1% and 5%."""
    T, L, Hk, W = gate_open.shape
    rep = card_logmass.shape[2] // Hk
    out = {"windows": 0, "within_1pct": 0, "within_5pct": 0}
    for li in range(L):
        rows = np.flatnonzero(gate_fired[1:, li]) + 1
        q = (tier[rows, li] == TIER_Q)[:, None, :]
        read = _per_head(gate_open[rows, li], rep) & q
        c = cache_step_mass[rows, li].astype(np.float64)
        lm = card_logmass[rows, li].astype(np.float64)
        pred = card_fill(c, lm, read)
        sel = q & ~read & (c >= floor) & np.isfinite(pred)
        r = np.abs(pred[sel] / c[sel] - 1.0)
        out["windows"] += int(sel.sum())
        out["within_1pct"] += int((r <= 0.01).sum())
        out["within_5pct"] += int((r <= 0.05).sum())
    return out


def int2_fidelity(tier_rec: np.ndarray, tier_arm: np.ndarray, gate_rec: np.ndarray,
                  gate_arm: np.ndarray, gate_fired: np.ndarray, card_logmass: np.ndarray,
                  cache_step_mass: np.ndarray, full_mass: np.ndarray,
                  exclude: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """R_Q, "FullKV attention mass preserved" over the int2 tier, per layer:
    ``[L]`` numerator and denominator summed over decode steps and query heads.

    For each query head and step, every int2 window's FullKV mass ``a_w`` is set
    against the mass the decode step itself gives it, ``c_w`` -- its 2-bit read
    where the gate opened it, its card fill where it did not (``cache_step_mass``)
    -- rescaled to the FullKV scale by the local band, which both read exactly.
    A window preserves ``min(c_w, a_w)``: credit it did not have in FullKV is not
    preserved mass. ``R_Q = sum min(c, a) / sum a`` over the int2 tier.

    ``tier_arm`` / ``gate_arm`` may be the replay's: on the steps where its int2
    tier or picks differ from the run's, every skipped window's fill is
    recomputed with :func:`card_fill` over the replay's read set; windows it
    reads are credited what the run recorded for them -- the 2-bit read, or for
    a window the run held in fp by promotion, the fp read of the dequantized
    payload promotion put there (the same values). Where none of the replay's
    read windows has a card (every slot went to windows the run had promoted),
    the fill is calibrated on the run's own read windows at that step instead
    (``fill_from_run``). A head-step is left out when an int2 window has no
    credit -- a window the run promoted that the replay's gate had no slot for,
    so neither a read nor a card -- or its local band carries no mass. Those
    head-steps come back as ``diag["dropped"]`` ``[T, L, H]``; pass them as
    ``exclude`` to the measured arm's call so both arms are scored on the same
    head-steps. Diagnostics also count the read windows whose recorded credit
    was a fill (``read_without_record``).
    """
    T, L, Hk, W = gate_rec.shape
    rep = full_mass.shape[2] // Hk
    num, den = np.zeros(L), np.zeros(L)
    diag: Dict[str, Any] = {"recomputed_rows": 0, "read_without_record": 0,
                            "fill_from_run": 0, "dropped_head_steps": 0,
                            "dropped": np.zeros((T, L, full_mass.shape[2]), bool)}
    for li in range(L):
        rows = np.flatnonzero(gate_fired[1:, li]) + 1
        a = full_mass[rows, li].astype(np.float64)                     # [n, H, W]
        c = cache_step_mass[rows, li].astype(np.float64)
        q_rec, q_arm = tier_rec[rows, li] == TIER_Q, tier_arm[rows, li] == TIER_Q
        r_rec = _per_head(gate_rec[rows, li], rep) & q_rec[:, None, :]
        r_arm = _per_head(gate_arm[rows, li], rep) & q_arm[:, None, :]
        moved = (q_rec != q_arm).any(-1) | (r_rec != r_arm).any((-1, -2))
        if moved.any():
            lm = card_logmass[rows[moved], li].astype(np.float64)
            carded = r_arm[moved] & q_rec[moved][:, None, :]
            fill = card_fill(c[moved], lm, carded)
            none = ~(carded & (c[moved] > 0) & np.isfinite(lm)).any(-1)        # [n', H]
            if none.any():
                fill = np.where(none[..., None], card_fill(c[moved], lm, r_rec[moved]), fill)
                diag["fill_from_run"] += int(none.sum())
            skip = q_arm[moved][:, None, :] & ~r_arm[moved]
            c[moved] = np.where(skip, fill, c[moved])
            diag["recomputed_rows"] += int(moved.sum())
            diag["read_without_record"] += int(
                (r_arm[moved] & q_rec[moved][:, None, :] & ~r_rec[moved]).sum())
        local = (tier_rec[rows, li] == TIER_LOCAL)[:, None, :]
        la = np.where(local, a, 0.0).sum(-1)
        lc = np.where(local, c, 0.0).sum(-1)
        scale = np.where(la > 1e-4, lc / np.maximum(la, 1e-30), np.nan)
        q3 = q_arm[:, None, :]
        credited = ~(q3 & ~np.isfinite(c)).any(-1)                      # [n, H]
        use = np.isfinite(scale) & credited
        if exclude is not None:
            use &= ~exclude[rows, li]
        drop = ~use & q_arm.any(-1)[:, None]
        diag["dropped"][rows, li] = drop
        diag["dropped_head_steps"] += int(drop.sum())
        ok = use[..., None] & q3
        chat = np.where(ok, c, 0.0) / np.where(use, scale, 1.0)[..., None]
        num[li] = float(np.where(ok, np.minimum(chat, a), 0.0).sum())
        den[li] = float(np.where(ok, a, 0.0).sum())
    return num, den, diag


# ---------------------------------------------------------------------------
# one pass over the zip
# ---------------------------------------------------------------------------


@dataclass
class Replay:
    """Everything one pass over a bidirectional run's zip produces."""
    meta: Dict[str, Any]
    arms: Dict[str, List[Dict[str, np.ndarray]]]   # arm -> per-sample OURS_KEYS
    qsa: Dict[str, List[np.ndarray]]                # arm -> per-sample [L]
    rq: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]  # arm -> per-sample ([L], [L])
    swap: List[Dict[str, np.ndarray]]               # per-sample swap_mass
    checks: Dict[str, int] = field(default_factory=dict)


def _geometry(meta: Dict[str, Any]) -> Tuple[int, int, int]:
    g = meta["resolved_geometry"]
    return int(g["top_k_fp"]), int(g["N_q"]), int(g["local_windows"])


def _read_meta(path: str | Path) -> Dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("meta.json"))


def _samples(zip_path: str | Path, keys: Sequence[str], limit: Optional[int]
             ) -> Iterator[Dict[str, np.ndarray]]:
    return itertools.islice(iter_zip_samples(zip_path, keys), limit)


def replay_run(zip_path: str | Path, horizon: int = 32,
               limit: Optional[int] = None) -> Replay:
    """Every arm, one sample at a time: the measured run (with gated
    promotion), the same tiers with every int2 window read (with promotion, gate
    off) and both gate brackets of the one-way replay (no promotion). The ground
    truth (``full_mass``) is read again by :func:`run` through ``parity_base``;
    nothing here holds more than one sample of it."""
    meta = _read_meta(zip_path)
    promo = meta["config"].get("quant_promotion", "bidir")
    if promo != "bidir":
        raise ValueError(f"{zip_path} is a {promo!r} run: the replay needs the "
                         "bidirectional run, and replays the one-way arm on it")
    k_fp, n_q, local_w = _geometry(meta)
    arms: Dict[str, List[Dict[str, np.ndarray]]] = {
        "with": [], "gate_off": [], **{f"without_{b}": [] for b in GATE_BRACKET}}
    qsa: Dict[str, List[np.ndarray]] = {"with": [], "without": []}
    rq: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {"with": [], "without": []}
    swap: List[Dict[str, np.ndarray]] = []
    checks = {"bidir_replay_events": 0, "bidir_replay_mismatch": 0,
              "gate_rows": 0, "gate_rows_reproduced": 0,
              "oneway_events": 0, "oneway_unscored": 0, "oneway_promotions": 0,
              "oneway_kept_differs": 0, "oneway_fp_differs": 0,
              "gate_rerun_rows": 0, "gate_blocked_windows": 0,
              "gate_over_budget_rows": 0, "fill_windows": 0, "fill_within_1pct": 0,
              "fill_within_5pct": 0, "rq_recomputed_rows": 0,
              "rq_read_without_record": 0, "rq_fill_from_run": 0,
              "rq_dropped_head_steps": 0, "rq_head_steps": 0}
    for i, a in enumerate(_samples(zip_path, REPLAY_KEYS, limit)):
        tier, ev_mask, rs = a["tier"], a["evict_step"], a["rank_signal"]
        rsteps = a["rank_signal_steps"]
        bidir, d_b = replay_tiers(tier, ev_mask, rs, rsteps, k_fp, n_q, local_w,
                                  oneway=False)
        ev = np.flatnonzero(ev_mask)
        checks["bidir_replay_events"] += d_b["events"]
        checks["bidir_replay_mismatch"] += int(
            (bidir[ev] != tier[ev]).any(-1).sum())
        same, tot = gate_check(tier, a["gate_open"], a["gate_fired"], a["card_logmass"])
        checks["gate_rows_reproduced"] += same
        checks["gate_rows"] += tot
        fc = fill_check(tier, a["gate_open"], a["gate_fired"], a["card_logmass"],
                        a["cache_step_mass"])
        for k, v in fc.items():
            checks[f"fill_{k}"] += v
        one, d_o = replay_tiers(tier, ev_mask, rs, rsteps, k_fp, n_q, local_w,
                                oneway=True)
        for k in ("events", "unscored", "promotions", "kept_differs", "fp_differs"):
            checks[f"oneway_{k}"] += d_o[k]
        common = {"gate_fired": a["gate_fired"], "evict_step": ev_mask,
                  "tokens": a["tokens"]}
        arms["with"].append({"tier": tier, "gate_open": a["gate_open"], **common})
        every = np.broadcast_to((tier == TIER_Q)[:, :, None, :], a["gate_open"].shape)
        arms["gate_off"].append({"tier": tier, "gate_open": every, **common})
        fid = (a["card_logmass"], a["cache_step_mass"], a["full_mass"])
        for b in GATE_BRACKET:
            g, d_g = replay_gate(tier, one, a["gate_open"], a["gate_fired"],
                                 a["card_logmass"], b, a["full_mass"])
            if b == "open":
                checks["gate_rerun_rows"] += d_g["rerun_rows"]
                checks["gate_blocked_windows"] += d_g["blocked_windows"]
                checks["gate_over_budget_rows"] += d_g["over_budget_rows"]
                n, d, d_f = int2_fidelity(tier, one, a["gate_open"], g,
                                          a["gate_fired"], *fid)
                rq["without"].append((n, d))
                for k in ("recomputed_rows", "read_without_record", "fill_from_run",
                          "dropped_head_steps"):
                    checks[f"rq_{k}"] += d_f[k]
                checks["rq_head_steps"] += int(
                    (a["gate_fired"][1:] & (tier[1:] == TIER_Q).any(-1)).sum()
                    * a["full_mass"].shape[2])
                # the measured arm, on the head-steps the replay could score
                n, d, _ = int2_fidelity(tier, tier, a["gate_open"], a["gate_open"],
                                        a["gate_fired"], *fid, exclude=d_f["dropped"])
                rq["with"].append((n, d))
            arms[f"without_{b}"].append({"tier": one, "gate_open": g, **common})
        qsa["with"].append(score_agreement(tier, ev_mask, rs, a["full_mass"]))
        qsa["without"].append(score_agreement(one, ev_mask, rs, a["full_mass"]))
        swap.append(swap_mass(tier, ev_mask, a["full_mass"], horizon))
        log.info("sample %d: bidir replay mismatches %d, one-way fp differs on %d "
                 "of %d layer-evictions", i, checks["bidir_replay_mismatch"],
                 d_o["fp_differs"], d_o["events"])
        del a
    return Replay(meta=meta, arms=arms, qsa=qsa, rq=rq, swap=swap, checks=checks)


def measured_arm(zip_path: str | Path, limit: Optional[int] = None
                 ) -> Tuple[Dict[str, Any], List[Dict[str, np.ndarray]], List[np.ndarray],
                            List[Tuple[np.ndarray, np.ndarray]]]:
    """A recorded run's own tiers and gate (for ``--oneway-zip``), its
    quantized-score agreement and its R_Q."""
    meta = _read_meta(zip_path)
    samples, qsa, rq = [], [], []
    for a in _samples(zip_path, REPLAY_KEYS, limit):
        samples.append({k: a[k] for k in OURS_KEYS})
        qsa.append(score_agreement(a["tier"], a["evict_step"], a["rank_signal"],
                                   a["full_mass"]))
        n, d, _ = int2_fidelity(a["tier"], a["tier"], a["gate_open"], a["gate_open"],
                                a["gate_fired"], a["card_logmass"], a["cache_step_mass"],
                                a["full_mass"])
        rq.append((n, d))
    return meta, samples, qsa, rq


# ---------------------------------------------------------------------------
# the suite's own metrics, per (prompt, layer) trace, for one arm
# ---------------------------------------------------------------------------


def arm_traces(base: Dict[str, Any], ours: Dict[str, Any], horizon: int, lir_m: int,
               ledger_only: bool = False) -> Dict[str, np.ndarray]:
    """Per-trace ``[M]`` values for one arm, from the suite's own functions."""
    inp = QO.build_observation_inputs(base, ours)
    reads = QO.analyse_decode_reads(base, ours, inp, n_boot=0)
    led = reads["ledger_by_trace"]                                   # [M, P]
    part = {n: led[:, i] for i, n in enumerate(QM.LEDGER_PARTS)}
    win = sum(part[n] for n in QM.LEDGER_PARTS if n != "sink")
    held = part["fp"] + part["q_read"] + part["q_skipped"]
    out = {
        "labels": np.asarray(inp.trace_labels),
        "groups": inp.trace_group,
        "fp": part["fp"], "q_read": part["q_read"], "q_skipped": part["q_skipped"],
        "evicted": part["evicted"],
        "read": sum(part[n] for n in QM.LEDGER_PARTS if n not in ("q_skipped", "evicted")),
        "fp_window_share": np.divide(part["fp"], win, out=np.full_like(win, np.nan),
                                     where=win > 0),
        "held_read": np.divide(part["fp"] + part["q_read"], held,
                               out=np.full_like(held, np.nan), where=held > 0),
        "q_recall": QM.nanmean(reads["head_recall"], axis=1),
    }
    if ledger_only:
        return out
    for name, key in (("fmm_kept", "measured_fp_plus_q"), ("fmm_exact", "measured_fp_only")):
        f = QM.future_missed_mass(inp.mass_step, inp.accessible[key], inp.event_steps,
                                  horizon, creation_steps=inp.creation)
        out[name] = QM.nanmean(f, axis=1)
    R, W = inp.event_steps.size, inp.mass_step.shape[2]
    creation_ev = np.minimum(np.searchsorted(inp.band.sum(1), np.arange(W) + 1,
                                             side="left"), R)
    lir = QM.episode_lir(inp.policy_selected, lir_m, None, creation_ev)
    out["lir_rescued"] = np.asarray(lir["rescued_by_trace"], float)
    out["lir_eligible"] = np.asarray(lir["eligible_by_trace"], float)
    counts = QM.tier_transitions(inp.states, 1)["counts_by_trace"]
    out["promotions"] = counts[:, QM.STATE_Q, QM.STATE_F].astype(float)
    out["demotions"] = counts[:, QM.STATE_F, QM.STATE_Q].astype(float)
    return out


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

#: (key, label, better, unit). ``better`` is "down" / "up" / "" (a count).
#: The first five are Table 18's rows, under the paper's own labels; the rest
#: say where the mass went.
ROWS: Tuple[Tuple[str, str, str, str], ...] = (
    ("fmm_kept", "R3 FMM", "down", "pct"),
    ("qsa", "Quantized-Score Agreement", "up", "num"),
    ("rq", "FullKV attention mass preserved, R_Q", "up", "pct"),
    ("lir", "Global LIR", "up", "pct"),
    ("promotions", "Recorded Q→F transitions", "", "count"),
    ("fmm_exact", "FMM of the tiers read every step (fp + local)", "down", "pct"),
    ("fp", "FullKV mass on the fp tier", "up", "pct"),
    ("fp_window_share", "  same, share of window mass (sinks excluded)", "up", "pct"),
    ("read", "FullKV mass the decode step reads (sinks + local + fp + read int2)",
     "up", "pct"),
    ("q_skipped", "FullKV mass reached only through the card fill", "down", "pct"),
    ("q_recall", "int2 FullKV mass in windows the decode reads (gate recall)", "up", "pct"),
    ("held_read", "fp + int2 FullKV mass the decode reads, (fp + read) / (fp + int2)",
     "up", "pct"),
)

#: Rows the gate cannot move under the replay: the gate-off column carries the
#: measured run's values for them (same tiers, same scores).
GATE_FREE = ("fmm_kept", "fmm_exact", "qsa", "lir_rescued", "lir_eligible",
             "promotions", "demotions", "fp", "fp_window_share")


def _paired(no: np.ndarray, yes: np.ndarray, groups: np.ndarray, n_boot: int,
            seed: int) -> Dict[str, float]:
    ok = np.isfinite(no) & np.isfinite(yes)
    g = groups[ok]
    a, b, d = (QM.group_reduce(x[ok], g) for x in (no, yes, yes - no))
    mu_a = QM.bootstrap_mean_ci(a, 0.95, n_boot, seed)
    mu_b = QM.bootstrap_mean_ci(b, 0.95, n_boot, seed + 1)
    mu_d = QM.bootstrap_mean_ci(d, 0.95, n_boot, seed + 2)
    return {"without": mu_a[0], "without_lo": mu_a[1], "without_hi": mu_a[2],
            "with": mu_b[0], "with_lo": mu_b[1], "with_hi": mu_b[2],
            "delta": mu_d[0], "delta_lo": mu_d[1], "delta_hi": mu_d[2],
            "traces": int(ok.sum())}


def _paired_ratio(num_a, den_a, num_b, den_b, n_boot: int, seed: int) -> Dict[str, float]:
    """Ratio-of-sums rates (LIR), bootstrapped over traces together."""
    def rate(n, d):
        return float(n.sum() / d.sum()) if d.sum() > 0 else np.nan
    out = {"without": rate(num_a, den_a), "with": rate(num_b, den_b),
           "without_n": [int(num_a.sum()), int(den_a.sum())],
           "with_n": [int(num_b.sum()), int(den_b.sum())]}
    out["delta"] = out["with"] - out["without"]
    rng = np.random.default_rng(seed)
    M = num_a.size
    boots = []
    for _ in range(max(n_boot, 0)):
        i = rng.integers(0, M, M)
        ra, rb = rate(num_a[i], den_a[i]), rate(num_b[i], den_b[i])
        boots.append((0.0 if np.isnan(ra) else ra, rb))
    if boots:
        bo = np.asarray(boots)
        out["without_lo"], out["without_hi"] = np.nanquantile(bo[:, 0], [0.025, 0.975])
        out["with_lo"], out["with_hi"] = np.nanquantile(bo[:, 1], [0.025, 0.975])
        out["delta_lo"], out["delta_hi"] = np.nanquantile(bo[:, 1] - bo[:, 0],
                                                          [0.025, 0.975])
    return out


def build_table(no: Dict[str, np.ndarray], yes: Dict[str, np.ndarray],
                n_boot: int = 2000, seed: int = 0,
                gate_off: Optional[Dict[str, np.ndarray]] = None
                ) -> Dict[str, Dict[str, Any]]:
    """Every row of :data:`ROWS`, paired per trace: no promotion (``no``) vs
    with gated promotion (``yes``), which is what ``delta`` is. ``gate_off``
    adds the with-promotion, gate-off column (``gate_off``: a mean, no CI); a
    row it cannot fill is ``None`` there."""
    if not np.array_equal(no["labels"], yes["labels"]):
        raise ValueError("the two arms are not on the same (prompt, layer) traces")
    groups = yes["groups"]
    rows: Dict[str, Dict[str, Any]] = {}
    for j, (key, *_rest) in enumerate(ROWS):
        if key == "lir":
            if "lir_rescued" in no:
                rows[key] = _paired_ratio(no["lir_rescued"], no["lir_eligible"],
                                          yes["lir_rescued"], yes["lir_eligible"],
                                          n_boot, seed + 10 * j)
        elif key == "promotions":
            if key in no:
                rows[key] = {"without": float(no[key].sum()), "with": float(yes[key].sum()),
                             "delta": float(yes[key].sum() - no[key].sum())}
        elif key in no and key in yes:
            rows[key] = _paired(no[key], yes[key], groups, n_boot, seed + 10 * j)
        if gate_off is not None and key in rows:
            off = None
            if key == "lir" and "lir_rescued" in gate_off:
                d = gate_off["lir_eligible"].sum()
                off = float(gate_off["lir_rescued"].sum() / d) if d > 0 else None
            elif key == "promotions" and key in gate_off:
                off = float(gate_off[key].sum())
            elif key in gate_off:
                v = QM.group_reduce(gate_off[key], groups)
                off = float(QM.nanmean(v)) if np.isfinite(v).any() else None
            rows[key]["gate_off"] = off
    return rows


# ---------------------------------------------------------------------------
# promotion-level view: what each Q -> F move carried
# ---------------------------------------------------------------------------


def swap_summary(swap: Sequence[Dict[str, np.ndarray]], n_boot: int = 2000,
                 seed: int = 0) -> Dict[str, Any]:
    """Pooled per-window future mass of promoted vs demoted windows, and a
    paired per-trace ratio (traces with both a promotion and a demotion)."""
    ps = np.concatenate([s["promoted_sum"] for s in swap])
    pn = np.concatenate([s["promoted_n"] for s in swap])
    ds = np.concatenate([s["demoted_sum"] for s in swap])
    dn = np.concatenate([s["demoted_n"] for s in swap])
    ok = (pn > 0) & (dn > 0)
    diff = ps[ok] / pn[ok] - ds[ok] / dn[ok]
    d_mu, d_lo, d_hi = QM.bootstrap_mean_ci(diff, 0.95, n_boot, seed)
    return {"promoted_windows": int(pn.sum()), "demoted_windows": int(dn.sum()),
            "traces_with_swaps": int(ok.sum()), "traces": int(pn.size),
            "promoted_mass": float(ps.sum() / pn.sum()) if pn.sum() else np.nan,
            "demoted_mass": float(ds.sum() / dn.sum()) if dn.sum() else np.nan,
            "per_trace_gain": d_mu, "per_trace_gain_lo": d_lo, "per_trace_gain_hi": d_hi}


def conditional_fp(ours_with: Dict[str, Any], ours_without: Dict[str, Any],
                   base: Dict[str, Any]) -> Dict[str, float]:
    """fp-tier FullKV mass on the (step, layer) cells where the two arms' fp
    sets differ -- where promotion acted -- per head, averaged over heads."""
    ids_a = ours_with["arrays"]["all_window_ids"]
    tg_a = ours_with["arrays"]["all_window_tier"]
    ids_b = ours_without["arrays"]["all_window_ids"]
    tg_b = ours_without["arrays"]["all_window_tier"]
    fm = base["arrays"]["step_window_scores_heads"]                   # [S,T,L,H,W]
    S, T, L, H, W = fm.shape
    tot_a = tot_b = 0.0
    cells = allc = 0
    for s in range(S):
        for li in range(L):
            fa = np.zeros((T, W), bool)
            fb = np.zeros((T, W), bool)
            for ids, tg, dst in ((ids_a, tg_a, fa), (ids_b, tg_b, fb)):
                t_i, o_i = np.nonzero((ids[s, :, li] >= 0) & (tg[s, :, li] == 0))
                dst[t_i, ids[s, t_i, li, o_i]] = True
            diff = (fa != fb).any(-1)
            diff[0] = False
            allc += T - 1
            if not diff.any():
                continue
            m = fm[s, diff, li].astype(np.float32).mean(1)            # [n, W]
            tot_a += float((m * fa[diff]).sum())
            tot_b += float((m * fb[diff]).sum())
            cells += int(diff.sum())
    return {"cells": cells, "cells_total": allc,
            "fp_mass_with": tot_a / cells if cells else np.nan,
            "fp_mass_without": tot_b / cells if cells else np.nan}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _fmt(v: Any, unit: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if unit == "pct":
        return f"{100 * v:.2f}%"
    if unit == "count":
        return f"{v:,.0f}"
    return f"{v:.4f}"


def _fmt_delta(r: Dict[str, Any], unit: str) -> str:
    d = r.get("delta")
    if d is None or not np.isfinite(d):
        return "--"
    if unit == "pct":
        s = f"{100 * d:+.3f} pp"
        if "delta_lo" in r and np.isfinite(r["delta_lo"]):
            s += f" [{100 * r['delta_lo']:+.3f}, {100 * r['delta_hi']:+.3f}]"
        return s
    if unit == "count":
        return f"{d:+,.0f}"
    s = f"{d:+.4f}"
    if "delta_lo" in r and np.isfinite(r["delta_lo"]):
        s += f" [{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]"
    return s


def render_table(rows: Dict[str, Dict[str, Any]], head_no: str, head_yes: str,
                 head_off: Optional[str] = None) -> List[str]:
    arrow = {"down": " ↓", "up": " ↑", "": ""}
    cols = [head_no] + ([head_off] if head_off else []) + [head_yes]
    out = [f"| Metric | {' | '.join(cols)} | Δ (gated promotion − no promotion) [95% CI] |",
           "|" + "---|" * (len(cols) + 2)]
    for key, label, better, unit in ROWS:
        r = rows.get(key)
        if r is None:
            continue
        vals = [_fmt(r["without"], unit)]
        if head_off:
            vals.append(_fmt(r.get("gate_off"), unit))
        vals.append(_fmt(r["with"], unit))
        out.append(f"| {label}{arrow[better]} | {' | '.join(vals)} | {_fmt_delta(r, unit)} |")
    return out


def render(result: Dict[str, Any]) -> str:
    g, c = result["geometry"], result["checks"]
    measured = result["no_promotion_source"] == "measured"
    head_no = "No promotion (measured)" if measured else "No promotion (one-way replay)"
    off_measured = result.get("gate_off_source") == "measured"
    L: List[str] = [
        "# Q -> F promotion, in attention mass",
        "",
        f"Run: `{result['zip']}` -- {g['samples']} prompts x {g['layers']} layers x "
        f"{g['steps']} decode steps; {g['k_fp']} fp + {g['n_q']} int2 windows of "
        f"{g['window_size']} tokens, gate opens {g['n_sel']} of {g['n_q']} per KV head. "
        f"Read-gate verdict: `{g['read_gate']}`.",
        "",
        "Mass is a share of each query head's FullKV attention per decode step "
        "(sinks included) unless a row says otherwise. FMM horizon "
        f"{result['horizon']} steps. Intervals: 95% percentile bootstrap over "
        f"{g['traces']} (prompt, layer) traces, paired.",
        "",
        "## Table 18, on this run",
        "",
        *render_table(result["table"], head_no, "With gated promotion (measured)",
                      "With promotion (gate off, measured)" if off_measured
                      else "With promotion (gate off)"),
        "",
        ("With promotion (gate off): a `--gate-ratio 1.0` collector run on the same "
         "prompts, on its own decode trajectory." if off_measured else
         "With promotion (gate off): the measured run's tiers and scores with every int2 "
         "window read. Rows the gate cannot move under the replay carry the measured "
         "values; R_Q is `--` because the 2-bit read of the windows the gate skipped was "
         "never recorded (pass a `--gate-ratio 1.0` collector run as `--gate-off-zip`)."),
        "",
    ]
    if not measured:
        L += ["No-promotion gate: `open` (the gate never misses a window promotion would "
              "have held in fp; favours no promotion). The `closed` bracket (never opens "
              "it; favours promotion; R_Q `--`, those windows have no recorded card):", "",
              *render_table(result["table_closed"], "No promotion (closed gate)",
                            "With gated promotion (measured)"), ""]
    sw = result["swap"]
    cf = result["conditional_fp"]
    L += [
        "## What each promotion carried",
        "",
        f"Next {result['horizon']} steps, per window, share of a head's attention per "
        "step (measured arm):",
        "",
        "| window | count | mass per step |",
        "|---|---|---|",
        f"| promoted Q -> F | {sw['promoted_windows']:,} | {_fmt(sw['promoted_mass'], 'pct')} |",
        f"| demoted F -> Q at the same evictions | {sw['demoted_windows']:,} | "
        f"{_fmt(sw['demoted_mass'], 'pct')} |",
        "",
        f"Per-trace gain (promoted − demoted, {sw['traces_with_swaps']} of "
        f"{sw['traces']} traces with swaps): {_fmt(sw['per_trace_gain'], 'pct')} "
        f"[{_fmt(sw['per_trace_gain_lo'], 'pct')}, {_fmt(sw['per_trace_gain_hi'], 'pct')}].",
        "",
        f"Where promotion acted ({cf['cells']:,} of {cf['cells_total']:,} (step, layer) "
        f"cells, {100 * cf['cells'] / max(cf['cells_total'], 1):.1f}%, the two fp sets "
        f"differ): fp-tier mass {_fmt(cf['fp_mass_with'], 'pct')} with promotion vs "
        f"{_fmt(cf['fp_mass_without'], 'pct')} without.",
        "",
        "## Checks",
        "",
        "| check | value |",
        "|---|---|",
    ]
    if not measured:
        L += [
            f"| bidirectional replay reproduces the recorded tiers | "
            f"{c['bidir_replay_events'] - c['bidir_replay_mismatch']:,} / "
            f"{c['bidir_replay_events']:,} layer-evictions |",
            f"| gate rule reproduces the recorded picks | {c['gate_rows_reproduced']:,} / "
            f"{c['gate_rows']:,} (step, layer, KV head) |",
            f"| one-way: kept set (fp + int2) differs from the run | "
            f"{c['oneway_kept_differs']:,} / {c['oneway_events']:,} layer-evictions |",
            f"| one-way: fp set differs from the run | {c['oneway_fp_differs']:,} / "
            f"{c['oneway_events']:,} layer-evictions |",
            f"| one-way: windows ranked without a score (unscored) | {c['oneway_unscored']:,} |",
            f"| one-way: promotions | {c['oneway_promotions']:,} |",
            f"| gate re-run rows / int2 windows without a card | {c['gate_rerun_rows']:,} / "
            f"{c['gate_blocked_windows']:,} |",
            f"| rows with more of them than the gate reads (`open`: best by FullKV mass) | "
            f"{c['gate_over_budget_rows']:,} |",
            f"| R_Q: rows re-credited / replay-read windows whose recorded credit was a fill | "
            f"{c['rq_recomputed_rows']:,} / {c['rq_read_without_record']:,} |",
            f"| R_Q: head-steps whose fill was calibrated on the run's read set (no carded "
            f"window read) | {c['rq_fill_from_run']:,} |",
            f"| R_Q: head-steps left out of both arms (an int2 window with neither a read "
            f"nor a card in the replay) | {c['rq_dropped_head_steps']:,} of "
            f"{c['rq_head_steps']:,} |",
        ]
    if c.get("fill_windows"):
        L += [f"| card fill formula reproduces the recorded fills (skipped windows, credit "
              f">= 1e-4): within 1% / 5% | {c['fill_within_1pct']:,} / "
              f"{c['fill_within_5pct']:,} of {c['fill_windows']:,} |"]
    L += [f"| read-gate verdict | `{g['read_gate']}` |", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _rq(parts: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Per-trace R_Q from per-sample ``([L] num, [L] den)``, sample-major like
    the suite's trace axis."""
    num = np.concatenate([n for n, _ in parts])
    den = np.concatenate([d for _, d in parts])
    return np.divide(num, den, out=np.full_like(den, np.nan), where=den > 0)


def run(zip_path: str | Path, out_dir: str | Path, oneway_zip: Optional[str] = None,
        horizon: int = 32, lir_m: int = 4, n_boot: int = 2000,
        limit: Optional[int] = None, gate_off_zip: Optional[str] = None
        ) -> Dict[str, Any]:
    zip_path = str(zip_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = replay_run(zip_path, horizon, limit)
    meta = rep.meta
    base = parity_base(meta, _samples(zip_path, BASE_KEYS, limit), zip_path)
    ours_with = parity_ours(meta, rep.arms["with"], zip_path)
    yes = arm_traces(base, ours_with, horizon, lir_m)
    yes["qsa"] = np.concatenate(rep.qsa["with"])
    yes["rq"] = _rq(rep.rq["with"])
    if gate_off_zip:
        meta_g, samples_g, qsa_g, rq_g = measured_arm(gate_off_zip, limit)
        cfg_g = meta_g["config"]
        if cfg_g.get("quant_promotion", "bidir") != "bidir" or \
                float(cfg_g.get("gate_ratio", 0.0)) != 1.0:
            raise ValueError(f"{gate_off_zip} is not a bidirectional --gate-ratio 1.0 run")
        base_g = parity_base(meta_g, _samples(gate_off_zip, BASE_KEYS, limit), gate_off_zip)
        gate_off = arm_traces(base_g, parity_ours(meta_g, samples_g, gate_off_zip),
                              horizon, lir_m)
        gate_off["qsa"] = np.concatenate(qsa_g)
        gate_off["rq"] = _rq(rq_g)
        del base_g
    else:
        ours_off = parity_ours(meta, rep.arms["gate_off"], zip_path, quant_gate_ratio=1.0,
                               gate_bracket="every int2 window read (replay)")
        gate_off = arm_traces(base, ours_off, horizon, lir_m, ledger_only=True)
        gate_off.update({k: yes[k] for k in GATE_FREE if k in yes})
    if oneway_zip:
        meta_o, samples_o, qsa_o, rq_o = measured_arm(oneway_zip, limit)
        if meta_o["config"].get("quant_promotion") != "oneway":
            raise ValueError(f"{oneway_zip} is not a one-way run")
        base_o = parity_base(meta_o, _samples(oneway_zip, BASE_KEYS, limit), oneway_zip)
        ours_no = parity_ours(meta_o, samples_o, oneway_zip)
        no = arm_traces(base_o, ours_no, horizon, lir_m)
        no["qsa"] = np.concatenate(qsa_o)
        no["rq"] = _rq(rq_o)
        table_closed = None
        source = "measured"
    else:
        ours_no = parity_ours(meta, rep.arms["without_open"], zip_path,
                              quant_promotion="oneway-replay", gate_bracket="open")
        no = arm_traces(base, ours_no, horizon, lir_m)
        no["qsa"] = np.concatenate(rep.qsa["without"])
        no["rq"] = _rq(rep.rq["without"])
        ours_closed = parity_ours(meta, rep.arms["without_closed"], zip_path,
                                  quant_promotion="oneway-replay", gate_bracket="closed")
        closed = arm_traces(base, ours_closed, horizon, lir_m, ledger_only=True)
        table_closed = build_table(closed, yes, n_boot)
        source = "replay"
    table = build_table(no, yes, n_boot, gate_off=gate_off)
    ax, g = meta["axes"], meta["resolved_geometry"]
    n_sel = int(np.max([_n_sel(s["gate_open"])[1:].max() for s in rep.arms["with"]]))
    result = {
        "zip": Path(zip_path).name,
        "oneway_zip": Path(oneway_zip).name if oneway_zip else None,
        "gate_off_zip": Path(gate_off_zip).name if gate_off_zip else None,
        "no_promotion_source": source,
        "gate_off_source": "measured" if gate_off_zip else "replay",
        "horizon": horizon, "lir_m": lir_m,
        "geometry": {"samples": len(rep.arms["with"]), "layers": meta["model"]["num_layers"],
                     "steps": ax["gen"], "window_size": ax["window_size"],
                     "k_fp": int(g["top_k_fp"]), "n_q": int(g["N_q"]), "n_sel": n_sel,
                     "traces": int(yes["labels"].shape[0]),
                     "quant_ratio": meta["config"]["quant_ratio"],
                     "cache_budget": meta["config"]["cache_budget"],
                     "gate_ratio": meta["config"]["gate_ratio"],
                     "read_gate": (meta.get("read_gate") or {}).get("verdict")},
        "checks": rep.checks,
        "table": table,
        "table_closed": table_closed,
        "swap": swap_summary(rep.swap, n_boot),
        "conditional_fp": conditional_fp(ours_with, ours_no, base),
    }
    (out / "promotion_ablation.json").write_text(
        json.dumps(QO._json_safe(result), indent=1))
    (out / "promotion_ablation.md").write_text(render(result))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", required=True, help="bidirectional collector run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--oneway-zip", default=None,
                    help="a collector run with --promotion oneway: the measured arm")
    ap.add_argument("--gate-off-zip", default=None,
                    help="a bidirectional collector run with --gate-ratio 1.0: the "
                         "measured with-promotion (gate off) column, R_Q included")
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lir-m", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=None,
                    help="only the first N prompts (a quick look)")
    a = ap.parse_args(argv)
    res = run(a.zip, a.out_dir, a.oneway_zip, a.horizon, a.lir_m, a.n_boot, a.samples,
              a.gate_off_zip)
    print(render(res))


if __name__ == "__main__":
    main()
