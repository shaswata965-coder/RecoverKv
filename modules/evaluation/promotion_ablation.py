"""What Q -> F promotion buys, in attention mass: the paper's Table 18 on one run.

Table 18 compares the three-tier cache with and without Q -> F promotion on two
pilot runs over *different* source articles, and says so -- directional
evidence, not a paired estimate. One observation-collector run of the shipped
policy (``quant_promotion="bidir"``) pairs it exactly:

* **with promotion** is the run as measured: its tiers, its read gate's picks;
* **without promotion** is the one-way policy (``quant_promotion="oneway"``,
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

**Every quantity is attention mass.** Future Missed Mass (FMM), the decode read
ledger (where each query head's FullKV attention landed: fp, int2 opened by the
gate, int2 reached only through the card fill, evicted), the mass a promoted
window carries against the fp window demoted for it, and -- beside them, as
Table 18 has them -- score agreement on the int2 tier, Global LIR and the
Q -> F transition count.

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
measured, paired per ``(prompt, layer)`` but on its own decode trajectory.

CLI::

    python -m modules.evaluation.promotion_ablation \\
        --zip outputs/obs_data/<run>.zip --out-dir outputs/promotion_ablation \\
        [--oneway-zip outputs/obs_data/<run>_oneway.zip] [--horizon 32]
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
    BASE_KEYS, OURS_KEYS, TIER_EVICTED, TIER_FP, TIER_Q, iter_zip_samples,
    parity_base, parity_ours)
from utils import qevict_metrics as QM
from utils.logger import get_logger

log = get_logger(__name__)

#: What one pass over a collector zip reads per sample.
REPLAY_KEYS = ("full_mass", "tier", "gate_open", "gate_fired", "evict_step",
               "tokens", "rank_signal", "rank_signal_steps", "card_logmass")
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
              n_sel: np.ndarray, n_kv: int) -> np.ndarray:
    """The read gate's selection rule, batched over ``N`` (step) rows.

    ``card`` ``[N, H, W]`` the card log-mass estimates; ``known`` ``[N, W]`` the
    int2 windows that have one; ``forced`` ``[N, W]`` (every KV head) or
    ``[N, n_kv, W]`` windows opened regardless -- they take their slots first;
    ``n_sel`` ``[N]`` windows read per KV head. Returns ``[N, n_kv, W]`` bool.

    ``sketch.group_share`` then ``select_windows``: each query head's log share
    of its own int2 mass (``logmass - logsumexp`` over the windows it has
    estimates for), the max over the query heads sharing a KV head, the top
    ``n_sel`` per KV head. Never raw logits across heads (CLAUDE.md).
    """
    N, H, W = card.shape
    rep = H // n_kv
    x = np.where(known[:, None, :], card.astype(np.float64), -np.inf)
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
    int2 windows both hold, at the recorded ``n_sel``; windows only the replay
    holds in int2 (the run had promoted them, so no card was read) are opened
    for every KV head (``blocked="open"``) or never (``"closed"``).

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
        out[rows, li] = gate_pick(card_logmass[rows, li], known, forced, ns, Hk)
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
# one pass over the zip
# ---------------------------------------------------------------------------


@dataclass
class Replay:
    """Everything one pass over a bidirectional run's zip produces."""
    meta: Dict[str, Any]
    arms: Dict[str, List[Dict[str, np.ndarray]]]   # arm -> per-sample OURS_KEYS
    qsa: Dict[str, List[np.ndarray]]                # arm -> per-sample [L]
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
    """Measured arm + both gate brackets of the one-way replay, one sample at a
    time. The ground truth (``full_mass``) is read again by :func:`run` through
    ``parity_base``; nothing here holds more than one sample of it."""
    meta = _read_meta(zip_path)
    promo = meta["config"].get("quant_promotion", "bidir")
    if promo != "bidir":
        raise ValueError(f"{zip_path} is a {promo!r} run: the replay needs the "
                         "bidirectional run, and replays the one-way arm on it")
    k_fp, n_q, local_w = _geometry(meta)
    arms: Dict[str, List[Dict[str, np.ndarray]]] = {
        "with": [], **{f"without_{b}": [] for b in GATE_BRACKET}}
    qsa: Dict[str, List[np.ndarray]] = {"with": [], "without": []}
    swap: List[Dict[str, np.ndarray]] = []
    checks = {"bidir_replay_events": 0, "bidir_replay_mismatch": 0,
              "gate_rows": 0, "gate_rows_reproduced": 0,
              "oneway_events": 0, "oneway_unscored": 0, "oneway_promotions": 0,
              "oneway_kept_differs": 0, "oneway_fp_differs": 0,
              "gate_rerun_rows": 0, "gate_blocked_windows": 0,
              "gate_over_budget_rows": 0}
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
        one, d_o = replay_tiers(tier, ev_mask, rs, rsteps, k_fp, n_q, local_w,
                                oneway=True)
        for k in ("events", "unscored", "promotions", "kept_differs", "fp_differs"):
            checks[f"oneway_{k}"] += d_o[k]
        common = {"gate_fired": a["gate_fired"], "evict_step": ev_mask,
                  "tokens": a["tokens"]}
        arms["with"].append({"tier": tier, "gate_open": a["gate_open"], **common})
        for b in GATE_BRACKET:
            g, d_g = replay_gate(tier, one, a["gate_open"], a["gate_fired"],
                                 a["card_logmass"], b, a["full_mass"])
            if b == "open":
                checks["gate_rerun_rows"] += d_g["rerun_rows"]
                checks["gate_blocked_windows"] += d_g["blocked_windows"]
                checks["gate_over_budget_rows"] += d_g["over_budget_rows"]
            arms[f"without_{b}"].append({"tier": one, "gate_open": g, **common})
        qsa["with"].append(score_agreement(tier, ev_mask, rs, a["full_mass"]))
        qsa["without"].append(score_agreement(one, ev_mask, rs, a["full_mass"]))
        swap.append(swap_mass(tier, ev_mask, a["full_mass"], horizon))
        log.info("sample %d: bidir replay mismatches %d, one-way fp differs on %d "
                 "of %d layer-evictions", i, checks["bidir_replay_mismatch"],
                 d_o["fp_differs"], d_o["events"])
        del a
    return Replay(meta=meta, arms=arms, qsa=qsa, swap=swap, checks=checks)


def measured_arm(zip_path: str | Path, limit: Optional[int] = None
                 ) -> Tuple[Dict[str, Any], List[Dict[str, np.ndarray]], List[np.ndarray]]:
    """A recorded run's own tiers and gate (for ``--oneway-zip``) and its
    quantized-score agreement."""
    meta = _read_meta(zip_path)
    samples, qsa = [], []
    for a in _samples(zip_path, REPLAY_KEYS, limit):
        samples.append({k: a[k] for k in OURS_KEYS})
        qsa.append(score_agreement(a["tier"], a["evict_step"], a["rank_signal"],
                                   a["full_mass"]))
    return meta, samples, qsa


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
ROWS: Tuple[Tuple[str, str, str, str], ...] = (
    ("fmm_kept", "FMM, fp + int2 (Table 18: R3 FMM)", "down", "pct"),
    ("fmm_exact", "FMM, tiers read every step (fp + local)", "down", "pct"),
    ("fp", "FullKV mass on the fp tier, per step", "up", "pct"),
    ("fp_window_share", "  same, share of window mass (sinks excluded)", "up", "pct"),
    ("read", "FullKV mass the decode step reads (sinks + local + fp + opened int2)",
     "up", "pct"),
    ("q_skipped", "FullKV mass reached only through the card fill", "down", "pct"),
    ("held_read", "R_F+Q: held-tier mass the decode reads, (fp + opened) / (fp + int2)",
     "up", "pct"),
    ("q_recall", "R_Q: int2-tier mass in gate-opened windows (per-head recall)", "up", "pct"),
    ("qsa", "Quantized-score agreement (Spearman, int2 tier)", "up", "num"),
    ("lir", "Global LIR, fp tier (m = 4, uncapped)", "up", "pct"),
    ("promotions", "Recorded Q -> F transitions (lag 1)", "", "count"),
)


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
                n_boot: int = 2000, seed: int = 0) -> Dict[str, Dict[str, Any]]:
    """Every row of :data:`ROWS`, paired per trace (no promotion vs with)."""
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


def render_table(rows: Dict[str, Dict[str, Any]], head_no: str, head_yes: str) -> List[str]:
    arrow = {"down": " ↓", "up": " ↑", "": ""}
    out = [f"| metric | {head_no} | {head_yes} | Δ (with − without) [95% CI] |",
           "|---|---|---|---|"]
    for key, label, better, unit in ROWS:
        r = rows.get(key)
        if r is None:
            continue
        out.append(f"| {label}{arrow[better]} | {_fmt(r['without'], unit)} | "
                   f"{_fmt(r['with'], unit)} | {_fmt_delta(r, unit)} |")
    return out


def render(result: Dict[str, Any]) -> str:
    g, c = result["geometry"], result["checks"]
    measured = result["no_promotion_source"] == "measured"
    head_no = "no promotion (measured)" if measured else "no promotion (one-way replay)"
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
        *render_table(result["table"], head_no, "with promotion (measured)"),
        "",
    ]
    if not measured:
        L += ["No-promotion gate: `open` (the gate never misses a window promotion would "
              "have held in fp; favours no promotion). The `closed` bracket (never opens "
              "it; favours promotion):", "",
              *render_table(result["table_closed"], "no promotion (closed gate)",
                            "with promotion (measured)"), ""]
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
        ]
    L += [f"| read-gate verdict | `{g['read_gate']}` |", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(zip_path: str | Path, out_dir: str | Path, oneway_zip: Optional[str] = None,
        horizon: int = 32, lir_m: int = 4, n_boot: int = 2000,
        limit: Optional[int] = None) -> Dict[str, Any]:
    zip_path = str(zip_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = replay_run(zip_path, horizon, limit)
    meta = rep.meta
    base = parity_base(meta, _samples(zip_path, BASE_KEYS, limit), zip_path)
    ours_with = parity_ours(meta, rep.arms["with"], zip_path)
    yes = arm_traces(base, ours_with, horizon, lir_m)
    yes["qsa"] = np.concatenate(rep.qsa["with"])
    if oneway_zip:
        meta_o, samples_o, qsa_o = measured_arm(oneway_zip, limit)
        if meta_o["config"].get("quant_promotion") != "oneway":
            raise ValueError(f"{oneway_zip} is not a one-way run")
        base_o = parity_base(meta_o, _samples(oneway_zip, BASE_KEYS, limit), oneway_zip)
        ours_no = parity_ours(meta_o, samples_o, oneway_zip)
        no = arm_traces(base_o, ours_no, horizon, lir_m)
        no["qsa"] = np.concatenate(qsa_o)
        table_closed = None
        source = "measured"
    else:
        ours_no = parity_ours(meta, rep.arms["without_open"], zip_path,
                              quant_promotion="oneway-replay", gate_bracket="open")
        no = arm_traces(base, ours_no, horizon, lir_m)
        no["qsa"] = np.concatenate(rep.qsa["without"])
        ours_closed = parity_ours(meta, rep.arms["without_closed"], zip_path,
                                  quant_promotion="oneway-replay", gate_bracket="closed")
        closed = arm_traces(base, ours_closed, horizon, lir_m, ledger_only=True)
        table_closed = build_table(closed, yes, n_boot)
        source = "replay"
    table = build_table(no, yes, n_boot)
    ax, g = meta["axes"], meta["resolved_geometry"]
    n_sel = int(np.max([_n_sel(s["gate_open"])[1:].max() for s in rep.arms["with"]]))
    result = {
        "zip": Path(zip_path).name,
        "oneway_zip": Path(oneway_zip).name if oneway_zip else None,
        "no_promotion_source": source,
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
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lir-m", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=None,
                    help="only the first N prompts (a quick look)")
    a = ap.parse_args(argv)
    res = run(a.zip, a.out_dir, a.oneway_zip, a.horizon, a.lir_m, a.n_boot, a.samples)
    print(render(res))


if __name__ == "__main__":
    main()
