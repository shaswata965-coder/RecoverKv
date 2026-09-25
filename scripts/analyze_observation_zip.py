"""Per-metric tables from one observation-collector zip. numpy only.

    python scripts/analyze_observation_zip.py outputs/obs_data/<run>.zip \\
        --out reports/<run>.md [--json reports/<run>.json] [--samples N]

Reads the zip ``modules/evaluation/observation_collector.py`` writes (one
``sample_NNN.npz`` per prompt, ``meta.json``, ``SCHEMA.md``) and prints one
table per observation, per layer band, pooled over every decode step, head and
sample. Nothing is re-run: every number is a reduction of arrays in the zip.

Three kinds of number, and the tables say which:

* **measured** -- read straight off the cache run (tiers, gate picks, card
  estimates, the cache's own window mass, the output-error ladder);
* **oracle** -- the same run's full-KV attention (``full_mass``) replayed
  through a policy that sees exact mass: an H2O-style cumulative ranker at the
  cache's own cadence and tier sizes, a per-step best-set, recency, and the
  fp-only split at the same bytes. These bound what a better *ranking* could
  buy at this budget; the trajectory (the queries) is still the cache's;
* **derived** -- ratios of the two.

Mass is a fraction of the query's full-KV attention (sinks included), so the
tier columns of a row sum to 1.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

TIER_NONE, TIER_FP, TIER_Q, TIER_LOCAL, TIER_EVICTED = -1, 0, 1, 2, 3
LADDER = ("kept_original", "kept_int2_dequant", "held", "actual")
STEPS = ("eviction", "int2_quantization", "promotion_payload", "read_gate")
MOVES = {"int2->fp (promote)": (1, 0), "fp->int2 (demote)": (0, 1),
         "int2->evicted": (1, 3), "fp->evicted": (0, 3),
         "local->fp": (2, 0), "local->int2": (2, 1), "local->evicted": (2, 3)}
GRAN_WS = (4, 8, 16, 32)
BIG = 1e-3          # a window "carries mass" above this (fill error is noise below)
POOL_CAP = 40_000   # window-level values kept per layer per sample
f32 = np.float32


def bands(L: int) -> List[Tuple[str, int, int]]:
    q = L // 4
    cuts = [(0, 2), (2, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, L)]
    return [(f"L{a}-{b - 1}" if b - a > 1 else f"L{a}", a, b) for a, b in cuts]


# ---------------------------------------------------------------------------
# oracle policies
# ---------------------------------------------------------------------------


def sim_evicted(rank_at, W_t: np.ndarray, ev_steps: np.ndarray, n_keep: int,
                local_w: int, W: int) -> np.ndarray:
    """``[T, W]`` evicted mask of a policy that, at every eviction step, ranks
    the evictable band (not evicted, older than the last ``local_w`` existing
    windows) by ``rank_at(t)`` and keeps ``n_keep``; the rest go for good --
    the cache's own constraint (an evicted window never comes back)."""
    T = len(W_t)
    out = np.zeros((T, W), bool)
    state = np.zeros(W, bool)
    evs = set(int(e) for e in ev_steps)
    for t in range(1, T):
        if t in evs:
            hi = max(int(W_t[t]) - local_w, 0)
            cand = np.flatnonzero(~state[:hi])
            if cand.size > n_keep:
                order = cand[np.argsort(-rank_at(t)[cand], kind="stable")]
                state[order[n_keep:]] = True
        out[t] = state
    return out


def instant_evicted(score: np.ndarray, W_t: np.ndarray, n_keep: int,
                    local_w: int) -> np.ndarray:
    """Per-step best set: every step keeps the ``n_keep`` evictable windows
    with the most mass THIS step (no permanence -- a ceiling, not a policy)."""
    n, W = score.shape
    cand = np.arange(W)[None, :] < (W_t[:, None] - local_w)
    s = np.where(cand, score, -np.inf)
    kept = np.zeros_like(cand)
    if n_keep < W:
        idx = np.argpartition(-s, n_keep - 1, axis=1)[:, :n_keep]
        np.put_along_axis(kept, idx, True, 1)
    else:
        kept[:] = True
    return cand & ~kept


def topk_mask(x: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-x, k - 1, axis=-1)[..., :k]
    m = np.zeros(x.shape, bool)
    np.put_along_axis(m, idx, True, -1)
    return m


def ranks(x: np.ndarray) -> np.ndarray:
    return x.argsort(-1).argsort(-1).astype(f32)


def pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(-1) / den


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------


def analyze_sample(a: Dict[str, np.ndarray], geom: Dict, ax: Dict,
                   rng: np.random.Generator) -> Dict[str, np.ndarray]:
    fm_all = a["full_mass"]
    T, L, H, W = fm_all.shape
    tier = a["tier"]
    Hk = a["gate_open"].shape[2]
    rep = H // Hk
    n = T - 1
    ws, ns, P = ax["window_size"], ax["num_sink_tokens"], ax["prefill"]
    local_w = int(geom["local_windows"])
    k_fp, k_q = int(geom["top_k_fp"]), int(geom["N_q"])
    k_fp8 = int(geom["top_k_windows"])
    ev = np.flatnonzero(a["evict_step"])
    W_t = (tier[:, 0] != TIER_NONE).sum(-1)
    g0 = (P - ns) // ws                  # first window holding generated tokens
    gen = np.arange(W) >= g0
    # queries that could see window w by forward t-1 (the prompt forward counts
    # every prompt row at or after the window's first token)
    first_tok = ns + ws * np.arange(W)
    rows_seen = np.maximum(P - first_tok, 0)

    def exposure(t):
        first_step = np.maximum(first_tok - P + 1, 1)
        return rows_seen + np.maximum(t - first_step, 0)

    R: Dict[str, np.ndarray] = {}

    def put(name, shape, fill=np.nan):
        R[name] = np.full(shape, fill, f32)
        return R[name]

    ph = (n, L, H)
    for k in ("sink", "m_local", "m_fp", "m_int2", "m_evict", "miss_h2o",
              "miss_inst", "miss_recent", "miss_h2o_fp8", "miss_agenorm",
              "m_evict_gen", "body_w90",
              "top1_share", "recall", "recall_oracle", "recall_perhead",
              "unread", "hot_in_pick", "card_spearman", "card_top_overlap",
              "card_hot_in_top", "fill_bias", "read_bias"):
        put(k, ph)
    put("out_err", (n, L, H, 4))
    put("out_step", (n, L, H, 4))
    R["out_err"][:] = a["out_err"][1:]
    R["out_step"][:] = a["out_step_err"][1:]
    R["sink"][:] = a["sink_mass"][1:]
    nev = len(ev)
    for k in ("ev_jacc", "ev_oracle_pct", "ev_regret", "ev_F8_evicted",
              "ev_F8_int2", "ev_F8_fp", "ev_F64_evicted", "ev_F64_int2",
              "ev_F64_fp", "ev_rank_spearman", "ev_F8_promoted",
              "ev_F8_demoted", "first_keep_vs_cum", "first_keep_vs_future",
              "first_keep_recency", "ev_gen_frac", "gen_kept_last"):
        put(k, (nev, L))
    for m in MOVES:
        put("mv:" + m, (nev, L))
    put("fp_dequant_frac", (n, L))
    pool_fill, pool_read = [], []

    fired = a["gate_fired"][1:]                                  # [n, L]
    for l in range(L):
        fm = fm_all[:, l].astype(f32)                            # [T, H, W]
        m = fm[1:]
        tl = tier[1:, l]
        # ---- tier mass (measured) -------------------------------------
        for name, code in (("m_local", TIER_LOCAL), ("m_fp", TIER_FP),
                           ("m_int2", TIER_Q), ("m_evict", TIER_EVICTED)):
            R[name][:, l] = np.einsum("thw,tw->th", m, (tl == code).astype(f32))
        R["m_evict_gen"][:, l] = np.einsum("thw,tw->th", m,
                                           ((tl == TIER_EVICTED) & gen).astype(f32))
        fpw = tl == TIER_FP
        with np.errstate(invalid="ignore"):
            R["fp_dequant_frac"][:, l] = (a["fp_dequant"][1:, l] & fpw).sum(-1) / fpw.sum(-1)
        # ---- attention shape (ground truth) ---------------------------
        body = m.sum(-1, keepdims=True)
        srt = -np.sort(-m, -1) / np.maximum(body, 1e-12)
        R["body_w90"][:, l] = (np.cumsum(srt, -1) < 0.9).sum(-1) + 1
        R["top1_share"][:, l] = srt[..., 0]
        # ---- oracle policies ------------------------------------------
        hm = fm.mean(1)                                          # [T, W]
        cum = np.cumsum(hm, 0)
        ev_h2o = sim_evicted(lambda t: cum[t - 1], W_t, ev, k_fp + k_q, local_w, W)
        ev_rec = sim_evicted(lambda t: np.arange(W, dtype=f32), W_t, ev,
                             k_fp + k_q, local_w, W)
        ev_fp8 = sim_evicted(lambda t: cum[t - 1], W_t, ev, k_fp8, local_w, W)
        ev_ins = instant_evicted(hm[1:], W_t[1:], k_fp + k_q, local_w)
        ev_age = sim_evicted(lambda t: cum[t - 1] / np.maximum(exposure(t), 1), W_t,
                             ev, k_fp + k_q, local_w, W)
        for name, msk in (("miss_h2o", ev_h2o[1:]), ("miss_recent", ev_rec[1:]),
                          ("miss_agenorm", ev_age[1:]),
                          ("miss_h2o_fp8", ev_fp8[1:]), ("miss_inst", ev_ins)):
            R[name][:, l] = np.einsum("thw,tw->th", m, msk.astype(f32))
        # ---- read gate (measured) vs same-count oracle picks -----------
        K = int((tl == TIER_Q).sum(1).max())
        if K:
            I = np.argsort(tl != TIER_Q, axis=1, kind="stable")[:, :K]
            valid = np.take_along_axis(tl == TIER_Q, I, 1)[:, None, :]
            X = np.take_along_axis(m, I[:, None, :], 2) * valid          # [n,H,K]
            op = np.take_along_axis(a["gate_open"][1:, l], I[:, None, :], 2) & valid
            oph = np.repeat(op, rep, 1)
            Mq = X.sum(-1)
            rd = (X * oph).sum(-1)
            ok = (Mq > 0) & fired[:, l][:, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                R["recall"][:, l] = np.where(ok, rd / Mq, np.nan)
                R["unread"][:, l] = np.where(fired[:, l][:, None], Mq - rd, np.nan)
                kk = int(op.sum(-1).max())
                if kk:
                    share = X / Mq[..., None]
                    g = np.where(valid, share, -np.inf).reshape(n, Hk, rep, K).max(2)
                    pick = np.repeat(topk_mask(np.nan_to_num(g, nan=-np.inf), kk), rep, 1)
                    R["recall_oracle"][:, l] = np.where(ok, (X * pick).sum(-1) / Mq, np.nan)
                    Xs = -np.sort(-X, -1)
                    R["recall_perhead"][:, l] = np.where(ok, Xs[..., :kk].sum(-1) / Mq, np.nan)
                    hot = X.argmax(-1)[..., None]
                    R["hot_in_pick"][:, l] = np.where(
                        ok, np.take_along_axis(oph, hot, -1)[..., 0], np.nan)
                    # ---- card ranking (E1) --------------------------------
                    C = np.take_along_axis(a["card_logmass"][1:, l].astype(f32),
                                           I[:, None, :], 2)
                    C = np.where(valid & np.isfinite(C), C, -np.inf)
                    R["card_spearman"][:, l] = np.where(ok, pearson(ranks(C), ranks(X)), np.nan)
                    mc, mx = topk_mask(C, kk), topk_mask(X, kk)
                    R["card_top_overlap"][:, l] = np.where(ok, (mc & mx).sum(-1) / kk, np.nan)
                    R["card_hot_in_top"][:, l] = np.where(
                        ok, np.take_along_axis(mc, hot, -1)[..., 0], np.nan)
                # ---- fill calibration ---------------------------------
                cs = a["cache_step_mass"][1:, l].astype(f32)
                CS = np.take_along_axis(cs, I[:, None, :], 2)
                loc = (tl == TIER_LOCAL)[:, None, :]
                norm = np.nansum(np.where(loc, cs, 0), -1) / np.sum(np.where(loc, m, 0), -1)
                Xn = X * norm[..., None]
                skip, rdw = valid & ~oph, valid & oph
                R["fill_bias"][:, l] = np.log2(np.nansum(np.where(skip, CS, 0), -1)
                                               / np.sum(np.where(skip, Xn, 0), -1))
                R["read_bias"][:, l] = np.log2(np.nansum(np.where(rdw, CS, 0), -1)
                                               / np.sum(np.where(rdw, Xn, 0), -1))
                r = np.log2((CS + 1e-8) / (Xn + 1e-8))
                big = X > BIG
                for pool, sel in ((pool_fill, skip & big), (pool_read, rdw & big)):
                    v = r[sel & np.isfinite(r)]
                    if v.size > POOL_CAP:
                        v = rng.choice(v, POOL_CAP, replace=False)
                    pool.append((l, v.astype(f32)))
        # ---- evictions: decision quality and tier moves ----------------
        pre = np.concatenate([np.zeros((1, W), f32), np.cumsum(hm[1:], 0)])  # pre[t] = sum hm[1..t]
        rs = a["rank_signal"]
        rsteps = list(a["rank_signal_steps"])
        for i, e in enumerate(ev):
            cur = tier[e, l]
            kept = (cur == TIER_FP) | (cur == TIER_Q)
            if i == 0:
                prev = tier[e - 1, l]
                new_ev = (cur == TIER_EVICTED) & (prev != TIER_EVICTED)
                cand = kept | new_ev
                cw = np.flatnonzero(cand)
                nk = int(kept.sum())
                for name, sc in (("first_keep_vs_cum", cum[e - 1]),
                                 ("first_keep_vs_future", pre[T - 1] - pre[e - 1]),
                                 ("first_keep_recency", np.arange(W, dtype=f32))):
                    top = cw[np.argsort(-sc[cw], kind="stable")[:nk]]
                    R[name][i, l] = kept[top].mean()
            else:
                # the state the previous eviction left (between evictions a
                # window leaving the local band is only relabelled, not moved)
                prev = tier[ev[i - 1], l]
                new_ev = (cur == TIER_EVICTED) & (prev != TIER_EVICTED)
                cand = kept | new_ev

                def fut(h):
                    hi = min(e + h - 1, T - 1)
                    return (pre[hi] - pre[e - 1]) / (hi - e + 1)
                F8, F64 = fut(8), fut(64)
                for name, F in (("F8", F8), ("F64", F64)):
                    for grp, msk in (("evicted", new_ev), ("int2", cur == TIER_Q),
                                     ("fp", cur == TIER_FP)):
                        if msk.any():
                            R[f"ev_{name}_{grp}"][i, l] = F[msk].mean()
                pk = (prev == TIER_FP) | (prev == TIER_Q)
                R["ev_jacc"][i, l] = (pk & kept).sum() / max((pk | kept).sum(), 1)
                for mname, (x0, x1) in MOVES.items():
                    R["mv:" + mname][i, l] = ((prev == x0) & (cur == x1)).sum()
                pro = (prev == TIER_Q) & (cur == TIER_FP)
                dem = (prev == TIER_FP) & (cur == TIER_Q)
                if pro.any():
                    R["ev_F8_promoted"][i, l] = F8[pro].mean()
                if dem.any():
                    R["ev_F8_demoted"][i, l] = F8[dem].mean()
                if new_ev.any() and cand.sum() > 1:
                    R["ev_gen_frac"][i, l] = gen[new_ev].mean()
                    # oracle percentile of what the cache evicted: 0 = the
                    # exact cumulative ranker's first choice to evict too
                    cw = np.flatnonzero(cand)
                    pct = ranks(cum[e - 1][cw][None, :])[0] / (cw.size - 1)
                    R["ev_oracle_pct"][i, l] = pct[new_ev[cw]].mean()
                    med = np.median(F64[cur == TIER_Q]) if (cur == TIER_Q).any() else np.inf
                    R["ev_regret"][i, l] = (F64[new_ev] > med).mean()
            if i == len(ev) - 1:
                R["gen_kept_last"][i, l] = (kept & gen).sum()
            if e in rsteps:
                sig = rs[rsteps.index(e), l].mean(0)
                cw = np.flatnonzero(cand & np.isfinite(sig))
                if cw.size > 2:
                    R["ev_rank_spearman"][i, l] = pearson(ranks(sig[cw][None]),
                                                          ranks(cum[e - 1][cw][None]))[0]
    R["_pool_fill"] = pool_fill
    R["_pool_read"] = pool_read

    # ---- int2 reconstruction error -----------------------------------------
    qr, qf = a["q_recon_err"], a["q_first_step"]
    R["_recon"] = [(l, qr[l][:, qf[l] >= 0]) for l in range(L)]   # [Hk, nw, 2]

    # ---- granularity: exact cumulative ranker at other window sizes ------
    tm = a["token_mass"].astype(f32)                             # [T, L, S]
    held_tok = (k_fp + k_q) * ws
    g = np.full((len(GRAN_WS), n, L), np.nan, f32)
    for gi, w2 in enumerate(GRAN_WS):
        if held_tok % w2 or (local_w * ws) % w2:
            continue
        Wg = -(-(tm.shape[2] - ns) // w2)
        post = np.zeros((T, L, Wg * w2), f32)
        post[:, :, :tm.shape[2] - ns] = tm[:, :, ns:]
        wm = post.reshape(T, L, Wg, w2).sum(-1)
        Wt_g = np.minimum(-(-(P + np.arange(T) - ns) // w2), Wg)
        Wt_g[0] = -(-(P - ns) // w2)
        for l in range(L):
            cum = np.cumsum(wm[:, l], 0)
            evm = sim_evicted(lambda t: cum[t - 1], Wt_g, ev, held_tok // w2,
                              local_w * ws // w2, Wg)
            g[gi, :, l] = (wm[1:, l] * evm[1:]).sum(-1)
    R["_gran"] = g

    # ---- tokens ---------------------------------------------------------------
    tok = a["tokens"]
    grams = [tuple(tok[i:i + 4]) for i in range(len(tok) - 3)]
    seen, rep4 = set(), 0
    for gr in grams:
        rep4 += gr in seen
        seen.add(gr)
    R["_tok"] = {"logprob": a["token_logprob"], "entropy": a["token_entropy"],
                 "top1p": np.exp(a["token_top5_logprob"][:, 0]),
                 "rep4": rep4 / max(len(grams), 1)}
    return R


# ---------------------------------------------------------------------------
# pooling and tables
# ---------------------------------------------------------------------------


def load(zip_path: str, limit: int | None):
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f"{zip_path}: member {bad} is corrupt")
        meta = json.loads(zf.read("meta.json"))
        files = [s["file"] for s in meta["samples"]][:limit]
        for f in files:
            with np.load(io.BytesIO(zf.read(f)), allow_pickle=False) as z:
                yield meta, f, {k: z[k] for k in z.files}


def fmt(v, pct=False, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if pct:
        return f"{100 * v:.{max(nd - 2, 1)}f}%"
    return f"{v:.{nd}f}"


def table(rows: List[Tuple[str, List[str]]], cols: List[str]) -> str:
    out = ["| " + " | ".join([""] + cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    out += ["| " + " | ".join([r] + v) + " |" for r, v in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip")
    ap.add_argument("--out", required=True, help="markdown tables")
    ap.add_argument("--json", default=None, help="also write every number here")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    per: List[Dict] = []
    meta = None
    for meta, f, a in load(args.zip, args.samples):
        print(f"analyzing {f} ...", flush=True)
        per.append(analyze_sample(a, meta["resolved_geometry"], meta["axes"], rng))
    assert meta is not None
    L = meta["model"]["num_layers"]
    B = bands(L)
    cols = [b[0] for b in B] + ["all"]
    S = len(per)
    J: Dict = {"meta": {k: meta[k] for k in ("config", "model", "axes",
                                            "resolved_geometry", "read_gate")}}

    def cat(k):
        return np.concatenate([p[k] for p in per], 0)

    def by_band(x, fn=np.nanmean):
        """x: [N, L, ...] -> per band + all."""
        return [float(fn(x[:, a:b])) for _, a, b in B] + [float(fn(x))]

    def q(p):
        return lambda x: np.nanpercentile(x, p)

    md: List[str] = []
    rg = meta["read_gate"]
    g = meta["resolved_geometry"]
    cfg = meta["config"]
    md.append(f"Run: `{Path(args.zip).name}` -- {S} prompt(s) x {meta['axes']['gen']} "
              f"decode steps x {L} layers x {meta['model']['num_heads']} heads; "
              f"read_gate `{rg.get('verdict')}` ({rg.get('gated')}/{rg.get('fired')} "
              f"layers gated, read fraction {rg.get('read_fraction', float('nan')):.3f}).")
    md.append(f"Geometry: {g['top_k_fp']} fp + {g['N_q']} int2 evictable windows, "
              f"{g['local_windows']} local, {cfg['num_sink_tokens']} sinks, ws "
              f"{meta['axes']['window_size']}; fp-only at the same bytes = "
              f"{g['top_k_windows']} windows.")

    # 1. attention shape
    rows = [("sink mass", [fmt(v, 1) for v in by_band(cat("sink"))]),
            ("windows for 90% of body mass (mean)", [fmt(v, nd=1) for v in by_band(cat("body_w90"))]),
            ("top window's share of body mass", [fmt(v, 1) for v in by_band(cat("top1_share"))])]
    md += ["\n## 1. Attention structure (ground truth)\n", table(rows, cols)]
    J["attention"] = {r: v for r, v in zip(["sink", "body_w90", "top1_share"],
                                           [by_band(cat(k)) for k in ("sink", "body_w90", "top1_share")])}

    # 2. tier mass
    names = [("sink", "sink"), ("local (fp)", "m_local"), ("fp (evictable)", "m_fp"),
             ("int2", "m_int2"), ("evicted = missed", "m_evict")]
    rows = [(r, [fmt(v, 1) for v in by_band(cat(k))]) for r, k in names]
    ev_ = cat("m_evict")
    rows += [("missed on windows holding generated tokens", [fmt(v, 1) for v in by_band(cat("m_evict_gen"))]),
             ("missed, p90 over (step, head)", [fmt(v, 1) for v in by_band(ev_, q(90))]),
             ("missed, p99", [fmt(v, 1) for v in by_band(ev_, q(99))]),
             ("share of (step, head) missing >10%", [fmt(v, 1) for v in by_band((ev_ > 0.1).astype(f32))])]
    md += ["\n## 2. Where the attention mass sits (measured)\n", table(rows, cols)]
    J["tier_mass"] = {k: by_band(cat(k)) for _, k in names}

    # 3. oracle
    names = [("cache (measured)", "m_evict"), ("exact cumulative ranker, same tiers", "miss_h2o"),
             ("per-step best set (ceiling)", "miss_inst"), ("recency, same count", "miss_recent"),
             ("exact ranker, fp-only at same bytes", "miss_h2o_fp8"),
             ("exact ranker, age-normalised (mass per query that saw it)", "miss_agenorm")]
    rows = [(r, [fmt(v, 1) for v in by_band(cat(k))]) for r, k in names]
    md += ["\n## 3. Missed mass: the cache against oracle policies\n", table(rows, cols)]
    J["missed"] = {k: by_band(cat(k)) for _, k in names}

    # 4. eviction decisions
    names = [("evicted window's rank under the exact ranker (0 = its first pick too)", "ev_oracle_pct", False),
             ("evicted windows out-drawing the median kept int2 window (next 64 steps)", "ev_regret", True),
             ("rank-signal vs exact cumulative mass, Spearman", "ev_rank_spearman", False),
             ("kept set unchanged across an eviction (Jaccard)", "ev_jacc", False),
             ("evicted windows that hold generated tokens", "ev_gen_frac", True)]
    rows = [(r, [fmt(v, p) for v in by_band(cat(k))]) for r, k, p in names]
    rows.append(("generated-token windows kept (fp+int2) at the last step, of "
                 f"{g['top_k_fp'] + g['N_q']}", [fmt(v, nd=2) for v in by_band(cat("gen_kept_last"))]))
    for grp in ("fp", "int2", "evicted"):
        rows.append((f"mean mass/step, next 8 steps: {grp} windows",
                     [f"{v:.2e}" for v in by_band(cat(f'ev_F8_{grp}'))]))
    for grp in ("fp", "int2", "evicted"):
        rows.append((f"mean mass/step, next 64 steps: {grp} windows",
                     [f"{v:.2e}" for v in by_band(cat(f'ev_F64_{grp}'))]))
    md += ["\n## 4. Eviction decisions (steady state, evictions 2..)\n", table(rows, cols)]
    rows = [(r, [fmt(v, 1) for v in by_band(cat(k))]) for r, k in (
        ("kept set = exact prompt-score top set", "first_keep_vs_cum"),
        ("kept set = the windows the whole decode attends most", "first_keep_vs_future"),
        ("kept set = the most recent windows", "first_keep_recency"))]
    md += ["\n### 4b. The first eviction (prompt compression)\n", table(rows, cols)]
    J["evictions"] = {k: by_band(cat(k)) for _, k, _ in names}

    # 5. tier moves
    rows = [(m_, [fmt(v, nd=2) for v in by_band(cat("mv:" + m_))]) for m_ in MOVES]
    rows += [("mass/step next 8 steps: promoted windows", [f"{v:.2e}" for v in by_band(cat("ev_F8_promoted"))]),
             ("mass/step next 8 steps: demoted windows", [f"{v:.2e}" for v in by_band(cat("ev_F8_demoted"))]),
             ("fp windows holding a dequantized payload", [fmt(v, 1) for v in by_band(cat("fp_dequant_frac"))])]
    md += ["\n## 5. Tier moves per eviction per layer (measured)\n", table(rows, cols)]
    J["moves"] = {m_: by_band(cat("mv:" + m_)) for m_ in MOVES}

    # 6. read gate
    rc = cat("recall")
    uni = math.ceil(cfg["gate_ratio"] * g["N_q"]) / g["N_q"]
    rows = [("int2 tier's share of attention", [fmt(v, 1) for v in by_band(cat("m_int2"))]),
            ("int2 mass the gate read (recall), mean", [fmt(v, 1) for v in by_band(rc)]),
            ("recall, p10 over (step, head)", [fmt(v, 1) for v in by_band(rc, q(10))]),
            ("recall, p1", [fmt(v, 1) for v in by_band(rc, q(1))]),
            ("heads with recall < 50%", [fmt(v, 1) for v in by_band((rc < 0.5).astype(f32))]),
            ("head's hottest int2 window was read", [fmt(v, 1) for v in by_band(cat("hot_in_pick"))]),
            ("oracle pick, same count, group share", [fmt(v, 1) for v in by_band(cat("recall_oracle"))]),
            ("oracle pick per head (no GQA sharing)", [fmt(v, 1) for v in by_band(cat("recall_perhead"))]),
            ("uniform pick (expected)", [fmt(uni, 1)] * len(cols)),
            ("unread int2 mass (of all attention), mean", [fmt(v, 2) for v in by_band(cat("unread"))]),
            ("unread int2 mass, p99", [fmt(v, 1) for v in by_band(cat("unread"), q(99))])]
    md += ["\n## 6. Read gate: what the step actually read (measured)\n", table(rows, cols)]
    J["gate"] = {"recall": by_band(rc), "oracle": by_band(cat("recall_oracle")),
                 "perhead": by_band(cat("recall_perhead")), "hot": by_band(cat("hot_in_pick"))}

    # 7. cards
    rows = [("Spearman(card estimate, true mass) over int2 windows", [fmt(v) for v in by_band(cat("card_spearman"))]),
            ("card top-7 vs true top-7 overlap", [fmt(v, 1) for v in by_band(cat("card_top_overlap"))]),
            ("head's hottest int2 window in the card's top 7", [fmt(v, 1) for v in by_band(cat("card_hot_in_top"))])]
    md += ["\n## 7. Card ranking (E1, measured)\n", table(rows, cols)]
    J["cards"] = {k: by_band(cat(k)) for k in ("card_spearman", "card_top_overlap", "card_hot_in_top")}

    # 8. fill calibration
    def pooled(key, fn):
        vals = {}
        for p in per:
            for l, v in p[key]:
                vals.setdefault(l, []).append(v)
        allv = {l: np.concatenate(v) for l, v in vals.items()}
        out = []
        for _, a_, b_ in B:
            v = np.concatenate([allv[l] for l in range(a_, b_) if l in allv] or [np.zeros(0)])
            out.append(float(fn(v)) if v.size else float("nan"))
        v = np.concatenate(list(allv.values()))
        out.append(float(fn(v)))
        return out
    rows = [("read windows: log2(cache mass / true mass), median", [fmt(v) for v in pooled("_pool_read", np.median)]),
            ("read windows: abs(log2 ratio), median", [fmt(v) for v in pooled("_pool_read", lambda v: np.median(np.abs(v)))]),
            ("skipped windows (fill): log2 ratio, median", [fmt(v) for v in pooled("_pool_fill", np.median)]),
            ("skipped windows (fill): abs(log2 ratio), median", [fmt(v) for v in pooled("_pool_fill", lambda v: np.median(np.abs(v)))]),
            ("skipped windows: fill within 2x of truth", [fmt(v, 1) for v in pooled("_pool_fill", lambda v: np.mean(np.abs(v) < 1))]),
            ("skipped set in total: log2(fill sum / true sum), median", [fmt(v) for v in by_band(cat("fill_bias"), np.nanmedian)]),
            ("read set in total: log2(cache sum / true sum), median", [fmt(v) for v in by_band(cat("read_bias"), np.nanmedian)])]
    md += ["\n## 8. Fill calibration: the cache's own window mass vs truth (int2 windows with >0.1% mass)\n",
           table(rows, cols)]
    J["fill"] = {"fill_abs": pooled("_pool_fill", lambda v: np.median(np.abs(v))),
                 "fill_bias_set": by_band(cat("fill_bias"), np.nanmedian)}

    # 9. output ladder
    oe, os_ = cat("out_err"), cat("out_step")
    rows = []
    for i, nm in enumerate(LADDER):
        rows.append((f"error: {nm}, mean", [fmt(v) for v in by_band(oe[..., i])]))
    rows.append(("error: actual, median", [fmt(v) for v in by_band(oe[..., 3], np.nanmedian)]))
    rows.append(("error: actual, p99", [fmt(v) for v in by_band(oe[..., 3], q(99))]))
    for i, nm in enumerate(STEPS):
        rows.append((f"step: {nm}, mean", [fmt(v) for v in by_band(os_[..., i])]))
    md += ["\n## 9. Attention-output error ladder: rel. L2 vs the full-KV output (measured)\n",
           table(rows, cols)]
    J["ladder"] = {nm: by_band(oe[..., i]) for i, nm in enumerate(LADDER)}
    J["ladder_steps"] = {nm: by_band(os_[..., i]) for i, nm in enumerate(STEPS)}

    # 10. reconstruction
    def recon(kind, fn):
        vals = {}
        for p in per:
            for l, v in p["_recon"]:
                vals.setdefault(l, []).append(v[..., kind].ravel())
        allv = {l: np.concatenate(v) for l, v in vals.items()}
        out = [float(fn(np.concatenate([allv[l] for l in range(a_, b_)]))) for _, a_, b_ in B]
        return out + [float(fn(np.concatenate(list(allv.values()))))]
    rows = [("K rel. error, median", [fmt(v) for v in recon(0, np.median)]),
            ("K rel. error, p90", [fmt(v) for v in recon(0, q(90))]),
            ("V rel. error, median", [fmt(v) for v in recon(1, np.median)]),
            ("V rel. error, p90", [fmt(v) for v in recon(1, q(90))])]
    md += ["\n## 10. int2 reconstruction error per window (measured at demotion)\n", table(rows, cols)]
    J["recon"] = {"K": recon(0, np.median), "V": recon(1, np.median)}

    # 11. granularity
    gr = np.concatenate([p["_gran"] for p in per], 1)                  # [G, N, L]
    rows = [(f"ws {w2}: {(g['top_k_fp'] + g['N_q']) * meta['axes']['window_size'] // w2} windows",
             [fmt(v, 1) for v in by_band(gr[i])]) for i, w2 in enumerate(GRAN_WS)]
    md += ["\n## 11. Granularity: exact cumulative ranker, same held tokens (oracle, head-mean mass)\n",
           table(rows, cols)]
    J["granularity"] = {str(w2): by_band(gr[i]) for i, w2 in enumerate(GRAN_WS)}

    # 12. time
    nstep = per[0]["m_evict"].shape[0]
    edges = list(range(0, nstep + 1, 64))
    tcols = [f"{a_ + 1}-{b_}" for a_, b_ in zip(edges[:-1], edges[1:])]

    def over_t(k, idx=None):
        x = np.stack([p[k] if idx is None else p[k][..., idx] for p in per])  # [S, n, L, H]
        return [float(np.nanmean(x[:, a_:b_])) for a_, b_ in zip(edges[:-1], edges[1:])]
    ent = np.stack([p["_tok"]["entropy"][1:] for p in per])
    rows = [("sink", [fmt(v, 1) for v in over_t("sink")]),
            ("int2", [fmt(v, 1) for v in over_t("m_int2")]),
            ("missed (cache)", [fmt(v, 1) for v in over_t("m_evict")]),
            ("missed (exact ranker)", [fmt(v, 1) for v in over_t("miss_h2o")]),
            ("missed on generated-token windows", [fmt(v, 1) for v in over_t("m_evict_gen")]),
            ("missed (age-normalised ranker)", [fmt(v, 1) for v in over_t("miss_agenorm")]),
            ("gate recall", [fmt(v, 1) for v in over_t("recall")]),
            ("output error, actual", [fmt(v) for v in over_t("out_err", 3)]),
            ("next-token entropy (nats)", [fmt(float(ent[:, a_:b_].mean()), nd=2)
                                           for a_, b_ in zip(edges[:-1], edges[1:])])]
    md += ["\n## 12. Over the decode (decode steps, all layers)\n", table(rows, tcols)]

    # 13. per sample
    rows = []
    for si, p in enumerate(per):
        t = p["_tok"]
        rows.append((f"sample {si}", [fmt(float(np.nanmean(p[k])), 1) for k in
                                       ("sink", "m_int2", "m_evict", "miss_h2o", "recall")]
                     + [fmt(float(np.nanmean(p["out_err"][..., 3])))]
                     + [fmt(float(t["logprob"].mean()), nd=2), fmt(float(t["entropy"].mean()), nd=2),
                        fmt(float((t["top1p"] < 0.5).mean()), 1), fmt(float(t["rep4"]), 1)]))
    md += ["\n## 13. Per prompt\n",
           table(rows, ["sink", "int2", "missed", "missed (exact)", "gate recall", "out err",
                        "token logprob", "entropy", "top-1 p<0.5", "repeated 4-grams"])]

    # appendix: per layer
    lrows = []
    for l in range(L):
        def lm(k, idx=None):
            x = cat(k)[:, l]
            return float(np.nanmean(x if idx is None else x[..., idx]))
        lrows.append((f"L{l}", [fmt(lm("sink"), 1), fmt(lm("m_local"), 1), fmt(lm("m_fp"), 1),
                                fmt(lm("m_int2"), 1), fmt(lm("m_evict"), 1), fmt(lm("miss_h2o"), 1),
                                fmt(lm("miss_inst"), 1), fmt(lm("recall"), 1),
                                fmt(lm("recall_oracle"), 1), fmt(lm("card_spearman")),
                                fmt(lm("out_err", 0)), fmt(lm("out_err", 3))]))
    md += ["\n## Appendix: per layer\n",
           table(lrows, ["sink", "local", "fp", "int2", "missed", "missed (exact)",
                         "missed (per-step best)", "gate recall", "oracle recall",
                         "card Spearman", "err: eviction", "err: actual"])]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"wrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(J, indent=1, default=float))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
