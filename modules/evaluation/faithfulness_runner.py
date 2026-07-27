"""FaithfulnessRunner — pure post-processing over parity npzs (Suite B).

Reads both base and ours npz files.  For every (step, layer) computes five
distribution-comparison metrics between ours' score vector and base's score
vector over the *retained* window set:

    cos_sim      — cosine similarity             ∈ [-1, 1]  (higher = better)
    pearson      — Pearson correlation           ∈ [-1, 1]  (higher = better)
    spearman     — Spearman rank correlation     ∈ [-1, 1]  (higher = better)
    kl_ours_base — KL divergence KL(ours ‖ base) ≥ 0        (lower  = better)
    mass_ratio   — base_mass / ours_mass                    (≈1 = well-matched)

It also runs the Sticky-K policy analytics (utils/sticky_metrics.py) over the
base run's window scores — the ground-truth attention masses — producing:

    global_lir            — Lazy Insertion Rescue rate (scalar, [L], [L, H])
    missed_mass           — Sticky-K absolute missed mass trajectory ([T], [T, L])
    missed_mass_fresh     — Fresh-K baseline missed-mass trajectory ([T])

No model loaded — pure tensor / numpy ops.
"""
from __future__ import annotations
import csv, json, math, time
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from utils.config import ExperimentConfig, ParityValidationError
from utils.hashing import sha256_file
from utils.logger import get_logger
from utils import metrics as M
from utils import sticky_metrics as SM

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two 1-D vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).clamp(-1, 1).squeeze()


def _pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation between two 1-D vectors."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    return (a_c * b_c).sum() / (a_c.norm() * b_c.norm()).clamp(min=eps)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation (Pearson on ranks)."""
    a_rank = a.argsort().argsort().float()
    b_rank = b.argsort().argsort().float()
    return _pearson(a_rank, b_rank)


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(P ‖ Q) from non-negative score vectors P, Q (normalised internally)."""
    p = p.clamp(min=0)
    q = q.clamp(min=0)
    n = p.shape[0]
    p_prob = (p + eps) / (p.sum() + eps * n)
    q_prob = (q + eps) / (q.sum() + eps * n)
    return (p_prob * (p_prob.log() - q_prob.log())).sum().clamp(min=0)


# ---------------------------------------------------------------------------
# Tier-aware Jaccard helpers (Suite A, two-tier).  Set ops over variable-size
# id sets — the vectorised M.jaccard_topk assumes a fixed K, so the two-tier
# comparison (whose fp / kept set sizes vary per (step, layer)) is done here
# with explicit sets.  (utils/metrics.py must stay loop-free; this module does
# not — see faithfulness_runner's role as pure post-processing.)
# ---------------------------------------------------------------------------

def _jaccard_sets(a: set, b: set) -> float:
    """Jaccard of two id sets; 1.0 when both are empty (matches jaccard_topk)."""
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _base_top_ids(base_hm: np.ndarray, ew: int, n: int) -> set:
    """Original ids of base's top-``n`` evictable windows by head-mean mass.

    ``base_hm`` is the base run's per-window head-mean score vector; the
    evictable region is ``[0, ew)`` (the recency tail is excluded, matching the
    ours-side tier tags where local windows are tier 2).  Returns at most ``n``
    ids; fewer if ``ew < n``.
    """
    if n <= 0 or ew <= 0:
        return set()
    ev = base_hm[:ew]
    k = min(n, ev.shape[0])
    # argsort descending, take top-k; ids ARE the evictable window indices.
    top = np.argpartition(ev, -k)[-k:]
    return set(int(i) for i in top)

# ---------------------------------------------------------------------------
# NPZ loader
# ---------------------------------------------------------------------------

def _load_npz(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ not found: {p}")
    data = np.load(str(p), allow_pickle=True)
    meta_str = str(data["metadata_json"][0])
    meta = json.loads(meta_str)
    arrays = {k: data[k] for k in data.files if k != "metadata_json"}
    return {"arrays": arrays, "metadata": meta, "path": str(p)}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class FaithfulnessRunner:
    """Suite B — faithfulness metrics from paired parity npzs."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        fc  = cfg.faithfulness
        log.info("=== Faithfulness Runner ===")
        base = _load_npz(fc.base_npz_path)
        ours = _load_npz(fc.ours_npz_path)
        self._validate_alignment(base["metadata"], ours["metadata"])
        results = self._compute_metrics(base, ours)
        return self._write(results, base, ours, cfg)

    # ------------------------------------------------------------------
    def _validate_alignment(self, bm: dict, om: dict) -> None:
        checks = ["article_sha", "seed", "prefill_len", "gen_len",
                  "window_size", "num_sink_tokens", "model_name"]
        mismatches = []
        for f in checks:
            bv, ov = bm.get(f), om.get(f)
            if bv is not None and ov is not None and bv != ov:
                mismatches.append(f"  {f}: base={bv!r}, ours={ov!r}")
        if mismatches:
            raise ParityValidationError(
                "Faithfulness alignment failed:\n" + "\n".join(mismatches))
        if bm.get("mode") != "parity_base":
            log.warning("Base npz mode is %r, expected 'parity_base'", bm.get("mode"))
        if om.get("mode") != "parity_ours":
            log.warning("Ours npz mode is %r, expected 'parity_ours'", om.get("mode"))

    # ------------------------------------------------------------------
    def _compute_metrics(self, base: dict, ours: dict) -> dict:
        ba, oa = base["arrays"], ours["arrays"]

        # Require new retained-window arrays from ours npz.
        if "retained_window_ids" not in oa or "retained_window_scores" not in oa:
            raise KeyError(
                "Ours npz is missing 'retained_window_ids' / 'retained_window_scores'. "
                "Re-run OursParityRunner to generate an updated npz."
            )

        # ── load tensors ──────────────────────────────────────────────
        # Window scores: legacy [T, L, H, W] → new [S, T, L, H, W]
        base_ws  = torch.from_numpy(ba["window_scores"].astype(np.float32))
        # Top-K indices for Jaccard: legacy [T, L, K] → new [S, T, L, K]
        base_tk  = torch.from_numpy(ba["top_window_indices"].astype(np.int64))
        ours_tk  = torch.from_numpy(oa["top_window_indices"].astype(np.int64))
        # New retained arrays: [S, T, L, M] and [S, T, L, H, M]
        ours_rid = torch.from_numpy(oa["retained_window_ids"].astype(np.int64))
        ours_rsc = torch.from_numpy(oa["retained_window_scores"].astype(np.float32))

        # Two-tier full-survivor arrays (schema >= 1.2).  Absent on legacy npzs →
        # tier-aware metrics fall back to "all fp, no Q" (jaccard_lift == 0,
        # recovered_mass_q == 0), i.e. identical to the single-tier result.
        has_tier = ("all_window_ids" in oa and "all_window_tier" in oa
                    and "window_scores" in oa)
        if has_tier:
            all_ids  = torch.from_numpy(oa["all_window_ids"].astype(np.int64))
            all_tier = torch.from_numpy(oa["all_window_tier"].astype(np.int64))
            ours_ws  = torch.from_numpy(oa["window_scores"].astype(np.float32))
        else:
            all_ids = all_tier = ours_ws = None

        # Normalise to per-sample form
        if base_tk.dim() == 3:
            base_tk  = base_tk.unsqueeze(0)
            ours_tk  = ours_tk.unsqueeze(0)
        if base_ws.dim() == 4:
            base_ws  = base_ws.unsqueeze(0)
        if ours_rid.dim() == 3:
            ours_rid = ours_rid.unsqueeze(0)
            ours_rsc = ours_rsc.unsqueeze(0)
        if has_tier:
            if all_ids.dim() == 3:
                all_ids  = all_ids.unsqueeze(0)
                all_tier = all_tier.unsqueeze(0)
            if ours_ws.dim() == 4:
                ours_ws = ours_ws.unsqueeze(0)

        num_samples = min(base_ws.shape[0], ours_rid.shape[0])

        # Align K for Jaccard (truncate to min side to avoid -1 inflation)
        bK, oK = base_tk.shape[-1], ours_tk.shape[-1]
        if bK != oK:
            minK    = min(bK, oK)
            base_tk = base_tk[..., :minK]
            ours_tk = ours_tk[..., :minK]

        om          = ours["metadata"]
        ws_sz       = int(om.get("window_size", 8))
        ns          = int(om.get("num_sink_tokens", 0))
        prefill_len = int(om.get("prefill_len", 0))
        # Two-tier split (design.md §7).  Defaults reproduce the single-tier run.
        top_k_windows = int(om.get("top_k_windows", 0))
        top_k_fp      = int(om.get("top_k_fp", top_k_windows))
        n_q_meta      = int(om.get("N_q", 0))
        lr_meta       = int(om.get("local_window_size_resolved", 0))
        local_windows = lr_meta // ws_sz if ws_sz > 0 else 0

        # ── per-sample accumulation ───────────────────────────────────
        per_sample_jaccard = []
        per_sample_cos, per_sample_prs, per_sample_spm = [], [], []
        per_sample_kl,  per_sample_mr                  = [], []
        # Tier-aware Jaccard + Q-tier fidelity (two-tier, Suite A / Phase 3)
        per_sample_jfp, per_sample_jkept    = [], []
        per_sample_qfid, per_sample_qfidcnt = [], []

        for s in range(num_samples):
            b_tk_s  = base_tk[s]    # [T, L, K]
            o_tk_s  = ours_tk[s]    # [T, L, K]
            b_ws_s  = base_ws[s]    # [T, L, H, W_pad]
            o_rid_s = ours_rid[s]   # [T, L, M]
            o_rsc_s = ours_rsc[s]   # [T, L, H, M]
            a_id_s  = all_ids[s]  if has_tier else None   # [T, L, W]
            a_ti_s  = all_tier[s] if has_tier else None   # [T, L, W]
            o_ws_s  = ours_ws[s]  if has_tier else None   # [T, L, H, W]

            # Jaccard (unchanged — uses top-K evictable indices)
            j = M.jaccard_topk(o_tk_s.unsqueeze(2), b_tk_s.unsqueeze(2))  # [T, L, 1]
            per_sample_jaccard.append(j)

            T, L, _, W_pad = b_ws_s.shape
            cos_s = torch.zeros(T, L)
            prs_s = torch.zeros(T, L)
            spm_s = torch.zeros(T, L)
            kl_s  = torch.zeros(T, L)
            mr_s  = torch.zeros(T, L)
            # Tier-aware Jaccard (default fp==kept when no tier data)
            jfp_s   = torch.zeros(T, L)
            jkept_s = torch.zeros(T, L)
            qfid_s  = torch.zeros(T, L)   # summed Q-tier cos-sim
            qcnt_s  = torch.zeros(T, L)   # count of (t,l) cells that had a Q tier

            for t in range(T):
                bws   = b_ws_s[t]    # [L, H, W_pad]
                # Trace index 0 is the PREFILL forward, so at index t the cache
                # holds prefill_len + t tokens (not + t + 1 — that modelled
                # index 0 as the first decode step and ran a flush ahead of the
                # recorded data; see utils.sticky_metrics.flush_geometry, which
                # is corrected in lockstep so Suite B and Suite E agree).
                Sp_t  = max(1, prefill_len + t - ns)
                W_act = min(math.ceil(Sp_t / ws_sz), W_pad)

                for li in range(L):
                    # ── tier-aware Jaccard + Q-tier fidelity (two-tier) ──
                    # Credit the Q tier: compare ours' fp-only vs fp∪Q retained
                    # sets against base's ground-truth top windows of matching
                    # size, so a kept-quantized window no longer scores as a drop.
                    if has_tier:
                        ids_tl  = a_id_s[t, li]                    # [W] orig ids, -1 pad
                        tier_tl = a_ti_s[t, li]                    # [W] 0=fp,1=Q,2=local
                        valid_m = ids_tl >= 0
                        fp_m = valid_m & (tier_tl == 0)
                        q_m  = valid_m & (tier_tl == 1)
                        ours_fp_ids = set(ids_tl[fp_m].tolist())
                        q_ids_t     = ids_tl[q_m]                  # [n_q] orig ids
                        ours_kept   = ours_fp_ids | set(q_ids_t.tolist())
                        ew = max(W_act - local_windows, 0)
                        base_hm = bws[li].mean(dim=0).cpu().numpy()  # [W_pad]
                        jfp_s[t, li] = _jaccard_sets(
                            ours_fp_ids, _base_top_ids(base_hm, ew, len(ours_fp_ids)))
                        jkept_s[t, li] = _jaccard_sets(
                            ours_kept, _base_top_ids(base_hm, ew, len(ours_kept)))
                        # Q-tier fidelity: cos-sim of ours' (dequantized) vs base's
                        # head-mean mass over the Q windows — how faithfully the
                        # int4 tier reproduces the true attention it holds.
                        if q_ids_t.numel() > 0:
                            q_cols = q_m.nonzero(as_tuple=True)[0]
                            o_q = o_ws_s[t, li, :, q_cols].mean(dim=0)   # [n_q]
                            b_q = bws[li, :, q_ids_t].mean(dim=0)        # [n_q]
                            qfid_s[t, li] = _cosine(o_q, b_q)
                            qcnt_s[t, li] = 1.0

                    # Retained window IDs (sorted by original pos, -1 padded)
                    rid_full = o_rid_s[t, li]                      # [M]
                    valid    = (rid_full >= 0) & (rid_full < W_act)
                    idx      = valid.nonzero(as_tuple=True)[0]     # positions in [M]
                    n_ret    = idx.shape[0]
                    if n_ret == 0:
                        continue

                    ret_ids = rid_full[idx]                        # [n_ret] original IDs

                    # Ours' scores for retained windows (mean over heads)
                    o_sc = o_rsc_s[t, li, :, idx].mean(dim=0)     # [n_ret]
                    # Base's scores for same windows (mean over heads)
                    b_sc = bws[li, :, ret_ids].mean(dim=0)         # [n_ret]

                    # 1. Cosine similarity
                    cos_s[t, li] = _cosine(o_sc, b_sc)

                    if n_ret < 2:
                        continue

                    # 2. Pearson correlation
                    prs_s[t, li] = _pearson(o_sc, b_sc)

                    # 3. Spearman rank correlation
                    spm_s[t, li] = _spearman(o_sc, b_sc)

                    # 4. KL(ours ‖ base)
                    kl_s[t, li] = _kl(o_sc, b_sc)

                    # 5. Mass ratio: base_mass / ours_mass over retained windows
                    mr_s[t, li] = b_sc.sum() / o_sc.sum().clamp(min=1e-8)

            per_sample_cos.append(cos_s)
            per_sample_prs.append(prs_s)
            per_sample_spm.append(spm_s)
            per_sample_kl.append(kl_s)
            per_sample_mr.append(mr_s)
            per_sample_jfp.append(jfp_s)
            per_sample_jkept.append(jkept_s)
            per_sample_qfid.append(qfid_s)
            per_sample_qfidcnt.append(qcnt_s)

        # ── stack & mean across samples ───────────────────────────────
        def _smean(lst: list) -> torch.Tensor:
            return torch.stack(lst, 0).mean(0)

        jaccard_stack = torch.stack(per_sample_jaccard, 0)  # [S, T, L, 1]
        jaccard       = jaccard_stack.mean(0)               # [T, L, 1]

        cos        = _smean(per_sample_cos)   # [T, L]
        pearson    = _smean(per_sample_prs)   # [T, L]
        spearman   = _smean(per_sample_spm)   # [T, L]
        kl         = _smean(per_sample_kl)    # [T, L]
        mass_ratio = _smean(per_sample_mr)    # [T, L]

        jaccard_per_layer = M.aggregate_per_layer(jaccard)   # [T, L]
        jaccard_global    = M.aggregate_global(jaccard)      # [T]
        heterogeneity     = M.final_step_heterogeneity(jaccard)  # [L]

        # ── tier-aware Jaccard (Suite A, two-tier) ────────────────────
        # Without tier data (legacy npz), fp == kept == legacy jaccard so the
        # lift is 0 and nothing changes.
        if has_tier:
            jaccard_fp   = _smean(per_sample_jfp)              # [T, L]
            jaccard_kept = _smean(per_sample_jkept)            # [T, L]
            # Mean Q-tier fidelity over cells that actually had a Q tier.
            qfid_sum = torch.stack(per_sample_qfid, 0).sum(0)      # [T, L]
            qfid_cnt = torch.stack(per_sample_qfidcnt, 0).sum(0)   # [T, L]
            q_tier_fidelity_per_layer = torch.where(
                qfid_cnt > 0, qfid_sum / qfid_cnt.clamp(min=1), torch.zeros_like(qfid_sum)
            ).mean(dim=0)                                      # [L]
            tot_cnt = float(qfid_cnt.sum())
            q_tier_fidelity = float(qfid_sum.sum() / tot_cnt) if tot_cnt > 0 else 0.0
        else:
            jaccard_fp   = jaccard_per_layer.clone()
            jaccard_kept = jaccard_per_layer.clone()
            q_tier_fidelity_per_layer = torch.zeros(jaccard_per_layer.shape[-1])
            q_tier_fidelity = 0.0
        jaccard_lift = jaccard_kept - jaccard_fp              # [T, L]
        jaccard_fp_global   = jaccard_fp.mean(dim=-1)         # [T]
        jaccard_kept_global = jaccard_kept.mean(dim=-1)       # [T]
        jaccard_lift_global = jaccard_lift.mean(dim=-1)       # [T]

        # ── Sticky-K policy analytics — Global LIR & absolute missed mass ──
        # Simulated on the base run's window scores (the ground-truth attention
        # masses, since base never evicts) over the same sample/step set used
        # above.  See utils/sticky_metrics.py.  The fp tier gets top_k_fp windows
        # and the Q tier N_q more; at q==0 (N_q==0, top_k_fp==top_k_windows) every
        # series collapses to the legacy single-tier numbers.
        lir_m         = int(om.get("lir_ignore_threshold", 3))
        sticky = SM.compute_sticky_metrics(
            base_ws[:num_samples].numpy(),
            prefill_len=prefill_len,
            num_sink=ns,
            window_size=ws_sz,
            local_windows=local_windows,
            history_budget_K=top_k_fp,
            m=lir_m,
            n_q=n_q_meta,
        )
        # Fidelity-discounted rescued mass: credit the Q tier only to the extent
        # its dequantized attention is faithful (Phase 3).  q_tier_fidelity is a
        # scalar in [0, 1]; scale the whole recovered trajectory by it.
        recovered_q            = np.asarray(sticky["recovered_mass_q"])
        recovered_q_discounted = recovered_q * q_tier_fidelity
        log.info(
            "Sticky-K: global_LIR=%.4f  missed_fp=%.4f  missed_kept=%.4f  "
            "recovered_q=%.4f  q_fid=%.3f  k_fp=%d N_q=%d local=%d",
            float(sticky["global_lir"]), float(sticky["missed_mass_total"]),
            float(sticky["missed_mass_kept_total"]),
            float(sticky["recovered_mass_q_total"]), q_tier_fidelity,
            top_k_fp, n_q_meta, local_windows,
        )

        return {
            "jaccard":           jaccard.numpy(),
            "jaccard_per_layer": jaccard_per_layer.numpy(),
            "jaccard_global":    jaccard_global.numpy(),
            "heterogeneity":     heterogeneity.numpy(),
            "cos_sim":           cos.numpy(),         # [T, L]
            "pearson":           pearson.numpy(),     # [T, L]
            "spearman":          spearman.numpy(),    # [T, L]
            "kl_ours_base":      kl.numpy(),          # [T, L]
            "mass_ratio":        mass_ratio.numpy(),  # [T, L]
            "global_lir":           sticky["global_lir"],            # scalar
            "lir_per_layer":        sticky["lir_per_layer"],         # [L]
            "lir_per_head":         sticky["lir_per_head"],          # [L, H]
            "missed_mass":          sticky["missed_mass"],           # [T] (fp-only)
            "missed_mass_per_layer": sticky["missed_mass_per_layer"], # [T, L]
            "missed_mass_fresh":    sticky["missed_mass_fresh"],     # [T]
            "missed_mass_total":    sticky["missed_mass_total"],     # scalar
            # ── two-tier: Q tier credited (Suite A/B, Phase 1–3) ──
            "jaccard_fp":           jaccard_fp.numpy(),              # [T, L]
            "jaccard_kept":         jaccard_kept.numpy(),            # [T, L]
            "jaccard_lift":         jaccard_lift.numpy(),            # [T, L]
            "jaccard_fp_global":    jaccard_fp_global.numpy(),       # [T]
            "jaccard_kept_global":  jaccard_kept_global.numpy(),     # [T]
            "jaccard_lift_global":  jaccard_lift_global.numpy(),     # [T]
            "missed_mass_kept":     sticky["missed_mass_kept"],      # [T]
            "missed_mass_kept_per_layer": sticky["missed_mass_kept_per_layer"],  # [T, L]
            "missed_mass_kept_total": sticky["missed_mass_kept_total"],          # scalar
            "recovered_mass_q":     sticky["recovered_mass_q"],      # [T]
            "recovered_mass_q_per_layer": sticky["recovered_mass_q_per_layer"],  # [T, L]
            "recovered_mass_q_total": sticky["recovered_mass_q_total"],          # scalar
            "recovered_mass_q_discounted": recovered_q_discounted,   # [T]
            "recovered_mass_q_discounted_total": np.array(
                float(recovered_q_discounted.mean()), dtype=np.float64),
            "missed_mass_fresh_kept": sticky["missed_mass_fresh_kept"],          # [T]
            "recovered_mass_q_fresh": sticky["recovered_mass_q_fresh"],          # [T]
            "q_tier_fidelity":      np.array(q_tier_fidelity, dtype=np.float64),  # scalar
            "q_tier_fidelity_per_layer": q_tier_fidelity_per_layer.numpy(),      # [L]
            "num_samples":       np.array([num_samples], dtype=np.int64),
            "per_sample_jaccard_global": jaccard_stack.mean(dim=(2, 3)).numpy(),  # [S, T]
        }

    # ------------------------------------------------------------------
    def _write(self, results: dict, base: dict, ours: dict,
               cfg: ExperimentConfig) -> Path:
        od = Path(cfg.telemetry.output_dir)
        od.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": "2.2",   # bumped: two-tier (Q-credited) Suite A/B arrays
            "base_npz_path":   base["path"],
            "base_npz_sha256": sha256_file(base["path"]),
            "ours_npz_path":   ours["path"],
            "ours_npz_sha256": sha256_file(ours["path"]),
            "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # ── run config propagated from ours npz (for display / auditing) ──
            "model_name":                 ours["metadata"].get("model_name"),
            "seed":                       ours["metadata"].get("seed"),
            "prefill_len":                ours["metadata"].get("prefill_len"),
            "gen_len":                    ours["metadata"].get("gen_len"),
            "window_size":                ours["metadata"].get("window_size"),
            "num_sink_tokens":            ours["metadata"].get("num_sink_tokens"),
            "local_window_size_resolved": ours["metadata"].get("local_window_size_resolved"),
            "top_k_windows":              ours["metadata"].get("top_k_windows"),
            "cache_budget":               ours["metadata"].get("cache_budget"),
            "quant_ratio":                ours["metadata"].get("quant_ratio", 0.0),
            "top_k_fp":                   ours["metadata"].get("top_k_fp"),
            "N_q":                        ours["metadata"].get("N_q", 0),
        }
        npz_path = od / "faithfulness_results.npz"
        if cfg.output_path:
            npz_path = Path(cfg.output_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        save_arrays = {
            "jaccard":           results["jaccard"],
            "jaccard_per_layer": results["jaccard_per_layer"],
            "jaccard_global":    results["jaccard_global"],
            "heterogeneity":     results["heterogeneity"],
            "cos_sim":           results["cos_sim"],
            "pearson":           results["pearson"],
            "spearman":          results["spearman"],
            "kl_ours_base":      results["kl_ours_base"],
            "mass_ratio":        results["mass_ratio"],
            "global_lir":            results["global_lir"],
            "lir_per_layer":         results["lir_per_layer"],
            "lir_per_head":          results["lir_per_head"],
            "missed_mass":           results["missed_mass"],
            "missed_mass_per_layer": results["missed_mass_per_layer"],
            "missed_mass_fresh":     results["missed_mass_fresh"],
            "missed_mass_total":     results["missed_mass_total"],
            "metadata_json":     np.array([json.dumps(meta)], dtype=object),
        }
        # Two-tier (Q-credited) arrays — additive, always present (collapse to the
        # single-tier values at q == 0).  See Suite A/B, Phase 1–3.
        for opt in (
            "jaccard_fp", "jaccard_kept", "jaccard_lift",
            "jaccard_fp_global", "jaccard_kept_global", "jaccard_lift_global",
            "missed_mass_kept", "missed_mass_kept_per_layer", "missed_mass_kept_total",
            "recovered_mass_q", "recovered_mass_q_per_layer", "recovered_mass_q_total",
            "recovered_mass_q_discounted", "recovered_mass_q_discounted_total",
            "missed_mass_fresh_kept", "recovered_mass_q_fresh",
            "q_tier_fidelity", "q_tier_fidelity_per_layer",
            "num_samples", "per_sample_jaccard_global",
        ):
            if opt in results:
                save_arrays[opt] = results[opt]

        np.savez_compressed(str(npz_path), **save_arrays)
        with open(npz_path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        self._write_summary(results, meta, npz_path)
        log.info("Saved faithfulness: %s", npz_path)
        return npz_path

    # ------------------------------------------------------------------
    @staticmethod
    def _write_summary(results: dict, meta: dict, npz_path: Path) -> None:
        """Write the per-layer CSV + headline markdown beside the npz.

        The npz is the machine-readable artifact and ``scripts/print_faithfulness.py``
        renders a console report, but a run should also leave behind something
        readable without Python — matching the QEvict suite's output contract.
        """
        def _tl(name: str) -> np.ndarray:      # [T, L] → per-layer mean
            arr = np.asarray(results[name], dtype=float)
            return arr.mean(axis=0) if arr.ndim == 2 else arr

        jac = _tl("jaccard_per_layer")
        cols = {
            "jaccard": jac,
            "jaccard_fp": _tl("jaccard_fp"),
            "jaccard_kept": _tl("jaccard_kept"),
            "jaccard_lift": _tl("jaccard_lift"),
            "cos_sim": _tl("cos_sim"),
            "pearson": _tl("pearson"),
            "spearman": _tl("spearman"),
            "kl_ours_base": _tl("kl_ours_base"),
            "mass_ratio": _tl("mass_ratio"),
            "missed_mass": _tl("missed_mass_per_layer"),
            "missed_mass_kept": _tl("missed_mass_kept_per_layer"),
            "recovered_mass_q": _tl("recovered_mass_q_per_layer"),
            "lir": np.asarray(results["lir_per_layer"], dtype=float),
            "q_tier_fidelity": np.asarray(
                results["q_tier_fidelity_per_layer"], dtype=float),
        }
        n_layers = int(jac.shape[0])
        csv_path = npz_path.with_name(npz_path.stem + "_per_layer.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["layer", *cols])
            for li in range(n_layers):
                writer.writerow([li] + [f"{float(v[li]):.6f}" if li < len(v)
                                        else "" for v in cols.values()])

        def _m(name: str) -> float:
            return float(np.mean(cols[name]))

        md = [
            "# Faithfulness results (Suite A/B)",
            "",
            f"- Model: `{meta.get('model_name')}`  |  prefill {meta.get('prefill_len')}"
            f" / gen {meta.get('gen_len')}  |  {n_layers} layers",
            f"- Cache: budget {meta.get('cache_budget')}, window "
            f"{meta.get('window_size')}, quant_ratio {meta.get('quant_ratio')}, "
            f"top_k_fp {meta.get('top_k_fp')}, N_q {meta.get('N_q')}",
            "",
            "| metric | value | reading |",
            "| --- | --- | --- |",
            f"| Jaccard (top-K overlap) | {_m('jaccard'):.4f} | higher = same windows as oracle |",
            f"| Jaccard fp / kept | {_m('jaccard_fp'):.4f} / {_m('jaccard_kept'):.4f} | kept credits the Q tier |",
            f"| Jaccard lift (Q tier) | {_m('jaccard_lift'):+.4f} | gain from crediting int4 survivors |",
            f"| Cosine similarity | {_m('cos_sim'):.4f} | higher = faithful scores |",
            f"| Pearson / Spearman | {_m('pearson'):.4f} / {_m('spearman'):.4f} | higher = better |",
            f"| KL(ours‖base) | {_m('kl_ours_base'):.4f} | lower = better |",
            f"| Mass ratio (base/ours) | {_m('mass_ratio'):.4f} | ~1.0 = well matched |",
            f"| Missed mass (fp only) | {_m('missed_mass'):.4f} | mass below the fp tier |",
            f"| Missed mass (fp+Q kept) | {_m('missed_mass_kept'):.4f} | honest two-tier miss |",
            f"| Recovered mass (Q tier) | {_m('recovered_mass_q'):.4f} | what int4 rescues |",
            f"| Q-tier fidelity | {_m('q_tier_fidelity'):.4f} | dequant faithfulness |",
            f"| Global LIR (Sticky-K, m=3) | {float(results['global_lir']):.4f} | pair-based; NOT the QEvict episode LIR |",
            "",
            f"Per-layer detail: `{csv_path.name}`.  Full arrays: `{npz_path.name}`.",
            "",
        ]
        npz_path.with_name(npz_path.stem + "_summary.md").write_text(
            "\n".join(md), encoding="utf-8")
