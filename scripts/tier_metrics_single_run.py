"""The tier study's seven metrics (M1-M7) for ONE observation-collector run.

    python scripts/tier_metrics_single_run.py \\
        --base-npz parity/parity_base.npz --ours-npz parity/parity_ours.npz \\
        --zip outputs/obs_data/<run>.zip --label q0.70 --out-json m1_m7_q0.70.json

``modules/evaluation/tier_study`` computes M1-M7 over a matrix of five separately
recorded runs (R0 full KV, R1/R2 evict-only, R3/R4 three-tier). A collector run
already carries R0 (its never-evicted shadow KV) and R3 (the cache itself), so this
script builds that package's per-condition view from the QEvict suite's
``ObservationInputs`` and calls the package's own metric code:

* R3 -- the measured three-tier cache (``measured_fp_plus_q`` / ``measured_fp_only``).
* R2 -- **simulated**, not recorded: evict-only fp16 at the same bytes (the run's
  ``top_k_windows`` evictable windows + the local tail), ranked on exact
  cumulative mass at the run's own eviction steps. A recorded R2 ranks on its own
  scores; the collector shows the cache's scores track exact mass (Spearman 0.99).
* S2 -- **simulated** StreamingLLM-style evict-only at the same bytes (recency).
* R1 / R4 (other window sizes) are not available from one run, so M1/M2's
  granularity contrasts are not produced here.

M5 needs the cache's own per-head scores, which the parity export drops, so it
reads ``rank_signal`` (the exact score each eviction ranked on) from the zip.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.evaluation import qevict_observations as Q          # noqa: E402
from modules.evaluation.tier_study import m2_selection_churn as M2  # noqa: E402
from modules.evaluation.tier_study import m3_tier_mass_distribution as M3  # noqa: E402
from modules.evaluation.tier_study import m4_jaccard_topk_overlap as M4  # noqa: E402
from modules.evaluation.tier_study import m7_global_lir as M7       # noqa: E402
from utils import qevict_metrics as QM                              # noqa: E402
from utils import sticky_metrics as SM                              # noqa: E402
from analyze_observation_zip import sim_evicted                     # noqa: E402

HORIZONS = (8, 32, 64)


@dataclass
class View:
    """The subset of ``tier_study.ConditionView`` the metric code reads."""
    event_steps: np.ndarray
    w_act: np.ndarray
    ew_act: np.ndarray
    creation: np.ndarray
    top_k_fp: int
    n_q: int
    num_windows: int
    acc_fp: np.ndarray
    acc_q: np.ndarray

    @property
    def acc_alive(self) -> np.ndarray:
        return self.acc_fp | self.acc_q

    @property
    def num_events(self) -> int:
        return int(self.event_steps.size)

    def band_mask(self) -> np.ndarray:
        cols = np.arange(self.num_windows)[None, :]
        return cols < self.ew_act[self.event_steps][:, None]


class Matrix:
    """The subset of ``tier_study.RunMatrix`` M7's selection builder reads."""

    def __init__(self, inp, views):
        self.inp, self.conditions = inp, views
        self.num_traces = inp.mass_cum.shape[0]

    def mass_cum(self, cid, head=None):
        return self.inp.mass_cum


def boot(values_mr, groups, seed=0, n=1000):
    """Per-trace mean over events, bootstrapped over traces -> (mean, lo, hi)."""
    per = QM.group_reduce(QM.nanmean(values_mr, axis=1), groups)
    mu, lo, hi = QM.bootstrap_mean_ci(per, 0.95, n, seed)
    return {"mean": float(mu), "lo": float(lo), "hi": float(hi)}


def simulated_access(inp, rank_fn, keep):
    M, T, W = inp.mass_cum.shape
    ev = np.asarray(inp.event_steps)
    out = np.zeros((M, ev.size, W), bool)
    local_w = int(inp.geometry.get("local_windows", 16))
    for m in range(M):
        e = sim_evicted(lambda t: rank_fn(m, t), np.asarray(inp.w_act), ev, keep, local_w, W)
        out[m] = inp.valid[m, ev] & ~e[ev]
    return out


def m5_fidelity(zip_path):
    """Cosine (cache's own score vs truth) over each eviction's int2 windows, per head."""
    cos, ratio = [], []
    with zipfile.ZipFile(zip_path) as zf:
        meta = json.loads(zf.read("meta.json"))
        for s in meta["samples"]:
            a = np.load(io.BytesIO(zf.read(s["file"])))
            tier, rs, steps = a["tier"], a["rank_signal"], list(a["rank_signal_steps"])
            fm = a["full_mass"]
            L = tier.shape[1]
            for l in range(L):
                cum = np.cumsum(fm[:, l].astype(np.float64), 0)          # [T, H, W]
                for r, e in enumerate(steps):
                    qw = np.flatnonzero(tier[e, l] == 1)
                    if qw.size < 2:
                        continue
                    ours = rs[r, l][:, qw].astype(np.float64)            # [H, nq]
                    truth = cum[e - 1][:, qw]
                    ok = np.isfinite(ours).all(-1)
                    num = (ours * truth).sum(-1)
                    den = np.linalg.norm(ours, axis=-1) * np.linalg.norm(truth, axis=-1)
                    c = np.where(ok & (den > 0), num / np.where(den > 0, den, 1), np.nan)
                    cos.append(c)
                    ratio.append(np.where(ok, truth.sum(-1) / np.where(ours.sum(-1) > 0, ours.sum(-1), np.nan), np.nan))
    cos, ratio = np.concatenate(cos), np.concatenate(ratio)
    return {"cosine_mean": float(np.nanmean(cos)), "cosine_median": float(np.nanmedian(cos)),
            "cosine_p10": float(np.nanpercentile(cos, 10)),
            "q_mass_ratio_median": float(np.nanmedian(ratio)), "n": int(np.isfinite(cos).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-npz", required=True)
    ap.add_argument("--ours-npz", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    base, ours = Q.load_parity_npz(a.base_npz), Q.load_parity_npz(a.ours_npz)
    inp = Q.build_observation_inputs(base, ours)
    om = ours["metadata"]
    M, T, W = inp.mass_cum.shape
    ev = np.asarray(inp.event_steps)
    groups = inp.trace_group
    k_fp, n_q, k_bytes = int(om["top_k_fp"]), int(om["N_q"]), int(om["top_k_windows"])
    inp.geometry.setdefault("local_windows", int(om.get("local_window_size_resolved", 128)) // 8)
    common = dict(event_steps=ev, w_act=np.asarray(inp.w_act), ew_act=np.asarray(inp.ew_act),
                  creation=np.asarray(inp.creation), num_windows=W)
    acc_all = inp.accessible["measured_fp_plus_q"]
    acc_fp_meas = inp.accessible["measured_fp_only"]
    acc_q = acc_all & ~acc_fp_meas
    cum = inp.mass_cum
    r2 = simulated_access(inp, lambda m, t: cum[m, t - 1], k_bytes)
    s2 = simulated_access(inp, lambda m, t: np.arange(W, dtype=float), k_bytes)
    views = {
        "R3": View(top_k_fp=k_fp, n_q=n_q, acc_fp=acc_fp_meas, acc_q=acc_q, **common),
        "R2sim": View(top_k_fp=k_bytes, n_q=0, acc_fp=r2, acc_q=np.zeros_like(r2), **common),
        "S2sim": View(top_k_fp=k_bytes, n_q=0, acc_fp=s2, acc_q=np.zeros_like(s2), **common),
    }
    band = views["R3"].band_mask()[None]
    out = {"label": a.label, "geometry": {"top_k_fp": k_fp, "N_q": n_q, "fp_only_same_bytes": k_bytes,
                                         "traces": M, "events": int(ev.size), "T": T, "W": W},
           "read_gate": om.get("read_gate")}

    # M1 Future Missed Mass (+ M6b paired reductions come from the same numbers)
    sets = {"R3 fp+int2 (measured)": acc_all, "R3 fp only (measured)": acc_fp_meas,
            "R2 evict-only, same bytes (sim, exact scores)": r2,
            "S2 StreamingLLM-style, same bytes (sim)": s2}
    out["M1_fmm"] = {}
    for H in HORIZONS:
        out["M1_fmm"][str(H)] = {k: boot(QM.future_missed_mass(inp.mass_step, v, ev, H,
                                                                creation_steps=inp.creation), groups, 1)
                                  for k, v in sets.items()}

    # M2 Selection Churn on the evictable band (+ per decode step, + mass-weighted)
    mass_ev = cum[:, ev, :]
    out["M2_churn"] = {}
    for k, v in sets.items():
        sel = v & band
        ch = QM.selection_churn(sel)
        mw = M2._mass_weighted_churn(sel, mass_ev)
        per_flush = boot(ch, groups, 2)
        out["M2_churn"][k] = {"per_flush": per_flush,
                              "per_decode_step": per_flush["mean"] * (ev.size - 1) / (T - 1),
                              "mass_weighted": boot(mw, groups, 3)}

    # M3 tier mass distribution + boundary sharpness
    out["M3_tiers"] = {}
    for cid in ("R3", "R2sim"):
        ser = M3.bucket_event_masses(mass_ev, views[cid])
        out["M3_tiers"][cid] = {k: float(np.nanmean(v)) for k, v in ser.items()
                                if k not in ("k_fp", "n_q", "band_windows")}

    # M4 alive-set Jaccard vs the same-size oracle, early / late
    out["M4_jaccard"] = {}
    traj = {}
    for name, cid, s in (("R3 fp+int2", "R3", acc_all), ("R3 fp only", "R3", acc_fp_meas),
                         ("R2 evict-only (sim)", "R2sim", r2), ("S2 StreamingLLM (sim)", "S2sim", s2)):
        j, _ = M4.overlap_trajectory(s & band, mass_ev, views[cid])
        traj[name] = j
        early, late = M4._halves(j)
        out["M4_jaccard"][name] = {"all": boot(j, groups, 4), "early": float(np.nanmean(early)),
                                   "late": float(np.nanmean(late))}
    lead = traj["R3 fp+int2"] > traj["R2 evict-only (sim)"]
    both = np.isfinite(traj["R3 fp+int2"]) & np.isfinite(traj["R2 evict-only (sim)"])
    out["M4_jaccard"]["share_of_flushes_three_tier_leads_R2"] = float(lead[both].mean())

    # M5 Q-tier fidelity
    out["M5_qtier_fidelity"] = m5_fidelity(a.zip) if n_q else None

    # M6 recovered mass: simulated Fresh-K / Sticky-K over the ground truth
    out["M6_recovered"] = {}
    for sticky in (False, True):
        fp3 = np.zeros((M, T)); kept3 = np.zeros((M, T)); fp2 = np.zeros((M, T))
        for m in range(M):
            _, fp3[m], kept3[m] = SM.simulate_policy(cum[m], k_fp, inp.w_act, inp.ew_act,
                                                     is_sticky=sticky, n_q=n_q)
            _, fp2[m], _ = SM.simulate_policy(cum[m], k_bytes, inp.w_act, inp.ew_act,
                                              is_sticky=sticky, n_q=0)
        out["M6_recovered"]["sticky" if sticky else "fresh"] = {
            "missed_fp_R3": float(fp3.mean()), "missed_kept_R3": float(kept3.mean()),
            "recovered_vs_nothing": float((fp3 - kept3).mean()),
            "missed_R2_capacity": float(fp2.mean()),
            "recovered_vs_same_bytes_fp16": float((fp2 - kept3).mean())}
    f32 = out["M1_fmm"]["32"]
    out["M6_recovered"]["paired_fmm_32"] = {
        "R2sim_minus_R3": f32["R2 evict-only, same bytes (sim, exact scores)"]["mean"] - f32["R3 fp+int2 (measured)"]["mean"],
        "R3fp_minus_R3": f32["R3 fp only (measured)"]["mean"] - f32["R3 fp+int2 (measured)"]["mean"]}

    # M7 Global LIR (uncapped), oracle vs the real fp tier, + lag-1 transitions
    mx = Matrix(inp, {"R3": views["R3"]})
    creation_ev = M7.band_creation_events(views["R3"])
    out["M7_lir"] = {}
    for name, sel in M7.selection_matrices(mx, "R3"):
        row = {}
        for m_in in (1, 2, 4, 8):
            res = QM.episode_lir(sel, m_in, None, creation_ev)
            row[f"m{m_in}"] = {"rate": float(res["global_rate"]), "eligible": int(res["eligible"]),
                               "rescued": int(res["rescued"])}
        tr = QM.binary_transition(sel, 1, creation_ev)
        p = np.asarray(tr["pooled_probabilities"])
        row["P01"], row["P10"] = float(p[0, 1]), float(p[1, 0])
        out["M7_lir"][name] = row

    Path(a.out_json).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
