"""Why promote Q -> F: the case, from recorded observation runs. numpy + matplotlib.

    python scripts/promotion_motivation.py \\
        --zip q0.70=<run q0.7>.zip --zip q0.20=<run q0.2>.zip \\
        --ablation q0.70=reports/promotion_ablation/q0.7.json \\
        --ablation q0.20=reports/promotion_ablation/q0.2.json \\
        --out-json reports/promotion_motivation.json \\
        --fig reports/figures/obs_promotion

Per run, the measured (promoting) cache against the one-way replay of the same
trajectory (``promotion_ablation.replay_tiers``: same queries, same bytes, the
policy's own ranking signal, int2 windows barred from fp), over the decode in
64-step stretches:

* **fp-tier attention** -- each query head's FullKV attention on the fp tier
  (per-head share, then averaged);
* **fp-tier aim** -- at each eviction, the share of the fp tier that is among
  the ``k_fp`` held windows the NEXT 32 steps attend to most (hindsight, head
  mean). Chance is ``k_fp / (k_fp + N_q)``;
* **staleness** -- the share of layer-evictions whose fp set differs between the
  two arms.

The figure is three print-size panels: (a) staleness over the decode, (b)
fp-tier aim over the decode, with vs without promotion, against chance, (c)
what one promotion carries (from the ablation JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.evaluation import promotion_ablation as PA          # noqa: E402

BIN, HORIZON = 64, 32
KEYS = ("tier", "evict_step", "rank_signal", "rank_signal_steps", "full_mass")


def analyse(zip_path: str) -> Dict[str, object]:
    meta = PA._read_meta(zip_path)
    k_fp, n_q, local_w = PA._geometry(meta)
    T = meta["axes"]["T"]
    nb = (T - 1) // BIN
    acc = {k: np.zeros(nb) for k in ("fp_with", "fp_without", "cells",
                                     "aim_with", "aim_without", "aim_n",
                                     "stale", "stale_n")}
    for a in PA.iter_zip_samples(zip_path, KEYS):
        tier, ev_mask = a["tier"], a["evict_step"]
        one, _ = PA.replay_tiers(tier, ev_mask, a["rank_signal"], a["rank_signal_steps"],
                                 k_fp, n_q, local_w, oneway=True)
        ev = np.flatnonzero(ev_mask)
        L = tier.shape[1]
        for li in range(L):
            fm = a["full_mass"][:, li].astype(np.float32)             # [T, H, W]
            hm = fm.mean(1)                                            # [T, W] head mean
            b = (np.arange(1, T) - 1) // BIN
            for arm, tr in (("with", tier[1:, li]), ("without", one[1:, li])):
                m = np.einsum("thw,tw->t", fm[1:], (tr == PA.TIER_FP).astype(np.float32))
                acc[f"fp_{arm}"] += np.bincount(b, m / fm.shape[1], nb)
            acc["cells"] += np.bincount(b, minlength=nb)
            for e in ev[1:]:
                if e + HORIZON > T - 1:
                    break
                j = (e - 1) // BIN
                held = np.flatnonzero((tier[e, li] == PA.TIER_FP) | (tier[e, li] == PA.TIER_Q))
                fut = hm[e + 1:e + HORIZON + 1].mean(0)
                best = set(held[np.argsort(-fut[held], kind="stable")[:k_fp]])
                for arm, tr in (("with", tier[e, li]), ("without", one[e, li])):
                    fp = np.flatnonzero(tr == PA.TIER_FP)
                    acc[f"aim_{arm}"][j] += len(best.intersection(fp)) / k_fp
                acc["aim_n"][j] += 1
            for e in ev:
                j = min((e - 1) // BIN, nb - 1)
                acc["stale"][j] += not np.array_equal(tier[e, li] == PA.TIER_FP,
                                                      one[e, li] == PA.TIER_FP)
                acc["stale_n"][j] += 1
        del a
    out = {"k_fp": k_fp, "n_q": n_q, "chance": k_fp / (k_fp + n_q),
           "bins": [f"{BIN * i + 1}-{BIN * (i + 1)}" for i in range(nb)]}
    for arm in ("with", "without"):
        out[f"fp_{arm}"] = (acc[f"fp_{arm}"] / acc["cells"]).tolist()
        out[f"aim_{arm}"] = (acc[f"aim_{arm}"] / np.maximum(acc["aim_n"], 1)).tolist()
        out[f"fp_{arm}_all"] = float(acc[f"fp_{arm}"].sum() / acc["cells"].sum())
        out[f"aim_{arm}_all"] = float(acc[f"aim_{arm}"].sum() / acc["aim_n"].sum())
    out["aim_bins_scored"] = [int(x) for x in acc["aim_n"]]
    out["stale"] = (acc["stale"] / np.maximum(acc["stale_n"], 1)).tolist()
    out["stale_all"] = float(acc["stale"].sum() / acc["stale_n"].sum())
    return out


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------

INK, MUTED, GRID = "#1c2026", "#5b6878", "#d9dde3"
WITH, WITHOUT, SOFT = "#6446a4", "#82aee3", "#c1b3f2"


def figure(res: Dict[str, dict], abl: Dict[str, dict], stem: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                         "legend.fontsize": 7.2, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    labs = list(res)
    ls = dict(zip(labs, ("-", "--", ":")))
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.3),
                           gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1, 0.82]})
    x = np.arange(len(res[labs[0]]["bins"])) * BIN + BIN / 2

    # (a) how often the fp tier without promotion is not the one the cache would pick
    for lab in labs:
        y = 100 * np.asarray(res[lab]["stale"])
        ax[0].plot(x, y, ls[lab], color=WITH, lw=1.5, marker="o", ms=2.6)
        ax[0].annotate(lab.replace("q", "q = "), (x[-1], y[-1]), xytext=(-2, 5),
                       textcoords="offset points", ha="right", fontsize=7.2, color=INK)
    ax[0].set_title("(a) Stale fp tier (no promotion)", loc="left")
    ax[0].set_ylabel("Layer-evictions stale (%)")
    ax[0].set_ylim(0, 100)

    # (b) is the fp tier on the windows the next 32 steps attend to most?
    for lab in labs:
        n = int(np.sum(np.asarray(res[lab]["aim_bins_scored"]) > 0))
        for arm, col in (("with", WITH), ("without", WITHOUT)):
            ax[1].plot(x[:n], 100 * np.asarray(res[lab][f"aim_{arm}"][:n]), ls[lab],
                       color=col, lw=1.5, marker="o", ms=2.6)
        ax[1].axhline(100 * res[lab]["chance"], color=MUTED, lw=0.7, ls=ls[lab], alpha=0.6)
    for lab in labs:
        ax[1].text(6, 100 * res[lab]["chance"] + 1.5, "chance", fontsize=6.8, color=MUTED)
    ax[1].set_title("(b) fp tier on top windows", loc="left")
    ax[1].set_ylabel("fp tier in next-32 top-$k$ (%)")
    ax[1].set_ylim(0, 100)
    for a_ in ax[:2]:
        a_.set_xlabel("Decode step")
        a_.set_xlim(0, 512)
        a_.set_xticks([0, 128, 256, 384, 512])
        a_.grid(axis="y", color=GRID, lw=0.6)
        a_.spines[["top", "right"]].set_visible(False)

    # (c) what one promotion carries
    lab_ab = list(abl)
    pos = np.arange(len(lab_ab))
    dem = np.array([100 * abl[k]["swap"]["demoted_mass"] for k in lab_ab])
    pro = np.array([100 * abl[k]["swap"]["promoted_mass"] for k in lab_ab])
    w = 0.38
    ax[2].bar(pos - w / 2, dem, w, color=WITHOUT)
    ax[2].bar(pos + w / 2, pro, w, color=WITH)
    for i in range(len(lab_ab)):
        ax[2].text(pos[i] + w / 2, pro[i] + 0.02, f"{pro[i] / dem[i]:.1f}\u00d7",
                   ha="center", va="bottom", fontsize=7.2, color=INK)
    ax[2].set_xticks(pos, [k.replace("q", "q = ") for k in lab_ab])
    ax[2].set_title("(c) One promotion", loc="left")
    ax[2].set_ylabel("Attention / step, next 32 (%)")
    ax[2].set_ylim(0, pro.max() * 1.3)
    ax[2].grid(axis="y", color=GRID, lw=0.6)
    ax[2].spines[["top", "right"]].set_visible(False)

    handles = [Line2D([], [], color=WITH, lw=5, label="With promotion / promoted window"),
               Line2D([], [], color=WITHOUT, lw=5,
                      label="No promotion / window it displaces"),
               *[Line2D([], [], color=MUTED, lw=1.2, ls=ls[lab],
                        label=lab.replace("q", "q = ")) for lab in labs]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.14), handlelength=2.2, columnspacing=1.4)
    written = []
    for ext in ("pdf", "svg", "png"):
        p = f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=600 if ext == "png" else None)
        written.append(p)
    plt.close(fig)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", action="append", default=[], help="LABEL=path")
    ap.add_argument("--ablation", action="append", default=[], help="LABEL=json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--fig", default=None, help="figure path stem (writes pdf/svg/png)")
    ap.add_argument("--from-json", default=None,
                    help="redraw from a previous --out-json instead of the zips")
    a = ap.parse_args()
    res = json.loads(Path(a.from_json).read_text()) if a.from_json else {}
    for spec in ([] if a.from_json else a.zip):
        lab, path = spec.split("=", 1)
        res[lab] = analyse(path)
        print(lab, json.dumps({k: v for k, v in res[lab].items()
                               if not isinstance(v, list)}), flush=True)
    abl = {s.split("=", 1)[0]: json.loads(Path(s.split("=", 1)[1]).read_text())
           for s in a.ablation}
    if not a.from_json:
        Path(a.out_json).write_text(json.dumps(res, indent=1))
    if a.fig:
        print("wrote", figure(res, abl, a.fig))


if __name__ == "__main__":
    main()
