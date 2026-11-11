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

The figure is three print-size panels, captioned under their x axes, with the
legend on the right: (a) staleness over the decode, (b) fp-tier aim over the
decode, with vs without promotion, against a random pick of ``k_fp`` of the
held windows, (c) what one promotion carries (from the ablation JSON).
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

INK, MUTED = "#1c2026", "#5b6878"
# Checked with the dataviz validator (light surface): lightness, chroma, CVD and
# normal-vision separation, 3:1 contrast all pass for this pair.
WITH, WITHOUT = "#6446a4", "#4a86d6"
CAPTION_Y = -0.29            # panel captions sit under each x axis, all at one height


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
    ls = dict(zip(labs, ("-", "--", "-.")))
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.3),
                           gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1, 0.8]})
    x = np.arange(len(res[labs[0]]["bins"])) * BIN + BIN / 2
    mk = dict(lw=1.5, marker="o", ms=3.2, markeredgewidth=0)

    # (a) how often the fp tier without promotion is not the one the cache would pick
    for lab in labs:
        y = 100 * np.asarray(res[lab]["stale"])
        ax[0].plot(x, y, ls[lab], color=WITH, **mk)
        ax[0].annotate(lab.replace("q", "q = "), (x[-1], y[-1]), xytext=(0, 6),
                       textcoords="offset points", ha="right", fontsize=7.2, color=INK)
    ax[0].set_ylabel("Layer-evictions stale (%)")
    ax[0].set_ylim(0, 100)

    # (b) is the fp tier on the windows the next 32 steps attend to most?
    for lab in labs:
        n = int(np.sum(np.asarray(res[lab]["aim_bins_scored"]) > 0))
        for arm, col in (("with", WITH), ("without", WITHOUT)):
            ax[1].plot(x[:n], 100 * np.asarray(res[lab][f"aim_{arm}"][:n]), ls[lab],
                       color=col, **mk)
        c = 100 * res[lab]["chance"]
        ax[1].axhline(c, color=MUTED, lw=0.8, ls=":")
        ax[1].text(508, c + 1.8, f"random pick, {lab.replace('q', 'q = ')}: {c:.0f}%",
                   ha="right", va="bottom", fontsize=6.8, color=MUTED)
    ax[1].set_ylabel("fp windows in next-32 top-$k$ (%)")
    ax[1].set_ylim(0, 100)
    for a_ in ax[:2]:
        a_.set_xlabel("Decode step")
        a_.set_xlim(0, 512)
        a_.set_xticks([0, 128, 256, 384, 512])

    # (c) what one promotion carries
    lab_ab = list(abl)
    pos = np.arange(len(lab_ab))
    dem = np.array([100 * abl[k]["swap"]["demoted_mass"] for k in lab_ab])
    pro = np.array([100 * abl[k]["swap"]["promoted_mass"] for k in lab_ab])
    w = 0.36
    ax[2].bar(pos - w / 2 - 0.01, dem, w, color=WITHOUT)
    ax[2].bar(pos + w / 2 + 0.01, pro, w, color=WITH)
    for i in range(len(lab_ab)):
        ax[2].text(pos[i] + w / 2, pro[i] + 0.02, f"{pro[i] / dem[i]:.1f}\u00d7",
                   ha="center", va="bottom", fontsize=7.2, color=INK)
    ax[2].set_xticks(pos, [k.lstrip("q") for k in lab_ab])
    ax[2].set_xlabel("Quant ratio")
    ax[2].set_ylabel("Attention / step, next 32 (%)")
    ax[2].set_ylim(0, pro.max() * 1.25)

    captions = ("(a) Stale fp tier, no promotion",
                "(b) fp tier on windows read next",
                "(c) One promotion")
    for a_, cap in zip(ax, captions):
        a_.spines[["top", "right"]].set_visible(False)
        a_.text(0.5, CAPTION_Y, cap, transform=a_.transAxes, ha="center", va="top",
                fontsize=8.5, color=INK)

    handles = [Line2D([], [], color=WITH, lw=5, label="With promotion\n(promoted window)"),
               Line2D([], [], color=WITHOUT, lw=5,
                      label="No promotion\n(window it replaces)"),
               *[Line2D([], [], color=MUTED, lw=1.3, ls=ls[lab],
                        label=lab.replace("q", "q = ")) for lab in labs],
               Line2D([], [], color=MUTED, lw=0.8, ls=":", label="Random pick")]
    fig.legend(handles=handles, loc="center left", frameon=False,
               bbox_to_anchor=(0.905, 0.55), handlelength=2.2, labelspacing=0.9)
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
