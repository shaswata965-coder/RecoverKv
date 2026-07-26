"""Pretty-print faithfulness results (schema v2.0).

Sections:
  1. Run configuration
  2. Jaccard similarity — global + per-layer (prefill / generation)
  3. Master layer scorecard — all 5 distribution metrics side-by-side
  4. Generation trend — each metric at prefill / Q1 / Q2 / Q3 / Q4 per layer
  5. Layer rankings — top-3 / bottom-3 per metric
  6. Head heterogeneity
  7. Per-sample breakdown

Usage (Kaggle cell):
    exec(open("scripts/print_faithfulness.py").read())
  or:
    %run scripts/print_faithfulness.py
"""
from __future__ import annotations
import glob as _glob
import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────
# Resolution order: $FAITHFULNESS_NPZ, then argv[1], then the Kaggle default,
# then a search of the usual output roots.  The path used to be hard-wired,
# which broke every invocation that wrote its npz anywhere else.
_DEFAULT = "/kaggle/working/outputs/faithfulness/faithfulness_results.npz"
_SEARCH_ROOTS = ("outputs", "/kaggle/working/outputs", ".")

# The report is drawn with box characters; a non-UTF-8 console (Windows cp1252)
# would otherwise raise UnicodeEncodeError mid-table.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _resolve_npz() -> str:
    explicit = os.environ.get("FAITHFULNESS_NPZ") or (
        sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".npz") else None)
    if explicit:
        return explicit
    if Path(_DEFAULT).exists():
        return _DEFAULT
    for root in _SEARCH_ROOTS:
        hits = sorted(_glob.glob(f"{root}/**/faithfulness_results.npz",
                                 recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "No faithfulness_results.npz found. Set FAITHFULNESS_NPZ=<path>, pass "
        "the path as an argument, or run the faithfulness suite first.")


NPZ_PATH = _resolve_npz()

# ── Load ──────────────────────────────────────────────────────────────
data = np.load(NPZ_PATH, allow_pickle=True)

jaccard_global    = data["jaccard_global"]       # [T]
jaccard_per_layer = data["jaccard_per_layer"]    # [T, L]
het               = data["heterogeneity"]        # [L]
cos_sim    = data["cos_sim"]        # [T, L]
pearson    = data["pearson"]        # [T, L]
spearman   = data["spearman"]       # [T, L]
kl         = data["kl_ours_base"]   # [T, L]
mass_ratio = data["mass_ratio"]     # [T, L]

T, L = jaccard_per_layer.shape

meta = {}
if "metadata_json" in data.files:
    meta = json.loads(str(data["metadata_json"][0]))

PREFILL_STEPS = 1
GEN_START, GEN_END = PREFILL_STEPS, T
N_GEN = GEN_END - GEN_START

W = 96   # print width

# ── Helpers ───────────────────────────────────────────────────────────

def _bar(v: float, lo: float = 0.0, hi: float = 1.0, width: int = 8) -> str:
    """Filled bar proportional to v in [lo, hi]."""
    frac = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-9)))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)

def _drift_arrow(delta: float, thr: float = 0.005) -> str:
    """↑ / → / ↓ based on delta magnitude."""
    if delta >  thr: return "↑"
    if delta < -thr: return "↓"
    return "→"

def _gen_quartiles(arr_t: np.ndarray):
    """Split a 1-D generation array into [Q1, Q2, Q3, Q4] means."""
    if N_GEN == 0:
        return [0.0, 0.0, 0.0, 0.0]
    g = arr_t[GEN_START:GEN_END]
    q = N_GEN // 4
    if q == 0:
        return [g.mean()] * 4
    segs = [g[:q], g[q:2*q], g[2*q:3*q], g[3*q:]]
    return [s.mean() for s in segs]

def _hdr(title: str) -> None:
    print()
    print("┌" + "─" * (W - 2) + "┐")
    print("│" + title.center(W - 2) + "│")
    print("└" + "─" * (W - 2) + "┘")

def _sep() -> None:
    print("─" * W)

# ══════════════════════════════════════════════════════════════════════
# 1. Configuration header
# ══════════════════════════════════════════════════════════════════════
print()
print("═" * W)
print("  FAITHFULNESS  RESULTS  SUMMARY".center(W))
print("═" * W)
if meta:
    print(f"  Model      : {meta.get('model_name','—')}")
    print(f"  Prefill    : {meta.get('prefill_len','—')} tokens    "
          f"Gen: {meta.get('gen_len','—')} tokens    "
          f"Total steps: {T}  (1 prefill + {T-1} gen)    Layers: {L}")
    print(f"  Window     : {meta.get('window_size','—')}    "
          f"Sink: {meta.get('num_sink_tokens','—')}    "
          f"Local: {meta.get('local_window_size_resolved','—')} tokens    "
          f"Top-K: {meta.get('top_k_windows','—')}    "
          f"Budget: {meta.get('cache_budget','—')}    "
          f"Schema: v{meta.get('schema_version','?')}")
print("═" * W)


# ══════════════════════════════════════════════════════════════════════
# 2. Jaccard
# ══════════════════════════════════════════════════════════════════════
_hdr("  JACCARD SIMILARITY  —  Top-K Window Overlap  (higher = better)")

jac_pre = jaccard_per_layer[:PREFILL_STEPS]         # [1, L]
jac_gen = jaccard_per_layer[GEN_START:GEN_END]       # [N_GEN, L]

print()
print(f"  Global  Prefill: {jac_pre.mean():.4f}   "
      f"Gen: {jac_gen.mean():.4f}   "
      f"Overall: {jaccard_global.mean():.4f}   "
      f"Min: {jaccard_global.min():.4f}   Max: {jaccard_global.max():.4f}")

print()
qdesc = ["Q1", "Q2", "Q3", "Q4"]
print(f"  {'Lyr':>4}  {'Pre':>6}  "
      + "  ".join(f"{q:>6}" for q in qdesc)
      + f"  {'Drift':>5}  {'Gen↓':>6}  {'Overall':>8}  Bar")
print(f"  {'─'*4}  {'─'*6}  "
      + "  ".join("─" * 6 for _ in qdesc)
      + f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}")

jac_gen_q_all = np.array([_gen_quartiles(jaccard_per_layer[:, li]) for li in range(L)])
for li in range(L):
    pre_v = jac_pre[:, li].mean() if len(jac_pre) else 0.0
    qs    = jac_gen_q_all[li]
    drift = qs[-1] - qs[0]
    gen_m = jac_gen[:, li].mean() if N_GEN else 0.0
    ovr   = jaccard_per_layer[:, li].mean()
    bar   = _bar(ovr, 0, 1)
    print(f"  {li:>4d}  {pre_v:>6.4f}  "
          + "  ".join(f"{q:>6.4f}" for q in qs)
          + f"  {drift:>+5.3f}  {gen_m:>6.4f}  {ovr:>8.4f}  {bar}")

print(f"  {'avg':>4s}  {jac_pre.mean():>6.4f}  "
      + "  ".join(f"{jac_gen_q_all[:, i].mean():>6.4f}" for i in range(4))
      + f"  {'—':>5}  {jac_gen.mean():>6.4f}  {jaccard_per_layer.mean():>8.4f}")


# ══════════════════════════════════════════════════════════════════════
# 3. Master layer scorecard — all 5 metrics, generation phase only
# ══════════════════════════════════════════════════════════════════════
_hdr("  MASTER LAYER SCORECARD  —  Generation-phase means  (cos/pearson/spearman ↑,  kl ↓,  mass_ratio ≈1)")

gen_cos  = cos_sim[GEN_START:GEN_END]
gen_prs  = pearson[GEN_START:GEN_END]
gen_spm  = spearman[GEN_START:GEN_END]
gen_kl   = kl[GEN_START:GEN_END]
gen_mr   = mass_ratio[GEN_START:GEN_END]

# Per-layer generation means
cos_l  = gen_cos.mean(axis=0)   # [L]
prs_l  = gen_prs.mean(axis=0)
spm_l  = gen_spm.mean(axis=0)
kl_l   = gen_kl.mean(axis=0)
mr_l   = gen_mr.mean(axis=0)
jac_l  = jac_gen.mean(axis=0) if N_GEN else jaccard_per_layer.mean(axis=0)

# Composite rank: normalise each metric to 0-1 (higher=better) then average
def _norm(v, invert=False):
    lo, hi = v.min(), v.max()
    n = (v - lo) / (hi - lo + 1e-9)
    return 1 - n if invert else n

norm_cos = _norm(cos_l)
norm_prs = _norm(prs_l)
norm_spm = _norm(spm_l)
norm_kl  = _norm(kl_l,  invert=True)   # lower KL = better
norm_mr  = _norm(-np.abs(mr_l - 1.0), invert=True)  # closer to 1 = better (already inverted)
composite = (norm_cos + norm_prs + norm_spm + norm_kl + norm_mr + _norm(jac_l)) / 6

print()
print(f"  {'Lyr':>4}  {'Jac':>7}  {'CosSim':>7}  {'Pearson':>7}  "
      f"{'Spearman':>8}  {'KL↓':>7}  {'MassRat':>7}  {'Score':>6}  Bar(score)")
print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}")

for li in range(L):
    sc = composite[li]
    print(f"  {li:>4d}  {jac_l[li]:>7.4f}  {cos_l[li]:>7.4f}  {prs_l[li]:>7.4f}  "
          f"{spm_l[li]:>8.4f}  {kl_l[li]:>7.4f}  {mr_l[li]:>7.4f}  "
          f"{sc:>6.4f}  {_bar(sc, 0, 1)}")

print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}")
print(f"  {'avg':>4s}  {jac_l.mean():>7.4f}  {cos_l.mean():>7.4f}  {prs_l.mean():>7.4f}  "
      f"{spm_l.mean():>8.4f}  {kl_l.mean():>7.4f}  {mr_l.mean():>7.4f}  "
      f"{composite.mean():>6.4f}")
print(f"  {'std':>4s}  {jac_l.std():>7.4f}  {cos_l.std():>7.4f}  {prs_l.std():>7.4f}  "
      f"{spm_l.std():>8.4f}  {kl_l.std():>7.4f}  {mr_l.std():>7.4f}")


# ══════════════════════════════════════════════════════════════════════
# 4. Generation trend — prefill + Q1/Q2/Q3/Q4 per metric per layer
# ══════════════════════════════════════════════════════════════════════
metrics_def = [
    ("Cosine Similarity",      cos_sim,    False, 0.0, 1.0,  "↑"),
    ("Pearson Correlation",    pearson,    False, 0.0, 1.0,  "↑"),
    ("Spearman Correlation",   spearman,   False, 0.0, 1.0,  "↑"),
    ("KL Divergence (ours‖base)", kl,      True,  0.0, None, "↓"),
    ("Mass Ratio (base/ours)", mass_ratio, False, 0.0, None, "≈1"),
]

for label, arr, invert_bar, bar_lo, bar_hi, direction in metrics_def:
    _hdr(f"  {label.upper()}  {direction}  —  Prefill + Generation Quartiles per Layer")

    # determine bar range from data
    gen_arr = arr[GEN_START:GEN_END]
    _bhi = bar_hi if bar_hi is not None else float(np.percentile(gen_arr, 95))
    _blo = bar_lo

    # per-layer quartile arrays [L, 4]
    layer_qs = np.array([_gen_quartiles(arr[:, li]) for li in range(L)])

    print()
    print(f"  {'Lyr':>4}  {'Prefill':>8}  "
          + "  ".join(f"{'Gen-'+q:>8}" for q in qdesc)
          + f"  {'Drift':>6}  {'GenMean':>8}  Bar(gen)")
    print(f"  {'─'*4}  {'─'*8}  "
          + "  ".join("─" * 8 for _ in qdesc)
          + f"  {'─'*6}  {'─'*8}  {'─'*8}")

    for li in range(L):
        pre_v = arr[:PREFILL_STEPS, li].mean() if PREFILL_STEPS else 0.0
        qs    = layer_qs[li]                   # [4]
        drift = qs[-1] - qs[0]
        gen_m = gen_arr[:, li].mean()
        arr_v = gen_m if not invert_bar else -gen_m
        bar   = _bar(arr_v, -_bhi if invert_bar else _blo, 0.0 if invert_bar else _bhi)
        arrow = _drift_arrow(drift if not invert_bar else -drift)
        print(f"  {li:>4d}  {pre_v:>8.4f}  "
              + "  ".join(f"{q:>8.4f}" for q in qs)
              + f"  {drift:>+5.3f}{arrow}  {gen_m:>8.4f}  {bar}")

    # averages row
    pre_avg  = arr[:PREFILL_STEPS].mean()
    qs_avg   = layer_qs.mean(axis=0)
    drift_avg = qs_avg[-1] - qs_avg[0]
    gen_avg  = gen_arr.mean()
    print(f"  {'─'*4}  {'─'*8}  "
          + "  ".join("─" * 8 for _ in qdesc)
          + f"  {'─'*6}  {'─'*8}")
    print(f"  {'avg':>4s}  {pre_avg:>8.4f}  "
          + "  ".join(f"{q:>8.4f}" for q in qs_avg)
          + f"  {drift_avg:>+5.3f}   {gen_avg:>8.4f}")

    # layer standard deviation row
    pre_std  = arr[:PREFILL_STEPS].std()
    qs_std   = layer_qs.std(axis=0)
    gen_std  = gen_arr.std()
    print(f"  {'std':>4s}  {pre_std:>8.4f}  "
          + "  ".join(f"{q:>8.4f}" for q in qs_std)
          + f"  {'':>6}   {gen_std:>8.4f}")


# ══════════════════════════════════════════════════════════════════════
# 5. Layer rankings — best/worst 3 per metric
# ══════════════════════════════════════════════════════════════════════
_hdr("  LAYER RANKINGS  —  Best and Worst 3 Layers per Metric  (generation phase)")

rank_metrics = [
    ("Jaccard",    jac_l,  False),
    ("CosSim",     cos_l,  False),
    ("Pearson",    prs_l,  False),
    ("Spearman",   spm_l,  False),
    ("KL",         kl_l,   True ),   # lower = better
    ("MassRatio",  np.abs(mr_l - 1.0), True),  # |mr-1| lower = better
    ("Composite",  composite, False),
]

K = min(3, L)
print()
print(f"  {'Metric':<12}  "
      f"{'Best-1':>8}  {'Best-2':>8}  {'Best-3':>8}    "
      f"{'Worst-1':>8}  {'Worst-2':>8}  {'Worst-3':>8}")
print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}    {'─'*8}  {'─'*8}  {'─'*8}")

for name, vals, lower_better in rank_metrics:
    if lower_better:
        order = np.argsort(vals)           # ascending → best first
    else:
        order = np.argsort(vals)[::-1]     # descending → best first
    best  = order[:K]
    worst = order[-K:][::-1]

    def _fmt(idx): return f"L{idx}({vals[idx]:.3f})"
    bstrs = [_fmt(i).rjust(8) for i in best]  + ["        "] * (K - len(best))
    wstrs = [_fmt(i).rjust(8) for i in worst] + ["        "] * (K - len(worst))
    print(f"  {name:<12}  {'  '.join(bstrs)}    {'  '.join(wstrs)}")


# ══════════════════════════════════════════════════════════════════════
# 6. Head heterogeneity
# ══════════════════════════════════════════════════════════════════════
_hdr("  HEAD HETEROGENEITY  —  Std across heads at final step  (higher = heads disagree more)")

het_max = het.argmax()
print()
print(f"  Mean: {het.mean():.4f}    Max: {het[het_max]:.4f} at layer {het_max}"
      f"    Min: {het.min():.4f} at layer {het.argmin()}")
print()
cols = 4
rows = (L + cols - 1) // cols
for r in range(rows):
    parts = []
    for c in range(cols):
        li = r + c * rows
        if li < L:
            marker = " ◄MAX" if li == het_max else "     "
            bar = _bar(het[li], 0, het.max())
            parts.append(f"  L{li:<3d} {het[li]:.4f} {bar}{marker}")
    print("".join(parts))


# ══════════════════════════════════════════════════════════════════════
# 6b. Sticky-K policy analytics — Global LIR + absolute missed mass
# ══════════════════════════════════════════════════════════════════════
if "global_lir" in data.files:
    _hdr("  STICKY-K ANALYTICS  —  Global LIR (rescue rate ↓ = stable)  +  Missed Mass (↓ = better)")

    global_lir    = float(data["global_lir"])
    lir_per_layer = data["lir_per_layer"]          # [L]
    lir_per_head  = data["lir_per_head"]           # [L, H]
    missed_mass   = data["missed_mass"]            # [T]
    missed_layer  = data["missed_mass_per_layer"]  # [T, L]
    missed_total  = float(data["missed_mass_total"])
    missed_fresh  = data.get("missed_mass_fresh")  # [T] or None

    print()
    print(f"  Global LIR (Lazy Insertion Rescue) : {global_lir:.4f}   "
          f"(fraction of windows ignored for m flushes that are later rescued)")
    fresh_str = ""
    if missed_fresh is not None:
        fresh_str = (f"   Fresh-K baseline: {float(np.asarray(missed_fresh).mean()):.4f}"
                     f"   Sticky overhead: {missed_total - float(np.asarray(missed_fresh).mean()):+.4f}")
    print(f"  Missed mass (Sticky-K, per-flush mean): {missed_total:.4f}{fresh_str}")

    print()
    print(f"  {'Lyr':>4}  {'LIR':>7}  {'MissedMass(gen)':>15}  {'Head LIR (min/mean/max)':>26}  Bar(LIR)")
    print(f"  {'─'*4}  {'─'*7}  {'─'*15}  {'─'*26}  {'─'*8}")
    for li in range(L):
        mm_gen = missed_layer[GEN_START:GEN_END, li].mean() if N_GEN else missed_layer[:, li].mean()
        hl = lir_per_head[li]
        print(f"  {li:>4d}  {lir_per_layer[li]:>7.4f}  {mm_gen:>15.4f}  "
              f"{hl.min():>7.4f} /{hl.mean():>7.4f} /{hl.max():>7.4f}  {_bar(lir_per_layer[li], 0, 1)}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*15}")
    print(f"  {'avg':>4s}  {lir_per_layer.mean():>7.4f}  {missed_layer.mean():>15.4f}")


# ══════════════════════════════════════════════════════════════════════
# 7. Per-sample breakdown
# ══════════════════════════════════════════════════════════════════════
if "per_sample_jaccard_global" in data.files:
    psjg = data["per_sample_jaccard_global"]   # [S, T]
    S_n  = psjg.shape[0]
    _hdr(f"  PER-SAMPLE BREAKDOWN  ({S_n} sample{'s' if S_n != 1 else ''})")
    print()
    print(f"  {'Sample':>8}  {'Jac overall':>12}  {'Jac prefill':>12}  {'Jac gen':>10}  "
          f"{'Jac drift':>10}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")
    for s in range(S_n):
        sj    = psjg[s]
        pre   = sj[:PREFILL_STEPS].mean()
        gen   = sj[GEN_START:].mean() if T > PREFILL_STEPS else 0.0
        drift = sj[-1] - sj[PREFILL_STEPS] if T > PREFILL_STEPS + 1 else 0.0
        print(f"  {s:>8d}  {sj.mean():>12.4f}  {pre:>12.4f}  {gen:>10.4f}  "
              f"  {drift:>+8.4f}")


# ══════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════
print()
print("═" * W)
print(f"  Source : {NPZ_PATH}")
print(f"  Arrays : {', '.join(data.files)}")
print("═" * W)
print()
