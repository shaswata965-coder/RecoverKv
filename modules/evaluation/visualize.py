"""Visualization — 8 plot types for evaluation results.

Each plot is independently callable via make_<name>(). VisualizeRunner
orchestrates all plots from config-specified npz paths. Gracefully
degrades: falls back to pure matplotlib if seaborn unavailable.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from utils.config import ExperimentConfig
from utils.logger import get_logger

log = get_logger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns
    sns.set_context("paper")
    sns.set_style("whitegrid")
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    if HAS_MPL:
        warnings.warn("seaborn unavailable — using pure matplotlib", RuntimeWarning)

# Consistent colors
COLORS = {
    "baseline": "#888888",
    "windowed_50pct": "#2196F3",
    "windowed_25pct": "#1565C0",
    "windowed_10pct": "#0D47A1",
    "baseline_flash": "#BDBDBD",
    "windowed_flash_50pct": "#FF9800",
    "windowed_flash_25pct": "#F57C00",
    "windowed_flash_10pct": "#E65100",
}

def _get_color(name: str) -> str:
    for key, color in COLORS.items():
        if key in name.lower(): return color
    return "#333333"

def _save_fig(fig, out_dir: Path, name: str, dpi: int = 300, save_pdf: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        try: fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        except Exception: pass
    plt.close(fig)
    log.info("Saved %s", out_dir / f"{name}.png")

def _load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    arrays = {k: data[k] for k in data.files if k != "metadata_json"}
    meta = {}
    if "metadata_json" in data.files:
        meta = json.loads(str(data["metadata_json"][0]))
    return {"arrays": arrays, "metadata": meta}

# --- Plot 1: Jaccard trajectory ---
def make_jaccard_trajectory(npz_paths: List[Path], out_dir: Path,
                            dpi: int = 300, save_pdf: bool = False) -> None:
    if not HAS_MPL: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for p in npz_paths:
        d = _load_npz(str(p))
        if "jaccard_global" not in d["arrays"]: continue
        jg = d["arrays"]["jaccard_global"]
        jpl = d["arrays"].get("jaccard_per_layer")
        name = d["metadata"].get("ours_npz_path", str(p))
        stem = Path(name).stem
        # Panel 1: Global line.  With a Q tier, overlay fp-only vs fp∪Q and shade
        # the lift between them — the Jaccard the Q tier buys (schema ≥ 2.2).
        jfp_g   = d["arrays"].get("jaccard_fp_global")
        jkept_g = d["arrays"].get("jaccard_kept_global")
        if jfp_g is not None and jkept_g is not None and not np.allclose(jfp_g, jkept_g):
            steps = range(len(jkept_g))
            axes[0].plot(jkept_g, linewidth=1.6, label=f"{stem} (fp∪Q kept)")
            axes[0].plot(jfp_g, linewidth=1.1, linestyle="--", alpha=0.8,
                         label=f"{stem} (fp only)")
            axes[0].fill_between(steps, jfp_g, jkept_g, alpha=0.15,
                                 label=f"{stem} (Q lift)")
        else:
            axes[0].plot(jg, label=stem, linewidth=1.5)
        axes[0].set_xlabel("Generation Step"); axes[0].set_ylabel("Jaccard")
        axes[0].set_title("Global Jaccard Trajectory"); axes[0].legend(fontsize=7)
        # Panel 2: Per-layer heatmap
        if jpl is not None and jpl.ndim == 2:
            im = axes[1].imshow(jpl.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            axes[1].set_xlabel("Step"); axes[1].set_ylabel("Layer")
            axes[1].set_title("Per-Layer Jaccard"); plt.colorbar(im, ax=axes[1])
        # Panel 3: Per-head violin at 3 time slices
        j_full = d["arrays"].get("jaccard")
        if j_full is not None and j_full.ndim >= 2:
            S = j_full.shape[0]
            slices = [0, S//2, S-1]
            data_v = [j_full[s].flatten() for s in slices if s < S]
            if data_v:
                axes[2].violinplot(data_v, showmeans=True)
                axes[2].set_xticks(range(1, len(data_v)+1))
                axes[2].set_xticklabels([f"t={s}" for s in slices[:len(data_v)]])
                axes[2].set_ylabel("Jaccard"); axes[2].set_title("Per-Head Distribution")
    fig.tight_layout()
    _save_fig(fig, out_dir, "jaccard_trajectory", dpi, save_pdf)

# --- Plot 2: LIR (Lazy Insertion Rescue) ---
def make_lir_trajectory(npz_paths: List[Path], out_dir: Path,
                        dpi: int = 300, save_pdf: bool = False) -> None:
    """Global LIR (bar) + per-layer LIR (line) + per-(layer,head) LIR heatmap.

    ``global_lir`` is a scalar, ``lir_per_layer`` is ``[L]`` and
    ``lir_per_head`` is ``[L, H]`` — the Sticky-K rescue-rate granularities
    emitted by the faithfulness runner (schema ≥ v2.1).
    """
    if not HAS_MPL: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for p in npz_paths:
        d = _load_npz(str(p))
        gl = d["arrays"].get("global_lir")
        lpl = d["arrays"].get("lir_per_layer")
        lph = d["arrays"].get("lir_per_head")
        if gl is None and lpl is None: continue
        label = Path(str(p)).stem
        # Panel 1: global LIR scalar as a bar
        if gl is not None:
            axes[0].bar(label, float(gl), alpha=0.8)
        axes[0].set_ylabel("Global LIR"); axes[0].set_ylim(0, 1)
        axes[0].set_title("Global LIR (rescue rate)")
        axes[0].tick_params(axis="x", rotation=45)
        # Panel 2: per-layer LIR
        if lpl is not None and lpl.ndim == 1:
            axes[1].plot(lpl, marker="o", markersize=3, linewidth=1.2, label=label)
            axes[1].set_xlabel("Layer"); axes[1].set_ylabel("LIR")
            axes[1].set_title("Per-Layer LIR"); axes[1].legend(fontsize=7)
            axes[1].set_ylim(0, 1)
        # Panel 3: per-(layer, head) LIR heatmap (last file wins)
        if lph is not None and lph.ndim == 2:
            im = axes[2].imshow(lph.T, aspect="auto", cmap="magma", vmin=0, vmax=1)
            axes[2].set_xlabel("Layer"); axes[2].set_ylabel("Head")
            axes[2].set_title("Per-Head LIR"); plt.colorbar(im, ax=axes[2])
    fig.tight_layout()
    _save_fig(fig, out_dir, "lir_trajectory", dpi, save_pdf)

# --- Plot 3: Missed mass trajectory ---
def make_missed_mass_distribution(npz_paths: List[Path], out_dir: Path,
                                   dpi: int = 300, save_pdf: bool = False) -> None:
    """Missed-mass trajectories with the Q tier's rescued-mass band.

    ``missed_mass`` (fp-only drop) and ``missed_mass_kept`` (two-tier) are
    ``[T]`` (lower = the retained set captures more of the true attention); the
    shaded gap between them is ``recovered_mass_q`` — the mass the int2 Q tier
    rescues that an fp-only drop would have lost.  ``missed_mass_per_layer`` is
    ``[T, L]`` and is shown as a heatmap.
    """
    if not HAS_MPL: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for p in npz_paths:
        d = _load_npz(str(p))
        mm = d["arrays"].get("missed_mass")
        mmf = d["arrays"].get("missed_mass_fresh")
        mmk = d["arrays"].get("missed_mass_kept")
        mmpl = d["arrays"].get("missed_mass_per_layer")
        if mm is None: continue
        stem = Path(str(p)).stem
        axes[0].plot(mm, linewidth=1.5, label=f"{stem} (fp-only drop)")
        # Q tier credited: shade the mass it rescues (schema ≥ 2.2).
        if mmk is not None and not np.allclose(mmk, mm):
            axes[0].plot(mmk, linewidth=1.6, color="C2",
                         label=f"{stem} (fp∪Q kept)")
            axes[0].fill_between(range(len(mm)), mmk, mm, alpha=0.18, color="C2",
                                 label=f"{stem} (Q rescued)")
        if mmf is not None:
            axes[0].plot(mmf, linewidth=1.0, linestyle="--", alpha=0.7,
                         label=f"{stem} (Fresh-K)")
        if mmpl is not None and mmpl.ndim == 2:
            im = axes[1].imshow(mmpl.T, aspect="auto", cmap="viridis")
            axes[1].set_xlabel("Flush (step)"); axes[1].set_ylabel("Layer")
            axes[1].set_title("Per-Layer Missed Mass"); plt.colorbar(im, ax=axes[1])
    axes[0].set_xlabel("Flush (step)"); axes[0].set_ylabel("Missed Mass")
    axes[0].set_title("Missed Mass — fp-only vs two-tier (Q rescued)")
    axes[0].legend(fontsize=7)
    _save_fig(fig, out_dir, "missed_mass_distribution", dpi, save_pdf)

# --- Plot 4: KL divergence heatmap ---
def make_kl_divergence_heatmap(npz_paths: List[Path], out_dir: Path,
                                dpi: int = 300, save_pdf: bool = False) -> None:
    if not HAS_MPL: return
    for p in npz_paths:
        d = _load_npz(str(p))
        jpl = d["arrays"].get("jaccard_per_layer")
        if jpl is None: continue
        fig, ax = plt.subplots(figsize=(12, 6))
        # Use 1-jaccard as proxy for divergence
        im = ax.imshow((1-jpl).T, aspect="auto", cmap="Reds")
        ax.set_xlabel("Step"); ax.set_ylabel("Layer")
        ax.set_title("Divergence Heatmap (1 - Jaccard)"); plt.colorbar(im, ax=ax)
        _save_fig(fig, out_dir, f"kl_heatmap_{Path(str(p)).stem}", dpi, save_pdf)

# --- Plot 5: Budget sweep ---
def make_budget_sweep(npz_paths: List[Path], out_dir: Path,
                      dpi: int = 300, save_pdf: bool = False) -> None:
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(8, 6))
    budgets, lirs = [], []
    for p in npz_paths:
        d = _load_npz(str(p))
        b = d["metadata"].get("cache_budget")
        gl = d["arrays"].get("global_lir")
        if b is not None and gl is not None:
            budgets.append(b); lirs.append(gl.mean())
    if budgets:
        ax.semilogx(budgets, lirs, "o-", markersize=8, linewidth=2)
        ax.set_xlabel("Cache Budget"); ax.set_ylabel("Mean LIR")
        ax.set_title("Cache Budget Sweep"); ax.grid(True, alpha=0.3)
    _save_fig(fig, out_dir, "budget_sweep", dpi, save_pdf)

# --- Plot 6: Perf summary ---
def make_perf_summary(npz_paths: List[Path], out_dir: Path,
                      dpi: int = 300, save_pdf: bool = False) -> None:
    if not HAS_MPL: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for p in npz_paths:
        d = _load_npz(str(p))
        names = d["arrays"].get("config_names")
        skip = d["arrays"].get("skipped_mask")
        if names is None: continue
        ttft = d["arrays"].get("ttft_ms")
        tpot_arr = d["arrays"].get("tpot_ms")
        mem = d["arrays"].get("peak_memory_mb")
        pf = d["metadata"].get("prefill_len", "?")
        for ci in range(len(names)):
            if skip is not None and skip[ci]: continue
            nm = str(names[ci])
            col = _get_color(nm)
            if ttft is not None:
                med = np.nanmedian(ttft[ci])
                axes[0].bar(nm, med, color=col, alpha=0.8)
            if tpot_arr is not None:
                med = np.nanmedian(tpot_arr[ci])
                axes[1].bar(nm, med, color=col, alpha=0.8)
            if mem is not None:
                med = np.nanmedian(mem[ci])
                axes[2].bar(nm, med, color=col, alpha=0.8)
    axes[0].set_title("TTFT (ms)"); axes[0].tick_params(axis="x", rotation=45)
    axes[1].set_title("TPOT (ms)"); axes[1].tick_params(axis="x", rotation=45)
    axes[2].set_title("Peak Memory (MB)"); axes[2].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save_fig(fig, out_dir, "perf_summary", dpi, save_pdf)

# --- Plot 7: Hook overhead bar ---
def make_hook_overhead_bar(npz_paths: List[Path], out_dir: Path,
                           dpi: int = 300, save_pdf: bool = False) -> None:
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in npz_paths:
        d = _load_npz(str(p))
        names = d["arrays"].get("config_names")
        tpot_arr = d["arrays"].get("tpot_ms")
        skip = d["arrays"].get("skipped_mask")
        if names is None or tpot_arr is None: continue
        # Find baseline
        base_tpot = None
        for ci in range(len(names)):
            nm = str(names[ci])
            if "baseline" in nm.lower() and "hook" not in nm.lower():
                if skip is None or not skip[ci]:
                    base_tpot = np.nanmedian(tpot_arr[ci]); break
        if base_tpot is None or base_tpot == 0: continue
        for ci in range(len(names)):
            if skip is not None and skip[ci]: continue
            nm = str(names[ci])
            med = np.nanmedian(tpot_arr[ci])
            normalized = med / base_tpot
            ax.bar(nm, normalized, color=_get_color(nm), alpha=0.8)
            ax.annotate(f"{med:.1f}ms", (nm, normalized), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("TPOT (normalized)"); ax.set_title("Hook Overhead")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save_fig(fig, out_dir, "hook_overhead", dpi, save_pdf)

# --- Plot 8: Top-K window age histogram ---
def make_topk_window_age_histogram(npz_paths: List[Path], out_dir: Path,
                                   dpi: int = 300, save_pdf: bool = False) -> None:
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in npz_paths:
        d = _load_npz(str(p))
        topk = d["arrays"].get("top_window_indices")
        if topk is None: continue
        # Pull units from metadata: a window covers `window_size` tokens,
        # and the first `num_sink` tokens are not represented in the
        # window grid. Without these, `t - window_id` is a unit-mismatched
        # quantity (steps minus windows).
        meta = d.get("metadata", {}) or {}
        window_size = int(meta.get("window_size", 1))
        num_sink = int(meta.get("num_sink_tokens", 0))
        prefill_len = int(meta.get("prefill_len", 0))
        ages = []
        # Schema v1.0: [num_steps, num_layers, K]
        # Schema v1.1: [num_samples, num_steps, num_layers, K]
        if topk.ndim == 4:
            num_samples, num_steps = topk.shape[0], topk.shape[1]
            for s in range(num_samples):
                for t in range(num_steps):
                    valid = topk[s, t][topk[s, t] >= 0]
                    if len(valid) > 0:
                        # Age in TOKENS = current absolute token position
                        # (post-sink) minus the window's first-token position.
                        age = (prefill_len + t - num_sink) - valid.astype(float) * window_size
                        ages.extend(age.tolist())
        else:
            num_steps = topk.shape[0]
            for t in range(num_steps):
                valid = topk[t][topk[t] >= 0]
                if len(valid) > 0:
                    age = (prefill_len + t - num_sink) - valid.astype(float) * window_size
                    ages.extend(age.tolist())
        if ages:
            ax.hist(ages, bins=50, alpha=0.6, label=Path(str(p)).stem)
    ax.set_xlabel("Window Age (tokens)"); ax.set_ylabel("Count")
    ax.set_title("Retained Window Age Distribution"); ax.legend()
    _save_fig(fig, out_dir, "topk_window_age", dpi, save_pdf)

class VisualizeRunner:
    """Orchestrates all 8 plot types."""
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> None:
        if not HAS_MPL:
            log.error("matplotlib not available — cannot generate plots")
            return
        vc = self.config.visualize
        out_dir = Path(vc.output_dir)
        dpi = vc.dpi
        pdf = vc.save_pdf
        # Collect npz paths
        all_paths = [Path(p) for p in vc.npz_paths if Path(p).exists()]
        parity_paths = [p for p in all_paths if "parity" in p.stem or "faithfulness" in p.stem]
        faith_paths = [Path(vc.faithfulness_npz)] if vc.faithfulness_npz and Path(vc.faithfulness_npz).exists() else []
        perf_paths = list(Path(vc.perf_npz_dir).glob("perf_prefill*.npz")) if Path(vc.perf_npz_dir).exists() else []
        ours_paths = [p for p in all_paths if "ours" in p.stem]
        log.info("Generating plots to %s", out_dir)
        if faith_paths or parity_paths:
            make_jaccard_trajectory(faith_paths or parity_paths, out_dir, dpi, pdf)
            make_lir_trajectory(faith_paths or parity_paths, out_dir, dpi, pdf)
            make_missed_mass_distribution(faith_paths or parity_paths, out_dir, dpi, pdf)
            make_kl_divergence_heatmap(faith_paths or parity_paths, out_dir, dpi, pdf)
        if ours_paths:
            make_budget_sweep(ours_paths, out_dir, dpi, pdf)
            make_topk_window_age_histogram(ours_paths, out_dir, dpi, pdf)
        if perf_paths:
            make_perf_summary(perf_paths, out_dir, dpi, pdf)
            make_hook_overhead_bar(perf_paths, out_dir, dpi, pdf)
        log.info("Visualization complete")
