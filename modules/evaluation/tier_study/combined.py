"""Combined tier study — M1 … M6 over one five-run matrix, in order.

Loads the run matrix once (``R0`` … ``R4``), runs the six metric modules in
their numbered order, and writes the combined artifacts beside each module's own
standalone outputs:

``all_metrics.npz``
    every module's arrays, keys prefixed ``m1__``, ``m2__``, … and stored in
    that order (M1's arrays first, then M2's, and so on), plus ``metadata_json``.
``all_metrics.json``
    ``{"metadata": …, "metrics": [ …one ordered entry per module… ]}``.
``all_metrics.md``
    the markdown report, sections in the same order.

The matrix is loaded once for all six metrics, so the npz sha256s, the resolved
geometry and the trace axis in every artifact describe the same load — a
per-module run and a combined run of the same inputs produce the same numbers.

CLI::

    python -m modules.evaluation.tier_study.combined \\
        --r0-npz outputs/parity_base_ws1.npz \\
        --r1-npz outputs/parity_ours_ws1_evictonly.npz \\
        --r2-npz outputs/parity_ours_ws8_evictonly.npz \\
        --r3-npz outputs/parity_ours_ws8_threetier.npz \\
        --r4-npz outputs/parity_ours_ws32_threetier.npz \\
        --output-dir outputs/tier_study --fmm-horizon 32

Or via the single entry point::

    python main.py --config configs/eval_tier_study.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from modules.evaluation.tier_study import (
    CONDITION_IDS,
    SCHEMA_VERSION,
    _json_safe,
    add_matrix_args,
    load_run_matrix,
    matrix_from_args,
    matrix_header_lines,
)
from modules.evaluation.tier_study import m1_future_missed_mass as m1
from modules.evaluation.tier_study import m2_selection_churn as m2
from modules.evaluation.tier_study import m3_tier_mass_distribution as m3
from modules.evaluation.tier_study import m4_jaccard_topk_overlap as m4
from modules.evaluation.tier_study import m5_qtier_fidelity as m5
from modules.evaluation.tier_study import m6_recovered_mass_q as m6
from utils.logger import get_logger

log = get_logger(__name__)

#: The numbered order every combined artifact preserves.
MODULES = (m1, m2, m3, m4, m5, m6)

DEFAULTS: Dict[str, Any] = {
    "fmm_horizon": m1.DEFAULTS["horizon"],
    "sticky_primary": False,
}


def _module_kwargs(mod, *, fmm_horizon: int, sticky_primary: bool,
                   confidence: float, n_boot: int, seed: int,
                   per_head: bool) -> Dict[str, Any]:
    """Per-module compute kwargs (the modules do not share one signature)."""
    common = {"confidence": confidence, "n_boot": n_boot, "per_head": per_head}
    # Distinct seed per module so the bootstraps are independent but reproducible.
    offset = 1000 * (MODULES.index(mod) + 1)
    if mod is m1:
        return {"horizon": fmm_horizon, "seed": seed + offset, **common}
    if mod is m6:
        return {"horizon": fmm_horizon, "sticky": sticky_primary,
                "seed": seed + offset, **common}
    return {"seed": seed + offset, **common}


def run_all(
    matrix,
    *,
    fmm_horizon: int = DEFAULTS["fmm_horizon"],
    sticky_primary: bool = DEFAULTS["sticky_primary"],
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    per_head: bool = True,
) -> List[Dict[str, Any]]:
    """Run M1 … M6 in order over one loaded matrix."""
    results: List[Dict[str, Any]] = []
    for mod in MODULES:
        t0 = time.time()
        log.info("=== %s (%s) ===", mod.METRIC_TITLE, mod.METRIC_QUESTION)
        kwargs = _module_kwargs(
            mod, fmm_horizon=fmm_horizon, sticky_primary=sticky_primary,
            confidence=confidence, n_boot=n_boot, seed=seed, per_head=per_head)
        res = mod.compute(matrix, **kwargs)
        res["elapsed_sec"] = round(time.time() - t0, 2)
        res["module"] = mod
        results.append(res)
        log.info("%s done in %.1fs", mod.METRIC_ID, res["elapsed_sec"])
    return results


def _payload(res: Dict[str, Any]) -> Dict[str, Any]:
    """One metric's JSON entry — everything except the raw arrays."""
    return {k: v for k, v in res.items()
            if k not in ("arrays", "module", "report")}


def build_report(matrix, results: Sequence[Dict[str, Any]], meta: Dict[str, Any]
                 ) -> str:
    """The combined markdown report, sections in module order."""
    lines = [
        "# RecoveryKV tier study — six observation metrics over five runs",
        "",
        "Two questions, one run matrix: **why decide over a window** (M1, M2) "
        "and **why three tiers** (M3–M6). Every condition below is a separate "
        "recorded run — a flush cadence changes cache state divergently, so no "
        "condition here is simulated from another.",
        "",
        *matrix_header_lines(matrix, list(CONDITION_IDS)),
        "",
        "| metric | question | runs read |",
        "| --- | --- | --- |",
    ]
    for res in results:
        lines.append(
            f"| {res['title']} | {res['question']} | "
            f"{', '.join('`' + c + '`' for c in res['runs_used'])} |")
    lines += ["", "---", ""]
    for res in results:
        lines += [res["report"], ""]
    diag = matrix.diagnostics
    lines += [
        "---",
        "",
        "## Reading notes",
        "",
        f"- Baseline mass source: `{diag.get('baseline_mass_source')}`"
        + ("" if diag.get("baseline_mass_source") == "recorded_step_mass" else
           " — the base npz predates schema 1.2, so the per-step mass was "
           "differenced from fp16 cumulative scores; re-run BaseParityRunner "
           "before reporting M1 or M6(b)."),
        f"- R0 was recorded at `window_size={matrix.base_window_size}` and "
        "re-bucketed per condition (`pool_factor` in the geometry block). R0 "
        "never evicts, so a coarse bucket is the sum of the fine ones inside it "
        "(up to fp16 storage of the recorded scores) — this is re-bucketing of "
        "one ground truth, not a substituted run.",
        "- Each ours run was teacher-forced from a base run at its own "
        "`window_size` (the parity validator requires that), so R0's sha will "
        "differ from the base npz behind R2/R3/R4. Those base runs are the same "
        "greedy decode; the loader verified that all five npzs carry an "
        "identical forced-token stream.",
        "- A metric that is undefined (no eligible flush, no full horizon, a "
        "policy that dropped nothing) is reported as `NA`, never as 0.",
        f"- Written: {meta['run_finished_utc']}.",
        "",
    ]
    return "\n".join(lines)


def write_combined(matrix, results: Sequence[Dict[str, Any]], out_dir: Path
                   ) -> Dict[str, Path]:
    """Write ``all_metrics.{npz,json,md}``, M1's data first, then M2's, …"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        **matrix.metadata_block(list(CONDITION_IDS)),
        "schema_version": SCHEMA_VERSION,
        "mode": "tier_study",
        "metrics": [
            {"id": r["metric_id"], "name": r["metric_name"], "title": r["title"],
             "question": r["question"], "runs_used": r["runs_used"],
             "knobs": r["knobs"], "elapsed_sec": r["elapsed_sec"]}
            for r in results
        ],
    }

    arrays: Dict[str, np.ndarray] = {}
    for res in results:
        prefix = f"{res['metric_id']}__"
        for key, arr in res["arrays"].items():
            arrays[prefix + key] = np.asarray(arr)
    npz_path = out_dir / "all_metrics.npz"
    np.savez_compressed(
        str(npz_path), **arrays,
        metadata_json=np.array([json.dumps(_json_safe(meta))], dtype=object))

    json_path = out_dir / "all_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe({"metadata": meta,
                              "metrics": [_payload(r) for r in results]}),
                  f, indent=2)

    report = build_report(matrix, results, meta)
    md_path = out_dir / "all_metrics.md"
    md_path.write_text(report, encoding="utf-8")

    # Sidecar, matching every other suite's output convention.
    with open(npz_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, indent=2, default=str)

    log.info("Combined tier study written to %s", out_dir.resolve())
    return {"npz": npz_path, "json": json_path, "md": md_path}


def run_study(
    paths: Dict[str, Any],
    out_dir: Path,
    *,
    fmm_horizon: int = DEFAULTS["fmm_horizon"],
    sticky_primary: bool = DEFAULTS["sticky_primary"],
    trace_axis: str = "sample_layer",
    max_samples: Optional[int] = None,
    layer_stride: int = 1,
    head_stride: int = 1,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    per_head: bool = True,
    module_outputs: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """End-to-end: load the matrix, run M1 … M6, write everything."""
    matrix = load_run_matrix(
        paths, required=CONDITION_IDS, trace_axis=trace_axis,
        max_samples=max_samples, layer_stride=layer_stride,
        head_stride=head_stride, keep_ours_scores=True, strict=strict)
    results = run_all(matrix, fmm_horizon=fmm_horizon,
                      sticky_primary=sticky_primary, confidence=confidence,
                      n_boot=n_boot, seed=seed, per_head=per_head)
    if module_outputs:
        for res in results:
            res["module"].write(matrix, res, out_dir)
    written = write_combined(matrix, results, out_dir)
    return {"matrix": matrix, "results": results, "written": written}


class TierStudyRunner:
    """``python main.py --config configs/eval_tier_study.yaml``.

    Reads the five npz paths from the ``tier_study:`` config section and writes
    under ``telemetry.output_dir/tier_study`` (or ``output_path``).  Analysis
    knobs come from the config when set and from the module defaults otherwise —
    the same wiring ``QEvictObservationRunner`` uses.
    """

    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        ts = getattr(cfg, "tier_study", None)
        if ts is None:
            raise ValueError(
                "config has no `tier_study:` section — it needs r0_npz … r4_npz "
                "(see configs/eval_tier_study.yaml).")
        paths = {cid: getattr(ts, f"{cid.lower()}_npz", "")
                 for cid in CONDITION_IDS}
        missing = [c for c, p in paths.items() if not p]
        if missing:
            raise ValueError(
                f"tier_study config is missing npz path(s) for {missing}. Each "
                "condition is a separate recorded run.")
        out_dir = Path(cfg.output_path) if cfg.output_path else (
            Path(cfg.telemetry.output_dir) / "tier_study")
        res = run_study(
            paths, out_dir,
            fmm_horizon=int(getattr(ts, "fmm_horizon", DEFAULTS["fmm_horizon"])),
            sticky_primary=bool(getattr(ts, "sticky_primary", False)),
            trace_axis=str(getattr(ts, "trace_axis", "sample_layer")),
            max_samples=getattr(ts, "max_samples", None),
            layer_stride=int(getattr(ts, "layer_stride", 1)),
            head_stride=int(getattr(ts, "head_stride", 1)),
            confidence=float(getattr(ts, "confidence", 0.95)),
            n_boot=int(getattr(ts, "bootstrap_samples", 2000)),
            seed=int(getattr(cfg.run, "seed", 0)),
            per_head=bool(getattr(ts, "per_head", True)),
        )
        return res["written"]["npz"]


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Combined RecoveryKV tier study (M1 … M6).")
    add_matrix_args(ap, CONDITION_IDS)
    ap.add_argument("--fmm-horizon", type=int, default=DEFAULTS["fmm_horizon"],
                    help="H for M1 and M6(b).")
    ap.add_argument("--sticky-primary", action="store_true",
                    help="M6: report Sticky-K as the primary simulated policy.")
    ap.add_argument("--no-module-outputs", action="store_true",
                    help="Write only the combined artifacts (skip each "
                         "module's own mN_<name>.{csv,json,npz}).")
    args = ap.parse_args(argv)

    matrix = matrix_from_args(args, CONDITION_IDS, keep_ours_scores=True)
    results = run_all(
        matrix, fmm_horizon=args.fmm_horizon,
        sticky_primary=args.sticky_primary, confidence=args.confidence,
        n_boot=args.bootstrap_samples, seed=args.seed,
        per_head=not args.no_per_head)
    if not args.no_module_outputs:
        for res in results:
            res["module"].write(matrix, res, args.output_dir)
    written = write_combined(matrix, results, args.output_dir)
    print(Path(written["md"]).read_text(encoding="utf-8"))
    print(f"\nSaved all outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
