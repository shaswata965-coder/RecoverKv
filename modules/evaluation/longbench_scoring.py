"""LongBench scoring pipeline — post-hoc metric computation.

Reads ``<dataset>.jsonl`` prediction files, applies the dataset-specific
metric from ``dataset2metric.json``, and produces a CSV comparison table.

Scoring is independent of the model — re-runnable in seconds.

Adapted from THUDM/LongBench/LongBench/eval.py.  The metric dispatch
follows their ``dataset2metric`` mapping exactly; per-example score is
``max(metric(pred, gt) for gt in answers)`` (THUDM convention, inherited
by DefensiveKV).

Can be invoked as:
    python -m modules.evaluation.longbench_scoring --predictions_dir <dir> --out_csv <path>
Or via main.py with mode=longbench_score.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from data.longbench_loader import TASK_CATEGORIES
from modules.evaluation.longbench_metrics import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Metric function registry (mirrors THUDM/LongBench eval.py exactly)
# ---------------------------------------------------------------------------

METRIC_FN_REGISTRY = {
    "qa_f1_score": qa_f1_score,
    "qa_f1_zh_score": qa_f1_zh_score,
    "rouge_score": rouge_score,
    "rouge_zh_score": rouge_zh_score,
    "classification_score": classification_score,
    "retrieval_score": retrieval_score,
    "retrieval_zh_score": retrieval_zh_score,
    "count_score": count_score,
    "code_sim_score": code_sim_score,
}

# Datasets where THUDM's eval.py applies first-line extraction in the scorer
# (see eval.py scorer() function).
_FIRST_LINE_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}


def _load_dataset2metric() -> Dict[str, str]:
    """Load the vendored dataset→metric mapping."""
    config_path = Path("data/longbench_configs/dataset2metric.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Vendored config not found: {config_path}. "
            "Ensure data/longbench_configs/ is populated."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_predictions(
    predictions_dir: Path,
    out_csv: Optional[Path] = None,
) -> Dict[str, float]:
    """Score all ``<dataset>.jsonl`` files in *predictions_dir*.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing per-dataset ``.jsonl`` prediction files.
    out_csv : Path, optional
        If provided, write a CSV with columns: dataset, num_examples, score.

    Returns
    -------
    dict[str, float]
        ``{dataset_name: score}`` where score is percentage (0–100).
    """
    dataset2metric = _load_dataset2metric()
    results: Dict[str, float] = {}
    details: List[Dict[str, Any]] = []

    for jsonl in sorted(predictions_dir.glob("*.jsonl")):
        name = jsonl.stem
        if name not in dataset2metric:
            log.warning("No metric mapping for %s — skipping", name)
            continue

        metric_name = dataset2metric[name]
        if metric_name not in METRIC_FN_REGISTRY:
            log.warning(
                "Unknown metric function %s for dataset %s — skipping",
                metric_name,
                name,
            )
            continue

        metric_fn = METRIC_FN_REGISTRY[metric_name]
        total = 0.0
        n = 0
        n_skipped = 0

        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)

            # Every example in the split contributes to the denominator (matches
            # THUDM eval.py, which divides by len(predictions)). A null/failed
            # prediction (e.g. skip_oom recorded pred=null) scores 0 rather than
            # being dropped — dropping it would silently inflate the mean and
            # break comparability with the published numbers.
            n += 1
            pred = ex.get("pred")
            if pred is None:
                n_skipped += 1
                # best = 0.0; nothing to add to `total`.
                continue

            # Apply first-line extraction for specific datasets
            # (matches THUDM eval.py scorer() exactly)
            if name in _FIRST_LINE_DATASETS:
                pred = pred.lstrip("\n").split("\n")[0]

            answers = ex.get("answers", [])
            # `classification_score` iterates over `all_classes`; a missing or
            # null field would crash with `'NoneType' is not iterable`.
            all_classes = ex.get("all_classes") or []

            # Per-example score = max over ground truths (THUDM convention).
            # `default=0.0` guards the (data-malformed) empty-answers case, which
            # would otherwise raise ValueError on max() of an empty iterable —
            # THUDM initialises score=0 before the loop, so this matches.
            best = max(
                (metric_fn(pred, gt, all_classes=all_classes) for gt in answers),
                default=0.0,
            )
            total += best

        if n_skipped > 0:
            log.warning(
                "%s: %d examples had pred=null (likely OOM) — scored 0 and "
                "counted in the denominator (matches THUDM eval.py)",
                name,
                n_skipped,
            )

        score = (total / n) * 100 if n > 0 else float("nan")
        results[name] = round(score, 2)
        details.append({"dataset": name, "num_examples": n, "score": score})
        log.info(
            "%-25s  n=%-4d  skipped=%-3d  score=%.2f",
            name,
            n,
            n_skipped,
            score,
        )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "num_examples", "score"])
            writer.writeheader()
            writer.writerows(details)
        log.info("Scores written to %s", out_csv)

    return results


def compute_category_averages(
    scores: Dict[str, float],
) -> Dict[str, float]:
    """Compute per-task-category averages (matches DefensiveKV Figure 5).

    Parameters
    ----------
    scores : dict[str, float]
        Per-dataset scores from ``score_predictions``.

    Returns
    -------
    dict[str, float]
        ``{category_name: average_score}``.
    """
    cat_avgs: Dict[str, float] = {}
    for cat_name, datasets in TASK_CATEGORIES.items():
        cat_scores = [scores[d] for d in datasets if d in scores and not np.isnan(scores[d])]
        if cat_scores:
            cat_avgs[cat_name] = round(float(np.mean(cat_scores)), 2)
        else:
            cat_avgs[cat_name] = float("nan")
    return cat_avgs


def compute_macro_average(scores: Dict[str, float]) -> float:
    """Compute overall macro average across all datasets.

    NaN scores (missing datasets) are excluded with a warning.
    """
    valid = [v for v in scores.values() if not np.isnan(v)]
    if len(valid) < len(scores):
        log.warning(
            "Macro average computed over %d/%d datasets (some missing/NaN)",
            len(valid),
            len(scores),
        )
    return round(float(np.mean(valid)), 2) if valid else float("nan")


def compute_relative_degradation(
    baseline_scores: Dict[str, float],
    variant_scores: Dict[str, float],
) -> float:
    """Compute relative degradation vs baseline.

    ``(baseline_avg - variant_avg) / baseline_avg * 100``

    The "X% drop" formulation from DefensiveKV Table 1.
    """
    baseline_avg = compute_macro_average(baseline_scores)
    variant_avg = compute_macro_average(variant_scores)
    if np.isnan(baseline_avg) or baseline_avg == 0:
        return float("nan")
    return round((baseline_avg - variant_avg) / baseline_avg * 100, 2)


def build_comparison_table(
    run_dirs: List[Path],
    out_path: Optional[Path] = None,
) -> str:
    """Build a comparison table across multiple runs.

    Parameters
    ----------
    run_dirs : list of Path
        Directories containing scored prediction runs.
    out_path : Path, optional
        Write CSV + Markdown to this path (+ .md for markdown).

    Returns
    -------
    str
        Markdown-formatted comparison table.
    """
    all_scores: Dict[str, Dict[str, float]] = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        csv_path = run_dir / "scores.csv"
        if csv_path.exists():
            scores = {}
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scores[row["dataset"]] = float(row["score"])
            all_scores[run_dir.name] = scores
        else:
            # Score from jsonls
            scores = score_predictions(run_dir, csv_path)
            all_scores[run_dir.name] = scores

    if not all_scores:
        return "No runs found to compare."

    # Collect all dataset names
    all_datasets = sorted(
        set(d for scores in all_scores.values() for d in scores.keys())
    )
    run_names = list(all_scores.keys())

    # Build markdown table
    lines = []
    header = "| Dataset | " + " | ".join(run_names) + " |"
    sep = "|---|" + "|".join(["---"] * len(run_names)) + "|"
    lines.append(header)
    lines.append(sep)

    for ds in all_datasets:
        row = f"| {ds} |"
        for rn in run_names:
            val = all_scores[rn].get(ds, float("nan"))
            row += f" {val:.2f} |" if not np.isnan(val) else " — |"
        lines.append(row)

    # Category averages
    lines.append("|---|" + "|".join(["---"] * len(run_names)) + "|")
    for cat_name in TASK_CATEGORIES:
        row = f"| **{cat_name}** |"
        for rn in run_names:
            cat_avg = compute_category_averages(all_scores[rn]).get(cat_name, float("nan"))
            row += f" {cat_avg:.2f} |" if not np.isnan(cat_avg) else " — |"
        lines.append(row)

    # Macro average
    lines.append("|---|" + "|".join(["---"] * len(run_names)) + "|")
    row = "| **Overall** |"
    for rn in run_names:
        macro = compute_macro_average(all_scores[rn])
        row += f" **{macro:.2f}** |" if not np.isnan(macro) else " — |"
    lines.append(row)

    md_table = "\n".join(lines)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write CSV
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset"] + run_names)
            for ds in all_datasets:
                writer.writerow(
                    [ds] + [all_scores[rn].get(ds, "") for rn in run_names]
                )
        # Write Markdown
        md_path = out_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# LongBench Comparison Table\n\n")
            f.write(md_table)
            f.write("\n")
        log.info("Comparison written to %s and %s", out_path, md_path)

    return md_table


# ---------------------------------------------------------------------------
# LongBenchScorer — runnable via main.py with mode=longbench_score
# ---------------------------------------------------------------------------


class LongBenchScorer:
    """Post-hoc scorer, invoked via ``main.py --config ... --override run.mode=longbench_score``."""

    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> None:
        longbench_cfg = getattr(self.config, "longbench", None)

        # Score individual runs
        base_dir = Path("outputs/longbench")
        if not base_dir.exists():
            log.error("No outputs/longbench/ directory found")
            return

        run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        if not run_dirs:
            log.error("No run directories found in %s", base_dir)
            return

        # Score each run
        for run_dir in run_dirs:
            jsonls = list(run_dir.glob("*.jsonl"))
            if jsonls:
                csv_path = run_dir / "scores.csv"
                log.info("Scoring %s (%d datasets)...", run_dir.name, len(jsonls))
                score_predictions(run_dir, csv_path)

        # Build comparison table
        md = build_comparison_table(run_dirs, base_dir / "comparison.csv")
        print("\n" + md + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="Score LongBench predictions")
    parser.add_argument(
        "--predictions_dir",
        type=str,
        help="Directory with <dataset>.jsonl files",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline scores.csv for comparison",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=[],
        help="Variant scores.csv files for comparison",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output comparison CSV path",
    )
    args = parser.parse_args()

    if args.predictions_dir:
        predictions_dir = Path(args.predictions_dir)
        out_csv = Path(args.out_csv) if args.out_csv else None
        scores = score_predictions(predictions_dir, out_csv)
        macro = compute_macro_average(scores)
        print(f"\nMacro average: {macro:.2f}")

    if args.out:
        # Build comparison from parent dirs of scores.csv files
        all_csvs = []
        if args.baseline:
            all_csvs.append(Path(args.baseline).parent)
        for v in args.variants:
            all_csvs.append(Path(v).parent)
        if all_csvs:
            md = build_comparison_table(all_csvs, Path(args.out))
            print("\n" + md + "\n")


if __name__ == "__main__":
    _cli_main()
