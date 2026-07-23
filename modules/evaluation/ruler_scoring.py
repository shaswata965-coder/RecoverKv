"""RULER scoring pipeline — post-hoc metric computation.

Reads ``<task>.jsonl`` prediction files written by ``RulerRunner`` and applies
the RULER string-match metric, then produces a CSV comparison table.

Metric functions (``string_match_part`` / ``string_match_all``) and the
task-category dispatch are ported verbatim from DefensiveKV's
``kv_cache/DefensiveKV/evaluation/ruler/calculate_metrics.py``
(NVIDIA, SPDX-Apache-2.0), so scores are directly comparable to DefensiveKV's
published RULER numbers. Like that reference implementation, null/failed
predictions are DROPPED before scoring (not counted as 0) — this differs from
``longbench_scoring.score_predictions``, which follows THUDM/LongBench's
convention of scoring null preds as 0 in the denominator.

Can be invoked as:
    python -m modules.evaluation.ruler_scoring --predictions_dir <dir> --out_csv <path>
Or via main.py with mode=ruler_score.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f]")


def string_match_part(preds: List[str], refs: List[List[str]]) -> float:
    """Per-example score = 1 if ANY reference substring appears in pred, else 0."""
    score = (
        sum(max(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) for pred, ref in zip(preds, refs))
        / len(preds)
        * 100
    )
    return round(score, 2)


def string_match_all(preds: List[str], refs: List[List[str]]) -> float:
    """Per-example score = fraction of references whose substring appears in pred."""
    score = (
        sum(sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref) for pred, ref in zip(preds, refs))
        / len(preds)
        * 100
    )
    return round(score, 2)


def _task_metric_fn(task: str):
    task_category = task.split("_")[0]
    return string_match_part if task_category == "qa" else string_match_all


def score_predictions(
    predictions_dir: Path,
    out_csv: Optional[Path] = None,
) -> Dict[str, float]:
    """Score all ``<task>.jsonl`` files in *predictions_dir*.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing per-task ``.jsonl`` prediction files.
    out_csv : Path, optional
        If provided, write a CSV with columns: task, num_examples, score, dropped.

    Returns
    -------
    dict[str, float]
        ``{task_name: score}`` where score is percentage (0-100).
    """
    results: Dict[str, float] = {}
    details: List[Dict[str, Any]] = []

    for jsonl in sorted(predictions_dir.glob("*.jsonl")):
        task = jsonl.stem
        preds: List[str] = []
        refs: List[List[str]] = []
        n_total = 0
        n_dropped = 0

        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            n_total += 1

            pred = ex.get("pred")
            if pred is None:
                n_dropped += 1
                continue

            pred = _CONTROL_CHAR_RE.sub("", pred.strip()).strip()
            preds.append(pred)
            refs.append(ex.get("answer", []))

        if n_dropped > 0:
            log.warning(
                "%s: %d/%d examples had pred=null (OOM/budget-too-small) — "
                "dropped before scoring (matches DefensiveKV calculate_metrics.py)",
                task, n_dropped, n_total,
            )

        if not preds:
            log.warning("%s: no scoreable predictions — skipping", task)
            continue

        try:
            metric_fn = _task_metric_fn(task)
            score = metric_fn(preds, refs)
        except Exception as e:
            log.error("Error scoring task %s: %s", task, e)
            continue

        results[task] = score
        details.append(
            {"task": task, "num_examples": n_total, "score": score, "dropped": n_dropped}
        )
        log.info(
            "%-20s  n=%-4d  dropped=%-3d  score=%.2f",
            task, n_total, n_dropped, score,
        )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["task", "num_examples", "score", "dropped"]
            )
            writer.writeheader()
            writer.writerows(details)
        log.info("Scores written to %s", out_csv)

    return results


def compute_macro_average(scores: Dict[str, float]) -> float:
    """Compute overall macro average across all scored tasks."""
    valid = [v for v in scores.values() if not np.isnan(v)]
    if len(valid) < len(scores):
        log.warning(
            "Macro average computed over %d/%d tasks (some missing/NaN)",
            len(valid), len(scores),
        )
    return round(float(np.mean(valid)), 2) if valid else float("nan")


def build_comparison_table(
    run_dirs: List[Path],
    out_path: Optional[Path] = None,
) -> str:
    """Build a comparison table across multiple RULER runs (e.g. omega sweep)."""
    all_scores: Dict[str, Dict[str, float]] = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        csv_path = run_dir / "scores.csv"
        if csv_path.exists():
            scores = {}
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scores[row["task"]] = float(row["score"])
            all_scores[run_dir.name] = scores
        else:
            scores = score_predictions(run_dir, csv_path)
            all_scores[run_dir.name] = scores

    if not all_scores:
        return "No runs found to compare."

    all_tasks = sorted(set(t for scores in all_scores.values() for t in scores.keys()))
    run_names = list(all_scores.keys())

    lines = []
    header = "| Task | " + " | ".join(run_names) + " |"
    sep = "|---|" + "|".join(["---"] * len(run_names)) + "|"
    lines.append(header)
    lines.append(sep)

    for t in all_tasks:
        row = f"| {t} |"
        for rn in run_names:
            val = all_scores[rn].get(t, float("nan"))
            row += f" {val:.2f} |" if not np.isnan(val) else " — |"
        lines.append(row)

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
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["task"] + run_names)
            for t in all_tasks:
                writer.writerow([t] + [all_scores[rn].get(t, "") for rn in run_names])
        md_path = out_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# RULER Comparison Table\n\n")
            f.write(md_table)
            f.write("\n")
        log.info("Comparison written to %s and %s", out_path, md_path)

    return md_table


# ---------------------------------------------------------------------------
# RulerScorer — runnable via main.py with mode=ruler_score
# ---------------------------------------------------------------------------


class RulerScorer:
    """Post-hoc scorer, invoked via ``main.py --override run.mode=ruler_score``."""

    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> None:
        base_dir = Path("outputs/ruler")
        if not base_dir.exists():
            log.error("No outputs/ruler/ directory found")
            return

        run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        if not run_dirs:
            log.error("No run directories found in %s", base_dir)
            return

        for run_dir in run_dirs:
            jsonls = list(run_dir.glob("*.jsonl"))
            if jsonls:
                csv_path = run_dir / "scores.csv"
                log.info("Scoring %s (%d tasks)...", run_dir.name, len(jsonls))
                score_predictions(run_dir, csv_path)

        md = build_comparison_table(run_dirs, base_dir / "comparison.csv")
        print("\n" + md + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="Score RULER predictions")
    parser.add_argument(
        "--predictions_dir", type=str, help="Directory with <task>.jsonl files",
    )
    parser.add_argument(
        "--out_csv", type=str, default=None, help="Output CSV path",
    )
    parser.add_argument(
        "--compare", type=str, nargs="*", default=[],
        help="Multiple run directories to build a comparison table across",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output comparison CSV path",
    )
    args = parser.parse_args()

    if args.predictions_dir:
        predictions_dir = Path(args.predictions_dir)
        out_csv = Path(args.out_csv) if args.out_csv else None
        scores = score_predictions(predictions_dir, out_csv)
        macro = compute_macro_average(scores)
        print(f"\nMacro average: {macro:.2f}")

    if args.compare:
        run_dirs = [Path(d) for d in args.compare]
        out_path = Path(args.out) if args.out else None
        md = build_comparison_table(run_dirs, out_path)
        print("\n" + md + "\n")


if __name__ == "__main__":
    _cli_main()
