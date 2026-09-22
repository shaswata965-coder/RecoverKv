"""Audit the GSM8K re-score: old (DefensiveKV) vs new scorer, on YOUR real data.

The fixed scorer in ``modules/evaluation/gsm8k_scoring.py`` is validated against
constructed outputs. This script validates it against the generations you actually
have, by showing every case where the two scorers disagree so you can eyeball who
is right. Twenty cases is usually enough to be sure.

It also prints the binomial noise floor, so you can tell which budget-to-budget
differences in your sweep are real and which are sampling noise.

Usage:
    python scripts/audit_gsm8k_rescore.py --predictions "results/*_df.csv"
    python scripts/audit_gsm8k_rescore.py --predictions "results/*_df.csv" --examples 30

Kaggle:
    %run scripts/audit_gsm8k_rescore.py --predictions "/kaggle/working/results/*_df.csv"
"""

from __future__ import annotations

import argparse
import glob as _glob
import math
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.evaluation.gsm8k_scoring import (  # noqa: E402
    _coerce_answer_cell,
    _normalize_ground_truth,
    answers_match,
    extract_answer,
    score_predictions,
)


# ---------------------------------------------------------------------------
# The DefensiveKV scorer, copied verbatim, for side-by-side comparison only.
# Do not import this into anything that reports a number.
# ---------------------------------------------------------------------------


def _legacy_extract(prediction: str) -> str:
    prediction = str(prediction).strip()
    m = re.search(r"[Aa]nswer:\s*([+-]?[\d,]+\.?\d*)", prediction)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"[Tt]he answer is\s*[\$]?\s*([+-]?[\d,]+\.?\d*)", prediction)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", prediction)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[+-]?[\d,]+\.?\d*", prediction)
    if nums:
        return nums[-1].replace(",", "")
    return ""


def _legacy_score_one(prediction: Any, ground_truths: Any) -> float:
    gts = ground_truths if isinstance(ground_truths, (list, tuple)) else [ground_truths]
    pred_num = _legacy_extract(prediction)
    if not pred_num:
        return 0.0
    best = 0.0
    for gt in gts:
        gt_num = str(gt).replace(",", "").strip()
        try:
            best = max(best, 1.0 if float(pred_num) == float(gt_num) else 0.0)
        except ValueError:
            best = max(best, 1.0 if pred_num == gt_num else 0.0)
    return best


# ---------------------------------------------------------------------------


def _binomial_ci95(p_pct: float, n: int) -> float:
    """Half-width of the 95% CI on an accuracy of p_pct% over n examples."""
    if n <= 0:
        return float("nan")
    p = p_pct / 100.0
    return round(100.0 * 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n), 2)


def _truncate(text: str, head: int = 220, tail: int = 160) -> str:
    text = str(text).replace("\n", "\\n")
    if len(text) <= head + tail + 5:
        return text
    return f"{text[:head]}  …[{len(text) - head - tail} chars]…  {text[-tail:]}"


def audit_file(path: Path, n_examples: int) -> Optional[dict]:
    import pandas as pd

    df = pd.read_csv(path)
    missing = {"predicted_answer", "answer"} - set(df.columns)
    if missing:
        print(f"  !! {path.name}: missing column(s) {sorted(missing)} -- skipped")
        return None

    preds = df["predicted_answer"].tolist()
    answers = [_coerce_answer_cell(c) for c in df["answer"].tolist()]

    rep = score_predictions(preds, answers)
    legacy_total = sum(_legacy_score_one(p, a) for p, a in zip(preds, answers))
    legacy_acc = round(100.0 * legacy_total / len(preds), 2) if preds else 0.0

    scoreable = rep.n_total - rep.n_generation_failed - rep.n_empty

    print(f"\n{'=' * 78}")
    print(f"  {path.name}")
    print(f"{'=' * 78}")
    print(f"  n rows                     {rep.n_total}")
    print(f"  scoreable                  {scoreable}   "
          f"(failed={rep.n_generation_failed}, empty={rep.n_empty})")
    print(f"  OLD (DefensiveKV)          {legacy_acc:6.2f}")
    print(f"  NEW                        {rep.accuracy:6.2f}  +/- {_binomial_ci95(rep.accuracy, scoreable)} (95% CI)")
    print(f"  NEW strict (marker only)   {rep.accuracy_strict:6.2f}")
    print(f"  marker rate                {rep.marker_rate:6.2f}%")
    print(f"  run-on trimmed             {rep.n_runon_trimmed}")
    print(f"  looks truncated            {rep.n_looks_truncated}")
    for w in rep.warnings:
        print(f"  !! {w}")

    # ---- disagreement audit -------------------------------------------------
    disagreements: List[tuple] = []
    for pred, gts in zip(preds, answers):
        old = _legacy_score_one(pred, gts)
        ex = extract_answer(pred)
        gt_list = gts if isinstance(gts, (list, tuple)) else [gts]
        new = 1.0 if any(
            answers_match(ex.value, _normalize_ground_truth(g)) for g in gt_list
        ) else 0.0
        if old != new:
            disagreements.append((pred, gt_list, _legacy_extract(pred), ex, old, new))

    print(f"\n  disagreements: {len(disagreements)}/{rep.n_total} "
          f"({100.0 * len(disagreements) / rep.n_total:.1f}%)")

    if disagreements and n_examples > 0:
        print(f"\n  --- first {min(n_examples, len(disagreements))} disagreements "
              f"(check which extraction is right) ---")
        for i, (pred, gt_list, old_val, ex, old, new) in enumerate(disagreements[:n_examples]):
            verdict = "NEW says correct" if new > old else "NEW says wrong"
            print(f"\n  [{i + 1}] gt={gt_list}   {verdict}")
            print(f"      old extracted: {old_val!r}")
            print(f"      new extracted: {ex.value!r}  (via {ex.method}"
                  f"{', run-on trimmed' if ex.runon_trimmed else ''}"
                  f"{', looks truncated' if ex.looks_truncated else ''})")
            print(f"      generation:    {_truncate(pred)}")

    return {
        "file": path.name,
        "n": rep.n_total,
        "scoreable": scoreable,
        "old": legacy_acc,
        "new": rep.accuracy,
        "new_strict": rep.accuracy_strict,
        "ci95": _binomial_ci95(rep.accuracy, scoreable),
        "marker_rate": rep.marker_rate,
        "failed": rep.n_generation_failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=str, nargs="+", required=True,
                        help="Prediction CSV file(s) or glob(s)")
    parser.add_argument("--examples", type=int, default=20,
                        help="Disagreement examples to print per file (0 to skip)")
    args = parser.parse_args()

    paths: List[Path] = []
    for pattern in args.predictions:
        hits = sorted(_glob.glob(pattern))
        paths.extend(Path(h) for h in hits) if hits else paths.append(Path(pattern))
    paths = [p for p in paths if p.exists()]
    if not paths:
        parser.error("no prediction files matched")

    rows = [r for r in (audit_file(p, args.examples) for p in paths) if r]
    if not rows:
        return

    print(f"\n\n{'=' * 96}")
    print("  SWEEP SUMMARY")
    print(f"{'=' * 96}")
    print(f"  {'file':<44} {'n':>5} {'OLD':>7} {'NEW':>7} {'+/-95%':>6} "
          f"{'strict':>7} {'mark%':>6} {'fail':>5}")
    print(f"  {'-' * 92}")
    for r in rows:
        print(f"  {r['file'][:44]:<44} {r['scoreable']:>5} {r['old']:>7.2f} "
              f"{r['new']:>7.2f} {r['ci95']:>6.2f} {r['new_strict']:>7.2f} "
              f"{r['marker_rate']:>6.1f} {r['failed']:>5}")

    worst_ci = max((r["ci95"] for r in rows), default=0.0)
    print(f"\n  Noise floor: with these sample sizes, two budgets differing by less")
    print(f"  than ~{2 * worst_ci:.1f} points are statistically indistinguishable.")
    if any(r["marker_rate"] < 90.0 for r in rows):
        print("  !! Some runs have marker_rate < 90% -- raise max_new_tokens or add a")
        print("     stop sequence before trusting the shape of the curve.")
    if any(r["failed"] for r in rows):
        print("  !! Some runs contain failed generations -- those budgets are not")
        print("     comparable to the clean ones until re-run.")


if __name__ == "__main__":
    main()
