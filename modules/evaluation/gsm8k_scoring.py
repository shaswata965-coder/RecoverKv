"""GSM8K scoring — post-hoc metric computation with extraction diagnostics.

This REPLACES the DefensiveKV ``evaluation/gsm8k/calculate_metrics.py`` that was
imported verbatim into this project. That implementation makes the reported
accuracy a function of *how the generation terminated* rather than of whether the
model got the arithmetic right, which makes the accuracy-vs-KV-budget curve
non-monotonic for every compression method at once. See ``docs`` block below.

What went wrong in the imported scorer
--------------------------------------
Its ladder was:

    1. re.search(r"[Aa]nswer:\\s*([+-]?[\\d,]+\\.?\\d*)")   # FIRST match
    2. re.search(r"[Tt]he answer is\\s*[\\$]?\\s*(...)")     # FIRST match
    3. re.search(r"####\\s*(...)")                          # FIRST match
    4. re.findall(...)[-1]                                  # LAST number anywhere

Rung 1 requires a **digit immediately after the colon**. Real Llama-3.1 output is
``Answer: $18``, ``Answer: **18**``, ``Answer: \\boxed{18}`` — all of which fail
rung 1 (and 2 and 3) and drop to rung 4, the last-number-anywhere fallback. Rung 4
is silently correct *only when generation stopped right after the answer*. GSM8K
here runs with ``max_new_tokens=256`` and no stop sequence, so a model that
answers correctly and then keeps going ("\\n\\nQuestion: ... Answer: 12") has rung
4 return **12**. Same math, same correct answer, opposite score.

Measured: holding the model's arithmetic perfectly constant at 100% correct and
varying only the fraction of generations that run past the answer, the old scorer
reports 100.0 / 90.0 / 75.0 / 50.0 / 25.0 / 0.0 percent for run-on rates of
0 / 10 / 25 / 50 / 75 / 100 percent. Accuracy tracks termination 1:1.

KV compression perturbs verbosity and EOS behaviour, so this error term moves with
the budget in an uncontrolled direction — hence "score drops past a certain budget,
across the board".

Two further defects inherited from the same file, both fixed here:

*   ``[\\d,]+`` matches a bare ``","``. ``"1, 2,".replace(",", "")`` → ``""`` →
    scored 0 even when right.
*   ``evaluate.py`` catches generation exceptions and writes the string
    ``"Failure: <msg>"`` into ``predicted_answer``. The old scorer feeds that to
    rung 4, which happily extracts a number out of an OOM message and scores it
    wrong. A crashed run is therefore indistinguishable from a bad method — this
    is why DefensiveKV's own published ``EfficientLayerDefensiveKV`` GSM8K numbers
    are ``0.0`` and ``0.08``. Here failures are counted and reported separately.

The fix
-------
1.  Cut hallucinated continuations *before* extraction (``_strip_runon``).
2.  Inside the kept segment take the **last** explicit answer marker (models
    self-correct mid-solution), not the first.
3.  Accept the formats models actually emit: ``$``, ``**bold**``, ``\\boxed{}``,
    ``####``, trailing units, thousands separators, trailing periods.
4.  Keep the last-number fallback — but **flag every use of it**, and report a
    ``strict`` accuracy that excludes it. If strict and lenient accuracy diverge,
    or the divergence itself moves with the budget, the number is not trustworthy
    and the report says so.
5.  Never let a generation failure masquerade as a wrong answer.

Can be invoked as:
    python -m modules.evaluation.gsm8k_scoring --predictions <path-or-glob>
Or imported as a drop-in replacement for the DefensiveKV module:
    from modules.evaluation.gsm8k_scoring import calculate_metrics
"""

from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import math
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Number grammar
# ---------------------------------------------------------------------------

# A number must START with a digit — this is what stops a bare "," from being
# matched as a number (the DefensiveKV `[\d,]+` bug). Optional sign, optional
# currency, thousands separators, optional decimal tail.
_NUM_CORE = r"[-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?"
_NUM_RE = re.compile(_NUM_CORE)

# Markers that introduce a final answer, in the forms models actually produce.
# Each alternative is followed by arbitrary decoration (**, $, \boxed{, spaces,
# "is", ":") before the number itself, handled by _first_number_after().
_ANSWER_MARKERS = [
    r"####",
    r"\\boxed\s*\{",
    r"(?:final\s+)?answer\s*(?:is)?\s*[:=]",
    r"the\s+(?:final\s+)?answer\s+is",
    r"answer\s*[:=]",
]
_MARKER_RE = re.compile("|".join(f"(?:{m})" for m in _ANSWER_MARKERS), re.IGNORECASE)

# Boundaries at which a completion has stopped answering *this* problem and
# started inventing a new one. Anchored to line starts so an in-reasoning use of
# the word ("the question asks...") does not trigger a cut.
_RUNON_RE = re.compile(
    r"\n\s*(?:"
    r"Question\s*:"
    r"|Problem\s*:"
    r"|Format\s*:"
    r"|Q\s*\d*\s*:"
    r"|Exercise\s*\d*\s*:"
    r"|#{1,6}\s*Question"
    r")",
    re.IGNORECASE,
)

# A generation that was cut off by max_new_tokens rather than by EOS. Heuristic:
# no answer marker AND no sentence-terminating punctuation at the end.
_CLEAN_END_RE = re.compile(r"[.!?)\]\}\"']\s*$")

_FAILURE_PREFIX = "Failure:"


def _strip_runon(text: str) -> Tuple[str, bool]:
    """Return (text up to the first hallucinated next-problem boundary, was_cut)."""
    m = _RUNON_RE.search(text)
    if m is None:
        return text, False
    return text[: m.start()], True


def _normalize_number(raw: str) -> Optional[float]:
    """Parse a matched number token into a float, or None if unparseable."""
    cleaned = raw.replace(",", "").replace("$", "").replace(" ", "")
    cleaned = cleaned.rstrip(".")
    if cleaned in ("", "-", "+"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _first_number_after(text: str, pos: int, window: int = 64) -> Optional[float]:
    """Find the first number within *window* chars after *pos*.

    The window keeps ``Answer:`` from reaching across half a paragraph to grab an
    unrelated number when the model wrote ``Answer: see above``.
    """
    segment = text[pos : pos + window]
    m = _NUM_RE.search(segment)
    if m is None:
        return None
    return _normalize_number(m.group(0))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass
class Extraction:
    """Result of pulling a final answer out of one generation."""

    value: Optional[float]
    #: "marker" (explicit Answer:/####/boxed), "fallback" (last number anywhere),
    #: "none" (nothing numeric found), "failed" (generation errored),
    #: "empty" (null/blank prediction)
    method: str
    #: True if a hallucinated follow-up problem was trimmed before extraction.
    runon_trimmed: bool = False
    #: True if the generation looks like it hit max_new_tokens mid-sentence.
    looks_truncated: bool = False
    raw: str = ""

    @property
    def is_confident(self) -> bool:
        """Whether the answer came from an explicit marker rather than a guess."""
        return self.method == "marker"


def extract_answer(prediction: Any) -> Extraction:
    """Extract the final numeric answer from one model generation.

    Unlike the DefensiveKV original this is *order-correct* (last marker wins,
    after run-on is trimmed) and *honest* (it says which rung it used).
    """
    if prediction is None or (isinstance(prediction, float) and math.isnan(prediction)):
        return Extraction(None, "empty", raw="")

    text = str(prediction)
    if text.strip().startswith(_FAILURE_PREFIX):
        return Extraction(None, "failed", raw=text)
    if not text.strip():
        return Extraction(None, "empty", raw=text)

    body, trimmed = _strip_runon(text)
    body = body.strip()
    if not body:
        # Everything after the first boundary — the model restated the format and
        # never answered. Fall back to the whole text rather than scoring blind.
        body, trimmed = text.strip(), False

    # Rung 1: explicit answer markers. Take the LAST one in this segment, because
    # models revise ("Answer: 12 — wait, that double counts. Answer: 18").
    matches = list(_MARKER_RE.finditer(body))
    for m in reversed(matches):
        val = _first_number_after(body, m.end())
        if val is not None:
            return Extraction(val, "marker", trimmed, False, text)

    truncated = not _CLEAN_END_RE.search(body)

    # Rung 2: no marker anywhere. Fall back to the last number in the segment,
    # but label it so the caller can discount it.
    nums = _NUM_RE.findall(body)
    if nums:
        val = _normalize_number(nums[-1])
        if val is not None:
            return Extraction(val, "fallback", trimmed, truncated, text)

    return Extraction(None, "none", trimmed, truncated, text)


def _normalize_ground_truth(ground_truth: Any) -> Optional[float]:
    """Parse a GSM8K reference answer (``'18'`` or ``'... #### 18'``) to float."""
    if ground_truth is None:
        return None
    gt = str(ground_truth)
    if "####" in gt:
        gt = gt.split("####")[-1]
    m = _NUM_RE.search(gt)
    if m is None:
        return None
    return _normalize_number(m.group(0))


def answers_match(pred: Optional[float], gt: Optional[float], abs_tol: float = 1e-6) -> bool:
    """Numeric equality with a tolerance, so ``18.00`` matches ``18``."""
    if pred is None or gt is None:
        return False
    return math.isclose(pred, gt, rel_tol=0.0, abs_tol=abs_tol)


# ---------------------------------------------------------------------------
# Per-example and aggregate scoring
# ---------------------------------------------------------------------------


@dataclass
class GSM8KReport:
    """Accuracy plus the diagnostics needed to tell a bad method from a bad run."""

    n_total: int = 0
    n_generation_failed: int = 0
    n_empty: int = 0
    n_no_number: int = 0
    n_marker: int = 0
    n_fallback: int = 0
    n_runon_trimmed: int = 0
    n_looks_truncated: int = 0
    n_correct_strict: int = 0
    n_correct_lenient: int = 0

    #: correct / (n_total - failures - empties). The headline number.
    accuracy: float = 0.0
    #: Same numerator but counting only marker-backed extractions. If this is far
    #: below `accuracy`, the headline is being propped up by the guess rung.
    accuracy_strict: float = 0.0
    #: correct / n_total, failures counted as wrong (DefensiveKV's denominator).
    accuracy_including_failures: float = 0.0
    #: Fraction of generations that carried an explicit answer marker.
    marker_rate: float = 0.0
    #: Correct answers that came only from the fallback rung — the untrustworthy
    #: part of the score. Large or budget-varying => curve is measuring format.
    n_correct_from_fallback: int = 0

    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def score_predictions(
    predictions: Sequence[Any],
    answers: Sequence[Any],
) -> GSM8KReport:
    """Score a list of generations against GSM8K references.

    ``answers[i]`` may be a scalar or a list of acceptable references (the
    DefensiveKV/LongBench convention); the max over references is taken.
    """
    if len(predictions) != len(answers):
        raise ValueError(
            f"predictions/answers length mismatch: {len(predictions)} vs {len(answers)}"
        )

    rep = GSM8KReport(n_total=len(predictions))

    for pred, gts in zip(predictions, answers):
        ex = extract_answer(pred)

        if ex.method == "failed":
            rep.n_generation_failed += 1
            continue
        if ex.method == "empty":
            rep.n_empty += 1
            continue

        if ex.runon_trimmed:
            rep.n_runon_trimmed += 1
        if ex.looks_truncated:
            rep.n_looks_truncated += 1

        if ex.method == "marker":
            rep.n_marker += 1
        elif ex.method == "fallback":
            rep.n_fallback += 1
        else:
            rep.n_no_number += 1

        gt_list = gts if isinstance(gts, (list, tuple)) else [gts]
        try:
            gt_list = list(gt_list)
        except TypeError:
            gt_list = [gts]

        hit = any(answers_match(ex.value, _normalize_ground_truth(g)) for g in gt_list)
        if hit:
            rep.n_correct_lenient += 1
            if ex.is_confident:
                rep.n_correct_strict += 1
            else:
                rep.n_correct_from_fallback += 1

    scoreable = rep.n_total - rep.n_generation_failed - rep.n_empty
    rep.accuracy = _pct(rep.n_correct_lenient, scoreable)
    rep.accuracy_strict = _pct(rep.n_correct_strict, scoreable)
    rep.accuracy_including_failures = _pct(rep.n_correct_lenient, rep.n_total)
    rep.marker_rate = _pct(rep.n_marker, scoreable)

    # ---- guard rails: refuse to report a clean number over a dirty run -------
    if rep.n_generation_failed:
        rep.warnings.append(
            f"{rep.n_generation_failed}/{rep.n_total} generations FAILED "
            f"(predicted_answer starts with '{_FAILURE_PREFIX}'). These are excluded "
            f"from `accuracy`. A run with failures is not comparable to one without -- "
            f"this is the defect that turns an OOM into a 0.0 'result'."
        )
    if rep.n_empty:
        rep.warnings.append(
            f"{rep.n_empty}/{rep.n_total} predictions were null/blank -- excluded from "
            f"`accuracy`. Check the runner, not the method."
        )
    if scoreable and rep.marker_rate < 90.0:
        rep.warnings.append(
            f"only {rep.marker_rate:.1f}% of generations contained an explicit answer "
            f"marker; {rep.n_fallback} fell back to last-number-in-text. Accuracy on "
            f"the fallback rung is a coin flip that tracks generation length, not "
            f"correctness."
        )
    if rep.n_correct_from_fallback and rep.n_correct_lenient:
        share = 100.0 * rep.n_correct_from_fallback / rep.n_correct_lenient
        if share > 5.0:
            rep.warnings.append(
                f"{share:.1f}% of the correct answers ({rep.n_correct_from_fallback}) "
                f"came from the fallback rung, not an explicit marker. Compare "
                f"`accuracy_strict` ({rep.accuracy_strict:.2f}) against `accuracy` "
                f"({rep.accuracy:.2f}) across budgets -- if the gap moves, the curve is "
                f"measuring output format, not model quality."
            )
    if scoreable and _pct(rep.n_looks_truncated, scoreable) > 5.0:
        rep.warnings.append(
            f"{rep.n_looks_truncated}/{scoreable} generations look truncated by "
            f"max_new_tokens (no marker, no terminal punctuation). Raise "
            f"max_new_tokens or add a stop sequence -- truncation rate co-varies with "
            f"KV budget and will bend the accuracy curve on its own."
        )

    return rep


# ---------------------------------------------------------------------------
# Drop-in replacement surface for DefensiveKV's gsm8k/calculate_metrics.py
# ---------------------------------------------------------------------------


def extract_predicted_number(prediction: str) -> str:
    """Signature-compatible with the DefensiveKV original; correct internals."""
    ex = extract_answer(prediction)
    if ex.value is None:
        return ""
    return str(int(ex.value)) if float(ex.value).is_integer() else str(ex.value)


def gsm8k_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Per-example 0/1 score. Signature-compatible with the original."""
    ex = extract_answer(prediction)
    return 1.0 if answers_match(ex.value, _normalize_ground_truth(ground_truth)) else 0.0


dataset2metric = {"gsm8k": gsm8k_score}


def get_score(dataset, predictions, answers, all_classes=None) -> float:
    """Signature-compatible aggregate. Returns the headline accuracy (0-100)."""
    del dataset, all_classes  # kept for call-site compatibility
    return score_predictions(list(predictions), list(answers)).accuracy


def calculate_metrics(df, verbose: bool = True) -> Dict[str, Any]:
    """Drop-in replacement for ``gsm8k.calculate_metrics.calculate_metrics``.

    Returns ``{task: accuracy}`` like the original, plus a ``"_diagnostics"`` key
    carrying the full per-task report. Callers that only read ``metrics["gsm8k"]``
    keep working; callers that care about validity can read the diagnostics.
    """
    scores: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}

    for task, df_task in df.groupby("task"):
        preds = df_task["predicted_answer"].tolist()
        answers = df_task["answer"].tolist()
        rep = score_predictions(preds, answers)
        scores[task] = rep.accuracy
        diagnostics[task] = rep.to_dict()

        if verbose:
            log.info(
                "%-10s n=%-5d acc=%.2f  strict=%.2f  marker_rate=%.1f%%  "
                "failed=%d  empty=%d  runon_trimmed=%d",
                task, rep.n_total, rep.accuracy, rep.accuracy_strict,
                rep.marker_rate, rep.n_generation_failed, rep.n_empty,
                rep.n_runon_trimmed,
            )
            for w in rep.warnings:
                log.warning("%s: %s", task, w)

    scores["_diagnostics"] = diagnostics
    return scores


# ---------------------------------------------------------------------------
# Re-scoring existing runs (no regeneration needed)
# ---------------------------------------------------------------------------


def _coerce_answer_cell(cell: Any) -> Any:
    """CSV round-trips turn ``['18']`` into the string ``"['18']"``.

    The DefensiveKV merge script used bare ``eval`` here, which is both unsafe and
    breaks on numpy's space-separated repr (``"['a' 'b']"``). Parse defensively.
    """
    if not isinstance(cell, str):
        return cell
    s = cell.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return s
    inner = s[1:-1].strip()
    if not inner:
        return []
    # Handles "'18'", "'18', '19'" and numpy's "'18' '19'" alike.
    parts = re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)", inner)
    return [a or b or c for a, b, c in parts]


def rescore_file(path: Path) -> GSM8KReport:
    """Re-score one prediction file (``*_df.csv`` or ``*.jsonl``) already on disk."""
    path = Path(path)
    if path.suffix == ".jsonl":
        preds, answers = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            preds.append(ex.get("pred", ex.get("predicted_answer")))
            answers.append(ex.get("answer", ex.get("answers")))
        return score_predictions(preds, answers)

    import pandas as pd  # local import: keeps the module importable without pandas

    df = pd.read_csv(path)
    missing = {"predicted_answer", "answer"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required column(s) {sorted(missing)}")
    answers = [_coerce_answer_cell(c) for c in df["answer"].tolist()]
    return score_predictions(df["predicted_answer"].tolist(), answers)


_TABLE_COLS = [
    "file", "n_total", "n_generation_failed", "n_empty",
    "accuracy", "accuracy_strict", "accuracy_including_failures",
    "marker_rate", "n_fallback", "n_correct_from_fallback",
    "n_runon_trimmed", "n_looks_truncated",
]


def rescore_many(paths: Iterable[Path], out_csv: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Re-score every prediction file and emit one comparison row per file."""
    rows: List[Dict[str, Any]] = []
    for p in paths:
        p = Path(p)
        try:
            rep = rescore_file(p)
        except Exception as e:  # a broken file must not silently become a score
            log.error("%s: could not be scored (%s)", p.name, e)
            continue
        d = rep.to_dict()
        d["file"] = p.name
        rows.append({k: d.get(k) for k in _TABLE_COLS})
        print(f"\n=== {p.name} ===")
        print(
            f"  accuracy               {rep.accuracy:6.2f}   "
            f"(n={rep.n_total - rep.n_generation_failed - rep.n_empty} scoreable)"
        )
        print(f"  accuracy_strict        {rep.accuracy_strict:6.2f}   (marker-backed only)")
        print(f"  incl. failures         {rep.accuracy_including_failures:6.2f}   (old denominator)")
        print(f"  marker rate            {rep.marker_rate:6.2f}%")
        print(
            f"  failed={rep.n_generation_failed}  empty={rep.n_empty}  "
            f"fallback={rep.n_fallback}  runon_trimmed={rep.n_runon_trimmed}  "
            f"truncated={rep.n_looks_truncated}"
        )
        for w in rep.warnings:
            print(f"  !! {w}")

    if out_csv is not None and rows:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_TABLE_COLS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("Comparison written to %s", out_csv)

    return rows


def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score GSM8K predictions with extraction diagnostics."
    )
    parser.add_argument(
        "--predictions", type=str, nargs="+", required=True,
        help="Prediction file(s) or glob(s): *_df.csv or *.jsonl",
    )
    parser.add_argument("--out_csv", type=str, default=None, help="Comparison CSV path")
    args = parser.parse_args()

    paths: List[Path] = []
    for pattern in args.predictions:
        hits = sorted(_glob.glob(pattern))
        paths.extend(Path(h) for h in hits) if hits else paths.append(Path(pattern))

    if not paths:
        parser.error("no prediction files matched")

    rescore_many(paths, Path(args.out_csv) if args.out_csv else None)


if __name__ == "__main__":
    _cli_main()
