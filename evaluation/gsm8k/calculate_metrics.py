"""GSM8K scoring — accuracy plus the diagnostics needed to tell a bad method from a bad run.

Signature-compatible with the other ``evaluation/*/calculate_metrics.py`` modules
(``calculate_metrics(df) -> dict``), so it can be dropped into ``evaluate.py``'s
``SCORER_DICT``. The internals are not a port of the usual pattern, for the reasons
below.

Why the obvious scorer is wrong
-------------------------------
The natural GSM8K extraction ladder is:

    1. re.search(r"[Aa]nswer:\\s*([+-]?[\\d,]+\\.?\\d*)")   # FIRST match
    2. re.search(r"[Tt]he answer is\\s*[\\$]?\\s*(...)")     # FIRST match
    3. re.search(r"####\\s*(...)")                          # FIRST match
    4. re.findall(...)[-1]                                  # LAST number anywhere

Rung 1 requires a **digit immediately after the colon**. Real Llama-3.1 output is
``Answer: $18``, ``Answer: **18**``, ``Answer: \\boxed{18}`` — all of which fail rungs
1-3 and drop to rung 4, the last-number-anywhere fallback. Rung 4 is silently correct
*only when generation stopped right after the answer*. A model that answers correctly
and then keeps going ("\\n\\nQuestion: ... Answer: 12") has rung 4 return **12**. Same
math, same correct answer, opposite score.

Holding the model's arithmetic perfectly constant at 100% correct and varying only the
fraction of generations that run past the answer, that ladder reports
100.0 / 90.0 / 75.0 / 50.0 / 25.0 / 0.0 percent for run-on rates of
0 / 10 / 25 / 50 / 75 / 100 percent. **Accuracy tracks termination 1:1.**

KV compression perturbs verbosity and EOS behaviour, so this error term moves with the
compression ratio in an uncontrolled direction — which is exactly how you get "the
score drops past a certain ratio, for every method at once".

Two further defects in the same family, both fixed here:

*   ``[\\d,]+`` matches a bare ``","``. ``"1, 2,".replace(",", "")`` -> ``""`` -> scored
    0 even when the answer was right.
*   ``evaluate.py`` catches generation exceptions and writes the string
    ``"Failure: <msg>"`` into ``predicted_answer``. Rung 4 happily extracts a number out
    of an OOM message and scores it wrong, so a crashed run is indistinguishable from a
    bad method. Here failures are counted and excluded, never scored.

The fix
-------
1.  Cut hallucinated continuations *before* extraction (``_strip_runon``).
2.  Inside the kept segment take the **last** explicit answer marker (models
    self-correct mid-solution), not the first.
3.  Accept the formats models actually emit: ``$``, ``**bold**``, ``\\boxed{}``,
    ``####``, trailing units, thousands separators, trailing periods.
4.  Keep the last-number fallback — but **flag every use of it**, and report a
    ``strict`` accuracy that counts only marker-backed extractions. If strict and
    lenient diverge, or the divergence itself moves with the compression ratio, the
    number is not trustworthy and the report says so.
5.  Never let a generation failure masquerade as a wrong answer.

Usage
-----
    python -m gsm8k.calculate_metrics --runs results/gsm8k/*/          # comparison table
    python -m gsm8k.calculate_metrics --predictions '*_df.csv'         # rescore old runs
"""

from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Number grammar
# ---------------------------------------------------------------------------

# A number must START with a digit — this is what stops a bare "," from being matched
# as a number. Optional sign, optional currency, thousands separators, decimal tail.
_NUM_CORE = r"[-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?"
_NUM_RE = re.compile(_NUM_CORE)

# Markers that introduce a final answer, in the forms models actually produce. Each
# alternative may be followed by arbitrary decoration (**, $, \boxed{, spaces, "is",
# ":") before the number itself, handled by _first_number_after().
_ANSWER_MARKERS = [
    r"####",
    r"\\boxed\s*\{",
    r"(?:final\s+)?answer\s*(?:is)?\s*[:=]",
    r"the\s+(?:final\s+)?answer\s+is",
    r"answer\s*[:=]",
]
_MARKER_RE = re.compile("|".join(f"(?:{m})" for m in _ANSWER_MARKERS), re.IGNORECASE)

# Boundaries at which a completion has stopped answering *this* problem and started
# inventing a new one. Anchored to line starts so an in-reasoning use of the word
# ("the question asks...") does not trigger a cut.
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

# A generation cut off by max_new_tokens rather than by EOS. Heuristic: no answer
# marker AND no sentence-terminating punctuation at the end.
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
    m = _NUM_RE.search(text[pos : pos + window])
    return None if m is None else _normalize_number(m.group(0))


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

    Order-correct (last marker wins, after run-on is trimmed) and honest (it reports
    which rung it used).
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
        # Everything sat after the first boundary — the model restated the format and
        # never answered. Fall back to the whole text rather than scoring blind.
        body, trimmed = text.strip(), False

    # Rung 1: explicit answer markers. Take the LAST one, because models revise
    # ("Answer: 12 — wait, that double counts. Answer: 18").
    for m in reversed(list(_MARKER_RE.finditer(body))):
        val = _first_number_after(body, m.end())
        if val is not None:
            return Extraction(val, "marker", trimmed, False, text)

    truncated = not _CLEAN_END_RE.search(body)

    # Rung 2: no marker anywhere. Fall back to the last number in the segment, but
    # label it so the caller can discount it.
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
    return None if m is None else _normalize_number(m.group(0))


def answers_match(
    pred: Optional[float], gt: Optional[float], abs_tol: float = 1e-6
) -> bool:
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
    #: Same denominator, but only marker-backed extractions count as correct. If this
    #: is far below `accuracy`, the headline is propped up by the guess rung.
    accuracy_strict: float = 0.0
    #: correct / n_total, failures counted as wrong.
    accuracy_including_failures: float = 0.0
    #: Fraction of scoreable generations that carried an explicit answer marker.
    marker_rate: float = 0.0
    #: Correct answers that came only from the fallback rung — the untrustworthy part
    #: of the score. Large or ratio-varying => the curve is measuring output format.
    n_correct_from_fallback: int = 0

    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def score_predictions(
    predictions: Sequence[Any], answers: Sequence[Any]
) -> GSM8KReport:
    """Score a list of generations against GSM8K references.

    ``answers[i]`` may be a scalar or a list of acceptable references (the
    DefensiveKV / LongBench convention); a hit against any reference counts.
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

        if any(answers_match(ex.value, _normalize_ground_truth(g)) for g in gt_list):
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

    # ---- guard rails: refuse to report a clean number over a dirty run -----------
    if rep.n_generation_failed:
        rep.warnings.append(
            f"{rep.n_generation_failed}/{rep.n_total} generations FAILED "
            f"(prediction starts with '{_FAILURE_PREFIX}'). These are excluded from "
            f"`accuracy`. A run with failures is not comparable to one without -- this "
            f"is the defect that turns an OOM into a 0.0 'result'."
        )
    if rep.n_empty:
        rep.warnings.append(
            f"{rep.n_empty}/{rep.n_total} predictions were null/blank -- excluded from "
            f"`accuracy`. Check the runner, not the method."
        )
    if scoreable and rep.marker_rate < 90.0:
        rep.warnings.append(
            f"only {rep.marker_rate:.1f}% of generations contained an explicit answer "
            f"marker; {rep.n_fallback} fell back to last-number-in-text. Accuracy on the "
            f"fallback rung is a coin flip that tracks generation length, not correctness."
        )
    if rep.n_correct_from_fallback and rep.n_correct_lenient:
        share = 100.0 * rep.n_correct_from_fallback / rep.n_correct_lenient
        if share > 5.0:
            rep.warnings.append(
                f"{share:.1f}% of the correct answers ({rep.n_correct_from_fallback}) came "
                f"from the fallback rung, not an explicit marker. Compare "
                f"`accuracy_strict` ({rep.accuracy_strict:.2f}) against `accuracy` "
                f"({rep.accuracy:.2f}) across compression ratios -- if the gap moves, the "
                f"curve is measuring output format, not model quality."
            )
    if scoreable and _pct(rep.n_looks_truncated, scoreable) > 5.0:
        rep.warnings.append(
            f"{rep.n_looks_truncated}/{scoreable} generations look truncated by "
            f"max_new_tokens (no marker, no terminal punctuation). Raise max_new_tokens or "
            f"add a stop sequence -- truncation rate co-varies with the KV budget and will "
            f"bend the accuracy curve on its own."
        )

    return rep


# ---------------------------------------------------------------------------
# Drop-in surface for evaluate.py's SCORER_DICT
# ---------------------------------------------------------------------------


def extract_predicted_number(prediction: str) -> str:
    """Return the extracted answer as a string (``""`` when nothing was found)."""
    ex = extract_answer(prediction)
    if ex.value is None:
        return ""
    return str(int(ex.value)) if float(ex.value).is_integer() else str(ex.value)


def gsm8k_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Per-example 0/1 score."""
    ex = extract_answer(prediction)
    return (
        1.0 if answers_match(ex.value, _normalize_ground_truth(ground_truth)) else 0.0
    )


dataset2metric = {"gsm8k": gsm8k_score}


def get_score(dataset, predictions, answers, all_classes=None) -> float:
    """Aggregate accuracy (0-100), signature-compatible with the LongBench helpers."""
    del dataset, all_classes  # kept for call-site compatibility
    return score_predictions(list(predictions), list(answers)).accuracy


def calculate_metrics(df, verbose: bool = True) -> Dict[str, Any]:
    """``evaluate.py``-compatible entry point.

    Returns ``{task: accuracy}`` like the other scorers, plus a ``"_diagnostics"`` key
    carrying the full per-task report. Callers that only read ``metrics["gsm8k"]`` keep
    working; callers that care about validity can read the diagnostics.
    """
    scores: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}

    for task, df_task in df.groupby("task"):
        rep = score_predictions(
            df_task["predicted_answer"].tolist(), df_task["answer"].tolist()
        )
        scores[task] = rep.accuracy
        diagnostics[task] = rep.to_dict()

        if verbose:
            print(
                f"{task:<10} n={rep.n_total:<5} acc={rep.accuracy:.2f}  "
                f"strict={rep.accuracy_strict:.2f}  marker_rate={rep.marker_rate:.1f}%  "
                f"failed={rep.n_generation_failed}  empty={rep.n_empty}  "
                f"runon_trimmed={rep.n_runon_trimmed}"
            )
            for w in rep.warnings:
                print(f"  !! {task}: {w}")

    scores["_diagnostics"] = diagnostics
    return scores


# ---------------------------------------------------------------------------
# Re-scoring existing runs (no regeneration needed)
# ---------------------------------------------------------------------------


def _coerce_answer_cell(cell: Any) -> Any:
    """CSV round-trips turn ``['18']`` into the string ``"['18']"``.

    Parsed defensively rather than with ``eval``, which is both unsafe and breaks on
    numpy's space-separated repr (``"['a' 'b']"``).
    """
    if not isinstance(cell, str):
        return cell
    s = cell.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return s
    inner = s[1:-1].strip()
    if not inner:
        return []
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


def rescore_many(
    paths: Iterable[Path], out_csv: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Re-score every prediction file and emit one comparison row per file."""
    rows: List[Dict[str, Any]] = []
    for p in paths:
        p = Path(p)
        try:
            rep = rescore_file(p)
        except Exception as e:  # a broken file must not silently become a score
            print(f"ERROR {p.name}: could not be scored ({e})")
            continue
        d = rep.to_dict()
        d["file"] = p.name
        rows.append({k: d.get(k) for k in _TABLE_COLS})
        scoreable = rep.n_total - rep.n_generation_failed - rep.n_empty
        print(f"\n=== {p.name} ===")
        print(f"  accuracy               {rep.accuracy:6.2f}   (n={scoreable} scoreable)")
        print(f"  accuracy_strict        {rep.accuracy_strict:6.2f}   (marker-backed only)")
        print(f"  incl. failures         {rep.accuracy_including_failures:6.2f}")
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
        print(f"\nComparison written to {out_csv}")

    return rows


# ---------------------------------------------------------------------------
# Run-directory scoring (run_gsm8k.py output) + press/ratio comparison
# ---------------------------------------------------------------------------


def _binomial_ci95(p_pct: float, n: int) -> float:
    """Half-width of the 95% CI on an accuracy of *p_pct*% over *n* examples."""
    if n <= 0:
        return float("nan")
    p = p_pct / 100.0
    return round(100.0 * 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n), 2)


def score_run_dir(run_dir: Path) -> Tuple[GSM8KReport, Dict[str, Any]]:
    """Score one runner output directory. Returns ``(report, meta)``."""
    run_dir = Path(run_dir)
    preds_path = run_dir / "predictions.jsonl"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} not found")

    rep = rescore_file(preds_path)

    meta_path = run_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return rep, meta


_COMPARE_COLS = [
    "run", "press", "compression_ratio", "n", "accuracy", "ci95", "accuracy_strict",
    "marker_rate", "n_fallback", "n_looks_truncated", "n_generation_failed",
    "measured_context_compression", "effective_retention",
    "pct_examples_degenerate", "pct_examples_press_skipped",
    "compress_questions", "batch_size", "n_batch_size_backoffs", "dataset_sha",
]


def build_comparison_table(
    run_dirs: Sequence[Path], out_csv: Optional[Path] = None
) -> str:
    """Score several runs and render a press/compression-ratio comparison table.

    Refuses to present the runs as comparable when their ``dataset_sha`` or their
    ``compress_questions`` setting differ — those are the failure modes where a
    "sweep" is really two different experiments, and neither is detectable from the
    accuracy numbers alone.
    """
    rows: List[Dict[str, Any]] = []
    for d in run_dirs:
        d = Path(d)
        try:
            rep, meta = score_run_dir(d)
        except Exception as e:
            print(f"ERROR {d}: could not be scored ({e})")
            continue
        scoreable = rep.n_total - rep.n_generation_failed - rep.n_empty
        rows.append({
            "run": d.name,
            "press": meta.get("press_name", "?"),
            "compression_ratio": meta.get("compression_ratio"),
            "n": scoreable,
            "accuracy": rep.accuracy,
            "ci95": _binomial_ci95(rep.accuracy, scoreable),
            "accuracy_strict": rep.accuracy_strict,
            "marker_rate": rep.marker_rate,
            "n_fallback": rep.n_fallback,
            "n_looks_truncated": rep.n_looks_truncated,
            "n_generation_failed": rep.n_generation_failed,
            "measured_context_compression": meta.get("measured_context_compression"),
            "effective_retention": meta.get("mean_effective_retention"),
            "pct_examples_degenerate": meta.get("pct_examples_degenerate"),
            "pct_examples_press_skipped": meta.get("pct_examples_press_skipped"),
            "compress_questions": meta.get("compress_questions"),
            "dataset_sha": (meta.get("dataset_sha") or "")[:12],
            "batch_size": (meta.get("batching") or {}).get("batch_size"),
            "n_batch_size_backoffs": meta.get("n_batch_size_backoffs") or 0,
            "_warnings": rep.warnings,
        })

    if not rows:
        return "(no runs scored)"

    lines: List[str] = []
    header = (
        f"  {'run':<26} {'press':<20} {'cr':>5} {'n':>5} {'acc':>7} {'+/-':>6} "
        f"{'strict':>7} {'mark%':>6} {'trunc':>6} {'ctxComp':>8} {'effRet':>7} {'degen%':>7}"
    )
    lines.append("=" * len(header))
    lines.append("  GSM8K PRESS / COMPRESSION-RATIO COMPARISON")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    def _fmt(v, spec="{:.3f}"):
        return "-" if v is None else spec.format(v)

    for r in rows:
        cr = "full" if r["compression_ratio"] in (None, 0) else f"{r['compression_ratio']:.2f}"
        lines.append(
            f"  {r['run'][:26]:<26} {str(r['press'])[:20]:<20} {cr:>5} {r['n']:>5} "
            f"{r['accuracy']:>7.2f} {r['ci95']:>6.2f} {r['accuracy_strict']:>7.2f} "
            f"{r['marker_rate']:>6.1f} {r['n_looks_truncated']:>6} "
            f"{_fmt(r['measured_context_compression']):>8} "
            f"{_fmt(r['effective_retention']):>7} "
            f"{_fmt(r['pct_examples_degenerate'], '{:.1f}'):>7}"
        )

    lines.append("")
    lines.append(
        "  ctxComp = measured fraction of CONTEXT KV actually pruned (from the cache, "
        "not from the requested ratio)."
    )
    lines.append(
        "  effRet  = retained KV / full-sequence KV, counting the uncompressed question "
        "and every generated token. This is what the method costs end to end."
    )

    # ---- validity gates -------------------------------------------------------
    lines.append("")
    shas = {r["dataset_sha"] for r in rows if r["dataset_sha"]}
    if len(shas) > 1:
        lines.append(
            "  !! FATAL: runs used DIFFERENT datasets (dataset_sha: "
            f"{', '.join(sorted(shas))}). These accuracies are not comparable. Rebuild "
            "once with `python -m gsm8k.create_huggingface_dataset` and re-run every "
            "press against that one directory."
        )
    else:
        # ASCII only: these tables get read over ssh on cp1252 consoles, where a stray
        # em-dash prints as a replacement char and makes the gate look broken.
        lines.append(
            f"  dataset_sha {shas.pop() if shas else 'unknown'} -- identical across runs. OK"
        )

    cq = {r["compress_questions"] for r in rows if r["compress_questions"] is not None}
    if len(cq) > 1:
        lines.append(
            "  !! FATAL: runs disagree on compress_questions "
            f"({sorted(str(c) for c in cq)}). One set compressed the problem statement "
            "and the other only compressed the constant system prompt -- the compression "
            "ratio does not mean the same thing in the two, so the rows are not comparable."
        )

    worst_ci = max((r["ci95"] for r in rows), default=0.0)
    lines.append(
        f"  Noise floor: runs differing by less than ~{2 * worst_ci:.1f} points are "
        f"statistically indistinguishable."
    )

    low_marker = [r for r in rows if r["marker_rate"] < 90.0]
    if low_marker:
        lines.append(
            "  !! marker_rate < 90% on: " + ", ".join(r["run"] for r in low_marker)
            + " -- those rows are measuring output format, not correctness."
        )
    degen = [
        r for r in rows
        if isinstance(r["pct_examples_degenerate"], (int, float))
        and r["pct_examples_degenerate"] > 10.0
    ]
    if degen:
        lines.append(
            "  !! budget collapsed into the observation window on >10% of examples for: "
            + ", ".join(r["run"] for r in degen)
            + " -- on those examples the press keeps (at most) its own force-kept window, "
            "so the row measures 'keep the last k tokens', not the scoring method. "
            "Lower the compression ratio or the press window_size."
        )
    skipped = [
        r for r in rows
        if isinstance(r["pct_examples_press_skipped"], (int, float))
        and r["pct_examples_press_skipped"] > 0.0
    ]
    if skipped:
        lines.append(
            "  !! press was SKIPPED (context shorter than its window) on some examples for: "
            + ", ".join(r["run"] for r in skipped)
            + " -- those examples ran at full cache and are baseline rows in disguise."
        )
    # Batched output is batch-size dependent (different cuBLAS kernels -> different bf16
    # rounding -> flipped near-tied argmaxes, amplified by a press's discrete top-k), so
    # cells run at different batch sizes are not strictly on the same footing.
    batch_sizes = {r["batch_size"] for r in rows if r["batch_size"] is not None}
    if len(batch_sizes) > 1:
        lines.append(
            "  !! runs used DIFFERENT batch sizes "
            f"({sorted(batch_sizes)}). Batched generation is batch-size dependent, so "
            "part of any difference between these rows is a batching artifact rather "
            "than a compression effect. Re-run the sweep at one batch size."
        )
    backed_off = [r for r in rows if r["n_batch_size_backoffs"]]
    if backed_off:
        lines.append(
            "  !! OOM batch-size backoff occurred on: "
            + ", ".join(f"{r['run']} ({r['n_batch_size_backoffs']}x)" for r in backed_off)
            + " -- some rows were generated at a smaller batch than the rest of their "
            "own run. Check `batch_size_used` per row in predictions.jsonl; consider "
            "re-running those cells at a batch size that fits."
        )

    failed = [r for r in rows if r["n_generation_failed"]]
    if failed:
        lines.append(
            "  !! generation failures on: " + ", ".join(r["run"] for r in failed)
            + " -- not comparable to clean runs until re-run."
        )

    for r in rows:
        for w in r["_warnings"]:
            lines.append(f"  [{r['run']}] {w}")

    table = "\n".join(lines)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COMPARE_COLS)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in _COMPARE_COLS})
        print(f"Comparison written to {out_csv}")

    return table


def _expand(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        hits = sorted(_glob.glob(pattern))
        if hits:
            out.extend(Path(h) for h in hits)
        else:
            out.append(Path(pattern))
    return out


def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Score GSM8K predictions with extraction diagnostics."
    )
    parser.add_argument(
        "--runs", type=str, nargs="*", default=[],
        help="Run directories to compare (each holds predictions.jsonl + meta.json)",
    )
    parser.add_argument(
        "--predictions", type=str, nargs="*", default=[],
        help="Raw prediction file(s) or glob(s): *_df.csv or *.jsonl",
    )
    parser.add_argument("--out_csv", type=str, default=None, help="Comparison CSV path")
    args = parser.parse_args()

    if not args.predictions and not args.runs:
        parser.error("pass --runs and/or --predictions")

    if args.runs:
        run_dirs = [d for d in _expand(args.runs) if d.is_dir()]
        print(
            "\n"
            + build_comparison_table(
                run_dirs, Path(args.out_csv) if args.out_csv else None
            )
            + "\n"
        )

    if args.predictions:
        paths = _expand(args.predictions)
        rescore_many(paths, Path(args.out_csv) if args.out_csv else None)


if __name__ == "__main__":
    _cli_main()
