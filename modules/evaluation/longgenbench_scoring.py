"""LongGenBench scoring: SCOPE's metric, verbatim, plus what it cannot see.

The headline metric (``acc``) is ``eval_longgenbench.py`` from SCOPE
(``Linking-ai/SCOPE``), reimplemented with the same regular expressions and the
same arithmetic, so a number here is comparable with its published table. That
is checked, not assumed: :func:`rescore_reference` re-scores SCOPE's own
full-cache predictions and must give back the scores its ``results.csv`` prints
(``tests/test_longgenbench.py`` and the notebook's preflight both run it).

What the reference metric does
------------------------------
For one example (one prompt with K questions):

* GSM8K: gold is ``#### (\\d+)`` from each GSM8K solution; predictions are
  ``re.findall(r'Answer_\\d+:\\s*.*?answer is (\\d+)', pred, re.DOTALL)``.
* MMLU / CSQA: gold is the letter; predictions are
  ``re.findall(r'Answer_\\d+:\\s*.*?answer is \\((.*?)\\)', pred)`` (no DOTALL).
* The i-th prediction is compared with the i-th gold (``zip``), and the
  example's accuracy is ``correct / K``. The task score is the mean over
  examples, times 100, rounded to 4 places.

So pairing is by POSITION. A model that skips one question shifts every later
answer onto the wrong gold, and one ``answer is $18`` (which ``(\\d+)`` does not
match) lets the lazy DOTALL match run on into the next answer's block and take
its number. Both are properties of the benchmark as published; the headline
keeps them.

What is reported beside it
--------------------------
``acc_aligned`` pairs each ``Answer_N`` block with question ``N`` (the demos
take the first numbers, ``FIRST_QUESTION_INDEX``) and applies the SAME answer
pattern inside that block only. It differs from ``acc`` in nothing but how
answers are paired with questions, so ``acc - acc_aligned`` is exactly the
alignment artefact. ``coverage`` is the fraction of questions that got an
``Answer_N`` block at all: a cache that loses its place in a long answer shows
up there first. If ``acc`` and ``acc_aligned`` disagree, or the disagreement
moves between arms, read the per-example records before quoting either.

CLI::

    python -m modules.evaluation.longgenbench_scoring --runs DIR [DIR ...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from data.longgenbench_loader import FIRST_QUESTION_INDEX, TASKS

# --- eval_longgenbench.py, verbatim patterns --------------------------------
_GSM8K_GOLD = re.compile(r'####\s*(\d+)')
_GSM8K_PRED = re.compile(r'Answer_\d+:\s*.*?answer is (\d+)', re.DOTALL)
_CHOICE_PRED = re.compile(r'Answer_\d+:\s*.*?answer is \((.*?)\)')

# --- per-block versions of the same answer patterns (alignment diagnostic) ---
_BLOCK_HEAD = re.compile(r'Answer_(\d+):')
_GSM8K_IN_BLOCK = re.compile(r'\s*.*?answer is (\d+)', re.DOTALL)
_CHOICE_IN_BLOCK = re.compile(r'\s*.*?answer is \((.*?)\)')


def _check_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"unknown LongGenBench task {task!r}; one of {TASKS}")


def expected_answers(task: str, answers: Sequence[Any]) -> List[Optional[str]]:
    """Gold answers as the reference compares them (strings; ``None`` = no gold)."""
    _check_task(task)
    if task == "gsm8k":
        out = []
        for a in answers:
            m = _GSM8K_GOLD.search(str(a))
            out.append(m.group(1) if m else None)
        return out
    return [None if a is None else str(a) for a in answers]


def scope_predictions(task: str, pred: Optional[str]) -> List[str]:
    """``extract_predicted_answers`` / ``extract_predicted_choices``, verbatim."""
    _check_task(task)
    if not pred:
        return []
    return (_GSM8K_PRED if task == "gsm8k" else _CHOICE_PRED).findall(pred)


def scope_example_accuracy(task: str, pred: Optional[str], answers: Sequence[Any]) -> float:
    """``compare_answers`` / ``compare_choices``: positional, ``correct / K``."""
    exp = expected_answers(task, answers)
    got = scope_predictions(task, pred)
    correct = sum(1 for e, p in zip(exp, got) if e == p)
    return correct / len(exp) if exp else 0.0


def scope_task_score(task: str, preds: Sequence[Optional[str]],
                     answers: Sequence[Sequence[Any]]) -> float:
    """``scorer()``: ``round(100 * mean(per-example accuracy), 4)``."""
    if not preds:
        return float("nan")
    accs = [scope_example_accuracy(task, p, a) for p, a in zip(preds, answers)]
    return round(100.0 * sum(accs) / len(accs), 4)


def answer_blocks(pred: Optional[str]) -> Dict[int, str]:
    """``{N: text after "Answer_N:" up to the next header}``; first block per N wins.

    First, not last: a model that loses its place and starts numbering again does
    not get a second attempt at a question it already answered.
    """
    if not pred:
        return {}
    heads = list(_BLOCK_HEAD.finditer(pred))
    out: Dict[int, str] = {}
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(pred)
        out.setdefault(int(m.group(1)), pred[m.end():end])
    return out


def aligned_example(task: str, pred: Optional[str], answers: Sequence[Any]) -> Dict[str, Any]:
    """Per-question result with answers paired to questions by their number."""
    exp = expected_answers(task, answers)
    blocks = answer_blocks(pred)
    pat = _GSM8K_IN_BLOCK if task == "gsm8k" else _CHOICE_IN_BLOCK
    first = FIRST_QUESTION_INDEX[task]
    got: List[Optional[str]] = []
    present = 0
    for j in range(len(exp)):
        body = blocks.get(first + j)
        if body is None:
            got.append(None)
            continue
        present += 1
        m = pat.match(body)
        got.append(m.group(1) if m else None)
    correct = [e is not None and e == g for e, g in zip(exp, got)]
    n = len(exp)
    return {
        "acc_aligned": (sum(correct) / n) if n else 0.0,
        "coverage": (present / n) if n else 0.0,
        "extracted": (sum(g is not None for g in got) / n) if n else 0.0,
        "n_blocks": len(blocks),
        "correct": correct,
    }


def score_example(task: str, pred: Optional[str], answers: Sequence[Any]) -> Dict[str, Any]:
    """Everything one example contributes: the reference accuracy and the diagnostics."""
    al = aligned_example(task, pred, answers)
    return {
        "acc": scope_example_accuracy(task, pred, answers),
        "acc_aligned": al["acc_aligned"],
        "coverage": al["coverage"],
        "extracted": al["extracted"],
        "n_scope_matches": len(scope_predictions(task, pred)),
        "n_questions": len(answers),
        "failed": pred is None,
    }


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(task: str, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Task-level numbers over runner records (``pred``, ``answers`` and stats).

    ``acc`` is the reference score over EVERY record; a failed generation
    (``pred`` null, e.g. an OOM under ``skip_oom``) scores 0 there, as an empty
    prediction does in the reference, and is counted in ``n_failed`` so it is
    never mistaken for a wrong answer.
    """
    per = [score_example(task, r.get("pred"), r["answers"]) for r in records]
    n = len(per)
    out: Dict[str, Any] = {
        "task": task,
        "n_examples": n,
        "n_questions": sum(p["n_questions"] for p in per),
        "n_failed": sum(p["failed"] for p in per),
        "acc": round(100.0 * _mean(p["acc"] for p in per), 4) if n else float("nan"),
        "acc_aligned": round(100.0 * _mean(p["acc_aligned"] for p in per), 4) if n else float("nan"),
        "coverage": round(100.0 * _mean(p["coverage"] for p in per), 2) if n else float("nan"),
        "extracted": round(100.0 * _mean(p["extracted"] for p in per), 2) if n else float("nan"),
    }
    gen = [r["gen_tokens"] for r in records if r.get("gen_tokens") is not None]
    if gen:
        out["mean_gen_tokens"] = round(_mean(gen), 1)
        out["pct_hit_token_cap"] = round(
            100.0 * _mean(1.0 if r.get("hit_token_cap") else 0.0 for r in records), 1)
    return out


# ---------------------------------------------------------------------------
# run directories written by LongGenBenchRunner
# ---------------------------------------------------------------------------


def read_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n")
            if line.strip()]


def score_run_dir(run_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """``{task: summary + meta essentials}`` for every ``<task>.jsonl`` in *run_dir*."""
    run_dir = Path(run_dir)
    out: Dict[str, Dict[str, Any]] = {}
    for task in TASKS:
        jl = run_dir / f"{task}.jsonl"
        if not jl.is_file():
            continue
        s = summarize(task, read_records(jl))
        meta_p = run_dir / f"{task}.meta.json"
        if meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            s["read_gate"] = (meta.get("read_gate") or {}).get("verdict")
            s["read_fraction"] = (meta.get("read_gate") or {}).get("read_fraction")
            s["cache_type"] = meta.get("cache_type")
            s["effective_retention"] = meta.get("effective_retention")
            s["setting"] = meta.get("setting")
        out[task] = s
    return out


def rescore_reference(data_dir: str | Path, setting: str, task: str) -> float:
    """Re-score SCOPE's own full-cache predictions with this module.

    Must equal ``published_reference_score(setting, task)``: that is the proof
    the metric here IS the reference metric, regexes and rounding included.
    """
    from data.longgenbench_loader import load_reference
    rows = load_reference(data_dir, setting, task)
    return scope_task_score(task, [r["pred"] for r in rows], [r["answers"] for r in rows])


def _cli_main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", required=True, help="run directories")
    p.add_argument("--json", default=None, help="also write the table as JSON here")
    a = p.parse_args()
    table = {str(d): score_run_dir(d) for d in a.runs}
    cols = ("n_examples", "acc", "acc_aligned", "coverage", "mean_gen_tokens",
            "pct_hit_token_cap", "n_failed", "read_gate")
    print(f"{'run':40s} {'task':6s} " + " ".join(f"{c:>12s}" for c in cols))
    for d, tasks in table.items():
        for t, s in tasks.items():
            print(f"{Path(d).name[:40]:40s} {t:6s} " +
                  " ".join(f"{str(s.get(c, '-')):>12s}" for c in cols))
    if a.json:
        Path(a.json).write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    _cli_main()
