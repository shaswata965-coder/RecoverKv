"""The paper's metrics, exactly as its code computes them. Standard library only.

Per document (``eval.py`` / ``static.ipynb``):

1. ``blocks = output.split('#*#')`` where ``output = prefix + generation``.
2. ``parse_blocks``: a block is keyed by the first ``"<type> (\\d+)"`` in it
   (``"Week 24"``, ``"Menu Week 12"``); the FIRST block with a number wins.
3. Completion: the share of ``1..number`` that got a block.
4. Every instruction in ``checks_once`` / ``checks_range`` / ``checks_periodic``
   whose entry exists is judged yes/no (``judge.py``); a missing entry is not
   judged.

The three metrics (paper §2.4), and the two derived Table 2 columns::

    CR     = mean over documents of (entries present / entries required)   x 100
    STIC-1 = yes / instructions judged    (static.csv "Ratio Yes/Results") x 100
    STIC-2 = yes / all instructions       (static.csv "Ratio Yes/Checks")  x 100
    wAvg   = CR x STIC-2 / 100                                    (Table 2)
    Len    = mean words over outputs with > 40 blocks and > 2000 words (static.ipynb)

These formulas reproduce 16 of the 18 rows of the paper's Table 2 from the
authors' own counts, digit for digit; the other two (GPT-4o, GPT-4o-mini) carry
typos in the paper itself (``tests/test_longgenbench_eval.py``).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .reference import BLOCK_SEP

CATEGORIES = ("once", "range", "periodic")


# ---------------------------------------------------------------------------
# one document
# ---------------------------------------------------------------------------


def split_blocks(output: str) -> List[str]:
    return output.split(BLOCK_SEP)


def parse_blocks(blocks: Sequence[str], type_: str) -> Dict[int, str]:
    """eval.py ``parse_blocks``: first ``"<type> N"`` per block; first block per N wins."""
    pattern = rf"{type_} (\d+)"
    out: Dict[int, str] = {}
    for block in blocks:
        m = re.search(pattern, block)
        if m:
            ident = int(m.group(1))
            if ident not in out or out[ident] is None:
                out[ident] = block
    return out


def completion_rate(type_to_block: Dict[int, str], number: int) -> float:
    """eval.py ``calculate_completion_rate`` (percent)."""
    expected = set(range(1, number + 1))
    missing = expected - set(type_to_block)
    return (len(expected) - len(missing)) / len(expected) * 100


def word_count(output: str) -> int:
    """inference.py: ``len(output.split())``."""
    return len(output.split())


def document_checks(doc: Dict[str, Any], output: Optional[str]) -> List[Dict[str, Any]]:
    """Every instruction of one document with the block it is judged on (None = not judged).

    Order is eval.py's: once, range, periodic; within each, the dataset's order.
    """
    blocks = parse_blocks(split_blocks(output), doc["type"]) if output else {}
    out = []
    for cat in CATEGORIES:
        for ident, desc in doc[f"checks_{cat}"].items():
            out.append({"category": cat, "identifier": int(ident), "desc": desc,
                        "block": blocks.get(int(ident))})
    return out


def doc_stats(doc: Dict[str, Any], output: Optional[str], verdicts: Sequence[str]) -> Dict[str, Any]:
    """One document's counts. ``verdicts``: "yes"/"no" for each judged instruction.

    A document with no output (a failed generation) has completion 0 and nothing
    judged, so it can only lower a score.
    """
    blocks = parse_blocks(split_blocks(output), doc["type"]) if output else {}
    n_checks = sum(len(doc[f"checks_{c}"]) for c in CATEGORIES)
    n_judged = sum(1 for c in document_checks(doc, output) if c["block"] is not None)
    if len(verdicts) != n_judged:
        raise ValueError(f"document {doc.get('id')}: {len(verdicts)} verdicts for {n_judged} judged instructions")
    long_enough = bool(output) and len(split_blocks(output)) > 40 and word_count(output) > 2000
    return {"cr": completion_rate(blocks, doc["number"]), "checks": n_checks, "judged": n_judged,
            "yes": sum(v == "yes" for v in verdicts), "len": word_count(output) if long_enough else None}


def aggregate(stats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """CR, STIC-1, STIC-2 and Table 2's Len and wAvg from per-document counts."""
    n = len(stats)
    if not n:
        return {"n_docs": 0}
    C = sum(s["checks"] for s in stats)
    J = sum(s["judged"] for s in stats)
    Y = sum(s["yes"] for s in stats)
    CR = sum(s["cr"] for s in stats) / n
    lens = [s["len"] for s in stats if s["len"] is not None]
    return from_counts(CR, C, J, Y, lens, n)


def from_counts(CR: float, checks: int, judged: int, yes: int,
                lens: Sequence[int] = (), n_docs: Optional[int] = None) -> Dict[str, Any]:
    """The aggregate formulas on their own, so they can be checked against the
    authors' published counts (static.csv) without any documents."""
    stic2 = 100.0 * yes / checks if checks else float("nan")
    return {
        "n_docs": n_docs, "checks": checks, "judged": judged, "yes": yes,
        "CR": CR,
        "STIC-1": 100.0 * yes / judged if judged else float("nan"),
        "STIC-2": stic2,
        "Len": sum(lens) / len(lens) if lens else None,
        "n_len": len(lens),
        "wAvg": CR * stic2 / 100.0,
    }


def rounded(s: Dict[str, Any]) -> Dict[str, Any]:
    """Table 2's precision: one decimal for percentages, whole words for Len."""
    out = dict(s)
    for k in ("CR", "STIC-1", "STIC-2", "wAvg"):
        if isinstance(out.get(k), float) and out[k] == out[k]:
            out[k] = round(out[k], 1)
    if out.get("Len") is not None:
        out["Len"] = int(round(out["Len"]))       # Table 2: 7295.7 -> 7296
    return out


def paired_bootstrap(sa: Sequence[Dict[str, Any]], sb: Sequence[Dict[str, Any]],
                     keys: Iterable[str] = ("CR", "STIC-1", "STIC-2", "wAvg"), iters: int = 2000,
                     seed: int = 0) -> Dict[str, Dict[str, float]]:
    """``b - a`` per metric with a paired cluster bootstrap over documents.

    ``sa`` / ``sb``: :func:`doc_stats` for the same documents in the same order.
    Each draw resamples documents and recomputes the ratio-of-sums metrics.
    """
    assert len(sa) == len(sb)
    keys = list(keys)
    pa, pb = aggregate(sa), aggregate(sb)
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(iters):
        pick = [rng.randrange(len(sa)) for _ in range(len(sa))]
        ga, gb = aggregate([sa[i] for i in pick]), aggregate([sb[i] for i in pick])
        for k in keys:
            if ga[k] == ga[k] and gb[k] == gb[k]:
                draws[k].append(gb[k] - ga[k])
    out = {}
    for k in keys:
        v = sorted(draws[k]) or [float("nan")]
        out[k] = {"delta": pb[k] - pa[k], "lo": v[int(0.025 * len(v))],
                  "hi": v[min(len(v) - 1, int(0.975 * len(v)))]}
    return out


# ---------------------------------------------------------------------------
# input files
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n") if line.strip()]


def load_generations(path: str | Path, docs: Sequence[Dict[str, Any]]) -> Dict[int, Optional[str]]:
    """``{id: output}`` from a generations file, in either accepted format.

    * JSONL, one object per document: ``{"id": <index in the dataset file>,
      "output": <prefix + generation>}`` (other fields ignored; ``output`` null =
      a failed generation). This is what any baseline should write.
    * The reference's own ``inference.py`` output (a JSON list with ``input``,
      ``output_blocks``, ...): matched to documents by prompt, and
      ``output = '#*#'.join(output_blocks)``, which is exactly the text it split.
    """
    path = Path(path)
    text = path.read_bytes().decode("utf-8")
    by_prompt = {d["prompt"]: d["id"] for d in docs}
    known = {d["id"] for d in docs}
    out: Dict[int, Optional[str]] = {}
    if text.lstrip().startswith("["):
        for r in json.loads(text):
            i = by_prompt.get(r.get("input"))
            if i is None:
                raise ValueError(f"{path}: a record's 'input' matches no prompt of this dataset")
            out[i] = BLOCK_SEP.join(r["output_blocks"])
        return out
    for r in _read_jsonl(path):
        i = int(r["id"])
        if i not in known:
            raise ValueError(f"{path}: id {i} is not a document of this dataset")
        if i in out:
            raise ValueError(f"{path}: id {i} appears twice")
        out[i] = r.get("output")
    return out


def load_verdicts(path: str | Path) -> Dict[int, List[Dict[str, Any]]]:
    """``{id: [verdict rows in eval.py order]}`` from a ``*.verdicts.jsonl``."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    for r in _read_jsonl(Path(path)):
        out.setdefault(int(r["id"]), []).append(r)
    return out
