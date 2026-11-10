"""LongGenBench metrics: the reference definitions, verbatim, plus one diagnostic.

Everything here mirrors ``mozhu621/LongGenBench`` @ ``45c320e``:
``Evalution/eval.py`` (block parsing, the judge prompt, the verdict, the
per-category accuracies) and ``Evalution/results/static.ipynb`` (STIC-1, STIC-2,
the length statistic). The paper's Table 2 is built from those.

The pipeline for one generated document
---------------------------------------
1. ``output = prefix + generation``; ``blocks = output.split('#*#')``.
2. ``parse_blocks``: each block is keyed by the first ``"<type> (\\d+)"`` in it
   (``"Week 24"``, ``"Floor 63"``, ``"Menu Week 12"``, ``"Block 51"``); the FIRST
   block with a given number wins.
3. Completion rate (CR): the share of ``1..number`` that got a block.
4. For every check in ``checks_once`` / ``checks_range`` / ``checks_periodic``
   (``{"24": "Attending a fashion week in Paris", ...}``) whose entry exists, a
   judge model reads that block and answers yes/no (:func:`judge_prompt`). A
   check whose entry is missing is not judged.

Aggregates (micro over checks; CR is a mean over documents)::

    STIC-1 = yes / judged          (static.ipynb "Ratio Yes/Results")
    STIC-2 = yes / all checks      (static.ipynb "Ratio Yes/Checks")
    wAvg   = CR * STIC-2           (paper, Table 2)
    acc_once / acc_range / acc_periodic = yes / judged, per category  (eval.py)
    Len    = mean word count over outputs with > 40 blocks and > 2000 words

The verdict is ``'yes' if 'yes' in response.strip().lower() else 'no'`` over up
to 50 generated tokens, so a judge that answers "no" and then runs on into an
invented "Example 4 ... Answer: yes" is counted as yes. That is the reference
rule and the headline keeps it. ``strict`` (the first word of the response) is
recorded beside it as a diagnostic: if the two disagree often, or differently
between arms, read the raw responses before quoting either.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from data.longgenbench_loader import BLOCK_SEP

CATEGORIES = ("once", "range", "periodic")

#: ``create_prompts`` in eval.py, verbatim (the paper's Appendix C: few-shot).
JUDGE_EXAMPLES = [
    "Example 1: Context: The district's new residential area will feature two 10-story apartment buildings, a shopping mall, and a park. The construction is set to begin at 9 AM, and the park will include a playground and jogging tracks. The plan also includes space for a small medical clinic, which will open from 10 AM to 6 PM. ### Instruction: Does this context include a medical clinic? Please answer with 'yes' or 'no' only. Answer: yes",  # noqa: E501
    "Example 2: Context: The menu for today's lunch at the office includes grilled turkey, mashed potatoes with gravy, roasted vegetables, and pumpkin pie for dessert. The meal will be served from 12 PM to 2 PM, and there will be a vegetarian option available. The meal is planned to accommodate 50 people, and the turkey will be served with cranberry sauce. ### Instruction: Does this context include mashed potatoes? Please answer with 'yes' or 'no' only. Answer: yes",  # noqa: E501
    "Example 3: Context: On April 15th, the weather was sunny with a high of 75°F. In the morning, I volunteered for a community cleanup from 9 AM to 12 PM. We collected trash and planted 20 new trees along the riverbank. After lunch, I helped organize the donation of clothes and food for a local shelter, where we served sandwiches and drinks. The day ended at 4 PM. ### Instruction: Does this context include long-distance running? Please answer with 'yes' or 'no' only. Answer: no",  # noqa: E501
]
#: ``SamplingParams(max_tokens=50)`` in eval.py.
JUDGE_MAX_NEW_TOKENS = 50
#: The paper's judge (§2.4). The repo's later eval.py switched to
#: Llama-3.3-70B-Instruct, which does not fit one A100 in fp16.
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# one document
# ---------------------------------------------------------------------------


def split_blocks(output: str) -> List[str]:
    """``process_output``: ``output.split('#*#')``."""
    return output.split(BLOCK_SEP)


def parse_blocks(blocks: Sequence[str], type_: str) -> Dict[int, str]:
    """``parse_blocks`` in eval.py: first block per entry number wins."""
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
    """``calculate_completion_rate``: percent of ``1..number`` present."""
    expected = set(range(1, number + 1))
    missing = expected - set(type_to_block)
    return (len(expected) - len(missing)) / len(expected) * 100


def judge_prompt(block: str, event_desc: str) -> str:
    """The judge prompt eval.py builds for one check, character for character."""
    return ("\n".join(JUDGE_EXAMPLES) + " \n ### Refer to the examples above for how to answer. \nContext: "
            + block + f"\n\n### Instruction: Now, for the following context, does it include the "
            f"{event_desc}? Please answer with 'yes' or 'no' only. Answer: ")


def verdict(response: str) -> str:
    """``evaluate_accuracy``: ``'yes' if 'yes' in response.strip().lower() else 'no'``."""
    return "yes" if "yes" in response.strip().lower() else "no"


def strict_verdict(response: str) -> Optional[str]:
    """Diagnostic: the response's first word, if it is yes or no; else None."""
    m = re.match(r"[^A-Za-z]*([A-Za-z]+)", response)
    w = m.group(1).lower() if m else ""
    return w if w in ("yes", "no") else None


def document_checks(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every check of one document, with the block it is judged on (or None).

    ``rec`` carries ``output``, ``type``, ``checks_once/range/periodic``. Order is
    eval.py's: once, range, periodic; within each, the dataset's dict order.
    """
    blocks = parse_blocks(split_blocks(rec["output"]), rec["type"]) if rec.get("output") else {}
    out = []
    for cat in CATEGORIES:
        for ident, desc in rec[f"checks_{cat}"].items():
            i = int(ident)
            out.append({"category": cat, "identifier": i, "desc": desc,
                        "block": blocks.get(i)})
    return out


def word_count(output: str) -> int:
    """``len(output.split())``, as inference.py records it."""
    return len(output.split())


def counts_length(rec: Dict[str, Any]) -> bool:
    """static.ipynb: an output enters the length mean if > 40 blocks and > 2000 words."""
    return len(split_blocks(rec["output"])) > 40 and word_count(rec["output"]) > 2000


# ---------------------------------------------------------------------------
# aggregates
# ---------------------------------------------------------------------------


def doc_stats(doc: Dict[str, Any], judged: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One document's contribution to every aggregate (parsed once, summed later).

    ``judged``: that document's judged checks, ``[{"category", "verdict",
    "strict"}, ...]``. A document whose ``output`` is None (a failed generation)
    has CR 0 and no judged check, so it can only lower a score.
    """
    out = doc.get("output")
    blocks = parse_blocks(split_blocks(out), doc["type"]) if out else {}
    st: Dict[str, Any] = {"cr": completion_rate(blocks, doc["number"]), "yes_strict": 0,
                          "len": word_count(out) if out and counts_length(doc) else None}
    for c in CATEGORIES:
        st[f"tot_{c}"] = len(doc[f"checks_{c}"])
        st[f"jud_{c}"] = 0
        st[f"yes_{c}"] = 0
    for j in judged:
        st[f"jud_{j['category']}"] += 1
        st[f"yes_{j['category']}"] += j["verdict"] == "yes"
        st["yes_strict"] += j.get("strict") == "yes"
    return st


def aggregate(stats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The reference aggregates from per-document stats (see the module docstring)."""
    n_docs = len(stats)
    if not n_docs:
        return {"n_docs": 0}
    tot = {c: sum(s[f"tot_{c}"] for s in stats) for c in CATEGORIES}
    jud = {c: sum(s[f"jud_{c}"] for s in stats) for c in CATEGORIES}
    yes = {c: sum(s[f"yes_{c}"] for s in stats) for c in CATEGORIES}
    T, J, Y = sum(tot.values()), sum(jud.values()), sum(yes.values())
    CR = sum(s["cr"] for s in stats) / n_docs
    stic2 = 100.0 * Y / T if T else float("nan")
    lens = [s["len"] for s in stats if s["len"] is not None]
    out = {
        "n_docs": n_docs,
        "n_checks": T, "n_judged": J, "n_yes": Y,
        "CR": round(CR, 2),
        "STIC-1": round(100.0 * Y / J, 2) if J else float("nan"),
        "STIC-2": round(stic2, 2),
        "wAvg": round(CR * stic2 / 100.0, 2),
        "STIC-2_strict": round(100.0 * sum(s["yes_strict"] for s in stats) / T, 2) if T else float("nan"),
        "Len": round(sum(lens) / len(lens)) if lens else None,
        "n_len": len(lens),
    }
    for c in CATEGORIES:
        out[f"acc_{c}"] = round(100.0 * yes[c] / jud[c], 2) if jud[c] else float("nan")
    return out


def summarize(docs: Sequence[Dict[str, Any]],
              judgments: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Reference aggregates over documents and their judged checks.

    ``docs``: runner records (``id``, ``type``, ``number``, ``output``, checks).
    ``judgments``: ``{id: [{"category", "identifier", "verdict", "strict"}, ...]}``,
    one entry per JUDGED check (checks without a block are not in it).
    """
    return aggregate([doc_stats(d, judgments.get(d["id"], [])) for d in docs])


def bootstrap_delta(docs_a: Sequence[Dict[str, Any]], judg_a: Dict[int, List[Dict[str, Any]]],
                    docs_b: Sequence[Dict[str, Any]], judg_b: Dict[int, List[Dict[str, Any]]],
                    keys: Iterable[str] = ("CR", "STIC-2", "wAvg"), iters: int = 2000,
                    seed: int = 0) -> Dict[str, Dict[str, float]]:
    """``summarize(b) - summarize(a)`` with a paired cluster bootstrap over documents.

    Documents are resampled (the same draw for both arms) and each statistic is
    recomputed from the resampled documents' counts, so the ratio-of-sums
    definitions are kept rather than averaged per document.
    """
    ids = [d["id"] for d in docs_a]
    assert ids == [d["id"] for d in docs_b], "arms must be paired document by document"
    keys = list(keys)
    sa = [doc_stats(d, judg_a.get(d["id"], [])) for d in docs_a]
    sb = [doc_stats(d, judg_b.get(d["id"], [])) for d in docs_b]
    point = {k: aggregate(sb)[k] - aggregate(sa)[k] for k in keys}
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = {k: [] for k in keys}
    n = len(ids)
    for _ in range(iters):
        pick = [rng.randrange(n) for _ in range(n)]
        ga, gb = aggregate([sa[i] for i in pick]), aggregate([sb[i] for i in pick])
        for k in keys:
            draws[k].append(gb[k] - ga[k])
    out = {}
    for k in keys:
        v = sorted(draws[k])
        out[k] = {"delta": round(point[k], 2), "lo": round(v[int(0.025 * len(v))], 2),
                  "hi": round(v[min(len(v) - 1, int(0.975 * len(v)))], 2)}
    return out


# ---------------------------------------------------------------------------
# files written by the runner and the judge
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n")
            if line.strip()]


def judgments_by_doc(rows: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(int(r["id"]), []).append(r)
    return out
