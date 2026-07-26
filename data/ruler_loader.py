"""RULER data loader — loads the vendored HF arrow splits.

Each RULER context-length split (4096 / 16384 / 32768) is a single
``datasets.Dataset`` saved to disk with 13 tasks × 500 examples::

    {
        "context": str,          # long synthetic context
        "question": str,         # the query
        "answer_prefix": str,    # prefix prepended to the generation turn
        "answer": list[str],     # acceptable reference answers
        "task": str,             # e.g. "niah_multikey_1"
        "max_new_tokens": int,   # per-example generation budget
    }

Matches the schema DefensiveKV's ``evaluation/evaluate.py`` consumes via
``load_from_disk(data_dir).to_pandas()`` — see
``kv_cache/DefensiveKV/kvpress/pipeline.py`` for the exact prompt protocol
this data feeds into.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

# All 13 RULER tasks (fixed set across every context length).
RULER_TASKS: List[str] = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "cwe",
    "fwe",
    "vt",
    "qa_1",
    "qa_2",
]


def load_ruler_dataset(
    data_dir: str, tasks: Optional[List[str]] = None
) -> List[Dict]:
    """Load a RULER context-length split, optionally filtered to *tasks*.

    Parameters
    ----------
    data_dir : str
        Path to a ``datasets.save_to_disk`` directory (e.g.
        ``.../defensivekv_dataset/ruler/4096``).
    tasks : list[str], optional
        Subset of ``RULER_TASKS`` to keep. ``None`` keeps all 13 tasks.

    Returns
    -------
    list[dict]
        Examples in on-disk order (stable, so ``num_samples`` capping is
        deterministic).
    """
    from datasets import load_from_disk

    ds = load_from_disk(data_dir)
    if tasks:
        unknown = set(tasks) - set(RULER_TASKS)
        if unknown:
            log.warning("ruler.tasks contains unknown task names: %s", sorted(unknown))
        ds = ds.filter(lambda ex: ex["task"] in tasks)

    examples = list(ds)
    log.info("Loaded RULER split %s: %d examples (tasks=%s)", data_dir, len(examples), tasks or "all")
    return examples
