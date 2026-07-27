"""Build the GSM8K chain-of-thought eval set, with a manifest that makes runs comparable.

This is the *only* place GSM8K data enters the project. It downloads ``openai/gsm8k``
(config ``main``), applies the CoT prompt below, and writes three artifacts:

    <out>/gsm8k_cot.jsonl   human-inspectable mirror; the SHA is computed over this
    <out>/hf/               ``datasets.save_to_disk`` dir (so ``evaluate.py --data_dir <out>/hf``
                            also works, and ``load_from_disk`` round-trips)
    <out>/manifest.json     prompt style, max_new_tokens, and ``dataset_sha``

Why the manifest matters
------------------------
The accuracy of a KV-compression run is only comparable across compression ratios if
every ratio saw byte-identical prompts. The runner records ``dataset_sha`` in its
``meta.json`` and the scorer REFUSES to build a comparison table across runs whose
``dataset_sha`` disagree. That is the mechanism that makes "snapkv@0.8 vs
defensivekv@0.8" a controlled comparison rather than two unrelated numbers.

Prompt design (three deliberate choices)
----------------------------------------
1.  ``max_new_tokens = 512``. At 256 a non-trivial share of generations are clipped
    mid-reasoning and never emit an answer, and the clip rate rises with the KV budget
    (a healthier cache reasons longer) — which bends the accuracy curve on its own,
    independent of whether the compression method is any good.
2.  The prompt demands GSM8K's native terminal marker ``#### <number>`` on its own
    final line. One unambiguous token sequence, no ``$`` / ``**`` / prose decoration to
    parse around, and it doubles as a natural stopping point.
3.  ``STOP_STRINGS`` is exported so the runner can halt the moment the model starts
    inventing a *follow-up* problem. This is what keeps the higher token cap from
    costing wall-clock: typical generations still stop in ~150 tokens, only the long
    ones use the headroom. It also removes run-on continuations at the source, which
    is the single largest source of scoring noise (see ``calculate_metrics.py``).

What ``context`` vs ``question`` means here (kvpress-specific, load-bearing)
---------------------------------------------------------------------------
The kvpress pipeline compresses **only** ``context``: it prefills the context under
the press, then feeds ``question`` uncompressed. So the split decides what the
compression ratio actually acts on.

*   ``context = SYSTEM_PROMPT`` and ``question = "Question: ..."`` (the DefensiveKV
    convention) means the budget knob acts on ~55 tokens of constant boilerplate and
    the problem statement is **never compressed**. Accuracy is then almost invariant
    to the compression ratio, and reporting it as "GSM8K under 80% compression" would
    be wrong.
*   Merging the problem into ``context`` (the runner's default, equivalent to
    ``evaluate.py --compress_questions True``) makes the budget act on the problem
    itself, which is the thing worth measuring.

Both columns are written so either mode is available; the runner records which one it
used in ``meta.json``. See ``press_budget.py`` for what the ratio then buys.

Usage
-----
    python -m gsm8k.create_huggingface_dataset --out data/gsm8k_cot
    python -m gsm8k.create_huggingface_dataset --out data/gsm8k_cot --limit 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

SYSTEM_PROMPT = (
    "Solve the following math word problem. Show your step-by-step reasoning, "
    "then give the final numeric answer.\n\n"
    "End your response with the final answer on its own line in exactly this "
    "form:\n#### <number>\n\n"
    "Write only the bare number after ####: no currency symbols, no units, no "
    "commas, no formatting."
)

QUESTION_TEMPLATE = "Question:\n{q}\n\nReasoning:\n"

#: Halt generation once the model starts a NEW problem. Note ``####`` is deliberately
#: absent: it introduces the answer, so stopping on it would cut the number off.
STOP_STRINGS = ["\n\nQuestion:", "\nQuestion:", "\n\nProblem:"]

MAX_NEW_TOKENS = 512

HF_DATASET = "openai/gsm8k"
HF_CONFIG = "main"
TASK_NAME = "gsm8k"

JSONL_NAME = "gsm8k_cot.jsonl"
MANIFEST_NAME = "manifest.json"
ARROW_SUBDIR = "hf"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def extract_numeric_answer(answer_text: str) -> str:
    """Pull the reference number out of a GSM8K ``answer`` field."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip()


def _records_to_jsonl(records: List[Dict]) -> str:
    """Serialise records deterministically (sorted keys, no whitespace drift)."""
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build_gsm8k_cot(
    out_dir: str | Path,
    split: str = "test",
    limit: Optional[int] = None,
) -> Path:
    """Download GSM8K and write the CoT prompt set to *out_dir*.

    Order is the HuggingFace on-disk order, never shuffled, so ``limit`` and the
    runner's sharding are reproducible across machines.
    """
    from datasets import Dataset, load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {HF_DATASET}/{HF_CONFIG} split={split} ...")
    raw = load_dataset(HF_DATASET, HF_CONFIG, split=split)

    records: List[Dict] = []
    for row in raw:
        records.append(
            {
                "context": SYSTEM_PROMPT,
                "question": QUESTION_TEMPLATE.format(q=row["question"]),
                "answer_prefix": "",
                "answer": [extract_numeric_answer(row["answer"])],
                "task": TASK_NAME,
                "max_new_tokens": MAX_NEW_TOKENS,
                "all_classes": [],
            }
        )

    if limit is not None:
        if limit < 1:
            raise ValueError(f"--limit must be >= 1, got {limit}")
        records = records[:limit]

    # 1. jsonl mirror — this is what the SHA is computed over, and what the runner reads.
    payload = _records_to_jsonl(records)
    (out_dir / JSONL_NAME).write_text(payload, encoding="utf-8")
    dataset_sha = _sha256_bytes(payload.encode("utf-8"))

    # 2. arrow dir, so `evaluate.py --data_dir <out>/hf` works too.
    Dataset.from_list(records).save_to_disk(str(out_dir / ARROW_SUBDIR))

    # 3. manifest
    manifest = {
        "hf_dataset": HF_DATASET,
        "hf_config": HF_CONFIG,
        "split": split,
        "num_examples": len(records),
        "limit": limit,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_style": "chain_of_thought",
        "system_prompt": SYSTEM_PROMPT,
        "question_template": QUESTION_TEMPLATE,
        "stop_strings": STOP_STRINGS,
        "dataset_sha": dataset_sha,
    }
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nGSM8K CoT dataset ready: {out_dir}")
    print(f"  examples       {len(records)}")
    print(f"  max_new_tokens {MAX_NEW_TOKENS}")
    print(f"  dataset_sha    {dataset_sha}")
    return out_dir


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def read_manifest(data_dir: str | Path) -> Dict:
    """Read the manifest written by :func:`build_gsm8k_cot`."""
    path = Path(data_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the dataset first:\n"
            f"    python -m gsm8k.create_huggingface_dataset --out {data_dir}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_gsm8k_dataset(
    data_dir: str | Path,
    num_samples: int | str = "max",
    shard: int = 0,
    num_shards: int = 1,
) -> List[Dict]:
    """Load the built CoT dataset in stable on-disk order.

    Parameters
    ----------
    num_samples
        ``"max"`` for the full split, or an int cap applied BEFORE sharding so that a
        capped run is the same set of problems at every compression ratio.
    shard, num_shards
        Contiguous sharding for multi-GPU splits — contiguous, not strided, so a
        single shard is still a coherent prefix of the dataset.
    """
    data_dir = Path(data_dir)
    manifest = read_manifest(data_dir)

    jsonl_path = data_dir / JSONL_NAME
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if isinstance(num_samples, int) and num_samples >= 0:
        records = records[:num_samples]

    if num_shards > 1:
        if not (0 <= shard < num_shards):
            raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")
        total = len(records)
        per = (total + num_shards - 1) // num_shards
        start, end = shard * per, min((shard + 1) * per, total)
        records = records[start:end]
        print(
            f"GSM8K shard {shard}/{num_shards}: examples [{start}, {end}) of {total}"
        )

    print(
        f"Loaded {len(records)} GSM8K examples from {data_dir} "
        f"(dataset_sha={manifest['dataset_sha'][:12]})"
    )
    return records


def split_context_question(
    example: Dict, compress_questions: bool = True
) -> "tuple[str, str]":
    """Return ``(context, question)`` for the kvpress pipeline.

    ``compress_questions=True`` (default) merges the problem statement into the
    compressed context, so the compression ratio acts on the problem rather than on
    ~55 tokens of constant boilerplate. This mirrors
    ``evaluate.py --compress_questions``; the empty question is fine because the
    pipeline still appends the chat template's assistant header to it.
    """
    if compress_questions:
        return example["context"] + example["question"], ""
    return example["context"], example["question"]


def _cli_main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="data/gsm8k_cot", help="Output directory")
    p.add_argument("--split", default="test", help="GSM8K split (default: test)")
    p.add_argument(
        "--limit", type=int, default=None,
        help="Keep only the first N examples (smoke runs). Default: all 1319.",
    )
    args = p.parse_args()
    build_gsm8k_cot(args.out, args.split, args.limit)


if __name__ == "__main__":
    _cli_main()
