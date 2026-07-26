"""GSM8K data loader — download once, build the CoT prompt set, load deterministically.

This is the *only* place GSM8K data enters the project. It downloads
``openai/gsm8k`` (config ``main``), applies the chain-of-thought prompt from
``modules/evaluation/gsm8k_dataset.py``, and writes a
``datasets.save_to_disk`` directory plus a jsonl mirror and a manifest carrying
the SHA-256 of the built data.

Why the manifest matters: the accuracy of a KV-compression run is only
comparable across budgets if every budget saw byte-identical prompts. The runner
records ``dataset_sha`` in its meta sidecar, and the scorer refuses to build a
comparison table across runs whose ``dataset_sha`` disagree. That is the
mechanism that makes "20% vs 80% budget" a controlled comparison rather than two
unrelated numbers.

Schema (matches ``data/ruler_loader.py`` and DefensiveKV's evaluate.py):

    {
        "context":        str,        # the CoT system prompt (constant)
        "question":       str,        # "Question:\\n<q>\\n\\nReasoning:\\n"
        "answer_prefix":  str,        # "" for GSM8K
        "answer":         list[str],  # ["18"] — the bare reference number
        "task":           str,        # "gsm8k"
        "max_new_tokens": int,        # 512
    }

Usage:
    python -m data.gsm8k_loader --out data/gsm8k_cot
    python -m data.gsm8k_loader --out data/gsm8k_cot --limit 200   # smoke subset
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from modules.evaluation.gsm8k_dataset import (
    MAX_NEW_TOKENS,
    QUESTION_TEMPLATE,
    SYSTEM_PROMPT,
    extract_numeric_answer,
)
from utils.logger import get_logger

log = get_logger(__name__)

HF_DATASET = "openai/gsm8k"
HF_CONFIG = "main"
TASK_NAME = "gsm8k"

#: Files written by :func:`build_gsm8k_cot` into the output directory.
JSONL_NAME = "gsm8k_cot.jsonl"
MANIFEST_NAME = "manifest.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _records_to_jsonl(records: List[Dict]) -> str:
    """Serialise records to jsonl deterministically (sorted keys, no ws drift)."""
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    )


def build_gsm8k_cot(
    out_dir: str | Path,
    split: str = "test",
    limit: Optional[int] = None,
) -> Path:
    """Download GSM8K and write the CoT prompt set to *out_dir*.

    Order is the HuggingFace on-disk order, never shuffled, so ``limit`` and the
    runner's sharding are reproducible across machines.

    Returns the output directory.
    """
    from datasets import Dataset, load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s/%s split=%s ...", HF_DATASET, HF_CONFIG, split)
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
            }
        )

    if limit is not None:
        if limit < 1:
            raise ValueError(f"--limit must be >= 1, got {limit}")
        records = records[:limit]

    # 1. jsonl mirror (human-inspectable, and what the SHA is computed over)
    payload = _records_to_jsonl(records)
    jsonl_path = out_dir / JSONL_NAME
    jsonl_path.write_text(payload, encoding="utf-8")
    dataset_sha = _sha256_bytes(payload.encode("utf-8"))

    # 2. arrow directory (what the runner loads)
    Dataset.from_list(records).save_to_disk(str(out_dir / "hf"))

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
        "dataset_sha": dataset_sha,
    }
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info(
        "Built %d GSM8K CoT examples in %s (dataset_sha=%s)",
        len(records), out_dir, dataset_sha[:12],
    )
    print(f"\nGSM8K CoT dataset ready: {out_dir}")
    print(f"  examples       {len(records)}")
    print(f"  max_new_tokens {MAX_NEW_TOKENS}")
    print(f"  dataset_sha    {dataset_sha}")
    return out_dir


def read_manifest(data_dir: str | Path) -> Dict:
    """Read the manifest written by :func:`build_gsm8k_cot`."""
    path = Path(data_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the dataset first:\n"
            f"    python -m data.gsm8k_loader --out {data_dir}"
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
    data_dir
        Directory produced by :func:`build_gsm8k_cot`.
    num_samples
        ``"max"`` for the full split, or an int cap applied BEFORE sharding so
        that a capped run is the same set of problems at every budget.
    shard, num_shards
        Contiguous sharding for multi-GPU splits. Contiguous (not strided) so a
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
            raise ValueError(
                f"shard must be in [0, {num_shards}), got {shard}"
            )
        total = len(records)
        per = (total + num_shards - 1) // num_shards
        start, end = shard * per, min((shard + 1) * per, total)
        records = records[start:end]
        log.info(
            "GSM8K shard %d/%d: examples [%d, %d) of %d",
            shard, num_shards, start, end, total,
        )

    log.info(
        "Loaded %d GSM8K examples from %s (dataset_sha=%s)",
        len(records), data_dir, manifest["dataset_sha"][:12],
    )
    return records


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
