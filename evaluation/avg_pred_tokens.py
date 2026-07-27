#!/usr/bin/env python
"""Average number of generated tokens per prediction in an evaluation .jsonl log.

Uses the same HF tokenizer the evaluation pipeline uses (AutoTokenizer), so the
counts match what the model actually generated.

Example:
    python avg_pred_tokens.py logs/nish_key3.jsonl --model ${MODELS_DIR}/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", type=Path, help="Path to the predictions .jsonl file")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model path or HF id whose tokenizer is used (default: %(default)s)",
    )
    parser.add_argument("--field", default="pred", help="JSON field holding the generation (default: %(default)s)")
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="Also break the statistics down by the 'task' field",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=8,
        help="Count predictions shorter than this many tokens (default: %(default)s)",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    lengths = []
    per_task = {}
    short_examples = []
    max_new_tokens = set()
    with args.jsonl.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping malformed line {line_no}")
                continue
            pred = row.get(args.field)
            if pred is None:
                continue
            # add_special_tokens=False: count only the generated tokens, no BOS/EOS
            n = len(tokenizer.encode(pred, add_special_tokens=False))
            lengths.append(n)
            per_task.setdefault(row.get("task", "unknown"), []).append(n)
            if n < args.short_threshold:
                short_examples.append((row.get("id", line_no), n, pred))
            if "max_new_tokens" in row:
                max_new_tokens.add(row["max_new_tokens"])

    if not lengths:
        raise SystemExit(f"no '{args.field}' entries found in {args.jsonl}")

    def report(name, values):
        short = sum(1 for n in values if n < args.short_threshold)
        print(
            f"{name:<24} n={len(values):<6} "
            f"avg={statistics.mean(values):7.2f}  "
            f"median={statistics.median(values):6.1f}  "
            f"std={statistics.pstdev(values):6.2f}  "
            f"min={min(values):<5} max={max(values):<5} "
            f"total={sum(values):<8} "
            f"<{args.short_threshold}tok={short} ({100 * short / len(values):.1f}%)"
        )

    print(f"file      : {args.jsonl}")
    print(f"tokenizer : {args.model}")
    if max_new_tokens:
        print(f"max_new_tokens in file: {sorted(max_new_tokens)}")
        cap = max(max_new_tokens)
        hit = sum(1 for n in lengths if n >= cap)
        print(f"predictions at the {cap}-token cap: {hit} ({100 * hit / len(lengths):.1f}%)")
    empty = sum(1 for n in lengths if n == 0)
    if empty:
        print(f"empty predictions: {empty}")
    if short_examples:
        print(
            f"predictions under {args.short_threshold} tokens: "
            f"{len(short_examples)} ({100 * len(short_examples) / len(lengths):.1f}%)"
        )
        for ex_id, n, pred in short_examples[:5]:
            print(f"    id={ex_id} n={n} pred={pred!r}")
        if len(short_examples) > 5:
            print(f"    ... {len(short_examples) - 5} more")
    print()
    report("ALL", lengths)
    if args.per_task and len(per_task) > 1:
        print()
        for task, values in sorted(per_task.items()):
            report(task, values)


if __name__ == "__main__":
    main()
