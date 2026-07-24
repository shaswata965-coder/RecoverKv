"""Build the GSM8K eval dataset with the truncation defect fixed.

Replaces DefensiveKV ``evaluation/gsm8k/create_huggingface_dataset.py``.

Three changes, all aimed at making the answer line always reachable:

1.  ``max_new_tokens`` 256 -> 512. At 256 a non-trivial share of generations are
    clipped mid-reasoning and never emit an answer, and the clip rate rises with
    KV budget (a healthier cache reasons longer), which bends the accuracy curve
    on its own.
2.  The prompt now demands GSM8K's native terminal marker ``#### <number>`` on
    its own final line. One unambiguous token sequence, no ``$``/``**``/prose
    decoration to parse around, and it doubles as a stop string.
3.  ``STOP_STRINGS`` is exported so the runner can halt the moment the answer is
    complete. This is what keeps the higher token cap from costing wall-clock:
    typical generations still stop in ~150 tokens, only the long ones use the
    headroom. It also removes run-on continuations at the source.

Note ``context`` is still the constant system prompt, matching DefensiveKV. Under
kvpress only ``context`` is compressed, so with this layout the budget knob acts
on ~60 tokens of boilerplate and the problem text is never compressed. Pass
``compress_questions=True`` in evaluate.py if you want the budget to act on the
problem itself.

Usage:
    python -m modules.evaluation.gsm8k_dataset --out ./gsm8k
"""

from __future__ import annotations

import argparse

SYSTEM_PROMPT = (
    "Solve the following math word problem. Show your step-by-step reasoning, "
    "then give the final numeric answer.\n\n"
    "End your response with the final answer on its own line in exactly this "
    "form:\n#### <number>\n\n"
    "Write only the bare number after ####: no currency symbols, no units, no "
    "commas, no formatting."
)

QUESTION_TEMPLATE = "Question:\n{q}\n\nReasoning:\n"

#: Halt generation once the answer line is complete, or if the model starts a new
#: problem. Pass to generate(stop_strings=..., tokenizer=...) — transformers>=4.39.
STOP_STRINGS = ["\n\nQuestion:", "\nQuestion:", "\n\nProblem:"]

MAX_NEW_TOKENS = 512


def extract_numeric_answer(answer_text: str) -> str:
    """Pull the reference number out of a GSM8K ``answer`` field."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip()


def build(out_dir: str = "./gsm8k", split: str = "test"):
    from datasets import Dataset, load_dataset

    df = load_dataset("openai/gsm8k", "main", split=split).to_pandas()

    df["context"] = SYSTEM_PROMPT
    df["question"] = df["question"].apply(lambda q: QUESTION_TEMPLATE.format(q=q))
    df["answer"] = df["answer"].apply(lambda a: [extract_numeric_answer(a)])
    df["answer_prefix"] = ""
    df["task"] = "gsm8k"
    df["max_new_tokens"] = MAX_NEW_TOKENS
    df["all_classes"] = [[] for _ in range(len(df))]

    df = df[
        ["context", "question", "answer_prefix", "answer",
         "task", "max_new_tokens", "all_classes"]
    ]
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.save_to_disk(out_dir)
    print(f"Saved {len(df)} GSM8K examples to {out_dir} "
          f"(max_new_tokens={MAX_NEW_TOKENS})")
    return ds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./gsm8k")
    p.add_argument("--split", default="test")
    build(p.parse_args().out, p.parse_args().split)
