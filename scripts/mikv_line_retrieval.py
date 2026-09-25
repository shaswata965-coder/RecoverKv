"""Line Retrieval — the MiKV paper's detail-preservation benchmark (§2.3, App. D.3).

Each sample is LongChat's format (Li et al., 2023): an instruction header,
``--n-lines`` lines of ``line <index>: REGISTER_CONTENT is <number>``, and a
request for one line's number. It is the task where eviction visibly fails —
the paper's Table 1 has H2O at 4.0% with a 20% cache — and where a retained
low-precision copy of the evicted tokens recovers it.

Every arm runs through the same MiKV cache and the same H2O selection; the
arms differ only in what happens to the tokens H2O would evict:

    full         DynamicCache (no MiKV code at all)
    h2o          dropped                      (lo_bits=0)
    rtn2         everything INT2, no importance cache   (importance_ratio=0)
    mikv2-nobal  retained at INT2, no balancer
    mikv2        retained at INT2 + balancer   <- the paper's headline
    mikv3, mikv4 retained at INT3 / INT4 + balancer
    mikv2-hi8    mikv2 with an INT8 importance cache (Table 3)

Usage (GPU, the paper's setting is an instruction-tuned model)::

    python scripts/mikv_line_retrieval.py \\
        --model meta-llama/Llama-3.1-8B-Instruct --samples 200 --n-lines 20

Runs on CPU with a small Llama-architecture model, e.g.
``--model HuggingFaceTB/SmolLM2-360M-Instruct --device cpu --dtype float32``.
Writes one JSON line per arm to ``--out`` (default: stdout only).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.mikv import MiKVCache, MiKVConfig, install_mikv_hooks  # noqa: E402
from modules.mikv.config import paper_cache_size  # noqa: E402

_ADJ = ("torpid", "moaning", "vivid", "brave", "quiet", "sturdy", "hollow", "eager",
        "gentle", "rapid", "silent", "bitter", "clever", "dusty", "frozen", "golden",
        "humble", "jolly", "lucky", "mighty", "nimble", "proud", "rusty", "sleepy")
_NOUN = ("kid", "conversation", "harbor", "lantern", "meadow", "pencil", "river",
         "castle", "engine", "garden", "island", "jacket", "kettle", "ladder",
         "mirror", "needle", "orchard", "planet", "quartz", "saddle", "tunnel", "violin")

HEADER = ("Below is a record of lines I want you to remember. Each line begins with "
          "'line <line index>' and contains a '<REGISTER_CONTENT>' at the end of the "
          "line as a numerical value. For each line index, memorize its corresponding "
          "<REGISTER_CONTENT>. At the end of the record, I will ask you to retrieve the "
          "corresponding <REGISTER_CONTENT> of a certain line index. Now the record "
          "start:\n\n")

ARMS: Dict[str, Optional[dict]] = {
    "full": None,
    "h2o": dict(lo_bits=0),
    "rtn2": dict(importance_ratio=0.0, lo_bits=2),
    "mikv2-nobal": dict(lo_bits=2, channel_balancer=False),
    "mikv2": dict(lo_bits=2),
    "mikv3": dict(lo_bits=3),
    "mikv4": dict(lo_bits=4),
    "mikv2-hi8": dict(lo_bits=2, hi_bits=8),
}


def make_sample(rng: random.Random, n_lines: int) -> Dict[str, str]:
    names = set()
    while len(names) < n_lines:
        names.add(f"{rng.choice(_ADJ)}-{rng.choice(_NOUN)}")
    names = list(names)
    values = [rng.randint(1, 50000) for _ in names]
    lines = "".join(f"line {n}: REGISTER_CONTENT is <{v}>\n" for n, v in zip(names, values))
    i = rng.randrange(n_lines)
    question = (f"\nNow the record is over. Tell me what is the <REGISTER_CONTENT> in "
                f"line {names[i]}? I need the number.")
    return {"prompt": HEADER + lines + question, "answer": str(values[i])}


def encode(tokenizer, prompt: str, chat: bool) -> torch.Tensor:
    if chat and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return tokenizer(prompt, return_tensors="pt").input_ids


def run_arm(model, tokenizer, samples: List[dict], arm: str, overrides: Optional[dict],
            importance_ratio: float, max_new_tokens: int, chat: bool) -> dict:
    correct = 0
    fractions: List[float] = []
    t0 = time.time()
    for s in samples:
        ids = encode(tokenizer, s["prompt"], chat).to(model.device)
        cache, hooks = None, None
        if overrides is not None:
            kw = {"importance_ratio": importance_ratio, **overrides}
            cache = MiKVCache(MiKVConfig(**kw), model.config.num_hidden_layers,
                              model.dtype, max_new_tokens)
            hooks = install_mikv_hooks(model, cache)
        try:
            with torch.no_grad():
                out = model.generate(
                    ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                    do_sample=False, num_beams=1,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    **({"past_key_values": cache} if cache is not None else {}))
        finally:
            if hooks is not None:
                hooks.remove()
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        correct += int(s["answer"] in text)
        if cache is not None:
            fractions.append(cache.memory_report()["kv_fraction"])
    row = {"arm": arm, "accuracy": correct / len(samples), "n": len(samples),
           "kv_fraction_measured": (sum(fractions) / len(fractions)) if fractions else 1.0,
           "seconds": round(time.time() - t0, 1)}
    if overrides is not None:
        cfg = MiKVConfig(**{"importance_ratio": importance_ratio, **overrides})
        head_dim = getattr(model.config, "head_dim", None) or (
            model.config.hidden_size // model.config.num_attention_heads)
        g = cfg.resolved_group_size(head_dim)
        row["kv_fraction_paper"] = paper_cache_size(
            cfg.importance_ratio, cfg.hi_bits, cfg.lo_bits, g, head_dim)
        row["config"] = cfg.to_dict()
    return row


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--n-lines", type=int, default=20)
    ap.add_argument("--importance-ratio", type=float, default=0.2)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-new-tokens", type=int, default=40,
                    help="instruct models restate the line before the number")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    ap.add_argument("--attn", default="sdpa", choices=("sdpa", "eager", "flash_attention_2"))
    ap.add_argument("--no-chat", action="store_true", help="skip the chat template")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="append JSON lines here")
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error(f"unknown arms {unknown}; choose from {list(ARMS)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn).to(args.device).eval()
    rng = random.Random(args.seed)
    samples = [make_sample(rng, args.n_lines) for _ in range(args.samples)]
    n_tok = encode(tokenizer, samples[0]["prompt"], not args.no_chat).shape[1]
    print(f"# {args.model}  {args.samples} samples x {args.n_lines} lines "
          f"(~{n_tok} prompt tokens)  importance_ratio={args.importance_ratio}",
          flush=True)
    print(f"{'arm':<12} {'acc':>7} {'kv (meas.)':>11} {'kv (paper)':>11} {'sec':>7}")
    for arm in arms:
        row = run_arm(model, tokenizer, samples, arm, ARMS[arm], args.importance_ratio,
                      args.max_new_tokens, not args.no_chat)
        row.update(model=args.model, n_lines=args.n_lines, seed=args.seed,
                   prompt_tokens=n_tok)
        paper = row.get("kv_fraction_paper")
        print(f"{arm:<12} {row['accuracy']:>7.1%} {row['kv_fraction_measured']:>11.1%} "
              f"{'' if paper is None else f'{paper:.1%}':>11} {row['seconds']:>7}",
              flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
