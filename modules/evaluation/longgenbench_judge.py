"""LongGenBench judge: one yes/no call per Specific Task Instruction check.

Runs in its own process, on a plain full-cache model, AFTER generation, so the
judge can never be touched by the cache under test (no hooks, no patched
attention). Both arms are judged by the same model with the same settings, and
the settings are written beside every verdict file.

The reference (``Evalution/eval.py``) calls vLLM on the raw prompt (one BOS, no
chat template) with ``max_tokens=50`` and samples at ``temperature=0.95,
top_p=0.95``. The paper names Llama-3.1-8B-Instruct as the judge and does not
state its decoding. Defaults here: that judge, greedy, one check per forward
pass (B = 1), so a verdict is a deterministic function of the block and the
question, not of a sampling draw or of what else shared its batch.
``--decoding repo_sampling`` reproduces eval.py's sampling (seeded per check).

Usage::

    python -m modules.evaluation.longgenbench_judge --runs DIR [DIR ...]
    python -m modules.evaluation.longgenbench_judge --selftest

Writes, next to each ``<type>.jsonl`` in DIR, ``<type>.judge.jsonl`` (one row per
judged check, with the raw response) and ``<type>.judge.meta.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data.longgenbench_loader import type_slug
from modules.evaluation.longgenbench_scoring import (
    JUDGE_MAX_NEW_TOKENS,
    JUDGE_MODEL,
    document_checks,
    judge_prompt,
    read_jsonl,
    strict_verdict,
    verdict,
)

JUDGE_DECODINGS = ("greedy", "repo_sampling")
REPO_JUDGE_SAMPLING = {"temperature": 0.95, "top_p": 0.95, "top_k": 0, "seed": 42}

#: Four trivially decidable checks with known answers, in the reference prompt
#: format. A judge that cannot get these right cannot score anything.
SELFTEST = [
    ("The skyscraper's 12th floor houses a public library with reading rooms, study carrels "
     "and a children's corner. Large windows face the river.", "public library", "yes"),
    ("The skyscraper's 12th floor houses a public library with reading rooms, study carrels "
     "and a children's corner. Large windows face the river.", "swimming pool", "no"),
    ("This week's menu: appetizers of bruschetta and calamari, a main course of grilled "
     "salmon with lemon butter, roasted asparagus, and tiramisu for dessert.", "grilled salmon", "yes"),
    ("This week's menu: appetizers of bruschetta and calamari, a main course of grilled "
     "salmon with lemon butter, roasted asparagus, and tiramisu for dessert.", "venison stew", "no"),
]


def prompt_template_sha() -> str:
    """Identity of the judge prompt (few-shot examples + wrapper)."""
    return hashlib.sha256(judge_prompt("<BLOCK>", "<DESC>").encode("utf-8")).hexdigest()[:16]


class Judge:
    def __init__(self, model_name: str = JUDGE_MODEL, dtype: str = "float16",
                 decoding: str = "greedy", attn_implementation: str = "flash_attention_2"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if decoding not in JUDGE_DECODINGS:
            raise ValueError(f"judge decoding must be one of {JUDGE_DECODINGS}, got {decoding!r}")
        self.torch = torch
        self.model_name, self.dtype, self.decoding = model_name, dtype, decoding
        self.attn_implementation = attn_implementation
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype],
            attn_implementation=attn_implementation, device_map="auto")
        self.model.eval()

    def ask(self, prompt: str, seed: int = 0) -> str:
        from utils.prompting import encode_prompt

        torch = self.torch
        enc = encode_prompt(self.tok, prompt, return_tensors="pt")   # one BOS, as vLLM adds
        ids = enc.input_ids.to(self.model.device)
        kw: Dict[str, Any] = dict(attention_mask=enc.attention_mask.to(self.model.device),
                                  max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                                  pad_token_id=self.tok.pad_token_id)
        if self.decoding == "greedy":
            kw.update(do_sample=False, num_beams=1, temperature=1.0, top_p=1.0)
        else:
            torch.manual_seed(REPO_JUDGE_SAMPLING["seed"] + seed)
            kw.update(do_sample=True, temperature=REPO_JUDGE_SAMPLING["temperature"],
                      top_p=REPO_JUDGE_SAMPLING["top_p"], top_k=REPO_JUDGE_SAMPLING["top_k"])
        with torch.no_grad():
            out = self.model.generate(ids, **kw)
        return self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    def record(self) -> Dict[str, Any]:
        from utils.env_capture import capture_environment
        from utils.hashing import sha256_tokenizer

        return {
            "judge_model": self.model_name,
            "judge_dtype": self.dtype,
            "judge_attn_implementation": self.attn_implementation,
            "judge_decoding": self.decoding,
            "judge_sampling": REPO_JUDGE_SAMPLING if self.decoding == "repo_sampling" else None,
            "judge_max_new_tokens": JUDGE_MAX_NEW_TOKENS,
            "judge_batch_size": 1,
            "judge_prompt_sha": prompt_template_sha(),
            "judge_tokenizer_sha": sha256_tokenizer(self.tok),
            "verdict_rule": "'yes' if 'yes' in response.strip().lower() else 'no'",
            **capture_environment(),
        }


def selftest(judge: Judge) -> Dict[str, Any]:
    rows = []
    for i, (ctx, desc, want) in enumerate(SELFTEST):
        r = judge.ask(judge_prompt(ctx, desc), seed=10_000 + i)
        rows.append({"desc": desc, "want": want, "response": r,
                     "verdict": verdict(r), "strict": strict_verdict(r)})
    return {"rows": rows,
            "strict_correct": sum(x["strict"] == x["want"] for x in rows),
            "official_correct": sum(x["verdict"] == x["want"] for x in rows),
            "n": len(rows)}


def judge_run_dir(judge: Judge, run_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Judge every ``<type>.jsonl`` in *run_dir*; returns ``[(slug, summary)]``."""
    done = []
    for meta_p in sorted(run_dir.glob("*.meta.json")):
        if meta_p.name.endswith(".judge.meta.json"):
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("benchmark") != "LongGenBench":
            continue
        slug = type_slug(meta["type"])
        recs = read_jsonl(run_dir / f"{slug}.jsonl")
        t0 = time.time()
        rows = []
        for rec in recs:
            if rec.get("output") is None:
                continue                  # a failed generation has nothing to judge
            for k, c in enumerate(document_checks(rec)):
                if c["block"] is None:
                    continue              # eval.py judges only checks whose entry exists
                resp = judge.ask(judge_prompt(c["block"], c["desc"]), seed=rec["id"] * 1000 + k)
                rows.append({"id": rec["id"], "category": c["category"],
                             "identifier": c["identifier"], "desc": c["desc"],
                             "response": resp, "verdict": verdict(resp),
                             "strict": strict_verdict(resp)})
        with open(run_dir / f"{slug}.judge.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_yes = sum(r["verdict"] == "yes" for r in rows)
        jm = {**judge.record(),
              "generation_meta_sha": hashlib.sha256(meta_p.read_bytes()).hexdigest()[:16],
              "generations_sha": hashlib.sha256((run_dir / f"{slug}.jsonl").read_bytes()).hexdigest()[:16],
              "ids": [r["id"] for r in recs],
              "n_judged": len(rows), "n_yes": n_yes,
              "n_strict_disagree": sum(r["strict"] != r["verdict"] for r in rows),
              "wall_s": round(time.time() - t0, 1)}
        (run_dir / f"{slug}.judge.meta.json").write_text(json.dumps(jm, indent=2, default=str),
                                                        encoding="utf-8")
        done.append((slug, {"n_judged": len(rows), "n_yes": n_yes}))
        print(f"judged {run_dir.name}/{slug}: {len(rows)} checks, {n_yes} yes, "
              f"{jm['n_strict_disagree']} where the first word disagrees, {jm['wall_s']} s", flush=True)
    return done


def _cli_main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    p.add_argument("--decoding", default="greedy", choices=JUDGE_DECODINGS)
    p.add_argument("--selftest", action="store_true",
                   help="run the four known-answer checks first; exit 1 unless the judge's "
                        "first word is right on all four")
    a = p.parse_args()
    judge = Judge(a.model, a.dtype, a.decoding)
    if a.selftest:
        st = selftest(judge)
        for r in st["rows"]:
            print(f"selftest  want {r['want']:3s}  strict {str(r['strict']):4s}  official "
                  f"{r['verdict']:3s}  | {r['response'][:80]!r}")
        print(f"JUDGE SELFTEST strict {st['strict_correct']}/{st['n']}, "
              f"official rule {st['official_correct']}/{st['n']}", flush=True)
        if st["strict_correct"] != st["n"]:
            raise SystemExit(1)
    for d in a.runs:
        judge_run_dir(judge, Path(d))


if __name__ == "__main__":
    _cli_main()
