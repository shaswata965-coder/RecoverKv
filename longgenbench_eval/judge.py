"""The LongGenBench judge, as the paper ran it: Llama-3.1-8B-Instruct under vLLM, batched.

    python -m longgenbench_eval.judge --setting 16K --data data/longgenbench \\
        --generations runs/ours.jsonl runs/fullkv.jsonl

For each generations file ``X.jsonl`` this writes ``X.verdicts.jsonl`` (one row
per judged instruction, with the judge's raw response) and
``X.verdicts.meta.json`` (judge, prompt, sampling, versions, hashes). All files
are judged in ONE vLLM engine and ONE batched ``generate`` call, so a few
thousand instructions take minutes, not hours.

What it reproduces (``reference.py``, eval.py @ 951769c): the zero-shot prompt
``"Context: " + block + "### Instruction: Does this context include the {X}?
Please answer with 'yes' or 'no' only."``, raw (one BOS, no chat template),
``SamplingParams(temperature=0.95, top_p=0.95, max_tokens=50, seed=42)``, and the
verdict ``'yes' in response.strip().lower()``. vLLM seeds each request on its own,
so a verdict depends on the block and the question, not on what else is in the
batch. ``--greedy`` and ``--prompt 2025`` (today's few-shot eval.py prompt) exist
for sensitivity checks; neither is the paper's.

A known-answer selftest (four trivially decidable questions) runs first in the
same engine; fewer than three right stops the run.

Needs ``vllm`` (``requirements-judge.txt``: 0.8.5.post1, run on its V0 engine; the
paper used 0.5.4). Everything else is standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import __version__
from .data import dataset_sha, load_all
from .metrics import document_checks, load_generations
from .reference import (JUDGE_CODE, JUDGE_MODEL, JUDGE_PROMPTS, JUDGE_SAMPLING,
                        verdict)

#: Four trivially decidable checks with known answers, in the paper's prompt format.
SELFTEST = [
    (" Floor 12:\nThe twelfth floor houses a public library with reading rooms, study carrels and a "
     "children's corner. Large windows face the river.\n", "public library", "yes"),
    (" Floor 12:\nThe twelfth floor houses a public library with reading rooms, study carrels and a "
     "children's corner. Large windows face the river.\n", "swimming pool", "no"),
    (" Menu Week 3 (January 15th - January 21st):\nAppetizers of bruschetta and calamari, a main course "
     "of grilled salmon with lemon butter, roasted asparagus, and tiramisu for dessert.\n",
     "grilled salmon", "yes"),
    (" Menu Week 3 (January 15th - January 21st):\nAppetizers of bruschetta and calamari, a main course "
     "of grilled salmon with lemon butter, roasted asparagus, and tiramisu for dessert.\n",
     "venison stew", "no"),
]

Engine = Callable[[List[str]], List[str]]


def prompt_sha(prompt_version: str) -> str:
    return hashlib.sha256(JUDGE_PROMPTS[prompt_version]("<BLOCK>", "<DESC>").encode("utf-8")).hexdigest()[:16]


def sha16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def build_requests(docs: Sequence[Dict[str, Any]], generations: Dict[int, Optional[str]],
                   prompt_version: str = "paper") -> List[Dict[str, Any]]:
    """One request per instruction whose entry exists, in eval.py's order."""
    make = JUDGE_PROMPTS[prompt_version]
    by_id = {d["id"]: d for d in docs}
    reqs = []
    for i in sorted(generations):
        out = generations[i]
        if out is None:
            continue                       # a failed generation has nothing to judge
        for c in document_checks(by_id[i], out):
            if c["block"] is not None:      # eval.py judges only entries that exist
                reqs.append({"id": i, "category": c["category"], "identifier": c["identifier"],
                             "desc": c["desc"], "prompt": make(c["block"], c["desc"])})
    return reqs


def selftest(engine: Engine, prompt_version: str = "paper") -> Dict[str, Any]:
    make = JUDGE_PROMPTS[prompt_version]
    resp = engine([make(ctx, desc) for ctx, desc, _ in SELFTEST])
    rows = [{"desc": d, "want": w, "response": r, "verdict": verdict(r)}
            for (_, d, w), r in zip(SELFTEST, resp)]
    return {"rows": rows, "correct": sum(r["verdict"] == r["want"] for r in rows), "n": len(rows)}


def vllm_engine(model: str, prompts_for_length: Sequence[str], greedy: bool = False,
                gpu_memory_utilization: float = 0.85, tensor_parallel_size: int = 1,
                dtype: str = "auto") -> tuple[Engine, Dict[str, Any]]:
    """A vLLM engine sized to the longest prompt, and the record of how it was built."""
    # The paper's judge ran vLLM 0.5.4, whose engine is what 0.8.x still ships as
    # "V0"; per-request seeds have been supported there all along. An explicit
    # VLLM_USE_V1 in the environment wins.
    os.environ.setdefault("VLLM_USE_V1", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import vllm

    tok = AutoTokenizer.from_pretrained(model)
    longest = max((len(tok(p).input_ids) for p in prompts_for_length), default=1024)
    max_model_len = min(131072, ((longest + JUDGE_SAMPLING["max_tokens"] + 64) // 1024 + 1) * 1024)
    llm = LLM(model=model, dtype=dtype, tensor_parallel_size=tensor_parallel_size,
              gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
    if greedy:
        sp = SamplingParams(temperature=0.0, max_tokens=JUDGE_SAMPLING["max_tokens"])
    else:
        sp = SamplingParams(temperature=JUDGE_SAMPLING["temperature"], top_p=JUDGE_SAMPLING["top_p"],
                            max_tokens=JUDGE_SAMPLING["max_tokens"], seed=JUDGE_SAMPLING["seed"])

    def run(prompts: List[str]) -> List[str]:
        return [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=True)]

    info: Dict[str, Any] = {"vllm_version": vllm.__version__, "vllm_use_v1": os.environ.get("VLLM_USE_V1"),
                            "max_model_len": max_model_len,
                            "longest_prompt_tokens": longest, "dtype": dtype,
                            "tensor_parallel_size": tensor_parallel_size}
    try:
        import torch
        info.update(torch_version=torch.__version__,
                    gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:  # pragma: no cover - informational only
        pass
    return run, info


def judge_files(setting: str, data_dir: str | Path, gen_paths: Sequence[str | Path],
                engine_factory: Callable[[List[str]], tuple[Engine, Dict[str, Any]]],
                model: str = JUDGE_MODEL, prompt_version: str = "paper", greedy: bool = False,
                out_dir: Optional[str | Path] = None, force: bool = False,
                min_selftest: int = 3) -> List[Path]:
    """Judge every generations file; returns the verdict file paths."""
    docs = load_all(data_dir, setting)
    settings = {"judge_model": model, "judge_prompt": prompt_version, "judge_prompt_sha": prompt_sha(prompt_version),
                "judge_sampling": ({"temperature": 0.0, "max_tokens": JUDGE_SAMPLING["max_tokens"]}
                                   if greedy else dict(JUDGE_SAMPLING)),
                "judge_reference": JUDGE_CODE, "setting": setting, "dataset_sha": dataset_sha(data_dir, setting)}
    todo, jobs = [], []
    for g in map(Path, gen_paths):
        dst = (Path(out_dir) if out_dir else g.parent) / (g.name[:-len(g.suffix)] + ".verdicts.jsonl")
        meta = dst.with_suffix(".meta.json")
        if not force and dst.is_file() and meta.is_file():
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("generations_sha") == sha16(g) and all(m.get(k) == v for k, v in settings.items()):
                print(f"up to date: {dst}")
                todo.append(dst)
                continue
        gens = load_generations(g, docs)
        jobs.append((g, dst, gens, build_requests(docs, gens, prompt_version)))
    if not jobs:
        return todo
    prompts = [r["prompt"] for _, _, _, reqs in jobs for r in reqs]
    t0 = time.time()
    engine, info = engine_factory(prompts or [""])
    st = selftest(engine, prompt_version)
    for r in st["rows"]:
        print(f"selftest  want {r['want']:3s}  got {r['verdict']:3s}  | {r['response'][:70]!r}")
    print(f"JUDGE SELFTEST {st['correct']}/{st['n']}", flush=True)
    if st["correct"] < min_selftest:
        raise SystemExit(f"judge selftest {st['correct']}/{st['n']} < {min_selftest}: the judge is not "
                         "answering yes/no questions; nothing was scored")
    t1 = time.time()
    responses = engine(prompts) if prompts else []
    t_gen = time.time() - t1
    k = 0
    for g, dst, gens, reqs in jobs:
        rows = []
        for r in reqs:
            resp = responses[k]
            k += 1
            rows.append({"id": r["id"], "category": r["category"], "identifier": r["identifier"],
                         "desc": r["desc"], "response": resp, "verdict": verdict(resp)})
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta = {**settings, **info, "harness_version": __version__,
                "generations": str(g), "generations_sha": sha16(g),
                "ids": sorted(gens), "n_documents": len(gens), "n_judged": len(rows),
                "n_yes": sum(r["verdict"] == "yes" for r in rows),
                "selftest": st, "judge_wall_s_all_files": round(t_gen, 1),
                "engine_and_selftest_s": round(t1 - t0, 1),
                "judged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        dst.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"judged {g}: {len(rows)} instructions, {meta['n_yes']} yes -> {dst}", flush=True)
        todo.append(dst)
    print(f"{len(prompts)} judge calls in {t_gen:.0f} s ({len(prompts) / max(t_gen, 1e-9):.1f}/s)")
    return todo


def _cli_main() -> None:
    p = argparse.ArgumentParser(description="Judge LongGenBench generations as the paper did (vLLM).")
    p.add_argument("--setting", choices=("16K", "32K"), default="16K")
    p.add_argument("--data", default="data/longgenbench", help="directory holding the pinned dataset file")
    p.add_argument("--generations", nargs="+", required=True)
    p.add_argument("--out-dir", default=None, help="default: next to each generations file")
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--prompt", choices=tuple(JUDGE_PROMPTS), default="paper")
    p.add_argument("--greedy", action="store_true", help="not the paper's: temperature 0")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--force", action="store_true", help="judge again even if verdicts are up to date")
    a = p.parse_args()
    judge_files(a.setting, a.data, a.generations,
                lambda prompts: vllm_engine(a.model, prompts, a.greedy, a.gpu_memory_utilization,
                                            a.tensor_parallel_size),
                model=a.model, prompt_version=a.prompt, greedy=a.greedy, out_dir=a.out_dir, force=a.force)


if __name__ == "__main__":
    _cli_main()
