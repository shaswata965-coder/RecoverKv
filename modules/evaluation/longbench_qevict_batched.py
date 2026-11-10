"""Batched QEvict on DeepSeek-V2(-Lite) for LongBench, under DefensiveKV's protocol.

The harness is the batched SnapKV baseline script (DefensiveKV's
``evaluation/`` folder, kvpress pipeline) with one thing swapped: the KV
compressor. Everything that decides what the model is asked and how it is
scored or batched is that script's, line for line:

* **Protocol.** DefensiveKV's ``context_prompt`` / ``question_prompt`` /
  ``max_new_tokens`` (read from ``evaluation/longbench/create_huggingface_dataset.py``
  with ``ast``, without running it), tokenized by DefensiveKV's own
  ``KVPressTextGenerationPipeline.preprocess`` (its source, extracted from
  ``kvpress/pipeline.py`` and executed unchanged -- importing ``kvpress`` needs
  transformers 4.57, the cache here needs 4.47.1). The context is prefilled and
  compressed; the question (question prompt + chat-template suffix) is appended
  afterwards; the answer is decoded greedily. No chat template and an empty BOS
  for ``trec``/``triviaqa``/``samsum`` (``evaluate.py``); ``samsum`` also stops at
  a newline. Optional token-id middle truncation (``max_prompt_tokens``).
  Scoring is DefensiveKV's ``evaluation/longbench/calculate_metrics.py``.
* **Batching.** Rows sorted by context length (longest first unless
  ``shortest_first``), grouped greedily into batches of at most ``batch_size``
  rows and ``max_batch_tokens`` padded prefill tokens. Contexts left-padded,
  per-row positions ``0 .. L_i - 1``; the question block left-padded at
  positions ``L_i ..``; decode tokens at ``L_i + Q_i + t``, every row in
  lockstep until all have emitted EOS or ``max_new_tokens``; each answer cut at
  its first EOS.

What QEvict changes:

* **The cache.** Each row gets its own RecoverKV ``WindowedCache`` (MLA latent),
  sized by the paper's definition: ``cache_budget x (L_i + Q_i + max_new)``.
  The padded batch runs every non-attention layer as one tensor; attention is
  split by row (``models/deepseek_v2/row_batch.py``), so a row's cache only ever
  receives its own tokens -- the keys a padding mask would leave visible.
* **The question is fed one column at a time.** QEvict's cache takes one prompt
  and then single-token steps (its read gate and fused decode kernel are
  single-query), so the left-padded question block is fed column by column,
  each row skipping its own pad columns. Every real question token attends to
  what it attends to in a one-shot question forward (the compressed context
  and the earlier question tokens), through QEvict's gated read.
* **When compression happens.** At ``first_eviction_step = 0`` the cache
  compresses before the first question token is attended: after the context,
  before the question, as SnapKV does. After that QEvict keeps re-routing every
  R steps over the question and the answer, as it does over generated tokens in
  the paper; the question is appended at full precision and stays there while
  it is inside the 128-token recent window.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import subprocess
import textwrap
import time
import types
from pathlib import Path
from typing import Callable, List, Sequence

import torch

NO_CHAT_TEMPLATE = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]  # evaluate.py
REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# DefensiveKV's own protocol code, read from a checkout
# ---------------------------------------------------------------------------


def defensivekv_dicts(dkv_dir: Path):
    """context_prompt / question_prompt / max_new_tokens, read without running the script."""
    path = Path(dkv_dir) / "evaluation" / "longbench" / "create_huggingface_dataset.py"
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("context_prompt", "question_prompt", "max_new_tokens"):
                out[name] = ast.literal_eval(node.value)
    return out["context_prompt"], out["question_prompt"], out["max_new_tokens"]


def defensivekv_preprocess(dkv_dir: Path) -> Callable:
    """``KVPressTextGenerationPipeline.preprocess`` from DefensiveKV's kvpress, unchanged.

    Called as the baseline script calls it: unbound, with a namespace whose only
    attribute is ``tokenizer``.
    """
    src = (Path(dkv_dir) / "kvpress" / "pipeline.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "KVPressTextGenerationPipeline":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "preprocess":
                    ns = {"logger": logging.getLogger("kvpress.pipeline")}
                    exec(compile(textwrap.dedent(ast.get_source_segment(src, fn)),
                                 str(Path(dkv_dir) / "kvpress" / "pipeline.py"), "exec"), ns)
                    return ns["preprocess"]
    raise RuntimeError(f"no KVPressTextGenerationPipeline.preprocess in {dkv_dir}/kvpress/pipeline.py")


def defensivekv_calculate_metrics(dkv_dir: Path) -> Callable:
    path = Path(dkv_dir) / "evaluation" / "longbench" / "calculate_metrics.py"
    spec = importlib.util.spec_from_file_location("defensivekv_longbench_calculate_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calculate_metrics


# ---------------------------------------------------------------------------
# Batching: the baseline script's, unchanged
# ---------------------------------------------------------------------------


def left_pad(seqs, pad_id, device):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s)
        mask[i, n - len(s):] = 1
    return ids.to(device), mask.to(device)


def make_batches(ctx_lens: Sequence[int], batch_size: int, max_batch_tokens: int,
                 shortest_first: bool) -> List[List[int]]:
    order = sorted(range(len(ctx_lens)), key=lambda i: ctx_lens[i], reverse=not shortest_first)
    batches, cur = [], []
    for i in order:
        width = max([ctx_lens[j] for j in cur] + [ctx_lens[i]])
        if cur and (len(cur) == batch_size or (len(cur) + 1) * width > max_batch_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    return batches


def decode_answers(tok, gen: List[List[int]], eos_ids: torch.Tensor) -> List[str]:
    answers = []
    eos = eos_ids.tolist()
    for row in gen:
        cut = next((j + 1 for j, x in enumerate(row) if x in eos), len(row))
        answers.append(tok.decode(row[:cut], skip_special_tokens=True))
    return answers


@torch.no_grad()
def run_batch(model, pad_id: int, make_row_cache: Callable[[int, int], object],
              ctx_ids: List[List[int]], q_ids: List[List[int]], max_new_tokens: int,
              eos_ids: torch.Tensor, trace: list = None):
    """The baseline's ``run_batch``, over per-row caches.

    ``make_row_cache(L_i, Q_i)`` builds row ``i``'s cache. Returns the generated
    token ids per row (lockstep, uncut) and the row caches. ``trace``, if given,
    collects each step's ``[B, vocab]`` fp32 logits (tests).
    """
    from models.deepseek_v2.row_batch import RowBatchCache

    dev = model.device
    bsz = len(ctx_ids)
    L = torch.tensor([len(c) for c in ctx_ids], device=dev)
    Q = torch.tensor([len(q) for q in q_ids], device=dev)
    ids, mask = left_pad(ctx_ids, pad_id, dev)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    rows = [make_row_cache(len(c), len(q)) for c, q in zip(ctx_ids, q_ids)]
    cache = RowBatchCache(rows)

    def forward(input_ids, position_ids, starts):
        cache.begin_call(starts)
        try:
            return model.model(input_ids=input_ids, position_ids=position_ids,
                               past_key_values=cache, use_cache=True).last_hidden_state
        finally:
            cache.end_call()

    # Context prefill: each row from its first real column.
    forward(ids, pos, [ids.shape[1] - len(c) for c in ctx_ids])

    # Question block, left-padded at positions L_i + ..., one column per call.
    qi, qm = left_pad(q_ids, pad_id, dev)
    qpos = L[:, None] + (qm.cumsum(-1) - 1).clamp(min=0)
    qpad = [qi.shape[1] - len(q) for q in q_ids]
    if qi.shape[1] == 0:
        raise ValueError("every row needs at least one question token")
    h = None
    for j in range(qi.shape[1]):
        h = forward(qi[:, j:j + 1], qpos[:, j:j + 1], [0 if j >= qpad[i] else 1 for i in range(bsz)])
    logits = model.lm_head(h[:, -1])
    if trace is not None:
        trace.append(logits.float())
    nxt = logits.argmax(-1)
    gen = [nxt]
    done = torch.isin(nxt, eos_ids)
    for t in range(max_new_tokens - 1):
        if bool(done.all()):
            break
        p = (L + Q + t)[:, None]
        h = forward(nxt[:, None], p, [0] * bsz)
        logits = model.lm_head(h[:, -1])
        if trace is not None:
            trace.append(logits.float())
        nxt = logits.argmax(-1)
        gen.append(nxt)
        done |= torch.isin(nxt, eos_ids)
    return torch.stack(gen, dim=1).tolist(), rows


# ---------------------------------------------------------------------------
# The method's cache
# ---------------------------------------------------------------------------


def qevict_cache_config(cache_cfg, cache_budget: float):
    """``WindowedCacheConfig`` exactly as ``LongBenchRunner._setup_windowed_cache`` builds it."""
    from modules.windowed_cache.config import WindowedCacheConfig
    from utils.cache_factory import (quant_budget_mode_kwargs, quant_gate_ratio_kwargs,
                                     quant_tier_policy_kwargs)
    from utils.config import FIRST_EVICTION_STEP_DEFAULT

    return WindowedCacheConfig(
        window_size=cache_cfg.window_size,
        num_sink_tokens=cache_cfg.num_sink_tokens,
        local_window_size=cache_cfg.local_window_size,
        cache_budget=cache_budget,
        rerotate_on_evict=getattr(cache_cfg, "rerotate_on_evict", False),
        quant_ratio=getattr(cache_cfg, "quant_ratio", 0.0),
        **quant_budget_mode_kwargs(WindowedCacheConfig,
                                   getattr(cache_cfg, "quant_budget_mode", "bytes")),
        quant_memoize_read=getattr(cache_cfg, "quant_memoize_read", None),
        **quant_gate_ratio_kwargs(WindowedCacheConfig,
                                  getattr(cache_cfg, "quant_gate_ratio", 0.25)),
        quant_card_bits=getattr(cache_cfg, "quant_card_bits", None),
        **quant_tier_policy_kwargs(cache_cfg),
        first_eviction_step=getattr(cache_cfg, "first_eviction_step", FIRST_EVICTION_STEP_DEFAULT),
    )


def _git_sha(path: Path):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main(
    model: str = "deepseek-ai/DeepSeek-V2-Lite-Chat",
    task: str = "2wikimqa",
    data_dir: str = "data/longbench",
    defensivekv_dir: str = "DefensiveKV",
    config: str = "configs/longbench_deepseek_v2_lite_flash.yaml",
    cache_budget: float = 0.10,
    batch_size: int = 8,
    out_dir: str = "outputs/longbench_deepseek",
    limit: int = 0,
    max_context_length: int = 0,
    shortest_first: bool = False,
    max_batch_tokens: int = 140000,
    max_prompt_tokens: int = 0,
    revision: str = None,
):
    """Arguments as the baseline script's, plus where DefensiveKV and the method's YAML live."""
    from models.deepseek_v2 import attention as mla_attention
    from models.deepseek_v2.loader import make_cache
    from modules.windowed_cache import flash_decode
    from utils.cache_factory import (assert_transformers_version_supported,
                                     quant_card_bits_record, quant_tier_policy_record)
    from utils.config import load_config
    from utils.model_loading import load_causal_lm, load_tokenizer
    import pandas as pd
    import transformers

    assert_transformers_version_supported()
    dkv = Path(defensivekv_dir)
    out = Path(out_dir)
    stem = f"{task}_qevict_b{cache_budget}_bs{batch_size}" + (f"_trunc{max_prompt_tokens}" if max_prompt_tokens else "")
    if (out / f"{stem}.meta.json").exists():
        print(f"{stem}: already done, skipping")
        return
    model_override = {"name": model}
    if revision is not None:
        model_override["revision"] = revision
    cfg = load_config(config, overrides={"cache": {"cache_budget": cache_budget}, "model": model_override})
    revision = cfg.model.revision
    if cfg.cache.backend != "windowed" or cfg.cache.backend_package != "flash_attn":
        raise ValueError(f"{config}: QEvict runs on the windowed flash_attn backend")
    cache_config = qevict_cache_config(cfg.cache, cache_budget)
    context_prompt, question_prompt, max_new = defensivekv_dicts(dkv)
    preprocess = defensivekv_preprocess(dkv)
    calculate_metrics = defensivekv_calculate_metrics(dkv)
    rows = [json.loads(l) for l in open(Path(data_dir) / f"{task}.jsonl")]
    if limit:
        rows = rows[:limit]

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    cuda = torch.cuda.is_available()
    tok = load_tokenizer(model, revision=revision)
    m = load_causal_lm(model, revision=revision, torch_dtype=dtypes[cfg.model.dtype],
                       attn_implementation=cfg.model.attn_implementation,
                       device_map="cuda:0" if cuda else None)
    max_ctx = max_context_length or m.config.max_position_embeddings
    if task in NO_CHAT_TEMPLATE:   # evaluate.py: no chat template, empty bos for these tasks
        tok.chat_template, tok.bos_token = None, ""
    eos = m.generation_config.eos_token_id
    eos_ids = torch.tensor(eos if isinstance(eos, list) else [eos], device=m.device)
    if task == "samsum":
        eos_ids = torch.cat([eos_ids, torch.tensor([tok.encode("\n", add_special_tokens=False)[-1]], device=m.device)])

    pre = types.SimpleNamespace(tokenizer=tok)
    for r in rows:
        t = preprocess(
            pre, context_prompt[task].format(context=r["context"]),
            questions=[question_prompt[task].format(input=r["input"])], answer_prefix="", max_context_length=max_ctx,
        )
        r["_ctx"] = t["context_ids"][0].tolist()
        r["_q"] = t["questions_ids"][0][0].tolist()
        r["_full_ctx"] = len(r["_ctx"])
        if max_prompt_tokens and len(r["_ctx"]) + len(r["_q"]) + max_new[task] > max_prompt_tokens:
            room = max_prompt_tokens - len(r["_q"]) - max_new[task]
            r["_ctx"] = r["_ctx"][: room // 2] + r["_ctx"][len(r["_ctx"]) - (room - room // 2):]

    def make_row_cache(n_ctx: int, n_q: int):
        return make_cache(m, cache_config, prefill_len=n_ctx, max_tokens=n_q + max_new[task])

    batches = make_batches([len(r["_ctx"]) for r in rows], batch_size, max_batch_tokens, shortest_first)
    flash_decode.reset_stats()
    mla_attention.reset_path_stats()
    t0 = time.time()
    for bi, idx in enumerate(batches):
        batch = [rows[i] for i in idx]
        tb = time.time()
        gen, caches = run_batch(m, tok.pad_token_id or 0, make_row_cache, [r["_ctx"] for r in batch],
                                [r["_q"] for r in batch], max_new[task], eos_ids)
        answers = decode_answers(tok, gen, eos_ids)
        for r, a, c in zip(batch, answers, caches):
            res = c.resolved
            r["predicted_answer"] = a
            r["kept_tokens"] = int(res.retained_tokens)
            r["budget_bytes"] = int(res.total_budget_bytes)
            r["int2_windows"] = int(res.N_q)
        del caches
        peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
        print(f"batch {bi + 1}/{len(batches)}: {len(batch)} rows, "
              f"ctx {len(batch[-1]['_ctx'])}-{len(batch[0]['_ctx'])} tok, {time.time() - tb:.1f}s, "
              f"peak {peak:.1f} GB", flush=True)
    elapsed = time.time() - t0

    expect = flash_decode.expect_gated(cfg.cache.backend_package, cache_config.quant_ratio, cuda)
    read_gate = flash_decode.gate_report(expect)
    print(f"read gate: {read_gate['verdict']} (fired {read_gate.get('fired')}, "
          f"gated {read_gate.get('gated')}, read fraction {read_gate.get('read_fraction')})", flush=True)

    df = pd.DataFrame([dict(task=task, _id=r["_id"], predicted_answer=r["predicted_answer"], answer=r["answers"],
                            all_classes=r["all_classes"], context_tokens=len(r["_ctx"]), question_tokens=len(r["_q"]),
                            kept_tokens=r["kept_tokens"], budget_bytes=r["budget_bytes"],
                            int2_windows=r["int2_windows"], untruncated_context_tokens=r["_full_ctx"])
                       for r in rows])
    score = calculate_metrics(df)[task]
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / f"{stem}.jsonl", orient="records", lines=True, force_ascii=False)
    c = cfg.cache
    meta = dict(task=task, score=score, n=len(df),
                method="qevict (RecoverKV two-tier cache, DeepSeek-V2 MLA latent, per-row caches in a left-padded batch)",
                read_gate=read_gate, attention_paths=mla_attention.path_stats(),
                cache_budget=cache_budget, budget_definition="cache_budget * (context + question + max_new_tokens) per row",
                window_size=c.window_size, num_sink_tokens=c.num_sink_tokens, local_window_size=c.local_window_size,
                quant_ratio=cache_config.quant_ratio, quant_budget_mode=getattr(c, "quant_budget_mode", "bytes"),
                quant_gate_ratio=getattr(c, "quant_gate_ratio", None),
                first_eviction_step=cache_config.first_eviction_step,
                quant_card_bits=quant_card_bits_record(c), **quant_tier_policy_record(c),
                batch_size=batch_size, max_batch_tokens=max_batch_tokens, n_batches=len(batches),
                shortest_first=shortest_first, attn=cfg.model.attn_implementation, dtype=cfg.model.dtype,
                max_prompt_tokens=max_prompt_tokens,
                n_truncated=sum(len(r["_ctx"]) < r["_full_ctx"] for r in rows), max_context_length=max_ctx,
                model=model, model_revision=revision, config=config,
                recoverkv_commit=_git_sha(REPO), defensivekv_commit=_git_sha(dkv),
                transformers=transformers.__version__, torch=torch.__version__, seconds=round(elapsed, 1),
                gpu=torch.cuda.get_device_name(0) if cuda else "cpu",
                peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1) if cuda else None)
    (out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))


def _cli():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    p.add_argument("--task", default="2wikimqa")
    p.add_argument("--data_dir", default="data/longbench")
    p.add_argument("--defensivekv_dir", default="DefensiveKV")
    p.add_argument("--config", default="configs/longbench_deepseek_v2_lite_flash.yaml")
    p.add_argument("--cache_budget", type=float, default=0.10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_dir", default="outputs/longbench_deepseek")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_context_length", type=int, default=0)
    p.add_argument("--shortest_first", action="store_true")
    p.add_argument("--max_batch_tokens", type=int, default=140000)
    p.add_argument("--max_prompt_tokens", type=int, default=0)
    p.add_argument("--revision", default=None)
    main(**vars(p.parse_args()))


if __name__ == "__main__":
    _cli()
