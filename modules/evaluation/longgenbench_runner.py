"""LongGenBench runner (mode ``longgenbench``): long-form generation, decode-phase cache.

Generates LongGenBench documents (Wu et al., ICLR'25) the way the runs behind the
paper's Table 2 did (``longgenbench_eval/reference.py``: ``inference.py`` @
``7b59ec6``), with this repository's cache in place of vLLM's:

* the dataset prompt verbatim, raw completion, one BOS (``encode_prompt`` adds
  it because the prompt carries none), no chat template;
* stop at ``*** finished`` (cut from the text, as vLLM does), at the model's
  EOS (the generation config's, as vLLM uses), or at ``max_new_tokens``
  (16,000 for ``16K``, 32,000 for ``32K``);
* ``output = prefix + generation``, written as ``{"id", "output", ...}`` lines
  that ``longgenbench_eval`` judges and scores like any baseline's;
* decoding ``reference`` (the code: temperature 0.95, top_p 0.95, no top-k, one
  seed set before every batch) or ``greedy`` (what the paper's text says);
  flash-attention-2.

**Batched, equal-length only.** Documents are grouped by exact prompt token
length and decoded ``batch_size`` at a time (``longgenbench_batching``): the
windowed cache keeps rows rectangular and takes no padding, and a padding mask
would send flash-attention down the varlen path the fused decode refuses. A
batch that runs out of memory is re-run in smaller batches down
``batch_fallback`` (16 -> 8 -> 6), and the smaller cap holds for the rest of the
run; every such event is in the sidecar.

The cache is the only thing that differs between arms: ``cache.backend:
dynamic`` is the full-cache baseline, ``windowed`` + ``flash_attn`` is this
repo's method, built exactly as the GSM8K and LongBench runners build it.

A run writes ``generations.jsonl`` (one record per document) and
``generations.meta.json``. The sidecar carries what makes the number
attributable: the read-gate verdict (counters reset per run), the operating
point, the resolved geometry, the dataset pin and hash, the decoding, the planned
and realised batches, and a compression record. The budget is ``cache_budget *
(prompt + max_new_tokens)``, the method's own definition; a document that stops
short of ``max_new_tokens`` holds more than ``cache_budget`` of its own sequence,
and ``kv_bytes_fraction_upper`` says how much.

Timing fields are wall clock inside a quality run: the first token carries the
process's compiles and tile searches, and one decode step serves every row of a
batch. They are not a performance measurement; ``mode: perf`` is.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from longgenbench_eval.data import dataset_sha, load_all, load_type
from longgenbench_eval.metrics import completion_rate, parse_blocks, split_blocks, word_count
from longgenbench_eval.reference import (
    CODE_REPO,
    FILES,
    GENERATION_CODE,
    GENERATION_SAMPLING,
    HF_DATASET,
    HF_REVISION,
    MAX_NEW_TOKENS,
    PROMPT_FORMAT,
    STOP_STRING,
    TYPES,
)
from modules.evaluation.longgenbench_batching import next_smaller, plan_batches, split_evenly
from utils.config import FIRST_EVICTION_STEP_DEFAULT, log_operating_point
from utils.env_capture import capture_environment
from utils.logger import get_logger
from utils.prompting import encode_prompt

log = get_logger(__name__)

#: "reference": the sampling of the runs behind the paper's Table 2;
#: "greedy": what the paper's text says (no run in the reference repo used it).
DECODINGS = ("reference", "greedy")


class _StepClock(StoppingCriteria):
    """Never stops; records when each decode step finishes.

    ``generate`` reads ``unfinished_sequences.max()`` on the host every step, so
    consecutive calls are a step apart in device time too. This separates what a
    document's wall clock mixes: the first token (prefill, plus every Triton and
    Inductor compile and tile search a fresh process does) and the steady decode
    rate that a long document is made of.
    """

    def __init__(self) -> None:
        self.t: List[float] = []

    def __call__(self, input_ids, scores, **kwargs):
        self.t.append(time.perf_counter())
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def summary(self, t0: float) -> Dict[str, Any]:
        if len(self.t) < 3:
            return {"first_token_s": round(self.t[0] - t0, 3) if self.t else None,
                    "decode_ms_per_token": None, "decode_ms_per_token_last_half_median": None}
        steps = [b - a for a, b in zip(self.t, self.t[1:])]
        half = sorted(steps[len(steps) // 2:])
        return {
            "first_token_s": round(self.t[0] - t0, 3),
            # every step after the first, evictions and tile searches included
            "decode_ms_per_token": round(1e3 * (self.t[-1] - self.t[0]) / len(steps), 2),
            # a typical step once the document is well under way
            "decode_ms_per_token_last_half_median": round(1e3 * half[len(half) // 2], 2),
        }


class LongGenBenchRunner:
    """End-to-end LongGenBench generation runner."""

    def __init__(self, config) -> None:
        self._assert_tracking_off(config)
        self.config = config
        self.lg = getattr(config, "longgenbench", None)
        if self.lg is None:
            raise ValueError("Config must have a 'longgenbench' section for longgenbench mode.")
        if self.lg.decoding not in DECODINGS:
            raise ValueError(f"longgenbench.decoding must be one of {DECODINGS}")

        cache_backend = getattr(config.cache, "backend", "dynamic")
        cache_package = getattr(config.cache, "backend_package", None)
        if cache_backend == "windowed" and cache_package:
            from utils.cache_factory import get_cache_classes, validate_backend_attn_pairing

            validate_backend_attn_pairing(cache_package, config.model.attn_implementation)
            (self.WindowedCache, self.WindowedCacheConfig,
             self.install_score_hooks) = get_cache_classes(cache_package)
            self.cache_backend_package = cache_package
            self.is_windowed = True
        else:
            self.WindowedCache = self.WindowedCacheConfig = self.install_score_hooks = None
            self.cache_backend_package = None
            self.is_windowed = False

        self.model = None
        self.tokenizer = None
        self._over_context_warned = False
        self._reset_type_state()

    def _reset_type_state(self) -> None:
        self._resolved_sample: Optional[Dict[str, Any]] = None
        self._n_no_eviction = 0
        self._n_no_int2 = 0
        self._retained: List[int] = []
        self._sequence: List[int] = []
        self._bytes_frac: List[float] = []
        self._step_ms: List[Optional[float]] = []

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_tracking_off(config) -> None:
        if getattr(getattr(config, "telemetry", None), "track_scores", False):
            raise ValueError(
                "track_scores must be False for LongGenBench runs: telemetry buffers "
                "grow with every decode step, and these generate up to 32,000 tokens.")

    def _load_model_and_tokenizer(self) -> Tuple:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                  "float32": torch.float32}
        log.info("Loading tokenizer: %s", cfg.model.name)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name, revision=getattr(cfg.model, "revision", None))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log.info("Loading model: %s (dtype=%s, attn=%s)", cfg.model.name, cfg.model.dtype,
                 cfg.model.attn_implementation)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            revision=getattr(cfg.model, "revision", None),
            torch_dtype=dtypes.get(cfg.model.dtype, torch.float16),
            attn_implementation=cfg.model.attn_implementation,
            device_map="auto",
        )
        model.eval()
        return model, tokenizer

    def _max_new_tokens(self) -> int:
        override = getattr(self.lg, "max_new_tokens", None)
        return int(override) if override else MAX_NEW_TOKENS[self.lg.setting]

    def _types(self) -> List[str]:
        types = self.lg.types or list(TYPES[self.lg.setting])
        for t in types:
            if t not in TYPES[self.lg.setting]:
                raise ValueError(f"{self.lg.setting} LongGenBench types are "
                                 f"{list(TYPES[self.lg.setting])}, got {t!r}")
        return list(types)

    def _eos_ids(self) -> List[int]:
        """The model's generation-config EOS ids: what vLLM stops on by default."""
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        return [int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos])]

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def _select_docs(self) -> List[Dict[str, Any]]:
        """``longgenbench.ids`` if given (any types, in that order), else every
        type's ``[start:end]`` slice."""
        lg = self.lg
        if lg.ids:
            by_id = {d["id"]: d for d in load_all(lg.data_dir, lg.setting)}
            missing = [i for i in lg.ids if i not in by_id]
            if missing:
                raise ValueError(f"longgenbench.ids not in the {lg.setting} dataset: {missing[:10]}")
            return [by_id[i] for i in lg.ids]
        docs: List[Dict[str, Any]] = []
        for t in self._types():
            docs += load_type(lg.data_dir, lg.setting, t, lg.start, lg.end)
        return docs

    def _plan(self, docs: List[Dict[str, Any]], lengths: Dict[int, int]) -> List[List[int]]:
        """The batches to run: ``longgenbench.batches`` if given (validated), else planned."""
        lg = self.lg
        if not lg.batches:
            return plan_batches(lengths, int(lg.batch_size), order=[d["id"] for d in docs])
        batches = [[int(i) for i in b] for b in lg.batches]
        flat = [i for b in batches for i in b]
        if sorted(flat) != sorted(d["id"] for d in docs) or len(flat) != len(set(flat)):
            raise ValueError("longgenbench.batches must cover exactly the selected documents, once each")
        for b in batches:
            if len(b) > int(lg.batch_size):
                raise ValueError(f"batch {b} is larger than longgenbench.batch_size={lg.batch_size}")
            if len({lengths[i] for i in b}) != 1:
                raise ValueError(f"batch {b} mixes prompt lengths {sorted({lengths[i] for i in b})}: "
                                 "batches must be equal-length (the cache does not take padding)")
        return batches

    def run(self) -> None:
        if self.is_windowed:
            from utils.cache_factory import assert_transformers_version_supported

            assert_transformers_version_supported()
        from modules.windowed_cache import flash_decode

        lg = self.lg
        dataset_sha(lg.data_dir, lg.setting)   # missing or altered data: fail before a model load
        docs = self._select_docs()
        if not docs:
            raise ValueError("no documents selected")
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        lengths = {d["id"]: int(encode_prompt(self.tokenizer, d["prompt"], truncation=False,
                                              return_tensors="pt").input_ids.shape[-1]) for d in docs}
        batches = self._plan(docs, lengths)
        by_id = {d["id"]: d for d in docs}
        max_new = self._max_new_tokens()
        self._cap = int(lg.batch_size)
        sizes = sorted({int(lg.batch_size), *map(int, lg.batch_fallback)}, reverse=True)
        output_dir = Path(lg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info("LongGenBench %s: %d documents in %d equal-length batches (cap %d, fallback %s), "
                 "max_new_tokens=%d, decoding=%s, windowed=%s, out=%s", lg.setting, len(docs),
                 len(batches), self._cap, sizes[1:], max_new, lg.decoding, self.is_windowed, output_dir)
        log_operating_point(self.config, self.is_windowed)

        # Process-global, cumulative counters: zero them so this run's sidecar
        # describes this run's decode steps and nothing else.
        flash_decode.reset_stats()
        self._reset_type_state()
        self._realised: List[Dict[str, Any]] = []
        self._oom_events: List[Dict[str, Any]] = []
        run_start = time.time()
        n_done = 0
        with open(output_dir / "generations.jsonl", "w", encoding="utf-8") as f:
            for bi, planned in enumerate(batches):
                for rec in self._run_planned(bi, [by_id[i] for i in planned], max_new, sizes):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_done += 1
                log.info("  batch %d/%d done (%d/%d documents)", bi + 1, len(batches), n_done, len(docs))
        self._report_compression("all")
        self._write_meta(docs, batches, lengths, max_new, run_start, time.time(), output_dir)
        log.info("LongGenBench run complete. Outputs in %s", output_dir)

    def _run_planned(self, bi: int, docs: List[Dict[str, Any]], max_new: int,
                     sizes: List[int]) -> List[Dict[str, Any]]:
        """One planned batch, split down the fallback sizes if it runs out of memory."""
        out: List[Dict[str, Any]] = []
        pending = split_evenly([d["id"] for d in docs], self._cap) if len(docs) > self._cap else [
            [d["id"] for d in docs]]
        by_id = {d["id"]: d for d in docs}
        while pending:
            ids = pending.pop(0)
            try:
                recs = self._generate_batch(bi, [by_id[i] for i in ids], max_new)
            except torch.cuda.OutOfMemoryError:
                self._recover_from_oom()
                smaller = next_smaller(len(ids), sizes)
                self._oom_events.append({"batch": bi, "ids": ids, "size": len(ids), "fell_back_to": smaller})
                if smaller is None:
                    if not getattr(self.lg, "skip_oom", False):
                        raise
                    log.warning("batch %d: %d documents ran out of memory at the smallest allowed size; "
                                "recording null outputs", bi, len(ids))
                    recs = [self._record(by_id[i], None, {"oom": True, "batch": bi, "batch_size": len(ids),
                                                          "batch_ids": ids}) for i in ids]
                    self._realised.append({"batch": bi, "ids": ids, "size": len(ids), "oom": True})
                    out += recs
                    continue
                log.warning("batch %d: out of memory at %d documents; batch size falls back to %d for the "
                            "rest of this run", bi, len(ids), smaller)
                self._cap = min(self._cap, smaller)
                pending = split_evenly(ids, smaller) + pending
                continue
            out += recs
        return out

    def _recover_from_oom(self) -> None:
        """Free what the failed batch held and disarm the fused path it left armed."""
        from modules.windowed_cache import flash_decode

        flash_decode._PENDING["ctx"] = None   # an OOM between arming and firing leaves it set
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _record(self, d: Dict[str, Any], output: Optional[str], stats: Dict[str, Any]) -> Dict[str, Any]:
        cr = completion_rate(parse_blocks(split_blocks(output), d["type"]), d["number"]) if output else 0.0
        return {
            "id": d["id"], "type": d["type"], "number": d["number"], "prefix": d["prefix"],
            "checks_once": d["checks_once"], "checks_range": d["checks_range"],
            "checks_periodic": d["checks_periodic"],
            "output": output,
            "n_blocks": len(split_blocks(output)) if output is not None else 0,
            "word_count": word_count(output) if output is not None else 0,
            "completion_rate": round(cr, 2),
            **stats,
        }

    # ------------------------------------------------------------------
    # generate one batch (equal-length prompts, no padding)
    # ------------------------------------------------------------------

    def _generate_batch(self, bi: int, docs: List[Dict[str, Any]], max_new: int) -> List[Dict[str, Any]]:
        model, tokenizer = self.model, self.tokenizer
        rows = [encode_prompt(tokenizer, d["prompt"], truncation=False, return_tensors="pt").input_ids[0]
                for d in docs]
        ctx = int(rows[0].shape[-1])
        if any(int(r.shape[-1]) != ctx for r in rows):
            raise ValueError(f"batch {bi}: prompts of different lengths {[int(r.shape[-1]) for r in rows]}; "
                             "batches must be equal-length (the cache does not take padding)")
        input_ids = torch.stack(rows).to(model.device)
        attention_mask = torch.ones_like(input_ids)      # no padding, so transformers never goes varlen
        self._warn_if_over_context(ctx, max_new)

        cache = hooks = None
        if self.is_windowed:
            cache, hooks = self._setup_windowed_cache(input_ids, max_new)

        eos_ids = self._eos_ids()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
        gen_kwargs: Dict[str, Any] = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_new,
            "eos_token_id": eos_ids,
            "pad_token_id": pad_id,
            "stop_strings": [STOP_STRING],
            "tokenizer": tokenizer,
            "num_beams": 1,
        }
        if self.lg.decoding == "greedy":
            gen_kwargs.update(do_sample=False, temperature=1.0, top_p=1.0)
        else:
            # One seed, set before every batch: a batch is a deterministic function
            # of its documents, and both arms of a comparison start each batch from
            # the same generator state.
            torch.manual_seed(int(self.lg.sample_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.lg.sample_seed))
            gen_kwargs.update(do_sample=True, temperature=GENERATION_SAMPLING["temperature"],
                              top_p=GENERATION_SAMPLING["top_p"], top_k=GENERATION_SAMPLING["top_k"])
        if self.cache_backend_package == "eager":
            gen_kwargs["output_attentions"] = True
        if cache is not None:
            gen_kwargs["past_key_values"] = cache
        clock = _StepClock()
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([clock])

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                out = model.generate(input_ids, **gen_kwargs)
        finally:
            if hooks is not None:
                hooks.remove()
        if cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        timing = clock.summary(t0)
        ids = [d["id"] for d in docs]
        batch_stats = {
            "batch": bi, "batch_size": len(docs), "batch_ids": ids, "batch_steps": len(clock.t),
            "batch_wall_s": round(wall, 3),
            "first_token_s": timing["first_token_s"],
            # one decode step serves every row of the batch
            "decode_ms_per_step": timing["decode_ms_per_token"],
            "decode_ms_per_step_last_half_median": timing["decode_ms_per_token_last_half_median"],
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if cuda else None,
        }
        self._realised.append({"batch": bi, "ids": ids, "size": len(docs), "prompt_tokens": ctx,
                               "steps": len(clock.t), "decode_ms_per_step": timing["decode_ms_per_token"]})
        self._step_ms.append(timing["decode_ms_per_token"])

        records = []
        r = cache.resolved if cache is not None else None
        for row, d in enumerate(docs):
            new = out[row][ctx:].tolist()
            # Generation ends at this row's first EOS; after a row finishes (EOS
            # or the stop string), generate fills it with the pad id, which is an
            # EOS id here, so the first EOS id also marks the end of a stop-string row.
            k = next((j for j, t in enumerate(new) if t in eos_ids), len(new))
            text = tokenizer.decode(new[:k], skip_special_tokens=True)
            cut = text.find(STOP_STRING)          # vLLM ends the text BEFORE the stop string
            if cut >= 0:
                text, stopped_on, n_gen = text[:cut], "stop_string", k
            elif k < len(new):
                stopped_on, n_gen = "eos", k + 1
            elif len(new) >= max_new:
                stopped_on, n_gen = "max_new_tokens", len(new)
            else:
                stopped_on, n_gen = "other", len(new)
            seq = ctx + n_gen
            stats: Dict[str, Any] = {
                "prompt_tokens": ctx, "gen_tokens": n_gen, "hit_token_cap": n_gen >= max_new,
                "stopped_on": stopped_on, **batch_stats,
                "prompt_sha": hashlib.sha256(d["prompt"].encode("utf-8")).hexdigest()[:16],
            }
            if r is not None:
                cap = int(r.retained_tokens)
                # Keys the fp16 tier (sink + local + top_k_fp windows) holds. A longer
                # sequence has int2 windows, so the read gate has something to select
                # over; longer than `cap`, keys have been dropped.
                fp_cap = int(r.num_sink_tokens + r.local_tokens + r.top_k_fp * r.window_size)
                stats.update({
                    "budget_tokens": int(r.total_budget_tokens), "retained_tokens_cap": cap,
                    "fp_tokens_cap": fp_cap, "N_q": int(r.N_q),
                    "int2_used": seq > fp_cap, "evicted": seq > cap,
                    # Upper bound on the KV bytes held when this row finished, as a
                    # fraction of what the full cache holds for the same sequence.
                    "kv_bytes_fraction_upper": round(
                        min(1.0, r.retained_bytes / max(seq * r.bytes_per_token, 1)), 4),
                })
                self._retained.append(min(cap, seq))
                self._sequence.append(seq)
                self._bytes_frac.append(stats["kv_bytes_fraction_upper"])
                self._n_no_eviction += int(not stats["evicted"])
                self._n_no_int2 += int(not stats["int2_used"])
                if self._resolved_sample is None:
                    self._resolved_sample = self._geometry(r, ctx)
            records.append(self._record(d, d["prefix"] + text, stats))
            log.info("    id %d (%s): %d new tokens, %d words, CR %.1f%%, stopped on %s", d["id"], d["type"],
                     n_gen, records[-1]["word_count"], records[-1]["completion_rate"], stopped_on)
        log.info("  batch %d: %d documents x %d prompt tokens, %d steps in %.0f s (first token %s s, then %s "
                 "ms/step; %s ms typical late)", bi, len(docs), ctx, len(clock.t), wall, timing["first_token_s"],
                 timing["decode_ms_per_token"], timing["decode_ms_per_token_last_half_median"])
        # Drop this frame's references before empty_cache, or the cache's
        # buffers are still alive when the allocator is asked to release them.
        del out, input_ids, attention_mask, rows
        cache = hooks = r = None
        self._cleanup_memory()
        return records

    @staticmethod
    def _geometry(r, prefill_len: int) -> Dict[str, Any]:
        return {
            "prefill_len": prefill_len,
            "quant_budget_mode": r.quant_budget_mode,
            "window_size": int(r.window_size),
            "num_sink_tokens": int(r.num_sink_tokens),
            "local_tokens": int(r.local_tokens),
            "top_k_windows": int(r.top_k_windows),
            "top_k_fp": int(r.top_k_fp),
            "N_q": int(r.N_q),
            "quant_card_bits": dict(r.quant_card_bits._asdict()),
            "bytes_per_gate_card": int(r.bytes_per_gate_card),
            "retained_windows": int(r.retained_windows),
            "retained_tokens": int(r.retained_tokens),
            "total_budget_tokens": int(r.total_budget_tokens),
            "retained_bytes": int(r.retained_bytes),
            "total_budget_bytes": int(r.total_budget_bytes),
            "budget_utilisation": round(r.budget_utilisation, 4),
        }

    def _setup_windowed_cache(self, input_ids: torch.Tensor, max_gen_len: int):
        """Windowed cache + hooks, built exactly as the GSM8K / LongBench runners build it."""
        from utils.cache_factory import (quant_budget_mode_kwargs,
                                         quant_gate_ratio_kwargs,
                                         quant_tier_policy_kwargs)

        cfg = self.config
        model = self.model
        budget = cfg.cache.cache_budget if cfg.cache.cache_budget is not None else 0.20
        cache_config = self.WindowedCacheConfig(
            window_size=cfg.cache.window_size,
            num_sink_tokens=cfg.cache.num_sink_tokens,
            local_window_size=cfg.cache.local_window_size,
            cache_budget=budget,
            rerotate_on_evict=getattr(cfg.cache, "rerotate_on_evict", False),
            quant_ratio=getattr(cfg.cache, "quant_ratio", 0.0),
            **quant_budget_mode_kwargs(
                self.WindowedCacheConfig,
                getattr(cfg.cache, "quant_budget_mode", "bytes")),
            quant_memoize_read=getattr(cfg.cache, "quant_memoize_read", None),
            **quant_gate_ratio_kwargs(
                self.WindowedCacheConfig,
                getattr(cfg.cache, "quant_gate_ratio", 0.25)),
            # Gate-card field widths (sketch.CardBits); None means the shipped card.
            quant_card_bits=getattr(cfg.cache, "quant_card_bits", None),
            **quant_tier_policy_kwargs(cfg.cache),
            first_eviction_step=getattr(cfg.cache, "first_eviction_step",
                                        FIRST_EVICTION_STEP_DEFAULT),
        )

        rope = None
        for name, mod in model.named_modules():
            if "rotary" in name.lower() or "rope" in name.lower():
                rope = mod
                break
        if rope is None:
            for name, mod in model.named_modules():
                if hasattr(mod, "rotary_emb"):
                    rope = mod.rotary_emb
                    break
        if rope is None:
            from utils.config import ConfigValidationError

            raise ConfigValidationError(
                "Could not locate a RoPE module on the model. WindowedCache "
                "requires a rotary embedding module.")

        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                  "float32": torch.float32}
        cache = self.WindowedCache(
            config=cache_config,
            prefill_len=int(input_ids.shape[-1]),
            model_config=model.config,
            kv_dtype=dtypes.get(cfg.model.dtype, torch.float16),
            rope_module=rope,
            num_layers=model.config.num_hidden_layers,
            max_tokens=max_gen_len,
        )
        hooks = self.install_score_hooks(model, cache, cache_config)
        return cache, hooks

    def _warn_if_over_context(self, ctx: int, max_new: int) -> None:
        model_max = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if model_max and ctx + max_new > model_max and not self._over_context_warned:
            log.warning("prompt + generation = %d tokens exceeds the model's %d-token "
                        "context; output past it is unreliable.", ctx + max_new, model_max)
            self._over_context_warned = True

    def _read_gate_report(self, label: str) -> dict:
        """Did the read gate actually run? Recorded into every sidecar.

        An ungated run reads the whole int2 tier, writes a correct document and
        scores normally, so the verdict is recorded rather than inferred from
        the config (``CLAUDE.md``, "It is the only read path").
        """
        from modules.windowed_cache import flash_decode

        cache_cfg = getattr(self.config, "cache", None)
        expect = flash_decode.expect_gated(
            getattr(cache_cfg, "backend_package", None),
            getattr(cache_cfg, "quant_ratio", 0.0),
            torch.cuda.is_available(),
        )
        return flash_decode.log_gate_report(log, label, expect)

    def _cleanup_memory(self) -> None:
        if getattr(self.lg, "aggressive_cache_clear", True) and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # diagnostics + meta
    # ------------------------------------------------------------------

    def _compression_summary(self) -> Dict[str, Any]:
        if not self._sequence:
            return {"compression_active": False if not self.is_windowed else None,
                    "n_examples_no_eviction": None}
        n = len(self._sequence)
        return {
            "compression_active": self._n_no_int2 < n,
            "n_examples_no_eviction": self._n_no_eviction,
            "n_examples_no_int2": self._n_no_int2,
            "mean_sequence_tokens": round(sum(self._sequence) / n, 1),
            "mean_retained_tokens": round(sum(self._retained) / n, 1),
            # Keys retained (either tier) over keys the full cache holds at the
            # end of each document. An int2 key costs ~1/4 of an fp16 one under
            # quant_budget_mode "bytes", so this is well above cache_budget.
            "effective_retention": round(sum(self._retained) / max(sum(self._sequence), 1), 4),
            # The memory claim: KV bytes held over the full cache's, per document
            # (an upper bound), averaged.
            "mean_kv_bytes_fraction_upper": round(sum(self._bytes_frac) / n, 4),
        }

    def _report_compression(self, type_: str) -> None:
        if not self.is_windowed:
            log.info("%s: full-cache baseline, no compression applied.", type_)
            return
        s = self._compression_summary()
        if s.get("n_examples_no_eviction") is None:
            return
        log.info("%s compression: mean sequence %.0f tok, mean retained %.0f tok "
                 "(keys %.3f, KV bytes <= %.3f of full); int2 tier unused on %d/%d, "
                 "no key dropped on %d/%d", type_, s["mean_sequence_tokens"],
                 s["mean_retained_tokens"], s["effective_retention"],
                 s["mean_kv_bytes_fraction_upper"], s["n_examples_no_int2"],
                 len(self._sequence), s["n_examples_no_eviction"], len(self._sequence))
        if s["n_examples_no_int2"]:
            log.warning("%s: %d document(s) never filled past the fp16 tier, so they were "
                        "generated with no int2 window and no read gate.", type_,
                        s["n_examples_no_int2"])

    def _write_meta(self, docs: List[Dict[str, Any]], planned: List[List[int]], lengths: Dict[int, int],
                    max_new: int, run_start: float, run_end: float, output_dir: Path) -> None:
        cfg, lg = self.config, self.lg
        budget = cfg.cache.cache_budget
        from utils.cache_factory import quant_card_bits_record, quant_tier_policy_record

        realised = self._realised
        done = [b for b in realised if not b.get("oom")]
        steps = sum(b["steps"] for b in done)
        meta = {
            "read_gate": self._read_gate_report(f"longgenbench {lg.setting}"),
            "quant_card_bits": quant_card_bits_record(cfg.cache),
            **quant_tier_policy_record(cfg.cache),
            "benchmark": "LongGenBench",
            "setting": lg.setting,
            "types": sorted({d["type"] for d in docs}),
            "dataset": HF_DATASET,
            "dataset_revision": HF_REVISION,
            "reference_code": CODE_REPO,
            "reference_generation": GENERATION_CODE,
            "data_file": FILES[lg.setting][0],
            "dataset_sha": dataset_sha(lg.data_dir, lg.setting),
            "prompt_format": PROMPT_FORMAT,
            "ids": sorted(d["id"] for d in docs),
            "num_examples": len(docs),
            "num_oom": sum(len(b["ids"]) for b in realised if b.get("oom")),
            "prompt_tokens": {str(i): lengths[i] for i in sorted(lengths)},
            "max_new_tokens": max_new,
            "max_new_tokens_override": getattr(lg, "max_new_tokens", None),
            "generation": {
                "decoding": lg.decoding,
                "sampling": ({**GENERATION_SAMPLING, "seed": int(lg.sample_seed),
                              "seed_rule": "the same seed set before every batch"}
                             if lg.decoding == "reference" else None),
                "stop_string": STOP_STRING,
                "stop_string_in_output": False,
                "eos_token_id": self._eos_ids(),
                "output": "prefix + generation",
                "batching": "equal-length prompts, no padding",
                "batch_size": int(lg.batch_size),
                "batch_fallback": [int(x) for x in lg.batch_fallback],
            },
            "batches_planned": planned,
            "batches_realised": realised,
            # Every out-of-memory batch: its size and the size it fell back to
            # (None: no smaller size was allowed, its documents scored as failed).
            "oom_events": self._oom_events,
            "final_batch_cap": self._cap,
            "model_name": cfg.model.name,
            "model_revision": getattr(cfg.model, "revision", None),
            "tokenizer_sha": self._get_tokenizer_sha(),
            "dtype": cfg.model.dtype,
            "attn_implementation": cfg.model.attn_implementation,
            "cache_type": "windowed" if self.is_windowed else "full_cache",
            "cache_backend_package": self.cache_backend_package,
            "cache_budget": budget,
            "budget_basis": "prefill_plus_max_new_tokens",
            "compression_ratio": round(1.0 - budget, 4) if budget else None,
            "window_size": cfg.cache.window_size,
            "num_sink_tokens": cfg.cache.num_sink_tokens,
            "local_window_size": cfg.cache.local_window_size,
            "quant_ratio": getattr(cfg.cache, "quant_ratio", 0.0),
            "quant_budget_mode": getattr(cfg.cache, "quant_budget_mode", "bytes"),
            "quant_gate_ratio": getattr(cfg.cache, "quant_gate_ratio", 0.25),
            "quant_memoize_read": getattr(cfg.cache, "quant_memoize_read", None),
            "first_eviction_step": getattr(cfg.cache, "first_eviction_step",
                                           FIRST_EVICTION_STEP_DEFAULT),
            "rerotate_on_evict": getattr(cfg.cache, "rerotate_on_evict", False),
            "resolved_geometry_first_example": self._resolved_sample,
            **self._compression_summary(),
            "seed": cfg.run.seed,
            "track_scores": False,
            # Wall clock inside a quality run (prefill + compiles included):
            # context for the reader, not a performance number.
            "total_decode_steps": steps,
            "decode_ms_per_step_by_batch": [b["decode_ms_per_step"] for b in done],
            **capture_environment(),
            "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_start)),
            "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_end)),
            "run_wall_s": round(run_end - run_start, 1),
        }
        (output_dir / "generations.meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def _get_tokenizer_sha(self) -> str:
        if self.tokenizer is None:
            return "unknown"
        try:
            from utils.hashing import sha256_tokenizer

            return sha256_tokenizer(self.tokenizer)
        except Exception:
            return "unknown"
