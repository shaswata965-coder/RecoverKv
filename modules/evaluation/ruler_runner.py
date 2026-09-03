"""RULER evaluation runner — one runner, both backends, any subset of tasks.

Mirrors ``LongBenchRunner`` exactly for model loading, windowed-cache setup,
and the generation loop; only the data source and prompt construction differ,
following DefensiveKV's RULER protocol
(``kv_cache/DefensiveKV/kvpress/pipeline.py`` + ``evaluation/evaluate.py``):
- RULER (vendored locally as ``datasets.save_to_disk`` arrow splits per
  context length, see ``data/ruler_loader.py``)
- Llama-3.1-8B-Instruct fp16, chat-templated prompt, greedy decoding
- Per-example ``max_new_tokens`` from the dataset (no fixed generation cap)

Prompt construction mirrors ``kvpress.pipeline.KVPressTextGenerationPipeline
.preprocess``: apply the chat template to a single user turn containing
``context + question``, then append ``answer_prefix`` verbatim after the
generation-prompt header.

Backend routing via ``utils/cache_factory.py`` (same as LongBenchRunner).

Optional per-example memory accounting via ``utils/cache_memory.py``
(``ruler.capture_memory: true``): after each generate call, before the cache
is torn down, the cache's exact byte breakdown (fp16 tier, int4 Q tier,
bookkeeping; retained tokens vs. observed context) is captured and written to
``<task>.memory.jsonl`` (one record per example) plus a
``<task>.memory_summary.json`` aggregate (mean/min/max across examples).
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from data.ruler_loader import RULER_TASKS, load_ruler_dataset
from utils.config import FIRST_EVICTION_STEP_DEFAULT, log_operating_point
from utils.prompting import encode_prompt
from utils.env_capture import capture_environment
from utils.logger import get_logger

log = get_logger(__name__)


class RulerRunner:
    """End-to-end RULER prediction runner.

    One runner handles all (or a configured subset of) the RULER tasks for a
    single context-length split, on either cache backend (flash_attn /
    eager), routed via the factory in ``utils/cache_factory.py``.
    """

    def __init__(self, config) -> None:
        self._assert_tracking_off(config)
        self.config = config

        self.ruler = getattr(config, "ruler", None)
        if self.ruler is None or not getattr(self.ruler, "data_dir", None):
            raise ValueError(
                "Config must have a 'ruler' section with 'data_dir' set for RULER mode."
            )

        cache_backend = getattr(config.cache, "backend", "dynamic")
        cache_package = getattr(config.cache, "backend_package", None)

        if cache_backend == "windowed" and cache_package:
            from utils.cache_factory import (
                get_cache_classes,
                validate_backend_attn_pairing,
            )

            validate_backend_attn_pairing(
                cache_package, config.model.attn_implementation
            )
            (
                self.WindowedCache,
                self.WindowedCacheConfig,
                self.install_score_hooks,
            ) = get_cache_classes(cache_package)
            self.cache_backend_package = cache_package
            self.is_windowed = True
        else:
            self.WindowedCache = None
            self.WindowedCacheConfig = None
            self.install_score_hooks = None
            self.cache_backend_package = None
            self.is_windowed = False

        self.model = None
        self.tokenizer = None
        self._over_context_warned = False
        self._memory_reports: List[Dict[str, Any]] = []
        self._last_report = None

    @staticmethod
    def _assert_tracking_off(config) -> None:
        """Guard: track_scores must be False for RULER runs (see LongBenchRunner)."""
        track = getattr(getattr(config, "telemetry", None), "track_scores", False)
        if track:
            raise ValueError(
                "track_scores must be False for RULER runs. "
                "Telemetry buffers grow linearly with run length and "
                "would distort throughput numbers + risk OOM. "
                "Set telemetry.track_scores: false in your config. "
                "Use the parity runner if you want telemetry."
            )

    def _load_model_and_tokenizer(self) -> Tuple:
        """Load model and tokenizer (lazy, called once)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config

        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_dtype = dtypes.get(cfg.model.dtype, torch.float16)

        log.info("Loading tokenizer: %s", cfg.model.name)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            revision=getattr(cfg.model, "revision", None),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info(
            "Loading model: %s (dtype=%s, attn=%s)",
            cfg.model.name,
            cfg.model.dtype,
            cfg.model.attn_implementation,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            revision=getattr(cfg.model, "revision", None),
            torch_dtype=model_dtype,
            attn_implementation=cfg.model.attn_implementation,
            device_map="auto",
        )
        model.eval()

        return model, tokenizer

    def run(self) -> None:
        """Run predictions on all configured RULER tasks."""
        # Fail fast on an unsupported transformers version: the windowed
        # cache's RoPE handling assumes monotonic cache_position
        # (transformers <= 4.47), see utils.cache_factory.
        if self.is_windowed:
            from utils.cache_factory import assert_transformers_version_supported

            assert_transformers_version_supported()
            self._warn_on_cache_window_disagreement()

        self.model, self.tokenizer = self._load_model_and_tokenizer()

        tasks = getattr(self.ruler, "tasks", None) or RULER_TASKS
        if isinstance(tasks, str):
            tasks = [tasks]

        examples = load_ruler_dataset(self.ruler.data_dir, tasks=tasks)

        output_dir = Path(getattr(self.ruler, "output_dir", "outputs/ruler"))
        output_dir.mkdir(parents=True, exist_ok=True)

        resume = getattr(self.ruler, "resume", False)

        # Group examples by task, preserving on-disk order within each task.
        by_task: Dict[str, List[Dict[str, Any]]] = {}
        for ex in examples:
            by_task.setdefault(ex["task"], []).append(ex)

        log.info(
            "RULER run: %d tasks (%s), output_dir=%s, windowed=%s",
            len(by_task),
            sorted(by_task.keys()),
            output_dir,
            self.is_windowed,
        )
        log_operating_point(self.config, self.is_windowed)

        for task_name, task_examples in by_task.items():
            jsonl_path = output_dir / f"{task_name}.jsonl"

            if resume and jsonl_path.exists():
                existing_lines = len(
                    jsonl_path.read_text(encoding="utf-8").strip().splitlines()
                )
                if existing_lines > 0:
                    log.info(
                        "Skipping %s (resume=true, %d lines exist)",
                        task_name,
                        existing_lines,
                    )
                    continue

            self._run_task(task_name, task_examples, output_dir)

        log.info("RULER run complete. Outputs in %s", output_dir)

    def _run_task(
        self, task_name: str, task_examples: List[Dict[str, Any]], output_dir: Path
    ) -> None:
        """Run predictions on a single RULER task."""
        log.info("=== Task: %s ===", task_name)

        ns = getattr(self.ruler, "num_samples", "max")
        if isinstance(ns, int) and ns >= 0:
            total = len(task_examples)
            if ns < total:
                log.info(
                    "%s: capping examples %d -> %d (ruler.num_samples=%d)",
                    task_name, total, ns, ns,
                )
                task_examples = task_examples[:ns]

        out_path = output_dir / f"{task_name}.jsonl"
        skip_oom = getattr(self.ruler, "skip_oom", False)
        capture_memory = self.is_windowed and getattr(self.ruler, "capture_memory", False)

        run_start = time.time()
        n_examples = 0
        n_oom = 0
        self._memory_reports = []

        with open(out_path, "w", encoding="utf-8") as f:
            for i, ex in enumerate(task_examples):
                mem_report = None
                try:
                    pred, mem_report = self._predict(ex, capture_memory=capture_memory)
                except torch.cuda.OutOfMemoryError:
                    if skip_oom:
                        log.warning(
                            "%s example %d: OOM — recording null pred", task_name, i
                        )
                        pred = None
                        n_oom += 1
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        raise

                record = {
                    "pred": pred,
                    "answer": ex["answer"],
                    "task": ex["task"],
                    "max_new_tokens": ex["max_new_tokens"],
                    "id": i,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_examples += 1

                if mem_report is not None:
                    self._memory_reports.append(mem_report)
                    if i == 0:
                        from utils.cache_memory import format_report

                        log.info(
                            "\n%s",
                            format_report(
                                self._last_report, label=f"{task_name} ex=0"
                            ),
                        )

                if (i + 1) % 50 == 0:
                    log.info(
                        "  %s: %d/%d examples done", task_name, i + 1, len(task_examples)
                    )

        run_end = time.time()
        elapsed = run_end - run_start
        eps = n_examples / elapsed if elapsed > 0 else 0

        log.info(
            "%s: %d examples, %d OOM, %.1fs (%.2f ex/s)",
            task_name,
            n_examples,
            n_oom,
            elapsed,
            eps,
        )

        self._write_meta(task_name, n_examples, run_start, run_end, eps, output_dir)
        if self._memory_reports:
            self._write_memory(task_name, output_dir)

    def _predict(
        self, ex: Dict[str, Any], capture_memory: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Generate a prediction for a single RULER example.

        Follows kvpress.pipeline.KVPressTextGenerationPipeline exactly: the
        final string is
        ``chat_template(context + question, add_generation_prompt=True) + answer_prefix``.

        Returns ``(pred, memory_report_dict_or_None)``.
        """
        cfg = self.config
        model = self.model
        tokenizer = self.tokenizer

        max_gen_len = int(ex["max_new_tokens"])

        # 1. Build the chat-templated prompt (RULER always uses the chat
        #    template — none of its task names are in DefensiveKV's
        #    no-template exception set, unlike some LongBench datasets).
        messages = [{"role": "user", "content": ex["context"] + ex["question"]}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{ex['context'] + ex['question']}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        prompt = prompt + ex["answer_prefix"]

        # 2. Tokenize (no truncation — RULER splits are sized to fit the
        #    model's context window at generation time; DefensiveKV doesn't
        #    pre-truncate RULER either).
        #    encode_prompt, not tokenizer(): the templated string above already
        #    carries the BOS, and tokenizer()'s add_special_tokens default would
        #    add a second — shifting every position and spending a protected
        #    sink slot on a duplicate. kvpress tokenizes without re-adding
        #    specials; this matches it.
        inputs = encode_prompt(tokenizer, prompt, truncation=False,
                               return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        # Explicit mask: pad_token == eos_token on Llama, so transformers warns
        # on every example otherwise. All-ones at batch size 1.
        attention_mask = inputs.attention_mask.to(model.device)
        context_length = input_ids.shape[-1]

        self._warn_if_over_context(context_length, max_gen_len)

        # 3. Set up cache
        cache = None
        hooks = None

        if self.is_windowed:
            cache, hooks = self._setup_windowed_cache(input_ids, max_gen_len)

        # 4. Generate
        try:
            gen_kwargs = {
                "attention_mask": attention_mask,
                "max_new_tokens": max_gen_len,
                "num_beams": 1,
                "do_sample": False,
                "temperature": 1.0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }

            if self.cache_backend_package == "eager":
                gen_kwargs["output_attentions"] = True

            if cache is not None:
                gen_kwargs["past_key_values"] = cache

            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)

        finally:
            if hooks is not None:
                hooks.remove()

        # 5. Capture memory BEFORE the cache is torn down. The actual
        #    generated length (may stop early on EOS) is the honest
        #    full-context baseline, not the requested max_gen_len.
        mem_dict = None
        if capture_memory and cache is not None:
            from utils.cache_memory import measure_cache_memory

            n_generated = output.shape[-1] - context_length
            report = measure_cache_memory(
                cache, full_context_len=context_length + n_generated
            )
            mem_dict = report.to_dict()
            mem_dict["task"] = ex["task"]
            self._last_report = report

        # 6. Decode only new tokens
        pred = tokenizer.decode(
            output[0][context_length:], skip_special_tokens=True
        )

        # 7. Memory hygiene
        self._cleanup_memory(cache)

        return pred, mem_dict

    def _setup_windowed_cache(self, input_ids: torch.Tensor, max_gen_len: int):
        """Create windowed cache and install hooks (identical to LongBenchRunner)."""
        from utils.cache_factory import quant_budget_mode_kwargs

        cfg = self.config
        model = self.model

        # RULER reads window params from cfg.cache.*; parity runners read
        # from cfg.window.*. Warn loudly if the two configs disagree so a user
        # who copied a parity template doesn't silently get cache defaults.
        self._warn_on_cache_window_disagreement()

        budget = cfg.cache.cache_budget if cfg.cache.cache_budget is not None else 0.20
        cache_config = self.WindowedCacheConfig(
            window_size=cfg.cache.window_size,
            num_sink_tokens=cfg.cache.num_sink_tokens,
            local_window_size=cfg.cache.local_window_size,
            cache_budget=budget,
            rerotate_on_evict=getattr(cfg.cache, "rerotate_on_evict", False),
            quant_ratio=getattr(cfg.cache, "quant_ratio", 0.0),
            # Without this the YAML knob is inert here and the run inherits
            # whatever the dataclass default happens to be — the failure that
            # produced ACCURACY_RECOVERY_PLAN.md §2 on LongBench. Routed through
            # the factory: the eager package has no such field.
            **quant_budget_mode_kwargs(
                self.WindowedCacheConfig,
                getattr(cfg.cache, "quant_budget_mode", "bytes")),
            quant_memoize_read=getattr(cfg.cache, "quant_memoize_read", None),
            first_eviction_step=getattr(cfg.cache, "first_eviction_step", FIRST_EVICTION_STEP_DEFAULT),
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
                "requires a rotary embedding module for key rerotation."
            )

        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        cache = self.WindowedCache(
            config=cache_config,
            prefill_len=input_ids.shape[-1],
            model_config=model.config,
            kv_dtype=dtypes.get(cfg.model.dtype, torch.float16),
            rope_module=rope,
            num_layers=model.config.num_hidden_layers,
            max_tokens=max_gen_len,
        )

        hooks = self.install_score_hooks(model, cache, cache_config)
        return cache, hooks

    def _warn_if_over_context(self, context_length: int, max_gen_len: int) -> None:
        """Warn (once) if prompt + generation exceeds the model's context window."""
        model_max = getattr(getattr(self.model, "config", None),
                            "max_position_embeddings", None)
        if not model_max:
            return
        needed = context_length + max_gen_len
        if needed > model_max and not self._over_context_warned:
            log.warning(
                "Prompt+generation = %d tokens exceeds the model's context "
                "window (%d). Positions beyond the window are out-of-distribution "
                "and scores will be unreliable. Suppressing further warnings.",
                needed, model_max,
            )
            self._over_context_warned = True

    def _cleanup_memory(self, cache=None) -> None:
        """Memory hygiene between examples."""
        if cache is not None:
            del cache
        aggressive = getattr(self.ruler, "aggressive_cache_clear", False)
        if aggressive and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _write_memory(self, task_name: str, output_dir: Path) -> None:
        """Write per-example memory records + an aggregate summary.

        ``<task>.memory.jsonl`` — one ``CacheMemoryReport.to_dict()`` per
        example. ``<task>.memory_summary.json`` — mean/min/max of the
        headline byte totals and the compression ratios across examples, so a
        window_size / cache_budget / quant_ratio sweep can be compared at a
        glance without re-parsing every jsonl line.
        """
        reports = self._memory_reports
        mem_path = output_dir / f"{task_name}.memory.jsonl"
        with open(mem_path, "w", encoding="utf-8") as f:
            for d in reports:
                f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")

        def _stats(key: str) -> Dict[str, float]:
            vals = [r[key] for r in reports if r.get(key) is not None]
            if not vals:
                return {"mean": None, "min": None, "max": None}
            return {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }

        summary = {
            "task": task_name,
            "num_examples": len(reports),
            "total_live": _stats("total_live"),
            "total_alloc": _stats("total_alloc"),
            "fp_content_live": _stats("fp_content_live"),
            "q_content_live": _stats("q_content_live"),
            "bookkeeping_live": _stats("bookkeeping_live"),
            "reduction_vs_full": _stats("reduction_vs_full"),
            "compression_vs_fp16": _stats("compression_vs_fp16"),
            # The summary is what anyone actually reads — the jsonl is for
            # re-analysis — so it must carry BOTH compression framings, not just
            # the memo-inclusive one. At B = 1 (where the memo defaults ON) the
            # memo is the largest line item and drags compression_vs_fp16 below
            # 1.0, i.e. the summary alone would report the two-tier cache as
            # bigger than fp16. See utils/cache_memory.py.
            "memo_bytes": _stats("memo_bytes"),
            "total_live_excl_memo": _stats("total_live_excl_memo"),
            "reduction_vs_full_excl_memo": _stats("reduction_vs_full_excl_memo"),
            "compression_vs_fp16_excl_memo": _stats(
                "compression_vs_fp16_excl_memo"
            ),
            "retained_tokens": _stats("retained_tokens"),
            "observed_context_len": _stats("observed_context_len"),
        }
        summary_path = output_dir / f"{task_name}.memory_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info(
            "%s: memory captured for %d examples -> %s / %s",
            task_name, len(reports), mem_path, summary_path,
        )

    def _write_meta(
        self,
        task_name: str,
        num_examples: int,
        run_start: float,
        run_end: float,
        eps: float,
        output_dir: Path,
    ) -> None:
        """Write per-task metadata sidecar JSON (mirrors LongBenchRunner)."""
        cfg = self.config
        env = capture_environment()

        budget = cfg.cache.cache_budget
        compression_ratio = round(1.0 - budget, 2) if budget else None

        meta = {
            "task": task_name,
            "data_dir": self.ruler.data_dir,
            "num_examples": num_examples,
            "model_name": cfg.model.name,
            "model_revision": getattr(cfg.model, "revision", None),
            "tokenizer_sha": self._get_tokenizer_sha(),
            "cache_type": "windowed" if self.is_windowed else "full_cache",
            "cache_backend_package": self.cache_backend_package,
            "cache_budget": budget,
            "compression_ratio": compression_ratio,
            "window_size": cfg.cache.window_size,
            "num_sink_tokens": cfg.cache.num_sink_tokens,
            "local_window_size": cfg.cache.local_window_size,
            "rerotate_on_evict": getattr(cfg.cache, "rerotate_on_evict", False),
            "quant_ratio": getattr(cfg.cache, "quant_ratio", 0.0),
            "quant_memoize_read": getattr(cfg.cache, "quant_memoize_read", None),
            "first_eviction_step": getattr(cfg.cache, "first_eviction_step", FIRST_EVICTION_STEP_DEFAULT),
            "track_scores": False,
            "attn_implementation": cfg.model.attn_implementation,
            "dtype": cfg.model.dtype,
            "num_samples_requested": getattr(self.ruler, "num_samples", "max"),
            "capture_memory": getattr(self.ruler, "capture_memory", False),
            "seed": cfg.run.seed,
            **env,
            "run_started_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_start)
            ),
            "run_finished_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_end)
            ),
            "examples_per_second": round(eps, 4),
        }

        meta_path = output_dir / f"{task_name}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    def _warn_on_cache_window_disagreement(self) -> None:
        """Warn if cfg.cache.* and cfg.window.* disagree on shared fields."""
        cfg = self.config
        pairs = [
            ("window_size", cfg.cache.window_size, cfg.window.window_size),
            ("num_sink_tokens", cfg.cache.num_sink_tokens, cfg.window.num_sink_tokens),
            ("local_window_size", cfg.cache.local_window_size, cfg.window.local_window_size),
        ]
        for name, cache_val, window_val in pairs:
            if cache_val != window_val:
                log.warning(
                    "cfg.cache.%s=%r != cfg.window.%s=%r — RULER uses "
                    "cfg.cache.* for the runtime cache. Update cfg.cache.%s "
                    "if you meant the cfg.window.* value.",
                    name, cache_val, name, window_val, name,
                )

    def _get_tokenizer_sha(self) -> str:
        """Get tokenizer SHA for reproducibility."""
        if self.tokenizer is None:
            return "unknown"
        try:
            from utils.hashing import sha256_tokenizer
            return sha256_tokenizer(self.tokenizer)
        except Exception:
            return "unknown"
