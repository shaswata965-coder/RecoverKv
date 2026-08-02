
from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from data.longbench_loader import (
    LONGBENCH_EN_DATASETS,
    load_longbench_dataset,
)
from utils.config import FIRST_EVICTION_STEP_DEFAULT, log_operating_point
from utils.prompting import encode_prompt
from utils.env_capture import capture_environment
from utils.hashing import sha256_file
from utils.logger import get_logger

log = get_logger(__name__)


class LongBenchRunner:

    NO_CHAT_TEMPLATE_DATASETS = frozenset(
        {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
    )

    def __init__(self, config) -> None:
        self._assert_tracking_off(config)
        self.config = config

        self.lb = getattr(config, "longbench", None)
        if self.lb is None:
            raise ValueError(
                "Config must have a 'longbench' section for LongBench mode."
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

        configs_dir = Path("data/longbench_configs")
        with open(configs_dir / "dataset2prompt.json", "r", encoding="utf-8") as f:
            self.dataset2prompt = json.load(f)
        with open(configs_dir / "dataset2maxlen.json", "r", encoding="utf-8") as f:
            self.dataset2maxlen = json.load(f)

        self._vendored_shas = {
            "longbench_dataset2prompt_sha": sha256_file(
                configs_dir / "dataset2prompt.json"
            ),
            "longbench_dataset2maxlen_sha": sha256_file(
                configs_dir / "dataset2maxlen.json"
            ),
            "longbench_dataset2metric_sha": sha256_file(
                configs_dir / "dataset2metric.json"
            ),
            "longbench_metrics_py_sha": self._compute_metrics_sha(),
        }

        self.model = None
        self.tokenizer = None
        self._over_context_warned = False

    @staticmethod
    def _compute_metrics_sha() -> str:
        metrics_path = Path("modules/evaluation/longbench_metrics.py")
        if metrics_path.exists():
            return sha256_file(metrics_path)
        return "unknown"

    @staticmethod
    def _assert_tracking_off(config) -> None:
        track = getattr(getattr(config, "telemetry", None), "track_scores", False)
        if track:
            raise ValueError(
                "track_scores must be False for LongBench runs. "
                "Telemetry buffers grow linearly with run length and "
                "would distort throughput numbers + risk OOM. "
                "Set telemetry.track_scores: false in your config. "
                "Use the parity runner if you want telemetry."
            )

    def _load_model_and_tokenizer(self) -> Tuple:
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
        if self.is_windowed:
            from utils.cache_factory import assert_transformers_version_supported

            assert_transformers_version_supported()
            self._warn_on_cache_window_disagreement()

        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self._warn_if_chat_gate_looks_wrong()

        datasets = getattr(self.lb, "datasets", LONGBENCH_EN_DATASETS)
        if isinstance(datasets, str):
            datasets = [datasets]

        output_dir = Path(getattr(self.lb, "output_dir", "outputs/longbench"))
        output_dir.mkdir(parents=True, exist_ok=True)

        resume = getattr(self.lb, "resume", False)

        log.info(
            "LongBench run: %d datasets, output_dir=%s, windowed=%s",
            len(datasets),
            output_dir,
            self.is_windowed,
        )
        log_operating_point(self.config, self.is_windowed)

        for dataset_name in datasets:
            jsonl_path = output_dir / f"{dataset_name}.jsonl"

            if resume and jsonl_path.exists():
                existing_lines = len(
                    jsonl_path.read_text(encoding="utf-8").strip().splitlines()
                )
                if existing_lines > 0:
                    log.info(
                        "Skipping %s (resume=true, %d lines exist)",
                        dataset_name,
                        existing_lines,
                    )
                    continue

            self._run_dataset(dataset_name, output_dir)

        log.info("LongBench run complete. Outputs in %s", output_dir)

    def _run_dataset(self, name: str, output_dir: Path) -> None:
        log.info("=== Dataset: %s ===", name)

        use_e = getattr(self.lb, "use_e_variants", False)
        examples = load_longbench_dataset(name, use_e_variant=use_e)
        examples_list = list(examples)

        ns = getattr(self.lb, "num_samples", "max")
        if isinstance(ns, int) and ns >= 0:
            total = len(examples_list)
            if ns < total:
                log.info(
                    "%s: capping examples %d → %d (longbench.num_samples=%d)",
                    name, total, ns, ns,
                )
                examples_list = examples_list[:ns]

        max_gen_len = self.dataset2maxlen.get(name, 128)
        prompt_template = self.dataset2prompt.get(name)
        if prompt_template is None:
            log.error("No prompt template for dataset %s — skipping", name)
            return

        out_path = output_dir / f"{name}.jsonl"
        skip_oom = getattr(self.lb, "skip_oom", False)

        run_start = time.time()
        n_examples = 0
        n_oom = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for i, ex in enumerate(examples_list):
                try:
                    pred = self._predict(ex, prompt_template, max_gen_len, name)
                except torch.cuda.OutOfMemoryError:
                    if skip_oom:
                        log.warning(
                            "%s example %d: OOM — recording null pred", name, i
                        )
                        pred = None
                        n_oom += 1
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        raise

                record = {
                    "pred": pred,
                    "answers": ex["answers"],
                    "all_classes": ex.get("all_classes"),
                    "length": ex["length"],
                    "_id": ex.get("_id", str(i)),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_examples += 1

                if (i + 1) % 50 == 0:
                    log.info(
                        "  %s: %d/%d examples done", name, i + 1, len(examples_list)
                    )

        run_end = time.time()
        elapsed = run_end - run_start
        eps = n_examples / elapsed if elapsed > 0 else 0

        log.info(
            "%s: %d examples, %d OOM, %.1fs (%.2f ex/s)",
            name,
            n_examples,
            n_oom,
            elapsed,
            eps,
        )

        self._write_meta(name, n_examples, max_gen_len, run_start, run_end, eps, output_dir)

    def _predict(
        self,
        ex: Dict[str, Any],
        prompt_template: str,
        max_gen_len: int,
        dataset_name: str,
    ) -> str:
        cfg = self.config
        model = self.model
        tokenizer = self.tokenizer

        prompt = prompt_template.format(
            context=ex["context"], input=ex.get("input", "")
        )

        max_length = getattr(self.lb, "max_length", None)
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if max_length and max_length > 0 and len(tokenized) > max_length:
            half = max_length // 2
            prompt = tokenizer.decode(
                tokenized[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized[-half:], skip_special_tokens=True)

        if self._should_apply_chat_template(cfg.model.name, dataset_name):
            messages = [{"role": "user", "content": prompt}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{prompt}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )

        inputs = encode_prompt(tokenizer, prompt, truncation=False,
                               return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        context_length = input_ids.shape[-1]

        self._warn_if_over_context(context_length, max_gen_len)

        cache = None
        hooks = None

        if self.is_windowed:
            cache, hooks = self._setup_windowed_cache(input_ids, max_gen_len)

        try:
            gen_kwargs = {
                "attention_mask": attention_mask,
                "max_new_tokens": max_gen_len,
                "num_beams": 1,
                "do_sample": False,
                "temperature": 1.0,
                "pad_token_id": tokenizer.pad_token_id
                or tokenizer.eos_token_id,
            }

            rep_pen = getattr(self.lb, "repetition_penalty", 1.0)
            if rep_pen is not None and rep_pen != 1.0:
                gen_kwargs["repetition_penalty"] = rep_pen

            if self.cache_backend_package == "eager":
                gen_kwargs["output_attentions"] = True

            if cache is not None:
                gen_kwargs["past_key_values"] = cache

            if dataset_name == "samsum":
                newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
                gen_kwargs["eos_token_id"] = [
                    tokenizer.eos_token_id,
                    newline_id,
                ]
                gen_kwargs["min_length"] = context_length + 1

            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)

        finally:
            if hooks is not None:
                hooks.remove()

        pred = tokenizer.decode(
            output[0][context_length:], skip_special_tokens=True
        )

        pred = self._post_process(pred, dataset_name)

        self._cleanup_memory(cache)

        return pred

    def _setup_windowed_cache(self, input_ids: torch.Tensor, max_gen_len: int):
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
            quant_memoize_read=getattr(cfg.cache, "quant_memoize_read", None),
            quant_promotion=getattr(cfg.cache, "quant_promotion", True),
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
        model_max = getattr(getattr(self.model, "config", None),
                            "max_position_embeddings", None)
        if not model_max:
            return
        needed = context_length + max_gen_len
        if needed > model_max and not self._over_context_warned:
            log.warning(
                "Prompt+generation = %d tokens exceeds the model's context "
                "window (%d). Positions beyond the window are out-of-distribution "
                "and scores will be unreliable. Either set longbench.max_length "
                "to truncate, or use a longer-context model (e.g. "
                "Llama-3.1-8B-Instruct, 128K). Suppressing further warnings.",
                needed, model_max,
            )
            self._over_context_warned = True

    @classmethod
    def _should_apply_chat_template(cls, model_name: str, dataset_name: str) -> bool:
        name = model_name.lower()
        is_chat_model = "instruct" in name or "chat" in name
        return is_chat_model and dataset_name not in cls.NO_CHAT_TEMPLATE_DATASETS

    def _warn_if_chat_gate_looks_wrong(self) -> None:
        tok = self.tokenizer
        if tok is None:
            return
        name = self.config.model.name
        name_says_chat = "instruct" in name.lower() or "chat" in name.lower()
        has_template = getattr(tok, "chat_template", None) is not None
        if has_template and not name_says_chat:
            log.warning(
                "model.name=%r contains neither 'instruct' nor 'chat', so the "
                "chat template is DISABLED for every dataset — but this "
                "tokenizer HAS a chat template. If this is an instruct model "
                "under a local path, rename the path (or symlink it) to include "
                "'instruct'; otherwise the 10 templated datasets are being run "
                "with raw prompts and their scores are not comparable.",
                name,
            )
        elif name_says_chat and not has_template:
            log.warning(
                "model.name=%r looks like an instruct model, so the chat "
                "template is enabled — but this tokenizer has none, so "
                "apply_chat_template will fall back to the hard-coded Llama-3 "
                "template. Check that is the right format for this model.",
                name,
            )

    @staticmethod
    def _post_process(pred: str, dataset_name: str) -> str:
        if dataset_name == "samsum":
            pred = pred.split("\n")[0].strip()
        return pred

    def _cleanup_memory(self, cache=None) -> None:
        if cache is not None:
            del cache
        aggressive = getattr(self.lb, "aggressive_cache_clear", False)
        if aggressive and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _write_meta(
        self,
        dataset_name: str,
        num_examples: int,
        max_gen_len: int,
        run_start: float,
        run_end: float,
        eps: float,
        output_dir: Path,
    ) -> None:
        cfg = self.config
        env = capture_environment()

        budget = cfg.cache.cache_budget
        compression_ratio = round(1.0 - budget, 2) if budget else None

        lws = cfg.cache.local_window_size
        if isinstance(lws, float) and budget:
            import math
            max_len = getattr(self.lb, "max_length", None)
            if not max_len:
                max_len = getattr(
                    getattr(self.model, "config", None),
                    "max_position_embeddings", 8192,
                )
            budget_tokens = int(budget * (max_len + max_gen_len))
            raw = lws * budget_tokens
            ceiled = math.ceil(raw)
            remainder = ceiled % cfg.cache.window_size
            if remainder:
                ceiled += cfg.cache.window_size - remainder
            lws_resolved = ceiled
        elif isinstance(lws, int):
            lws_resolved = lws
        else:
            lws_resolved = None

        meta = {
            "dataset": dataset_name,
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
            "rerotate_on_evict": getattr(cfg.cache, "rerotate_on_evict", False),
            "first_eviction_step": getattr(cfg.cache, "first_eviction_step", FIRST_EVICTION_STEP_DEFAULT),
            "quant_ratio": getattr(cfg.cache, "quant_ratio", 0.0),
            "quant_memoize_read": getattr(cfg.cache, "quant_memoize_read", None),
            "local_window_size": lws,
            "local_window_size_resolved_at_max_length": lws_resolved,
            "track_scores": False,
            "attn_implementation": cfg.model.attn_implementation,
            "dtype": cfg.model.dtype,
            "max_length": getattr(self.lb, "max_length", 7500),
            "max_gen_len": max_gen_len,
            "num_samples_requested": getattr(self.lb, "num_samples", "max"),
            "seed": cfg.run.seed,
            **self._vendored_shas,
            **env,
            "run_started_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_start)
            ),
            "run_finished_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_end)
            ),
            "examples_per_second": round(eps, 4),
        }

        meta_path = output_dir / f"{dataset_name}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    def _warn_on_cache_window_disagreement(self) -> None:
        cfg = self.config
        pairs = [
            ("window_size", cfg.cache.window_size, cfg.window.window_size),
            ("num_sink_tokens", cfg.cache.num_sink_tokens, cfg.window.num_sink_tokens),
            ("local_window_size", cfg.cache.local_window_size, cfg.window.local_window_size),
        ]
        for name, cache_val, window_val in pairs:
            if cache_val != window_val:
                log.warning(
                    "cfg.cache.%s=%r != cfg.window.%s=%r — LongBench uses "
                    "cfg.cache.* for the runtime cache. Update cfg.cache.%s "
                    "if you meant the cfg.window.* value.",
                    name, cache_val, name, window_val, name,
                )

    def _get_tokenizer_sha(self) -> str:
        if self.tokenizer is None:
            return "unknown"
        try:
            from utils.hashing import sha256_tokenizer
            return sha256_tokenizer(self.tokenizer)
        except Exception:
            return "unknown"
