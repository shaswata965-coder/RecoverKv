"""GSM8K evaluation runner — chain-of-thought, greedy, one runner for all budgets.

Mirrors ``RulerRunner``: same cache-backend routing, same chat-template protocol,
same meta sidecar discipline. GSM8K-specific parts are the stop strings (so a
512-token budget costs ~150 tokens of wall-clock) and the *compression
diagnostic* described below.

The compression diagnostic
--------------------------
GSM8K prompts are short — roughly 120-180 tokens. A cache budget is sized against
``prefill_len + max_new_tokens``, so on a 150-token prompt with a 512-token
generation budget, ``cache_budget=0.80`` resolves to ~530 retained tokens: more
than the sequence ever reaches. **The cache never evicts, and the run is
byte-identical to the full-cache baseline.** That is not a bug, but silently
reporting it as "accuracy at 80% budget" would be. So for every run this runner
resolves the policy against the real per-example prefill length and records:

* ``n_examples_no_eviction`` — examples where the budget exceeded the sequence
* ``mean_retained_tokens`` / ``mean_sequence_tokens``

and logs a loud warning when the first exceeds 10% of the run. Read it before
you read the accuracy.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from data.gsm8k_loader import load_gsm8k_dataset, read_manifest
from modules.evaluation.gsm8k_dataset import STOP_STRINGS
from utils.config import FIRST_EVICTION_STEP_DEFAULT, log_operating_point
from utils.generation import unmappable_token_ids
from utils.model_loading import describe_model_load
from utils.prompting import encode_prompt
from utils.env_capture import capture_environment
from utils.logger import get_logger

log = get_logger(__name__)


class GSM8KRunner:
    """End-to-end GSM8K prediction runner (mode ``gsm8k``)."""

    def __init__(self, config) -> None:
        self._assert_tracking_off(config)
        self.config = config

        self.gs = getattr(config, "gsm8k", None)
        if self.gs is None:
            raise ValueError("Config must have a 'gsm8k' section for gsm8k mode.")

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
        self._stop_strings_supported: Optional[bool] = None

        # Compression diagnostic accumulators
        self._n_no_eviction = 0
        self._retained_tokens: List[int] = []
        self._sequence_tokens: List[int] = []

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_tracking_off(config) -> None:
        track = getattr(getattr(config, "telemetry", None), "track_scores", False)
        if track:
            raise ValueError(
                "track_scores must be False for GSM8K runs — telemetry buffers "
                "grow with run length and distort throughput. Set "
                "telemetry.track_scores: false."
            )

    def _load_model_and_tokenizer(self) -> Tuple:
        """Load model and tokenizer (lazy, called once).

        Delegates to :func:`utils.model_loading.load_model_and_tokenizer`, shared
        with the LongBench and RULER runners. It owns the prompt-format warnings
        this method used to duplicate, the context/RoPE overrides, and the reset
        of any sampling defaults the checkpoint ships (Qwen2.5's
        repetition_penalty=1.05 applies under greedy decoding; Llama-3.1 and
        Mistral-v0.2 ship none) — so the three model columns decode identically.
        """
        from utils.model_loading import load_model_and_tokenizer

        return load_model_and_tokenizer(
            self.config,
            raw_prompt_hint="GSM8K templates every example, so this affects all of them.",
            is_windowed=self.is_windowed,
        )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self) -> None:
        if self.is_windowed:
            from utils.cache_factory import assert_transformers_version_supported

            assert_transformers_version_supported()

        data_dir = getattr(self.gs, "data_dir", "data/gsm8k_cot")
        manifest = read_manifest(data_dir)

        examples = load_gsm8k_dataset(
            data_dir,
            num_samples=getattr(self.gs, "num_samples", "max"),
            shard=getattr(self.gs, "shard", 0),
            num_shards=getattr(self.gs, "num_shards", 1),
        )
        if not examples:
            log.error("No GSM8K examples loaded from %s", data_dir)
            return

        self.model, self.tokenizer = self._load_model_and_tokenizer()

        output_dir = Path(getattr(self.gs, "output_dir", "outputs/gsm8k/run"))
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "predictions.jsonl"

        log.info(
            "GSM8K run: %d examples, budget=%s, windowed=%s, out=%s",
            len(examples),
            self.config.cache.cache_budget,
            self.is_windowed,
            output_dir,
        )
        log_operating_point(self.config, self.is_windowed)

        skip_oom = getattr(self.gs, "skip_oom", False)
        run_start = time.time()
        n_examples = n_oom = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for i, ex in enumerate(examples):
                try:
                    pred, stats = self._predict(ex)
                except torch.cuda.OutOfMemoryError:
                    if not skip_oom:
                        raise
                    log.warning("example %d: OOM — recording null pred", i)
                    pred, stats = None, {}
                    n_oom += 1
                    torch.cuda.empty_cache()
                    gc.collect()

                record = {
                    "id": i,
                    "pred": pred,
                    "answer": ex["answer"],
                    "task": ex["task"],
                    "max_new_tokens": ex["max_new_tokens"],
                    **stats,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_examples += 1

                if (i + 1) % 50 == 0:
                    log.info("  %d/%d examples done", i + 1, len(examples))

        run_end = time.time()
        elapsed = run_end - run_start
        eps = n_examples / elapsed if elapsed > 0 else 0.0

        log.info(
            "GSM8K: %d examples, %d OOM, %.1fs (%.2f ex/s)",
            n_examples, n_oom, elapsed, eps,
        )

        self._report_compression_diagnostic(n_examples)
        self._write_meta(
            manifest, n_examples, n_oom, run_start, run_end, eps, output_dir
        )
        log.info("Predictions written to %s", out_path)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def _predict(self, ex: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate one GSM8K answer. Returns ``(pred, per_example_stats)``."""
        model, tokenizer = self.model, self.tokenizer
        override = getattr(self.gs, "max_new_tokens", None)
        max_gen_len = int(override) if override else int(ex["max_new_tokens"])

        # 1. Chat template over context + question, then answer_prefix.
        #    Identical protocol to RulerRunner / kvpress pipeline.
        #
        #    Applied only when the tokenizer ships a template, and with NO
        #    fallback if rendering fails. The `except Exception` here used to
        #    substitute a hand-written LLaMA-3 wrapper, whose <|begin_of_text|> /
        #    <|start_header_id|> markers are ordinary text to any other tokenizer
        #    — on Mistral that is a silently mis-formatted prompt that still
        #    produces plausible-looking (and worthless) scores.
        messages = [{"role": "user", "content": ex["context"] + ex["question"]}]
        if getattr(tokenizer, "chat_template", None) is not None:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = ex["context"] + ex["question"]
        prompt = prompt + ex.get("answer_prefix", "")

        # encode_prompt, not tokenizer(): the templated string above already
        # carries the BOS, and tokenizer()'s add_special_tokens default would
        # add a second — shifting every position and spending a protected sink
        # slot on a duplicate.
        inputs = encode_prompt(tokenizer, prompt, truncation=False,
                               return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        # Pass the mask explicitly: pad_token == eos_token for Llama, so
        # transformers cannot infer it and warns on every example. At batch
        # size 1 the mask is all-ones and the result is unchanged, but leaving
        # it implicit buries real warnings under thousands of spurious ones.
        attention_mask = inputs.attention_mask.to(model.device)
        context_length = input_ids.shape[-1]

        # 2. Cache
        cache = hooks = None
        resolved = None
        if self.is_windowed:
            cache, hooks, resolved = self._setup_windowed_cache(
                input_ids, max_gen_len
            )

        # 3. Generate — pure greedy, matching the LongBench/RULER protocol.
        try:
            gen_kwargs: Dict[str, Any] = {
                "attention_mask": attention_mask,
                "max_new_tokens": max_gen_len,
                "num_beams": 1,
                "do_sample": False,
                "temperature": 1.0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            # Ids the model can emit but the tokenizer cannot map (Qwen2.5 pads
            # its vocab to 152064 against a 151665-id tokenizer). They decode to
            # '' AND make stop_strings crash — StopStringCriteria sizes its
            # embedding by len(tokenizer). None for Llama/Mistral, which have no
            # gap, so their generation is untouched.
            suppress = unmappable_token_ids(model, tokenizer)
            if suppress:
                gen_kwargs["suppress_tokens"] = suppress
            if self._supports_stop_strings():
                gen_kwargs["stop_strings"] = STOP_STRINGS
                gen_kwargs["tokenizer"] = tokenizer
            if self.cache_backend_package == "eager":
                gen_kwargs["output_attentions"] = True
            if cache is not None:
                gen_kwargs["past_key_values"] = cache

            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)
        finally:
            if hooks is not None:
                hooks.remove()

        n_generated = int(output.shape[-1] - context_length)
        pred = tokenizer.decode(
            output[0][context_length:], skip_special_tokens=True
        )

        # 4. Per-example stats — these are what make a run auditable after the
        #    fact without re-running it.
        seq_tokens = context_length + n_generated
        stats: Dict[str, Any] = {
            "prompt_tokens": context_length,
            "gen_tokens": n_generated,
            "hit_token_cap": n_generated >= max_gen_len,
        }
        if resolved is not None:
            retained = min(resolved.total_budget_tokens, seq_tokens)
            evicted = resolved.total_budget_tokens < seq_tokens
            stats["budget_tokens"] = resolved.total_budget_tokens
            stats["evicted"] = evicted
            self._retained_tokens.append(retained)
            self._sequence_tokens.append(seq_tokens)
            if not evicted:
                self._n_no_eviction += 1

        self._cleanup_memory(cache)
        return pred, stats

    def _supports_stop_strings(self) -> bool:
        """Whether this transformers version accepts ``generate(stop_strings=...)``."""
        if self._stop_strings_supported is None:
            try:
                from transformers import GenerationConfig

                self._stop_strings_supported = hasattr(
                    GenerationConfig(), "stop_strings"
                )
            except Exception:
                self._stop_strings_supported = False
            if not self._stop_strings_supported:
                log.warning(
                    "transformers does not support generate(stop_strings=...); "
                    "generations will run to max_new_tokens and may contain "
                    "hallucinated follow-up problems. The scorer trims these, "
                    "but wall-clock will be several times higher."
                )
        return self._stop_strings_supported

    def _setup_windowed_cache(self, input_ids: torch.Tensor, max_gen_len: int):
        """Create windowed cache + hooks. Returns ``(cache, hooks, resolved)``."""
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
                "requires a rotary embedding module."
            )

        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        kv_dtype = dtypes.get(cfg.model.dtype, torch.float16)
        prefill_len = int(input_ids.shape[-1])

        resolved = cache_config.resolve(
            prefill_len=prefill_len,
            model_config=model.config,
            kv_dtype=kv_dtype,
            max_tokens=max_gen_len,
        )

        cache = self.WindowedCache(
            config=cache_config,
            prefill_len=prefill_len,
            model_config=model.config,
            kv_dtype=kv_dtype,
            rope_module=rope,
            num_layers=model.config.num_hidden_layers,
            max_tokens=max_gen_len,
        )
        hooks = self.install_score_hooks(model, cache, cache_config)
        return cache, hooks, resolved

    def _cleanup_memory(self, cache=None) -> None:
        if cache is not None:
            del cache
        if getattr(self.gs, "aggressive_cache_clear", False) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # ------------------------------------------------------------------
    # diagnostics + meta
    # ------------------------------------------------------------------

    def _compression_summary(self, n_examples: int) -> Dict[str, Any]:
        if not self._sequence_tokens:
            return {
                "compression_active": bool(self.is_windowed),
                "n_examples_no_eviction": None,
            }
        n = len(self._sequence_tokens)
        return {
            "compression_active": self._n_no_eviction < n,
            "n_examples_no_eviction": self._n_no_eviction,
            "pct_examples_no_eviction": round(100.0 * self._n_no_eviction / n, 2),
            "mean_sequence_tokens": round(sum(self._sequence_tokens) / n, 1),
            "mean_retained_tokens": round(sum(self._retained_tokens) / n, 1),
            "effective_retention": round(
                sum(self._retained_tokens) / max(sum(self._sequence_tokens), 1), 4
            ),
        }

    def _report_compression_diagnostic(self, n_examples: int) -> None:
        """Print, and warn about, how much compression actually happened."""
        if not self.is_windowed:
            log.info("Full-cache baseline: no compression applied.")
            return

        s = self._compression_summary(n_examples)
        if s.get("n_examples_no_eviction") is None:
            return

        log.info(
            "compression diagnostic: mean_seq=%.1f tok, mean_retained=%.1f tok, "
            "effective_retention=%.3f, no-eviction on %d/%d examples (%.1f%%)",
            s["mean_sequence_tokens"], s["mean_retained_tokens"],
            s["effective_retention"], s["n_examples_no_eviction"],
            len(self._sequence_tokens), s["pct_examples_no_eviction"],
        )

        if s["pct_examples_no_eviction"] > 10.0:
            log.warning(
                "cache_budget=%s never bound on %.1f%% of examples: the budget "
                "(%.0f tokens on average) exceeded the whole sequence (%.0f "
                "tokens). Those examples are byte-identical to the full-cache "
                "baseline, so this run's accuracy is NOT a measurement of "
                "compression at that budget. GSM8K prompts are ~150 tokens; a "
                "budget only bites once it falls below prefill+generation. "
                "Lower cache_budget, or report this run as a control.",
                self.config.cache.cache_budget,
                s["pct_examples_no_eviction"],
                s["mean_retained_tokens"], s["mean_sequence_tokens"],
            )

    def _write_meta(
        self,
        manifest: Dict[str, Any],
        n_examples: int,
        n_oom: int,
        run_start: float,
        run_end: float,
        eps: float,
        output_dir: Path,
    ) -> None:
        cfg = self.config
        budget = cfg.cache.cache_budget

        meta = {
            "task": "gsm8k",
            "run_name": output_dir.name,
            "num_examples": n_examples,
            "num_oom": n_oom,
            # Dataset identity — the scorer refuses to compare runs whose
            # dataset_sha disagree.
            "dataset_dir": str(getattr(self.gs, "data_dir", "data/gsm8k_cot")),
            "dataset_sha": manifest["dataset_sha"],
            "prompt_style": manifest["prompt_style"],
            "dataset_max_new_tokens": manifest["max_new_tokens"],
            "max_new_tokens_override": getattr(self.gs, "max_new_tokens", None),
            "shard": getattr(self.gs, "shard", 0),
            "num_shards": getattr(self.gs, "num_shards", 1),
            "num_samples_requested": getattr(self.gs, "num_samples", "max"),
            # Model / cache
            "model_name": cfg.model.name,
            "model_revision": getattr(cfg.model, "revision", None),
            "tokenizer_sha": self._get_tokenizer_sha(),
            "dtype": cfg.model.dtype,
            "attn_implementation": cfg.model.attn_implementation,
            # What the load path did to the checkpoint's own config, and the
            # greedy pinning — without these a YaRN-scaled run is indistinguishable
            # from an un-scaled one in the result files.
            **describe_model_load(cfg),
            "model_max_position_embeddings_effective": getattr(
                getattr(self.model, "config", None), "max_position_embeddings", None
            ),
            "model_rope_scaling_effective": getattr(
                getattr(self.model, "config", None), "rope_scaling", None
            ),
            "cache_type": "windowed" if self.is_windowed else "full_cache",
            "cache_backend_package": self.cache_backend_package,
            "cache_budget": budget,
            "compression_ratio": round(1.0 - budget, 4) if budget else None,
            "window_size": cfg.cache.window_size,
            "num_sink_tokens": cfg.cache.num_sink_tokens,
            "local_window_size": cfg.cache.local_window_size,
            "quant_ratio": getattr(cfg.cache, "quant_ratio", 0.0),
            "first_eviction_step": getattr(cfg.cache, "first_eviction_step", FIRST_EVICTION_STEP_DEFAULT),
            "rerotate_on_evict": getattr(cfg.cache, "rerotate_on_evict", False),
            "stop_strings": STOP_STRINGS if self._supports_stop_strings() else None,
            "decoding": "greedy",
            "seed": cfg.run.seed,
            **self._compression_summary(n_examples),
            **capture_environment(),
            "run_started_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_start)
            ),
            "run_finished_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_end)
            ),
            "examples_per_second": round(eps, 4),
        }
        (output_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

    def _get_tokenizer_sha(self) -> str:
        if self.tokenizer is None:
            return "unknown"
        try:
            from utils.hashing import sha256_tokenizer

            return sha256_tokenizer(self.tokenizer)
        except Exception:
            return "unknown"
