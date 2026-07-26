"""LongBench evaluation runner — one runner, both backends, all 16 datasets.

Follows DefensiveKV's exact protocol:
- LongBench v1 (THUDM/LongBench), 16 English datasets
- Llama-3.1-8B-Instruct (128K) fp16, full context (no pre-truncation)
- Greedy decoding, per-dataset max gen length
- Optional middle truncation (longbench.max_length) for short-context models
- Output jsonl schema matches THUDM/LongBench/pred.py exactly

Backend routing via ``utils/cache_factory.py``.
"""

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
from utils.env_capture import capture_environment
from utils.hashing import sha256_file
from utils.logger import get_logger

log = get_logger(__name__)


class LongBenchRunner:
    """End-to-end LongBench prediction runner.

    One runner handles all 16 datasets and both cache backends
    (flash_attn / eager), routed via the factory in ``utils/cache_factory.py``.
    """

    # Few-shot in-context-learning datasets whose prompt body IS a series of
    # worked examples ("input\nanswer\n\ninput\nanswer\n...").  THUDM/LongBench
    # pred.py (and DefensiveKV) deliberately do NOT wrap these in the chat
    # template — doing so flips an instruct model out of "continue the format"
    # mode into chat-assistant mode, so it emits a meta-preamble ("Here are the
    # summaries:", "Here is the completed code:") instead of imitating the
    # examples, destroying the score on exact-/edit-match metrics.
    #   THUDM/LongBench/pred.py:
    #     if dataset not in ["trec","triviaqa","samsum","lsht","lcc","repobench-p"]:
    #         prompt = build_chat(...)
    NO_CHAT_TEMPLATE_DATASETS = frozenset(
        {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
    )

    def __init__(self, config) -> None:
        self._assert_tracking_off(config)
        self.config = config

        # Extract longbench-specific config
        self.lb = getattr(config, "longbench", None)
        if self.lb is None:
            raise ValueError(
                "Config must have a 'longbench' section for LongBench mode."
            )

        # Determine cache type
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

        # Load vendored configs (DO NOT reimplement)
        configs_dir = Path("data/longbench_configs")
        with open(configs_dir / "dataset2prompt.json", "r", encoding="utf-8") as f:
            self.dataset2prompt = json.load(f)
        with open(configs_dir / "dataset2maxlen.json", "r", encoding="utf-8") as f:
            self.dataset2maxlen = json.load(f)

        # Compute SHA-256 of vendored files for reproducibility
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
        # Auto-fit bookkeeping (see _resolve_max_length): chat-template overhead
        # is per-dataset because NO_CHAT_TEMPLATE_DATASETS skip the template.
        self._chat_overhead_cache: dict[str, int] = {}
        self._auto_fit_logged: set[str] = set()
        self._prompt_preview_logged: set[str] = set()

    @staticmethod
    def _compute_metrics_sha() -> str:
        """SHA-256 of the vendored metrics module."""
        metrics_path = Path("modules/evaluation/longbench_metrics.py")
        if metrics_path.exists():
            return sha256_file(metrics_path)
        return "unknown"

    @staticmethod
    def _assert_tracking_off(config) -> None:
        """Guard: track_scores must be False for LongBench runs.

        Telemetry buffers grow linearly with
        ``num_layers × H_q × num_windows × num_steps``; on long-context tasks
        (~7.5k tokens prompt, up to 512 tokens generation), that's gigabytes
        of CPU-resident tensors per example.  Distorts throughput numbers and
        risks OOM.
        """
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

        if tokenizer.chat_template is None:
            log.warning(
                "Tokenizer for %s ships no chat template — every dataset will be "
                "prompted raw. Correct for a base model; on an instruct model it "
                "costs most of the score on the generative tasks.",
                cfg.model.name,
            )

        # Mistral-7B-Instruct-v0.1 sets sliding_window=4096 (v0.2/v0.3 set null).
        # Both the FA2 kernel and the sdpa mask builder band attention by
        # *cache-slot* distance, which stops matching real token distance once
        # the cache is compacted. Inert while the compacted cache is shorter than
        # the window, but not something to discover from a score table.
        sw = getattr(model.config, "sliding_window", None)
        if sw and self.is_windowed:
            log.warning(
                "%s sets sliding_window=%s. The windowed cache keeps survivors at "
                "their original positions but stores them compacted, so a banded "
                "attention mask no longer corresponds to real token distance. "
                "Prefer a model with sliding_window=null (e.g. "
                "Mistral-7B-Instruct-v0.2) for comparable numbers.",
                cfg.model.name, sw,
            )

        return model, tokenizer

    def run(self) -> None:
        """Run predictions on all configured datasets."""
        # Fail fast on an unsupported transformers version: the windowed cache's
        # RoPE handling assumes monotonic cache_position (transformers <= 4.47).
        if self.is_windowed:
            from utils.cache_factory import assert_transformers_version_supported

            assert_transformers_version_supported()

        # Lazy-load model
        self.model, self.tokenizer = self._load_model_and_tokenizer()

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

        for dataset_name in datasets:
            jsonl_path = output_dir / f"{dataset_name}.jsonl"

            # Resume support: skip if output already exists with data
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
        """Run predictions on a single dataset."""
        log.info("=== Dataset: %s ===", name)

        use_e = getattr(self.lb, "use_e_variants", False)
        examples = load_longbench_dataset(name, use_e_variant=use_e)
        examples_list = list(examples)

        # Cap to num_samples per dataset. "max" (default) keeps the full split.
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

                # Output schema matches THUDM/LongBench/pred.py exactly
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

        # Write metadata sidecar
        self._write_meta(name, n_examples, max_gen_len, run_start, run_end, eps, output_dir)

    def _predict(
        self,
        ex: Dict[str, Any],
        prompt_template: str,
        max_gen_len: int,
        dataset_name: str,
    ) -> str:
        """Generate a prediction for a single example.

        Follows THUDM/LongBench/pred.py + kvpress protocol exactly.
        """
        model = self.model
        tokenizer = self.tokenizer

        # 1. Format prompt from template
        prompt = prompt_template.format(
            context=ex["context"], input=ex.get("input", "")
        )

        # 2. Tokenize and (optionally) middle-truncate.
        #    A positive longbench.max_length reproduces official THUDM/LongBench
        #    middle-truncation. null auto-fits to the model's context window so a
        #    prompt never scores on out-of-distribution RoPE positions; 0 opts out.
        #    See _resolve_max_length.
        max_length = self._resolve_max_length(tokenizer, dataset_name, max_gen_len)
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if max_length and max_length > 0 and len(tokenized) > max_length:
            if not getattr(self.lb, "max_length", None):
                self._note_auto_fit(dataset_name, len(tokenized), max_length)
            half = max_length // 2
            # Middle truncation — byte-for-byte identical to THUDM/LongBench
            # pred.py and DefensiveKV: decode the head and tail halves
            # SEPARATELY and string-concatenate. (Concatenating token ids and
            # decoding once produces a different prompt at the head/tail seam,
            # so it must not be used if results are to match the published
            # numbers.) The prompt is re-tokenized below regardless.
            prompt = tokenizer.decode(
                tokenized[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized[-half:], skip_special_tokens=True)

        # 3. Apply the model's own chat template.
        #    Skipped for the few-shot ICL datasets (matches THUDM/LongBench +
        #    DefensiveKV) — wrapping their worked-example prompts in a chat
        #    turn breaks few-shot continuation. See NO_CHAT_TEMPLATE_DATASETS.
        templated = self._should_apply_chat_template(tokenizer, dataset_name)
        if templated:
            messages = [{"role": "user", "content": prompt}]
            # No fallback: a template that fails to render must be a hard error.
            # The old `except Exception` here substituted a hand-written LLaMA-3
            # template, whose <|begin_of_text|> / <|start_header_id|> markers are
            # ordinary text to any other tokenizer — a silently mis-formatted
            # prompt that still produces plausible-looking (and worthless) scores.
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        self._log_prompt_preview(dataset_name, prompt, templated)

        # 4. Tokenize final prompt.
        #    add_special_tokens is OFF once the chat template has run: every
        #    template emits its own BOS (Mistral "<s> [INST]", LLaMA-3
        #    "<|begin_of_text|>"), so letting the tokenizer prepend another
        #    yields a doubled BOS the model never saw in training. THUDM's
        #    pred.py builds the LLaMA-2 wrapper as a bare "[INST]...[/INST]"
        #    string and relies on the tokenizer for the single BOS; kvpress
        #    (and therefore DefensiveKV) uses the template and passes
        #    add_special_tokens=False. Both end up with exactly one.
        inputs = tokenizer(
            prompt, truncation=False, return_tensors="pt",
            add_special_tokens=not templated,
        )
        input_ids = inputs.input_ids.to(model.device)
        context_length = input_ids.shape[-1]

        # Guard: a prompt longer than the model's positional range yields
        # out-of-distribution RoPE positions (garbage / errors). Surface it
        # loudly instead of silently scoring noise — almost always means
        # truncation was disabled (max_length null) on a SHORT-context model.
        # Llama-3.1-8B (128K) fits every LongBench prompt; the original
        # Llama-3-8B (8K) does not.
        self._warn_if_over_context(context_length, max_gen_len)

        # 5. Set up cache
        cache = None
        hooks = None

        if self.is_windowed:
            cache, hooks = self._setup_windowed_cache(input_ids, max_gen_len)

        # 6. Generate
        try:
            gen_kwargs = {
                "max_new_tokens": max_gen_len,
                "num_beams": 1,
                "do_sample": False,
                "temperature": 1.0,
                "pad_token_id": tokenizer.pad_token_id
                or tokenizer.eos_token_id,
            }

            # THUDM/LongBench + DefensiveKV use PURE greedy with NO repetition
            # penalty. Only pass one if the user explicitly opted in to a
            # non-1.0 value; otherwise the default must stay absent so results
            # match the published protocol exactly.
            rep_pen = getattr(self.lb, "repetition_penalty", 1.0)
            if rep_pen is not None and rep_pen != 1.0:
                gen_kwargs["repetition_penalty"] = rep_pen

            # output_attentions only for eager backend
            if self.cache_backend_package == "eager":
                gen_kwargs["output_attentions"] = True

            if cache is not None:
                gen_kwargs["past_key_values"] = cache

            # samsum special handling: stop at newline
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
            # 7. Clean up hooks (no leakage between examples)
            if hooks is not None:
                hooks.remove()

        # 8. Decode only new tokens
        pred = tokenizer.decode(
            output[0][context_length:], skip_special_tokens=True
        )

        # 9. Post-processing (dataset-specific, matches THUDM pred.py)
        pred = self._post_process(pred, dataset_name)

        # 10. Memory hygiene
        self._cleanup_memory(cache)

        return pred

    def _setup_windowed_cache(self, input_ids: torch.Tensor, max_gen_len: int):
        """Create windowed cache and install hooks."""
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
        )

        # Get RoPE module
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

    # Slack reserved on top of generation + chat template when auto-fitting a
    # prompt to the context window. Middle truncation decodes two halves and
    # re-encodes the concatenation, so the final token count drifts a few tokens
    # from the requested length; this absorbs that drift.
    _AUTO_FIT_SAFETY_MARGIN = 64

    def _model_context_window(self) -> Optional[int]:
        """The model's usable positional range, or None if it can't be read."""
        model_max = getattr(getattr(self.model, "config", None),
                            "max_position_embeddings", None)
        return int(model_max) if model_max else None

    def _chat_template_overhead(self, tokenizer, dataset_name: str) -> int:
        """Tokens the chat template wraps around the prompt body (0 if unused).

        Truncation happens *before* the chat template is applied, so its wrapper
        tokens have to be reserved up front or an auto-fitted prompt can still
        cross the context window after templating.
        """
        if not self._should_apply_chat_template(tokenizer, dataset_name):
            return 0
        cached = self._chat_overhead_cache.get(dataset_name)
        if cached is not None:
            return cached
        try:
            wrapped = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False,
                add_generation_prompt=True,
            )
            # add_special_tokens=False mirrors how _predict tokenizes a
            # templated prompt; otherwise the reserve is off by the BOS.
            overhead = len(
                tokenizer(
                    wrapped, truncation=False, add_special_tokens=False
                ).input_ids
            )
        except Exception:
            # Same conservative fallback as the manual template in _predict.
            overhead = 64
        self._chat_overhead_cache[dataset_name] = overhead
        return overhead

    def _resolve_max_length(
        self, tokenizer, dataset_name: str, max_gen_len: int
    ) -> Optional[int]:
        """Resolve ``longbench.max_length`` against the model's real context window.

        Returns the token budget for the *pre-template* prompt, or None for no
        truncation. See LongBenchConfig.max_length for the three modes; the point
        of the auto mode is that a prompt never silently runs past
        ``max_position_embeddings``, where RoPE positions are out of distribution
        and the resulting scores are noise rather than a measurement.
        """
        configured = getattr(self.lb, "max_length", None)

        if configured is not None and configured > 0:
            return int(configured)          # explicit — reproduces published protocol

        if configured == 0:                 # explicit opt-out
            if not self._over_context_warned:
                model_max = self._model_context_window()
                log.warning(
                    "longbench.max_length=0 disables truncation entirely. Prompts "
                    "longer than the model's context window (%s) will use "
                    "out-of-distribution RoPE positions and their scores will be "
                    "unreliable. Set max_length to null to auto-fit instead.",
                    model_max,
                )
                self._over_context_warned = True
            return None

        # auto (max_length: null)
        model_max = self._model_context_window()
        if not model_max:
            return None                     # can't read a window — leave untouched
        reserve = (max_gen_len
                   + self._chat_template_overhead(tokenizer, dataset_name)
                   + self._AUTO_FIT_SAFETY_MARGIN)
        return max(model_max - reserve, 1)

    def _note_auto_fit(self, dataset_name: str, original: int, budget: int) -> None:
        """Log once per dataset when auto-fit actually truncates something."""
        if dataset_name in self._auto_fit_logged:
            return
        self._auto_fit_logged.add(dataset_name)
        log.info(
            "%s: middle-truncating prompts to %d tokens to fit the model's "
            "context window (%s). First hit was %d tokens. Set "
            "longbench.max_length explicitly to pin a different budget, or 0 to "
            "disable truncation (scores then become unreliable past the window).",
            dataset_name, budget, self._model_context_window(), original,
        )

    def _warn_if_over_context(self, context_length: int, max_gen_len: int) -> None:
        """Last-resort net: prompt + generation still past the context window.

        With auto-fit this should not fire; if it does, the reserve above was too
        small for this tokenizer/template, so report it at ERROR — the example's
        score is not trustworthy and that must not be silent.
        """
        model_max = self._model_context_window()
        if not model_max:
            return
        needed = context_length + max_gen_len
        if needed > model_max and not self._over_context_warned:
            log.error(
                "Prompt+generation = %d tokens exceeds the model's context "
                "window (%d). Positions beyond the window are out-of-distribution "
                "and these scores are unreliable. Set longbench.max_length to a "
                "value below %d, or use a longer-context model. Suppressing "
                "further warnings.",
                needed, model_max, model_max - max_gen_len,
            )
            self._over_context_warned = True

    @classmethod
    def _should_apply_chat_template(cls, tokenizer, dataset_name: str) -> bool:
        """Whether to wrap the prompt in the model's chat template.

        Decided by the TOKENIZER — a model is a chat model iff it ships a chat
        template — not by substring-matching the model name. The name test this
        replaced ("instruct" in name or "chat" in name) reads whatever string
        `model.name` happens to hold, which on a cluster is a checkout directory:
        `.../mistral-7b-v0.2` matches neither substring, so the template was
        silently dropped and an instruct model was prompted as a base LM. That
        is invisible in the output files and costs most of the score on the
        generative datasets. kvpress (and therefore DefensiveKV) gates on
        `tokenizer.chat_template is None` for exactly this reason.

        Still False for the few-shot ICL datasets (NO_CHAT_TEMPLATE_DATASETS),
        which must stay raw so the model continues the worked-example format
        instead of switching into chat-assistant mode — matches THUDM/LongBench
        + DefensiveKV, which blanks `tokenizer.chat_template` on those tasks.
        """
        if dataset_name in cls.NO_CHAT_TEMPLATE_DATASETS:
            return False
        return getattr(tokenizer, "chat_template", None) is not None

    def _log_prompt_preview(
        self, dataset_name: str, prompt: str, templated: bool
    ) -> None:
        """Log the head/tail of the first prompt of each dataset, once.

        The whole class of bug this guards against (wrong template, no template,
        doubled BOS) is invisible in the prediction files and only shows up as a
        depressed score, so the actual string fed to the model is worth one log
        line per dataset.
        """
        if dataset_name in self._prompt_preview_logged:
            return
        self._prompt_preview_logged.add(dataset_name)
        head = prompt[:120].replace("\n", "\\n")
        tail = prompt[-80:].replace("\n", "\\n")
        log.info(
            "%s: chat_template=%s | prompt head %r ... tail %r",
            dataset_name, "applied" if templated else "SKIPPED", head, tail,
        )

    @staticmethod
    def _post_process(pred: str, dataset_name: str) -> str:
        """Dataset-specific post-processing (matches THUDM pred.py).

        - samsum: first line only
        - code datasets: preserve whitespace
        - all others: return as-is
        """
        if dataset_name == "samsum":
            # Take first line only (prevents illegal repeating output)
            pred = pred.split("\n")[0].strip()
        # Code datasets: preserve all whitespace (no stripping)
        # All others: return as-is (metric functions handle normalization)
        return pred

    def _cleanup_memory(self, cache=None) -> None:
        """Memory hygiene between examples."""
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
        """Write per-dataset metadata sidecar JSON."""
        cfg = self.config
        env = capture_environment()

        budget = cfg.cache.cache_budget
        compression_ratio = round(1.0 - budget, 2) if budget else None

        # Resolve local_window_size if possible
        lws = cfg.cache.local_window_size
        if isinstance(lws, float) and budget:
            import math
            # Mirror WindowedCacheConfig.resolve: a float local_window_size is a
            # fraction of the cache BUDGET (not the full context), resolved here
            # at the max_length upper bound (the runtime resolves against each
            # example's own prefill length).
            # Under auto-fit (max_length null) the upper bound is the resolved
            # per-dataset budget; fall back to the model's context window if that
            # can't be read. The runtime policy resolves local_window_size against
            # each example's real prefill length regardless.
            max_len = self._resolve_max_length(
                self.tokenizer, dataset_name, max_gen_len
            )
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
            "local_window_size": lws,
            # NOTE: resolved against `max_length` (upper bound), not the
            # per-example truncated prefill; the actual policy resolves
            # against each example's own prefill length at runtime.
            "local_window_size_resolved_at_max_length": lws_resolved,
            "track_scores": False,
            "attn_implementation": cfg.model.attn_implementation,
            "dtype": cfg.model.dtype,
            # Both the knob and what it actually resolved to, so a result file is
            # self-describing under auto-fit (max_length null).
            "max_length": getattr(self.lb, "max_length", 7500),
            "max_length_effective": self._resolve_max_length(
                self.tokenizer, dataset_name, max_gen_len
            ),
            "model_context_window": self._model_context_window(),
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

    def _get_tokenizer_sha(self) -> str:
        """Get tokenizer SHA for reproducibility."""
        if self.tokenizer is None:
            return "unknown"
        try:
            from utils.hashing import sha256_tokenizer
            return sha256_tokenizer(self.tokenizer)
        except Exception:
            return "unknown"
