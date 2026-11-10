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
* decoding ``reference`` (the code: temperature 0.95, top_p 0.95, no top-k, the
  same seed for every document, as vLLM's per-request seed) or ``greedy`` (what
  the paper's text says); batch size 1, flash-attention-2.

The cache is the only thing that differs between arms: ``cache.backend:
dynamic`` is the full-cache baseline, ``windowed`` + ``flash_attn`` is this
repo's method, built exactly as the GSM8K and LongBench runners build it.

Each type writes ``<type>.jsonl`` and ``<type>.meta.json``. The sidecar carries
what makes the number attributable: the read-gate verdict (counters reset per
type), the operating point, the resolved geometry, the dataset pin and hash, the
decoding, and a compression record. The budget is ``cache_budget * (prompt +
max_new_tokens)``, the method's own definition; a document that stops short of
``max_new_tokens`` holds more than ``cache_budget`` of its own sequence, and
``kv_bytes_fraction_upper`` says how much.

Timing fields (``wall_s``, ``decode_tokps_wallclock``) are wall clock inside a
quality run: prefill, eviction compiles and Python overhead included. They are
not a performance measurement; ``mode: perf`` is.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from longgenbench_eval.data import dataset_sha, load_type, type_slug
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
from utils.config import FIRST_EVICTION_STEP_DEFAULT, log_operating_point
from utils.env_capture import capture_environment
from utils.logger import get_logger
from utils.prompting import encode_prompt

log = get_logger(__name__)

#: "reference": the sampling of the runs behind the paper's Table 2;
#: "greedy": what the paper's text says (no run in the reference repo used it).
DECODINGS = ("reference", "greedy")


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

    def run(self) -> None:
        if self.is_windowed:
            from utils.cache_factory import assert_transformers_version_supported

            assert_transformers_version_supported()
        lg = self.lg
        types = self._types()
        dataset_sha(lg.data_dir, lg.setting)   # missing or altered data: fail before a model load

        self.model, self.tokenizer = self._load_model_and_tokenizer()
        output_dir = Path(lg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info("LongGenBench %s: types=%s, documents [%s, %s) of each, decoding=%s, "
                 "windowed=%s, out=%s", lg.setting, types, lg.start, lg.end, lg.decoding,
                 self.is_windowed, output_dir)
        log_operating_point(self.config, self.is_windowed)
        for t in types:
            self._run_type(t, output_dir)
        log.info("LongGenBench run complete. Outputs in %s", output_dir)

    def _run_type(self, type_: str, output_dir: Path) -> None:
        from modules.windowed_cache import flash_decode

        lg = self.lg
        docs = load_type(lg.data_dir, lg.setting, type_, lg.start, lg.end)
        if not docs:
            raise ValueError(f"{type_}: no documents in [{lg.start}, {lg.end})")
        max_new = self._max_new_tokens()
        log.info("=== %s %s: %d documents (ids %d..%d), %d entries each, max_new_tokens=%d ===",
                 lg.setting, type_, len(docs), docs[0]["id"], docs[-1]["id"],
                 docs[0]["number"], max_new)

        # Process-global, cumulative counters: zero them so this type's sidecar
        # describes this type's decode steps and nothing else.
        flash_decode.reset_stats()
        self._reset_type_state()

        slug = type_slug(type_)
        run_start = time.time()
        n_oom = tot_gen = 0
        tot_wall = peak = 0.0
        with open(output_dir / f"{slug}.jsonl", "w", encoding="utf-8") as f:
            for i, d in enumerate(docs):
                try:
                    output, stats = self._generate(d, max_new)
                except torch.cuda.OutOfMemoryError:
                    if not getattr(lg, "skip_oom", False):
                        raise
                    log.warning("%s id %d: OOM, recording a null output", type_, d["id"])
                    output, stats = None, {"oom": True}
                    n_oom += 1
                    gc.collect()
                    torch.cuda.empty_cache()
                cr = (completion_rate(parse_blocks(split_blocks(output), type_), d["number"])
                      if output is not None else 0.0)
                record = {
                    "id": d["id"], "type": type_, "number": d["number"], "prefix": d["prefix"],
                    "checks_once": d["checks_once"], "checks_range": d["checks_range"],
                    "checks_periodic": d["checks_periodic"],
                    "output": output,
                    "n_blocks": len(split_blocks(output)) if output is not None else 0,
                    "word_count": word_count(output) if output is not None else 0,
                    "completion_rate": round(cr, 2),
                    **stats,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                tot_gen += int(stats.get("gen_tokens") or 0)
                tot_wall += float(stats.get("wall_s") or 0.0)
                peak = max(peak, float(stats.get("peak_mem_gib") or 0.0))
                log.info("  %s %d/%d (id %d): %s new tokens in %.0f s, %d words, CR %.1f%%, "
                         "stopped on %s", type_, i + 1, len(docs), d["id"], stats.get("gen_tokens"),
                         stats.get("wall_s") or 0.0, record["word_count"], cr, stats.get("stopped_on"))
        self._report_compression(type_)
        self._write_meta(type_, docs, max_new, n_oom, run_start, time.time(),
                         tot_gen, tot_wall, peak, output_dir)

    # ------------------------------------------------------------------
    # generate one document
    # ------------------------------------------------------------------

    def _generate(self, d: Dict[str, Any], max_new: int) -> Tuple[str, Dict[str, Any]]:
        model, tokenizer = self.model, self.tokenizer
        prompt = d["prompt"]
        inputs = encode_prompt(tokenizer, prompt, truncation=False, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        ctx = int(input_ids.shape[-1])
        self._warn_if_over_context(ctx, max_new)

        cache = hooks = None
        if self.is_windowed:
            cache, hooks = self._setup_windowed_cache(input_ids, max_new)

        eos_ids = self._eos_ids()
        gen_kwargs: Dict[str, Any] = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_new,
            "eos_token_id": eos_ids,
            "pad_token_id": tokenizer.pad_token_id or eos_ids[0],
            "stop_strings": [STOP_STRING],
            "tokenizer": tokenizer,
            "num_beams": 1,
        }
        if self.lg.decoding == "greedy":
            gen_kwargs.update(do_sample=False, temperature=1.0, top_p=1.0)
        else:
            # vLLM's SamplingParams(seed=...) seeds every request with the same
            # seed; so does this, and both arms of a comparison draw from the
            # same stream until the cache makes their distributions differ.
            torch.manual_seed(int(self.lg.sample_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.lg.sample_seed))
            gen_kwargs.update(do_sample=True, temperature=GENERATION_SAMPLING["temperature"],
                              top_p=GENERATION_SAMPLING["top_p"], top_k=GENERATION_SAMPLING["top_k"])
        if self.cache_backend_package == "eager":
            gen_kwargs["output_attentions"] = True
        if cache is not None:
            gen_kwargs["past_key_values"] = cache

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

        new = out[0][ctx:]
        n_gen = int(new.shape[-1])
        text = tokenizer.decode(new, skip_special_tokens=True)
        # vLLM's stop string ends the text BEFORE the string; HF keeps it.
        cut = text.find(STOP_STRING)
        if cut >= 0:
            text = text[:cut]
            stopped_on = "stop_string"
        elif n_gen and int(new[-1]) in eos_ids:
            stopped_on = "eos"
        elif n_gen >= max_new:
            stopped_on = "max_new_tokens"
        else:
            stopped_on = "other"
        output = d["prefix"] + text              # inference.py: input['prefix'] + output.text
        seq = ctx + n_gen
        stats: Dict[str, Any] = {
            "prompt_tokens": ctx,
            "gen_tokens": n_gen,
            "hit_token_cap": n_gen >= max_new,
            "stopped_on": stopped_on,
            "wall_s": round(wall, 3),
            "decode_tokps_wallclock": round(n_gen / wall, 2) if wall > 0 else None,
            "peak_mem_gib": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                             if cuda else None),
            "prompt_sha": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        }
        if cache is not None:
            r = cache.resolved
            cap = int(r.retained_tokens)
            # Keys the fp16 tier (sink + local + top_k_fp windows) can hold. A
            # longer sequence has int2 windows, so the read gate has something to
            # select over; longer than `cap`, keys have been dropped.
            fp_cap = int(r.num_sink_tokens + r.local_tokens + r.top_k_fp * r.window_size)
            stats.update({
                "budget_tokens": int(r.total_budget_tokens),
                "retained_tokens_cap": cap,
                "fp_tokens_cap": fp_cap,
                "N_q": int(r.N_q),
                "int2_used": seq > fp_cap,
                "evicted": seq > cap,
                # Upper bound on the KV bytes held at the end of this document,
                # as a fraction of what the full cache holds for the same sequence.
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

        # Drop this frame's references before empty_cache, or the cache's
        # buffers are still alive when the allocator is asked to release them.
        del out, new, inputs, input_ids, attention_mask
        cache = hooks = None
        self._cleanup_memory()
        return output, stats

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

    def _write_meta(self, type_: str, docs: List[Dict[str, Any]], max_new: int, n_oom: int,
                    run_start: float, run_end: float, tot_gen: int, tot_wall: float,
                    peak: float, output_dir: Path) -> None:
        cfg, lg = self.config, self.lg
        budget = cfg.cache.cache_budget
        from utils.cache_factory import quant_card_bits_record, quant_tier_policy_record

        meta = {
            "read_gate": self._read_gate_report(f"longgenbench {lg.setting} {type_}"),
            "quant_card_bits": quant_card_bits_record(cfg.cache),
            **quant_tier_policy_record(cfg.cache),
            "benchmark": "LongGenBench",
            "setting": lg.setting,
            "type": type_,
            "number": docs[0]["number"],
            "dataset": HF_DATASET,
            "dataset_revision": HF_REVISION,
            "reference_code": CODE_REPO,
            "reference_generation": GENERATION_CODE,
            "data_file": FILES[lg.setting][0],
            "dataset_sha": dataset_sha(lg.data_dir, lg.setting),
            "prompt_format": PROMPT_FORMAT,
            "start": lg.start,
            "end": lg.end,
            "ids": [d["id"] for d in docs],
            "num_examples": len(docs),
            "num_oom": n_oom,
            "max_new_tokens": max_new,
            "max_new_tokens_override": getattr(lg, "max_new_tokens", None),
            "generation": {
                "decoding": lg.decoding,
                "sampling": ({**GENERATION_SAMPLING, "seed": int(lg.sample_seed),
                              "seed_rule": "the same seed for every document"}
                             if lg.decoding == "reference" else None),
                "stop_string": STOP_STRING,
                "stop_string_in_output": False,
                "eos_token_id": self._eos_ids(),
                "output": "prefix + generation",
                "batch_size": 1,
            },
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
            "total_gen_tokens": tot_gen,
            "total_generate_wall_s": round(tot_wall, 1),
            "decode_tokps_wallclock": round(tot_gen / tot_wall, 2) if tot_wall > 0 else None,
            "peak_mem_gib_max": round(peak, 3),
            **capture_environment(),
            "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_start)),
            "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_end)),
        }
        (output_dir / f"{type_slug(type_)}.meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def _get_tokenizer_sha(self) -> str:
        if self.tokenizer is None:
            return "unknown"
        try:
            from utils.hashing import sha256_tokenizer

            return sha256_tokenizer(self.tokenizer)
        except Exception:
            return "unknown"
