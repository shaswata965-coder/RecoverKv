"""PerfRunner — wall-clock benchmarks (Suite C).

TTFT, throughput (tok/s), TPOT, peak memory across the backend x budget
factorial. Supports eager (Kaggle-safe) and flash_attn (Ampere+) backends.
Gracefully skips configs that OOM or require unavailable flash-attn.
"""
from __future__ import annotations
import json, time, gc
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from utils.config import ExperimentConfig
from utils.env_capture import capture_environment
from utils.logger import get_logger

log = get_logger(__name__)

def _flash_attn_available() -> bool:
    try:
        import flash_attn  # type: ignore
        return True
    except ImportError:
        return False

class PerfRunner:
    """Suite C — TTFT, throughput, TPOT benchmarks."""
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> List[Path]:
        cfg = self.config
        pc = cfg.perf
        log.info("=== Performance Runner ===")
        flash_ok = _flash_attn_available()
        if not flash_ok:
            log.info("flash-attn not available — flash configs will be skipped")
        env = capture_environment()
        # Try to lock GPU clocks
        clocks_locked = False
        if pc.enable_clock_locking:
            clocks_locked = self._try_lock_clocks()
        # Build (prefill_len, gen_len) grid. Explicit `grid:` wins; otherwise
        # fall back to legacy `prefill_lengths` x scalar `gen_len`.
        if pc.grid:
            cells = [
                (int(g["prefill_len"]), int(g["gen_len"]),
                 int(g.get("batch_size", pc.batch_size)))
                for g in pc.grid
            ]
        else:
            cells = [(p, pc.gen_len, pc.batch_size) for p in pc.prefill_lengths]
        output_paths = []
        for prefill_len, gen_len, batch_size in cells:
            log.info("--- Cell: prefill=%d gen=%d batch=%d ---", prefill_len, gen_len, batch_size)
            result = self._run_prefill(prefill_len, gen_len, batch_size, pc, cfg, flash_ok, env)
            result["clocks_locked"] = clocks_locked
            result["gen_len"] = gen_len
            result["batch_size"] = batch_size
            path = self._save(result, prefill_len, gen_len, batch_size, cfg, env, clocks_locked)
            output_paths.append(path)
        return output_paths

    def _run_prefill(self, prefill_len: int, gen_len: int, batch_size: int, pc, cfg, flash_ok: bool, env: dict) -> dict:
        configs = pc.configs
        n_configs = len(configs)
        n_runs = pc.num_measurement_runs
        ttft = np.full((n_configs, n_runs), np.nan)
        throughput = np.full((n_configs, n_runs), np.nan)
        tpot = np.full((n_configs, n_runs), np.nan)
        e2e_latency = np.full((n_configs, n_runs), np.nan)
        peak_mem = np.full((n_configs, n_runs), np.nan)
        skipped = np.zeros(n_configs, dtype=bool)
        names = []
        attn_impls = []
        for ci, c in enumerate(configs):
            name = c.get("name", f"config_{ci}")
            names.append(name)
            attn_impls.append(c.get("attn_implementation", "eager"))
            # Check flash requirement
            if c.get("requires_flash_attn", False) and not flash_ok:
                if pc.skip_if_flash_attn_unavailable:
                    log.info("Skipping %s (flash-attn unavailable)", name)
                    skipped[ci] = True
                    continue
            log.info("Running config: %s", name)
            try:
                measurements = self._measure_config(c, prefill_len, gen_len, batch_size, pc, cfg)
                for ri, m in enumerate(measurements):
                    ttft[ci, ri] = m["ttft_ms"]
                    throughput[ci, ri] = m["throughput_tokps"]
                    tpot[ci, ri] = m["tpot_ms"]
                    e2e_latency[ci, ri] = m["e2e_latency_ms"]
                    peak_mem[ci, ri] = m["peak_memory_mb"]
            except torch.cuda.OutOfMemoryError:
                if pc.skip_if_oom:
                    log.warning("OOM on %s at prefill %d — skipping", name, prefill_len)
                    skipped[ci] = True
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
            except Exception as e:
                log.warning("Error on %s: %s — skipping", name, e)
                skipped[ci] = True
        return {"names": names, "attn_impls": attn_impls, "ttft": ttft,
                "throughput": throughput, "tpot": tpot, "e2e_latency": e2e_latency,
                "peak_mem": peak_mem, "skipped": skipped}

    def _measure_config(self, c: dict, prefill_len: int, gen_len: int, batch_size: int, pc, cfg) -> List[dict]:
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtypes.get(cfg.model.dtype, torch.float16)
        attn_impl = c.get("attn_implementation", "eager")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, revision=cfg.model.revision,
            torch_dtype=torch_dtype, attn_implementation=attn_impl,
            device_map="auto")
        model.eval()
        # Create input. batch_size>1 measures batched throughput; the windowed
        # cache evicts each row independently (equal-length inputs, no padding).
        batch_size = max(1, int(batch_size))
        data_source = c.get("data_source") or getattr(pc, "data_source", None)
        if data_source:
            input_ids = self._build_input_ids(
                data_source, tokenizer, prefill_len, batch_size, cfg.run.seed
            ).to(model.device)
        else:
            input_ids = torch.randint(
                100, 30000, (batch_size, prefill_len), device=model.device
            )
        # Setup cache
        cache_backend = c.get("cache_backend", "dynamic")
        cache_pkg = c.get("cache_package")
        budget = c.get("cache_budget")
        # Windowed-cache setup: resolve classes + RoPE once, but DO NOT install
        # hooks here. Each warmup/measurement iteration creates a fresh cache,
        # so hooks must be (re)installed on the cache the model actually
        # receives, then removed before the next iteration.
        WC = WCC = install_hooks = None
        cc = None
        rope = None
        if cache_backend == "windowed" and cache_pkg:
            from utils.cache_factory import (
                assert_transformers_version_supported,
                get_cache_classes,
            )
            # Fail fast: the windowed cache's RoPE handling assumes monotonic
            # cache_position (transformers <= 4.47).
            assert_transformers_version_supported()
            WC, WCC, install_hooks = get_cache_classes(cache_pkg)
            w = cfg.window
            # quant_ratio is per-config (like cache_budget), falling back to the
            # shared cache.quant_ratio; 0.0 keeps the pure-fp16 path (design.md §7).
            quant_ratio = c.get("quant_ratio", getattr(cfg.cache, "quant_ratio", 0.0))
            # None (default) = auto: memoize the dequantized Q tier at B=1, not
            # above. The default is set by the memory bound (the memo costs
            # ~149 MB/row against ~131 MB/row of actual KV, halving max-B); what
            # it trades that for — ~8x fewer Q-tier dequants per step — is a
            # wall-clock question only this suite can answer. Set it explicitly
            # to measure both sides. See BATCHING_PLAN.md §5.
            memoize = c.get("quant_memoize_read",
                            getattr(cfg.cache, "quant_memoize_read", None))
            cc = WCC(window_size=w.window_size, num_sink_tokens=w.num_sink_tokens,
                      local_window_size=w.local_window_size,
                      cache_budget=budget if budget is not None else 0.5,
                      rerotate_on_evict=getattr(cfg.cache, "rerotate_on_evict", False),
                      quant_ratio=quant_ratio,
                      quant_memoize_read=memoize)
            # Two-pass RoPE discovery (mirrors ours_parity_runner.py).
            for nm, mod in model.named_modules():
                if "rotary" in nm.lower() or "rope" in nm.lower():
                    rope = mod; break
            if rope is None:
                for nm, mod in model.named_modules():
                    if hasattr(mod, "rotary_emb"):
                        rope = mod.rotary_emb; break
            if rope is None:
                from utils.config import ConfigValidationError
                raise ConfigValidationError(
                    "Could not locate a RoPE module on the model. WindowedCache "
                    "requires a rotary embedding module for key rerotation."
                )

        def _make_windowed_cache():
            # max_tokens sizes the budget against the FULL expected sequence
            # (prefill + generation) per design.md §7, which is what
            # ours_parity_runner and longbench_runner both pass. This used to
            # pass 0 on the premise that perf was "prefill-only, no generation" —
            # but the measurement loop below generates gen_len tokens, so the
            # budget came out sized on the prompt alone. At gen_len >> prefill
            # that is a different (much smaller) cache than the quality suites
            # measure, so their numbers would not correspond.
            return WC(config=cc, prefill_len=prefill_len,
                      model_config=model.config, kv_dtype=torch_dtype,
                      rope_module=rope,
                      num_layers=model.config.num_hidden_layers,
                      max_tokens=gen_len)

        gen_kwargs = {}
        if attn_impl == "eager" and c.get("install_hooks_for_measurement"):
            gen_kwargs["output_attentions"] = True
        # Warmup
        for _ in range(pc.num_warmup_runs):
            hooks = None
            try:
                with torch.no_grad():
                    if cache_backend == "windowed" and cache_pkg:
                        pkv = _make_windowed_cache()
                        hooks = install_hooks(model, pkv, cc)
                    else:
                        pkv = DynamicCache() if cache_backend == "dynamic" else None
                    model(input_ids=input_ids, past_key_values=pkv,
                          use_cache=True, return_dict=True, **gen_kwargs)
            finally:
                if hooks is not None:
                    hooks.remove()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Measurement runs
        measurements = []
        for ri in range(pc.num_measurement_runs):
            hooks = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                # TTFT — synchronize BEFORE t0 so prior async work doesn't taint
                # the measurement and the sync's own latency isn't timed.
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    if cache_backend == "windowed" and cache_pkg:
                        pkv = _make_windowed_cache()
                        hooks = install_hooks(model, pkv, cc)
                    else:
                        pkv = DynamicCache() if cache_backend == "dynamic" else None
                    out = model(input_ids=input_ids, past_key_values=pkv,
                                use_cache=True, return_dict=True, **gen_kwargs)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = time.perf_counter()
                ttft_ms = (t1 - t0) * 1000
                # Generate tokens for TPOT
                pkv = out.past_key_values
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t2 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(gen_len - 1):
                        out = model(input_ids=next_tok, past_key_values=pkv,
                                    use_cache=True, return_dict=True, **gen_kwargs)
                        pkv = out.past_key_values
                        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t3 = time.perf_counter()
                gen_time = t3 - t2
                tpot_ms = (gen_time / max(gen_len - 1, 1)) * 1000
                e2e_latency_ms = (t3 - t0) * 1000
                # End-to-end throughput includes prefill (TTFT) + decode time; this
                # mirrors the legacy field name but is NOT decode-only. Counts all
                # batch_size rows (B=1 ⇒ identical to the legacy value).
                throughput_tokps = (batch_size * gen_len) / max(gen_time + (t1-t0), 1e-9)
                peak_mb = 0.0
                if torch.cuda.is_available():
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                measurements.append({"ttft_ms": ttft_ms, "throughput_tokps": throughput_tokps,
                                     "tpot_ms": tpot_ms, "e2e_latency_ms": e2e_latency_ms,
                                     "peak_memory_mb": peak_mb})
            finally:
                if hooks is not None:
                    hooks.remove()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return measurements

    def _save(self, result: dict, prefill_len: int, gen_len: int, batch_size: int, cfg, env: dict, clocks_locked: bool) -> Path:
        od = Path(cfg.telemetry.output_dir); od.mkdir(parents=True, exist_ok=True)
        npz_path = od / f"perf_prefill{prefill_len}_gen{gen_len}_bs{batch_size}.npz"
        meta = {
            "prefill_len": prefill_len,
            "gen_len": gen_len,
            "batch_size": batch_size,
            **env,
            "clocks_locked": clocks_locked,
        }
        np.savez_compressed(
            str(npz_path),
            config_names=np.array(result["names"], dtype=object),
            attn_implementations=np.array(result["attn_impls"], dtype=object),
            ttft_ms=result["ttft"],
            throughput_tokps=result["throughput"],
            tpot_ms=result["tpot"],
            e2e_latency_ms=result["e2e_latency"],
            peak_memory_mb=result["peak_mem"],
            skipped_mask=result["skipped"],
            metadata_json=np.array([json.dumps(meta)], dtype=object),
        )
        log.info("Saved perf: %s", npz_path)
        return npz_path

    def _build_input_ids(self, data_source: str, tokenizer, prefill_len: int,
                         batch_size: int, seed: int) -> torch.Tensor:
        import random
        from data.longbench_loader import load_longbench_dataset
        log.info("Loading dataset '%s' for perf inputs (prefill_len=%d, batch_size=%d)",
                 data_source, prefill_len, batch_size)
        ds = load_longbench_dataset(data_source)
        parts = [s["context"] + "\n\n" + s["input"] for s in ds]
        full_text = "\n\n".join(parts)
        all_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False)
        total = all_ids.shape[1]
        chunks = [all_ids[:, i:i + prefill_len]
                  for i in range(0, total - prefill_len + 1, prefill_len)]
        if len(chunks) < batch_size:
            raise ValueError(
                f"Dataset '{data_source}' yields only {len(chunks)} chunks of "
                f"{prefill_len} tokens but batch_size={batch_size}. "
                f"Reduce batch_size or prefill_len."
            )
        rng = random.Random(seed)
        selected = rng.sample(chunks, batch_size)
        return torch.cat(selected, dim=0)  # [batch_size, prefill_len]

    def _try_lock_clocks(self) -> bool:
        import subprocess
        try:
            r = subprocess.run(["nvidia-smi", "-lgc", "1400,1400"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                log.info("GPU clocks locked"); return True
            log.warning("nvidia-smi -lgc failed: %s", r.stderr)
        except Exception as e:
            log.warning("Clock locking failed: %s", e)
        return False
