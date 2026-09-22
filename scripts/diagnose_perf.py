"""Locate the wall-clock bottleneck in a LongBench run.

Answers one question: is the time going into StickyKV, or into the model?

Runs three probes in increasing order of involvement:

1. Placement  — where the weights actually live (``hf_device_map``).
2. Bare model — HF ``generate`` with a plain DynamicCache, no windowed cache,
   no hooks, no quant. This is the floor: StickyKV cannot beat it, and no
   optimization inside StickyKV can move it.
3. Windowed  — the same generate with the configured StickyKV backend.

If probe 2 is already at the observed per-sample time, the bottleneck is the
model's placement, not the cache — see the ms/token vs. bandwidth reading.

Usage:
    python scripts/diagnose_perf.py --config configs/longbench_ours_flash_attn.yaml
    python scripts/diagnose_perf.py --config configs/longbench_ours_flash_attn.yaml \
        --prefill 4900 --gen 32
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import torch

from utils.config import load_config


PARAM_BYTES = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_generate(model, input_ids, gen_len, **gen_kwargs):
    """Wall-clock a greedy generate, returning (seconds, tokens_emitted)."""
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=gen_len,
            min_new_tokens=gen_len,  # defeat early EOS so the timing is comparable
            num_beams=1,
            do_sample=False,
            **gen_kwargs,
        )
    _sync()
    return time.perf_counter() - t0, out.shape[-1] - input_ids.shape[-1]


def report_placement(model) -> bool:
    """Print where each layer lives. Returns True if anything is off-GPU."""
    print("\n" + "=" * 72)
    print("PROBE 1 — weight placement")
    print("=" * 72)

    dmap = getattr(model, "hf_device_map", None)
    if dmap is None:
        print("  no hf_device_map (model loaded without device_map)")
        print(f"  first parameter device: {next(model.parameters()).device}")
        return next(model.parameters()).device.type != "cuda"

    tally = Counter(str(d) for d in dmap.values())
    print("  hf_device_map summary (module count per device):")
    for dev, n in sorted(tally.items()):
        flag = "  <-- OFF-GPU" if dev in ("cpu", "disk") else ""
        print(f"    {dev:<12} {n:>4} modules{flag}")

    offloaded = {k: v for k, v in dmap.items() if str(v) in ("cpu", "disk")}
    if offloaded:
        print(f"\n  {len(offloaded)} modules are NOT on the GPU. Examples:")
        for k in list(offloaded)[:8]:
            print(f"    {k} -> {offloaded[k]}")
        print(
            "\n  Every one of these streams over PCIe on EVERY forward pass —\n"
            "  i.e. once per generated token. This dominates decode and is\n"
            "  completely independent of KV-cache size."
        )
    else:
        print("\n  All modules are GPU-resident. Offload is NOT the bottleneck.")
    return bool(offloaded)


def report_hardware(model) -> None:
    print("\n" + "-" * 72)
    print("hardware / model")
    print("-" * 72)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(
                f"  GPU{i}: {p.name}  total={total/2**30:.1f} GiB  "
                f"free={free/2**30:.1f} GiB"
            )
    else:
        print("  NO CUDA AVAILABLE — running on CPU.")

    n_params = sum(p.numel() for p in model.parameters())
    dtype = next(model.parameters()).dtype
    gb = n_params * PARAM_BYTES.get(dtype, 2) / 1e9
    print(f"  params: {n_params/1e9:.2f} B  dtype={dtype}  weights={gb:.2f} GB")
    print(
        "  decode reads every weight once per token, so:\n"
        f"    ms/token floor = {gb:.2f} GB / (device bandwidth GB/s) * 1000"
    )


def interpret(ms_per_tok: float, weights_gb: float) -> None:
    """Turn ms/token into an implied bandwidth and name the regime."""
    implied = weights_gb / (ms_per_tok / 1000.0)
    print(f"    -> implied weight-read bandwidth: {implied:.1f} GB/s")
    if implied < 40:
        print(
            "    -> PCIe-class bandwidth. The weights are being streamed from\n"
            "       host RAM (or disk) every token. The model is NOT resident\n"
            "       on the GPU. No KV-cache work can fix this."
        )
    elif implied < 250:
        print("    -> low-end GPU memory bandwidth, or heavy per-step overhead.")
    else:
        print("    -> GPU-resident. Decode is running at device bandwidth.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefill", type=int, default=4900,
                    help="synthetic prompt length (qasper avg ~4.9k)")
    ap.add_argument("--gen", type=int, default=32,
                    help="tokens to generate per probe (qasper real = 128)")
    ap.add_argument("--skip-windowed", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}
    print(f"loading {cfg.model.name} (dtype={cfg.model.dtype}, "
          f"attn={cfg.model.attn_implementation}) ...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtypes.get(cfg.model.dtype, torch.float16),
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto",
    )
    model.eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    report_hardware(model)
    report_placement(model)

    n_params = sum(p.numel() for p in model.parameters())
    weights_gb = n_params * PARAM_BYTES.get(next(model.parameters()).dtype, 2) / 1e9

    # Synthetic prompt at qasper's real length — content is irrelevant to timing.
    input_ids = torch.randint(
        1000, 20000, (1, args.prefill), device=model.device, dtype=torch.long
    )
    pad = tok.pad_token_id or tok.eos_token_id

    # ---- PROBE 2: bare model, zero StickyKV ----------------------------
    print("\n" + "=" * 72)
    print(f"PROBE 2 — bare model, DynamicCache, no hooks "
          f"(prefill={args.prefill}, gen={args.gen})")
    print("=" * 72)
    _timed_generate(model, input_ids[:, :64], 2, pad_token_id=pad)  # warmup
    bare_s, n = _timed_generate(model, input_ids, args.gen, pad_token_id=pad)
    bare_ms = bare_s / max(n, 1) * 1000
    print(f"  {bare_s:.2f}s for {n} tokens = {bare_ms:.0f} ms/token")
    interpret(bare_ms, weights_gb)
    print(f"\n  extrapolated to a real qasper sample (128 tokens): "
          f"{bare_ms*128/1000:.1f}s")
    print(f"  extrapolated to 20 samples: {bare_ms*128*20/1000/60:.1f} min")
    print("  ^ this is the FLOOR. StickyKV cannot go faster than this.")

    # ---- PROBE 3: windowed cache ---------------------------------------
    backend = getattr(cfg.cache, "backend", "dynamic")
    pkg = getattr(cfg.cache, "backend_package", None)
    if args.skip_windowed or backend != "windowed" or not pkg:
        print("\n(skipping windowed probe — config is not a windowed backend)")
        return

    from utils.cache_factory import get_cache_classes

    WindowedCache, WindowedCacheConfig, install_score_hooks = get_cache_classes(pkg)

    rope = None
    for name, mod in model.named_modules():
        if "rotary" in name.lower() or "rope" in name.lower():
            rope = mod
            break

    q = getattr(cfg.cache, "quant_ratio", 0.0)
    print("\n" + "=" * 72)
    print(f"PROBE 3 — windowed cache (budget={cfg.cache.cache_budget}, "
          f"quant_ratio={q})")
    print("=" * 72)

    cache_config = WindowedCacheConfig(
        window_size=cfg.cache.window_size,
        num_sink_tokens=cfg.cache.num_sink_tokens,
        local_window_size=cfg.cache.local_window_size,
        cache_budget=cfg.cache.cache_budget or 0.20,
        rerotate_on_evict=getattr(cfg.cache, "rerotate_on_evict", False),
        quant_ratio=q,
    )
    cache = WindowedCache(
        config=cache_config,
        prefill_len=args.prefill,
        model_config=model.config,
        kv_dtype=dtypes.get(cfg.model.dtype, torch.float16),
        rope_module=rope,
        num_layers=model.config.num_hidden_layers,
        max_tokens=args.gen,
    )
    hooks = install_score_hooks(model, cache, cache_config)
    try:
        win_s, n = _timed_generate(
            model, input_ids, args.gen, pad_token_id=pad, past_key_values=cache
        )
    finally:
        hooks.remove()
    win_ms = win_s / max(n, 1) * 1000
    print(f"  {win_s:.2f}s for {n} tokens = {win_ms:.0f} ms/token")

    # ---- verdict --------------------------------------------------------
    overhead = win_s - bare_s
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  bare model (irreducible):  {bare_s:6.2f}s  ({bare_ms:.0f} ms/tok)")
    print(f"  windowed cache:            {win_s:6.2f}s  ({win_ms:.0f} ms/tok)")
    print(f"  StickyKV's share:          {overhead:+6.2f}s  "
          f"({overhead/win_s*100:+.1f}% of wall clock)")
    print()
    share = max(overhead, 0) / win_s
    print(f"  Amdahl ceiling: making ALL StickyKV code infinitely fast would")
    print(f"  cut a 34-minute run to {34*(1-share):.1f} minutes.")
    if share < 0.15:
        print("\n  StickyKV is a minority of the runtime. Optimizing it further")
        print("  is capped by the number above — fix the model path first.")


if __name__ == "__main__":
    main()
