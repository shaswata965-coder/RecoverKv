"""Kernel-level profile of one steady-state decode step.

Answers the question TPOT cannot: of the ~85 ms a decode step costs, how much
is the GPU actually running kernels, and which kernels?

Why this exists. Suite C reports wall clock only, and the one structural
measurement behind the perf work so far was a hand-count of ATen dispatches
inside ``WindowedCache.update``. That count is a *share of the cache update*,
not a share of TPOT, and the two were never reconciled: the eviction was 81% of
the update's launches and ~10% of the step. Compiling it moved TPOT 101.6 -> 85
ms, which is what its own arithmetic predicted, and left the other ~61 ms of
excess over the fp16 baseline untouched and unexplained.

The flat-TPOT signature the decision rested on (perf_runner.py:1074 -- "flat
across a 32x change in batch ... only happens when the GPU is idle waiting on
the host") does NOT identify a host bottleneck. A single kernel whose critical
path is batch-invariant -- e.g. one launched on a ``B * H_kv`` grid, which is 8
blocks on a 108-SM A100 at B=1 -- produces the identical signature. The two
hypotheses are distinguished by exactly one number, printed first below:

    GPU busy fraction = (CUDA kernel self time) / wall time

  << 1.0   host-bound. The launch/sync path is the budget; the dispatch-count
           argument was right and the remaining launches are the target.
  ~= 1.0   kernel-bound. The GPU is saturated and the kernels themselves are
           slow. No amount of launch collapsing can help; fix the kernel.

Then the A/B that names the kernel, if it is one:

    python scripts/profile_decode.py --config <cfg> --fused 1     # Triton path
    python scripts/profile_decode.py --config <cfg> --fused 0     # materialize

Usage:
    python scripts/profile_decode.py --config configs/perf_ours.yaml \
        --prefill 1048 --batch 1 --steps 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _env_gate(args) -> None:
    """Set the path flags BEFORE anything reads them (hooks latch at install)."""
    if args.fused is not None:
        os.environ["STICKYKV_FUSED_DECODE"] = str(args.fused)
    if args.compile_evict is not None:
        os.environ["STICKYKV_COMPILE_EVICT"] = str(args.compile_evict)


def _self_device_us(ev) -> float:
    """Per-event CUDA self time. The attribute was renamed in torch 2.x."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(ev, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def _perf_cell_quant(cfg) -> float:
    """quant_ratio as the benchmarked cell sets it (perf.configs[0] first)."""
    try:
        c = dict(cfg.perf.configs[0])
    except Exception:
        c = {}
    return float(c.get("quant_ratio", getattr(cfg.cache, "quant_ratio", 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefill", type=int, default=1048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20, help="profiled decode steps")
    ap.add_argument("--warmup", type=int, default=12,
                    help="decode steps before profiling; must exceed window_size "
                         "so at least one eviction is compiled/autotuned away")
    ap.add_argument("--fused", type=int, choices=(0, 1), default=None,
                    help="STICKYKV_FUSED_DECODE. 0 = materialize path (A/B the "
                         "Triton decode kernel out of the picture)")
    # Defaults to 1, which is what the library itself picks on CUDA (the device
    # decides -- see _compile_evict_enabled). This used to be the one place the
    # two disagreed: the library defaulted to 0 while run_perf_table.sh exported
    # 1, so an unset profile attributed the EAGER eviction against a table that
    # ran it compiled -- a different method, and not by a little: compiling moved
    # TPOT 101.6 -> 85 ms (DECODE_HISTORY.md §1). Pass 0 deliberately to A/B it.
    ap.add_argument("--compile-evict", type=int, choices=(0, 1), default=1)
    ap.add_argument("--trace", default=None, help="write a chrome trace here")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    _env_gate(args)

    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from utils.config import load_config

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required -- this profile is meaningless on CPU.")

    cfg = load_config(args.config)
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}
    dtype = dtypes.get(cfg.model.dtype, torch.float16)

    # Backend from the PERF cell, not cfg.model: the generated perf config's
    # model section carries only name/revision/dtype, so cfg.model.
    # attn_implementation silently yields ModelConfig's "eager" default and
    # both the model load and the cache package come out wrong. Shared with
    # audit_e2e so the two tools cannot disagree about what they measured.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_e2e import resolve_backend
    pkg, attn_impl = resolve_backend(cfg)

    print(f"loading {cfg.model.name} (attn={attn_impl}, cache_package={pkg}) ...")
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=dtype,
        attn_implementation=attn_impl, device_map="auto")
    model.eval()

    input_ids = torch.randint(1000, 20000, (args.batch, args.prefill),
                              device=model.device, dtype=torch.long)

    # -- cache + hooks: same construction as scripts/diagnose_perf.py --------
    from utils.cache_factory import get_cache_classes
    WindowedCache, WindowedCacheConfig, install_score_hooks = get_cache_classes(pkg)

    rope = None
    for name, mod in model.named_modules():
        if "rotary" in name.lower() or "rope" in name.lower():
            rope = mod
            break

    total_steps = args.warmup + args.steps + 1
    # Shared with audit_e2e: budget and quant settings live in perf.configs[0],
    # not cfg.cache, and first_eviction_step must be carried or this profiles a
    # different method than the table.
    from audit_e2e import resolve_cache_kwargs
    q = _perf_cell_quant(cfg)
    cache_config = WindowedCacheConfig(**resolve_cache_kwargs(cfg, q, pkg))
    cache = WindowedCache(
        config=cache_config, prefill_len=args.prefill, model_config=model.config,
        kv_dtype=dtype, rope_module=rope,
        num_layers=model.config.num_hidden_layers, max_tokens=total_steps)
    hooks = install_score_hooks(model, cache, cache_config)

    # cache_position must be passed EXPLICITLY and advanced monotonically. Left
    # to itself, transformers derives it from `past_key_values.get_seq_length()`,
    # which for an evicting cache is the RETAINED key count, not the absolute
    # token index — so it jumps backwards after the first eviction and the store
    # ends up with duplicated, non-monotonic positions. `generate()` does this
    # for us, which is why every quality runner is unaffected and only this
    # hand-written decode loop trips the cache's position contract.
    device = input_ids.device
    pos = 0                      # absolute token index, advanced by _step

    def _step(ids, past, n_new):
        nonlocal pos
        o = model(input_ids=ids, past_key_values=past, use_cache=True,
                  return_dict=True,
                  cache_position=torch.arange(pos, pos + n_new, device=device))
        pos += n_new
        return o

    try:
        with torch.no_grad():
            out = _step(input_ids, cache, input_ids.shape[1])
            pkv = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Warm up past compile / autotune / the first eviction.
            for _ in range(args.warmup):
                out = _step(nxt, pkv, 1)
                pkv = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                for _ in range(args.steps):
                    out = _step(nxt, pkv, 1)
                    pkv = out.past_key_values
                    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                torch.cuda.synchronize()
            wall_us = (time.perf_counter() - t0) * 1e6
    finally:
        hooks.remove()

    evs = prof.key_averages()
    gpu_us = sum(_self_device_us(e) for e in evs)
    n_launch = sum(e.count for e in evs if _self_device_us(e) > 0)

    # Host-side stalls. A sync does not cost a launch's ~5 us -- it drains the
    # queue, so it converts every downstream launch's CPU cost from hidden to
    # exposed. This is the term a dispatch COUNT cannot see.
    sync_names = ("cudaDeviceSynchronize", "cudaStreamSynchronize",
                  "cudaMemcpyAsync", "Memcpy DtoH", "cudaHostAlloc",
                  "cudaStreamWaitEvent", "cudaEventSynchronize")
    syncs = [(e.key, e.count, e.cpu_time_total) for e in evs
             if any(s in e.key for s in sync_names)]

    n = max(args.steps, 1)
    print("\n" + "=" * 74)
    print(f"DECODE PROFILE  batch={args.batch} prefill={args.prefill} "
          f"steps={n}  fused={os.environ.get('STICKYKV_FUSED_DECODE', '1')} "
          f"compile_evict={os.environ.get('STICKYKV_COMPILE_EVICT', 'device')}")
    if os.environ.get("STICKYKV_COMPILE_EVICT", "").strip() == "0":
        print("  !! eviction is EAGER here. The library compiles it on CUDA with "
              "nothing set,\n     so this profiles a different method than a "
              "default run. The elementwise/copy\n     kernels below are "
              "inflated accordingly.")
    print("=" * 74)
    print(f"  wall            {wall_us / n / 1000:8.2f} ms/step")
    print(f"  CUDA kernels    {gpu_us / n / 1000:8.2f} ms/step")
    busy = gpu_us / max(wall_us, 1.0)
    print(f"  GPU busy        {busy * 100:8.1f} %   <-- THE number")
    print(f"  kernel launches {n_launch / n:8.0f} /step")
    if busy < 0.5:
        print("  -> HOST-BOUND. The GPU idles most of the step; launches and "
              "syncs are the budget.")
    elif busy > 0.85:
        print("  -> KERNEL-BOUND. The GPU is saturated; the kernels themselves "
              "are slow. Collapsing launches cannot help.")
    else:
        print("  -> MIXED. Both terms are real; fix the larger one first.")

    # Which method was actually profiled. The fused kernel running does not mean
    # the read gate ran: a store with no sketch cards reads the whole tier,
    # correctly, and looks identical in every number above.
    try:
        from modules.windowed_cache import flash_decode
        st = flash_decode.stats()
        if st["armed"] or st["fired"]:
            rf = st["read_fraction"]
            print(f"\n  path        armed={st['armed']} fired={st['fired']} "
                  f"gated={st['gated']}"
                  + (f"  read_fraction={rf:.3f}" if rf is not None else ""))
            if st["fired"] and not st["gated"]:
                print("  -> the gate NEVER RAN: this profile is the ungated "
                      "tier (no sketch cards, or quant_gate_ratio >= 1.0).")
            elif st["gated"] and st["gated"] != st["fired"]:
                print(f"  -> the gate ran on only {st['gated']}/{st['fired']} "
                      "fused steps; the rest read the whole tier.")
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        pass

    if syncs:
        print("\n  host stalls (each drains the queue and exposes downstream "
              "launch cost):")
        for k, c, us in sorted(syncs, key=lambda r: -r[2])[:8]:
            print(f"    {k:<32} {c / n:7.1f} /step   {us / n / 1000:7.2f} ms/step")

    print(f"\n  top {args.top} kernels by CUDA self time:")
    ranked = sorted(((e, _self_device_us(e)) for e in evs), key=lambda r: -r[1])
    for e, us in ranked[:args.top]:
        if us <= 0:
            break
        print(f"    {us / n / 1000:7.3f} ms/step  {us / max(gpu_us, 1) * 100:5.1f}%  "
              f"n={e.count / n:6.1f}  {e.key[:58]}")

    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"\n  chrome trace -> {args.trace}")


if __name__ == "__main__":
    main()
