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


#: Substrings that mark a PROFILER RANGE rather than a kernel.
#:
#: `torch._dynamo.utils.dynamo_timed` opens a `record_function` range around
#: compilation, and those ranges come back carrying CUDA self time -- a range's
#: self device time is the CUDA time of everything nested inside it. Counting
#: them alongside the kernels they enclose double-counts those kernels.
#:
#: Observed on a 2026-09-19 A100 profile at 4096/B=32: five such rows summed to
#: 726.824 ms/step, three of them with **n < 1 launch in the whole window** (a
#: kernel with no launches cannot have time -- that is the tell). They pushed
#: `GPU busy` to 203.9%, which is impossible on one stream, and the script then
#: printed "KERNEL-BOUND ... the kernels themselves are slow" for a step whose
#: kernels were in fact the only thing that had got faster.
#:
#: Kernels launched *inside* one of these ranges are unaffected: they appear
#: under their own names as their own rows and are still counted. Only the
#: enclosing range is dropped.
_NON_KERNEL_MARKERS = (
    "(dynamo_timed)",           # every row observed on that profile
    "CachingAutotuner",         # ... and the individual names, in case the
    "compile_fx",               #     "(dynamo_timed)" suffix ever changes
    "create_aot_dispatcher",
    "compile_inner",
    "_recursive_",
    "Torch-Compiled Region",
    "TorchDynamo Cache Lookup",
    "CompiledFunction",
    "AOTAutograd",
    "dynamo",
    "inductor",
)


def _is_compile_range(ev) -> bool:
    """True for a compilation / autotuning profiler range (see
    :data:`_NON_KERNEL_MARKERS`)."""
    return any(m in str(ev.key) for m in _NON_KERNEL_MARKERS)


def _is_device_event(ev) -> bool:
    """True for a CUDA KERNEL event; false for the ATen op that launched it.

    ``key_averages()`` returns BOTH views, and both carry self device time. A
    leaf op's self device time IS the time of the kernels it launched, and those
    kernels then report it again under their own names. Summing the lot
    double-counts every kernel in the step -- which is what ``gpu_us`` did here:
    it summed every event unconditionally, so ``CUDA kernels``, ``kernel
    launches`` and therefore the ``GPU busy %`` this script exists to report
    were all inflated by close to 2x.

    ``device_type`` is the real signal. The name check is a fallback for a build
    that does not expose it: the operator view is namespaced (``aten::``,
    ``autograd::``), device events are not. Memcpy/Memset are device events and
    are deliberately NOT excluded by it.
    """
    # FIRST, and ahead of device_type: a compilation range is never a kernel,
    # whatever the profiler tags it as. On the build that produced the profile
    # above these ranges passed the device-type arm.
    if _is_compile_range(ev):
        return False
    dt = getattr(ev, "device_type", None)
    if dt is not None:
        try:
            from torch.autograd import DeviceType
            return dt == DeviceType.CUDA
        except Exception:  # pragma: no cover - torch-version dependent
            pass
    return not str(ev.key).startswith(
        ("aten::", "autograd::", "torch::", "nn.Module", "Optimizer",
         "ProfilerStep", "cudaLaunch", "cudaMemcpy", "cudaStream",
         "cudaDevice", "cudaEvent", "cudaHost"))


def _dynamo_counters() -> dict:
    """Dynamo's own compile tally, or ``{}``. Read-only, no side effects."""
    try:
        from torch._dynamo.utils import counters
        return {k: int(v) for k, v in dict(counters.get("stats", {})).items()}
    except Exception:  # pragma: no cover - torch-version dependent
        return {}


def _print_compile_contamination(evs, wall_us: float, n: int) -> None:
    """Say, out loud, if COMPILATION happened inside the measured window.

    Excluding the ranges from the kernel totals without reporting them would
    remove the symptom and keep the disease: a window in which Inductor is still
    compiling and autotuning is ~10x slower than a steady step and would
    otherwise carry no line saying why.

    The signature to recognise:
      * `CachingAutotuner.benchmark_all_configs` ranges at all;
      * `Memcpy DtoD` and `cudaMemcpyAsync` at hundreds per step -- Inductor
        cloning mutated arguments between benchmark repetitions;
      * a `cudaDeviceSynchronize` per repetition.

    The eviction fires once per `window_size` steps, so a short warmup holds
    only one or two of them. If a NEW graph is built per eviction, no warmup
    length absorbs it and the recompile axis is the bug, not the warmup.
    """
    ranges = [(_self_device_us(e), e.count, e.key)
              for e in evs if _is_compile_range(e) and _self_device_us(e) > 0]
    counters = _dynamo_counters()
    if not ranges and not counters:
        return
    print("\n  -- COMPILATION INSIDE THE MEASURED WINDOW " + "-" * 31)
    if counters:
        print(f"  dynamo/inductor counters: {counters}")
    if ranges:
        total = sum(us for us, _, _ in ranges)
        print(f"  compile/autotune ranges:  {total / n / 1000:.2f} ms/step "
              f"across {len(ranges)} range(s), EXCLUDED from the totals above.")
        for us, cnt, key in sorted(ranges, reverse=True)[:6]:
            print(f"      {us / n / 1000:8.3f} ms/step  n={cnt / n:7.1f}  "
                  f"{str(key)[:52]}")
        print(f"  -> THIS PROFILE IS NOT A STEADY-STATE STEP. Compilation is "
              f"~{total / max(wall_us, 1.0) * 100:.0f}% of the wall,\n"
              "     so every ms/step above is an average of the step and the "
              "compiler.\n"
              "     Do NOT quote it. Before reaching for --warmup: if a new "
              "graph is built\n"
              "     per eviction, no warmup is long enough and the recompile "
              "axis is the bug.")


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
    ap.add_argument("--compile-evict", type=int, choices=(0, 1), default=None)
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

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=cache,
                        use_cache=True, return_dict=True)
            pkv = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Warm up past compile / autotune / the first eviction.
            for _ in range(args.warmup):
                out = model(input_ids=nxt, past_key_values=pkv,
                            use_cache=True, return_dict=True)
                pkv = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                for _ in range(args.steps):
                    out = model(input_ids=nxt, past_key_values=pkv,
                                use_cache=True, return_dict=True)
                    pkv = out.past_key_values
                    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                torch.cuda.synchronize()
            wall_us = (time.perf_counter() - t0) * 1e6
    finally:
        hooks.remove()

    evs = prof.key_averages()
    # KERNELS ONLY. This used to sum every event, which counted each kernel
    # twice (once under its own name, once under the aten:: op that launched
    # it) and counted compilation ranges as kernels on top. See
    # _is_device_event / _NON_KERNEL_MARKERS.
    kernels = [e for e in evs if _is_device_event(e) and _self_device_us(e) > 0]
    gpu_us = sum(_self_device_us(e) for e in kernels)
    n_launch = sum(e.count for e in kernels)
    # The same work seen from the operator side. Reported, never added.
    op_us = sum(_self_device_us(e) for e in evs
                if not _is_device_event(e) and not _is_compile_range(e)
                and _self_device_us(e) > 0)

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
          f"compile_evict={os.environ.get('STICKYKV_COMPILE_EVICT', '0')}")
    print("=" * 74)
    print(f"  wall            {wall_us / n / 1000:8.2f} ms/step")
    print(f"  CUDA kernels    {gpu_us / n / 1000:8.2f} ms/step")
    if op_us > 0:
        print(f"  (ATen op view   {op_us / n / 1000:8.2f} ms/step  -- the same "
              "kernels seen from\n                            the operator "
              "side. NOT added: key_averages()\n                            "
              "returns both, and summing them double-counts.)")
    busy = gpu_us / max(wall_us, 1.0)
    print(f"  GPU busy        {busy * 100:8.1f} %   <-- THE number")
    # NOTE: `wall_us` is the PROFILED wall, so this UNDER-reads -- CPU+CUDA
    # activity tracing costs tens of microseconds per launch and all of it
    # lands in the denominator. A mixed step can therefore read as host-bound
    # here. Treat `busy` as a lower bound until this times an unprofiled window
    # separately.
    print(f"  kernel launches {n_launch / n:8.0f} /step")
    # A verdict is only printed when the number it rests on is POSSIBLE. Kernel
    # self time cannot exceed wall time on one stream, so busy > 100% does not
    # mean "very busy" -- it means something that is not a kernel is in the
    # numerator, and every number below inherits the error.
    if busy > 1.0:
        print("  -> !! IMPOSSIBLE: kernel self time EXCEEDS the wall. One "
              "stream cannot be\n"
              "        more than 100% busy, so a non-kernel event is being "
              "counted as a\n"
              "        kernel and NOTHING below can be trusted. Check the rows "
              "with\n"
              "        near-zero launches (n~0.0) in the top-N list -- a range "
              "carries the\n"
              "        CUDA time of everything nested in it. Add the offender "
              "to\n"
              "        _NON_KERNEL_MARKERS. No host/kernel verdict for this "
              "run.")
    elif busy < 0.5:
        print("  -> HOST-BOUND. The GPU idles most of the step; launches and "
              "syncs are the budget.")
    elif busy > 0.85:
        print("  -> KERNEL-BOUND. The GPU is saturated; the kernels themselves "
              "are slow. Collapsing launches cannot help.")
    else:
        print("  -> MIXED. Both terms are real; fix the larger one first.")

    _print_compile_contamination(evs, wall_us, n)

    if syncs:
        print("\n  host stalls (each drains the queue and exposes downstream "
              "launch cost):")
        for k, c, us in sorted(syncs, key=lambda r: -r[2])[:8]:
            print(f"    {k:<32} {c / n:7.1f} /step   {us / n / 1000:7.2f} ms/step")

    print(f"\n  top {args.top} kernels by CUDA self time:")
    ranked = sorted(((e, _self_device_us(e)) for e in kernels),
                    key=lambda r: -r[1])
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
