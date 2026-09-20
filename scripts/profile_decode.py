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

    GPU busy fraction = (CUDA kernel self time) / UNPROFILED wall time

  << 1.0   host-bound. The launch/sync path is the budget; the dispatch-count
           argument was right and the remaining launches are the target.
  ~= 1.0   kernel-bound. The GPU is saturated and the kernels themselves are
           slow. No amount of launch collapsing can help; fix the kernel.

The denominator is load-bearing and was wrong until it was measured separately.
Kernel time comes from the profiler, so the obvious thing is to take the wall
from the same block -- but the profiler charges tens of microseconds per event,
and a decode step that issues ~1,650 launches pays that ~1,650 times. The cost
lands wholly in the denominator (the kernels themselves are unchanged), so it
pushes `busy` down by close to 2x and turns a mixed step into a host-bound
verdict. This script therefore times `steps` steps with the profiler OFF, uses
that as the denominator, and prints the profiled wall beside it as overhead.
Anything comparing `busy` against TPOT must use the unprofiled figure.

Then the A/B that names the kernel, if it is one:

    python scripts/profile_decode.py --config <cfg> --fused 1     # Triton path
    python scripts/profile_decode.py --config <cfg> --fused 0     # materialize

What to read, in order:

1. ``GPU busy %`` -- host-bound or kernel-bound. Everything else is secondary.
2. ``WHERE THE STEP GOES`` -- the bucketed rollup. ``ours (cache)`` vs ``model``
   is the split that decides whether more cache work is worth doing at all.
   ``other`` is always listed by name; if it is large, the fix is a new pattern
   in ``_BUCKETS``, never a subtraction.
3. The flat top-N, for the individual kernel once the bucket says which one.

Usage (the shape the perf table's 4096/256 batch-32 row runs at):

    python scripts/profile_decode.py --config configs/perf_ours.yaml \
        --prefill 4096 --batch 32 --steps 24 --top 40

Add ``--trace out.json`` to get a chrome trace; that is the only thing that can
split the ``shared/other`` bucket, because it carries the launching stack.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _self_device_us(ev) -> float:
    """Per-event CUDA self time. The attribute was renamed in torch 2.x."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(ev, attr, None)
        if v is not None:
            return float(v)
    return 0.0


#: Ordered classification of CUDA kernel names into the buckets the decode
#: question is actually about. FIRST MATCH WINS, so the specific patterns come
#: before the generic ones -- our Triton kernels are named before "elementwise"
#: can claim anything, and the model's GEMMs before "reduction" can.
#:
#: This exists because the flat top-N list below cannot answer "how much of the
#: step is the cache?". Reading it, you subtract the rows you recognise from the
#: total and call the remainder a tail -- which is how a 10.4 ms "unattributed"
#: block got quoted in this repo that was never a block at all, just every kernel
#: ranked below the cut. Hence the rule this table is built around: the `other`
#: bucket is ALWAYS printed by name. A tail you cannot see is a tail you will
#: eventually explain with a guess.
#:
#: Matching is on the kernel name, so it is a heuristic and can mis-file. Two
#: consequences are called out in the output rather than hidden: `memory:` is
#: shared between the cache and the model and is attributed to NEITHER, and
#: anything unmatched is listed individually.
_BUCKETS = (
    # -- ours: the three Triton kernels this project owns ------------------
    ("ours: two-tier decode", ("_two_tier_decode_kernel",)),
    ("ours: read gate",       ("_gate_kernel",)),
    ("ours: prefill score",   ("_score_kernel",)),
    # Inductor names its generated kernels triton_{poi,red,tem,for,unk}_fused_*.
    # Nothing else here is compiled, so these are the eviction body.
    ("ours: compiled evict",  ("triton_poi_fused", "triton_red_fused",
                               "triton_tem_fused", "triton_for_fused",
                               "triton_unk_fused", "triton_mm_fused")),
    # -- the model --------------------------------------------------------
    ("model: attention",      ("flash_fwd", "flash::", "fmha", "mha_fwd",
                               "attention", "scaled_dot")),
    ("model: GEMM",           ("gemm", "gemv", "cutlass", "cublas", "xmma",
                               "sm80_", "sm90_", "ampere_", "turing_",
                               "volta_", "splitKreduce", "dot_kernel")),
    ("model: norm/softmax",   ("layer_norm", "layernorm", "rms_norm",
                               "rmsnorm", "softmax")),
    ("model: activation",     ("silu", "gelu", "swiglu", "sigmoid")),
    # -- shared: cannot be attributed to either side by name ---------------
    ("memory: copy/cat",      ("copy_device_to_device", "direct_copy",
                               "CatArrayBatched", "Memcpy", "Memset",
                               "vectorized_copy")),
    ("memory: index/gather",  ("indexSelect", "index_elementwise",
                               "index_put", "gather", "scatter", "take_",
                               "gatherTopK")),
    ("sort/topk",             ("radix", "bitonic", "sort", "Sort", "topk",
                               "TopK")),
    ("elementwise",           ("elementwise_kernel", "unrolled_elementwise",
                               "CUDAFunctor")),
    ("reduction",             ("reduce_kernel", "ReduceOp", "cub::")),
)

#: Buckets whose time belongs to this project, for the headline split.
_OURS = tuple(name for name, _ in _BUCKETS if name.startswith("ours:"))
#: Buckets that are the model's own work.
_MODEL = tuple(name for name, _ in _BUCKETS if name.startswith("model:"))


def _bucket_of(key: str) -> str:
    """Which bucket a CUDA kernel name falls in. ``"other"`` if none match."""
    low = key.lower()
    for name, pats in _BUCKETS:
        for pat in pats:
            if pat.lower() in low:
                return name
    return "other"


#: The cache this run built, so the rollup can read its counters. One process,
#: one profiled cache; a dict rather than a global rebind keeps it importable.
_PROFILED_CACHE: dict = {"cache": None}
#: The decode-graph runner this run built, so the rollup can report it.
_PROFILED_GRAPH: dict = {"runner": None}


def _print_buckets(evs, n: int, gpu_us: float) -> None:
    """The rollup: where the step's GPU time and launches actually go."""
    agg: dict = {}
    for e in evs:
        us = _self_device_us(e)
        if us <= 0:
            continue
        b = _bucket_of(e.key)
        rec = agg.setdefault(b, {"us": 0.0, "n": 0, "kernels": []})
        rec["us"] += us
        rec["n"] += e.count
        rec["kernels"].append((us, e.count, e.key))

    def share(rec):
        return rec["us"] / max(gpu_us, 1.0) * 100

    ours = sum(agg[b]["us"] for b in _OURS if b in agg)
    model = sum(agg[b]["us"] for b in _MODEL if b in agg)
    shared = gpu_us - ours - model

    print("\n" + "-" * 74)
    print("  WHERE THE STEP GOES")
    print("-" * 74)
    print(f"    {'bucket':<26} {'ms/step':>9} {'% GPU':>7} {'launches/step':>14}")
    for b, _ in _BUCKETS:
        if b not in agg:
            continue
        r = agg[b]
        print(f"    {b:<26} {r['us']/n/1000:9.3f} {share(r):6.1f}% "
              f"{r['n']/n:14.0f}")
    if "other" in agg:
        r = agg["other"]
        print(f"    {'other':<26} {r['us']/n/1000:9.3f} {share(r):6.1f}% "
              f"{r['n']/n:14.0f}")

    print(f"\n    ours (cache)   {ours/n/1000:8.3f} ms/step  "
          f"{ours/max(gpu_us,1)*100:5.1f}%")
    print(f"    model          {model/n/1000:8.3f} ms/step  "
          f"{model/max(gpu_us,1)*100:5.1f}%")
    print(f"    shared/other   {shared/n/1000:8.3f} ms/step  "
          f"{shared/max(gpu_us,1)*100:5.1f}%")
    print("      ^ copy / index / sort / elementwise / reduction, issued by BOTH")
    print("        sides. A kernel name alone cannot say which, so it is")
    print("        attributed to neither. Shrinking it needs the chrome trace")
    print("        (--trace), which carries the launching stack.")

    # The rule this table exists for: `other` is never anonymous.
    if "other" in agg:
        ks = sorted(agg["other"]["kernels"], reverse=True)
        plural = "kernel" if len(ks) == 1 else "kernels"
        print(f"\n    'other' in full ({len(ks)} distinct {plural}) -- if this is "
              "large, add a\n    pattern to _BUCKETS rather than calling it a tail:")
        for us, cnt, key in ks[:12]:
            print(f"      {us/n/1000:8.3f} ms/step  n={cnt/n:7.1f}  {key[:52]}")
        if len(ks) > 12:
            rest = sum(u for u, _, _ in ks[12:])
            print(f"      {rest/n/1000:8.3f} ms/step  ... and {len(ks)-12} more")

    _print_evict_fusion_verdict(agg, n)

    total = sum(r["us"] for r in agg.values())
    if abs(total - gpu_us) > 1.0:   # us; float noise only
        print(f"\n    !! buckets sum to {total/n/1000:.3f} ms but CUDA self time "
              f"is {gpu_us/n/1000:.3f} ms/step -- the rollup is dropping work.")


def _print_evict_fusion_verdict(agg, n: int) -> None:
    """Say whether the eviction FUSED, not merely whether it compiled.

    These are different claims and only the first one was ever reported.
    ``_run_compiled_evict`` is kernel-or-error, so a run that finishes proves the
    compiled callable ran -- and the banner duly prints ``COMPILED ... ACTIVE
    [OK]``. It does not prove Inductor generated anything. A body that traces,
    graph-breaks, and lowers each fragment back to stock ATen kernels satisfies
    every check this repo had, reports itself as compiled, and delivers none of
    what compiling is for.

    That is what a 2026-09-16 GPU profile showed: ``ours (cache)`` came to
    7.917 ms against ``_two_tier_decode_kernel`` 7.104 + ``_gate_kernel`` 0.813
    -- the two hand-written Triton kernels to the milligram, so ``ours: compiled
    evict`` contributed exactly 0.000. The eviction was landing as
    ``at::native::elementwise_kernel`` and friends with fractional launch counts
    (n < 5/step, i.e. one step in ``window_size``), which is the unfused
    signature the design notes describes.

    Root cause was eight ``untyped_storage().data_ptr()`` graph breaks in
    now pins the count at zero. This printer is the other half: the instrument
    that would have said so at the time.
    """
    try:
        from modules.windowed_cache.cache import evict_path_stats
    except Exception:  # pragma: no cover - import-path dependent
        return
    stats = evict_path_stats()
    compiled_runs, eager_runs = int(stats.get("compiled", 0)), int(stats.get("eager", 0))
    arm_runs = int(stats.get("control_arm", 0))
    if arm_runs:
        # The CONTROL ARM. Not a fault, and not a fused/unfused question at all
        # -- there is nothing to fuse. Reported in its own words so a reference
        # run can never be read as a broken shipped run, which is the mistake
        # this whole printer exists to prevent in the other direction.
        print(f"\n    eviction: CONTROL ARM -- {arm_runs} eager run(s), no "
              "compile, no autotune.\n"
              "      This is the REFERENCE arm (--compile-evict 0). Its ms/step "
              "is what the\n"
              "      compiled eviction has to beat. Compare against the same "
              "command with\n"
              "      --compile-evict 1 and state the claim against it; a tie "
              "means the\n"
              "      compile is buying nothing (DISTANCE_TO_GOAL §10.14 found "
              "exactly that).")
        if compiled_runs:
            print("      !! MIXED: compiled runs in a control-arm window "
                  f"({compiled_runs}). Not a clean arm.")
        return
    if compiled_runs == 0 and eager_runs == 0:
        print("\n    eviction: NO EVICTION RAN in the profiled window. Raise "
              "--steps above window_size, or this profile is not measuring the "
              "eviction at all.")
        return

    fused_us = agg.get("ours: compiled evict", {}).get("us", 0.0)
    fused_n = agg.get("ours: compiled evict", {}).get("n", 0.0)
    print(f"\n    eviction: {compiled_runs} compiled / {eager_runs} eager runs, "
          f"Inductor kernels {fused_us/n/1000:.3f} ms/step over {fused_n/n:.0f} "
          "launches/step")
    if compiled_runs > 0 and fused_us <= 0.0:
        print("      !! COMPILED BUT NOT FUSED. The compiled callable ran and "
              "Inductor\n"
              "         emitted no kernel this rollup can see. The eviction is "
              "paying\n"
              "         compile overhead for eager kernels. Check:\n"
              "         and bound the cost with one --compile-evict 0 run "
              "(the control\n"
              "         arm, which now exists): if eager and 'compiled' tie, "
              "the compile is\n"
              "         doing nothing either way.")
    elif eager_runs > 0 and compiled_runs > 0:
        print("      !! MIXED eviction paths in one window -- the ms/step above "
              "is an\n         average of two methods. Re-run with "
              "--compile-evict pinned.")


#: Substrings that mark a PROFILER RANGE rather than a kernel. These are
#: rejected before any other test, including ``device_type``.
#:
#: Why a hard reject and not a device-type test: `torch._dynamo.utils.
#: dynamo_timed` opens a `record_function` range around compilation, and on the
#: 2026-09-19 A100 profile those ranges came back carrying CUDA self time and
#: passed BOTH arms of `_is_device_event` -- so `compile_fx_inner` was counted
#: as a 268 ms/step "kernel" with **n = 0.0 launches**, and five such rows made
#: up 726.824 ms/step of the `other` bucket. A range's self device time is the
#: CUDA time of everything nested inside it, so adding it to the kernels it
#: encloses double-counts them: exactly the failure this function was written
#: for (see the aten::mm case below), one level up.
#:
#: The damage was not cosmetic. It pushed `GPU busy` to **203.9%** -- which is
#: impossible on one stream, and is the tell -- and the script then printed
#: "KERNEL-BOUND ... the kernels themselves are slow" for a step whose real
#: kernel time was 345 ms of a 526 ms wall (65.6% busy, i.e. a ~180 ms host
#: gap from the autotuner's per-rep `cudaDeviceSynchronize`). Acting on that
#: verdict sends you into the two kernels, which were the only part of that
#: profile that had got FASTER.
#:
#: Kernels launched *inside* one of these ranges are unaffected: they appear
#: under their own names as their own rows and are still counted. Only the
#: enclosing range is dropped.
_NON_KERNEL_MARKERS = (
    "(dynamo_timed)",           # every row observed on the 2026-09-19 profile
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
    :data:`_NON_KERNEL_MARKERS`). Reported separately -- a window containing
    these is not measuring a steady-state step."""
    return any(m in str(ev.key) for m in _NON_KERNEL_MARKERS)


def _is_device_event(ev) -> bool:
    """True for a CUDA KERNEL event; false for the ATen op that launched it.

    ``key_averages()`` returns BOTH, and both carry self device time. A leaf op's
    self device time is the time of the kernels it launched, and those kernels
    then report it again under their own names. Summing the lot double-counts
    every kernel in the step.

    This was live in every number this script has ever printed. It surfaced when
    the rollup listed its `other` bucket by name and `aten::mm` came back at
    11.052 ms / 225.0 launches per step against four `ampere_*gemm*` kernels
    summing to 10.831 ms / **exactly** 225.0. Same launches, same time, counted
    twice -- which inflated `CUDA kernels`, `kernel launches`, and therefore the
    `GPU busy %` this whole script exists to report, by close to 2x.

    ``device_type`` is the real signal. The name check is only a fallback for a
    build that does not expose it: the operator view is namespaced (``aten::``,
    ``autograd::``), device events are not. Memcpy/Memset are device events and
    are deliberately NOT excluded by it.
    """
    # FIRST, and ahead of device_type: a compilation range is never a kernel,
    # whatever the profiler tags it as. See _NON_KERNEL_MARKERS for the profile
    # that made this necessary and what it did to `GPU busy`.
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
        st = dict(counters.get("stats", {}))
        return {k: int(v) for k, v in st.items()}
    except Exception:  # pragma: no cover - torch-version dependent
        return {}


def _print_compile_contamination(evs, steady_us: float, n: int) -> None:
    """Say, out loud, if COMPILATION happened inside the measured window.

    This is the half of the instrument that was missing. `perf_runner` prints
    Dynamo's counters for its measured runs; this script printed nothing, so a
    window in which Inductor was still compiling and autotuning looked exactly
    like a steady-state window -- only ~10x slower, with no line saying why.

    That is not hypothetical. The 2026-09-19 profile at 4096/B=32 reported
    525.81 ms/step against the 45.69 ms/step the SAME command produced days
    earlier, and the reason was visible only because the compile ranges had
    leaked into the kernel table (see _NON_KERNEL_MARKERS). Now that they are
    correctly excluded from the kernels, they would vanish from the output
    entirely -- and a 10x-slow profile would carry no explanation at all. So
    they are excluded from the kernels AND reported here.

    The signature to recognise, all of it visible in that profile:
      * `CachingAutotuner.benchmark_all_configs` ranges at all;
      * `Memcpy DtoD` and `cudaMemcpyAsync` at hundreds per step -- Inductor
        cloning mutated arguments between benchmark repetitions;
      * a `cudaDeviceSynchronize` per repetition (196 ms/step, there).

    What to do about it is NOT "raise --warmup" by reflex. The eviction fires
    once per `window_size` steps, so a 12-step warmup contains one or two of
    them; if a fresh graph is being built for each, no warmup is long enough
    and the recompile axis is the bug (DISTANCE_TO_GOAL §10.10 named `T_fp` and
    `W`, which are `int()`-forced and therefore specialise even under
    `dynamic=True`). The message says both.
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
              f"across {len(ranges)} range(s), EXCLUDED from the kernel totals "
              "above.")
        for us, cnt, key in sorted(ranges, reverse=True)[:6]:
            print(f"      {us / n / 1000:8.3f} ms/step  n={cnt / n:7.1f}  "
                  f"{str(key)[:52]}")
        share = total / max(steady_us, 1.0)
        print(f"  -> THIS PROFILE IS NOT A STEADY-STATE STEP. Compilation is "
              f"~{share * 100:.0f}% of the\n"
              "     unprofiled wall, so every ms/step below is an average of "
              "the step and\n"
              "     the compiler. Do NOT quote it, and do not read the kernel "
              "ranking as a\n"
              "     ranking of the step's real costs.\n"
              "     Before reaching for --warmup: the eviction fires once per "
              "window_size\n"
              "     steps, so a short warmup holds only one or two of them. If "
              "a NEW graph is\n"
              "     built per eviction, no warmup is long enough and the "
              "recompile axis is\n"
              "     the bug -- see DISTANCE_TO_GOAL.md §10.10 (T_fp and W are "
              "int()-forced,\n"
              "     so they specialise even under dynamic=True).")


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
    ap.add_argument("--compile-evict", type=int, choices=(0, 1), default=1,
                    help="1 (default) = the shipped compiled eviction. "
                         "0 = the CONTROL ARM: the same body run eagerly, so "
                         "the compiled path has a reference to be measured "
                         "against. The rollup below has advertised this flag "
                         "since before it existed; it exists now. A 0 run "
                         "reports eager numbers and says so.")
    ap.add_argument("--warmup", type=int, default=12,
                    help="decode steps before profiling; must exceed window_size "
                         "so at least one eviction is compiled/autotuned away")
    # Defaults to 1, which is what the library itself picks on CUDA (the device
    # decides -- see _compile_evict_enabled). This used to be the one place the
    # two disagreed: the library defaulted to 0 while run_perf_table.sh exported
    # 1, so an unset profile attributed the EAGER eviction against a table that
    # ran it compiled -- a different method, and not by a little: compiling moved
    # TPOT 101.6 -> 85 ms (DECODE_HISTORY.md §1). Pass 0 deliberately to A/B it.
    ap.add_argument("--trace", default=None, help="write a chrome trace here")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()


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

    # The control arm, set BEFORE any eviction: the compiled callable is built
    # lazily on first use, and the point of the arm is that it is never built,
    # so the run pays no compile and no autotune. That is the quantity being
    # measured.
    try:
        from modules.windowed_cache.cache import set_evict_control_arm
        set_evict_control_arm(args.compile_evict == 0)
    except Exception:  # pragma: no cover - import-path dependent
        if args.compile_evict == 0:
            raise

    # -- cache + hooks: same construction as scripts/diagnose_perf.py --------
    from utils.cache_factory import get_cache_classes
    WindowedCache, WindowedCacheConfig, install_score_hooks = get_cache_classes(pkg)

    rope = None
    for name, mod in model.named_modules():
        if "rotary" in name.lower() or "rope" in name.lower():
            rope = mod
            break

    # Deliberately counts ONE window of `steps`, not the two that actually run.
    # `max_tokens` is not a capacity: the buffers are sized to the eviction
    # budget (cache.py's "Sized to the EVICTION BUDGET" note), and the only thing
    # this value reaches is `config.resolve`, where the budget is taken against
    # `prefill_len + max_tokens`. Counting the second (unprofiled) window here
    # would widen the budget by ~0.6% and quietly profile a different method than
    # every earlier run of this script. The extra steps cost budget nothing --
    # they evict against the same target like any other step.
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
    # The rollup is printed by a module-level function, so it needs a handle to
    # the live cache to read its eviction-width counters.
    _PROFILED_CACHE["cache"] = cache

    # cache_position must be passed EXPLICITLY and advanced monotonically. Left
    # to itself, transformers derives it from `past_key_values.get_seq_length()`,
    # which for an evicting cache is the RETAINED key count, not the absolute
    # token index — so it jumps backwards after the first eviction and the store
    # ends up with duplicated, non-monotonic positions. `generate()` does this
    # for us, which is why every quality runner is unaffected and only this
    # hand-written decode loop trips the cache's position contract.
    device = input_ids.device
    pos = 0                      # absolute token index, advanced by _step

    _decode_step = [-1]

    def _step(ids, past, n_new):
        nonlocal pos
        cp = torch.arange(pos, pos + n_new, device=device)
        pos += n_new
        _decode_step[0] += 1
        return model(input_ids=ids, past_key_values=past, use_cache=True,
                     return_dict=True, cache_position=cp)

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

            # ---- window 1: the STEADY wall, with the profiler OFF ------------
            # `GPU busy` is kernel time over wall time, and the only wall that
            # answers the question is the one the step costs when nothing is
            # watching. Timing it inside the `profile` block instead charges the
            # denominator for the profiler's own per-event cost -- which on this
            # workload is not a rounding error: CPU+CUDA activity tracing pays
            # roughly tens of microseconds per launch, and a step that issues
            # ~1,650 of them absorbs tens of milliseconds it does not otherwise
            # spend. That lands entirely in the denominator and nowhere in the
            # numerator, so it drives `busy` DOWN and makes a mixed step read as
            # host-bound -- the one reading this script exists to rule on.
            #
            # The counters are snapshotted ACROSS this window, not just at the
            # end. `busy` divides window 2's kernel time by window 1's wall, so
            # the two windows have to be the same KIND of step. If Inductor
            # compiles in window 1 and is done by window 2, they are not: the
            # 2026-09-20 run measured window 1 at 2357.34 ms/step and window 2
            # at 136.32, and `busy` came out at 1.7% with a confident
            # "HOST-BOUND. The GPU idles most of the step" underneath. The GPU
            # was not idling; window 1 was compiling 18 graphs. Nothing in the
            # kernel table could show it, because the compile happened in the
            # window the profiler was not watching.
            dyn_before = _dynamo_counters()
            t0 = time.perf_counter()
            for _ in range(args.steps):
                out = _step(nxt, pkv, 1)
                pkv = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            steady_us = (time.perf_counter() - t0) * 1e6
            dyn_after = _dynamo_counters()

            # ---- window 2: the attribution, with the profiler ON -------------
            # Same length as window 1 so the two see the same number of eviction
            # steps, which are ~2.7x a steady step and would otherwise bias
            # whichever window held more of them.
            #
            # Zero the eviction path counters here, not at startup: the verdict
            # printed with the rollup has to describe the window the kernels were
            # measured in, and warmup alone runs several evictions.
            try:
                from modules.windowed_cache.cache import reset_evict_path_stats
                reset_evict_path_stats()
            except Exception:  # pragma: no cover - import-path dependent
                pass
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
    kernels = [e for e in evs if _is_device_event(e) and _self_device_us(e) > 0]
    gpu_us = sum(_self_device_us(e) for e in kernels)
    n_launch = sum(e.count for e in kernels)
    # The same work seen from the operator side. Reported, never added -- see
    # _is_device_event for what adding it did to every number below.
    op_us = sum(_self_device_us(e) for e in evs
                if not _is_device_event(e) and _self_device_us(e) > 0)

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
          f"steps={n}")
    print("=" * 74)
    print(f"  wall            {steady_us / n / 1000:8.2f} ms/step   "
          "(profiler OFF -- the real step)")
    print(f"  wall, profiled  {wall_us / n / 1000:8.2f} ms/step   "
          f"({(wall_us - steady_us) / n / 1000:+.2f} ms of profiler overhead, "
          "NOT the step's cost)")
    print(f"  CUDA kernels    {gpu_us / n / 1000:8.2f} ms/step")
    if op_us > 0:
        print(f"  (ATen op view   {op_us / n / 1000:8.2f} ms/step  -- the same "
              "kernels seen from the\n                            operator side. "
              "NOT added: key_averages()\n                            returns "
              "both, and summing them double-counts.)")
    busy = gpu_us / max(steady_us, 1.0)
    print(f"  GPU busy        {busy * 100:8.1f} %   <-- THE number")
    print(f"  (against the profiled wall it would read "
          f"{gpu_us / max(wall_us, 1.0) * 100:.1f}% -- that figure is an "
          "artifact\n   of the measurement and must not be quoted.)")
    print(f"  kernel launches {n_launch / n:8.0f} /step")
    # A verdict is only printed when the number it rests on is POSSIBLE. Kernel
    # self time cannot exceed wall time on one stream, so busy > 100% does not
    # mean "very busy" -- it means something that is not a kernel is in the
    # numerator, and every bucket below inherits the error. The 2026-09-19
    # profile read 203.9% and still printed "KERNEL-BOUND ... the kernels
    # themselves are slow"; the kernels were in fact the only thing that had
    # improved. Refusing the verdict is the whole fix.
    # The SAME class of failure as busy > 100%, in the other direction, and it
    # needs its own guard because the ratio's two halves come from two
    # different windows. `gpu_us` is window 2; `steady_us` is window 1. If they
    # are not the same kind of step the ratio is meaningless, and it fails
    # SILENTLY -- a low `busy` reads as a discovery ("host-bound!"), not as an
    # error. Two tells, both cheap:
    #
    #   * profiling made the step FASTER. Tracing only ever adds cost, so a
    #     negative overhead means window 1 was doing something window 2 was
    #     not;
    #   * Dynamo's graph count MOVED during window 1. That is compilation
    #     inside the denominator.
    #
    # The 2026-09-20 run at 4096/B=32 hit both: 2357.34 ms/step unprofiled
    # against 136.32 profiled, 18 unique graphs built, and `GPU busy 1.7% ->
    # HOST-BOUND. The GPU idles most of the step`. Window 2 was clean and good
    # (40.51 ms of kernels over 1726 launches); only the denominator was junk.
    compiled_in_w1 = {k: dyn_after.get(k, 0) - v
                      for k, v in (dyn_before or {}).items()
                      if dyn_after.get(k, 0) != v}
    for k, v in (dyn_after or {}).items():
        if k not in (dyn_before or {}):
            compiled_in_w1[k] = v
    windows_differ = wall_us < steady_us or bool(compiled_in_w1)
    if windows_differ:
        print("  -> !! THE TWO WINDOWS ARE NOT COMPARABLE, so `GPU busy` above "
              "is NOT a\n"
              "        measurement. Kernel time comes from the profiled "
              "window and the wall\n"
              "        from the unprofiled one; the ratio only means something "
              "when both\n"
              "        ran the same kind of step.")
        if wall_us < steady_us:
            print(f"        * profiling made the step FASTER by "
                  f"{(steady_us - wall_us) / n / 1000:.2f} ms/step. Tracing "
                  "only ever ADDS\n"
                  "          cost, so the unprofiled window was doing "
                  "something the profiled one\n"
                  "          was not.")
        if compiled_in_w1:
            print(f"        * Dynamo compiled during the unprofiled window: "
                  f"{compiled_in_w1}.\n"
                  "          That is compilation inside the denominator.")
        print("        The kernel table below is still valid -- it is the "
              "profiled window\n"
              "        alone. Only `GPU busy` and the wall are spoiled. Raise "
              "--warmup until\n"
              "        no graphs are built here, then re-read busy.")
    elif busy > 1.0:
        print("  -> !! IMPOSSIBLE: kernel self time EXCEEDS the unprofiled "
              "wall. One stream\n"
              "        cannot be more than 100% busy, so a non-kernel event is "
              "being counted\n"
              "        as a kernel and NO bucket below can be trusted. Check "
              "the rows with\n"
              "        near-zero launches (n~0.0) in the top-N list -- a range "
              "carries the\n"
              "        CUDA time of everything nested in it. Add the offender "
              "to\n"
              "        _NON_KERNEL_MARKERS. No host/kernel verdict is printed "
              "for this run.")
    elif busy < 0.5:
        print("  -> HOST-BOUND. The GPU idles most of the step; launches and "
              "syncs are the budget.")
    elif busy > 0.85:
        print("  -> KERNEL-BOUND. The GPU is saturated; the kernels themselves "
              "are slow. Collapsing launches cannot help.")
    else:
        print("  -> MIXED. Both terms are real; fix the larger one first.")

    _print_compile_contamination(evs, steady_us, n)

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
    except Exception:
        pass

    # The tile rung, asked for rather than caught. The kernel prints its choice
    # once per geometry, which happens in warmup and scrolls away above whatever
    # is being read -- so the one line DECODE_NEXT.md §5 step 1 says to read was
    # the one line a profile did not carry.
    def _print_search(title, choice_fn, timings_fn, fmt_sig, fmt_rung):
        """One tuner's verdict AND its margin. Never swallowed silently.

        This block used to sit inside a bare ``except Exception: pass`` while
        importing a ``fit_choice`` that did not exist -- so the one line the
        tuning step needs was the one line a profile never carried, and nothing
        said why. A diagnostic must not fail the run, but it must say when it
        cannot answer.
        """
        try:
            chosen, timed = choice_fn(), timings_fn()
            if not chosen:
                print(f"\n  {title}: nothing tuned (the kernel never ran)")
                return
            print(f"\n  {title}:")
            # Formatting is INSIDE the try too. A signature's arity is part of
            # the tuner's contract, so a stale or renamed one unpacks wrong here
            # -- and a diagnostic that kills the run it is diagnosing is worse
            # than one that says it cannot answer.
            for sig, rung in sorted(chosen.items(), key=lambda kv: str(kv[0])):
                ranked = timed.get(sig, [])
                best_ms = ranked[0][1] if ranked else None
                # The margin is the point: a rung that won by 30% and one that
                # won by 0.5% call for different next moves.
                margin = ""
                if len(ranked) > 1 and best_ms:
                    margin = (f"   ({100.0 * (ranked[1][1] - best_ms) / best_ms:.1f}% "
                              f"clear of {fmt_rung(ranked[1][0])})")
                ms = f"  {best_ms:.3f} ms" if best_ms else ""
                print(f"    {fmt_sig(sig)}  ->  {fmt_rung(rung)}{ms}{margin}")
                for r, t in ranked:
                    mark = " *" if r == rung else "  "
                    print(f"        {mark} {fmt_rung(r):<18} {t:.3f} ms")
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"\n  {title}: unavailable ({type(exc).__name__}: {exc})")

    try:
        from modules.windowed_cache import decode_kernel as _dk
        from modules.windowed_cache import gate_kernel as _gk
    except Exception as exc:  # pragma: no cover - diagnostics only
        _dk = _gk = None
        print(f"\n  tile searches: unavailable ({type(exc).__name__}: {exc})")

    if _dk is not None:
        _print_search(
            "decode tiling (target_keys x num_stages x num_warps), per geometry",
            _dk.fit_choice, _dk.fit_timings,
            lambda s: (f"ws={s[0]} head_dim={s[1]} BLOCK_R={s[2]} "
                       f"q_tier={bool(s[3])} gated={s[4]} B={s[5]} "
                       f"Sfp~2^{s[6]} n_sel~2^{s[7]}"),
            lambda r: f"{r[0]}x{r[1]}x{r[2]}w",
        )
        _print_search(
            "read gate tiling (BLOCK_W x num_warps), per geometry",
            _gk.gate_choice, _gk.gate_timings,
            lambda s: (f"NW={s[0]} H_kv={s[1]} head_dim={s[2]} ws={s[3]} "
                       f"B={s[4]} rep={s[5]}"),
            lambda r: f"BLOCK_W={r[0]}x{r[1]}w",
        )

    # What the eviction's worst-case width actually bought (DECODE_NEXT.md D1).
    # Printed here, after both timed windows, because reading it syncs once.
    try:
        from modules.windowed_cache.cache import evict_width_stats
        w = evict_width_stats(_PROFILED_CACHE["cache"])
        if w:
            print(f"\n  eviction widths over {w['evictions']} evictions "
                  f"(n_q={w['n_q']}, W_retained={w['W_retained']}):")
            print(f"    fresh (quantize+sketch)  max={w['fresh_max']:4d}  "
                  f"mean={w['fresh_mean']:6.2f}  of n_q={w['n_q']}"
                  + (f"   fill={w['fresh_fill']:.1%}"
                     if w["fresh_fill"] is not None else ""))
            print(f"    promote (dequant+RoPE)   max={w['promote_max']:4d}  "
                  f"mean={w['promote_mean']:6.2f}"
                  + (f"   fill={w['promote_fill']:.1%}"
                     if w.get("promote_fill") is not None else ""))
            print(f"    reactivate (free)        max={w['react_max']:4d}")
            # Both sides are now sized by the measured max rather than by their
            # cap, so these fills are what the OLD code wasted, not what the
            # current one does. A low fill means the saving is large.
            for label, fill, cap, work in (
                ("demote", w["fresh_fill"], w["n_q"],
                 "un-rotates, quantizes and sketches"),
                ("promote", w.get("promote_fill"), w["n_q"],
                 "dequantizes, RoPEs and splices"),
            ):
                if fill is None:
                    continue
                if fill < 0.5:
                    print(f"    -> D1 ({label}) is live: sizing by the cap "
                          f"would {work}\n       [L*B, {cap}, H_kv, ws, D] "
                          f"per eviction and mask away {1 - fill:.0%} of it.")
                else:
                    print(f"    -> D1 ({label}) buys little here: the cap is "
                          "already close to the real count.")
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        pass

    if _PROFILED_GRAPH["runner"] is not None:
        print(f"\n  {_PROFILED_GRAPH['runner'].report()}")

    if syncs:
        print("\n  host stalls (each drains the queue and exposes downstream "
              "launch cost):")
        for k, c, us in sorted(syncs, key=lambda r: -r[2])[:8]:
            print(f"    {k:<32} {c / n:7.1f} /step   {us / n / 1000:7.2f} ms/step")

    _print_buckets(kernels, n, gpu_us)

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
