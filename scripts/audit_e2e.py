"""One GPU session that answers all four questions, by bisection.

The four questions -- is prefill 3x faster, is decode 4x faster, does batching
work, does the OOM go away -- have so far been answered by comparing numbers
across separate runs and separate reports, which is how a metric mismatch
(``tpot_ms`` vs ``tpot_steady_ms``) survived long enough to be quoted as a 16%
win. This runs every comparison in ONE process against ONE model load, so the
deltas are differences of like for like.

**The finding this is built to localise.** Against the Flash baseline at
1048/1048, our overhead is:

    batch  1 : prefill +338.0 ms   decode +77.6 ms/step
    batch 32 : prefill +446.0 ms   decode +61.5 ms/step

Splitting that into a batch-proportional part and a fixed part gives 3.5 ms
proportional and **334 ms invariant** in prefill, and -0.5 / **78 ms invariant**
in decode. 99-100% of what we add does not change when the data does. That
rules out -- not "makes unlikely", rules out -- every explanation proportional
to volume: KV bytes, bandwidth, FLOPs, and the compression ratio itself. TPOT
here is not a function of cache size, which is why shrinking the cache has never
moved it. What remains is fixed per-layer cost: launches, Python, host syncs, or
a kernel whose critical path does not depend on B.

**The ladder.** Each rung adds exactly one layer of the design, using only
documented switches -- no logic edits, and every rung is a supported
configuration that produces correct output:

    0  DynamicCache + flash          the floor; none of our code runs
    1  windowed, q=0                 + cache bookkeeping, scoring, eviction
    2  windowed, q>0, fused OFF      + the int2 Q tier via the materialize path
    3  windowed, q>0, fused ON       + the fused Triton decode kernel (shipped)

Rung 3 minus rung 2 is the fused kernel. Rung 2 minus rung 1 is the Q tier.
Rung 1 minus rung 0 is everything else we add. Whichever gap holds the ~78 ms
is the answer, and it is one subtraction, not an inference.

**What the ladder does NOT resolve, and why ``--profile`` exists.** Gap 0->1
contains everything we add outside the Q tier: ``cache.update``, the score
hook, ``compute_lse``, the Triton score kernel and the eviction. ALL 334 ms of
the fixed prefill cost lands in that one gap, undivided, and the decode cost may
too. The ladder would narrow decode to one of three buckets and prefill to a
single bucket that is the whole overhead -- useful, not sufficient.

An env A/B on L-reuse cannot fill that hole either: ``STICKYKV_SCORE_LSE_FROM_
FORWARD=1`` currently MISSES, so both arms would recompute and the delta would
be zero. There is no switch that isolates ``compute_lse``.

``--profile`` does, by op name. ``torch.logsumexp`` appears exactly once in the
codebase -- score_kernel.py:318, inside ``compute_lse`` -- so ``aten::logsumexp``
in a trace is an unambiguous marker for it, and its chain (matmul -> _to_copy ->
mul -> masked_fill -> logsumexp) runs over one ``[B, H_kv, rep, chunk, S]``
tensor, making those op totals directly comparable. Prefill and decode are
profiled SEPARATELY because they are different regimes, and each reports CUDA
self time against wall time -- the number that separates "the host is idle
waiting on kernels" from "the GPU is idle waiting on the host".

Usage:
    # attribution by layer
    python scripts/audit_e2e.py --config outputs/perf_table/_perf_table.generated.yaml \\
        --prefill 1048 --gen 64 --batches 1 32

    # attribution by op, inside the gap the ladder points at
    python scripts/audit_e2e.py --config ... --batches 1 --profile --rungs 3_q70_fused
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Rung definitions. Env is applied BEFORE hooks install, which is when
# STICKYKV_FUSED_DECODE and STICKYKV_SCORE_LSE_FROM_FORWARD are latched.
# --------------------------------------------------------------------------
RUNGS = [
    {"id": "0_baseline", "windowed": False, "q": 0.0, "env": {},
     "what": "DynamicCache + flash — the floor, none of our code runs"},
    {"id": "1_windowed_q0", "windowed": True, "q": 0.0, "env": {},
     "what": "+ cache bookkeeping, scoring, eviction (no quant, no fused kernel)"},
    {"id": "2_q70_materialize", "windowed": True, "q": 0.70,
     "env": {"STICKYKV_FUSED_DECODE": "0"},
     "what": "+ int2 Q tier via the materialize path"},
    {"id": "3_q70_fused", "windowed": True, "q": 0.70,
     "env": {"STICKYKV_FUSED_DECODE": "1"},
     "what": "+ the fused Triton decode kernel (as shipped)"},
]


def _sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _kernel_metadata(torch) -> Dict[str, Any]:
    """Register / spill counts for our two Triton kernels, if they compiled.

    Spills are the cheap decisive check on a hand-written kernel: a decode
    kernel carrying ``[BLOCK_R, HEAD_DIM]`` fp32 accumulators at the default
    4 warps can exceed the register file, and every spilled accumulator update
    then goes to local memory. Seconds to read, and it either implicates the
    kernel or clears it.
    """
    out: Dict[str, Any] = {}
    for modname, attr in (("modules.windowed_cache.decode_kernel",
                           "_two_tier_decode_kernel"),
                          ("modules.windowed_cache.score_kernel",
                           "_score_kernel")):
        try:
            mod = __import__(modname, fromlist=[attr])
            fn = getattr(mod, attr, None)
            cache = getattr(fn, "cache", None)
            if not cache:
                out[attr] = "not compiled (kernel never launched)"
                continue
            entries = []
            for per_device in (cache.values() if isinstance(cache, dict) else []):
                for compiled in (per_device.values()
                                 if isinstance(per_device, dict) else []):
                    entries.append({
                        "n_regs": getattr(compiled, "n_regs", None),
                        "n_spills": getattr(compiled, "n_spills", None),
                        "shared": getattr(compiled, "shared", None),
                        "num_warps": getattr(compiled, "num_warps", None),
                    })
            out[attr] = entries or "compiled, but metadata unavailable"
        except Exception as e:
            out[attr] = f"unavailable: {type(e).__name__}: {e}"
    return out


def _perf_cell(cfg) -> dict:
    """The first ``perf.configs`` entry — where the generated perf config puts
    ``cache_budget`` and the quant settings. They are NOT under ``cfg.cache``,
    so reading them from there silently gives defaults (cache_budget 0.5 instead
    of 0.20) and the ladder measures a different method than the table."""
    try:
        return dict(cfg.perf.configs[0])
    except Exception:
        return {}


def resolve_cache_kwargs(cfg, q: float) -> Dict[str, Any]:
    """WindowedCacheConfig kwargs, resolved EXACTLY as perf_runner resolves them.

    Kept as a pure function so it can be checked without a GPU or a model: this
    is the single place where the ladder could silently diverge from the
    benchmarked method, and a divergence here would not surface as an error —
    it would surface as deltas that quietly fail to add up to the table.

    ``q`` is the ladder's own axis and overrides whatever the config says.
    """
    from utils.config import FIRST_EVICTION_STEP_DEFAULT

    c = _perf_cell(cfg)
    w = cfg.window
    budget = c.get("cache_budget", getattr(cfg.cache, "cache_budget", None))
    return {
        "window_size": int(c.get("window_size", w.window_size)),
        "num_sink_tokens": int(c.get("num_sink_tokens", w.num_sink_tokens)),
        "local_window_size": c.get("local_window_size", w.local_window_size),
        "cache_budget": budget if budget is not None else 0.5,
        "rerotate_on_evict": getattr(cfg.cache, "rerotate_on_evict", False),
        "quant_ratio": q,
        "quant_budget_mode": c.get(
            "quant_budget_mode",
            getattr(cfg.cache, "quant_budget_mode", "tokens")),
        "quant_memoize_read": c.get(
            "quant_memoize_read",
            getattr(cfg.cache, "quant_memoize_read", None)),
        "first_eviction_step": c.get(
            "first_eviction_step",
            getattr(cfg.cache, "first_eviction_step",
                    FIRST_EVICTION_STEP_DEFAULT)),
    }


def _build_cache(cfg, model, prefill_len: int, gen_len: int, q: float,
                 dtype, pkg: str):
    """A fresh windowed cache + hooks, resolved EXACTLY as perf_runner does.

    Mirrors modules/evaluation/perf_runner.py's resolution block field for
    field. Any divergence — a missing ``first_eviction_step``, a different
    ``quant_budget_mode``, a budget read from the wrong section — makes every
    rung measure a method adjacent to the benchmarked one, and the ladder's
    deltas would then not add up to the table they are meant to explain.
    """
    from utils.cache_factory import get_cache_classes
    from utils.config import FIRST_EVICTION_STEP_DEFAULT

    WC, WCC, install_hooks = get_cache_classes(pkg)
    c = _perf_cell(cfg)
    w = cfg.window

    # Two-pass RoPE discovery, as perf_runner does: some models expose the
    # module only as a submodule attribute, and a None rope breaks the cache.
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

    cc = WCC(**resolve_cache_kwargs(cfg, q))
    cache = WC(config=cc, prefill_len=prefill_len, model_config=model.config,
               kv_dtype=dtype, rope_module=rope,
               num_layers=model.config.num_hidden_layers, max_tokens=gen_len)
    return cache, install_hooks(model, cache, cc)


def _time_rung(torch, model, input_ids, rung: dict, cfg, prefill_len: int,
               gen_len: int, dtype, pkg: str, warmup_steps: int) -> Dict[str, Any]:
    """Prefill + steady-state decode for one rung. Never raises: a rung that
    fails records why and the ladder continues, because a partial ladder still
    localises the cost and a dead run localises nothing."""
    from transformers import DynamicCache

    saved = {k: os.environ.get(k) for k in rung["env"]}
    os.environ.update(rung["env"])
    hooks = None
    rec: Dict[str, Any] = {"id": rung["id"], "what": rung["what"],
                           "env": dict(rung["env"])}
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if rung["windowed"]:
            pkv, hooks = _build_cache(cfg, model, prefill_len, gen_len,
                                      rung["q"], dtype, pkg)
        else:
            pkv = DynamicCache()

        with torch.no_grad():
            _sync(torch)
            t0 = time.perf_counter()
            out = model(input_ids=input_ids, past_key_values=pkv,
                        use_cache=True, return_dict=True)
            _sync(torch)
            rec["ttft_s"] = time.perf_counter() - t0

            pkv = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            # Cross the first compaction and any JIT/autotune before timing.
            for _ in range(warmup_steps):
                out = model(input_ids=nxt, past_key_values=pkv,
                            use_cache=True, return_dict=True)
                pkv = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync(torch)

            n_steps = max(gen_len - warmup_steps - 2, 1)
            t1 = time.perf_counter()
            for _ in range(n_steps):
                out = model(input_ids=nxt, past_key_values=pkv,
                            use_cache=True, return_dict=True)
                pkv = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync(torch)
            rec["tpot_steady_s"] = (time.perf_counter() - t1) / n_steps
            rec["decode_steps_timed"] = n_steps

        if torch.cuda.is_available():
            rec["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            rec["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
    except Exception as e:
        # Every crucial detail, on the first failure — see the OOM autopsy.
        rec["failed"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                rec["free_gb_at_failure"] = free / 1024**3
                rec["total_gb"] = total / 1024**3
                rec["memory_summary"] = torch.cuda.memory_summary()
            except Exception:
                pass
    finally:
        if hooks is not None:
            try:
                hooks.remove()
            except Exception:
                pass
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rec


#: ``compute_lse``'s op chain. ``aten::logsumexp`` is UNAMBIGUOUS — torch.logsumexp
#: occurs exactly once in the codebase, inside compute_lse. The rest of the chain
#: runs over the same tensor and is reported alongside it, flagged as shared
#: because matmul/_to_copy/mul are issued by other code too.
_LSE_MARKER = "aten::logsumexp"
_LSE_CHAIN = ("aten::logsumexp", "aten::masked_fill", "aten::_to_copy",
              "aten::mul", "aten::matmul", "aten::bmm")


def _self_device_us(ev) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(ev, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def _profile_phase(torch, label: str, fn, n_iter: int, top: int) -> None:
    """Profile one phase and attribute it by op name.

    Prefill and decode are profiled separately: one is a single large forward,
    the other a stream of tiny ones, and averaging them hides whichever regime
    is the problem.
    """
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    wall_us = (time.perf_counter() - t0) * 1e6

    evs = prof.key_averages()
    gpu_us = sum(_self_device_us(e) for e in evs)
    launches = sum(e.count for e in evs if _self_device_us(e) > 0)
    n = max(n_iter, 1)

    print(f"\n  --- {label} ---")
    print(f"    wall          {wall_us / n / 1000:9.3f} ms")
    print(f"    CUDA kernels  {gpu_us / n / 1000:9.3f} ms")
    busy = gpu_us / max(wall_us, 1.0)
    print(f"    GPU busy      {busy * 100:9.1f} %"
          f"   -> {'HOST-BOUND' if busy < 0.5 else 'KERNEL-BOUND' if busy > 0.85 else 'MIXED'}")
    print(f"    launches      {launches / n:9.0f}")

    lse = {e.key: (_self_device_us(e), e.count) for e in evs if e.key in _LSE_CHAIN}
    marker = lse.get(_LSE_MARKER, (0.0, 0))
    if marker[1]:
        chain_us = sum(v[0] for v in lse.values())
        print(f"    compute_lse   {marker[0] / n / 1000:9.3f} ms in {_LSE_MARKER} "
              f"({marker[1] / n:.0f} calls) — UNAMBIGUOUS marker")
        print(f"                  {chain_us / n / 1000:9.3f} ms across its whole chain "
              f"({chain_us / max(gpu_us, 1) * 100:.1f}% of GPU) — chain ops are shared,"
              f" treat as an upper bound")
    else:
        print(f"    compute_lse   not present in this phase "
              f"(no {_LSE_MARKER} — L-reuse hit, or wrong phase)")

    for title, key in (("CUDA self time", _self_device_us),
                       ("CPU self time", lambda e: float(e.self_cpu_time_total))):
        print(f"    top {top} by {title}:")
        for e in sorted(evs, key=lambda x: -key(x))[:top]:
            if key(e) <= 0:
                break
            print(f"      {key(e) / n / 1000:8.3f} ms  n={e.count / n:7.1f}  {e.key[:52]}")


def _profile_rung(torch, model, input_ids, rung: dict, cfg, prefill_len: int,
                  gen_len: int, dtype, pkg: str, warmup_steps: int,
                  profile_steps: int, top: int) -> None:
    """Op-level attribution for one rung, prefill and decode profiled apart.

    Same env discipline as :func:`_time_rung` — a leaked var here would
    mislabel the attribution as well as the timing.
    """
    from transformers import DynamicCache

    saved = {k: os.environ.get(k) for k in rung["env"]}
    os.environ.update(rung["env"])
    hooks = None
    try:
        print(f"\n  === {rung['id']} — {rung['what']} ===")
        if rung["windowed"]:
            pkv, hooks = _build_cache(cfg, model, prefill_len, gen_len,
                                      rung["q"], dtype, pkg)
        else:
            pkv = DynamicCache()

        with torch.no_grad():
            state = {}

            def _prefill():
                out = model(input_ids=input_ids, past_key_values=pkv,
                            use_cache=True, return_dict=True)
                state["pkv"] = out.past_key_values
                state["nxt"] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            _profile_phase(torch, "PREFILL (1 forward)", _prefill, 1, top)

            for _ in range(warmup_steps):
                out = model(input_ids=state["nxt"], past_key_values=state["pkv"],
                            use_cache=True, return_dict=True)
                state["pkv"] = out.past_key_values
                state["nxt"] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()

            def _decode():
                for _ in range(profile_steps):
                    out = model(input_ids=state["nxt"],
                                past_key_values=state["pkv"],
                                use_cache=True, return_dict=True)
                    state["pkv"] = out.past_key_values
                    state["nxt"] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            _profile_phase(torch, f"DECODE (per step, {profile_steps} steps)",
                           _decode, profile_steps, top)
    finally:
        if hooks is not None:
            try:
                hooks.remove()
            except Exception:
                pass
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _report(results: List[Dict[str, Any]], batch: int) -> None:
    print("\n" + "=" * 78)
    print(f"LADDER  batch={batch}   (each rung adds exactly one layer)")
    print("=" * 78)
    ok = [r for r in results if "tpot_steady_s" in r]
    print(f"  {'rung':<20} {'TTFT (s)':>10} {'TPOT (s)':>10} {'peak GB':>9}   what")
    for r in results:
        if "failed" in r:
            print(f"  {r['id']:<20} {'FAILED':>10} {'—':>10} {'—':>9}   {r['failed'][:44]}")
            continue
        print(f"  {r['id']:<20} {r['ttft_s']:10.4f} {r['tpot_steady_s']:10.4f} "
              f"{r.get('peak_alloc_gb', float('nan')):9.2f}   {r['what']}")

    if len(ok) >= 2:
        print("\n  attribution — each gap is the cost of the layer that rung adds:")
        for a, b in zip(ok, ok[1:]):
            dt = (b["ttft_s"] - a["ttft_s"]) * 1000
            dp = (b["tpot_steady_s"] - a["tpot_steady_s"]) * 1000
            print(f"    {a['id']:>20} -> {b['id']:<20} "
                  f"prefill {dt:+8.1f} ms   decode {dp:+7.2f} ms/step "
                  f"({dp / 32:+5.2f} ms/layer)")
        base, top = ok[0], ok[-1]
        print(f"\n    TOTAL  prefill x{top['ttft_s'] / max(base['ttft_s'], 1e-9):.2f}"
              f"   decode x{top['tpot_steady_s'] / max(base['tpot_steady_s'], 1e-9):.2f}"
              f"   (vs the no-StickyKV floor)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefill", type=int, default=1048)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--warmup-steps", type=int, default=12,
                    help="decode steps before timing; must exceed window_size")
    ap.add_argument("--rungs", nargs="+", default=None,
                    help="subset of rung ids to run (default: all)")
    ap.add_argument("--profile", action="store_true",
                    help="also attribute each rung BY OP NAME, prefill and decode "
                         "separately — the only thing that isolates compute_lse")
    ap.add_argument("--profile-steps", type=int, default=16)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.config import load_config

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — this audit is meaningless on CPU.")

    cfg = load_config(args.config)
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}
    dtype = dtypes.get(cfg.model.dtype, torch.float16)
    pkg = "flash_attn" if cfg.model.attn_implementation != "eager" else "eager"

    print(f"loading {cfg.model.name} (attn={cfg.model.attn_implementation}) ...")
    AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=dtype,
        attn_implementation=cfg.model.attn_implementation, device_map="auto")
    model.eval()

    rungs = RUNGS if args.rungs is None else [
        r for r in RUNGS if r["id"] in set(args.rungs)]

    all_results: Dict[int, List[Dict[str, Any]]] = {}
    for batch in args.batches:
        ids = torch.randint(1000, 20000, (batch, args.prefill),
                            device=model.device, dtype=torch.long)
        results = [_time_rung(torch, model, ids, r, cfg, args.prefill, args.gen,
                              dtype, pkg, args.warmup_steps) for r in rungs]
        all_results[batch] = results
        _report(results, batch)

        if args.profile:
            print("\n" + "=" * 78)
            print(f"OP ATTRIBUTION  batch={batch}")
            print("=" * 78)
            for rung in rungs:
                try:
                    _profile_rung(torch, model, ids, rung, cfg, args.prefill,
                                  args.gen, dtype, pkg, args.warmup_steps,
                                  args.profile_steps, args.top)
                except Exception as e:
                    print(f"\n  {rung['id']}: profiling FAILED — "
                          f"{type(e).__name__}: {e}")
                    print(traceback.format_exc())

    # Batch scaling — question 3, answered by the ladder itself.
    if len(args.batches) >= 2:
        print("\n" + "=" * 78)
        print("BATCH SCALING  (throughput = batch / TPOT; ideal is linear)")
        print("=" * 78)
        lo = min(args.batches)
        for rung in rungs:
            rid = rung["id"]
            pts = []
            for b in args.batches:
                r = next((x for x in all_results[b] if x["id"] == rid), None)
                if r and "tpot_steady_s" in r:
                    pts.append((b, b / r["tpot_steady_s"]))
            if len(pts) >= 2:
                base = next((t for b, t in pts if b == lo), pts[0][1])
                s = "  ".join(f"B={b}: {t:7.1f} tok/s (x{t / base:5.2f})"
                              for b, t in pts)
                print(f"  {rid:<20} {s}")

    print("\n" + "=" * 78)
    print("TRITON KERNEL METADATA  (n_spills > 0 implicates the kernel directly)")
    print("=" * 78)
    print(json.dumps(_kernel_metadata(torch), indent=2, default=str))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in all_results.items()}, f,
                      indent=2, default=str)
        print(f"\nfull results -> {args.json_out}")


if __name__ == "__main__":
    main()
