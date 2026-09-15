"""How many DISTINCT Q windows does a generation actually need?

The digest gate shows that ~75% of the int2 tier goes unread on any **given**
decode step. That alone says nothing about whether the tier is oversized: if
every step wants a *different* 110 of 439 windows, all 439 are still needed and
none can be dropped. The question this script answers is the union:

    over the steps of one eviction epoch, how many distinct windows were admitted?

    union_frac ~ 1.0  -> every resident window was wanted by some step. The tier
                         is not oversized; per-step sparsity cannot become a
                         memory saving, and the "different 110 each time" story
                         is confirmed rather than assumed.
    union_frac  small -> `N_q` could shrink outright. That is a real memory win
                         needing NO recall path, which matters because
                         DIGEST_GATED_DECODE_PLAN.md §1.3 rules recall out.

**Why per epoch.** Eviction rebuilds the resident set, so a union taken across
evictions would grow for a trivial reason — new windows entered the tier, not
that more windows were needed. Within an epoch the Q tier is frozen (design §10),
so union / resident is a clean ratio. Epochs with a single step are dropped: their
union is `k` by construction and shows no reuse.

Usage:
    PYTHONPATH=. STICKYKV_DIGEST_TRACE=1 python scripts/digest_union.py \
        --config configs/eval_perf_digest_gated.yaml --prefill 7105 --steps 32

Tracing syncs once per layer per step. This script is never a timing measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _perf_cell_quant(cfg) -> float:
    c = dict(cfg.perf.configs[0])
    return float(c.get("quant_ratio", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefill", type=int, default=7105)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--json", default=None, help="write the full per-epoch table here")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required: the gate lives in the fused decode path.")

    from utils.config import load_config
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_e2e import resolve_backend, resolve_cache_kwargs
    from modules.windowed_cache import flash_decode

    if not flash_decode.trace_enabled():
        raise SystemExit(
            "STICKYKV_DIGEST_TRACE is not set, so nothing would be recorded. "
            "Re-run with STICKYKV_DIGEST_TRACE=1.")

    cfg = load_config(args.config)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(cfg.model.dtype, torch.float16)
    pkg, attn_impl = resolve_backend(cfg)

    print(f"loading {cfg.model.name} (attn={attn_impl}, cache_package={pkg}) ...")
    AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=dtype,
        attn_implementation=attn_impl, device_map="auto").eval()

    from utils.cache_factory import get_cache_classes
    WindowedCache, WindowedCacheConfig, install_score_hooks = get_cache_classes(pkg)

    rope = None
    for name, mod in model.named_modules():
        if "rotary" in name.lower() or "rope" in name.lower():
            rope = mod
            break

    cache_config = WindowedCacheConfig(
        **resolve_cache_kwargs(cfg, _perf_cell_quant(cfg), pkg))
    if cache_config.digest_gate_frac is None:
        raise SystemExit(
            "this config has no digest_gate_frac, so nothing is ever gated and "
            "there is no selection to trace. Use configs/eval_perf_digest_gated.yaml.")

    cache = WindowedCache(
        config=cache_config, prefill_len=args.prefill, model_config=model.config,
        kv_dtype=dtype, rope_module=rope,
        num_layers=model.config.num_hidden_layers, max_tokens=args.steps + 1)
    hooks = install_score_hooks(model, cache, cache_config)

    ids = torch.randint(1000, 20000, (args.batch, args.prefill),
                        device=model.device, dtype=torch.long)
    flash_decode.trace_reset()

    # cache_position explicitly and monotonically: an evicting cache's
    # get_seq_length() is the RETAINED count, not the token index.
    pos = 0

    def step(x, past, n_new):
        nonlocal pos
        o = model(input_ids=x, past_key_values=past, use_cache=True,
                  return_dict=True,
                  cache_position=torch.arange(pos, pos + n_new, device=model.device))
        pos += n_new
        return o

    try:
        with torch.no_grad():
            out = step(ids, cache, ids.shape[1])
            pkv = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            for _ in range(args.steps):
                out = step(nxt, pkv, 1)
                pkv = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    finally:
        hooks.remove()

    summary = flash_decode.trace_summary()
    rows, agg = summary["rows"], summary["aggregate"]
    if not rows:
        raise SystemExit(
            "no multi-step epoch was recorded. Either the gate never engaged "
            "(k >= n_active), or every eviction epoch was one step long — try "
            "more --steps.")

    print()
    print("=" * 78)
    print(f"DISTINCT-WINDOW UNION  prefill={args.prefill} batch={args.batch} "
          f"steps={args.steps}")
    print("=" * 78)
    print(f"  epoch-layer pairs      {agg['n_epoch_layer']}")
    print(f"  steps per epoch (mean) {agg['mean_steps_per_epoch']:.1f}")
    print(f"  k / resident           {agg['mean_k_frac']:.3f}   <- read per STEP")
    print(f"  union / resident       {agg['mean_union_frac']:.3f}   <- THE number")
    print(f"    min / max            {agg['min_union_frac']:.3f} / {agg['max_union_frac']:.3f}")
    print()
    u = agg["mean_union_frac"]
    if u > 0.95:
        print("  => Essentially EVERY resident window was wanted by some step.")
        print("     Per-step sparsity does NOT make the tier oversized; N_q cannot")
        print("     be reduced on this evidence, and the memory door is closed.")
    elif u < 0.6:
        print(f"  => Only {u:.0%} of the tier was EVER wanted. N_q looks oversized:")
        print(f"     retaining ~{u:.0%} of it needs no recall path and is a real")
        print("     memory saving. Worth testing directly by lowering quant_ratio.")
    else:
        print(f"  => {u:.0%} of the tier was wanted at least once. Partial reuse;")
        print("     a smaller N_q would cost some windows that were genuinely used.")
    print()
    print("  per-epoch detail (first 10):")
    print(f"    {'epoch':>6} {'layer':>6} {'steps':>6} {'resident':>9} "
          f"{'k':>5} {'union':>6} {'union/res':>10}")
    for r in rows[:10]:
        print(f"    {r['epoch']:>6} {r['layer']:>6} {r['steps']:>6} "
              f"{r['resident']:>9} {r['k_per_step']:>5} {r['union']:>6} "
              f"{r['union_frac']:>10.3f}")

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, default=list))
        print(f"\n  full table -> {args.json}")


if __name__ == "__main__":
    main()
