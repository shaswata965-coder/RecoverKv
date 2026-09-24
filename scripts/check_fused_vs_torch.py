"""Does the fused two-tier decode kernel compute attention? Checked against PyTorch.

Every GPU check of the decode path so far compared the fused kernel with ITSELF
(gate 0.25 vs sel=None through the same kernel), so a kernel that is wrong on
the GPU -- bf16, real magnitudes, Qwen2's anchored keys, rep 7 -- passes all of
them. The Triton interpreter checks ran fp32 on synthetic caches only.

This runs the normal LongBench path on a few examples and, for the first
``--max-steps`` decode steps of every fused layer, recomputes the same step in
plain PyTorch from the same cache:

    K = [ fp tier (as flash received it, GQA un-repeated) | store.effective_q_tier ]
    out_ref = softmax(scale * q K^T) V            (fp32)

and compares it with the kernel at ``sel=None`` (every window read) and with
the step the model actually used. ``effective_q_tier`` is the store's own
PyTorch dequantize + re-rotate, i.e. QEvict's read path.

    python scripts/check_fused_vs_torch.py \\
        --config configs/longbench_qwen_ours_gate100.yaml --dataset triviaqa

Reading it: ``kernel_vs_torch`` should be ~1e-3 (bf16 inputs, TF32 dots). A
layer at 1e-1 or worse is a kernel that does not compute attention there. Run
the Llama config too: a Qwen-only failure points at the paths only Qwen takes
(KEY_ANCHOR, bf16 tables, the GQA un-repeat).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install(max_steps: int):
    import torch
    from modules.windowed_cache import flash_decode
    from modules.windowed_cache.decode_kernel import fused_two_tier_decode

    orig = flash_decode._run_fused
    stats = defaultdict(lambda: defaultdict(list))
    seen = {"cache": None, "step": 0}

    def wrapped(ctx, q_flash, k_flash, v_flash):
        out = orig(ctx, q_flash, k_flash, v_flash)
        cache, layer = ctx.get("cache"), ctx.get("layer_idx")
        if cache is not seen["cache"]:
            seen["cache"], seen["step"] = cache, 0
        if layer == 0:
            seen["step"] += 1
        if seen["step"] > max_steps:
            return out
        with torch.no_grad():
            q_hd = q_flash.transpose(1, 2)[:, :, 0, :]
            k_fp = k_flash.transpose(1, 2)
            v_fp = v_flash.transpose(1, 2)
            n_kv = ctx.get("n_kv_heads")
            if n_kv is not None and k_fp.shape[1] != n_kv:
                rep_ = k_fp.shape[1] // n_kv
                k_fp = k_fp[:, ::rep_].contiguous()
                v_fp = v_fp[:, ::rep_].contiguous()
            num_sink, ws = ctx["num_sink"], ctx["window_size"]
            n_body_win = -(-max(k_fp.shape[2] - num_sink, 0) // ws)
            out_full, _ = fused_two_tier_decode(
                q_hd, k_fp, v_fp, ctx["qtier"], ctx["scaling"], num_sink, n_body_win)
            out_full = out_full.clone()

            # PyTorch reference from the store (joint or per-layer).
            store = getattr(cache, "_joint_store", None)
            if store is not None and getattr(cache, "_joint", None) is not None:
                eff = store.effective_q_tier(cache.rope_module, k_fp.dtype)
                B = k_fp.shape[0]
                kq, vq = (t[layer * B:(layer + 1) * B] for t in eff[:2])
            else:
                eff = cache._stores[layer].effective_q_tier(cache.rope_module, k_fp.dtype)
                kq, vq = eff[0], eff[1]
            K = torch.cat([k_fp, kq], 2).float()
            V = torch.cat([v_fp, vq], 2).float()
            B, Hq, D = q_hd.shape
            rep = Hq // K.shape[1]
            Kx, Vx = K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1)
            lg = torch.einsum("bhd,bhsd->bhs", q_hd.float(), Kx) * ctx["scaling"]
            ref = torch.einsum("bhs,bhsd->bhd", torch.softmax(lg, -1), Vx)

            def rel(a):
                return ((a.float() - ref).norm(dim=-1)
                        / ref.norm(dim=-1).clamp_min(1e-12)).flatten().tolist()

            stats[layer]["kernel_vs_torch"].extend(rel(out_full))
            stats[layer]["shipped_vs_torch"].extend(rel(out[:, 0]))
            # Q-tier share of attention in the reference: how much the int2 tier
            # matters at this head, so a large error on a tiny share is legible.
            qs = torch.softmax(lg, -1)[..., k_fp.shape[2]:].sum(-1)
            stats[layer]["q_share"].extend(qs.flatten().tolist())
        return out

    flash_decode._run_fused = wrapped
    return stats, lambda: setattr(flash_decode, "_run_fused", orig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--out", default="outputs/fused_vs_torch.json")
    args = ap.parse_args()

    import torch
    from utils.config import load_config
    from utils.seed import seed_everything
    from modules.evaluation.longbench_runner import LongBenchRunner

    config = load_config(args.config, overrides={"longbench": {
        "datasets": [args.dataset], "num_samples": args.examples,
        "output_dir": str(Path(args.out).with_suffix("")) + "_predictions"}})
    seed_everything(config.run.seed)
    stats, uninstall = install(args.max_steps)
    try:
        LongBenchRunner(config).run()
    finally:
        uninstall()

    def q(xs, p):
        t = torch.tensor(xs, dtype=torch.float64)
        return float(t.quantile(p)) if t.numel() else float("nan")

    summary = {int(L): {f"{k}_{n}": q(v, p) for k, v in d.items()
                        for n, p in (("median", .5), ("p90", .9), ("max", 1.0))}
               for L, d in stats.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    if not summary:
        print("No fused decode layer ran -- check read_gate in the sidecar.")
        return
    print(f"\n{args.config} / {args.dataset}: relative output error vs PyTorch "
          "attention over the same cache -- median (p90, max)")
    print(f"{'layer':>5}  {'kernel (sel=None)':>28}  {'shipped step':>28}  {'q_share med':>11}")
    for L in sorted(summary):
        r = summary[L]
        print(f"{L:>5}  {r['kernel_vs_torch_median']:9.2e} ({r['kernel_vs_torch_p90']:.1e}, "
              f"{r['kernel_vs_torch_max']:.1e})  {r['shipped_vs_torch_median']:9.2e} "
              f"({r['shipped_vs_torch_p90']:.1e}, {r['shipped_vs_torch_max']:.1e})  "
              f"{r['q_share_median']:11.3f}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
