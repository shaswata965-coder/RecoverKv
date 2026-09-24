"""Measure the read gate's approximation on the real model -- minutes, not a LongBench run.

Runs the normal LongBench path (same config, same prompts, same generation) on a
few examples of one dataset, with every fused decode layer additionally
compared against the full read of the same step
(``modules/windowed_cache/gate_diagnosis.py`` defines the four metrics).
Predictions are the shipped method's; nothing about the run changes.

    python scripts/diagnose_gate_error.py \\
        --config configs/longbench_qwen_ours_flash_attn.yaml --dataset triviaqa

Run the same command with the Llama config for the comparison that matters:
the gate was validated on Llama-shaped keys, and the question is whether it is
as accurate on Qwen2.5's.

Reading it: ``out_err`` is what the approximation does to each layer's
attention output; ``card_tv`` says whether the card's weights for the windows
it does NOT read are right (0 exact, 1 disjoint); ``recall`` says whether it
reads the right windows; ``q_share`` says how much the int2 tier matters for
that head at all; ``skip_share`` is how much of the head's attention the
centroid path carries, and ``skip_mass_err`` whether it carries the right total
(log ratio, nats: + over-weighted, - under-weighted).

Run it at a larger ratio to see how much of the error is the SELECTION's
capacity rather than the card: ``--override cache.quant_gate_ratio=0.5``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=16,
                    help="decode steps measured per example (the rest run unmeasured)")
    ap.add_argument("--out", default="outputs/gate_diagnosis.json")
    ap.add_argument("--override", nargs="*", default=[],
                    help="extra key=value overrides, as main.py takes them")
    args = ap.parse_args()

    from main import _parse_value
    from utils.config import load_config
    from utils.seed import seed_everything
    from modules.evaluation.longbench_runner import LongBenchRunner
    from modules.windowed_cache import gate_diagnosis

    overrides = {"longbench": {
        "datasets": [args.dataset],
        "num_samples": args.examples,
        "output_dir": str(Path(args.out).with_suffix("")) + "_predictions",
    }}
    for item in args.override:
        key, val = item.split("=", 1)
        d = overrides
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _parse_value(val)

    config = load_config(args.config, overrides=overrides)
    seed_everything(config.run.seed)
    stats, uninstall = gate_diagnosis.install(max_steps=args.max_steps)
    try:
        LongBenchRunner(config).run()
    finally:
        uninstall()

    summary = gate_diagnosis.summarize(stats)
    summary["config"] = args.config
    summary["dataset"] = args.dataset
    summary["examples"] = args.examples
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    if not summary["layers"]:
        print("No fused decode layer was measured: the run never reached the "
              "gated read (check read_gate in the sidecar).")
        return
    print(f"\n{args.config} / {args.dataset}: per layer, medians (p90) over "
          "query heads x measured steps")
    print(f"{'layer':>5}  {'out_err':>15}  {'card_tv':>15}  {'recall':>15}  "
          f"{'q_share':>15}  {'skip_share':>15}  {'skip_mass_err':>17}")
    for layer, r in summary["layers"].items():
        print(f"{layer:>5}  "
              f"{r['out_err_median']:7.4f} ({r['out_err_p90']:.4f})  "
              f"{r['card_tv_median']:7.3f} ({r['card_tv_p90']:.3f})  "
              f"{r['recall_median']:7.3f} ({r['recall_p10']:.3f}*)  "
              f"{r['q_share_median']:7.3f} ({r['q_share_p90']:.3f})  "
              f"{r['skip_share_median']:7.3f} ({r['skip_share_p90']:.3f})  "
              f"{r['skip_mass_err_median']:+7.2f} ({r['skip_mass_err_p10']:+.2f}, "
              f"{r['skip_mass_err_p90']:+.2f})")
    o = summary["overall"]
    print(f"\nall layers: out_err median {o['out_err_median']:.4f} p90 {o['out_err_p90']:.4f} | "
          f"card_tv median {o['card_tv_median']:.3f} | recall median {o['recall_median']:.3f} "
          f"p10 {o['recall_p10']:.3f} | q_share median {o['q_share_median']:.3f} | "
          f"skip_share median {o['skip_share_median']:.3f} | skip_mass_err median "
          f"{o['skip_mass_err_median']:+.2f} nats (p10 {o['skip_mass_err_p10']:+.2f}, "
          f"p90 {o['skip_mass_err_p90']:+.2f})")
    print("(* recall's LOW decile: the heads the selection serves worst. "
          "skip_mass_err: + = centroids over-weighted, - = under-weighted)")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
