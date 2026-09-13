#!/usr/bin/env python3
"""Run the three-shape x three-batch decode benchmark and print the result.

    prefill=1024  decode=1024    generation-dominated
    prefill=2048  decode= 512    balanced
    prefill=4096  decode= 256    prompt-dominated

at batch = 1, 32, 64. Nine cells, nothing else.

    python scripts/perf_shapes.py --model <hf-id-or-path> --dataset 2wikimqa

Self-contained: builds the config in memory, drives PerfRunner, then prints the
table from the npz it just wrote. No YAML, no second script.

WHY THE TABLE DERIVES ITS OWN THROUGHPUT COLUMNS
------------------------------------------------
Two fields the runner stores are not the ones a decode claim can be built on:

* ``batch / (tpot_ms/1000)`` -- what eval_perf_batched.yaml's header recommends
  -- is wrong, because ``tpot_ms`` averages in DECODE STEP 0, and step 0 is
  where this design compacts the whole prompt (perf_runner.py splits step0 from
  steady state for exactly this reason). Charging a one-off O(prefill) cost into
  a per-token rate makes the same method report a different decode throughput
  purely as a function of gen_len; the bias decays as 1/gen_len. This script
  uses ``tpot_steady_ms``, as DECODE_PLAN.md's success criteria require.

* ``throughput_tokps`` divides by ``gen_time + ttft``, which EXCLUDES the
  interval between end-of-prefill and start-of-decode (the LSE reuse/recompute
  and the first argmax). ``e2e_latency_ms`` is ``t3 - t0`` and includes it, so
  the two stored fields disagree whenever that interval is nonzero. This script
  reports the e2e-consistent rate and prints the disagreement as ``gap%``.

Aggregation is a true median over measurement runs.

MEASUREMENT HYGIENE PINNED HERE, NOT LEFT TO THE CONFIG DEFAULTS
----------------------------------------------------------------
* ``prefill_logits: last_only`` -- at prefill=4096 the full [B, L, V] logits
  tensor is ~1.05 GB PER ROW in fp16 (~67 GB at B=64). It is identical for every
  method and is enough on its own to OOM the B=64 rungs, stopping the ladder for
  reasons that have nothing to do with the KV cache.
* ``cooldown_s`` + clock locking -- nine cells run the part hot; without them
  later cells are measured on a slower GPU and the drift reads as method
  variance.
* ``allow_shared_gpu: false`` -- a co-tenant process is indistinguishable from a
  slow method in these numbers.
* ``quant_budget_mode: tokens`` -- this is a LATENCY comparison, so rows must do
  equal work; the repo default ('bytes') is the mode a memory claim is stated in.
* ``warmup_decode_steps: None`` (auto) -- runs enough decode steps to cross the
  first eviction, so the Triton eviction kernel JITs and autotunes in WARMUP
  rather than inside measurement run 0.

Requires transformers 4.47.x (utils/cache_factory.py refuses newer) and
flash-attn: every config here is flash_attention_2, because the eager backend's
score hook needs output_attentions, i.e. a [B, H, L, L] tensor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# (prefill_len, decode_steps). gen_len = decode_steps + 1: the prefill forward
# emits the first token, so a cell asking for 1024 DECODE STEPS is gen_len 1025.
# This is the classic off-by-one in this comparison.
SHAPES = [
    (1024, 1024, "A  generation-dominated"),
    (2048, 512,  "B  balanced            "),
    (4096, 256,  "C  prompt-dominated    "),
]


def build_raw_config(a) -> dict:
    """The full config dict, equivalent to a YAML file that does not exist."""
    configs = []
    if a.baseline:
        configs.append({
            "name": "full_cache",
            "cache_backend": "dynamic",
            "cache_package": None,
            "attn_implementation": "flash_attention_2",
            "cache_budget": None,
            "requires_flash_attn": True,
        })
    configs.append({
        "name": f"ours_b{int(a.budget * 100):02d}_q{int(a.quant_ratio * 100):02d}",
        "cache_backend": "windowed",
        "cache_package": "flash_attn",
        "attn_implementation": "flash_attention_2",
        "cache_budget": a.budget,
        "quant_ratio": a.quant_ratio,
        "requires_flash_attn": True,
    })
    return {
        "run": {"mode": "perf", "seed": a.seed},
        "model": {"name": a.model, "revision": a.revision, "dtype": a.dtype},
        "window": {
            "window_size": a.window,
            "num_sink_tokens": a.sink,
            "local_window_size": a.local,
        },
        "cache": {"quant_budget_mode": "tokens"},
        "perf": {
            "configs": configs,
            "grid": [
                {"prefill_len": p, "gen_len": d + 1, "batch_size": b}
                for (p, d, _) in SHAPES for b in a.batches
            ],
            "data_source": a.dataset,
            "prompt_mode": "dataset",
            "prefill_logits": "last_only",
            "num_warmup_runs": 1,
            "warmup_decode_steps": None,
            "num_measurement_runs": a.runs,
            "cooldown_s": a.cooldown,
            "enable_clock_locking": not a.no_clock_lock,
            "clock_settle_s": 5.0,
            "allow_shared_gpu": False,
            "skip_if_oom": True,
            "skip_if_flash_attn_unavailable": True,
        },
        "telemetry": {"track_scores": False, "output_dir": a.out},
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _med(data, name, ci):
    import numpy as np
    arr = data.get(name)
    if arr is None:
        return None
    row = np.asarray(arr)[ci]
    valid = row[np.isfinite(row)]
    return float(np.median(valid)) if valid.size else None


def _f(v, nd=2, w=10):
    return f"{'--':>{w}}" if v is None else f"{v:>{w}.{nd}f}"


def report(out_dir: Path, batches: list[int], baseline_name: str) -> str:
    import numpy as np
    lines: list[str] = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 118)
    emit("DECODE THROUGHPUT — 3 shapes x batch " + "/".join(map(str, batches)))
    emit("=" * 118)
    emit("dec tok/s = batch / TPOT_steady          step 0 excluded: it compacts the prompt")
    emit("e2e tok/s = batch*gen_len / e2e_latency  includes prefill and the prefill->decode gap")
    emit("gap%      = how far the runner's stored throughput_tokps overstates e2e tok/s")
    emit("")

    meta_seen: dict = {}
    errors: list[str] = []

    for prefill, decode, label in SHAPES:
        gen = decode + 1
        for bs in batches:
            path = out_dir / f"perf_prefill{prefill}_gen{gen}_bs{bs}.npz"
            if not path.exists():
                emit(f"{label}  bs={bs:<4}  -- no npz (cell did not run)")
                continue
            data = dict(np.load(str(path), allow_pickle=True))
            meta_seen = meta_seen or json.loads(str(data["metadata_json"][0]))
            names = [str(n) for n in data["config_names"]]
            oom = np.asarray(data.get("oom_mask", np.zeros(len(names), bool)))
            err = np.asarray(data.get("error_mask", np.zeros(len(names), bool)))
            why = [str(r) for r in data.get("skip_reason", [""] * len(names))]

            emit("-" * 118)
            emit(f"{label}   prefill={prefill} decode={decode}   |   batch = {bs}")
            emit("-" * 118)
            emit(f"{'config':<22}{'TTFT s':>9}{'step0 ms':>10}{'TPOT_st ms':>12}"
                 f"{'dec tok/s':>12}{'e2e tok/s':>12}{'gap%':>8}{'peak MB':>10}{'%dev':>7}")

            rows: dict[str, dict] = {}
            for ci, name in enumerate(names):
                if err[ci]:
                    errors.append(f"{label} bs={bs} {name}: {why[ci]}")
                    emit(f"{name:<22}{'ERROR':>9}   {why[ci]}")
                    continue
                if oom[ci]:
                    emit(f"{name:<22}{'OOM':>9}   does not fit")
                    continue
                tpot_s = _med(data, "tpot_steady_ms", ci)
                if tpot_s is None:
                    emit(f"{name:<22}{'SKIP':>9}   {why[ci] or 'no runs'}")
                    continue
                ttft = _med(data, "ttft_ms", ci)
                step0 = _med(data, "decode_step0_ms", ci)
                e2e = _med(data, "e2e_latency_ms", ci)
                stored = _med(data, "throughput_tokps", ci)
                peak = (_med(data, "peak_device_used_mb", ci)
                        or _med(data, "peak_memory_mb", ci))
                total = _med(data, "device_total_mb", ci)

                dec = bs / (tpot_s / 1000.0)
                e2t = (bs * gen) / (e2e / 1000.0) if e2e else None
                gap = 100.0 * (stored - e2t) / e2t if (stored and e2t) else None
                pct = 100.0 * peak / total if (peak and total) else None
                rows[name] = {"dec": dec, "e2e": e2t}
                emit(f"{name:<22}{_f(ttft / 1000 if ttft else None, 3, 9)}"
                     f"{_f(step0, 1)}{_f(tpot_s, 3, 12)}{_f(dec, 1, 12)}"
                     f"{_f(e2t, 1, 12)}{_f(gap, 1, 8)}{_f(peak, 0, 10)}{_f(pct, 0, 7)}")

            base = rows.get(baseline_name)
            if base:
                for name, r in rows.items():
                    if name == baseline_name:
                        continue
                    dx = r["dec"] / base["dec"] if base["dec"] else None
                    ex = r["e2e"] / base["e2e"] if base["e2e"] else None
                    emit(f"{'  vs ' + baseline_name:<22}"
                         f"{'':>31}{_f(dx, 3, 12)}{_f(ex, 3, 12)}      (x)")
            emit("")

    emit("=" * 118)
    emit("PROVENANCE")
    emit("=" * 118)
    for k in ("model_name", "dtype", "prompt_mode", "data_source", "budget_basis",
              "prefill_logits", "num_measurement_runs", "warmup_decode_steps",
              "cooldown_s", "clocks_locked", "score_backend"):
        if k in meta_seen:
            emit(f"  {k:<22}{meta_seen[k]}")
    if not meta_seen.get("clocks_locked", False):
        emit("  !! clocks NOT locked — later cells ran on a hotter, slower part.")
    if errors:
        emit("")
        emit("  ERRORED CELLS (the sweep is not trustworthy until these are fixed):")
        for e in errors:
            emit(f"    {e}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="3 shapes x batch 1/32/64 decode benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True,
                   help="HF id or local path, e.g. meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dataset", default="2wikimqa",
                   help="corpus for the prompt: a LongBench name, wikitext-103, "
                        "pg19, or a local file/directory path")
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--batches", default="1,32,64")
    p.add_argument("--budget", type=float, default=0.20)
    p.add_argument("--quant-ratio", type=float, default=0.7)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--sink", type=int, default=5)
    p.add_argument("--local", type=int, default=64)
    p.add_argument("--runs", type=int, default=5, help="measurement runs per cell")
    p.add_argument("--cooldown", type=float, default=3.0)
    p.add_argument("--no-clock-lock", action="store_true")
    p.add_argument("--no-baseline", dest="baseline", action="store_false",
                   help="skip the full-cache row (then there is no speedup column)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/perf_shapes")
    p.add_argument("--report-only", action="store_true",
                   help="re-print from existing npz without running")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved plan and exit; imports no torch")
    a = p.parse_args()

    a.batches = [int(b) for b in a.batches.split(",") if b.strip()]
    raw = build_raw_config(a)
    out_dir = Path(a.out)
    base_name = "full_cache" if a.baseline else ""

    if a.dry_run:
        print(json.dumps(raw, indent=2, default=str))
        print(f"\n{len(raw['perf']['grid'])} cells x "
              f"{len(raw['perf']['configs'])} configs x "
              f"{1 + a.runs} passes")
        return 0

    if not a.report_only:
        from utils.config import _dict_to_config
        from utils.seed import seed_everything
        from modules.evaluation.perf_runner import PerfRunner

        cfg = _dict_to_config(raw)
        seed_everything(cfg.run.seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plan.json").write_text(json.dumps(raw, indent=2, default=str))
        PerfRunner(cfg).run()

    text = report(out_dir, a.batches, base_name)
    (out_dir / "report.txt").write_text(text)
    print(f"\nwrote {out_dir / 'report.txt'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
