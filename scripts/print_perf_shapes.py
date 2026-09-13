"""Print the shape sweep (configs/eval_perf_shapes.yaml) as a throughput table.

Reads every ``perf_prefill*_gen*_bs*.npz`` under --npz-dir and prints, per
(shape x batch) cell, one row per config.

THREE DERIVED COLUMNS THIS PRINTER COMPUTES ITSELF, because the fields the
runner stores are not the ones a throughput claim needs:

* ``dec tok/s`` = batch_size / (tpot_steady_ms / 1000).
  NOT ``batch_size / (tpot_ms/1000)``, which is what eval_perf_batched.yaml's
  header recommends. ``tpot_ms`` averages in DECODE STEP 0, and step 0 is where
  this design compacts the whole prompt (perf_runner.py splits them for exactly
  this reason). Charging a one-off O(prefill) compaction into a per-token rate
  makes the same method report a different decode throughput purely as a
  function of gen_len — the penalty decays as 1/gen_len. DECODE_PLAN.md says to
  report TPOT_steady; this column is that, expressed as a rate.

* ``e2e tok/s`` = batch_size * gen_len / (e2e_latency_ms / 1000).
  NOT the stored ``throughput_tokps``, which is
  ``batch*gen_len / (gen_time + ttft)`` and therefore EXCLUDES the interval
  between the end of prefill and the start of decode (the LSE
  reuse/recompute and the first argmax). ``e2e_latency_ms`` is ``t3 - t0`` and
  includes it, so the two stored fields are mutually inconsistent whenever that
  gap is nonzero. The ``gap%`` column prints the disagreement: a large value
  means real work is falling between the two timed regions and the stored
  throughput is optimistic.

* Aggregation is a true MEDIAN over measurement runs. ``print_perf_batched.py``
  names its helper ``_median`` but returns ``np.mean``; with n=2 and no outlier
  rejection one thermally-throttled run moves the number by half its excess.

Usage:
    python scripts/print_perf_shapes.py --npz-dir outputs/perf_shapes
    python scripts/print_perf_shapes.py --npz-dir outputs/perf_shapes --out report.txt
    python scripts/print_perf_shapes.py --npz-dir outputs/perf_shapes --baseline full_cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SHAPE_LABEL = {
    (1024, 1025): "A  generation-dominated  prefill=1024 decode=1024",
    (2048, 513):  "B  balanced              prefill=2048 decode= 512",
    (4096, 257):  "C  prompt-dominated      prefill=4096 decode= 256",
}


def _key(path: Path) -> tuple[int, int, int]:
    stem = path.stem
    try:
        return (int(stem.split("prefill")[1].split("_")[0]),
                int(stem.split("_gen")[1].split("_")[0]),
                int(stem.split("_bs")[1]))
    except (IndexError, ValueError):
        return (0, 0, 0)


def _med(data: dict, name: str, ci: int) -> float | None:
    """True median over measurement runs; None if the cell produced nothing."""
    arr = data.get(name)
    if arr is None:
        return None
    row = np.asarray(arr)[ci]
    valid = row[np.isfinite(row)]
    return float(np.median(valid)) if valid.size else None


def _fmt(v: float | None, nd: int = 2, width: int = 10) -> str:
    return f"{'--':>{width}}" if v is None else f"{v:>{width}.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="outputs/perf_shapes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default="full_cache",
                    help="config name used as the speedup denominator")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    files = sorted(npz_dir.glob("perf_prefill*_gen*_bs*.npz"), key=_key)
    if not files:
        print(f"no npz found under {npz_dir}", file=sys.stderr)
        return 1

    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("=" * 132)
    emit("SHAPE SWEEP — decode throughput at B = 1 / 32 / 64")
    emit("=" * 132)
    emit("dec tok/s = batch / TPOT_steady   (step 0 excluded: it compacts the prompt)")
    emit("e2e tok/s = batch * gen_len / e2e_latency   (includes prefill AND the prefill->decode gap)")
    emit("gap%      = disagreement with the runner's stored throughput_tokps; large => untimed work between phases")
    emit("")

    meta_seen: dict = {}
    errored_cells: list[str] = []

    for path in files:
        try:
            data = dict(np.load(str(path), allow_pickle=True))
        except Exception as e:                                  # pragma: no cover
            print(f"[warn] {path}: {e}", file=sys.stderr)
            continue

        meta = json.loads(str(data["metadata_json"][0])) if "metadata_json" in data else {}
        meta_seen = meta_seen or meta
        prefill, gen, bs = _key(path)
        names = [str(n) for n in data["config_names"]]
        oom = np.asarray(data.get("oom_mask", np.zeros(len(names), bool)))
        err = np.asarray(data.get("error_mask", np.zeros(len(names), bool)))
        reason = [str(r) for r in data.get("skip_reason", [""] * len(names))]

        label = SHAPE_LABEL.get((prefill, gen), f"prefill={prefill} gen={gen}")
        emit("-" * 132)
        emit(f"{label}   |   batch = {bs}")
        emit("-" * 132)
        emit(f"{'config':<24}{'TTFT s':>10}{'step0 ms':>10}{'TPOT_st ms':>12}"
             f"{'dec tok/s':>12}{'e2e tok/s':>12}{'gap%':>8}"
             f"{'peak MB':>11}{'%dev':>7}{'  note'}")

        rows: dict[str, dict] = {}
        for ci, name in enumerate(names):
            if err[ci]:
                errored_cells.append(f"{label} bs={bs} {name}: {reason[ci]}")
                emit(f"{name:<24}{'ERROR':>10}{'':>61}  {reason[ci]}")
                continue
            if oom[ci]:
                emit(f"{name:<24}{'OOM':>10}{'':>61}  does not fit")
                continue
            if not np.any(np.isfinite(np.asarray(data['tpot_steady_ms'])[ci])):
                emit(f"{name:<24}{'SKIP':>10}{'':>61}  {reason[ci] or 'no runs'}")
                continue

            ttft = _med(data, "ttft_ms", ci)
            step0 = _med(data, "decode_step0_ms", ci)
            tpot_s = _med(data, "tpot_steady_ms", ci)
            e2e = _med(data, "e2e_latency_ms", ci)
            stored = _med(data, "throughput_tokps", ci)
            peak = _med(data, "peak_device_used_mb", ci) or _med(data, "peak_memory_mb", ci)
            total = _med(data, "device_total_mb", ci)

            dec_tps = bs / (tpot_s / 1000.0) if tpot_s else None
            e2e_tps = (bs * gen) / (e2e / 1000.0) if e2e else None
            gap = (100.0 * (stored - e2e_tps) / e2e_tps
                   if (stored and e2e_tps) else None)
            pct = (100.0 * peak / total) if (peak and total) else None

            rows[name] = {"dec": dec_tps, "e2e": e2e_tps, "tpot": tpot_s}
            emit(f"{name:<24}{_fmt(ttft/1000 if ttft else None, 3)}{_fmt(step0, 1)}"
                 f"{_fmt(tpot_s, 3, 12)}{_fmt(dec_tps, 1, 12)}{_fmt(e2e_tps, 1, 12)}"
                 f"{_fmt(gap, 1, 8)}{_fmt(peak, 0, 11)}{_fmt(pct, 0, 7)}")

        base = rows.get(args.baseline)
        if base and len(rows) > 1:
            emit("")
            emit(f"{'  vs ' + args.baseline:<24}{'dec x':>22}{'e2e x':>12}")
            for name, r in rows.items():
                if name == args.baseline:
                    continue
                dx = r["dec"] / base["dec"] if (r["dec"] and base["dec"]) else None
                ex = r["e2e"] / base["e2e"] if (r["e2e"] and base["e2e"]) else None
                emit(f"{'  ' + name:<24}{_fmt(dx, 3, 22)}{_fmt(ex, 3, 12)}")
        emit("")

    emit("=" * 132)
    emit("PROVENANCE")
    emit("=" * 132)
    for k in ("model_name", "dtype", "protocol", "prompt_mode", "data_source",
              "budget_basis", "prefill_logits", "num_warmup_runs",
              "num_measurement_runs", "warmup_decode_steps", "cooldown_s",
              "clocks_locked", "score_backend"):
        if k in meta_seen:
            emit(f"  {k:<24}{meta_seen[k]}")
    if meta_seen.get("prefill_logits") == "full":
        emit("  !! prefill_logits=full — the [B,L,V] logits tensor is in the peak-memory")
        emit("     column and is method-independent. Memory rows are not comparable.")
    if not meta_seen.get("clocks_locked", False) and not meta_seen.get("cooldown_s"):
        emit("  !! clocks unlocked AND cooldown_s=0 — later cells ran on a hotter, "
             "slower part.")
    if errored_cells:
        emit("")
        emit("  ERRORED CELLS (the sweep is not trustworthy until these are fixed):")
        for c in errored_cells:
            emit(f"    {c}")

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
