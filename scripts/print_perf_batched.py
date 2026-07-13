"""Print a human-readable summary of batched perf benchmark results.

Reads all perf_prefill*_gen*_bs*.npz files produced by eval_perf_batched.yaml
and emits a table of median TPOT (ms/token), TTFT (s), and end-to-end
throughput (token/s) for each config × scenario cell.

Usage:
    python scripts/print_perf_batched.py --npz-dir outputs/perf_batched
    python scripts/print_perf_batched.py --npz-dir outputs/perf_batched --out summary.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCENARIO_LABELS = {
    (512, 32,   1): "short_decode    (prefill=512, decode=32,   bs=1)",
    (512, 512,  1): "medium_decode   (prefill=512, decode=512,  bs=1)",
    (512, 1024, 1): "long_decode     (prefill=512, decode=1024, bs=1)",
}

COL_W = 24   # config name column width
NUM_W = 14   # numeric column width


def _load_npz(path: Path) -> dict | None:
    try:
        return dict(np.load(str(path), allow_pickle=True))
    except Exception as e:
        print(f"[warn] could not load {path}: {e}", file=sys.stderr)
        return None


def _meta(data: dict) -> dict:
    raw = str(data.get("metadata_json", np.array(["{}"]))[0])
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _median(arr: np.ndarray, ci: int) -> float | None:
    row = arr[ci]
    valid = row[~np.isnan(row)]
    return float(np.mean(valid)) if len(valid) else None


def _fmt(v: float | None, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if v is not None else "skipped"


def _print_scenario(data: dict, label: str, sink: list[str]) -> None:
    meta = _meta(data)
    batch_size = int(meta.get("batch_size", 1))
    names = [str(n) for n in data["config_names"]]
    skipped = data["skipped_mask"].astype(bool)
    ttft_ms    = data["ttft_ms"]       # shape (n_configs, n_runs)
    tpot_ms    = data["tpot_ms"]
    tput       = data["throughput_tokps"]
    e2e_ms_arr = data.get("e2e_latency_ms")

    header = (
        f"\n{'='*96}\n"
        f"SCENARIO: {label}\n"
        f"{'='*96}\n"
        f"{'Config':<{COL_W}} {'TTFT (s)':>{NUM_W}} {'TPOT (ms/tok)':>{NUM_W}} {'E2E Latency (ms)':>{NUM_W}} {'Throughput (tok/s)':>{NUM_W}}\n"
        f"{'-'*COL_W} {'-'*NUM_W} {'-'*NUM_W} {'-'*NUM_W} {'-'*NUM_W}"
    )
    sink.append(header)

    for ci, name in enumerate(names):
        if skipped[ci]:
            row = f"{name:<{COL_W}} {'skipped':>{NUM_W}} {'skipped':>{NUM_W}} {'skipped':>{NUM_W}} {'skipped':>{NUM_W}}"
        else:
            ttft_s   = _median(ttft_ms, ci)
            ttft_val = ttft_s / 1000.0 if ttft_s is not None else None
            tpot_val = _median(tpot_ms, ci)
            e2e_val  = _median(e2e_ms_arr, ci) if e2e_ms_arr is not None else None
            tput_val = _median(tput, ci)
            row = (
                f"{name:<{COL_W}}"
                f" {_fmt(ttft_val, 3):>{NUM_W}}"
                f" {_fmt(tpot_val, 2):>{NUM_W}}"
                f" {_fmt(e2e_val, 1):>{NUM_W}}"
                f" {_fmt(tput_val, 1):>{NUM_W}}"
            )
        sink.append(row)

    # Speedup vs. first non-skipped eager baseline
    baseline_idx = next(
        (i for i, n in enumerate(names)
         if "baseline" in n and "flash" not in n and not skipped[i]),
        None,
    )
    if baseline_idx is not None:
        base_tpot = _median(tpot_ms, baseline_idx)
        base_tput = _median(tput, baseline_idx)
        sink.append(f"\n  Relative to '{names[baseline_idx]}':")
        for ci, name in enumerate(names):
            if ci == baseline_idx or skipped[ci]:
                continue
            t = _median(tpot_ms, ci)
            tp = _median(tput, ci)
            tpot_ratio = f"{base_tpot/t:.2f}x faster" if t and base_tpot else "n/a"
            tput_ratio = f"{tp/base_tput:.2f}x" if tp and base_tput else "n/a"
            sink.append(f"    {name:<{COL_W-4}}  TPOT {tpot_ratio:<18}  throughput {tput_ratio}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", required=True, help="Directory with perf_*.npz files")
    parser.add_argument("--out", default=None, help="Optional file to write summary to")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    npz_files = sorted(npz_dir.glob("perf_prefill*_gen*_bs*.npz"))

    if not npz_files:
        print(f"No perf_prefill*_gen*_bs*.npz files found in {npz_dir}", file=sys.stderr)
        sys.exit(1)

    lines: list[str] = ["StickyKV — Batched Performance Summary", "=" * 80]

    for path in npz_files:
        data = _load_npz(path)
        if data is None:
            continue
        meta = _meta(data)
        prefill = int(meta.get("prefill_len", 0))
        gen     = int(meta.get("gen_len", 0))
        bs      = int(meta.get("batch_size", 1))
        key     = (prefill, gen, bs)
        label   = SCENARIO_LABELS.get(
            key, f"prefill={prefill}, decode={gen}, bs={bs}"
        )
        _print_scenario(data, label, lines)

    output = "\n".join(lines) + "\n"
    print(output)
    if args.out:
        Path(args.out).write_text(output)
        print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()
