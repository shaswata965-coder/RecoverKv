#!/usr/bin/env python3
"""Combine a `run_perf_sweep.sh` directory into one table: config x shape x batch.

Each config lives in its own subdirectory because perf_runner names its npz after
the SHAPE and BATCH only — two configs in one directory overwrite each other.
This walks those subdirectories and prints the comparison the sweep was run for,
with the knobs parsed back out of `run_perf_table.env` rather than out of the
directory name, so a row says what it actually ran.

    python scripts/print_perf_sweep.py --sweep-dir outputs/sweep_v7
    python scripts/print_perf_sweep.py --sweep-dir outputs/sweep_v7 --sort tpot

A cell that OOMed or errored prints as such instead of being dropped: a sweep
whose hard cells vanish silently reads as a uniformly good method.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MB_PER_GB = 1024.0
KNOBS = ("cache_budget", "quant_ratio", "quant_gate_ratio",
         "local_window", "window_size", "num_sink")


def _env(cell: Path) -> dict:
    """Knobs as `run_perf_table.sh` recorded them. Falls back to the dir name."""
    out = {}
    f = cell / "run_perf_table.env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                out[k.strip().lower()] = v.strip().strip('"')
    return out


def _label(cell: Path) -> str:
    e = _env(cell)
    if not e:
        return cell.name
    g = lambda *names: next((e[n] for n in names if n in e), "?")
    return (f"b{g('cache_budget')} q{g('quant_ratio')} "
            f"g{g('gate_ratio', 'quant_gate_ratio')} "
            f"l{g('local_window', 'local_window_size')} w{g('window_size')}")


def _cells(cell: Path):
    for p in sorted(cell.glob("perf_prefill*_gen*_bs*.npz")):
        stem = p.stem
        try:
            pf = int(stem.split("prefill")[1].split("_")[0])
            gen = int(stem.split("_gen")[1].split("_")[0])
            bs = int(stem.split("_bs")[1])
        except (IndexError, ValueError):
            continue
        yield pf, gen, bs, p


def _row(path: Path, stat):
    try:
        d = dict(np.load(str(path), allow_pickle=True))
    except Exception as e:                                   # corrupt / partial
        return {"note": f"unreadable ({e.__class__.__name__})"}
    names = [str(n) for n in d.get("config_names", [])]
    idx = next((i for i, n in enumerate(names)
                if "full" not in n.lower() and "baseline" not in n.lower()), 0)
    if len(names) == 0:
        return {"note": "no configs"}
    if bool(np.asarray(d.get("oom_mask", np.zeros(len(names), bool)))[idx]):
        return {"note": "OOM"}
    if bool(np.asarray(d.get("error_mask", np.zeros(len(names), bool)))[idx]):
        why = [str(r) for r in d.get("skip_reason", [""] * len(names))][idx]
        return {"note": f"ERROR: {why[:40]}"}

    def col(name):
        if name not in d:
            return float("nan")
        row = np.asarray(d[name], dtype=float)[idx]
        row = row[~np.isnan(row)]
        return float(stat(row)) if row.size else float("nan")

    tpot = col("tpot_steady_ms")
    return {
        "tpot": tpot / 1000.0 if tpot == tpot else float("nan"),
        "dec": col("throughput_decode_tokps"),
        "peak": col("peak_memory_mb") / MB_PER_GB,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--stat", default="median", choices=["median", "mean"])
    ap.add_argument("--sort", default="config", choices=["config", "tpot"],
                    help="'tpot' ranks every (config, cell) by decode latency")
    a = ap.parse_args()
    stat = {"median": np.median, "mean": np.mean}[a.stat]

    root = Path(a.sweep_dir)
    cells = [p for p in sorted(root.iterdir()) if p.is_dir()] if root.is_dir() else []
    if not cells:
        print(f"no config directories under {root}", file=sys.stderr)
        return 1

    rows = []
    for cell in cells:
        for pf, gen, bs, path in _cells(cell):
            r = _row(path, stat)
            rows.append((_label(cell), f"{pf}/{gen}", bs, r))
    if not rows:
        print(f"no perf npz under {root}/*/", file=sys.stderr)
        return 1

    if a.sort == "tpot":
        rows.sort(key=lambda r: (r[3].get("tpot", float("inf")) if "note" not in r[3]
                                 else float("inf")))

    w = max(len(r[0]) for r in rows) + 2
    print("=" * (w + 52))
    print(f"{'config':<{w}}{'shape':>11}{'batch':>7}{'TPOT_st (s)':>13}"
          f"{'dec tok/s':>12}{'peak_GB':>9}")
    print("=" * (w + 52))
    last = None
    for label, shape, bs, r in rows:
        if a.sort == "config" and label != last:
            if last is not None:
                print("-" * (w + 52))
            last = label
        if "note" in r:
            print(f"{label:<{w}}{shape:>11}{bs:>7}{r['note']:>34}")
            continue
        print(f"{label:<{w}}{shape:>11}{bs:>7}{r['tpot']:>13.4f}"
              f"{r['dec']:>12.1f}{r['peak']:>9.2f}")
    print("=" * (w + 52))
    print(f"{len(cells)} configs, {len(rows)} cells, stat={a.stat}")
    print("TPOT_steady EXCLUDES decode step 0, which compacts the prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
