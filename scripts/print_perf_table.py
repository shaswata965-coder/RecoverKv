"""Print the 7-column decode table: shape, batch, TTFT, TPOT_steady, throughput,
peak_GB, steadyKV_GB — one row per (prefill, gen, batch) cell.

This is the compact operator-facing view of a perf run: the same columns as the
shared screenshot, nothing else. The full split (step-0 vs steady, per-phase
peaks, provenance) lives in print_efficiency.py; this is the headline table.

Column provenance (all from the npz the perf runner writes):
    TTFT (s)            ttft_ms / 1000
    TPOT_steady (s)     tpot_steady_ms / 1000   <- steady state, EXCLUDES step 0
    throughput (tok/s)  throughput_tokps        <- includes prefill (perf_runner)
    peak_GB             peak_memory_mb / 1024    (torch ALLOCATED peak)
    steadyKV_GB         peak_decode_steady_mb / 1024   (DEVICE-USED in the steady
                        phase; includes the CUDA context, so it can read HIGHER
                        than peak_GB even though the cache is smaller -- the two
                        columns are different quantities, not the same one twice)

A cell that OOMed prints "OOM" under TTFT and "-" elsewhere; a cell skipped for
another reason (flash-attn missing, config error) prints "-" across. max-B is
read off the OOM mask, exactly as print_perf_batched does.

Usage:
    python scripts/print_perf_table.py --npz-dir outputs/perf_table
    python scripts/print_perf_table.py --npz-dir outputs/perf_table --config ours \
        --stat median --out table.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MB_PER_GB = 1024.0

#: Cells written further apart than this did not come from one run.
STALE_S = 3600.0


def _cells(npz_dir: Path):
    """Yield (prefill, gen, batch, path) for every perf npz, ordered like the
    screenshot: prefill descending, then batch ascending."""
    rows = []
    for path in npz_dir.glob("perf_prefill*_gen*_bs*.npz"):
        stem = path.stem
        try:
            p = int(stem.split("prefill")[1].split("_")[0])
            g = int(stem.split("_gen")[1].split("_")[0])
            b = int(stem.split("_bs")[1])
        except (IndexError, ValueError):
            continue
        rows.append((p, g, b, path))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    return rows


def _load(path: Path):
    try:
        return dict(np.load(str(path), allow_pickle=True))
    except Exception as e:  # pragma: no cover - corrupt file
        print(f"[warn] could not load {path}: {e}", file=sys.stderr)
        return None


def _col(data: dict, name: str, n_cfg: int, n_run: int) -> np.ndarray:
    if name not in data:
        return np.full((n_cfg, n_run), np.nan)
    return np.asarray(data[name], dtype=float)


def _stat(row: np.ndarray, fn) -> float:
    row = row[~np.isnan(row)]
    return float(fn(row)) if row.size else float("nan")


def _match_index(names, wanted: str | None) -> int | None:
    """Row index for the requested config name (substring, case-insensitive),
    or the single non-baseline row, or 0. None if nothing matches."""
    if wanted is not None:
        for i, n in enumerate(names):
            if wanted.lower() in n.lower():
                return i
        return None
    # No name given: prefer the one config that is not a full-cache baseline.
    non_base = [i for i, n in enumerate(names)
                if "full" not in n.lower() and "baseline" not in n.lower()]
    if len(non_base) == 1:
        return non_base[0]
    return 0 if len(names) else None


def build_table(npz_dir: Path, config: str | None, stat: str) -> str:
    fn = {"median": np.median, "mean": np.mean}[stat]
    header = ["shape", "batch", "TTFT (s)", "TPOT_steady (s)",
              "throughput (tok/s)", "peak_GB", "steadyKV_GB"]
    widths = [11, 6, 9, 16, 19, 9, 12]

    def fmt_row(vals) -> str:
        return "  ".join(str(v).ljust(w) if i < 2 else str(v).rjust(w)
                         for i, (v, w) in enumerate(zip(vals, widths)))

    lines = [fmt_row(header), "-" * (sum(widths) + 2 * (len(widths) - 1))]

    any_row = False
    notes: list = []
    ages: list = []
    for prefill, gen, batch, path in _cells(npz_dir):
        data = _load(path)
        if data is None:
            continue
        names = [str(n) for n in data["config_names"]]
        n_cfg = len(names)
        n_run = int(np.asarray(data["ttft_ms"]).shape[1]) if n_cfg else 0
        ci = _match_index(names, config)
        if ci is None:
            continue
        any_row = True
        shape = f"{prefill}/{gen}"
        try:
            ages.append((path.stat().st_mtime, shape, batch))
        except OSError:
            pass

        oom = bool(np.asarray(data.get("oom_mask", np.zeros(n_cfg)))[ci]) \
            if "oom_mask" in data else False
        err = bool(np.asarray(data.get("error_mask", np.zeros(n_cfg)))[ci]) \
            if "error_mask" in data else False
        skipped = bool(np.asarray(data.get("skipped_mask", np.zeros(n_cfg)))[ci]) \
            if "skipped_mask" in data else False

        reason = ""
        if "skip_reason" in data:
            try:
                reason = str(np.asarray(data["skip_reason"], dtype=object)[ci])
            except Exception:
                reason = ""

        if oom:
            lines.append(fmt_row([shape, batch, "OOM", "-", "-", "-", "-"]))
            notes.append((shape, batch, "OOM", reason))
            continue
        if err or skipped:
            # NOT dashes. A row of dashes reads as "nothing happened", which is
            # how a deliberate hard error (the strict L-reuse miss) got mistaken
            # for a slow-but-successful cell. The reason is in the npz and has
            # always been; it was simply never printed.
            lines.append(fmt_row([shape, batch, "ERROR", "-", "-", "-", "-"]))
            notes.append((shape, batch, "ERROR" if err else "SKIPPED", reason))
            continue

        ttft = _stat(_col(data, "ttft_ms", n_cfg, n_run)[ci], fn) / 1000.0
        tpot_ss = _stat(_col(data, "tpot_steady_ms", n_cfg, n_run)[ci], fn) / 1000.0
        thru = _stat(_col(data, "throughput_tokps", n_cfg, n_run)[ci], fn)
        peak = _stat(_col(data, "peak_memory_mb", n_cfg, n_run)[ci], fn) / MB_PER_GB
        skv = _stat(_col(data, "peak_decode_steady_mb", n_cfg, n_run)[ci], fn) / MB_PER_GB

        def g(v, p):
            return "-" if v != v else f"{v:.{p}f}"

        lines.append(fmt_row([shape, batch, g(ttft, 3), g(tpot_ss, 4),
                              g(thru, 1), g(peak, 2), g(skv, 2)]))

    if not any_row:
        hint = f" matching --config {config!r}" if config else ""
        lines.append(f"(no perf npz found in {npz_dir}{hint})")
        return "\n".join(lines)

    if notes:
        lines.append("")
    for shape, batch, kind, reason in notes:
        lines.append(f"  {kind}  {shape} batch={batch}: "
                     f"{reason or '(no reason recorded in the npz)'}")

    # Provenance. This printer globs the WHOLE directory, so a run that
    # re-measures one cell leaves every other cell showing its previous
    # result -- with nothing on screen to say so. Cells written more than
    # STALE_S apart did not come from the same run and must not be read as
    # one table.
    if len(ages) >= 2:
        newest = max(a[0] for a in ages)
        stale = [(t, s, b) for t, s, b in ages if newest - t > STALE_S]
        if stale:
            lines.append("")
            lines.append(f"  !! MIXED RUNS: {len(stale)} of {len(ages)} cells "
                         f"are older than the newest by more than "
                         f"{STALE_S / 3600:.0f}h. This is not one table.")
            for t, s, b in sorted(stale, key=lambda r: r[0]):
                hrs = (newest - t) / 3600.0
                lines.append(f"     stale  {s} batch={b}   {hrs:.1f}h older "
                             f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(t))})")
            lines.append("     Re-run those shapes, or point --npz-dir at a "
                         "clean directory, before comparing rows.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", required=True, type=Path)
    ap.add_argument("--config", default=None,
                    help="Config-name substring to print (default: the single "
                         "non-baseline config in each file).")
    ap.add_argument("--stat", default="median", choices=["median", "mean"],
                    help="Statistic across measurement runs (default: median).")
    ap.add_argument("--out", default=None, type=Path,
                    help="Also write the table to this file.")
    args = ap.parse_args()

    table = build_table(args.npz_dir, args.config, args.stat)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"\n[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
