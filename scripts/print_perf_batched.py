"""Print a human-readable summary of batched perf benchmark results.

Reads all perf_prefill*_gen*_bs*.npz files produced by eval_perf_batched.yaml
and emits, per config × scenario cell: median TPOT (ms/token), TTFT (s),
end-to-end throughput (token/s), and the peak memory the cell reached.

It then prints the **max-B summary** — the point of the batched suite. For each
(prefill, gen) scenario and config it reports the largest batch size that did not
OOM, the decode throughput there, and how full the GPU was at peak. That is the
headline the method is measured by: tokens/s at the largest batch that fits, not
B=1 latency (BATCHING_PLAN.md §5).

max-B is read off `oom_mask`, never `skipped_mask`: a config can also be skipped
because flash-attn is missing or because it errored, and neither is evidence
about what fits. Cells that errored are called out separately — if any appear,
the ladder is not trustworthy until they are fixed.

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


def _ladder_key(path: Path) -> tuple[int, int, int]:
    """``(prefill, gen, batch)`` parsed from the filename, for numeric ordering."""
    stem = path.stem  # perf_prefill512_gen1024_bs32
    try:
        prefill = int(stem.split("prefill")[1].split("_")[0])
        gen = int(stem.split("_gen")[1].split("_")[0])
        bs = int(stem.split("_bs")[1])
        return prefill, gen, bs
    except (IndexError, ValueError):
        return 0, 0, 0


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


def _mask(data: dict, key: str, n: int) -> np.ndarray:
    """A boolean per-config mask, or all-False when the npz predates the field."""
    arr = data.get(key)
    if arr is None:
        return np.zeros(n, dtype=bool)
    return np.asarray(arr).astype(bool)


def _skip_label(data: dict, ci: int, oom: np.ndarray, err: np.ndarray) -> str:
    """Why this cell has no numbers — the distinction max-B depends on."""
    if oom[ci]:
        return "OOM"
    if err[ci]:
        return "ERROR"
    reasons = data.get("skip_reason")
    if reasons is not None and str(reasons[ci]):
        return str(reasons[ci])[:14]
    return "skipped"


def _print_scenario(data: dict, label: str, sink: list[str]) -> None:
    meta = _meta(data)
    batch_size = int(meta.get("batch_size", 1))
    names = [str(n) for n in data["config_names"]]
    skipped = data["skipped_mask"].astype(bool)
    oom = _mask(data, "oom_mask", len(names))
    err = _mask(data, "error_mask", len(names))
    ttft_ms    = data["ttft_ms"]       # shape (n_configs, n_runs)
    tpot_ms    = data["tpot_ms"]
    tput       = data["throughput_tokps"]
    e2e_ms_arr = data.get("e2e_latency_ms")
    peak_dev   = data.get("peak_device_used_mb")
    peak_alloc = data.get("peak_memory_mb")

    header = (
        f"\n{'='*110}\n"
        f"SCENARIO: {label}\n"
        f"{'='*110}\n"
        f"{'Config':<{COL_W}} {'TTFT (s)':>{NUM_W}} {'TPOT (ms/tok)':>{NUM_W}} {'E2E Latency (ms)':>{NUM_W}} {'Throughput (tok/s)':>{NUM_W}} {'Peak mem (MB)':>{NUM_W}}\n"
        f"{'-'*COL_W} {'-'*NUM_W} {'-'*NUM_W} {'-'*NUM_W} {'-'*NUM_W} {'-'*NUM_W}"
    )
    sink.append(header)

    for ci, name in enumerate(names):
        if skipped[ci]:
            why = _skip_label(data, ci, oom, err)
            cells = " ".join(f"{why:>{NUM_W}}" for _ in range(5))
            row = f"{name:<{COL_W}} {cells}"
        else:
            ttft_s   = _median(ttft_ms, ci)
            ttft_val = ttft_s / 1000.0 if ttft_s is not None else None
            tpot_val = _median(tpot_ms, ci)
            e2e_val  = _median(e2e_ms_arr, ci) if e2e_ms_arr is not None else None
            tput_val = _median(tput, ci)
            # Prefer the device-level peak (what an OOM is decided on); fall back
            # to the torch allocated peak on CPU runs and on older npz files.
            mem_val = _median(peak_dev, ci) if peak_dev is not None else None
            if not mem_val and peak_alloc is not None:
                mem_val = _median(peak_alloc, ci)
            row = (
                f"{name:<{COL_W}}"
                f" {_fmt(ttft_val, 3):>{NUM_W}}"
                f" {_fmt(tpot_val, 2):>{NUM_W}}"
                f" {_fmt(e2e_val, 1):>{NUM_W}}"
                f" {_fmt(tput_val, 1):>{NUM_W}}"
                f" {_fmt(mem_val, 0):>{NUM_W}}"
            )
        sink.append(row)

    if err.any():
        sink.append(
            "\n  [!] non-OOM ERRORs in this cell: "
            + ", ".join(names[i] for i in range(len(names)) if err[i])
            + " — these are NOT max-B evidence."
        )

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


def _collect_cells(npz_files: list[Path]) -> list[dict]:
    """One record per (scenario, batch, config) cell across every npz."""
    cells: list[dict] = []
    for path in npz_files:
        data = _load_npz(path)
        if data is None:
            continue
        meta = _meta(data)
        names = [str(n) for n in data["config_names"]]
        skipped = data["skipped_mask"].astype(bool)
        oom = _mask(data, "oom_mask", len(names))
        err = _mask(data, "error_mask", len(names))
        for ci, name in enumerate(names):
            cells.append({
                "prefill": int(meta.get("prefill_len", 0)),
                "gen": int(meta.get("gen_len", 0)),
                "bs": int(meta.get("batch_size", 1)),
                "config": name,
                "ran": not skipped[ci],
                "oom": bool(oom[ci]),
                "error": bool(err[ci]),
                "tpot_ms": _median(data["tpot_ms"], ci),
                "tput": _median(data["throughput_tokps"], ci),
                "peak_dev_mb": (
                    _median(data["peak_device_used_mb"], ci)
                    if "peak_device_used_mb" in data else None
                ),
                "device_total_mb": (
                    _median(data["device_total_mb"], ci)
                    if "device_total_mb" in data else None
                ),
                "peak_prefill_mb": (
                    _median(data["peak_prefill_mb"], ci)
                    if "peak_prefill_mb" in data else None
                ),
                "peak_decode_mb": (
                    _median(data["peak_decode_mb"], ci)
                    if "peak_decode_mb" in data else None
                ),
            })
    return cells


def _print_max_b(cells: list[dict], sink: list[str]) -> None:
    """The headline: largest batch that fits, and the throughput there.

    Decode throughput is ``batch_size / (tpot_ms / 1000)`` — NOT the npz's
    ``throughput_tokps``, which includes prefill (perf_runner says so). At a
    long-prompt shape the prefill term dominates and would hide the decode win
    this suite exists to measure.
    """
    if not cells:
        return
    scenarios = sorted({(c["prefill"], c["gen"]) for c in cells})
    sink.append(f"\n{'='*110}")
    sink.append("MAX-B SUMMARY — largest batch that fits, and decode tok/s there")
    sink.append("=" * 110)

    for prefill, gen in scenarios:
        rows = [c for c in cells if (c["prefill"], c["gen"]) == (prefill, gen)]
        configs = sorted({c["config"] for c in rows})
        ladder = sorted({c["bs"] for c in rows})
        sink.append(f"\nscenario: prefill={prefill}  gen={gen}   "
                    f"ladder={ladder}")
        sink.append(
            f"{'Config':<{COL_W}} {'max-B':>8} {'decode tok/s':>14} "
            f"{'tok/s/row':>12} {'peak GPU MB':>13} {'util':>7} {'peak phase':>12}"
        )
        sink.append(f"{'-'*COL_W} {'-'*8} {'-'*14} {'-'*12} {'-'*13} {'-'*7} {'-'*12}")
        best: dict[str, float] = {}
        for name in configs:
            fitted = [c for c in rows if c["config"] == name and c["ran"]]
            if not fitted:
                oomed = any(c["oom"] for c in rows if c["config"] == name)
                sink.append(
                    f"{name:<{COL_W}} {'—':>8} "
                    f"{('OOM at every B' if oomed else 'never ran'):>14}"
                )
                continue
            top = max(fitted, key=lambda c: c["bs"])
            # Did the ladder actually find the ceiling, or just run out of rungs?
            higher = [c for c in rows if c["config"] == name and c["bs"] > top["bs"]]
            bounded = any(c["oom"] for c in higher)
            per_row = 1000.0 / top["tpot_ms"] if top["tpot_ms"] else None
            decode = per_row * top["bs"] if per_row else None
            util = (
                top["peak_dev_mb"] / top["device_total_mb"]
                if top.get("peak_dev_mb") and top.get("device_total_mb") else None
            )
            phase = ""
            if top.get("peak_prefill_mb") and top.get("peak_decode_mb"):
                phase = ("prefill" if top["peak_prefill_mb"] >= top["peak_decode_mb"]
                         else "decode")
            if decode:
                best[name] = decode
            sink.append(
                f"{name:<{COL_W}} {str(top['bs']) + ('' if bounded else '+'):>8} "
                f"{_fmt(decode, 1):>14} {_fmt(per_row, 2):>12} "
                f"{_fmt(top.get('peak_dev_mb'), 0):>13} "
                f"{(f'{util*100:.0f}%' if util else '—'):>7} {phase:>12}"
            )

        sink.append(
            "  (a '+' on max-B means no higher rung OOMed — the ladder ran out, "
            "so this is a LOWER BOUND)"
        )
        baseline = next(
            (n for n in configs if "full" in n or "baseline" in n), None
        )
        if baseline and baseline in best:
            for name in configs:
                if name == baseline or name not in best:
                    continue
                sink.append(
                    f"    {name:<{COL_W-4}} {best[name]/best[baseline]:.2f}x "
                    f"decode throughput vs {baseline}"
                )
    errs = [c for c in cells if c["error"]]
    if errs:
        sink.append(
            f"\n[!] {len(errs)} cell(s) failed for non-OOM reasons and were "
            f"excluded from max-B. Fix these before quoting the table:"
        )
        for c in errs[:12]:
            sink.append(
                f"      prefill={c['prefill']} gen={c['gen']} bs={c['bs']} "
                f"{c['config']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", required=True, help="Directory with perf_*.npz files")
    parser.add_argument("--out", default=None, help="Optional file to write summary to")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    # Sort by (prefill, gen, batch) NUMERICALLY. A lexicographic glob interleaves
    # the ladder as bs1, bs128, bs256, bs32, bs512, bs8 — which makes a max-B
    # search unreadable, since the whole point is to walk B upward.
    npz_files = sorted(
        npz_dir.glob("perf_prefill*_gen*_bs*.npz"), key=_ladder_key
    )

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

    _print_max_b(_collect_cells(npz_files), lines)

    output = "\n".join(lines) + "\n"
    print(output)
    if args.out:
        Path(args.out).write_text(output)
        print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()
