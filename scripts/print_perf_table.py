"""Print the decode table: shape, batch, TTFT, TPOT_steady, throughput(s),
peak_GB, steadyKV_GB — one row per (prefill, gen, batch) cell.

This is the compact operator-facing view of a perf run. The full split (step-0
vs steady, per-phase peaks, provenance) lives in print_efficiency.py; this is
the headline table.

Which throughput column you get
-------------------------------
``--throughput`` selects it, and the choice is load-bearing:

* ``decode`` — ``batch / TPOT_steady``. **The decode claim.** Built on the
  steady state, so the one-off prompt compaction this design charges to decode
  step 0 is excluded.
* ``e2e`` — ``batch * gen_len / e2e_latency``. **The end-to-end claim.**
  Consistent with ``e2e_latency_ms`` by construction; includes prefill and the
  prefill->decode gap.
* ``both`` (default) — both columns, which is what a decode table should show:
  the two rates differ by a factor that grows with prefill/gen and with batch,
  so one number cannot stand for both.
* ``legacy`` — the npz's ``throughput_tokps``, in the original 7-column layout.
  Its denominator is ``gen_time + TTFT``, which omits ``prefill_to_decode_gap_ms``
  (the L-reuse/recompute and the first argmax) that ``e2e_latency_ms`` does
  include, so it is neither a clean e2e figure nor a decode figure and it
  overstates the rate by ``gap / (e2e - gap)``. perf_runner documents it as
  legacy and says not to quote it. It is kept here for ONE purpose: reproducing
  a table printed before this change, byte-for-byte, so old output can be
  identified rather than guessed at. A warning goes to stderr; stdout stays
  identical.
* ``all`` — decode, e2e and legacy side by side. Use it once to reconcile an old
  screenshot against the corrected figures.

This printer used to hard-code the legacy field. The corrections in fbcf368
added ``throughput_decode_tokps`` / ``throughput_e2e_tokps`` to the runner and
reached print_perf_batched.py and perf_shapes.py, but not this script, so the
headline table kept quoting the field the runner tells you not to quote.

Column provenance (all from the npz the perf runner writes):
    TTFT (s)            ttft_ms / 1000
    TPOT_steady (s)     tpot_steady_ms / 1000   <- steady state, EXCLUDES step 0
    dec tok/s           throughput_decode_tokps, else batch / tpot_steady_ms
    e2e tok/s           throughput_e2e_tokps, else batch*gen / e2e_latency_ms
    throughput (tok/s)  throughput_tokps        <- legacy, --throughput legacy only
    peak_GB             peak_memory_mb / 1024    (torch ALLOCATED peak)
    steadyKV_GB         peak_decode_steady_mb / 1024   (DEVICE-USED in the steady
                        phase; includes the CUDA context, so it can read HIGHER
                        than peak_GB even though the cache is smaller -- the two
                        columns are different quantities, not the same one twice)

The two corrected columns prefer the runner's own fields and fall back to the
identical formula over ``tpot_steady_ms`` / ``e2e_latency_ms`` for npz files
written before those fields existed (``batch`` and ``gen_len`` come from the
filename, which is also what the shape column prints). A derivation is footnoted,
never silent. What is NOT done is substituting ``throughput_tokps`` for a missing
e2e figure: that is a different quantity, so the cell prints "-" and says why.

A cell that OOMed prints "OOM" under TTFT and "-" elsewhere; a cell skipped for
another reason (flash-attn missing, config error) prints "-" across. max-B is
read off the OOM mask, exactly as print_perf_batched does.

Usage:
    python scripts/print_perf_table.py --npz-dir outputs/perf_table
    python scripts/print_perf_table.py --npz-dir outputs/perf_table --throughput decode
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

#: header label, column width, kind — per --throughput mode. The shared columns
#: keep their original widths so `legacy` reproduces the old table byte-for-byte.
THROUGHPUT_COLUMNS = {
    "both":   [("dec tok/s", 12, "decode"), ("e2e tok/s", 12, "e2e")],
    "decode": [("dec tok/s", 12, "decode")],
    "e2e":    [("e2e tok/s", 12, "e2e")],
    "legacy": [("throughput (tok/s)", 19, "legacy")],
    "all":    [("dec tok/s", 12, "decode"), ("e2e tok/s", 12, "e2e"),
               ("legacy tok/s", 13, "legacy")],
}


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


def _positive(row: np.ndarray) -> np.ndarray:
    """A denominator row with the unusable entries turned into NaN.

    A zero or negative duration is not a fast cell, it is a missing measurement;
    dividing by it would print an infinity where the table promises a rate.
    """
    row = np.asarray(row, dtype=float).copy()
    row[~(row > 0)] = np.nan
    return row


def _rate(data: dict, ci: int, n_cfg: int, n_run: int, fn, *,
          field: str, denom: str, numer: float):
    """``(value, source)`` for one rate column.

    Prefers the rate the runner recorded; falls back to recomputing it per run
    from ``denom`` (the same formula perf_runner uses) for npz files written
    before that field existed, and reports which of the two it used so the
    caller can footnote a derivation instead of passing it off as recorded.
    """
    direct = _col(data, field, n_cfg, n_run)[ci]
    if np.isfinite(direct).any():
        return _stat(direct, fn), "recorded"
    if denom not in data:
        return float("nan"), "unavailable"
    derived = numer / (_positive(_col(data, denom, n_cfg, n_run)[ci]) / 1000.0)
    if not np.isfinite(derived).any():
        return float("nan"), "unavailable"
    return _stat(derived, fn), "derived"


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


def build_table(npz_dir: Path, config: str | None, stat: str,
                throughput: str = "both") -> str:
    fn = {"median": np.median, "mean": np.mean}[stat]
    thru_cols = THROUGHPUT_COLUMNS[throughput]
    header = (["shape", "batch", "TTFT (s)", "TPOT_steady (s)"]
              + [label for label, _, _ in thru_cols]
              + ["peak_GB", "steadyKV_GB"])
    widths = ([11, 6, 9, 16] + [w for _, w, _ in thru_cols] + [9, 12])

    def fmt_row(vals) -> str:
        return "  ".join(str(v).ljust(w) if i < 2 else str(v).rjust(w)
                         for i, (v, w) in enumerate(zip(vals, widths)))

    lines = [fmt_row(header), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    #: one dash per column an OOM/ERROR row has no number for: everything after
    #: shape, batch and the OOM/ERROR marker itself. Sized off `widths` so adding
    #: a throughput column cannot leave a short row shifting under the header.
    blank = ["-"] * (len(widths) - 3)

    any_row = False
    notes: list = []
    ages: list = []
    sources: dict = {kind: set() for _, _, kind in thru_cols}
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
            lines.append(fmt_row([shape, batch, "OOM"] + blank))
            notes.append((shape, batch, "OOM", reason))
            continue
        if err or skipped:
            # NOT dashes. A row of dashes reads as "nothing happened", which is
            # how a deliberate hard error (the strict L-reuse miss) got mistaken
            # for a slow-but-successful cell. The reason is in the npz and has
            # always been; it was simply never printed.
            lines.append(fmt_row([shape, batch, "ERROR"] + blank))
            notes.append((shape, batch, "ERROR" if err else "SKIPPED", reason))
            continue

        ttft = _stat(_col(data, "ttft_ms", n_cfg, n_run)[ci], fn) / 1000.0
        tpot_ss = _stat(_col(data, "tpot_steady_ms", n_cfg, n_run)[ci], fn) / 1000.0
        peak = _stat(_col(data, "peak_memory_mb", n_cfg, n_run)[ci], fn) / MB_PER_GB
        skv = _stat(_col(data, "peak_decode_steady_mb", n_cfg, n_run)[ci], fn) / MB_PER_GB

        rates = []
        for _, _, kind in thru_cols:
            if kind == "decode":
                # batch / TPOT_steady. Deliberately NOT tpot_ms: that averages in
                # decode step 0, where this design compacts the whole prompt, so a
                # rate built on it charges a one-off O(prefill) cost into every
                # token with a bias decaying as 1/gen_len.
                value, source = _rate(data, ci, n_cfg, n_run, fn,
                                      field="throughput_decode_tokps",
                                      denom="tpot_steady_ms", numer=float(batch))
            elif kind == "e2e":
                value, source = _rate(data, ci, n_cfg, n_run, fn,
                                      field="throughput_e2e_tokps",
                                      denom="e2e_latency_ms",
                                      numer=float(batch * gen))
            else:
                value = _stat(_col(data, "throughput_tokps", n_cfg, n_run)[ci], fn)
                source = "recorded" if value == value else "unavailable"
            sources[kind].add(source)
            rates.append(value)

        def g(v, p):
            return "-" if v != v else f"{v:.{p}f}"

        lines.append(fmt_row([shape, batch, g(ttft, 3), g(tpot_ss, 4)]
                             + [g(r, 1) for r in rates]
                             + [g(peak, 2), g(skv, 2)]))

    if not any_row:
        hint = f" matching --config {config!r}" if config else ""
        lines.append(f"(no perf npz found in {npz_dir}{hint})")
        return "\n".join(lines)

    if notes:
        lines.append("")
    for shape, batch, kind, reason in notes:
        lines.append(f"  {kind}  {shape} batch={batch}: "
                     f"{reason or '(no reason recorded in the npz)'}")

    # Where each rate came from. A derived figure is the same formula the runner
    # would have written, but the reader should not have to assume that.
    provenance = {
        "decode": ("dec tok/s derived as batch / TPOT_steady",
                   "dec tok/s unavailable: the npz has neither "
                   "throughput_decode_tokps nor tpot_steady_ms"),
        "e2e": ("e2e tok/s derived as batch*gen_len / e2e_latency",
                "e2e tok/s unavailable: the npz has neither throughput_e2e_tokps "
                "nor e2e_latency_ms. throughput_tokps is NOT substituted -- its "
                "denominator omits the prefill->decode gap, so it is a different "
                "quantity"),
        # The legacy column gets no footnote in either direction: it has no
        # derivation, and a note the old printer never printed would break the
        # byte-identity that is this mode's whole reason to exist. A missing
        # value prints "-", exactly as it did before.
        "legacy": ("", ""),
    }
    footnotes = []
    for _, _, kind in thru_cols:
        derived_msg, missing_msg = provenance[kind]
        if "derived" in sources[kind] and derived_msg:
            footnotes.append(f"  note: {derived_msg} (npz predates the "
                             f"runner's own field; same formula).")
        if "unavailable" in sources[kind] and missing_msg:
            footnotes.append(f"  note: {missing_msg}.")
    if footnotes:
        lines.append("")
        lines.extend(footnotes)

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
    ap.add_argument("--throughput", default="both",
                    choices=sorted(THROUGHPUT_COLUMNS),
                    help="Which throughput column(s) to print: both (default), "
                         "decode (batch/TPOT_steady), e2e (batch*gen/e2e_latency), "
                         "legacy (the npz's throughput_tokps, in the original "
                         "7-column layout -- for reproducing an old table only), "
                         "or all three side by side.")
    ap.add_argument("--out", default=None, type=Path,
                    help="Also write the table to this file.")
    args = ap.parse_args()

    table = build_table(args.npz_dir, args.config, args.stat, args.throughput)
    if args.throughput == "legacy":
        # stderr, so stdout and --out stay byte-identical to the old table.
        print("[warn] --throughput legacy prints throughput_tokps, whose "
              "denominator (gen_time + TTFT) omits prefill_to_decode_gap_ms. It "
              "is neither the decode rate nor the e2e rate and it overstates "
              "both; perf_runner documents it as legacy and says not to quote "
              "it. Use --throughput both for a figure you can publish.",
              file=sys.stderr)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"\n[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
