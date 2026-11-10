#!/usr/bin/env python3
"""Refuse a finished run whose sidecars do not prove the read gate ran.

``CLAUDE.md``: "If that verdict is missing or is not ``gated``, the number
beside it is not a gated number." Every LongBench (``<dataset>.meta.json``) and
GSM8K (``meta.json``) sidecar records ``read_gate``. This reads them and exits
non-zero unless each is acceptable:

* a windowed run whose configuration expected the gate must say ``gated``;
* ``not-expected`` passes only for a FullKV (``cache_type: full_cache``) run, or
  with ``--allow-not-expected`` (an eager-backend run, which is ungated by
  design).

GSM8K sidecars also carry ``pct_examples_with_int2_tier``: the gate can only run
on examples whose cache holds an int2 tier, so it is printed beside the verdict.

    python scripts/check_read_gate.py outputs/longbench/deepseek_v2_lite_ours_flash_compression_0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _sidecars(run_dir: Path):
    return sorted(set(run_dir.glob("*.meta.json")) | set(run_dir.glob("meta.json")))


def check(run_dirs, allow_not_expected: bool = False) -> int:
    bad = 0
    seen = 0
    for d in run_dirs:
        d = Path(d)
        files = _sidecars(d)
        if not files:
            print(f"!! {d}: no sidecar (*.meta.json / meta.json) -- nothing proves what ran")
            bad += 1
            continue
        for f in files:
            seen += 1
            meta = json.loads(f.read_text(encoding="utf-8"))
            rg = meta.get("read_gate") or {}
            verdict = rg.get("verdict", "MISSING")
            full = meta.get("cache_type") == "full_cache"
            ok = verdict == "gated" or (
                verdict == "not-expected" and (full or allow_not_expected))
            extra = ""
            if "pct_examples_with_int2_tier" in meta:
                extra = f"  int2 tier on {meta['pct_examples_with_int2_tier']}% of examples"
            rf = rg.get("read_fraction")
            rf_s = "n/a" if rf is None else f"{rf:.3f}"
            print(f"{'ok' if ok else '!!'} {f}: {verdict} (fired={rg.get('fired')}, "
                  f"gated={rg.get('gated')}, read_fraction={rf_s}){extra}")
            bad += not ok
    if bad:
        print(f"\n{bad} sidecar(s) do not prove a gated run; those numbers are not "
              "the gated method's.")
    elif seen:
        print(f"\nall {seen} sidecar(s) OK")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--allow-not-expected", action="store_true",
                   help="accept `not-expected` for windowed runs (eager backend)")
    a = p.parse_args()
    return check(a.run_dirs, a.allow_not_expected)


if __name__ == "__main__":
    sys.exit(main())
