#!/usr/bin/env python
"""Do the speed runs and the quality runs describe the SAME method?

Run this before quoting any pair of numbers together.

    python scripts/check_operating_point.py

It reads every shipped config plus the perf table's generator and prints the
cache operating point each one resolves, against ``utils.config.OPERATING_POINT``.
It exits non-zero if any non-exempt config diverges.

Why it exists
-------------
Until it did, these two things were both true at once:

* ``configs/longbench_ours_flash_attn.yaml`` -- the LongBench accuracy run --
  set ``quant_ratio: 0.0``. No int2 tier. No read gate. Pure fp16 eviction.
* ``scripts/run_perf_table.sh`` -- the throughput table -- ran
  ``quant_ratio 0.70`` in ``quant_budget_mode tokens``.

So the accuracy number described a cache with no quantized tier, and the speed
number described a cache that is ~57% int2 windows. Nothing in the repo compared
the two, and nothing printed the difference. A paper quoting both would be
describing a method that was never run.

This does not *change* any config. A run is allowed to sit somewhere else on
purpose -- that is what ``OPERATING_POINT_EXEMPT`` records, with a reason each.
What it removes is the case where nobody noticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import OPERATING_POINT, OPERATING_POINT_EXEMPT  # noqa: E402

#: Fields that decide what cache a run actually builds.
FIELDS = list(OPERATING_POINT)


def _load_cache_block(path: Path) -> dict | None:
    """The ``cache:`` block of a YAML config, or ``None`` if it has none."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a hard dep of main.py
        print("PyYAML is required.", file=sys.stderr)
        raise
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"  !! {path.name}: unreadable ({exc})")
        return None
    block = doc.get("cache")
    return block if isinstance(block, dict) else None


def _perf_table_point() -> dict:
    """What ``run_perf_table.sh`` generates, read from its own defaults.

    Read rather than restated: a second copy of these numbers is exactly the
    drift this script exists to catch, and the script's header comment has
    already been wrong about two of them (it claimed budget 0.50 / local 64
    while the body did 0.20 / 128 -- the design notes).
    """
    import re

    text = (ROOT / "scripts" / "run_perf_table.sh").read_text(encoding="utf-8")
    want = {
        "cache_budget": "CACHE_BUDGET",
        "window_size": "WINDOW_SIZE",
        "num_sink_tokens": "NUM_SINK",
        "local_window_size": "LOCAL_WINDOW",
        "quant_ratio": "QUANT_RATIO",
        "quant_budget_mode": "QUANT_MODE",
        "quant_gate_ratio": "GATE_RATIO",
    }
    out: dict = {"first_eviction_step": 0}   # the generated YAML hardcodes it
    for field, var in want.items():
        m = re.search(rf'^{var}="\$\{{{var}:-([^}}]*)\}}"', text, re.M)
        if m:
            raw = m.group(1).strip()
            out[field] = raw
    return out


#: How the shell spells "derive it" for a tri-state knob: unset.
_UNSET_MEANS_NONE = {"", "derive", "derived"}


def _norm(v):
    """Compare 0.2 and '0.20' and 0.20 as the same number."""
    if v is None:
        return None
    if isinstance(v, str):
        low = v.strip().lower()
        if low in _UNSET_MEANS_NONE:
            return None
        if low in ("off", "false", "no", "0"):
            return False
        if low in ("on", "true", "yes", "1"):
            return True
        try:
            return round(float(v), 6)
        except ValueError:
            return v.strip()
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 6)
    return v


def _compare(name: str, point: dict) -> list[str]:
    """Fields that differ from the reference. Missing = not set = not compared
    against a guess: an absent field means the config inherits the schema
    default, which may or may not be the operating point, and saying "missing"
    is more honest than substituting one."""
    bad = []
    for f in FIELDS:
        if f not in point:
            continue
        if _norm(point[f]) != _norm(OPERATING_POINT[f]):
            bad.append(f"{f}={point[f]!r} (want {OPERATING_POINT[f]!r})")
    return bad


def main() -> int:
    print("Reference operating point (utils.config.OPERATING_POINT):")
    for k, v in OPERATING_POINT.items():
        print(f"    {k:22s} {v!r}")

    rows, diverged = [], []

    perf = _perf_table_point()
    bad = _compare("run_perf_table.sh", perf)
    rows.append(("run_perf_table.sh  [SPEED]", perf, bad))
    if bad:
        diverged.append(("run_perf_table.sh", bad))

    for path in sorted((ROOT / "configs").glob("*.yaml")):
        block = _load_cache_block(path)
        if block is None:
            continue
        exempt = path.name in OPERATING_POINT_EXEMPT
        bad = _compare(path.name, block)
        tag = "  [exempt]" if exempt else ""
        rows.append((path.name + tag, block, [] if exempt else bad))
        if bad and not exempt:
            diverged.append((path.name, bad))

    print("\nPer config (only the fields it sets):")
    for name, point, bad in rows:
        mark = "OK " if not bad else "!! "
        vals = "  ".join(f"{f.split('_')[-1]}={point[f]}" for f in FIELDS if f in point)
        print(f"  {mark}{name:44s} {vals}")
        for b in bad:
            print(f"        -> {b}")

    if not diverged:
        print("\nAll non-exempt configs are at the operating point.")
        return 0

    print(f"\n{len(diverged)} config(s) diverge from the operating point.\n")
    print("This is what it means for a paper: the numbers those configs produce")
    print("describe a DIFFERENT cache than the ones that match. Quoting a speed")
    print("row and an accuracy row from two different points is not a result")
    print("about one method.\n")
    print("Fix it in one of two ways, and say which in the paper:")
    print("  a) edit the config to the operating point and re-run it, or")
    print("  b) add it to utils.config.OPERATING_POINT_EXEMPT with the reason,")
    print("     which is how an intentional ablation is recorded.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
