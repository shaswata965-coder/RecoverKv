#!/usr/bin/env python
"""Would turning the CUDA decode graph on actually buy anything? Usually: no.

    python scripts/check_decode_graph.py        # exit 0 = usable, 1 = refused

Run by ``run_perf_table.sh`` before a sweep whenever ``--graph`` / ``DECODE_GRAPH``
asks for the graph, so the answer arrives in the first second instead of after
minutes of work.

Why it exists
-------------
``OUT_DIR=outputs/table_v5_graph ... --graph on`` came back with six ERROR rows
and no usable number. Three separate things went wrong, and only one of them was
the graph's fault in an obvious way:

* ``4096/257 batch=1`` died with ``it->second->use_count > 0 INTERNAL ASSERT
  FAILED`` in the CUDA caching allocator, on **slot 1** -- the first capture
  taken after an invalidation. The captured outputs were still alive when the
  graphs (and so their shared memory pool) were destroyed.
* ``2048/513 batch=32`` ran out of memory, with 16.18 GiB reserved-but-unallocated.
* Every remaining row then reported *"already failed on this build"*, because
  that OOM had been filed as a torch.compile lowering failure, which is sticky.

The second and third are fixed in ``modules/windowed_cache/cache.py``. The first
is fixed in ``graph_decode.invalidate``. But fixing the crash does not make the
graph *worth running*, and that is what this checks: the capture schedule never
replays anything, so the best case after the crash is fixed is a slower run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.windowed_cache.graph_decode import (  # noqa: E402
    EpochSchedule, graph_mode, replay_is_reachable,
)
from utils.config import FIRST_EVICTION_STEP_DEFAULT, OPERATING_POINT  # noqa: E402


def main() -> int:
    mode = graph_mode()
    if mode == "off":
        print("[decode-graph] off — the ordinary decode loop. Nothing to check.")
        return 0

    sched = EpochSchedule(
        window_size=int(OPERATING_POINT["window_size"]),
        first_eviction_step=int(OPERATING_POINT.get(
            "first_eviction_step", FIRST_EVICTION_STEP_DEFAULT)),
    )
    why = replay_is_reachable(sched)
    if why is None:
        print("[decode-graph] on — captures are reachable for replay.")
        return 0

    print("")
    print("=" * 72)
    print("REFUSED: STICKYKV_DECODE_GRAPH / --graph is on, and it cannot help.")
    print("=" * 72)
    print("")
    print(f"  {why}")
    print("")
    print("  Running it anyway would capture ~900 graphs over a 1024-token")
    print("  generation, replay none of them, and tear down the graph memory")
    print("  pool once per window epoch. That is strictly slower than the")
    print("  default path, and it is what produced six ERROR rows in")
    print("  outputs/table_v5_graph.")
    print("")
    print("  Do this instead — drop --graph / DECODE_GRAPH and rerun:")
    print("")
    print("    the default path IS the fast path, and it is also the loop")
    print("    model.generate() uses, so the throughput table and the LongBench")
    print("    scores describe the same code.")
    print("")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
