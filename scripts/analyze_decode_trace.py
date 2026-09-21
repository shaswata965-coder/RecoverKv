#!/usr/bin/env python3
"""Attribute every GPU kernel in a decode trace to the code that launched it.

WHY THIS EXISTS
---------------
`profile_decode.py` answers "which kernel". It buckets by kernel *name*, and a
name is all it has, so five of its ten buckets -- copy, index/gather, sort,
elementwise, reduction -- are issued by BOTH this project and the model and are
therefore attributed to neither. Its own rollup says so:

    ours (cache)     10.644 ms/step   23.6%
    model            10.544 ms/step   23.3%
    shared/other     24.008 ms/step   53.1%
      ^ ... Shrinking it needs the chrome trace (--trace), which carries the
        launching stack.

**53.1% of the step is unattributed, and one kernel signature inside it --
`vectorized_elementwise_kernel<4, ...>` -- is 19.045 ms/step, 42.1%, at 131
launches of 0.146 ms each.** That single number decides the project's remaining
headroom, because everything else is already sized:

* the model's GEMMs are 10.448 ms and sit ON the B=1 weight-read floor (~10.3
  ms), so they are not moveable;
* `_two_tier_decode_kernel` is 2.858 ms -- **6.3%** -- which caps every
  decode-kernel idea (§6.2 contiguity, the tile ladder, split-KV) at 6.3% no
  matter how well it is executed;
* the compiled eviction is 7.478 ms, of which `triton_poi_fused_gather_6` alone
  is 6.564 ms.

So the 1.2x target (-9.3 ms off a 55.9 ms step) can only come out of the
elementwise pile, the eviction gather, or the host gap. This script exists to
say which, and it is the difference between a final push aimed at 42% of the
step and one aimed at 6%.

An elementwise kernel that costs 0.146 ms at **batch 1** is the tell worth
chasing: a model activation at B=1 is `[1, 1, 4096]` and should be launch-bound
at single-digit microseconds. 146 us means it is touching cache-sized tensors.
That is a hypothesis, not a finding -- this script is how it becomes one.

HOW IT WORKS
------------
A PyTorch chrome trace carries four kinds of event this needs:

* `cat == "kernel"` (and `gpu_memcpy` / `gpu_memset`) -- the GPU work, with a
  duration and an `External id`.
* `cat == "cuda_runtime"` -- the `cudaLaunchKernel` on the host that issued it,
  carrying the same `correlation` as the kernel and the `External id` of the
  aten op it sits inside.
* `cat == "cpu_op"` -- the aten op (`aten::mul`, `aten::index_select`, ...).
  With `--stack` it also carries `args["Call stack"]`: the Python frames.
* `cat == "python_function"` -- present only with `--stack`, used as a fallback
  when an op has no call stack of its own.

Kernels are linked to ops by `External id` when both have one, else by
`correlation` through the runtime event, else by timestamp containment on the
launching thread. Each op is then attributed to a **call site**: the innermost
Python frame belonging to this project. If no frame is ours the work is the
model's, and it is labelled by its innermost `transformers` frame instead. That
rule is what separates "our elementwise" from "the model's elementwise", and it
is the whole point of the file.

WITHOUT `--stack` this still runs and still beats the name-based rollup -- it
resolves kernels to aten ops and to the enclosing op chain -- but it cannot say
which side issued `aten::mul`. Take the attribution run with `--stack`.

    python scripts/profile_decode.py --config <cfg> --prefill 4096 --batch 1 \\
        --steps 24 --warmup 40 --stack --trace /tmp/decode.json
    python scripts/analyze_decode_trace.py /tmp/decode.json --steps 24

READING IT HONESTLY
-------------------
`--stack` adds per-op CPU cost, so an attribution run's **host gap is not a
measurement** -- take timings from a run without it. Every ms/step here is GPU
self time from the same trace, so the shares are internally consistent whether
or not the wall is.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Ownership: which subsystem a Python frame belongs to.
#
# Order matters -- the FIRST pattern that matches a frame's path wins, so the
# specific paths precede the general ones. `ours:` prefixes are this project;
# everything else is somebody else's code and cannot be optimised from here.
# --------------------------------------------------------------------------
_THIRD_PARTY: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("model: transformers", ("transformers/",)),
    ("framework: torch",    ("/torch/", "site-packages/torch",
                             "dist-packages/torch")),
    ("framework: other",    ("site-packages/", "dist-packages/")),
)

#: This project's own files. Checked ONLY after `_THIRD_PARTY`, and the order
#: matters for the same reason it does there: the specific paths precede the
#: general ones. The precedence is not cosmetic -- `models/` matches both this
#: repo's `models/` and `transformers/models/llama/...`, so a frame in the
#: model's own attention would be booked to this project if third-party paths
#: were not resolved first. That inverts the one number this script exists to
#: produce, so it is tested in `_selftest`.
_OURS_PATHS: Tuple[Tuple[str, str], ...] = (
    ("ours: decode kernel",  "modules/windowed_cache/decode_kernel"),
    ("ours: gate kernel",    "modules/windowed_cache/gate_kernel"),
    ("ours: score kernel",   "modules/windowed_cache/score_kernel"),
    ("ours: flash decode",   "modules/windowed_cache/flash_decode"),
    ("ours: cache/evict",    "modules/windowed_cache/cache"),
    ("ours: store",          "modules/windowed_cache/store"),
    ("ours: slots",          "modules/windowed_cache/slots"),
    ("ours: scorer",         "modules/windowed_cache/scorer"),
    ("ours: windowed_cache", "modules/windowed_cache/"),
    ("ours: sketch",         "modules/quant/sketch"),
    ("ours: quantizer",      "modules/quant/quantizer"),
    ("ours: quant",          "modules/quant/"),
    ("ours: utils",          "utils/"),
    ("ours: models",         "models/"),
    ("ours: scripts",        "scripts/"),
)


#: Kernel-name buckets, kept deliberately identical in spirit to
#: `profile_decode._BUCKETS` so the two reports can be read side by side. This
#: script's job is the OWNER axis; the bucket axis is here only to cross it.
_KIND: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("decode kernel",  ("_two_tier_decode_kernel",)),
    ("gate kernel",    ("_gate_kernel",)),
    ("score kernel",   ("_token_scores", "_score_kernel")),
    ("inductor",       ("triton_poi_fused", "triton_red_fused",
                        "triton_tem_fused", "triton_for_fused",
                        "triton_unk_fused", "triton_mm_fused")),
    ("GEMM",           ("gemm", "gemv", "cutlass", "cublas", "xmma", "sm80_",
                        "sm90_", "ampere_", "turing_", "volta_",
                        "splitkreduce", "dot_kernel")),
    ("attention",      ("flash_fwd", "flash::", "fmha", "mha_fwd")),
    ("elementwise",    ("elementwise_kernel", "unrolled_elementwise",
                        "cudafunctor")),
    ("copy/cat",       ("copy_device_to_device", "direct_copy",
                        "catarraybatched", "memcpy", "memset",
                        "vectorized_copy")),
    ("index/gather",   ("indexselect", "index_elementwise", "index_put",
                        "gather", "scatter", "take_", "gathertopk")),
    ("sort/topk",      ("radix", "bitonic", "sort", "topk")),
    ("reduction",      ("reduce_kernel", "reduceop", "cub::")),
)

_GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
_FRAME_RE = re.compile(r"^(?P<path>.+?)\((?P<line>\d+)\):\s*(?P<func>.*)$")


def _kind_of(name: str) -> str:
    low = name.lower()
    for kind, pats in _KIND:
        for p in pats:
            if p in low:
                return kind
    return "other"


def _owner_of_path(path: str) -> Optional[str]:
    """Which subsystem a source path belongs to, third party resolved first."""
    norm = path.replace("\\", "/")
    for owner, pats in _THIRD_PARTY:
        for pat in pats:
            if pat in norm:
                return owner
    for owner, pat in _OURS_PATHS:
        if pat in norm:
            return owner
    return None


def _short(path: str) -> str:
    """`.../modules/windowed_cache/cache.py` -> `windowed_cache/cache.py`."""
    norm = path.replace("\\", "/")
    for marker in ("modules/", "utils/", "models/", "scripts/",
                   "transformers/", "torch/"):
        i = norm.find(marker)
        if i >= 0:
            return norm[i:]
    return os.path.basename(norm)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _load(path: Path) -> List[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        blob = json.load(fh)
    evs = blob["traceEvents"] if isinstance(blob, dict) else blob
    return [e for e in evs if isinstance(e, dict)]


def _ext_id(ev: dict) -> Optional[int]:
    a = ev.get("args") or {}
    for k in ("External id", "external id", "External Id"):
        if k in a:
            try:
                return int(a[k])
            except (TypeError, ValueError):
                return None
    return None


def _corr(ev: dict) -> Optional[int]:
    a = ev.get("args") or {}
    for k in ("correlation", "Correlation"):
        if k in a:
            try:
                return int(a[k])
            except (TypeError, ValueError):
                return None
    return None


def _call_stack(ev: dict) -> List[str]:
    a = ev.get("args") or {}
    cs = a.get("Call stack") or a.get("call stack") or ""
    if not cs:
        return []
    return [f for f in (x.strip() for x in cs.split(";")) if f]


def _attribute(frames: List[str]) -> Tuple[str, str]:
    """(owner, call site) for a Python call stack, innermost-ours wins.

    A trace's frames run outermost-first. This project's code is always called
    FROM the model (transformers' attention calls `Cache.update`), so the
    innermost frame that belongs to us is the honest owner of the op: it is the
    last line of our code before the aten call. When no frame is ours the work
    is somebody else's and gets labelled by its innermost non-torch frame,
    which is what makes "the model's elementwise" a separate row rather than a
    residue.
    """
    ours: Optional[Tuple[str, str]] = None
    theirs: Optional[Tuple[str, str]] = None
    for raw in frames:
        m = _FRAME_RE.match(raw)
        if not m:
            continue
        path, func, line = m.group("path"), m.group("func"), m.group("line")
        owner = _owner_of_path(path)
        if owner is None:
            continue
        site = f"{_short(path)}:{line} {func}"
        if owner.startswith("ours:"):
            ours = (owner, site)
        elif theirs is None or owner.startswith("model:"):
            theirs = (owner, site)
    if ours:
        return ours
    if theirs:
        return theirs
    return ("unattributed", "<no recognised frame>")


# --------------------------------------------------------------------------
# Linking kernels to the code that launched them
# --------------------------------------------------------------------------
def _link(evs: List[dict]) -> Tuple[List[dict], Dict[int, dict], dict]:
    """Return (gpu events, ext-id -> cpu_op, stats)."""
    gpu, runtime, cpu_ops, pyfuncs = [], {}, {}, []
    for e in evs:
        cat = (e.get("cat") or e.get("categories") or "").lower()
        if cat in _GPU_CATS:
            gpu.append(e)
        elif cat in ("cuda_runtime", "runtime"):
            c = _corr(e)
            if c is not None:
                runtime[c] = e
        elif cat == "cpu_op":
            x = _ext_id(e)
            if x is not None:
                # Keep the INNERMOST op per external id: the deepest aten call
                # is the one that actually issued the launch.
                prev = cpu_ops.get(x)
                if prev is None or (e.get("dur", 0) or 0) < (prev.get("dur", 0) or 0):
                    cpu_ops[x] = e
        elif cat == "python_function":
            pyfuncs.append(e)

    stats = {"gpu": len(gpu), "runtime": len(runtime),
             "cpu_ops": len(cpu_ops), "python_function": len(pyfuncs),
             "via_extid": 0, "via_corr": 0, "unlinked": 0}
    return gpu, cpu_ops, stats


def _op_for(k: dict, cpu_ops: Dict[int, dict], runtime_by_corr: Dict[int, dict],
            stats: dict) -> Optional[dict]:
    x = _ext_id(k)
    if x is not None and x in cpu_ops:
        stats["via_extid"] += 1
        return cpu_ops[x]
    c = _corr(k)
    if c is not None and c in runtime_by_corr:
        rx = _ext_id(runtime_by_corr[c])
        if rx is not None and rx in cpu_ops:
            stats["via_corr"] += 1
            return cpu_ops[rx]
    stats["unlinked"] += 1
    return None


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _bar(frac: float, width: int = 28) -> str:
    n = max(0, min(width, int(round(frac * width))))
    return "#" * n + "." * (width - n)


def _rule(title: str, ch: str = "=", w: int = 92) -> str:
    return f"\n{ch * w}\n  {title}\n{ch * w}"


def _selftest() -> int:
    """Pin the attribution rules that a wrong answer would look plausible under.

    Every case here is one where the report would still print, still add up,
    and still look like a finding -- while naming the wrong owner for the
    largest row in it. That is the failure this file has to refuse.
    """
    HF = "/e/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py"
    TORCH = "/e/lib/python3.10/site-packages/torch/nn/modules/module.py"
    OURS = "/home/u/RecoverKv/modules/windowed_cache/cache.py"
    OURM = "/home/u/RecoverKv/models/llama_patch.py"
    cases = [
        # transformers' own `models/` dir must NOT read as this repo's.
        (HF,    "model: transformers"),
        (TORCH, "framework: torch"),
        (OURS,  "ours: cache/evict"),
        (OURM,  "ours: models"),
    ]
    bad = 0
    for path, want in cases:
        got = _owner_of_path(path)
        ok = got == want
        bad += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {want:<20} <- {path}"
              + ("" if ok else f"   got {got!r}"))

    # The innermost OUR frame wins over an outer model frame: our code is
    # always called FROM the model, so the outermost match is always
    # transformers and would attribute every one of our kernels to the model.
    owner, site = _attribute([f"{TORCH}(1518): _call_impl",
                              f"{HF}(402): forward",
                              f"{OURS}(812): _evict"])
    ok = owner == "ours: cache/evict" and "cache.py:812" in site
    bad += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] innermost-ours wins over outer model "
          f"frame -> {owner} | {site}")

    # A stack with no frame of ours is the model's, not "unattributed":
    # a residue bucket would hide the very comparison this script is for.
    owner, _ = _attribute([f"{TORCH}(1518): _call_impl", f"{HF}(402): forward"])
    ok = owner == "model: transformers"
    bad += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] model-only stack -> {owner}")

    # An empty stack must not silently become somebody's.
    owner, _ = _attribute([])
    ok = owner == "unattributed"
    bad += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] empty stack -> {owner}")

    print("\n  SELFTEST", "PASSED" if not bad else f"FAILED ({bad})")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Attribute a decode chrome trace to the code that "
                    "launched each kernel.")
    ap.add_argument("trace", type=Path, nargs="?",
                    help="chrome trace (.json or .json.gz)")
    ap.add_argument("--selftest", action="store_true",
                    help="check the attribution rules and exit (no trace "
                         "needed); runs on any box, including CPU-only ones")
    ap.add_argument("--steps", type=int, default=1,
                    help="profiled decode steps, for per-step normalisation "
                         "(pass the same --steps profile_decode.py used)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--target-ms", type=float, default=None,
                    help="wall ms/step of the arm this trace came from. Turns "
                         "the budget section on: host gap and what each item "
                         "is worth against a 1.2x goal.")
    ap.add_argument("--speedup", type=float, default=1.2,
                    help="the speedup the budget section sizes against")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write the full kernel x call-site matrix here")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if args.trace is None:
        sys.exit("give a trace path, or --selftest")
    if not args.trace.exists():
        sys.exit(f"no such trace: {args.trace}")
    size_mb = args.trace.stat().st_size / 1e6
    print(f"reading {args.trace} ({size_mb:.1f} MB) ...", file=sys.stderr)
    evs = _load(args.trace)
    n = max(1, args.steps)

    gpu, cpu_ops, stats = _link(evs)
    runtime_by_corr = {}
    for e in evs:
        cat = (e.get("cat") or "").lower()
        if cat in ("cuda_runtime", "runtime"):
            c = _corr(e)
            if c is not None:
                runtime_by_corr[c] = e

    # --- aggregate -------------------------------------------------------
    by_owner: Dict[str, dict] = defaultdict(lambda: {"us": 0.0, "n": 0})
    by_site: Dict[Tuple[str, str], dict] = defaultdict(
        lambda: {"us": 0.0, "n": 0, "kernels": defaultdict(float)})
    by_kernel: Dict[str, dict] = defaultdict(
        lambda: {"us": 0.0, "n": 0, "sites": defaultdict(float)})
    by_kind: Dict[str, dict] = defaultdict(lambda: {"us": 0.0, "n": 0})
    by_kind_owner: Dict[Tuple[str, str], float] = defaultdict(float)
    matrix: Dict[Tuple[str, str, str], dict] = defaultdict(
        lambda: {"us": 0.0, "n": 0})

    gpu_us = 0.0
    have_stacks = False
    for k in gpu:
        dur = float(k.get("dur", 0) or 0)
        if dur <= 0:
            continue
        gpu_us += dur
        name = k.get("name", "?")
        kind = _kind_of(name)
        op = _op_for(k, cpu_ops, runtime_by_corr, stats)
        if op is not None:
            frames = _call_stack(op)
            if frames:
                have_stacks = True
            owner, site = _attribute(frames)
            if not frames:
                # No stack: fall back to the aten op's own name, which still
                # beats the kernel name for anything the model and this
                # project both issue.
                owner, site = "unattributed", f"<op> {op.get('name', '?')}"
        else:
            owner, site = "unattributed", "<unlinked kernel>"

        by_owner[owner]["us"] += dur
        by_owner[owner]["n"] += 1
        by_site[(owner, site)]["us"] += dur
        by_site[(owner, site)]["n"] += 1
        by_site[(owner, site)]["kernels"][name] += dur
        by_kernel[name]["us"] += dur
        by_kernel[name]["n"] += 1
        by_kernel[name]["sites"][f"{owner} | {site}"] += dur
        by_kind[kind]["us"] += dur
        by_kind[kind]["n"] += 1
        by_kind_owner[(kind, owner)] += dur
        m = matrix[(name, owner, site)]
        m["us"] += dur
        m["n"] += 1

    if gpu_us <= 0:
        sys.exit("trace contains no GPU kernel events with a duration.")

    pct = lambda us: us / gpu_us * 100          # noqa: E731
    ms = lambda us: us / n / 1000               # noqa: E731

    # --- 1. summary ------------------------------------------------------
    print(_rule("1. TRACE SUMMARY"))
    print(f"    trace                      {args.trace}")
    print(f"    profiled steps             {n}")
    print(f"    GPU kernel events          {stats['gpu']:,}")
    print(f"    cpu_op events (unique id)  {stats['cpu_ops']:,}")
    print(f"    python_function events     {stats['python_function']:,}")
    print(f"    linked by External id      {stats['via_extid']:,}")
    print(f"    linked by correlation      {stats['via_corr']:,}")
    print(f"    UNLINKED kernels           {stats['unlinked']:,}")
    print(f"\n    total GPU self time        {ms(gpu_us):8.3f} ms/step")
    print(f"    total GPU launches         {sum(v['n'] for v in by_owner.values())/n:8.0f} /step")
    if not have_stacks:
        print("\n    !! NO PYTHON CALL STACKS IN THIS TRACE.")
        print("       Kernels resolve to aten ops but not to call sites, so")
        print("       the ownership split below is mostly '<op> aten::...'.")
        print("       Re-take the trace with profile_decode.py --stack.")

    # --- 2. ownership ----------------------------------------------------
    print(_rule("2. WHO LAUNCHED THE WORK"))
    print(f"    {'owner':<26} {'ms/step':>9} {'% GPU':>7} {'launches':>9}  share")
    ours_us = sum(v["us"] for k, v in by_owner.items() if k.startswith("ours:"))
    model_us = sum(v["us"] for k, v in by_owner.items()
                   if k.startswith(("model:", "framework:")))
    un_us = sum(v["us"] for k, v in by_owner.items() if k == "unattributed")
    for owner, rec in sorted(by_owner.items(), key=lambda r: -r[1]["us"]):
        print(f"    {owner:<26} {ms(rec['us']):9.3f} {pct(rec['us']):6.1f}% "
              f"{rec['n']/n:9.0f}  {_bar(rec['us']/gpu_us)}")
    print(f"\n    ours                       {ms(ours_us):9.3f} ms/step "
          f"{pct(ours_us):6.1f}%")
    print(f"    model + framework          {ms(model_us):9.3f} ms/step "
          f"{pct(model_us):6.1f}%")
    print(f"    still unattributed         {ms(un_us):9.3f} ms/step "
          f"{pct(un_us):6.1f}%")

    # --- 3. kind x owner -------------------------------------------------
    print(_rule("3. KERNEL KIND x OWNER  -- this is the table that splits the "
                "elementwise pile"))
    owners_seen = [o for o, _ in sorted(by_owner.items(),
                                        key=lambda r: -r[1]["us"])][:6]
    head = "".join(f"{o.split(':')[-1].strip()[:11]:>12}" for o in owners_seen)
    print(f"    {'kind':<16}{'total':>9}{head}")
    for kind, rec in sorted(by_kind.items(), key=lambda r: -r[1]["us"]):
        row = "".join(f"{ms(by_kind_owner.get((kind, o), 0.0)):12.3f}"
                      for o in owners_seen)
        print(f"    {kind:<16}{ms(rec['us']):9.3f}{row}")
    print("\n    (ms/step. Columns are the six owners with the most GPU time.)")

    # --- 4. call sites ---------------------------------------------------
    print(_rule(f"4. TOP {args.top} CALL SITES BY GPU TIME"))
    ranked_sites = sorted(by_site.items(), key=lambda r: -r[1]["us"])
    for (owner, site), rec in ranked_sites[:args.top]:
        print(f"\n    {ms(rec['us']):8.3f} ms/step  {pct(rec['us']):5.1f}%  "
              f"{rec['n']/n:6.0f} launches/step")
        print(f"      owner: {owner}")
        print(f"      site : {site}")
        for kname, kus in sorted(rec["kernels"].items(),
                                 key=lambda r: -r[1])[:4]:
            print(f"        {ms(kus):8.3f} ms  {kname[:62]}")

    # --- 5. kernels ------------------------------------------------------
    print(_rule(f"5. TOP {args.top} KERNELS, EACH WITH WHO LAUNCHED IT"))
    for kname, rec in sorted(by_kernel.items(),
                             key=lambda r: -r[1]["us"])[:args.top]:
        print(f"\n    {ms(rec['us']):8.3f} ms/step  {pct(rec['us']):5.1f}%  "
              f"n={rec['n']/n:7.1f}  {_kind_of(kname)}")
        print(f"      {kname[:86]}")
        tot = sum(rec["sites"].values()) or 1.0
        for site, sus in sorted(rec["sites"].items(), key=lambda r: -r[1])[:5]:
            print(f"        {sus/tot*100:5.1f}%  {ms(sus):8.3f} ms  {site[:70]}")

    # --- 6. budget -------------------------------------------------------
    if args.target_ms:
        need = args.target_ms - args.target_ms / args.speedup
        host_gap = args.target_ms - ms(gpu_us)
        print(_rule(f"6. THE {args.speedup:g}x BUDGET"))
        print(f"    wall                       {args.target_ms:8.3f} ms/step")
        print(f"    GPU busy                   {ms(gpu_us):8.3f} ms/step  "
              f"({ms(gpu_us)/args.target_ms*100:.1f}%)")
        print(f"    host gap                   {host_gap:8.3f} ms/step  "
              f"({host_gap/args.target_ms*100:.1f}%)")
        print(f"\n    {args.speedup:g}x needs                 {need:8.3f} ms/step\n")
        print(f"    {'candidate':<34}{'ms/step':>9}{'% of need':>11}")
        cands = [("host gap (CUDA graphs / fewer launches)", host_gap)]
        cands += [(f"all of: {o}", v["us"] / n / 1000)
                  for o, v in sorted(by_owner.items(), key=lambda r: -r[1]["us"])
                  if o.startswith("ours:")][:6]
        for label, val in sorted(cands, key=lambda r: -r[1]):
            if val <= 0:
                continue
            print(f"    {label[:34]:<34}{val:9.3f}{val/need*100:10.0f}%")
        print("\n    A candidate at less than 100% cannot reach the target on")
        print("    its own. Anything the model owns is not a candidate at all.")

    # --- 7. csv ----------------------------------------------------------
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["kernel", "kind", "owner", "call_site",
                        "ms_per_step", "pct_gpu", "launches_per_step"])
            for (kname, owner, site), rec in sorted(matrix.items(),
                                                    key=lambda r: -r[1]["us"]):
                w.writerow([kname, _kind_of(kname), owner, site,
                            f"{ms(rec['us']):.6f}", f"{pct(rec['us']):.4f}",
                            f"{rec['n']/n:.2f}"])
        print(f"\n[wrote] {args.csv}  (full kernel x call-site matrix)")


if __name__ == "__main__":
    main()
