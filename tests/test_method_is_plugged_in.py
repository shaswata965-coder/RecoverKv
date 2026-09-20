"""Every stage of the method must be reachable FROM THE DECODE STEP.

This repository has already shipped the opposite. Between `3658f72` and
`9d131e4` the read gate was built, tested and documented with **zero non-test
callers**: the step read 100% of the int2 tier while `build_sketch` ran on
every eviction and the cards sat in memory unread. Every number taken in that
window was an ungated number, and nothing in the suite, the docs or the
profile said so -- because presence in a module, a test and a design doc is not
presence in the step.

`flash_decode.gate_report` is the runtime half of that check and covers the
gate only. This is the static half, and it covers all four stages:

    eviction       update -> _evict_two_tier -> the compiled body
    quantization   the eviction body -> demote_many -> build_sketch
    gating         update -> set_pending -> _run_fused -> fused_gate
    representative fused_two_tier_decode consumes the centroids

It is deliberately a REACHABILITY test, not a "does this symbol exist" test.
A grep for `fused_gate` passed throughout the eight commits above; what failed
was the path from the step to it.

No torch, no GPU: this parses the source. That is the point -- it is the check
that still runs on a box where the kernels cannot.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
#: Production source only. A test calling a function is exactly the evidence
#: that misled this repo before, so tests are excluded by construction.
PKGS = ("modules", "utils")


def _iter_sources():
    for pkg in PKGS:
        for p in sorted((ROOT / pkg).rglob("*.py")):
            rel = p.relative_to(ROOT).as_posix()
            if "/test" in rel or rel.split("/")[-1].startswith("test_"):
                continue
            yield rel, p


def _callees(fn: ast.AST) -> set:
    """Names this function calls: `f()`, `self.f()`, `mod.f()` all give "f"."""
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _call_graph():
    """{function name -> set of names it calls}, merged across the package.

    Merged by bare name, so it over-approximates (two classes with the same
    method name share a node). That is the safe direction for this test: it can
    only make an unreachable stage look reachable, never the reverse, so a PASS
    is weaker than it looks and a FAIL is real. The runtime counters
    (`gate_report`, `evict_path_stats`) are what make a pass meaningful.
    """
    graph: dict = {}
    for _rel, path in _iter_sources():
        try:
            tree = ast.parse(io.open(path, encoding="utf-8").read())
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                graph.setdefault(node.name, set()).update(_callees(node))
    return graph


def _reaches(graph, start: str, target: str):
    """A call path start -> ... -> target, or None."""
    if start not in graph:
        return None
    seen, stack = {start}, [(start, [start])]
    while stack:
        cur, path = stack.pop()
        for nxt in sorted(graph.get(cur, ())):
            if nxt == target:
                return path + [nxt]
            if nxt in graph and nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, path + [nxt]))
    return None


@pytest.fixture(scope="module")
def graph():
    return _call_graph()


#: (stage, entry point, the function that stage cannot happen without)
STAGES = [
    ("eviction: the step evicts",
     "update", "_evict_two_tier"),
    ("eviction: the eviction runs the compiled-or-control-arm body",
     "_evict_two_tier", "_run_compiled_evict"),
    ("quantization: the eviction demotes windows into the int2 tier",
     "_evict_two_tier_impl", "demote_many"),
    ("quantization: a demoted window gets a card",
     "demote_many", "build_sketch"),
    ("gating: the step hands the decode path its context",
     "update", "set_pending"),
    ("gating: the fused path runs the read gate",
     "_run_fused", "fused_gate"),
    ("gating: the gate's selection reaches the decode kernel",
     "_run_fused", "fused_two_tier_decode"),
]


@pytest.mark.parametrize("label,start,target",
                         STAGES, ids=[s[0] for s in STAGES])
def test_the_stage_is_reachable_from_the_step(graph, label, start, target):
    path = _reaches(graph, start, target)
    assert path is not None, (
        f"{label}: no call path from {start}() to {target}() in production "
        f"code.\nThis is the 3658f72..9d131e4 failure: the stage exists and is "
        f"tested, but the step never reaches it, so every number measured is "
        f"measuring a method that does not include it."
    )


def test_the_gate_has_a_non_test_caller():
    """The literal check CLAUDE.md prescribes, as a test rather than a habit."""
    hits = []
    for rel, path in _iter_sources():
        if rel.endswith("gate_kernel.py"):
            continue                      # its own definition does not count
        src = io.open(path, encoding="utf-8").read()
        for i, line in enumerate(src.splitlines(), 1):
            if "fused_gate(" in line and "def fused_gate" not in line:
                hits.append(f"{rel}:{i}")
    assert hits, (
        "fused_gate has NO non-test caller. The decode step is reading the "
        "whole int2 tier and every perf and quality number is ungated.")


def test_the_gated_kernel_refuses_to_run_without_the_representative(graph):
    """Representative attention cannot be silently absent.

    The skipped windows attend through their value centroids. If those were
    allowed to default, ~75% of the tier would vanish from the output at
    `quant_gate_ratio=0.25` and the run would still finish and still score.
    `fused_two_tier_decode` must RAISE instead, and the zeroed fallback must be
    reachable only on the ungated path, where the GATED blocks are compiled out.
    """
    src = io.open(ROOT / "modules/windowed_cache/decode_kernel.py",
                  encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_fused_two_tier_decode_impl"
              or isinstance(n, ast.FunctionDef) and n.name == "fused_two_tier_decode")

    # find `if centroids is None: raise ...` anywhere in the module
    raises_on_missing = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                and t.left.id == "centroids"
                and any(isinstance(o, ast.Is) for o in t.ops)
                and any(isinstance(c, ast.Constant) and c.value is None
                        for c in t.comparators)):
            if any(isinstance(s, ast.Raise) for s in ast.walk(node)):
                raises_on_missing = True
    assert raises_on_missing, (
        "a gated fused decode no longer raises when `centroids` is None. The "
        "windows the gate skips attend through the centroids; defaulting them "
        "drops most of the compressed tier from the output silently.")
    assert fn is not None


def test_the_runtime_proof_counters_still_exist():
    """A static path is necessary, not sufficient -- Dynamo declining a frame
    and a card-less store both look fine to a parser. These are the counters
    that catch what this file cannot."""
    fd = io.open(ROOT / "modules/windowed_cache/flash_decode.py",
                 encoding="utf-8").read()
    ca = io.open(ROOT / "modules/windowed_cache/cache.py",
                 encoding="utf-8").read()
    for needle, where in (("def gate_report", "flash_decode"),
                          ("def expect_gated", "flash_decode")):
        assert needle in fd, f"{needle} gone from {where}: a run can no longer prove it gated"
    for needle in ("def evict_path_stats", "def evict_path_mode",
                   "class EvictionRanEager"):
        assert needle in ca, f"{needle} gone from cache.py: the eviction's proof is gone"
