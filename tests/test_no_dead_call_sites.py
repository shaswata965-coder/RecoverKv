"""Every name the production modules call has to exist.

This file exists because of one bug and the shape of it. ``0974687`` deleted 21
environment knobs; two of their READERS survived the deletion:

    score_kernel.py:537   use_exp2 = _score_exp2_enabled()
    hooks.py:487          elif _lse_strict():

Both are on CUDA-only paths -- ``_token_scores_triton`` refuses non-CUDA tensors
two lines above the first, and the second only runs when an L-reuse capture
misses -- so no CPU test could reach either. They shipped, and the next perf
table raised ``NameError: name '_score_exp2_enabled' is not defined`` on every
cell of a six-cell run: a GPU box, a loaded 8B model, and every row lost to a
name that was never going to resolve.

A name that does not exist is not a runtime question. Pyflakes builds the same
scope graph the interpreter does and answers it statically, for every line,
including the ones a CPU box cannot execute -- which is exactly the set the rest
of this suite cannot cover. So this is the one check here that is *not* about
what the code computes, and it is the cheapest insurance in the repo.

Zero tolerance, no allowlist. The two false positives this used to have were
removed at the source instead (`perf_runner`'s ``del model`` under closures that
read it, and ``base_parity_runner``'s loop-carried ``next_tok``), because an
allowlist is where the next real one would hide.
"""

from __future__ import annotations

import pathlib

import pytest

pyflakes_api = pytest.importorskip(
    "pyflakes.api", reason="pyflakes not installed (pip install pyflakes)")
from pyflakes import reporter as pyflakes_reporter  # noqa: E402

#: Production code. `tests/` is deliberately out: a stale test is a failing
#: test, which is visible; a stale call site in a module is a GPU hour.
ROOTS = ("modules", "utils", "scripts", "main.py")


class _Collect(pyflakes_reporter.Reporter):
    """Keep only undefined-name reports; pyflakes has no structured filter."""

    def __init__(self):
        self.found: list[str] = []

    def unexpectedError(self, filename, msg):        # noqa: N802
        self.found.append(f"{filename}: {msg}")

    def syntaxError(self, filename, msg, lineno, offset, text):  # noqa: N802
        self.found.append(f"{filename}:{lineno}: syntax error: {msg}")

    def flake(self, message):
        if type(message).__name__ in ("UndefinedName", "UndefinedLocal",
                                      "UndefinedExport"):
            self.found.append(str(message))


def test_no_undefined_names_in_production_code():
    root = pathlib.Path(__file__).resolve().parent.parent
    rep = _Collect()
    for target in ROOTS:
        pyflakes_api.checkRecursive([str(root / target)], rep)
    assert not rep.found, (
        "a call site names something that does not exist:\n  "
        + "\n  ".join(rep.found)
        + "\n\nIf this is the remains of a deleted knob, delete the READER too "
          "and make its default the behaviour -- that is what the deletion "
          "meant. A CUDA-only line is exactly the line this check is for."
    )
