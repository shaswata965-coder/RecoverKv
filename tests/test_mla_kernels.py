"""The MLA gate + fused decode kernels under the Triton interpreter, on CPU.

Runs ``tests/interp_mla_kernels.py`` in a subprocess with ``TRITON_INTERPRET=1``
(it must be set before Triton is imported). That script checks, against the
shipped CPU oracle ``decode_kernel.two_tier_window_reference`` over a real int2
latent tier with real cards:

* the gate kernel's log-mass and its share-union pick at ``rep = H_q``;
* the MLA decode kernel's latent output and every window score, gated and not;
* the production route end to end on a tiny DeepSeek-V2 -- owned forward ->
  ``run_pending`` -> ``_run_fused`` -> both kernels -> scores in the cache --
  reproducing the materialize path at ``quant_gate_ratio = 1.0`` and proving
  ``gated == fired`` with the ``gated`` verdict at 0.25.

The interpreter checks arithmetic, masking and lane maps, not GPU lowering or
speed (``CLAUDE.md``: Triton cannot run on this dev box).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_mla_kernels_under_the_triton_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1",
               PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, str(REPO / "tests" / "interp_mla_kernels.py")],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=3000)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("ok", "FAIL"))]
    assert proc.returncode == 0, "\n".join(lines) + "\n" + proc.stderr[-4000:]
    assert any(ln.startswith("ok e2e") for ln in lines), proc.stdout[-2000:]
