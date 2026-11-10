"""``scripts/deepseek_gpu_preflight.py`` under the Triton interpreter, on CPU.

The script exists to run on a GPU before a long DeepSeek-V2-Lite benchmark (the
Colab notebook calls it). Its ``--interpret`` mode runs the checks that do not need
FA2 -- the fused decode against the materialize path, the gate verdict at the
run's ratio, the compiled eviction -- on CPU tensors through the same kernels, so
the script itself cannot rot unnoticed. The geometry comes from the vendored
``config.json`` (V2-Lite's real attention dims), and the remote-code hash check
runs against the vendored ``modeling_deepseek.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_preflight_passes_under_the_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1",
               PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "deepseek_gpu_preflight.py"),
         "--model-dir", str(REPO / "models" / "deepseek_v2" / "upstream"),
         "--budget", "0.10", "--prompt", "2200", "--steps", "12", "--interpret"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=3000)
    lines = proc.stdout.strip().splitlines()
    assert proc.returncode == 0, "\n".join(lines[-20:]) + "\n" + proc.stderr[-4000:]
    summary = json.loads(lines[-1])
    assert summary["ok"]
    checks = summary["checks"]
    assert checks["checkpoint remote code is the pinned oracle"]["pass"]
    fused = checks["3 fused decode (gate 1.0) vs materialize"]
    # Compared through at least the first joint eviction, at fp32 interpreter precision.
    assert fused["steps_compared"] >= 8 and fused["attn_rel_err"] <= 1e-4
    assert checks["4 gate at ratio 0.25"]["verdict"] == "gated"
    assert checks["5 compiled eviction"]["eager"] == 0
