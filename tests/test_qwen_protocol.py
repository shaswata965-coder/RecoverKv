"""The Qwen2.5 column is run on the QEvict column's protocol and operating point.

The QEvict column (``int2_qwen``'s code) and this branch's Qwen2.5 column were
both run at the shared operating point (quant_ratio 0.70, budget 0.20) on bf16 +
YaRN 128K, untruncated -- the ``_yarn128k`` config here. So the gap between
them is the machinery, and the configs must not drift away from that pairing
without saying so:

* every Qwen config is bf16 + YaRN 128K (fp16 is the overflow trap
  ``utils/model_loading.py`` warns about, and a row on a different RoPE is a
  different model). The default-named config was fp16 at 32K under the same
  name ``int2_qwen`` uses for bf16 + YaRN; it is now the run that was measured,
  and ``_yarn128k`` inherits it;
* the method configs sit at ``OPERATING_POINT`` -- the split QEvict ran too.
  (A diagnosis that assumed QEvict ran at int2_qwen's YAML default of 0.5 moved
  them to 0.5 for one commit; it was wrong, see QWEN_PORT_PLAN.md.)
* the gate's control arm is a one-field diff.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from utils.config import OPERATING_POINT, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

#: QEvict's Qwen protocol, read off int2_qwen's configs.
QEVICT_MODEL = {
    "dtype": "bfloat16",
    "max_position_embeddings": 131072,
    "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                     "original_max_position_embeddings": 32768},
}

QWEN_METHOD_CONFIGS = [
    "longbench_qwen_ours_flash_attn.yaml",
    "longbench_qwen_ours_flash_attn_yarn128k.yaml",
    "gsm8k_qwen_budget20.yaml",
    "ruler_qwen_niah_mk3_omega16.yaml",
]
QWEN_ALL_CONFIGS = QWEN_METHOD_CONFIGS + [
    "longbench_qwen_full_cache.yaml",
    "gsm8k_qwen_full_cache.yaml",
    "longbench_qwen_ours_gate100.yaml",
    "longbench_qwen_ours_gate50.yaml",
    "longbench_qwen_ours_q0.yaml",
]


def _load(name):
    return load_config(CONFIGS / name)


@pytest.mark.parametrize("name", QWEN_ALL_CONFIGS)
def test_every_qwen_config_runs_qevicts_model_setup(name):
    """bf16 + YaRN 128K on every Qwen row, baselines included. fp16 is the trap
    utils/model_loading.py warns about (a bf16-native model overflowing at long
    context), and a row on a different RoPE is a different model."""
    m = _load(name).model
    assert m.dtype == QEVICT_MODEL["dtype"], f"{name}: dtype {m.dtype}"
    assert m.max_position_embeddings == QEVICT_MODEL["max_position_embeddings"]
    assert dict(m.rope_scaling) == QEVICT_MODEL["rope_scaling"]


@pytest.mark.parametrize("name", QWEN_METHOD_CONFIGS)
def test_the_qwen_column_is_at_the_operating_point_qevict_ran_at(name):
    """The QEvict column was run at quant_ratio 0.70 as well, so the Qwen column
    stays on OPERATING_POINT's split; comparing it at any other split confounds
    the machinery with the operating point."""
    cache = _load(name).cache
    assert cache.quant_ratio == OPERATING_POINT["quant_ratio"] == 0.70, (
        f"{name}: quant_ratio {cache.quant_ratio}")


@pytest.mark.parametrize("name", ["longbench_qwen_ours_flash_attn.yaml",
                                  "longbench_qwen_full_cache.yaml"])
def test_longbench_qwen_rows_are_not_truncated(name):
    """QEvict's max_length: null never cut a prompt under YaRN's 131072 window."""
    assert _load(name).longbench.max_length is None


def test_the_rest_of_the_qwen_cache_is_the_operating_point():
    """Only the split departs from OPERATING_POINT; the geometry does not."""
    cache = _load("longbench_qwen_ours_flash_attn.yaml").cache
    for field in ("cache_budget", "window_size", "num_sink_tokens",
                  "local_window_size", "quant_budget_mode", "quant_gate_ratio",
                  "first_eviction_step"):
        assert getattr(cache, field) == OPERATING_POINT[field], field


def test_the_yarn128k_alias_is_the_same_run():
    base, alias = (_load("longbench_qwen_ours_flash_attn.yaml"),
                   _load("longbench_qwen_ours_flash_attn_yarn128k.yaml"))
    assert base.model == alias.model and base.cache == alias.cache
    assert base.longbench.output_dir != alias.longbench.output_dir


@pytest.mark.parametrize("arm,field,value", [
    ("longbench_qwen_ours_gate100.yaml", "quant_gate_ratio", 1.0),
    ("longbench_qwen_ours_gate50.yaml", "quant_gate_ratio", 0.5),
    ("longbench_qwen_ours_q0.yaml", "quant_ratio", 0.0),
])
def test_each_ablation_arm_changes_exactly_one_cache_field(arm, field, value):
    """An arm that moved two things measures neither."""
    raw = yaml.safe_load((CONFIGS / arm).read_text())
    assert raw["_base_"] == "longbench_qwen_ours_flash_attn.yaml"
    assert raw["cache"] == {field: value}
    assert set(raw) == {"_base_", "cache", "longbench"}
    assert set(raw["longbench"]) == {"output_dir"}
    base = _load("longbench_qwen_ours_flash_attn.yaml")
    got = _load(arm)
    assert got.model == base.model
    assert getattr(got.cache, field) == value


def test_the_ablation_script_runs_its_arms_into_separate_dirs():
    src = (ROOT / "scripts/run_longbench_qwen_ablation.sh").read_text()
    for cfg in ("longbench_qwen_ours_flash_attn.yaml",
                "longbench_qwen_ours_gate100.yaml",
                "longbench_qwen_ours_gate50.yaml",
                "longbench_qwen_ours_q0.yaml",
                "longbench_qwen_full_cache.yaml"):
        assert f"configs/{cfg}" in src
    assert 'longbench.output_dir=$out' in src
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts/run_longbench_qwen_ablation.sh")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
