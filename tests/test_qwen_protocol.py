"""The Qwen2.5 column is run on the QEvict column's protocol and split.

For three rounds the Qwen2.5 comparison against QEvict (``int2_qwen``) was not
like for like, and nothing said so: this branch's
``configs/longbench_qwen_ours_flash_attn.yaml`` -- the SAME filename QEvict's
branch uses -- ran fp16 at native 32K truncated to 31500 at quant_ratio 0.70,
while QEvict's ran bf16 + YaRN 128K, untruncated, at 0.5. At 0.70 the
full-precision tier holds ~40% fewer tokens, which is where the retrieval-heavy
tasks lost 4-8 points (QWEN_PORT_PLAN.md "Round 3").

These pin the protocol, so a config edit that re-opens the gap fails here with
the reason rather than showing up weeks later as a score.
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
QEVICT_SPLIT = 0.5

QWEN_METHOD_CONFIGS = [
    "longbench_qwen_ours_flash_attn.yaml",
    "longbench_qwen_ours_flash_attn_yarn128k.yaml",
    "gsm8k_qwen_budget20.yaml",
    "ruler_qwen_niah_mk3_omega16.yaml",
]
QWEN_ALL_CONFIGS = QWEN_METHOD_CONFIGS + [
    "longbench_qwen_full_cache.yaml",
    "gsm8k_qwen_full_cache.yaml",
    "longbench_qwen_ours_q070.yaml",
    "longbench_qwen_ours_gate100.yaml",
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
def test_the_qwen_column_is_at_qevicts_split(name):
    cache = _load(name).cache
    assert cache.quant_ratio == QEVICT_SPLIT, (
        f"{name}: quant_ratio {cache.quant_ratio}. The Qwen column is compared "
        "against QEvict at 0.5; the shared 0.70 is the q070 ablation arm.")
    assert OPERATING_POINT["quant_ratio"] != QEVICT_SPLIT, (
        "the shared operating point moved to QEvict's split; the Qwen "
        "exemptions in OPERATING_POINT_EXEMPT are no longer needed")


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
    ("longbench_qwen_ours_q070.yaml", "quant_ratio", OPERATING_POINT["quant_ratio"]),
    ("longbench_qwen_ours_gate100.yaml", "quant_gate_ratio", 1.0),
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


def test_the_ablation_script_runs_the_three_arms_into_separate_dirs():
    src = (ROOT / "scripts/run_longbench_qwen_ablation.sh").read_text()
    for cfg in ("longbench_qwen_ours_flash_attn.yaml",
                "longbench_qwen_ours_q070.yaml",
                "longbench_qwen_ours_gate100.yaml"):
        assert f"configs/{cfg}" in src
    assert 'longbench.output_dir=$out' in src
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts/run_longbench_qwen_ablation.sh")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
