"""The read gate had no control arm. This is it, and these are its guardrails.

``e7bc158`` deliberately made every ``quant_gate_ratio`` take the same route, so
``1.0`` still runs the ``fused_gate`` launch and its ``topk`` + ``sort``, the
``SEL`` indirection in the Q loop, the ``GATED`` prologue over every Q column,
the §5 fill pass, and ``build_sketch`` on every eviction. ``1.0`` is a control
for **selectivity**. It is not a control for **the machinery**
(``TARGET_GAP.md`` §4), and the machinery is what every remaining speed item
accuses.

``quant_read_gate=False`` is the control for the machinery: ``GATED`` is a
Triton ``constexpr``, so ``sel=None`` compiles the indirection, the prologue and
the fill pass out of the kernel entirely. It also restores the whole-tier read
memo, which a live gate disables unconditionally.

It is an ARM, not a default, and the tests below are what keep it one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]

from modules.windowed_cache.config import WindowedCacheConfig  # noqa: E402

_MODEL = SimpleNamespace(num_key_value_heads=8, num_attention_heads=32,
                         hidden_size=4096, num_hidden_layers=32, head_dim=128)
_POINT = dict(window_size=8, num_sink_tokens=5, local_window_size=128,
              cache_budget=0.20, quant_ratio=0.70, quant_budget_mode="bytes",
              quant_gate_ratio=0.25)


def _resolve(**over):
    cfg = WindowedCacheConfig(**{**_POINT, **over})
    return cfg.resolve(prefill_len=4096, model_config=_MODEL,
                       kv_dtype=torch.float16, max_tokens=256)


# ---------------------------------------------------------------------------
# The knob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asked,expect", [(None, True), (True, True), (False, False)])
def test_the_arm_resolves_three_ways(asked, expect):
    assert _resolve(quant_read_gate=asked).quant_sketch_enabled is expect


def test_the_default_is_unchanged():
    """The whole point of a tri-state: the shipped point must not move.

    ``None`` has to resolve to exactly what the old derivation produced, or
    every existing number was taken on a different cache than the next one.
    """
    assert _resolve().quant_sketch_enabled is True
    assert _resolve(quant_read_gate=None).quant_sketch_enabled is True


def test_no_q_tier_means_no_gate_whatever_is_asked():
    """``q > 0`` stays a necessary condition. There is nothing to gate at q=0,
    and a config that claimed otherwise would build cards for an empty tier."""
    assert _resolve(quant_ratio=0.0, quant_read_gate=True).quant_sketch_enabled is False
    assert _resolve(quant_ratio=0.0, quant_read_gate=None).quant_sketch_enabled is False


def test_a_non_bool_is_rejected():
    with pytest.raises(ValueError, match="quant_read_gate must be None"):
        WindowedCacheConfig(**_POINT, quant_read_gate="off")


# ---------------------------------------------------------------------------
# What turning it off actually buys: the read memo comes back
# ---------------------------------------------------------------------------


def test_the_memo_is_rejected_with_a_live_gate_and_allowed_without():
    """The memo caches the WHOLE dequantized tier keyed on ``store.version``,
    which only moves at eviction -- while a live gate's selected set moves every
    step, so it would serve a set the gate did not choose. Ungated there is no
    per-step selected set, so the memo is legitimate again."""
    with pytest.raises(ValueError, match="LIVE read gate"):
        WindowedCacheConfig(**_POINT, quant_memoize_read=True)
    with pytest.raises(ValueError, match="LIVE read gate"):
        WindowedCacheConfig(**_POINT, quant_read_gate=True, quant_memoize_read=True)
    WindowedCacheConfig(**_POINT, quant_read_gate=False, quant_memoize_read=True)


def test_the_ungated_arm_restores_the_auto_memo_at_batch_one():
    """``_resolve_memoization`` kills the memo outright when the gate is live.

    Ungated there is no per-step selected set, so the auto rule (``on at B=1,
    off above``) applies again and the whole-tier dequant runs once per
    eviction instead of every step.

    **Scope, stated here so it is not over-read:** the memo covers
    ``effective_q_tier``, reached only via ``_joint_qtier_read`` ->
    ``_materialize_joint``. On CUDA the fused two-tier kernel takes raw int2 and
    dequantizes inside itself, so that path is never taken and this buys
    **nothing on a GPU decode step**. It is an eager/materialize-path saving.
    What the arm buys on CUDA is the gate launch, its ``topk`` + ``sort``, the
    ``SEL`` indirection, the ``GATED`` prologue and the §5 fill pass -- none of
    which this test can reach without a GPU.
    """
    from modules.windowed_cache import cache as cache_mod

    class _Store:
        memoize_read = None

    def _memo_for(sketch_enabled: bool, batch_size: int, pref=None):
        holder = SimpleNamespace(
            resolved=SimpleNamespace(quant_memoize_read=pref,
                                     quant_sketch_enabled=sketch_enabled),
            _memoization_resolved=False,
            _stores=[_Store()])
        cache_mod.WindowedCache._resolve_memoization(holder, batch_size)
        return holder._stores[0].memoize_read

    assert _memo_for(sketch_enabled=True, batch_size=1) is False, (
        "a live gate is supposed to disable the memo")
    assert _memo_for(sketch_enabled=False, batch_size=1) is True, (
        "the ungated arm must get the auto memo back at B=1 on the "
        "eager/materialize path")
    # Above B=1 the memo is off either way: it costs ~149 MB/row against
    # ~131 MB/row of real KV, so it would halve max-B.
    assert _memo_for(sketch_enabled=False, batch_size=32) is False


# ---------------------------------------------------------------------------
# It must reach every runner, or it is silently inert in some of them
# ---------------------------------------------------------------------------


def test_the_factory_omits_none_and_passes_a_bool():
    """``None`` means "derive", which every existing caller wants, so it is
    omitted rather than passed -- a backend without the field keeps working."""
    from utils.cache_factory import quant_read_gate_kwargs

    assert quant_read_gate_kwargs(WindowedCacheConfig, None) == {}
    assert quant_read_gate_kwargs(WindowedCacheConfig, False) == {
        "quant_read_gate": False}
    assert quant_read_gate_kwargs(WindowedCacheConfig, True) == {
        "quant_read_gate": True}

    class _NoField:
        __dataclass_fields__: dict = {}

    assert quant_read_gate_kwargs(_NoField, False) == {}


@pytest.mark.parametrize("runner", [
    "perf_runner.py", "longbench_runner.py", "gsm8k_runner.py",
    "ours_parity_runner.py",
])
def test_every_runner_threads_the_arm(runner):
    """A knob a runner forgets to pass is a knob that is silently inert.

    That is not hypothetical: ``quant_budget_mode`` did nothing in three
    runners at once (ACCURACY_RECOVERY_PLAN.md §2), and ``quant_gate_ratio``
    did nothing in LongBench. The speed table and the accuracy runs have to be
    able to take the SAME arm, or they describe different methods again.
    """
    src = (ROOT / "modules" / "evaluation" / runner).read_text(encoding="utf-8")
    assert "quant_read_gate_kwargs" in src, (
        f"{runner} does not thread quant_read_gate; a config asking for the "
        "ungated arm would be accepted and ignored here")


def test_the_yaml_config_carries_the_field():
    """``6b8a188``'s standing lesson: the YAML loader is a SEPARATE class, and a
    field added only to ``WindowedCacheConfig`` is unreachable from a config
    file."""
    from utils.config import CacheConfig

    assert "quant_read_gate" in CacheConfig.__dataclass_fields__
    assert CacheConfig().quant_read_gate is None


# ---------------------------------------------------------------------------
# It is an ARM: the checker must say so
# ---------------------------------------------------------------------------


def test_the_operating_point_pins_the_gate_as_derived():
    from utils.config import OPERATING_POINT

    assert "quant_read_gate" in OPERATING_POINT
    assert OPERATING_POINT["quant_read_gate"] is None


def test_the_checker_flags_the_ungated_arm_as_a_divergence():
    """Running ungated is legitimate. Running ungated *without saying so* is the
    defect ``8ef579a`` fixed for ``quant_ratio``: a speed row and an accuracy row
    describing two different caches. So the checker must fail on it."""
    src = (ROOT / "scripts" / "run_perf_table.sh").read_text(encoding="utf-8")
    assert 'READ_GATE="${READ_GATE:-}"' in src, (
        "the perf table must default to the derived gate")

    from scripts.check_operating_point import _compare, _norm

    assert _norm("") is None and _norm("off") is False and _norm("on") is True
    assert _compare("x", {"quant_read_gate": ""}) == []
    bad = _compare("x", {"quant_read_gate": "off"})
    assert bad and "quant_read_gate" in bad[0], bad


def test_the_shipped_checker_still_exits_clean():
    r = subprocess.run([sys.executable, "scripts/check_operating_point.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_ungated_longbench_config_differs_in_exactly_one_field():
    """The arm's accuracy half must be a one-field diff, or the pair is not a
    controlled comparison and neither number means anything about the other."""
    import yaml

    gated = yaml.safe_load(
        (ROOT / "configs" / "longbench_ours_flash_attn.yaml").read_text())
    ungated = yaml.safe_load(
        (ROOT / "configs" / "longbench_ours_ungated.yaml").read_text())

    a, b = gated["cache"], ungated["cache"]
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"quant_read_gate"}, (
        f"the ungated arm differs from its gated twin in {sorted(diff)}; only "
        "quant_read_gate may differ or the two runs are not comparable")
    assert b["quant_read_gate"] is False
    assert gated["window"] == ungated["window"]
    assert gated["model"] == ungated["model"]
    assert (gated["longbench"]["datasets"] == ungated["longbench"]["datasets"]), (
        "the two arms must score the SAME datasets")


def test_the_ungated_arm_is_recorded_as_an_exempt_arm_with_a_reason():
    from utils.config import OPERATING_POINT_EXEMPT

    reason = OPERATING_POINT_EXEMPT.get("longbench_ours_ungated.yaml")
    assert reason and "UNGATED" in reason, (
        "an off-point config without a recorded reason is an accident, not an "
        "ablation -- which is the distinction OPERATING_POINT_EXEMPT exists for")
