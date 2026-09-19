"""The compiled eviction's CONTROL ARM — and the guarantees that keep it one.

This repo's standing rule is that every perf-affecting change gets a control
arm and states its claim against it. The compiled eviction never had one, and
the cost of that gap is the best argument for these tests: between 2026-09-18
and 2026-09-19 the eviction did not compile at all on this build while
``_EVICT_STATS`` reported ``compiled`` throughout (DISTANCE_TO_GOAL §10.9,
§10.13), and nothing caught it, because there was nothing to compare against.

The danger of adding one is the opposite failure: an escape hatch that quietly
becomes a fallback, so that "compiled" numbers are sometimes eager numbers.
Every test below pins one of the four properties that stop that:

1. it is OFF unless explicitly asked for;
2. it does not disarm the ``eager`` tripwire (it has its own counter);
3. it is visible in provenance (``evict_path_mode``);
4. a compile FAILURE still raises — the arm is never reached for as a rescue.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modules.windowed_cache import cache as cache_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Each test starts with the arm unresolved and the environment clean."""
    monkeypatch.delenv(cache_mod._EVICT_CONTROL_ARM_ENV, raising=False)
    monkeypatch.setattr(cache_mod, "_EVICT_CONTROL_ARM",
                        {"on": None, "announced": False})
    monkeypatch.setattr(cache_mod, "_EVICT_STATS",
                        {"eager": 0, "compiled": 0, "control_arm": 0})
    monkeypatch.setattr(cache_mod, "_EVICT_ANNOUNCED", {"done": True})
    monkeypatch.setattr(cache_mod, "_EVICT_COMPILE_FAILED", None)
    monkeypatch.setattr(cache_mod, "_COMPILED_EVICT_FN", None)


# --- 1. off by default ------------------------------------------------------

def test_the_arm_is_off_unless_asked_for():
    assert cache_mod.evict_control_arm_enabled() is False
    assert cache_mod.evict_path_mode() == "compiled"


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " on "])
def test_the_env_var_turns_it_on(monkeypatch, raw):
    monkeypatch.setenv(cache_mod._EVICT_CONTROL_ARM_ENV, raw)
    assert cache_mod.evict_control_arm_enabled() is True
    assert cache_mod.evict_path_mode() == "control-arm-eager"


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "maybe"])
def test_anything_else_leaves_it_off(monkeypatch, raw):
    """Not truthy-by-default. A typo must not silently switch the arm on."""
    monkeypatch.setenv(cache_mod._EVICT_CONTROL_ARM_ENV, raw)
    assert cache_mod.evict_control_arm_enabled() is False


def test_the_setter_overrides_the_environment(monkeypatch):
    monkeypatch.setenv(cache_mod._EVICT_CONTROL_ARM_ENV, "1")
    cache_mod.set_evict_control_arm(False)
    assert cache_mod.evict_control_arm_enabled() is False
    cache_mod.set_evict_control_arm(True)
    assert cache_mod.evict_control_arm_enabled() is True


def test_the_env_var_is_read_once_and_then_cached(monkeypatch):
    assert cache_mod.evict_control_arm_enabled() is False
    monkeypatch.setenv(cache_mod._EVICT_CONTROL_ARM_ENV, "1")
    assert cache_mod.evict_control_arm_enabled() is False, (
        "the environment is resolved once per process; a mid-run change to it "
        "must not move the arm under a measurement already in flight")


# --- 2. the tripwire keeps its meaning --------------------------------------

def test_the_arm_has_its_own_counter_and_does_not_touch_eager(monkeypatch):
    """``eager`` must stay a tripwire: nonzero = an ACCIDENTAL eager run.

    Counting the control arm as ``eager`` would make the tripwire fire on the
    one run that is supposed to be eager, and the alarm would be ignored from
    then on — which is how a tripwire dies.
    """
    sentinel = object()
    monkeypatch.setattr(cache_mod.WindowedCache, "_evict_two_tier_impl",
                        staticmethod(lambda *a, **k: sentinel))
    cache_mod.set_evict_control_arm(True)

    out = cache_mod._run_compiled_evict(None, None, None, None, step=7)

    assert out is sentinel
    assert cache_mod._EVICT_STATS["control_arm"] == 1
    assert cache_mod._EVICT_STATS["eager"] == 0, (
        "the control arm incremented the accidental-eager tripwire")
    assert cache_mod._EVICT_STATS["compiled"] == 0


def test_reset_zeroes_the_arm_counter_too():
    cache_mod._EVICT_STATS["control_arm"] = 5
    cache_mod.reset_evict_path_stats()
    assert cache_mod.evict_path_stats()["control_arm"] == 0


# --- 3. it never builds the compiled callable -------------------------------

def test_the_arm_pays_no_compile_and_no_autotune(monkeypatch):
    """The quantity being measured is the compile's cost, so it must not run.

    ``_build_compiled_evict`` is where torch.compile is invoked. If the arm
    touched it, the reference run would carry the very cost it exists to price.
    """
    def _boom(*a, **k):                       # pragma: no cover - must not run
        raise AssertionError("the control arm built the compiled callable")

    monkeypatch.setattr(cache_mod, "_build_compiled_evict", _boom)
    monkeypatch.setattr(cache_mod.WindowedCache, "_evict_two_tier_impl",
                        staticmethod(lambda *a, **k: "ok"))
    cache_mod.set_evict_control_arm(True)

    assert cache_mod._run_compiled_evict(None, None, None, None, step=1) == "ok"
    assert cache_mod._COMPILED_EVICT_FN is None


# --- 4. it is NOT a fallback ------------------------------------------------

def test_a_compile_failure_still_raises_when_the_arm_is_off(monkeypatch):
    """Kernel-or-error is unchanged for every run that did not ask for the arm."""
    monkeypatch.setattr(cache_mod, "_EVICT_COMPILE_FAILED", "Boom: nope")
    with pytest.raises(RuntimeError, match="already failed on this build"):
        cache_mod._run_compiled_evict(None, None, None, None, step=3)


def test_nothing_switches_the_arm_on_by_itself(monkeypatch):
    """A failure must not reach for the arm. Only an explicit request does."""
    monkeypatch.setattr(cache_mod, "_EVICT_COMPILE_FAILED", "Boom: nope")
    with pytest.raises(RuntimeError):
        cache_mod._run_compiled_evict(None, None, None, None, step=3)
    assert cache_mod.evict_control_arm_enabled() is False
    assert cache_mod.evict_path_mode() == "compiled"
    assert cache_mod._EVICT_STATS["control_arm"] == 0


def test_the_arm_is_checked_before_the_sticky_failure_gate(monkeypatch):
    """Deliberate: a build that cannot lower the body is exactly the build on
    which someone needs the reference number."""
    monkeypatch.setattr(cache_mod, "_EVICT_COMPILE_FAILED", "Boom: nope")
    monkeypatch.setattr(cache_mod.WindowedCache, "_evict_two_tier_impl",
                        staticmethod(lambda *a, **k: "ok"))
    cache_mod.set_evict_control_arm(True)
    assert cache_mod._run_compiled_evict(None, None, None, None, step=3) == "ok"


# --- the tripwire condition itself ------------------------------------------

def test_an_unrequested_eager_run_still_stops_the_process():
    """The exemption is the arm, and ONLY the arm.

    ``_evict_two_tier_impl`` opens with
    ``if not is_compiling() and not evict_control_arm_enabled(): raise``. With
    the arm off, calling it outside a trace must still raise EvictionRanEager —
    that check is the only thing standing between "compiled" in the log and
    eager numbers in the table.
    """
    assert cache_mod.evict_control_arm_enabled() is False
    with pytest.raises(cache_mod.EvictionRanEager):
        cache_mod.WindowedCache._evict_two_tier_impl(None, None, None, None)
