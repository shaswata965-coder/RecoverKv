"""L-reuse is fail-loud: a miss raises AT the miss, it does not degrade.

The rest of the prefill score path is already kernel-or-error --
``compute_token_scores`` has no PyTorch fallback for ``T > 1`` because the
reference materialises a ``[B, H_q, chunk, S]`` block that OOMs the shapes this
method targets. The L-capture was the one piece that still degraded quietly,
and it degrades into exactly that reference: a second ``O(N^2)`` pass per layer
plus a transient that is 32 GB at 4096/batch-32. That is how a whole campaign of
prefill numbers got recorded on the recompute path while the banner said
``installed source: 'flashinfer'``.

Equally important is what strictness must NOT do: fire when reuse was never
requested, or on decode, or after an eviction. A strict mode that cries wolf
gets turned off, and then it protects nothing.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("STICKYKV_LSE_STRICT", "STICKYKV_LSE_BACKEND",
              "STICKYKV_SCORE_LSE_FROM_FORWARD"):
        monkeypatch.delenv(k, raising=False)


class TestStrictDefaultsOnAndAgreesEverywhere:
    """Three modules read this knob. If they disagree, one half of the chain
    raises while the other silently degrades, which is worse than either."""

    def _gates(self):
        from modules.windowed_cache import flash_lse, flashinfer_lse, hooks
        return {"hooks": hooks._lse_strict(), "flash": flash_lse._strict(),
                "flashinfer": flashinfer_lse.strict()}

    def test_default_is_strict(self):
        assert all(self._gates().values()), "L-reuse must fail loud by default"

    @pytest.mark.parametrize("val,want", [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("", False),
    ])
    def test_all_three_modules_agree(self, monkeypatch, val, want):
        monkeypatch.setenv("STICKYKV_LSE_STRICT", val)
        g = self._gates()
        assert set(g.values()) == {want}, f"modules disagree at {val!r}: {g}"

    def test_the_driver_script_no_longer_forces_the_degrade(self):
        """run_perf_table.sh exported STICKYKV_LSE_STRICT=0, which made the
        library default irrelevant for every table run."""
        src = open("scripts/run_perf_table.sh", encoding="utf-8").read()
        assert 'LSE_STRICT="${LSE_STRICT:-1}"' in src
        assert 'LSE_STRICT="${LSE_STRICT:-0}"' not in src


class TestCaptureFailuresRaiseAtTheCallSite:
    """A post-hoc per-cell warning cannot say which layer broke or why. The
    raise has to happen where the exception happened."""

    def test_flash_lse_raises_when_the_build_rejects_return_attn_probs(
            self, monkeypatch):
        import types
        from modules.windowed_cache import flash_lse

        monkeypatch.setenv("STICKYKV_LSE_STRICT", "1")

        def _orig(*a, **k):
            if "return_attn_probs" in k:
                raise TypeError("unexpected keyword 'return_attn_probs'")
            return "out"

        mod = types.SimpleNamespace(flash_attn_func=_orig)
        h = flash_lse.enable(mod)
        assert h is not None
        try:
            with pytest.raises(RuntimeError, match="return_attn_probs"):
                mod.flash_attn_func("q", "k", "v")
        finally:
            h.restore()

    def test_the_same_failure_degrades_when_strict_is_off(self, monkeypatch):
        import types
        from modules.windowed_cache import flash_lse

        monkeypatch.setenv("STICKYKV_LSE_STRICT", "0")

        def _orig(*a, **k):
            if "return_attn_probs" in k:
                raise TypeError("nope")
            return "out"

        mod = types.SimpleNamespace(flash_attn_func=_orig)
        h = flash_lse.enable(mod)
        try:
            assert mod.flash_attn_func("q", "k", "v") == "out"
        finally:
            h.restore()

    def test_the_reason_is_recorded_either_way(self, monkeypatch):
        """Strict or not, the cause must be retrievable — the autopsy and the
        perf-runner warning both read it."""
        import types
        from modules.windowed_cache import flash_lse

        monkeypatch.setenv("STICKYKV_LSE_STRICT", "0")

        def _orig(*a, **k):
            if "return_attn_probs" in k:
                raise TypeError("distinctive-marker")
            return "out"

        mod = types.SimpleNamespace(flash_attn_func=_orig)
        h = flash_lse.enable(mod)
        try:
            mod.flash_attn_func("q", "k", "v")
            assert "distinctive-marker" in (flash_lse.broken_reason() or "")
        finally:
            h.restore()


class TestStrictnessMustNotCryWolf:
    """The conditions under which a recompute is legitimate. If strictness
    fires on these, it gets disabled and stops protecting anything."""

    def test_reuse_not_requested_is_not_a_miss(self, monkeypatch):
        """STICKYKV_SCORE_LSE_FROM_FORWARD=0 means the user asked to recompute.
        _install_lse_source returns capture=False, and the strict branch is
        gated behind lse_capture, so it cannot fire."""
        from modules.windowed_cache import hooks
        monkeypatch.setenv("STICKYKV_SCORE_LSE_FROM_FORWARD", "0")
        assert hooks._lse_from_forward() is False

    def test_the_guard_conditions_are_all_present_in_the_hook(self):
        """The strict raise sits under `lse_capture and T > 1 and score_meta is
        None`. Losing any one of those makes it fire on a legitimate path:
        decode (T == 1) computes L per step by design, and a post-eviction
        prefill pass has reordered keys so flash's L does not apply."""
        src = open("modules/windowed_cache/hooks.py", encoding="utf-8").read()
        assert "if lse_capture and q.shape[2] > 1 and score_meta is None:" in src
        i = src.index("if lse_capture and q.shape[2] > 1 and score_meta is None:")
        assert "_lse_strict()" in src[i:i + 1200], (
            "the strict check must live INSIDE the reuse-requested branch")


class TestTheMissMessageIsActionable:
    """Three distinguishable causes with different fixes. A message that does
    not separate them sends the reader to the wrong one."""

    def _hook_src(self):
        src = open("modules/windowed_cache/hooks.py", encoding="utf-8").read()
        i = src.index("elif _lse_strict():")
        # To the end of the raise, not a fixed window — a fixed slice silently
        # stops asserting on whatever falls past its edge.
        j = src.index("compute_token_scores(", i)
        return src[i:j]

    def test_separates_never_reached_from_latched_from_wrong_shape(self):
        s = self._hook_src()
        assert "NEVER REACHED" in s
        assert "latched off earlier" in s
        assert "expected" in s

    def test_names_the_layer(self):
        assert "layer {lidx}" in self._hook_src()

    def test_points_at_the_cheapest_remedy_first(self):
        """STICKYKV_LSE_BACKEND=flash keeps the attention output bit-identical
        because it reuses the same kernel; flashinfer replaces the kernel."""
        s = self._hook_src()
        assert "STICKYKV_LSE_BACKEND=flash" in s
        assert "bit-identical" in s

    def test_says_how_to_turn_it_off(self):
        assert "STICKYKV_LSE_STRICT=0" in self._hook_src()


class TestFlashinferMustNotDropAttentionArguments:
    """flashinfer_lse REPLACES the attention call. Any result-changing argument
    it does not forward silently changes the model's output — and it forwards
    only q/k/v, causal and softmax_scale. Inert on Llama-3 at eval; NOT inert on
    the Mistral branches, whose sliding window arrives as `window_size` and
    whose absence computes full attention.
    """

    def _wrapper(self, monkeypatch, strict="1"):
        import sys
        import types

        import torch as _t
        from modules.windowed_cache import flashinfer_lse

        monkeypatch.setenv("STICKYKV_LSE_STRICT", strict)
        # A flashinfer stub, so the guard is reached without the real library.
        fake = types.ModuleType("flashinfer")
        fake.BatchPrefillWithRaggedKVCacheWrapper = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "flashinfer", fake)

        called = {}

        def _orig(*a, **k):
            called["hit"] = True
            return "real-kernel-output"

        mod = types.SimpleNamespace(flash_attn_func=_orig)
        h = flashinfer_lse.enable(mod)
        assert h is not None, "stub flashinfer should let enable() succeed"
        q = _t.zeros(1, 4, 2, 8)          # [B, S, H, D], S > 1 => prefill
        return mod, q, called, h

    def test_sliding_window_raises_rather_than_being_dropped(self, monkeypatch):
        mod, q, _, h = self._wrapper(monkeypatch)
        try:
            with pytest.raises(RuntimeError, match="window_size"):
                mod.flash_attn_func(q, q, q, 0.0, causal=True,
                                    window_size=(4096, 0))
        finally:
            h.restore()

    def test_nonzero_dropout_raises(self, monkeypatch):
        mod, q, _, h = self._wrapper(monkeypatch)
        try:
            with pytest.raises(RuntimeError, match="dropout_p"):
                mod.flash_attn_func(q, q, q, 0.1, causal=True)
        finally:
            h.restore()

    def test_alibi_and_softcap_raise(self, monkeypatch):
        mod, q, _, h = self._wrapper(monkeypatch)
        try:
            with pytest.raises(RuntimeError, match="alibi_slopes"):
                mod.flash_attn_func(q, q, q, 0.0, alibi_slopes=[1.0])
            with pytest.raises(RuntimeError, match="softcap"):
                mod.flash_attn_func(q, q, q, 0.0, softcap=30.0)
        finally:
            h.restore()

    def test_default_flash_arguments_are_not_flagged(self, monkeypatch):
        """The guard must not fire on the ordinary Llama call, or it blocks the
        very path it is protecting."""
        mod, q, called, h = self._wrapper(monkeypatch)
        try:
            # Reaches the FlashInfer body (stub wrapper -> plan() fails) and
            # raises the CAPTURE error, not the unsupported-argument error.
            with pytest.raises(RuntimeError) as ei:
                mod.flash_attn_func(q, q, q, 0.0, causal=True,
                                    window_size=(-1, -1))
            assert "does not forward" not in str(ei.value)
        finally:
            h.restore()

    def test_non_strict_hands_the_call_back_to_the_real_kernel(self, monkeypatch):
        """Degrading must return the REAL kernel's output, never a partial one."""
        mod, q, called, h = self._wrapper(monkeypatch, strict="0")
        try:
            out = mod.flash_attn_func(q, q, q, 0.0, window_size=(4096, 0))
            assert out == "real-kernel-output"
            assert called.get("hit") is True
        finally:
            h.restore()


class TestTheStrictErrorCarriesTheCauseInline:
    """`raise ... from exc` puts the real cause in a SEPARATE traceback block
    above the new one — the first thing to scroll off a cluster log, and the
    only part that says what FlashInfer actually objected to. The message must
    stand alone.
    """

    def _raise_it(self, monkeypatch):
        import sys
        import types

        import torch as _t
        from modules.windowed_cache import flashinfer_lse

        monkeypatch.setenv("STICKYKV_LSE_STRICT", "1")
        fake = types.ModuleType("flashinfer")

        def _boom(*a, **k):
            raise ValueError("plan() got an unexpected keyword 'head_dim_qk'")

        fake.BatchPrefillWithRaggedKVCacheWrapper = _boom
        monkeypatch.setitem(sys.modules, "flashinfer", fake)

        mod = types.SimpleNamespace(flash_attn_func=lambda *a, **k: "real")
        h = flashinfer_lse.enable(mod)
        q = _t.zeros(2, 16, 4, 8)
        try:
            with pytest.raises(RuntimeError) as ei:
                mod.flash_attn_func(q, q, q, 0.0, causal=True)
            return str(ei.value)
        finally:
            h.restore()

    def test_the_flashinfer_message_is_in_the_error_text(self, monkeypatch):
        msg = self._raise_it(monkeypatch)
        assert "head_dim_qk" in msg, "the actual cause is not in the message"
        assert "ValueError" in msg

    def test_the_raising_frame_is_named(self, monkeypatch):
        assert "Raised at:" in self._raise_it(monkeypatch)

    def test_the_shape_is_reported(self, monkeypatch):
        msg = self._raise_it(monkeypatch)
        assert "B=2" in msg and "S_q=16" in msg

    def test_the_cheapest_remedy_leads(self, monkeypatch):
        msg = self._raise_it(monkeypatch)
        assert msg.index("STICKYKV_LSE_BACKEND=flash") < msg.index(
            "STICKYKV_LSE_STRICT=0")

    def test_the_full_traceback_is_still_retrievable(self, monkeypatch):
        from modules.windowed_cache import flashinfer_lse
        self._raise_it(monkeypatch)
        tb = flashinfer_lse.broken_traceback()
        assert tb and "head_dim_qk" in tb
