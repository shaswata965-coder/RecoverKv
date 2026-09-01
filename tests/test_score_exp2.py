"""The score kernel's exponential is the score pass, and it changed base.

Why it changed. The kernel issues one exponential per (query, key) pair —
8.6e9 across the model at 4096/batch-1 — against 2.2 TFLOP of ``tl.dot`` and
~11 GB of Q traffic. Budgeted on an A100 that is ~7 ms of tensor core, ~11 ms
of memory, and ~36 ms of exponential if the exponential is the accurate
``expf``. The measured score-pass overhead was 46 ms, so the transcendental was
the kernel. ``ex2.approx.f32`` is one hardware instruction instead of ~ten.

What must hold. The identity is exact in real arithmetic:

    exp(x) == exp2(x · log2 e)
    exp(s·scale − L) == exp2(s·(scale·log2 e) − L·log2 e)

so the only question is float behaviour, and the only thing that can silently
break is the FOLDING — if ``scale`` reaches the kernel without ``log2 e``, every
score is wrong by an exponent factor and nothing raises. That is what most of
this file checks, on CPU, because the kernel itself needs a GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from modules.windowed_cache.score_kernel import LOG2E, _score_exp2_enabled


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("STICKYKV_SCORE_EXP2", raising=False)


class TestTheGate:
    def test_exp2_is_the_default(self):
        assert _score_exp2_enabled() is True

    @pytest.mark.parametrize("val,want", [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("", False),
    ])
    def test_env_override(self, monkeypatch, val, want):
        monkeypatch.setenv("STICKYKV_SCORE_EXP2", val)
        assert _score_exp2_enabled() is want

    def test_log2e_is_exact(self):
        """A hand-typed constant here silently rescales every score."""
        assert LOG2E == math.log2(math.e)


class TestTheIdentityHolds:
    """Real-arithmetic identity, checked in float at realistic magnitudes."""

    @pytest.fixture
    def case(self):
        torch.manual_seed(0)
        B, Hq, T, S, D = 2, 4, 64, 64, 128
        q = torch.randn(B, Hq, T, D) * 0.5
        k = torch.randn(B, Hq, S, D) * 0.5
        scale = D ** -0.5
        s = (q @ k.transpose(-1, -2)) * scale
        lse = torch.logsumexp(s, dim=-1)
        return q, k, scale, s, lse

    def test_per_element_agreement(self, case):
        q, k, scale, s, lse = case
        old = torch.exp(s - lse[..., None])
        new = torch.exp2((q @ k.transpose(-1, -2)) * (scale * LOG2E)
                         - lse[..., None] * LOG2E)
        torch.testing.assert_close(old, new, atol=1e-6, rtol=1e-5)

    def test_the_colsum_is_what_actually_feeds_eviction(self, case):
        """Scores are a column sum over queries, then ranked. Summing averages
        the per-element noise down, so this is the tolerance that matters."""
        q, k, scale, s, lse = case
        old = torch.exp(s - lse[..., None]).sum(-2)
        new = torch.exp2((q @ k.transpose(-1, -2)) * (scale * LOG2E)
                         - lse[..., None] * LOG2E).sum(-2)
        rel = ((old - new).abs() / (old.abs() + 1e-30)).max().item()
        assert rel < 1e-5, f"colsum relative error {rel:.2e}"

    def test_it_is_still_a_softmax(self, case):
        q, k, scale, s, lse = case
        new = torch.exp2((q @ k.transpose(-1, -2)) * (scale * LOG2E)
                         - lse[..., None] * LOG2E)
        torch.testing.assert_close(new.sum(-1), torch.ones_like(new.sum(-1)),
                                   atol=1e-5, rtol=1e-5)

    def test_dropping_the_fold_is_catastrophic_not_subtle(self, case):
        """Guards the failure mode this file exists for: exp2 without log2(e)
        folded into scale does not error, it silently returns different scores.
        If this ever stops failing, the identity tests above are vacuous."""
        q, k, scale, s, lse = case
        old = torch.exp(s - lse[..., None])
        unfolded = torch.exp2(s - lse[..., None])          # missing LOG2E
        rel = ((old - unfolded).abs() / (old.abs() + 1e-30)).max().item()
        assert rel > 0.1, "a missing fold must be loud in the values"


class TestTheFoldReachesTheKernel:
    """The dispatcher multiplies `scaling` by log2(e) only when exp2 is on.
    Checked by source, because the launch itself needs CUDA."""

    def _src(self):
        return open("modules/windowed_cache/score_kernel.py",
                    encoding="utf-8").read()

    def test_scale_is_folded_and_gated_together(self):
        src = self._src()
        assert "eff_scale = scaling * LOG2E if use_exp2 else scaling" in src
        assert "use_exp2 = _score_exp2_enabled()" in src

    def test_the_folded_scale_is_what_is_passed(self):
        """`common` must carry eff_scale, not the raw scaling."""
        src = self._src()
        i = src.index("common = (")
        block = src[i:src.index(")", src.index("num_groups,", i))]
        assert "eff_scale," in block
        assert "\n        scaling," not in block

    def test_both_launch_paths_pass_the_flag(self):
        """Autotuned and pinned launches both take USE_EXP2, or one of them
        silently runs the other branch."""
        src = self._src()
        assert src.count("USE_EXP2=use_exp2") == 2

    def test_the_kernel_applies_log2e_to_the_lse_vector(self):
        """Folded into scale (per call) and the [BLOCK_M] LSE vector — never
        into the [BLOCK_M, BLOCK_N] tile, or the base change costs per element."""
        src = self._src()
        i = src.index("if USE_EXP2:")
        body = src[i:i + 200]
        assert "tl.exp2(s - lse_i[:, None] * 1.4426950408889634)" in body

    def test_the_expf_path_is_still_reachable(self):
        """The escape hatch has to actually compute the historic value."""
        src = self._src()
        i = src.index("if USE_EXP2:")
        assert "p = tl.exp(s - lse_i[:, None])" in src[i:i + 400]


class TestAutotuneConfigs:
    """BLOCK_N divides Q traffic: the key-outer kernel re-reads Q once per key
    block, so 32x redundancy at 64 and 8x at 256. With the exponential no longer
    dominating, that traffic is the next bound."""

    def test_large_block_n_configs_exist(self):
        src = open("modules/windowed_cache/score_kernel.py",
                   encoding="utf-8").read()
        i = src.index("_SCORE_AUTOTUNE_CONFIGS = [")
        block = src[i:src.index("]", i)]
        assert '"BLOCK_N": 256' in block, (
            "without a 256 config the autotuner cannot get below 16x Q re-reads")

    def test_configs_are_unique(self):
        src = open("modules/windowed_cache/score_kernel.py",
                   encoding="utf-8").read()
        i = src.index("_SCORE_AUTOTUNE_CONFIGS = [")
        block = src[i:src.index("\n    ]", i)]
        lines = [l.strip() for l in block.splitlines()
                 if l.strip().startswith("triton.Config")]
        assert len(lines) == len(set(lines)), f"duplicate configs: {lines}"
