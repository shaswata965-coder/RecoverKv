"""CPU unit tests for bounding-volume digests (modules/quant/digest.py).

The contract these pin is one-sided: a digest's estimate must never be BELOW
the true best dot product in the group it summarizes. Under-estimating is the
only failure that loses information — a window whose bound reads low is skipped,
so an under-estimate silently discards a key the model wanted. Over-estimating
only costs unnecessary work.

All tests run on CPU with synthetic data; no model loads, no GPU.
"""

from __future__ import annotations

import pytest
import torch

from modules.quant.digest import (
    build_aabb,
    build_sphere,
    estimate_aabb,
    estimate_sphere,
)


def _true_max(q: Tensor, keys: Tensor) -> torch.Tensor:  # noqa: F821
    """Exact ``max_k q . k`` per (b, h_q, t, g), the thing the bound must cover.

    Deliberately the naive form — expand KV heads, take every dot product — so
    it shares no code with the bound it is checking.
    """
    B, G, H_kv, S, D = keys.shape
    _, H_q, T, _ = q.shape
    rep = H_q // H_kv
    k = keys.permute(0, 2, 1, 3, 4).reshape(B, H_kv, G * S, D)
    k = k.repeat_interleave(rep, dim=1)                      # [B, H_q, G*S, D]
    scores = (q.float() @ k.float().transpose(-2, -1))       # [B, H_q, T, G*S]
    return scores.reshape(B, H_q, T, G, S).amax(dim=-1)


Tensor = torch.Tensor


@pytest.fixture
def shapes():
    # H_q = 8, H_kv = 2 exercises the GQA reshape (rep = 4) rather than MHA.
    return dict(B=2, G=5, H_kv=2, S=8, D=16, H_q=8, T=3)


@pytest.fixture
def keys(shapes):
    torch.manual_seed(0)
    s = shapes
    return torch.randn(s["B"], s["G"], s["H_kv"], s["S"], s["D"])


@pytest.fixture
def q(shapes):
    torch.manual_seed(1)
    s = shapes
    return torch.randn(s["B"], s["H_q"], s["T"], s["D"])


class TestAABB:
    def test_encloses_every_key(self, keys):
        lo, hi = build_aabb(keys)
        assert (lo.unsqueeze(3) <= keys).all()
        assert (keys <= hi.unsqueeze(3)).all()

    def test_shapes(self, keys, shapes):
        lo, hi = build_aabb(keys)
        s = shapes
        assert lo.shape == (s["B"], s["G"], s["H_kv"], s["D"])
        assert hi.shape == lo.shape

    def test_estimate_is_an_upper_bound(self, q, keys):
        lo, hi = build_aabb(keys)
        est = estimate_aabb(q, lo, hi)
        assert (est >= _true_max(q, keys) - 1e-4).all()

    def test_estimate_shape(self, q, keys, shapes):
        lo, hi = build_aabb(keys)
        s = shapes
        assert estimate_aabb(q, lo, hi).shape == (s["B"], s["H_q"], s["T"], s["G"])

    def test_bound_is_tight_for_a_single_key_group(self, q):
        # S = 1 collapses the box to a point, so the bound must be the exact
        # dot product — this is what proves the relu decomposition is right
        # and not merely conservative.
        torch.manual_seed(2)
        keys = torch.randn(1, 3, 2, 1, 16)
        lo, hi = build_aabb(keys)
        qq = torch.randn(1, 4, 2, 16)
        assert torch.allclose(
            estimate_aabb(qq, lo, hi), _true_max(qq, keys), atol=1e-4
        )


class TestSphere:
    def test_encloses_every_key(self, keys):
        center, radius = build_sphere(keys)
        dist = (keys - center.unsqueeze(3)).norm(dim=-1)
        assert (dist <= radius.unsqueeze(3) + 1e-5).all()

    def test_shapes(self, keys, shapes):
        center, radius = build_sphere(keys)
        s = shapes
        assert center.shape == (s["B"], s["G"], s["H_kv"], s["D"])
        assert radius.shape == (s["B"], s["G"], s["H_kv"])

    def test_estimate_is_an_upper_bound(self, q, keys):
        center, radius = build_sphere(keys)
        est = estimate_sphere(q, center, radius)
        assert (est >= _true_max(q, keys) - 1e-4).all()

    def test_estimate_shape(self, q, keys, shapes):
        center, radius = build_sphere(keys)
        s = shapes
        assert estimate_sphere(q, center, radius).shape == (
            s["B"], s["H_q"], s["T"], s["G"]
        )


class TestRelativeTightness:
    """Why AABB is the choice, given it costs 2D against the sphere's D+1.

    The answer is data-dependent, and these two tests pin both sides of it so
    the choice is not folklore. On an isotropic cloud the sphere is actually
    the *tighter* of the two at half the memory — a box drawn round a ball in
    high dimensions is mostly corner. AABB wins only when the channels differ
    in scale, because it bounds each one separately.

    Real key channels do differ in scale, sharply: it is exactly why this
    codebase quantizes keys on a **per-channel** grid (``key_scale`` is
    ``[B, N, H_kv, D]`` in ``modules/quant/slots.py``, one scale per channel)
    while values get a per-token one. The store's own layout is the evidence
    for the regime, so AABB is the choice that matches the data.
    """

    @staticmethod
    def _slack(q, keys):
        lo, hi = build_aabb(keys)
        center, radius = build_sphere(keys)
        truth = _true_max(q, keys)
        return (
            (estimate_aabb(q, lo, hi) - truth).mean(),
            (estimate_sphere(q, center, radius) - truth).mean(),
        )

    def test_aabb_wins_when_channels_differ_in_scale(self):
        torch.manual_seed(4)
        scale = torch.ones(64)
        scale[:4] = 12.0                       # outlier channels, as keys have
        keys = torch.randn(2, 8, 2, 8, 64) * scale
        q = torch.randn(2, 8, 4, 64)
        slack_aabb, slack_sphere = self._slack(q, keys)
        assert slack_aabb < slack_sphere

    def test_sphere_wins_on_an_isotropic_cloud(self):
        """The counter-case, pinned so the AABB choice stays honest.

        If a future digest is ever built over data without per-channel
        structure (a projected or whitened space, say), this is the reminder
        that the sphere is then both tighter and half the size.
        """
        torch.manual_seed(5)
        keys = torch.randn(2, 8, 2, 8, 64)
        q = torch.randn(2, 8, 4, 64)
        slack_aabb, slack_sphere = self._slack(q, keys)
        assert slack_sphere < slack_aabb


class TestGQAMapping:
    def test_query_head_maps_to_its_kv_group(self, shapes):
        """Query head ``h`` must bound against KV head ``h // rep``.

        Built so the two KV heads are far apart in scale: if the mapping were
        transposed (or strided the other way), the bound for the first ``rep``
        query heads would come from the wrong KV head and the enclosure check
        against that head's own keys would fail.
        """
        torch.manual_seed(3)
        keys = torch.randn(1, 2, 2, 4, 8)
        keys[:, :, 1] *= 50.0
        lo, hi = build_aabb(keys)
        q = torch.randn(1, 4, 1, 8)           # H_q = 4, H_kv = 2 -> rep = 2
        est = estimate_aabb(q, lo, hi)
        assert (est >= _true_max(q, keys) - 1e-4).all()


class TestValidation:
    def test_build_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\[B, G, H_kv, S, D\]"):
            build_aabb(torch.randn(2, 3, 4, 5))

    def test_estimate_rejects_indivisible_head_count(self, keys):
        lo, hi = build_aabb(keys)          # H_kv = 2
        with pytest.raises(ValueError, match="not divisible"):
            estimate_aabb(torch.randn(2, 3, 1, 16), lo, hi)
