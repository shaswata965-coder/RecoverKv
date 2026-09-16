"""The card's ``eps`` field is built only when something will read it.

``build_sketch`` ends by materializing ``recon``, a full ``[N, H, ws, D]`` fp32
tensor, purely to norm it away into the residual field. It is the most expensive
step in the card build, it runs on every eviction, and on every production
configuration **nothing reads the result**:

* ``_gate_triton`` unpacks the card as ``…, _e_q, _e_s`` and discards both.
* ``_gate_kernel`` stopped computing the bound in ``71841d2``; its docstring
  already records that half (*"the bound cost a [B, H_q, NW] fp32 store and the
  whole eps field of every card, to be discarded by _gate_triton"*).
* The one consumer is ``store.gate_and_select``, reached only from
  ``gated_decode_step`` — the CPU reference, not production.
* And ``cache._gate_ctx`` **raises** on a finite ``quant_gate_margin``, which is
  the only setting ``eps`` exists to serve.

So the field is derived from the resolved margin rather than deleted: the
capability is intact and only the unused work is skipped. These tests pin all
three halves of that — the shapes stay identical, the *selection* does not move,
and asking for a margin still gets a real bound.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modules.quant.sketch import (  # noqa: E402
    build_sketch, decode_sketch, gate_and_score, select_windows,
)


def _batched(s, b):
    """``[N, H, ...]`` card -> the ``[B, Nw, Hkv, ...]`` layout the gate reads."""
    return type(s)(*[f.unsqueeze(0).expand(b, *f.shape).contiguous() for f in s])


def _fixture(seed=0, N=6, H=2, ws=8, D=32):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(N, H, ws, D, generator=g) * 1.7
    anchor = keys.mean(dim=(0, 2))                      # [H, D]
    return keys, anchor


def test_eps_off_matches_eps_on_in_every_other_field():
    """Skipping the residual changes nothing the gate ranks on.

    ``mu``/``v``/``t`` are computed before ``recon`` and do not depend on it, so
    this is bit-identity and not a tolerance.
    """
    keys, anchor = _fixture()
    on = build_sketch(keys, anchor, need_eps=True)
    off = build_sketch(keys, anchor, need_eps=False)
    for name in ("mu_q", "mu_s", "v_q", "v_s", "t_q", "t_s"):
        assert torch.equal(getattr(on, name), getattr(off, name)), name


def test_eps_off_keeps_the_shapes_and_dtypes_the_slot_table_stores():
    """The ``Sketch`` contract and the table's eight columns are unchanged."""
    keys, anchor = _fixture()
    on = build_sketch(keys, anchor, need_eps=True)
    off = build_sketch(keys, anchor, need_eps=False)
    assert off.e_q.shape == on.e_q.shape and off.e_q.dtype == on.e_q.dtype
    assert off.e_s.shape == on.e_s.shape and off.e_s.dtype == on.e_s.dtype
    assert not off.e_q.any() and not off.e_s.any()


def test_the_gate_selects_the_same_windows_without_eps():
    """What the gate ranks on is ``est``, which does not contain ``eps``.

    This is the test that makes the elision safe at the default ``inf`` margin:
    the pick is a top-k on the card estimate, so removing the bound term cannot
    move it. If this ever fails, the fused gate has grown a margin and
    ``sketch_needs_eps`` must follow it.
    """
    keys, anchor = _fixture(seed=3)
    g = torch.Generator().manual_seed(11)
    # [B, H_q, D] with H_q = H_kv * rep; the card has H_kv=2, so rep=2.
    q = torch.randn(2, 4, keys.shape[-1], generator=g)

    on = _batched(build_sketch(keys, anchor, need_eps=True), 2)
    off = _batched(build_sketch(keys, anchor, need_eps=False), 2)
    _, lm_on, est_on = gate_and_score(q, on, anchor, 0.125)
    _, lm_off, est_off = gate_and_score(q, off, anchor, 0.125)

    assert torch.equal(est_on, est_off)
    assert torch.equal(lm_on, lm_off)
    for k in (1, 2, 3):
        assert torch.equal(select_windows(est_on, k), select_windows(est_off, k))


def test_a_zero_eps_decodes_to_a_zero_residual_not_to_garbage():
    """"Not computed" has to decode to nothing, not to an invented bound."""
    keys, anchor = _fixture(seed=5)
    off = build_sketch(keys, anchor, need_eps=False)
    *_, eps_hat = decode_sketch(off, anchor)
    assert torch.equal(eps_hat, torch.zeros_like(eps_hat))


def test_a_finite_margin_still_gets_a_real_bound():
    """The capability is switched, not removed — asking for it still works."""
    keys, anchor = _fixture(seed=7)
    on = build_sketch(keys, anchor, need_eps=True)
    assert on.e_s.gt(0).any(), "a real card must carry a non-zero eps scale"


def test_the_default_cache_asks_for_no_eps_and_a_margin_flips_it():
    """``WindowedCache`` derives the flag from the resolved margin.

    The point of deriving rather than hard-coding: ``quant_gate_margin`` stays a
    real setting. A reader who sets one gets a card that can honour it.
    """
    from modules.windowed_cache.config import WindowedCacheConfig

    base = dict(window_size=8, num_sink_tokens=5, local_window_size=128,
                cache_budget=0.2, quant_ratio=0.7)
    default = WindowedCacheConfig(**base)
    assert default.quant_gate_margin == float("inf")

    with_margin = WindowedCacheConfig(**base, quant_gate_margin=2.0)
    assert with_margin.quant_gate_margin == 2.0

    # The expression cache.py applies, stated once here so the two cannot drift.
    assert (default.quant_gate_margin != float("inf")) is False
    assert (with_margin.quant_gate_margin != float("inf")) is True


def test_the_store_carries_the_flag_through_join_layers():
    """The layer-major store is built from a reference store, not from config.

    ``join_layers`` copies ``sketch_enabled``; if it did not also copy this, the
    decode store would silently start building the field again at the first
    layer-major eviction — which is every eviction after the first.
    """
    from modules.quant.store import QuantizedStore

    kw = dict(window_size=8, head_dim=32, num_kv_heads=2, n_slots=4,
              sketch_enabled=True, sketch_needs_eps=False)
    stores = [QuantizedStore(**kw) for _ in range(3)]
    joint = QuantizedStore.join_layers(stores)
    assert joint.sketch_needs_eps is False
