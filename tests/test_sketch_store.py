"""The gate as the store exposes it: cards through the slot table, end to end.

``tests/test_sketch.py`` pins the estimator's maths on raw tensors. This pins the
plumbing — that a card survives the write/gather round trip, that it rides
eviction and reactivation like the codes it sits beside, and that the bound still
holds after it has been through the table rather than only in a local variable.

CPU-only; no GPU, no Triton, no model.
"""

from __future__ import annotations

import math

import pytest
import torch

from modules.quant.slots import SKETCH_FIELDS, QuantSlotTable
from modules.quant.sketch import Sketch, gate_and_score
from modules.quant.store import QuantizedStore

B, N, H, S, D = 2, 6, 4, 8, 128
SCALING = 1.0 / math.sqrt(D)


def _store(sketch=True, n_slots=32):
    st = QuantizedStore(window_size=S, head_dim=D, num_kv_heads=H,
                        n_slots=n_slots, memoize_read=False,
                        sketch_enabled=sketch)
    st.ensure(B, torch.device("cpu"))
    return st


def _demote(st, n=N, seed=0, hot=6.0):
    g = torch.Generator().manual_seed(seed)
    k_post = torch.randn(B, n, H, S, D, generator=g)
    k_post[:, :, :, 0, :] += torch.randn(B, n, H, D, generator=g) * hot
    st.demote_many(
        st.table.free_slots(n),
        torch.ones(B, n, dtype=torch.bool),
        torch.arange(n).reshape(1, n).expand(B, n).contiguous(),
        torch.randn(B, n, H, S, D, generator=g),                 # pre-RoPE codes
        torch.randn(B, n, H, S, D, generator=g),                 # values
        torch.arange(n * S).reshape(1, n, S).expand(B, n, S).contiguous(),
        keys_post_rope=k_post,
    )
    st.commit_active_count(n)
    return k_post


def _true_max_logit(k_post, order):
    """``[B, Hq, n]`` exact per-window max logit, windows in gate column order."""
    kk = k_post[torch.arange(B)[:, None], order]                 # [B,n,H,S,D]
    return kk


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def test_disabled_store_allocates_no_cards():
    """q>0 with the gate off must cost nothing it did not cost before."""
    st = _store(sketch=False)
    assert not st.table.sketch
    for f in SKETCH_FIELDS:
        assert not hasattr(st.table, f)


def test_demote_without_post_rope_keys_raises_when_enabled():
    st = _store()
    with pytest.raises(ValueError, match="keys_post_rope"):
        st.demote_many(
            st.table.free_slots(N), torch.ones(B, N, dtype=torch.bool),
            torch.arange(N).reshape(1, N).expand(B, N).contiguous(),
            torch.randn(B, N, H, S, D), torch.randn(B, N, H, S, D),
            torch.zeros(B, N, S, dtype=torch.long),
        )


def test_gate_before_any_demotion_is_none():
    assert _store().gate_and_select(torch.randn(B, H, D), SCALING, 1.0) is None


def test_anchor_is_frozen_at_the_first_demotion_batch():
    """A later anchor would silently reinterpret every card already written."""
    st = _store()
    _demote(st, n=3, seed=1)
    first = st._anchor.clone()
    st._n_active = 0
    _demote(st, n=3, seed=99, hot=40.0)          # wildly different key stats
    assert torch.equal(st._anchor, first)


def test_card_survives_the_table_round_trip():
    st = _store()
    _demote(st)
    slots = st.table.active_order(st._n_active)
    card = Sketch(*st.table.gather_sketch(slots))
    # mu and vbar are int4, packed two per byte along D; v and t stay int8.
    assert card.mu_q.shape == (B, N, H, D // 2)
    assert card.vm_q.shape == (B, N, H, D // 2)
    assert card.v_q.shape == (B, N, H, D)
    assert card.t_q.shape == (B, N, H, S)
    assert card.mu_q.dtype == torch.uint8 and card.vm_q.dtype == torch.uint8
    assert card.v_q.dtype == torch.int8 and card.t_q.dtype == torch.int8


def test_estimate_ranks_the_true_hottest_window_through_the_store():
    """The card still finds the right window after write + gather.

    The Cauchy-Schwarz *bound* this used to assert is gone with ``eps``: the
    fused gate is a top-k on the point estimate and never implemented a margin,
    so the bound was built, stored and discarded. What has to survive the round
    trip is the ranking, which is what the cap consumes.
    """
    st = _store()
    k_post = _demote(st)
    q = torch.randn(B, H * 2, D, generator=torch.Generator().manual_seed(7)) * 1.5
    slots = st.table.active_order(st._n_active)
    card = Sketch(*st.table.gather_sketch(slots))
    _, est = gate_and_score(q, card, st._anchor, SCALING)

    order = st.table.slot_wid.gather(1, slots)
    kk = _true_max_logit(k_post, order)
    qg = q.reshape(B, H, 2, D)
    true = SCALING * torch.einsum("bnhwd,bhrd->bhrnw", kk, qg).reshape(
        B, H * 2, N, S).amax(-1)
    assert est.argmax(-1).eq(true.argmax(-1)).float().mean() > 0.9


# ---------------------------------------------------------------------------
# selection behaviour through the store
# ---------------------------------------------------------------------------


def test_ratio_one_selects_every_window():
    """The no-op arm, through the store: 1.0 is "read everything" on the one
    implementation, which is what makes it a provable no-op rather than a second
    code path that has to be kept in step."""
    st = _store()
    _demote(st)
    keep, logmass, slots = st.gate_and_select(
        torch.randn(B, H * 2, D), SCALING, 1.0)
    assert keep.shape == (B, H, N) and keep.all()
    assert logmass.shape == (B, H * 2, N) and slots.shape == (B, N)


def test_ratio_is_monotone_through_the_store():
    st = _store()
    _demote(st)
    q = torch.randn(B, H * 2, D, generator=torch.Generator().manual_seed(8))
    counts = [st.gate_and_select(q, SCALING, r)[0].sum().item()
              for r in (0.25, 0.5, 0.75, 1.0)]
    assert counts == sorted(counts)


def test_ratio_bounds_the_work_through_the_store():
    """The cap binds on the KV head, which is the unit of work: one program
    loads a window once for its whole GQA group."""
    st = _store()
    _demote(st)
    ratio = 2.0 / N
    keep, _, _ = st.gate_and_select(torch.randn(B, H * 2, D), SCALING, ratio)
    import math as _m
    assert keep.sum(-1).max().item() <= max(1, _m.ceil(ratio * N))


def test_logmass_is_returned_for_skipped_windows_too():
    """Without this the skipped tier scores zero, ranks last, and is evicted."""
    st = _store()
    _demote(st)
    keep, logmass, _ = st.gate_and_select(
        torch.randn(B, H * 2, D, generator=torch.Generator().manual_seed(9)),
        SCALING, 0.25)
    assert not keep.all(), "ratio too loose for this test to mean anything"
    assert torch.isfinite(logmass).all()


def test_gate_without_cards_raises_rather_than_degrading():
    """Kernel-or-error: no silent fallback to reading the whole tier."""
    st = _store(sketch=False)
    st._n_active = 1
    with pytest.raises(RuntimeError, match="sketch"):
        st.gate_and_select(torch.randn(B, H, D), SCALING, 1.0)


# ---------------------------------------------------------------------------
# lifecycle: cards must move with the codes they belong to
# ---------------------------------------------------------------------------


def test_retain_only_frees_a_card_with_its_window():
    st = _store()
    _demote(st)
    keep_ids = torch.tensor([[0, 2, 4], [0, 2, 4]])
    _, _, _, keep = st.table.lookup(keep_ids)
    st.table.retain_only(keep)
    assert int(st.table.n_live.min()) == 3


def test_reactivation_does_not_recompute_the_card():
    """Re-demotion is a bit flip (§10) -- the card must be bit-identical after."""
    st = _store()
    _demote(st)
    slots = st.table.active_order(st._n_active)
    before = [f.clone() for f in st.table.gather_sketch(slots)]
    st.table.set_active(slots, torch.ones_like(slots, dtype=torch.bool), False)
    st.table.set_active(slots, torch.ones_like(slots, dtype=torch.bool), True)
    after = st.table.gather_sketch(slots)
    for b, a in zip(before, after):
        assert torch.equal(b, a)


def test_join_layers_carries_the_anchor_row_major():
    stores = []
    for i in range(3):
        st = _store()
        _demote(st, seed=i)
        stores.append(st)
    joint = QuantizedStore.join_layers(stores)
    assert joint.sketch_enabled and joint.table.sketch
    assert joint._anchor.shape == (3 * B, H, D)
    assert torch.equal(joint._anchor[B:2 * B], stores[1]._anchor)
    assert joint.table.sk_mu_q.shape[0] == 3 * B


# ---------------------------------------------------------------------------
# the shipped operating point: gate on by default, 25% of windows read
# ---------------------------------------------------------------------------


def test_ratio_caps_windows_as_a_fraction_of_the_live_tier():
    """The cap is derived per step, not pinned: N_q moves with the shape."""
    st = _store(n_slots=64)
    _demote(st, n=16)
    q = torch.randn(B, H * 2, D, generator=torch.Generator().manual_seed(11))
    keep, _, _ = st.gate_and_select(q, SCALING, ratio=0.25)
    assert keep.sum(-1).max().item() <= 4           # ceil(0.25 * 16)
    assert keep.sum(-1).min().item() >= 1


def test_ratio_never_starves_a_tiny_tier():
    """ceil, and a floor of one: a 3-window tier at 25% still reads one."""
    st = _store()
    _demote(st, n=3)
    keep, _, _ = st.gate_and_select(torch.randn(B, H * 2, D), SCALING, ratio=0.25)
    assert keep.sum(-1).min().item() == 1


def test_ratio_of_one_is_a_no_op():
    st = _store()
    _demote(st)
    keep, _, _ = st.gate_and_select(torch.randn(B, H * 2, D), SCALING, ratio=1.0)
    assert keep.all()


def test_ratio_keeps_the_window_holding_the_true_best_key():
    """What a capped gate must not do: drop the window a query head actually wants.

    Checked for EVERY query head of every group: the window holding that head's
    own hottest key survives. It used to be checked once per group, on the
    group's hottest key by RAW logit -- which is the loudest head's key, the one
    the old raw-max union already served, so the check could not see a quieter
    head losing its window (``tests/test_gate_union_is_per_head.py``).
    """
    st = _store()
    k_post = _demote(st)
    q = torch.randn(B, H * 2, D, generator=torch.Generator().manual_seed(12)) * 1.5
    for ratio in (0.5, 0.75):
        keep, _, slots = st.gate_and_select(q, SCALING, ratio=ratio)
        order = st.table.slot_wid.gather(1, slots)
        kk = k_post[torch.arange(B)[:, None], order]
        qg = q.reshape(B, H, 2, D)
        true = torch.einsum("bnhwd,bhrd->bhrnw", kk, qg).amax(-1)      # [B,H,r,n]
        best = true.argmax(-1)                                         # [B,H,r]
        assert keep.gather(-1, best).all(), f"dropped a head's best at {ratio}"


def test_default_config_turns_the_gate_on_where_there_is_a_tier():
    from modules.windowed_cache.config import WindowedCacheConfig

    class _Model:
        num_key_value_heads, num_attention_heads = 8, 32
        hidden_size, head_dim = 4096, 128

    base = dict(window_size=8, num_sink_tokens=5, local_window_size=64,
                cache_budget=0.2)
    on = WindowedCacheConfig(**base, quant_ratio=0.7).resolve(
        4096, _Model(), torch.float16, 256)
    assert on.quant_sketch_enabled and on.quant_gate_ratio == 0.25
    # q=0 has no Q tier, so AUTO switches itself off rather than erroring.
    off = WindowedCacheConfig(**base, quant_ratio=0.0).resolve(
        4096, _Model(), torch.float16, 256)
    assert not off.quant_sketch_enabled
