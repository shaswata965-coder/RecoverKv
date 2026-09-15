"""Digest storage in the Q-tier slot table (DIGEST_GATED_DECODE_PLAN.md §5.2).

Pure CPU: :mod:`modules.quant.slots` imports nothing but ``torch``, so these
collect and run on a GPU-less box (§9). The decode gate that consumes what is
stored here is tested in ``tests/test_digest_gate.py``.

What these pin:

- the digest survives a write -> gather round trip unchanged;
- it is **write-once** in the same sense as the record — a window that goes
  Q -> fp -> Q keeps the digest built from its original post-RoPE keys, because
  reactivation flips a bit and never re-enters ``write()``;
- with storage off, nothing is allocated and a digest cannot be smuggled in;
- the stored box actually **encloses** its keys, which is the one property the
  whole gate rests on (an over-estimate costs a dequant, an under-estimate
  silently drops a key the model wanted).
"""

import pytest
import torch

from modules.quant.digest import build_aabb, estimate_aabb
from modules.quant.slots import FREE, QuantSlotTable

B, N, WS, D, H = 2, 6, 8, 16, 4


def _table(store_digest=True, batch=B):
    return QuantSlotTable(
        batch_size=batch, n_slots=N, window_size=WS, head_dim=D,
        num_kv_heads=H, device=torch.device("cpu"), store_digest=store_digest,
    )


def _payload(n, batch=B, seed=0):
    """A full ``write()`` argument set for ``n`` lanes per row."""
    g = torch.Generator().manual_seed(seed)
    return dict(
        k_codes=torch.randint(0, 255, (batch, n, H, D, WS // 4), dtype=torch.uint8, generator=g),
        k_scale=torch.randn(batch, n, H, D, generator=g).half(),
        k_zero=torch.randn(batch, n, H, D, generator=g).half(),
        v_codes=torch.randint(0, 255, (batch, n, H, WS, D // 4), dtype=torch.uint8, generator=g),
        v_scale=torch.randn(batch, n, H, WS, generator=g).half(),
        v_zero=torch.randn(batch, n, H, WS, generator=g).half(),
        pos=torch.arange(n * WS).reshape(1, n, WS).expand(batch, n, WS).contiguous(),
    )


def _keys(n, batch=B, seed=7):
    """Post-RoPE keys ``[B, n, H_kv, ws, D]`` with per-channel scale spread.

    Channels deliberately differ in scale: that is this data's regime and the
    reason the plan settles on an AABB over a sphere (§1.2).
    """
    g = torch.Generator().manual_seed(seed)
    scale = torch.logspace(-1, 1, D).reshape(1, 1, 1, 1, D)
    return torch.randn(batch, n, H, WS, D, generator=g) * scale


def test_digest_survives_write_then_gather():
    t = _table()
    n = 3
    keys = _keys(n)
    lo, hi = build_aabb(keys)
    slots = t.free_slots(n)
    valid = torch.ones(B, n, dtype=torch.bool)
    wid = torch.arange(n).unsqueeze(0).expand(B, n).contiguous()

    t.write(slots, valid, wid, **_payload(n), k_digest_lo=lo, k_digest_hi=hi)
    *_, g_lo, g_hi = t.gather(slots)

    assert torch.equal(g_lo.reshape(B, n, H, D), lo.half())
    assert torch.equal(g_hi.reshape(B, n, H, D), hi.half())


def test_stored_digest_encloses_its_own_keys():
    """The bound property, read back out of the table rather than off build_aabb.

    This is what makes the gate safe: every key of the window must lie inside
    the box the table hands the scorer, so the estimate can only ever be too
    high. fp16 storage rounds the corners, so the check is against the rounded
    box — which is exactly the box the gate will use.
    """
    t = _table()
    n = 4
    keys = _keys(n, seed=11)
    lo, hi = build_aabb(keys)
    slots = t.free_slots(n)
    valid = torch.ones(B, n, dtype=torch.bool)
    wid = torch.arange(n).unsqueeze(0).expand(B, n).contiguous()
    t.write(slots, valid, wid, **_payload(n), k_digest_lo=lo, k_digest_hi=hi)

    *_, g_lo, g_hi = t.gather(slots)
    g_lo = g_lo.reshape(B, n, H, D).float()
    g_hi = g_hi.reshape(B, n, H, D).float()

    # fp16 rounds the corners either way, so allow the half-ULP the round costs.
    tol = torch.finfo(torch.float16).eps * keys.abs().amax()
    assert (keys.amin(dim=3) >= g_lo - tol).all(), "a key sits below the stored lo"
    assert (keys.amax(dim=3) <= g_hi + tol).all(), "a key sits above the stored hi"


def test_estimate_from_stored_digest_upper_bounds_real_dot_products():
    """End-to-end direction check: estimate >= max_k q.k, for real queries.

    The digest is a *selector*: over-estimating costs an unnecessary dequant,
    under-estimating silently drops a window the model wanted. So the assert is
    one-sided on purpose.
    """
    t = _table(batch=1)
    n, H_q = 4, H * 2                      # GQA: 2 query heads per KV head
    keys = _keys(n, batch=1, seed=23)
    lo, hi = build_aabb(keys)
    slots = t.free_slots(n)
    t.write(slots, torch.ones(1, n, dtype=torch.bool),
            torch.arange(n).reshape(1, n), **_payload(n, batch=1),
            k_digest_lo=lo, k_digest_hi=hi)
    *_, g_lo, g_hi = t.gather(slots)
    g_lo = g_lo.reshape(1, n, H, D).float()
    g_hi = g_hi.reshape(1, n, H, D).float()

    q = torch.randn(1, H_q, 1, D, generator=torch.Generator().manual_seed(31))
    est = estimate_aabb(q, g_lo, g_hi)                       # [1, H_q, 1, n]

    # Real max over each window's keys, per query head (query head h reads KV
    # head h // rep, the convention estimate_aabb itself uses).
    rep = H_q // H
    q_g = q.reshape(1, H, rep, D)                            # head h -> KV h//rep
    kk = keys.permute(0, 2, 1, 3, 4)                         # [1, H, n, WS, D]
    real = torch.einsum("bhrd,bhgsd->bhrgs", q_g, kk).amax(-1)   # [1, H, rep, n]
    real = real.reshape(1, H_q, 1, n)
    assert (est >= real - 1e-3).all(), "the digest under-estimated a real q.k"


def test_reactivation_keeps_the_original_digest():
    """Q -> fp -> Q must not re-derive the box (design §10, plan §3.1).

    A re-demotion is a bit flip through ``set_active``; it never reaches
    ``write()``, so there is no path on which a second digest could be built.
    Re-deriving one would let it disagree with the codes it summarizes.
    """
    t = _table()
    n = 2
    keys = _keys(n, seed=5)
    lo, hi = build_aabb(keys)
    slots = t.free_slots(n)
    valid = torch.ones(B, n, dtype=torch.bool)
    wid = torch.arange(n).unsqueeze(0).expand(B, n).contiguous()
    t.write(slots, valid, wid, **_payload(n), k_digest_lo=lo, k_digest_hi=hi)
    *_, before, _ = t.gather(slots)

    t.set_active(slots, valid, False)          # promote to fp (dormant)
    assert not t.slot_active.gather(1, slots).any()
    t.set_active(slots, valid, True)           # re-demote: pure reactivation
    assert t.slot_active.gather(1, slots).all()

    *_, after, _ = t.gather(slots)
    assert torch.equal(before, after)


def test_invalid_lanes_leave_a_neighbours_digest_untouched():
    """``write()`` masks the value, not the index — digests must obey that too.

    Trailing lanes are sized by the worst case and can point at an occupied
    slot, so a masked-off lane must write nothing rather than zero a live box.
    """
    t = _table()
    keys = _keys(2, seed=9)
    lo, hi = build_aabb(keys)
    slots = t.free_slots(2)
    t.write(slots, torch.ones(B, 2, dtype=torch.bool),
            torch.arange(2).unsqueeze(0).expand(B, 2).contiguous(),
            **_payload(2), k_digest_lo=lo, k_digest_hi=hi)
    *_, before, _ = t.gather(slots)

    # Re-write the SAME slots with all lanes invalid and a different box.
    other_lo, other_hi = build_aabb(_keys(2, seed=99) + 50.0)
    t.write(slots, torch.zeros(B, 2, dtype=torch.bool),
            torch.full((B, 2), FREE), **_payload(2, seed=3),
            k_digest_lo=other_lo, k_digest_hi=other_hi)

    *_, after, _ = t.gather(slots)
    assert torch.equal(before, after)


def test_storage_off_allocates_nothing_and_gathers_none():
    t = _table(store_digest=False)
    assert t.key_digest_lo is None and t.key_digest_hi is None
    slots = t.free_slots(2)
    t.write(slots, torch.ones(B, 2, dtype=torch.bool),
            torch.arange(2).unsqueeze(0).expand(B, 2).contiguous(), **_payload(2))
    *_, lo, hi = t.gather(slots)
    assert lo is None and hi is None


def test_mismatched_digest_and_storage_raise_rather_than_drop():
    """Both directions are silent-wrongness bugs, so both are hard errors."""
    keys = _keys(2)
    lo, hi = build_aabb(keys)
    args = (torch.zeros(B, 2, dtype=torch.long),
            torch.ones(B, 2, dtype=torch.bool),
            torch.arange(2).unsqueeze(0).expand(B, 2).contiguous())

    on = _table(store_digest=True)
    with pytest.raises(ValueError, match="all-zero box"):
        on.write(*args, **_payload(2))

    off = _table(store_digest=False)
    with pytest.raises(ValueError, match="dropped silently"):
        off.write(*args, **_payload(2), k_digest_lo=lo, k_digest_hi=hi)


def test_join_layers_carries_digests_and_rejects_a_mismatch():
    n = 2
    keys = _keys(n, batch=1, seed=13)
    lo, hi = build_aabb(keys)
    tables = []
    for _ in range(3):
        t = _table(batch=1)
        slots = t.free_slots(n)
        t.write(slots, torch.ones(1, n, dtype=torch.bool),
                torch.arange(n).reshape(1, n), **_payload(n, batch=1),
                k_digest_lo=lo, k_digest_hi=hi)
        tables.append(t)

    joint = QuantSlotTable.join_layers(tables)
    assert joint.store_digest
    assert joint.key_digest_lo.shape == (3, N, H, D)
    for i in range(3):
        assert torch.equal(joint.key_digest_lo[i], tables[i].key_digest_lo[0])

    with pytest.raises(RuntimeError, match="geometry differs"):
        QuantSlotTable.join_layers([tables[0], _table(store_digest=False, batch=1)])


def test_join_layers_without_digests_stays_none():
    tables = [_table(store_digest=False, batch=1) for _ in range(2)]
    joint = QuantSlotTable.join_layers(tables)
    assert not joint.store_digest
    assert joint.key_digest_lo is None and joint.key_digest_hi is None
