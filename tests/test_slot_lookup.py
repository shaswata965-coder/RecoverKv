"""``lookup`` answers four questions from one ``[B, W, N]`` match — not six.

That match tensor is the largest transient in the eviction: at ``B·L = 1024``
rows, ``W = 512`` windows and ``N = 318`` slots it is 167 MB, and the old form
walked it six times and materialised two more copies of it (the ``&`` against
``slot_active``, and the ``uint8`` cast for ``argmax``).

Three of those passes were answering questions the ``[B, W]`` results already
answer, and the rewrite leans on the table's own invariant — **at most one slot
per row carries a given window id** — to do it:

* ``max(-1)`` returns the value and its index, so ``has_entry`` and ``slot_of``
  come from one pass.
* "is that entry active" is ``slot_active`` read at ``slot_of``.
* "does some retained id live in slot j" is the same map inverted: a scatter.

Each of those is exact only because of the invariant, so this file checks them
against the reductions they replace, on random tables that satisfy it — including
the case that broke the first attempt, where a row matches nothing and
``argmax`` returns slot 0, which a naive scatter then stamps ``False`` over a
real hit at slot 0.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modules.quant.slots import FREE, QuantSlotTable  # noqa: E402


def _random_table(B, N, W, g):
    """A table that satisfies the invariant, plus ids to look up."""
    slot_wid = torch.randint(-1, W + 1, (B, N), generator=g)
    for b in range(B):                       # unique ids per row
        seen = set()
        for j in range(N):
            v = int(slot_wid[b, j])
            if v == FREE:
                continue
            if v in seen:
                slot_wid[b, j] = FREE
            else:
                seen.add(v)
    slot_active = (torch.rand(B, N, generator=g) < 0.5) & (slot_wid != FREE)
    wids = torch.stack([torch.randperm(W + 1, generator=g)[:W] for _ in range(B)])
    return slot_wid, slot_active, wids


def _table(slot_wid, slot_active):
    t = QuantSlotTable(batch_size=slot_wid.shape[0], n_slots=slot_wid.shape[1],
                       window_size=4, head_dim=8, num_kv_heads=2,
                       device=torch.device("cpu"))
    t.slot_wid = slot_wid.clone()
    t.slot_active = slot_active.clone()
    return t


@pytest.mark.parametrize("seed", range(12))
def test_lookup_matches_the_reductions_it_replaced(seed):
    g = torch.Generator().manual_seed(seed)
    slot_wid, slot_active, wids = _random_table(3, 9, 6, g)
    t = _table(slot_wid, slot_active)

    match = slot_wid.unsqueeze(1) == wids.unsqueeze(2)          # the old form
    has_entry, is_active, slot_of, keep = t.lookup(wids)

    assert torch.equal(has_entry, match.any(-1))
    assert torch.equal(is_active, (match & slot_active.unsqueeze(1)).any(-1))
    assert torch.equal(slot_of, match.to(torch.uint8).argmax(-1))
    assert torch.equal(keep, match.any(1)), "the keep mask is retain_only's input"


def test_a_row_that_matches_nothing_cannot_erase_slot_zero():
    """The bug the dump column exists for.

    ``argmax`` over an all-False row returns 0, so a lane with no entry points at
    slot 0 exactly like a lane that really matched it. Scattering ``has_entry``
    there writes ``False`` over the real hit; routing the no-entry lanes to a
    dump column instead is what keeps the two apart.
    """
    slot_wid = torch.tensor([[7, FREE, FREE]])        # id 7 lives in slot 0
    slot_active = torch.tensor([[True, False, False]])
    t = _table(slot_wid, slot_active)

    # window 7 is retained (it matches slot 0); 99 and 100 match nothing.
    _, _, _, keep = t.lookup(torch.tensor([[7, 99, 100]]))
    assert keep.tolist() == [[True, False, False]], (
        "slot 0's hit was erased by a lane that matched nothing")


def test_it_still_finds_a_dormant_entry():
    """``has_entry`` covers active AND dormant; only ``is_active`` splits them."""
    slot_wid = torch.tensor([[5, 6]])
    slot_active = torch.tensor([[False, True]])        # 5 dormant, 6 active
    t = _table(slot_wid, slot_active)

    has, active, slot, keep = t.lookup(torch.tensor([[5, 6]]))
    assert has.tolist() == [[True, True]]
    assert active.tolist() == [[False, True]]
    assert slot.tolist() == [[0, 1]]
    assert keep.tolist() == [[True, True]]


def test_retain_only_frees_exactly_the_slots_the_mask_drops():
    slot_wid = torch.tensor([[5, 6, 7]])
    slot_active = torch.tensor([[True, True, False]])
    t = _table(slot_wid, slot_active)

    _, _, _, keep = t.lookup(torch.tensor([[5, 7]]))   # 6 is dropped
    t.retain_only(keep)
    assert t.slot_wid.tolist() == [[5, FREE, 7]]
    assert t.slot_active.tolist() == [[True, False, False]]
