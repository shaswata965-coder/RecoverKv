"""``stable_partition`` is ``argsort(~mask, stable=True)``, and must stay so.

The eviction compacts four different things by partitioning a mask, and every
one of them used a sort to do it. Replacing that with a cumsum is only
legitimate if the permutation is **identical** — not "equivalent up to the part
we look at". Three of the four callers slice a prefix and never read the rest,
but the fourth writes through the whole width (``QuantSlotTable.write`` sizes by
a worst-case width and masks the value, not the index), so two trailing lanes
landing on one slot would be a lost demotion, not a cosmetic difference.

So this file asserts the whole permutation, on shapes and densities chosen to hit
the edges: no trues, all trues, one true, ragged rows, and more than two axes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modules.quant.compact import stable_partition  # noqa: E402


def _ref(mask):
    return torch.argsort(~mask, dim=-1, stable=True)


@pytest.mark.parametrize("shape", [(1, 1), (4, 16), (3, 2, 9), (2, 3, 4, 7), (5, 64)])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_it_is_the_sort_it_replaces(shape, p):
    g = torch.Generator().manual_seed(hash((shape, p)) % (2 ** 31))
    mask = torch.rand(shape, generator=g) < p
    assert torch.equal(stable_partition(mask), _ref(mask))


def test_rows_partition_independently():
    """A row with no trues and a row with all of them, side by side."""
    mask = torch.tensor([[True, False, True, False],
                         [False, False, False, False],
                         [True, True, True, True]])
    got = stable_partition(mask)
    assert got.tolist() == [[0, 2, 1, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    assert torch.equal(got, _ref(mask))


def test_every_index_appears_exactly_once():
    """It is a permutation. The trailing lanes are what makes this matter.

    ``write`` pushes masked-off lanes through as real indices; two of them
    aliasing one slot would make the read-modify-write nondeterministic between a
    fresh demotion and the slot's old contents.
    """
    g = torch.Generator().manual_seed(7)
    for p in (0.0, 0.25, 0.75, 1.0):
        mask = torch.rand(6, 33, generator=g) < p
        order = stable_partition(mask)
        for row in order:
            assert sorted(row.tolist()) == list(range(33))


def test_the_selected_prefix_keeps_column_order():
    """The property the callers actually rely on: selection does not reorder.

    ``gated_q_tier`` gathers positions with this prefix and hands them to
    ``compute_score_meta_gated``, which assumes the selected columns are still
    ascending by window id.
    """
    g = torch.Generator().manual_seed(11)
    mask = torch.rand(4, 20, generator=g) < 0.4
    n = int(mask[0].sum())
    for row_mask, row_order in zip(mask, stable_partition(mask)):
        picked = row_order[: int(row_mask.sum())].tolist()
        assert picked == sorted(picked)
        assert all(bool(row_mask[i]) for i in picked)
    assert n >= 0


def test_it_does_not_sort():
    """The point of the exercise: no ``aten.sort`` reaches the dispatcher.

    Inductor has no lowering for sort, so one left here would keep the compiled
    eviction split around an extern kernel — which is the thing this helper
    exists to stop.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    seen = []

    class Watch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            if "sort" in name or "topk" in name:
                seen.append(name)
            return func(*args, **(kwargs or {}))

    mask = torch.rand(3, 32, generator=torch.Generator().manual_seed(2)) < 0.5
    with Watch():
        stable_partition(mask)
    assert seen == [], f"stable_partition dispatched {seen}"
