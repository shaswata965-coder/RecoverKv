"""Compaction indices for the eviction, without a sort.

Four places in the eviction and the read path want the same thing: *the columns
where a mask is true, in their existing order, first; everything else after*.
All four expressed it as ``argsort(~mask, stable=True)`` — a boolean is 0 or 1,
so a stable sort of it is a partition, and a partition's prefix is the compaction
they actually wanted.

It works, and it is the wrong instrument. A partition is a **running count**, and
a running count is a cumsum:

* a true column's destination is how many trues precede it;
* a false column's destination is the total number of trues plus how many falses
  precede it.

Both come out of one cumsum and some arithmetic, and writing the permutation is
one scatter. That is ``O(n)`` in one scan against a radix sort's ``O(n log n)``
in several passes, and it matters for a reason beyond the asymptotics:
**Inductor has no lowering for sort.** ``aten.sort`` leaves the compiled eviction
as an extern kernel call on every backend, and fusion cannot cross it; ``cumsum``
lowers to a Triton scan on CUDA and fuses with the pointwise work around it. The
eviction is compiled precisely so that this chain does not land as a string of
separate kernels (``cache.py:_run_compiled_evict``), so an op that cannot be
lowered is worth removing where an equivalent one can be.

What is NOT claimed: a measured speedup. This box has no GPU, so the change is
justified by op count and by what each op does, not by a timing. The
equivalence, on the other hand, is exact and is what ``tests/test_compact.py``
pins — every index, on every shape, against the sort it replaces.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["stable_partition"]


def stable_partition(mask: Tensor) -> Tensor:
    """``argsort(~mask, stable=True)`` over the last axis, without the sort.

    Returns an index tensor of ``mask``'s shape: for each row, the indices of the
    true columns in ascending order, followed by the indices of the false ones in
    ascending order. Callers take a prefix of it (the compaction) and gather
    whatever they need onto that axis.

    The full permutation is returned rather than only the prefix because the
    trailing lanes are load-bearing even when their *contents* are not: they must
    stay **distinct**, so a caller that writes through them (``QuantSlotTable``
    sizes its writes by a worst-case width and masks the value, not the index)
    cannot have two lanes aliasing one slot.
    """
    # CONCRETE. Every caller is inside the compiled eviction, where a raw
    # `.shape` read is a SymInt, and `torch.arange` below needs a real length --
    # a symbolic one reaches Inductor as an integer EXPRESSION (the window axis
    # is a token count over `window_size`, so it arrives as `((I)//8)`) and the
    # lowering fails. `int()` forces specialization here, which is a no-op in
    # eager and byte-identical, and costs no extra recompiles: every width passed
    # in is already specialized by its caller.
    n = int(mask.shape[-1])
    # `cumsum` on a bool promotes to int64 on its own, so the explicit `.to()`
    # this used to do was a whole extra elementwise pass over the mask for a
    # dtype the next op would have produced anyway.
    csum = mask.cumsum(-1)                               # trues at or before i
    n_true = csum[..., -1:]                              # trues in the row
    # NOT cached in a module-global dict. That saved one small `arange` kernel in
    # EAGER and was a hazard everywhere else: the eviction is always compiled, so
    # the tensor this line produces is created while Dynamo traces, and stashing
    # it in a global keyed by a (possibly symbolic) length means a later call can
    # read back a tensor from a dead tracing context. Inside Inductor an `arange`
    # is not a kernel at all -- it fuses into the consumer as an index expression
    # -- so the cache bought nothing on the path that matters.
    idx = torch.arange(n, device=mask.device).expand_as(csum)
    # true  -> its rank among trues;  false -> n_true + its rank among falses.
    # (falses at or before i) - 1 == i - csum[i], which saves the second scan.
    dest = torch.where(mask, csum - 1, n_true + idx - csum)
    order = torch.empty_like(csum)
    return order.scatter_(-1, dest, idx)
