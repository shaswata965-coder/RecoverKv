# Adapting FlashAttention-2 to produce eviction scores

Compact record of the design discussion. Scope: **only** how to get per-key
eviction scores out of a FlashAttention-style pass. Excludes the decode-path
change, the effective-KV arena, and the transformers-4.47.1 integration.

Reference: Dao, *FlashAttention-2* (arXiv:2307.08691), Algorithms 1 and 2.


## The problem

Eviction needs, per cached key, the total attention it received summed over all
queries:

```
token_scores[s] = Σ_i P[i, s]           per key, summed over queries
```

FlashAttention keeps nothing of this shape. Everything it keeps is **per query**:

```
ℓ_i = Σ_s exp(S[i,s] − m_i)             per query, summed over keys
```

`ℓ_i` is the exact transpose of what we need — it has already collapsed the key
axis we want to keep, and kept the query axis we want to collapse. So the score
cannot be read off flash's saved state; it has to be computed.

This is why `modules/windowed_cache/hooks.py` exists as a **separate pass** after
flash runs (today: recompute `S = QKᵀ`, softmax, sum over queries, chunked over
query rows to bound peak memory — `hooks.py:334-356`). The eager backend gets the
same number for free (`windowed_eager_cache/hooks.py:150-168`) only because eager
materializes the full `P` matrix, so summing it is trivial.


## Dead ends (and why)

**Store `ℓ` and correct at the end.** Wrong axis (see above): `ℓ` is per-query,
we need per-key. Unrecoverable — `score` depends on all `T·S` matrix entries,
`ℓ` is only `T` numbers.

**Sum into a per-key accumulator inside flash's key loop.** At key block `j` you
hold `P̃ = exp(S − m^(j))` against a *running* max. The correction to true
softmax is `exp(m_i^(j) − L_i)`, indexed by query `i` — and it sits inside the
`Σ_i`. You cannot pull a per-`i` factor out of a sum over `i`. Correcting it
means keeping the un-summed `[BLOCK_M, BLOCK_N]` tile until `L` is final — i.e.
materializing, the thing flash exists to avoid.

**Shrink the query chunk until the tile stays in SRAM.** Chunking bounds *peak*
memory, not *total* bytes moved:

```
total P elements = ⌈T/chunk⌉ · chunk · S = T · S    (chunk cancels)
peak memory      = chunk · S                         (chunk matters)
```

Solving for SRAM residency: `chunk · S · 2 ≤ ~164 KB` at `S≈18000` gives
`chunk ≤ ~4` query rows — a dead matmul, `T/4 ≈ 4500` iterations. SRAM residency
therefore *forces* tiling the key axis, which brings back the running max, which
kills the single-pass sum. **Single-pass and SRAM-resident are mutually
exclusive** unless you change what you accumulate (next section).


## Design A — window-tiling in one pass (primary)

**Key idea: don't collapse `ℓ` all the way.** `ℓ_i` throws away *which* keys the
attention went to. Keep a vector per query — one slot per window — instead of a
scalar:

```
scalar:   ℓ_i    ← exp(m_old − m_i)·ℓ_i          + rowsum(P̃_i^(j))
vector:   W_i[:,w(j)] ← exp(m_old − m_i)·W_i[:,w(j)] + rowsum(P̃_i^(j))
```

`ℓ_i = Σ_w W_i[:,w]` — identical computation, just not summed to the bottom. The
per-row rescale `exp(m_old − m_i)` applies to `W_i` for the **same reason** it
applies to `O_i`: both are indexed by query row, so a per-row scalar broadcasts
cleanly. Equivalently, `W_i` is `O_i` computed against a fake "value" that is a
one-hot marker of which window each key belongs to (`O_i = P̃ᵀV_j`,
`W_i = P̃ᵀ·window-indicator`). Divide by the final `ℓ_i` at row end — the same
normalization `O_i` gets — and the per-query-per-window scores are exact.

**Why it fuses cheaply: `BLOCK_N = window_size`.** Size the key tile to the
window and flash's *existing* inner loop already processes exactly one window per
step, so `w(j) = j` — no new indexing, just relabel a column flash already
produces. This *is* "per query, how much attention it paid to each past window,
then add across queries at the end."

**The combine.** Different query blocks run on different programs and finish
their own `W_i` independently — collisions happen only once, at the end, adding
each block's finished window totals into the per-window total:

```
combine writes = num_query_blocks · num_windows ≈ (T/BLOCK_M)·(S/window_size)
               = (18000/64)·(18000/32) ≈ 282 · 562 ≈ 158K atomic adds / layer
```

Two orders of magnitude below the ~80M atomics of fusing at key-block
granularity, because contention is once-per-row-after-finish, not every inner
step.

**The binding constraint — `W_i` must stay resident all through the row** (the
running-max rescale touches every slot when the max moves, so no slot can be
evicted mid-row). Size, `BLOCK_M=64`, vs. ~164 KB SRAM/SM:

```
num_windows = S / window_size
window_size=32:  64·562·4 ≈ 144 KB fp32 / 72 KB bf16   → fits (bf16 comfortable)
window_size=8:   64·2250·4 ≈ 576 KB fp32 / 288 KB bf16 → does NOT fit (even H100 ~228 KB)
```

At `window_size=8` you cannot tile `W_i` smaller (residency is mandatory); the
only lever is `BLOCK_M ≤ ~22`, i.e. `BLOCK_M=16`, which starves the tensor cores
and may erase the win. **So Design A is the primary path for `window_size ≥ 32`
(the configs actually run — `eval_parity_base.yaml:20` uses 32) and stops paying
off at the `base.yaml` default of 8.**


## Design B — recompute from `L`, outer-loop over keys (fallback)

For small windows where `W_i` won't fit, fall back to a two-pass design that
holds no per-window buffer on-chip, so `window_size` is irrelevant to it.

**Shape: this is flash's own backward pass.** FA2 Algorithm 2 loops **outer over
key blocks, inner over query blocks**, recomputing `P` from the saved `L`:

```
P_i^(j) = exp(S_ij − L_i)          exact softmax, L final
dV_j    = Σ_i P_i^(j)ᵀ · dO_i      per-key accumulator
```

Set `dO = 𝟙` and the matmul degenerates to a column sum:

```
score_j = Σ_i P_i^(j)ᵀ · 𝟙 = Σ_i colsum(P_i^(j))
```

**Our score is `dV` with the V-dimension collapsed to 1.** With `L` final,
`P = exp(S − L)` is exact on first touch — no running max, no rescale, no
overflow guard (`ℓ ≥ 1` ⟹ `S − L ≤ 0` ⟹ `exp ≤ 1`). Grid over key blocks ⟹
disjoint output slices ⟹ **no atomics** (same reason FA2's backward reserves
atomics for `dQ` alone). `K_j` loads once, reused across every query block and
all `n_rep` GQA heads.

**Why recompute beats store-and-reload:**

```
extra QKᵀ recompute:  1.33 TFLOP / ~150 TFLOPS ≈  8.9 ms
store + reload P:      41.4 GB   / ~1.5 TB/s   ≈ 27.6 ms   (~3× worse)
```

FA2's backward makes the identical choice for the identical reason.

**Stage A for both designs** needs `L` kept from the forward.
`flash_attn_func(..., return_attn_probs=True)` returns `softmax_lse` = `L`;
**verify empirically** before relying on it (the flag is documented as
test-oriented). If it holds, Stage A is a one-line change.


## Causal geometry (both designs)

Convention (`hooks.py:346-351`): query `t` sits at absolute position `S−T+t`,
attends keys `0 … S−T+t`. Per (query block `i`, key block `j`):

| condition | action |
|---|---|
| `j0+BN−1 ≤ i0+(S−T)` | compute, no mask |
| `j0 > i0+BM−1+(S−T)` | **skip entirely** |
| otherwise | compute, mask this tile |

Paper reports 1.7–1.8× from skipping fully-masked tiles; only ~1 tile per query
row needs the mask applied. Today's loop computes the full tile then masks
(`hooks.py:344-352`) — it pays for the skipped half.

**Prefill-only simplification:** at prefill the Q tier is empty
(`store.num_active_windows == 0`), so `materialize_effective_kv` returns the fp
store unchanged (`quant/effective.py:288-289`). The unsorted `[sink ‖ body ‖ Q]`
layout that would break "key index = position" only appears after the first
eviction, a decode-step event (`policy.py:124-128`). So the kernel never sees a
permuted key axis and needs no `order`-aware masking.


## Why windows must be fixed-length (both designs)

1. **Tile-to-window alignment.** The fusion relies on `BLOCK_N = window_size` so
   each step is one window. Variable windows straddle tile boundaries → one
   `rowsum` mixes two windows → segmented reduction inside every tile, losing the
   "free relabel" property.
2. **Static shared memory.** `W_i` width is `num_windows`, which must be known at
   compile time to allocate. Variable windows make it data-dependent and, worse,
   per-row divergent (rows evict different windows, BATCHING_PLAN §3) → ragged,
   un-allocatable.
3. **Downstream reduction hard-codes it.** `scorer.py:87-89` is
   `reduce(post_sink, "b h (w s) -> b h w", "sum", s=window_size)` — a reshape
   that requires `S` to factor into equal `window_size` groups. Variable windows
   break it outright.

Fixed `window_size` is the shared assumption that lets the tile align, the buffer
be statically sized, and the reduction be a reshape. The size may change; it must
be one size.


## Honest framing of the win

Neither design is faster than a *bare* flash forward — both add work (Design A: a
wider resident buffer + per-step rescale of it + a small combine; Design B: a
second key-outer pass). What they beat is **today's pipeline** — flash forward
*plus* a separate `O(T·S)` reconstruction pass — by deleting that second pass (A)
or replacing it with a cheaper recompute that never materializes `P` (B).

The scoring cost is `O(T·S)` and unavoidable; both designs are about paying it
once, on-chip, instead of as a memory-bound eager chain that round-trips a ~20 GB
`P` tensor through HBM per layer. **Benchmark the ratio, don't trust the
absolutes.**


## Numerics flag

Not bit-exact: fp32 registers vs. today's bf16 default (`hooks.py:330`), and a
different summation order (per-key-block / per-window tiles vs. fixed 1024-row
chunks). FP addition is non-associative, so it won't match even
`STICKYKV_SCORE_SOFTMAX_DTYPE=float32`. More accurate, but different — validate
on **retained window sets** (all the policy consumes), not raw scores.
