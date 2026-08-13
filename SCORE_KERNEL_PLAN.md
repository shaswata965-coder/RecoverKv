# Fused eviction-score kernel — design

Goal: replace the auxiliary score reconstruction in
`modules/windowed_cache/hooks.py` (lines 331–359) with a Triton kernel that
gets FlashAttention-2's speed while producing the per-key attention sum the
eviction policy needs.

Nothing downstream changes. The kernel still produces `token_scores`
`[B, H_q, S]`, and it still feeds `reduce_two_tier_scores` /
`reduce_token_scores_to_windows` exactly as today (`hooks.py:359-374`).

Reference: Dao, *FlashAttention-2* (arXiv:2307.08691), Algorithms 1 and 2.


## 1. Why you can't just fuse scoring into the normal flash-attn forward pass

**In plain terms:** Flash attention processes keys in chunks and keeps a
"running best guess" of the softmax as it goes, correcting that guess as
each new chunk arrives. The correction it applies is *per query token*. Our
score is a sum *across* query tokens, for each key. By the time we've summed
across queries, the per-query correction factor is gone — there's no way to
fix up the sum after the fact, because the thing you'd need to fix it with
was specific to each query row, and those rows have already been collapsed
away. So you can't get a correct running score out of the same loop that
computes the running output. It's not a missing optimization — the two
quantities need the correction applied at different points, and one of those
points no longer exists once you've summed.

**In equations:** Flash attention tracks, per query row `i`, a running max
`m_i` and running sum `ℓ_i` as it walks through key blocks `j = 1, 2, …`:

```
m_i ← max(m_i, rowmax(S_i))
ℓ_i ← exp(m_i_old − m_i)·ℓ_i + rowsum(P̃_i)
O_i ← exp(m_i_old − m_i)·O_i + P̃_i · V_j        (unnormalized output)
```

The correction term `exp(m_i_old − m_i)` is indexed by query row `i`. Since
`O_i` is also indexed by `i`, that correction is just "rescale this row" —
cheap and exact.

Our target is a sum over `i` (queries), for each key `s`:

```
token_scores[s] = Σ_i P[i, s]
```

The correction that would make a running partial sum exact is
`exp(m_i − L_i)`, which still depends on `i` — but `i` is exactly the axis
we're summing over. You cannot pull a `Σ_i` out of `Σ_i (something depending
on i)` after the fact. The only way to apply it correctly is to keep the full
un-summed `[queries, keys]` tile around until every row has its final `L_i` —
which is the large matrix we're trying to avoid materializing in the first
place.

**Where this shows up in code today:** this is exactly why
`modules/windowed_cache/hooks.py` exists as a *separate* pass after flash
attention runs, instead of pulling the score out of flash's own forward.


## 2. The fix: borrow flash attention's *backward* pass instead of its forward

**In plain terms:** Flash attention already has a step, in its backward pass
(used for training), that computes something summed *over queries, for each
key* — because that's exactly the shape of the gradient it needs for V. We
don't need gradients, but we need a sum with the identical shape. So instead
of inventing something new, we just run that half of flash's backward pass,
with the "gradient" input replaced by a vector of all 1s. The matmul that
would normally combine per-query gradients into a per-key gradient just adds
them up unweighted instead — which is exactly the count we want.

**In equations:** Flash attention's backward computes, for each key block
`j`, accumulated over all query blocks `i`:

```
dV_j = Σ_i  Pᵀ_i · dO_i
```

If we set `dO_i = 1` (a column of ones, one per query row) instead of the
real output-gradient, this becomes:

```
dV_j = Σ_i  Pᵀ_i · 1  =  Σ_i  (column-sum of P_i)  =  token_scores_j
```

That's our score, exactly. It's the `dV` computation with the output
dimension collapsed to a single "1", so the "gradient matmul" degenerates
into a plain column-sum.

**Why the loop order matters:** flash's backward loops **outer over key
blocks, inner over query blocks** — the opposite of the forward, which loops
outer over query blocks. That's because `dV_j` (and our score) is a per-key
accumulator: you want to finish one key's total before moving to the next,
same as we do.


## 3. Why the second pass is cheap, not another full attention computation

**In plain terms:** Normally, computing softmax needs a running max and a
running-sum trick, because you don't know the biggest score until you've
seen everything. But by the time we're doing this second, backward-style
pass, flash attention has *already finished* the forward pass and already
knows the true final normalizer for every query row. So instead of tracking
a running estimate and correcting it later, we can just plug in the true
answer from the start. That deletes almost all of the bookkeeping — no
running max, no rescaling, no risk of overflow.

**In equations:** flash's forward already computes and can save, per query
row `i`:

```
L_i = m_i + log(ℓ_i)          (log-sum-exp — the exact softmax normalizer)
```

Given `L_i`, the exact softmax probability is just:

```
P[i, s] = exp(S[i, s] − L_i)
```

with no running max, no rescaling, and no overflow risk (because
`S[i, s] ≤ L_i` always, so the exponent is always ≤ 0). The second pass is
then just: recompute `S = Q·Kᵀ` for the block, subtract `L`, exponentiate,
and sum. Plain matmul + exp + reduce — nothing else.

**Cost to enable it:** flash's forward has to actually *keep* `L_i` around
(one number per query row) instead of throwing it away after the forward
pass finishes. That's a tiny `[B, H_q, T]` array — a few MB — not a
second full attention computation.


## 4. The two-stage design

**In plain terms:** Stage A is completely normal flash attention — same
speed, same output — except it saves that one small `L` array instead of
discarding it. Stage B is a second, smaller kernel that reuses `L` to
recompute just the per-key sums, one key-block at a time, using the "no
bookkeeping needed" shortcut from §3.

**Stage A — flash forward, unchanged except for `L`:**

```
O, L = flash_forward(Q, K, V)     # same as today, plus L is kept instead of discarded
```

**Stage B — the score kernel, pseudocode:**

```
program (key_block_j, batch, kv_head):
    load K_j into SRAM                       # once, reused below
    acc = 0                                   # one running total per key in this block

    for each query_block_i that can see key_block_j (see §5):
        for each query head sharing this kv_head (GQA group):
            load Q_i, L_i
            S = scale * Q_i @ K_jᵀ
            P = exp(S − L_i)                  # exact softmax, no running max needed
            acc += column_sum(P)              # sum over the query axis

    write token_scores[batch, kv_head, key_block_j] = acc
```

Because each program owns one key block and writes only to that block's slice
of `token_scores`, different programs never write to the same memory — plain
stores, no atomics, no race conditions. `K_j` is loaded once and reused
across every query block and every query head in the group, instead of being
re-read from memory for each one (which is what today's chunked-loop version
effectively does).


## 5. Skipping key blocks that are entirely in the future (causal masking)

**In plain terms:** With causal attention, a query can only see keys at or
before its own position. For a given block of keys, if every query in some
query-block is *before* that whole key-block, none of those queries can see
any of those keys — the entire tile is wasted work today. Right now the code
computes the full tile and *then* masks it out; the better approach is to
never compute it at all. Roughly half of all tiles fall into this
"completely masked, skip it" category.

**In equations / code today:** the existing mask logic
(`hooks.py:344-352`) is:

```python
causal = torch.triu(ones(blk, S), diagonal=S - T + start + 1)
aw = aw.masked_fill(causal, float("-inf"))
```

This computes the full `[blk, S]` tile first, then zeroes out the invalid
part. The kernel instead checks, before doing any matmul for a given
`(query_block, key_block)` pair, whether the whole tile is invalid — and if
so, skips it:

| relationship between the two blocks | what the kernel does |
|---|---|
| every query in the block is *after* every key in the block | compute normally, no mask needed |
| every query in the block is *before* every key in the block | **skip entirely — no matmul** |
| the blocks overlap on the diagonal | compute, then mask just this one tile |

Only one tile per query-block ever needs the "compute-then-mask" treatment;
everything else is either fully valid or fully skipped.


## 6. Why decode (T=1) doesn't use any of this

**In plain terms:** all of the above exists to avoid materializing a big
grid of query-vs-key scores. During decode you generate one token at a time,
so there is no grid — just one query against all past keys. There's nothing
to tile, so none of flash attention's tricks (or this kernel's tricks) apply.
Decode instead just runs one plain matmul → softmax → matmul, and reads the
resulting probabilities directly for scoring — no `L`, no second pass, no
kernel needed at all. This kernel is a prefill-only optimization.


## 7. Flags — things this design does NOT preserve for free

### Flag 1: this changes the actual numbers (not bit-exact)

**In plain terms:** today's score is computed in a way that gives one
specific rounding of the answer. This kernel computes the mathematically
same thing, but by adding the numbers up in a different order and at higher
precision. Floating-point math isn't perfectly consistent about that — add
the same numbers in a different order, or at different precision, and you
can get a very slightly different final digit. So the two versions won't
match bit-for-bit, even though both are "correct."

**Concretely:**
- Today: softmax computed in bf16 by default, accumulated in query dtype
  (`hooks.py:330-355`).
- Kernel: softmax and the sum computed in fp32, accumulated tile-by-tile in
  a different traversal order (by key block, not by fixed 1024-row chunks).

**What this means practically:** don't test this change by asserting the
scores match exactly. Instead check that the *decisions* made from those
scores — which windows get evicted — come out the same, since that's the
only thing downstream actually depends on.

### Flag 2: this can no longer be "just a hook"

**In plain terms:** today's setup works by letting the normal attention
forward run untouched, then peeking at what it computed after the fact.
This design needs `L` — a number that only exists *inside* the forward
pass and gets thrown away before anything outside could see it. So instead
of peeking from the outside, we need to change the forward pass itself
to hand `L` out.

**Concretely:** this means replacing the attention forward function used
during prefill (via transformers' pluggable-attention mechanism, or a
monkeypatch — whichever exists in the pinned transformers version, this
needs to be checked, not assumed). Once that's done, several things the
current hook does by necessity become unnecessary and can be deleted: the
`q_proj` stash (`hooks.py:213-234`), the RoPE recompute
(`hooks.py:290-299`), and the effective-K handoff (`hooks.py:269-278`) —
because the new forward pass already has all of that in hand, it doesn't
need to be smuggled out to a separate hook.

### Flag 3: a scoping simplification that's worth knowing about

**In plain terms:** this kernel only ever has to handle the simple case
where "key number 5" really is the 5th key in time order. There's a more
complicated internal layout (sink tokens + quantized tokens + recent tokens,
possibly reordered) that the cache uses — but that reordering only exists
*after* the first eviction has happened, which only happens during decode.
During prefill, none of that reordering exists yet, so the kernel can use
the simple, position-equals-index assumption without any extra bookkeeping
for the reordering.


## 8. Rough cost picture (measure before trusting)

| | today | proposed |
|---|---|---|
| real attention (Stage A) | flash, fused, fast | same, plus a small `L` write |
| score pass (Stage B) | separate eager pass, computes-then-masks, re-reads K from memory repeatedly, bf16 | fused kernel, skips fully-masked tiles, K loaded once and reused, fp32 |

The FLOP count only drops modestly. The real win is that the score pass
stops being a slow, memory-bound Python loop and becomes a fast,
GPU-resident kernel — closer to flash-attention's own speed instead of
running as a much slower bolt-on. This needs to be benchmarked, not assumed.


## 9. Deliberately not doing (footnotes, not day-one work)

- **Fusing Stage B directly into Stage A's own loop:** possible in theory
  (by the time Stage A finishes a query block, it already has that block's
  final `L`), but the score accumulator is organized per-key while Stage A's
  loop is organized per-query-block — so multiple query blocks would be
  writing to overlapping key ranges at the same time, requiring atomic
  (contended) writes instead of the clean, non-overlapping writes Stage B
  gets by looping per-key. Not worth the complexity unless proven necessary.
- **Splitting further for GPU occupancy:** only relevant if the number of
  independent kernel launches is too small to fill the GPU, which is not
  expected to be an issue at the sequence lengths this repo runs at.
