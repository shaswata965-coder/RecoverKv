# What is new here, and why it would matter in a paper

This report covers the **kernel and throughput work** — the custom GPU kernels
built on top of the two-tier int2 cache, and then the per-window *card* and read
gate built on top of those. It says each idea in plain words first, then explains
how the kernel works and what it replaced, then gives the algorithm and the maths
in a form a paper could use directly.

**What already existed, and is not claimed here.** The cache itself: the sequence
is cut into fixed-size windows; every few generation steps each window's
accumulated attention decides a three-way outcome — the best windows stay at full
precision, the next band is compressed to 2 bits per number with a scale that is
fixed the first time the window is written and never recomputed, and the rest are
dropped. Eviction compacts memory but never renumbers positions, so a surviving
key keeps the rotary phase it was born with forever. The memory budget is divided
in **bytes**, so the compressed tier buys roughly four times as many windows as
full precision does at the same cost. All of that predates the work below. It is
the thing the kernels accelerate, and the constraint they have to respect: **none
of these kernels changes which tokens the cache keeps, or the arithmetic that
decides it.**

* **Part A —** the three kernels, and what each one replaced.
* **Part B —** this branch: the card, the read gate, and what measurement said
  about it.
* **Part C —** how the core architecture changed between the two.

---
---

# Part A — The three kernels

## A1. At a glance

**1. The eviction score stopped being a second attention pass.**
To decide what to evict, the cache needs to know how much attention each key
received, totalled over every query that looked at it. The obvious way to get
that is to compute attention a second time and add up the columns — which is what
the code did, and it was expensive enough to be one of the two largest costs in
processing a prompt. It turns out you do not need a second pass at all. Flash
attention's *training* code already computes a quantity with exactly the right
shape — a sum over queries, per key — because that is the shape of the gradient it
needs. Feed it ones where the gradient would go, and the same machinery produces
the eviction score. And because the forward pass has already finished by then, it
already knows the exact normaliser for every query, so all of the running-maximum
bookkeeping that makes attention hard simply disappears. **Nothing new was
invented; an existing correct reduction was repurposed by changing its input.**

**2. The compressed tier is never decompressed into memory.**
Reading a compressed window used to mean expanding it back to 16-bit numbers,
writing that copy to GPU memory, gluing it onto the uncompressed tier, running
attention over the result, and then running attention *again* to get the scores.
Four costs, every layer, every token. The fused kernel does all of it in one
launch: it reads the 2-bit codes, expands and rotates them **inside the chip's
registers**, and never writes the expanded version anywhere. Generating a token is
bandwidth-bound — the GPU streams nearly the whole model for every single token —
so bytes that never needed to exist in memory are pure waste. **The memory the
compression saved was being handed back, briefly, on every step. Now it is not.**
And because a single query row makes the softmax weight *identical to* the
importance score, the eviction score falls out of the same dot product for free.

**3. Eviction stopped being thirty-two separate operations and became one.**
Every layer evicts on the same step, on the same schedule, at exactly the same
shapes — because those shapes come from the configuration, not from the data. So
thirty-two layers were issuing thirty-two small, identically-shaped, independent
passes back to back. Each GPU launch carries a fixed dispatch cost no matter how
little work it does, so that is the same cost paid thirty-two times for no extra
parallelism. Since every operation in the eviction body already treats rows
independently, a *layer* can simply be more rows. One pass replaces thirty-two.
**This is a launch-count argument, not an arithmetic one** — not one byte fewer is
moved.

**The result.** On an 8B model with a 4096-token prompt at batch 32, time per
generated token fell **2.81×**, taking the method from **2.05× slower than an
uncompressed flash-attention cache to 1.37× faster**, with peak memory unchanged.
The three kernels act on different phases (prompt scoring, periodic eviction,
per-token decode) and on different bottlenecks (special-function throughput,
launch overhead, memory bandwidth), which is why their gains compound instead of
overlapping.

---

## A2. How each kernel works

### A2.1 Kernel 1 — the eviction score, from attention's backward pass

**What it replaced.** A separate pass that ran after flash attention returned. It
walked the query rows in chunks, rebuilt the full query×key score tile for each
chunk, masked out the half of it that causality forbids, exponentiated, and summed
down the columns. That is a second attention-sized computation, and unlike flash
attention it *materialises* the tile it is summing.

**Why you cannot simply fuse it into the forward pass.** This is the crux, and it
is worth a paragraph in any paper because it looks like a missing optimisation and
is not. Flash attention walks key blocks while keeping, for each query row, a
running maximum and running sum, and corrects its running estimate whenever a
larger value appears. That correction factor is *indexed by query row*. Our score
is a sum **across** query rows. You cannot pull a per-row factor out of a sum over
rows after the rows have been collapsed away — the thing you would need to correct
with is specific to each row, and those rows no longer exist. The only way to
apply it correctly is to keep the whole un-summed tile until every row has its
final normaliser, which is precisely the tile flash attention exists to avoid.
**The two quantities need their correction applied at different points, and one of
those points is gone.**

**The fix.** Flash attention's backward pass, used only in training, accumulates
for each key block a contribution summed over every query block — because the
gradient with respect to the values has exactly that shape. Substituting a vector
of ones for the real output gradient degenerates that gradient matmul into a plain,
unweighted column sum. That column sum *is* the eviction score.

**Why the second pass is nearly free.** The naive version of this pass would still
need the running-maximum machinery, because the true normaliser for a query is not
known until every key block has been seen. But by the time this pass runs, the
forward pass has already finished and already knows each query's exact log-sum-exp.
Handing that value over turns the softmax probability into one direct formula with
no running maximum, no rescaling, and no overflow risk — the exponent is always
negative by construction. The only cost to the forward pass is keeping one small
number per query row instead of discarding it: a few megabytes, not a second
attention.

**The loop order is the other half.** Flash's forward loops outer over query
blocks; this kernel loops **outer over key blocks**, the same way the backward
pass does, because the score being accumulated belongs to keys. Each program owns
one key block, loads it into on-chip memory once, and reuses it across every query
block and every query head sharing it. Different programs write to disjoint slices
of the output, so there are no atomics and no race conditions — and the key block
is read once rather than re-read per query head, which is what the chunked loop it
replaced effectively did.

**Skipping work that is known to be empty.** Under causal masking, a tile of
(query block, key block) is in one of three states: entirely visible, entirely
invisible, or straddling the diagonal. Roughly half of all tiles are entirely
invisible. The old code computed them and then masked the result; the kernel checks
before doing any matmul and skips them outright. Only one tile per query block ever
needs the compute-then-mask treatment.

**The bottleneck that was found by arithmetic, not by profiling.** Once shipped,
the kernel was still about 4× less efficient than the attention pass running beside
it. Instead of profiling blind, the three plausible costs — matmul FLOPs, memory
traffic, and the exponentials — were each sized by hand from the kernel's actual
shapes. Two were small; the exponentials accounted for nearly all of the measured
time. The cause is a hardware detail: a natural exponential costs roughly ten
special-function-unit instructions, a base-2 exponential costs one. Folding a
constant into the softmax scale **once, at launch** makes every exponential in the
kernel base-2 for free, changing nothing per element. That moved the kernel from a
transcendental-bound cost toward its matmul-and-memory floor — a 4–6× reduction on
the term that dominated. **The transferable lesson is to size each candidate cost
before measuring, because the answer named the fix.**

### A2.2 Kernel 2 — fused two-tier decode, with nothing written back

**What it replaced.** For each compressed window: look it up, expand the 2-bit
codes to 16-bit, apply the rotary encoding, concatenate every expanded window with
the full-precision tier into one contiguous tensor, hand that to the standard
attention path — and then run a *second* query-key product to recompute the scores.
Four costs per layer per token: the 16-bit write-back, the concatenation, a
full-budget 16-bit read, and a duplicate score computation.

**Why that is worse than it sounds.** Generating one token at a time is
bandwidth-bound, not compute-bound. The GPU streams almost the entire model's
weights for every single token, and the arithmetic done per byte read is tiny by
comparison. Under that constraint, expanding the compressed cache and writing the
copy to memory — even briefly, even only to re-read it immediately — is pure waste:
**those bytes never needed to exist in memory at all.**

**The fused design.** One kernel reads a window's packed codes directly, expands
them in registers using that window's fixed scale and offset, applies the rotary
rotation in registers using the window's original absolute position, and runs the
query-key product and the weighted accumulation — all before any expanded value
touches memory. The only bytes that cross memory are the small codes and their
compact per-window grid.

**Why the score is free here.** The prefill kernel exists because a score summed
over many query rows cannot be recovered from a running softmax. At decode there is
only *one* query row, so there is nothing to sum over: the per-key softmax weight
**is** the importance score. It falls out of the same product the output needed.
There is no second pass, no saved normaliser, and no separate kernel.

**Rotation without a gather.** The rotary encoding is normally applied by building
a rotated copy of the key. Instead the kernel loads the two halves of the head
dimension separately and combines them with the cosine and sine directly — pure
register arithmetic, no shuffle, no temporary. The angles are fixed for a window's
whole lifetime, because eviction never renumbers positions, so they are computed
once and reused.

**Tiling that matches the data's own structure.** A tile is sized to whole windows.
That is not cosmetic: a window is also the unit that owns one scale and offset, so
a tile that never straddles a window boundary needs no bookkeeping across tiles.
Widening the tile from one window per serial step to several windows per step let
the kernel also emit the **per-window attention totals** the eviction step needs,
from a register epilogue, instead of a separate traversal of the whole sequence.

That change is worth more than it looks. The traffic for scoring goes from
proportional to sequence length to proportional to window count, and the host-side
reduction that followed it — pad, reshape, sum, window-reduce, concatenate, gather
— collapses to a **single gather**. Measured, it was worth about 11 ms per step,
almost identically at every shape and every batch size. *That uniformity is itself
the evidence*: a saving that does not move with how much data there is was never a
traffic saving. It was a fixed per-step cost, and the two earlier estimates of its
value were both wrong because both were derived from traffic alone.

**A second consequence of whole-window tiling** is that the window size stops being
constrained. An earlier design concluded that the tile width had to divide evenly
into the window size, which forces a power of two. Tiling in whole windows, padded
up to a power-of-two block, removes the constraint entirely: a window never
straddles a tile for *any* size, and the only cost is unused lanes — full
utilisation at powers of two, 94% at a window size of 12. Window sizes from 1 to
128 all work.

**The hardware limit it hit, and the fix.** Widening the tile also multiplies how
much operand data the matrix-multiply units must stage on-chip at once. The first
wide version needed about 128 KB of staged operands against an A100's ~163 KB per
streaming multiprocessor — and the compiler's default pipelining depth took that to
172 KB. Every cell of the first GPU run failed outright with an out-of-resources
error. The fix is a **fit ladder**: try the largest, fastest configuration first,
step down through a short list on failure, cache the winner per GPU and per shape
so the search runs once. Every rung is numerically identical, so this is a
resource-fitting search, not a correctness fallback — and if nothing fits it raises
with the whole ladder rather than silently degrading. The arithmetic that predicted
the failure was computable up front and had not been computed: **a tile-size change
is a shared-memory change.**

**The accuracy bug it exposed.** One shared helper built the rotary lookup tables
at full precision while every other rotation in the system used half precision —
nothing had depended on that value's precision before, so it went unnoticed. The
effect: a compressed window was un-rotated in half precision and re-rotated in
full, so **a token's key depended on which tier happened to be holding it.**
Matching the precision cut the worst round-trip error by roughly **660×** and
recovered a long-document benchmark regression. No equivalence test could have
caught it — every one of them compared the kernel against an oracle built with the
*same* convention.

### A2.3 Kernel 3 — one eviction pass instead of thirty-two

**What it replaced.** The eviction body ran once per layer. At thirty-two layers,
an evicting step issued thirty-two small, independent, identically-shaped
sequences of operations, one after another.

**The observation that makes it collapsible.** Every layer evicts on the same step,
and every shape involved — window size, budget, how many windows go to each tier —
comes from the configuration rather than from the data. So those thirty-two passes
are not merely similar; they are shape-identical. And every operation in the
eviction body already treats rows independently. Therefore a layer can be folded
into the row axis: with thirty-two layers and a batch of `B`, the joint store has
`32·B` rows, layer *i* owning a contiguous block of them. **The body did not change
at all — only its row count.**

**Why this is worth doing when it moves no bytes.** A GPU launch carries a largely
fixed dispatch cost independent of how much work it does. Thirty-two small
identical launches pay that cost thirty-two times and gain no parallelism for it,
because the work was independent to begin with. Folding them into one pays it once.
The measured effect on dispatched operations: the eviction body went from **13,824
to 426** (32.5×), and amortised across every generated token from **5,592 to
1,177** (4.75×).

**The ordering flip this forced, and why it is safe.** With one store spanning
every layer, eviction must run **before any layer appends this step's token** —
when the first layer is ready to compact, the later layers have not produced this
step's token yet. So "append, then compact" became "compact, then append". The
safety argument is short and is the whole justification: which windows survive is
decided **only** from already-accumulated scores; this step's token has no score
yet and lands in the most-recent region, which is always kept at full precision and
is never eligible for eviction. Therefore the surviving set cannot depend on
whether the token has been appended. That was not merely argued — the batched path
is pinned byte-for-byte identical to the per-layer one across tier ratios, layer
counts, batch sizes, every window size from 1 to 128, with and without sink tokens,
and through promotion and dormancy.

**What was deliberately *not* folded.** The first compaction, during prompt
processing, stays layer-by-layer. Folding it would require holding the whole
thirty-two-layer prompt buffer and the whole thirty-two-layer steady store
simultaneously — several gigabytes of peak memory, taken straight out of the
maximum batch size. Joining *after* the first eviction, once the per-layer prompt
buffers have been released, costs nothing.

**The compiled follow-on, and where its win actually is.** The eviction body was
later compiled into a single fused graph. Two things had to be fixed first: the
quantiser had been excluded from tracing to dodge a compiler bug since fixed
upstream — so the graph traced and then fused *nothing*, landing as sixteen
separate multi-millisecond elementwise kernels, about 18% of GPU time — and the
compiler does not, by default, reproduce an intermediate precision rounding step
that the storage format depends on for correctness. Both fixed, the body traces as
one graph with no breaks.

Measured against its own control arm — the identical run with compilation disabled
— it is worth about **10% of per-token time**, in the same direction at every
shape and batch. The instructive part is *where*: its GPU kernels cost roughly
**1 ms/step more** than the eager work they replace, and it wins anyway, on the
host, from 36 fewer launches per step. **The gain is in launches, not arithmetic**
— the same lesson as the batching itself, arriving by a different route.

---

## A3. Algorithms and maths

### Notation

`B` batch, `L` layers, `H_kv`/`H_q` key/query heads, `rep = H_q/H_kv` the group
size, `D` head dimension, `T` query length, `S` key length, `ws` window size.

### Kernel 1 — prefill eviction score

The identity, with the output gradient replaced by ones:

$$
dV_j=\sum_i P_i^{\top}dO_i
\;\xrightarrow{\;dO=\mathbf 1\;}\;
\sum_i \mathbf 1^{\top}P_i=\sum_i P[i,\cdot]=\text{score}_j
$$

and, since the forward has already produced the exact normaliser
`L_i = m_i + log ℓ_i`,

$$
P[i,s]=\exp\big(S[i,s]-L_i\big),\qquad
S[i,s]\le L_i \;\;\Rightarrow\;\; \text{no running max, no rescale, no overflow}
$$

```
Stage A — ordinary flash forward, one line changed:
    O, L ← flash_forward(Q, K, V)          # L is KEPT, not discarded

Stage B — the score kernel, one program per (key block, batch, head):
    load K_j into on-chip memory                    # once; reused below
    acc ← 0
    for each query block i:
        if block i is entirely before block j:  skip      # ~half of all tiles
        for each query head sharing this key head:
            S ← scale · Q_i · K_jᵀ
            P ← exp2(S − L_i)                            # base-2: 1 SFU op, not ~10
            if the blocks straddle the diagonal: mask this tile only
            acc ← acc + colsum(P)
    score[batch, head, block j] ← acc                    # disjoint writes, no atomics
```

**Cost model.** One key-block load per key block, reused over every query block,
plus `⌈T/BLOCK_M⌉` tiles of shape `[BLOCK_M, BLOCK_N]`. The full `[B, H_q, T, S]`
probability tensor — an `O(T·S)` write — is never materialised. The base-2 swap
divides the per-element transcendental cost by ~10 and was measured to cut the
dominant term 4–6×.

### Kernel 2 — fused two-tier decode

For a single query row, the softmax weight and the importance score are the same
quantity:

$$
\text{score}_s \;=\; p_s \;=\;
\frac{\exp(\text{scale}\cdot q\cdot k_s)}{\sum_{s'}\exp(\text{scale}\cdot q\cdot k_{s'})}
$$

so no second pass exists to remove — it was never needed.

```
one program per (batch, key head); the whole query-head group rides along
    m, ℓ, acc ← −∞, 0, 0
    for each tile of the full-precision tier:              # already rotated
        s ← scale · q·kᵀ
        m, ℓ, acc ← online_softmax(m, ℓ, acc, s, v)
    for each tile of WHOLE compressed windows:
        k ← expand_2bit_in_registers(codes, scale, offset)
        k ← rotate_in_registers(k, cos, sin at the window's fixed position)
        s ← scale · q·kᵀ
        m, ℓ, acc ← online_softmax(m, ℓ, acc, s, v)
        per-window running total and running max            # scores, in-register
    epilogue: rescale the per-window totals by (max − lse);  out ← acc / ℓ
```

with the online update in base 2 throughout:

$$
m \leftarrow \max(m,\ \text{rowmax}(s)),\qquad
\ell \leftarrow 2^{\,m_{\text{old}}-m}\,\ell+\text{rowsum}(\tilde p),\qquad
\text{acc}\leftarrow 2^{\,m_{\text{old}}-m}\,\text{acc}+\tilde p\cdot v
$$

**Cost model.** Decode has an arithmetic intensity of roughly 2 FLOP/byte, far
below the machine's ridge point, so wall-clock tracks bytes moved almost directly.
A materialise-then-attend design pays for the codes **and** a full 16-bit
write-back of the expanded tier — about 4× the compressed bytes — on every step.
Fusing expansion, rotation, and the dot product removes that round trip entirely:
the only bytes crossing memory are the codes and their grid. Scoring traffic goes
from `O(S)` to `O(W)` where `W` is the window count.

**Shared-memory budget** (the arithmetic that should have been done first):

$$
\text{staged bytes}\;\approx\;2\cdot\tfrac{D}{2}\cdot \text{BLOCK\_T}\cdot 4
\;+\;\text{BLOCK\_T}\cdot D\cdot 4,
\qquad\text{then}\;\times\;\text{num\_stages}
$$

At `BLOCK_T = 64, D = 128` that is ~128 KB before pipelining, against ~163 KB per
SM — and the default pipelining depth took it to the observed 172 KB.

### Kernel 3 — batched cross-layer eviction

```
R ← L · B                     # layer i owns rows [i·B, (i+1)·B) of one joint store

at each decode step t:
    if this step evicts:
        run the UNCHANGED eviction body once, over R rows      # not L times over B
    then every layer appends token t                            # order FLIPPED
```

**Safety lemma.** The retained set is a function of the accumulated window scores
alone. Token `t` has no score column at step `t`, and it lies in the most-recent
region, which is always full precision and never evictable. Therefore the retained
set is invariant to whether token `t` has been appended, and "compact then append"
is equivalent to "append then compact".

**Cost model.** With a per-launch dispatch cost `c`, the eviction body pays `L·c`
before and `c` after. Bytes moved — proportional to `L·B·T·D` — are **unchanged**.
This is purely a launch-count argument. Measured: 13,824 → 426 dispatched
operations at 32 layers, and 5,592 → 1,177 amortised per generated token.

### Why the three compose

They act on disjoint phases and on disjoint bottlenecks:

| kernel | phase | bottleneck removed |
|---|---|---|
| 1 — backward-pass score | prompt processing | an `O(T·S)` redundant pass, then special-function throughput |
| 2 — fused decode | every generated token | one full memory round trip per layer per token |
| 3 — batched eviction | every eviction step | fixed per-launch dispatch cost, ×32 |

Nothing in the list overlaps, which is why the measured gains multiply rather than
cancel: **2.81× on time per generated token**, from 2.05× slower than an
uncompressed cache to 1.37× faster, at unchanged peak memory.

---
---

# Part B — This branch: the card and the read gate

## B1. At a glance

**1. Stop reading the compressed tier to find out you did not need it.**
Every generated token expanded **every** compressed window, so it could be attended
over and could accumulate a score. But the compressed tier is by construction the
band ranked *below* the full-precision tier — most of those windows contribute
almost nothing, and we were paying full price to discover that they were cheap.
This branch writes a small **card** for each window, once, at the moment the window
is compressed. The decode step scans every card, picks the windows worth opening,
and expands only those — a quarter of the tier at the shipped setting. Scanning a
card costs two dot products.

**2. The windows it skips still take part in the answer.**
This is the strongest claim on the branch and it is easy to miss. A gate that reads
a quarter of the tier and returns that softmax unchanged is not an approximation of
attention over the cache — it is **exact attention over a different cache**, one
where the other three quarters do not exist and the surviving weights have been
inflated to cover for them. So each card also stores the window's **average
value vector**, and a skipped window now contributes that average at the weight its
own card estimated, calibrated against the windows that *were* opened; the output
is then renormalised to account for the mass added back. It rides on work the
kernel was already doing — one extra matrix product on a tile already in registers,
no new launch and no new pass over the tier.

**3. The two halves of the card answer different questions, so they are built
differently.**
The key half decides **whether** a window can matter, so it needs to find outliers:
the card models a window as a *line* rather than a point — an average plus a
direction plus one number per token saying how far along that direction the token
sits — and the direction points at the token **farthest from the average**. A plain
average would hide the one hot token, which is exactly the case the gate exists to
catch: exponentials are convex, so an average always understates a window that
mixes one large value with several small ones. The value half only answers **what**
a window contributes once we have already decided it does not matter much — and
inside a window we chose not to open, the attention weights are near-uniform by
construction, so a plain average is the correct first-order answer there.
*Outlier model for keys, average for values, with a stated reason on each side.*

**4. The finding that governs the whole idea — a negative result with a clean
explanation.**
A card must be read for **every** window on **every** step; a window is only opened
if selected. So the gate costs *one card plus the fraction of windows you open*,
against *every window*. Break-even is therefore `1 − card/window`, and that single
line is the whole economics: **the card's width, not the gate's cleverness, decides
whether the feature pays.** At the shipped geometry a window's token data is 804
bytes per head and the card is 272, so break-even sits at a **0.66 read ratio** and
the shipped quarter costs 59% of an ungated read — which is a saving of only about
**3% of what the decode kernel actually moves**, because the compressed tier is
just ~13% of the read and the full-precision tier is the other 87%.

**Measured, reading everything ties with reading a quarter — to within 0.2–1.6% of
time per token.** Selection currently buys nothing measurable, and it is not free:
looking each window's address up at runtime turns every field into a scattered
fetch, and the gated kernel runs at **12× off its own bandwidth roofline** where
the model's matrix math runs at 1.4×.

**The cause is the memory layout, not the idea.** The card has already been
narrowed once — the two average vectors went from 8-bit to 4-bit, taking it from
400 to 272 bytes per head and break-even from 0.50 to 0.66, at no measurable
accuracy cost. What is left is making the selected reads **contiguous**: a window's
data currently lives in six separate tensors plus four card tensors, so one
selected window is ten scattered fetches.

Two further changes are logic rather than method, but they decide whether any of
the numbers above mean anything:

**5. One production path, and no ungated arm.** Several environment switches could
route a production run into different code, and none of them appeared in the output
— a number produced under the wrong one still looks like a number. They are gone;
the device decides. The gate is now the **only** decode read path wherever a
compressed tier exists: a layer that arrives without cards **raises**, rather than
reading the whole tier correctly and silently. Asking for a full read is a setting
(ratio = 1.0), which selects every window through the same code and is therefore a
*provable* no-op — and the control arm.

**6. The compiled cache is bit-identical to the eager one, and the budget now
prices what is actually held.** The compiler does not reproduce an intermediate
precision rounding step that the storage format depends on, so three quantisers
diverged from eager once the eviction body was compiled. Separately, the byte
accounting counted codes and scales but not the cards — which are resident for the
life of the window — so every reported memory budget **understated the compressed
tier by 38%**. Both fixed, and the budget resolver now *enforces* that the leftover
is too small to buy another window of either tier.

---

## B2. How it works

### B2.1 The card, and what it costs

Written once, at the moment a window is compressed, from the **already-rotated**
keys the eviction body is holding one line before it strips their rotation — so the
card costs no extra rotation work. It is frozen for the window's life and
reactivated if the window is compressed again after a spell at full precision, for
the same reason the codes are: eviction never renumbers positions, so a statistic
taken at compression time stays valid forever.

Four fields per window per key head, 272 bytes in total:

| field | length | precision | bytes | what it is |
|---|---|---|---|---|
| average key | `D` | **4-bit**, 2 per byte + a scale | 66 | the window's mean key, stored as a difference from a frozen reference |
| direction | `D` | 8-bit + a scale | 130 | unit vector toward the farthest-out token |
| projections | `ws` | 8-bit + a scale | 10 | how far each token sits along that direction |
| average value | `D` | **4-bit**, 2 per byte + a scale | 66 | the window's mean value, as a difference from its own reference |

The 4-bit/8-bit split is measured, not stylistic. **The direction must stay 8-bit:**
at 4 bits the hot token stops being the best-fit token in its window, which is the
single property the card exists for. (A 1-bit sign vector is catastrophic — the hot
token ends up with the *largest* reconstruction error in its window, 38.0 against a
window average of 15.0; 4-bit gives 7.0; 8-bit gives 0.39 against 9.2.) The two
averages need no such range, and coarsening them is accuracy-neutral: over 8 seeds,
median relative output error at the shipped ratio is **0.00004 for both encodings**
— identical to five decimals — and the two encodings choose the same windows
98.5–98.8% of the time. The per-token projections are 10 bytes and cannot shrink
either: without them the card is a plain average again, which is the exact failure
it was designed to avoid.

**Storing averages as differences from a frozen reference** is what makes 4 bits
enough. Certain channels in these models carry massive activations that are nearly
identical across *every* window — so they separate nothing while consuming the
entire numeric range. Subtracting a per-layer, per-head reference first cut the
average's reconstruction error **17×**. The reference is frozen at the first
compression batch, not running: a card is written once and never revisited, so a
later change to the reference would silently reinterpret every card already stored.

**The guarantee that was designed, measured, and retired.** The card originally
carried a fifth field — a per-token residual — which turned the estimate into a
**provable upper bound** via Cauchy–Schwarz: a window whose bound fell below the
bar provably carried no attention logit above it, so the gate could over-select but
could not miss. It was removed because **nothing on the production path ever read
it.** The bound exists to serve a threshold rule; the shipped gate is a top-*k* on
the point estimate, with no threshold. And ranking on the estimate is not a
weakening: the bound carries a per-window slack term, which lets a loosely-bounded
window outrank a tighter one carrying a *higher* true logit. Measured, the estimate
holds **100.0%** of the worst head's attention mass at a 0.15 ratio against the
bound's **99.7%** — and it costs less, because the ranking path never touches the
residual field. It was removed rather than zero-filled, because a zero residual is
not a weaker bound; it is not a bound at all, and leaving the column would invite a
reader to think one was available.

For a paper this is a clean story in its own right: *a provable guarantee was
designed, measured against the cheaper heuristic it was meant to protect, found to
be both looser and more expensive in the only regime that ships, and retired.*

### B2.2 The gate kernel, and where selection had to live

**Where it runs, and why there is no choice.** Selecting windows needs the
**query**. The cache interface that the model calls is handed keys and values only
— that is the shape of the library's API, not a decision this project can revisit.
So the cache hands over the full-precision tier plus the cards, and selection runs
one layer later, inside the attention call, which is exactly where the fused decode
path already patches in. Worth stating in a paper as a structural constraint: *the
tier decision is query-independent and happens at eviction; the read decision is
query-dependent and must happen inside attention.*

**One program per (batch, key head)** — the same grid as the decode kernel, so the
group union is free. Card vectors stream through registers and only three scalars
per window leave the loop, so the on-chip footprint is proportional to the tile
width, not to the number of windows times the head dimension.

It is a **separate** kernel rather than a prologue inside the decode kernel, for a
concrete reason: the decode kernel's compressed-tier loop already stages ten large
tiles, and the tile-fit ladder **falls back silently** when the budget is exceeded.
A rung lost to a prologue would cost more than the gate saves and would produce no
error at all. Land it standalone, price the extra launch, then fold it in with the
rung pinned.

**Two selection rules that measurement forced.** Both are easy to get wrong and
both change the answer by a lot:

*The cap must bind on the key head, after the group union.* One program owns a key
head and every query head sharing it, so a window is loaded once for the whole
group — the key head is the unit of work. If each query head picks its own windows
and the picks are unioned afterwards, a 25% cap can become **100% loaded** at a
group size of 4, and the gate buys nothing. So the group's estimates are combined
first, then the cap is applied.

*Selection is per (row, key head), not per row.* Measured on a 271-window tier with
8 key heads and 32 query heads, at a 0.25 ratio:

| granularity | windows read | worst head's mass recalled |
|---|---|---|
| per (row, key head) | 68 / 271 | **100.00%** |
| per row, union of the heads' picks | 240 / 271 | 100.00% |
| per row, top-k of the row's best | 68 / 271 | 97.54% |

The union reads 89% of the tier — no saving at all. A row-level top-*k* gives up
2.5% of the worst head's mass for identical traffic. Per-head selection costs only
that the compressed positions become per-head; the counts stay equal across rows
and heads, so the result is still dense, with no padding and no keep-mask.

**How the selection reaches the decode kernel.** As an indirection, not as a
different tensor: the compressed tier stays whole and gathered, and the decode
kernel's inner loop dereferences a selection list. **The gate declines to read
bytes rather than shrinking the store**, so the bytes it skips are never moved and
nothing is repacked. The gated and ungated kernels are one kernel with a
compile-time flag, so the ungated path is not perturbed by the feature's existence.

### B2.3 Representatives attend — the epilogue that closes the gap

The kernel's epilogue already computed, for every window, the card's estimate
rescaled by the deviation the opened windows give for free. It then used that only
for the eviction score. It now also uses it for the **output**.

The gap it closes is real and structural. Renormalising a softmax over a selected
subset does not approximate attention over the whole cache; it computes a
different, self-consistent distribution in which the unselected mass has been
redistributed onto the survivors. Adding each skipped window's average value back,
at the weight its card estimated, restores the missing mass to the right place at
first order, and the result is renormalised by the total mass added.

The cost is small by construction: the epilogue's pass over the window columns is
unchanged, the only new traffic is the average-value tile — 66 bytes per head
against the 804-byte window *not* read — and the only new arithmetic is one matrix
product on a tile the loop already holds, the same shape as the one the compressed
loop was already doing.

**And the scores still have to be filled**, for the reason they always did: scores
are what eviction ranks on, so a skipped window scoring zero would rank last and be
evicted — a change meant only to *read* the tier less often would silently
*destroy* it, and at the shipped tier split that is 70% of the evictable cache.
Measured against truth, the rescaled fill gives rank correlation **0.996** and
recovers **99.5%** of total mass, against **0.32** correlation if skipped windows
are left at zero.

**Why cards alone cannot replace opening windows.** Opening nothing is not a
speed/accuracy trade — it is a 0.908 relative error. The calibration factor is
*measured on the windows that were opened*, so with nothing opened there is nothing
to calibrate against. The cards are a selector with a first-order correction, not a
substitute for reading.

### B2.4 The byte accounting, restated honestly

Two format changes moved every figure, and one accounting bug meant the budget had
never priced what was held.

**The compressed tier's scale grid is one byte per entry, not two.** The 16-bit
scale-and-offset grid was 48.5% of a window's codes-plus-grid — two bytes of
precision wrapping a two-bit code. It is now 8-bit against a 16-bit scale shared by
32 consecutive entries. This is nearly free because the codes are fitted to the
*stored* grid: a rounded scale merely repositions the four levels and the codes
re-fit against them, so the 2-bit error (~13% on keys) swamps the grid's own
rounding. Over 12 seeds:

| grid | key error | query·key error | bytes/head |
|---|---|---|---|
| 16-bit (was) | 0.13348 | 0.13541 | 512 |
| 8-bit, one scale per head | 0.13370 | 0.13568 | 260 |
| **8-bit, groups of 32** | **0.13368** | **0.13562** | **272** |

**The grouping is what makes it safe, and aggregate error cannot see why.** On keys
whose channel gains span a factor of a thousand — which is what a massive-activation
channel *is* — a single scale per head leaves the worst channel at **1.082**
relative error (worse than returning zeros) while the Frobenius norm moves 0.14%
and reports nothing at all. Groups of 32 put that channel back at 0.306 — the
16-bit grid's own figure — for 12 bytes per head. *Aggregate error is the wrong
instrument for a per-channel failure*, which is a transferable methodological point.

**The card is part of the window, and the budget now says so.** A card is resident
for the window's life and allocated wherever a compressed tier exists, so excluding
it understated that tier by **38%** — on the exact tier the memory claim is made
about. Two floor divisions also left up to one window of each tier unbought and
never spent; the resolver now tops up one window at a time, each time choosing the
tier that keeps the realised byte split closest to the requested one, and then
**enforces** the result: the leftover must be too small to buy another window of
either tier, or it raises. Granularity-exact rather than a flat percentage, because
a percentage is the wrong shape — one full-precision window is 15% of a six-window
budget and 0.4% of a 263-window one.

| | before this branch | **now** |
|---|---|---|
| compressed window, codes + grid | 8448 B | **6432 B** (−23.9%) |
| card | not charged | **2176 B** |
| window as charged to the budget | 8448 B | **8608 B** |
| price ratio, full precision : compressed | 3.88 | **3.81** |
| windows bought at a fixed budget | — | **+21% from the grid change alone** |

### B2.5 Why it does not pay yet, in one number

The gate's whole premise is that a summary is cheaper to read than the thing it
summarises. **A summary is only cheap if it is coarser than what it summarises —
and here what it summarises was already crushed to 2 bits per number.** At 8-bit
encoding the card was 400 bytes against 804 bytes of token data: **78% of what it
replaces.** The summary was written at four times the precision of the thing
summarised. Going to 4 bits on the two averages brought that to 34%, which is what
moved break-even from 0.50 to 0.66.

Two costs the gate adds on top, both measured:

* **Scattered reads.** Ungated, the kernel walks windows in order and the hardware
  fetches them in long contiguous runs. Gated, each window's address is resolved at
  runtime, so every field becomes a scattered fetch — **12× off the kernel's own
  bandwidth roofline**, where the model's matrix math runs at 1.4× its roofline.
* **An extra full-tier pass.** The score-fill epilogue walks *every* window, not
  the selected ones, plus a prologue that seeds every column.

Net: save ~3% of the kernel's traffic; pay for scattered reads on the rest plus two
extra full-tier passes. **The remedy is contiguity, not a weaker gate.**

There is a second, sharper result in the same data, and a paper should lead its
latency discussion with it: **decode time barely tracks cache size at all.** A
configuration holding **2.5× more keys and 2.9× more compressed windows** ran
**22% faster**. If per-window decompression dominated, that could not happen. What
dominates is paid once per step per layer, and it moves only 1.4× across a 32×
batch range.

---

## B3. Algorithms and maths

### The card

For a window with post-rotation keys `{k_i}` and values `{v_i}`, and frozen
per-head references `a_K`, `a_V`:

$$
\mu=\tfrac{1}{ws}\sum_i k_i,\qquad
i^\star=\arg\max_i\|k_i-\mu\|,\qquad
v=\frac{k_{i^\star}-\mu}{\|k_{i^\star}-\mu\|},\qquad
t_i=\langle k_i-\mu,\,v\rangle,\qquad
\bar v=\tfrac{1}{ws}\sum_i v_i
$$

stored as `4-bit(μ − a_K)`, `8-bit(v)`, `8-bit(t)`, `4-bit(v̄ − a_V)` with one
16-bit scale each. This is **greedy *k*-center with one center**, not Lloyd's: the
selector wants the window's extreme point, and *k*-center minimises exactly the
radius a max-style estimate depends on. One pass, fixed instruction count, no
convergence loop.

### Reading a card

With `m = ⟨q, a_K⟩ + ⟨q, μ − a_K⟩` and `g = ⟨q, v⟩` — **two dot products of length
`D`** — every per-token estimate in the window follows without touching the tokens:

$$
x_i=\text{scale}\cdot\big(m+t_i\,g\big),
\qquad
\underbrace{\max_i x_i}_{\text{what selection ranks on}},
\qquad
\underbrace{\log\sum_i e^{x_i}}_{\text{what a skipped window contributes}}
$$

Both live in the log domain, so they are comparable across windows and cannot
overflow. The projections are 10 bytes and ride in the cache line the vectors
already pulled in, so all of this comes out of the same two dots.

*Retired, and reported as such:* with a per-token residual `ε_i ≥ ‖k_i − k̂_i‖`
measured against the **decoded** card and rounded **up**, Cauchy–Schwarz gives
`|q·k_i − (m + t_i g)| ≤ ‖q‖ ε_i`, hence a provable upper bound on every logit in
the window. Nothing consumed it and the slack-free estimate ranks better, so it is
gone.

### Selection

$$
\hat x[b,h,w]=\max_{r\in\text{group}(h)} x[b,h,r,w],
\qquad
n_{\text{sel}}=\max\big(1,\lceil r\cdot n_{\text{active}}\rceil\big),
\qquad
\text{keep}(w)=\big[\,\text{rank}_{\hat x}(w)<n_{\text{sel}}\,\big]
$$

The group reduction **must** precede the cap; reversing them admits up to
`r × group_size` windows, i.e. the whole tier at a group size of 4. The cap
resolves against the **live** active count rather than a fixed integer, because the
tier size moves with the shape — 115 windows at a 1024-token prompt, 271 at 4096 —
so a pinned count would not be the same fraction anywhere.

### What a skipped window contributes

Let `R` be the opened set, `e_w` the card's estimated mass, `s_w` the real score:

$$
c=\frac{\sum_{w\in R}s_w}{\sum_{w\in R}e_w},
\qquad
f_w=c\,e_w\ \ (w\notin R)
$$

and both outputs complete from the same `f`:

$$
\text{score}_w=\begin{cases}s_w & w\in R\\[2pt] f_w & w\notin R\end{cases}
\qquad\qquad
\text{out}=\frac{\displaystyle\sum_{w\in R}\sum_{i\in w}p_i v_i \;+\; \sum_{w\notin R}f_w\,\bar v_w}
{1+\displaystyle\sum_{w\notin R}f_w}
$$

The calibration factor costs one reduction and needs nothing the step did not
already compute: the opened windows carry **both** a real and an estimated score,
so it is free and self-calibrating. Exponentiation is taken relative to each head's
own maximum — the offset cancels in the ratio, so it is an overflow guard, not an
approximation. In the kernel the normalisation collapses to a max and a sum, so no
logarithm is taken and every exponential stays base 2.

### The governing law

Per window per step, in bytes moved:

$$
C_{\text{gated}}(r)=\underbrace{B_{\text{card}}}_{\text{every window}}
\;+\;r\cdot\underbrace{B_{\text{win}}}_{\text{opened only}},
\qquad
C_{\text{ungated}}=B_{\text{win}}
\qquad\Longrightarrow\qquad
\boxed{\;r^\star=1-\frac{B_{\text{card}}}{B_{\text{win}}}\;}
$$

At the shipped geometry, `B_win = 804` bytes per head (512 of codes + 292 of grid)
and `B_card = 272`, so `r* = 0.66` and the shipped `r = 0.25` costs **58.8%** of an
ungated read. Under the previous 8-bit card (`B_card = 400`), `r* = 0.50`.

Two corollaries worth stating explicitly:

1. **A gate over a compressed tier is bounded by its own summary's compression
   ratio.** The tighter the underlying storage, the narrower the card must be for
   selection to pay *at all*. At 2 bits per number, an 8-bit summary made of three
   full-length vectors cannot win.
2. **The saving is diluted by whatever the gate does not govern.** Here the gated
   tier is ~13% of the decode read and the full-precision tier is 87%, so a 41%
   saving on the gated part is ~3% overall. Clearing break-even is necessary and
   nowhere near sufficient.

### One gated decode step

```
if the layer has no cards: RAISE
    # a compressed tier that cannot be selected over is a misconfiguration,
    # not a mode — reading it whole would be CORRECT and would look like success

1  estimate every window from its card                   # two dot products each
   combine the query-head group, then keep the top n_sel # union BEFORE the cap
2  expand and rotate ONLY the kept windows               # the point of all this
3  one fused launch: attend over [sink | full-precision body | kept windows],
   dereferencing the selection rather than repacking the store
4  full-precision windows: the usual contiguous reduce into per-window scores
5  epilogue, over every window column:
     c ← (real mass over kept) / (estimated mass over kept)
     skipped w:  score_w ← c·e_w ;  out ← out + c·e_w · (that window's average value)
     out, scores ← (out, scores) / (1 + total added mass)
```

At ratio 1.0 this selects every window and is **byte-identical** to attending over
the whole tier — the no-op proof, and the statement that this is a *selection*
rather than a second implementation of attention.

### The compiled-eviction numerics fix

The storage format requires a round trip through 16-bit when fitting codes to their
scale, so that the grid the codes were fitted to is bit-identical to the grid every
later read uses. The compiler drops that round trip by default and keeps the wider
value in a register. Measured over 24 seeds: the key quantiser diverged from eager
on 11 seeds, the value quantiser on 14, and the card's encoder on **all 24**. The
fix restores the emulation around the one compile site, entered per call and
restored on exit, so nothing else the host compiles is affected. All rows go to
zero, and a multi-eviction end-to-end run is now identical tensor for tensor —
codes, scales, offsets, slot table, and every card field.

---
---

# Part C — How the core architecture changed

## C1. The one-sentence version

**Before, the system asked one question: what should the cache keep? Now it asks a
second, different one — what should this step actually read? — and the two have
different inputs, different frequencies, and different homes in the code.**

## C2. Side by side

| | **before this branch** | **now** |
|---|---|---|
| decisions in the system | one: which tier each window goes to | two: tier assignment **+** per-step read selection |
| what decides | accumulated attention scores | tier: the same scores; read: **the current query** |
| when | every few steps, at eviction | tier as before; read **every step** |
| where | inside the cache's update call | tier as before; read **inside the attention call**, because the cache API is never handed a query |
| compressed-tier read per step | the whole active tier | a fixed card scan plus a selected fraction of windows |
| frozen per-window data | codes, scale, offset, position range | **+ the card** |
| attention over the compressed tier | every window attends exactly | selected windows attend exactly; **skipped windows attend through their average value** |
| eviction scores | every entry exact | exact where opened; calibrated card estimate where not |
| selection granularity | n/a | per (row, key head); group union before the cap |
| compressed window price | 8448 B | **6432 B** of data, **8608 B** as charged with its card |
| budget honesty | codes and scales only | cards included; leftover **enforced** unspendable |
| production paths | three environment-variable forks | **one**; a card-less layer raises rather than reading the tier whole |
| compiled vs eager | equal under a shallow compile check; **not** under the real compiler | **bit-identical**, pinned by test |
| compiled eviction's worth | asserted | **measured against a control arm: ~10% of per-token time** |

## C3. What stayed identical — and why that matters

Every kernel in Part A is untouched in its contract, which is what makes the gate a
cleanly separable contribution:

* positions are still never renumbered, so cards and codes stay valid for life;
* a window's scale is still pinned at first compression, and re-compressing a
  window that was promoted back is still an exact reactivation, not a recompute;
* the tier split is still **rectangular** — equal counts across rows and heads — so
  the working tensor is still dense, with no padding and no keep-mask;
* the budget is still divided in **bytes**, so what the cache *keeps* at a given
  split is decided the same way (the prices moved; the rule did not);
* the decode kernel is still **one launch** per layer per step. The gate is an
  indirection into its inner loop and a compile-time flag on its epilogue, not a
  second kernel on the critical path.

So the gate changes **what is read**, never **what is retained** — except through
the eviction scores, which is exactly why the calibrated fill exists and why it is
the part of the design most worth defending in review.

## C4. The structural insight that only shows up in the comparison

Before, the compressed tier's cost was tied to its **size**. The whole economic
argument for compressing at all is that 2-bit storage buys roughly four times as
many retained windows at the same bytes — but the read path then paid for all of
them, every step, whether or not the query cared. **Compression bought retention and
handed the bill straight back to decode.**

The gate was designed to break that coupling, and structurally it does. Retention
still scales with the inverse of a window's price; bytes read per step become *a
card for every window plus a chosen fraction of full windows*. The two levers
separate, and the read fraction is a free parameter independent of the budget.

But the **measured** answer is that this has not yet converted into time, and the
reason is contained in the same two expressions: the card is 34% of a window, which
puts break-even at a two-thirds read ratio; the tier the gate governs is only ~13%
of what decode reads; and what the gate does save is currently eaten by the
scattered fetches that per-window selection introduces.

So the honest one-line comparison is: **the kernels made the cache fast; the gate
made holding more stop costing more to read *in principle*, and the work that
remains is making that true in bytes and in address order rather than only in
asymptotics.** That is a normal place for a method to be, and it is more useful in
a paper stated that way than as a claim.

## C5. Status, claim by claim

| claim | status |
|---|---|
| the three kernels' speedup: 2.81× per-token, from 2.05× slower to 1.37× faster than an uncompressed cache | **measured on GPU** |
| batched eviction is byte-identical to the per-layer path | **pinned by tests** across tier ratios, layer counts, batch sizes, every window size, sinks, promotion |
| the rotary-precision fix recovered the long-document regression | measured; the run bundles more than one change, so not a controlled attribution |
| card is 272 B against a 804 B window; break-even at a 0.66 read ratio | **arithmetic**, from the shipped byte formulas |
| the gate's selection is worth 0.2–1.6% of per-token time — i.e. a tie | **measured** against the read-everything control arm |
| the gated kernel is 12× off its roofline, and scattered reads are why | **measured** — GPU profile |
| 4-bit cards are accuracy-neutral at the shipped ratio | **measured on a CPU fixture**, 8 seeds |
| 4-bit cards are *faster* | **not measured.** Justified by bytes only |
| the compiled eviction is worth ~10% | **measured** against its own control arm |
| "the read gate is a 24% regression" | **not attributable** — a configuration change sits inside the same range. A floor, not a measurement |
| comparison against the two external compressed-cache baselines | **never run — neither is implemented.** Nothing here may be written as beating them |

Two framing points that belong in any write-up of these numbers:

* **Batch 1 is not winnable, and that is fine.** Every model weight is read once per
  generated token — about 16 GB, a ~10 ms/token floor on this GPU — and that floor
  is independent of cache size. At batch 1 the claim is **memory**, not latency.
* **Across the six shape-and-batch cells tested, the method beats an uncompressed
  flash-attention cache in one** (two at its best-ever operating point). The
  comfortable win is the long-prompt, large-batch cell — which is the cell the
  method is *for*. Saying that plainly is stronger than quoting the headline cell
  alone.
