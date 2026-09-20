# What is new here, and why it would matter in a paper

Three GPU kernels built on the two-tier int2 cache, then the per-window *card* and read
gate built on those. Each idea plainly first, then against what it replaced, then as
algorithm and maths.

**Not claimed here (it predates this work).** The sequence is cut into fixed-size windows;
every few steps each window's accumulated attention decides a three-way outcome — best stay
full precision, the next band is compressed to 2 bits against a scale fixed at first write,
the rest are dropped. Positions are never renumbered, and the budget is divided in **bytes**,
so the compressed tier buys ~4× as many windows per byte. The kernels accelerate that, under
one constraint: **none changes which tokens are kept.**

---

# Part A — The three kernels

## A1. At a glance

**1. The eviction score stopped being a second attention pass.** Deciding what to evict needs
each key's total received attention, summed over every query — so the code ran attention a
second time and summed columns. But flash attention's *training* code already computes a sum
over queries per key, because that is the gradient's shape; feed it ones and it produces the
score. **Nothing new was invented — an existing correct reduction was repurposed by changing
its input.**

**2. The compressed tier is never decompressed into memory.** Reading it meant expanding to
16-bit, writing that copy out, concatenating, attending, then attending *again* for scores.
One fused launch now expands and rotates **inside registers** and writes the expanded form
nowhere. Decode is bandwidth-bound, so **the memory compression saved was being handed back
every step.** And at one query row the softmax weight *is* the score, so scoring is free.

**3. Eviction stopped being thirty-two operations and became one.** Every layer evicts on the
same step at identical shapes, because the shapes come from configuration, not data — so
thirty-two identical launches each paid the fixed dispatch overhead for no parallelism, and
the body already treats rows independently. **A launch-count argument, not an arithmetic
one**: not one byte fewer moves.

**Result.** 8B model, 4096-token prompt, batch 32: time per generated token fell **2.81×**,
from **2.05× slower than an uncompressed flash cache to 1.37× faster**, peak memory unchanged.
They act on different phases and bottlenecks, so they compound.

## A2. How each works

### Kernel 1 — the score, from attention's backward pass

**Replaced:** a pass after flash attention returned that rebuilt the full query×key tile in
chunks, masked it, exponentiated and summed columns — a second attention-sized computation
that *materialises* the tile flash attention avoids.

**Why it cannot be fused into the forward pass.** Flash keeps a running maximum and sum per
query row, correcting whenever a larger value appears. That correction is *indexed by query
row*; our score sums **across** query rows, and you cannot pull a per-row factor out of a sum
over rows once collapsed. The two need their correction at different points; one is gone.

**The fix.** The backward pass accumulates, per key block, a contribution summed over every
query block; substituting ones for the gradient degenerates that matmul into a plain column
sum — the score. And since the forward already produced each query's exact log-sum-exp, the
second pass needs no running maximum, no rescale, no overflow guard. The only cost is keeping
one number per query row.

**Mechanics.** The loop runs **outer over key blocks**, because the score belongs to keys —
each program owns one key block and writes a disjoint slice, no atomics — and half of causal
tiles are entirely invisible, skipped before any matmul. Still ~4× less efficient than the
attention beside it, so the candidate costs were sized by hand: FLOPs and traffic were small,
the exponentials nearly all of it. A natural exponential costs ~10 special-function
instructions and a base-2 one costs one, so folding a constant into the softmax scale **once
at launch** took 4–6× off it.

### Kernel 2 — fused decode, with nothing written back

**Replaced:** expand each window to 16-bit, rotate, concatenate, attend — then a *second*
product for scores. Decode is bandwidth-bound, so writing an expanded copy out only to re-read
it is waste: **those bytes never needed to exist.**

**The design.** One kernel reads packed codes, expands them in registers against the window's
fixed scale and offset, rotates in registers at the window's original position, and
accumulates — all before any expanded value touches memory. Rotation needs no gather: the two
head-dimension halves load separately and combine with cosine and sine directly.

**Whole-window tiling, and what it bought.** A tile is sized to whole windows — also the unit
owning one scale and offset, so no cross-tile bookkeeping. Widening from one window per step
to several let the kernel emit the **per-window totals** eviction needs from a register
epilogue instead of a separate traversal: scoring traffic drops from proportional to sequence
length to proportional to window count, and a six-step host reduction collapses to **one
gather**. Measured ~11 ms/step, almost identically at every shape and batch — *and that
uniformity is the evidence*: a saving that does not move with data volume was never a traffic
saving, which is why two earlier traffic-derived estimates were wrong. It also lifts the
constraint that tile width divide the window size.

**The hardware limit.** Widening the tile multiplies the operands staged on-chip: ~128 KB
against an A100's ~163 KB per SM, and default pipelining took that to 172 KB — every cell of
the first GPU run failed. The fix is a **fit ladder**: fastest configuration first, step down
on failure, cache the winner. Every rung is numerically identical, so it is a resource search,
not a correctness fallback. **A tile-size change is a shared-memory change**, and that
arithmetic was computable up front.

**The accuracy bug it exposed.** One helper built the rotary tables at full precision while
every other rotation used half, so a compressed window was un-rotated in half and re-rotated
in full: **a token's key depended on which tier held it.** Matching the precision cut the
worst round-trip error ~**660×** and recovered a long-document regression. No equivalence test
could have caught it — all compared the kernel to an oracle using the *same* convention.

### Kernel 3 — one eviction pass instead of thirty-two

**Replaced:** the body ran once per layer, so an evicting step issued thirty-two small,
identically-shaped sequences back to back.

**Why it collapses.** The shapes come from configuration, so the passes are shape-identical,
and the body already treats rows independently — so a layer folds into the row axis, giving
`32·B` rows. **The body did not change at all, only its row count.** Dispatched operations:
eviction body **13,824 → 426** (32.5×), per generated token **5,592 → 1,177**.

**The ordering flip, and why it is safe.** With one store spanning every layer, eviction must
run **before any layer appends this step's token**, since the first layer cannot wait for the
later ones. Survival is decided **only** from already-accumulated scores, and
this step's token has no score yet and lands in the never-evictable most-recent region — so
the retained set cannot depend on whether it was appended. Pinned byte-for-byte against the
per-layer path across every configuration axis. The first compaction stays per-layer: folding
it would hold the whole prompt buffer and the steady store at once.

**The compiled follow-on.** The body was later compiled into one fused graph, after two
blockers: the quantiser had been excluded from tracing to dodge a compiler bug since fixed
upstream, so the graph traced and fused *nothing* (~18% of GPU time); and the compiler does
not reproduce an intermediate precision rounding step the storage format depends on. Measured
against its own control arm it is worth **~10% of per-token time**. The instructive part is
*where*: its GPU kernels cost ~**1 ms/step more** than the eager work they replace, and it
wins anyway on the host, from 36 fewer launches per step. **The gain is in launches, not
arithmetic.**

## A3. Algorithms and maths

### Kernel 1 — prefill score

With the gradient replaced by ones, and the forward's normaliser `L_i = m_i + log ℓ_i`:

$$
dV_j=\sum_i P_i^{\top}dO_i \;\xrightarrow{\,dO=\mathbf 1\,}\; \sum_i \mathbf 1^{\top}P_i=\text{score}_j,
\qquad
P[i,s]=\exp\big(S[i,s]-L_i\big)\le 1
$$

```
Stage A:  O, L ← flash_forward(Q, K, V)              # L is KEPT, not discarded
Stage B:  program per (key block j, batch, head):
            load K_j on-chip once; acc ← 0
            for each query block i:
                if i entirely before j: skip                 # ~half of all tiles
                for each query head sharing this key head:
                    P ← exp2(scale · Q_i·K_jᵀ − L_i)         # 1 SFU op, not ~10
                    mask only if the blocks straddle the diagonal
                    acc ← acc + colsum(P)
            score[batch, head, j] ← acc                      # disjoint writes
```

The `O(T·S)` probability tensor is never materialised.

### Kernel 2 — fused decode

At one query row, weight and score are one quantity: `score_s = softmax(scale·q·k)_s`.

```
program per (batch, key head); the whole query-head group rides along
    m, ℓ, acc ← −∞, 0, 0
    for each tile of the full-precision tier:              # already rotated
        m, ℓ, acc ← online_softmax(m, ℓ, acc, scale·q·kᵀ, v)
    for each tile of WHOLE compressed windows:
        k ← expand_2bit_in_registers(codes, scale, offset)
        k ← rotate_in_registers(k, cos, sin at the window's fixed position)
        m, ℓ, acc ← online_softmax(m, ℓ, acc, scale·q·kᵀ, v)
        accumulate per-window running total and max        # scores, in-register
    epilogue: rescale window totals by (max − lse);  out ← acc / ℓ
```

$$
m \leftarrow \max(m,\text{rowmax}(s)),\quad
\ell \leftarrow 2^{\,m_{\text{old}}-m}\ell+\text{rowsum}(\tilde p),\quad
\text{acc}\leftarrow 2^{\,m_{\text{old}}-m}\text{acc}+\tilde p\cdot v
$$

Arithmetic intensity ~2 FLOP/byte, far below the ridge point, so wall-clock tracks bytes
almost directly, and the write-back a materialise-then-attend design pays is ~4× the
compressed bytes.

### Kernel 3 — batched eviction

```
R ← L · B                  # layer i owns rows [i·B, (i+1)·B) of one joint store
at each evicting step:  run the UNCHANGED body once over R rows   # not L times over B
then every layer appends this step's token                        # order FLIPPED
```

With per-launch dispatch cost `c`, the body pays `L·c` before and `c` after; bytes moved are
**unchanged**.

---

# Part B — This branch: the card and the read gate

## B1. At a glance

**1. Stop reading the compressed tier to find out you did not need it.** Every token expanded
**every** compressed window so it could be attended over and accumulate a score. But that tier
is by construction the band ranked *below* full precision — most windows contribute almost
nothing, and we paid full price to discover they were cheap. Each window now gets a small
**card**, written once when it is compressed; decode scans every card and opens only a quarter
of the tier.

**2. The windows it skips still take part in the answer.** A gate that reads a quarter of the
tier and returns that softmax is not an approximation of attention over the cache — it is
**exact attention over a different cache**, one where the other three quarters do not exist
and the surviving weights have been inflated to cover for them. So each card also stores the
window's **average value vector**, and a skipped window contributes it at its estimated
weight, with the output renormalised for the mass added back.

**3. The two halves of the card answer different questions.** The key half decides **whether**
a window can matter, so it must find outliers: the card models a window as a *line* — an
average, a direction, one number per token — with the direction pointing at the token
**farthest from the average**, because a plain average hides the one hot token, exactly the
case the gate exists to catch. The value half only answers **what** a window contributes once
we have decided it does not matter much, and there the weights are near-uniform, so a plain
average is the right first-order answer.

**4. The finding that governs the whole idea — a negative result with a clean explanation.** A
card is read for **every** window on **every** step; a window is opened only if selected. So
the gate costs *one card plus the fraction you open* against *every window*, and break-even is
`1 − card/window`. **The card's width, not the gate's cleverness, decides whether it pays.**
At 804 B/head against a 272 B card that is a **0.66 read ratio**, so the shipped quarter saves
~**3% of what decode moves**. **Measured, reading everything ties with reading a quarter to
within 0.2–1.6% of time per token** — and runtime address resolution makes every field a
scattered fetch, putting the gated kernel **12× off its bandwidth roofline**. **The cause is
the memory layout, not the idea:** what remains is making the selected reads **contiguous**.

**5. One production path, no ungated arm.** Environment switches could route a production run
into different code and none appeared in the output. They are gone; the device decides, and a
layer arriving without cards now **raises** rather than reading the whole tier correctly and
silently. A full read is a setting through the same code — a *provable* no-op, and the control
arm.

**6. The compiled cache is bit-identical to the eager one, and the budget prices what is
held.** The compiler does not reproduce an intermediate precision rounding step the storage
format depends on, so three quantisers diverged from eager once the body was compiled. And the
byte accounting omitted cards, **understating the compressed tier by 38%**.

## B2. How it works

### The card

Written once when a window is compressed, from the **already-rotated** keys the eviction body
holds one line before stripping their rotation — so it costs no extra rotation, and is frozen
for life because positions never move.

| field | length | precision | bytes | what it is |
|---|---|---|---|---|
| average key | `D` | **4-bit** + scale | 66 | mean key, as a difference from a frozen reference |
| direction | `D` | 8-bit + scale | 130 | unit vector toward the farthest-out token |
| projections | `ws` | 8-bit + scale | 10 | how far each token sits along it |
| average value | `D` | **4-bit** + scale | 66 | mean value, as a difference from its own reference |

* **The direction must stay 8-bit.** At 4 bits the hot token stops being the best-fit token in
  its window — the single property the card exists for. A 1-bit sign vector is catastrophic:
  the hot token ends up with the *largest* error in its window, 38.0 against a window average
  of 15.0; 4-bit gives 7.0, 8-bit 0.39.
* **The averages need no such range.** Over 8 seeds, median relative output error at the
  shipped ratio is **0.00004 for both encodings**. The projections are 10 B and cannot shrink.
* **Averages are stored as differences from a frozen reference.** Certain channels carry
  massive activations nearly identical across *every* window, so they separate nothing while
  consuming the whole range; a per-layer reference cut the average's error **17×**.

**The guarantee that was designed, measured, and retired.** The card originally carried a
per-token residual, turning the estimate into a **provable upper bound** via Cauchy–Schwarz:
the gate could over-select but not miss. It was removed because **nothing on the production
path read it** — the bound serves a threshold rule, and the shipped gate is a top-*k*. Not a weakening: its slack lets a loosely-bounded window outrank a tighter one
with a *higher* true logit, and the estimate holds **100.0%** of worst-head mass at a 0.15
ratio against the bound's **99.7%**.

### The gate kernel, and where selection had to live

Selection needs the **query**, and the cache interface the model calls is handed keys and
values only — the library's API, not a choice. So the cache hands over the tier plus the cards
and selection runs one layer later, inside the attention call. *The tier decision is
query-independent and happens at eviction; the read decision is query-dependent and must
happen inside attention.*

One program per (batch, key head), the same grid as the decode kernel, so the group union is
free. It is a **separate** kernel rather than a prologue because the decode kernel's tile-fit
ladder **falls back silently** when its budget is exceeded — a lost rung would cost more than
the gate saves, with no error. Two selection rules measurement forced:

* **The cap binds on the key head, after the group union.** One program owns a key head and
  every query head sharing it, so a window loads once for the whole group. If each query head
  picks separately and the picks are unioned after, a 25% cap becomes **100% loaded** at group
  size 4.
* **Selection is per (row, key head), not per row.** On a 271-window tier, 8 key heads, 32
  query heads, at ratio 0.25:

| granularity | windows read | worst head's mass recalled |
|---|---|---|
| per (row, key head) | 68 / 271 | **100.00%** |
| per row, union of the heads' picks | 240 / 271 | 100.00% |
| per row, top-k of the row's best | 68 / 271 | 97.54% |

The union reads 89% of the tier — no saving; a row-level top-*k* gives up 2.5% of the worst
head's mass for identical traffic. Counts stay equal across rows and heads, so the result is
still dense. **Selection reaches the kernel as an indirection:** the tier stays whole and the
inner loop dereferences a selection list, so **the gate declines to read bytes rather than
shrinking the store**.

### Representatives attend

The epilogue already computed, per window, the card's estimate rescaled by the deviation the
opened windows give for free, and used it only for the score; it now also uses it for the
**output**. Renormalising a softmax over a subset redistributes the unselected mass onto the
survivors, so adding each skipped window's average value back at its estimated weight restores
that mass at first order — for one tile of new traffic (66 B/head against the 804 B window
*not* read).

**The scores still must be filled**, because scores are what eviction ranks on: a skipped
window scoring zero would rank last and be evicted, so reading the tier less often would
silently *destroy* it. The rescaled fill gives rank correlation **0.996** and recovers
**99.5%** of mass, against **0.32** if skipped windows are left at zero. **Cards alone cannot
replace opening windows:** opening nothing is a 0.908 relative error, because the calibration
factor is measured on the windows that *were* opened.

### The byte accounting, restated honestly

**The scale grid is one byte per entry, not two.** It was 48.5% of a window's
codes-plus-grid — two bytes of precision wrapping a two-bit code. It is now 8-bit against a
16-bit scale shared by 32 entries, nearly free because the codes are fitted to the *stored*
grid: a rounded scale repositions the four levels and the codes re-fit.

| grid | key error | query·key error | bytes/head |
|---|---|---|---|
| 16-bit (was) | 0.13348 | 0.13541 | 512 |
| 8-bit, one scale per head | 0.13370 | 0.13568 | 260 |
| **8-bit, groups of 32** | **0.13368** | **0.13562** | **272** |

**The grouping is what makes it safe, and aggregate error cannot see why.** On a
massive-activation channel, a single scale per head leaves the worst channel at **1.082**
relative error — worse than returning zeros — while the Frobenius norm moves 0.14% and reports
nothing. Groups of 32 put it back at 0.306 for 12 B/head. *Aggregate error is the wrong
instrument for a per-channel failure.*

| | before | **now** |
|---|---|---|
| compressed window, codes + grid | 8448 B | **6432 B** (−23.9%) |
| card | not charged | **2176 B** |
| window as charged | 8448 B | **8608 B** |
| price ratio, full : compressed | 3.88 | **3.81** |
| windows bought at fixed budget | — | **+21% from the grid change alone** |

And for any latency discussion: **decode time barely tracks cache size** — a configuration
holding **2.5× more keys** ran **22% faster**, so per-window decompression cannot dominate.

## B3. Algorithms and maths

### Building and reading a card

$$
\mu=\tfrac{1}{ws}\sum_i k_i,\quad
i^\star=\arg\max_i\|k_i-\mu\|,\quad
v=\frac{k_{i^\star}-\mu}{\|k_{i^\star}-\mu\|},\quad
t_i=\langle k_i-\mu,v\rangle,\quad
\bar v=\tfrac{1}{ws}\sum_i v_i
$$

stored as `4-bit(μ − a_K)`, `8-bit(v)`, `8-bit(t)`, `4-bit(v̄ − a_V)`. This is **greedy
*k*-center with one center**, not Lloyd's: the selector wants the window's extreme point, and
*k*-center minimises exactly the radius a max-style estimate depends on. At read, with
`m = ⟨q, a_K⟩ + ⟨q, μ − a_K⟩` and `g = ⟨q, v⟩` — **two dot products** — every per-token
estimate follows:

$$
x_i=\text{scale}\cdot(m+t_i g),
\qquad
\underbrace{\max_i x_i}_{\text{selection ranks on this}},
\qquad
\underbrace{\log\sum_i e^{x_i}}_{\text{a skipped window's contribution}}
$$

*Retired:* with a per-token residual `ε_i ≥ ‖k_i − k̂_i‖` measured against the **decoded**
card and rounded **up**, Cauchy–Schwarz gives `|q·k_i − (m + t_i g)| ≤ ‖q‖ ε_i` — a provable
upper bound on every logit in the window. Nothing consumed it.

### Selection

$$
\hat x[b,h,w]=\max_{r\in\text{group}(h)} x[b,h,r,w],
\quad
n_{\text{sel}}=\max\big(1,\lceil r\cdot n_{\text{active}}\rceil\big),
\quad
\text{keep}(w)=\big[\text{rank}_{\hat x}(w)<n_{\text{sel}}\big]
$$

The group reduction **must** precede the cap; reversing them admits up to `r × group_size`
windows. The cap resolves against the **live** active count, because the tier size moves with
the shape — 115 windows at a 1024-token prompt, 271 at 4096.

### What a skipped window contributes

With `R` the opened set, `e_w` the estimated mass, `s_w` the real score:

$$
c=\frac{\sum_{w\in R}s_w}{\sum_{w\in R}e_w},\qquad f_w=c\,e_w\ (w\notin R)
$$
$$
\text{score}_w=\begin{cases}s_w & w\in R\\ f_w & w\notin R\end{cases}
\qquad
\text{out}=\frac{\sum_{w\in R}\sum_{i\in w}p_i v_i+\sum_{w\notin R}f_w\bar v_w}{1+\sum_{w\notin R}f_w}
$$

`c` needs nothing the step did not already compute — the opened windows carry **both** a real
and an estimated score, so it is free and self-calibrating.

### The governing law

$$
C_{\text{gated}}(r)=\underbrace{B_{\text{card}}}_{\text{every window}}+\;r\cdot\underbrace{B_{\text{win}}}_{\text{opened only}},
\qquad
C_{\text{ungated}}=B_{\text{win}}
\qquad\Longrightarrow\qquad
\boxed{\;r^\star=1-\frac{B_{\text{card}}}{B_{\text{win}}}\;}
$$

`B_win = 804` B/head (512 codes + 292 grid), `B_card = 272`, so `r* = 0.66` and `r = 0.25`
costs **58.8%** of an ungated read. Two corollaries:

1. **A gate over a compressed tier is bounded by its own summary's compression ratio.** At 2
   bits per number, an 8-bit summary of three full-length vectors cannot win.
2. **The saving is diluted by whatever the gate does not govern.** The gated tier is ~13% of
   the read, so a 41% saving on it is ~3% overall.

### One gated decode step

```
if the layer has no cards: RAISE     # reading the tier whole would be CORRECT and
                                     # would look exactly like success
1  estimate every window from its card                    # two dot products each
   combine the query-head group, then keep the top n_sel  # union BEFORE the cap
2  expand and rotate ONLY the kept windows                # the point of all this
3  one fused launch over [sink | full-precision body | kept windows],
   dereferencing the selection rather than repacking the store
4  full-precision windows: contiguous reduce into per-window scores
5  epilogue, over every window column:
     c ← (real mass over kept) / (estimated mass over kept)
     skipped w: score_w ← c·e_w ;  out ← out + c·e_w · (window's average value)
     out, scores ← (out, scores) / (1 + total added mass)
```

At ratio 1.0 this selects every window and is **byte-identical** to attending over the whole
tier — the no-op proof that this is a *selection*, not a second attention.

### The compiled-eviction numerics fix

The storage format requires a round trip through 16-bit when fitting codes to their scale, so
the grid the codes were fitted to is bit-identical to the grid every later read uses. The
compiler drops it and keeps the wider value in a register. Over 24 seeds: the key quantiser
diverged from eager on 11, the value quantiser on 14, the card's encoder on **all 24**.
Restoring the emulation around the one compile site takes all rows to zero.

---

# Part C — How the core architecture changed

**Before, the system asked one question: what should the cache keep? Now it asks a second —
what should this step actually read? — and the two have different inputs, frequencies and
homes in the code.**

| | **before** | **now** |
|---|---|---|
| decisions | one: which tier each window goes to | two: tier assignment **+** read selection |
| what decides | accumulated attention scores | tier: same; read: **the current query** |
| when | every few steps, at eviction | tier as before; read **every step** |
| where | the cache's update call | read **inside the attention call**, since the cache API never sees a query |
| compressed read per step | the whole active tier | a card scan plus a fraction |
| attention over that tier | every window attends exactly | selected exactly; **skipped through their average value** |
| eviction scores | every entry exact | exact where opened; calibrated estimate where not |
| compressed window price | 8448 B | **6432 B** of data, **8608 B** with its card |
| compiled vs eager | equal under a shallow check, not the real compiler | **bit-identical**, pinned |

**What stayed identical**, which makes the gate cleanly separable: positions are still never
renumbered, a window's scale is still pinned at first compression, the tier split is still
**rectangular** so the tensor stays dense, and the decode kernel is still **one launch** per
layer per step. The gate changes **what is read**, never **what is retained** — except through
the eviction scores, which is why the calibrated fill exists.

**The structural insight.** Before, the compressed tier's cost was tied to its **size**: 2-bit
storage buys ~4× more retained windows per byte, but the read path paid for all of them every
step, whether or not the query cared. **Compression bought retention and handed the bill back
to decode.** The gate breaks that coupling in principle — bytes read become a card per window
plus a chosen fraction, free of the budget — but **measured**, it has not converted into time:
the card is 34% of a window, the governed tier is ~13% of the read, and what it saves is eaten
by scattered fetches. **The kernels made the cache fast; making holding more stop costing more
to read is now a bytes-and-address-order problem, not an asymptotic one.**

## Status, claim by claim

| claim | status |
|---|---|
| the three kernels: 2.81× per-token, 2.05× slower → 1.37× faster | **measured on GPU** |
| batched eviction is byte-identical to the per-layer path | **pinned by tests**, every config axis |
| the rotary-precision fix recovered the long-document regression | measured; the run bundles more |
| card is 272 B against an 804 B window; break-even at 0.66 | **arithmetic**, from the byte formulas |
| the gate's selection is worth 0.2–1.6% of per-token time — a tie | **measured** against the read-everything control arm |
| the gated kernel is 12× off its roofline | **measured**, GPU profile |
| 4-bit cards are accuracy-neutral at the shipped ratio | **measured on CPU**, 8 seeds |
| 4-bit cards are *faster* | **not measured** — bytes only |
| the compiled eviction is worth ~10% | **measured** against its own control arm |
| "the read gate is a 24% regression" | **not attributable** — a configuration change sits in the same range |
| comparison against the two external baselines | **never run — neither is implemented** |

Two framing points for any write-up:

* **Batch 1 is not winnable, and that is fine.** Every model weight is read once per token —
  ~16 GB, a ~10 ms/token floor independent of cache size. At batch 1 the claim is **memory**,
  not latency.
* **Across the six cells tested, the method beats an uncompressed flash cache in one** (two at
  its best operating point) — the long-prompt, large-batch cell the method is *for*.
