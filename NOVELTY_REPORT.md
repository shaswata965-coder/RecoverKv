# What is new here, and why it would matter in a paper

Two branches, read end to end, and then compared.

* **Part A — `quant_batched_fixed_int2_eviction`.** Two spans, one branch.
  The **foundation**, everything up to `37f5837` (29 Jul) — the two-tier int2
  cache itself. Then the **kernel and throughput span**, `c88581c..13f8b9b`
  (13 Aug → 9 Sep) — the fused score kernel, the fused decode kernel, and the
  batched eviction. The 3 Aug sticky-Q commits (`06f8c5a`, `b668230`) are
  deliberately **out of scope**: they branched off `37f5837` onto
  `int2_no_promote` and were never merged here, so this branch's foundation is
  the state *before* them.
* **Part B — `int2_clustered_rep`.** This branch, read at `d562940` (20 Sep):
  97 commits on top of `13f8b9b` — the per-window *card*, the read gate built on
  it, the measurement that says what the gate is currently worth, and the logic
  changes that came with them.
* **Part C —** how the core architecture changed from A to B.

Each part says the idea in plain words first, then explains it, then gives the
algorithm and the maths in a form a paper could use directly.

`HARDWARE_INTEGRATION_REPORT.md` covers the same kernel work as Part A §A2.4–A2.6
from an engineering angle; this document is the novelty angle and does not repeat
it. `DISTANCE_TO_GOAL.md` is the live status source for every measured number
quoted in Part B and §C5.

---
---

# Part A — `quant_batched_fixed_int2_eviction`

## A1. The five things that mattered, at a glance

**1. A token gets three outcomes, not two — and the third one is free to undo.**
Every other eviction method decides *keep* or *drop*. This one adds *demote*: a
window too weak for fp16 but too useful to throw away is re-stored at 2 bits.
The part that is actually new is what happens next. A window's quantisation grid
is fixed the first time it is written and never recomputed, so if the window
later earns its way back to fp16 and then falls back again, it is **reactivated,
not re-quantised**. Promote/demote cycling therefore costs zero compute and adds
*exactly* zero extra error — not "bounded" error, zero. Error cannot compound,
because no arithmetic path in the system ever runs `quantise(dequantise(x))`.

**2. Eviction moves memory. It never moves positions.**
Compaction gathers the survivors so they sit contiguously, but every surviving
key keeps the absolute position it was born with. Nothing is renumbered, nothing
is re-rotated, and the query needs no position override. This removes a whole
class of bookkeeping that position-rebasing caches need — and it is also the
reason point 1 works: because a window's content and its positions are both
frozen, its 2-bit codes are valid forever.

**3. The budget is spent in bytes, so compression converts directly into
retention.**
The split knob `q` divides the *memory* allowance, not the window count, and
each tier buys windows at its own price. An int2 window costs 3.88× less than an
fp16 one, so at the same byte budget the quantised tier holds ~3.9× as many
windows. This is the sentence the whole method exists to support: *compression
is not a memory result, it is an accuracy result, because the bytes you save buy
back context you would otherwise have dropped.* The branch also carries the
counter-example — when the default silently flipped to splitting by token count
instead, every one of twelve LongBench datasets moved down together, which is a
clean natural ablation of exactly this claim.

**4. The eviction score stops being a second pass.**
Scoring which windows to keep normally means recomputing attention. Two separate
results remove that cost. At **prefill**, the score is obtained by running half
of FlashAttention's *backward* pass with the gradient replaced by ones — it has
the exact shape needed (a sum over queries, per key) and, because the forward
already produced the log-sum-exp, the second pass needs no online-softmax
bookkeeping at all. At **decode**, the score is free outright: with one query
row, the per-key softmax weight *is* the H2O importance score, so the fused
kernel emits it from the same `q·kᵀ` it already computed.

**5. Eviction is one batched operation across all layers, not a loop over
layers.**
Every layer evicts on the same step at identical shapes — because the shapes come
from configuration, not from data. So the layer axis folds into the row axis and
one pass replaces thirty-two. This is a scheduling result, not an approximation:
the two paths are byte-identical.

**Result.** At Llama-3.1-8B, prefill 4096 / 256 decode, batch 32, one A100:
TPOT **0.1760 s → 0.0626 s (2.81×)**, taking the method from **2.05× slower than
full-cache FlashAttention to 1.37× faster**, with peak memory flat
(49.20 → 48.44 GB).

---

## A2. How each one works

### A2.1 Three tiers and a frozen grid

A *window* is `window_size` consecutive tokens — the unit of both scoring and
quantisation. Every `window_size` decode steps the cache ranks windows by
accumulated attention and partitions them:

| outcome | stored as | eligible |
|---|---|---|
| **K tier** | fp16, rotated in place | top-ranked evictable windows, **plus** sink and local windows which are force-fp |
| **Q tier** | int2 codes, **pre-RoPE** | the next band down |
| dropped | — | the rest |

The two tiers are two *separate gap-free dense stores*, not one tensor with holes.
A zero-padded full-length tensor was rejected for a reason worth stating in a
paper: zero keys are not softmax-neutral, since `exp(q·0) = 1`.

Quantisation is asymmetric affine, per-channel for keys (group = the window along
the token axis) and per-token for values (group = head_dim), computed in fp32 but
**fitted against the fp16-stored scale and zero**, so the grid the codes were fit
to is bit-identical to the grid every later dequantisation reads. Four 2-bit codes
that share a scale are packed into one byte, so a byte load is one scale load.

The *pinned grid* is the load-bearing invariant. A window's `(scale, zero)` is set
once, at first demotion, and never recomputed. Its ledger entry — codes, grid, and
immutable position range — survives promotion in a dormant state, so a later
demotion is "drop the fp16 copy, flip the entry active". Zero drift and zero
compounding are therefore **structural**, not numerical.

### A2.2 Positions are never rebased

Keys in the Q tier are stored *pre*-RoPE and rotated at read time using each
window's original absolute positions, which eviction never changes. The fp tier is
already rotated and is never re-rotated. The query keeps HuggingFace's natural
monotonic position. Consequence: query↔key relative phase is exact across both
tiers with no per-eviction position bookkeeping, no `arange` renumbering, and no
position override.

Two second-order consequences are what make the rest of the design close:

* Because positions never move, a demoted window's content is a fixed tensor, so
  its codes are frozen for life — which is what licenses reactivation (A2.1).
* The fp store must keep **physical window order = chronological window order**,
  because that is what lets a window's token range be computed as
  `num_sink + rank_fp · window_size + offset`. A promoted window is therefore
  *spliced* into its chronological slot, not appended. The Q store has no such
  constraint: its ledger `offset` locates each window's codes, so any physical
  order is fine.

All per-window bookkeeping lives on one **merged chronological window axis**
(both tiers, sorted by original window id), which is also the order the effective
K/V is materialised in — so the existing window scorer's physical chunking stays
valid unchanged.

### A2.3 The byte budget

The resolver is folded into the existing token-granular one so that `q = 0`
reproduces the single-tier configuration field for field. Only the *evictable*
band is split; sink and local are carved first, in tokens, and are always fp16.

At `H_kv = 8, D = 128, window_size = 8`: `b_fp = 32768 B`, `b_q = 8448 B`.
The ratio is **3.88×**, not 8×, because the fp16 scale/zero grid does not shrink
with bit width — at int2 the grid is **51% of a quantised window's cost**. That is
a publishable observation in its own right: *at 2 bits the lever is amortising the
grid (larger windows, coarser key groups), not shrinking the codes.*

### A2.4 Scoring without a second attention pass

**Prefill — the backward-pass trick.** You cannot pull the eviction score out of
FlashAttention's forward loop. The forward's rescaling correction `exp(m_i − m_i')`
is indexed by query row `i`, and the score is a sum *over* `i`; you cannot pull a
per-`i` factor out of a sum over `i` after the rows have been collapsed. But
FlashAttention's *backward* already computes a quantity with exactly the right
shape — `dV_j = Σ_i Pᵀ_i dO_i`, summed over queries, per key — so setting
`dO = 1` degenerates the gradient matmul into a plain column sum, which is the
score. And because the forward has already finished, its true normaliser `L_i` is
known, so the second pass needs no running max and no rescaling: just
`S = QKᵀ`, subtract `L`, exponentiate, reduce. The loop is key-outer, so each
program owns one key block, writes only that block, and needs no atomics. Fully
masked causal tiles (about half of them) are skipped before any matmul.

**Decode — the score is already there.** With `T = 1` there is no query grid to
tile, and the per-key softmax weight is itself the H2O score. The fused kernel
emits it from the `q·kᵀ` it computed for the output.

**Later refinement (window scores from the kernel).** The kernel was changed to
emit per-*window* sums directly, plus a per-window max companion, and rescale in a
register epilogue — traffic `4·S → 5·W`, and the host-side reduction collapsed
from pad + reshape + sum + einops-reduce + concat + gather down to **one gather**.
Worth ~11 ms/step, uniformly across every shape and batch — and that uniformity is
itself the signature of a removed *fixed* cost, not a traffic term.

### A2.5 The fused two-tier decode kernel

One launch replaces four per-step per-layer costs: the fp16 write-back of the Q
tier, the concatenation, a full-budget fp16 attention read, and a duplicate
`q·kᵀ` for scoring. It reads the Q tier **as int2** and dequantises plus rotates
**in registers**.

Mechanics folded in: one program per `(batch, KV head)` so each key tile and each
dequantisation happens once per KV head rather than once per query head (4× less
work at Llama-3.1-8B's GQA); online softmax so the `[S]` score row is never held;
tiled `tl.dot` over both tiers; RoPE applied from the two head-dim halves in
register arithmetic with no rotate-gather; base-2 softmax with `log2 e` folded
into the scale at launch.

Tiling in **whole windows** padded to a power-of-two block is what makes
`window_size ∈ [1, 128]`, non-powers-of-two included, all work — a window never
straddles a tile for any `ws`, and the cost is lane padding alone (100% lane use
at powers of two, 94% at `ws = 12`).

### A2.6 Cross-layer batched eviction

All `L` layers evict on the same step at identical shapes, because `ws`, budget,
`n_q` and `k_fp` come from config and not from data. So fold `L` into the row axis
(`R = L·B`, layer `i` owning rows `[i·B, (i+1)·B)`) and one batched pass replaces
`L` sequential ones. The eviction body did not change at all — only its row count.

The one thing this forces is an **ordering flip**: with a joint store, eviction
must run *before* any layer appends this step's token, because when layer 0's
`update` is called, layers `1..L−1` have not produced token *t* yet. So "append
then compact" becomes "compact then append". That is safe — and the argument is
exactly the kind a reviewer will want: the retained set reads **only**
`window_scores`; this step's token has no score column yet and lands in the local
region, which is force-fp and never evictable. Byte-identity against the
per-layer path is pinned across `q ∈ {0, 0.5}`, `L ∈ {1,3}`, `B ∈ {1,2}`,
`ws ∈ [1,128]`, sinks on and off, and promotion/dormancy.

Prefill deliberately stays per-layer: joining before the first eviction would hold
the `L·B`-row prompt and the `L·B`-row steady store simultaneously, about +4 GB of
peak straight out of max batch. Joining *after* the first eviction, once every
prompt buffer is released, costs nothing.

Measured on CPU dispatch counts (`L=32, B=4, ws=8`): eviction body
**13,824 → 426 ops (32.5×)**; amortised per decode token **5,592 → 1,177 (4.75×)**.

---

## A3. Algorithms and maths

### Notation

`B` batch, `L` layers, `H_kv`/`H_q` KV/query heads, `rep = H_q/H_kv`, `D` head dim,
`ws` window size, `W` merged window count, `q ∈ [0,1]` the tier split ratio,
`β` the cache budget fraction.

### M1 — Pinned-grid int2 quantiser

For a quantisation group `G` (window along tokens for keys; head_dim for values):

$$
s=\frac{\max_G x-\min_G x}{3},\qquad z=\min_G x,\qquad
c=\mathrm{clamp}\!\big(\mathrm{round}_{\text{half-even}}\tfrac{x-z}{s},\,0,\,3\big),
\qquad \hat x = c\,s+z
$$

with the degenerate case `max = min ⇒ s = 1` giving `ĉ = 0`, `x̂ = z` exactly. The
grid is stored fp16 and the codes are fit against the **fp16** values, i.e.
`s ← fp32(fp16(s))` before the round, so the fit grid equals the read grid bit for
bit. The float-offset form (`z = min`) is used rather than an integer zero point
because K/V groups frequently exclude zero, and an integer zero point clamped to
`[0,3]` cannot represent an offset outside the group's own span.

### M2 — Zero-drift theorem (the reason reactivation is exact)

Let window `w` be demoted at time `t₁`, promoted at `t₂ > t₁`, demoted again at
`t₃ > t₂`. Because eviction never rebases positions, the pre-RoPE content `K_w`
and the position range `P_w` are both invariant in `t`. With the grid `(s_w, z_w)`
pinned at `t₁`, the codes at `t₃` are *by identity* the codes at `t₁`:

$$
c_w^{(t_3)} \equiv c_w^{(t_1)},\qquad
\varepsilon_{\text{quant}}\big(t_3\big)=\varepsilon_{\text{quant}}\big(t_1\big)
$$

so error after `n` promote/demote cycles is independent of `n`. Implementation:
the dormant ledger entry is reactivated, so the re-quantisation is not merely
idempotent — it is never executed. (A re-quantisation path would *not* be exact:
it would pick up fp16 rounding from the intermediate dequantise/re-rotate on
promotion and could flip boundary codes.)

### M3 — Tier-aware byte budget

$$
b_{fp} = 4\,H_{kv} D\,ws
$$
$$
b_q = \underbrace{\tfrac{1}{2}H_{kv} D\,ws}_{\text{packed int2 K+V}}
    + \underbrace{4 H_{kv} D}_{\text{key grid}}
    + \underbrace{4 H_{kv} ws}_{\text{value grid}}
$$
$$
M_{\text{evict}}=\big(\lfloor \beta\,(T_{\text{prefill}}+T_{\text{gen}})\rfloor - n_{\text{sink}} - T_{\text{local}}\big)\cdot 4H_{kv}D
$$
$$
k_{fp}=\Big\lfloor \frac{(1-q)\,M_{\text{evict}}}{b_{fp}}\Big\rfloor,
\qquad
N_q=\Big\lfloor \frac{q\,M_{\text{evict}}}{b_q}\Big\rfloor
$$

At `H_kv=8, D=128, ws=8`: `b_fp = 32768`, `b_q = 4096+4096+256 = 8448`, ratio
**3.88×**, grid share **51%** of `b_q`. Effective key bits ≈ `2 + 32/ws`, so the
grid term *exceeds* the code term for `ws ≤ 16`.

`q = 0` gives `N_q = 0` and `k_fp = (budget − sink − local)//ws`, i.e. exactly the
single-tier configuration — parity is by construction, not by test.

### Algorithm 1 — One eviction (per layer, every `ws` steps)

```
Input : window_scores S [B,H_q,W] on the merged axis, ledger, k_fp, N_q
1  rank the EVICTABLE band of the merged axis by S            # sink/local excluded
2  fp  ← top k_fp of that band     ∪ sink ∪ local             # force-fp
   Qs  ← next N_q of that band
   drop← the rest
3  for w in (Qs \ current_Q):                                 # demote
       if ledger[w] dormant:  reactivate(w)                   # O(1), exact  (M2)
       else: un-rotate K_w at its ORIGINAL positions (negated sin)
             (s,z) ← pin_grid(K_w); codes ← quantise(K_w; s,z)
             record immutable position_range(w)
   for w in (fp ∩ current_Q):                                 # promote
       K_w ← RoPE(dequantise(codes_w), position_range_w)
       splice K_w into the fp store at its CHRONOLOGICAL slot
       keep ledger[w] DORMANT (do not free)
4  fp store  ← [ sink ‖ fp windows in chronological order ]   # invariant
   Q  store  ← compact by shifting ledger offsets (any order)
5  align S and original_window_ids to the merged survivors
```

Invariant maintained by step 4: *the fp store's physical window order equals
chronological window order*, which is what makes index translation
`num_sink + rank_fp·ws + offset` correct at the next eviction.

### Algorithm 2 — Prefill eviction score (FlashAttention-backward form)

Identity, with `dO = 1`:

$$
dV_j=\sum_i P_i^{\top}dO_i \;\xrightarrow{\;dO=\mathbf 1\;}\;
\sum_i \mathbf{1}^{\top}P_i=\text{token\_scores}_j
$$

and, since the forward already produced `L_i = m_i + log ℓ_i`,

$$
P[i,s]=\exp\!\big(S[i,s]-L_i\big),\qquad S[i,s]\le L_i \;\Rightarrow\; \text{no overflow, no running max}
$$

```
Stage A:  O, L ← flash_forward(Q,K,V)                # unchanged; L is KEPT, not discarded
Stage B:  program(key_block j, batch b, kv_head h):
            load K_j into SRAM                       # once, reused below
            acc ← 0
            for each query_block i visible to j:      # fully-masked tiles SKIPPED
              for each query head r in the GQA group of h:
                S ← scale · Q_i Kⱼᵀ ;  P ← exp(S − L_i)
                acc += colsum(P)
            token_scores[b,h,j] ← acc                 # disjoint writes, no atomics
```

Cost to enable: the forward must retain `L`, a `[B, H_q, T]` array — megabytes,
not a second attention.

### Algorithm 3 — Fused two-tier decode (one launch, per layer, per step)

For `T = 1` the softmax weight is the score:

$$
\text{score}_s=p_s=\frac{\exp(\text{scale}\cdot q\cdot k_s)}{\sum_{s'}\exp(\text{scale}\cdot q\cdot k_{s'})}
$$

```
program(batch b, kv_head h):                          # grid = B · H_kv
  m,l,acc ← -inf,0,0
  for tile over fp tier:                              # already rotated
      s ← scale · q·kᵀ  ;  online_softmax_update(m,l,acc,s,v)
  for tile over Q tier (whole windows, padded to BLOCK_WS):
      k ← dequantise_int2_in_registers(codes, s_grid, z_grid)
      k ← rope_halves(k, cos,sin at the window's ORIGINAL positions)
      s ← scale · q·kᵀ  ;  online_softmax_update(m,l,acc,s,v)
      wsum[w], wmax[w] ← running per-WINDOW mass          # traffic 4S → 5W
  epilogue: rescale wsum by (wmax − lse); out ← acc / l
```

with `log2 e` folded into `scale` so every exponential is `exp2` (one
`ex2.approx.f32` instead of ~ten SFU ops).

### Algorithm 4 — Cross-layer batched eviction

```
R ← L·B                                               # layer i owns rows [i·B,(i+1)·B)
at every decode step t:
  if should_evict(t):  Algorithm 1 over the JOINT R-row store     # one pass, not L
  then every layer appends token t                                # order FLIPPED
```

Safety condition (stated as a lemma, because it is the whole argument):
`retain(t)` is a function of `window_scores` only; token `t` has no score column
at step `t` and lies in the local region, which is force-fp; therefore
`retain(t)` is invariant to whether token `t` has been appended.

---
---

# Part B — `int2_clustered_rep` (this branch)

> Read at commit `d562940` (20 Sep). Part A's byte figures are Part A's: this
> branch changed both the grid format and the card, and §B2.4 states the new ones.

## B1. The four things that matter, at a glance

**1. Stop reading the compressed tier to find out you did not need it.**
Before this branch, every decode step dequantised **every** active int2 window so
it could be attended over and could accrue a score. But the Q tier is *by
construction* the band ranked below the fp tier — most of those windows contribute
almost nothing, and we were paying full price to discover they were cheap. This
branch writes a tiny **card** for each window, once, when the window is sealed.
Decode scans every card, picks the windows worth opening, and dequantises only
those — 25% of the tier at the shipped operating point. Reading a card costs two
dot products.

**2. The windows it skips still take part in the answer.**
This is the strongest claim on the branch and it is easy to miss. A gate that
reads a quarter of the tier and returns that softmax unchanged is not an
approximation of attention over the cache — it is **exact attention over a
different cache**, one where the other three quarters do not exist and the
surviving weights have been inflated to cover for them. So each card also carries
the window's **value centroid**, and a skipped window now contributes that
centroid at the weight the card estimated for it, calibrated against the windows
that *were* opened; the output and every window score are renormalised by
`1 + Σ fill`. It costs one extra `tl.dot` on a tile the loop already holds — no
new launch, no new pass over the tier.

**3. The two halves of the card answer different questions, so they are built
differently.**
The key side decides **whether** a window can matter, so it needs an
outlier-seeking direction: the card models the window as a line, `k_i ≈ μ + t_i·v`,
with `v` pointing at the token **farthest from the mean** — one pass of greedy
*k*-center, no convergence loop. A centroid would hide the one hot token, which is
exactly the case the gate exists to catch (`exp` is convex, so a mean always
understates a mixed window). The value side only answers **what** the window
contributes once we have already decided it does not matter much — and inside a
window the gate declined to open, the softmax weights are near-uniform by
construction, so the unweighted mean *is* the correct first-order term. Outlier
model for keys, centroid for values, for a stated reason on each side.

**4. The finding that governs the whole idea — and it is a negative result with a
clean explanation.**
A card must be read for **every** window on **every** step; a window is only
opened if selected. So the gate costs `card + ratio × window` against `window`,
and break-even is `1 − card/window`. That one line is the whole economics, and it
says the card's *width* — not the gate's cleverness — decides whether the feature
pays. At `H=8, D=128, ws=8` a window's token data is 804 B per head and the card
is 272 B, so break-even sits at a **0.66 read ratio** and the shipped 0.25 costs
59% of an ungated read — a saving of ~3% of what the decode kernel actually moves,
because the compressed tier is only ~13% of the read and the full-detail tier is
the other 87%. **Measured:** `--gate-ratio 1.0` (read everything) ties with 0.25
to within **0.2–1.6% of TPOT**. Selection currently buys nothing measurable, and
it is not free: runtime address lookup makes every field a scattered fetch, and
the kernel runs **7.03 ms against a 0.58 ms roofline — 12× off** — where the
model's own matrix math runs at 1.4× its roofline.

**The cause is the memory layout, not the idea.** The card has already been
narrowed once (int8 → int4 on the two mean vectors: 400 → 272 B/head, break-even
0.50 → 0.66) at no measurable accuracy cost. What is left is making the selected
reads contiguous — a window's fields live in six tensors plus four card tensors,
so one selected window is ten scattered fetches.

Two further changes are logic rather than method, but they are the kind that
decide whether any of the numbers above mean anything:

**5. One production path, and no ungated arm.** Environment switches could route a
production run to different code, and none appeared in the output — a number
produced under the wrong one still looks like a number. They are gone; the device
decides. The gate is now the **only** decode read path where a Q tier exists:
`_run_fused` **raises** on a card-less fused layer rather than reading the whole
tier correctly and silently. `quant_gate_ratio = 1.0` is how you ask for a full
read — it selects every window through the same code, which makes it a *provable*
no-op and therefore the control arm.

**6. The compiled cache is bit-identical to the eager one, and the cache now
prices what it actually holds.** Inductor does not emulate intermediate precision
casts, so it elided the `fp32→fp16→fp32` round trip the pinned-grid design depends
on — three quantisers diverged from eager. Separately, `bytes_per_q_window`
counted codes and scales only, while the card is resident for the life of the
window: every reported `cache_budget` understated the Q tier by **38%**. Both are
fixed, and the budget resolver now *enforces* that the leftover is too small to
buy another window of any tier.

---

## B2. How each one works

### B2.1 The card

Written once, at demotion, from the **already-rotated** keys the eviction body is
holding one line before it un-rotates them — so the card costs no extra RoPE. It
is frozen for life and reactivated on re-demotion, exactly like the codes and the
pinned grid (M2), and for the same reason: eviction never rebases positions, so a
statistic taken at demotion stays valid forever.

Four fields per `(window, head)`, 272 B in total:

| field | shape | dtype | bytes | what |
|---|---|---|---|---|
| `μ` | `D` | **int4**, 2/byte + fp16 scale | 66 | window mean of keys, as a residual from a frozen anchor |
| `v` | `D` | int8 + fp16 scale | 130 | unit direction of the farthest deviation |
| `t` | `ws` | int8 + fp16 scale | 10 | each token's projection onto `v` |
| `v̄` | `D` | **int4**, 2/byte + fp16 scale | 66 | window mean of **values**, residual from its own anchor |

The int4/int8 split is measured, not stylistic. `v` **must** stay int8: it is the
deviation *direction*, and at int4 the hot token stops being the best-fit token in
its window — the single property the card exists for. (1-bit is catastrophic: the
hot token ends up with the *largest* residual in its window, 38.0 against a 15.0
mean; int4 is 7.0; int8 is 0.39 against 9.2.) `μ` and `v̄` are both means, both
stored as residuals from a frozen per-`(layer, head)` anchor, and neither needs
the range — massive-activation channels are common-mode across windows, so they
consume int8 range without separating anything (anchoring alone cut `μ` error
**17×**, 0.078 → 0.0046). Measured over 8 seeds, median relative output error at
ratio 0.25 is **0.00004 for both int4 and int8** — identical to five decimals; at
0.10 it is 0.00166 vs 0.00151, and selection overlap between the encodings is
98.5–98.8%. `t` is 10 B and cannot shrink either: without it the card is a plain
average again.

**`eps` is gone, and the honest framing matters.** The card originally carried a
per-token residual norm, which turned the estimate into a **Cauchy–Schwarz upper
bound**: `|q·k_i − (q·μ̂ + t_i q·v̂)| ≤ ‖q‖ε_i`, so a window whose bound fell below
the bar provably carried no logit above it — the gate could over-select but could
not miss. It was removed because **nothing on the production path ever read it**:
the bound exists to serve a finite `quant_gate_margin`, and the fused gate is a
top-*k* on the point estimate with no margin term. It is removed rather than
zero-filled, because a zero `eps` is not a weaker bound — it is not a bound at
all, and keeping the column would invite a reader to think one was available.

Ranking on the estimate is **not** a weakening. The bound's slack term lets a
loosely-bounded window outrank a tighter one carrying a higher true logit; measured
at the shape-C geometry (271 windows, 32 query heads over 8 KV heads) the estimate
holds **100.0%** of worst-head attention mass at ratio 0.15 against the bound's
**99.7%** — and it costs less, since the ranking path never touches `ε`. For a
paper this is a clean story: *a provable guarantee was designed, measured against
the cheaper heuristic it was meant to protect, found to be both looser and more
expensive in the only regime that ships, and retired.*

### B2.2 Selection, and where it had to live

Selecting windows needs the **query**. `transformers`' `Cache.update()` is handed
keys and values only — that is the shape of the API, not a choice. So `update()`
hands over the fp tier plus the cards, and the gate runs one layer later, inside
the attention step, which is exactly where the fused path already patches in. The
live path is `cache.update()` → `flash_decode.set_pending` → `_run_fused` →
`fused_gate` → `fused_two_tier_decode(..., sel=…)`. Worth naming in a paper as a
structural constraint: *the tier decision is query-independent and happens at
eviction; the read decision is query-dependent and must happen inside attention.*

Two selection rules that measurement forced, both easy to get wrong:

**The cap must bind on the KV head, after the GQA union.** One kernel program owns
a KV head and every query head sharing it, so a window is loaded once for the whole
group — the KV head is the unit of work. Capping per query head and unioning
afterwards lets a ratio `r` turn into as much as `r·rep` windows read; at
Llama-3.1-8B's `rep = 4` a 25% cap becomes **100% loaded** and the gate buys
nothing. So: union first (max over the group), then select.

**Selection is per `(row, KV head)`, not per row.** Measured at the shape-C tier,
ratio 0.25:

| granularity | windows read | worst-head mass recall |
|---|---|---|
| per `(row, KV head)` | 68 / 271 | **100.00%** |
| per row, union of picks | 240 / 271 | 100.00% |
| per row, top-k of row max | 68 / 271 | 97.54% |

The union reads 89% of the tier — the gate buys nothing. A row-level top-*k* pays
2.5% of the worst head's mass for identical traffic. The cost of per-head selection
is that Q positions become per-head, but counts stay equal across rows and heads,
so the result is still **dense with no padding and no keep-mask** — the same
rectangularity argument the tier split already relies on.

### B2.3 Representatives attend

The gate's epilogue already computed, for every window, the card's estimate
rescaled by the deviation the opened windows give for free — and then used it only
for the eviction score. It now also uses it for the **output**.

This closes a real gap. Renormalising a softmax over a selected subset does not
approximate the full attention; it computes a different, self-consistent
distribution in which the unselected mass has been redistributed onto the
survivors. Adding the centroids restores the missing mass to the right place at
first order:

$$
\text{out}=\frac{\sum_{w\in R}\sum_{i\in w}p_i\,v_i \;+\; \sum_{w\notin R}f_w\,\bar v_w}{1+\sum_{w\notin R}f_w}
$$

The only new traffic is the centroid tile — `D` int4 plus one fp16 scale per
skipped window, 66 B/head against the 804 B/head window not read — and the only
new arithmetic is one `tl.dot` of a tile the loop already holds, the same shape as
the Q loop's `tl.dot(p, vv)`. `GATED` is a `constexpr`, so the ungated kernel is
untouched.

Why a centroid is the right model *here* when it was the wrong model for keys:
inside a window the gate declined to open, the softmax weights are by construction
near-uniform, so the unweighted mean is the correct first-order term and a rank-1
value model would buy a second order there is no budget for.

And the scores still have to be filled, for the reason they always did: a skipped
window scoring zero would rank last and be evicted, so a change meant only to
*read* the tier less often would silently *destroy* it — 70% of the evictable
cache at `quant_ratio = 0.7`. Measured against truth, the rescaled fill gives rank
correlation **0.996** and recovers **99.5%** of total mass, against **0.32**
correlation if skipped windows are left at zero.

### B2.4 The byte accounting, restated honestly

Two format changes moved every figure in Part A §A2.3, and one accounting bug meant
the budget had never priced what was held.

**The int2 grid is one byte per entry, not two.** The fp16 scale/zero grid was
48.5% of a window's codes-plus-grid — two bytes of precision wrapping a two-bit
code. It is now int8 against an fp16 scale shared by 32 consecutive entries.
Near-free, because the codes are pinned to the *stored* grid (M1): a rounded scale
merely repositions the four int2 levels and the codes re-fit against them, so the
int2 error (~13% on keys) swamps the grid's own rounding. Measured over 12 seeds:

| grid | K rel. err | `q·k` rel. err | B/head |
|---|---|---|---|
| fp16 (was) | 0.13348 | 0.13541 | 512 |
| int8, one scale per head | 0.13370 | 0.13568 | 260 |
| **int8, groups of 32** | **0.13368** | **0.13562** | **272** |

**The grouping is what makes it safe, and aggregate error cannot see why.** On keys
whose channel gains span 1000× — which is what a massive-activation channel *is* —
a single scale per head leaves the worst channel at **1.082** relative error (worse
than returning zeros) while the Frobenius norm moves 0.14% and reports nothing.
Groups of 32 put it back at 0.306, the fp16 grid's own figure, for 12 B/head.
*Aggregate error is the wrong instrument for a per-channel failure* — a
transferable methodological point.

**The card is part of the window, and the budget now says so.** The card is
resident for the life of the window and allocated wherever there is a Q tier, so
excluding it understated the Q tier by 3200 B against the codes' 8448 — **38% on
the exact tier the memory claim is made about.** At budget 0.50 / `q = 0.70` the
honest resolve is `N_q = 518`, not 715. Two floor divisions also left up to one
window of each tier unbought; `resolve` now tops up one window at a time, each time
taking the tier that keeps the realised byte split closest to `q`, and then
**enforces** the result — under `bytes` mode the leftover must be too small to buy
another window of any tier, or it raises. Granularity-exact rather than a flat
percentage, because a percentage is the wrong shape: one fp window is 15% of a
6-window budget and 0.4% of a 263-window one.

Where that leaves the arithmetic, at `H=8, D=128, ws=8`:

| | Part A | **this branch** |
|---|---|---|
| int2 window, codes + grid | 8448 B | **6432 B** (−23.9%) |
| card | — | **2176 B** (272 B/head) |
| window as charged | 8448 B | **8608 B** |
| fp/q price ratio | 3.88 | **3.81** |
| `N_q` at budget 0.50 / `q=0.70` | — | 518 → **627** (+21% from the grid alone) |

### B2.5 Why it does not pay yet, in one number

The gate's whole premise is that a summary is cheaper to read than the thing it
summarises. **It is only cheap if it is coarser than what it summarises — and here
what it summarises was already crushed to 2 bits per number.** At int8 the card
was 400 B against 804 B of token data: **78% of what it replaces.** The summary
was written at four times the precision of the thing summarised. int4 on the two
mean vectors brought that to 272/804 = **34%**, which is what moved break-even
from 0.50 to 0.66.

Two costs the gate adds on top, both measured:

* **Scattered reads.** Ungated, the kernel walks windows in order and the hardware
  fetches them in long contiguous runs. Gated, each window's address is looked up
  at runtime, so every field becomes a scattered fetch: **7.03 ms against a 0.58 ms
  roofline, 12× off.**
* **An extra full-tier pass.** The score-fill epilogue walks *every* window, not
  the selected ones, plus a prologue that seeds every column.

Net: save ~3% of the kernel's traffic, pay for scattered reads on the rest plus two
extra full-tier passes. The remedy is contiguity, not a weaker gate.

There is a second, sharper result hiding in the same data, and it is the one a
paper should lead the latency discussion with: **decode time barely tracks cache
size at all.** A configuration holding **2.5× more keys and 2.9× more compressed
windows** ran **22% faster** (TPOT 0.0626 against 0.0763). If per-window
decompression dominated, that could not happen. What dominates is paid once per
step per layer — ~300–430 µs, moving only 1.4× across a 32× batch range.

---

## B3. Algorithms and maths

### M4 — The card

Key side, per window `w` with post-RoPE keys `{k_i}` and values `{v_i}`:

$$
\mu=\tfrac{1}{ws}\textstyle\sum_i k_i,\quad
i^\star=\arg\max_i\|k_i-\mu\|,\quad
v=\frac{k_{i^\star}-\mu}{\|k_{i^\star}-\mu\|},\quad
t_i=\langle k_i-\mu,\,v\rangle,\quad
\bar v=\tfrac{1}{ws}\textstyle\sum_i v_i
$$

Writing `m = ⟨q, a_K⟩ + ⟨q, μ−a_K⟩` and `g = ⟨q, v̂⟩`, two dot products of length
`D` give both read-path quantities:

$$
x_i=\text{scale}\cdot(m+t_i g),\qquad
\underbrace{\textstyle\max_i x_i}_{\text{est — what selection ranks on}},\qquad
\underbrace{\log\textstyle\sum_i e^{x_i}}_{\text{logmass — what a skipped window contributes}}
$$

`t` is 10 B and rides in the cache line the vectors already pulled in, so all of
it comes out of the same two dots.

*Historical, and worth reporting as such:* with a per-token residual
`ε_i ≥ ‖k_i − k̂_i‖` (measured against the **decoded** card and rounded **up**),
Cauchy–Schwarz gave `|q·k_i − (m + t_i g)| ≤ ‖q‖ε_i`, hence a provable upper bound
on every logit in the window. It was retired: nothing consumed it, and the
slack-free estimate ranks better (100.0% vs 99.7% worst-head mass at ratio 0.15).

### M5 — Selection

$$
\text{est}_{\text{kv}}[b,h,w]=\max_{r\in\text{group}(h)}\text{est}[b,h,r,w],
\qquad
n_{\text{sel}}=\max\big(1,\lceil r\cdot n_{\text{active}}\rceil\big)
$$
$$
\text{keep}(w)=\big[\text{rank}_{\text{est}_{\text{kv}}}(w)<n_{\text{sel}}\big]
$$

The group union **must** precede the cap: capping per query head and unioning
afterwards admits up to `r·rep` windows, i.e. 100% of the tier at `rep = 4`. The
cap resolves against the **live** `n_active`, because `N_q` moves with the shape
(115 windows at prefill 1024, 271 at 4096) — a pinned integer would not be the
same fraction anywhere. There is no margin rule on the production path; `r = 1.0`
selects everything through the same code and is the control arm.

### M6 — What a skipped window contributes

Let `R` be the selected set, `e_w = exp(logmass_w)` the card estimate, `s_w` the
real per-window score from attention, and

$$
c=\frac{\sum_{w\in R}s_w}{\sum_{w\in R}e_w},
\qquad
f_w=c\,e_w \;(w\notin R)
$$

Then both outputs are completed from the same `f`:

$$
\text{score}_w=\begin{cases}s_w & w\in R\\ f_w & w\notin R\end{cases}
\qquad\qquad
\text{out}=\frac{\displaystyle\sum_{w\in R}\sum_{i\in w}p_i v_i+\sum_{w\notin R}f_w\bar v_w}{1+\displaystyle\sum_{w\notin R}f_w}
$$

`c` costs one reduction and needs nothing the step did not already compute: the
selected windows carry **both** a real and an estimated score, so the correction is
free and self-calibrating. Exponentiation is taken relative to each head's own
maximum — the offset cancels in the ratio, so it is a pure overflow guard, not an
approximation; in the kernel the explicit normalisation collapses to a max and a
sum, so no logarithm is taken and every exponential stays base 2.

**Why cards alone cannot replace opening windows.** Setting `R = ∅` is not a
speed/accuracy trade — it is a 0.908 relative error. `c` is *measured on the
windows that were opened*, so with nothing opened there is nothing to calibrate
against. The cards are a selector with a first-order correction, not a substitute.

### M7 — The gate's economics (the governing law)

Per window per step, in bytes moved:

$$
C_{\text{gated}}(r)=\underbrace{B_{\text{card}}}_{\text{every window}}+\;r\cdot \underbrace{B_{\text{win}}}_{\text{opened only}},
\qquad
C_{\text{ungated}}=B_{\text{win}}
$$
$$
\boxed{\;r^\star=1-\frac{B_{\text{card}}}{B_{\text{win}}}\;}
$$

At `H=8, D=128, ws=8`: `B_win = 804 B/head` (512 codes + 292 grid),
`B_card = 272 B/head` ⇒ `r* = 0.66`, and the shipped `r = 0.25` costs
**58.8%** of an ungated read. Against int8 cards (`B_card = 400`), `r* = 0.50`.

Two corollaries a paper should state:

1. **A gate over a compressed tier is bounded by its summary's compression
   ratio.** The tighter the underlying storage, the narrower the card must be for
   selection to pay at all. At 2 bits per number, an 8-bit summary of three
   full-length vectors cannot win.
2. **The traffic saving is diluted by whatever the gate does not govern.** Here
   the gated tier is ~13% of the decode read and the full-precision tier is 87%,
   so a 41% saving on the gated part is ~3% overall. `r*` is necessary but nowhere
   near sufficient.

### Algorithm 5 — Card construction (at demotion, one pass)

```
build_card(K_w [ws,D], V_w [ws,D], anchors a_K, a_V):
  μ ← mean_i K_w[i]                                   # sweep 1
  d ← K_w − μ ;  i* ← argmax_i ‖d_i‖                  # sweep 2 (greedy k-center)
  v ← d_{i*} / ‖d_{i*}‖
  t ← ⟨d, v⟩                                          # sweep 3
  v̄ ← mean_i V_w[i]
  return  int4(μ − a_K), int8(v), int8(t), int4(v̄ − a_V)   + four fp16 scales
```

Fixed instruction count, no convergence loop. This is greedy *k*-center, not
Lloyd's: the selector wants the window's extreme point, and *k*-center minimises
exactly the radius a max-style estimate depends on.

### Algorithm 6 — One gated decode step

```
gated_step(q, k_fp, v_fp, store, ratio r):
  if no cards: RAISE        # a Q tier that cannot be selected over is a misconfig
  1  est, logmass ← gate(q, cards, anchors, scale)                 # 2 dots/window
     keep ← top-k( group_max(est), n_sel = ⌈r · n_active⌉ )        # union THEN cap
  2  K_q, V_q ← dequantise + RoPE ONLY the kept windows            # the point
  3  out, key_scores ← fused_kernel(q, [sink‖body‖K_q], [..‖V_q], sel=keep)
  4  body windows ← contiguous window reduce of key_scores
  5  epilogue: c ← Σ exact(kept) / Σ est(kept)                     (M6)
     for w ∉ keep:  score_w ← c·e_w ;  out += c·e_w · v̄_w
     out, scores ← (out, scores) / (1 + Σ_{w∉keep} c·e_w)
```

`r = 1.0` selects every window and is **byte-identical** to attending over the
whole tier — the no-op proof, and the statement that this is a *selection* rather
than a second implementation of attention. In the kernel the selection is passed
as an indirection `SEL` into the Q loop: the tier stays whole and gathered, and the
gate declines to *read* bytes rather than shrinking the store.

### M8 — Compiled-eviction numerics

M1 requires the round trip `s ← fp32(fp16(s))`. Inductor drops it by default and
keeps the wider value in a register. Measured on torch 2.14, CPU Inductor, 24 seeds:
`quantize_key_windows` diverged from eager on 11/24 seeds, `quantize_value_windows`
on 14/24, and the card's symmetric encoder on **24/24**. The fix wraps the single
compile site in `emulate_precision_casts`, entered per call and restored on exit;
all rows go to zero, and a 14-step / 5-eviction end-to-end run is now identical
tensor for tensor — codes, scales, zeros, slot table and every card field.

Landing it also un-blocked the fusion the workaround had been preventing: the
quantiser had carried `@torch.compiler.disable` to dodge an Inductor lowering
failure since fixed upstream, which was costing a ~195 ms eviction step against a
~72 ms steady step — **~15.4 ms/step amortised, 18% of GPU time** — landing as
sixteen separate multi-millisecond elementwise kernels. Undecorated, it traces with
0 graph breaks in 1 graph, and the compiled eviction is now **measured against its
own control arm at ~10% of TPOT** (median +9.8% slower eager, every cell in the
same direction, 6.4–11.2%). That win is host-side, from 36 fewer launches per step;
its GPU kernels are ~1 ms/step *more* expensive than the eager work they replace.

---
---

# Part C — How the core architecture changed, A → B

## C1. The one-sentence version

**Branch A asks "what should the cache keep?" Branch B adds a second, different
question — "what should this step actually read?" — and the two questions have
different inputs, different frequencies, and different homes in the code.**

## C2. The shift, side by side

| | **A — `quant_batched_fixed_int2_eviction`** | **B — `int2_clustered_rep`** |
|---|---|---|
| **Decisions in the system** | one: tier assignment | two: tier assignment **+** per-step read selection |
| **What decides** | accumulated `window_scores` | tier: `window_scores`; read: **the current query** |
| **When** | every `ws` steps | tier every `ws` steps; read **every step** |
| **Where** | `update()` / the eviction body | tier as before; read **inside the attention call**, because `update()` never sees a query |
| **Q-tier read per step** | `O(N_q)` — the whole active tier | `O(card·N_q + r·N_q·win)` — a fixed scan plus a selected read |
| **Per-window frozen artifacts** | codes + pinned grid + position range | **+ card** (`μ`, `v`, `t`, `v̄`) |
| **Attention over the Q tier** | every window attends | selected windows attend exactly; **skipped windows attend through their value centroid** |
| **`window_scores`** | every entry exact | exact where opened; rescaled card estimate where not |
| **Selection granularity** | n/a | per `(row, KV head)`; GQA union before the cap |
| **int2 window price** | 8448 B (fp16 grid) | **6432 B** grid at int8/32-groups; **8608 B** as charged, card included |
| **Budget honesty** | codes + scales only | card in the budget; leftover **enforced** unspendable |
| **Production paths** | 3 env-var forks | **one**; no ungated arm — a card-less fused layer raises |
| **Compiled vs eager** | equal under `aot_eager`; **not** under Inductor | **bit-identical** under Inductor |
| **Compiled eviction's worth** | asserted | **measured against a control arm: ~10% of TPOT** |
| **Reported throughput** | one legacy, prefill-inclusive field | decode (`B/TPOT_steady`) and e2e (`B·gen/(t₃−t₀)`) as separate columns |

## C3. What stayed exactly the same — and why that matters

Everything in Part A's foundation is untouched by Part B, which is what makes the
gate a cleanly separable contribution:

* positions are still never rebased;
* the pinned grid is still pinned, and re-demotion is still an exact reactivation;
* the tier split is still **rectangular** — equal counts across rows and heads — so
  the effective K/V is still dense with no padding and no keep-mask;
* the budget is still split in **bytes**, so what the cache *keeps* at a given `q`
  is decided the same way (the prices moved, the rule did not);
* the fused kernel is still one launch per layer per step; the gate is an
  *indirection* into its Q loop, not a different kernel.

The gate therefore changes **what is read**, never **what is retained** — except
through `window_scores`, which is exactly why M6 exists and is the part of the
design most worth defending in review.

## C4. The structural insight that only shows up in the comparison

In A, the compressed tier's cost was tied to its *size*. A's whole economic
argument (M3) is that 2-bit storage buys ~3.9× more retained windows at the same
bytes — but the read path then paid for all of them, every step, whether or not the
query cared. **Compression bought retention and handed the bill back to decode.**

B was designed to break that coupling, and structurally it does:

$$
\text{retained windows}\;\propto\;\frac{q\,M_{\text{evict}}}{b_q}\quad\text{(accuracy)}
\qquad\qquad
\text{bytes read per step}\;=\;N_q\big(B_{\text{card}}+r\,B_{\text{win}}\big)\quad\text{(latency)}
$$

The lever separates. But the *measured* answer is that this branch has not yet
converted the separation into time, and the reason is contained in the same two
formulas: `B_card/B_win = 0.34` puts break-even at `r* = 0.66`, the gated tier is
only ~13% of what decode reads, and what the gate does buy is currently eaten by
the scattered fetches that per-window selection introduces (12× off roofline).

So the honest one-line comparison is: **A made the cache hold more; B made holding
more stop costing more to read *in principle*, and the remaining work is making
that true in bytes and in address order rather than only in asymptotics.** That is
a normal place for a method to be, and it is more useful in a paper stated that way
than as a claim.

## C5. Status, claim by claim

| claim | status |
|---|---|
| A's kernel/throughput speedup (2.81× TPOT; 2.05× slower → 1.37× faster vs Flash) | **measured on GPU**, A100 |
| A's cross-layer eviction is byte-identical to per-layer | **pinned by tests** across `q`, `L`, `B`, `ws`, sinks, promotion |
| A's LongBench recovery after the RoPE-dtype fix | measured; bundles more than one change, so not a controlled attribution |
| B: card 272 B vs window 804 B/head; break-even 0.66 | **arithmetic**, from `sketch_bytes_per_head` / `bytes_per_q_window` |
| B: the gate's selection saves 0.2–1.6% of TPOT — i.e. ties | **measured** — the `--gate-ratio 1.0` control arm |
| B: the gated kernel is 12× off its roofline; scattered reads are why | **measured** — GPU profile |
| B: int4 cards are accuracy-neutral at ratio 0.25 | **measured on a CPU fixture**, 8 seeds |
| B: int4 cards are *faster* | **not measured** — justified by bytes only |
| B: the compiled eviction is worth ~10% | **measured** against its control arm, 2026-09-20 |
| B: "the read gate is a 24% regression" | **not attributable** — a config change (budget 0.50→0.20, local 64→128) sits inside the same range. A floor, not a measurement |
| Comparison against int2 KIVI or QEvict | **never run — there is no implementation of either.** Nothing here may be written as "beats KIVI" |

Two framing points that belong in any write-up of these numbers:

* **B=1 is not winnable and that is fine.** Every model weight is read once per
  step — 16.06 GB, a ~10.3 ms/token floor on an A100 — and that floor is
  independent of cache size. At B=1 the claim is **memory**, not latency.
* **Across all six shape/batch cells the method beats Flash FullKV in one** (and
  in two at its best-ever operating point). The comfortable win is the long-prompt,
  large-batch cell, which is the cell the method is *for*. Saying that plainly is
  stronger than quoting the headline cell alone.
