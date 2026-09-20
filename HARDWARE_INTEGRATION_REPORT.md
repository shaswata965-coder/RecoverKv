# StickyKV Hardware Integration Report

*2026-09-19*

## Overview

This project (internally called StickyKV) lets a large language model read very long documents without its working memory — the "KV cache" of everything it has read so far — growing without bound and exhausting the GPU. It watches which chunks of text ("windows") the model actually leans on, keeps the important ones at full precision, compresses the moderately useful ones down to 2 bits per number instead of 16, and drops the rest.

Since the batched int2 eviction work landed (the branch this report covers, starting around August 3rd), the project shifted from proving the compression idea correct into making it fast on real GPU hardware: three custom fused kernels, a restructured eviction step that runs across all model layers in one pass, and careful numerical profiling to find out where GPU time actually goes rather than assuming it. The headline result is real: decode (token-generation) speed improved **2.8x** at the target configuration, enough to beat an uncompressed cache on the long-prompt, large-batch shape the method is designed for.

In short: the core compression idea works, it is now fast on real hardware through a small set of purpose-built GPU kernels, and a silent accuracy regression introduced along the way was found and fixed before it could go unnoticed.

## Technical dive

### 1. The algorithm being accelerated

The sequence is chopped into fixed-size windows. Every few generation steps, each window's accumulated attention score (an H2O-style running sum: every query token votes for every key it attends to) determines a three-way ranking: the top windows stay full precision (fp16), the next band is compressed to int2 with a pinned per-window scale and zero-point, and the rest are dropped. This is a correctness-first design: it is expressible in plain PyTorch and was validated on CPU before any of the GPU work below existed. The kernel work in this section is entirely about removing the overhead that a naive implementation of that same algorithm would pay, without changing which tokens the model keeps or the arithmetic that decides it.

### 2. Kernel 1 — recovering a per-key attention score from flash-attention's backward pass

**The problem.** FlashAttention never materializes the full attention matrix; it streams key blocks and keeps a running maximum and running sum *per query row* to compute a numerically stable softmax on the fly. The eviction policy needs the opposite reduction: a sum *per key*, over all queries. Because the correction that keeps the running sum exact is indexed by query row, and query rows are exactly the axis being summed away, there is no way to patch up a per-key running total after the fact — the only way to get it exactly right the naive way is to keep the whole (large) query×key score matrix around, which is precisely what flash-attention exists to avoid.

**The insight.** FlashAttention's *backward* pass, used for training, already computes a reduction with exactly the right shape: for each key block, it accumulates a gradient contribution summed over every query block, `dV_j = \sum_i P_i^{\top} dO_i`. Feeding it a vector of all ones in place of the real output gradient turns that matmul into a plain, unweighted column sum — `dV_j` becomes `\sum_i P_i^{\top} \mathbf{1}`, which is exactly the per-key attention total the eviction policy needs. No new mathematical machinery was invented; an existing, already-correct reduction was repurposed by swapping its input.

**Why the second pass is nearly free.** The naive way to build this second pass would still need the same running-max bookkeeping flash's forward pass uses, because the true softmax normalizer for a query isn't known until every key block has been seen. But by the time this second pass runs, the *first* pass has already finished and already knows each query's exact final log-sum-exp value, `L_i = m_i + \log \ell_i`. Handing `L_i` to the second pass turns the softmax probability into a single direct formula, `P[i,s] = \exp(S[i,s] - L_i)`, with no running maximum, no rescaling, and no risk of overflow (because `S[i,s] \le L_i` always). The only added cost to the *first* pass is keeping that one small per-query number instead of discarding it.

**The two-stage kernel.** Stage A is an ordinary fused attention forward pass, changed only to keep `L` instead of throwing it away (this is what FlashInfer's prefill kernel exposes directly, replacing an earlier, more fragile way of capturing it that some builds silently failed to support). Stage B is a second, smaller kernel with the *loop order reversed* from a normal attention kernel — outer over key blocks, inner over query blocks — because the score being accumulated belongs to keys, not queries. Each program owns one key block, loads it into on-chip memory once, and reuses it across every query block and every attention head that shares it, instead of re-reading it from memory repeatedly the way a naive per-key loop would.

**Skipping known-empty work.** Because attention is causal, a whole tile of (query block, key block) pairs is either entirely visible or entirely invisible — only the tile straddling the diagonal boundary needs masking. The kernel checks this before doing any matrix multiplication, and skips roughly half of all tile pairs outright rather than computing them and masking the result afterward.

**A hardware bottleneck found by budgeting, not guessing.** Once implemented, this kernel was still about 4x less efficient than the attention pass running beside it. Rather than profiling blindly, the three plausible costs — matrix-multiply FLOPs, memory traffic, and the exponentials in the softmax — were each estimated by hand from the kernel's actual shapes. Two of the three were small; the exponential term alone accounted for the overwhelming majority of the measured time. The cause: computing a natural-log exponential costs the GPU roughly ten special-function-unit instructions, while a base-2 exponential is one. The fix folds a constant (`log2 e`) into the softmax scale factor once, at kernel launch, so every exponential in the kernel becomes a base-2 one for free — not by changing anything per element.

### 3. Kernel 2 — collapsing 32 independent layer-eviction passes into one

**The observation.** Every transformer layer evicts on the same schedule, at shapes fixed entirely by configuration (window size, budget, quantized-tier size) rather than by data. So all 32 layers, at a given step, are doing the *identical-shaped* operation independently.

**Why that matters on a GPU.** Each kernel launch carries a largely fixed dispatch cost, independent of how much work it does. Running 32 small, independent, identically-shaped operations back to back pays that fixed cost 32 times for no extra parallelism gained. Any operation that is already row-independent can instead be reshaped so a "layer" is just another row: folding the layer axis into the batch axis (`R = layers × batch` rows, instead of `batch` rows repeated 32 times) turns 32 sequential launches into one.

**The correctness wrinkle this forced.** With one shared store spanning every layer, the eviction for a given step now has to run *before* that step's new token is appended, rather than after — because by the time layer 0 is ready to compact, layers 1–31 have not yet produced this step's token to append. This reordering ("compact, then append" instead of "append, then compact") is safe only because which tokens survive an eviction is decided purely from *already-accumulated* historical scores; the newest, not-yet-scored token always lands in the region that is never evicted regardless of order. This was not just argued but pinned by tests asserting the batched path is byte-for-byte identical to running each layer separately, across many configurations.

**Result.** Eviction-side kernel launches fell by roughly 32x, and the fraction of decode time spent on eviction bookkeeping fell about 4.75x, with the very first compaction (during prompt processing) deliberately kept layer-by-layer to avoid briefly holding all 32 layers' full, uncompressed prompt data in memory at once.

### 4. Kernel 3 — a decode kernel that never writes decompressed keys back to memory

**Why this matters more than it sounds.** Generating one token at a time is memory-bandwidth-bound, not compute-bound: the GPU must stream almost the entire model's weights from memory for every single token, regardless of cache size, and the arithmetic done per byte read is tiny by comparison. Under that constraint, decompressing the int2 cache to fp16 and writing that copy back to memory — even briefly, even just to immediately re-read it for attention — is pure waste: those bytes never needed to exist in memory at all.

**The fused design.** A single kernel reads a window's packed 2-bit codes directly from memory, dequantizes them in on-chip registers using that window's pinned scale and zero-point, applies the rotary positional encoding in-register (using the window's fixed original position, whose cosine/sine values are computed once and reused for the window's whole lifetime, since eviction never renumbers positions), and immediately runs the query·key dot product and softmax-weighted accumulation — all before any decompressed value ever touches memory. The only bytes moved are the small int2 codes and their compact per-window grid.

**Tiling to match the data's own structure.** A kernel tile is sized to exactly one window, so it owns exactly one scale/zero-point pair with no bookkeeping across tile boundaries. Widening a tile to cover several windows per iteration (rather than one window per serial step) let the kernel also emit the per-window attention totals that the eviction step needs directly from an in-register epilogue, instead of a separate reduction pass over the whole sequence — removing an entire second traversal of the data and cutting iteration counts roughly 8x at the shipped window size.

**Hitting, and then working around, a hardware limit.** Widening that tile also multiplies how much operand data the GPU's matrix-multiply units need staged on-chip at once, and an early version of the wider tile exceeded the GPU's shared-memory budget outright — a hard launch failure, not a slow result. Rather than hand-picking one safe tile size, the fix tries the largest, fastest configuration first and steps down through a small ladder of smaller ones until one fits, caching the winning configuration per GPU and per shape so the search only ever runs once. Every rung of that ladder is numerically identical; it is a resource-fitting search, not an approximation.

### 5. A tier-consistency bug caught by tracing dtypes, not by a failing test

One shared utility that builds the rotary-position lookup tables was, by accident, computing them in a different floating-point precision than every other place in the code that applies the same rotation — nothing had depended on that value's precision before, so it went unnoticed. The effect was subtle: a token's key ended up rotated with a slightly different numerical convention depending on which precision tier happened to be holding it when it was read, a purely incidental inconsistency invisible to any test that checks one tier at a time. Matching the precision closed the resulting error by roughly 660x and fully recovered a benchmark regression on long-document question-answering — accepted deliberately even though it cost real decode latency, because it fixed a genuine accuracy defect and the method still beat its speed target after paying for it.

### 6. Net measured effect

Measured end-to-end at the shipped configuration (steady-state decode time per generated token, in seconds, one GPU):

| shape / batch | before any kernel work | + batched eviction (§2) | + fused window-score kernel (§3–§4) | vs. an uncompressed cache |
|---|---|---|---|---|
| 4096-token prompt, batch 32 | 0.176 | 0.0735 | **0.0626** | **1.37x faster** |
| 1024-token prompt, batch 32 | 0.105 | 0.0607 | **0.0488** | slightly slower |

Peak GPU memory did not regress alongside the speed gain (49.2 GB → 48.4 GB at the headline shape). The two kernels together account for essentially the entire 2.8x improvement quoted in the overview; the accuracy fix in §5 cost a small amount of that gain back in exchange for closing the correctness defect.

## Kernel algorithms, shapes, and the throughput math

Each kernel is a strict reduction in bytes moved or launches issued for an *unchanged* arithmetic result — none alters the eviction decision or the attention output. `B`=batch, `H_q`/`H_kv`=query/KV heads, `rep=H_q/H_kv` (GQA group size), `T`=query length, `S`=key length, `D`=head dim, `L`=layers, `ws`=window size.

### Kernel 1 — prefill score kernel

| | |
|---|---|
| grid | `(\lceil S/\text{BLOCK\_N}\rceil,\ B \cdot H_q)` — one program per (key block, batch×query-head) |
| tiles / program | `Q_m \in \mathbb{R}^{\text{BLOCK\_M}\times D}`, `K_n \in \mathbb{R}^{\text{BLOCK\_N}\times D}` (loaded once, reused across all `m`), `L_m \in \mathbb{R}^{\text{BLOCK\_M}}` |
| output | `token\_scores \in \mathbb{R}^{B\times H_q\times S}` |

```latex
\text{acc}[n] \;=\; \sum_{m\,:\,m \text{ causal-visible to } n} \;\sum_{i \in m} \exp_2\!\big(\text{scale}\cdot(Q_i \!\cdot\! K_n^\top)\cdot\log_2 e - L_i \cdot \log_2 e\big)
```

Throughput math: a per-key colsum this way costs one `K` load per key block (reused over every `m`) plus `\lceil T/\text{BLOCK\_M}\rceil` `[\text{BLOCK\_M},\text{BLOCK\_N}]` tiles — versus materializing the full `[B,H_q,T,S]` probability tensor the naive way, an `O(T\cdot S)` memory write the kernel never performs. Swapping `\exp` for `\exp_2` cuts the per-element cost from ~10 special-function-unit ops to 1; measured, this moved the kernel from a 46 ms/step transcendental-bound cost toward its ~7–11 ms matmul/memory floor — a ~4–6x reduction on the term that dominated.

### Kernel 2 — batched cross-layer eviction

| | |
|---|---|
| reshape | `L` stores of `[B,H_{kv},T,D]` → one joint store `[R,H_{kv},T,D]`, `R=L\cdot B`, layer `i` owns rows `[iB,(i{+}1)B)` |
| per-row op | unchanged: `\text{top\_k}(\overline{\text{window\_scores}})` → retained window indices → gather/compact |

Throughput math: the eviction body's launch count goes from `L` dispatches of a `k`-op sequence to `1` dispatch of the same `k`-op sequence over `R` rows instead of `B`. Per-launch dispatch overhead `c` is paid `L\cdot c` times → `c` once; bytes moved (`\propto L\cdot B\cdot T\cdot D`) are unchanged. This is a launch-count argument, not a FLOP argument — measured `32\times` fewer eviction-body launches at `L=32`.

### Kernel 3 — fused int2 decode kernel

| | |
|---|---|
| grid | `(B\cdot H_{kv},)` — one program per (batch, KV head), all `rep` query heads at once, `\text{BLOCK\_R}\geq rep` |
| tiles / program | sink prefix, fp-tier windows, and Q-tier (int2) windows, each streamed `\text{BLOCK\_NW}` windows at a time (`\text{BLOCK\_T}=` pow2-padded `\text{BLOCK\_NW}\cdot ws`) |
| per-tile Q-only step | `\tilde K,\tilde V = \text{dequant}(\text{code},\text{scale},\text{zero})`, rotated in-register by `(\cos,\sin)` at the window's fixed position, before the dot product |

```latex
m \leftarrow \max(m,\ \text{rowmax}(S))\qquad
\ell \leftarrow \exp_2(m_{\text{old}}\!-\!m)\,\ell + \text{rowsum}(\tilde P)\qquad
\text{acc} \leftarrow \exp_2(m_{\text{old}}\!-\!m)\,\text{acc} + \tilde P \cdot V
```

Output: attention `\in \mathbb{R}^{B\times H_q\times D}` plus window sums `\in \mathbb{R}^{B\times H_q\times W_{\text{phys}}}` from the same tile loop (no second pass over `S`). Throughput math: decode is bandwidth-bound (arithmetic intensity `\approx 2` FLOP/byte, far below the GPU's compute/bandwidth ridge point), so wall-clock time tracks bytes moved almost directly. A materialize-then-attend design pays for the int2 codes *and* a full fp16 write-back of the dequantized Q tier (`\approx 4\times` the int2 bytes) every step; fusing dequantization, RoPE, and the dot product into one kernel removes that fp16 round trip entirely — the only bytes crossing HBM are the codes and their compact per-window grid.

### How the three compose into 2.8x

Kernel 1 removes an `O(T\cdot S)` redundant pass from prefill. Kernel 2 divides eviction's fixed per-launch overhead by `L=32`. Kernel 3 removes one full extra HBM round trip from every decode step. They act on disjoint phases (prefill scoring, periodic eviction, per-token decode) and disjoint cost terms (transcendental-bound, launch-bound, bandwidth-bound respectively), which is why their measured gains compound rather than overlap.

## What did not work

- **"Sticky-Q" (one-way demotion) was built, fully tested, and shelved.** An early-August experiment made compression a one-way door — once a window was compressed it could never buy back full precision. It passed its full test suite but was superseded by the two-way promotion design that shipped instead.
- **A silent accuracy regression from a budget-accounting default.** A one-line default change caused the cache to quietly spend under half of its allotted memory budget instead of all of it, shrinking effective context and measurably lowering scores on every one of twelve benchmark datasets. It was root-caused by tracing the metric back through the code history and fixed.
- **A memory-saving micro-optimization caused an out-of-memory crash.** Reusing a shared per-step object to save a fraction of a millisecond created a hidden reference cycle that kept old data alive far longer than intended, causing a long-running benchmark to run out of memory partway through. It was reverted.
- **Running the eviction step layer-by-layer during the very first compaction was rejected** because it would have required holding two full copies of the largest data structure in memory at once, adding several gigabytes of peak memory for no benefit.
- **The alternate route to the log-sum-exp value carried a silent risk.** Reusing the softmax normalizer from a third-party attention library was, for five days, preferred automatically whenever that library was installed — but that library's convention for the value (natural log versus base-2 log) is a per-build detail, and using the wrong convention without noticing would have corrupted every eviction decision in a run while still producing plausible-looking output. The automatic preference was turned off before this was ever confirmed to have happened.
- **A hand-rolled low-level kernel-launch shortcut was tried and abandoned.** Bypassing the standard kernel-launch wrapper to shave off its argument-count overhead was tested, but the underlying launch API changes between GPU-kernel-compiler versions and could not be validated without a GPU on hand — too fragile to justify against a small, unconfirmed gain.

## What's implemented now — impactful changes

- **2.8x faster decode** at the target configuration, enough to beat an uncompressed cache by 37% on the long-prompt, large-batch shape the method is built for.
- **Cross-layer batched eviction**: every layer's cache-compaction step now runs as one combined operation instead of 32 separate ones, cutting eviction-related GPU launches by roughly 32x.
- **A fused decode kernel** that unpacks, positions, and attends over compressed keys entirely inside GPU registers, eliminating a separate decompress-then-write-back step.
- **A fused prefill scoring kernel**, built on flash-attention's own backward-pass math, that removed a redundant full attention-sized computation from every prompt processed — recovering most of the speed gap to an uncompressed cache during prompt processing.
- **A rotary-position accuracy fix** that closed a benchmark score regression across most tested datasets, caused by a token's positional encoding subtly depending on which precision tier it was stored in.
- **A fixed budget-accounting bug** that had been silently letting the cache use less than half of its allotted memory, which was suppressing quality scores across every benchmark dataset tested.
- **Automatic, safe GPU tuning**: kernels now pick their own launch configuration and back off automatically when a setting doesn't fit the GPU's on-chip memory, rather than crashing or needing to be hand-tuned per GPU.
- **Fail-fast multi-GPU handling**: configurations that would silently produce a meaningless timing number (because the model is split across GPUs) now stop with a clear explanation instead of quietly reporting an invalid result.
- **Every run now records which kernels and settings were actually active** (a captured environment snapshot alongside each result), so a reported number can always be traced back to the exact code path that produced it.
