# Decode speed plan — remaining work

Cross-layer batched eviction and the windowed-score kernel rewrite are landed
and default-on. What they cost, what they bought, and everything that went wrong
getting there is in `DECODE_HISTORY.md`. **This document is only what is left.**

## 0. The cell, stated exactly

Llama-3.1-8B (L=32, H_q=32, H_kv=8, D=128, 16.06 GB fp16), prefill 4096,
B=32, `ws=8`, `num_sink=5`, `local_window=64`, budget 0.50, q=0.70.
**One GPU** — `perf_runner` and `run_perf_table.sh` both refuse a sharded model
now (a pipeline-sharded decode step times the interconnect, not the cache).

```
top_k = 256 windows  ->  n_q = 179 int2 windows (1432 tok) | k_fp = 77 fp windows
S_fp  = 5 sink + (77 + 8 local) * 8 = 685 tok        S = 2117 keys
bytes/token/layer:  fp16 4096 B    int2 1056 B (25.8%)
                    [K codes 256 | K scale+zero 512 | V codes 256 | V scale+zero 32]
```

**Note the second line.** int2 gives 3.9× compression, not 8×, because the
per-`(window, head, dim)` fp16 scale+zero grid (512 B/token) is **twice the K
codes it describes** (256 B/token). That is a `ws=8` group-size artifact, priced
in §4.5.

## 0.1 Correctness invariants — non-negotiable

Every change below is a launch, layout or scheduling change. None may alter what
the cache holds. Each invariant is enforced by a test that fails loudly:

1. **The retained set is a function of `window_scores` alone.** Not of when the
   eviction runs relative to the append. Pinned by
   `tests/test_layer_major_evict.py` — whole-cache byte-identity against the
   per-layer path across `q ∈ {0, 0.5}`, `L ∈ {1,3}`, `B ∈ {1,2}`,
   `ws ∈ {1…128}`, sinks on and off, promotion/dormancy.

2. **`cache_position` is the absolute token index, and it is verified.** An
   evicting cache breaks a transformers assumption — see `DECODE_HISTORY.md` §2.
   `WindowedCache._check_position_contract` refuses a wrong base; free until the
   retained count and the absolute index diverge, then one sync, once per cache.

3. **No silent correctness fallbacks.** A path that cannot produce the right
   answer raises. Tuning searches that are numerically identical at every rung
   (the shared-memory tile ladder) are not fallbacks and may proceed, but must
   announce which rung they took.

## 1. Where we are

`run_perf_table.sh`, one GPU, shipped defaults:

| shape / B | ours | Flash (FullKV) | |
|---|---|---|---|
| **4096/256, B=32** | **0.0626** | 0.086 | **1.37× faster** |
| 2048/512, B=32 | 0.0531 | 0.055 | 1.04× faster |
| 1024/1024, B=32 | 0.0488 | 0.043 | 1.13× slower |
| 4096/256, B=1 | 0.0471 | 0.025 | 1.88× slower |
| 2048/512, B=1 | 0.0468 | 0.024 | 1.95× slower |
| 1024/1024, B=1 | 0.0467 | 0.024 | 1.95× slower |

| target @ 4096/B=32 | | status |
|---|---|---|
| beat Flash | ≤ 0.086 | **met** (1.37×) |
| 20% faster than Flash | ≤ 0.0688 | **met** (0.0626) |
| stretch | ≤ 0.050 | gap **12.6 ms** |
| approach SnapKV | ~0.033 | gap **29.6 ms** |
| beat KIVI-int2 @ ~1024/B=32 | < 0.044 | gap **4.8 ms** (0.0488) |

**The two headline targets are met.** What is left is three separable problems:

- **B=1 is ~1.9× slower than Flash at every context length**, and flat across
  context — the classic fixed-per-step-cost signature. §4.1 (split-K) is the
  lever: the grid is `B·H_kv`, which is **8 programs on a 108-SM A100 at B=1**.
- **Short context at B=32** (1024: 1.13× slower). Weight-bound regime, so the
  per-step fixed costs dominate — same lever as B=1, plus §4.3.
- **The 12.6 ms to the stretch target** at the headline cell.

### The roofline, so the remaining targets are bounded honestly

| | per-step traffic | @2.04 TB/s |
|---|---|---|
| FullKV | 16.06 GB weights + 17.18 GB KV = 33.2 GB | 16.3 ms |
| ours | 16.06 + 4.42 GB KV ≈ 20.5 GB | 10.0 ms |

We are at 62.6 ms against a ~10 ms roofline — **6.2× off**. Flash is 5.3× off its
own. So we are now close to the harness's own efficiency, and the residue is
largely HF's per-layer Python, which every method in the table pays and which
only §5 removes (with the fairness caveat there).

## 2. Stage 0 — the profile that has still never been run

Every remaining ms estimate below is a launch/traffic count with an unverified
translation to time. The window-score rewrite is the cautionary tale: it was
sized at ~0.5 ms from traffic and delivered **~11 ms**, because the value was in
launches and serial-iteration latency, neither of which was measured. Do not
repeat that.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_decode.py \
    --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 32 --steps 24 --warmup 12 --trace outputs/decode.json
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_decode.py \
    --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 1 --steps 24 --warmup 12
```

Read, in order:

1. **GPU busy %** at B=1 and B=32. B=1 being flat across context says a fixed
   cost still dominates there; the profile says whether it is host or an
   under-occupied kernel (§4.1 vs §4.3).
2. **CPU self time, top 20.** If `JITFunction.run` still dominates, the launcher
   bypass (`kernel.warmup` + a cached `CompiledKernel`) is worth the version
   risk; the argument-count cut reduced 50 args to 41 but left the launcher.
3. **The chosen tile rung.** The kernel prints
   `[StickyKV] fused decode tiling: target_keys=… num_stages=…`. At
   `target_keys=64` the Q-tier tiling got its full 7.8×; at 32 it got 4×; at 16,
   2×. If it is not 64, raising it is free performance — §4.4.
4. **`STICKYKV_COMPILE_EVICT=1` vs `=0`.** If small, §3.3.

## 3. The eviction — what's left

### 3.1 Per-device grouping (multi-GPU is currently refused)

`device_map="auto"` shards by layer, so on a multi-GPU box layers `0..k` sit on
`cuda:0` and the rest elsewhere. The layer-major join concatenates all `L` layers
into one row axis and cannot span that; both `perf_runner` and
`run_perf_table.sh` now refuse a sharded model outright, for measurement-validity
reasons as much as correctness (`DECODE_HISTORY.md` §4).

The fix is to **partition layers by device and build one joint store per group**,
looping the batched eviction over groups. On 8 GPUs and 32 layers that is 4
layers/group — a 4× launch cut instead of 32×, so it is worth far less than it
sounds, and it only matters for a model that genuinely does not fit on one card.
Llama-3.1-8B peaks near 48 GB at the largest cell, so it does not.

**Not worth doing for this model.** Revisit only for one that needs sharding.

### 3.2 Decouple the eviction period from `window_size`

`policy.py:92` hard-codes the cadence to `step % window_size == 0`. Add
`eviction_interval` (a multiple of `ws`, default `ws`). At `4·ws` the amortized
eviction cost drops **4×**, for +24 tokens/row of peak KV (+1.2% on a 2048-token
budget).

This is **not** a `window_size` change. `ws` sets which tokens are *scored
together*, and shrinking it is what degrades H2O (`no-observation-window-scoring`).
`eviction_interval` leaves `ws` alone and only changes *when* the same per-window
scores are acted on — between evictions the cache holds strictly more context and
each decision rests on 4× more accumulated evidence.

**It changes outputs.** Quality should be ≥, not =. Gate on LongBench + GSM8K at
`interval ∈ {ws, 2ws, 4ws}` before shipping a non-default.

### 3.3 Close the `_affine_quantize` graph break

If Stage 0.4 shows the compiled eviction buys little, fix the break rather than
route around it: compute the affine range on a **gathered, materialised** tensor
so the reduction's read index is not a `StarDep`, or write the range reduction as
a small Triton kernel. The `aot_eager` eviction test already pins byte identity.
Applies unchanged to the batched body — only its row count is wider.

## 4. The fused decode kernel — what's left

### 4.1 Split-K — the B=1 and short-context lever

`grid = (B * H_kv,)` is **8 programs on a 108-SM A100 at B=1**, and 256 (2.4
waves) at B=32. The kernel's own docstring has flagged split-K over the sequence
as "not yet" since it was written.

This is now the **highest-value remaining item**: B=1 is 1.9× slower than Flash
at every context length and flat across context, which is exactly what an
under-occupied grid plus a fixed per-step cost looks like. Emit partial
`(out, lse)` per KV-split and combine. Nothing about the window-score epilogue
blocks it — window sums combine across splits with the same `exp(m − lse)`
rescale the tiles already use.

### 4.2 `BLOCK_R = 16` for `rep = 4`

`_pow2_at_least(4) = 16`, so **every `tl.dot` is 25% useful rows**. Either pack 4
KV heads into one `BLOCK_R=16` tile (grid becomes `B·H_kv/4`) or drop to an FMA
reduction for the Q tier. Interacts with §4.1 — both change the grid, so decide
them together.

### 4.3 The "free ones" — reviewed, and two of them are not

They were written against the **pre-rewrite** kernel and only two survive review.

**(a) `tl.exp` → `exp2` — do it, but it is ~0.42 ms, not 0.75, and not free.**
The base-2 fold works with the window epilogue: put `log2(e)` into `scale`, keep
`m`/`wmax` in base-2 units, and use `lse = m + log2(l)` — `p`, `l`, `acc` and
`out` are then bit-for-bit the same quantities, and the epilogue's
`exp2(wmax − lse)` equals `exp(m_tile − lse)`. Verified on CPU (max rel
`1.4e-6`, window sums still normalise to 1).

Two corrections to the old entry. The exponential count is **78 M/step, not
139 M** — the rewrite deleted the second pass over `S`, so the count fell 1.78×
and the saving with it. And it is *not* numerically free: `1.4e-6` max relative
is ~10× the window-score rewrite's `1e-7` and the same order as the smallest
observed ranking gap (`8e-6`), so it needs a quality check, not just a tolerance
test.

**(b) `cos`/`sin` in fp16 — do it, but for consistency, not bandwidth.**

The bandwidth claim is right (0.75 → 0.38 GB/step, ~0.18 ms) and it may buy a
bigger tile rung (§4.4). But fp16 `cos`/`sin` perturbs a rotated key by ~`5e-4`
mean relative — **~5000× the window-score change and ~60× the ranking gap.** On
those numbers alone it would be a bad trade.

It is still the right change, because **the kernel is currently the odd one out**:

| path | `ref` passed to the rotary | dtype |
|---|---|---|
| HF's own RoPE for the fp tier | the fp16 hidden state | **fp16** |
| `unrotate_key_window` on demote | the fp16 key (`_rope_cos_sin(rope, k, …)`) | **fp16** |
| `rotate_key_window` on promote | the fp16 key | **fp16** |
| **the decode kernel** (`rope_cos_sin_halves`) | `torch.empty(1,1,1)` → default dtype | **fp32** |

So the Q tier is un-rotated in fp16, stored, and then re-rotated by the kernel in
fp32 — the round trip does not return the key that was stored. Measured
round-trip error on a window of 512 positions:

| kernel `cos`/`sin` | mean rel | max rel |
|---|---|---|
| fp16 (matched) | 2.1e-04 | **6.9e-04** |
| fp32 (current) | 3.4e-04 | **4.5e-01** |

Matching the kernel to fp16 cuts the **max** round-trip error ~660×. Today a
token's key differs depending on which tier it is in, which is not a property a
two-tier cache should have — the tiers are meant to be interchangeable
representations of the same key. `torch.empty(1, 1, 1)` picking the global
default dtype looks incidental rather than intended.

> **This is worth a look from the accuracy side independently of speed.**
> `ACCURACY_RECOVERY_PLAN.md`'s causes B–D are still unmeasured, and a
> tier-dependent key representation at `q > 0` is exactly the shape of thing that
> would cost LongBench points without showing up in any equivalence test — every
> test here compares the kernel against an oracle built with *the same* fp32
> convention, so none of them can see it.

**(c) Drop the `q_proj` stash hook when fused is active — NO. The claim is wrong.**
The score hook's early return is gated on `hidden_states.shape[1] == 1`
(`hooks.py:494`), so it only fires on **decode** steps with a non-empty Q tier.
**Prefill (`T > 1`) falls through and consumes the stash** at `hooks.py:553`.
Removing the hook would break prefill scoring outright. It could be made
conditional inside the hook, but the prize is one dict store per layer per step
plus ~8 MB of live tensor at B=32 — far below measurement noise. **Struck.**

**(d) Preallocate the `set_pending` dict — not worth it.** Measured: 177 ns to
build the 7-key dict, ×32 layers = **5.7 µs/step, 0.009% of a 62.6 ms step.**
There are also two call sites, and mutating a shared dict trades that for
aliasing risk. **Struck.**

**Revised total for §4.3: ~0.6 ms** (0.42 + 0.18), both items carrying a
numerical change that needs a quality run — not the "~1 ms, costs nothing" the
old entry claimed.

### 4.4 Raise the tile rung if Stage 0 says it was capped

The shared-memory ladder picks the largest tile that fits. If it landed below
`target_keys=64`, the Q-tier loop is doing 2–4× more serial iterations than the
tiling intended. Options, cheapest first: `cos`/`sin` in fp16 (§4.3 — halves
those operands), stage `vv` in fp16 for the `tl.dot`, or split the fp and Q tiers
into separate `BLOCK_T` constexprs so the memory-hungry Q tier can shrink alone.

### 4.5 Priced but not recommended: the quantization grid

Per-`(window, head, dim)` fp16 scale+zero is 512 B/token against 256 B of K
codes. A 4-window group (G=32, KIVI's setting) would cut int2 to 672 B/token —
**36% more retained tokens at equal bytes, or 36% less Q-tier traffic at equal
tokens.** But it couples four eviction windows into one quantization group, which
the write-once store (design §10) keeps independent, and it changes quantization
error directly. An accuracy decision, not a perf one.

## 5. CUDA graphs, and the fairness problem

The decode loop is a manual `model(...)` loop with no per-step sync, so it is
graph-capturable in principle: `_reslice` already views a preallocated buffer, and
at steady state the eviction leaves `W`, `k_fp`, `n_q` constant, so the
non-eviction steps have static shapes. Missing: tensor-valued lengths (the kernel
takes `Sfp` as a Python int it specializes on) and a fixed-capacity K/V view with
masking.

**Do not do this to win the table.** It removes overhead every other row still
pays, and a win sourced from that is not a KV-cache result. If done, it must be
offered to every method in the sweep and reported as a separate harness column.

## 6. What each remaining change does to the scores

| change | effect | gate |
|---|---|---|
| §3.3 close the graph break | none | existing `aot_eager` eviction test |
| §4.1 split-K | fp-level (softmax reassociation across splits) | tolerance vs `two_tier_window_reference` |
| §4.2 `BLOCK_R`, §4.4 tile rung | none (shape only) | `tests/test_decode_kernel.py` |
| §4.3 `exp2` | ~2 ulp, as in prefill | decode analog of `tests/test_score_exp2.py` |
| §4.3 `cos`/`sin` fp16 | fp16 rounding on a rotation | tolerance + one LongBench dataset |
| **§3.2 `eviction_interval`** | **changes outputs** | **full LongBench + GSM8K at `{ws, 2ws, 4ws}`** |
| **§4.5 quant group size** | **changes quantization error** | **full LongBench** |

The default configuration must stay score-identical. Note that the quality
runners override `cache.quant_ratio` on the command line
(`KAGGLE_RUNBOOK.md:159`), so they **do** exercise the fused kernel — the YAML's
`0.0` is not what runs.

## 7. Order of work

| # | item | risk | expected |
|---|---|---|---|
| 1 | Stage 0 profile (§2) | none | the only thing that can size items 2–4 honestly |
| 2 | §4.1 split-K | fp-level | the B=1 and short-context deficit; highest value |
| 3 | §4.3 (a) `exp2` + (b) fp16 `cos`/`sin` | **fp-level, needs a quality run** | ~0.6 ms + 0.4 GB; (b) also fixes a tier-dependent key representation, and may unlock §4.4. (c) and (d) struck on review |
| 4 | §4.4 tile rung, §4.2 `BLOCK_R` | none | recovers the Q-tier tiling's full 7.8× if capped |
| 5 | §3.2 `eviction_interval` (opt-in) | **outputs** | 4× on the eviction, + quality run |
| 6 | §3.3 graph break | none | only if Stage 0.4 says so |
| 7 | §4.5 quant grid / §5 CUDA graphs / §3.1 multi-GPU | **outputs / fairness / scope** | decisions, not tasks |

## 8. Housekeeping

- `DECODE_PLAN.md` is superseded (its cell is budget 0.20; the table runs 0.50 /
  q=0.70). Delete it or mark it stale.
- `reports/decode_throughput_benchmarks.html` predates every change here and its
  second memory column is `peak_decode_steady_mb` (device-used, can exceed the
  peak-allocated figure), which the caption misdescribes. Regenerate before
  sharing it again.
