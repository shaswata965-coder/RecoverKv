# Decode speed plan — get competitive without touching the scores

`PREFILL_SPEED_PLAN.md` closed prefill to 1.09–1.14× FullKV. Decode is still
**2.05× slower than FullKV and 5.3× slower than SnapKV** at the headline cell.

This plan replaces `DECODE_PLAN.md`, whose arithmetic was written against a
`cache_budget=0.20` / 20%-of-keys cell that **is not the cell the table
benchmarks** — `scripts/run_perf_table.sh` defaults to `CACHE_BUDGET=0.50`,
`QUANT_RATIO=0.70`, `QUANT_MODE=tokens`. Every number below is re-derived at
that geometry.

## 0. The cell, stated exactly

Llama-3.1-8B (L=32, H_q=32, H_kv=8, D=128, 16.06 GB fp16), prefill 4096,
B=32, `ws=8`, `num_sink=5`, `local_window=64`, budget 0.50, q=0.70.

```
top_k = 256 windows  ->  n_q = 179 int2 windows (1432 tok) | k_fp = 77 fp windows
S_fp  = 5 sink + (77 + 8 local) * 8 = 685 tok        S = 2117 keys
bytes/token/layer:  fp16 4096 B    int2 1056 B (25.8%)
                    [K codes 256 | K scale+zero 512 | V codes 256 | V scale+zero 32]
```

**Note the second line.** int2 gives 3.9× compression, not 8×, because the
per-`(window, head, dim)` fp16 scale+zero grid (512 B/token) is **twice the K
codes it describes** (256 B/token). That is a `ws=8` group-size artifact and it
is priced in §5.7, not assumed away.

## 1. Where the time goes — the table already answers it

| shape / B | ours | Flash | ratio | ours − Flash |
|---|---|---|---|---|
| 1048 / 1 | 0.1016 | 0.024 | 4.23× | 77.6 ms |
| 2048 / 1 | 0.1024 | 0.024 | 4.27× | 78.4 ms |
| 4096 / 1 | 0.1011 | 0.025 | 4.04× | 76.1 ms |
| 1048 / 32 | 0.1045 | 0.043 | 2.43× | 61.5 ms |
| 2048 / 32 | 0.1142 | 0.055 | 2.08× | 59.2 ms |
| **4096 / 32** | **0.1760** | **0.086** | **2.05×** | **90.0 ms** |

Two sensitivities settle the diagnosis:

```
1048 -> 4096 at B=1 :  ours  -0.5%      flash  +4.2%
   B=1 -> B=32 @1048:  ours  +2.9%      flash +79.2%
```

**A 32× increase in batch costs us 2.9%.** A 4× increase in context costs us
nothing. The GPU is idle. There is a **fixed ~101 ms/step** that depends on
neither B nor S; Flash's own harness floor is ~24.5 ms, so

> **our own host overhead is 76.6 ms/step = 2.39 ms per layer per step.**

Only at 4096/B=32 does GPU work finally surface: 176 − 101 = **75 ms of exposed
GPU** for 4.42 GB of KV, against Flash's 62 ms for 17.18 GB — **4.7× less
efficient per byte** than the flash decode path.

So the excess is two independent terms, and they are close to equal at the
headline cell: **77 ms host + 75 ms GPU.** `DECODE_PLAN.md`'s Fact B (a roughly
even split) survives the re-derivation. Its Fact A ("decode cost barely depends
on cache size") does not: it is true only at B=1, where the host floor hides
everything. At B=32 the cost is **+68%** from 1048 to 4096.

### The roofline, so the target is bounded honestly

| | per-step traffic | @2.04 TB/s |
|---|---|---|
| FullKV | 16.06 GB weights + 17.18 GB KV = 33.2 GB | 16.3 ms |
| ours | 16.06 + 4.42 GB KV = 20.5 GB | 10.0 ms |
| ours + avoidable overhead traffic (§1.1) | 22.4 GB | 11.0 ms |

Measured: FullKV 86 ms (**5.3× off**), ours 176 ms (**17.5× off**). The 5.3× is
the harness — per-layer Python, no CUDA graphs — and every method in the table
pays it. The gap from 5.3× to 17.5× is ours alone, and it is what this plan
removes.

### 1.1 Traffic we pay for nothing

| | per layer / step | per step | share of our KV |
|---|---|---|---|
| `scores` fp32 `[B,H_q,S]`, written + re-read + rewritten + reduced | 34.7 MB | **1.11 GB** | 25% |
| `cos`/`sin` fp32 `[B, n·ws, D/2]` ×2 | 23.5 MB | **0.75 GB** | 17% |
| | | **1.86 GB** | **42%** |

Both are pure overhead. `decode_kernel.py:353` allocates the score buffer in
**fp32 at token granularity**, `decode_kernel.py:302` reads the whole thing back
for a second normalizing pass, and `scorer.py:93` reads it a third time to
reduce `S → W`. The RoPE tables (`decode_kernel.py:124`) are fp32 because
`rope_cos_sin_halves` builds them off `torch.empty(1,1,1)` (default dtype), and
they are **frozen between evictions** yet re-read every step.

## 2. The launch budget, and what each fix is worth

`perf_runner.py:1217-1226` counted ATen dispatches through `update()`:
273 launching ops/layer on an eviction step, 9 on a normal step. Adding the ~10
the fused epilogue issues *outside* `update()` (`flash_decode.py:139` →
`scorer.py:93` → `accumulate`):

```
eviction step   273 ops/layer x 32 = 8,736 launches, every ws=8 steps
normal step      19 ops/layer x 32 =   608 launches, every step
amortized                            1,700 launches/step   <- eviction is 64%
```

76.6 ms / 1,700 = **45 µs per launching op.** That is 5–10× a bare
`cudaLaunchKernel`, and it is the single most important thing Stage 0 must
confirm or refute — it says the cost is Python / dispatch / Triton launcher, not
the driver.

| after | launches/step | host cost @45 µs |
|---|---|---|
| today | 1,700 | 76.6 ms |
| + cross-layer eviction (§4.1) | 642 | 28.9 ms |
| + score epilogue in the kernel (§5.1) | 226 | 10.2 ms |
| + `eviction_interval = 4·ws` (§4.2) | 201 | 9.1 ms |

**Landing arithmetic.** Host 76.6 → ~9 ms and GPU 75 → ~18 ms (flash-class
efficiency on 4.42 GB) gives **24.5 + 9 + 18 ≈ 51 ms** at 4096/B=32 — 1.7×
faster than FullKV and 1.5× off SnapKV. At B=1 the same work gives ~0.033 s,
which beats KIVI (0.043) and lands within 30% of FullKV.

## 3. Stage 0 — the one run that gates everything

`scripts/profile_decode.py` and `scripts/audit_e2e.py` exist and are unblocked.
Run **at the shape the table reports**, not at B=1:

```bash
python scripts/profile_decode.py --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 32 --steps 24 --warmup 12 --trace outputs/decode.json
```

```bash
python scripts/profile_decode.py --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 1 --steps 24 --warmup 12
```

Read four things, in order:

1. **GPU busy %.** §1 predicts ~10% at B=1 and ~45% at B=32. If B=1 comes back
   >70%, the host-bound reading is wrong, §5 is the whole plan, and §4.1 is
   wasted work — stop and re-plan.
2. **CPU self time, top 20.** If `JITFunction.run` / Triton frames dominate,
   §5.2 moves to the front. The profiler cannot name Python that isn't an ATen
   op, so pair it with a `cProfile` over 24 steps — that is the tool that answers
   "45 µs of *what*".
3. **`STICKYKV_COMPILE_EVICT=1` vs `=0` at this shape.** `run_perf_table.sh`
   defaults it on and `perf_runner.py:1242` force-enables it on CUDA, so every
   published number already used it. If the A/B is <5%, the
   `torch._dynamo.disable` on `_affine_quantize` (`quant/quantizer.py:48-61`) is
   breaking the graph once per layer per eviction, and the compiled path is
   issuing eager launch counts under a compiled name — §4.3.
4. **Eviction wall time per call, split from the normal step.** Everything in §4
   is sized off it.

Do not start §4 or §5 before this returns. That is the discipline the eviction
compile skipped, and it cost a campaign.

## 4. Stage 1 — the eviction (64% of the launch budget)

### 4.1 Batch the eviction across layers — the single biggest lever

> **LANDED (default on).** `WindowedCache` folds `L` into the row axis after the
> first eviction: one `[L·B, …]` state and one `[L·B, …]` slot table, so all `L`
> layers evict in one pass. Measured on CPU at `L=32, B=4, ws=8`, counting ATen
> dispatches through a whole decode step:
>
> | | per-layer | layer-major | |
> |---|---|---|---|
> | the eviction body itself | 13,824 | 426 | **32.5×** |
> | eviction step, all 32 layers | 17,664 | 1,549 | 11.4× |
> | normal step, all 32 layers | 3,840 | 1,123 | 3.4× |
> | **amortized per decode token** | **5,592** | **1,177** | **4.75×** |
>
> The eviction body hits the predicted 32×. The step-level ratios are lower
> because what remains is irreducibly per layer (each layer's K/V write, its
> return slice, its `set_pending`). These are *dispatch* counts on the CPU
> materialize path, not GPU launch counts on the fused path — the ratio is the
> transferable part; the wall-clock number needs `run_perf_table.sh` on a GPU.
>
> The ordering change §4.1 requires (compact, then append) is pinned byte-for-byte
> against the per-layer path by `tests/test_layer_major_evict.py` — the whole
> cache, every step, at `q ∈ {0, 0.5}`, `L ∈ {1, 3}`, `B ∈ {1, 2}`, with and
> without sink tokens, and across a fixture that exercises promotion/dormancy.
> `STICKYKV_LAYER_MAJOR_DECODE=0` is the control arm, not a fallback.
>
> **Prefill is deliberately NOT layer-major** (`STICKYKV_PREFILL_LAYER_MAJOR`,
> default off). A layer-major prompt buffer would force the first compaction to
> hold the whole `L·B`-row prompt and the whole `L·B`-row steady store at once —
> about +4 GB of peak at 4096/batch-32, straight out of max batch. Keeping prefill
> per-layer and joining *after* the first eviction, once every prompt buffer has
> been released, costs nothing: the join peaks at two steady-state stores, well
> under the prompt peak it replaces.

All 32 layers evict **on the same step, at identical shapes** (`ws`, budget,
`n_q`, `k_fp` are config, not data). Today that is 32 sequential Python calls of
273 ops each. Stack the per-layer state into `[L, B, …]` and it becomes **273 ops
total: a 32× cut on 64% of the budget** (1,700 → 642 launches/step), and it
changes no math whatsoever.

The blocker is ordering: `_evict_two_tier` is called from `update()`
(`cache.py:673`) *inside* layer *i*'s forward, so layer 0's eviction must finish
before layer 0 attends. The unlock is to **move eviction to the end of the
step**: after layer 31's forward at step *t*, evict all 32 layers at once, for
use at step *t+1*.

Why that is the same decision: at step *t*, `update()` appends token *t*, then
accumulates the scores the *previous* step's attention wrote, then evicts. Doing
it after layer 31 of step *t−1* uses the same scores over the same scored
windows; the only difference is that token *t* is not yet appended — and token
*t* is the newest token, which lives in the local window and is never evictable.

**Hypothesis: the retained set is identical.** That is testable, not assumable —
gate it on byte-identity of `retained_idx` / `new_tier` across a 64-step run at
`ws=8`, alongside the existing `aot_eager` eviction test.

Cost: a real refactor of `WindowedCacheState` and `QuantizedStore` to a
layer-major layout. It is the largest engineering item here and the largest
payoff. Do §4.2 and §5 first if time is short — they are cheaper per ms.

### 4.2 Decouple the eviction period from `window_size`

`policy.py:92` hard-codes the cadence to `step % window_size == 0`. Add
`eviction_interval` (a multiple of `ws`, default `ws`, so nothing changes unless
set). At `4·ws` the amortized eviction cost drops **4×**, for +24 tokens/row of
peak KV (+1.2% on a 2048-token budget).

This is **not** the `window_size` change `DECODE_PLAN.md` §1c rules out, and the
distinction matters. `ws` sets which tokens are *scored together*, and shrinking
it is what degrades H2O (see `no-observation-window-scoring`). `eviction_interval`
leaves `ws` at 8 — scoring is untouched — and only changes *when* the same
per-window scores are acted on. Between evictions the cache holds strictly more
context, and each decision is made on 4× more accumulated evidence.

**It changes outputs.** Quality should be ≥, not =, but "should" is not a
result: gate on LongBench + GSM8K at `interval ∈ {ws, 2ws, 4ws}` before shipping
a non-default.

### 4.3 Close the `_affine_quantize` graph break

If Stage 0.3 shows the compiled eviction buys <5%, fix the break rather than
route around it: compute the affine range on a **gathered, materialised** tensor
so the reduction's read index is not a `StarDep`, or write the range reduction as
a small Triton kernel. The `aot_eager` eviction test already pins byte identity.

Lower priority than §4.1 — cross-layer batching makes the compiled body's launch
count largely irrelevant — but it is cheap, and it unblocks compiling the batched
body later.

## 5. Stage 2 — the fused decode kernel

### 5.1 Emit window scores, not token scores (host **and** GPU)

`decode_kernel.py:235-310` writes `[B,H_q,S]` fp32 logits to HBM, reads them back
for a normalizing second pass, and `scorer.py:93` reads them a third time to
reduce `S → W` and scatter by `order`. Four passes over 8.67 MB/layer.

The kernel already holds the logits in registers, and windows are `ws`-aligned
with `BLOCK_N % ws == 0`, so **each window lives entirely inside one tile.** Per
tile, store the per-window partial sum `Σ exp(logit − m_tile)` *and* `m_tile`;
the epilogue rescales by `exp(m_tile − lse)`. Traffic drops from `4·S` to
`5·W = 5·S/8` — a **6.4× cut, 1.11 GB → 0.17 GB/step** — and the `order` gather
plus the `+=` into `state.window_scores` fold into the same epilogue, removing
~13 launches/layer/step (**416/step**).

Numerics: identical summands, different association in the online-softmax
rescale. The bar is fp tolerance against `two_tier_decode_reference`, same as the
prefill `exp2` change.

### 5.2 Stop paying Triton's launcher 32× per step

`_decode_triton` (`decode_kernel.py:319`) passes **50 arguments**, 25 of them
`.stride()` calls made fresh on every launch. Triton's `JITFunction.run` hashes
and specialization-checks every one. Fix: contract the tensors to contiguous,
pass shapes instead of strides, and cache the compiled handle (`kernel.warmup(…)`
once, then launch the `CompiledKernel` directly). Sized by Stage 0.2 — if
`JITFunction.run` is not in the top CPU frames, skip this.

### 5.3 Tile the Q-tier loop

`decode_kernel.py:260` runs **one window (8 keys) per serial iteration** —
`n_active = 179` dependent iterations, against the fp tier's 11 (`BLOCK_N=64`
over `S_fp=685`). Every iteration is a full online-softmax dependency: 4
scale/zero loads, 2 code loads, 2 fp32 `cos`/`sin` loads, 2 tiny `tl.dot`s, an
`exp`, a store.

Tile `BLOCK_NW` windows per iteration (8 windows → 64 keys): **179 → 23
iterations**, and the `tl.dot` goes from `[16,64]×[64,16]` to `[16,64]×[64,64]`.

### 5.4 `BLOCK_R = 16` for `rep = 4`

`decode_kernel.py:372`: `_pow2_at_least(4) = 16`, so **every `tl.dot` is 25%
useful rows**. Either pack 4 KV heads into one `BLOCK_R=16` tile (grid becomes
`B·H_kv/4`) or drop to an FMA reduction for the Q tier. Do it after §5.3 — that
fixes the N dimension, this fixes M.

### 5.5 The free ones

- **`tl.exp` → `exp2`.** Three sites (`decode_kernel.py:245,246,308`). Same SFU
  fix as commit `17d866f`; fold `log2(e)` into `scale` and the `[BLOCK_R]` LSE
  vector, never into the `[BLOCK_R,BLOCK_N]` tile. ~139 M exponentials/step at
  this cell — worth ~0.75 ms, and it costs nothing.
- **`cos`/`sin` in fp16.** `rope_cos_sin_halves` returns fp32 only because `ref`
  is fp32. **0.75 → 0.38 GB/step** for a `.half()`, with the kernel's math still
  in fp32.
- **Drop the `q_proj` stash hook when fused is active** (`hooks.py:424`). On the
  fused path the score hook returns before consuming it (`hooks.py:494`), so it
  is a dict store and a live tensor per layer per step, for nothing.
- **Preallocate the `set_pending` dict** (`cache.py:678`) instead of building a
  7-key dict per layer per step.

### 5.6 Split-K, for every shape that is not B=32

`grid = (B * H_kv,)` (`decode_kernel.py:374`) is **8 programs on a 108-SM A100 at
B=1**, and 256 (2.4 waves) at B=32. The kernel's own docstring already flags
split-K over the sequence as "not yet". It is what makes B=1 and B≤8 not
embarrassing, and it is orthogonal to everything above.

### 5.7 Priced but not recommended yet: the quantization grid

Per-`(window, head, dim)` fp16 scale+zero is 512 B/token against 256 B of K
codes. A 4-window group (G=32, KIVI's setting) would cut int2 to 672 B/token —
**36% more retained tokens at equal bytes, or 36% less Q-tier traffic at equal
tokens.** But it couples four eviction windows into one quantization group, which
the write-once store (design §10) is built to keep independent, and it changes
quantization error directly. This is an accuracy decision, not a perf one. Park
it behind the §7 gate.

## 6. Stage 3 — CUDA graphs, and the fairness problem

If Stages 1–2 land, our own host excess is ~9 ms and the residue is the harness's
own 24.5 ms — HF's per-layer Python, paid identically by FullKV, SnapKV and KIVI.

The decode loop (`perf_runner.py:1660-1674`) is a manual `model(...)` loop with no
per-step sync, so it is graph-capturable in principle: `_reslice` (`state.py:123`)
already views a preallocated buffer, and at steady state the eviction leaves `W`,
`k_fp`, `n_q` constant, so the 7 non-eviction steps of each cycle have static
shapes. What is missing is tensor-valued lengths (the kernel takes `Sfp` as a
Python int it specializes on) and a fixed-capacity K/V view with masking.

**Do not do this to win the table.** CUDA-graphing our path removes overhead every
other row still pays, and a 2× win sourced from that is not a KV-cache result. If
it is done, it has to be offered to every method in the sweep and reported as a
separate harness column. It is listed here so that it is a decision and not an
accident.

## 7. What each change does to the scores

| change | effect on scores | gate |
|---|---|---|
| §4.1 cross-layer eviction | none claimed — **identical retained set** | byte-identity of `retained_idx`/`new_tier` over 64 steps |
| §4.3 close the graph break | none | existing `aot_eager` eviction test |
| §5.1 window-score epilogue | fp-level (softmax reassociation) | tolerance vs `two_tier_decode_reference` |
| §5.2–5.4, §5.6 | none (launch / shape only) | `tests/test_decode_kernel.py` |
| §5.5 `exp2` | ~2 ulp, as in prefill | decode analog of `tests/test_score_exp2.py` |
| §5.5 `cos`/`sin` fp16 | fp16 rounding on a rotation | tolerance + one LongBench dataset |
| **§4.2 `eviction_interval`** | **changes outputs** (more context, more evidence) | **full LongBench + GSM8K at `{ws, 2ws, 4ws}`** |
| **§5.7 quant group size** | **changes quantization error** | **full LongBench; park until §4/§5 land** |

The default configuration must stay score-identical. `eviction_interval` defaults
to `ws` and ships off until the quality run says otherwise —
`ACCURACY_RECOVERY_PLAN.md` §1 is the standing reminder of what a silent default
change costs.

## 8. Order of work

| # | item | risk to score | launches/step | expected |
|---|---|---|---|---|
| 0 | Stage 0 profile at 4096/B=32 | none | — | gates everything |
| 1 | §5.5 freebies (`exp2`, fp16 rope, dead hook) | fp-level | 1,700 → 1,650 | −1 ms host, −0.4 GB traffic |
| 2 | §5.1 window-score epilogue | fp-level | 1,650 → 1,230 | −19 ms host, −0.94 GB |
| 3 | §5.3 + §5.4 Q-tier tiling | none | — | the GPU half |
| 4 | §4.1 cross-layer eviction — **LANDED** | none (gated, byte-identity pinned) | 1,230 → 172 | −48 ms host |
| 5 | §5.6 split-K | none | — | the B=1 and B≤8 rows |
| 6 | §4.2 `eviction_interval` (opt-in) | **outputs** | 172 → 147 | −1 ms, + quality run |
| 7 | §5.7 quant grid / §6 CUDA graphs | **outputs / fairness** | — | decisions, not tasks |

Items 1–5 are score-safe and are the whole of the projected 176 → ~51 ms.

## 9. Success criteria

| | target | how verified |
|---|---|---|
| TPOT @ 4096/256, B=32 | **≤ 0.050** (beats Flash 0.086 by 1.7×) | `run_perf_table.sh`, fresh `OUT_DIR` |
| TPOT @ 4096/256, B=1 | ≤ 0.035 (beats KIVI 0.043) | same table |
| beats a named peer | < 0.086 (Flash), < 0.043 (KIVI) | same table |
| stretch | approach 0.033 (SnapKV) | same table |
| GPU busy % @ B=32 | > 80% | `profile_decode.py` |
| scores unchanged | identical at the default config | LongBench + GSM8K before / after |

Report `TPOT_steady` at a `gen` long enough to contain several evictions — a
short `gen` puts zero evictions in the steady window and flatters the number by
~40%.

## 10. Housekeeping found on the way

- `DECODE_PLAN.md` is superseded: its target cell (budget 0.20, "20% keys, 48%
  bytes", TPOT 0.1345) is not what `run_perf_table.sh` runs (budget 0.50, q=0.70,
  TPOT 0.176). Delete it or mark it stale.
- `reports/decode_throughput_benchmarks.html` describes its second memory figure
  as "the steady-state KV cache footprint in GB". It is `peak_decode_steady_mb` —
  **device-used** memory in the steady phase — which is why 4096/B=32 reads
  `49.20 / steady 61.24` with the second number larger than the first. The caption
  should say so; `print_perf_table.py:13-15` already does.
