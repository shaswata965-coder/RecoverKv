# Decode speed plan — remaining work

`PREFILL_SPEED_PLAN.md` closed prefill to 1.09–1.14× FullKV. Cross-layer
batched eviction (formerly this plan's §4.1) is landed and default-on — see
`git log` on `modules/windowed_cache/cache.py` and
`tests/test_layer_major_evict.py`. This document is what is left.

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

## 1. Current baseline and the gap to close

`run_perf_table.sh` at the shipped defaults (`STICKYKV_LAYER_MAJOR_DECODE=1`):

| shape / B | ours (TPOT) | Flash (FullKV) | ratio |
|---|---|---|---|
| 4096/256, B=1 | 0.0586 | 0.0250 | 2.34× slower |
| **4096/256, B=32** | **0.0735** | 0.0860 | **1.17× faster** |
| 2048/512, B=1 | 0.0588 | 0.0240 | 2.45× slower |
| 2048/512, B=32 | 0.0641 | 0.0550 | 1.17× slower |
| 1024/1024, B=1 | 0.0594 | 0.0240 | 2.48× slower |
| 1024/1024, B=32 | 0.0607 | 0.0430 | 1.41× slower |

**The headline cell (4096/B=32) already beats FullKV.** What is left is closing
the gap to the harder targets:

| | target | current | gap |
|---|---|---|---|
| 20% faster than Flash @ 4096/B=32 | ≤ 0.0688 | 0.0735 | **4.7 ms** |
| stretch @ 4096/B=32 | ≤ 0.050 | 0.0735 | **23.5 ms** |
| beat KIVI-int2 @ ~1024/B=32 | < 0.044 | 0.0607 | **16.3 ms** |
| approach SnapKV @ 4096/B=32 | ~ 0.033 | 0.0735 | **40.5 ms** |

Sensitivities at the current baseline:

```
B=1, 1024 -> 4096  :  ours -1.3%      (flat — decode cost still ~independent of context at B=1)
B=1 -> B=32 @4096  :  ours +25.4%
B=1 -> B=32 @1024  :  ours  +2.2%
```

At B=1 the excess is still nearly flat across context length, which is what a
fixed per-step host/launch cost looks like. At B=32/4096 it grows faster in
*relative* terms than before the eviction change, because what remains scales
more with the traffic and GPU work the sections below target — not because the
launch cut stopped helping.

### The roofline, so the targets are bounded honestly

| | per-step traffic | @2.04 TB/s |
|---|---|---|
| FullKV | 16.06 GB weights + 17.18 GB KV = 33.2 GB | 16.3 ms |
| ours | 16.06 + 4.42 GB KV = 20.5 GB | 10.0 ms |
| ours + avoidable overhead traffic (§1.1) | 22.4 GB | 11.0 ms |

Flash measures 86 ms against its own 16.3 ms roofline (5.3× off) — that is the
harness's own per-layer Python cost, paid by every method in the table, not
something this plan can remove without a CUDA-graph change that would have to
be offered to every method (§6). Our 73.5 ms against a ~10–11 ms roofline is
what §2 and §5 are for.

### 1.1 Traffic we pay for nothing — the concrete target for §5.1

| | per layer / step | per step | share of our KV |
|---|---|---|---|
| `scores` fp32 `[B,H_q,S]`, written + re-read + rewritten + reduced | 34.7 MB | **1.11 GB** | 25% |
| `cos`/`sin` fp32 `[B, n·ws, D/2]` ×2 | 23.5 MB | **0.75 GB** | 17% |
| | | **1.86 GB** | **42%** |

Both are pure overhead. `decode_kernel.py:353` allocates the score buffer in
**fp32 at token granularity**, `decode_kernel.py:302` reads the whole thing
back for a second normalizing pass, and `scorer.py:93` reads it a third time to
reduce `S → W`. The RoPE tables (`decode_kernel.py:124`) are fp32 because
`rope_cos_sin_halves` builds them off `torch.empty(1,1,1)` (default dtype), and
they are **frozen between evictions** yet re-read every step.

## 2. The launch budget still on the table

The remaining excess is a mix of host-side launch overhead (§5.1, §5.2, §5.5)
and exposed GPU work (§5.1's traffic cut, §5.3, §5.4, §5.6). The pre-eviction-fix
version of this document derived a host/GPU split and a µs-per-launch rate from
the old (much larger) excess; that derivation should **not** be reused now — the
excess it was calibrated against was ~2.4× the current one, and a rate fit to a
bigger gap does not transfer cleanly to a smaller one. **Stage 0 (§3) has to be
re-run before any further ms projection in this document is trusted.** Until
then, treat every "expected" figure in §5 and §8 as a launch/byte-count fact
(real, architecture-level) with an *unverified* ms translation.

## 3. Stage 0 — the one run that gates everything else

`scripts/profile_decode.py` and `scripts/audit_e2e.py` exist and are unblocked.
Run **at the shape the table reports**, not at B=1, on the current default
(layer-major decode, already on):

```bash
python scripts/profile_decode.py --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 32 --steps 24 --warmup 12 --trace outputs/decode.json
```

```bash
python scripts/profile_decode.py --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 1 --steps 24 --warmup 12
```

Read four things, in order:

1. **GPU busy %.** With the eviction batched, host overhead should be smaller
   than before at every batch size — get the real number rather than assuming
   it. If B=32 already reads >80%, §5.1–§5.6 (GPU/traffic-bound) matter more
   than §5.2/§5.5 (launch-count-bound); if it is still well under that, the
   opposite.
2. **CPU self time, top 20.** If `JITFunction.run` / Triton frames dominate,
   §5.2 moves to the front. The profiler cannot name Python that isn't an ATen
   op, so pair it with a `cProfile` over 24 steps.
3. **`STICKYKV_COMPILE_EVICT=1` vs `=0` at this shape.** `run_perf_table.sh`
   defaults it on and `perf_runner.py:1242` force-enables it on CUDA. If the A/B
   is small, the `torch._dynamo.disable` on `_affine_quantize`
   (`quant/quantizer.py:48-61`) is still breaking the graph inside the (now
   batched) eviction body — §4.3.
4. **Eviction wall time per call, split from the normal step.** Sizes whether
   §4.2/§4.3 are still worth doing versus §5's per-step fixed costs.

Do not start §4 or §5 changes before this returns.

## 4. The eviction — what's left

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

If Stage 0.3 shows the compiled eviction buys little, fix the break rather than
route around it: compute the affine range on a **gathered, materialised** tensor
so the reduction's read index is not a `StarDep`, or write the range reduction as
a small Triton kernel. The `aot_eager` eviction test already pins byte identity.

This now applies to the batched (layer-major) eviction body, which is unchanged
in its own logic — only its row count is wider — so the same fix and the same
gate apply without modification.

## 5. The fused decode kernel

### 5.0 The `window_size` range these three changes support — verified

The question "can §5.1–§5.3 take any `ws` from 1 to 128?" has a measured answer
and a derived one, and they differ.

**Measured (what works today, before any of §5.1–§5.3).**
`tests/test_layer_major_evict.py::test_window_size_range_*` drives prefill +
decode end-to-end at `ws ∈ {1,2,3,4,6,8,12,16,32,64,128}` and asserts the whole
cache is byte-identical between the per-layer and layer-major paths:

| | supported `ws` | why |
|---|---|---|
| `q = 0` (fp only) | **1 – 128, all of them** | nothing constrains it; the eviction's row axis is orthogonal to the window axis |
| `q > 0` (int2 Q tier) | **multiples of 4 only** | int2 crumb packing, `config.py:354` — a hard `ValueError`, not a silent degrade |

**Derived (what §5.1–§5.3 add on top).**

| | extra constraint | supported `ws` |
|---|---|---|
| §5.1 window-score epilogue | `ws ≥ 2` (below that it is a pessimization) **and** power-of-2 for the clean form | `q>0`: **4, 8, 16, 32, 64, 128**;  `q=0`: also 2 |
| §5.2 Triton launcher | none — `ws` never enters it | anything the config accepts |
| §5.3 Q-tier tiling | none beyond `q>0`'s multiple-of-4; benefit shrinks as `ws` grows | 4 – 128 |

So: **not "any ws 1–128" for §5.1.** For §5.2 and §5.3, yes.

**Why §5.1 wants a power of 2.** Triton's `tl.arange(0, N)` requires `N` to be a
power of two — that is why `BLOCK_WS = _pow2_at_least(ws)` exists at all
(`decode_kernel.py:373`). So `BLOCK_N` is a power of two, and "each window lives
entirely inside one tile" needs `BLOCK_N % ws == 0`, which only a power-of-2 `ws`
satisfies. At `ws ∈ {3, 6, 12, 24, …}` windows straddle tile boundaries and the
clean form does not apply — see the fallback below. Note masking does **not**
rescue this: `tmask` handles *padding*, not *alignment*.

**Why `ws = 1` is a pessimization.** With one window per token, `W = S` and the
epilogue writes `5·W = 5·S` against the current `4·S` — 25% *more* traffic. Break
-even is `ws > 1.25`. At the benchmark cell:

| `ws` | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| §5.1 traffic vs today | **0.8× (worse)** | 1.6× | 3.2× | **6.4×** | 12.8× | 25.7× | 51.3× | 102.6× |

The 6.4× the rest of this plan quotes is the `ws = 8` column — the shipped
default.

**Fallback for non-power-of-2 `ws`,** if it is ever needed: carry the straddling
window's partial sum and its `m` across the tile boundary instead of flushing per
tile, reconciling at the next iteration. It costs one extra register pair and a
carry, and it removes the alignment constraint entirely. Not worth building until
a config actually needs `ws = 12`.

### 5.1 Emit window scores, not token scores (host **and** GPU)

`decode_kernel.py:235-310` writes `[B,H_q,S]` fp32 logits to HBM, reads them back
for a normalizing second pass, and `scorer.py:93` reads them a third time to
reduce `S → W` and scatter by `order`. Four passes over 8.67 MB/layer.

The kernel already holds the logits in registers, so for each tile store the
per-window partial sum `Σ exp(logit − m_tile)` *and* `m_tile`; the epilogue
rescales by `exp(m_tile − lse)`. At the shipped `ws = 8` traffic drops from `4·S`
to `5·W = 5·S/8` — a **6.4× cut, 1.11 GB → 0.17 GB/step** — and the `order`
gather plus the `+=` into `state.window_scores` fold into the same epilogue,
removing ~13 launches/layer/step (**416/step**).

**Two things the original sketch of this got wrong, both now pinned by §5.0:**

1. `window_size` stays a runtime `tl.constexpr` — this change does *not* fix it
   to one value. What it needs is `BLOCK_N` (today a fixed 64, independent of
   `ws` — `decode_kernel.py:326`) **derived from `ws` per launch** as
   `max(64, _pow2_at_least(ws))`, the same way `BLOCK_WS` already is. That also
   covers `ws = 128 > 64`, where a window would otherwise span two tiles.
2. The fp-tier loop must be **`num_sink`-offset aware.** Windows in the fp store
   begin at `num_sink`, not at 0 (`scorer.py:76-77` strips the sink prefix before
   windowing), so `range(0, Sfp, BLOCK_N)` from offset zero misaligns every
   window by `num_sink % ws`. Handle `[0, num_sink)` as a prologue tile that
   contributes to the output and the LSE but emits no window score, then tile
   `[num_sink, Sfp)` on window-aligned boundaries. This works for any `num_sink`,
   including the benchmark cell's 5.

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

**`ws`-independent.** Nothing here touches the window axis, so it applies at
every `ws` the config accepts, and it is the only one of the three that is safe
to land without re-checking the geometry.

### 5.3 Tile the Q-tier loop

`decode_kernel.py:260` runs **one window per serial iteration** — at the
benchmark cell that is `n_active = 179` dependent iterations, against the fp
tier's 11 (`BLOCK_N=64` over `S_fp=685`). Every iteration is a full
online-softmax dependency: 4 scale/zero loads, 2 code loads, 2 fp32 `cos`/`sin`
loads, 2 tiny `tl.dot`s, an `exp`, a store.

Tile `BLOCK_NW = max(1, BLOCK_N // ws)` windows per iteration so the tile is a
fixed ~64 keys regardless of `ws`, rather than a fixed window count. At `ws = 8`
that is 8 windows/iteration: **179 → 23 iterations**, and the `tl.dot` goes from
`[16,64]×[64,16]` to `[16,64]×[64,64]`.

**The payoff is `ws`-dependent and vanishes at large `ws`,** because the loop is
already coarse there — worth knowing before spending the effort at a non-default
`ws`:

| `ws` | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| iterations today | 358 | 179 | 89 | 44 | 22 | 11 |
| iterations after §5.3 | 23 | 23 | 23 | 22 | 22 | 11 |
| speedup on the loop | 15.6× | 7.8× | 3.9× | 2.0× | 1.0× | 1.0× |

At `ws ≥ 64` this is a no-op: one window already fills the tile. Register
pressure does not get worse either — `BLOCK_WS` is unchanged, and the codebase
already runs `BLOCK_WS = 128` at `ws = 128` today.

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
- **Preallocate the `set_pending` dict** (`cache.py`, `flash_decode.set_pending`
  call site) instead of building a 7-key dict per layer per step.

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

## 6. CUDA graphs, and the fairness problem

The decode loop (`perf_runner.py:1660-1674`) is a manual `model(...)` loop with
no per-step sync, so it is graph-capturable in principle: `_reslice`
(`state.py:123`) already views a preallocated buffer, and at steady state the
eviction leaves `W`, `k_fp`, `n_q` constant, so the non-eviction steps of each
cycle have static shapes. What is missing is tensor-valued lengths (the kernel
takes `Sfp` as a Python int it specializes on) and a fixed-capacity K/V view with
masking.

**Do not do this to win the table.** CUDA-graphing our path removes overhead every
other row still pays, and a win sourced from that is not a KV-cache result. If
it is done, it has to be offered to every method in the sweep and reported as a
separate harness column. It is listed here so that it is a decision and not an
accident.

## 7. What each remaining change does to the scores

| change | effect on scores | gate |
|---|---|---|
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

| # | item | risk to score | expected |
|---|---|---|---|
| 1 | Stage 0 profile, re-run at 4096/B=32 | none | attribute the remaining ~35 ms/step between host and GPU — nothing below should be trusted for ms until this runs |
| 2 | §5.5 freebies (`exp2`, fp16 rope, dead hook, prealloc dict) | fp-level | ~−1 ms host, −0.4 GB traffic |
| 3 | §5.1 window-score epilogue | fp-level | ~−19 ms host, −0.94 GB — projected to close most of the remaining gap to the 20%-faster and stretch targets |
| 4 | §5.3 + §5.4 Q-tier tiling | none | the GPU-bound half of the remaining excess |
| 5 | §5.6 split-K | none | the B=1 and B≤8 rows, currently 2.3–2.5× slower than Flash |
| 6 | §4.2 `eviction_interval` (opt-in) | **outputs** | small further cut, + quality run |
| 7 | §5.7 quant grid / §6 CUDA graphs | **outputs / fairness** | decisions, not tasks |

Items 1–6 are score-safe. Every "expected" figure past item 1 is a
launch/traffic-count projection, not yet re-verified against the current
(smaller) excess — that is exactly what item 1 is for.

## 9. Success criteria

| | target | current gap |
|---|---|---|
| TPOT @ 4096/256, B=32 | ≤ 0.050 (beats Flash by 1.7×) | 23.5 ms |
| stretch: 20% faster than Flash @ 4096/256, B=32 | ≤ 0.0688 | 4.7 ms |
| TPOT @ 4096/256, B=1 | ≤ 0.035 (beats KIVI 0.043) | 23.6 ms |
| beat KIVI @ ~1024, B=32 | < 0.044 | 16.3 ms |
| stretch | approach 0.033 (SnapKV) | 40.5 ms |
| GPU busy % @ B=32 | > 80% | unmeasured — Stage 0 (§3) |
| scores unchanged at the default config | identical | full LongBench/GSM8K rerun still pending for the final (all-of-§5) configuration |

Verified with `run_perf_table.sh`, fresh `OUT_DIR`, same shapes/batches as §1's
table. Report `TPOT_steady` at a `gen` long enough to contain several
evictions — a short `gen` puts zero evictions in the steady window and flatters
the number by ~40%.

## 10. Housekeeping found on the way

- `DECODE_PLAN.md` is superseded: its target cell (budget 0.20, "20% keys, 48%
  bytes") is not what `run_perf_table.sh` runs (budget 0.50, q=0.70). Delete it
  or mark it stale.
- `reports/decode_throughput_benchmarks.html` describes its second memory figure
  as "the steady-state KV cache footprint in GB". It is `peak_decode_steady_mb` —
  **device-used** memory in the steady phase, which can read higher than the
  first (peak-allocated) figure. The caption should say so;
  `print_perf_table.py:13-15` already does. The table itself also predates the
  layer-major decode change and should be regenerated before it is shared again.
