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

## 0.1 Correctness invariants — non-negotiable

Every change below is a launch, layout or scheduling change. None of them may
alter what the cache holds. Three invariants make that checkable rather than
aspirational, and each is enforced by a test that fails loudly:

1. **The retained set is a function of `window_scores` alone.** Not of when the
   eviction runs relative to the append, not of `policy.total_tokens`. This is
   what let the eviction move to the start of the step; it is pinned by
   `tests/test_layer_major_evict.py`, whole-cache byte-identity against the
   per-layer path across `q ∈ {0, 0.5}`, `L ∈ {1,3}`, `B ∈ {1,2}`,
   `ws ∈ {1…128}`, with and without sinks, and across promotion/dormancy.

2. **`cache_position` is the absolute token index, and it is verified.** An
   evicting cache breaks a transformers assumption: with no explicit
   `cache_position`, `LlamaModel.forward` derives it as
   `arange(get_seq_length(), …)` — treating "keys held" as "tokens processed".
   Those diverge the moment the budget binds, because `get_seq_length()` becomes
   a sawtooth that drops by `ws-1` at every eviction. The step after an eviction
   is then told it sits ~`ws` tokens *earlier* than the one before it, and the
   store accumulates duplicated, non-monotonic positions (measured:
   `… 70, 71, 72, 65, 66, 67`).

   `generate()` is immune — it advances `cache_position` itself and never
   re-derives it — so every quality runner here was always safe. A hand-written
   decode loop is not, and `perf_runner`'s was not. **Both are now fixed** (it
   passes an explicit monotonic position in the warmup and measured loops), and
   `WindowedCache._check_position_contract` refuses a wrong base outright rather
   than letting it corrupt the store. The check is free until the retained count
   and the absolute index actually diverge, then costs one sync, once per cache.

   *This never moved the reported numbers* — store length is config arithmetic
   (`k_fp*ws + …`), not data-dependent, so shapes, launch counts and KV traffic
   were identical either way (verified: only HF's own `arange` differed, one
   dispatch per step). What it moved is whether the harness being optimized runs
   the semantics that ship.

3. **No silent fallbacks.** A path that cannot run raises. This already holds for
   the Triton kernels, the compiled eviction and the L-capture; §5 must not
   introduce an exception. In particular a kernel that cannot satisfy an
   alignment precondition (§5.0) must refuse, not quietly take a slow or
   different-answer path.

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

**Where the TPOT actually is, before picking an item.** At the benchmark cell the
per-step total is ~22.3 GB of which **16.06 GB (72%) is the weights** — untouchable
by any KV method. Roofline is ~10.9 ms; measured is 73.5 ms. So **~63 ms of the
73.5 is not traffic at all**, and no amount of traffic reduction reaches it:

| lever | what it moves | size |
|---|---|---|
| §5.1 traffic half | 1.11 → 0.17 GB/step of score traffic | **~0.5 ms** — 6.4× of a 5% term |
| §5.5 `exp2` | SFU-bound exponentials | ~0.75 ms |
| §5.5 fp16 `cos`/`sin` | 0.75 → 0.38 GB/step | ~0.18 ms |
| §5.1 launch half | −416 launches/step | **unsized** — needs Stage 0 |
| §5.2 Triton launcher | 32 launches/step × 50 args of Python | **unsized** — needs Stage 0 |
| §5.3 Q-tier tiling | 179 → 23 **serial dependent** iterations | **unsized** — latency, not traffic |

Adding up everything that *is* sized gives ~1.4 ms against a 4.7 ms gap to the
20%-faster target. **So the target is not reachable on the traffic terms alone —
it depends entirely on the three unsized launch/latency items, and which of them
dominates is exactly what Stage 0 (§3) measures.** Resist the temptation to read
the 6.4× in §5.1 as the headline; it is 6.4× of a small number.


### 5.0–5.3 — LANDED

> Implemented together, as one kernel rewrite: they touch the same loops and
> splitting them would have meant two rewrites. **Nothing outside the kernel and
> its immediate caller changed** — the cache, the eviction and the scoring
> semantics are untouched.
>
> **§5.1 — window scores, not token scores.** The kernel now emits
> ``wsum[B, H_q, W_phys]`` (per-window softmax mass, physical order: body windows
> then Q windows) plus a ``wmax`` companion, and rescales in a register epilogue
> over ``W_phys`` instead of a second pass over ``S``. Traffic ``4·S → 5·W``.
> ``_run_fused``'s reduction collapses from pad + reshape + sum + einops-reduce +
> cat + gather to **one gather** by ``order``.
>
> **§5.2 — fewer kernel arguments.** All 21 stride arguments for the Q-tier
> tensors, ``cos``/``sin``, ``OUT`` and the score buffers are gone; the kernel
> derives them from shapes. ``q``/``k_fp``/``v_fp`` keep explicit strides because
> they are transposed views and forcing them contiguous would copy the whole fp
> tier every step. The dispatcher **checks** contiguity and raises rather than
> assuming it. 50 args → 41.
>
> **§5.3 — whole-window tiling, both tiers.** The Q loop went from one window per
> serial iteration to ``BLOCK_NW`` windows, and the fp body was re-tiled the same
> way. At ``ws=8`` that is 179 → 23 Q-tier iterations.
>
> **`window_size` 1–128 all work, including non-powers-of-2.** §5.0 previously
> concluded §5.1 needed a power-of-2 ``ws``. That was wrong, and the fix is
> better: tile in **whole windows** padded up to a power-of-2 block
> (:func:`window_tiling`) and mask the tail. A window then never straddles a tile
> for any ``ws``, and the ``num_sink`` offset stops mattering too, because the
> body loop starts at ``num_sink`` and steps in window units. Cost is the padding
> alone — 100% lane utilisation at ``ws ∈ {1,2,4,8,16,32,64,128}``, 94% at
> ``ws=12``. The sink prefix gets its own prologue tile that feeds the softmax
> and emits no window score, matching ``reduce_token_scores_to_windows``.
>
> **Padding does not change what a window sums**, and that is verified rather
> than argued. Two independent defences: the lane mask zeroes a padding lane's
> contribution, and — structurally — a padding lane sits at
> ``offs_t >= BLOCK_NW*ws`` so its ``t_win = offs_t // ws`` is ``>= BLOCK_NW``,
> while the per-window store loop only runs ``j`` over ``0 … BLOCK_NW-1``. No
> window's selector can name it even with the mask removed. The CPU oracle
> materialises the padding lanes too (rather than slicing the exact range), so
> the end-to-end comparisons exercise the masking; on top of that,
> ``test_padding_lanes_are_unreachable_by_any_window`` checks the structural
> property **exhaustively for every ``ws`` from 1 to 128**, both directions —
> no padding lane is reachable, and every window owns exactly ``ws`` real lanes.
>
> **What is verified, and what is not.** Triton cannot run on this repo's CPU dev
> box, so the *lowering* ships unvalidated — the same contract the score kernel
> and the original decode kernel already ship under. The *algorithm* does not:
> :func:`two_tier_window_reference` reproduces the kernel tile for tile (sink
> prologue, window tiling, per-tile sums against the running max, the
> ``exp(m_tile − lse)`` epilogue) and ``tests/test_window_scores.py`` pins it —
> 52 tests covering ``ws ∈ {1…128}`` × ``num_sink ∈ {0,1,5}``, partial trailing
> windows, softmax-mass normalisation, overflow safety, and **five end-to-end
> cases against a real two-tier cache** where the old ``reduce_two_tier_scores``
> path and the new one-gather path are driven off the same store, the same
> effective K/V and the same ``score_meta``. Keep the oracle in step with the
> kernel; it is the only thing standing between this and an unverified rewrite.
>
> **Expected gain, honestly.** The traffic half is ~0.5 ms (see §5.1's table) —
> the value is in the launch and latency halves, which remain unsized until
> Stage 0 (§3) runs on a GPU. Do not assume this closed the 4.7 ms gap; measure it.

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
| — | §5.1 + §5.2 + §5.3 | fp-level | **DONE** — algorithm verified on CPU, Triton lowering ships unvalidated (§5.0–5.3) |
| 1 | Stage 0 profile, re-run at 4096/B=32 | none | attribute the remaining excess between host and GPU, **and measure what §5.1–5.3 actually bought** — nothing below should be trusted for ms until this runs |
| 2 | §5.5 freebies (`exp2`, fp16 rope, dead hook, prealloc dict) | fp-level | ~−1 ms host, −0.4 GB traffic |
| 3 | §5.4 `BLOCK_R` packing | none | the Q-tier `tl.dot` is 25% useful rows at `rep=4`; pack 4 KV heads per tile |
| 4 | §5.6 split-K | none | the B=1 and B≤8 rows, currently 2.3–2.5× slower than Flash |
| 5 | §4.2 `eviction_interval` (opt-in) | **outputs** | small further cut, + quality run |
| 6 | §5.7 quant grid / §6 CUDA graphs | **outputs / fairness** | decisions, not tasks |

Items 1–4 are score-safe. Every "expected" figure is a launch/traffic-count
projection, not yet re-verified on a GPU — including what §5.1–5.3 just landed,
which is precisely what item 1 now has to measure first.

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
