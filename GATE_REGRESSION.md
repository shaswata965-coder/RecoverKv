# The read gate is a 24% decode regression — findings

Working notes from the session that measured it. Everything here is either a
measurement or is labelled as an inference. Where an earlier claim in this repo
is wrong, it is named.

---

## 1. The regression

Baseline is the published artifact (`QEvict` rows, batch 32). Current is the
run of 2026-09-16.

| shape, b32 | baseline TPOT | current | | baseline e2e tok/s | current | |
|---|---|---|---|---|---|---|
| 4096/256 | 0.0626 | **0.0777** | +24% | 294.3 | **256.1** | −13% |
| 2048/512 | 0.0531 | **0.0644** | +21% | 497.1 | **421.6** | −15% |
| 1048/1048 | 0.0488 | **0.0546**\* | +12% | 619.7 | **556.2**\* | −10% |

\* those two rows were flagged stale (17 h older than the rest of the table), so
the true gap may be larger.

**Baseline commit is `fbcf368` (2026-09-13).** At that commit
`_two_tier_decode_kernel` already existed and `modules/quant/sketch.py` did not.
Everything between `fbcf368` and HEAD is the read-gate work. The regression is
the gate.

---

## 2. Why reading *less* got *slower*

Three compounding effects. The byte accounting is the core one.

### 2a. The gate skips 75% of the windows but only 28% of the bytes

Per Q window at `ws=8`, `H_kv=8`, `D=128`:

| | bytes | share |
|---|---|---|
| K int2 codes | 2,048 | 24% |
| **K fp16 scale/zero grid** | **4,096** | **48%** |
| V int2 codes | 2,048 | 24% |
| V scale/zero | 256 | 3% |
| **total** | **8,448** | |

- ungated read: 1,651 MB/step → 0.810 ms roofline
- gated (0.25) read: 1,186 MB/step → 0.582 ms roofline
- **saving: 0.23 ms**

The fp tier is **87% of what the gated kernel reads** and the gate never touches
it. The scale/zero grid is *half of a Q window* — two bytes of precision
wrapping a two-bit code.

### 2b. `SEL` turns coalesced streams into gathers

Ungated, `widx = w0 + t_win` is affine, so Triton emits contiguous vector loads.
Gated, `widx = tl.load(SEL + …)` is a runtime value, so `KS`, `KZ`, `KC`, `VC`,
`VS`, `VZ` and `COS`/`SIN` all become scattered gathers with an address
dependency on the SEL load.

The kernel measures **7.03 ms against a 0.58 ms roofline — 12×**. That is what a
gather-bound kernel looks like. (Model GEMMs, for contrast, run at 1.4× theirs.)

### 2c. The gate added a full-tier pass that did not exist before

`decode_kernel.py` §5 loops `range(n_body_win, W_phys, BLOCK_W)` — over **every**
Q window, not the selected ones — to fill scores for the windows it skipped,
plus a prologue seeding `WSUM`/`WMAX` across all Q columns. The gate reduced the
read but not the score pass; it added one.

**Net:** save 0.23 ms of traffic; pay for gathers on the remaining 0.58 ms plus
two full-tier passes.

---

## 3. The measurement tooling was wrong, in both directions

Two separate errors. Both invalidated conclusions that were acted on.

### 3a. The profiler counted every kernel twice

`key_averages()` returns the ATen operator view **and** the CUDA kernel events,
and both carry self device time. `gpu_us = sum(...)` over all of them
double-counted. Caught by the new rollup printing its `other` bucket by name:

```
aten::mm                    11.052 ms/step   n=225.0
ampere_*gemm* (4 kernels)   10.831 ms/step   n=225.0   <- exactly the same 225
```

| | reported | real |
|---|---|---|
| CUDA kernel time | 87.30 ms/step | **48.05** |
| kernel launches | 3,132 | **1,656** |
| GPU busy | 58.7% "MIXED" | **32.3% — HOST-BOUND** |

This also retires the "model is 45.8 ms, 4.8× the hardware floor" reading that
never reconciled. **The model's GEMMs are 11.0 ms**, ~1.4× the fp16
weight-bandwidth floor. Fixed in `33ae5f9`.

### 3b. The wrong baseline file was used for three replies

`reports/decode_throughput_benchmarks.html` in the repo is an **earlier
revision** of the published artifact. Its own successor's footer says so:

> *Earlier revisions of this table listed decode-only `batch / TPOT` for the
> non-QEvict rows, which excluded prefill and so flattered the long-prefill
> shapes.*

Comparisons made against that file concluded the current build was ~2× faster
than baseline and ahead of FullKV-flash. **Both conclusions were false.** Use
the published artifact, not the checked-in HTML.

---

## 4. Corrected profile, 4096 prefill / batch 32

Real kernel time 48.05 ms/step, 1,656 launches.

| bucket | ms/step | share |
|---|---|---|
| ours: two-tier decode | 7.03 | 14.6% |
| ours: read gate | 0.81 | 1.7% |
| model: GEMM | 11.04 | 23.0% |
| elementwise | 14.20 | 29.6% |
| memory: copy/cat | 7.50 | 15.6% |
| memory: index/gather | 4.44 | 9.2% |
| reduction | 2.51 | 5.2% |

- There is **no `model: attention` row** — `_two_tier_decode_kernel` *is* the
  attention now.
- `elementwise` + `copy` + `reduce` is 60% of GPU time and is issued by **both**
  sides; kernel names cannot split it. Stock HF Llama's unfused RMSNorm and RoPE
  are most of it (`aten::neg` n=64.6 = 2/layer, `aten::mean` n=65.2).
- The cache's own launch budget, measured separately: **80 launching ops per
  steady step** (64 of them the KV append) and **493 on an eviction step** —
  ~142/step amortized, ~8.6% of the real total. There is no launch reduction
  left to find on the cache side.

---

## 5. Narrowing the scale/zero grid: measured

Reconstruction error, 12 seeds, keys with outlier channels + DC offset.

| grid | K err | V err | vs fp16 | B/token |
|---|---|---|---|---|
| fp16 (shipped) | 0.1330 | 0.5005 | — | 1056 |
| **int8 + shared fp16 scale** | **0.1338** | 0.5006 | **+0.6%** | **784** |
| fp8 e4m3 | 0.1351 | 0.5006 | +1.6% | 784 |
| fp8 e5m2 | 0.1390 | 0.4998 | +4.5% | 784 |

**Speed impact: ≤1%, and that is an upper bound.**

| | bytes saved | kernel | saves |
|---|---|---|---|
| gated 0.25 | −3.4% | 7.03 → 6.80 ms | 0.24 ms/step (0.30%) |
| ungated 1.0 | −9.7% | 7.03 → 6.35 ms | 0.68 ms/step (0.88%) |

Assumes time scales with bytes; the kernel is 12× off roofline, so it does not.

**Real payoff is memory:** 26% smaller Q windows → **1.35× more Q-tier tokens at
the same byte budget** under `quant_budget_mode=bytes`. Accuracy-positive.

**Why the accuracy cost is near zero:** design §2 pins codes to the *stored*
grid, so a rounded scale/zero just repositions the four levels and the codes
re-fit. The int2 error (13% for keys) swamps a 6% grid rounding. The prediction
that `zero` would be dangerous — an absolute anchor unrelated to the range — did
not survive measurement.

**On A100 (sm80) use int8, not fp8.** There is no hardware fp8 convert before
sm89; Triton would emit software bit-twiddling in a kernel already 12× off
roofline. int8 also measured *better* (+0.6% vs +1.6%).

**Do not raise `ws` to save the same bytes** — it costs 20× more accuracy:

| | B/token | K err |
|---|---|---|
| ws=8, fp16 (now) | 1056 | 0.1330 |
| ws=8, 1-byte grid | 784 (−26%) | 0.1351 (+1.6%) |
| ws=16, fp16 | 800 (−24%) | 0.1732 (**+30%**) |
| ws=32, fp16 | 672 (−36%) | 0.1818 (+37%) |

---

## 6. Landed this session

| commit | what |
|---|---|
| `5186353` | Fuse the int2 quantiser into the eviction graph; scope `emulate_precision_casts` to the compiled eviction |
| `0091e9c` | Delete the decode read path's second implementation (`STICKYKV_COMPILE_READ`) |
| `d43040a` | Bucketed rollup in `profile_decode.py`; `other` is always named |
| `33ae5f9` | Stop counting every kernel twice |

### `5186353` also fixed a live correctness bug

Inductor does not emulate intermediate precision casts by default — it drops an
`fp32 → fp16 → fp32` round trip. Three quantisers in the eviction graph take
that round trip deliberately. Measured over 24 seeds without the fix:

| | diverged from eager |
|---|---|
| `quantize_key_windows` | 11/24 seeds |
| `quantize_value_windows` | 14/24 |
| `sketch._q_sym` | 24/24 |
| `sketch._q_up` | 24/24, **plus 115 violations of `dequant >= x`** |

The last row is **not** caused by that change — the sketch helpers were never
behind a graph break, so every compiled eviction since the cards landed built
them on the elided grid. `_q_up`'s `dequant >= x` is the only reason the card's
Cauchy–Schwarz score is an *upper* bound, i.e. the only reason the gate may skip
a window. With the fix: 0 divergences, 0 violations, and a 14-step/5-eviction
run is identical tensor-for-tensor compiled vs eager.

---

## 7. Open — in priority order

1. ~~**`--gate-ratio 1.0` control arm.**~~ **Run. See §8.1 — it ties, which is the
   "machinery costs more than the selection saves" branch.**
2. **Profile `fbcf368` vs HEAD** with the fixed bucketed profiler; diff the
   `ours:` rows. Confirms 2a–2c quantitatively. **Still open, and now blocked on
   §8.2** — the two commits are not at the same operating point.
3. ~~**Re-run the four stale cells** before comparing anything.~~ Done; §8.1 is a
   fresh pair of full-table runs.
4. If the gate is confirmed net-negative, the design question is whether a gate
   over an int2 tier can pay at all — the codes are already 8× smaller than fp16,
   so the tier is 38% of the ungated read and half of *that* is the grid.
   **§8.1 confirms it; the question is live.**
5. Grid → int8. Park until 1–2 land; it is a memory change, not a speed one.

### Not measured, do not assume

- The §5 accuracy numbers are **reconstruction error on synthetic tensors**, not
  LongBench. A 0.6% move is far below what LongBench resolves — that is an
  inference from a proxy, not a result.
- The "realistic keys" distribution is a model of LLM key statistics (outlier
  channels + DC offset), not dumped tensors.
- GPU cost of `emulate_precision_casts` is unmeasured. The truncations are
  pointwise and should fuse into the same kernels — instructions, not launches.
- The `elementwise`/`copy`/`reduce` 60% has not been split between the model and
  the cache. Only a chrome trace (`--trace`) can do that.

---

## 8. The control arm ran — and what it exposed

Three findings. The first answers §7.1. The second and third say that two of the
numbers this document reasons from were measured against the wrong reference.

### 8.1 gate 1.0 vs 0.25: the selection is worth ~0.8%

Two full table runs, identical but for `GATE_RATIO`. TPOT_steady, seconds:

| shape | B | gate 1.0 | gate 0.25 | Δ | e2e 1.0 | e2e 0.25 |
|---|---|---|---|---|---|---|
| 4096/257 | 1 | 0.0575 | 0.0566 | **−1.6%** | 16.6 | 16.8 |
| 4096/257 | 32 | 0.0770 | 0.0763 | **−0.9%** | 257.5 | 259.3 |
| 2048/513 | 1 | 0.0578 | 0.0574 | −0.7% | 17.0 | 17.1 |
| 2048/513 | 32 | 0.0645 | 0.0644 | −0.2% | 421.2 | 421.6 |
| 1024/1025 | 1 | 0.0576 | 0.0573 | −0.5% | 17.2 | 17.3 |
| 1024/1025 | 32 | 0.0590 | 0.0586 | −0.7% | 516.0 | 519.6 |

**Validity check first:** TTFT agrees across the two runs to within 0.1% in every
cell (11.820/11.812, 5.577/5.576, 2.714/2.707). A decode read gate cannot touch
prefill, so matching TTFT is what says these are the same configuration differing
in one variable. Without that the table would be two runs, not a control arm.

**Reading 25% of the windows instead of 100% is worth 0.2–1.6% of TPOT.** §2a
predicted it: 0.23 ms of traffic on a ~76 ms step is 0.3%, and the measurement
comes out the same order. The byte accounting was right, and what it was right
about is that there is nothing there.

**Memory does not move either.** `peak_GB` is identical in all six cells;
`steadyKV_GB` is identical in four and within 0.5% in the other two. That is
correct and expected — this is a *read* gate, not a residency gate; it changes
what a step dequantizes, never what the cache keeps. It was never going to buy
memory and it does not.

**So §7.1's decision rule fires.** 1.0 does not beat 0.25 — it ties it, and a tie
is the same verdict: the machinery costs essentially everything the selection
saves. Every millisecond of §2b's gathers and §2c's extra full-tier pass is being
paid for a ~0.8% return.

### 8.2 §1's baseline comparison is confounded — the operating point moved

`ecc0792` *"Adjust CACHE_BUDGET and LOCAL_WINDOW parameters"* (2026-09-15) sits
**inside** the `fbcf368..HEAD` window and changes `scripts/run_perf_table.sh`:

| | before | after |
|---|---|---|
| `CACHE_BUDGET` | 0.50 | **0.20** |
| `LOCAL_WINDOW` | 64 | **128** |

§1 says *"Everything between `fbcf368` and HEAD is the read-gate work. The
regression is the gate."* That is not true of `ecc0792`. It is not gate work at
all — it is a change to the benchmark's own operating point, and the published
artifact §1 compares against was produced at budget 0.50 / local 64.

**The direction is the opposite of what "LOCAL_WINDOW doubled" suggests, and it
matters.** `CACHE_BUDGET` is the outer term: `config.resolve` computes
`remaining = cache_budget * (prefill + max_tokens) - num_sink - local_tokens`
and splits *that* between the tiers, so a 2.5× budget cut shrinks the fp body and
the int2 tier together, and swamps the +64 tokens `LOCAL_WINDOW` adds. Worked
through for the 4096 cell at `quant_mode=tokens`, `q=0.70`:

| | published 0.50 / 64 | current 0.20 / 128 |
|---|---|---|
| `total_budget_tokens` | 2,176 | 870 |
| `top_k_windows` | 263 | 92 |
| `N_q` (int2 windows) | **184** | **64** |
| `top_k_fp` (fp windows) | 79 | 28 |
| `S_fp` (fp tokens/row) | **701** | **357** |
| read/step, gated 0.25 | **3,338 MB** | **1,636 MB** |

(`N_q = 184` against `DECODE_SPEED_PLAN.md`'s `n_active ≈ 179`, and
`b_q = 8,448 B` against §2a's window — two independent checks that this is the
right arithmetic.)

So the fp tier did not double, it **halved**, and the int2 tier fell to a third.
**The current table reads 2.04× LESS than the baseline it is scored against and
is 22% slower.** Which retires "reading less got slower" as a traffic story
altogether — see §8.6 — and makes the regression *larger* than §1 states rather
than smaller, because the baseline was carrying twice the bytes when it posted
0.0626.

Prediction for the re-run, worth writing down before it is made: at 0.50/64 the
Q-tier loop visits 46 selected windows instead of 16 and the body loop 79 fp
windows instead of 28, so HEAD should land **slower** than its own 0.0763 — call
it 0.080–0.085 — and the true matched-config regression against 0.0626 is then
~30%, not 24%. **The 24% in §1 is not attributable to the gate
until this is re-run at the published operating point:**

```bash
CACHE_BUDGET=0.50 LOCAL_WINDOW=64 GATE_RATIO=0.25 CUDA_VISIBLE_DEVICES=0 scripts/run_perf_table.sh
CACHE_BUDGET=0.50 LOCAL_WINDOW=64 GATE_RATIO=1.0  CUDA_VISIBLE_DEVICES=0 scripts/run_perf_table.sh
```

### 8.3 `GPU busy` was divided by a wall the profiler had inflated

§3a fixed the numerator — `key_averages()` double-counting every kernel — and
left the denominator wrong in the same direction.

`wall_us` was measured **inside** the `with profile(...)` block. The profiler's
per-event cost is charged to that wall and to nothing else, so it lands wholly in
the denominator and drives `busy` down:

| | ms/step |
|---|---|
| CUDA kernel self time (numerator) | 48.05 |
| wall, profiler **ON** | 147.34 |
| `tpot_steady_ms` at the same cell, profiler **OFF** | **76.3** |
| difference = 71.0 ms over 1,656 launches | **43 µs/launch** |

- 48.05 / 147.34 = **32.6% → "HOST-BOUND"**
- 48.05 / 76.3 = **63% → "MIXED"**

The two are comparable: `tpot_steady_ms` is `(t3 − t_step0) / (n_decode − 1)`,
which includes eviction steps exactly as the profiled window does.

**This flips the instrument's verdict, and the verdict is what the priority list
keys off.** "Host-bound" says collapse launches. "63%, mixed" says the GPU is busy
two-thirds of the step and the kernels are at least as large a lever as the host —
so `DECODE_SPEED_PLAN.md`'s standing premise that decode is host-bound does not
survive the corrected denominator, and neither does §4's reading of it here.

Fixed: `profile_decode.py` now times `steps` steps with the profiler **off**,
reports `GPU busy` against that, and prints the profiled wall beside it labelled
as overhead.

### 8.4 `sel` is sorted again — the one change §2b's mechanism asks for

§2b names the mechanism: under `SEL` the Q-tier loop's `widx` stops being affine,
`KS`/`KZ`/`KC`/`VC`/`VS`/`VZ`/`COS`/`SIN` all become gathers, and the kernel runs
**7.03 ms against a 0.58 ms roofline — 12× off**.

`e7bc158` had dropped the ascending sort on the gate's pick, correctly reasoning
that the decode kernel places each window's score at its own column so the order
cannot change the result, and therefore that the sort was "a launch for nothing".
The result is not what the order was buying. `topk` returns indices in descending
*score* order, which is arbitrary in *column* space — a tile of `BLOCK_NW` windows
then touches `BLOCK_NW` segments scattered across the whole tier:

```
unsorted topk : [14, 13, 19,  6, 15]
sorted        : [ 6, 13, 14, 15, 19]
```

Same set, same scores, same eviction decisions — monotone addresses instead of
random ones, with a mean stride of `1/ratio` windows. Restored in
`gate_kernel._sorted_pick`, which carries the argument.

Cost is one sort of `[B, H_kv, n_sel]` per layer per step (~11.5k int32 at the
headline cell) against a cache-side launch budget §4 measures at 8.6% of the
step. **Unmeasured on GPU** — it is a memory-locality argument, and §2b's 12× is
the only evidence that locality is what the kernel is losing to.

### 8.6 The ceiling on every read-side optimisation is ~1% of the step

The single number that reorders this whole document. At the 4096/B=32 cell the
gated read is **1,636 MB/step**; an A100 moves that in **0.80 ms**. The step is
**76.3 ms**.

> **The entire KV read — both tiers, every byte the gate exists to avoid — is
> 1.05% of TPOT.**

At the published operating point it is 1.64 ms of 62.6 ms, i.e. 2.6%. Neither is
a budget anything can be won back from.

That bounds, by construction and regardless of implementation:

| claim | what it is really worth |
|---|---|
| §2a's 0.23 ms gate saving | 0.30% of TPOT |
| §5's int8 grid, gated | 0.24 ms → 0.31% |
| §5's int8 grid, ungated | 0.68 ms → 0.89% |
| a *perfect* gate (`n_sel = 0`, vs **ungated**) | 0.27 ms → **0.35%** |
| the headroom left from `ratio = 0.25` | 0.07 ms → **0.09%** |

And it explains why §8.1 measured 0.8% for the gate and §2a predicted 0.3%: both
are right, and both are noise against the step. It also explains the 12×: the
two-tier kernel spends **7.03 ms moving 0.80 ms of bytes**, so it is not
bandwidth-limited at all — the gathers cost latency and issue slots, not
bandwidth, and "bytes saved" was never the unit that predicted its time.

**The corollary is the useful part.** The fp body loop covers `top_k_fp * ws +
local + sink` = 357 tokens against the Q loop's 128 (at `n_sel=16`), so **73% of
the kernel's token work is in the tier the gate cannot touch.** No setting of
`quant_gate_ratio` reaches it.

Everything worth having is therefore in the other two places: the **48.05 ms of
kernels** and the **28.25 ms of host gap** (§8.3). `DECODE_PLAN`-style byte
accounting should not be used to size decode work again.

### 8.5 The scratch round trip — checked, and too small to chase

`WSUM`/`WMAX` are `[B, H_q, W_phys]` fp32 in **global** memory, and the kernel
makes about five passes over them: the `GATED` prologue seeds the Q columns, the
body loop writes, the Q loop writes, the §4 epilogue reads and rewrites all of
them, and the §5 fill reads and rewrites the skipped ones. At `W_phys ≈ 200`
(`DECODE_SPEED_PLAN.md`'s `n_active` at the 4096 cell) that is 2 × 32 × 32 × 200 ×
4 B ≈ 1.6 MB per layer per pass, ~256 MB/step across 32 layers — **~0.13 ms/step
at A100 bandwidth**, against the 1,186 MB/step of the gated read.

Real, and 0.2% of the step. Written down so nobody spends a week on it.

---

## 9. Revised priority

| # | item | kind | why it is here |
|---|---|---|---|
| 1 | Re-run at budget 0.50 / local 64 (§8.2) | measurement | nothing in §1 is attributable until this lands |
| 2 | Re-profile with the fixed denominator (§8.3) | measurement | decides whether kernels or host lead |
| 3 | Measure `sel` ascending (§8.4) | landed, unverified | the only change §2b's mechanism directly asks for |
| 4 | Split the `elementwise`/`copy`/`reduce` block | measurement | ~24 ms of 48 ms — bigger than the GEMMs, bigger than our kernel, and still unattributed. Needs `--trace` |
| 5 | Decide the gate on §8.1 | design | a ~0.8% return does not pay for §2b + §2c |
| 6 | Grid → int8 (§5) | memory | 1.35× Q tokens at the same bytes; not a speed item |

Items 1–4 are measurements, and they come first because the three findings above
are all the same failure: a number was compared against a reference that had
moved. The remaining speed question is genuinely open — §8.3 means the 7.03 ms
two-tier kernel is 14.6% of GPU time on a step that is 63% GPU, i.e. ~9% of TPOT,
and the 24 ms elementwise block is three times that and still belongs to nobody.

### Added to "do not assume"

- **§8.4 is unmeasured on GPU.** Sorting `sel` is an argument about coalescing and
  L2, pinned on CPU only for what it must not change (the selected set, the column
  each score lands on). It reorders an online-softmax accumulation, so `out` moves
  in the last bits exactly as any tile-order change does.
- **§8.2 is a confound, not a result.** It says the 24% is not attributable as
  claimed. It does **not** say the gate is innocent, and it does not predict the
  size or even the sign of what the re-run will show.
