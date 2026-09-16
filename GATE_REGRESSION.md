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

1. **`--gate-ratio 1.0` control arm.** Keeps `GATED`, `SEL`, the epilogue and the
   cards; removes only the selection. If 1.0 beats 0.25, the machinery costs more
   than the selection saves. Decisive and cheap. **Run this first.**
2. **Profile `fbcf368` vs HEAD** with the fixed bucketed profiler; diff the
   `ours:` rows. Confirms 2a–2c quantitatively.
3. **Re-run the four stale cells** before comparing anything.
4. If the gate is confirmed net-negative, the design question is whether a gate
   over an int2 tier can pay at all — the codes are already 8× smaller than fp16,
   so the tier is 38% of the ungated read and half of *that* is the grid.
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
