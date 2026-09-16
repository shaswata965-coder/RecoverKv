# Decode speed — where the time is, and what to do next

A handoff document. It assumes you have read nothing else, and it names the file
and section for every claim it borrows. Read `GATE_REGRESSION.md` §8 next; this
is the plan built on top of it.

Every number labelled **measured** came off a GPU. Every number labelled
**estimated** is arithmetic on a measured number and is marked with a confidence.
No projection in the table in §4 has been run.

---

## 1. The headline cell, and the targets

Everything below is Llama-3.1-8B, one A100 80 GB, **prefill 4096 / gen 257,
batch 32**, `flash_attn` backend, `cache_budget 0.20`, `local_window 128`,
`window_size 8`, `num_sink 5`, `quant_ratio 0.70`, `quant_gate_ratio 0.25`,
`compile_evict 1`. That is what `scripts/run_perf_table.sh` runs at today.

| | TPOT | source |
|---|---|---|
| **now** | **0.0763** | measured, 2026-09-16 table |
| gate 1.0 control arm | 0.0770 | measured, same day |
| published QEvict artifact | 0.0626 | measured, **at budget 0.50 / local 64** |
| FullKV flash | 0.086 | measured |
| `DECODE_SPEED_PLAN.md` stretch | 0.050 | target, never met |

The published 0.0626 is **not** at this operating point — see §3.3. At matched
config, HEAD is expected to be *worse* than 0.0763, not better.

---

## 2. Where the 76.3 ms goes

Measured, `scripts/profile_decode.py`, after the double-count fix (`33ae5f9`)
and the denominator fix (`GATE_REGRESSION.md` §8.3).

```
TPOT_steady            76.3 ms/step        100%
├─ CUDA kernels        48.05 ms/step      63.0%   <- GPU busy
└─ host gap            28.25 ms/step      37.0%   1,656 launches @ ~17 us exposed
```

The 48.05 ms of kernels, by bucket:

| bucket | ms/step | % GPU | % TPOT | whose |
|---|---|---|---|---|
| elementwise | 14.20 | 29.6% | 18.6% | **unattributed** |
| model: GEMM | 11.04 | 23.0% | 14.5% | model, ~1.4× roofline — near optimal |
| memory: copy/cat | 7.50 | 15.6% | 9.8% | **unattributed** |
| ours: two-tier decode | 7.03 | 14.6% | 9.2% | ours |
| memory: index/gather | 4.44 | 9.2% | 5.8% | **unattributed** |
| reduction | 2.51 | 5.2% | 3.3% | **unattributed** |
| ours: read gate | 0.81 | 1.7% | 1.1% | ours |
| rest (sort/topk, …) | ~0.52 | 1.1% | 0.7% | mixed |

**Ours is 7.84 ms — 16.3% of GPU time, 10.3% of TPOT.** Even deleting the entire
two-tier kernel and gate leaves 68.5 ms.

**28.65 ms (60% of GPU) is in the four unattributed buckets.** Kernel names
cannot split model from cache. Only a chrome trace can, and it has never been
taken. This is the largest single unknown in the project.

---

## 3. Three things that are settled — do not re-litigate

### 3.1 The read gate is not a speed lever, at any ratio

Measured (`GATE_REGRESSION.md` §8.1): gate 0.25 vs 1.0, two full table runs,
TTFT matching to 0.1% in every cell. Reading 25% of the windows instead of 100%
is worth **0.2–1.6% of TPOT** (mean ~0.8%). `peak_GB` identical in all six cells.

Why it cannot be more (§8.6): the whole gated KV read is **1,636 MB/step**, which
an A100 moves in **0.80 ms** — **1.05% of the step**. The entire int2 tier,
ungated, is 553 MB = 0.27 ms, so a **perfect** gate — one that read *nothing*,
against reading the whole tier — has a ceiling of **0.35% of TPOT**. From where
the gate already sits at 0.25, the remaining headroom is 0.09%. And 73% of the
kernel's token work is in the fp body loop, which the gate structurally cannot
reach at any ratio.

The gate may still be worth keeping for accuracy or as a byte-budget lever. It is
not worth keeping for decode speed, and no tuning of `quant_gate_ratio` changes
that.

### 3.2 Byte accounting does not predict decode time

The two-tier kernel spends **7.03 ms moving 0.80 ms of bytes** — 8.8× off its own
roofline. It is bound by gather latency and issue slots, not bandwidth.
`GATE_REGRESSION.md` §2a and §5's speed column, and anything else that sizes
decode work in MB, are bounded by §3.1's ~1% and should not be used again.

### 3.3 The published baseline is at a different operating point

`ecc0792` changed `scripts/run_perf_table.sh`: `CACHE_BUDGET` 0.50 → 0.20,
`LOCAL_WINDOW` 64 → 128. `cache_budget` is the outer term in `config.resolve`, so
it dominates:

| | published 0.50/64 | current 0.20/128 |
|---|---|---|
| `N_q` (int2 windows) | 184 | 64 |
| `S_fp` (fp tokens/row) | 701 | 357 |
| read/step (gated) | 3,338 MB | 1,636 MB |

**The current table reads 2.04× less and is 22% slower.** So the regression is
larger than `GATE_REGRESSION.md` §1's 24%, not smaller, and it is definitively
not a traffic story.

---

## 4. Expected gains

Baseline 76.3 ms. **All estimated**; confidence is mine, and the "how to falsify"
column is how you find out cheaply.

| # | lever | targets | est. Δ ms | est. TPOT | conf. | how to falsify |
|---|---|---|---|---|---|---|
| L1 | CUDA graphs on the steady step | host gap 28.25 | **−15 to −22** | 0.054–0.061 | med-high | capture one step; count launches |
| L2 | Collapse the eviction's elementwise tail | ~15.4 amortized | **−5 to −10** | 0.066–0.071 | low-med | `--trace`, then eviction-step kernel count |
| L3a | `sel` ascending *(landed, unmeasured)* | two-tier 7.03 | **−2 to −3** | 0.073–0.074 | med | re-profile `ours: two-tier decode` |
| L3b | Drop the gate from the read path | two-tier + gate 7.84 | **−3 to −4.5** | 0.072–0.073 | med | `quant_ratio` unchanged, `sel=None` |
| L4 | A/B `emulate_precision_casts` | unknown | **0 to −8** | 0.068–0.076 | low | neuter the decorator, one table run |
| L5 | The two latched arms (`DECODE_SPEED_PLAN` §"Open") | ~5 ms constant | **0 to −5** | 0.071–0.076 | med | two env-var runs, already wired |
| L6 | int8 scale/zero grid (`GATE_REGRESSION.md` §5) | Q bytes | **−0.2** | 0.076 | high | it is a memory item, not a speed one |

**L3a and L3b are alternatives, not additive** — they target the same kernel.
L2 and L4 overlap; L4 is a plausible *cause* of L2.

Two combined paths:

| path | levers | est. TPOT | vs now | vs published 0.0626 | vs FullKV 0.086 |
|---|---|---|---|---|---|
| **A — no CUDA graphs** | L3 + L2 + L4/L5 + L6 | **~0.062** | −19% | ties | 1.39× faster |
| **B — with CUDA graphs** | A + L1 | **~0.043** | −44% | −31% | 2.0× faster |

Path A roughly recovers the published number **at an operating point that reads
half as much**, so it is a weaker result than it looks. Path B clears the 0.050
stretch target that `DECODE_SPEED_PLAN.md` declared unreachable.

**The fairness caveat on L1 is real and answerable.** `DECODE_SPEED_PLAN.md` cut
CUDA graphs because they remove per-layer Python that every method in the table
pays, so a win sourced from them is a harness result, not a KV-cache result. That
is correct *for a comparison against FullKV*. It is not a reason to leave 28 ms on
the floor: capture the baselines the same way and report both columns. L1 is the
largest lever by a factor of two and the only one that reaches the stretch target.

---

## 5. Do these in this order

Steps 1–3 are measurements. They come first because §3.3 means nothing is
attributable yet, and because L2/L4 — the second-largest lever — is currently a
guess.

**1. Re-run at the published operating point.** Removes the §3.3 confound.

```bash
export MODEL_PATH=<hf-id-or-path>
CACHE_BUDGET=0.50 LOCAL_WINDOW=64 GATE_RATIO=0.25 CUDA_VISIBLE_DEVICES=0 scripts/run_perf_table.sh
CACHE_BUDGET=0.50 LOCAL_WINDOW=64 GATE_RATIO=1.0  CUDA_VISIBLE_DEVICES=0 scripts/run_perf_table.sh
```

Prediction on record: HEAD lands **0.080–0.085**, i.e. worse than its own 0.0763,
and the matched-config regression against 0.0626 is ~30%. If it lands at or below
0.0763, §3.3's arithmetic is wrong and everything keyed to it needs revisiting.

**2. Re-profile with the fixed denominator.** One run now yields both the kernel
time and the unprofiled wall, so `GPU busy` is finally self-consistent.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_decode.py \
    --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 32 --steps 24 --warmup 12 --top 40
```

Read: the `wall (profiler OFF)` line against the table's `TPOT_steady` — they
should agree within a few percent, and if they do not, the profile is not
measuring the table's cell. Then `GPU busy`, then `ours: two-tier decode` (L3a's
verdict against the 7.03 ms baseline).

**3. Take the chrome trace — the highest-information step in this document.**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_decode.py \
    --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 32 --steps 24 --warmup 12 \
    --trace /tmp/decode.json
```

Open in `chrome://tracing` or Perfetto and split the 28.65 ms of
`elementwise` + `copy/cat` + `index/gather` + `reduction` by launching stack:
`WindowedCache.update` / the compiled eviction (ours) vs `LlamaDecoderLayer`
(model's). **This decides whether L2 exists.** If that block is the model's
unfused RMSNorm/RoPE, L2 is worth ~0 and Path A tops out around 0.070.

**4. Then, cheapest first:** L5 (two env-var runs, already wired), L4 (one-line
edit), L3 verdict from step 2, L2 if step 3 says it is ours, L1 last because it
is the largest change.

### L5, exactly

```bash
CUDA_VISIBLE_DEVICES=0 STICKYKV_DECODE_EXP2=0        scripts/run_perf_table.sh
CUDA_VISIBLE_DEVICES=0 STICKYKV_ROPE_STORE_DTYPE=0   scripts/run_perf_table.sh
```

If the RoPE arm is the cause: **keep it anyway.** It is the fix that closed the
9.5-point qasper regression (`DECODE_SPEED_PLAN.md` introduction). Record the cost.

### L4, exactly

There is no env latch. Neuter the decorator in `modules/windowed_cache/cache.py`:

```python
def _emulating_precision_casts(fn):
    return fn          # A/B ONLY — revert before committing
```

`tests/test_compiled_eviction_numerics.py` will fail (4 of 5) while it is
neutered. That is the safety net working, not a problem to route around: the flag
exists because without it `sketch._q_up` violated `dequant >= x` 115 times over 24
seeds, and that inequality is the only reason the card's score is an upper bound
and therefore the only reason the gate may skip a window. **Measure, then revert.**

---

## 6. What is ruled out, and why

| item | verdict | why |
|---|---|---|
| tuning `quant_gate_ratio` | dead | §3.1 — ≤0.35% even at `n_sel = 0` |
| narrowing the scale/zero grid *for speed* | dead | §3.2 — 0.31% gated. Keep it as a **memory** item: 1.35× Q tokens at the same bytes |
| raising `window_size` to save bytes | dead | `GATE_REGRESSION.md` §5 — 20× the accuracy cost of the int8 grid |
| the `WSUM`/`WMAX` scratch round trip | dead | `GATE_REGRESSION.md` §8.5 — ~0.13 ms/step |
| further cache-side launch cuts | dead | §4 of that doc — 142/step amortized of 1,656 |
| split-K over the sequence | cut | reassociates the softmax; payoff only at B=1, a regime the method does not claim |
| `BLOCK_R` packing | impossible | block-diagonal product `tl.dot` cannot express |
| `eviction_interval`, quant group size | cut | both change which tokens survive |
| multi-GPU per-device grouping | cut | 4×, not 32×, and the model fits on one card |

---

## 7. Open risks, and what is not known

- **The 63% GPU-busy figure mixes two runs.** 48.05 ms came from a profile run,
  76.3 ms from a table run. They *should* be the same cell; nothing has confirmed
  it. Step 2 in §5 closes this, and until it does, treat 63% as approximate.
- **L3a is unmeasured on GPU.** Sorting `sel` is an argument about coalescing and
  L2, pinned on CPU only for what it must *not* change (the selected set, the
  column each score lands on). It reorders an online-softmax accumulation, so
  `out` moves in the last bits as any tile-order change does.
- **The GPU cost of `emulate_precision_casts` has never been measured** and it
  landed in the same commit that fused the quantiser. It is the best unexamined
  candidate for a large constant.
- **`GATE_REGRESSION.md` §5's accuracy numbers are reconstruction error on
  synthetic tensors**, not LongBench. A 0.6% move is far below what LongBench
  resolves.
- **The RoPE-dtype fix perturbed Q-tier keys by ~5e-4** and has not been
  re-scored on LongBench. `DECODE_SPEED_PLAN.md` calls this worth more than any
  remaining millisecond, and that is still true.
- **Accuracy has not been re-measured since the gate landed at all.** Every
  number in this document is speed.

---

## 8. Reproducing the environment

The dev box for this work is CPU-only; every Triton kernel here ships unvalidated
by construction and is pinned by a pure-PyTorch reference that runs on CPU.

```bash
pip install 'transformers==4.47.1' einops numpy pytest torch
python -m pytest tests/ -q        # ~1,050 pass, ~4 min
```

- **Pin `transformers==4.47.1`.** On 5.x, 26 tests in `test_backend_e2e.py` and
  `test_layer_major_evict.py` fail with `AttributeError` at `cache_utils.py:1584`.
  Those failures are environmental, not regressions — they reproduce on a clean
  checkout of any commit.
- `download.pytorch.org` is blocked behind the agent proxy in the cloud container;
  plain PyPI `torch` works.
- The CPU suite cannot exercise `_two_tier_decode_kernel`, `_gate_kernel` or
  `_score_kernel`. `two_tier_window_reference` and `gate_reference` are their
  oracles; a change to either kernel must be mirrored there or it is untested.

---

## 9. One-paragraph summary

Decode is 76.3 ms/step: 48 ms of kernels and 28 ms of host gap. Our cache is
7.8 ms of that — 10% — and the entire KV read is 0.8 ms, about 1%. So the read
gate, the byte accounting behind it, and every remaining bytes-saved idea are
bounded at roughly 1% of TPOT, and the measured gate saving of 0.8% is already
at that ceiling. The time is in three places nobody has attributed: 28.65 ms of
elementwise/copy/index/reduce kernels split between the model and our eviction,
28 ms of host gap across 1,656 launches, and a two-tier kernel running 8.8× off
its own roofline. Take the chrome trace first — it is the only thing that can say
which of those is ours.
