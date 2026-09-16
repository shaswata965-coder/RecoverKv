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
| L3b | Drop the gate from the read path | two-tier + gate 7.84 | **−3 to −4.5** | 0.072–0.073 | med | pass `sel=None`; re-profile the same bucket |
| L3b+ | …and the card build it deletes from the eviction | `build_sketch`, eviction-side | **−3 to −6** | 0.066–0.070 | low | `sketch_enabled=False`; time an eviction step |
| L4a | A/B `emulate_precision_casts` alone | reduced tensors only | **0 to −1** | ~0.076 | low-med | neuter the decorator; see §4b for why it is small |
| L4b | Did tracing `_affine_quantize` actually fuse? | eviction graph | **0 to −8** | 0.068–0.076 | low | `--compile-evict 0` first; it bounds both arms |
| L5 | The two latched arms (`DECODE_SPEED_PLAN` §"Open") | ~5 ms constant | **0 to −5** | 0.071–0.076 | med | two env-var runs, already wired |
| L6 | int8 scale/zero grid (`GATE_REGRESSION.md` §5) | Q bytes | **−0.2** | 0.076 | high | it is a memory item, not a speed one |

**L3a and L3b are alternatives, not additive** — they target the same kernel.
L2 and L4b overlap; L4b is a plausible *cause* of L2. **L4a is a price, not a
saving** — the flag is a correctness fix, so the only legitimate way to stop
paying it is to remove its clients, which L3b and L2's target 2 both do. **L3b+ overlaps L2's target 2**
— both delete card work from the eviction, so count it once. See §4b for what each
one actually does; L3b is the only lever here that is also expected to *improve*
accuracy, and L2 is the largest one that is unambiguously ours.

Two combined paths:

| path | levers | est. TPOT | vs now | vs published 0.0626 | vs FullKV 0.086 |
|---|---|---|---|---|---|
| **A — no CUDA graphs** | L3 + L2 + L4b/L5 + L6 | **~0.062** | −19% | ties | 1.38× faster |
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

## 4b. L3b and L2, in detail

These two are the ones worth understanding before touching anything, because
they overlap: **L3b deletes a large part of L2's target as a side effect.**

### L3b — drop the read gate from the decode path

**What it is.** Stop passing `sel` to `_two_tier_decode_kernel`, so the Q-tier
loop walks `n_active` windows with an affine `widx = w0 + t_win` instead of
dereferencing `SEL`. The kernel already supports this: `GATED` is a
`tl.constexpr`, and `sel=None` compiles the ungated variant that shipped before
the gate existed. Nothing in the kernel has to be written.

**What it removes.** Four separate things, only the first of which is obvious:

1. **The `SEL` indirection.** `GATE_REGRESSION.md` §2b: gated, `KS`, `KZ`, `KC`,
   `VC`, `VS`, `VZ` and `COS`/`SIN` all become gathers with an address dependency
   on the `SEL` load. Ungated they are contiguous vector loads.
2. **The `GATED` prologue** (`decode_kernel.py` §3): a pass over every Q column
   seeding `WSUM = 0`, `WMAX = -inf` so the epilogue can detect what was skipped.
3. **The §5 fill pass**: a second full-tier pass writing card-estimated scores
   onto skipped windows. Ungated there are no skipped windows, so the epilogue's
   `num`/`gmx`/`gsm` reductions and the whole `LOGM` tensor disappear too.
4. **`build_sketch` from the eviction.** This is the one that is not on the decode
   path at all. `store.sketch_enabled` gates the card build in
   `cache._evict_two_tier_impl` §3c; with no gate there are no cards to build, and
   `build_sketch` is a ~60-op chain (mean, deviation, norm, argmax, two gathers,
   three `_q_sym`, three `_dq_sym`, a full `[N,H,ws,D]` `recon` outer product,
   another norm, `_q_up`) running on the same padded width as the quantiser — see
   L2. **The table's `−3 to −4.5 ms` for L3b counts only items 1–3.** Item 4 is
   eviction-side and plausibly worth as much again.

**What it costs.** The Q-tier loop visits 64 windows instead of 16, so the read
grows by 405 MB/step — **0.20 ms** (§3.2). That is the entire downside on the
traffic axis, and it is noise.

**It should also be accuracy-positive**, which is the part that makes this more
than a speed trade. The gate is an approximation in two places: skipped windows
do not contribute to the attention output, and their eviction scores are the
card's rescaled `logmass` estimate rather than the real softmax mass. Ungated,
both are exact. Removing the gate removes approximation error; it does not add
any. Nothing about the stored cache changes — the tier is quantized either way —
so `peak_GB` and `steadyKV_GB` are untouched.

**The real cost is structural, not numerical.** `676c781` deliberately collapsed
this to one production path, and `config.py` derives
`quant_sketch_enabled = q > 0.0` with the comment *"the gate IS the read path
wherever there is a Q tier, and there is no way to ask for the ungated one."*
L3b means reintroducing that choice — a config field threaded to
`store.sketch_enabled` and `cache._gate_ctx`. Small and localized, but it is a
second path returning, and that was removed on purpose.

### L2 — the eviction, which is 20% of TPOT

**What it is.** `EvictionPolicy.should_evict` fires every `window_size = 8` steps.
`DECODE_SPEED_PLAN.md` §3 measured an eviction step at **~195 ms against a ~72 ms
steady step** — so the eviction itself is ~123 ms of extra work, **~15.4 ms/step
amortized, ~20% of TPOT**. That is more than twice the two-tier decode kernel,
and it is not read traffic, so §3.2's 1% ceiling does not apply to it.

It also shows up as launches: **493 on an eviction step against 80 steady**
(`GATE_REGRESSION.md` §4).

**Target 1 — the demote/promote width is the worst case, not the count.** This is
the big one. In `_evict_two_tier_impl` §3b/§3c:

```python
p_max = min(self.resolved.N_q, n_fp)
prom_slot, prom_valid = self._compact(fp_slot, fp_prom, p_max)
fresh_wid,  fresh_valid = self._compact(wids, fresh, n_q)   # n_q = 64 here
```

`_compact`'s own docstring says why: *"Allocating by count would therefore need a
host sync to learn the max; allocating by rank into a worst-case-width tensor does
not."* So every eviction quantizes, un-rotates and sketches a `[B, n_q, H_kv, ws,
D]` tensor — `n_q = 64` at budget 0.20, **`n_q = 184` at the published 0.50** —
when at steady state only a handful of windows per row are actually fresh (one new
window is sealed every `ws` steps, plus whatever crosses the tier boundary on
re-ranking). **The invalid lanes are dequantized, rotated, quantized and sketched,
then masked away.** At `B=32` that is a 33.5 MB `k_post` gather per layer and
~67 MB fp32 temporaries through a ~10-op quantiser and a ~60-op card build,
roughly 1 GB of traffic per layer per eviction — of which ~98% is garbage.

The fix is a bucketed width: sync once per eviction (1 step in 8) to learn
`fresh.sum(1).max()`, round up to a small ladder of rungs so `torch.compile` sees
a handful of shapes rather than a new one each time, and use that instead of
`n_q`. One host sync every eighth step against a ~64× overcompute is a good trade;
the "no host syncs" rule in that docstring was written against a per-layer
`.tolist()` in the inner loop, which is a different thing. The first eviction,
where everything genuinely is fresh, still takes the full width.

**Target 2 — the `eps` field is computed every eviction and never read.**
`build_sketch` ends with:

```python
recon = mu_h.unsqueeze(-2) + t_h.unsqueeze(-1) * v_h.unsqueeze(-2)
e_q, e_s = _q_up((k - recon).norm(dim=-1))
```

That `recon` materializes a full `[N, H, ws, D]` fp32 tensor and is the single
most expensive step in the card build. Nothing on the production path reads what
it produces: `_gate_triton` unpacks the card as `…, _e_q, _e_s` and discards both,
and `gate_reference` throws away `bound` (`_, logmass, est = gate_and_score(…)`).
The only consumer is `store.gate_and_select`, which is called from
`gated_decode.py` — the CPU reference, not the fused path — and even there
`_gate_ctx` refuses a finite margin, so the threshold keeps everything and the cap
decides on `est` alone. `gate_kernel._gate_kernel`'s docstring already says this
about its own half of it: *"No `bound` is computed … the bound cost a
`[B, H_q, NW]` fp32 store and the whole `eps` field of every card, to be discarded
by `_gate_triton`."* The build side was never followed through.

There is a sting in the tail: `_q_up` is exactly the function whose `dequant >= x`
guarantee `emulate_precision_casts` was turned on to protect (115 violations over
24 seeds). **A flag whose GPU cost is unmeasured is guarding a field nothing
reads.** Deleting `e_q`/`e_s` removes one of the four reasons that flag exists —
which is also L4a.

**Target 3 — whether the graph fuses at all.** `DECODE_SPEED_PLAN.md` §3 reports
the eviction landing as *"sixteen separate multi-millisecond elementwise kernels —
round, clamp, div, sub, where, scatter — which is the signature of a graph that
traced and then fused nothing."* `5186353` removed the `torch._dynamo.disable` on
`_affine_quantize` to fix that, but in the same commit added
`emulate_precision_casts`, whose GPU cost has never been measured. Whether the
fusion actually took is unknown, and the chrome trace (§5 step 3) answers it.

**Order within L2:** target 2 first (a deletion, ~10 lines, no behaviour change on
the production path), then target 3 (a measurement), then target 1 (the real work,
and the real win).

### L4 — `emulate_precision_casts`, and what it is really asking

**Revised down from the table's first draft.** Writing §4b forced a closer read of
what the flag actually touches, and the honest estimate for the flag *alone* is
much smaller than the `0 to −8 ms` first entered. The `0 to −8` belongs to a
different question that landed in the same commit. Both are below.

**What the flag does.** `cache._emulating_precision_casts` wraps the compiled
eviction in `torch._inductor.config.patch("emulate_precision_casts", True)`.
Inductor normally *elides* an `fp32 -> fp16 -> fp32` round trip and keeps the
wider value in a register. Three quantisers in the eviction take that round trip
**deliberately**, to fit codes to the fp16-**stored** grid so the grid the codes
were fit to is bit-identical to the grid every later dequant reads (design §2).
The flag forces Inductor to emit the truncation.

**Why the direct cost is almost certainly near zero.** Every value the flag
narrows is a **reduced** tensor, not a data tensor:

| quantiser | narrowed tensor | shape | size vs its data |
|---|---|---|---|
| `_affine_quantize`, keys | `scale`, `zero` | `[N, H_kv, D]` | **1/8** (reduced over `ws`) |
| `_affine_quantize`, values | `scale`, `zero` | `[N, H_kv, ws]` | **1/128** (reduced over `D`) |
| `sketch._q_sym` ×3 | `scale` | `[N, H, 1]` | 1/D or 1/ws |
| `sketch._q_up` | `scale` | `[N, H, 1]` | 1/ws |

Two convert instructions on between 1/8 and 1/128 of the elements, pointwise, and
they fuse into kernels that already run. The per-call overhead is a dict swap —
Inductor keys its code cache on config, so the wrapped callable compiles once.
**Estimate for the flag alone: 0 to −1 ms/step, low confidence but bounded.**

**The real question in that commit is the other half.** `5186353` did two things.
It turned the flag on, *and* it removed the `@_compile_disable` from
`_affine_quantize` so the int2 quantiser is traced into the eviction graph instead
of sitting behind a graph break. The second is the structural change — the graph
got substantially bigger, and Inductor's fusion, scheduling and memory planning
for the whole eviction changed with it. `DECODE_SPEED_PLAN.md` §3 predicted that
would *help*, because the graph break was what split the eviction into "sixteen
separate multi-millisecond elementwise kernels". The profile taken afterwards
still shows `elementwise` at 14.20 ms. So either it did not help, or that block
was never ours — which is exactly what the chrome trace decides.

The two are coupled in one direction only: `quantizer._quant_may_be_traced()`
requires `hasattr(inductor_config, "emulate_precision_casts")`, so the quantiser
cannot be traced on a build without the knob. But the knob can be *neutered*
while tracing continues, because that check is `hasattr`, not the value. So all
four arms are reachable.

**The four arms, and what each one answers:**

| arm | how | answers |
|---|---|---|
| 1. current | — | baseline |
| 2. flag off | `_emulating_precision_casts` → `return fn` | the flag's cost, alone |
| 3. graph break back | force `_compile_disable = torch.compiler.disable` | whether tracing the quantiser helped |
| 4. eager | `--compile-evict 0` | what compiling the eviction buys at all |

Arm 4 is already wired and costs one flag. Run it first: it bounds arms 2 and 3
together, and if compiled and eager are close, the whole compiled-eviction story
needs revisiting rather than tuning.

**Arm 2 is correctness-unsafe and is a speed measurement only.**
`tests/test_compiled_eviction_numerics.py` fails (4 of 5) while the flag is
neutered — that is the safety net working. Without it, `_affine_quantize`'s packed
codes diverge from eager on 11/24 seeds (keys) and 14/24 (values), and
`sketch._q_up`'s `dequant >= x` guarantee is violated 115 times in 24 seeds. That
inequality is the only reason the card's Cauchy–Schwarz score is an **upper**
bound, and therefore the only reason the gate may skip a window at all. **Measure
speed, then revert. Draw no accuracy conclusion from that run**, and do not
report its LongBench.

**The status this lever really has.** `emulate_precision_casts` is a correctness
fix, not an optimisation to be removed. If it costs time, that is a price, not a
saving — the only legitimate ways to stop paying it are to remove its *clients*
or to restore the graph break (arm 3). Which is where this connects to §4b:

- **L2 target 2** (delete the dead `eps` field) removes `_q_up` — the client with
  the 115 violations.
- **L3b** (no gate → no cards) removes `_q_sym` and `_q_up` entirely, leaving
  `_affine_quantize` as the flag's only remaining client.

So doing §4b's work first **shrinks L4 rather than competing with it.** Run arm 4
now because it is free; leave arms 2 and 3 until after the trace, which may make
them unnecessary.

## 5. Do these in this order

Steps 1–3 are measurements. They come first because §3.3 means nothing is
attributable yet, and because L2/L4b — the second-largest lever — is currently a
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

**4. Then, cheapest first:** L4b arm 4 (`--compile-evict 0`, one flag, and it
bounds the whole compiled-eviction question), L5 (two env-var runs, already
wired), L2's target 2 (delete the dead `eps` field), the L3 verdict from step 2,
then L2's target 1. L1 last, because it is the largest change. L4a only if the
trace says the eviction graph is still unfused — §4b explains why it is small.

### L5, exactly

```bash
CUDA_VISIBLE_DEVICES=0 STICKYKV_DECODE_EXP2=0        scripts/run_perf_table.sh
CUDA_VISIBLE_DEVICES=0 STICKYKV_ROPE_STORE_DTYPE=0   scripts/run_perf_table.sh
```

If the RoPE arm is the cause: **keep it anyway.** It is the fix that closed the
9.5-point qasper regression (`DECODE_SPEED_PLAN.md` introduction). Record the cost.

### L4a, exactly

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
