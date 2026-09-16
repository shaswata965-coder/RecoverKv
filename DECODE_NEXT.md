# Decode speed — where the time is, and what to do next

A handoff document. It assumes you have read nothing else, and it names the file
and section for every claim it borrows. Read `GATE_REGRESSION.md` §8 next; this
is the plan built on top of it.

Every number labelled **measured** came off a GPU. Every number labelled
**estimated** is arithmetic on a measured number and is marked with a confidence.
No projection in the table in §4 has been run.

**Scope: optimization only.** §4 carries only directions that are score-neutral
by construction *and* fit the current architecture. Anything that would change
what the cache keeps, what it stores, or how many production paths exist is in
§4c with its reason, and is not to be re-proposed as a speed lever. Correctness
work waits for the benchmark results.

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

## 4. Expected gains — the safe envelope

**Scope rule for this section:** every direction below is *score-neutral by
construction* (it cannot change which tokens the cache keeps) **and** fits the
current architecture (no second production path, no new config surface, no
change to what is stored). Anything that fails either test is in §4c, cut.

Baseline 76.3 ms at the headline cell. All figures **estimated** from the
measured profile; none has been run.

| # | direction | target bucket | est. Δ ms | conf. | why it cannot change scores |
|---|---|---|---|---|---|
| **D1** | Tighten the eviction's demote/promote width | eviction, ~15.4 amortized | **−4 to −8** | med | masked-out lanes are discarded today; a tighter *valid* bound is bit-identical for the lanes that survive |
| **D2** | Stop building the card's dead `eps` field | `build_sketch` | **−1 to −3** | med | nothing on the fused path reads it |
| **D3** | Raise the tile rung if the **gated** kernel was capped | two-tier, 7.03 | **0 to −4** | med | the ladder's own contract — rungs differ in tiling only |
| **D4** | `sel` ascending *(landed, unmeasured)* | two-tier, 7.03 | **−2 to −3** | med | same set, same column per score |
| **D5** | Persistent `wsum`/`wmax` scratch | host path | **−0.5 to −1.5** | low | same values written to the same shapes |
| **D6** | Drop the redundant `SEL` scalar reloads | two-tier, 7.03 | **0 to −0.5** | low | reloads a value already in a register |

**Overlaps — do not add these up.**

- **D1 and D2 overlap.** D2's saving lives *inside* the padded width D1 shrinks,
  so whichever lands second is worth less. Together: **−5 to −10**.
- **D3, D4 and D6 all target the same 7.03 ms kernel.** Together: **−2 to −4**.

| realistic total | est. TPOT | vs now | vs published 0.0626 | vs FullKV 0.086 |
|---|---|---|---|---|
| −7.5 to −15.5 ms | **0.061 – 0.069** | −9% to −20% | ties at best | 1.25–1.41× faster |

**State this plainly: the safe envelope does not reach 0.050.** The stretch target
needed the host gap (28.25 ms, 37% of the step), and closing that needs CUDA
graphs, which §4c cuts. Within the current architecture, ~0.061 is the floor.

---

## 4b. The safe directions, in detail

### D1 — the eviction's demote/promote width is the worst case, not the count

**The largest item here, and unambiguously ours.** `should_evict` fires every
`window_size = 8` steps, and `DECODE_SPEED_PLAN.md` §3 measured an eviction step
at **~195 ms against a ~72 ms steady step** — ~15.4 ms/step amortized, **~20% of
TPOT**, twice the decode kernel. It is not read traffic, so §3.2's 1% ceiling does
not apply.

In `_evict_two_tier_impl` §3b/§3c:

```python
p_max = min(self.resolved.N_q, n_fp)
prom_slot, prom_valid  = self._compact(fp_slot, fp_prom, p_max)
fresh_wid, fresh_valid = self._compact(wids, fresh, n_q)   # n_q = 64 here
```

`_compact`'s docstring says why the width is the bound: *"Allocating by count
would therefore need a host sync to learn the max; allocating by rank into a
worst-case-width tensor does not."* So every eviction un-rotates, quantizes and
sketches a `[B, n_q, H_kv, ws, D]` tensor — **`n_q = 64` at budget 0.20, `184` at
the published 0.50** — when at steady state only a handful of windows per row are
actually fresh: one new window is sealed per `ws` steps, plus whatever crosses the
tier boundary on re-ranking. At `B=32` that is a 33.5 MB `k_post` gather per layer
plus ~67 MB fp32 temporaries through a ~10-op quantiser and a ~60-op card build —
roughly **1 GB per layer per eviction, of which ~98% is masked away.**

**Why it is score-neutral.** The invalid lanes are already discarded. The same
docstring: *"The invalid lanes dequantize garbage from free slots and are dropped
by the mask below — bit-identical for the valid ones, since every op here is
per-window or per-token pointwise."* A smaller width that is **still a valid upper
bound** changes nothing about the surviving lanes.

**The one invariant it touches, stated honestly.** A tight width needs the actual
count, which needs one `.item()` per eviction. The body's docstring says *"No
Python loops and no host syncs"* — but that rule was written against host-side
Python sets over `.tolist()`ed window ids, *per layer, inside the decision logic*.
This is one scalar read on an eviction step, i.e. **one sync per eight steps**,
and on the layer-major path one sync for all 32 layers at once. A microsecond-scale
sync against milliseconds of overcompute is the trade; the rule's actual target —
per-layer host logic that cannot carry a batch axis — is untouched.

**Do not guess the bound instead.** `_compact` routes overflow lanes to a dump
column, so a bound that is ever too small **silently drops work**. The width must
come from the data, not from config arithmetic.

**Shape churn is the real implementation risk.** A fresh count that varies
per eviction gives `torch.compile` a new shape every time and it will recompile
itself to death. Round the measured count **up to a small ladder of rungs**
(e.g. 1, 2, 4, 8, 16, then `n_q`) so the eviction sees a handful of shapes. The
first eviction, where everything genuinely is fresh, still takes the full width.

### D2 — the card's `eps` field is built every eviction and never read

`build_sketch` ends with:

```python
recon = mu_h.unsqueeze(-2) + t_h.unsqueeze(-1) * v_h.unsqueeze(-2)
e_q, e_s = _q_up((k - recon).norm(dim=-1))
```

`recon` materializes a full `[N, H, ws, D]` fp32 tensor and is the single most
expensive step in the card build.

**Nothing on the fused path consumes it.** `_gate_triton` unpacks the card as
`…, _e_q, _e_s` and discards both. `gate_reference` throws away `bound`
(`_, logmass, est = gate_and_score(…)`). The only consumer is
`store.gate_and_select`, which is called from `gated_decode.py` — the CPU
reference, not production — and even there `cache._gate_ctx` **refuses a finite
margin**, so the threshold keeps everything and the cap decides on `est` alone.
`_gate_kernel`'s own docstring already says this about its half: *"the bound cost
a `[B, H_q, NW]` fp32 store and the whole `eps` field of every card, to be
discarded by `_gate_triton`."* The build side was never followed through.

**Keep the capability, skip the work.** Do **not** delete `e_q`/`e_s` from the
`Sketch` contract — `quant_gate_margin` is a real config field, and a finite
margin is what `eps` exists for. Thread a `need_eps` flag from the resolved margin
down through `store.demote_many` to `build_sketch`, and return zero-filled fields
when it is off. Architecture unchanged, config surface unchanged, and the default
configuration — the only one the fused path will accept — stops paying for it.

### D3 — read the tile rung, and raise it only if the gated kernel was capped

The lever `DECODE_SPEED_PLAN.md` §2 designated as score-free, and it has still
never been read. `decode_kernel.py` picks the largest tile that fits in shared
memory, stepping down `_FIT_LADDER = [(64,2), (32,2), (32,1), (16,2), (16,1)]` on
`OutOfResources`, and the contract is explicit: *"every rung computes
bit-identical results, only the tiling and the pipeline depth differ."*

| rung | Q-tier iterations at `ws=8` |
|---|---|
| `target_keys=64` | 23 |
| `target_keys=32` | 45 |
| `target_keys=16` | 90 |

**The reason to look now is new.** `gated` joins the fit signature:

```python
sig = (int(ws), int(D), int(BLOCK_R), int(W_phys > 0), bool(gated))
```

with the comment *"GATED is a constexpr, so the two variants are separate compiles
with different register and staging pressure, and a rung that fit one is not
evidence about the other."* The gated variant stages strictly more — the `SEL`
loads, `LOGM`, the prologue and the §5 pass. **So the gated kernel may have
dropped to a lower rung than the ungated one used, and that would be a mechanism
for the whole 7.03 ms that has nothing to do with gathers** — 90 serial iterations
instead of 23 is a latency story, which is exactly what an 8.8×-off-roofline
kernel looks like when it is not bandwidth-bound.

**This is free to check.** The kernel prints
`[StickyKV] fused decode tiling: target_keys=… num_stages=…` once per geometry.
Read it off the profile run. If it says 64, close the item. If it says 16 or 32,
the score-neutral way up is to free shared memory **without touching the
arithmetic**: shrink `BLOCK_R` (query-head rows are independent, so this cannot
reassociate anything), or give the fp and Q tiers separate `BLOCK_T` so the
memory-hungry Q tier shrinks alone. **Do not** stage `vv` in fp16 for its
`tl.dot` — `DECODE_SPEED_PLAN.md` suggests it, but it changes the dot's precision
and therefore the scores, so it fails this section's scope rule.

### D4 — `sel` ascending (landed, unmeasured)

In `gate_kernel._sorted_pick`. `topk` returns indices in descending *score* order,
arbitrary in *column* space, so each tile gathers `BLOCK_NW` segments scattered
across the tier; ascending makes them monotone with a mean stride of `1/ratio`
windows. Same set, same score per column, same eviction decisions. Verified on CPU:
`[14, 13, 19, 6, 15]` → `[6, 13, 14, 15, 19]`, identical as a set.

Needs nothing but a re-profile of the `ours: two-tier decode` bucket against 7.03.

### D5 — persistent `wsum`/`wmax` scratch

`_decode_triton` allocates `out`, `wsum` and `wmax` fresh on every call — 96
allocator calls per step across 32 layers, on a path where the host is 37% of the
step. The shapes are stable at steady state, so a cache-owned scratch buffer
reused across layers would remove them. Same values, same shapes: score-neutral.

**Carry the precedent.** `bfbdb1f` — *"revert the reusable ctx dict — it caused
the narrativeqa OOM"* — is the same idea failing once already. Reused buffers hold
memory the allocator would otherwise recycle, and the headline cell already peaks
near 48 GB. Size it once against the worst cell and check `peak_GB` in the table,
or leave this one alone; it is the smallest item here.

### D6 — the redundant `SEL` scalar reloads

In the Q loop's score-store, `col` re-loads from `SEL` for each of `BLOCK_NW`
lanes a value the loop head already has in `widx`:

```python
widx = tl.load(SEL + … + tl.minimum(slot, n_sel - 1))      # loop head, [BLOCK_T]
for j in tl.static_range(BLOCK_NW):
    col = n_body_win + tl.load(SEL + … + tl.minimum(w0 + j, n_sel - 1))
```

Listed for completeness and sized honestly: `SEL` is a tiny array that will be
L1-resident, so this is likely **free already**. Only worth touching if the trace
shows the Q loop stalled on scalar loads.

---

## 4c. Cut — breaks the architecture, the numerics, or both

Removed from the plan, with the reason, so they are not re-proposed:

| cut | why |
|---|---|
| **Drop the read gate / ungated read path** | Reintroduces a second production path that `676c781` collapsed on purpose (`config.py`: *"there is no way to ask for the ungated one"*), and changes what the decode reads and how skipped windows are scored — so it moves eviction decisions. Architecture **and** numerics. |
| **CUDA graphs on the decode step** | The largest lever (28.25 ms) and the only route to 0.050, but capture needs static shapes and no host syncs, and the cache has neither: `n_active` moves every eviction, evictions are ragged per row, and `store.version` invalidates memos mid-step. A fundamental restructure. Also already cut in `DECODE_SPEED_PLAN.md` as a harness result rather than a KV-cache one. |
| **Neutering `emulate_precision_casts`** | Correctness-breaking by construction — 4 of 5 tests in `test_compiled_eviction_numerics.py` fail by design, `_affine_quantize` diverges from eager on 11–14 of 24 seeds, and `_q_up`'s `dequant >= x` is violated 115 times. It is a correctness fix, so its cost is a price, not a saving. **D2 is the legitimate version of this**: remove a client rather than the guard. |
| **Restoring the `_affine_quantize` graph break** | A revert of a deliberate architectural change, and it cannot be evaluated as an optimization until the trace says whether the eviction graph fuses at all. Keep it as a *measurement* arm only (§5). |
| **int8 / fp8 scale-zero grid** | Changes the stored grid, so reconstruction and therefore scores move. Worth 0.24 ms (0.31%) on speed — it is a **memory** item (1.35× Q tokens at the same bytes), not a speed one. |
| **`STICKYKV_DECODE_EXP2=0` / `STICKYKV_ROPE_STORE_DTYPE=0`** | Reverts of numerics changes, one of which is the fix that closed the 9.5-point qasper regression. Measurement arms, never optimizations. |
| **Staging `vv` in fp16 for its `tl.dot`** | Suggested in `DECODE_SPEED_PLAN.md` §2 as the way to raise the tile rung; it changes the dot's precision and therefore the scores. Use `BLOCK_R` or a split `BLOCK_T` instead (D3). |
| **`eviction_interval`, quantization group size, `window_size`** | All change which tokens survive or how they are represented. |
| **Split-K over the sequence, `BLOCK_R` packing, multi-GPU grouping** | Already cut upstream: reassociates the softmax; not expressible in `tl.dot`; 4× not 32× on a model that fits on one card. |

## 5. Do these in this order

Steps 1–3 are measurements and cost no code. They come first because §3.3 means
nothing is attributable yet, and because **D1 and D3 — the two largest safe
directions — are each gated on one number a measurement produces.**

**1. Re-profile with the fixed denominator, and read two lines off it.**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_decode.py \
    --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 32 --steps 24 --warmup 12 --top 40
```

Read, in order:

- `[StickyKV] fused decode tiling: target_keys=…` — **this is the whole of D3.**
  64 closes the item; 32 or 16 means the gated kernel is running 45 or 90 serial
  Q-tier iterations instead of 23, and that is the largest kernel-side win here.
- `wall (profiler OFF)` against the table's `TPOT_steady`. They should agree
  within a few percent; if not, the profile is not measuring the table's cell and
  the 63% busy figure in §2 is not trustworthy.
- `GPU busy`, then `ours: two-tier decode` against 7.03 ms — **D4's verdict.**

**2. Take the chrome trace — the highest-information step in this document.**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_decode.py \
    --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 32 --steps 24 --warmup 12 \
    --trace /tmp/decode.json
```

Open in `chrome://tracing` or Perfetto and split the 28.65 ms of
`elementwise` + `copy/cat` + `index/gather` + `reduction` by launching stack:
`WindowedCache.update` and the compiled eviction (ours) vs `LlamaDecoderLayer`
(the model's). **This sizes D1.** If that block is mostly the model's unfused
RMSNorm/RoPE, D1 is worth far less than the estimate and the safe envelope tops
out nearer 0.070.

While the trace is open, also note whether the eviction's kernels are fused or
still land as the sixteen separate elementwise launches `DECODE_SPEED_PLAN.md` §3
described. That is the one thing worth knowing about the compiled eviction, and
`--compile-evict 0` bounds it in one more run if the trace is ambiguous.

**3. Re-run at the published operating point.** Removes the §3.3 confound so the
D-series has a valid reference to be measured against.

```bash
export MODEL_PATH=<hf-id-or-path>
CACHE_BUDGET=0.50 LOCAL_WINDOW=64 GATE_RATIO=0.25 CUDA_VISIBLE_DEVICES=0 scripts/run_perf_table.sh
```

Prediction on record: HEAD lands **0.080–0.085**, i.e. worse than its own 0.0763,
and the matched-config regression against 0.0626 is ~30%. If it lands at or below
0.0763, §3.3's arithmetic is wrong and everything keyed to it needs revisiting.
Note that D1 scales with `n_q`, which is 184 here against 64 at budget 0.20 — so
the padded-width waste is ~3× larger at the published operating point, and D1 is
worth correspondingly more there.

**4. Then implement, in this order:**

| order | direction | why here |
|---|---|---|
| 1 | **D2** (`eps`) | Cheapest real change, no measurement needed, and it shrinks D1's payload before D1 lands |
| 2 | **D3** (tile rung) | Free if step 1 said 64; if capped, the biggest kernel win |
| 3 | **D1** (demote width) | The largest item, and the most implementation work |
| 4 | **D4** | Already landed — this is just recording the verdict from step 1 |
| 5 | **D5 / D6** | Only if the trace points at them; both are small and D5 carries the `bfbdb1f` OOM precedent |

Re-run the table after each, not at the end. Six changes measured together is one
data point; six measured separately is six.

---

## 6. Also dead — on the merits, not on architecture

§4c lists what was cut for breaking the architecture or the numerics. These are
different: they are compatible with both, and simply are not worth anything.

| item | why |
|---|---|
| tuning `quant_gate_ratio` | §3.1 — ≤0.35% even at `n_sel = 0`, and 0.09% of headroom left from 0.25 |
| the `WSUM`/`WMAX` scratch round trip | `GATE_REGRESSION.md` §8.5 — ~0.13 ms/step, 0.2% |
| further cache-side launch cuts | `GATE_REGRESSION.md` §4 — 142/step amortized of 1,656 |
| shrinking the KV read by any means | §3.2 — the whole read is 0.80 ms, 1.05% of the step |

---

## 7. Open risks, and what is not known

- **The 63% GPU-busy figure mixes two runs.** 48.05 ms came from a profile run,
  76.3 ms from a table run. They *should* be the same cell; nothing has confirmed
  it. Step 1 in §5 closes this, and until it does, treat 63% as approximate.
- **D4 is unmeasured on GPU.** Sorting `sel` is an argument about coalescing and
  L2, pinned on CPU only for what it must *not* change (the selected set, the
  column each score lands on). It reorders an online-softmax accumulation, so
  `out` moves in the last bits as any tile-order change does — the same class of
  perturbation the tile ladder already accepts.
- **D1's estimate rests on the fresh count actually being small.** One new window
  is sealed per `ws` steps, but boundary crossers from re-ranking are not bounded
  by anything, and `_compact`'s docstring notes rows diverge (*"row 0 may demote 4
  windows while row 1 demotes none"*). If the real count is routinely close to
  `n_q`, D1 is worth little. The trace and one printed count settle it.
- **The GPU cost of `emulate_precision_casts` has never been measured.** It is not
  a lever (§4c), but it is an unknown sitting inside D1's and D2's target.
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
7.8 ms of that — 10% — and the entire KV read is 0.8 ms, about 1%, which bounds
the read gate and every other bytes-saved idea at roughly 1% of TPOT. Within the
current architecture the time worth chasing is in two places: the **eviction**,
~15.4 ms/step amortized, where every eviction quantizes and sketches a
worst-case-width tensor of which ~98% is masked away (D1) and builds a card field
nothing reads (D2); and the **two-tier kernel**, 7.03 ms at 8.8× off roofline,
where the gated variant may have dropped to a lower shared-memory tile rung than
the ungated one (D3) and is gathering in an order that has just been fixed (D4).
That envelope is worth an estimated 0.061–0.069 and does **not** reach 0.050 —
the stretch target needed the host gap, and closing that needs CUDA graphs, which
§4c cuts. Two commands decide almost everything: read `target_keys=` off the
profile, and take the chrome trace.
