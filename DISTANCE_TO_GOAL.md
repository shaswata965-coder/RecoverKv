# Distance to the goal

Updated 2026-09-24 (§13: the observation path reads what the decode reads). §12 as of 2026-09-23. §11 as of 2026-09-22. Sections 1–10 as of 2026-09-18/19.

**The goal:** beat Flash FullKV, int2 KIVI and QEvict across all six shape/batch
cells. Today we beat Flash in **one of six**. At our best numbers ever we beat it
in **two of six**. KIVI and QEvict have never been run — there is no
implementation of either.

**The headline finding of this session:** the read gate — the whole point of the
current design — does not pay, and we now know exactly why in one number. A
window's summary card costs 400 bytes. The 8 tokens it summarises cost 512 bytes.
Skimming is not cheap when the thing you are skimming was already crushed to 2
bits per number. Everything below follows from that.

> Section references like `TARGET_GAP §2` point at working notes deleted in
> `0974687`. They are kept as provenance for each measurement.

---

## 1. Where we stand

TPOT_steady in seconds, 4096-token prompts unless stated. **Flash** is FullKV
with `flash_attention_2`. **Best** is the published table (budget 0.50, local 64).
**Now** is the 2026-09-16 run (budget 0.20, local 128).

| shape | B | Flash | best ever | vs Flash | **now (v9, 2026-09-19)** | vs Flash |
|---|---|---|---|---|---|---|
| 4096/256 | 32 | 0.086 | 0.0626 | **1.37× faster** | **0.0589** | **1.46× faster** |
| 2048/512 | 32 | 0.055 | 0.0531 | 1.04× faster | 0.0561 | 1.02× slower |
| 1024/1024 | 32 | 0.043 | 0.0488 | 1.13× slower | 0.0562 | 1.31× slower |
| 4096/256 | 1 | ~0.024 | 0.0471 | 1.96× slower | 0.0553 | 2.30× slower |
| 2048/512 | 1 | ~0.024 | 0.0468 | 1.95× slower | 0.0557 | 2.32× slower |
| 1024/1024 | 1 | ~0.024 | 0.0467 | 1.95× slower | 0.0551 | 2.30× slower |

Three things this says plainly:

**"Beat Flash across all" was never close.** Even at our best, four of six cells
lost. The one comfortable win is the long-prompt, large-batch cell — which is the
cell the method is *for*, and the only one the headline ever quoted.

**The "now" column moved for the first time since 2026-09-16, and only because
the 2026-09-16 numbers were taken at a different operating point.** The v9 row
is `bytes` mode at budget 0.20 / local 128, the same as the 0.0763 it replaces,
so those two are comparable: 0.0763 → 0.0589 at the headline cell. The `best
ever` column is budget 0.50 / local 64 and is **not** comparable to either.

**Almost none of that came from this session's code.** §10.0.1 has the
controlled comparison — against an identically-configured run (v8: 0.0570 at the
headline cell), the current head is a **wash**: four cells worse by +0.9% …
+3.3%, one unchanged, one better by −1.4%. What the session produced is
measurement,
not speed: §10.0 is the summary, and three of its findings are corrections of
claims elsewhere in this document.

---

## 2. The one thing to understand

Everything in the gate's design rests on an assumption that turns out to be
false: that a summary of 8 tokens is much cheaper to read than the 8 tokens.

Per KV head, for one window of 8 tokens:

| | bytes |
|---|---|
| the 8 tokens themselves (keys and values, at **2 bits** per number) | **512** |
| the grid needed to decode them | 292 |
| **a full window** | **804** |
| **the card that summarises those 8 tokens** | **400** |

The card is **78% of the token data it replaces.**

The reason is a precision mismatch. The tokens are stored at **2 bits** per
number. The card is stored at **8 bits** per number, and it holds three
full-length vectors — the mean, the deviation direction, and the value centroid,
128 numbers each. One of those vectors costs 128 B, which is what *four tokens*
of one tier cost. Three of them, plus scales, and the summary costs nearly what
the thing it summarises costs.

**The summary is written at four times the precision of what it summarises.**
That is the whole problem, and it is fixable — see §6.1.

### What that does to the gate

The card of *every* window must be read on *every* step. That is what selecting
means. So per window, per step:

```
cost of the gate  =  every card  +  the windows you actually open
                  =  400  +  ratio × 804
```

| read ratio | cost vs reading everything |
|---|---|
| 0.25 (shipped) | 75% |
| 0.50 | 100% — break-even |
| 1.00 | 150% |

**Break-even is a 0.50 read ratio.** At the shipped 0.25 the gate saves 25% of
the compressed tier's bytes — and the compressed tier is only ~13% of what the
decode kernel reads, because the full-detail tier is the other 87%. So the
saving is roughly **3% of the kernel's traffic**.

### It has been measured, and it ties

`--gate-ratio 1.0` (read everything, no selection) was run against 0.25 at
identical budget, shape and batch. **They tie: 0.2–1.6% of TPOT** (`130de24`).
Reading a quarter of the windows instead of all of them is worth nothing
measurable.

### And the gate is not free

Two costs it adds, both measured:

* **Scattered reads.** Ungated, the kernel walks windows in order and the
  hardware fetches them in long contiguous runs. Gated, each window's address is
  looked up at runtime, so every field becomes a scattered fetch. The kernel
  measures **7.03 ms against a 0.58 ms roofline — 12× off**. The model's own
  matrix math runs at 1.4× its roofline.
* **An extra full-tier pass.** The score-fill epilogue walks every window, not
  the selected ones, plus a prologue that seeds every column.

**So: save ~3% of traffic, pay for scattered reads on the rest plus two extra
full-tier passes. The gate is currently a net negative, and the cause is the
memory layout, not the idea.**

---

## 3. Why keeping MORE cache ran FASTER

This looks backwards and is worth stating, because it rules out a whole class of
explanation.

Both runs used `tokens` mode (the budget-mode default only changed on 2026-09-17,
after both). Resolving each operating point:

| | budget | local | windows kept | compressed windows | **keys read per step** | TPOT |
|---|---|---|---|---|---|---|
| best ever | 0.50 | 64 | 247 | 173 | **2045** | **0.0626** |
| 2026-09-16 | 0.20 | 128 | 85 | 59 | **813** | 0.0763 |

The **faster** configuration holds **2.5× more keys and 2.9× more compressed
windows**.

If the unpack-and-decompress arithmetic were the dominant cost, the run with
nearly three times as many compressed windows would have been far slower. It was
22% faster. So:

* **Per-window decompression is not what dominates.** Cutting the cache to 40% of
  its size made decode *slower*, not faster.
* **Decode time barely tracks cache size at all.** What dominates is paid once per
  step regardless of how much cache there is.
* **The difference between those two runs is the gate.**

---

## 4. What is measured, and what is not

Being strict about this is the point of the document.

| claim | status |
|---|---|
| The gate's selection saves 0.2–1.6% of TPOT | **measured** — the `--gate-ratio 1.0` control arm, `130de24` |
| Card 400 B vs window 804 B per head; break-even at 0.50 | **arithmetic**, from `sketch_bytes_per_head` and `bytes_per_q_window` |
| The kernel runs 12× off its roofline; scattered reads are why | **measured** — the 2026-09-16 GPU profile |
| int4 cards are accuracy-neutral at ratio 0.25 | **measured on a CPU fixture** — `two_tier_window_reference`, the gated kernel's own statement, against itself ungated |
| int4 cards are faster | **not measured.** No GPU here. Justified by bytes only |
| "The 24% regression is the gate" | **not attributable** — see below |
| The step is kernel-bound: **97.2% GPU busy** at 4096/B=32 | **measured** — the 2026-09-19 profile (§10) |
| The decode kernel is 13.7× off roofline, the gate 11.0× | **measured** (time) against **arithmetic** (bytes) — §10.1 |
| The compiled eviction was running **eager** on every recorded run | **measured** — `Inductor kernels 0.000 ms/step`, §10.2 |
| The decode kernel gets `target_keys=64 num_stages=2` | **measured** — §10.3. It is also the ladder's worst-occupancy rung |
| Timing the tile ladder makes the decode kernel **1.45×** faster, the gate **1.05×** | **measured** — the full ladder, both kernels, §10.0.3 |
| That win reaching TPOT | **NOT measured to arrive.** 6.1% available at 2048/B=32, 1.4% observed; ~4.7 points unexplained (§10.0.3) |
| The compiled eviction is fully fused with zero recompiles | **measured** — `graph_breaks: 0, unique_graphs: 0` after warmup (§10.0.4) |
| The compiled eviction is worth anything | **measured to be worth nothing** — v9 ≈ v8 at every cell. And it still has no control arm |
| Extrapolating a tuned rung across geometries | **measured to cost 12%.** Reverted (§10.15) |
| This session's net effect on TPOT | **measured: a wash** (4 cells +0.9% … +3.3%, 1 unchanged, 1 at −1.4%) |
| The gate's GQA union starved every query head but the loudest of its group | **measured on CPU fixtures**, the kernels' own oracles: worst head 0.05% of its mass read at 0.25 on the recall fixture; retrieval needle read on ~50% of steps at ws=32 (§12) |
| The per-head share union fixes that | **measured on the same CPU fixtures**: worst head 99.95%; needle read on every seed (§12) |
| It recovers RULER | **not measured.** No GPU run on this commit yet (§12) |
| The share kernel costs ~1 launch + `2·rep·NW` fp32 per program per layer | **arithmetic.** Compiles for sm80 (Triton 3.3.1); never executed |

### The confound in the 24% number

`192c001` recorded the gate as a 24% regression (0.0626 → 0.0777) and reasoned
that "everything between `fbcf368` and HEAD is the gate." It is not. `ecc0792`
sits in that range and changed the measurement script's defaults: budget
0.50 → 0.20, local window 64 → 128. Local window sizes the full-detail tier,
which is 87% of the read — so the dominant term roughly doubled while the
compressed tier shrank (§3's key counts: 2045 → 813).

By §3's arithmetic the budget change should have made things *faster*, so the
gate's real cost is probably larger than 24%, not smaller. Either way **24% is a
floor, not a clean measurement**, and it stays that way until the table is re-run
at the published operating point.

### GPU busy was wrong

The profiler divided kernel time by a wall clock measured *inside* the profiled
block, so its own ~43 µs/launch overhead landed in the denominator. Against the
real 76.3 ms step, GPU busy is **63% — mixed**, not the 32.6% that read
"host-bound". `profile_decode.py` now times an unprofiled window. Anything that
reasoned from "host-bound" needs re-reading.

**And at the headline cell it is 97.2%** (§10.1), which closes the question
rather than merely correcting it: at 4096/B=32 there is no host gap to attack.
The flat-TPOT-across-batch signature that first suggested one has a second and
now confirmed cause — a kernel whose critical path is batch-invariant, which is
what `grid = (B * H_kv,)` over a serial tile loop produces.

### Never run at all

Stage-0-style questions that no GPU session has answered. Two were answered on
2026-09-19 and one of them answered "it never ran":

* ~~the tile rung the decode kernel actually gets~~ — **`target_keys=64
  num_stages=2`** (§10.3), which is the ladder's worst-occupancy rung and was
  chosen by first-fit, never by time.
* ~~the eviction's fused kernel timing on CUDA~~ — **there was no fused kernel**
  (§10.2). Dynamo had fallen back to eager, silently, on every run recorded in
  this document. The fused eviction's cost remains unmeasured, but for a
  different reason than before.
* the two external baselines (§7) — still never run.

---

## 5. What one decode step costs

Measured 2026-09-18 with a dispatch counter (launching ops only), and separately
by bytes at a real head geometry (H=8, D=128, ws=8, N_q=276, one row, one layer).

| | launching ops |
|---|---|
| ordinary decode step | **5** (the KV append; already minimal) |
| eviction step | **186** (all 32 layers at once, every 8 steps) |

So the eviction is ~82% of the cache's amortized launches. By bytes, one eviction
touches 27.7 MB:

| MB | what | where |
|---|---|---|
| 8.3 | the position table rebuilt for the **whole** compressed tier | `decode_kernel.py:185` |
| 8.6 | the full-detail store rebuild | `cache.py`, `state.py` |
| 3.4 | the cards gather | `slots.py` |
| 1.7 | the codes + grid gather | `slots.py` |

**The largest single item is a redundant pass.** A compressed window's positions
are frozen for life, so its position table never changes — yet it is rebuilt for
every active window at every eviction, when only the handful that just arrived
are new. Making that rebuild incremental is worth more than any proposal in §6
and changes nothing about what the cache keeps. It is not done because it is a
kernel-visible change and this box has no GPU.

---

## 5.1 Optimization review, 2026-09-18 — unit, module, design

Done on a CPU box with a `TorchDispatchMode` counter attributing every launching
op to the repo line that issued it. **Read the caveat at the end before quoting
any byte figure from it.**

### Unit: the ordinary decode step is already minimal

Measured on the PRODUCTION path (layer-major store, gate live, both Triton
kernels stubbed so the host work around them runs), L=32, B=1, N_q=99, ws=8:

| per layer, per step | |
|---|---|
| `state.write_rows` — K and V into the joint buffer | 2 copies |
| `fused_gate` | **1 launch** |
| `fused_two_tier_decode` | **1 launch** |
| `torch.gather` — physical → merged-id score permutation | **1 launch** |

Once per step, not per layer: `_begin_step`, `scorer.accumulate`,
`state.reserve` — 3 ops total for all 32 layers.

**That is 3 launches and 2 copies per layer per step and nothing else.** There is
no per-layer Python building tensors, no per-step re-gather of the tier, no
device sync anywhere on the step. Every memo (`_fused_ctx`, `score_meta`, the
gate's cards) is keyed on the store's version and hits between evictions. The
`_scratch` buffers mean `est` and `logmass` are not reallocated per layer.

**Conclusion: there is nothing left to win at unit level on the ordinary step.**
Anyone arriving with a launch-count idea for this path should stop here.

### Unit: the eviction's widths are measured, not bounded

`_evict_widths` takes ONE host sync for the whole eviction (three masks stacked
into one `.item()`) and returns the true max over rows. At steady state, against
a bound of `N_q = 99`:

| | promote `n_p` | reactivate `n_r` | demote `n_d` |
|---|---|---|---|
| steady-state evictions | 8–16 | 0–8 | 4–16 |

So D1 landed and works: the quantizer round-trip runs at roughly the real count,
not at 99. (The first eviction is different by construction — it demotes the
whole Q tier at once, `n_d = N_q`.) **A previous session's hypothesis that this
was a ~99x overcompute is wrong and is retired here.**

### Module: the eviction is the only heavy cache phase, and it is already batched

Marginal cost of an eviction over an ordinary step: **557 host dispatches**,
across all 32 layers at once. It is layer-major (one pass, not 32), compiled
unconditionally, and takes one sync. The per-call-site ranking, by dispatches:
`slots.put` (64), the quantizer's `decode` / `_quantize_grid` / `pack_crumbs_last`
(~60), `effective._rope_cos_sin` (14).

### Design: two fusions are open, and together they are worth about 2%

Both are score-neutral — they move where a computation happens, not what it
computes — and each removes one launch per layer per step:

1. **Fold the gate into the decode kernel as a prologue.** `gate_kernel.py` says
   this itself ("A separate kernel, for now"): one program already owns the right
   `(batch, KV head)`. It removes 32 launches/step *and* the `est`/`logmass`
   round trip through HBM. The reason it has not been done is real — a prologue
   competes for the main kernel's register budget, and the autotuner falls back
   silently when that budget is exceeded — so it needs the tile rung pinned and a
   GPU to confirm the rung did not drop.
2. **Fold the score permutation into the kernel's epilogue.** The kernel already
   writes `wsum`; passing it `ORDER` (memoized per window epoch) lets it store
   straight to the permuted column, removing the host `gather` and one
   `[B, H_q, W]` read-plus-write per layer per step.

**Sizing, honestly.** Together that is 64 of the ~96 cache-side launches per
step. At the ~17 us exposed per launch this document measures elsewhere, it is
~1.1 ms against a ~57 ms step: **about 2%.** Worth doing, not a headline, and
*not* where the remaining factor of two lives.

### What the review did NOT find

No unbatched per-layer work. No redundant per-step gather. No device sync on the
decode path. No dead allocation. The host side of this cache is in good shape,
and the gap to Flash is not sitting in it.

**The dominant term is still inside the decode kernel** — 7.03 ms against a
0.58 ms roofline, from the gather-bound reads §2 describes. That is §6.2, and it
is a memory-layout change, not a launch-count one.

### Caveat on the byte figures

The eviction is compiled, and a `TorchDispatchMode` observes the op chain
**before** Inductor fuses it: the counts above are identical with
`STICKYKV_COMPILE_EVICT` set to 1 and to 0. So the eager chain's byte totals
(the quantizer round-trip at ~235 MB, RoPE at ~134 MB per eviction across 32
layers) are an upper bound on a fused kernel's real traffic, not a measurement of
it. **Do not quote them as GPU numbers.** The dispatch counts are graph-size
facts; the launch counts on the *uncompiled* per-step path above are real.

While checking this, two dead controls were found and removed: the eviction
banner printed `(STICKYKV_COMPILE_EVICT=1)` for a variable deleted in `0974687`
and read nowhere, and `perf_runner` set that same variable on CUDA. Both looked
like a control arm and were not one.

---

## 5.2 Will the gate fire on the next run? — verified 2026-09-18

The question this section answers is "if I run the sweep now, am I timing the
gated method at the config I asked for?" It is asked because the honest answer
has been *no* before, and nothing in a latency table showed it.

### The Q tier is non-empty in every cell, so the gate arms

The gate only arms where `store.num_active_windows > 0`. Resolved through the
real `WindowedCacheConfig.resolve` for the whole grid — budget 0.20, sink 5,
q in {0.70, 0.30}, gate in {0.10, 0.25}, local in {64, 128}, ws in {8, 16, 32},
across all three shapes — **72 of 72 cells resolve with a live Q tier**, and the
realised read fraction tracks the request:

| q | ws | local | N_q @4096 | n_sel @0.25 | realised |
|---|---|---|---|---|---|
| 0.70 | 8 | 128 | 247 | 62 | 25.1% |
| 0.70 | 32 | 128 | 103 | 26 | 25.2% |
| 0.30 | 8 | 128 | 107 | 27 | 25.2% |
| 0.30 | 32 | 128 | 42 | 11 | 26.2% |

At gate 0.10 the realised fraction runs 10.0–11.9%; the drift is `max(1, ceil())`
on a small `N_q` and is arithmetic, not slippage. The thinnest cell is q=0.30,
ws=32, local=128 at **N_q=42** — still a real tier, but it is the cell where the
gate has least to select from, so read its row with that in mind.

### The knob reaches the cache

`--gate-ratio` → `GATE_RATIO` → `quant_gate_ratio` in the generated YAML →
`perf_runner` → `quant_gate_ratio_kwargs(WCC, ...)` → `WindowedCacheConfig`.
`run_perf_table.sh` re-reads its own generated config and asserts the value
survived, specifically so the `6b8a188` bug (the knob inert in every runner)
cannot recur silently.

### The run now PROVES it, which it did not before

`perf_runner` recorded the eviction path's counters and **not**
`flash_decode.stats()`. So a run that never gated — reading the whole tier,
correct output, plausible TPOT — was indistinguishable from one that did. That
is not a hypothetical: it is what this branch shipped for eight commits.

Fixed: the gate's counters are reset after warmup, recorded into the npz under
`diagnostics.<config>.gate`, and printed under the table as one of three lines.

```
read gate: ran on all 192 fused layers, realised read fraction 0.252
!! read gate ran 0 times against 192 fused layers -- ... not the shipped method.
!! the fused decode kernel never fired: ... the materialize path, NOT the gated one.
```

**If the first line is not there, the number above it is not a gated number.**

### Every runner, not just the perf suite

The gate is the decode path wherever the flash backend meets a Q tier, so the
same question — *did it actually gate?* — has to be answerable from a LongBench,
GSM8K, RULER or parity run, not only from a latency table. It was not: none of
the quality runners touched `flash_decode.stats()`.

An ungated quality run is the harder version of the same failure. It reads the
whole int2 tier, generates correct text, and scores normally; quality moves for a
hundred reasons and nobody re-derives them, so a score quoted for "the gated
method" when the gate never ran survives review in a way a latency number might
not.

Now: one definition, `flash_decode.expect_gated(backend, quant_ratio, cuda)`,
restating the three conditions `hooks.install_score_hooks` uses to pick the path
— flash backend, `quant_ratio > 0`, CUDA — so a perf row and a LongBench sidecar
cannot disagree about what the run was. `gate_report` turns the counters into one
of four stable verdicts (`gated`, `not-expected`, `NOT-GATED-kernel-never-fired`,
`NOT-GATED-partial`) plus `ok`, and every runner records it:

| runner | where it lands |
|---|---|
| LongBench | `read_gate` in the per-dataset metadata sidecar |
| GSM8K | `read_gate` in the metadata sidecar |
| RULER | `read_gate` in the per-task metadata sidecar |
| perf | `diagnostics.<config>.gate` in the npz, printed under the table |

`not-expected` is a pass, not a failure: the eager backend and `quant_ratio = 0`
have no Q tier to select over, and **eager is the supported way to ask for an
ungated decode**. What the verdict removes is the case where nobody could tell
which had happened.

### RULER was scoring a different method

`configs/ruler_niah_mk3_omega16.yaml` sat at **`quant_ratio: 0.0`** while
LongBench and GSM8K ran at 0.70 — a pure fp16 windowed cache, no int2 tier, and
therefore no read gate, scored under the same branch's name. It had an
`OPERATING_POINT_EXEMPT` entry reading "RULER uses int4, a different tier", which
was not true of anything in the repo and was covering the divergence rather than
recording it.

It is now at the operating point (`quant_ratio: 0.70`, `quant_gate_ratio: 0.25`)
and exempt only on `window_size=16`, which is the arm the file exists to be.
`scripts/check_operating_point.py` passes with that as the stated reason.

**Any RULER number taken before this change is not comparable with one taken
after it** — it was a different cache.

### It is the fastest path this repo has — not the fastest possible

What the next run measures is: int4 cards (272 B/head, the §6.1 frontrunner,
landed), the gate as the only Q-tier read path with no ungated arm, a host path
measured at 3 launches and 2 copies per layer per step (§5.1), and a
layer-major compiled eviction.

What it does **not** include, and what therefore still stands between this and
the goal:

* **§6.2, the scattered reads** — the kernel at 7.03 ms against a 0.58 ms
  roofline. Untouched. This is the factor that matters.
* the two open fusions (gate as a kernel prologue, score permutation into the
  epilogue) — ~2% together, §5.1.

### Three preconditions, all outside the config

The fused path needs **flash-attn** and **Triton on CUDA**; without either, the
cell errors rather than quietly running something else. Prompts must be
equal-length — a padding `attention_mask` sends transformers down
`flash_attn_varlen_func`, which this patch does not own, and the run raises
`FusedDecodeNotReached`. The sweep's `--data-source wikitext-103` chunks to
exactly `prefill_len`, so this holds.

### This table is a new baseline

int4 moved `bytes_per_q_window` 9,632 → 8,608, so the cache holds more at the
same budget. Like `ac128e8` and `8a8cdc0` before it, **rows either side of this
change are not comparable cell-for-cell.**

---

## 6. What to do next

### 6.1 Shrink the card to int4 — **FRONTRUNNER**

The card is the one lever that pushes both directions at once: **more context per
byte AND less read per step.** Nothing else on this list does both.

Store the mean and the value centroid at 4 bits instead of 8. They are 64% of the
card and, measured, they do not need the precision.

| | now (all int8) | **mu + vbar at int4** |
|---|---|---|
| card, per head | 400 B | **272 B** |
| a compressed window | 9,632 B | **8,608 B** |
| break-even read ratio | 0.50 | **0.66** |
| cost at ratio 0.25 | 75% of ungated | **59% of ungated** |
| retained context at the same budget | — | **+12%** |

**Accuracy: neutral.** Measured on `two_tier_window_reference` against exact
attention over the same cache — median relative error at ratio 0.25 is 0.0001
with int8 and 0.0001 with int4, unchanged to four decimals.

Two caveats on that number, both in our favour. The ablation simulates int4 by
coarsening the existing int8 codes against the same scale; a real int4 encoder
re-fits the scale to the narrower range, so this is conservative. And no seed
count was recorded for it — the 12-seed figure in the grid work (§9) is a
different measurement and does not cover this one.

**The one field that cannot shrink** is `t`, the per-token projection. It is 10 B
of 400 and dropping it costs four orders of magnitude of accuracy — without it
the card is a plain average again, which is the exact failure the card was
designed to avoid (an average always understates a window holding one hot token
and seven cold ones).

**Pair it with re-pinning the read ratio.** `quant_gate_ratio` is a *fraction*, so
every window the budget buys is automatically spent on more work in the kernel's
inner loop — 46 → 55 → 62 selected windows across the changes we have landed, on
a kernel that is loop-bound, not byte-bound. Holding the selected *count* fixed
instead converts a memory win into context at flat decode work:

| shape | variant | keys | selected | compressed-tier read/step |
|---|---|---|---|---|
| 4096/256 | today | 2117 | 55 | 1036 KB |
| | + int4 cards at ratio 0.25 | 2325 (+28%) | 62 (+35%) | 914 KB (−4%) |
| | + int4 cards, **ratio 0.186** | 2325 (+28%) | 55 (0%) | **814 KB (−14%)** |

That second row is a config change, not a code change. Nobody has turned this
dial.

### 6.2 Make the selected reads contiguous

The gate's other cost is that the windows it picks are scattered, so the hardware
cannot fetch them in runs (§2). Compacting the selection so the chosen windows sit
next to each other would restore that. This attacks the 7.03 ms, where §6.1
attacks the 3%. Larger prize, larger change, and it needs a GPU to verify.

Partially addressed already: `130de24` sorts the picks back into address order,
which is free and monotone. Unmeasured on GPU.

### 6.3 Wire the two missing baselines

See §7. This should arguably come before any further latency work, because it may
change which number we are trying to move.

### Parked, with reasons

| idea | why not |
|---|---|
| **Freeze the window ranking between evictions** | ~2% of a step, and it is the only proposal that **changes which windows the cache keeps** — so it cannot land without a full LongBench run. Worst ratio of the three. |
| **Use cards only, open nothing** | Not a speed/accuracy trade — it is a 0.908 relative error. The cards are a *selector*, not a substitute: the correction that makes them work is measured on the windows that *were* opened, so with nothing opened there is nothing to calibrate against. |
| **Drop the gate entirely, keep cards for scoring** | Live option if §6.1 and §6.2 both disappoint. |
| **CUDA graphs** | Deleted. It captured every steady step and replayed none — a slot dies at the next eviction — so it paid capture cost, returned nothing, and died in the allocator. That is where six ERROR rows came from. The host gap is real and must be closed by issuing fewer launches, not by recording them. |

---

## 7. Two of the three baselines have never been run

| baseline | status |
|---|---|
| Flash FullKV | runs — `cache_backend: dynamic` + `flash_attention_2` |
| int2 KIVI | no implementation, no config, no numbers |
| QEvict | an observation suite only; no runnable cache backend |

`perf_runner` has the seam for both (`cache_backend: external` plus a
`method_factory`). Nothing implements one.

This matters more than it looks. The two unmeasured baselines are the ones this
method should be **strongest** against, because both are memory arguments and
memory is where our work actually landed. KIVI at int2 with a full-precision
per-channel grid carries the same grid overhead we just cut by 17%. Against
Flash we are arguing about latency, which is the argument we are weakest at and
the one every session so far has spent itself on.

**Wiring them is small work — one existing seam, no method changes.**

---

## 8. B=1 is not winnable, and that is half the goal

At batch 1 a decode step reads every model weight once: 16.06 GB in fp16, a
**~10.3 ms/token floor on an A100**, independent of cache size. Flash sits at
~24 ms. We sit at ~57 ms.

The gap is not KV traffic — at B=1 there is barely any. It is the per-step
machinery: eviction bookkeeping, the card scan, the selection, the score
write-back. All of it per layer per step, none of it proportional to cache size.
`TARGET_GAP §2` measured the signature: **~300–430 µs per layer per step, moving
only 1.4× across a 32× batch range.**

This is the method working as designed. `eval_perf_batched.yaml` has said so in
writing the whole time:

> At B=1 KV compression CANNOT show a speedup, and that is not a bug.

Three options, only one of which is both true and keeps B=1 in the story:

1. **Drop B=1 from the goal.** State the claim at B≥32. Honest; costs only the
   headline's breadth.
2. **Get the per-step machinery near zero at B=1.** The whole list is worth
   −7 to −13 ms against a 33 ms deficit. It does not close.
3. **Claim memory at B=1, not latency** — quote max-batch or tokens-held rather
   than TPOT, and say so. ✅ **This is the one to take.**

**The realistic goal:** beat Flash at every B=32 shape, beat KIVI and QEvict on
memory everywhere, and say plainly that B=1 latency is not what this method is
for.

---

## 9. What recent sessions changed

| change | worth | confidence |
|---|---|---|
| Gate card enters the byte budget | corrects a **38% overrun** on the compressed tier; `N_q` 715 → 518 at budget 0.50/q=0.70 | measured |
| Budget remainder spent and enforced | 99.8%+ utilisation under `bytes` | measured |
| One-byte scale/zero grid (`8a8cdc0`) | window 11,648 → 9,632 B; **+21% context** for +0.2% error | measured |
| Grid grouped per 32 entries | the worst channel on 1000×-gain keys stays at full-precision's error instead of decoding to noise, for 12 B/head | measured |
| Skipped windows attend via their centroid | accuracy, not speed — ~2.3× lower median output error than dropping them | measured, CPU |
| Compaction by counting, not sorting (`c65ccff`) | `aten.sort` 21 → 6 calls in the compiled eviction | op count |
| `lookup` walks its match 3× not 6× (`513b3d1`) | ~500 MB traffic, ~330 MB peak transient per eviction | measured |
| CUDA graph deleted | removes the OOM and six ERROR rows; loses nothing | measured |
| 21 env knobs, eager fork, FlashInfer, PyTorch oracle deleted | no speed change; every number now says which method it came from | structural |
| Eviction no longer specialises the graph on `layer_idx` / `step`; eager fallback now raises (§10.2) | the compiled eviction was **running eager on every recorded run** — its real value is still unmeasured | measured (that it was eager); unmeasured (the fix) |
| Decode tile ladder timed instead of first-fit, `num_warps` searched (§10.3) | **1.45× on the decode kernel** — the first-fit rung is 9th of 11 (§10.0.3) | measured |
| Read gate's `BLOCK_W` / `num_warps` searched instead of derived (§10.4) | **1.05× on the gate** — the derived heuristic is 4th of 8 | measured |
| Eviction body made pure w.r.t. the fp buffers (§10.13) | the eviction **compiles for the first time on this build** — fully fused, zero recompiles | measured |
| `_LAST_WINNER`, extrapolating a tuned rung across geometries (§10.15) | **−12% TPOT. Reverted.** The rung's trade moves with the geometry | measured |
| Net effect of the whole session on TPOT | **a wash** vs an identically-configured run (§10.0.1) | measured |

**None of these moved a latency cell.** The grid work is a memory change; the
op-count work removes ~15 extern calls from a step that has ~1,650 launches. An
earlier projection that assumed otherwise was double-counting a recovery that had
already happened in `6e83a2c`, and is **withdrawn**:

| | now vs Flash | was projected | actually expected |
|---|---|---|---|
| 4096/256 B=32 | 1.13× faster | ~1.3–1.5× faster | **unchanged** |
| 2048/512 B=32 | 1.17× slower | ~parity | **unchanged** |
| 1024/1024 B=32 | 1.36× slower | ~1.1× slower | **unchanged** |
| all B=1 | ~2.4× slower | ~2.1× slower | **unchanged** |

**Any quality comparison across `ac128e8` or `8a8cdc0` is invalid.** Both moved
the stored format and the price of a window, so a row before and a row after are
not the same cache.

### The perf table could not run at all (fixed, `014c40b` + `52fc367`)

The first GPU table after this work ERRORed on all six cells: the compiled
eviction could not be lowered. Values the code needs as real integers were left
symbolic inside the compiled region, the "static" retry silently reused the
failed state so it was not a retry, and the error text sent you after three
knobs that had been deleted.

**It took two commits, and the first one is a lesson worth keeping.** `014c40b`
named the offending value correctly — `W`, the scored-window count in
`policy.compute_two_tier_retain` — and then put the `int()` on
`compute_retain_window_indices`: a different function forty lines up, and one
the compiled body never calls. Every cell still ERRORed with the identical
message. `52fc367` fixed the real one.

The guard test added alongside it made that worse rather than catching it. It
matched `x = t.shape[i]` and skipped whole-shape tuple unpacks, on the stated
reasoning that an unpack is "a different pattern handled by the callers". It is
not a different pattern — it is the same symbolic read wearing a different
shape in the syntax tree — and the live offender was an unpack. So the guard
came back green, the commit message asserted the root cause was fixed, and the
GPU still died. The guard now matches both forms; run against `014c40b`'s tree
it reports ten offenders, `W` among them.

Two things generalise. A check that cannot see the form the bug takes is worse
than no check, because it converts "unverified" into "verified". And naming a
root cause correctly in prose is not the same as editing it — the guard existed
precisely to close that gap and had a hole in exactly the place it was needed.

---

## Appendix — two utilisation numbers that look contradictory

Both statements are in this repo's history and both are true.

**"Under `bytes`, utilisation is 0.99 at every `q`; under `tokens` it is
0.99 / 0.69 / 0.57."** This is about what the two modes *mean*. Under `bytes`, `q`
splits the byte allowance and each tier buys windows at its own price, so the
cache costs what it was granted at every `q`. Under `tokens` the window *count* is
pinned, so cheaper keys buy fewer bytes rather than more windows and the spend
falls with `q`. Still true today.

**"The card was not in the window's price, so `bytes` mode was overspending."**
This is about the *meter*, one level down. Utilisation is
`retained_bytes / budget`, and `retained_bytes` priced a window at codes and grid
only while the window also carried a card:

| | it reported | it actually held |
|---|---|---|
| `bytes`, q=0.5 | 0.997 | **1.180** |
| `bytes`, q=0.7 | 1.000 | **1.257** |
| `tokens`, q=0.5 | 0.638 | 0.686 |
| `tokens`, q=0.7 | 0.497 | 0.563 |

Same defect, opposite symptom. Under `bytes` the resolver *spends to the meter*,
so an undercounted price made it hold 18–26% more than the budget granted — a
real overspend, and why `N_q` fell 715 → 518 when the card was priced in. Under
`tokens` the count is pinned, so the missing bytes could not buy anything and
showed up only as under-reporting.

So "bytes spends 99%" was never the claim that got corrected. What was corrected
is what 99% was 99% *of*. **Utilisation is a check on the resolver, not on the
format.** Only the window's price can be wrong about the format, which is why it
now lives in the module that defines the layout.

---

## 10. The 2026-09-19 decode session — findings, values, meaning

**Read §10.0 for the conclusions. §10.1–10.15 are the record of how they were
reached, seven GPU runs including the wrong turns, kept because three of the
conclusions are corrections of earlier ones in this same document.**

---

## 10.0 What seven runs established

### 10.0.1 The table, end to end

Llama-3.1-8B, A100-80GB, `bytes` budget mode, `cache_budget=0.20`, `q=0.70`,
`ws=8`, gate at `quant_gate_ratio=0.25`. TPOT_steady, seconds:

| cell | **v8** eager evict, untuned tile | **v9** compiled evict, tuned tile | **v10** + `_LAST_WINNER` *(reverted)* |
|---|---|---|---|
| 4096/257 B=1 | 0.0545 | 0.0553 | 0.0614 |
| 4096/257 B=32 | **0.0570** | 0.0589 | 0.0655 |
| 2048/513 B=1 | 0.0550 | 0.0557 | 0.0620 |
| 2048/513 B=32 | 0.0569 | **0.0561** | 0.0634 |
| 1048/1049 B=1 | 0.0546 | 0.0551 | 0.0616 |
| 1048/1049 B=32 | 0.0562 | 0.0562 | 0.0635 |

**v9 is the current head and is a wash against v8** — four cells worse by
+0.9% … +3.3%, one unchanged, one better by −1.4%. **v10 was +11–13% everywhere
and is reverted** (§10.15).

TTFT moved within ±6% with mixed signs, and `peak_GB` is identical to the
0.01 GB at every cell. `steadyKV_GB` drifted at the three B=32 cells (−1.50,
−0.25, −0.49 GB; B=1 unchanged), which is the allocator, not a footprint change.
**Nothing here is a prefill or a memory story.**

### 10.0.2 Where a decode step goes — the profile that had never been run

`profile_decode.py`, 4096/B=32, 45.69 ms/step wall, 44.41 ms of CUDA kernels:

| bucket | ms/step | share | launches |
|---|---|---|---|
| **our two-tier decode kernel** | **17.74** | **40.0%** | 32 |
| model GEMM | 11.18 | 25.2% | 290 |
| **our read gate** | **3.63** | 8.2% | 32 |
| elementwise | 4.12 | 9.3% | 822 |
| index / gather | 3.66 | 8.2% | 79 |
| copy / cat | 2.08 | 4.7% | 339 |
| reduction | 1.04 | 2.4% | 70 |
| sort / topk | 0.84 | 1.9% | 65 |

**GPU busy 97.2%. 1,762 launches/step, all covered by the work they issue.**

*Meaning.* Three things follow, and they govern everything else here:

1. **The step is kernel-bound, not host-bound.** The flat-TPOT-across-batch
   signature that suggested a host gap has a second cause, which the profiler's
   own docstring predicted: a kernel whose critical path is batch-invariant,
   which is what `grid = (B * H_kv,)` over a serial tile loop produces. *Nothing
   is won by issuing fewer launches.*
2. **Our cache costs ~3× what the model does.** The model side is finished —
   11.18 ms against a 10.3 ms weights-read floor is ~92% of theoretical. The
   two-tier decode kernel alone costs more than every GEMM in the model.
3. **Both our kernels are ~12× off their own rooflines, the same way.** Priced
   from bytes at this geometry: decode 2.02 GB/step → 1.30 ms roofline vs 17.74
   measured (**13.7×**); gate 0.51 GB → 0.33 ms vs 3.63 (**11.0×**). That
   independently reproduces §2's 12× from a different direction, and says the
   problem is not one bad kernel but one shared shape: gather-bound,
   occupancy-starved reads. **That is §6.2, and it is still the main event.**

### 10.0.3 The tile rungs — measured, and the first-fit rung is bad

Both kernels now time every rung that fits, once per geometry, and publish the
whole ladder. At 2048/B=32, ms per layer per step:

```
decode  (32,2,4)=0.242  (32,1,4)=0.281  (64,2,8)=0.301  (64,1,4)=0.303
        (16,2,4)=0.303  (64,1,8)=0.329  (32,2,8)=0.332  (32,1,8)=0.335
        (64,2,4)=0.351  <- what first-fit always took
        (16,1,8)=0.406  (16,2,8)=0.417
gate    (16,4)=0.076  (16,8)=0.079  (32,8)=0.079  (32,4)=0.080  <- the heuristic
        (64,8)=0.080  (64,4)=0.090  (128,8)=0.092  (128,4)=0.102
```

| | old choice | measured winner | gain |
|---|---|---|---|
| decode kernel | `(64,2,4)` 0.351 ms — **9th of 11** | `(32,2,4)` 0.242 ms | **1.45×** |
| read gate | `(32,4)` 0.080 ms — 4th of 8 | `(16,4)` 0.076 ms | 1.05× |

*Meaning.* §10.3 predicted exactly this: the top rung stages ~128 KB of `tl.dot`
operands against an A100's 163 KB/SM, so **one block is resident per SM** and
the Q-tier loop runs with nothing to hide its latency. The ladder's own docstring
called its ordering an assertion — *"it never benchmarks, so that ordering… run
the table twice, compare"*. It was an assertion, and it was wrong by 45%.

**But the win does not reach TPOT.** At 2048/B=32, 0.351 → 0.242 ms/layer is
3.49 ms of a 56.9 ms step: **6.1% available, 1.4% observed.** Two candidate
explanations, unresolved: only some signatures are tuned (a signature must recur
`_TUNE_AFTER`=3 times), and a kernel timed back-to-back in isolation may not
shorten a step proportionally. **Extrapolating one geometry's winner to others
was tried to close this and cost 12%** (§10.15) — the trade a rung selects moves
with the geometry, which is the whole reason the search is keyed per signature.

### 10.0.4 The compiled eviction — three findings, all against prior belief

```
eviction path: {'eager': 0, 'compiled': 285}
dynamo: {'graph_breaks': 0, 'frames_total': 0, 'frames_ok': 0, 'unique_graphs': 0}
```

`perf_runner` zeroes those counters **after warmup**, so they cover the measured
window only.

1. **It had never compiled on this build** (§10.9, §10.13), by two independent
   mechanisms — `capture_scalar_outputs` defaulting True since torch 2.6 made
   `_evict_widths`' `.tolist()` an unresolvable guard that abandoned the frame;
   and with that fixed, an aliased-view mutation hit an AOTAutograd bug.
   `_EVICT_STATS` reported `compiled` throughout, because it counts *calls to
   the compiled callable*, not compiles. **Every decode number in this document
   before v9 was measured with an eager eviction.**
2. **Now that it compiles, it is fully fused with zero recompiles** — not one
   graph break, no compilation inside the timings.
3. **And it is worth nothing measurable.** v9 ≈ v8 at every cell.

*Meaning.* Consistent with §10.0.2: the compiled eviction exists to collapse
~360 launches, and launches are not the bottleneck on a 97.2%-busy step. **This
is now a design question, not a bug** — "compiled unconditionally, one
production path, no fallbacks" is a deliberate stance that, on this build,
produced a five-run total outage in service of an optimisation the profile says
is not needed. It still has **no control arm**, which is the one thing this
repo's own standing rules require of every perf-affecting change, and precisely
why it went this long with nobody knowing it neither ran nor paid.

### 10.0.5 The eviction's widths, measured

```
fresh (quantize+sketch)  max=230  mean=188.95  of n_q=230   fill=100.0%
promote (dequant+RoPE)   max=  2  mean=  0.21             fill=0.9%
```

*Meaning.* The demote cap is already right — a hypothesis that it was a ~99×
overcompute is retired. The promote side was sized by its bound and masked away
99%; that one was live and is fixed.

### 10.0.6 The gate fires, everywhere, at the configured ratio

`gated=49152 fired=49152`, `read_fraction 0.251–0.258` against
`quant_gate_ratio=0.25`, on all 24,576 fused layers in every run. *Meaning:* the
failure that once left the gate unreachable for eight commits is not recurring,
and every number in §10.0.1 is a gated number.

### 10.0.7 Three of my own diagnoses, refuted by measurement

Recorded because each was stated confidently in this document before being
disproved, and the corrections are more useful than the claims:

| claim | what killed it |
|---|---|
| "Decode is host-bound; collapse launches" | GPU busy **97.2%** (§10.1) |
| "The eviction recompiles per eviction; that is the TPOT cost" | `graph_breaks: 0, unique_graphs: 0` in the measured window (§10.14) |
| "The counting backend caused the IndexError" | the next run raised it under **stock** `backend='inductor'` (§10.11) |

### 10.0.8 What is open

1. **§6.2, making the selected reads contiguous.** Both kernels are ~12× off
   roofline from scattered fetches. This is the only item sized to matter.
2. **The 4.7 points of tile win that do not reach TPOT** (§10.0.3).
3. **A control arm for the compiled eviction** — ~30 lines, one run, settles
   whether to keep it at all.
4. **The gate's +7.6% across all eight rungs** between v9 and v10 — a
   fixed-config kernel timing, so not a code change. Unexplained; ~0.6 points.

---

## 10.1 The profile in full

`scripts/profile_decode.py --prefill 4096 --batch 32 --steps 24 --warmup 12` on
the shipped config (`n_q=230`, `W_retained=273`, gate live at `read_fraction
0.252`). This is the first profile taken at the headline cell on the gated path,
and it answers two of the three questions §4 listed as **never run at all**.

```
  wall               45.69 ms/step   (profiler OFF -- the real step)
  CUDA kernels       44.41 ms/step
  GPU busy            97.2 %
  kernel launches     1762 /step
  path        armed=1920 fired=1920 gated=1920  read_fraction=0.252
```

| bucket | ms/step | share | launches |
|---|---|---|---|
| **ours: two-tier decode** | **17.74** | **40.0%** | 32 |
| model: GEMM | 11.18 | 25.2% | 290 |
| **ours: read gate** | **3.63** | 8.2% | 32 |
| elementwise | 4.12 | 9.3% | 822 |
| memory: index/gather | 3.66 | 8.2% | 79 |
| memory: copy/cat | 2.08 | 4.7% | 339 |
| reduction | 1.04 | 2.4% | 70 |
| sort/topk | 0.84 | 1.9% | 65 |

**Three readings, in order of how much they change what to do next.**

**(a) The step is kernel-bound, not host-bound, and now firmly.** 97.2% GPU busy
retires the last of the host-gap framing for good — §4's correction of the
32.6% figure to 63% was already moving this way; at the headline cell there is
essentially no gap left to close. *Nothing is to be won by issuing fewer
launches.* 1,762 launches per step are fully covered by the work they issue.

**(b) Our cache costs three times what the model does.** The model side is
finished: 11.18 ms against the 10.3 ms weights-read floor is ~92% of
theoretical, and no KV method touches it. Ours is 21.37 ms of named kernels plus
most of the 11.75 ms of shared copy/index/sort/elementwise. The two-tier decode
kernel alone costs **more than every GEMM in the model put together**.

**(c) Both of our kernels are ~12× off their own roofline, in the same way.**
Priced from bytes at this exact geometry:

| | traffic/step | roofline @1555 GB/s | measured | off by |
|---|---|---|---|---|
| two-tier decode | 2.02 GB | 1.30 ms | 17.74 ms | **13.7×** |
| read gate | 0.51 GB | 0.33 ms | 3.63 ms | **11.0×** |

That independently reproduces §2's 12×-off measurement from a different
direction, and it says the problem is not one bad kernel — it is the same
gather-bound, occupancy-starved shape in both. §6.2 (contiguous reads) is the
structural answer and still needs a GPU. What follows is the part that did not.

### 10.2 The compiled eviction was not compiling — and could not be seen

```
torch._dynamo hit config.cache_size_limit (8)
   function: '_evict_two_tier_impl' (cache.py:2187)
   last reason: 0/0: L['layer_idx'] == 0   # self._states[layer_idx]
...
  eviction: 3 compiled / 0 eager runs, Inductor kernels 0.000 ms/step
```

`_evict_two_tier_impl` took `layer_idx` and opened with
`self._evict_targets(layer_idx)`, which indexes `self._states` — **a Python
list**. Dynamo specialises on the integer's value, so it built one graph per
layer. The first eviction runs per layer, 32 times, which blew the default
`cache_size_limit` of 8 at layer 8; Dynamo's response to that limit is to run
the frame **eagerly**. Every eviction after it — including all 39 joint ones,
which are the ones that matter — ran eager under a "compiled" label.

**`_EVICT_STATS` cannot see this and never could.** It counts calls to the
compiled *callable*, and the callable was called; it was Dynamo underneath that
declined. So the counter said `3 compiled / 0 eager` while Inductor emitted zero
kernels. A `TorchDispatchMode` could not see it either — §5.1's caveat already
notes that a dispatch counter observes the op chain *before* Inductor fuses it,
which is exactly why its counts were "identical with the compile flag on and
off". That identity was not evidence of anything; it was the blind spot.

This invalidates one claim outright and softens another:

* §5.1's *"it is layer-major, compiled unconditionally, and takes one sync"* —
  the first and third held, the second did not, on the runs that produced these
  numbers.
* the eviction's contribution to the step has never been measured **fused**.
  Whatever the eager chain costs today is an upper bound on what the fused one
  will.

**Fix, three parts, all at the source:**

1. **`layer_idx` and `step` no longer cross into the graph.** The body takes the
   resolved `(state, store, policy)` plus one `joint` bool. Resolving the
   targets, dropping the stale `_fused_ctx` entries and recording telemetry all
   moved to `_evict_two_tier`, outside the compiled region. `step` was only ever
   the telemetry argument and is gone from the body entirely — an integer that
   increments every eviction is an integer Dynamo specialises on, and it had no
   business in a graph. Telemetry's input (the pre-eviction score axis, which
   step 5 of the body overwrites) is returned rather than recorded in place.
2. **`_EVICT_CACHE_SIZE_LIMIT = 64`**, applied to whichever of Dynamo's
   recompile-limit config names the installed version has. The remaining
   specialisations are the shape ones the body asks for *on purpose* — it makes
   its control ints concrete with `int()` so Inductor gets integers instead of
   symbols. A limit is not a budget to spend; it is where Dynamo stops
   compiling, so it is set well clear of the handful expected.
3. **An eager eviction is now a hard error.** The body opens with
   `if not torch.compiler.is_compiling(): raise`. Under Dynamo that is a
   trace-time constant, so the branch folds away and nothing reaches the graph;
   on a cache *hit* the Python body is never entered at all, so the steady-state
   cost is exactly zero. It executes only when Dynamo has fallen back — which is
   precisely the case that used to be silent. The message names the three things
   that cause it (recompile limit, compilation disabled process-wide, an
   untraceable construct) and says plainly that no knob makes an eager eviction
   acceptable.

Point 3 is the durable half. Points 1 and 2 fix the cause we found; point 3
means the *class* of failure cannot recur unseen.

### 10.3 The decode tile was chosen by first-fit, never by time

`_FIT_LADDER` existed to pick a tile that fits in shared memory, and its own
docstring conceded the rest: *"The ladder only falls on `OutOfResources`, it
never benchmarks, so that ordering [is an assertion] — run the table twice,
compare."* Nobody ran it twice. The profile shows it took the top rung,
`target_keys=64 num_stages=2`.

That is the rung with the **worst occupancy in the ladder**. At `BLOCK_T=64` the
`tl.dot` operands stage ~128 KB against an A100's 163 KB/SM, so exactly **one
block is resident per SM** and the Q-tier loop's ~14 dependent tiles run with
nothing co-resident to hide their latency. A kernel 13.7× off roofline at 97%
busy is what that looks like. Whether a smaller tile's extra iterations cost
less than its extra occupancy buys is **not derivable** from the staging
arithmetic — which is all the ladder ever had.

So the ladder is now timed. `_search_rungs` times every rung that fits, caches
the winner per geometry, and re-launches the winner so the caller's values come
from the rung the steady state will use. `num_warps` joins the search: it was
never passed at all, so every launch to date ran at Triton's default of 4 — a
value inherited, never chosen. `score_kernel` already autotunes `num_warps` in
prefill, so this is the repo's own precedent, not a new idea.

The signature gained `B`, `Sfp` and `n_sel`, which first-fit did not need: a
*fit* is shape-independent (staging is a function of the tile), but a *time* is
not — the grid is `B * H_kv` and the serial chain is `Sfp` and `n_sel` long.
`Sfp` and `n_sel` are bucketed by bit length so the fp store's one-token-per-step
growth between evictions does not re-trigger the search.

### 10.4 The read gate was never tuned at all

3.63 ms/step, 11× off roofline, and its two launch parameters were a heuristic
and a default: `gate_window_tiles` derived `BLOCK_W` from an occupancy rule of
thumb, and `num_warps=4` was hardcoded. It now goes through the same
`_search_rungs`, with the derived value kept as the ladder's **first** entry —
the heuristic encodes a real floor ("fill the machine", which at B=1's 8 rows is
the whole argument for splitting the window axis), it just cannot price it.

### 10.5 What this costs, and the one property it moves

**Runtime: nothing.** At steady state a search is one dict hit and one launch —
the same work first-fit did. The search itself is ~19 launches of a
sub-millisecond kernel, once per signature.

**Warmup: one Triton compile per rung — at the shipped geometry, all 12 of
them.** That is the real price, and it is the one number in this section that
was checked rather than asserted. An earlier draft of this paragraph claimed the
rung dedup reduced it: it does not, here. `window_tiling` does collapse
`target_keys` 64/32/16 onto one `(BLOCK_NW, BLOCK_T)` at a large `ws`, but
`BLOCK_W` is derived from `target_keys` as well, so those rungs still differ in a
constexpr and still compile separately. The dedup only bites when the tiling
*and* `BLOCK_W` both collapse — a large `ws` with `W_phys <= 16`. It is kept
because that regime is real, not because it helps the common case.

So: ~12 decode compiles and 8 gate compiles on a cold Triton cache, once per
machine, in the warmup the harness already reserves for JIT. If that is ever too
slow somewhere, `_FIT_LADDER` is one list literal — but trim it by measurement,
not by guessing which rungs matter, which is the mistake that made the ladder
first-fit in the first place.

**The one property that moves: `num_warps` changes the lane layout of the
reductions, so `wsum`, `est` and `logmass` move in their last bits.** Stated
plainly because it is the only thing here that is not byte-identical:

* `BLOCK_W` in the gate is **provably** bit-identical — each program owns a
  disjoint span of output columns and every reduction lives inside one window.
* `BLOCK_T`/`BLOCK_NW` in the decode kernel reassociate the online softmax. That
  is not new: the ladder has always been free to step rungs, and `_sorted_pick`
  already documents the same effect for selection order.
* `num_warps` is a new axis of that same kind. Where it lands on a discrete
  decision — `est` feeds a top-k, `wsum` feeds the eviction ranking — a last-bit
  move can only change the outcome for two windows already tied to ~1e-7.

**No quality claim crosses this change** until a LongBench run says so, and a
`--gate-ratio 1.0` control still controls what it always did.

### 10.6 Considered and not done, with reasons

| idea | why not |
|---|---|
| **Fold the gate in as a decode-kernel prologue** (§5.1 fusion 1) | ~1.3% by §5.1's own sizing, and the top-k cannot fold in at all — the gate splits the window axis across programs, so no program sees the whole card set. It also risks dropping the decode kernel's rung, which is now a *measured* quantity and therefore an expensive one to disturb. |
| **Fold the score permutation into the kernel epilogue** (§5.1 fusion 2) | 0.23 ms/step measured (`_scatter_gather_elementwise_kernel`, n=32/step) — 0.5%. It needs the *inverse* of `order` (a `gather` maps destination→source; a kernel store needs source→destination), and an `order` that is not a strict permutation would scatter scores onto the wrong windows silently. Not worth that failure mode for 0.5%. |
| **Replace the gate's `topk` + `sort` with a Triton selection kernel** | 1.22 ms/step (`gatherTopK` 0.43 + two `radixSortKVInPlace` 0.79) = 2.7%. An exact in-kernel top-k needs its own tie-break rule, and `torch.topk`'s is unspecified — so this trades a measured 2.7% for a change in *which windows are read* in the tie case. Revisit after §6.2, which may remove the reason to care. |
| **Shrink the demote width** | Retired by measurement: `fresh mean=188.95 of n_q=230, fill=100%`. The cap is already close to the real count. (The promote side was live and is already fixed.) |
| **CUDA graphs** | Still deleted, and 97.2% GPU busy is the strongest form of the argument yet: there is no host gap left to record. |

### 10.7 What to check on the next GPU run

1. `[StickyKV] fused decode tiling tuned sig=... -> ...` and
   `[StickyKV] read gate tiling tuned sig=... -> ...`, plus the per-rung
   milliseconds `profile_decode.py` now prints under each. **The margin is the
   result**: a rung that wins by 30% and one that wins by 0.5% call for
   different next moves.
2. `eviction: N compiled / 0 eager runs, Inductor kernels X ms/step over Y
   launches/step` with **X and Y non-zero**. If they are still zero the eviction
   is still not fusing, and the body now raises rather than letting that pass.
3. Whether the "shared/other" 11.75 ms bucket shrinks once the eviction fuses.
   That is the first honest measurement of what the compiled eviction is worth —
   every previous one measured the eager chain.

### 10.8 The first GPU run, 2026-09-19 — it broke the table (`table_v9`)

Reported honestly because §10.7 said what to check and the answer was "worse".

| cell | result |
|---|---|
| 4096/257 B=1 | ran. **TPOT 0.1917** |
| 4096/257 B=32 | `RuntimeError: the two-tier eviction ran EAGER` at decode step 48 |
| everything after | sticky "already failed on this build" |

The gate was fine throughout — `ran on all 24576 fused layers, realised read
fraction 0.251`.

**Two separate faults, both introduced by §10.2–10.4.**

**(a) The tile search fired during the fill phase, and that is the 0.1917.** The
signature carried `n_sel.bit_length()`, and a decode run does not hold one
geometry: the Q tier fills at one window per `ws` steps, and the perf runner
warns in this very log that the cell is shorter than the fill (*"needs ~744
steps ... and this cell runs 512"*). So `n_sel` climbed across the whole run,
the signature changed roughly five times **inside the measured window**, and
each change ran a fresh 12-rung timed search — up to twelve Triton JIT compiles
per change. Before the search existed, exactly one rung was compiled per
geometry. This was a pure self-inflicted regression, and it also tuned at
`n_sel = 1`, where every rung runs one Q-tier iteration and the comparison says
nothing about the loop being tuned.

**Fixed by `_TUNE_AFTER`: a geometry is only timed once it has been seen three
times.** A fill-phase shape is seen once and takes the first rung that fits —
exactly the old cost. A steady-state shape recurs every step, so it is tuned
once, at a representative point, and the winner serves the rest of the run.

**(b) The eviction still ran eager, and the error could not say why.** The check
named three causes and gave no way to choose between them — which cost this
entire run. The failure timing rules out the obvious reading: the B=1 cell
survived ~32 distinct fill-phase shapes, so the limit raise to 64 *did* take
effect, and the B=32 cell then failed at its **7th** eviction. A simple
"limit is still 8" cannot produce both.

**Fixed by making the error carry Dynamo's own state** (`_dynamo_diagnosis`):
the live limits and whether our raise actually took (cause 1, now *provable*
rather than inferred), `config.disable` / `TORCHDYNAMO_DISABLE` (cause 2), and
the top graph-break reasons with frame counts (cause 3). Verified against four
stubbed states — healthy flags nothing, each cause is named — and it cannot
raise while diagnosing. The limit is also now re-applied before **every**
compiled eviction, not once at build, since the static retry performs a
`torch._dynamo.reset()` in between.

**What this run did establish, and it matters:** every decode number in this
document up to `table_v8` was measured with an **eager** eviction. The compiled
eviction has never run for more than a handful of evictions in any recorded
sweep. Reverting §10.2 restores those numbers — it does not restore their
meaning.

### 10.9 Resolved, second run — a torch default flipped under the code

The diagnosis did its job on the first try. `table_v9`, 2026-09-19 09:30:

```
frames: {'total': 10, 'ok': 9}
top graph breaks (CAUSE 3 if one of these is new):
    9x  Could not guard on data-dependent expression u0 <= 0 (unhinted: u0 <= 0)
Caused by: if need <= 0 or bound <= 0:   # cache.py:168 in _evict_widths
```

**CAUSE 3, and it was never about anything I changed.** `_evict_widths` reads
three widths with one `.tolist()` and then branches on them. Its own docstring
states the intended cost — *"that `.item()` costs one graph break in the
compiled eviction — two Inductor graphs instead of one"* — and
`tests/test_evict_graph_breaks.py:258` pins it down to the recorded message,
*"Unsupported Tensor.item() call with `capture_scalar_outputs=False`"*.

**`capture_scalar_outputs` has defaulted to `True` since torch 2.6.** Under it
`.tolist()` stops graph-breaking and returns an *unbacked* symint; the branch
becomes a guard on `u0 <= 0` that Dynamo cannot resolve; and Dynamo's response
to an unresolvable guard is not a graph break — it abandons the frame and runs
the whole eviction eager.

So the eviction has been running eager on this torch build for as long as this
build has been in use, and `_EVICT_STATS` reported "compiled" the entire time.
§10.2's fix was necessary and insufficient: it removed a real specialisation
blow-up, but the frame was being abandoned for a different reason underneath.

**Why the config is the right fix and not a workaround.** The width is what
`_compact` allocates with, and `_evict_widths` exists precisely to pay one host
sync for a real integer — a width that is ever too small routes overflow to a
dump column and silently drops work. A symbolic width cannot size that
allocation without a guard, and the guard is the thing that cannot be resolved.
The sync is the point; the break that comes with it is the designed structure.
`_scalar_outputs_off` restores it as a **scoped** `config.patch` around the
compiled callable, mirroring `_emulating_precision_casts` — nothing else the
host compiles is affected.

The diagnosis now recognises this failure by name, so if the patch ever stops
reaching the trace it says so rather than re-deriving it.

**What is still unmeasured:** whether a genuinely compiled eviction is faster.
Every decode number in this document predates one.

### 10.10 Third run — the fix worked and revealed the real problem

`table_v9` again. Still one cell, still an error at B=32 — and **TPOT went UP
again**: 0.1917 → 0.1845 → **0.2162**. That third number is the finding.

**One root cause, two symptoms, and neither was introduced by §10.2–10.4.** The
compiled body forces `int()` on two shapes that move at *every* eviction:

```
cache.py:2551  T_fp = int(state.key_states.shape[2])      <- grows while filling
cache.py:2554  W    = int(state.window_scores.shape[2])   <- grows while filling
```

`H_kv`, `D` and `H_q` are model constants, and the `_evict_widths` widths are
quantised onto `_EVICT_WIDTH_LADDER`, so region 2 is bounded. `T_fp` and `W`
are not. At 4096/257 B=1 that is ~32 evictions during the fill, so up to 32
distinct `(T_fp, W)` pairs, **× 2 regions** either side of `_evict_widths`'
deliberate break — roughly **64 Inductor compiles, all inside the measured
window**.

* **TPOT 0.2162** — ~64 Inductor compiles over 255 steps is ~0.15–0.25 s/step.
  It got *worse* across the three runs because §10.9 finally let it compile at
  all; before that it bailed to eager and paid no compile cost.
* **The B=32 error at step 40** (was 48, was 80) — the B=1 cell spends the
  **process-wide** recompile budget and B=32 bails almost immediately. The step
  number moves run to run because the budget is shared and the shape order
  differs.

**This inverts the premise of the compiled eviction.** `_evict_two_tier`'s
docstring argues *"it fires once per `ws` steps, so a compile / recompile
amortizes"*. That is false while the store is filling — and the perf runner
warns, in the same log, that these cells never leave the fill phase. On this
build, at these shapes, **eager (~0.055) beats compiled (0.216)** because
compile cost dominates.

**What landed, and what deliberately did not.**

The obvious fix is to drop the `int()` forcing and let `dynamic=True` make
`T_fp` and `W` symbolic — 32 graphs collapse to one. It is **not** in this
commit. It contradicts a named hazard class in `_EVICT_COMPILE_HELP` *and*
`test_scalar_shape_reads_in_the_compiled_region_are_int_forced`, and this
session has been confident and wrong three times. It needs the graph count
first. So:

1. **`_counting_inductor` tallies the graphs Inductor actually builds.**
   `_EVICT_STATS["compiled"]` counts *calls*, which says nothing about compiles.
   If `graphs` reads ~2× the eviction count, the diagnosis above is confirmed
   and dropping the `int()`s is the fix. If it reads ~2, the diagnosis is wrong
   and the cost is elsewhere. One number instead of another argument.
2. **A Dynamo bail is no longer filed as a build failure.** `EvictionRanEager`
   is now its own type. A *lowering* failure is a build property and stays
   sticky; a *bail* is a run property — its commonest cause is a budget spent by
   an earlier cell — and routing it into the sticky flag is what turned one
   failing cell into five untried ones.
3. **One reset-and-retry per measured run.** On a bail, `torch._dynamo.reset()`
   (the only thing that returns a fresh budget), rebuild, retry once. Safe on
   the same step: the check sits at the top of the body, before any mutation.
   Cleared by `reset_evict_path_stats`, so each cell can recover once from
   another cell's spending while a cell that bails twice still fails loudly.

Routing verified against six cases — happy path, bail-then-recover, bail-twice,
rescue-already-spent, lowering failure, OOM — confirming a bail never sets the
sticky flag and that lowering/OOM behaviour is unchanged.

### 10.11 Fourth run — I blamed the counter. **It was not the counter.**

```
BackendCompilerFailed: backend='_counting_inductor' raised:
IndexError: list assignment index out of range
```

Every cell. I concluded the counting wrapper from §10.10 was at fault, removed
it, and wrote a standing rule on that basis.

**The fifth run raised the identical error under stock `backend='inductor'`.**
The wrapper was never the cause, the rule was not earned, and both are corrected
here — §10.13 has what it actually is.

Removing the wrapper was still right, on its own merits: the graph tally it
existed to collect is already published read-only in
`torch._dynamo.utils.counters`, which `_dynamo_diagnosis` was *already* reading
for `frames`. It bought a free number and added a moving part to a path that
cannot afford one. The tally now comes from `counters["stats"]`.

One number did improve: TPOT at 4096/257 B=1 went 0.2162 → **0.1700**, which is
the §10.10 recurrence gate and bail-retry taking effect. It is still ~3× the
~0.055 that the eager eviction produced, which is the point §10.10 makes.

### 10.12 Where this actually stands

Four GPU runs, four failures, and the table still has one row. What is now
established, and what it implies:

1. **The eviction never compiled on this build** (§10.9) — so every decode number
   in this document was measured eager, and `_EVICT_STATS` said "compiled"
   throughout.
2. **When it does compile, it recompiles per eviction** (§10.10) and costs
   0.17–0.22 TPOT against eager's ~0.055.
3. **The step is 97.2% GPU busy and kernel-bound** (§10.1). The compiled
   eviction exists to collapse *launches*. If launches are not the bottleneck,
   its premise is weak on this build — and it demonstrably costs compile time.

Points 2 and 3 together are the uncomfortable conclusion: **on this build, at
these shapes, the compiled eviction may not be worth having.** That is a
design-level call, not a bug fix, and it contradicts "compiled unconditionally,
one production path" — so it is recorded here rather than acted on.

The measurement that decides the cheaper question — whether the recompiles are
the cost — is one line in the next failure's diagnosis (`GRAPHS BUILT`). If it
reads ~2× the eviction count, dropping the `int()` forcing on `T_fp` and `W`
collapses them to one graph, and that is the fix. That change is still not
shipped, for the reason given in §10.10.

---

## 10.13 The eviction is not compilable on this build, by two independent routes

The fifth run's full traceback, under **stock** `backend='inductor'`:

```
cache.py:2757 in torch_dynamo_resume_in__evict_two_tier_impl_at_2701
    store.demote_many(
  _aot_autograd/runtime_wrappers.py:1023 in pre_compile
    flat_args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(
  _aot_autograd/runtime_wrappers.py:1442 in merge_view_inputs
    post_processed_calling_convention_meta[k] = v
IndexError: list assignment index out of range
```

This is an **upstream AOTAutograd bug**. It builds "synthetic bases" when graph
inputs are views aliasing a common base and at least one is mutated, and
mis-sizes its metadata list. This body is made of precisely that shape:
`_LayerStateView` → `joint.key_states` → `_key_buf`, plus the `body_k` /
`body_v` / `body_pos` slices of the same buffer.

**It fires only in the RESUMED region** — the half created by `_evict_widths`'
`.tolist()` break. Which closes the loop on §10.9:

| `capture_scalar_outputs` | what happens |
|---|---|
| `True` (torch 2.6 default) | no break; Dynamo abandons the whole frame → **eager** |
| `False` (§10.9's patch) | break; region 2 → `merge_view_inputs` → **IndexError** |

**Both roads end at "not compiled", by two independent mechanisms, and neither
is caused by anything in this branch.** §10.9's patch only moved which one
fires. The stickiness is now *correct*: this is a genuine build property, so
cells 2–6 refusing is the designed behaviour working as intended.

### The fix, 2026-09-19 — the aliasing pair, named exactly

`state.replace_body` does:

```python
self._key_buf[:, :, num_sink:n] = body_key
```

It **mutates the base that `state.key_states` is a view of** — and the same
region reads that view, as `body_k = state.key_states[:, :, num_sink:, :]`.
Three such pairs (key / value / positions). The `replace` branch is worse still:
it reads `state.key_states` *and* reallocates the buffer in one expression.

That is the synthetic-base pattern verbatim, so this was candidate (B), and it
did not need guessing which views to detach — the traceback named the call and
the buffer write is the only mutation of an aliased base in the region.

**The compiled body is now pure with respect to the fp buffers.** It returns
`(scores_before, retained_idx, new_k, new_v, new_pos, n_q)`; `_evict_two_tier`
installs the compacted body, commits the active count, and sets the policy total
— all outside the graph. Three slice-assignments gain nothing from being traced,
and `replace_body`'s own aliasing assertion is a host check that only really runs
out there.

**Two things fell out of it.** The `joint` bool selected `replace_body` versus
`replace`; with both in the caller the body no longer branches on it, so the
parameter is gone and with it a specialisation axis — **one graph per shape
instead of two**. And the body now takes exactly `(self, state, store, policy)`:
no `layer_idx`, no `step`, no `joint`.

**The invariant this establishes**, recorded in `_EVICT_COMPILE_HELP` and beside
the code: *the compiled eviction may mutate the slot table — whose fields are
separate allocations — but must never write a buffer it reads a view of.*
Putting such a write back re-breaks the build.

Verified here at AST level (no GPU): the body's signature is exactly those four
arguments; `layer_idx`, `step` and `joint` appear in no live `Name` node;
`.replace_body`, `.replace`, `.telemetry`, `.commit_active_count` and
`.set_total_after_compaction` are called nowhere in it; the caller performs every
one of them; and the body's 6-tuple return matches the caller's unpack. What is
**not** verified is that it compiles — that needs the GPU.

**The graph break stays.** It is the designed structure (`_evict_widths` pays one
host sync for a real integer) and `tests/test_evict_graph_breaks.py` pins it. It
was never the bug; it only decided which region the aliasing blew up in.

---

## 10.14 The table ran. Two findings, one of them against me.

Sixth run, `table_v9`, all six cells:

| cell | v8 TPOT (eager evict, untuned tile) | now | Δ |
|---|---|---|---|
| 4096/257 B=1 | 0.0545 | 0.0553 | +1.5% |
| **4096/257 B=32** | **0.0570** | **0.0589** | **+3.3%** |
| 2048/513 B=1 | 0.0550 | 0.0557 | +1.3% |
| **2048/513 B=32** | **0.0569** | **0.0561** | **−1.4%** |
| 1048/1049 B=1 | 0.0546 | 0.0551 | +0.9% |
| 1048/1049 B=32 | 0.0562 | 0.0562 | 0.0% |

A wash. But the two instrumented lines underneath it are not.

### §10.10's "recompile storm" is REFUTED

```
eviction path: {'eager': 0, 'compiled': 285}
dynamo: {'graph_breaks': 0, 'frames_total': 0, 'frames_ok': 0, 'unique_graphs': 0}
```

`perf_runner` zeroes those counters **after warmup**, so they measure the
measured window only. All four are zero. The eviction is **fully fused — not one
graph break — with zero recompiles inside the timings.**

So §10.10's diagnosis was wrong. There was no recompile storm; the 0.17–0.22
TPOT of runs 3–5 was compilation churning against a broken build, not a steady
state. **The compiled eviction, working exactly as designed, is worth nothing
measurable** — which is what §10.1's 97.2% GPU-busy reading predicted, because
collapsing launches cannot help a step that is kernel-bound.

### The tile search works, and first-fit was picking the 9th-best rung

```
fused decode tiling tuned -> (32, 2, 4)
  (32,2,4)=0.242  (32,1,4)=0.281  (64,2,8)=0.301  (64,1,4)=0.303  (16,2,4)=0.303
  (64,1,8)=0.329  (32,2,8)=0.332  (32,1,8)=0.335  (64,2,4)=0.351  (16,*,8)=0.41
read gate tiling tuned -> (16, 4)
  (16,4)=0.076  (16,8)=0.079  (32,8)=0.079  (32,4)=0.080  ...  (128,4)=0.102
```

**`(64, 2, 4)` — the rung first-fit always took — came 9th of 11, 1.45× slower
than the winner.** §10.3 predicted exactly this (the top rung is the one whose
~128 KB of staged operands leaves one block per SM) and the ladder's own
docstring had called its ordering an assertion. It was, and it was wrong.

The gate's derived heuristic came 4th of 8; the winner is 5% better.

### Why the win did not reach TPOT — and the leak it exposes

At 2048/B=32, 0.351 → 0.242 ms/layer is 3.49 ms of a 56.9 ms step: **6.1%
available, 1.4% observed.** ~4.7 points missing.

**Only one decode signature was announced as tuned in that cell.** A signature
must recur `_TUNE_AFTER` times to earn a search, and every step on a signature
that has not runs `_first_fit` — which walks the ladder from the top and lands
on `(64, 2, 4)`, now measured as the 9th-best rung.

So the fix: **`_LAST_WINNER` — an untuned geometry now starts from the best rung
already measured for that kernel, not from the top of the ladder.** What a rung
really selects is the occupancy / iteration-count trade for this kernel on this
card, which moves with the hardware, not with the window count, so a nearby
tuned winner is a far better prior than a position in a hand-ordered list. If it
does not fit at the new geometry it raises `OutOfResources` and the ladder walk
proceeds exactly as before — this can only change which rung is tried *first*.

Verified against a stub seeded with this run's real measurements: a tuned
geometry still costs one dict hit and one launch; a brand-new geometry takes the
prior in one launch instead of the ladder's 1.45×-slower top; a prior that does
not fit falls through; and a different kernel does not inherit it.

### Where that leaves the two changes

| change | verdict |
|---|---|
| **Measured tile rungs** | **works** — 31% off the decode kernel, 5% off the gate, both measured. Worth more once `_LAST_WINNER` extends it to untuned geometries. |
| **Compiled eviction** | **buys nothing** — fully fused, zero recompiles, TPOT unchanged. Consistent with a kernel-bound step. |

The compiled eviction still has **no control arm**, which is why it went this
long without anyone knowing it neither ran nor paid. The repo's own standing
rule — *every perf-affecting change gets a control arm, and a claim is stated
against it* — is the thing it has never had. Giving it one is the honest next
step, and it is a design decision, not a bug fix.

### 10.15 `_LAST_WINNER` cost 12% and is reverted

The seventh run, with §10.14's prior-seeded default as the only change:

| cell | v9 | with `_LAST_WINNER` | Δ |
|---|---|---|---|
| 4096/257 B=1 | 0.0553 | 0.0614 | +11.0% |
| 4096/257 B=32 | 0.0589 | 0.0655 | +11.2% |
| 2048/513 B=1 | 0.0557 | 0.0620 | +11.3% |
| 2048/513 B=32 | 0.0561 | 0.0634 | +13.0% |
| 1048/1049 B=1 | 0.0551 | 0.0616 | +11.8% |
| 1048/1049 B=32 | 0.0562 | 0.0635 | +13.0% |

**The paired rung timings are what make this attributable**, and they are worth
keeping as a technique — both runs publish the whole ladder, so every rung is a
matched pair at a fixed geometry:

| | median drift, run to run |
|---|---|
| decode rungs (11 paired) | **−0.7%** |
| gate rungs (8 paired) | +7.6% |
| TTFT (6 cells) | −1.2% … +6.0% |

So the decode kernel is **identical** between the runs and prefill is
**unchanged** — the machine and the model path are fine. Nothing that gets
*printed* moved, and yet the step slowed 12%. The only quantity that can move
without appearing in any printed number is the rung used at geometries the
search never tuned, which is exactly and only what `_LAST_WINNER` changed.

**The reasoning behind it was wrong**, and the run says so directly. It assumed
a rung selects "the occupancy / iteration-count trade for this kernel on this
card, which moves with the hardware, not with the window count." If that were
true, exporting one geometry's winner everywhere would be free. It is not: the
trade evidently **does** move with the geometry — which is the whole reason the
search is keyed per signature in the first place. Seeding one geometry's answer
across all of them contradicts the premise of the thing it was trying to help.

Reverted. The ladder walk is the default again for an untuned geometry: it takes
a rung known to be mediocre at the tuned geometry (9th of 11 there), but
mediocre-and-local beats good-somewhere-else, and 12% is the price of the
difference.

**What stands:** per-signature tuning (§10.14's 31% and 5% are real, measured at
their own geometries) and `_TUNE_AFTER`. **What does not:** extrapolating a
winner across geometries.

The gate's +7.6% across all eight rungs is unexplained and is not this change —
it is a kernel timing at a fixed config. Worth watching; ~8% of the step, so
~0.6 points of the 12.

### Where five runs leave this

| run | result |
|---|---|
| 1–2 | eager bail, no table |
| 3 | eager bail at B=32; TPOT 0.2162 |
| 4 | wrapper blamed, wrongly; TPOT 0.1700 |
| 5 | same error under stock inductor; TPOT 0.2037 |

The one row that prints is 3–4× *slower* than the eager baseline, because the
compiled path pays recompiles it never amortises (§10.10).

Three established facts now point one way:

1. the eviction has **never** compiled on this build (§10.9, §10.13);
2. when forced toward compiling, it costs 0.17–0.22 TPOT against eager's ~0.055
   (§10.10);
3. the step is **97.2% GPU busy and kernel-bound** (§10.1), so the launch
   collapsing the compiled eviction exists to buy is not the bottleneck.

**This is now a design question, not a bug.** "Compiled unconditionally, one
production path, no fallbacks" is a deliberate stance, and on this build it
yields a total outage in service of an optimisation the profile says is not
needed. Changing it means saying so openly and renaming the counters that
currently claim "compiled" — it is not something to do quietly, and it is not
mine to decide.

---

**None of 10.2–10.4 has been measured on a GPU, and the one run that tried made
things worse before §10.8's fixes.** They are a silent-fallback
repair and two searches replacing two assumptions; the searches cannot be wrong
about which rung is faster, but whether a faster rung *exists* is exactly what no
one has ever asked the hardware.

---

## 11. The RULER 32k regression — two decode defects, 2026-09-22

RULER 32k at `ws=32` (budget 0.20, q 0.70 bytes, gate 0.25) against the
QEvict reference: `niah_single_3` 99.0 → 71.4, `cwe` 43.98 → 18.56,
`niah_multivalue` 95.9 → 87.1, `niah_multikey_2` 97.0 → 91.0, `qa_1` 84.6 →
78.0, while `niah_single_2` held (100 → 99.2). At `ws=8` it was worse:
`niah_single_3` 2.40, with the model copying the first ~8 characters of each
UUID and inventing the rest. That pattern — the first tokens of a copy right,
the rest wrong, and the task with the longest verbatim copy hurt most — pointed
at the decode step, not the prompt compression.

### Defect 1: the skipped windows were credited with the needle's mass

The gate reads 25% of the int2 tier. Every skipped window is credited with
`c · exp(logmass)` — for its eviction score **and** as the weight its value
centroid attends with — and `c` was `Σ exact / Σ estimate` over the read
windows. That ratio of sums is right only when the card is off by one common
factor. On a retrieval step it is not: a head copying a token out of a READ
int2 window puts nearly all its mass on that one token, which a rank-1 card
does not model, so `Σ exact` *is* that token and `c` becomes that token's
estimation error. Every skipped window was then credited with a share of the
needle — about 3× the needle's own mass at ratio 0.25 — and the output was
renormalised by `1 + Σ fill`: the needle's value diluted by that factor and
replaced by a blend of centroids. The sharper the head, the more mass it handed
to windows it did not read. That is backwards, and it is exactly the step RULER
is made of.

`c` is now the **mean log ratio**, `exp(mean(ln exact − logmass))` over the
read windows: it still recovers a common factor exactly, and one outlier moves it
by `1/n_sel` of its log instead of by all of it. Kernel (`GATED` epilogue), CPU
oracle (`two_tier_window_reference`) and spec (`fill_skipped_window_scores`)
changed together. `ratio = 1.0` never had the defect — nothing is skipped — so
the control arm could not have shown it.

**CPU-only, on the kernel's own oracle with real cards** (synthetic keys with a
massive-activation common mode; the hot token is a random token of a random
window, not the farthest point the card is built on). Relative output error vs
full attention, 840 windows × `ws=32`, ratio 0.25, 4 seeds:

| head | shipped fill | mean-log fill | no fill |
|---|---|---|---|
| diffuse | 0.257 | 0.257 | 1.505 |
| one hot token in a read int2 window, sharpness 8 / 12 / 20 | 0.253 / 0.530 / 0.301 | 0.021 / 0.002 / 0.000 | 0.925 / 0.042 / 0.000 |
| hot token in the fp tier, sharpness 8 / 12 | 0.019 / 0.002 | 0.020 / 0.003 | — |

Same shape at 1600 × `ws=8` and at ratio 0.10. The credited mass now tracks
what the skipped windows truly hold in every regime (e.g. 2.34 vs 2.34 diffuse,
0.42 vs 0.43 at sharpness 8, ~0 vs ~0 when sharp);
`tests/test_gate_fill_calibration.py` pins that, and fails on the old fill.

### Defect 2: the running window score was fp16

`hooks.py` handed the cache prefill scores in `q.dtype`, and
`state.window_scores` takes its dtype from them. At 32k a window's prefill total
is a sum over up to 32k query rows; a decode step adds well under 1e-2 per
window, which is below half an fp16 ULP of any total above ~32 and rounds to
nothing. So every re-eviction ranked on the prompt alone. `446a7eb` measured
this on the July int2 branches (52% of per-step contributions lost in fp16, the
kept set ~9% off over 512 steps) and fixed it on the Qwen branch only.
`SCORE_ACCUM_DTYPE = float32` now. Arithmetic, not measured on a GPU.

### What this does not change, and what is left

* **Scores move — this is a quality change.** Quality rows do not compare
  across it, and it has no LongBench or RULER run behind it yet. The Triton
  epilogue compiles for sm80 (`tests/test_kernel_compiles.py`); it has not run.
* **The card's bytes are in the budget** (`ac128e8`), so at 32k/`ws=32` the
  cache holds 59 fp + 840 int2 windows and **drops 97** (~10% of the prompt);
  at `ws=8` it drops 1,653 (41%). That is honest accounting, not a defect, and it
  is most of why `ws=8` is so much worse than `ws=32`.
* **Gate recall is untested on real attention.** A rank-1 card models the
  window's farthest token; a copy step's target is usually some other token. If
  the fixed build still trails QEvict on NIAH, `quant_gate_ratio=1.0` on the same
  build separates gate recall from everything else.

Next GPU run, in this order: RULER 32k `ws=32` at gate 0.25 on this build; the
same at gate 1.0; LongBench at the operating point for the quality-change record.

---

## 12. The RULER 32k regression was the gate's GQA union — 2026-09-23

### §11 did not move the numbers, and the way it did not is the clue

RULER 32k at `ws=32` on `0b21c9f` (the §11 build), against the build before it:

| task | QEvict | before §11 | after §11 |
|---|---|---|---|
| cwe | 43.98 | 18.56 | **18.56** |
| niah_single_3 | 99.00 | 71.40 | **71.40** |
| niah_multikey_2 | 97.00 | 91.00 | **91.00** |
| niah_single_2 | 100.00 | 99.20 | **99.20** |
| niah_multivalue | 95.90 | 87.10 | 87.55 |
| niah_multiquery | 97.00 | 95.70 | 95.40 |
| qa_1 | 84.60 | 78.00 | 78.20 |

cwe, niah_single_3 and niah_multikey_2 (three of the four largest drops) and
niah_single_2 did not change to the second decimal, out of 500 examples each.
The other three moved by under half a point. §11 changed the weight the skipped
windows attend with, so on these tasks what the skipped windows carried was not
what decided the answer. Which windows got read was.

### The defect

The gate reads `n_sel = ceil(0.25 · n_active)` windows per **KV head**, for the
four query heads that share it. It chose them by the group max of each query
head's raw card estimate, `est_h(w) = max_i scale · q_h·k̂_i`. That is a raw
logit, and it carries two things the head's own softmax ignores:

* the head's **baseline**, `scale · q_h·anchor`, the same for every window of
  that head, and different between the heads of a group by several logit units;
* the head's **scale**, `|q_h|`, which sets the spread of its logits.

So the max over the group was decided by whichever head had the largest
baseline or the widest spread. The other three were read only where that head
happened to agree with them. A retrieval head that shares its KV head with a
louder one loses its needle window whenever the louder head does not also rank
it in its top quarter.

That fits the RULER pattern. Retrieval is a *copy*, one step per answer token,
and each step needs the needle window read for the heads doing the copy:

* niah_single_3 copies a ~25-token UUID and lost 28 points;
* niah_single_2 copies a ~3-token number and lost 0.8;
* cwe lost 25. An aggregation head gets a quarter of the list read, and under
  the old union that quarter was chosen for whichever head of its group was
  loudest. This one is a consistent explanation, not a demonstrated one.

### Why nothing caught it

The recall test that justified ratio 0.25 (`test_recall_at_the_shipped_25_percent_ratio`)
measured recall of the group's *summed raw* mass, `Σ_h exp(logit_h)`. That sum
is dominated by the same loudest head the union served, so metric and selection
shared one blind spot. It read **1.000**. Per query head, on the same fixture:

| ratio | worst query head, old union | worst query head, share union |
|---|---|---|
| 0.10 | 0.0000 | 0.977 |
| 0.15 | 0.0001 | 0.995 |
| 0.25 (shipped) | **0.0005** | **0.9995** |
| 0.50 | 0.0020 | 1.0000 |

Over 8 seeds × 32 heads at 0.25, the old union left **6.6%** of query heads with
less than half their mass read, and 17.6% under 99%. The share union left none
under 99%.

The §11 fixtures could not see it either: every query head of a group there
points at the same target, so they agree.

### The fix

Rank each window by the largest **share of its own int2 mass** any head of the
group puts on it:

```
share_h(w) = logmass_h(w) − logsumexp_w' logmass_h(w')      per query head
score(w)   = max_h share_h(w)                                the GQA union
```

This is what each head's own softmax computes, so it ignores baselines and
scales exactly as the softmax does, and it is shift-invariant per head
(`test_selection_ignores_a_per_head_baseline`). A head's shares sum to 1, so
every window the cards estimate to hold more than about `rep / n_sel` of ANY
member's int2 mass is read. That is about 2% at the 32k operating point. It ranks on `logmass`
(the whole window) rather than `est` (one token's rank-1 estimate). At `ws=32`
the token a query wants is rarely the card's farthest point, and on the
oracle `logmass` read the needle on every seed where `est − logsumexp` read it
60–90%.

Changed together, as the kernel-or-error contract requires:

* spec — `sketch.head_log_share`, `sketch.group_share`;
* CPU reference — `gate_kernel.gate_reference`, `QuantizedStore.gate_and_select`;
* Triton — `_gate_kernel` now emits only `logm`, and a new `_gate_share_kernel`
  (one program per `(row, KV head)`, two passes over `logm`) writes the union.
  The union needs each head's logsumexp over every window, and the card kernel
  splits windows across programs, so this is a second launch. It moves
  `2 · rep · NW` fp32 per program, about 3% of the card kernel's traffic.
  `gate_share_tiled_reference` mirrors its tiling and masking on CPU.

Nothing else moves. The card, `ratio`, the budget, the eviction ranking and the
fill are all unchanged, and `quant_gate_ratio = 1.0` still selects every window.

### Evidence, CPU only

These use the kernels' own oracles and real cards from `build_sketch`.
`tests/test_gate_union_is_per_head.py` (48 tests) pins them. 10 of them fail on
the parent's selection, and the file's guards run the old rule on each fixture
and require it to fail.

**RULER-shaped** (840 int2 windows at `ws=32`; per group, three ordinary heads
with random directions and a per-head baseline spread of 0, 4 or 8 logit units,
plus one retrieval head pointing at a random token of a random int2 window;
6 seeds; the ranges span the three spreads):

| needle share of the retrieval head's int2 mass | needle window read, old | needle window read, share |
|---|---|---|
| 0.02 | 0.35–0.44 | 0.98 |
| 0.36 | 0.42–0.52 | 1.00 |
| 0.95 | 0.50–0.62 | 1.00 |

**Decode output**, `two_tier_window_reference`, relative error against reading
the whole tier (240 windows at `ws=32`, per-head scales 0.5–2× and baselines,
5 seeds):

| head | old union | share union |
|---|---|---|
| retrieval, needle in one window | 0.501 | **0.005** |
| retrieval, needle across two windows (a UUID over a boundary) | 0.534 | **0.077** |
| diffuse heads (mean over all heads) | 0.654 | 0.524 |
| sink-heavy heads (~99% of mass on a sink) | ≤ 0.006 | ≤ 0.003 |

An oracle that also knew each head's true int2-tier fraction of its total mass
(the gate cannot, without reading the fp tier) improved diffuse heads by
≤ 0.05 and changed nothing else. The plain share takes nearly all of the gain.

**What the union fix cannot reach: a rank-1 card at `ws=32`.** The fixtures
above give the needle's window a shared key component, which is what
neighbouring tokens that share context would produce. With it removed, the needle token is
the only thing in its window aligned with the query. The card's mean then
carries it at `1/ws` weight, and its direction points elsewhere. Needle window
read, 8 seeds × 4 groups, needle logit margin 8–24:

| keys | `ws` | share union | old union |
|---|---|---|---|
| shared window component | 8 or 32 | **1.00** | 0.38–0.97 |
| no window component | 8 | **0.97–1.00** | 0.44–0.69 |
| no window component | 32 | **0.56–0.66** | 0.22–0.28 |

So at `ws=32` a residual miss rate is possible that no union rule can remove.
It is bounded by how much window structure real needle keys have, which only
a GPU run can say. Step 2 below measures it.

### What is verified and what is not

| check | result |
|---|---|
| `tests/test_gate_union_is_per_head.py` (new) | 48 passed; 10 fail on the parent's selection |
| the gate, sketch, store, fill-calibration and read-path suites, the new file included | 138 passed. `test_gate_fill_calibration.py` now runs the shipped selection, not a restatement of it |
| `tests/test_kernel_compiles.py` (Triton 3.3.1, sm80) | 24 passed, including `_gate_share_kernel` at `rep` 1/2/3/4/8. The decode-kernel cases were repaired: they passed `BLOCK_R=4`, which `tl.dot` rejects and the dispatcher never launches |
| GPU execution, the Triton numbers, any RULER or LongBench score | **not done** |

### Next GPU run

1. **RULER 32k, `ws=32`, gate 0.25, this commit**, with the same command as
   `scripts/run_ruler32k_windows_8gpu.sh`. Check `read_gate == gated` in every
   sidecar.
2. If a task still trails QEvict, run **the same at `quant_gate_ratio=1.0`**.
   What remains at 1.0 is not the gate. It is the int2 tier and the eviction
   (97 windows dropped at `ws=32`, §11). The gap between 0.25 and 1.0 is the
   card's `ws=32` capacity limit above. If it is large, the next lever is the
   selection's evidence, not its union. One candidate is keeping a window read
   while the previous step measured it holding a head's mass, which fits how a
   copy walks one window token by token.
3. **LongBench at the operating point.** The gate reads different windows now,
   so quality rows do not compare across this commit.

---

## 13. The observation path now reads what the decode reads — 2026-09-24

The QEvict observation suite (`modules/evaluation/qevict_observations.py`) and
the parity runners behind it had not followed §11 or §12. They scored the cache
as its tier tags describe it, not as a decode step reads it, and their oracle
carried the §11 fp16 defect. Five things, all fixed together:

| # | what was wrong | effect |
|---|---|---|
| 1 | `BaseParityRunner` accumulated the ground-truth cumulative scores in the attention dtype (fp16) | the oracle ranking behind Observations I, III and the tier study was the prompt's ranking — §11 defect 2, left standing in the ground truth |
| 2 | nothing recorded which int2 windows the read gate opened | Future Missed Mass counted the whole Q tier as readable; on the flash path a decode opens ~25% of it per KV head |
| 3 | the policy's Observation III `P01`/`P10` were reported as promotion/demotion | they are fp-membership flips: `0` pools Q with evicted windows, so an evicted window is a "promotable" 0 at every later event |
| 4 | the parity ours run dropped `quant_budget_mode` and recorded no `read_gate` verdict | a `tokens` request ran `bytes`; a gated and an ungated run produced indistinguishable npzs |
| 5 | articles shorter than `prefill_len` were used as-is | every observation assumes the metadata `prefill_len`, so a short article shifted its whole window geometry |

### What changed

* **Base, schema 1.3.** fp32 running sum (storage stays fp16); articles shorter
  than the prefill are skipped and `article_indices` recorded (ours replays them);
  opt-in `step_window_scores_heads` `[S,T,L,H,W]` — per-query-head step mass,
  because anything summed across query heads must be normalised per head first.
* **Ours, schema 1.3.** `gate_read` `[S,T,L,H_kv,W]` and `gate_fired` `[S,T,L]`:
  the gate's pick, captured by an evaluation-only observer in `_run_fused`
  (`flash_decode.set_gate_observer`; one dict read per fused layer when unset)
  and mapped to window ids through `active_ids()` at the moment the gate reads
  them. Metadata carries `read_gate`, `quant_gate_ratio`, `quant_budget_mode`,
  `quant_card_bits`, and the GQA head counts.
* **Observation IV — the decode read ledger.** Each query head's ground-truth
  mass per decode step, split into fp / local / fresh (exact), `q_read`
  (opened by the gate for that head's KV group), `q_skipped` (centroid only),
  `evicted`, `sink`. Gate recall is per head, with the worst head and tail
  fractions, beside a same-size hindsight pick (best per-step group share).
  This is the §12 per-head recall table on real attention, which §11 listed as
  untested.
* **Observation V — tier dynamics.** The F / Q / E chain with promotion,
  demotion and eviction rates, and whether each move landed on the next `H`
  steps' attention (lift, hindsight hit rates, swap gain, eviction regret), plus
  decision fidelity: the fp set vs the ground-truth ranking of the same
  survivors, which is where gate-filled scores for skipped windows would show.
  Tier study M7 gains the same chain rates.
* **`scripts/run_qevict_observations.sh`** runs one dataset end to end at the
  operating point, on the flash backend, with the gate 1.0 control arm, for
  wikitext/pg19, any LongBench jsonl (`context` field) or a RULER task.

### What is verified and what is not

| check | result |
|---|---|
| `tests/test_observation_gate_and_tiers.py` (new, 37) | pass on CPU — including the observer driven through the real cache and `_run_fused` (CPU gate reference, stubbed decode kernel) on the per-layer and layer-major paths |
| the observation, metric, parity and corpus suites | pass |
| any number from a real run, any GPU execution of the recording path | **not done** |

No decode output, eviction decision or score moves: the only production-path
change is the unset observer check. The base run's oracle *does* move (item 1),
so observation numbers do not compare across this commit.

### 13.1 Every experiment, one command — and the two ablation knobs it needed

`scripts/run_recoverkv_all.sh` runs the whole programme on one GPU queue,
resumably, then scores and writes `report/report.html`
(`modules/evaluation/recoverkv_report.py`): Observations I–V on a dataset panel
(arms `gate0.10/0.25/0.50/1.0`, `card4`, `oneway`, `original`), and the accuracy
grids for E1 (gate ratio), E2 (Bidir vs OneWay, LongBench 16 × 5/10/20% +
RULER 13 × 20%) and E3 (A dequantized / B original oracle / C no promotion).
`DRY_RUN=1` prints the 224-job list; `SMOKE=1` runs a trimmed grid in about an
hour. The report refuses any accuracy file whose `read_gate` is not `gated` or
whose sidecar knobs contradict its arm, and lists it.

Two knobs, both **quality changes**, both off by default (the shipped method is
`bidir` / `dequant`, byte- and code-path-identical to before):

| knob | values | what it is |
|---|---|---|
| `quant_promotion` | `bidir` \| `oneway` | `oneway` blocks int2 → fp promotion (`F → Q → E`) at exactly matched tier sizes: fp slots go to the best non-int2 windows, int2 slots to the best of the rest. One extra `lookup` per eviction, on that arm only. |
| `quant_promote_source` | `dequant` \| `original` | `original` keeps each demoted window's exact fp K (post-RoPE) and V in a slot-table shadow and promotes from it. The shadow is **outside the byte budget**: an evaluation-only oracle, never a method; config construction warns. |

Pinned on CPU through the real compiled eviction (`tests/test_promotion_knobs.py`,
15): Bidir promotes and OneWay never does, at identical fp/int2 counts every
step; the oracle's promoted windows are bit-identical to what was fed while
dequant's differ by the int2 error. Not run on a GPU.

### 13.2 One run, every datapoint: the observation collector

`scripts/collect_observations.sh` (`modules/evaluation/observation_collector.py`)
takes a model path, a dataset, the window size, cache budget, gate ratio and
quant ratio, runs `--num-samples` prompts of `--prefill` tokens (default 512)
plus `--gen` greedy decode steps (default 512) through the real cache, and
writes one zip: `meta.json`, `SCHEMA.md`, `sample_NNN.npz`.

No second run is needed for ground truth: `update()` is wrapped so a shadow of
the full, never-evicted KV sits beside the cache, and at every step and layer
the model's own post-RoPE query (off `q_proj`) is scored against it. So every
per-step array -- full-KV window mass per head, tiers, the gate's per-KV-head
pick and the card estimates it ranked on, the cache's own per-step mass (fill
included), the exact ranking signal of every eviction, int2 reconstruction
error, and a five-rung attention-output ladder (full / kept-original /
kept-int2 / held / actual) that isolates eviction, quantization, the promotion
payload and the read gate per head -- comes from one trajectory.
`observation_collector export` writes the parity-npz pair the observation suite
reads. CPU-only so far (`tests/test_observation_collector.py`, 9: the taps
reproduce a tiny Llama's own attention output to 1e-5; the recorder on the real
cache and gate; zip -> export -> Observations I-V); never run on a GPU.

### 13.3 Promotion, priced in attention mass: Table 18 on one run — 2026-09-26

`modules/evaluation/promotion_ablation.py` builds the paper's Table 18 (no promotion
vs with promotion) from **one** bidirectional collector run, paired on the same
prompts and queries. The paper's pilot compares runs on different articles. The
no-promotion column replays the one-way policy on the run's own `rank_signal`, and
the read gate is re-run by its own rule on the recorded cards. Windows the run had
promoted get no card, so the gate is bracketed: `open` (always read, favours no
promotion, the primary arm) and `closed`. Write-up:
`reports/promotion_mass_q0.7_wikitext.md`; paper table:
`reports/tables/promotion_ablation.tex`.

| q = 0.70 (2 fp + 26 int2) | no promotion | with | Δ (`open` … `closed`) |
|---|---|---|---|
| R3 FMM (fp + int2) | 28.89% | 28.89% | 0 (kept set identical at every eviction) |
| FullKV mass on the fp tier | 1.00% | 1.07% | +0.064 pp |
| mass reached only through the card fill | 2.90% | 2.87% | −0.03 … −0.10 pp |
| R_Q (int2 mass in opened windows) | 59.07% | 59.24% | +0.17 … +0.84 pp |
| Global LIR (fp) / Q→F transitions | 0 / 0 | 0.69% / 233 | |
| promoted vs demoted window, mass per step | 0.39% | 0.78% | 2.0× |

Every mass row favours promotion, and every CI excludes zero, even under the bracket
that favours no promotion. **At q = 0.70 the aggregate effect is small**, for three
reasons: promotion reorders the kept set rather than changing it; the fp tier is two
windows; and the gate already reads the window promotion would pin 92% of the time.
At q = 0.20 the same table moves R_Q +3.4 … +9.4 pp and fill-only mass −17 … −34%.
The pilot's magnitudes (R_Q +6.8 pp, FMM −0.7 pp, QSA +0.038) do not reproduce at
q = 0.70. QSA cannot show them in a replay, because both arms carry the same scores.

Checks, both runs: replaying `bidir` reproduces the recorded tiers at 16,384 / 16,384
layer-evictions; the gate rule reproduces 99.6% / 99.9% of the recorded picks (fp16
card rounding); no replayed window lacked a score. `tests/test_promotion_ablation.py`
(17) pins the replay against `EvictionPolicy.compute_two_tier_retain` for both
arms, and the gate against `sketch.group_share` + `select_windows`. **Not done:** a
measured `--promotion oneway` collector run (the command is in the write-up). The
export was split into `parity_base` / `parity_ours` for this; output is
byte-identical to before, checked on a 2-prompt zip cut from the real run.
