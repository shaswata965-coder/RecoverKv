# Distance to the goal

Updated 2026-09-18.

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

| shape | B | Flash | best ever | vs Flash | now | vs Flash |
|---|---|---|---|---|---|---|
| 4096/256 | 32 | 0.086 | 0.0626 | **1.37× faster** | 0.0763 | **1.13× faster** |
| 2048/512 | 32 | 0.055 | 0.0531 | 1.04× faster | 0.0644 | 1.17× slower |
| 1024/1024 | 32 | 0.043 | 0.0488 | 1.13× slower | 0.0586 | 1.36× slower |
| 4096/256 | 1 | ~0.024 | 0.0471 | 1.96× slower | 0.0566 | 2.36× slower |
| 2048/512 | 1 | ~0.024 | 0.0468 | 1.95× slower | 0.0574 | 2.39× slower |
| 1024/1024 | 1 | ~0.024 | 0.0467 | 1.95× slower | 0.0573 | 2.39× slower |

Two things this says plainly:

**"Beat Flash across all" was never close.** Even at our best, four of six cells
lost. The one comfortable win is the long-prompt, large-batch cell — which is the
cell the method is *for*, and the only one the headline ever quoted.

**Nothing in this table has moved since 2026-09-16.** Every commit since is a
memory change, a correctness fix or an op-count cut. The next
`run_perf_table.sh` replaces this table; this document does not.

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
| Timing the tile ladder / gate launch makes anything faster | **not measured.** No GPU here. The searches are landed, unrun (§10.7) |

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
| Decode tile ladder timed instead of first-fit, `num_warps` searched (§10.3) | the chosen rung is the ladder's worst-occupancy one; unknown whether a better exists | unmeasured |
| Read gate's `BLOCK_W` / `num_warps` searched instead of derived (§10.4) | 8.2% of the step, never tuned | unmeasured |

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

## 10. The 2026-09-19 GPU profile, and the three things it changed

`scripts/profile_decode.py --prefill 4096 --batch 32 --steps 24 --warmup 12` on
the shipped config (`n_q=230`, `W_retained=273`, gate live at `read_fraction
0.252`). This is the first profile taken at the headline cell on the gated path,
and it answers two of the three questions §4 listed as **never run at all**.

### 10.1 What it says

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

Two candidate fixes, **neither attempted**:

- **(A) Remove the break.** Compute the widths in the *uncompiled* caller,
  between two compiled halves, so the host sync happens in Python and no region
  carries a data-dependent branch. This is the correct structure — it makes
  explicit what the graph break does implicitly — and it is a substantial
  refactor of a 300-line body.
- **(B) Break the input aliasing** feeding the resumed region. Smaller, but
  which views to detach is not knowable without running it.

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
