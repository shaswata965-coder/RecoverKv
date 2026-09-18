# Distance to the goal

2026-09-17, updated the same day after items 1 and 2 of "The ceiling" landed.
Their entries there say what they cost and what they did not buy; the latency
projection they were sized against is withdrawn.

The goal is to beat Flash FullKV, int2 KIVI and QEvict across all six cells.
Today the branch beats Flash in **one of six**; at the best numbers this project
has ever recorded it beat Flash in **two of six**. The three B=1 cells are not
reachable by this method and never were — decode at B=1 reads every weight once
per step, a ~10.3 ms/token floor that no cache change touches. KIVI and QEvict
remain unmeasurable: the seam exists, nothing implements them.

> Section references below (`DECODE_HISTORY §1`, `TARGET_GAP §2/§5/§6`,
> `GATE_REGRESSION §2/§5`) point at working notes deleted in `0974687`. They are
> kept as provenance for each measurement; the substance is in `README.md`.

---

## Against Flash, cell by cell

TPOT_steady in seconds. **Flash** is FullKV with `flash_attention_2`. **Best** is
the published table (`DECODE_HISTORY §1`, budget 0.50 / local 64) — the fastest
this project has ever been. **Now** is the 2026-09-16 run.

**Nothing in this table has moved since 2026-09-16, and nothing below it has
been timed on a GPU.** Every commit since then is either a memory change, a
correctness fix or an op-count cut; the arithmetic for what they do to a decode
step is in §"What the landed work and proposal 3 do to the operating point", and
it comes to **±0.4 ms on a 76.3 ms step** — inside run-to-run noise. The next
`run_perf_table.sh` is what replaces this table, not this document.

| shape | B | Flash | best ever | vs Flash | now | vs Flash |
| --- | --- | --- | --- | --- | --- | --- |
| 4096/256 | 32 | 0.086 | 0.0626 | **1.37x faster** | 0.0763 | **1.13x faster** |
| 2048/512 | 32 | 0.055 | 0.0531 | 1.04x faster | 0.0644 | 1.17x slower |
| 1024/1024 | 32 | 0.043 | 0.0488 | 1.13x slower | 0.0586 | 1.36x slower |
| 4096/256 | 1 | ~0.024 | 0.0471 | 1.96x slower | 0.0566 | 2.36x slower |
| 2048/512 | 1 | ~0.024 | 0.0468 | 1.95x slower | 0.0574 | 2.39x slower |
| 1024/1024 | 1 | ~0.024 | 0.0467 | 1.95x slower | 0.0573 | 2.39x slower |

Two things this table says that the project's own documents have not said
plainly.

**"Beat Flash across all" has never been true, and was never close.** At the best
numbers ever recorded, four of six cells lost. The one comfortable win is the
long-prompt, large-batch cell — which is the cell the method is *for*, and the
only one the published headline quotes.

**The current gap to "best" is not the main problem.** It is +9.5 to +13.7 ms,
and `TARGET_GAP §6` sizes the recovery at −11.5 to −27.5 ms for B=32 — so the
B=32 row of the published table is recoverable. Recovering it still leaves three
B=1 cells roughly 2x slower than Flash and one B=32 cell 1.13x slower. **Closing
the regression and reaching the goal are different problems, and only the first
one is on any current plan.**

---

## B=1 is out of reach, and that is half the goal

At B=1 a decode step reads every model weight once: 16.06 GB in fp16, a **~10.3
ms/token floor on an A100** that is independent of cache size. Flash sits at
~24 ms. Ours sits at ~57 ms.

The gap is not KV traffic, because at B=1 there is barely any. It is the
machinery: eviction bookkeeping, the card scan, the gate's top-k, the score
write-back — all of it per layer per step, all of it launch-bound, none of it
proportional to how much cache there is. `TARGET_GAP §2` measured exactly this
signature: the gap is **~300–430 µs per layer per step and does not know how much
data there is**, moving only 1.4x across a 32x batch range.

This is the method working as designed. A KV-compression method cannot win at
B=1 — there is nothing to compress that is on the critical path — and
`eval_perf_batched.yaml` has said so in writing the whole time:

> At B=1 KV compression CANNOT show a speedup, and that is not a bug.

So one of three things has to give:

- **Drop B=1 from the goal.** State the claim at B≥32, where the method has a
  real mechanism. This is the honest option and it costs nothing but the
  headline's breadth.
- **Get the per-step machinery to near zero at B=1.** The whole D-series plus
  Block A is worth −7.0 to −13.3 ms there, against a 33 ms deficit to Flash. It
  does not close.
- **Beat Flash on memory instead of latency at B=1**, and say so — quote max-B or
  tokens-held rather than TPOT.

The third is the only one that both keeps B=1 in the story and is true. It is
also the one the byte-budget work in this session directly serves.

---

## The 25% read does not buy 25%

This is the finding that decides whether the current direction can work, and it
is already measured (`GATE_REGRESSION §2`). The gate landed as a **24% decode
regression**: 0.0626 → 0.0777 at 4096/B=32.

**Reading a quarter of the windows does not read a quarter of the bytes.** An
int2 window at `ws=8, D=128, H_kv=8`:

| field | bytes (fp16 grid) | now | share now |
| --- | --- | --- | --- |
| K int2 codes | 4,096 | 4,096 | 43% |
| **K scale/zero grid** | **4,096** | **2,176** | 23% |
| V scale/zero | 256 | 160 | 2% |
| **gate card** | 3,200 | 3,200 | 33% |
| total | 11,648 | **9,632** | |

The card has to be read for **every** window every step — that is what selecting
means. So per step the gate reads all the cards plus a quarter of the windows:

```
ungated   627 windows x 6,432 B = 4.0 MB/step
gated     627 cards  x 3,200 B
        + 157 windows x 6,432 B = 3.0 MB/step   = 75% of ungated
```

**A 25% traffic saving, not 75%** — and narrowing the grid made this *worse*,
not better, because it shrank the part the gate can skip and left the card, which
it cannot, untouched. That is the honest reading and it does not change the
decision: the grid change is a memory win (+21% context per byte) that happens to
move the gate's break-even against it. The measured saving was already smaller
than the traffic figure — 0.23 ms — because the fp tier, which the gate never
touches, is 87% of what the gated kernel reads.

Against that 0.23 ms the gate pays two costs:

- **The indirection destroys coalescing.** Ungated, `widx = w0 + t_win` is affine
  and Triton emits contiguous vector loads. Gated, `widx = tl.load(SEL + …)` is a
  runtime value with an address dependency, so every field becomes a scattered
  gather. The kernel runs **7.03 ms against a 0.58 ms roofline — 12x off**. The
  model's own GEMMs run at 1.4x theirs.
- **It added a full-tier pass.** The score-fill loop walks every Q window, not the
  selected ones, plus a prologue seeding all Q columns.

So: save 0.23 ms of traffic, pay for gathers on the remaining 0.58 ms plus two
full-tier passes. **The gate is currently a net negative, and the reason is the
memory layout, not the idea.**

The uncomfortable part: the card is now **a third of a window**, and both of the
last two changes pushed it that way — the centroid added bytes to the card, the
grid work removed bytes from everything else. Both bought something real
(accuracy, and +21% context per byte); neither helped the gate, and the gate's
break-even moved against it twice. It should be judged on that, with §3 below.

---

## What the 2026-09-17 session changed

| change | what it is worth | confidence |
| --- | --- | --- |
| The gate card enters the byte budget | Corrects a **38% overrun** on the int2 tier. `N_q` at budget 0.50 / q=0.70 goes 715 → 518 | measured; arithmetic |
| Budget remainder is spent and enforced | 99.8%+ utilisation under `bytes`; `tokens` now warns with its real 56–69% | measured |
| Skipped windows attend via their centroid | Accuracy, not speed. ~2.3x lower median output error against truncation at ratio 0.25 | measured on a CPU fixture |
| `eps` deleted, replaced by the centroid | Removes a full `[N,H,ws,D]` fp32 `recon` per eviction that nothing read | structural |
| CUDA graph deleted | Removes the OOM and the six ERROR rows; loses nothing, it replayed nothing | the repo's own measurement |
| 21 env knobs, the eager fork, FlashInfer, the PyTorch oracle deleted | No speed change. Every number now says which method it came from | structural |

Two of these need to be read carefully.

**The budget fix makes the cache smaller, and that is correct.** Holding 715 Q
windows against a budget that priced 518 was not free performance — it was an
unreported 38% more memory. Quality numbers taken before this fix were measured
at a larger cache than their `cache_budget` claimed. **Any LongBench comparison
across this commit is invalid.**

**The centroid is an accuracy change that costs bytes.** It should be kept only
if a LongBench run shows it earning its 27% of the window — or, better, if the
grid work below pays for it.

One real regression of mine, found and fixed within the session: the
`quant_read_gate` removal in `ac128e8` deleted every field of `CacheConfig`
rather than one, and the LSE rewrite spanned the transformers attention imports.
Both restored in `0974687`. The Triton front end now type-checks the new kernel
for an sm80 target and passes.

---

## What the follow-up session changed — items 1 and 2

| change | what it is worth | confidence |
| --- | --- | --- |
| One-byte scale/zero grid (`8a8cdc0`) | An int2 window 11,648 → **9,632 B**; `N_q` at 0.50/q=0.70 **518 → 627 (+21%)** for +0.2% reconstruction and +0.2% on `q·k` | measured, CPU fixture + `resolve` |
| Grouping that grid per 32 entries | The worst channel on 1000x-gain keys stays at fp16's 0.306 instead of going to 1.082, for 12 B/head | measured |
| Two grid underflow guards | A group of uniformly small entries, and a scale code rounding to zero, both decoded the grid to zero and made the fit divide by it | measured, pinned |
| Compaction by cumsum, not sort (`c65ccff`) | `aten.sort` in the compiled eviction 21 → 6 calls; `sort` has no Inductor lowering on any backend, `cumsum` has one on CUDA | op count, not time |
| The eviction's fusion, re-measured | 3 breaks (all the deliberate `.item()`), 22 fused kernels, nothing extern but `sort`/`searchsorted`/`cumsum`. **Item 2's 9.34 ms was already recovered by `6e83a2c`** | measured, CPU Inductor |
| `--gate-ratio` reaching the config (`a39124f`) | It was parsed, printed and operating-point-checked, but never written into the generated YAML: both arms of the gate A/B ran at 0.25 | read off the generated config |
| `eval_efficiency.yaml`'s fourth arm | `quant_memoize_read: false` priced a memo that is unreachable, so it duplicated `ours_b50_q50`. Now `quant_gate_ratio: 1.0`, the control `TARGET_GAP §4` said did not exist | structural |
| Two dead call sites (`c51c1ae`) | `0974687` deleted 21 knobs and left two of their READERS. Both on CUDA-only lines, so the CPU suite could not see them; a six-cell perf table died on every row with `NameError: _score_exp2_enabled`. Guarded now by a static undefined-name check over `modules`/`utils`/`scripts` | measured, the hard way |
| `lookup` walks its match 3x, not 6x (`513b3d1`) | ~500 MB of traffic and ~330 MB of peak transient per eviction | measured (op + byte profile) |
| Three micro-fixes in the eviction | −6 launching ops of 186 | measured |

**What did not change: any latency cell.** Item 1 is a memory change and item 2
turned out to be a verification. See "Where that lands" below, where the estimate
that assumed otherwise is withdrawn.

**Any quality comparison across `8a8cdc0` is invalid**, for the same reason it
was across the card fix: the stored format and the price of a window both moved,
so a row before it and a row after it are not the same cache. The two together
make an int2 window 17% cheaper than it was yesterday and the budget honest about
all of it.

---

## The ceiling, and the order to get there

Ordered by measured value, not by how interesting it is.

**1. Narrow the scale/zero grid to int8. — LANDED (`8a8cdc0`).**

The fp16 grid was **48.5% of an int2 window's codes-plus-grid** — two bytes of
precision wrapping a two-bit code. It is now one byte per entry against an fp16
scale shared by 32 of them.

| | fp16 grid | one byte, shared per 32 |
| --- | --- | --- |
| grid, per head | 512 B | **272 B** |
| bytes per int2 window | 11,648 | **9,632 (−17.3%)** |
| `N_q` at budget 0.50 / q=0.70 | 518 | **627 (+21.0%)** |

**+21% retained context at the same byte budget.** The prediction above was
+23% at 9,472 B; the difference is that this counts the shared fp16 scales, which
the estimate did not. `N_q` is read off `config.resolve`, not asserted.

Accuracy, 12 seeds, keys with massive-activation channels and a hot token per
window: **+0.17% reconstruction error and +0.20% on `q·k`** — better than the
+0.6% `GATE_REGRESSION §5` predicted, because `scale` is non-negative and takes
the full uint8 range where that measurement used a symmetric int8 for both
fields.

**The grouping is the part that was not in the estimate, and it is the part that
matters.** One scale per head is +0.2% in aggregate and hides a per-channel
failure: on keys whose channel gains span 1000x — which is what a
massive-activation channel *is* — the worst channel's own reconstruction error
goes 0.305 → 1.082, i.e. that channel decodes to noise, while the Frobenius norm
moves 0.14% and reports nothing. Groups of 32 put it back at 0.306, fp16's own
figure, for 12 B/head. **Aggregate error is the wrong instrument for a shared
exponent.** Two underflow hazards were found and closed the same way (a group of
uniformly small entries whose amax/255 underflows fp16; a scale code rounding to
zero); both would have produced a zero grid and a divide-by-zero in the fit.

**2. Make the eviction actually fuse. — the premise was stale; what was left is
done (`c65ccff`).**

`TARGET_GAP §5.2`'s 9.34 ms/step of unfused ATen kernels was measured on a build
that predates `6e83a2c`, which found the cause — eight `data_ptr` graph breaks
from `CacheState`'s aliasing guards — and removed it. Re-measured here on torch
2.14 CPU Inductor over one eviction cycle, **before** this session's change:

| | |
| --- | --- |
| graph breaks | 3, all the deliberate `.item()` width read |
| fused kernels | 22 |
| extern calls | `sort` 21, `searchsorted` 9, `cumsum` 9, nothing else |

So the body fuses, and what is left outside the fused region is the handful of
ops Inductor has no lowering for. Half the sorts were not sorting: four places
wanted "the true columns, in order, first" and wrote it as
`argsort(~mask, stable=True)`. A partition is a running count, so they are now
one cumsum and one scatter (`modules/quant/compact.py`), which is `O(n)` in one
pass instead of `O(n log n)` in several — and, the reason it is worth doing,
`cumsum` lowers to a Triton scan on CUDA and fuses with the pointwise work
around it while `sort` never lowers at all. **`sort` 21 → 6.** The rest are real
orderings, and the score ranking stays a sort deliberately: `topk` breaks ties
differently, and which window survives a tie is an eviction decision.

**Not a measured speedup.** No GPU here, so this is justified by op count and by
what each op does. The arithmetic bounds the prize at well under a millisecond
per step — these are ~500-column tensors — so it claims no part of the 13.7 ms
gap. The 9.34 ms it was aimed at was already gone.

**3. Decide the gate on evidence.** It is a 24% regression today. Either fix the
gather problem — the selected windows are scattered, so a compaction that makes
them contiguous would restore coalescing — or remove the gate and keep the cards
only as a scoring aid. The centroid work makes it *more correct*, not faster, and
correctness does not rescue a negative.

**4. Re-measure everything.** Every number in this document predates the commits
it describes, and the two budget fixes together change what `cache_budget` buys
by more than 20%.

### Where that lands

The estimate this section used to carry is **withdrawn**, and the withdrawal is
the most useful thing items 1 and 2 produced.

| | now vs Flash | was projected after 1+2 | actually expected |
| --- | --- | --- | --- |
| 4096/256 B=32 | 1.13x faster | ~1.3–1.5x faster | **unchanged** |
| 2048/512 B=32 | 1.17x slower | ~parity to 1.1x faster | **unchanged** |
| 1024/1024 B=32 | 1.36x slower | ~1.1x slower | **unchanged** |
| all B=1 | ~2.4x slower | ~2.1x slower | **unchanged** |

The projection rested on item 2 recovering the eviction's 9.34 ms. That recovery
had already happened in `6e83a2c`, before the table was written, so it was being
counted twice: once in the "now" column, which was measured after it, and again
in the "after" column. **Items 1 and 2 as landed move no latency cell.** Item 1
is a memory change (+21% context per byte, +0.2% error); item 2 removes ~15
extern kernel calls per eviction cycle from a step that has ~1,650 launches.

What that leaves is a plainer statement of the position than this document had:

* **Latency.** The 9.5–13.7 ms gap to the best-ever numbers is unexplained by
  anything on this list. `TARGET_GAP §3` splits it into a flat 4.3–5.3 ms
  (Block A: the `exp2` and RoPE-dtype changes, control arms latched, never run)
  and a gate-era remainder. The gate is a 24% regression whose cause is measured
  and structural (§3, the `SEL` gathers). Neither is a tuning item.
* **Memory.** This is where the method actually moved this week: an int2 window
  went 11,648 → 9,632 B, so the same budget holds 21% more context, and the
  budget is now honest about every byte of it. That is the axis KIVI and QEvict
  are strongest on and the one the repo cannot yet compare on, which is why
  wiring those two baselines outranks another millisecond.

The realistic goal is unchanged: *beat Flash at every B=32 shape, and beat KIVI
and QEvict on memory everywhere* — and say plainly that B=1 latency is not what
this method is for.

---

## Two utilisation numbers that look contradictory and are not

Raised as a direct question, and worth writing down because both statements are
in this repo's history and both are true.

**Earlier:** *"under `bytes`, `budget_utilisation` is 0.99 at every q; under
`tokens` it is 0.99 / 0.69 / 0.57."* That is a statement about what the two modes
MEAN. Under `bytes`, `q` splits the byte allowance and each tier buys windows at
its own price, so the cache costs what it was granted at every `q`. Under
`tokens`, the window COUNT is pinned, so cheaper int2 keys buy fewer bytes
rather than more windows and the spend falls with `q`. Nothing about that has
changed — the same configuration still reports 0.999 and 0.686 today.

**Later:** *"the card was not in `bytes_per_q_window`, so `bytes` mode was
overspending."* That is a statement about the METER, one level down. The
utilisation above is `retained_bytes / total_budget_bytes`, and `retained_bytes`
priced a Q window at its codes and grid only — 8,448 B — while the window also
carried a 3,200 B gate card. So:

At `cache_budget=0.50`, prefill 4096, local 64, `ws=8`, as the pre-fix resolver
would have bought it:

| | it reported | it actually held |
| --- | --- | --- |
| `bytes`, q=0.5 | 0.997 | **1.180** |
| `bytes`, q=0.7 | 1.000 | **1.257** |
| `tokens`, q=0.5 | 0.638 | 0.686 |
| `tokens`, q=0.7 | 0.497 | 0.563 |

**Same defect, opposite symptom.** Under `bytes` the resolver *spends* to the
meter, so an under-counted price made it hold 18–26% more bytes than the budget
granted — a real overspend, and the reason `N_q` at 0.50/q=0.70 fell 715 → 518
when the card was priced in. Under `tokens` the count is pinned, so the missing
3,200 B could not buy anything; it showed up only as under-reporting, and the
cache was inside its budget the whole time.

So "bytes spends 99% and tokens spends 70%" was never the claim that got
corrected. What was corrected is what 99% was 99% *of*. Both fixes since — the
card entering the budget, and the grid narrowing — move that price, and the
utilisation figure sits at 0.999 under `bytes` after each one, because that is
what the mode does. **`budget_utilisation` is a check on the resolver, not on the
format.** Only `bytes_per_q_window` can be wrong about the format, which is why
it now lives in the module that defines the layout.

---

## What the step is made of, and which of three proposals is worth it

Measured on 2026-09-18 with a dispatch counter (launching ops only — `view`,
`slice`, `expand` are metadata and are not a launch) and, separately, by bytes
at a real head geometry (H=8, D=128, ws=8, N_q=276, one row, one layer).

| | launching ops |
|---|---|
| steady decode step | **5** (the KV append; already minimal) |
| eviction step | **186** (layer-major: once for all L, every `ws` steps) |

So the eviction is ~82% of the cache's amortized launches, which is what the
repo has said all along. By bytes, one eviction touches 27.7 MB:

| MB | what | where |
|---|---|---|
| 8.3 | the RoPE table for the **whole** Q tier | `decode_kernel.py:185` |
| 8.6 | the fp store rebuild + replace | `cache.py`, `state.py` |
| 3.4 | the cards gather (8 fields) | `slots.py` |
| 1.7 | the codes + grid gather (11 fields) | `slots.py` |

**The largest single item is a redundant pass.** A Q window's positions are
frozen for life (§5), so its `cos`/`sin` never change — yet the table is rebuilt
for every active window at every eviction, when only the handful that entered
the tier are new. The two gathers have the same shape of waste. Making the fused
context *incremental* is worth more than any of the three proposals below and
changes nothing about what the cache keeps; it is not done here because it is a
kernel-visible change and this box has no GPU.

Landed instead (`513b3d1`): `lookup` now walks its `[B, W, N]` match three times
instead of six — 167 MB at B·L=1024, and it was reduced six ways and handed out
for a seventh. ~500 MB of traffic and ~330 MB of peak transient per eviction.

### The three proposals

**1. Freeze the rank between evictions.** A frozen rank skips the score sort, the
tier re-assignment and most of the id→slot resolution — but not sealing the new
window, not the fp-store rebuild, and not the context rebuild, which are the
expensive two-thirds. Optimistically ~30% of an eviction, and only on the frozen
ones: at one re-rank in four that is ~2% of a decode step. It is also the only
one of the three that **changes which windows the cache keeps**, so it cannot
land without a LongBench run. Worst ratio of the three.

**2. Only use cards for generation.** Measured on `two_tier_window_reference`
(the CPU statement of the kernel, gated epilogue included), against the same
function ungated — exact attention over the keys the cache holds:

| read ratio | windows read | median rel err | p95 | B/head/window read |
|---|---|---|---|---|
| 1.00 | 96 | 0.0000 | 0.0000 | 1204 |
| 0.50 | 48 | 0.0000 | 0.162 | 802 |
| 0.25 (shipped) | 24 | 0.0001 | 0.575 | 601 |
| 0.10 | 10 | 0.0063 | 0.942 | 483 |
| 0.03 | 3 | 0.532 | 0.976 | 425 |
| cards only (1) | 1 | **0.908** | 0.982 | 400 |

A 0.9 relative error is an output unrelated to the truth. The cards are a
**selector, not a substitute**: the epilogue's calibration is measured *on the
windows that were read*, so with nothing read there is nothing to calibrate
against, and a sum of centroids cannot represent variation inside a window. The
fixture is pessimistic (its fp tier is 8 windows of 104, where production's holds
the top-ranked ones), so the absolute numbers are a lower bound on quality — but
the trend is the answer, and it is steep well before the cards-only end.

**3. Simpler cards.** Same harness, same truth:

| card | B/head | err @ 0.25 | err @ 1 window |
|---|---|---|---|
| shipped (mu, v, t, vbar int8) | 400 | 0.0001 | 0.908 |
| drop `t` | 390 | **0.787** | 1.041 |
| `mu` int4 | 336 | 0.0001 | 0.909 |
| **`mu` + `vbar` int4** | **272** | **0.0001** | 0.910 |
| `mu` int4, no `t` | 326 | 0.788 | 1.054 |

**The smallest field in the card is the one that cannot go.** `t` is 10 bytes of
400 and dropping it costs four orders of magnitude of accuracy — without the
per-token projection the model is a centroid again, which is the failure the card
was designed around. The two `D`-length vectors, which are 64% of the card, take
int4 for nothing measurable: **400 → 272 B/head**, a window 9632 → 8608 B, i.e.
**+12% more retained context at the same byte budget**, on top of the +21% the
grid narrowing bought. It also cuts the gate's fixed cost by a third — the card
is the one thing read for *every* window on *every* step, which is exactly what
makes the 25% read buy only 25% of the bytes (§3).

(The ablation simulates int4 by coarsening the existing int8 codes against the
same scale; a real int4 encoder re-fits the scale to the narrower range, so these
are conservative.)

### What the landed work and proposal 3 do to the operating point

Resolved off `config.resolve` at the perf table's point (budget 0.20, local 128,
q 0.70, `bytes`, ws 8, H=8, D=128). `keys` is what a decode step attends over per
row; `Q KB/step` is what the gated kernel reads from the Q tier per row per layer
— **every card, plus the codes of the selected windows**; `sel` is how many
windows the Q loop iterates over.

| shape | variant | keys | sel | Q KB/step |
|---|---|---|---|---|
| 4096/256 | 2026-09-16 (fp16 grid, 400 B card) | 1813 | 46 | 951 |
| | one-byte grid — **landed** | 2117 (+17%) | 55 (+20%) | 1036 (+9%) |
| | + simpler cards, ratio 0.25 | 2325 (+28%) | 62 (+35%) | 914 (−4%) |
| | + simpler cards, **same `sel`** (ratio 0.186) | 2325 (+28%) | 46 (0%) | **814 (−14%)** |
| 2048/512 | 2026-09-16 | 989 | 23 | 480 |
| | one-byte grid — landed | 1149 (+16%) | 28 | 529 (+10%) |
| | + simpler cards, same `sel` (0.181) | 1261 (+28%) | 23 (0%) | **414 (−14%)** |
| 1048/1048 | 2026-09-16 | 789 | 18 | 374 |
| | one-byte grid — landed | 909 (+15%) | 22 | 410 (+10%) |
| | + simpler cards, same `sel` (0.184) | 997 (+26%) | 18 (0%) | **321 (−14%)** |

Two things fall out of this that were not obvious before the arithmetic:

**The grid narrowing raised the per-step read.** More windows fit, and the card
of *every* window is read on *every* step, so cards went 572 → 691 KB while the
codes read fell 377 → 347. Net +9%. The memory win was real and the decode cost
of it was not free — that is the gate's fixed cost, and §"The 25% read does not
buy 25%" predicted exactly this.

**The card is the only lever that pushes both ways.** Proposal 3 cuts the fixed
cost by a third, so it is the first change here that gives *more* context and
*less* per-step read at once: −4% at the shipped ratio, −14% if the ratio is
re-pinned to hold the selected count.

**And the ratio is the dial nobody has turned.** `quant_gate_ratio` is a
fraction, so every window the budget buys is automatically spent on more Q-loop
iterations — 46 → 55 → 62 across the rows above, on a kernel measured at 12x off
its roofline and therefore iteration-bound, not byte-bound. Holding the *count*
instead converts a memory win into context at flat decode work. That is a config
choice, not a code change.

### Verdict

**Do 3, skip 1 and 2.** It is the only one of the three that is accuracy-neutral
as measured, it is a storage change rather than a policy change, and it lands on
the memory axis — which is where the goal is winnable at all (§"Where that
lands") and where KIVI and QEvict are strongest. 1 buys ~2% of a step for an
accuracy risk that needs a GPU run to price. 2 is not a speed-versus-accuracy
trade at all; it is a 0.9 relative error.

---

## Two of the three baselines have never been run

The goal names Flash, int2 KIVI and QEvict. Only Flash is measurable in this
repo.

| baseline | status |
| --- | --- |
| Flash FullKV | runs — `cache_backend: dynamic` + `flash_attention_2` |
| int2 KIVI | no implementation, no config, no numbers |
| QEvict | an observation suite that scores routing decisions; no runnable cache backend, so no latency and no memory |

`perf_runner` has the seam for both — `cache_backend: external` plus a
`method_factory`. Nothing implements one.

This matters more than it looks. The two unmeasured baselines are the ones this
method should be *strongest* against, because both are memory arguments and
memory is where the byte-budget work just landed. KIVI at int2 with a fp16
per-channel grid carries the same 48% grid overhead this document recommends
attacking — which is a comparison worth having, and currently cannot be made in
either direction.

Against Flash, a KV-compression method is arguing about latency, which is the
argument it is weakest at and the one this project has spent every session on.

**Wiring the two external baselines is small work — one existing seam, no method
changes — and it is the only way to find out whether the current operating point
is competitive at all.** It should come before any further latency work, because
it may change which number the project is trying to move.
