# Distance to the goal

2026-09-17

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

| field | bytes | share |
| --- | --- | --- |
| K int2 codes | 4,096 | 35% |
| **K fp16 scale/zero grid** | **4,096** | **35%** |
| V scale/zero | 256 | 2% |
| **gate card** (after this session) | **3,200** | **27%** |
| total | 11,648 | |

The card has to be read for **every** window every step — that is what selecting
means. So per step the gate reads all the cards plus a quarter of the windows:

```
ungated   518 windows x 8,448 B = 4.4 MB/step
gated     518 cards  x 3,200 B
        + 130 windows x 8,448 B = 2.8 MB/step   = 63% of ungated
```

**A 37% traffic saving, not 75%.** And the measured saving is smaller still —
0.23 ms — because the fp tier, which the gate never touches, is 87% of what the
gated kernel reads.

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

The uncomfortable part for this session's work: adding the value centroid made
the card **27% of a window in absolute terms and 38% of the codes**, which moves
break-even in the wrong direction. It bought accuracy, not speed, and it should
be judged on that.

---

## What this session changed

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

## The ceiling, and the order to get there

Ordered by measured value, not by how interesting it is.

**1. Narrow the scale/zero grid to int8. This is the biggest unexploited lever in
the repo and it is already measured.**

The fp16 grid is **48.5% of an int2 window's codes-plus-grid** — two bytes of
precision wrapping a two-bit code. `GATE_REGRESSION §5` measured int8 + a shared
fp16 scale at **+0.6% reconstruction error**, against fp8 e4m3's +1.6%. On sm80
int8 is also the only sane choice: there is no hardware fp8 convert before sm89.

| | now | int8 grid |
| --- | --- | --- |
| bytes per int2 window | 11,648 | 9,472 (−18.7%) |
| `N_q` at budget 0.50 / q=0.70 | 518 | **~637 (+23%)** |

That is 23% more retained context at the same byte budget, for 0.6% more
quantization error, on a design that pins codes to the *stored* grid so a rounded
scale just repositions the four levels and the codes re-fit. It helps gated and
ungated reads alike, and it partly pays for the centroid. **Do not raise `ws` to
save the same bytes — measured at 20x the accuracy cost.**

**2. Make the eviction actually fuse.** `TARGET_GAP §5.2` found **9.34 ms/step —
18.5% of real kernel time — of unfused ATen kernels with fractional launch
counts** (`n < 5/step`, i.e. one step in eight, i.e. the eviction), and Inductor
contributing *exactly zero*. The compiled-evict bucket does not appear in the
rollup at all. Largest single B=32 item; worth ~0 at B=1.

**3. Decide the gate on evidence.** It is a 24% regression today. Either fix the
gather problem — the selected windows are scattered, so a compaction that makes
them contiguous would restore coalescing — or remove the gate and keep the cards
only as a scoring aid. The centroid work makes it *more correct*, not faster, and
correctness does not rescue a negative.

**4. Re-measure everything.** Every number in this document predates this
session's commits, and the budget fix alone changes what `cache_budget` means.

### Where that lands

| | now vs Flash | after 1+2 | after 1+2+3 |
| --- | --- | --- | --- |
| 4096/256 B=32 | 1.13x faster | ~1.3–1.5x faster | 1.4–1.6x faster |
| 2048/512 B=32 | 1.17x slower | ~parity to 1.1x faster | 1.1–1.2x faster |
| 1024/1024 B=32 | 1.36x slower | ~1.1x slower | ~parity |
| all B=1 | ~2.4x slower | ~2.1x slower | ~2x slower |

The B=32 row is winnable across all three shapes. **B=1 is not**, by roughly
30 ms that no item on any list addresses. The realistic goal is *beat Flash at
every B=32 shape, and beat KIVI and QEvict on memory everywhere* — and to say
plainly that B=1 latency is not what this method is for.

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
