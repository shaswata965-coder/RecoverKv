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

### Never run at all

Stage-0-style questions that no GPU session has answered: the eviction's fused
kernel timing on CUDA, the tile rung the decode kernel actually gets, and the
two external baselines (§7).

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
