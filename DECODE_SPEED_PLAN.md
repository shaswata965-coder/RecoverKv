# Decode speed plan — what is left

## Introduction

Both stated decode targets are met. At the headline cell (Llama-3.1-8B,
prefill 4096, B=32, budget 0.50, q=0.70, one GPU):

| shape / B | ours | Flash (FullKV) | |
|---|---|---|---|
| **4096/256, B=32** | **0.0626** | 0.086 | **1.37× faster** |
| 2048/512, B=32 | 0.0531 | 0.055 | 1.04× faster |
| 1024/1024, B=32 | 0.0488 | 0.043 | 1.13× slower |
| B=1, any length | ~0.047 | ~0.024 | ~1.9× slower |

- beat FullKV — **met** (1.37×)
- beat FullKV by 20% (≤ 0.0688) — **met**

Started at 0.1760, i.e. 2.05× *slower* than FullKV. How that was done, and
everything that went wrong doing it, is in `DECODE_HISTORY.md`.

**The LongBench regression is closed.** The RoPE-dtype fix (`DECODE_HISTORY.md`
§2.5) repaired a tier-dependent key representation — the Q tier was un-rotated in
fp16 and re-rotated by the kernel in fp32, so a token's key depended on which
tier held it. Against `ACCURACY_RECOVERY_PLAN.md` §1's table:

| dataset | before | regressed | now |
|---|---|---|---|
| qasper | 42.30 | 32.78 | **42.43** |
| multifieldqa_en | 55.74 | 50.89 | **55.86** |
| gov_report | 33.71 | 31.28 | **33.72** |
| trec | 71.00 | 69.00 | **71.50** |

8 of 11 complete datasets are at or above the pre-regression baseline; the two
headline losses (−9.52 and −4.85) are fully recovered. The run bundles more than
one change, so this is not a controlled attribution — but it closes the
regression, and it is the reason the fp16 RoPE change stays despite perturbing
Q-tier keys by ~5e-4.

**The governing constraint from here is that scores must not move.** That is a
stronger bar than "fp tolerance", and it reclassifies most of what this plan used
to propose. An optimisation now has to be **score-neutral by construction** —
shape, launch count, or scheduling only, with no change to what the cache keeps
or to the arithmetic that decides it. Anything that alters eviction decisions is
out regardless of how much it would buy, because the remaining speedups are small
and the accuracy position is the asset.

**A second reassessment matters as much.** The two remaining deficits are both in
regimes this method explicitly does not claim. `eval_perf_batched.yaml` states it
directly: *"At B=1 KV compression CANNOT show a speedup, and that is not a bug —
decode reads every weight once per step (16.06 GB = a 10.3 ms/token floor on an
A100) and that floor is independent of cache size."* The same argument covers
short context at B=32, where attention is a small share of the step. Chasing
either would mean spending score risk on shapes the method is not for.

## What is left

| item | score-neutral? | verdict |
|---|---|---|
| **Stage 0 profile** | yes — read-only | **do it** |
| **Raise the shared-memory tile rung** | yes — a launch constant | **do it if capped** |
| Close the `_affine_quantize` graph break | yes — byte-identical, test-pinned | only if Stage 0 says so |
| Split-K over the sequence | no — reassociates the softmax | **cut** (see below) |
| `BLOCK_R` packing | — | **cut, it is not possible** |
| `eviction_interval` | **no — changes which tokens survive** | **cut** |
| Quantization group size | **no — changes quantization error** | **cut** |
| CUDA graphs | yes, but unfair to the baselines | **cut** |
| Multi-GPU per-device grouping | yes | **cut** — 4×, not 32×, and this model fits on one card |

## The work

### 1. Stage 0 — the profile, which has still never been run

Read-only, about two minutes, no code change.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_decode.py \
    --config outputs/perf_table/_perf_table.generated.yaml \
    --prefill 4096 --batch 32 --steps 24 --warmup 12
```

Two things to read:

1. **The tile rung**, printed by the kernel as
   `[StickyKV] fused decode tiling: target_keys=… num_stages=…`. This is the
   whole of item 2 — no experiment needed, just the line.
2. **`GPU busy    xx.x %`.** Not to act on, but to record. Every ms estimate this
   plan ever made was a byte or launch count with an unverified translation to
   time, and one of them was wrong by 20× (the window-score rewrite: sized at
   ~0.5 ms, delivered ~11 ms, because the value was in launches and serial
   latency, which a traffic count cannot see). If anyone later proposes decode
   work, this number is what makes the proposal checkable.

### 2. Raise the tile rung, if it was capped

The kernel picks the largest Q-tier tile that fits in shared memory, stepping
down a ladder on `OutOfResources`. Every rung is numerically identical — only the
serial iteration count changes — so this is the one remaining item that is both
free of score risk *and* potentially worth real time:

| rung | Q-tier iterations at `ws=8` | vs the pre-tiling 179 |
|---|---|---|
| `target_keys=64` | 23 | 7.8× |
| `target_keys=32` | 45 | 4.0× |
| `target_keys=16` | 90 | 2.0× |

If Stage 0 reports 64, this is already banked and there is nothing to do. If it
reports 32 or 16, the cheapest way up is to shrink the operands rather than the
work: `cos`/`sin` are now fp16 (which already halved theirs), and staging `vv` in
fp16 for its `tl.dot` is the next candidate. A separate `BLOCK_T` for the fp and
Q tiers would let the memory-hungry Q tier shrink alone.

### 3. Close the `_affine_quantize` graph break — only if Stage 0 says so

If `STICKYKV_COMPILE_EVICT=1` versus `=0` shows little difference, the
`torch._dynamo.disable` on `_affine_quantize` is still splitting the compiled
eviction. Fixing it is byte-identical and the `aot_eager` test already pins that.
Low value, no risk, entirely optional.

## What was cut, and why

- **Split-K.** The strongest remaining lever *on paper* — the grid is
  `B·H_kv` = 8 programs on a 108-SM A100 at B=1. Cut for two reasons that
  compound: it reassociates the softmax across splits (a score change), and its
  entire payoff is at B=1 and short context, the two regimes the method's own
  documentation says it cannot win. It is also a substantial kernel change
  (partial `(out, lse)` per split, a combine pass, window sums recombined) whose
  premise — that B=1 is occupancy-bound rather than host-bound — is unverified.
- **`BLOCK_R` packing.** Listed twice in earlier drafts as "pack 4 KV heads into
  one tile". **It is not possible.** `hq = kv * rep + offs_r` means every row of a
  `BLOCK_R` tile belongs to one KV head and shares one `k_rlo`; packing four
  would need four different K tiles against four row groups, which is a
  block-diagonal product `tl.dot` cannot express. The 25% row utilisation at
  `rep=4` is real but not addressable this way, and the kernel is not
  tensor-core-bound anyway.
- **`eviction_interval`, quantization group size.** Both change what the cache
  keeps. Out under the constraint, not on their merits.
- **CUDA graphs.** Would remove per-layer Python that every method in the table
  pays, so a win sourced from it is a harness result, not a KV-cache result.
- **Multi-GPU per-device grouping.** 4 layers per group on 8 GPUs is a 4× launch
  cut, not 32×, and Llama-3.1-8B peaks near 48 GB — one 80 GB card is enough.
  Sharded runs are refused outright because a pipeline-sharded decode step times
  the interconnect, not the cache.

## Expected outcome

**If nothing further is done:** the result stands as it is —
**0.0626 s at 4096/B=32, 1.37× faster than FullKV, both targets met**, with the
eviction path score-neutral by construction and the LongBench regression closed
(introduction). Speed and accuracy are both in hand; this is a shippable
position.

**If items 1–2 are done:** one of three outcomes, all cheap to reach.

| Stage 0 reports | what it means | expected |
|---|---|---|
| `target_keys=64` | the tiling is already at full effect | **no change** — banked, close the item |
| `target_keys=32` | the Q-tier loop runs 45 iterations instead of 23 | a few ms at best; worth one fp16 `vv` attempt, then stop |
| `target_keys=16` | 90 instead of 23 — most of the tiling's benefit is unrealised | the largest score-free gain available; worth the separate-`BLOCK_T` change |

**Ceiling.** With split-K and everything output-changing cut, the score-neutral
work left is bounded by the tile rung. The stretch target (≤ 0.050, a further
12.6 ms) is **not reachable** without either the score-risky items or CUDA
graphs, and should be treated as closed rather than pending. The honest ceiling
under this constraint is **~0.055–0.0626** at the headline cell, i.e. between
1.37× and 1.56× faster than FullKV.

**The one open accuracy question** is not a speed item at all: the RoPE dtype fix
changed the Q tier's keys by ~5e-4 and repaired a tier-dependent key
representation that no equivalence test could see. It may move LongBench in
either direction, and finding out is worth more than any remaining millisecond.
