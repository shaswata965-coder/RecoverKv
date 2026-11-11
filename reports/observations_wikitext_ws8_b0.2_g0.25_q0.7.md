# RecoverKV data report: wikitext, ws 8, budget 0.20, gate 0.25, q 0.70

**Data:** `wikitext_test_articles_ws8_b0.2_g0.25_q0.7_p512_n512.zip` (1,877,430,880 bytes, sha256
verified after transfer). Llama-3.1-8B-Instruct, the first 8 wikitext-103 test articles with at least
512 tokens (the test set split on its ` = Title = ` headings), 512-token prompt + 512 greedy decode
steps each, flash-attn backend, every decode step x 32 layers x 32 query heads.

**Every number is one of four kinds.**
* **Measured:** read off the cache run: tiers, gate picks, card estimates, the cache's own window mass,
  the attention-output error ladder.
* **Ground truth:** the full-KV attention of the same query, from the collector's never-evicted shadow KV.
* **Oracle / simulated baseline:** the same run's ground-truth attention replayed through another policy
  (an exact-score ranker, recency, fp-only at the same bytes, a per-step best set). These use the cache's
  own trajectory (the same queries); they are not separate runs. The "H2O-style" baseline is given exact
  attention scores, which favours it.
* **Arithmetic:** derived from the resolved cache geometry or from other numbers here; labelled where used.

Mass is a share of the query's full-KV attention, sinks included, unless a table says "share of window
mass" (sinks excluded). Band columns average layers; `all` averages all 32. Intervals in the paper-suite
tables (Part A) are 95% percentile-bootstrap CIs over 256 (prompt, layer) traces: within-run variability,
not population CIs.

## Run integrity

| check | value |
|---|---|
| read-gate verdict | `gated`: 131,072 / 131,072 fused layer-steps (8 prompts x 512 steps x 32 layers) |
| realised read fraction | 0.2692 = 7 of 26 int2 windows per KV head, every step (917,504 of 3,407,872 windows opened) |
| resolved geometry | 5 sinks + 16 local + 2 fp + 26 int2 windows of 8 tokens; fp-only at the same bytes = 8 windows |
| bytes per window | fp16 32,768; int2 8,608 (codes + grid 6,432, card 2,176 = 272 B/head, bits mu4/v8/t8/vm4) |
| occupancy | 26 int2 windows at every decode step and layer; 64 evictions (steps 1, 9, ..., 505) |
| mass accounting | sink + all windows = 1 within 4e-4 at every step |

## Headline comparison at equal memory

| metric | RecoverKV (measured) | H2O-style fp16, same bytes | StreamingLLM-style fp16, same bytes | where |
|---|---|---|---|---|
| missed mass, mean | **9.9%** | 15.2% | 15.6% | §3 |
| missed mass, p90 over (step, head) | **26.0%** | 39.6% | 40.5% | §3 |
| (step, head) pairs missing >10% | **34.0%** | 47.1% | 47.0% | §3 |
| future missed mass, next 32 steps | **28.9%** [28.4, 29.4] | 45.9% [45.2, 46.6] | 47.9% [47.0, 48.7] | A.II |
| selection churn | **0.044** | 0.080 | 0.080 | A.II |
| evictable windows held | **28** (2 fp + 26 int2) | 8 | 8 | geometry |
| windows held incl. local | **44** (47.9% of existing) | 24 (26.1%, arithmetic) | 24 | A.I |

RecoverKV misses 35% less attention than the H2O-style baseline (9.9 vs 15.2) and 36% less than
StreamingLLM-style (9.9 vs 15.6); its future missed mass is 37% and 40% lower, its churn 45% lower.

## Summary

1. **At equal memory the two-tier cache misses a third less attention than fp16 eviction** (9.9% vs
   15.2% / 15.6%), and the next 32 steps lose 37% less (28.9% vs 45.9% / 47.9%). It is below both
   baselines in every 64-step stretch of the decode (§12).
2. **Its evictions equal an exact-score cumulative ranker's** (9.9% vs 10.0% missed; ranking Spearman
   0.988). The score is accurate; the cumulative rule is what loses mass. The best 28 windows re-chosen
   every step would miss 5.5%.
3. **The int2 tier pays for itself.** It holds 7.3% of attention and 28 windows where fp16 holds 8; with
   the same exact ranking it turns 15.2% missed into 10.0%.
4. **The read gate opens 27% of int2 windows and gets 59% of their attention** (Observation IV; 94% of a
   hindsight pick of the same size); cards rank int2 windows with Spearman 0.91.
5. **Promotion is rare but well aimed:** promoted windows draw 2.34x the average future attention, 36.7%
   of them land in the hindsight-best fp set (about 5x chance), at 0.004 output error and no extra bytes.
6. **Most of the output error is the budget itself:** eviction alone gives 0.182 of the final 0.213;
   int2 storage, promotion and gated reading add 0.031.
7. **Open item:** generated text leaving the 128-token local band is mostly evicted (91.5% of windows
   leaving it), so missed mass grows over the decode (5.2% -> 14.5%). An age-normalised ranker would cut
   the last 64 steps to 12.4% (oracle).

---

# Part A. The paper's observations (modules.evaluation.qevict_observations)

Run on the collector's parity export (`observation_collector export`), default knobs (FMM horizon 32,
LIR m = 4, H = 8), 256 traces = 8 prompts x 32 layers, 64 routing events.

## A.I Skewed importance

| top share of windows | share of importance mass | gap over uniform |
|---|---|---|
| 5% | 13.6% | 8.6 points |
| 10% | 22.4% [22.1, 22.7] | 12.4 points |
| 20% | 36.8% | 16.8 points |
| 40% | 59.2% | 19.2 points |

| importance mass to cover | share of windows needed |
|---|---|
| 50% | 32.2% [31.7, 32.6] |
| 80% | 64.9% [64.5, 65.3] |
| 90% | 79.4% [79.1, 79.7] |

* Measured capacity on this axis: the fp tier (with local) keeps 19.6% of existing windows, fp + int2
  keeps 47.9%. fp16 at the same bytes would keep 24 windows, 26.1% (arithmetic: 24 / (44 / 0.479)).
* **Importance is broad, not peaked:** the top 10% of windows carry 22%, and 80% of it needs 65% of the
  windows. Holding many windows cheaply (int2) matters more than holding a few exactly.

## A.II Window-level decisions: Future Missed Mass and Selection Churn

Future Missed Mass (FMM): at each eviction, of the attention the next H steps put on windows that
already existed, the share on windows that eviction dropped (sinks excluded). Churn: Jaccard distance
between consecutive accessible sets.

Suite output (H = 32, 60 events scored):

| policy | FMM | churn | windows kept | dropped |
|---|---|---|---|---|
| RecoverKV fp tier only (int2 dropped) | 52.3% [51.5, 53.1] | 0.107 | 18.0 | 77.5 |
| **RecoverKV fp + int2** | **28.9% [28.4, 29.4]** | **0.044** | 44.0 | 51.5 |
| simulated 8-token decisions (18 kept) | 52.3% [51.5, 53.2] | 0.106 | 18.0 | 77.5 |
| simulated 16-token decisions (18 kept) | 52.7% [51.9, 53.5] | 0.106 | 18.0 | 77.5 |

* Adding the int2 tier to the same fp tier cuts FMM by 0.234 [0.229, 0.239] (44.8%) and churn by 0.062
  (58.3%). The suite flags this pair as not byte-matched; the byte-matched contrast is below.
* 16-token decisions miss 0.4 points more than 8-token ones at the same kept count.

Byte-matched (the suite's `future_missed_mass` and `selection_churn` on its own inputs; fp16 baselines
simulated at 8 evictable windows + local, same eviction steps; bootstrap 1000):

| horizon | RecoverKV (measured) | H2O-style fp16, same bytes | StreamingLLM-style fp16, same bytes | exact ranker, RecoverKV tiers | RecoverKV fp tier only |
|---|---|---|---|---|---|
| next 8 steps | **24.2%** [23.7, 24.7] | 37.9% [37.1, 38.7] | 39.2% [38.3, 40.0] | 24.5% | 42.9% |
| next 32 steps | **28.9%** [28.4, 29.4] | 45.9% [45.2, 46.6] | 47.9% [47.0, 48.7] | 29.2% | 52.3% |
| next 64 steps | **30.7%** [30.2, 31.1] | 49.7% [49.1, 50.3] | 52.4% [51.6, 53.2] | 31.0% | 56.9% |
| churn | **0.044** | 0.080 | 0.080 | 0.044 | 0.107 |
| windows kept (incl. local) | 44 | 24 | 24 | 44 | 18 |

* At the same bytes RecoverKV's evictions throw away 37% less of the next 32 steps' attention than
  H2O-style eviction with exact scores, and 40% less than StreamingLLM-style.
* The measured cache matches the exact-score ranker on its own tiers within 0.3 points at every horizon.

## A.III Historical importance revives

Episodes: a window outside the important set for 4 consecutive evictions; revival = back in it within 8.

| selection | LIR (revived) | revived / episodes | median time to revival | cold -> hot, lag 1 | hot -> cold, lag 1 |
|---|---|---|---|---|---|
| oracle important set | 0.18% [0.12, 0.24] | 45 / 25,207 | 3 events (IQR 2-7) | 0.015% | 0.56% |
| RecoverKV fp tier | 0.2% [0.2, 0.3] | 58 / 25,254 | 3.5 events (IQR 2-6) | 0.0% | 0.7% |

* Cold windows almost never come back, so permanent eviction of the int2 tail costs little; the
  headroom a promotion path can recover is small in this run.

## A.IV The decode read ledger

Each query head's ground-truth attention at every step, split by how the decode step read it:

| part | share of head attention | share of window attention |
|---|---|---|
| sink (exact) | 57.3% [55.2, 59.2] | - |
| local (exact) | 24.5% [23.3, 25.8] | 57.1% |
| fp (exact) | 1.1% [1.0, 1.1] | 2.6% |
| int2, gate opened (read at 2 bits) | 4.4% [4.2, 4.7] | 10.4% |
| int2, gate skipped (card fill only) | 2.9% [2.7, 3.0] | 6.8% |
| evicted | 9.9% [9.3, 10.4] | 23.1% |
| decode missed (skipped + evicted) | 12.7% [12.0, 13.4] | - |

| gate measure | value |
|---|---|
| per-head recall (int2 attention in opened windows / int2 attention) | **59.2%** [58.6, 59.9] |
| hindsight pick of the same size | 63.1% [62.4, 63.7] |
| gate efficiency (recall / hindsight) | **93.9%** [93.7, 94.1] |
| mass per opened window vs a uniform pick | 2.20x |
| head-steps with >= 1% int2 attention: median / 5th-percentile recall | 54.3% / 35.0% |
| head-steps under 50% recall | 38.3% |
| worst head overall | 29.8% |

* The suite counts only head-steps with at least 1% of their attention in int2 and averages per head;
  §6 below averages every head-step (56.3%). Both describe the same picks.

## A.V Tier dynamics (F / Q / E)

Rates P(to | from) between routing events:

| lag | stay fp | promote Q->F | demote F->Q | stay int2 | evict from int2 | evict from fp |
|---|---|---|---|---|---|---|
| 1 | 99.3% | 0.06% (233) | 0.7% (233) | 99.6% | 0.33% (1,375) | 0 |
| 2 | 98.7% | 0.10% (428) | 1.3% (428) | 99.3% | 0.64% (2,660) | 0 |
| 4 | 97.5% | 0.19% (760) | 2.5% (760) | 98.6% | 1.24% (4,934) | 0 |

Windows leaving the local band enter as: fp 0%, int2 8.5% (1,375), evicted 91.5% (14,753).

Did each move land on future attention? (next 32 steps, 59 events; lift = the moved window's future
attention over the mean of the windows that eviction decided on)

| move | count | share of candidates' future attention | lift |
|---|---|---|---|
| promote | 221 | 0.1% | **2.34** [2.10, 2.59] |
| stay fp | 29,987 | 11.1% | 1.63 [1.58, 1.67] |
| demote | 221 | 0.1% | 1.19 [1.09, 1.29] |
| stay int2 | 391,147 | 84.3% | 0.94 [0.94, 0.95] |
| evict from int2 | 1,336 | 0.2% | 0.66 [0.63, 0.70] |

| measure | value |
|---|---|
| swap gain (promote lift - demote lift) | +1.15 [0.92, 1.39] |
| promotions into the hindsight-best fp set | 36.7% [29.3, 43.8] (n = 221) |
| demotions out of it | 95.6% [92.8, 97.9] (n = 221) |
| evictions outside the hindsight-best retained set | 7.3% [6.6, 8.0] (n = 15,104) |
| fp precision (fp windows in the hindsight-best fp set) | 23.6% [21.7, 25.8] |
| eviction regret (candidates' future attention on evicted windows) | 0.2% |
| decision fidelity (fp set vs true ranking of the same survivors) | 94.5% [93.3, 95.7] |

---

# Part B. Promotion (int2 -> fp): the evidence

Promotion reuses the window's dequantized int2 payload (`quant_promote_source: dequant`). It does not
restore precision; it makes the window always read, by every query head, instead of read only when the
gate opens it, and frees a gate slot.

| evidence | value | source |
|---|---|---|
| promoted windows' future attention vs candidate average | **2.34x** [2.10, 2.59] | A.V |
| promoted vs displaced (demoted) window, next 8 steps | 1.03e-2 vs 3.87e-3 per step (**2.7x**) | §5 |
| promoted vs a typical fp / int2 window, next 8 steps | 2.2x (vs 4.76e-3) / 3.7x (vs 2.80e-3) | §4, §5 |
| promotions into the hindsight-best fp set | **36.7%** vs about 7% by chance (2 of ~29 candidates, arithmetic): 5.3x | A.V |
| the same rate for fp windows overall | 23.6%: promoted windows are 1.55x as precise | A.V |
| demotions that really leave the best fp set | 95.6% (76.4% for a random fp window) | A.V |
| swap gain | +1.15 [0.92, 1.39] | A.V |
| output-error cost of the dequantized payload | **0.004** per step (2% of eviction's 0.182); rung 0.199 -> 0.199 | §9 |
| extra memory | **0 bytes** (no fp16 copy is kept) | design |
| fp windows holding a promoted payload | 15.7% (26.0% in layers 8-15) | §5 |
| persistence | net promotions over 2 / 4 evictions are 1.84x / 3.26x the 1-eviction count (2x / 4x if none reverted; arithmetic) | A.V |

* **Frequency:** 0.014 promotions per eviction per layer (221 scored), because the fp tier has only 2
  slots at q = 0.70. Promoted windows carry 0.1% of the candidates' future attention.
* **Not shown by this run:** the effect on accuracy (needs the E2 one-way and E3 dequant / original / none
  runs in `scripts/run_recoverkv_all.sh`), the per-window gate-skip rate before promotion, and a one-way
  replay of the fp tier's attention.
* A setting with more fp slots exercises promotion more: q = 0.21 gives 7 fp + 7 int2 windows (Part D).

---

# Part C. Detailed tables (scripts/analyze_observation_zip.py)

## 1. Attention structure (ground truth)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| sink mass | 46.8% | 65.6% | 37.3% | 65.3% | 65.6% | 57.3% |
| windows for 90% of body mass (mean) | 38.8 | 31.2 | 32.1 | 38.6 | 37.9 | 35.4 |
| top window's share of body mass | 33.9% | 37.0% | 28.7% | 27.5% | 31.2% | 30.9% |

* **57% of all attention lands on the 5 sink tokens.** Layers 8-15 are the exception (37%), and they are where the most mass is later missed.
* **Attention outside the sinks is spread out.** 90% of it needs about 35 windows (out of 64-128), and the hottest window holds 31%.

## 2. Where the attention mass sits (measured)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| sink | 46.8% | 65.6% | 37.3% | 65.3% | 65.6% | 57.3% |
| local (fp) | 35.4% | 21.8% | 35.2% | 18.9% | 18.8% | 24.5% |
| fp (evictable) | 0.7% | 0.8% | 1.6% | 0.9% | 1.1% | 1.1% |
| int2 | 5.4% | 5.2% | 11.2% | 6.3% | 6.5% | 7.3% |
| evicted = missed | 11.7% | 6.6% | 14.8% | 8.7% | 8.1% | 9.9% |
| missed on windows holding generated tokens | 5.3% | 2.8% | 6.2% | 3.7% | 3.5% | 4.2% |
| missed, p90 over (step, head) | 31.6% | 18.6% | 35.0% | 22.1% | 21.1% | 26.0% |
| missed, p99 | 47.4% | 42.4% | 60.1% | 46.3% | 44.2% | 51.0% |
| share of (step, head) missing >10% | 43.2% | 21.2% | 51.5% | 29.7% | 28.1% | 34.0% |

* **The local band holds 24.5%, the 26 int2 windows 7.3%, and the 2 fp windows 1.1%; 9.9% is missed.** The int2 tier carries about 7x the mass of the fp tier.
* **The miss is heavy-tailed:** 34% of (step, head) pairs miss more than 10%, 1% miss more than 51%.
* **4.2 of the 9.9 points are on windows holding generated tokens** (§4). Layers 8-15 lose the most (14.8%), layers 2-7 the least (6.6%).

## 3. Missed mass: the cache against oracle policies

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| cache (measured) | 11.7% | 6.6% | 14.8% | 8.7% | 8.1% | 9.9% |
| exact cumulative ranker, same tiers | 11.8% | 6.6% | 15.0% | 8.8% | 8.2% | 10.0% |
| per-step best set (ceiling) | 8.1% | 4.3% | 7.2% | 4.7% | 4.7% | 5.5% |
| recency, same count | 8.9% | 7.5% | 17.3% | 9.9% | 9.7% | 11.2% |
| exact ranker, fp-only at same bytes | 15.8% | 10.2% | 23.3% | 13.3% | 12.8% | 15.2% |
| recency, fp-only at same bytes | 14.3% | 10.6% | 23.6% | 13.6% | 13.5% | 15.6% |
| exact ranker, age-normalised (mass per query that saw it) | 8.5% | 6.1% | 14.0% | 8.3% | 7.4% | 9.1% |

Tails (all layers): p90 of missed mass 26.0% (cache), 39.6% (exact ranker, fp-only, same bytes), 40.5%
(recency, fp-only, same bytes), 13.9% (per-step best); share of (step, head) missing >10%: 34.0%, 47.1%,
47.0%, 17.1%; p99: 51.0%, 69.9%, 74.7%, 29.9%.

* **At the same bytes, fp16 eviction misses 15.2% (exact scores) or 15.6% (recency) against the cache's 9.9%**, lower in every band.
* **The cache equals the exact cumulative ranker to 0.2 points in every band.** Its running score is not the source of the loss.
* **The same 28 windows re-chosen every step would miss 5.5%.** That 4.4-point gap is the cost of a frozen cumulative ranking and permanent eviction.
* **Recency at the same window count misses 11.2%,** 1.3 points more than the cache overall, but less in layers 0-1 (8.9% vs 11.7%).
* **Dividing each window's score by the number of queries that saw it misses 9.1%,** lower in every band and 2.3 points lower over the last 64 steps (§12).
* **The int2 tier recovers 5.2 points** at this budget: 15.2% (fp-only) vs 10.0% (with int2), same exact ranker.

## 4. Eviction decisions (steady state, evictions 2..)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| evicted window's rank under the exact ranker (0 = its first pick too) | 0.000 | 0.002 | 0.002 | 0.001 | 0.001 | 0.002 |
| evicted windows out-drawing the median kept int2 window (next 64 steps) | 95.9% | 55.1% | 52.6% | 56.8% | 51.3% | 56.5% |
| rank-signal vs exact cumulative mass, Spearman | 0.985 | 0.989 | 0.983 | 0.989 | 0.990 | 0.988 |
| kept set unchanged across an eviction (Jaccard) | 0.998 | 0.993 | 0.993 | 0.995 | 0.995 | 0.994 |
| evicted windows that hold generated tokens | 74.4% | 70.9% | 71.5% | 72.7% | 73.1% | 72.3% |
| generated-token windows kept (fp+int2) at the last step, of 28 | 1.12 | 3.33 | 2.94 | 2.19 | 1.95 | 2.46 |
| mean mass/step, next 8 steps: fp windows | 2.68e-03 | 3.61e-03 | 6.93e-03 | 3.93e-03 | 4.81e-03 | 4.76e-03 |
| mean mass/step, next 8 steps: int2 windows | 2.05e-03 | 1.99e-03 | 4.30e-03 | 2.40e-03 | 2.48e-03 | 2.80e-03 |
| mean mass/step, next 8 steps: evicted windows | 5.30e-03 | 2.53e-03 | 5.01e-03 | 2.97e-03 | 2.89e-03 | 3.52e-03 |
| mean mass/step, next 64 steps: fp windows | 2.63e-03 | 3.59e-03 | 6.49e-03 | 3.78e-03 | 4.80e-03 | 4.60e-03 |
| mean mass/step, next 64 steps: int2 windows | 1.98e-03 | 1.91e-03 | 4.10e-03 | 2.30e-03 | 2.39e-03 | 2.68e-03 |
| mean mass/step, next 64 steps: evicted windows | 4.04e-03 | 1.98e-03 | 4.24e-03 | 2.52e-03 | 2.38e-03 | 2.91e-03 |

* **The cache evicts the window the exact ranker would evict first** (mean rank 0.002 on a 0-1 scale).
* **After the first eviction the kept set is essentially frozen** (Jaccard 0.994); each eviction removes about one window per layer.
* **72% of evicted windows hold generated tokens;** at the last step only 2.5 of 28 kept windows are generated text.
* **Those windows are not cold:** over the next 64 steps they draw 2.9e-3 per step against 2.7e-3 for kept int2 windows, and 56.5% of them out-draw the median kept int2 window (96% in layers 0-1).
* **Why:** at the last eviction the weakest kept window's running score is about 3.4 (median over layers, 3 prompts), the evicted window's about 1.8 after 130-plus steps. Older, mostly prompt, windows win on accumulated exposure.
* This differs from A.V's eviction regret (0.2%), which scores only windows the eviction ranked; windows leaving the local band enter as evicted (91.5%) without being candidates there.

### 4b. The first eviction (prompt compression)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| kept set = exact prompt-score top set | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| kept set = the windows the whole decode attends most | 72.8% | 78.3% | 73.5% | 74.6% | 79.1% | 76.0% |
| kept set = the most recent windows | 36.4% | 40.5% | 40.7% | 38.7% | 37.7% | 39.1% |

* **The first eviction keeps exactly the prompt-attention top set.** 76% of it is among the 28 windows the whole 512-step decode attends most, against 39% for the most recent windows.

## 5. Tier moves per eviction per layer (measured)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| int2->fp (promote) | 0.01 | 0.01 | 0.02 | 0.02 | 0.01 | 0.01 |
| fp->int2 (demote) | 0.01 | 0.01 | 0.02 | 0.02 | 0.01 | 0.01 |
| int2->evicted | 0.03 | 0.11 | 0.11 | 0.08 | 0.07 | 0.09 |
| fp->evicted | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| local->fp | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| local->int2 | 0.03 | 0.11 | 0.11 | 0.08 | 0.07 | 0.09 |
| local->evicted | 0.97 | 0.89 | 0.89 | 0.92 | 0.93 | 0.91 |
| mass/step next 8 steps: promoted windows | 5.77e-03 | 3.63e-03 | 1.93e-02 | 5.72e-03 | 6.03e-03 | 1.03e-02 |
| mass/step next 8 steps: demoted windows | 2.65e-03 | 2.17e-03 | 4.81e-03 | 3.35e-03 | 4.61e-03 | 3.87e-03 |
| fp windows holding a dequantized payload | 7.9% | 11.5% | 26.0% | 16.8% | 9.5% | 15.7% |

* Moves are counted between consecutive evictions' post-states.
* **About 0.91 windows per layer per eviction go from local straight to evicted; 0.09 go into int2, displacing 0.09 int2 windows.**
* **Promotion and demotion are rare** (0.01 each), and promoted windows draw 2.7x the displaced ones (Part B).

## 6. Read gate: what the step actually read (measured)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| int2 tier's share of attention | 5.4% | 5.2% | 11.2% | 6.3% | 6.5% | 7.3% |
| int2 mass the gate read (recall), mean | 46.1% | 54.5% | 59.8% | 56.9% | 56.0% | 56.3% |
| recall, p10 over (step, head) | 33.3% | 38.2% | 41.0% | 40.1% | 39.6% | 39.0% |
| recall, p1 | 26.4% | 28.6% | 31.0% | 30.8% | 29.6% | 29.5% |
| heads with recall < 50% | 68.5% | 39.8% | 29.3% | 34.8% | 36.3% | 36.8% |
| head's hottest int2 window was read | 87.2% | 91.6% | 93.9% | 93.0% | 90.7% | 92.0% |
| oracle pick, same count, group share | 49.4% | 57.2% | 62.4% | 59.3% | 59.3% | 59.1% |
| oracle pick per head (no GQA sharing) | 55.5% | 63.8% | 67.8% | 65.1% | 65.3% | 65.0% |
| uniform pick (expected) | 26.9% | 26.9% | 26.9% | 26.9% | 26.9% | 26.9% |
| unread int2 mass (of all attention), mean | 3.1% | 2.3% | 4.1% | 2.3% | 2.6% | 2.9% |
| unread int2 mass, p99 | 15.5% | 16.9% | 18.9% | 12.8% | 15.2% | 16.5% |

* **The gate opens 27% of int2 windows and reads 56% of int2 mass** here (every head-step; 59.2% in A.IV's definition). It includes the head's hottest int2 window 92% of the time.
* **Against an exact-mass pick of the same 7 windows** it gets 56.3 of 59.1 points, about 95%. Sharing one pick across 4 query heads costs about 6 points (65.0% per head).
* **Unread int2 mass is 2.9% of all attention** (p99 16.5%), scored by the fill (§8). Layers 0-1 are weakest (46%).

## 7. Card ranking (E1, measured)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| Spearman(card estimate, true mass) over int2 windows | 0.857 | 0.911 | 0.914 | 0.917 | 0.905 | 0.908 |
| card top-7 vs true top-7 overlap | 75.9% | 81.1% | 81.8% | 82.1% | 80.4% | 81.0% |
| head's hottest int2 window in the card's top 7 | 93.5% | 96.7% | 97.3% | 97.0% | 95.4% | 96.4% |

* **A 272-byte card (34% of the 804 bytes/head it summarises) ranks int2 windows with Spearman 0.91;** its top 7 shares 81% with the true top 7 (27% for a random 7) and holds the hottest window 96% of the time.

## 8. Fill calibration: the cache's own window mass vs truth (int2 windows with >0.1% mass)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| read windows: log2(cache mass / true mass), median | 0.035 | 0.042 | 0.058 | 0.050 | 0.052 | 0.049 |
| read windows: abs(log2 ratio), median | 0.077 | 0.112 | 0.139 | 0.130 | 0.140 | 0.126 |
| skipped windows (fill): log2 ratio, median | -0.058 | -0.070 | -0.072 | -0.081 | -0.109 | -0.081 |
| skipped windows (fill): abs(log2 ratio), median | 0.262 | 0.314 | 0.356 | 0.345 | 0.386 | 0.345 |
| skipped windows: fill within 2x of truth | 93.6% | 89.4% | 87.7% | 88.3% | 83.9% | 87.6% |
| skipped set in total: log2(fill sum / true sum), median | -0.115 | -0.113 | -0.123 | -0.115 | -0.143 | -0.123 |
| read set in total: log2(cache sum / true sum), median | 0.041 | 0.051 | 0.072 | 0.059 | 0.062 | 0.059 |

* Ratios are normalised by the local band (exact keys), so held-set renormalisation cancels.
* **Read int2 windows come out about 3.5% high** (2^0.049). **The fill is within 2x of truth for 87.6%** of skipped windows (median error about 27%) and **under-credits the skipped set by about 8% in total** (2^-0.123), in every band.

## 9. Attention-output error ladder: rel. L2 vs the full-KV output (measured)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| error: kept_original, mean | 0.144 | 0.139 | 0.181 | 0.212 | 0.195 | 0.182 |
| error: kept_int2_dequant, mean | 0.156 | 0.152 | 0.202 | 0.227 | 0.214 | 0.199 |
| error: held, mean | 0.156 | 0.152 | 0.203 | 0.228 | 0.214 | 0.199 |
| error: actual, mean | 0.175 | 0.164 | 0.216 | 0.240 | 0.227 | 0.213 |
| error: actual, median | 0.108 | 0.098 | 0.175 | 0.198 | 0.180 | 0.162 |
| error: actual, p99 | 0.735 | 0.861 | 0.790 | 0.835 | 0.849 | 0.824 |
| step: eviction, mean | 0.144 | 0.139 | 0.181 | 0.212 | 0.195 | 0.182 |
| step: int2_quantization, mean | 0.054 | 0.052 | 0.076 | 0.071 | 0.074 | 0.068 |
| step: promotion_payload, mean | 0.002 | 0.002 | 0.007 | 0.004 | 0.003 | 0.004 |
| step: read_gate, mean | 0.092 | 0.044 | 0.052 | 0.050 | 0.057 | 0.054 |

* **Actual output error per head is 0.213** (median 0.162, p99 0.82); **eviction alone gives 0.182 (85%).**
* Steps are norms of consecutive differences, so they do not add up exactly. The read-gate step includes the fp16 kernel's own rounding, which this data cannot separate; it is largest in layers 0-1 (0.092), where recall is lowest.

## 10. int2 reconstruction error per window (measured at demotion)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| K rel. error, median | 0.098 | 0.112 | 0.123 | 0.120 | 0.118 | 0.118 |
| K rel. error, p90 | 0.134 | 0.138 | 0.148 | 0.143 | 0.144 | 0.144 |
| V rel. error, median | 0.614 | 0.586 | 0.570 | 0.530 | 0.524 | 0.550 |
| V rel. error, p90 | 0.874 | 0.710 | 0.663 | 0.635 | 0.643 | 0.677 |

* **K reconstructs with 11.8% median error, V with 55.0%.** The output impact of int2 storage is +0.017 on the mean error (0.182 -> 0.199) because the tier carries 7.3% of attention.

## 11. Granularity: exact cumulative ranker, same held tokens (oracle, head-mean mass)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| ws 4: 56 windows | 10.8% | 6.0% | 13.9% | 8.3% | 7.5% | 9.2% |
| ws 8: 28 windows | 11.8% | 6.6% | 15.0% | 8.8% | 8.2% | 10.0% |
| ws 16: 14 windows | 12.4% | 7.1% | 15.7% | 9.2% | 8.8% | 10.5% |
| ws 32: 7 windows | 13.2% | 7.7% | 16.6% | 9.7% | 9.4% | 11.2% |

* Finer windows miss slightly less at the same 224 held tokens. The ws 8 row reproduces §3's exact-ranker row (a consistency check). Token count only: smaller windows pay more grid and card bytes per token.

## 12. Over the decode (decode steps, all layers)

|  | 1-64 | 65-128 | 129-192 | 193-256 | 257-320 | 321-384 | 385-448 | 449-512 |
|---|---|---|---|---|---|---|---|---|
| sink | 59.5% | 58.8% | 58.8% | 58.2% | 56.8% | 56.1% | 54.5% | 55.4% |
| int2 | 9.7% | 9.0% | 7.3% | 6.6% | 7.1% | 6.4% | 6.1% | 6.1% |
| missed (cache) | 5.2% | 6.4% | 7.4% | 8.8% | 11.1% | 12.2% | 13.3% | 14.5% |
| missed (exact ranker) | 5.2% | 6.4% | 7.5% | 8.9% | 11.3% | 12.4% | 13.5% | 14.7% |
| missed on generated-token windows | 0.0% | 0.0% | 1.3% | 3.0% | 5.1% | 6.8% | 8.0% | 9.5% |
| missed (exact ranker, fp-only, same bytes) | 12.1% | 13.0% | 12.8% | 13.7% | 16.4% | 17.0% | 17.9% | 19.1% |
| missed (recency, fp-only, same bytes) | 13.5% | 14.1% | 13.4% | 13.8% | 16.4% | 16.8% | 17.5% | 18.9% |
| missed (age-normalised ranker) | 5.7% | 7.0% | 7.5% | 8.4% | 10.0% | 10.4% | 11.4% | 12.4% |
| gate recall | 56.6% | 55.7% | 55.6% | 55.2% | 56.4% | 56.7% | 56.5% | 57.5% |
| output error, actual | 0.160 | 0.175 | 0.178 | 0.200 | 0.233 | 0.241 | 0.249 | 0.266 |
| next-token entropy (nats) | 1.59 | 1.37 | 1.18 | 0.81 | 0.62 | 0.63 | 0.42 | 0.56 |

* **The cache is below both same-bytes fp16 baselines in every stretch** (5.2% vs 12.1% / 13.5% early; 14.5% vs 19.1% / 18.9% late); against the H2O-style baseline the gap narrows from 6.9 to 4.6 points.
* **Missed mass grows from 5.2% to 14.5%,** almost all of it on generated-text windows (0% -> 9.5%); output error follows (0.160 -> 0.266).
* **Gate recall is flat at 55-58%.** Next-token entropy falls from 1.59 to 0.56 nats as outputs become repetitive (§13); there is no full-KV generation to tell whether the cache causes this.

## 13. Per prompt

|  | sink | int2 | missed | missed (exact) | gate recall | out err | token logprob | entropy | top-1 p<0.5 | repeated 4-grams |
|---|---|---|---|---|---|---|---|---|---|---|
| sample 0 | 56.4% | 8.3% | 10.2% | 10.2% | 57.2% | 0.215 | -0.28 | 0.80 | 14.2% | 20.2% |
| sample 1 | 60.5% | 6.5% | 7.7% | 7.8% | 55.9% | 0.195 | -0.53 | 1.30 | 31.8% | 7.5% |
| sample 2 | 58.2% | 7.0% | 8.9% | 9.0% | 56.6% | 0.207 | -0.40 | 0.99 | 23.4% | 28.4% |
| sample 3 | 56.7% | 7.0% | 9.8% | 10.0% | 57.2% | 0.200 | -0.31 | 0.80 | 18.5% | 42.2% |
| sample 4 | 58.5% | 7.8% | 8.7% | 8.8% | 56.6% | 0.209 | -0.44 | 1.09 | 25.9% | 13.5% |
| sample 5 | 54.0% | 8.1% | 12.5% | 12.8% | 58.1% | 0.234 | -0.22 | 0.57 | 12.9% | 73.7% |
| sample 6 | 56.9% | 6.9% | 11.2% | 11.3% | 53.5% | 0.231 | -0.38 | 0.91 | 22.2% | 14.1% |
| sample 7 | 56.9% | 6.8% | 9.8% | 9.9% | 55.2% | 0.212 | -0.29 | 0.75 | 15.8% | 25.3% |

* **Prompts agree closely:** missed 7.7-12.5%, gate recall 53.5-58.1%, output error 0.195-0.234. Sample 5 is the outlier: 12.5% missed, the highest error, and 74% repeated 4-grams (a repetition loop).

## Appendix: per layer

|  | sink | local | fp | int2 | missed | missed (exact) | missed (per-step best) | gate recall | oracle recall | card Spearman | err: eviction | err: actual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 21.9% | 50.7% | 0.9% | 8.0% | 18.4% | 18.7% | 12.6% | 44.4% | 47.2% | 0.851 | 0.168 | 0.220 |
| L1 | 71.7% | 20.1% | 0.5% | 2.7% | 5.0% | 5.0% | 3.7% | 47.9% | 51.5% | 0.864 | 0.121 | 0.130 |
| L2 | 81.2% | 10.7% | 0.5% | 2.8% | 4.8% | 4.9% | 2.4% | 56.5% | 60.1% | 0.910 | 0.187 | 0.218 |
| L3 | 78.0% | 13.6% | 0.6% | 3.3% | 4.5% | 4.5% | 3.6% | 53.5% | 55.4% | 0.923 | 0.123 | 0.143 |
| L4 | 68.4% | 20.7% | 0.7% | 4.4% | 5.7% | 5.8% | 4.1% | 51.6% | 54.3% | 0.905 | 0.130 | 0.149 |
| L5 | 61.5% | 23.0% | 1.1% | 6.9% | 7.4% | 7.4% | 4.7% | 56.9% | 59.7% | 0.909 | 0.144 | 0.181 |
| L6 | 56.0% | 29.2% | 0.8% | 6.0% | 8.0% | 8.1% | 5.3% | 52.8% | 55.1% | 0.916 | 0.123 | 0.143 |
| L7 | 48.6% | 33.8% | 1.1% | 7.6% | 8.9% | 9.0% | 5.6% | 55.4% | 58.5% | 0.902 | 0.126 | 0.152 |
| L8 | 39.9% | 32.9% | 1.8% | 11.5% | 13.9% | 14.1% | 8.0% | 57.0% | 59.7% | 0.908 | 0.163 | 0.202 |
| L9 | 39.2% | 31.5% | 1.7% | 11.6% | 16.0% | 16.2% | 9.2% | 55.9% | 58.2% | 0.915 | 0.203 | 0.234 |
| L10 | 42.4% | 31.3% | 1.3% | 10.5% | 14.5% | 14.7% | 6.4% | 61.4% | 64.2% | 0.911 | 0.205 | 0.243 |
| L11 | 32.5% | 35.8% | 1.7% | 12.7% | 17.3% | 17.4% | 9.3% | 57.7% | 60.4% | 0.906 | 0.205 | 0.248 |
| L12 | 35.6% | 47.4% | 0.8% | 6.8% | 9.3% | 9.6% | 5.4% | 56.1% | 58.3% | 0.914 | 0.113 | 0.129 |
| L13 | 32.0% | 31.2% | 2.3% | 15.5% | 19.0% | 19.2% | 7.5% | 65.1% | 67.4% | 0.924 | 0.205 | 0.251 |
| L14 | 34.2% | 35.7% | 1.7% | 12.5% | 15.8% | 16.0% | 6.0% | 65.5% | 68.4% | 0.917 | 0.177 | 0.215 |
| L15 | 42.4% | 35.4% | 1.1% | 8.6% | 12.5% | 12.8% | 6.1% | 60.0% | 62.7% | 0.915 | 0.178 | 0.209 |
| L16 | 47.8% | 31.9% | 1.3% | 8.7% | 10.4% | 10.6% | 4.8% | 62.6% | 65.2% | 0.921 | 0.156 | 0.191 |
| L17 | 52.7% | 24.8% | 1.2% | 8.7% | 12.6% | 12.7% | 6.9% | 56.3% | 58.9% | 0.916 | 0.196 | 0.222 |
| L18 | 68.5% | 17.6% | 0.9% | 5.6% | 7.5% | 7.6% | 4.1% | 56.2% | 58.3% | 0.918 | 0.202 | 0.228 |
| L19 | 69.4% | 15.5% | 0.8% | 5.7% | 8.7% | 8.8% | 4.8% | 54.8% | 57.0% | 0.919 | 0.223 | 0.249 |
| L20 | 66.9% | 16.3% | 0.9% | 6.6% | 9.4% | 9.5% | 4.9% | 56.6% | 58.5% | 0.923 | 0.242 | 0.273 |
| L21 | 66.8% | 19.1% | 0.8% | 5.3% | 8.0% | 8.1% | 4.5% | 56.4% | 59.3% | 0.907 | 0.201 | 0.225 |
| L22 | 72.2% | 14.4% | 0.8% | 5.4% | 7.3% | 7.3% | 3.9% | 57.3% | 59.4% | 0.920 | 0.249 | 0.283 |
| L23 | 77.9% | 11.5% | 0.6% | 4.2% | 5.8% | 5.9% | 3.4% | 55.5% | 58.1% | 0.914 | 0.226 | 0.252 |
| L24 | 77.8% | 11.3% | 0.7% | 4.5% | 5.7% | 5.8% | 3.1% | 58.2% | 61.2% | 0.913 | 0.214 | 0.250 |
| L25 | 80.4% | 10.7% | 0.6% | 4.0% | 4.3% | 4.3% | 2.5% | 58.4% | 62.4% | 0.902 | 0.185 | 0.226 |
| L26 | 66.2% | 17.0% | 1.2% | 7.0% | 8.6% | 8.8% | 4.5% | 58.6% | 61.2% | 0.917 | 0.221 | 0.258 |
| L27 | 69.0% | 14.8% | 1.0% | 6.9% | 8.3% | 8.4% | 4.6% | 56.2% | 58.8% | 0.909 | 0.245 | 0.283 |
| L28 | 63.1% | 21.8% | 0.8% | 5.8% | 8.4% | 8.5% | 5.2% | 53.6% | 56.3% | 0.916 | 0.173 | 0.189 |
| L29 | 69.2% | 15.4% | 1.5% | 6.7% | 7.2% | 7.3% | 4.1% | 54.0% | 58.6% | 0.888 | 0.179 | 0.209 |
| L30 | 48.0% | 25.8% | 1.7% | 10.8% | 13.7% | 13.9% | 7.8% | 55.9% | 59.3% | 0.899 | 0.193 | 0.232 |
| L31 | 51.2% | 33.1% | 1.0% | 6.0% | 8.6% | 8.8% | 5.6% | 53.3% | 56.6% | 0.898 | 0.149 | 0.170 |

---

# Part D. A setting with equal fp and int2 counts: q = 0.21

Arithmetic from `WindowedCacheConfig.resolve()` (bytes mode) at this run's other settings; not run.

| quantity | value |
|---|---|
| evictable byte budget (71 tokens x 4,096 B) | 290,816 |
| one fp window + one int2 window | 32,768 + 8,608 = 41,376 |
| most equal pairs that fit | 7 (289,632 bytes) |
| q for equal counts | about 8,608 / 41,376 = 0.208 |

| `quant_ratio` | fp windows | int2 windows |
|---|---|---|
| below about 0.110 | 8 | 3 |
| **0.111 - 0.233 (use 0.21)** | **7** | **7** |
| above about 0.234 | 6 | 10 and up |
| 0.70 (this run) | 2 | 26 |

At q = 0.21: 14 evictable windows, int2 is 20.8% of the evictable bytes, and the gate opens ceil(0.25 x 7)
= 2 of 7 int2 windows per KV head (read fraction 0.286). Check before a run:

```bash
python -c "
from types import SimpleNamespace; import torch
from modules.windowed_cache.config import WindowedCacheConfig
mc = SimpleNamespace(num_key_value_heads=8, num_attention_heads=32, hidden_size=4096, head_dim=128)
for q in (0.12, 0.21, 0.23, 0.24, 0.70):
    r = WindowedCacheConfig(window_size=8, num_sink_tokens=5, local_window_size=128, cache_budget=0.20,
                            quant_ratio=q, quant_budget_mode='bytes').resolve(512, mc, torch.float16, 512)
    print(q, 'fp', r.top_k_fp, 'int2', r.N_q)"
```

---

# Scope

* One configuration, one corpus (wikitext continuation), 8 prompts. The data explains mechanisms; it is not
  an accuracy or reasoning score.
* Baselines (H2O-style, StreamingLLM-style, per-step best, age-normalised) are simulations on this run's own
  attention, not runs of those methods. int2 KIVI and QEvict have not been run.
* The output-error ladder is per head, relative L2 against the full-KV output of the same query; its last
  step includes the fp16 kernel's rounding.
* Not covered: the E2 one-way replay, the E3 payload arms, token-level output against a full-KV generation.

# Reproduce

```bash
# corpus: one wikitext article per line (the loader's own splitter)
python -c "import json,sys; from data.corpus_loader import CorpusLoader as C; a=C._split_into_articles(open(sys.argv[1],encoding='utf-8',errors='replace').read()); open(sys.argv[2],'w',encoding='utf-8').writelines(json.dumps({'text':x})+'\n' for x in a); print(len(a),'articles')" wikitext_test.txt wikitext_test_articles.jsonl
# collect (GPU) and move the zip off the machine
scripts/collect_observations.sh --model-path <llama-3.1-8b-instruct> --dataset wikitext_test_articles.jsonl \
    --window-size 8 --cache-budget 0.20 --gate-ratio 0.25 --quant-ratio 0.70 --num-samples 8
scripts/ship_observations.sh outputs/obs_data/wikitext_test_articles_ws8_b0.2_g0.25_q0.7_p512_n512.zip
# Part C tables
python scripts/analyze_observation_zip.py <zip> --out report.md --json report.json
# Part A (the paper's suite)
python -m modules.evaluation.observation_collector export --zip <zip> --out-dir parity
python -m modules.evaluation.qevict_observations --base-npz parity/parity_base.npz \
    --ours-npz parity/parity_ours.npz --output-dir parity/qevict
```

The byte-matched FMM rows in A.II apply `utils.qevict_metrics.future_missed_mass` and `selection_churn` to
the suite's `ObservationInputs`, with the fp16 baselines simulated by
`scripts/analyze_observation_zip.sim_evicted` (8 evictable windows + 16 local, the run's eviction steps,
exact cumulative scores or recency); that driver is not yet in the repository.

Observations page: `reports/recoverkv_observations_wikitext_ws8_b0.2_g0.25_q0.7.html`.
