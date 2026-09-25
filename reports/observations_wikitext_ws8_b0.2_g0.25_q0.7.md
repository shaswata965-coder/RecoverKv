# Observation run: wikitext, ws 8, budget 0.20, gate 0.25, q 0.70

**Data:** `wikitext_test_articles_ws8_b0.2_g0.25_q0.7_p512_n512.zip` (1.88 GB, sha256 verified after transfer).
Llama-3.1-8B-Instruct, 8 wikitext-103 test articles, 512-token prompt + 512 greedy decode steps each,
every step x 32 layers x 32 heads. Tables: `python scripts/analyze_observation_zip.py <zip> --out <md>`.

**Every number is one of three kinds.**
* **Measured:** read off the cache run: tiers, gate picks, card estimates, the cache's own mass, the error ladder.
* **Oracle:** the same run's full-KV attention replayed through a policy that sees exact mass. These use the cache's trajectory, i.e. the same queries.
* **Ground truth:** full-KV attention of the same query, from the collector's shadow KV.

Mass is a fraction of the query's full-KV attention (sinks included). Band columns average layers; `all` averages all 32.

## Run integrity

| check | value |
|---|---|
| read-gate verdict | `gated`: 131,072 / 131,072 fused layers (8 prompts x 512 steps x 32 layers) |
| realised read fraction | 0.269 = 7 of 26 int2 windows per KV head, every step |
| geometry (resolved) | 5 sinks + 16 local + 2 fp + 26 int2 windows; fp-only at the same bytes = 8 windows |
| occupancy | 26 int2 windows at every decode step, every layer; 64 evictions (steps 1, 9, ..., 505) |
| mass accounting | sink + all windows = 1 within 4e-4 at every step |

## Summary

1. **The cache misses 9.9% of attention** (p90 26%, p99 51% per step and head). The loss grows through the decode, from 5.2% to 14.5%, and the output error grows with it, from 0.160 to 0.266.
2. **Its evictions are exactly what an exact-mass cumulative ranker would pick:** 9.9% vs 10.0% missed, and its ranking agrees with the exact one (Spearman 0.988). **The score is accurate; the rule is what loses mass.** Keeping the best 28 windows afresh at each step would miss 5.5%.
3. **The rule evicts generated text as soon as it leaves the local band.** 72% of steady-state evictions are generated-text windows, and only 2.5 of the 28 kept windows are generated text at the end. Those windows are not cold: over the next 64 steps they draw as much attention as the kept int2 windows (2.9e-3 vs 2.7e-3 per step).
4. **The int2 tier pays for itself.** It holds 7.3% of attention. With exact ranking, an fp-only cache at the same bytes would miss 15.2% instead of 10.0% (oracle). The gate opens 27% of int2 windows and gets 56% of their mass; an exact pick of the same count would get 59%. It includes the head's hottest window 92% of the time.
5. **Most of the output error is eviction.** Removing the evicted windows alone gives 0.182 of the final 0.213. int2 quantization adds a step of 0.068; V reconstructs with 55% error. The read gate adds 0.054 and the promotion payload 0.004.

## 1. Attention structure (ground truth)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| sink mass | 46.8% | 65.6% | 37.3% | 65.3% | 65.6% | 57.3% |
| windows for 90% of body mass (mean) | 38.8 | 31.2 | 32.1 | 38.6 | 37.9 | 35.4 |
| top window's share of body mass | 33.9% | 37.0% | 28.7% | 27.5% | 31.2% | 30.9% |

* **57% of all attention lands on the 5 sink tokens.** Layers 8-15 are the exception (37%), and they are where the most mass is later missed.
* **Attention outside the sinks is spread out.** 90% of it needs about 35 windows (out of 64-128), and the hottest window holds 31%. A 28-window evictable budget cannot hold it all, which sets the scale of the losses below.

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
* **The miss is heavy-tailed.** The mean is 9.9%, but 34% of (step, head) pairs miss more than 10%, and 1% miss more than 51%.
* **4.2 of the 9.9 points are on windows holding generated tokens** (see §4).
* **Layers 8-15 lose the most (14.8%); layers 2-7 the least (6.6%).**

## 3. Missed mass: the cache against oracle policies

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| cache (measured) | 11.7% | 6.6% | 14.8% | 8.7% | 8.1% | 9.9% |
| exact cumulative ranker, same tiers | 11.8% | 6.6% | 15.0% | 8.8% | 8.2% | 10.0% |
| per-step best set (ceiling) | 8.1% | 4.3% | 7.2% | 4.7% | 4.7% | 5.5% |
| recency, same count | 8.9% | 7.5% | 17.3% | 9.9% | 9.7% | 11.2% |
| exact ranker, fp-only at same bytes | 15.8% | 10.2% | 23.3% | 13.3% | 12.8% | 15.2% |
| exact ranker, age-normalised (mass per query that saw it) | 8.5% | 6.1% | 14.0% | 8.3% | 7.4% | 9.1% |

* **The cache equals the exact cumulative ranker to 0.1 points in every band.** Its running score is not the source of the loss.
* **The same 28 windows re-chosen every step would miss 5.5%.** That is 4.4 points less: the cost of a frozen cumulative ranking and permanent eviction, not of score noise.
* **Recency alone misses 11.2%.** That is only 1.3 points worse than the cache overall, and better in layers 0-1 (8.9% vs 11.7%).
* **Dividing each window's score by the number of queries that saw it misses 9.1%.** That is better in every band, and 2.3 points better over the last 64 steps (§12).
* **fp-only at the same bytes misses 15.2%, against 10.0% for the same exact ranker with the int2 tier.** The int2 tier recovers 5.2 points of attention at this budget (oracle against oracle).

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
* **After the first eviction the kept set is essentially frozen** (Jaccard 0.994 between evictions). Each eviction removes about one window per layer.
* **72% of evicted windows hold generated tokens.** At the last step only 2.5 of 28 kept windows are generated text.
* **Those evicted windows are not cold.** Over the next 64 steps they draw 2.9e-3 per step against 2.7e-3 for kept int2 windows, and 56.5% of them out-draw the median kept int2 window (96% in layers 0-1).
* **Why:** at the last eviction the weakest kept window's running score is about 3.4 (median over layers, 3 prompts). The window evicted there has about 1.8 after 130-plus steps. Older, mostly prompt, windows win on accumulated exposure.

### 4b. The first eviction (prompt compression)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| kept set = exact prompt-score top set | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| kept set = the windows the whole decode attends most | 72.8% | 78.3% | 73.5% | 74.6% | 79.1% | 76.0% |
| kept set = the most recent windows | 36.4% | 40.5% | 40.7% | 38.7% | 37.7% | 39.1% |

* **The first eviction keeps exactly the prompt-attention top set (100%).** Before any eviction the cache's score is the full prompt attention.
* **76% of that set matches the 28 windows the whole 512-step decode attends most.** Only 39% matches the most recent windows.

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
* **About 0.91 windows per layer per eviction go straight from local to evicted; 0.09 go into int2, displacing 0.09 int2 windows.**
* **Promotion and demotion are rare** (0.01 each per eviction per layer). The two-way promotion path is barely exercised in this run.
* **When a promotion happens it is right:** promoted windows draw 1.0e-2 per step over the next 8 steps, against 3.9e-3 for the demoted ones (2.7x).
* **15.7% of fp windows hold a dequantized (promoted) payload**, and it costs little: the promotion step is 0.004 in §9.

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

* **The gate opens 27% of int2 windows and reads 56% of int2 mass** (uniform would get 27%). It includes the head's hottest int2 window 92% of the time.
* **Against an exact-mass pick of the same 7 windows** (same group-share rule), it gets 56.3 of 59.1 points, about 95%.
* **Sharing one pick across the 4 query heads of a KV head costs about 6 points:** a per-head exact pick would get 65.0%.
* **Unread int2 mass is 2.9% of all attention on average** (p99 16.5%). That mass is scored by the fill (§8), not read.
* **Layers 0-1 are weakest:** 46% recall, and 68.5% of heads under 50%.

## 7. Card ranking (E1, measured)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| Spearman(card estimate, true mass) over int2 windows | 0.857 | 0.911 | 0.914 | 0.917 | 0.905 | 0.908 |
| card top-7 vs true top-7 overlap | 75.9% | 81.1% | 81.8% | 82.1% | 80.4% | 81.0% |
| head's hottest int2 window in the card's top 7 | 93.5% | 96.7% | 97.3% | 97.0% | 95.4% | 96.4% |

* **The card's estimate ranks the int2 windows with Spearman 0.91 against true mass.** Its top 7 shares 81% with the true top 7, and holds the hottest window 96% of the time.
* **Layers 0-1 are weakest** (0.86 / 76% / 93.5%), matching their lower gate recall.

## 8. Fill calibration: the cache's own window mass vs truth

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
* **Read int2 windows come out about 3.5% high** (median ratio 2^0.049). Dequantized keys shift mass only slightly.
* **The fill for skipped windows is within 2x of truth for 87.6% of windows** that carry more than 0.1% mass; the median error is about 27% (abs log2 ratio 0.345).
* **In total the fill under-credits the skipped set by about 8%** (2^-0.123), in every band. That lowers the running score of int2 windows the gate skips, relative to read ones.

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

* **Actual output error per head is 0.213 on average** (median 0.162, p99 0.82).
* **Eviction alone gives 0.182** (kept tokens at original values). int2 quantization adds a step of 0.068, the promotion payload 0.004, and the read gate 0.054.
* Steps are norms of consecutive differences, so they do not add up exactly.
* **The read-gate step also includes the fp16 kernel's own rounding**, which this data cannot separate. It is largest in layers 0-1 (0.092), where recall is lowest.
* **Layers 16-31 carry the largest eviction error** (0.195-0.212), despite missing less mass than layers 8-15.

## 10. int2 reconstruction error per window (measured at demotion)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| K rel. error, median | 0.098 | 0.112 | 0.123 | 0.120 | 0.118 | 0.118 |
| K rel. error, p90 | 0.134 | 0.138 | 0.148 | 0.143 | 0.144 | 0.144 |
| V rel. error, median | 0.614 | 0.586 | 0.570 | 0.530 | 0.524 | 0.550 |
| V rel. error, p90 | 0.874 | 0.710 | 0.663 | 0.635 | 0.643 | 0.677 |

* **K reconstructs with 11.8% median relative error; V with 55.0% (p90 67.7%).** int2 V is very lossy.
* The output impact is moderate (quantization step 0.068) because the int2 tier carries only 7.3% of attention.

## 11. Granularity: exact cumulative ranker, same held tokens (oracle, head-mean mass)

|  | L0-1 | L2-7 | L8-15 | L16-23 | L24-31 | all |
|---|---|---|---|---|---|---|
| ws 4: 56 windows | 10.8% | 6.0% | 13.9% | 8.3% | 7.5% | 9.2% |
| ws 8: 28 windows | 11.8% | 6.6% | 15.0% | 8.8% | 8.2% | 10.0% |
| ws 16: 14 windows | 12.4% | 7.1% | 15.7% | 9.2% | 8.8% | 10.5% |
| ws 32: 7 windows | 13.2% | 7.7% | 16.6% | 9.7% | 9.4% | 11.2% |

* **At the same 224 held tokens, finer windows miss less:** 9.2% at ws 4, 10.0% at ws 8, 10.5% at ws 16, 11.2% at ws 32. The ws 8 row reproduces §3's exact-ranker row, a consistency check.
* **Token count only:** a smaller window pays more grid and card bytes per token, so this is not a same-bytes comparison.

## 12. Over the decode (decode steps, all layers)

|  | 1-64 | 65-128 | 129-192 | 193-256 | 257-320 | 321-384 | 385-448 | 449-512 |
|---|---|---|---|---|---|---|---|---|
| sink | 59.5% | 58.8% | 58.8% | 58.2% | 56.8% | 56.1% | 54.5% | 55.4% |
| int2 | 9.7% | 9.0% | 7.3% | 6.6% | 7.1% | 6.4% | 6.1% | 6.1% |
| missed (cache) | 5.2% | 6.4% | 7.4% | 8.8% | 11.1% | 12.2% | 13.3% | 14.5% |
| missed (exact ranker) | 5.2% | 6.4% | 7.5% | 8.9% | 11.3% | 12.4% | 13.5% | 14.7% |
| missed on generated-token windows | 0.0% | 0.0% | 1.3% | 3.0% | 5.1% | 6.8% | 8.0% | 9.5% |
| missed (age-normalised ranker) | 5.7% | 7.0% | 7.5% | 8.4% | 10.0% | 10.4% | 11.4% | 12.4% |
| gate recall | 56.6% | 55.7% | 55.6% | 55.2% | 56.4% | 56.7% | 56.5% | 57.5% |
| output error, actual | 0.160 | 0.175 | 0.178 | 0.200 | 0.233 | 0.241 | 0.249 | 0.266 |
| next-token entropy (nats) | 1.59 | 1.37 | 1.18 | 0.81 | 0.62 | 0.63 | 0.42 | 0.56 |

* **Missed mass grows steadily from 5.2% to 14.5%**, and generated-text windows account for nearly all of the growth (0% to 9.5%). Output error follows it (0.160 to 0.266).
* **The age-normalised ranker's advantage grows over the decode:** 12.4% vs 14.7% missed in the last 64 steps. It is slightly worse over the first 64 steps (5.7% vs 5.2%).
* **Gate recall is flat at 55-58%.** The gate does not degrade with length.
* **Next-token entropy falls from 1.59 to 0.56 nats** as outputs become repetitive (§13). There is no full-KV generation to tell whether the cache causes this.

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

* **Prompts agree closely:** missed 7.7-12.5%, gate recall 53.5-58.1%, output error 0.195-0.234.
* **Sample 5 is the outlier:** the most mass missed (12.5%), the highest error, and 74% repeated 4-grams (a repetition loop).

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

## Not covered here

* The E2 one-way counterfactual (a replay of `rank_signal` with int2 windows blocked from fp).
* Observations I-V in the parity suite's own format. `python -m modules.evaluation.observation_collector export --zip <zip> --out-dir <dir>` produces that input.
* Any accuracy score: this run records attention and outputs, not task answers.
