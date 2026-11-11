# QEvict metrics, q = 0.70 vs q = 0.20 (wikitext, ws 8, budget 0.20, gate 0.25)

Two observation-collector runs that differ **only** in `quant_ratio`. Same model (Llama-3.1-8B-Instruct),
same 8 wikitext-103 prompts (identical article hashes), 512 prompt + 512 greedy decode tokens, same
20% byte budget, same local band (128 tokens) and sinks (5).

| run | fp windows | int2 windows | gate opens | read-gate verdict | data |
|---|---|---|---|---|---|
| q = 0.70 | 2 | 26 | 7 of 26 per KV head (0.269) | `gated` 131,072 / 131,072 | 1.88 GB, sha256 OK |
| q = 0.20 | 7 | 7 | 2 of 7 per KV head (0.286) | `gated` 131,072 / 131,072 | 1.55 GB, sha256 OK |

Both spend the same bytes (289,344 and 289,632 of 290,816 evictable bytes); an fp16-only cache at this budget
holds 8 evictable windows.

**How the numbers were made.** The paper suite (`modules.evaluation.qevict_observations`, Observations I-V)
on each run's parity export, and the tier study's seven metrics M1-M7 (`modules/evaluation/tier_study`) via
`scripts/tier_metrics_single_run.py`, which feeds one collector run into that package's own metric code.
R3 is the measured cache. **R2 (evict-only fp16 at the same bytes) and S2 (StreamingLLM-style recency at the
same bytes) are simulated** on each run's own ground-truth attention; R2 is given exact scores, which favours
it. R1 / R4 (other window sizes) need separate runs and are not here. Intervals are 95% bootstrap CIs over
256 (prompt, layer) traces (within-run variability).

---

## Research goal and verdict

`DISTANCE_TO_GOAL.md`: beat Flash FullKV, int2 KIVI and QEvict on decode speed, with the memory claim
(same bytes) and quality held. These runs measure the quality and memory side (what the cache keeps and
reads), not latency.

| observation | better | q = 0.70 | q = 0.20 | why it matters for the goal |
|---|---|---|---|---|
| M1 Future Missed Mass, H = 32 | **q 0.70** | **28.9%** | 40.5% | what each eviction costs the next 32 steps |
| M1 advantage over fp16 at the same bytes | **q 0.70** | **-17.0 pts** (vs 45.9%) | -5.9 pts (vs 46.4%) | the memory claim |
| missed attention per step (measured) | **q 0.70** | **9.9%** | 14.5% | direct quality loss |
| attention-output error (actual, mean) | **q 0.70** | **0.213** | 0.264 | what the next layer sees |
| M3 band importance held (fp + int2) | **q 0.70** | **53.8%** | 33.4% | vs 21.4% / 22.0% for fp16 at the same bytes |
| M4 alive-set Jaccard vs oracle | **q 0.70** | **0.884** | 0.826 | sustained selection quality |
| M6 importance recovered vs fp16, same bytes | **q 0.70** | **29.1 pts** | 10.3 pts | the case for int2 over fp16 |
| paper Obs IV gate recall | **q 0.70** | **59.2%** | 57.0% | gate efficiency equal (93.9% / 94.1%) |
| card ranking (Spearman) | **q 0.70** | **0.908** | 0.861 | |
| decode bytes read per layer-step (arithmetic) | **q 0.70** | **711,904** | 802,240 | 11% less traffic: the latency side |
| repeated 4-grams in the generation | **q 0.70** | **28.1%** | 48.8% | same prompts; more loops at q 0.20 |
| **promotion (Obs V): hits into the best fp set** | **q 0.20** | 36.7% | **82.2%** | the "why promote" argument |
| **promotion lift / demotion lift** | **q 0.20** | 2.34 / 1.19 | **1.92 / 0.79** | only q 0.20 meets promote > 1 > demote |
| **fp precision (fp windows in the best fp set)** | **q 0.20** | 23.6% | **69.7%** | |
| int2 quantization step in the output error | **q 0.20** | 0.068 | **0.029** | less int2 exposure |
| read-gate step in the output error | **q 0.20** | 0.054 | **0.025** | |
| M2 churn, evictable band only | **q 0.20** | 0.0059 | **0.0027** | both tiny |
| M5 int2-tier score fidelity (cosine) | tie | 0.9989 | 0.9991 | |
| M7 revival headroom (oracle LIR, m = 4) | q 0.20 has more | 0.55% | 2.13% | more for promotion to catch |

**Answer.** For the research goal (quality at equal memory, and decode traffic) **q = 0.70 stays the
better operating point**: its evictions lose 29% less of the next 32 steps' attention than q = 0.20's, it
beats same-bytes fp16 by 17 points of FMM where q = 0.20 beats it by 6, and it reads 11% fewer bytes per
step. **q = 0.20 is better on one observation family, promotion (Observation V / M7):** with 7 fp slots,
82% of promotions land in the hindsight-best fp set, promoted windows draw 1.92x the average while demoted
ones draw 0.79x, and the fp tier is 70% precise. That is the cleanest evidence the promotion path works;
q = 0.70 barely exercises it (2 fp slots).

---

## M1 Future Missed Mass

At each eviction: of the attention the next H steps put on windows that already existed, the share on
windows the eviction dropped (sinks excluded). Lower is better.

| horizon | policy | q = 0.70 | q = 0.20 |
|---|---|---|---|
| 8 | **RecoverKV fp + int2 (measured)** | **24.2%** [23.7, 24.7] | **34.4%** [33.6, 35.3] |
| 8 | RecoverKV fp tier only (measured) | 42.9% [42.1, 43.7] | 39.3% [38.4, 40.2] |
| 8 | R2 evict-only fp16, same bytes (sim) | 37.9% [37.1, 38.6] | 39.3% [38.4, 40.3] |
| 8 | S2 StreamingLLM-style, same bytes (sim) | 39.2% [38.3, 40.0] | 41.8% [40.7, 42.9] |
| 32 | **RecoverKV fp + int2 (measured)** | **28.9%** [28.4, 29.4] | **40.5%** [39.8, 41.2] |
| 32 | RecoverKV fp tier only (measured) | 52.3% [51.5, 53.1] | 46.4% [45.6, 47.2] |
| 32 | R2 evict-only fp16, same bytes (sim) | 45.9% [45.2, 46.6] | 46.4% [45.6, 47.2] |
| 32 | S2 StreamingLLM-style, same bytes (sim) | 47.9% [47.0, 48.7] | 49.6% [48.7, 50.5] |
| 64 | **RecoverKV fp + int2 (measured)** | **30.7%** [30.2, 31.2] | **43.4%** [42.8, 44.1] |
| 64 | RecoverKV fp tier only (measured) | 56.9% [56.2, 57.6] | 50.1% [49.4, 50.8] |
| 64 | R2 evict-only fp16, same bytes (sim) | 49.7% [49.0, 50.4] | 50.0% [49.3, 50.6] |
| 64 | S2 StreamingLLM-style, same bytes (sim) | 52.4% [51.6, 53.2] | 53.9% [53.0, 54.7] |

* Both settings beat fp16 at the same bytes at every horizon; q = 0.70 by 37% (H = 32), q = 0.20 by 13%.
* The measured values match the paper suite exactly (28.9%, 52.3%, 40.5%, 46.4%).

## M2 Selection Churn

Jaccard distance between the retained sets of consecutive evictions (0 = unchanged).

| set | q = 0.70 per flush | q = 0.20 per flush | q = 0.70 per step | q = 0.20 per step | q = 0.70 mass-weighted | q = 0.20 mass-weighted |
|---|---|---|---|---|---|---|
| RecoverKV fp + int2 | 0.0059 | 0.0027 | 0.00072 | 0.00033 | 0.0042 | 0.0022 |
| RecoverKV fp only | 0.0096 | 0.0128 | 0.00119 | 0.00158 | 0.0089 | 0.0105 |
| R2 evict-only (sim) | 0.0003 | 0.0015 | 0.00003 | 0.00018 | 0.0002 | 0.0013 |
| S2 StreamingLLM (sim) | 0.2222 | 0.2222 | 0.02734 | 0.02734 | 0.2230 | 0.2236 |

* **Definitions matter here.** M2's primary set is the evictable band only. On it, exact-score evict-only
  fp16 churns *less* than RecoverKV because it is frozen: every window leaving the local band is dropped,
  so its kept set never changes. RecoverKV admits some of those windows into int2 (9% at q 0.70).
* The paper suite's Observation II churn includes the local tail: RecoverKV 0.044 (q 0.70) / 0.065
  (q 0.20) against 0.080 for fp16 at the same bytes. Quote the definition with the number.

## M3 Tier mass distribution and boundary sharpness

Each eviction's band ranked by true cumulative importance and cut at the run's own tier sizes.

| quantity | RecoverKV q = 0.70 | RecoverKV q = 0.20 | fp16 same bytes (q 0.70 run) | fp16 same bytes (q 0.20 run) |
|---|---|---|---|---|
| band share in fp | 6.9% | 19.9% | 21.4% | 22.0% |
| band share in int2 | 46.9% | 13.5% | - | - |
| band share held (fp + int2) | **53.8%** | 33.4% | 21.4% | 22.0% |
| band share evicted | 46.2% | 66.6% | 78.6% | 78.0% |
| mean score ratio fp / int2 / evicted | 2.62 / 1.38 / 0.73 | 2.18 / 1.48 / 0.82 | 2.04 / - / 0.88 | 2.11 / - / 0.87 |
| fp -> int2 boundary sharpness | **20.5** | 6.8 | - | - |
| kept -> evicted boundary sharpness | 1.38 | 3.31 | 5.61 | 6.51 |
| kept -> evicted gap ratio | 1.016 | 1.028 | 1.040 | 1.043 |

* Three regimes are visible at both settings (fp > int2 > evicted per-window importance).
* q = 0.70's 2 fp slots sit on a cliff (sharpness 20.5): the top two windows are far above the rest.
* Its kept/evicted cut falls on a smooth stretch (1.38, about the median step): the first evicted window
  is nearly as important as the last kept one, the same finding as "evicted windows are not cold".

## M4 Alive-set Jaccard vs the same-size oracle (evictable band)

| set | q = 0.70 all | early | late | q = 0.20 all | early | late |
|---|---|---|---|---|---|---|
| RecoverKV fp + int2 | **0.884** [0.877, 0.890] | 0.928 | 0.840 | 0.826 [0.815, 0.837] | 0.889 | 0.763 |
| RecoverKV fp only | 0.945 | 0.971 | 0.919 | 0.892 | 0.929 | 0.855 |
| R2 evict-only (sim) | 0.807 | 0.861 | 0.753 | 0.775 | 0.842 | 0.709 |
| S2 StreamingLLM (sim) | 0.001 | 0.002 | 0.000 | 0.004 | 0.008 | 0.000 |

* The three-tier alive set tracks the oracle better than evict-only fp16 in both runs, and leads it on 60.0%
  (q 0.70) and 53.6% (q 0.20) of evictions. All sets decay from the early half to the late half.

## M5 int2-tier fidelity

Cosine between the cache's own cumulative score for its int2 windows (per head, at each eviction) and the
true cumulative mass of the same windows.

| | q = 0.70 | q = 0.20 |
|---|---|---|
| cosine, mean / median / p10 | 0.9989 / 0.9997 / 0.9972 | 0.9991 / 0.9998 / 0.9978 |
| true mass / cache's own mass, median | 0.989 | 0.983 |
| (eviction, layer, head) samples | 524,288 | 524,288 |

* The int2 tier's scores track the truth closely in both. The cumulative score includes the prompt's
  exact attention, which pushes cosine toward 1; read it as an upper bound on int2 scoring damage.

## M6 Recovered mass

(a) Simulated per the M6 definition: share of cumulative importance (sinks excluded) on evictable windows
outside the kept tiers. Fresh-K and Sticky-K give identical results in both runs (the cumulative ranking
barely changes between evictions).

| | q = 0.70 | q = 0.20 |
|---|---|---|
| outside the fp tier | 84.4% | 73.0% |
| outside fp and int2 | 42.2% | 60.7% |
| recovered by int2 (vs nothing) | 42.2 pts | 12.2 pts |
| outside the kept tier, fp16 at the same bytes | 71.3% | 71.0% |
| **recovered vs fp16 at the same bytes** | **29.1 pts** | **10.3 pts** |

(b) Paired future missed mass, H = 32: fp16 same bytes minus RecoverKV = **17.0 pts** (q 0.70) and 5.9 pts
(q 0.20); RecoverKV's fp tier alone minus fp + int2 = 23.4 pts and 5.9 pts.

## M7 Global LIR (uncapped) and transitions

Episodes: a window outside the selection for m consecutive evictions; revived if it re-enters anywhere later.

| selection | m | q = 0.70 rate (revived / episodes) | q = 0.20 rate (revived / episodes) |
|---|---|---|---|
| oracle top-k_fp | 1 | 0.64% (181 / 28,085) | 2.75% (753 / 27,377) |
| oracle top-k_fp | 4 | 0.55% (151 / 27,277) | 2.13% (561 / 26,400) |
| oracle top-k_fp | 8 | 0.47% (124 / 26,214) | 1.72% (435 / 25,228) |
| RecoverKV fp tier | 1 | 0.83% (233 / 28,137) | 2.65% (726 / 27,350) |
| RecoverKV fp tier | 4 | 0.69% (188 / 27,318) | 1.92% (505 / 26,346) |
| RecoverKV fp tier | 8 | 0.58% (153 / 26,249) | 1.46% (368 / 25,166) |

| lag-1 transition | q = 0.70 oracle / fp tier | q = 0.20 oracle / fp tier |
|---|---|---|
| cold -> hot (P01) | 0.015% / 0.019% | 0.065% / 0.063% |
| hot -> cold (P10) | 0.56% / 0.72% | 0.74% / 0.73% |

* With 7 fp slots the oracle's important set is larger and revives 4x more often; the real fp tier reaches
  90% of the oracle's rate at m = 4 (1.92% vs 2.13%). This is the headroom promotion exists for.

---

## The paper suite, side by side (Observations I-V)

| observation | measure | q = 0.70 | q = 0.20 |
|---|---|---|---|
| I | top 10% of windows hold | 22.4% | 23.1% |
| I | windows for 80% of importance | 64.9% | 64.2% |
| I | windows held: fp / fp + int2 (with local) | 19.6% / 47.9% | 25.1% / 32.7% |
| II | FMM fp only / fp + int2 (H 32) | 52.3% / 28.9% | 46.4% / 40.5% |
| II | FMM reduction from int2 | 44.8% | 12.8% |
| II | churn fp only / fp + int2 (tail-inclusive) | 0.107 / 0.044 | 0.087 / 0.065 |
| II | 16-token vs 8-token decisions (FMM) | 52.7% vs 52.3% | 48.5% vs 47.3% |
| III | oracle LIR (m 4, H 8) | 0.18% | 0.9% |
| IV | read exactly: sink / local / fp | 57.3% / 24.5% / 1.1% | 56.2% / 23.7% / 3.7% |
| IV | int2 opened / skipped (fill) | 4.4% / 2.9% | 1.1% / 0.9% |
| IV | evicted (hard missed) / decode missed | 9.9% / 12.7% | 14.5% / 15.4% |
| IV | gate recall / hindsight / efficiency | 59.2% / 63.1% / 93.9% | 57.0% / 60.5% / 94.1% |
| V | promotions, lag 1 (count) | 233 | 726 |
| V | lift: promote / stay fp / demote / stay int2 / evict from int2 | 2.34 / 1.63 / 1.19 / 0.94 / 0.66 | 1.92 / 1.27 / 0.79 / 0.72 / 0.53 |
| V | swap gain | 1.15 [0.92, 1.39] | 1.12 [0.99, 1.27] |
| V | promotions into / demotions out of the best fp set | 36.7% / 95.6% | 82.2% / 75.6% |
| V | fp precision | 23.6% | 69.7% |
| V | evictions outside the best retained set | 7.3% | 15.6% |
| V | eviction regret | 0.2% | 0.1% |
| V | decision fidelity | 94.5% | 93.8% |
| V | windows leaving local: to fp / int2 / evicted | 0% / 8.5% / 91.5% | 0.6% / 1.4% / 98.0% |

## Other measured differences (scripts/analyze_observation_zip.py)

| measure | q = 0.70 | q = 0.20 |
|---|---|---|
| missed attention, mean / p90 / steps x heads >10% | 9.9% / 26.0% / 34.0% | 14.5% / 37.1% / 46.7% |
| missed, exact ranker at the same tiers | 10.0% | 14.6% |
| missed, per-step best set of the same size (ceiling) | 5.5% | 10.3% |
| missed, fp16 same bytes: exact ranker / recency | 15.2% / 15.6% | 16.5% / 17.3% |
| missed over the decode, steps 1-64 -> 449-512 | 5.2% -> 14.5% | 9.3% -> 18.4% |
| output error: eviction / + int2 / + promotion / actual | 0.182 / 0.199 / 0.199 / 0.213 | 0.256 / 0.259 / 0.260 / 0.264 |
| output error steps: int2 / promotion / read gate | 0.068 / 0.004 / 0.054 | 0.029 / 0.014 / 0.025 |
| int2 tier's share of attention / unread int2 | 7.3% / 2.9% | 1.9% / 0.9% |
| head's hottest int2 window opened | 92.0% | 78.2% |
| card Spearman / hottest window in the card's top 7 | 0.908 / 96.4% | 0.861 / 89.3% |
| first eviction: kept set = decode's most-attended windows | 76.0% | 62.6% |
| promotions / demotions per eviction per layer | 0.01 / 0.01 | 0.05 / 0.05 |
| next-token entropy (mean over prompts) | 0.90 nats | 0.68 nats |
| repeated 4-grams (mean over prompts) | 28.1% | 48.8% |

* Prompt by prompt, q = 0.20 misses more attention in all 8 and has higher output error in all 8. Two of
  its generations are near-total loops (96.5% and 99.4% repeated 4-grams, vs 28.4% and 25.3% at q 0.70).
  Without a full-KV generation this is not proof the cache caused them.

## Decode bytes read per layer per step (arithmetic, not measured)

The fused kernel reads every fp window, the int2 windows the gate opens (codes + grid), every card, and the
sinks + local band.

| component | q = 0.70 | q = 0.20 |
|---|---|---|
| fp windows (32,768 B each) | 2 -> 65,536 | 7 -> 229,376 |
| opened int2 windows (6,432 B each) | 7 -> 45,024 | 2 -> 12,864 |
| cards (2,176 B each) | 26 -> 56,576 | 7 -> 15,232 |
| sinks + local (133 tokens) | 544,768 | 544,768 |
| **total** | **711,904** | **802,240** |

q = 0.70 reads 11% fewer bytes per layer per step. Not a latency measurement: `CLAUDE.md` records that
decode barely tracks bytes read at these shapes.

---

## Scope

* One corpus (wikitext continuation), 8 prompts, two settings. Mechanism, not accuracy.
* R2 and S2 are simulations on each run's attention, not recorded runs; R1 / R4 (other window sizes) are
  absent, so M1/M2's granularity contrasts are not produced. int2 KIVI and QEvict have not been run.
* M5's cosine uses cumulative scores that include the prompt's exact attention (an upper bound on damage).
* M6(a) is a share of cumulative importance (sinks excluded), not per-step attention; compare it only
  within this table.

## Reproduce

```bash
python -m modules.evaluation.observation_collector export --zip <run>.zip --out-dir parity
python -m modules.evaluation.qevict_observations --base-npz parity/parity_base.npz \
    --ours-npz parity/parity_ours.npz --output-dir parity/qevict
python scripts/tier_metrics_single_run.py --base-npz parity/parity_base.npz \
    --ours-npz parity/parity_ours.npz --zip <run>.zip --label q0.70 --out-json m1_m7.json
python scripts/analyze_observation_zip.py <run>.zip --out tables.md --json tables.json
```

Data: `obs-data/wikitext_test_articles_ws8_b0.2_g0.25_q0.7_p512_n512` and
`obs-data/wikitext_test_articles_ws8_b0.2_g0.25_q0.2_p512_n512`.
Single-run detail for q = 0.70: `reports/observations_wikitext_ws8_b0.2_g0.25_q0.7.md`.
