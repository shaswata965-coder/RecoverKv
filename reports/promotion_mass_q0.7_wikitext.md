# Q → F promotion in attention mass: the paper's Table 18 on the q = 0.70 run

**Data:** `wikitext_test_articles_ws8_b0.2_g0.25_q0.7_p512_n512.zip` (sha256 verified), the
run behind `reports/observations_wikitext_ws8_b0.2_g0.25_q0.7.md` and
`reports/qevict_metrics_q0.7_vs_q0.2_wikitext.md`. Llama-3.1-8B-Instruct, 8 wikitext-103
articles, 512-token prompt + 512 greedy decode steps, window 8, 5 sinks, local 128, 20% byte
budget: **2 fp + 26 int2 windows**, the read gate opens 7 of 26 per KV head. Read-gate verdict
`gated` (131,072 / 131,072 fused layer-steps). For context, the q = 0.20 run (7 fp + 7 int2,
gate opens 2 of 7), same prompts.

Produced by `python -m modules.evaluation.promotion_ablation` (this commit); machine-readable
output in `reports/promotion_ablation/q0.7.json` and `q0.2.json`.

## How the two columns are made, and why they are paired

Table 18 of the paper compares two pilot runs on **different source articles** and calls
the result directional. Here both columns are the **same 8 prompts, the same queries, the
same bytes**:

* **With promotion — measured.** The recorded run (shipped `quant_promotion: bidir`).
* **No promotion — one-way replay.** At each of the run's 64 evictions (× 32 layers × 8
  prompts), the one-way policy (`quant_promotion: oneway`, `F → Q → E`, the rule
  `policy.compute_two_tier_retain` implements) is replayed on the run's own `rank_signal`
  (exactly what each eviction ranked on) at the same tier sizes. The windows it holds in int2
  cannot take an fp slot.

Both are scored against the same ground truth, the same queries over the never-evicted KV
(`full_mass`), by the observation suite's own code (`qevict_observations`,
`utils.qevict_metrics`). Deltas are paired per (prompt, layer) trace; intervals are 95%
percentile bootstraps over the 256 traces (within-run variability, not population CIs).

The replay is checked against the run before anything is scored:

| check | q = 0.70 | q = 0.20 |
|---|---|---|
| replaying `bidir` reproduces the recorded tiers | **16,384 / 16,384** layer-evictions | 16,384 / 16,384 |
| gate rule re-run on the recorded cards reproduces the recorded picks | 1,044,545 / 1,048,576 (99.6%) | 1,047,576 / 1,048,576 (99.9%) |
| one-way: kept set (fp + int2) differs from the run | **0** / 16,384 | 0 / 16,384 |
| one-way: fp set differs from the run | 5,014 / 16,384 (30.6%) | 11,449 / 16,384 (69.9%) |
| one-way: windows needing a score the run never recorded | 0 | 0 |

The misses in the gate check are fp16 rounding of the stored card estimates.

**What is replayed, not measured.** (1) A window keeps the score the run gave it. Under
one-way, a window the run promoted would have stayed int2 and scored through its gated read
or fill instead of an fp read. (2) Where the one-way int2 tier differs, the gate is re-run by
its own rule (`sketch.group_share` → top-`n_sel` per KV head) on the recorded card estimates.
A window the run had promoted has no card estimate at those steps, so it is bracketed:

* **`open`** (primary): the gate reads it for every KV head, i.e. never misses a window
  promotion would have made permanent. **This favours no promotion**, so every gate-dependent
  delta below is a lower bound.
* **`closed`**: the gate never reads it. This favours promotion and gives an upper bound.

A collector run with `--promotion oneway` removes both assumptions (see *Next GPU run*).

---

## Table 18, on this run (q = 0.70)

| metric | no promotion (one-way replay) | with promotion (measured) | Δ [95% CI] |
|---|---|---|---|
| R3 FMM (fp + int2, next 32 steps) ↓ | 28.89% | 28.89% | **0.00 pp** (identical kept set) |
| Quantized-score agreement (Spearman, int2 tier) ↑ | 0.9830 | 0.9830 | −0.0001 [−0.0001, −0.0000] |
| R_Q: int2-tier FullKV mass in gate-opened windows ↑ | 59.07% | 59.24% | **+0.17 pp** [+0.12, +0.21] |
| Global LIR, fp tier (m = 4, uncapped) ↑ | 0.00% (0 / 27,136) | 0.69% (188 / 27,318) | **+0.69 pp** [+0.58, +0.80] |
| Recorded Q → F transitions (lag 1) | 0 | 233 | +233 |

The same run, in attention mass (share of each query head's FullKV attention per decode step):

| mass row | no promotion | with promotion | Δ [95% CI] | relative |
|---|---|---|---|---|
| on the fp tier ↑ | 1.00% | 1.07% | **+0.064 pp** [+0.048, +0.081] | +6.4% |
| on the fp tier, share of window mass (sinks excluded) ↑ | 2.45% | 2.58% | +0.13 pp [+0.10, +0.16] | +5.4% |
| FMM of the tiers read every step (fp + local), next 32 ↓ | 52.46% | 52.32% | −0.14 pp [−0.18, −0.11] | −0.3% |
| read by the decode step (sinks + local + fp + opened int2) ↑ | 87.24% | 87.27% | +0.029 pp [+0.023, +0.035] | |
| reached only through the card fill ↓ | 2.90% | 2.87% | **−0.029 pp** [−0.035, −0.023] | −1.0% |
| R_F+Q: held-tier mass read, (fp + opened) / (fp + int2) ↑ | 64.20% | 64.49% | +0.29 pp [+0.24, +0.35] | |

With the gate bracket reversed (`closed`: the replayed gate never reads a blocked window),
only the gate-dependent rows move. Fill-only mass goes 2.97% → 2.87% (−0.099 pp
[−0.120, −0.079], −3.3%), R_Q 58.41% → 59.24% (+0.84 pp [+0.69, +1.00]), and R_F+Q
63.47% → 64.49% (+1.02 pp [+0.84, +1.21]). The true effect lies between the two brackets and
sits near `open`, because the gate already opens a head's hottest int2 window 92.0% of the time
(`observations_wikitext_…_q0.7.md` §6).

### What each promotion carried

The 221 promotions with a full 32-step horizon, each against the fp window demoted at the same
eviction. Under one-way the demoted window would have kept that fp slot, so this pair is the
per-decision counterfactual:

| window | count | FullKV mass per step, next 32 steps |
|---|---|---|
| promoted Q → F | 221 | **0.78%** |
| demoted F → Q at the same evictions | 221 | 0.39% |

**Each promotion puts a window with 2.0× the future attention into the fp slot** (per-trace
gain +0.44 pp [+0.34, +0.55], 126 of 256 traces promote at least once). Where promotion acted,
i.e. the 30.6% of (step, layer) cells whose fp set differs between the arms, the fp tier holds
**1.21%** of attention with promotion against 1.00% without (+21%).

---

## Reading it

**The benefit is real and one-directional.** Every mass row moves in promotion's favour, and
every interval excludes zero, under the bracket that favours no promotion. Promotion is what
lets the fp tier follow attention (LIR 0 → 0.69%). Without it, the fp tier stays on the first
eviction's top two windows for the whole decode: the replayed fp set equals that pair at all
16,128 later layer-evictions of all 8 prompts.

**At q = 0.70 it is small in aggregate.** Three reasons, each visible above:

1. **Promotion reorders the kept set; it does not change it.** The one-way replay keeps
   exactly the run's fp + int2 windows at every eviction, so R3 FMM is identical (Table 18's
   pilot: −0.7 pp). What promotion changes is which of the kept windows is read on every step.
2. **The tier it feeds is two windows.** They hold 1.07% of attention, so a +21% gain where
   promotion acts is +0.06 pp overall.
3. **The gate already reads what promotion would pin.** The promoted window is usually a head's
   hottest int2 window, which the gate opens 92% of the time without promotion. The `open`
   bracket gives it 100%.

**The paper's Table 18 magnitudes do not reproduce on this cache and should not be quoted for
it.** R_Q is +0.17 to +0.84 pp here against the pilot's +6.8 pp; R3 FMM is 0.0 pp against
−0.7 pp; QSA is −0.0001 against +0.038. The definitions differ (next section), and so does the
setting: the pilot used a 256 / 1024 prompt / decode split on different articles per arm.

**Quantized-score agreement cannot show the paper's effect in a replay.** Under assumption (1)
both arms carry the same scores, so QSA moves only through *which* windows sit in int2. At
q = 0.20 that makes it go the other way (−0.0135): promotion removes the top of the int2 tier,
and a Spearman over a narrower range of importance is lower. Score damage from int2 reads,
which is what the pilot's QSA gain reflects, needs the measured one-way run.

### Context: the same table at q = 0.20 (7 fp slots)

| row | no promotion | with promotion | Δ [95% CI] (`open`) | Δ (`closed`) |
|---|---|---|---|---|
| R3 FMM (fp + int2) ↓ | 40.49% | 40.49% | 0.00 pp | |
| FMM of fp + local ↓ | 47.35% | 46.42% | −0.93 pp [−1.10, −0.77] | |
| FullKV mass on the fp tier ↑ | 3.33% | 3.71% | +0.38 pp [+0.31, +0.46] (+11.5%) | |
| reached only through the card fill ↓ | 1.03% | 0.86% | −0.17 pp [−0.20, −0.15] (−17%) | −0.45 pp (−34%) |
| R_Q ↑ | 53.57% | 56.98% | **+3.41 pp** [+2.66, +4.21] | +9.35 pp [+8.43, +10.32] |
| R_F+Q ↑ | 81.05% | 83.89% | +2.83 pp [+2.49, +3.19] | +6.56 pp [+5.77, +7.42] |
| QSA ↑ | 0.9396 | 0.9260 | −0.0135 [−0.0179, −0.0092] | |
| Global LIR, fp tier ↑ | 0.00% (0 / 25,856) | 1.92% (505 / 26,346) | +1.92 pp [+1.76, +2.08] | |
| Q → F transitions | 0 | 726 | +726 | |
| promoted vs demoted window, mass per step | 0.32% | 0.77% | 2.4× (704 vs 706 windows) | |

With 7 fp slots and a gate that reads only 2 of 7 int2 windows, promotion moves R_Q 11–20×
more than at q = 0.70, and fp-tier and read mass about 5–6× more. On 5,856 of the 91,592 re-run gate rows, the
one-way int2 tier held more card-less windows than the gate reads. There `open` read the ones
with the most FullKV mass, which still favours no promotion. **If the paper needs one setting
that shows promotion's benefit in mass, it is this one**, which matches the earlier finding
that q = 0.70 barely exercises promotion (`qevict_metrics_q0.7_vs_q0.2_wikitext.md`).

---

## Definitions, and how they map to Table 18

| Table 18 row | here | definition |
|---|---|---|
| R3 FMM | FMM, fp + int2 | at each eviction, of the next 32 steps' attention on windows that already existed, the share on windows outside fp + int2 (sinks excluded, head mean), `qevict_metrics.future_missed_mass`; same number as Observation II |
| Quantized-Score Agreement | QSA | at each eviction, Spearman between the cache's ranking signal and the FullKV cumulative mass through the previous step (both head means) over the windows that eviction puts in int2; mean over evictions |
| FullKV attention mass preserved, R_Q | R_Q | of each query head's FullKV attention on the int2 tier, the share in windows the read gate opened for that head's KV group (per head, then averaged: Observation IV's recall). In this cache the int2 tier is *read* only where opened; the rest reaches the output through the card fill |
| — | R_F+Q | the same over fp + int2: (fp + opened) / (fp + int2), where promotion moves mass between the two |
| Global LIR | Global LIR | episodes of a window outside the fp tier for 4 consecutive evictions, revived if it re-enters fp anywhere later (uncapped), ratio of summed counts; same number as the tier study's M7 |
| Recorded Q→F transitions | Q → F transitions | int2 at one eviction and fp at the next (Observation V, lag 1) |

## Next GPU run

This replaces the replay's two assumptions with a measurement, about 22 minutes on an
A100. Same prompts; the no-promotion column is then paired per (prompt, layer) but runs on its
own decode trajectory:

```bash
scripts/collect_observations.sh --model-path <llama-3.1-8b-instruct> \
    --dataset outputs/corpora/wikitext_test_articles.jsonl \
    --window-size 8 --cache-budget 0.20 --gate-ratio 0.25 --quant-ratio 0.70 --promotion oneway
python -m modules.evaluation.promotion_ablation \
    --zip outputs/obs_data/wikitext_test_articles_ws8_b0.2_g0.25_q0.7_p512_n512.zip \
    --oneway-zip outputs/obs_data/wikitext_test_articles_ws8_b0.2_g0.25_q0.7_p512_n512_oneway.zip \
    --out-dir outputs/promotion_ablation_q0.7_measured
```

## Reproduce

```bash
python -m modules.evaluation.promotion_ablation \
    --zip <run>.zip --out-dir outputs/promotion_ablation_q0.7     # ~3 min on CPU
```

## Scope

* One corpus, 8 prompts, two settings. This measures mechanism, not accuracy; nothing here is a
  LongBench number.
* Every number is CPU arithmetic on one recorded GPU run. The no-promotion column is a replay
  (assumptions above), not a recorded run.
* The gate bracket bounds the replay's gate, not the measured one; the measured arm's gate is
  the recorded one.
