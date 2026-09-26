# Why promote Q → F: the case from the recorded runs

Figure: `reports/figures/obs_promotion.{pdf,svg,png}` (print size, 6.5 in). Numbers:
`reports/promotion_motivation.json` (`scripts/promotion_motivation.py`, about 1 min on CPU from the
two collector zips) and `reports/promotion_ablation/q0.{7,2}.json`. Llama-3.1-8B-Instruct,
8 wikitext-103 prompts, 512-token prefill + 512 decode steps, ω = 8, 20% byte budget, same
prompts and queries in every arm. "No promotion" is the one-way policy replayed on each run's own
ranking signal: `modules/evaluation/promotion_ablation.py`, which reproduces the recorded tiers
at 16,384 / 16,384 layer-evictions.

## The argument, in four numbers

**1. Importance drifts, so an exact tier chosen once goes stale, and it gets worse the longer
the model generates** (panel a). Without promotion, the fp tier is decided at the first
eviction and never revisited: the replayed fp set equals the step-1 pair at all 16,128 later
layer-evictions at q = 0.70. We count a layer-eviction as stale when that frozen tier is not
the one the cache's own ranking now picks. With promotion it always is, by construction.

| decode steps | 1–64 | 129–192 | 257–320 | 385–448 | 449–512 | all |
|---|---|---|---|---|---|---|
| q = 0.70 (2 fp slots) | 6.7% | 26.4% | 34.1% | 44.0% | **45.6%** | 30.6% |
| q = 0.20 (7 fp slots) | 25.8% | 65.8% | 80.3% | 86.7% | **87.3%** | 69.9% |

Staleness climbs through the whole decode and has not levelled off at q = 0.70 by step 512.
Long generations, such as reasoning chains, are where a one-way cache is most out of date.

**2. Promotion keeps the exact tier aimed at what the model reads next** (panel b). This is the
share of the fp tier that lies among the windows the *next 32 steps* attend to most (hindsight,
same held set in both arms):

| | no promotion | with promotion | chance | gap, steps 1–64 → 449–512 |
|---|---|---|---|---|
| q = 0.70 | 22.2% | **25.4%** | 7.1% | +0.8 → +3.6 pp |
| q = 0.20 | 65.6% | **73.3%** | 50.0% | +2.7 → **+11.0 pp** |

Without promotion the aim stays flat (q = 0.20: 64% → 66%). With it, the aim improves as the
decode goes on (67% → 77%): the cache learns where attention has moved and follows it.

**3. Every promotion is a better pick than the window it replaces** (panel c). Over the next 32
steps, a promoted window draws **2.0×** (q = 0.70) and **2.4×** (q = 0.20) the attention of the
fp window it displaces (0.78% vs 0.39% and 0.77% vs 0.32% of a head's attention per step; 221
and 704 promotions). Promoted windows draw 2.34× / 1.92× the average candidate. 36.7% / 82.2%
of promotions land in the hindsight-best fp set, about 5× chance at q = 0.70
(`qevict_metrics_q0.7_vs_q0.2_wikitext.md`, Observation V).

**4. It costs nothing in bytes and almost nothing in error.**

- **Bytes:** tier sizes are fixed and the kept set is unchanged, so R3 FMM is identical with and
  without promotion. Promotion only re-ranks windows the cache already holds.
- **Frequency:** it is rare, 0.01 / 0.05 moves per eviction per layer.
- **Error:** the promoted window's int2 payload adds 0.004 / 0.014 relative output error, against
  0.182 / 0.256 from eviction itself.

## One paragraph for the paper

> Attention importance is not static: a one-way (F→Q→E) cache fixes its full-precision tier at the
> first eviction and never revisits it, and on the same queries that tier diverges from the
> cache's own importance ranking in 6.7% of layer-evictions over the first 64 decode steps but in
> 45.6% over the last 64 (87.3% with seven fp slots). Q→F promotion keeps the exact tier on the
> windows the next 32 steps attend to most (73.3% vs 65.6%; the gap grows from 2.7 to 11.0 points
> over the decode), and each promotion replaces an fp window with one drawing 2.0–2.4× its
> future attention, at no byte cost: tier sizes, and hence the retained set, are unchanged.

## What not to claim

* **The aggregate attention effect is small at q = 0.70.** The fp tier holds 1.07% vs 1.00% of
  attention (+6.4%), because it is two windows. At q = 0.20 it is 3.71% vs 3.33% (+11.5%).
  Promotion does **not** raise R_Q (int2 fidelity) at either setting
  (`promotion_mass_q0.7_wikitext.md`). The case is **aim and freshness**, not mass or fidelity.
* **Staleness beyond 512 steps is not measured.** The q = 0.70 curve is still rising at 512;
  say "grows over the measured decode", not a number for longer ones.
* **"No promotion" is a replay** of the one-way rule on the recorded trajectory, not a recorded
  run. A `--promotion oneway` collector run (about 22 min on an A100) turns it into a measurement.
* One corpus, 8 prompts. Mechanism, not accuracy.
