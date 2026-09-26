# Q → F promotion in attention mass: the paper's Table 18 on the q = 0.70 run

**Data:** `wikitext_test_articles_ws8_b0.2_g0.25_q0.7_p512_n512.zip` (sha256 verified), the
run behind `reports/observations_wikitext_ws8_b0.2_g0.25_q0.7.md` and
`reports/qevict_metrics_q0.7_vs_q0.2_wikitext.md`. Llama-3.1-8B-Instruct, 8 wikitext-103
articles, 512-token prompt + 512 greedy decode steps, window 8, 5 sinks, local 128, 20% byte
budget: **2 fp + 26 int2 windows**, the read gate opens 7 of 26 per KV head. Read-gate verdict
`gated` (131,072 / 131,072 fused layer-steps). For context, the q = 0.20 run (7 fp + 7 int2,
gate opens 2 of 7), same prompts.

Produced by `python -m modules.evaluation.promotion_ablation`; machine-readable output in
`reports/promotion_ablation/q0.7.json` and `q0.2.json`; the paper table is
`reports/tables/promotion_ablation.tex`.

## The three columns

Table 18 of the paper compares two pilot runs on **different source articles** and calls the
result directional. Here every column uses **the same 8 prompts, the same queries, the same
bytes**:

| column | what it is | how it is made |
|---|---|---|
| **No promotion** | one-way policy (`quant_promotion: oneway`, `F → Q → E`), card gate on | **replayed**: at each of the run's 64 evictions (× 32 layers × 8 prompts), the rule `policy.compute_two_tier_retain` implements, applied to the run's own `rank_signal` at the same tier sizes. The windows it holds in int2 cannot take an fp slot. The gate is re-run by its own rule on the recorded cards |
| **With promotion** | shipped promotion, **gate off**: every int2 window read, as in the paper's pilot | the measured run's tiers and scores with every int2 window read |
| **With gated promotion** | shipped promotion, card gate on (7 of 26) | **measured**: the recorded run |

Δ is *with gated promotion − no promotion*, paired per (prompt, layer) trace, with 95%
percentile bootstrap intervals over 256 traces (within-run variability, not population CIs).
All three columns are scored against the same ground truth, the same queries over the
never-evicted KV, by the observation suite's own code.

**The gate-off column is only partly filled.** Under the replay the gate moves nothing but what
is read, so the rows it cannot move carry the measured values: FMM, Quantized-Score Agreement,
LIR, transitions and fp mass. **R_Q is empty**: it needs each int2 window's 2-bit read, and a
gated run records that only for the windows the gate opened. A `--gate-ratio 1.0` collector run,
passed as `--gate-off-zip`, measures the whole column, R_Q included (see *Next GPU runs*).

---

## Table 18, on this run (q = 0.70)

| Metric | No promotion | With promotion | With gated promotion | Δ [95% CI] |
|---|---|---|---|---|
| R3 FMM ↓ | 28.89% | 28.89% | 28.89% | 0.00 pp (identical kept set) |
| Quantized-Score Agreement ↑ | 0.9830 | 0.9830 | 0.9830 | −0.0000 [−0.0001, −0.0000] |
| FullKV attention mass preserved, R_Q ↑ | 88.52% | — | 88.54% | +0.025 pp [+0.010, +0.040] |
| Global LIR ↑ | 0.00% (0 / 27,136) | 0.69% | 0.69% (188 / 27,318) | +0.69 pp [+0.57, +0.79] |
| Recorded Q→F transitions | 0 | 233 | 233 | +233 |

Where the attention mass went (a share of each query head's FullKV attention per decode step):

| row | No promotion | With promotion | With gated promotion | Δ [95% CI] |
|---|---|---|---|---|
| FMM of the tiers read every step (fp + local), next 32 ↓ | 52.46% | 52.32% | 52.32% | −0.14 pp [−0.17, −0.11] |
| on the fp tier ↑ | 1.00% | 1.07% | 1.07% | **+0.064 pp** [+0.049, +0.082] (+6.4%) |
| on the fp tier, share of window mass (sinks excluded) ↑ | 2.45% | 2.58% | 2.58% | +0.13 pp [+0.10, +0.17] |
| read by the decode step (sinks + local + fp + read int2) ↑ | 87.24% | 90.14% | 87.27% | +0.028 pp [+0.023, +0.034] |
| reached only through the card fill ↓ | 2.90% | 0.00% | 2.87% | −0.028 pp [−0.035, −0.023] (−1.0%) |
| int2 mass in windows the decode reads (gate recall) ↑ | 59.08% | 100% | 59.24% | +0.17 pp [+0.12, +0.21] |
| fp + int2 mass the decode reads, (fp + read) / (fp + int2) ↑ | 64.20% | 100% | 64.49% | +0.29 pp [+0.24, +0.34] |

**The no-promotion gate is bracketed.** A window the run had promoted has no card estimate at
those steps (an fp window's card is not read). The replay above reads it for every KV head
(`open`), which favours no promotion, so every gate-dependent Δ above is a lower bound. With
the reverse bracket (`closed`: never read), the no-promotion values move as follows:

- **reached only through the card fill:** 2.97%, Δ −0.098 pp [−0.122, −0.079], −3.3%;
- **gate recall:** 58.41%, Δ +0.83 pp [+0.70, +0.99];
- **fp + int2 read:** 63.48%, Δ +1.02 pp [+0.84, +1.21].

R_Q has no `closed` value, because those windows would be credited by a card nobody recorded.
The true value sits near `open`: the gate opens a head's hottest int2 window 92.0% of the time
(`observations_wikitext_…_q0.7.md` §6).

### What each promotion carried

The 221 promotions with a full 32-step horizon, each against the fp window demoted at the same
eviction. Without promotion that window keeps the fp slot, so this pair is the per-decision
counterfactual:

| window | count | FullKV mass per step, next 32 steps |
|---|---|---|
| promoted Q → F | 221 | **0.78%** |
| demoted F → Q at the same evictions | 221 | 0.39% |

**Each promotion puts a window with 2.0× the future attention into the fp slot** (per-trace
gain +0.44 pp [+0.34, +0.55], 126 of 256 traces promote at least once). On the 30.6% of
(step, layer) cells whose fp set differs between the arms, the fp tier holds 1.21% of attention
with promotion against 1.00% without (+21%).

---

## Reading it

**Promotion changes which kept windows are read every step, not what is kept, and not how well
int2 preserves attention.**

* **R3 FMM is identical.** The one-way replay keeps exactly the run's fp + int2 windows at all
  16,384 layer-evictions. Without promotion, the fp tier stays on the first eviction's top two
  windows for the whole decode: the replayed fp set equals that pair at all 16,128 later
  layer-evictions of all 8 prompts.
* **R_Q barely moves.** 88.52% → 88.54% at q = 0.70; 88.11% → 87.99% at q = 0.20, which is not
  significant. Two reasons. A promoted window carries its int2 payload into fp (`dequant`), so
  promotion does not restore precision. And the 2-bit read and the calibrated card fill already
  give the int2 windows about 88% of their FullKV attention in either arm.
* **What promotion does move is small at q = 0.70:** fp-tier attention +6.4%, card-fill-only
  attention −1.0% to −3.3%, gate recall +0.17 to +0.83 pp, LIR 0 → 0.69%. The fp tier is two
  windows, and the gate already reads the window promotion would pin.
* **The gate costs far more read mass than promotion recovers.** Reading every int2 window
  (gate off) leaves 0.00% to the card fill against 2.87% gated. Promotion's effect on the same
  row is −0.03 pp.

**The pilot's Table 18 magnitudes do not reproduce on this cache and should not be quoted for
it.** They were R3 FMM −0.7 pp, QSA +0.038 and R_Q +6.8 pp; here they are 0.0 pp, −0.0001 and
+0.02 pp. The pilot's setting was 256 / 1024 prompt / decode on different articles per arm, and
QSA cannot show its effect in a replay: both arms carry the same scores, so QSA moves only by
*which* windows sit in int2 (at q = 0.20 it drops by 0.014 for that reason).

### Context: the same table at q = 0.20 (7 fp slots)

| Metric | No promotion | With promotion | With gated promotion | Δ [95% CI] |
|---|---|---|---|---|
| R3 FMM ↓ | 40.49% | 40.49% | 40.49% | 0.00 pp |
| Quantized-Score Agreement ↑ | 0.9396 | 0.9260 | 0.9260 | −0.0135 [−0.0179, −0.0096] |
| FullKV attention mass preserved, R_Q ↑ | 88.11% | — | 87.99% | −0.12 pp [−0.36, +0.09] |
| Global LIR ↑ | 0.00% (0 / 25,856) | 1.92% | 1.92% (505 / 26,346) | +1.92 pp [+1.75, +2.08] |
| Recorded Q→F transitions | 0 | 726 | 726 | +726 |
| FMM of fp + local ↓ | 47.35% | 46.42% | 46.42% | −0.93 pp [−1.11, −0.76] |
| on the fp tier ↑ | 3.33% | 3.71% | 3.71% | +0.38 pp [+0.32, +0.46] (+11.5%) |
| reached only through the card fill ↓ | 1.03% | 0.00% | 0.86% | −0.18 pp [−0.20, −0.15] (−17%; `closed` −34%) |
| gate recall ↑ | 53.55% | 100% | 56.98% | +3.43 pp [+2.71, +4.24] (`closed` +9.33 pp) |
| promoted vs demoted window, mass per step | 0.32% | 0.77% | 0.77% | 2.4× (704 vs 706 windows) |

With 7 fp slots and a gate that reads 2 of 7 int2 windows, promotion's effect on *what is read*
is several times larger than at q = 0.70:

- fp-tier and read mass about 6× larger;
- gate recall 11–20× larger;
- card-fill-only mass 17–34% lower.

R_Q still does not move. R_Q here is scored on 95.4% of head-steps. The rest are steps where the
one-way arm held more card-less windows than the gate reads, and they are left out of both arms.
On 23.5% of head-steps, every window the replay read was card-less, so its fill is calibrated
on the run's own read windows. If the paper needs a setting that shows promotion's benefit in
mass, it is q = 0.20, and the benefit is in what is read, not in R_Q.

---

## Definitions, and how they map to Table 18

| Table 18 row | definition here |
|---|---|
| R3 FMM | at each eviction, of the next 32 steps' attention on windows that already existed, the share on windows outside fp + int2 (sinks excluded, head mean): `qevict_metrics.future_missed_mass`, the same number as Observation II |
| Quantized-Score Agreement | at each eviction, Spearman between the cache's ranking signal and the FullKV cumulative mass through the previous step (both head means) over the windows that eviction puts in int2; mean over evictions |
| FullKV attention mass preserved, R_Q | per query head and decode step, each int2 window's FullKV attention `a` against the attention the decode step gives it, `c`: its 2-bit read where the gate opened it, its card fill where it did not (`cache_step_mass`, rescaled to FullKV by the local band, which both read exactly). `R_Q = Σ min(c, a) / Σ a` over the int2 tier: credit a window did not have in FullKV is not preserved mass. For the replay, the fill of every skipped window is recomputed with the kernel's own formula over the replay's read set. That formula reproduces the run's recorded fills on 99.998% of 65.8 M skipped windows within 1% |
| Global LIR | episodes of a window outside the fp tier for 4 consecutive evictions, revived if it re-enters fp anywhere later (uncapped); ratio of summed counts; the same number as the tier study's M7 |
| Recorded Q→F transitions | int2 at one eviction and fp at the next (Observation V, lag 1) |

## Checks

| check | q = 0.70 | q = 0.20 |
|---|---|---|
| replaying `bidir` reproduces the recorded tiers | **16,384 / 16,384** layer-evictions | 16,384 / 16,384 |
| gate rule re-run on the recorded cards reproduces the recorded picks | 1,044,545 / 1,048,576 (99.6%) | 1,047,576 / 1,048,576 (99.9%) |
| card-fill formula reproduces the recorded fills (within 1%) | 99.998% of 65.8 M | 99.999% of 18.0 M |
| one-way: kept set (fp + int2) differs from the run | **0** / 16,384 | 0 / 16,384 |
| one-way: fp set differs from the run | 5,014 / 16,384 (30.6%) | 11,449 / 16,384 (69.9%) |
| one-way: windows needing a score the run never recorded | 0 | 0 |
| R_Q: head-steps scored in both arms | 99.9% | 95.4% |
| R_Q: replay-read windows whose recorded credit was a fill, not a read | 1,556 | 320 |

The gate-check misses are fp16 rounding of the stored card estimates.

## Next GPU runs

Each is about 22 minutes on an A100, on the same prompts. The first replaces the no-promotion
replay with a measurement; the second measures the gate-off column, R_Q included. Both are
paired per (prompt, layer), each on its own decode trajectory.

```bash
M=<llama-3.1-8b-instruct>; D=outputs/corpora/wikitext_test_articles.jsonl
scripts/collect_observations.sh --model-path $M --dataset $D --window-size 8 \
    --cache-budget 0.20 --gate-ratio 0.25 --quant-ratio 0.70 --promotion oneway
scripts/collect_observations.sh --model-path $M --dataset $D --window-size 8 \
    --cache-budget 0.20 --gate-ratio 1.0 --quant-ratio 0.70
R=outputs/obs_data/wikitext_test_articles_ws8_b0.2
python -m modules.evaluation.promotion_ablation --zip ${R}_g0.25_q0.7_p512_n512.zip \
    --oneway-zip ${R}_g0.25_q0.7_p512_n512_oneway.zip \
    --gate-off-zip ${R}_g1_q0.7_p512_n512.zip \
    --out-dir outputs/promotion_ablation_q0.7_measured
```

## Reproduce

```bash
python -m modules.evaluation.promotion_ablation \
    --zip <run>.zip --out-dir outputs/promotion_ablation_q0.7     # a few minutes on CPU
```

## Scope

* One corpus, 8 prompts, two settings. This measures mechanism, not accuracy; nothing here is a
  LongBench number.
* Every number is CPU arithmetic on one recorded GPU run. Only *with gated promotion* is
  measured; the other two columns are replays under the assumptions above.
