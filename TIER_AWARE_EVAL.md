# Tier-Aware Evaluation — Verification & Guide

How the Jaccard (Suite A) and attention-mass (Suite B) suites treat the int2
**Q tier** on `quant_batched`, how to run them, and what each metric means.
This answers three questions, verified against the code.

---

## 1. Do we count quantized tokens in every calculation, even though they don't match exactly due to quantization loss?

**Yes — in the retention/benefit metrics a quantized window counts as fully kept,
and we do *not* pretend the loss is zero: it is measured on a separate axis.**

The key idea: **"is this window kept?" is decided by the window's identity (its id),
not by its values.** Quantization changes a window's KV *values*, never its id, so a
kept-but-quantized window can never be demoted out of the "kept" set by quant loss.
The *quality* cost of that loss is then reported separately, so nothing is hidden.

| Metric | Counts a Q window as kept? | Does quant loss change it? |
|---|---|---|
| `jaccard_kept` (fp∪Q) | **Yes** — set membership by window id | No — identity, not values |
| `jaccard_fp` (fp only) | No, by design (the fp-only baseline) | No |
| `missed_mass_kept` / `recovered_mass_q` | **Yes** — Q counts as retained | No — uses base (full-precision) mass |
| `q_tier_fidelity` | — | **Yes — this is where loss is measured** |
| `recovered_mass_q_discounted` | Yes, then debited | **Yes — `= recovered_mass_q × fidelity`** |
| `cos_sim` / `pearson` / `spearman` / `kl_ours_base` | the top Q windows that fall in the retained top-K | **Yes — uses ours' dequantized scores** |

Reading the table by group:

- **Selection / benefit (identity-based, loss-blind by design).** `jaccard_kept`
  compares *which* windows we physically keep (fp ∪ Q, by id) against the windows the
  full model considers most important — so keeping a window quantized scores exactly
  like keeping it in fp16. `recovered_mass_q = missed_mass_fp − missed_mass_kept`
  (`utils/sticky_metrics.py`) is a **policy simulation over the base run's
  ground-truth masses**: it credits the Q tier with the full-precision attention mass
  it holds that an fp-only drop would have lost. Guaranteed ≥ 0 (kept ⊇ fp). This is
  the *potential* benefit assuming the right windows are chosen.

- **Quality (loss is measured, not assumed away).** `q_tier_fidelity`
  (`faithfulness_runner`) is the mean cosine similarity between ours' **dequantized**
  Q-window scores and base's scores over the same windows — precisely "how much do
  they fail to match because of quantization?", in `[0, 1]`.
  `recovered_mass_q_discounted = recovered_mass_q × q_tier_fidelity` re-credits the
  benefit only to the extent the dequantized KV is faithful. So you get both the raw
  ("kept, not dropped") and the honest ("discounted for loss") numbers.

- **Distribution metrics still show the degradation.** `cos_sim` / `pearson` / `kl`
  run over the actual retained top-K, which at `q > 0` is the fp windows **plus** the
  top few Q windows. Those use ours' quantized scores, so a Q window whose dequantized
  attention drifted from the truth *lowers* these — the loss is visible here too, not
  swept under the rug.

**In one line:** a quantized window is counted as kept by its identity (loss can't
demote it), and the loss itself is surfaced by `q_tier_fidelity` /
`recovered_mass_q_discounted` and by the distribution metrics — never silently
ignored, never silently penalising retention.

---

## 2. How do we run it?

Single entry point, YAML-driven: `python main.py --config <yaml>` routes on
`run.mode` (`main.py:21`). The tier-aware numbers come out of a **three-step
pipeline** (base → ours → faithfulness), plus an optional plot step.

```bash
# 1. BASE — full cache, never evicts: the ground-truth attention masses.
python main.py --config configs/eval_parity_base.yaml
#   → outputs/parity_base_<dataset>.npz

# 2. OURS — our windowed cache WITH the Q tier on.  quant_ratio > 0 turns on the
#    two-tier split; it needs a window_size divisible by 4 (int2 crumb packing).
#    The eager config
#    uses window_size 32, so just override the ratio:
python main.py --config configs/eval_parity_ours_eager.yaml \
       --override cache.quant_ratio=0.5
#   → outputs/parity_ours_eager_<dataset>.npz
#     (now also carries all_window_ids / all_window_tier and quant_ratio/top_k_fp/N_q)

# 3. FAITHFULNESS — pure post-processing over the two npzs: computes every metric.
python main.py --config configs/eval_faithfulness.yaml
#   → outputs/faithfulness_results.npz
#     (jaccard_fp/kept/lift, missed_mass_kept, recovered_mass_q,
#      recovered_mass_q_discounted, q_tier_fidelity, …)

# 4. (optional) PLOTS — fp-vs-kept Jaccard overlay + shaded rescued-mass band.
python main.py --config configs/eval_visualize.yaml
```

Notes:
- The knob is **`cache.quant_ratio`** (`configs/base.yaml:24`). `0.0` = pure fp16
  (byte-identical to the single-tier path); `> 0` splits the evictable budget so the
  int2 Q tier holds more windows per byte. The factor is **not** the naive 8×: the
  fp16 scale/zero grid is fixed overhead that the 2-bit codes no longer dominate, so
  it grows with `window_size` as the grid amortizes — ~3.9× at `window_size` 8, ~6.1×
  at the `window_size` 32 this eager config uses (head_dim 128; see `resolve`'s
  `b_q`). Set it in the yaml or via `--override`.
- Step 3's config (`configs/eval_faithfulness.yaml`) points `base_npz_path` /
  `ours_npz_path` at the two outputs above.
- **q = 0 sanity / back-compat:** run steps 2–3 with `--override cache.quant_ratio=0.0`
  and every new series collapses to the legacy value (`jaccard_kept == jaccard`,
  `recovered_mass_q == 0`).
- **Fast logic check without a model** (unit + synthetic end-to-end):
  ```bash
  python -m pytest tests/test_sticky_metrics.py \
         modules/evaluation/test_faithfulness.py \
         modules/evaluation/test_ours_parity.py -q
  ```

---

## 3. Which metrics are we measuring?

### In simple terms

- **Jaccard (fp / kept / lift)** — *Did we keep the same windows the full model cares
  about?* `fp` = using only the full-precision tier; `kept` = counting the quantized
  tier too; `lift` = how much the Q tier adds.
- **Missed mass / recovered mass** — *How much important attention did the cache throw
  away, and how much did the Q tier save?* `recovered_mass_q` is the headline: the
  attention mass quantization rescues that a plain drop would have lost.
- **Q-tier fidelity** — *When we keep a window quantized, how close is it to the real
  thing?* Feeds the "discounted" benefit so we don't overstate it.
- **Score-distribution metrics (cosine / Pearson / Spearman / KL / mass-ratio)** — *Do
  our per-window importance scores match the full model's?*
- **LIR (Lazy Insertion Rescue)** — *Does the policy thrash* (drop then re-admit the
  same windows)?

### In detail

All arrays land in `outputs/faithfulness_results.npz` (schema 2.2). `T` = generation
steps, `L` = layers, `H` = heads. Two-tier arrays collapse to their legacy
counterparts at `q = 0`.

**Suite A — Jaccard (window-selection agreement).** Base's "ground truth" is its top
windows by attention mass; ours is what the cache retains.
- `jaccard` `[T,L,1]`, `jaccard_per_layer` `[T,L]`, `jaccard_global` `[T]` — legacy
  single-tier top-K overlap (`utils/metrics.py:jaccard_topk`), unchanged.
- `jaccard_fp` `[T,L]` — overlap of ours' **fp-only** survivors with base's top-`|fp|`.
- `jaccard_kept` `[T,L]` — overlap of ours' **fp ∪ Q** survivors with base's
  top-`|fp∪Q|` (set sizes matched so the ratio stays interpretable).
- `jaccard_lift = jaccard_kept − jaccard_fp` `[T,L]` — the selection quality the Q
  tier buys. Usually ≥ 0; **can be slightly negative** if the Q tier holds windows
  base ranks low — a useful signal that the tier is keeping junk, not a bug.
- `*_global` `[T]` variants = mean over layers.
- Computed in `faithfulness_runner._compute_metrics` via `_jaccard_sets` /
  `_base_top_ids`, using the ours npz's `all_window_ids` / `all_window_tier`
  (0 = fp, 1 = Q, 2 = local).

**Suite B — attention (missed / recovered) mass.** A policy simulation over the base
run's ground-truth masses (`utils/sticky_metrics.py`). fp tier = `top_k_fp` windows,
Q tier = the next `N_q` by score (mirrors `policy.compute_two_tier_retain`). All
normalised to `[0,1]` by total valid-window mass.
- `missed_mass` (= `missed_mass_fp`) `[T]`, `missed_mass_per_layer` `[T,L]` — mass
  below the **fp** tier (the fp-only-drop baseline).
- `missed_mass_kept` `[T]` / `_per_layer` / `_total` — mass dropped by **neither**
  tier (the honest two-tier miss; ≤ `missed_mass`).
- `recovered_mass_q` `[T]` / `_per_layer` / `_total` = `missed_mass − missed_mass_kept`
  — **mass the Q tier rescues** (the visible benefit).
- `recovered_mass_q_discounted` `[T]` / `_total` = `recovered_mass_q × q_tier_fidelity`
  — rescued mass debited for quantization loss.
- `missed_mass_fresh` / `missed_mass_fresh_kept` / `recovered_mass_q_fresh` `[T]` —
  Fresh-K (re-pick-every-flush) counterparts; Fresh-K mirrors the production two-tier
  eviction's per-flush re-rank and is a lower bound on missed mass.

**Quantization fidelity (Phase 3).**
- `q_tier_fidelity` scalar, `q_tier_fidelity_per_layer` `[L]` — mean cosine similarity
  of ours' dequantized vs base's head-mean mass over the Q windows, in `[0,1]`. A
  score-space proxy (not raw-KV reconstruction); a true demote-time round-trip
  fidelity is a possible follow-up.

**Score-distribution metrics (unchanged; reflect quant loss where Q windows appear).**
Over the retained top-K set, per `[T,L]`:
- `cos_sim`, `pearson`, `spearman` — agreement of ours' vs base's window scores
  (higher = better).
- `kl_ours_base` — KL(ours ‖ base) over the retained windows (lower = better).
- `mass_ratio` — base_mass / ours_mass over retained windows (≈ 1 = well-matched).

**LIR — Lazy Insertion Rescue (thrashing).**
- `global_lir` scalar, `lir_per_layer` `[L]`, `lir_per_head` `[L,H]` — fraction of
  windows ignored for `m` flushes that are later re-admitted. High = thrashing. This
  stays an fp-tier quantity (the Q tier is not marked in the LIR selection).

See `EVALUATION_GUIDE.md` (Metrics 1 and 8) for the underlying formulas.
