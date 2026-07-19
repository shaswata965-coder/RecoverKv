# Evaluation Guide — Corpus Loading to Final Scores

This document traces the complete evaluation pipeline: how articles are loaded,
how each evaluation suite runs, and how every metric (Jaccard, cosine similarity,
Pearson, Spearman, KL divergence, mass ratio, LongBench) is computed step by step.

---

## 1. Corpus Loading

### Entry point

Every evaluation suite that needs text begins by constructing a `CorpusLoader`.

**File:** `data/corpus_loader.py`  
**Class:** `CorpusLoader`

```python
loader = CorpusLoader(dataset="wikitext-103")  # or "pg19"
```

### `CorpusLoader.load()` (corpus_loader.py:84)

The first call triggers the actual dataset fetch; subsequent calls return the
cached list.

**wikitext-103 path:**
1. `load_dataset("wikitext", "wikitext-103-raw-v1", split="test")` from HuggingFace.
2. All rows concatenated into one large string.
3. `_split_into_articles()` (line 70): splits on `\n(?= = [^=])` — every level-1
   heading `= Title =` starts a new article. Empty articles are dropped.
4. Returns `List[str]` — one entry per Wikipedia article.

**PG19 path:**
1. `load_dataset("deepmind/pg19", split="test")`.
2. Returns `[row["text"] for row in ds]` — one entry per book.

### `CorpusLoader.sample_articles(n, seed)` (corpus_loader.py:118)

```python
rng = random.Random(seed)
indices = rng.sample(range(len(articles)), n)
indices.sort()          # deterministic ordering regardless of sample order
return [articles[i] for i in indices]
```

The sort ensures that running with the same `(n, seed)` always returns articles
in the same order, making multi-run comparisons reproducible.

### Article identity — `ArticleRegistry`

**File:** `data/article_registry.py`  
**Class:** `ArticleRegistry`

After articles are sampled, each one is registered:

```python
registry.register_article(dataset, article_id, text)
# Computes sha256(text.encode("utf-8"))
# Stores (dataset, article_id) → sha256 mapping
```

The SHA is embedded in every output NPZ's metadata. When `OursParityRunner` loads
a base NPZ it calls `sha256_file()` and cross-checks per-article SHAs so that a
mismatch in corpus version is caught before any GPU work begins.

---

## 2. Suite A — Parity (Baseline Run)

### What it measures

The base parity run is a **reference run** with no eviction. Its purpose is to
record the "ground truth" top-K window selections so that the ours run can be
compared against them.

### Runner

**File:** `modules/evaluation/base_parity_runner.py`  
**Class:** `BaseParityRunner`  
**Method:** `run()` (line 26)

### Step-by-step flow

```
1. Load config, corpus loader, article registry
2. Load model with attn_implementation="eager", DynamicCache (no eviction)
3. For each sample article:
   a. Tokenize (truncate to prefill_len tokens)
   b. Prefill: model(input_ids, past_key_values, output_attentions=True)
   c. For each of gen_len generation steps:
      i.  Forward: model(last_token, past_key_values, output_attentions=True)
      ii. For each layer, collect attn_weights [B, H_q, 1, S]
      iii. Accumulate cumulative attention scores across query steps (H2O style)
      iv. Compute window scores:
            compute_window_scores(attn, num_sink, window_size)
      v.  Select top-K windows from evictable region (topk on evictable slice)
      vi. Store: top_window_indices[step, layer, :] = topk indices
4. Save NPZ:
   - top_window_indices  [num_samples, num_steps, num_layers, K]
   - window_scores       [num_samples, num_steps, num_layers, H_q, W]
   - generated_tokens    [num_samples, num_steps]
   - eviction_step_mask  [num_samples, num_steps]  (all False for base)
   - metadata_json       JSON blob with run parameters + SHA fingerprints
```

### Suite A — Ours Run

**File:** `modules/evaluation/ours_parity_runner.py`  
**Class:** `OursParityRunner`

1. Validates backend/attn_implementation pairing via
   `validate_backend_attn_pairing()` (`utils/cache_factory.py:88`).
2. Loads and validates the base NPZ — checks `article_sha`, `model_name`,
   `window_size`, `num_sink_tokens`, `seed`, `prefill_len`, `gen_len` all match.
3. Loads corpus, cross-checks per-sample SHA.
4. Loads model with chosen `attn_implementation`.
5. Calls `install_score_hooks(model, cache, resolved_config)`.
6. For each sample, runs generation **teacher-forced** from `base["generated_tokens"]`
   (so both runs decode identical token sequences — the only difference is the cache).
7. Saves same NPZ schema as base, plus `retained_window_ids` and
   `retained_window_scores` arrays needed by the faithfulness runner.

---

## 3. Suite B — Faithfulness (Score Distribution Comparison)

**File:** `modules/evaluation/faithfulness_runner.py`  
**Class:** `FaithfulnessRunner`  
**Method:** `run()` (line 85)

This suite loads **no model**. It is pure tensor arithmetic on the two NPZ files.

### Input loading

```python
base = _load_npz(fc.base_npz_path)    # parity_base_*.npz
ours = _load_npz(fc.ours_npz_path)    # parity_ours_*.npz
```

`_load_npz()` (line 65): reads `metadata_json` as a JSON string and all array
arrays from the NPZ into a dict. Metadata alignment is validated before any
metric computation (matching `article_sha`, `seed`, `prefill_len`, `gen_len`,
`window_size`, `num_sink_tokens`, `model_name`).

### Metric computation — `_compute_metrics()` (line 113)

Arrays loaded:
- `base_ws`   — `[S, T, L, H, W]` window scores from base run
- `base_tk`   — `[S, T, L, K]`    top-K window indices from base run
- `ours_tk`   — `[S, T, L, K]`    top-K window indices from ours run
- `ours_rid`  — `[S, T, L, M]`    retained window IDs (mapped back to original positions)
- `ours_rsc`  — `[S, T, L, H, M]` ours scores at retained windows

For each sample `s`, step `t`, layer `li`:

```
retained IDs = ours_rid[s, t, li]          (M values, -1 padded)
valid        = (retained IDs >= 0) & (retained IDs < W_act)
ret_ids      = retained IDs at valid positions    [n_ret]

o_sc = ours_rsc[s, t, li, :, valid].mean(dim=0)  [n_ret]  (mean over heads)
b_sc = base_ws [s, t, li, :, ret_ids].mean(dim=0) [n_ret]  (base scores at same windows)
```

Both vectors are over the **same set of windows** (the ones ours chose to retain,
looked up in the base's score array at their original positions via `ret_ids`).

#### Metric 1 — Jaccard similarity

**File:** `utils/metrics.py:32`  
**Function:** `jaccard_topk(ours_topk, base_topk) → [T, L, H]`

Measures whether ours and base select the same top-K window indices.

```python
ours_exp = ours_topk.unsqueeze(-1)   # [T, L, H, K, 1]
base_exp = base_topk.unsqueeze(-2)   # [T, L, H, 1, K]
matches  = (ours_exp == base_exp)    # [T, L, H, K, K]

# Intersection: for each ours element, does any base element match?
intersection = matches.any(dim=-1).sum(dim=-1).float()  # [T, L, H]

# Union = |A| + |B| - |A∩B| (both sets have exactly K elements)
union    = 2.0 * K - intersection
jaccard  = intersection / union      # [T, L, H] ∈ [0, 1]
```

**Aggregation:**
- `aggregate_per_layer()` (metrics.py:68): `j.mean(dim=-1)` → `[T, L]`
- `aggregate_global()` (metrics.py:73): `j.mean(dim=(-2,-1))` → `[T]`
- `final_step_heterogeneity()` (metrics.py:78): `j[-1].std(dim=-1)` → `[L]`
  (std across heads at the last step — measures per-layer agreement)

**Tier-aware Jaccard (two-tier, schema ≥ 2.2).** The legacy Jaccard above slices
ours' single-tier top-K, so with `quant_ratio > 0` a window kept in the int4 **Q
tier** gets no credit — it scores as if dropped. The faithfulness runner therefore
also computes, per `(step, layer)`, the overlap of ours' *retained* set against
base's top-`n` windows of **matching size** (`faithfulness_runner._jaccard_sets` /
`_base_top_ids`), split by tier:
- `jaccard_fp`   — ours **fp-only** survivors vs base top-`|fp|` (≈ what an fp-only
  drop would capture).
- `jaccard_kept` — ours **fp ∪ Q** survivors vs base top-`|fp∪Q|` (Q credited — a
  kept-quantized window no longer counts as a miss).
- `jaccard_lift` = `jaccard_kept − jaccard_fp` — the Jaccard the Q tier buys.
  (Usually ≥ 0, but negative if the Q tier holds windows base ranks low — itself a
  useful signal that the tier is retaining junk.)

The full survivor axis + tier tag comes from the ours npz's `all_window_ids` /
`all_window_tier` (0=fp, 1=Q, 2=local); at `quant_ratio == 0` there is no Q tier,
so `jaccard_fp == jaccard_kept == jaccard` and `jaccard_lift == 0`.

#### Metric 2 — Cosine similarity

**File:** `modules/evaluation/faithfulness_runner.py:33`  
**Function:** `_cosine(o_sc, b_sc)`

```python
F.cosine_similarity(o_sc.unsqueeze(0), b_sc.unsqueeze(0), dim=1).clamp(-1, 1)
```

Range: `[-1, 1]`. Higher is better (1 = identical direction in score space).

#### Metric 3 — Pearson correlation

**File:** `faithfulness_runner.py:38`  
**Function:** `_pearson(o_sc, b_sc, eps=1e-8)`

```python
a_c = a - a.mean()
b_c = b - b.mean()
(a_c * b_c).sum() / (a_c.norm() * b_c.norm()).clamp(min=eps)
```

Range: `[-1, 1]`. Measures linear correlation of score magnitudes.

#### Metric 4 — Spearman rank correlation

**File:** `faithfulness_runner.py:45`  
**Function:** `_spearman(o_sc, b_sc)`

```python
a_rank = a.argsort().argsort().float()
b_rank = b.argsort().argsort().float()
_pearson(a_rank, b_rank)
```

Pearson applied to ranks. Robust to monotone non-linearities in score magnitude.
Range: `[-1, 1]`.

#### Metric 5 — KL divergence KL(ours ‖ base)

**File:** `faithfulness_runner.py:52`  
**Function:** `_kl(p=o_sc, q=b_sc, eps=1e-8)`

```python
p = o_sc.clamp(min=0)
q = b_sc.clamp(min=0)
p_prob = (p + eps) / (p.sum() + eps * n)   # normalize to distribution
q_prob = (q + eps) / (q.sum() + eps * n)
kl = (p_prob * (p_prob.log() - q_prob.log())).sum().clamp(min=0)
```

`p` = ours distribution, `q` = base distribution.  
Range: `[0, ∞)`. Lower is better (0 = distributions identical).

#### Metric 6 — Mass ratio

**File:** `faithfulness_runner.py:217`

```python
mr = b_sc.sum() / o_sc.sum().clamp(min=1e-8)
```

Ratio of total attention mass base assigned to the retained windows vs what ours
assigned. `≈1` means ours and base agree on total importance of the retained set.

#### Metric 7 — Global LIR (Lazy Insertion Rescue)

**File:** `utils/sticky_metrics.py`
**Functions:** `flush_geometry`, `simulate_policy`, `lir_counts`, `compute_sticky_metrics`

This is a **policy-simulation** metric, not a pairwise comparison. It runs the
production Sticky-K eviction policy over the **base run's window scores** (the
ground-truth attention masses, since base never evicts) and asks: *how often does
the policy drop a window and then re-admit it later?*

```
A "flush" is one decode step t. At flush t there are w_act = ceil((prefill_len
+ t + 1 - sink) / window_size) history windows; the last `local_windows` are an
always-kept recency tail, the rest are evictable candidates competing for
`top_k_windows` slots via a sticky set (one best-swap per flush, on strict
improvement) — exactly policy.compute_retain_window_indices.

selection_matrix[t, k] = 1 if window k is retained at flush t.

A (flush r, window k) pair is ELIGIBLE when:
  - the lookback [r-m+1, r] is in range,            (default m = 3)
  - window k existed at the lookback start,
  - window k was NOT retained in any of those m flushes.
It is RESCUED if window k is retained at some later flush > r.

global_lir = total_rescued / total_eligible     (ratio of summed counts)
```

High LIR ⇒ the policy thrashes (drops then re-admits the same windows);
low LIR ⇒ stable selection. Aggregations (ratio-of-summed-counts):
- `global_lir`    — scalar over all layers and heads (head-mean simulation).
- `lir_per_layer` — `[L]`, per layer (head-mean simulation, matches the cache).
- `lir_per_head`  — `[L, H]`, each head simulated independently.

#### Metric 8 — Absolute missed mass

**File:** `utils/sticky_metrics.py:simulate_policy`

The same Sticky-K simulation also records, per flush, the raw ground-truth
attention mass sitting on evictable windows the policy did **not** retain:

```
missed[t] = Σ  truth_mass[t, w]   for evictable windows w not in the retained set
```

The local recency tail is always kept, so it never contributes missed mass.
A **Fresh-K** baseline (re-pick the top-K every flush, no memory) is computed
alongside as a lower bound — Fresh-K is greedy-optimal per flush, so
`missed_mass_fresh ≤ missed_mass` always. Outputs:
- `missed_mass`           — `[T]`, Sticky-K trajectory (mean over layers/samples).
- `missed_mass_per_layer` — `[T, L]`.
- `missed_mass_fresh`     — `[T]`, Fresh-K baseline.
- `missed_mass_total`     — scalar, per-flush mean of `missed_mass`.

`local_windows` = `local_window_size_resolved // window_size`, and `m` = metadata
`lir_ignore_threshold` (default 3) — all read from the ours npz metadata.

**Two-tier missed mass (schema ≥ 2.2).** `simulate_policy` models the fp tier
(`history_budget_K = top_k_fp` windows) **and** an int4 **Q tier** of the next `N_q`
strongest evictable windows (`n_q` arg), mirroring `policy.compute_two_tier_retain`.
Because the Q tier is physically kept, counting it as "missed" would penalise
exactly the mass quantization *rescues*. So the runner reports:
- `missed_mass` (== `missed_mass_fp`) — mass below the **fp** tier (fp-only-drop
  baseline). At `q > 0` this uses `top_k_fp`, not `top_k_windows`.
- `missed_mass_kept` — mass dropped by **neither** tier (the honest two-tier
  number). Guaranteed `≤ missed_mass` (kept ⊇ fp).
- `recovered_mass_q` = `missed_mass − missed_mass_kept` — mass the Q tier rescues
  (**the visible benefit**), plus `_per_layer` / `_total` and Fresh-K counterparts
  (`missed_mass_fresh_kept`, `recovered_mass_q_fresh`).
- `recovered_mass_q_discounted` = `recovered_mass_q × q_tier_fidelity`, where
  `q_tier_fidelity ∈ [0, 1]` is the mean cos-sim of ours' (dequantized) vs base's
  head-mean mass over the Q windows — credits the Q tier only as far as its
  quantized KV is faithful.

At `quant_ratio == 0` (`N_q == 0`, `top_k_fp == top_k_windows`) every series above
collapses to the legacy single-tier value: `missed_mass_kept == missed_mass`,
`recovered_mass_q == 0`.

### Output

```
np.savez_compressed("outputs/faithfulness_results.npz",
    jaccard           [T, L, 1]      per-(step, layer) Jaccard
    jaccard_per_layer [T, L]         mean over heads
    jaccard_global    [T]            mean over heads + layers
    heterogeneity     [L]            std across heads at last step
    cos_sim           [T, L]         cosine similarity
    pearson           [T, L]         Pearson correlation
    spearman          [T, L]         Spearman rank correlation
    kl_ours_base      [T, L]         KL(ours ‖ base)
    mass_ratio        [T, L]         base_mass / ours_mass
    global_lir            scalar     Sticky-K rescue rate (global)
    lir_per_layer        [L]         rescue rate per layer
    lir_per_head         [L, H]      rescue rate per (layer, head)
    missed_mass          [T]         fp-only-drop missed-mass trajectory
    missed_mass_per_layer[T, L]      missed mass per layer
    missed_mass_fresh    [T]         Fresh-K baseline missed-mass trajectory
    missed_mass_total    scalar      per-flush mean of missed_mass
    # ── two-tier: Q tier credited (schema ≥ 2.2; collapse to above at q=0) ──
    jaccard_fp / jaccard_kept / jaccard_lift          [T, L]   tier-aware Jaccard
    jaccard_fp_global / jaccard_kept_global / _lift   [T]      + global means
    missed_mass_kept [T] / _per_layer [T,L] / _total  scalar   two-tier missed mass
    recovered_mass_q [T] / _per_layer / _total                 mass the Q tier rescues
    recovered_mass_q_discounted [T] / _total                   × q_tier_fidelity
    missed_mass_fresh_kept / recovered_mass_q_fresh   [T]      Fresh-K counterparts
    q_tier_fidelity  scalar / _per_layer [L]                   Q dequant faithfulness
    metadata_json     JSON string    provenance + SHA checksums  (schema v2.2)
)
```

The ours parity npz (schema ≥ 1.2) carries the tier inputs: `all_window_ids`
`[S,T,L,W]` and `all_window_tier` `[S,T,L,W]` (0=fp, 1=Q, 2=local, -1=pad), plus
`quant_ratio` / `top_k_fp` / `N_q` in its metadata.

---

## 4. Suite D — LongBench

### Runner

**File:** `modules/evaluation/longbench_runner.py` (not fully shown in exploration,
but orchestrated identically to the parity runners)

**Config:** `LongBenchConfig` in `utils/config.py`  
**Datasets:** Multi-document QA, single-document QA, summarization, code, etc.

### Prediction generation

1. Loads each LongBench dataset from HuggingFace (e.g. `THUDM/LongBench`).
2. For each example: concatenate context + instruction → tokenize.
3. Runs `model.generate()` with either full `DynamicCache` or `WindowedCache`.
4. Decodes prediction and writes `<dataset>.jsonl` to `predictions/` dir.

### Scoring — `score_predictions()` (longbench_scoring.py:78)

**File:** `modules/evaluation/longbench_scoring.py`  
**Class:** `LongBenchScorer`  
**Entry:** `score_predictions(predictions_dir, output_csv)`

```
1. Load dataset2metric.json   (maps each dataset name → metric function name)
2. For each <dataset>.jsonl in predictions_dir:
   a. Load all examples: {pred, answers: List[str], all_classes?: List[str]}
   b. For first-line-only datasets (trec, triviaqa, samsum, lsht):
         pred = pred.split("\n")[0].strip()
   c. score = mean over examples of:
         max(metric_fn(pred, gt, all_classes) for gt in answers)
   d. Write to CSV: dataset, num_examples, score (× 100)
3. Compute category averages across datasets
```

### Metric functions (longbench_metrics.py — vendored from THUDM/LongBench verbatim)

Do not modify these. They are kept identical to the published LongBench codebase
so results are directly comparable to the literature.

| Metric function | Datasets | Method |
|---|---|---|
| `qa_f1_score` (line 150) | hotpotqa, triviaqa, multifieldqa_en, 2wikimqa, musique | Token-level F1 after normalization |
| `qa_f1_zh_score` | multifieldqa_zh | Same, Chinese tokenization |
| `rouge_score` (line ~120) | gov_report, qasper, multi_news, vcsum, trec, samsum, lsht | ROUGE-L F1 |
| `rouge_zh_score` | vcsum, lsht | ROUGE-L on Chinese text |
| `classification_score` | trec, lsht | Match prediction in `all_classes` list |
| `retrieval_score` | passage_count, passage_retrieval_en | Paragraph ID from regex |
| `retrieval_zh_score` | passage_retrieval_zh | Same, Chinese |
| `count_score` | passage_count | Extract digit, compare |
| `code_sim_score` | lcc, repobench-p | Fuzzy string match on code |

**`normalize_answer()` (longbench_metrics.py:24):**
```python
lower → remove articles (a/an/the) → remove punctuation → collapse whitespace
```
Applied before all English QA metrics.

**`f1_score()` (longbench_metrics.py:139):**
```python
# Token-level F1 via Counter intersection
common = Counter(pred_tokens) & Counter(gold_tokens)
intersection = sum(common.values())
precision = intersection / len(pred_tokens)
recall    = intersection / len(gold_tokens)
f1 = (2 * precision * recall) / (precision + recall)  if denom > 0 else 0
```

**`qa_f1_score()` (longbench_metrics.py:150):**
```python
prediction = normalize_answer(prediction)
ground_truth = normalize_answer(ground_truth)
f1_score(prediction.split(), ground_truth.split())
```

---

## 5. Metric Quick Reference

| Suite | Metric | Range | Better |
|---|---|---|---|
| A | Jaccard (top-K window overlap) | [0, 1] | Higher |
| B | Cosine similarity | [-1, 1] | Higher |
| B | Pearson correlation | [-1, 1] | Higher |
| B | Spearman rank correlation | [-1, 1] | Higher |
| B | KL(ours ‖ base) | [0, ∞) | Lower |
| B | Mass ratio (base/ours) | (0, ∞) | ≈ 1 |
| B | Global LIR (rescue rate) | [0, 1] | Lower (stable) |
| B | Absolute missed mass | [0, ∞) | Lower |
| D | Dataset-specific (F1/ROUGE/exact) | [0, 100] | Higher |

---

## 6. End-to-End Evaluation Pipeline

```
CorpusLoader.load()
  └─► _load_wikitext103() or _load_pg19()
        └─► HuggingFace datasets
              └─► _split_into_articles()   [wikitext only]

CorpusLoader.sample_articles(n, seed)
  └─► random.Random(seed).sample() + sort

ArticleRegistry.register_article()
  └─► sha256(text) → identity record

BaseParityRunner.run()                     [Suite A base]
  └─► model.generate()  DynamicCache
        └─► attn_weights per step per layer
              └─► compute_window_scores()  scorer.py:17
                    └─► topk on evictable slice
                          └─► save NPZ (top_window_indices, window_scores, …)

OursParityRunner.run()                     [Suite A ours]
  └─► install_score_hooks()
        └─► model.generate()  WindowedCache
              └─► hooks → compute_window_scores() → cache.update() → eviction
                    └─► save NPZ (+ retained_window_ids, retained_window_scores)

FaithfulnessRunner.run()                   [Suite B]
  └─► load base NPZ + ours NPZ
        └─► _validate_alignment()
              └─► _compute_metrics()
                    ├─► jaccard_topk()        metrics.py:32
                    ├─► _cosine()             faithfulness_runner.py:33
                    ├─► _pearson()            faithfulness_runner.py:38
                    ├─► _spearman()           faithfulness_runner.py:45
                    ├─► _kl()                 faithfulness_runner.py:52
                    └─► mass ratio            faithfulness_runner.py:217
                          └─► save faithfulness_results.npz

LongBenchRunner.run()                      [Suite D]
  └─► model.generate() for each example
        └─► write predictions/<dataset>.jsonl

LongBenchScorer.run()                      [Suite D scoring]
  └─► score_predictions()                  longbench_scoring.py:78
        └─► dataset2metric.json → metric function dispatch
              └─► normalize_answer() + f1/rouge/exact  longbench_metrics.py
                    └─► write scores CSV
```
