# Kaggle Runbook — both metric suites, end to end

Clone → run → formatted results, for a **corpus you attach** and a **model
already on Kaggle**. One pair of model runs feeds both suites:

| Suite | Question | Outputs |
|---|---|---|
| **A/B — Faithfulness** | Does our cache pick and score the same windows as the oracle? | `faithfulness_results.npz`, `_per_layer.csv`, `_summary.md` |
| **E — QEvict observations** | Is the design premise true (skew / window-level decisions / revival)? | 9 CSVs, `paper_results.md`, `all_results.json`, `qevict_observations.npz`, 3 PDFs |

The expensive part is the two parity runs (steps 4–5). Both suites are then
pure CPU post-processing over their npzs — seconds, re-runnable with different
knobs without touching the GPU.

---

## Before you start — four things that will bite

1. **`transformers` must be ≤ 4.47.1.** The windowed cache keeps survivors at
   their original RoPE positions and relies on `cache_position` advancing
   monotonically; newer transformers re-derive it from the shrunken cache and
   silently corrupt the RoPE phase after the first eviction. `OursParityRunner`
   refuses to run rather than emit wrong numbers. Kaggle ships something much
   newer, so **cell 2 pins it** — which needs Internet **ON** for that cell.
2. **Attention memory, not weights, is the binding constraint.** The base run
   needs `output_attentions=True`, which materialises `[B, H, S, S]` per layer.
   At `L=32, H=32, prefill=2048, fp16` that is ~8.6 GB of returned attention on
   top of the weights. Size with the table in step 4.
3. **Documents shorter than `prefill_len` are silently truncated-to-shorter**,
   which makes window geometry differ per sample. `parity.min_article_tokens`
   exists in the config but **no runner reads it** — cell 3 filters for you.
4. **`num_samples` is your statistical axis.** With one document, both suites'
   confidence intervals are over *layers of one prompt* — within-run
   variability, not population CIs. Use ≥ 8 documents for anything reported.

---

## Notebook setup

- **Accelerator:** GPU T4 ×2 (preferred — `device_map="auto"` shards across
  both) or P100.
- **Internet:** ON for cell 2. Turn it off afterwards if you like; nothing
  later needs it.
- **Add data:** your corpus dataset, and the model (Kaggle Models or a dataset
  containing the HF folder).

---

### Cell 1 — Clone the repo

```python
!git clone --depth 1 --branch int4_qwen \
    https://github.com/shaswata965-coder/StickyKV.git /kaggle/working/StickyKV
%cd /kaggle/working/StickyKV
!git log --oneline -3
```

### Cell 2 — Pin the environment (needs Internet ON)

```python
!pip install -q "transformers==4.47.1" "tokenizers>=0.21,<0.22" "accelerate>=0.26" "matplotlib>=3.7"
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(), torch.cuda.device_count())
```

`matplotlib` is only needed for the Suite E figures — every table and JSON is
written without it. **Restart the session** if pip replaced an already-imported
package, then re-run cell 1's `%cd`.

### Cell 3 — Point at your corpus and model, and filter short docs

`parity.dataset` accepts a **path** as well as `wikitext-103`/`pg19`. Layouts:
`.jsonl`/`.ndjson` (one record per line), `.json` (array), `.txt` (one article),
or a directory of those walked in sorted order. The text field is auto-detected
from `text`/`content`/`document`/`article`/`body`/`input`/`prompt`/`context`.

```python
import json, os
from pathlib import Path
from transformers import AutoTokenizer

# ── EDIT THESE TWO ──────────────────────────────────────────────────────
RAW_CORPUS = "/kaggle/input/<your-dataset>/docs.jsonl"   # file or directory
MODEL_DIR  = "/kaggle/input/<your-model>/transformers/<variant>/<version>"
# ────────────────────────────────────────────────────────────────────────

PREFILL, GEN, SAMPLES = 1024, 256, 8      # see the sizing table in step 4
WORK = Path("/kaggle/working/outputs"); WORK.mkdir(parents=True, exist_ok=True)

assert Path(MODEL_DIR).exists(), f"model not found: {MODEL_DIR}"
print("model files:", sorted(p.name for p in Path(MODEL_DIR).glob("*"))[:8])

# Keep only documents long enough to fill the prefill — otherwise the run
# silently uses a shorter prefill for those samples and the geometry drifts.
from data.corpus_loader import CorpusLoader
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
docs = CorpusLoader(RAW_CORPUS).load()
long_enough = [d for d in docs
               if len(tok(d, add_special_tokens=True).input_ids) >= PREFILL]
print(f"{len(long_enough)}/{len(docs)} documents have >= {PREFILL} tokens")
assert len(long_enough) >= SAMPLES, (
    f"need >= {SAMPLES} documents of >= {PREFILL} tokens; "
    f"lower PREFILL/SAMPLES or supply more text")

CORPUS = str(WORK / "corpus_filtered.jsonl")
with open(CORPUS, "w", encoding="utf-8") as f:
    for d in long_enough[:max(SAMPLES, 50)]:
        f.write(json.dumps({"text": d}) + "\n")
print("filtered corpus →", CORPUS)
```

### Cell 4 — Suite A base run (full cache, the ground truth)

Sizing — measured shape, `output_attentions` memory only (add the weights):

| layers × heads | prefill 512 | 1024 | 2048 |
|---|---|---|---|
| 32 × 32 (8B) | 0.5 GB | 2.1 GB | 8.6 GB |
| 16 × 16 (1–3B) | 0.13 GB | 0.5 GB | 2.1 GB |

An 8B model in fp16 is ~16 GB of weights, so `prefill=2048` does **not** fit a
single 16 GB T4 — use T4 ×2, drop to `prefill=1024`, or use a smaller model.

```python
%env PYTHONHASHSEED=0
%env TOKENIZERS_PARALLELISM=false

GEOM = (
    f"model.name={MODEL_DIR} model.revision=none model.dtype=float16 "
    f"parity.dataset={CORPUS} data.dataset={CORPUS} "
    f"parity.prefill_len={PREFILL} parity.gen_len={GEN} "
    f"data.prefill_len={PREFILL} data.gen_len={GEN} data.num_samples={SAMPLES} "
    f"window.window_size=32 window.num_sink_tokens=4 window.local_window_size=256 "
    f"cache.window_size=32 cache.num_sink_tokens=4 cache.local_window_size=256 "
    f"cache.cache_budget=0.25 "
    f"telemetry.output_dir={WORK}"
)
BASE_NPZ = f"{WORK}/parity_base.npz"
OURS_NPZ = f"{WORK}/parity_ours.npz"

!python main.py --config configs/eval_parity_base.yaml \
    --override {GEOM} output_path={BASE_NPZ}
```

`model.revision=none` matters: the configs pin `revision: main`, which is a hub
concept and meaningless for a local directory.

> **Geometry constraints** (enforced, fail-fast): with `quant_ratio > 0` the
> int4 tier needs an even `window_size` (nibble packing, 2 codes per byte;
> `head_dim` is always even, so it never binds). An integer `local_window_size`
> must be a multiple of `window_size`; a float in `(0, 1]` is a ratio instead.
> `window_size=32` suits an 8B model; 8 or 16 is more sensible for a small one.

### Cell 5 — Suite A ours run (two-tier cache, teacher-forced from base)

```python
!python main.py --config configs/eval_parity_ours_eager.yaml \
    --override {GEOM} cache.quant_ratio=0.5 \
    base_run_npz={BASE_NPZ} output_path={OURS_NPZ}
```

`quant_ratio=0.5` splits the retained band fp16/int4. Use `0.0` for a
single-tier baseline; the tier-aware metrics then collapse to the legacy
numbers by construction. Both runs must share every geometry field — both
suites hard-fail on a mismatched pair rather than silently comparing two
different experiments.

### Cell 6 — Suite A/B: faithfulness scores

```python
!python main.py --config configs/eval_faithfulness.yaml --override \
    faithfulness.base_npz_path={BASE_NPZ} \
    faithfulness.ours_npz_path={OURS_NPZ} \
    telemetry.output_dir={WORK} output_path={WORK}/faithfulness_results.npz

from IPython.display import Markdown, display
display(Markdown(open(f"{WORK}/faithfulness_results_summary.md").read()))
```

Console report with per-layer scorecards, generation-trend quartiles and layer
rankings:

```python
%env FAITHFULNESS_NPZ=/kaggle/working/outputs/faithfulness_results.npz
!python scripts/print_faithfulness.py
```

### Cell 7 — Suite E: QEvict observations

```python
!python -m modules.evaluation.qevict_observations \
    --base-npz {BASE_NPZ} --ours-npz {OURS_NPZ} \
    --output-dir {WORK}/qevict_observations \
    --fmm-horizon 32 --pool-factor 2 \
    --primary-inactivity 4 --primary-lir-horizon 8 --primary-transition-delta 1 \
    --trace-axis sample --bootstrap-samples 2000 --seed 42

display(Markdown(open(f"{WORK}/qevict_observations/paper_results.md").read()))
```

Knobs worth knowing:

| flag | meaning |
|---|---|
| `--fmm-horizon H` | how far ahead a routing decision is judged. Events with fewer than `H` future steps are **censored**, not truncated — `events_scored` shows how many survive, so keep `H` well under `GEN`. |
| `--trace-axis sample` | bootstrap over documents (what you want with `SAMPLES ≥ 8`). `sample_layer` bootstraps over layers — only defensible for a single document. |
| `--primary-inactivity m` / `--primary-lir-horizon H` | the revival operating point. Needs `m - 1 + H < R` (routing events) or every episode is censored and LIR is NA. |
| `--pool-factor 2` | coarse-vs-fine decision granularity. For a true **token**-vs-window comparison, re-run cell 4 with `window.window_size=1` and pass `--pool-factor 32`. |
| `--layer-stride 4` / `--max-samples N` | memory/time relief on a constrained box. |

### Cell 8 — Collect the outputs

```python
import shutil, glob
for f in sorted(glob.glob(f"{WORK}/**/*", recursive=True)):
    print(f.replace(str(WORK) + "/", ""))
shutil.make_archive("/kaggle/working/stickykv_results", "zip", WORK)
print("\n→ /kaggle/working/stickykv_results.zip")
```

Add `/kaggle/working` as a notebook output, or push the zip to a Kaggle
dataset, then pull it down for the HPC-scale comparison.

---

## What you get

```
outputs/
  parity_base.npz  .meta.json            ground truth (schema 1.2: step_window_scores)
  parity_ours.npz  .meta.json            two-tier run + tier tags per window
  faithfulness_results.npz               Suite A/B arrays
  faithfulness_results_summary.md        headline table  ← read this
  faithfulness_results_per_layer.csv     14 metrics × layer
  qevict_observations/
    paper_results.md                     Suite E report + manuscript sentences  ← read this
    observation1_mass_table.csv          C(p) at p ∈ {5,10,20,40}% + CIs
    observation1_coverage_table.csv      windows needed for {50,80,90}% of mass
    observation1_curve.csv               the full concentration curve + band
    observation2_summary_table.csv       FMM + churn per policy, units kept/dropped
    observation2_paired_comparison.csv   paired reductions with CIs
    observation3_lir_grid_oracle.csv     revival rate over (m, H)
    observation3_lir_grid_policy_fp.csv  same, for our real fp tier
    observation3_transition_table_*.csv  P01/P10 per lag
    observation3_quantifiable_result_*.csv
    observation3_primary_episodes_*.csv  one row per inactive episode
    all_results.json                     everything, machine-readable
    qevict_observations.npz .meta.json   raw arrays + provenance SHAs
    observation{1,2,3}_*.pdf             figures
```

**Do not compare the two LIR numbers.** Suite A/B's `global_lir` counts every
`(event, window)` lookback pair with an unbounded rescue window and `m=3`
hard-wired; Suite E's episode LIR counts each inactive run once, bounds the
rescue by `H`, and right-censors. Different estimators of related quantities.

---

## Scaling to HPC

Same repo, same commands — only the geometry grows. The bundled script does the
whole chain:

```bash
PROFILE=hpc MODEL=/path/to/model CORPUS=/path/to/corpus.jsonl bash scripts/run_qevict_observations.sh
```

`PROFILE=hpc` is prefill 2048 / gen 1024 / 8 samples; `STAGE=observe` re-runs
only the analysis over existing npzs. Every knob is an env var — see the header
of [`scripts/run_qevict_observations.sh`](scripts/run_qevict_observations.sh).
For the faithfulness suite on the same pair, `bash scripts/run_faithfulness.sh`
with the two npz paths overridden.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ConfigValidationError: transformers 4.5x.x is newer than 4.47.1` | Cell 2 didn't take. Re-run it, restart the session, `%cd` back. |
| `CUDA out of memory` during the base run | `output_attentions` at your prefill. Halve `PREFILL`, use T4 ×2, or a smaller model. |
| `ParityValidationError: ... article_sha` | Cells 4 and 5 saw different corpora or geometry. Re-run both with the same `GEOM`. |
| `Batched parity requires equal-length prefills` | Documents shorter than `PREFILL`. Re-run cell 3's filter, or set `data.batch_size=1`. |
| `ours npz is missing 'all_window_tier'` | Ours npz predates schema 1.2. Re-run cell 5. |
| `no 'step_window_scores' … falling back to differencing` | Base npz predates schema 1.2. Re-run cell 4 — differencing fp16 cumulative scores is noise past a few hundred decode steps. |
| Suite E reports `units dropped = 0` | Nothing was evicted: `N_q` covers the whole band at this geometry. Lower `cache.cache_budget` or `cache.quant_ratio`, or raise `PREFILL`. |
| Global LIR is `NA` | Every episode right-censored: needs `m - 1 + H < R`. Lower `--primary-inactivity`/`--primary-lir-horizon`, or raise `GEN`. |
| `cannot tell which field holds the article text` | Ambiguous JSON records. Pass an explicit field: `CorpusLoader(path, text_field="body")` in cell 3. |
| Figures missing, tables present | `matplotlib` absent — cell 2 installs it; harmless otherwise. |
