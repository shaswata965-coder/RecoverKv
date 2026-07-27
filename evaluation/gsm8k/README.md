# GSM8K for kvpress presses (SnapKV / AdaKV / DefensiveKV)

Chain-of-thought GSM8K evaluation that produces numbers you can act on: a corrected
answer extractor, a pre-flight budget gate, and compression measured off the cache
rather than assumed from the requested ratio.

Run everything from `evaluation/` so that `gsm8k.*` and `kvpress` are both importable —
the same convention `evaluate.py` uses for `longbench.*` / `ruler.*`.

## Quick start

```bash
cd evaluation

# 1. Build the dataset once. Every run must read this exact directory.
python -m gsm8k.create_huggingface_dataset --out data/gsm8k_cot

# 2. Full-cache baseline (the reference every press is measured against)
python -m gsm8k.run_gsm8k \
    --model /models/Meta-Llama-3.1-8B-Instruct \
    --press_name none \
    --data_dir data/gsm8k_cot \
    --output_dir results/gsm8k/full_cache

# 3. A press
python -m gsm8k.run_gsm8k \
    --model /models/Meta-Llama-3.1-8B-Instruct \
    --press_name defensivekv --compression_ratio 0.5 \
    --data_dir data/gsm8k_cot \
    --output_dir results/gsm8k/defensivekv_cr0.5

# 4. Compare
python -m gsm8k.calculate_metrics --runs results/gsm8k/*/ \
    --out_csv results/gsm8k/comparison.csv
```

Or the whole thing in one command:

```bash
bash gsm8k/run_gsm8k_e2e.sh                      # full sweep
SAMPLES=20 bash gsm8k/run_gsm8k_e2e.sh           # ~5-minute smoke
PRESSES="snapkv defensivekv" RATIOS="0.5" bash gsm8k/run_gsm8k_e2e.sh
```

## Presses

| `--press_name` | Class | Head-wise | Notes |
|---|---|---|---|
| `none` | — | — | Full-cache control. Required before any press run is interpretable. |
| `snapkv` | `SnapKVPress` | no | Uniform budget per head. |
| `adakv` | `AdaKVPress(SnapKVPress())` | selection only | Mask-based simulation — no real memory saving, warns per layer. |
| `efficient_adasnapkv` | `EfficientAdaSnapKVPress` | yes | Real head-wise; flattened cache + varlen flash attn. |
| `defensivekv` | `EfficientDefensiveKVPress` | yes | Per-layer adaptive allocation. |
| `layer_defensivekv` | `EfficientLayerDefensiveKVPress` | yes | Cross-layer global allocation. |
| `streaming_llm` | `StreamingLLMPress` | no | Position-only reference; no observation window. |

Aliases accepted: `full_cache`/`baseline` → `none`, `adasnapkv` → `adakv`,
`efficient_defensivekv` → `defensivekv`, `efficient_layer_defensivekv` → `layer_defensivekv`.

`compression_ratio` follows the DefensiveKV convention: **fraction removed**. `0.8`
keeps 20%.

## Three things to read before the accuracy

### 1. What the ratio acts on (`--compress_questions`)

kvpress compresses **only `context`**, then feeds `question` uncompressed. The GSM8K
prompt splits naturally into a constant ~55-token system prompt and the problem
statement, so the split decides what the knob touches:

* `--compress_questions True` (**default**) merges the problem into the compressed
  context. The ratio acts on the problem. This is what you want.
* `--compress_questions False` reproduces the stock layout, where the ratio acts on ~55
  tokens of boilerplate and **the problem is never compressed**. Accuracy is then nearly
  invariant to the ratio, and reporting it as "GSM8K under compression" would be wrong.

The comparison table refuses to put runs with different settings in the same table.

### 2. The budget can collapse into the observation window

Every score-based press here force-keeps its last `window_size` (default 32) tokens at
max score. GSM8K contexts are ~150 tokens, so:

```
n_kept = int(150 * (1 - 0.8)) = 29   <   window_size = 32
```

At that point the entire budget *is* the window, top-k has no free slots, and SnapKV,
AdaKV and DefensiveKV all reduce to "keep the last k tokens" — the run cannot
distinguish them. Worse, DefensiveKV's CriticalKV stage-1 clamp
(`count.clamp_(max=int(q_len*(1-cr) - window_size))`) goes **negative**, and
`sorted_indices[b, h, :k]` with negative `k` selects everything *except* the last `|k|`.
The defensive mechanism inverts, silently.

`run_gsm8k.py` computes this before any GPU work and **refuses to start**, naming the
fix. To go above ~0.75, shrink the window too:

```bash
python -m gsm8k.run_gsm8k ... --compression_ratio 0.8 --press_window_size 8
```

`--allow_degenerate True` records the cell anyway; it is flagged in `meta.json` and in
the table.

### 3. The nominal ratio is not the end-to-end saving

kvpress compresses the context **once, at prefill**. The question tokens and every
generated token are appended uncompressed and never pruned. On a 150-token prompt with
a 200-token CoT answer, `compression_ratio=0.8` removes 121 of 350 KV entries — a **34%**
end-to-end saving, not 80%.

`meta.json` reports `mean_effective_retention` and the table prints it as `effRet`.
Quote that, not the nominal ratio.

## Outputs

```
results/gsm8k/<run>/predictions.jsonl   one record per example
results/gsm8k/<run>/meta.json           identity, config, compression diagnostic
results/gsm8k/comparison.csv            one row per run
```

Each `predictions.jsonl` record carries `context_tokens`, `gen_tokens`, `stop_reason`,
`n_kept`, `free_slots`, `degenerate`, `measured_context_compression` and
`effective_retention`, so a run is auditable after the fact without re-running it.

## Scoring

`calculate_metrics.py` reports three numbers:

* `accuracy` — headline; failures and blank predictions are **excluded**, never scored
  as wrong.
* `accuracy_strict` — same denominator, but only answers backed by an explicit marker
  (`####`, `Answer:`, `\boxed{}`) count. If this diverges from `accuracy`, or if the gap
  moves with the compression ratio, the curve is measuring output format.
* `marker_rate` — share of generations that emitted a marker at all. Below 90% the
  number is being carried by the last-number-in-text fallback and is not trustworthy.

It is signature-compatible with the other scorers (`calculate_metrics(df) -> dict`), so
it can be dropped into `evaluate.py`'s `SCORER_DICT` if you want that entry point too —
though you would lose stop strings and the budget gate.

The extractor exists because the obvious ladder (first `Answer:` match, else last number
anywhere) makes accuracy track *how generation terminated* rather than whether the math
was right. Holding arithmetic constant at 100% correct and varying only the run-on rate,
that ladder reports 100/90/75/50/25/0 for run-on rates of 0/10/25/50/75/100 percent. KV
compression perturbs verbosity, so the error term moves with the ratio — which is how
you get "every method degrades past ratio X".

## Known limitation: batch size 1

Everything runs at `B=1`. Decode dominates GSM8K and is weight-bound at that batch size,
so this is the obvious speed lever — but it is blocked in the library, not here:

* `kvpress/ada_attn.py:219,373,696` — `assert bsz == 1` in the varlen head-wise
  attention. Blocks `defensivekv`, `layer_defensivekv`, `efficient_adasnapkv`.
* `efficient_ada_global_scorer_press.py:86,174` — two more asserts, plus a **global
  top-k that flattens the batch away**. At `B>1` sequences would compete for each
  other's budget, making per-row retention depend on batch composition. That one is a
  semantic redesign, not an assert to delete.
* The window-attention scorers build a purely causal mask with no padding term
  (`snapkv_press.py:68` and both DefensiveKV variants), and the pipeline never passes an
  `attention_mask`. Left-padded batches would give pad keys real softmax mass and let
  them win retention slots.

If batching is added, prefer **equal-length bucketing over padding**: GSM8K prompts
cluster tightly, so exact-length groups need no padding, no mask changes, and make the
batched path bit-identical to `B=1` — which is what makes it verifiable.

## Tests

```bash
cd evaluation && pytest gsm8k/test_gsm8k.py -q     # 52 tests, no GPU, no network
```
