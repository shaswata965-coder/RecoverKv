# StickyKV — Windowed KV-Cache

A modular, reproducible evaluation framework for windowed KV-cache eviction in large language models. Supports two attention backends (flash-attn-2 and eager) across multiple hardware targets.

## Quick Start

### 1. Create Environment

```bash
conda env create -f environment.yml
conda activate stickykv
```

### 2. (Optional) Install Flash Attention 2

Required only for the `flash_attn` backend. Needs CUDA 12.x and Ampere+ GPU (A100, H100).

```bash
pip install flash-attn --no-build-isolation
```

### 3. Run an Evaluation

```bash
# Base parity run (full cache, eager attention)
python main.py --config configs/eval_parity_base.yaml

# Ours parity run (windowed cache, eager backend)
python main.py --config configs/eval_parity_ours_eager.yaml

# Override config values from CLI
python main.py --config configs/base.yaml --override run.seed=123 data.prefill_len=200
```

### 4. Reproduce All Results

```bash
bash scripts/reproduce_all.sh
```

---

## Project Structure

```
StickyKV/
├── configs/             # YAML experiment configs (inherit from base.yaml)
├── data/                # Corpus loaders (wikitext-103, PG19) + article registry
├── models/              # Model-related code (placeholder)
├── modules/
│   ├── windowed_cache/        # Flash-attn-2 backend (Prompt 02)
│   ├── windowed_eager_cache/  # Eager-attention backend (Prompt 02)
│   └── evaluation/            # Runners: parity, faithfulness, perf, LongBench
├── scripts/             # Bash scripts for each evaluation suite
├── utils/               # Seed, logging, config, hashing, env capture
├── tests/               # Cross-cutting tests
├── main.py              # Single entry point (--config routing)
└── environment.yml      # Conda environment spec
```

---

## Hardware / Backend Matrix

| Backend        | `attn_implementation` | `cache.backend_package` | GPU Requirement      | Flash-Attn Required? |
|---------------|-----------------------|-------------------------|----------------------|---------------------|
| **Eager**      | `eager`               | `eager`                 | Any CUDA GPU (T4+)   | No                  |
| **Flash-Attn** | `flash_attention_2`   | `flash_attn`            | Ampere+ (A100/H100)  | Yes                 |
| **Baseline**   | `eager`               | `null` (DynamicCache)   | Any CUDA GPU (T4+)   | No                  |

### Kaggle T4 Compatibility

The eager backend runs on Kaggle's free T4 GPUs without flash-attn installed:

```bash
python main.py --config configs/eval_parity_ours_eager.yaml
```

### A100/H100 Setups

Use the flash-attn backend for maximum throughput:

```bash
python main.py --config configs/eval_parity_ours_flash.yaml
```

---

## Evaluation Suites

| Suite            | Mode              | Config                          | Description                                      |
|-----------------|-------------------|---------------------------------|--------------------------------------------------|
| **Parity A**     | `parity_base`     | `eval_parity_base.yaml`        | Full-cache baseline generation                    |
|                  | `parity_ours`     | `eval_parity_ours_*.yaml`      | Windowed-cache generation (both backends)         |
| **Faithfulness B**| `faithfulness`   | `eval_faithfulness.yaml`       | Compares base vs ours npz outputs                 |
| **Perf C**       | `perf`           | `eval_perf.yaml`               | Throughput/latency benchmarks                     |
|                  | `perf`           | `eval_efficiency.yaml`         | Same, under the published comparison protocol     |
| **LongBench D**  | `longbench`      | `longbench_*.yaml`             | Downstream benchmark (16 English datasets)        |
| **Visualization**| `visualize`      | `eval_visualize.yaml`          | Plots from telemetry npz files                    |

---

## Reproducibility

Every output file includes a `.meta.json` sidecar with:
- Schema version, seed, dataset/article identity, tokenizer SHA
- Full window config (window_size, sink tokens, local window, top_k)
- Model name, revision, dtype, attention implementation
- Cache backend + package, budget ratio
- Library versions (transformers, torch, flash-attn, CUDA)
- GPU name/memory, git commit SHA
- UTC timestamps (start/finish)

### Deterministic Execution

All scripts enforce:
```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

And `seed_everything()` sets:
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

---

## Testing

```bash
# Run all CPU tests (fast, <30s)
pytest tests/ data/ modules/evaluation/ -v -m "not gpu"

# Run GPU tests (requires model loads)
pytest -v -m gpu
```

---

## Configuration

Configs use YAML with single-level inheritance via `_base_`:

```yaml
# configs/my_experiment.yaml
_base_: base.yaml

run:
  mode: parity_ours
  seed: 123

cache:
  backend: windowed
  backend_package: eager
  cache_budget: 0.40
```

---

## WindowedCache Usage

```python
from utils.cache_factory import get_cache_classes, validate_backend_attn_pairing

# --- Flash-attn backend (A100/H100) ---
backend, attn_impl = "flash_attn", "flash_attention_2"
validate_backend_attn_pairing(backend, attn_impl)
CacheClass, ConfigClass, install_hooks = get_cache_classes(backend)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn_impl)
cfg = ConfigClass(window_size=8, num_sink_tokens=4, local_window_size=0.25, cache_budget=0.40)
cache = CacheClass(cfg, prefill_len, model.config, torch.float16, model.model.rotary_emb, model.config.num_hidden_layers)
handles = install_hooks(model, cache, cfg)
try:
    output = model.generate(input_ids, past_key_values=cache, max_new_tokens=50)
finally:
    handles.remove()

# --- Eager backend (Kaggle T4) ---
backend, attn_impl = "eager", "eager"
validate_backend_attn_pairing(backend, attn_impl)
CacheClass, ConfigClass, install_hooks = get_cache_classes(backend)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn_impl)
cfg = ConfigClass(window_size=8, num_sink_tokens=4, local_window_size=0.25, cache_budget=0.40)
cache = CacheClass(cfg, prefill_len, model.config, torch.float16, model.model.rotary_emb, model.config.num_hidden_layers)
handles = install_hooks(model, cache, cfg)
try:
    output = model.generate(input_ids, past_key_values=cache, max_new_tokens=50, output_attentions=True)
finally:
    handles.remove()
```

---

## Architecture Decisions

See the locked algorithmic decisions in the project specification:
- Cache layout: `[sink | evictable windows | local windows]`
- H2O-style cumulative attention scoring
- Byte-based budget computation (GQA-aware)
- RoPE rerotation using model's own `apply_rotary_pos_emb`
- Two backends sharing identical public APIs
- Vectorized hot paths (no Python loops over batch/heads/tokens)

---

## Out of Scope (v1)

- Quantization integration
- Cross-layer score sharing
- Speculative decoding
- Multi-GPU cache sharding
- Beam search (`reorder_cache` raises `NotImplementedError`)
- Chinese LongBench subsets
- LongBench v2

---

## License

Research use only. See LICENSE for details.
