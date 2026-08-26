# FlashInfer L-source for the prefill score path

## What this changes and why

Our prefill TTFT is ~4.5× FullKV-flash at the same shape (0.44 vs 0.10 s at
1048/1048, B=1), while SnapKV / KIVI sit ~1×. The cause is the H2O score pass,
and specifically that it costs **two** `O(N²)` passes per layer instead of one:

1. `score_kernel.compute_lse` — a full `matmul(q,kᵀ) → logsumexp` in naive
   PyTorch, materialising fp32 logit blocks.
2. `score_kernel._token_scores_triton` — the actual per-key column-sum.

Pass 1 is only skipped when the softmax normaliser `L` is reused from the real
attention forward. The historic reuse path (`flash_lse`) calls the installed
`flash_attn_func` with `return_attn_probs=True` and stashes `softmax_lse` — a
**fragile capture** that many flash-attn builds reject, silently latching off and
making every prefill layer recompute `L`.

`flashinfer_lse` replaces that with FlashInfer's prefill kernel, which returns
`L` as a **first-class output** (`return_lse=True`) — no probability-mask hack to
reject, no silent latch-off — and runs the prefill attention itself on
FlashInfer's (faster) kernel in the same pass. One pass now yields both the
attention output the model needs **and** `L`, eliminating `compute_lse`.

The score `q·kᵀ` column-sum pass (2) is inherent to H2O-style scoring — no flash
library exposes a column-sum of the attention matrix — so it stays. Matching
SnapKV's ~1× TTFT entirely would require observation-window scoring, which is a
different scoring rule (not done here).

## How to turn it on

Selected by `STICKYKV_LSE_BACKEND` (default `auto`):

| value        | behaviour                                                                    |
|--------------|------------------------------------------------------------------------------|
| `auto`       | use FlashInfer if importable, else the `flash_attn` capture, else recompute  |
| `flashinfer` | **require** FlashInfer — raise at hook install if it is not importable       |
| `flash`      | the historic `flash_attn_func(return_attn_probs=True)` capture               |
| (`STICKYKV_SCORE_LSE_FROM_FORWARD=0` forces the `compute_lse` recompute, as before) |

The once-per-process banner reports which is live:

```
[StickyKV] score path: FLASH ... | L-reuse: flashinfer
```

### Confirm it is actually working

A counter tracks how often the `compute_lse` recompute ran. On a healthy reuse
run it stays **0** across the whole prefill; a nonzero count means every prefill
layer paid the second `O(N²)` pass.

```python
from modules.windowed_cache import score_kernel
score_kernel.reset_lse_recompute_count()   # before the prefill
# ... run prefill ...
assert score_kernel.lse_recompute_count() == 0   # L-reuse held
```

## Environment — what you need to install

FlashInfer is **CUDA-only** (no CPU build). On the GPU box (this repo's pinned
`transformers==4.47.1` env):

```bash
# Match the wheel to your CUDA + torch. See https://docs.flashinfer.ai/installation.html
# Example (CUDA 12.4, torch 2.4+); pick the index URL for your CUDA/torch:
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
# or build from source:
#   pip install flashinfer --no-build-isolation
```

Requirements:
- NVIDIA GPU, **compute capability ≥ 8.0** (Ampere/Ada/Hopper — sm80+).
- A CUDA toolkit matching your torch build (for JIT/AOT kernel compilation).
- `transformers==4.47.1` (this repo's pinned API), torch ≥ 2.2.

`flash-attn` is still used for the attention on non-prefill / fallback paths, so
keep it installed. FlashInfer only takes over the `T > 1` prefill call; the
decode path stays on our fused two-tier int2 kernel (FlashInfer does not do
int2-in-register two-tier decode + score emission).

## Validate before trusting the numbers

This integration is authored on a CPU-only box and ships **GPU-unvalidated**,
like the Triton kernels. Run the parity test on the GPU first:

```bash
pytest tests/test_flashinfer_lse.py -q
```

It checks two things and tells you exactly what to fix:

1. **LSE base.** FlashInfer's LSE must equal `compute_lse` (natural-log
   logsumexp of the scaled logits). If the test reports the ratio is ~`1.4427`
   (`1/ln2`), the installed build returns **log2** LSE — set
   `STICKYKV_FLASHINFER_LSE_LOG2=1`.
2. **Output parity.** The attention output must match
   `scaled_dot_product_attention` to fp rounding (else the model degrades).

Other GPU-validation points (see the header of
`modules/windowed_cache/flashinfer_lse.py`):
- **API version** — targets the `plan`/`run` wrapper API (FlashInfer ≥ 0.2). If
  you are on 0.1.x (`begin_forward`/`forward`), the wrapper will latch to
  fallback (correct, but no speedup); upgrade FlashInfer.
- **Scale** — we pass `sm_scale = softmax_scale` from the model's own flash call;
  do not double-apply `head_dim**-0.5`.

If FlashInfer is not importable and you did **not** request it explicitly
(`auto`), everything degrades to the previous behaviour with no error.

---

# Further prefill optimization — the score pass itself

FlashInfer L-reuse removes the *second* `O(N²)` pass (`compute_lse`). What
remains is **one** score pass: the key-outer Triton kernel
(`score_kernel._score_kernel`) that computes `Σ_q exp(scale·q·kᵀ − L_q)` per key.
This pass is inherent — it is the exact H2O cumulative score (every query scores
every key), which the method requires. It is already cheaper than a full
attention pass (tensor-core `tl.dot`, triangular via causal-tile skip, and no
`p·V` second matmul), so with L-reuse working, prefill lands ~1.6× FullKV-flash
rather than the ~4.5× we started from.

## Score-kernel autotuning (`STICKYKV_SCORE_AUTOTUNE`, default ON)

The kernel is **key-outer**: each key block re-reads all of `Q`, so total `Q`
traffic is `⌈S/BLOCK_N⌉ × T × D`. The best `BLOCK_M`/`BLOCK_N`/`num_warps`/
`num_stages` — and in particular a larger `BLOCK_N`, which cuts the number of `Q`
re-reads — depends on the GPU and on `(S, T, D)`. `triton.autotune` benchmarks a
config set once per new shape key and caches the winner. The math is unchanged
(autotune only *selects* among correct configs), so the CPU torch reference stays
the oracle. Set `STICKYKV_SCORE_AUTOTUNE=0` to pin the historic fixed 64×64
launch (byte-stable timing, or if a Triton build's autotuner misbehaves).

**Warmup note:** the first prefill at each new shape benchmarks the config set,
so keep `num_warmup_runs ≥ 1` (the perf runner already does) to keep that cost
out of the measured runs.

## What is NOT worth doing (and why)

- **Fusing the score into the attention pass.** Tempting — the score is the
  column-sum of the same `p_qk` flash computes — but attention is *query-outer*
  (needs running softmax) while the exact column-sum is *key-outer* (needs the
  final per-query `L`). The loop orders conflict, so one kernel cannot do both
  efficiently; the two-pass split (FlashInfer attn+L, then key-outer score) is
  the right factorization given we already have `L`.
- **Observation-window scoring (SnapKV-style).** Rejected: it degrades as the
  window shrinks. The method keeps full cumulative scoring.
