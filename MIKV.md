# MiKV on Llama

Yang et al., *No Token Left Behind: Reliable KV Cache Compression via
Importance-Aware Mixed Precision Quantization*, arXiv:2402.18096 (2024).

**Status, 2026-09-25: implemented end to end and CPU-tested, never run on a GPU
or on Llama-3.1-8B.** No number in this file is a Llama-3.1-8B number. Every
claim below is marked *measured (CPU)*, *arithmetic*, or *not measured*, as
`DISTANCE_TO_GOAL.md` §4 asks.

---

## 1. Why this baseline matters here

MiKV is the closest published relative of this repository's method. Both keep
an exact tier and a low-precision tier and rank tokens by accumulated
attention. Both keep, at low precision, what an eviction cache would drop. The
differences are exactly the things this repository exists to argue:

| | MiKV | RecoverKV |
|---|---|---|
| unit of selection | token, per KV head | window of `ws` tokens |
| low tier | every non-important token, INT2-4, **nothing evicted** | the next `N_q` windows at int2; the rest **dropped** |
| read path | every token, every step | rank-1 card gate, a fraction of windows |
| outliers | channel balancer (Eq. 2) | anchored residual cards; per-channel key grid |

So MiKV at a matched **byte** budget answers the question the other two
baselines (int2 KIVI, QEvict; `DISTANCE_TO_GOAL.md` §7) cannot: *is it worth
dropping tokens at all, or should every token be kept at low precision?*

## 2. What the paper specifies, and where it lives

| paper | here |
|---|---|
| Eq. 1: per-token asymmetric N-bit RTN, `alpha = (max-min)/(2^N-1)`, `beta = min` | `modules/mikv/quant.py:quantize` |
| §3.2: group size = half the head dimension ("mitigates the artifact of RoPE") | `MiKVConfig.group_size = None` → `D/2` |
| Eq. 2: `b = sqrt(max\|q\| / max\|k\|)` per (layer, head, channel), at prefill | `quant.channel_balancer`, `MiKVCache._prefill` |
| Eqs. 3-4: `k_hat = I(k*b)`, `q_hat = q/b` | `_TokenStore.encode` / `.decode` (see §3.1) |
| §3.1: evicted KVs retained at INT2/3/4 | `lo_bits` ∈ {2, 3, 4, 8} |
| §3.3: importance cache at FP16/INT8/INT4 | `hi_bits` ∈ {16, 8, 4} |
| importance policy: H2O (heavy hitters + recent) | `recent_ratio` split; scores from the captured query |
| baselines: H2O eviction, RTN | `lo_bits=0`; `importance_ratio=0` |
| cache size (Tables 1-3) | `config.paper_cache_size` — reproduces Table 3's 33/23/18/16% exactly (tested) |

Table 3's sizes fall out *only* with an fp16 `alpha` and an fp16 `beta` per
64-element group: 0.2·16 + 0.8·(2 + 32/64) = 5.2 bits = 32.5%. That is how
this implementation stores the grid. *Arithmetic.*

## 3. Decisions the paper leaves open, and what was chosen

### 3.1 The balancer divides the dequantized key, not the query

`sum_c (q_c / b_c) k_hat_c == sum_c q_c (k_hat_c / b_c)` exactly. Applying `1/b`
on the way *out* of the key store keeps the model's attention untouched, so
eager, SDPA and flash-attention-2 all work unmodified. Tokens stored at 16 bits
are stored exact and never balanced: the balancer shares a *quantizer's*
burden, and they have none. That is also what makes the control arm exact.

### 3.2 GQA

The paper's GQA models (Llama-2-70B, Mistral-7B) are not described at this
resolution. Here, a KV head's importance is the **sum** over the query heads
sharing it, and its balancer uses the **max** `|q|` over those heads. Selection
is per (row, KV head), as H2O's is per head.

### 3.3 The importance ratio holds at every step

`importance_ratio = r` means the importance cache holds `floor(r·S)` of the `S`
tokens seen, **recomputed every step**, so a generated token demotes an old one
only as often as the ratio requires. H2O's real-drop code instead freezes
`hh_size + recent_size` at prefill. The paper's reported cache sizes are shares
of the whole sequence, and only the per-step reading makes `r = 1.0` the
full cache. With it frozen, `r = 1.0` would still demote a token per step.
*Tested: `r = 1.0` is token-identical to `DynamicCache` on eager and SDPA.*

`cache_budget` (this repository's byte knob) reduces to the same thing: the `r`
at which codes + grids + exact tokens are that fraction of fp16 K+V. Since MiKV
keeps every token, that fraction holds at every length.

### 3.4 Scores are the attention the model computed

`Cache.update` receives keys and values but not the query. A forward hook on
`q_proj` captures the model's own query, and `update` re-applies RoPE from the
`cos`/`sin` HF passes it. *Tested: prefill + decode scores match eager
`attn_weights` column sums to 1e-5.* A missing hook raises
`MiKVHooksMissing`. Scoring without the query would demote tokens by nothing
and still look like MiKV.

### 3.5 Prefill attends exactly, and demotion happens after the step's scores

The prompt's own K/V are returned untouched. Tiers are formed from them
afterwards: scores over every prompt row, then the balancer, then selection,
then quantization. At decode, the step reads the victim before demoting it, as
H2O does. No prompt token is quantized twice.

### 3.6 Everything else

* Positions are never renumbered; the dense read is in position order, so HF's
  causal and padding masks index it unchanged. Left padding is tested to be
  invisible to scores and balancer.
* The balancer is per batch row (a "dynamic" balancer per request).
* A demoted token is never promoted: its exact value is gone.
* Beam search (`reorder_cache`) raises.

## 4. What it costs

Per token, per layer, per KV head, Llama-3.1-8B (`D = 128`, fp16). *Arithmetic.*

| | K+V payload | bookkeeping | total vs fp16 |
|---|---|---|---|
| fp16 token | 512 B | — | — |
| INT2 token (g=64) | 80 B | 4 B (int32 position) | |
| importance token | 512 B | 12 B (int64 position + fp32 score) | |
| **paper point, r=0.20** | 166.4 B (**32.5%**) | 5.6 B | **33.6%** |
| **bytes-matched, `cache_budget=0.20`** (r=0.052) | 102.4 B (**20.0%**) | 4.4 B | **20.9%** |
| RTN INT2 floor, r=0 | 80 B (15.6%) | 4 B | 16.4% |

`memory_report()` counts every tensor (tested byte for byte), with
`kv_fraction` being the paper's accounting and `total_fraction` including
bookkeeping.

**Decode latency is not a MiKV claim.** Each step dequantizes the whole cache
into one layer's dense K/V and hands it to the model's attention: a transient,
never stored, but a full-cache read and write per layer per step. The paper's
§3.4 only sketches kernels, and neither it nor this implementation ships one.
The prefill score pass is O(L²) per layer (H2O's is too), done in row blocks of
`score_chunk_bytes`.

## 5. Running it

```bash
# Unit + integration tests (CPU, ~1 min)
python -m pytest tests/test_mikv.py tests/test_mikv_runners.py -q

# The paper's Line Retrieval, every arm through one cache
python scripts/mikv_line_retrieval.py --model meta-llama/Llama-3.1-8B-Instruct \
    --samples 200 --n-lines 20

# Quality: GSM8K at the paper point, LongBench at this repo's 20% byte budget
python main.py --config configs/gsm8k_mikv.yaml
python main.py --config configs/longbench_mikv.yaml

# Perf: full cache, MiKV control arm, paper point, bytes-matched
python main.py --config configs/eval_perf_mikv.yaml
```

All three quality runners take `cache.backend: external` with
`cache.method_factory: "modules.mikv:make_mikv_method"` and the MiKV knobs in
`cache.method_kwargs`. Their sidecars record `cache_type: external`, the
resolved ratio, and the mean measured `kv_fraction`. `read_gate` is
`not-expected`, since there is no windowed Q tier. The perf suite reaches the
same factory through its existing `cache_backend: external` seam.

Arms, via `--override cache.method_kwargs.<k>=<v>`:

| arm | kwargs |
|---|---|
| control (full cache through MiKV's code) | `importance_ratio=1.0` |
| H2O (the paper's eviction baseline) | `lo_bits=0` |
| RTN (uniform per-token INT2) | `importance_ratio=0.0` |
| no outlier awareness (Table 2) | `channel_balancer=false` |
| INT8 / INT4 importance cache (Table 3) | `hi_bits=8` / `hi_bits=4` |

## 6. What has been measured

*Measured (CPU)*, `tests/test_mikv.py` + `tests/test_mikv_runners.py`, 49 tests:
the quantizer is Eq. 1 on the stored grid at every width; packing is exact
(int3 costs 3 bits); the balancer preserves logits and cuts INT2 logit RMSE
under synthetic query outliers; `r = 1.0` equals `DynamicCache` token for token;
scores equal eager attention; tiers partition the sequence and the recent
window is never demoted; the dense read is `I(k·b)/b` bit for bit; error falls
monotonically with `lo_bits` and with `r`; the GSM8K runner runs MiKV end to end
and its sidecar says so.

*Measured (CPU)*, Line Retrieval on a small Llama-architecture model (§7), and
the perf seam running all four MiKV rows (§8).

*Not measured*: anything on a GPU; anything on Llama-3.1-8B; GSM8K, LongBench,
RULER or perf numbers for MiKV.

## 7. Line Retrieval on SmolLM2-1.7B-Instruct (CPU)

A Llama-architecture model small enough for CPU (24 layers, 32 heads,
`D = 64`, so groups of 32 and 3 bits per INT2 element). fp32, SDPA, 30 samples
× 20 lines (~570 prompt tokens), `importance_ratio = 0.2`, seed 0:

```bash
python scripts/mikv_line_retrieval.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --device cpu --dtype float32 --samples 30 \
    --arms full,h2o,rtn2,mikv2-nobal,mikv2,mikv4,mikv2-hi8
```

| arm | what happens to H2O's victims | accuracy | KV size (paper accounting) |
|---|---|---|---|
| full | — (`DynamicCache`) | **96.7%** (29/30) | 100% |
| h2o | dropped | **3.3%** (1/30) | 20.0% |
| rtn2 | no importance cache; all INT2 | 33.3% (10/30) | 18.8% |
| mikv2-nobal | INT2, no balancer | 46.7% (14/30) | 35.0% |
| **mikv2** | INT2 + balancer (the paper's point) | **56.7%** (17/30) | 35.0% |
| mikv2-hi8 | mikv2 with an INT8 importance cache | 56.7% (17/30) | 26.3% |
| **mikv4** | INT4 + balancer | **96.7%** (29/30) | 45.0% |

*Measured (CPU)*, one seed, 30 samples. At n = 30 one sample is 3.3 points
and the binomial standard error near 50% is ~9 points.

What reproduces the paper, qualitatively:

* **Eviction collapses and retention recovers it.** H2O at a 20% cache: 3.3%
  here against the paper's 4.0% (Table 1). Keeping the same victims at INT4
  instead returns the full-cache score exactly (Table 1: 100%).
* **INT2 is where it gets hard.** 56.7% with the balancer, 46.7% without, 33.3%
  with no importance cache at all. The order matches Table 2 (64.0% → 92.6%),
  but the balancer's +10 points is three samples, **inside the noise**. It is
  not evidence that the balancer works at this scale, only that it does not
  hurt.
* **The importance cache tolerates INT8.** mikv2-hi8 scores exactly what mikv2
  does (Table 3: 92.6% → 92.4%).

Why the sizes are larger than the paper's: this model's heads are `D = 64`, so a
`D/2` group is 32 channels and INT2 costs 3 bits per element, not 2.5. The
"paper accounting" column is `paper_cache_size(...)` at 16-bit; the script
also prints the measured fraction against this fp32 run's dense cache.

**What this does not show:** anything about Llama-3.1-8B, a GPU, or LongBench.
Llama-3.1-8B has `D = 128` (so INT2 is 2.5 bits) and is far stronger at
retrieval, so the INT2 rows should move. The command is in §5.

## 8. The perf seam, exercised (CPU)

`configs/eval_perf_mikv.yaml`'s four rows, run on SmolLM2-135M at prefill 256 /
gen 32 / B = 2 (CPU, fp32, SDPA), all completed through
`cache_backend: external`. The bytes-matched row resolved `cache_budget: 0.20`
to `budget_tokens = 57` → `cache_budget = 0.1979` → `r = 0.115` for this
model's `D = 64` fp32 heads, and `describe()` says "bytes-matched".

Its timings are not reported. The run shared four CPU cores with the §7
sweep, and it measured the control arm at 4.6× the full cache. Timed alone
(one thread, same model, 256-token prompt, 16 steps), a decode step costs
118 / 159 / 207 ms for `DynamicCache` / MiKV `r = 1.0` / MiKV `r = 0.2`, i.e.
+35% for the machinery and +76% with the INT2 read. `torch.profiler` puts the
difference in dequantization (`__rshift__`), `scatter_` and `copy_`. *Measured
(CPU)*; a statement about the reference implementation on a 135M model, not
about MiKV on a GPU.
