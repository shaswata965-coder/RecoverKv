# Digest-Gated Dequantization — implementation plan

**Goal.** Attach a bounding-volume digest to every Tier-2 (int2) window, score
the incoming query against those digests each decode step, and let only the
top-scoring windows into the attention computation. Windows whose digest says
they cannot matter this step stay compressed and are skipped entirely.

This document is a **standalone handoff**: it assumes no knowledge of the
conversation or repo state that produced it. Read §0 and §2 before writing code.

---

## 0. HARD CONSTRAINTS — do not violate these

These are **settled design choices that work in this method's favour**, not
accidents, not tech debt, and not optimization targets. They are listed first
because assistants reliably try to "improve" them unprompted. Any proposal that
touches them is out of scope for this plan and should be rejected without
debate.

### 0.1 Do not change the sink / middle / local structure

The cache layout is `[sink tokens ‖ scored evictable middle ‖ local window]`:

- **sink** (`num_sink_tokens`, e.g. 5) — unconditionally retained, full precision.
- **middle** — the evictable region, ranked by cumulative attention and split
  across Tier 1 / Tier 2 / dropped.
- **local** (`local_window_size`, e.g. 128 tokens) — the most recent tokens,
  unconditionally retained, full precision, never scored for eviction.

Do **not** propose making sinks evictable, making the local window adaptive,
folding local into the scored region, quantizing sinks or locals, gating them by
digest, or "unifying" the three regions under one policy. The gate in this plan
applies to **Tier-2 windows in the middle region only**. Sink and local windows
are never gated, never digested, and never skipped — they go into attention
every step exactly as they do today.

The eviction code depends on this structurally: the local region is identified
**positionally** as the trailing columns of the window axis, and sinks sit
outside the window axis entirely (`body = key_states[:, :, num_sink:, :]`).
Changing the layout is not a tuning change, it is a rewrite of
`_evict_two_tier_impl`.

### 0.2 Do not touch the prefill n×n structure

Prefill computes full quadratic attention and derives the initial window scores
from it. Do **not** propose chunked prefill, windowed/sliding prefill,
approximate or low-rank prefill scoring, digest-gating the prefill, or reusing
the decode gate during prefill.

The whole method rests on the first eviction being made from **exact** prompt
attention: that is the operating point every prompt-compression baseline
(SnapKV / AdaKV / DefensiveKV) is measured at, and it is what makes the
comparison a comparison. Approximating prefill scores would change what the
method *is*, and every accuracy number on record would become incomparable.

### 0.3 Scope of this plan

In scope: digests for Tier-2 windows, per-step digest scoring, and top-k gating
of **Tier-2 windows during decode attention**. Nothing else.

---

## 1. Starting state — what already exists

Everything below is **already in the codebase** and must not be re-derived.

### 1.1 The two-tier cache (QEvict)

`modules/windowed_cache/` implements window-based eviction with cumulative
attention scoring and a three-tier hierarchy:

- **Tier 1** — full-precision fp16 windows (`top_k_fp` of them).
- **Tier 2** — int2 quantized windows (`N_q` of them), still attended over
  every step, still accumulating scores.
- **Tier 3** — dropped outright.

Key files:

| file | what it owns |
|---|---|
| `modules/windowed_cache/cache.py` | `WindowedCache`, `_evict_two_tier_impl` (the eviction), the fused hand-off |
| `modules/windowed_cache/policy.py` | `compute_two_tier_retain` — rank → tier assignment |
| `modules/windowed_cache/decode_kernel.py` | the fused two-tier decode Triton kernel |
| `modules/windowed_cache/flash_decode.py` | patches `flash_attn_func`; calls the fused kernel |
| `modules/quant/slots.py` | `QuantSlotTable` — the write-once int2 ledger |
| `modules/quant/store.py` | `QuantizedStore` — demote / promote / reactivate |
| `modules/quant/quantizer.py` | int2 pack/unpack, affine quantize/dequantize |

Read `EVICTION_LOGIC.md` for the end-to-end token flow.

### 1.2 Digest primitives — **already present and tested, do not rewrite**

`modules/quant/digest.py` + `tests/test_digest.py` (14 tests, CPU, passing).

> Both files are **already in this repo**, copied in and verified. Nothing to
> port. They are untracked until you commit them. `digest.py` imports nothing
> but `torch`, and `modules/quant/__init__.py` does **not** re-export it, so
> import by full path — `from modules.quant.digest import ...` — exactly as
> `tests/test_digest.py` does.
>
> Run this before writing anything else. It is the cheapest confirmation that
> the environment is sound and the primitives arrived intact:
>
> ```bash
> ~/.conda/envs/sticky_env/bin/python -m pytest tests/test_digest.py -q   # expect 14 passed
> ```

```python
build_aabb(keys)      # [B, G, H_kv, S, D] post-RoPE -> (lo, hi) each [B, G, H_kv, D]
build_sphere(keys)    # -> (center [B,G,H_kv,D], radius [B,G,H_kv])
estimate_aabb(q, lo, hi)          # [B,H_q,T,D] -> [B,H_q,T,G] upper bounds
estimate_sphere(q, center, radius)
```

The contract is one-sided: `estimate(q, digest(K)) >= max_{k in K} q·k`. An
over-estimate costs an unnecessary dequant; an **under-estimate silently drops a
key the model wanted**, so the bound direction is the thing the tests pin.

Two settled facts, both with tests:

- **Use AABB, not sphere.** On isotropic data the sphere is *tighter* at half the
  memory (a box round a ball in 128 dims is mostly corner). AABB wins only when
  channels differ in scale — which is this data's regime, and the codebase's own
  evidence is that keys are quantized on a **per-channel** grid (`key_scale` is
  `[B, N, H_kv, D]`) while values get a per-token one.
- **Keys must be POST-RoPE.** The Q tier stores keys *pre*-RoPE and rotates at
  read time. A bound built in the pre-RoPE frame bounds nothing. Post-RoPE keys
  are available in `_evict_two_tier_impl` as `k_post`, gathered from the fp body
  immediately *before* `unrotate_key_window` is applied.

### 1.3 Out of scope: recalling evicted windows

There is **no recall mechanism in this plan**, and none should be built. Tier 3
stays exactly as it is: dropped, permanently, with no backup, no host offload
and no way back. Digests exist here for **one purpose only** — gating which
*resident* Tier-2 windows enter the decode attention.

An earlier experiment did build a recallable tier on top of host-offloaded
records. It is discontinued and deliberately not carried over. Port
`modules/quant/digest.py` and its tests (§1.2); port nothing else from it, and
do not reintroduce host-memory offload, page backup, or recall in any form.

---

## 2. What is already solved — read before building

The motivation usually given for this design has three parts. **Two are already
implemented.** `modules/windowed_cache/decode_kernel.py`'s own header:

> "reading the Q tier **as int2** and dequantizing + RoPE-ing **in registers** —
> no fp16 tier is written to HBM, no concat, and the eviction score falls out of
> the same `q·kᵀ`"

| claimed motivation | status |
|---|---|
| "avoid blanket dequantization of the int2 cache each step" | **already done.** Dequant happens per-tile in registers, never to HBM. The `materialize_effective_kv` path that did write fp16 still exists for the eager backend and CPU tests, but the fused path replaced it. |
| "remove the eager-SDPA fallback used for routing scores" | **already done.** For a single decode query the per-key softmax weight *is* the H2O score, so it falls out of the same `q·kᵀ`. There is no duplicate pass. |
| **"shorter attention sequences"** | **NOT done — this is the real work.** The fused kernel attends over **every** active Q window each step. |

**The actual opportunity, quantified.** At a measured LongBench operating point
(Llama-3.1-8B, `cache_budget=0.20`, `quant_ratio=0.70`, prefill 7105):

```
top_k_fp =  48 windows      (fp tier)
N_q      = 439 windows      (int2 tier)   <- 90% of the retained set
retained = 487 windows = 4029 tokens
```

The Q tier is **90% of what the kernel reads every step**. Gating it to a top-k
is therefore where nearly all the available saving is. Build for this; do not
build for the first two rows of the table.

---

## 3. Design

### 3.1 Digest creation on demotion (Phase 1)

When a window is demoted Tier 1 → Tier 2 in `_evict_two_tier_impl`, compute its
AABB from the **post-RoPE** keys and store it alongside the int2 record.

- Source: `k_post` in the demote block, *before* `unrotate_key_window`.
- Storage: two new fields on `QuantSlotTable`, `key_digest_lo` / `key_digest_hi`,
  both `[B, N, H_kv, D]` fp16, written by `QuantSlotTable.write`.
- Cost: `2 * H_kv * D * 2` bytes per slot = **4 KB/window/layer** at
  `H_kv=8, D=128`. At `n_slots ≈ top_k_fp + N_q + 8` that is ~2 MB/layer,
  ~64 MB across 32 layers at the operating point above. **This is resident GPU
  memory that `cache_budget` does not account for** — state it in any memory
  claim.
- Write-once, like the record itself (design §10). A *reactivated* (dormant →
  active) window already has its digest; do not recompute. A window promoted to
  Tier 1 and later re-demoted reuses the frozen digest — its keys have not
  changed, and re-deriving one would make it disagree with the codes.

### 3.2 Fast-pass query match (Phase 2)

Each decode step, score every active Q window's digest against the query:

```python
est = estimate_aabb(q, lo, hi)      # [B, H_q, 1, n_active]
rank = est.amax(dim=1).squeeze(1)   # [B, n_active]  — max over query heads
```

Reduce over heads with **max, not mean**: a window one head wants badly is worth
dequantizing even if the others are indifferent. That is the failure mode the
whole mechanism exists to catch.

Cost: `n_active * H_kv * D` MACs — equivalent to attending over `2` tokens per
window instead of `window_size` (8), so the *ranking* is ~4× cheaper than the
attention it gates. That ratio is why `window_size = 8` matters (§7).

### 3.3 Selective dequantization (Phase 3)

Select the top-`k` windows by `rank` and hand **only those** to the fused kernel.

**Integration point: `flash_decode._run_fused`, not `cache.update`.** The gate
needs the post-RoPE query, and `update()` runs *before* attention — it has no
query. `_run_fused` already receives `q_hd` (`[B, H_q, D]`) and `ctx["qtier"]`.
Do the selection there:

1. `cache.update` puts the digests in the `qtier` hand-off alongside the codes.
2. `_run_fused` scores them against `q_hd`, takes top-k, and gathers the
   selected windows' codes/scale/zero before calling `fused_two_tier_decode`.
3. The kernel itself **needs no changes** — it just receives fewer windows.

**The score-scatter map must be subset to match.** `ctx["score_meta"]` is
`(order, q_token_len)`, permuting the kernel's emitted window axis onto the
merged score axis; it is built by `compute_score_meta` in
`modules/quant/effective.py` and handed over by `WindowedCache._fused_meta`.
The kernel emits `n_body_win + n_selected` columns instead of
`n_body_win + n_active`, so `order` has to be rebuilt for the selected set.
`flash_decode._run_fused` already asserts `order.shape[1] == wsum.shape[-1]` and
raises on mismatch — that guard will fire if this is got wrong, which is the
desired behaviour, not something to work around.

**Only the Q-tier half of the emitted axis is gated.** The kernel's window axis
is `[body (fp) windows ‖ Q windows]`. The gate selects among the Q windows only;
the `n_body_win` fp columns are untouched, which is the kernel-level statement
of §0.1 (sinks and locals live in the fp body and are never gated).

### 3.4 The eager / materialize path stays ungated

The gate lives in `flash_decode._run_fused`, so it applies to the **fused decode
path only**. The other path — `materialize_effective_kv`, used by the eager
backend, by CPU tests, and whenever the fused kernel is unavailable — continues
to dequantize the whole Q tier and attend over all of it.

This is deliberate, and it must be **stated in any result**: the eager backend
is the reference/quality path, so with gating on, the two backends compute
*different attention* and their outputs will legitimately diverge. Do not treat
that divergence as a bug, and do not "fix" it by gating the materialize path as
well unless there is a reason to — an ungated reference is useful precisely
because it bounds what the gate costs.

Practical consequence: a quality A/B for this feature must run **flash vs
flash**, never flash-gated vs eager-ungated.

---

## 4. Open decisions — resolve these before coding

### 4.1 What happens to a gated-out window's score? **(most important)**

A skipped window receives no attention that step, so its **cumulative score
stops growing** — and cumulative score is exactly what
`compute_two_tier_retain` ranks on at the next eviction. Left alone, gated
windows decay and get evicted *because they were gated*: a feedback loop where
one step's approximation becomes permanent deletion.

ArkVale does not face this — there, the digest score *is* the ranking. QEvict
ranks on accumulated attention, so the two mechanisms disagree.

Options:

| option | consequence |
|---|---|
| **(a) Freeze** — leave the score untouched while gated | No decay, but a window that stops being relevant never decays either; eviction goes stale. |
| **(b) Credit the digest estimate** — add a scaled `rank` to the score | Keeps ranking live, but mixes units: the bound is an unnormalized dot-product ceiling, the score is accumulated softmax mass. Needs a calibration nobody can derive a priori. |
| **(c) Accept the decay** — score 0 for gated steps | Simplest and self-consistent (unattended really did contribute nothing), but couples the gate to eviction: raising `k` changes *what gets evicted*, not just speed. |

**RESOLVED: (a) and (c) are the same operation here, so there is one policy and
no flag.** `scorer.accumulate` is a plain `state_scores += new_scores` with **no
decay term anywhere** (`modules/windowed_cache/scorer.py:150`), and
`compute_two_tier_retain` ranks on `window_scores.mean(dim=1)` of that pure
running sum. So writing 0 for a gated window *is* freezing it: its accumulated
mass cannot shrink, it only stops growing while its neighbours grow. The
"feedback loop where one step's approximation becomes permanent deletion" the
table warns about needs a decay term to bite, and there isn't one.

What remains true, and is the thing to actually watch: the gate still couples to
eviction through the **relative** ranking, because unattended windows fall behind
attended ones. So `k` changes *what gets evicted*, not only decode speed — which
is consequence (c), minus the decay. Implemented in `flash_decode._run_fused`;
the reasoning is repeated at the line that does it. Option (b) remains untried
and still needs a calibration experiment before it would mean anything.

### 4.2 How is `k` chosen?

- **Fixed `k`** — simplest, predictable kernel shapes (matters: the fused kernel
  tiles over the window axis, and varying `n` may force recompiles).
- **Threshold on the bound** — adaptive, but makes the sequence length
  data-dependent and ragged across rows in a batch, which this codebase avoids
  deliberately (`BATCHING_PLAN.md` §3: tiers stay dense and rectangular).

**RESOLVED: fixed `k`, as the fraction `digest_gate_frac` of `N_q`.** Resolved
once into `ResolvedConfig.digest_top_k` (ceil, clamped to `[1, N_q]`), so the
kernel's window count is constant between evictions and cannot go ragged across
a batch. `None` is off — and off means *nothing allocated*, not merely nothing
gated, so a build with the gate off is memory-identical too. Still to sweep (§6
predicts a unimodal curve, so expect an interior optimum).

### 4.3 Accuracy is not automatically preserved

The plan's claim — "a low bound means the model would have ignored it anyway" —
**does not follow**, and any writeup should not assert it.

The bound guarantees no *individual* key in a skipped window scores high. But
softmax is normalized over **all** keys: dropping hundreds of individually-low
windows removes probability mass from the denominator, inflating every retained
weight. With `N_q = 439` windows skipped down to, say, 64, the aggregate dropped
mass can be significant even when no single window mattered.

This is **approximate attention with measurable error**, exactly as ArkVale
treats it. Budget for measuring the error, not for asserting it away.

---

## 5. Implementation order

> **Status (2026-09-13).** Steps 1-7 are done. §10 is what landed; **§11 is the
> measured outcome** — the gate is a quality win and a latency loss, and §3.3's
> "the kernel needs no changes" is the reason for the second half.

1. **Port `modules/quant/digest.py` + `tests/test_digest.py`.** Already written
   and passing; no changes needed.
2. **Add digest storage to `QuantSlotTable`** (`key_digest_lo/hi`), written in
   `write()` alongside the record. Extend `gather()` to return them.
   Tests: a digest survives write→gather; a reactivated window keeps its
   original digest (write-once).
3. **Compute the digest at demotion** in `_evict_two_tier_impl`, from `k_post`
   before `unrotate_key_window`. Pass it into `demote_many`.
   Test: the stored digest encloses the window's real post-RoPE keys.
4. **Carry digests into the fused hand-off** (`_fused_qtier`). No behaviour
   change yet — verify decode output is bit-identical to before.
5. **Implement the gate in `_run_fused`** behind a config flag, default OFF:
   score → top-k → gather → subset `score_meta`. With the flag off, everything
   must stay bit-identical.
6. **Resolve §4.1** and implement the chosen scoring policy.
7. **Validate** (§8), then sweep `k`.

Keep every step bit-identical-when-disabled. That property is what makes a
regression attributable.

---

## 6. Prior evidence, from a discontinued experiment

Retained only because it is evidence about **digests**, which this plan reuses.
The mechanism it came from is not part of this plan (§1.3).

- **The digest ranks real relevance — the key result.** Sweeping how many
  digest-selected windows a mechanism acted on, over LongBench 2wikimqa
  (Llama-3.1-8B, budget 0.20, q=0.70, n=200), produced a clean unimodal curve:
  47.47 (off) → 48.94 → 49.10 → **49.36** → 49.28 → 48.50 — rising to an
  optimum, then declining once the mechanism grew too aggressive. A digest that
  ranked nothing useful would have given a flat or erratic curve. This is the
  main reason to expect the gate in this plan to work, and to expect `k` to have
  an optimum rather than being monotone in either direction.
- **The pipeline is deterministic.** Two identical-config runs produced
  byte-identical predictions, so a paired bootstrap over examples is a valid CI
  and A/B differences are attributable to the change.
- **Method-off must be byte-identical.** The strongest check available: with the
  mechanism disabled, predictions matched the baseline exactly across all 200
  examples. Reproduce that discipline here.

---

## 7. Risks and caveats

- **At `B=1`, decode is weight-bound** (`BATCHING_PLAN.md` §5). LongBench runs
  at `B=1`. Cutting KV attention work may therefore buy little TPOT there; the
  win should appear at larger batch or much longer context. **Profile where
  decode time actually goes before building**, so the effort targets a real
  bottleneck. This is the single biggest risk to the premise.
- **Digest memory is not free** and is not counted by `cache_budget`: ~4 KB per
  window per layer. At `window_size = 8` a per-window fp16 AABB is roughly half
  the size of the ~8.25 KB int2 record it summarizes, because that record is
  dominated by its own per-channel scale/zero grid. If resident cost becomes a
  problem, group `P` consecutive windows under one digest (`P=4` matches the
  32-token page ArkVale validated) — at the price of a looser bound and a
  coarser gate.
- **Ragged shapes.** Gating changes the kernel's window count. Fixed `k` keeps it
  rectangular; anything adaptive risks per-row raggedness, which this cache
  avoids by design.
- **The `score_meta` guard will fire** if the scatter map is not subset
  correctly. Treat that as the feature it is.

---

## 8. Validation

1. **Flag-off equality** — with gating disabled, decode output and generated
   predictions bit-identical to the pre-change build.
2. **Bound property** — the stored digest encloses the window's post-RoPE keys
   (unit test, CPU).
3. **Gate correctness** — with `k = n_active` (gate everything in), output must
   match ungated exactly. This separates "the gate plumbing is right" from "the
   gate's selection is good".
4. **Layer-major parity** — `tests/test_layer_major_evict.py` compares the
   per-layer and layer-major eviction paths tensor-for-tensor; keep it passing.
5. **Latency** — measure TPOT at several batch sizes, not just `B=1` (§7).
   Do **not** write a new timing harness: `scripts/run_perf.sh`,
   `scripts/run_perf_table.sh` and `scripts/print_perf_table.py` already
   measure TPOT across a prefill × batch grid and record provenance,
   `scripts/profile_decode.py` breaks a step down by phase, and
   `DECODE_SPEED_PLAN.md` documents where the per-token budget currently goes.
   Start from `scripts/profile_decode.py` for the §7 question (is decode
   weight-bound at this operating point?), then use the perf table for the
   before/after.
6. **Quality** — LongBench, three arms: gate off / gate on at large `k` / gate on
   at small `k`, with a paired bootstrap CI per arm.

---

## 9. Environment notes (this cluster)

- **Always use `~/.conda/envs/sticky_env/bin/python`.** A bare `python` resolves
  to a shared cluster Anaconda missing the project's deps.
- **Cache-level tests need a GPU to even collect.**
  `modules/windowed_cache/score_kernel.py` builds its Triton autotuner at
  *import* time, and `_HAS_TRITON` only checks that triton imports — not that a
  driver exists. On a GPU-less box, importing `modules.windowed_cache.cache`
  raises `RuntimeError: 0 active drivers`, which kills collection for every
  cache-level test including ones whose docstrings say "CPU only". Pure-CPU
  suites (`tests/test_digest.py`, `tests/test_quant.py`) are unaffected.
- **Eval runners take an explicit kwarg list.** `longbench_runner.py`,
  `ruler_runner.py` and `gsm8k_runner.py` construct `WindowedCacheConfig` field
  by field, so **a new YAML knob is silently inert until added there**. This has
  already caused a shipped regression (`ACCURACY_RECOVERY_PLAN.md` §2). Add new
  fields to the runners *and* record them in the run's `.meta.json` sidecar, or
  two arms of an ablation will write identical provenance.
- **Budget resolution is counter-intuitive.** In the default `bytes` mode an int2
  window is ~4× cheaper, so the retained *window count* grows with `quant_ratio`
  and can exceed the prompt: at prefill 64 / budget 0.25 / q=0.5 the budget
  resolves to 22 retained windows against 16 that exist, and **nothing is ever
  evicted**. Use `quant_budget_mode="tokens"` for small test configs, and always
  assert a test actually evicted something rather than passing vacuously.
- LongBench data: `LONGBENCH_LOCAL_DIR=/home/ee/phd/eez228470/kv_cache/qwen_longbench_final_int2/RecoverKv/data/longbench`
- Model: `/home/ee/phd/eez228470/llama-3.1-8b-instruct`
- Operating point used throughout: `cache_budget=0.20`, `quant_ratio=0.70`,
  `window_size=8`, `num_sink_tokens=5`, `local_window_size=128`, flash backend.


---

## 10. What landed (steps 1-6)

One config knob turns the whole feature on: **`digest_gate_frac`** — `None` (the
default) is off, a float in `(0, 1]` admits that fraction of `N_q` Q-tier windows
per decode step. It is rejected at `quant_ratio = 0` (no Q tier to gate).

| file | change |
|---|---|
| `modules/windowed_cache/config.py` | `digest_gate_frac` + validation; `ResolvedConfig.digest_top_k` resolves it to an int |
| `modules/quant/slots.py` | `key_digest_lo/hi` `[B, N, H_kv, D]` fp16, allocated only under `store_digest`; written in `write()`, returned by `gather()` |
| `modules/quant/store.py` | `demote_many(..., key_digest=(lo, hi))`; `store_digest` through `__init__` / `ensure` / `join_layers` |
| `modules/windowed_cache/cache.py` | `build_aabb(k_post)` at demotion, **before** `unrotate_key_window`; digests carried into `_fused_qtier` and `_build_joint_fused_ctx` (both memos) |
| `modules/windowed_cache/flash_decode.py` | `_gate_qtier` (score → top-k → gather) and the subset score-scatter in `_run_fused` |
| runners + `scripts/audit_e2e.py` | the knob reaches `WindowedCacheConfig` and the `.meta.json` sidecar (§9's trap) |

Three design points worth not re-deriving:

- **`window_scores` stays full width.** Gating shortens what the *kernel emits*,
  not what the *scorer is indexed by*: a gated-out window keeps its column and
  receives 0. So `score_meta`'s `order` is **not** subset (§3.3 expected it to
  be); instead `_run_fused` builds a physical→emitted map, `-1` where the gate
  skipped, and scatters through it. The §3.3 guard is now two guards — one on the
  full physical width, one on the emitted width — and both still fire loudly.
- **A no-op gate is the same object.** At `k >= n_active` `_gate_qtier` returns
  the untouched `qtier`, so §8.3 is a real bit-identity check and not an
  "equal-by-luck" one. Selected indices are sorted ascending for the same reason.
- **Max over query heads, not mean** (§3.2), pinned by a test built so the two
  reductions actually disagree.

### Outstanding: step 7 (needs a GPU)

`tests/test_digest_gate.py` collects only on a GPU box — importing anything under
`modules.windowed_cache` builds `score_kernel`'s Triton autotuner at import time
(§9). Nothing in it runs on the GPU; the kernel is stubbed.

```bash
# 1. unit (both run on CPU today; the gate file needs a GPU box to COLLECT)
~/.conda/envs/sticky_env/bin/python -m pytest \
    tests/test_digest.py tests/test_digest_storage.py tests/test_digest_gate.py -q

# 2. §8.1 flag-off equality + §8.4 layer-major parity
~/.conda/envs/sticky_env/bin/python -m pytest \
    tests/test_quant_cache.py tests/test_layer_major_evict.py \
    tests/test_backend_e2e.py -q

# 3. §7 — the premise. Is decode weight-bound at this operating point? Answer
#    this BEFORE sweeping k: at B=1 the gate may buy no TPOT at all.
~/.conda/envs/sticky_env/bin/python scripts/profile_decode.py

# 4. §8.5 before/after TPOT across the prefill x batch grid
bash scripts/run_perf_table.sh
```

§8.3 (gate correctness) is `digest_gate_frac: 1.0` vs the gate off: identical
predictions, since a full-`k` gate returns the qtier untouched. That is the check
that separates "the plumbing is right" from "the selection is good", and it
should be run before any quality arm.

Do not skip straight to §8.6: §7 names the premise risk plainly — LongBench runs
at `B=1`, where decode is weight-bound, so a correct gate can still buy nothing.
Profile first.


---

## 11. Measured outcome (step 7)

All at the §9 operating point, verified from the run sidecars: Llama-3.1-8B,
prefill 7105, `cache_budget=0.20`, `quant_ratio=0.70`, `quant_budget_mode=bytes`
-> `top_k_fp=48`, `N_q=439`. LongBench 2wikimqa, n=200, A100-80GB.

### 11.1 Quality — the gate works, and at k=110 it *helps*

| arm | `digest_gate_frac` | k | score | delta | preds differ | 95% CI (paired bootstrap) |
|---|---|---|---|---|---|---|
| off | `None` | — | 47.47 | — | — | — |
| k100 | 1.0 | 439 | 47.47 | +0.00 | **0/200** | [+0.00, +0.00] |
| k50 | 0.5 | 220 | 47.44 | -0.03 | 5/200 | [-0.17, +0.06] |
| k25 | 0.25 | 110 | **48.37** | **+0.90** | 18/200 | **[+0.05, +2.02]** |

The four arms' `.meta.json` sidecars differ in **exactly one field**
(`digest_gate_frac`), so §9's "two arms write identical provenance" trap is
avoided and the deltas are attributable.

- **§8.3 passed in the strongest available form.** `k100` is *byte-identical* to
  `off` — same sha256 over the predictions file, 0/200 differing. `off` allocates
  no digests at all while `k100` builds an AABB at every demotion, carries it
  through both fused memos, scores it each step and rebuilds the scatter map — so
  identical output proves the scatter map is right, proves digest construction is
  inert, and is what makes the other rows trustworthy rather than possibly
  measuring a plumbing bug.
- **Gating out half the Q tier is free**: 219 of 439 windows skipped moved 5
  predictions and 0.03 points. The digest is skipping windows the model did not
  need, which §4.3 explicitly warned not to *assume*.
- **Gating out three quarters is better than baseline.** Direction and rough
  magnitude match §6's discontinued-mechanism curve, from the same 47.47.

**Do not over-read the +0.90.** The CI's lower bound is +0.05 — it clears zero by
a hair. One dataset, n=200, and the gain rests on 18 flipped predictions. It is
evidence worth pursuing, not a headline.

**The mechanism is unidentified, and there are two candidates.** (a) Attention:
skipping low-bound windows removes mass that was noise. (b) Eviction: per §4.1 a
gated window stops accumulating score, so the *relative* ranking
`compute_two_tier_retain` reads shifts and a different set survives. These are
not separable from this data, and (b) means `k` is partly an eviction-policy
knob. Resolving it needs an arm that gates attention but credits the score as if
ungated.

**The peak has not been found.** §6 predicts unimodal and k25 is the most
aggressive point measured, still rising. `frac` of 0.125 and 0.375 are the
obvious next rungs.

### 11.2 Latency — a clear regression, and §3.3 is why

`scripts/profile_decode.py`, prefill 7105, 20 steps, fused:

| | gate off | gate on (k=110) | |
|---|---|---|---|
| `_two_tier_decode_kernel` B=1 | 22.379 ms | 6.795 ms | **-70%** |
| `_two_tier_decode_kernel` B=8 | 23.428 ms | 7.384 ms | **-68%** |
| wall B=1 | 84.63 ms | 153.60 ms | **+81%** |
| wall B=8 | 97.59 ms | 176.13 ms | **+80%** |
| launches B=1 | 2813 | 5536 | +2723 (+97%) |
| launches B=8 | 2954 | 5678 | +2724 (+92%) |
| CUDA total B=8 | 91.94 ms | 118.65 ms | +26.7 ms |

The gate cuts its target kernel by ~70%, exactly as the window reduction
predicts. It then loses 4x that much elsewhere.

**Root cause: §3.3's "The kernel itself needs no changes — it just receives fewer
windows."** Honouring that means `_gate_qtier` *materializes a compacted Q tier*:
eight `index_select` copies per layer per step over tensors like `k_codes`
`[B, 439, H_kv, D, ws/4]`. That is a full read-and-write of the selected data
before the kernel reads it again, so selection costs roughly what the shortened
kernel saves — plus ~85 extra launches per layer per step (~2723 across 32
layers), which is fatal at B=1 where decode is host-bound.

**The fix is to stop copying.** Pass the kernel a `[B, k]` index vector and let it
read the windows it wants — it already indexes the window axis, so it needs an
indirection, not a pre-compacted input. That contradicts §3.3 and is a real Triton
change, so it is worth doing only if the quality result above holds up.

**Correction to §2 for the record.** "The Q tier is 90% of what the kernel reads
every step. Gating it to a top-k is therefore where nearly all the available
saving is" — the first clause is right, the second does not follow. At this
operating point the fused kernel is 40.8% of CUDA time at B=1 and CUDA is 64.9%
of wall, so the whole Q-tier read is ~26% of a decode step; and the kernel is
latency-bound (B=1 -> B=8 is 8x the rows for +2.8% time), so shortening its window
axis does not shorten it proportionally. The saving is real but bounded, and
smaller than the cost of the delivery mechanism chosen to capture it.

### 11.3 Defects found in paths §8 depends on

All pre-existing, all confirmed against a stashed baseline, none caused by this work:

- `tests/test_backend_e2e.py::_tiny_model` never sets `attn_implementation`, so
  transformers 4.47.1 gives it sdpa and the flash patch has nothing to intercept —
  12 flash-backend tests cannot pass as written. **Not fixed** (out of scope).
- `scripts/profile_decode.py` had no repo-root bootstrap (`ModuleNotFoundError:
  utils` under its own documented invocation; needs `PYTHONPATH=.`) and drove a
  hand-written decode loop **without `cache_position`**, which an evicting cache
  rejects because `get_seq_length()` is the retained count, not the token index.
  **Fixed** (the `cache_position` half; the `PYTHONPATH` half left alone).
- `utils/config.py`'s `CacheConfig` had no `digest_gate_frac`, so the knob was
  unsettable from YAML and every arm would silently have run ungated — §9's trap,
  and the one that would actually have shipped a dead ablation. **Fixed.**
- `configs/eval_perf_*.yaml` inherit `quant_budget_mode: tokens`, which resolves
  to `N_q=113`, not the `N_q=439` every figure in this plan quotes. The first
  profile was taken at the wrong operating point and had to be redone.
  **Fixed** by pinning `bytes` in `configs/eval_perf_digest.yaml`.


---

## 12. Open question: what is the +0.90 actually measuring?

§11.1 measured +0.90 at k=110 with a CI that clears zero by 0.05. **The mechanism
is not established**, and until it is, the number cannot be written up as
"digest-gated attention improves quality" — because there are two candidate
causes and the data so far cannot tell them apart:

1. **Attention.** Skipping windows whose bound is low removes mass that was
   noise, so the surviving distribution is sharper.
2. **Eviction.** A gated window accumulates no score that step, so the *relative*
   ranking `compute_two_tier_retain` reads shifts and a **different set of
   windows survives**. On this reading the digest is not improving attention at
   all — it is acting as an eviction policy, and `k` is a policy knob.

These have very different consequences. If (2) dominates, the method's honest
name is *digest-informed eviction*, the latency work in §11.2 is beside the
point, and the right comparison is against other eviction policies rather than
against approximate-attention methods.

### 12.1 The decomposition

`digest_gate_mode` (config; `both` | `attention` | `eviction`) splits them.

| mode | attention | eviction scores | isolates |
|---|---|---|---|
| `both` | gated | gated | — (production; effects entangled) |
| `attention` | **gated** | ungated (2nd kernel run) | effect 1 |
| `eviction` | **ungated** | gated (skipped columns zeroed) | effect 2 |

`eviction` mode attends over every window, so its output is bit-identical to the
gate-off run and the only variable is which windows accumulate score. `attention`
mode runs the kernel twice per layer per step — deliberately slow, diagnostic
only, never production.

Arms: `configs/longbench_digest_k25_{attn,evict}.yaml`, both at k=110, read
against `off`=47.47 and `both`=48.37.

| outcome | reading |
|---|---|
| `evict` ~ 48.37, `attn` ~ 47.47 | the gain is eviction; rename the method |
| `attn` ~ 48.37, `evict` ~ 47.47 | the gain is attention; §3's framing holds |
| both ~ 47.9 | additive; neither alone explains it |
| both ~ 47.47 | interaction or noise — get the second dataset first |

### 12.1.1 RESULT: the eviction effect is zero

| arm | score | delta | preds differ | sha256 of predictions |
|---|---|---|---|---|
| `off` | 47.47 | — | — | `9d1478ab…` |
| `k100` (k=439) | 47.47 | +0.00 | 0/200 | `9d1478ab…` |
| **`k25_evict`** | **47.47** | **+0.00** | **0/200** | **`9d1478ab…`** |
| `k25` (`both`) | 48.37 | +0.90 | 18/200 | `709ac82a…` |

Isolating the eviction effect reproduces the gate-off run **byte for byte**.
Suppressing the skipped windows' score contributions flips no eviction ranking at
all.

The geometry says why. Eviction fires at steps 0, 8, 16, 24 for
``max_gen_len=32``. **Step 0 consumes prefill scores** — exact n x n attention
over 7105 prompt tokens, which no decode-time gate can touch (§0.2) — and the
three later evictions rank on cumulative scores still dominated by that prefill
mass. Eight steps of decode contribution is a rounding error against it.

So §4.1's headline worry — "gated windows decay and get evicted *because* they
were gated: a feedback loop where one step's approximation becomes permanent
deletion" — **does not occur at this operating point**. That is now the second
independent reason the §4.1 options were moot: they were already identical under
a decay-free `+=` accumulator (§4.1), and the effect they differ about is null
here anyway.

**Scope.** This holds at ``gen_len=32``. Decode contributions accumulate, so at
long generation — 512 tokens, many more evictions, prefill mass proportionally
smaller — the eviction path could become live. The claim is "negligible at this
operating point", not "structurally impossible". A long-generation arm would
settle it.

### 12.1.2 RESULT: the decomposition is exact — the gain is all attention

| arm | score | delta | preds differ | sha256 |
|---|---|---|---|---|
| `off` | 47.47 | — | — | `9d1478ab…` |
| `k25_evict` | 47.47 | +0.00 | 0/200 | `9d1478ab…` — **identical to `off`** |
| `k25_attn` | **48.37** | **+0.90** | 18/200 | `709ac82a…` — **identical to `both`** |
| `k25` (`both`) | 48.37 | +0.90 | 18/200 | `709ac82a…` |

The two isolation arms partition the effect with **nothing left over**, and not
approximately — byte-identically. `attention` mode (gated attention, ungated
eviction scores) reproduces production exactly; `eviction` mode reproduces
gate-off exactly.

**100% of the +0.90 is the attention path; 0% is eviction.** §4.1 is closed: its
options were already identical under a decay-free `+=` accumulator, and the
effect they disagreed about is null at this operating point anyway (§12.1.1).

This also self-checks the implementation. `attn` and `both` differ *only* in
whether the eviction scores are gated; §12.1.1 showed gating them flips no
ranking, so the two runs **must** coincide — and they do, to the byte. A
divergence would have meant one of the modes was not doing what it claims.

**Consequence for the writeup.** The method is what §3 says it is: attention
gated on a geometric bound. It is not an eviction policy wearing a digest, and it
does not need to be defended as one.

### 12.1.3 RESULT: generality — the +0.90 does NOT replicate

A second dataset at a genuinely different operating point (hotpotqa, prefill
11380, **N_q=729**, so k=183 and **546 windows skipped per step**):

| dataset | arm | score | delta | preds differ | 95% CI |
|---|---|---|---|---|---|
| 2wikimqa | `off` | 47.47 | — | — | — |
| 2wikimqa | `k25` | 48.37 | +0.90 | 18/200 | [+0.05, +2.02] — significant |
| hotpotqa | `off` | 57.97 | — | — | — |
| hotpotqa | `k25` | 58.09 | +0.12 | 5/200 | [-0.01, +0.33] — **not significant** |

**Do not claim a quality improvement.** One dataset clearing zero by 0.05 and a
second straddling it reads as a favourable draw, not an effect. §11.1 already
warned the +0.90 rested on 18 flipped predictions; the second dataset is what that
warning was for, and it did not hold up.

**What replicates is the preservation claim, and it is strong:**

> Skipping ~75% of the Tier-2 int2 windows every decode step changes 18/200
> predictions on 2wikimqa and 5/200 on hotpotqa, with no quality loss on either.

That holds across two datasets, two cache sizes (N_q 439 and 729) and two prefill
lengths, and it is the substantive claim the digest was built to support: the
bound ranks relevance well enough that three quarters of the compressed cache is
demonstrably not needed on any given step. §4.3 said accuracy preservation "does
not follow" and had to be measured rather than asserted — it was measured, and it
holds, which is a result. "Improves quality" is not.

### 12.2 Also outstanding

- **The peak.** §6 predicts unimodal and k=110 was still rising.
  `configs/longbench_digest_k{125,375}.yaml` bracket it.
- ~~**Generality.**~~ **Done — see §12.1.3.** The +0.90 did not replicate; the
  preservation claim did. A third dataset would firm up the preservation result,
  which is now the headline.
- **What the preservation claim is worth depends on §11.2.** Skipping 75% of the
  Q tier for free is only interesting if skipping it *buys* something. It does not
  buy TPOT (§11.2, §12.3). It would buy memory if the skipped windows need not be
  resident — but they do: this gate selects among RESIDENT windows and frees
  nothing (§1.3 forbids recall, and the digests themselves add ~4 KB/window/layer
  that `cache_budget` does not count). **State this plainly in any writeup**: as
  it stands the method demonstrates that 75% of the Q tier is unnecessary per
  step, without yet converting that into either speed or memory.

### 12.3 Why the latency work stopped here

§11.2's regression was chased through three implementations — host gather, kernel
indirection, then an unsorted `sel` with an fp16 digest — landing at B=8 TPOT
18.01 ms/token against a 12.20 baseline. At that point the cost is no longer GPU
work (CUDA 91.94 -> 95.38 ms, near break-even) but **launches**: 2954 -> 4574,
and the relationship to wall time is non-linear. The ungated baseline runs at
94.2% occupancy with the queue permanently full; a gate that must complete before
each layer's kernel can launch drains that queue 32 times a step, and exposed
host time jumps 5.65 -> 48.7 ms. Shaving ops does not remove the stall.

The only remaining path to a TPOT win is **fusing the gate into
`_two_tier_decode_kernel`** so there is no host round-trip. Ceiling:
`(97.59 - 15.16) / 8 = 10.3 ms/token` vs 12.20 — about **16%**, for a substantial
Triton rewrite of a kernel that still has no independent GPU correctness test
beyond `tests/test_digest_gate_kernel.py`. Not worth starting until §12.1 says
what the feature actually is.


---

## 13. The union experiment, and what it settled

### 13.1 Only ~42% of the Q tier is ever selected

`scripts/digest_union.py` (opt-in via `STICKYKV_DIGEST_TRACE`) records which
**window ids** each step admits, grouped per eviction epoch — ids because `sel`
indexes an axis that is rebuilt at every eviction, per epoch because the resident
set changes across them.

```
prefill 7105, B=1, 32 steps, 128 epoch-layer pairs, 8.0 steps/epoch
  k / resident       0.251     110 read per STEP
  union / resident   0.419     ~184 distinct over an epoch  (range 0.333-0.551)
```

880 selections (8 steps x 110) drawn from ~184 distinct windows — each chosen ~4.8
times. Consecutive queries want the **same** windows, consistently across all 32
layers. So the per-step sparsity is not churn.

### 13.2 But most of that was over-provisioning

`configs/longbench_union_control.yaml` shrinks Tier 2 to the measured union
(`budget 0.126, q 0.49` -> `top_k_fp=48` unchanged, `N_q=181`) with **no gate**:

| arm | N_q | k | score | vs `digest_off` |
|---|---|---|---|---|
| `digest_off` | 439 | — | 47.47 | — |
| `digest_k25` | 439 | 110 | 48.37 | +0.90 [+0.06, +2.06] |
| `union_control` | 181 | — | 47.24 | -0.23 [-2.28, +1.77] |
| `union_control_gated` | 181 | 46 | 47.67 | +0.21 [-1.73, +2.13] |

**Dropping 439 -> 181 int2 windows costs nothing detectable.** `cache_budget=0.20,
q=0.70` was simply more Tier 2 than this workload needs: 37% less retained cache,
no digest, no gate, no kernel work. The union finding is mostly a **budget-tuning**
result, not a digest result — and any writeup must say so, because every gated
number before §13.2 was measured on a cache with slack in it.

### 13.3 The gate survives right-sizing — weakly

`union_control_gated` vs its **own** ungated control: **+0.43 [+0.00, +1.24]**.
Smaller than the +0.90 on the over-provisioned cache, so part of that was slack;
but not zero. The gate finds per-step structure that a correctly sized budget does
not already capture.

Four arms, two budgets, consistent weak positive, every CI lower bound at or just
above zero. That is the honest summary: **a real but small effect that has never
cleared significance convincingly**, plus a replication failure on hotpotqa
(§12.1.3). Do not quote +0.90.

### 13.4 The digest's own footprint eats the saving

Per layer at `N_q=181`:

```
union_control         KV 3.65 MB + digest 0.00 MB = 3.65 MB   63% of baseline
union_control_gated   KV 3.65 MB + digest 1.71 MB = 5.36 MB   92%
  with the duplicate below removed                 = 4.39 MB   75%
```

**BUG: digests are stored twice.** `QuantSlotTable.key_digest_lo/hi` holds the raw
`[B, N, H_kv, D]`, and `prepare_aabb` builds a second transposed
`[B, H_kv, 1, D, G]` copy held in the fused memo — which is why B=8 measured
+1.09 GB where one copy is ~0.5 GB. Fix: store the digest **already transposed**
at `write()` time and drop the raw field. Halves the overhead and removes a
per-eviction transpose.

Even fixed, the plain budget reduction (63%, no machinery) beats the gated
right-sized arm (75%) on memory at indistinguishable quality. **At
`window_size=8` the digest cannot pay for itself on memory either**, for the same
reason it could not on speed (§11.2): a 4 KB box against an 8.4 KB record.

### 13.5 What would actually change the verdict

Every negative result here traces to one number: the digest is ~48% the size of
what it filters. §7 already names the fix — group `P` consecutive windows under one
box. At `P=4` the box costs 1 KB per window equivalent (12% of the record), which
moves both the memory and the bandwidth arithmetic decisively, at the price of a
coarser selection unit. **That is the one change left that could flip any of
§11.2, §12.1.3 or §13.4**, and it should be tried before the method is written off
or written up.


---

## 14. P-grouping: negative on all three axes

§7 and §13.5 named grouping as the one change that might flip the verdict. It was
implemented (`digest_group_size`, `prepare_aabb_grouped`, group->window expansion
in `_gate_qtier`) and measured. **It did not.**

| | baseline | P=1 gate | P=4 gate |
|---|---|---|---|
| 2wikimqa | 47.47 | **48.37** (+0.90) | **47.09** (-0.38) |
| TPOT B=1 | 0.0534 | 0.0724 (+36%) | **0.0814 (+52%)** |
| TPOT B=8 | 0.0713 | 0.0925 (+30%) | **0.0962 (+35%)** |
| steadyKV B=8 | 56.23 | 57.32 | 57.32 (unchanged) |

P=4 vs P=1 head to head: **-1.29 [-2.94, +0.12]**.

### 14.1 Why the byte argument did not transfer

The case for grouping was the ratio: a 4096 B box against an 8448 B record. That
is the right diagnosis for **memory** (§13.4) and it was wrongly extended to
**speed**. Bytes were never the latency bottleneck:

```
digest read at P=1, B=8   ~14.4 MB/layer  ->  P=4  ~3.6 MB
saving 345 MB/step / ~1.5 TB/s            ~=   0.23 ms/step
```

0.23 ms against a 71 ms baseline — while the group->window expansion
(`sel_g * P + arange`, reshape, tail arange, cat, cast) adds ~6 host ops per layer
per step, ~190 per step, of exactly the thing that has been the bottleneck since
§11.2. Grouping **trades a negligible bandwidth saving for more host ops**, which
is why it is worst at B=1 (+52%), the most host-bound arm.

Memory did not move either: per-window digests remain in the slot table
(write-once, design §10); only the memoized prepared copy shrank.

### 14.2 Why quality fell

A group's box is the union of its members', so it scores as high as its **best**
member and admits all P. At P=4 each genuinely relevant window drags in three
passengers, so an admitted-window budget of ~115 buys far less relevance than 110
individually chosen ones. §13.3 had left only +0.43 of headroom on a right-sized
cache; grouping spends more than that.

### 14.3 The complete latency record

| implementation | B=8 TPOT | vs baseline |
|---|---|---|
| baseline, no gate | 0.0713 | — |
| host gather | ~0.22* | +209% |
| `sel` indirection into the kernel | ~0.183* | +157% |
| + fused scoring kernel, collapsed scatter | 0.0925 | +30% |
| + P=4 grouping | 0.0962 | +35% |

\* profiler-measured and inflated ~37%; see §14.4.

Five implementations, each a real improvement to the mechanism, none flipping the
sign. The floor is structural: **32 per-layer host dependencies per token against
a GPU already at 94% occupancy.** Only moving selection fully inside
`_two_tier_decode_kernel` would remove it, and top-k has no Triton primitive while
`BLOCK_NW` tiling means a scattered selection skips only ~10% of tiles.

### 14.4 Methodology note — `profile_decode` cannot measure deltas

Its wall timer sits **inside** the `with profile(...)` block, so per-op profiler
overhead scales with op count and is charged to whichever arm issues more ops. It
reported 84.63 ms/step at B=1 where `run_perf_table.sh` measures 53.4 — **58%
inflation** — and its CUDA column exceeded real wall time. Use it for "where does
time go" (§7), never for before/after. The perf table is the instrument §8.5
specified, and switching to it changed several conclusions.
