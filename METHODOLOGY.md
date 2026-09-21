# Prefill & Decode Methodology — as the code actually runs

**Branch:** `int2_clustered_rep` · **Commit:** `dc694be` · **Date:** 2026-09-21

This document describes how prefill and decode work **as implemented in the
current code**, read straight from the source rather than from the design/plan
docs (`design.md`, `DECODE_PLAN.md`, `PREFILL_PLAN.md`, …), several of which
predate the force-rewrite of this branch and no longer match the code. Every
non-obvious claim is anchored to a `file:line` so it can be checked against the
tree at this commit. Line numbers drift with edits; treat them as pointers.

The last section answers a specific question: what `store.window_centroids` and
`scorer.expand_keep_to_query_heads` are for, and why nothing calls them.

---

## 0. One-paragraph summary

The cache keeps a **fixed byte budget** of the prompt's KV. Every token's
received attention is scored; on a fixed cadence the lowest-scoring windows are
either dropped or **demoted from fp16 to a 2-bit "Q tier"**, and the highest
kept in fp16. Decode attends over `[sink ‖ fp survivors ‖ int2 survivors]`. At
`quant_ratio > 0` a per-window **rank-1 "card"** lets each decode step *skip*
dequantising most int2 windows and instead attend to them through a cheap
**value centroid**, correcting the skipped windows' eviction scores from the
card so the gate does not cannibalise the tier it exists to read less often.
Everything after the first eviction runs **layer-major** (all `L` layers in one
batched pass) and the eviction body runs **compiled-or-error** (never silently
eager).

---

## 1. Architecture at a glance

- **Two backends**, selected by config and validated as a pair
  (`utils/cache_factory.py:203` `validate_backend_attn_pairing`):
  - `flash_attn` ↔ `attn_implementation=flash_attention_2` → `modules/windowed_cache/`.
    This is the default and the only backend with the fused CUDA decode/score
    kernels. **Everything below describes this backend** unless stated.
  - `eager` ↔ `attn_implementation=eager` → `modules/windowed_eager_cache/`.
    Same cache/policy/state/store logic; it reads attention weights directly from
    the eager forward instead of reconstructing them, and it has no fused kernel
    (it always uses the materialize read path).
- **HuggingFace integration:** `WindowedCache` subclasses the HF cache base and
  implements `update()` (`modules/windowed_cache/cache.py:1481`), so the model's
  own `generate()` drives it. Scores are produced by **forward hooks** installed
  on the attention modules (`hooks.py:187` `install_score_hooks`).
- **transformers is pinned** to `<= 4.47.1`
  (`utils/cache_factory.py:44` `MAX_SUPPORTED_TRANSFORMERS`,
  enforced by `assert_transformers_version_supported`). Newer versions re-derive
  `cache_position` from the shrunken cache after eviction, which would desync the
  query's RoPE phase from the retained keys. This is an **accuracy guard**, not a
  convenience — it refuses to run rather than emit wrong numbers.

---

## 2. What the config knobs mean (resolved once at construction)

`WindowedCacheConfig` → `ResolvedConfig` turns fractions into concrete integer
counts (`modules/windowed_cache/config.py`). The ones that shape the paths below:

- `cache_budget` ∈ (0,1] — fraction of the full cache to keep, **by bytes**.
- `quant_ratio` (`q`) — the fraction of the retained budget carried in the 2-bit
  Q tier vs fp16. `q == 0` ⇒ single-tier (fp only); `q > 0` ⇒ two-tier.
- `quant_budget_mode` — `"bytes"` (default): `q` splits the **byte** budget, so
  cheaper int2 keys buy *more* retained windows at the same bytes. `"tokens"`:
  `q` splits the **window count** (used only by the latency suite, where both
  arms must attend over equal key counts). The flash and eager backends must
  agree here; the factory refuses `"tokens"` on eager
  (`cache_factory.py:169` `quant_budget_mode_kwargs`).
- `window_size` (`ws`) — tokens per window; the granularity of scoring & eviction.
- `num_sink_tokens` — immutable prefix, always kept, never scored.
- `local_tokens` / `local_windows` — most-recent windows always kept.
- `quant_gate_ratio` — fraction of the int2 tier a decode step actually reads
  when the gate is on (`1.0` = read everything, the gate's no-op control arm).

---

## 3. Two lifecycle phases (one switch, never switched back)

`update()` routes to one of two bodies (`cache.py:1481`, `:1521`):

1. **Per-layer** (`_update_per_layer`, `cache.py:1529`) — prefill and the *first*
   eviction. Each layer owns its own prompt-sized buffer.
2. **Layer-major / "joint"** (`_update_joint`, `cache.py:2074`) — every step after
   the first eviction. One store holds all `L` layers on a single `L·B` row axis;
   scoring, eviction and the position advance happen **once per step** for all
   layers instead of `L` times.

The switch happens in `_migrate_to_joint` (`cache.py:1780`), on the first
`update()` of the step *after* every layer has evicted once
(`cache.py:1509`–`1520`). It requires that **all** `L` layers evicted on the
first eviction step; a partial eviction raises, because a batched decode path
cannot be built on layers that are not in lockstep.

---

## 4. Prefill (strictly from the code)

### 4.1 Install
`install_score_hooks` (`hooks.py:187`) walks the model and, per attention module:
- registers a `q_proj` forward hook that **stashes the projected query**
  (`hooks.py:313` `make_qproj_stash_hook`) so the score pass reuses it instead of
  re-running the projection;
- registers the main `score_hook` (`hooks.py:356`, installed at `:563`);
- on CUDA, installs the **flash-LSE capture** (`hooks.py:263`
  `_install_lse_source` → `flash_lse.enable()`) and, if the fused decode kernel is
  enabled, arms it (`hooks.py:275`–`283`).

On CUDA the backend is **committed** (`hooks.py:231`–`232` `_committed = _cuda`):
every "scoring couldn't run" branch that would otherwise degrade to sink+local
instead **raises**. There is no silent unscored run under this method's name.

### 4.2 Append
The model runs a normal flash-attention-2 prefill. As each layer calls
`update()`, `_update_per_layer` (`cache.py:1529`):
1. appends this layer's K/V to its buffer (`state.append`, `cache.py:1560`) —
   keys are stored **post-RoPE**, at their real absolute positions;
2. records prefill length into the policy (`initialize_after_prefill`).

### 4.3 Score (the auxiliary pass)
flash-attention-2 never materialises the attention matrix, so scores are
reconstructed **after** each layer's attention forward, inside `score_hook`
(`hooks.py:356`):
1. rebuild the post-RoPE query from the stashed `q_proj` output + this layer's
   `position_embeddings` (`hooks.py:442`–`447`);
2. reuse flash's own softmax normaliser **L** captured this forward
   (`hooks.py:476`–`484`). If the capture misses, it **raises** — recomputing L is
   a second `O(N²)` pass whose fp32 block OOMs the big shapes, so it is never done
   silently (`hooks.py:485`–`533`);
3. `compute_token_scores` (`score_kernel.py`) → per-key received attention
   `[B, H_q, S]`. Prefill (`T>1`) runs the **Triton prefill score kernel** and
   raises if Triton/CUDA is unavailable — the PyTorch fallback was removed
   (`hooks.py:450`–`463`);
4. reduce to per-window scores (`scorer.reduce_token_scores_to_windows`, or
   `reduce_two_tier_scores` when `q>0`) and write them to
   `cache.cache_kwargs[layer_idx]["window_scores"]` (`hooks.py:550`–`559`).

Scoring is **H2O-style cumulative**: every query row contributes; there is no
observation window (`hooks.py:18`–`21`).

### 4.4 First eviction
Prefill itself cannot evict — the score hook runs *after* attention, so no scores
exist during the prefill `update()`. The prefill scores first become readable on
**decode step 0**, and `FIRST_EVICTION_STEP = 0` (`policy.py:41`) makes the first
compaction fire then: the prompt is cut to budget **before** step 0's query
attends, so every generated token is produced against the budgeted cache. That is
the operating point the SnapKV/AdaKV/DefensiveKV baselines are measured at.

---

## 5. Decode (strictly from the code)

After the first eviction the cache is layer-major. A decode step is: layer 0's
`update()` opens the step for everyone, then each layer writes its own rows and
gets its read tensors back.

### 5.1 `_begin_step` — once per step, all layers (`cache.py:1846`)
1. **Accumulate scores.** Concatenate every layer's pending `window_scores`
   (written by the previous step's hooks) in layer order and add into the running
   joint scores (`cache.py:1875`–`1893`). A partial set (some layers scored, some
   not) **raises** (`cache.py:1883`).
2. **Evict**, if the schedule says so (`_evict_joint`, `cache.py:2005`) — see §7.
   Eviction runs **before** this step's token is appended; `_check_score_width`
   (`cache.py:1920`) enforces that the score axis describes the store the eviction
   is about to compact.
3. **Reserve** this step's token slot for all layers and advance positions
   (`cache.py:1902`).

### 5.2 `_update_joint` — per layer (`cache.py:2074`)
1. Write this layer's K/V into its rows (`write_rows`, `cache.py:2138`).
2. If `q <= 0`, return the fp rows directly (`cache.py:2142`).
3. If `q > 0`, produce the **effective K/V** the attention will run over. Two
   routes:
   - **Fused (CUDA):** if `_fused_decode_active` and the Q tier is non-empty
     (`cache.py:2147`), build the kernel hand-off — gathered raw int2 fields +
     RoPE halves (`_fused_qtier_joint`), the score-scatter map
     (`_fused_meta_joint`), and the **gate context** (`_gate_ctx`) — and stash it
     via `flash_decode.set_pending(...)` (`cache.py:2150`–`2161`). The real flash
     call then dispatches into the fused two-tier decode kernel, which
     dequantises int2 **in registers**, attends, **and writes this step's window
     scores itself**. Because the kernel produces the scores, the score hook
     returns early on this exact condition (`hooks.py:383`–`388`) — no auxiliary
     pass runs.
   - **Materialize (eager/CPU reference):** otherwise `_materialize_joint`
     (`cache.py:2292`) dequantises + RoPEs the **whole** active Q tier
     (`store.effective_q_tier`) and concatenates `[sink ‖ body ‖ Q]`
     (`modules/quant/effective.py:235` `materialize_effective_kv`). This path does
     **not** gate — it reads every active window — so it needs no cards and no
     centroids. The score hook then scores the materialised K as in prefill.

### 5.3 Memoisation
Both the fused hand-off (`_fused_ctx`, rebuilt in `_build_joint_fused_ctx`,
`cache.py:2171`) and the dequantised Q tier (`_joint_qtier_read`, `cache.py:2274`)
are memoised on `store.version`, which only moves at eviction. So `ws-1` of every
`ws` steps are pure cache hits on the large tier; the work concentrates on the
eviction step.

---

## 6. The read gate (how decode skips int2 windows)

Only relevant at `q > 0` **on the fused path**. Motivation: the Q tier is by
construction the band ranked *below* the fp tier, so most of its windows
contribute almost nothing to a given step — yet a naïve decode dequantises all of
them (`sketch.py:1`–`8`).

### 6.1 The card (`modules/quant/sketch.py`)
When a window is sealed at demotion, a **rank-1 card** is written once and frozen
for life (valid forever because eviction never rebases positions):

    k_i ≈ mu + t_i · v          v = direction of the farthest deviation

`mu`/`vbar` are **int4** (packed 2/byte), `v`/`t` int8, each with an fp16 scale —
**272 B/head** at `H_kv=8, D=128, ws=8` (`sketch.py:41`–`54`). `mu` is stored as a
residual from a frozen per-(layer,head) **anchor** to spend its range on what
separates windows rather than on massive-activation channels
(`sketch.py:70`–`74`). A mean alone fails on the one-hot-token case, which is why
the model is rank-1, not a centroid (`sketch.py:18`–`32`).

### 6.2 Select
`_gate_ctx` (`cache.py:2374`) gathers the cards for the active slots and hands the
kernel `{card, anchor, centroid, n_sel}` where `n_sel = ceil(ratio · n_active)`.
The gate scores every window from its card (`sketch.gate_and_score`), unions the
GQA group so selection binds per **KV head** (the unit of kernel work), and picks
the top `n_sel` (`store.gate_and_select`, `store.py:369`;
`sketch.select_windows`/`group_max`). `ratio = 1.0` selects everything — the
gate's no-op control arm on the *same* code, not a second path
(`cache.py:2381`–`2385`).

### 6.3 Skipped windows still count, two ways
A skipped window is not simply ignored — that would corrupt both attention and
eviction:
- **Attention:** skipped windows **attend through their value centroid**
  (int4 `vbar`, dequantised in-register), added at the calibrated weight the read
  set implies (`decode_kernel.py:547`–`557`; eager reference of the same math).
  This is the "representatives attend" feature.
- **Eviction scores:** a skipped window is credited the estimate from its own
  card, rescaled onto the same footing as the real scores
  (`c = Σ exact / Σ estimate` over the read set), so it does not score zero, rank
  last, and get evicted next cycle (`scorer.fill_skipped_window_scores`
  docstring at `scorer.py:169`; the live implementation is inline in
  `decode_kernel.py:534`–`545`).

---

## 7. Eviction (two-tier, `cache.py:2574` → `:2706`)

### 7.1 Trigger
`policy.should_evict(step)` (`policy.py:94`): the first eviction at
`first_eviction_step` (0 by default), then every `step % ws == 0`.

### 7.2 The body (`_evict_two_tier_impl`, `cache.py:2706`)
Per row, entirely on device, **no Python loops and no host syncs** except one
pooled `.item()` for the ragged demote/promote widths (`cache.py:2906`):
1. **Rank + assign tiers** on the merged (fp-body ‖ Q) window axis
   (`policy.compute_two_tier_retain`) → `retained_idx`, `new_tier` (0=fp, 1=Q).
2. **Tier counts** `k_fp, n_q, local_w` from `policy.tier_counts(W)`
   (`cache.py:2843`), all host ints identical across rows.
3. Resolve retained window ids → slots once (`store.lookup`), then decide with
   masks (`cache.py:2861`–`2864`):
   - `demote` = fp now, Q wanted; split into `fresh` (never quantised → quantise
     now) and `react` (dormant entry → **reactivate with no requant**, exactly
     zero added error, design §10);
   - `promote` = Q now, fp wanted → dequantise + RoPE back to fp16.
4. Free dropped slots (`store.retain_only`), rebuild the fp store as
   `[sink ‖ fp survivors in ascending-id order]` **keeping every survivor's
   original absolute position** (`cache.py:2868`–`2918`; positions never
   renumbered).
5. Install the compacted body *outside* the graph (`cache.py:2664`–`2685`).

### 7.3 Compiled-or-error, layer-major
The whole body runs under a lazily-built `torch.compile`
(`_run_compiled_evict`, `cache.py:968`). It is **never allowed to run eager**: a
`torch.compiler.is_compiling()` tripwire inside the body raises `EvictionRanEager`
if Dynamo fell back (`cache.py:2776`), because an eager run would report eager
timings under a "compiled" label. `layer_idx`, `step`, and the `joint` bool are
kept **out** of the compiled signature (`cache.py:2608`–`2635`) so Dynamo builds
one graph, not one per layer/step. Passing `layer_idx=None` evicts all `L` layers
in one pass with identical arithmetic at `1/L` the launches (`cache.py:2729`).
`q == 0` takes the single-tier retain instead (`_evict_joint`, `cache.py:2021`).

---

## 8. Quantization & storage

- **Q tier:** hand-rolled **KIVI-style affine int2** (`quantizer.py:1`), codes in
  `[0,3]`, 4 crumbs per byte, grid one byte per entry. Keys stored **pre-RoPE**;
  RoPE applied at read from each window's frozen `position_range`. Values carry no
  RoPE (asymmetric store) (`store.py:19`–`20`).
- **Store:** `QuantizedStore` (one per layer, or one joint store) over a dense
  `QuantSlotTable`; all ops are per-**slot**, masked, ragged-width-safe
  (`store.py:1`–`31`). `demote_many` / `reactivate_many` / `promote_many` /
  `effective_q_tier` are the mutators/readers; every mutator bumps
  `store.version`.
- **Effective read** (`effective.py:235` `materialize_effective_kv`): emits
  `[sink ‖ fp body ‖ Q]` **unsorted** — attention is order-free, so the per-token
  reorder is skipped and only a per-*window* scatter map (`compute_score_meta`,
  `effective.py:204`) is handed to the scorer to undo the layout on the tiny score
  axis.

---

## 9. The two functions you asked about

Both are **orphaned leftovers of commit `0974687` "refactor: one production path,
no fallbacks"**, which consolidated the gated-decode logic into `decode_kernel.py`
(the Triton kernel plus its eager reference). Neither is work-in-progress
scaffolding, and neither is on any live or reference path. The *features* they
relate to are shipped — just implemented elsewhere.

### 9.1 `scorer.expand_keep_to_query_heads` (`scorer.py:218`)
- **Purpose:** expand a per-KV-head keep mask `[B, H_kv, W]` → per-query-head
  `[B, H_q, W]` (a window read for a KV head was read for every query head in its
  GQA group), to feed `fill_skipped_window_scores`.
- **Status: dead.** Zero callers (code or test). Its job now happens **inline** in
  the decode reference: `sel_q = sel.repeat_interleave(H_q // H_kv, dim=1)`
  (`decode_kernel.py:518`). Git: it had a caller until `0974687` removed it.
- **Note:** `scorer.fill_skipped_window_scores` (`scorer.py:169`), the function it
  fed, is in the *same* state — reimplemented inline at `decode_kernel.py:534`–`545`
  and now cited only in comments. It is the spec/reference for that block.

### 9.2 `store.window_centroids` (`store.py:422`) → `sketch.value_centroid` (`sketch.py:303`)
- **Purpose:** produce each active window's dequantised **value centroid** — "what
  a skipped window contributes to the attention output" — so skipped windows can
  still attend.
- **Status: dead wrapper.** The "representatives attend" feature is **live**, but
  the production path never calls it: `_gate_ctx` passes the **raw int4 centroid
  columns** straight to the kernel (`cache.py:2407`
  `"centroid": (card[-2], card[-1], store._v_anchor)`), which dequantises them
  in-register (`decode_kernel.py:547`–`557`). The materialize path reads the whole
  tier and needs no centroids at all. So `window_centroids` has zero callers, and
  its only dependency `value_centroid` is reachable solely through it plus one
  unit test (`tests/test_sketch.py`).

**Bottom line:** removing `window_centroids` and `expand_keep_to_query_heads`
(and, if desired, the now-inline `fill_skipped_window_scores`) is
accuracy-neutral — they are superseded, not pending. `value_centroid` would need
its unit test removed with it, so it is best left unless that test is also
retired.

---

## 10. Invariants that protect accuracy (why the cleanups above are safe)

- **Positions are never renumbered** across eviction; survivors keep their
  original absolute RoPE positions (`cache.py:2740`, `effective.py`). This is why
  the transformers version is pinned (§1).
- **Kernel-or-error, everywhere it matters:** prefill scoring (`hooks.py:450`),
  L-reuse (`hooks.py:485`), and the compiled eviction (`cache.py:2776`) all raise
  rather than silently degrade — so a run either does the method or stops.
- **Rectangularity:** retained tier counts are equal across rows
  (`tier_counts`), so divergent per-row eviction stays dense — no padding, no
  keep-mask on the effective K/V (`store.py:22`–`31`).
- **Gate never destroys the tier:** skipped windows attend via centroids and are
  score-credited from their cards (§6.3), so reading less of the tier does not
  change what the tier retains.
- **The gate's `ratio=1.0` is the shipped path with nothing skipped** — the
  control arm and production run one implementation (`cache.py:2381`).

---

## Appendix — file map (current tree)

| Path | Role |
|---|---|
| `modules/windowed_cache/cache.py` | `WindowedCache`: HF `update()`, per-layer→joint migration, decode step, two-tier eviction, fused hand-off |
| `modules/windowed_cache/hooks.py` | Prefill/decode score hooks, q_proj stash, flash-LSE capture, fused-decode arming |
| `modules/windowed_cache/policy.py` | `EvictionPolicy`: trigger cadence, tier counts, retain indices |
| `modules/windowed_cache/state.py` | `CacheState`: fp buffers, append/reserve/write_rows/slice_and_keep/replace_body |
| `modules/windowed_cache/scorer.py` | Pure score reductions; `expand_keep_to_query_heads`/`fill_skipped_window_scores` (orphaned) |
| `modules/windowed_cache/score_kernel.py` | Triton prefill score kernel (`compute_token_scores`) |
| `modules/windowed_cache/decode_kernel.py` | Fused two-tier decode: Triton kernel + eager reference (gate, centroids, score write-back) |
| `modules/windowed_cache/flash_decode.py` / `flash_lse.py` | Dispatch seam into the decode kernel / softmax-L capture |
| `modules/quant/store.py` | `QuantizedStore`: demote/reactivate/promote, effective/gated Q tier; `window_centroids` (orphaned) |
| `modules/quant/quantizer.py` | int2 affine quantiser + int-value quantiser |
| `modules/quant/sketch.py` | Rank-1 cards, `gate_and_score`, `select_windows`, `value_centroid` |
| `modules/quant/effective.py` | `materialize_effective_kv`, RoPE primitives, `compute_score_meta` (CPU reference) |
| `modules/quant/slots.py` / `compact.py` | Dense slot table; `join`/`stable_partition` |
| `utils/cache_factory.py` | Backend selection + pairing/version guards |
