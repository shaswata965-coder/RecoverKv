# StickyKV — Quantization Design

Two-tier windowed KV cache: **top-K** windows in full precision (fp16) plus
**top-Q** windows in **int4** (hand-rolled KIVI-style), with a **per-window pinned
scale/zero-point**. The int4 (Q) tier stores keys **pre-RoPE**; RoPE is applied
fresh at read using each window's **original absolute positions** (which eviction
never changes). This document records only **fixed** design choices. Supporting analysis, trade-offs, and open items live in
[design_rationale.md](design_rationale.md); the decision record (original prompt,
amendment log, rejected alternatives) lives in [design_history.md](design_history.md).

---

## 1. Architecture overview

A window is a fixed-size chunk of the sequence (`window_size` tokens). Every
`window_size` steps the cache scores each window by accumulated attention and ranks
them into three outcomes:

- **K tier (fp16)** — the highest-ranked windows, plus the always-kept **sink**
  (first tokens) and **local** (most recent) windows. Stored full precision.
- **Q tier (int4)** — the next band of windows: not good enough for fp16 but too
  useful to drop. Stored quantized at ¼ the memory.
- **Dropped** — everything else.

The two tiers live in two separate **gap-free** dense stores (§4). Windows migrate
between tiers by ranking (promotion K←Q, demotion K→Q) and can be dropped. On **every**
eviction the cache **compacts** survivors in memory — jointly ranked across both tiers
(§5) — but **does not renumber positions**: surviving keys keep the RoPE rotation baked
in at their original absolute positions, and keys are never stripped and re-rotated.
The query keeps its natural monotonic (absolute) position from HuggingFace, so the
query↔key relative phase is preserved without any position override. Both attention
backends (`windowed_cache`, `windowed_eager_cache`) share this logic (§9).

The design is realised in three kernel phases (§11); Phase 1 (materialize-then-
interleave) is the shippable v1 and is fully CPU-testable.

---

## 2. Quantization scheme

- **Granularity.** Keys are quantized **per-channel at the window-index level** —
  one scale/zero per `(head, channel, window)`. Values are quantized **per-token**.
  Quant error is set by a group's **dynamic range (max−min), not its count**: a
  single global scale is pinned by the largest outlier and obliterates small/median
  values, so groups are kept fine to localise range. But not arbitrarily fine — each
  group costs a scale + zero, so over-fine grouping eats the int4 savings.
- **Affine, asymmetric — numerics pinned.** The distributions are skewed, so use
  asymmetric affine int4 quantization, KIVI-reference float-offset form, computed in
  fp32:

  ```
  scale = (mx − mn) / 15                # mx/mn over the quant group
  zero  = mn                            # float offset (not an integer zero-point)
  q     = clamp(round((x − zero) / scale), 0, 15)   # round-half-even; clamp BEFORE the uint cast
  x̂     = q · scale + zero
  ```

  Degenerate group (`mx == mn`): set `scale = 1` → all codes 0 and `x̂ = mn`
  exactly. **Scales and zeros are stored fp16 (pinned).** The fp8-scale option is
  dropped: it saves ~1.5% of Q-tier bytes while injecting scale-quantization noise
  into every dequant. Quantization runs against the **fp16-stored** `scale`/`zero`
  (not the fp32 intermediates), so the grid the codes were fit to is bit-identical
  to the grid used at every dequant. The float-offset form is used rather than an
  integer zero-point because K/V groups often exclude zero — an integer zero-point
  clamped to `[0, 15]` cannot represent an offset outside the group's own span.
- **Pinned grid.** Each window's **scale and zero-point are pinned at first
  quantization and never recomputed**. Codes are written exactly once in a window's
  lifetime: re-reads only dequantize, and a re-demotion **reactivates** the stored
  ledger entry instead of re-quantizing (§10). Zero drift and zero compounding are
  therefore **structural** guarantees — no arithmetic path ever runs
  `quant(dequant(·))` — not numerical ones (§3, §8).
- **Nibble packing (decided).** Pack the two int4 codes that **share a scale** into
  one byte — i.e. pack along each tier's quantization-group axis:
  - **Keys** (per-channel scale, group = window along the token axis): store the Q
    store **channel-major** `[H_kv, D, T_q]` and pack 2 consecutive tokens per byte →
    `[H_kv, D, ceil(window/2)]` uint8, scale/zero `[H_kv, D]` per window.
  - **Values** (per-token scale, group = head_dim): keep token-major `[H_kv, T_q, D]`
    and pack 2 consecutive channels per byte → `[H_kv, T_q, ceil(D/2)]` uint8,
    scale/zero `[H_kv, T_q]`.

  Both nibbles in every byte then share one scale → a single scale load per byte pair,
  branchless vectorized dequant, and (Phase 2) tile = window = scale group = packing
  group. Nibbles are unsigned `[0,15]` (asymmetric zero-point); even-index code in the
  low 4 bits. `window_size` must be even (head_dim always is), so no tail padding.
  **Bring-up path:** validate with unpacked codes (one 4-bit value per byte, so
  `torch.gather` works token-wise), then switch to this packed layout; Phase 2 repacks
  into uint32 words (8 nibbles) for 32-bit-aligned loads, same group axis.
- **Outlier handling (v1: none).** v1 ships **no outlier machinery**: per-channel
  key scales, pre-RoPE storage, and the two-tier split (the largest spikes sit in
  the fp tier, and the sink is always fp16) already absorb the dominant outlier
  structure. NF4-style key codebooks and the value Hadamard fold are **deferred
  beyond v1 altogether** (decision record in [design_history.md](design_history.md);
  analysis retained in [design_rationale.md](design_rationale.md)). The sole
  contingency — a micro dense-and-sparse fp16 side-list (~0.1–0.25%) — is added
  **only if** the int4 LongBench gate misses.

The quantizer is a shared module implementing exactly the scheme above; no numeric
details remain open.

---

## 3. No re-quantization; scoring is read-path only

Past KV is immutable. There is **no re-quantization** of stored windows:

- The Q tier is **dequantized for read/attention each step** so it continues to
  accrue `window_scores`. This dequant-for-scoring is a **read-path cost** (§8), not
  a re-quant.
- The new/local window is born in fp16 and is quantized **at most once** — only if it
  is later demoted into the Q tier. Codes are computed at the **first** demotion
  only; any later re-demotion reactivates the stored ledger entry (§10).
- **Promotion/demotion decisions are pure score-ranking arithmetic** ("where does
  this window land in the ranking?"). No dequantization and no concatenation is
  needed for the *decision*; those happen only in the attention read path.

---

## 4. Two dense stores (not a zero-padded tensor)

Two separate, gap-free dense stores per layer:

- **fp store** — `[B, H_kv, T_fp, D]` fp16 keys/values + `position_ids`. Keys are
  rotated in place.
- **Q store** — int4 codes + per-window scales/zeros + `position_ids`. Keys are
  stored **pre-RoPE** and rotated **at read** using each window's **original absolute
  positions** (fixed at window creation; eviction never rebases them).

A full-length fp tensor with zeros in the Q slots is **rejected**: it wastes memory
and zero keys are not softmax-neutral (`exp(q·0) = 1`). RoPE needs only a
`position_id`, not physical co-location, so at read the dequantized Q keys are merged
with the fp store into one effective tensor. The merge is **chronological by window
id** (§5) — correct for attention regardless of order, but chronological so the window
scorer's physical chunking stays valid. The shared cross-store layout is a **logical
index/tier map** (the per-window ledger, §6), not a tensor. A window's tier is
implicit: tier *is* which store holds it.

---

## 5. Eviction cycle and effective-KV materialization

Every eviction runs a single compaction that spans **both tiers jointly** — but
**positions are never renumbered**. Each surviving window keeps its original absolute
positions forever; eviction only *compacts memory* (gathers survivors contiguous) and
migrates windows between tiers. Joint handling is required for two reasons, neither of
them positional: **(a)** ranking and tier assignment must consider both tiers at once
(a Q window can outrank an fp window and get promoted); **(b)** the window scorer chunks
the physical key axis, so the materialized effective K/V must be in chronological window
order (below).

**The cycle, per eviction:**

1. **Rank the evictable windows** by accumulated `window_scores` (on the merged window
   axis below). Sink and local windows are **not** ranked — exactly as today
   ([policy.py:135](modules/windowed_cache/policy.py:135) `topk`s only the evictable
   slice and force-appends local).
2. **Assign tiers.** Sink + local are **force-fp** (never eligible for Q — a
   low-scoring local window must stay full precision, not fall to int4). Only the
   **evictable band** is partitioned by rank: top `top_k_fp` → fp; next `N_q` → Q; the
   rest → dropped (`top_k_fp`, `N_q` from the budget resolver, §7). The fp store then
   holds `sink + local + top_k_fp`; the Q store holds `N_q`.
3. **Move boundary-crossers.**
   - **Demote (K→Q), first time:** un-rotate the window's fp keys **once** (apply
     RoPE with negated sin at the window's *original absolute positions*, the
     `position_ids` `slice_and_keep` already carries), quantize against a freshly-pinned
     grid, append
     to the Q store, remove from the fp store. Record the window's original absolute
     positions as its immutable `position_range` (§6). A window that has been demoted
     before is **reactivated, not re-quantized**: its dormant ledger entry (codes +
     pinned grid + `position_range`) is still live, so demotion is drop-the-fp-copy +
     mark the entry active (§10).
   - **Promote (Q→K):** dequantize the window's codes, apply RoPE at the window's
     original absolute positions, and **splice the window into the fp store at its
     chronological (`original_window_id`) position** — *not* appended at the end (§4
     invariant below) — removing it from the Q store **but keeping its ledger entry
     dormant** (codes + pinned grid + `position_range`) for a possible future
     re-demotion (§10).
4. **Reassemble each store in memory.** Because demotion *removes* fp windows and
   promotion *inserts* one, `slice_and_keep` alone (which only drops) is insufficient
   when a promotion occurs. Reassemble the **fp store as
   `[sink ‖ fp windows in chronological `original_window_id` order]`**: gather the
   surviving-fp tokens from the old fp store (via `slice_and_keep` on the fp-partition
   token indices, keeping their original absolute `position_ids` — no re-rotate, no
   renumber) and splice each promoted window's freshly-rotated keys at its chronological
   slot. **Invariant (required by the §5 index translation): the fp store's physical
   window order equals chronological window order at all times** — this is what makes
   `rank_fp = cumsum(fp_mask)` and `num_sink + rank_fp·window_size + offset` correct on
   the *next* eviction. The Q store has **no** such constraint: it compacts by shifting
   each surviving window's byte `offset` (any physical order is fine — the ledger
   `offset` locates each window's codes); `codes`, `scale/zero`, and `position_range`
   are untouched. When no promotion occurs, the fp reassembly degenerates to today's
   single `slice_and_keep`, and at `q = 0` the whole step is byte-identical to `main`.

**No interleaved position map, no query override.** Because positions are never rebased,
there is no `arange(T_total)` renumbering, no `rerotate_keys(new_pos)`, and no
`position_override`. The query gets its natural monotonic absolute position from
HuggingFace (correct on the pinned transformers ≤ 4.47.1, exactly as the single-tier
cache relies on today), and each surviving key — fp or Q — carries its true absolute
RoPE phase, so query↔key relative distance is exact across both tiers with no
per-eviction position bookkeeping.

**Why pre-RoPE (in the keep-original-positions world).** Since positions never change,
a demoted window's post-RoPE fp keys are already a *fixed* tensor — so post-RoPE codes
would also be frozen, and pre-RoPE is **no longer a correctness requirement** (it was,
under the earlier rerotation design, because positions were rebased every eviction —
see [design_history.md](design_history.md) Amendments 2–3, now superseded by Amendment
5). Pre-RoPE is retained for two standing reasons: **(1) quantization quality** —
per-channel key ranges are tighter and more consistent pre-RoPE, since RoPE smears the
inter-channel structure the per-channel scales exploit (see
[design_rationale.md](design_rationale.md)); and **(2) Phase-2 kernel consistency** —
the fused Triton tile applies RoPE in-register from cos/sin (§11), so the store layout
is pre-RoPE end-to-end. Either way the codes are frozen once written: because the
window's pre-RoPE content is immutable **and** its `position_range` never changes,
pinned-grid idempotence holds and a re-demotion is an exact reactivation, never a
re-quant (§10). Values carry no RoPE in either tier (asymmetric store).

**Effective K/V is materialized in chronological window order.** *Attention* is
order-free — RoPE has baked each key's logical position into its values, so Q·Kᵀ is
correct regardless of physical order. But the window **scorer** chunks the physical key
axis into windows (`b h (w s) -> b h w`), so it requires physical order = chronological
window order. Therefore `materialize_effective_kv` interleaves the fp and Q windows by
`original_window_id` (a cheap gather over `N_fp + N_q` windows, not a plain
`[fp ‖ Q]` concat), yielding `[sink ‖ windows in chronological order]`. This keeps the
existing scorer, sink-strip, and cumulative score accumulation unchanged.

### New/changed primitives (relative to the current cache)

- **Demotion un-rotate** (one-time, at first demotion). To produce pre-RoPE codes from
  a window's post-RoPE fp keys, apply RoPE with **negated sin** at the window's
  *original absolute positions* (`cos(−θ)=cos θ`, `sin(−θ)=−sin θ`). Read the window's
  `position_ids` slice from the fp store **before** `slice_and_keep` drops that window
  (the slice is needed both for the un-rotate angles and to record the immutable
  `position_range`). This is a standalone op — the removed `rerotate_keys` is **not**
  reintroduced; there is no re-rotate half, because the fp tier is never re-rotated. It
  needs the model's rope module, so the two-tier path (only when `q > 0`) re-introduces
  a rope-module dependency that the single-tier `q = 0` path does not have.
- `materialize_effective_kv(fp_store, q_store, ledger, rope)` — dequantizes the Q
  store, applies RoPE at each Q window's **immutable** `position_range` (its original
  absolute positions), and interleaves with the fp store by `original_window_id` into
  chronological order, emitting `[sink ‖ windows in chronological order]` — the **sink
  prefix (the fp store's first `num_sink` tokens) is carried through unchanged** so the
  scorer's `reduce_token_scores_to_windows(·, num_sink, window_size)` still strips
  exactly the sink. Returns effective K/V. Used **both** as `update()`'s return and by
  the flash score hook (which currently reads raw `key_states`). Operates **per row**
  (rows may retain different windows, as the current cache already does).
- `cache.get_seq_length()` must report the **effective** length `T_total = T_fp + T_q`,
  since HF uses it to size the causal mask over the returned effective K/V and the flash
  hook uses it as the key count `S`. This is **only** a key-count report for masking —
  there is **no** query-position override (removed with rerotation); query positioning
  follows HF's monotonic absolute positions exactly as in the single-tier cache.
- `expand_to_token_indices` becomes **tier-aware** (see the merged window axis
  below): it expands only the **fp partition** of the retained merged indices into
  fp-store token indices (for `slice_and_keep`); Q windows are handled entirely through
  the ledger and never through a token gather.

### The merged window axis (one index space for scores, ranking, and retention)

All per-window bookkeeping lives on **one axis**: the merged chronological window
axis — every surviving window from **both tiers**, sorted by `original_window_id`,
exactly the order `materialize_effective_kv` emits. Invariant: merged-axis index `i`
↔ the `i`-th chronological surviving window ↔ the `i`-th physical window chunk of
the effective K/V. Like `original_window_ids` today, the axis is **per row** (rows
may evict different windows). Concretely:

- `window_scores` `[B, H_q, W]` and `original_window_ids` `[B, W]` index the merged
  axis (`W = W_fp + W_q`). The scorer chunks the effective K/V — materialized in
  merged order — so its output aligns with this axis by construction; `accumulate`
  and `update()`'s W-growth bookkeeping run unchanged, just on the merged axis.
- **Ranking and tier assignment** (steps 1–2 above) run on the merged axis and
  partition its indices into fp survivors, Q survivors, and dropped.
- **Index translation is tier-aware.** A merged index no longer maps to a physical
  offset by arithmetic alone; it resolves through the tier map (§4):
  - **fp window** → its token range in the fp store. This relies on the §5 step-4
    **invariant** that the fp store's physical window order equals chronological
    (`original_window_id`) order, so the fp-store window rank is a cumsum over the
    fp-tier mask and token indices are `num_sink + rank_fp · window_size + offset`.
    (Promotion must therefore splice into the fp store at the chronological slot, not
    append — §5 step 3.)
  - **Q window** → its ledger entry (codes / scales / `position_range`), located by
    `offset`. The Q store needs **no** ordering constraint and the window has **no**
    fp-store token indices.
- The sink prefix sits outside the axis (never scored, always fp), and the local
  windows are the trailing merged indices (newest ids), so the local-protection
  slice in `compute_retain_window_indices` is unchanged.

**Policy bookkeeping under a physically split store.** `EvictionPolicy` today derives
its region counts from a **single** contiguous length: `total_tokens` →
`post_sink_tokens` → `num_total_windows` → `num_evictable_windows`
([policy.py:64](modules/windowed_cache/policy.py:64)). With the store split, two
distinct lengths must not be conflated:

- **Merged-axis counts (windows).** Ranking, local protection, and the evictable/local
  split operate on the **merged window count** `W = W_fp + W_q`, which
  `compute_retain_window_indices` already reads off `window_scores.shape[2]` — *not*
  off `total_tokens`. So the ranking path is correct as long as the policy's window
  math is driven by `W` (the scores-tensor width), which is the merged logical length,
  **not** the fp store's physical token count.
- **fp-store count (tokens).** `expand_to_token_indices`'s fp-partition expansion and
  its OOB trim (`idx < total_tokens`) must use **`T_fp`** (the fp store's physical
  length, what `state.key_states.shape[2]` reports), since only fp windows have
  physical token indices. Q windows never enter this expansion.

Concretely: keep `EvictionPolicy`'s logical `total_tokens` tracking the **merged**
length (`T_fp + T_q`) so `num_total_windows` / `num_evictable_windows` / local counts
stay aligned with the merged axis and with `get_seq_length()` (§5 primitives), and pass
the fp-store physical length `T_fp` explicitly to the tier-aware
`expand_to_token_indices` for the fp gather. `set_total_after_compaction` therefore
records `T_fp + T_q`, **not** `state.seq_length` (which is now fp-only). At `q = 0`,
`T_q = 0` so `total_tokens == state.seq_length` and every count reduces to today's.

**First eviction (single store → split).** Prefill builds one fp store and scores it;
the **first** eviction is where the Q tier is born — the evictable band that lands in
the `N_q` band is demoted (un-rotated once, quantized), everything above it stays fp,
everything below is dropped. Every later eviction is the steady-state cycle above. At
`q = 0` there is never an `N_q` band, so no store is ever split.

---

## 6. Per-window ledger

A small record keyed by `original_window_id` tracks each surviving Q window across
evictions. The fp tier needs no ledger — it is a plain dense tensor.

| field | mutable? | purpose |
|---|---|---|
| `original_window_id` | no | chronological identity; used for the interleaved sort |
| `codes` (int4) | no | packed quantized bits; never change after demotion |
| `scale`, `zero` | no | pinned affine grid; set once at demotion |
| `position_range` | no | the window's **original absolute positions**; set once at demotion; what the read path feeds to RoPE |
| `offset` | yes | byte offset into the Q store; shifts as the Q store compacts |

`offset` is the only field that changes at eviction cadence — the Q store compacts, so
each entry's byte offset shifts (an O(Q_windows) integer update, no tensor movement, no
re-quant). `position_range` is **immutable**: because eviction never renumbers, a
window's absolute positions are fixed at creation, so the read-path RoPE angles for a Q
window never change — only its physical `offset` in the store does.

Entries **persist through promotion**: a promoted window's entry goes **dormant**
(codes + scale/zero retained; `offset` invalid; excluded from reads and the
interleave) rather than being freed, so a later re-demotion is a pure reactivation
(§10). An entry is freed only when its window is dropped outright. Dormant codes are
a small, freeable overhead — bounded by the fp tier's window count at int4 size —
which the v1 budget resolver may ignore.

---

## 7. Tier-aware budget resolver

A `quant_ratio` knob `q` splits the retained budget between tiers. The split is
expressed in **memory**, but it must be folded into the **existing token-granular
`resolve()`** ([config.py](modules/windowed_cache/config.py)) so that `q = 0`
reproduces today's `ResolvedConfig` **field-for-field** — not layered on top as a
separate window-count formula. The current resolver works entirely in tokens:

```
bytes_per_token     = 4 · H_kv · D            # K + V, fp16   (= 2·H_kv·D·2)
total_budget_bytes  = β · (prefill_len + max_tokens) · bytes_per_token
total_budget_tokens = total_budget_bytes // bytes_per_token
                    = ⌊ β · (prefill_len + max_tokens) ⌋
# sink and local are carved as TOKENS, before any //window_size:
top_k_windows       = (total_budget_tokens − num_sink_tokens − local_tokens) // window_size
```

Two structural facts the tier split must respect:

- **`num_sink_tokens` is a raw token count, not a window multiple** (e.g. 4 tokens with
  `window_size = 8`). The sink lives **outside** the window axis (§5) and is carved at
  token granularity. It is **always fp** and is *not* a member of either tier's window
  count.
- **`local_tokens` is already a `window_size` multiple** (int input validated, or a
  float ratio of `total_budget_tokens` snapped up — [config.py:208](modules/windowed_cache/config.py:208)).
  `num_local_windows = local_tokens / window_size`. Local windows are **always fp** (§5).

**The tier split (fold into `resolve()`).** Only the **evictable** window budget — what
`top_k_windows` counts today — is divided between tiers; sink and local are untouched
and stay fp:

```
b_fp = bytes_per_token · window_size = 4 · H_kv · D · window_size    # one fp window
b_q  = H_kv · D · window_size      # packed int4 codes, K + V   (= ¼ · b_fp)
     + 4 · H_kv · D                # key scale+zero, per (head, channel), fp16
     + 4 · H_kv · window_size      # value scale+zero, per (head, token), fp16

# evictable budget in bytes, then split by q:
M_evict = (total_budget_tokens − num_sink_tokens − local_tokens) · bytes_per_token
top_k_fp = ⌊ (1 − q) · M_evict / b_fp ⌋     # fp evictable windows  (Q tier disabled ⇒ q=0 ⇒ = today's top_k_windows)
N_q      = ⌊     q   · M_evict / b_q ⌋      # int4 windows           (uses b_q, NOT b_fp)
```

`N_fp = num_sink_windows_equivalent + num_local_windows + top_k_fp` describes the fp
store, but the resolver only needs `top_k_fp` (the evictable fp count) and `N_q`; sink
and local are already handled by the untouched `num_sink_tokens` / `local_tokens`.

- The resolver **must use `b_q`, not `b_fp`, for the Q tier** — the int4 tier holds ~4×
  the windows of equal fp memory (minus overhead). Scale dtype is **fixed fp16** (§2),
  so `N_q` is deterministic with no dtype input.
- **`q = 0` parity is exact by construction:** with `q = 0`, `N_q = 0` and
  `top_k_fp = ⌊ M_evict / b_fp ⌋ = (total_budget_tokens − num_sink_tokens − local_tokens)
  // window_size` — identical to today's `top_k_windows`. Keep the existing
  `bytes_per_token` / `total_budget_bytes` / `total_budget_tokens` / `top_k_windows`
  fields so the pure-fp16 path is byte-identical; add `top_k_fp` (= `top_k_windows` at
  `q=0`) and `N_q` (= 0 at `q=0`) as new fields.
- **Float `local_window_size` is resolved first, unchanged.** `local_tokens` is fixed
  before the tier split (it is a fraction of `total_budget_tokens`, the whole retained
  budget — not of `M_fp`), so local sizing is independent of `q` and the local region
  can never exceed the budget.

*Example* (β = 0.25, q = 0.5): the evictable budget splits 50/50 by memory ⇒
`N_q ≈ 4 · top_k_fp` ⇒ ~4× as many evictable windows kept in int4 as in fp, at the same
byte cost.

New config knobs: `β` (exists as `cache_budget`), `q` (default **0.0**), bit-width
(fixed 4 in v1), group size (= `window_size`). Scale dtype is fixed fp16 (§2) — not a
knob.

---

## 8. Read / attention path and per-step cost

**Read path (v1, materialize).** For each Q window: look it up in the ledger, take
the int4 codes, dequantize to fp16, apply RoPE at the window's **original absolute
positions** (the immutable `position_range`), then interleave the fp and Q windows
chronologically by window id (§5) and hand the result to the standard attention path.
The fp tier is already rotated and ready. `update()` returns one normal fp tensor.

**Cost.** Recent/local + sink + top-K stay fp, so the most-attended tokens skip the
slow path. The per-step Q cost is: dequant + one RoPE apply (arithmetic on already-
dequantized data — bandwidth-trivial). The real v1 cost is the **fp16 write-back**:
the Q tier blooms to fp16 transiently, but attention runs **layer-by-layer**, so only
one layer's Q tier is live in fp16 at any moment — peak impact ≈ `(Q-fp size) /
num_layers` (~1–2% of the full cache at 32 layers), freed immediately after each
layer. Phase 2 eliminates the write-back entirely (§11).

**Scoring.** Because the effective K/V is materialized in **chronological window
order** (§5), the window scorer's physical chunking (`b h (w s) -> b h w`) stays
valid unchanged and both tiers accrue `window_scores` — the scores land on the
merged window axis (§5). This does require the effective
K/V — not the raw fp `key_states` — to be what gets scored: the eager backend already
attends over `update()`'s return, but the flash score hook currently reads raw
`cache._states[l].key_states` and must instead source the effective K via
`materialize_effective_kv` (§9).

**Benchmark gate.** Suite C (`perf_runner.py`) must confirm memory savings outweigh
TPOT impact in v1 before moving to Phase 2.

---

## 9. Backend mirroring

"Mirror" means the **byte-identical** `cache.py` / `state.py` (+ the new shared quant
module and ledger), **not** `hooks.py`. The two backends diverge only in hooks: flash
recomputes scores via an auxiliary SDPA; eager reads materialized attention weights —
no aux SDPA is added to eager.

- There is **no** shared query-position override (`utils.position_override` was removed
  with rerotation); both backends rely on HF's monotonic absolute query positions. The
  shared change is `get_seq_length()` reporting `T_total` (§5) so the mask covers both
  tiers.
- Shared `update()` returns the effective K/V via `materialize_effective_kv` (Q tier
  is pre-RoPE, so the read path dequantizes, applies RoPE at each window's original
  absolute positions, and interleaves with the fp tier in chronological order — §5).
  The eager backend attends over this return, so its real-attention scoring works
  unchanged.
- The flash score hook must **also** source effective K via `materialize_effective_kv`
  — it currently reads raw `key_states` (`hooks.py`), which would miss the Q tier and
  the interleaving. This is the one required flash-hook change.
- Transient dequant is **per-layer** (§8).

**Implementation note:** `update()` currently returns the live `state.key_states /
value_states`. With the Q tier it returns a freshly-built interleaved effective tensor
instead; callers must not assume the returned tensor aliases the stored fp cache.

---

## 10. Key design choices

- **Full bidirectional promotion in v1** (per the original prompt). This accepts the
  ledger bookkeeping and the score-feedback risk; both are instrumented via promotion-
  frequency telemetry and **Suite A Jaccard-vs-fp-only** over long sequences. (The
  score-feedback loop and the not-chosen fallback are documented in
  [design_history.md](design_history.md).)
- **Ledger entries persist through promotion; re-demotion is a reactivation.** A
  promoted window's ledger entry (codes + pinned scale/zero + immutable
  `position_range`) is **retained dormant**, not freed (§6). Because past KV is
  immutable and positions are never rebased, the window's pre-RoPE content *and* its
  read-path RoPE positions can never change — so a later demotion does **not**
  un-rotate or re-quantize: it drops the fp copy and reactivates the stored codes. Zero
  compute, and **exactly** zero added error. (A re-quantization path would pick up fp16
  rounding from the intermediate fp-tier dequant/re-rotate on promotion and could flip
  boundary codes — this is why reactivation, not "re-quantize against the old grid", is
  the spec.) This is what "pinned grid by identity" means operationally, and it makes
  promote→demote oscillation free without explicit hysteresis.
- **Quant group = the eviction window.** This pins one grid per window, which
  promotion requires. `window_size` and bit-width are empirical knobs swept in
  Suite C / LongBench — **no hardcoded floor**; pick by measured effective-
  bits-vs-quality. (Scale dtype is fixed fp16, §2.) Effective key bits ≈
  `4 + 32/window_size` (expectation-setting, not a rule).
- **Precision: fp16 K tier, int4 Q tier.** Full-precision windows are fp16; the
  quantized tier is int4 (§2, §7).
- **v1 is batch-size 1 only.** The single-tier cache it builds on is already B=1 (it
  relies on HF's monotonic absolute query position and evicts per row), and
  `materialize_effective_kv` runs per row; B>1 is **out of scope for v1** and gated
  behind the ragged left-padded batching design. Keep the per-row primitives in §5 so
  B>1 is a later extension, not a rewrite.

---

## 11. Kernel roadmap (three phases)

### Phase 1 — v1: materialize-then-interleave (ship first)

Dequantize the entire Q store to fp16, interleave with the fp store chronologically by
window id (§5), pass to the standard attention path unchanged. No custom kernels; fully
CPU-testable;
correctness is the only goal.

- **Q-tier RoPE:** pre-RoPE — codes stored un-rotated; read path is `dequantize →
  apply RoPE at the window's original absolute positions → interleave by window id`.
- **Memory peak:** a transient fp16 copy of one layer's Q tier at a time
  (≈ `(Q-fp size)/num_layers`), freed immediately. The fp16 write-back is the real
  cost (~4× the int4 read); Phase 2 eliminates it.
- **Exit criterion:** Suite C confirms net memory savings; Suite A Jaccard holds;
  LongBench quality acceptable at int4. Only then move to Phase 2.

### Phase 2 — Triton GEMV tile: fused dequant-inside-attention (future work)

A Triton decode kernel that loads int4 codes tile-by-tile, dequantizes to fp16 **in
registers**, and runs `Q·Kᵀ` before any write-back. The fp16 materialization is
eliminated — only int4 codes are read from HBM.

- **Storage: pre-RoPE** (same as Phase 1). Tile kernel: `load int4 → unpack → scale →
  apply RoPE from cos/sin → MAC`. RoPE is arithmetic on already-loaded data — zero
  extra memory traffic in the bandwidth-bound decode regime. `cos/sin` for a window's
  **fixed** original absolute positions are computed once and reused — eviction never
  rebases them, and the codes never change — so the tile carries one pinned
  `(scale, zero)` and one pinned position range per window.
- **Scope:** decode path only (GEMV, one query token at a time). Prefill stays on the
  Phase 1 materialize path — fine, since the Q tier is a decode-phase construct.
- **Layout fit:** tile boundary = window boundary = scale-group boundary. One tile
  reads one window's codes and one pinned `(scale, zero)` — no cross-tile scale
  bookkeeping.

### Phase 3 — FlashInfer integration (production ceiling, not in scope)

Replace the custom GEMV tile with FlashInfer's paged quantized decode attention
(online softmax, GQA, paged blocks). **Note: FlashInfer is fp8/fp4-native — int4 is
*not* a native FlashInfer path**; the int4 decode kernels in the literature (Atom)
are custom builds on top of it. Phase 3 therefore either (a) re-targets the Q tier
to fp8/nvfp4 for the native path, or (b) ports an Atom-style int4 kernel — decided
when Phase 2 is profiled. Requires aligning `QuantizedStore`'s block layout with
FlashInfer's paged KV convention. Strictly better than Phase 2 but adds a
significant dependency and layout constraint. Deferred until Phase 2 is profiled.

---

## Environment caveat

The **target** across eval devices is **transformers 4.47.1**; `environment.yml` is
pinned to the 4.47.x line. transformers 5.x builds the causal mask via
`create_causal_mask` → `Cache.get_mask_sizes()`, which `WindowedCache` does not
implement, so a full-model forward crashes on 5.x. The current dev machine runs
transformers 5.8.1 / torch 2.12 / Python 3.12, so until it is brought to 4.47.x,
verify cache/quant logic via **CPU unit tests** (`pytest -m "not gpu"`), not full-
model runs. (`utils/cache_factory.py` refuses to run on > 4.47.1 rather than crash
mid-run.)

---

## Implementation outline

New `QuantizedStore` + hand-rolled KIVI-style quantizer module; two-tier
`update()` / eviction with the per-window ledger (§5, §6); a one-time demotion
un-rotate + `materialize_effective_kv` helper (RoPE at each Q window's fixed original
positions); `get_seq_length()` reporting `T_total`; tier-aware budget resolver (§7);
mirrored into both backends (§9).

CPU unit tests: round-trip error, degenerate-group exactness (`mx == mn` → `x̂ = mn`),
promote→demote reactivation (codes bit-identical, no recompute), position-invariance
(attention output unchanged by physical interleave order), merged-axis score alignment
(`window_scores` index ↔ chronological window id across a mixed-tier eviction),
flash/eager parity.

Gates: Suite C (peak memory + throughput/TPOT), Suite A (Jaccard drift vs fp-only),
LongBench (quality at int4).

---

## Appendix: the whole thing in plain English

**The problem.** When the model reads a long prompt and generates, it remembers every
token it has seen — that memory is the KV cache. It keeps growing and eventually eats
the whole GPU, so we have to throw stuff away. The whole game is throwing away the
right stuff and keeping what matters.

**Windows.** We chop the sequence into fixed-size chunks called windows (say 32
tokens each). Every `window_size` steps we pause, look at how much attention each
window has been pulling, and rank them. A couple of windows never get ranked — the
sink (first few tokens) and the local window (the most recent one) are always kept.

**Three buckets instead of two.** Normally a window is either kept or deleted. We add
a middle bucket. The best windows stay in full precision fp16 — that's the K tier.
The ones not good enough for fp16 but still too useful to throw away, we squeeze down
to int4, a quarter of the memory — that's the Q tier. Everything else is dropped.

**What happens at every eviction.**

1. We rank all the windows — both tiers together, because a squeezed-down Q window can
   turn out more important than a full-precision one and get promoted back up.
2. The windows crossing into the Q tier get quantized — but right before we quantize,
   we strip RoPE off them (RoPE is the position stamp on a token). We store the
   stripped, un-stamped version, and we remember the window's original position numbers.
   (A window that has been in the Q tier before skips all of this — we kept its old
   codes and its old positions and just switch them back on.)
3. The survivors get squished together in memory so there are no gaps — but we **do not
   renumber anything**. Every surviving key keeps the exact position it always had. A
   window that used to sit at positions 900–931 still says 900–931; it just moved to a
   lower shelf in memory. That is the whole difference from the old design: we compact
   storage, we never touch positions.
4. Because nobody is renumbered, there is no puzzle about where window 3 goes relative
   to windows 1 and 5 — they all still carry their true original positions, so the
   model sees the right distances no matter where each one physically sits. The one
   thing we still do jointly across both tiers is glue them back in chronological order
   when we build the tensor to attend over, because the scorer counts attention window
   by window and needs them in order.
5. The only thing that changes for a Q window at eviction is its **byte offset** — where
   its codes live in the packed store as the store compacts. Its codes, its scale, and
   its position numbers are all frozen for life.

**Why un-stamped (pre-RoPE).** Not for correctness anymore — since we never renumber, a
stamped copy would be just as frozen as an un-stamped one. We keep it un-stamped for two
reasons: the un-stamped keys quantize **more cleanly** (RoPE smears the per-channel
structure our scales lean on), and it lines up with the future fused kernel, which
stamps RoPE on the fly inside attention. Storing codes un-stamped and stamping fresh only
at read
means the codes are frozen forever and never drift.

**The forward pass.**

- The fp16 windows are already stamped and ready.
- For each Q window we look it up in the ledger, grab the int4 codes, blow them back
  up to fp16 (dequantize), and stamp them with RoPE using the window's original
  position numbers (its frozen `position_range`).
- We glue the fp16 and freshly-stamped Q windows into one tensor and hand it to normal
  attention. The glue order doesn't matter — each key already carries its correct
  position inside its own values, so attention gets the right distances no matter
  where a key physically sits.

In Phase 1 we do this the simple way — blow up the whole Q tier, glue, attend. In
Phase 2 we do the blow-up one window at a time *inside* the attention kernel, so the
fp16 version is never written to memory; only the small int4 version ever lives in
HBM.
