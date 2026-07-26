# Eviction Logic — End-to-End Token Flow

This document traces the exact path a token takes from the moment it arrives during
prefill through every function that touches it, ending with how the KV cache is
compacted and keys are repaired after an eviction event.

---

## 1. High-Level Picture

```
Tokens (prefill or generation step)
        │
        ▼
  Attention forward pass
        │
        ▼
  Score hooks capture attention weights
  (modules/windowed_cache/hooks.py  OR  modules/windowed_eager_cache/hooks.py)
        │
        ▼
  compute_window_scores()
  (modules/windowed_cache/scorer.py:17)
        │
        ▼
  WindowedCache.update()
  (modules/windowed_cache/cache.py:99)
   ├── CacheState.append()            — grow the KV store
   ├── accumulate()                   — add new scores to running total
   ├── EvictionPolicy.should_evict()  — decide whether to compact now
   │       (first at decode step 0, then every window_size steps)
   └── [if evicting]
       ├── compute_retain_window_indices()  — top-K window selection
       ├── expand_to_token_indices()        — windows → absolute token positions
       ├── CacheState.slice_and_keep()      — compact K/V tensors
       ├── CacheState.rerotate_keys()       — repair RoPE encoding
       └── gather window_scores             — keep scores aligned with survivors
```

---

## 2. Configuration and Budget Resolution

Before any token arrives, `WindowedCacheConfig.resolve()` converts the high-level
YAML parameters into concrete integer counts.

**File:** `modules/windowed_cache/config.py:146`  
**Class:** `WindowedCacheConfig`  
**Method:** `resolve(prefill_len, model_config, kv_dtype) → ResolvedConfig`

```
cache_budget  (float, e.g. 0.5)
         │
         ▼
bytes_per_token = num_kv_heads × head_dim × element_size × 2   (K + V)
total_budget_bytes = cache_budget × prefill_len × bytes_per_token
total_budget_tokens = total_budget_bytes // bytes_per_token      (floor)
         │
         ▼
local_tokens  = resolved from local_window_size (int or float ratio)
top_k_windows = (total_budget_tokens - num_sink_tokens - local_tokens) // window_size
         │
         ▼
ResolvedConfig (frozen dataclass, lines 22-37):
  .window_size       int
  .num_sink_tokens   int
  .local_tokens      int
  .top_k_windows     int
```

`ResolvedConfig` is handed to both `EvictionPolicy` and `CacheState` at construction
time inside `WindowedCache.__init__()` (`cache.py:47`).

---

## 3. Step 0 — Prefill

All prompt tokens arrive in one batch.

### 3a. Hook installation (before `model.generate()`)

**Flash-attention backend:**  
`modules/windowed_cache/hooks.py:install_score_hooks()` (lines 99–224)  
- Registers a `forward_hook` (with kwargs) on each `LlamaAttention` /
  `Qwen2Attention` module. Flash-attention-2 never materializes the attention
  matrix, so the hook reconstructs it:
  1. Recompute the post-RoPE query from the layer inputs (`hidden_states`
     + `position_embeddings`); read the post-RoPE keys from the cache.
  2. Run an auxiliary scaled-dot-product attention on `(q, k_current)` → `[B, H_q, T, S]`,
     causally masked so prefill query rows do not attend to future keys.
  3. Call `compute_window_scores()` → `[B, H_q, W]`.
  4. Store the result in `cache.cache_kwargs[layer_idx]["window_scores"]`.

**Eager backend:**  
`modules/windowed_eager_cache/hooks.py:install_score_hooks()` (lines 78–178)  
- Registers a plain `register_forward_hook` on each attention module.
- Reads `attn_weights` directly from the output tuple (requires
  `output_attentions=True` to be passed to `model.generate()`).
- Calls `compute_window_scores()` and stores into `cache.cache_kwargs`.

### 3b. Score computation inside the hook

**File:** `modules/windowed_cache/scorer.py`  
**Function:** `compute_window_scores(attn, num_sink, window_size)` (lines 17–61)

Input `attn` shape: `[B, H_q, T, S]` (post-softmax attention matrix for the
current forward pass).

```python
# Step 1 — collapse query dimension: every query row contributes to the
#           "importance" of the key it attends to (H2O cumulative scoring)
token_scores = attn.sum(dim=-2)          # [B, H_q, S]  (sum over T queries)

# Step 2 — strip sink tokens (they are never evicted, never scored)
post_sink = token_scores[..., num_sink:]  # [B, H_q, S_post]

# Step 3 — right-pad so S_post is divisible by window_size
if S_post % window_size != 0:
    pad_size = window_size - (S_post % window_size)
    post_sink = F.pad(post_sink, (0, pad_size), value=0.0)

# Step 4 — reduce windows via einops: sum tokens within each window
window_scores = einops.reduce(post_sink, "b h (w s) -> b h w", "sum", s=window_size)
# shape: [B, H_q, W]   — sink is NOT represented here
```

### 3c. `WindowedCache.update()` — prefill path

**File:** `modules/windowed_cache/cache.py`

Called once per layer (each attention block calls `past_key_values.update()`).
The score hook runs *after* the attention forward, so during the prefill
`update()` no scores are available yet:

```python
# 1. Append prefill K/V
state.append(key_states, value_states, pos)        # state.py
policy.extend_total_after_append(n_new)            # policy.py

# is_prefill=True (first call to this layer)
policy.initialize_after_prefill(state.seq_length)  # policy.py
self._prefill_done[layer_idx] = True

# 2. Pull pre-computed window scores left by the hook.
#    cache_kwargs is still empty here — the prefill hook has not run yet —
#    so new_window_scores is None and state.window_scores stays None.
new_window_scores = merged_kwargs.get("window_scores")

# 3. No eviction during prefill: should_evict requires `not is_prefill`.
```

**Prefill does not evict — and cannot.** The prefill hook computes the prefill
window scores *after* this `update()` returns; they are consumed — and
`state.window_scores` is initialized — on the first generation step's
`update()`. Decode step 0 is therefore the *earliest* point at which any
eviction is possible, and at the default `first_eviction_step = 0` it is where
the first one fires: the cache grows to `prefill_len`, then compacts to budget
before the step-0 query attends. Every generated token from the first decode
step onward comes from the budgeted cache.

The one token this does not cover is the token the prefill forward itself
produces, which is read off the full prompt. That is true of the
prompt-compression baselines too (SnapKV / AdaKV / CriticalKV / DefensiveKV
prune in a hook that also fires after prefill attention), so the comparison is
like for like.

---

## 4. Step N — Generation (one token per step)

Each new token runs a forward pass. The hook fires again and produces a
`window_scores` of shape `[B, H_q, W]` covering every window currently in the
cache — H2O-style, every query row contributes.

### 4a. `WindowedCache.update()` — generation path

```python
# 1. Append new token's K/V
state.append(key_states, value_states, pos)   # grows seq dim by 1

# 2. Pull new window scores from hook
new_window_scores = merged_kwargs.get("window_scores")

# 3. Accumulate into running total
if W_new > W_old:
    # pad existing scores and extend original_window_ids
    state.window_scores = cat([state.window_scores, pad], dim=-1)
accumulate(state.window_scores, new_window_scores)   # scorer.py:64  (in-place +=)

# 4. Check trigger
step = self._generation_step[layer_idx]   # starts at 0, increments each gen step
should_evict = policy.should_evict(step)  # policy.py:62
#   → the FIRST eviction fires at a fixed step (first_eviction_step, default 8),
#     INDEPENDENT of window_size; nothing evicts before it. After that the natural
#     window cadence resumes: True iff step % window_size == 0. So ws=12 → 8, 12,
#     24, 36…; ws=8 → uniform 8, 16, 24…. (Edge: ws<8 suppresses sub-8 boundaries,
#     then resumes at the next multiple — ws=3 → 8, 9, 12…)
```

**Key insight:** scores accumulate monotonically. Every query in every forward pass
votes for every key it attended to. There is no sliding observation window — the
full history of attention is summed.

---

## 5. The Eviction Block (runs every `window_size` generation steps)

All code lives in `cache.py:197-234`.

### 5a. Two-step retain: window indices → token indices

**Step 1 — Window-level selection**  
`modules/windowed_cache/policy.py:87`  
`EvictionPolicy.compute_retain_window_indices(window_scores) → [B, W_retained]`

```
Cache region layout (in window space):
  [0 … evictable_w-1]  |  [evictable_w … W_total-1]
      evictable               local (always kept)

mean_scores = window_scores.mean(dim=1)         # [B, W_total]  mean over heads
evictable_scores = mean_scores[:, :evictable_w] # [B, evictable_w]

_, topk_idx = torch.topk(evictable_scores, k, dim=-1)  # k = top_k_windows
topk_sorted, _ = torch.sort(topk_idx, dim=-1)          # chronological order!

local_idx = arange(W_total - local_w, W_total)
retained  = cat([topk_sorted, local_idx], dim=-1)       # [B, k + local_w]
```

Sorting by chronological position (not by score) ensures the surviving windows
stay in natural sequence order, which matters for correct RoPE after eviction.

**Step 2 — Token-level expansion**  
`modules/windowed_cache/policy.py:158`  
`EvictionPolicy.expand_to_token_indices(retained_window_idx) → [B, T_retained]`

```
sink_idx   = arange(0, num_sink_tokens)                    # always prepended
offsets    = arange(0, window_size)                        # [window_size]

token_idx  = num_sink_tokens
           + retained_window_idx.unsqueeze(-1) * window_size
           + offsets                                       # [B, W_retained, window_size]
token_idx  = token_idx.reshape(B, -1)                     # [B, W_retained * window_size]

all_idx    = cat([sink_idx, token_idx], dim=-1)            # [B, total]

# Mask out OOB indices (partial last window)
valid_mask = all_idx < self.total_tokens
order      = argsort(~valid_mask, stable=True)             # valid-first order
all_idx    = gather(all_idx, 1, order)[:, :min_valid]
```

### 5b. Snapshot old positions

```python
# cache.py:213
old_positions = state.position_ids[retain_token_idx[0]].clone()
# These are the *pre-compaction* absolute positions of retained tokens.
# They are needed so RoPE can be un-applied with the correct angles.
```

### 5c. Compact K/V — `CacheState.slice_and_keep()`

**File:** `modules/windowed_cache/state.py:103`

```python
def slice_and_keep(self, retain_token_indices):
    # Gather K along seq dim using token indices
    idx = retain_token_indices.unsqueeze(1).unsqueeze(-1)  # [B, 1, T_ret, 1]
    idx = idx.expand(-1, H_kv, -1, D)
    self.key_states   = torch.gather(self.key_states,   dim=2, index=idx)
    self.value_states = torch.gather(self.value_states, dim=2, index=idx)

    # Gather position_ids to the survivors' ORIGINAL absolute positions,
    # per row (rows may evict different windows).
    self.position_ids = torch.gather(self.position_ids, 1, retain_token_indices)
```

After this call `key_states` and `value_states` have shape
`[B, H_kv, T_retained, D]` and `position_ids` holds each survivor's **original**
absolute position — sparse, with gaps where windows were dropped.

> **This is the default and it is deliberate.** Surviving keys keep the RoPE
> rotation they were stored with, so their positions must stay original for the
> query↔key relative phase to be right. Rebasing to `arange(T_retained)` is only
> valid if the keys are *also* re-rotated **and** the query's position is rebased
> to the compacted length every step — which HuggingFace `generate` does not do
> (it advances `cache_position` monotonically on transformers ≤ 4.47). Keeping
> original positions matches KVPress / H2O and is correct on any version.
> `rerotate_keys` below is the opt-in `rerotate_on_evict: true` path, off by
> default; it is the only thing that rebases `position_ids`.

### 5d. Repair RoPE — `CacheState.rerotate_keys()` (opt-in, `rerotate_on_evict: true`)

**File:** `modules/windowed_cache/state.py:137`

Compaction creates a gap: the key at original position 47 is now at compacted
position 12. Without correction the model's attention scores will be computed
with wrong position-pair distances. `rerotate_keys` fixes this in two passes:

```python
def rerotate_keys(self, rope_module, old_positions):
    # 1. Un-apply old RoPE (original absolute positions)
    cos_old, sin_old = rope_module(self.key_states, old_positions)
    keys_raw = apply_rotary_pos_emb(self.key_states, cos_old, -sin_old)

    # 2. Re-apply new RoPE (contiguous compacted positions)
    new_positions = self.position_ids          # [0, 1, ..., T_retained-1]
    cos_new, sin_new = rope_module(keys_raw, new_positions)
    self.key_states = apply_rotary_pos_emb(keys_raw, cos_new, sin_new)
```

The negated sin (`-sin_old`) is the mathematical inverse of RoPE rotation.
`rope_module` carries the model's NTK/YaRN scaling factors, so extended-context
models are handled correctly.

### 5e. Align window scores and original_window_ids

```python
# cache.py:222-231
idx_w = retained_window_idx.unsqueeze(1).expand(B, H_q, -1)
state.window_scores = torch.gather(state.window_scores, dim=-1, index=idx_w)
# Now state.window_scores is [B, H_q, W_retained]

# Keep the mapping from compact-index to original-sequence-window alive
state.original_window_ids = state.original_window_ids[retained_window_idx[0]]
```

`original_window_ids` is what lets the faithfulness runner compare the ours run
against the base run at the same absolute window positions.

### 5f. Policy state update

```python
# cache.py:234
policy.set_total_after_compaction(state.seq_length)
# Updates policy.total_tokens to match the compacted cache length.
```

---

## 6. CacheState Internal Structure

**File:** `modules/windowed_cache/state.py`

| Attribute | Shape | Meaning |
|---|---|---|
| `key_states` | `[B, H_kv, T, D]` | Current K tensors |
| `value_states` | `[B, H_kv, T, D]` | Current V tensors |
| `position_ids` | `[B, T]` | Survivors' ORIGINAL absolute positions, gathered per row at each eviction (only rebased to `arange(T)` on the opt-in `rerotate_on_evict` path) |
| `window_scores` | `[B, H_q, W]` | Cumulative sum of attention received per window |
| `original_window_ids` | `[B, W]` | Maps compact window index → original sequence window id, per row |

### `CacheState.append()` (state.py:62)

```python
def append(self, key, value, pos=None):
    self.key_states   = cat([self.key_states,   key],   dim=2)
    self.value_states = cat([self.value_states, value], dim=2)
    if pos is not None:
        self.position_ids = cat([self.position_ids, pos])
    else:
        # Auto-increment: next positions follow sequentially
        T_new = key.shape[2]
        next_start = self.position_ids[-1] + 1 if len(self.position_ids) > 0 else 0
        self.position_ids = cat([self.position_ids, arange(next_start, next_start + T_new)])
```

---

## 7. Eviction Trigger Summary

The first eviction fires at a fixed `first_eviction_step`, independent of
`window_size`; the natural cadence resumes afterward.

| Condition | `should_evict` result |
|---|---|
| Prefill step (first call to `update()`) | False (`not is_prefill` required) |
| Generation step N where `N < first_eviction_step` | False (nothing evicts before it) |
| Generation step N where `N == first_eviction_step` | True (the forced first eviction) |
| Generation step N > `first_eviction_step`, `N % window_size == 0` | True |
| Generation step N > `first_eviction_step`, `N % window_size != 0` | False |

**At the default `first_eviction_step = 0`** the two rules coincide (`0 % ws ==
0` for every `ws`) and the schedule collapses to a uniform `0, ws, 2·ws, …` with
no short first window. The prompt is compacted on the first decode step and the
cache stays at budget for the rest of the run, with the hook cost amortized
across `window_size` steps as before.

**A positive value is an ablation, not a tuning knob.** With
`first_eviction_step = N` the prompt stays whole through decode steps `0 … N-1`,
so any answer that terminates inside that window is measured at FULL cache no
matter what `cache_budget` says. At the old default of 8 that covered, on
LongBench, every `max_gen_length = 32` dataset (hotpotqa, 2wikimqa, musique,
triviaqa, passage_count, passage_retrieval_en) plus trec and narrativeqa — their
scores came out budget-invariant because all four runs were the same computation
up to the scored token. Label any such run as an ablation; never put it in a
comparison row.

---

## 8. Backend Differences

| | Flash-attention backend | Eager backend |
|---|---|---|
| **Module** | `modules/windowed_cache/` | `modules/windowed_eager_cache/` |
| **Hook type** | `forward_hook` (with kwargs) | `forward_hook` |
| **Attention source** | Auxiliary causal SDPA (recomputed query + cached keys) | Materialized `attn_weights` from output tuple |
| **Requires `output_attentions`** | No | Yes |
| **attn_implementation** | `"flash_attention_2"` | `"eager"` |

Both backends call the same `compute_window_scores()` function from `scorer.py`
and store results in the same `cache.cache_kwargs[layer_idx]["window_scores"]` slot.
The `WindowedCache.update()` method is identical for both backends.

---

## 9. Full Function Call Sequence (one generation step, one layer)

```
model.generate() — one step
  │
  └─► LlamaAttention.forward(q, k, v, ...)
        │
        ├─► attention computation (flash or eager)
        │
        └─► [FORWARD HOOK]
              ├─► flash: recompute post-RoPE query, run auxiliary causal SDPA
              │   eager: read materialized attn_weights from the output tuple
              ├─► compute_window_scores(attn, num_sink, window_size)
              │     scorer.py:17
              └─► cache.cache_kwargs[layer_idx]["window_scores"] = scores

  └─► WindowedCache.update(k, v, layer_idx, cache_kwargs)
        cache.py:99
        │
        ├─► CacheState.append(k, v, pos)
        │     state.py:62
        │
        ├─► EvictionPolicy.extend_total_after_append(1)
        │     policy.py:45
        │
        ├─► accumulate(state.window_scores, new_window_scores)
        │     scorer.py:64
        │
        ├─► EvictionPolicy.should_evict(step)
        │     policy.py:62
        │
        └─► [if True]
              ├─► EvictionPolicy.compute_retain_window_indices(window_scores)
              │     policy.py:87
              │
              ├─► EvictionPolicy.expand_to_token_indices(retained_window_idx)
              │     policy.py:158
              │
              ├─► old_positions = state.position_ids[retain_token_idx[0]].clone()
              │     cache.py:213
              │
              ├─► CacheState.slice_and_keep(retain_token_idx)
              │     state.py:103
              │
              ├─► CacheState.rerotate_keys(rope_module, old_positions)
              │     state.py:137
              │
              ├─► gather state.window_scores by retained_window_idx
              │     cache.py:222-225
              │
              ├─► update state.original_window_ids
              │     cache.py:228-231
              │
              └─► EvictionPolicy.set_total_after_compaction(state.seq_length)
                    policy.py:54
```
