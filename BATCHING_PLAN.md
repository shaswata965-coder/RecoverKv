# Batching the two-tier cache (`quant_unpressed`)

Plan to lift `quant_ratio > 0` from B=1 to B>1, and why batching is the regime
where KV compression actually pays.

Status of every claim below: **verified against the tree at `3fb5a45`**, not
inherited from notes.

---

## 1. Where we actually are

| | state |
|---|---|
| `position_ids` | `[B, T]`, per-row — batch-ready |
| `original_window_ids` | `[B, W]`, per-row gather at eviction — batch-ready |
| `state.slice_and_keep` | per-row gather — batch-ready |
| `q = 0`, equal-length B>1 | **works** (verified byte-identical B∈{2,3,4,8}) |
| `q > 0`, any B>1 | **raises** `NotImplementedError` (`cache.py:179`) |
| ragged / left-pad + keep-mask | **not implemented** — blocks LongBench at *any* q |

So batching is half-built: the fp tier is per-row already; the Q tier has no
batch axis at all.

### The loop guard has a hole

`tests/test_windowed_cache.py::test_no_python_loops_in_hot_path` rejects `for`
loops whose target is named `b/h/t/w/n/batch/head/token/window` in
`cache/state/policy/scorer`. It matches on **variable name**, not semantics — so

```python
for wid in new_fp:            # passes the guard; is exactly a per-window loop
```

slips through. Every such loop in `_evict_two_tier` is a B>1 blocker. Widen the
guard to flag any `ast.For` in `_evict_two_tier` / `_materialize` regardless of
target name, or the batching work will regress silently.

---

## 2. What actually blocks `q > 0` at B>1

1. **`cache.py:179`** — the explicit guard.
2. **`policy.compute_two_tier_retain`** — `window_scores.mean(dim=1)[0]`, row 0
   only. Mechanical to vectorize (`argsort(dim=-1)`); no design question here.
3. **`_evict_two_tier`** — every access is row 0 (`ri`, `owids`, `pos_row`,
   `key_states[0]`), and the tier bookkeeping is **host-side Python sets**
   (`new_fp`, `new_q`, `cur_is_q`, `store.has_entry`).
4. **`QuantLedger` / `QuantizedStore`** — `Dict[int, LedgerEntry]`, one entry per
   window, **no batch axis**. This is the real blocker: rows evict divergently,
   so row 0 may hold windows `{1,5}` in int4 while row 1 holds `{4,11}`.
5. **`_window_spans`** — single-row `tok_wid`.

(3) and (4) are one problem wearing two hats: **the Q tier's identity
bookkeeping lives on the host, so it cannot carry a batch axis.**

---

## 3. The realization that makes this tractable

**Divergent eviction stays rectangular.**

`compute_two_tier_retain` keeps exactly `k_fp = min(top_k_fp, evictable_w)` fp
windows and `n_q = min(N_q, evictable_w - k_fp)` Q windows. Both derive from
config + `W_total`, which is **shared across rows**. So for equal-length prompts
every row retains the *same count* of fp and Q windows — only *which* windows
differ.

⇒ `T_fp` and `T_q` are equal across rows. The effective K/V stays a dense
`[B, H_kv, T_total, D]`. **No padding, no keep-mask, no raggedness** — as long as
prompts are equal-length.

That is why this splits cleanly into a cheap phase and an expensive one, and why
the cheap phase is worth doing first.

---

## 4. Plan

### Phase 1 — dense slot-table Q store (the enabling refactor)

Replace the host-side dict ledger with tensors carrying a batch axis:

```
key_codes   [B, N_slots, H_kv, D, ws//2]  uint8
key_scale   [B, N_slots, H_kv, D]         fp16
key_zero    [B, N_slots, H_kv, D]         fp16
val_codes   [B, N_slots, H_kv, ws, D//2]  uint8
val_scale   [B, N_slots, H_kv, ws]        fp16
val_zero    [B, N_slots, H_kv, ws]        fp16
slot_wid    [B, N_slots]  int64   # -1 = free
slot_active [B, N_slots]  bool    # active vs dormant (§10)
slot_pos    [B, N_slots, ws] int64
```

`N_slots` is **bounded**: `retain_only` drops every non-retained window, so live
entries ≤ `top_k_fp + N_q + local_windows`. Size it from `ResolvedConfig`; no
growth policy needed.

This is the shape the `quantization` branch already proved out (`active_view()`
returns tensors, `gather_keys(slots)` is one `index_select`). Do it at B=1 first
and hold byte-identity — it is a pure refactor.

**It pays for itself immediately, before any batching:**
- kills the residual eviction host syncs (§2 of the perf work; 5/layer/eviction → ~0);
- `promote_many`/`demote_many` become masked scatter/gather, dropping the
  eviction step's remaining 765 dispatches;
- removes every `for wid in ...` loop, so the widened guard can be enforced.

### Phase 2 — vectorize the tier decisions over B

- `compute_two_tier_retain`: `mean(dim=1)` → `[B, W]`; `argsort(dim=-1)`;
  `fp_sel = order[:, :k_fp]`, `q_sel = order[:, k_fp:k_fp+n_q]`. Loop-free.
- Tier transitions become set algebra on `[B, W]` bool masks:
  `demote = is_q_new & ~is_q_cur`, `promote = ~is_q_new & is_q_cur`,
  `drop = ~retained`. No Python sets.
- `_window_spans` → per-row. `tok_wid` is non-decreasing **per row**, so the run
  boundaries generalize: compute `is_start` on `[B, T_fp]` and use a batched
  `argsort` rather than a host map.
- The interleave **already generalizes**: `torch.argsort(merged_wids, dim=-1,
  stable=True)` over `[B, T]` needs only the dim change. This is the one piece
  the on-device rewrite bought us for free.
- Drop the `cache.py:179` guard. Keep `B=1` byte-identity as a hard test.

Ships equal-length B>1 at `q>0` ⇒ **perf + parity suites batch at int4**.

### Phase 3 — ragged / left-pad (shared with the deferred q=0 work)

Only LongBench needs this, and it is already blocked at `q = 0`, so it is not a
quant problem — do it once, for both tiers:

- left-pad `[B, H, T, D]` + cache-managed `key_padding_mask [B, T_keep]`;
- per-row `pad_offset` in scorer/policy so windows align to each row's real start;
- **rectangularity choice**: resolve `top_k` against the batch's *shortest* valid
  length (uniform retained count, no filler) — low-risk, and §3 above shows the
  tier split then stays rectangular for free. Filler+mask is the general fallback;
- replace `expand_to_token_indices`' `min_valid` cross-row truncation — it drops
  real high-score tokens from longer rows (a live correctness bug for ragged
  batches, independent of quant);
- flash hook aux-SDPA must apply causal **and** per-row padding mask;
- `longbench_runner` is hardcoded B=1: needs left-pad, per-row EOS/stop, and
  mask↔`generate` reconciliation after eviction shrinks the cache.
- Add length-bucketing to cut padding waste.

**Gate:** Phase 3 needs the live 4.47.1 env. Do not ship it blind.

---

## 5. Why batching is where compression pays

This explains the result that started this whole investigation: **at B=1, KV
compression cannot show a speedup, and that is not a bug.**

Decode reads **every weight, once, per step** — 16.06 GB for Llama-3.1-8B fp16.
On an A100 (1555 GB/s) that is a **10.3 ms/token floor** at B=1, and it is
completely independent of KV-cache size. Shrinking the cache 5× moves nothing,
because you were never waiting on the cache. Arithmetic intensity at B=1 is ~2
FLOP/byte against the A100's ~200 FLOP/byte ridge — the GPU is idle by design.

Batching is the only lever that changes this: the same 16.06 GB read serves B
tokens.

| B | ms/step | ms/token effective | throughput |
|---|---|---|---|
| 1 | 10.3 | 10.30 | 1× |
| 8 | 10.3 | 1.29 | 8× |
| 32 | 10.3 | 0.32 | 32× |
| 128 | 10.3 | 0.08 | 128× |

**And B is capped by KV memory — which is exactly what we compress.** Measured
steady state at qasper (4900 ctx, budget 0.20, q=0.5): `T_fp=565`, `T_q=1136`.

| | per-row KV | B on 80GB A100* |
|---|---|---|
| full cache (4900 tok fp16) | 612 MB | 93 |
| StickyKV (565 fp + 1136 int4) | 125 MB | **458** |

<sub>*16.1 GB weights + ~8 GB activations ⇒ ~56 GB for KV.</sub>

**~4.9× the batch at equal VRAM ⇒ ~4.9× the decode throughput.** That is the
thesis. It is invisible at B=1 and only measurable batched — so the perf suite
should report **tokens/s at max-B-that-fits**, not B=1 latency, or it will keep
showing our method winning by ~0%.

### Preallocation: negligible at B=1, ~1.9× at max-B

`state.append` does `torch.cat([key_states, key], dim=2)` every decode step, per
layer — reallocating and copying the **whole fp store** each time (`state.py:92`).
The fp tier is 57% of per-row KV bytes, so the copy is not small.

| B | weights | KV read | cat copy | total/step | prealloc speedup |
|---|---|---|---|---|---|
| 1 | 16.1 G | 0.1 G | 0.1 G | 16.3 G | 1.01× |
| 32 | 16.1 G | 4.2 G | 4.7 G | 25.0 G | 1.23× |
| 128 | 16.1 G | 16.8 G | 19.0 G | 51.8 G | 1.58× |
| 458 | 16.1 G | 60.0 G | **67.8 G** | 143.9 G | **1.89×** |

At B=1 it is 19 GB/sample ≈ 12 ms ≈ **0.2%** of runtime (plus 12,288 dispatches,
2.4%) — real but invisible. At max-B the cat is **4× the weight traffic** and
becomes the single largest term. It also doubles peak VRAM at the moment of copy,
which directly caps B.

HF's `DynamicCache` cats identically, so the full-cache baseline pays it too —
this is not a StickyKV regression. `StaticCache`-style preallocation to
`prefill + max_new_tokens` is the standard fix, and it is a **prerequisite for
max-B**, not a micro-optimization. Do it with Phase 1.

The Q slot table (§4) is the same idea for the other tier: bounded size, allocate
once, no per-eviction churn.

### The grid overhead is NOT the lever (correcting an earlier claim)

An earlier draft claimed the int4 key grid "costs ~30% of achievable B". **That
was wrong.** `config.py:293` computes

```python
b_q = (num_kv_heads*head_dim*ws          # codes
       + 4*num_kv_heads*head_dim         # key scale+zero fp16
       + 4*num_kv_heads*ws)              # value scale+zero fp16
```

— the budget **already includes the grid**, and it is enforced in **bytes**. So
per-row memory is pinned by `cache_budget` alone. The grid does not cost batch
capacity; it costs **retained tokens**, i.e. accuracy.

The observation itself holds — at `ws=8` the key grid is 512 B/token against
512 B of codes, so the key tier is effectively int8 (2.0× compression, not 4×;
values get 3.8×). It is 1:1 because the quant group is only 8 tokens. But every
way of fixing it is roughly a wash, measured on outlier-channel key-like data:

| option | b_q/win | retained tokens | int4 rel-err |
|---|---|---|---|
| now (ws=8, fp16 grids) | 12544 | 1701 | 1.00× |
| fp8 **scale** only | 11456 | 1805 (+6%) | 1.10× |
| ws=16 | 20992 | 1925 (+13%) | 1.30× |
| ws=32 | 37888 | 2053 (+21%) | 1.57× |
| ws=64 | 71680 | 2053 (+21%) | 1.79× |

Token gain **saturates at +21%** (once the grid is amortized the codes dominate)
while error grows without bound. Raising `window_size` also coarsens eviction
granularity — a second accuracy cost not in that column.

Also ruled out concretely:
- **Full fp8 grid → NaN.** `e4m3fn` maxes at 448; the zero point is a raw value
  magnitude and overflows once channel spread ≥ ~100. `e5m2` survives but doubles
  the error (2 mantissa bits). Only the *scale* is safe in fp8 (it is a small
  ratio) — and scale-only buys just +6% tokens for +10% error.
- **Tier-wide key grid.** This is the `ws → ∞` limit of the table above: the group
  would span ~1136 tokens instead of 8, on exactly the same error curve, and key
  channels have persistent outliers that would dominate the range for every other
  token. It also breaks §10 — the grid is frozen at first demotion, but windows
  demote at different times, so a tier-wide grid must either re-fit (forbidden;
  flips boundary codes) or clip late arrivals. **Do not pursue.**

**Conclusion: leave the grid alone.** The lever for batch capacity is
`cache_budget` (which *is* per-row memory, 1:1); the lever for decode throughput
is preallocation + the dense slot store.

---

## 6. Sequencing

1. Widen the AST loop guard (cheap; stops regression).
2. **Phase 1** dense slot store + fp-cache preallocation at B=1, byte-identical.
   *Also finishes the perf work and lifts the VRAM cap on B.*
3. **Phase 2** vectorize over B ⇒ equal-length B>1 at `q>0`.
4. Re-point the perf suite at tokens/s @ max-B (B=1 latency cannot show this
   method working — see above).
5. **Phase 3** ragged, on 4.47.1, shared with the q=0 gap.

Steps 2 and 3 are CPU-verifiable against the existing byte-identity harness.
Step 5 is the only one that needs the live model.

The grid work is **not** in this list, by the measurement above.
