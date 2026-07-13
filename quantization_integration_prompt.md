/# Implementation Prompt — Integrate the StickyKV Quantization Design

You are implementing the two-tier windowed KV cache described in [design.md](design.md)
into the existing StickyKV codebase. This prompt translates that design into a
concrete, ordered work plan bound to the files that already exist. Read
[design.md](design.md) in full first — it is the spec. This document tells you
_where_ each piece lands and _which invariants must not break_.

Supporting rationale is in [design_rationale.md](design_rationale.md); the decision
record is in [design_history.md](design_history.md). Do not re-open decisions those
files mark as fixed.

---

## 0. Ground rules (violating any of these is a regression)

1. **`p=1` / `q=0` byte-identical.** With the quant tier disabled (`quant_ratio q = 0`,
   or the feature flag off) the cache must behave **bit-identically** to today.
   Every new code path is gated so the pure-fp16 path is untouched. Confirm with the
   existing `modules/evaluation/test_*_parity.py` suites.
2. **Backend mirroring.** `modules/windowed_cache/{cache.py,state.py}` and
   `modules/windowed_eager_cache/{cache.py,state.py}` are **byte-identical** twins
   (see the header banners in those files). Any edit to one is mirrored verbatim to
   the other. The **only** legitimate divergence stays in `hooks.py` (§9 of the
   design). The new quantizer/ledger/store is a **single shared module imported by
   both backends** — do not duplicate it per-backend.
3. **B=1 only in v1.** The single-tier cache is already B=1 (it relies on HF's
   monotonic absolute query position and evicts **per row**). Keep every new primitive
   per-row so B>1 is a later extension, not a rewrite. Do not add a B>1 code path.
   **Note:** eviction no longer re-rotates keys or overrides the query position —
   `state.rerotate_keys` and `utils/position_override.py` were removed (commit
   `ef9c84f`). Surviving keys keep their **original absolute** RoPE positions; the two
   tier design must preserve that (§5 of design.md), not reintroduce renumbering.
4. **CPU-testable / transformers pin.** The dev box runs transformers 5.8.1, but the
   target is 4.47.1 and `utils/cache_factory.py` refuses to run model-backed on
   > 4.47.1. So **all new logic must be exercised by `pytest -m "not gpu"` CPU unit
   > tests** — no full-model forward is required to prove correctness. Design them so
   > they run with a tiny fake rope module and hand-built tensors, as the current
   > `tests/test_windowed_cache.py` does.
5. **No re-quantization, ever (§3, §10).** Codes are written exactly once per window
   lifetime. Re-demotion is a _reactivation_ of the dormant ledger entry, never a
   recompute. There must be no arithmetic path that runs `quant(dequant(·))`.
6. **Pinned grid stored in fp16 (§2).** Quantize against the **fp16-rounded**
   `scale`/`zero`, not the fp32 intermediates, so the grid the codes were fit to is
   bit-identical to the grid used at dequant.

---

## 1. Current-code map (what you are extending)

| Concern                | File(s)                                          | Relevant today                                                                                                                                                                     |
| ---------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF Cache orchestration | `modules/windowed_cache/cache.py` (+ eager twin) | `update()` appends, accumulates scores, evicts (rank→retain window idx→token idx→`slice_and_keep`→gather scores/ids), returns live `state.key_states/value_states`. **No re-rotation, no query override.** |
| Tensor storage         | `modules/windowed_cache/state.py` (+ twin)       | `key_states/value_states/position_ids/window_scores/original_window_ids`; `append`, `slice_and_keep` (compacts + gathers **original absolute** `position_ids`; keys never re-rotated). `rerotate_keys` was **removed** (commit `ef9c84f`).             |
| Eviction index math    | `modules/windowed_cache/policy.py` (+ twin)      | `compute_retain_window_indices`, `expand_to_token_indices` (arithmetic `num_sink + w*window_size + offset`)                                                                        |
| Scoring                | `modules/windowed_cache/scorer.py` (+ twin)      | `compute_window_scores`, `reduce_token_scores_to_windows`, `accumulate` — pure cumulative (H2O) sums on the merged window axis                                                                  |
| Budget resolver        | `modules/windowed_cache/config.py` (+ twin)      | `WindowedCacheConfig` / `ResolvedConfig`; `resolve()` computes `bytes_per_token`, `total_budget_bytes`, `top_k_windows`                                                            |
| Flash score hook       | `modules/windowed_cache/hooks.py`                | reads **raw** `cache._states[lidx].key_states` for the aux SDPA                                                                                                                    |
| Eager score hook       | `modules/windowed_eager_cache/hooks.py`          | reads `attn_weights` from the module output — i.e. attends over `update()`'s return                                                                                                |
| Query position         | HF-native (no module)                            | eviction keeps original absolute positions; query uses HF's monotonic position. `position_override.py` and the `rope_module` ctor param were **removed** (commit `ef9c84f`).       |
| Backend/version gate   | `utils/cache_factory.py`                         | `get_cache_classes`, `assert_transformers_version_supported`                                                                                                                       |

---

## 2. New shared module to create

Create a **single shared package** `modules/quant/` (imported by both backends):

- `quantizer.py` — the hand-rolled KIVI-style affine int4 quantizer. Implement
  **exactly** the numerics in design §2:
  - `scale = (mx − mn)/15`, `zero = mn` (float offset), computed in fp32, then
    **stored/rounded to fp16** and re-loaded before the codes are fit.
  - `q = clamp(round_half_even((x − zero)/scale), 0, 15)`, clamp **before** the uint
    cast; `x̂ = q·scale + zero`.
  - Degenerate `mx == mn` → `scale = 1`, all codes 0, `x̂ = mn` exactly.
  - Key granularity: per-`(head, channel, window)` (channel-major store
    `[H_kv, D, T_q]`, pack 2 tokens/byte → `[H_kv, D, ceil(window/2)]`; scale/zero
    `[H_kv, D]` per window). Value granularity: per-token (token-major
    `[H_kv, T_q, D]`, pack 2 channels/byte; scale/zero `[H_kv, T_q]`).
  - **Bring-up path:** implement an _unpacked_ codes variant first (one 4-bit value
    per uint8 so `torch.gather` works token-wise), get all round-trip/parity tests
    green, then switch to the nibble-packed layout behind the same API. `window_size`
    even is required (assert it) so there is no tail padding.
- `store.py` — `QuantizedStore`: gap-free dense int4 code store + per-window fp16
  scales/zeros + `position_ids` (each window's **original absolute** positions), keys
  stored **pre-RoPE**. Supports append, compact (offset shift), and dequant-at-read.
- `ledger.py` — the per-window ledger keyed by `original_window_id` (design §6 table):
  `original_window_id` (immutable), `codes` (immutable), `scale`/`zero` (immutable,
  pinned), `position_range` (**immutable** — the window's original absolute positions,
  set once at demotion; the read path feeds it to RoPE), `offset` (**mutable** — byte
  offset, shifts as the Q store compacts). `offset` is the *only* per-eviction write;
  there is **no** per-eviction `position_range` update, because eviction never renumbers.
  Entries persist through promotion as **dormant** (codes+grid+`position_range`
  retained, `offset` invalid, excluded from reads/interleave); freed only on outright
  drop. Re-demotion = reactivate.

Everything in `modules/quant/` must be pure-tensor and CPU-testable with no
transformers dependency beyond the rope module passed in at read time.

---

## 3. Ordered workstreams

Do these in order; each ends green before the next starts. Keep the quant tier behind
a config flag / `q=0` default so `main` stays shippable throughout.

### WS-1 — Quantizer + store + ledger (pure, no cache wiring)

- Implement `modules/quant/{quantizer,store,ledger}.py` per §2 above.
- CPU tests: round-trip error bounds; degenerate-group exactness (`mx==mn` → `x̂=mn`);
  fp16-grid idempotence (`quant` against fp16 grid, dequant, re-quant against the same
  grid → **bit-identical codes**); pack/unpack round-trip equals the unpacked variant.
- No changes to `cache.py`/`state.py` yet.

### WS-2 — Tier-aware budget resolver (design §7)

- Extend `WindowedCacheConfig`/`ResolvedConfig.resolve()` in `config.py` (mirror to
  twin) with new knobs: `quant_ratio` (`q`, default **0.0** = feature off), bit-width
  (fix 4 in v1), group size (= `window_size`). Scale dtype is **fixed fp16 — not a
  knob** (§2, §7).
- **Fold the split into the existing token-granular `resolve()`** — do not layer a
  separate window-count formula (that would break `q=0` parity). Keep the current
  token math: `total_budget_tokens = ⌊β·(prefill_len+max_tokens)⌋`; **sink and local
  are carved as tokens, before any `//window_size`** (`num_sink_tokens` is a raw token
  count *outside* the window axis and always fp; `local_tokens` is a `window_size`
  multiple, always fp). Split **only the evictable budget** by `q`:
  ```
  M_evict  = (total_budget_tokens − num_sink_tokens − local_tokens) · bytes_per_token
  top_k_fp = ⌊ (1−q)·M_evict / b_fp ⌋      # b_fp = bytes_per_token·window_size
  N_q      = ⌊    q ·M_evict / b_q ⌋       # b_q per §7; use b_q NOT b_fp
  ```
- Add `top_k_fp` and `N_q` fields; **keep** `bytes_per_token/total_budget_bytes/
  total_budget_tokens/top_k_windows` unchanged. At `q=0`: `N_q=0` and
  `top_k_fp = top_k_windows` **by construction** (both floor `M_evict/b_fp`).
- Resolve float `local_window_size` first (fraction of `total_budget_tokens`, the whole
  retained budget — **not** of `M_fp`), so local sizing is independent of `q`.
- CPU tests: the §7 worked example (β=0.25, q=0.5 → N_q≈4·top_k_fp); `q=0` equals the
  current resolver output **field-for-field** (including a case where `num_sink_tokens`
  is not a `window_size` multiple, to lock the carve-before-floor order).

### WS-3 — Merged window axis + tier-aware index translation (design §5)

This is the structural core. **There is no interleaved position map and no
`rerotate_keys`** — positions are never renumbered (design §5). Add to
`state.py`/`policy.py` (mirror to twins) and `modules/quant`:

- **Q-window original positions.** When a window is demoted, capture its **original
  absolute** `position_ids` (the values `slice_and_keep` already carries) as the
  ledger's immutable `position_range`. These are frozen for the window's life; there is
  no per-eviction position update — only the byte `offset` shifts as the Q store
  compacts.
- **Demotion un-rotate** (one-time): to produce pre-RoPE codes from post-RoPE fp keys,
  apply RoPE with **negated sin** (`cos(−θ)=cos θ`, `sin(−θ)=−sin θ`) at the window's
  original absolute positions, using the model's rope module. This is a standalone op —
  do **not** reintroduce the removed `rerotate_keys`; the fp tier is never re-rotated.
- Make `expand_to_token_indices` **tier-aware**: expand only the **fp partition** of
  the retained merged indices to fp-store token indices (fp-store window rank =
  cumsum over the fp-tier mask → `num_sink + rank_fp·window_size + offset`); Q windows
  resolve through the ledger and get **no** token gather.
- **Policy bookkeeping — do not conflate two lengths** (design §5): ranking / local
  protection / the evictable split run on the **merged window count** `W = W_fp + W_q`
  (already read off `window_scores.shape[2]`, not `total_tokens`), while
  `expand_to_token_indices`'s fp gather and OOB trim use the **fp-store physical length
  `T_fp`** (`state.key_states.shape[2]`). Keep `EvictionPolicy.total_tokens` = merged
  `T_fp + T_q` (so `num_total_windows` / local counts / `get_seq_length()` align), and
  pass `T_fp` explicitly to the fp expansion. At `q=0`, `T_q=0` ⇒ both collapse to today.
- CPU tests: `slice_and_keep` leaves fp survivors at their **original absolute**
  `position_ids` (unchanged from `main`); demotion un-rotate (reading the window's
  `position_ids` slice **before** it is dropped) then read-time re-rotate round-trips
  the key; per-row divergent eviction; tier-aware `expand_to_token_indices` selects
  exactly the fp partition against `T_fp`.

### WS-4 — `materialize_effective_kv` + two-tier `update()`/eviction (design §5, §8, §9)

- Add `materialize_effective_kv(fp_store, q_store, ledger, rope)` (shared): dequantize
  the Q store, apply RoPE at each Q window's **immutable** `position_range` (its
  original absolute positions), interleave with the fp store **by `original_window_id`**
  into chronological order (a gather over `N_fp+N_q` windows, **not** a `[fp ‖ Q]`
  concat). Returns effective K/V. The `rope` module dependency is reintroduced **only**
  on the `q>0` path.
- Rewire the eviction cycle in `cache.py` (mirror to twin) to the steps in §5:
  rank on the merged axis → assign tiers (`N_fp`/`N_q` from WS-2) → move
  boundary-crossers (demote: un-rotate once at the window's original absolute positions,
  quantize against a freshly-pinned fp16 grid, record `position_range`, append to Q
  store, drop fp copy; **re-demote = reactivate dormant ledger entry, no re-quant, no
  un-rotate**; promote: dequantize + re-rotate at original positions, **splice into the
  fp store at the window's chronological `original_window_id` slot — NOT append at the
  end**, keep ledger entry **dormant**) → **reassemble** each store (fp = `[sink ‖ fp
  windows in chronological order]`: `slice_and_keep` the surviving fp tokens + splice any
  promoted window at its chronological slot, keeping original `position_ids`; Q via
  `offset` shift, any physical order) → `set_total_after_compaction` records **merged**
  `T_fp + T_q` (not `state.seq_length`, now fp-only). **No position map, no fp
  re-rotation, no ledger `position_range` update, no query override.**
- **fp-store ordering INVARIANT (do not violate):** the fp store's physical window order
  must equal chronological `original_window_id` order at all times, or the tier-aware
  `expand_to_token_indices` cumsum (`num_sink + rank_fp·window_size + offset`) desyncs on
  the next eviction. Add a CPU test: after a promotion, `expand_to_token_indices` returns
  the correct fp token ranges. When no promotion occurs the reassembly degenerates to
  today's single `slice_and_keep`; at `q=0` no store is ever split (first eviction is
  where a split is born — prefill is a single fp store).
- `update()` returns `materialize_effective_kv(...)` (a freshly-built interleaved
  tensor) instead of the live `state.key_states/value_states`. Callers must not assume
  the return aliases the stored fp cache — document this at the return site.
- `cache.get_seq_length()` must report the **effective** `T_total` (fp+Q), since HF
  consumes it to size the causal mask over the returned effective K/V (and the flash
  hook uses it as `S`). This is a **key-count** report only — there is no query-position
  override; query positioning stays HF-native (monotonic absolute).
- Gate the whole thing on `q>0`; `q=0` takes the legacy single-tier path unchanged
  (rope-free, no materialize, byte-identical to `main`).
- CPU tests: promote→demote reactivation (codes **bit-identical**, no recompute, no
  un-rotate call); position-invariance of attention under interleave order; merged-axis
  score alignment (`window_scores` index ↔ chronological window id across a
  **mixed-tier** eviction); `get_seq_length()` returns `T_fp + T_q`.

### WS-5 — Flash hook sources effective K (design §8, §9)

- The **one required flash-hook change**: `modules/windowed_cache/hooks.py` currently
  reads raw `cache._states[lidx].key_states` for the aux SDPA. With a live Q tier this
  misses the Q windows and the interleaving. Source the effective K via
  `materialize_effective_kv` instead so scoring covers both tiers on the merged axis.
- The eager hook already attends over `update()`'s return, so it needs **no** change
  beyond confirming that return is now the effective K/V. Do **not** add an aux SDPA
  to eager.
- CPU test: flash/eager parity of `window_scores` across a mixed-tier eviction.

### WS-6 — Config surface + gates

- Expose `quant_ratio` (and bit-width/group-size if you surface them) in the YAML
  config (`utils/config.py`) and the LongBench runner, matching how the existing
  `cache` knobs (`cache_budget`, `window_size`, `num_sink_tokens`) are threaded through.
  Default `q=0`.
- Wire telemetry: promotion-frequency counters (design §10) and dormant-entry counts.
- Do **not** change `MAX_SUPPORTED_TRANSFORMERS`; the mask-API gap on 5.x is separate.

---

## 4. Test matrix (all `pytest -m "not gpu"`, CPU)

Mirror the style of `tests/test_windowed_cache.py` and
`modules/evaluation/test_*_parity.py`. Required cases (design "CPU unit tests" list):

- Quantizer round-trip error within bound; degenerate-group exactness; fp16-grid
  idempotence; pack/unpack equals unpacked.
- Promote→demote **reactivation**: codes bit-identical, zero recompute calls
  (assert the quantizer is not re-invoked, e.g. via a call counter/spy).
- Position-invariance: attention output equal regardless of physical interleave order.
- Pre-RoPE round-trip: demotion un-rotate then read-time re-rotate at the window's
  original absolute positions recovers the key; fp survivors keep original positions.
- Merged-axis score alignment across a mixed-tier eviction; `get_seq_length()` = `T_total`.
- Flash/eager parity on the merged axis.
- **Regression parity:** with `q=0`, `resolve()` output and full `update()`/eviction
  behaviour are byte-identical to `main` (run the existing parity suites).

---

## 5. Deliverable shape

- New `modules/quant/` package (shared, single copy).
- Edits mirrored byte-for-byte across `windowed_cache` and `windowed_eager_cache`
  for `cache.py`/`state.py`/`policy.py`/`config.py`/`scorer.py`; **only `hooks.py`
  diverges** (flash gets the `materialize_effective_kv` source change; eager unchanged).
- **No** query-position override (`utils/position_override.py` was removed): the mask
  covers both tiers via `get_seq_length()` returning `T_total`; query positioning stays
  HF-native. Do not reintroduce a position-override hook.
- All new behaviour gated so `q=0` is the untouched legacy path (rope-free, no
  materialize).
- Phase 1 only (materialize-then-interleave, §11). **Do not** build the Triton
  (Phase 2) or FlashInfer (Phase 3) kernels — but keep the store layout
  (tile=window=scale-group, pre-RoPE codes) so Phase 2 is a drop-in later.

Work WS-by-WS, keeping the suite green and `q=0` byte-identical at every step.
Report which design section each commit implements.
