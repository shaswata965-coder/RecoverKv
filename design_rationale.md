# StickyKV Quantization — Rationale & Open Items

Supporting analysis and trade-off studies behind the **fixed** choices in
[design.md](design.md). Nothing here is ratified spec — it is the "why", the cost
comparisons, the options still under evaluation, the integration audit, and the items
left to decide. Once an item is settled it graduates to design.md; once an alternative
is rejected outright it moves to [design_history.md](design_history.md).

Target precision is **int4** (the int8 milestone was dropped).

---

## Outlier strategy (int4)

**Ratified for v1 (2026-07-06): no outlier machinery.** NF4 keys and the value
Hadamard fold are deferred beyond v1 altogether — see the v1 decision at the end of
this section. The analysis below is retained as the record and governs the post-v1
revisit.

### Why the burden is already light

Three properties mean StickyKV needs far less than a uniform-quant cache (KVQuant
keeps ~1% of entries in fp16; that is overkill here):

1. **Per-channel keys** give each channel its own scale, neutralising the dominant
   inter-channel key-outlier structure.
2. **Pre-RoPE storage** keeps that per-channel structure consistent (RoPE smears it),
   so per-channel scales stay tight.
3. **Two-tier split** puts the highest-attention windows in fp16; the Q tier holds
   only mid-importance windows, so its worst outliers are disproportionately *already*
   in the fp tier.

### The three option families

| option | mechanism | wrinkle in this system |
|---|---|---|
| **A. Dense-and-sparse** (KVQuant) | keep top ~1% entries in an fp16 side-list | irregular gather/scatter breaks the clean layout; best quality |
| **B. Hadamard rotation** (QuaRot/RotateKV) | rotate channels to spread outliers | conflicts with per-channel keys; **does not commute with read-time RoPE on keys** — but is free on *values* |
| **C. Non-uniform codebook** (NF4/NUQ) | distribution-matched 16-level LUT | cost is phase-dependent — see below |

### Is NF4 costly? — it depends on the phase (this corrects an earlier over-statement)

The earlier draft deferred NF4 by charging it the *Phase 2 tile* cost. That was wrong
for v1. Split by phase:

- **Phase 1 (v1, materialize):** NF4 dequant is `codebook[codes] * scale` — a single
  vectorized gather (`torch.take`/`F.embedding`) over the Q tier, memory-bound, the
  same order as the affine multiply. **Effectively free in v1.** So there is no v1 cost
  reason to withhold it — this is the answer to "why not do it right away": *for v1, do
  it.*
- **Phase 2 (fused Triton tile):** here it matters. Affine dequant is one FMA (pure
  ALU, hidden under memory latency — the property that makes the fused tile free). NF4
  replaces that FMA with a 16-entry LUT lookup (register shuffle, or shared-mem load
  with possible **bank conflicts**) that can *serialize*. So Phase 2 revisits whether
  to keep NF4 in-tile or fall back to affine keys for the kernel — a kernel decision,
  not a reason to withhold NF4 from v1.
- **Calibration:** a fixed NF4 codebook (QLoRA constants) has zero calibration cost; a
  per-channel NUQ codebook (k-means) adds offline work per demotion. Use the fixed
  codebook to keep the pinned-grid freeze trivial.

### Earlier recommendation (superseded for v1 — retained as the post-v1 record)

Adopt both cheap pieces up front; keep only the genuinely-costly piece contingent:

- **Values → Hadamard rotation folded into the weights (free, always-on).** Values
  carry **no RoPE**, so a shared Hadamard `H` on head_dim folds entirely offline: `H`
  into `W_v` (values born rotated, `V·H = x·(W_v·H)`) and `Hᵀ` into `W_o` (output
  un-rotates, `(P·V·H)·(Hᵀ W_o) = P·V·W_o`). Zero extra decode FLOPs, zero extra
  storage, lossless on the fp tier. Adopt.
- **Keys → NF4-style non-uniform codebook, per-channel, no rotation (cheap in v1).**
  Keep the pinned per-channel absmax scale; swap uniform levels for a fixed 16-entry
  codebook. Free in the v1 materialize path (above); revisit for the Phase 2 tile.
- **Contingency (the one with real operational cost even in v1): micro dense-sparse
  (~0.1–0.25%).** An fp16 side-list needs an irregular gather and breaks the clean
  chronological layout, so add it **only if** the int4 LongBench gate misses — kept far
  smaller than KVQuant's ~1% because the fp tier already holds the biggest spikes.

So the split is now by *actual cost*, not caution: value-fold and NF4 keys are the
baseline (both cheap in v1); dense-sparse is the sole contingency.

### Cost: recommended baseline vs no outlier handling (v1 / materialize)

| axis | no outlier | value Hadamard fold | NF4 keys | (contingency) micro-sparse |
|---|---|---|---|---|
| **v1 decode compute** | baseline | **+0** (offline fold) | ~0 (vectorized gather) | +irregular gather (~0.1–0.25%) |
| **storage** | baseline | **+0** | ≈0 (may drop the zero-point) | +~0.1–0.25% fp16 |
| **offline prep** | — | fold `H` into `W_v`,`W_o` once | pick 16-level codebook | select outliers per demotion |
| **Phase 2 tile** | 1 FMA | +0 | LUT lookup (may serialize) | breaks tile locality |

**Bottom line:** the recommended int4 baseline costs ≈ no-outlier in v1 (both additions
are free/near-free in the materialize path). Real cost appears only in the Phase 2 tile
(NF4 LUT) and in the optional sparse net. And the dominant v1 cost overall is the fp16
write-back of the materialized Q tier (design.md §8) — outlier handling is second-order.

### v1 decision: no outlier machinery (NF4 + value-fold deferred beyond v1)

**Ratified 2026-07-06.** v1 ships with the pinned asymmetric affine int4 quantizer
(design.md §2) and **no outlier handling**. NF4 keys and the value Hadamard fold —
recommended above as near-free in the materialize path — are **deferred beyond v1
altogether**:

- **The value fold is model surgery** (integration audit item 11): a
  weight-modification step at model load with its own correctness surface (per-head
  block-diagonal fold, GQA head mapping, tied-weight edge cases), for a benefit the
  SAW-INT4 ablations suggest is second-order (key-side rotation dominates
  value-side, and our keys are already per-channel pre-RoPE).
- **NF4 keys change the grid** the quantizer tests must certify and add a codebook
  choice; the free-in-v1 analysis above stands, but v1 correctness is served better
  by one quantizer with a byte-comparable KIVI reference.
- **The structural mitigations are the heavy lifters** per the burden analysis at
  the top of this section: per-channel pre-RoPE keys, the fp tier holding the
  largest-spike windows, and the always-fp16 sink.

The sole contingency remains the micro dense-sparse side-list (~0.1–0.25%), added
only if the int4 LongBench gate misses. NF4 keys and the value fold re-enter
consideration post-v1, alongside the Phase 2 in-tile kernel decision.

---

## Integration audit — required changes to current StickyKV code

Grounded against the current `windowed_cache` (mirrored in `windowed_eager_cache`).
Roughly ordered from cache-core outward.

1. **`policy.py` — two-tier split.** `compute_retain_window_indices` does one
   `torch.topk` over the **evictable** slice and force-appends local. Two-tier keeps
   sink + local **force-fp** (never ranked, never eligible for Q) and partitions only
   the evictable band: top `top_k_fp` → fp, next `N_q` → Q, rest dropped. Return both
   retained sets (fp = local ∪ top_k_fp evictable; Q = next `N_q` evictable). A
   low-scoring local window must stay fp — do not let it fall to int4.
2. **`config.py` — resolver.** Add `quant_ratio q`, bit-width, scale dtype; compute
   `N_fp`, `N_q`, `b_q`. `ResolvedConfig` gains fields (design.md §7).
3. **`state.py` — Q store + ledger.** New packed-int4 store (channel-major keys,
   token-major values) + per-window ledger (`original_window_id`, `codes`, `scale`,
   `zero`, `offset`, **immutable** `position_range` = original absolute positions).
4. **`state.py` — demotion un-rotate (one-time).** Un-rotate a demoting window's fp
   keys once, applying RoPE with negated sin at the window's **original absolute
   positions**, to produce pre-RoPE codes. This is a standalone op — `rerotate_keys`
   was **removed** with the rerotation revert (design.md §5); do not reintroduce it, and
   do not renumber positions.
5. ~~New `build_interleaved_position_map`~~ — **not needed.** Positions are never
   renumbered, so there is no interleaved map, no `arange(T_total)`, and no fp
   re-rotation. `expand_to_token_indices` just becomes tier-aware (fp partition only).
6. **`cache.py` — `update()` eviction.** Insert tier assignment + demote/promote data
   movement + Q-store compaction (`offset` shift). **Return value changes**: from the
   live `state.key_states/value_states` to `materialize_effective_kv(...)` — a
   freshly-built chronological interleave, not the raw fp cache.
7. **New `materialize_effective_kv`** (shared) — dequant Q + RoPE at each window's
   immutable `position_range` + chronological interleave with fp by `original_window_id`.
8. **`cache.py` — `get_seq_length` returns effective `T_total`** (fp + Q), since HF uses
   it to size the causal mask over the returned effective K/V (and the flash hook uses it
   as `S`). This is a key-count report only — there is **no** query-position override to
   feed. Reporting fp-only length would undersize the mask against the effective keys.
9. ~~`position_override.py`~~ — **removed** (commit `ef9c84f`); query positioning is
   HF-native (monotonic absolute). Do not reintroduce it (design.md §5).
10. **`hooks.py` (flash) — score hook** reads `cache._states[l].key_states` directly
    (line ~219); must instead source effective K via `materialize_effective_kv`, or it
    scores only the fp tier and misses the interleave. **The one required flash-hook
    change.** (Eager attends over `update()`'s return, so its real-attention scoring
    needs no hook change — but it *does* depend on (6)/(7)/(8).)
11. **Value Hadamard fold — DEFERRED beyond v1** (see the v1 decision above; decision
    record in design_history.md Amendment 4). If revived post-v1 it is model surgery,
    outside the cache: folding `H`→`W_v` and `Hᵀ`→`W_o` at model load (e.g. in
    `cache_factory`/model setup); no weight-modification step exists today. Larger
    integration surface than the cache-local key changes — scope explicitly then.
12. **Mirror to `windowed_eager_cache`.** All `cache.py`/`state.py` edits stay
    byte-identical; the quant module + ledger are shared.
13. **`telemetry.py` — tier-migration counters.** Per layer per eviction: windows
    promoted / demoted (first-time) / reactivated / dropped, plus Q-tier occupancy.
    Required by design.md §10's instrumentation (promotion frequency + Suite A
    Jaccard-vs-fp-only).

Already anticipated by design.md §5–§9: (1)-(3), (6)-(8), (10). Simplified by the
keep-original-positions revert (commit `ef9c84f`): (4) demotion un-rotate is now a
one-time standalone op, not a per-eviction `rerotate_keys`; (5) and (9) are **dropped**
— no interleaved position map, no query override, because positions are never
renumbered (design.md §5, and Amendment 5 in design_history.md). (11) value-fold as
model surgery remains deferred beyond v1.

---

## Open items (resolve before the affected milestone)

*(Resolved 2026-07-06 and graduated to design.md §2: scale dtype = fp16; the exact
asymmetric affine quant/dequant formula is pinned; NF4 is deferred beyond v1. See
design_history.md Amendment 4.)*

- **int4 sparse-net gate.** Decide the LongBench threshold that triggers the micro
  dense-sparse escalation.
- **Explicit hysteresis.** Deferred; revisit only if Suite A Jaccard shows measurable
  K/Q boundary churn.
