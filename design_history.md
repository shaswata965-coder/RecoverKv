# StickyKV Quantization — Decision History

This file is the **decision record** for the two-tier quantization design: the
original prompt, the amendment log (how the design changed over time), the options
that were analysed and rejected, and the supporting rationale behind the locked
choices.

Nothing here is required to *implement* the feature — the current, self-contained
design lives in [design.md](design.md). This file exists so the "why", "what
changed", and "what we deliberately did not do" are not re-litigated during
implementation.

---

## Origin note: the `#N` / `G-N` numbering

Earlier drafts of the design tagged each resolution with a secondary number
(`#1`–`#13`) and referenced "gaps" as `G4`, `G5`, etc. Those tags pointed into a
challenge-analysis planning document
(`~/.claude/plans/to-integrate-quantization-into-witty-stonebraker.md`) that is **no
longer present in this repo**. The clean design drops the tags entirely; they are
preserved below only inside quoted historical material (the amendments and the
rejected-optimizations rationale), where removing them would corrupt the quote.

---

## Initial prompt (verbatim)

> To integrate quantization into our wqorkflow what would be the major challenges?
> How I intend to integrate quantization:
> we will be retaining top K+ top Q windows, these q windows will be stored in quantized form
> While decoding when we have a new window it will be compared against full precision and then dequantized window, if any dequantized window is more important than it will promoted and stored along with the full precision windows and a new window will be quantized
> During generation we will dequantize and quantize on the fly to store their Updated cummulated attention as well
> Finally we want to implement a pre Rope quantization, meaning during presses stripping of rope and re applying rope we want to do the quantization operation (if possible) so that rope does not accumulte quantization error

---

## Amendment log

The clean design already folds the **outcome** of every amendment inline. The blocks
below are retained as the change record only — they narrate what the design *was*
before each amendment and why it changed.

### Amendment 2 — rerotation RESTORED; compact + re-rotate + query-position override (superseded the `f80326b` amendment)

> Verified against the real NVIDIA/kvpress source: KVPress **does** re-rotate
> evicted-survivor keys (`KeyRerotationPress` rebases survivors to contiguous
> `[0..N_survivor-1]`, computes `delta_pos = idx − selected_positions`, and
> rebuilds cos/sin) **and** overrides the query position in its pipeline
> (`context_length = cache.get_seq_length()` ⇒ the query lands at the compacted
> length, the "N_survivor+1" slot — not its original position). The earlier
> `f80326b` amendment (keep-original-positions, rerotation off) was based on the
> **opposite, incorrect** reading and has been **reverted**. On **every** eviction
> the cache now: (1) compacts survivors contiguous, (2) **re-rotates** their keys
> to contiguous positions (`state.rerotate_keys`), and (3) a forward pre-hook
> (`utils.position_override`) overrides the query's `position_ids`/`cache_position`
> to the compacted cache length each step. The `rerotate_on_evict` knob is gone —
> this is the only path. Because the query position is set **explicitly**, this is
> correct independent of the transformers version. Implications for quantization:
>
> - **The original-prompt goal "pre-RoPE so RoPE does not accumulate quantization
>   error" (#4) is back in force — and is now a *correctness* requirement for the
>   Q tier, not merely a quality lever.** A strip→re-rotate cycle runs at **every**
>   eviction. If Q-tier keys were stored **post-RoPE**, keeping them consistent
>   with the rebased positions would require
>   dequant→un-rotate→re-rotate→re-quant each eviction; re-rotation changes the
>   values, so re-quantizing against the pinned grid is **no longer idempotent**
>   and quantization error **accumulates** across evictions (breaking #13).
>   Therefore the Q tier stores **pre-RoPE** codes: un-rotated **once** at
>   demotion, pinned grid, with RoPE applied **fresh at read** using the window's
>   current (contiguous) positions. The stored codes never change across evictions
>   — only the cos/sin applied at read do — so pinned-grid idempotence (#13) holds
>   and there is **zero** rope-driven quant-error accumulation.
> - **#2 / #8 — pre-RoPE is now the Q-tier default in BOTH phases** (was: post-RoPE
>   v1, pre-RoPE Phase 2). v1 materialize path: `dequant → apply RoPE at current
>   positions → concat [fp ‖ Q]` (one extra RoPE apply on already-dequantized data
>   — no custom kernel). Phase 2 tile: `load int4 → unpack → scale → apply RoPE
>   from cos/sin → MAC`. The demotion-time un-rotate **reuses `rerotate_keys`'
>   un-rotate half**, which now always runs.
> - **#3 / #9 — positions ARE rebased to contiguous every eviction** (not "never
>   rebased"). The Q store records each surviving window's **current** position
>   range and updates it at each eviction; because the codes are pre-RoPE
>   (position-independent) this costs only a cos/sin recompute at read, never a
>   re-quant. The query carries the **overridden compacted** `cache_position`, so
>   query↔key relative phase is exact (not the old compaction approximation).
> - **#7 / #10 — per-step Q cost gains one RoPE apply** (pre-RoPE read), still
>   bandwidth-trivial; the fp16 write-back remains the real v1 cost Phase 2
>   eliminates.
> - **G5 restored:** both tiers re-rotate every eviction. The fp tier re-rotates
>   in place (negligible fp error); the Q tier avoids compounding precisely
>   *because* it stores pre-RoPE codes (above).
> - **Both backends mirror this** (`windowed_cache` + `windowed_eager_cache`),
>   consistent with #10.
>
> The affected resolutions (#2, #3, #7, #8, #9, #10) were updated inline to match
> this amendment.

### Amendment 3 — Interleaved position map: compaction must span both tiers jointly

> **The problem with fp-only compaction.** `rerotate_keys` currently assigns
> `arange(T_fp_retained)` to fp survivors only, treating the Q tier as an
> afterthought appended at the end. This corrupts relative positions. Example:
> original windows W1(fp), W2(Q), W5(fp) survive an eviction. Fp-only compaction
> assigns W1→[0..ws-1], W5→[ws..2ws-1], then W2 is appended at [2ws..3ws-1].
> But W2 is chronologically **between** W1 and W5 — placing it last makes the
> model treat it as the most recent context, corrupting every Q·Kᵀ dot product
> involving W2.
>
> **The fix — interleaved position map.** At each eviction, merge ALL surviving
> windows (both tiers) sorted by `original_window_id` and assign contiguous
> positions `arange(T_total)` across the merged set:
>
> ```
> Surviving windows in chronological order: W1(fp), W2(Q), W5(fp)
>
> Interleaved position assignment:
>   W1 (fp) → [0 .. ws-1]       fp keys re-rotated to these positions
>   W2 (Q)  → [ws .. 2ws-1]     Q-tier ledger position_range updated (no re-quant)
>   W5 (fp) → [2ws .. 3ws-1]    fp keys re-rotated to these positions (gap for W2)
>
> Query overridden to: T_total = 3ws  (fp + Q combined, not just fp)
> ```
>
> Fp keys for W5 skip over W2's slot in position space even though no fp key
> occupies [ws..2ws-1]. `rerotate_keys` must receive the explicit interleaved
> target positions for each fp survivor, not just `arange(T_fp)`.
>
> **Concat is physically order-free.** The resulting concat `[fp_store ‖ dequant_Q]`
> can appear in any physical order because RoPE has already baked the correct
> logical position into each key at read time. W2's keys are dequantized and
> rotated to [ws..2ws-1]; they produce correct Q·Kᵀ dot products regardless of
> where they sit in the tensor.
>
> **Implementation impact:**
> - New helper `build_interleaved_position_map(fp_window_ids, q_window_ids,
>   window_size, num_sink)` — sorts all surviving window ids jointly, assigns
>   contiguous positions, returns: (a) fp-survivor target position tensor fed
>   to `rerotate_keys`, (b) per-Q-window `position_range` assignments written
>   to ledger.
> - `rerotate_keys(rope, old_pos, new_pos)` — gains explicit `new_pos` argument
>   (was implicit `arange(T_fp)`).
> - `position_override.py` — `cache_position` uses `T_total` (fp + Q tokens),
>   not `T_fp`.
> - Q-tier ledger — `position_range` updated from this map every eviction. Cost:
>   one integer assignment per surviving Q window; no re-quant (codes are
>   pre-RoPE and position-independent).
>
> The affected resolutions (#9 and #12) were updated inline.

### Amendment 4 — v1 quantizer pinned (fp16 scales, affine-only), ledger reactivation, merged window axis

> Ratified 2026-07-06, following an end-to-end design review against the current
> literature (KIVI/KVQuant lineage, RDKV, SAW-INT4, KVSink, Atom/FlashInfer):
>
> - **Scale dtype pinned to fp16; the fp8-scale option is removed.** fp8 scales
>   save ~1.5% of Q-tier bytes while injecting scale-quantization noise into every
>   dequant. This unblocks a deterministic `b_q` / `N_q`; the exact per-window byte
>   formula is now in design.md §7.
> - **Quantizer numerics ratified** (design.md §2): KIVI-reference asymmetric
>   affine, float-offset form (`zero = mn`), fp32 compute, round-half-even,
>   clamp-before-cast, degenerate-group rule `scale = 1` when `mx == mn`.
>   Quantization runs against the fp16-*stored* scale/zero so the fitted grid is
>   bit-identical to the read-path grid. The float-offset form was chosen over an
>   integer zero-point because K/V groups often exclude zero — an integer
>   zero-point clamped to `[0, 15]` cannot represent an offset outside the group's
>   own span (catastrophic for strictly-positive/negative channels).
> - **NF4 keys and the value Hadamard fold are deferred beyond v1 altogether**
>   (were the rationale file's recommended baseline). The value fold is model
>   surgery with its own correctness surface for a benefit SAW-INT4's ablations
>   suggest is second-order on values; NF4 changes the grid the tests must certify.
>   v1 ships one quantizer with a byte-comparable KIVI reference and no outlier
>   machinery; the micro dense-sparse side-list remains the sole contingency,
>   gated on LongBench.
> - **Ledger entries persist through promotion; re-demotion is a reactivation, not
>   a re-quantization.** The earlier claim — "a promote→demote round trip
>   re-quantizes against the old grid → idempotent → identical codes" — was only
>   epsilon-true: intermediate fp-tier re-rotations add fp16 rounding that can flip
>   boundary codes on re-quantization. Since pre-RoPE window content is immutable,
>   the stored codes never need recomputing: promotion keeps the ledger entry
>   dormant, re-demotion drops the fp copy and reactivates it. Exactly lossless,
>   zero compute, and it makes pinned-grid idempotence a structural property (no
>   path ever runs `quant(dequant(·))`) instead of a numerical one.
> - **Merged window axis specified** (design.md §5): `window_scores` /
>   `original_window_ids` / ranking all live on one merged chronological window
>   axis spanning both tiers (the order `materialize_effective_kv` emits);
>   `expand_to_token_indices` becomes tier-aware — only the fp partition expands to
>   fp-store token indices, Q windows resolve through the ledger. This closes the
>   previously-undeclared gap between the merged scoring axis and the per-store
>   physical token axis.
> - **Phase 3 claim corrected:** FlashInfer is fp8/fp4-native; the int4 decode
>   kernels in the literature (Atom) are custom builds on top of it. Phase 3 either
>   re-targets the Q tier to fp8/nvfp4 or ports an Atom-style int4 kernel.

### Amendment 5 — rerotation REMOVED again; keep-original-positions is the cache; pre-RoPE re-justified (supersedes Amendments 2–3)

> **Ratified 2026-07-13, following the code revert in commit `ef9c84f`
> ("eviction: remove KVPress rerotation; keep original RoPE positions").** The
> single-tier cache no longer re-rotates survivors or overrides the query position:
> `CacheState.rerotate_keys`, `utils/position_override.py`, and the `rope_module`
> ctor param were **deleted**. Eviction now only **compacts** memory; surviving keys
> keep the RoPE rotation baked in at their **original absolute positions**, and the
> query keeps HF's monotonic absolute position (correct on transformers ≤ 4.47.1).
> This is the reading Amendment 2 had reverted as "incorrect" — it is now the shipped
> cache, and the quantization design is reconciled to it. Implications:
>
> - **Amendments 2 and 3 are superseded.** There is **no** interleaved position map,
>   no `arange(T_total)` renumbering, no `rerotate_keys(new_pos)`, and no query
>   override. `build_interleaved_position_map` is not built. The §5 rewrite in
>   design.md reflects this.
> - **Pre-RoPE is retained, but its justification changes from correctness to
>   quality.** Amendment 2 made pre-RoPE a *correctness requirement* ("positions are
>   rebased every eviction, so post-RoPE would need re-quant → error accumulation").
>   That argument is now **void**: positions are never rebased, so a demoted window's
>   post-RoPE content is already frozen, and post-RoPE codes would be equally valid.
>   Pre-RoPE is kept for **(1) quantization quality** — per-channel key ranges are
>   tighter pre-RoPE, since RoPE smears the inter-channel structure the scales exploit
>   — and **(2) Phase-2 kernel consistency** (the fused tile applies RoPE in-register).
>   The original-prompt goal "pre-RoPE so RoPE does not accumulate quantization error"
>   is honoured, now as a quality/kernel lever rather than a per-eviction necessity.
> - **The ledger `position_range` becomes immutable.** It holds each Q window's
>   original absolute positions, set once at demotion, fed to RoPE at read. The only
>   per-eviction ledger write is the byte `offset` (Q-store compaction). The "one
>   integer write per Q window per eviction, must never be skipped" hazard from
>   Amendment 3 is gone.
> - **Demotion un-rotate is a one-time standalone op**, not a reuse of a
>   `rerotate_keys` "un-rotate half" (which no longer exists): apply RoPE with negated
>   sin at the window's original absolute positions, once, at first demotion.
>   Reactivation on re-demotion stays exactly lossless — content **and** read-path
>   positions are both immutable.
> - **`get_seq_length` still reports effective `T_total`** (fp + Q), but now purely to
>   size the causal mask over the materialized effective K/V and to give the flash hook
>   its key count `S` — **not** to feed a query-position override (there is none).
> - **G5 below ("both tiers re-rotate every eviction") is void**; neither tier
>   re-rotates. The "Async eviction" and "Hysteresis" rejected-optimization notes that
>   assume a per-eviction strip→re-rotate cycle are correspondingly weakened (the
>   per-eviction work is now only demotion un-rotate on newly-crossed windows), but the
>   dropped verdicts stand.
>
> The self-contained spec is design.md as of this amendment; the integration prompt
> and design_rationale.md integration audit are updated to match.

---

## Considered and explicitly rejected optimizations

The following were analysed and dropped before v1. Recorded here so they are not
re-debated during implementation.

- **Async eviction** (overlap the RoPE strip + re-rotate + quantize with FFN via
  CUDA streams): a strip→re-rotate cycle now runs every eviction, but the per-step
  work is still bounded (a few newly-demoted windows). Overlapping it requires
  decoupling this step's attention from the compaction (Suite A parity break),
  raises peak memory during overlap (pre- and post-eviction buffers coexist), is
  GPU-only, and adds stream/determinism risk. **Dropped.**

- **Deferred memory movement** (flag migrations, batch copies every N steps):
  movement already happens only at eviction cadence, batched, boundary-crossers
  only. Pushing N beyond `window_size` overshoots the memory budget during deferral
  and splits logical tier from physical store (ambiguous precision in read/score
  paths). Safe substitute: hysteresis (see below). **Dropped.**

- **Prefetch next-layer Q-tier dequant during FFN** (overlap dequant with the
  adjacent layer's FFN): decode is memory-bandwidth-bound on weight loading already;
  FFN weight traffic saturates HBM, leaving no free bandwidth shadow. Running dequant
  concurrently on a second stream adds bytes moved rather than hiding them. Q-tier
  dequant is ~1% of step bandwidth — real but unhideable by scheduling. The right
  lever is eliminating the write-back (Phase 2 fused kernel), not prefetching it.
  **Dropped.**

- **Hysteresis at the K/Q boundary** (require a score margin before migrating):
  demotion costs one un-rotate + quantize and promotion one dequant — both cheap.
  With the pinned **pre-RoPE** grid, an oscillating window's codes are unchanged
  across migrations (idempotent, zero error). Hysteresis adds a tunable margin knob
  for marginal migration savings. Its only remaining value is damping the score-
  feedback drift (see G4 below), already instrumented via Suite A Jaccard. **Dropped
  as a v1 concern;** revisit if Suite A shows measurable boundary churn.

---

## Supporting rationale

### G4 — score-feedback loop

`window_scores` are accumulated from attention weights computed over the dequantized
Q tier. Because int4 is lossy, those attention weights are slightly wrong, so the
score increments for Q-tier windows are noisy. Windows near the K/Q score boundary
are the most exposed — small noise can flip a demotion decision, causing spurious
promotion/demotion churn. Bounded naturally because the fp tier (sink + local +
top-K) dominates attention mass and anchors most scores. Detected via **Suite A
Jaccard**: compare which windows survive in a two-tier run vs a pure-fp baseline
over long sequences; large divergence flags feedback drift. Hysteresis is the
surgical fix if detected, but is deferred (see rejected optimizations).

This is why v1 keeps **full bidirectional promotion** (per the initial prompt) but
**instruments promotion frequency + Suite A Jaccard-vs-fp-only over long sequences**.
The documented fallback, **not chosen**, was one-way demotion + frozen Q-scores.

### G5 — both tiers re-rotate every eviction ~~(VOID — see Amendment 5)~~

> **Superseded by Amendment 5 (2026-07-13).** Neither tier re-rotates any more —
> eviction keeps original absolute positions. Retained only as the record of the
> rerotation-era rationale.

The fp tier re-rotates in place (negligible fp error); the Q tier avoids compounding
error precisely *because* it stores pre-RoPE codes — the stored codes never change
across evictions, only the cos/sin applied at read.

---

## Environment / version decision

Target: **transformers 4.47.1** across all eval devices (`environment.yml` is pinned
to the 4.47.x line). The current live status of the dev machine, and the reason a
newer transformers cannot be used, are documented as an implementation prerequisite
in [design.md](design.md) under **Environment caveat** (kept there because it is a
live blocker an implementer must act on, not merely a past decision). In short: the
4.47.1 pin exists because transformers 5.x builds the causal mask via
`create_causal_mask` → `Cache.get_mask_sizes()`, which `WindowedCache` does not
implement, so a full-model forward crashes on 5.x.
