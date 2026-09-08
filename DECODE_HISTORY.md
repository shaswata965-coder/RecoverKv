# Decode optimisation — what was tried, what landed, what went wrong

Companion to `DECODE_SPEED_PLAN.md`, which holds only the *remaining* work. This
file is the record: the changes that shipped, the numbers they moved, and every
failure along the way — including the ones that were mine, because those are the
ones worth not repeating.

---

## 1. The result

Llama-3.1-8B, `run_perf_table.sh` at shipped defaults (budget 0.50, q=0.70,
`ws=8`), one GPU. TPOT_steady, seconds:

| shape / B | before | + batched eviction | + window-score kernel | total |
|---|---|---|---|---|
| **4096/256, B=32** | 0.1760 | 0.0735 | **0.0626** | **2.81×** |
| 2048/512, B=32 | 0.1142 | 0.0641 | **0.0531** | 2.15× |
| 1024/1024, B=32 | 0.1045 | 0.0607 | **0.0488** | 2.14× |
| 4096/256, B=1 | 0.1011 | 0.0586 | **0.0471** | 2.15× |
| 2048/512, B=1 | 0.1024 | 0.0588 | **0.0468** | 2.19× |
| 1024/1024, B=1 | 0.1016 | 0.0594 | **0.0467** | 2.18× |

Against FullKV (Flash) at the headline cell: **2.05× slower → 1.37× faster.**
Both stated targets met — beat FullKV, and beat it by 20% (≤ 0.0688).

Peak memory did not regress: 49.20 → 48.44 GB at 4096/B=32.

---

## 2. What landed

### 2.1 Cross-layer batched eviction

All 32 layers evict on the same step at identical shapes (`ws`, budget, `n_q`,
`k_fp` come from config, not data), so folding `L` into the row axis
(`R = L·B`, layer *i* owning rows `[i·B, (i+1)·B)`) lets one batched pass replace
32 sequential ones. Every op in the eviction body is already per-row, so the body
itself did not change — only its row count.

**The ordering change this forced.** With one joint store the eviction must run
*before any layer appends this step's token*, because when layer 0's `update`
is called at step *t*, layers `1..L-1` have not produced token *t* yet. So
"append then compact" became "compact then append".

That is safe because the retained set does not depend on this step's token:
`compute_two_tier_retain` reads **only** `window_scores`, and this step's token
has no score column yet and lands in the local region, which is force-fp and
never evictable. Not assumed — `tests/test_layer_major_evict.py` pins whole-cache
byte-identity against the per-layer path across `q ∈ {0, 0.5}`, `L ∈ {1,3}`,
`B ∈ {1,2}`, `ws ∈ {1…128}`, sinks on and off, and promotion/dormancy.

Measured on CPU (ATen dispatches, L=32 B=4 ws=8): eviction body **13,824 → 426
(32.5×)**; amortized per decode token **5,592 → 1,177 (4.75×)**.

Prefill stays per-layer deliberately (`STICKYKV_PREFILL_LAYER_MAJOR`, default
off): a layer-major prompt buffer would force the first compaction to hold the
whole `L·B`-row prompt and the whole `L·B`-row steady store at once, ~+4 GB of
peak straight out of max batch. Joining *after* the first eviction, once every
prompt buffer is released, costs nothing.

`STICKYKV_LAYER_MAJOR_DECODE=0` is the control arm, not a fallback.

### 2.2 Window scores from the kernel, whole-window tiling, fewer args

One kernel rewrite, because all three touched the same loops.

- **Window scores.** The kernel emits `wsum[B,H_q,W_phys]` (per-window softmax
  mass, physical order: body windows then Q windows) plus a `wmax` companion, and
  rescales in a register epilogue over `W_phys` instead of a second pass over
  `S`. Traffic `4·S → 5·W`. The caller's reduction collapsed from
  pad + reshape + sum + einops-reduce + cat + gather to **one gather**.
- **Whole-window tiling, both tiers.** The Q loop went from one window per serial
  iteration to `BLOCK_NW` windows; the fp body was re-tiled the same way. At
  `ws=8` that is 179 → 23 Q-tier iterations at the top tile rung.
- **Fewer kernel arguments.** All 21 stride arguments for the Q-tier tensors,
  `cos`/`sin`, `OUT` and the score buffers are derived from shapes in-kernel.
  `q`/`k_fp`/`v_fp` keep explicit strides — they are transposed views, and
  forcing them contiguous would copy the whole fp tier every step. The dispatcher
  **checks** contiguity and raises rather than assuming it. 50 args → 41.

**`window_size` 1–128 all work, non-powers-of-2 included.** Tiling in *whole
windows* padded up to a power-of-2 block means a window never straddles a tile
for any `ws`, and the `num_sink` offset stops mattering because the body loop
starts at `num_sink` and steps in window units. Cost is padding alone: 100% lane
use at `ws ∈ {1,2,4,8,16,32,64,128}`, 94% at `ws=12`. The sink prefix gets its
own prologue tile that feeds the softmax and emits no window score.

**How it was verified without a GPU.** Triton cannot run on the dev box, so the
*lowering* shipped unvalidated — the contract the score kernel and the original
decode kernel already ship under. The *algorithm* did not:
`two_tier_window_reference` reproduces the kernel tile for tile, including the
power-of-2 padding lanes, and `tests/test_window_scores.py` pins it — `ws`
1–128 × `num_sink ∈ {0,1,5}`, partial trailing windows, softmax-mass
normalisation, overflow safety, an exhaustive padding-exclusion proof for every
`ws` from 1 to 128, and **five end-to-end cases against a real two-tier cache**
where the old reduce path and the new one-gather path are driven off the same
store, effective K/V and `score_meta`.

### 2.3 `cache_position` contract

An evicting cache breaks an assumption transformers makes. With no explicit
`cache_position`, `LlamaModel.forward` derives it as
`arange(get_seq_length(), …)` — treating "keys held" as "tokens processed". Those
diverge the moment the budget binds, because `get_seq_length()` returns the
*retained key count*, which sawtooths down by `ws-1` at every eviction. The step
after an eviction is then told it sits ~`ws` tokens **earlier** than the one
before it, so the query's RoPE phase goes backward and the store accumulates
duplicated, non-monotonic positions (measured: `… 70, 71, 72, 65, 66, 67`).

`generate()` is immune — it advances `cache_position` itself and never re-derives
it — so the quality runners were always safe. `perf_runner`'s hand-written decode
loop was not. It now passes an explicit monotonic position in both the warmup and
measured loops, and `WindowedCache._check_position_contract` refuses a wrong base
outright.

This never moved any reported number: store length is config arithmetic
(`k_fp*ws + …`), not data-dependent, so shapes, launch counts and KV traffic were
identical either way — only HF's own `arange` differed, one dispatch per step.
What it moved is whether the harness being optimised runs the semantics that ship.

### 2.4 Measurement gates

`run_perf_table.sh` rejects a comma in `CUDA_VISIBLE_DEVICES` before loading 16 GB
of weights; `perf_runner._assert_single_device` re-checks after the model loads.
Reason in §4.4 below.

---

## 3. What was tried and rejected

| idea | why not |
|---|---|
| **Layer-major *prefill*** | Would hold the `L·B`-row prompt and the `L·B`-row steady store simultaneously at the first compaction: ~+4 GB peak, straight out of max batch. Joining after the first eviction is free. Gated off. |
| **Power-of-2 `ws` requirement for window scores** | The first design concluded `BLOCK_N % ws == 0` forced a power-of-2 `ws`. Wrong — tiling in whole windows with a padded block removes the constraint entirely and costs only lane padding. |
| **Per-device grouping for multi-GPU** | Correct fix, but on 8 GPUs / 32 layers it is 4 layers per group — a 4× launch cut, not 32×. Llama-3.1-8B fits on one 80 GB card, so sharding is unnecessary here. Refused instead, with both escapes named. |
| **Triton launcher bypass** (`CompiledKernel` direct launch) | The API moves between Triton versions and cannot be tested on a CPU box. Only the argument count was cut. Revisit if Stage 0 shows `JITFunction.run` dominating. |
| **Forcing `k_fp`/`v_fp` contiguous** to derive their strides too | They are transposed views; `.contiguous()` would copy the entire fp tier every step. Kept explicit strides for those three tensors only. |

---

## 4. What went wrong

### 4.1 Shared memory: the tiling regression *(GPU run 1 — all six cells failed)*

```
OutOfResources: shared memory, Required: 176128, Hardware limit: 166912
```

Widening the Q-tier tile from one window (`BLOCK_WS=16`) to `BLOCK_NW` windows
took `BLOCK_T` to 64, which quadrupled the `tl.dot` operand staging —
`k_rlo`/`k_rhi` go `[64,16] → [64,64]` and `vv` goes `[16,128] → [64,128]`. That
is ~128 KB of dot operands before Triton's pipelining multiplies it, against an
A100's 163 KB/SM. The launch also used Triton's *default* `num_stages` (3+),
which took 128 KB to the observed 172 KB.

**Fix:** a shared-memory **fit ladder** — try `(target_keys, num_stages)`
fastest-first, step down on `OutOfResources`, cache the winner per geometry,
announce which rung won. Every rung is numerically identical, so this is a tuning
search, not a correctness fallback; if none fits it raises with the whole ladder.

**Lesson:** a tile-size change is a shared-memory change. The arithmetic
(`2·HALF·BLOCK_T·4 + BLOCK_T·D·4 + …`) was computable up front and was not
computed.

### 4.2 Multi-GPU: the join cannot span devices *(GPU run 2)*

```
batch=1   RuntimeError: Expected all tensors to be on the same device,
                        cuda:0 and cuda:1  (in wrapper_CUDA_cat)
batch=32  RuntimeError: CUDA error: device-side assert triggered
later     NotImplementedError: flash_attn::_flash_attn_forward … 'CPU' backend
```

`device_map="auto"` shards **by layer**. `CacheState.join_layers` and
`QuantSlotTable.join_layers` concatenate all `L` layers into one row axis, which
cannot span devices.

**All three error shapes are one cause.** The `batch=32` device-side assert is the
same mismatch surfacing through `torch.compile`; the later `CPU backend` errors
are not a separate bug at all — a device-side assert **poisons the CUDA context**,
so every subsequent cell in the sweep fails in whatever way it happens to fail.
Reading them as three independent problems wasted a diagnosis round.

**Fix:** `_migrate_to_joint` checks all layers share one device and raises with
both escapes named.

**Lesson:** `torch.cat` across a list of per-layer tensors carries an unstated
single-device assumption. Any code that folds a layer axis into a tensor axis
inherits it.

### 4.3 The advice that did not take effect *(GPU run 3 — 8 GPUs)*

The guard fired correctly and named `cuda:0 … cuda:7`. But the recommended fix —
"use `CUDA_VISIBLE_DEVICES=0`" — had no effect, because `run_perf_table.sh` only
*defaults* it:

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
```

An existing export in the shell wins. It has to be set on the invocation.

**Lesson:** when advising an environment variable, check whether the script sets
it or merely defaults it.

### 4.4 The framing that was too narrow *(same run)*

Treating this as "the join can't span devices" was the smaller half. The real
problem is that a **pipeline-sharded decode step measures the interconnect, not
the KV cache** — each token serialises 32 layers across N devices with a transfer
between each. Even with every code path working, that TPOT is not comparable to
the single-GPU baselines printed beside it in the same table. Numbers that look
like results without being results are worse than an error.

**Fix:** two fail-fast gates (§2.4), justified on measurement validity rather than
on capability.

### 4.5 Estimation errors

| claim | reality |
|---|---|
| Window scores worth "~19 ms host" | That used a µs-per-launch rate calibrated against an excess 2.4× larger. Retired. |
| Then re-sized to "~0.5 ms" from traffic | **Actual: ~11 ms**, uniformly across every shape and batch. The traffic half really was ~0.5 ms; the launch and serial-latency halves — which were left unsized — were 20× that. The near-constant gain across all shapes is itself the signature: a fixed per-step cost was removed, not a traffic term. |
| "LongBench/GSM8K run `quant_ratio: 0.0`, so the kernel is never reached" | Wrong. The YAMLs say 0.0 but the runners override `cache.quant_ratio` on the command line (`KAGGLE_RUNBOOK.md:159`). The quality runs **do** exercise the fused kernel. Read the invocation, not the config default. |

**Lesson:** the two estimates that were wrong were both traffic-only. At this
cell 72% of per-step traffic is model weights, so a traffic-derived estimate is
structurally an underestimate of anything that touches launches or latency.

### 4.6 Bugs caught before shipping

- **Body-end overrun.** Deriving the fp body's end from `n_body_win * ws`
  overruns into the Q tier whenever the newest body window is partial — the
  common case. The kernel masks the body at `Sfp`; the oracle initially did not.
  Caught by the end-to-end tests, not the unit ones.
- **A verification gap in the oracle.** It sliced the exact `[lo, hi)` range, so
  it modelled the *tiling* while silently assuming the *masking* — and the
  masking is precisely what could fold a padding lane into a window. Now it
  materialises the padding lanes, and an exhaustive test covers every `ws` from 1
  to 128 in both directions.
- **Two harness bugs of my own** while sweeping `ws`: a prompt drawn from the
  advancing global RNG (so the two paths compared different prompts), and a
  manual decode loop without `cache_position` (which produced the corruption of
  §2.3 and looked at first like a cache bug).

---

## 5. Standing lessons

1. **Size launches and latency separately from traffic.** Traffic is the easy
   number and, here, the small one.
2. **A CPU oracle must mirror the kernel's mechanics, not just its maths** —
   including the parts that exist only because of hardware constraints (padding,
   masking, tiling). An oracle that models the safe half and assumes the risky
   half verifies nothing.
3. **Cascading CUDA failures are one bug.** After a device-side assert, every
   later error is noise.
4. **Compute the resource budget before changing a tile size.**
5. **Read the invocation, not the config default.**
