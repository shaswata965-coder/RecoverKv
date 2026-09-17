# RecoverKV — a two-tier evicting KV cache

A KV cache for long-context decode that keeps a small fp16 tier exactly, holds a
much larger int2 tier approximately, and reads only a quarter of that int2 tier
on any given step. The aim is more batch at the same memory, and decode latency
at or below FullKV-flash.

Target model: Llama-3.1-8B (GQA, 32 query heads over 8 KV heads, `D=128`), one
A100-80GB, `window_size=8`, `num_sink=5`.

---

## 1. The method

The context is cut into **windows** of `ws` tokens. Every window carries a
cumulative attention score. At each eviction the windows are ranked and split
into three bands:

| band | storage | rule |
|---|---|---|
| sink + local | fp16, never evictable | first `num_sink` tokens, last `local` tokens |
| **fp tier** | fp16, exact | the top `top_k_fp` evictable windows by score |
| **int2 tier** | 2-bit codes + a grid | the next `N_q` windows |
| dropped | — | everything below |

`quant_ratio` (`q`) splits the evictable budget between the two tiers.
`quant_budget_mode` decides what it splits: `bytes` (the memory claim — both
tiers divide a byte allowance, cheaper keys buy more windows) or `tokens` (the
latency claim — the window count is pinned so two rows do equal work).

Eviction runs every `ws` decode steps, **compact-then-append**, for all 32 layers
at once: the layer axis is folded into the row axis (`R = L·B`) so one batched
pass replaces 32 sequential ones. Positions are never renumbered, which is what
lets a demoted window's encoding be frozen for life.

### The read gate

Dequantizing the whole int2 tier every step is most of the cost, and most of it
is wasted: the int2 tier is by construction the band ranked *below* the fp tier.
So each window carries a **card** written once at demotion, and the decode step
reads only the windows whose cards say they matter.

A card is a rank-1 model of the window's post-RoPE keys, plus a value centroid:

```
k_i ≈ mu + t_i · v        v = direction of the farthest deviation from the mean
vbar = mean of the window's values
```

`mu`, `v`, `t` and `vbar` are int8 against a per-(window, head) scale, and `mu` /
`vbar` are stored as residuals from a frozen per-(layer, head) anchor — massive
activation channels are near-identical across windows, so anchoring cuts `mu`
error 17×. 400 B/head, 3200 B/window at `H_kv=8`.

**A centroid alone does not work as the selector.** `exp` is convex, so a mean
always understates a window's mass and understates it most when the window holds
one hot token and seven cold ones — exactly the case the gate must not miss.
`v` points at the token furthest from the mean, which in a sparse window is the
hot one, so the model spends its one direction where it is needed. This is why
`v` must be int8: at 1 bit the hot token ends up with the *largest* residual in
its window, worse than useless.

### One decode step

1. **Score every window from its card** — two `D`-length dots per (window, head)
   against the rank-1 model, no int2 codes touched. Yields `logmass` (what a
   window is worth) and `est` (what to rank on). One Triton launch.
2. **Select** the top 25% per KV head — per *KV head*, not per row: a program
   owns a KV head and every query head sharing it, so that is the unit of work.
   Measured at `ratio=0.25` on a 271-window tier: per (row, KV head) reads
   68/271 at 100% mass recall; a per-row union reads 240/271 and the gate buys
   nothing.
3. **Attend** over `[sink | fp body | the selected int2 windows]`. The selection
   reaches the kernel as an indirection (`widx = tl.load(SEL + …)`), not a
   re-gather, and the int2 unpack + dequant + RoPE all happen in registers.
4. **Credit the rest.** The windows that were not read still contribute. The read
   windows have both a real mass and a card estimate, so their ratio is the
   step's *deviation* — how wrong the cards were against this query. Every
   skipped window is credited with `deviation × its estimate`, which goes to both
   its eviction score **and** the attention output via its value centroid. `out`
   and every score are then renormalized by `1 + Σ(restored mass)`.

Step 4 rides in an epilogue pass that already ran: the only added traffic is the
centroid tile, and the only added arithmetic is one `tl.dot` of a tile the loop
already holds. At `ratio = 1.0` every term in it is an exact no-op.

**Why crediting matters.** A skipped window scored zero would rank last and be
evicted — the gate would destroy the tier it exists to read less often, and at
`q=0.7` that is 70% of the evictable cache. And a skipped window absent from the
output is not an approximation of attention over the cache; it is exact attention
over a *different* cache, with the surviving weights inflated to cover the gap.

---

## 2. One path

There is one implementation of each thing, and no fallbacks. A path that cannot
run raises.

* **Decode** is the fused Triton kernel. There is no PyTorch decode.
* **Prefill scoring** is the fused Triton kernel. There is no PyTorch prefill.
* **The gate** is the read path wherever there is an int2 tier. `quant_gate_ratio
  = 1.0` selects every window — the no-op proof on the one implementation, not a
  second one.
* **Eviction** is compiled-or-raise, layer-major, always.
* **L-reuse** always. Recomputing the softmax normaliser is a second O(N²) pass
  whose fp32 block is 32 GB at 4096/B=32 — larger than the weights, and the
  reason that cell OOMed.
* **Both backends share one cache.** `flash_attn` and `eager` differ only in
  where eviction scores come from (the fused kernel vs `attn_weights`), which
  `install_score_hooks` picks from the attention implementation.

Everything above was once selectable. 21 `STICKYKV_*` environment variables, a
`quant_read_gate` arm, a `quant_memoize_read` memo, a FlashInfer LSE backend, a
CUDA-graph runner, a whole second cache package and a pure-PyTorch decode oracle
have been deleted. The cost of keeping them was not the code — it was that no
reported number said which of them it came from.

### Why the CUDA graph is gone

It targeted the right thing: the decode step is **host-bound**, 32.3% GPU busy,
with ~28 ms of launch gap. But as built it captured every steady step and
replayed none of them — a slot is destroyed at the next eviction and each slot
value occurs once per epoch — so it paid capture time plus a per-epoch memory
pool teardown, returned nothing, and died in the CUDA allocator. That is where
the six ERROR rows came from. The host gap is real and still has to be closed;
it has to be closed by issuing fewer launches, not by recording them.

---

## 3. Challenges, and what each one cost

**The budget did not count the card.** `bytes_per_q_window` counted codes and
scales only, so every reported `cache_budget` understated what was held by 3200 B
against the codes' 8448 — 38% on the exact tier the memory claim is about. Fixed;
at budget 0.50 / q=0.70 the honest resolve is `N_q=518`, not 715. `resolve` now
also spends the remainder (two floor divisions were leaving a window of each tier
unbought) and *enforces* it: the leftover must be too small to buy another
window, or it raises.

**Reading less got slower.** The gate landed as a **24% decode regression**
(TPOT 0.0626 → 0.0777 at 4096/B=32). Three compounding causes:

1. *It skips 75% of the windows but only 28% of the bytes.* The fp16 scale/zero
   grid is **48% of an int2 window** — two bytes of precision wrapping a two-bit
   code — and the fp tier, which the gate never touches, is 87% of what the gated
   kernel reads. Traffic saved: 0.23 ms.
2. *`SEL` turns coalesced streams into gathers.* Ungated, `widx = w0 + t_win` is
   affine and Triton emits contiguous vector loads. Gated, it is a runtime value
   with an address dependency, so every field becomes a scattered gather. The
   kernel runs **7.03 ms against a 0.58 ms roofline — 12× off**.
3. *It added a full-tier pass.* The score-fill loop runs over every Q window, not
   the selected ones, plus a prologue seeding all Q columns.

Net: save 0.23 ms of traffic, pay for gathers plus two full-tier passes.

**The profiler counted every kernel twice.** `key_averages()` returns both the
ATen operator view and the CUDA kernel events, and both carry self device time.
Reported 87.30 ms/step and 3,132 launches; real is **48.05 ms and 1,656**, and
GPU busy 58.7% "mixed" is really **32.3% — host-bound**. Three replies were also
made against a stale checked-in HTML copy of the results table rather than the
published one, and concluded the build was 2× faster than baseline. Both
conclusions were false.

**A traffic-only estimate is structurally an underestimate.** Two sizing errors,
both from counting bytes and not launches: the window-score rewrite was sized at
~0.5 ms and delivered ~11 ms. At these shapes 72% of per-step traffic is model
weights, so anything touching launches or serial latency is invisible to a
traffic count.

**Other things that bit, briefly.** A tile-size change is a shared-memory change
(the first GPU run died at 176 KB against an A100's 163 KB — now a fit ladder).
`torch.cat` over per-layer tensors carries an unstated single-device assumption,
so the layer-major join cannot span a `device_map="auto"` shard. A device-side
assert poisons the CUDA context, so every later error in a sweep is noise from
one bug. An evicting cache breaks `cache_position`: transformers derives it from
"keys held", which sawtooths down at every eviction, so a hand-written decode
loop must pass an explicit monotonic position. And a 0.009% micro-optimisation
(parking a reused context dict on the cache) closed a reference cycle and cost an
OOM at narrativeqa 109/200 plus 6% TPOT.

---

## 4. Where the numbers are

Published target (budget 0.50 / local 64) against the 2026-09-16 run
(budget 0.20 / local 128), TPOT_steady in seconds:

| shape | B | target | measured | gap |
|---|---|---|---|---|
| 4096/256 | 1 | 0.0471 | 0.0566 | +9.5 ms |
| 4096/256 | 32 | 0.0626 | 0.0763 | +13.7 ms |
| 2048/512 | 1 | 0.0468 | 0.0574 | +10.6 ms |
| 2048/512 | 32 | 0.0531 | 0.0644 | +11.3 ms |
| 1024/1024 | 1 | 0.0467 | 0.0573 | +10.6 ms |
| 1024/1024 | 32 | 0.0488 | 0.0586 | +9.8 ms |

The gap is **+9.5 to +13.7 ms across a 32× batch range and a 4× context range** —
a spread of 1.4× end to end. It is ~300–430 µs per layer per step and it does not
know how much data there is, which rules out every traffic explanation.

Corrected profile at 4096/B=32 — 48.05 ms/step of kernel, 1,656 launches:

| bucket | ms/step | share |
|---|---|---|
| elementwise | 14.20 | 29.6% |
| model GEMM | 11.04 | 23.0% |
| memory: copy/cat | 7.50 | 15.6% |
| **ours: two-tier decode** | **7.03** | 14.6% |
| memory: index/gather | 4.44 | 9.2% |
| reduction | 2.51 | 5.2% |
| ours: read gate | 0.81 | 1.7% |

There is no `model: attention` row — the two-tier kernel *is* the attention. The
cache's own launch budget is 80 ops per steady step and 493 on an eviction step,
~142/step amortized, ~8.6% of the total.

---

## 5. Running it

```bash
MODEL_PATH=/models/llama-3.1-8b-instruct scripts/run_perf_table.sh
python scripts/profile_decode.py --config <generated.yaml> --prefill 4096 --batch 32
scripts/run_longbench_ours_flash_attn.sh      # quality
```

Requires transformers 4.47.x (`utils/cache_factory.py` refuses newer),
flash-attn, and Triton on CUDA. `quant_budget_mode: tokens` is for latency tables
only; quote memory and quality against `bytes`.

Two lines the run prints are worth reading every time: `[StickyKV] eviction path:
{'eager': n, 'compiled': m}` — `eager > 0` means the run measured the path it was
trying to replace — and `[StickyKV] fused decode tiling: target_keys=…`, which
says whether the kernel got its top shared-memory rung.

---

## 6. Layout

```
modules/windowed_cache/   the cache: eviction, hooks, and the Triton kernels
  cache.py                update(), the layer-major eviction, the fused hand-off
  decode_kernel.py        the fused two-tier decode + gated Q loop + CPU oracle
  score_kernel.py         the fused prefill scorer
  gate_kernel.py          the card scan and the top-k
  flash_decode.py         the flash_attn_func seam; where the gate runs
  policy.py state.py scorer.py config.py
modules/quant/            the int2 tier: quantizer, slot table, cards
utils/                    config, cache factory, memory probe
scripts/                  perf table, decode profile, quality runners
```
