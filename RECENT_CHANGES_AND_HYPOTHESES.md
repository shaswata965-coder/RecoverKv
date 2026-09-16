# Recent changes and open hypotheses

Branch `int2_clustered_rep`, HEAD `d43040a`, report written 2026-09-16.

Window covered: **`7f6ee8c..d43040a` — 24 commits, 2026-09-13 → 2026-09-16,
41 files, +5,492 / −263.**

This document has three parts. §1–§5 are **what changed and what was measured**.
§6 is **the hypotheses** — every claim in this branch that is currently believed
rather than demonstrated, with the one measurement that would settle each. §7 is
**drift found while writing this**: four places where a document or a tool now
describes a configuration the code no longer has. §8 is the run order for a GPU
session.

The distinction matters because of this branch's recurring failure mode, which
appears four separate times in the window below: **a knob that is accepted,
reported, and inert.** `quant_gate_ratio` was rejected by the YAML loader and
dropped by five runners (`6b8a188`); it was then dropped again by the profiler's
cache builder (`57313dc`); `STICKYKV_COMPILE_EVICT` defaulted to 0 while the perf
table exported 1 (`1434b2c`, `676c781`); `STICKYKV_COMPILE_READ` printed a banner
naming a read path no CUDA run has taken since the fused kernel landed
(`0091e9c`). None of these produce an error. Each produces a number under the
wrong configuration that looks exactly like a number under the right one.

---

## 1. The headline feature — the rank-1 read gate

Eight commits, `3658f72 → e7bc158`. The int2 Q tier used to be dequantized
**whole, every decode step**. The gate makes the decode step read a configurable
fraction of it — default **0.25** — by scoring each window from a small card
written once at demotion, and dequantizing only the windows that rank.

### 1.1 The card (`3658f72`)

Each int2 window carries a rank-1 model `k_i ≈ mu + t_i · v`, plus a per-token
residual `eps_i`. Two properties carry the design:

* **`v` points at the token farthest from the mean**, so that token is the
  best-fit token in its window. In a sparse window that is the hot one — a
  mean-only card averages it away, which is the exact failure the gate exists to
  avoid.
* **`|q·k_i − (q·mu + t_i q·v)| ≤ ‖q‖ · eps_i` by Cauchy–Schwarz**, so the gate
  computes a **bound, not an estimate**: it can over-select, it cannot miss.
  `eps` is measured against the *decoded* card and quantized by rounding **up**,
  so the bound holds for the bytes actually stored.

Three encoding choices were settled by measurement and are pinned by tests so
they cannot be "optimised" back:

| choice | alternative | measured |
|---|---|---|
| `v` as **int8** | 1-bit sign vector | hot token's residual **38.0** against a 15.0 window mean — worse than useless |
| | int4 | 7.0 |
| | **int8** | **0.39** against a 9.2 mean |
| `mu` as a **residual from a per-(layer, head) anchor** frozen at first demotion | raw `mu` | reconstruction error 0.078 → **0.0046** (17×) — massive-activation channels are near-identical across windows, so they separate nothing and eat the whole int8 range |
| **no Hadamard rotation** | rotate before coding | 0.0046 → 0.0042, mass estimate identical to three decimals. Load-bearing only for the 1-bit variant, so it is gone: one fewer matmul on a launch-bound eviction path |

Cost: **280 B/head, 2,240 B/window** at `H_kv=8, D=128, ws=8` — **26.5% of
`b_q`**. Cards ride the existing `QuantSlotTable` as eight slot-major columns, so
write / gather / `retain_only` / `set_active` / `join_layers` are inherited
unchanged, and a card is frozen at first demotion, dropped with its window and
reactivated on re-demotion for free. It is built from the **rotated** keys
`cache.py` already holds one line before un-rotating them, so it costs no extra
RoPE.

### 1.2 Selection granularity — forced by measurement (`fc6c729`)

Selection is **per (row, KV head)**, and the alternatives were priced first. At a
0.25 ratio on a shape-C tier (271 windows, 8 KV heads, 32 query heads):

| scheme | windows read | worst-head mass recall |
|---|---|---|
| **per (row, KV head)** | **68 / 271** | **100.00%** |
| per row, union of picks | 240 / 271 | 100.00% |
| per row, top-k row-max | 68 / 271 | 97.54% |

The union reads 89% of the tier — the gate buys nothing. A row-level top-k pays
2.5% of the worst head's mass for identical traffic. So each KV head picks its
own windows; the cost is that Q positions become per-head
(`[B, H_kv, n_sel·ws]` instead of a shared `[B, n·ws]`). Counts stay equal across
rows and heads, so the result is still dense — no padding, no keep-mask — which
is the same rectangularity argument the tier split already relies on.

**The cap must bind on the KV head, not the query head.** The first version
capped per query head and ran the GQA union afterwards; at `rep=4`
(Llama-3.1-8B) a 25% cap could therefore load **up to 100%** of windows and the
gate would buy nothing. The union now runs first, on bounds.

### 1.3 The write-back that makes it safe (`fc6c729`)

`scorer.fill_skipped_window_scores` credits every skipped window with its card's
estimate, rescaled by `c = Σ(exact over selected) / Σ(estimated over selected)`.
The selected windows carry both a real and an estimated score, so the correction
is free and self-calibrating.

This is not an optimisation, it is a correctness requirement: **without it a
skipped window scores zero, ranks last, and is evicted.** At `quant_ratio=0.7`
that is 70% of the evictable cache destroyed by a change meant only to read it
less often. Measured on synthetic scores: rank correlation with truth **0.996**
and **99.5%** of total mass recovered, against **0.32** correlation if skipped
windows are left at zero.

### 1.4 End-to-end accuracy, on CPU (`9d131e4`)

`gated_decode_step` is the whole method in one function and the Triton path's
oracle. Six seeds at `n_q=64`:

| ratio | out rel err (median) | score corr | top window kept |
|---|---|---|---|
| 0.15 | 0.0081 | 0.9935 | 100% |
| **0.25** | **0.0020** | **0.9995** | **100%** |
| 0.50 | 0.0004 | 1.0000 | 100% |
| 1.00 | 0.0000 | 1.0000 | 100% |

`ratio=1.0` is byte-identical to attending over the whole tier — the no-op proof,
and the test that says this is a *selection* rather than a second implementation
of attention.

**There is a tail, and it is asserted rather than hidden:** p95 of the relative
output error is ~0.1 at ratio 0.25, max ~0.38, on rows whose mass is spread thin
enough that dropping 75% of the tier costs real output. The fixture is
pessimistic about that tail by construction — its fp tier is noise, where the
real one holds the sink, the local window and the top-ranked windows. In
production most of the mass sits **outside** the gated tier; in the fixture it
sits inside. **The real number has to be read off LongBench** (→ H-H).

### 1.5 Onto the production path (`7a2d404`), then made cheap (`71841d2`)

`_two_tier_decode_kernel` takes a `SEL` selection and dereferences the Q loop
through it. **The tier stays whole and gathered — the gate skips reads rather
than shrinking the store, so the bytes it declines to read are never moved.**
`GATED` is a `constexpr`, so the ungated kernel is byte-unchanged.

The gate had to live in `_run_fused` and not in `update()`: selecting windows
needs the **query**, and transformers' `Cache.update()` is handed keys and values
only. That is the shape of the API, not a choice.

It shipped correct and **~49% slower**: +23 ms/token at B=1, +30 ms at B=32,
near-identical across a 4× context range and a 32× batch range. *A cost that
ignores how much data there is is not a data cost.* Measured: **28 extra GPU
calls per layer per step, 896 per token at L=32**, on a path this repo's own
notes say is bound by launch count. Four causes, all fixed:

1. The epilogue's explicit normalisation collapses to a max and a sum — the
   offset cancels in the ratio, so no logarithm is taken and every exponential
   stays base 2.
2. Card loads sat **inside** the query-head loop, so a 4-head GQA group read each
   card 4×, defeating the one reason the group is grouped. Loads hoisted, anchor
   with them.
3. The kernel computed and stored `bound`, a full `[B, H_q, NW]` fp32 output that
   `fused_gate` discards, and loaded the whole `eps` field of every card solely
   to compute it. Both gone.
4. Occupancy: one program per (row, KV head) walking every window serially is **8
   programs at B=1 on a 108-SM A100**. The window axis is now split across
   programs (`gate_window_tiles`) — 136 blocks at B=1, still coarse at B=32 where
   rows already fill the machine.

**Net: 28 → 3 host calls per layer per step, 896 → 96 per token.**

### 1.6 Verified without a GPU — further than assumed

Every kernel in this window was validated on a CPU-only box, and the checks were
non-vacuous:

* `two_tier_window_reference` models the `SEL` indirection tile for tile.
  Identity `SEL` is byte-identical to the ungated path; a real subset matches a
  plain softmax over `[sink | body | selected]`, with a different pick per
  (row, KV head).
* `_run_fused` is driven end to end with the kernel replaced by that oracle and
  reproduces `gated_decode_step` through a non-identity `order`.
* Both Triton kernels are **compiled for an sm_80 target with no CUDA present**,
  which type-checks the front end. Confirmed non-vacuous against a deliberate
  type error.

That last check found a real bug: `_gate_kernel` indexed its token axis with
`tl.arange(0, WS)`, so **any non-power-of-2 `window_size` could not compile at
all**. The axis is now padded to `BLOCK_WS` and masked, with the padding forced
to `-inf` rather than zero — zero would have scored every window as if it held
one extra token sitting at its mean.

One fixture bug is worth recording because it produced convincing-looking
garbage: the first version demoted unrelated random `keys_pre_rope` while
building cards from `k_post`, so the gate selected on one tensor while attention
ran over another — score correlation 0.25, output error 2.2. Deriving
`keys_pre_rope` by un-rotating `k_post`, as `cache.py` does, took those to 0.9995
and 0.002. **An accuracy number from a fixture whose codes and cards disagree
measures nothing.**

---

## 2. Measurement correctness — Suite C

Four commits, `7f6ee8c / f3519e0 / fbcf368 / 981212a`. These change no kernel and
no cache; they change what the harness *reports*.

> **Numbers recorded before `fbcf368` carry the defects below and are not
> comparable to numbers recorded after it.**

| defect | effect |
|---|---|
| `throughput_tokps` divides by `gen_time + TTFT` and omits `prefill_to_decode_gap_ms` | neither the decode rate nor the e2e rate. On a 4096/257 B=32 cell: **230.1 quoted against an actual 345.9 tok/s (1.5×)**, with 11.7 s of prefill in the denominator. The error grows with prefill *and* with batch, so it distorts comparisons **between rows**, not just their scale |
| `_median` returned `np.mean` despite its name and the module docstring | at `num_measurement_runs: 2`, one throttled run moves a figure by half its excess. Synthetic cell `[81.3, 42.8, 42.8, 42.8]`: **52.43 before, 42.80 now** |
| every rate built on `tpot_ms` | `tpot_ms` averages in decode step 0, which is where this design compacts the whole prompt — so a per-token rate charges a one-off O(prefill) cost into every token, with a bias decaying as `1/gen_len`. **The same method then reports a different decode throughput purely as a function of how many tokens the cell asked for.** All columns now use `tpot_steady_ms` |
| `prefill_logits: full` | the `[B, L, V]` prefill logits tensor is ~1.05 GB per row in fp16 at prefill 4096 — **~67 GB at B=64**, method-independent, and large enough on its own to stop the max-B ladder. **Every max-B recorded with `full` is understated.** Now `last_only` |
| `num_measurement_runs: 2` in the batched suite | a median of two is a mean. Now 5, with `cooldown_s 3.0`, clock locking on, `allow_shared_gpu false` |

Three new fields — `prefill_to_decode_gap_ms`, `throughput_e2e_tokps`,
`throughput_decode_tokps` — sit beside `throughput_tokps`, which keeps its
formula bit-for-bit and is now documented as legacy. `print_perf_table.py`
(`981212a`) gained `--throughput both|decode|e2e|all|legacy`; `legacy` reproduces
the pre-change table byte-for-byte, verified by diffing both printers over one
npz dir, and exists so an old screenshot can be *identified* rather than guessed
at. A missing e2e figure prints `-` rather than borrowing `throughput_tokps` —
that substitution is the bug wearing a new label.

`f3519e0` collapsed the config + driver + printer trio into one self-contained
`scripts/perf_shapes.py` running exactly nine cells: 1024/1024
(generation-dominated), 2048/512 (balanced), 4096/256 (prompt-dominated) at
batch 1 / 32 / 64.

---

## 3. One production path

Six commits, `b8b5fa8 → 676c781` and `0091e9c`. The theme is provenance: **a
number produced under the wrong switch still looks like a number.**

| commit | what was wrong |
|---|---|
| `6b8a188` | `quant_gate_ratio` was written into the generated config and verified against `modules.windowed_cache.config.WindowedCacheConfig` — but the YAML loader uses `utils.config.CacheConfig`, which had no such field. **Every run died before the model loaded.** Worse, none of the five runners that build a cache config passed it on, so once the loader accepted it, `--gate-ratio 1.0` would have reported an "ungated control arm" that was fully gated |
| `57313dc` | `resolve_cache_kwargs`, which the profiler and `audit_e2e` build their cache from, dropped it too — its own docstring calls it "the single place where the ladder could silently diverge from the benchmarked method" |
| `1434b2c` | `profile_decode` left `STICKYKV_COMPILE_EVICT` unset; the library default was 0 while `run_perf_table.sh` exported 1, so **the profile ran the eviction eager against a table that ran it compiled** — and compiling the eviction is worth TPOT 101.6 → 85 ms, landing in exactly the elementwise/copy/cat rows a profile is read to interpret |
| `676c781` | `STICKYKV_FUSED_DECODE` is **gone**. `fused_decode_enabled()` returns `True` and reads no environment; the score hook's `enabled and cuda` decides, so **CUDA runs the fused gated kernel and CPU runs `_materialize`**. `STICKYKV_COMPILE_EVICT` now lets the device decide (CUDA compiles, CPU does not — compiling a launch-coalescing pass on CPU cost the suite 4× its runtime). An explicit value still wins, for the documented compile-failure bisection |
| `0091e9c` | `STICKYKV_COMPILE_READ` selected nothing — on CUDA `dequant_rotate_q_keys` is unreachable by two independent routes — but printed a banner naming a "decode read path" the run was not taking, **in every profile and perf log in this repo** |
| `e7bc158` | `quant_gate_ratio = 1.0` returned `gate=None` and skipped the gate entirely, so the shipped config and the control arm ran *different code*. Now every ratio takes the same route and 1.0 selects every window — the no-op proof becomes a property of one implementation rather than evidence about a second |

`_materialize` **stays**. It is not a fallback; it is the CPU oracle 13 test
files check the kernel against, and without it nothing here is verifiable off a
GPU. It is simply not reachable as a production alternative. The varlen case that
used to justify the switch still fails loudly (`FusedDecodeNotReached`) rather
than silently running different code.

`tests/test_default_operating_point.py` pins the result: a no-flag run resolves
to **budget 0.20, window 8, gate 0.25, quant_ratio 0.70, local window 128** on
the fused gated kernel, and **the profiler resolves the SAME cache as the
table**. The values are read out of `run_perf_table.sh` rather than restated, so
the script and the tools cannot drift apart again.

Observability added alongside (`9ce4519`, `b8b5fa8`): `stats()` now reports
`gated` beside `armed`/`fired`, plus `read_fraction` — windows actually
dequantized over windows available, i.e. `quant_gate_ratio` **as realised**
rather than as configured. Both come from values already in hand, so this is
Python integer arithmetic: no launch, no `.item()`, no device sync on the decode
path. **First thing to read on the GPU box: `gated == fired` says the gate served
every step; `read_fraction ≈ 0.25` says it selected what it was asked to.**

---

## 4. Eviction numerics under Inductor (`5186353`)

The largest single correctness find in the window, and it was found while chasing
speed.

`_affine_quantize` carried `@torch.compiler.disable` to dodge an Inductor ≤ 2.6
lowering failure (`aten.amin` behind a data-dependent gather). That bug is fixed
upstream, and the break was costing far more than it was scored at: **the
eviction step is ~195 ms against a ~72 ms steady step, ~15.4 ms/step amortized,
18% of GPU time**, landing as sixteen separate multi-millisecond elementwise
kernels — round, clamp, div, sub, where, scatter. That is the signature of a
graph that traced and then fused nothing. Undecorated, the function traces with
**0 graph breaks in 1 graph**.

**Landing that alone would have corrupted the cache.** `DECODE_SPEED_PLAN.md` §3
said the change was "byte-identical and the `aot_eager` test already pins that".
That was wrong: `aot_eager` runs eager kernels, so it can never observe a codegen
difference.

Inductor does not emulate intermediate precision casts by default — it drops an
`fp32 → fp16 → fp32` round trip and keeps the wider value in a register. Three
quantisers in the eviction graph take that round trip **on purpose**, to fit
codes to the fp16-stored grid so the fit grid is bit-identical to the grid every
later dequant reads (design §2). Measured on torch 2.14, CPU Inductor, 24 seeds:

| | diverges from eager |
|---|---|
| `quantize_key_windows` | 11 / 24 seeds |
| `quantize_value_windows` | 14 / 24 seeds |
| `sketch._q_sym` | 24 / 24 seeds |
| `sketch._q_up` | 24 / 24 seeds, **and 115 violations of `dequant ≥ x`** |

**That last row is the serious one and is not caused by this change.** The sketch
helpers were never behind a graph break, so **every compiled eviction since the
cards landed has built them on the elided grid**. `_q_up` rounds up precisely so
`dequant ≥ x` holds, which is the only reason the card's Cauchy–Schwarz score is
an *upper bound* — and therefore the only reason the read gate may skip a window
on a low score at all (§1.1). Its `(1 + 2⁻⁹)` nudge is sized for an fp16 grid;
against an un-narrowed one it is sized for the wrong grid.

Fix: one wrapper at the single compile site, `_emulating_precision_casts`, which
patches `torch._inductor.config.emulate_precision_casts` around the compiled
eviction — entered per call, restored on exit, so nothing else the host compiles
is affected. All four rows go to zero, and a 14-step / 5-eviction end-to-end run
compiled vs eager is **identical tensor for tensor**: int2 codes, scales, zeros,
slot table and all eight card fields. Without the wrapper that same run already
differed (2 elements of `sk_t_q`) on a deliberately tiny fixture.

`_quant_may_be_traced()` now requires **both** torch ≥ 2.7 **and** the presence
of `emulate_precision_casts`. A build with no such knob keeps the graph break,
because the alternative is silently writing a different cache under
`torch.compile` than without it.

Suite after: **1015 passed, 12 failed** (the same CUDA-only failures as before),
58 s. `tests/test_compiled_eviction_numerics.py` pins it; 4 of its 5 tests fail
if the wrapper is reduced to the identity.

---

## 5. The profile can now name its own tail (`d43040a`)

`profile_decode.py` did no categorisation: it printed the top-N kernels by CUDA
self time. Reading that means subtracting the rows you recognise and calling the
remainder a tail — which is how **a "10.4 ms unattributed tail" got quoted in
this repo that was never a block at all**, just every kernel ranked below the
cut.

The rollup answers what the flat list cannot — how much of the step is the cache
and how much is the model:

* **`ours (cache)`** — two-tier decode / read gate / prefill score / compiled
  evict (Inductor's `triton_*_fused_*` names)
* **`model`** — attention / GEMM / norm+softmax / activation
* **`shared/other`** — copy+cat, index+gather, sort+topk, elementwise, reduction

`shared/other` is reported as attributed to **neither** side rather than guessed:
both the cache and the model issue those kernels and a kernel name cannot say
which. Splitting it needs `--trace`, which carries the launching stack. The rule
the table is built around: **`other` is always printed by name, with a note to
add a pattern rather than subtract.** A rollup that hides its own misses is worse
than no rollup, because it looks authoritative. There is also a reconciliation
check against measured CUDA self time.

Classification is a name heuristic, so it is unit-tested against the names an
A100 actually emits. That test immediately caught a real bug: **flash-attn
kernels carry `cutlass::half_t` in their template arguments, so a
GEMM-before-attention order filed every attention kernel as a GEMM.** Attention
is now matched first, three more two-bucket names are pinned by order, and a test
asserts every bucket's own first pattern reaches it rather than being shadowed.

Related measurement from `0091e9c`: the cache issues **80 launching ops per
steady step** (64 of them the KV append) and **493 on an eviction step**, against
**~3,384 per step for the whole model**.

---

## 6. The hypotheses

Everything below is currently **believed, not demonstrated**, on this branch.
Each row names the single measurement that settles it.

### 6.1 Accuracy family

Origin: `ACCURACY_RECOVERY_PLAN.md` §2–§5, the twelve-dataset LongBench
regression (`qasper` −9.52, `multifieldqa_en` −4.85, every dataset down, none
spared).

---

**H-A — the regression was `quant_budget_mode` silently flipping `bytes` →
`tokens`.**

*Status: fix landed, attribution never confirmed.*

Supporting: `f71fec0` changed the default; no YAML set it; the two modes differ
by 3.88× per-window price, so at `q=0.5` the run attends over **2.2× fewer keys
at the same nominal budget**. Measured `budget_utilisation` under `bytes` is 0.99
at q = 0.0 / 0.5 / 0.7; under `tokens` it is 0.99 / 0.69 / 0.57 — i.e. tokens
mode **grants a byte budget and then declines to spend 37–52% of it**. The
uniform, monotone, context-sensitivity-ordered shape of the §1 table is the
signature of the cache getting smaller, not of a kernel bug (which is lumpy).

Against: the recovery run bundles more than one change. The post-fix numbers
(`qasper` 42.43, `multifieldqa_en` 55.86, `gov_report` 33.72, `trec` 71.50 — 8 of
11 at or above baseline) landed **together with the RoPE-dtype fix**, so this is
not a controlled attribution. *"The number came back" is a coincidence that has
not been ruled out.*

**Settles it:** rung 1 of §5 — `cache.quant_budget_mode: tokens` at HEAD on
`qasper`, one dataset, must **reproduce 32.78**. Rung 0 recovering without rung 1
reproducing is not a confirmation.

---

**H-B — the fused two-tier decode kernel is numerically equivalent to its
reference, on a GPU.**

*Status: **the experiment that was to settle this no longer exists.** See §7.2 —
this is the most consequential finding in this report.*

The kernel produces the attention output the next token is sampled from, and it
has never been checked against its reference on a GPU. The check was
`audit_e2e.py` rungs `2_q70_materialize` vs `3_q70_fused` — greedy decode is
deterministic, so the same method by two routes must emit an identical token
sequence.

Those two rungs differ **only** by `STICKYKV_FUSED_DECODE`, which `676c781`
deleted. `fused_decode_enabled()` now returns `True` unconditionally, so on CUDA
**both rungs run the fused kernel** and the equivalence check prints
`OK — identical for all N decode tokens` while comparing a path against itself.
The acceptance gate in `ACCURACY_RECOVERY_PLAN.md` §7 ("Decode parity:
`audit_e2e.py` rungs 2 and 3 emit identical token sequences") is now **vacuously
true**.

**Settles it:** rung 2 must be rebuilt to force the materialize path by a route
that still exists — a config/constructor argument, not an environment variable —
or the gate must be replaced by a direct kernel-vs-`_materialize` comparison on
device. **Until then the branch has no GPU validation of its only decode path,
and no test that would notice.**

---

**H-C — FlashInfer's LSE is natural-log, not log2.**

*Status: routed around, never verified.*

`STICKYKV_FLASHINFER_LSE_LOG2` defaults to 0 while the module's own docstring
warns some builds historically returned log2. If the installed build does, every
`L` is off by `ln 2`, every `exp(S − L)` is wrong, and **every eviction decision
in the run is made on corrupted scores** — no error, no warning, plausible
output. `f8c442d` made `auto` prefer `flash`, which sidesteps it; `flash_lse` is
also strictly safer, because it keeps the `softmax_lse` the same
`flash_attn_func` already computed and leaves the attention output
**bit-identical**, whereas FlashInfer *replaces the attention kernel*.

**Settles it:** `STICKYKV_LSE_BACKEND=flashinfer python -m pytest
tests/test_flashinfer_lse.py -v` on the GPU box. Minutes. A factor-`ln 2`
mismatch **is** cause C.

---

**H-D — the Triton prefill score kernel and `exp2` are score-neutral.**

*Status: reads correct, never run on a GPU.*

The `exp2` change reads correct by inspection: `scaling` carries `log2 e` from
the dispatcher, the kernel applies it to the `[BLOCK_M]` LSE vector and not to
the tile, and `exp(s·scale − L) ≡ exp2(s·scale·log2e − L·log2e)` holds.
`ex2.approx` costs ~2 ulp into a sum that feeds a ranking — **the least likely
cause in the document.** But the kernel as a whole ships GPU-unvalidated and now
carries two autotune configs (`BLOCK_M=32/64, BLOCK_N=256`) that no correctness
run has exercised.

**Settles it:** `pytest tests/test_score_kernel.py tests/test_score_kernel_ab.py
tests/test_score_exp2.py -v` on the GPU box — `triton == token_scores_from_lse ==
token_scores_torch`. These skip the kernel on CPU, so this would be **the first
real validation the prefill kernel has had.** `STICKYKV_SCORE_EXP2=0` and
`STICKYKV_SCORE_AUTOTUNE=0` both still exist as control arms.

---

**H-H — the read gate at 0.25 does not cost LongBench accuracy.**

*Status: new with this branch, entirely unmeasured on a real model.*

Supporting: median rel err 0.0020 and score corr 0.9995 at ratio 0.25 over six
seeds; top window kept 100% at every ratio; the selection is a Cauchy–Schwarz
**bound**, so it over-selects rather than missing; skipped windows are credited
with a rescaled estimate at rank correlation 0.996.

Against, and stated by its own author: **p95 rel err ~0.1 and max ~0.38** at
ratio 0.25. The fixture is pessimistic by construction — its fp tier is noise
where production holds the sink, local window and top-ranked windows — but
"pessimistic by construction" is an argument, not a measurement.

There is also an **interaction with H-B and §4**: the gate's right to skip a
window rests on `_q_up`'s `dequant ≥ x` guarantee, which was **violated 115 times
under compiled eviction** until `5186353`. Any gate accuracy number taken from a
compiled-eviction run before that commit is measuring a gate whose bound did not
hold.

**Settles it:** LongBench at HEAD, `GATE_RATIO=0.25` against `GATE_RATIO=1.0`
(the true ungated control since `e7bc158`), same budget, same shapes.

---

### 6.2 Speed family

Origin: `DECODE_SPEED_PLAN.md`. Both stated decode targets are met — 0.0626 at
4096/256 B=32 against FullKV's 0.086, **1.37× faster**, from a starting point of
0.1760 (2.05× *slower*). The governing constraint from here is that **scores must
not move**: an optimisation must be score-neutral *by construction* — shape,
launch count or scheduling only.

---

**H-E — the open ~5 ms/step regression is the base-2 softmax, or the RoPE dtype,
or neither.**

*Status: measured, unexplained, control arms latched and unrun. The single most
important open speed item.*

| cell | 0.0626-era | now | delta |
|---|---|---|---|
| 4096/256 B=32 | 0.0626 | 0.0669 | +4.3 ms |
| 2048/512 B=32 | 0.0531 | 0.0575 | +4.4 ms |
| 1024/1024 B=32 | 0.0488 | 0.0537 | +4.9 ms |
| 4096/256 B=1 | 0.0471 | 0.0516 | +4.5 ms |
| 2048/512 B=1 | 0.0468 | 0.0516 | +4.8 ms |
| 1024/1024 B=1 | 0.0467 | 0.0520 | +5.3 ms |

**+4.3 to +5.3 ms, flat across a 32× batch range and a 4× context range** — a
fixed per-step cost (~156 µs/layer), not traffic, and not the Q-tier loop: it
does not track `n_active` (179 / 89 / 44), which it would if the tile rung had
changed. Only two code changes are in scope and **neither has a mechanism that
predicts a constant of that size**, which is why it is recorded as unattributed
rather than explained away.

**Settles it:** one run each.
`STICKYKV_DECODE_EXP2=0 scripts/run_perf_table.sh` isolates (a);
`STICKYKV_ROPE_STORE_DTYPE=0 scripts/run_perf_table.sh` isolates (b).

* If **(a)** — drop it. Worth ~0.42 ms in theory, buying nothing.
* If **(b)** — **keep it anyway.** It is the accuracy fix; a ~5 ms/step cost for
  closing a 9.5-point `qasper` regression is a trade worth making. Record it as
  the price, and note 0.0669 still beats FullKV by 1.29× and still clears the 20%
  target (≤ 0.0688).
* If **neither** — the cause is outside these two changes, and Stage 0's
  `GPU busy %` becomes the next instrument.

---

**H-F — the shared-memory tile rung is already at `target_keys=64`.**

*Status: read-only, two minutes, never run.* This is the entire remaining
score-free speed budget.

| rung | Q-tier iterations at `ws=8` | vs the pre-tiling 179 |
|---|---|---|
| `target_keys=64` | 23 | 7.8× |
| `target_keys=32` | 45 | 4.0× |
| `target_keys=16` | 90 | 2.0× |

Every rung is numerically identical — only the serial iteration count changes.

**Settles it:** the line the kernel already prints,
`[StickyKV] fused decode tiling: target_keys=… num_stages=…`. If 64, the item is
banked and closed. If 32, a few ms at best — worth one fp16 `vv` attempt, then
stop. If 16, most of the tiling's benefit is unrealised and the separate-`BLOCK_T`
change is worth doing.

---

**H-G — the read gate pays for itself at B=32.**

*Status: unmeasured after the 28 → 3 launch cut, and its author flagged the
ceiling as the open question.*

At B=1 the **baseline was already flat across context** (0.0471 / 0.0468 /
0.0467), so the KV read was never the bottleneck there and **reading less of it
cannot help** — the gate can only lose at B=1. The only cell with real context
sensitivity is B=32, where **~14 ms of the step moves with context. That is the
ceiling, and it is what the next run has to beat.**

**Settles it:** `GATE_RATIO=0.25` against `GATE_RATIO=1.0` at identical budget,
shape and batch — the same cache, gated vs not, which since `e7bc158` is one code
path at two values rather than two code paths. Read `gated`/`read_fraction` from
`stats()` first to confirm the gate actually fired.

---

**H-J — the decode step is host-bound rather than kernel-bound.**

*Status: the assumption behind every launch-count optimisation in this repo, and
explicitly unverified.*

The flat-TPOT signature the decision rested on — "flat across a 32× change in
batch … only happens when the GPU is idle waiting on the host" — **does not
identify a host bottleneck.** A single kernel whose critical path is
batch-invariant (e.g. one on a `B·H_kv` grid, which is 8 blocks on a 108-SM A100
at B=1) produces the identical signature. Note that H-E's regression has exactly
this shape too.

**Settles it:** `GPU busy % = CUDA kernel self time / wall time`, printed first by
`profile_decode.py`. **`≪ 1.0`** → host-bound, the dispatch-count argument was
right, remaining launches are the target. **`≈ 1.0`** → kernel-bound, no amount of
launch collapsing can help and the kernel itself is the target.

This is also the number that makes future proposals checkable at all: *every ms
estimate this plan ever made was a byte or launch count with an unverified
translation to time, and one of them was wrong by 20×* (the window-score rewrite:
sized at ~0.5 ms, delivered ~11 ms, because the value was in launches and serial
latency, which a traffic count cannot see).

---

**H-I — `emulate_precision_casts` is free on a GPU.**

*Status: stated as expected, explicitly unmeasured.* The truncations it
reinstates are pointwise and fuse into the same kernels, so it *should* be
instructions, not launches. It sits on the eviction path, which is ~15.4 ms/step
amortized — so "should" is worth one measurement.

**Settles it:** the `ours (cache)` bucket in the new profile rollup, with and
without the wrapper.

---

### 6.3 Hypothesis status at a glance

| id | claim | evidence | settles it | cost |
|---|---|---|---|---|
| **H-A** | `quant_budget_mode` caused the LongBench regression | strong, uncontrolled | rung 1 reproduces 32.78 on `qasper` | 1 dataset |
| **H-B** | fused decode ≡ its reference on GPU | **none — the A/B was deleted** | rebuild rung 2 without the dead env var | code + 1 run |
| **H-C** | FlashInfer LSE is natural-log | none | `pytest tests/test_flashinfer_lse.py` on GPU | minutes |
| **H-D** | score kernel + `exp2` are score-neutral | reads correct | 3 score-kernel test files on GPU | minutes |
| **H-E** | the +5 ms is `exp2` or RoPE dtype | neither mechanism predicts it | two latched perf-table runs | 2 runs |
| **H-F** | tile rung is at 64 | none | read one printed line | 2 min |
| **H-G** | the gate pays at B=32 | ~14 ms is the whole ceiling | `GATE_RATIO` 0.25 vs 1.0 | 2 runs |
| **H-H** | gate at 0.25 is accuracy-neutral | CPU fixture, p95 tail ~0.1 | LongBench 0.25 vs 1.0 | 1 suite |
| **H-I** | `emulate_precision_casts` is free | argued | profile bucket A/B | 1 profile |
| **H-J** | decode is host-bound | signature is ambiguous | `GPU busy %` | 2 min |

---

## 7. Drift found while writing this report

Four places where a document or tool describes a configuration the code no longer
has. None of these are hypothetical; each is verifiable at HEAD.

### 7.1 `run_perf_table.sh` header contradicts its own defaults

`ecc0792` changed the values without the header comments:

| knob | header says | code does |
|---|---|---|
| `CACHE_BUDGET` | `default: 0.50` (line 34) | `0.20` (line 115) |
| `LOCAL_WINDOW` | `default: 64` (line 40) | `128` (line 121) |

`tests/test_default_operating_point.py` reads the **values**, so the pinned
operating point (0.20 / 128) is correct and the `--help` text is what is wrong.

### 7.2 `audit_e2e.py` rungs 2/3 and `profile_decode.py --fused` are inert

Both still set `STICKYKV_FUSED_DECODE`, which `676c781` made unread.

* `audit_e2e.py` rungs `2_q70_materialize` / `3_q70_fused` now run **the same
  path**, and `_report_equivalence` prints `OK — identical for all N decode
  tokens` for a comparison that cannot fail. This is the acceptance gate for
  **H-B**, so H-B currently has a green check that proves nothing.
* `profile_decode.py --fused 0` sets the same dead variable and then prints
  `fused=0` in the profile header — **a label naming a path the run did not
  take**, which is precisely the bug `0091e9c` deleted from the read path one
  commit earlier.

Several error messages also still advise `set STICKYKV_FUSED_DECODE=0 to fall
back` (`decode_kernel.py:181`, `gate_kernel.py:300`, `flash_decode.py:158`) —
advice that now does nothing.

### 7.3 The headline tables are quoted at a budget the branch no longer runs

`DECODE_HISTORY.md:12` and `DECODE_SPEED_PLAN.md:6` both label the result table
*"budget 0.50, q=0.70"*. The shipped default is now **0.20**. The numbers are not
wrong, but they are **not at the shipped operating point**, and nothing in either
document says so — so a reader comparing a fresh `run_perf_table.sh` against
0.0626 is comparing two different caches.

### 7.4 `ACCURACY_RECOVERY_PLAN.md` §3 "Ruled out" is stale

It rules out compile-related causes on the grounds that *"`STICKYKV_COMPILE_EVICT`
defaults `0` … and `STICKYKV_COMPILE_READ` defaults `0` — the compiled-eviction
work is not in a default run."* At HEAD, **CUDA compiles the eviction by
default** (`676c781`) and `STICKYKV_COMPILE_READ` does not exist (`0091e9c`). The
compiled eviction *is* in a default run, and §4 of this report shows it was
building sketch cards on an elided fp16 grid until `5186353`. **That is a new
candidate cause for any accuracy number taken on CUDA between the cards landing
and `5186353`.**

One smaller note: `gate_and_select` has no production caller — only
`gated_decode_step`, the CPU reference — so a finite `quant_gate_margin` is
honoured nowhere. `bcc68e9` corrected the error message to say so, and
`cache.py:1798` raises rather than accepting it silently on the fused path. This
is recorded, not a defect.

---

## 8. Run order for one GPU session

Ordered by information per minute. Items 1–4 cost minutes and can each
invalidate a whole branch of §6.

1. **`profile_decode.py` at 4096 / B=32** — read `GPU busy %` (**H-J**), the tile
   rung line (**H-F**), the `ours (cache)` vs `model` split, and
   `gated`/`read_fraction` (**H-G** precondition). Two minutes, four hypotheses
   touched.
2. **`pytest tests/test_score_kernel.py tests/test_score_kernel_ab.py
   tests/test_score_exp2.py`** — first GPU validation the prefill kernel has had
   (**H-D**).
3. **`STICKYKV_LSE_BACKEND=flashinfer pytest tests/test_flashinfer_lse.py`** —
   settles **H-C** outright.
4. **Fix §7.2 first, then run `audit_e2e.py`** — without the fix this run
   produces a false pass on **H-B**. This is the item to do before any accuracy
   number is quoted from this branch.
5. **Two latched perf-table runs** — `STICKYKV_DECODE_EXP2=0` and
   `STICKYKV_ROPE_STORE_DTYPE=0` (**H-E**).
6. **`GATE_RATIO` 0.25 vs 1.0 on the perf table** (**H-G**), then on LongBench
   (**H-H**).
7. **Rung 1 on `qasper`** — `quant_budget_mode: tokens` must reproduce 32.78
   (**H-A**).

**Where the branch stands if none of it is run:** both decode targets are met,
the LongBench regression is closed, the eviction path is score-neutral by
construction and numerically equal compiled vs eager, and there is one production
decode path that no environment variable can redirect. What is missing is that
**the only decode path on CUDA has never been validated on CUDA** — and as of
`676c781` the test that was supposed to do it silently cannot.
