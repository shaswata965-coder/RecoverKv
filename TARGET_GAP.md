# Reaching the published QEvict numbers — the gap, its shape, and what closes it

Written against the 2026-09-16 table (`GATE_REGRESSION.md` §8.1, gate 0.25 arm)
and the published QEvict row set. It answers one question: **what has to change
in this repo for a fresh `run_perf_table.sh` to print the published numbers
again.**

It is a companion to `DECODE_NEXT.md`, not a replacement. Where it disagrees
with that document it says so and shows the arithmetic. The disagreement is
about *scope*: `DECODE_NEXT.md` §4 sizes its envelope at one cell (4096 / B=32),
and three of the six target cells are B=1, where the largest item on its list is
worth approximately nothing.

Every number below is arithmetic on a measurement already in this repo. **No new
measurement was taken** — the dev box is CPU-only (`DECODE_NEXT.md` §8). Claims
that rest on inference are labelled.

---

## 1. The gap, per cell

Target = `DECODE_HISTORY.md` §1, the "total" column — the published table, taken
at **budget 0.50 / local 64**. Current = the 2026-09-16 run at **budget 0.20 /
local 128**. TPOT_steady, seconds.

| shape | B | target | now | gap (ms) | gap | target dec tok/s | now | needed |
|---|---|---|---|---|---|---|---|---|
| 4096/256 | 1 | 0.0471 | 0.0566 | **+9.5** | +20.2% | 21.2 | 17.7 | +20.0% |
| 4096/256 | 32 | 0.0626 | 0.0763 | **+13.7** | +21.9% | 511.2 | 419.6 | +21.8% |
| 2048/512 | 1 | 0.0468 | 0.0574 | **+10.6** | +22.6% | 21.4 | 17.4 | +22.8% |
| 2048/512 | 32 | 0.0531 | 0.0644 | **+11.3** | +21.3% | 602.6 | 497.0 | +21.3% |
| 1024/1024 | 1 | 0.0467 | 0.0573 | **+10.6** | +22.7% | 21.4 | 17.5 | +22.4% |
| 1024/1024 | 32 | 0.0488 | 0.0586 | **+9.8** | +20.1% | 655.7 | 546.0 | +20.1% |

e2e tok/s at B=32 (`GATE_REGRESSION.md` §1): target 294.3 / 497.1 / 619.7 against
259.3 / 421.6 / 519.6 — the same gap, +13% to +19%, as e2e dilutes TPOT with
prefill.

**The memory columns are already at target and are not part of this.** The
published run peaked at 48.44 GB at 4096/B=32 (`DECODE_HISTORY.md` §1); the
current table prints 48.70. `GATE_REGRESSION.md` §8.1 measured `peak_GB`
identical across the gate arms in all six cells. Nothing below is a memory
change, and nothing below should move those columns.

---

## 2. The signature: this is a fixed per-step cost

The gap is **+9.5 to +13.7 ms**, and its spread is the whole finding:

| | B=1 | B=32 | ratio |
|---|---|---|---|
| work per step | 1× | 32× | **32×** |
| gap | 9.5–10.6 ms | 9.8–13.7 ms | **1.13×** |

Across a 32× batch range and a 4× context range the gap moves by 1.4× end to
end. It is **~300–430 µs per layer per step** and it does not know how much data
there is.

`RECENT_CHANGES_AND_HYPOTHESES.md` H-E states the rule this repo has already
paid to learn: *"A cost that ignores how much data there is is not a data
cost."* Applied here it rules out, by construction and without further
measurement:

- **anything sized in bytes.** Already known from a different direction —
  `GATE_REGRESSION.md` §8.6 bounds the entire KV read at ~1% of TPOT — but the
  batch invariance is the stronger argument, because it does not depend on a
  roofline estimate. At B=1 the step moves 1/32 the KV bytes and carries the
  same +10 ms.
- **anything sized in tile iterations that tracks `n_active`.** `n_active` is
  179 / 89 / 44 across the three shapes (`DECODE_SPEED_PLAN.md`); the gap is
  9.5 / 10.6 / 10.6 at B=1. A tile-rung or Q-loop-length story predicts a 4×
  spread across those columns. There is none.
- **anything whose cost scales with rows**, which is most of the eviction — see
  §5, D1.

What is left is per-layer fixed work: kernel launches, host-side Python between
them, and any kernel whose critical path is invariant in both B and context.
`DECODE_NEXT.md` §2 measures 28.25 ms of host gap over 1,656 launches at ~17 µs
exposed each, so ~600 extra exposed launches per step, or ~18 per layer, would
account for the whole gap on its own. That is the size of thing being looked
for.

---

## 3. The gap splits into two blocks, and the plan targets only one

`RECENT_CHANGES_AND_HYPOTHESES.md` H-E records an intermediate table, taken
after the `exp2` + RoPE-dtype changes and **before** the gate landed on the
fused path (`7a2d404`, 2026-09-14), and therefore before `ecc0792` moved the
operating point (2026-09-15). Splicing it in:

| cell | published | H-E era | now | H-E block | gate era | H-E share |
|---|---|---|---|---|---|---|
| 4096/256 B=32 | 0.0626 | 0.0669 | 0.0763 | +4.3 | +9.4 | 31% |
| 2048/512 B=32 | 0.0531 | 0.0575 | 0.0644 | +4.4 | +6.9 | 39% |
| 1024/1024 B=32 | 0.0488 | 0.0537 | 0.0586 | +4.9 | +4.9 | 50% |
| 4096/256 B=1 | 0.0471 | 0.0516 | 0.0566 | +4.5 | +5.0 | 47% |
| 2048/512 B=1 | 0.0468 | 0.0516 | 0.0574 | +4.8 | +5.8 | 45% |
| 1024/1024 B=1 | 0.0467 | 0.0520 | 0.0573 | +5.3 | +5.3 | 50% |

**Block A — H-E, 4.3–5.3 ms, 43% of the gap.** Dead flat: 1.23× spread over 32×
batch and 4× context. Two changes landed together, neither with a mechanism that
predicts a constant of that size. Both control arms exist and are latched:
`STICKYKV_DECODE_EXP2=0` and `STICKYKV_ROPE_STORE_DTYPE=0`, wired into
`scripts/bisect_decode_regression.sh`, which runs all three arms plus a profile
in one command. **It has never been run.**

**Block B — the gate era, 4.9–9.4 ms, 57% of the gap.** Grows with batch and
context, so part of it is a real data cost. It is a **lower bound**: it is
measured across `ecc0792`, which cut the budget 0.50 → 0.20 and so cut
`N_q` 184 → 64 and `S_fp` 701 → 357 (`DECODE_NEXT.md` §3.3). The current table
does less work than the published one and is still 22% slower.

**Why this reordering matters.** `DECODE_NEXT.md` §4 lists six directions, D1–D6.
Every one of them targets Block B. Block A is 43% of the gap and appears in
`DECODE_NEXT.md` only in §4c, cut, with the note *"Measurement arms, never
optimizations."*

That cut is correct as a *scope* rule and wrong as a *plan*. H-E's own decision
rule is already pre-committed in the bisect script:

- **`exp2` owns it** → drop it. Worth ~0.42 ms in theory, buying nothing. **4.3–5.3
  ms recovered for free.**
- **RoPE dtype owns it** → keep it. It is the fix that closed a 9.5-point `qasper`
  regression, and scores outrank 5 ms. **4.3–5.3 ms is then permanently forfeit,
  and the published numbers become unreachable** — the floor moves to
  0.0669 / 0.0575 / 0.0537 at B=32 and 0.0516 / 0.0516 / 0.0520 at B=1, i.e. ~8%
  above target, for ever, whatever happens to Block B.
- **neither** → the cause is a third thing, and the profile the same script
  prints is the next instrument.

**So one 2-run script decides whether the target is reachable at all.** Nothing
else on any list in this repo has that property. It is the first thing to run.

---

## 4. The instrument that does not exist: there is no ungated arm

`GATE_REGRESSION.md` §8.1 ran gate 0.25 against gate 1.0, found a 0.2–1.6% tie,
and concluded *"the machinery costs essentially everything the selection
saves."* `DECODE_NEXT.md` §3.1 promotes that to settled — *"do not
re-litigate"* — and §4c cuts removing the gate.

**The conclusion is an inference, and the measurement that would support it
cannot be taken today.** `e7bc158` deliberately made every ratio take the same
route, so `GATE_RATIO=1.0` still runs:

- the `fused_gate` launch and its `topk` + `sort` (`gate_kernel._sorted_pick`),
- the `SEL` indirection in the Q loop, with `KS`/`KZ`/`KC`/`VC`/`VS`/`VZ` and
  `COS`/`SIN` all gathered rather than affine (`decode_kernel.py` §3),
- the `GATED` prologue over every Q column (`decode_kernel.py:759–770`),
- the `GATED` §5 fill pass over every Q column (`decode_kernel.py:904–930`),
- `build_sketch` on every eviction, `recon` and all.

1.0 is a control for **selectivity**. It is not a control for **the machinery**,
and the machinery is what §2b, §2c and D1–D4 all accuse.

The kernel's ungated variant is fully implemented and reachable — `sel=None`
gives `gated=False` and `GATED` is a `constexpr`, so the indirection, the
prologue and the §5 pass compile out entirely (`decode_kernel.py:1091–1093`,
and its own docstring at :652). The only thing standing between the harness and
it is one derived line:

```python
# modules/windowed_cache/config.py:615-617
# Derived, not configured: the gate IS the read path wherever there
# is a Q tier, and there is no way to ask for the ungated one.
quant_sketch_enabled=q > 0.0,
```

With `quant_sketch_enabled=False` the store builds no cards (`store.py:282`),
`_gate_ctx` returns `None` (`cache.py:1785`), the kernel takes the ungated path,
and every window gets a true score instead of a `fill_skipped_window_scores`
estimate. **The published 0.0626 was produced by exactly that configuration**,
at twice the read traffic.

**Recommendation: add the arm, as an arm.** A tri-state override on
`WindowedCacheConfig` (and its twin on `utils.config.CacheConfig` — `6b8a188`
is the standing lesson that the YAML loader is a separate class), defaulting to
the current derivation so the shipped operating point is unchanged and
`tests/test_default_operating_point.py` still passes. This is squarely inside
`DECODE_NEXT.md` §4c's own rules, which keep measurement arms while cutting
production paths: it is the same status `STICKYKV_DECODE_EXP2=0` already has.

It prices Block B in one run. Without it, Block B — 57% of the gap — has no
instrument at all, and D1–D6 are being sized against a cost nobody has measured.

**Two things it is not.** It is not a proposal to ship ungated: that decision
needs H-H (LongBench at 0.25 vs 1.0), which is also unrun. And it is not
score-neutral — ungated scores every window truly where the gate credits skipped
windows with a rescaled estimate — so it changes eviction decisions, in the
direction of *more* accuracy, not less. That is why it is an arm and not a
default.

---

## 5. What each direction is worth, per cell

`DECODE_NEXT.md` §4's estimates are all taken at 4096 / B=32. The column that
matters for this table is the one it does not have: **B=1**, which is three of
the six cells and needs −9.5 to −10.6 ms of its own.

Scaling is inference from each item's mechanism, not measurement.

| | direction | B=32 (§4) | B=1 | why it scales that way |
|---|---|---|---|---|
| **D1** | eviction demote/promote width | −4 to −8 | **~0 to −1** | the waste is a `[L·B, n_q, H_kv, ws, D]` gather plus fp32 temporaries — pure bytes, and `L·B` is 1024 at B=32 against 32 at B=1. The op *count* is unchanged, so the launch-bound B=1 eviction keeps its cost |
| **D2** | the card's dead `eps` field | −1 to −3 | **~0 to −0.5** | `recon` is a `[N, H, ws, D]` fp32 materialization; same 32× argument |
| **D3** | tile rung, if the gated kernel was capped | 0 to −4 | **0 to −4** | serial iterations per program, invariant in B. If anything worth *more* at B=1, where the kernel is latency-bound at 8–136 blocks on 108 SMs |
| **D4** | `sel` ascending (landed, unmeasured) | −2 to −3 | −1 to −3 | gather locality; helps wherever the gathers are issued |
| **D5** | persistent `wsum`/`wmax` scratch | −0.5 to −1.5 | −0.5 to −1.5 | 96 allocator calls per step, host-side, identical at every B |
| **D6** | redundant `SEL` scalar reloads | 0 to −0.5 | 0 to −0.5 | L1-resident; likely free already |
| **A** | **Block A, if `exp2` owns it** | **−4.3 to −4.9** | **−4.5 to −5.3** | flat by measurement, not by inference |

Applying `DECODE_NEXT.md` §4's own overlap rules (D1+D2 together −5 to −10;
D3+D4+D6 together −2 to −4):

| | needed | D-series only | + Block A recovered |
|---|---|---|---|
| **B=32** | −9.8 to −13.7 | −7.5 to −15.5 | −11.8 to −20.4 |
| **B=1** | −9.5 to −10.6 | **−2.5 to −6.0** | **−7.0 to −11.3** |

**The D-series alone cannot reach the B=1 rows.** Its two largest items, D1 and
D2, are byte-scaled and are worth ~nothing at B=1, and what is left is short by
4 to 8 ms — which is Block A, to within the precision of either estimate. This
is the concrete consequence of §2's signature: a plan built at one batch size
inherits that cell's cost structure.

At B=32 the D-series alone *might* reach it, at the top of its range, if every
estimate lands well. `DECODE_NEXT.md` §4 already says as much — *"ties at
best"* — and that was before the operating-point confound (§3) made the true
matched-config gap larger than the table's.

---

## 6. The order to do this in

Steps 1–3 cost no code and decide what steps 4+ are worth. `DECODE_NEXT.md` §5
has the same instinct and the same first two commands; this adds the arm from §4
and reorders around §3's finding that 43% of the gap is not on its list.

**1. Run the bisect. One command, and it can end the project.**

```bash
bash scripts/bisect_decode_regression.sh --model /path/to/llama-3.1-8b
```

Three table arms plus a profile. Decides Block A — 43% of the gap — under a
pre-committed rule, and if RoPE dtype owns it, says immediately that the
published numbers cannot be reproduced without giving back the `qasper` fix.
Everything else is sized against this answer.

**2. Read three lines off the profile it prints.**

- `[StickyKV] fused decode tiling: target_keys=…` — **this is all of D3.** 64
  closes it; 16 or 32 means the gated kernel runs 90 or 45 serial Q-tier
  iterations against the ungated 23, and D3 becomes the largest kernel item.
- `GPU busy` against the table's `TPOT_steady` — settles H-J, which decides
  whether launches or kernels lead.
- `gated` / `read_fraction` from `stats()` — confirms the gate fired at all
  before anything is attributed to it.

**3. Build the ungated arm (§4) and run it.** Prices Block B — the other 57% —
against the one configuration that produced the target numbers. Small, and the
only new code before step 5.

**4. Re-run at the published operating point.** `CACHE_BUDGET=0.50
LOCAL_WINDOW=64` removes the §3 confound so the D-series has a valid reference.
Note D1 scales with `n_q`, which is 184 there against 64 now, so the padded-width
waste is ~3× larger at the operating point the target was set at.

**5. Then implement, re-running the table after each — not at the end.**

| order | item | conditioned on |
|---|---|---|
| 1 | D2 (`eps`) | nothing — cheapest real change, and it shrinks D1's payload |
| 2 | D3 (tile rung) | step 2's printed line; free if it says 64 |
| 3 | D1 (demote width) | step 4's `n_q`; the largest item and the most work |
| 4 | Block A disposition | step 1 |
| 5 | D5 / D6 | the chrome trace pointing at them |

**6. Also fix, because they cost nothing and each one has already produced a
wrong number once.**

- `scripts/run_perf_table.sh` header says `CACHE_BUDGET default: 0.50` (line 34)
  and `LOCAL_WINDOW default: 64` (line 40); the code does 0.20 (line 115) and 128
  (line 121). This is the exact confound of `DECODE_NEXT.md` §3.3, still sitting
  in the `--help` text a reader would check it against.
  (`RECENT_CHANGES_AND_HYPOTHESES.md` §7.1 records it; it is still there.)
- `audit_e2e.py` rungs 2/3 compare the fused path against itself and print
  `OK — identical`, so H-B's acceptance gate is vacuously true (§7.2). Fix
  before quoting any accuracy number from this branch.
- The published table's third shape is **1048/1048**; the current table's is
  **1024/1025**. A 2.3% difference in prefill, and `cache_budget` multiplies
  `prefill + max_tokens`, so the two rows are not the same cell. Re-run that row
  at 1048/1048 before reading its 20.1% as the same quantity as the others.

---

## 7. Confounds, and what was not checked

- **The target numbers here are the repo's own record, not the artifact.** The
  published artifact could not be read from this session (access denied to a
  non-member reader). `DECODE_HISTORY.md` §1 is used instead; its three B=32
  cells match `GATE_REGRESSION.md` §1's baseline column exactly (0.0626 /
  0.0531 / 0.0488), and its e2e figures reconcile with its TPOT and TTFT to
  within a percent, so it is self-consistent. **If the artifact's QEvict rows
  differ from `DECODE_HISTORY.md` §1, every gap in §1 moves and this document
  should be re-derived.**
- **Operating point.** Target at 0.50/64, current at 0.20/128. The current
  configuration does less work, so every gap in §1 is a **lower bound** on the
  matched-config gap. `DECODE_NEXT.md` §3.3 predicts HEAD at 0.50/64 lands
  0.080–0.085.
- **Block A's splice assumes the H-E table predates `ecc0792`.** H-E's arms are
  the `exp2` and RoPE-dtype changes, which land before `7a2d404` (2026-09-14);
  `ecc0792` is 2026-09-15. Consistent, but the H-E run's own config was not
  recorded. If it was taken at 0.20/128, Block A and Block B swap some of their
  share and §5's B=1 conclusion gets *stronger*, not weaker.
- **Nothing here was measured.** Every ms in §5 is arithmetic on
  `DECODE_NEXT.md` §2's profile and §4's estimates, scaled by a mechanism
  argument. The batch-invariance finding in §2 is the exception: it is a ratio of
  measured table cells and rests on no model.
- **This document is entirely about speed.** Accuracy has not been re-measured
  since the gate landed (H-H), and the only decode path on CUDA has never been
  validated on CUDA (H-B). Neither is affected by anything proposed here, and
  both outrank it.
