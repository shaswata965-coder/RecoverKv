# Prefill speed plan — parity, then beating FullKV

`PREFILL_PLAN.md` is done: the second `O(N²)` pass is gone, the 4096/batch-32
OOM is gone, and prefill went from 2–4× slower than FullKV-flash to **1.09–1.14×**.
This plan is about the next target: *beat* FullKV, or beat named methods.

## The arithmetic that bounds the target

Our prefill is FullKV's prefill **plus** a score pass. We do strictly more work,
so we cannot be faster unless we make the *model's own* work smaller. What is
that work?

Llama-3-8B FLOPs, by term:

| N | linear layers | attention | attention share | our score pass |
|---|---|---|---|---|
| 1048 | 14.6 T | 0.3 T | 1.9% | 1.0% |
| 4096 | 57.2 T | 4.4 T | **7.1%** | 3.6% |
| 16384 | 228.7 T | 70.4 T | 23.5% | 11.8% |
| 32768 | 457.4 T | 281.5 T | 38.1% | 19.0% |

**At 4096, prefill is 93% linear layers.** No KV-cache method touches those.
Even deleting attention *entirely* would buy 7.1%. So:

> **20% faster than FullKV is arithmetically impossible at 4096.**
> Not hard — impossible, by a bound that does not depend on implementation.

Any published claim of faster-than-FullKV prefill is either at much longer
context, or is not running a full forward.

## Where we actually stand

The prefill overhead worth attacking is small and specific. At 4096, batch 1:

```
ours 0.381 s   FullKV-flash 0.335 s   -> +13.7%
our score pass is 3.6% of prefill FLOPs
=> it runs at ~26% of the efficiency of everything else
```

The score kernel is the whole gap, and it is **~4× less efficient** than the
flash forward it runs beside. That is the only headroom below parity, and it is
worth ~10 points.

We already beat several methods (4096/256, batch 1):

| | TTFT | vs ours |
|---|---|---|
| StreamingLLM | 0.333 | we lose |
| Flash (FullKV) | 0.335 | we lose |
| SnapKV | 0.339 | we lose |
| KIVI int2 | 0.360 | we lose by 5.8% |
| KIVI int4 | 0.367 | we lose by 3.8% |
| **ours** | **0.381** | — |
| DefensiveKV | 0.430 | **we win** |
| Eager | 0.744 | **we win** |

At 4096/batch-32 we also beat Eager, KIVI-int2 and KIVI-int4 outright — all
three **OOM** at that shape and we no longer do.

---

## Stage A — reach parity by fixing the score kernel (no score change)

Target: **0.381 → ~0.347 s**, i.e. 1.04× FullKV. That overtakes KIVI-int2,
KIVI-int4, DefensiveKV and Eager — 4 of 7 methods — and leaves us within 4% of
FullKV, StreamingLLM and SnapKV.

> **FOUND — it was none of the three suspects below.** Budgeting the kernel's
> three resources at 4096/batch-1 settles it without a profiler:
>
> | term | work | time |
> |---|---|---|
> | `tl.dot` | 2.2 TFLOP | ~7 ms |
> | Q traffic (BLOCK_N=128) | 17 GB | ~11 ms |
> | **`tl.exp`** | **8.6e9 exponentials** | **~36 ms** |
>
> measured overhead: **46 ms**. The kernel is **SFU-bound on the exponential**,
> not memory- or tensor-core-bound. `tl.exp` lowers to the accurate `expf`
> (~ten SFU ops); `ex2.approx.f32` is one instruction, which is why every
> FlashAttention implementation uses it. Fixed by folding `log2(e)` into
> `scale` and the `[BLOCK_M]` LSE vector — never into the `[BLOCK_M, BLOCK_N]`
> tile — so the base change costs nothing per element. `STICKYKV_SCORE_EXP2=0`
> restores `expf`. Larger `BLOCK_N` autotune configs were added too, since Q
> traffic is the next bound once the exponential stops dominating.
>
> Numerical note: `ex2.approx` carries ~2 ulp against `expf`'s ~1. Measured
> colsum relative error is **2.4e-7**, and scores are used for a *ranking*, so
> this can only matter where two windows are already tied to ~1e-6.

The kernel computes 3.6% of the FLOPs in 13.7% of the time. Suspects, in the
order the profile should test them (kept for the record — all three were
wrong, which is why the resource budget above is the better first move):

1. **It runs as a separate pass over K.** Flash reads K once for the forward;
   we read it again. At 4096/batch-1 the fp16 K tensor is small, so this is
   latency, not bandwidth — consistent with the prefill overhead being
   batch-invariant.
2. **Autotune config coverage.** `_score_kernel` autotunes over six
   `(BLOCK_M, BLOCK_N, warps, stages)` configs. Whether the winner is any good
   at `S=4096, T=4096, D=128` is unmeasured. Check `n_regs`/`n_spills` — a
   spilling kernel would explain a 4× efficiency gap on its own.
3. **The `IS_CAUSAL` skip.** The kernel starts each key block at
   `m_start = (m_lo // BLOCK_M) * BLOCK_M`, so it should skip fully-masked
   query blocks. Confirm it actually does; a kernel doing the full `N²` instead
   of `N²/2` is exactly a 2× efficiency gap.

`scripts/audit_e2e.py --profile` already reports the per-op CUDA time and the
Triton metadata; the score kernel's share of prefill is what to read.

**Ceiling: parity.** Stage A cannot beat FullKV, because our work is a superset
of FullKV's. Anyone proposing otherwise should be asked which of the 93% they
intend to remove.

---

## Stage B — beat FullKV, which needs a longer context

The only lever that makes the *model's* prefill cheaper is to stop attending
over the full prefix: process the prompt in chunks and evict between them, so
chunk *i* attends over a compressed prefix instead of a growing one. Attention
becomes `O(N·K)` rather than `O(N²)`.

With budget 0.20 and chunk = N/16, attention work drops to **40.9%** of full
causal — and what that is worth depends entirely on the context length:

| N | attention share | ⇒ prefill saved vs FullKV |
|---|---|---|
| 4096 | 7.1% | 4.2% |
| 8192 | 13.3% | 7.9% |
| 16384 | 23.5% | 13.9% |
| **32768** | 38.1% | **22.5%** ← clears the 20% target |
| 65536 | 55.2% | 32.6% |

**So the 20% target is reachable, at ≥32k context, via chunked prefill.** At
4096 the same change buys 4.2% and is not worth the risk.

### What this costs

Chunked prefill with progressive eviction **changes the model's output**: a
token in chunk *i* attends over an evicted prefix rather than the full one, so
this is a different method, not an optimisation of the current one. It needs
the full quality suite re-run, not a numerical-equivalence check. That is a
real decision, not a detail — it trades the "identical scores" property that
every change so far has preserved.

It also interacts with scoring: the score pass currently runs once over the
whole prompt. Chunked prefill needs per-chunk scoring, and the eviction
decision for chunk *i* is made without seeing chunks *i+1…*, which is
strictly less information than the current one-shot scoring has. Expect a
quality cost, and measure it before assuming it is small.

### The strategic point

At 4096 with a 20% budget, this method's premise barely bites: attention is
7% of prefill and the cache saves ~3 GB. At 32k the premise is real — attention
is 38% of prefill, and the memory saving is what makes the shape runnable at
all. **The benchmark shapes are shorter than the regime the method is designed
for**, which is why it looks like overhead rather than a win. SnapKV,
StreamingLLM and DefensiveKV are all evaluated at LongBench lengths for exactly
this reason.

---

## Recommended order

| # | action | risk to score | outcome |
|---|---|---|---|
| 1 | profile the score kernel; check `n_spills` | none | names the 4× gap |
| 2 | fix what the profile finds | none | ~1.04× FullKV; beats 4 of 7 methods |
| 3 | re-run the table at 16k and 32k | none | shows the method in its own regime |
| 4 | chunked prefill + progressive eviction | **changes outputs** | ≥20% faster than FullKV at ≥32k |

Steps 1–3 are safe and answer "can we beat some method" with yes. Step 4 is the
only route to "20% faster than FullKV", it only pays above ~16k, and it is a
method change that must be justified on quality as well as speed.

**If the requirement is 20% faster than FullKV at 4096, the honest answer is
that no implementation achieves it, and the requirement should move to a longer
context or to beating named methods instead.**
