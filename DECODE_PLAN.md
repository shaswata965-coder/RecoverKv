# Decode plan — beat FullKV by 20%, at the shape where that is possible

## Pick the shape first, because at 1048 the target is arithmetically unreachable

Decode reads the weights every step whether or not the KV is compressed. So a
KV-compression method's ceiling is set by how much of the traffic the KV is:

| shape | FullKV traffic | ours (20% keys, 48% bytes) | **best possible speedup** |
|---|---|---|---|
| 1048 / B=32 | 19.05 GiB | 15.35 GiB | **1.24×** |
| 4096 / B=32 | 30.96 GiB | 16.49 GiB | **1.88×** |
| 8192 / B=32 | 46.96 GiB | 18.03 GiB | **2.60×** |
| 4096 / B=1 | 15.46 GiB | 15.01 GiB | **1.03×** |

"20% faster than FullKV" needs a 1.25× advantage. At 1048/B=32 the ceiling is
1.24× — the target is unreachable **at infinite efficiency**, because the
weights are 79% of the traffic. At B=1 it is hopeless at any length: one row's
KV is a rounding error next to 15 GiB of weights.

**So the target shape is 4096 and 8192 at B=32**, and that is not moving the
goalposts — it is where a KV-compression method can win at all. Every number
below is 4096/B=32 unless stated.

## The target is proven reachable by a peer on this exact harness

| method | TPOT @ 4096/256, B=32 | × off its own roofline |
|---|---|---|
| SnapKV / StreamingLLM | **0.033** | ~2.8× |
| Flash (FullKV) | 0.086 | ~4.2× |
| **ours** | **0.1345** | **~11.4×** |

SnapKV compresses too, and gets 2.6× *faster* than FullKV on the same GPU and
harness. So the target is not a physics problem — it is an implementation gap.
We need 11.4× → 5.8× off roofline. SnapKV shows 2.8× is attainable.

Goal: **134.5 ms → ≤68.8 ms** (a 1.95× improvement).

## Two facts the last run established, which redirect the work

**A. Decode cost barely depends on cache size.** At B=1, going 1048 → 4096
gives 4× the context, 5× the fused kernel's serial iterations (12 → 65 Q-tier
windows), and 5.5× the eviction's window count — and TPOT went *down* 4%
(0.0851 → 0.0816). **Neither the fused Triton kernel nor the KV reads are the
budget.** Anything proportional to kept keys is ruled out by this, and that
includes further compression: a smaller cache cannot make this faster.

> Caveat carried forward from prefill: "invariant to X" does not mean
> "unrelated to X" when the work is launch-bound at these sizes. `compute_lse`
> looked batch-invariant and was still the prefill problem. Treat A as strong
> evidence, not proof, and let Stage 0 confirm it.

**B. The excess splits cleanly in two.** At B=1, 4096:

```
normal step          51.2 ms   vs FullKV 25.0 ms  ->  26.2 ms on EVERY step
eviction amortized   30.4 ms   (~243 ms per eviction, every 8 steps)
total excess         56.6 ms   (measured 0.0816 - 0.025 = 56.6)
```

The two halves are roughly equal. **Fixing either one alone cannot reach the
target**, which is why every previous single-lever attempt returned ~1.5%.

## Stage 0 — attribute both halves (one run, prerequisite)

`scripts/audit_e2e.py` is unblocked now. Run it at the target shape:

```bash
python scripts/audit_e2e.py --config outputs/prefill_flash/_perf_table.generated.yaml \
    --prefill 4096 --gen 64 --batches 32 --profile
```

Read, in order:

1. **GPU busy %** on the decode phase. Fact A predicts host-bound (<50%). If it
   comes back >85%, A is wrong, the kernels are the cost, and Stages 1–2 below
   are aimed wrong — stop and re-plan.
2. **Which ladder gap holds the 26 ms normal-step overhead.** Rung 2→3 is the
   fused kernel; 1→2 the Q tier; 0→1 everything else.
3. **The op names** under the top CPU self-time list. A host-bound step with few
   CUDA kernels and no dominant ATen op means Python — which neither list names,
   and which then needs a `cProfile` follow-up rather than more profiler work.

Do not start Stage 1 or 2 before this returns. That is the discipline the
eviction compile skipped, and it cost a campaign.

## Stage 1 — the eviction (~30 ms/step, 53% of the excess)

~243 ms per eviction, 7.6 ms/layer, ~6× a normal step. Its op count is fixed
(~360 pointwise/gather/scatter per layer) regardless of window count — which is
why it did not grow 5.5× from 1048 to 4096, and why it reads as launch-bound.

**1a. Find out why compiling it bought 1.5%.** The prime suspect is in the tree:
`3b311c5` excludes `_affine_quantize` from the graph with
`torch._dynamo.disable`, because Inductor ≤2.6 could not schedule the `amin`
over a data-dependent gather. That is a **graph break per layer per eviction**,
and `perf_runner`'s own warning says a broken region runs eagerly — "the
compiled eviction may be issuing the eager launch count under a compiled name".
The commit argued the quantizer is only ~10 ops per demoted window so the win is
"essentially intact"; that reasoning was never checked against a clock. Check it:
compare `STICKYKV_COMPILE_EVICT=1` against `0` at this shape and see whether the
gap is 1.5% or 30%.

**1b. Close the break rather than route around it.** If 1a shows the break is
the cost, the fix is to make the quantizer lowerable — compute the affine range
on a **gathered, materialised** tensor so the reduction's read index is not a
StarDep, or write the range reduction as a small Triton kernel. Byte-identity is
testable: the `aot_eager` eviction test already pins it.

**1c. Do not reach for `window_size`.** Halving the eviction frequency by
doubling `ws` changes which tokens are scored together and therefore the scores.
Out of scope under the same-logic constraint. Say so out loud when it gets
suggested, because it is the obvious tempting move.

## Stage 2 — the normal step (~26 ms/step, 47% of the excess)

Every decode step costs 2× FullKV *before* any eviction happens. Per layer that
is 0.8 ms, and by Fact A it is not the kernel and not the KV.

Candidates, in the order Stage 0's op list will discriminate:

- **The score hook's Python.** It runs per layer per step, extracts args from
  the module call, and on the fused path returns early — but only after the
  extraction work.
- **The `flash_decode` hand-off.** `set_pending` builds a dict per layer per
  step; the pre-hook clears it; the wrapper transposes q/k/v both ways.
  Individually cheap, 32× per step, on the critical path.
- **Triton's Python launcher.** `_two_tier_decode_kernel[grid](...)` passes ~40
  arguments; Triton hashes and specialisation-checks every one on every launch.
  This is a real, known cost at this argument count and it is invisible to a
  CUDA-time profile — it shows up as CPU self time.
- **Anything that syncs.** One `.item()` on the decode path serialises the
  pipeline and exposes every downstream launch.

## Stage 3 — only if Stages 1 and 2 fall short

The arithmetic: 51.2 − 26.2 (Stage 2) = 25 ms normal step, plus a Stage-1
eviction at ~50% off = 15 ms amortized ⇒ ~40 ms at B=1, scaling to roughly
70–80 ms at B=32. That lands at or just past the 68.8 ms target with no margin.
If Stage 0 finds a third term, it goes here. If not, the remaining lever is the
fused kernel's efficiency — `BLOCK_R=16` with `rep=4` and `BLOCK_WS=16` with
`ws=8` means every Q-tier `tl.dot` is 12.5% useful work — but Fact A says that
is not where the time is, so it is last, not first.

## Constraints

1. **Scores unchanged.** Every step here is a launch/scheduling change, not a
   math change. Where a change touches numerics (a Triton range reduction), the
   bar is fp tolerance against the eager reference, and the existing
   byte-identity eviction test is the gate.
2. **Kernel-or-error.** No silent fallbacks. A path that cannot run must raise,
   as the score kernel, the fused decode kernel and now the L-capture do.
3. **Measure, then fix.** Every number in this document is measured or derived
   from a measurement, and every stage is gated on Stage 0. The one prior
   attempt that skipped this — sizing the eviction from a dispatch count taken
   inside one function — targeted 81% of the wrong denominator.

## Success criteria

| | target | how verified |
|---|---|---|
| TPOT @ 4096/256 B=32 | ≤ 0.0688 (0.8 × FullKV) | `run_perf_table.sh`, fresh `OUT_DIR` |
| beats a peer method | < 0.086 (Flash) | same table |
| scores unchanged | identical | LongBench / GSM8K parity before and after |
| stretch | approach 0.033 (SnapKV) | same table |

Report TPOT_steady at a generation long enough to include evictions — a short
`gen` puts zero evictions in the steady window and flatters the number by ~40%,
which is exactly how 0.0512 got mistaken for a decode win.
