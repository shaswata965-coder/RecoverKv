# RULER 32k regression: what was found and fixed

Date: 2026-09-22 · Fix commit: `0b21c9f` on `Clustered_int2_paused_fast`

> **Update 2026-09-23: the fix below did not move RULER.** On the `0b21c9f`
> build, cwe (18.56), niah_single_3 (71.40), niah_multikey_2 (91.00) and
> niah_single_2 (99.20) scored exactly what they scored before it. The fill it
> corrects is real, but it was not what decided those tasks.
>
> **The cause was which windows the gate reads, not how it weights the rest.**
> The gate picks windows per KV head for four query heads at once, and it
> merged them by the max of each head's *raw* logit estimate. A raw logit
> carries the head's baseline and scale, which its softmax ignores, so the
> loudest head of each group chose for all four. A retrieval head sharing a KV
> head with a louder one had its needle window read on about half the steps.
> The recall test could not see this because it summed raw mass across the
> group, which the same loudest head dominates. It read 1.000 while the worst
> query head had 0.05% of its mass read.
>
> Fixed by ranking each window on the largest share of its own int2 mass that
> any head of the group puts on it (`sketch.group_share`, and a second small
> Triton launch, `_gate_share_kernel`). Full write-up: `DISTANCE_TO_GOAL.md` §12.
> Tests: `tests/test_gate_union_is_per_head.py`. CPU-verified; not yet run on a
> GPU.

## The report

RULER 32k at `window_size=32`, compared with the QEvict reference.

| task | QEvict | this branch (before fix) |
|---|---|---|
| cwe | 43.98 | 18.56 |
| niah_multikey_1 | 99.80 | 98.20 |
| niah_multikey_2 | 97.00 | 91.00 |
| niah_multiquery | 97.00 | 95.70 |
| niah_multivalue | 95.90 | 87.10 |
| niah_single_2 | 100.00 | 99.20 |
| niah_single_3 | 99.00 | 71.40 |
| qa_1 | 84.60 | 78.00 |

Operating point, from `scripts/run_ruler32k_windows_8gpu.sh`: budget 0.20, `quant_ratio` 0.70 (bytes), gate 0.25, local 128, sink 5, first eviction at step 0.

The biggest clue was a note already in that script. At `ws=8`, `niah_single_3` scored 2.40, and the model "reproduces the first ~8 chars of each UUID and invents the rest". Getting the start of a copy right and then drifting points at the decode step, not at prompt compression. The tasks with the longest verbatim copies (UUIDs, multiple values) were hurt most, while `niah_single_2` (a 7-digit number of about 3 tokens) held up.

## Summary

I found two logic bugs in the decode path and fixed both.

1. **Windows the gate skips were credited with the needle's weight.** This is the main cause. On a retrieval step, the weight given to the skipped int2 windows grew with how sharply the head focused on the needle. That diluted the needle's value in the attention output.
2. **Window scores were accumulated in fp16.** Almost every decode step's contribution to the eviction ranking was rounded away.

Neither bug shows up at `quant_gate_ratio=1.0`, the control setting, which is why that setting could not have revealed them.

**Status: verified on CPU only.** Nothing has run on a GPU. The edited Triton kernel type-checks for an A100 but has never executed. How much of the RULER gap these fixes close has not been measured.

---

## What I checked and ruled out

| area | finding |
|---|---|
| RULER runner and prompt (`ruler_runner.py`) | Matches DefensiveKV/kvpress: chat template plus answer prefix, no duplicate BOS, same cache setup as the LongBench runner. |
| RULER scorer (`ruler_scoring.py`) | Faithful port of the reference `string_match_all` / `string_match_part`. |
| Budget resolution (`config.resolve`) | Correct. Numbers below. |
| Eviction mechanics (`_evict_two_tier_impl`, joint migration, `_begin_step`, score-width check, partial-window and unscored-tail handling) | Consistent. A CPU `generate()` run went through five evictions cleanly. |
| Prefill score kernel (`score_kernel.py`) | GQA head mapping, causal mask, exp2 base change and log-sum-exp reuse are all correct. |
| RoPE un-rotate / re-rotate for int2 keys | Correct pure rotation at the store dtype. |
| Gate kernel (`gate_kernel.py`) | Matches `gate_reference`, including strides and int4 unpacking. |
| Score-scatter map (`compute_score_meta`) | Consistent with the body's window chunking. |
| **Skipped-window weighting in the gated decode** | **Defect 1** |
| **Dtype of the window-score accumulator** | **Defect 2** |

### Budget at 32k (arithmetic from `config.resolve`, assuming a prompt of about 32,000 tokens)

| ws | fp16 windows | int2 windows | total windows | dropped windows | share of prompt dropped |
|---|---|---|---|---|---|
| 8 | 236 | 2,095 | 4,000 | 1,653 | 41% |
| 32 | 59 | 840 | 1,000 | 97 | 10% |
| 64 | 29 | 470 | 500 | 0 | 0% |

Every int2 window's budget cost includes its gate card (added in `ac128e8`). The card is proportionally more expensive at small `ws`. That is most of why `ws=8` is far worse than `ws=32`. This is honest memory accounting, not a bug, so I did not change it.

---

## Defect 1: skipped windows inherited the needle's weight

### Mechanism

Each decode step, the gate reads 25% of the int2 windows exactly. Every window it skips still counts in two places:

- as its eviction score;
- in the attention output, through its value average (centroid).

In both places the weight is `c · exp(logmass)`, where `logmass` is the card's estimate of the window's attention mass. `c` was calibrated as a **ratio of sums** over the windows that were read:

```
c = Σ_read exact / Σ_read estimate
```

That ratio is only right when the card is off by the same factor everywhere. On a retrieval step it isn't:

1. A head copying a token out of a read int2 window puts nearly all its attention on that one token.
2. The card models a window as its mean plus one direction (rank 1), and that direction points at the token farthest from the mean. The token being copied is usually a different one, so the card underestimates it badly.
3. So `Σ_read exact` is essentially that one token's mass, and `c` becomes that token's estimation error.
4. Every skipped window was then credited a share of the needle's mass: roughly 3× the needle's own mass at ratio 0.25.
5. The output is renormalised by `1 + Σ fill`, so the needle's value was diluted by that factor and replaced by a blend of centroids.

The sharper the head, the more weight it handed to windows it never read. That is backwards, and it is exactly the step RULER tests.

### Fix

`c` is now the **mean log ratio** over the read windows:

```
log c = mean_read( ln exact_j − logmass_j )
fill_i = exp( logmass_i + log c )
```

- It still corrects a card that is off by one common factor exactly.
- One outlier window moves `log c` by only `1/n_sel` of its own log error, instead of setting `c` outright.
- The softmax normaliser cancels out, so the kernel doesn't need it for this.
- The log of each read window's mass is computed as `log2(ssum) + smax − lse`, never as the log of an exponentiated value. That way a window far below the step's peak cannot underflow to zero and drag the mean to −∞.
- Read windows with zero mass are left out. If no read window can be measured, the skipped windows get nothing.

Three places changed together:

- the Triton kernel's `GATED` epilogue (`modules/windowed_cache/decode_kernel.py`);
- its CPU reference (`two_tier_window_reference`);
- the spec function (`modules/windowed_cache/scorer.py::fill_skipped_window_scores`).

### Evidence (CPU only)

These use the repo's own CPU reference of the fused kernel, with real cards from `build_sketch` / `gate_and_score`. Keys are synthetic, with a large common component in a few channels (like real massive-activation channels) plus per-window and per-token variation. The hot token is a random token in a random window, not the one the card's direction points at. Numbers are relative output error against full attention, averaged over 4 seeds. "No fill" means the skipped windows are ignored entirely.

**840 windows × ws=32, gate ratio 0.25 (the shipped geometry):**

| head type | no fill | old fill | new fill |
|---|---|---|---|
| diffuse attention | 1.505 | 0.257 | 0.257 |
| 1 hot token in an int2 window, sharpness 8 | 0.925 | 0.253 | **0.021** |
| 1 hot token in an int2 window, sharpness 12 | 0.042 | 0.530 | **0.002** |
| 1 hot token in an int2 window, sharpness 20 | 0.000 | 0.301 | **0.000** |
| 3 hot tokens in int2 windows, sharpness 8 | 1.335 | 0.314 | 0.313 |
| 3 hot tokens in int2 windows, sharpness 12 | 0.795 | 0.270 | **0.157** |
| 3 hot tokens in int2 windows, sharpness 20 | 0.177 | 0.378 | **0.138** |
| hot token in the fp16 tier, sharpness 8 | 0.876 | 0.019 | 0.020 |
| hot token in the fp16 tier, sharpness 12 | 0.039 | 0.002 | 0.003 |

At sharpness 12, the old fill (0.530) was worse than ignoring the skipped windows altogether (0.042).

**The same pattern holds in other settings:**

| setting, 1 hot token in an int2 window | old fill (sharpness 8 / 12 / 20) | new fill |
|---|---|---|
| 840 × ws=32, ratio 0.10 | 0.489 / 0.758 / 0.497 | 0.025 / 0.002 / 0.000 |
| 1600 × ws=8, ratio 0.25 | 0.373 / 0.452 / 0.110 | 0.015 / 0.001 / 0.000 |

**The weight credited to skipped windows now matches what they actually hold** (smaller test geometry):

| case | credited (new) | actually held |
|---|---|---|
| diffuse | 2.34 | 2.34 |
| sharpness 8 | 0.42 | 0.43 |
| sharpness 12 | ~0.03 | ~0.03 |
| sharpness 20 | ~0 | ~0 |

Under the old fill, the sharp cases were credited 0.2–1.3 extra when the skipped windows actually held less than 0.05.

---

## Defect 2: fp16 score accumulation

`hooks.py` created the window scores in the model's dtype (`q.dtype`, fp16). The running sum `state.window_scores` takes its dtype from the first scores it receives, so the whole eviction ranking ran in fp16.

At 32k, a window's prefill score sums over up to 32k query rows. A decode step then adds well under 0.01 per window. In fp16, adding a number smaller than half the spacing between representable values changes nothing, and above a total of about 32 that spacing is already larger than 0.02. So nearly every decode step's contribution was discarded, and every re-eviction ranked windows on the prompt alone.

This exact bug was diagnosed and fixed in July on `int2_qwen` (`446a7eb`). It measured 52% of per-step contributions lost in fp16, and the set of kept windows drifting about 9% off over 512 steps. The fix never reached this branch.

**Fix:** `SCORE_ACCUM_DTYPE = torch.float32` in `modules/windowed_cache/hooks.py`, used for the scores the hook hands to the cache. The fused decode path already produced fp32 scores, so the accumulator is now fp32 end to end. The extra memory is one fp32 `[B, H_q, W]` tensor per layer.

---

## Considered and not changed

- **Card bytes in the budget** (`ac128e8`). About 10% of the prompt is dropped at ws=32 and 41% at ws=8. This is honest accounting; changing it would misstate the memory claim.
- **How well the gate picks windows on real attention.** A rank-1 card models one token per window, and a copy step usually needs a different one. This is untested on real data. The next runs below separate it from everything else.
- **H2O-style prefill scoring.** Every prompt row adds to every key's score, with no observation window. This is question-blind at 32k, but it is the method's scoring rule (the observation window was removed in `b1e44b3`), not a defect.
- The standing rule that the read gate stays the only int2 read path is kept. No weaker or ungated path was added.

---

## Verification

| check | result |
|---|---|
| `tests/test_gate_fill_calibration.py` (new) | 42 passed. 24 of them fail on the parent commit `2f06bc4`, so they detect the old behaviour. |
| `tests/test_kernel_compiles.py` (Triton, both `GATED` variants, A100 target) | 19 passed. Type-checks only; not executed. |
| gate / sketch / window-score / decode test files | Same 12 failures before and after, all in stale tests of deleted control-arm and tile-override code. |
| CPU `generate()` smoke test: tiny fp16 Llama, q=0.7, ws=8, 40 tokens | Five evictions (step 0 per layer, then all layers together), compiled eviction, fp32 scores throughout. |
| Full suite (`tests`, `modules/evaluation`, `data`), old vs new code | Old: 1,229 passed, 123 failed, 6 errors. New: 1,271 passed, 123 failed, 6 errors. **The two failure lists are identical**; the extra 42 passes are the new test file. The 123 are pre-existing failures in stale tests (see `CLAUDE.md`, "Two traps in the harness"). |

Not verified: anything on a GPU, the Triton kernel's actual numbers, and any RULER or LongBench score.

The new tests pin these properties:

- the weight credited to skipped windows tracks their true mass, from diffuse to sharp heads;
- a needle in a read int2 window is not diluted;
- the old fill fails the same check (so the test can tell the two apart);
- a needle in the fp16 tier is unaffected;
- diffuse heads still get meaningful weight from skipped windows;
- the spec recovers a common factor exactly, moves by `1/n` for one outlier, and tolerates a read window with zero mass;
- the CPU reference and the spec agree;
- the hook hands the cache fp32 scores on an fp16 model.

---

## What to run next (GPU)

1. **RULER 32k, ws=32, gate 0.25, on `0b21c9f`**, with the same command as `scripts/run_ruler32k_windows_8gpu.sh`. Check that each task's metadata sidecar has `read_gate` equal to `gated`.
2. **The same run at `quant_gate_ratio=1.0`.** If NIAH still trails QEvict in run 1 but matches here, the remaining gap is how well the gate picks windows. Gate 1.0 was measured within 0.2–1.6% of 0.25 on decode speed (`130de24`).
3. **LongBench at the operating point.** Scores move with this change, so it needs its own quality record.

Quality rows do not compare across this commit; `CLAUDE.md` and `DISTANCE_TO_GOAL.md` §11 say so.

---

## Files changed in `0b21c9f`

| file | change |
|---|---|
| `modules/windowed_cache/decode_kernel.py` | `GATED` epilogue and `two_tier_window_reference`: mean-log-ratio fill |
| `modules/windowed_cache/scorer.py` | `fill_skipped_window_scores`: same rule, with the reasoning in the docstring |
| `modules/windowed_cache/hooks.py` | `SCORE_ACCUM_DTYPE = float32` for the scores handed to the cache |
| `tests/test_gate_fill_calibration.py` | new, 42 tests |
| `DISTANCE_TO_GOAL.md` | new §11 with the finding, evidence and next runs |
| `README.md` | "Credit the rest" step describes the new rule |
| `CLAUDE.md` | quality rows do not compare across this fix; do not bring back a `Σ exact / Σ estimate` calibration |
