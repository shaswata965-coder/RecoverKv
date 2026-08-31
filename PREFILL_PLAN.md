# Prefill plan — remove the second `O(N²)` pass, clear the OOM, keep the score

## What is actually wrong

Prefill runs **three** `O(N²)` passes per layer where it should run two:

1. flash's own forward — the attention output. Necessary.
2. `score_kernel.compute_lse` — recomputes the softmax normaliser `L` that
   pass 1 already computed internally. **Pure waste.**
3. `_token_scores_triton` — the per-key column sum. Inherent to H2O scoring;
   no flash library exposes a column-sum of the attention matrix, so it stays.

Pass 2 is skipped only when `L` is reused from pass 1. It is not being skipped:

```
compute_lse ran 32x this prefill — L did NOT come from the forward
(installed source: 'flashinfer')
```

32 = one per layer. Every prefill layer pays it, at every batch size.

### It is also the OOM

`compute_lse` materialises `[B, H_kv, rep, chunk, S]` in fp32, and none of the
ops in `matmul(...).float() * scaling` → `aw.masked_fill(...)` are in place, so
two full blocks are live at the crossover. Per `modules/evaluation/memory_model.py`:

| prefill 4096, batch 32 | GB |
|---|---|
| weights | 14.96 |
| prefill KV (uncompressed — eviction is on decode step 0) | 16.00 |
| **`compute_lse` transient** | **32.00** |
| other | 7.86 |
| device used | **86.36** vs 79.25 available |

The wasted pass is larger than the model weights. Removing it takes the cell to
**47.3 GB**. So pass 2 is one bug with two symptoms, and this plan fixes the bug.

### What removing it will and will not buy

Measured against the Flash baseline at 1048/1048, our prefill overhead splits
into **3.5 ms batch-proportional and 334 ms batch-invariant**. `compute_lse` is
the *entire* proportional term.

- **Batch 32:** removes ~108 ms of a 446 ms overhead (~24%).
- **Batch 1:** removes ~3.5 ms of a 338 ms overhead (~1%).
- **This does not deliver 3× prefill.** The 334 ms invariant term is untouched
  by everything in this document and is still unattributed; that is
  `scripts/audit_e2e.py --profile`'s job, not this plan's.

Stating that up front because sizing this fix as "the prefill fix" would repeat
the error that made the eviction compile look like a 4× lever when it was worth
1.5%.

---

## Stage 0 — Find out why the capture misses (one run, ~10 min)

Nothing below should be guessed at; the instrumentation to answer it landed in
`flashinfer_lse.broken_reason()` and the perf runner's warning. Run any perf
cell and read the L-reuse warning line. It now distinguishes two cases with
opposite fixes:

| the warning says | meaning | go to |
|---|---|---|
| `LATCHED OFF ... <Type: msg>` | FlashInfer ran and raised; capture disabled for the process | Stage 1B |
| `never REACHED` | no exception — the patched symbol is not the one the model calls | Stage 1C |

Do not skip this. The previously recorded cause ("batch>1 takes the VARLEN
path") is **wrong for this harness**: `perf_runner` passes no `attention_mask`,
so `_update_causal_mask` hands flash\_attention\_2 a `None` mask and
`_flash_attention_forward` calls the patched `flash_attn_func` at every batch
size. Two hypotheses already ruled out by reading, so they need no run:

- the consumer-side `score_meta is None` guard — `_materialize` returns
  `score_meta=None` on an empty Q tier (`cache.py:812`), and prefill has no Q
  tier under `first_eviction_step=0`;
- the wrapper's `q.dim()==4 and q.shape[1] > 1` bail — flash layout is
  `[B, S, H, D]`, so that is `S > 1`, true in prefill.

---

## Stage 1 — Get `L` from the forward

### 1A. Try the `flash` backend first — it costs one environment variable

```bash
STICKYKV_LSE_BACKEND=flash
```

**This has never been tried on the box.** `hooks._install_lse_source` defaults
to `auto`, and `auto` prefers FlashInfer whenever it imports. FlashInfer does
import there — the banner says so — so `flash_lse` was never reached.

It is also the **safer** option for the score constraint, and the existing
design note undersells this. The two sources are not equivalent:

- `flash_lse` calls the *same* `flash_attn_func` with `return_attn_probs=True`
  and keeps the `softmax_lse` the kernel already computed. **The attention
  output is bit-identical** — nothing about the model's forward changes.
- `flashinfer_lse` **replaces the attention kernel**. The model's attention
  output then comes from a different kernel with a different accumulation
  order. `FLASHINFER_LSE.md` frames this as a bonus ("runs the prefill
  attention faster"); for a run whose requirement is "the exact same score" it
  is a numerics change to the forward, not just to `L`.

Failure mode is clean: a build that rejects `return_attn_probs` raises
`TypeError`, `flash_lse` latches off, output stays correct. Shape is
`[B, H_q, S_q]` in natural log of the scaled scores — exactly what
`compute_lse` returns, and what the consumer's shape check expects.

**Gate:** `score_kernel.lse_recompute_count() == 0` across the prefill. If it
is 0, stop here — pass 2 is gone at zero risk to the score.

### 1B. If FlashInfer latched — fix the recorded failure

Only if 1A also fails. Use the reason string; do not re-derive it. The likely
classes, given the call site in `_planned_wrapper`:

- `plan()` signature drift (`head_dim_qk` vs `head_dim`, `q_data_type`) across
  FlashInfer versions;
- `run(..., return_lse=True)` returning a different arity or LSE layout;
- workspace too small — `STICKYKV_FLASHINFER_WORKSPACE_MB` (default 128).

If the LSE comes back in **log2** units, that is
`STICKYKV_FLASHINFER_LSE_LOG2=1`, not a code change. Getting this wrong is
silent: `L` off by a factor of ln2 does not crash, it *shifts every score*, so
it must be checked by the Stage 3 assertion rather than by eye.

### 1C. If the capture was never reached — find the real symbol

`flashinfer_lse.enable()` patches
`transformers.modeling_flash_attention_utils.flash_attn_func`. If the installed
transformers resolves its flash entry point differently (a direct
`from flash_attn import flash_attn_func` inside the attention module, a
different util module), the patch is on the wrong object. Confirm by checking
identity of the symbol the attention module actually holds, then patch that.

---

## Stage 2 — Make the OOM independent of Stage 1

Stage 1 removes the transient by removing the pass. Stage 2 makes the *fallback*
survivable, so a future capture regression costs TTFT and not the run. Do both;
they are independent.

### 2A. Immediate, zero code, bit-identical

```bash
STICKYKV_PREFILL_SCORE_CHUNK=128
```

86.4 GB → **52.2 GB**. `compute_lse` chunks over **query rows** and each row's
`logsumexp` reduces only over its own keys, so a smaller chunk is
**bit-identical**, not merely close. Same total FLOPs and same total memory
traffic — **this buys memory, not time.** Ship it as the default for large
`B × S` cells regardless of what Stage 1 concludes.

### 2B. In-place the fallback chain

In `compute_lse`, `matmul(...).float() * scaling` then `aw.masked_fill(...)`
holds two full fp32 blocks. `mul_` and `masked_fill_` on the already-copied
fp32 tensor halve the peak. Numerically identical — same values, same order,
one fewer allocation. Cheap and low-risk.

### 2C. A Triton `compute_lse` (the real fix for the fallback)

A one-pass, online-max kernel that never materialises the logit block: `O(N²)`
FLOPs, `O(1)` memory. This is a flash forward without the value matmul, and the
project already has the shape of it in `_score_kernel`. It removes the transient
term **whether or not the capture ever works**, and it removes the dependence of
the OOM on a fragile monkeypatch.

Worth doing even after Stage 1 succeeds: it turns "L-reuse regressed" from a
run-ending OOM into a TTFT regression.

---

## Stage 3 — Prove the score did not change

Non-negotiable, and it is the part that makes this plan safe to land.

1. **`L` equality.** With the capture live, assert the reused `L` matches
   `compute_lse` on the same inputs to fp tolerance. `flashinfer_lse`'s own
   docstring already claims this check exists (§"asserts FlashInfer's LSE ==
   compute_lse"); make it a test that runs, not a claim.
2. **Score equality.** `token_scores` from the reused `L` vs the recomputed
   `L` — `allclose`. This is what actually feeds eviction; equal `L` and unequal
   scores would mean a units or layout bug.
3. **End-to-end tokens.** Greedy decode from an identical prefill must emit an
   identical sequence with the capture on and off. `scripts/audit_e2e.py`
   already does this comparison for the decode routes; reuse the mechanism.
4. **The counter is the gate.** `lse_recompute_count() == 0` is the only
   evidence the pass is actually gone. A green test suite with a nonzero count
   means the optimisation is off and the timings are the old path.

Run the LongBench / GSM8K parity suites before and after. Bit-identical is the
bar for 1A and 2A/2B; fp-tolerance is the bar for 1B, because FlashInfer changes
the attention kernel.

---

## Order of work

| # | action | risk to score | buys |
|---|---|---|---|
| 0 | read the recorded L-reuse reason | none | decides 1B vs 1C |
| 1 | `STICKYKV_PREFILL_SCORE_CHUNK=128` | none (bit-identical) | the 4096/32 cell |
| 2 | `STICKYKV_LSE_BACKEND=flash` | none (same kernel) | pass 2, if the build allows |
| 3 | Stage 3 assertions | — | the right to trust 1–2 |
| 4 | fix FlashInfer (1B/1C) *only if 2 failed* | forward numerics change | pass 2 |
| 5 | in-place chain (2B) | none | half the fallback transient |
| 6 | Triton `compute_lse` (2C) | fp tolerance | fallback OOM-proof |

Steps 1 and 2 are environment variables. If both hold, the OOM is gone and the
second pass is gone before a single line of kernel code is written — which is
the outcome to aim for, and the reason Stage 0 comes first.
