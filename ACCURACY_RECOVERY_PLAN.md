# Accuracy recovery plan — LongBench regression across the efficiency span

Status: diagnosis complete from code + history. **Cause A is fixed — the
default is back on `bytes` and the knob now reaches the cache (§2, "Landed").**
Causes B–D are unmeasured and the isolation protocol in §5 still has to run.
The efficiency span is `c88581c..HEAD` (2026-08-13 → 2026-09-01), from the
fused Design-B score kernel through `auto` preferring flash.

---

## 1. The delta, stated

Llama-3.1-8B-Instruct, LongBench, `compression_ratio=0.8`. Earlier run
(`Unpressed_quant INT 2` / `QEvict_final`) against the current run:

| dataset          | n   | before | after | Δ         |
|------------------|-----|--------|-------|-----------|
| qasper           | 200 | 42.30  | 32.78 | **−9.52** |
| multifieldqa_en  | 150 | 55.74  | 50.89 | **−4.85** |
| gov_report       | 200 | 33.71  | 31.28 | −2.43     |
| trec             | 200 | 71.00  | 69.00 | −2.00     |
| samsum           | 200 | 41.77  | 39.82 | −1.95     |
| 2wikimqa         | 200 | 48.05  | 46.20 | −1.85     |
| multi_news       | 200 | 26.89  | 25.62 | −1.27     |
| musique          | 200 | 32.99  | 31.88 | −1.11     |
| qmsum            | 200 | 25.10  | 24.46 | −0.64     |
| narrativeqa      | 200 | 29.69  | 29.15 | −0.54     |
| hotpotqa         | 200 | 57.97  | 57.53 | −0.44     |
| triviaqa         | 200 | 91.21  | 91.09 | −0.12     |

**Every dataset moved, all in the same direction, none spared.** That shape
matters. A kernel bug in scoring or decode is lumpy — it wrecks the cases that
trip it and leaves the rest alone. A uniform, monotone slide across twelve
heterogeneous tasks is the signature of **the cache getting smaller**, with the
magnitude tracking how much each task depends on retained context (single-doc
QA hurts most; `triviaqa`, where the answer is a memorised fact, barely
notices).

The configs did not change: `configs/longbench_ours_flash_attn.yaml` has been
untouched since `fefdca2` (2026-07-27), *before* the span. So whatever shrank
the cache did it from code, under an unchanged `cache_budget: 0.20`.

---

## 2. Prime cause — `quant_budget_mode` silently stopped spending the budget

`f71fec0` (2026-08-29) changed the default of `quant_budget_mode` from `bytes`
to `tokens` (`modules/windowed_cache/config.py:220`). No YAML sets it, so every
run since has inherited `tokens`.

The two modes divide the evictable allowance differently
(`config.py:486-505`). At `window_size=8`, `H_kv=8`, `D=128`: `b_fp = 32768`
bytes per fp16 window against `b_q = 8448` per int2 window — an int2 window is
**3.88x cheaper**.

* `bytes` — `q` splits the byte budget; each tier buys windows at its own
  price. Bytes spent: **100% of the allowance**. Windows retained:
  `top_k · (1 + 2.879q)`.
* `tokens` — `q` splits the *window count*, which is held q-invariant. Windows
  retained: `top_k`. Bytes spent: **`(1−q) + 0.258q` of the allowance**.

Measured at the LongBench operating point:

| q   | tokens-mode byte-budget utilisation | bytes-mode windows vs tokens-mode |
|-----|-------------------------------------|-----------------------------------|
| 0.3 | 77.7%                               | 1.86x                             |
| 0.5 | **62.9%**                           | **2.44x**                         |
| 0.7 | **48.0%**                           | **3.02x**                         |
| 0.9 | 33.2%                               | 3.59x                             |

End to end, at `cache_budget=0.20`, sinks 5, local 128, `ws=8`:

| prefill | q   | bytes-mode kept | tokens-mode kept |
|---------|-----|-----------------|------------------|
| 3600    | 0.5 | 1621 (45.0%)    | 741 (20.6%)      |
| 4600    | 0.5 | 2069 (45.0%)    | 925 (20.1%)      |
| 8700    | 0.5 | 4293 (49.3%)    | 1837 (21.1%)     |

**At `quant_ratio=0.5` the current run attends over 2.2x fewer keys than the
earlier run did — same nominal budget, same config file.** That alone is enough
to produce the entire table in §1, and it produces it in exactly the observed
shape: uniform, monotone, worst where context matters most.

### Why the `tokens` default is wrong *here* specifically

`f71fec0` is a good fix aimed at the wrong scope. Its complaint is real and
worth keeping: under `bytes`, the retained key count grows with `q`, so a
latency row at `q=0.7` does more attention work than one at `q=0.0`, and a "50%
cache" can end up attending over more keys than the prompt it compressed. For
the **perf table**, holding the key count fixed is the only way rows compare.

But it was made the **global** default, so it also governs the quality
evaluation — and there it inverts the method's entire premise. The point of an
int2 tier is that a key costs 3.88x less, so *the same memory buys more
context*. `tokens` mode forbids exactly that: it grants a byte budget, then
declines to spend 37–52% of it, and hands the model less context than its own
budget paid for. That is not a fair operating point; it is an underspent one.

Note that `bytes` mode is not cheating the budget — both modes spend at most
the granted bytes, and `bytes` spends it precisely. The two differ only in what
`q` denotes: share of bytes, or share of windows. For a memory-vs-quality
claim, share of bytes is the meaningful one.

### Landed

Verified before the change: under `bytes`, `budget_utilisation` is 0.99 at
q = 0.0 / 0.5 / 0.7; under `tokens` it is 0.99 / 0.69 / 0.57. Suite after:
684 passed, the same 22 pre-existing `test_backend_e2e` failures as a clean
tree.

* Default back to `bytes` in `modules/windowed_cache/config.py`
  (`WindowedCacheConfig` and `ResolvedConfig`) and `utils/config.py`
  (`CacheConfig`).
* `quant_budget_mode` now forwarded by all three quality runners
  (LongBench / GSM8K / RULER) via `cache_factory.quant_budget_mode_kwargs`,
  which omits the kwarg for the eager package (it has no such field and
  computes `bytes` unconditionally) and **raises** if a config asks eager for
  `tokens` rather than silently dropping it.
* `ResolvedConfig.retained_bytes` and `.budget_utilisation` added — the second
  is ~1.0 when the cache costs what the budget granted, and is the single
  number that would have caught this.
* The LongBench sidecar now records `quant_budget_mode` and
  `resolved_geometry_first_example` (`top_k_fp`, `N_q`, `retained_windows`,
  `retained_tokens`, `retained_bytes`, `total_budget_bytes`,
  `budget_utilisation`).
* `capture_environment()` now records every set `STICKYKV_*` variable, so every
  suite's run record says which kernels and L source were active.
* Mode pinned explicitly in every config, so no suite inherits it again:
  `bytes` in the eight quality configs, `tokens` in the four perf/efficiency
  ones.
* `tests/test_quant_budget_mode.py` — 17 tests over the default, both modes'
  budget utilisation, q=0 equivalence, the eager omit/raise split, and the
  YAML→cache forwarding that was the inert half of this bug.

**A third divergence this surfaced:** the eager package
(`windowed_eager_cache/config.py:406-407`) has no mode branch — it has always
computed the byte split. So from `f71fec0` until now, flash defaulted to
`tokens` while eager stayed on `bytes`, and the two backends resolved the same
YAML to caches differing by 2.2x in retained keys at q=0.5. Any eager-vs-flash
comparison at `quant_ratio > 0` from that window is void.

### The fix

1. **Make `quant_budget_mode` per-suite, not global.** Quality suites
   (LongBench, RULER, GSM8K) run `bytes` — equal memory per row, which is what
   a KV-compression accuracy claim means. The perf table keeps `tokens` — equal
   work per row, which is what a latency claim means. Set it explicitly in both
   YAMLs so neither depends on a default.
2. Keep `perf_runner.describe_tier_geometry`'s expansion warning: a quality row
   under `bytes` should still *report* its `S_eff` expansion, so nobody reads
   45% retention as 20%.
3. Report the operating point as **retained bytes / full-cache bytes**, which
   is q-invariant and honest under either mode, instead of `1 − cache_budget`.

### Two defects that block even testing this

Found while tracing the knob:

* **`quant_budget_mode` is inert in the LongBench runner.**
  `modules/evaluation/longbench_runner.py:476-495` constructs
  `WindowedCacheConfig(...)` and forwards `quant_ratio`, `quant_memoize_read`
  and `first_eviction_step` — but **not** `quant_budget_mode`. Setting it in
  YAML today changes nothing and says nothing. Only `perf_runner.py` honours
  it. This is the same failure the surrounding comments already document twice,
  for `first_eviction_step` and `quant_memoize_read`; it recurred.
  **Fix first — the §5 A/B cannot run without it.**
* **The run manifest does not record which mode ran.**
  `longbench_runner.py:695-720` records `quant_ratio` but not
  `quant_budget_mode`, and not the resolved geometry (`top_k_fp`, `N_q`,
  `retained_windows`, `retained_bytes`). So neither run's sidecar can be asked
  what cache it actually held — which is why this had to be reconstructed from
  `git log` instead of read off the artifacts. Add all five fields, plus the
  active `STICKYKV_*` values.

---

## 3. Secondary causes — rank them, do not skip them

These are live, all default-on, and each is capable of taking points off on its
own. Cause A explains the *shape* of §1; these explain any residual after A is
corrected.

### B. The fused two-tier decode kernel has never been validated on a GPU

`44fe53c` / `cd0c8a5` / `59f854b` shipped a Triton kernel that computes decode
attention directly out of the int2 tier. `STICKYKV_FUSED_DECODE` defaults to
**`1`** (`decode_kernel.py:74`); on CUDA it fires on every decode step where
`quant_ratio > 0`. It does not merely score — it produces the attention output
the next token is sampled from.

The repo says plainly that it is unchecked:

* `tests/test_decode_kernel.py:109-120` is a **comment describing** the
  validation, not a test.
* `scripts/audit_e2e.py:533-557`: *"The fused kernel is the default on CUDA and
  has never been checked against its reference on a GPU, so this is the first
  evidence either way."*

`audit_e2e.py` already implements the check — rungs `2_q70_materialize` and
`3_q70_fused` are the same method by two routes and must emit an identical
greedy token sequence. It has not been run to conclusion.

### C. `auto` selected FlashInfer for five days, and its LSE base is unverified

Between `020c8e4` (2026-08-27) and `f8c442d` (2026-09-01 22:01), `auto`
preferred FlashInfer whenever it imported. That path **replaces the attention
call** — a different kernel with a different accumulation order — so it changes
the model's output, not just the score. Two compounding risks:

* `STICKYKV_FLASHINFER_LSE_LOG2` defaults to **`0`** (`flashinfer_lse.py:102`)
  while the module's own docstring (line 46) warns that *"some builds
  historically returned log2"*. If the installed build does, every `L` is off
  by a factor `ln 2`, every `exp(S − L)` is wrong, and **every eviction
  decision in the run is made on corrupted scores** — with no error, no
  warning, and plausible-looking output. The check exists
  (`tests/test_flashinfer_lse.py`) but needs a GPU and is skipped on the dev
  box.
* `f8c442d` already fixed the selection. **If the current numbers were produced
  before 22:01 on 2026-09-01, they are stale regardless of everything else in
  this document** — re-run before drawing any conclusion from them.

### D. The Triton prefill score kernel, and `exp2`

`c88581c` replaced the PyTorch reconstruction (bf16 softmax, fp16 accumulate)
with the fused Design-B kernel (fp32 throughout); `9a651ee` autotuned it;
`17d866f` switched `expf` → `ex2.approx`.

The `exp2` change reads correct: `scaling` carries `log2 e` from the dispatcher
(`score_kernel.py:562`), the kernel applies it to the `[BLOCK_M]` LSE vector and
not to the tile (`score_kernel.py:452-457`), and the identity
`exp(s·scale − L) ≡ exp2(s·scale·log2e − L·log2e)` holds. `ex2.approx` costs
~2 ulp into a sum that feeds a ranking. **It is the least likely cause in this
document** — but it is also the cheapest to eliminate
(`STICKYKV_SCORE_EXP2=0`), and it is where the user-visible framing points, so
rule it out by measurement rather than by argument.

The kernel *as a whole* is a different matter: like the decode kernel it ships
GPU-unvalidated, and it now carries two new autotune configs (`BLOCK_M=32/64,
BLOCK_N=256`) that no correctness run has exercised.
`STICKYKV_SCORE_AUTOTUNE=0` pins the fixed 64x64 path.

### Ruled out

* `STICKYKV_COMPILE_EVICT` defaults `0` (`cache.py:117`) and
  `STICKYKV_COMPILE_READ` defaults `0` (`quant/effective.py:68`) — the
  compiled-eviction work (`c14a1f5`…`3b311c5`) is not in a default run.
* The Inductor-lowerability rewrite (`_clamp_index`, `d44baba`/`4ee2aa0`) is
  semantically identical: `hi − relu(hi − x)` after `clamp_min(0)` reproduces
  `clamp(x, 0, hi)` on both branches, including `hi < 0`.
* Hyperparameters: `window_size`, `num_sink_tokens`, `local_window_size`,
  `cache_budget` and `first_eviction_step` are unchanged since 2026-07-27.

---

## 4. Step 0 — provenance, before anything else

Two facts decide which half of this document applies. Both are already on disk.

1. **`quant_ratio` in each run.** Read it from the sidecar JSON in each run's
   `output_dir`. If it is `0.0` in both, §2 is a no-op (`bytes` and `tokens`
   are bit-identical at `q=0`) and the whole investigation moves to §3 — start
   at B. If it is `> 0`, §2 is the headline and its magnitude is predicted by
   the tables above.
2. **`commit_sha` in each run.** Compare against `f8c442d` (2026-09-01 22:01)
   and `f71fec0` (2026-08-29). This dates each run against the two changes that
   matter and settles cause C outright.

Both fields are already written by `capture_environment()` /
`_write_metadata()`. If the current run predates `f8c442d`, stop and re-run it
at HEAD before proceeding — nothing else is worth measuring against a run whose
attention came from a different kernel.

---

## 5. The isolation protocol

**One dataset, not twelve.** `qasper` carries the largest drop (−9.52), runs
200 examples at ~3.6k tokens, and has `max_gen_length=128` — the cheapest
configuration that still exercises prompt compression, the Q tier and a real
generation. Add `multifieldqa_en` (150 @ ~4.6k, −4.85) as the confirmation arm
once a rung moves the number.

Everything below is an env A/B against one config. No logic edits; every rung
is a supported configuration.

Note the polarity: HEAD is now `bytes`, so **rung 0 is the candidate fix and
rung 1 reproduces the regression**. Rung 1 is what confirms the attribution —
without it, "the number came back" is a coincidence you have not ruled out.

| # | rung | change from HEAD | isolates |
|---|------|------------------|----------|
| 0 | baseline at HEAD (`bytes`) | none | should recover most of the 9.52 |
| 1 | `cache.quant_budget_mode: tokens` | budget semantics | **cause A** — must reproduce 32.78 |
| 2 | `STICKYKV_FUSED_DECODE=0` | materialize read instead of the fused kernel | **cause B** |
| 3 | `STICKYKV_LSE_BACKEND=flash` (explicit) | pins bit-identical attention | **cause C** |
| 4 | `STICKYKV_SCORE_EXP2=0` | `expf` instead of `ex2.approx` | **cause D** |
| 5 | `STICKYKV_SCORE_AUTOTUNE=0` | fixed 64x64 score tile | **cause D** |

Run rungs 1–5 **one at a time against rung 0**, not cumulatively — the point is
attribution, not recovery. Expected: rung 0 back at ~42 and rung 1 back down at
~32.8 if `quant_ratio > 0`; rungs 2–5 should each move the number by ~0.

If rung 0 does **not** recover and rung 1 does not reproduce, cause A is not the
story (check `quant_ratio` — at 0.0 the two modes are identical by construction)
and the answer is among B–D.

Two kernel-equivalence checks run first, and cost minutes rather than
GPU-hours:

```bash
python scripts/audit_e2e.py --config configs/longbench_ours_flash_attn.yaml --prefill 4096 --gen 64 --batches 1 --rungs 2_q70_materialize 3_q70_fused
```

Greedy decode is deterministic, so the two rungs must emit an identical token
sequence. A divergence *is* cause B, found without a LongBench run at all.

```bash
STICKYKV_LSE_BACKEND=flashinfer python -m pytest tests/test_flashinfer_lse.py -v
```

On the GPU box, against the installed build. A factor-`ln 2` mismatch *is*
cause C; the remedy is `STICKYKV_FLASHINFER_LSE_LOG2=1` — or, better, stay on
`flash`, which `f8c442d` already made the `auto` default.

And the score kernel's own oracle, which has never been run on a GPU:

```bash
python -m pytest tests/test_score_kernel.py tests/test_score_kernel_ab.py tests/test_score_exp2.py -v
```

`triton == token_scores_from_lse == token_scores_torch`. On CPU these skip the
kernel; on the GPU box they are the first real validation the prefill kernel
has had.

---

## 6. Order of work

1. ~~**Forward `quant_budget_mode`** and record it, plus the resolved geometry
   and the `STICKYKV_*` env, in the sidecar.~~ **Done** — see §2, "Landed".
2. **Read the two sidecars** — `quant_ratio`, `commit_sha`. Branch on §4. If
   the current run predates `f8c442d`, re-run at HEAD first.
3. **Run the three offline checks** in §5 (audit ladder, FlashInfer base, score
   kernel oracle). These are cheap and can invalidate a whole cause each.
4. **Run rungs 0 and 1 on `qasper`.** Rung 0 recovering ≥ 8 of the 9.52 points
   *and* rung 1 reproducing 32.78 is what confirms cause A. One without the
   other is not.
5. **Run rungs 2–5 on `qasper`** regardless of step 4's outcome. "Cause A
   explains it" is a hypothesis until the other four are measured at zero — the
   requirement is that efficiency must not cost accuracy, and that is a claim
   about each change, not about their sum.
6. **Confirm on `multifieldqa_en`**, then re-run the full twelve.
7. **Re-run anything that compared eager against flash at `quant_ratio > 0`**
   after 2026-08-29 — those two backends were on different budget semantics
   (§2, "Landed").

---

## 7. Acceptance gates

Nothing merges as "efficiency" until all four hold:

* **Score parity.** The Triton prefill kernel matches `token_scores_from_lse`
  on the GPU box, at every autotune config, with and without `exp2`.
* **Decode parity.** `audit_e2e.py` rungs 2 and 3 emit identical token
  sequences. Wire this into CI on the GPU box so the fused kernel cannot ship
  unvalidated a second time.
* **Operating-point parity.** Every quality row records `quant_budget_mode`,
  `retained_windows` and `retained_bytes`, and two rows are only ever compared
  at equal `retained_bytes / full_bytes`.
* **End to end.** LongBench at the recovered operating point is within noise of
  the 2026-08-13 numbers on all twelve datasets — *and* the prefill speedup
  from `17d866f` is still there. The speedup and the scores are not in tension:
  §2 is a budget-accounting change, not a compute one, and rungs 2–5 exist to
  prove the compute changes were free.
