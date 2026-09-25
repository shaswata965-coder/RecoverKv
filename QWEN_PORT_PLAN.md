# Qwen port plan — bring `int2_qwen`'s logic and structure onto `int2_clustered_rep`

## The LongBench-20% gap against `int2_qwen` (QEvict)

| task | QEvict | run 1 | run 2: key anchor (`efba5db`) | run 3: head-unit gate (`0db8eb6`) |
|---|---|---|---|---|
| trec | 67.00 | 64.50 | 62.00 | 62.50 |
| triviaqa | 89.63 | 82.40 | 79.97 | 81.40 |
| qasper | 35.74 | 35.44 | 36.96 | 36.83 |
| multifieldqa_en | 44.28 | 43.86 | 44.40 | 44.95 |
| hotpotqa | 49.15 | 48.53 | 47.92 | 48.76 |
| 2wikimqa | 37.43 | 39.31 | 36.55 | 36.83 |
| musique | 24.37 | 22.14 | 21.58 | 20.71 |
| samsum | 44.17 | 45.00 | 44.18 | 44.13 |

**Read the runs against their noise first.** n = 200 per task, so a task moves
~2.5-3 points between runs for no reason at all. Rounds 1 and 2 each fixed a
real defect and each moved nothing beyond that: across three runs the gap is
stable, confined to trec (-4.5), triviaqa (-8.2) and musique (-3.7), while
qasper, multifieldqa_en, hotpotqa, 2wikimqa and samsum sit at parity -- with
qasper and multifieldqa_en slightly ABOVE QEvict. Neither round's hypothesis
predicted anything going up.

### Round 11 — the native-RoPE arm came back identical to the YaRN arm: one of them did not run its config

| dataset | full (YaRN) | full_native | ours (YaRN) | ours_native |
|---|---|---|---|---|
| triviaqa | 87.43 | 87.43 | 81.15 | 80.85 |
| trec | 65.50 | 65.50 | 62.50 | 62.50 |
| qasper | 44.66 | 44.66 | 37.13 | 37.21 |
| musique | 27.72 | 27.72 | 20.78 | 21.79 |
| narrativeqa | 24.92 | 23.74 | 17.79 | 16.02 |

The FullKV rows are identical to the digit on four of five datasets. The one
that moved, narrativeqa, is the one whose prompts exceed the native arm's
31500-token truncation. A real RoPE change cannot leave 800 greedy
generations bit-identical, so **one of the two FullKV runs did not run the
positional setup its config names.** On this branch the two configs load
different models: with Qwen2.5-0.5B through the same loader, YaRN gives a
`yarn` rotary at a = 1.1386 and native a `default` one, and their last-token
logits on one prompt differ by up to 10 with a different argmax. So the
difference is in how the GPU box runs them (its local ablation scripts, its
model-path override, or a local checkpoint `config.json` carrying its own
YaRN), and the Round 10 question -- does YaRN handicap Qwen -- is NOT answered
by this table.

What it does say: in both pairs compression costs the same ~6 points
(49.81 -> 43.67 native, 50.05 -> 43.87 YaRN), so whatever RoPE actually ran,
it ran in both arms of each pair.

Changes so this cannot recur unseen:
- the LongBench sidecar records `model_load_requested` (the YAML's overrides)
  AND `model_rope_realised` (read off the loaded model's rotary module: type,
  attention_scaling, first/last inv_freq). A row whose two disagree is not the
  row its config names. Before this, nothing in a sidecar said which RoPE ran.
- the native configs are EXPLICIT (`max_position_embeddings: 32768`,
  `rope_scaling: {rope_type: default, factor: 1.0}`) instead of null, because
  null keeps whatever the checkpoint's config.json says -- and a local copy
  edited per Qwen's long-context instructions carries YaRN. Checked: on a
  YaRN-edited config the native arm now loads a `default` rotary at a = 1.0.

Next: the run logs already carry `Loading model: ... rope_scaling=...` for
each arm (utils/model_loading.py); grep them, and re-run the two native arms
on this commit so their sidecars carry `model_rope_realised`.

### Round 10 — every Qwen column runs static YaRN; published baselines do not

Published LongBench baselines on Qwen2.5-7B-Instruct (DefensiveKV and others,
as tabulated in the QEvict paper) score ABOVE this repo's uncompressed Qwen
column even at 20% compression. So the handicap is not only in the compressed
rows: it is in the setup every Qwen row shares.

Every Qwen config since Round 3 runs static YaRN x4 with
`max_position_embeddings: 131072` (QEvict's protocol). Measured against the
checkpoint's native RoPE (rope_theta 1e6, window 32768, no scaling):
- every logit is multiplied by a^2 = 1.296 (YaRN's attention_scaling folded
  into cos/sin) -- sharper attention at every length;
- 33 of 64 RoPE frequency pairs are changed, 17 of them stretched 4x; at a
  6000-token distance 8 pairs' angles move by more than 0.5 rad;
- on transformers 4.47.1 `original_max_position_embeddings` is ignored, so this
  is not even Qwen's documented YaRN (which changes 40 pairs).

Qwen's model card advises adding rope_scaling only when long context is
needed, because static YaRN can degrade shorter inputs -- and LongBench's
prompts fit the native 32K window after THUDM's standard truncation. It also
plausibly makes COMPRESSION cost more on Qwen than on Llama (Llama-3.1's rope
has attention_factor 1): the 1.296 logit scale multiplies every int2 key error
in nats by 1.3, and the bf16 rounding of a = 1.1386 is what leaves layer 0's
int2 tier +0.17 nats hot (Round 8). With a = 1 both go away.

Arms (`scripts/run_longbench_qwen_ablation.sh`, default now
`ARMS="full_native ours_native"`): `longbench_qwen_full_cache_native.yaml` and
`longbench_qwen_ours_native.yaml`, the same two rows on native RoPE with
prompts middle-truncated to 31500. What they say:
- full_native well above full -> the YaRN setup handicaps every Qwen column,
  QEvict's included; re-baseline Qwen on native RoPE.
- ours_native within ~1 of full_native (Llama's cost) -> YaRN was also why
  compression cost 6 on Qwen. Still ~6 -> the int2 tier on Qwen is next
  (`q0`).

### Round 9 — both GPU kernels verified against PyTorch: the machinery matches QEvict everywhere it has been measured

`scripts/check_fused_vs_torch.py`, gate100 config, triviaqa, GPU:
- **Prefill score kernel vs an exact causal softmax sum, 84 scoring calls:**
  relative L1 median 1.5e-6 (worst 5.7e-6); top-20% window Jaccard 1.000 on
  every call. The kept set is exactly what exact attention would choose.
- **Fused decode kernel vs PyTorch attention over the same cache:** median
  1.8e-3 to 4.8e-3 per layer (bf16/TF32 rounding), p90 under 1.4e-2 except
  layer 0 (median 1.3e-2, p90 7.1e-2, max 1.0; layer 27 max 0.46). The layer-0
  tail is the REFERENCE's rounding: `effective_q_tier` returns bf16 keys, and in
  layer 0's massive static channels (~230 post-RoPE) bf16 steps by 1.0. Read at
  fp32, the store's own layer-0 int2 error drops from 0.169 / 0.855 (median /
  p90) to 0.132 / 0.747 while layers 14 and 27 do not move. The kernel reads at
  fp32 (TF32), so at layer 0 it is closer to exact than a bf16 materialize path
  -- QEvict's -- not further.

So every component that selects or reads the cache has now been measured equal
to `int2_qwen`'s or to exact attention: prefill scores, policy, schedule
(`should_evict` is the same code), geometry, int2 storage, the decode read.
And the column still sits ~6 points under full where QEvict's does not. The
remaining unmeasured difference is **the QEvict column itself**: it was not
produced in this environment. Its pattern is that of a column whose short
answers were produced at FULL cache -- above this runner's full column on
triviaqa (+2.2) and trec (+1.5), where answers are a few tokens, and BELOW
this branch on qasper (35.74 vs 37.13), where answers run to 128 tokens.
`first_eviction_step > 0`, a different runner, or a different commit would
each do that. Its `<dataset>.meta.json` sidecars record the invocation
(`first_eviction_step`, `quant_ratio`, `cache_budget`, the resolved geometry);
read them before any more machinery work, and re-run `int2_qwen` on this
runner if they do not match.

### Round 8 — gate 1.0 is QEvict's read path and still loses: the gap is the rest of the machinery

GPU, 200 examples per dataset, commit `bce166c`:

| dataset | full | ours (gate 0.25) | gate100 (1.00) | QEvict |
|---|---|---|---|---|
| triviaqa | 87.43 | 81.15 | 79.94 | 89.63 |
| trec | 65.50 | 62.50 | 62.50 | 67.00 |
| qasper | 44.66 | 37.13 | 37.64 | 35.74 |
| musique | 27.72 | 20.78 | 21.65 | 24.37 |
| narrativeqa | 24.92 | 17.79 | 17.66 | -- |
| average (5) | 50.05 | 43.87 | 43.88 | |

At gate 1.0 every int2 window is read exactly -- QEvict's read path -- and the
column does not move (43.88 vs 43.87). So the Qwen gap to QEvict is in what the
two branches do NOT share, and it is not the gate. The same cache costs Llama
~1 point against its own full column; it costs Qwen 6.

**Ruled out on CPU, both branches' OWN code on the same real Qwen2.5-7B
activations** (4 LongBench prompts, ~6K tokens, all 28 layers;
`scripts/measure_gate_real_cpu.py --dump-dir`, `scripts/compare_int2_store.py`
run once per branch):
- scoring semantics: both sum `softmax(q k^T)` over prefill query rows per
  window from q_proj's output (bias included) + RoPE;
- the eviction policy: identical code (rank by mean-over-heads score, top k_fp
  fp, next N_q int2);
- the geometry: same `top_k_fp`, `N_q` within 1.7% (the card is priced in);
- **the int2 storage**: the one-byte grid + key anchor here against
  `int2_qwen`'s per-entry bf16 grid, each through its own unrotate ->
  `demote_many` -> `effective_q_tier`, give the same attention output error to
  three digits at every layer (pooled median 0.0354 vs 0.0356, p90 0.323 vs
  0.325), the same logit bias and the same int2-tier mass shift (+0.107 vs
  +0.108 nats median).

**Found, shared by both branches, so not the gap:** YaRN folds a = 1.1386 into
cos/sin and HF hands them back in bf16, where 1.1386 rounds to 1.140625. The
int2 path un-rotates with the rounded tables, divides out the EXACT a^2, and
re-rotates with the rounded tables, so an int2 key comes back ~0.36% large in
the static RoPE pairs -- where Qwen keeps its largest key biases (206 per pair
at layer 0). At layer 0 that is a +0.17 nat logit bias on every int2 token
(both branches); elsewhere it is under 0.05. Llama is fp16 with a = 1 and has
none of it. Cheap to fix later (divide by the rounded a^2, or un-rotate in
fp32); it does not explain the difference between the two columns.

**What is left runs only on the GPU**, and nothing has ever checked it against
anything but itself:
- the fused two-tier decode kernel (bf16 inputs, the KEY_ANCHOR and rep-7
  specializations): every GPU check compared it with the same kernel at another
  gate ratio;
- the Triton prefill score kernel, which decides the kept set from flash's
  captured softmax normaliser L -- a mismatched L (layout, scale, the GQA
  expansion Qwen2FlashAttention2 does before flash) changes every eviction and
  raises nothing;
- the compiled eviction and the decode-time promote/demote dynamics.

`scripts/check_fused_vs_torch.py` checks the first two against PyTorch on real
decode and prefill steps (minutes); the `q0` arm separates the int2 tier from
the eviction.

### Round 7 — the gate sweep is flat: the gate is NOT the Qwen gap

LongBench, Qwen2.5-7B, flash, q = 0.70, budget 0.20, GPU; the realised read
fraction was checked against each ratio (the knob is live), and 0.10 was run
too and is flat with the rest:

| dataset | gate 0.25 | gate 0.50 | gate 0.75 |
|---|---|---|---|
| triviaqa | 81.40 | 80.39 | 79.95 |
| trec | 62.50 | 63.00 | 62.50 |
| hotpotqa | 48.76 | 46.95 | 48.03 |
| multifieldqa_en | 44.95 | 45.86 | 44.25 |
| samsum | 44.16 | 43.45 | 43.81 |
| qasper | 36.83 | 37.09 | 37.44 |
| 2wikimqa | 36.52 | 37.67 | 37.02 |
| musique | 20.71 | 20.65 | 20.69 |
| **average (8)** | **46.98** | **46.88** | **46.71** |

The GPU diagnostic (same config, same four gap datasets) puts the gate's output error at 0.75 about
10x below 0.25 (median ~0.001-0.003, p90 0.01-0.03), and 1.00 reads exactly
0. Cutting the gate's error tenfold moves no task beyond noise, and none of the
three gap tasks (triviaqa, trec, musique) moves toward QEvict. **The gate is
ruled out as the cause.** Round 6's measurements stand as measurements -- the
card IS less accurate on Qwen and in its last layers -- but that error does not
cost LongBench points. Rounds 2, 4, 5 and 6 all pointed at the gate; the sweep
is what refutes them.

What is left is anything that does not depend on the gate ratio OR the budget
(triviaqa is ~81 at every budget 5-20% and every ratio 0.10-0.75, against
QEvict's 89.63): the protocol, the eviction/fp path, or the int2 tier itself
(one-byte grid + anchor, fused two-tier kernel). Two config arms separate them
(`scripts/run_longbench_qwen_ablation.sh`, default `ARMS="full q0"`):
- `full` (no compression) on this branch's runner. If it scores ~81 on
  triviaqa, the gap is not the method: the runner/protocol differs from the one
  QEvict's column was produced with, and every earlier round was chasing
  machinery.
- `q0` (same eviction, quant_ratio 0.0: no int2 tier, no card, no fused
  kernel). q0 ~ full -> the int2 tier is the gap; q0 ~ ours -> the eviction
  or the fp path is.

### Round 6 — measured on the real models: why Qwen and not Llama

Real weights, real LongBench prompts (triviaqa x2, narrativeqa, musique at
~6K tokens), every layer, on CPU (`scripts/measure_gate_real_cpu.py`: weights
streamed by range reads, linears bf16, attention fp32). Each layer's cache is
laid out as after the first eviction -- tier sizes from `resolve()` at the
operating point, ranking by the policy's prefill scores -- and the repo's own
`build_sketch` / `gate_reference` / decode oracle (with centroids) are compared
against reading every window. Keys are not int2-quantized in either arm, so the
difference is the gate alone. Shipped settings: head-unit union on Qwen, raw
on Llama. Llama-3.1-8B from the ungated `unsloth` copy of the same weights.

**The harness reproduces the GPU diagnostic on Qwen** (GPU at 0.25: triviaqa
recall 0.846, output-error p90 0.10-0.14 on triviaqa/narrativeqa/musique and
0.22 on trec; CPU: 0.839 and 0.089 pooled), so its Llama column is a fair
comparison.

At `quant_gate_ratio` 0.25, per (query head x query x layer x prompt):

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| output error, median / p90 | 0.005 / 0.089 | 0.008 / 0.072 |
| heads with output error > 0.3 | 2.1% | 1.3% |
| recall: solo oracle -> shared oracle -> card (median) | 0.951 -> 0.905 -> 0.839 | 0.867 -> 0.832 -> 0.772 |
| same, p10 | 0.788 -> 0.691 -> 0.569 | 0.700 -> 0.644 -> 0.533 |
| card_tv median / p90 | **0.274 / 0.509** | 0.170 / 0.304 |
| skip_mass_err median / p10 (nats) | **-0.42 / -1.51** | -0.20 / -0.64 |
| **last two layers: output error p90** | **0.206** (4.6% > 0.3) | 0.068 (0.3% > 0.3) |

What it says:
- **Averaged over layers, the gate is barely worse on Qwen** (p90 0.089 vs
  0.072), and Qwen's selection recall is HIGHER than Llama's -- its attention
  over the int2 tier is more concentrated. Average error does not explain why
  one model loses points and the other does not.
- **GQA sharing is real but secondary.** Sharing one selection across the
  query heads costs 0.046 median / 0.097 p10 of recall on Qwen (rep 7) against
  0.035 / 0.056 on Llama (rep 4).
- **The card is ~2x worse on Qwen.** Its weights over the unread windows are
  further from the truth (TV 0.27 vs 0.17), and it UNDER-weights them twice as
  hard: in the worst decile of heads they get e^-1.5 = 22% of their true weight,
  against 53% on Llama.
- **And the error lands in Qwen's LAST layers**, the ones that write the next
  token: layers 26-27 have output-error p90 0.20-0.21 (Llama's last two:
  0.09 / 0.05), card TV 0.35-0.39, and a p10 skip_mass_err of -2.0 / -2.2 --
  unread windows get ~11% of their weight. Their card recall p10 is 0.54-0.55
  against 0.72-0.76 for a perfect card at the same sharing, so there the card,
  not the sharing, is what loses. Qwen's layers 0-2 are also bad (p90
  0.31-0.44), but Llama's layer 0 is worse (1.0) and Llama is fine, so early
  layers are tolerated.
- Layer 27 carries the model's largest key bias (511 per RoPE pair, 920 in
  total) and a query bias with a third of its energy rotating inside a window;
  layer 0 is second. But the biases are almost all in STATIC RoPE pairs (93-100%
  of the big ones' energy), so the card's post-RoPE anchor cancels them, and
  layer 26 is bad with a small bias -- the card error there is not the anchor.
- **At 0.5 Qwen's last layers fall to 0.064-0.072 p90, Llama's level at
  0.25**, and the pooled share over 0.3 to 0.5%. 0.5 is under the 0.66
  break-even. The `gate50` arm tests whether that recovers the scores.

**Correction to Round 5:** `skip_mass_err` is NEGATIVE on both models -- the
unread windows are under-weighted, not over-weighted, so the output is too
concentrated on the read windows and the fp tier, not blurred toward the tier
mean. The inverse-budget pattern is not explained by this measurement; with
n = 200 it may be partly noise (narrativeqa's -5.4 is ~2 sigma).

### Round 5 — the budget sweep says the loss scales with the UNREAD windows

Three rounds fixed real defects and none moved the column beyond noise,
because each was chosen from a synthetic argument and none was checked against
the one run that attributes the gap (the gate's control arm, below). The
budget sweep is the first new evidence since, and it narrows things. Flash,
q = 0.70, head-unit gate + key anchor on; the 20% column reproduces run 3 (six
of eight tasks to the digit, the other two within 0.31):

| dataset | 20% / l128 | 10% / l128 | 5% / l32 |
|---|---|---|---|
| triviaqa | 81.40 | 81.90 | 81.66 |
| trec | 62.50 | 62.25 | 58.50 |
| hotpotqa | 48.76 | 49.84 | 47.60 |
| multifieldqa_en | 44.95 | 41.08 | 36.31 |
| samsum | 44.16 | 43.72 | 33.14 |
| qasper | 36.83 | 34.57 | 32.58 |
| 2wikimqa | 36.52 | 36.75 | 38.01 |
| gov_report | 28.06 | 30.22 | 26.07 |
| multi_news | 25.29 | 23.39 | 24.24 |
| qmsum | 21.95 | 22.31 | 21.54 |
| musique | 20.71 | 23.51 | 25.15 |
| narrativeqa | 18.28 | 23.69 | 22.68 |
| **average (12)** | **39.12** | **39.44** | **37.29** |

What it says, against n = 200 noise of ~2.5-3 points per task:
- **Doubling the budget from 10% to 20% buys nothing on average** (-0.3).
  The tasks that should use the extra cache do (multifieldqa +3.9, qasper
  +2.3); the longest-context ones LOSE with it -- narrativeqa -5.4 (about
  twice the noise), musique -2.8 (monotone over all three budgets),
  gov_report -2.2.
  Retaining more windows making an answer worse means the retained windows are
  being read wrongly, and the longest prompts have the most of them.
- **triviaqa is flat at ~81.6 from 5% to 20%**, 8 points under QEvict's 20%.
  Its loss does not depend on how much is kept, so it is in how every step
  reads, not in what eviction chose.
- At q = 0.70 ~88% of retained windows are int2 and the gate reads 25% of
  them, so **~66% of what the cache retains reaches attention only as a value
  centroid at a card-estimated weight.** That share is fixed across budgets;
  the NUMBER of such windows roughly doubles from 10% to 20%. An error per
  unread window -- the card's weight, or the centroid standing in for the
  window -- grows the way the long-context tasks fell. int2 noise also grows
  with window count, and QEvict carries the same noise; it scores higher at 20%
  on these tasks (musique 24.37), but whether ITS long-context scores also
  fall with budget is unmeasured -- QEvict's 10% column would settle that.

So the next measurement is the attribution, and nothing about the method changes
until it has run:
- `ARMS="gate100 gate50" scripts/run_longbench_qwen_ablation.sh` -- trec,
  triviaqa, musique, narrativeqa, qasper (control) at every window read
  (1.0) and at half (0.5). The 0.25 row is the table above.
  - gate100 recovers, gate50 recovers most: the selection's CAPACITY. rep 7
    shares one selection among seven query heads where Llama shares it among
    four; the ratio should scale with rep. Still under the 0.66 break-even.
  - gate100 recovers, gate50 does not: the CARD. The unread windows' weight or
    centroid is wrong on Qwen, and reading more of them only shrinks the error.
  - gate100 does not recover: the gate is exonerated. The rest of the machinery
    (one-byte grid + anchor, Triton score/decode kernels) is next.
- `scripts/diagnose_gate_error.py --dataset narrativeqa` (and triviaqa), with
  and without `--override cache.quant_gate_ratio=0.5`, minutes each. New this
  round: **`skip_mass_err`**, the log ratio of the unread windows' total weight
  as the kernel fills it to their true total, relative to the read windows'
  (pinned to the oracle's applied fill in `tests/test_gate_diagnosis.py`), and
  `skip_share`, the head's attention that rides the centroid path. `card_tv`
  measures the card's shape over the tier and cannot see a wrong TOTAL, which
  is what pulls the output toward the tier's mean value. On the plain synthetic
  fixture it already reads -0.32 nats median at 0.25 with a p10-p90 of -1.26 to
  +0.39: the fill is right on average and off by e^1 for some heads even on
  benign keys.

If gate100 recovers the column, `quant_gate_ratio: 1.0` is the immediate Qwen
setting (CLAUDE.md: 1.0 ties 0.25 in TPOT, `130de24`) while the card is fixed
against the diagnostic's numbers.

### Round 4 — where the gap can still be: the decode READ

Established with the operating point and protocol equal (both columns
q=0.70, bf16, YaRN 128K, untruncated):
- **The first generated token is the same** on both branches: it comes from the
  prefill forward, before any eviction.
- **The kept set is the same** from the first eviction on: it is ranked by the
  prefill scores (same H2O semantics; the Triton score kernel was EXECUTED under
  Triton's interpreter against its reference at Qwen's shapes, rep 7: 1e-6 in
  fp32), and 32-64 decode steps barely move a cumulative score summed over
  thousands of prompt queries.
- So tokens 2+ differ only in **how each decode step reads** the kept cache:
  - this branch reads 25% of the int2 tier exactly and the other 75% through
    each window's card (estimated weight) and value centroid; QEvict reads every
    window exactly;
  - kernel numerics are not worse: the fused kernel dots in fp32 (TF32 on A100
    -- exact for bf16 inputs, finer than the bf16 cast the materialize path
    applies to dequantized keys); the int2 storage is equal or better (above).
- Rounds 1 and 2 changed the precision of the windows READ and WHICH windows are
  read. Neither touched the 75% approximated through their cards.

**Why the card may be the Qwen-specific part (synthetic, CPU, magnitudes
guessed):** a card models a window as one mean plus one deviation direction.
A q_proj bias makes some query channels large, and then a key's content in
those channels dominates the logit -- content a rank-1 card does not capture.
On keys + queries with a Qwen-style bias through the real Qwen2 RoPE, the
card's per-window log-mass error and its misallocated share of each head's
attention (TV):

| bias (low-frequency RoPE band) | median error | TV |
|---|---|---|
| none | 0.28 nats | 0.09 |
| k 20 / q 5 | 0.63 | 0.21 |
| k 60 / q 10 | 1.42 | 0.56 |
| k 150 / q 20 | 3.49 | 0.89 |

int8 `mu` gives the same numbers in this band (in a mid-frequency band int4 is
worse, e.g. median 3.34 vs 1.43 at k 60). The per-head union (Round 2) cannot
help an estimate that is wrong. **Whether Qwen2.5's real biases are in this
range is exactly what is not known here.**

**Two ways to settle it on the GPU:**
- minutes: `python scripts/diagnose_gate_error.py --config
  configs/longbench_qwen_ours_flash_attn.yaml --dataset triviaqa` (and the same
  with the Llama config). Per layer: the gated step's output error against the
  full read of the same step, the card's TV against the true window weights,
  the selection's recall, and the int2 tier's share of attention
  (`modules/windowed_cache/gate_diagnosis.py`). A large `card_tv` on Qwen and a
  small one on Llama is this mechanism.
- one quality run: `ARMS=gate100 scripts/run_longbench_qwen_ablation.sh trec
  triviaqa musique` -- every window read exactly, same code otherwise.
  Predicted: trec/triviaqa/musique back within noise of QEvict. If so,
  `quant_gate_ratio: 1.0` is the immediate fix for Qwen (CLAUDE.md: 1.0 ties 0.25
  in TPOT today) and a card that models the large-query channels is the real
  one. If not, the gate is ruled out and the one-byte grid is next.

### Round 3 — the operating point (RETRACTED: the premise was wrong)

Round 3 read `int2_qwen`'s YAML (`quant_ratio: 0.5`) as the split the QEvict
column was measured at, and moved this branch's Qwen configs to 0.5 for one
commit (`1db0488`). **It was not:** the QEvict column was run at
`quant_ratio 0.70`, and the Current columns with the `_yarn128k` config (bf16 +
YaRN 128K, untruncated). The two columns share the operating point AND the
model setup. The q=0.5 change is reverted; the lesson is in CLAUDE.md -- ask
how the reference was actually invoked before reading its config.

What still stands from Round 3's side-by-side checks, and so is ruled out:
- the budget split resolves the same geometry on both branches at equal q
  (identical fp window count, int2 within 1.5% -- the gate's card is priced in);
- the eviction policy and the scorer are semantically identical (the diff is
  compile refactoring plus the gate's skipped-window fill);
- the LongBench runner treats every dataset identically (few-shot template
  skip, no BOS on Qwen, first-line extraction);
- values are quantized the same way (per token over 128 channels), and the
  one-byte grid stores a value zero to the same ~0.4% as QEvict's per-entry
  bf16 grid; the key zero is exact for a bias channel here (anchor) where
  QEvict's bf16 zero rounds it.

Kept from Round 3 because it is right independently: every Qwen config is
bf16 + YaRN 128K (the default-named files were fp16 at 32K under the names
`int2_qwen` uses for the YaRN setup), `_yarn128k` is an alias of the default,
and the operating-point checker's three Qwen divergences are exempted.

**So the gap is the machinery.** Same operating point, same protocol, same kept
set up to the first eviction (prefill scores decide it; decode-time additions
barely move H2O's cumulative ranking over 32-64 generated tokens). What
differs is how a decode step READS the kept cache: this branch reads 25% of the
int2 tier exactly and the other 75% through each window's card (estimated mass)
and value centroid, where QEvict reads every window exactly -- plus the Triton
prefill score kernel and fused decode kernel in place of PyTorch. Rounds 1 and 2
changed the precision of the windows that are READ and WHICH windows are read;
neither touched the approximation of the 75% that are not, which is the largest
remaining difference. `longbench_qwen_ours_gate100.yaml` (every window read,
same code) isolates it: `scripts/run_longbench_qwen_ablation.sh`.

### Round 2 — the read gate's GQA union over raw logits (fixed: a real defect, not the gap)

Run 3 is this fix: triviaqa +1.4, trec +0.5, musique -0.9 against run 2 --
inside the noise. The defect is real and stays fixed; the account below of
why it would explain the Qwen gap was wrong.

QEvict has no read gate: it dequantizes the whole int2 tier on every step, so
every query head attends to its own hot tokens exactly. Here the gate reads
`quant_gate_ratio` = 25% of the tier per **KV head**, chosen by a max over the
`rep` query heads sharing it (`group_max`), and the other 75% attend only through
each window's value centroid. A max over query heads is only a union if the
heads are in the same units, and raw logits are not: each head carries its own
constant `q_h . k_common` (its query's projection onto the keys' common mode),
which its own softmax discards and a cross-head max does not. Whichever head has
the largest offset picks the whole group's windows; its siblings' hot windows are
read only as centroids -- the mean of 8 values, which for a head attending to one
token (retrieval, induction, answer copying) is the wrong vector.

Why Qwen2.5 and why these tasks:
- **rep = 7** (28 q / 4 kv heads) puts seven heads behind each union, not four.
- **q/k biases** make the offsets larger by construction (a token-independent
  key component each query projects onto differently), and through the
  mid-frequency RoPE pairs they also put large smooth positional swings on some
  heads' logits, which outvote a sibling's content match the same way.
- **The Q tier is ~88% of the retained windows** at q = 0.70.
- The tasks that fell are the ones that need exact in-context retrieval
  (trec's label copying, triviaqa's few-shot answers, musique's multi-hop);
  summarization (samsum) and single-doc QA did not move.
- The key anchor could not help: it made the READ windows' logits exact, and for
  six of seven heads the windows that mattered were not read.

It was invisible because the test that set the 0.25 operating point
(`test_sketch.py::test_recall_at_the_shipped_25_percent_ratio`) pools raw
`exp(logit)` over a KV head's query heads before measuring recall -- a quantity
dominated by the same head the raw union serves.

**Measured (CPU), on that test's own fixture** (keys with a massive-activation
common mode, one hot token per window; 271 windows, 8 seeds), share of each
query head's OWN Q-tier attention mass in the windows read:

| ratio 0.25 | raw union: mean / p10 / worst | head units: mean / p10 / worst |
|---|---|---|
| rep 4 (Llama-3.1 GQA) | 0.947 / 0.966 / **0.000** | 0.9999 / 0.9998 / 0.998 |
| rep 7 (Qwen2.5-7B GQA) | 0.857 / **0.164** / **0.000** | 0.9997 / 0.9992 / 0.998 |

Attention OUTPUT, through the decode kernel's CPU oracle
(`two_tier_window_reference`: skipped windows attend via centroids at the
calibrated weight) against a full read, synthetic Qwen-like groups (rep 7,
per-head offsets sd 40 nats, retrieval / moderate / diffuse heads), error in
units of |v|, mean over heads:

| union | retrieval heads | all heads |
|---|---|---|
| raw logits (shipped) | 0.433 | 0.231 |
| minus `q . anchor` only | 0.004 (0.393 with positional-swing heads) | 0.042 |
| **head units (fix)** | **0.004** (0.036 with positional-swing heads) | **0.041** |

Subtracting the anchor term alone fixes the constant offset but not the
positional swings; per-head normalization fixes both.

**Fix -- `quant_gate_head_norm`.** Before the group max, each query head's
estimate is taken relative to that head's own Q-tier log-partition
(`sketch.in_head_units`: `est - logsumexp_w logmass`), i.e. heads are compared
in probability, the only unit they share. Within a head it is a constant shift,
so each head's own ranking, every `rep == 1` selection, `logmass` (what skipped
windows score and attend with), the card, the budget and `N_q` are all
unchanged; `quant_gate_ratio = 1.0` still selects everything. On the flash path
the scan kernel gains a `HEAD_NORM` constexpr (per-head `est` + per-tile
`(max, sum)` partials) and a small `_gate_union_kernel` merges the partials and
takes the max: **one extra launch per layer per step, Qwen only, zero bytes per
window**; without it the scan runs the arithmetic it always did.

- **Auto-on where the attention projections carry a bias** (the same
  `key_projection_has_bias` test as the key anchor): Qwen2.5 on, Llama-3.1 and
  Mistral off, so their columns are byte-identical. Every runner threads the YAML
  knob (`cache.quant_gate_head_norm: null|true|false`) and the LongBench sidecar
  records it (`resolved_geometry_first_example.quant_gate_head_norm`). Every
  runner's `read_gate` record also says which union the layers that actually
  ran used (`"union": "per-head" | "raw" | "mixed"`); **a Qwen2.5 row whose
  `read_gate.union` is not `per-head` is not this method.**
- **Llama is affected too** (rep 4 row above: worst head 0.000). It stays off
  there only because CLAUDE.md forbids moving the Llama/Mistral scores without a
  LongBench run. To measure it: `quant_gate_head_norm: true` on the Llama config.
- **Verified:** both Triton kernels compile for sm80 in both specializations
  (`test_kernel_compiles.py`) and were **executed** under Triton's CPU
  interpreter against `gate_reference`, selection identical at rep 3/4/7 with
  window counts off every tile boundary (`test_gate_head_units.py`). The fused
  decode kernel that consumes the selection -- including round 1's `KEY_ANCHOR`
  -- was executed the same way against `two_tier_window_reference` on a real
  Qwen2 cache after eviction: 7e-6 relative output error in fp32, anchored and
  not, gated and not (`test_decode_kernel_interpreted.py`; swapping the anchor's
  halves takes it to 1.5). So round 1's kernel change is correct as written and
  is not what moved the second run down. Not yet run on a GPU, and the LongBench
  effect is not yet measured.

**The decisive check on the next GPU run** is the gate's own control arm:
Qwen2.5 at `quant_gate_ratio: 1.0` (every window read -- QEvict's read path) next
to the default 0.25 with this fix. If 1.0 recovers the QEvict column, the gate
was the gap and 0.25 in head units should sit close to it; whatever 1.0 does NOT
recover is elsewhere -- the remaining differences from QEvict are the operating
point (q = 0.70 vs 0.5: 70 vs 117 fp windows at a 10K prompt) and the YaRN
flag below. Qwen2.5 quality rows do not compare across this commit.

### Round 1 — the key zero-point anchor (kept: correct, but not the gap)

Diagnosed first, and it is a real defect, but the run after it (table above)
shows it was not what separated the columns.

**Defect — the one-byte grid (`8a8cdc0`) cannot hold Qwen2's biased keys.**
Qwen2's `k_proj` has a bias, which puts a large, near-CONSTANT value on some key
channels: pre-RoPE, such a channel's per-window range is ~0 and its zero-point
(the window min) is the bias, hundreds. `int2_qwen` stored the zero per entry in
the KV dtype, which holds a KV-dtype number exactly. The one-byte grid stores it
as int8 x a scale shared by 32 channels, i.e. only to within `max|zero|/254`.
That error is the SAME in every window (same bias, same code), so it is not
noise: times the query's own massive value in that channel it shifts every int2
logit against every fp logit by several nats, scaling the whole tier's attention
mass (~88% of the retained windows at q=0.7) up or down by orders of magnitude.
Measured on synthetic Qwen-like keys (`tests/test_key_anchor.py`): 6 bias
channels of |40..240| per head in the low-frequency RoPE band, taken through
the real demotion round trip (Qwen2 rotary, rope_theta 1e6, rotate + unrotate
in the cache dtype), mean tier-wide logit shift over 5 seeds, in nats:

| grid | bf16, q +20 | bf16, q +150 | fp16, q +20 | fp16, q +150 |
|---|---|---|---|---|
| one-byte (as shipped) | 0.95 | 7.1 | 1.18 | 8.8 |
| per-entry (`int2_qwen`) | 0.035 | 0.24 | 0.017 | 0.12 |
| **anchored (fix)** | **0.010** | **0.061** | **0.012** | **0.078** |

("q +20" = the query's added magnitude in the bias channels.) Per-key noise is
the same in every row; the one-byte grid adds a BIAS, not noise, which is why no
aggregate error metric saw it. Llama-like keys (no bias): 0.02 either way —
which is why the one-byte grid passed its own validation, and why nothing on
the Llama column ever showed it.

**Fix — the key zero-point is stored relative to a frozen anchor**
(`quant_key_anchor`). `QuantizedStore` freezes a per-(row, head, channel)
pre-RoPE key mean at the first demotion (the same idiom the gate card already
uses for `mu`), the quantizer encodes `zero - anchor`, and every reader adds it
back: `promote_many`, `effective_q_tier`, `gated_q_tier`, and the fused Triton
kernel (`KEY_ANCHOR` constexpr, one [H_kv, D] load per program). A bias
channel's residual is then exactly 0 and round-trips bit-exact; tier shift
0.01 nat. **Zero bytes per window** — the budget, N_q and every compression
ratio are unchanged. **Auto-on only where `k_proj` has a bias**
(`key_projection_has_bias`: Qwen2 yes; Llama-3.1 `attention_bias=False`, Mistral
no), so the Llama/Mistral columns are byte-identical; the un-anchored kernel
compiles exactly as before. The sidecar records it
(`resolved_geometry_first_example.quant_key_anchor`).

**Two smaller bf16 defects, fixed alongside** (both no-ops on fp16):
- The slot table stored every grid group scale as fp16 regardless of the grid
  dtype, and `write` casts to the buffer: a bf16 cache's grid was fit to a bf16
  scale and read back from an fp16 copy, so the Stage-5 "grid follows the KV
  dtype" fix held inside the quantizer and was undone at storage. The table now
  allocates `grid_dtype_for(kv_dtype)`.
- Window scores accumulated in the KV dtype: in bf16 ~76% of per-step
  contributions vanish (`int2_qwen` `446a7eb`, which QEvict had). Scores are now
  produced — hence accumulated — in fp32 on a bf16 cache
  (`hooks.score_accum_dtype`); fp16 keeps fp16, byte for byte (widening it is
  still the Stage-6 item and moves Llama/Mistral).

**Not changed, flagged:** the Qwen configs (all of them since Round 3) set
`max_position_embeddings: 131072`. On the pinned transformers 4.47.1,
`_compute_yarn_parameters` takes the YaRN correction range from
`max_position_embeddings`, so this is NOT Qwen's YaRN (which uses the original
32768, what Qwen's own config.json keeps): 23 of 64 RoPE frequency pairs differ
by >1%, up to 2.2x at pair 40. `int2_qwen`'s config carries the identical
override, so it does not explain the gap and changing it would move the
column off the protocol QEvict was measured at; fix it for both together.

Status: **CPU-validated** (mechanism, fix, store/cache/kernel plumbing, and all
32 decode-kernel specializations compile for sm80 via `test_kernel_compiles.py`);
the Qwen2.5 LongBench number needs a GPU run. Quality rows for Qwen2.5 do not
compare across this commit.

## Execution status (as-built on `int2_clustered_qwen`)

Implemented and CPU-validated (torch + transformers 4.47.1) — `test_qwen_port.py`
(17), `test_qwen_numerics.py` (7), `test_quant.py`, `test_prompting.py`,
`test_windowed_cache.py` all green except the repo's pre-existing stale tests
(they import the deleted `windowed_eager_cache`; CLAUDE.md documents that
backlog). A live Qwen run needs a GPU + the checkpoint and was not possible in
the web container.

- **Stage 1** ModelConfig overrides + dtype hardening — done.
- **Stage 2** `utils/model_loading.py` shared load path — done.
- **Stage 3** `utils/generation.py` + BOS-aware prompting — done.
- **Stage 4** YaRN `a²` RoPE inverse — done. Only the two PyTorch unrotate sites
  needed it (`effective.unrotate_key_window`, `state.rerotate_keys`); the fused
  kernel re-rotates only, so it is correct for free — pinned by the YaRN
  round-trip test. **There is no eager twin to mirror**: `windowed_eager_cache`
  was deleted on this branch (`0974687`), so the plan's "×2" sites collapse to one.
- **Stage 5** grid dtype — **adapted**. This branch stores a *one-byte* grid
  (`QGrid`, commit `8a8cdc0`), so the fp16→inf overflow is on the grid's fp16
  *group scale* in `_quantize_grid`, not a flat scale/zero pair. The KV dtype is
  inferable from `x.dtype` inside `_affine_quantize`, so `grid_dtype_for` threads
  in with **no** store/slots/sketch changes. fp16 stays byte-identical.
- **Stage 6-arg** forward-arg-index fix — done. **Stage 6-accum (fp32 window-score
  accumulation) DEFERRED**: this branch accumulates in the KV/query dtype, so
  moving to fp32 changes the existing Llama/Mistral scores, which CLAUDE.md
  forbids without a LongBench run behind it (not runnable here). It is a quality
  refinement, not a correctness blocker; revisit with a GPU LongBench validation.
- **Stage 7** runners routed through the shared loader + `suppress_tokens` — done.
  The samsum-EOS / max_length / eager-OOM-ceiling refinements were **skipped**:
  they risk moving the existing Llama scores and need LongBench validation.
- **Stage 8** six Qwen configs (LongBench/GSM8K/RULER, 32K + a YaRN-128K bf16
  variant) — done. **Stage 9** tests — done.

**Parity with the sibling Mistral port (`int2_mistral_clustered`, same base `5e2b9e8`).**
The Mistral port added two model-agnostic robustness fixes to the *shared*
clustered modules; they are now carried here too so the shared modules match the
reference architecture (both are **inert for Qwen**, which passes both args, so
they change nothing functionally on the Qwen path — they are pure parity):
- `cache.py` `_resolve_cache_position` — derive `cache_position` from the cache's
  own monotonic token count when a caller omits it (Qwen2's three attention
  variants all pass it, so the derive-branch never fires).
- `hooks.py` `rotary_emb` fallback — recompute `(cos, sin)` from
  `module.rotary_emb` when `position_embeddings` is absent (Qwen2 passes it).

**One remaining architectural difference from the Mistral sibling:** this port
adds `utils/model_loading.py` + reroutes the runners, which the Mistral port did
not. It is justified by Qwen-genuine load needs Mistral did not have — YaRN
`rope_scaling`/`max_position_embeddings` overrides applied to the AutoConfig, and
neutralizing the shipped `repetition_penalty=1.05` — but it is a heavier load
path than the sibling's per-runner loaders. See the closing note in chat.

**GQA un-repeat (found by the first GPU run — the rep=7 risk, now fixed).**
`Qwen2FlashAttention2` calls `repeat_kv` (4 KV heads → 28) *before* the flash
call; `LlamaFlashAttention2` does not. So the monkeypatched fused-decode path
received a 28-head fp tier while the store, int2 Q tier and gate cards are keyed
by the true `H_kv=4`, and the kernel (`H_kv = k_fp.shape[1] = 28`) rejected the
`H_kv=4` `sel` — `RuntimeError: fused decode gate expects sel [1, 28, *]; got
(1, 4, ...)`, failing every LongBench job at decode. Fix: `_run_fused`
un-repeats `k_fp`/`v_fp` back to `H_kv` (carried in the ctx as `n_kv_heads` from
the store; `repeat_kv` lays out head `h = kv*rep + r`, so `[:, ::rep]` recovers
the originals bit-exactly). `rep == 1` (Llama) is a no-op → byte-identical.
Pinned by `test_gqa_unrepeat_recovers_true_kv_heads` against the real
`repeat_kv`. This is the one place the fused path assumed Llama's no-repeat FA2
layout; the rest of the method is head-count-agnostic.

The sections below are the original design plan, kept as the rationale of record.

## What this is

`int2_qwen` already made this method run on **Qwen2.5-7B-Instruct** end to end —
LongBench, RULER, GSM8K — alongside the Llama-3.1 and Mistral columns, with "the
three columns differ only by model" made a property of the code. This branch
(`int2_clustered_rep`) has since diverged onto a heavier, faster path (fused
Triton decode/score kernels, the gated read, clustered replication) and **never
received the Qwen work**. This plan ports that work over.

It is *not* a cherry-pick. `int2_clustered_rep` and `int2_qwen` share **no common
ancestor** (`git merge-base` returns nothing — the histories were re-based/rewritten
independently), so every change below is re-implemented onto the current files, using
the `int2_qwen` commits as the reference for *intent and exact arithmetic*, not as
patches to `git cherry-pick`.

Reference commits on `int2_qwen`, newest first — read each with `git show <sha>`:

| sha | title |
|---|---|
| `446a7eb` | qwen: fail on OOM, and stop accumulating window scores in the KV dtype |
| `1ce59a5` | configs: the Qwen2.5 matrix at YaRN 128K, flash and eager, with baselines |
| `3d83a6a` | quant: the int2 grid follows the KV dtype, because bf16 can overflow fp16 |
| `5f072c1` | eval: one load path, so "greedy" means greedy on every model column |
| `54f0441` | quant: YaRN folds a scalar into cos/sin, so the RoPE inverse was not one |
| `f0fd6ee` | qwen: fix the vocab-padding crash and a wrong forward-arg index |
| `e7ad39d` | qwen: integrate Qwen2.5-7B into LongBench, RULER and GSM8K |

The whole port range is `git show 46b7523..446a7eb` (its parent to its tip):
**36 files, +4266 / −229**.

## What is already true on this branch

Some of the plumbing the port assumes is present — this branch inherited or
re-grew it, so those parts are verify-not-build:

- **`Qwen2Attention` is already a hook target.** `modules/windowed_cache/hooks.py`
  imports `Qwen2Attention` and `_get_attn_classes()` returns it; the eager twin
  does the same. The `isinstance` match covers `Qwen2FlashAttention2` /
  `Qwen2SdpaAttention` because both subclass `Qwen2Attention`.
- **`effective.py` already imports Qwen2's `apply_rotary_pos_emb`** (falls back
  from Llama to Qwen2), and `_apply_rotary()` is model-agnostic.
- **`ModelConfig` has `name`, `revision`, `dtype`, `attn_implementation`** — but
  none of the RoPE/context override fields, and no validation.
- **`kv_dtype` is already threaded** through the runners and the budget resolver.

What is **missing** and must be built (confirmed absent by symbol search):
`grid_dtype_for`, `rope_attention_scaling`, `tokenizer_has_bos`,
`unmappable_token_ids`, `pin_greedy_generation_defaults`,
`utils/model_loading.py`, `utils/generation.py`, and the Qwen configs.

## The one thing that is harder here than it was on `int2_qwen`

**The fused Triton decode kernel bakes RoPE in.** `int2_qwen` only ever applied
RoPE to keys in two PyTorch functions. This branch also re-applies RoPE *inside*
`modules/windowed_cache/decode_kernel.py` (`rope_cos_sin_halves` →
`gate_kernel`/`decode_kernel`, on the dequantized int2 Q-tier keys, on device).

The YaRN correction (Stage 4) still lands in the **same two PyTorch unrotate
sites**, and this is safe because — verified in the code — **the kernel never
unrotates; it only re-rotates**, and its own docstring says it relies on
`unrotate_key_window` / `rotate_key_window` for the round trip:

- RoPE-strip (unrotate, negated sin) exists **only** at
  `modules/quant/effective.py:170` (`unrotate_key_window`) and
  `modules/windowed_cache/state.py:529` (`rerotate_keys`) — both PyTorch.
- The kernel's `rope_cos_sin_halves` calls `rope_module(ref, pos)`, so it
  receives `a·cosθ` / `a·sinθ` already (transformers folds `a` into cos/sin).

So the invariant to preserve is: **the stored int2 codes must be fit to the bare
pre-RoPE key `k`** (not `a²·k`). If the unrotate sites divide `a²` back out, the
codes hold `k`, and *every* re-rotation — the PyTorch `rotate_key_window`,
`dequant_rotate_q_keys`, and the in-kernel rotate — then applies `a·R(θ)` and
lands on the correct `a·R(θ)·k`. Fixing the two unrotate sites fixes the kernel
path for free. **This must be asserted, not assumed** (Stage 4 test + Stage 9
end-to-end YaRN check).

---

## Stages

Order matters: config schema first (everything reads it), then the load path,
then numerics, then the model-specific fixes, then configs and tests. Each stage
is independently testable and independently committable.

### Stage 1 — `ModelConfig` schema + dtype hardening (`utils/config.py`)

Ref: `5f072c1`. Add to `ModelConfig`:

```python
max_position_embeddings: Optional[int] = None
rope_scaling: Optional[Dict[str, Any]] = None   # {"rope_type":"yarn","factor":4.0,
                                                 #  "original_max_position_embeddings":32768}
sliding_window: Optional[int] = None
use_sliding_window: Optional[bool] = None
```

Add `DTYPE_NAMES = ("float16", "bfloat16", "float32")` at module scope and a
`__post_init__` that:
- raises `ConfigValidationError` if `dtype not in DTYPE_NAMES` (kills the silent
  `dtypes.get(name, torch.float16)` fallback — a typo must not read as an fp16 run);
- validates `attn_implementation ∈ {eager, flash_attention_2, sdpa}`;
- range-checks the four override fields (positive int / bool / null; reject `bool`
  masquerading as int for `max_position_embeddings` and `sliding_window`).

Confirm the loader passes overrides through unchanged (it does dot-notation merge,
so nested `rope_scaling` dicts survive).

### Stage 2 — one load path (`utils/model_loading.py`, **new**)

Ref: `5f072c1`. Port the file verbatim in behaviour. Five owned concerns:

1. **`resolve_dtype(name)`** — name→`torch.dtype`, raises on unknown.
2. **`apply_model_config_overrides(hf_config, model_cfg)`** — applies
   `max_position_embeddings`, `rope_scaling`, `sliding_window`,
   `use_sliding_window` to the `AutoConfig` **before** weights load; returns one
   log line per change. Two non-obvious rules, keep both:
   - mirror legacy `"type"` → `"rope_type"` in the rope dict, or a legacy-spelled
     block validates as `default` (no scaling);
   - re-apply Qwen2's own `sliding_window = sw if use_sliding_window else None`
     disable rule *after* setting the pair, so `use_sliding_window: false` nulls
     the window.
   Then `_revalidate_rope(hf_config)` (try `validate_rope()`, fall back to
   `rope_config_validation`; treat *validator-absent* as different from
   *config-wrong*).
   Note the pinned-transformers-4.47.1 caveat: `_compute_yarn_parameters` reads
   `max_position_embeddings`, **ignores `original_max_position_embeddings`** — so
   setting `max_position_embeddings` alongside the rope block is mandatory, not
   cosmetic.
3. **`pin_greedy_generation_defaults(model)`** — reset the checkpoint's sampling
   knobs (`GREEDY_GENERATION_DEFAULTS`) so greedy is bare greedy. Qwen2.5 ships
   `repetition_penalty: 1.05`, built in `_get_logits_processor` (a **processor**,
   not a warper), so it applies under `do_sample=False`; Llama-3.1 / Mistral ship
   none → returns `{}` → those two columns stay byte-identical. **Never touch
   `eos/pad/bos_token_id`** (the real stopping contract).
4. **`_warn_on_dtype_mismatch`** — warn when run dtype ≠ checkpoint `torch_dtype`
   (the Qwen2.5 fp16-overflow trap).
5. **`load_model_and_tokenizer(cfg, *, raw_prompt_hint, is_windowed)`** — the
   single entry point: tokenizer (+ pad=eos fallback), `AutoConfig`, overrides,
   model, `.eval()`, dtype warn, greedy pin, chat-template warn, active
   `sliding_window` warn (only when `is_windowed`).

Also port `describe_model_load(cfg)` for the meta sidecar.

### Stage 3 — generation safety (`utils/generation.py`, **new**) + prompting

Ref: `f0fd6ee`, `e7ad39d`.

- **`utils/generation.unmappable_token_ids(model, tokenizer)`** — returns
  `list(range(len(tokenizer), vocab_size))` or `None`. Qwen2.5 pads vocab to
  152064 vs a 151665-id tokenizer → 399 ids that decode to `""` and **crash
  `generate(stop_strings=...)`** via `StopStringCriteria` (`IndexError` in
  `F.embedding`). Llama/Mistral: `vocab_size == len(tokenizer)` → `None` → no
  logits processor added → byte-identical.
- **`utils/prompting.py`**: add `tokenizer_has_bos(tokenizer)` and change
  `encode_prompt`'s default to
  `add_special_tokens = tokenizer_has_bos(tokenizer) and not prompt_carries_bos(...)`.
  Qwen2 has no BOS (`bos_token is None`; the config's `bos_token_id 151643` is
  `<|endoftext|>`), so both Qwen rows decide `False`, matching kvpress. Llama /
  Mistral: `has_bos` is `True`, so the rule reduces to the old one exactly.

### Stage 4 — YaRN RoPE inverse (`modules/quant/effective.py`, `state.py` ×2)

Ref: `54f0441`. **The subtle one.** YaRN folds `a = 0.1·ln(factor)+1` (1.13862 at
factor 4) into cos/sin, so the forward map is `a·R(θ)`. The negated-sin inverse
then lands on `a²·k`, and re-rotating gives `a³·R(θ)·k` vs the true `a·R(θ)·k` —
**every int2 Q-tier key and every Q→fp promotion 1.296× too large**, silently.

- Add `rope_attention_scaling(rope_module) -> float`: reads `attention_scaling`
  off the module (not recomputed from config — track what transformers built);
  returns `1.0` for non-finite/≤0.
- In **`unrotate_key_window`** (`effective.py`): after the negated-sin apply,
  `a = rope_attention_scaling(...)`; `if a != 1.0: k_un = k_un / (a*a)`. At
  `a == 1.0` it is a **skipped branch**, so Llama/Mistral/un-scaled-Qwen stay
  bit-identical (pin with `torch.equal`, not `assert_close`).
- Apply the **same `a²` division** in `CacheState.rerotate_keys`
  (`modules/windowed_cache/state.py` and `modules/windowed_eager_cache/state.py`)
  — it does the same strip→re-rotate round trip.
- In `_rope_cos_sin` (`effective.py`), add `_check_rope_scaling_consistency`:
  cross-check `cos²+sin² ≈ a²` once per process (tol 2%), warn loudly if a rope
  variant scales without exposing `attention_scaling`.
- **Do not touch the kernel** (`decode_kernel.py`), per the analysis above — but
  add the end-to-end assertion in Stage 9 that a real YaRN-scaled
  `Qwen2RotaryEmbedding` round-trips exactly through the *fused* path too.

### Stage 5 — int2 grid dtype follows KV dtype (`modules/quant/*`)

Ref: `3d83a6a`. Moving Qwen to native bf16 (Stage 8) makes a key past fp16's
65504 representable in the cache but **`inf` the moment the fp16 grid is cast** →
a whole window dequantizes to `nan`, silently.

- Add **`grid_dtype_for(kv_dtype)`** to `modules/quant/quantizer.py` (export from
  `__init__.py`): `float16` KV → `float16`; **anything else → `bfloat16`** (fp32
  exponent range in the same **2 bytes** — the budget resolver charges the grid at
  2 bytes/element, so the width must not change or every compression ratio moves).
- Thread `grid_dtype` through `_affine_quantize(x, group_dim, grid_dtype=...)`;
  pin the grid in `grid_dtype`, quantize against the **stored** values so
  fit-grid == read-grid bit for bit.
- **`QuantizedStore` owns the dtype** and hands the *same* value to the slot
  buffers (`slots.py`) and to every `quantize` call, so scatter cannot downcast
  the grid. Widening only the quantizer would not work.
- Close the two pre-existing nan paths while here: scale that underflows to zero
  (`0/0` → re-apply the degenerate `where` after the cast, no host sync); a
  non-finite input reaching the `uint8` cast (map to a defined code; let the fp
  tier keep the nan so divergence stays visible).
- Keep the bounded representability guard as a should-never-fire assertion
  (`STICKYKV_GRID_CHECKS`, reports not repairs).
- **Divergence to handle:** this branch also has `modules/quant/sketch.py`
  (absent on `int2_qwen`). Audit it — if it stores or casts a `(scale, zero)`
  grid, it takes `grid_dtype_for(kv_dtype)` too; otherwise leave it. fp16 runs
  must stay bit-identical (pin with `torch.equal` on codes/scale/zero).

### Stage 6 — hooks: forward-arg index + fp32 score accumulation

Ref: `f0fd6ee`, `446a7eb`. Apply to both `modules/windowed_cache/hooks.py` and
`modules/windowed_eager_cache/hooks.py`.

- **Forward-arg fix.** Change `_extract_arg` to `pos: Optional[int] = None`
  (name-only when `pos is None`). The flash hook currently reads
  `_extract_arg(args, kwargs, "position_embeddings", 1)` — index **1 is
  `attention_mask`**, not `position_embeddings` (whose real index is 7 on
  Llama/Qwen2, absent on Mistral). Make it name-only, and add the documented
  fallback: `if position_embeddings is None and hasattr(module, "rotary_emb")`
  recompute, else the loud warning. Inert on 4.47.1 (all args passed by keyword)
  but removes a wrong-tensor-to-RoPE class of silent bug.
- **fp32 window-score accumulator.** `state.window_scores` is a running sum over
  T query rows then over every decode step; in a reduced-precision accumulator
  ~76% of per-step contributions vanish in bf16, ~52% in fp16, and the retained
  top-k diverges ~9% on a 512-step generation. Widen the **reduced** per-chunk
  score result to fp32 before it feeds the accumulator (in this branch that is
  the `[B,H_kv,rep,S]` / `[B,H_q,W]` reduce in `scorer.py` /
  `reduce_two_tier_scores`, **not** the `[..,blk,S]` softmax block — leave
  `_score_softmax_dtype()` alone, torch already reduces softmax with an fp32
  internal accumulator). **Check first whether this branch already accumulates in
  fp32** (its numerics work may have pre-empted this); if so, this is verify-only.
  Note: this makes Llama/Mistral **not** byte-identical to their old numbers —
  they must be re-run to share a table.

### Stage 7 — route the generation runners through the load path

Ref: `5f072c1`, `446a7eb`. For `longbench_runner.py`, `ruler_runner.py`,
`gsm8k_runner.py`:

- Replace each inline `_load_model_and_tokenizer` with
  `utils.model_loading.load_model_and_tokenizer(cfg, raw_prompt_hint=..., is_windowed=self.is_windowed)`.
- Pass `suppress_tokens=unmappable_token_ids(model, tokenizer)` into `generate()`
  (only added when non-`None`).
- **GSM8K**: it passes `stop_strings` → the vocab-gap crash is reachable there
  today; suppress-tokens is the fix.
- **LongBench**: (a) log the `max_length` policy at run start and warn when a
  configured value discards headroom; (b) **samsum EOS**: *extend* the model's
  EOS set with the newline id, do not *replace* it (replacing drops Qwen's
  `<|endoftext|>` 151643, which a raw continuation actually emits); (c) OOM
  ceiling must carry a **`num_hidden_layers` factor** — eager returns all layers'
  `[B,H_q,T,T]` at once (`output_attentions=True`), so the old one-layer estimate
  understated by 28× on Qwen and stayed quiet through LongBench's whole range;
  warn with the computed figure and tell the user to set `skip_oom: false`.
- Not converting the **parity** and **perf** runners (they keep their own loaders,
  as on `int2_qwen`) — unless a Qwen parity/perf column is wanted, in which case
  fold them in as a follow-up.

### Stage 8 — the Qwen config matrix (`configs/*qwen*.yaml`)

Ref: `e7ad39d` (6 configs at 32K), `1ce59a5` (YaRN 128K matrix), `446a7eb`
(`skip_oom`/`resume`). Mirror the existing Llama/Mistral configs so **columns
differ only by model**:

- `longbench_qwen_{full_cache,full_cache_eager,ours_flash_attn,ours_eager}.yaml`
- `ruler_qwen_{full_cache,full_cache_eager,niah_mk3_omega16,niah_mk3_omega16_eager}.yaml`
- `gsm8k_qwen_{full_cache,full_cache_eager,budget20,budget20_eager}.yaml`

Rules baked into these:
- `model.name: Qwen/Qwen2.5-7B-Instruct`, geometry matched to the Llama set.
- **dtype note**: Qwen2.5 is bf16-native; `float16` keeps the three columns
  comparable, but must be applied to *all three* models together or not at all.
  For the YaRN-128K matrix, use `bfloat16` (fp16 overflows at long context) — and
  then Stage 5's `grid_dtype_for` is what keeps the quantizer safe.
- **YaRN block** for the 128K configs:
  `rope_scaling: {rope_type: yarn, factor: 4.0, original_max_position_embeddings: 32768}`
  with `max_position_embeddings: 131072` (mandatory alongside, per Stage 2). The
  32K configs leave rope alone (`max_length` truncates as with Mistral-v0.2).
- **On both Qwen "ours" configs**: `skip_oom: false`, `resume: false` — an OOM
  must fail the run, not return a uniformly-depressed table with `skipped=N` as
  the only tell; `resume: true` would accept a crash-truncated jsonl as finished.

### Stage 9 — tests

Ref: all. Port/adapt:

- **`tests/test_qwen_protocol.py`** — greedy-pinning, bias-in-q/k/v (perturb the
  q_proj bias, require the surviving window set to move), BOS rule truth table,
  sliding-window-quiet-on-Qwen / real-on-Mistral-v0.1.
- **`tests/test_generation_safety.py`** — the 399-id vocab gap, the
  `StopStringCriteria` mechanism, arg-extraction indices against the *real* 4.47.1
  signatures (`hidden_states@0`, `position_ids@2`, `position_embeddings@7`).
- **`tests/test_score_precision.py`** — fp32-accumulator arithmetic + the eager
  OOM-guard `num_layers` factor (the three guard assertions must fail against the
  pre-fix guard).
- **`tests/test_qwen_yarn_e2e.py`** — the RoPE round trip through a **real**
  YaRN-scaled `Qwen2RotaryEmbedding` is exact; disabling the `a²` correction
  reproduces the 1.296× inflation. **Extend beyond `int2_qwen`**: assert the same
  through the **fused decode kernel** path (unrotate PyTorch → dequant+rotate
  in-kernel), so the Stage-4 "the kernel is correct for free" claim is pinned, not
  assumed.
- Extend **`tests/test_backend_e2e.py`** to run each case on `llama` / `mistral` /
  `qwen2` (config-built tiny model + the real Qwen2.5 tokenizer, CPU).
- Update **`tests/test_prompting.py`** for the new BOS rule.

Gate the model-driven tests on the transformers version (`4.47.1`): past it the
Cache ABI changed and even a bare `model.forward` with this cache raises from the
mask builder — skip, don't fail.

---

## Validation checklist (the parity invariants)

Run these before calling any stage done:

1. **fp16 quant is bit-identical.** Full existing quant suite green untouched;
   `torch.equal` on codes/scale/zero against the pre-change algorithm (Stage 5).
2. **`a == 1` is a skipped branch.** Llama/Mistral/un-scaled-Qwen RoPE round trip
   `torch.equal` to the pre-change expression (Stage 4).
3. **Greedy pin is `{}` on Llama/Mistral.** Those two runs untouched by Stage 2.
4. **`unmappable_token_ids` is `None` on Llama/Mistral.** No logits processor
   added; generation byte-identical (Stage 3).
5. **BOS rule reduces to the old one on Llama/Mistral** (Stage 3).
6. **YaRN end-to-end**: real Qwen2.5 tokenizer + tiny config-built Qwen2, all
   three suites, `skipped == 0` / `dropped == 0`, and the fused-path YaRN round
   trip exact (Stage 9).
7. **CPU suite** green in the pinned 4.47.1 env; on an unpinned box, the failure
   list must be **identical before and after** (all pre-existing).

Two intended, documented non-identities:
- Stage 6 (fp32 accumulator) changes Llama/Mistral scores — re-run them to share
  a table.
- Stage 5 costs bf16 runs 3 grid mantissa bits — dominated by int2's 4 levels,
  and the fit==read exactness invariant is preserved.

## Suggested commit sequence

Each maps to a reviewable unit; keep the `int2_qwen` titles so the lineage is
searchable.

1. `config: ModelConfig context/RoPE overrides + dtype hardening` (Stage 1)
2. `eval: one load path, so "greedy" means greedy on every model column` (Stages 2, 7)
3. `qwen: vocab-padding suppress_tokens + BOS-aware prompting + forward-arg index` (Stages 3, 6-arg)
4. `quant: the int2 grid follows the KV dtype` (Stage 5)
5. `quant: YaRN folds a scalar into cos/sin, so the RoPE inverse was not one` (Stage 4)
6. `eval: fp32 window-score accumulation + eager OOM ceiling` (Stage 6-accum, 7-OOM)
7. `configs: the Qwen2.5 matrix (32K + YaRN 128K), flash and eager` (Stage 8)
8. `test: qwen protocol, generation safety, score precision, YaRN e2e (+ fused path)` (Stage 9)

## Open questions to settle before starting

- **bf16 or fp16 for the headline Qwen table?** fp16 keeps columns comparable at
  32K; bf16 is required for the YaRN-128K matrix. This decides whether Stage 5's
  bf16 grid is exercised on the main table or only the long-context one.
  **Settled in Round 3:** bf16 + YaRN 128K on every Qwen config, because that is
  the protocol the QEvict column was measured on; the fp16/32K choice made the
  two columns different model setups under the same filenames.
- **Do we want a Qwen parity/perf column?** If yes, fold the parity/perf runners
  into `utils/model_loading` too (out of scope above; `int2_qwen` left them).
- **Does this branch already accumulate window scores in fp32?** If its numerics
  work pre-empted Stage 6's accumulator change, that stage is verify-only and
  Llama/Mistral stay identical.
