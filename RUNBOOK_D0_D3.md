# D0–D3: the exact commands, in order, with the number each one must move

Four changes landed on `int2_clustered_rep`. Every one is CPU-verified; **none is
GPU-verified**, because the dev box has no GPU. This file is the sequence that
turns them from code into numbers.

The rule, and it is the whole point of the ordering: **run one step, read its
acceptance number, then move on.** Each step below names exactly what to look at
and what verdict each outcome carries. A step whose number does not move is a
step that did not work, and the next step is then a different one.

Set this once:

```bash
export MODEL_PATH=<hf-id-or-path>
export CUDA_VISIBLE_DEVICES=0
```

The config every profile below points at is the one `run_perf_table.sh`
generates, so the profile and the table are the same cache
(`tests/test_default_operating_point.py` pins that):

```
outputs/perf_table/_perf_table.generated.yaml
```

If it does not exist yet, step 0 creates it.

---

## Step 0 — the baseline table, at HEAD, before reading anything

```bash
scripts/run_perf_table.sh
```

Writes `outputs/perf_table/`. This is the row set every later step is compared
against. Copy the printed table somewhere; the npz files are per-shape and a
later partial run overwrites only the shapes it ran.

**Acceptance: none.** This is the reference.

---

## Step 1 — D0: is the eviction fusing now?

```bash
python scripts/profile_decode.py \
  --config outputs/perf_table/_perf_table.generated.yaml \
  --prefill 4096 --batch 32 --steps 24 --warmup 12 --top 40
```

**Read four things, in this order.**

1. **The eviction verdict** — new, printed under the bucket table:

   ```
   eviction: 3 compiled / 0 eager runs, Inductor kernels X.XXX ms/step over N launches/step
   ```

   - `Inductor kernels 0.000` plus `!! COMPILED BUT NOT FUSED` → **D0 did not
     land on GPU.** The CPU fix removed 8 `data_ptr` graph breaks; if the GPU
     still emits nothing, there is a second cause. Run
     `python -m pytest tests/test_evict_graph_breaks.py -q` on the box and then
     bound it with `--compile-evict 0`: if eager and "compiled" tie, the compile
     is inert either way.
   - `Inductor kernels` non-zero → **D0 landed.** That time was previously
     spread through `elementwise` / `memory: index/gather` as
     `at::native::elementwise_kernel` rows with fractional launch counts.

2. **`ours (cache)`** — was `7.917 ms` (= 7.104 two-tier + 0.813 gate, exactly).
   It must now be **larger**, because the compiled-evict bucket is no longer
   zero. That increase is not a regression; it is work moving from
   *unattributed* to *ours*.

3. **`CUDA kernels` and `GPU busy`** — this build has the `33ae5f9` double-count
   fix and the `130de24` denominator fix, so expect roughly **50 ms/step over
   ~1,700 launches, busy ~60%**, not the 87.41 / 3132 / 59.3% of the
   2026-09-16 profile. If it still prints 87 ms and 3132 launches, the box is
   running an old checkout — `git log --oneline -1` should show `fd395e0` or
   later.

4. **The tile rung** — new, printed with the rollup:

   ```
   decode tiling (target_keys x num_stages), per geometry:
     ws=8 head_dim=128 BLOCK_R=4 q_tier=True gated=True  ->  64x2
   ```

   This is D3's whole verdict and it no longer requires catching a warmup line.

**Then the number that decides D1**, printed after both timed windows:

```
eviction widths over N evictions (n_q=64, W_retained=...):
  fresh (quantize+sketch)  max=  ..  mean= ....  of n_q=64   fill=..%
```

- `fill` well under 50% → D1 is live and worth the two-phase refactor.
- `fill` near 100% → **D1 is worth nothing**; drop it from the plan.

---

## Step 2 — D0 + D2 + D3, priced together on the table

D0, D2 and D3 all land in the same build and all three are score-neutral, so one
table run prices them as a block against step 0.

```bash
scripts/run_perf_table.sh
```

**Acceptance: `TPOT_steady` at 4096/B=32 must fall from 0.0763.**

| outcome | reading |
|---|---|
| ≤ 0.070 | D0 was most of the batch-32 gap, as §5.2 of `TARGET_GAP.md` predicts |
| 0.070–0.075 | D0 landed but is smaller than `1434b2c`'s 101.6 → 85 ms suggests; the trace (step 5) is next |
| ≥ 0.076 | nothing moved — go back to step 1's verdict line before writing any more code |

**Watch the B=1 rows too.** D0 and D2 are both eviction-side and scale with
`L·B`, so they should move B=32 much more than B=1. If B=1 (0.0566 / 0.0574 /
0.0573) barely moves, that is the *expected* result and it is also the finding:
it confirms `TARGET_GAP.md` §6 — the batch-1 cells need Block A, not these.

---

## Step 3 — D3 alone, if step 1 said the rung is not `64x2`

Only run this if step 1 printed a rung below `64x2`. Every rung is
bit-identical, so this is pure speed and a clean A/B:

```bash
SHAPES="4096/256" BATCHES="1 32" OUT_DIR=outputs/tile_64x2 \
  STICKYKV_DECODE_TILE=64x2 scripts/run_perf_table.sh

SHAPES="4096/256" BATCHES="1 32" OUT_DIR=outputs/tile_64x1 \
  STICKYKV_DECODE_TILE=64x1 scripts/run_perf_table.sh

SHAPES="4096/256" BATCHES="1 32" OUT_DIR=outputs/tile_32x2 \
  STICKYKV_DECODE_TILE=32x2 scripts/run_perf_table.sh
```

A pinned rung that does not fit **raises** rather than stepping down, so a run
that completes is a run at the rung on its label. The `64x2` arm will fail to
launch if that rung is why the ladder fell — which is itself the answer.

**Acceptance:** whichever rung wins should become the ladder's first entry. If
`64x1` beats `32x2`, the new rung paid for itself; if `32x2` wins, the ladder's
"widest tile first" ordering is wrong and the comment saying so needs correcting.

---

## Step 4 — Block A: the run that decides whether the target is reachable at all

Independent of D0–D3, and **more important than all of them** (`TARGET_GAP.md`
§3: 43% of the gap, dead flat across a 32× batch range, never measured):

```bash
bash scripts/bisect_decode_regression.sh --model "$MODEL_PATH"
```

Three table arms plus a profile, in one command. The decision is pre-committed in
the script's own header:

- **`exp2` owns it** → drop it. 4.3–5.3 ms recovered free, at every batch size.
- **RoPE dtype owns it** → keep it; it closed a 9.5-point `qasper` regression.
  Then the published numbers are **unreachable** and the floor is ~8% above
  target for ever, whatever D0–D3 do.
- **neither** → a third cause, and the profile it prints is the next instrument.

---

## Step 5 — the chrome trace, once D0 is settled

```bash
python scripts/profile_decode.py \
  --config outputs/perf_table/_perf_table.generated.yaml \
  --prefill 4096 --batch 32 --steps 24 --warmup 12 \
  --trace /tmp/decode.json
```

The only thing that can split the `shared/other` block by launching stack. Worth
more *after* D0 than before it: if the eviction now fuses, whatever remains in
`elementwise` is the model's, and that changes what is left to do.

---

## Step 6 — D1, only if step 1's `fill` justified it

Not implemented, deliberately. `_compact`'s width needs the real count, the real
count needs `.item()`, and `.item()` inside the eviction body reintroduces the
graph break D0 removed — measured, not assumed:

```
Unsupported Tensor.item() call with capture_scalar_outputs=False
```

So D1 means splitting the eviction into a *decide* phase (ranking, lookup, masks
— small tensors) and an *apply* phase (promote, demote, rebuild — the payload),
with the one sync between them and the width passed to the second as a Python
int rounded up to a small ladder of rungs so `torch.compile` sees a handful of
shapes rather than a new one per eviction.

That is a real refactor and it should not be started until step 1 prints a `fill`
that pays for it.

---

## What to re-run after any change

```bash
python -m pytest tests/ -q
```

Baseline is **1073 passed, 12 failed** on a CPU box. Those 12 are all
`tests/test_backend_e2e.py` and all say `has_triton=True, is_cuda=False` — the
prefill score kernel is CUDA-only. On the GPU box they should pass; if they do
not, that is a real failure and not the documented one.

The four guards that pin D0–D3 specifically:

```bash
python -m pytest tests/test_evict_graph_breaks.py \
                 tests/test_sketch_eps_elision.py \
                 tests/test_profile_buckets.py \
                 tests/test_decode_kernel.py -q
```
