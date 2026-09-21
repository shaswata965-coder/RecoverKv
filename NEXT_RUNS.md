# The next three runs — commands, and what each one decides

Companion to `DISTANCE_TO_GOAL.md` §11. That says what is open; this says what
to type and what the output has to look like for the number beside it to mean
anything.

**Every flag here was checked against the parser before being written down.**
A runbook that advertises a knob the script does not have is how `6b8a188`
happened (`quant_gate_ratio` inert in every runner) and how the previous
version of `run_perf_table.sh --help` came to truncate its own banner.

Set this once:

```bash
export MODEL=/home/ee/phd/eez228470/llama-3.1-8b-instruct
```

---

## 0. Preflight — 20 seconds, and it has already saved a session

```bash
git branch --show-current                                   # int2_clustered_rep
git fetch origin int2_clustered_rep
git rev-list --left-right --count int2_clustered_rep...origin/int2_clustered_rep
grep -rn "fused_gate" --include="*.py" . | grep -v "^./tests/"
python scripts/check_operating_point.py
```

**Expect:** the branch name; `0 0` from the third command; the grep returning
`flash_decode.py:51` and `:319` (plus `gate_kernel.py`'s own definition); and
`All non-exempt configs are at the operating point.`

**Stop if:** the right-hand count is nonzero (your ref is behind — anything you
conclude describes a different codebase), or the grep returns only
`gate_kernel.py` (you are on the abandoned Sep-14 line where the gate has no
callers and every number is ungated).

---

## 1. Run A — does the measured tile ladder actually pay?

Item 1 (`c647ab3`) reordered `_FIT_LADDER` by measured time. Order is
irrelevant to a *tuned* geometry — every rung is timed — and decides everything
for an *untuned* one, because `_first_fit` returns the first rung that fits and
a fit is shape-independent. The old head was `(64,2,4)`, which three profiles
rank 5th–9th of 11.

**This is a guess about untuned geometries and it ships with a control arm.**
Both arms, two directories, one flag apart:

```bash
CUDA_VISIBLE_DEVICES=0 OUT_DIR=outputs/v11_measured bash scripts/run_perf_table.sh \
  --model "$MODEL" \
  --shapes "4096/256 1048/1048 2048/512" --batches "1 32" \
  --data-source wikitext-103 --throughput all \
  --cooldown 3 --clock-lock true \
  --evict-control-arm false --fit-ladder measured

CUDA_VISIBLE_DEVICES=0 OUT_DIR=outputs/v11_legacy bash scripts/run_perf_table.sh \
  --model "$MODEL" \
  --shapes "4096/256 1048/1048 2048/512" --batches "1 32" \
  --data-source wikitext-103 --throughput all \
  --cooldown 3 --clock-lock true \
  --evict-control-arm false --fit-ladder legacy
```

**Two `OUT_DIR`s is not optional.** The npz is named
`perf_prefill{P}_gen{G}_bs{B}.npz` — shape and batch, nothing about the config
— so one directory would silently keep the survivor.

### What to expect

* `=== eviction: compiled eviction (shipped) ===` on **both** (this run varies
  the ladder, not the eviction).
* On the legacy arm only, once, in the Python log (it prints when
  `decode_kernel` is imported, not from the shell wrapper):
  `[StickyKV] decode tile ladder: LEGACY tile-major order …`. Its **absence**
  on the measured arm is how you know the arms differ.
* Per tuned geometry: `[StickyKV] fused decode tiling tuned sig=(…) -> (32, 2, 4)`
  with the whole ladder. **Both arms should land on the same winner here** —
  the search times every rung that fits, so only the fallback moved. That
  holds at this geometry because no two rungs collapse onto one launch: the
  dedup `key` includes `num_stages` and `num_warps`, and at `ws=8, W_phys=273`
  all 12 rungs are distinct compiles. At a large `ws` with `W_phys <= 16` some
  rungs do collapse, and there the order decides which representative of a
  collapsed group gets timed — so the "same winner" expectation is specific to
  this shape, not general.
* `read gate: ran on all 24576 fused layers, realised read fraction 0.251`.
* `eviction path: … | mode: compiled | tile ladder: measured|legacy | dynamo: …`
* Six rows, `TPOT_steady` near **0.055–0.059**.

### How to read it

| result | meaning |
|---|---|
| measured **faster** than legacy | the fallback was the leak; §10.14's missing 4.7 points are recovered in proportion |
| a **tie** | untuned geometries are a small share of the steps here; the reorder is harmless but not the leak |
| measured **slower** | the §10.15 failure recurring — the trade does move with geometry. **Revert to `legacy` as the default**; do not argue with it |

That third row is a real possibility and the arm exists to catch it. A short
Q-tier loop at fill-phase geometries may genuinely prefer a larger tile.

**Falsifier:** if the two arms announce *different* tuned winners at the same
`sig`, the search has an order dependence and the reorder is not the clean
change it is described as. Report that rather than the TPOT delta.

---

## 2. Run B — is the decode kernel scatter-bound or occupancy-starved?

This decides what item 3 actually is, and it needs **no code change**. The
ladder already publishes every rung per geometry and `B` is already in the
signature (§11.10).

```bash
CUDA_VISIBLE_DEVICES=0 OUT_DIR=outputs/v11_batchsweep bash scripts/run_perf_table.sh \
  --model "$MODEL" \
  --shapes "4096/256" --batches "1 8 32" \
  --data-source wikitext-103 --throughput all \
  --cooldown 3 --clock-lock true \
  --evict-control-arm false --fit-ladder measured
```

### What to expect

Three `fused decode tiling tuned sig=…` announcements, one per batch, each with
its full ladder. `grid = B * H_kv` is 8, 64 and 256 blocks against an A100's
108 SMs — 0.07, 0.59 and 2.37 waves.

### How to read it — this is the whole point of the run

Compare the **spread** (slowest rung ÷ fastest rung) across the three batches:

| observation | mechanism | what item 3 becomes |
|---|---|---|
| spread roughly **constant** across B | latency inside a block; the tile choice matters equally when the machine is empty | **§6.2, contiguous reads** — interleave the 10 per-window tensors |
| spread **collapses at B=1/8** (rungs converge) | the grid; with 8 blocks on 108 SMs nothing the tile does matters | **split-KV** — partition keys, `grid = B*H_kv*S`, log-sum-exp merge |

Current evidence favours the second: `(32,2,4)` beating `(64,2,4)` is an
occupancy signature, and TPOT moves only +6.9% across a 32× batch.

**Also worth recording:** `TPOT_steady` at B=1 vs B=8 vs B=32. If B=8 is
roughly B=32, the kernel is grid-limited and §8's "B=1 is not winnable" is an
empty GPU rather than a bandwidth floor — which changes what can be claimed.

---

## 3. Run C — a compiled-arm profile with a valid wall

The 2026-09-20 compiled-arm profile has **no usable `GPU busy`**: Dynamo built
18 graphs inside the unprofiled window, so the denominator was compiling while
the numerator was not. The kernel table was fine; the wall was not.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_decode.py \
  --config outputs/v11_measured/_perf_table.generated.yaml \
  --prefill 4096 --batch 32 --steps 24 --warmup 40 --top 40 \
  --compile-evict 1
```

`--warmup 40` (was 12) is the change: the eviction fires once per
`window_size`=8 steps, so 40 covers ~5 evictions and should let compilation
finish before window 1.

### What to expect

* `wall, profiled` **greater** than `wall` — a positive overhead.
* **No** `-- COMPILATION INSIDE THE MEASURED WINDOW` block.
* **No** `!! THE TWO WINDOWS ARE NOT COMPARABLE` and no `!! IMPOSSIBLE`.
* `GPU busy` in the **93–98%** range, with `-> KERNEL-BOUND`.
* `eviction: N compiled / 0 eager runs, Inductor kernels ~3.0 ms/step over ~14
  launches/step`.
* Buckets near: decode **14.4**, gate **2.4**, GEMM **11.0**, compiled evict
  **3.0**; **CUDA kernels ≈ 40.5 ms/step**.

### If the guard still fires

The message names which tell tripped. If it is *"Dynamo compiled during the
unprofiled window"* even at `--warmup 40`, **stop raising warmup** — that means
a new graph is built per eviction, which no warmup length absorbs, and the bug
is the recompile axis: `T_fp` and `W` (`cache.py:2695`, `:2698`) are
`int()`-forced, so they specialise even under `dynamic=True`, and neither is on
a ladder the way `_EVICT_WIDTH_LADDER` already puts the widths. That is then
the next fix, not a harness setting.

### The control-arm profile, for the matched pair

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_decode.py \
  --config outputs/v11_measured/_perf_table.generated.yaml \
  --prefill 4096 --batch 32 --steps 24 --warmup 40 --top 40 \
  --compile-evict 0
```

Expect `eviction: CONTROL ARM -- 3 eager run(s), no compile, no autotune.`,
**CUDA kernels ≈ 39.5 ms/step** and `GPU busy ≈ 91%`. The eager arm's kernels
are *cheaper* and its step is *slower* — that is §11.9's finding, and this pair
re-measures it with both walls valid.

---

## 3b. Run D — split-KV, the one that is supposed to pay

Item 3 (§11.11). `grid = (B*H_kv,)` is 8 blocks at B=1 on a 108-SM A100;
split-KV partitions the window axis so it becomes `B*H_kv*S`.

**Read this before running it.** The kernel could not be executed on the box it
was written on — `triton` imports there, `torch.cuda.is_available()` is False.
What *is* validated is the algorithm (the CPU reference carries `splits` and 23
tests pin it, including that `splits=1` is bit-identical and that the window
ranking does not move) and the wiring (22 tests, each negative-tested by
breaking it). The kernel itself rests on that plus review. **`--decode-splits 1`
is the default and a provable no-op**, so the shipped path is unchanged until
you ask for otherwise.

```bash
CUDA_VISIBLE_DEVICES=0 OUT_DIR=outputs/v12_split1 bash scripts/run_perf_table.sh \
  --model "$MODEL" \
  --shapes "4096/256 1048/1048 2048/512" --batches "1 32" \
  --data-source wikitext-103 --throughput all \
  --cooldown 3 --clock-lock true --decode-splits 1

CUDA_VISIBLE_DEVICES=0 OUT_DIR=outputs/v12_splitauto bash scripts/run_perf_table.sh \
  --model "$MODEL" \
  --shapes "4096/256 1048/1048 2048/512" --batches "1 32" \
  --data-source wikitext-103 --throughput all \
  --cooldown 3 --clock-lock true --decode-splits auto
```

### The first thing to check is CORRECTNESS, not speed

A split that drops or double-counts a window produces a plausible number, not a
crash. Before reading any timing:

* `read gate: ran on all 24576 fused layers, realised read fraction 0.251` must
  appear on **both**, unchanged. The gate runs before the split and must be
  untouched by it.
* `decode splits: 1` / `auto` in the `eviction path:` log line, and
  `decode_splits` in each `run_perf_table.env`. If both say 1, the arms never
  differed.
* **If the two arms' TPOT differ by more than ~15% in either direction at
  B=32, suspect correctness before celebrating or despairing.** The honest
  check is a quality run; the cheap one is that `peak_GB` and `steadyKV_GB`
  should be identical between arms — split-KV changes the grid, not the cache.

### Then the speed

| B | blocks at S=1 | at S=auto | expectation |
|---|---|---|---|
| 1 | 8 | up to 128 | **the big one** — 7.4% of the machine today |
| 32 | 256 | 256 (auto stops) | little or nothing; it is already 2.37 waves |

So the shape to look for is **B=1 improving and B=32 roughly flat**. That
asymmetry is the prediction; if B=32 moves a lot and B=1 does not, the model of
§11.11 is wrong and the number needs explaining before it is quoted.

A profile of the split arm, for where the time actually went:

```bash
CUDA_VISIBLE_DEVICES=0 STICKYKV_DECODE_SPLITS=auto python scripts/profile_decode.py \
  --config outputs/v12_splitauto/_perf_table.generated.yaml \
  --prefill 4096 --batch 1 --steps 24 --warmup 40 --top 40 --compile-evict 1
```

B=1, not 32 — that is where the grid is empty and where the change is supposed
to show.

## 3c. Run D is CLOSED, and Run E is the only one that matters now

### What Run D settled

The S sweep ran all sixteen values at 4096/257 bs1, one `OUT_DIR` each
(`v12_splitsweep/s01..s16`), gate verdict identical across every one:

| | TPOT | vs S=1 | spread |
|---|---|---|---|
| **S=1** | **0.0559** | — | the control, and the winner |
| S=2–9 | 0.0599 | **+7.1%** | 1.8% |
| S=10–16 | 0.0653 | **+16.9%** | 2.1% |

TTFT across the same sixteen runs — which this knob cannot touch — moved 1.8%
(1 sd), so that is the noise floor, the plateaus are real, and the variation
inside each one is not. **Split-KV is closed. `S=1` stays the default.**

The shape is the finding: the penalty is flat across S=2–9, so it is the cost
of *having* splits, not of how many. And the B=1 profile says why it could
never have paid — `_two_tier_decode_kernel` is **2.858 ms of a 45.196 ms step,
6.3%**, and that reading is of the *split* arm, so the control is lower still.

**This also retires §11.11's decomposition.** It inferred "~11 ms of that
kernel is paid regardless of batch" from a whole-step fit and built split-KV on
it. The batch-invariance it measured is real; the attribution was not.

### Run E — attribute the 42%

One kernel signature, `vectorized_elementwise_kernel<4, ...>`, is **19.045
ms/step, 42.1%, at 131 launches of 0.146 ms each**, and the name-based rollup
cannot say whose it is. Everything else is already sized: the model's GEMMs are
on the B=1 weight floor, the decode kernel is 6.3%, the eviction is 7.478 ms of
which one gather is 6.564. So **1.2× can only come out of the elementwise pile,
that gather, or the host gap**, and only Run E says which.

Take BOTH runs. The first is the timing, the second is the attribution, and
they are not interchangeable — `--stack` adds per-op CPU cost, so its host gap
is not a measurement.

```bash
cd ~/kv_cache/Clustered_efficiency/RecoverKv
export MODEL=/home/ee/phd/eez228470/llama-3.1-8b-instruct
CFG=v12_splitsweep/s01/_perf_table.generated.yaml    # the S=1 control arm

# E1 -- timing + trace, no --stack. This is the valid wall.
CUDA_VISIBLE_DEVICES=0 STICKYKV_DECODE_SPLITS=1 python scripts/profile_decode.py \
  --config "$CFG" --prefill 4096 --batch 1 --steps 24 --warmup 40 \
  --top 40 --compile-evict 1 --trace outputs/e1_notstack.json \
  2>&1 | tee outputs/e1_timing.txt

# E2 -- attribution. --stack makes every cpu_op carry its Python frames.
CUDA_VISIBLE_DEVICES=0 STICKYKV_DECODE_SPLITS=1 python scripts/profile_decode.py \
  --config "$CFG" --prefill 4096 --batch 1 --steps 24 --warmup 40 \
  --top 40 --compile-evict 1 --stack --trace outputs/e2_stack.json \
  2>&1 | tee outputs/e2_profile.txt

# The analysis. --target-ms is the S=1 TPOT in ms, which turns on the budget.
python scripts/analyze_decode_trace.py outputs/e2_stack.json \
  --steps 24 --top 40 --target-ms 55.9 --speedup 1.2 \
  --csv outputs/e2_matrix.csv | tee outputs/e2_attribution.txt
```

`analyze_decode_trace.py --selftest` runs anywhere, including a CPU-only box,
and checks the attribution rules before you spend GPU time on them.

### How to read it — the decision it makes

Section 3, **KERNEL KIND × OWNER**, is the whole run. Find the `elementwise`
row and read across:

* **Mostly `ours:`** → the headroom is real. The 19 ms is this project's eager
  cache work, it sits next to an Inductor region that already compiles, and
  extending that region is the final push. 1.2× is reachable.
* **Mostly `model: transformers`** → we are near the ceiling at B=1. Stop
  buying TPOT and take the remaining small wins (host gap, the eviction gather,
  the two §5.1 fusions), then move the claim to B=32 throughput and memory.

Section 6 prices each candidate against the −9.3 ms that 1.2× needs. **Anything
under 100% cannot get there alone.**

### Two traps specific to this run

* **`--stack` distorts the host gap.** Take the gap from E1. Section 6 of the
  analysis will happily compute one from E2's wall; it is not a measurement.
* **The trace is large** — tens to hundreds of MB for 24 steps. Write it under
  `outputs/`, not `/tmp`, and do not commit it; the CSV and the text report are
  the artifacts worth keeping.

## 4. Reprinting without re-measuring

Any npz directory can be re-tabled on a CPU:

```bash
python scripts/print_perf_table.py --npz-dir outputs/v11_measured --throughput all
```

---

## What none of these runs establish

* **Quality.** Every row here is TPOT. No LongBench, GSM8K or RULER run is
  implied, and the tile ladder moves `wsum`/`est` in their last bits
  (§10.5), so a quality claim needs its own run.
* **The two missing baselines.** int2 KIVI and QEvict still have no
  implementation (§7). Nothing here licenses "beats KIVI".
* **`--cooldown 3 --clock-lock true` breaks comparability with `table_v9`**,
  which was taken without them. The arms above are comparable to each other,
  which is what the questions need.

---

## Verification status of this file

Checked on 2026-09-20 at `e82a1ad`, on the box that has no GPU:

* every flag in every command exists in its parser (checked against
  `run_perf_table.sh`'s case block and the two scripts' `add_argument` calls);
* all three table commands were executed and reach the model load — past
  argument parsing, config generation and the arm banner — printing the
  expected `=== eviction: … ===` line, writing 6/6/3 grid cells respectively,
  and recording `fit_ladder` and `evict_control_arm` in `run_perf_table.env`;
* the ladder resolver was exercised over 18 cases: default `measured`, five
  spellings of `legacy`, and four typos, which raise rather than fall back.

**Not verified here, and not claimable until the runs happen:** every expected
*number*. The ranges quoted are what the 2026-09-20 profiles measured, carried
forward as predictions. If a run disagrees with them, the run is right.
