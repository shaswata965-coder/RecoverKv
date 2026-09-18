# RecoverKv — working notes for Claude

`DISTANCE_TO_GOAL.md` is the live status document. Read it before proposing any
decode work; it holds the current numbers and what is measured vs. argued. This
file holds the things that are easy to get wrong *while* reading it.

## The most important thing in this repository

**The rank-1 card read gate (`modules/quant/sketch.py`) is what this project is
for, and it must be the ONLY decode read path wherever a Q tier exists.**

Everything else — the int2 two-tier store, the batched eviction, the fused decode
kernel — is infrastructure that exists so the gate can run. Every int2 window
carries a card written once at demotion; the decode step scores all cards from
two dot products and dequantizes only the windows worth opening.

### It is the only read path, and that is enforced

`_run_fused` **raises** when a fused layer arrives without cards. There is no
ungated arm: both arming sites already require a non-empty Q tier, so a
card-less fused layer is a store holding a tier it cannot select over — a
misconfiguration, not a mode. Reading the whole tier there would be *correct*
and would look exactly like success, which is the one failure this path must
refuse. `quant_gate_ratio = 1.0` is how you ask for a full read; it selects
every window through the same code.

### It is wired. Check, don't assume.

The live path is `cache.update()` → `flash_decode.set_pending` →
`_run_fused` → `fused_gate` → `fused_two_tier_decode(..., sel=...)`. The gate
runs in `_run_fused` because selecting windows needs the **query**, and
`Cache.update()` is handed keys and values only — that is the shape of the
transformers API, not a choice this repo can revisit.

It has been unreachable once already. Between `3658f72` and `9d131e4` the gate
was built, tested and documented with **zero non-test callers**: the step read
100% of the tier while `build_sketch` ran every eviction and the cards sat in
memory unread. Presence in a module, a test and a design doc is not presence in
the step. So:

```bash
grep -rn "fused_gate" --include="*.py" . | grep -v "^./tests/"
```

`flash_decode.stats()` is the runtime version of the same check: `gated ==
fired` says the gate ran on every fused layer, and `read_fraction` is
`quant_gate_ratio` **as realised**. `gated == 0` with a large `fired` is the
silent no-cards case, which looks exactly like success everywhere else.

**Every run now carries that proof — quality as well as perf.**
`flash_decode.expect_gated(backend, quant_ratio, cuda)` is the single definition
of "should this have gated" (flash backend, `quant_ratio > 0`, CUDA), and
`gate_report` turns the counters into one of four stable verdicts — `gated`,
`not-expected`, `NOT-GATED-kernel-never-fired`, `NOT-GATED-partial`. LongBench,
GSM8K and RULER record it as `read_gate` in their metadata sidecars; perf records
it under `diagnostics.<config>.gate` and prints it under the table. **If that
verdict is missing or is not `gated`, the number beside it is not a gated
number.** `not-expected` is a pass: eager, or `quant_ratio = 0`, has no Q tier to
select over, and **eager is the supported way to ask for an ungated decode**.

An ungated *quality* run is the harder version of this failure — it scores
normally, and quality moves for a hundred reasons — which is why the verdict is
recorded rather than inferred from the config.

### There is no on/off knob, by design

`quant_sketch_enabled` is derived (`quant_ratio > 0`). `quant_gate_ratio = 1.0`
already expresses "read everything" through the same code, which makes it a
*provable* no-op and therefore the control arm. Shipped operating point: **0.25**
(`utils.config.OPERATING_POINT`).

## The finding that governs decode work

**A card must be read for every window on every step, while a window is only
opened if selected. So the gate costs `card + ratio * window` against `window`,
and break-even is `1 - card/window`.** That one line is the whole economics, and
it is why the card's *width* — not the gate's cleverness — decides whether the
feature pays.

At int8 the card was 400 B/head against the 804 B of token data it stands in for
(512 B of 2-bit K+V plus a 292 B grid): **78% of what it replaces**, break-even
at a 0.50 read ratio, and measurably worth nothing at the shipped 0.25. The
summary was written at four times the precision of what it summarised.

**`mu` and `vbar` are now int4** (packed two per byte), which is the §6.1
frontrunner landed:

| | int8 | **int4 (now)** |
|---|---|---|
| card / head | 400 B | **272 B** |
| `bytes_per_q_window` | 9,632 B | **8,608 B** |
| break-even ratio | 0.50 | **0.66** |
| fp/q price ratio | 3.40 | **3.81** |

`v` stays int8 and must: it is the deviation *direction*, and at int4 the hot
token stops being the best-fit token, which is the one property the card exists
for. `t` is 10 B and cannot shrink either. Accuracy at ratio 0.25 is identical
to five decimals (0.00004 median relative output error, both encodings, 8
seeds); at 0.10 it is 0.00166 vs 0.00151. **Not yet measured on a GPU** — the
speed claim is bytes-only.

Consequences that still hold, from `DISTANCE_TO_GOAL.md` §2:

* break-even is a **read ratio**, not something small — 0.66 now, not 0.50;
* at 0.25 the gate saves ~3% of the kernel's traffic, because the compressed tier
  is only ~13% of what decode reads — the full-detail tier is the other 87%;
* **measured, `--gate-ratio 1.0` ties with 0.25 to within 0.2–1.6% of TPOT**
  (`130de24`). Selection currently buys nothing measurable;
* and it is not free: runtime address lookup makes every field a scattered fetch
  (**7.03 ms against a 0.58 ms roofline, 12× off**), plus a full-tier score-fill
  pass.

**The cause is the memory layout, not the idea.** Do not respond to this by
weakening the gate. The card is cheap now (int4, above); **what is left is §6.2,
making the selected reads contiguous** — that attacks the 7.03 ms, where int4
attacked the 3%. A window's fields live in six separate tensors plus four card
tensors, so one selected window is ten scattered fetches; interleaving them into
one per-window block is the real fix and needs a GPU to verify.

## How to read a decode number here

* **Decode barely tracks cache size.** A run holding 2.5× more keys was 22%
  *faster* (§3). What dominates is paid once per step per layer: ~300–430 µs,
  moving only 1.4× across a 32× batch range.
* **Size launches and latency separately from traffic, and do launches first.**
  Traffic is the easy number and here it is the small one. A step has ~1,650
  launches; removing 15 extern calls moved nothing.
* **B=1 is not winnable and that is fine.** Every weight is read once per step:
  16.06 GB → a ~10.3 ms/token A100 floor, independent of cache size. Claim
  *memory* at B=1, not latency (§8).
* **A change that adds per-layer host work to save Q-tier bytes is a net loss by
  default.**
* **The host path is already tight — do not go hunting launches.** The
  2026-09-18 review (`DISTANCE_TO_GOAL.md` §5.1) measured the production step at
  **3 launches and 2 copies per layer**: the gate, the decode kernel, the score
  gather, and the two K/V writes. Nothing else. No unbatched per-layer work, no
  per-step re-gather, no device sync. The two fusions still open (gate as a
  kernel prologue; score permutation into the epilogue) are worth ~2% together.
  The remaining factor of two is inside the kernel, not around it.
* **A `TorchDispatchMode` sees the eviction's op chain BEFORE Inductor fuses
  it** — counts are identical with the compile flag on and off. Eviction byte
  totals taken that way are an upper bound, not a GPU measurement.

## Two traps in the harness

* **The test suite is ~87/880 failing on this branch, and they are stale tests,
  not broken code** — deleted modules (`windowed_eager_cache`), deleted symbols
  (`_tile_override`, `_compile_evict_enabled`), deleted control arms, changed
  signatures. It is not a usable regression net until they are repaired. Repair
  the file you are about to change before changing it; `test_sketch.py` and
  `test_sketch_store.py` were done that way and are green. Two assertions in
  them had been vacuous for months because `.norm(-1)` is a p-norm over every
  axis, not `dim=-1`.
* **A perf npz is named `perf_prefill{P}_gen{G}_bs{B}.npz` — shape and batch,
  nothing about the config.** Two configs written to one `OUT_DIR` overwrite
  each other and the table prints the survivor under whichever name it finds.
  Sweeps go through `scripts/run_perf_sweep.sh`, which gives each config its own
  directory, validates the grid up front, and is resumable.

## Standing rules

* **Every perf-affecting change gets a control arm**, and a claim is stated
  against it. `--gate-ratio 1.0` is the gate's. The "24% regression" number is
  **confounded** — `ecc0792` moved budget 0.50→0.20 and local 64→128 inside the
  same range — and is a floor, not a measurement.
* **Say whether a number is measured, arithmetic, or CPU-only.**
  `DISTANCE_TO_GOAL.md` §4 keeps that table; keep it honest rather than tidy.
* **Kernel-or-error.** No silent fallbacks. A path that cannot run raises.
* **Triton cannot run on this dev box.** Kernels ship unvalidated by contract;
  the *algorithm* does not. Every kernel has a CPU reference mirroring its tiling
  and masking (`two_tier_window_reference`, `gate_reference`,
  `gated_decode_step`), and tests pin the reference.
* **Scores must not move** unless the change is explicitly a quality change with
  a LongBench run behind it. Anything that alters which windows the cache keeps
  is out regardless of what it would buy.
* **Read the invocation, not the config default.** `quant_gate_ratio` was
  rejected at load and inert in every runner once (`6b8a188`); `quant_ratio: 0.0`
  in the quality YAMLs does not mean the kernel is unreached.
* **Quality rows do not compare across `ac128e8` or `8a8cdc0`.** Both moved the
  stored format and the price of a window.
* **Two of the three baselines have never been run.** int2 KIVI and QEvict have
  no implementation (§7). Do not write "beats KIVI" anywhere.
