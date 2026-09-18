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

### There is no on/off knob, by design

`quant_sketch_enabled` is derived (`quant_ratio > 0`). `quant_gate_ratio = 1.0`
already expresses "read everything" through the same code, which makes it a
*provable* no-op and therefore the control arm. Shipped operating point: **0.25**
(`utils.config.OPERATING_POINT`).

## The finding that governs decode work

**A window's card costs 400 B/head. The 8 tokens it summarises cost 512 B, and a
full window 804 B. The card is 78% of the data it replaces.**

The tokens are stored at 2 bits per number; the card at 8, and it holds three
full-length int8 vectors. The summary is written at four times the precision of
what it summarises.

Consequences, all in `DISTANCE_TO_GOAL.md` §2:

* break-even is a **0.50** read ratio, not something small;
* at 0.25 the gate saves ~3% of the kernel's traffic, because the compressed tier
  is only ~13% of what decode reads — the full-detail tier is the other 87%;
* **measured, `--gate-ratio 1.0` ties with 0.25 to within 0.2–1.6% of TPOT**
  (`130de24`). Selection currently buys nothing measurable;
* and it is not free: runtime address lookup makes every field a scattered fetch
  (**7.03 ms against a 0.58 ms roofline, 12× off**), plus a full-tier score-fill
  pass.

**The cause is the memory layout, not the idea.** Do not respond to this by
weakening the gate; respond by making the card cheap (int4 — §6.1, the
frontrunner, accuracy measured neutral) and the selected reads contiguous (§6.2).

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
