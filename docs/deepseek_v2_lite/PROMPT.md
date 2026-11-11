# Design prompt: a faithful port of RecoverKV to DeepSeek-V2-Lite

Paste everything below the line into a new Claude Code session opened in this
repository. It is written to drive the whole port phase by phase, stopping at
each exit gate. The numbers it relies on, and the reasoning behind each default,
are in [`PLAN.md`](PLAN.md).

---

You are porting **RecoverKV**, the two-tier evicting KV cache with the rank-1
card read gate in this repository, from Llama-3.1-8B (GQA) to
**DeepSeek-V2-Lite** (Multi-head Latent Attention, MoE). The port must be
**faithful**. Every part of the method is carried over, with the same
semantics, the same contracts and the same proof that it ran. Only what MLA
forces may change, and each such change is stated, tested against a CPU
reference, and recorded.

## 0. Read before writing anything

In this order. Do not skim `CLAUDE.md`; most of it is a list of ways this
project has fooled itself before.

1. `CLAUDE.md`, all of it.
2. `DISTANCE_TO_GOAL.md` §1–§2, §4, §5.1, §8, §10.0, §11, §12, §13.
3. `README.md` §1–§3.
4. `docs/deepseek_v2_lite/PLAN.md`: the component map, re-derived economics,
   decisions D1–D13, traps 1–10 and timeline.
5. The code you are porting: `modules/quant/{sketch,quantizer,store,slots,effective,compact}.py`,
   `modules/windowed_cache/{cache,config,policy,state,scorer,flash_decode,decode_kernel,gate_kernel,score_kernel,hooks,flash_lse}.py`,
   `utils/{config,cache_factory,cache_memory}.py`, and one runner end to end
   (`modules/evaluation/longbench_runner.py`).
6. The target: `modeling_deepseek.py` and `config.json` from
   `deepseek-ai/DeepSeek-V2-Lite-Chat@85864749cd611b4353ce1decdb286193298f64c7`
   (base model: `deepseek-ai/DeepSeek-V2-Lite@604d5664dddd88a0433dbae533b7fe9472482de0`).
   Vendor the modeling file, unmodified with its Apache-2.0 header, under
   `models/deepseek_v2/upstream/`. It is the **reference oracle**, never the path we run.

Then verify the gate is wired today before touching it:

```bash
grep -rn "fused_gate" --include="*.py" . | grep -v "^./tests/"
```

## 1. What "faithful" means here

**Preserved exactly (the method):**

* windows of `ws` tokens; H2O cumulative window scores in an **fp32** running sum,
  ranked by the mean over query heads; first eviction at decode step 0;
* bands: sink + local (fp, never evictable) / fp tier (top `top_k_fp`) /
  int2 tier (next `N_q`) / dropped. The budget is split by `quant_ratio` under
  `quant_budget_mode` (`bytes` for memory and quality, `tokens` for latency
  only). The card is priced in the window. The remainder is spent and enforced
  (≥ 0.99);
* bidirectional promotion; reactivation of a dormant window is a bit flip with
  frozen codes, frozen grid and frozen card; `retain_only` frees before allocating;
* **positions are never renumbered.** Keys are stored pre-RoPE in the int2 tier
  and rotated at their frozen positions, in registers, at read;
* the int2 quantizer rules: KIVI-style affine 2-bit codes, a one-byte grid with
  an fp16 scale per `GRID_GROUP=32`, codes fit to the **stored** grid,
  `emulate_precision_casts` at the compile site;
* the rank-1 card: `k_i ≈ mu + t_i·v` over **post-RoPE** keys, `v` the
  farthest-point direction, `mu` and `vbar` as residuals from frozen anchors,
  built once at demotion from keys already in hand, and the `quant_card_bits` knob;
* the gate: card scan → `logmass`; selection by the **per-query-head share**,
  `max_h [logmass_h(w) − logsumexp_w' logmass_h(w')]`, capped per **KV head** at
  `n_sel = ceil(ratio · n_active)`; picks sorted ascending; passed to the kernel
  as an indirection, not a re-gather;
* the fill: deviation = **mean log ratio** over the read set; every skipped window
  is credited `exp(logmass + deviation)` to its score **and** to the output via
  its value centroid; output and scores renormalised by `1 + Σ restored`.
  Never a `Σ exact / Σ estimate` calibration;
* layer-major eviction (`R = L·B`), compiled-or-raise, pure with respect to the fp
  buffers, one host sync (`_evict_widths`), compaction by counting, no
  `torch.cat` in the compiled region;
* L-reuse always (the prefill softmax normaliser comes from the flash forward,
  never recomputed); the prefill score is a FA2-backward-style Triton kernel;
* **kernel-or-error** everywhere; `quant_gate_ratio = 1.0` is the control arm
  and a provable no-op; there is no on/off knob for the gate;
* the proof: `flash_decode._STATS` (`armed`, `fired`, `gated`, `windows_read`,
  `windows_active`), `expect_gated` / `gate_report` and its four verdicts, the
  gate observer, `FusedDecodeNotReached`, `EvictionRanEager`, and the sidecar
  fields `read_gate`, `quant_card_bits`, `quant_promotion`,
  `quant_promote_source`, `gate_read`;
* the ablation knobs `quant_promotion: oneway` and
  `quant_promote_source: original`, with their semantics unchanged;
* the operating point `utils.config.OPERATING_POINT` (budget 0.20, ws 8, sink 5,
  local 128, q 0.70 `bytes`, gate 0.25).

**Allowed to change, because MLA forces it.** Each one is score-neutral against
its CPU reference unless it is listed as a quality change:

* what a token stores (the latent), the int2 layout, the card's field layout;
* the model seam (an owned attention forward instead of a monkeypatch);
* kernel geometry (split-K, D=576/512, `rep=16`);
* byte pricing.

**Quality changes** need a LongBench and a RULER run behind them before they
ship. Examples: switching the latent grid to per-token, changing the read ratio,
dropping `v` to int4. A quality change is never folded into a layout commit.

## 2. The MLA specification

Per layer ℓ, row `b`, token `t`, query head `h ∈ [0, 16)`:

```
c_t   = kv_a_layernorm(kv_a_proj_with_mqa(x_t)[:512])          ∈ R^512   (no RoPE)
r_t   = RoPE_π(kv_a_proj_with_mqa(x_t)[512:], pos_t)            ∈ R^64    (post-RoPE)
q_h   = q_proj(x_t) viewed [16, 192] → q_nope,h ∈ R^128, q_pe,h ∈ R^64
q_pe,h ← RoPE_π(q_pe,h, pos_t)
W = kv_b_proj.weight.view(16, 256, 512):  W_UK,h = W[h, :128],  W_UV,h = W[h, 128:]
q̃_h  = [ W_UK,hᵀ q_nope,h  ‖  q_pe,h ]                           ∈ R^576  (absorbed query)
key_t = [ c_t ‖ r_t ] ∈ R^576            value_t = c_t ∈ R^512  (same storage)
logit_h,t = s · q̃_h · key_t,     s = module.softmax_scale  (= 192^-0.5 · mscale² = 0.11472)
o_h   = W_UV,h · Σ_t p_h,t c_t ;   out = o_proj(concat_h o_h)
```

* **One latent "KV head" per layer: `H_kv = 1`, `rep = 16`.** The card, the
  anchor, the gate's cap and the decode program all bind to it.
* `π` is DeepSeek's interleave permutation (`view(d/2, 2).transpose(-1, -2)`). Apply
  it **once** when `r_t` and `q_pe` are produced, so every downstream RoPE is the
  half-split form the kernels already implement. Dot products are invariant
  under a shared permutation. Pin this with a test against the vendored
  `apply_rotary_pos_emb`.
* YaRN `cos`/`sin` tables are indexed by **absolute position** and never sliced
  by a held length. Keep the `_mscale` factor general (it is 1.0 for V2-Lite
  because `mscale == mscale_all_dim`).
* `s` comes from the module. **Never** derive it from a cache's head dim:
  `cache.py:1110` (`head_dim ** -0.5`) is wrong here by 1.59× at 192 and 2.75× at 576.
* **Prefill** runs non-absorbed: decompress per-head K (192) and V (128, padded
  to 192 for FA2) for the prompt in hand. Call `flash_attn_func(...,
  softmax_scale=s, causal=True, return_attn_probs=True)` **from our own
  forward** to get `out, softmax_lse`. Run the score kernel at `D=192`, `rep=1`,
  reusing that LSE. Cache only `c` and `r`.
* **Decode** runs absorbed, through `_run_fused`'s contract (raise if a fused
  layer arrives without cards; bump the same `_STATS`).
* **Eager** is the materialize path. It builds `[c ‖ r]` for the fp tier plus the
  dequantized int2 windows and computes absorbed attention in PyTorch. It is the
  CPU oracle and the supported ungated decode, and `expect_gated` returns False
  for it, exactly as today.

### The int2 tier on the latent

* `c` is stored **once** and serves as K (its first 512 dims) and V. Default
  granularity: **per-channel over the window's tokens** (KIVI's key rule: `c` is
  a key first, and its errors are exponentiated). The alternative is per-token
  with 32-channel scale groups. Choose it only from the P0 measurement, and
  treat the switch as a quality change.
* `r` (64) is stored int2, per-channel, **pre-RoPE** (un-rotated at demotion
  from the post-RoPE copy already in hand; rotated in registers at read), as
  keys are today.
* Layout default: one **contiguous payload block per window** (all codes and
  grids) and one **contiguous card array**, so the card scan streams and a
  selected window is a single gather instead of ten (`DISTANCE_TO_GOAL.md` §6.2). This
  is layout only. The CPU reference must agree bitwise with an SoA mirror of the
  same data.
* Price it in the format's own module (`bytes_per_q_window_mla`,
  `sketch_bytes_per_mla_window`), and have `resolve()` call those functions.
  Never re-derive bytes in the resolver. `bytes_per_token = (512 + 64) · 2 = 1152`;
  **never** from `config.num_key_value_heads` (16, meaningful only for the
  decompressed path; it misprices by 7.1×).

### The card on the latent

* Built from the post-RoPE `key_t = [c_t ‖ r_t]` (576) at demotion. Fields:
  `mu` (int4, residual from a per-layer anchor ∈ R^576), `v` (int8), `t` (int8).
* **`vbar ≡ mu[:512]`** because the value is `c`. Do not store a `vm` field.
  Price it at zero, and keep `CardBits.vm` as an alias that must equal
  `CardBits.mu` (raise otherwise). `v_anchor = anchor[:512]`.
* `mu`'s 576 dims mix latent and RoPE scales. If the RoPE slice dominates `amax`,
  give `mu` two fp16 scales (512 | 64) for 2 B. Decide it in P0 from the fixtures.
* Re-measure `v` at int4 vs int8 on real latent keys (it is 576 B, 66% of the card).
  The Llama finding that `v` must be int8 is evidence, not a law, on a 576-d key.

### The gate on the latent

* `logmass_h(w) = logsumexp_i s·(q̃_h·(anchor + mu_w) + t_w,i · q̃_h·v_w)`, the same
  two dots at D=576. The share-union is over all **16** query heads, and the cap
  binds on the single latent head. Capping per query head and unioning
  afterwards would let ratio 0.25 read the whole tier (0.25 × 16 ≥ 1).
* `_gate_share_kernel` at `rep=16` takes one program per row.

### The decode kernel

* Grid `B · n_splits`. **Split-K is mandatory**: one KV head means B programs,
  so at B=1 a single program would walk the whole cache. Align split boundaries
  to window boundaries so a window's score is produced by exactly one split.
* `BLOCK_R = 16` exactly fills the MMA (on Llama it padded 4 heads to 16).
  The accumulator is `[16, 512]` fp32. Expect register pressure; the tile
  ladder (keys per tile, stages, warps, splits) is **timed, never first-fit,
  never extrapolated across geometries** (§10.3, §10.15).
* Per selected int2 window: unpack and dequantize the `c` tile **once** and use
  it for both QK (with the dequantized, in-register-rotated `r` tile) and PV.
* The **fill and the window-score epilogue move into the split-K combine
  launch**, because the deviation is a mean over the read set across all
  splits. The combine computes global `m`/`l`, the deviation, the skipped
  windows' credit, their centroid contribution `exp(logmass+dev)·(anchor[:512] +
  mu_w[:512])`, and the renormalisation. Write a CPU reference that mirrors
  the split and the combine (`mla_two_tier_window_reference`).
* Count launches per layer per step and report them next to Llama's measured
  "3 launches + 2 copies": absorb-q, gate scan, share-union, decode, combine,
  `W_UV`, score gather, and the two latent writes. Fold where score-neutral;
  never add per-layer host work.

### The model seam

* Install an **owned MLA attention forward** at load on every
  `model.model.layers[i].self_attn`. Recognise the class by name
  (`DeepseekV2Attention` / `DeepseekV2FlashAttention2`) **and**
  `config.model_type == "deepseek_v2"`, and raise on anything else. Extend
  `_get_attn_classes` so a CPU run cannot silently install nothing.
* The remote module imports `flash_attn_func` into its own namespace, so any
  patch of transformers' copy is a silent no-op. Do not patch; own the call.
* Keep `_run_fused` (or a function with its exact contract) as the **only**
  decode read path when a Q tier exists, sharing `flash_decode._STATS`, so
  `stats()`, `expect_gated()` and `gate_report()` stay the single source of the
  verdict.
* Position and length contract with the remote code. Audit every call site at
  the pinned revision: `get_usable_length`, `seen_tokens`, `get_max_length`,
  `get_seq_length`. Define each on `WindowedCache` deliberately (seen =
  monotonic, held = shrinking). Pass absolute positions in `cache_kwargs` so
  `_check_position_contract` has something to check. Equal-length prompts
  only; a padding mask on the fused path raises.
* Model dtype bf16. RoPE tables at the store dtype, unless P0 shows the bf16
  demote→promote round trip on `r` moves keys measurably, in which case use fp32 RoPE in
  **both** tiers.

## 3. Work plan, one exit gate per phase

Work on a new branch. Commit at each gate. Keep
`docs/deepseek_v2_lite/STATUS.md`, in the style of `DISTANCE_TO_GOAL.md` with its
measured / arithmetic / CPU-only table, current at every gate. **Repair a stale
test file before changing the module it covers** (`CLAUDE.md`, "Two traps in the
harness").

**P0. Ground truth and claim map** (needs ~1 GPU-day; ask the user for it).
Load at transformers 4.47.1 with `trust_remote_code` at the pinned revisions and
record any API drift. Run a FullKV smoke: 2 LongBench sets, RULER 4k and 32k
NIAH. Dump fixtures for 4 layers at 4k and 32k: `c`, `k_pe` pre- and post-RoPE,
the absorbed `q̃`, and true per-head attention. Keep them small enough to commit,
or store a hash and a path. On CPU, from the fixtures, measure D4 (latent grid),
D5 (`r` precision), D7 (`v` width; `mu`'s split scale), D8 (per-query-head gate
recall at `rep=16`, worst head and tail fractions, at ratios
0.10/0.25/0.50/1.0, against a same-size hindsight pick) and D9 (bf16 RoPE round
trip). Confirm or correct `PLAN.md` §3.3.
*Gate:* every decision has a number or says "default, unmeasured". **Stop and
report** if per-head recall at 0.25 is materially worse than the §12 Llama
table, or if the claim map rules out a latency claim at every shape the GPU can
hold.

**P1. Geometry, adapter, CPU oracle** (CPU). Add `models/deepseek_v2/` with the
vendored upstream file, an `MLAGeometry` dataclass (every dim, `s`, the RoPE
convention), the owned forward, and absolute-position YaRN. Generalise the
store, quantizer, slot table, card and budget from `(H_kv, D)` to a geometry
object, keeping Llama byte-identical. The Llama tests are the regression net.
Add the MLA materialize path. Add one model-loading helper that every runner
uses (`trust_remote_code`, pinned revision, bf16, install the owned forward,
record `model_family` and the geometry in sidecars); never thread it by hand
into some runners and not others (`CLAUDE.md`, "Read the invocation"). Check BOS
handling in `utils/prompting.py` for this tokenizer, and add LongBench max-length
entries for a 32k context.
*Gate:* on a tiny random DeepSeek-V2 config (2 layers, 4 heads, latent 64, rope
16), the owned forward equals the vendored forward (fp32 exact, bf16 within a
stated tolerance). RoPE rotate/unrotate is bit-identical to the vendored
function, permutation included. `bytes_per_token == 1152` and the window/card
byte counts are pinned. Eager end-to-end `generate()` runs through ≥ 3 evictions
with monotonic positions. At budget 1.0 and q = 0 the output equals the vendored model.

**P2. Kernels, interpreter first** (CPU, `TRITON_INTERPRET=1`). The prefill score
kernel at D=192 (128 + 64 tiles), `rep=1`, with the LSE from our own flash call;
the gate scan and share-union at `H_kv=1`, `rep=16`, D=576; the MLA two-tier
decode with split-K and the combine. Each gets a CPU reference that mirrors its
tiling and masking, and tests pin the reference.
*Gate:* interpreter runs agree with the references, the kernels compile for sm80
(`tests/test_kernel_compiles.py`), and `tests/interp_card_kernels.py` has MLA
cases at the default and non-default `quant_card_bits`.

**P3. Compiled eviction and end-to-end** (CPU, then 1–2 GPU days). Layer-major
eviction over the latent: promote (dequantize, rotate `r` at frozen positions),
demote (un-rotate `r`, quantize, build the card from the post-RoPE key in hand),
reactivate, per-layer anchors frozen at the first eviction, and the shadow
store for `quant_promote_source: original`, outside the budget.
*Gate:* flash vs eager per-step parity (K/V, active window ids, effective
length), the analogue of `test_flash_eager_two_tier_parity`. `quant_gate_ratio
= 1.0` reproduces the full read exactly. `gated == fired` and `read_fraction`
tracks the ratio. The `grep` above shows a non-test caller. No
`EvictionRanEager`, and `profile_decode.py` shows nonzero Inductor time.

**P4. GPU bring-up and tuning** (1 GPU, ~3 days). Kernels against references on
GPU, timed ladders for every kernel with each rung's time printed, and
`profile_decode.py` buckets for MLA.
*Gate:* GPU matches references; each kernel's time against its own roofline.

**P5. Baselines and perf harness** (1 GPU). FullKV = latent cache + absorbed
decode from a maintained kernel that runs on this GPU. The remote-code
`DynamicCache` (12,288 B/token/layer) appears only as a labelled context row.
Use one **sync-free MoE** path for every arm, recorded as `moe_path`; the remote
`moe_infer` syncs once per MoE layer. Run the perf grid from `PLAN.md` D11
through `scripts/run_perf_sweep.sh` (one directory per config), with the
`--gate-ratio 1.0` control arm.
*Gate:* every row carries a `gated` verdict, and every claim is stated against
the control arm.

**P6. Quality** (8 GPUs, ~3 days). Run `SMOKE=1 scripts/run_recoverkv_all.sh`
first, then LongBench (16 sets; budgets 0.05/0.10/0.20), RULER 32k (13 tasks),
GSM8K, E1/E2/E3, and observations I–V. Observation IV is the per-head read ledger
at `rep=16`.
*Gate:* every sidecar records `read_gate == gated` (or `not-expected` for eager
and FullKV), plus `quant_card_bits`, `quant_promotion`, `quant_promote_source`,
`model_family`, the latent grid choice and `moe_path`.

**P7. Write-up.** `STATUS.md` with the measured / arithmetic / CPU-only table,
kept honest rather than tidy.

## 4. Rules that carry over, restated for this port

* **Say whether every number is measured, arithmetic, or CPU-only.** Speed
  claims from bytes alone are labelled as such.
* **Every perf-affecting change gets a control arm.** For the gate it is
  `--gate-ratio 1.0`.
* **Launches before traffic.** Size launches and latency separately from bytes,
  launches first. A change that adds per-layer host work to save int2 bytes is a
  net loss by default.
* **Anything that compares or adds attention quantities across query heads
  normalises per head first.** This covers selection and recall metrics alike. At
  `rep=16` a raw-logit union or a summed-raw-mass recall metric is the §12 defect
  waiting to recur.
* **Scores must not move** unless the change is declared a quality change with
  runs behind it. Layout, seam and kernel changes are score-neutral, and their
  CPU reference proves it.
* **Do not write "beats KIVI" or "beats QEvict".** Neither baseline exists, on
  either model.
* **Do not compare a DeepSeek number to a Llama number** as if it were the same
  method at a different size. The cache, the KV share of traffic and the
  baseline all differ (`PLAN.md` §3.3–§3.4).
* **B=1 is not a latency claim.** The floor is ~3.2 ms; claim memory there.
* **Stop and ask** before buying GPU time, before any quality change, and when a
  P0 measurement contradicts a default in `PLAN.md` §4.

## 5. What not to do

* Do not cache the decompressed per-head K/V. It is 10.7× the latent and turns
  every compression ratio into a comparison with a straw man.
* Do not monkeypatch `flash_attn_func` to reach this model; own the forward.
* Do not compute RoPE with Llama's `apply_rotary_pos_emb`, and do not index
  `cos`/`sin` by held length.
* Do not price the cache from `num_key_value_heads`, or the softmax scale from a
  head dim.
* Do not add an ungated fused arm, a PyTorch decode fallback, or an environment
  knob that selects a different path. Eager is the ungated path; a fused layer
  without cards raises.
* Do not weaken the gate to make it pay. If the card's economics are poor on the
  latent, the levers are its width (`quant_card_bits`) and the layout, as on Llama.
