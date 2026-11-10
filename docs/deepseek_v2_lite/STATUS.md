# DeepSeek-V2-Lite port: status

Updated 2026-11-10 (verification pass, §6). The plan is [`PLAN.md`](PLAN.md); the design prompt is
[`PROMPT.md`](PROMPT.md). This is the `DISTANCE_TO_GOAL.md` of the port: what
landed, what each claim rests on, and what has not been run.

**Nothing here has run on a GPU or on the real 16B checkpoint.** Every check
below is on CPU: a tiny random DeepSeek-V2 built from the vendored upstream
code, the real `config.json` for byte arithmetic, and the Triton kernels under
`TRITON_INTERPRET=1`.

---

## 1. The mapping that makes the port faithful

An MLA layer is handed to the shipped cache as **one KV head of width
`Dc + Dr = 576` whose value is its latent dims** (`models/deepseek_v2/geometry.py`):

* `H_kv = 1`, `H_q = 16`, so `rep = 16`. The card, anchor, gate cap and decode
  program bind to that one latent head.
* **Packed key layout** `[c[:256] | r[:32] | c[256:] | r[32:]]`. The cache's HF
  half-split RoPE then pairs latent with latent (`cos = 1, sin = 0`, an exact
  identity) and RoPE with RoPE (DeepSeek's own tables). Every existing
  rotate/unrotate primitive runs **unmodified** and rotates only the 64 RoPE
  dims, bit for bit as upstream.
* **Value aliasing** (`kv_value_alias`). The fp tier stores the latent once (no
  value buffer). The int2 tier quantizes it once, as a key (per channel,
  pre-RoPE). The card's value centroid **is** `mu`.

So `_evict_two_tier_impl`, the policy, the slot table's bookkeeping, promotion,
demotion, reactivation, `retain_only`, the layer-major join, the compiled body,
the budget resolver's split and the gate's share-union are the shipped code.
The only change to the eviction body is that it skips the value-side gathers
when the value is an alias; that branch is a trace-time constant.

---

## 2. What is verified, and how

| claim | status | evidence |
|---|---|---|
| Owned MLA forward == upstream at full cache (latent `DynamicCache`, absorbed decode) | **CPU, fp32**: max abs logit difference < 1e-5, greedy tokens identical | `test_owned_forward_matches_upstream_at_full_cache` |
| RecoverKV cache at budget 1.0, q = 0 == upstream | **CPU, fp32**: max abs logit difference < 1e-5 | same test |
| DeepSeek RoPE through the cache's own `rotate/unrotate_key_window` | **bit-identical** to upstream `apply_rotary_pos_emb` (fp32 and bf16); latent dims untouched | `test_rope_is_upstreams_bit_for_bit` |
| Prefill eviction score == the model's own attention | **CPU**: equals upstream eager probabilities summed over queries, per head, windowed | `test_prefill_scores_are_the_models_own_attention` |
| `out.attentions` == upstream at full cache | **CPU**, prefill and decode | `test_output_attentions_match_upstream_at_full_cache` |
| **Aliasing changes no eviction decision** | **CPU**: identical fp keys, positions, window ids, scores, active sets, int2 key codes, grids, cards (`mu`, `v`, `t`), anchors and slot tables vs a twin that stores an explicit value copy, over 40 steps with promotions and fresh demotions both exercised | `test_aliasing_changes_no_eviction_decision` |
| Layer-major == per-layer eviction (the shipped pin) | **CPU**: 44/45 pass after repairing the stale knob. The one failure (`test_cache_is_freed_by_refcounting`) also fails at the base commit | `tests/test_layer_major_evict.py` |
| RecoverKV runs on the latent through many evictions | **CPU**: compiled layer-major eviction, monotonic positions, no value buffer, no value codes, no `vm` card field | `test_recoverkv_runs_through_evictions_on_the_latent` |
| `generate()` through the runner path | **CPU**: `model.config` + rotary module into `WindowedCache` + `install_score_hooks` + `generate()` decodes token for token what an explicit-position loop decodes | `test_runner_path_generate_matches_the_explicit_loop` |
| Gate at `rep = 16` | **CPU**: one cap per latent head; selection == `select_windows(group_share(logmass, 1))`, the shipped rule | `test_gate_selects_per_latent_head_over_all_query_heads` |
| Gate Triton kernel at D = 96, rep = 16 | **interpreter**: `logmass` vs `gate_and_score` 3e-6; pick == share-union reference | `tests/interp_mla_kernels.py` |
| MLA fused decode kernel | **interpreter**: latent output vs `two_tier_window_reference` within fp16 rounding (7.8e-3 abs on fp16 output); every window score within 4.5e-4; gated and ungated | same |
| Fused path end to end (owned forward → `run_pending` → `_run_fused` → both kernels → scores) | **interpreter**: at gate 1.0 equals the materialize path, max abs logit difference 1.3e-7 over 14 steps across evictions; at 0.25 `armed == fired == gated`, verdict `gated` | same, `ok e2e` |
| Bytes at V2-Lite's real config | **arithmetic, through `resolve()`**: 1,152 B/token; int2 window 2,376 B + card 878 B; `top_k_fp = 27`, `N_q = 184` at 4096/256; utilisation 0.998 | `test_latent_bytes_at_the_real_config` |
| Raw DeepSeek config refused by `resolve()` (would price 8,192 B/token) | **CPU** | `test_raw_deepseek_config_is_refused` |
| Softmax scale = 0.11472 (module's, not a head dim's) | **CPU** | `test_softmax_scale_is_upstreams`; `install_recoverkv_mla` asserts module == geometry |

The vendored upstream files are hash-pinned (`test_vendored_upstream_is_unmodified`),
and on 2026-11-10 they were re-downloaded from the pinned Hub revision and hash
identically (`modeling_deepseek.py`, `configuration_deepseek.py`, `config.json`).

**At V2-Lite's real MLA dims** (§6.1: `kv_lora_rank` 512, RoPE 64, nope 128,
v 128, 16 heads, YaRN; only width, depth and the MoE shrunk), against the
UNMODIFIED pinned remote code loaded with `trust_remote_code`:

| claim | status | evidence |
|---|---|---|
| Owned forward == upstream, latent FullKV and RecoverKV at budget 1.0 | **CPU**: fp32 max abs logit diff 7.5e-7; bf16 8.3e-3 (one bf16 step at that scale); greedy `generate()` tokens identical in both | §6.1 |
| Gate + MLA decode kernels at D = 576, halves 256 + 32, rep = 16 | **interpreter**: gate logmass 7.6e-6 vs `gate_and_score`; latent output and every window score vs `two_tier_window_reference`, gated and ungated | `tests/interp_mla_kernels.py` (`ok real dims`) |
| Fused route through `generate()`, cache built as the runners build it | **interpreter**: gate 1.0 == materialize, fp32 8.9e-7 / bf16 1.6e-2 (argmax identical); gate 0.25: 93/93 fused layers gated, read fraction 0.250, verdict `gated`, no NaN | §6.1 |
| Compiled eviction over a long decode | **CPU**: 160 steps, 22 compiled evictions, 9 Dynamo graphs, no recompile after the fill phase | §6.1 |

**Regression net** (CPU, torch 2.15, transformers 4.47.1, Triton 3.9):

| | `tests/` | `modules/evaluation` + `data` |
|---|---|---|
| base commit `78a995b` | 93 failed / 1071 passed | 36 failed + 6 errors / 421 passed |
| this branch | **64 failed / 1122 passed** | 36 failed + 6 errors / 421 passed (identical set) |

**No test that passed at the base fails here.** Every one of the 64 failures was
already failing at the base commit: these are the stale tests `CLAUDE.md` describes.
That includes the shipped kernel's interpreter test, which hits a Triton 3.9
interpreter quirk with Python-int loop indices (the MLA kernel avoids it). The
+51 passes are the 22 new DeepSeek tests and 29 repaired layer-major tests.

---

## 3. Decisions (PLAN.md §4) as implemented

| id | implemented as | measured? |
|---|---|---|
| D1 cache the latent | yes. Both RecoverKV and the FullKV baseline hold 1,152 B/token | arithmetic + CPU |
| D2 absorbed decode / non-absorbed prefill | yes | CPU equivalence |
| D3 owned forward | yes (`models/deepseek_v2/attention.py`); `flash_decode.run_pending` is the second door into the same `_run_fused` | CPU + interpreter |
| D4 latent int2 granularity | **per-channel** (the key rule), the faithful default | **not measured** on real activations. Per-token is not implemented |
| D5 `k_pe` storage | int2, per channel, pre-RoPE, de-interleaved order | **not measured** on real activations |
| D6 store layout | **the shipped SoA slot table**, value columns absent. The per-window interleaved block (§6.2) was **not** done | — |
| D7 card widths | mu 4 / v 8 / t 8, `vm` aliased to `mu` (priced zero, `resolve()` refuses vm ≠ mu) | **not re-measured** on a 576-d key |
| D8 gate rule | share-union unchanged; ratio 0.25 | per-head recall at rep = 16 on real attention: **not measured** |
| D9 dtype / RoPE precision | bf16 store, RoPE tables at the store dtype | synthetic N(0,1) keys: bf16 demote→promote RoPE round trip ≤ 0.016 abs (the latent dims are exact). Not measured on real keys |
| D10 operating point | unchanged (`configs/longbench_deepseek_v2_lite_flash.yaml` passes `check_operating_point.py`) | — |
| D11 perf grid | **not done** | — |
| D12 sync-free MoE | **not done**. The remote `moe_infer` still syncs once per MoE layer | — |
| D13 FullKV baseline | latent `DynamicCache` through the owned forward (`configs/longbench_deepseek_v2_lite_full_cache.yaml`). A third-party MLA decode kernel baseline is **not** wired | CPU equivalence to upstream |

---

## 4. What changed in shared code (Llama path unchanged)

Every change is gated on `kv_value_alias` or `model_type == "deepseek_v2"`.
With neither set, the code takes its original path.

* `state.py`: `CacheState(value_alias=)` keeps no value buffer, and `value_states` is the key view.
* `quant/slots.py`, `quant/store.py`, `quant/sketch.py`, `quant/quantizer.py`: an aliased table holds no value codes, grid, centroid or shadow. Value reads return the key; `value_centroid` falls back to `mu`; pricing drops the value terms.
* `quant/effective.py`: an aliased effective value is the effective key, built once.
* `windowed_cache/config.py`: alias-aware `bytes_per_token`, card and window pricing; refuses a raw DeepSeek config and an aliased card with `vm ≠ mu`.
* `windowed_cache/cache.py`: reads the latent view; the stated softmax scale replaces `head_dim**-0.5` when given. It converts a raw DeepSeek config + rotary module at its boundary, skips value-side ops in the eviction under alias, and passes `value_alias` / `mla_split` to the fused hand-off. The gate's centroid under alias is `(mu_q, mu_s, anchor)`. It adds `get_max_cache_shape` (None) and `_seen_tokens` (monotonic), which the remote generation code calls.
* `windowed_cache/flash_decode.py`: `pending()` / `run_pending()` reach the existing `_run_fused` from an owned forward.
* `windowed_cache/decode_kernel.py`: `fused_two_tier_decode` routes an aliased tier to `mla_decode_kernel.mla_decode_triton`.
* `windowed_cache/mla_decode_kernel.py` (new): the shipped kernel with three changes:
  * each half is tiled as latent (HA) + RoPE (HB) lanes, both powers of two;
  * RoPE is applied on the HB lanes only;
  * V comes from the dequantized latent tile.
* `windowed_cache/gate_kernel.py`: the card-layout check skips an absent centroid.
* `windowed_cache/hooks.py`: `install_score_hooks` delegates a DeepSeek model to the owned forward and arms the fused decode with the same Triton-or-raise gates.
* `utils/model_loading.py` (new) + 8 runners/scripts: one loader. It runs `trust_remote_code` at the pinned revision and installs the owned forward, for DeepSeek only.
* `tests/test_layer_major_evict.py`: repaired. It selected the per-layer reference arm through a deleted env knob.

---

## 5. Not done, in the order it should happen

(§6.3 adds what the 2026-11-10 verification pass found and did not fix.)


1. **P0 on a GPU, with the real checkpoint.** Load at 4.47.1, FullKV smoke, and activation fixtures. Measure D4, D5, D7, D8 and D9 on real latents and real attention.
2. **The kernels on a GPU.** Compile and run the MLA decode and gate kernels against the CPU references, and time the tile ladder. The prefill path (FA2 + score kernel on per-head 192, zero-padded to 256) has never executed.
3. **Split-K for the MLA decode kernel.** With one KV head the grid is `B` programs, so a GPU at small B is mostly idle. The fill and score epilogue must move into a combine launch.
4. **Sync-free MoE** for every perf arm, then the perf grid (16k / 32k prompts) with the `--gate-ratio 1.0` control.
5. **Quality**: LongBench / RULER 32k / GSM8K with `read_gate == gated` in every sidecar, plus the observation suite (per-head read ledger at rep = 16).
6. The per-window interleaved layout (§6.2) and a third-party latent FullKV decode baseline.

Quality and latency numbers from this port do **not** compare with Llama rows. The cache, the KV share of traffic and the baseline all differ (PLAN.md §3).

---

## 6. Verification pass, 2026-11-10

Asked: will the branch produce verified LongBench and GSM8K results, does it run
without runtime errors, is it faithful to `Clustered_int2_paused_fast`, and is it
correct to DeepSeek-V2-Lite's structure. Everything here is CPU or the Triton
interpreter; nothing has run on a GPU or the real weights.

### 6.1 How it was checked

* A tiny random checkpoint with V2-Lite's **real** MLA dims (above), the pinned
  remote code and the real Lite-Chat tokenizer, saved to disk and loaded through
  `utils.model_loading.load_causal_lm` exactly as the runners load it.
* `main.py` end to end on it: LongBench (windowed eager and latent FullKV, on
  synthetic and on real LongBench JSONLs with middle truncation) and GSM8K (on the
  real 1,319-problem CoT set), then `longbench_score` / `gsm8k_score`.
* The fused/gated route under `TRITON_INTERPRET=1`, through `generate()`, with the
  cache constructed as `LongBenchRunner._setup_windowed_cache` constructs it.
* Every LongBench example (3,750) prompted as `LongBenchRunner._predict` prompts
  it, with the real tokenizer: all resolve with an int2 tier, none exceeds 32,768
  tokens with its generation, none tokenizes to a duplicate BOS.
* Regression net: `tests/` fails 64 at this branch vs 93 at the base; every
  failure here also fails at the base. `modules/evaluation` + `data`: the same 34
  failures at both. After the §6.2 fixes, the affected suites (the DeepSeek and
  MLA-kernel tests, layer-major eviction, quant cache, sketch, card bits,
  prompting, dead-call-sites, utils, `modules/evaluation`, `data`; 732 tests) fail
  24, every one of them also failing at the base.

### 6.2 Defects found and fixed

| defect | effect | fix |
|---|---|---|
| `GSM8KRunner._setup_windowed_cache` resolved the budget against the RAW DeepSeek config | **every windowed DeepSeek GSM8K run raised on its first example** (`resolve()` refuses a raw config) | read `cache.resolved`, which the cache resolves on the latent view; identical for Llama |
| GSM8K compression diagnostic reported `total_budget_tokens` as retained | under-reports what the cache holds whenever sink + local exceed the budget (the GSM8K regime) or int2 windows buy extra tokens | report `resolved.retained_tokens`; record `int2_windows` per example and `n_examples_with_int2_tier` |
| the latent FullKV baseline ran the Triton score kernel at prefill | a second O(N^2) pass per layer on the baseline, inflating its prefill time; outputs unchanged | score only for a RecoverKV cache |
| `ensure_table` claimed upstream's table "never" regrows | it regrows on every decode step in every layer (upstream sets `max_seq_len_cached = None` after init); ~20 kernels per layer per step | grow in 4,096-position chunks; every row bit-identical (tested) |
| `PackedRope` index buffers built on CPU | an H2D copy from pageable memory (a stream sync) on every call, inside the compiled eviction | built on the rotary module's device |
| no DeepSeek GSM8K configs, no DeepSeek launchers, no sidecar check | nothing to run GSM8K with; nothing refused an ungated result | `configs/gsm8k_deepseek_v2_lite_{budget20,full_cache}.yaml`, `scripts/run_longbench_deepseek_v2_lite.sh`, `CFG_PREFIX`/`ARMS` in `scripts/run_gsm8k_e2e.sh`, `scripts/check_read_gate.py` |

### 6.3 Found and NOT fixed (decisions, or GPU-only)

* **GSM8K at the operating point barely exercises the method.** Measured with the
  real tokenizer on all 1,319 problems: prompts are 110-276 tokens, so
  `0.20 * (prompt + 512)` is ~131 tokens at the median, below sink + local = 133.
  Only **231** problems (prompt >= 168 tokens) get an int2 tier, and **1** gets an
  evictable fp window; the rest run as a sink + recent-128 cache. On a GPU the
  verdict will read `gated` because those 231 fire, so `meta.json` now records
  `n_examples_with_int2_tier`. The Llama GSM8K configs pin the same point. Moving
  it is an operating-point decision with quality runs behind it, not a fix.
* **31.5k-token prefill needs an 80 GB card.** The remote `DeepseekV2ForCausalLM`
  computes fp32 logits for every prompt position (~13 GB at 31.5k, plus the bf16
  copy) on top of ~31 GB of weights. Same for both arms; `skip_oom: false`.
* **The eager backend cannot run LongBench lengths.** Its prefill materialises
  `[16, T, T]` fp32 probabilities (as upstream eager does). The shipped configs
  use flash; eager is for CPU tests.
* **GPU-only paths still unexecuted:** FA2 prefill + score kernel at the
  zero-padded 256 head, the gate kernel at `BLOCK_D = 1024`, the MLA decode
  kernel's lowering, register pressure and tile ladder (`CLAUDE.md`: Triton
  cannot run on this box). The interpreter checks arithmetic, not lowering.
* **Perf tooling prices DeepSeek from the raw config** (`modules/evaluation/memory_model.py`
  reads `num_key_value_heads`). Not on the LongBench/GSM8K path; the perf grid
  (D11) is not done.
* **The base branch moved on.** `Clustered_int2_paused_fast` has three newer
  commits (the Reviewer-2 completion-term ablation). The shipped decode is
  unchanged by them (`quant_completion = null` is the pre-knob call), so the
  method this branch ports is current; the ablation knob itself is not ported
  and would need `out_dtype` in the MLA kernel.

### 6.4 The paper run: LongBench at a 10% budget on Colab (2026-11-10)

`notebooks/longbench_deepseek_v2_lite_b010_colab.ipynb` runs QEvict at `cache_budget 0.10` (the paper's Table 2
"10%" row; local 128, everything else the operating point) and the latent Full-KV reference on the 11 Table 2
LongBench tasks other than NarrativeQA, on one pinned commit, checkpoint revision and stack (torch 2.6.0, Triton
3.2.0, transformers 4.47.1, flash-attn 2.7.4.post1, A100 80GB). It runs the benchmark only: one example per
`generate()` call (batch size 1, as in the paper's LongBench runs), results per task on Drive, resumable by task, and
each QEvict task's sidecar must report `read_gate: gated`. Larger batches are not offered: the cache batches only
equal-length prompts, the owned MLA forward rejects padding masks, and at 31.5k tokens the prefill's fp32 logits are
~13 GB per row (arithmetic), so even 4 rows exceed 80 GB with the weights.

* **`scripts/deepseek_gpu_preflight.py`** is run separately, before GPU time is spent on the benchmark. On a
  random model with the checkpoint's own attention geometry it checks, on the GPU: the owned forward (FA2 prefill)
  against upstream; the Triton score kernel against eager scores; the fused decode against the materialize path, in
  bf16 and in fp32 (per-layer attention outputs and per-step window scores, not just logits, which in a random model
  hide kernel errors); the gate verdict at rho = 0.25; a compiled eviction. Under the interpreter
  (`tests/test_deepseek_gpu_preflight.py`) it catches a 1% error injected into the MLA kernel's output.
* **On torch 2.6 (the GPU stack), CPU:** `tests/test_deepseek_mla.py` passes, including the compiled eviction over
  the aliased latent, with `ATEN_CPU_CAPABILITY=default`. Without it torch 2.6's CPU Inductor emits an AVX-512
  `convert<uint8_t,1,int64_t,2>` that does not compile (a CPU codegen bug; the GPU compiles through Triton).
* **Triton 3.2's interpreter cannot run any of the repo's kernels** (it passes scalar kernel arguments as
  one-element arrays: `for r in range(0, REP)` fails in the shipped gate kernel). The kernels pass under Triton
  3.9's interpreter; on 3.2 only the compiled GPU check (the preflight) applies.
* **Budget geometry at 10%** (real tokenizer, all 2,150 examples): 14 examples have no INT2 tier (11 MultiNews,
  3 2WikiMQA: sink + local exceed the budget); KV memory is 9.99% nominal (the resolver's active allocation) and
  11.61% physical (Appendix B.4's Eq. 9 with this cache's slot table, slack, positions and ledger).
