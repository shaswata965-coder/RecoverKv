# Porting RecoverKV to DeepSeek-V2-Lite: plan, re-derived economics, timeline

Drafted 2026-11-08. **Nothing in this file has been measured on DeepSeek-V2-Lite.**
Every number is labelled *config* (read from the model's `config.json`), *code*
(read from its `modeling_deepseek.py`), or *arithmetic* (derived here with the
repo's own pricing formulas). The paste-ready design prompt is
[`PROMPT.md`](PROMPT.md).

Sources, pinned: `deepseek-ai/DeepSeek-V2-Lite-Chat@85864749cd611b4353ce1decdb286193298f64c7`
and `deepseek-ai/DeepSeek-V2-Lite@604d5664dddd88a0433dbae533b7fe9472482de0`.
`modeling_deepseek.py` sha256 `7d8e5221…b45b9c`, Apache-2.0 header.

---

## 0. The short version

DeepSeek-V2-Lite does not have KV heads in the sense this repo assumes. It has
**Multi-head Latent Attention (MLA)**: one 512-wide latent `c` plus one 64-wide
RoPE key `k_pe` per token per layer, shared by all 16 query heads. In the
absorbed form used for decode, the key is `[c ‖ k_pe]` (576 wide) and the value
is `c` itself (512 wide). Three things follow, and together they set the shape
of the port:

1. **It is a 16-way MQA, not a 4-way GQA.** `H_kv = 1`, `rep = 16`. Each
   component that works per KV head (the gate's union, the decode grid, the
   card, the anchor) now has one KV head per layer. The §12 lesson, that the union
   must be over each query head's *share*, matters more here than on Llama:
   16 heads now share one quarter of the tier.
2. **K and V share storage.** `c` is both the key's first 512 dims and the
   value. The int2 tier stores it **once**. The card's value centroid `vbar`
   is `mu[:512]`, so it costs nothing extra. The quantizer's key/value
   asymmetry (per-channel K, per-token V) becomes a single choice that has to be
   measured (§4 D4).
3. **The latency case moves to long context.** The latent cache is 1,152 B per
   token per layer against Llama-3.1-8B's 4,096 B (8 KV heads × 128 × K+V × 2 B): 31 KB
   per token across the model against 128 KB. At
   the repo's headline cell (4096/B=32) the whole FullKV cache is about 12% of a
   decode step's bytes, compared with about 52% on Llama. Even an infinitely
   compressed cache cannot win much latency there. The cell the method is for
   moves to 16k–32k prompts (§3.3). Lead with the memory claim.

The method itself (windows, cumulative scores, fp/int2/drop bands,
bidirectional promotion, frozen positions, the rank-1 card gate, mean-log fill,
layer-major compiled eviction, kernel-or-error, proof counters) carries over
unchanged. **The data layout and the model seam change; the method does not.**

---

## 1. The target model, as the cache sees it

| fact | value | source |
|---|---|---|
| layers | 27 (layer 0 dense MLP, 1–26 MoE) | config `num_hidden_layers`, `first_k_dense_replace=1` |
| query heads | 16 | config `num_attention_heads` |
| query projection | direct `q_proj` 2048 → 16×192 (`q_lora_rank: null`) | config, code l.955 |
| per-head q/k dim | 192 = 128 nope + 64 rope | config `qk_nope_head_dim`, `qk_rope_head_dim` |
| per-head v dim | 128 | config `v_head_dim` |
| latent | `kv_a_proj_with_mqa` 2048 → 512 + 64; `kv_a_layernorm` (RMSNorm) on the 512 | code l.966–975 |
| decompression | `kv_b_proj` 512 → 16 × (128 + 128) | code |
| `num_key_value_heads` | **16 — meaningful only for the decompressed path** | config |
| softmax scale | `192^-0.5 · mscale²`, `mscale = 0.1·0.707·ln 40 + 1 = 1.2608` → **0.11472** | code l.745–751; arithmetic |
| RoPE | YaRN, factor 40, original 4096, `mscale = mscale_all_dim = 0.707` → cos/sin multiplier ratio 1.0 | config, code l.316–326 |
| RoPE layout | **interleaved**: `view(d/2, 2).transpose` before `rotate_half`; tables `cat(freqs, freqs)` | code l.339–371 |
| dtype | bfloat16 | config |
| MoE | 64 routed (6/token) + 2 shared, expert width 1408 | config |
| context | 32k | model card |

What the HF remote code does that matters here:

| behaviour | consequence for the port | where |
|---|---|---|
| imports `flash_attn_func` **into its own module namespace** | the repo's `flash_lse` / `flash_decode` monkeypatch of `transformers.modeling_flash_attention_utils` is never reached (`hooks.py`'s own L-reuse error message names this exact case) | l.62 |
| flash path caches **decompressed** per-head K (16×192) and V **padded to 192** | its `DynamicCache` holds 12,288 B/token/layer, 10.7× the latent | l.995–1004 |
| `kv_seq_len += past_key_value.get_usable_length(...)`; `cos_cached[:seq_len]`; then `cos[position_ids]` | after the first eviction the held length is below the absolute position, so the table is sliced shorter than the index → device-side assert, which poisons the CUDA context (README §3) | l.845–854, l.152 |
| `cache_kwargs = {"sin", "cos"}` only, no `cache_position` | `WindowedCache._check_position_contract` gets nothing to check | l.1001 |
| `prepare_inputs_for_generation` reads `seen_tokens`, `get_max_length`; position_ids from `attention_mask.cumsum` | positions are monotonic under `generate()` (good); `WindowedCache` has no `seen_tokens` | l.1715–1762 |
| `moe_infer`: `tokens_per_expert.cpu().numpy()` then a Python loop over experts | one device sync per MoE layer per step (26/step) and up to 64 small GEMMs per layer. Before the cache does anything, the decode step already breaks "no device sync on the decode path" | l.630 |
| `0 in attention_mask` in the model forward | one more host sync per step | l.1509 |

transformers 4.47.1 (the repo's pin) predates native DeepSeek support, so the
model loads through `trust_remote_code` at a pinned revision.

---

## 2. Component map: Llama-3.1-8B today → DeepSeek-V2-Lite

| component | today (Llama-3.1-8B) | port (DeepSeek-V2-Lite) | method change? |
|---|---|---|---|
| stored per token | K, V: `[H_kv=8, D=128]` each, fp16 | `c` (512, post-`kv_a_layernorm`) + `k_pe` (64, post-RoPE), bf16 — **one latent "KV head"** | no |
| decode attention | GQA, `rep=4` | **absorbed** MLA: `q̃_h = [W_UK,hᵀ q_nope,h ‖ q_pe,h]` (576), keys `[c ‖ k_pe]`, values `c`; `o_h = W_UV,h · Σ p·c`; `rep=16` | no (mathematically identical logits) |
| prefill attention + L | FA2 via HF; L captured by patching `flash_attn_func` | FA2 over decompressed per-head K (192) / V (128 → padded 192), called **by our own forward** with `return_attn_probs=True` | no |
| prefill scoring | Triton FA2-backward-style kernel, `D=128`, GQA | same kernel at `D=192` (not a power of two: 128 + 64 tiles), `rep=1`, same L | no |
| window scores | `[B, H_q=32, W]`, fp32 running sum, rank by head-mean | `[B, H_q=16, W]`, same | no |
| bands / promotion / reactivation / frozen positions | as shipped | same | no |
| int2 codes | K per-channel **pre-RoPE**, V per-token, separate | `c` **once** (granularity: D4), `k_pe` per-channel pre-RoPE | layout |
| grid | 1 B/entry, fp16 scale per 32 (`GRID_GROUP`), codes fit to stored grid | same rules (512 and 64 are multiples of 32) | no |
| RoPE | full 128-d head, HF half-split, in registers at frozen positions | 64-d `k_pe` only; DeepSeek interleave fixed by a one-time permutation (D5); YaRN tables indexed by absolute position | convention |
| card | rank-1 over each KV head's 128-d post-RoPE key; `vbar` per head; anchor per (layer, head) | rank-1 over the **576-d** latent key; **`vbar ≡ mu[:512]`** (no `vm` field); anchor per layer | layout |
| gate | card scan + per-KV-head share-union, `n_sel = ceil(0.25·n_active)` | same rule, one KV head, `rep=16` | no |
| fill | mean-log deviation, centroid credit, renormalise | same | no |
| decode kernel | grid `B·H_kv`, `BLOCK_R=16` padding 4 heads | grid `B·splits`. **split-K becomes mandatory** (one KV head ⇒ B programs). `BLOCK_R=16` is exactly full. acc `[16, 512]` fp32 | kernel |
| eviction | layer-major `R=32·B`, compiled-or-raise | `R=27·B`, same contracts | no |
| budget | `bytes_per_token = 8·128·2·2 = 4096` | **`(512+64)·2 = 1152`**, never from `num_key_value_heads` | pricing |
| model seam | monkeypatched `flash_attn_func` + forward hooks | **owned MLA attention forward** installed at load; same `_run_fused` contract, same `_STATS`, same verdicts | seam |
| FullKV baseline | `DynamicCache` + FA2 | latent cache + absorbed decode from a third-party kernel that runs on the target GPU (D13); remote-code `DynamicCache` as a context row only | baseline |

---

## 3. Economics, re-derived (*arithmetic*, repo formulas, `ws=8`)

### 3.1 What a window and its card cost (per layer; there is one latent head)

`codes = (512 + 64) · 8 · 2 bits / 8 = 1152 B`. The card is `mu` int4 (576) +
`v` int8 (576) + `t` int8 (8), each with an fp16 scale: **878 B** (1,136 B if
`vbar` were stored separately; aliasing it saves 258 B).

| latent grid (D4) | grid B | window B | card/window | break-even ratio | gate cost at 0.25 | fp/q price ratio |
|---|---|---|---|---|---|---|
| Llama today, per head | 292 | 804 | 0.34 | 0.66 | 0.59 | 3.81 |
| `c` per-channel (KIVI key rule) | 1,224 | 2,376 | **0.37** | **0.63** | 0.62 | 2.83 |
| `c` per-token | 156 | 1,308 | **0.67** | **0.33** | 0.92 | 4.22 |

So D4 has two effects. Per-token is 1.8× cheaper per window, which buys more
context under `bytes`, but it makes the card two-thirds of the window and
moves break-even to 0.33. **`v` at int8 is 576 B, 66% of the card.** The §6.1
finding that `v` must be int8 was measured on Llama's 128-d keys. On a 576-d
latent it has to be re-measured with `quant_card_bits`: `v` at int4 gives a
590 B card.

### 3.2 Tier sizes at the operating point (budget 0.20, q 0.70 bytes, sink 5, local 128)

| prompt/gen | latent grid | `top_k_fp` | `N_q` | keys held | `n_sel` at 0.25 |
|---|---|---|---|---|---|
| 4096/256 | per-channel | 27 | 182 | 1,805 (41%) | 46 |
| 4096/256 | per-token | 27 | 271 | 2,517 (58%) | 68 |
| 32768/256 | per-channel | 242 | 1,603 | 14,893 (45%) | 401 |
| 32768/256 | per-token | 242 | 2,387 | 21,165 (64%) | 597 |

### 3.3 Where a decode step's bytes go: the claim map

Weights read per step assume uniform routing (distinct experts touched per MoE
layer = `64·(1 − (58/64)^B)`). Bandwidth is the repo's effective 1.56 TB/s.

| B | prompt | weights | FullKV latent | KV share | traffic floor |
|---|---|---|---|---|---|
| 1 | 4k / 32k | 4.8 GB | 0.13 / 1.02 GB | 3% / 18% | 3.2 / 3.7 ms |
| 8 | 4k / 16k / 32k | 18.3 GB | 1.0 / 4.1 / 8.2 GB | 5% / 18% / 31% | 12.4 / 14.3 / 17.0 ms |
| 32 | 4k / 16k / 32k | 30.2 GB | 4.1 / 16.3 / 32.6 GB | **12%** / 35% / **52%** | 22.0 / 29.8 / 40.2 ms |

On Llama-3.1-8B, 4096/B=32 is 52% KV. On DeepSeek-V2-Lite the cell with that
share is **32k/B=32**, which needs about 64 GB for weights plus cache. That still fits an
A100-80GB but leaves little headroom. Consequences:

* **Perf grid:** drop 1024/1024 and add 16k and 32k prompts at B ∈ {8, 16, 32}.
  4096/B=32 stays only as continuity with the Llama table.
* **B=1 is less winnable than on Llama.** The floor is ~3.2 ms. Llama's measured
  per-step machinery (300–430 µs per layer) × 27 layers is 8–12 ms. Claim memory
  at B=1, as §8 of the status document already does.
* **FLOPs, not only bytes.** With `rep=16` the fp tier runs at ~30 FLOP/B, which is
  memory-bound. The int2 tier, at ~297 B/token plus dequant, is ~117 FLOP/B,
  near the A100 ridge (~156). A skipped window also saves the 16-head QK and PV
  MMAs. The gate's MLA break-even may therefore be set by issue rate, not by
  `1 − card/window`. That is a hypothesis for the first GPU profile, not a claim.

### 3.4 Baseline memory, so nobody quotes a straw man

| cache | B/token/layer | vs latent |
|---|---|---|
| latent (`c` + `k_pe`, bf16): **the FullKV reference** | 1,152 | 1.0× |
| `resolve()` as-is from `config.json` (16 × 128 × K+V × 2) | 8,192 | 7.1× (wrong) |
| remote-code `DynamicCache` (16 × 192 K + 16 × 192 padded V) | 12,288 | 10.7× |

---

## 4. Decisions: default, and what may overturn it

| id | decision | default | what overturns it |
|---|---|---|---|
| D1 | what is cached | the latent `c` + `k_pe` | nothing. A decompressed cache is a 10.7× straw man |
| D2 | attention form | absorbed at decode, non-absorbed at prefill (and prefill scoring) | nothing; the logits are identical, and this split is the cheap way to compute each |
| D3 | model seam | our own MLA attention forward, installed at load | nothing. The remote module imports `flash_attn_func` directly, so there is no seam to patch |
| D4 | latent int2 granularity | **per-channel** (KIVI's key rule: `c` is a key first, and its errors are exponentiated) | P0 measurement on real activations: logit error, output error and needle retrieval, per-channel vs per-token (with 32-channel scale groups). A switch to per-token is a quality change |
| D5 | `k_pe` storage and RoPE convention | int2, per-channel, **pre-RoPE**; stored in **de-interleaved (half-split) order** with `q_pe` permuted identically, so every kernel keeps the existing half-split in-register RoPE | P0: if int2 `k_pe` measurably breaks positional retrieval, an int8 `k_pe` ablation (DeepSeek's own FP8 MLA cache in FlashMLA keeps the 64 rope dims in bf16; that is precedent, not evidence at int2) |
| D6 | store layout | **one contiguous payload block per window** (codes + grids) plus a **contiguous card array**. §6.2 lands by construction because nothing on MLA has an old layout to preserve | if it delays bring-up, mirror the Llama SoA layout first. Score-neutral either way; the CPU reference must agree bitwise |
| D7 | card widths | `mu` 4, `v` 8, `t` 8; `vm` aliased to `mu[:512]` and priced zero | P0 re-measures `v` at 4 bits on the 576-d latent key |
| D8 | gate rule and ratio | share-union, unchanged; ratio 0.25 | never change the rule. P0 measures per-head recall at `rep=16`. Re-pinning the ratio is a config change, and a quality change with runs behind it |
| D9 | dtype | bf16 store; RoPE tables at the store dtype | P0 measures the demote→promote round-trip error on bf16 `k_pe`. If it is large, use fp32 RoPE in **both** tiers, so a key never depends on its tier |
| D10 | operating point | `utils.config.OPERATING_POINT` unchanged | — |
| D11 | perf grid | 4k (continuity), 16k, 32k × B ∈ {1, 8, 16, 32} | memory fit on the target GPU |
| D12 | MoE in perf runs | one **sync-free** fused MoE path, identical for every arm, recorded as `moe_path` in sidecars | — |
| D13 | FullKV baseline | latent cache + absorbed decode from a maintained kernel that runs on the box (FlashMLA on sm90; on A100 a Triton or FlashInfer MLA decode). Remote-code `DynamicCache` is a context row only | — |

---

## 5. Traps found while reading, each a test before it is a fix

1. **Budget priced from `num_key_value_heads`.** `resolve()` would price 8,192 B/token
   from the config (7.1× the latent). Pin `bytes_per_token == 1152` in a test.
2. **Softmax scale.** `cache.py:1110` sets `_attn_scaling = head_dim ** -0.5`.
   The correct value is 0.11472 (`module.softmax_scale`). `192^-0.5` is 1.59× too
   small and `576^-0.5` is 2.75× too small. Thread `softmax_scale` from the module
   into every kernel and reference.
3. **The seam is not where the repo looks.** `flash_attn_func` is imported into
   the remote module's namespace. Any monkeypatch of transformers' copy is a
   silent no-op. That is the failure mode `CLAUDE.md` describes, a gate built and tested but
   never called, arriving by a new route.
4. **RoPE table sliced by held length.** It crashes after the first eviction
   (device assert). The owned forward indexes YaRN tables by absolute position.
5. **Wrong RoPE convention.** `effective.py` and `hooks.py` import Llama's
   `apply_rotary_pos_emb`, which is half-split. DeepSeek's is interleaved. Un-rotating with the
   wrong one corrupts silently. Rotate/unrotate must be bit-identical to the
   model's own function, including the permutation.
6. **No `cache_position`** in the remote call. Pass absolute positions into
   `cache_kwargs` so `_check_position_contract` has something to check.
7. **`seen_tokens` / `get_usable_length` / `get_max_length`.** Audit every call
   site in the pinned modeling file and define each on `WindowedCache`
   deliberately (seen = monotonic, held = shrinking). Add a test that runs `generate()`
   through ≥3 evictions.
8. **MoE host syncs.** These are not the cache's, but they make "the step is
   kernel-bound" false before the cache is involved, and would make every perf
   comparison measure the MoE loop.
9. **One KV head ⇒ B programs.** Without split-K, B=1 decode is one program
   walking the whole cache.
10. **`_get_attn_classes` knows Llama and Qwen2 only.** On CPU it warns and
    installs nothing. The adapter must recognise the remote class by name plus
    `model_type == "deepseek_v2"`, and raise otherwise.

---

## 6. Timeline

Assumptions: one engineer working with Claude Code sessions; a CPU dev box
daily (as now); one A100-80GB for short sessions from week 2; 8×A100 for about
three days in weeks 9–11. Weeks are calendar weeks of focused work.

| phase | weeks | where | delivers | exit gate |
|---|---|---|---|---|
| **P−1 (parallel)**: verify the method on Llama first | 1–2 | 8×A100, ~1 day | §12 "Next GPU run" items 1–3: RULER 32k at gate 0.25 and 1.0, LongBench at the operating point | the union fix and fill fix are measured. **Otherwise the port carries forward three CPU-only fixes as its reference behaviour** |
| **P0**: ground truth and claim map | 1–2 | 1 GPU-day + CPU | pinned load at 4.47.1; FullKV smoke (2 LongBench sets, RULER 4k/32k NIAH); activation fixtures (`c`, `k_pe` pre/post-RoPE, absorbed `q̃`, true attention) for 4 layers at 4k and 32k; D4/D5/D7/D8/D9 measured on CPU from the fixtures; `docs/deepseek_v2_lite/STATUS.md` with §3.3 confirmed or corrected | every decision in §4 carries a number or an explicit "default, unmeasured"; go/no-go on the latency claim |
| **P1**: geometry, adapter, CPU oracle | 2–4 | CPU | `models/deepseek_v2/` (vendored pinned modeling file, `MLAGeometry`, owned attention forward); MLA store, quantizer, card and budget; materialize (eager) path | owned forward == remote forward on a tiny random config (fp32 exact, bf16 ≤ tolerance); RoPE bit-identity tests; budget pins; eager e2e through ≥3 evictions with monotonic positions |
| **P2**: kernels, interpreter-first | 4–6 | CPU (`TRITON_INTERPRET=1`) | prefill score kernel at D=192; gate kernels at H_kv=1, rep=16, D=576; MLA two-tier decode (split-K, shared `c` tile for QK and PV, 64-d RoPE in registers, gated loop, fill + score epilogue); a CPU reference per kernel that mirrors its tiling | interpreter tests agree with references; `test_kernel_compiles` at sm80 |
| **P3**: compiled eviction and end-to-end | 5–7 | CPU, then 1–2 GPU days | layer-major compiled eviction over the latent (promote, demote, reactivate, card build, anchors); ablation knobs; proof counters; sidecar fields | flash vs eager per-step parity; `quant_gate_ratio=1.0` is a provable no-op; `gated == fired`; `grep fused_gate` finds a non-test caller; no `EvictionRanEager` |
| **P4**: GPU bring-up and tuning | 7–8 | 1 GPU, ~3 days | kernels against references on GPU; timed tile ladders (never first-fit, no cross-geometry reuse); `profile_decode` buckets for MLA | GPU matches references; a profile with kernel-vs-roofline per kernel |
| **P5**: baselines and perf harness | 8–9 | 1 GPU | MLA FullKV baseline (D13); sync-free MoE (D12); perf sweep over the D11 grid with the gate-1.0 control arm | perf table with a `gated` verdict on every row |
| **P6**: quality | 9–11 | 8×A100, ~3 days | `SMOKE=1` first, then LongBench (16 sets, budgets 0.05/0.10/0.20), RULER 32k (13 tasks), GSM8K, E1/E2/E3, observations I–V (per-head read ledger at rep=16), tier study | every sidecar `read_gate == gated` (or `not-expected` for eager / FullKV) |
| **P7**: write-up | 12 | — | status document in the `DISTANCE_TO_GOAL.md` style; measured / arithmetic / CPU-only table | — |

**About 12 weeks end to end**, with the critical path through P2→P4 (kernels).
With a second person on P5 (baseline kernel, MoE, perf harness) from week 4, it
is about 8–9 weeks. P0 is the cheapest place to find out the port should claim
memory only, so it goes first and blocks nothing else.

### Risk register

| risk | likelihood | what it costs | caught in |
|---|---|---|---|
| share-union recall collapses at `rep=16` (16 heads, one quarter of the tier) | medium | the operating ratio moves up; break-even (§3.1) leaves little room | P0 (fixtures), P6 (observation IV) |
| per-channel latent grid is required, so windows stay expensive | medium | less context per byte than §3.2's per-token rows | P0 |
| decode kernel register pressure (`acc [16, 512]` fp32) or occupancy | high | the first rung that fits is slow; split-K count becomes a tuning axis | P2 (compile), P4 (timed) |
| MoE dominates the step, so cache effects are invisible | high (remote code) | perf table measures the MoE loop | P5 (D12) |
| bf16 RoPE round trip makes promoted keys drift | low–medium | fp32 RoPE in both tiers | P0 |
| remote code vs transformers 4.47.1 API drift | medium | load failures, deprecated cache API | P0 day 1 |
| no latency win below 16k (arithmetic says so) | high | the headline is memory, not TPOT | P0 (accept it early) |
