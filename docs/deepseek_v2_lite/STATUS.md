# DeepSeek-V2-Lite port: status

Updated 2026-11-09. The plan is [`PLAN.md`](PLAN.md); the design prompt is
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

The vendored upstream files are hash-pinned (`test_vendored_upstream_is_unmodified`).

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

1. **P0 on a GPU, with the real checkpoint.** Load at 4.47.1, FullKV smoke, and activation fixtures. Measure D4, D5, D7, D8 and D9 on real latents and real attention.
2. **The kernels on a GPU.** Compile and run the MLA decode and gate kernels against the CPU references, and time the tile ladder. The prefill path (FA2 + score kernel on per-head 192, zero-padded to 256) has never executed.
3. **Split-K for the MLA decode kernel.** With one KV head the grid is `B` programs, so a GPU at small B is mostly idle. The fill and score epilogue must move into a combine launch.
4. **Sync-free MoE** for every perf arm, then the perf grid (16k / 32k prompts) with the `--gate-ratio 1.0` control.
5. **Quality**: LongBench / RULER 32k / GSM8K with `read_gate == gated` in every sidecar, plus the observation suite (per-head read ledger at rep = 16).
6. The per-window interleaved layout (§6.2) and a third-party latent FullKV decode baseline.

Quality and latency numbers from this port do **not** compare with Llama rows. The cache, the KV share of traffic and the baseline all differ (PLAN.md §3).
