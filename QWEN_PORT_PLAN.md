# Qwen port plan — bring `int2_qwen`'s logic and structure onto `int2_clustered_rep`

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
- **Do we want a Qwen parity/perf column?** If yes, fold the parity/perf runners
  into `utils/model_loading` too (out of scope above; `int2_qwen` left them).
- **Does this branch already accumulate window scores in fp32?** If its numerics
  work pre-empted Stage 6's accumulator change, that stage is verify-only and
  Llama/Mistral stay identical.
