"""What makes the Qwen2.5 column comparable with the Llama and Mistral columns.

Four independent hazards, each of which produced (or would produce) a plausible
score table built on a different protocol from the other model columns:

1. **RoPE with a non-unit ``attention_scaling``.** YaRN folds
   ``a = 0.1·ln(factor)+1`` into cos/sin, so the forward map is ``a·R(θ)`` and the
   negated-sin "inverse" the two-tier read path uses lands on ``a²·k``. At
   ``factor=4`` that is 1.296x on every int2 key. ``rope_type`` default/llama3
   have ``a = 1``, which is exactly why Llama-3.1 and Mistral never exercised it.
2. **Sampling defaults the checkpoint ships.** Qwen2.5-7B-Instruct's
   ``generation_config.json`` sets ``repetition_penalty: 1.05``;
   ``RepetitionPenaltyLogitsProcessor`` is a logits PROCESSOR, so it applies under
   greedy decoding. Llama-3.1 and Mistral-v0.2 ship none.
3. **Context/RoPE overrides** having to be hand-edited into a cached config.json,
   where they are invisible to every result file.
4. **``model.dtype`` falling back to fp16 on an unknown name**, which made a typo
   indistinguishable from an intended fp16 run on a bf16-native model.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from modules.quant.effective import (           # noqa: E402
    rope_attention_scaling,
    rotate_key_window,
    unrotate_key_window,
)
from utils.config import ConfigValidationError, ModelConfig  # noqa: E402
from utils.model_loading import (               # noqa: E402
    GREEDY_GENERATION_DEFAULTS,
    apply_model_config_overrides,
    pin_greedy_generation_defaults,
    resolve_dtype,
)

H, T, D = 2, 8, 16


class _ScaledRope(torch.nn.Module):
    """A rotary module with the transformers contract, at a chosen scaling.

    Deliberately not a real ``Qwen2RotaryEmbedding``: this pins *our* inversion
    math against the contract (``forward`` returns cos/sin already multiplied by
    ``attention_scaling``) independently of the installed transformers. The
    integration test below then checks that the real module honours that
    contract.
    """

    def __init__(self, attention_scaling: float = 1.0, base: float = 10000.0,
                 applied_scaling: float | None = None):
        super().__init__()
        #: what the module REPORTS
        self.attention_scaling = attention_scaling
        #: what it actually folds into cos/sin — normally the same. They differ
        #: only in the test that checks a hidden scaling is reported, not absorbed.
        self._applied = (
            attention_scaling if applied_scaling is None else applied_scaling
        )
        inv = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x, position_ids):
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq  # [B, T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)                     # [B, T, D]
        cos = emb.cos() * self._applied
        sin = emb.sin() * self._applied
        return cos.to(x.dtype), sin.to(x.dtype)


def _key_and_pos(seed=0):
    torch.manual_seed(seed)
    k = torch.randn(1, H, T, D, dtype=torch.float32)
    pos = torch.arange(T, dtype=torch.long).unsqueeze(0)
    return k, pos


YARN_FACTOR_4 = 0.1 * math.log(4.0) + 1.0        # 1.13862... — transformers' formula


class TestRopeRoundTripUnderScaling:
    """rotate ∘ unrotate must be the identity at ANY attention_scaling."""

    @pytest.mark.parametrize(
        "a", [1.0, YARN_FACTOR_4, 0.1 * math.log(8.0) + 1.0],
        ids=["unscaled", "yarn-factor-4", "yarn-factor-8"],
    )
    def test_unrotate_then_rotate_recovers_the_key(self, a):
        rope = _ScaledRope(attention_scaling=a)
        k_post, pos = _key_and_pos()
        k_pre = unrotate_key_window(k_post, pos, rope)
        k_back = rotate_key_window(k_pre, pos, rope)
        torch.testing.assert_close(k_back, k_post, rtol=1e-5, atol=1e-5)

    def test_without_the_correction_the_error_is_a_squared(self):
        """Pins the size of the bug, so the fix is not guarding a hypothetical.

        The uncorrected inverse returns a²·k_pre; re-rotating that gives
        a²·k_post. At YaRN factor 4 every int2 key would be 29.6% too large.
        """
        a = YARN_FACTOR_4
        rope = _ScaledRope(attention_scaling=a)
        k_post, pos = _key_and_pos()

        corrected = unrotate_key_window(k_post, pos, rope)
        uncorrected = corrected * (a * a)         # what the old code returned

        ratio = (rotate_key_window(uncorrected, pos, rope) / k_post).mean().item()
        assert ratio == pytest.approx(a * a, rel=1e-3)
        assert ratio == pytest.approx(1.2965, rel=1e-3)

    def test_unscaled_path_is_bit_identical_to_no_correction(self):
        """Llama-3.1 (llama3 rope) and Mistral must not move by a single bit.

        ``a == 1.0`` takes a skipped branch, not a division by 1.0 — so this is
        an identity of the code path, not of the arithmetic.
        """
        rope = _ScaledRope(attention_scaling=1.0)
        k_post, pos = _key_and_pos(seed=3)

        from modules.quant.effective import _apply_rotary, _rope_cos_sin

        cos, sin = _rope_cos_sin(rope, k_post, pos)
        _, legacy = _apply_rotary()(k_post, k_post, cos, -sin)
        assert torch.equal(unrotate_key_window(k_post, pos, rope), legacy)

    def test_batched_and_unbatched_agree(self):
        rope = _ScaledRope(attention_scaling=YARN_FACTOR_4)
        k_post, pos = _key_and_pos(seed=7)
        batched = unrotate_key_window(k_post, pos, rope)
        unbatched = unrotate_key_window(k_post[0], pos[0], rope)
        torch.testing.assert_close(batched[0], unbatched, rtol=0, atol=0)


class TestRopeAttentionScaling:
    def test_missing_attribute_means_unscaled(self):
        assert rope_attention_scaling(torch.nn.Module()) == 1.0

    def test_reads_the_attribute(self):
        assert rope_attention_scaling(_ScaledRope(1.5)) == 1.5

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "x", None])
    def test_unusable_values_fall_back_to_one(self, bad):
        """A scaling we cannot invert must not silently rescale the tier."""
        rope = _ScaledRope(1.0)
        rope.attention_scaling = bad
        assert rope_attention_scaling(rope) == 1.0

    def test_a_hidden_scaling_is_reported_not_absorbed(self):
        """cos²+sin² disagreeing with the declared a² must warn, once."""
        import modules.quant.effective as eff

        # Folds 1.6 into cos/sin while reporting 1.0 — the shape of a rope
        # variant that scales without exposing `attention_scaling`.
        rope = _ScaledRope(attention_scaling=1.0, applied_scaling=1.6)
        eff._warned_rope_scaling[0] = False
        try:
            with pytest.warns(RuntimeWarning, match="attention_scaling"):
                unrotate_key_window(*_key_and_pos(), rope)
        finally:
            eff._warned_rope_scaling[0] = False


@pytest.mark.parametrize("rope_type,factor,expected", [
    ("yarn", 4.0, YARN_FACTOR_4),
    ("default", None, 1.0),
])
def test_real_qwen2_rotary_honours_the_contract(rope_type, factor, expected):
    """The installed transformers must actually fold ``attention_scaling`` in.

    Our inversion is analytic, so if a transformers release stopped exposing the
    scalar (or stopped applying it) the Q tier would be wrong with nothing
    raising. Not version-pinned — it asserts the contract, not a version.
    """
    pytest.importorskip("transformers")
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    kw = dict(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
              max_position_embeddings=4096, rope_theta=10000.0)
    if rope_type == "yarn":
        kw["rope_scaling"] = {"rope_type": "yarn", "factor": factor}
    cfg = Qwen2Config(**kw)
    rope = Qwen2RotaryEmbedding(config=cfg)

    assert rope_attention_scaling(rope) == pytest.approx(expected, rel=1e-6)

    # …and the returned cos/sin really carry it: cos² + sin² == a².
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0)
    cos, sin = rope(torch.empty(1, 1, 1, dtype=torch.float32), pos)
    measured = float((cos[..., :1] ** 2 + sin[..., :1] ** 2).flatten()[0])
    assert measured == pytest.approx(expected ** 2, rel=1e-4)


class _GenCfg:
    def __init__(self, **kw):
        for k, v in GREEDY_GENERATION_DEFAULTS.items():
            setattr(self, k, v)
        for k, v in kw.items():
            setattr(self, k, v)


class _Model:
    def __init__(self, gen_cfg):
        self.generation_config = gen_cfg


class TestGreedyPinning:
    def test_qwen_shaped_config_is_neutralised(self):
        """The real Qwen2.5-7B-Instruct generation_config.json values."""
        gc = _GenCfg(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                     repetition_penalty=1.05)
        changed = pin_greedy_generation_defaults(_Model(gc))

        assert changed["repetition_penalty"] == (1.05, 1.0)
        assert gc.repetition_penalty == 1.0
        assert gc.do_sample is False
        assert gc.temperature == 1.0
        assert set(changed) == {"do_sample", "temperature", "top_p", "top_k",
                                "repetition_penalty"}

    def test_llama_and_mistral_shaped_configs_are_untouched(self):
        """Empty dict, so those two runs stay byte-identical to before this existed.

        Llama-3.1-8B-Instruct ships do_sample/temperature/top_p but no penalty;
        Mistral-v0.2 ships only token ids. Neither reaches a logits processor
        under an explicit do_sample=False, and neither is what broke Qwen.
        """
        assert pin_greedy_generation_defaults(_Model(_GenCfg())) == {}

    def test_stopping_contract_is_never_touched(self):
        """eos/pad/bos are the model's real stopping contract, not a preference."""
        gc = _GenCfg(eos_token_id=[151645, 151643], pad_token_id=151643,
                     bos_token_id=151643, repetition_penalty=1.05)
        pin_greedy_generation_defaults(_Model(gc))
        assert gc.eos_token_id == [151645, 151643]
        assert gc.pad_token_id == 151643
        assert "eos_token_id" not in GREEDY_GENERATION_DEFAULTS

    def test_missing_generation_config_is_tolerated(self):
        assert pin_greedy_generation_defaults(object()) == {}


class _HFCfg:
    """Minimal stand-in for a Qwen2Config after __init__ has resolved the pair."""

    def __init__(self, **kw):
        self.max_position_embeddings = 32768
        self.rope_scaling = None
        self.sliding_window = None
        self.use_sliding_window = False
        for k, v in kw.items():
            setattr(self, k, v)


class TestConfigOverrides:
    def test_no_declared_overrides_changes_nothing(self):
        hf = _HFCfg()
        assert apply_model_config_overrides(hf, ModelConfig()) == []
        assert hf.max_position_embeddings == 32768
        assert hf.rope_scaling is None

    def test_yarn_block_is_applied(self):
        hf = _HFCfg()
        m = ModelConfig(
            max_position_embeddings=131072,
            rope_scaling={"rope_type": "yarn", "factor": 4.0,
                          "original_max_position_embeddings": 32768},
        )
        changes = apply_model_config_overrides(hf, m)
        assert hf.max_position_embeddings == 131072
        assert hf.rope_scaling["rope_type"] == "yarn"
        assert hf.rope_scaling["factor"] == 4.0
        assert any("max_position_embeddings" in c for c in changes)

    def test_legacy_type_key_is_mirrored_to_rope_type(self):
        """Otherwise transformers validates and computes it as 'default' — i.e.
        the scaling silently does not happen."""
        hf = _HFCfg()
        apply_model_config_overrides(
            hf, ModelConfig(rope_scaling={"type": "yarn", "factor": 4.0})
        )
        assert hf.rope_scaling["rope_type"] == "yarn"

    def test_declared_sliding_window_stays_disabled(self):
        """Qwen2.5 ships sliding_window=131072 AND use_sliding_window=false.

        Qwen2Config resolves that pair to None at construction; setting the
        attributes afterwards would switch banded attention back ON, which the
        windowed cache cannot be correct under (the band is measured in cache
        SLOTS, not token distance).
        """
        hf = _HFCfg()
        apply_model_config_overrides(
            hf, ModelConfig(sliding_window=131072, use_sliding_window=False)
        )
        assert hf.sliding_window is None
        assert hf.use_sliding_window is False

    def test_an_explicitly_enabled_window_is_left_alone(self):
        hf = _HFCfg()
        apply_model_config_overrides(
            hf, ModelConfig(sliding_window=4096, use_sliding_window=True)
        )
        assert hf.sliding_window == 4096


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestSamsumEosSet:
    """samsum stops at a newline — ADDED to the model's EOS set, not replacing it.

    ``[tokenizer.eos_token_id, newline]`` drops every other id the checkpoint
    stops on. On Qwen2.5 that discards ``<|endoftext|>`` (151643) and keeps only
    ``<|im_end|>`` (151645) — and samsum is prompted RAW (it is in
    NO_CHAT_TEMPLATE_DATASETS), so ``<|endoftext|>`` is the one a raw
    continuation actually emits.
    """

    @staticmethod
    def _call(runner_state, newline_id=198):
        from modules.evaluation.longbench_runner import LongBenchRunner

        return LongBenchRunner._eos_ids_plus(runner_state, newline_id)

    def test_qwen_keeps_both_stop_ids(self):
        state = _Stub(
            model=_Stub(generation_config=_Stub(eos_token_id=[151645, 151643])),
            tokenizer=_Stub(eos_token_id=151645),
        )
        assert self._call(state) == [151645, 151643, 198]

    def test_llama_keeps_all_three(self):
        """Llama-3.1-8B-Instruct stops on 128001/128008/128009."""
        state = _Stub(
            model=_Stub(generation_config=_Stub(
                eos_token_id=[128001, 128008, 128009])),
            tokenizer=_Stub(eos_token_id=128009),
        )
        assert self._call(state) == [128001, 128008, 128009, 198]

    def test_scalar_eos_still_works(self):
        """Mistral-v0.2 ships a single id."""
        state = _Stub(
            model=_Stub(generation_config=_Stub(eos_token_id=2)),
            tokenizer=_Stub(eos_token_id=2),
        )
        assert self._call(state) == [2, 198]

    def test_falls_back_to_the_tokenizer(self):
        state = _Stub(model=_Stub(generation_config=None),
                      tokenizer=_Stub(eos_token_id=7))
        assert self._call(state) == [7, 198]

    def test_duplicates_are_collapsed(self):
        state = _Stub(
            model=_Stub(generation_config=_Stub(eos_token_id=[198, 2])),
            tokenizer=_Stub(eos_token_id=2),
        )
        assert self._call(state) == [198, 2]


class TestTruncationPolicyIsAnnounced:
    """``longbench.max_length`` decides how much of each prompt is ever seen and
    DEFAULTS to 7500, and nothing in the prediction files records which mode was
    in force. A config that simply omits the key produced a plausible score table
    built on quarter-length prompts."""

    @staticmethod
    def _run(max_length, model_window, datasets=("gov_report",)):
        from modules.evaluation.longbench_runner import LongBenchRunner

        # A real LongBenchRunner with its __init__ skipped, rather than a stub:
        # the method reads a class constant and two collaborators, and a stub
        # that has to re-declare each of them drifts from the class it is
        # standing in for.
        runner = LongBenchRunner.__new__(LongBenchRunner)
        runner.lb = _Stub(max_length=max_length, datasets=list(datasets))
        runner.model = _Stub(config=_Stub(max_position_embeddings=model_window,
                                          num_attention_heads=28))
        # Flash backend throughout: the eager attention ceiling is a separate
        # concern with its own cases (TestEagerAttentionCeiling), and mixing it in
        # here would make these assertions about two things at once.
        runner.config = _Stub(model=_Stub(attn_implementation="flash_attention_2"))
        runner.cache_backend_package = "flash_attn"
        runner.dataset2maxlen = {"gov_report": 512, "triviaqa": 32}
        runner._over_context_warned = False
        runner._log_truncation_policy()

    def test_the_inherited_default_on_a_long_context_model_warns(self, caplog):
        """7500 against a YaRN-scaled 131072 window — the exact failure."""
        with caplog.at_level("WARNING"):
            self._run(7500, 131072)
        assert "discards" in caplog.text
        assert "DEFAULT" in caplog.text

    def test_the_mistral_style_budget_does_not_warn(self, caplog):
        """31500 against a 32768 window is the published protocol, not a bug."""
        with caplog.at_level("WARNING"):
            self._run(31500, 32768)
        assert "discards" not in caplog.text

    def test_auto_fit_does_not_warn(self, caplog):
        with caplog.at_level("WARNING"):
            self._run(None, 131072)
        assert "discards" not in caplog.text

    def test_the_reserve_uses_the_longest_configured_generation(self, caplog):
        """A QA-only run reserves 32, not 512, so the headroom test is fair."""
        with caplog.at_level("WARNING"):
            self._run(31500, 32768, datasets=("triviaqa",))
        assert "discards" not in caplog.text


class TestModelConfigValidation:
    def test_unknown_dtype_raises_instead_of_becoming_fp16(self):
        with pytest.raises(ConfigValidationError, match="model.dtype"):
            ModelConfig(dtype="bloat16")

    @pytest.mark.parametrize("name", ["float16", "bfloat16", "float32"])
    def test_known_dtypes_resolve(self, name):
        assert resolve_dtype(name) is getattr(torch, name)

    def test_rope_scaling_without_a_type_raises(self):
        with pytest.raises(ConfigValidationError, match="rope_type"):
            ModelConfig(rope_scaling={"factor": 4.0})

    def test_rope_scaling_without_a_factor_raises(self):
        with pytest.raises(ConfigValidationError, match="factor"):
            ModelConfig(rope_scaling={"rope_type": "yarn"})

    def test_int_factor_is_normalised_to_float(self):
        """transformers' _validate_yarn_parameters only WARNS on a non-float,
        and a warning in a 9-hour run is a warning nobody reads."""
        m = ModelConfig(rope_scaling={"rope_type": "yarn", "factor": 4})
        assert isinstance(m.rope_scaling["factor"], float)

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5])
    def test_bad_max_position_embeddings_raises(self, bad):
        with pytest.raises(ConfigValidationError):
            ModelConfig(max_position_embeddings=bad)


class TestInt2GridUnderBfloat16:
    """The int2 scale/zero grid vs a bf16 KV cache.

    Under fp16 KV the grid cannot overflow (inputs are fp16-bounded, so
    ``scale = (mx-mn)/3 <= 43669``). Under bf16 KV it can, because bf16 carries
    fp32's exponent range — and switching the Qwen columns to bf16 is what made
    it reachable. Unguarded it is silent: the window dequantizes to inf/nan.

    Fixed structurally rather than by warning — ``grid_dtype_for`` gives a bf16
    cache a bf16 grid, which has the same exponent range and the same 2 bytes.
    """

    @staticmethod
    def _q(x, grid_dtype=None):
        from modules.quant.quantizer import (
            _affine_quantize, grid_dtype_for, reset_grid_guard,
        )

        if grid_dtype is None:
            grid_dtype = grid_dtype_for(x.dtype)
        reset_grid_guard()
        try:
            return _affine_quantize(x, group_dim=-1, grid_dtype=grid_dtype)
        finally:
            reset_grid_guard()

    def test_grid_dtype_follows_the_kv_dtype(self):
        from modules.quant import grid_dtype_for

        assert grid_dtype_for(torch.float16) is torch.float16
        assert grid_dtype_for(torch.bfloat16) is torch.bfloat16
        # fp32 caches (the CPU test path) also get bf16: it covers fp32's range.
        assert grid_dtype_for(torch.float32) is torch.bfloat16

    @pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16, torch.float32])
    def test_every_grid_dtype_is_two_bytes(self, dt):
        """A wider grid would move every compression ratio the method reports."""
        from modules.quant import grid_dtype_for

        g = grid_dtype_for(dt)
        assert torch.tensor([], dtype=g).element_size() == 2

    def test_a_key_past_fp16_range_no_longer_corrupts_the_window(self):
        """The actual bug: bf16 holds 3e5, fp16's grid cannot, so it became nan."""
        x = torch.tensor([[-2.0e5, 0.0, 1.0, 3.0e5]], dtype=torch.bfloat16)

        codes, scale, zero = self._q(x)                    # bf16 grid (the fix)
        assert scale.dtype is torch.bfloat16
        assert torch.isfinite(scale).all() and torch.isfinite(zero).all()

        from modules.quant.quantizer import _affine_dequantize

        back = _affine_dequantize(codes, scale, zero, torch.float32)
        assert torch.isfinite(back).all()
        # …and it is a real reconstruction, not a degenerate collapse.
        assert back.max() > 1.0e5 and back.min() < -1.0e5

    def test_the_old_fp16_grid_would_have_produced_nan(self):
        """Pins what was actually wrong, so the fix cannot be quietly reverted."""
        x = torch.tensor([[-2.0e5, 0.0, 1.0, 3.0e5]], dtype=torch.bfloat16)
        with pytest.warns(RuntimeWarning, match="not representable"):
            codes, scale, zero = self._q(x, grid_dtype=torch.float16)
        # It now falls back to the degenerate grid rather than storing inf —
        # information-free for that group, but never nan, and it warns.
        assert torch.isfinite(scale).all() and torch.isfinite(zero).all()

    def test_fp16_cache_path_is_bit_identical_to_the_original(self):
        """Llama-3.1 / Mistral-v0.2 results must not move by a single bit.

        Recomputes the pre-change algorithm inline and requires exact equality.
        """
        torch.manual_seed(0)
        x = torch.randn(3, 8, dtype=torch.float16) * 50

        codes, scale, zero = self._q(x)
        assert scale.dtype is torch.float16

        x32 = x.to(torch.float32)
        mx, mn = x32.amax(-1, keepdim=True), x32.amin(-1, keepdim=True)
        sc = torch.where(mx == mn, torch.ones_like(mx), (mx - mn) / 3.0)
        s16, z16 = sc.to(torch.float16), mn.to(torch.float16)
        q = torch.round((x32 - z16.float()) / s16.float()).clamp(0.0, 3.0)
        assert torch.equal(codes, q.to(torch.uint8))
        assert torch.equal(scale, s16) and torch.equal(zero, z16)

    def test_a_diverged_model_gives_defined_codes_not_undefined_behaviour(self):
        """nan in the KV must not reach `.to(uint8)`, which is UB."""
        x = torch.tensor([[float("nan"), 1.0, 2.0, 3.0]], dtype=torch.float32)
        codes, scale, zero = self._q(x)
        assert codes.dtype is torch.uint8
        assert int(codes.max()) <= 3

    def test_a_spread_below_the_grid_subnormal_does_not_divide_by_zero(self):
        """scale underflowing the grid to 0 used to give 0/0 = nan -> UB."""
        x = torch.tensor([[0.0, 1e-44, 2e-44, 3e-44]], dtype=torch.float32)
        codes, scale, zero = self._q(x)
        assert torch.isfinite(scale).all() and (scale != 0).all()
        assert int(codes.max()) <= 3

    def test_the_store_hands_one_dtype_to_both_the_table_and_the_quantizer(self):
        """A mismatch would silently down-cast the grid on scatter."""
        from modules.quant import QuantizedStore, grid_dtype_for

        store = QuantizedStore(
            window_size=8, head_dim=16, num_kv_heads=2, n_slots=4,
            grid_dtype=grid_dtype_for(torch.bfloat16),
        )
        table = store.ensure(1, torch.device("cpu"))
        assert store.grid_dtype is torch.bfloat16
        for t in (table.key_scale, table.key_zero, table.val_scale, table.val_zero):
            assert t.dtype is torch.bfloat16
