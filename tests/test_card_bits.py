"""The ``quant_card_bits`` knob: every gate-card field at 8, 4 or 2 bits.

What has to hold, in order of how badly it fails if it does not:

1. **The default is the shipped card.** Same shapes, same dtypes, same 272
   B/head, same budget. A knob that moved the default would move every number in
   the repo without a config saying so.
2. **Storage, price and encoder agree at every width.** The slot table allocates
   what ``sketch_bytes_per_head`` charges for, and ``build_sketch`` writes what
   the table allocated. A field that is stored wider than it is priced is a
   window the budget does not know about.
3. **The knob reaches the cache from every runner** -- the failure
   ``quant_budget_mode`` and ``quant_gate_ratio`` each had once.
4. **The kernels unpack every width the way the encoder packs it.** Run under
   Triton's CPU interpreter when Triton is installed; see
   ``tests/interp_card_kernels.py``.

CPU-only.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from modules.quant.sketch import (CARD_BITS_ALLOWED, CARD_BITS_DEFAULT,
                                  CardBits, Sketch, _dq_symb, _q_symb, _width,
                                  build_sketch,
                                  card_field_layout, decode_sketch,
                                  gate_and_score, parse_card_bits,
                                  sketch_bytes_per_head)
from modules.quant.slots import SKETCH_FIELDS, QuantSlotTable
from modules.quant.store import QuantizedStore

REPO = Path(__file__).resolve().parents[1]
B, N, H, S, D = 2, 6, 4, 8, 128
SCALING = 1.0 / math.sqrt(D)
WIDTHS = [None, 8, 4, 2, "mu=2,v=4,t=2,vm=8", {"v": 4}]


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec, want", [
    (None, (4, 8, 8, 4)),
    (4, (4, 4, 4, 4)),
    (2, (2, 2, 2, 2)),
    ("int2", (2, 2, 2, 2)),
    ("8", (8, 8, 8, 8)),
    ({"v": 4}, (4, 4, 8, 4)),
    ({"mu": "int2", "vm": 8}, (2, 8, 8, 8)),
    ("mu=2, t=4", (2, 8, 4, 4)),
    ([2, 4, 8, 2], (2, 4, 8, 2)),
    (CardBits(8, 8, 8, 8), (8, 8, 8, 8)),
])
def test_every_accepted_form_parses(spec, want):
    assert tuple(parse_card_bits(spec)) == want


@pytest.mark.parametrize("spec", [
    3, 1, 16, True, "int3", "mu=5", {"x": 4}, {"mu": True}, [4, 4], "v4", 4.0,
])
def test_anything_else_raises(spec):
    """A typo must not fall back to the default: the run would say one card and
    use another."""
    with pytest.raises(ValueError):
        parse_card_bits(spec)


def test_the_default_is_the_shipped_card():
    assert CARD_BITS_DEFAULT == CardBits(mu=4, v=8, t=8, vm=4)
    assert parse_card_bits(None) is CARD_BITS_DEFAULT


# ---------------------------------------------------------------------------
# the codec
# ---------------------------------------------------------------------------


def test_int2_is_ternary_with_the_crumb_lane_map():
    """Codes {-1, 0, +1}, stored +2 so every crumb is unsigned, element j at
    shift 2*(j % 4) -- the int2 packer's convention, low lanes first."""
    x = torch.tensor([[1.0, 0.0, -1.0, 0.0]])
    packed, scale = _q_symb(x, 2)
    assert packed.dtype == torch.uint8 and packed.shape == (1, 1)
    assert int(packed) == (3 | (2 << 2) | (1 << 4) | (2 << 6))
    assert torch.equal(_dq_symb(packed, scale, 2), x)


@pytest.mark.parametrize("bits", [8, 4, 2])
def test_codes_use_the_whole_symmetric_range(bits):
    qmax = (1 << (bits - 1)) - 1
    x = torch.randn(16, 4, D) * 3.0
    packed, scale = _q_symb(x, bits)
    width, dtype = card_field_layout(D, bits)
    assert packed.shape == (16, 4, width) and packed.dtype == dtype
    codes = torch.round(_dq_symb(packed, scale, bits) / scale.float().unsqueeze(-1))
    assert codes.min() == -qmax and codes.max() == qmax


def test_4_bit_path_is_the_old_int4_encoder():
    """`_q_sym4` is the wrapper the eviction and the tests used before the knob;
    at 4 bits the generic encoder must be it exactly."""
    from modules.quant.sketch import _dq_sym4, _q_sym4
    x = torch.randn(32, 4, D)
    p4, s4 = _q_sym4(x)
    pg, sg = _q_symb(x, 4)
    assert torch.equal(p4, pg) and torch.equal(s4, sg)
    assert torch.equal(_dq_sym4(p4, s4), _dq_symb(pg, sg, 4))


def test_a_field_that_does_not_fill_whole_bytes_raises():
    with pytest.raises(ValueError):
        card_field_layout(6, 2)


def _symbolic(v: int):
    """``v`` as a SymInt: what a width read off ``store.card_bits`` becomes
    inside the compiled eviction (``torch.compile(dynamic=True)``) on torch
    2.6, where Dynamo unspecializes ints reached from a frame local."""
    from torch._dynamo.source import ConstantSource
    from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
    return ShapeEnv().create_unspecified_symint_and_symbol(
        v, ConstantSource(f"bits{v}"), DimDynamic.DYNAMIC)


@pytest.mark.parametrize("bits", CARD_BITS_ALLOWED)
def test_width_turns_a_symbolic_width_into_a_python_int(bits):
    w = _width(_symbolic(bits))
    assert type(w) is int and w == bits


@pytest.mark.parametrize("spec", [None, 2, "mu=2,v=4,t=2,vm=8"])
def test_a_symbolic_card_width_packs_the_same_codes(spec):
    """The encoder must pin a symbolic width before doing arithmetic on it.

    Unpinned, ``demote_many``'s ``build_sketch`` raised ``cannot determine
    truth value of Relational`` on torch 2.6 (``1 << (s - 1)``), which knocked
    the eviction off its ``dynamic=True`` compile, at every width including
    the default. torch 2.14 accepts the arithmetic and silently packs
    DIFFERENT codes at 4 and 2 bits, so this compares bytes, not just "runs".
    """
    bits = parse_card_bits(spec)
    k, a, v, va = _fixture()
    want = build_sketch(k, a, v, va, bits)
    got = build_sketch(k, a, v, va, CardBits(*map(_symbolic, bits)))
    for name, x, y in zip(Sketch._fields, got, want):
        assert x.dtype == y.dtype and torch.equal(x, y), name
    # The decoder takes the same pin (the gate's CPU reference unpacks with it).
    assert torch.equal(_dq_symb(want.mu_q, want.mu_s, _symbolic(bits.mu)),
                       _dq_symb(want.mu_q, want.mu_s, bits.mu))


# ---------------------------------------------------------------------------
# price == storage == encoder
# ---------------------------------------------------------------------------


def test_card_bytes_at_the_shipped_geometry():
    """Llama-3.1-8B at ws=8: 272 B/head shipped, 204 all-int4, 106 all-int2."""
    assert sketch_bytes_per_head(128, 8) == 272
    assert sketch_bytes_per_head(128, 8, 4) == 204
    assert sketch_bytes_per_head(128, 8, 2) == 106
    assert sketch_bytes_per_head(128, 8, 8) == 400


@pytest.mark.parametrize("spec", WIDTHS)
def test_the_table_stores_exactly_what_the_budget_prices(spec):
    bits = parse_card_bits(spec)
    t = QuantSlotTable(1, 1, S, D, H, torch.device("cpu"), sketch=True,
                       card_bits=bits)
    stored = sum(getattr(t, f).numel() * getattr(t, f).element_size()
                 for f in SKETCH_FIELDS)
    assert stored == H * sketch_bytes_per_head(D, S, bits)


@pytest.mark.parametrize("spec", WIDTHS)
def test_build_sketch_writes_what_the_table_allocated(spec):
    bits = parse_card_bits(spec)
    t = QuantSlotTable(1, 3, S, D, H, torch.device("cpu"), sketch=True,
                       card_bits=bits)
    k, v = torch.randn(3, H, S, D), torch.randn(3, H, S, D)
    card = build_sketch(k, k.mean((0, 2)), v, v.mean((0, 2)), bits)
    for name, field in zip(SKETCH_FIELDS, card):
        col = getattr(t, name)
        assert field.shape == col.shape[1:] and field.dtype == col.dtype, name


def test_default_table_shapes_are_unchanged():
    t = QuantSlotTable(1, 1, S, D, H, torch.device("cpu"), sketch=True)
    assert (t.sk_mu_q.shape[-1], t.sk_mu_q.dtype) == (D // 2, torch.uint8)
    assert (t.sk_v_q.shape[-1], t.sk_v_q.dtype) == (D, torch.int8)
    assert (t.sk_t_q.shape[-1], t.sk_t_q.dtype) == (S, torch.int8)
    assert (t.sk_vm_q.shape[-1], t.sk_vm_q.dtype) == (D // 2, torch.uint8)


def test_join_layers_refuses_tables_at_different_widths():
    a = QuantSlotTable(1, 2, S, D, H, torch.device("cpu"), sketch=True)
    b = QuantSlotTable(1, 2, S, D, H, torch.device("cpu"), sketch=True,
                       card_bits=2)
    with pytest.raises(RuntimeError):
        QuantSlotTable.join_layers([a, b])
    assert QuantSlotTable.join_layers([b, b]).card_bits == CardBits(2, 2, 2, 2)


# ---------------------------------------------------------------------------
# the estimator at each width
# ---------------------------------------------------------------------------


def _fixture(seed=0, n=48):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(H, D, generator=g) * 0.4
    a[:, 3] += 40.0
    k = a[None, :, None, :] + torch.randn(n, H, S, D, generator=g)
    k[:, :, 0, :] += torch.randn(n, H, D, generator=g) * 6.0
    v = torch.randn(n, H, S, D, generator=g)
    return k, a, v, v.mean(dim=(0, 2))


@pytest.mark.parametrize("spec", [8, 4, 2])
def test_every_width_decodes_to_the_same_shapes_and_finite_values(spec):
    bits = parse_card_bits(spec)
    k, a, v, va = _fixture()
    s = build_sketch(k, a, v, va, bits)
    for f in decode_sketch(s, a, bits):
        assert torch.isfinite(f).all()
    mu, vv, t = decode_sketch(s, a, bits)
    assert mu.shape == vv.shape == (k.shape[0], H, D) and t.shape == (k.shape[0], H, S)
    card = Sketch(*(f.unsqueeze(0) for f in s))
    q = torch.randn(1, H * 2, D)
    logmass, est = gate_and_score(q, card, a, SCALING, bits)
    assert logmass.shape == (1, H * 2, k.shape[0]) and torch.isfinite(logmass).all()


def test_wider_is_closer_to_the_unquantised_estimate():
    """Not a quality claim -- a sanity ordering. The mass estimate at each width
    against the same card built at fp32 precision (int8 is the closest proxy the
    encoder offers): int8 < int4 < int2 in error, averaged over seeds."""
    errs = {8: 0.0, 4: 0.0, 2: 0.0}
    for seed in range(4):
        k, a, v, va = _fixture(seed)
        q = torch.randn(1, H * 2, D, generator=torch.Generator().manual_seed(seed))
        # Exact logsumexp over each window's tokens: what the card estimates.
        qg = q.reshape(1, H, 2, D)
        x = SCALING * torch.einsum("nhwd,bhrd->bhrnw", k, qg)
        exact = torch.logsumexp(x, -1).reshape(1, H * 2, k.shape[0])
        for w in errs:
            bits = CardBits(w, w, w, w)
            card = Sketch(*(f.unsqueeze(0) for f in build_sketch(k, a, v, va, bits)))
            lm, _ = gate_and_score(q, card, a, SCALING, bits)
            errs[w] += (lm - exact).abs().mean().item()
    assert errs[8] < errs[4] < errs[2], errs


# ---------------------------------------------------------------------------
# through the store
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", WIDTHS)
def test_the_store_demotes_selects_and_centroids_at_every_width(spec):
    st = QuantizedStore(window_size=S, head_dim=D, num_kv_heads=H, n_slots=32,
                        memoize_read=False, sketch_enabled=True, card_bits=spec)
    st.ensure(B, torch.device("cpu"))
    assert st.table.card_bits == parse_card_bits(spec)
    g = torch.Generator().manual_seed(0)
    st.demote_many(
        st.table.free_slots(N), torch.ones(B, N, dtype=torch.bool),
        torch.arange(N).reshape(1, N).expand(B, N).contiguous(),
        torch.randn(B, N, H, S, D, generator=g), torch.randn(B, N, H, S, D, generator=g),
        torch.arange(N * S).reshape(1, N, S).expand(B, N, S).contiguous(),
        keys_post_rope=torch.randn(B, N, H, S, D, generator=g))
    st.commit_active_count(N)
    keep, logmass, slots = st.gate_and_select(torch.randn(B, H * 2, D), SCALING, 0.5)
    assert keep.shape == (B, H, N) and int(keep[0, 0].sum()) == 3
    assert torch.isfinite(logmass).all()
    cent = st.window_centroids(slots)
    assert cent.shape == (B, N, H, D) and torch.isfinite(cent).all()
    # ratio 1.0 still selects every window, at every width.
    keep_all, _, _ = st.gate_and_select(torch.randn(B, H * 2, D), SCALING, 1.0)
    assert bool(keep_all.all())


# ---------------------------------------------------------------------------
# through the layer-major cache, into _run_fused
# ---------------------------------------------------------------------------


class _Rope(torch.nn.Module):
    """Deterministic rotary stub, shaped like HF's. ``x`` is not always a
    head-dim tensor (HF passes it for dtype/device only), so ``d`` is fixed."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, x, position_ids):
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * torch.arange(
            1, self.d // 2 + 1, dtype=torch.float32) * 0.01
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


# Two widths, not all: the defect is width-independent and each case pays
# ~90 s of CPU Inductor compiles for the eviction.
@pytest.mark.parametrize("spec", [None, 2])
def test_the_layer_major_decode_hands_run_fused_the_card_widths(spec, monkeypatch):
    """Every decode step the production cache arms reaches ``fused_gate`` with
    the widths the card was built at.

    The layer-major rebuild re-slices ``_gate_ctx``'s dict per layer. It used to
    list the keys by hand and dropped ``bits``, so ``_run_fused`` raised
    ``KeyError: 'bits'`` on the first gated step of every run at every width,
    the default included. The store-level tests above never build that dict.

    The gate runs for real (its CPU reference, at ``spec``'s widths); only the
    decode kernel is stubbed, because it is Triton-or-raise.
    """
    from modules.windowed_cache import flash_decode
    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig

    L, Bt, h_kv, h_q, d, ws, P = 2, 2, 2, 4, 16, 4, 32
    model_cfg = types.SimpleNamespace(
        num_key_value_heads=h_kv, num_attention_heads=h_q,
        hidden_size=h_q * d, head_dim=d)
    cache = WindowedCache(
        config=WindowedCacheConfig(
            window_size=ws, num_sink_tokens=0, local_window_size=ws,
            cache_budget=0.35, quant_ratio=0.5, first_eviction_step=0,
            quant_card_bits=spec),
        prefill_len=P, model_config=model_cfg, kv_dtype=torch.float32,
        rope_module=_Rope(d), num_layers=L, max_tokens=P + 64)
    cache._fused_decode_active = True      # what hooks.py sets on a CUDA flash run
    want = parse_card_bits(spec)

    armed, gate_bits, vm_seen = [], [], []
    real_gate = flash_decode.fused_gate

    def spy_gate(*a, **kw):
        gate_bits.append(kw.get("bits"))
        return real_gate(*a, **kw)

    def stub_decode(q_hd, k_fp, v_fp, qtier, scaling, num_sink, n_body_win,
                    *, sel, logmass, centroids, vm_bits: int = 4):
        vm_seen.append(vm_bits)
        w = n_body_win + qtier["k_codes"].shape[1]
        return (torch.zeros_like(q_hd),
                torch.rand(q_hd.shape[0], q_hd.shape[1], w))

    monkeypatch.setattr(flash_decode, "set_pending", armed.append)
    monkeypatch.setattr(flash_decode, "fused_gate", spy_gate)
    monkeypatch.setattr(flash_decode, "fused_two_tier_decode", stub_decode)

    def merged_windows(i):
        st, store = cache._states[i], cache._stores[i]
        return -(-st.seq_length // ws) + store.num_active_windows

    g = torch.Generator().manual_seed(0)
    for i in range(L):
        k = torch.randn(Bt, h_kv, P, d, generator=g)
        cache.update(k, torch.randn_like(k), i,
                     cache_kwargs={"cache_position": torch.arange(P)})
    for i in range(L):
        cache.cache_kwargs[i]["window_scores"] = torch.rand(
            Bt, h_q, merged_windows(i), generator=g)

    # Step 0 arms on the per-layer path, steps 1-2 on the joint one.
    fused = joint_fused = 0
    for t in range(3):
        for i in range(L):
            k = torch.randn(Bt, h_kv, 1, d, generator=g)
            k_fp, v_fp = cache.update(
                k, torch.randn_like(k), i,
                cache_kwargs={"cache_position": torch.tensor([P + t])})
            if not armed:
                # No Q tier yet: the score hook's job, emulated.
                cache.cache_kwargs[i]["window_scores"] = torch.rand(
                    Bt, h_q, merged_windows(i), generator=g)
                continue
            ctx = armed.pop()
            assert not armed
            joint = cache._joint is not None
            store = cache._joint_store if joint else cache._stores[i]
            n = store.num_active_windows
            # The gate dict must carry everything _gate_ctx builds.
            ref = cache._gate_ctx(store, store.table.active_order(n), n)
            assert ctx["gate"].keys() == ref.keys()
            assert ctx["gate"]["bits"] == want
            q = torch.randn(Bt, 1, h_q, d, generator=g)
            flash_decode._run_fused(ctx, q, k_fp.transpose(1, 2),
                                    v_fp.transpose(1, 2))
            fused += 1
            joint_fused += joint

    assert joint_fused > 0, (
        "no layer-major step armed the fused decode: the test proves nothing")
    assert gate_bits == [want] * fused
    assert vm_seen == [want.vm] * fused


# ---------------------------------------------------------------------------
# config and budget
# ---------------------------------------------------------------------------

MODEL_CONFIG = types.SimpleNamespace(
    num_key_value_heads=8, head_dim=128, num_hidden_layers=32,
    hidden_size=4096, num_attention_heads=32,
)


def _resolve(card_bits=None, mode="bytes", q=0.7):
    from modules.windowed_cache.config import WindowedCacheConfig
    cfg = WindowedCacheConfig(
        window_size=8, num_sink_tokens=5, local_window_size=128,
        cache_budget=0.20, quant_ratio=q, quant_budget_mode=mode,
        quant_card_bits=card_bits,
    )
    return cfg.resolve(3600, MODEL_CONFIG, torch.float16, 128)


def test_default_resolve_is_unchanged():
    r = _resolve()
    assert r.quant_card_bits == CARD_BITS_DEFAULT
    assert r.bytes_per_gate_card == 8 * 272 == 2176
    assert r.bytes_per_q_window == 6432 + 2176


@pytest.mark.parametrize("spec, card", [(4, 8 * 204), (2, 8 * 106), (8, 8 * 400)])
def test_the_budget_prices_the_card_it_will_store(spec, card):
    r = _resolve(spec)
    assert r.bytes_per_gate_card == card
    assert r.bytes_per_q_window == 6432 + card


def test_a_cheaper_card_buys_more_int2_windows_under_bytes_mode():
    n_default, n4, n2 = (_resolve(s).N_q for s in (None, 4, 2))
    assert n_default < n4 < n2


def test_tokens_mode_keeps_the_window_count():
    assert _resolve(None, "tokens").N_q == _resolve(2, "tokens").N_q


def test_the_yaml_config_rejects_a_bad_width_at_load():
    from utils.config import CacheConfig, ConfigValidationError
    assert CacheConfig(quant_card_bits=2).quant_card_bits == 2
    with pytest.raises(ConfigValidationError):
        CacheConfig(quant_card_bits=3)


def test_the_sidecar_record_is_the_parsed_widths():
    from utils.cache_factory import quant_card_bits_record
    rec = quant_card_bits_record(types.SimpleNamespace(quant_card_bits=None))
    assert rec == {"mu": 4, "v": 8, "t": 8, "vm": 4}
    assert quant_card_bits_record(types.SimpleNamespace(quant_card_bits="int2")) == {
        "mu": 2, "v": 2, "t": 2, "vm": 2}


@pytest.mark.parametrize("runner", [
    "gsm8k_runner", "longbench_runner", "ours_parity_runner", "perf_runner",
    "ruler_runner",
])
def test_every_runner_forwards_the_knob(runner):
    """A YAML key a runner never forwards is inert, and the run record then
    names a card the run did not use."""
    src = (REPO / "modules" / "evaluation" / f"{runner}.py").read_text()
    assert "quant_card_bits=" in src


# ---------------------------------------------------------------------------
# the kernels, under Triton's CPU interpreter
# ---------------------------------------------------------------------------


def test_kernels_unpack_every_width_under_the_triton_interpreter():
    """Both Triton kernels, executed by ``TRITON_INTERPRET=1`` on CPU.

    For each width: packed codes must give BITWISE the same ``logmass``,
    selection, output and window scores as the same integer codes stored as
    int8 -- which isolates the packed unpack -- and the gate must match the
    PyTorch reference. This is the interpreter, not a GPU compile: it checks the
    kernels' arithmetic and lane maps, not their lowering.
    """
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1",
               PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, str(REPO / "tests" / "interp_card_kernels.py")],
        env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
