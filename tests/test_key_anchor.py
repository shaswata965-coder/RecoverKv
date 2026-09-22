"""The int2 key zero-point is stored relative to an anchor on Qwen2 — and why.

The Qwen2.5 LongBench column fell well short of the ``int2_qwen`` (QEvict) run,
worst on triviaqa, a task that barely notices a smaller cache on Llama. The
cause is the one-byte grid (``8a8cdc0``), and the mechanism is specific to a
model whose ``k_proj`` has a **bias**:

* A key bias gives some channels a large value that barely moves from token to
  token. Pre-RoPE, such a channel's per-window range is ~0 while its zero-point
  (the window min) is the bias itself — hundreds.
* The one-byte grid stores a zero as an int8 code times a scale shared by 32
  channels, i.e. only to within ``max|zero| / 254`` of the group. The per-entry
  grid ``int2_qwen`` used stored the window min in the KV dtype, which holds a
  KV-dtype number *exactly*.
* The resulting error is the same in EVERY window (same bias, same code), so it
  is not noise: times the query's own massive value in that channel it shifts
  every int2 logit against every fp logit by several nats — scaling the whole
  int2 tier's attention mass (~88% of the retained windows at q = 0.7) up or
  down by orders of magnitude.

Storing ``zero - anchor`` (a frozen per-(row, head, channel) pre-RoPE mean)
makes a bias channel's residual exactly 0, and the shared scale is then set by
genuine per-window deviation. It costs no bytes per window. It is switched on
where the key projection has a bias, so the Llama/Mistral columns are
byte-identical.

The fixtures are synthetic — no Qwen weights on this box — built to the property
that matters (large near-constant key channels aligned with large query
channels). What they establish is the mechanism and that the fix removes it;
the LongBench number needs a GPU run.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from modules.quant.quantizer import (  # noqa: E402
    dequantize_key_windows,
    grid_dtype_for,
    quantize_key_windows,
)
from modules.quant.slots import QuantSlotTable  # noqa: E402
from modules.quant.store import QuantizedStore  # noqa: E402

H, WS, D = 4, 8, 128


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _qwen_like(n=128, dtype=torch.float16, b_max=250.0, n_bias=6, seed=0):
    """Pre-RoPE keys with bias-dominated channels, and queries aligned to them.

    ``n_bias`` channels per head carry a CONSTANT value of magnitude 40..b_max
    (a k_proj bias with ~no content weight); the rest are ordinary content
    channels. Queries carry a massive value (50..250) in the same channels,
    which is what makes an error there expensive.
    """
    g = torch.Generator().manual_seed(seed)
    off = torch.randn(H, D, generator=g) * 1.5
    sig = torch.rand(H, D, generator=g) * 2.0 + 0.5
    idx = torch.stack([torch.randperm(D, generator=g)[:n_bias] for _ in range(H)])
    mag = torch.rand(H, n_bias, generator=g) * (b_max - 40.0) + 40.0
    sgn = torch.where(torch.rand(H, n_bias, generator=g) > 0.5, 1.0, -1.0)
    q = torch.randn(64, H, D, generator=g) * 1.5
    for h in range(H):
        off[h, idx[h]] = mag[h] * sgn[h]
        sig[h, idx[h]] = 0.0
        q[:, h, idx[h]] += (torch.rand(n_bias, generator=g) * 200 + 50) * torch.where(
            torch.rand(n_bias, generator=g) > 0.5, 1.0, -1.0)
    k = off[None, :, None, :] + torch.randn(n, H, WS, D, generator=g) * sig[None, :, None, :]
    return k.to(dtype), q, idx


def _llama_like(n=128, dtype=torch.float16, seed=0):
    """No bias: a few outlier channels with a DC offset, all of them varying."""
    g = torch.Generator().manual_seed(seed)
    off = torch.randn(H, D, generator=g) * 1.5
    idx = torch.randperm(D, generator=g)[:4]
    off[:, idx] += 12.0
    sig = torch.rand(H, D, generator=g) * 2.0 + 0.5
    sig[:, idx] = 3.0
    k = off[None, :, None, :] + torch.randn(n, H, WS, D, generator=g) * sig[None, :, None, :]
    q = torch.randn(64, H, D, generator=g) * 1.5
    q[:, :, idx] += 8.0
    return k.to(dtype), q


def _anchor_of(k):
    """What the store freezes: the per-(head, channel) mean over windows and tokens."""
    return k.float().mean(dim=(0, 2))                               # [H, D]


def _per_entry_roundtrip(k):
    """The ``int2_qwen`` grid: scale and zero stored per entry in the KV dtype."""
    x = k.float()
    mx, mn = x.amax(2, keepdim=True), x.amin(2, keepdim=True)
    s = torch.where(mx == mn, torch.ones_like(mx), (mx - mn) / 3.0)
    s = s.to(grid_dtype_for(k.dtype)).float()
    z = mn.to(grid_dtype_for(k.dtype)).float()
    return torch.round((x - z) / s).clamp_(0, 3) * s + z


def _roundtrip(k, anchor=None):
    a = None if anchor is None else anchor.unsqueeze(0).expand(k.shape[0], -1, -1)
    codes, s, z = quantize_key_windows(k, anchor=a)
    return dequantize_key_windows(codes, s, z, WS, out_dtype=torch.float32, anchor=a)


def _tier_shift(k, k_hat, q):
    """Mean |tier-wide logit shift| over (query, head), and the per-key residual.

    The shift is the part of the logit error common to every key of the tier —
    what moves the tier's total attention mass against the fp tier. The
    residual is ordinary per-key quantisation noise.
    """
    ref = torch.einsum("nhwd,bhd->bhnw", k.float(), q) / math.sqrt(D)
    got = torch.einsum("nhwd,bhd->bhnw", k_hat.float(), q) / math.sqrt(D)
    d = got - ref
    common = d.mean(dim=(2, 3))
    return common.abs().mean().item(), (d - common[..., None, None]).abs().mean().item()


# ---------------------------------------------------------------------------
# the mechanism, and the fix, at the quantizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_the_one_byte_grid_shifts_the_whole_tier_on_bias_channels(dtype):
    """The regression, measured: several nats of tier-wide logit shift.

    The per-entry grid it replaced shifts the same keys by ~0.01, and the
    per-key noise is the same either way — the one-byte grid did not add
    noise, it added a BIAS, which is why it hid behind every aggregate error.
    """
    k, q, _ = _qwen_like(dtype=dtype)
    shift_1b, noise_1b = _tier_shift(k, _roundtrip(k), q)
    shift_pe, noise_pe = _tier_shift(k, _per_entry_roundtrip(k), q)
    assert shift_pe < 0.05
    assert shift_1b > 1.0, shift_1b
    assert noise_1b == pytest.approx(noise_pe, rel=0.15)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_anchoring_removes_the_tier_shift(dtype):
    """Anchored, the one-byte grid is as good as the per-entry grid it replaced."""
    k, q, _ = _qwen_like(dtype=dtype)
    shift, noise = _tier_shift(k, _roundtrip(k, _anchor_of(k)), q)
    shift_pe, noise_pe = _tier_shift(k, _per_entry_roundtrip(k), q)
    assert shift <= max(2.0 * shift_pe, 0.05), (shift, shift_pe)
    assert noise == pytest.approx(noise_pe, rel=0.15)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_a_bias_channel_round_trips_exactly_when_anchored(dtype):
    """The constant channels come back bit-exact; un-anchored they do not."""
    k, _, idx = _qwen_like(dtype=dtype)
    exact = k.float()
    got = _roundtrip(k, _anchor_of(k))
    raw = _roundtrip(k)
    for h in range(H):
        cols = idx[h]
        assert torch.equal(got[:, h][..., cols], exact[:, h][..., cols])
    worst_raw = max(
        (raw[:, h][..., idx[h]] - exact[:, h][..., idx[h]]).abs().max().item()
        for h in range(H))
    assert worst_raw > 0.05, worst_raw


def test_even_a_modest_query_sees_the_shift_through_the_real_round_trip():
    """No extreme values needed. Bias channels in the low-frequency band (where
    Qwen2.5's massive q/k values sit), taken through the ACTUAL demotion round
    trip -- a real Qwen2 rotary at rope_theta 1e6, rotate then unrotate, in the
    cache dtype -- with the query only +20 in those channels: the one-byte grid
    still moves the whole tier by most of a nat (a ~2x mass error), the
    per-entry grid and the anchored one by a few hundredths.
    """
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    from modules.quant.effective import rotate_key_window, unrotate_key_window

    rope = Qwen2RotaryEmbedding(config=transformers.Qwen2Config(
        hidden_size=512, num_attention_heads=4, num_key_value_heads=4,
        rope_theta=1.0e6))
    low = torch.cat([torch.arange(40, 64), torch.arange(104, 128)])
    n = 128
    got = {"one": [], "anchored": [], "per_entry": []}
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        k_pre = torch.randn(H, n * WS, D, generator=g) * 1.5
        cols = low[torch.randperm(len(low), generator=g)[:6]]
        k_pre[:, :, cols] = (torch.rand(6, generator=g) * 200 + 40) * torch.where(
            torch.rand(6, generator=g) > 0.5, 1.0, -1.0)
        pos = torch.arange(n * WS) + 3000
        for dtype in (torch.float16, torch.bfloat16):
            back = unrotate_key_window(
                rotate_key_window(k_pre.to(dtype), pos, rope), pos, rope)
            k = back.reshape(H, n, WS, D).permute(1, 0, 2, 3).contiguous()
            q = torch.randn(32, H, D, generator=g)
            q[:, :, cols] += 20.0
            got["one"].append(_tier_shift(k, _roundtrip(k), q)[0])
            got["anchored"].append(_tier_shift(k, _roundtrip(k, _anchor_of(k)), q)[0])
            got["per_entry"].append(_tier_shift(k, _per_entry_roundtrip(k), q)[0])
    mean = {name: sum(v) / len(v) for name, v in got.items()}
    assert mean["one"] > 0.5, mean
    assert mean["anchored"] < 0.05, mean
    assert mean["per_entry"] < 0.1, mean


def test_llama_like_keys_need_no_anchor():
    """The control. Without a key bias the one-byte grid never had this problem,
    which is why it passed its own validation and why the anchor is off there."""
    k, q = _llama_like()
    shift_1b, _ = _tier_shift(k, _roundtrip(k), q)
    shift_pe, _ = _tier_shift(k, _per_entry_roundtrip(k), q)
    assert shift_1b < 0.1
    assert shift_1b == pytest.approx(shift_pe, abs=0.05)


def test_a_zero_anchor_is_the_absolute_encoding_bit_for_bit():
    """``anchor=0`` stores exactly what no anchor stores — the anchored path
    degenerates to the historic one, it is not a second approximation of it."""
    k, _, _ = _qwen_like(dtype=torch.float16)
    zero = torch.zeros(k.shape[0], H, D)
    c0, s0, z0 = quantize_key_windows(k)
    c1, s1, z1 = quantize_key_windows(k, anchor=zero)
    assert torch.equal(c0, c1)
    for a, b in ((s0, s1), (z0, z1)):
        assert torch.equal(a.q, b.q) and torch.equal(a.s, b.s)
    assert torch.equal(
        dequantize_key_windows(c0, s0, z0, WS, out_dtype=torch.float32),
        dequantize_key_windows(c1, s1, z1, WS, out_dtype=torch.float32, anchor=zero),
    )


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


def _demote(store, k_pre, valid=None, first_wid=0):
    """Demote ``[B, n, H, WS, D]`` pre-RoPE keys into free slots."""
    B, n = k_pre.shape[:2]
    store.ensure(B, torch.device("cpu"), grid_dtype=grid_dtype_for(k_pre.dtype))
    if valid is None:
        valid = torch.ones(B, n, dtype=torch.bool)
    wid = torch.arange(first_wid, first_wid + n).unsqueeze(0).expand(B, -1).clone()
    pos = (torch.arange(n * WS).reshape(n, WS) + first_wid * WS).unsqueeze(0).expand(B, -1, -1)
    store.demote_many(
        store.table.free_slots(n), valid, wid, k_pre, torch.randn_like(k_pre.float()).to(k_pre.dtype),
        pos.clone(),
    )
    store.commit_active_count(store.num_active_windows + n)
    return wid


def _store(key_anchor=True):
    return QuantizedStore(WS, D, H, n_slots=64, memoize_read=False,
                          sketch_enabled=False, key_anchor=key_anchor)


def test_the_store_freezes_the_anchor_at_the_first_demotion():
    """A later demotion must not move it: every code already stored was fit
    against it, so moving it would shift the whole tier."""
    k1, _, _ = _qwen_like(n=16, seed=1)
    k2, _, _ = _qwen_like(n=16, seed=2, b_max=90.0)
    st = _store()
    _demote(st, k1.unsqueeze(0))
    frozen = st.key_anchor_rows().clone()
    assert torch.allclose(frozen[0], _anchor_of(k1))
    _demote(st, k2.unsqueeze(0), first_wid=16)
    assert torch.equal(st.key_anchor_rows(), frozen)


def test_the_anchor_ignores_invalid_lanes():
    """Invalid lanes ride the demote batch with whatever the gather put there;
    they must not steer an encoding every later window inherits."""
    k, _, _ = _qwen_like(n=8, seed=3)
    junk = k.clone()
    junk[4:] = 1.0e4
    valid = torch.tensor([[True] * 4 + [False] * 4])
    st = _store()
    _demote(st, junk.unsqueeze(0), valid=valid)
    assert torch.allclose(st.key_anchor_rows()[0], _anchor_of(k[:4]))


def test_promotion_and_the_read_path_add_the_anchor_back():
    """promote_many (Q -> fp) returns the bias channels exactly."""
    k, _, idx = _qwen_like(n=8, seed=4)
    st = _store()
    wid = _demote(st, k.unsqueeze(0))
    has, _, slot, _ = st.lookup(wid)
    assert bool(has.all())
    keys, _, _ = st.promote_many(slot, torch.ones_like(has), torch.float32)
    for h in range(H):
        assert torch.equal(keys[0][:, h][..., idx[h]], k.float()[:, h][..., idx[h]])


def test_an_unanchored_store_is_the_plain_quantizer():
    """``key_anchor=False`` (Llama, Mistral): no anchor, codes as before."""
    k, _, _ = _qwen_like(n=8, seed=5)
    st = _store(key_anchor=False)
    wid = _demote(st, k.unsqueeze(0))
    assert st.key_anchor_rows() is None
    _, _, slot, _ = st.lookup(wid)
    codes, _, _ = quantize_key_windows(k)
    assert torch.equal(st.table.key_codes[0, slot[0]], codes)


def test_an_anchored_store_with_nothing_demoted_refuses_to_decode():
    st = _store()
    with pytest.raises(RuntimeError, match="has not demoted"):
        st.key_anchor_rows()


def test_join_layers_carries_the_anchor_row_major():
    ka, _, _ = _qwen_like(n=8, seed=6)
    kb, _, _ = _qwen_like(n=8, seed=7)
    sa, sb = _store(), _store()
    _demote(sa, ka.unsqueeze(0))
    _demote(sb, kb.unsqueeze(0))
    joint = QuantizedStore.join_layers([sa, sb])
    assert joint.key_anchor
    assert torch.equal(joint.key_anchor_rows(),
                       torch.cat([sa.key_anchor_rows(), sb.key_anchor_rows()]))


def test_join_layers_refuses_a_partial_anchor():
    ka, _, _ = _qwen_like(n=8, seed=8)
    sa, sb = _store(), _store()
    _demote(sa, ka.unsqueeze(0))
    sb.ensure(1, torch.device("cpu"), grid_dtype=torch.float16)
    sb.commit_active_count(sa.num_active_windows)
    with pytest.raises(RuntimeError, match="key anchor"):
        QuantizedStore.join_layers([sa, sb])


# ---------------------------------------------------------------------------
# the slot table keeps the grid dtype the quantizer fit against
# ---------------------------------------------------------------------------


def test_the_slot_table_stores_group_scales_in_the_grid_dtype():
    """A bf16 cache's grid was fit to bf16 scales and stored into fp16 buffers.

    ``write`` casts to the buffer dtype, so the port's "grid dtype follows the
    KV dtype" held inside the quantizer and was undone at storage.
    """
    fp = QuantSlotTable(1, 4, WS, D, H, torch.device("cpu"))
    bf = QuantSlotTable(1, 4, WS, D, H, torch.device("cpu"), grid_dtype=torch.bfloat16)
    for name in ("key_scale_s", "key_zero_s", "val_scale_s", "val_zero_s"):
        assert getattr(fp, name).dtype == torch.float16          # historic, unchanged
        assert getattr(bf, name).dtype == torch.bfloat16

    k, _, _ = _qwen_like(n=4, dtype=torch.bfloat16, seed=9)
    st = _store(key_anchor=False)
    wid = _demote(st, k.unsqueeze(0))
    _, _, slot, _ = st.lookup(wid)
    _, _, zero = quantize_key_windows(k)
    stored = st.table.key_zero_s[0, slot[0]]
    assert stored.dtype == torch.bfloat16
    assert torch.equal(stored, zero.s)


# ---------------------------------------------------------------------------
# config: on where the key projection has a bias, off elsewhere
# ---------------------------------------------------------------------------


def _resolve(model_cfg, **kw):
    from modules.windowed_cache.config import WindowedCacheConfig

    cfg = WindowedCacheConfig(window_size=8, num_sink_tokens=5,
                              local_window_size=128, cache_budget=0.2,
                              quant_ratio=kw.pop("quant_ratio", 0.7), **kw)
    return cfg.resolve(4096, model_cfg, torch.float16, 32)


def _hf(name, **kw):
    transformers = pytest.importorskip("transformers")
    base = dict(hidden_size=512, num_attention_heads=4, num_key_value_heads=4,
                num_hidden_layers=2, intermediate_size=64, vocab_size=64)
    base.update(kw)
    return getattr(transformers, name)(**base)


def test_key_projection_has_bias_truth_table():
    from modules.windowed_cache.config import key_projection_has_bias

    assert key_projection_has_bias(_hf("Qwen2Config"))
    assert not key_projection_has_bias(_hf("LlamaConfig"))
    assert not key_projection_has_bias(_hf("MistralConfig"))
    # An explicit flag is believed over the model type, both ways.
    assert key_projection_has_bias(_hf("LlamaConfig", attention_bias=True))


def test_the_anchor_is_auto_on_for_qwen_and_off_for_llama_and_mistral():
    assert _resolve(_hf("Qwen2Config")).quant_key_anchor is True
    assert _resolve(_hf("LlamaConfig")).quant_key_anchor is False
    assert _resolve(_hf("MistralConfig")).quant_key_anchor is False


def test_the_anchor_knob_overrides_and_needs_a_q_tier():
    assert _resolve(_hf("LlamaConfig"), quant_key_anchor=True).quant_key_anchor
    assert not _resolve(_hf("Qwen2Config"), quant_key_anchor=False).quant_key_anchor
    assert not _resolve(_hf("Qwen2Config"), quant_ratio=0.0).quant_key_anchor
    with pytest.raises(ValueError, match="quant_key_anchor"):
        _resolve(_hf("Qwen2Config"), quant_key_anchor=1)


# ---------------------------------------------------------------------------
# the cache: resolved flag -> store -> eviction -> both read paths
# ---------------------------------------------------------------------------


def _kernel_qtier_keys(qtier):
    """The fused decode kernel's in-register Q-tier key read, restated on CPU.

    Line for line with ``_two_tier_decode_kernel``'s Q loop: unpack the crumb
    at ``(byte t//4, shift 2*(t%4))``, rebuild the one-byte grid as code times
    its group's scale, ADD THE KEY ANCHOR under ``KEY_ANCHOR``, dequantize as
    ``crumb * scale + zero``, then rotate the two halves by the cos/sin tables.
    Returns ``[B, H_kv, n*ws, D]`` in window-major token order -- the layout
    ``QuantizedStore.effective_q_tier`` returns -- so the two read paths can be
    compared directly.
    """
    kc = qtier["k_codes"].long()                         # [B, n, H, D, ws//4]
    B, n, Hk, Dk, _ = kc.shape
    ws = qtier["window_size"]
    t = torch.arange(ws)
    crumb = (kc[..., t // 4] >> (2 * (t % 4))) & 3       # [B, n, H, D, ws]

    def grid(g):
        grp = Dk // g.s.shape[-1]
        return g.q.float() * g.s.float().repeat_interleave(grp, dim=-1)

    ks, kz = grid(qtier["k_scale"]), grid(qtier["k_zero"])   # [B, n, H, D]
    anc = qtier.get("k_anchor")
    if anc is not None:
        kz = kz + anc[:, None]                            # [B, 1, H, D]
    k = crumb.float() * ks[..., None] + kz[..., None]     # [B, n, H, D, ws]
    k = k.permute(0, 2, 1, 4, 3).reshape(B, Hk, n * ws, Dk)
    half = Dk // 2
    c = qtier["cos"].float()[:, None]                     # [B, 1, n*ws, D/2]
    s = qtier["sin"].float()[:, None]
    lo, hi = k[..., :half], k[..., half:]
    return torch.cat([lo * c - hi * s, hi * c + lo * s], dim=-1)


def _qwen_cache(key_anchor=None):
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig

    mcfg = transformers.Qwen2Config(
        hidden_size=1024, num_attention_heads=8, num_key_value_heads=2,
        num_hidden_layers=1, intermediate_size=64, vocab_size=64,
        max_position_embeddings=4096, rope_theta=1.0e6)
    ccfg = WindowedCacheConfig(
        window_size=8, num_sink_tokens=4, local_window_size=16,
        cache_budget=0.5, quant_ratio=0.5, quant_key_anchor=key_anchor)
    rope = Qwen2RotaryEmbedding(config=mcfg)
    cache = WindowedCache(
        config=ccfg, prefill_len=260, model_config=mcfg, kv_dtype=torch.float32,
        rope_module=rope, num_layers=1, max_tokens=8)
    return cache, rope


def _seed_and_evict(cache, rope, seed=0):
    """Prompt keys with Qwen-style bias channels, then the first eviction."""
    from modules.quant.effective import rotate_key_window

    Hk, Dk, ws, sink = 2, 128, 8, 4
    n_win = 32
    T = sink + n_win * ws
    g = torch.Generator().manual_seed(seed)
    k_pre = torch.randn(Hk, T, Dk, generator=g) * 1.5
    # Low-frequency channels (the tail of each RoPE half) carry a constant bias.
    bias_cols = torch.tensor([60, 61, 62, 63, 124, 125, 126, 127])
    k_pre[:, :, bias_cols] = torch.tensor([180.0, -130.0, 95.0, -240.0,
                                           210.0, -75.0, 150.0, -60.0])
    pos = torch.arange(T)
    k_post = rotate_key_window(k_pre, pos, rope)
    v = torch.randn(Hk, T, Dk, generator=g)
    st = cache._states[0]
    st.replace(k_post.unsqueeze(0).clone(), v.unsqueeze(0).clone(), pos.unsqueeze(0).clone())
    st.window_scores = torch.linspace(100.0, 1.0, n_win).expand(1, 8, n_win).clone()
    st.original_window_ids = torch.arange(n_win).unsqueeze(0)
    cache._policies[0].initialize_after_prefill(T)
    cache._prefill_done[0] = True
    cache._evict_two_tier(0, step=0)
    store = cache._stores[0]
    store.commit_active_count(store.num_active_windows)
    return k_post, pos, bias_cols


def _true_q_keys(cache, k_post):
    """The fp keys the Q tier stands in for, in its window-major order."""
    store = cache._stores[0]
    _, _, qpos = store.effective_q_tier(cache.rope_module, torch.float32)
    return k_post[:, qpos[0]].unsqueeze(0), qpos


def test_the_cache_resolves_and_builds_an_anchored_store_for_qwen():
    cache, _ = _qwen_cache()
    assert cache.resolved.quant_key_anchor is True
    assert all(s.key_anchor for s in cache._stores)
    off, _ = _qwen_cache(key_anchor=False)
    assert not any(s.key_anchor for s in off._stores)


def test_the_qwen_cache_reads_its_int2_keys_without_a_tier_shift():
    """End to end on CPU: resolve -> eviction (compiled) -> materialize read.

    Measured against the fp keys each int2 window replaced, with a query that is
    massive in the bias channels as Qwen's is. Anchored, the tier-wide shift is
    per-entry-grid small; forced off, the same cache shows the regression.
    """
    torch.manual_seed(0)
    shifts = {}
    for flag in (None, False):
        cache, rope = _qwen_cache(key_anchor=flag)
        k_post, _, bias_cols = _seed_and_evict(cache, rope)
        store = cache._stores[0]
        assert store.num_active_windows > 0
        got, _, _ = store.effective_q_tier(cache.rope_module, torch.float32)
        want, _ = _true_q_keys(cache, k_post)
        q = torch.randn(16, 2, 128)
        q[:, :, bias_cols] += 150.0
        d = (torch.einsum("bhtd,qhd->qht", got.float(), q)
             - torch.einsum("bhtd,qhd->qht", want.float(), q)) / math.sqrt(128)
        shifts[flag] = d.mean(dim=-1).abs().mean().item()
    assert shifts[None] < 0.1, shifts
    assert shifts[False] > 1.0, shifts


def test_the_kernel_reads_the_same_keys_as_the_host_path():
    """The fused hand-off carries the anchor, and the kernel's arithmetic applied
    to it reproduces the materialize path -- anchored and not."""
    for flag in (None, False):
        cache, rope = _qwen_cache(key_anchor=flag)
        _seed_and_evict(cache, rope)
        store = cache._stores[0]
        n = store.num_active_windows
        qtier = cache._fused_qtier(0, store, 1, n, 8)
        if flag is None:
            assert torch.equal(qtier["k_anchor"], store.key_anchor_rows())
        else:
            assert qtier["k_anchor"] is None
        host, _, _ = store.effective_q_tier(cache.rope_module, torch.float32)
        kern = _kernel_qtier_keys(qtier)
        assert torch.allclose(kern, host.float(), atol=1e-3, rtol=1e-5), (
            (kern - host.float()).abs().max())
