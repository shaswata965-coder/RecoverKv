"""Cache-level two-tier quantization tests (design.md §5, §8, §10).

Drive WindowedCache's two-tier eviction directly with hand-set state so the
demote / promote / drop / materialize paths are exercised in isolation from the
score-timing machinery. CPU only, B = 1.
"""

from __future__ import annotations

import pytest
import torch

from modules.windowed_cache.cache import WindowedCache
from modules.windowed_cache.config import WindowedCacheConfig
from modules.quant.effective import rotate_key_window, unrotate_key_window


class _FakeModelConfig:
    num_attention_heads = 2
    num_key_value_heads = 2
    hidden_size = 8
    head_dim = 4
    num_hidden_layers = 1


class _RealRoPE(torch.nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        B = position_ids.shape[0]
        inv = self.inv_freq[None, :, None].float().expand(B, -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _make_cache(quant_ratio=0.5, ws=4, num_sink=0, prefill_len=16):
    cfg = WindowedCacheConfig(
        window_size=ws, num_sink_tokens=num_sink, local_window_size=ws,
        cache_budget=0.5, quant_ratio=quant_ratio,
    )
    return WindowedCache(
        config=cfg, prefill_len=prefill_len, model_config=_FakeModelConfig(),
        kv_dtype=torch.float32, rope_module=_RealRoPE(4),
        num_layers=1, max_tokens=8,
    )


def _active(store):
    """Active Q window ids for row 0, as a plain sorted list."""
    ids = store.active_ids()
    return [] if ids is None else ids[0].tolist()


def _codes_of(store, wid, row=0):
    """The stored int2 key codes for a window id (active or dormant)."""
    has, _, slot, _ = store.lookup(torch.tensor([[wid]], dtype=torch.long))
    assert bool(has[0, 0]), f"window {wid} has no slot-table entry"
    return store.table.key_codes[row, slot[0, 0]]


def _seed_prefill_state(cache, n_win=4, ws=4, num_sink=0, H=2, D=4):
    """Populate layer-0 state as if prefill produced n_win full windows."""
    T = num_sink + n_win * ws
    torch.manual_seed(0)
    rope = cache.rope_module
    pos = torch.arange(T, dtype=torch.long)
    k_pre = torch.randn(H, T, D)
    v = torch.randn(H, T, D)
    k_post = rotate_key_window(k_pre, pos, rope)
    # replace(), not attribute assignment: the fp store is preallocated and the
    # buffers are what append/evict actually read.
    cache._states[0].replace(
        k_post.unsqueeze(0).clone(), v.unsqueeze(0).clone(), pos.unsqueeze(0).clone()
    )
    return k_pre, v, k_post, pos


# ---------------------------------------------------------------------------


def test_b_gt_1_with_quant_is_supported():
    """B>1 at q>0 runs. Was NotImplementedError until the tier decision batched."""
    cache = _make_cache(quant_ratio=0.5)
    k = torch.randn(2, 2, 3, 4)  # B=2
    ek, ev = cache.update(k, k.clone(), 0, cache_kwargs={"cache_position": torch.arange(3)})
    assert ek.shape[0] == 2 and ev.shape[0] == 2


def test_first_eviction_demotes_and_materializes():
    """w0→fp (top), w1→Q (next), w2 dropped, w3 local-fp. Effective = [w0,w1,w3]."""
    cache = _make_cache(quant_ratio=0.5, ws=4, num_sink=0)
    H, D, ws = 2, 4, 4
    k_pre, v, k_post, pos = _seed_prefill_state(cache, n_win=4, ws=ws)
    st = cache._states[0]
    st.window_scores = torch.zeros(1, 2, 4)
    st.window_scores[0, :, 0] = 100.0
    st.window_scores[0, :, 1] = 50.0
    st.window_scores[0, :, 2] = 10.0
    st.original_window_ids = torch.tensor([[0, 1, 2, 3]])

    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1

    cache._evict_two_tier(0, step=2)
    store = cache._stores[0]

    # w1 is the sole Q window; w2 dropped; fp store holds w0 + w3.
    assert _active(store) == [1]
    assert st.position_ids[0].tolist() == [0, 1, 2, 3, 12, 13, 14, 15]  # w0 ‖ w3
    assert torch.equal(st.key_states[0][:, :4], k_post[:, 0:4])    # w0 untouched fp
    assert torch.equal(st.key_states[0][:, 4:], k_post[:, 12:16])  # w3 untouched fp

    # Effective K/V is the UNSORTED [body: w0,w3 ‖ Q: w1] layout (num_sink=0).
    eff_k, eff_v, score_meta = cache._materialize(0)
    assert eff_k.shape == (1, H, 12, D)                        # T_total = 8 + 4
    assert cache.get_seq_length(0) == 12
    # fp body windows are byte-identical, at their physical offsets: w0 | w3.
    assert torch.equal(eff_k[0][:, 0:4], k_post[:, 0:4])       # w0
    assert torch.equal(eff_k[0][:, 4:8], k_post[:, 12:16])     # w3
    # Q window w1 follows the body, reconstructed within int2 error.
    assert (eff_k[0][:, 8:12] - k_post[:, 4:8]).abs().max() < 1.0
    # Score-scatter: physical ids [w0,w3 | w1] = [0,3,1]; argsort → [0,2,1]
    # scatters back to ascending merged id [0,1,3].
    order, q_token_len = score_meta
    assert q_token_len == 4
    assert order[0].tolist() == [0, 2, 1]


def test_redemotion_after_promotion_reactivates():
    """Demote w1, promote it (dormant), then re-demote — codes unchanged (§10)."""
    cache = _make_cache(quant_ratio=0.5, ws=4, num_sink=0)
    H, D, ws = 2, 4, 4
    k_pre, v, k_post, pos = _seed_prefill_state(cache, n_win=4, ws=ws)
    st = cache._states[0]
    st.window_scores = torch.zeros(1, 2, 4)
    st.window_scores[0, :, [0, 1, 2]] = torch.tensor([100.0, 50.0, 10.0])
    st.original_window_ids = torch.tensor([[0, 1, 2, 3]])
    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1
    cache._evict_two_tier(0, step=2)
    store = cache._stores[0]
    codes_before = _codes_of(store, 1).clone()

    # Now promote w1 back: fp store = [w0(0-3), w1(4-7), w3(12-15)] merged [0,1,3],
    # rank so w1 stays top → both w0,w1 fp (top_k_fp=2), w3 local.
    st.window_scores = torch.zeros(1, 2, 3)
    st.window_scores[0, :, [0, 1]] = torch.tensor([10.0, 100.0])  # w1 outranks w0
    st.original_window_ids = torch.tensor([[0, 1, 3]])
    pol.top_k_fp, pol.N_q, pol.local_windows = 2, 0, 1
    cache._evict_two_tier(0, step=4)

    # w1 promoted → dormant, fp store now holds it again at positions 4..7.
    assert _active(store) == []
    assert int(store.table.n_live[0]) == 1  # dormant, retained not freed
    assert st.position_ids[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]

    # Re-demote w1 (drop w0, keep w1 in Q): codes must be bit-identical (no requant).
    st.window_scores = torch.zeros(1, 2, 3)
    st.window_scores[0, :, [0, 1]] = torch.tensor([10.0, 100.0])
    st.original_window_ids = torch.tensor([[0, 1, 3]])
    pol.top_k_fp, pol.N_q, pol.local_windows = 0, 1, 1
    cache._evict_two_tier(0, step=6)
    assert _active(store) == [1]
    assert torch.equal(_codes_of(store, 1), codes_before)


def test_q0_store_is_none():
    cache = _make_cache(quant_ratio=0.0)
    assert cache._stores[0] is None
    assert cache._q == 0.0


# ---------------------------------------------------------------------------
# Fused decode kernel glue (design §11 Phase 2) — the CPU-checkable half of the
# int2-in-register decode kernel: everything but the Triton launch itself.
# ---------------------------------------------------------------------------


def test_two_half_rope_matches_primitive():
    """The decode kernel's two-half RoPE reconstructs rotate_key_window exactly.

    The kernel avoids a rotate-half gather by loading the two head-dim halves and
    applying ``k_lo*c − k_hi*s`` / ``k_hi*c + k_lo*s`` with the cos/sin FIRST
    halves. That must equal the read path's ``apply_rotary_pos_emb``
    (rotate_key_window) the fp store and effective_q_tier use — otherwise the
    fused decode attends mis-rotated Q keys.
    """
    from modules.windowed_cache.decode_kernel import rope_cos_sin_halves

    torch.manual_seed(0)
    H, T, D = 2, 6, 4
    rope = _RealRoPE(D)
    k_pre = torch.randn(H, T, D)
    pos = torch.arange(T, dtype=torch.long)

    k_ref = rotate_key_window(k_pre, pos, rope)                 # [H, T, D] full RoPE
    cos_h, sin_h = rope_cos_sin_halves(rope, pos)              # [1, T, D//2]
    half = D // 2
    c, s = cos_h[0], sin_h[0]                                   # [T, half]
    k_lo, k_hi = k_pre[..., :half], k_pre[..., half:]
    k_two_half = torch.cat([k_lo * c - k_hi * s, k_hi * c + k_lo * s], dim=-1)
    assert torch.allclose(k_two_half, k_ref, atol=1e-5)


def test_fused_ctx_dequant_matches_effective_q_tier():
    """The int2 fields the fused path gathers, dequantized + two-half RoPE'd as the
    kernel does, equal ``effective_q_tier`` — so fused decode attends the SAME Q
    keys/values as the materialize path. This proves the whole int2-in-register
    read except the Triton launch (validated on a GPU against the reference).
    """
    from modules.windowed_cache.decode_kernel import rope_cos_sin_halves
    from modules.quant.quantizer import (
        dequantize_key_windows,
        dequantize_value_windows,
    )

    cache = _make_cache(quant_ratio=0.5, ws=4, num_sink=0)
    H, D, ws, B = 2, 4, 4, 1
    _seed_prefill_state(cache, n_win=4, ws=ws)
    st = cache._states[0]
    st.window_scores = torch.zeros(1, 2, 4)
    st.window_scores[0, :, 0] = 100.0
    st.window_scores[0, :, 1] = 50.0
    st.window_scores[0, :, 2] = 10.0
    st.original_window_ids = torch.tensor([[0, 1, 2, 3]])
    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1
    cache._evict_two_tier(0, step=2)
    store = cache._stores[0]
    n = store.num_active_windows
    assert n >= 1

    # Read-path reference: dequant + RoPE via the tested primitives.
    k_ref, v_ref, _ = store.effective_q_tier(cache.rope_module, torch.float32)  # [B,H,n*ws,D]

    # Mirror cache.update()'s fused-branch gather (raw int2, no dequant).
    idx = store.table.active_order(n)
    kc, ks, kz, vc, vs, vz, qpos = store.table.gather(idx)      # flat [B*n, ...]
    qpos_flat = qpos.reshape(B, n * ws)
    cos_h, sin_h = rope_cos_sin_halves(cache.rope_module, qpos_flat)  # [B, n*ws, half]

    # Kernel-style key reconstruction: int2 dequant (pre-RoPE) -> two-half RoPE.
    k_pre = dequantize_key_windows(kc, ks, kz, ws, out_dtype=torch.float32)  # [B*n,H,ws,D]
    k_pre = k_pre.reshape(B, n, H, ws, D).permute(0, 2, 1, 3, 4).reshape(B, H, n * ws, D)
    half = D // 2
    c, s = cos_h[:, None], sin_h[:, None]                       # [B,1,n*ws,half]
    k_lo, k_hi = k_pre[..., :half], k_pre[..., half:]
    k_roped = torch.cat([k_lo * c - k_hi * s, k_hi * c + k_lo * s], dim=-1)
    assert torch.allclose(k_roped, k_ref, atol=1e-4)

    # Values: int2 dequant, no RoPE, window-major -> [B,H,n*ws,D].
    v_pre = dequantize_value_windows(vc, vs, vz, D, out_dtype=torch.float32)  # [B*n,H,ws,D]
    v_deq = v_pre.reshape(B, n, H, ws, D).permute(0, 2, 1, 3, 4).reshape(B, H, n * ws, D)
    assert torch.allclose(v_deq, v_ref, atol=1e-4)


# ---------------------------------------------------------------------------
# Slot table + fp-rebuild invariants (Phase 1)
# ---------------------------------------------------------------------------


class TestReadMemoization:
    """The Q-tier read memo is configurable and OFF by default above B=1.

    It caches the dequantized + RoPE'd Q tier between evictions — correct and
    free at B=1, where decode is weight-bound (BATCHING_PLAN.md §5) and the memo
    saves 7 of every 8 steps' dequant at window_size=8. Above B=1 it is charged
    per row: measured at the qasper steady state it holds ~149 MB/row against
    ~131 MB/row of actual two-tier KV, so it more than doubles per-row memory and
    halves the batch that fits — and batch capacity is the entire thesis.
    """

    def _cache(self, memo, q=0.5):
        cfg = WindowedCacheConfig(
            window_size=4, num_sink_tokens=0, local_window_size=4,
            cache_budget=0.5, quant_ratio=q, quant_memoize_read=memo,
        )
        return WindowedCache(
            config=cfg, prefill_len=16, model_config=_FakeModelConfig(),
            kv_dtype=torch.float32, rope_module=_RealRoPE(4),
            num_layers=1, max_tokens=8,
        )

    def test_auto_default_is_on_at_b1_off_above(self):
        c = self._cache(None)
        c._resolve_memoization(1)
        assert c._stores[0].memoize_read is True
        c = self._cache(None)
        c._resolve_memoization(4)
        assert c._stores[0].memoize_read is False, (
            "the memo must default OFF at B>1 — it costs ~149 MB/row and halves "
            "max batch, which is what the method exists to raise"
        )

    def test_explicit_setting_overrides_the_batch_heuristic(self):
        c = self._cache(True)
        c._resolve_memoization(8)
        assert c._stores[0].memoize_read is True
        c = self._cache(False)
        c._resolve_memoization(1)
        assert c._stores[0].memoize_read is False

    def test_memo_is_a_pure_optimization(self):
        """Memo on vs off must be byte-identical — it caches, it never decides.

        Codes and grids are written once at first demotion and position_range is
        never rebased (§10), so the dequant + RoPE cannot move between evictions.
        If these ever diverged, the memo would be serving stale reads.
        """
        ws, H, D, prefill = 4, 2, 4, 16
        outs = []
        for memo in (True, False):
            c = self._cache(memo)
            c._policies[0].top_k_fp, c._policies[0].N_q, c._policies[0].local_windows = 2, 2, 1
            torch.manual_seed(5)
            kp = torch.randn(1, H, prefill, D)
            got = []
            ek, ev = c.update(kp, kp.clone(), 0, cache_kwargs={
                "cache_position": torch.arange(prefill),
                "window_scores": torch.rand(1, H, _merged_W(c, ws, 0, prefill)),
            })
            got.append((ek.clone(), ev.clone()))
            for t in range(prefill, prefill + 14):
                k1 = torch.randn(1, H, 1, D)
                W = _merged_W(c, ws, 0, c._states[0].seq_length + 1)
                ek, ev = c.update(k1, k1.clone(), 0, cache_kwargs={
                    "cache_position": torch.arange(t, t + 1),
                    "window_scores": torch.rand(1, H, W),
                })
                got.append((ek.clone(), ev.clone()))
            outs.append(got)

        assert len(outs[0]) == len(outs[1])
        for i, ((ka, va), (kb, vb)) in enumerate(zip(*outs)):
            assert torch.equal(ka, kb), f"memo changed effective K at step {i}"
            assert torch.equal(va, vb), f"memo changed effective V at step {i}"

    def test_memo_off_holds_nothing_between_reads(self):
        c = self._cache(False)
        c._policies[0].top_k_fp, c._policies[0].N_q, c._policies[0].local_windows = 1, 1, 1
        _seed_prefill_state(c, n_win=4, ws=4)
        st = c._states[0]
        st.window_scores = torch.zeros(1, 2, 4)
        st.window_scores[0, :, [0, 1, 2]] = torch.tensor([100.0, 50.0, 10.0])
        st.original_window_ids = torch.tensor([[0, 1, 2, 3]])
        c._evict_two_tier(0, step=2)
        c._materialize(0)
        assert c._stores[0]._read_cache is None, "memo=False still retained a tier"


def _windows_of(positions, num_sink, ws):
    """Group absolute positions into {window_id: sorted positions}."""
    out = {}
    for p in positions:
        if p < num_sink:
            continue
        out.setdefault((p - num_sink) // ws, []).append(p)
    return out


def test_eviction_preserves_whole_windows():
    """Every retained window survives WHOLE, in both tiers, with no dup or loss.

    This is the real check on the fp rebuild. Its new-body length is pure config
    arithmetic (``k_fp*ws + local_tokens``) rather than a mask count — no host
    sync — which is only correct if the fp store really is
    ``[sink ‖ full windows by id ‖ possibly-partial newest]``. If that assumption
    ever breaks, the rebuild silently slices at the wrong offsets and windows
    come out shredded — so assert on window integrity, not on the length.
    """
    ws, num_sink, H, D = 4, 0, 2, 4
    prefill = 16
    cache = _make_cache(quant_ratio=0.5, ws=ws, num_sink=num_sink, prefill_len=prefill)
    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 2, 2, 1
    store = cache._stores[0]

    torch.manual_seed(7)
    kp = torch.randn(1, H, prefill, D)
    cache.update(kp, kp.clone(), 0, cache_kwargs={
        "cache_position": torch.arange(prefill),
        "window_scores": torch.rand(1, H, _merged_W(cache, ws, num_sink, prefill)),
    })

    saw_q = False
    for t in range(prefill, prefill + 20):
        k1 = torch.randn(1, H, 1, D)
        W = _merged_W(cache, ws, num_sink, cache._states[0].seq_length + 1)
        cache.update(k1, k1.clone(), 0, cache_kwargs={
            "cache_position": torch.arange(t, t + 1),
            "window_scores": torch.rand(1, H, W),
        })
        st = cache._states[0]
        fp_pos = st.position_ids[0].tolist()
        assert fp_pos == sorted(fp_pos), "fp store must stay chronological"
        assert len(set(fp_pos)) == len(fp_pos), "fp store duplicated a token"

        q_pos = []
        if store.num_active_windows:
            saw_q = True
            order = store.table.active_order(store.num_active_windows)
            q_pos = store.table.slot_pos[0][order[0]].reshape(-1).tolist()

        assert not (set(fp_pos) & set(q_pos)), "a token is in BOTH tiers"

        # A window lives wholly in one tier and is never split or truncated —
        # except the newest, which is still filling up.
        newest = max(_windows_of(fp_pos + q_pos, num_sink, ws), default=-1)
        for wid, ps in _windows_of(fp_pos, num_sink, ws).items():
            if wid != newest:
                assert len(ps) == ws, f"fp window {wid} has {len(ps)}/{ws} tokens"
        for wid, ps in _windows_of(q_pos, num_sink, ws).items():
            assert len(ps) == ws, f"Q window {wid} has {len(ps)}/{ws} tokens"

        store.validate()

    assert saw_q, "Q tier never populated — demotion path not exercised"


def test_slot_table_stays_within_its_bound():
    """Live entries never exceed top_k_fp + N_q — the bound n_slots_for relies on.

    If this ever failed, free_slots() would hand out an occupied slot and a live
    window's codes would be silently overwritten, so it is worth asserting
    rather than trusting the derivation.
    """
    ws, num_sink, H, D = 4, 0, 2, 4
    prefill = 16
    cache = _make_cache(quant_ratio=0.5, ws=ws, num_sink=num_sink, prefill_len=prefill)
    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 2, 2, 1
    store = cache._stores[0]
    bound = pol.top_k_fp + pol.N_q

    torch.manual_seed(11)
    kp = torch.randn(1, H, prefill, D)
    cache.update(kp, kp.clone(), 0, cache_kwargs={
        "cache_position": torch.arange(prefill),
        "window_scores": torch.rand(1, H, _merged_W(cache, ws, num_sink, prefill)),
    })
    for t in range(prefill, prefill + 24):
        k1 = torch.randn(1, H, 1, D)
        W = _merged_W(cache, ws, num_sink, cache._states[0].seq_length + 1)
        cache.update(k1, k1.clone(), 0, cache_kwargs={
            "cache_position": torch.arange(t, t + 1),
            "window_scores": torch.rand(1, H, W),
        })
        if store.table is not None:
            assert int(store.table.n_live.max()) <= bound, (
                f"live entries exceeded top_k_fp + N_q = {bound}"
            )
            assert store.table.n_slots >= bound
            store.validate()


# ---------------------------------------------------------------------------
# End-to-end through update() (prefill + generation with real evictions)
# ---------------------------------------------------------------------------


def _merged_W(cache, ws, num_sink, after_append_tfp):
    wq = cache._stores[0].num_active_windows if cache._stores[0] else 0
    body = max(after_append_tfp - num_sink, 0)
    return (body + ws - 1) // ws + wq


def test_end_to_end_generation_with_quant():
    """Full prefill + many decode steps at q>0. Asserts the two-tier machine
    stays self-consistent every step: T_fp + T_q == get_seq_length, effective
    K/V matches that length, positions valid, budget bounded, Q tier used."""
    ws, num_sink, H, D = 4, 0, 2, 4
    prefill = 16
    cache = _make_cache(quant_ratio=0.5, ws=ws, num_sink=num_sink, prefill_len=prefill)
    # Controlled budget: keep 2 fp + 2 Q evictable windows + 1 local.
    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 2, 2, 1

    torch.manual_seed(1)
    kdt = torch.float32

    def scores(W):
        s = torch.rand(1, H, W)
        return s

    # Prefill
    kp = torch.randn(1, H, prefill, D, dtype=kdt)
    vp = torch.randn(1, H, prefill, D, dtype=kdt)
    W0 = _merged_W(cache, ws, num_sink, prefill)
    ek, ev = cache.update(kp, vp, 0, cache_kwargs={
        "cache_position": torch.arange(prefill), "window_scores": scores(W0),
    })
    assert ek.shape[2] == cache.get_seq_length(0)

    used_q = False
    for t in range(prefill, prefill + 16):
        k1 = torch.randn(1, H, 1, D, dtype=kdt)
        v1 = torch.randn(1, H, 1, D, dtype=kdt)
        tfp_after = cache._states[0].seq_length + 1
        W = _merged_W(cache, ws, num_sink, tfp_after)
        ek, ev = cache.update(k1, v1, 0, cache_kwargs={
            "cache_position": torch.arange(t, t + 1), "window_scores": scores(W),
        })
        store = cache._stores[0]
        tfp = cache._states[0].seq_length
        tq = store.num_active_tokens
        # invariants
        assert cache.get_seq_length(0) == tfp + tq
        assert ek.shape == (1, H, tfp + tq, D)
        assert ev.shape == (1, H, tfp + tq, D)
        # fp positions strictly increasing (no renumber, no dup)
        p = cache._states[0].position_ids[0]
        assert torch.all(p[1:] > p[:-1])
        step = t - prefill
        if step < pol.first_eviction_step:
            # Nothing evicts before the fixed first-eviction step: the whole
            # prompt stays resident and the cache grows one fp token per step.
            assert tq == 0 and tfp == prefill + step + 1
        else:
            # From the first eviction on, the two-tier budget bounds the cache:
            # post-eviction fp is (top_k_fp + local)*ws, plus up to ws growth
            # between ws-spaced evictions, plus the Q tier (N_q*ws) and a window
            # of slack.
            cap = num_sink + (pol.top_k_fp + pol.N_q + pol.local_windows + 2) * ws
            assert tfp + tq <= cap
        assert store.num_active_windows <= pol.N_q
        if store.num_active_windows > 0:
            used_q = True

    # Compaction actually happened (retained ≪ uncompacted prefill+steps).
    assert cache.get_seq_length(0) < prefill + 16
    assert used_q, "Q tier was never populated — demotion path not exercised"


# ===========================================================================
# B > 1 at q > 0 (BATCHING_PLAN.md §3, §4 Phase 2)
# ===========================================================================


def _batched_cache(B, ws=4, num_sink=0, prefill=16, q=0.5, memo=None):
    cfg = WindowedCacheConfig(
        window_size=ws, num_sink_tokens=num_sink, local_window_size=ws,
        cache_budget=0.5, quant_ratio=q, quant_memoize_read=memo,
    )
    c = WindowedCache(
        config=cfg, prefill_len=prefill, model_config=_FakeModelConfig(),
        kv_dtype=torch.float32, rope_module=_RealRoPE(4),
        num_layers=1, max_tokens=16,
    )
    c._policies[0].top_k_fp, c._policies[0].N_q, c._policies[0].local_windows = 2, 2, 1
    return c


def _drive(cache, kp, vp, kd, vd, scores, ws=4, num_sink=0):
    """Prefill + decode a cache with pre-generated tensors. Returns per-step outs."""
    B, H, prefill, D = kp.shape
    outs = []
    W = _merged_W(cache, ws, num_sink, prefill)
    ek, ev = cache.update(kp, vp, 0, cache_kwargs={
        "cache_position": torch.arange(prefill),
        "window_scores": scores[0][:, :, :W].clone(),
    })
    outs.append((ek.clone(), ev.clone()))
    for i in range(kd.shape[0]):
        t = prefill + i
        W = _merged_W(cache, ws, num_sink, cache._states[0].seq_length + 1)
        ek, ev = cache.update(kd[i], vd[i], 0, cache_kwargs={
            "cache_position": torch.arange(t, t + 1),
            "window_scores": scores[i + 1][:, :, :W].clone(),
        })
        outs.append((ek.clone(), ev.clone()))
    return outs


def _rows_diverge_scores(B, H, steps, prefill_W=4, maxW=32):
    """Scores that force each row to rank the evictable band DIFFERENTLY.

    Row r pins its own windows to the top, so rows retain different sets — the
    case a host-side dict keyed by window id cannot express at all. Drawn from a
    fixed-size pool so the number of random values never depends on W.
    """
    g = torch.Generator().manual_seed(31)
    s = torch.rand(steps + 1, B, H, maxW, generator=g)
    for r in range(B):
        s[:, r, :, r % prefill_W] += 100.0
        s[:, r, :, (r + 2) % prefill_W] += 50.0
    return s


def test_b_gt_1_rows_evict_divergently():
    """Rows retain DIFFERENT window sets in the Q tier, simultaneously."""
    B, H, D, ws, prefill = 4, 2, 4, 4, 16
    cache = _batched_cache(B, ws=ws, prefill=prefill)
    g = torch.Generator().manual_seed(3)
    kp = torch.randn(B, H, prefill, D, generator=g)
    kd = torch.randn(12, B, H, 1, D, generator=g)
    sc = _rows_diverge_scores(B, H, 12)
    _drive(cache, kp, kp.clone(), kd, kd.clone(), sc, ws=ws)

    store = cache._stores[0]
    store.validate()
    ids = store.active_ids()
    assert ids.shape[0] == B
    per_row = [tuple(ids[r].tolist()) for r in range(B)]
    assert len(set(per_row)) > 1, (
        f"rows did not diverge ({per_row}) — the test is not exercising the "
        f"case a host-side dict ledger could not express"
    )
    # Rectangular: same COUNT per row, different WINDOWS (BATCHING_PLAN.md §3).
    assert len({len(p) for p in per_row}) == 1


def test_b_gt_1_row_equivalence_vs_solo_runs():
    """Row i of a batched run == a solo run of that same sequence.

    The property that makes batching trustworthy: rows must not interact. Any
    cross-row leak — a shared slot, a mask reduced over the wrong axis, a count
    taken from row 0 — breaks this and almost nothing else.
    """
    B, H, D, ws, prefill, steps = 4, 2, 4, 4, 16, 12
    g = torch.Generator().manual_seed(17)
    kp = torch.randn(B, H, prefill, D, generator=g)
    vp = torch.randn(B, H, prefill, D, generator=g)
    kd = torch.randn(steps, B, H, 1, D, generator=g)
    vd = torch.randn(steps, B, H, 1, D, generator=g)
    sc = _rows_diverge_scores(B, H, steps)

    batched = _drive(_batched_cache(B, ws=ws, prefill=prefill),
                     kp, vp, kd, vd, sc, ws=ws)

    for r in range(B):
        solo = _drive(
            _batched_cache(1, ws=ws, prefill=prefill),
            kp[r:r + 1], vp[r:r + 1], kd[:, r:r + 1], vd[:, r:r + 1],
            sc[:, r:r + 1], ws=ws,
        )
        assert len(solo) == len(batched)
        for i, ((bk, bv), (sk, sv)) in enumerate(zip(batched, solo)):
            assert torch.equal(bk[r:r + 1], sk), (
                f"row {r} step {i}: batched effective K != solo run"
            )
            assert torch.equal(bv[r:r + 1], sv), (
                f"row {r} step {i}: batched effective V != solo run"
            )


def _seed_rows(cache, B, H=2, D=4, seed=23, ws=4, n_win=4):
    """Seed layer-0 with B rows of n_win full windows at positions 0..n_win*ws-1."""
    torch.manual_seed(seed)
    st = cache._states[0]
    T = n_win * ws
    pos = torch.arange(T, dtype=torch.long)
    k_pre = torch.randn(B, H, T, D)
    k_post = rotate_key_window(k_pre, pos.unsqueeze(0).expand(B, -1), cache.rope_module)
    st.replace(k_post, torch.randn(B, H, T, D),
               pos.unsqueeze(0).expand(B, -1).contiguous())
    st.original_window_ids = torch.arange(n_win).unsqueeze(0).expand(B, -1).contiguous()
    return st


def test_b_gt_1_ragged_demote_counts_allocate_correctly():
    """Rows demote DIFFERENT numbers of windows in ONE eviction.

    §3's rectangularity covers the RETAINED counts; the transition counts are
    ragged, because `demote = is_q_new & ~is_q_cur` depends on where each row's Q
    tier already sits. Slots are therefore allocated by rank into a worst-case
    tensor rather than by count, and this asserts the ragged case is real and
    that no row's allocation clobbers another's.
    """
    B, H, D, ws = 3, 2, 4, 4
    cache = _batched_cache(B, ws=ws, prefill=16)
    st = _seed_rows(cache, B)
    store = cache._stores[0]
    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1

    # Eviction 1: each row sends a DIFFERENT window to the Q tier.
    st.window_scores = torch.zeros(B, H, 4)
    st.window_scores[0, :, [0, 1]] = torch.tensor([100.0, 50.0])   # row 0: Q <- w1
    st.window_scores[1, :, [1, 0]] = torch.tensor([100.0, 50.0])   # row 1: Q <- w0
    st.window_scores[2, :, [0, 2]] = torch.tensor([100.0, 50.0])   # row 2: Q <- w2
    cache._evict_two_tier(0, step=2)
    store.validate()
    first = [store.active_ids()[r].tolist() for r in range(B)]
    assert first == [[1], [0], [2]], f"rows did not diverge: {first}"
    assert int(store.table.n_live.max()) == 1

    # Eviction 2 on the merged axis. Row 0 keeps its existing Q window (0 fresh
    # demotions); the other rows send one they have never quantized (1 fresh
    # each) -> ragged fresh-demote counts within a single pass.
    owids = st.original_window_ids
    st.window_scores = torch.zeros(B, H, owids.shape[1])
    st.window_scores[:, :, 0] = 100.0
    cache._evict_two_tier(0, step=4)
    store.validate()
    assert int(store.table.n_live.max()) <= pol.top_k_fp + pol.N_q, \
        "live entries exceeded the slot bound"
    assert int(store.table.n_live.min()) >= 1, \
        "a row lost its entry to another row's allocation"


def test_b_gt_1_dormant_survives_promote_then_redemote():
    """Codes written exactly once per (row, window_id), across B>1 churn.

    design.md §10: a re-demotion REACTIVATES; it never re-quantizes (re-quantizing
    would pick up fp16 rounding from the promote-side dequant and could flip
    boundary codes). Asserted on the codes tensor itself, per row.
    """
    B, H, D, ws = 3, 2, 4, 4
    cache = _batched_cache(B, ws=ws, prefill=16)
    st = _seed_rows(cache, B, seed=29)
    store = cache._stores[0]
    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1

    # Demote w1 into the Q tier on every row.
    st.window_scores = torch.zeros(B, H, 4)
    st.window_scores[:, :, 0] = 100.0
    st.window_scores[:, :, 1] = 50.0
    st.window_scores[:, :, 2] = 10.0
    cache._evict_two_tier(0, step=2)
    assert [store.active_ids()[r].tolist() for r in range(B)] == [[1]] * B

    def codes_for(wid):
        has, _, slot, _ = store.lookup(torch.full((B, 1), wid, dtype=torch.long))
        assert bool(has.all())
        return torch.stack([store.table.key_codes[r, slot[r, 0]] for r in range(B)])

    codes0 = codes_for(1).clone()

    # Promote w1 back to fp on every row -> entries go dormant, not freed.
    st.window_scores = torch.zeros(B, H, 3)
    st.window_scores[:, :, 0] = 10.0
    st.window_scores[:, :, 1] = 100.0
    st.original_window_ids = torch.tensor([[0, 1, 3]] * B)
    pol.top_k_fp, pol.N_q = 2, 0
    cache._evict_two_tier(0, step=4)
    assert [store.active_ids()[r].tolist() for r in range(B)] == [[]] * B
    assert int(store.table.n_live.min()) == 1, "dormant entry was freed"
    assert torch.equal(codes_for(1), codes0), "promotion mutated the codes"

    # Re-demote w1 -> must REACTIVATE the dormant slot, bit-for-bit.
    st.window_scores = torch.zeros(B, H, 3)
    st.window_scores[:, :, 0] = 10.0
    st.window_scores[:, :, 1] = 100.0
    st.original_window_ids = torch.tensor([[0, 1, 3]] * B)
    pol.top_k_fp, pol.N_q = 0, 1
    cache._evict_two_tier(0, step=6)
    assert [store.active_ids()[r].tolist() for r in range(B)] == [[1]] * B
    assert torch.equal(codes_for(1), codes0), (
        "re-demotion re-quantized instead of reactivating (design.md §10)"
    )
    store.validate()


@pytest.mark.parametrize("B", [2, 3, 4, 8])
def test_b_gt_1_end_to_end_invariants(B):
    """The two-tier machine stays self-consistent per row at every batch size."""
    H, D, ws, prefill, steps = 2, 4, 4, 16, 14
    cache = _batched_cache(B, ws=ws, prefill=prefill)
    store = cache._stores[0]
    g = torch.Generator().manual_seed(41 + B)
    kp = torch.randn(B, H, prefill, D, generator=g)
    kd = torch.randn(steps, B, H, 1, D, generator=g)
    sc = _rows_diverge_scores(B, H, steps)
    outs = _drive(cache, kp, kp.clone(), kd, kd.clone(), sc, ws=ws)

    for ek, ev in outs:
        assert ek.shape[0] == B and ev.shape[0] == B
        assert ek.shape == ev.shape
    # get_seq_length reports the CURRENT length, so only the last output can be
    # checked against it — it is what HF sizes the causal mask from.
    assert outs[-1][0].shape[2] == cache.get_seq_length(0)
    st = cache._states[0]
    assert st.key_states.shape[0] == B
    for r in range(B):
        p = st.position_ids[r]
        assert torch.all(p[1:] > p[:-1]), f"row {r} positions not strictly increasing"
    store.validate()
    assert store.num_active_windows <= cache._policies[0].N_q


# ---------------------------------------------------------------------------
# Flash / eager backend parity (shared cache.py ⇒ identical two-tier results)
# ---------------------------------------------------------------------------

from modules.windowed_eager_cache.cache import WindowedCache as EagerWindowedCache
from modules.windowed_eager_cache.config import WindowedCacheConfig as EagerCfg


@pytest.mark.parametrize("B", [1, 4])
def test_flash_eager_two_tier_parity(B):
    """The twins share cache.py, so their two-tier results must be identical —
    at B>1 as well as B=1, since batching is a cache.py-level change."""
    ws, num_sink, H, D, prefill = 4, 0, 2, 4, 16

    def build(EWC, ECfg):
        cfg = ECfg(window_size=ws, num_sink_tokens=num_sink, local_window_size=ws,
                   cache_budget=0.5, quant_ratio=0.5)
        c = EWC(config=cfg, prefill_len=prefill, model_config=_FakeModelConfig(),
                kv_dtype=torch.float32, rope_module=_RealRoPE(4), num_layers=1, max_tokens=8)
        c._policies[0].top_k_fp, c._policies[0].N_q, c._policies[0].local_windows = 2, 2, 1
        return c

    flash = build(WindowedCache, WindowedCacheConfig)
    eager = build(EagerWindowedCache, EagerCfg)

    torch.manual_seed(3)
    kp = torch.randn(B, H, prefill, D); vp = torch.randn(B, H, prefill, D)
    sc0 = _rows_diverge_scores(B, H, 12)[0][:, :, :_merged_W(flash, ws, num_sink, prefill)]
    fk, _ = flash.update(kp.clone(), vp.clone(), 0,
                         cache_kwargs={"cache_position": torch.arange(prefill), "window_scores": sc0.clone()})
    ek, _ = eager.update(kp.clone(), vp.clone(), 0,
                         cache_kwargs={"cache_position": torch.arange(prefill), "window_scores": sc0.clone()})
    assert torch.equal(fk, ek)

    pool = _rows_diverge_scores(B, H, 12)
    for i, t in enumerate(range(prefill, prefill + 12)):
        k1 = torch.randn(B, H, 1, D); v1 = torch.randn(B, H, 1, D)
        Wf = _merged_W(flash, ws, num_sink, flash._states[0].seq_length + 1)
        sc = pool[i + 1][:, :, :Wf]
        fk, fv = flash.update(k1.clone(), v1.clone(), 0,
                              cache_kwargs={"cache_position": torch.arange(t, t + 1), "window_scores": sc.clone()})
        ek, ev = eager.update(k1.clone(), v1.clone(), 0,
                              cache_kwargs={"cache_position": torch.arange(t, t + 1), "window_scores": sc.clone()})
        assert torch.equal(fk, ek) and torch.equal(fv, ev)
        fa, ea = flash._stores[0].active_ids(), eager._stores[0].active_ids()
        assert (fa is None and ea is None) or torch.equal(fa, ea)
        assert flash.get_seq_length(0) == eager.get_seq_length(0)


# ---------------------------------------------------------------------------
# Fused hand-off memoization — the per-layer/per-step launch-overhead fix.
# The claim under test is EQUIVALENCE, not speed: what the memo hands back must
# be what a recompute would have produced, tensor for tensor.
# ---------------------------------------------------------------------------


def _fresh_fused_ctx(cache, layer_idx=0):
    """Rebuild the fused hand-off from scratch — the pre-memo code path, verbatim."""
    from modules.windowed_cache.decode_kernel import rope_cos_sin_halves
    from modules.quant.effective import compute_score_meta

    state = cache._states[layer_idx]
    store = cache._stores[layer_idx]
    n = store.num_active_windows
    B = state.key_states.shape[0]
    ws = cache.resolved.window_size
    idx = store.table.active_order(n)
    kc, ks, kz, vc, vs, vz, qpos = store.table.gather(idx)
    qtier = {
        "k_codes": kc.reshape(B, n, *kc.shape[1:]),
        "k_scale": ks.reshape(B, n, *ks.shape[1:]),
        "k_zero": kz.reshape(B, n, *kz.shape[1:]),
        "v_codes": vc.reshape(B, n, *vc.shape[1:]),
        "v_scale": vs.reshape(B, n, *vs.shape[1:]),
        "v_zero": vz.reshape(B, n, *vz.shape[1:]),
    }
    qpos_flat = qpos.reshape(B, n * ws)
    qtier["cos"], qtier["sin"] = rope_cos_sin_halves(cache.rope_module, qpos_flat)
    meta = compute_score_meta(
        state.position_ids, qpos_flat, cache.resolved.num_sink_tokens, ws
    )
    return qtier, meta


def _seeded_fused_cache(ws=4):
    """A layer-0 cache past its first two-tier eviction, armed for fused decode."""
    cache = _make_cache(quant_ratio=0.5, ws=ws, num_sink=0)
    _seed_prefill_state(cache, n_win=4, ws=ws)
    st = cache._states[0]
    st.window_scores = torch.zeros(1, 2, 4)
    st.window_scores[0, :, 0] = 100.0
    st.window_scores[0, :, 1] = 50.0
    st.window_scores[0, :, 2] = 10.0
    st.original_window_ids = torch.tensor([[0, 1, 2, 3]])
    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1
    cache._evict_two_tier(0, step=2)
    cache._prefill_done[0] = True
    cache._fused_decode_active = True
    return cache


def _drive_decode_step(cache, pos):
    """One decode update(); returns the ctx the fused path armed (and disarms it)."""
    from modules.windowed_cache import flash_decode

    k = torch.randn(1, 2, 1, 4)
    cache.update(k, k.clone(), 0,
                 cache_kwargs={"cache_position": torch.tensor([pos])})
    ctx = flash_decode._PENDING["ctx"]
    # Stand in for the wrapper consuming it (the Triton launch is GPU-only).
    flash_decode._PENDING["ctx"] = None
    return ctx


def test_memoized_fused_ctx_equals_a_fresh_rebuild_every_step():
    """Every memoized hand-off is tensor-for-tensor what a rebuild would produce.

    This is the whole safety argument for hoisting the gather / RoPE / argsort out
    of the per-layer-per-step path: design §10 freezes Q entries between evictions,
    so the memo can only return the same values — never merely equivalent ones.
    """
    cache = _seeded_fused_cache(ws=4)
    compared = 0
    for step, pos in enumerate(range(16, 21)):
        ctx = _drive_decode_step(cache, pos)
        if ctx is None:                       # Q tier emptied — nothing to compare
            continue
        compared += 1
        want_q, want_meta = _fresh_fused_ctx(cache)
        for field, got in ctx["qtier"].items():
            if field == "window_size":
                continue
            assert torch.equal(got, want_q[field]), f"step {step}: {field} diverged"
        assert torch.equal(ctx["score_meta"][0], want_meta[0]), f"step {step}: order"
        assert ctx["score_meta"][1] == want_meta[1], f"step {step}: q_token_len"
    # Guard against the loop quietly going vacuous if the fixture drifts.
    assert compared >= 4, f"only {compared} fused steps exercised"


def test_fused_ctx_is_reused_between_evictions_and_dropped_by_one():
    """The memo actually hits (same object) and an eviction actually drops it.

    Without the first half the fix does nothing; without the second it would serve
    a stale Q tier after the active set moves.
    """
    cache = _seeded_fused_cache(ws=4)
    first = _drive_decode_step(cache, 16)
    assert first is not None
    second = _drive_decode_step(cache, 17)
    # Same epoch -> the identical dict, not a rebuilt copy.
    assert second["qtier"] is first["qtier"]

    cache._evict_two_tier(0, step=4)
    after = _drive_decode_step(cache, 18)
    if after is not None:
        assert after["qtier"] is not first["qtier"]
        want_q, _ = _fresh_fused_ctx(cache)
        assert torch.equal(after["qtier"]["k_codes"], want_q["k_codes"])


def test_store_version_bump_invalidates_the_memo():
    """Any Q-store mutation invalidates, not just the eviction entry point."""
    cache = _seeded_fused_cache(ws=4)
    first = _drive_decode_step(cache, 16)
    assert first is not None
    cache._stores[0]._invalidate()            # what every store mutator calls
    again = _drive_decode_step(cache, 17)
    assert again["qtier"] is not first["qtier"]
    want_q, _ = _fresh_fused_ctx(cache)
    assert torch.equal(again["qtier"]["k_codes"], want_q["k_codes"])


# ---------------------------------------------------------------------------
# Triton-or-error, proven at RUNTIME (not just at hook install).
# ---------------------------------------------------------------------------


class TestFusedDecodeProofOfExecution:
    """Arming the kernel and never running it is a hard error, never a silent skip.

    ``assert_decode_kernel_available`` only proves triton *imports* on a CUDA box.
    It cannot see whether transformers actually routed through the patched
    ``flash_attn_func`` — and the varlen path silently does not, dropping the Q
    tier from attention and writing no eviction scores. These lock the gap shut.
    """

    def setup_method(self):
        from modules.windowed_cache import flash_decode
        flash_decode._PENDING["ctx"] = None
        flash_decode.reset_stats()

    def teardown_method(self):
        from modules.windowed_cache import flash_decode
        flash_decode._PENDING["ctx"] = None
        flash_decode.reset_stats()

    def test_clear_raises_when_the_kernel_never_ran(self):
        from modules.windowed_cache import flash_decode

        flash_decode.set_pending({"layer_idx": 7})
        with pytest.raises(flash_decode.FusedDecodeNotReached) as e:
            flash_decode.clear()          # next layer's pre-hook
        assert "layer 7" in str(e.value)
        assert "varlen" in str(e.value)   # the actual cause, named

    def test_arming_twice_without_a_run_raises(self):
        from modules.windowed_cache import flash_decode

        flash_decode.set_pending({"layer_idx": 0})
        with pytest.raises(flash_decode.FusedDecodeNotReached):
            flash_decode.set_pending({"layer_idx": 1})

    def test_a_consumed_hand_off_is_silent_and_counted(self):
        from modules.windowed_cache import flash_decode

        for lidx in range(3):
            flash_decode.clear()                       # pre-hook: nothing pending
            flash_decode.set_pending({"layer_idx": lidx})
            flash_decode._PENDING["ctx"] = None        # wrapper consumes
            flash_decode._STATS["fired"] += 1
        flash_decode.clear()
        assert flash_decode.stats() == {"armed": 3, "fired": 3}

    def test_wrapper_passes_through_and_counts_only_fused_calls(self):
        from modules.windowed_cache import flash_decode

        calls = []
        wrapper = flash_decode._make_wrapper(lambda *a, **k: calls.append(a) or "orig")
        assert wrapper(1, 2, 3) == "orig"              # no ctx -> original flash
        assert flash_decode.stats()["fired"] == 0
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Optional compiled eviction (STICKYKV_COMPILE_EVICT) — semantics preserved.
# torch.compile is semantics-preserving by construction; this pins it against
# regressions in how the body is factored. Uses the aot_eager backend so it
# traces + runs without a C++/Triton compiler (Inductor codegen is the GPU
# concern, exercised there, not the tracing/semantics this asserts).
# ---------------------------------------------------------------------------


def _run_decode_across_eviction(compile_backend=None, seed=1234):
    """Drive one cache across a full eviction cycle; return its final state."""
    import os
    from modules.windowed_cache import cache as cache_mod
    from modules.windowed_cache import flash_decode

    prev = os.environ.get("STICKYKV_COMPILE_EVICT")
    prev_b = os.environ.get("STICKYKV_COMPILE_EVICT_BACKEND")
    if compile_backend is not None:
        os.environ["STICKYKV_COMPILE_EVICT"] = "1"
        os.environ["STICKYKV_COMPILE_EVICT_BACKEND"] = compile_backend
    else:
        os.environ["STICKYKV_COMPILE_EVICT"] = "0"
    # Reset the process-global compiled fn + once-banner so the flag is honoured.
    cache_mod._COMPILED_EVICT_FN = None
    cache_mod._EVICT_ANNOUNCED["done"] = True   # silence the banner in tests
    try:
        cache = _make_cache(quant_ratio=0.5, ws=8, num_sink=0, prefill_len=256)
        H, D, ws = 2, 4, 8
        g = torch.Generator().manual_seed(seed)
        T = 256
        pos = torch.arange(T, dtype=torch.long)
        k_pre = torch.randn(H, T, D, generator=g)
        v = torch.randn(H, T, D, generator=g)
        k_post = rotate_key_window(k_pre, pos, cache.rope_module)
        cache._states[0].replace(
            k_post.unsqueeze(0).clone(), v.unsqueeze(0).clone(), pos.unsqueeze(0).clone()
        )
        st = cache._states[0]
        nwin = T // ws
        st.window_scores = torch.rand(1, H, nwin, generator=g) * 100
        st.original_window_ids = torch.arange(nwin).unsqueeze(0)
        p = cache._policies[0]
        p.top_k_fp, p.N_q, p.local_windows = 2, 2, 1
        cache._prefill_done[0] = True
        cache._fused_decode_active = True
        g2 = torch.Generator().manual_seed(seed + 1)
        for i in range(1, ws + 2):
            kk = torch.randn(1, H, 1, D, generator=g2)
            cache.update(kk, kk.clone(), 0,
                         cache_kwargs={"cache_position": torch.tensor([T + i])})
            flash_decode._PENDING["ctx"] = None
        stt, store = cache._states[0], cache._stores[0]
        return {
            "key": stt.key_states.clone(), "val": stt.value_states.clone(),
            "pos": stt.position_ids.clone(), "wscore": stt.window_scores.clone(),
            "wids": stt.original_window_ids.clone(),
            "n": store.num_active_windows,
            "slot_wid": store.table.slot_wid.clone(),
            "key_codes": store.table.key_codes.clone(),
        }
    finally:
        cache_mod._COMPILED_EVICT_FN = None
        if prev is None:
            os.environ.pop("STICKYKV_COMPILE_EVICT", None)
        else:
            os.environ["STICKYKV_COMPILE_EVICT"] = prev
        if prev_b is None:
            os.environ.pop("STICKYKV_COMPILE_EVICT_BACKEND", None)
        else:
            os.environ["STICKYKV_COMPILE_EVICT_BACKEND"] = prev_b


def test_compiled_eviction_is_byte_identical_to_eager():
    """STICKYKV_COMPILE_EVICT must not change the eviction outcome.

    Runs the same decode sequence across an eviction twice -- eager, then
    torch.compile'd (aot_eager) -- and asserts every stored tensor matches. The
    compiled path is a launch-count optimisation only; the retained windows,
    tier moves, quantized codes, and rebuilt fp store are identical.
    """
    pytest.importorskip("torch")
    eager = _run_decode_across_eviction(compile_backend=None)
    try:
        comp = _run_decode_across_eviction(compile_backend="aot_eager")
    except Exception as e:  # torch.compile unavailable on this build
        pytest.skip(f"torch.compile(aot_eager) unavailable: {type(e).__name__}: {e}")
    for k in eager:
        a, b = eager[k], comp[k]
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), f"{k} diverged under compiled eviction"
        else:
            assert a == b, f"{k} diverged under compiled eviction"


def test_evict_path_stats_split_eager_from_compiled():
    """``evict_path_stats`` must say which body ran, not which flag was set.

    Setting ``STICKYKV_COMPILE_EVICT`` is not evidence that the compiled eviction
    ran: the flag is read per call, the compiled fn is built lazily, and a
    benchmark harness that trusts the flag will report eager launch counts under
    a compiled name. The counters are the proof-of-execution contract
    ``flash_decode`` already applies to the decode kernel (armed vs fired), and
    ``perf_runner`` fails a row on them, so they have to actually discriminate.
    """
    pytest.importorskip("torch")
    from modules.windowed_cache.cache import (
        evict_path_stats, reset_evict_path_stats,
    )

    reset_evict_path_stats()
    _run_decode_across_eviction(compile_backend=None)
    eager = evict_path_stats()
    assert eager["eager"] >= 1, "the eager body ran but was not counted"
    assert eager["compiled"] == 0

    reset_evict_path_stats()
    try:
        _run_decode_across_eviction(compile_backend="aot_eager")
    except Exception as e:  # torch.compile unavailable on this build
        pytest.skip(f"torch.compile(aot_eager) unavailable: {type(e).__name__}: {e}")
    comp = evict_path_stats()
    assert comp["compiled"] == eager["eager"], (
        "the compiled run must take the compiled body for exactly the evictions "
        f"the eager run took eagerly (compiled={comp}, eager={eager})"
    )
    assert comp["eager"] == 0, "an eviction fell back to the eager body silently"
    reset_evict_path_stats()
