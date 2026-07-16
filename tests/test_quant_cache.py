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


def _make_cache(quant_ratio=0.5, ws=2, num_sink=0, prefill_len=8):
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
    """The stored int4 key codes for a window id (active or dormant)."""
    has, _, slot, _ = store.lookup(torch.tensor([[wid]], dtype=torch.long))
    assert bool(has[0, 0]), f"window {wid} has no slot-table entry"
    return store.table.key_codes[row, slot[0, 0]]


def _seed_prefill_state(cache, n_win=4, ws=2, num_sink=0, H=2, D=4):
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


def test_b_gt_1_with_quant_raises():
    cache = _make_cache(quant_ratio=0.5)
    k = torch.randn(2, 2, 3, 4)  # B=2
    with pytest.raises(NotImplementedError):
        cache.update(k, k.clone(), 0, cache_kwargs={"cache_position": torch.arange(3)})


def test_first_eviction_demotes_and_materializes():
    """w0→fp (top), w1→Q (next), w2 dropped, w3 local-fp. Effective = [w0,w1,w3]."""
    cache = _make_cache(quant_ratio=0.5, ws=2, num_sink=0)
    H, D, ws = 2, 4, 2
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
    assert st.position_ids[0].tolist() == [0, 1, 6, 7]         # w0 ‖ w3
    assert torch.equal(st.key_states[0][:, :2], k_post[:, 0:2])  # w0 untouched fp
    assert torch.equal(st.key_states[0][:, 2:], k_post[:, 6:8])  # w3 untouched fp

    # Effective K/V re-inserts w1 chronologically at positions 2,3.
    eff_k, eff_v = cache._materialize(0)
    assert eff_k.shape == (1, H, 6, D)                          # T_total = 4 + 2
    assert cache.get_seq_length(0) == 6
    # fp windows byte-identical inside the effective tensor
    assert torch.equal(eff_k[0][:, 0:2], k_post[:, 0:2])        # w0
    assert torch.equal(eff_k[0][:, 4:6], k_post[:, 6:8])        # w3
    # Q window w1 (positions 2,3) reconstructed within int4 error
    assert (eff_k[0][:, 2:4] - k_post[:, 2:4]).abs().max() < 0.3


def test_redemotion_after_promotion_reactivates():
    """Demote w1, promote it (dormant), then re-demote — codes unchanged (§10)."""
    cache = _make_cache(quant_ratio=0.5, ws=2, num_sink=0)
    H, D, ws = 2, 4, 2
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

    # Now promote w1 back: fp store = [w0(0,1), w1(2,3), w3(6,7)] merged [0,1,3],
    # rank so w1 stays top → both w0,w1 fp (top_k_fp=2), w3 local.
    st.window_scores = torch.zeros(1, 2, 3)
    st.window_scores[0, :, [0, 1]] = torch.tensor([10.0, 100.0])  # w1 outranks w0
    st.original_window_ids = torch.tensor([[0, 1, 3]])
    pol.top_k_fp, pol.N_q, pol.local_windows = 2, 0, 1
    cache._evict_two_tier(0, step=4)

    # w1 promoted → dormant, fp store now holds it again at positions 2,3.
    assert _active(store) == []
    assert int(store.table.n_live[0]) == 1  # dormant, retained not freed
    assert st.position_ids[0].tolist() == [0, 1, 2, 3, 6, 7]

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
            window_size=2, num_sink_tokens=0, local_window_size=2,
            cache_budget=0.5, quant_ratio=q, quant_memoize_read=memo,
        )
        return WindowedCache(
            config=cfg, prefill_len=8, model_config=_FakeModelConfig(),
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
        ws, H, D, prefill = 2, 2, 4, 8
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
        _seed_prefill_state(c, n_win=4, ws=2)
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
    ws, num_sink, H, D = 2, 0, 2, 4
    prefill = 8
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
    ws, num_sink, H, D = 2, 0, 2, 4
    prefill = 8
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
    ws, num_sink, H, D = 2, 0, 2, 4
    prefill = 8
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
        # Total retained stays bounded (compaction works): post-eviction fp is
        # (top_k_fp + local)*ws, plus up to ws growth between ws-spaced evictions,
        # plus the Q tier (N_q*ws), plus one window of slack.
        cap = num_sink + (pol.top_k_fp + pol.N_q + pol.local_windows + 2) * ws
        assert tfp + tq <= cap
        assert store.num_active_windows <= pol.N_q
        if store.num_active_windows > 0:
            used_q = True

    # Compaction actually happened (retained ≪ uncompacted prefill+steps).
    assert cache.get_seq_length(0) < prefill + 16
    assert used_q, "Q tier was never populated — demotion path not exercised"


# ---------------------------------------------------------------------------
# Flash / eager backend parity (shared cache.py ⇒ identical two-tier results)
# ---------------------------------------------------------------------------

from modules.windowed_eager_cache.cache import WindowedCache as EagerWindowedCache
from modules.windowed_eager_cache.config import WindowedCacheConfig as EagerCfg


def test_flash_eager_two_tier_parity():
    ws, num_sink, H, D, prefill = 2, 0, 2, 4, 8

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
    kp = torch.randn(1, H, prefill, D); vp = torch.randn(1, H, prefill, D)
    sc0 = torch.rand(1, H, _merged_W(flash, ws, num_sink, prefill))
    fk, _ = flash.update(kp.clone(), vp.clone(), 0,
                         cache_kwargs={"cache_position": torch.arange(prefill), "window_scores": sc0.clone()})
    ek, _ = eager.update(kp.clone(), vp.clone(), 0,
                         cache_kwargs={"cache_position": torch.arange(prefill), "window_scores": sc0.clone()})
    assert torch.equal(fk, ek)

    for t in range(prefill, prefill + 12):
        k1 = torch.randn(1, H, 1, D); v1 = torch.randn(1, H, 1, D)
        Wf = _merged_W(flash, ws, num_sink, flash._states[0].seq_length + 1)
        sc = torch.rand(1, H, Wf)
        fk, fv = flash.update(k1.clone(), v1.clone(), 0,
                              cache_kwargs={"cache_position": torch.arange(t, t + 1), "window_scores": sc.clone()})
        ek, ev = eager.update(k1.clone(), v1.clone(), 0,
                              cache_kwargs={"cache_position": torch.arange(t, t + 1), "window_scores": sc.clone()})
        assert torch.equal(fk, ek) and torch.equal(fv, ev)
        assert _active(flash._stores[0]) == _active(eager._stores[0])
        assert flash.get_seq_length(0) == eager.get_seq_length(0)
