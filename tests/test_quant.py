"""CPU unit tests for the two-tier quantization package (design.md §2–§8).

All tests run on CPU with synthetic data; no model loads, no GPU.
"""

from __future__ import annotations

import pytest
import torch

from modules.quant.quantizer import (
    dequantize_key_window,
    dequantize_value_window,
    grid_group,
    pack_crumbs_last,
    quantize_key_window,
    quantize_value_window,
    unpack_crumbs_last,
)


# ---------------------------------------------------------------------------
# Crumb packing (int2: 4 codes per byte)
# ---------------------------------------------------------------------------


class TestPacking:
    def test_pack_unpack_roundtrip(self):
        codes = torch.randint(0, 4, (3, 4, 8), dtype=torch.uint8)
        packed = pack_crumbs_last(codes)
        assert packed.shape == (3, 4, 2)  # 8 codes / 4 per byte
        out = unpack_crumbs_last(packed, 8)
        assert torch.equal(out, codes)

    def test_index_order_low_to_high_bits(self):
        # codes [1,2,3,0] -> byte = 1 | (2<<2) | (3<<4) | (0<<6) = 1+8+48 = 57
        codes = torch.tensor([[1, 2, 3, 0]], dtype=torch.uint8)
        packed = pack_crumbs_last(codes)
        assert packed.item() == 57

    def test_pack_rejects_non_multiple_of_four(self):
        with pytest.raises(ValueError):
            pack_crumbs_last(torch.zeros(6, dtype=torch.uint8))


# ---------------------------------------------------------------------------
# Key quantization
# ---------------------------------------------------------------------------


class TestKeyQuant:
    def test_shapes(self):
        H, W, D = 8, 8, 128
        # fp16 KV — a real key dtype (windows are fp16 or bf16, never fp32). The
        # grid group scale follows the KV dtype (quantizer.grid_dtype_for): fp16
        # KV keeps the fp16 grid, byte for byte, so this is the bit-identity path.
        k = torch.randn(H, W, D, dtype=torch.float16)
        packed, scale, zero = quantize_key_window(k)
        assert packed.shape == (H, D, W // 4)
        assert packed.dtype == torch.uint8
        # The grid is one byte per (head, channel) over an fp16 scale shared by
        # GRID_GROUP of them -- see QGrid.
        assert scale.q.shape == (H, D) and scale.q.dtype == torch.uint8
        assert zero.q.shape == (H, D) and zero.q.dtype == torch.int8
        assert scale.s.shape == (H, D // grid_group(D))
        assert scale.s.dtype == torch.float16 and zero.s.dtype == torch.float16
        # A scale code is never 0: the fit divides by the decoded grid.
        assert int(scale.q.min()) >= 1

        k_hat = dequantize_key_window(packed, scale, zero, W, out_dtype=torch.float32)
        assert k_hat.shape == (H, W, D)

    def test_roundtrip_error_bounded(self):
        # int2 over a group of dynamic range R has max error <= scale/2 = R/6.
        H, W, D = 4, 8, 64
        k = torch.randn(H, W, D) * 3.0
        packed, scale, zero = quantize_key_window(k)
        k_hat = dequantize_key_window(packed, scale, zero, W, out_dtype=torch.float32)
        # Per (head, channel) range along the token axis.
        rng = k.amax(dim=1, keepdim=True) - k.amin(dim=1, keepdim=True)  # [H,1,D]
        max_err = (k_hat - k).abs().amax(dim=1)  # [H, D]
        # Allow a hair over R/6 for the one-byte grid's rounding: the stored
        # scale can exceed the true one by up to half a group quantum, which
        # widens the four levels and so the worst-case error with them.
        bound = (rng.squeeze(1) / 6.0) * 1.20 + 1e-3
        assert torch.all(max_err <= bound)

    def test_degenerate_group_exact(self):
        # All tokens identical along the token axis -> mx == mn -> x_hat == mn.
        H, W, D = 2, 4, 16
        const = torch.randn(H, 1, D).expand(H, W, D).contiguous()
        packed, scale, zero = quantize_key_window(const)
        k_hat = dequantize_key_window(packed, scale, zero, W, out_dtype=torch.float32)
        # Every code is 0, so the dequant returns the stored `zero` exactly --
        # which is now the DECODED grid, not an fp16 copy of the constant.
        assert torch.equal(k_hat, zero.decode().unsqueeze(1).expand(H, W, D))

    def test_fp16_grid_idempotence(self):
        # Dequantizing twice against the pinned fp16 grid is bit-identical, and
        # re-quantizing the dequant reproduces the same codes (pinned-grid
        # idempotence — the basis for reactivation-not-requant, design §10).
        H, W, D = 4, 8, 32
        k = torch.randn(H, W, D) * 2.0
        packed, scale, zero = quantize_key_window(k)
        k_hat1 = dequantize_key_window(packed, scale, zero, W, out_dtype=torch.float16)
        k_hat2 = dequantize_key_window(packed, scale, zero, W, out_dtype=torch.float16)
        assert torch.equal(k_hat1, k_hat2)
        # Re-quantize the fp16 dequant: codes must be identical (idempotent).
        packed2, scale2, zero2 = quantize_key_window(k_hat1.to(torch.float32))
        assert torch.equal(packed2, packed)


# ---------------------------------------------------------------------------
# Value quantization
# ---------------------------------------------------------------------------


class TestValueQuant:
    def test_shapes(self):
        H, W, D = 8, 6, 128
        v = torch.randn(H, W, D)
        packed, scale, zero = quantize_value_window(v)
        assert packed.shape == (H, W, D // 4)
        assert scale.q.shape == (H, W) and scale.q.dtype == torch.uint8
        assert zero.q.shape == (H, W) and zero.q.dtype == torch.int8
        # ws < GRID_GROUP, so a value window's grid shares one scale per head.
        assert scale.s.shape == (H, W // grid_group(W))

        v_hat = dequantize_value_window(packed, scale, zero, D, out_dtype=torch.float32)
        assert v_hat.shape == (H, W, D)

    def test_roundtrip_error_bounded(self):
        H, W, D = 4, 8, 64
        v = torch.randn(H, W, D) * 3.0
        packed, scale, zero = quantize_value_window(v)
        v_hat = dequantize_value_window(packed, scale, zero, D, out_dtype=torch.float32)
        rng = v.amax(dim=2, keepdim=True) - v.amin(dim=2, keepdim=True)  # [H,W,1]
        max_err = (v_hat - v).abs().amax(dim=2)  # [H, W]
        bound = (rng.squeeze(2) / 6.0) * 1.05 + 1e-3
        assert torch.all(max_err <= bound)

    def test_degenerate_group_exact(self):
        H, W, D = 2, 4, 16
        const = torch.randn(H, W, 1).expand(H, W, D).contiguous()
        packed, scale, zero = quantize_value_window(const)
        v_hat = dequantize_value_window(packed, scale, zero, D, out_dtype=torch.float32)
        assert torch.equal(v_hat, zero.decode().unsqueeze(2).expand(H, W, D))


# ---------------------------------------------------------------------------
# Store / ledger / effective-KV read path
# ---------------------------------------------------------------------------

from modules.quant.store import QuantizedStore
from modules.quant.effective import (
    materialize_effective_kv,
    rotate_key_window,
    unrotate_key_window,
)


class _RealRoPE(torch.nn.Module):
    """Standard Llama RoPE: cos/sin [B, T, head_dim], unit circle (exact inverse)."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        B = position_ids.shape[0]
        inv = self.inv_freq[None, :, None].float().expand(B, -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)          # [B, T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)      # [B, T, head_dim]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class TestRoPEInverse:
    def test_unrotate_then_rotate_is_identity(self):
        H, W, D = 2, 4, 8
        rope = _RealRoPE(D)
        pos = torch.arange(100, 100 + W, dtype=torch.long)
        k_post = torch.randn(H, W, D)
        k_pre = unrotate_key_window(k_post, pos, rope)
        k_back = rotate_key_window(k_pre, pos, rope)
        assert torch.allclose(k_back, k_post, atol=1e-5)


def _store(ws=4, D=8, H=2, n_slots=8, B=1):
    s = QuantizedStore(window_size=ws, head_dim=D, num_kv_heads=H, n_slots=n_slots)
    s.ensure(B, torch.device("cpu"))
    return s


def _demote_one(store, wid, k_pre, v, pos, row=0):
    """Demote a single window through the batched API (B=1, one lane)."""
    B = store.table.batch_size
    slot = store.table.free_slots(1)
    valid = torch.zeros(B, 1, dtype=torch.bool)
    valid[row, 0] = True
    w = torch.full((B, 1), -1, dtype=torch.long)
    w[row, 0] = wid
    store.demote_many(
        slot, valid, w,
        k_pre.unsqueeze(0).unsqueeze(0).expand(B, 1, *k_pre.shape),
        v.unsqueeze(0).unsqueeze(0).expand(B, 1, *v.shape),
        pos.unsqueeze(0).unsqueeze(0).expand(B, 1, pos.shape[0]),
    )
    store.commit_active_count(store.num_active_windows + 1)
    return slot


def _slot_of(store, wid):
    w = torch.tensor([[wid]], dtype=torch.long)
    has, active, slot, _ = store.lookup(w)
    return has[0, 0].item(), active[0, 0].item(), slot


class TestStore:
    def _make_window(self, H=2, W=4, D=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(H, W, D, generator=g)

    def test_demote_promote_roundtrip(self):
        store = _store()
        k_pre = self._make_window(seed=1)
        v = self._make_window(seed=2)
        pos = torch.arange(8, 12, dtype=torch.long)
        _demote_one(store, 5, k_pre, v, pos)
        assert store.num_active_windows == 1

        has, active, slot = _slot_of(store, 5)
        assert has and active
        valid = torch.ones(1, 1, dtype=torch.bool)
        k_hat, v_hat, pos_out = store.promote_many(slot, valid, out_dtype=torch.float32)
        store.commit_active_count(0)
        assert torch.equal(pos_out[0, 0], pos)
        # promotion made it dormant — entry retained, not freed (§6, §10)
        assert store.num_active_windows == 0
        has, active, _ = _slot_of(store, 5)
        assert has and not active
        assert int(store.table.n_live[0]) == 1
        # dequant error bounded
        k_hat = k_hat[0, 0]
        rngk = k_pre.amax(1, keepdim=True) - k_pre.amin(1, keepdim=True)
        assert torch.all((k_hat - k_pre).abs().amax(1) <= rngk.squeeze(1) / 6.0 * 1.05 + 1e-3)

    def test_redemotion_reactivates_no_requantize(self):
        store = _store()
        k_pre = self._make_window(seed=3)
        v = self._make_window(seed=4)
        pos = torch.arange(0, 4, dtype=torch.long)
        slot = _demote_one(store, 7, k_pre, v, pos)
        codes_before = store.table.key_codes[0, slot[0, 0]].clone()

        # promote (dormant), then re-demote: reactivation is a bit flip and must
        # not touch the codes or the frozen position_range (§10).
        valid = torch.ones(1, 1, dtype=torch.bool)
        store.promote_many(slot, valid, out_dtype=torch.float32)
        store.commit_active_count(0)

        has, active, slot2 = _slot_of(store, 7)
        assert has and not active
        store.reactivate_many(slot2, valid)
        store.commit_active_count(1)

        has, active, _ = _slot_of(store, 7)
        assert has and active
        codes_after = store.table.key_codes[0, slot[0, 0]]
        assert torch.equal(codes_before, codes_after)              # codes never changed
        assert torch.equal(store.table.slot_pos[0, slot[0, 0]], pos)  # positions frozen
        store.validate()

    def test_slot_table_invariants_hold(self):
        store = _store()
        k = self._make_window()
        _demote_one(store, 1, k, k, torch.arange(4))
        _demote_one(store, 2, k, k, torch.arange(4, 8))
        store.validate()
        # distinct ids land in distinct slots; free slots stay free
        assert int(store.table.n_live[0]) == 2
        assert torch.equal(store.active_ids(), torch.tensor([[1, 2]]))

    def test_retain_only_frees_dropped_and_keeps_dormant(self):
        """A dropped window's slot is freed; a promoted (dormant) one is not (§6)."""
        store = _store()
        k = self._make_window()
        _demote_one(store, 1, k, k, torch.arange(4))
        _demote_one(store, 2, k, k, torch.arange(4, 8))
        assert int(store.table.n_live[0]) == 2

        # Retain only window 1 -> window 2's slot is freed outright.
        _, _, _, match = store.lookup(torch.tensor([[1]]))
        store.retain_only(match)
        store.commit_active_count(1)
        assert int(store.table.n_live[0]) == 1
        assert not _slot_of(store, 2)[0]
        assert _slot_of(store, 1)[0]
        store.validate()


class TestMaterialize:
    def test_empty_qtier_returns_fp_unchanged(self):
        H, T, D = 2, 10, 8
        store = _store(ws=4, D=8, H=2)
        rope = _RealRoPE(D)
        fp_k = torch.randn(1, H, T, D)
        fp_v = torch.randn(1, H, T, D)
        fp_pos = torch.arange(T, dtype=torch.long).unsqueeze(0)
        eff_k, eff_v, score_meta = materialize_effective_kv(
            fp_k, fp_v, fp_pos, store, num_sink=2, window_size=4, rope_module=rope
        )
        assert torch.equal(eff_k, fp_k)
        assert torch.equal(eff_v, fp_v)
        assert score_meta is None  # empty Q tier ⇒ no score-scatter needed

    def test_interleave_reconstructs_full_sequence(self):
        """Demote windows 1 & 3; materialize lays out the UNSORTED effective K/V
        as [sink ‖ fp body (w0,w2) ‖ Q (w1,w3)] with Q windows rotated at their
        ORIGINAL positions (within int2 error), and returns the score-scatter map
        that reorders the physical [body ‖ Q] window axis back to ascending id."""
        torch.manual_seed(0)
        H, D, ws, num_sink = 2, 8, 4, 2
        rope = _RealRoPE(D)
        n_win = 4
        T = num_sink + n_win * ws  # 18

        # pre-RoPE keys/values for the whole sequence, positions 0..T-1
        pos_all = torch.arange(T, dtype=torch.long)
        k_pre_all = torch.randn(H, T, D)
        v_all = torch.randn(H, T, D)
        # reference full-fp keys (post-RoPE)
        k_post_all = rotate_key_window(k_pre_all, pos_all, rope)

        store = _store(ws=ws, D=D, H=H)
        q_windows = {1, 3}
        # build fp store = sink + fp windows (ids 0,2) in chronological order
        fp_k_parts = [k_post_all[:, :num_sink]]
        fp_v_parts = [v_all[:, :num_sink]]
        fp_pos_parts = [pos_all[:num_sink]]
        for w in range(n_win):
            s = num_sink + w * ws
            e = s + ws
            if w in q_windows:
                # demote: store pre-RoPE keys + values + original positions
                _demote_one(store, w, k_pre_all[:, s:e], v_all[:, s:e], pos_all[s:e])
            else:
                fp_k_parts.append(k_post_all[:, s:e])
                fp_v_parts.append(v_all[:, s:e])
                fp_pos_parts.append(pos_all[s:e])
        fp_k = torch.cat(fp_k_parts, dim=1).unsqueeze(0)
        fp_v = torch.cat(fp_v_parts, dim=1).unsqueeze(0)
        fp_pos = torch.cat(fp_pos_parts, dim=0).unsqueeze(0)

        eff_k, eff_v, score_meta = materialize_effective_kv(
            fp_k, fp_v, fp_pos, store, num_sink=num_sink, window_size=ws,
            rope_module=rope, out_dtype=torch.float32,
        )
        assert eff_k.shape == (1, H, T, D)
        eff_k, eff_v = eff_k[0], eff_v[0]

        # Physical layout is [sink ‖ body: w0,w2 ‖ Q: w1,w3] — NOT sorted.
        # Reference token slices per window id (0-based, within the full seq).
        def win(wid):
            s = num_sink + wid * ws
            return slice(s, s + ws)

        # sink + fp body windows (w0, w2) are byte-identical, at their PHYSICAL
        # (unsorted) offsets: sink | w0 | w2.
        assert torch.equal(eff_k[:, :num_sink], k_post_all[:, :num_sink])
        assert torch.equal(eff_k[:, num_sink:num_sink + ws], k_post_all[:, win(0)])
        assert torch.equal(eff_k[:, num_sink + ws:num_sink + 2 * ws], k_post_all[:, win(2)])
        assert torch.equal(eff_v[:, num_sink:num_sink + ws], v_all[:, win(0)])
        assert torch.equal(eff_v[:, num_sink + ws:num_sink + 2 * ws], v_all[:, win(2)])

        # Q windows (w1, w3) follow the body, rotated at ORIGINAL positions, so
        # they match within int2 error only (coarser than int4 — 4 levels).
        q_start = num_sink + 2 * ws
        assert torch.allclose(eff_k[:, q_start:q_start + ws], k_post_all[:, win(1)], atol=1.0)
        assert torch.allclose(eff_k[:, q_start + ws:q_start + 2 * ws], k_post_all[:, win(3)], atol=1.0)
        assert (eff_v[:, q_start:q_start + ws] - v_all[:, win(1)]).abs().max() < 1.5
        assert (eff_v[:, q_start + ws:q_start + 2 * ws] - v_all[:, win(3)]).abs().max() < 1.5

        # Score-scatter map: physical per-window ids are [w0,w2 | w1,w3] = [0,2,1,3];
        # argsort → [0,2,1,3] scatters them back to ascending merged id [0,1,2,3].
        order, q_token_len = score_meta
        assert q_token_len == 2 * ws
        assert order[0].tolist() == [0, 2, 1, 3]


def test_affine_quantize_inplace_matches_the_out_of_place_form():
    """The in-place chain must be BIT-identical to the expression it replaced.

    `_affine_quantize` used to build `round((x32 - zero) / scale)` then clamp
    out of place, holding three fp32 buffers of x's size live at its peak --
    12 bytes per element to quantize a 2-byte one. At 2048/batch-32 the first
    eviction asked for 7.25 GiB there and the allocator refused. Chaining in
    place keeps one buffer.

    That is only a legitimate change if nothing about the result moves, so this
    re-implements the old expression here and compares codes, scale and zero
    exactly -- across dtypes, group axes, and the degenerate mx == mn case the
    scale guard exists for.
    """
    import torch
    from modules.quant.quantizer import (
        _affine_quantize, _quantize_grid, _LEVELS, grid_dtype_for,
    )

    def _reference(x, group_dim):
        x32 = x.to(torch.float32)
        mx = x32.amax(dim=group_dim, keepdim=True)
        mn = x32.amin(dim=group_dim, keepdim=True)
        scale = (mx - mn) / _LEVELS
        scale = torch.where(mx == mn, torch.ones_like(scale), scale)
        # The grid group scale follows the KV dtype (grid_dtype_for): fp16 KV
        # keeps fp16, bf16 KV takes bf16 so it cannot overflow. Mirror that here
        # so the reference stays bit-identical to the code across all dtypes.
        gdt = grid_dtype_for(x.dtype)
        s_g = _quantize_grid(scale.squeeze(group_dim), signed=False, grid_dtype=gdt)
        z_g = _quantize_grid(mn.squeeze(group_dim), signed=True, grid_dtype=gdt)
        q = torch.round((x32 - z_g.decode().unsqueeze(group_dim))
                        / s_g.decode().unsqueeze(group_dim))
        q = torch.clamp(q, 0.0, _LEVELS)
        return q.to(torch.uint8), s_g, z_g

    g = torch.Generator().manual_seed(20260917)
    cases = []
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        cases.append(torch.randn(3, 5, 32, generator=g).to(dtype))
        cases.append((torch.randn(2, 4, 16, generator=g) * 1e3).to(dtype))
    # Degenerate group: every element of the group axis identical => mx == mn.
    cases.append(torch.full((2, 3, 8), 0.75, dtype=torch.float16))
    # Mixed: one degenerate group beside live ones.
    mixed = torch.randn(2, 3, 8, generator=g).to(torch.float16)
    mixed[0, 1, :] = -1.25
    cases.append(mixed)

    for x in cases:
        for group_dim in (-1, 1):
            before = x.clone()
            codes, scale_g, zero_g = _affine_quantize(x, group_dim)
            r_codes, r_scale, r_zero = _reference(before, group_dim)
            tag = f"dtype={x.dtype} group_dim={group_dim}"
            assert torch.equal(x, before), f"{tag}: the input was mutated"
            assert torch.equal(codes, r_codes), f"{tag}: codes moved"
            assert torch.equal(scale_g.q, r_scale.q), f"{tag}: scale codes moved"
            assert torch.equal(scale_g.s, r_scale.s), f"{tag}: scale group moved"
            assert torch.equal(zero_g.q, r_zero.q), f"{tag}: zero codes moved"
            assert torch.equal(zero_g.s, r_zero.s), f"{tag}: zero group moved"
            assert codes.max() <= _LEVELS, f"{tag}: code above the top level"


def test_affine_quantize_does_not_mutate_an_fp32_input():
    """`.to(float32)` is a NO-OP on an fp32 tensor, so the in-place chain would
    write through to the caller's tensor. The clone guard is the whole reason
    this is safe; without it a caller that quantized an fp32 key window would
    find its keys replaced by int2 codes, silently."""
    import torch
    from modules.quant.quantizer import _affine_quantize

    x = torch.randn(2, 3, 16, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(4))
    before = x.clone()
    _affine_quantize(x, -1)
    assert torch.equal(x, before), "an fp32 input was quantized in place"


def test_affine_dequantize_inplace_matches_the_out_of_place_form():
    """Same contract on the read side: one fp32 buffer, identical values."""
    import torch
    from modules.quant.quantizer import _affine_quantize, _affine_dequantize

    g = torch.Generator().manual_seed(11)
    for dtype in (torch.float16, torch.float32):
        x = torch.randn(3, 4, 32, generator=g).to(dtype)
        codes, scale_g, zero_g = _affine_quantize(x, -1)
        scale = scale_g.decode().unsqueeze(-1)   # the reduced group axis
        zero = zero_g.decode().unsqueeze(-1)
        ref = ((codes.to(torch.float32) * scale) + zero).to(dtype)
        codes_before = codes.clone()
        got = _affine_dequantize(codes, scale, zero, dtype)
        assert torch.equal(got, ref), f"dequant moved for {dtype}"
        assert torch.equal(codes, codes_before), "the codes were mutated"


def test_apply_rotary_one_matches_huggingface():
    """Rotating ONE tensor must be bit-identical to HF rotating two.

    `unrotate_key_window` / `rotate_key_window` used to call
    `apply_rotary_pos_emb(k, k, cos, sin)` -- passing the key as both q and k,
    so HF computed the identical result twice and one copy was discarded. That
    is double the elementwise work and roughly five key-sized fp buffers live at
    the peak to produce one. At 2048/batch-32 the demote path asked for 3.62 GiB
    there and the allocator refused.

    `_apply_rotary_one` computes HF's expression once, with `rotate_half`
    imported from HF so the convention is still theirs. This asserts the two
    agree exactly -- not `allclose`. If a transformers upgrade changes the RoPE
    convention, this is the test that says so.
    """
    import torch
    from modules.quant.effective import _apply_rotary, _apply_rotary_one

    apply_rotary_pos_emb = _apply_rotary()
    g = torch.Generator().manual_seed(913)
    for dtype in (torch.float32, torch.float16):
        for B, H, W, D in [(1, 2, 4, 8), (3, 4, 16, 32), (2, 1, 1, 64)]:
            k = torch.randn(B, H, W, D, generator=g).to(dtype)
            cos = torch.randn(B, W, D, generator=g).to(dtype)
            sin = torch.randn(B, W, D, generator=g).to(dtype)
            k_ref = k.clone()
            _, expect = apply_rotary_pos_emb(k_ref, k_ref, cos, sin)
            got = _apply_rotary_one(k, cos, sin)
            tag = f"dtype={dtype} shape={(B, H, W, D)}"
            assert torch.equal(got, expect), f"{tag}: rotation moved"
            assert torch.equal(k, k_ref), f"{tag}: the input key was mutated"


def test_rotate_and_unrotate_are_unchanged_by_the_single_tensor_path():
    """The two public helpers must still agree with HF end to end."""
    import torch
    from modules.quant.effective import _apply_rotary

    apply_rotary_pos_emb = _apply_rotary()
    H, W, D = 3, 6, 16
    rope = _RealRoPE(D)
    pos = torch.arange(40, 40 + W, dtype=torch.long)
    k_post = torch.randn(H, W, D, generator=torch.Generator().manual_seed(5))

    k4 = k_post.unsqueeze(0)
    cos, sin = rope(k4, pos.unsqueeze(0))
    _, expect_un = apply_rotary_pos_emb(k4, k4, cos, -sin)
    assert torch.equal(unrotate_key_window(k_post, pos, rope),
                       expect_un.squeeze(0))

    _, expect_rot = apply_rotary_pos_emb(k4, k4, cos, sin)
    assert torch.equal(rotate_key_window(k_post, pos, rope),
                       expect_rot.squeeze(0))
