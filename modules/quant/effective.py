"""Effective K/V materialization — the two-tier read path (design.md §5, §8).

Phase 1 (materialize-then-interleave): dequantize the Q store, apply RoPE at
each Q window's **original absolute positions** (its immutable
``position_range`` — eviction never rebases them), and interleave with the fp
store **chronologically by window id** into one effective tensor for the
standard attention path.

Two RoPE primitives, both built on the model's own ``apply_rotary_pos_emb`` so
NTK/YaRN scaling is preserved:

- :func:`unrotate_key_window` — one-time, at first demotion: strip RoPE off a
  window's fp keys (negated sin) at its original positions to get pre-RoPE codes.
- :func:`rotate_key_window` — every read: apply RoPE to dequantized Q keys at
  the window's original positions.

Everything here is per row (B = 1 in v1) and shape-agnostic to head count.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


def _apply_rotary():
    """Lazily import the model's ``apply_rotary_pos_emb`` (Llama, else Qwen2)."""
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except ImportError:  # pragma: no cover - depends on installed model families
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    return apply_rotary_pos_emb


def _rope_cos_sin(
    rope_module: torch.nn.Module, ref: Tensor, position_range: Tensor
) -> Tuple[Tensor, Tensor]:
    """cos/sin for one window's positions. ``ref`` supplies device/dtype only."""
    pos = position_range.to(torch.long)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)  # [1, window]
    return rope_module(ref, pos)


def unrotate_key_window(
    key_post_rope: Tensor,
    position_range: Tensor,
    rope_module: torch.nn.Module,
) -> Tensor:
    """Strip RoPE from a window's fp keys (design §5, one-time demotion).

    ``cos(−θ)=cos θ``, ``sin(−θ)=−sin θ`` — undo the rotation at the window's
    *original absolute positions*.

    Parameters
    ----------
    key_post_rope : ``[H_kv, window, D]`` — post-RoPE keys from the fp store.
    position_range : ``[window]`` int64 — the window's original positions.
    """
    apply_rotary_pos_emb = _apply_rotary()
    k = key_post_rope.unsqueeze(0)  # [1, H_kv, window, D]
    cos, sin = _rope_cos_sin(rope_module, k, position_range)
    _, k_un = apply_rotary_pos_emb(k, k, cos, -sin)
    return k_un.squeeze(0)


def rotate_key_window(
    key_pre_rope: Tensor,
    position_range: Tensor,
    rope_module: torch.nn.Module,
) -> Tensor:
    """Apply RoPE to a window's pre-RoPE keys at its original positions (§5, §8)."""
    apply_rotary_pos_emb = _apply_rotary()
    k = key_pre_rope.unsqueeze(0)  # [1, H_kv, window, D]
    cos, sin = _rope_cos_sin(rope_module, k, position_range)
    _, k_rot = apply_rotary_pos_emb(k, k, cos, sin)
    return k_rot.squeeze(0)


def window_id_of(position: int, num_sink: int, window_size: int) -> int:
    """The chronological window id owning an original absolute ``position``.

    Windows are fixed chunks of the sequence and eviction never renumbers, so a
    token's window id is a pure function of its original position (design §1,
    §5): ``(position − num_sink) // window_size`` for post-sink tokens. This is
    the single source of truth for the interleave order — it needs no separate
    id bookkeeping and is immune to score/state skew.
    """
    return (int(position) - num_sink) // window_size


def materialize_effective_kv(
    fp_keys: Tensor,
    fp_values: Tensor,
    fp_positions: Tensor,
    store,
    num_sink: int,
    window_size: int,
    rope_module: torch.nn.Module,
    out_dtype: torch.dtype = None,
) -> Tuple[Tensor, Tensor]:
    """Interleave the fp and Q tiers into one effective K/V (design §5, §8).

    Emits ``[sink ‖ windows in chronological (window-id) order]``. Window ids
    are derived from ``fp_positions`` / each Q window's frozen ``position_range``
    via :func:`window_id_of`, so no external id map is needed and the result is
    exact regardless of any transient score/state skew. The sink prefix (fp
    store's first ``num_sink`` tokens) is carried through unchanged so the
    scorer still strips exactly the sink.

    Attention is order-free (RoPE bakes each key's position into its values), so
    this order matters only for the window **scorer**, which chunks the physical
    key axis — hence the chronological interleave.

    Parameters
    ----------
    fp_keys, fp_values : ``[H_kv, T_fp, D]``
        The fp store for one row (post-RoPE keys).
    fp_positions : ``[T_fp]`` int64
        Original absolute positions of every fp token.
    store : QuantizedStore
        The Q tier. Active entries carry codes + frozen ``position_range``.
    num_sink, window_size : int
    rope_module : nn.Module
    out_dtype : torch.dtype, optional
        Dequant output dtype; defaults to ``fp_keys.dtype``.

    Returns
    -------
    (eff_k, eff_v) : each ``[H_kv, T_total, D]`` where ``T_total = T_fp + T_q``.
    """
    if out_dtype is None:
        out_dtype = fp_keys.dtype

    # Fast path: empty Q tier ⇒ effective K/V is the fp store, byte-identical.
    if store.num_active_windows == 0:
        return fp_keys, fp_values

    sink_k = fp_keys[:, :num_sink]
    sink_v = fp_values[:, :num_sink]
    body_k = fp_keys[:, num_sink:]
    body_v = fp_values[:, num_sink:]
    body_pos = fp_positions[num_sink:]
    n_body = body_k.shape[1]

    chunks: List[Tuple[int, Tensor, Tensor]] = []

    # fp windows: group contiguous body tokens by window id (ascending along the
    # body — fp windows are stored in ascending-id order, with gaps where Q
    # windows were removed).
    i = 0
    while i < n_body:
        w = window_id_of(body_pos[i].item(), num_sink, window_size)
        j = i + 1
        while j < n_body and window_id_of(body_pos[j].item(), num_sink, window_size) == w:
            j += 1
        chunks.append((w, body_k[:, i:j], body_v[:, i:j]))
        i = j

    # Q windows: dequantize, RoPE at their original positions, tag by window id.
    _, q_keys_pre, q_values, q_positions = store.gather_active(out_dtype=out_dtype)
    for idx in range(q_keys_pre.shape[0]):
        pos = q_positions[idx]
        k_rot = rotate_key_window(q_keys_pre[idx], pos, rope_module)
        w = window_id_of(pos[0].item(), num_sink, window_size)
        chunks.append((w, k_rot.to(out_dtype), q_values[idx].to(out_dtype)))

    # Interleave by chronological window id.
    chunks.sort(key=lambda c: c[0])
    body_k2 = torch.cat([c[1] for c in chunks], dim=1)
    body_v2 = torch.cat([c[2] for c in chunks], dim=1)

    eff_k = torch.cat([sink_k, body_k2], dim=1)
    eff_v = torch.cat([sink_v, body_v2], dim=1)
    return eff_k, eff_v
