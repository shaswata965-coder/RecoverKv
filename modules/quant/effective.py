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

Both RoPE primitives accept either an unbatched ``[H_kv, T, D]`` window with
``[T]`` positions or a batched ``[B, H_kv, T, D]`` with ``[B, T]``. The unbatched
form is exactly the batched one at ``B = 1`` — it just unsqueezes — so the two
are bit-identical, not merely equivalent. Everything here is shape-agnostic to
head count.
"""

from __future__ import annotations

from typing import Tuple

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
    key_post_rope : ``[H_kv, window, D]`` or ``[B, H_kv, window, D]``
        Post-RoPE keys from the fp store.
    position_range : ``[window]`` or ``[B, window]`` int64
        The window's original positions.
    """
    apply_rotary_pos_emb = _apply_rotary()
    batched = key_post_rope.dim() == 4
    k = key_post_rope if batched else key_post_rope.unsqueeze(0)
    cos, sin = _rope_cos_sin(rope_module, k, position_range)
    _, k_un = apply_rotary_pos_emb(k, k, cos, -sin)
    return k_un if batched else k_un.squeeze(0)


def rotate_key_window(
    key_pre_rope: Tensor,
    position_range: Tensor,
    rope_module: torch.nn.Module,
) -> Tensor:
    """Apply RoPE to a window's pre-RoPE keys at its original positions (§5, §8).

    Accepts ``[H_kv, window, D]`` with ``[window]`` positions, or
    ``[B, H_kv, window, D]`` with ``[B, window]``.
    """
    apply_rotary_pos_emb = _apply_rotary()
    batched = key_pre_rope.dim() == 4
    k = key_pre_rope if batched else key_pre_rope.unsqueeze(0)
    cos, sin = _rope_cos_sin(rope_module, k, position_range)
    _, k_rot = apply_rotary_pos_emb(k, k, cos, sin)
    return k_rot if batched else k_rot.squeeze(0)


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
    fp_keys, fp_values : ``[B, H_kv, T_fp, D]``
        The fp store (post-RoPE keys). Per row — rows retain different windows.
    fp_positions : ``[B, T_fp]`` int64
        Original absolute positions of every fp token.
    store : QuantizedStore
        The Q tier. Active entries carry codes + frozen ``position_range``.
    num_sink, window_size : int
    rope_module : nn.Module
    out_dtype : torch.dtype, optional
        Dequant output dtype; defaults to ``fp_keys.dtype``.

    Returns
    -------
    (eff_k, eff_v) : each ``[B, H_kv, T_total, D]`` where ``T_total = T_fp + T_q``.

    ``T_total`` is the same for every row: the tier split keeps exactly
    ``k_fp``/``n_q`` windows and both derive from config + ``W``, which are shared
    across rows — so divergent eviction stays **rectangular** and the effective
    K/V needs no padding or keep-mask (BATCHING_PLAN.md §3).
    """
    if out_dtype is None:
        out_dtype = fp_keys.dtype

    # Fast path: empty Q tier ⇒ effective K/V is the fp store, byte-identical.
    if store.num_active_windows == 0:
        return fp_keys, fp_values

    B, H, _, D = fp_keys.shape
    sink_k = fp_keys[:, :, :num_sink]
    sink_v = fp_values[:, :, :num_sink]
    body_k = fp_keys[:, :, num_sink:]
    body_v = fp_values[:, :, num_sink:]
    body_pos = fp_positions[:, num_sink:]

    # Q windows: dequantized + RoPE'd at their frozen original positions.
    # RoPE is per-token pointwise, so rotating all N_q windows in ONE call
    # (flattened window-major) is bit-identical to N_q separate per-window calls
    # at a single kernel launch. The store may memoize this across steps — the
    # tier cannot change between evictions (§10).
    k_q, v_q, q_pos_flat = store.effective_q_tier(rope_module, out_dtype)

    # Interleave chronologically, entirely ON DEVICE: concatenate the fp body
    # with the Q tokens and sort the merged token axis by window id, per row.
    #
    # The rerotation design could scatter by absolute position (positions were
    # renumbered to tile arange(T_total)); keeping original positions leaves gaps
    # where windows were evicted, so they don't tile. But chronological order
    # only needs the window-id ORDERING, not the positions themselves — so an
    # argsort gets us there without the host ever seeing the ids. A stable sort
    # keeps each window's tokens in position order, and a window is always
    # wholly fp or wholly Q (eviction assigns tiers per window), so no window is
    # ever split across the two sources.
    #
    # Doing this on the host instead — grouping the body, .tolist()ing the ids,
    # sorting chunks in Python — costs ~4 GPU→CPU syncs per layer PER DECODE
    # STEP (~16k per sample), which is pure pipeline stall. The row axis rides
    # through for free: argsort just gains a dim.
    merged_k = torch.cat([body_k, k_q], dim=2)
    merged_v = torch.cat([body_v, v_q], dim=2)
    merged_wids = torch.cat([
        (body_pos.to(torch.long) - num_sink) // window_size,       # [B, n_body]
        (q_pos_flat.to(torch.long) - num_sink) // window_size,     # [B, N*S]
    ], dim=1)
    order = torch.argsort(merged_wids, dim=1, stable=True)         # [B, T_m]
    idx = order.unsqueeze(1).unsqueeze(-1).expand(B, H, order.shape[1], D)
    eff_k = torch.cat([sink_k, torch.gather(merged_k, 2, idx)], dim=2)
    eff_v = torch.cat([sink_v, torch.gather(merged_v, 2, idx)], dim=2)
    return eff_k, eff_v
