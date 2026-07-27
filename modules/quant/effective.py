"""Effective K/V materialization — the two-tier read path (design.md §5, §8).

Phase 1 (materialize-then-concat): dequantize the Q store, apply RoPE at each Q
window's **original absolute positions** (its immutable ``position_range`` —
eviction never rebases them), and concatenate with the fp store into one
effective tensor laid out ``[sink ‖ fp body ‖ Q]`` for the standard attention
path. The tiers are **not** globally id-sorted: attention is order-free, so the
only consumer that cares about chronological order is the window scorer, and it
undoes the layout cheaply on the ``[B, H, W]`` score axis (see
:func:`materialize_effective_kv`) rather than on the K/V tensors.

Two RoPE primitives, both built on the model's own ``apply_rotary_pos_emb`` so
NTK/YaRN scaling is preserved:

- :func:`unrotate_key_window` — one-time, at first demotion: strip RoPE off a
  window's fp keys (negated sin) at its original positions to get pre-RoPE codes.
- :func:`rotate_key_window` — every read: apply RoPE to dequantized Q keys at
  the window's original positions.

**The negated-sin inverse is only an inverse for a unit rotation.** Some RoPE
variants fold a scalar *attention scaling* ``a`` into cos/sin — YaRN sets
``a = 0.1·ln(factor) + 1`` (1.1386 at ``factor=4``) and the rotary module returns
``a·cos θ`` / ``a·sin θ``. Then the forward map is ``a·R(θ)``, not ``R(θ)``, and
applying ``(cos, −sin)`` to a post-RoPE key gives ``a·R(−θ)·a·R(θ)·k = a²·k`` —
the pre-RoPE key inflated by ``a²``. Re-rotating that on the read path yields
``a³·R(θ)·k`` against a true ``a·R(θ)·k``, so **every int4 Q-tier key and every
Q→fp promotion would be 1.296× too large** at YaRN ``factor=4``, silently
skewing the attention logits of the entire quantized tier.

:func:`rope_attention_scaling` reads ``a`` off the module and
:func:`unrotate_key_window` divides it out. ``a == 1`` for ``rope_type`` default
/ linear / dynamic / **llama3** (Llama-3.1's), so the fp16 and Llama/Mistral
paths stay byte-identical — the correction is a no-op branch there, not a
rescaled one. :func:`_rope_cos_sin` cross-checks ``cos²+sin² ≈ a²`` once per
process so a variant that scales cos/sin without exposing ``attention_scaling``
is reported instead of silently corrupting the tier.

Both RoPE primitives accept either an unbatched ``[H_kv, T, D]`` window with
``[T]`` positions or a batched ``[B, H_kv, T, D]`` with ``[B, T]``. The unbatched
form is exactly the batched one at ``B = 1`` — it just unsqueezes — so the two
are bit-identical, not merely equivalent. Everything here is shape-agnostic to
head count.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor

from .quantizer import dequantize_key_windows


def _apply_rotary():
    """Lazily import the model's ``apply_rotary_pos_emb`` (Llama, else Qwen2)."""
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except ImportError:  # pragma: no cover - depends on installed model families
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    return apply_rotary_pos_emb


# ---------------------------------------------------------------------------
# Fused dequant → RoPE for the Q-tier read path (design.md §8, "Phase 2")
# ---------------------------------------------------------------------------
#
# The per-step read path dequantizes every active int4 key window and RoPE's it
# (:meth:`QuantizedStore.effective_q_tier`). Eager, that is a chain of ~20 small
# elementwise kernels per layer per step — unpack, upcast to fp32, affine
# rescale, cast back, then the rotate_half/mul/add of RoPE — each launched
# separately and each writing its fp32 intermediate out to HBM only for the next
# kernel to read back. The whole chain is elementwise (plus shape-only
# reshape/permute), so fusing it is numerically safe: torch.compile lowers it to
# a single kernel that keeps the fp32 temporaries in registers.
#
# Opt-in via ``STICKYKV_COMPILE_READ`` (default off). The dev box has no GPU, so
# compiling in the CPU test suite only buys first-call Inductor latency and
# flakiness for no launch-count win; the perf runner sets it for GPU decode. The
# eager path below is the op-for-op sequence the store used before this change,
# so with the flag off the read path stays byte-identical.


def _compile_read_enabled() -> bool:
    return os.environ.get("STICKYKV_COMPILE_READ", "0").lower() in (
        "1", "true", "yes", "on",
    )


def _dequant_rotate_flat(
    k_codes: Tensor,
    k_scale: Tensor,
    k_zero: Tensor,
    window: int,
    cos: Tensor,
    sin: Tensor,
    out_dtype: torch.dtype,
    B: int,
    n: int,
    H: int,
    D: int,
    apply_rotary,
) -> Tensor:
    """Dequantize ``B*n`` key windows and RoPE them, as one elementwise chain.

    Mirrors the eager sequence in :meth:`QuantizedStore.effective_q_tier`
    op-for-op — affine dequant to ``out_dtype``, window-major → token-major
    permute, then ``apply_rotary_pos_emb`` with the model's own ``cos``/``sin`` —
    so it is bit-identical whether run eager or under ``torch.compile``. ``cos``/
    ``sin`` are passed in (position-only, value-independent) so the compiled
    region is pure tensor math with no module call to trace through.
    """
    keys_pre = dequantize_key_windows(
        k_codes, k_scale, k_zero, window, out_dtype=out_dtype
    )                                                        # [B*n, H, window, D]
    k_flat = (
        keys_pre.reshape(B, n, H, window, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(B, H, n * window, D)
    )
    _, k_rot = apply_rotary(k_flat, k_flat, cos, sin)
    return k_rot.to(out_dtype)


_COMPILED_READ_FN = None


def _read_fn():
    """Return the (optionally compiled) fused dequant→RoPE callable.

    Compiles once, lazily, with ``dynamic=True`` so the varying active-window
    count ``n`` does not force a recompile every eviction. Falls back to eager if
    compilation is unavailable or raises.
    """
    global _COMPILED_READ_FN
    if not _compile_read_enabled():
        return _dequant_rotate_flat
    if _COMPILED_READ_FN is None:
        try:
            _COMPILED_READ_FN = torch.compile(_dequant_rotate_flat, dynamic=True)
        except Exception:  # pragma: no cover - environment-dependent
            _COMPILED_READ_FN = _dequant_rotate_flat
    return _COMPILED_READ_FN


def dequant_rotate_q_keys(
    k_codes: Tensor,
    k_scale: Tensor,
    k_zero: Tensor,
    window: int,
    pos_flat: Tensor,
    rope_module: torch.nn.Module,
    out_dtype: torch.dtype,
    B: int,
    n: int,
    H: int,
    D: int,
) -> Tensor:
    """Read-path Q-tier keys: dequantize + RoPE, fused.

    Equivalent to ``rotate_key_window(dequant(...).reshape/permute, pos_flat)`` in
    the store, but routed through :func:`_dequant_rotate_flat` so the whole chain
    can be a single compiled kernel. ``cos``/``sin`` depend only on the positions
    and the ``out_dtype``/device (not the key values), so we build them from a
    tiny reference tensor — bit-identical to computing them from the dequantized
    keys as the eager path did.
    """
    ref = torch.empty(1, 1, 1, dtype=out_dtype, device=pos_flat.device)
    cos, sin = _rope_cos_sin(rope_module, ref, pos_flat)
    apply_rotary = _apply_rotary()
    fn = _read_fn()
    try:
        return fn(
            k_codes, k_scale, k_zero, window, cos, sin,
            out_dtype, B, n, H, D, apply_rotary,
        )
    except Exception:  # pragma: no cover - compiled-path fallback
        return _dequant_rotate_flat(
            k_codes, k_scale, k_zero, window, cos, sin,
            out_dtype, B, n, H, D, apply_rotary,
        )


def rope_attention_scaling(rope_module: torch.nn.Module) -> float:
    """The scalar ``a`` this rotary module folds into its cos/sin.

    ``1.0`` for ``rope_type`` default / linear / dynamic / llama3 — i.e. Llama-3.1,
    Mistral and un-scaled Qwen2.5 — so every caller's correction is a no-op branch
    on those and their numerics are byte-identical to before this existed.

    ``0.1·ln(factor) + 1`` for YaRN unless the config pins ``attention_factor``
    (transformers ``_compute_yarn_parameters``), and a config-supplied value for
    longrope. Read off the module rather than recomputed from the config so it
    tracks whatever transformers actually built.
    """
    a = getattr(rope_module, "attention_scaling", 1.0)
    try:
        a = float(a)
    except (TypeError, ValueError):
        return 1.0
    # A non-finite or non-positive scaling is not a rotation we can invert;
    # 1.0 keeps the legacy behaviour and the consistency check below will fire.
    if not (a > 0.0) or a != a or a in (float("inf"), float("-inf")):
        return 1.0
    return a


_warned_rope_scaling = [False]
# cos² + sin² is a² to within fp16 rounding of a squared magnitude; 2% is far
# below the 29.6% error a missed YaRN a² would produce and far above fp16 noise.
_ROPE_SCALING_TOL = 0.02


def _check_rope_scaling_consistency(cos: Tensor, sin: Tensor, a: float) -> None:
    """Warn once if ``cos²+sin²`` disagrees with ``a²`` from the module.

    The RoPE round-trip in this module inverts ``a·R(θ)`` analytically, so a
    module that scales cos/sin by something it does not expose as
    ``attention_scaling`` would corrupt every Q-tier key with nothing raising.
    One scalar comparison per process is a cheap way to make that loud.
    """
    if _warned_rope_scaling[0]:
        return
    try:
        measured = float((cos[..., :1] ** 2 + sin[..., :1] ** 2).flatten()[0])
    except (IndexError, RuntimeError):        # pragma: no cover - degenerate shapes
        return
    expected = a * a
    if expected <= 0 or abs(measured - expected) <= _ROPE_SCALING_TOL * expected:
        return
    _warned_rope_scaling[0] = True
    warnings.warn(
        f"RoPE module reports attention_scaling={a} (a^2={expected:.4f}) but its "
        f"cos^2+sin^2 is {measured:.4f}. The two-tier read path inverts RoPE "
        f"analytically as a*R(theta), so a scaling this module does not expose "
        f"would inflate every dequantized int4 key by the mismatch. Q-tier "
        f"numerics are NOT trustworthy for this rope_type — check "
        f"modules.quant.effective.rope_attention_scaling against the installed "
        f"transformers before using these results.",
        RuntimeWarning,
        stacklevel=2,
    )


def _rope_cos_sin(
    rope_module: torch.nn.Module, ref: Tensor, position_range: Tensor
) -> Tuple[Tensor, Tensor]:
    """cos/sin for one window's positions. ``ref`` supplies device/dtype only."""
    pos = position_range.to(torch.long)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)  # [1, window]
    cos, sin = rope_module(ref, pos)
    _check_rope_scaling_consistency(cos, sin, rope_attention_scaling(rope_module))
    return cos, sin


def unrotate_key_window(
    key_post_rope: Tensor,
    position_range: Tensor,
    rope_module: torch.nn.Module,
) -> Tensor:
    """Strip RoPE from a window's fp keys (design §5, one-time demotion).

    ``cos(−θ)=cos θ``, ``sin(−θ)=−sin θ`` — undo the rotation at the window's
    *original absolute positions*.

    Exact inverse of :func:`rotate_key_window` for **any** ``attention_scaling``:
    the forward map is ``a·R(θ)``, so applying ``(cos, −sin)`` lands on ``a²·k``
    and the ``a²`` is divided back out. At ``a == 1`` (Llama-3.1's llama3 rope,
    Mistral, un-scaled Qwen2.5) the division is skipped entirely, so those paths
    are byte-identical rather than merely equivalent.

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
    a = rope_attention_scaling(rope_module)
    if a != 1.0:
        k_un = k_un / (a * a)
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
    (eff_k, eff_v, score_meta) : the effective K/V and a per-window scatter map.

    ``eff_k``/``eff_v`` are each ``[B, H_kv, T_total, D]`` where
    ``T_total = T_fp + T_q``, laid out **unsorted** as ``[sink ‖ fp body ‖ Q]``
    (see below). ``T_total`` is the same for every row: the tier split keeps
    exactly ``k_fp``/``n_q`` windows and both derive from config + ``W``, which
    are shared across rows — so divergent eviction stays **rectangular** and the
    effective K/V needs no padding or keep-mask (BATCHING_PLAN.md §3).

    ``score_meta`` is ``(order, q_token_len)`` — the input the window scorer
    needs to undo the unsorted layout on the (tiny) score axis — or ``None`` when
    the Q tier is empty (the fast path returns the fp store, and its post-sink
    axis is already ascending-id, so no scatter is needed).
    """
    if out_dtype is None:
        out_dtype = fp_keys.dtype

    # Fast path: empty Q tier ⇒ effective K/V is the fp store, byte-identical.
    if store.num_active_windows == 0:
        return fp_keys, fp_values, None

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

    # Concatenate the two tiers UNSORTED: [sink ‖ fp body ‖ Q]. Attention is
    # order-free (RoPE bakes each key's absolute position into its value), so it
    # does not care that body and Q are not globally id-sorted — which lets us
    # skip the per-token argsort + two [B, H, T_total, D] gathers the sorted
    # layout used to pay every decode step. That reorder was never for attention;
    # it was only so the window scorer, which chunks the physical key axis into
    # window-size groups, saw windows in chronological id order. We hand that
    # reorder to the scorer instead as a per-WINDOW permutation (~W entries, on
    # the [B, H, W] score axis) rather than a per-TOKEN one on the K/V tensors.
    eff_k = torch.cat([sink_k, body_k, k_q], dim=2)
    eff_v = torch.cat([sink_v, body_v, v_q], dim=2)

    # Score-scatter map. Every window is wholly one tier, and within a tier the
    # windows are already ascending by id and window-size-aligned (fp body: the
    # eviction rebuild lays it out [sink ‖ windows by ascending id]; Q: the store
    # is window-major over active_order, ascending id). So the id at each
    # window's start token, tier-concatenated, is the physical per-window id
    # sequence, and ``argsort`` of it is exactly the placement that puts each
    # tier's per-window score back in ascending-merged-id order — the order
    # ``state.window_scores`` / ``original_window_ids`` are kept in. A stable
    # sort is not required (window ids are unique) but costs nothing.
    ws = window_size
    body_win_ids = (body_pos[:, ::ws].to(torch.long) - num_sink) // ws   # [B, n_body]
    q_win_ids = (q_pos_flat[:, ::ws].to(torch.long) - num_sink) // ws    # [B, n_q]
    phys_win_ids = torch.cat([body_win_ids, q_win_ids], dim=1)           # [B, W]
    order = torch.argsort(phys_win_ids, dim=1)                           # [B, W]
    return eff_k, eff_v, (order, q_pos_flat.shape[1])
