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

Both RoPE primitives accept either an unbatched ``[H_kv, T, D]`` window with
``[T]`` positions or a batched ``[B, H_kv, T, D]`` with ``[B, T]``. The unbatched
form is exactly the batched one at ``B = 1`` — it just unsqueezes — so the two
are bit-identical, not merely equivalent. Everything here is shape-agnostic to
head count.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor

from .quantizer import QGrid, dequantize_key_windows


def _apply_rotary():
    """Lazily import the model's ``apply_rotary_pos_emb`` (Llama, else Qwen2)."""
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except ImportError:  # pragma: no cover - depends on installed model families
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    return apply_rotary_pos_emb


def _rotate_half():
    """HF's own ``rotate_half``, from the same module as ``apply_rotary_pos_emb``.

    Imported rather than restated so the RoPE convention stays HF's. The only
    thing :func:`_apply_rotary_one` changes is *how many tensors get rotated*,
    never which halves go where or with what sign.

    **Not used on the compiled eviction path** -- see :func:`_rotate_half_no_cat`,
    which is bit-identical and is what that path calls. This stays as the
    reference the equality test compares against.
    """
    try:
        from transformers.models.llama.modeling_llama import rotate_half
    except ImportError:  # pragma: no cover - depends on installed model families
        from transformers.models.qwen2.modeling_qwen2 import rotate_half
    return rotate_half


def _rotate_half_no_cat(x: Tensor) -> Tensor:
    """``rotate_half`` without ``torch.cat``. Bit-identical; see why it must be.

    HF writes the rotation as ``torch.cat((-x2, x1), dim=-1)``. That one line is
    what made every GPU perf cell ERROR with

        ValueError: The argument '((I)//8)' is not comparable.

    and the message is not about shapes at all, which is why two rounds of
    forcing ``int()`` on ``.shape`` reads changed nothing. Decoded:

    * ``I`` is **not** the imaginary unit and not a shape symbol. It is
      ``torch.utils._sympy.functions.Identity`` -- Inductor's "do not expand
      this" wrapper -- printed as ``I`` because sympy's ``StrPrinter`` has a
      ``_print_Identity`` written for the identity *matrix* and dispatches on the
      class *name*. So ``((I)//8)`` is ``FloorDiv(Identity(<expr>), ws)``.
    * Inductor creates that wrapper in exactly one place:
      ``_inductor/lowering.py::pointwise_cat``, as
      ``Identity(idx[dim] - inputs_ranges[i][0])`` -- the shifted index of a
      ``torch.cat`` lowered as a fused pointwise read. **A cat is the only way to
      get one.**
    * It blows up in ``_simplify_loops`` -> ``stride_vars``, which removes the
      offset by substituting every index var with 0. ``Identity(i - h)`` then
      becomes ``Identity(-h)`` -- no free symbols -- and sympy rebuilds the
      enclosing ``Min``/``Max``, whose ``_new_args_filter`` rejects any arg that
      ``is_number`` but not ``is_comparable``. ``Identity`` is a ``Function``
      sympy cannot evaluate, so it is exactly that.

    Two consequences worth keeping in mind, because both were got wrong before:

    * ``is_number`` is True on the failing argument, so the value was **already
      concrete**. A symbolic shape is not what this is, and no amount of
      ``int()`` can fix it. It is also why the ``dynamic=False`` retry failed
      with the identical message instead of a different one.
    * It cannot reproduce on CPU no matter what is compiled, because
      ``lowering.py::cat`` returns a ``ConcatKernel`` unconditionally for CPU
      devices ("negative performance impact of pointwise_cat on CPU") and so
      never builds an ``Identity`` at all. The CPU/GPU split is that branch, not
      anything about the C++ backend tolerating symbolic ints.

    The rotation itself is pure index motion, so it does not need a cat:
    view the last axis as ``[2, h]`` (half 0 = ``x1``, half 1 = ``x2``), ``flip``
    that axis to get ``[x2, x1]``, and scale by ``[-1, +1]``. ``flip`` lowers to
    an index transform, not a concatenation, so no ``Identity`` is created.

    Bit-identity, not approximate equality: ``x * -1`` flips the sign bit and
    ``x * 1`` is the identity, for every finite, infinite and subnormal value in
    every float dtype, so each output element is the same bit pattern HF's
    ``cat((-x2, x1))`` produces. ``test_rotate_half_no_cat_is_bit_identical``
    pins it against HF directly.
    """
    # int(): a concrete half-width, for the same reason the rest of the compiled
    # region forces its shape reads. HF writes `x.shape[-1] // 2` raw.
    h = int(x.shape[-1]) // 2
    # [..., 2, h] -- reshape, not unflatten, so a non-contiguous caller is handled.
    u = x.reshape(*x.shape[:-1], 2, h)
    sign = torch.tensor([[-1.0], [1.0]], dtype=x.dtype, device=x.device)
    return (u.flip(-2) * sign).reshape(x.shape)


def _apply_rotary_one(k: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """RoPE **one** tensor, in place where it is safe to.

    ``apply_rotary_pos_emb(q, k, cos, sin)`` rotates two tensors and returns
    both. Every caller here wants only the key, and passed the key as *both*
    arguments -- so the function computed the identical result twice, threw one
    copy away, and peaked at roughly five key-sized buffers to produce one.
    At 2048/batch-32 the demote path asked for 3.62 GiB in exactly that spot and
    the allocator refused (``cache.py`` demote, first eviction, full width).

    This computes HF's expression once:

        (k * cos) + (rotate_half(k) * sin)

    with ``rotate_half`` imported from HF, and with the two adds done in place
    on tensors this function allocated, so the peak is the input plus two
    buffers instead of five. Every operation and its order is unchanged, so the
    result is bit-for-bit what ``apply_rotary_pos_emb(k, k, cos, sin)[1]``
    returns -- which is asserted directly in
    ``test_apply_rotary_one_matches_huggingface``.
    """
    cos = cos.unsqueeze(1)    # HF's unsqueeze_dim default: the head axis
    sin = sin.unsqueeze(1)
    out = k * cos                         # allocation 1 (ours; safe in place)
    # `_rotate_half_no_cat`, NOT HF's `rotate_half`: bit-identical, and free of
    # the `torch.cat` whose Inductor lowering emitted the `Identity` node that
    # made every compiled eviction fail to lower. The whole derivation is in
    # that function's docstring; this call site is the only reason it exists.
    tmp = _rotate_half_no_cat(k) * sin    # allocation 2 (ours)
    return out.add_(tmp)


# ---------------------------------------------------------------------------
# Fused dequant → RoPE for the Q-tier read path (design.md §8, "Phase 2")
# ---------------------------------------------------------------------------
#
# The per-step read path dequantizes every active int2 key window and RoPE's it
# (:meth:`QuantizedStore.effective_q_tier`). Eager, that is a chain of ~20 small
# elementwise kernels per layer per step — unpack, upcast to fp32, affine
# rescale, cast back, then the rotate_half/mul/add of RoPE — each launched
# separately and each writing its fp32 intermediate out to HBM only for the next
# kernel to read back. The whole chain is elementwise (plus shape-only
# reshape/permute), so fusing it is numerically safe: torch.compile lowers it to
# a single kernel that keeps the fp32 temporaries in registers.
#
# This chain no longer runs in production, and the torch.compile fork that used
# fused two-tier decode kernel hands raw int2 straight to Triton and dequantizes
# inside the kernel, so ``dequant_rotate_q_keys`` is unreachable there by two
# independent routes: ``update()`` only falls through to ``_materialize_joint``
# when the Q tier is EMPTY, and ``effective_q_tier`` returns ``None`` on an empty
# tier before reaching this. The flag was set to 1 on every CUDA perf run and
# printed a banner naming a "decode read path" that the run was not taking.
#
# What is left is the CPU reference: ``_materialize_joint`` on a box with no
# fused kernel builds the effective K through here, op for op, and every
# accuracy test validates against it. That is worth keeping and is not worth
# compiling -- on CPU a compile buys first-call Inductor latency and nothing else.


def _dequant_rotate_flat(
    k_codes: Tensor,
    k_scale: QGrid,
    k_zero: QGrid,
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


def dequant_rotate_q_keys(
    k_codes: Tensor,
    k_scale: QGrid,
    k_zero: QGrid,
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
    the store. ``cos``/``sin`` depend only on the positions
    and the ``out_dtype``/device (not the key values), so we build them from a
    tiny reference tensor — bit-identical to computing them from the dequantized
    keys as the eager path did.
    """
    ref = torch.empty(1, 1, 1, dtype=out_dtype, device=pos_flat.device)
    cos, sin = _rope_cos_sin(rope_module, ref, pos_flat)
    apply_rotary = _apply_rotary()
    return _dequant_rotate_flat(
        k_codes, k_scale, k_zero, window, cos, sin,
        out_dtype, B, n, H, D, apply_rotary,
    )


def rope_attention_scaling(rope_module: torch.nn.Module) -> float:
    """The scalar ``a`` this rotary module folds into its cos/sin.

    ``1.0`` for ``rope_type`` default / linear / dynamic / **llama3** — i.e.
    Llama-3.1, Mistral and un-scaled Qwen2.5 — so every caller's correction is a
    no-op branch on those and their numerics are byte-identical to before this
    existed.

    ``0.1·ln(factor) + 1`` for **YaRN** (1.13862 at factor 4.0) unless the config
    pins ``attention_factor`` — transformers folds it into cos/sin so the forward
    map is ``a·R(θ)``, not ``R(θ)``. Read off the module rather than recomputed
    from the config so it tracks whatever transformers actually built.
    """
    a = getattr(rope_module, "attention_scaling", 1.0)
    try:
        a = float(a)
    except (TypeError, ValueError):
        return 1.0
    # A non-finite or non-positive scaling is not a rotation we can invert; 1.0
    # keeps the legacy behaviour and the consistency check below will fire.
    if not (a > 0.0) or a != a or a in (float("inf"), float("-inf")):
        return 1.0
    return a


_warned_rope_scaling = [False]
# cos² + sin² == a² to within fp16 rounding; 2% is far below the 29.6% error a
# missed YaRN a² would produce and far above fp16 noise.
_ROPE_SCALING_TOL = 0.02


def _check_rope_scaling_consistency(cos: Tensor, sin: Tensor, a: float) -> None:
    """Warn once if ``cos²+sin²`` disagrees with ``a²`` from the module.

    The RoPE round-trip here inverts ``a·R(θ)`` analytically, so a module that
    scales cos/sin by something it does not expose as ``attention_scaling`` would
    corrupt every dequantized int2 key with nothing raising. One scalar
    comparison per process is a cheap way to make that loud.
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
        f"would inflate every dequantized int2 key by the mismatch. Q-tier "
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

    Parameters
    ----------
    key_post_rope : ``[H_kv, window, D]`` or ``[B, H_kv, window, D]``
        Post-RoPE keys from the fp store.
    position_range : ``[window]`` or ``[B, window]`` int64
        The window's original positions.
    """
    batched = key_post_rope.dim() == 4
    k = key_post_rope if batched else key_post_rope.unsqueeze(0)
    cos, sin = _rope_cos_sin(rope_module, k, position_range)
    k_un = _apply_rotary_one(k, cos, -sin)
    # The negated-sin pass is an exact inverse only for a UNIT rotation. When the
    # module folds a scalar `a` into cos/sin (YaRN: a = 0.1·ln(factor)+1), the
    # forward map is a·R(θ), so (cos, −sin) lands on a²·k — divide the a² back
    # out so the stored codes are the true pre-RoPE key. At a == 1 (Llama-3.1's
    # llama3 rope, Mistral, un-scaled Qwen2.5) this branch is SKIPPED, so those
    # paths stay byte-identical. Without it every int2 Q-tier key would be 1.296×
    # too large at YaRN factor 4, silently. See `rope_attention_scaling`.
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
    batched = key_pre_rope.dim() == 4
    k = key_pre_rope if batched else key_pre_rope.unsqueeze(0)
    cos, sin = _rope_cos_sin(rope_module, k, position_range)
    k_rot = _apply_rotary_one(k, cos, sin)
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


def compute_score_meta(
    fp_positions: Tensor,
    q_pos_flat: Tensor,
    num_sink: int,
    window_size: int,
) -> Tuple[Tensor, int]:
    """The ``(order, q_token_len)`` scatter map the two-tier scorer needs.

    The physical effective-K order is ``[sink ‖ fp body ‖ Q]``; the running
    ``window_scores`` live in ascending merged-window-id order. ``order`` is the
    ``argsort`` of the physical per-window ids, so
    :func:`scorer.reduce_two_tier_scores` can scatter each tier's per-window sum
    back to merged-id order. Extracted so both the materialize path **and** the
    fused-decode path (which builds no effective tensor) produce identical meta.

    Parameters
    ----------
    fp_positions : ``[B, T_fp]`` original absolute positions of the fp store
        (sink prefix included; sliced off here).
    q_pos_flat : ``[B, n_q * ws]`` original absolute positions of the Q tier,
        window-major ascending (from :meth:`QuantizedStore.effective_q_tier`).
    """
    ws = window_size
    body_pos = fp_positions[:, num_sink:]
    body_win_ids = (body_pos[:, ::ws].to(torch.long) - num_sink) // ws   # [B, n_body]
    q_win_ids = (q_pos_flat[:, ::ws].to(torch.long) - num_sink) // ws    # [B, n_q]
    phys_win_ids = torch.cat([body_win_ids, q_win_ids], dim=1)          # [B, W]
    order = torch.argsort(phys_win_ids, dim=1)                          # [B, W]
    return order, q_pos_flat.shape[1]


def materialize_effective_kv(
    fp_keys: Tensor,
    fp_values: Tensor,
    fp_positions: Tensor,
    store,
    num_sink: int,
    window_size: int,
    rope_module: torch.nn.Module,
    out_dtype: torch.dtype = None,
    q_tier: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
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

    # Q windows: dequantized + RoPE'd at their frozen original positions.
    # RoPE is per-token pointwise, so rotating all N_q windows in ONE call
    # (flattened window-major) is bit-identical to N_q separate per-window calls
    # at a single kernel launch. The store may memoize this across steps — the
    # tier cannot change between evictions (§10).
    #
    # ``q_tier`` lets the caller supply that triple instead. The layer-major
    # decode store (DECODE_SPEED_PLAN §4.1) holds all L layers in one row axis,
    # so it dequantizes every layer's Q tier in ONE call and hands each layer its
    # row slice here — the same tensors ``effective_q_tier`` would have produced
    # for that layer, at 1/L the launches. Passing it is not optional plumbing:
    # a layer-major store has no per-layer view to call the method on.
    if q_tier is None:
        k_q, v_q, q_pos_flat = store.effective_q_tier(rope_module, out_dtype)
    else:
        k_q, v_q, q_pos_flat = q_tier

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
    order, q_token_len = compute_score_meta(
        fp_positions, q_pos_flat, num_sink, window_size
    )
    return eff_k, eff_v, (order, q_token_len)
