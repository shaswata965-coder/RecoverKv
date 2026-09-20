"""Fused two-tier decode attention + eviction score (design.md §11 Phase 2).

What this replaces
------------------
Phase 1 decode reads the Q tier by dequantizing every active int2 window to
fp16, RoPE-ing it, and concatenating it with the fp tier into one effective K/V
that the model attends over (``materialize_effective_kv``); a separate score pass
recomputes ``softmax(q·kᵀ)`` for the eviction scores. Four per-step costs, every
layer (design §8): the fp16 write-back of the Q tier, the concat, a full-budget
fp16 attention read, and a **duplicate** ``q·kᵀ`` for the score.

This kernel does all of it in one launch, reading the Q tier **as int2** and
dequantizing + RoPE-ing **in registers** — no fp16 tier is written to HBM, no
concat, and the eviction score falls out of the same ``q·kᵀ`` (for a single
decode query the per-key softmax weight IS the H2O score). Nothing is held in
fp16, so the read-memo can stay off at ``B>1`` with no penalty — max-B AND speed.

Fast-decode mechanics folded in (flash-decoding)
------------------------------------------------
* **GQA load-once.** One program owns one ``(batch, KV head)`` and handles all
  ``rep = num_groups`` query heads sharing it, so each key/value tile — and each
  int2 dequant — is read/done **once per KV head**, not once per query head
  (``num_groups``× less traffic; 4× on Llama-3.1-8B). Hence grid ``B·H_kv``.
* **Online softmax** with a running max — the ``[S]`` score row is never held;
  a second pass turns the stored logits into normalized per-key scores.
* **Tiled ``tl.dot``** over the fp tier (``BLOCK_N`` keys) and the Q tier (one
  window, padded to ``BLOCK_WS``); query heads padded to ``BLOCK_R``.
* **RoPE without a rotate gather.** Load the two head-dim halves separately and
  apply ``k_lo·c − k_hi·s`` / ``k_hi·c + k_lo·s`` from the frozen per-position
  ``cos``/``sin`` halves — pure register math (HF stores ``cos=cat(h,h)``, so the
  first half is the distinct part).
* (Not yet: split-K over the sequence for occupancy when ``B·H_kv`` is small — a
  tuning follow-up: partial ``(out, lse)`` per KV-split + a combine.)

Backend contract (mirrors :mod:`score_kernel`)
----------------------------------------------
* :func:`two_tier_decode_reference` — pure-PyTorch oracle over an *already
  effective* K/V. Defines correctness (output **and** score); runs on CPU; is the
  Triton kernel's test oracle (build ``k_eff`` with ``materialize_effective_kv``
  on the same store, then compare).
* ``_two_tier_decode_kernel`` + :func:`_decode_triton` — the fused GPU kernel.
  **GPU-only, ships unvalidated by construction** (CPU-only dev box), like
  ``score_kernel._token_scores_triton``.
* :func:`fused_two_tier_decode` — dispatcher: **Triton-or-raise**. No PyTorch
  decode fallback in production; the flash backend is gated to refuse to run
  without the kernel (:func:`assert_decode_kernel_available`).
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
from torch import Tensor

from modules.quant.quantizer import grid_group

try:  # pragma: no cover - import guard, exercised only where triton is present
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # ImportError, or a broken triton build
    _HAS_TRITON = False


_LOG2E = 1.4426950408889634
"""``log2(e)``. Folded into ``scale`` so the kernel's softmax runs in base 2."""


#: The kernel's two numeric conventions, both fixed. Base-2 softmax: log2(e) is
#: folded into `scale`, so every logit is already a base-2 exponent and the base
#: change costs nothing per element. RoPE tables at the KV STORE's dtype, not
#: fp32: every other RoPE in this cache runs at the store dtype, and an fp32
#: table here made a token's key depend on which tier held it (max round-trip
#: error 4.5e-01 -> 6.9e-04). Both were environment knobs; neither had a correct
#: "off", so neither is a knob.
_EXP2 = [True]
_ROPE_STORE_DTYPE = [True]


def fused_decode_enabled() -> bool:
    """Always ``True``. There is one decode on the flash backend.

    quietly served by the materialize path instead — a second, slower, entirely
    different implementation of the same method, selected by an environment
    variable nobody reads back when quoting a number.

    What decides the path now is the machine, not a setting: the score hook
    computes ``fused_decode_enabled() and cuda``, so CUDA runs the fused gated
    kernel and CPU runs :meth:`WindowedCache._materialize`. The materialize path
    is kept for exactly that — it is the CPU oracle 13 test files check the
    kernel against, and without it nothing here is verifiable off a GPU. It is
    not reachable as a production alternative.

    The varlen case (a padding ``attention_mask`` reaching
    ``_flash_attention_forward``) is the one thing that used to justify the
    switch. It still cannot be served fused, and it still fails loudly —
    :class:`~modules.windowed_cache.flash_decode.FusedDecodeNotReached` — rather
    than silently routing to different code. Use equal-length prompts.
    """
    return True


# ---------------------------------------------------------------------------
# Availability / gating
# ---------------------------------------------------------------------------


def decode_kernel_available(cuda: bool) -> bool:
    """True iff the fused decode Triton kernel can actually launch."""
    return bool(_HAS_TRITON and cuda)


def describe_decode_backend(cuda: bool) -> str:
    """Human string for the decode attention backend, given CUDA availability."""
    if not fused_decode_enabled():
        return "materialize (fused decode disabled via STICKYKV_FUSED_DECODE=0)"
    if decode_kernel_available(cuda):
        return "TRITON fused two-tier decode kernel (required)"
    reason = "triton not installed" if not _HAS_TRITON else "not running on CUDA"
    return f"WILL ERROR: fused decode needs the Triton kernel on CUDA ({reason})"


def assert_decode_kernel_available(cuda: bool) -> None:
    """Raise unless the fused decode kernel can launch (flash-backend gate).

    Choosing the flash backend commits to the Triton path for the decode kernel
    just as for the prefill score kernel, so a box that cannot launch it must fail
    at setup rather than silently on the first quantized decode step.
    """
    if not fused_decode_enabled():
        return
    if not decode_kernel_available(cuda):
        reason = "triton not installed" if not _HAS_TRITON else "not running on CUDA"
        raise RuntimeError(
            "The flash backend requires the fused two-tier decode Triton kernel "
            f"on CUDA, and it cannot launch ({reason}). This is the decode analog "
            "of the prefill score kernel's Triton-or-error contract: there is no "
            "PyTorch decode fallback. Install triton and run on GPU, use the eager "
            "backend, or set STICKYKV_FUSED_DECODE=0 to fall back to the Phase-1 "
            "materialize path."
        )


# ---------------------------------------------------------------------------
# RoPE halves for the Q positions (built on the host; passed to the kernel)
# ---------------------------------------------------------------------------


def rope_cos_sin_halves(
    rope_module: torch.nn.Module, pos_flat: Tensor,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """``cos``/``sin`` **first halves** ``[B, T, D//2]`` for the Q positions.

    HF builds ``cos``/``sin`` as ``cat(h, h)`` along head_dim, so the first half
    carries the distinct per-frequency values; the kernel applies RoPE from those
    halves directly (no rotate-half gather). Position-only, so this is cheap and
    holds no key data — the point of the fused path.

    ``dtype`` must be **the KV store's dtype**, and passing it is not optional
    tuning. HF's rotary returns ``cos.to(dtype=x.dtype)``, so every other RoPE in
    this cache runs in fp16: the model's own rotation of the fp tier, and
    ``unrotate_key_window`` / ``rotate_key_window``, which both hand
    ``_rope_cos_sin`` the fp16 *key* as their reference. This function used to
    hand it ``torch.empty(1, 1, 1)`` and so got fp32 — apparently incidentally,
    since nothing depended on it.

    That made the kernel the odd one out, and it broke a round trip: the Q tier is
    un-rotated in fp16, stored, then re-rotated here in fp32, so the key the
    kernel reconstructs is **not** the key that was demoted. Measured over 512
    positions, matching the dtype cuts the maximum round-trip error from 4.5e-01
    to 6.9e-04. It also halves this tensor's traffic (0.75 -> 0.38 GB/step at
    4096/batch-32) and its ``tl.dot`` operand footprint, but that is the side
    effect, not the reason.
    """
    want = dtype if (dtype is not None and _ROPE_STORE_DTYPE[0]) else torch.float32
    ref = torch.empty(1, 1, 1, device=pos_flat.device, dtype=want)
    pos = pos_flat.to(torch.long)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    cos, sin = rope_module(ref, pos)                       # [B, T, D]
    half = cos.shape[-1] // 2
    return cos[..., :half].contiguous(), sin[..., :half].contiguous()


# ---------------------------------------------------------------------------
# PyTorch reference / oracle — over an ALREADY-EFFECTIVE K/V
# ---------------------------------------------------------------------------


def two_tier_decode_reference(
    q: Tensor,
    k_eff: Tensor,
    v_eff: Tensor,
    scaling: float,
) -> Tuple[Tensor, Tensor]:
    """Single-query attention output **and** per-key eviction score (the oracle).

    For a single decode query the per-key softmax weight is the H2O score, so both
    outputs come from one softmax.

    q : ``[B, H_q, D]`` or ``[B, H_q, 1, D]`` post-RoPE query.
    k_eff, v_eff : ``[B, H_kv, S, D]`` effective K/V (``[sink ‖ body ‖ Q]``).
    Returns ``(out [B,H_q,D], token_scores [B,H_q,S])``.
    """
    if q.dim() == 3:
        q = q.unsqueeze(2)
    B, H_q, _, D = q.shape
    H_kv, S = k_eff.shape[1], k_eff.shape[2]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} not divisible by H_kv={H_kv}")
    rep = H_q // H_kv

    q5 = q.reshape(B, H_kv, rep, 1, D)
    k = k_eff.unsqueeze(2)
    v = v_eff.unsqueeze(2)
    logits = torch.matmul(q5, k.transpose(-2, -1)).float() * scaling
    p = torch.softmax(logits, dim=-1)
    out = torch.matmul(p.to(v.dtype), v)

    out = out.reshape(B, H_q, D)
    token_scores = p.reshape(B, H_q, S)
    return out.to(v_eff.dtype), token_scores.to(v_eff.dtype)


# ---------------------------------------------------------------------------
# Window-score emission (DECODE_SPEED_PLAN §5.1) — tiling + CPU oracle
# ---------------------------------------------------------------------------


def window_tiling(window_size: int, target_keys: int = 64) -> Tuple[int, int]:
    """``(BLOCK_NW, BLOCK_T)`` — how many windows per tile, padded to a power of 2.

    The kernel tiles the fp **body** and the Q tier in whole windows rather than
    in keys, which is what makes "every window lives entirely inside one tile"
    true for **any** ``window_size``, not just powers of two. ``tl.arange``
    requires a power-of-2 length, so a tile of ``BLOCK_NW * ws`` keys is padded up
    to ``BLOCK_T`` and the tail lanes are masked off.

    That padding is the entire cost of supporting a non-power-of-2 ``ws``:

    =====  ========  =========  =========
    ``ws``  BLOCK_NW   BLOCK_T   utilisation
    =====  ========  =========  =========
    1            64         64      100%
    8             8         64      100%
    12            5         64       94%
    128           1        128      100%
    =====  ========  =========  =========

    Tiling by windows also makes the ``num_sink`` offset a non-issue: the body
    loop starts at key ``num_sink`` (window 0's first token) and steps in window
    units, so tile boundaries land on window boundaries by construction. The sink
    prefix is handled by its own prologue tile, which contributes to the softmax
    but emits no window score — matching ``reduce_token_scores_to_windows``,
    which strips the sink before windowing.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    block_nw = max(1, target_keys // window_size)
    return block_nw, _pow2_at_least(block_nw * window_size, floor=1)


def check_gate_selection(
    sel: Optional[Tensor], B: int, H_kv: int, n_active: int,
) -> int:
    """Validate the gate's pick; return ``n_sel``, or ``0`` when ungated.

    Split out of :func:`_decode_triton` for the same reason :func:`window_tiling`
    is a function: the launch path cannot run on a CPU-only box, so anything left
    inside it ships untested. This is the contract that keeps a malformed
    selection an **error** rather than silently wrong output — a ``sel`` of the
    wrong shape, dtype or layout would not fail on a GPU, it would index the
    wrong windows and emit scores that still look exactly like probabilities.

    The kernel derives ``SEL``'s innermost stride as 1 and indexes it with
    ``b * selb + kv * selh + slot``, so contiguity is a checked contract here,
    not an assumption there.
    """
    if sel is None or n_active == 0:
        return 0
    if sel.dim() != 3 or sel.shape[0] != B or sel.shape[1] != H_kv:
        raise RuntimeError(
            f"fused decode gate expects sel [B, H_kv, n_sel] = [{B}, {H_kv}, *]; "
            f"got {tuple(sel.shape)}.")
    n_sel = int(sel.shape[-1])
    if not 0 < n_sel <= n_active:
        raise RuntimeError(
            f"fused decode gate selected {n_sel} of {n_active} active windows; it "
            "must pick at least one and no more than the tier holds. A larger "
            "n_sel means sel was built against a different store version than "
            "qtier, which would index past the gathered codes.")
    if sel.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            f"fused decode gate requires an int32 or int64 sel, got {sel.dtype}.")
    if not sel.is_contiguous():
        raise RuntimeError(
            "fused decode requires a contiguous sel; its innermost stride is "
            "assumed to be 1 inside the kernel.")
    return n_sel


def two_tier_window_reference(
    q: Tensor,
    k_eff: Tensor,
    v_eff: Tensor,
    scaling: float,
    num_sink: int,
    window_size: int,
    n_body_win: int,
    Sfp: Optional[int] = None,
    exp2: bool = False,
    sel: Optional[Tensor] = None,
    logmass: Optional[Tensor] = None,
    centroids: Optional[Tensor] = None,
    splits: int = 1,
) -> Tuple[Tensor, Tensor]:
    """CPU oracle for the kernel's **windowed** output, tile for tile.

    This is deliberately not a shortcut. It reproduces the kernel's actual
    algorithm — sink prologue, window-tiled body, window-tiled Q tier, the online
    softmax's running max, per-tile window sums taken against that running max,
    and the epilogue that rescales each window by ``exp(m_tile - lse)`` — so that
    the parts of §5.1 that cannot be executed on a CPU-only box (the Triton
    lowering) are the *only* parts left unverified. The arithmetic, the tiling,
    ``tests/test_window_scores.py`` against the plain
    ``softmax -> strip sink -> window-sum`` path.

    ``k_eff``/``v_eff`` are the effective ``[sink ‖ body ‖ Q]`` store, exactly
    what ``two_tier_decode_reference`` takes. ``Sfp`` is where the fp tier ends
    and the Q tier begins; it defaults to "all of it" (no Q tier).

    ``sel`` is the **gate's** selection: ``[B, H_kv, n_sel]`` int, the active Q
    columns this step actually reads, ascending. ``None`` means read the whole
    tier, which is the ungated path and leaves every line below unchanged.

    When ``sel`` is given, ``k_eff``/``v_eff`` still carry the **full** Q tier —
    the gate skips *reads*, it does not shrink the store — so the Q region is
    still ``n_active`` windows and ``W_phys`` is still ``n_body_win + n_active``.
    The loop then runs over ``n_sel`` physical slots and dereferences each
    through ``sel``, which is exactly the kernel's ``widx = tl.load(SEL + …)``.
    Two consequences the kernel shares and this models:

    * A window's score lands on column ``n_body_win + sel[b, kv, j]``, its own
      column, not on the ``j``-th one. Getting this wrong would credit a read
      window's mass to a skipped one — plausible-looking scores on the wrong
      windows, which no shape check would catch.
    * A skipped column is never written, so it must be *pre-set* rather than
      left at whatever the buffer held: the epilogue rescales every column it
      reads. Here they start at ``0``/``-inf``; the kernel stores the same
      sentinel in a prologue (see its ``GATED`` block).

    Returns ``(out [B,H_q,D], wsum [B,H_q,W_phys])`` where ``W_phys`` is
    ``n_body_win + n_q_win`` in **physical** order — body windows then Q windows,
    which is the order ``scorer.reduce_two_tier_scores`` scatters from.
    """
    if q.dim() == 3:
        q = q.unsqueeze(2)
    B, H_q, _, D = q.shape
    H_kv, S = k_eff.shape[1], k_eff.shape[2]
    rep = H_q // H_kv
    ws = window_size
    block_nw, _ = window_tiling(ws)

    # The fp store's REAL length. Deriving it as num_sink + n_body_win*ws would
    # overrun into the Q tier whenever the newest body window is partial, which
    # is the common case. The kernel masks the body at Sfp for the same reason.
    body_end = S if Sfp is None else min(Sfp, S)
    n_q_win = -(-(S - body_end) // ws)
    W_phys = n_body_win + n_q_win

    q5 = q.reshape(B, H_kv, rep, 1, D).float()
    k = k_eff.unsqueeze(2).float()
    # `exp2=True` mirrors the kernel's base-2 softmax: log2(e) folds into the
    # scale, so every logit becomes a base-2 exponent and m / wmax / lse are all
    # in base-2 units. p, l, acc and out are unchanged quantities either way.
    _e = math.exp2 if exp2 else math.exp
    scale_eff = scaling * _LOG2E if exp2 else scaling
    logits = torch.matmul(q5, k.transpose(-2, -1)) * scale_eff    # [B,H_kv,rep,1,S]
    logits = logits.reshape(B, H_q, S)
    v_flat = (v_eff.reshape(B, H_kv, 1, S, D)
              .expand(B, H_kv, rep, S, D).reshape(B, H_q, S, D).float())

    NEG = float("-inf")
    m = torch.full((B, H_q), NEG, dtype=torch.float32)
    l = torch.zeros((B, H_q), dtype=torch.float32)
    acc = torch.zeros((B, H_q, D), dtype=torch.float32)
    wsum = torch.zeros((B, H_q, W_phys), dtype=torch.float32)
    wmax = torch.full((B, H_q, W_phys), NEG, dtype=torch.float32)

    _, BLOCK_T = window_tiling(ws)
    offs_t = torch.arange(BLOCK_T)
    t_win = offs_t // ws                                   # window index in tile
    t_tok = offs_t % ws                                    # token index in window
    in_tile = offs_t < (block_nw * ws)                     # masks the pow2 padding

    def tile(key0: int, end: int, win_base: int, n_win_limit: int) -> None:
        """One tile, **including the power-of-2 padding lanes the kernel has**.

        This is deliberately not the tidy "slice ``[lo, hi)``" version. A Triton
        block must be a power of two, so a tile of ``BLOCK_NW * ws`` keys is
        materialised as ``BLOCK_T >= BLOCK_NW * ws`` lanes and the surplus is
        masked. Modelling the exact slice instead would verify the tiling while
        silently assuming the masking — and the masking is the part that could
        fold padding lanes into a window and quietly change what a "window" sums.

        Two independent things keep padding out of the window sums, and this
        function reproduces both so the tests exercise them:

        1. ``nmask`` zeroes every padding lane's ``p``, so it contributes nothing
           to the running sum, to ``acc``, or to any window.
        2. A padding lane has ``offs_t >= BLOCK_NW*ws``, hence
           ``t_win = offs_t // ws >= BLOCK_NW``, while the per-window loop only
           runs ``j`` over ``0 … BLOCK_NW-1``. So no padding lane is reachable by
           any window's selector even if (1) were removed.

        ``win_base < 0`` means "softmax only" — the sink prologue, which feeds
        the LSE and emits no window score.
        """
        nonlocal m, l, acc
        offs_n = key0 + offs_t
        nmask = in_tile & (offs_n < end) & (offs_n >= 0)
        if not bool(nmask.any()):
            return
        safe = offs_n.clamp(0, S - 1)
        lg = logits[:, :, safe]                            # [B,H_q,BLOCK_T]
        lg = torch.where(nmask, lg, torch.full_like(lg, float("-inf")))
        m_new = torch.maximum(m, lg.max(dim=-1).values)
        corr = (torch.exp2 if exp2 else torch.exp)(m - m_new)
        corr = torch.nan_to_num(corr, nan=0.0)             # first tile: -inf - -inf
        p = torch.where(nmask, (torch.exp2 if exp2 else torch.exp)(lg - m_new.unsqueeze(-1)),
                        torch.zeros_like(lg))
        acc = acc * corr.unsqueeze(-1) + torch.matmul(
            p.unsqueeze(-2), v_flat[:, :, safe]).squeeze(-2)
        l = l * corr + p.sum(dim=-1)
        m = m_new
        if win_base < 0:
            return
        for j in range(block_nw):                          # mirrors tl.static_range
            w = win_base + j
            if w - win_base >= n_win_limit or w >= W_phys:
                continue
            sel = (t_win == j) & nmask
            if not bool(sel.any()):
                continue
            wsum[:, :, w] = torch.where(sel, p, torch.zeros_like(p)).sum(dim=-1)
            wmax[:, :, w] = m_new

    def q_tile_gated(w0: int, sel_q: Tensor, n_sel: int) -> None:
        """One Q-tier tile **under SEL indirection** — the kernel's gated loop.

        Same online softmax as :func:`tile`, but the lane -> key map goes through
        ``sel`` instead of being ``key0 + offs_t``. Every lane of a tile is
        resolved independently, per ``(row, KV head)``, because the gate selects
        per KV head (see ``QuantizedStore.gated_q_tier`` for why the union or a
        row-level top-k will not do).
        """
        nonlocal m, l, acc
        slot = w0 + t_win                                  # physical slot in sel
        live = in_tile & (slot < n_sel)                    # [BLOCK_T]
        if not bool(live.any()):
            return
        # Clamp rather than mask the dereference: `live` already decides what
        # counts, and a clamped index keeps every derived offset (the key, and in
        # the kernel the RoPE row) inside the tier. The kernel clamps for the
        # same reason.
        col = sel_q[:, :, slot.clamp(0, n_sel - 1)]        # [B,H_q,BLOCK_T]
        idx = body_end + col * ws + t_tok                  # key index per lane
        nmask = live.expand_as(idx) & (idx < S) & (idx >= body_end)
        safe = idx.clamp(0, S - 1)

        lg = torch.gather(logits, -1, safe)                # [B,H_q,BLOCK_T]
        lg = torch.where(nmask, lg, torch.full_like(lg, float("-inf")))
        m_new = torch.maximum(m, lg.max(dim=-1).values)
        corr = (torch.exp2 if exp2 else torch.exp)(m - m_new)
        corr = torch.nan_to_num(corr, nan=0.0)
        p = torch.where(nmask, (torch.exp2 if exp2 else torch.exp)(
            lg - m_new.unsqueeze(-1)), torch.zeros_like(lg))
        vg = torch.gather(v_flat, 2, safe.unsqueeze(-1).expand(*safe.shape, D))
        acc = acc * corr.unsqueeze(-1) + torch.matmul(
            p.unsqueeze(-2), vg).squeeze(-2)
        l = l * corr + p.sum(dim=-1)
        m = m_new

        for j in range(block_nw):                          # mirrors tl.static_range
            if w0 + j >= n_sel:
                continue
            selm = (t_win == j) & nmask
            if not bool(selm.any()):
                continue
            # The j-th slot of THIS tile scores onto its own active column.
            wcol = (n_body_win + sel_q[:, :, w0 + j]).unsqueeze(-1)   # [B,H_q,1]
            wsum.scatter_(-1, wcol, torch.where(
                selm, p, torch.zeros_like(p)).sum(dim=-1, keepdim=True))
            wmax.scatter_(-1, wcol, m_new.unsqueeze(-1))

    sel_q = None
    n_sel = 0
    if sel is not None:
        if sel.shape[0] != B or sel.shape[1] != H_kv:
            raise ValueError(
                f"sel must be [B, H_kv, n_sel] = [{B}, {H_kv}, *], got "
                f"{tuple(sel.shape)}")
        n_sel = int(sel.shape[-1])
        if n_sel > n_q_win:
            raise ValueError(
                f"sel picks {n_sel} windows but the Q tier holds only {n_q_win}. "
                "The gate selects a subset of the tier it is handed; a larger "
                "selection means sel was built against a different store.")
        sel_q = sel.to(torch.long).repeat_interleave(H_q // H_kv, dim=1)

    # ---- SPLIT-KV: partition the WINDOW axis across `splits` programs --------
    #
    # Mirrors the kernel's phase 1. Each split owns a contiguous chunk of the
    # body windows and a contiguous chunk of the Q windows, accumulates its own
    # `(acc, l, m)`, and writes `wsum`/`wmax` for **its own windows only** --
    # which is why the partition is over windows and not over keys: a window
    # split across two programs would need its score reduced across them, and
    # every column here is written exactly once.
    #
    # The sinks go to split 0 alone. They emit no window score (the scorer
    # strips them before windowing), so they only have to land in exactly one
    # split's softmax, and splitting them would buy nothing -- `num_sink` is 5.
    #
    # Everything BELOW the merge is untouched by this. `out = acc / l` and the
    # whole epilogue need the GLOBAL lse and a full pass over `W_phys`, so they
    # stay where they are and run once, after the partials are combined. That
    # is the property that makes split-KV tractable on this kernel at all.
    for _split_bound, _name in ((splits, "splits"),):
        if not isinstance(_split_bound, int) or _split_bound < 1:
            raise ValueError(f"{_name} must be a positive int, got {_split_bound!r}")

    def _chunk(n: int, i: int, k: int) -> Tuple[int, int]:
        """Contiguous chunk ``i`` of ``n`` items over ``k`` splits.

        Remainder goes to the low splits, so chunk sizes differ by at most one
        and the partition is a partition -- every window in exactly one chunk.
        """
        base, rem = divmod(n, k)
        lo = i * base + min(i, rem)
        return lo, lo + base + (1 if i < rem else 0)

    parts = []
    for sp in range(splits):
        # Reset this split's running softmax. The closures mutate these through
        # `nonlocal`, so rebinding here is what gives each split its own state.
        # Constructed exactly as lines above do -- no `device=`, because this
        # oracle is CPU-side and the originals it mirrors take none.
        m = torch.full((B, H_q), NEG, dtype=torch.float32)
        l = torch.zeros((B, H_q), dtype=torch.float32)
        acc = torch.zeros((B, H_q, D), dtype=torch.float32)

        # 1. sink prologue -- softmax only, split 0 only
        if sp == 0:
            for s0 in range(0, max(num_sink, 0), BLOCK_T):
                tile(s0, num_sink, -1, 0)
        # 2. fp body -- this split's chunk of the body windows
        lo_b, hi_b = _chunk(n_body_win, sp, splits)
        for w0 in range(lo_b, hi_b, block_nw):
            tile(num_sink + w0 * ws, num_sink + hi_b * ws, w0, hi_b - w0)
        # 3. Q tier -- this split's chunk, through the gate's selection when given
        if sel is None:
            lo_q, hi_q = _chunk(n_q_win, sp, splits)
            for w0 in range(lo_q, hi_q, block_nw):
                tile(body_end + w0 * ws, min(S, body_end + hi_q * ws),
                     n_body_win + w0, hi_q - w0)
        else:
            lo_q, hi_q = _chunk(n_sel, sp, splits)
            for w0 in range(lo_q, hi_q, block_nw):
                q_tile_gated(w0, sel_q, hi_q)
        parts.append((acc, l, m))

    # ---- merge the partials: the standard log-sum-exp combine ---------------
    #
    # At `splits == 1` this is `acc, l, m = parts[0]` with one multiply by
    # exp(m - m) == 1, so it is a **provable no-op** -- the property that makes
    # `splits=1` the control arm for this change.
    if splits == 1:
        acc, l, m = parts[0]
    else:
        _exp = torch.exp2 if exp2 else torch.exp
        m = torch.stack([p[2] for p in parts], 0).amax(0)
        l = torch.zeros_like(parts[0][1])
        acc = torch.zeros_like(parts[0][0])
        for a_s, l_s, m_s in parts:
            # A split that saw no key has m_s == -inf and l_s == 0. exp(-inf -
            # m) is 0 for finite m, but -inf - -inf is NaN, so the empty-
            # everywhere case is selected away rather than computed.
            w = torch.where(m_s > NEG, _exp(m_s - m), torch.zeros_like(m_s))
            l = l + l_s * w
            acc = acc + a_s * w.unsqueeze(-1)

    out = acc / l.unsqueeze(-1)
    lse = m + (torch.log2 if exp2 else torch.log)(l)
    live = wmax > NEG
    wsum = torch.where(live, wsum * (torch.exp2 if exp2 else torch.exp)(wmax - lse.unsqueeze(-1)),
                       torch.zeros_like(wsum))

    # Skipped windows get their card's estimate, scaled onto the same footing as
    # the real scores — mirroring the kernel's GATED epilogue. `live` over the Q
    # columns IS the read set (a skipped column was never written, so its wmax is
    # still the -inf sentinel), which is why neither this nor the kernel consults
    # `sel` again. The offset in `fill_skipped_window_scores` cancels in the
    # ratio, so a max and a sum suffice and no logarithm is taken.
    if sel is not None and logmass is not None:
        read = live[..., n_body_win:]                          # [B,H_q,n_active]
        num = (wsum[..., n_body_win:] * read).sum(-1, keepdim=True)
        gmx = torch.where(read, logmass, torch.full_like(logmass, NEG)).amax(
            -1, keepdim=True)
        rel = (logmass - gmx).exp()
        gsm = (rel * read).sum(-1, keepdim=True).clamp_min(1e-30)
        fill = num * rel / gsm                                 # calibrated mass
        wsum = torch.cat([
            wsum[..., :n_body_win],
            torch.where(read, wsum[..., n_body_win:], fill),
        ], dim=-1)

        # The skipped windows also ATTEND, through their value centroids, at the
        # weight `fill` just measured. Mirrors the kernel's §5 block: a `tl.dot`
        # of the same tile, then one renormalization of `out` and of every score.
        if centroids is not None:
            skipped = (~read).to(fill.dtype)
            mhat = fill * skipped                              # [B,H_q,n_active]
            out = out.float() + torch.einsum(
                "bhw,bhwd->bhd", mhat, centroids.to(torch.float32))
            den = 1.0 + mhat.sum(-1, keepdim=True)
            out = out / den
            wsum = wsum / den
    return out.to(v_eff.dtype), wsum


# ---------------------------------------------------------------------------
# Triton fused kernel — GQA-grouped, int2-in-register, RoPE-in-register.
# GPU-only, ships UNVALIDATED (validate on a GPU vs two_tier_decode_reference).
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _two_tier_decode_kernel(
        Q, KFP, VFP,
        # Q keys: codes u8 [B,n,H_kv,D,ws//4]; grid = one byte per (head, channel)
        # (u8 scale, i8 zero) over an fp16 scale per GROUP_K of them, [B,n,H_kv,D]
        # and [B,n,H_kv,D//GROUP_K]. Values the same over (head, token) and WS.
        KC, KS, KSS, KZ, KZS,
        VC, VS, VSS, VZ, VZS,
        COS, SIN,                  # RoPE halves [B, n*ws, D//2]
        SEL,                       # gate's pick: int32/int64 [B, H_kv, n_sel]
        LOGM,                      # gate's per-window log-mass: fp32 [B, H_q, n]
        VM, VMS, VANC,             # value centroids: int4-packed u8 [B,n,H_kv,D//2], fp16 [B,n,H_kv], fp32 [B,H_kv,D]
        OUT, WSUM, WMAX,
        PACC, PL, PM,              # split-KV partials: fp32 [B,H_kv,SPLITS,BLOCK_R,D] / [..,BLOCK_R] x2
        scale,
        H_q, H_kv, n_active, n_sel, Sfp, rep, num_sink, n_body_win, W_phys,
        sqb, sqh, sqd,
        kfb, kfh, kfs, kfd,
        vfb, vfh, vfs, vfd,
        selb, selh,
        HEAD_DIM: tl.constexpr, HALF: tl.constexpr, WS: tl.constexpr,
        BLOCK_R: tl.constexpr, BLOCK_NW: tl.constexpr, BLOCK_T: tl.constexpr,
        BLOCK_W: tl.constexpr, PACK_K: tl.constexpr, PACK_V: tl.constexpr,
        GROUP_K: tl.constexpr, GROUP_V: tl.constexpr,
        GATED: tl.constexpr, LOG2E: tl.constexpr,
        SPLITS: tl.constexpr, PHASE: tl.constexpr,
    ):
        """One program == one (batch, KV head); all ``rep`` query heads at once.

        Emits **per-window** softmax mass (DECODE_SPEED_PLAN §5.1) instead of a
        ``[B,H_q,S]`` per-token buffer that then has to be read back twice and
        reduced on the host. Both tiers are tiled in **whole windows** (§5.3), so
        no window is ever split across a tile and the scheme holds for any
        ``window_size`` -- see :func:`window_tiling`.

        The exact algorithm, tile for tile, is mirrored on CPU by
        ``tests/test_window_scores.py``. **Keep the two in step**: that oracle is
        the only thing standing between this kernel and an unverified rewrite,
        because Triton cannot run on the CPU box this repo is developed on.

        ``GATED`` — the read gate
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        When ``GATED``, the Q-tier loop runs over ``n_sel`` **physical slots** and
        dereferences each through ``SEL`` (``widx = tl.load(SEL + …)``) instead of
        walking the tier in order. ``KC``/``KS``/``VC``/``COS``/… are still the
        **whole** gathered tier, indexed by active column — the gate skips *reads*,
        it does not shrink the store — so nothing is re-gathered per step and the
        traffic saved is exactly the windows not named by ``SEL``. This is why the
        selection is an indirection and not a host-side ``index_select``: the
        latter would move the very bytes the gate exists not to move.

        Two things follow, and both are load-bearing:

        * A window's score is stored at ``n_body_win + widx``, its own column —
          not at the slot's position in the tile. Crediting slot ``j``'s mass to
          column ``j`` would put real scores on the wrong windows and still look
          entirely plausible downstream.
        * Skipped columns are never written by the Q loop, so the ``GATED``
          prologue below seeds them (``WSUM = 0``, ``WMAX = -inf``) before it
          runs. Without that, the epilogue would rescale whatever the buffer
          happened to hold — ``wsum``/``wmax`` are ``torch.empty``.

        ``GATED`` is a ``constexpr``, so the ungated kernel is a separate compile
        with the indirection folded away — the gate costs the ungated path
        nothing, not even a predicated load.

        Strides for the Q-tier tensors, ``COS``/``SIN``, ``OUT``, ``WSUM`` and
        ``WMAX`` are DERIVED from shapes rather than passed (§5.2); the dispatcher
        asserts those tensors are contiguous, so it is a checked contract rather
        than an assumption.
        """
        # SPLIT-KV. `SPLITS` and `PHASE` are constexpr, so at the shipped
        # SPLITS=1 / PHASE=0 every branch below folds away and this kernel is
        # byte-for-byte the pre-split one: `sp` is 0, the chunk bounds are the
        # full ranges, no partial is written or read, and the epilogue runs
        # inline exactly as before. That is what makes SPLITS=1 the control arm
        # (§11.11) rather than a nominally-similar second path.
        #
        # PHASE 0 = attend + epilogue in one launch (the SPLITS=1 path)
        #       1 = attend only; write this split's (acc, l, m) partial
        #       2 = merge the partials, then epilogue. Grid is B*H_kv here.
        pid = tl.program_id(0)
        sp = pid % SPLITS
        bh = pid // SPLITS
        b = bh // H_kv
        kv = bh % H_kv

        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < rep
        hq = kv * rep + offs_r                              # [BLOCK_R] query heads
        offs_hl = tl.arange(0, HALF)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_t = tl.arange(0, BLOCK_T)
        t_win = offs_t // WS                                # window index within tile
        t_tok = offs_t % WS                                 # token index within window
        in_tile = offs_t < (BLOCK_NW * WS)                  # masks the pow2 padding

        qg = tl.load(Q + b * sqb + hq[:, None] * sqh + offs_d[None, :] * sqd,
                     mask=r_mask[:, None], other=0.0).to(tl.float32)   # [BLOCK_R, D]
        q_lo = tl.load(Q + b * sqb + hq[:, None] * sqh + offs_hl[None, :] * sqd,
                       mask=r_mask[:, None], other=0.0).to(tl.float32)
        q_hi = tl.load(Q + b * sqb + hq[:, None] * sqh + (offs_hl + HALF)[None, :] * sqd,
                       mask=r_mask[:, None], other=0.0).to(tl.float32)

        m = tl.full([BLOCK_R], -float("inf"), tl.float32)
        l = tl.zeros([BLOCK_R], tl.float32)
        acc = tl.zeros([BLOCK_R, HEAD_DIM], tl.float32)

        # This split's contiguous chunk of the WINDOW axis. The partition is
        # over windows, not keys, so every WSUM/WMAX column is written by
        # exactly one split and needs no cross-split reduction -- see the CPU
        # oracle's `_chunk`, which this mirrors term for term:
        #     lo(i) = i*base + min(i, rem);  hi(i) = lo(i+1)
        # Written with two `minimum`s and no conditional so it is the same
        # expression on both sides. At SPLITS=1 it is (0, n) exactly.
        _bb = n_body_win // SPLITS
        _br = n_body_win % SPLITS
        lo_b = sp * _bb + tl.minimum(sp, _br)
        hi_b = (sp + 1) * _bb + tl.minimum(sp + 1, _br)
        # Keys past this split's last window belong to the next split. Masking
        # by Sfp alone would let the chunk's final tile read into it and double
        # count those keys in the merged softmax.
        body_hi = tl.minimum(Sfp, num_sink + hi_b * WS)

        # Derived strides -- contiguous by contract (see the dispatcher's check).
        wsb = H_q * W_phys
        wsh = W_phys
        lgb = H_q * n_active
        lgh = n_active
        ob = H_q * HEAD_DIM
        kcb = n_active * H_kv * HEAD_DIM * PACK_K
        kcn = H_kv * HEAD_DIM * PACK_K
        kch = HEAD_DIM * PACK_K
        ksb = n_active * H_kv * HEAD_DIM
        ksn = H_kv * HEAD_DIM
        ksh = HEAD_DIM
        GK: tl.constexpr = HEAD_DIM // GROUP_K
        ksgb = n_active * H_kv * GK
        ksgn = H_kv * GK
        ksgh = GK
        vcb = n_active * H_kv * WS * PACK_V
        vcn = H_kv * WS * PACK_V
        vch = WS * PACK_V
        vsb = n_active * H_kv * WS
        vsn = H_kv * WS
        vsh = WS
        GV: tl.constexpr = WS // GROUP_V
        vsgb = n_active * H_kv * GV
        vsgn = H_kv * GV
        vsgh = GV
        cob = n_active * WS * HALF
        # VM is int4, packed two per byte along D, so its row is HALF wide.
        vmb = n_active * H_kv * HALF
        vmn = H_kv * HALF
        vmh = HALF
        vmsb = n_active * H_kv
        vmsn = H_kv
        vab = H_kv * HEAD_DIM
        vah = HEAD_DIM

        # ---- 1. sink prologue: softmax only, emits no window score -----------
        # Sinks are not represented in window scores (the scorer strips them
        # before windowing), so they contribute to `out` and the LSE, nothing else.
        # Split 0 alone takes the sinks. They emit no window score (the scorer
        # strips them before windowing), so they only have to land in exactly
        # one split's softmax; num_sink is 5, so splitting them buys nothing.
        # At SPLITS=1, `sp` is `pid % 1` and this folds to always-true.
        _sink_n = num_sink if sp == 0 else 0
        for s0 in range(0, _sink_n, BLOCK_T):
            offs_n = s0 + offs_t
            nmask = offs_n < _sink_n
            kf = tl.load(KFP + b * kfb + kv * kfh + offs_n[:, None] * kfs
                         + offs_d[None, :] * kfd,
                         mask=nmask[:, None], other=0.0).to(tl.float32)
            logit = tl.dot(qg, tl.trans(kf)) * scale
            logit = tl.where(nmask[None, :], logit, -float("inf"))
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp2(m - m_new)
            p = tl.where(nmask[None, :], tl.exp2(logit - m_new[:, None]), 0.0)
            vf = tl.load(VFP + b * vfb + kv * vfh + offs_n[:, None] * vfs
                         + offs_d[None, :] * vfd,
                         mask=nmask[:, None], other=0.0).to(tl.float32)
            acc = acc * corr[:, None] + tl.dot(p, vf)
            l = l * corr + tl.sum(p, axis=1)
            m = m_new

        # ---- 2. fp body: whole-window tiles, starting at num_sink -------------
        for w0 in range(lo_b, hi_b, BLOCK_NW):
            key0 = num_sink + w0 * WS
            offs_n = key0 + offs_t
            nmask = in_tile & (offs_n < body_hi)
            kf = tl.load(KFP + b * kfb + kv * kfh + offs_n[:, None] * kfs
                         + offs_d[None, :] * kfd,
                         mask=nmask[:, None], other=0.0).to(tl.float32)
            logit = tl.dot(qg, tl.trans(kf)) * scale
            logit = tl.where(nmask[None, :], logit, -float("inf"))
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp2(m - m_new)
            p = tl.where(nmask[None, :], tl.exp2(logit - m_new[:, None]), 0.0)
            vf = tl.load(VFP + b * vfb + kv * vfh + offs_n[:, None] * vfs
                         + offs_d[None, :] * vfd,
                         mask=nmask[:, None], other=0.0).to(tl.float32)
            acc = acc * corr[:, None] + tl.dot(p, vf)
            l = l * corr + tl.sum(p, axis=1)
            m = m_new
            # Per-window sums against THIS tile's running max. One WMAX per
            # window is what lets the epilogue rescale to the global LSE.
            for j in tl.static_range(BLOCK_NW):
                w = w0 + j
                sel = (t_win == j) & nmask
                pj = tl.sum(tl.where(sel[None, :], p, 0.0), axis=1)
                keep = r_mask & (w < n_body_win)
                tl.store(WSUM + b * wsb + hq * wsh + w, pj, mask=keep)
                tl.store(WMAX + b * wsb + hq * wsh + w, m_new, mask=keep)

        # ---- 3. Q tier: int2 dequant + RoPE in registers, whole-window tiles ---
        # Under GATED, seed the Q columns this program owns: the loop below writes
        # only the selected ones and the epilogue reads them all.
        # SPLITS > 1 moves this to the HOST. Every split would otherwise re-seed
        # the Q columns, and nothing orders the splits within one launch, so a
        # later split could zero a column an earlier one had already scored.
        # The dispatcher seeds once before phase 1 instead; at SPLITS=1 there is
        # only one program per (b, kv) and the in-kernel seed is kept exactly as
        # it was.
        if GATED and SPLITS == 1:
            offs_wi = tl.arange(0, BLOCK_W)
            for wi0 in range(n_body_win, W_phys, BLOCK_W):
                icols = wi0 + offs_wi
                imask = r_mask[:, None] & (icols < W_phys)[None, :]
                iptr = b * wsb + hq[:, None] * wsh + icols[None, :]
                tl.store(WSUM + iptr, tl.zeros([BLOCK_R, BLOCK_W], tl.float32),
                         mask=imask)
                tl.store(WMAX + iptr,
                         tl.full([BLOCK_R, BLOCK_W], -float("inf"), tl.float32),
                         mask=imask)

        cbyte = (offs_d // 4)
        cshift = (2 * (offs_d % 4)).to(tl.uint8)
        # int4 lane map for the value centroids (§5): two per byte along D.
        vbyte = (offs_d // 2)
        vshift = (4 * (offs_d % 2)).to(tl.uint8)
        byte_t = (t_tok // 4)
        shift_t = (2 * (t_tok % 4)).to(tl.uint8)
        # A statement, not a ternary: `if` on a constexpr is the form Triton's
        # frontend is guaranteed to fold, and only the taken branch is traced.
        if GATED:
            n_q_iter = n_sel
        else:
            n_q_iter = n_active
        _qb = n_q_iter // SPLITS
        _qr = n_q_iter % SPLITS
        lo_q = sp * _qb + tl.minimum(sp, _qr)
        hi_q = (sp + 1) * _qb + tl.minimum(sp + 1, _qr)
        for w0 in range(lo_q, hi_q, BLOCK_NW):
            if GATED:
                # Slot -> active column. Clamp the dereference rather than mask
                # it: `qmask` already decides what counts, and a clamped widx
                # keeps every offset derived from it (the codes, and `crow` into
                # COS/SIN) inside the tier for the dead lanes too.
                slot = w0 + t_win                            # [BLOCK_T] slots
                qmask = in_tile & (slot < n_sel)
                widx = tl.load(SEL + b * selb + kv * selh
                               + tl.minimum(slot, n_sel - 1)).to(tl.int32)
            else:
                widx = w0 + t_win                            # [BLOCK_T] window ids
                qmask = in_tile & (widx < n_active)
            # The grid is one byte per entry times the fp16 scale its group of
            # GROUP_K channels shares, in that order -- the same two ops the
            # store's dequant does, so the kernel reads the exact grid the codes
            # were fit to. The group scales are few (D // GROUP_K per window and
            # head) and every lane of a group hits the same address, so the
            # repeated load is an L1 hit, not traffic.
            kg_lo = (offs_hl // GROUP_K)[:, None]
            kg_hi = ((offs_hl + HALF) // GROUP_K)[:, None]
            gptr = KSS + b * ksgb + widx[None, :] * ksgn + kv * ksgh
            zptr = KZS + b * ksgb + widx[None, :] * ksgn + kv * ksgh
            kptr = KS + b * ksb + widx[None, :] * ksn + kv * ksh
            kzptr = KZ + b * ksb + widx[None, :] * ksn + kv * ksh
            ks_lo = (tl.load(kptr + offs_hl[:, None],
                             mask=qmask[None, :], other=0).to(tl.float32)
                     * tl.load(gptr + kg_lo,
                               mask=qmask[None, :], other=0.0).to(tl.float32))
            kz_lo = (tl.load(kzptr + offs_hl[:, None],
                             mask=qmask[None, :], other=0).to(tl.float32)
                     * tl.load(zptr + kg_lo,
                               mask=qmask[None, :], other=0.0).to(tl.float32))
            ks_hi = (tl.load(kptr + (offs_hl + HALF)[:, None],
                             mask=qmask[None, :], other=0).to(tl.float32)
                     * tl.load(gptr + kg_hi,
                               mask=qmask[None, :], other=0.0).to(tl.float32))
            kz_hi = (tl.load(kzptr + (offs_hl + HALF)[:, None],
                             mask=qmask[None, :], other=0).to(tl.float32)
                     * tl.load(zptr + kg_hi,
                               mask=qmask[None, :], other=0.0).to(tl.float32))
            kb_lo = tl.load(KC + b * kcb + widx[None, :] * kcn + kv * kch
                            + offs_hl[:, None] * PACK_K + byte_t[None, :],
                            mask=qmask[None, :], other=0)
            kb_hi = tl.load(KC + b * kcb + widx[None, :] * kcn + kv * kch
                            + (offs_hl + HALF)[:, None] * PACK_K + byte_t[None, :],
                            mask=qmask[None, :], other=0)
            k_lo = ((kb_lo >> shift_t[None, :]) & 3).to(tl.float32) * ks_lo + kz_lo
            k_hi = ((kb_hi >> shift_t[None, :]) & 3).to(tl.float32) * ks_hi + kz_hi
            crow = widx * WS + t_tok
            c = tl.load(COS + b * cob + crow[None, :] * HALF + offs_hl[:, None],
                        mask=qmask[None, :], other=0.0).to(tl.float32)
            s = tl.load(SIN + b * cob + crow[None, :] * HALF + offs_hl[:, None],
                        mask=qmask[None, :], other=0.0).to(tl.float32)
            k_rlo = k_lo * c - k_hi * s
            k_rhi = k_hi * c + k_lo * s
            logit = (tl.dot(q_lo, k_rlo) + tl.dot(q_hi, k_rhi)) * scale
            logit = tl.where(qmask[None, :], logit, -float("inf"))
            m_new = tl.maximum(m, tl.max(logit, axis=1))
            corr = tl.exp2(m - m_new)
            p = tl.where(qmask[None, :], tl.exp2(logit - m_new[:, None]), 0.0)
            vg = t_tok // GROUP_V
            vs = (tl.load(VS + b * vsb + widx * vsn + kv * vsh + t_tok,
                          mask=qmask, other=0).to(tl.float32)
                  * tl.load(VSS + b * vsgb + widx * vsgn + kv * vsgh + vg,
                            mask=qmask, other=0.0).to(tl.float32))
            vz = (tl.load(VZ + b * vsb + widx * vsn + kv * vsh + t_tok,
                          mask=qmask, other=0).to(tl.float32)
                  * tl.load(VZS + b * vsgb + widx * vsgn + kv * vsgh + vg,
                            mask=qmask, other=0.0).to(tl.float32))
            vb = tl.load(VC + b * vcb + widx[:, None] * vcn + kv * vch
                         + t_tok[:, None] * PACK_V + cbyte[None, :],
                         mask=qmask[:, None], other=0)
            vv = ((vb >> cshift[None, :]) & 3).to(tl.float32) * vs[:, None] + vz[:, None]
            acc = acc * corr[:, None] + tl.dot(p, vv)
            l = l * corr + tl.sum(p, axis=1)
            m = m_new
            for j in tl.static_range(BLOCK_NW):
                lane = (t_win == j) & qmask
                pj = tl.sum(tl.where(lane[None, :], p, 0.0), axis=1)
                if GATED:
                    # Slot w0+j scores onto ITS OWN column, not onto column j.
                    sj = w0 + j
                    col = n_body_win + tl.load(
                        SEL + b * selb + kv * selh
                        + tl.minimum(sj, n_sel - 1)).to(tl.int32)
                    keep = r_mask & (sj < n_sel)
                else:
                    w = w0 + j
                    col = n_body_win + w
                    keep = r_mask & (w < n_active)
                tl.store(WSUM + b * wsb + hq * wsh + col, pj, mask=keep)
                tl.store(WMAX + b * wsb + hq * wsh + col, m_new, mask=keep)

        # ---- phase boundary --------------------------------------------------
        if PHASE == 1:
            # Attend-only: hand this split's running softmax to phase 2 and
            # stop. The epilogue needs the GLOBAL lse and a full pass over
            # W_phys, so it cannot run per split.
            _pb = H_kv * SPLITS * BLOCK_R
            _ph = SPLITS * BLOCK_R
            _pr = b * _pb + kv * _ph + sp * BLOCK_R + offs_r
            tl.store(PL + _pr, l, mask=r_mask)
            tl.store(PM + _pr, m, mask=r_mask)
            tl.store(PACC + _pr[:, None] * HEAD_DIM + offs_d[None, :],
                     acc, mask=r_mask[:, None])
            return

        if PHASE == 2:
            # Merge: the standard log-sum-exp combine, mirroring the oracle.
            # A split that saw no key has m == -inf and l == 0; exp2(-inf - M)
            # is 0 for finite M but -inf - -inf is NaN, so an all-empty row is
            # selected away rather than computed.
            _pb = H_kv * SPLITS * BLOCK_R
            _ph = SPLITS * BLOCK_R
            m = tl.full([BLOCK_R], -float("inf"), tl.float32)
            for _i in range(0, SPLITS):
                _r = b * _pb + kv * _ph + _i * BLOCK_R + offs_r
                m = tl.maximum(m, tl.load(PM + _r, mask=r_mask,
                                          other=-float("inf")))
            l = tl.zeros([BLOCK_R], tl.float32)
            acc = tl.zeros([BLOCK_R, HEAD_DIM], tl.float32)
            for _i in range(0, SPLITS):
                _r = b * _pb + kv * _ph + _i * BLOCK_R + offs_r
                _m = tl.load(PM + _r, mask=r_mask, other=-float("inf"))
                _l = tl.load(PL + _r, mask=r_mask, other=0.0)
                _a = tl.load(PACC + _r[:, None] * HEAD_DIM + offs_d[None, :],
                             mask=r_mask[:, None], other=0.0)
                _w = tl.where(_m > -float("inf"), tl.exp2(_m - m), 0.0)
                l = l + _l * _w
                acc = acc + _a * _w[:, None]

        out = acc / l[:, None]
        lse = m + tl.log2(l)
        # OUT is NOT stored here. Under GATED the windows this step declined to
        # read still contribute to the attention output, through their value
        # centroids at the weight the epilogue is about to measure -- so the
        # output is not final until §5 has run. The ungated kernel stores at the
        # bottom unchanged; `GATED` is a constexpr, so it costs it nothing.

        # ---- 4. epilogue: rescale each window from its tile max to the LSE ----
        # Runs over W_phys values, not S. That is the whole of §5.1's traffic cut.
        #
        # Under GATED this pass also accumulates what the skipped windows need,
        # so their scores never leave the kernel. Doing it on the host cost 22
        # torch ops per layer per step -- 704 launches per token at L=32 -- to
        # shuffle a [B, H_q, W] tensor, on a decode path that is bound by launch
        # count. The two reductions ride along in a pass that already runs.
        #
        # `smax > -inf` IS the selected set: the GATED prologue seeded every Q
        # column to -inf and only the visited ones were written, so no SEL lookup
        # is needed here.
        offs_w = tl.arange(0, BLOCK_W)
        num = tl.zeros([BLOCK_R], tl.float32)          # mass the read windows hold
        gmx = tl.full([BLOCK_R], -float("inf"), tl.float32)   # running max of logmass
        gsm = tl.zeros([BLOCK_R], tl.float32)          # sum exp(logmass - gmx)
        for w0 in range(0, W_phys, BLOCK_W):
            cols = w0 + offs_w
            cmask = cols < W_phys
            ptr = WSUM + b * wsb + hq[:, None] * wsh + cols[None, :]
            mptr = WMAX + b * wsb + hq[:, None] * wsh + cols[None, :]
            sm = r_mask[:, None] & cmask[None, :]
            ssum = tl.load(ptr, mask=sm, other=0.0)
            smax = tl.load(mptr, mask=sm, other=-float("inf"))
            scaled = tl.where(smax > -float("inf"),
                              ssum * tl.exp2(smax - lse[:, None]), 0.0)
            tl.store(ptr, scaled, mask=sm)
            if GATED:
                live = sm & (smax > -float("inf")) & (cols[None, :] >= n_body_win)
                num += tl.sum(tl.where(live, scaled, 0.0), axis=1)
                qc = tl.maximum(cols - n_body_win, 0)
                lg = tl.load(LOGM + b * lgb + hq[:, None] * lgh + qc[None, :],
                             mask=live, other=-float("inf"))
                gmx_new = tl.maximum(gmx, tl.max(tl.where(live, lg, -float("inf")),
                                                 axis=1))
                # `gmx` is -inf until the first selected column is seen, and a
                # tile of body columns alone sees none; exp(-inf - -inf) is NaN,
                # so the first-tile case is selected away rather than computed.
                corr = tl.where(gmx == -float("inf"), 0.0, tl.exp2(
                    (gmx - gmx_new) * LOG2E))
                gsm = gsm * corr + tl.sum(
                    tl.where(live, tl.exp2((lg - gmx_new[:, None]) * LOG2E), 0.0),
                    axis=1)
                gmx = gmx_new

        # ---- 5. GATED only: score the windows this step did not read ----------
        # A skipped window credited with zero would rank last and be evicted, so
        # the gate would destroy the tier it exists to read less often. It gets
        # its card's estimate instead, put on the same footing as the real scores
        # by the ratio the read windows give for free:
        #
        #     score_i = (mass the read windows hold) * softmax(logmass)_i
        #
        # over the read set. The arbitrary offset in `fill_skipped_window_scores`
        # cancels in that ratio, which is why only a max and a sum are needed and
        # no logarithm is taken. `LOG2E` keeps the base-e logmass exact while
        # every exponential in this kernel stays base 2.
        if GATED:
            gsm = tl.maximum(gsm, 1e-30)
            # The skipped windows also ATTEND, through their value centroids.
            #
            # Reading 25% of the tier and returning that softmax unchanged is not
            # an approximation of attention over the cache -- it is exact
            # attention over a DIFFERENT cache, one where the other 75% does not
            # exist and the surviving weights have been inflated to cover for it.
            # `fill` is already the calibrated mass those windows carry, so the
            # honest output adds each one's representative at that weight and
            # renormalizes.
            #
            # This rides in the pass that was already running. The only new
            # traffic is the centroid tile (D int8 + one fp16 scale per skipped
            # window, 130 B/head against the 6432 B/window not read), and the
            # only new arithmetic is one `tl.dot` of a tile the loop already
            # holds -- the same shape as the Q loop's `tl.dot(p, vv)`.
            vacc = tl.zeros([BLOCK_R, HEAD_DIM], tl.float32)
            msum = tl.zeros([BLOCK_R], tl.float32)
            vanc = tl.load(VANC + b * vab + kv * vah + offs_d).to(tl.float32)
            for w0 in range(n_body_win, W_phys, BLOCK_W):
                cols = w0 + offs_w
                cmask = cols < W_phys
                ptr = WSUM + b * wsb + hq[:, None] * wsh + cols[None, :]
                mptr = WMAX + b * wsb + hq[:, None] * wsh + cols[None, :]
                sm = r_mask[:, None] & cmask[None, :]
                smax = tl.load(mptr, mask=sm, other=0.0)
                skipped = sm & (smax == -float("inf"))
                qc = tl.maximum(cols - n_body_win, 0)
                lg = tl.load(LOGM + b * lgb + hq[:, None] * lgh + qc[None, :],
                             mask=skipped, other=-float("inf"))
                fill = (num[:, None]
                        * tl.exp2((lg - gmx[:, None]) * LOG2E) / gsm[:, None])
                tl.store(ptr, fill, mask=skipped)

                f = tl.where(skipped, fill, 0.0)           # [BLOCK_R, BLOCK_W]
                msum += tl.sum(f, axis=1)
                # Centroid tile: [BLOCK_W, HEAD_DIM]. Dequantized in registers,
                # exactly as the Q loop dequantizes codes -- no fp16 centroid
                # tensor is ever built.
                qmask = cmask & (cols >= n_body_win)
                # int4 nibbles, +8 biased: element d is byte d//2, shift 4*(d%2)
                # -- the same unpack as the Q loop's int2 crumbs, one width up.
                vb_ = tl.load(VM + b * vmb + qc[:, None] * vmn + kv * vmh
                              + vbyte[None, :],
                              mask=qmask[:, None], other=0)
                vq = (((vb_ >> vshift[None, :]) & 0xF).to(tl.float32) - 8.0)
                vsc = tl.load(VMS + b * vmsb + qc * vmsn + kv,
                              mask=qmask, other=0.0).to(tl.float32)
                vbar = vq * vsc[:, None] + vanc[None, :]
                vacc = vacc + tl.dot(f, vbar)

            # Renormalize: `out` and every stored window score are on a softmax
            # that summed to 1 over the read set, and `msum` of mass has just
            # been added to it.
            den = 1.0 + msum
            out = (out + vacc) / den[:, None]
            for w0 in range(0, W_phys, BLOCK_W):
                cols = w0 + offs_w
                sm = r_mask[:, None] & (cols < W_phys)[None, :]
                ptr = WSUM + b * wsb + hq[:, None] * wsh + cols[None, :]
                tl.store(ptr, tl.load(ptr, mask=sm, other=0.0) / den[:, None],
                         mask=sm)

        tl.store(OUT + b * ob + hq[:, None] * HEAD_DIM + offs_d[None, :],
                 out.to(OUT.dtype.element_ty), mask=r_mask[:, None])


#: Per-(shape, dtype, device) scratch, reused across layers within a step.
#:
#: the design notesD5 sized this against three allocations per layer per step
#: and rated it the smallest item on its list. It counted the decode side only;
#: the read gate brought two more (`est`, `logm`), and `topk`/`sort` two outputs
#: each, so the fused path allocates ~11 per layer per step -- **352 per step at
#: L=32**, on a path where the host gap is 37% of the step.
#:
#: Only tensors that are DEAD by the end of `_run_fused` live here. `out` does
#: not: it is returned to the model as the attention output, so reusing it would
#: let layer i+1 overwrite a tensor layer i's `o_proj` may still be reading. That
#: is precisely the class of silent aliasing bug `CacheState.replace` guards
#: against, and it is why this cache is opt-in per buffer rather than applied to
#: everything allocated here.
#:
#: `bfbdb1f` -- *"revert the reusable ctx dict -- it caused the narrativeqa OOM"*
#: -- is the standing precedent. A reused buffer holds memory the caching
#: allocator would otherwise recycle, and the headline cell already peaks near
#: 48 GB. Keying on the exact shape means a changed geometry allocates a new
#: turns the whole thing off if `peak_GB` moves in the perf table.
_SCRATCH: dict = {}

_SCRATCH_ON = True


def _scratch(name: str, shape, dtype, device):
    """A reusable buffer for ``name`` at this exact geometry.

    Contents are undefined on entry, exactly as ``torch.empty`` leaves them, so
    every caller must fully write what it later reads. The two-tier kernel does:
    the body loop writes the fp columns, the ``GATED`` prologue seeds every Q
    column, and the epilogue reads only what those two wrote.
    """
    if not _SCRATCH_ON:
        return torch.empty(shape, dtype=dtype, device=device)
    key = (tuple(shape), dtype, str(device))
    held = _SCRATCH.get(name)
    if held is not None and held[0] == key:
        return held[1]
    # ONE buffer per name, replaced when the geometry moves -- never accumulated.
    # `W_phys` tracks `n_active`, which moves at every eviction, so a cache keyed
    # on shape would keep a buffer per distinct window count for the whole
    # generation. That is the `bfbdb1f` narrativeqa OOM with extra steps. Holding
    # one means a changed shape costs exactly the allocation it costs today, and
    # the steady state -- where every step has the same geometry -- still reuses.
    buf = torch.empty(shape, dtype=dtype, device=device)
    _SCRATCH[name] = (key, buf)
    return buf


def release_decode_scratch() -> None:
    """Drop every reused buffer. For tests and for a caller that has finished a
    generation and wants the memory back before the next shape arrives."""
    _SCRATCH.clear()


def _pow2_at_least(x: int, floor: int = 16) -> int:
    v = floor
    while v < x:
        v *= 2
    return v


# ---------------------------------------------------------------------------
# Tile search -- shared-memory fit AND measured time
# ---------------------------------------------------------------------------

#: ``(target_keys, num_stages, num_warps)`` rungs. Bigger tiles mean fewer
#: serial iterations (§5.3's whole point) but more ``tl.dot`` operand staging in
#: shared memory. An A100 has 163 KB/SM and BLOCK_T=64 needs ~128 KB of operands
#: before staging, so the top rung does not fit everywhere -- hence a ladder
#: rather than a constant.
#:
#: ``num_stages`` is not free: Triton's software pipeliner allocates that many
#: copies of the loop's shared-memory operands, so dropping 2 -> 1 roughly halves
#: the staging at the SAME tile. Without ``(64, 1)`` a kernel that just missed
#: ``(64, 2)`` fell straight to ``target_keys=32`` and **doubled its serial
#: Q-tier iterations** -- 23 -> 45 at ``ws=8`` -- to buy shared memory one fewer
#: stage would also have bought.
#:
#: ``num_warps`` was never passed at all, so every launch ran at Triton's
#: default of 4 regardless of tile. At ``BLOCK_T=64`` and ``HEAD_DIM=128`` that
#: is a guess, not a choice, and it is the one launch parameter with no cost
#: model here at all -- so it is searched rather than asserted.
#:
#: **The ladder is now TIMED, not first-fit.** It used to stop at the first rung
#: that merely *launched*, which asserts that a full tile at one stage beats a
#: half tile at two -- i.e. that serial iteration count dominates latency hiding.
#: The 2026-09-19 profile says that assertion was wrong in the direction that
#: matters: at ``target_keys=64`` the ~128 KB of staged operands leave room for
#: **one block per SM**, so the Q-tier loop's ~14 dependent tiles run with
#: nothing co-resident to cover their latency, and the kernel lands at ~190 GB/s
#: on a 1555 GB/s part (97.2% GPU busy, so this is the kernel's own time, not a
#: host gap). A smaller tile trades iteration count for occupancy, and which way
#: that trade falls is a measurement, not a derivation.
#:
#: **Ordering still carries meaning, and the claim that it did not was wrong.**
#: It is irrelevant to a TUNED geometry -- every rung is timed and the fastest
#: wins, whatever order they were tried in. But a geometry that has not yet been
#: seen :data:`_TUNE_AFTER` times runs :func:`_first_fit`, which returns the
#: first rung that *fits*; a fit is shape-independent, so that fallback is a
#: CONSTANT across every untuned geometry in the run. Under the tile-major order
#: that constant was ``(64, 2, 4)`` -- measured 5th-to-9th of 11 on three
#: separate profiles and 1.20-1.45x off the winner. §10.14 measured 6.1% of tile
#: win available at 2048/B=32 and only 1.4% observed; untuned geometries taking
#: a known-mediocre rung is the leak.
#:
#: So the ladder is now ordered by MEASURED time, best first, from the
#: 2026-09-20 control-arm profile (the first fully valid one -- positive
#: profiler overhead, no compilation in either window). The three profiles taken
#: at ``ws=8, D=128, B=32, n_sel~2^6`` rank the rungs identically:
#:
#:     (32,2,4) 0.486  (32,1,4) 0.538  (64,2,8) 0.561  (64,1,8) 0.575
#:     (64,2,4) 0.582  (64,1,4) 0.583  (16,2,4) 0.614  (16,1,4) 0.629
#:     (32,1,8) 0.647  (32,2,8) 0.654  (16,1,8) 0.807  (16,2,8) 0.823
#:
#: **What this is NOT.** `_LAST_WINNER` (§10.15) exported one geometry's tuned
#: winner across all geometries at runtime and cost 12%, because the trade a
#: rung selects does move with the geometry. This is a different thing: a static,
#: deterministic reordering of the fallback walk, which still walks, and which
#: replaces an ordering that was *asserted* with one that was *measured*. Both
#: are guesses at an untuned geometry. Only one of them is informed.
#:
#: **It is a guess, so it has a control arm.** `STICKYKV_FIT_LADDER=legacy`
#: restores the tile-major order exactly, which makes the A/B one environment
#: variable and the claim statable against it -- the thing `_LAST_WINNER` shipped
#: without. The risk this prices: at a fill-phase geometry (`n_sel` small) the
#: Q-tier loop is short, the fp body dominates, and a larger tile may win there
#: even though it loses at the shipped shape. Nobody has measured that, and this
#: comment is not a substitute for doing so.
_FIT_LADDER_MEASURED = [
    (32, 2, 4), (32, 1, 4), (64, 2, 8), (64, 1, 8),
    (64, 2, 4), (64, 1, 4), (16, 2, 4), (16, 1, 4),
    (32, 1, 8), (32, 2, 8), (16, 1, 8), (16, 2, 8),
]

#: The pre-2026-09-20 tile-major order. The control arm, and nothing else.
_FIT_LADDER_LEGACY = [
    (64, 2, 4), (64, 2, 8),
    (64, 1, 4), (64, 1, 8),
    (32, 2, 4), (32, 2, 8),
    (32, 1, 4), (32, 1, 8),
    (16, 2, 4), (16, 2, 8),
    (16, 1, 4), (16, 1, 8),
]


#: How many programs share one (batch, KV head)'s window axis. **1 = off.**
#:
#: `STICKYKV_DECODE_SPLITS`: an integer, or `auto`. Default 1.
#:
#: Off by default on purpose. §11.11 measured the case FOR splitting -- the
#: grid is 8 blocks at B=1 on a 108-SM A100, the step is 93% batch-invariant,
#: ~11 ms of a 14.42 ms kernel is paid regardless of batch -- but the kernel
#: that implements it cannot be run on the box it was written on. An
#: unvalidatable change does not become the shipped path on the strength of an
#: argument, however good the argument. `splits=1` is a provable no-op (every
#: branch is constexpr-folded, no buffer is allocated, one launch on the old
#: grid), so it is both the default and the control arm.
#:
#: `auto` maximises the fraction of the machine the grid actually fills, and
#: never splits finer than the work allows -- a split with no window in it
#: costs a launch and contributes an empty partial.
#:
#: **It is wave quantisation that matters, not raw block count**, and the first
#: version of this got that wrong: it targeted a flat 216 blocks and so
#: returned S=1 at B=32, where 256 blocks over 108 SMs is 2.37 waves and the
#: tail wave fills 40 of 108 -- 79% of the machine, which the flat target
#: called finished. It also picked S=16 at B=1 (128 blocks, 2 waves, 59%) when
#: S=13 (104 blocks, one wave) fills 96%. Both were worse than available.
#:
#: Utilisation is `blocks / (ceil(blocks / SMs) * SMs)`. Among the split counts
#: within `_SPLITS_TIE` points of the best, the SMALLEST wins: each extra split
#: adds a partial to store, a partial to reload and a merge iteration, and
#: that cost is **unmeasured**. Which is also why `auto` is a starting point
#: and not an answer -- NEXT_RUNS.md sweeps S explicitly at the headline cell,
#: because where the real optimum sits depends on exactly that overhead.
_DECODE_SPLITS_ENV = "STICKYKV_DECODE_SPLITS"
_SPLITS_MAX = 16
_SPLITS_TIE = 5.0
_SPLITS_FALLBACK_SMS = 108          # A100; only used if the device cannot say


def _device_sms(default: int = _SPLITS_FALLBACK_SMS) -> int:
    try:
        import torch as _t
        if _t.cuda.is_available():
            return int(_t.cuda.get_device_properties(
                _t.cuda.current_device()).multi_processor_count)
    except Exception:  # pragma: no cover - device dependent
        pass
    return default


def _resolve_splits(B: int, H_kv: int, n_q_iter: int, n_body_win: int) -> int:
    """Splits for this geometry. 1 unless asked for."""
    raw = os.environ.get(_DECODE_SPLITS_ENV, "").strip().lower()
    if raw in ("", "1", "off", "false", "no"):
        return 1
    # Never split finer than there is work: the window axis is what is being
    # partitioned, so `splits > windows` just makes empty programs.
    cap = max(1, min(_SPLITS_MAX, max(n_q_iter, n_body_win, 1)))
    if raw == "auto":
        sms = _device_sms()
        base = max(B * H_kv, 1)

        def _util(n: int) -> float:
            blocks = base * n
            waves = -(-blocks // sms)          # ceil
            return blocks / (waves * sms) * 100.0

        scored = [(n, _util(n)) for n in range(1, cap + 1)]
        best = max(u for _, u in scored)
        # Smallest split count that gets within _SPLITS_TIE points of the best:
        # extra splits buy occupancy and cost merge work, and only the first
        # half of that trade is known.
        return min(n for n, u in scored if u >= best - _SPLITS_TIE)
    try:
        want = int(raw)
    except ValueError:
        raise RuntimeError(
            f"{_DECODE_SPLITS_ENV}={raw!r} is not an integer or 'auto'. A "
            "misspelled knob that silently falls back is how two arms come to "
            "measure the same thing.") from None
    if want < 1:
        raise RuntimeError(
            f"{_DECODE_SPLITS_ENV}={want} is not a split count; it must be >= 1 "
            "(1 = off, the control arm).")
    return min(want, cap)


def decode_splits_setting() -> str:
    """What this process was ASKED for, for a run's provenance."""
    return os.environ.get(_DECODE_SPLITS_ENV, "").strip().lower() or "1"


def _resolve_fit_ladder():
    """Latched at import: this must not be read on the hot path."""
    import os
    want = os.environ.get("STICKYKV_FIT_LADDER", "").strip().lower()
    if want in ("legacy", "tile-major", "tile_major"):
        print("[StickyKV] decode tile ladder: LEGACY tile-major order "
              "(STICKYKV_FIT_LADDER=legacy) -- this is the CONTROL ARM for the "
              "measured ordering, not the shipped default.", flush=True)
        return list(_FIT_LADDER_LEGACY)
    if want not in ("", "measured"):
        raise RuntimeError(
            f"STICKYKV_FIT_LADDER={want!r} is not a ladder. Use 'measured' "
            "(default) or 'legacy'. A misspelled knob that silently falls back "
            "is how two arms come to measure the same thing.")
    return list(_FIT_LADDER_MEASURED)


_FIT_LADDER = _resolve_fit_ladder()

#: Which ordering this process is using, for a run's provenance.
def fit_ladder_order() -> str:
    return "legacy" if _FIT_LADDER == list(_FIT_LADDER_LEGACY) else "measured"


#: Winning rung per geometry signature, so the search runs once per process.
_FIT_CHOICE: dict = {}

#: Every rung's measured milliseconds, per signature, kept so a profile can print
#: the search rather than just its verdict. A rung that did not fit is absent --
#: that absence IS the shared-memory result.
_FIT_TIMINGS: dict = {}

_FIT_ANNOUNCED: set = set()

#: How many times each signature has been seen, so a geometry is only tuned once
#: it has proved it will recur (:data:`_TUNE_AFTER`).
_FIT_SEEN: dict = {}


def fit_choice() -> dict:
    """``{sig: rung}`` -- the tile rung chosen per geometry. A copy.

    ``scripts/profile_decode.py`` prints this: the kernel announces its choice
    once per geometry, in warmup, which scrolls away above whatever a profile is
    being read for. It asked for this accessor before it existed, inside a bare
    ``except Exception: pass``, so the one line the tuning step needs was the one
    line a profile did not carry.
    """
    return dict(_FIT_CHOICE)


def fit_timings() -> dict:
    """``{sig: [(rung, ms), ...]}`` -- what the search measured. A copy.

    The verdict without the margin is not reviewable: a rung that won by 30% and
    a rung that won by 0.5% call for different next moves, and only this says
    which happened.
    """
    return {k: list(v) for k, v in _FIT_TIMINGS.items()}


#: Launches per timed sample, and samples per rung. Small on purpose: the whole
#: search is ``len(_FIT_LADDER) * (_TUNE_WARMUP + _TUNE_ITERS)`` launches of a
#: ~0.5 ms kernel, i.e. tens of milliseconds, and it runs ONCE per geometry.
#:
#: **The cost that is not small is Triton's JIT: one compile per rung, and at the
#: shipped geometry that is all 12.** The rung dedup does not reduce it there
#: (see :func:`_search_rungs`). This is why the harness's warmup exists --
#: ``perf_shapes.py`` runs enough decode steps to cross the first eviction
#: precisely so JIT and autotune land in warmup rather than in measurement run 0
#: -- and why Triton's on-disk cache makes it a once-per-machine cost rather than
#: a once-per-run one. If a cold-cache warmup of a few minutes is not acceptable
#: somewhere, ``_FIT_LADDER`` is one list literal and trimming it is the knob;
#: trim it by *measurement*, not by guessing which rungs matter.
_TUNE_WARMUP = 1
_TUNE_ITERS = 3

#: How many times a geometry must RECUR before it is worth timing.
#:
#: The first version of this searched on a signature's first sight, and that was
#: measured wrong on 2026-09-19: at 4096/B=1 it took TPOT to 0.1917. The cause is
#: that a decode run does not hold one geometry. The Q tier fills at one window
#: per ``ws`` steps, so ``n_sel`` climbs across the whole run, and the perf
#: runner itself warns when a cell is shorter than the fill ("needs ~744 steps
#: ... and this cell runs 512"). Every signature change was a fresh 12-rung
#: search, i.e. up to 12 Triton JIT compiles, landing INSIDE the measured window.
#: Before the search existed only one rung was ever compiled per geometry, so
#: this was a pure regression of my own making.
#:
#: Requiring recurrence fixes both halves. A fill-phase geometry is seen once and
#: never tuned -- it takes the first rung that fits, exactly the old cost. A
#: steady-state geometry recurs every step, so it is tuned once and the winner
#: serves the rest of the run. It also tunes at a REPRESENTATIVE point: tuning on
#: first sight measured the rungs at ``n_sel = 1``, where every rung runs one
#: Q-tier iteration and the comparison is dominated by the fp body -- a tuning
#: point that says nothing about the loop being tuned.
_TUNE_AFTER = 3


def _time_launch(launch, warmup: int = _TUNE_WARMUP,
                 iters: int = _TUNE_ITERS) -> float:
    """Mean device milliseconds for ``launch()``, or raise what it raised.

    One event pair around ``iters`` launches, not ``iters`` pairs: the kernel
    under test is sub-millisecond and per-launch event overhead would be a
    visible share of what is being compared. The warmup launch absorbs Triton's
    JIT for a rung this process has not compiled yet, so the compile never lands
    inside the timed region.

    **The launches are performed on the caller's real tensors.** That is safe
    because the kernel only ever ``tl.store``s to ``OUT`` / ``WSUM`` / ``WMAX``
    and never loads them -- it accumulates in registers -- so running it n times
    leaves exactly what running it once leaves. Timing against scratch copies
    would need a full duplicate of the fp tier, which is the largest thing on the
    step, to measure a kernel whose whole problem is memory.
    """
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / max(iters, 1)


def _first_fit(ladder, launch, label: str, sig, key=None):
    """Launch the first rung that fits, compiling nothing else. The old policy.

    This is what a geometry gets until it has proved it will recur
    (:data:`_TUNE_AFTER`). It is not a fallback -- it is the cheap arm of the
    search, and it exists because compiling twelve rungs for a shape that is
    about to change is strictly worse than compiling one.
    """
    seen_keys = set()
    last: Optional[BaseException] = None
    for rung in ladder:
        k = rung if key is None else key(rung)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        try:
            launch(rung)
        except BaseException as exc:                     # noqa: BLE001
            if not _is_out_of_resources(exc):
                raise
            last = exc
            continue
        return rung
    raise RuntimeError(
        f"{label}: no tile fits in shared memory at any rung. Tried "
        f"{ladder} for sig={sig}. Last error: {last}"
    )


def _search_rungs(choice: dict, timed: dict, announced: set, seen: dict, sig,
                  ladder, launch, label: str, key=None):
    """Time every rung that fits; cache and return the fastest. Once per ``sig``.

    ``launch(rung)`` must run the kernel and return ``None``; it is called
    repeatedly, so it must be idempotent (see :func:`_time_launch`).

    ``key(rung)`` maps a rung to the launch it actually produces, and rungs that
    collapse onto one are timed once. Distinct rungs are not always distinct
    kernels: ``window_tiling`` floors ``target_keys // ws``, so at a large ``ws``
    several ``target_keys`` yield the same ``(BLOCK_NW, BLOCK_T)``.

    **At the shipped geometry this fires for nothing, and that was measured, not
    assumed.** ``BLOCK_W`` is derived from ``target_keys`` too
    (``_pow2_at_least(min(W_phys, target_keys), 16)``), so those rungs still
    differ in a constexpr and are still separate compiles. The dedup only bites
    where the tiling AND ``BLOCK_W`` both collapse -- a large ``ws`` with a short
    window axis (``W_phys <= 16``). It is kept because that regime is real and
    the check is free, NOT because it makes the common case cheaper: at
    ``ws=8, W_phys=273`` the ladder is 12 rungs and 12 compiles.

    A rung that raises ``OutOfResources`` is skipped -- that is the shared-memory
    half of the search and it is unchanged. Any other exception propagates: a
    tuner that swallowed a real error would silently tune around a bug.

    **The winner is re-launched before returning**, so the values the caller goes
    on to use come from the rung every subsequent step will use. Rungs differ in
    tile order, which reassociates the online-softmax accumulation and moves the
    output in its last bits; returning a losing rung's numbers would make the
    step that happened to run the search numerically different from its
    neighbours for no reason.

    Kernel-or-error: if no rung fits, this raises with the whole ladder, because
    a decode kernel that cannot launch must fail loudly (invariant 3).
    """
    known = choice.get(sig)
    if known is not None:
        launch(known)
        return known

    # Only time a geometry that has proved it will recur -- see _TUNE_AFTER.
    n = seen.get(sig, 0) + 1
    seen[sig] = n
    if n < _TUNE_AFTER:
        return _first_fit(ladder, launch, label, sig, key)

    timings = []
    # NOT `seen` -- that is the sighting-count dict this function was handed,
    # and rebinding it here shadowed the parameter for the rest of the body.
    # Harmless today only because the one write to the dict happens above this
    # line; any future read of `seen` below it would have silently queried a
    # set of rungs instead of the recurrence counter.
    deduped = set()
    last: Optional[BaseException] = None
    for rung in ladder:
        k = rung if key is None else key(rung)
        if k in deduped:
            continue
        deduped.add(k)
        try:
            ms = _time_launch(lambda r=rung: launch(r))
        except BaseException as exc:                     # noqa: BLE001
            if not _is_out_of_resources(exc):
                raise
            last = exc
            continue
        timings.append((rung, ms))

    if not timings:
        raise RuntimeError(
            f"{label}: no tile fits in shared memory at any rung. Tried "
            f"{ladder} for sig={sig}. Last error: {last}"
        )

    best = min(timings, key=lambda t: t[1])[0]
    choice[sig] = best
    timed[sig] = sorted(timings, key=lambda t: t[1])
    launch(best)
    if sig not in announced:
        announced.add(sig)
        ranked = " ".join(
            f"{r}={ms:.3f}ms" for r, ms in sorted(timings, key=lambda t: t[1])
        )
        print(f"[StickyKV] {label} tuned sig={sig} -> {best} "
              f"(after {n} sightings; {len(timings)}/{len(ladder)} rungs timed, "
              f"the rest did not fit or duplicate one that did) | {ranked}")
    return best


def _is_out_of_resources(exc: BaseException) -> bool:
    """True for Triton's shared-memory/registers exhaustion, across versions.

    Triton has moved this exception between modules (``triton.runtime.autotuner``
    -> ``triton.runtime.errors``) and wraps it differently by version, so match on
    the type NAME and the message rather than importing a moving target. A
    mis-detection here would either mask a real bug (if too broad) or abort a
    launch that a smaller tile would have served (if too narrow), so it checks
    both.
    """
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ == "OutOfResources":
            return True
        msg = str(cur).lower()
        if "out of resource" in msg and ("shared memory" in msg or "register" in msg):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _decode_triton(
    q: Tensor,
    k_fp: Tensor,
    v_fp: Tensor,
    qtier: Optional[dict],
    scaling: float,
    num_sink: int = 0,
    n_body_win: Optional[int] = None,
    sel: Optional[Tensor] = None,
    logmass: Optional[Tensor] = None,
    centroids: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Launch the fused decode kernel.

    Returns ``(out [B,H_q,D], wsum [B,H_q,W_phys])`` — per-**window** softmax mass
    in physical order (body windows, then Q windows), which is the order
    :func:`scorer.reduce_two_tier_scores` scatters from. §5.1 replaced a
    ``[B,H_q,S]`` token-score buffer that was written, read back, rewritten and
    then reduced on the host; the reduction now happens in registers, and the
    caller is left with one gather.

    ``qtier`` (or None for an empty Q tier) carries the gathered active-window
    int2 fields shaped ``[B, n, H_kv, ...]`` plus RoPE halves ``cos``/``sin``
    ``[B, n*ws, D//2]`` and ``window_size`` — the int2 unpack + affine dequant +
    RoPE all happen inside the kernel, so no fp16 Q tensor is built.

    ``sel`` is the gate's ``[B, H_kv, n_sel]`` int32/int64 pick of active columns
    (ascending). ``None`` reads the whole tier. ``qtier`` is the **full** tier
    either way — ``sel`` is an indirection inside the kernel, not a pre-gather,
    which is what keeps the skipped windows' bytes off the wire.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton not available; fused decode requires CUDA+triton.")
    if not q.is_cuda:
        raise RuntimeError("Fused decode kernel requires CUDA tensors.")

    B, H_q, D = q.shape
    H_kv, Sfp = k_fp.shape[1], k_fp.shape[2]
    rep = H_q // H_kv
    half = D // 2

    if qtier is not None:
        n_active = int(qtier["k_codes"].shape[1])
        ws = int(qtier["window_size"])
    else:
        n_active, ws = 0, 4  # WS is a constexpr; the Q loop runs 0 times

    body = max(Sfp - num_sink, 0)
    if n_body_win is None:
        n_body_win = -(-body // ws)
    if n_body_win * ws < body:
        raise ValueError(
            f"n_body_win={n_body_win} covers {n_body_win * ws} tokens but the fp "
            f"body holds {body} (Sfp={Sfp}, num_sink={num_sink}). Every body token "
            "must fall inside a scored window or its attention mass would be "
            "dropped from the eviction scores."
        )
    W_phys = n_body_win + n_active

    # `out` is NOT scratch: it leaves this function as the model's attention
    # output and stays live until that layer's o_proj has consumed it.
    out = torch.empty((B, H_q, D), device=q.device, dtype=q.dtype)
    # `wsum` is consumed by the score gather inside `_run_fused` and `wmax` is
    # never read outside the kernel, so both are dead before the next layer runs.
    wsum = _scratch("wsum", (B, H_q, W_phys), torch.float32, q.device)
    wmax = _scratch("wmax", (B, H_q, W_phys), torch.float32, q.device)

    # Dummies for the empty-Q case: valid tensors so the pointers exist; never
    # indexed (the Q loop runs only while w0 < n_active == 0).
    dev = q.device
    gk, gv = grid_group(D), grid_group(ws)
    if qtier is None:
        kc = torch.zeros((B, 1, H_kv, D, max(ws // 4, 1)), dtype=torch.uint8, device=dev)
        kq = torch.zeros((B, 1, H_kv, D), dtype=torch.uint8, device=dev)
        kgs = torch.zeros((B, 1, H_kv, D // gk), dtype=torch.float16, device=dev)
        vc = torch.zeros((B, 1, H_kv, ws, max(D // 4, 1)), dtype=torch.uint8, device=dev)
        vq = torch.zeros((B, 1, H_kv, ws), dtype=torch.uint8, device=dev)
        vgs = torch.zeros((B, 1, H_kv, ws // gv), dtype=torch.float16, device=dev)
        cs = torch.zeros((B, 1, half), dtype=q.dtype, device=dev)
        KC, VC = kc, vc
        KS, KSS, KZ, KZS = kq, kgs, kq, kgs
        VS, VSS, VZ, VZS = vq, vgs, vq, vgs
        COS, SIN = cs, cs
    else:
        KC, VC = qtier["k_codes"], qtier["v_codes"]
        KS, KSS = qtier["k_scale"]
        KZ, KZS = qtier["k_zero"]
        VS, VSS = qtier["v_scale"]
        VZ, VZS = qtier["v_zero"]
        COS, SIN = qtier["cos"], qtier["sin"]

    # §5.2 passes shapes instead of strides for these, so contiguity stops being
    # an assumption and becomes a checked contract. It holds by construction --
    # they come from `QuantSlotTable.gather` (a fresh index_select) reshaped, and
    # from `rope_cos_sin_halves`, which calls `.contiguous()` -- so this never
    # fires in practice and costs one flag read per launch. It is here because a
    # silently non-contiguous tensor would read garbage rather than fail.
    for name, t in (("k_codes", KC), ("k_scale", KS), ("k_scale_s", KSS),
                    ("k_zero", KZ), ("k_zero_s", KZS),
                    ("v_codes", VC), ("v_scale", VS), ("v_scale_s", VSS),
                    ("v_zero", VZ), ("v_zero_s", VZS),
                    ("cos", COS), ("sin", SIN)):
        if not t.is_contiguous():
            raise RuntimeError(
                f"fused decode requires a contiguous {name}; its strides are "
                f"derived from its shape inside the kernel (DECODE_SPEED_PLAN "
                f"§5.2). Got shape {tuple(t.shape)} strides {tuple(t.stride())}."
            )

    n_sel = check_gate_selection(sel, B, H_kv, n_active)
    gated = n_sel > 0
    if gated:
        SEL, selb, selh = sel, sel.stride(0), sel.stride(1)
        if logmass is None:
            raise RuntimeError(
                "a gated fused decode needs the gate's logmass: the kernel scores "
                "the windows it skipped from their cards, and without it they "
                "would leave as zeros, rank last, and be evicted.")
        if logmass.shape != (B, H_q, n_active) or logmass.dtype != torch.float32:
            raise RuntimeError(
                f"logmass must be fp32 [B, H_q, n_active] = [{B}, {H_q}, "
                f"{n_active}]; got {tuple(logmass.shape)} {logmass.dtype}.")
        if not logmass.is_contiguous():
            raise RuntimeError(
                "fused decode requires a contiguous logmass; its strides are "
                "derived from its shape inside the kernel.")
        LOGM = logmass
        if centroids is None:
            raise RuntimeError(
                "a gated fused decode needs the value centroids: the windows it "
                "skips attend through them, and without them 75% of the tier "
                "would be silently absent from the attention output.")
        VM, VMS, VANC = centroids
        # int4, packed two per byte along D (modules/quant/sketch._q_sym4), so
        # the stored row is D//2 wide. Checked rather than inferred: a full-width
        # int8 centroid would read the right number of BYTES and the wrong
        # VALUES, which is the failure mode a shape check exists to catch.
        if VM.shape != (B, n_active, H_kv, D // 2):
            raise RuntimeError(
                f"centroid codes must be int4-packed [B, n_active, H_kv, D//2] = "
                f"[{B}, {n_active}, {H_kv}, {D // 2}]; got {tuple(VM.shape)}.")
        if VM.dtype != torch.uint8:
            raise RuntimeError(
                f"centroid codes must be uint8 (packed int4 nibbles); got "
                f"{VM.dtype}.")
        if VMS.shape != (B, n_active, H_kv):
            raise RuntimeError(
                f"centroid scales must be [B, n_active, H_kv]; got {tuple(VMS.shape)}.")
        if VANC.shape != (B, H_kv, D):
            raise RuntimeError(
                f"the value anchor must be [B, H_kv, D] = [{B}, {H_kv}, {D}]; "
                f"got {tuple(VANC.shape)}.")
        for name, t in (("vm", VM), ("vms", VMS), ("vanc", VANC)):
            if not t.is_contiguous():
                raise RuntimeError(
                    f"fused decode requires a contiguous {name}; its strides are "
                    "derived from its shape inside the kernel.")
    else:
        # Valid pointers the kernel never dereferences: the Q loop runs while
        # w0 < n_sel == 0, and the GATED blocks are compiled out entirely.
        SEL = torch.zeros((1,), dtype=torch.int32, device=dev)
        LOGM = torch.zeros((1,), dtype=torch.float32, device=dev)
        VM = torch.zeros((1,), dtype=torch.uint8, device=dev)
        VMS = torch.zeros((1,), dtype=torch.float16, device=dev)
        VANC = torch.zeros((1,), dtype=torch.float32, device=dev)
        selb = selh = 0

    BLOCK_R = _pow2_at_least(rep)

    # ---- SPLIT-KV -----------------------------------------------------------
    # `grid = (B * H_kv,)` is 8 blocks at B=1 on a 108-SM A100 and 256 at B=32
    # (2.37 waves). §11.11 measured the consequence: the step is 93%
    # batch-invariant and ~11 ms of this 14.42 ms kernel is paid regardless of
    # batch, on work that is entirely proportional to it. Splitting the window
    # axis multiplies the grid by `splits`.
    #
    # DEFAULT IS 1, i.e. OFF, and that is deliberate. At splits=1 the constexpr
    # folds every split branch away, no partial buffer is allocated, one kernel
    # launches on the old grid, and the shipped path is the pre-split path. The
    # win is a hypothesis until the A/B runs (NEXT_RUNS.md), and an
    # unvalidatable kernel change does not get to be the default on the
    # strength of an argument.
    splits = _resolve_splits(B, H_kv, n_sel if sel is not None else n_active,
                             n_body_win)
    if splits > 1:
        # Seeded on the host, not in the kernel: with several programs per
        # (b, kv) nothing orders them, so an in-kernel seed could zero a column
        # another split had already scored. See the SPLITS==1 guard there.
        if sel is not None:
            wsum[..., n_body_win:].zero_()
            wmax[..., n_body_win:].fill_(-float("inf"))
        pacc = torch.empty((B, H_kv, splits, BLOCK_R, D),
                           dtype=torch.float32, device=dev)
        pl = torch.empty((B, H_kv, splits, BLOCK_R),
                         dtype=torch.float32, device=dev)
        pm = torch.empty((B, H_kv, splits, BLOCK_R),
                         dtype=torch.float32, device=dev)
    else:
        # Valid pointers the kernel never dereferences: PHASE 0 touches neither.
        pacc = pl = pm = torch.zeros((1,), dtype=torch.float32, device=dev)
    grid = (B * H_kv * splits,)

    # Base-2 softmax: fold log2(e) into the SCALE, which is a scalar multiplied
    # into the logits the kernel already computes — so the base change costs
    # nothing per element. Folding it into the [BLOCK_R, BLOCK_T] logit tile
    # instead would add a multiply per key per query head, which is the whole
    # cost the change exists to avoid.
    #
    # The env read is latched at import, not done here: this function runs once
    # per layer per step (32x at L=32), and `os.environ.get(...).strip().lower()`
    # allocates two strings each time. A knob that is read on the hot path is a
    # cost, not a feature.
    if _EXP2[0]:
        scaling = scaling * _LOG2E

    # Tile search (see _FIT_LADDER). §5.3 widened the Q-tier tile from one window
    # (BLOCK_WS=16 in the pre-§5.3 kernel) to BLOCK_NW windows, which quadrupled
    # the `tl.dot` operand staging -- k_rlo/k_rhi go from [64,16] to [64,64] and
    # vv from [16,128] to [64,128]. At BLOCK_T=64 that is ~128 KB of dot operands
    # before Triton's pipelining multiplies it, and an A100 has 163 KB of shared
    # memory per SM. The first GPU run of this kernel hit exactly that:
    # "Required: 176128, Hardware limit: 166912".
    #
    # That arithmetic says which rungs FIT. It does not say which is FASTEST, and
    # until 2026-09-19 this loop conflated the two: it took the first rung that
    # launched. The profile settles it -- at ~128 KB of operands only one block is
    # resident per SM, so the Q-tier loop runs with nothing to hide its latency,
    # and the kernel sits at ~12% of memory bandwidth at 97.2% GPU busy. Whether
    # a smaller tile's extra iterations cost less than its extra occupancy buys
    # is not derivable from the staging arithmetic, so it is measured.
    #
    # `_search_rungs` owns the whole policy: it times every rung that fits, caches
    # the winner per signature, re-launches the winner so the caller's values come
    # from the rung the steady state will use, and raises if nothing fits. At
    # steady state this is one dict hit and one launch -- the same work the
    # first-fit path did.
    #
    # `gated` joins the signature: GATED is a constexpr, so the two variants are
    # separate compiles with different register and staging pressure, and a rung
    # that fit one is not evidence about the other. `B`, `Sfp` and `n_sel` join it
    # too, and they did not have to under first-fit: a FIT is shape-independent
    # (staging is a function of the tile), but a TIME is not -- the grid is
    # `B * H_kv` and the serial chain is `Sfp` and `n_sel` long. `Sfp` and `n_sel`
    # are bucketed by bit length so the fp store's one-token-per-step growth
    # between evictions does not re-trigger the search every step; `B` is exact
    # because it moves the grid.
    sig = (int(ws), int(D), int(BLOCK_R), int(W_phys > 0), bool(gated),
           int(B), int(Sfp).bit_length(), int(n_sel).bit_length())

    def _launch(rung):
        target_keys, num_stages, num_warps = rung
        BLOCK_NW, BLOCK_T = window_tiling(ws, target_keys)
        BLOCK_W = _pow2_at_least(min(W_phys, target_keys), floor=16)
        if splits == 1:
            _one(rung, BLOCK_NW, BLOCK_T, BLOCK_W, grid, 0)
        else:
            # Two launches, and the ORDER is the correctness argument: phase 2
            # merges what phase 1 wrote. They are separate kernel launches on
            # the same stream, so the second is ordered after the first -- a
            # single launch could not give that, which is why this is not one
            # kernel with an internal barrier.
            _one(rung, BLOCK_NW, BLOCK_T, BLOCK_W, grid, 1)
            _one(rung, BLOCK_NW, BLOCK_T, BLOCK_W, (B * H_kv,), 2)

    def _one(rung, BLOCK_NW, BLOCK_T, BLOCK_W, g, phase):
        target_keys, num_stages, num_warps = rung
        _two_tier_decode_kernel[g](
            q, k_fp, v_fp, KC, KS, KSS, KZ, KZS, VC, VS, VSS, VZ, VZS,
            COS, SIN, SEL, LOGM,
            VM, VMS, VANC,
            out, wsum, wmax,
            pacc, pl, pm,
            scaling,
            H_q, H_kv, n_active, n_sel, Sfp, rep, num_sink, n_body_win, W_phys,
            q.stride(0), q.stride(1), q.stride(2),
            k_fp.stride(0), k_fp.stride(1), k_fp.stride(2), k_fp.stride(3),
            v_fp.stride(0), v_fp.stride(1), v_fp.stride(2), v_fp.stride(3),
            selb, selh,
            HEAD_DIM=D, HALF=half, WS=ws,
            BLOCK_R=BLOCK_R, BLOCK_NW=BLOCK_NW, BLOCK_T=BLOCK_T,
            BLOCK_W=BLOCK_W,
            PACK_K=max(ws // 4, 1), PACK_V=max(D // 4, 1),
            GROUP_K=gk, GROUP_V=gv,
            GATED=gated, LOG2E=_LOG2E,
            SPLITS=splits, PHASE=phase,
            num_stages=num_stages, num_warps=num_warps,
        )

    _search_rungs(
        _FIT_CHOICE, _FIT_TIMINGS, _FIT_ANNOUNCED, _FIT_SEEN, sig, _FIT_LADDER,
        _launch, "fused decode tiling",
        key=lambda r: (window_tiling(ws, r[0]),
                       _pow2_at_least(min(W_phys, r[0]), floor=16), r[1], r[2]),
    )
    return out, wsum



# ---------------------------------------------------------------------------
# Public dispatcher — Triton-or-raise
# ---------------------------------------------------------------------------


def fused_two_tier_decode(
    q: Tensor,
    k_fp: Tensor,
    v_fp: Tensor,
    qtier: Optional[dict],
    scaling: float,
    num_sink: int = 0,
    n_body_win: Optional[int] = None,
    sel: Optional[Tensor] = None,
    logmass: Optional[Tensor] = None,
    centroids: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Fused decode attention + score. **Triton-or-raise** — no PyTorch fallback.

    q : ``[B, H_q, D]`` post-RoPE decode query.
    k_fp, v_fp : ``[B, H_kv, S_fp, D]`` fp tier (post-RoPE keys), ``[sink ‖ body]``.
    qtier : int2 Q-tier context (see :func:`_decode_triton`) or ``None`` (empty).
    num_sink : sink tokens at the head of ``k_fp``; they carry no window score.
    n_body_win : scored windows the fp body spans. Defaults to
        ``ceil((S_fp - num_sink) / ws)``; pass it when the caller already knows
        it, so the kernel's window axis matches the caller's score axis exactly.
    sel : the gate's ``[B, H_kv, n_sel]`` int32/int64 pick of active columns, or None
        to read the whole tier.

    Returns ``(out [B,H_q,D], wsum [B,H_q,W_phys])`` — per-window softmax mass in
    physical order (body windows, then Q windows). ``W_phys`` counts the **whole**
    Q tier whether or not ``sel`` gated it; a skipped window's column comes back
    ``0`` for the caller to fill from its card (see
    :func:`scorer.fill_skipped_window_scores`, and why leaving it at zero would
    evict the tier the gate exists to preserve).
    """
    if not (_HAS_TRITON and q.is_cuda):
        reason = "triton not installed" if not _HAS_TRITON else "not on CUDA"
        raise RuntimeError(
            "fused_two_tier_decode requires the Triton kernel on CUDA "
            f"({reason}); there is no PyTorch decode fallback in production "
            "(the reference is for tests only). See assert_decode_kernel_available."
        )
    return _decode_triton(q, k_fp, v_fp, qtier, scaling, num_sink, n_body_win,
                          sel, logmass, centroids)
