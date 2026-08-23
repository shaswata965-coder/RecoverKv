"""Per-key eviction scores, computed the FlashAttention-2 *backward* way.

Background — the problem in one paragraph
-----------------------------------------
Eviction needs, for every cached key ``s``, the total attention it received
summed over *all* query rows::

    token_scores[s] = Σ_i softmax(qᵢ·kᵀ)[s]        # sum DOWN the query axis

FlashAttention's forward keeps nothing of this shape — everything it keeps is
per *query* (its running sum ``ℓ_i`` has already collapsed the key axis we want
to keep). So today the flash backend reconstructs the score in a *separate*
PyTorch pass (``hooks.py``) that rebuilds ``softmax(q·kᵀ)`` and sums it — a slow,
memory-bound bolt-on that round-trips a ~20 GB probability matrix through HBM
per layer.

The fix (Design B / ``SCORE_KERNEL_PLAN.md``)
---------------------------------------------
Our score has the *exact* shape of FlashAttention-2's **backward** pass output
``dV`` — a sum over queries, for each key — with the value replaced by a column
of ones, so the "gradient matmul" degenerates into "add up each key's attention".
Running that half of the backward pass gives the score directly.

Two things make it cheap and exact:

* **``L`` is already known.** ``L_i = m_i + log(ℓ_i)`` is the exact softmax
  normaliser. Given it, ``P[i,s] = exp(scale·qᵢ·kᵀ_s − L_i)`` is exact on first
  touch — no running max, no rescale, no overflow guard (``S − L ≤ 0`` always).
* **Loop key-outer.** One program owns one key block and writes only that
  block's slice of ``token_scores`` → plain stores, **no atomics**.

The window size never enters here. This module produces ``token_scores``
``[B, H_q, S]`` at full per-key resolution; the caller reshapes it into windows
downstream (``scorer.reduce_*``). That is what keeps ``window_size`` a free knob.

What's in this file
-------------------
* :func:`compute_token_scores` — the public entry point / dispatcher.
* :func:`token_scores_torch` — pure-PyTorch reference. **Byte-identical to the
  old inline hook loop** when called with its defaults. Now used only for the
  DECODE pass (``T == 1``) and as the CPU test oracle — it is no longer a prefill
  fallback (see below). Runs on CPU.
* :func:`token_scores_from_lse` — the Design-B ``exp(S − L)`` formulation in
  PyTorch. Mathematically equal to :func:`token_scores_torch`; it exists to
  prove the kernel's math on CPU and to serve as the kernel's test oracle.
* :func:`_token_scores_triton` + ``_score_kernel`` — the fused Triton Stage-B
  kernel. GPU-only; **must be validated on a GPU against the reference** (this
  repo's dev box is CPU-only, so it ships unvalidated by construction).

Prefill scoring (``T > 1``) is **Triton-only**: :func:`compute_token_scores`
runs the fused kernel or RAISES — there is no PyTorch prefill fallback, because
that path materialises a ``[B, H_q, chunk, S]`` probability block (~8.6 GB fp16
at prefill=4096, batch=32) that OOMs the batched long-context shapes this method
exists to serve, and a silent degrade to it is worse than a hard failure. Decode
(``T == 1``) uses the PyTorch path (no grid to tile — design §6).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

# Triton is optional: absent on CPU-only / non-CUDA installs. The reference path
# never touches it, so importing this module is always safe.
try:  # pragma: no cover - import guard, exercised only where triton is present
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # ImportError, or a broken triton build
    _HAS_TRITON = False


# --- One-time, explicit terminal announcement of the chosen scoring path ------
# So a run never *silently* falls back to the reference (e.g. triton missing on
# the GPU box): the actual choice is printed once per process.

_ANNOUNCED = {"done": False}


def _prefill_backend(cuda: bool):
    """Resolve what PREFILL scoring will do: ``('triton', '')`` or ``('error', reason)``.

    Prefill scoring is Triton-only by contract (see :func:`compute_token_scores`):
    the PyTorch reconstruction materialises a ``[B, H_q, chunk, S]`` probability
    block that OOMs the batched long-context shapes this method targets, so when
    the kernel is unavailable the prefill path RAISES rather than degrading. This
    reports which case it is so the one-time banner can warn up front instead of
    the run only discovering it at the first prefill.
    """
    if _HAS_TRITON and cuda:
        return "triton", ""
    if not _HAS_TRITON:
        return "error", "triton not installed"
    if not cuda:
        return "error", "not running on CUDA"
    return "error", "unavailable"


def describe_prefill_backend(cuda: bool) -> str:
    """Human string for the prefill scoring backend, given CUDA availability."""
    kind, reason = _prefill_backend(cuda)
    if kind == "triton":
        return "TRITON fused kernel (required)"
    return f"WILL ERROR: prefill needs the Triton kernel on CUDA ({reason})"


def _announce_backend_once(q: Tensor) -> None:
    """Print, once per process, whether prefill scoring can use the Triton path."""
    if _ANNOUNCED["done"]:
        return
    _ANNOUNCED["done"] = True
    cuda = bool(getattr(q, "is_cuda", False))
    kind, reason = _prefill_backend(cuda)
    if kind == "triton":
        print(
            "[StickyKV] score kernel: TRITON path ACTIVE for prefill scoring [OK]",
            flush=True,
        )
    else:
        print(
            "[StickyKV] score kernel: TRITON UNAVAILABLE "
            f"({reason}) — prefill scoring will RAISE (no PyTorch fallback)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# GQA helper — score at KV-head granularity without expanding K
# ---------------------------------------------------------------------------
#
# repeat_kv would materialise a num_groups×-larger [B, H_q, S, D] key copy just
# to average it back to a per-window mean later. Instead we view the query heads
# as (H_kv, n_rep) and let the S-dim matmul broadcast over n_rep against the
# un-expanded keys — identical per-head scores, no num_groups× key copy. This is
# exactly what the old hook did (hooks.py:304-316), lifted here unchanged.


def _as_grouped(q: Tensor, k: Tensor):
    """Return ``(q5, kt, B, H_q, H_kv, rep, T, S, D)`` for the broadcast matmul.

    ``q5`` is ``[B, H_kv, rep, T, D]`` and ``kt`` is ``[B, H_kv, 1, D, S]`` so
    ``q5 @ kt`` broadcasts to ``[B, H_kv, rep, T, S]`` without copying K.
    """
    B, H_q, T, D = q.shape
    H_kv, S = k.shape[1], k.shape[2]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} not divisible by H_kv={H_kv}")
    rep = H_q // H_kv
    q5 = q.reshape(B, H_kv, rep, T, D)          # [B, H_kv, rep, T, D]
    kt = k.transpose(-2, -1).unsqueeze(2)       # [B, H_kv, 1, D, S]
    return q5, kt, B, H_q, H_kv, rep, T, S, D


def _causal_mask(blk_rows: Tensor, S: int, offset: int, device) -> Tensor:
    """Boolean ``[blk, S]`` mask, ``True`` where a key is in the query's future.

    Convention (unchanged from ``hooks.py``): global query row ``t`` sits at
    absolute position ``S - T + t`` (``offset = S - T``) and may attend keys
    ``0 … offset + t``. So key ``s`` is masked out for query row ``t`` iff
    ``s > offset + t``. Broadcasts over the leading head dims.
    """
    col = torch.arange(S, device=device)
    return col[None, :] > (offset + blk_rows[:, None])


# ---------------------------------------------------------------------------
# PyTorch reference — byte-identical to the old inline hook loop
# ---------------------------------------------------------------------------


def token_scores_torch(
    q: Tensor,
    k: Tensor,
    scaling: float,
    *,
    chunk: int = 1024,
    softmax_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Per-key received attention ``[B, H_q, S]`` via chunked softmax.

    This is the historic computation, extracted verbatim: recompute
    ``scale·q·kᵀ`` in query-row blocks, causally mask, softmax in
    ``softmax_dtype``, and sum down the query axis — accumulating so the full
    ``[B, H_q, T, S]`` matrix is never materialised (peak ``O(chunk · S)``).

    Called with ``softmax_dtype=bfloat16`` and ``chunk=1024`` it reproduces the
    old hook **bit-for-bit**; that is why swapping the hook to this function is a
    no-op until a different backend is selected.

    Parameters
    ----------
    q : ``[B, H_q, T, D]`` post-RoPE queries.
    k : ``[B, H_kv, S, D]`` post-RoPE (effective) keys.
    scaling : softmax logit scale (``head_dim ** -0.5`` unless the module overrides).
    chunk : query-row block size bounding peak memory.
    softmax_dtype : dtype of the softmax intermediate (bf16 halves it vs fp32
        with the same exponent range; set fp32 for the exact reduction).
    """
    q5, kt, B, H_q, H_kv, rep, T, S, D = _as_grouped(q, k)
    offset = S - T
    out = torch.zeros(B, H_kv, rep, S, device=q.device, dtype=q.dtype)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        aw = torch.matmul(q5[:, :, :, start:end, :], kt) * scaling  # [B,H_kv,rep,blk,S]
        if T > 1:
            rows = torch.arange(start, end, device=aw.device)
            aw = aw.masked_fill(_causal_mask(rows, S, offset, aw.device), float("-inf"))
        aw = F.softmax(aw, dim=-1, dtype=softmax_dtype).to(q.dtype)
        out += aw.sum(dim=-2)                                       # [B,H_kv,rep,S]
        del aw
    return out.reshape(B, H_q, S)


def token_scores_decode(
    q: Tensor,
    k: Tensor,
    scaling: float,
    *,
    softmax_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Per-key received attention for a single decode query (``T == 1``).

    The ``T == 1`` special case of :func:`token_scores_torch`: one query row, so
    there is no causal mask and the "sum over query rows" is the identity. This
    drops the ``torch.zeros`` accumulator and the chunk loop that the general path
    carries — **byte-identical** to ``token_scores_torch(q, k, ...)`` at ``T == 1``
    (a size-1 sum over the query axis changes nothing), just without the per-step
    allocation. It fires on every generated token, every layer, so the saved
    allocator traffic is not nothing at batch.
    """
    q5, kt, B, H_q, H_kv, rep, T, S, D = _as_grouped(q, k)
    if T != 1:
        raise ValueError(f"token_scores_decode is the T==1 path only, got T={T}")
    aw = torch.matmul(q5, kt) * scaling                # [B, H_kv, rep, 1, S]
    p = F.softmax(aw, dim=-1, dtype=softmax_dtype).to(q.dtype)
    return p.reshape(B, H_q, S)                        # size-1 query axis folds away


# ---------------------------------------------------------------------------
# Design-B formulation in PyTorch — the exp(S − L) math, and its L source
# ---------------------------------------------------------------------------


def compute_lse(
    q: Tensor,
    k: Tensor,
    scaling: float,
    *,
    chunk: int = 1024,
) -> Tensor:
    """Softmax normaliser ``L = logsumexp_s(scale·q·kᵀ)`` → ``[B, H_q, T]`` (fp32).

    This is the one row-reduction the Stage-B kernel needs up front. It is
    computed here in bounded query-row chunks (never materialising the full
    logit matrix).

    In the intended end state this is **not computed at all** — flash's forward
    already produces ``L`` and can hand it out (``softmax_lse``), making Stage B a
    single extra ``q·kᵀ`` pass instead of two. Until that handoff is wired, this
    provides ``L`` self-contained so the kernel is usable and testable in
    isolation. See ``SCORE_KERNEL_PLAN.md`` §4 / Flag 2.
    """
    q5, kt, B, H_q, H_kv, rep, T, S, D = _as_grouped(q, k)
    offset = S - T
    lse = torch.empty(B, H_kv, rep, T, device=q.device, dtype=torch.float32)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        aw = torch.matmul(q5[:, :, :, start:end, :], kt).float() * scaling
        if T > 1:
            rows = torch.arange(start, end, device=aw.device)
            aw = aw.masked_fill(_causal_mask(rows, S, offset, aw.device), float("-inf"))
        lse[:, :, :, start:end] = torch.logsumexp(aw, dim=-1)
        del aw
    return lse.reshape(B, H_q, T)


def token_scores_from_lse(
    q: Tensor,
    k: Tensor,
    scaling: float,
    lse: Tensor,
    *,
    chunk: int = 1024,
    out_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Design-B score in PyTorch: ``Σ_i exp(scale·qᵢ·kᵀ − Lᵢ)`` → ``[B, H_q, S]``.

    Mathematically identical to :func:`token_scores_torch` (because
    ``exp(S − logsumexp(S)) ≡ softmax(S)``), but expressed the way the Triton
    kernel computes it — from a precomputed ``L``, with no running max. It is the
    kernel's CPU oracle: ``triton == this == token_scores_torch`` (to fp rounding).

    ``lse`` is ``[B, H_q, T]`` (fp32), e.g. from :func:`compute_lse` or the
    forward pass.
    """
    q5, kt, B, H_q, H_kv, rep, T, S, D = _as_grouped(q, k)
    lse5 = lse.reshape(B, H_kv, rep, T)
    offset = S - T
    out = torch.zeros(B, H_kv, rep, S, device=q.device, dtype=out_dtype)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        aw = torch.matmul(q5[:, :, :, start:end, :], kt).float() * scaling  # [B,H_kv,rep,blk,S]
        p = torch.exp(aw - lse5[:, :, :, start:end, None])                  # exact softmax
        if T > 1:
            rows = torch.arange(start, end, device=aw.device)
            # exp of −inf-masked logits is already ~0, but mask explicitly so
            # future keys contribute exactly 0 regardless of rounding.
            p = p.masked_fill(_causal_mask(rows, S, offset, aw.device), 0.0)
        out += p.sum(dim=-2).to(out_dtype)                                  # [B,H_kv,rep,S]
        del aw, p
    return out.reshape(B, H_q, S)


# ---------------------------------------------------------------------------
# Triton Stage-B kernel — the fused fast path (GPU-only, validate on GPU)
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _score_kernel(
        Q, K, LSE, OUT,
        scale,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_lb, stride_lh, stride_lt,
        stride_ob, stride_oh, stride_os,
        T, S, OFFSET, H_q, num_groups,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """One program == one key block, for one (batch, query-head).

        Walks the query blocks that can see this key block, accumulates
        ``Σ_i exp(scale·qᵢ·kᵀ − Lᵢ)`` per key (a colsum over queries), and writes
        this key block's slice of ``token_scores``. Disjoint key slices ⇒ plain
        stores, no atomics. ``exp(S − L)`` needs no running max because ``L`` is
        already final.
        """
        pid_n = tl.program_id(0)        # key block
        pid_bh = tl.program_id(1)       # b * H_q + h_q
        b = pid_bh // H_q
        h = pid_bh % H_q
        kv = h // num_groups            # GQA: which KV head this query head reads

        ks = pid_n * BLOCK_N
        offs_n = ks + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        n_mask = offs_n < S

        # K for this block/head, loaded once and reused across every query block.
        k_ptrs = (K + b * stride_kb + kv * stride_kh
                  + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)   # [BLOCK_N, D]

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        # Causal skip: query t sees key s iff s ≤ OFFSET + t. The smallest key in
        # this block is ks, so no query below (ks − OFFSET) can see any of it —
        # start the loop at that block boundary and skip everything before it.
        if IS_CAUSAL:
            m_lo = ks - OFFSET
            m_lo = tl.maximum(m_lo, 0)
            m_start = (m_lo // BLOCK_M) * BLOCK_M
        else:
            m_start = 0

        for m in range(m_start, T, BLOCK_M):
            offs_m = m + tl.arange(0, BLOCK_M)
            m_mask = offs_m < T
            q_ptrs = (Q + b * stride_qb + h * stride_qh
                      + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd)
            q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)   # [BLOCK_M, D]

            s = tl.dot(q_tile, tl.trans(k_tile)) * scale               # [BLOCK_M, BLOCK_N]

            lse_i = tl.load(LSE + b * stride_lb + h * stride_lh + offs_m * stride_lt,
                            mask=m_mask, other=0.0)                     # [BLOCK_M]
            p = tl.exp(s - lse_i[:, None])                             # exact softmax

            valid = m_mask[:, None] & n_mask[None, :]
            if IS_CAUSAL:
                # key ≤ OFFSET + query  ⇔  offs_n − offs_m ≤ OFFSET
                valid = valid & ((offs_n[None, :] - offs_m[:, None]) <= OFFSET)
            p = tl.where(valid, p, 0.0)

            acc += tl.sum(p, axis=0)                                    # colsum → [BLOCK_N]

        out_ptrs = OUT + b * stride_ob + h * stride_oh + offs_n * stride_os
        tl.store(out_ptrs, acc, mask=n_mask)


def _token_scores_triton(
    q: Tensor,
    k: Tensor,
    scaling: float,
    lse: Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    out_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Launch the fused Stage-B kernel. Returns ``[B, H_q, S]`` (``out_dtype``).

    ``lse`` is ``[B, H_q, T]`` fp32. Grid is ``(⌈S/block_n⌉, B·H_q)``: one
    program per key block per (batch, query-head).
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton not available; use backend='torch'.")
    if not q.is_cuda:
        raise RuntimeError("Triton score kernel requires CUDA tensors.")

    B, H_q, T, D = q.shape
    H_kv, S = k.shape[1], k.shape[2]
    num_groups = H_q // H_kv
    lse = lse.contiguous().float()

    out = torch.empty(B, H_q, S, device=q.device, dtype=out_dtype)
    grid = (triton.cdiv(S, block_n), B * H_q)
    _score_kernel[grid](
        q, k, lse, out,
        scaling,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        T, S, S - T, H_q, num_groups,
        HEAD_DIM=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=(T > 1),
    )
    return out


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def compute_token_scores(
    q: Tensor,
    k: Tensor,
    scaling: float,
    *,
    lse: Optional[Tensor] = None,
    chunk: int = 1024,
    softmax_dtype: torch.dtype = torch.bfloat16,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Per-key received attention ``[B, H_q, S]`` — the score the policy needs.

    Prefill passes (``T > 1``) run the fused Triton kernel or RAISE — there is no
    PyTorch prefill fallback (it OOMs the batched long-context shapes this method
    targets). Decode passes (``T == 1``) always use the PyTorch path — there is no
    grid to tile, so the kernel would only add launch overhead (design §6).

    Parameters
    ----------
    q : ``[B, H_q, T, D]`` post-RoPE queries.
    k : ``[B, H_kv, S, D]`` post-RoPE (effective) keys.
    scaling : logit scale.
    lse : optional ``[B, H_q, T]`` softmax normaliser from the forward pass. Only
        used by the kernel path; when ``None`` it is computed via
        :func:`compute_lse`. Feeding it from the forward is the remaining
        optimisation that turns Stage B into a single extra ``q·kᵀ`` pass.
    chunk : query-row block size for the PyTorch paths.
    softmax_dtype : softmax intermediate dtype for the ``torch`` backend
        (bf16 default reproduces the historic hook exactly).
    out_dtype : dtype of the returned scores. Defaults to ``q.dtype`` for the
        ``torch`` backend (byte-identical history) and fp32 for the kernel.

    Returns
    -------
    ``[B, H_q, S]`` — hand straight to ``scorer.reduce_*``. Window size never
    enters this function.
    """
    T = q.shape[2]
    _announce_backend_once(q)

    # -- Prefill (T > 1): Triton-only, by contract. --------------------------
    # No PyTorch fallback: token_scores_torch would materialise a
    # [B, H_q, chunk, S] probability block (~8.6 GB fp16 at prefill=4096, bs=32,
    # chunk=1024) that OOMs the very batched long-context shapes this method
    # targets. A silent degrade to that path is worse than a hard failure, so
    # when the kernel is unavailable we RAISE instead of running it.
    if T > 1:
        if not (_HAS_TRITON and q.is_cuda):
            raise RuntimeError(
                "Prefill score computation requires the Triton kernel on CUDA; "
                "the PyTorch prefill fallback has been removed "
                f"(has_triton={_HAS_TRITON}, "
                f"is_cuda={bool(getattr(q, 'is_cuda', False))}). The reference "
                "path materialises a [B, H_q, chunk, S] score matrix that OOMs "
                "batched long-context prefills. Install triton and run on GPU; "
                "for CPU/eager smoke tests use the eager backend, which reads "
                "attn_weights directly and never calls this path."
            )
        if lse is None:
            # L was not handed down from the forward
            # (STICKYKV_SCORE_LSE_FROM_FORWARD off): recompute it. NOTE this
            # still materialises a chunked logit block; feed `lse` from flash's
            # softmax_lse to make Stage B a single extra q·kᵀ pass with no such
            # transient.
            lse = compute_lse(q, k, scaling, chunk=chunk)
        return _token_scores_triton(
            q, k, scaling, lse, out_dtype=out_dtype or torch.float32
        )

    # -- Decode (T == 1): PyTorch fast path. ---------------------------------
    # There is no grid to tile for a single query row, so the kernel would only
    # add launch overhead (design §6). token_scores_decode is the T==1 special
    # case of token_scores_torch — byte-identical, minus the accumulator + loop.
    scores = token_scores_decode(q, k, scaling, softmax_dtype=softmax_dtype)
    if out_dtype is not None and scores.dtype != out_dtype:
        scores = scores.to(out_dtype)
    return scores
