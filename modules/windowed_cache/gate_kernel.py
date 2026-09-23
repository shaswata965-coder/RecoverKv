"""Triton gate: score every int2 window from its card, pick the top ``n_sel``.

What it replaces
----------------
Without the gate the decode step dequantizes every active int2 window to find
out which ones mattered (design §8). This kernel reads each window's card --
at the default widths ``mu`` int4 and ``v``/``t`` int8 with fp16 scales, 272 B
per head with the value centroid, against a 6432 B window; ``quant_card_bits``
sets each field to 8, 4 or 2 bits -- and emits the compacted slot list the
Q-tier loop should actually visit, plus the log-mass estimate that every
**skipped** window contributes to ``window_scores``.

Shape of the work
-----------------
Two launches. ``_gate_kernel`` scores the cards: one program per
``(batch, KV head, window tile)``, reading a window's card once for all ``rep``
query heads that share it. Two dot products per window give::

    m = q.anchor + q.mu        g = q.v
    x_i     = scale * (m + t_i * g)                per-token estimate
    logmass = logsumexp_i x_i                      what a window is worth to a head

``_gate_share_kernel`` then turns ``logmass`` into the selection score, one
program per ``(batch, KV head)``::

    share_h(w) = logmass_h(w) - logsumexp_w' logmass_h(w')   per query head
    score(w)   = max over the group's heads of share_h(w)    the GQA union

The union has to bind on the KV head, because that is the unit of work: one
decode program loads a window once for the whole group. Capping per query head
and unioning afterwards would let a 0.25 ratio read up to ``0.25 * rep`` of the
tier (100% at Llama-3.1-8B's rep=4).

**Why the union is over shares and not over raw logits.** It used to be the max
of the raw ``max_i x_i`` over the group, computed inside ``_gate_kernel``. A raw
logit carries its head's baseline (``q.anchor``) and scale (``|q|``), and a
softmax is invariant to both, so that max was decided by whichever head had the
largest baseline, and the other heads were read only where it agreed. The share
is what a head's own softmax computes, so it compares heads on one footing. The
normaliser is a reduction over every window, which the card kernel cannot do
because it splits windows across programs. That is the only reason for the
second launch. :func:`modules.quant.sketch.group_share` has the measurement.

A separate kernel, for now
--------------------------
This could be a prologue inside ``_two_tier_decode_kernel`` -- one program
already owns the right ``(batch, KV head)`` -- and that is where it belongs
eventually, because the decode path is launch-bound and the card scan costs
one extra launch per layer per step (the share union is a second, small one).
It is separate first because a prologue competes for
the main kernel's register budget: its Q-tier loop already stages ten
``[D/2, BLOCK_T]`` fp32 tiles and the autotuner falls back silently when that
budget is exceeded, so a rung dropped by the prologue would cost more than the
gate saves and would produce no error. Land it standalone, measure the launch
cost against the traffic saved, then fold it in with the rung pinned.

Backend contract (mirrors ``score_kernel`` / ``decode_kernel``)
--------------------------------------------------------------
* :func:`gate_reference` -- pure-PyTorch oracle. Defines correctness, runs on
  CPU, and is the Triton kernel's test oracle.
* ``_gate_kernel`` + ``_gate_share_kernel`` + :func:`_gate_triton` -- the GPU
  kernels. **GPU-only, ship unvalidated by construction** (this repo's dev box
  is CPU-only), like every other Triton kernel here.
  :func:`gate_share_tiled_reference` mirrors the second one's tiling on CPU.
* :func:`fused_gate` -- dispatcher: Triton on CUDA, reference otherwise.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from modules.quant.sketch import _as_bits, card_field_layout, parse_card_bits

try:  # pragma: no cover - import guard, exercised only where triton is present
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # ImportError, or a broken triton build
    _HAS_TRITON = False


def gate_reference(
    q: Tensor,
    mu_q: Tensor, mu_s: Tensor,
    v_q: Tensor, v_s: Tensor,
    t_q: Tensor, t_s: Tensor,
    vm_q: Tensor, vm_s: Tensor,
    anchor: Tensor,
    scaling: float,
    n_sel: int,
    bits=None,
) -> Tuple[Tensor, Tensor]:
    """``(sel, logmass)`` — the oracle.

    Parameters
    ----------
    q : ``[B, H_q, D]`` post-RoPE decode query.
    mu_q .. vm_s : card fields, ``[B, Nw, H_kv, ...]``. ``vm_*`` is the value
        centroid; the gate does not read it (the decode kernel does, for the
        windows this selection skips) and it is taken only to keep the card one
        object.
    anchor : ``[B, H_kv, D]`` or ``[H_kv, D]``.
    n_sel : windows to keep per ``(row, KV head)``.
    bits : the card's :class:`~modules.quant.sketch.CardBits`; ``None`` is the
        shipped card.

    Returns
    -------
    sel : ``[B, H_kv, n_sel]`` int32 column indices, **ascending**. See
        :func:`_sorted_pick` for why the order is free to the result and not
        free to the memory system.
    logmass : ``[B, H_q, Nw]`` the estimate for every window.

    The pick is the top ``n_sel`` by :func:`~modules.quant.sketch.group_share`:
    the largest share of its own int2 mass that any query head of the group puts
    on a window. It used to be the top ``n_sel`` by the group max of the RAW
    estimate, which let the head with the largest baseline logit choose for the
    whole group; see ``group_share`` for the measurement.
    """
    from modules.quant.sketch import Sketch, gate_and_score, group_share

    card = Sketch(mu_q, mu_s, v_q, v_s, t_q, t_s, vm_q, vm_s)
    logmass, _ = gate_and_score(q, card, anchor, scaling, parse_card_bits(bits))
    hkv = mu_q.shape[2]
    top = group_share(logmass, hkv).topk(n_sel, dim=-1).indices    # [B,Hkv,n_sel]
    return _sorted_pick(top), logmass


def _sorted_pick(top: Tensor) -> Tensor:
    """Top-k indices as an ascending int32 ``sel``.

    The decode kernel places each window's score at ``n_body_win + widx``, its
    own column, so **the result cannot depend on this order** — that is what
    ``test_selection_order_does_not_change_the_result`` pins, and it is why the
    sort was dropped as "a launch for nothing".

    The launch was for something, and it was not the arithmetic. the design notes: under ``SEL`` the Q-tier loop's ``widx`` stops being affine, so ``KS``,
    ``KZ``, ``KC``, ``VC``, ``VS``, ``VZ`` and ``COS``/``SIN`` all become gathers,
    and the kernel runs **7.03 ms against a 0.58 ms roofline — 12x off**, which is
    the signature of a gather-bound kernel and nothing else in that measurement
    explains it. ``topk`` returns indices in descending *score* order, which is
    arbitrary in *column* space: each tile of ``BLOCK_NW`` windows then touches
    ``BLOCK_NW`` segments scattered across the whole tier. Ascending order makes
    the same segments monotone and locally clustered (mean stride ``1/ratio``
    windows at ratio ``r``), which is what L2, the TLB and the coalescer are
    built for.

    Cost is one small sort per layer per step — ``[B, H_kv, n_sel]``, ~11.5k
    elements at the headline cell — against a cache-side launch budget the same
    document measures at 8.6% of the step's total. It does reorder an
    online-softmax accumulation, so ``out`` moves in the last bits exactly as any
    tile-order change does; it does not move *which* windows are read or *which*
    column each score lands on, so no eviction decision is reassociated.

    **The int32 cast is gone (W6).** ``topk`` returns int64 indices, so casting
    cost a whole extra kernel launch per layer per step — 32 per step at L=32 —
    to save 4 bytes per selected window, which at the headline cell is ~46 KB of
    a 1,636 MB step. The kernel already does ``tl.load(SEL + ...).to(tl.int32)``
    after the load, and Triton's pointer arithmetic is in elements, so an int64
    ``SEL`` indexes identically; ``check_gate_selection`` now accepts either.
    That trade was upside-down on a path where the design notes measures the
    host gap at 37% of the step and ~17 us exposed per launch.
    """
    return top.sort(dim=-1).values


#: Winning ``(BLOCK_W, num_warps)`` per geometry, its measured ladder, and the
#: signatures already announced. Same contract as ``decode_kernel``'s: the search
#: runs once per signature and every later launch is one dict hit.
#:
#: **What a rung can and cannot move.** ``BLOCK_W`` is provably bit-identical:
#: each program owns a disjoint span of output columns and every reduction here
#: -- the ``logsumexp`` over ``WS`` -- lives entirely inside one window, so which
#: program owns which window changes nothing about the arithmetic. ``num_warps``
#: is NOT: it changes the lane layout of the ``tl.sum`` over ``HEAD_DIM``, so
#: ``logmass`` moves in its last bits, and ``logmass`` feeds the share the top-k
#: ranks on. A last-bit move can therefore swap the k-th
#: and (k+1)-th window when the two are already tied to ~1e-7 -- windows whose
#: estimated mass is equal to seven digits. That is the same class of
#: perturbation the tile ladder next door already ships, and the same one
#: ``score_kernel`` already autotunes ``num_warps`` across in prefill; it is
#: recorded here rather than hidden because it is the ONE property this search
#: moves.
_GATE_CHOICE: dict = {}
_GATE_TIMINGS: dict = {}
_GATE_ANNOUNCED: set = set()
_GATE_SEEN: dict = {}


def gate_choice() -> dict:
    """``{sig: (BLOCK_W, num_warps)}`` -- the gate's chosen launch, per geometry."""
    return dict(_GATE_CHOICE)


def gate_timings() -> dict:
    """``{sig: [((BLOCK_W, num_warps), ms), ...]}`` -- what the search measured."""
    return {k: list(v) for k, v in _GATE_TIMINGS.items()}


def gate_window_tiles(rows: int, n_windows: int, sm_count: int) -> int:
    """``BLOCK_W`` — how many windows one gate program owns.

    The gate used to run one program per ``(row, KV head)`` and walk every window
    inside it. At ``B=1`` on Llama-3.1-8B that is **8 programs**, which on a
    108-SM A100 leaves 93% of the machine idle while a single serial loop grinds
    through the whole card set. Splitting the window axis across programs costs
    nothing in coordination — each program owns disjoint output columns and the
    top-k runs afterwards on the host — so the only question is how finely.

    Small enough to fill the machine, large enough that the per-program prologue
    (the query, the anchor) is not the whole cost: aim for roughly two waves of
    programs, then clamp to a power of two in ``[16, 64]``.

    ``rows`` is ``B * H_kv``, the programs a single tile already gives.
    """
    if n_windows <= 0:
        raise ValueError(f"n_windows must be positive, got {n_windows}")
    want = max(1, -(-2 * max(sm_count, 1) // max(rows, 1)))
    for bw in (64, 32, 16):
        if -(-n_windows // bw) >= want:
            return bw
    return 16


#: Windows per tile in ``_gate_share_kernel``'s two passes. Any power of two is
#: correct -- the union is exact whatever the tiling -- and the tensor it walks is
#: ``[REP, NW]`` fp32 per program, so this only trades loop trips for registers.
GATE_SHARE_BLOCK_W = 128


if _HAS_TRITON:  # pragma: no cover - GPU-only

    @triton.jit
    def _card_codes(ROW, lane, mask, BITS: tl.constexpr):
        """One card field's signed codes, as fp32, at its stored width.

        ``ROW`` points at each window's field (``[BLOCK_W, 1]`` pointers),
        ``lane`` is the element index (``[1, n]``). ``BITS`` is a constexpr, so
        only one branch is traced and a field at its shipped width compiles to
        the load it always was: int8 is a plain load; a packed width reads byte
        ``lane // (8 // BITS)`` and shifts ``BITS * (lane % (8 // BITS))``, then
        removes the ``2**(BITS-1)`` bias -- ``modules.quant.sketch._q_symb``'s
        lane map, and at 4 bits exactly the nibble unpack this kernel had
        inline. Masked lanes decode to a finite value that every caller already
        multiplies by a masked-to-zero operand or forces to ``-inf``.
        """
        if BITS == 8:
            codes = tl.load(ROW + lane, mask=mask, other=0).to(tl.float32)
        else:
            PER: tl.constexpr = 8 // BITS
            raw = tl.load(ROW + lane // PER, mask=mask, other=0)
            shift = (BITS * (lane % PER)).to(tl.uint8)
            codes = (((raw >> shift) & ((1 << BITS) - 1)).to(tl.float32)
                     - (1 << (BITS - 1)))
        return codes

    @triton.jit
    def _gate_kernel(
        Q, MU, MUS, V, VS, T, TS, ANCH,
        LOGM,
        qb, qh, pmb, pmn, pmh, mub, mun, muh, sb, sn, sh, tb, tn, th, ab, ah,
        lb, lh,
        NW, REP, SCALE,
        HEAD_DIM: tl.constexpr, WS: tl.constexpr, BLOCK_WS: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_W: tl.constexpr,
        MU_BITS: tl.constexpr, V_BITS: tl.constexpr, T_BITS: tl.constexpr,
    ):
        """One program per ``(row, KV head, window tile)``.

        Three things about the shape of this loop are load-bearing.

        **The cards are loaded once, outside the query-head loop.** One program
        owns a whole GQA group precisely so a window's card is read once for all
        ``REP`` query heads sharing it; reading it inside the ``r`` loop instead
        would multiply the card traffic by ``REP`` (4x on Llama-3.1-8B) and throw
        away the only reason the group is grouped.

        **The window axis is split across programs**, not walked serially inside
        one. See :func:`gate_window_tiles`.

        **Only ``logmass`` leaves, per query head.** This kernel used to also
        store the group max of the raw estimate, and the top-k ranked on it. That
        max compared raw logits across query heads, which is the defect
        :func:`modules.quant.sketch.group_share` describes. The union needs each
        head's logsumexp over EVERY window, and this kernel splits windows across
        programs, so the union moved to :func:`_gate_share_kernel`.

        ``BLOCK_WS`` is ``WS`` rounded up to a power of two, because ``tl.arange``
        admits nothing else. The surplus lanes are masked out of the ``t`` load
        **and** forced to ``-inf`` in ``x``: a padding lane that merely loaded
        zero would still carry the real ``SCALE * m`` into the max and the
        logsumexp, so a ``ws`` of 12 would score every window as if it held a
        13th token sitting exactly at its mean.

        No ``bound`` is computed. The margin rule that needed it has no caller on
        this path (``WindowedCache._gate_ctx`` refuses a finite margin), so the
        bound cost a ``[B, H_q, NW]`` fp32 store and the whole ``eps`` field of
        every card, to be discarded by :func:`_gate_triton`.
        """
        b = tl.program_id(0)
        kv = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        dm = d < HEAD_DIM
        w = tl.arange(0, BLOCK_WS)
        wm = w < WS

        cols = tl.program_id(2) * BLOCK_W + tl.arange(0, BLOCK_W)
        cm = cols < NW

        # ---- the cards for this tile: read ONCE for the whole query group ----
        # Each field at its `quant_card_bits` width (`_card_codes`). At the
        # shipped card mu is int4 packed two per byte along D -- element `d` is
        # the nibble at byte `d // 2`, shift `4 * (d % 2)`, biased by +8 so it
        # is unsigned, the decode kernel's int2 crumb idiom one width up -- and
        # v and t are int8.
        mu = _card_codes(MU + b * pmb + cols[:, None] * pmn + kv * pmh, d[None, :],
                         cm[:, None] & dm[None, :], MU_BITS)
        mus = tl.load(MUS + b * sb + cols * sn + kv * sh,
                      mask=cm, other=0.0).to(tl.float32)
        vv = _card_codes(V + b * mub + cols[:, None] * mun + kv * muh, d[None, :],
                         cm[:, None] & dm[None, :], V_BITS)
        vs = tl.load(VS + b * sb + cols * sn + kv * sh,
                     mask=cm, other=0.0).to(tl.float32)
        tt = _card_codes(T + b * tb + cols[:, None] * tn + kv * th, w[None, :],
                         cm[:, None] & wm[None, :], T_BITS)
        ts = tl.load(TS + b * sb + cols * sn + kv * sh,
                     mask=cm, other=0.0).to(tl.float32)

        # The anchor is indexed by KV head alone, so it is constant in r too.
        anc = tl.load(ANCH + b * ab + kv * ah + d, mask=dm, other=0.0).to(tl.float32)

        for r in range(0, REP):
            hq = kv * REP + r
            q_v = tl.load(Q + b * qb + hq * qh + d, mask=dm, other=0.0).to(tl.float32)
            base = tl.sum(q_v * anc, axis=0)

            m = base + mus * tl.sum(mu * q_v[None, :], axis=1)       # [BLOCK_W]
            g = vs * tl.sum(vv * q_v[None, :], axis=1)               # [BLOCK_W]

            x = SCALE * (m[:, None] + (ts[:, None] * tt) * g[:, None])
            x = tl.where(wm[None, :], x, -float("inf"))
            xm = tl.max(x, axis=1)
            # logsumexp, written out: the online-softmax form the decode kernel
            # uses, and `torch.logsumexp(` must stay unique in modules/ for
            # audit_e2e.py's attribution (tests/test_audit_e2e.py).
            lm = xm + tl.log(tl.sum(tl.exp(x - xm[:, None]), axis=1))
            tl.store(LOGM + b * lb + hq * lh + cols,
                     tl.where(cm, lm, -float("inf")), mask=cm)

    @triton.jit
    def _gate_share_kernel(
        LOGM, EST,
        lb, lh, eb, eh,
        NW, REP,
        BLOCK_R: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        """The GQA union, on shares. One program per ``(row, KV head)``.

        Pass 1 is an online logsumexp over EVERY window of each of the group's
        ``REP`` query heads, which is each head's int2-tier normaliser. Pass 2
        stores, per window, the max over those heads of
        ``logmass - normaliser``: the largest share of its own int2 mass any head
        of the group puts on the window. The top-k then ranks on that.

        Both passes read only ``LOGM``, which ``_gate_kernel`` wrote in the
        previous launch, so ordering is the stream's, not this program's. The
        traffic is ``2 * REP * NW`` fp32 per program against the card kernel's
        ``NW`` cards, about 3% of it at ``D = 128``.

        Masking, both kinds, and why neither can produce a NaN:

        * a padding query-head lane (``r >= REP``, when ``REP`` is not a power of
          two) loads ``-inf`` everywhere, so its running max stays ``-inf``. Its
          normaliser is forced to ``0`` and its shares to ``-inf``, so it never
          wins the max and never evaluates ``-inf - -inf``.
        * a padding window lane (``cols >= NW``) loads ``-inf``, which adds
          ``exp(-inf) = 0`` to the sum. The running max is shifted through
          ``m_safe`` so ``exp(m - m_new)`` is never ``exp(-inf - -inf)``.

        :func:`gate_share_tiled_reference` mirrors this tile for tile on CPU.
        """
        b = tl.program_id(0)
        kv = tl.program_id(1)
        r = tl.arange(0, BLOCK_R)
        rm = r < REP
        hq = kv * REP + r
        offs_w = tl.arange(0, BLOCK_W)

        m = tl.full([BLOCK_R], -float("inf"), tl.float32)
        s = tl.zeros([BLOCK_R], tl.float32)
        for w0 in range(0, NW, BLOCK_W):
            cols = w0 + offs_w
            msk = rm[:, None] & (cols < NW)[None, :]
            x = tl.load(LOGM + b * lb + hq[:, None] * lh + cols[None, :],
                        mask=msk, other=-float("inf"))
            m_new = tl.maximum(m, tl.max(x, axis=1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            s = s * tl.exp(m - m_safe) + tl.sum(tl.exp(x - m_safe[:, None]), axis=1)
            m = m_new
        lse = tl.where(rm & (m > -float("inf")), m + tl.log(s), 0.0)

        for w0 in range(0, NW, BLOCK_W):
            cols = w0 + offs_w
            cm = cols < NW
            msk = rm[:, None] & cm[None, :]
            x = tl.load(LOGM + b * lb + hq[:, None] * lh + cols[None, :],
                        mask=msk, other=-float("inf"))
            share = tl.where(msk, x - lse[:, None], -float("inf"))
            tl.store(EST + b * eb + kv * eh + cols, tl.max(share, axis=0), mask=cm)


def gate_share_tiled_reference(logmass: Tensor, num_kv_heads: int,
                               block_w: int = GATE_SHARE_BLOCK_W) -> Tensor:
    """CPU mirror of ``_gate_share_kernel``, tile for tile. ``[B,Hq,NW] -> [B,Hkv,NW]``.

    Not a shortcut to :func:`modules.quant.sketch.group_share`. It walks the
    window axis in ``block_w`` tiles with the kernel's online logsumexp,
    including the ``m_safe`` shift and the ``-inf`` padding lanes, so the
    tiling and masking are what the tests check against the spec. That is the
    part a CPU box can verify. The Triton lowering is not.
    """
    B, Hq, NW = logmass.shape
    rep = Hq // num_kv_heads
    x = logmass.to(torch.float32).reshape(B, num_kv_heads, rep, NW)
    ninf = float("-inf")
    m = torch.full((B, num_kv_heads, rep), ninf)
    s = torch.zeros((B, num_kv_heads, rep))
    for w0 in range(0, NW, block_w):
        cols = torch.arange(w0, w0 + block_w)
        valid = cols < NW
        tile = torch.full((B, num_kv_heads, rep, block_w), ninf)
        tile[..., valid] = x[..., cols[valid]]
        m_new = torch.maximum(m, tile.amax(-1))
        m_safe = torch.where(m_new == ninf, torch.zeros_like(m_new), m_new)
        s = s * torch.exp(m - m_safe) + torch.exp(tile - m_safe[..., None]).sum(-1)
        m = m_new
    lse = torch.where(m > ninf, m + torch.log(s), torch.zeros_like(m))
    return (x - lse[..., None]).amax(2)


def _sm_count(device) -> int:  # pragma: no cover - GPU-only
    try:
        return int(torch.cuda.get_device_properties(device).multi_processor_count)
    except Exception:
        return 64


def _check_card_layout(card, bits, D: int, ws: int) -> None:
    """Validate every card field against its width.

    A packed field's shape cannot say its own width once ``D`` is not known, and
    the kernel derives nothing from the shapes it is handed but these extents.
    So they are checked rather than inferred: a field at the wrong width reads
    the right number of BYTES and the wrong VALUES, which is the failure a shape
    check exists to catch. Run once per launch signature, not per step: every
    card in a run comes from one slot table, built at one width.
    """
    mu_q, _, v_q, _, t_q, _, vm_q, _ = card
    for name, t, n, w in (("mu", mu_q, D, bits.mu), ("v", v_q, D, bits.v),
                          ("t", t_q, ws, bits.t), ("vm", vm_q, D, bits.vm)):
        width, dtype = card_field_layout(n, w)
        if t.shape[-1] != width or t.dtype != dtype:
            raise RuntimeError(
                f"card field {name} at {w} bits must be {dtype} with a last axis "
                f"of {width} (n={n}); got {t.dtype} {tuple(t.shape)}. The card "
                "and the `bits` it was passed with disagree.")


def _gate_triton(q, card, anchor, scaling, n_sel, bits):  # pragma: no cover - GPU-only
    if not _HAS_TRITON:
        raise RuntimeError("fused_gate requires triton")
    mu_q, mu_s, v_q, v_s, t_q, t_s, _vm_q, _vm_s = card
    # The head dim comes from the query: every card field may be packed, and
    # reading it off one would shrink every downstream extent by its pack.
    B, NW, HKV = mu_q.shape[0], mu_q.shape[1], mu_q.shape[2]
    D = q.shape[-1]
    HQ = q.shape[1]
    ws = t_q.shape[-1] * 8 // bits.t           # t may be packed along ws
    if anchor.dim() == 2:
        anchor = anchor.unsqueeze(0).expand(B, HKV, D).contiguous()
    # Both are dead by the end of `_run_fused`: `est` (the union score) is
    # consumed by the topk below, `logm` by the decode kernel's §5 fill in the
    # same call. Reused across layers within a step -- see
    # `decode_kernel._scratch` (W4/D5).
    from .decode_kernel import _scratch
    logm = _scratch("logm", (B, HQ, NW), torch.float32, q.device)
    est = _scratch("est", (B, HKV, NW), torch.float32, q.device)
    # `BLOCK_W` and `num_warps` are MEASURED, not derived -- the same change
    # `decode_kernel` §5.3 made to its tile ladder, for the same reason. This
    # kernel reads 16 MB of cards per layer at the headline cell and takes 3.63 ms
    # doing it (~140 GB/s on a 1555 GB/s part, 2026-09-19 profile), which is the
    # same gather-bound, occupancy-starved shape the decode kernel is in. Until
    # now it ran at whatever `gate_window_tiles` derived and at Triton's DEFAULT
    # four warps -- a value never chosen, only inherited.
    #
    # `gate_window_tiles` is kept as the ladder's first entry rather than deleted:
    # it encodes the "fill the machine" floor, and at B=1 (8 rows) that floor is
    # the whole argument for splitting the window axis at all. What it cannot do
    # is price the split, which is what the search adds.
    from .decode_kernel import _search_rungs
    derived = gate_window_tiles(B * HKV, NW, _sm_count(q.device))
    ladder = [(derived, 4)]
    ladder += [(bw, nw) for bw in (16, 32, 64, 128) for nw in (4, 8)
               if (bw, nw) != (derived, 4)]

    def _launch(rung):
        block_w, num_warps = rung
        _gate_kernel[(B, HKV, -(-NW // block_w))](
            q, mu_q, mu_s, v_q, v_s, t_q, t_s, anchor,
            logm,
            q.stride(0), q.stride(1),
            mu_q.stride(0), mu_q.stride(1), mu_q.stride(2),
            v_q.stride(0), v_q.stride(1), v_q.stride(2),
            mu_s.stride(0), mu_s.stride(1), mu_s.stride(2),
            t_q.stride(0), t_q.stride(1), t_q.stride(2),
            anchor.stride(0), anchor.stride(1),
            logm.stride(0), logm.stride(1),
            NW, HQ // HKV, scaling,
            HEAD_DIM=D, WS=ws, BLOCK_WS=triton.next_power_of_2(ws),
            BLOCK_D=triton.next_power_of_2(D),
            BLOCK_W=block_w,
            MU_BITS=bits.mu, V_BITS=bits.v, T_BITS=bits.t,
            num_warps=num_warps,
        )

    # `NW` is exact in the signature, not bucketed as `Sfp` is in the decode
    # kernel: the active window count is FROZEN between evictions (design §10
    # moves the active set only there), so it takes one value per budget regime
    # and an exact key costs no extra searches while a bucketed one could serve a
    # tile chosen for a tier 40% larger.
    # The widths join it because they are constexprs: each is its own compile,
    # and a tile timed on one is not evidence about another.
    sig = (int(NW), int(HKV), int(D), int(ws), int(B), int(HQ // HKV), bits)
    if sig not in _GATE_CHOICE:
        _check_card_layout(card, bits, D, ws)
    _search_rungs(_GATE_CHOICE, _GATE_TIMINGS, _GATE_ANNOUNCED, _GATE_SEEN,
                  sig, ladder, _launch, "read gate tiling")
    # The GQA union, on each head's share of its own int2 mass. It needs every
    # window's `logm`, so it is its own launch after the tile search's final
    # launch has written them; see `_gate_share_kernel`. Not tuned: it moves
    # 2 * REP * NW floats per program, ~3% of the card kernel's traffic.
    rep = HQ // HKV
    _gate_share_kernel[(B, HKV)](
        logm, est,
        logm.stride(0), logm.stride(1), est.stride(0), est.stride(1),
        NW, rep,
        BLOCK_R=triton.next_power_of_2(rep), BLOCK_W=GATE_SHARE_BLOCK_W,
        num_warps=4,
    )
    # `est` now carries the union score, so this is a bare top-k. Sorted
    # ascending -- free to the result, not free to the memory system; the whole
    # argument is in `_sorted_pick`.
    return _sorted_pick(est.topk(n_sel, dim=-1).indices), logm


def fused_gate(
    q: Tensor,
    card,
    anchor: Tensor,
    scaling: float,
    n_sel: int,
    force_reference: bool = False,
    bits=None,
) -> Tuple[Tensor, Tensor]:
    """Dispatch: Triton on CUDA (**required**), the reference on CPU.

    ``bits`` is the card's :class:`~modules.quant.sketch.CardBits` (``None`` =
    the shipped card). It must be the widths the card was BUILT with; the
    Triton path checks every field against it.

    This used to fall back silently on CUDA, on the argument that the reference
    is not a *degraded* path -- same arithmetic, same selectivity, just without
    the fused launch. That argument is wrong about cost, and the cost is the
    entire point of the gate. Measured at the benchmarked shape, the reference is
    **62 torch ops per layer per step** against the kernels' two launches, and it
    materialises a ``[B, Nw, H_kv, rep, ws]`` intermediate the kernel never
    builds. A silent fallback therefore turns a read-traffic optimisation into a
    large decode regression, while every number downstream still looks like the
    gate ran -- exactly the "timed against a method it is not running" failure
    ``flash_decode.FusedDecodeNotReached`` exists to refuse.

    So on CUDA it raises, like ``fused_two_tier_decode``. ``force_reference``
    remains for tests that want the oracle on purpose.
    """
    bits = _as_bits(bits)                      # one isinstance on the hot path
    if force_reference or not q.is_cuda:
        return gate_reference(*((q,) + tuple(card) + (anchor, scaling, n_sel)),
                              bits=bits)
    if not _HAS_TRITON:
        raise RuntimeError(
            "the read gate requires the Triton kernel on CUDA and triton is not "
            "installed. There is no CUDA fallback: the PyTorch reference costs "
            "~62 ops per layer per step against the kernels' two launches, so "
            "falling back would silently replace the optimisation with a "
            "regression. Install triton, or set STICKYKV_FUSED_DECODE=0 to run "
            "the materialize path (which does not gate at all)."
        )
    return _gate_triton(q, card, anchor, scaling, n_sel, bits)
