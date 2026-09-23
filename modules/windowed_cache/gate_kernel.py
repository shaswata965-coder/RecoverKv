"""Triton gate: score every int2 window from its card, pick the top ``n_sel``.

What it replaces
----------------
Without the gate the decode step dequantizes every active int2 window to find
out which ones mattered (design §8). This kernel reads each window's card --
``mu``/``v`` int8 plus two tiny scalars, 400 B per head against a 6432 B window
-- and emits the compacted slot list the Q-tier loop should actually visit, plus
the log-mass estimate that every **skipped** window contributes to
``window_scores``.

Shape of the work
-----------------
One program owns one ``(batch, KV head)``, matching ``decode_kernel``'s grid, so
the GQA group's union is free: the program reduces over its own ``rep`` query
heads before selecting, which is what makes the cap bind on the KV head -- the
real unit of work, since one program loads a window once for the whole group.
Capping per query head and unioning afterwards would let a 0.25 ratio read up to
``0.25 * rep`` of the tier (100% at Llama-3.1-8B's rep=4).

Two dot products per window give both outputs::

    m = q.anchor + q.mu        g = q.v
    x_i   = scale * (m + t_i * g)                  per-token estimate
    bound = max_i (x_i + scale * ||q|| * eps_i)    upper bound, cannot miss
    est   = max_i x_i                              slack-free, what the cap ranks on
    logmass = logsumexp_i x_i                      what a skipped window scores

``bound`` and ``est`` are different quantities with different jobs; see
:func:`modules.quant.sketch.select_windows` for why the cap ranks on the second.

The union, in each head's own units (``head_norm``)
---------------------------------------------------
The group's union is a max over its ``rep`` query heads, and a max is only a
union if the heads are measured in the same units. Raw logits are not: each head
carries its own constant ``q . k_common``, which its softmax discards and a cross-
head max does not, so the head with the largest offset picks the whole group's
windows (:func:`modules.quant.sketch.in_head_units` has the mechanism and the
per-query-head recall). With ``head_norm`` each head's ``est`` is taken relative
to its own Q-tier log-partition first. That needs a reduction over the WHOLE
window axis, which is split across programs, so it is two launches: the scan
writes per-head ``est`` and a per-tile ``(max, sum)`` of ``logmass``; a small
union kernel merges the partials into each head's partition and takes the group
max. Without ``head_norm`` the scan does exactly the arithmetic it always did
(``HEAD_NORM`` is a constexpr; the partials' pointers ride along unused) and
there is no second launch, so a raw-union selection is unchanged bit for bit.

A separate kernel, for now
--------------------------
This could be a prologue inside ``_two_tier_decode_kernel`` -- one program
already owns the right ``(batch, KV head)`` -- and that is where it belongs
eventually, because the decode path is launch-bound and this costs one extra
launch per layer per step. It is separate first because a prologue competes for
the main kernel's register budget: its Q-tier loop already stages ten
``[D/2, BLOCK_T]`` fp32 tiles and the autotuner falls back silently when that
budget is exceeded, so a rung dropped by the prologue would cost more than the
gate saves and would produce no error. Land it standalone, measure the launch
cost against the traffic saved, then fold it in with the rung pinned.

Backend contract (mirrors ``score_kernel`` / ``decode_kernel``)
--------------------------------------------------------------
* :func:`gate_reference` -- pure-PyTorch oracle. Defines correctness, runs on
  CPU, and is the Triton kernel's test oracle.
* ``_gate_kernel`` + :func:`_gate_triton` -- the GPU kernel. **GPU-only, ships
  unvalidated by construction** (this repo's dev box is CPU-only), like every
  other Triton kernel here.
* :func:`fused_gate` -- dispatcher: Triton on CUDA, reference otherwise.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

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
    head_norm: bool = False,
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
    head_norm : take the group union in each query head's own units
        (:func:`modules.quant.sketch.in_head_units`) instead of over raw logits.
        At ``rep == 1`` that is a constant shift of one head's ranking, so it is
        skipped there -- by the kernel dispatcher too, so the two agree exactly.

    Returns
    -------
    sel : ``[B, H_kv, n_sel]`` int32 column indices, **ascending**. See
        :func:`_sorted_pick` for why the order is free to the result and not
        free to the memory system.
    logmass : ``[B, H_q, Nw]`` the estimate for every window.
    """
    from modules.quant.sketch import Sketch, gate_and_score, group_max, in_head_units

    card = Sketch(mu_q, mu_s, v_q, v_s, t_q, t_s, vm_q, vm_s)
    logmass, est = gate_and_score(q, card, anchor, scaling)
    hkv = mu_q.shape[2]
    if head_norm and q.shape[1] > hkv:
        est = in_head_units(est, logmass)
    top = group_max(est, hkv).topk(n_sel, dim=-1).indices          # [B,Hkv,n_sel]
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
#: -- the ``logsumexp`` over ``WS`` and the ``max`` over ``REP`` -- lives entirely
#: inside one window, so which program owns which window changes nothing about
#: the arithmetic. ``num_warps`` is NOT: it changes the lane layout of the
#: ``tl.sum`` over ``HEAD_DIM``, so ``est`` and ``logmass`` move in their last
#: bits, and ``est`` feeds a top-k. A last-bit move can therefore swap the k-th
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


if _HAS_TRITON:  # pragma: no cover - GPU-only

    @triton.jit
    def _gate_kernel(
        Q, MU, MUS, V, VS, T, TS, ANCH,
        EST, LOGM, PMX, PSM,
        qb, qh, pmb, pmn, pmh, mub, mun, muh, sb, sn, sh, tb, tn, th, ab, ah,
        lb, lh, eb, eh, xb, xh,
        NW, REP, SCALE,
        HEAD_DIM: tl.constexpr, WS: tl.constexpr, BLOCK_WS: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_W: tl.constexpr, HEAD_NORM: tl.constexpr,
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

        **``est`` leaves as the GQA group's max**, already reduced over ``r``, so
        the output is ``[B, H_kv, NW]`` rather than ``[B, H_q, NW]`` — a quarter
        of the write traffic, and it saves the host a reshape and an ``amax``
        before the top-k. ``logmass`` stays per query head because every query
        head's skipped windows need their own estimate.

        ``BLOCK_WS`` is ``WS`` rounded up to a power of two, because ``tl.arange``
        admits nothing else. The surplus lanes are masked out of the ``t`` load
        **and** forced to ``-inf`` in ``x``: a padding lane that merely loaded
        zero would still carry the real ``SCALE * m`` into the max and the
        logsumexp, so a ``ws`` of 12 would score every window as if it held a
        13th token sitting exactly at its mean.

        No ``bound`` is computed. The margin rule that needed it has no caller on
        this path (``WindowedCache._gate_ctx`` refuses a finite margin), so the
        bound cost a ``[B, H_q, NW]`` fp32 store and the whole ``eps`` field of
        every card, to be discarded by :func:`_gate_triton`. The cap ranks on the
        estimate; see :func:`modules.quant.sketch.select_windows`.

        **Under ``HEAD_NORM`` ``est`` leaves per query head instead**, ``[B, H_q,
        NW]``, with this tile's ``(max, sum exp(logmass - max))`` per head in
        ``PMX``/``PSM`` at column ``program_id(2)``. The group max cannot be taken
        here, because putting the heads in common units needs each head's
        partition over the whole window axis, which no one program sees;
        :func:`_gate_union_kernel` finishes it. The cards are still read once.
        """
        b = tl.program_id(0)
        kv = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        dm = d < HEAD_DIM
        w = tl.arange(0, BLOCK_WS)
        wm = w < WS

        cols = tl.program_id(2) * BLOCK_W + tl.arange(0, BLOCK_W)
        cm = cols < NW
        # mu is int4 packed two per byte along D: element `d` is the nibble at
        # byte `d // 2`, shift `4 * (d % 2)`, biased by +8 so it is unsigned.
        # Same idiom as the decode kernel's int2 crumbs, one width up.
        dbyte = d // 2
        dshift = (4 * (d % 2)).to(tl.uint8)

        # ---- the cards for this tile: read ONCE for the whole query group ----
        mu_b = tl.load(MU + b * pmb + cols[:, None] * pmn + kv * pmh + dbyte[None, :],
                       mask=cm[:, None] & dm[None, :], other=0)
        mu = (((mu_b >> dshift[None, :]) & 0xF).to(tl.float32) - 8.0)
        mus = tl.load(MUS + b * sb + cols * sn + kv * sh,
                      mask=cm, other=0.0).to(tl.float32)
        vv = tl.load(V + b * mub + cols[:, None] * mun + kv * muh + d[None, :],
                     mask=cm[:, None] & dm[None, :], other=0).to(tl.float32)
        vs = tl.load(VS + b * sb + cols * sn + kv * sh,
                     mask=cm, other=0.0).to(tl.float32)
        tt = tl.load(T + b * tb + cols[:, None] * tn + kv * th + w[None, :],
                     mask=cm[:, None] & wm[None, :], other=0).to(tl.float32)
        ts = tl.load(TS + b * sb + cols * sn + kv * sh,
                     mask=cm, other=0.0).to(tl.float32)

        # The anchor is indexed by KV head alone, so it is constant in r too.
        anc = tl.load(ANCH + b * ab + kv * ah + d, mask=dm, other=0.0).to(tl.float32)

        est_g = tl.full([BLOCK_W], -float("inf"), tl.float32)
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
            if HEAD_NORM:
                tl.store(EST + b * eb + hq * eh + cols,
                         tl.where(cm, xm, -float("inf")), mask=cm)
                # Every tile holds at least one real column (the grid is a
                # ceil-div), so `pmx` is finite and the masked lanes' exp is 0.
                lv = tl.where(cm, lm, -float("inf"))
                pmx = tl.max(lv, axis=0)
                tl.store(PMX + b * xb + hq * xh + tl.program_id(2), pmx)
                tl.store(PSM + b * xb + hq * xh + tl.program_id(2),
                         tl.sum(tl.exp(lv - pmx), axis=0))
            else:
                est_g = tl.maximum(est_g, xm)

        if HEAD_NORM:
            pass                     # the union kernel writes the group's EST
        else:
            tl.store(EST + b * eb + kv * eh + cols,
                     tl.where(cm, est_g, -float("inf")), mask=cm)

    @triton.jit
    def _gate_union_kernel(
        ESTH, PMX, PSM, EST,
        hb, hh, xb, xh, eb, eh,
        NW, NT, REP,
        BLOCK_R: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_U: tl.constexpr,
    ):
        """The GQA union in each query head's own units. One program per
        ``(row, KV head, column tile)``.

        1. Merge the scan's ``NT`` per-tile ``(max, sum)`` partials into each of
           the group's heads' log-partition over the whole Q tier -- the online
           softmax merge, one tile of partials at a time.
        2. ``EST[kv, w] = max_r (est_r[w] - lse_r)``: the largest share of its
           OWN head's Q-tier mass that window ``w``'s hot token carries.

        Every program redoes step 1 for its ``rep`` heads. That is ``rep * NT``
        values (``NT = NW / BLOCK_W`` of the scan: 10 to 39 at 624 windows) against the
        ``rep * BLOCK_U`` it then reads for step 2, so sharing it would cost a
        third launch to save less than it reads anyway.

        Padded head lanes (``r >= REP``) keep ``m = -inf, s = 0`` without ever
        forming ``-inf - -inf``: the merge exponentiates against ``m_ref``, which
        is 0 where the running max is still ``-inf``, and their ``lse`` is pinned
        to a finite 0. They are removed from the max by the ``em`` select, never
        by arithmetic. (Run under ``TRITON_INTERPRET=1`` this is checked: numpy
        warns on any ``log(0)`` or ``inf - inf``, even in a lane selected away.)
        """
        b = tl.program_id(0)
        kv = tl.program_id(1)
        r = tl.arange(0, BLOCK_R)
        rm = r < REP
        hq = kv * REP + r

        m = tl.full([BLOCK_R], -float("inf"), tl.float32)
        s = tl.zeros([BLOCK_R], tl.float32)
        for t0 in range(0, NT, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)
            tm = rm[:, None] & (t < NT)[None, :]
            pm = tl.load(PMX + b * xb + hq[:, None] * xh + t[None, :],
                         mask=tm, other=-float("inf"))
            ps = tl.load(PSM + b * xb + hq[:, None] * xh + t[None, :],
                         mask=tm, other=0.0)
            m_new = tl.maximum(m, tl.max(pm, axis=1))
            m_ref = tl.where(m_new == -float("inf"), 0.0, m_new)
            s = s * tl.exp(m - m_ref) + tl.sum(
                tl.where(tm, ps * tl.exp(pm - m_ref[:, None]), 0.0), axis=1)
            m = m_new
        # A padded lane has s == 0; give it a finite 0 rather than log(0), so no
        # lane ever forms -inf - -inf below (the select would drop it, but a NaN
        # that is only ever selected away is one refactor from being read).
        lse = tl.where(rm, m, 0.0) + tl.log(tl.where(rm, s, 1.0))

        cols = tl.program_id(2) * BLOCK_U + tl.arange(0, BLOCK_U)
        cm = cols < NW
        em = rm[:, None] & cm[None, :]
        e = tl.load(ESTH + b * hb + hq[:, None] * hh + cols[None, :],
                    mask=em, other=-float("inf"))
        crit = tl.max(tl.where(em, e - lse[:, None], -float("inf")), axis=0)
        tl.store(EST + b * eb + kv * eh + cols, crit, mask=cm)


def _sm_count(device) -> int:  # pragma: no cover - GPU-only
    try:
        return int(torch.cuda.get_device_properties(device).multi_processor_count)
    except Exception:
        return 64


#: Column tile and partials tile of :func:`_gate_union_kernel`. Not searched: the
#: kernel reads ``rep * NW`` floats that the scan has just written (L2-resident),
#: a few percent of the scan's card traffic, and its tiling cannot move a result
#: bit -- every reduction in it is a max, or a merge whose order is fixed by the
#: loop over ``NT`` rather than by the grid.
_UNION_BLOCK_U = 256
_UNION_BLOCK_T = 64


def _gate_triton(q, card, anchor, scaling, n_sel,
                 head_norm=False):  # pragma: no cover - GPU-only
    if not _HAS_TRITON:
        raise RuntimeError("fused_gate requires triton")
    mu_q, mu_s, v_q, v_s, t_q, t_s, _vm_q, _vm_s = card
    # mu_q's last axis is D//2 (int4, packed); the head dim comes from `v_q`,
    # which is still full-width int8. Reading it off mu_q would halve every
    # downstream extent.
    B, NW, HKV = mu_q.shape[0], mu_q.shape[1], mu_q.shape[2]
    D = v_q.shape[-1]
    HQ = q.shape[1]
    REP = HQ // HKV
    ws = t_q.shape[-1]
    # A group of one has nothing to put in common units: the shift would move
    # its one head's ranking by a constant. Skipped, exactly as `gate_reference`
    # skips it, so rep == 1 is the raw kernel with no second launch.
    head_norm = bool(head_norm) and REP > 1
    if anchor.dim() == 2:
        anchor = anchor.unsqueeze(0).expand(B, HKV, D).contiguous()
    # Both are dead by the end of `_run_fused`: `est` is consumed by the topk
    # below, `logm` by the decode kernel's §5 fill in the same call. Reused
    # across layers within a step -- see `decode_kernel._scratch` (W4/D5).
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

    if head_norm:
        # The scan's `est` goes out per query head; the union kernel folds it
        # into `est`. Partials are sized for the narrowest rung, so every rung
        # the search tries writes into the same buffers.
        est_h = _scratch("est_h", (B, HQ, NW), torch.float32, q.device)
        nt_max = -(-NW // min(bw for bw, _ in ladder))
        pmx = _scratch("gate_pmx", (B, HQ, nt_max), torch.float32, q.device)
        psm = _scratch("gate_psm", (B, HQ, nt_max), torch.float32, q.device)
        scan_est, xb, xh = est_h, pmx.stride(0), pmx.stride(1)
    else:
        # Never dereferenced: HEAD_NORM is a constexpr and compiles them out.
        scan_est, pmx, psm, xb, xh = est, est, est, 0, 0

    def _launch(rung):
        block_w, num_warps = rung
        n_tiles = -(-NW // block_w)
        _gate_kernel[(B, HKV, n_tiles)](
            q, mu_q, mu_s, v_q, v_s, t_q, t_s, anchor,
            scan_est, logm, pmx, psm,
            q.stride(0), q.stride(1),
            mu_q.stride(0), mu_q.stride(1), mu_q.stride(2),
            v_q.stride(0), v_q.stride(1), v_q.stride(2),
            mu_s.stride(0), mu_s.stride(1), mu_s.stride(2),
            t_q.stride(0), t_q.stride(1), t_q.stride(2),
            anchor.stride(0), anchor.stride(1),
            logm.stride(0), logm.stride(1),
            scan_est.stride(0), scan_est.stride(1),
            xb, xh,
            NW, REP, scaling,
            HEAD_DIM=D, WS=ws, BLOCK_WS=triton.next_power_of_2(ws),
            BLOCK_D=triton.next_power_of_2(D),
            BLOCK_W=block_w, HEAD_NORM=head_norm,
            num_warps=num_warps,
        )
        if head_norm:
            # Inside the rung, not after the search: the partials are laid out by
            # THIS rung's tile count, and the search re-launches its winner last,
            # so the `est` the top-k reads always comes from one consistent pair.
            _gate_union_kernel[(B, HKV, -(-NW // _UNION_BLOCK_U))](
                est_h, pmx, psm, est,
                est_h.stride(0), est_h.stride(1), xb, xh,
                est.stride(0), est.stride(1),
                NW, n_tiles, REP,
                BLOCK_R=triton.next_power_of_2(REP), BLOCK_T=_UNION_BLOCK_T,
                BLOCK_U=_UNION_BLOCK_U,
                num_warps=4,
            )

    # `NW` is exact in the signature, not bucketed as `Sfp` is in the decode
    # kernel: the active window count is FROZEN between evictions (design §10
    # moves the active set only there), so it takes one value per budget regime
    # and an exact key costs no extra searches while a bucketed one could serve a
    # tile chosen for a tier 40% larger. `head_norm` is a different compile (and
    # a second launch), so it tunes separately.
    sig = (int(NW), int(HKV), int(D), int(ws), int(B), int(REP), head_norm)
    _search_rungs(_GATE_CHOICE, _GATE_TIMINGS, _GATE_ANNOUNCED, _GATE_SEEN,
                  sig, ladder, _launch, "read gate tiling")
    # `est` already carries the group max, so this is a bare top-k. Sorted
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
    head_norm: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Dispatch: Triton on CUDA (**required**), the reference on CPU.

    This used to fall back silently on CUDA, on the argument that the reference
    is not a *degraded* path -- same arithmetic, same selectivity, just without
    the fused launch. That argument is wrong about cost, and the cost is the
    entire point of the gate. Measured at the benchmarked shape, the reference is
    **62 torch ops per layer per step** against the kernel's one launch, and it
    materialises a ``[B, Nw, H_kv, rep, ws]`` intermediate the kernel never
    builds. A silent fallback therefore turns a read-traffic optimisation into a
    large decode regression, while every number downstream still looks like the
    gate ran -- exactly the "timed against a method it is not running" failure
    ``flash_decode.FusedDecodeNotReached`` exists to refuse.

    So on CUDA it raises, like ``fused_two_tier_decode``. ``force_reference``
    remains for tests that want the oracle on purpose.

    ``head_norm`` takes the GQA union in each query head's own units; see the
    module docstring. Both backends honour it identically, including skipping
    it at ``rep == 1``.
    """
    if force_reference or not q.is_cuda:
        return gate_reference(*((q,) + tuple(card) + (anchor, scaling, n_sel)),
                              head_norm=head_norm)
    if not _HAS_TRITON:
        raise RuntimeError(
            "the read gate requires the Triton kernel on CUDA and triton is not "
            "installed. There is no CUDA fallback: the PyTorch reference costs "
            "~62 ops per layer per step against the kernel's one launch, so "
            "falling back would silently replace the optimisation with a "
            "regression. Install triton, or set STICKYKV_FUSED_DECODE=0 to run "
            "the materialize path (which does not gate at all)."
        )
    return _gate_triton(q, card, anchor, scaling, n_sel, head_norm=head_norm)
