"""Batched matched-state recording of the completion term, inside a real generate() call.

Experiment 1 of the completion ablation asks how close the calibrated completion
term is to the true unread-window mass, and how much attention-output error every
variant of it causes, on the SAME decode state. ``observation_collector`` answers
that for one prompt at a time (B = 1) and keeps every array the observation suite
needs; at a long generation that is tens of GB per document. This module records
only what the completion analysis reads, for every row of a batched
``model.generate`` over the windowed cache, every ``stride``-th decode step:

* **The state is the step's own.** At each attention layer, after the fused decode
  ran, it takes the step's post-RoPE query and the kernel's real output (before
  ``o_proj``), the fp tier as the cache holds it, the int2 tier dequantized
  through the slot table, the read gate's pick and the cards' estimates (the gate
  observer), and the cards' value centroids (the cache's fused hand-off).
* **Every variant on that state.** Each row goes through
  :func:`~modules.windowed_cache.completion.completion_rungs`, the function
  ``observation_collector.record_completion_rungs`` uses, so all
  :data:`~modules.windowed_cache.completion.COMPLETION_RUNGS` (off, uncal, cal, the
  five scales, the four noise levels, the three oracles) are scored against
  ``o_held``, attention over every held window. Nothing is fed back: the run's
  output is the shipped method's.
* **Reduced on the device.** Per (row, layer, query head) it keeps the rel. L2
  error of every rung and the per-head mass statistics of
  :func:`head_step_stats`. That is ~100 KB per recorded row-step instead of a
  [T, L, H, W] array per quantity.

Two checks are recorded with the data, and a reader must apply them:
``cal_vs_actual``, the recomputed calibrated output against the kernel's real
output for that row, and the ``full_oracle`` rung, which must reproduce
``o_held``. A row paired with the wrong row's tiers, or a window paired with the
wrong card, fails the first by orders of magnitude.

Errors are normalised by ``|o_held|``. The B = 1 collector normalises by the
full-KV output, which needs a shadow copy of every key; no shadow is kept here.

Rows that have finished (EOS) are not recorded: ``generate`` keeps decoding them
with pad tokens until the batch ends. :class:`RowTracker`, a stopping criterion
that never stops anything, marks a row dead as soon as it emits an EOS id.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from transformers import StoppingCriteria

from modules.windowed_cache.completion import COMPLETION_RUNGS

#: Per-head statistics, in this order on the last axis of ``stats``. The same
#: quantities the notebook's Experiment 1 analysis takes from the collector's
#: per-window arrays, except ``uncal_shift``, kept in log space (its exponential
#: overflows fp16 and is only ever summarised by a median).
STAT_NAMES = ("unread", "E_w", "E_unw", "bias_w", "bias_unw", "tot_err", "sd_U",
              "E_w_uncal", "uncal_shift", "cal_vs_actual")
#: Windows below this exact mass are left out of the unweighted statistics.
FLOOR = 1e-6
_CAL = COMPLETION_RUNGS.index("cal")


# ---------------------------------------------------------------------------
# the per-head reduction
# ---------------------------------------------------------------------------


def head_step_stats(read: Tensor, log_mstar: Tensor, logerr: Tensor, shift: Tensor,
                    cva: Tensor, floor: float = FLOOR):
    """One (row, step, layer): ``(stats [H, len(STAT_NAMES)] f32, valid [H] bool)``.

    ``read [H, n]`` the gate's pick per query head; ``log_mstar [H, n]`` each
    int2 window's exact mass over ``Z_A`` (natural log); ``logerr [H, n]``
    ``log m~_cal - log m*``; ``shift [H]`` ``log m~_uncal - log m~_cal``; ``cva
    [H]`` the calibrated output's error against the kernel's. Every window here
    is in the int2 tier. A head is valid when it read at least one window and
    left at least one finite-mass window unread; invalid heads are NaN.
    """
    lms, e = log_mstar.double(), logerr.double()
    fin = torch.isfinite(lms) & torch.isfinite(e)
    m = torch.where(fin, lms, torch.full_like(lms, float("-inf"))).exp()
    U = ~read
    Uw = U & fin
    Uf = Uw & (m >= floor)
    nU = Uf.sum(1)
    wU = torch.where(Uw, m, torch.zeros_like(m))
    sw = wU.sum(1)
    eU = torch.where(Uw, e, torch.zeros_like(e))
    eF = torch.where(Uf, e, torch.zeros_like(e))
    nan = torch.full_like(sw, float("nan"))
    nUd = nU.clamp_min(1).double()
    mean_f = eF.sum(1) / nUd
    var = torch.where(Uf, (e - mean_f.unsqueeze(1)) ** 2, torch.zeros_like(e)).sum(1) \
        / (nU - 1).clamp_min(1).double()
    sh = shift.double()
    out = torch.stack([
        sw / (1.0 + sw),                                                     # unread
        (wU * eU.abs()).sum(1) / sw,                                         # E_w
        torch.where(nU > 0, eF.abs().sum(1) / nUd, nan),                     # E_unw
        (wU * eU).sum(1) / sw,                                               # bias_w
        torch.where(nU > 0, mean_f, nan),                                    # bias_unw
        torch.log((wU * eU.exp()).sum(1)) - torch.log(sw),                   # tot_err
        torch.where(nU >= 4, var.sqrt(), nan),                               # sd_U
        (wU * torch.where(Uw, e + sh.unsqueeze(1), torch.zeros_like(e)).abs()).sum(1) / sw,
        sh,                                                                  # uncal_shift
        cva.double(),                                                        # cal_vs_actual
    ], dim=-1)
    valid = (read.sum(1) > 0) & (Uw.sum(1) > 0)
    out = torch.where(valid.unsqueeze(1) & torch.isfinite(out), out, torch.full_like(out, float("nan")))
    return out.float(), valid


# ---------------------------------------------------------------------------
# the int2 tier as held, every row
# ---------------------------------------------------------------------------


def held_int2_tier_batched(store, rope, ws: int, D: int, dtype: torch.dtype):
    """``observation_collector.held_int2_tier`` for every row of a batch.

    Returns ``(keys, vals [B, Hk, n*ws, D] f32, pos [B, n*ws], ids [B, n])`` in
    the gate's active order (ascending window id), or None when the tier is
    empty. Rows are rectangular (the cache holds equal-length rows only), so
    ``n`` is shared. Reads through the slot table: works on a per-layer store
    and on the layer-major path's per-layer view alike.
    """
    if store is None or store.num_active_windows == 0 or store.table is None:
        return None
    from modules.quant.effective import dequant_rotate_q_keys
    from modules.quant.quantizer import dequantize_value_windows
    table = store.table
    n = int(store.num_active_windows)
    idx = table.active_order(n)                                          # [B, n]
    B = idx.shape[0]
    kc, ks, kz, vc, vs, vz, pos = table.gather(idx)
    Hk = table.num_kv_heads
    pos_flat = pos.reshape(B, n * ws)
    keys = dequant_rotate_q_keys(kc, ks, kz, ws, pos_flat, rope, dtype, B, n, Hk, D)
    vals = dequantize_value_windows(vc, vs, vz, D, out_dtype=dtype)
    vals = vals.reshape(B, n, Hk, ws, D).permute(0, 2, 1, 3, 4).reshape(B, Hk, n * ws, D)
    ids = table.slot_wid.gather(1, idx)
    return keys.float(), vals.float(), pos_flat, ids.long()


# ---------------------------------------------------------------------------
# which rows are still generating
# ---------------------------------------------------------------------------


class RowTracker(StoppingCriteria):
    """Never stops anything; marks a row dead once it has emitted an EOS id.

    ``generate`` calls its stopping criteria after appending each new token and
    before the next forward, so ``alive`` read during a forward says whether
    that forward's input token is still part of the row's text. A row that ends
    on a stop string is padded with an EOS id from the next step on, and dies one
    step later; a reader also cuts records at the row's generated length.
    """

    def __init__(self, eos_ids: Iterable[int]):
        self.eos = sorted({int(e) for e in eos_ids})
        self.alive: Optional[Tensor] = None
        self._eos_t: Optional[Tensor] = None

    def reset(self, batch: int, device) -> None:
        self.alive = torch.ones(batch, dtype=torch.bool, device=device)
        self._eos_t = torch.tensor(self.eos, dtype=torch.long, device=device)

    def __call__(self, input_ids, scores, **kwargs):
        if self.alive is None or self.alive.shape[0] != input_ids.shape[0]:
            self.reset(input_ids.shape[0], input_ids.device)
        self.alive &= ~torch.isin(input_ids[:, -1], self._eos_t)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


# ---------------------------------------------------------------------------
# the recorder
# ---------------------------------------------------------------------------


class CompletionRecorder:
    """Attach to one ``model.generate`` over ``cache``; see the module docstring.

    Usage::

        rec = CompletionRecorder(model, cache, stride=16, eos_ids=eos)
        rec.attach()
        try:
            model.generate(..., stopping_criteria=StoppingCriteriaList([rec.tracker, ...]))
        finally:
            rec.detach()
        arrays = rec.result()
    """

    def __init__(self, model, cache, stride: int, eos_ids: Sequence[int],
                 floor: float = FLOOR) -> None:
        if int(stride) < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        mc = model.config
        self.model, self.cache, self.stride, self.floor = model, cache, int(stride), floor
        self.H = mc.num_attention_heads
        self.Hk = getattr(mc, "num_key_value_heads", self.H)
        self.D = getattr(mc, "head_dim", None) or mc.hidden_size // self.H
        self.L = mc.num_hidden_layers
        self.scale = self.D ** -0.5
        self.ws = int(cache.resolved.window_size)
        self.tracker = RowTracker(eos_ids)
        self.t = -1
        self.recording = False
        self._rows: Optional[List[int]] = None
        self._q: Dict[int, Tensor] = {}
        self._o: Dict[int, Tensor] = {}
        self._handles: list = []
        self._gate = None
        self._buf: Optional[Dict[str, Tensor]] = None
        self._out: Dict[str, list] = {"steps": [], "rows": [], "err": [], "stats": [], "fired": []}

    # -- wiring ------------------------------------------------------------

    def attach(self) -> None:
        from modules.evaluation.ours_parity_runner import GateRecorder
        from modules.windowed_cache import flash_decode
        from modules.windowed_cache.hooks import _get_attn_classes

        self._gate = GateRecorder()
        self._gate.bind(self.cache)
        flash_decode.set_gate_observer(self._observe)
        try:
            for m in self.model.modules():
                if not isinstance(m, _get_attn_classes()):
                    continue
                li = m.layer_idx
                self._handles.append(m.q_proj.register_forward_hook(self._q_hook(li)))
                self._handles.append(m.o_proj.register_forward_pre_hook(self._o_hook(li)))
                self._handles.append(m.register_forward_hook(self._attn_hook(li), with_kwargs=True))
        except Exception:
            self.detach()
            raise
        if not self._handles:
            self.detach()
            raise RuntimeError("no attention module found: the recorder would record nothing")

    def detach(self) -> None:
        from modules.windowed_cache import flash_decode

        flash_decode.clear_gate_observer()
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._gate is not None:
            self._gate.bind(None)
        self._flush()

    def _observe(self, layer_idx: int, sel: Tensor, logmass: Tensor) -> None:
        if self.recording:
            self._gate(layer_idx, sel, logmass)

    def _q_hook(self, li: int):
        def hook(_m, args, out):
            if li == 0:
                self._step(out.shape[1])
            if self.recording:
                self._q[li] = out.detach()
        return hook

    def _o_hook(self, li: int):
        def hook(_m, args):
            if self.recording:
                self._o[li] = args[0].detach()
        return hook

    def _attn_hook(self, li: int):
        def hook(_m, args, kwargs, _out):
            if not self.recording:
                return
            pe = kwargs.get("position_embeddings")
            if pe is None and len(args) > 1:
                pe = args[1]
            if pe is None:
                raise RuntimeError(f"layer {li}: no position_embeddings in the attention call; "
                                   "the recorder cannot rebuild the post-RoPE query")
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            q_raw, o_raw = self._q.pop(li), self._o.pop(li)
            B = q_raw.shape[0]
            q = q_raw.view(B, 1, self.H, self.D).transpose(1, 2)
            cos, sin = pe
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            self.record_layer(li, q[:, :, 0], o_raw.view(B, self.H, self.D))
            for d in (self._gate.picks, self._gate.logmass, self._gate.active_ids):
                d.pop(li, None)
        return hook

    # -- steps -------------------------------------------------------------

    def _step(self, tq: int) -> None:
        """Layer 0's q_proj: a new forward. Prefill is t = 0; decode steps count from 1."""
        self._flush()
        self.t = 0 if tq > 1 else self.t + 1
        self._q.clear()
        self._o.clear()
        self.recording = False
        if tq != 1 or self.t < 1 or self.t % self.stride:
            return
        alive = self.tracker.alive
        rows = list(range(self._batch())) if alive is None else \
            torch.nonzero(alive).flatten().tolist()
        if rows:
            self._begin(rows, next(self.model.parameters()).device)

    def _begin(self, rows: List[int], dev) -> None:
        """Open a recorded step for ``rows`` (``self.t`` already set)."""
        self._gate.clear()
        self.recording = True
        self._rows = rows
        B, K, S = self._batch(), len(COMPLETION_RUNGS), len(STAT_NAMES)
        nanf = float("nan")
        self._buf = {
            "err": torch.full((B, self.L, self.H, K), nanf, device=dev),
            "stats": torch.full((B, self.L, self.H, S), nanf, device=dev),
            "fired": torch.zeros(B, self.L, dtype=torch.bool, device=dev),
        }

    def _batch(self) -> int:
        st = self.cache._states[0]
        return int(st.key_states.shape[0])

    def _flush(self) -> None:
        if self._buf is None:
            return
        buf, rows, t = self._buf, self._rows, self.t
        self._buf = None
        err, stats, fired = buf["err"].cpu().numpy(), buf["stats"].cpu().numpy(), buf["fired"].cpu().numpy()
        for r in rows:
            self._out["steps"].append(t)
            self._out["rows"].append(r)
            self._out["err"].append(err[r])
            self._out["stats"].append(stats[r])
            self._out["fired"].append(fired[r])

    # -- one layer ---------------------------------------------------------

    def record_layer(self, layer: int, q: Tensor, o_actual: Tensor) -> None:
        """Every rung, for every live row, at this layer. ``q``/``o_actual`` ``[B, H, D]``."""
        from modules.evaluation.observation_collector import _rel, attend
        from modules.quant.sketch import _dq_symb
        from modules.windowed_cache.completion import completion_rungs

        gate, cache = self._gate, self.cache
        if layer not in gate.picks:
            return                                            # no int2 tier gated at this layer yet
        st = cache._states[layer]
        tier = held_int2_tier_batched(cache._stores[layer], cache.rope_module, self.ws, self.D,
                                      st.key_states.dtype)
        if tier is None:
            return
        qK, qV, _, q_ids = tier
        act = gate.active_ids[layer].long()                   # [B, n]
        if act.shape != q_ids.shape or not torch.equal(act, q_ids.to(act.device)):
            raise RuntimeError(
                f"layer {layer} step {self.t}: the gate's active windows are not the held int2 tier "
                "in the same order; the rungs would pair cards with the wrong windows.")
        pick = gate.picks[layer].long()                       # [B, Hk, n_sel] window ids
        read_kv = (pick.unsqueeze(-1) == act[:, None, None, :]).any(2)    # [B, Hk, n]
        fctx = getattr(cache, "_fused_ctx", None)
        slot = fctx[layer] if fctx is not None and layer < len(fctx) else None
        g = slot.get("gate") if isinstance(slot, dict) else None
        if g is None:
            raise RuntimeError(f"layer {layer} step {self.t}: no gate context on the cache; "
                               "the recorder needs the fused decode path.")
        vm_q, vm_s, v_anc = g["centroid"]
        rep = self.H // self.Hk
        buf = self._buf
        for b in self._rows:
            nu = _dq_symb(vm_q[b], vm_s[b], g["bits"].vm)                      # [n, Hk, D]
            va = v_anc[b] if v_anc.dim() == 3 else v_anc
            nu = (nu + va.float().unsqueeze(0)).permute(1, 0, 2)              # [Hk, n, D]
            fpK, fpV = st.key_states[b].float(), st.value_states[b].float()
            o_held, _ = attend(q[b], torch.cat([fpK, qK[b]], 1), torch.cat([fpV, qV[b]], 1),
                               self.scale, rep)
            r = completion_rungs(q[b], fpK, fpV, qK[b], qV[b], self.ws, read_kv[b],
                                 gate.logmass[layer][b].float(), nu, self.scale, act[b], layer)
            outs = r["outs"]                                                   # [K, H, D]
            err = _rel(outs, o_held.unsqueeze(0), o_held.unsqueeze(0)).T       # [H, K]
            cva = _rel(outs[_CAL], o_actual[b].float(), o_held)
            stats, valid = head_step_stats(r["read"], r["log_mstar"], r["logerr_cal"],
                                           r["uncal_shift"], cva, self.floor)
            buf["err"][b, layer] = torch.where(valid.unsqueeze(1), err, torch.full_like(err, float("nan")))
            buf["stats"][b, layer] = stats
            buf["fired"][b, layer] = True

    # -- output ------------------------------------------------------------

    def result(self) -> Dict[str, np.ndarray]:
        """``steps [R] i32``, ``rows [R] i16``, ``err [R, L, H, K] f32``, ``stats [R, L, H, S] f32``,
        ``fired [R, L] bool``: one record per (recorded step, live row)."""
        self._flush()
        o = self._out
        K, S = len(COMPLETION_RUNGS), len(STAT_NAMES)
        if not o["steps"]:
            return {"steps": np.zeros(0, np.int32), "rows": np.zeros(0, np.int16),
                    "err": np.zeros((0, self.L, self.H, K), np.float32),
                    "stats": np.zeros((0, self.L, self.H, S), np.float32),
                    "fired": np.zeros((0, self.L), bool)}
        return {"steps": np.asarray(o["steps"], np.int32), "rows": np.asarray(o["rows"], np.int16),
                "err": np.stack(o["err"]).astype(np.float32),
                "stats": np.stack(o["stats"]).astype(np.float32),
                "fired": np.stack(o["fired"]).astype(bool)}
