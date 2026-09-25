"""MiKV — a mixed-precision KV cache that evicts nothing (Yang et al., 2024).

Every token the model has seen stays in the cache. H2O's importance policy
decides only *how precisely* each token is kept:

=================  =========================  ==============================
tier               holds                      stored at
=================  =========================  ==============================
importance cache   the ``n_recent`` newest     ``hi_bits`` — the model dtype
                   tokens + the ``n_heavy``    at 16, else per-token INT8/INT4
                   heaviest by accumulated     (Table 3)
                   attention
retained cache     every other token           ``lo_bits`` per-token INT2-4
                                               (Tables 1-2)
=================  =========================  ==============================

Selection is per (row, KV head), as in H2O, so both tiers keep the same token
*count* in every head and stay rectangular. Where H2O would evict, MiKV demotes:
the victim is re-quantized at ``lo_bits`` and moved to the retained cache, and
it keeps contributing to attention for the rest of the sequence. A demoted token
is never promoted back — its full-precision value is gone, so there would be
nothing to promote.

``lo_bits = 0`` drops the demoted token instead: that is H2O, the paper's
eviction baseline, run through the same scores and the same selection, so the
two arms differ in exactly the one thing the paper is about. It returns a
compacted K/V (surviving keys keep their original RoPE positions, as H2O's
do), which HF's position-indexed padding masks cannot address, so it refuses
padded batches.

Keys are stored outlier-balanced (Eqs. 2-4): ``I(k * b)`` with ``b`` computed
once per (row, KV head, channel) from the prompt's maxima. Dequantization
divides by ``b`` again, which is the same logit as dividing the query — see
:mod:`modules.mikv.quant`. Tokens stored at 16 bits are stored exactly; the
balancer exists to share a quantizer's burden, and they have none.

One decode step, per layer, all inside :meth:`MiKVCache.update`:

1. **Read.** Dequantize both tiers and scatter them, with the new token, into a
   dense ``[B, H, S, D]`` K and V in *position order*, which is what the
   model's own attention (eager / SDPA / flash) then consumes. Position order
   keeps HF's padding and causal masks valid without any permutation.
2. **Score.** ``softmax(q K^T / sqrt(D))`` for the new query rows, summed over
   the query heads sharing each KV head, is added to the accumulated H2O
   score of every importance-cache token. The query comes from
   :func:`modules.mikv.hooks.install_mikv_hooks`, which captures the model's
   own ``q_proj`` output; RoPE is re-applied here from the ``cos`` / ``sin``
   HF passes to ``update``. The scores are therefore the attention the model
   is about to compute, not an approximation of it.
3. **Demote.** If the importance cache is over its size, its lowest-scoring
   tokens outside the recent window move to the retained cache. As in H2O,
   this happens after the step's scores are known, so the current step still
   reads the victim at full precision.

The prefill is exact: ``update`` returns the prompt's own K/V untouched, and
tiers are formed from it afterwards (scores over every prompt row, then the
balancer, then selection and quantization). No prompt token is quantized
twice.

**What this is not.** It is the reference implementation, in PyTorch. Step 1
materializes one layer's dense fp16 K/V per step — a transient, never stored —
so decode latency here measures dequantize-then-attend, not a fused kernel.
The paper's §3.4 sketches kernels; it does not ship any, and neither does this
module. Memory (what the cache *holds*) is real: codes are bit-packed and
:meth:`MiKVCache.memory_report` counts every tensor.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

try:
    from transformers import Cache as _HFCacheBase
except ImportError:  # pragma: no cover - transformers is a hard dependency of runs
    _HFCacheBase = object  # type: ignore[assignment,misc]

from .config import EVICT, MiKVBudget, MiKVConfig
from .quant import QTensor, channel_balancer, dequantize, quantize


class MiKVHooksMissing(RuntimeError):
    """``update`` ran for a layer whose query was never captured."""


# ---------------------------------------------------------------------------
# a per-layer token store at one precision
# ---------------------------------------------------------------------------


class _TokenStore:
    """Tokens of one tier of one layer, ``[B, H, n, ...]`` along axis 2.

    At 16 bits the parts are ``k`` / ``v`` in the model dtype. Below, each of K
    and V is ``codes`` / ``alpha`` / ``beta`` (see :class:`QTensor`), and keys
    are balanced before quantization when a balancer is set.

    ``grow=True`` keeps a capacity ahead of the length and appends in place
    (the retained cache, which gains a token every step). ``grow=False``
    rebuilds exactly-sized tensors (the importance cache, which is rebuilt by
    selection every step anyway).
    """

    def __init__(self, bits: int, group_size: int, head_dim: int,
                 dtype: torch.dtype, pos_dtype: torch.dtype, grow: bool) -> None:
        self.bits = bits
        self.group_size = group_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.pos_dtype = pos_dtype
        self.grow = grow
        self.balancer: Optional[Tensor] = None  # [B, H, D] fp32, keys only
        self.parts: Dict[str, Tensor] = {}
        self.pos: Optional[Tensor] = None       # [B, H, cap]
        self.length = 0

    # -- encoding ---------------------------------------------------------

    def encode(self, k: Tensor, v: Tensor) -> Dict[str, Tensor]:
        """``[B, H, m, D]`` K/V (original space) -> this tier's parts."""
        if self.bits == 16:
            return {"k": k.to(self.dtype), "v": v.to(self.dtype)}
        if self.balancer is not None:
            k = k.float() * self.balancer.unsqueeze(2)
        qk = quantize(k, self.bits, self.group_size)
        qv = quantize(v, self.bits, self.group_size)
        return {"k_codes": qk.codes, "k_alpha": qk.alpha, "k_beta": qk.beta,
                "v_codes": qv.codes, "v_alpha": qv.alpha, "v_beta": qv.beta}

    def decode(self, parts: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Parts -> ``[B, H, m, D]`` K/V in the original space, model dtype."""
        if self.bits == 16:
            return parts["k"], parts["v"]
        k = dequantize(QTensor(parts["k_codes"], parts["k_alpha"], parts["k_beta"],
                               self.bits, self.group_size, self.head_dim))
        v = dequantize(QTensor(parts["v_codes"], parts["v_alpha"], parts["v_beta"],
                               self.bits, self.group_size, self.head_dim))
        if self.balancer is not None:
            k = k / self.balancer.unsqueeze(2)
        return k.to(self.dtype), v.to(self.dtype)

    # -- views ------------------------------------------------------------

    def live_parts(self) -> Dict[str, Tensor]:
        n = self.length
        return {name: t[:, :, :n] for name, t in self.parts.items()}

    def live_pos(self) -> Tensor:
        assert self.pos is not None
        return self.pos[:, :, : self.length]

    def read(self) -> Tuple[Tensor, Tensor]:
        return self.decode(self.live_parts())

    # -- mutation ---------------------------------------------------------

    def set(self, parts: Dict[str, Tensor], pos: Tensor, reserve: int = 0) -> None:
        """Replace the contents; keep ``reserve`` spare slots if growable."""
        n = pos.shape[-1]
        if self.grow and reserve > 0:
            self.parts = {}
            for name, t in parts.items():
                buf = t.new_empty(*t.shape[:2], n + reserve, *t.shape[3:])
                buf[:, :, :n] = t
                self.parts[name] = buf
            self.pos = torch.empty(*pos.shape[:2], n + reserve,
                                   dtype=self.pos_dtype, device=pos.device)
            self.pos[:, :, :n] = pos
        else:
            self.parts = {name: t.contiguous() for name, t in parts.items()}
            self.pos = pos.to(self.pos_dtype).contiguous()
        self.length = n

    def append(self, parts: Dict[str, Tensor], pos: Tensor) -> None:
        m = pos.shape[-1]
        if m == 0:
            return
        if self.pos is None:
            self.set(parts, pos, reserve=0)
            return
        cap = self.pos.shape[-1]
        need = self.length + m
        if need > cap:
            new_cap = max(need, int(cap * 1.5) + 16) if self.grow else need
            for name, t in self.parts.items():
                buf = t.new_empty(*t.shape[:2], new_cap, *t.shape[3:])
                buf[:, :, : self.length] = t[:, :, : self.length]
                self.parts[name] = buf
            pbuf = self.pos.new_empty(*self.pos.shape[:2], new_cap)
            pbuf[:, :, : self.length] = self.pos[:, :, : self.length]
            self.pos = pbuf
        for name, t in parts.items():
            self.parts[name][:, :, self.length:need] = t
        self.pos[:, :, self.length:need] = pos.to(self.pos_dtype)
        self.length = need

    # -- accounting -------------------------------------------------------

    def kv_bytes(self, live: bool = True) -> int:
        total = 0
        for t in self.parts.values():
            n = self.length if live else t.shape[2]
            per_slot = t.numel() // max(t.shape[2], 1) * t.element_size()
            total += per_slot * n
        return total

    def pos_bytes(self, live: bool = True) -> int:
        if self.pos is None:
            return 0
        n = self.length if live else self.pos.shape[-1]
        return self.pos.shape[0] * self.pos.shape[1] * n * self.pos.element_size()


# ---------------------------------------------------------------------------
# per-layer state
# ---------------------------------------------------------------------------


class _Layer:
    def __init__(self) -> None:
        self.length = 0                     # tokens seen (== held, unless evicting)
        self.hi: Optional[_TokenStore] = None
        self.lo: Optional[_TokenStore] = None   # None under lo_bits=0 (H2O)
        self.dropped = 0                    # tokens evicted, lo_bits=0 only
        self.hi_score: Optional[Tensor] = None  # [B, H, n_hi] fp32, H2O score
        self.balancer: Optional[Tensor] = None  # [B, H, D] fp32


# ---------------------------------------------------------------------------
# RoPE and the score pass
# ---------------------------------------------------------------------------


def _rotate_half(x: Tensor) -> Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(q: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Llama's ``apply_rotary_pos_emb`` for the query alone.

    ``q`` is ``[B, H, L, D]``; ``cos`` / ``sin`` are what HF hands ``update``,
    ``[B, L, D]`` (or ``[L, D]``). Same arithmetic, same dtype, as
    ``transformers.models.llama.modeling_llama.apply_rotary_pos_emb``, so the
    query scored here is the query the model attends with.
    """
    if cos.dim() == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin)


def key_validity(mask: Optional[Tensor], batch: int, length: int,
                 device: torch.device) -> Optional[Tensor]:
    """``[B, S]`` bool — which key positions are real tokens, or ``None`` = all.

    Reads the mask the attention module was called with: the 2-D ``[B, S]``
    0/1 mask flash-attention gets, or the 4-D additive ``[B, 1, L, S]`` mask
    eager / SDPA get (whose LAST query row sees every real key). Only padding
    is taken from it — causality is imposed by the score pass itself, so SDPA's
    un-masking of fully-padded rows cannot leak into the scores.

    Never asks the device whether the mask is all-ones: that is a sync per
    layer per step, and an all-True mask is merely a no-op below.
    """
    if mask is None:
        return None
    if mask.dim() == 2:
        valid = mask[:, :length].bool()
    elif mask.dim() == 4:
        row = mask[:, 0, -1, :length]
        if row.dtype == torch.bool:
            valid = row
        else:
            valid = row > torch.finfo(row.dtype).min / 2
    else:
        raise ValueError(f"unexpected attention_mask shape {tuple(mask.shape)}")
    if valid.shape != (batch, length):
        raise ValueError(
            f"attention_mask covers {tuple(valid.shape)} keys, expected "
            f"{(batch, length)}")
    return valid.to(device)


def _refuse_padding_when_evicting(valid: Optional[Tensor]) -> None:
    """H2O mode returns fewer keys than positions, so a padding mask cannot line up."""
    if valid is not None and not bool(valid.all()):
        raise NotImplementedError(
            "lo_bits=0 (H2O eviction) returns a compacted K/V, which HF's "
            "position-indexed padding mask cannot address. Run it unpadded "
            "(batch size 1 or equal-length prompts).")


def attention_colsum(q: Tensor, k: Tensor, q_start: int,
                     valid: Optional[Tensor], chunk_bytes: int) -> Tensor:
    """H2O's per-key score contribution: ``sum_rows sum_group softmax(qK^T)``.

    ``q`` is ``[B, Hq, L, D]`` (post-RoPE) for absolute positions
    ``q_start .. q_start + L - 1``; ``k`` is ``[B, H, S, D]``, every key held.
    Row ``i`` sees keys ``<= q_start + i`` (causal) that ``valid`` marks real;
    rows that are themselves padding contribute nothing. Returns ``[B, H, S]``
    fp32, summed over the ``Hq / H`` query heads of each KV head — GQA shares a
    KV head, so its importance is the group's.

    The logit is ``q . k`` in the model dtype, then fp32 from the scale on, as
    in HF's eager attention. Rows run in blocks so the fp32 transient stays
    under ``chunk_bytes`` — the prefill pass is O(L^2), as H2O's is.
    """
    b, hq, n_rows, d = q.shape
    h, s = k.shape[1], k.shape[2]
    if hq % h:
        raise ValueError(f"{hq} query heads do not group onto {h} KV heads")
    g = hq // h
    qg = q.reshape(b, h, g, n_rows, d)
    kt = k.unsqueeze(2).transpose(-1, -2)  # [B, H, 1, D, S]
    scale = 1.0 / math.sqrt(d)
    out = torch.zeros(b, h, s, dtype=torch.float32, device=k.device)
    key_pos = torch.arange(s, device=k.device)
    rows_per = max(1, int(chunk_bytes // max(b * hq * s * 4, 1)))
    neg = torch.finfo(torch.float32).min
    for r0 in range(0, n_rows, rows_per):
        r1 = min(n_rows, r0 + rows_per)
        logits = torch.matmul(qg[:, :, :, r0:r1], kt).float() * scale
        q_pos = torch.arange(q_start + r0, q_start + r1, device=k.device)
        allowed = key_pos.view(1, -1) <= q_pos.view(-1, 1)          # [r, S]
        allowed = allowed.view(1, 1, 1, r1 - r0, s)
        if valid is not None:
            allowed = allowed & valid.view(b, 1, 1, 1, s)
        logits = logits.masked_fill(~allowed, neg)
        probs = torch.softmax(logits, dim=-1)
        if valid is not None:
            row_ok = valid[:, q_start + r0:q_start + r1].to(probs.dtype)
            probs = probs * row_ok.view(b, 1, 1, r1 - r0, 1)
        out += probs.sum(dim=(2, 3))
    return out


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------


class MiKVCache(_HFCacheBase):
    """HF ``Cache`` implementing MiKV. Pair it with :func:`install_mikv_hooks`.

    Parameters
    ----------
    config : MiKVConfig
    num_layers : int
    dtype : torch.dtype
        The model dtype; tokens at 16 bits are stored in it.
    max_new_tokens : int, optional
        Generation horizon. Only pre-sizes the retained cache, so decode does
        not reallocate it; the budget itself is a ratio of the current length.
    """

    def __init__(self, config: MiKVConfig, num_layers: int,
                 dtype: torch.dtype = torch.float16,
                 max_new_tokens: Optional[int] = None) -> None:
        if isinstance(_HFCacheBase, type) and _HFCacheBase is not object:
            try:
                super().__init__()
            except (TypeError, ValueError):  # pragma: no cover - transformers >= 4.50
                pass
        self.config = config
        self.num_layers = num_layers
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens or 0)
        self.budget: Optional[MiKVBudget] = None
        self._layers: List[_Layer] = [_Layer() for _ in range(num_layers)]
        self._seen_tokens = 0
        # Filled by the hooks, consumed by update(): layer -> tensor.
        self._pending_q: Dict[int, Tensor] = {}
        self._pending_mask: Dict[int, Optional[Tensor]] = {}
        self.hooks_installed = False
        self.stats = {"prefills": 0, "decode_updates": 0, "demoted": 0}

    # -- HF Cache interface ------------------------------------------------

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        idx = layer_idx or 0
        if idx >= len(self._layers):
            return 0
        return self._layers[idx].length

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_max_length(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: Tensor) -> None:
        raise NotImplementedError(
            "MiKVCache does not support beam search (reorder_cache). Use greedy "
            "or sampling decoding, as the paper does.")

    def __len__(self) -> int:
        return sum(1 for layer in self._layers if layer.length)

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int,
               cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Tensor, Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        layer = self._layers[layer_idx]
        q = self._take_query(layer_idx, key_states, cache_kwargs)
        mask = self._pending_mask.pop(layer_idx, None)
        if layer.length == 0:
            return self._prefill(layer, key_states, value_states, q, mask)
        return self._decode(layer, key_states, value_states, q, mask)

    # -- hook plumbing -----------------------------------------------------

    def _stash_query(self, layer_idx: int, q_proj_out: Tensor) -> None:
        self._pending_q[layer_idx] = q_proj_out

    def _stash_mask(self, layer_idx: int, mask: Optional[Tensor]) -> None:
        self._pending_mask[layer_idx] = mask

    def _take_query(self, layer_idx: int, key_states: Tensor,
                    cache_kwargs: Optional[Dict[str, Any]]) -> Tensor:
        raw = self._pending_q.pop(layer_idx, None)
        if raw is None:
            raise MiKVHooksMissing(
                f"MiKVCache.update(layer {layer_idx}) has no captured query. H2O "
                "importance needs the query the model attends with; call "
                "modules.mikv.install_mikv_hooks(model, cache) before generate(). "
                "Scoring without it would demote tokens by nothing and still "
                "look like MiKV.")
        if not cache_kwargs or "cos" not in cache_kwargs or "sin" not in cache_kwargs:
            raise ValueError(
                "MiKVCache needs the RoPE cos/sin HF passes to update() "
                "(cache_kwargs['cos'/'sin']); this model does not pass them")
        b, _, n_new, d = key_states.shape
        q = raw.reshape(b, n_new, -1, d).transpose(1, 2)
        return apply_rope(q, cache_kwargs["cos"], cache_kwargs["sin"])

    # -- prefill -------------------------------------------------------------

    def _new_store(self, bits: int, head_dim: int, grow: bool,
                   pos_dtype: torch.dtype) -> _TokenStore:
        assert self.budget is not None
        return _TokenStore(bits, self.budget.group_size, head_dim, self.dtype,
                           pos_dtype, grow)

    def _prefill(self, layer: _Layer, k: Tensor, v: Tensor, q: Tensor,
                 mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        cfg = self.config
        b, h, n, d = k.shape
        if self.budget is None:
            elem = torch.empty((), dtype=self.dtype).element_size()
            self.budget = cfg.resolve(d, elem)
        budget = self.budget
        valid = key_validity(mask, b, n, k.device)

        evicting = cfg.lo_bits == EVICT
        if evicting:
            _refuse_padding_when_evicting(valid)

        # Balancer (Eq. 2): per-channel maxima over the prompt's real tokens.
        quantizing = not evicting or cfg.hi_bits < 16
        if cfg.channel_balancer and quantizing:
            g = q.shape[1] // h
            qa = q.abs().reshape(b, h, g, n, d)
            ka = k.abs()
            if valid is not None:
                qa = qa * valid.view(b, 1, 1, n, 1).to(qa.dtype)
                ka = ka * valid.view(b, 1, n, 1).to(ka.dtype)
            layer.balancer = channel_balancer(qa.amax(dim=(2, 3)), ka.amax(dim=2))

        scores = attention_colsum(q, k, 0, valid, cfg.score_chunk_bytes)

        # Selection: the newest n_recent, then the n_heavy best of the rest.
        n_hi = budget.n_hi(n)
        n_recent = budget.n_recent(n)
        n_heavy = n_hi - n_recent
        is_hi = torch.zeros(b, h, n, dtype=torch.bool, device=k.device)
        if n_recent:
            is_hi[:, :, n - n_recent:] = True
        if n_heavy:
            older = scores[:, :, : n - n_recent]
            heavy = older.topk(n_heavy, dim=-1).indices
            is_hi.scatter_(2, heavy, True)
        # One stable sort yields both tiers' positions, each ascending.
        order = torch.sort(is_hi.to(torch.int8), dim=-1, stable=True).indices
        lo_pos, hi_pos = order[:, :, : n - n_hi], order[:, :, n - n_hi:]

        layer.hi = self._new_store(cfg.hi_bits, d, grow=False, pos_dtype=torch.int64)
        layer.hi.balancer = layer.balancer if cfg.hi_bits < 16 else None
        hk, hv = _gather_tokens(k, hi_pos), _gather_tokens(v, hi_pos)
        layer.hi.set(layer.hi.encode(hk, hv), hi_pos)
        layer.hi_score = scores.gather(2, hi_pos)
        if evicting:
            layer.dropped = n - n_hi
        else:
            layer.lo = self._new_store(cfg.lo_bits, d, grow=True, pos_dtype=torch.int32)
            layer.lo.balancer = layer.balancer
            lk, lv = _gather_tokens(k, lo_pos), _gather_tokens(v, lo_pos)
            # Reserve what decode can demote into (at most one token per step),
            # so the retained cache does not reallocate mid-generation.
            layer.lo.set(layer.lo.encode(lk, lv), lo_pos, reserve=self.max_new_tokens)
        layer.length = n
        self.stats["prefills"] += 1
        # The prompt attends exactly; tiers only shape what later steps read.
        return k, v

    # -- decode ----------------------------------------------------------------

    def _decode(self, layer: _Layer, k_new: Tensor, v_new: Tensor, q: Tensor,
                mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        assert layer.hi is not None and layer.hi_score is not None
        assert self.budget is not None
        b, h, m, d = k_new.shape
        s_old = layer.length
        s_new = s_old + m
        hi, lo = layer.hi, layer.lo
        held = hi.length + (lo.length if lo is not None else layer.dropped)
        if held != s_old:
            raise AssertionError(f"tiers account for {held} of {s_old} tokens")
        valid = key_validity(mask, b, s_new, k_new.device)

        # 1. Read: both tiers + the new tokens. The new tokens are read as the
        #    importance cache will store them.
        new_parts = hi.encode(k_new, v_new)
        nk, nv = hi.decode(new_parts)
        hk, hv = hi.read()
        new_pos = torch.arange(s_old, s_new, device=k_new.device).expand(b, h, m)
        all_pos = torch.cat([hi.live_pos(), new_pos], dim=-1)
        all_score = torch.cat([layer.hi_score, torch.zeros(
            b, h, m, dtype=torch.float32, device=k_new.device)], dim=-1)
        if lo is None:
            # H2O: only the importance cache exists. Its positions are kept
            # ascending, so importance-then-new IS position order, compacted.
            _refuse_padding_when_evicting(valid)
            if m != 1:
                raise NotImplementedError(
                    "lo_bits=0 (H2O) scores one decode token per step")
            dense_k = torch.cat([hk, nk], dim=2)
            dense_v = torch.cat([hv, nv], dim=2)
            # 2. Score: every held key precedes the new query.
            all_score += attention_colsum(q, dense_k, dense_k.shape[2] - 1, None,
                                          self.config.score_chunk_bytes)
        else:
            # Dense, in position order, so HF's masks index it unchanged.
            dense_k = k_new.new_empty(b, h, s_new, d)
            dense_v = v_new.new_empty(b, h, s_new, d)
            if lo.length:
                lk, lv = lo.read()
                idx = _expand_pos(lo.live_pos(), d)
                dense_k.scatter_(2, idx, lk)
                dense_v.scatter_(2, idx, lv)
            if hi.length:
                idx = _expand_pos(hi.live_pos(), d)
                dense_k.scatter_(2, idx, hk)
                dense_v.scatter_(2, idx, hv)
            dense_k[:, :, s_old:] = nk
            dense_v[:, :, s_old:] = nv
            # 2. Score the new query rows against exactly what the model reads.
            contrib = attention_colsum(q, dense_k, s_old, valid,
                                       self.config.score_chunk_bytes)
            all_score += contrib.gather(2, all_pos)

        # 3. Demote the importance cache back to its size, H2O-style.
        excess = all_pos.shape[-1] - self.budget.n_hi(s_new)
        all_parts = {name: torch.cat([t, new_parts[name]], dim=2)
                     for name, t in hi.live_parts().items()}
        if excess > 0:
            recent_floor = s_new - self.budget.n_recent(s_new)
            cand = torch.where(all_pos < recent_floor, all_score,
                               torch.full_like(all_score, float("inf")))
            victims = cand.topk(excess, dim=-1, largest=False).indices
            keep_mask = torch.ones_like(all_score, dtype=torch.bool)
            keep_mask.scatter_(2, victims, False)
            # Stable sort keeps both groups in their existing order.
            order = torch.sort((~keep_mask).to(torch.int8), dim=-1, stable=True).indices
            n_keep = all_pos.shape[-1] - excess
            keep, victims = order[:, :, :n_keep], order[:, :, n_keep:]
            if lo is None:
                layer.dropped += excess
            else:
                vk = _gather_tokens(torch.cat([hk, nk], dim=2), victims)
                vv = _gather_tokens(torch.cat([hv, nv], dim=2), victims)
                lo.append(lo.encode(vk, vv), all_pos.gather(2, victims))
            all_parts = {name: _gather_tokens(t, keep) for name, t in all_parts.items()}
            all_pos = all_pos.gather(2, keep)
            all_score = all_score.gather(2, keep)
            self.stats["demoted"] += excess
        hi.set(all_parts, all_pos)
        layer.hi_score = all_score
        layer.length = s_new
        self.stats["decode_updates"] += 1
        return dense_k, dense_v

    # -- accounting ------------------------------------------------------------

    def memory_report(self) -> Dict[str, Any]:
        """Exact bytes held, by tier, against a dense fp16 cache of every token.

        ``kv_*`` are K+V payload (codes, grids, 16-bit tokens) — the quantity the
        paper's "cache size" measures. ``meta_bytes`` is what the paper leaves
        out: positions, H2O scores and the balancer. ``*_alloc`` counts
        reserved capacity as well as live tokens.
        """
        elem = torch.empty((), dtype=self.dtype).element_size()
        hi_kv = lo_kv = lo_kv_alloc = meta = meta_alloc = 0
        dense = 0
        for layer in self._layers:
            if layer.hi is None:
                continue
            hi_kv += layer.hi.kv_bytes()
            m = m_alloc = layer.hi.pos_bytes()
            if layer.lo is not None:
                lo_kv += layer.lo.kv_bytes()
                lo_kv_alloc += layer.lo.kv_bytes(live=False)
                m += layer.lo.pos_bytes()
                m_alloc += layer.lo.pos_bytes(live=False)
            for t in (layer.hi_score, layer.balancer):
                if t is not None:
                    m += t.numel() * t.element_size()
                    m_alloc += t.numel() * t.element_size()
            meta += m
            meta_alloc += m_alloc
            bsz, heads = layer.hi.pos.shape[:2]
            dense += 2 * bsz * heads * layer.length * layer.hi.head_dim * elem
        kv = hi_kv + lo_kv
        first = self._layers[0] if self._layers else None
        return {
            "layers": sum(1 for layer in self._layers if layer.length),
            "tokens": first.length if first else 0,
            # Per (row, KV head); every head and layer holds the same counts.
            "hi_tokens": first.hi.length if first and first.hi else 0,
            "lo_tokens": first.lo.length if first and first.lo else 0,
            "dropped_tokens": first.dropped if first else 0,
            "hi_kv_bytes": hi_kv,
            "lo_kv_bytes": lo_kv,
            "kv_bytes": kv,
            "meta_bytes": meta,
            "total_bytes": kv + meta,
            "total_alloc_bytes": hi_kv + lo_kv_alloc + meta_alloc,
            "dense_fp16_bytes": dense,
            "kv_fraction": kv / dense if dense else float("nan"),
            "total_fraction": (kv + meta) / dense if dense else float("nan"),
            "budget": None if self.budget is None else dict(vars(self.budget)),
        }


def _expand_pos(pos: Tensor, d: int) -> Tensor:
    return pos.long().unsqueeze(-1).expand(*pos.shape, d)


def _gather_tokens(x: Tensor, idx: Tensor) -> Tensor:
    """``x[b, h, idx[b, h, i], ...]`` for ``x`` of any trailing shape."""
    idx = idx.long()
    trail = x.shape[3:]
    view = idx.view(*idx.shape, *([1] * len(trail))).expand(*idx.shape, *trail)
    return x.gather(2, view)
