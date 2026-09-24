"""One run, every observation: per-step datapoints from a single decode, zipped.

A single long-context, long-generation decode (default 512 prompt tokens + 512
generated) through the real cache at one operating point, recording at EVERY
step and EVERY layer everything the observation metrics need. Nothing here
needs a second run:

* **Ground truth comes from the same run.** The cache's ``update()`` is wrapped
  so a *shadow* copy of the full, never-evicted, never-quantized KV (post-RoPE
  keys, exactly as the model handed them over) sits beside it. At every step
  and layer the model's own post-RoPE query is taken off ``q_proj`` and its
  full-context attention over the shadow is computed. So "missed mass" is the
  mass THIS query would have put on what the cache no longer holds -- the same
  query, the same trajectory, no teacher-forced second model.
* **The cache is read, not re-simulated.** Tier of every window (fp / int2 /
  local / evicted), which int2 windows the read gate opened per KV head, the
  card estimates it ranked on, the cache's own per-step window mass (exact for
  what it read, the calibrated fill for what it skipped), its running eviction
  scores, and the exact ranking signal at every eviction.
* **Where the output error comes from, per head, per step.** Five attention
  outputs from the same query: full KV; the kept tokens at their original
  values; kept, with the int2 tier dequantized; kept as actually held (promoted
  windows carry their dequantized payload); and what the fused kernel actually
  produced. Consecutive differences isolate eviction, int2 quantization, the
  promotion payload and the read gate.

Output: ``<out>.zip`` holding ``meta.json``, ``SCHEMA.md`` (every array, its
shape, dtype and which metric it feeds) and one ``sample_NNN.npz`` per prompt.

Command::

    python -m modules.evaluation.observation_collector \\
        --model-path /path/to/llama-3.1-8b-instruct --dataset hotpotqa \\
        --window-size 8 --cache-budget 0.20 --gate-ratio 0.25 --quant-ratio 0.70

``--dataset`` takes a LongBench name (``$LONGBENCH_LOCAL_DIR/<name>.jsonl``,
field ``context``), ``ruler:<task>`` (``$RULER_DATA``), ``wikitext-103``,
``pg19`` or a path. The prompt is the first ``--prefill`` tokens of a document
(shorter documents are skipped); the continuation is the cache's own greedy
decode, ``--gen`` decode steps.

``python -m modules.evaluation.observation_collector export --zip X --out-dir D``
turns a zip into the parity-npz pair the existing observation suite reads
(``qevict_observations``: Observations I-V).

Requires CUDA + flash-attn for the gated path (``--backend flash_attn``, the
default). ``--backend eager`` runs the same cache ungated.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from utils.logger import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = "1.0"

#: Tier codes on the absolute window axis.
TIER_NONE, TIER_FP, TIER_Q, TIER_LOCAL, TIER_EVICTED = -1, 0, 1, 2, 3

#: Attention-output ladder, each against the full-KV output of the same query.
LADDER = ("kept_original", "kept_int2_dequant", "held", "actual")
#: Consecutive differences of the ladder -- one cause each.
LADDER_STEPS = ("eviction", "int2_quantization", "promotion_payload", "read_gate")

LB_NAMES = {
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class CollectorConfig:
    model_path: str
    dataset: str
    window_size: int
    cache_budget: float
    gate_ratio: float
    quant_ratio: float
    prefill: int = 512
    gen: int = 512
    num_samples: int = 8
    article_index: int = 0
    num_sink_tokens: int = 5
    local_window_size: int = 128
    quant_budget_mode: str = "bytes"
    quant_card_bits: Any = None
    quant_promotion: str = "bidir"
    quant_promote_source: str = "dequant"
    backend: str = "flash_attn"
    dtype: str = "float16"
    token_heads: bool = False
    seed: int = 42
    out: Optional[str] = None
    longbench_dir: str = ""
    ruler_data: str = ""
    text_field: Optional[str] = None
    record_filter: Optional[str] = None

    def validate(self) -> None:
        if self.window_size % 4:
            raise ValueError("window_size must be a multiple of 4 (int2 packing)")
        if self.local_window_size % self.window_size:
            raise ValueError(f"local_window_size {self.local_window_size} is not a "
                             f"multiple of window_size {self.window_size}")
        if not (0.0 < self.quant_ratio <= 1.0):
            raise ValueError("quant_ratio must be in (0, 1]: this collector "
                             "observes the two-tier cache")
        if not (0.0 < self.gate_ratio <= 1.0):
            raise ValueError("gate_ratio must be in (0, 1]")
        if self.backend not in ("flash_attn", "eager"):
            raise ValueError("backend must be flash_attn or eager")

    def resolve_dataset(self) -> Tuple[str, Optional[str], Optional[str], str]:
        """``(corpus, text_field, record_filter, slug)`` for CorpusLoader."""
        d = self.dataset
        if d in ("wikitext-103", "pg19"):
            return d, self.text_field, self.record_filter, d
        if d.startswith("ruler:"):
            task = d.split(":", 1)[1]
            root = self.ruler_data or os.environ.get("RULER_DATA", "")
            if not root or not (Path(root) / "dataset_info.json").exists():
                raise FileNotFoundError(
                    f"RULER save_to_disk dir not found ({root!r}); pass "
                    "--ruler-data or set RULER_DATA")
            return root, self.text_field or "context", f"task={task}", \
                f"ruler{Path(root).name}-{task}"
        if d in LB_NAMES:
            root = self.longbench_dir or os.environ.get("LONGBENCH_LOCAL_DIR", "")
            path = Path(root) / f"{d}.jsonl"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found; pass --longbench-dir or set "
                    "LONGBENCH_LOCAL_DIR")
            return str(path), self.text_field or "context", self.record_filter, \
                f"lb-{d}"
        if Path(d).expanduser().exists():
            return d, self.text_field, self.record_filter, Path(d).stem
        raise ValueError(f"dataset {d!r}: not wikitext-103, pg19, a LongBench "
                         "name, ruler:<task>, or a path")

    def default_out(self, slug: str) -> Path:
        tag = (f"{slug}_ws{self.window_size}_b{self.cache_budget:g}"
               f"_g{self.gate_ratio:g}_q{self.quant_ratio:g}"
               f"_p{self.prefill}_n{self.gen}")
        if self.quant_promotion != "bidir":
            tag += f"_{self.quant_promotion}"
        if self.quant_promote_source != "dequant":
            tag += f"_{self.quant_promote_source}"
        return Path("outputs") / "obs_data" / f"{tag}.zip"


# ---------------------------------------------------------------------------
# the shadow full KV
# ---------------------------------------------------------------------------


class ShadowKV:
    """Every token's post-RoPE K and V, per layer, never evicted or quantized.

    Index ``p`` holds absolute position ``p`` (B=1, no padding, tokens arrive in
    order), which is what lets the cache's ``position_ids`` and the int2 tier's
    frozen positions index straight into it.
    """

    def __init__(self, L: int, H_kv: int, S_max: int, D: int, dtype, device):
        self.k = torch.zeros(L, H_kv, S_max, D, dtype=dtype, device=device)
        self.v = torch.zeros(L, H_kv, S_max, D, dtype=dtype, device=device)
        self.n = [0] * L

    def append(self, layer: int, k: Tensor, v: Tensor) -> None:
        if k.shape[0] != 1:
            raise ValueError("the collector records batch size 1 only")
        t = k.shape[2]
        s = self.n[layer]
        if s + t > self.k.shape[2]:
            raise RuntimeError(f"shadow KV overflow at layer {layer}: {s}+{t} > "
                               f"{self.k.shape[2]}")
        self.k[layer, :, s:s + t] = k[0].to(self.k.dtype)
        self.v[layer, :, s:s + t] = v[0].to(self.v.dtype)
        self.n[layer] = s + t

    def get(self, layer: int) -> Tuple[Tensor, Tensor]:
        n = self.n[layer]
        return self.k[layer, :, :n], self.v[layer, :, :n]


# ---------------------------------------------------------------------------
# per-sample buffers
# ---------------------------------------------------------------------------


class SampleBuffers:
    """Everything recorded for one prompt, preallocated on device.

    Axes: ``T`` = forwards (index 0 is the prompt, ``t >= 1`` decode step
    ``t``), ``L`` layers, ``H`` query heads, ``Hk`` KV heads, ``W`` absolute
    windows (window ``w`` = post-sink tokens ``[ns + w*ws, ns + (w+1)*ws)``),
    ``S`` absolute token positions.
    """

    def __init__(self, T, L, H, Hk, W, S, token_heads, device):
        f16, f32 = torch.float16, torch.float32
        nan = float("nan")

        def full(shape, value, dtype):
            return torch.full(shape, value, dtype=dtype, device=device)

        self.full_mass = full((T, L, H, W), 0.0, f16)
        self.sink_mass = full((T, L, H), nan, f32)
        self.token_mass = full((T, L, S), 0.0, f16)
        self.token_mass_heads = full((T, L, H, S), 0.0, f16) if token_heads else None
        self.cache_step_mass = full((T, L, H, W), nan, f16)
        self.running_score_mean = full((T, L, W), nan, f32)
        self.tier = full((T, L, W), TIER_NONE, torch.int8)
        self.fp_dequant = full((T, L, W), False, torch.bool)
        self.gate_open = full((T, L, Hk, W), False, torch.bool)
        self.gate_fired = full((T, L), False, torch.bool)
        self.card_logmass = full((T, L, H, W), nan, f16)
        self.out_err = full((T, L, H, len(LADDER)), nan, f32)
        self.out_step_err = full((T, L, H, len(LADDER_STEPS)), nan, f32)
        self.out_norm = full((T, L, H), nan, f32)
        self.n_fp_tokens = full((T, L), -1, torch.int32)
        self.n_q_windows = full((T, L), -1, torch.int32)
        self.q_recon_err = full((L, Hk, W, 2), nan, f32)
        self.q_first_step = full((L, W), -1, torch.int32)
        self.evict_step = full((T,), False, torch.bool)
        self.rank_signal: List[Tuple[int, Tensor]] = []      # (t, [L, H, W] f32)
        self.tokens: List[int] = []
        self.token_logprob: List[float] = []
        self.token_entropy: List[float] = []
        self.token_top5_ids: List[List[int]] = []
        self.token_top5_logprob: List[List[float]] = []

    def to_numpy(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k, v in vars(self).items():
            if isinstance(v, Tensor):
                out[k] = v.detach().cpu().numpy()
        if self.rank_signal:
            out["rank_signal_steps"] = np.array([t for t, _ in self.rank_signal],
                                                dtype=np.int32)
            out["rank_signal"] = torch.stack(
                [x for _, x in self.rank_signal]).cpu().numpy()
        else:
            out["rank_signal_steps"] = np.zeros(0, np.int32)
        out["tokens"] = np.asarray(self.tokens, np.int64)
        out["token_logprob"] = np.asarray(self.token_logprob, np.float32)
        out["token_entropy"] = np.asarray(self.token_entropy, np.float32)
        out["token_top5_ids"] = np.asarray(self.token_top5_ids, np.int64)
        out["token_top5_logprob"] = np.asarray(self.token_top5_logprob, np.float32)
        return out


# ---------------------------------------------------------------------------
# the maths, standalone so it is testable without a model
# ---------------------------------------------------------------------------


def _gqa(x: Tensor, rep: int) -> Tensor:
    """``[Hk, n, D] -> [H, n, D]``: query head ``h`` reads KV head ``h // rep``."""
    return x.repeat_interleave(rep, dim=0) if rep > 1 else x


def attend(q: Tensor, K: Tensor, V: Tensor, scale: float, rep: int
           ) -> Tuple[Tensor, Tensor]:
    """One decode query over a key set. ``q [H, D]``, ``K, V [Hk, n, D]``;
    returns ``(o [H, D], p [H, n])`` in fp32."""
    Kr, Vr = _gqa(K, rep).float(), _gqa(V, rep).float()
    p = torch.softmax(torch.einsum("hd,hnd->hn", q.float(), Kr) * scale, dim=-1)
    return torch.einsum("hn,hnd->hd", p, Vr), p


def prefill_token_mass(q: Tensor, K: Tensor, scale: float, rep: int,
                       chunk: int = 256) -> Tensor:
    """Causal attention of ``P`` prompt queries over ``P`` keys, summed over the
    query rows: ``[H, P]`` (fp32). Chunked over rows to bound memory."""
    H, P, _ = q.shape
    Kr = _gqa(K, rep).float()
    out = torch.zeros(H, K.shape[1], device=q.device)
    keys = torch.arange(K.shape[1], device=q.device)
    off = K.shape[1] - P
    for s in range(0, P, chunk):
        qs = q[:, s:s + chunk].float()
        logits = torch.einsum("hqd,hkd->hqk", qs, Kr) * scale
        rows = torch.arange(s, s + qs.shape[1], device=q.device) + off
        logits.masked_fill_(keys[None, None, :] > rows[None, :, None], float("-inf"))
        out += torch.softmax(logits, dim=-1).sum(dim=1)
    return out


def window_mass(tok: Tensor, ns: int, ws: int, W: int) -> Tensor:
    """``[H, S]`` token mass -> ``[H, W]`` post-sink window mass."""
    post = tok[:, ns:]
    n = post.shape[1]
    if n < W * ws:
        post = F.pad(post, (0, W * ws - n))
    return post[:, :W * ws].reshape(tok.shape[0], W, ws).sum(-1)


def _rel(a: Tensor, b: Tensor, ref: Tensor) -> Tensor:
    return (a - b).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)


def held_int2_tier(store, rope, ws: int, D: int, dtype: torch.dtype
                   ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """The active int2 tier as the cache holds it: dequantized K (post-RoPE at
    the frozen positions) and V ``[Hk, n*ws, D]`` fp32, positions ``[n*ws]`` and
    window ids ``[n]`` (ascending, the gate's active order). Dequantized and
    rotated at the KV store's ``dtype``, as the cache's own read path does.

    Reads through the slot table, so it works on a per-layer store and on the
    layer-major path's read-only per-layer view alike.
    """
    if store is None or store.num_active_windows == 0 or store.table is None:
        return None
    from modules.quant.effective import dequant_rotate_q_keys
    from modules.quant.quantizer import dequantize_value_windows
    table = store.table
    n = int(store.num_active_windows)
    idx = table.active_order(n)
    kc, ks, kz, vc, vs, vz, pos = table.gather(idx)
    Hk = table.num_kv_heads
    pos_flat = pos.reshape(1, n * ws)
    keys = dequant_rotate_q_keys(kc, ks, kz, ws, pos_flat, rope, dtype,
                                 1, n, Hk, D)[0]
    vals = dequantize_value_windows(vc, vs, vz, D, out_dtype=dtype)
    vals = vals.reshape(1, n, Hk, ws, D).permute(0, 2, 1, 3, 4).reshape(Hk, n * ws, D)
    ids = table.slot_wid.gather(1, idx)[0]
    return keys.float(), vals.float(), pos_flat[0], ids.long()


def _merged_ids(orig: Tensor, width: int) -> Tensor:
    """Window ids of a ``width``-column score tensor on the merged axis: the
    survivors' ids, then new windows (consecutive after the newest)."""
    n = orig.shape[0]
    if width <= n:
        return orig[:width]
    nxt = (int(orig.max()) + 1) if n else 0
    extra = torch.arange(nxt, nxt + width - n, device=orig.device)
    return torch.cat([orig, extra])


def record_decode_layer(buf: SampleBuffers, t: int, layer: int, q: Tensor,
                        o_actual: Tensor, cache, shadow: ShadowKV, gate,
                        ns: int, ws: int, scale: float) -> None:
    """Record one layer of one decode step. ``q``/``o_actual`` are ``[H, D]``."""
    H, D = q.shape
    K, V = shadow.get(layer)
    S = K.shape[1]
    Hk = K.shape[0]
    rep = H // Hk
    W = buf.full_mass.shape[-1]
    dev = q.device

    # ---- ground truth: this query over everything ----------------------
    o_full, p = attend(q, K, V, scale, rep)                         # p [H, S]
    buf.full_mass[t, layer] = window_mass(p, ns, ws, W).to(torch.float16)
    buf.sink_mass[t, layer] = p[:, :ns].sum(-1)
    buf.token_mass[t, layer, :S] = p.mean(0).to(torch.float16)
    if buf.token_mass_heads is not None:
        buf.token_mass_heads[t, layer, :, :S] = p.to(torch.float16)
    buf.out_norm[t, layer] = o_full.norm(dim=-1)

    # ---- what the cache holds -------------------------------------------
    st = cache._states[layer]
    fpK, fpV = st.key_states[0].float(), st.value_states[0].float()
    fpPos = st.position_ids[0].long()
    store = cache._stores[layer]
    q_tier = held_int2_tier(store, cache.rope_module, ws, D, st.key_states.dtype)
    buf.n_fp_tokens[t, layer] = fpPos.numel()
    buf.n_q_windows[t, layer] = 0 if q_tier is None else q_tier[3].numel()

    Kfo, Vfo = K[:, fpPos].float(), V[:, fpPos].float()
    if q_tier is None:
        qK = qV = torch.zeros(Hk, 0, D, device=dev)
        qPos = torch.zeros(0, dtype=torch.long, device=dev)
        q_ids = torch.zeros(0, dtype=torch.long, device=dev)
    else:
        qK, qV, qPos, q_ids = q_tier
        qPos = qPos.long()
        if bool((q_ids >= W).any()):
            raise RuntimeError(f"int2 window id beyond the window axis W={W}")

    # ---- the output-error ladder ----------------------------------------
    o_kept, _ = attend(q, torch.cat([Kfo, K[:, qPos].float()], 1),
                       torch.cat([Vfo, V[:, qPos].float()], 1), scale, rep)
    o_qdeq, _ = attend(q, torch.cat([Kfo, qK], 1), torch.cat([Vfo, qV], 1),
                       scale, rep)
    o_held, _ = attend(q, torch.cat([fpK, qK], 1), torch.cat([fpV, qV], 1),
                       scale, rep)
    o_act = o_actual.float()
    ladder = (o_kept, o_qdeq, o_held, o_act)
    buf.out_err[t, layer] = torch.stack([_rel(o, o_full, o_full) for o in ladder], -1)
    prev = (o_full,) + ladder[:-1]
    buf.out_step_err[t, layer] = torch.stack(
        [_rel(a, b, o_full) for a, b in zip(ladder, prev)], -1)

    # ---- tiers on the absolute window axis ------------------------------
    w_exist = min(W, max(0, -(-(S - ns) // ws)))
    tier = torch.full((W,), TIER_NONE, dtype=torch.int8, device=dev)
    tier[:w_exist] = TIER_EVICTED
    body = fpPos >= ns
    fp_w = torch.unique((fpPos[body] - ns) // ws)
    fp_w = fp_w[fp_w < W]
    tier[fp_w] = TIER_FP
    orig = st.original_window_ids
    if orig is not None:
        orig = orig[0].long()
        W_scored = orig.numel()
        local_w = int(cache._policies[layer].tier_counts(W_scored)[2])
        loc = orig[W_scored - local_w:] if local_w else orig[:0]
        newer = fp_w[fp_w > (int(orig.max()) if orig.numel() else -1)]
        loc = torch.cat([loc, newer])
        loc = loc[loc < W]
        tier[loc] = TIER_LOCAL
    if q_ids.numel():
        qi = q_ids[q_ids < W].long()
        tier[qi] = TIER_Q
    buf.tier[t, layer] = tier

    # Promoted windows: fp tokens that differ from what the model produced.
    if fpPos.numel():
        diff = ((fpK - Kfo).abs().amax(dim=(0, 2)) > 0) | \
               ((fpV - Vfo).abs().amax(dim=(0, 2)) > 0)
        dq = fpPos[diff & body]
        if dq.numel():
            dw = torch.unique((dq - ns) // ws)
            buf.fp_dequant[t, layer][dw[dw < W]] = True

    # int2 reconstruction error, once per window (codes are frozen for life).
    if q_ids.numel():
        n = q_ids.numel()
        kq, vq = qK.reshape(Hk, n, ws, D), qV.reshape(Hk, n, ws, D)
        ko = K[:, qPos].float().reshape(Hk, n, ws, D)
        vo = V[:, qPos].float().reshape(Hk, n, ws, D)
        ek = (kq - ko).flatten(2).norm(dim=-1) / ko.flatten(2).norm(dim=-1).clamp_min(1e-12)
        ev = (vq - vo).flatten(2).norm(dim=-1) / vo.flatten(2).norm(dim=-1).clamp_min(1e-12)
        new = buf.q_first_step[layer][q_ids] < 0
        if bool(new.any()):
            wi = q_ids[new]
            buf.q_first_step[layer][wi] = t
            buf.q_recon_err[layer, :, :, 0][:, wi] = ek[:, new]
            buf.q_recon_err[layer, :, :, 1][:, wi] = ev[:, new]

    # ---- the cache's own scores ------------------------------------------
    if orig is not None and st.window_scores is not None:
        rs = st.window_scores[0].float().mean(0)                      # [W_scored]
        buf.running_score_mean[t, layer][orig[orig < W]] = rs[orig < W]
    inc = (cache.cache_kwargs.get(layer) or {}).get("window_scores")
    if inc is not None and orig is not None:
        mids = _merged_ids(orig, inc.shape[-1])
        keep = mids < W
        buf.cache_step_mass[t, layer][:, mids[keep]] = \
            inc[0].float()[:, keep].to(torch.float16)

    # ---- the read gate ---------------------------------------------------
    if gate is not None and layer in gate.picks:
        buf.gate_fired[t, layer] = True
        pick = gate.picks[layer][0].long()                           # [Hk, n_sel]
        for kv in range(pick.shape[0]):
            buf.gate_open[t, layer, kv][pick[kv][pick[kv] < W]] = True
        act = gate.active_ids[layer][0].long()
        lm = gate.logmass[layer][0].float()                          # [H, n]
        keep = act < W
        buf.card_logmass[t, layer][:, act[keep]] = lm[:, keep].to(torch.float16)


def record_prefill_layer(buf: SampleBuffers, layer: int, q: Tensor, cache,
                         shadow: ShadowKV, ns: int, ws: int, scale: float) -> None:
    """Step 0: the prompt's causal attention, summed over its query rows (the
    same quantity the H2O-style cumulative score starts from)."""
    K, _ = shadow.get(layer)
    rep = q.shape[0] // K.shape[0]
    W = buf.full_mass.shape[-1]
    tok = prefill_token_mass(q, K, scale, rep)                      # [H, P]
    S = tok.shape[1]
    buf.full_mass[0, layer] = window_mass(tok, ns, ws, W).to(torch.float16)
    buf.sink_mass[0, layer] = tok[:, :ns].sum(-1)
    buf.token_mass[0, layer, :S] = tok.mean(0).to(torch.float16)
    if buf.token_mass_heads is not None:
        buf.token_mass_heads[0, layer, :, :S] = tok.to(torch.float16)
    tier = torch.full((W,), TIER_NONE, dtype=torch.int8, device=q.device)
    tier[:min(W, max(0, -(-(S - ns) // ws)))] = TIER_FP
    buf.tier[0, layer] = tier
    buf.n_fp_tokens[0, layer] = S
    buf.n_q_windows[0, layer] = 0
    inc = (cache.cache_kwargs.get(layer) or {}).get("window_scores")
    if inc is not None:
        w = min(W, inc.shape[-1])
        buf.cache_step_mass[0, layer, :, :w] = inc[0, :, :w].to(torch.float16)


def snapshot_rank_signal(buf: SampleBuffers, t: int, cache, L: int) -> None:
    """What the eviction at step ``t`` ranks on, for every layer, taken before
    any layer's ``update()`` runs: the running scores plus the increment the
    previous step left pending (``update()`` adds it, then evicts)."""
    H, W = buf.full_mass.shape[2], buf.full_mass.shape[3]
    snap = torch.full((L, H, W), float("nan"), device=buf.full_mass.device)
    for layer in range(L):
        st = cache._states[layer]
        inc = (cache.cache_kwargs.get(layer) or {}).get("window_scores")
        if st.window_scores is None:
            # The first eviction: nothing accumulated yet, so update() starts
            # the running score from the prompt's scores, still pending here.
            if inc is None:
                continue
            tot = inc[0].float()
            ids = torch.arange(tot.shape[-1], device=tot.device)
        else:
            ws_ = st.window_scores[0].float()
            width = ws_.shape[-1] if inc is None else max(ws_.shape[-1],
                                                          inc.shape[-1])
            tot = F.pad(ws_, (0, width - ws_.shape[-1]))
            if inc is not None:
                tot = tot + F.pad(inc[0].float(), (0, width - inc.shape[-1]))
            ids = _merged_ids(st.original_window_ids[0].long(), width)
        keep = ids < W
        snap[layer][:, ids[keep]] = tot[:, keep]
    buf.rank_signal.append((t, snap))


# ---------------------------------------------------------------------------
# wiring: the shadow, the model taps, the zip
# ---------------------------------------------------------------------------


def attach_shadow(cache, shadow: ShadowKV):
    """Route every ``cache.update()`` through ``shadow`` first. Returns the
    original bound method (reassign it to detach). The keys HF hands to
    ``update()`` are already RoPE'd, which is what the shadow must hold."""
    orig = cache.update

    def update(k, v, layer_idx, cache_kwargs=None):
        shadow.append(layer_idx, k, v)
        return orig(k, v, layer_idx, cache_kwargs)

    cache.update = update
    return orig


def post_rope_query(q_raw: Tensor, position_embeddings, H: int, D: int) -> Tensor:
    """``q_proj`` output ``[1, Tq, H*D]`` -> post-RoPE query ``[H, Tq, D]``,
    exactly as the attention forward builds it."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    Tq = q_raw.shape[1]
    q = q_raw.view(1, Tq, H, D).transpose(1, 2)
    cos, sin = position_embeddings
    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
    return q[0]


class ModelTaps:
    """Forward hooks that hand ``on_layer(layer, q [H,Tq,D], o [H,Tq,D])`` the
    post-RoPE query and the attention output BEFORE ``o_proj`` -- per head,
    whatever path produced it (the fused kernel included) -- after each
    attention module returns. ``on_layer_pre(layer)`` runs before it starts.

    Register AFTER ``install_score_hooks`` so the per-step scores the cache
    writes during the forward are in place when ``on_layer`` reads them.
    """

    def __init__(self, model, H: int, D: int, on_layer, on_layer_pre=None):
        from modules.windowed_cache.hooks import _get_attn_classes
        self.H, self.D = H, D
        self._q: Dict[int, Tensor] = {}
        self._o: Dict[int, Tensor] = {}
        self.handles = []
        for m in model.modules():
            if not isinstance(m, _get_attn_classes()):
                continue
            li = m.layer_idx
            self.handles.append(m.q_proj.register_forward_hook(self._qh(li)))
            self.handles.append(m.o_proj.register_forward_pre_hook(self._oh(li)))
            if on_layer_pre is not None:
                self.handles.append(m.register_forward_pre_hook(
                    lambda _m, _a, li=li: on_layer_pre(li)))
            self.handles.append(m.register_forward_hook(
                self._ah(li, on_layer), with_kwargs=True))

    def _qh(self, li):
        def hook(_m, _inp, out):
            self._q[li] = out.detach()
        return hook

    def _oh(self, li):
        def hook(_m, args):
            self._o[li] = args[0].detach()
        return hook

    def _ah(self, li, on_layer):
        def hook(_m, args, kwargs, _out):
            pe = kwargs.get("position_embeddings")
            if pe is None and len(args) > 1:
                pe = args[1]
            if pe is None:
                raise RuntimeError(
                    f"layer {li}: no position_embeddings in the attention call; "
                    "the collector cannot rebuild the post-RoPE query")
            q = post_rope_query(self._q.pop(li), pe, self.H, self.D)
            o_raw = self._o.pop(li)
            o = o_raw.view(1, o_raw.shape[1], self.H, self.D)[0].transpose(0, 1)
            on_layer(li, q, o)
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


class ZipWriter:
    """Streams ``sample_NNN.npz`` members into ``<path>.partial`` and renames
    it on :meth:`close`, so an interrupted run never leaves a zip that looks
    complete."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = self.path.with_suffix(self.path.suffix + ".partial")
        self.zf = zipfile.ZipFile(self.tmp, "w", compression=zipfile.ZIP_STORED)

    def add_sample(self, name: str, arrays: Dict[str, np.ndarray]) -> None:
        bio = io.BytesIO()
        np.savez_compressed(bio, **arrays)
        self.zf.writestr(name, bio.getvalue())

    def close(self, meta: Dict[str, Any]) -> Path:
        self.zf.writestr("meta.json", json.dumps(meta, indent=2, default=str))
        self.zf.writestr("SCHEMA.md", SCHEMA.format(version=SCHEMA_VERSION))
        self.zf.close()
        self.tmp.replace(self.path)
        return self.path


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


SCHEMA = """# Observation data -- schema {version}

One zip per run: `meta.json` (the configuration, the resolved cache geometry,
the read-gate verdict, per-sample identity) and one `sample_NNN.npz` per prompt.

Axes: `T` forwards (index 0 = the prompt, `t >= 1` = decode step t), `L` layers,
`H` query heads, `Hk` KV heads (query head h reads KV head h // (H/Hk)), `W`
absolute windows (window w = post-sink positions [ns + w*ws, ns + (w+1)*ws)),
`S` absolute token positions, `R` eviction events.

Ground truth is the SAME query's attention over the full, never-evicted KV kept
in a shadow beside the cache (no second run).

| array | shape, dtype | what it is |
|---|---|---|
| `full_mass` | T,L,H,W f16 | full-KV attention mass per window (t=0: summed over prompt rows) |
| `sink_mass` | T,L,H f32 | full-KV mass on the sink tokens |
| `token_mass` | T,L,S f16 | full-KV mass per token, head-mean (re-bucket to any window size) |
| `token_mass_heads` | T,L,H,S f16 | per head (only with --token-heads) |
| `tier` | T,L,W i8 | -1 not yet created, 0 fp, 1 int2, 2 local, 3 evicted |
| `fp_dequant` | T,L,W bool | fp window holding a DEQUANTIZED payload (was promoted from int2) |
| `gate_open` | T,L,Hk,W bool | int2 windows the read gate opened for that KV head |
| `gate_fired` | T,L bool | the gate ran for that layer and step |
| `card_logmass` | T,L,H,W f16 | the card's log-mass estimate the gate ranked on (NaN off-tier) |
| `cache_step_mass` | T,L,H,W f16 | the cache's own per-step window mass: exact where read, the calibrated fill where skipped |
| `running_score_mean` | T,L,W f32 | the cache's running eviction score, head-mean, after the step's update |
| `rank_signal` | R,L,H,W f32 | exactly what each eviction ranked on (per head; the policy ranks the head-mean) |
| `rank_signal_steps` | R i32 | the step of each eviction |
| `evict_step` | T bool | an eviction fired at this step |
| `out_err` | T,L,H,4 f32 | rel. L2 error vs the full-KV output of the same query: kept_original, kept_int2_dequant, held, actual |
| `out_step_err` | T,L,H,4 f32 | consecutive rungs: eviction, int2_quantization, promotion_payload, read_gate |
| `out_norm` | T,L,H f32 | norm of the full-KV attention output |
| `n_fp_tokens`, `n_q_windows` | T,L i32 | occupancy |
| `q_recon_err` | L,Hk,W,2 f32 | int2 reconstruction error of each window (K, V), measured when first seen in int2 |
| `q_first_step` | L,W i32 | step a window was first seen in int2 (-1 never) |
| `tokens` | T i64 | the token each forward produced (greedy, through the cache) |
| `token_logprob`, `token_entropy` | T f32 | its log-probability; the entropy of the next-token distribution |
| `token_top5_ids`, `token_top5_logprob` | T,5 | the top five |
| `prompt_ids` | P i64 | the prompt |

## Which array feeds which observation

| observation | arrays |
|---|---|
| skewed importance (I) | `full_mass` (cumsum over t = the oracle's H2O score) |
| missed mass (instantaneous) | `full_mass`, `sink_mass`, `tier` |
| future missed mass (II), selection churn | `full_mass` over t+1..t+H, `tier` at `evict_step` |
| tier mass distribution | `full_mass` x `tier` |
| Jaccard / top-k overlap vs oracle | `full_mass` cumsum vs `tier` |
| int2 fidelity | `cache_step_mass` vs `full_mass` on tier 1; `q_recon_err` |
| recovered mass (int2 vs evict) | `full_mass` on tier 1 |
| revival / LIR (III) | `tier` sequences, `full_mass` ranking |
| decode read ledger + per-head gate recall (IV) | `full_mass`, `sink_mass`, `tier`, `gate_open` |
| tier moves: promote / demote / evict and their payoff (V) | `tier` sequences, `full_mass` future windows |
| E1 card ranking vs truth | `card_logmass` vs `full_mass` on tier 1; `gate_open` |
| E2 one-way counterfactual | replay `rank_signal` with int2 windows blocked from fp; score with `full_mass` |
| E3 promotion payload | `fp_dequant`, `out_step_err[..., 2]` (promotion_payload), `q_recon_err` |
| fill calibration | `cache_step_mass` vs `full_mass` where tier 1 and not `gate_open` |
| granularity (any window size) | `token_mass` re-bucketed |
"""


class ObservationCollector:
    def __init__(self, cfg: CollectorConfig) -> None:
        cfg.validate()
        self.cfg = cfg

    def run(self) -> Path:
        cfg = self.cfg
        from data.corpus_loader import CorpusLoader
        from modules.evaluation.base_parity_runner import select_parity_articles
        from modules.evaluation.ours_parity_runner import GateRecorder, forward_at
        from modules.windowed_cache import (WindowedCache, WindowedCacheConfig,
                                            flash_decode, install_score_hooks)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from utils.cache_factory import (assert_transformers_version_supported,
                                         validate_backend_attn_pairing)
        from utils.hashing import sha256_string

        assert_transformers_version_supported()
        attn_impl = "flash_attention_2" if cfg.backend == "flash_attn" else "eager"
        validate_backend_attn_pairing(cfg.backend, attn_impl)
        torch.manual_seed(cfg.seed)

        corpus, text_field, record_filter, slug = cfg.resolve_dataset()
        out_path = Path(cfg.out) if cfg.out else cfg.default_out(slug)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        articles = CorpusLoader(corpus, text_field=text_field,
                                record_filter=record_filter).load()

        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                  "float32": torch.float32}
        kv_dtype = dtypes[cfg.dtype]
        tok = AutoTokenizer.from_pretrained(cfg.model_path)
        indices = select_parity_articles(articles, tok, cfg.article_index,
                                         cfg.num_samples, cfg.prefill)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype=kv_dtype, attn_implementation=attn_impl,
            device_map="auto").eval()
        mc = model.config
        L = mc.num_hidden_layers
        H = mc.num_attention_heads
        Hk = getattr(mc, "num_key_value_heads", H)
        D = getattr(mc, "head_dim", None) or mc.hidden_size // H
        scale = D ** -0.5
        rope = getattr(model.model, "rotary_emb", None)
        if rope is None:
            raise RuntimeError("model has no model.rotary_emb; the collector "
                               "needs the RoPE module the cache re-rotates with")
        dev = next(model.parameters()).device
        ws, ns = cfg.window_size, cfg.num_sink_tokens
        T = cfg.gen + 1
        S_max = cfg.prefill + cfg.gen
        W = -(-(S_max - ns) // ws)

        meta: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "config": asdict(cfg),
            "dataset_resolved": {"corpus": corpus, "text_field": text_field,
                                 "record_filter": record_filter, "slug": slug},
            "model": {"num_layers": L, "num_heads": H, "num_kv_heads": Hk,
                      "head_dim": D, "dtype": cfg.dtype,
                      "attn_implementation": attn_impl},
            "axes": {"T": T, "S_max": S_max, "W": W, "window_size": ws,
                     "num_sink_tokens": ns, "prefill": cfg.prefill,
                     "gen": cfg.gen},
            "tier_codes": {"none": TIER_NONE, "fp": TIER_FP, "int2": TIER_Q,
                           "local": TIER_LOCAL, "evicted": TIER_EVICTED},
            "ladder": list(LADDER), "ladder_steps": list(LADDER_STEPS),
            "samples": [],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            from utils.env_capture import capture_environment
            meta["environment"] = capture_environment()
        except Exception:                                     # noqa: BLE001
            pass

        state: Dict[str, Any] = {"t": 0, "buf": None, "cache": None,
                                 "shadow": None}
        gate = GateRecorder()
        flash_decode.reset_stats()

        def on_layer_pre(layer: int) -> None:
            # Layer 0, before any layer's update(): every layer's pending
            # eviction ranks on what is recorded here.
            cache, t = state["cache"], state["t"]
            if layer != 0 or t == 0:
                return
            due = any(cache._policies[i].should_evict(cache._generation_step[i])
                      for i in range(L))
            if due:
                state["buf"].evict_step[t] = True
                snapshot_rank_signal(state["buf"], t, cache, L)

        def on_layer(layer: int, q: Tensor, o: Tensor) -> None:
            if q.shape[1] > 1:
                record_prefill_layer(state["buf"], layer, q, state["cache"],
                                     state["shadow"], ns, ws, scale)
            else:
                record_decode_layer(state["buf"], state["t"], layer, q[:, 0],
                                    o[:, 0], state["cache"], state["shadow"],
                                    gate, ns, ws, scale)
            for d in (gate.picks, gate.logmass, gate.active_ids):
                d.pop(layer, None)

        zpath = out_path.with_suffix(".zip")
        writer = ZipWriter(zpath)
        for si, art_idx in enumerate(indices):
            t0 = time.time()
            ids = tok.encode(articles[art_idx], return_tensors="pt",
                             add_special_tokens=True)[:, :cfg.prefill].to(dev)
            cache_cfg = WindowedCacheConfig(
                window_size=ws, num_sink_tokens=ns,
                local_window_size=cfg.local_window_size,
                cache_budget=cfg.cache_budget, quant_ratio=cfg.quant_ratio,
                quant_budget_mode=cfg.quant_budget_mode,
                quant_gate_ratio=cfg.gate_ratio, first_eviction_step=0,
                quant_card_bits=cfg.quant_card_bits,
                quant_promotion=cfg.quant_promotion,
                quant_promote_source=cfg.quant_promote_source)
            # max_tokens = the decode tokens the cache will receive: the budget
            # is sized on prefill + gen, as the quality runners size it.
            cache = WindowedCache(config=cache_cfg, prefill_len=cfg.prefill,
                                  model_config=mc, kv_dtype=kv_dtype,
                                  rope_module=rope, num_layers=L,
                                  max_tokens=cfg.gen)
            shadow = ShadowKV(L, Hk, S_max, D, kv_dtype, dev)
            buf = SampleBuffers(T, L, H, Hk, W, S_max, cfg.token_heads, dev)
            state.update(cache=cache, shadow=shadow, buf=buf, t=0)
            attach_shadow(cache, shadow)
            hooks = install_score_hooks(model, cache, cache_cfg)
            taps = ModelTaps(model, H, D, on_layer, on_layer_pre)
            gate.bind(cache)
            flash_decode.set_gate_observer(gate)
            fwd_kw = {"output_attentions": True} if cfg.backend == "eager" else {}
            try:
                with torch.no_grad():
                    inp, pos = ids, 0
                    for t in range(T):
                        state["t"] = t
                        gate.clear()
                        # Explicit, monotonic cache_position: derived, it would
                        # be the RETAINED key count and run backward after the
                        # first eviction (the cache refuses that).
                        out, pos = forward_at(model, inp, cache, pos, **fwd_kw)
                        lp = torch.log_softmax(out.logits[0, -1].float(), -1)
                        nxt = int(lp.argmax())
                        top = lp.topk(5)
                        buf.tokens.append(nxt)
                        buf.token_logprob.append(float(lp[nxt]))
                        buf.token_entropy.append(float(-(lp.exp() * lp).sum()))
                        buf.token_top5_ids.append(top.indices.tolist())
                        buf.token_top5_logprob.append(top.values.tolist())
                        inp = torch.tensor([[nxt]], device=dev)
                        if (t + 1) % 128 == 0:
                            log.info("sample %d/%d: step %d/%d", si + 1,
                                     len(indices), t + 1, T)
            finally:
                flash_decode.clear_gate_observer()
                taps.remove()
                hooks.remove()
                gate.bind(None)

            arrays = buf.to_numpy()
            arrays["prompt_ids"] = ids[0].cpu().numpy()
            r = cache.resolved
            meta["resolved_geometry"] = {
                "top_k_fp": int(r.top_k_fp), "N_q": int(r.N_q),
                "top_k_windows": int(r.top_k_windows),
                "local_windows": int(r.local_tokens // ws),
                "bytes_per_fp_window": int(r.bytes_per_fp_window),
                "bytes_per_q_window": int(r.bytes_per_q_window),
                "bytes_per_gate_card": int(r.bytes_per_gate_card),
                "quant_card_bits": dict(r.quant_card_bits._asdict())}
            meta["samples"].append({
                "file": f"sample_{si:03d}.npz", "article_index": int(art_idx),
                "article_sha": sha256_string(articles[art_idx]),
                "prompt_len": int(ids.shape[1]),
                "seconds": round(time.time() - t0, 1)})
            writer.add_sample(f"sample_{si:03d}.npz", arrays)
            del cache, shadow, buf, arrays
            state.update(cache=None, shadow=None, buf=None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("sample %d/%d done (%.0f s)", si + 1, len(indices),
                     meta["samples"][-1]["seconds"])

        expect = flash_decode.expect_gated(cfg.backend, cfg.quant_ratio,
                                           torch.cuda.is_available())
        meta["read_gate"] = flash_decode.log_gate_report(log, "collector", expect)
        meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        path = writer.close(meta)
        log.info("wrote %s", path)
        return path


# ---------------------------------------------------------------------------
# reading a zip back, and exporting it to the parity-npz pair
# ---------------------------------------------------------------------------


def load_zip(path: str | Path) -> Tuple[Dict[str, Any], List[Dict[str, np.ndarray]]]:
    """``(meta, [sample arrays...])``."""
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("meta.json"))
        samples = []
        for s in meta["samples"]:
            with np.load(io.BytesIO(zf.read(s["file"])), allow_pickle=False) as z:
                samples.append({k: z[k] for k in z.files})
    return meta, samples


def export_parity(zip_path: str | Path, out_dir: str | Path) -> Tuple[Path, Path]:
    """Write the ``parity_base`` / ``parity_ours`` npz pair the observation
    suite reads (``qevict_observations``), from one collector zip.

    Base: the full-KV masses of the same run (cumulative per head, per-step
    head-mean, per-step per head). Ours: the survivor axis (every window the
    cache holds, ascending), its tiers and the gate's pick on it.
    """
    meta, samples = load_zip(zip_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ax, cfg = meta["axes"], meta["config"]
    geom = meta.get("resolved_geometry", {})
    T = ax["T"]
    L = meta["model"]["num_layers"]
    Hk = meta["model"]["num_kv_heads"]

    cum, step_hm, step_h, toks = [], [], [], []
    ids_all, tier_all, gate_all, fired_all, evict_all = [], [], [], [], []
    for s in samples:
        fm = s["full_mass"].astype(np.float32)                   # [T,L,H,W]
        step_h.append(fm.astype(np.float16))
        step_hm.append(fm.mean(axis=2))
        cum.append(np.cumsum(fm, axis=0).astype(np.float16))
        toks.append(s["tokens"])
        tier = s["tier"]
        held = (tier == TIER_FP) | (tier == TIER_Q) | (tier == TIER_LOCAL)
        Wo = int(held.sum(-1).max())
        ids = np.full((T, L, Wo), -1, np.int64)
        tg = np.full((T, L, Wo), -1, np.int64)
        gr = np.zeros((T, L, Hk, Wo), bool)
        for t in range(T):
            for li in range(L):
                w = np.flatnonzero(held[t, li])
                ids[t, li, :w.size] = w
                code = tier[t, li, w]
                tg[t, li, :w.size] = np.where(code == TIER_Q, 1,
                                              np.where(code == TIER_LOCAL, 2, 0))
                gr[t, li, :, :w.size] = s["gate_open"][t, li][:, w]
        ids_all.append(ids)
        tier_all.append(tg)
        gate_all.append(gr)
        fired_all.append(s["gate_fired"])
        evict_all.append(s["evict_step"])

    def stack(xs, fill):
        m = max(x.shape[-1] for x in xs)
        return np.stack([np.concatenate(
            [x, np.full(x.shape[:-1] + (m - x.shape[-1],), fill, x.dtype)], -1)
            for x in xs])

    ident = {
        "seed": cfg["seed"], "dataset": meta["dataset_resolved"]["corpus"],
        "article_id": cfg["article_index"],
        "article_index_start": cfg["article_index"],
        "article_indices": [x["article_index"] for x in meta["samples"]],
        "article_shas": [x["article_sha"] for x in meta["samples"]],
        "article_sha": meta["samples"][0]["article_sha"],
        "num_samples": len(samples), "prefill_len": ax["prefill"],
        "gen_len": T, "window_size": ax["window_size"],
        "num_sink_tokens": ax["num_sink_tokens"],
        "local_window_size_resolved": geom.get("local_windows", 0) * ax["window_size"],
        "model_name": cfg["model_path"],
        "num_attention_heads": meta["model"]["num_heads"],
        "num_key_value_heads": Hk, "score_accum_dtype": "float32",
        "source_zip": str(zip_path),
    }
    base_meta = {**ident, "schema_version": "1.3", "mode": "parity_base",
                 "top_k_windows": geom.get("top_k_windows", 0),
                 "record_head_step_mass": True,
                 "note": "full-KV attention of the cache run's own queries "
                         "(observation_collector shadow), not a separate run"}
    ours_meta = {**ident, "schema_version": "1.3", "mode": "parity_ours",
                 "top_k_windows": geom.get("top_k_windows", 0),
                 "top_k_fp": geom.get("top_k_fp", 0), "N_q": geom.get("N_q", 0),
                 "quant_ratio": cfg["quant_ratio"],
                 "quant_gate_ratio": cfg["gate_ratio"],
                 "quant_budget_mode": cfg["quant_budget_mode"],
                 "quant_card_bits": geom.get("quant_card_bits"),
                 "quant_promotion": cfg["quant_promotion"],
                 "quant_promote_source": cfg["quant_promote_source"],
                 "cache_budget": cfg["cache_budget"],
                 "read_gate": meta.get("read_gate"), "gate_recorded": True,
                 "first_eviction_step": 0,
                 "bytes_per_q_window": geom.get("bytes_per_q_window"),
                 "bytes_per_gate_card": geom.get("bytes_per_gate_card")}
    base_p, ours_p = out / "parity_base.npz", out / "parity_ours.npz"
    tokens = np.stack(toks)
    np.savez_compressed(
        base_p, window_scores=np.stack(cum), step_window_scores=np.stack(step_hm),
        step_window_scores_heads=np.stack(step_h),
        eviction_step_mask=np.zeros((len(samples), T), bool),
        generated_tokens=tokens,
        top_window_indices=np.zeros((len(samples), T, L, 1), np.int64),
        metadata_json=np.array([json.dumps(base_meta, default=str)], dtype=object))
    np.savez_compressed(
        ours_p, all_window_ids=stack(ids_all, -1), all_window_tier=stack(tier_all, -1),
        gate_read=stack(gate_all, False), gate_fired=np.stack(fired_all),
        eviction_step_mask=np.stack(evict_all), generated_tokens=tokens,
        top_window_indices=np.zeros((len(samples), T, L, 1), np.int64),
        metadata_json=np.array([json.dumps(ours_meta, default=str)], dtype=object))
    return base_p, ours_p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _card_bits(v: Optional[str]):
    if v is None:
        return None
    return int(v) if v.isdigit() else v


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    ex = sub.add_parser("export", help="collector zip -> parity npz pair")
    ex.add_argument("--zip", required=True)
    ex.add_argument("--out-dir", required=True)

    for p in (ap,):
        p.add_argument("--model-path")
        p.add_argument("--dataset")
        p.add_argument("--window-size", type=int)
        p.add_argument("--cache-budget", type=float)
        p.add_argument("--gate-ratio", type=float)
        p.add_argument("--quant-ratio", type=float)
        p.add_argument("--prefill", type=int, default=512)
        p.add_argument("--gen", type=int, default=512)
        p.add_argument("--num-samples", type=int, default=8)
        p.add_argument("--article-index", type=int, default=0)
        p.add_argument("--sink", type=int, default=5)
        p.add_argument("--local", type=int, default=128)
        p.add_argument("--quant-budget-mode", default="bytes")
        p.add_argument("--card-bits", default=None)
        p.add_argument("--promotion", default="bidir", choices=("bidir", "oneway"))
        p.add_argument("--promote-source", default="dequant",
                       choices=("dequant", "original"))
        p.add_argument("--backend", default="flash_attn",
                       choices=("flash_attn", "eager"))
        p.add_argument("--dtype", default="float16")
        p.add_argument("--token-heads", action="store_true",
                       help="also store per-head per-token attention [T,L,H,S]")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--out", default=None, help="output .zip path")
        p.add_argument("--longbench-dir", default="")
        p.add_argument("--ruler-data", default="")
        p.add_argument("--text-field", default=None)
        p.add_argument("--record-filter", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        b, o = export_parity(a.zip, a.out_dir)
        print(f"base: {b}\nours: {o}\nthen: python -m "
              f"modules.evaluation.qevict_observations --base-npz {b} "
              f"--ours-npz {o} --output-dir {Path(a.out_dir) / 'observations'}")
        return
    missing = [n for n in ("model_path", "dataset", "window_size", "cache_budget",
                           "gate_ratio", "quant_ratio") if getattr(a, n) is None]
    if missing:
        ap.error("required: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    cfg = CollectorConfig(
        model_path=a.model_path, dataset=a.dataset, window_size=a.window_size,
        cache_budget=a.cache_budget, gate_ratio=a.gate_ratio,
        quant_ratio=a.quant_ratio, prefill=a.prefill, gen=a.gen,
        num_samples=a.num_samples, article_index=a.article_index,
        num_sink_tokens=a.sink, local_window_size=a.local,
        quant_budget_mode=a.quant_budget_mode, quant_card_bits=_card_bits(a.card_bits),
        quant_promotion=a.promotion, quant_promote_source=a.promote_source,
        backend=a.backend, dtype=a.dtype, token_heads=a.token_heads, seed=a.seed,
        out=a.out, longbench_dir=a.longbench_dir, ruler_data=a.ruler_data,
        text_field=a.text_field, record_filter=a.record_filter)
    path = ObservationCollector(cfg).run()
    print(path)


if __name__ == "__main__":
    main()
