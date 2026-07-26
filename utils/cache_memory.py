"""cache_memory — drop-in memory accounting for the two-tier windowed KV cache.

One file, two ways to use it.

**As a probe inside any test / eval.** After a run has populated the cache
(prefill + at least one eviction), hand the cache object to
:func:`report_cache_memory`::

    from utils.cache_memory import report_cache_memory
    ...
    out = model.generate(..., past_key_values=pkv)
    report_cache_memory(pkv, label="qasper bs=4 q=0.5")

It prints an exact byte breakdown — fp16 tier, int2 Q tier (codes + grid),
read memo, and bookkeeping — split into what is *semantically retained* (live)
vs. what is *reserved in memory* (allocated), plus the two compression ratios
the method exists to move: against an fp16 cache at the **same retention** (the
quantization win) and against a dense fp16 cache over the **full observed
context** (eviction + quantization together). It also snapshots CUDA / RSS.

It degrades gracefully: a plain ``DynamicCache`` / ``StaticCache`` is walked via
its ``key_cache`` / ``value_cache`` lists, so the *same* call measures your
baseline cache and the numbers line up apples-to-apples.

**As a standalone script.** Run it to build a model, drive a real
prefill + decode long enough to trigger eviction, and print the report::

    python -m utils.cache_memory --model meta-llama/Llama-3.1-8B \
        --backend eager --prefill 512 --gen 256 --batch-size 4 \
        --quant-ratio 0.5 --cache-budget 0.5

Introspection reads the cache's own tensors (``_states`` / ``_stores`` /
``resolved``), so it needs no GPU and adds no allocation of its own — the byte
figures are computed, not sampled, and are correct on CPU or CUDA alike.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

__all__ = [
    "CacheMemoryReport",
    "measure_cache_memory",
    "format_report",
    "report_cache_memory",
    "snapshot_device_memory",
]

_MB = 1024.0 * 1024.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _nbytes(t: Optional[Tensor]) -> int:
    """Bytes a tensor's storage carries (``0`` for ``None``)."""
    if t is None:
        return 0
    return t.numel() * t.element_size()


def _per_row_slot_bytes(t: Tensor) -> int:
    """Bytes one (row, slot) lane of a ``[B, N, ...]`` table tensor occupies.

    Used to charge the Q tier by *active* windows rather than by the full
    (dormant + free + margin) slot table — the live footprint, not the reserved
    one.
    """
    if t.dim() < 2:
        return 0
    lane = t[0, 0]
    return lane.numel() * t.element_size()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@dataclass
class CacheMemoryReport:
    """Byte accounting for one cache instance, summed over all layers and rows.

    ``*_live`` fields are the **semantically retained** content — the KV the
    cache currently represents. ``*_alloc`` fields are what is **reserved in
    memory** right now, which is larger when the fp buffer still holds the
    prefill-sized allocation, or when the Q slot table carries dormant / free
    slots (see ``state.py`` / ``slots.py``). Token counts are per row, taken
    from layer 0 (identical across layers at steady state); byte totals are
    summed exactly over every layer.
    """

    kind: str                       # "windowed" | "dynamic" | "static" | "unknown"
    num_layers: int
    batch_size: int
    device: str
    kv_dtype: str
    window_size: int

    # per-row token / window counts (layer 0, representative)
    fp_live_tokens: int
    fp_alloc_tokens: int
    q_active_windows: int
    q_active_tokens: int
    q_slots: int
    retained_tokens: int            # T_fp + T_q per row (effective seq length)
    observed_context_len: int       # max absolute position + 1 (full context seen)

    # byte totals — all layers, all rows
    fp_content_live: int
    fp_content_alloc: int
    q_content_live: int
    q_content_alloc: int
    bookkeeping_live: int           # positions + window scores + window ids + slot meta
    bookkeeping_alloc: int
    memo_bytes: int

    total_live: int
    total_alloc: int

    # analytic baselines (all layers, all rows, fp16-equivalent)
    fp16_equiv_retained: int        # same retained tokens, but all fp16 dense
    dense_full_context: int         # full observed context, all fp16 dense

    resolved: Dict[str, Any] = field(default_factory=dict)
    device_mem: Dict[str, Any] = field(default_factory=dict)
    per_layer: List[Dict[str, Any]] = field(default_factory=list)

    # -- derived ratios ------------------------------------------------------

    @property
    def compression_vs_fp16(self) -> Optional[float]:
        """fp16-at-same-retention ÷ actual live. The quantization win in isolation."""
        return self.fp16_equiv_retained / self.total_live if self.total_live else None

    @property
    def reduction_vs_full(self) -> Optional[float]:
        """Dense fp16 over the full observed context ÷ actual live.

        Eviction and quantization together, against the cache you would have
        kept had you never evicted or quantized.
        """
        return self.dense_full_context / self.total_live if self.total_live else None

    # -- the same two ratios with the read memo taken out --------------------
    #
    # The memo is a *derived* fp16 copy of the Q tier, not cache state: dropping
    # it costs recomputation, never correctness or a token. It is also charged
    # per row and, at B = 1 (where the auto rule turns it ON), it is by far the
    # largest line item — measured on a RULER example it was 16.3 of 21.4 MB,
    # which drags `compression_vs_fp16` to 0.91x, i.e. the report says the
    # two-tier cache is BIGGER than fp16.
    #
    # Both framings are honest and neither is the whole story, so publish the
    # pair rather than picking one: `*_excl_memo` is the cache-state figure a
    # compression claim should quote, `compression_vs_fp16` is the resident
    # footprint you actually pay at that batch size. For a headline memory
    # table, run the capture with quant_memoize_read: false and the two agree.

    @property
    def total_live_excl_memo(self) -> int:
        return self.total_live - self.memo_bytes

    @property
    def compression_vs_fp16_excl_memo(self) -> Optional[float]:
        t = self.total_live_excl_memo
        return self.fp16_equiv_retained / t if t else None

    @property
    def reduction_vs_full_excl_memo(self) -> Optional[float]:
        t = self.total_live_excl_memo
        return self.dense_full_context / t if t else None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["compression_vs_fp16"] = self.compression_vs_fp16
        d["reduction_vs_full"] = self.reduction_vs_full
        d["total_live_excl_memo"] = self.total_live_excl_memo
        d["compression_vs_fp16_excl_memo"] = self.compression_vs_fp16_excl_memo
        d["reduction_vs_full_excl_memo"] = self.reduction_vs_full_excl_memo
        return d


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def measure_cache_memory(
    cache: Any,
    full_context_len: Optional[int] = None,
) -> CacheMemoryReport:
    """Introspect a cache instance and return a :class:`CacheMemoryReport`.

    Parameters
    ----------
    cache
        A ``WindowedCache`` (either backend), or any HF cache exposing
        ``key_cache`` / ``value_cache`` lists (``DynamicCache``, ``StaticCache``).
    full_context_len
        Override for the full-context baseline (prefill + tokens generated). If
        ``None``, it is inferred from the largest absolute position the cache
        still holds — reliable, because the newest (local) tokens are never
        evicted.
    """
    if hasattr(cache, "_states"):
        return _measure_windowed(cache, full_context_len)
    if hasattr(cache, "key_cache"):
        return _measure_hf_dense(cache, full_context_len)
    raise TypeError(
        f"don't know how to measure {type(cache).__name__}: expected a "
        f"WindowedCache (has ._states) or an HF cache (has .key_cache)"
    )


def _measure_windowed(
    cache: Any, full_context_len: Optional[int]
) -> CacheMemoryReport:
    states = cache._states
    stores = getattr(cache, "_stores", [None] * len(states))
    resolved = cache.resolved
    ws = resolved.window_size

    # locate the first populated layer to read shapes / dtype / device from
    ref = next((s for s in states if s.key_states is not None), None)
    if ref is None:
        raise ValueError(
            "cache is empty — run a prefill through the model before measuring"
        )
    B, H_kv, _, D = ref.key_states.shape
    kv_dtype = ref.key_states.dtype
    elsize = ref.key_states.element_size()
    device = str(ref.key_states.device)
    dense_bytes_per_token = H_kv * D * elsize * 2  # K + V, per row per layer

    fp_content_live = fp_content_alloc = 0
    q_content_live = q_content_alloc = 0
    book_live = book_alloc = 0
    memo_bytes = 0
    retained_per_row_sum = 0        # Σ_layers (T_fp + T_q)  — for the fp16 baseline
    observed_ctx = 0
    per_layer: List[Dict[str, Any]] = []

    # representative per-row counts from layer 0
    fp_live_tokens = ref.key_states.shape[2]
    fp_alloc_tokens = ref.buffer_capacity
    q_active_windows = 0
    q_slots = 0

    for li, state in enumerate(states):
        if state.key_states is None:
            continue

        # -- fp16 tier: live views vs. reserved buffers --
        k_live = _nbytes(state.key_states) + _nbytes(state.value_states)
        k_alloc = _nbytes(state._key_buf) + _nbytes(state._val_buf)
        fp_content_live += k_live
        fp_content_alloc += k_alloc

        # -- bookkeeping: positions, window scores, window ids --
        b_live = (
            _nbytes(state.position_ids)
            + _nbytes(state.window_scores)
            + _nbytes(state.original_window_ids)
        )
        b_alloc = (
            _nbytes(state._pos_buf)
            + _nbytes(state.window_scores)
            + _nbytes(state.original_window_ids)
        )

        t_fp = state.key_states.shape[2]
        t_q = 0
        n_active = 0

        # -- int2 Q tier --
        store = stores[li] if li < len(stores) else None
        table = getattr(store, "table", None) if store is not None else None
        if table is not None:
            n_active = store.num_active_windows
            t_q = store.num_active_tokens
            if li == 0 or q_slots == 0:
                q_slots = table.n_slots
                q_active_windows = n_active

            kv_tensors = (
                table.key_codes, table.key_scale, table.key_zero,
                table.val_codes, table.val_scale, table.val_zero,
            )
            meta_tensors = (table.slot_pos, table.slot_wid, table.slot_active)

            # allocated = the whole table; live = only the active windows
            q_content_alloc += sum(_nbytes(t) for t in kv_tensors)
            per_slot_kv = sum(_per_row_slot_bytes(t) for t in kv_tensors)
            q_content_live += per_slot_kv * B * n_active

            b_alloc += sum(_nbytes(t) for t in meta_tensors)
            per_slot_meta = sum(_per_row_slot_bytes(t) for t in meta_tensors)
            b_live += per_slot_meta * B * n_active

            # read memo: whole Q tier dequantized to fp16, if enabled + warm
            rc = getattr(store, "_read_cache", None)
            if rc is not None:
                keys, vals, pos_flat = rc[1]
                memo_bytes += _nbytes(keys) + _nbytes(vals) + _nbytes(pos_flat)

        book_live += b_live
        book_alloc += b_alloc

        retained_row = t_fp + t_q
        retained_per_row_sum += retained_row

        # full-context estimate: newest tokens are never evicted, so the largest
        # surviving absolute position + 1 is the true context length.
        if state.position_ids is not None and state.position_ids.numel():
            observed_ctx = max(observed_ctx, int(state.position_ids.max().item()) + 1)

        per_layer.append({
            "layer": li,
            "t_fp": t_fp,
            "t_q": t_q,
            "q_active_windows": n_active,
            "fp_live_mb": k_live / _MB,
            "q_live_mb": (per_slot_kv * B * n_active) / _MB if table is not None else 0.0,
        })

    total_live = fp_content_live + q_content_live + book_live + memo_bytes
    total_alloc = fp_content_alloc + q_content_alloc + book_alloc + memo_bytes

    # analytic fp16 baselines (K+V dense, all layers, all rows)
    fp16_equiv = retained_per_row_sum * B * dense_bytes_per_token
    if full_context_len is not None:
        observed_ctx = max(observed_ctx, int(full_context_len))
    n_layers_pop = len(per_layer)
    dense_full = observed_ctx * B * dense_bytes_per_token * n_layers_pop

    return CacheMemoryReport(
        kind="windowed",
        num_layers=cache.num_layers,
        batch_size=B,
        device=device,
        kv_dtype=str(kv_dtype).replace("torch.", ""),
        window_size=ws,
        fp_live_tokens=fp_live_tokens,
        fp_alloc_tokens=fp_alloc_tokens,
        q_active_windows=q_active_windows,
        q_active_tokens=q_active_windows * ws,
        q_slots=q_slots,
        retained_tokens=fp_live_tokens + q_active_windows * ws,
        observed_context_len=observed_ctx,
        fp_content_live=fp_content_live,
        fp_content_alloc=fp_content_alloc,
        q_content_live=q_content_live,
        q_content_alloc=q_content_alloc,
        bookkeeping_live=book_live,
        bookkeeping_alloc=book_alloc,
        memo_bytes=memo_bytes,
        total_live=total_live,
        total_alloc=total_alloc,
        fp16_equiv_retained=fp16_equiv,
        dense_full_context=dense_full,
        resolved={
            "quant_ratio": resolved.quant_ratio,
            "cache_budget_tokens": resolved.total_budget_tokens,
            "window_size": ws,
            "num_sink_tokens": resolved.num_sink_tokens,
            "local_tokens": resolved.local_tokens,
            "top_k_windows": resolved.top_k_windows,
            "top_k_fp": resolved.top_k_fp,
            "N_q": resolved.N_q,
            "quant_memoize_read": bool(memo_bytes),
        },
        device_mem=snapshot_device_memory(ref.key_states.device),
        per_layer=per_layer,
    )


def _measure_hf_dense(
    cache: Any, full_context_len: Optional[int]
) -> CacheMemoryReport:
    """Walk a ``DynamicCache`` / ``StaticCache`` for an apples-to-apples baseline."""
    keys = [k for k in cache.key_cache if k is not None and k.numel()]
    vals = [v for v in cache.value_cache if v is not None and v.numel()]
    if not keys:
        raise ValueError("HF cache is empty — run a prefill before measuring")

    ref = keys[0]
    B, H_kv, T, D = ref.shape
    elsize = ref.element_size()
    dense_bytes_per_token = H_kv * D * elsize * 2

    content = sum(_nbytes(k) for k in keys) + sum(_nbytes(v) for v in vals)
    n_layers = len(keys)
    # StaticCache preallocates to max; its "live" length is a prefix. DynamicCache
    # is exact. Either way, alloc == what the tensors carry; live == what's used.
    seq_len = T
    observed_ctx = int(full_context_len) if full_context_len is not None else seq_len
    retained_sum = seq_len * n_layers
    fp16_equiv = retained_sum * B * dense_bytes_per_token
    dense_full = observed_ctx * B * dense_bytes_per_token * n_layers

    kind = type(cache).__name__.replace("Cache", "").lower() or "dense"
    return CacheMemoryReport(
        kind=kind,
        num_layers=getattr(cache, "num_hidden_layers", n_layers),
        batch_size=B,
        device=str(ref.device),
        kv_dtype=str(ref.dtype).replace("torch.", ""),
        window_size=0,
        fp_live_tokens=seq_len,
        fp_alloc_tokens=seq_len,
        q_active_windows=0,
        q_active_tokens=0,
        q_slots=0,
        retained_tokens=seq_len,
        observed_context_len=observed_ctx,
        fp_content_live=content,
        fp_content_alloc=content,
        q_content_live=0,
        q_content_alloc=0,
        bookkeeping_live=0,
        bookkeeping_alloc=0,
        memo_bytes=0,
        total_live=content,
        total_alloc=content,
        fp16_equiv_retained=fp16_equiv,
        dense_full_context=dense_full,
        resolved={"quant_ratio": 0.0},
        device_mem=snapshot_device_memory(ref.device),
        per_layer=[],
    )


# ---------------------------------------------------------------------------
# device / process snapshot
# ---------------------------------------------------------------------------


def snapshot_device_memory(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """CUDA allocator stats (if on GPU) and process RSS (best effort).

    The cache byte figures above are computed exactly and stand on their own;
    this is the surrounding context — total process footprint, peak allocator
    high-water mark — that a test set usually wants alongside them.
    """
    out: Dict[str, Any] = {}
    is_cuda = device is not None and torch.device(device).type == "cuda"
    if is_cuda and torch.cuda.is_available():
        idx = torch.device(device).index or 0
        out["cuda_allocated_mb"] = torch.cuda.memory_allocated(idx) / _MB
        out["cuda_reserved_mb"] = torch.cuda.memory_reserved(idx) / _MB
        out["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(idx) / _MB
    try:
        import psutil  # optional

        out["process_rss_mb"] = psutil.Process().memory_info().rss / _MB
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def _mb(n: int) -> str:
    """Human-readable bytes: scales B / KB / MB / GB so small configs don't
    all collapse to ``0.00 MB`` and 8B-model caches don't overflow the column."""
    n = float(n)
    for unit, scale in (("GB", _MB * 1024), ("MB", _MB), ("KB", 1024.0)):
        if n >= scale:
            return f"{n / scale:,.2f} {unit}"
    return f"{n:,.0f} B"


def format_report(report: CacheMemoryReport, label: Optional[str] = None) -> str:
    r = report
    lines: List[str] = []
    title = "KV cache memory"
    if label:
        title += f" - {label}"
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)
    lines.append(
        f"kind={r.kind}  layers={r.num_layers}  batch={r.batch_size}  "
        f"device={r.device}  kv_dtype={r.kv_dtype}"
    )
    if r.kind == "windowed":
        rc = r.resolved
        lines.append(
            f"config: q={rc['quant_ratio']}  ws={rc['window_size']}  "
            f"sink={rc['num_sink_tokens']}  local={rc['local_tokens']}  "
            f"top_k_fp={rc['top_k_fp']}  N_q={rc['N_q']}  "
            f"memo={'on' if rc['quant_memoize_read'] else 'off'}"
        )
    lines.append("-" * 72)
    lines.append(f"{'tier':<26}{'live':>20}{'allocated':>22}")
    lines.append(f"{'-'*26}{'-'*20:>20}{'-'*22:>22}")
    lines.append(
        f"{'fp16 KV':<26}{_mb(r.fp_content_live):>20}{_mb(r.fp_content_alloc):>22}"
    )
    if r.q_content_alloc or r.kind == "windowed":
        lines.append(
            f"{'int2 Q (codes+grid)':<26}"
            f"{_mb(r.q_content_live):>20}{_mb(r.q_content_alloc):>22}"
        )
    lines.append(
        f"{'bookkeeping':<26}"
        f"{_mb(r.bookkeeping_live):>20}{_mb(r.bookkeeping_alloc):>22}"
    )
    if r.memo_bytes:
        lines.append(
            f"{'read memo (fp16)':<26}{_mb(r.memo_bytes):>20}{_mb(r.memo_bytes):>22}"
        )
    lines.append(f"{'-'*26}{'-'*20:>20}{'-'*22:>22}")
    lines.append(
        f"{'TOTAL':<26}{_mb(r.total_live):>20}{_mb(r.total_alloc):>22}"
    )
    lines.append("-" * 72)

    # token accounting (per row)
    lines.append(
        f"retained/row: T_fp={r.fp_live_tokens} + T_q={r.q_active_tokens} "
        f"({r.q_active_windows} win x ws={r.window_size}) = {r.retained_tokens} tok"
    )
    if r.fp_alloc_tokens != r.fp_live_tokens:
        lines.append(
            f"  (fp buffer still reserves {r.fp_alloc_tokens} tok/row - "
            f"prefill-sized, released at first eviction)"
        )
    lines.append(f"observed context: {r.observed_context_len} tok/row")

    # baselines
    c1 = r.compression_vs_fp16
    c2 = r.reduction_vs_full
    lines.append("-" * 72)
    lines.append(
        f"vs fp16 at same retention: {_mb(r.fp16_equiv_retained)} live  ->  "
        f"{c1:.2f}x compression" if c1 else "vs fp16: n/a"
    )
    lines.append(
        f"vs dense fp16 @ full context: {_mb(r.dense_full_context)}  ->  "
        f"{c2:.2f}x smaller" if c2 else "vs full context: n/a"
    )
    if r.memo_bytes:
        # Say it out loud: at B = 1 the memo is usually the biggest line item,
        # and both ratios above are computed WITH it. Quoting only the first
        # number here would understate the method; quoting only the second
        # would hide what the run actually occupies. Print both.
        c1x = r.compression_vs_fp16_excl_memo
        c2x = r.reduction_vs_full_excl_memo
        share = 100.0 * r.memo_bytes / r.total_live if r.total_live else 0.0
        lines.append("-" * 72)
        lines.append(
            f"NB the read memo is {share:.0f}% of TOTAL live. It is a derived "
            f"fp16 copy of the Q tier,"
        )
        lines.append(
            "   not cache state - recomputable, and OFF by default at B > 1. "
            "Excluding it:"
        )
        lines.append(
            f"   cache state {_mb(r.total_live_excl_memo)} live  ->  "
            f"{c1x:.2f}x vs fp16 at same retention, "
            f"{c2x:.2f}x vs full context" if c1x and c2x else "   n/a"
        )
        lines.append(
            "   For a headline memory table run with quant_memoize_read: false, "
            "where the two agree."
        )
    if r.device_mem:
        parts = [f"{k}={v:,.1f}" for k, v in r.device_mem.items()]
        lines.append("-" * 72)
        lines.append("process/device: " + "  ".join(parts))
    if r.kind == "windowed" and r.q_slots and r.q_active_windows == 0:
        lines.append("-" * 72)
        lines.append(
            "NOTE: Q tier is empty (no eviction has run yet). Increase --gen or "
            "lower --cache-budget so the cache compacts and the int2 tier fills."
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def report_cache_memory(
    cache: Any,
    label: Optional[str] = None,
    full_context_len: Optional[int] = None,
    as_json: bool = False,
    file=sys.stdout,
) -> CacheMemoryReport:
    """Measure *cache*, print the report, and return it. The one-call entry point."""
    report = measure_cache_memory(cache, full_context_len=full_context_len)
    if as_json:
        print(json.dumps(report.to_dict(), indent=2), file=file)
    else:
        print(format_report(report, label=label), file=file)
    return report


# ---------------------------------------------------------------------------
# standalone CLI — build a model, drive prefill + decode, print the report
# ---------------------------------------------------------------------------


def _build_and_run(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM

    from utils.cache_factory import (
        assert_transformers_version_supported,
        get_cache_classes,
    )

    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtypes[args.dtype]
    attn_impl = "flash_attention_2" if args.backend == "flash_attn" else "eager"

    assert_transformers_version_supported()
    WC, WCC, install_hooks = get_cache_classes(args.backend)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, attn_implementation=attn_impl,
        device_map="auto" if args.device == "auto" else None,
    )
    if args.device != "auto":
        model = model.to(args.device)
    model.eval()

    device = model.device
    B = max(1, args.batch_size)
    # Bound token ids by the model's real vocab so this works for tiny test
    # models as well as production ones. The content is irrelevant — memory
    # accounting depends on shapes, not values.
    vocab = int(getattr(model.config, "vocab_size", 30000))
    input_ids = torch.randint(0, vocab, (B, args.prefill), device=device)

    cfg = WCC(
        window_size=args.window_size,
        num_sink_tokens=args.num_sink,
        local_window_size=args.local,
        cache_budget=args.cache_budget,
        quant_ratio=args.quant_ratio,
        quant_memoize_read=args.memoize,
    )

    rope = None
    for nm, mod in model.named_modules():
        if "rotary" in nm.lower() or "rope" in nm.lower():
            rope = mod
            break
    if rope is None:
        for _, mod in model.named_modules():
            if hasattr(mod, "rotary_emb"):
                rope = mod.rotary_emb
                break

    pkv = WC(
        config=cfg, prefill_len=args.prefill, model_config=model.config,
        kv_dtype=torch_dtype, rope_module=rope,
        num_layers=model.config.num_hidden_layers, max_tokens=args.gen,
    )
    # The eager scorer reads attn_weights from the forward output, so it needs
    # output_attentions=True; flash_attn derives scores from its own softmax
    # stats and must NOT be given it (see the runners). Without scores flowing,
    # eviction never fires and the Q tier stays empty — so this is what makes
    # the tool show a *compacted* two-tier cache rather than a full one.
    gen_kwargs = {"output_attentions": True} if args.backend == "eager" else {}
    hooks = install_hooks(model, pkv, cfg)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=pkv,
                        use_cache=True, return_dict=True, **gen_kwargs)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(args.gen - 1):
                out = model(input_ids=next_tok, past_key_values=out.past_key_values,
                            use_cache=True, return_dict=True, **gen_kwargs)
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    finally:
        if hooks is not None:
            hooks.remove()

    full_ctx = args.prefill + args.gen
    report_cache_memory(
        pkv,
        label=f"{args.model} q={args.quant_ratio} bs={B} "
              f"prefill={args.prefill} gen={args.gen}",
        full_context_len=full_ctx,
        as_json=args.json,
    )


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Reveal two-tier windowed KV-cache memory usage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="sshleifer/tiny-gpt2",
                   help="HF model id (tiny default so the script runs anywhere)")
    p.add_argument("--backend", choices=["eager", "flash_attn"], default="eager")
    p.add_argument("--prefill", type=int, default=512)
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--quant-ratio", type=float, default=0.5)
    p.add_argument("--cache-budget", type=float, default=0.5)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--num-sink", type=int, default=4)
    p.add_argument("--local", type=float, default=0.25,
                   help="local window: float=fraction of budget, or pass an int")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                   default="float32")
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', 'cuda', 'cuda:0', ...")
    p.add_argument("--memoize", type=lambda s: {"true": True, "false": False}.get(
        s.lower(), None), default=None,
        help="force read memo on/off; default None = auto (on at B=1)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = p.parse_args(argv)

    # allow integer --local
    if args.local is not None and float(args.local).is_integer() and args.local > 1:
        args.local = int(args.local)

    _build_and_run(args)


if __name__ == "__main__":
    main()
