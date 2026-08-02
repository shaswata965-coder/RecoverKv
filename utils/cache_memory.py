
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
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
    "MemoryProbe",
    "PeakMemoryReport",
    "PhasePeak",
    "format_peak_report",
    "host_rss_bytes",
    "host_peak_rss_bytes",
]

_MB = 1024.0 * 1024.0


def _nbytes(t: Optional[Tensor]) -> int:
    if t is None:
        return 0
    return t.numel() * t.element_size()


def _per_row_slot_bytes(t: Tensor) -> int:
    if t.dim() < 2:
        return 0
    lane = t[0, 0]
    return lane.numel() * t.element_size()


@dataclass
class CacheMemoryReport:

    kind: str
    num_layers: int
    batch_size: int
    device: str
    kv_dtype: str
    window_size: int

    fp_live_tokens: int
    fp_alloc_tokens: int
    q_active_windows: int
    q_active_tokens: int
    q_slots: int
    retained_tokens: int
    observed_context_len: int

    fp_content_live: int
    fp_content_alloc: int
    q_content_live: int
    q_content_alloc: int
    bookkeeping_live: int
    bookkeeping_alloc: int
    memo_bytes: int

    total_live: int
    total_alloc: int

    fp16_equiv_retained: int
    dense_full_context: int

    resolved: Dict[str, Any] = field(default_factory=dict)
    device_mem: Dict[str, Any] = field(default_factory=dict)
    per_layer: List[Dict[str, Any]] = field(default_factory=list)


    @property
    def compression_vs_fp16(self) -> Optional[float]:
        return self.fp16_equiv_retained / self.total_live if self.total_live else None

    @property
    def reduction_vs_full(self) -> Optional[float]:
        return self.dense_full_context / self.total_live if self.total_live else None


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


def measure_cache_memory(
    cache: Any,
    full_context_len: Optional[int] = None,
) -> CacheMemoryReport:
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

    ref = next((s for s in states if s.key_states is not None), None)
    if ref is None:
        raise ValueError(
            "cache is empty — run a prefill through the model before measuring"
        )
    B, H_kv, _, D = ref.key_states.shape
    kv_dtype = ref.key_states.dtype
    elsize = ref.key_states.element_size()
    device = str(ref.key_states.device)
    dense_bytes_per_token = H_kv * D * elsize * 2

    fp_content_live = fp_content_alloc = 0
    q_content_live = q_content_alloc = 0
    book_live = book_alloc = 0
    memo_bytes = 0
    retained_per_row_sum = 0
    observed_ctx = 0
    per_layer: List[Dict[str, Any]] = []

    fp_live_tokens = ref.key_states.shape[2]
    fp_alloc_tokens = ref.buffer_capacity
    q_active_windows = 0
    q_slots = 0

    for li, state in enumerate(states):
        if state.key_states is None:
            continue

        k_live = _nbytes(state.key_states) + _nbytes(state.value_states)
        k_alloc = _nbytes(state._key_buf) + _nbytes(state._val_buf)
        fp_content_live += k_live
        fp_content_alloc += k_alloc

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

            q_content_alloc += sum(_nbytes(t) for t in kv_tensors)
            per_slot_kv = sum(_per_row_slot_bytes(t) for t in kv_tensors)
            q_content_live += per_slot_kv * B * n_active

            b_alloc += sum(_nbytes(t) for t in meta_tensors)
            per_slot_meta = sum(_per_row_slot_bytes(t) for t in meta_tensors)
            b_live += per_slot_meta * B * n_active

            rc = getattr(store, "_read_cache", None)
            if rc is not None:
                keys, vals, pos_flat = rc[1]
                memo_bytes += _nbytes(keys) + _nbytes(vals) + _nbytes(pos_flat)

        book_live += b_live
        book_alloc += b_alloc

        retained_row = t_fp + t_q
        retained_per_row_sum += retained_row

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


def host_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    return _os_rss(peak=False)


def host_peak_rss_bytes() -> int:
    return _os_rss(peak=True)


def _os_rss(peak: bool) -> int:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
            get_info.restype = wintypes.BOOL

            counters = _PMC()
            counters.cb = ctypes.sizeof(_PMC)
            if not get_info(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            ):
                return 0
            return int(
                counters.PeakWorkingSetSize if peak else counters.WorkingSetSize
            )
        except Exception:
            return 0
    try:
        if peak:
            import resource

            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(kb if sys.platform == "darwin" else kb * 1024)
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def snapshot_device_memory(device: Optional[torch.device] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    is_cuda = device is not None and torch.device(device).type == "cuda"
    if is_cuda and torch.cuda.is_available():
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        out["cuda_allocated_mb"] = torch.cuda.memory_allocated(idx) / _MB
        out["cuda_reserved_mb"] = torch.cuda.memory_reserved(idx) / _MB
        out["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(idx) / _MB
        out["cuda_max_reserved_mb"] = torch.cuda.max_memory_reserved(idx) / _MB
        try:
            free_b, total_b = torch.cuda.mem_get_info(idx)
            out["cuda_device_used_mb"] = (total_b - free_b) / _MB
            out["cuda_device_total_mb"] = total_b / _MB
        except Exception:
            pass
        try:
            stats = torch.cuda.memory_stats(idx)
            out["cuda_num_ooms"] = float(stats.get("num_ooms", 0))
            out["cuda_alloc_retries"] = float(stats.get("num_alloc_retries", 0))
        except Exception:
            pass
    rss = host_rss_bytes()
    if rss:
        out["process_rss_mb"] = rss / _MB
    peak_rss = host_peak_rss_bytes()
    if peak_rss:
        out["process_peak_rss_mb"] = peak_rss / _MB
    return out


@dataclass
class PhasePeak:

    name: str
    seconds: float
    samples: int
    torch_alloc_peak: int
    torch_reserved_peak: int
    device_used_peak: int
    host_rss_peak: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PeakMemoryReport:

    label: Optional[str]
    device: str
    device_name: str
    duration_s: float
    samples: int
    poll_interval_s: Optional[float]
    hooked: bool

    torch_alloc_peak: int
    torch_alloc_end: int
    torch_reserved_peak: int
    torch_reserved_end: int
    device_used_peak: int
    device_used_end: int
    device_total: int
    host_rss_peak: int
    host_rss_end: int
    num_ooms: int
    num_alloc_retries: int

    phases: List[PhasePeak] = field(default_factory=list)


    @property
    def fragmentation_peak(self) -> int:
        return max(self.torch_reserved_peak - self.torch_alloc_peak, 0)

    @property
    def device_headroom(self) -> int:
        if not self.device_total:
            return 0
        return max(self.device_total - self.device_used_peak, 0)

    @property
    def device_utilization(self) -> Optional[float]:
        if not self.device_total:
            return None
        return self.device_used_peak / self.device_total

    @property
    def peak_phase(self) -> Optional[str]:
        if not self.phases:
            return None
        key = (
            (lambda p: p.device_used_peak)
            if self.device_total
            else (lambda p: max(p.torch_alloc_peak, p.host_rss_peak))
        )
        return max(self.phases, key=key).name

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["phases"] = [p.to_dict() for p in self.phases]
        d["fragmentation_peak"] = self.fragmentation_peak
        d["device_headroom"] = self.device_headroom
        d["device_utilization"] = self.device_utilization
        d["peak_phase"] = self.peak_phase
        return d


class MemoryProbe:

    def __init__(
        self,
        label: Optional[str] = None,
        device: Optional[torch.device] = None,
        model: Optional[torch.nn.Module] = None,
        poll_interval_s: Optional[float] = 0.05,
    ) -> None:
        if device is None:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = torch.device(device)
        self.label = label
        self.poll_interval_s = poll_interval_s
        self._model = model

        self._is_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self._idx = (
            (self.device.index if self.device.index is not None
             else torch.cuda.current_device())
            if self._is_cuda else None
        )

        self._carried_alloc = 0
        self._carried_reserved = 0
        self._device_used_peak = 0
        self._host_rss_peak = 0
        self._samples = 0

        self._phases: List[PhasePeak] = []
        self._phase_open: Optional[tuple] = None
        self._phase_device_start = 0
        self._phase_host_start = 0
        self._t0: Optional[float] = None
        self._t1: Optional[float] = None
        self._ooms_at_start = 0
        self._retries_at_start = 0

        self._hook = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None


    def start(self) -> "MemoryProbe":
        if self._is_cuda:
            torch.cuda.synchronize(self._idx)
            torch.cuda.reset_peak_memory_stats(self._idx)
            stats = torch.cuda.memory_stats(self._idx)
            self._ooms_at_start = int(stats.get("num_ooms", 0))
            self._retries_at_start = int(stats.get("num_alloc_retries", 0))
        self._t0 = time.perf_counter()
        self.sample()
        if self._model is not None:
            self.attach(self._model)
        if self.poll_interval_s:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._poll, name="MemoryProbe", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> "MemoryProbe":
        if self._t1 is not None:
            return self
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._end_phase()
        self.detach()
        if self._is_cuda:
            torch.cuda.synchronize(self._idx)
        self.sample()
        self._t1 = time.perf_counter()
        return self

    def __enter__(self) -> "MemoryProbe":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


    def attach(self, module: torch.nn.Module):
        self.detach()

        def _hook(_module, _inp, _out):
            self.sample()

        self._hook = module.register_forward_hook(_hook)
        return self._hook

    def detach(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


    def sample(self) -> None:
        self._samples += 1
        if self._is_cuda:
            try:
                free_b, total_b = torch.cuda.mem_get_info(self._idx)
                self._device_used_peak = max(
                    self._device_used_peak, total_b - free_b
                )
            except Exception:
                pass
        rss = host_rss_bytes()
        if rss:
            self._host_rss_peak = max(self._host_rss_peak, rss)

    def _poll(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            self.sample()


    class _Phase:
        def __init__(self, probe: "MemoryProbe", name: str) -> None:
            self._probe, self._name = probe, name

        def __enter__(self):
            self._probe._begin_phase(self._name)
            return self._probe

        def __exit__(self, *exc):
            self._probe._end_phase()

    def phase(self, name: str) -> "MemoryProbe._Phase":
        return MemoryProbe._Phase(self, name)

    def _begin_phase(self, name: str) -> None:
        self._end_phase()
        self._fold_allocator_peaks()
        if self._is_cuda:
            torch.cuda.synchronize(self._idx)
            torch.cuda.reset_peak_memory_stats(self._idx)
        self._phase_open = (name, time.perf_counter(), self._samples)
        self._phase_device_start = self._device_used_peak
        self._phase_host_start = self._host_rss_peak
        self._device_used_peak = 0
        self._host_rss_peak = 0
        self.sample()

    def _end_phase(self) -> None:
        open_phase = getattr(self, "_phase_open", None)
        if open_phase is None:
            return
        name, t_start, s_start = open_phase
        self._phase_open = None
        if self._is_cuda:
            torch.cuda.synchronize(self._idx)
        self.sample()
        alloc_peak = self._allocator_peak()
        reserved_peak = self._allocator_peak(reserved=True)
        self._phases.append(
            PhasePeak(
                name=name,
                seconds=time.perf_counter() - t_start,
                samples=self._samples - s_start,
                torch_alloc_peak=alloc_peak,
                torch_reserved_peak=reserved_peak,
                device_used_peak=self._device_used_peak,
                host_rss_peak=self._host_rss_peak,
            )
        )
        self._fold_allocator_peaks()
        self._device_used_peak = max(
            self._device_used_peak, self._phase_device_start
        )
        self._host_rss_peak = max(self._host_rss_peak, self._phase_host_start)


    def _allocator_peak(self, reserved: bool = False) -> int:
        if not self._is_cuda:
            return 0
        return int(
            torch.cuda.max_memory_reserved(self._idx)
            if reserved
            else torch.cuda.max_memory_allocated(self._idx)
        )

    def _fold_allocator_peaks(self) -> None:
        self._carried_alloc = max(self._carried_alloc, self._allocator_peak())
        self._carried_reserved = max(
            self._carried_reserved, self._allocator_peak(reserved=True)
        )


    def report(self) -> PeakMemoryReport:
        self._fold_allocator_peaks()
        end = snapshot_device_memory(self.device)
        total_b = int(end.get("cuda_device_total_mb", 0) * _MB)
        used_end = int(end.get("cuda_device_used_mb", 0) * _MB)
        ooms = int(end.get("cuda_num_ooms", 0)) - self._ooms_at_start
        retries = int(end.get("cuda_alloc_retries", 0)) - self._retries_at_start
        t1 = self._t1 if self._t1 is not None else time.perf_counter()
        return PeakMemoryReport(
            label=self.label,
            device=str(self.device),
            device_name=(
                torch.cuda.get_device_name(self._idx) if self._is_cuda else "cpu"
            ),
            duration_s=t1 - (self._t0 if self._t0 is not None else t1),
            samples=self._samples,
            poll_interval_s=self.poll_interval_s,
            hooked=self._model is not None,
            torch_alloc_peak=self._carried_alloc,
            torch_alloc_end=int(end.get("cuda_allocated_mb", 0) * _MB),
            torch_reserved_peak=self._carried_reserved,
            torch_reserved_end=int(end.get("cuda_reserved_mb", 0) * _MB),
            device_used_peak=max(self._device_used_peak, used_end),
            device_used_end=used_end,
            device_total=total_b,
            host_rss_peak=max(self._host_rss_peak, host_peak_rss_bytes()),
            host_rss_end=host_rss_bytes(),
            num_ooms=max(ooms, 0),
            num_alloc_retries=max(retries, 0),
            phases=list(self._phases),
        )

    def format(self) -> str:
        return format_peak_report(self.report())

    def to_dict(self) -> Dict[str, Any]:
        return self.report().to_dict()


def _mb(n: int) -> str:
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


def format_peak_report(report: PeakMemoryReport) -> str:
    r = report
    lines: List[str] = []
    title = "Peak memory"
    if r.label:
        title += f" - {r.label}"
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)
    lines.append(
        f"device={r.device} ({r.device_name})  elapsed={r.duration_s:.2f}s  "
        f"samples={r.samples}"
        + (f"  poll={r.poll_interval_s}s" if r.poll_interval_s else "")
        + ("  hook=on" if r.hooked else "")
    )
    lines.append("-" * 72)
    lines.append(f"{'':<26}{'peak':>20}{'at end':>22}")
    lines.append(f"{'-'*26}{'-'*20:>20}{'-'*22:>22}")
    lines.append(
        f"{'torch allocated':<26}{_mb(r.torch_alloc_peak):>20}"
        f"{_mb(r.torch_alloc_end):>22}"
    )
    lines.append(
        f"{'torch reserved':<26}{_mb(r.torch_reserved_peak):>20}"
        f"{_mb(r.torch_reserved_end):>22}"
    )
    if r.device_total:
        lines.append(
            f"{'GPU device used':<26}{_mb(r.device_used_peak):>20}"
            f"{_mb(r.device_used_end):>22}"
        )
    lines.append(
        f"{'host RSS':<26}{_mb(r.host_rss_peak):>20}{_mb(r.host_rss_end):>22}"
    )
    lines.append("-" * 72)

    if r.device_total:
        util = r.device_utilization or 0.0
        lines.append(
            f"GPU {_mb(r.device_used_peak)} / {_mb(r.device_total)} at peak "
            f"({util*100:.1f}%)  headroom {_mb(r.device_headroom)}"
        )
        if util >= 0.95:
            lines.append(
                "  -> at the ceiling: the next rung of the batch ladder will OOM."
            )
        elif util <= 0.60:
            lines.append(
                f"  -> {(1.0/util if util else 0):.1f}x headroom: batch size is NOT "
                f"memory-bound here; raise it before quoting a max-B."
            )
    lines.append(
        f"fragmentation at peak (reserved - allocated): "
        f"{_mb(r.fragmentation_peak)}"
    )
    if r.num_ooms or r.num_alloc_retries:
        lines.append(
            f"allocator pressure: {r.num_ooms} OOM(s), "
            f"{r.num_alloc_retries} retr(y/ies) - a retry means the allocator had "
            f"to flush and re-request; the run survived but is at its limit."
        )

    if r.phases:
        lines.append("-" * 72)
        lines.append(
            f"{'phase':<14}{'secs':>8}{'alloc peak':>16}{'GPU peak':>16}"
            f"{'RSS peak':>16}"
        )
        for p in r.phases:
            lines.append(
                f"{p.name:<14}{p.seconds:>8.2f}"
                f"{_mb(p.torch_alloc_peak):>16}"
                f"{_mb(p.device_used_peak):>16}"
                f"{_mb(p.host_rss_peak):>16}"
            )
        peak_phase = r.peak_phase
        if peak_phase and not r.device_total:
            lines.append(
                f"peak phase: {peak_phase} (by host RSS - CPU run, so this is "
                f"NOT a max-B signal)"
            )
        elif peak_phase:
            lines.append(f"peak phase: {peak_phase}")
            if peak_phase == "prefill":
                lines.append(
                    "  -> the UN-EVICTED prompt is the binding constraint, not the "
                    "steady state."
                )
                lines.append(
                    "     Lowering cache_budget / raising quant_ratio will not "
                    "raise max-B at this shape;"
                )
                lines.append(
                    "     only a shorter prompt, more decode, or chunked prefill "
                    "will."
                )
            else:
                lines.append(
                    "  -> the steady state binds, so cache_budget maps ~1:1 onto "
                    "max-B. This is the regime"
                )
                lines.append(
                    "     where compression buys batch capacity."
                )
    lines.append("=" * 72)
    return "\n".join(lines)


def report_cache_memory(
    cache: Any,
    label: Optional[str] = None,
    full_context_len: Optional[int] = None,
    as_json: bool = False,
    file=sys.stdout,
    probe: Optional["MemoryProbe"] = None,
) -> CacheMemoryReport:
    report = measure_cache_memory(cache, full_context_len=full_context_len)
    if as_json:
        out = report.to_dict()
        if probe is not None:
            out["peak"] = probe.to_dict()
        print(json.dumps(out, indent=2), file=file)
    else:
        print(format_report(report, label=label), file=file)
        if probe is not None:
            print(format_peak_report(probe.report()), file=file)
    return report


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
    vocab = int(getattr(model.config, "vocab_size", 30000))
    input_ids = torch.randint(0, vocab, (B, args.prefill), device=device)

    cfg = WCC(
        window_size=args.window_size,
        num_sink_tokens=args.num_sink,
        local_window_size=args.local,
        cache_budget=args.cache_budget,
        quant_ratio=args.quant_ratio,
        quant_memoize_read=args.memoize,
        quant_promotion=not args.no_quant_promotion,
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
    gen_kwargs = {"output_attentions": True} if args.backend == "eager" else {}
    hooks = install_hooks(model, pkv, cfg)
    label = (f"{args.model} q={args.quant_ratio} bs={B} "
             f"prefill={args.prefill} gen={args.gen}")
    probe = MemoryProbe(label=label, device=device,
                        poll_interval_s=None if args.no_poll else args.poll)
    try:
        with torch.no_grad(), probe:
            with probe.phase("prefill"):
                out = model(input_ids=input_ids, past_key_values=pkv,
                            use_cache=True, return_dict=True, **gen_kwargs)
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            with probe.phase("decode"):
                for _ in range(args.gen - 1):
                    out = model(input_ids=next_tok,
                                past_key_values=out.past_key_values,
                                use_cache=True, return_dict=True, **gen_kwargs)
                    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    finally:
        if hooks is not None:
            hooks.remove()

    full_ctx = args.prefill + args.gen
    report_cache_memory(
        pkv,
        label=label,
        full_context_len=full_ctx,
        as_json=args.json,
        probe=probe,
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
    p.add_argument("--no-quant-promotion", action="store_true",
                   help="sticky Q: a demoted window never returns to fp, it "
                        "stays int2 until it is evicted")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.add_argument("--poll", type=float, default=0.05,
                   help="MemoryProbe background poll interval, seconds")
    p.add_argument("--no-poll", action="store_true",
                   help="disable the background poller (allocator peaks only)")
    args = p.parse_args(argv)

    if args.local is not None and float(args.local).is_integer() and args.local > 1:
        args.local = int(args.local)

    _build_and_run(args)


if __name__ == "__main__":
    main()
