"""Analytic prefill-memory model — predict an OOM before spending the GPU hour.

Every term here is arithmetic over the model config and the run shape, so this
runs on any machine and can be validated against a run that already happened.
It exists because a 4096/32 OOM currently costs a full model load plus a full
prefill and yields one line of log ("OOM ... — skipping"), which is the most
expensive way possible to learn a number that was computable in advance.

**The four terms, and which one is the surprise.**

``weights``      params x dtype. Fixed; the floor.
``prefill_kv``   2 (K,V) x L x B x H_kv x S x D x kv_bytes. NOTE this is the
                 FULL, uncompressed cache: with ``first_eviction_step=0`` the
                 compaction runs on decode step 0, so the whole prompt is
                 resident for every row throughout prefill. Compression buys
                 steady-state batch capacity, not prefill headroom.
``lse_recompute`` the surprise. When L-reuse misses, every prefill layer calls
                 ``score_kernel.compute_lse``, which materialises
                 ``[B, H_kv, rep, chunk, S]`` in fp32 -- i.e.
                 ``B x H_q x min(chunk,S) x S x 4`` bytes. It is LINEAR in
                 batch and (once ``chunk`` saturates) LINEAR in context, and at
                 4096/32 it is larger than the weights. None of the ops in that
                 chain are in-place (``matmul(...).float() * scaling``, then
                 ``aw.masked_fill(...)``), so two full fp32 blocks are live at
                 the crossover -- see ``NONINPLACE_FACTOR``.
``other``        attention/MLP activations, logits, allocator slack. Not
                 derived -- CALIBRATED from one measured run (see
                 ``CALIBRATION``) and scaled linearly in B*S. Treat it as the
                 error bar on the whole prediction, not as a known quantity.

The actionable consequence is in ``lse_recompute``: it is the only term that is
pure waste. ``STICKYKV_PREFILL_SCORE_CHUNK`` divides it linearly with no logic
change and no change in total FLOPs (same work, smaller blocks, more
iterations), and a working L-reuse removes it outright.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

GB = 1024 ** 3

#: How many full fp32 blocks ``compute_lse`` holds live at its crossover.
#:
#: This was a guess (2.0, reasoning that ``matmul(...).float()``, ``* scaling``
#: and ``masked_fill`` each allocate while the previous is still referenced).
#: It is now SOLVED from two measurements at the same shape — 1048/batch-32,
#: L recomputed vs L reused — which differ only by this term:
#:
#:     L recomputed : torch alloc peak 29.25 GB
#:     L reused     : torch alloc peak 24.73 GB
#:     difference   :                   4.52 GB   <- the transient, measured
#:     one fp32 block at that shape:     4.09 GB
#:
#: So the real factor is 1.10, not 2.0: the allocator reuses the freed block
#: for the next op far more often than the source reading suggests. Measured
#: beats reasoned; the guess was 82% too high.
NONINPLACE_FACTOR = 4.52 / 4.094

#: The single measured point this model's ``other`` term is calibrated against.
#: Meta-Llama-3-8B-Instruct, A100-80GB, fp16, from the run that produced
#: outputs/perf_table/perf_prefill1048_gen1049_bs32.npz. ONE point: the model
#: interpolates honestly and extrapolates on trust.
#: NOTE ``measured_end_alloc_gb`` is NOT the weights: at end-of-run the
#: compacted KV is still resident, so it reads ~1 GiB above the derived weight
#: figure. The model derives weights from the config and treats the difference
#: as evidence the derivation is right, not as a discrepancy to paper over.
CALIBRATION = {
    "prefill_len": 1048,
    "batch_size": 32,
    "measured_alloc_peak_gb": 29.25,
    "measured_device_peak_gb": 35.67,
    "measured_end_alloc_gb": 16.00,
    #: The SAME cell with L-reuse working (STICKYKV_LSE_BACKEND=flash). Differs
    #: from the row above only by the compute_lse transient, which is what makes
    #: both that term and ``other`` solvable rather than fitted. Buffer sizing
    #: does not depend on gen length (cache.py:367), so the shorter generation
    #: this was measured at does not affect the prefill peak.
    "measured_alloc_peak_lse_reused_gb": 24.73,
}

#: Fraction by which the device's used peak exceeds torch's allocated peak
#: (CUDA context + reserved-but-unallocated + fragmentation). Modelled as a
#: FRACTION, not a constant: the calibration run showed 5.89 GB of
#: reserved-minus-allocated at a 29 GB allocation, and fragmentation tracks
#: allocation volume rather than sitting still as the shape grows.
OVERHEAD_FRACTION = (
    CALIBRATION["measured_device_peak_gb"] / CALIBRATION["measured_alloc_peak_gb"] - 1.0
)

#: Predictions within this fraction of device total are reported MARGINAL
#: rather than FITS/OOM. The ``other`` term is fitted from one point and
#: fragmentation is not deterministic, so finer resolution than this would be
#: false precision.
MARGINAL_BAND = 0.10


@dataclass
class MemoryBreakdown:
    """Per-term prefill memory prediction, in GB."""

    weights: float
    prefill_kv: float
    lse_recompute: float
    other: float
    #: Measured gap between torch's allocated peak and the device's used peak
    #: (CUDA context + reserved-but-unallocated + fragmentation).
    allocator_overhead: float
    device_total: Optional[float] = None
    notes: list = field(default_factory=list)

    @property
    def alloc_total(self) -> float:
        """Predicted torch *allocated* peak."""
        return self.weights + self.prefill_kv + self.lse_recompute + self.other

    @property
    def device_total_used(self) -> float:
        """Predicted *device used* peak — the figure an OOM is decided on."""
        return self.alloc_total + self.allocator_overhead

    @property
    def fits(self) -> Optional[bool]:
        if self.device_total is None:
            return None
        return self.device_total_used <= self.device_total

    @property
    def verdict(self) -> Optional[str]:
        """``"fits"`` / ``"marginal"`` / ``"oom"``, or None without a device size.

        ``marginal`` is a real answer, not a hedge: the ``other`` term is fitted
        from a single measured point and fragmentation is not deterministic, so
        a prediction inside :data:`MARGINAL_BAND` of the device total does not
        distinguish the two outcomes and should not pretend to.
        """
        if self.device_total is None:
            return None
        band = self.device_total * MARGINAL_BAND
        if self.device_total_used > self.device_total + band:
            return "oom"
        if self.device_total_used > self.device_total - band:
            return "marginal"
        return "fits"

    def format(self) -> str:
        rows = [
            ("weights", self.weights),
            ("prefill KV (uncompressed)", self.prefill_kv),
            ("compute_lse transient", self.lse_recompute),
            ("other (calibrated)", self.other),
        ]
        w = max(len(k) for k, _ in rows)
        out = [f"  {k:<{w}}  {v:8.2f} GB" for k, v in rows]
        out.append(f"  {'-' * (w + 12)}")
        out.append(f"  {'torch allocated peak':<{w}}  {self.alloc_total:8.2f} GB")
        out.append(f"  {'allocator overhead':<{w}}  {self.allocator_overhead:8.2f} GB")
        out.append(f"  {'DEVICE USED peak':<{w}}  {self.device_total_used:8.2f} GB")
        if self.device_total is not None:
            head = self.device_total - self.device_total_used
            label = {"fits": "FITS", "marginal": "MARGINAL (inside the error bar)",
                     "oom": "OOM PREDICTED"}[self.verdict]
            out.append(f"  {'device total':<{w}}  {self.device_total:8.2f} GB")
            out.append(f"  -> {label}  (headroom {head:+.2f} GB)")
        for n in self.notes:
            out.append(f"  ! {n}")
        return "\n".join(out)


def _geometry(model_config: Any) -> Dict[str, int]:
    """Pull (L, H_q, H_kv, D) off a HF config, tolerating missing head_dim."""
    h_q = int(getattr(model_config, "num_attention_heads", 0) or 0)
    h_kv = int(getattr(model_config, "num_key_value_heads", 0) or h_q)
    hidden = int(getattr(model_config, "hidden_size", 0) or 0)
    d = int(getattr(model_config, "head_dim", 0) or 0) or (
        hidden // h_q if h_q else 0)
    return {
        "layers": int(getattr(model_config, "num_hidden_layers", 0) or 0),
        "h_q": h_q,
        "h_kv": h_kv,
        "head_dim": d,
    }


def prefill_kv_gb(model_config: Any, batch_size: int, prefill_len: int,
                  kv_bytes: int = 2) -> float:
    """Full uncompressed K+V resident through prefill, in GB."""
    g = _geometry(model_config)
    n = (2 * g["layers"] * max(1, batch_size) * g["h_kv"]
         * max(0, prefill_len) * g["head_dim"])
    return n * kv_bytes / GB


def lse_recompute_gb(model_config: Any, batch_size: int, prefill_len: int,
                     chunk: int = 1024,
                     noninplace_factor: float = NONINPLACE_FACTOR) -> float:
    """Peak live bytes of ``compute_lse``'s fp32 block chain, in GB.

    One block is ``B x H_q x min(chunk, S) x S x 4``. ``noninplace_factor``
    accounts for the two blocks that coexist across the non-in-place ops.
    """
    g = _geometry(model_config)
    block = (max(1, batch_size) * g["h_q"]
             * min(max(chunk, 1), max(prefill_len, 1)) * max(prefill_len, 1) * 4)
    return block * noninplace_factor / GB


def weights_gb(model_config: Any, param_bytes: int = 2,
               n_params: Optional[int] = None) -> float:
    """Weight bytes. Pass ``n_params`` when the real count is known."""
    if n_params is not None:
        return n_params * param_bytes / GB
    g = _geometry(model_config)
    hidden = int(getattr(model_config, "hidden_size", 0) or 0)
    inter = int(getattr(model_config, "intermediate_size", 0) or 0)
    vocab = int(getattr(model_config, "vocab_size", 0) or 0)
    per_layer = (hidden * g["h_q"] * g["head_dim"]          # q_proj
                 + 2 * hidden * g["h_kv"] * g["head_dim"]   # k_proj, v_proj
                 + g["h_q"] * g["head_dim"] * hidden        # o_proj
                 + 3 * hidden * inter)                      # gate, up, down
    return (g["layers"] * per_layer + 2 * vocab * hidden) * param_bytes / GB


def _calibrated_other_gb(model_config: Any, batch_size: int, prefill_len: int,
                         chunk: int, noninplace_factor: float,
                         w_gb: float) -> float:
    """Residual of the calibration point, scaled linearly in ``B * S``.

    Everything the four named terms do not cover: attention and MLP
    activations, the logit tensor, transient copies. Fitted, not derived.

    Fitted against the L-REUSED measurement, where the transient is absent, so
    the residual is not contaminated by the term whose size we are also
    estimating. With both measurements at one shape the system is exactly
    determined: their difference gives the transient, and the L-reused point
    gives ``other`` directly.
    """
    c = CALIBRATION
    residual = (
        c["measured_alloc_peak_lse_reused_gb"]
        - w_gb
        - prefill_kv_gb(model_config, c["batch_size"], c["prefill_len"])
    )
    residual = max(residual, 0.0)
    base = c["batch_size"] * c["prefill_len"]
    scale = (max(1, batch_size) * max(1, prefill_len)) / base if base else 1.0
    return residual * scale


def predict_prefill_peak(
    model_config: Any,
    batch_size: int,
    prefill_len: int,
    *,
    lse_recomputes: bool = True,
    chunk: int = 1024,
    kv_bytes: int = 2,
    param_bytes: int = 2,
    n_params: Optional[int] = None,
    device_total_gb: Optional[float] = None,
    allocator_overhead_gb: Optional[float] = None,
    noninplace_factor: float = NONINPLACE_FACTOR,
) -> MemoryBreakdown:
    """Predicted prefill peak for one (shape, batch) cell.

    ``lse_recomputes=False`` models a working L-reuse — the term drops out
    entirely, which is the single largest lever at large B*S.
    """
    w = weights_gb(model_config, param_bytes=param_bytes, n_params=n_params)
    kv = prefill_kv_gb(model_config, batch_size, prefill_len, kv_bytes=kv_bytes)
    lse = (lse_recompute_gb(model_config, batch_size, prefill_len, chunk=chunk,
                            noninplace_factor=noninplace_factor)
           if lse_recomputes else 0.0)
    other = _calibrated_other_gb(model_config, batch_size, prefill_len, chunk,
                                 noninplace_factor, w)
    if allocator_overhead_gb is None:
        # Scaled with the allocation, not held constant — see OVERHEAD_FRACTION.
        allocator_overhead_gb = (w + kv + lse + other) * OVERHEAD_FRACTION

    notes = []
    if lse_recomputes and lse > w:
        notes.append(
            f"the compute_lse transient ({lse:.1f} GB) EXCEEDS the model weights "
            f"({w:.1f} GB) at this shape — it is pure waste from the L-reuse miss")
    if lse_recomputes:
        smaller = lse_recompute_gb(model_config, batch_size, prefill_len,
                                   chunk=max(chunk // 8, 1),
                                   noninplace_factor=noninplace_factor)
        notes.append(
            f"STICKYKV_PREFILL_SCORE_CHUNK={max(chunk // 8, 1)} would cut that term "
            f"to {smaller:.1f} GB (same FLOPs, more iterations, no logic change)")
    return MemoryBreakdown(
        weights=w, prefill_kv=kv, lse_recompute=lse, other=other,
        allocator_overhead=allocator_overhead_gb,
        device_total=device_total_gb, notes=notes,
    )
