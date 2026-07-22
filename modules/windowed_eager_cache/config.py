"""WindowedCacheConfig and ResolvedConfig — typed, validated cache configuration.

``WindowedCacheConfig`` is the user-facing configuration dataclass.
``ResolvedConfig`` is the resolved (frozen) form with concrete integer counts
derived from byte-based budget accounting.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from .policy import FIRST_EVICTION_STEP


# ---------------------------------------------------------------------------
# ResolvedConfig (frozen, output of resolve())
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """Resolved cache configuration with concrete integer counts.

    Produced by :meth:`WindowedCacheConfig.resolve`.  All fields are ints
    (or the original ``window_size`` / ``num_sink_tokens``).
    """

    window_size: int
    num_sink_tokens: int
    local_tokens: int          # resolved post percentage-rounding
    top_k_windows: int         # may be 0 (legal — sink + local only)
    bytes_per_token: int
    total_budget_bytes: int
    total_budget_tokens: int
    rerotate_on_evict: bool = False
    # Decode step at which the FIRST eviction fires, independent of window_size
    # (EvictionPolicy.should_evict). Carried verbatim from WindowedCacheConfig.
    first_eviction_step: int = FIRST_EVICTION_STEP
    # --- two-tier quantization (design.md §7) ---
    # quant_ratio q splits the EVICTABLE window budget between the fp16 (K) tier
    # and the int4 (Q) tier by memory. q=0 disables the Q tier entirely and every
    # field below reduces to today's single-tier config (top_k_fp == top_k_windows,
    # N_q == 0), so the pure-fp16 path stays byte-identical.
    quant_ratio: float = 0.0
    top_k_fp: int = -1         # evictable fp windows; sentinel -1 → top_k_windows
    N_q: int = 0               # evictable int4 windows (0 when q=0)
    # None → decide from the batch size at the first forward (on at B=1, off
    # above). See WindowedCacheConfig.quant_memoize_read.
    quant_memoize_read: Optional[bool] = None

    def __post_init__(self) -> None:
        # top_k_fp defaults to top_k_windows so direct constructions (and every
        # q=0 path) mirror the single-tier count without extra plumbing.
        if self.top_k_fp < 0:
            object.__setattr__(self, "top_k_fp", self.top_k_windows)


# ---------------------------------------------------------------------------
# WindowedCacheConfig (user-facing)
# ---------------------------------------------------------------------------


@dataclass
class WindowedCacheConfig:
    """User-facing configuration for the windowed KV cache.

    Parameters
    ----------
    window_size : int
        Size of each scoring window in tokens.  Must be > 0.
    num_sink_tokens : int
        Number of sink tokens always retained at the start.  Must be >= 0.
    local_window_size : int | float
        If int: number of local tokens (must be a multiple of *window_size*).
        If float: ratio in (0, 1] of the **cache budget** (not the full
        context) -- ``local ~= ratio * total_budget_tokens``, ``ceil`` then
        snap up to the nearest *window_size* multiple.  Guarantees the local
        region can never exceed the cache budget.
    cache_budget : float
        Fraction of full-cache memory to retain, in (0, 1].
        Must be ``float`` — ``int`` and ``bool`` are rejected with clear errors.
    track_scores : bool
        Enable telemetry recording.  Default ``False``.
    rerotate_on_evict : bool
        If ``True``, re-rotate surviving keys to contiguous positions after
        each eviction (StreamingLLM-style).  **Default ``False``.**  Re-rotation
        is only correct if the model also rebases the query's RoPE position to
        the compacted cache length every step; HuggingFace ``generate``
        (transformers <= 4.47) advances ``cache_position`` monotonically
        instead, so re-rotating keys while the query keeps its original absolute
        position corrupts RoPE phase after the first eviction.  Leaving this off
        keeps original key positions (KVPress / H2O behaviour), correct on any
        version.
    quant_ratio : float
        Two-tier split ``q`` in ``[0, 1]`` (design.md §7).  **Default 0.0** —
        the Q (int4) tier is disabled and the cache is byte-identical to the
        single-tier fp16 path.  When ``q > 0``, the evictable window budget is
        split by memory: ``(1-q)`` to the fp16 tier, ``q`` to the int4 tier
        (which holds ~4× the windows per byte).  Requires an even ``window_size``
        (int4 nibble packing).
    quant_memoize_read : bool, optional
        Cache the dequantized + RoPE'd Q tier between evictions, per layer.
        **Default ``None`` = auto: on at ``B == 1``, off above.**

        The memo is free at ``B == 1`` and pure waste of batch capacity above it.
        At ``B == 1`` decode is weight-bound — every weight is read once per step
        (16.06 GB for Llama-3.1-8B fp16, a 10.3 ms/token floor on an A100) and
        that floor is independent of KV size — so the memo's footprint costs
        nothing and it saves 7 of every 8 steps' dequant at ``window_size = 8``.

        At ``B > 1`` it is charged **per row**, and batch capacity is the whole
        thesis: batching is the only thing that amortizes the weight read, and
        ``B`` is capped by KV memory, which is exactly what this method
        compresses. Measured at the qasper steady state (``T_fp=565``,
        ``T_q=1136``, 32 layers, ``H_kv=8``, ``D=128``, fp16) the memo holds
        ~4.65 MB/layer/row ⇒ **~149 MB/row**, against ~125 MB/row for the actual
        two-tier KV — it more than doubles per-row memory and cuts max-``B`` from
        ~458 to ~204.

        Set ``True`` to force it on at ``B > 1``: it trades that batch capacity
        for ~8x fewer Q-tier dequants per step (the Phase-1 read path
        rematerializes the whole tier every step without it — design.md §8, which
        Phase 2's fused kernel is what actually fixes). Which side wins at max-``B``
        is a measurement the perf suite must make; the default is set by the
        memory bound, which is the one we can compute.
    quant_ratio : float
        (see above)

    Notes
    -----
    Scoring is H2O-style cumulative: every query row contributes to the
    per-key score at every step.  There is no observation window.
    """

    window_size: int
    num_sink_tokens: int
    local_window_size: Union[int, float]
    cache_budget: float
    track_scores: bool = False
    rerotate_on_evict: bool = False
    quant_ratio: float = 0.0
    quant_memoize_read: Optional[bool] = None
    # Decode step of the FIRST eviction, independent of window_size (default 8).
    # See EvictionPolicy.should_evict; must be a non-negative int (0 = fire from
    # the first decode step, the pre-fixed-offset behaviour).
    first_eviction_step: int = FIRST_EVICTION_STEP

    def __post_init__(self) -> None:
        # -- window_size --
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool):
            raise ValueError(
                f"window_size must be a positive int, got {self.window_size!r}"
            )
        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be > 0, got {self.window_size}"
            )

        # -- num_sink_tokens --
        if not isinstance(self.num_sink_tokens, int) or isinstance(self.num_sink_tokens, bool):
            raise ValueError(
                f"num_sink_tokens must be a non-negative int, got {self.num_sink_tokens!r}"
            )
        if self.num_sink_tokens < 0:
            raise ValueError(
                f"num_sink_tokens must be >= 0, got {self.num_sink_tokens}"
            )

        # -- cache_budget (must be float, not int, not bool) --
        if isinstance(self.cache_budget, bool):
            raise ValueError(
                f"cache_budget must be a float in (0, 1], got bool {self.cache_budget!r}. "
                f"bool is rejected because it subclasses int."
            )
        if isinstance(self.cache_budget, int):
            raise ValueError(
                f"cache_budget must be a float ratio in (0, 1], got int {self.cache_budget}. "
                f"Use e.g. 0.40 instead of 40."
            )
        if not isinstance(self.cache_budget, float):
            raise ValueError(
                f"cache_budget must be a float in (0, 1], got {type(self.cache_budget).__name__}"
            )
        if not (0.0 < self.cache_budget <= 1.0):
            raise ValueError(
                f"cache_budget must be in (0, 1], got {self.cache_budget}"
            )

        # -- local_window_size --
        if isinstance(self.local_window_size, bool):
            raise ValueError("local_window_size must be int or float, got bool")
        if isinstance(self.local_window_size, int):
            if self.local_window_size <= 0:
                raise ValueError(
                    f"local_window_size as int must be > 0, got {self.local_window_size}"
                )
            if self.local_window_size % self.window_size != 0:
                raise ValueError(
                    f"local_window_size as int ({self.local_window_size}) must be a "
                    f"multiple of window_size ({self.window_size})"
                )
        elif isinstance(self.local_window_size, float):
            if not (0.0 < self.local_window_size <= 1.0):
                raise ValueError(
                    f"local_window_size as float must be in (0, 1], "
                    f"got {self.local_window_size}"
                )
        else:
            raise ValueError(
                f"local_window_size must be int or float, "
                f"got {type(self.local_window_size).__name__}"
            )

        # -- quant_ratio (two-tier split, design.md §7) --
        if isinstance(self.quant_ratio, bool):
            raise ValueError("quant_ratio must be a float in [0, 1], got bool")
        if isinstance(self.quant_ratio, int) and not isinstance(self.quant_ratio, bool):
            # allow the literal 0 / 1 as a convenience; promote to float
            self.quant_ratio = float(self.quant_ratio)
        if not isinstance(self.quant_ratio, float):
            raise ValueError(
                f"quant_ratio must be a float in [0, 1], "
                f"got {type(self.quant_ratio).__name__}"
            )
        if not (0.0 <= self.quant_ratio <= 1.0):
            raise ValueError(
                f"quant_ratio must be in [0, 1], got {self.quant_ratio}"
            )
        if self.quant_ratio > 0.0 and self.window_size % 2 != 0:
            # int4 nibble packing needs an even window (2 codes/byte, design §2).
            raise ValueError(
                f"quant_ratio > 0 requires an even window_size (int4 packing), "
                f"got window_size={self.window_size}"
            )

        # -- quant_memoize_read --
        if self.quant_memoize_read is not None and not isinstance(
            self.quant_memoize_read, bool
        ):
            raise ValueError(
                f"quant_memoize_read must be None (auto) or bool, got "
                f"{type(self.quant_memoize_read).__name__}"
            )

        # -- first_eviction_step (non-negative int; bool rejected before int) --
        if isinstance(self.first_eviction_step, bool) or not isinstance(
            self.first_eviction_step, int
        ):
            raise ValueError(
                f"first_eviction_step must be a non-negative int, got "
                f"{self.first_eviction_step!r}"
            )
        if self.first_eviction_step < 0:
            raise ValueError(
                f"first_eviction_step must be >= 0, got {self.first_eviction_step}"
            )

    # -----------------------------------------------------------------
    # resolve() — pure function, no mutation
    # -----------------------------------------------------------------

    def resolve(
        self,
        prefill_len: int,
        model_config: Any,
        kv_dtype: torch.dtype,
        max_tokens: int,
    ) -> ResolvedConfig:
        """Return a :class:`ResolvedConfig` with concrete int counts.

        Pure function; doesn't mutate *self*.  Floor-division on byte→token
        conversion guarantees the retained cache never exceeds the byte budget.

        Parameters
        ----------
        prefill_len : int
            Number of tokens in the prefill (prompt).
        model_config
            HuggingFace ``PretrainedConfig`` (or compatible object) with
            ``num_key_value_heads``, ``num_attention_heads``, ``hidden_size``,
            and optionally ``head_dim``.
        kv_dtype : torch.dtype
            Data type of the KV cache tensors (e.g. ``torch.float16``).
        max_tokens : int
            Maximum number of tokens to be generated.  The budget is sized
            against the full expected sequence (prefill + generation) so the
            cache is not undersized when the output is long.
        """
        num_kv_heads = getattr(
            model_config,
            "num_key_value_heads",
            getattr(model_config, "num_attention_heads", None),
        )
        if num_kv_heads is None:
            raise ValueError(
                "model_config must have num_key_value_heads or num_attention_heads"
            )
        head_dim = getattr(model_config, "head_dim", None)
        if head_dim is None:
            num_heads = getattr(model_config, "num_attention_heads", None)
            hidden = getattr(model_config, "hidden_size", None)
            if num_heads is None or hidden is None:
                raise ValueError(
                    "model_config must provide head_dim or (num_attention_heads + hidden_size)"
                )
            head_dim = hidden // num_heads

        element_size = torch.tensor([], dtype=kv_dtype).element_size()
        # K + V, each shaped [num_kv_heads, head_dim] per token
        bytes_per_token = num_kv_heads * head_dim * element_size * 2

        # Total byte budget and token budget
        total_budget_bytes = int(self.cache_budget * (prefill_len + max_tokens) * bytes_per_token)
        total_budget_tokens = total_budget_bytes // bytes_per_token

        # Resolve local_window_size to concrete int.
        # Float local_window_size is a fraction of the CACHE BUDGET (not the
        # full post-sink context): local ~= ratio * total_budget_tokens, then
        # ceil and snap up to a window_size multiple. This guarantees the local
        # region can never exceed the budget. (The previous post-sink-relative
        # formula could make the local region alone larger than the whole
        # budget, which either crashed resolve() or starved top-K retention.)
        if isinstance(self.local_window_size, float):
            raw = self.local_window_size * total_budget_tokens
            ceiled = math.ceil(raw)
            remainder = ceiled % self.window_size
            if remainder != 0:
                ceiled += self.window_size - remainder
            local_tokens = ceiled
        else:
            local_tokens = self.local_window_size

        # Top-K evictable windows.
        #
        # Sink + local may exceed the budget. That used to raise; it now clamps
        # to zero evictable windows and proceeds, retaining sink + local only —
        # an explicitly legal ResolvedConfig (see top_k_windows above).
        #
        # The clamp is not cosmetic. `remaining` feeds both top_k_windows and
        # m_evict below, and Python floor division takes a negative remaining
        # NEGATIVE (-5 // 8 == -1), which would propagate through top_k_fp / N_q
        # into n_slots_for() and CacheState(capacity=...) as negative sizes —
        # failing later, and far less legibly, than the check that was here.
        #
        # THE BUDGET IS THEN EXCEEDED, and silently would be the wrong kind of
        # quiet: the retained cache is num_sink + local_tokens, not
        # total_budget_tokens, so any bytes/compression figure derived from the
        # requested budget is wrong for this config. Over-retaining against a
        # nominal budget is the exact accounting error we fault the
        # LongBenchSticky baseline for, so say so out loud rather than let a
        # quiet over-retain look like a legitimate result.
        remaining = total_budget_tokens - self.num_sink_tokens - local_tokens
        if remaining < 0:
            warnings.warn(
                f"cache_budget yields total_budget_tokens={total_budget_tokens}, "
                f"below num_sink_tokens ({self.num_sink_tokens}) + local_tokens "
                f"({local_tokens}). Proceeding with 0 evictable windows: the cache "
                f"will retain {self.num_sink_tokens + local_tokens} tokens/row "
                f"(sink + local), EXCEEDING the requested budget by "
                f"{-remaining} tokens/row. Reported compression for this config is "
                f"not the requested budget.",
                RuntimeWarning,
                stacklevel=2,
            )
            remaining = 0
        top_k_windows = remaining // self.window_size

        # --- Two-tier split (design.md §7) --------------------------------
        # Only the EVICTABLE window budget is divided between tiers; sink and
        # local stay fp and are already carved as tokens above. At q=0 this
        # yields top_k_fp == top_k_windows and N_q == 0 by construction, so the
        # ResolvedConfig is field-for-field identical to the single-tier path.
        q = self.quant_ratio
        # one fp window vs one int4 window, in bytes:
        b_fp = bytes_per_token * self.window_size                       # K+V fp16
        b_q = (
            num_kv_heads * head_dim * self.window_size                  # int4 codes, K+V
            + 4 * num_kv_heads * head_dim                               # key scale+zero fp16
            + 4 * num_kv_heads * self.window_size                       # value scale+zero fp16
        )
        m_evict = remaining * bytes_per_token                           # evictable bytes
        top_k_fp = int(((1.0 - q) * m_evict) // b_fp)
        N_q = int((q * m_evict) // b_q) if q > 0.0 else 0
        # q=0 must reproduce today's count exactly (guard float floor drift).
        if q == 0.0:
            top_k_fp = top_k_windows

        return ResolvedConfig(
            window_size=self.window_size,
            num_sink_tokens=self.num_sink_tokens,
            local_tokens=local_tokens,
            top_k_windows=top_k_windows,
            bytes_per_token=bytes_per_token,
            total_budget_bytes=total_budget_bytes,
            total_budget_tokens=total_budget_tokens,
            rerotate_on_evict=self.rerotate_on_evict,
            quant_ratio=q,
            top_k_fp=top_k_fp,
            N_q=N_q,
            quant_memoize_read=self.quant_memoize_read,
            first_eviction_step=self.first_eviction_step,
        )
