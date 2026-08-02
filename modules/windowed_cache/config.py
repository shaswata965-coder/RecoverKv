
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from .policy import FIRST_EVICTION_STEP


@dataclass(frozen=True)
class ResolvedConfig:

    window_size: int
    num_sink_tokens: int
    local_tokens: int
    top_k_windows: int
    bytes_per_token: int
    total_budget_bytes: int
    total_budget_tokens: int
    rerotate_on_evict: bool = False
    first_eviction_step: int = FIRST_EVICTION_STEP
    quant_ratio: float = 0.0
    top_k_fp: int = -1
    N_q: int = 0
    quant_memoize_read: Optional[bool] = None
    quant_promotion: bool = True

    def __post_init__(self) -> None:
        if self.top_k_fp < 0:
            object.__setattr__(self, "top_k_fp", self.top_k_windows)


@dataclass
class WindowedCacheConfig:

    window_size: int
    num_sink_tokens: int
    local_window_size: Union[int, float]
    cache_budget: float
    track_scores: bool = False
    rerotate_on_evict: bool = False
    quant_ratio: float = 0.0
    quant_memoize_read: Optional[bool] = None
    quant_promotion: bool = True
    first_eviction_step: int = FIRST_EVICTION_STEP

    def __post_init__(self) -> None:
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool):
            raise ValueError(
                f"window_size must be a positive int, got {self.window_size!r}"
            )
        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be > 0, got {self.window_size}"
            )

        if not isinstance(self.num_sink_tokens, int) or isinstance(self.num_sink_tokens, bool):
            raise ValueError(
                f"num_sink_tokens must be a non-negative int, got {self.num_sink_tokens!r}"
            )
        if self.num_sink_tokens < 0:
            raise ValueError(
                f"num_sink_tokens must be >= 0, got {self.num_sink_tokens}"
            )

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

        if isinstance(self.quant_ratio, bool):
            raise ValueError("quant_ratio must be a float in [0, 1], got bool")
        if isinstance(self.quant_ratio, int) and not isinstance(self.quant_ratio, bool):
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
        if self.quant_ratio > 0.0 and self.window_size % 4 != 0:
            raise ValueError(
                f"quant_ratio > 0 requires a window_size divisible by 4 (int2 "
                f"packing), got window_size={self.window_size}"
            )

        if self.quant_memoize_read is not None and not isinstance(
            self.quant_memoize_read, bool
        ):
            raise ValueError(
                f"quant_memoize_read must be None (auto) or bool, got "
                f"{type(self.quant_memoize_read).__name__}"
            )

        if not isinstance(self.quant_promotion, bool):
            raise ValueError(
                f"quant_promotion must be a bool, got "
                f"{type(self.quant_promotion).__name__}"
            )

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


    def resolve(
        self,
        prefill_len: int,
        model_config: Any,
        kv_dtype: torch.dtype,
        max_tokens: int,
    ) -> ResolvedConfig:
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

        if self.quant_ratio > 0.0 and head_dim % 4 != 0:
            raise ValueError(
                f"quant_ratio > 0 requires head_dim divisible by 4 (int2 crumb "
                f"packing over the value channel axis), got head_dim={head_dim}"
            )

        element_size = torch.tensor([], dtype=kv_dtype).element_size()
        bytes_per_token = num_kv_heads * head_dim * element_size * 2

        total_budget_bytes = int(self.cache_budget * (prefill_len + max_tokens) * bytes_per_token)
        total_budget_tokens = total_budget_bytes // bytes_per_token

        if isinstance(self.local_window_size, float):
            raw = self.local_window_size * total_budget_tokens
            ceiled = math.ceil(raw)
            remainder = ceiled % self.window_size
            if remainder != 0:
                ceiled += self.window_size - remainder
            local_tokens = ceiled
        else:
            local_tokens = self.local_window_size

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

        q = self.quant_ratio
        b_fp = bytes_per_token * self.window_size
        b_q = (
            (num_kv_heads * head_dim * self.window_size) // 2
            + 4 * num_kv_heads * head_dim
            + 4 * num_kv_heads * self.window_size
        )
        m_evict = remaining * bytes_per_token
        top_k_fp = int(((1.0 - q) * m_evict) // b_fp)
        N_q = int((q * m_evict) // b_q) if q > 0.0 else 0
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
            quant_promotion=self.quant_promotion,
            first_eviction_step=self.first_eviction_step,
        )
