"""MiKV configuration and budget resolution.

Two ways to size the importance cache, exactly one per run. Both come down to
one number, the importance ratio ``r``: after ``S`` tokens, the importance
cache holds ``floor(r * S)`` of them per (row, KV head), and every other token
is retained at ``lo_bits``. The ratio holds at every step — prefill and each
decode token — so a generated token demotes an old one only as often as the
ratio requires (four steps in five at ``r = 0.2``).

``importance_ratio`` — **the paper's knob.** ``r`` itself. The headline point
is 0.20 (Tables 2-3, Fig. 6). ``1.0`` at ``hi_bits = 16`` keeps every token
exact and never demotes: the full cache through the same code, which makes it
the control arm. ``0.0`` quantizes every token uniformly: the paper's RTN arm.

``cache_budget`` — **this repository's knob**, for a like-for-like memory
comparison with the windowed cache: a fraction of the dense fp16 K+V bytes.
``r`` is solved so codes + grids + exact tokens are exactly that fraction —
and because MiKV keeps every token, that fraction then holds at every length,
not just at the end of the horizon. The floor is the all-``lo_bits`` cache
(~15.6% of fp16 at INT2 / ``D/2`` groups); a budget below it raises rather than
quietly evicting.

Neither mode counts the per-token bookkeeping (positions, scores, the
balancer) — the paper does not, and H2O needs the scores anyway.
:meth:`MiKVCache.memory_report` reports those bytes separately.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .quant import SUPPORTED_BITS, bits_per_element, token_bytes

#: ``lo_bits`` value that drops demoted tokens: H2O, the paper's baseline.
EVICT = 0

#: The paper's headline operating point: 20% of the prompt in the importance
#: cache, fp16, the rest at INT2 with the balancer (Tables 2-3).
PAPER_IMPORTANCE_RATIO = 0.20


class MiKVConfigError(ValueError):
    """A MiKV configuration that cannot describe a valid cache."""


@dataclass
class MiKVConfig:
    #: Fraction of the prompt held in the importance cache (the paper's knob).
    #: ``None`` with ``cache_budget`` also ``None`` means the paper's 0.20.
    importance_ratio: Optional[float] = None
    #: Byte budget as a fraction of the dense fp16 cache over the horizon.
    cache_budget: Optional[float] = None
    #: Share of the importance cache reserved for the most recent tokens. H2O
    #: splits its budget evenly between heavy hitters and recency; so does this.
    recent_ratio: float = 0.5
    #: Importance-cache precision: 16 (the model dtype, exact), 8 or 4 (Table 3).
    hi_bits: int = 16
    #: Retained-cache precision: 2, 3 or 4 (Tables 1-2). 0 drops the demoted
    #: tokens instead of keeping them — H2O itself, the paper's eviction
    #: baseline, through the same selection code (B = 1 / unpadded only).
    lo_bits: int = 2
    #: Channels per quantization group. ``None`` = ``head_dim // 2`` (§3.2).
    group_size: Optional[int] = None
    #: The outlier-aware channel balancer (Eqs. 2-4). Off = plain per-token RTN.
    channel_balancer: bool = True
    #: Cap on the fp32 transient of the prefill score pass, in bytes. The pass
    #: is a sum over query rows, so it runs in row blocks sized to this.
    score_chunk_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.importance_ratio is not None and self.cache_budget is not None:
            raise MiKVConfigError(
                "set importance_ratio OR cache_budget, not both: they are two "
                "ways of sizing the same importance cache")
        if self.importance_ratio is None and self.cache_budget is None:
            self.importance_ratio = PAPER_IMPORTANCE_RATIO
        for name in ("importance_ratio", "cache_budget", "recent_ratio"):
            v = getattr(self, name)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise MiKVConfigError(f"{name} must be a float, got {v!r}")
            setattr(self, name, float(v))
        if self.importance_ratio is not None and not 0.0 <= self.importance_ratio <= 1.0:
            raise MiKVConfigError(
                f"importance_ratio must be in [0, 1], got {self.importance_ratio}. "
                "0 is uniform per-token RTN at lo_bits (the paper's RTN arm); "
                "1.0 at hi_bits=16 is the full cache (the control arm).")
        if self.cache_budget is not None and not 0.0 < self.cache_budget <= 1.0:
            raise MiKVConfigError(
                f"cache_budget must be in (0, 1], got {self.cache_budget}")
        if not 0.0 <= self.recent_ratio <= 1.0:
            raise MiKVConfigError(
                f"recent_ratio must be in [0, 1], got {self.recent_ratio}")
        if self.hi_bits not in (4, 8, 16):
            raise MiKVConfigError(f"hi_bits must be 16, 8 or 4, got {self.hi_bits}")
        if self.lo_bits != EVICT and (self.lo_bits not in SUPPORTED_BITS
                                      or self.lo_bits == 16):
            raise MiKVConfigError(
                f"lo_bits must be 0 (evict, H2O) or one of "
                f"{SUPPORTED_BITS[:-1]}, got {self.lo_bits}")
        if self.lo_bits > self.hi_bits:
            raise MiKVConfigError(
                f"lo_bits={self.lo_bits} exceeds hi_bits={self.hi_bits}: the "
                "retained cache would be MORE precise than the important one")
        if self.group_size is not None and (self.group_size <= 0 or self.group_size % 8):
            raise MiKVConfigError(
                f"group_size must be a positive multiple of 8, got {self.group_size}")
        if self.score_chunk_bytes <= 0:
            raise MiKVConfigError("score_chunk_bytes must be positive")

    def resolved_group_size(self, head_dim: int) -> int:
        g = self.group_size if self.group_size is not None else head_dim // 2
        if head_dim % g:
            raise MiKVConfigError(f"group_size {g} does not divide head_dim {head_dim}")
        if g % 8:
            raise MiKVConfigError(f"group_size {g} is not a multiple of 8")
        return g

    def resolve(self, head_dim: int, elem_bytes: int = 2) -> "MiKVBudget":
        """Settle the importance ratio and group size for this model's heads.

        Both sizing modes reduce to one ratio ``r``: the importance cache holds
        ``floor(r * S)`` of the ``S`` tokens seen so far, at every step. Under
        ``cache_budget`` it is the ``r`` at which K+V bytes are exactly
        ``cache_budget`` of the dense fp16 cache, which — MiKV keeping every
        token — holds at every ``S``, not only at the end of the horizon.
        """
        g = self.resolved_group_size(head_dim)
        if self.importance_ratio is not None:
            ratio = self.importance_ratio
            mode = "importance_ratio"
        else:
            hi_tok = token_bytes(self.hi_bits, g, head_dim, elem_bytes)
            lo_tok = (0 if self.lo_bits == EVICT
                      else token_bytes(self.lo_bits, g, head_dim, elem_bytes))
            full_tok = head_dim * elem_bytes
            floor_frac = lo_tok / full_tok
            if self.cache_budget < floor_frac:
                raise MiKVConfigError(
                    f"cache_budget={self.cache_budget} is below MiKV's floor: "
                    f"every token is kept at lo_bits={self.lo_bits}, which alone "
                    f"is {floor_frac:.4f} of the fp16 cache. MiKV does not "
                    "evict, so it cannot go lower.")
            if hi_tok == lo_tok:
                ratio = 1.0
            else:
                ratio = (self.cache_budget * full_tok - lo_tok) / (hi_tok - lo_tok)
            mode = "cache_budget"
        return MiKVBudget(
            mode=mode,
            importance_ratio=min(max(ratio, 0.0), 1.0),
            recent_ratio=self.recent_ratio,
            group_size=g,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MiKVBudget:
    """What :meth:`MiKVConfig.resolve` settled on for one model."""

    mode: str
    #: Fraction of the tokens seen that the importance cache holds.
    importance_ratio: float
    #: Fraction of the importance cache reserved for the newest tokens.
    recent_ratio: float
    group_size: int

    def n_hi(self, seq_len: int) -> int:
        """Importance-cache size, per (row, KV head), once ``seq_len`` are seen.

        A fixed fraction of the CURRENT length — the paper's importance ratio
        is a share of the cache, and its reported cache sizes (Table 3) are
        shares of the whole sequence. Non-decreasing in ``seq_len``, and it
        grows by at most one per token, so a token that has left the recent
        window never re-enters it.
        """
        return min(seq_len, int(math.floor(self.importance_ratio * seq_len + 1e-9)))

    def n_recent(self, seq_len: int) -> int:
        return int(math.floor(self.n_hi(seq_len) * self.recent_ratio + 1e-9))


def paper_cache_size(importance_ratio: float, hi_bits: int, lo_bits: int,
                     group_size: int = 64, head_dim: int = 128) -> float:
    """KV cache size as a fraction of fp16, in the paper's accounting.

    ``(r * bits(hi) + (1 - r) * bits(lo)) / 16`` with the fp16 grid counted
    (:func:`bits_per_element`). Reproduces Table 3 at ``r = 0.2``, INT2, ``g =
    64``: 32.5% / 23.1% / 18.1% / 15.6% for fp16 / INT8 / INT4 / INT2
    importance caches — printed there as 33 / 23 / 18 / 16%.
    """
    hi = bits_per_element(hi_bits, group_size, head_dim)
    lo = 0.0 if lo_bits == EVICT else bits_per_element(lo_bits, group_size, head_dim)
    return (importance_ratio * hi + (1.0 - importance_ratio) * lo) / 16.0
