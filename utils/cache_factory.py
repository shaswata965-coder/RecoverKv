"""Cache factory — single source of truth for backend selection.

Shell only for Prompt 01. The cache packages it routes to are built in
Prompt 02.  Two functions:

- ``get_cache_classes(backend)`` — lazy-imports and returns the
  ``(WindowedCache, WindowedCacheConfig, install_score_hooks)`` trio for
  either ``'flash_attn'`` or ``'eager'``.
- ``validate_backend_attn_pairing(backend, attn_implementation)`` — raises
  ``ConfigValidationError`` on mismatched backend/attention pairs.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Type

from utils.config import ConfigValidationError

__all__ = [
    "ConfigValidationError",
    "get_cache_classes",
    "quant_budget_mode_kwargs",
    "validate_backend_attn_pairing",
    "assert_transformers_version_supported",
    "is_transformers_version_supported",
    "MAX_SUPPORTED_TRANSFORMERS",
]


_BACKEND_TO_ATTN_IMPL = {
    "flash_attn": ("flash_attention_2",),
    "eager": ("eager",),
}

# Highest transformers version the windowed KV cache is known-correct on.
# The cache keeps surviving keys at their ORIGINAL RoPE positions after an
# eviction (rerotate_on_evict defaults False) and relies on HF ``generate``
# advancing the query's ``cache_position`` MONOTONICALLY — the transformers
# <= 4.47 behaviour. Newer transformers re-derive ``cache_position`` from the
# (now shrunken) cache length each step, so after the first eviction the query
# and the retained keys disagree on absolute position → corrupted RoPE phase
# and silently degraded results. Until the rerotation path is reworked, refuse
# to run on a newer version rather than emit wrong numbers.
MAX_SUPPORTED_TRANSFORMERS: Tuple[int, int, int] = (4, 47, 1)


def _parse_version(version: str) -> Tuple[int, int, int]:
    """Parse a version string to a ``(major, minor, patch)`` int tuple.

    Tolerant of suffixes like ``4.47.1.dev0`` / ``4.47.1+cu121`` — only the
    leading numeric ``major.minor.patch`` is used.
    """
    parts = []
    for chunk in version.split(".")[:3]:
        # Take only the LEADING run of digits, so "1+cu121" → 1 and
        # "1rc2" → 1 (not "1121" / "12").
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def is_transformers_version_supported(
    version: str,
    max_version: Tuple[int, int, int] = MAX_SUPPORTED_TRANSFORMERS,
) -> bool:
    """Return ``True`` iff *version* is <= *max_version* (inclusive)."""
    return _parse_version(version) <= max_version


def assert_transformers_version_supported(version: Optional[str] = None) -> None:
    """Raise ``ConfigValidationError`` if transformers is newer than supported.

    Called at the start of any *real* windowed-cache run (model-backed). Not
    invoked from pure-logic unit tests, which must stay runnable on dev boxes
    that have a newer transformers installed.

    Parameters
    ----------
    version : str, optional
        Version string to check. Defaults to the installed
        ``transformers.__version__``.
    """
    if version is None:
        import transformers

        version = transformers.__version__

    if not is_transformers_version_supported(version):
        supported = ".".join(str(p) for p in MAX_SUPPORTED_TRANSFORMERS)
        raise ConfigValidationError(
            f"transformers {version} is newer than the supported {supported}. "
            "The windowed KV cache keeps evicted-survivor keys at their original "
            "RoPE positions and relies on HF generate advancing cache_position "
            "monotonically (transformers <= 4.47). Newer versions re-derive "
            "cache_position from the compacted cache length after eviction, "
            "corrupting RoPE phase and silently degrading results. "
            f"Pin transformers=={supported} (see environment.yml), or rework the "
            "rerotation path and raise MAX_SUPPORTED_TRANSFORMERS before bumping."
        )


def get_cache_classes(backend: str) -> Tuple[Type, Type, Callable]:
    """Return ``(WindowedCache, WindowedCacheConfig, install_score_hooks)``
    for the requested backend.

    Parameters
    ----------
    backend : str
        Either ``'flash_attn'`` or ``'eager'``.

    Returns
    -------
    tuple[type, type, callable]
        The cache class, its config class, and the hook installer.

    Raises
    ------
    ValueError
        If *backend* is not recognized.
    ConfigValidationError
        If the required package is not installed (flash_attn only).

    Notes
    -----
    Imports are lazy: ``flash_attn`` is **never** imported on the eager
    path, so the eager backend runs without flash-attn installed.
    """
    if backend == "flash_attn":
        try:
            from modules.windowed_cache import (  # type: ignore[attr-defined]
                WindowedCache,
                WindowedCacheConfig,
                install_score_hooks,
            )
        except ImportError as e:
            raise ConfigValidationError(
                "flash_attn backend requested but modules.windowed_cache is not "
                "available.  Ensure Prompt 02 has been implemented."
            ) from e
        return WindowedCache, WindowedCacheConfig, install_score_hooks

    elif backend == "eager":
        try:
            from modules.windowed_eager_cache import (  # type: ignore[attr-defined]
                WindowedCache,
                WindowedCacheConfig,
                install_score_hooks,
            )
        except ImportError as e:
            raise ConfigValidationError(
                "eager backend requested but modules.windowed_eager_cache is not "
                "available.  Ensure Prompt 02 has been implemented."
            ) from e
        return WindowedCache, WindowedCacheConfig, install_score_hooks

    else:
        raise ConfigValidationError(
            f"Unknown cache backend: {backend!r}.  Must be 'flash_attn' or 'eager'."
        )


def quant_budget_mode_kwargs(cache_config_cls: Type, requested: str) -> dict:
    """``{"quant_budget_mode": requested}``, or ``{}`` if the backend lacks it.

    The flash package supports both splits of ``quant_ratio``; the eager package
    implements the byte split ONLY (``windowed_eager_cache/config.py:406-407``
    has no mode branch). So the kwarg has to be omitted for eager rather than
    passed and rejected as an unexpected argument.

    Omitting it is only safe when the two agree. ``'bytes'`` is what eager
    computes, so asking for it and dropping it is a no-op; asking for
    ``'tokens'`` and dropping it would run a DIFFERENT operating point under the
    config's name — at ``cache_budget=0.20, q=0.5`` the two differ by 2.2x in
    retained keys — so that RAISES.

    This asymmetry is also why the flash default went back to ``'bytes'``: from
    f71fec0 until now the flash backend defaulted to ``'tokens'`` while eager
    stayed on ``'bytes'``, so the two backends were silently resolving the same
    YAML to different caches. See ACCURACY_RECOVERY_PLAN.md §2.
    """
    if "quant_budget_mode" in getattr(cache_config_cls, "__dataclass_fields__", {}):
        return {"quant_budget_mode": requested}
    if requested != "bytes":
        raise ConfigValidationError(
            f"quant_budget_mode={requested!r} was requested, but "
            f"{cache_config_cls.__module__}.{cache_config_cls.__name__} has no "
            "such field — the eager cache package implements the BYTE split "
            "only. Dropping it silently would run a different operating point "
            "under this config's name (2.2x the retained keys at "
            "cache_budget=0.20, quant_ratio=0.5). Use the flash_attn backend "
            "for quant_budget_mode='tokens', or set it to 'bytes'."
        )
    return {}


def validate_backend_attn_pairing(
    backend: str, attn_implementation: str
) -> None:
    """Validate that the cache backend and attention implementation are compatible.

    Rules:
    - ``'flash_attn'`` backend requires ``'flash_attention_2'`` attention.
    - ``'eager'`` backend requires ``'eager'`` attention.

    Called before model load so a mismatched config fails fast.

    Raises
    ------
    ConfigValidationError
        On mismatch.
    """
    if backend not in _BACKEND_TO_ATTN_IMPL:
        raise ConfigValidationError(
            f"Unknown cache backend: {backend!r}.  Must be 'flash_attn' or 'eager'."
        )

    allowed = _BACKEND_TO_ATTN_IMPL[backend]
    if attn_implementation not in allowed:
        raise ConfigValidationError(
            f"Cache backend {backend!r} requires "
            f"attn_implementation in {allowed!r}, but got "
            f"{attn_implementation!r}.  Fix your config."
        )
