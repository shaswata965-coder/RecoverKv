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
    "ExternalCacheMethod",
    "get_cache_classes",
    "resolve_method_factory",
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
    if backend not in ("flash_attn", "eager"):
        raise ConfigValidationError(
            f"Unknown cache backend: {backend!r}.  Must be 'flash_attn' or 'eager'."
        )
    # ONE cache. The two backends differ only in where the eviction scores come
    # from -- flash recomputes the post-RoPE query and runs the fused kernel,
    # eager reads `attn_weights` off the module output -- and `install_score_hooks`
    # picks that from the attention implementation. They used to be two packages,
    # and the eager one drifted into a stale fork: 736 lines against 2494, with no
    # read gate, no fused decode, no layer-major eviction and no byte-split mode.
    # A backend that quietly runs a different cache is a different method.
    from modules.windowed_cache import (  # type: ignore[attr-defined]
        WindowedCache,
        WindowedCacheConfig,
        install_score_hooks,
    )
    return WindowedCache, WindowedCacheConfig, install_score_hooks


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


def quant_gate_ratio_kwargs(cache_config_cls: Type, requested: float) -> dict:
    """``{"quant_gate_ratio": requested}``, or ``{}`` if the backend lacks it.

    The read gate lives on the fused flash decode path only: the eager package
    dequantizes every active int2 window every step and has no gate to configure.
    So the kwarg is omitted there rather than passed and rejected.

    Unlike :func:`quant_budget_mode_kwargs`, dropping it does not silently change
    the operating point — it changes how much of the tier is *read*, not what the
    cache *keeps*, and eager's answer ("all of it") is the same at every ratio.
    So this omits rather than raising: an eager run is correct at any ratio, just
    ungated, and raising would break every existing eager config for a knob that
    could never have applied to it.

    What it does prevent is the reverse failure — the flash backend quietly
    ignoring a configured ratio because the caller forgot to thread it through,
    which is exactly how ``quant_budget_mode`` came to be inert in three runners
    at once (ACCURACY_RECOVERY_PLAN.md §2).
    """
    if "quant_gate_ratio" in getattr(cache_config_cls, "__dataclass_fields__", {}):
        return {"quant_gate_ratio": requested}
    return {}


# ---------------------------------------------------------------------------
# External methods (``cache.backend: external``)
# ---------------------------------------------------------------------------


def resolve_method_factory(spec: str):
    """Import a ``"package.module:callable"`` spec and return the callable.

    This is the seam that lets Suite C benchmark a method it knows nothing
    about. It matters more than it looks: the published efficiency protocols in
    this space are UNDER-SPECIFIED (papers report "peak memory and decoding
    latency" without stating context length, batch size, warmup rounds, dtype or
    prompt), so quoting a number out of a paper and putting it beside ours is not
    a controlled comparison. The only sound way to get a baseline is to run the
    baseline yourself, in-process, under the identical protocol -- which requires
    being able to plug one in.

    The factory is called as::

        factory(model=..., tokenizer=..., prefill_len=..., gen_len=...,
                batch_size=..., budget_tokens=..., dtype=..., **method_kwargs)

    and must return a *method handle* implementing:

        new_cache()                  -> a fresh ``past_key_values`` per run (required)
        install_hooks(model, cache)  -> object with .remove(), or None (optional)
        requires_output_attentions   -> bool attribute (optional, default False)
        describe()                   -> str for the npz metadata (optional)

    Nothing here touches this project's cache packages, so an external method
    needs no knowledge of them.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"method_factory must be 'package.module:callable', got {spec!r}")
    mod_name, _, attr = spec.partition(":")
    import importlib
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise ValueError(f"method_factory module {mod_name!r} is not importable: {e}") from e
    try:
        factory = getattr(mod, attr)
    except AttributeError as e:
        raise ValueError(f"method_factory {spec!r}: {mod_name} has no {attr!r}") from e
    if not callable(factory):
        raise ValueError(f"method_factory {spec!r} resolved to a non-callable")
    return factory


class ExternalCacheMethod:
    """A ``method_factory`` handle, as the quality runners drive it.

    The perf suite builds its external method inline (``perf_runner``); the
    GSM8K / LongBench / RULER runners share this instead, so the three stay one
    code path. Built once after the model loads; then per example
    :meth:`new_cache` returns ``(past_key_values, hooks_or_None)`` and
    :meth:`record` collects the cache's own memory report, if it has one, for
    the metadata sidecar.

    The runner does not size the method. A quality run's budget is the
    method's own business, stated in ``cache.method_kwargs`` — which is why
    :class:`~utils.config.CacheConfig` rejects ``cache_budget`` beside
    ``backend: external``: forwarded, it could only be forwarded as the
    eviction methods' token count, which a method that evicts nothing would
    have to reinterpret; ignored, it would be a knob that does nothing.
    """

    def __init__(self, cache_cfg: Any, model: Any, tokenizer: Any = None,
                 dtype: Any = None) -> None:
        import inspect

        spec = getattr(cache_cfg, "method_factory", None)
        if not spec:
            raise ConfigValidationError(
                "cache.backend='external' requires cache.method_factory: "
                "'package.module:callable'")
        factory = resolve_method_factory(spec)
        self.spec = spec
        self.kwargs = dict(getattr(cache_cfg, "method_kwargs", None) or {})
        self.method = factory(model=model, tokenizer=tokenizer, prefill_len=None,
                              gen_len=None, batch_size=1, budget_tokens=None,
                              dtype=dtype, **self.kwargs)
        if not hasattr(self.method, "new_cache"):
            raise TypeError(
                f"method_factory {spec!r} returned {type(self.method).__name__}, "
                "which has no new_cache() -- see resolve_method_factory")
        params = inspect.signature(self.method.new_cache).parameters
        self._takes_horizon = "max_new_tokens" in params
        self.model = model
        self._reports: list = []

    @property
    def requires_output_attentions(self) -> bool:
        return bool(getattr(self.method, "requires_output_attentions", False))

    def describe(self) -> str:
        d = getattr(self.method, "describe", None)
        return d() if d is not None else type(self.method).__name__

    def new_cache(self, max_new_tokens: Optional[int] = None):
        """``(past_key_values, hooks_or_None)`` for one example."""
        if self._takes_horizon:
            cache = self.method.new_cache(max_new_tokens=max_new_tokens)
        else:
            cache = self.method.new_cache()
        inst = getattr(self.method, "install_hooks", None)
        return cache, (inst(self.model, cache) if inst is not None else None)

    def record(self, cache: Any) -> Optional[dict]:
        """Keep this example's ``memory_report()``; return it for per-example stats."""
        report = getattr(cache, "memory_report", None)
        if report is None:
            return None
        r = report()
        self._reports.append(r)
        return r

    def clear(self) -> None:
        """Forget recorded reports (LongBench writes one sidecar per dataset)."""
        self._reports.clear()

    def summary(self) -> dict:
        """What the sidecar records: the method, as configured and as measured."""
        out: dict = {"method_factory": self.spec, "method_kwargs": self.kwargs,
                     "describe": self.describe()}
        meta = getattr(self.method, "metadata", None)
        if meta is not None:
            out["metadata"] = meta()
        if self._reports:
            n = len(self._reports)
            for key in ("kv_fraction", "total_fraction"):
                vals = [r[key] for r in self._reports if key in r]
                if vals:
                    out[f"mean_{key}"] = round(sum(vals) / len(vals), 6)
            out["n_memory_reports"] = n
        return out
