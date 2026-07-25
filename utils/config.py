from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ParityValidationError(ValueError):
    """Raised when base and ours configs disagree on identicality fields."""


class ConfigValidationError(ValueError):
    """Raised when a config value is invalid."""


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str = "meta-llama/Meta-Llama-3-8B"
    revision: Optional[str] = None
    dtype: str = "float16"
    attn_implementation: str = "eager"  # "eager" | "flash_attention_2"


@dataclass
class CacheConfig:

    backend: str = "dynamic"
    backend_package: Optional[str] = None  # "flash_attn" | "eager" | None
    cache_budget: Optional[float] = None  # float ratio in (0, 1]; None for baseline
    window_size: int = 8
    num_sink_tokens: int = 4
    local_window_size: Union[int, float] = 0.25  # int (multiple of window_size) or ratio
    rerotate_on_evict: bool = False  # StreamingLLM-style key re-rotation on eviction (default off)
    quant_ratio: float = 0.0  # two-tier int2 split q in [0,1] (design.md §7); 0 disables the Q tier
    first_eviction_step: int = 8  # decode step of the FIRST eviction, independent of window_size
    # None = auto (memoize the dequantized Q tier at B=1, not above). The memo
    # costs ~149 MB/row vs ~131 MB/row of actual KV, so it halves max-B; it buys
    # ~8x fewer Q-tier dequants per step. Set explicitly to measure both sides.
    quant_memoize_read: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.cache_budget is not None:
            # Type guards mirror WindowedCacheConfig.__post_init__: reject
            # bool before int (bool subclasses int) and reject non-float
            # types up-front so users don't pass 40 meaning 40%.
            if isinstance(self.cache_budget, bool):
                raise ConfigValidationError(
                    f"cache_budget must be a float ratio in (0, 1], got bool "
                    f"{self.cache_budget!r}. bool is rejected because it "
                    f"subclasses int."
                )
            if isinstance(self.cache_budget, int):
                raise ConfigValidationError(
                    f"cache_budget must be a float ratio in (0, 1], got int "
                    f"{self.cache_budget}. Use e.g. 0.40 instead of 40."
                )
            if not isinstance(self.cache_budget, float):
                raise ConfigValidationError(
                    f"cache_budget must be a float ratio in (0, 1], got "
                    f"{type(self.cache_budget).__name__}"
                )
            if not (0.0 < self.cache_budget <= 1.0):
                raise ConfigValidationError(
                    f"cache_budget must be in (0, 1], got {self.cache_budget}"
                )

        if isinstance(self.local_window_size, bool):
            # bool is a subclass of int in Python; explicitly reject it
            # so True/False can't silently pass as 1/0.
            raise ConfigValidationError(
                f"local_window_size must be int or float, got bool {self.local_window_size!r}"
            )
        if isinstance(self.local_window_size, int):
            if self.local_window_size % self.window_size != 0:
                raise ConfigValidationError(
                    f"local_window_size as int ({self.local_window_size}) must be a "
                    f"multiple of window_size ({self.window_size})"
                )
        elif isinstance(self.local_window_size, float):
            if not (0.0 < self.local_window_size <= 1.0):
                raise ConfigValidationError(
                    f"local_window_size as float must be in (0, 1], "
                    f"got {self.local_window_size}"
                )

        # quant_ratio: two-tier int2 split (design.md §7). Full validation
        # (window_size % 4 when q>0, range) lives in WindowedCacheConfig; here
        # we only guard the obvious type/range so eval configs fail fast.
        if isinstance(self.quant_ratio, bool):
            raise ConfigValidationError("quant_ratio must be a float in [0, 1], got bool")
        if isinstance(self.quant_ratio, int):
            self.quant_ratio = float(self.quant_ratio)
        if not isinstance(self.quant_ratio, float) or not (0.0 <= self.quant_ratio <= 1.0):
            raise ConfigValidationError(
                f"quant_ratio must be a float in [0, 1], got {self.quant_ratio!r}"
            )

    def resolve_local_window_size(self, budget_tokens: int) -> int:
        """Resolve local_window_size to a concrete token count.

        A float ``local_window_size`` is a fraction of the CACHE BUDGET (not the
        full post-sink context), matching ``WindowedCacheConfig.resolve``, so the
        local region can never exceed the budget.  ``budget_tokens`` is the total
        token budget (``cache_budget * (prefill_len + max_tokens)``).  An int
        ``local_window_size`` is returned verbatim.
        """
        if isinstance(self.local_window_size, int):
            return self.local_window_size

        raw = self.local_window_size * budget_tokens
        ceiled = math.ceil(raw)
        # Snap upward to nearest multiple of window_size
        remainder = ceiled % self.window_size
        if remainder != 0:
            ceiled += self.window_size - remainder
        return ceiled


@dataclass
class DataConfig:

    dataset: str = "wikitext-103"  # "wikitext-103" | "pg19"
    article_id: int = 0
    prefill_len: int = 100
    gen_len: int = 50
    # Global knobs (apply to parity runners; LongBench has its own num_samples).
    num_samples: int = 1
    # Samples per teacher-forced forward pass. 1 (default) ⇒ sequential, which
    # is byte-identical to the legacy single-sample path. >1 batches that many
    # equal-length samples through one cache/forward for speed.
    batch_size: int = 1
    max_tokens: Optional[int] = None
    ratio: float = 1.0   # prefill fraction of max_tokens; 1-ratio is gen

    def __post_init__(self) -> None:
        if not (0.0 < self.ratio <= 1.0):
            raise ConfigValidationError(
                f"data.ratio must be in (0, 1], got {self.ratio!r}"
            )
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ConfigValidationError(
                f"data.max_tokens must be a positive int, got {self.max_tokens!r}"
            )
        if self.num_samples < 1:
            raise ConfigValidationError(
                f"data.num_samples must be >= 1, got {self.num_samples!r}"
            )
        if self.batch_size < 1:
            raise ConfigValidationError(
                f"data.batch_size must be >= 1, got {self.batch_size!r}"
            )

    def resolved_lengths(
        self, default_prefill: int, default_gen: int
    ) -> Tuple[int, int]:
        if self.max_tokens is None:
            return int(default_prefill), int(default_gen)
        eff_prefill = int(self.max_tokens * self.ratio)
        eff_prefill = max(1, eff_prefill)
        eff_gen = max(0, int(self.max_tokens) - eff_prefill)
        return eff_prefill, eff_gen


@dataclass
class TelemetryConfig:

    track_scores: bool = False
    output_dir: str = "outputs"


@dataclass
class RunConfig:
    """Top-level run configuration."""

    mode: str = "parity_base"
    seed: int = 42


# ---------------------------------------------------------------------------
# Parity-specific config
# ---------------------------------------------------------------------------


@dataclass
class ParityConfig:
    """Parity run configuration (Suite A)."""

    dataset: str = "wikitext-103"
    num_articles: int = 50
    article_index: int = 0
    min_article_tokens: int = 4096
    prefill_len: int = 2048
    gen_len: int = 1024
    decoding: str = "greedy"
    record_full_attention: bool = False
    full_attention_sample_rate: int = 10


@dataclass
class WindowConfig:

    window_size: int = 32
    num_sink_tokens: int = 4
    local_window_size: Union[int, float] = 256
    top_k_windows: Optional[int] = None

    def resolved_top_k(
        self, cache_budget: Optional[float], prefill_len: int, max_tokens: int
    ) -> int:
        if self.top_k_windows is not None:
            return int(self.top_k_windows)

        if cache_budget is None:
            raise ConfigValidationError(
                "Cannot derive top_k_windows: window.top_k_windows is unset and "
                "cache.cache_budget is None. Set cache.cache_budget to the target "
                "compression ratio (e.g., 0.25) — base parity runs use it as the "
                "comparison target even though they do not evict."
            )

        # Budget is sized against the full expected sequence (prefill + generation),
        # mirroring WindowedCacheConfig.resolve(), so the derived K matches the K the
        # production eviction policy actually keeps.
        budget_tokens = int(cache_budget * (prefill_len + max_tokens))

        # Resolve local_window_size to a concrete int (mirrors WindowedCacheConfig):
        # a float local_window_size is a fraction of the CACHE BUDGET (not the
        # post-sink context), so the local region can never exceed the budget.
        lws = self.local_window_size
        if isinstance(lws, float):
            raw = lws * budget_tokens
            ceiled = math.ceil(raw)
            remainder = ceiled % self.window_size
            if remainder:
                ceiled += self.window_size - remainder
            local_tokens = ceiled
        else:
            local_tokens = int(lws)

        # Mirrors WindowedCacheConfig.resolve(): sink + local over budget clamps
        # to zero evictable windows rather than raising. Clamped, not just
        # un-raised — a negative remaining floor-divides NEGATIVE and would hand
        # back a negative top_k. See that resolver for the full note; the cache
        # will over-retain against the requested budget and warns when it does.
        remaining = budget_tokens - self.num_sink_tokens - local_tokens
        if remaining < 0:
            log.warning(
                "cache_budget=%s on prefill_len=%s + max_tokens=%s yields "
                "budget_tokens=%s, below num_sink_tokens (%s) + local_tokens (%s). "
                "Proceeding with top_k_windows=0: sink + local alone retain %s "
                "tokens/row, exceeding the requested budget by %s tokens/row.",
                cache_budget, prefill_len, max_tokens, budget_tokens,
                self.num_sink_tokens, local_tokens,
                self.num_sink_tokens + local_tokens, -remaining,
            )
            remaining = 0
        return remaining // self.window_size


# ---------------------------------------------------------------------------
# Performance config
# ---------------------------------------------------------------------------


@dataclass
class PerfConfig:
    """Performance benchmark configuration (Suite C)."""

    configs: List[Dict[str, Any]] = field(default_factory=list)
    # ChunkKV-style (prefill_len, gen_len) grid. If empty, falls back to the
    # cartesian product of `prefill_lengths` x [`gen_len`].
    grid: List[Dict[str, int]] = field(default_factory=list)
    prefill_lengths: List[int] = field(default_factory=lambda: [2048, 4096])
    gen_len: int = 256
    batch_size: int = 1            # sequences per forward; >1 measures batched throughput
    num_warmup_runs: int = 2
    num_measurement_runs: int = 10
    allow_shared_gpu: bool = True
    skip_if_oom: bool = True
    skip_if_flash_attn_unavailable: bool = True
    enable_clock_locking: bool = False
    data_source: Optional[str] = None  # LongBench dataset name (e.g. "2wikimqa"); None = synthetic


# ---------------------------------------------------------------------------
# Faithfulness config
# ---------------------------------------------------------------------------


@dataclass
class FaithfulnessConfig:
    """Faithfulness evaluation configuration (Suite B)."""

    base_npz_path: str = ""
    ours_npz_path: str = ""


# ---------------------------------------------------------------------------
# Visualization config
# ---------------------------------------------------------------------------


@dataclass
class VisualizeConfig:
    """Visualization configuration."""

    npz_paths: List[str] = field(default_factory=list)
    parity_base_npz: str = ""
    parity_ours_npz: str = ""
    faithfulness_npz: str = ""
    perf_npz_dir: str = "outputs"
    output_dir: str = "outputs/figures"
    save_pdf: bool = False
    dpi: int = 300


# ---------------------------------------------------------------------------
# LongBench config
# ---------------------------------------------------------------------------


@dataclass
class LongBenchConfig:

    datasets: List[str] = field(
        default_factory=lambda: [
            "narrativeqa", "qasper", "multifieldqa_en",
            "hotpotqa", "2wikimqa", "musique",
            "gov_report", "qmsum", "multi_news",
            "trec", "triviaqa", "samsum",
            "passage_count", "passage_retrieval_en",
            "lcc", "repobench-p",
        ]
    )
    include_chinese: bool = False
    use_e_variants: bool = False    # LongBench-E length-stratified variants
    # Prompt truncation budget. A positive int reproduces official LongBench
    # middle-truncation for short-context models. None / 0 disables truncation
    # (full-context runs — matches DefensiveKV, which uses Llama-3.1-8B's 128K
    # window and does NOT pre-truncate). Generation lengths are separate and
    # come per-dataset from dataset2maxlen.json.
    max_length: Optional[int] = 7500
    output_dir: str = "outputs/longbench/full_cache"
    seed: int = 42
    resume: bool = False            # skip datasets whose jsonl already exists
    skip_oom: bool = False          # record OOM'd examples as pred=null
    aggressive_cache_clear: bool = False  # essential on Kaggle T4
    num_samples: Union[int, str] = "max"  # "max" = all examples; int = cap per dataset

    def __post_init__(self) -> None:
        ns = self.num_samples
        if isinstance(ns, bool):
            # bool is a subclass of int in Python; explicitly reject it.
            raise ConfigValidationError(
                f"longbench.num_samples must be 'max' or a non-negative int, "
                f"got bool {ns!r}"
            )
        if isinstance(ns, str):
            if ns.strip().lower() != "max":
                raise ConfigValidationError(
                    f"longbench.num_samples string must be 'max', got {ns!r}"
                )
            # Normalise so downstream code can do a literal comparison
            self.num_samples = "max"
        elif isinstance(ns, int):
            if ns < 0:
                raise ConfigValidationError(
                    f"longbench.num_samples int must be >= 0, got {ns!r}"
                )
        else:
            raise ConfigValidationError(
                f"longbench.num_samples must be 'max' or a non-negative int, "
                f"got {type(ns).__name__}: {ns!r}"
            )


# ---------------------------------------------------------------------------
# RULER config
# ---------------------------------------------------------------------------


def _validate_num_samples(ns: Union[int, str], field_label: str) -> Union[int, str]:
    """Shared 'max' | non-negative-int validation for num_samples fields."""
    if isinstance(ns, bool):
        # bool is a subclass of int in Python; explicitly reject it.
        raise ConfigValidationError(
            f"{field_label} must be 'max' or a non-negative int, got bool {ns!r}"
        )
    if isinstance(ns, str):
        if ns.strip().lower() != "max":
            raise ConfigValidationError(f"{field_label} string must be 'max', got {ns!r}")
        return "max"
    if isinstance(ns, int):
        if ns < 0:
            raise ConfigValidationError(f"{field_label} int must be >= 0, got {ns!r}")
        return ns
    raise ConfigValidationError(
        f"{field_label} must be 'max' or a non-negative int, got {type(ns).__name__}: {ns!r}"
    )


@dataclass
class RulerConfig:

    data_dir: str = ""  # e.g. .../defensivekv_dataset/ruler/4096
    tasks: Optional[List[str]] = None  # None = all 13 RULER tasks
    output_dir: str = "outputs/ruler"
    seed: int = 42
    resume: bool = False            # skip tasks whose jsonl already exists
    skip_oom: bool = False          # record OOM'd examples as pred=null
    aggressive_cache_clear: bool = False
    num_samples: Union[int, str] = "max"  # cap per task, applied after task filtering
    # Capture a utils/cache_memory.py byte-accounting report per example and
    # write <task>.memory.jsonl + <task>.memory_summary.json alongside the
    # predictions. Cheap (introspects existing tensors, allocates nothing).
    capture_memory: bool = False

    def __post_init__(self) -> None:
        self.num_samples = _validate_num_samples(self.num_samples, "ruler.num_samples")


# ---------------------------------------------------------------------------
# ExperimentConfig (top-level)
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    data: DataConfig = field(default_factory=DataConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    parity: ParityConfig = field(default_factory=ParityConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
    faithfulness: FaithfulnessConfig = field(default_factory=FaithfulnessConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)
    longbench: LongBenchConfig = field(default_factory=LongBenchConfig)
    ruler: RulerConfig = field(default_factory=RulerConfig)

    # Paths
    base_run_npz: Optional[str] = None
    output_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def _dict_to_config(d: dict[str, Any]) -> ExperimentConfig:
    """Convert a flat/nested dict to an ``ExperimentConfig``."""
    # Parse perf configs list if present
    perf_raw = d.get("perf", {})
    perf_configs_raw = perf_raw.get("configs", [])
    perf_kwargs = {k: v for k, v in perf_raw.items() if k != "configs"}
    perf_kwargs["configs"] = perf_configs_raw if perf_configs_raw else []

    # Parse visualize npz_paths
    vis_raw = d.get("visualize", {})

    # Parse longbench config
    lb_raw = d.get("longbench", {})

    # Parse ruler config
    ruler_raw = d.get("ruler", {})

    return ExperimentConfig(
        run=RunConfig(**d.get("run", {})),
        model=ModelConfig(**d.get("model", {})),
        cache=CacheConfig(**d.get("cache", {})),
        data=DataConfig(**d.get("data", {})),
        telemetry=TelemetryConfig(**d.get("telemetry", {})),
        parity=ParityConfig(**d.get("parity", {})),
        window=WindowConfig(**d.get("window", {})),
        perf=PerfConfig(**perf_kwargs),
        faithfulness=FaithfulnessConfig(**d.get("faithfulness", {})),
        visualize=VisualizeConfig(**vis_raw),
        longbench=LongBenchConfig(**lb_raw),
        ruler=RulerConfig(**ruler_raw),
        base_run_npz=d.get("base_run_npz"),
        output_path=d.get("output_path"),
    )


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config file and return a typed ``ExperimentConfig``.

    If the YAML contains a ``_base_`` key, that file is loaded first and
    the current file's values are merged on top (single-level inheritance).

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.
    overrides : dict, optional
        Additional key-value overrides applied after file loading.

    Returns
    -------
    ExperimentConfig
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Handle single-level config inheritance
    if "_base_" in raw:
        base_path = path.parent / raw.pop("_base_")
        with open(base_path, "r") as f:
            base_raw = yaml.safe_load(f) or {}
        raw = _merge_dicts(base_raw, raw)

    if overrides:
        raw = _merge_dicts(raw, overrides)

    config = _dict_to_config(raw)
    log.info("Loaded config from %s (mode=%s)", path, config.run.mode)
    return config


# ---------------------------------------------------------------------------
# Cross-config validator
# ---------------------------------------------------------------------------

# Fields that MUST be identical between a base run and an ours run
_PARITY_IDENTITY_FIELDS = [
    "seed",
    "dataset",
    "article_id",
    "article_sha",
    "prefill_len",
    "gen_len",
    "window_size",
    "num_sink_tokens",
    "local_window_size_resolved",
    "model_name",
    "model_revision",
    "tokenizer_sha",
    "transformers_version",
]


def validate_parity_pair(
    base_meta: dict[str, Any],
    ours_config: ExperimentConfig,
) -> None:
    eff_prefill, eff_gen = ours_config.data.resolved_lengths(
        ours_config.parity.prefill_len, ours_config.parity.gen_len
    )

    # Build a comparable dict from ours_config
    ours_flat: dict[str, Any] = {
        "seed": ours_config.run.seed,
        "dataset": ours_config.parity.dataset,
        "article_id": ours_config.parity.article_index,
        "prefill_len": eff_prefill,
        "gen_len": eff_gen,
        "window_size": ours_config.window.window_size,
        "num_sink_tokens": ours_config.window.num_sink_tokens,
        "model_name": ours_config.model.name,
        "model_revision": ours_config.model.revision,
    }

    # Fields checked at runtime only (not in config)
    runtime_fields = {"tokenizer_sha", "transformers_version",
                      "article_sha", "local_window_size_resolved"}

    mismatches: list[str] = []
    for field_name in _PARITY_IDENTITY_FIELDS:
        if field_name in runtime_fields:
            # These are checked at runtime from the base npz metadata
            # against the actual runtime values, not from ours_config
            continue
        base_val = base_meta.get(field_name)
        ours_val = ours_flat.get(field_name)
        if base_val is None or ours_val is None:
            log.warning(
                "Parity identicality field %r missing on one side "
                "(base=%r, ours=%r) — skipping comparison.",
                field_name, base_val, ours_val,
            )
            continue
        if base_val != ours_val:
            mismatches.append(
                f"  {field_name}: base={base_val!r}, ours={ours_val!r}"
            )

    if mismatches:
        detail = "\n".join(mismatches)
        raise ParityValidationError(
            f"Parity validation failed — identicality fields differ:\n{detail}"
        )
