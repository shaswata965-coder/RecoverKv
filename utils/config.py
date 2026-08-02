from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from utils.logger import get_logger

log = get_logger(__name__)


class ParityValidationError(ValueError):
    pass


class ConfigValidationError(ValueError):
    pass


FIRST_EVICTION_STEP_DEFAULT = 0


def log_operating_point(config, is_windowed: bool) -> None:
    cache = getattr(config, "cache", None)
    if cache is None:
        return
    if not is_windowed:
        log.info("operating point: FULL CACHE (no eviction, no quantization)")
        return

    q = getattr(cache, "quant_ratio", 0.0)
    promote = getattr(cache, "quant_promotion", True)
    log.info(
        "operating point: budget=%s  window_size=%s  num_sink=%s  local=%s  "
        "quant_ratio=%s (%s)  first_eviction_step=%s (%s)  backend=%s/%s",
        getattr(cache, "cache_budget", None),
        getattr(cache, "window_size", None),
        getattr(cache, "num_sink_tokens", None),
        getattr(cache, "local_window_size", None),
        q,
        ("two-tier fp16+int2"
         + ("" if promote else ", STICKY-Q: demotion is one-way"))
        if q > 0 else "SINGLE-TIER fp16, Q tier disabled",
        getattr(cache, "first_eviction_step", None),
        "prompt compressed on decode step 0"
        if getattr(cache, "first_eviction_step", 0) == 0
        else "DELAYED — short answers measured at full cache",
        getattr(cache, "backend_package", None),
        getattr(getattr(config, "model", None), "attn_implementation", None),
    )


@dataclass
class ModelConfig:
    name: str = "meta-llama/Meta-Llama-3-8B"
    revision: Optional[str] = None
    dtype: str = "float16"
    attn_implementation: str = "eager"


@dataclass
class CacheConfig:

    backend: str = "dynamic"
    backend_package: Optional[str] = None
    cache_budget: Optional[float] = None
    window_size: int = 8
    num_sink_tokens: int = 4
    local_window_size: Union[int, float] = 0.25
    rerotate_on_evict: bool = False
    quant_ratio: float = 0.0
    first_eviction_step: int = FIRST_EVICTION_STEP_DEFAULT
    quant_memoize_read: Optional[bool] = None
    quant_promotion: bool = True

    def __post_init__(self) -> None:
        if self.cache_budget is not None:
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

        if isinstance(self.quant_ratio, bool):
            raise ConfigValidationError("quant_ratio must be a float in [0, 1], got bool")
        if isinstance(self.quant_ratio, int):
            self.quant_ratio = float(self.quant_ratio)
        if not isinstance(self.quant_ratio, float) or not (0.0 <= self.quant_ratio <= 1.0):
            raise ConfigValidationError(
                f"quant_ratio must be a float in [0, 1], got {self.quant_ratio!r}"
            )

        if not isinstance(self.quant_promotion, bool):
            raise ConfigValidationError(
                f"quant_promotion must be a bool, got {self.quant_promotion!r}"
            )

    def resolve_local_window_size(self, budget_tokens: int) -> int:
        if isinstance(self.local_window_size, int):
            return self.local_window_size

        raw = self.local_window_size * budget_tokens
        ceiled = math.ceil(raw)
        remainder = ceiled % self.window_size
        if remainder != 0:
            ceiled += self.window_size - remainder
        return ceiled


@dataclass
class DataConfig:

    dataset: str = "wikitext-103"
    article_id: int = 0
    prefill_len: int = 100
    gen_len: int = 50
    num_samples: int = 1
    batch_size: int = 1
    max_tokens: Optional[int] = None
    ratio: float = 1.0

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

    mode: str = "parity_base"
    seed: int = 42


@dataclass
class ParityConfig:

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

        budget_tokens = int(cache_budget * (prefill_len + max_tokens))

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


@dataclass
class PerfConfig:

    configs: List[Dict[str, Any]] = field(default_factory=list)
    grid: List[Dict[str, int]] = field(default_factory=list)
    prefill_lengths: List[int] = field(default_factory=lambda: [2048, 4096])
    gen_len: int = 256
    batch_size: int = 1
    num_warmup_runs: int = 2
    num_measurement_runs: int = 10
    allow_shared_gpu: bool = True
    skip_if_oom: bool = True
    skip_if_flash_attn_unavailable: bool = True
    enable_clock_locking: bool = False
    data_source: Optional[str] = None


@dataclass
class FaithfulnessConfig:

    base_npz_path: str = ""
    ours_npz_path: str = ""


@dataclass
class TierStudyConfig:

    r0_npz: str = ""
    r1_npz: str = ""
    r2_npz: str = ""
    r3_npz: str = ""
    r4_npz: str = ""
    fmm_horizon: int = 32
    sticky_primary: bool = False
    trace_axis: str = "sample_layer"
    max_samples: Optional[int] = None
    layer_stride: int = 1
    head_stride: int = 1
    per_head: bool = True
    confidence: float = 0.95
    bootstrap_samples: int = 2000


@dataclass
class VisualizeConfig:

    npz_paths: List[str] = field(default_factory=list)
    parity_base_npz: str = ""
    parity_ours_npz: str = ""
    faithfulness_npz: str = ""
    perf_npz_dir: str = "outputs"
    output_dir: str = "outputs/figures"
    save_pdf: bool = False
    dpi: int = 300


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
    use_e_variants: bool = False
    max_length: Optional[int] = 7500
    output_dir: str = "outputs/longbench/full_cache"
    seed: int = 42
    resume: bool = False
    skip_oom: bool = False
    aggressive_cache_clear: bool = False
    num_samples: Union[int, str] = "max"

    def __post_init__(self) -> None:
        ns = self.num_samples
        if isinstance(ns, bool):
            raise ConfigValidationError(
                f"longbench.num_samples must be 'max' or a non-negative int, "
                f"got bool {ns!r}"
            )
        if isinstance(ns, str):
            if ns.strip().lower() != "max":
                raise ConfigValidationError(
                    f"longbench.num_samples string must be 'max', got {ns!r}"
                )
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


def _validate_num_samples(ns: Union[int, str], field_label: str) -> Union[int, str]:
    if isinstance(ns, bool):
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

    data_dir: str = ""
    tasks: Optional[List[str]] = None
    output_dir: str = "outputs/ruler"
    seed: int = 42
    resume: bool = False
    skip_oom: bool = False
    aggressive_cache_clear: bool = False
    num_samples: Union[int, str] = "max"
    capture_memory: bool = False

    def __post_init__(self) -> None:
        self.num_samples = _validate_num_samples(self.num_samples, "ruler.num_samples")


@dataclass
class GSM8KConfig:

    data_dir: str = "data/gsm8k_cot"
    output_dir: str = "outputs/gsm8k/run"
    results_dir: str = "outputs/gsm8k"
    num_samples: Union[int, str] = "max"
    shard: int = 0
    num_shards: int = 1
    skip_oom: bool = False
    aggressive_cache_clear: bool = False
    max_new_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        self.num_samples = _validate_num_samples(self.num_samples, "gsm8k.num_samples")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ConfigValidationError(
                f"gsm8k.max_new_tokens must be >= 1 or None, got {self.max_new_tokens!r}"
            )
        if self.num_shards < 1:
            raise ConfigValidationError(
                f"gsm8k.num_shards must be >= 1, got {self.num_shards!r}"
            )
        if not (0 <= self.shard < self.num_shards):
            raise ConfigValidationError(
                f"gsm8k.shard must be in [0, {self.num_shards}), got {self.shard!r}"
            )


@dataclass
class ExperimentConfig:

    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    data: DataConfig = field(default_factory=DataConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    parity: ParityConfig = field(default_factory=ParityConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
    faithfulness: FaithfulnessConfig = field(default_factory=FaithfulnessConfig)
    tier_study: TierStudyConfig = field(default_factory=TierStudyConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)
    longbench: LongBenchConfig = field(default_factory=LongBenchConfig)
    ruler: RulerConfig = field(default_factory=RulerConfig)
    gsm8k: GSM8KConfig = field(default_factory=GSM8KConfig)

    base_run_npz: Optional[str] = None
    output_path: Optional[str] = None


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def _dict_to_config(d: dict[str, Any]) -> ExperimentConfig:
    perf_raw = d.get("perf", {})
    perf_configs_raw = perf_raw.get("configs", [])
    perf_kwargs = {k: v for k, v in perf_raw.items() if k != "configs"}
    perf_kwargs["configs"] = perf_configs_raw if perf_configs_raw else []

    vis_raw = d.get("visualize", {})

    lb_raw = d.get("longbench", {})

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
        tier_study=TierStudyConfig(**d.get("tier_study", {})),
        visualize=VisualizeConfig(**vis_raw),
        longbench=LongBenchConfig(**lb_raw),
        ruler=RulerConfig(**ruler_raw),
        gsm8k=GSM8KConfig(**d.get("gsm8k", {})),
        base_run_npz=d.get("base_run_npz"),
        output_path=d.get("output_path"),
    )


def _load_raw_with_bases(
    path: Path, _seen: Optional[List[Path]] = None
) -> dict[str, Any]:
    path = path.resolve()
    seen = list(_seen or [])
    if path in seen:
        chain = " -> ".join(p.name for p in seen + [path])
        raise ConfigValidationError(f"Cyclic config inheritance: {chain}")
    seen.append(path)

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw

    base_path = path.parent / base_ref
    if not base_path.exists():
        raise FileNotFoundError(
            f"{path.name} declares _base_: {base_ref}, which does not exist "
            f"({base_path})"
        )
    return _merge_dicts(_load_raw_with_bases(base_path, seen), raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_raw_with_bases(path)

    if overrides:
        raw = _merge_dicts(raw, overrides)

    config = _dict_to_config(raw)
    log.info("Loaded config from %s (mode=%s)", path, config.run.mode)
    return config


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

    runtime_fields = {"tokenizer_sha", "transformers_version",
                      "article_sha", "local_window_size_resolved"}

    mismatches: list[str] = []
    for field_name in _PARITY_IDENTITY_FIELDS:
        if field_name in runtime_fields:
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
