"""StickyKV — Single entry point.

Parses ``--config`` (path to YAML) and routes to the appropriate Runner
class based on ``config.run.mode``. No business logic lives here.

Usage:
    python main.py --config configs/eval_parity_base.yaml
    python main.py --config configs/longbench_full_cache.yaml
"""

from __future__ import annotations

import argparse
import sys

from utils.config import load_config
from utils.logger import get_logger
from utils.seed import seed_everything

log = get_logger("main")
_RUNNER_REGISTRY: dict[str, str] = {
    "parity_base": "modules.evaluation.base_parity_runner.BaseParityRunner",
    "parity_ours": "modules.evaluation.ours_parity_runner.OursParityRunner",
    "faithfulness": "modules.evaluation.faithfulness_runner.FaithfulnessRunner",
    "qevict_observations": (
        "modules.evaluation.qevict_observations.QEvictObservationRunner"),
    "tier_study": "modules.evaluation.tier_study.combined.TierStudyRunner",
    "perf": "modules.evaluation.perf_runner.PerfRunner",
    "longbench": "modules.evaluation.longbench_runner.LongBenchRunner",
    "longbench_score": "modules.evaluation.longbench_scoring.LongBenchScorer",
    "ruler": "modules.evaluation.ruler_runner.RulerRunner",
    "ruler_score": "modules.evaluation.ruler_scoring.RulerScorer",
    "gsm8k": "modules.evaluation.gsm8k_runner.GSM8KRunner",
    "gsm8k_score": "modules.evaluation.gsm8k_scoring.GSM8KScorer",
    "visualize": "modules.evaluation.visualize.VisualizeRunner",
}


def _import_runner(mode: str):
    if mode not in _RUNNER_REGISTRY:
        available = ", ".join(sorted(_RUNNER_REGISTRY.keys()))
        log.error("Unknown mode: %r.  Available: %s", mode, available)
        sys.exit(1)

    dotted_path = _RUNNER_REGISTRY[mode]
    module_path, class_name = dotted_path.rsplit(".", 1)

    try:
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        log.error(
            "Failed to import runner for mode %r (%s): %s",
            mode,
            dotted_path,
            e,
        )
        sys.exit(1)


def _parse_value(val: str):
    """Try to parse a CLI override value as list, int, float, bool, or string.

    Lists use an explicit bracket syntax -- ``[4096,8192]``, ``[]`` -- rather than
    bare commas, so a value that legitimately contains a comma (a prompt string,
    say) is never silently split into a list.
    """
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(item) for item in inner.split(",")]
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() == "none":
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StickyKV — Windowed KV-Cache Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help=("Key=value overrides (e.g. run.seed=123 data.prefill_len=200). "
              "Lists use brackets: perf.prefill_lengths=[4096,8192], perf.grid=[]."),
    )
    args = parser.parse_args()

    # Parse dot-notation overrides into a nested dict
    overrides: dict = {}
    for item in args.override:
        if "=" not in item:
            log.error("Override must be key=value, got: %r", item)
            sys.exit(1)
        key, val = item.split("=", 1)
        parts = key.split(".")
        d = overrides
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # Try to parse as int/float/bool
        d[parts[-1]] = _parse_value(val)

    # Load config
    config = load_config(args.config, overrides=overrides if overrides else None)

    # Seed everything
    seed_everything(config.run.seed)
    log.info("Seed set to %d", config.run.seed)

    # Import and run
    RunnerClass = _import_runner(config.run.mode)
    log.info("Running mode: %s", config.run.mode)

    runner = RunnerClass(config)
    runner.run()


if __name__ == "__main__":
    main()
