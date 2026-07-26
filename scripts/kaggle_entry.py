"""Kaggle convenience entry point.

Thin wrapper that maps --suite name to the appropriate config and invokes
main.py. No logic replicated — everything routes through main.py.

Usage in Kaggle notebook cells:
    !python scripts/kaggle_entry.py --suite parity_base
    !python scripts/kaggle_entry.py --suite parity_ours
    !python scripts/kaggle_entry.py --suite faithfulness
    !python scripts/kaggle_entry.py --suite qevict_observations
    !python scripts/kaggle_entry.py --suite perf
    !python scripts/kaggle_entry.py --suite visualize
    !python scripts/kaggle_entry.py --suite ruler
    !python scripts/kaggle_entry.py --suite gsm8k_budget20

Scoring is post-hoc and runs off the written predictions, so it goes through the
scorer modules rather than a suite:

    !python -m modules.evaluation.gsm8k_scoring --runs outputs/gsm8k/*
    !python -m modules.evaluation.ruler_scoring --predictions_dir outputs/ruler/...
    !python scripts/kaggle_entry.py --suite longbench_ours --override \\
        run.mode=longbench_score

Every generation suite here runs at the repo-default first_eviction_step = 0:
the prompt is compressed on decode step 0, before that step's query attends. To
run the delayed-eviction ablation instead, pass it explicitly:

    !python scripts/kaggle_entry.py --suite longbench_ours \\
        --override cache.first_eviction_step=8
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

_SUITE_TO_CONFIG = {
    "parity_base": "configs/eval_parity_base.yaml",
    "parity_ours": "configs/eval_parity_ours_eager.yaml",
    "parity_ours_eager": "configs/eval_parity_ours_eager.yaml",
    "parity_ours_flash": "configs/eval_parity_ours_flash.yaml",
    "faithfulness": "configs/eval_faithfulness.yaml",
    "qevict_observations": "configs/eval_qevict.yaml",
    "perf": "configs/eval_perf.yaml",
    "visualize": "configs/eval_visualize.yaml",
    "longbench_full": "configs/longbench_full_cache.yaml",
    "longbench_ours": "configs/longbench_ours_eager.yaml",
    "longbench_ours_eager": "configs/longbench_ours_eager.yaml",
    "longbench_ours_flash": "configs/longbench_ours_flash_attn.yaml",
    "longbench_ours_step0": "configs/longbench_ours_step0.yaml",
    # RULER (needs ruler.data_dir pointing at an unpacked RULER length bucket)
    "ruler": "configs/ruler_niah_mk3_omega16.yaml",
    # GSM8K CoT — generation then scoring. Build the corpus first with
    #   python -m data.gsm8k_loader --out data/gsm8k_cot
    "gsm8k_full": "configs/gsm8k_full_cache.yaml",
    "gsm8k_budget20": "configs/gsm8k_budget20.yaml",
    "gsm8k_budget80": "configs/gsm8k_budget80.yaml",
}

def main():
    parser = argparse.ArgumentParser(description="StickyKV Kaggle entry point")
    parser.add_argument("--suite", required=True, choices=list(_SUITE_TO_CONFIG.keys()),
                        help="Which evaluation suite to run.")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra key=value overrides passed to main.py.")
    args = parser.parse_args()

    config_path = _SUITE_TO_CONFIG[args.suite]

    # Find project root (parent of scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Set deterministic env vars
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    cmd = [
        sys.executable, os.path.join(project_root, "main.py"),
        "--config", os.path.join(project_root, config_path),
    ]
    if args.override:
        cmd.extend(["--override"] + args.override)

    print(f"[kaggle_entry] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
