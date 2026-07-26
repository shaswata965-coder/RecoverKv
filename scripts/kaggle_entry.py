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
