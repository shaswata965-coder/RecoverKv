"""GSM8K chain-of-thought evaluation for kvpress presses (SnapKV / AdaKV / DefensiveKV).

Layout mirrors the other benchmarks in ``evaluation/``:

    create_huggingface_dataset.py   build the CoT prompt set (+ manifest / dataset_sha)
    calculate_metrics.py            scoring, diagnostics, budget-comparison table
    press_budget.py                 what a compression_ratio actually buys on a short prompt
    pipeline.py                     kvpress pipeline + stop strings + cache telemetry
    run_gsm8k.py                    the runner (one press, one compression ratio, one run dir)
    run_gsm8k_e2e.sh                baseline -> gate -> press sweep -> comparison

Run everything from the ``evaluation/`` directory so ``gsm8k.*`` is importable, the
same convention ``evaluate.py`` uses for ``longbench.*`` / ``ruler.*``.
"""
