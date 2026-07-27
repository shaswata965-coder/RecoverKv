"""Pytest configuration for the GSM8K harness tests.

Registers the ``gpu`` marker so the weight-backed tests (the batched-vs-``B=1``
equivalence contract, chiefly) declare themselves instead of raising
``PytestUnknownMarkWarning``. They skip on their own when no model is configured, so
the marker is for selection (``-m "not gpu"``), not for correctness.
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: needs model weights and a CUDA device; set GSM8K_TEST_MODEL to run",
    )
