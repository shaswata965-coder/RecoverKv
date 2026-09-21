"""Unified logging.

Usage:
    from utils.logger import get_logger

    log = get_logger(__name__)
    log.info("Training started", extra={"epoch": 1})
"""

from __future__ import annotations

import logging
import sys

# ---------------------------------------------------------------------------
# Console handler setup
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_root_configured = False


def _configure_root_logger() -> None:
    """Configure the root logger once (idempotent)."""
    global _root_configured
    if _root_configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    _root_configured = True


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Return a named logger with console output configured.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    level : int
        Logging level for this specific logger.
    """
    _configure_root_logger()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
