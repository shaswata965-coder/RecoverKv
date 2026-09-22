"""Unified logging with optional Weights & Biases integration.

Usage:
    from utils.logger import get_logger

    log = get_logger(__name__)
    log.info("Training started", extra={"epoch": 1})
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

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
# Weights & Biases wrapper (graceful fallback)
# ---------------------------------------------------------------------------

_wandb_available: Optional[bool] = None


def _check_wandb() -> bool:
    """Return True if wandb can be imported, caching the result."""
    global _wandb_available
    if _wandb_available is None:
        try:
            import wandb  # noqa: F401

            _wandb_available = True
        except ImportError:
            _wandb_available = False
    return _wandb_available


def init_wandb(project: str, config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    """Initialize a wandb run if available; returns the run object or None.

    Parameters
    ----------
    project : str
        W&B project name.
    config : dict, optional
        Run configuration to log.
    **kwargs
        Forwarded to ``wandb.init``.
    """
    if not _check_wandb():
        log = get_logger("utils.logger")
        log.warning("wandb not installed — skipping W&B initialization")
        return None

    import wandb

    return wandb.init(project=project, config=config, **kwargs)


def log_wandb(data: dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to wandb if a run is active; no-op otherwise."""
    if not _check_wandb():
        return
    import wandb

    if wandb.run is not None:
        wandb.log(data, step=step)


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
