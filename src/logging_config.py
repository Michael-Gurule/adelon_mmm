"""Centralized logging configuration for Adelon MMM."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure root logger with structured format.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to file in addition
            to stderr.

    Raises:
        ValueError: If level is not a valid logging level.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % level)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
