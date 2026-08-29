"""Logging setup.

Configuration happens when an entry point calls ``setup_logging`` -- never at
import time (Rules.md section 4).  Library modules just call
``logging.getLogger(__name__)`` and inherit whatever the entry point set up.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger for a CLI or API process.

    Safe to call more than once; existing handlers are replaced so a second call
    does not double every line.

    Args:
        level: Threshold name, e.g. ``"DEBUG"`` or ``"INFO"``.
        log_file: Optional file to tee output into.  Its parent is created.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(level.upper())
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for ``name``."""
    return logging.getLogger(name)
