"""Device resolution.

Resolved once at an entry point and threaded through.  Nothing else in the
package calls ``.cuda()`` or checks availability (Architecture.md section 1).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(preference: str | None = None) -> torch.device:
    """Pick a compute device, falling back to CPU rather than failing.

    CPU is this project's supported target, so an unavailable accelerator is a
    warning and never an error (Rules.md section 5).

    Args:
        preference: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``None``/``"auto"`` to
            pick the best available.

    Returns:
        The device to use.

    Raises:
        ValueError: If ``preference`` names a device this code does not know.
    """
    if preference in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            logger.warning("cuda requested but unavailable; falling back to cpu")
            return torch.device("cpu")
        return torch.device("cuda")

    if preference == "mps":
        if not torch.backends.mps.is_available():
            logger.warning("mps requested but unavailable; falling back to cpu")
            return torch.device("cpu")
        return torch.device("mps")

    raise ValueError(f"unknown device preference {preference!r}; expected auto/cpu/cuda/mps")


def describe_device(device: torch.device) -> str:
    """Return a human-readable description, for logs and the API /health route."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type
