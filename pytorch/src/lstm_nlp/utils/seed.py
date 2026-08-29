"""Seeding, for the reproducibility guarantee in PRD S10.

The split seed and the training seed are deliberately separate: the split seed is
pinned at 10 to stay comparable with the frozen TensorFlow reference, while the
training seed is free to vary.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` so a run is reproducible.

    Args:
        seed: Non-negative seed applied to all three generators.
        deterministic: Ask cuDNN for deterministic kernels.  Has no effect on
            CPU, which is this project's supported target.  On GPU the cuDNN
            LSTM kernel remains non-deterministic regardless; that caveat is
            documented rather than worked around (Rules.md section 7).

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int | None) -> torch.Generator | None:
    """Build an explicit RNG for sampling, or ``None`` to use the global one.

    Library code must never reach for the global RNG (Rules.md section 4), so
    samplers take a generator argument instead.  Passing a seed here is what
    makes generation reproducible (PRD FR-23).
    """
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
