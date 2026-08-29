"""Phase 0: seeding and device resolution."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from lstm_nlp.utils.device import describe_device, resolve_device
from lstm_nlp.utils.seed import make_generator, set_seed


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_gives_same_draws() -> None:
    """The floor of the PRD S10 reproducibility guarantee."""
    set_seed(42)
    first = _draw()
    set_seed(42)
    assert _draw() == first


def test_different_seed_gives_different_draws() -> None:
    set_seed(42)
    first = _draw()
    set_seed(43)
    assert _draw() != first


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)


def test_generator_is_reproducible_and_independent() -> None:
    """Samplers take an explicit generator so they never touch the global RNG."""
    a = torch.rand(5, generator=make_generator(7))
    b = torch.rand(5, generator=make_generator(7))
    assert torch.equal(a, b)
    assert make_generator(None) is None


def test_resolve_device_cpu_always_available() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    assert resolve_device(None).type in {"cpu", "cuda", "mps"}


def test_unavailable_accelerator_falls_back_not_raises() -> None:
    """CPU is the supported target: a missing accelerator is a warning."""
    assert resolve_device("cuda").type in {"cpu", "cuda"}
    assert resolve_device("mps").type in {"cpu", "mps"}


def test_unknown_device_rejected() -> None:
    with pytest.raises(ValueError, match="unknown device preference"):
        resolve_device("tpu")


def test_describe_device_returns_string() -> None:
    assert isinstance(describe_device(torch.device("cpu")), str)
