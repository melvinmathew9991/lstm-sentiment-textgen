"""Phase 9: temperature scaling, and the property that makes it safe.

The load-bearing claim is that calibration **cannot change a decision**.
Dividing logits by a positive scalar is monotonic, so ``argmax`` is fixed and
every threshold-0.5 metric is bit-identical before and after. That is what let
this land without re-measuring a single figure in ``PARITY.md``, so it is
asserted here rather than argued in a docstring.

Measured on the shipped model: validation ECE 0.0668 -> 0.0198 at T = 2.6715,
and on the test block the fit never saw, 0.0803 -> 0.0223. Zero of 3,382
decisions changed.

This docstring read 0.0558 -> 0.0313 and 0.0609 -> 0.0324 until 2026-08-30 --
the v1.0.0 fit, superseded when deduplication changed the corpus and the
temperature was refitted. Same numbers, same omission, as ``PARITY.md`` 6.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lstm_nlp.engine.calibration import (
    ECE_BINS,
    MAX_TEMPERATURE,
    calibrate,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_table,
)
from lstm_nlp.errors import DataError


def synthetic_logits(n: int = 4000, scale: float = 3.0, seed: int = 0):
    """Over-confident two-class logits and their labels.

    ``scale`` inflates the margins, which is what an over-confident model does:
    the ranking is informative but the magnitudes overstate it. That is the
    situation ``class_weighting: balanced`` produces on this corpus.
    """
    generator = torch.Generator().manual_seed(seed)
    truth = torch.randint(0, 2, (n,), generator=generator)
    signal = torch.randn(n, generator=generator) * 0.8 + (truth.float() - 0.5) * 2.0
    logits = torch.stack([-signal * scale, signal * scale], dim=1)
    return logits, truth


# --------------------------------------------------------------------------- #
# expected calibration error
# --------------------------------------------------------------------------- #


def test_perfect_calibration_scores_zero() -> None:
    """A predictor whose confidence matches its frequency has no error."""
    probabilities = np.repeat([0.25, 0.75], 400)
    labels = np.concatenate([
        np.repeat([1, 0], [100, 300]),   # 25% positive where p = 0.25
        np.repeat([1, 0], [300, 100]),   # 75% positive where p = 0.75
    ])
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0, abs=1e-9)


def test_maximally_wrong_calibration_scores_one() -> None:
    """Certain and always wrong is the far end of the scale."""
    probabilities = np.ones(100)
    labels = np.zeros(100)
    assert expected_calibration_error(probabilities, labels) == pytest.approx(1.0)


def test_ece_rejects_mismatched_or_empty_input() -> None:
    with pytest.raises(DataError):
        expected_calibration_error(np.array([0.5]), np.array([1, 0]))
    with pytest.raises(DataError):
        expected_calibration_error(np.array([]), np.array([]))


def test_ece_bins_are_the_reported_convention() -> None:
    """Ten equal-width bins, so the figure is comparable to published ones."""
    assert ECE_BINS == 10


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def test_fitting_reduces_calibration_error_on_an_overconfident_model() -> None:
    logits, targets = synthetic_logits()
    result = calibrate(logits, targets)
    assert result["ece_after"] < result["ece_before"]
    assert result["temperature"] > 1.0, "an over-confident model needs softening"


def test_fitting_sharpens_an_underconfident_model() -> None:
    """The converse, so the fit is not merely biased upward.

    A method that could only ever soften would pass the test above by
    construction rather than by working.
    """
    logits, targets = synthetic_logits(scale=0.25)
    assert fit_temperature(logits, targets) < 1.0


def test_fitting_leaves_a_calibrated_model_alone() -> None:
    """The null case: nothing to fix, so barely any change."""
    logits, targets = synthetic_logits(scale=1.0)
    calibrated = fit_temperature(logits, targets)
    logits_scaled = logits / calibrated
    assert fit_temperature(logits_scaled, targets) == pytest.approx(1.0, abs=0.05)


def test_fitting_minimises_negative_log_likelihood() -> None:
    """The fitted point must beat its neighbours, or it is not a minimum."""
    logits, targets = synthetic_logits()
    best = fit_temperature(logits, targets)
    here = negative_log_likelihood(logits, targets, best)
    for other in (best * 0.8, best * 1.2, 1.0, MAX_TEMPERATURE):
        assert here <= negative_log_likelihood(logits, targets, other) + 1e-9


def test_fitting_is_deterministic() -> None:
    """A scan, not an optimiser: the same input gives the same scalar."""
    logits, targets = synthetic_logits()
    assert fit_temperature(logits, targets) == fit_temperature(logits, targets)


def test_fitting_rejects_degenerate_input() -> None:
    with pytest.raises(DataError):
        fit_temperature(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long))
    with pytest.raises(DataError):
        fit_temperature(torch.zeros(4, 2), torch.zeros(3, dtype=torch.long))


# --------------------------------------------------------------------------- #
# the property that makes this safe to ship
# --------------------------------------------------------------------------- #


def test_calibration_changes_no_decision() -> None:
    """Monotonic scaling cannot move argmax, so no metric moves.

    This is why calibration landed without re-measuring anything in PARITY.md.
    If it ever fails, every published figure needs revisiting -- which is
    exactly why it is asserted rather than reasoned about.
    """
    logits, targets = synthetic_logits()
    temperature = fit_temperature(logits, targets)

    before = logits.argmax(dim=1)
    after = (logits / temperature).argmax(dim=1)
    assert torch.equal(before, after)

    accuracy_before = (before == targets).float().mean().item()
    accuracy_after = (after == targets).float().mean().item()
    assert accuracy_before == accuracy_after


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 7.5])
def test_no_temperature_whatsoever_changes_a_decision(temperature: float) -> None:
    """Not just at the fitted value -- for any positive scalar."""
    logits, targets = synthetic_logits()
    assert torch.equal(logits.argmax(dim=1), (logits / temperature).argmax(dim=1))


def test_calibration_reports_what_it_bought() -> None:
    """A temperature without its effect is an unfalsifiable claim."""
    logits, targets = synthetic_logits()
    result = calibrate(logits, targets)
    assert set(result) == {
        "temperature", "ece_before", "ece_after", "nll_before", "nll_after",
    }
    assert result["nll_after"] <= result["nll_before"]


def test_reliability_table_states_the_number_the_bars_would_show() -> None:
    """The text alternative to a reliability diagram."""
    logits, targets = synthetic_logits()
    probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
    table = reliability_table(probabilities, targets.numpy())
    assert "expected calibration error" in table
    assert "observed" in table
