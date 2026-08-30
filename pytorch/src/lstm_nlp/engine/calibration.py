"""Temperature scaling: make the reported probabilities mean what they say.

The Phase 8 review measured an Expected Calibration Error of **0.066** on the
test split: inputs the model scored around 0.75 were positive about 46% of the
time. That is not a bug in the model, it is the price of
``class_weighting: balanced`` (4.130:1 on the deduplicated corpus, 3.884:1 when
that ECE was measured), which is the right trade for macro-F1
on a 79.5%-negative corpus and the wrong one for calibration -- upweighting the
minority class deliberately decouples the outputs from the data's prior.

The frontend prints those numbers under the word "probability", and
``Design.md`` principle 1 is *make the model's uncertainty legible*. A number
labelled as a probability that is not one fails that on its own terms.

Temperature scaling (Guo et al., 2017) fits a single scalar ``T`` on held-out
data and reports ``softmax(logits / T)``. Two properties make it the right tool
here rather than isotonic regression or a Platt fit:

* **It cannot change any decision.** Dividing logits by a positive scalar is
  monotonic, so ``argmax`` is unchanged and accuracy, macro-F1, the confusion
  matrix and ROC-AUC are all *bit-identical* before and after. Calibration here
  buys honesty about confidence and costs nothing in measured performance --
  which is why it needed no re-measurement of anything in ``PARITY.md``.
* **It is one parameter.** Fitted on 1,212 validation rows, a single scalar
  cannot meaningfully overfit; isotonic regression on that much data could.

``T`` is fitted on the **validation** block. Fitting it on test would recreate
the defect Phase 8 fixed, in the module written to fix a related one.
"""

from __future__ import annotations

import numpy as np
import torch

from lstm_nlp.errors import DataError

#: Bounds for the fitted temperature.
#:
#: Below 1 the model would be *under*-confident and scaling would sharpen it;
#: that is legitimate and allowed. The upper bound is a guard against a
#: degenerate fit on a tiny validation block, not a modelling belief.
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.05, 10.0

#: Bins used for the Expected Calibration Error.
#:
#: Ten equal-width bins is the convention the ECE literature reports against,
#: so the number stays comparable to published figures.
ECE_BINS = 10


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """Mean gap between confidence and accuracy, weighted by bin population.

    Args:
        probabilities: ``(N,)`` predicted probability of the positive class.
        labels: ``(N,)`` 0/1 ground truth.
        n_bins: Equal-width bins over [0, 1].

    Returns:
        ECE in [0, 1]. Zero means every bin's average confidence matched its
        observed frequency.

    Raises:
        DataError: If the inputs are empty or of different lengths.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probabilities.shape != labels.shape:
        raise DataError(
            f"probabilities and labels differ in shape: "
            f"{probabilities.shape} vs {labels.shape}"
        )
    if probabilities.size == 0:
        raise DataError("cannot compute calibration error on an empty set")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (probabilities >= low) & (probabilities < high if high < 1.0 else probabilities <= 1.0)
        if not in_bin.any():
            continue
        confidence = probabilities[in_bin].mean()
        observed = labels[in_bin].mean()
        error += in_bin.mean() * abs(confidence - observed)
    return float(error)


def negative_log_likelihood(logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> float:
    """Mean NLL of ``targets`` under ``softmax(logits / temperature)``."""
    scaled = logits / max(temperature, MIN_TEMPERATURE)
    return float(torch.nn.functional.cross_entropy(scaled, targets).item())


def fit_temperature(
    logits: torch.Tensor, targets: torch.Tensor, *, tolerance: float = 1e-4
) -> float:
    """Find the temperature minimising validation NLL.

    A deterministic coarse-to-fine scan rather than a gradient optimiser. The
    objective is smooth and one-dimensional, so a scan finds the same optimum
    without an optimiser's learning rate, iteration count and random behaviour
    -- three things that would have to be pinned for the fit to be reproducible.

    Args:
        logits: ``(N, C)`` raw model outputs on the **validation** block.
        targets: ``(N,)`` class indices.
        tolerance: Stop once the bracket is narrower than this.

    Returns:
        The fitted temperature, clamped to the supported range.

    Raises:
        DataError: If there is nothing to fit on.
    """
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise DataError(f"expected a non-empty (N, C) logit tensor, got {tuple(logits.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise DataError(
            f"logits and targets differ in length: {logits.shape[0]} vs {targets.shape[0]}"
        )

    low, high = MIN_TEMPERATURE, MAX_TEMPERATURE
    best = 1.0
    while high - low > tolerance:
        candidates = np.linspace(low, high, 25)
        losses = [negative_log_likelihood(logits, targets, float(t)) for t in candidates]
        index = int(np.argmin(losses))
        best = float(candidates[index])
        step = candidates[1] - candidates[0]
        low, high = max(MIN_TEMPERATURE, best - step), min(MAX_TEMPERATURE, best + step)
    return best


def calibrate(
    logits: torch.Tensor, targets: torch.Tensor, positive_index: int = 1
) -> dict[str, float]:
    """Fit a temperature and report what it bought.

    Args:
        logits: ``(N, C)`` validation logits.
        targets: ``(N,)`` validation labels.
        positive_index: Column treated as the positive class for ECE.

    Returns:
        ``temperature``, plus ``ece_before``/``ece_after`` and
        ``nll_before``/``nll_after`` so the fit can be judged rather than
        trusted. A temperature reported without its effect is the same
        unfalsifiable claim this project keeps removing.
    """
    temperature = fit_temperature(logits, targets)
    labels = targets.numpy()

    before = torch.softmax(logits, dim=1)[:, positive_index].numpy()
    after = torch.softmax(logits / temperature, dim=1)[:, positive_index].numpy()

    return {
        "temperature": round(temperature, 4),
        "ece_before": round(expected_calibration_error(before, labels), 4),
        "ece_after": round(expected_calibration_error(after, labels), 4),
        "nll_before": round(negative_log_likelihood(logits, targets, 1.0), 4),
        "nll_after": round(negative_log_likelihood(logits, targets, temperature), 4),
    }


def reliability_table(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS
) -> str:
    """Human-readable reliability diagram, for the CLI and the run log."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = [f"  {'bin':<11}{'n':>6}{'mean p':>9}{'observed':>10}{'gap':>8}"]
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (probabilities >= low) & (probabilities < high if high < 1.0 else probabilities <= 1.0)
        if not in_bin.any():
            continue
        confidence, observed = probabilities[in_bin].mean(), labels[in_bin].mean()
        rows.append(
            f"  {low:.1f}-{high:.1f}{in_bin.sum():>10}{confidence:>9.3f}"
            f"{observed:>10.3f}{confidence - observed:>+8.3f}"
        )
    ece = expected_calibration_error(probabilities, labels, n_bins)
    rows.append(f"\n  expected calibration error {ece:.4f}   (0 = perfectly calibrated)")
    return "\n".join(rows)
