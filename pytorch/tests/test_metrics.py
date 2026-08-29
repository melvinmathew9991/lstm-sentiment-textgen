"""Phase 2: metrics. Regression tests for D4 and C11."""

from __future__ import annotations

import numpy as np
import pytest

from lstm_nlp.engine.metrics import (
    classification_metrics,
    majority_baseline,
    perplexity,
    uniform_perplexity_baseline,
)
from lstm_nlp.errors import DataError

# 80/20 imbalance, the shape of the real data (which is 79.53/20.47).
Y_IMBALANCED = np.array([0] * 80 + [1] * 20)


def test_majority_baseline_on_imbalanced_data() -> None:
    """The number every reported metric must be read against."""
    accuracy, macro_f1 = majority_baseline(Y_IMBALANCED)
    assert accuracy == pytest.approx(0.80)
    # The minority class scores F1 exactly 0, which halves the macro average.
    # That collapse is what accuracy alone conceals (D4).
    assert macro_f1 == pytest.approx(0.4444, abs=1e-4)


def test_baseline_is_computed_not_hardcoded() -> None:
    """Change the class balance and the baseline must move with it."""
    a90, _ = majority_baseline(np.array([0] * 90 + [1] * 10))
    a50, _ = majority_baseline(np.array([0] * 50 + [1] * 50))
    assert a90 == pytest.approx(0.90)
    assert a50 == pytest.approx(0.50)


def test_empty_labels_rejected() -> None:
    with pytest.raises(DataError, match="zero labels"):
        majority_baseline(np.array([], dtype=int))


def test_all_negative_predictor_exactly_equals_the_baseline() -> None:
    """The D4 demonstration, as an assertion.

    A model that has learned nothing scores 0.80 accuracy here, which reads as
    respectable. Its macro-F1 of 0.44 is what gives it away.
    """
    y_pred = np.zeros_like(Y_IMBALANCED)
    report = classification_metrics(Y_IMBALANCED, y_pred)
    assert report.accuracy == report.baseline_accuracy
    assert report.macro_f1 == report.baseline_macro_f1
    assert report.accuracy_lift == pytest.approx(0.0)
    assert report.macro_f1_lift == pytest.approx(0.0)
    assert report.per_class["positive"]["f1"] == 0.0


def test_report_includes_baselines() -> None:
    """C11: a metric is never reported without the baseline it must beat."""
    report = classification_metrics(Y_IMBALANCED, np.zeros_like(Y_IMBALANCED))
    payload = report.to_dict()
    for key in ("accuracy", "macro_f1", "baseline_accuracy", "baseline_macro_f1",
                "accuracy_lift", "macro_f1_lift"):
        assert key in payload, f"{key} missing from the metrics payload"


def test_formatted_output_shows_value_baseline_and_lift() -> None:
    """C11 applies to what a human reads, not only to the JSON."""
    text = classification_metrics(Y_IMBALANCED, np.zeros_like(Y_IMBALANCED)).format()
    assert "baseline" in text and "lift" in text
    assert "accuracy" in text and "macro-F1" in text
    assert "confusion" in text
    for name in ("negative", "positive"):
        assert name in text


def test_perfect_predictions() -> None:
    report = classification_metrics(Y_IMBALANCED, Y_IMBALANCED)
    assert report.accuracy == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.macro_f1_lift > 0.5


def test_confusion_matrix_orientation() -> None:
    """Rows are true, columns predicted -- stated in the output and asserted."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    report = classification_metrics(y_true, y_pred)
    assert report.confusion == [[1, 1], [0, 2]]


def test_roc_auc_optional_and_computed_when_scores_given() -> None:
    y_true = np.array([0, 0, 1, 1])
    assert classification_metrics(y_true, y_true).roc_auc is None
    report = classification_metrics(y_true, y_true, y_score=np.array([0.1, 0.2, 0.8, 0.9]))
    assert report.roc_auc == pytest.approx(1.0)


def test_roc_auc_skipped_when_one_class_present() -> None:
    y = np.zeros(4, dtype=int)
    assert classification_metrics(y, y, y_score=np.array([0.1, 0.2, 0.3, 0.4])).roc_auc is None


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(DataError, match="shape mismatch"):
        classification_metrics(np.array([0, 1]), np.array([0]))


def test_empty_predictions_rejected() -> None:
    with pytest.raises(DataError, match="zero predictions"):
        classification_metrics(np.array([], dtype=int), np.array([], dtype=int))


def test_support_counts_rows_per_class() -> None:
    report = classification_metrics(Y_IMBALANCED, np.zeros_like(Y_IMBALANCED))
    assert report.support == {"negative": 80, "positive": 20}


# --------------------------------------------------------------------------- #
# language-model metrics (used by Phase 3)
# --------------------------------------------------------------------------- #


def test_perplexity_is_exp_of_cross_entropy() -> None:
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(np.log(50)) == pytest.approx(50.0)


def test_uniform_baseline_equals_vocab_size() -> None:
    """Guessing uniformly gives a perplexity of exactly the vocabulary size."""
    assert uniform_perplexity_baseline(2436) == pytest.approx(2436.0)
    assert perplexity(float(np.log(2436))) == pytest.approx(2436.0, rel=1e-6)


def test_uniform_baseline_rejects_empty_vocab() -> None:
    with pytest.raises(DataError, match="vocab_size"):
        uniform_perplexity_baseline(0)
