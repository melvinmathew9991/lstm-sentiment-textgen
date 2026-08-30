"""Metrics, each reported beside the baseline it must beat.

`Rules.md` C11: no metric without its baseline. The reference reported 0.909
validation accuracy on data whose majority class already scores 0.795, and the
gap between those two numbers is the only part that was evidence of anything
(D4).

Baselines are **computed from the labels**, never hardcoded. A baseline written
as a literal drifts the moment the split changes, and a stale baseline is worse
than none -- it looks like corroboration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from lstm_nlp.errors import DataError


@dataclass
class ClassificationReport:
    """Classification metrics with their majority-class baselines."""

    accuracy: float
    macro_f1: float
    baseline_accuracy: float
    baseline_macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    roc_auc: float | None
    support: dict[str, int]
    class_names: tuple[str, ...] = field(default=("negative", "positive"))

    @property
    def accuracy_lift(self) -> float:
        """Accuracy above the majority-class baseline. The part that is news."""
        return self.accuracy - self.baseline_accuracy

    @property
    def macro_f1_lift(self) -> float:
        """Macro-F1 above the majority-class baseline."""
        return self.macro_f1 - self.baseline_macro_f1

    def to_dict(self) -> dict:
        """JSON-safe payload for ``metrics.json`` and checkpoints."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "baseline_accuracy": self.baseline_accuracy,
            "baseline_macro_f1": self.baseline_macro_f1,
            "accuracy_lift": self.accuracy_lift,
            "macro_f1_lift": self.macro_f1_lift,
            "roc_auc": self.roc_auc,
            "per_class": self.per_class,
            "confusion": self.confusion,
            "support": self.support,
        }

    def format(self) -> str:
        """Render as a text block. Every metric sits beside its baseline."""
        lines = [
            "                    value    baseline      lift",
            f"  accuracy        {self.accuracy:7.4f}    {self.baseline_accuracy:7.4f}   {self.accuracy_lift:+7.4f}",
            f"  macro-F1        {self.macro_f1:7.4f}    {self.baseline_macro_f1:7.4f}   {self.macro_f1_lift:+7.4f}",
        ]
        if self.roc_auc is not None:
            lines.append(
                f"  ROC-AUC         {self.roc_auc:7.4f}    {0.5:7.4f}   {self.roc_auc - 0.5:+7.4f}"
            )
        lines.append("")
        lines.append("  per class        prec     recall        F1    support")
        for name in self.class_names:
            m = self.per_class[name]
            lines.append(
                "    {:<12s} {:7.4f}    {:7.4f}   {:7.4f}    {:7d}".format(
                    name, m["precision"], m["recall"], m["f1"], self.support[name]
                )
            )
        lines.append("")
        lines.append("  confusion (rows = true, cols = predicted)")
        header = "               " + "".join(f"{n:>12s}" for n in self.class_names)
        lines.append(header)
        for name, row in zip(self.class_names, self.confusion, strict=True):
            lines.append(f"    {name:<11s}" + "".join(f"{v:12d}" for v in row))
        return "\n".join(lines)


def majority_baseline(y_true: np.ndarray, n_classes: int = 2) -> tuple[float, float]:
    """Accuracy and macro-F1 of always predicting the most common class.

    This is the number any real model has to beat. On this dataset it is
    0.8048 accuracy and 0.4459 macro-F1 -- the second is far lower because the
    minority class scores an F1 of exactly zero, which is precisely what
    accuracy alone conceals (D4).

    Args:
        y_true: True labels.
        n_classes: Number of classes.

    Returns:
        ``(baseline_accuracy, baseline_macro_f1)``.

    Raises:
        DataError: If ``y_true`` is empty.
    """
    y_true = np.asarray(y_true)
    if y_true.size == 0:
        raise DataError("cannot compute a baseline from zero labels")

    counts = np.bincount(y_true, minlength=n_classes)
    majority = int(np.argmax(counts))
    y_pred = np.full_like(y_true, majority)

    accuracy = float((y_pred == y_true).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return accuracy, macro_f1


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    class_names: tuple[str, ...] = ("negative", "positive"),
) -> ClassificationReport:
    """Compute the full classification report, baselines included.

    Args:
        y_true: True labels, shape ``(N,)``.
        y_pred: Predicted labels, shape ``(N,)``.
        y_score: Optional positive-class scores for ROC-AUC, shape ``(N,)``.
        class_names: Names in label order.

    Returns:
        A report carrying every metric alongside its baseline.

    Raises:
        DataError: If the arrays are empty or their lengths disagree.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        raise DataError("cannot compute metrics from zero predictions")
    if y_true.shape != y_pred.shape:
        raise DataError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    n_classes = len(class_names)
    labels = list(range(n_classes))

    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    base_acc, base_f1 = majority_baseline(y_true, n_classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i, name in enumerate(class_names)
    }

    roc_auc = None
    if y_score is not None and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, np.asarray(y_score)))

    return ClassificationReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        baseline_accuracy=base_acc,
        baseline_macro_f1=base_f1,
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        roc_auc=roc_auc,
        support={name: int(support[i]) for i, name in enumerate(class_names)},
        class_names=class_names,
    )


def perplexity(mean_cross_entropy: float) -> float:
    """Convert mean cross-entropy in nats to perplexity."""
    return float(math.exp(mean_cross_entropy))


def uniform_perplexity_baseline(vocab_size: int) -> float:
    """Perplexity of guessing uniformly at random: exactly the vocabulary size.

    The number a language model must beat. For the 2,436-token Alice vocabulary
    that is 2,436, from a cross-entropy of ln(2436) = 7.798 nats.
    """
    if vocab_size < 1:
        raise DataError(f"vocab_size must be >= 1, got {vocab_size}")
    return float(vocab_size)
