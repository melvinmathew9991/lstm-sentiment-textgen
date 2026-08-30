"""Task-agnostic training loop.

Roughly 80 lines of actual mechanics, which is why no framework is used here
(`Rules.md` §2): a framework would hide precisely the parts this project exists
to make legible -- when the optimiser steps, when gradients are clipped, which
epoch's weights survive.

The loop knows nothing about sentiment or text generation. A task supplies two
callables:

* ``step_fn(model, batch, device) -> (loss, logits, targets)``
* ``metrics_fn(logits, targets) -> dict[str, float]`` (optional)

Phase 3 reuses this unchanged. If it ever needs a task-specific branch, the
abstraction is wrong and the abstraction gets fixed, not the loop.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lstm_nlp.engine.callbacks import BestWeights, EarlyStopping
from lstm_nlp.errors import TrainingError
from lstm_nlp.utils.logging import get_logger

logger = get_logger(__name__)

StepFn = Callable[[nn.Module, tuple, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
MetricsFn = Callable[[np.ndarray, np.ndarray], dict[str, float]]


@dataclass
class TrainingHistory:
    """Per-epoch record of everything the run measured."""

    epochs: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    stopped_early: bool = False
    total_seconds: float = 0.0

    def append(self, record: dict[str, float]) -> None:
        """Add one epoch's metrics."""
        self.epochs.append(record)

    @property
    def best_epoch_number(self) -> int | None:
        """The best epoch as it is *numbered* in ``epochs``, or ``None``.

        ``best_epoch`` is a zero-based index into ``epochs``; the ``epoch`` field
        inside each record is one-based, and so is every log line and every
        printed summary.  Persisting the raw index beside one-based records made
        ``history.json`` contradict itself -- a run that restored epoch 9 wrote
        ``"best_epoch": 8`` next to the record labelled ``"epoch": 9``.  Nothing
        computed with it, so nothing broke; a reader of the artifact simply could
        not tell which convention it was in.  Everything persisted is one-based
        from 2026-08-30.
        """
        return None if self.best_epoch < 0 else self.best_epoch + 1

    def to_dict(self) -> dict:
        """JSON-safe payload for ``history.json``."""
        return {
            "epochs": self.epochs,
            "best_epoch": self.best_epoch_number,
            "stopped_early": self.stopped_early,
            "total_seconds": round(self.total_seconds, 2),
            "n_epochs_run": len(self.epochs),
        }

    def save(self, path: Path) -> None:
        """Write ``history.json``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class Trainer:
    """Epoch loop with gradient clipping, early stopping and best-weight restore.

    Args:
        model: The module to train.
        optimizer: Its optimiser.
        step_fn: Runs one batch, returning ``(loss, logits, targets)``.
        device: Compute device.
        clip_grad_norm: Max gradient norm; ``None`` disables clipping.
        early_stopping: Stopping policy. ``None`` runs every epoch.
        best_weights: Best-epoch tracker. ``None`` keeps the final weights,
            which is the D5 behaviour and should only be used in tests.
        metrics_fn: Optional extra validation metrics from predictions.
        max_steps: Cap on optimiser steps per epoch, for smoke tests.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step_fn: StepFn,
        device: torch.device,
        clip_grad_norm: float | None = 5.0,
        early_stopping: EarlyStopping | None = None,
        best_weights: BestWeights | None = None,
        metrics_fn: MetricsFn | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.step_fn = step_fn
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.early_stopping = early_stopping
        self.best_weights = best_weights
        self.metrics_fn = metrics_fn
        self.max_steps = max_steps

    def train_epoch(self, loader: DataLoader, epoch: int, total: int) -> float:
        """Run one training epoch and return its mean loss.

        Raises:
            TrainingError: If the loss becomes NaN or infinite. Continuing past
                that point only produces a longer run with no model at the end.
        """
        self.model.train()
        running, seen = 0.0, 0
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{total}", leave=False, unit="batch")

        for i, batch in enumerate(bar):
            if self.max_steps is not None and i >= self.max_steps:
                break

            self.optimizer.zero_grad(set_to_none=True)
            loss, _, targets = self.step_fn(self.model, batch, self.device)

            if not math.isfinite(loss.item()):
                raise TrainingError(
                    f"loss became {loss.item()} at epoch {epoch + 1}, batch {i}. "
                    f"Aborting rather than continuing with a diverged model."
                )

            loss.backward()
            if self.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            n = int(targets.shape[0])
            running += loss.item() * n
            seen += n
            bar.set_postfix(loss=f"{running / max(seen, 1):.4f}")

        return running / max(seen, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        """Run the model over ``loader`` without updating it.

        Returns:
            ``(mean_loss, logits, targets)`` with both arrays on CPU.
        """
        self.model.eval()
        running, seen = 0.0, 0
        all_logits, all_targets = [], []

        for batch in loader:
            loss, logits, targets = self.step_fn(self.model, batch, self.device)
            n = int(targets.shape[0])
            running += loss.item() * n
            seen += n
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

        return (
            running / max(seen, 1),
            np.concatenate(all_logits) if all_logits else np.empty((0,)),
            np.concatenate(all_targets) if all_targets else np.empty((0,)),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> TrainingHistory:
        """Train, validate each epoch, then restore the best weights.

        Returns:
            The full per-epoch history, with ``best_epoch`` and
            ``stopped_early`` recorded.
        """
        history = TrainingHistory()
        started = time.perf_counter()

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch, epochs)
            val_loss, logits, targets = self.evaluate(val_loader)

            record: dict[str, float] = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
            }
            if self.metrics_fn is not None:
                record.update(self.metrics_fn(logits, targets))
            history.append(record)

            extra = "  ".join(
                f"{k}={v:.4f}" for k, v in record.items() if k not in ("epoch",)
            )
            logger.info("epoch %d/%d  %s", epoch + 1, epochs, extra)

            if self.best_weights is not None and self.best_weights.step(
                epoch, record, self.model
            ):
                logger.info("  new best %s=%.4f", self.best_weights.monitor,
                            self.best_weights.best)

            if self.early_stopping is not None and self.early_stopping.step(epoch, record):
                logger.info(
                    "early stopping at epoch %d: no improvement in %s for %d epochs",
                    epoch + 1, self.early_stopping.monitor, self.early_stopping.patience,
                )
                history.stopped_early = True
                break

        history.total_seconds = time.perf_counter() - started

        # The D5 fix, at the one moment it matters: what leaves this function is
        # the best epoch's model, never simply the last one.
        if self.best_weights is not None and self.best_weights.has_snapshot:
            restored = self.best_weights.restore(self.model)
            history.best_epoch = restored
            logger.info(
                "restored weights from epoch %d (%s=%.4f)",
                restored + 1, self.best_weights.monitor, self.best_weights.best,
            )
        else:
            history.best_epoch = len(history.epochs) - 1

        return history
