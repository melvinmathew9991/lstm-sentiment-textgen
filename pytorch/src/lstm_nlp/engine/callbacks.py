"""Early stopping and best-weight tracking.

The fix for D5. The reference trained a fixed 50 epochs with no callbacks and
saved whatever it had at the end -- by which point validation loss had risen
from 0.215 to 0.526 while training loss fell to 0.015. The saved artifact was
the *worst* model the run produced.

``BestWeights`` keeps a copy of the best epoch's ``state_dict`` in memory and
restores it when the run ends, so the weights that get saved are never simply
the last ones (``Rules.md`` C12).
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from lstm_nlp.errors import ConfigError


class EarlyStopping:
    """Stop when the monitored metric has not improved for ``patience`` epochs.

    Args:
        monitor: Key to watch in the metrics dict passed to :meth:`step`.
        mode: ``"min"`` if lower is better, ``"max"`` if higher is better.
        patience: Epochs without improvement before stopping.
        min_delta: Minimum change that counts as an improvement.

    Raises:
        ConfigError: If ``mode`` is not ``"min"`` or ``"max"``.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ConfigError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best: float | None = None
        self.best_epoch: int = -1
        self.epochs_without_improvement: int = 0
        self.stopped_epoch: int | None = None

    def is_improvement(self, value: float) -> bool:
        """Whether ``value`` beats the best seen so far by ``min_delta``."""
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Record an epoch's metrics and report whether training should stop.

        Args:
            epoch: Zero-based epoch index.
            metrics: Must contain ``self.monitor``.

        Returns:
            ``True`` if training should stop now.

        Raises:
            ConfigError: If the monitored key is absent -- a silent fallback
                here would mean training never stops for the right reason.
        """
        if self.monitor not in metrics:
            raise ConfigError(
                f"early stopping monitors {self.monitor!r}, which is not among "
                f"the reported metrics {sorted(metrics)}"
            )
        value = float(metrics[self.monitor])

        if self.is_improvement(value):
            self.best = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    @property
    def stopped_early(self) -> bool:
        """Whether stopping was triggered rather than the epoch cap reached."""
        return self.stopped_epoch is not None


class BestWeights:
    """Hold the best epoch's weights and restore them at the end of the run.

    Kept in memory rather than written per epoch: the models here are under
    1.4M parameters, and a deep copy costs far less than repeated disk writes.

    Args:
        monitor: Metric key to track.
        mode: ``"min"`` or ``"max"``.
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ConfigError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.best: float | None = None
        self.best_epoch: int = -1
        self._state: dict[str, torch.Tensor] | None = None

    def step(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> bool:
        """Snapshot the model if this epoch is the best so far.

        Returns:
            ``True`` if a snapshot was taken.

        Raises:
            ConfigError: If the monitored key is absent.
        """
        if self.monitor not in metrics:
            raise ConfigError(
                f"best-weight tracking monitors {self.monitor!r}, which is not "
                f"among the reported metrics {sorted(metrics)}"
            )
        value = float(metrics[self.monitor])
        better = (
            self.best is None
            or (self.mode == "min" and value < self.best)
            or (self.mode == "max" and value > self.best)
        )
        if not better:
            return False

        self.best = value
        self.best_epoch = epoch
        self._state = copy.deepcopy(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        )
        return True

    @property
    def has_snapshot(self) -> bool:
        """Whether any epoch has been recorded."""
        return self._state is not None

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The best epoch's weights.

        Raises:
            ConfigError: If no epoch was ever recorded.
        """
        if self._state is None:
            raise ConfigError("no epoch was recorded; nothing to restore")
        return self._state

    def restore(self, model: nn.Module) -> int:
        """Load the best weights into ``model``.

        Returns:
            The epoch the restored weights came from.
        """
        model.load_state_dict(self.state_dict())
        return self.best_epoch
