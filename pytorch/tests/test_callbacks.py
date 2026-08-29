"""Phase 2: early stopping and best-weight restore. Regression tests for D5."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lstm_nlp.engine.callbacks import BestWeights, EarlyStopping
from lstm_nlp.errors import ConfigError


def tiny_model(value: float) -> nn.Module:
    """A one-parameter model whose weight identifies the epoch it came from."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


# --------------------------------------------------------------------------- #
# EarlyStopping
# --------------------------------------------------------------------------- #


def test_stops_after_patience_without_improvement() -> None:
    es = EarlyStopping("val_loss", "min", patience=3)
    assert es.step(0, {"val_loss": 1.0}) is False
    assert es.step(1, {"val_loss": 1.1}) is False
    assert es.step(2, {"val_loss": 1.2}) is False
    assert es.step(3, {"val_loss": 1.3}) is True
    assert es.stopped_early
    assert es.best_epoch == 0


def test_improvement_resets_the_counter() -> None:
    es = EarlyStopping("val_loss", "min", patience=2)
    es.step(0, {"val_loss": 1.0})
    es.step(1, {"val_loss": 1.5})
    assert es.step(2, {"val_loss": 0.5}) is False
    assert es.epochs_without_improvement == 0
    assert es.best_epoch == 2


def test_max_mode_tracks_increases() -> None:
    es = EarlyStopping("val_macro_f1", "max", patience=2)
    assert es.step(0, {"val_macro_f1": 0.5}) is False
    assert es.step(1, {"val_macro_f1": 0.7}) is False
    assert es.best == pytest.approx(0.7)
    assert es.step(2, {"val_macro_f1": 0.6}) is False
    assert es.step(3, {"val_macro_f1": 0.6}) is True


def test_min_delta_ignores_negligible_gains() -> None:
    es = EarlyStopping("val_loss", "min", patience=2, min_delta=0.1)
    es.step(0, {"val_loss": 1.0})
    assert es.step(1, {"val_loss": 0.95}) is False  # gain below min_delta
    assert es.step(2, {"val_loss": 0.94}) is True
    assert es.best == pytest.approx(1.0)


def test_never_stops_while_improving() -> None:
    es = EarlyStopping("val_loss", "min", patience=2)
    for epoch in range(20):
        assert es.step(epoch, {"val_loss": 1.0 - epoch * 0.01}) is False
    assert not es.stopped_early


def test_missing_monitor_key_raises() -> None:
    """A silent fallback would mean training never stops for the right reason."""
    es = EarlyStopping("val_macro_f1", "max", patience=2)
    with pytest.raises(ConfigError, match="val_macro_f1"):
        es.step(0, {"val_loss": 1.0})


def test_bad_mode_rejected() -> None:
    with pytest.raises(ConfigError, match="min.*max"):
        EarlyStopping("val_loss", "sideways")


# --------------------------------------------------------------------------- #
# BestWeights -- the D5 fix
# --------------------------------------------------------------------------- #


def test_best_not_last_checkpoint_restored() -> None:
    """The D5 regression test.

    Simulates the reference's exact failure curve: validation improves briefly,
    then degrades for the rest of the run while training loss keeps falling.
    The reference saved the final epoch -- the worst model it produced. What
    must survive here is epoch 1.
    """
    tracker = BestWeights("val_loss", "min")
    curve = [0.50, 0.20, 0.35, 0.42, 0.55, 0.61]  # best at index 1
    for epoch, val_loss in enumerate(curve):
        tracker.step(epoch, {"val_loss": val_loss}, tiny_model(float(epoch)))

    assert tracker.best_epoch == 1
    assert tracker.best == pytest.approx(0.20)

    final = tiny_model(99.0)  # the "last epoch" weights
    restored_epoch = tracker.restore(final)
    assert restored_epoch == 1
    assert final.weight.detach().item() == pytest.approx(1.0), "restored the last epoch, not the best"


def test_max_mode_keeps_the_highest() -> None:
    tracker = BestWeights("val_macro_f1", "max")
    for epoch, f1 in enumerate([0.10, 0.90, 0.40]):
        tracker.step(epoch, {"val_macro_f1": f1}, tiny_model(float(epoch)))
    model = tiny_model(0.0)
    tracker.restore(model)
    assert model.weight.detach().item() == pytest.approx(1.0)


def test_snapshot_is_a_copy_not_a_reference() -> None:
    """Later training must not mutate the stored best weights."""
    tracker = BestWeights("val_loss", "min")
    model = tiny_model(1.0)
    tracker.step(0, {"val_loss": 0.1}, model)
    with torch.no_grad():
        model.weight.fill_(42.0)
    tracker.restore(model)
    assert model.weight.detach().item() == pytest.approx(1.0)


def test_step_reports_whether_it_snapshotted() -> None:
    tracker = BestWeights("val_loss", "min")
    assert tracker.step(0, {"val_loss": 1.0}, tiny_model(0.0)) is True
    assert tracker.step(1, {"val_loss": 2.0}, tiny_model(1.0)) is False


def test_restore_without_a_snapshot_raises() -> None:
    tracker = BestWeights("val_loss", "min")
    assert not tracker.has_snapshot
    with pytest.raises(ConfigError, match="nothing to restore"):
        tracker.restore(tiny_model(0.0))


def test_missing_monitor_key_raises_in_tracker() -> None:
    tracker = BestWeights("val_macro_f1", "max")
    with pytest.raises(ConfigError, match="val_macro_f1"):
        tracker.step(0, {"val_loss": 1.0}, tiny_model(0.0))
