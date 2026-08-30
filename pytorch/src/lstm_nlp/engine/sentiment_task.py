"""Wiring for the sentiment task: config in, trained checkpoint out.

Holds everything task-specific that ``Trainer`` deliberately does not know:
how a batch unpacks, what the loss is, which metrics matter.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lstm_nlp.config import SentimentConfig, dump_config
from lstm_nlp.data.sentiment import (
    LABEL_NAMES,
    SentimentSplits,
    collate_sentiment,
    prepare_sentiment_data,
)
from lstm_nlp.engine.calibration import calibrate
from lstm_nlp.engine.callbacks import BestWeights, EarlyStopping
from lstm_nlp.engine.metrics import ClassificationReport, classification_metrics
from lstm_nlp.engine.trainer import StepFn, Trainer, TrainingHistory
from lstm_nlp.inference.checkpoint import save_checkpoint
from lstm_nlp.models.sentiment_lstm import SentimentLSTM
from lstm_nlp.utils.device import describe_device, resolve_device
from lstm_nlp.utils.logging import get_logger
from lstm_nlp.utils.seed import set_seed

logger = get_logger(__name__)


def make_sentiment_step(criterion: nn.Module) -> StepFn:
    """Build the trainer's per-batch callable, closing over the loss function.

    The criterion is deliberately **not** attached to the model. ``nn`` losses
    are themselves modules, so ``model.criterion = CrossEntropyLoss(weight=...)``
    registers a submodule and leaks ``criterion.weight`` into the model's
    ``state_dict`` -- which then fails to load into a freshly constructed model.
    A loss is a property of the training run, not of the architecture, and
    keeping it out of the model keeps checkpoints loadable (D8).
    """

    def step(
        model: nn.Module, batch: tuple, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one batch. Returns ``(loss, logits, targets)``.

        ``CrossEntropyLoss`` consumes **logits**, which is why the model must
        not apply softmax (``Rules.md`` C1, the structural fix for D2).
        """
        ids, lengths, labels = batch
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids, lengths)
        return criterion(logits, labels), logits, labels

    return step


def sentiment_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Per-epoch validation metrics for the trainer's history."""
    preds = logits.argmax(axis=1)
    report = classification_metrics(targets, preds)
    return {
        "val_accuracy": round(report.accuracy, 6),
        "val_macro_f1": round(report.macro_f1, 6),
    }


def build_loaders(
    splits: SentimentSplits, batch_size: int, seed: int, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders with a seeded shuffle generator.

    The generator is explicit so shuffling is reproducible without touching the
    global RNG (``Rules.md`` §4).

    Three loaders come back, in the order they may be used: train to fit, val to
    select, test to report. Three rather than two so a caller cannot hand the
    test loader to ``fit`` by accident -- which is precisely the defect this
    shape replaced.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        splits.train, batch_size=batch_size, shuffle=True, generator=generator,
        collate_fn=collate_sentiment, num_workers=num_workers,
    )
    val_loader = DataLoader(
        splits.val, batch_size=batch_size, shuffle=False,
        collate_fn=collate_sentiment, num_workers=num_workers,
    )
    test_loader = DataLoader(
        splits.test, batch_size=batch_size, shuffle=False,
        collate_fn=collate_sentiment, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def evaluate_report(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[ClassificationReport, float]:
    """Evaluate and return the full report plus mean loss."""
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.0),
                      make_sentiment_step(nn.CrossEntropyLoss()), device, clip_grad_norm=None)
    loss, logits, targets = trainer.evaluate(loader)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    report = classification_metrics(
        targets, logits.argmax(axis=1), y_score=probabilities[:, 1], class_names=LABEL_NAMES
    )
    return report, loss


def train_sentiment(cfg: SentimentConfig, max_steps: int | None = None) -> Path:
    """Train a sentiment classifier end to end and save its checkpoint.

    Args:
        cfg: Validated configuration.
        max_steps: Cap optimiser steps per epoch, for smoke tests.

    Returns:
        Path to the run directory containing ``best.pt``, ``history.json``,
        ``metrics.json`` and the resolved ``config.yaml``.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("device: %s", describe_device(device))

    splits = prepare_sentiment_data(
        cfg.data.csv,
        test_size=cfg.data.test_size,
        split_seed=cfg.data.split_seed,
        stratify=cfg.data.stratify,
        val_size=cfg.data.val_size,
        min_freq=cfg.data.min_freq,
        max_len=cfg.data.max_len,
    )
    logger.info(
        "train %d / val %d / test %d  vocabulary %d  test OOV %.2f%%",
        len(splits.train), len(splits.val), len(splits.test), len(splits.vocab),
        100 * splits.test.unknown_rate(),
    )

    model = SentimentLSTM(
        vocab_size=len(splits.vocab),
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        num_classes=splits.num_classes,
        pad_idx=splits.vocab.pad_index or 0,
    ).to(device)
    logger.info("model: %s parameters", f"{model.num_parameters():,}")

    weights = splits.class_weights.to(device) if cfg.train.class_weighting == "balanced" else None
    if weights is not None:
        logger.info("class weights: %s", [round(w, 3) for w in weights.tolist()])
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader, val_loader, test_loader = build_loaders(
        splits, cfg.train.batch_size, cfg.seed, cfg.train.num_workers
    )

    es = cfg.train.early_stopping
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        ),
        step_fn=make_sentiment_step(criterion),
        device=device,
        clip_grad_norm=cfg.train.clip_grad_norm,
        early_stopping=EarlyStopping(es.monitor, es.mode, es.patience, es.min_delta),
        best_weights=BestWeights(es.monitor, es.mode),
        metrics_fn=sentiment_metrics,
        max_steps=max_steps,
    )

    # Selection sees `val` and only `val`. `test` is scored once, below.
    history: TrainingHistory = trainer.fit(train_loader, val_loader, cfg.train.epochs)

    val_report, _ = evaluate_report(model, val_loader, device)

    # Temperature scaling, fitted on validation and never on test. It is
    # monotonic, so every metric above and below is bit-identical either
    # way -- it buys honest confidence, not a better score.
    _, val_logits, val_targets = trainer.evaluate(val_loader)
    calibration = calibrate(
        torch.from_numpy(val_logits), torch.from_numpy(val_targets)
    )
    logger.info(
        "calibration: T=%.4f  ECE %.4f -> %.4f",
        calibration["temperature"], calibration["ece_before"],
        calibration["ece_after"],
    )
    report, _ = evaluate_report(model, test_loader, device)
    print("\nValidation metrics (early stopping selected on these)\n" + val_report.format() + "\n")
    print("Test metrics (held out; scored once)\n" + report.format() + "\n")

    run_dir = Path(cfg.output.dir) / datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        run_dir / "best.pt",
        task="sentiment",
        model=model,
        model_cfg=model.config(),
        vocab=splits.vocab,
        preprocess={"max_len": cfg.data.max_len, "min_freq": cfg.data.min_freq},
        metrics=report.to_dict(),
        train_info={
            "seed": cfg.seed,
            "best_epoch": history.best_epoch,
            "stopped_early": history.stopped_early,
            "epochs_run": len(history.epochs),
            "class_weights": weights.tolist() if weights is not None else None,
            # Non-None marks a truncated smoke run. Recorded so a caller can
            # tell one from a real run, and so checkpoint resolution can refuse
            # to serve it (Rules.md B7).
            "max_steps": max_steps,
            "val_metrics": val_report.to_dict(),
            "calibration": calibration,
        },
    )
    history.save(run_dir / "history.json")
    (run_dir / "metrics.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    dump_config(cfg, run_dir / "config.yaml")

    logger.info("run written to %s", run_dir)
    return run_dir
