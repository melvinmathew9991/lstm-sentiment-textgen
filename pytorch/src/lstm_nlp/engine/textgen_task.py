"""Wiring for the text-generation task: config in, trained checkpoint out.

Deliberately parallel to ``sentiment_task``. ``engine.trainer.Trainer`` is
reused **unchanged** -- if this task had needed a branch inside the loop, the
abstraction would have been wrong and the abstraction would have been fixed
(``Phases.md`` Phase 3 exit criteria).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lstm_nlp.config import TextGenConfig, dump_config
from lstm_nlp.data.textgen import TextGenSplits, prepare_textgen_data
from lstm_nlp.engine.callbacks import BestWeights, EarlyStopping
from lstm_nlp.engine.metrics import perplexity, uniform_perplexity_baseline
from lstm_nlp.engine.trainer import StepFn, Trainer, TrainingHistory
from lstm_nlp.inference.checkpoint import save_checkpoint
from lstm_nlp.models.textgen_lstm import TextGenLSTM
from lstm_nlp.utils.device import describe_device, resolve_device
from lstm_nlp.utils.logging import get_logger
from lstm_nlp.utils.seed import set_seed

logger = get_logger(__name__)


def make_textgen_step(criterion: nn.Module) -> StepFn:
    """Build the trainer's per-batch callable, closing over the loss function.

    As in the sentiment task, the criterion is not attached to the model: ``nn``
    losses are modules, and assigning one would leak its buffers into the
    model's ``state_dict`` and break checkpoint loading (D8).
    """

    def step(
        model: nn.Module, batch: tuple, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one batch. Returns ``(loss, logits, targets)``."""
        windows, targets = batch
        windows, targets = windows.to(device), targets.to(device)
        logits = model(windows)
        return criterion(logits, targets), logits, targets

    return step


def textgen_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Per-epoch validation metrics.

    Perplexity is recomputed from the logits rather than taken from the running
    loss, so it stays correct if the criterion is ever weighted or reduced
    differently.
    """
    tensor_logits = torch.from_numpy(logits)
    tensor_targets = torch.from_numpy(targets).long()
    cross_entropy = nn.functional.cross_entropy(tensor_logits, tensor_targets).item()
    top1 = float((tensor_logits.argmax(dim=1) == tensor_targets).float().mean())
    return {
        "val_perplexity": round(perplexity(cross_entropy), 4),
        "val_top1": round(top1, 6),
    }


def build_loaders(
    splits: TextGenSplits, batch_size: int, seed: int, num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """Build train/validation loaders with a seeded shuffle generator.

    Windows are shuffled for training -- the *split* is contiguous (so no window
    straddles the boundary, D6), but the order in which training windows are
    presented carries no information worth preserving.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        splits.train, batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=num_workers,
    )
    val_loader = DataLoader(
        splits.val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader


def evaluate_textgen(
    model: nn.Module, loader: DataLoader, device: torch.device, vocab_size: int
) -> dict[str, float]:
    """Evaluate and return loss, perplexity and top-1, each with its baseline."""
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.0),
        make_textgen_step(nn.CrossEntropyLoss()), device, clip_grad_norm=None,
    )
    loss, logits, targets = trainer.evaluate(loader)
    extra = textgen_metrics(logits, targets)
    baseline = uniform_perplexity_baseline(vocab_size)
    return {
        "val_loss": loss,
        "perplexity": extra["val_perplexity"],
        "baseline_perplexity": baseline,
        "baseline_cross_entropy": math.log(vocab_size),
        "perplexity_ratio": baseline / extra["val_perplexity"],
        "top1_accuracy": extra["val_top1"],
        "baseline_top1": 1.0 / vocab_size,
        "vocab_size": vocab_size,
    }


def format_textgen_metrics(metrics: dict[str, float]) -> str:
    """Render evaluation results with every metric beside its baseline (C11)."""
    return "\n".join(
        [
            "                    value    baseline      ratio",
            "  perplexity     {:8.2f}   {:9.0f}   {:8.2f}x better".format(
                metrics["perplexity"], metrics["baseline_perplexity"],
                metrics["perplexity_ratio"],
            ),
            "  cross-entropy  {:8.4f}   {:9.4f}".format(
                math.log(metrics["perplexity"]), metrics["baseline_cross_entropy"],
            ),
            "  top-1 accuracy {:8.4f}   {:9.6f}".format(
                metrics["top1_accuracy"], metrics["baseline_top1"],
            ),
        ]
    )


def train_textgen(cfg: TextGenConfig, max_steps: int | None = None) -> Path:
    """Train a next-word model end to end and save its checkpoint.

    Args:
        cfg: Validated configuration.
        max_steps: Cap optimiser steps per epoch, for smoke tests.

    Returns:
        Path to the run directory.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("device: %s", describe_device(device))

    splits = prepare_textgen_data(
        cfg.data.text,
        seq_len=cfg.data.seq_len,
        stride=cfg.data.stride,
        val_fraction=cfg.data.val_fraction,
        min_freq=cfg.data.min_freq,
        strip_boilerplate=cfg.data.strip_gutenberg,
    )
    vocab_size = len(splits.vocab)
    storage_mb = (splits.train.storage_nbytes() + splits.val.storage_nbytes()) / 1e6
    onehot_mb = (
        splits.train.onehot_nbytes(vocab_size) + splits.val.onehot_nbytes(vocab_size)
    ) / 1e6
    logger.info(
        "tokens %d (train %d / val %d)  vocabulary %d  windows %d/%d",
        splits.n_tokens, splits.train.n_tokens, splits.val.n_tokens,
        vocab_size, len(splits.train), len(splits.val),
    )
    logger.info(
        "index storage %.2f MB; the same windows one-hot would be %.0f MB (D9)",
        storage_mb, onehot_mb,
    )

    model = TextGenLSTM(
        vocab_size=vocab_size,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)
    logger.info("model: %s parameters", f"{model.num_parameters():,}")
    logger.info(
        "uniform baseline: cross-entropy %.3f, perplexity %d",
        math.log(vocab_size), vocab_size,
    )

    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = build_loaders(
        splits, cfg.train.batch_size, cfg.seed, cfg.train.num_workers
    )

    es = cfg.train.early_stopping
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        ),
        step_fn=make_textgen_step(criterion),
        device=device,
        clip_grad_norm=cfg.train.clip_grad_norm,
        early_stopping=EarlyStopping(es.monitor, es.mode, es.patience, es.min_delta),
        best_weights=BestWeights(es.monitor, es.mode),
        metrics_fn=textgen_metrics,
        max_steps=max_steps,
    )

    history: TrainingHistory = trainer.fit(train_loader, val_loader, cfg.train.epochs)

    metrics = evaluate_textgen(model, val_loader, device, vocab_size)
    print("\nValidation metrics\n" + format_textgen_metrics(metrics) + "\n")

    run_dir = Path(cfg.output.dir) / datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        run_dir / "best.pt",
        task="textgen",
        model=model,
        model_cfg=model.config(),
        vocab=splits.vocab,
        preprocess={
            "seq_len": cfg.data.seq_len,
            "min_freq": cfg.data.min_freq,
            "strip_gutenberg": cfg.data.strip_gutenberg,
        },
        metrics=metrics,
        train_info={
            "seed": cfg.seed,
            "best_epoch": history.best_epoch,
            "stopped_early": history.stopped_early,
            "epochs_run": len(history.epochs),
        },
    )
    history.save(run_dir / "history.json")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    dump_config(cfg, run_dir / "config.yaml")

    logger.info("run written to %s", run_dir)
    return run_dir
