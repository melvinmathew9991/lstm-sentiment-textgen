"""Self-contained checkpoints.

The fix for D8. The reference saved ``sentiment_model.h5`` and nothing else --
no ``word_to_int``, no ``int_to_word``, no ``sequence_length``. Those three
objects *define the model's input space*, so without them the 5 MB of weights
cannot be used: there is no way to turn a string into the indices the embedding
table expects. A model you cannot run is not a saved model.

Every checkpoint here therefore carries weights, model config, the full
vocabulary, the preprocessing contract and its version, plus the metrics and
library versions of the run that produced it. Loading one requires no other
file (``Rules.md`` C9, PRD FR-25).

``PREPROCESS_VERSION`` is checked on load. A mismatch raises rather than loads,
because stored indices would otherwise be silently reinterpreted under different
tokenisation rules -- a wrong answer, confidently given, which is the failure
mode this whole project exists to correct.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import lstm_nlp
from lstm_nlp.errors import CheckpointError, PreprocessVersionMismatch
from lstm_nlp.vocab import Vocab

FORMAT_VERSION = 1

REQUIRED_KEYS = (
    "format_version", "task", "model_class", "model_cfg", "model_state",
    "vocab", "preprocess",
)


def _library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "lstm_nlp": lstm_nlp.__version__,
    }


def save_checkpoint(
    path: str | Path,
    *,
    task: str,
    model: torch.nn.Module,
    model_cfg: dict[str, Any],
    vocab: Vocab,
    preprocess: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    train_info: dict[str, Any] | None = None,
) -> Path:
    """Write a self-contained checkpoint.

    Args:
        path: Destination ``.pt`` file. Parent directories are created.
        task: ``"sentiment"`` or ``"textgen"``.
        model: Model whose ``state_dict`` is saved. Pass it **after** best-weight
            restoration -- what lands here must never be merely the last epoch
            (D5).
        model_cfg: Constructor arguments, enough to rebuild the architecture.
        vocab: The vocabulary. Without this the weights are unusable (D8).
        preprocess: Tokenisation contract, e.g. ``max_len``.
        metrics: Final metrics, baselines included.
        train_info: Seed, best epoch, class weights, early-stopping outcome.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": FORMAT_VERSION,
        "task": task,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lib_versions": _library_versions(),
        "model_class": type(model).__name__,
        "model_cfg": dict(model_cfg),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "vocab": vocab.to_dict(),
        "preprocess": {**preprocess, "version": lstm_nlp.PREPROCESS_VERSION},
        "metrics": metrics or {},
        "train": train_info or {},
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(path: str | Path, *, strict_version: bool = True) -> dict[str, Any]:
    """Load a checkpoint, verifying it is complete and current.

    Args:
        path: The ``.pt`` file.
        strict_version: Enforce the preprocessing-version guard. Only tests
            that deliberately exercise a stale checkpoint should disable it.

    Returns:
        The payload dict, with ``vocab`` rehydrated into a :class:`Vocab`.

    Raises:
        CheckpointError: If the file is missing, unreadable, or lacks a
            required key.
        PreprocessVersionMismatch: If it was written under different
            tokenisation rules.
    """
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint not found: {source.resolve()}")

    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch raises a variety of types for bad files
        raise CheckpointError(f"could not read checkpoint {source}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CheckpointError(f"{source}: expected a dict payload, got {type(payload).__name__}")

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise CheckpointError(
            f"{source}: incomplete checkpoint, missing {missing}. "
            f"A checkpoint must be loadable with no other file present."
        )

    stored = payload["preprocess"].get("version")
    if strict_version and stored != lstm_nlp.PREPROCESS_VERSION:
        raise PreprocessVersionMismatch(str(stored), lstm_nlp.PREPROCESS_VERSION)

    payload["vocab"] = Vocab.from_dict(payload["vocab"])
    return payload


def build_model(payload: dict[str, Any]) -> torch.nn.Module:
    """Reconstruct the model described by a loaded checkpoint.

    Args:
        payload: Output of :func:`load_checkpoint`.

    Returns:
        The model in eval mode with its weights loaded.

    Raises:
        CheckpointError: If ``model_class`` is not one this package defines.
    """
    from lstm_nlp.models.sentiment_lstm import SentimentLSTM

    registry: dict[str, type[torch.nn.Module]] = {"SentimentLSTM": SentimentLSTM}

    name = payload["model_class"]
    if name not in registry:
        raise CheckpointError(
            f"unknown model_class {name!r}; this build knows {sorted(registry)}"
        )

    model = registry[name](**payload["model_cfg"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def describe(payload: dict[str, Any]) -> str:
    """One-block human summary of a checkpoint, for the CLI and API."""
    metrics = payload.get("metrics", {})
    train = payload.get("train", {})
    lines = [
        f"  task            {payload['task']}",
        f"  model           {payload['model_class']}",
        f"  vocabulary      {len(payload['vocab']):,}",
        f"  created         {payload.get('created_utc', '?')}",
        f"  preprocess ver  {payload['preprocess'].get('version')}",
    ]
    if "best_epoch" in train:
        lines.append(f"  best epoch      {train['best_epoch'] + 1}"
                     f"{'  (stopped early)' if train.get('stopped_early') else ''}")
    if "macro_f1" in metrics:
        lines.append(
            f"  macro-F1        {metrics['macro_f1']:.4f}"
            f"   baseline {metrics.get('baseline_macro_f1', float('nan')):.4f}"
        )
    if "accuracy" in metrics:
        lines.append(
            f"  accuracy        {metrics['accuracy']:.4f}"
            f"   baseline {metrics.get('baseline_accuracy', float('nan')):.4f}"
        )
    return "\n".join(lines)
