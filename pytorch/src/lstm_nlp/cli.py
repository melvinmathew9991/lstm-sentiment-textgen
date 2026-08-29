"""Command-line entry point: ``train``, ``eval``, ``predict``, ``generate``.

Phase 0 wires argument parsing and dispatch only; the handlers raise
``NotImplementedError`` until their phases land (see ``Phases.md``).

Dispatch is deliberately explicit and covered by a test.  The reference
implementation died at ``engine.py:48`` because a call site passed 4 of 7
required arguments and nothing checked (D1) -- after ~100 epochs of training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lstm_nlp import __version__
from lstm_nlp.errors import CheckpointError, LstmNlpError
from lstm_nlp.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BAD_ARGS = 2


# --------------------------------------------------------------------------- #
# handlers  (implemented in later phases)
# --------------------------------------------------------------------------- #


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model from a config file.

    Text generation lands in Phase 3; until then that task reports cleanly
    rather than half-running.
    """
    from lstm_nlp.config import load_config

    cfg = load_config(args.config)
    if cfg.task == "sentiment":
        from lstm_nlp.engine.sentiment_task import train_sentiment

        run_dir = train_sentiment(cfg, max_steps=args.max_steps)
        print(f"\ncheckpoint: {run_dir / 'best.pt'}")
        return EXIT_OK

    raise NotImplementedError(f"training for task {cfg.task!r} lands in Phase 3")


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a checkpoint, printing every metric beside its baseline."""
    from lstm_nlp.data.sentiment import prepare_sentiment_data
    from lstm_nlp.engine.sentiment_task import build_loaders, evaluate_report
    from lstm_nlp.inference.checkpoint import build_model, describe, load_checkpoint
    from lstm_nlp.utils.device import resolve_device

    payload = load_checkpoint(args.ckpt)
    print("Checkpoint\n" + describe(payload) + "\n")

    if payload["task"] != "sentiment":
        raise NotImplementedError(f"evaluation for task {payload['task']!r} lands in Phase 3")

    run_config = Path(args.ckpt).parent / "config.yaml"
    if not run_config.is_file():
        raise CheckpointError(
            f"cannot re-evaluate without the run's config.yaml (looked in "
            f"{run_config.parent}); the checkpoint alone does not record where "
            f"its data came from"
        )

    from lstm_nlp.config import load_config

    cfg = load_config(run_config)
    splits = prepare_sentiment_data(
        cfg.data.csv,
        test_size=cfg.data.test_size,
        split_seed=cfg.data.split_seed,
        stratify=cfg.data.stratify,
        min_freq=cfg.data.min_freq,
        max_len=cfg.data.max_len,
    )
    model = build_model(payload)
    device = resolve_device(cfg.device)
    model.to(device)

    _, test_loader = build_loaders(splits, cfg.train.batch_size, cfg.seed)
    report, _ = evaluate_report(model, test_loader, device)
    print(f"Metrics on the {args.split} split\n" + report.format())
    return EXIT_OK


def cmd_predict(args: argparse.Namespace) -> int:
    """Classify one or more texts with a sentiment checkpoint.  Phase 4."""
    raise NotImplementedError("predict lands in Phase 4")


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate text from a seed with temperature sampling.  Phase 4."""
    raise NotImplementedError("generate lands in Phase 4")


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="lstm-nlp",
        description=(
            "Many-to-one LSTMs for sentiment detection and text generation (PyTorch)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"lstm-nlp {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging threshold (default: INFO)",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # -- train ------------------------------------------------------------- #
    p_train = sub.add_parser("train", help="train a model from a YAML config")
    p_train.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    p_train.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="cap optimiser steps per epoch; for smoke tests (Rules.md B7)",
    )
    p_train.set_defaults(func=cmd_train)

    # -- eval -------------------------------------------------------------- #
    p_eval = sub.add_parser("eval", help="evaluate a checkpoint against its baselines")
    p_eval.add_argument("--ckpt", type=Path, required=True, help="path to a .pt checkpoint")
    p_eval.add_argument(
        "--split", default="test", choices=["train", "val", "test"], help="split to evaluate"
    )
    p_eval.set_defaults(func=cmd_eval)

    # -- predict ----------------------------------------------------------- #
    p_predict = sub.add_parser("predict", help="classify text with a sentiment checkpoint")
    p_predict.add_argument("--ckpt", type=Path, required=True, help="path to a .pt checkpoint")
    p_predict.add_argument("text", nargs="+", help="one or more texts to classify")
    p_predict.set_defaults(func=cmd_predict)

    # -- generate ---------------------------------------------------------- #
    p_gen = sub.add_parser("generate", help="generate text from a seed")
    p_gen.add_argument("--ckpt", type=Path, required=True, help="path to a .pt checkpoint")
    p_gen.add_argument("--seed", dest="seed_text", default=None, help="seed text")
    p_gen.add_argument("--n-words", type=int, default=None, help="words to generate")
    p_gen.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="softmax temperature applied to LOGITS; >0. Lower is greedier.",
    )
    p_gen.add_argument("--top-k", type=int, default=None, help="restrict sampling to top k logits")
    p_gen.add_argument(
        "--rng-seed", type=int, default=None, help="RNG seed, for reproducible generation"
    )
    p_gen.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch.

    Returns:
        ``0`` on success, ``1`` on a handled runtime error, ``2`` on bad
        arguments.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "func", None) is None:
        parser.print_help()
        return EXIT_BAD_ARGS

    setup_logging(args.log_level)

    try:
        return args.func(args)
    except LstmNlpError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return EXIT_ERROR
    except NotImplementedError as exc:
        logger.error("not implemented yet -- %s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
