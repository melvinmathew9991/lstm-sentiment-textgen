"""Command-line entry point: ``train``, ``eval``, ``predict``, ``generate``.

Dispatch is deliberately explicit and covered by a test.  The reference
implementation died at ``engine.py:48`` because a call site passed 4 of 7
required arguments and nothing checked (D1) -- after ~100 epochs of training.

Generation defaults come from the run's ``config.yaml``; an explicit flag
overrides. No sampling literal lives in this file (D10, ``Rules.md`` C13).
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

    Dispatches on the config's ``task`` discriminator; both tasks share the
    same trainer.
    """
    from lstm_nlp.config import load_config

    cfg = load_config(args.config)
    if cfg.task == "sentiment":
        from lstm_nlp.engine.sentiment_task import train_sentiment

        run_dir = train_sentiment(cfg, max_steps=args.max_steps)
    else:
        from lstm_nlp.engine.textgen_task import train_textgen

        run_dir = train_textgen(cfg, max_steps=args.max_steps)

    print(f"\ncheckpoint: {run_dir / 'best.pt'}")
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a checkpoint, printing every metric beside its baseline."""
    from lstm_nlp.data.sentiment import prepare_sentiment_data
    from lstm_nlp.engine.sentiment_task import build_loaders, evaluate_report
    from lstm_nlp.inference.checkpoint import build_model, describe, load_checkpoint
    from lstm_nlp.utils.device import resolve_device

    payload = load_checkpoint(args.ckpt)
    print("Checkpoint\n" + describe(payload) + "\n")

    run_config = Path(args.ckpt).parent / "config.yaml"
    if not run_config.is_file():
        raise CheckpointError(
            f"cannot re-evaluate without the run's config.yaml (looked in "
            f"{run_config.parent}); the checkpoint alone does not record where "
            f"its data came from"
        )

    from lstm_nlp.config import load_config

    cfg = load_config(run_config)

    if payload["task"] == "textgen":
        from lstm_nlp.data.textgen import prepare_textgen_data
        from lstm_nlp.engine.textgen_task import build_loaders as build_textgen_loaders
        from lstm_nlp.engine.textgen_task import evaluate_textgen, format_textgen_metrics

        splits = prepare_textgen_data(
            cfg.data.text,
            seq_len=cfg.data.seq_len,
            stride=cfg.data.stride,
            val_fraction=cfg.data.val_fraction,
            min_freq=cfg.data.min_freq,
            strip_boilerplate=cfg.data.strip_gutenberg,
        )
        model = build_model(payload)
        device = resolve_device(cfg.device)
        model.to(device)
        _, val_loader = build_textgen_loaders(splits, cfg.train.batch_size, cfg.seed)
        metrics = evaluate_textgen(model, val_loader, device, len(splits.vocab))
        print("Validation metrics\n" + format_textgen_metrics(metrics))
        return EXIT_OK

    splits = prepare_sentiment_data(
        cfg.data.csv,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        split_seed=cfg.data.split_seed,
        stratify=cfg.data.stratify,
        deduplicate=cfg.data.deduplicate,
        min_freq=cfg.data.min_freq,
        max_len=cfg.data.max_len,
    )
    model = build_model(payload)
    device = resolve_device(cfg.device)
    model.to(device)

    _, val_loader, test_loader = build_loaders(splits, cfg.train.batch_size, cfg.seed)
    # `--split` now selects a real block. Validation is the set early stopping
    # saw; test is the one it did not, and is the only honest headline.
    loader = val_loader if args.split == "val" else test_loader
    report, _ = evaluate_report(model, loader, device)
    print(f"Metrics on the {args.split} split\n" + report.format())
    return EXIT_OK


def cmd_predict(args: argparse.Namespace) -> int:
    """Classify one or more texts with a sentiment checkpoint."""
    from lstm_nlp.inference.predictor import SentimentPredictor

    predictor = SentimentPredictor(args.ckpt)
    baseline = predictor.metrics.get("baseline_accuracy")

    for text in args.text:
        result = predictor.predict(text)
        print(f"  {text!r}")
        print(f"    {result.label.upper():<9s} p(positive)={result.probabilities['positive']:.3f}")
        if result.n_unk:
            print(
                f"    {result.n_unk} of {result.n_tokens} tokens unknown "
                f"({100 * result.unk_rate:.0f}%) -- treat with caution"
            )
    if baseline is not None:
        print(f"\n  model test accuracy {predictor.metrics['accuracy']:.4f} "
              f"(majority-class baseline {baseline:.4f}, "
              f"macro-F1 {predictor.metrics['macro_f1']:.4f} vs "
              f"{predictor.metrics['baseline_macro_f1']:.4f})")
    return EXIT_OK


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate text from a seed, sampling logits at a temperature.

    Defaults come from the run's config so no literal lives here; an explicit
    flag overrides. This is what replaced the reference's
    ``input_words[-28701]`` magic index (D10).
    """
    from lstm_nlp.config import load_config
    from lstm_nlp.inference.predictor import TextGenerator

    generator = TextGenerator(args.ckpt)

    seed_text, n_words, temperature, top_k = args.seed_text, args.n_words, args.temperature, args.top_k
    run_config = Path(args.ckpt).parent / "config.yaml"
    if run_config.is_file():
        defaults = load_config(run_config).generate
        seed_text = seed_text if seed_text is not None else defaults.seed_text
        n_words = n_words if n_words is not None else defaults.n_words
        temperature = temperature if temperature is not None else defaults.temperature
        top_k = top_k if top_k is not None else defaults.top_k

    if seed_text is None:
        raise CheckpointError(
            "no --seed given and the run's config.yaml is not beside the "
            "checkpoint, so there is no default seed to fall back on"
        )

    result = generator.generate(
        seed_text=seed_text,
        n_words=n_words if n_words is not None else 40,
        temperature=temperature if temperature is not None else 0.7,
        top_k=top_k,
        rng_seed=args.rng_seed,
    )
    print(f"  temperature {result.temperature}   top-k {result.top_k}   "
          f"vocabulary {generator.vocab_size:,}")
    if result.n_unk_in_seed:
        print(f"  {result.n_unk_in_seed} seed word(s) unknown to the model")
    print()
    print(result.text)
    print()

    distribution = generator.next_word_distribution(seed_text, result.temperature, top_k, n=8)
    print(f"  next-word distribution at T={result.temperature} "
          f"(uniform would be {1 / generator.vocab_size:.6f}):")
    for token, probability in distribution:
        bar = "#" * max(1, round(probability * 40))
        print(f"    {token:<16s} {probability:6.4f}  {bar}")
    return EXIT_OK


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
        # Expected, typed failure: report the message, not a traceback.
        logger.error("%s: %s", type(exc).__name__, exc)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        logger.error("file not found: %s", exc)
        return EXIT_ERROR
    except OSError as exc:
        logger.error("could not read or write a file: %s", exc)
        return EXIT_ERROR
    except NotImplementedError as exc:
        logger.error("not implemented yet -- %s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return EXIT_ERROR
    except Exception:
        # An unexpected failure is a bug. Log it with the traceback for
        # diagnosis, but still exit through the documented code rather than
        # dumping a raw trace at the user (Rules.md section 5).
        logger.exception("unexpected error; this is a bug, please report it")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
