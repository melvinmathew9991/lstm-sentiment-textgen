"""Phase 0: CLI parsing and dispatch.

The reference implementation crashed at ``engine.py:48`` because a call site
passed 4 of 7 required arguments and nothing verified it -- after ~100 epochs of
training (D1).  These tests assert that every subcommand actually reaches its
handler with the arguments the handler expects, so that class of failure is
caught at test time rather than at the end of a training run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from lstm_nlp import __version__
from lstm_nlp.cli import EXIT_BAD_ARGS, build_parser, main

EXPECTED_COMMANDS = {"train", "eval", "predict", "generate"}


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no public API for this
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return action.choices
    raise AssertionError("parser has no subcommands")


def test_all_four_subcommands_exist() -> None:
    assert set(_subparsers(build_parser())) == EXPECTED_COMMANDS


def test_every_subcommand_binds_a_handler() -> None:
    """A subcommand with no ``func`` would fall through to the help text."""
    parser = build_parser()
    for name, sub in _subparsers(parser).items():
        assert callable(sub.get_default("func")), f"{name} has no handler bound"


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    assert "COMMAND" in capsys.readouterr().out


@pytest.mark.parametrize("command", sorted(EXPECTED_COMMANDS))
def test_subcommand_help_exits_zero(command: str) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "--help"])
    assert exc.value.code == 0


def test_version_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert __version__


def test_no_command_prints_help_and_returns_bad_args() -> None:
    assert main([]) == EXIT_BAD_ARGS


# --------------------------------------------------------------------------- #
# argument binding -- the D1 regression surface
# --------------------------------------------------------------------------- #


def test_train_args_parse() -> None:
    args = build_parser().parse_args(["train", "--config", "c.yaml", "--max-steps", "20"])
    assert args.config == Path("c.yaml")
    assert args.max_steps == 20
    assert args.func.__name__ == "cmd_train"


def test_eval_args_parse() -> None:
    args = build_parser().parse_args(["eval", "--ckpt", "best.pt"])
    assert args.ckpt == Path("best.pt")
    assert args.split == "test"


def test_predict_takes_multiple_texts() -> None:
    args = build_parser().parse_args(["predict", "--ckpt", "b.pt", "one text", "two text"])
    assert args.text == ["one text", "two text"]


def test_generate_binds_every_sampling_argument() -> None:
    """Every knob the sampler needs must survive parsing (fixes the D1 shape)."""
    args = build_parser().parse_args(
        [
            "generate", "--ckpt", "b.pt",
            "--seed", "alice was",
            "--n-words", "25",
            "--temperature", "0.9",
            "--top-k", "50",
            "--rng-seed", "7",
        ]
    )
    assert args.seed_text == "alice was"
    assert args.n_words == 25
    assert args.temperature == pytest.approx(0.9)
    assert args.top_k == 50
    assert args.rng_seed == 7


def test_generate_defaults_are_none_so_config_can_supply_them() -> None:
    args = build_parser().parse_args(["generate", "--ckpt", "b.pt"])
    assert (args.seed_text, args.n_words, args.temperature, args.top_k) == (None,) * 4


@pytest.mark.parametrize(
    "argv",
    [
        ["train"],                      # missing --config
        ["eval"],                       # missing --ckpt
        ["predict", "--ckpt", "b.pt"],  # missing text
        ["generate"],                   # missing --ckpt
        ["eval", "--ckpt", "b.pt", "--split", "holdout"],  # bad choice
    ],
)
def test_missing_required_arguments_exit_two(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == EXIT_BAD_ARGS


# --------------------------------------------------------------------------- #
# Phase 0 placeholders
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--config", "c.yaml"],
        ["eval", "--ckpt", "b.pt"],
        ["predict", "--ckpt", "b.pt", "hello"],
        ["generate", "--ckpt", "b.pt"],
    ],
)
def test_handlers_report_not_implemented_cleanly(argv: list[str]) -> None:
    """Unimplemented commands exit 1 with a log line -- never a raw traceback."""
    assert main(argv) == 1
