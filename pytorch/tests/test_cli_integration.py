"""Phase 5: the CLI as a user actually invokes it.

Two levels, chosen by what each claim actually requires:

* **In-process** (``main(argv)``) for argument handling, exit codes and error
  reporting. These are behaviours of the code, and a subprocess adds ~5s of
  interpreter and torch startup to prove nothing extra.
* **Subprocess**, marked ``slow``, for the claims that are genuinely about the
  process: running from an unrelated working directory, and reproducibility
  across separate invocations. The reference only ran if you happened to be
  standing in ``modular_code/``, so directory independence has to be tested for
  real.

Getting that split wrong is what pushed the suite past its 60s budget (NFR-6)
the first time this file was written.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from lstm_nlp.cli import main

PKG_ROOT = Path(__file__).resolve().parents[1]
RUNS = PKG_ROOT / "runs"
CONFIGS = PKG_ROOT / "configs"
NEUTRAL_CWD = Path(tempfile.gettempdir())

EXIT_OK, EXIT_ERROR, EXIT_BAD_ARGS = 0, 1, 2


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke the CLI in a real subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "lstm_nlp.cli", *args],
        cwd=cwd or NEUTRAL_CWD, capture_output=True, text=True, timeout=900,
    )


def _latest(task: str) -> Path | None:
    found = sorted((RUNS / task).glob("*/best.pt")) if (RUNS / task).is_dir() else []
    return found[-1] if found else None


@pytest.fixture(scope="module")
def sentiment_ckpt() -> Path:
    path = _latest("sentiment")
    if path is None:
        pytest.skip("no sentiment checkpoint; run: lstm-nlp train --config configs/sentiment.yaml")
    return path


@pytest.fixture(scope="module")
def textgen_ckpt() -> Path:
    path = _latest("textgen")
    if path is None:
        pytest.skip("no textgen checkpoint; run: lstm-nlp train --config configs/textgen.yaml")
    return path


# --------------------------------------------------------------------------- #
# arguments and exit codes -- in-process
# --------------------------------------------------------------------------- #


def test_no_command_returns_bad_args(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_BAD_ARGS
    assert "train" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["predict"], ["eval"], ["generate"], ["train"]])
def test_missing_required_flag_exits_two(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == EXIT_BAD_ARGS


@pytest.mark.parametrize("command", ["train", "eval", "predict", "generate"])
def test_subcommand_help_exits_zero(command: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == EXIT_OK


# --------------------------------------------------------------------------- #
# D1 -- the handlers actually execute
# --------------------------------------------------------------------------- #


def test_predict_runs(sentiment_ckpt: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A signature mismatch inside cmd_predict surfaces here in milliseconds.

    In the reference the equivalent line sat after ~100 epochs of training.
    """
    assert main(["predict", "--ckpt", str(sentiment_ckpt), "the flight was not great"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "NEGATIVE" in out or "POSITIVE" in out
    assert "baseline" in out  # C11 reaches the terminal


def test_predict_handles_several_texts(
    sentiment_ckpt: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(
        ["predict", "--ckpt", str(sentiment_ckpt), "great crew", "awful delay", "it was fine"]
    ) == EXIT_OK
    assert capsys.readouterr().out.count("p(positive)") == 3


def test_generate_runs_with_every_sampling_flag(
    textgen_ckpt: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exact call shape that broke the reference, exercised for real."""
    assert main([
        "generate", "--ckpt", str(textgen_ckpt), "--seed", "alice was beginning to",
        "--n-words", "12", "--temperature", "0.8", "--top-k", "20", "--rng-seed", "3",
    ]) == EXIT_OK
    out = capsys.readouterr().out
    assert "alice was beginning to" in out
    assert "next-word distribution" in out


def test_generate_falls_back_to_config_defaults(
    textgen_ckpt: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D10: the demo seed is configuration, not a literal in the source."""
    assert main(["generate", "--ckpt", str(textgen_ckpt), "--n-words", "5"]) == EXIT_OK
    assert "temperature" in capsys.readouterr().out


def test_eval_runs_for_both_tasks(
    sentiment_ckpt: Path, textgen_ckpt: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "--ckpt", str(sentiment_ckpt)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "macro-F1" in out and "baseline" in out

    assert main(["eval", "--ckpt", str(textgen_ckpt)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "perplexity" in out and "baseline" in out


# --------------------------------------------------------------------------- #
# failure modes -- a clear message, never a traceback
# --------------------------------------------------------------------------- #


def _stderr(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().err


def test_missing_checkpoint_reported_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["predict", "--ckpt", "no/such/model.pt", "hello"]) == EXIT_ERROR
    err = _stderr(capsys)
    assert "Traceback" not in err and "not found" in err


def test_missing_config_reported_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["train", "--config", "no/such/config.yaml"]) == EXIT_ERROR
    err = _stderr(capsys)
    assert "Traceback" not in err and "not found" in err


def test_missing_data_file_reported_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A config that points at data which is not there.

    Before Phase 5 this leaked a raw FileNotFoundError traceback: main() caught
    only LstmNlpError, and FileNotFoundError is a builtin.
    """
    config = tmp_path / "bad.yaml"
    config.write_text(
        yaml.safe_dump({"task": "sentiment", "data": {"csv": "absent.csv"}}), encoding="utf-8"
    )
    assert main(["train", "--config", str(config)]) == EXIT_ERROR
    err = _stderr(capsys)
    assert "Traceback" not in err and "not found" in err


def test_malformed_config_reported_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("task: sentiment\ndata:\n  csv: x.csv\n  min_freqq: 3\n", encoding="utf-8")
    assert main(["train", "--config", str(config)]) == EXIT_ERROR
    err = _stderr(capsys)
    assert "Traceback" not in err and "min_freqq" in err


def test_corrupt_checkpoint_reported_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "corrupt.pt"
    bad.write_text("not a torch archive", encoding="utf-8")
    assert main(["predict", "--ckpt", str(bad), "hello"]) == EXIT_ERROR
    assert "Traceback" not in _stderr(capsys)


def test_wrong_task_checkpoint_reported_cleanly(
    textgen_ckpt: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["predict", "--ckpt", str(textgen_ckpt), "hello"]) == EXIT_ERROR
    err = _stderr(capsys)
    assert "Traceback" not in err and "textgen" in err


def test_empty_generation_seed_reported_cleanly(
    textgen_ckpt: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["generate", "--ckpt", str(textgen_ckpt), "--seed", "!!! ???"]) == EXIT_ERROR
    err = _stderr(capsys)
    assert "Traceback" not in err and "empty after cleaning" in err


# --------------------------------------------------------------------------- #
# genuinely about the process -- subprocess, marked slow
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_runs_from_an_unrelated_working_directory(sentiment_ckpt: Path) -> None:
    """Paths resolve from config and __file__, never the current directory."""
    result = run_cli("predict", "--ckpt", str(sentiment_ckpt), "the crew were great")
    assert result.returncode == EXIT_OK, result.stderr
    assert "p(positive)" in result.stdout


@pytest.mark.slow
def test_generation_is_reproducible_across_processes(textgen_ckpt: Path) -> None:
    """FR-23 across process boundaries, not just within one interpreter."""
    args = ("generate", "--ckpt", str(textgen_ckpt), "--seed", "alice was",
            "--n-words", "15", "--rng-seed", "99")
    assert run_cli(*args).stdout == run_cli(*args).stdout


@pytest.mark.slow
def test_help_works_as_an_installed_console_script() -> None:
    result = subprocess.run(
        ["lstm-nlp", "--help"], cwd=NEUTRAL_CWD, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == EXIT_OK
    for command in ("train", "eval", "predict", "generate"):
        assert command in result.stdout


@pytest.mark.slow
@pytest.mark.parametrize("task", ["sentiment", "textgen"])
def test_train_smoke_runs(task: str, tmp_path: Path) -> None:
    """--max-steps keeps this to seconds (Rules.md B7).

    The run is redirected to a temporary directory. Writing into ``runs/`` would
    make this the newest checkpoint, and every test that resolves "the latest
    checkpoint" would then silently assert against a two-step model -- which is
    exactly what happened the first time this test was written.
    """
    from lstm_nlp.config import dump_config, load_config

    cfg = load_config(CONFIGS / f"{task}.yaml")
    cfg = cfg.model_copy(update={
        "train": cfg.train.model_copy(update={"epochs": 1}),
        "output": cfg.output.model_copy(update={"dir": tmp_path / "runs"}),
    })
    config_path = tmp_path / f"{task}.yaml"
    dump_config(cfg, config_path)

    result = run_cli("train", "--config", str(config_path), "--max-steps", "2", cwd=PKG_ROOT)
    assert result.returncode == EXIT_OK, f"{task}: {result.stderr[-800:]}"
    assert "checkpoint:" in result.stdout
    assert list((tmp_path / "runs").glob("*/best.pt")), "checkpoint not written to tmp"
    assert str(RUNS) not in result.stdout, "smoke run wrote into the real runs/ directory"
