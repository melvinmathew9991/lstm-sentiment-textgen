"""Phase 0: config loading, validation and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pydantic import ValidationError

from lstm_nlp.config import SentimentConfig, TextGenConfig, dump_config, load_config
from lstm_nlp.errors import ConfigError

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


# --------------------------------------------------------------------------- #
# the shipped configs
# --------------------------------------------------------------------------- #


def test_sentiment_config_loads() -> None:
    cfg = load_config(CONFIG_DIR / "sentiment.yaml")
    assert isinstance(cfg, SentimentConfig)
    assert cfg.data.min_freq == 2
    assert cfg.data.split_seed == 10
    assert cfg.data.stratify is True
    assert cfg.train.class_weighting == "balanced"
    # Accuracy is not a safe stopping monitor here: the majority baseline is 0.795.
    assert cfg.train.early_stopping.monitor == "val_macro_f1"
    assert cfg.train.early_stopping.mode == "max"


def test_textgen_config_loads() -> None:
    cfg = load_config(CONFIG_DIR / "textgen.yaml")
    assert isinstance(cfg, TextGenConfig)
    assert cfg.data.seq_len == 10
    assert cfg.data.strip_gutenberg is True
    assert cfg.generate.temperature == 0.7


def test_shipped_configs_point_at_real_data() -> None:
    """Paths resolve against the config's directory, not the CWD."""
    sentiment = load_config(CONFIG_DIR / "sentiment.yaml")
    textgen = load_config(CONFIG_DIR / "textgen.yaml")
    assert sentiment.data.csv.is_absolute()
    assert sentiment.data.csv.is_file(), f"missing: {sentiment.data.csv}"
    assert textgen.data.text.is_file(), f"missing: {textgen.data.text}"
    assert sentiment.output.dir.is_absolute()


def test_discriminator_picks_the_right_model() -> None:
    assert load_config(CONFIG_DIR / "sentiment.yaml").task == "sentiment"
    assert load_config(CONFIG_DIR / "textgen.yaml").task == "textgen"


# --------------------------------------------------------------------------- #
# failure modes  (Rules.md section 5: fail loudly, never default past a bad value)
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _valid_sentiment() -> dict:
    return {"task": "sentiment", "data": {"csv": "x.csv"}}


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_non_mapping_raises(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="expected a YAML mapping"):
        load_config(path)


def test_unknown_task_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, {"task": "translation", "data": {}}))


def test_typo_in_key_is_rejected(tmp_path: Path) -> None:
    """extra='forbid': a silently-ignored typo is how hyperparameters go missing."""
    payload = _valid_sentiment()
    payload["data"]["min_freqq"] = 3
    with pytest.raises(ConfigError, match="min_freqq"):
        load_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("block", "key", "value"),
    [
        ("data", "test_size", 1.5),
        ("data", "min_freq", 0),
        ("data", "max_len", 0),
        ("model", "dropout", 1.0),
        ("model", "embed_dim", 0),
        ("train", "lr", 0.0),
        ("train", "batch_size", 0),
        ("train", "clip_grad_norm", -1.0),
    ],
)
def test_out_of_range_values_rejected(tmp_path: Path, block: str, key: str, value: object) -> None:
    payload = _valid_sentiment()
    payload.setdefault(block, {})[key] = value
    with pytest.raises(ConfigError, match=key):
        load_config(_write(tmp_path, payload))


def test_unknown_early_stopping_monitor_rejected(tmp_path: Path) -> None:
    payload = _valid_sentiment()
    payload["train"] = {"early_stopping": {"monitor": "val_bleu", "mode": "max"}}
    with pytest.raises(ConfigError, match="val_bleu"):
        load_config(_write(tmp_path, payload))


def test_temperature_must_be_positive(tmp_path: Path) -> None:
    """Guards C2 at the config boundary: T=0 would divide logits by zero."""
    payload = {"task": "textgen", "data": {"text": "x.txt"}, "generate": {"temperature": 0.0}}
    with pytest.raises(ConfigError, match="temperature"):
        load_config(_write(tmp_path, payload))


# --------------------------------------------------------------------------- #
# round-trip
# --------------------------------------------------------------------------- #


def test_dump_then_load_round_trips(tmp_path: Path) -> None:
    """A run records the resolved config, and that record is itself loadable."""
    original = load_config(CONFIG_DIR / "sentiment.yaml")
    out = tmp_path / "resolved.yaml"
    dump_config(original, out)
    assert load_config(out) == original


def test_config_is_frozen() -> None:
    cfg = load_config(CONFIG_DIR / "sentiment.yaml")
    with pytest.raises(ValidationError):
        cfg.seed = 999
