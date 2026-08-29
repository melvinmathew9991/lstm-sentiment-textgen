"""Configuration schema and loader.

Every hyperparameter, path and threshold lives in YAML and is validated here.
No literal that a user might reasonably want to change belongs in code
(Rules.md C13) -- the reference implementation's ``input_type = 2`` and
``input_words[-28701]`` are the anti-patterns this module exists to prevent.

Path convention
---------------
Relative paths in a config file are resolved against **the directory containing
that config file**.  One rule, applied everywhere, so a config is portable and
never depends on the current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from lstm_nlp.errors import ConfigError


class _Base(BaseModel):
    """Shared strictness: unknown keys are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# shared blocks
# --------------------------------------------------------------------------- #


class EarlyStoppingConfig(_Base):
    """When to stop, and which epoch to keep (PRD FR-13, fixes D5)."""

    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    patience: int = Field(5, ge=1)
    min_delta: float = Field(0.0, ge=0.0)


class TrainConfig(_Base):
    batch_size: int = Field(64, ge=1)
    epochs: int = Field(40, ge=1)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    clip_grad_norm: float = Field(5.0, gt=0)
    class_weighting: Literal["none", "balanced"] = "none"
    num_workers: int = Field(0, ge=0)
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


class OutputConfig(_Base):
    dir: Path = Path("../runs")


class ModelConfig(_Base):
    embed_dim: int = Field(64, ge=1)
    hidden_dim: int = Field(64, ge=1)
    num_layers: int = Field(1, ge=1)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)


# --------------------------------------------------------------------------- #
# task-specific data blocks
# --------------------------------------------------------------------------- #


class SentimentDataConfig(_Base):
    csv: Path
    test_size: float = Field(0.30, gt=0, lt=1)
    split_seed: int = 10  # pinned to match the frozen TF reference
    stratify: bool = True
    min_freq: int = Field(2, ge=1)
    max_len: int = Field(30, ge=1)


class TextGenDataConfig(_Base):
    text: Path
    seq_len: int = Field(10, ge=1)
    stride: int = Field(1, ge=1)
    val_fraction: float = Field(0.10, gt=0, lt=1)
    min_freq: int = Field(2, ge=1)
    strip_gutenberg: bool = True  # fixes D6; switchable only for testing


class GenerateConfig(_Base):
    """Defaults for the ``generate`` command.

    ``seed_text`` replaces the reference's ``input_words[-28701]`` magic index
    (D10) -- a demo seed is a configuration value, not a literal.
    """

    seed_text: str = "alice was beginning to"
    n_words: int = Field(40, ge=1, le=500)
    temperature: float = Field(0.7, gt=0, le=5.0)
    top_k: int | None = Field(None, ge=1)


# --------------------------------------------------------------------------- #
# top-level configs
# --------------------------------------------------------------------------- #


class SentimentConfig(_Base):
    task: Literal["sentiment"]
    seed: int = Field(42, ge=0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    data: SentimentDataConfig
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    output: OutputConfig = OutputConfig()

    @field_validator("train")
    @classmethod
    def _monitor_is_known(cls, v: TrainConfig) -> TrainConfig:
        allowed = {"val_loss", "val_accuracy", "val_macro_f1"}
        if v.early_stopping.monitor not in allowed:
            raise ValueError(
                f"early_stopping.monitor={v.early_stopping.monitor!r} "
                f"is not one of {sorted(allowed)}"
            )
        return v


class TextGenConfig(_Base):
    task: Literal["textgen"]
    seed: int = Field(42, ge=0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    data: TextGenDataConfig
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    generate: GenerateConfig = GenerateConfig()
    output: OutputConfig = OutputConfig()

    @field_validator("train")
    @classmethod
    def _monitor_is_known(cls, v: TrainConfig) -> TrainConfig:
        allowed = {"val_loss", "val_perplexity"}
        if v.early_stopping.monitor not in allowed:
            raise ValueError(
                f"early_stopping.monitor={v.early_stopping.monitor!r} "
                f"is not one of {sorted(allowed)}"
            )
        return v


AnyConfig = Annotated[SentimentConfig | TextGenConfig, Field(discriminator="task")]
_ADAPTER: TypeAdapter[AnyConfig] = TypeAdapter(AnyConfig)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _resolve_paths(cfg: SentimentConfig | TextGenConfig, base: Path) -> dict:
    """Return ``cfg`` as a dict with every relative path made absolute."""
    data = cfg.model_dump()
    if isinstance(cfg, SentimentConfig):
        data["data"]["csv"] = _abs(cfg.data.csv, base)
    else:
        data["data"]["text"] = _abs(cfg.data.text, base)
    data["output"]["dir"] = _abs(cfg.output.dir, base)
    return data


def _abs(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> SentimentConfig | TextGenConfig:
    """Load, validate, and path-resolve a YAML config.

    Args:
        path: Path to the YAML file.

    Returns:
        A frozen ``SentimentConfig`` or ``TextGenConfig``, chosen by the ``task``
        key, with all relative paths resolved against the config's directory.

    Raises:
        ConfigError: If the file is missing, is not a YAML mapping, or fails
            schema validation.  The message names the offending field -- a
            config is never silently defaulted past a value the user set wrongly
            (Rules.md section 5).
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path.resolve()}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: expected a YAML mapping, got {type(raw).__name__}")

    try:
        cfg = _ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise ConfigError(f"{config_path}: {_format_errors(exc)}") from exc

    resolved = _resolve_paths(cfg, config_path.parent.resolve())
    return _ADAPTER.validate_python(resolved)


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"{loc}: {err['msg']}")
    return "; ".join(lines)


def dump_config(cfg: SentimentConfig | TextGenConfig, path: Path) -> None:
    """Write the fully resolved config beside a run's artifacts.

    Saves the merged, defaulted, validated object -- not the input file -- so a
    run records exactly what it did (Rules.md section 7).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
