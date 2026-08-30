"""Phase 9 (S10): the same seed must produce the same model, twice.

Reproducibility is the property every other measured claim in this project
rests on. `PARITY.md` says its figures reproduce; without this test that is an
assertion about the code rather than a fact about it.

Two full training runs from the same config, compared on their metrics. Marked
`slow` because it trains twice; CI runs `-m ""` and therefore runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from lstm_nlp.config import load_config
from lstm_nlp.data.sentiment import prepare_sentiment_data
from lstm_nlp.engine.sentiment_task import train_sentiment
from lstm_nlp.models.sentiment_lstm import SentimentLSTM
from lstm_nlp.utils.seed import set_seed

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# --------------------------------------------------------------------------- #
# the cheap determinism properties, checked without training
# --------------------------------------------------------------------------- #


def test_the_split_is_deterministic(sample_csv: Path) -> None:
    """Same seed, same rows in every block -- or nothing downstream compares."""
    a = prepare_sentiment_data(sample_csv, min_freq=1)
    b = prepare_sentiment_data(sample_csv, min_freq=1)
    for block in ("train", "val", "test"):
        assert list(getattr(a, block).texts) == list(getattr(b, block).texts)
    assert a.vocab.itos == b.vocab.itos


def test_a_different_seed_gives_a_different_split(sample_csv: Path) -> None:
    """The control: if every seed gave the same split, the test above is empty."""
    a = prepare_sentiment_data(sample_csv, min_freq=1, split_seed=10)
    b = prepare_sentiment_data(sample_csv, min_freq=1, split_seed=11)
    assert list(a.test.texts) != list(b.test.texts)


def test_weight_initialisation_is_reproducible() -> None:
    set_seed(42)
    first = SentimentLSTM(vocab_size=50, embed_dim=8, hidden_dim=8, num_layers=1)
    set_seed(42)
    second = SentimentLSTM(vocab_size=50, embed_dim=8, hidden_dim=8, num_layers=1)
    for (name, a), (_, b) in zip(
        first.state_dict().items(), second.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), f"{name} differs between identically seeded builds"


# --------------------------------------------------------------------------- #
# S10 -- two real runs, identical metrics
# --------------------------------------------------------------------------- #


def _smoke_config(tmp_path: Path, sample_csv: Path, run_dir: Path) -> Path:
    """A tiny but complete training config pointed at the fixture corpus."""
    cfg = yaml.safe_load((CONFIGS / "sentiment.yaml").read_text(encoding="utf-8"))
    cfg["data"]["csv"] = str(sample_csv)
    cfg["data"]["min_freq"] = 1
    cfg["model"] = {**cfg["model"], "embed_dim": 16, "hidden_dim": 16, "num_layers": 1}
    cfg["train"] = {**cfg["train"], "epochs": 2, "batch_size": 16}
    cfg["output"]["dir"] = str(run_dir)
    path = tmp_path / f"{run_dir.name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.slow
def test_two_seeded_runs_produce_identical_metrics(
    tmp_path: Path, sample_csv: Path
) -> None:
    """S10. Every measured claim in this repository depends on this holding."""
    metrics = []
    for name in ("first", "second"):
        run_root = tmp_path / name
        config = _smoke_config(tmp_path, sample_csv, run_root)
        run_dir = train_sentiment(load_config(config))
        metrics.append(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")))

    first, second = metrics
    assert first["macro_f1"] == second["macro_f1"]
    assert first["accuracy"] == second["accuracy"]
    assert first["confusion"] == second["confusion"]
    assert first["roc_auc"] == second["roc_auc"]


@pytest.mark.slow
def test_the_fitted_temperature_is_reproducible(tmp_path: Path, sample_csv: Path) -> None:
    """Calibration is part of the artifact, so it has to reproduce too."""
    temperatures = []
    for name in ("cal_a", "cal_b"):
        config = _smoke_config(tmp_path, sample_csv, tmp_path / name)
        run_dir = train_sentiment(load_config(config))
        payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        temperatures.append(payload["train"]["calibration"]["temperature"])
    assert temperatures[0] == temperatures[1]
