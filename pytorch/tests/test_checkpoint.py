"""Phase 2: checkpoints. Regression tests for D8.

The reference saved sentiment_model.h5 and nothing else. Its word_to_int,
int_to_word and sequence_length -- the three objects that define the model's
input space -- were never persisted, so the 5 MB of weights cannot be used at
all. These tests exist to make sure that cannot recur.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

import lstm_nlp
from lstm_nlp.errors import CheckpointError, PreprocessVersionMismatch
from lstm_nlp.inference.checkpoint import (
    build_model,
    describe,
    load_checkpoint,
    save_checkpoint,
)
from lstm_nlp.models.sentiment_lstm import SentimentLSTM
from lstm_nlp.vocab import Vocab


@pytest.fixture
def vocab() -> Vocab:
    return Vocab.build(Counter({"flight": 5, "late": 3, "not": 4, "great": 2}), min_freq=1)


@pytest.fixture
def saved(tmp_path: Path, vocab: Vocab) -> Path:
    torch.manual_seed(0)
    model = SentimentLSTM(vocab_size=len(vocab), embed_dim=8, hidden_dim=8, num_layers=2)
    return save_checkpoint(
        tmp_path / "best.pt",
        task="sentiment",
        model=model,
        model_cfg=model.config(),
        vocab=vocab,
        preprocess={"max_len": 30, "min_freq": 1},
        metrics={"accuracy": 0.8972, "macro_f1": 0.8485,
                 "baseline_accuracy": 0.7953, "baseline_macro_f1": 0.4430},
        train_info={"seed": 42, "best_epoch": 5, "stopped_early": True},
    )


# --------------------------------------------------------------------------- #
# completeness -- the D8 fix
# --------------------------------------------------------------------------- #


def test_checkpoint_carries_the_vocabulary(saved: Path, vocab: Vocab) -> None:
    """Without this the weights are unusable. This is the whole of D8."""
    payload = load_checkpoint(saved)
    assert isinstance(payload["vocab"], Vocab)
    assert payload["vocab"].itos == vocab.itos
    assert payload["vocab"].unk_index == vocab.unk_index


def test_checkpoint_carries_the_preprocessing_contract(saved: Path) -> None:
    payload = load_checkpoint(saved)
    assert payload["preprocess"]["max_len"] == 30
    assert payload["preprocess"]["version"] == lstm_nlp.PREPROCESS_VERSION


def test_checkpoint_records_provenance(saved: Path) -> None:
    payload = load_checkpoint(saved)
    assert payload["lib_versions"]["torch"] == torch.__version__
    assert payload["lib_versions"]["lstm_nlp"] == lstm_nlp.__version__
    assert payload["created_utc"].endswith("+00:00")
    assert payload["train"]["best_epoch"] == 5
    assert payload["train"]["stopped_early"] is True


def test_metrics_travel_with_their_baselines(saved: Path) -> None:
    """C11 survives serialisation: a stored metric keeps its baseline."""
    metrics = load_checkpoint(saved)["metrics"]
    assert metrics["baseline_macro_f1"] == pytest.approx(0.4430)
    assert metrics["macro_f1"] > metrics["baseline_macro_f1"]


def test_checkpoint_is_self_contained(saved: Path, tmp_path: Path) -> None:
    """The honest version of the D8 test: load it in a fresh process.

    The checkpoint is copied alone into an empty directory and loaded from a
    subprocess whose working directory holds nothing else -- no config, no
    dataset, no vocabulary file. If loading needs any other artifact, this
    fails.
    """
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    target = isolated / "only.pt"
    target.write_bytes(saved.read_bytes())

    script = (
        "import sys, torch\n"
        "from lstm_nlp.inference.checkpoint import load_checkpoint, build_model\n"
        "p = load_checkpoint('only.pt')\n"
        "m = build_model(p)\n"
        "v = p['vocab']\n"
        "ids = torch.tensor([v.encode(['flight','not','great'])])\n"
        "out = m(ids, torch.tensor([3]))\n"
        "assert out.shape == (1, 2), out.shape\n"
        "assert len(v) > 2\n"
        "print('SELF_CONTAINED_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=isolated, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "SELF_CONTAINED_OK" in result.stdout


def test_weights_survive_the_round_trip(saved: Path) -> None:
    """Restored predictions must match the saved model bit for bit (S6)."""
    payload = load_checkpoint(saved)
    restored = build_model(payload)
    restored.eval()

    ids = torch.tensor([[2, 3, 4]])
    lengths = torch.tensor([3])
    with torch.no_grad():
        first = restored(ids, lengths)
        second = build_model(load_checkpoint(saved))(ids, lengths)
    assert torch.allclose(first, second, atol=0.0)


def test_build_model_returns_eval_mode(saved: Path) -> None:
    """Dropout active at inference would make predictions nondeterministic."""
    assert build_model(load_checkpoint(saved)).training is False


# --------------------------------------------------------------------------- #
# the version guard -- FR-26
# --------------------------------------------------------------------------- #


def test_stale_preprocess_version_raises(saved: Path) -> None:
    """Loading under different tokenisation rules would silently misread indices."""
    payload = torch.load(saved, map_location="cpu", weights_only=False)
    payload["preprocess"]["version"] = "0-ancient"
    torch.save(payload, saved)

    with pytest.raises(PreprocessVersionMismatch) as exc:
        load_checkpoint(saved)
    assert "0-ancient" in str(exc.value)
    assert lstm_nlp.PREPROCESS_VERSION in str(exc.value)
    assert "retrain" in str(exc.value).lower()


def test_version_guard_can_be_disabled_deliberately(saved: Path) -> None:
    payload = torch.load(saved, map_location="cpu", weights_only=False)
    payload["preprocess"]["version"] = "0-ancient"
    torch.save(payload, saved)
    assert load_checkpoint(saved, strict_version=False)["task"] == "sentiment"


# --------------------------------------------------------------------------- #
# failure modes
# --------------------------------------------------------------------------- #


def test_missing_file_raises() -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint("no/such/checkpoint.pt")


def test_unreadable_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_text("this is not a torch archive", encoding="utf-8")
    with pytest.raises(CheckpointError, match="could not read"):
        load_checkpoint(bad)


@pytest.mark.parametrize("key", ["vocab", "model_state", "model_cfg", "preprocess", "task"])
def test_incomplete_checkpoint_rejected(saved: Path, key: str) -> None:
    """An incomplete checkpoint fails loudly rather than half-loading."""
    payload = torch.load(saved, map_location="cpu", weights_only=False)
    del payload[key]
    torch.save(payload, saved)
    with pytest.raises(CheckpointError, match="incomplete"):
        load_checkpoint(saved)


def test_unknown_model_class_rejected(saved: Path) -> None:
    payload = load_checkpoint(saved)
    payload["model_class"] = "TransformerXL"
    with pytest.raises(CheckpointError, match="unknown model_class"):
        build_model(payload)


def test_describe_shows_metrics_with_baselines(saved: Path) -> None:
    text = describe(load_checkpoint(saved))
    assert "sentiment" in text
    assert "baseline" in text
    assert "best epoch" in text
