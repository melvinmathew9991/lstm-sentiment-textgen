"""Phase 3: assertions against the actually-trained language model.

Skipped unless a checkpoint exists under ``runs/textgen/``. Produce one with:

    lstm-nlp train --config configs/textgen.yaml

The perplexity gate is deliberately expressed against the uniform baseline
rather than as a bare number: a language model's perplexity is meaningless
without the vocabulary size it was measured over (``Rules.md`` C11).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from lstm_nlp.inference.checkpoint import build_model, load_checkpoint

RUNS = Path(__file__).resolve().parents[1] / "runs" / "textgen"

GUTENBERG_WORDS = ("copyright", "donations", "foundation", "ebook", "license",
                   "gutenberg", "trademark", "royalty")


def _latest_checkpoint() -> Path | None:
    if not RUNS.is_dir():
        return None
    found = sorted(RUNS.glob("*/best.pt"))
    return found[-1] if found else None


@pytest.fixture(scope="module")
def trained():
    path = _latest_checkpoint()
    if path is None:
        pytest.skip("no trained checkpoint; run: lstm-nlp train --config configs/textgen.yaml")
    payload = load_checkpoint(path)
    return build_model(payload), payload


# --------------------------------------------------------------------------- #
# S4 -- the perplexity gate, stated against its baseline
# --------------------------------------------------------------------------- #


def test_beats_the_uniform_baseline_by_a_wide_margin(trained) -> None:
    """A model no better than uniform guessing has learned nothing."""
    _, payload = trained
    metrics = payload["metrics"]
    assert metrics["perplexity"] < metrics["baseline_perplexity"]
    assert metrics["perplexity_ratio"] > 5.0, (
        f"only {metrics['perplexity_ratio']:.2f}x better than guessing at random"
    )


def test_meets_the_perplexity_gate(trained) -> None:
    """PRD S4: validation perplexity <= 400 against a 2,436 baseline."""
    _, payload = trained
    perplexity = payload["metrics"]["perplexity"]
    assert perplexity <= 400.0, f"perplexity {perplexity:.2f} above the 400 gate"


def test_baseline_equals_the_vocabulary_size(trained) -> None:
    """Guessing uniformly gives perplexity exactly V. Guards against a
    baseline that silently drifts away from the model it contextualises."""
    _, payload = trained
    metrics = payload["metrics"]
    assert metrics["baseline_perplexity"] == pytest.approx(len(payload["vocab"]))
    assert metrics["baseline_cross_entropy"] == pytest.approx(math.log(len(payload["vocab"])), abs=1e-6)


def test_top1_accuracy_beats_chance(trained) -> None:
    _, payload = trained
    metrics = payload["metrics"]
    assert metrics["top1_accuracy"] > 50 * metrics["baseline_top1"]


def test_early_stopping_fired(trained) -> None:
    """PRD S14 for this task -- evidence D5 is closed, not merely configured."""
    _, payload = trained
    train = payload["train"]
    assert train["stopped_early"] is True
    assert train["best_epoch"] < train["epochs_run"] - 1


# --------------------------------------------------------------------------- #
# D6 -- the model never learned the licence text
# --------------------------------------------------------------------------- #


def test_no_gutenberg_vocabulary_in_the_model(trained) -> None:
    """S15 at the vocabulary level.

    The reference validated entirely on the Project Gutenberg licence and
    seeded generation from it. Here those words have no index at all, so the
    model cannot emit them under any temperature.
    """
    _, payload = trained
    vocab = payload["vocab"]
    for word in GUTENBERG_WORDS:
        assert word not in vocab, f"boilerplate token {word!r} has an index (D6 regressed)"


def test_vocabulary_contains_story_words(trained) -> None:
    """The complement of the check above: the actual book did survive."""
    _, payload = trained
    vocab = payload["vocab"]
    for word in ("alice", "rabbit", "queen", "hatter", "she", "said"):
        assert word in vocab, f"story word {word!r} is missing from the vocabulary"


# --------------------------------------------------------------------------- #
# shape and determinism
# --------------------------------------------------------------------------- #


def test_model_scores_the_whole_vocabulary(trained) -> None:
    model, payload = trained
    seq_len = payload["preprocess"]["seq_len"]
    window = torch.randint(0, len(payload["vocab"]), (1, seq_len))
    with torch.no_grad():
        logits = model(window)
    assert logits.shape == (1, len(payload["vocab"]))


def test_scoring_is_deterministic(trained) -> None:
    """Dropout must be off; only the sampler introduces randomness."""
    model, payload = trained
    window = torch.randint(0, len(payload["vocab"]), (1, payload["preprocess"]["seq_len"]))
    with torch.no_grad():
        first = model(window)
        for _ in range(3):
            assert torch.allclose(model(window), first, atol=0.0)


def test_checkpoint_is_self_contained_for_textgen(trained) -> None:
    """D8 applies to both tasks: the vocabulary travels with the weights."""
    _, payload = trained
    assert len(payload["vocab"]) > 1000
    assert payload["preprocess"]["seq_len"] == 10
    assert payload["preprocess"]["strip_gutenberg"] is True
    assert payload["model_class"] == "TextGenLSTM"
