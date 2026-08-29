"""Phase 2: assertions against the actually-trained model.

Skipped unless a checkpoint exists under ``runs/sentiment/``, so the suite stays
fast and offline by default (NFR-6). Produce one with:

    lstm-nlp train --config configs/sentiment.yaml

This is where D3 stops being a preprocessing detail and becomes a measurable
property of the model: a classifier trained on negation-stripped text cannot
tell "great" from "not great", because it never saw the word that matters.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lstm_nlp.data.preprocess import clean_tweet, tokenize
from lstm_nlp.inference.checkpoint import build_model, load_checkpoint

RUNS = Path(__file__).resolve().parents[1] / "runs" / "sentiment"

NEGATIVE, POSITIVE = 0, 1


def _latest_checkpoint() -> Path | None:
    if not RUNS.is_dir():
        return None
    found = sorted(RUNS.glob("*/best.pt"))
    return found[-1] if found else None


@pytest.fixture(scope="module")
def trained():
    path = _latest_checkpoint()
    if path is None:
        pytest.skip("no trained checkpoint; run: lstm-nlp train --config configs/sentiment.yaml")
    payload = load_checkpoint(path)
    return build_model(payload), payload


def predict(trained, text: str) -> tuple[int, float]:
    """Return ``(label, p_positive)`` for one string."""
    model, payload = trained
    vocab = payload["vocab"]
    max_len = payload["preprocess"]["max_len"]

    ids = vocab.encode(tokenize(clean_tweet(text)))[:max_len] or [vocab.unk_index]
    with torch.no_grad():
        logits = model(torch.tensor([ids]), torch.tensor([len(ids)]))
        probs = torch.softmax(logits, dim=1)[0]
    return int(probs.argmax()), float(probs[POSITIVE])


# --------------------------------------------------------------------------- #
# S8 -- the negation test. The payoff for D3.
# --------------------------------------------------------------------------- #


def test_negation_changes_the_prediction(trained) -> None:
    """"great" and "not great" must not look the same to the model.

    Under the reference's preprocessing these two strings were *identical* after
    stopword removal, so no model trained that way could possibly separate them.
    """
    positive_label, p_pos = predict(trained, "the flight was great")
    negated_label, p_neg = predict(trained, "the flight was not great")

    gap = p_pos - p_neg
    assert positive_label != negated_label or gap > 0.15, (
        f"negation barely moved the model: p(positive) {p_pos:.3f} -> {p_neg:.3f} "
        f"(gap {gap:.3f}); D3 may have regressed"
    )


@pytest.mark.parametrize(
    ("plain", "negated"),
    [
        ("the flight was great", "the flight was not great"),
        ("service was good", "service was not good"),
        ("i would recommend this airline", "i would not recommend this airline"),
    ],
)
def test_negation_lowers_positive_probability(trained, plain: str, negated: str) -> None:
    """Adding a negation must reduce p(positive), on several phrasings."""
    _, p_plain = predict(trained, plain)
    _, p_negated = predict(trained, negated)
    assert p_negated < p_plain, (
        f"negating {plain!r} did not lower p(positive): {p_plain:.3f} -> {p_negated:.3f}"
    )


def test_negations_are_in_the_vocabulary(trained) -> None:
    """The structural precondition: the model must have an index for "not"."""
    _, payload = trained
    vocab = payload["vocab"]
    for word in ("not", "no", "never"):
        assert word in vocab, f"{word!r} has no index -- stopword removal has returned (D3)"


# --------------------------------------------------------------------------- #
# quality gates
# --------------------------------------------------------------------------- #


def test_meets_the_macro_f1_gate(trained) -> None:
    """PRD S3: macro-F1 >= 0.75 against a 0.4430 baseline."""
    _, payload = trained
    metrics = payload["metrics"]
    assert metrics["macro_f1"] >= 0.75, f"macro-F1 {metrics['macro_f1']:.4f} below the 0.75 gate"
    assert metrics["macro_f1"] > metrics["baseline_macro_f1"] + 0.30


def test_detects_the_minority_class(trained) -> None:
    """What class weighting bought. A baseline model scores 0 recall here."""
    _, payload = trained
    positive = payload["metrics"]["per_class"]["positive"]
    assert positive["recall"] > 0.5, f"positive recall {positive['recall']:.3f} is baseline-like"
    assert positive["f1"] > 0.5


def test_early_stopping_fired(trained) -> None:
    """PRD S14, and the evidence that D5 is closed rather than merely configured."""
    _, payload = trained
    train = payload["train"]
    assert train["stopped_early"] is True, "training ran to the epoch cap"
    assert train["best_epoch"] < train["epochs_run"] - 1, (
        "the best epoch was the last one, so best-weight restore proved nothing"
    )


def test_unknown_words_do_not_crash(trained) -> None:
    """D7 at the model boundary."""
    label, prob = predict(trained, "qwertyuiop zxcvbnm flurbulate")
    assert label in (NEGATIVE, POSITIVE)
    assert 0.0 <= prob <= 1.0


def test_empty_input_does_not_crash(trained) -> None:
    """Cleans to nothing, so it encodes to a single <unk>."""
    label, prob = predict(trained, "!!! ???")
    assert label in (NEGATIVE, POSITIVE)
    assert 0.0 <= prob <= 1.0


def test_prediction_is_deterministic(trained) -> None:
    """Dropout must be off at inference."""
    first = predict(trained, "the flight was delayed and the staff were rude")
    for _ in range(3):
        assert predict(trained, "the flight was delayed and the staff were rude") == first
