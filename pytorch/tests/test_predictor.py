"""Phase 4: predictors against the trained checkpoints.

Skipped unless both checkpoints exist. Produce them with:

    lstm-nlp train --config configs/sentiment.yaml
    lstm-nlp train --config configs/textgen.yaml
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from lstm_nlp.errors import DataError
from lstm_nlp.inference.predictor import SentimentPredictor, TextGenerator
from lstm_nlp.inference.sampler import distribution_entropy

RUNS = Path(__file__).resolve().parents[1] / "runs"

GUTENBERG_WORDS = ("copyright", "donations", "foundation", "ebook",
                   "license", "gutenberg", "trademark", "royalty")


def _latest(task: str) -> Path | None:
    found = sorted((RUNS / task).glob("*/best.pt")) if (RUNS / task).is_dir() else []
    return found[-1] if found else None


@pytest.fixture(scope="module")
def sentiment() -> SentimentPredictor:
    path = _latest("sentiment")
    if path is None:
        pytest.skip("no sentiment checkpoint; run: lstm-nlp train --config configs/sentiment.yaml")
    return SentimentPredictor(path)


@pytest.fixture(scope="module")
def generator() -> TextGenerator:
    path = _latest("textgen")
    if path is None:
        pytest.skip("no textgen checkpoint; run: lstm-nlp train --config configs/textgen.yaml")
    return TextGenerator(path)


# --------------------------------------------------------------------------- #
# SentimentPredictor
# --------------------------------------------------------------------------- #


def test_predict_returns_a_complete_result(sentiment: SentimentPredictor) -> None:
    result = sentiment.predict("the flight was delayed and the crew were rude")
    assert result.label in ("negative", "positive")
    assert set(result.probabilities) == {"negative", "positive"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert result.n_tokens > 0


def test_unknown_rate_is_surfaced(sentiment: SentimentPredictor) -> None:
    """A prediction resting on mostly-unknown input must say so."""
    known = sentiment.predict("the flight was late")
    unknown = sentiment.predict("qwertyuiop zxcvbnm flurbulate")
    assert known.unk_rate < unknown.unk_rate
    assert unknown.unk_rate > 0.5
    assert "unk_rate" in unknown.to_dict()


def test_unknown_words_do_not_raise(sentiment: SentimentPredictor) -> None:
    """D7 at the inference boundary."""
    assert sentiment.predict("zzzz yyyy xxxx").label in ("negative", "positive")


def test_empty_text_does_not_raise(sentiment: SentimentPredictor) -> None:
    result = sentiment.predict("!!! ???")
    assert result.n_tokens == 0
    assert result.label in ("negative", "positive")


def test_batch_matches_individual_calls(sentiment: SentimentPredictor) -> None:
    texts = ["great service", "terrible delay", "the flight was fine"]
    batch = sentiment.predict_batch(texts)
    assert [r.label for r in batch] == [sentiment.predict(t).label for t in texts]


def test_wrong_task_checkpoint_rejected() -> None:
    path = _latest("textgen")
    if path is None:
        pytest.skip("no textgen checkpoint")
    with pytest.raises(DataError, match="requested"):
        SentimentPredictor(path)


def test_metrics_travel_with_the_predictor(sentiment: SentimentPredictor) -> None:
    """C11: the caller can always reach the baseline beside the metric."""
    assert sentiment.metrics["macro_f1"] > sentiment.metrics["baseline_macro_f1"]


# --------------------------------------------------------------------------- #
# TextGenerator
# --------------------------------------------------------------------------- #


def test_generate_produces_the_requested_length(generator: TextGenerator) -> None:
    result = generator.generate("alice was beginning to", n_words=15, rng_seed=0)
    assert len(result.generated_tokens) == 15
    assert result.text.startswith("alice was beginning to")


def test_generate_is_reproducible_with_a_seed(generator: TextGenerator) -> None:
    """PRD FR-23, end to end rather than at the sampler."""
    a = generator.generate("alice was", n_words=20, temperature=1.0, rng_seed=7)
    b = generator.generate("alice was", n_words=20, temperature=1.0, rng_seed=7)
    c = generator.generate("alice was", n_words=20, temperature=1.0, rng_seed=8)
    assert a.text == b.text
    assert a.text != c.text


def test_temperature_changes_the_output(generator: TextGenerator) -> None:
    cold = generator.generate("alice was", n_words=25, temperature=0.2, rng_seed=1)
    hot = generator.generate("alice was", n_words=25, temperature=2.0, rng_seed=1)
    assert cold.text != hot.text


def test_entropy_rises_with_temperature_on_the_real_model(generator: TextGenerator) -> None:
    """S5 against the trained model, not a synthetic distribution."""
    entropies = [
        distribution_entropy(generator.distribution_at("alice was beginning to", t))
        for t in (0.2, 0.5, 1.0, 1.5, 2.0)
    ]
    assert entropies == sorted(entropies)
    assert entropies[-1] > entropies[0]
    assert entropies[-1] < math.log(generator.vocab_size)


def test_greedy_is_deterministic(generator: TextGenerator) -> None:
    first = generator.greedy("alice was beginning to", n_words=12)
    assert first.text == generator.greedy("alice was beginning to", n_words=12).text


def test_short_seed_is_padded_not_rejected(generator: TextGenerator) -> None:
    """The window length is a model constraint, not the caller's problem."""
    result = generator.generate("alice", n_words=5, rng_seed=0)
    assert len(result.generated_tokens) == 5
    assert result.seed_tokens == ["alice"]


def test_long_seed_is_truncated_to_the_window(generator: TextGenerator) -> None:
    long_seed = " ".join(["the"] * 40)
    result = generator.generate(long_seed, n_words=3, rng_seed=0)
    assert len(result.seed_tokens) == 40
    assert len(result.generated_tokens) == 3


def test_unknown_seed_word_does_not_raise(generator: TextGenerator) -> None:
    """D7: a seed may contain words the model has never seen."""
    result = generator.generate("qwertyuiop alice was", n_words=5, rng_seed=0)
    assert result.n_unk_in_seed >= 1
    assert len(result.generated_tokens) == 5


def test_empty_seed_rejected(generator: TextGenerator) -> None:
    with pytest.raises(DataError, match="empty after cleaning"):
        generator.generate("!!! ???", n_words=5)


def test_negative_word_count_rejected(generator: TextGenerator) -> None:
    with pytest.raises(DataError, match="n_words must be >= 0"):
        generator.generate("alice was", n_words=-1)


def test_top_k_reduces_diversity(generator: TextGenerator) -> None:
    """top_k restricts the choice at EACH step, not across the passage.

    Over 200 steps the context changes 200 times, so the union of chosen words
    is far larger than k -- the meaningful property is comparative: at the same
    temperature and seed, restricting to the top 3 must produce fewer distinct
    words than sampling the full vocabulary.
    """
    restricted = generator.generate(
        "alice was", n_words=200, temperature=2.0, top_k=3, rng_seed=0
    )
    unrestricted = generator.generate(
        "alice was", n_words=200, temperature=2.0, top_k=None, rng_seed=0
    )
    assert len(set(restricted.generated_tokens)) < len(set(unrestricted.generated_tokens))


def test_next_word_distribution_is_sorted(generator: TextGenerator) -> None:
    pairs = generator.next_word_distribution("alice was beginning to", temperature=0.7, n=8)
    probabilities = [p for _, p in pairs]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(isinstance(token, str) for token, _ in pairs)


# --------------------------------------------------------------------------- #
# S15 -- no Gutenberg boilerplate can be generated
# --------------------------------------------------------------------------- #


def test_generated_text_never_contains_boilerplate(generator: TextGenerator) -> None:
    """S15, and a structural rather than statistical guarantee.

    The reference validated on the licence text and seeded generation from it.
    Here those words have no index at all, so no temperature can produce them.
    Five passages at T=0.7 confirm it end to end.
    """
    for seed in range(5):
        text = generator.generate(
            "alice was beginning to", n_words=40, temperature=0.7, rng_seed=seed
        ).text
        for word in GUTENBERG_WORDS:
            assert word not in text, f"generated boilerplate token {word!r}"


def test_generated_words_come_from_the_vocabulary(generator: TextGenerator) -> None:
    result = generator.generate("alice was", n_words=50, temperature=1.2, rng_seed=3)
    for token in result.generated_tokens:
        assert token in generator.vocab
