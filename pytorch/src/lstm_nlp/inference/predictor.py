"""Loadable predictors: a checkpoint in, usable predictions out.

Each class is constructed from a checkpoint path and needs nothing else -- the
vocabulary and preprocessing contract travel inside the file (D8). This is the
layer both the CLI and the FastAPI backend call, so there is exactly one
inference path in the system.

Both predictors report how many input tokens were unknown. That is not a debug
field: a prediction resting on mostly-``<unk>`` input is uninformative, and the
caller has to be able to see that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from lstm_nlp.data.preprocess import clean_tweet, tokenize
from lstm_nlp.data.sentiment import LABEL_NAMES
from lstm_nlp.errors import DataError
from lstm_nlp.inference.checkpoint import build_model, load_checkpoint
from lstm_nlp.inference.sampler import (
    greedy_from_logits,
    sample_from_logits,
    temperature_distribution,
    top_tokens,
)
from lstm_nlp.utils.seed import make_generator
from lstm_nlp.vocab import Vocab


@dataclass
class SentimentPrediction:
    """One classification, with the evidence needed to judge it."""

    label: str
    label_id: int
    probabilities: dict[str, float]
    n_tokens: int
    n_unk: int

    @property
    def unk_rate(self) -> float:
        """Fraction of input tokens the model had never seen."""
        return self.n_unk / self.n_tokens if self.n_tokens else 0.0

    def to_dict(self) -> dict:
        """JSON-safe payload for the API."""
        return {
            "label": self.label,
            "label_id": self.label_id,
            "probabilities": self.probabilities,
            "n_tokens": self.n_tokens,
            "n_unk": self.n_unk,
            "unk_rate": round(self.unk_rate, 4),
        }


@dataclass
class Generation:
    """One generated passage and the settings that produced it."""

    text: str
    seed_tokens: list[str]
    generated_tokens: list[str]
    temperature: float
    top_k: int | None
    n_unk_in_seed: int

    def to_dict(self) -> dict:
        """JSON-safe payload for the API."""
        return {
            "text": self.text,
            "seed_tokens": self.seed_tokens,
            "generated_tokens": self.generated_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "n_unk_in_seed": self.n_unk_in_seed,
        }


class _Predictor:
    """Shared checkpoint loading."""

    expected_task: str = ""

    def __init__(self, checkpoint_path: str | Path) -> None:
        payload = load_checkpoint(checkpoint_path)
        if self.expected_task and payload["task"] != self.expected_task:
            raise DataError(
                f"{Path(checkpoint_path).name} holds a {payload['task']!r} model, "
                f"but a {self.expected_task!r} model was requested"
            )
        self.payload = payload
        self.vocab: Vocab = payload["vocab"]
        self.model = build_model(payload)
        self.metrics: dict = payload.get("metrics", {})

    @property
    def vocab_size(self) -> int:
        """Number of tokens the model can represent."""
        return len(self.vocab)


class SentimentPredictor(_Predictor):
    """Classify text with a trained sentiment checkpoint."""

    expected_task = "sentiment"

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__(checkpoint_path)
        self.max_len: int = self.payload["preprocess"]["max_len"]

    def predict(self, text: str) -> SentimentPrediction:
        """Classify one string.

        Unknown words become ``<unk>`` and are counted rather than raising
        (D7). Text that cleans to nothing becomes a single ``<unk>``, since a
        zero-length sequence is not a valid LSTM input.

        Args:
            text: Raw input, cleaned with the same function used in training.

        Returns:
            The prediction, its class probabilities, and the unknown-token count.
        """
        tokens = tokenize(clean_tweet(text))[: self.max_len]
        n_unk = self.vocab.count_unknown(tokens)
        ids = self.vocab.encode(tokens) or [self.vocab.unk_index]

        with torch.no_grad():
            logits = self.model(torch.tensor([ids]), torch.tensor([len(ids)]))
            probabilities = torch.softmax(logits, dim=1)[0]

        label_id = int(probabilities.argmax())
        return SentimentPrediction(
            label=LABEL_NAMES[label_id],
            label_id=label_id,
            probabilities={
                name: round(float(probabilities[i]), 6) for i, name in enumerate(LABEL_NAMES)
            },
            n_tokens=len(tokens),
            n_unk=n_unk,
        )

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        """Classify several strings."""
        return [self.predict(t) for t in texts]


class TextGenerator(_Predictor):
    """Generate text with a trained next-word checkpoint."""

    expected_task = "textgen"

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__(checkpoint_path)
        self.seq_len: int = self.payload["preprocess"]["seq_len"]

    def _prepare_seed(self, seed_text: str) -> tuple[list[str], list[int], int]:
        """Tokenise a seed and pad or truncate it to exactly ``seq_len``.

        A short seed is left-padded with ``<unk>`` and a long one truncated to
        its final ``seq_len`` words -- both documented rather than treated as
        errors, since the window length is a model constraint and not something
        a caller should have to know.
        """
        tokens = tokenize(clean_tweet(seed_text))
        if not tokens:
            raise DataError("seed text is empty after cleaning; provide at least one word")

        n_unk = self.vocab.count_unknown(tokens)
        window = tokens[-self.seq_len :]
        if len(window) < self.seq_len:
            window = ["<unk>"] * (self.seq_len - len(window)) + window
        return tokens, self.vocab.encode(window), n_unk

    def next_word_logits(self, window_ids: list[int]) -> torch.Tensor:
        """Raw logits for the word following ``window_ids``."""
        with torch.no_grad():
            return self.model(torch.tensor([window_ids]))[0]

    def generate(
        self,
        seed_text: str,
        n_words: int = 40,
        temperature: float = 0.7,
        top_k: int | None = None,
        rng_seed: int | None = None,
    ) -> Generation:
        """Generate ``n_words`` words continuing ``seed_text``.

        Args:
            seed_text: Starting text. Padded or truncated to the window length.
            n_words: How many words to generate.
            temperature: Applied to **logits**. Lower is greedier.
            top_k: Optional restriction to the k most likely tokens.
            rng_seed: Pass an integer for reproducible output (PRD FR-23).

        Returns:
            The passage, its tokens, and the settings used.

        Raises:
            DataError: If ``n_words`` is negative or the seed cleans to nothing.
        """
        if n_words < 0:
            raise DataError(f"n_words must be >= 0, got {n_words}")

        seed_tokens, window, n_unk = self._prepare_seed(seed_text)
        generator = make_generator(rng_seed)

        generated: list[str] = []
        for _ in range(n_words):
            logits = self.next_word_logits(window)
            index = sample_from_logits(logits, temperature, top_k, generator)
            generated.append(self.vocab.token(index))
            window = window[1:] + [index]

        return Generation(
            text=" ".join(seed_tokens + generated),
            seed_tokens=seed_tokens,
            generated_tokens=generated,
            temperature=temperature,
            top_k=top_k,
            n_unk_in_seed=n_unk,
        )

    def greedy(self, seed_text: str, n_words: int = 40) -> Generation:
        """Generate deterministically, always taking the most likely word."""
        seed_tokens, window, n_unk = self._prepare_seed(seed_text)
        generated: list[str] = []
        for _ in range(n_words):
            index = greedy_from_logits(self.next_word_logits(window))
            generated.append(self.vocab.token(index))
            window = window[1:] + [index]
        return Generation(
            text=" ".join(seed_tokens + generated),
            seed_tokens=seed_tokens,
            generated_tokens=generated,
            temperature=0.0,
            top_k=None,
            n_unk_in_seed=n_unk,
        )

    def next_word_distribution(
        self, seed_text: str, temperature: float = 0.7, top_k: int | None = None, n: int = 12
    ) -> list[tuple[str, float]]:
        """The ``n`` most likely next words and their probabilities at ``T``.

        Backs the frontend's temperature chart (PRD FR-34) and makes the
        entropy claim inspectable rather than asserted.
        """
        _, window, _ = self._prepare_seed(seed_text)
        logits = self.next_word_logits(window)
        return [(self.vocab.token(i), p) for i, p in top_tokens(logits, temperature, top_k, n)]

    def distribution_at(
        self, seed_text: str, temperature: float, top_k: int | None = None
    ) -> torch.Tensor:
        """Full sampling distribution for the next word, for analysis."""
        _, window, _ = self._prepare_seed(seed_text)
        return temperature_distribution(self.next_word_logits(window), temperature, top_k)
