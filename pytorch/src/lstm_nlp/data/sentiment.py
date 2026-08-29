"""Sentiment dataset: airline tweets -> binary label.

Many-to-one: a variable-length token sequence produces one label.

Split-then-build is the rule here.  The vocabulary is constructed inside
:func:`prepare_sentiment_data` from the *training indices only*; there is no
other code path that builds one, so the D7 leak cannot be reintroduced by
accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from lstm_nlp.data.preprocess import clean_tweet, count_tokens, tokenize
from lstm_nlp.errors import DataError
from lstm_nlp.vocab import PADDED_SPECIALS, Vocab

LABEL_NAMES: tuple[str, str] = ("negative", "positive")
REQUIRED_COLUMNS = ("airline_sentiment", "text")


@dataclass(frozen=True)
class SentimentSplits:
    """Everything the trainer needs, with the vocabulary built from train only."""

    train: SentimentDataset
    test: SentimentDataset
    vocab: Vocab
    class_weights: torch.Tensor

    @property
    def num_classes(self) -> int:
        """Number of sentiment classes (2: negative, positive)."""
        return len(LABEL_NAMES)


class SentimentDataset(Dataset):
    """Encoded tweets and their labels.

    Sequences are stored as int64 index lists and padded per batch by
    :func:`collate_sentiment` -- never one-hot, and never padded globally.  The
    reference padded every tweet to 26 tokens, making 58.1% of its input matrix
    zeros.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocab,
        max_len: int,
    ) -> None:
        if len(texts) != len(labels):
            raise DataError(f"texts/labels length mismatch: {len(texts)} vs {len(labels)}")
        self.vocab = vocab
        self.max_len = max_len
        self.texts = texts
        self.labels = labels
        self._encoded = [self._encode(t) for t in texts]

    def _encode(self, text: str) -> list[int]:
        ids = self.vocab.encode(tokenize(text))[: self.max_len]
        # pack_padded_sequence rejects zero-length sequences, and a tweet can
        # clean to nothing.  A lone <unk> is the honest representation: we know
        # there was a tweet, and we know nothing about its contents.
        return ids or [self.vocab.unk_index]

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        ids = self._encoded[idx]
        return torch.tensor(ids, dtype=torch.long), len(ids), self.labels[idx]

    @property
    def label_counts(self) -> dict[str, int]:
        """Rows per class, keyed by label name."""
        return {name: self.labels.count(i) for i, name in enumerate(LABEL_NAMES)}

    def unknown_rate(self) -> float:
        """Fraction of tokens that fall outside the vocabulary."""
        total = unknown = 0
        for text in self.texts:
            toks = tokenize(text)
            total += len(toks)
            unknown += self.vocab.count_unknown(toks)
        return unknown / total if total else 0.0


def collate_sentiment(
    batch: list[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch to its own longest sequence.

    Returns:
        ``(ids, lengths, labels)`` where ``ids`` is ``(B, L_max)`` int64 padded
        with 0, ``lengths`` is ``(B,)`` int64 for ``pack_padded_sequence``, and
        ``labels`` is ``(B,)`` int64.
    """
    sequences, lengths, labels = zip(*batch, strict=True)
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def load_sentiment_frame(csv_path: str | Path) -> pd.DataFrame:
    """Read and clean the sentiment CSV.

    Args:
        csv_path: Path to ``airline_sentiment.csv``.

    Returns:
        The frame with an added ``clean`` column.

    Raises:
        FileNotFoundError: If the file is missing.
        DataError: If required columns are absent or labels are not 0/1.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"sentiment csv not found: {path.resolve()}")

    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataError(f"{path.name}: missing column(s) {missing}; found {list(frame.columns)}")

    labels = set(frame["airline_sentiment"].unique().tolist())
    if not labels <= {0, 1}:
        raise DataError(f"{path.name}: expected labels in {{0, 1}}, found {sorted(labels)}")

    frame = frame.copy()
    frame["clean"] = frame["text"].astype(str).map(clean_tweet)
    return frame


def compute_class_weights(labels: list[int], num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights, normalised so the majority class weighs 1.0.

    On this data that yields ``[1.0, 3.884]``.  Without it the model can score
    0.795 accuracy by never predicting the minority class (D4).
    """
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    if (counts == 0).any():
        raise DataError(f"class absent from labels: counts={counts.tolist()}")
    weights = counts.max() / counts
    return weights / weights.min()


def prepare_sentiment_data(
    csv_path: str | Path,
    *,
    test_size: float = 0.30,
    split_seed: int = 10,
    stratify: bool = True,
    min_freq: int = 2,
    max_len: int = 30,
) -> SentimentSplits:
    """Load, split, build the vocabulary, and wrap both splits as datasets.

    The order is deliberate and is the fix for D7: **split first, then build the
    vocabulary from the training rows only.**

    Args:
        csv_path: Path to the sentiment CSV.
        test_size: Test fraction.
        split_seed: Split RNG seed; pinned at 10 to match the frozen reference.
        stratify: Preserve the class ratio across splits.
        min_freq: Minimum training frequency for a token to earn an index.
        max_len: Truncation length (p99 of this data is 30 tokens).

    Returns:
        Train/test datasets, the vocabulary, and class weights.
    """
    frame = load_sentiment_frame(csv_path)
    texts = frame["clean"].tolist()
    labels = frame["airline_sentiment"].astype(int).tolist()

    train_idx, test_idx = train_test_split(
        range(len(frame)),
        test_size=test_size,
        random_state=split_seed,
        stratify=labels if stratify else None,
    )

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    # ---- the D7 fix: counts come from training rows and nowhere else ----
    vocab = Vocab.build(
        count_tokens(train_texts), min_freq=min_freq, specials=PADDED_SPECIALS
    )

    return SentimentSplits(
        train=SentimentDataset(train_texts, train_labels, vocab, max_len),
        test=SentimentDataset(test_texts, test_labels, vocab, max_len),
        vocab=vocab,
        class_weights=compute_class_weights(train_labels),
    )
