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
from lstm_nlp.utils.logging import get_logger
from lstm_nlp.vocab import PADDED_SPECIALS, Vocab

logger = get_logger(__name__)

LABEL_NAMES: tuple[str, str] = ("negative", "positive")
REQUIRED_COLUMNS = ("airline_sentiment", "text")


@dataclass(frozen=True)
class SentimentSplits:
    """Everything the trainer needs, with the vocabulary built from train only.

    Three blocks, not two. ``val`` is what early stopping and best-weight
    selection are allowed to see; ``test`` is scored exactly once, at the end.
    Until Phase 8 the trainer passed ``test`` as its validation loader, so the
    reported macro-F1 was the maximum over every epoch's score on the very rows
    it was reported for -- measured at **+0.0094** optimism against a proper
    held-out estimate. A metric chosen on the set it is quoted for is not a
    held-out metric, whatever the variable is called.
    """

    train: SentimentDataset
    val: SentimentDataset
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


def load_sentiment_frame(csv_path: str | Path, *, deduplicate: bool = True) -> pd.DataFrame:
    """Read and clean the sentiment CSV, optionally removing duplicate rows.

    Deduplication happens **before** anything else touches the frame, which is
    what makes it work: a duplicate that never reaches the splitter cannot
    straddle the train/test boundary. Measured on this corpus, 86 of 3,463 test
    rows (2.48%) previously shared their cleaned text with a training row, so
    the model was partly rewarded for memorising ``"<user> thanks"``.

    Duplicates are identified on the **cleaned** text, not the raw text, because
    cleaning is what makes two differently-typed tweets the same input. Two rows
    the model cannot tell apart are one row as far as evaluation is concerned.

    Rows whose cleaned text carries **contradictory labels** are dropped
    entirely rather than resolved. There are five such texts here; keeping
    either label would be inventing an annotation, and keeping both would put
    the same input in two classes.

    Args:
        csv_path: Path to ``airline_sentiment.csv``.
        deduplicate: Drop repeated and contradictory cleaned texts. Off only for
            tests that need the raw row count.

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
    if deduplicate:
        frame = _drop_duplicate_texts(frame)
    return frame


def _drop_duplicate_texts(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per cleaned text, and none where the labels disagree.

    Order is preserved and the *first* occurrence is kept, so the result is a
    deterministic function of the input file rather than of dict iteration.
    """
    distinct_labels = frame.groupby("clean")["airline_sentiment"].nunique()
    contradictory = set(distinct_labels[distinct_labels > 1].index)

    conflicted_rows = int(frame["clean"].isin(contradictory).sum())
    kept = frame[~frame["clean"].isin(contradictory)].drop_duplicates(
        subset="clean", keep="first"
    )

    removed = len(frame) - len(kept)
    if removed:
        logger.info(
            "deduplicated: %d of %d rows removed -- %d repeated, %d contradictory "
            "across %d texts",
            removed, len(frame), removed - conflicted_rows, conflicted_rows,
            len(contradictory),
        )
    return kept.reset_index(drop=True)


def compute_class_weights(labels: list[int], num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights, normalised so the majority class weighs 1.0.

    On the deduplicated corpus that yields ``[1.0, 4.130]`` (``[1.0, 3.884]``
    before deduplication).  Without it the model can score 0.8048 accuracy by
    never predicting the minority class (D4).
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
    val_size: float = 0.15,
    split_seed: int = 10,
    stratify: bool = True,
    deduplicate: bool = True,
    min_freq: int = 2,
    max_len: int = 30,
) -> SentimentSplits:
    """Load, split three ways, build the vocabulary, and wrap each block.

    The order is deliberate and is the fix for D7: **split first, then build the
    vocabulary from the training rows only.**

    The test block is carved off first, with ``test_size`` and ``split_seed``
    unchanged, so it holds exactly the rows it always did and results stay
    comparable across this change. Validation is then taken out of what
    remains, never out of test -- which is the point: the model may be selected
    on ``val`` and is scored once on ``test``.

    Args:
        csv_path: Path to the sentiment CSV.
        test_size: Test fraction of the whole dataset.
        val_size: Validation fraction **of the training block**, not of the
            whole dataset, so changing it never moves the test rows.
        split_seed: Split RNG seed; pinned at 10 to match the frozen reference.
        stratify: Preserve the class ratio across splits.
        deduplicate: Remove repeated and contradictory cleaned texts before
            splitting, so no duplicate can straddle the train/test boundary.
        min_freq: Minimum training frequency for a token to earn an index.
        max_len: Truncation length (p99 of this data is 30 tokens).

    Returns:
        Train/val/test datasets, the vocabulary, and class weights.

    Raises:
        DataError: If ``val_size`` is not a fraction strictly between 0 and 1.
    """
    if not 0.0 < val_size < 1.0:
        raise DataError(
            f"val_size must be a fraction in (0, 1), got {val_size}. Model "
            f"selection needs a block that is not the test block."
        )
    frame = load_sentiment_frame(csv_path, deduplicate=deduplicate)
    texts = frame["clean"].tolist()
    labels = frame["airline_sentiment"].astype(int).tolist()

    train_idx, test_idx = train_test_split(
        range(len(frame)),
        test_size=test_size,
        random_state=split_seed,
        stratify=labels if stratify else None,
    )

    pool_texts = [texts[i] for i in train_idx]
    pool_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    # Validation comes out of the training pool. Selection may see this; it may
    # never see `test`.
    inner_idx, val_idx = train_test_split(
        range(len(pool_texts)),
        test_size=val_size,
        random_state=split_seed,
        stratify=pool_labels if stratify else None,
    )
    train_texts = [pool_texts[i] for i in inner_idx]
    train_labels = [pool_labels[i] for i in inner_idx]
    val_texts = [pool_texts[i] for i in val_idx]
    val_labels = [pool_labels[i] for i in val_idx]

    # ---- the D7 fix: counts come from training rows and nowhere else ----
    vocab = Vocab.build(
        count_tokens(train_texts), min_freq=min_freq, specials=PADDED_SPECIALS
    )

    return SentimentSplits(
        train=SentimentDataset(train_texts, train_labels, vocab, max_len),
        val=SentimentDataset(val_texts, val_labels, vocab, max_len),
        test=SentimentDataset(test_texts, test_labels, vocab, max_len),
        vocab=vocab,
        class_weights=compute_class_weights(train_labels),
    )
