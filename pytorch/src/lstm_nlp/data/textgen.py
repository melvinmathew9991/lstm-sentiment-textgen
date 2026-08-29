"""Text-generation dataset: a 10-word window -> the next word.

Many-to-one: a fixed-length token window produces one word.

Two memory/correctness notes:

* Windows are produced by **slicing a single 1-D index tensor**, so storage is
  O(n_tokens) rather than O(n_windows x seq_len).  The reference materialised a
  one-hot ``(27419, 10, V)`` bool array -- 707 MB resident before batching (D9).
* The train/validation split happens **on tokens, before windowing**, and no
  window straddles the boundary.  The reference used Keras'
  ``validation_split=0.1``, which takes the trailing 10% unshuffled; in this file
  that slice was entirely Project Gutenberg licence text (D6).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lstm_nlp.data.preprocess import load_corpus, tokenize
from lstm_nlp.errors import DataError
from lstm_nlp.vocab import UNPADDED_SPECIALS, Vocab


@dataclass(frozen=True)
class TextGenSplits:
    """Train/validation window datasets sharing one train-built vocabulary."""

    train: WindowDataset
    val: WindowDataset
    vocab: Vocab
    seq_len: int

    @property
    def n_tokens(self) -> int:
        return self.train.n_tokens + self.val.n_tokens


class WindowDataset(Dataset):
    """Sliding windows over a token-index stream.

    Item ``i`` is ``(ids[i : i+seq_len], ids[i+seq_len])`` -- a ``(seq_len,)``
    int64 tensor and a scalar target.  Nothing is precomputed.
    """

    def __init__(self, token_ids: list[int], seq_len: int, stride: int = 1) -> None:
        if seq_len < 1:
            raise DataError(f"seq_len must be >= 1, got {seq_len}")
        if stride < 1:
            raise DataError(f"stride must be >= 1, got {stride}")
        if len(token_ids) <= seq_len:
            raise DataError(
                f"need more than seq_len={seq_len} tokens to form a window, "
                f"got {len(token_ids)}"
            )
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride
        self._starts = range(0, len(token_ids) - seq_len, stride)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._starts[idx]
        return self.ids[start : start + self.seq_len], self.ids[start + self.seq_len]

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    def storage_nbytes(self) -> int:
        """Bytes actually held. O(n_tokens), not O(n_windows x seq_len)."""
        return self.ids.element_size() * self.ids.numel()

    def dense_nbytes(self) -> int:
        """Bytes a materialised ``(N, seq_len)`` int64 tensor would take."""
        return len(self) * self.seq_len * 8

    def onehot_nbytes(self, vocab_size: int) -> int:
        """Bytes the reference's ``(N, seq_len, V)`` bool array would take."""
        return len(self) * self.seq_len * vocab_size


def split_tokens(tokens: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    """Split a token stream into contiguous train/validation blocks.

    The validation block is the trailing ``val_fraction`` of the stream.  That is
    standard practice for language modelling (Penn Treebank, WikiText) and is
    safe *here only because* ``strip_gutenberg`` already removed the licence
    text that used to occupy this slice -- see D6.  Run without stripping and
    this block is boilerplate again.

    Args:
        tokens: Full token stream.
        val_fraction: Fraction held out, in (0, 1).

    Returns:
        ``(train_tokens, val_tokens)``.

    Raises:
        DataError: If ``val_fraction`` is out of range or a block comes out empty.
    """
    if not 0.0 < val_fraction < 1.0:
        raise DataError(f"val_fraction must be in (0, 1), got {val_fraction}")

    n_val = int(len(tokens) * val_fraction)
    if n_val == 0 or n_val >= len(tokens):
        raise DataError(
            f"val_fraction={val_fraction} yields {n_val} validation tokens "
            f"from {len(tokens)}; choose a different fraction"
        )
    cut = len(tokens) - n_val
    return tokens[:cut], tokens[cut:]


def prepare_textgen_data(
    text_path: str | Path,
    *,
    seq_len: int = 10,
    stride: int = 1,
    val_fraction: float = 0.10,
    min_freq: int = 2,
    strip_boilerplate: bool = True,
) -> TextGenSplits:
    """Load a corpus and build train/validation window datasets.

    Order is deliberate: strip -> clean -> tokenise -> **split** -> build vocab
    from train only -> window each block independently.  Windowing last is what
    guarantees no window spans the train/validation boundary.

    Args:
        text_path: Corpus file.
        seq_len: Window length (the "many" in many-to-one).
        stride: Step between window starts.
        val_fraction: Trailing fraction held out.
        min_freq: Minimum training frequency for a token to earn an index.
        strip_boilerplate: Remove Gutenberg header/footer.  Leave this on.

    Returns:
        Train/validation datasets, the vocabulary, and the window length.
    """
    tokens = tokenize(load_corpus(text_path, strip_boilerplate=strip_boilerplate))
    train_tokens, val_tokens = split_tokens(tokens, val_fraction)

    # ---- vocabulary from the training block only (D7) ----
    vocab = Vocab.build(
        Counter(train_tokens), min_freq=min_freq, specials=UNPADDED_SPECIALS
    )

    return TextGenSplits(
        train=WindowDataset(vocab.encode(train_tokens), seq_len, stride),
        val=WindowDataset(vocab.encode(val_tokens), seq_len, stride),
        vocab=vocab,
        seq_len=seq_len,
    )
