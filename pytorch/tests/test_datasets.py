"""Phase 1: datasets. Regression tests for D7 (leak) and D9 (memory).

The ``realdata`` tests assert the measured values in ``Phases.md`` so the
pipeline cannot drift away from the audited numbers unnoticed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from lstm_nlp.data.preprocess import load_corpus, tokenize
from lstm_nlp.data.sentiment import (
    LABEL_NAMES,
    SentimentDataset,
    collate_sentiment,
    compute_class_weights,
    load_sentiment_frame,
    prepare_sentiment_data,
)
from lstm_nlp.data.textgen import WindowDataset, prepare_textgen_data, split_tokens
from lstm_nlp.errors import DataError
from lstm_nlp.vocab import Vocab, UNPADDED_SPECIALS

# --------------------------------------------------------------------------- #
# sentiment -- fast
# --------------------------------------------------------------------------- #


def test_load_frame_adds_clean_column(sample_csv: Path) -> None:
    frame = load_sentiment_frame(sample_csv)
    assert "clean" in frame.columns
    assert frame["clean"].str.isupper().sum() == 0


def test_missing_csv_raises() -> None:
    with pytest.raises(FileNotFoundError, match="sentiment csv not found"):
        load_sentiment_frame("no/such.csv")


def test_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DataError, match="missing column"):
        load_sentiment_frame(path)


def test_bad_labels_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("airline_sentiment,text\n2,hello\n", encoding="utf-8")
    with pytest.raises(DataError, match="expected labels"):
        load_sentiment_frame(path)


def test_splits_are_disjoint_and_complete(sample_csv: Path) -> None:
    splits = prepare_sentiment_data(sample_csv, min_freq=1)
    assert len(splits.train) + len(splits.test) == 200
    assert set(splits.train.texts).isdisjoint(set(splits.test.texts) - set(splits.train.texts))


def test_stratification_preserves_class_ratio(sample_csv: Path) -> None:
    splits = prepare_sentiment_data(sample_csv, min_freq=1)
    train_pos = splits.train.label_counts["positive"] / len(splits.train)
    test_pos = splits.test.label_counts["positive"] / len(splits.test)
    assert abs(train_pos - test_pos) < 0.02


def test_vocab_built_from_train_only(sample_csv: Path) -> None:
    """The D7 fix: a token appearing only in test must not have an index."""
    splits = prepare_sentiment_data(sample_csv, min_freq=1)
    train_tokens = {t for text in splits.train.texts for t in tokenize(text)}
    test_tokens = {t for text in splits.test.texts for t in tokenize(text)}
    test_only = test_tokens - train_tokens
    assert test_only, "fixture too small to exercise the leak"
    for token in test_only:
        assert token not in splits.vocab
        assert splits.vocab.index(token) == splits.vocab.unk_index


def test_empty_text_becomes_unk_not_zero_length() -> None:
    """pack_padded_sequence rejects length-0 sequences."""
    vocab = Vocab.build(Counter({"a": 1}))
    dataset = SentimentDataset(["!!!", "a"], [0, 1], vocab, max_len=10)
    ids, length, _ = dataset[0]
    assert length == 1
    assert ids.tolist() == [vocab.unk_index]


def test_truncation_respects_max_len() -> None:
    vocab = Vocab.build(Counter({"w": 100}))
    dataset = SentimentDataset([" ".join(["w"] * 50)], [0], vocab, max_len=7)
    ids, length, _ = dataset[0]
    assert length == 7 and ids.shape == (7,)


def test_length_mismatch_rejected() -> None:
    vocab = Vocab.build(Counter({"a": 1}))
    with pytest.raises(DataError, match="length mismatch"):
        SentimentDataset(["a", "b"], [0], vocab, max_len=5)


def test_collate_pads_to_batch_max_not_global(sample_csv: Path) -> None:
    """Per-batch padding: the reference padded everything to 26 (58.1% zeros)."""
    splits = prepare_sentiment_data(sample_csv, min_freq=1)
    loader = DataLoader(splits.train, batch_size=8, collate_fn=collate_sentiment)
    ids, lengths, labels = next(iter(loader))
    assert ids.dtype == torch.long and lengths.dtype == torch.long
    assert ids.shape == (8, int(lengths.max()))
    assert labels.shape == (8,)
    assert ids.shape[1] <= 30


def test_collate_pads_with_zero_beyond_each_length(sample_csv: Path) -> None:
    splits = prepare_sentiment_data(sample_csv, min_freq=1)
    loader = DataLoader(splits.train, batch_size=16, collate_fn=collate_sentiment)
    ids, lengths, _ = next(iter(loader))
    for row, length in zip(ids, lengths, strict=True):
        assert (row[length:] == 0).all(), "padding must be the <pad> index 0"


def test_class_weights_normalised_to_majority_one() -> None:
    weights = compute_class_weights([0] * 80 + [1] * 20)
    assert weights[0].item() == pytest.approx(1.0)
    assert weights[1].item() == pytest.approx(4.0)


def test_class_weights_reject_absent_class() -> None:
    with pytest.raises(DataError, match="class absent"):
        compute_class_weights([0, 0, 0])


def test_label_names_order() -> None:
    assert LABEL_NAMES == ("negative", "positive")


# --------------------------------------------------------------------------- #
# textgen -- fast
# --------------------------------------------------------------------------- #


def test_split_tokens_is_contiguous_and_complete() -> None:
    tokens = [str(i) for i in range(100)]
    train, val = split_tokens(tokens, 0.10)
    assert len(train) == 90 and len(val) == 10
    assert train + val == tokens


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_bad_val_fraction_rejected(fraction: float) -> None:
    with pytest.raises(DataError, match="val_fraction"):
        split_tokens([str(i) for i in range(100)], fraction)


def test_window_dataset_shapes_and_dtype() -> None:
    ds = WindowDataset(list(range(100)), seq_len=10)
    x, y = ds[0]
    assert x.shape == (10,) and x.dtype == torch.int64
    assert y.shape == () and y.dtype == torch.int64
    assert x.tolist() == list(range(10)) and int(y) == 10
    assert len(ds) == 90


def test_windows_are_int_indices_not_onehot() -> None:
    """The D9 fix asserted at the item level."""
    ds = WindowDataset(list(range(100)), seq_len=10)
    x, _ = ds[0]
    assert x.ndim == 1, "a one-hot window would be 2-D"
    assert x.dtype == torch.int64


def test_storage_is_o_tokens_not_o_windows() -> None:
    """The stronger D9 assertion: nothing is precomputed per window."""
    ds = WindowDataset(list(range(10_000)), seq_len=10)
    assert ds.storage_nbytes() == 10_000 * 8
    assert ds.storage_nbytes() < ds.dense_nbytes() / 9


def test_stride_reduces_window_count() -> None:
    assert len(WindowDataset(list(range(100)), seq_len=10, stride=5)) == 18


@pytest.mark.parametrize(("seq_len", "stride"), [(0, 1), (1, 0)])
def test_bad_window_params_rejected(seq_len: int, stride: int) -> None:
    with pytest.raises(DataError):
        WindowDataset(list(range(100)), seq_len=seq_len, stride=stride)


def test_too_few_tokens_rejected() -> None:
    with pytest.raises(DataError, match="need more than seq_len"):
        WindowDataset(list(range(5)), seq_len=10)


def test_no_window_straddles_the_split(mini_book: Path) -> None:
    """Windowing happens per block, after the split -- so context never leaks."""
    splits = prepare_textgen_data(mini_book, seq_len=10, min_freq=1)
    assert len(splits.train) == splits.train.n_tokens - 10
    assert len(splits.val) == splits.val.n_tokens - 10
    tokens = tokenize(load_corpus(mini_book))
    assert len(splits.train) + len(splits.val) == len(tokens) - 20


def test_textgen_vocab_built_from_train_only(mini_book: Path) -> None:
    splits = prepare_textgen_data(mini_book, seq_len=10, min_freq=1)
    tokens = tokenize(load_corpus(mini_book))
    train_tokens, val_tokens = split_tokens(tokens, 0.10)
    for token in set(val_tokens) - set(train_tokens):
        assert token not in splits.vocab


def test_textgen_vocab_has_no_pad(mini_book: Path) -> None:
    """Fixed windows need no padding, so <pad> would be a dead row."""
    splits = prepare_textgen_data(mini_book, seq_len=10, min_freq=1)
    assert splits.vocab.pad_index is None
    assert splits.vocab.specials == UNPADDED_SPECIALS


# --------------------------------------------------------------------------- #
# measured values (Phases.md Phase 1)
# --------------------------------------------------------------------------- #


@pytest.mark.realdata
def test_sentiment_measured_values(sentiment_splits) -> None:
    s = sentiment_splits
    assert (len(s.train), len(s.test)) == (8078, 3463)
    assert s.train.label_counts == {"negative": 6424, "positive": 1654}
    assert s.train.label_counts["positive"] / len(s.train) == pytest.approx(0.2048, abs=1e-4)
    assert s.test.label_counts["positive"] / len(s.test) == pytest.approx(0.2047, abs=1e-4)
    assert len(s.vocab) == 4505
    assert s.class_weights.tolist() == pytest.approx([1.0, 3.884], abs=1e-3)
    assert s.test.unknown_rate() == pytest.approx(0.0523, abs=1e-4)
    # The training rate must be NONZERO: min_freq=2 drops training hapax, and
    # that is exactly what gives the <unk> embedding row gradient signal. At 0,
    # <unk> would be a randomly-initialised row first used at inference time.
    assert s.train.unknown_rate() == pytest.approx(0.0338, abs=1e-4)
    assert 0.0 < s.train.unknown_rate() < s.test.unknown_rate()


@pytest.mark.realdata
def test_textgen_measured_values(textgen_splits) -> None:
    t = textgen_splits
    assert t.n_tokens == 27_429
    assert (t.train.n_tokens, t.val.n_tokens) == (24_687, 2_742)
    assert len(t.vocab) == 2_436  # train-only @ min_freq=1
    assert len(t.train) + len(t.val) == 27_409  # 27,429 - 2 x seq_len
    # min_freq=1 => every training token earned an index
    assert int((t.train.ids == t.vocab.unk_index).sum()) == 0


@pytest.mark.realdata
def test_textgen_memory_footprint(textgen_splits) -> None:
    """D9: 707 MB of one-hot becomes 0.22 MB of lazily-sliced indices."""
    t = textgen_splits
    storage = t.train.storage_nbytes() + t.val.storage_nbytes()
    dense = t.train.dense_nbytes() + t.val.dense_nbytes()
    onehot = t.train.onehot_nbytes(len(t.vocab)) + t.val.onehot_nbytes(len(t.vocab))
    assert storage / 1e6 == pytest.approx(0.22, abs=0.01)
    assert dense / 1e6 == pytest.approx(2.19, abs=0.01)
    assert onehot / storage > 2_000


@pytest.mark.realdata
def test_no_gutenberg_vocabulary_in_textgen(textgen_splits) -> None:
    """D6 at the vocabulary level -- the licence words have no index at all."""
    for word in ("copyright", "donations", "foundation", "ebook", "license", "gutenberg"):
        assert word not in textgen_splits.vocab
