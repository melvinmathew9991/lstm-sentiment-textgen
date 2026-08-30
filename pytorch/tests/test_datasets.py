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
from lstm_nlp.vocab import UNPADDED_SPECIALS, Vocab

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
    """Three blocks now, and every row lands in exactly one of them.

    The third block is what lets early stopping select a model without seeing
    the rows the model is then reported on.
    """
    splits = prepare_sentiment_data(sample_csv, min_freq=1)

    # Against the deduplicated row count, not a literal: deduplication decides
    # how many rows there are, and hardcoding 200 would assert the fixture's
    # size rather than the property.
    kept = len(load_sentiment_frame(sample_csv, deduplicate=True))
    assert len(splits.train) + len(splits.val) + len(splits.test) == kept

    # Disjoint, and now genuinely so: deduplication runs before the split, so a
    # shared cleaned text between blocks is impossible rather than merely rare.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(getattr(splits, a).texts) & set(getattr(splits, b).texts), (
            f"{a}/{b} share a cleaned text"
        )


def test_validation_is_carved_out_of_train_never_out_of_test(sample_csv: Path) -> None:
    """The whole point of the change: test size must not move when val does.

    If tuning ``val_size`` could resize the test block, every comparison across
    runs would be against a different yardstick.
    """
    a = prepare_sentiment_data(sample_csv, min_freq=1, val_size=0.10)
    b = prepare_sentiment_data(sample_csv, min_freq=1, val_size=0.30)
    assert len(a.test) == len(b.test)
    assert list(a.test.texts) == list(b.test.texts)
    assert len(a.val) < len(b.val)
    assert len(a.train) > len(b.train)


def test_val_size_must_be_a_fraction(sample_csv: Path) -> None:
    """Selection needs a block that is not the test block."""
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(DataError):
            prepare_sentiment_data(sample_csv, min_freq=1, val_size=bad)


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
    train, val, test = split_tokens(tokens, 0.05, 0.05)
    assert (len(train), len(val), len(test)) == (90, 5, 5)
    assert train + val + test == tokens, "blocks must tile the stream in order"


def test_the_test_block_is_carved_from_held_out_never_from_train() -> None:
    """The design choice that made a held-out block cost nothing.

    At 90/5/5 the training block is byte-identical to the old 90/10 one, so the
    vocabulary stays at 2,436 and the uniform baseline stays at ln V = 7.7981 --
    the number the whole D2 demonstration is quoted against. Taking test out of
    train instead would have moved it to 2,288 / 7.7354.
    """
    tokens = [str(i) for i in range(100)]
    two_way_train, _, empty = split_tokens(tokens, 0.10, 0.0)
    three_way_train, _, _ = split_tokens(tokens, 0.05, 0.05)
    assert two_way_train == three_way_train
    assert empty == []


@pytest.mark.parametrize("fraction", [1.0, -0.1, 1.5])
def test_bad_test_fraction_rejected(fraction: float) -> None:
    with pytest.raises(DataError, match="test_fraction"):
        split_tokens([str(i) for i in range(100)], 0.05, fraction)


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
    for block in ("train", "val", "test"):
        dataset = getattr(splits, block)
        assert len(dataset) == dataset.n_tokens - 10, f"{block} window count"
    tokens = tokenize(load_corpus(mini_book))
    # Three blocks, each losing seq_len windows at its own boundary.
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(tokens) - 30


def test_textgen_vocab_built_from_train_only(mini_book: Path) -> None:
    splits = prepare_textgen_data(mini_book, seq_len=10, min_freq=1)
    tokens = tokenize(load_corpus(mini_book))
    train_tokens, val_tokens, test_tokens = split_tokens(tokens, 0.05, 0.05)
    for token in (set(val_tokens) | set(test_tokens)) - set(train_tokens):
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
    """Measured 2026-08-30 under the three-way split, after deduplication.

    Validation is carved out of *train*, never out of test, so introducing it
    moved no test row: the block holds the 3,382 rows deduplication left, at a
    0.1952 positive rate. That is what keeps every number comparable across the
    change. (This docstring described the pre-deduplication 3,463 / 0.2047 block
    until 2026-08-30, while the assertions below already carried the corrected
    values -- the test was right and its own prose was not.)
    """
    s = sentiment_splits
    assert (len(s.train), len(s.val), len(s.test)) == (6705, 1184, 3382)
    assert len(s.train) + len(s.val) + len(s.test) == 11271
    assert s.train.label_counts == {"negative": 5398, "positive": 1307}
    assert s.train.label_counts["positive"] / len(s.train) == pytest.approx(0.1949, abs=1e-4)
    assert s.val.label_counts["positive"] / len(s.val) == pytest.approx(0.1951, abs=1e-4)
    assert s.test.label_counts["positive"] / len(s.test) == pytest.approx(0.1952, abs=1e-4)
    assert len(s.vocab) == 4045
    assert s.class_weights.tolist() == pytest.approx([1.0, 4.130], abs=1e-3)
    assert s.test.unknown_rate() == pytest.approx(0.0577, abs=1e-4)
    # The training rate must be NONZERO: min_freq=2 drops training hapax, and
    # that is exactly what gives the <unk> embedding row gradient signal. At 0,
    # <unk> would be a randomly-initialised row first used at inference time.
    assert s.train.unknown_rate() == pytest.approx(0.0370, abs=1e-4)
    assert 0.0 < s.train.unknown_rate() < s.test.unknown_rate()


@pytest.mark.realdata
def test_textgen_measured_values(textgen_splits) -> None:
    t = textgen_splits
    assert t.n_tokens == 27_429
    assert (t.train.n_tokens, t.val.n_tokens, t.test.n_tokens) == (24_687, 1_371, 1_371)
    # The training block is byte-identical to the old two-way 90/10 split, which
    # is why V and the uniform baseline did not move when test was added.
    assert len(t.vocab) == 2_436  # train-only @ min_freq=1
    assert len(t.train) + len(t.val) + len(t.test) == 27_399  # 27,429 - 3 x seq_len
    # min_freq=1 => every training token earned an index
    assert int((t.train.ids == t.vocab.unk_index).sum()) == 0


@pytest.mark.realdata
def test_textgen_memory_footprint(textgen_splits) -> None:
    """D9: 667 MB of one-hot becomes 0.22 MB of lazily-sliced indices.

    667 MB is what these windows would cost one-hot at V=2,436; the reference
    actually allocated 931 MB at its own larger vocabulary. This said "707 MB"
    until 2026-08-30 -- a hypothetical from the original plan, retired as a
    stale figure in Phase 1 and still surviving here because the audit's
    stale-figure gate read documents only.
    """
    t = textgen_splits
    blocks = (t.train, t.val, t.test)
    storage = sum(b.storage_nbytes() for b in blocks)
    dense = sum(b.dense_nbytes() for b in blocks)
    onehot = sum(b.onehot_nbytes(len(t.vocab)) for b in blocks)
    assert storage / 1e6 == pytest.approx(0.22, abs=0.01)
    assert dense / 1e6 == pytest.approx(2.19, abs=0.01)
    assert onehot / storage > 2_000


@pytest.mark.realdata
def test_no_gutenberg_vocabulary_in_textgen(textgen_splits) -> None:
    """D6 at the vocabulary level -- the licence words have no index at all."""
    for word in ("copyright", "donations", "foundation", "ebook", "license", "gutenberg"):
        assert word not in textgen_splits.vocab


# --------------------------------------------------------------------------- #
# deduplication -- no row may appear in two blocks
# --------------------------------------------------------------------------- #


def test_no_cleaned_text_appears_in_two_blocks(sample_csv: Path) -> None:
    """The property deduplication exists to guarantee.

    Previously, 86 of 3,463 test rows (2.48%) shared their cleaned text with
    a training row -- mostly stubs like "<user> thanks" -- so the model was partly
    scored on inputs it had memorised.
    """
    s = prepare_sentiment_data(sample_csv, min_freq=1)
    train, val, test = set(s.train.texts), set(s.val.texts), set(s.test.texts)
    assert not train & test
    assert not train & val
    assert not val & test


def test_without_deduplication_the_leak_returns(sample_csv: Path) -> None:
    """Negative control: the guarantee comes from the flag, not from luck.

    If the corpus happened to hold no duplicates, the test above would pass
    whatever the code did.
    """
    raw = load_sentiment_frame(sample_csv, deduplicate=False)
    deduped = load_sentiment_frame(sample_csv, deduplicate=True)
    assert len(deduped) < len(raw), "the fixture must contain duplicates to control against"


def test_deduplication_keeps_one_row_per_cleaned_text(sample_csv: Path) -> None:
    frame = load_sentiment_frame(sample_csv, deduplicate=True)
    assert frame["clean"].is_unique


def test_contradictory_labels_are_dropped_not_resolved() -> None:
    """A text labelled both ways is an annotation nobody can adjudicate.

    Keeping either label would be inventing one; keeping both would put the
    same input in two classes.
    """
    import pandas as pd

    from lstm_nlp.data.sentiment import _drop_duplicate_texts

    frame = pd.DataFrame({
        "clean": ["same text", "same text", "unique one", "twice", "twice"],
        "airline_sentiment": [0, 1, 1, 0, 0],
        "text": ["a", "b", "c", "d", "e"],
    })
    kept = _drop_duplicate_texts(frame)
    assert "same text" not in set(kept["clean"]), "contradictory text must be dropped"
    assert sorted(kept["clean"]) == ["twice", "unique one"]
    assert len(kept) == 2


def test_deduplication_is_order_stable(sample_csv: Path) -> None:
    """First occurrence wins, so the result depends on the file, not on chance."""
    a = load_sentiment_frame(sample_csv, deduplicate=True)
    b = load_sentiment_frame(sample_csv, deduplicate=True)
    assert list(a["clean"]) == list(b["clean"])
