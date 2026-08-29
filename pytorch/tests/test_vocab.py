"""Phase 1: vocabulary. Regression tests for D7."""

from __future__ import annotations

from collections import Counter

import pytest

from lstm_nlp.errors import VocabError
from lstm_nlp.vocab import (
    PAD_TOKEN,
    PADDED_SPECIALS,
    UNK_TOKEN,
    UNPADDED_SPECIALS,
    Vocab,
)


def make(counts: dict[str, int], **kw) -> Vocab:
    return Vocab.build(Counter(counts), **kw)


# --------------------------------------------------------------------------- #
# D7 -- unknown tokens must never raise
# --------------------------------------------------------------------------- #


def test_unknown_word_maps_to_unk() -> None:
    """``word_to_int[word]`` raising KeyError is what made the reference
    incapable of inference on new text (D7)."""
    vocab = make({"a": 5})
    assert vocab.index("qwertyuiop") == vocab.unk_index
    assert vocab.encode(["qwertyuiop"]) == [vocab.unk_index]


def test_encode_never_raises_on_arbitrary_input() -> None:
    vocab = make({"a": 5, "b": 3})
    weird = ["", "zzz", "123", "<pad>", "a", "éè"]
    assert len(vocab.encode(weird)) == len(weird)


def test_count_unknown() -> None:
    vocab = make({"a": 5, "b": 3})
    assert vocab.count_unknown(["a", "b", "x", "y"]) == 2
    assert vocab.count_unknown([]) == 0


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_specials_occupy_leading_indices() -> None:
    vocab = make({"z": 9}, specials=PADDED_SPECIALS)
    assert vocab.itos[0] == PAD_TOKEN
    assert vocab.itos[1] == UNK_TOKEN
    assert vocab.pad_index == 0
    assert vocab.unk_index == 1


def test_unpadded_vocab_has_no_pad() -> None:
    vocab = make({"z": 9}, specials=UNPADDED_SPECIALS)
    assert vocab.pad_index is None
    assert vocab.unk_index == 0


def test_ordered_by_frequency_then_alphabetically() -> None:
    """The tie-break keeps indices stable across runs (PRD S10)."""
    vocab = make({"rare": 1, "common": 10, "beta": 5, "alpha": 5}, min_freq=1)
    assert vocab.itos[2:] == ("common", "alpha", "beta", "rare")


def test_build_is_deterministic_regardless_of_insertion_order() -> None:
    pairs = [("a", 3), ("b", 3), ("c", 3), ("d", 3)]
    first = Vocab.build(Counter(dict(pairs)), min_freq=1)
    second = Vocab.build(Counter(dict(reversed(pairs))), min_freq=1)
    assert first.itos == second.itos


@pytest.mark.parametrize(("min_freq", "expected"), [(1, 4), (2, 3), (3, 2), (10, 0)])
def test_min_freq_filters(min_freq: int, expected: int) -> None:
    counts = {"a": 5, "b": 3, "c": 2, "d": 1}
    vocab = make(counts, min_freq=min_freq, specials=UNPADDED_SPECIALS)
    assert len(vocab) == expected + 1  # +1 for <unk>


def test_specials_are_not_duplicated_by_counts() -> None:
    vocab = make({UNK_TOKEN: 99, PAD_TOKEN: 99, "real": 5})
    assert vocab.itos.count(UNK_TOKEN) == 1
    assert vocab.itos.count(PAD_TOKEN) == 1


def test_len_and_contains() -> None:
    vocab = make({"a": 5, "b": 3}, min_freq=1)
    assert len(vocab) == 4
    assert "a" in vocab and "zzz" not in vocab


# --------------------------------------------------------------------------- #
# failure modes
# --------------------------------------------------------------------------- #


def test_min_freq_below_one_rejected() -> None:
    with pytest.raises(VocabError, match="min_freq must be >= 1"):
        make({"a": 1}, min_freq=0)


def test_specials_without_unk_rejected() -> None:
    with pytest.raises(VocabError, match="must include"):
        make({"a": 1}, specials=(PAD_TOKEN,))


def test_out_of_range_index_raises() -> None:
    """A bad index is a caller bug, not user input -- it must not be absorbed."""
    vocab = make({"a": 5})
    with pytest.raises(VocabError, match="out of range"):
        vocab.token(999)


def test_duplicate_itos_rejected() -> None:
    with pytest.raises(VocabError, match="duplicate"):
        Vocab(itos=(UNK_TOKEN, "a", "a"), min_freq=1, specials=(UNK_TOKEN,))


def test_misplaced_specials_rejected() -> None:
    with pytest.raises(VocabError, match="leading indices"):
        Vocab(itos=("a", UNK_TOKEN), min_freq=1, specials=(UNK_TOKEN,))


def test_vocab_is_immutable() -> None:
    vocab = make({"a": 5})
    with pytest.raises(Exception):
        vocab.min_freq = 99


# --------------------------------------------------------------------------- #
# encode / decode
# --------------------------------------------------------------------------- #


def test_encode_decode_round_trip() -> None:
    vocab = make({"the": 9, "cat": 5, "sat": 3}, min_freq=1)
    tokens = ["the", "cat", "sat"]
    assert vocab.decode(vocab.encode(tokens)) == tokens


def test_decode_skips_specials_by_default() -> None:
    vocab = make({"cat": 5}, min_freq=1)
    ids = [vocab.pad_index, vocab.unk_index, vocab.index("cat")]
    assert vocab.decode(ids) == ["cat"]
    assert vocab.decode(ids, skip_specials=False) == [PAD_TOKEN, UNK_TOKEN, "cat"]


# --------------------------------------------------------------------------- #
# serialisation -- the vocabulary travels inside the checkpoint (D8)
# --------------------------------------------------------------------------- #


def test_dict_round_trip_preserves_everything() -> None:
    original = make({"a": 5, "b": 3, "c": 1}, min_freq=2)
    restored = Vocab.from_dict(original.to_dict())
    assert restored == original
    assert restored.itos == original.itos
    assert restored.unk_index == original.unk_index


def test_to_dict_is_json_safe() -> None:
    import json

    payload = make({"a": 5}).to_dict()
    assert Vocab.from_dict(json.loads(json.dumps(payload))).itos == Vocab.from_dict(payload).itos


def test_from_dict_missing_key_raises() -> None:
    with pytest.raises(VocabError, match="missing key"):
        Vocab.from_dict({"itos": ["<unk>"]})
