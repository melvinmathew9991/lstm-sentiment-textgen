"""Phase 1: text cleaning. Regression tests for D3 and D6."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lstm_nlp.data.preprocess import (
    URL_TOKEN,
    USER_TOKEN,
    clean_book,
    clean_tweet,
    count_tokens,
    load_corpus,
    strip_gutenberg,
    tokenize,
)

# --------------------------------------------------------------------------- #
# D3 -- negation survival.  This is the whole reason stopword removal is gone.
# --------------------------------------------------------------------------- #

NEGATIONS = ["not", "no", "nor", "never", "don't", "isn't", "wasn't", "didn't",
             "doesn't", "won't", "can't", "couldn't", "shouldn't", "haven't"]


@pytest.mark.parametrize("word", NEGATIONS)
def test_negations_survive_cleaning(word: str) -> None:
    """NLTK's stopword list contains all of these; we must keep every one (D3)."""
    assert word in tokenize(clean_tweet(f"the flight was {word} good"))


def test_real_examples_keep_their_polarity() -> None:
    """The exact tweets whose meaning the reference pipeline inverted."""
    cases = [
        ("@VirginAmerica Your chat support is not working on your site", "not"),
        ("@VirginAmerica not worried, it's been a great ride", "not"),
        ("Called and emailed with no response", "no"),
    ]
    for raw, negation in cases:
        assert negation in tokenize(clean_tweet(raw)), f"{negation!r} lost from {raw!r}"


def test_apostrophes_are_preserved() -> None:
    """Contractions must stay single tokens, or don't/doesn't become don/doesn."""
    assert tokenize(clean_tweet("it doesn't work")) == ["it", "doesn't", "work"]


# --------------------------------------------------------------------------- #
# clean_tweet mechanics
# --------------------------------------------------------------------------- #


def test_lowercases() -> None:
    assert clean_tweet("FLIGHT Was LATE") == "flight was late"


def test_urls_become_placeholder() -> None:
    out = tokenize(clean_tweet("check http://t.co/abc123 and www.example.com now"))
    assert out.count(URL_TOKEN) == 2
    assert "t" not in out and "co" not in out


def test_handles_become_placeholder() -> None:
    out = tokenize(clean_tweet("@VirginAmerica and @united are late"))
    assert out.count(USER_TOKEN) == 2
    assert "virginamerica" not in out


def test_html_entity_expanded() -> None:
    assert tokenize(clean_tweet("crew &amp; staff")) == ["crew", "and", "staff"]


def test_digits_kept() -> None:
    assert "30" in tokenize(clean_tweet("waited 30 minutes"))


def test_punctuation_dropped_and_whitespace_collapsed() -> None:
    assert clean_tweet("  wow!!!  really???   bad...  ") == "wow really bad"


def test_empty_and_punctuation_only_yield_empty_string() -> None:
    assert clean_tweet("") == ""
    assert clean_tweet("!!! ??? ...") == ""


def test_is_idempotent() -> None:
    once = clean_tweet("@user says http://x.co it's GREAT!!")
    assert clean_tweet(once) == once


# --------------------------------------------------------------------------- #
# D6 -- Gutenberg boilerplate
# --------------------------------------------------------------------------- #


def test_strip_gutenberg_removes_header_and_footer(mini_book: Path) -> None:
    raw = mini_book.read_text(encoding="utf-8")
    body = raw.lower()
    assert "copyright laws" in body and "donations" in body  # present before
    stripped = strip_gutenberg(raw).lower()
    assert "copyright laws" not in stripped
    assert "donations" not in stripped
    assert "*** start of" not in stripped
    assert "*** end of" not in stripped
    assert len(stripped) < len(raw)


def test_gutenberg_boilerplate_absent_after_full_pipeline(mini_book: Path) -> None:
    """The D6 regression test at the level callers actually use."""
    vocab = set(tokenize(load_corpus(mini_book)))
    for word in ("copyright", "donations", "foundation", "ebook", "license", "gutenberg"):
        assert word not in vocab, f"boilerplate token {word!r} survived"


def test_strip_is_idempotent_and_safe_without_markers() -> None:
    plain = "Just some prose with no markers at all."
    assert strip_gutenberg(plain) == plain
    once = strip_gutenberg("*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody\n")
    assert strip_gutenberg(once) == once


def test_marker_matching_is_case_insensitive() -> None:
    text = "head\n*** start of the project gutenberg ebook x ***\nbody\n"
    assert strip_gutenberg(text).strip() == "body"


def test_load_corpus_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="corpus not found"):
        load_corpus("no/such/file.txt")


def test_disabling_strip_keeps_boilerplate(mini_book: Path) -> None:
    """Proves the strip is what removes it -- not some other stage."""
    kept = set(tokenize(load_corpus(mini_book, strip_boilerplate=False)))
    assert "donations" in kept and "copyright" in kept


# --------------------------------------------------------------------------- #
# clean_book / tokenize / count_tokens
# --------------------------------------------------------------------------- #


def test_clean_book_drops_blank_lines_and_line_structure() -> None:
    assert clean_book("First line.\n\n\nSecond  line!\n") == "first line second line"


def test_clean_book_keeps_apostrophes_drops_angle_brackets() -> None:
    out = tokenize(clean_book("Alice's <b>tale</b>"))
    assert "alice's" in out and "b" in out and "<b>" not in out


def test_count_tokens_aggregates() -> None:
    counts = count_tokens(["a b a", "a c"])
    assert counts["a"] == 3 and counts["b"] == 1 and counts["c"] == 1


# --------------------------------------------------------------------------- #
# Rules.md section 4: importable without torch
# --------------------------------------------------------------------------- #


def test_preprocess_imports_without_torch() -> None:
    """Blocks torch at the import hook and imports the module anyway."""
    script = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch is blocked for this test')\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from lstm_nlp.data.preprocess import clean_tweet, strip_gutenberg\n"
        "assert clean_tweet('Not GOOD!') == 'not good'\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --------------------------------------------------------------------------- #
# measured values (Phases.md Phase 1)
# --------------------------------------------------------------------------- #


@pytest.mark.realdata
def test_alice_measured_values(alice_txt: Path) -> None:
    raw = alice_txt.read_text(encoding="utf-8-sig")
    body = strip_gutenberg(raw)
    assert len(raw) == 164_045
    assert len(body) == 144_607
    assert round(100 * len(body) / len(raw), 1) == 88.2

    tokens = tokenize(load_corpus(alice_txt))
    assert len(tokens) == 27_429
    assert len(set(tokens)) == 2_578


@pytest.mark.realdata
def test_sentiment_token_lengths(sentiment_csv: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(sentiment_csv)
    lengths = frame["text"].astype(str).map(lambda t: len(tokenize(clean_tweet(t))))
    assert len(frame) == 11_541
    assert int(lengths.median()) == 20
    assert int(lengths.quantile(0.95)) == 27
    assert int(lengths.max()) == 35
    assert int((lengths == 0).sum()) == 0
