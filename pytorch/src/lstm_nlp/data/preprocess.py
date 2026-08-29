"""Text cleaning and tokenisation.

Pure ``str -> str`` functions with no state and no torch dependency, so they stay
trivially testable (Rules.md section 4).

Two decisions here are load-bearing corrections to the TensorFlow reference:

* **No stopword removal.**  NLTK's English stopword list contains all 14
  negations (``not``, ``no``, ``nor``, ``don't``, ``isn't``, ...).  Stripping them
  before a *sentiment* model turns ``"chat support is not working"`` into
  ``"chat support working"`` -- the opposite meaning (D3).
* **Gutenberg boilerplate is removed before tokenisation.**  In ``alice.txt`` the
  story ends 88.2% of the way through the file; the remainder is the Project
  Gutenberg licence.  The reference trained and validated on it (D6).

Changing anything in this module changes what stored vocabulary indices mean, so
bump ``lstm_nlp.PREPROCESS_VERSION`` when you do.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# Project Gutenberg wraps the work in these markers.  Matched case-insensitively
# because the exact casing varies between releases.
_GUTENBERG_START = "*** start of the project gutenberg"
_GUTENBERG_END = "*** end of the project gutenberg"

_URL_RE = re.compile(r"http\S+|www\.\S+")
_USER_RE = re.compile(r"@\w+")
_AMP_RE = re.compile(r"&amp;")
# Keep digits, ASCII letters, apostrophes, and the angle brackets that delimit
# placeholder tokens.  Apostrophes are kept so "don't" survives as one token.
_TWEET_KEEP_RE = re.compile(r"[^0-9a-z<>'\s]+")
_BOOK_KEEP_RE = re.compile(r"[^0-9a-z'\s]+")
_WS_RE = re.compile(r"\s+")

URL_TOKEN = "<url>"
USER_TOKEN = "<user>"


def strip_gutenberg(raw: str) -> str:
    """Return only the work itself, dropping Project Gutenberg's header/footer.

    Slices between the ``*** START OF ... ***`` and ``*** END OF ... ***``
    markers.  A missing marker is not an error -- the corresponding end of the
    text is kept -- so this is safe on an already-stripped corpus.

    Args:
        raw: Full file contents.

    Returns:
        The body text between the markers.
    """
    lowered = raw.lower()

    start = lowered.find(_GUTENBERG_START)
    if start == -1:
        body_start = 0
    else:
        newline = lowered.find("\n", start)
        body_start = len(raw) if newline == -1 else newline + 1

    end = lowered.find(_GUTENBERG_END, body_start)
    body_end = len(raw) if end == -1 else end

    return raw[body_start:body_end]


def clean_tweet(text: str) -> str:
    """Normalise a tweet for sentiment classification.

    Lowercases, replaces URLs with ``<url>`` and @handles with ``<user>``,
    expands ``&amp;``, then drops every character that is not a digit, ASCII
    letter, apostrophe, angle bracket or space.

    Negations are preserved -- deliberately, and this is the point (D3).

    Args:
        text: Raw tweet.

    Returns:
        Cleaned text; may be empty if the input held no retainable characters.
    """
    text = text.lower()
    text = _URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _USER_RE.sub(f" {USER_TOKEN} ", text)
    text = _AMP_RE.sub(" and ", text)
    text = _TWEET_KEEP_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def clean_book(text: str) -> str:
    """Normalise prose for language modelling.

    Drops blank lines, lowercases, removes punctuation except apostrophes, and
    collapses the text to a single whitespace-separated string.  Line structure
    is discarded: the model sees one continuous token stream.
    """
    lines = [line.strip().lower() for line in text.split("\n") if line.strip()]
    joined = " ".join(lines)
    joined = _BOOK_KEEP_RE.sub(" ", joined)
    return _WS_RE.sub(" ", joined).strip()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens.

    Whitespace splitting is sufficient *because* the clean functions already
    normalised separators.  Do not call this on raw text.
    """
    return text.split()


def load_corpus(path: str | Path, *, strip_boilerplate: bool = True) -> str:
    """Read a text file and return its cleaned token stream as a string.

    Args:
        path: File to read.  Decoded as UTF-8, tolerating a BOM.
        strip_boilerplate: Remove the Project Gutenberg header/footer first.

    Returns:
        Cleaned text, ready for :func:`tokenize`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"corpus not found: {file_path.resolve()}")

    raw = file_path.read_text(encoding="utf-8-sig")
    if strip_boilerplate:
        raw = strip_gutenberg(raw)
    return clean_book(raw)


def count_tokens(texts: list[str]) -> Counter[str]:
    """Count token frequencies across already-cleaned texts."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return counts
