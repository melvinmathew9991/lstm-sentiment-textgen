"""Shared fixtures.

Tests split into two groups:

* **fast** -- run against ``tests/fixtures/`` (200 rows, ~2k tokens), no real data.
* **realdata** -- assert the measured numbers in ``Phases.md`` against the actual
  corpora.  Skipped automatically if ``data/`` is absent, so the suite still runs
  on a checkout without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "realdata: needs the full corpora in data/")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def sample_csv() -> Path:
    """200 stratified rows of the sentiment data."""
    return FIXTURES / "sentiment_sample.csv"


@pytest.fixture(scope="session")
def mini_book() -> Path:
    """~2k tokens of prose wrapped in real Project Gutenberg markers."""
    return FIXTURES / "mini_book.txt"


@pytest.fixture(scope="session")
def sentiment_csv() -> Path:
    path = DATA_DIR / "airline_sentiment.csv"
    if not path.is_file():
        pytest.skip(f"real data not present: {path}")
    return path


@pytest.fixture(scope="session")
def alice_txt() -> Path:
    path = DATA_DIR / "alice.txt"
    if not path.is_file():
        pytest.skip(f"real data not present: {path}")
    return path


@pytest.fixture(scope="session")
def sentiment_splits(sentiment_csv: Path):
    """Prepared splits from the real CSV, built once for the session."""
    from lstm_nlp.data.sentiment import prepare_sentiment_data

    return prepare_sentiment_data(sentiment_csv, min_freq=2, max_len=30)


@pytest.fixture(scope="session")
def textgen_splits(alice_txt: Path):
    """Prepared splits from the real corpus, built once for the session."""
    from lstm_nlp.data.textgen import prepare_textgen_data

    return prepare_textgen_data(alice_txt, seq_len=10, min_freq=1)
