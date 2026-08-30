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


# --------------------------------------------------------------------------- #
# tiny checkpoints -- what lets the API and frontend suites run in CI
# --------------------------------------------------------------------------- #

#: Vocabulary for the fixture checkpoints. Real words from both corpora, so a
#: seed like "alice was" is in-vocabulary and a nonsense string is not.
TINY_WORDS = [
    "flight", "late", "crew", "rude", "great", "thanks", "delayed", "bag",
    "alice", "was", "beginning", "to", "get", "very", "tired", "of", "sitting",
]

#: Preprocessing contract for the fixture checkpoints.
TINY_SEQ_LEN = 6
TINY_MAX_LEN = 12


@pytest.fixture(scope="session")
def tiny_vocab():
    """A small but real vocabulary -- the same class production loads."""
    from collections import Counter

    from lstm_nlp.vocab import Vocab

    return Vocab.build(Counter({w: 5 for w in TINY_WORDS}), min_freq=1)


@pytest.fixture(scope="session")
def tiny_runs(tmp_path_factory: pytest.TempPathFactory, tiny_vocab) -> Path:
    """A ``runs/`` tree holding one tiny checkpoint per task.

    Weights are random and that is fine: nothing built on this asserts what the
    model *says*, only what the service and the UI do with it. What matters is
    that the files are complete and loadable with no other file present, which
    is the D8 contract both the API and the frontend depend on.

    This exists because ``test_predictor.py`` and friends skip wherever no model
    has been trained -- which means they skip in CI. Suites built on this
    fixture run on every push instead.
    """
    import torch

    from lstm_nlp.inference.checkpoint import save_checkpoint
    from lstm_nlp.models.sentiment_lstm import SentimentLSTM
    from lstm_nlp.models.textgen_lstm import TextGenLSTM

    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("tiny_runs")

    sentiment = SentimentLSTM(vocab_size=len(tiny_vocab), embed_dim=8, hidden_dim=8, num_layers=1)
    save_checkpoint(
        root / "sentiment" / "20260101T000000" / "best.pt",
        task="sentiment",
        model=sentiment,
        model_cfg=sentiment.config(),
        vocab=tiny_vocab,
        preprocess={"max_len": TINY_MAX_LEN, "min_freq": 1},
        # Deliberately synthetic. These held the real headline numbers of
        # the day until 2026-08-30, which meant a reader could lift a
        # measured-looking figure out of a fixture for a random-weight
        # model, and the stale-figure gate had one more place to chase.
        metrics={"accuracy": 0.5000, "baseline_accuracy": 0.2500,
                 "macro_f1": 0.5000, "baseline_macro_f1": 0.2500},
        train_info={"seed": 42, "best_epoch": 3},
    )

    textgen = TextGenLSTM(vocab_size=len(tiny_vocab), embed_dim=8, hidden_dim=8, num_layers=1)
    save_checkpoint(
        root / "textgen" / "20260101T000000" / "best.pt",
        task="textgen",
        model=textgen,
        model_cfg=textgen.config(),
        vocab=tiny_vocab,
        preprocess={"seq_len": TINY_SEQ_LEN, "min_freq": 1},
        # Synthetic, for the same reason as the sentiment block above.
        metrics={"perplexity": 100.00, "baseline_perplexity": 1000.0},
        train_info={"seed": 42, "best_epoch": 3},
    )
    return root
