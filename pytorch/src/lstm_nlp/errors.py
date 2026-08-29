"""Typed exceptions.

Every failure mode that is *expected* gets a named class here, so callers can
distinguish "the config was wrong" from "the checkpoint is stale" from a genuine
bug.  Bare ``except:`` and ``except Exception: pass`` are prohibited (Rules.md
section 5); catch one of these instead.
"""

from __future__ import annotations


class LstmNlpError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(LstmNlpError):
    """A configuration file is missing, malformed, or semantically invalid."""


class DataError(LstmNlpError):
    """Input data is missing, empty, or does not have the expected shape."""


class CheckpointError(LstmNlpError):
    """A checkpoint could not be read, or is missing required fields."""


class PreprocessVersionMismatch(CheckpointError):
    """A checkpoint was written by a different tokenisation contract.

    Raised instead of loading, because loading would silently produce garbage:
    the stored vocabulary indices would no longer mean what the running
    preprocessing code thinks they mean (PRD FR-26).
    """

    def __init__(self, checkpoint_version: str, running_version: str) -> None:
        self.checkpoint_version = checkpoint_version
        self.running_version = running_version
        super().__init__(
            f"Checkpoint was written with preprocess version "
            f"{checkpoint_version!r} but this code runs version "
            f"{running_version!r}. Token indices would be misinterpreted. "
            f"Retrain the model, or check out the matching code revision."
        )


class VocabError(LstmNlpError):
    """A vocabulary is malformed or used inconsistently."""


class TrainingError(LstmNlpError):
    """Training could not proceed (e.g. loss diverged to NaN/inf)."""
