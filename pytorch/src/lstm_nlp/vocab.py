"""Token <-> index mapping.

Hand-rolled rather than pulled from ``torchtext``, which is unmaintained and
hard-pins exact torch versions (Rules.md section 2).

Two invariants this class exists to enforce:

* **Built from the training split only.**  The reference built its vocabulary
  from 100% of the data before splitting, so 2,338 words occurred only in the
  test set and got embedding rows that were never trained (D7).  ``Vocab`` has no
  idea what a split is -- the caller must hand it training counts, and
  ``data.sentiment`` is the only place that does.
* **Never raises on an unknown token.**  Every miss maps to ``<unk>``.  The
  reference's ``word_to_int[word]`` raised ``KeyError`` on any unseen word, which
  made inference on new text impossible (D7, Rules.md C4).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from lstm_nlp.errors import VocabError

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

#: Specials for padded, variable-length tasks (sentiment).  ``<pad>`` must be
#: index 0 to match ``nn.Embedding(padding_idx=0)`` (Rules.md C10).
PADDED_SPECIALS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN)
#: Specials for fixed-window tasks (text generation), which need no padding.
UNPADDED_SPECIALS: tuple[str, ...] = (UNK_TOKEN,)


@dataclass(frozen=True)
class Vocab:
    """An immutable token/index mapping.

    Attributes:
        itos: Index -> token.  Specials occupy the leading positions.
        min_freq: Frequency threshold that produced this vocabulary.
        specials: The reserved tokens, in index order.
    """

    itos: tuple[str, ...]
    min_freq: int
    specials: tuple[str, ...]

    def __post_init__(self) -> None:
        if UNK_TOKEN not in self.specials:
            raise VocabError(f"specials must include {UNK_TOKEN!r}; got {self.specials}")
        if len(set(self.itos)) != len(self.itos):
            raise VocabError("itos contains duplicate tokens")
        if self.itos[: len(self.specials)] != self.specials:
            raise VocabError("specials must occupy the leading indices of itos")
        object.__setattr__(self, "_stoi", {tok: i for i, tok in enumerate(self.itos)})

    # -- construction ------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        counts: Counter[str],
        *,
        min_freq: int = 1,
        specials: Sequence[str] = PADDED_SPECIALS,
    ) -> Vocab:
        """Build a vocabulary from token counts.

        Tokens are ordered by descending frequency, ties broken alphabetically.
        The tie-break matters: without it the ordering depends on dict insertion
        order, and two runs at the same seed could produce different indices,
        breaking the reproducibility guarantee (PRD S10).

        Args:
            counts: Token frequencies, **from the training split only**.
            min_freq: Keep tokens occurring at least this many times.
            specials: Reserved tokens placed at the leading indices.

        Returns:
            The built vocabulary.

        Raises:
            VocabError: If ``min_freq`` < 1 or ``specials`` omits ``<unk>``.
        """
        if min_freq < 1:
            raise VocabError(f"min_freq must be >= 1, got {min_freq}")

        specials = tuple(specials)
        reserved = set(specials)
        kept = sorted(
            (tok for tok, n in counts.items() if n >= min_freq and tok not in reserved),
            key=lambda tok: (-counts[tok], tok),
        )
        return cls(itos=specials + tuple(kept), min_freq=min_freq, specials=specials)

    # -- lookup ------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi  # type: ignore[attr-defined]

    @property
    def unk_index(self) -> int:
        """Index of ``<unk>``."""
        return self._stoi[UNK_TOKEN]  # type: ignore[attr-defined]

    @property
    def pad_index(self) -> int | None:
        """Index of ``<pad>``, or ``None`` for an unpadded vocabulary."""
        return self._stoi.get(PAD_TOKEN)  # type: ignore[attr-defined]

    def index(self, token: str) -> int:
        """Return the index of ``token``, or ``<unk>``'s index. Never raises."""
        return self._stoi.get(token, self.unk_index)  # type: ignore[attr-defined]

    def token(self, index: int) -> str:
        """Return the token at ``index``.

        Raises:
            VocabError: If ``index`` is out of range -- that is a bug in the
                caller, not user input, so it is not silently absorbed.
        """
        if not 0 <= index < len(self.itos):
            raise VocabError(f"index {index} out of range for vocabulary of size {len(self.itos)}")
        return self.itos[index]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Map tokens to indices, unknowns to ``<unk>``."""
        return [self.index(tok) for tok in tokens]

    def decode(self, indices: Iterable[int], *, skip_specials: bool = True) -> list[str]:
        """Map indices back to tokens.

        Args:
            indices: Indices to convert.
            skip_specials: Drop reserved tokens from the output.  Padding is
                noise in rendered text, so this defaults to on.
        """
        out = []
        for i in indices:
            tok = self.token(int(i))
            if skip_specials and tok in self.specials:
                continue
            out.append(tok)
        return out

    def count_unknown(self, tokens: Iterable[str]) -> int:
        """Count tokens absent from the vocabulary.

        Surfaced to API callers so they can tell when a prediction rests on
        mostly-unknown input (Architecture.md section 6).
        """
        return sum(1 for tok in tokens if tok not in self)

    # -- serialisation ----------------------------------------------------- #

    def to_dict(self) -> dict:
        """Return a JSON-safe dict for embedding in a checkpoint (PRD FR-25)."""
        return {"itos": list(self.itos), "min_freq": self.min_freq, "specials": list(self.specials)}

    @classmethod
    def from_dict(cls, payload: dict) -> Vocab:
        """Rebuild from :meth:`to_dict` output.

        Raises:
            VocabError: If a required key is missing.
        """
        try:
            return cls(
                itos=tuple(payload["itos"]),
                min_freq=int(payload["min_freq"]),
                specials=tuple(payload["specials"]),
            )
        except KeyError as exc:
            raise VocabError(f"vocabulary payload missing key {exc}") from exc
