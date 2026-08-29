"""Many-to-one LSTM for next-word prediction.

A fixed 10-word window in, one word out. The "many-to-one" reduction is taking
``h_n[-1]`` -- the final hidden state of the top layer -- rather than the full
output sequence.

Two things distinguish this from the reference implementation:

* **Input is integer indices through an embedding, not one-hot.** The reference
  built a ``(30664, 10, 3036)`` bool array, 931 MB resident before batching
  (D9). An embedding layer is the same computation with a lookup instead of a
  matrix multiply against a mostly-zero vector.
* **``forward`` returns raw logits.** The reference ended in
  ``Dense(V, activation='softmax')``, and its sampler then divided those
  probabilities by the temperature and passed them to a function expecting
  logits -- which is D2. Returning logits makes that bug unrepresentable here:
  there is no probability vector lying around to divide.
"""

from __future__ import annotations

import torch
from torch import nn


class TextGenLSTM(nn.Module):
    """Embedding -> LSTM -> linear over the vocabulary.

    At the default configuration and a 2,436-token vocabulary this is 1,333,124
    parameters: 311,808 embedding, 395,264 LSTM, 626,052 head. The head is the
    largest part, which is what a word-level softmax costs.

    Args:
        vocab_size: Vocabulary size, including ``<unk>``.
        embed_dim: Embedding width.
        hidden_dim: LSTM hidden width.
        num_layers: Stacked LSTM layers.
        dropout: Dropout between LSTM layers and before the head.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Score the next word for each window in the batch.

        Args:
            ids: ``(B, seq_len)`` int64 token indices. Windows are fixed-length,
                so there is no padding and no packing to do.

        Returns:
            ``(B, vocab_size)`` raw logits over the vocabulary. Never
            probabilities -- see the module docstring.
        """
        embedded = self.embedding(ids)
        _, (h_n, _) = self.lstm(embedded)
        return self.head(self.dropout(h_n[-1]))

    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config(self) -> dict[str, int | float]:
        """Constructor arguments, for embedding in a checkpoint (PRD FR-25)."""
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embedding.embedding_dim,
            "hidden_dim": self.lstm.hidden_size,
            "num_layers": self.lstm.num_layers,
            "dropout": self.dropout.p,
        }
