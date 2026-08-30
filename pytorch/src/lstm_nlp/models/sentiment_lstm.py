"""Many-to-one LSTM for sentiment classification.

A variable-length token sequence in, one label out.  The "many-to-one" reduction
is taking the final hidden state of the top LSTM layer -- and taking it at each
sequence's *true* final token, not at the end of the padded row, which is what
``pack_padded_sequence`` buys.

``forward`` returns **raw logits**.  Softmax belongs to ``CrossEntropyLoss`` and
to the sampler, never to the model (``Rules.md`` C1).  This is the structural fix
for D2: the reference's model ended in ``Dense(2, activation='softmax')``, and
downstream code then treated its probability output as though it were logits.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentLSTM(nn.Module):
    """Embedding -> stacked LSTM -> dropout -> linear.

    At the default configuration and the 4,045-token vocabulary the deduplicated
    training block produces, this is 325,570 parameters: 258,880 embedding,
    66,560 LSTM, 130 head.  It was documented as 355,010 until 2026-08-30 --
    the count at the V=4,505 of the pre-deduplication two-way split.

    Args:
        vocab_size: Number of embedding rows, including ``<pad>`` and ``<unk>``.
        embed_dim: Embedding width.
        hidden_dim: LSTM hidden width.
        num_layers: Stacked LSTM layers.
        dropout: Dropout between LSTM layers; PyTorch ignores it when
            ``num_layers == 1``, so it is also applied before the head.
        num_classes: Output classes.
        pad_idx: Index of ``<pad>``. Its embedding row is fixed at zero and
            receives no gradient (``Rules.md`` C10).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Classify a padded batch.

        Args:
            ids: ``(B, L)`` int64 token indices, right-padded with ``pad_idx``.
            lengths: ``(B,)`` true lengths. Must be on CPU -- a PyTorch
                requirement of ``pack_padded_sequence``.

        Returns:
            ``(B, num_classes)`` raw logits. Never probabilities.
        """
        embedded = self.embedding(ids)

        # Packing is what makes this correct rather than merely convenient:
        # without it the LSTM would consume the padding and h_n would describe
        # the end of the padded row instead of the end of the sentence.
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        # h_n is (num_layers, B, hidden); the last layer is the many-to-one
        # reduction of the whole sequence.
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
            "num_classes": self.head.out_features,
            "pad_idx": self.pad_idx,
        }
