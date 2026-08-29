"""Phase 2: the sentiment model. Regression tests for C1, C10."""

from __future__ import annotations

import pytest
import torch

from lstm_nlp.models.sentiment_lstm import SentimentLSTM


@pytest.fixture
def model() -> SentimentLSTM:
    torch.manual_seed(0)
    return SentimentLSTM(vocab_size=50, embed_dim=8, hidden_dim=8, num_layers=2, dropout=0.1)


def test_forward_shape(model: SentimentLSTM) -> None:
    ids = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 0, 0, 0]])
    out = model(ids, torch.tensor([3, 2]))
    assert out.shape == (2, 2)
    assert out.dtype == torch.float32


def test_returns_logits_not_probabilities(model: SentimentLSTM) -> None:
    """C1: softmax belongs to the loss and the sampler, never to the model.

    The reference ended in Dense(2, activation='softmax') and downstream code
    then treated its probability output as logits -- the shape of D2.
    """
    model.eval()
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    with torch.no_grad():
        out = model(ids, torch.tensor([5]))
    assert abs(out.sum().item() - 1.0) > 1e-3, "output sums to 1: the model is softmaxing"
    assert bool((out < 0).any()) or out.max().item() > 1.0


def test_padding_does_not_change_the_answer(model: SentimentLSTM) -> None:
    """C10: pack_padded_sequence means h_n describes the sentence, not the row.

    Without packing, the LSTM consumes the padding and the final hidden state
    reflects trailing zeros -- so the same sentence padded to different widths
    would classify differently.
    """
    model.eval()
    sentence = [7, 8, 9]
    short = torch.tensor([sentence + [0] * 2])
    long = torch.tensor([sentence + [0] * 20])
    with torch.no_grad():
        a = model(short, torch.tensor([3]))
        b = model(long, torch.tensor([3]))
    assert torch.allclose(a, b, atol=1e-6), "padding width changed the prediction"


def test_pad_embedding_row_is_zero_and_frozen(model: SentimentLSTM) -> None:
    """padding_idx=0 keeps the pad row at zero and gradient-free."""
    assert torch.allclose(model.embedding.weight[0], torch.zeros(model.embedding.embedding_dim))

    model.train()
    ids = torch.tensor([[1, 2, 0, 0]])
    out = model(ids, torch.tensor([2]))
    out.sum().backward()
    assert torch.allclose(
        model.embedding.weight.grad[0], torch.zeros(model.embedding.embedding_dim)
    ), "the <pad> row received a gradient"


def test_batch_items_are_independent(model: SentimentLSTM) -> None:
    """A row's prediction must not depend on its neighbours in the batch."""
    model.eval()
    with torch.no_grad():
        alone = model(torch.tensor([[4, 5, 6]]), torch.tensor([3]))
        batched = model(
            torch.tensor([[4, 5, 6, 0], [1, 2, 3, 4]]), torch.tensor([3, 4])
        )
    assert torch.allclose(alone[0], batched[0], atol=1e-6)


def test_parameter_count_matches_the_specification() -> None:
    """Architecture.md section 3.1 states 355,010 at the real vocabulary."""
    model = SentimentLSTM(vocab_size=4505, embed_dim=64, hidden_dim=64, num_layers=2)
    assert model.num_parameters() == 355_010


def test_config_round_trips_through_the_constructor() -> None:
    """model_cfg must be enough to rebuild the architecture (PRD FR-25)."""
    original = SentimentLSTM(vocab_size=101, embed_dim=16, hidden_dim=12, num_layers=2, dropout=0.2)
    rebuilt = SentimentLSTM(**original.config())
    assert rebuilt.config() == original.config()
    assert rebuilt.num_parameters() == original.num_parameters()
    rebuilt.load_state_dict(original.state_dict())


def test_single_layer_model_is_valid() -> None:
    model = SentimentLSTM(vocab_size=30, embed_dim=8, hidden_dim=8, num_layers=1, dropout=0.5)
    out = model(torch.tensor([[1, 2, 3]]), torch.tensor([3]))
    assert out.shape == (1, 2)


def test_length_one_sequence(model: SentimentLSTM) -> None:
    """An empty tweet encodes to a single <unk>; that must not crash."""
    out = model(torch.tensor([[1]]), torch.tensor([1]))
    assert out.shape == (1, 2)
