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
    """Architecture.md section 3.1 states 325,570 at the real vocabulary.

    The vocabulary here must stay in step with the one the config actually
    produces. Until 2026-08-30 this used a superseded vocabulary size
    -- the count from before the validation split and before deduplication --
    so it asserted the *sentence* in the document rather than the *artifact*,
    and it would have passed unchanged however far the two drifted apart. They
    drifted: the shipped model has held 325,570 parameters since v1.1.0 while
    section 3.1 previously totalled 258,880 + 66,560 + 130 as 355,010 -- a
    superseded count that contradicted its own line items.
    ``test_real_vocabulary_produces_the_documented_parameter_count`` is the one
    that pins the artifact; this one pins the arithmetic cheaply.
    """
    model = SentimentLSTM(vocab_size=4045, embed_dim=64, hidden_dim=64, num_layers=2)
    assert model.num_parameters() == 325_570


def test_parameter_count_is_the_sum_of_its_parts() -> None:
    """A total that contradicts its own line items is the defect shape here.

    Checked at an arbitrary vocabulary so it holds for any configuration, not
    just the one in the document.
    """
    v, e, h, layers = 1_234, 64, 64, 2
    model = SentimentLSTM(vocab_size=v, embed_dim=e, hidden_dim=h, num_layers=layers)
    embedding = v * e
    lstm = sum(4 * h * (inp + h) + 2 * 4 * h for inp in (e, h))
    head = h * 2 + 2
    assert model.num_parameters() == embedding + lstm + head


@pytest.mark.realdata
def test_real_vocabulary_produces_the_documented_parameter_count(
    sentiment_splits,
) -> None:
    """Pin the artifact: the model built from the real split has 325,570 params."""
    model = SentimentLSTM(
        vocab_size=len(sentiment_splits.vocab), embed_dim=64, hidden_dim=64, num_layers=2
    )
    assert len(sentiment_splits.vocab) == 4_045
    assert model.num_parameters() == 325_570


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


# =========================================================================== #
# TextGenLSTM  (Phase 3)
# =========================================================================== #

from lstm_nlp.models.textgen_lstm import TextGenLSTM  # noqa: E402


@pytest.fixture
def lm() -> TextGenLSTM:
    torch.manual_seed(0)
    return TextGenLSTM(vocab_size=60, embed_dim=16, hidden_dim=24, num_layers=1)


def test_textgen_forward_shape(lm: TextGenLSTM) -> None:
    """A batch of 10-word windows scores every word in the vocabulary."""
    out = lm(torch.randint(0, 60, (4, 10)))
    assert out.shape == (4, 60)
    assert out.dtype == torch.float32


def test_textgen_returns_logits_not_probabilities(lm: TextGenLSTM) -> None:
    """C1, and the structural half of the D2 fix.

    With no probability vector in existence, the reference's mistake -- dividing
    probabilities by the temperature -- cannot be written here.
    """
    lm.eval()
    with torch.no_grad():
        out = lm(torch.randint(0, 60, (2, 10)))
    assert abs(out[0].sum().item() - 1.0) > 1e-3, "output sums to 1: the model is softmaxing"
    assert bool((out < 0).any()), "all-positive output suggests a softmax or relu head"


def test_textgen_uses_only_the_final_hidden_state(lm: TextGenLSTM) -> None:
    """Many-to-one: one prediction per window, not one per timestep."""
    out = lm(torch.randint(0, 60, (3, 10)))
    assert out.ndim == 2, "a many-to-many model would return (B, seq_len, V)"


def test_textgen_windows_are_independent(lm: TextGenLSTM) -> None:
    lm.eval()
    window = torch.randint(0, 60, (1, 10))
    other = torch.randint(0, 60, (1, 10))
    with torch.no_grad():
        alone = lm(window)
        batched = lm(torch.cat([window, other]))
    assert torch.allclose(alone[0], batched[0], atol=1e-6)


def test_textgen_parameter_count_matches_the_specification() -> None:
    """Architecture.md section 3.2 states 1,333,124 at the real vocabulary."""
    model = TextGenLSTM(vocab_size=2436, embed_dim=128, hidden_dim=256, num_layers=1)
    assert model.num_parameters() == 1_333_124


def test_textgen_config_round_trips() -> None:
    original = TextGenLSTM(vocab_size=77, embed_dim=32, hidden_dim=48, num_layers=2, dropout=0.1)
    rebuilt = TextGenLSTM(**original.config())
    assert rebuilt.config() == original.config()
    rebuilt.load_state_dict(original.state_dict())


def test_textgen_input_is_indices_not_onehot(lm: TextGenLSTM) -> None:
    """D9 at the model boundary: the model consumes int64 indices."""
    out = lm(torch.randint(0, 60, (2, 10), dtype=torch.int64))
    assert out.shape == (2, 60)
    with pytest.raises((RuntimeError, IndexError)):
        lm(torch.zeros(2, 10, 60))  # a one-hot tensor must not be accepted


# =========================================================================== #
# the trainer is task-agnostic  (Phase 3 exit criterion)
# =========================================================================== #


def test_trainer_contains_no_task_specific_code() -> None:
    """Phase 3 had to reuse engine/trainer.py unchanged.

    If the loop ever needs to know which task it is running, the abstraction is
    wrong and the abstraction gets fixed -- not the loop.
    """
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "src" / "lstm_nlp" / "engine" / "trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    # Scan CODE, not prose. The module docstring legitimately says the loop
    # knows nothing about sentiment or text generation; that sentence is not a
    # dependency on either.
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]

    code = ast.unparse(tree).lower()
    for word in ("sentiment", "textgen", "vocab", "tweet", "perplexity", "macro_f1"):
        assert word not in code, f"trainer.py code references {word!r}; it must stay task-agnostic"
