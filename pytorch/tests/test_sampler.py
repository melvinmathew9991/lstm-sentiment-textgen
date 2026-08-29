"""Phase 4: temperature and top-k sampling. Regression tests for D2.

PRD S5 is the headline gate of the whole project, because D2 is the defect the
reference presented as a *feature*: its notebook has a cell captioned as a
demonstration of softmax temperature that in fact samples uniformly at random.

These tests assert the property that caption claimed.
"""

from __future__ import annotations

import math

import pytest
import torch

from lstm_nlp.errors import DataError
from lstm_nlp.inference.sampler import (
    apply_top_k,
    distribution_entropy,
    greedy_from_logits,
    sample_from_logits,
    temperature_distribution,
    top_tokens,
)

TEMPERATURES = [0.2, 0.5, 1.0, 1.5, 2.0]


@pytest.fixture
def logits() -> torch.Tensor:
    """A realistically peaked distribution over 50 tokens."""
    torch.manual_seed(0)
    values = torch.randn(50) * 2.0
    values[7] += 6.0  # one clear favourite, as a trained LM produces
    return values


# --------------------------------------------------------------------------- #
# S5 -- the headline gate
# --------------------------------------------------------------------------- #


def test_entropy_monotonic_in_temperature(logits: torch.Tensor) -> None:
    """Entropy must increase strictly with temperature.

    This is what "temperature controls randomness" means, stated so it can be
    measured. The reference's sampler produced near-uniform output at every
    setting, so its entropy curve was flat -- this test would have caught it.
    """
    entropies = [
        distribution_entropy(temperature_distribution(logits, t)) for t in TEMPERATURES
    ]
    for lower, higher, t_lo, t_hi in zip(
        entropies, entropies[1:], TEMPERATURES, TEMPERATURES[1:], strict=False
    ):
        assert higher > lower, (
            f"entropy did not increase from T={t_lo} ({lower:.4f}) to T={t_hi} ({higher:.4f})"
        )


def test_entropy_spans_from_near_certain_to_near_uniform(logits: torch.Tensor) -> None:
    """The endpoints are the two behaviours users need to feel."""
    vocab_size = logits.shape[0]
    cold = distribution_entropy(temperature_distribution(logits, 0.05))
    hot = distribution_entropy(temperature_distribution(logits, 50.0))
    assert cold < 0.1, f"T=0.05 should be nearly certain, entropy was {cold:.4f}"
    assert hot > 0.95 * math.log(vocab_size), (
        f"a very high temperature should approach uniform entropy "
        f"{math.log(vocab_size):.4f}, got {hot:.4f}"
    )


def test_low_temperature_equals_argmax(logits: torch.Tensor) -> None:
    """As T approaches zero, sampling converges to greedy decoding (PRD S5)."""
    generator = torch.Generator().manual_seed(0)
    expected = greedy_from_logits(logits)
    draws = [sample_from_logits(logits, temperature=0.01, generator=generator) for _ in range(1000)]
    agreement = sum(d == expected for d in draws) / len(draws)
    assert agreement >= 0.99, f"T=0.01 matched argmax only {agreement:.1%} of the time"


def test_high_temperature_spreads_the_draws(logits: torch.Tensor) -> None:
    """The other end: high T must actually explore the vocabulary."""
    generator = torch.Generator().manual_seed(0)
    cold = {sample_from_logits(logits, 0.1, generator=generator) for _ in range(300)}
    hot = {sample_from_logits(logits, 5.0, generator=generator) for _ in range(300)}
    assert len(hot) > 5 * len(cold)


def test_top_k_restricts_support(logits: torch.Tensor) -> None:
    """With top_k=5, ten thousand draws must produce at most five tokens."""
    generator = torch.Generator().manual_seed(0)
    drawn = {
        sample_from_logits(logits, temperature=1.5, top_k=5, generator=generator)
        for _ in range(10_000)
    }
    assert len(drawn) <= 5, f"top_k=5 produced {len(drawn)} distinct tokens"
    assert drawn <= set(logits.topk(5).indices.tolist())


def test_same_rng_seed_same_output(logits: torch.Tensor) -> None:
    """Reproducible generation (PRD FR-23)."""
    def draws(seed: int) -> list[int]:
        g = torch.Generator().manual_seed(seed)
        return [sample_from_logits(logits, 1.0, generator=g) for _ in range(50)]

    assert draws(123) == draws(123)
    assert draws(123) != draws(456)


# --------------------------------------------------------------------------- #
# the D2 mistake itself, demonstrated
# --------------------------------------------------------------------------- #


def test_dividing_probabilities_by_temperature_destroys_the_signal(
    logits: torch.Tensor,
) -> None:
    """Reproduces the reference's bug to show what it cost.

    The reference computed ``softmax(logits)`` in the model and then handed
    ``probabilities / T`` to a function that applies its own softmax. Dividing a
    vector that sums to 1 by T=10 leaves every entry in a tiny range, and the
    softmax of a near-constant vector is near-uniform -- so it sampled words at
    random, at every temperature.

    Correct scaling keeps the signal; the reference's does not.
    """
    vocab_size = logits.shape[0]
    uniform_entropy = math.log(vocab_size)

    correct = distribution_entropy(temperature_distribution(logits, 10.0))

    probabilities = torch.softmax(logits, dim=-1)
    reference_bug = distribution_entropy(torch.softmax(probabilities / 10.0, dim=-1))

    assert reference_bug > 0.999 * uniform_entropy, (
        "expected the reference's formulation to be indistinguishable from uniform"
    )
    assert correct < reference_bug, "correct scaling must retain more signal than the bug"


def test_temperature_scales_logits_not_probabilities(logits: torch.Tensor) -> None:
    """C2 as an identity: the distribution IS softmax(logits / T)."""
    expected = torch.softmax(logits / 0.7, dim=-1)
    assert torch.allclose(temperature_distribution(logits, 0.7), expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# apply_top_k
# --------------------------------------------------------------------------- #


def test_top_k_masks_everything_below_the_kth() -> None:
    values = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    masked = apply_top_k(values, 2)
    assert torch.isinf(masked[[0, 2, 3]]).all()
    assert masked[1] == 5.0 and masked[4] == 4.0


def test_top_k_none_or_oversized_is_a_no_op() -> None:
    values = torch.tensor([1.0, 5.0, 3.0])
    assert torch.equal(apply_top_k(values, None), values)
    assert torch.equal(apply_top_k(values, 3), values)
    assert torch.equal(apply_top_k(values, 99), values)


def test_top_k_does_not_mutate_its_input() -> None:
    values = torch.tensor([1.0, 5.0, 3.0])
    apply_top_k(values, 1)
    assert torch.equal(values, torch.tensor([1.0, 5.0, 3.0]))


def test_top_k_below_one_rejected() -> None:
    with pytest.raises(DataError, match="top_k must be >= 1"):
        apply_top_k(torch.tensor([1.0, 2.0]), 0)


def test_masked_tokens_have_zero_probability() -> None:
    probabilities = temperature_distribution(torch.tensor([1.0, 5.0, 3.0, 2.0]), 1.0, top_k=2)
    assert probabilities[0] == 0.0 and probabilities[3] == 0.0
    assert probabilities.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# validation and shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("temperature", [0.0, -1.0, -0.5])
def test_non_positive_temperature_rejected(logits: torch.Tensor, temperature: float) -> None:
    """T=0 would divide by zero. The message points at the greedy alternative."""
    with pytest.raises(DataError, match="temperature must be > 0"):
        temperature_distribution(logits, temperature)


def test_distribution_sums_to_one(logits: torch.Tensor) -> None:
    for t in TEMPERATURES:
        assert temperature_distribution(logits, t).sum() == pytest.approx(1.0, abs=1e-5)


def test_sample_requires_one_dimensional_logits() -> None:
    with pytest.raises(DataError, match="1-D logits"):
        sample_from_logits(torch.randn(2, 10))


def test_top_tokens_are_sorted_and_sum_below_one(logits: torch.Tensor) -> None:
    pairs = top_tokens(logits, 1.0, n=5)
    probabilities = [p for _, p in pairs]
    assert probabilities == sorted(probabilities, reverse=True)
    assert 0 < sum(probabilities) <= 1.0 + 1e-6
    assert pairs[0][0] == greedy_from_logits(logits)


def test_entropy_of_a_one_hot_distribution_is_zero() -> None:
    assert distribution_entropy(torch.tensor([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)


def test_entropy_of_a_uniform_distribution_is_log_v() -> None:
    v = 8
    assert distribution_entropy(torch.full((v,), 1.0 / v)) == pytest.approx(math.log(v), abs=1e-6)
