"""Temperature and top-k sampling. The fix for D2.

The reference implementation did this:

    pred = model.predict(x)[0]                    # softmax output: probabilities
    tf.random.categorical(pred / temperature)     # expects LOGITS

``tf.random.categorical`` applies its own softmax, so it must be given logits.
Feeding it probabilities divided by T=10 puts every input in roughly
[0, 0.007]; the softmax of a near-constant vector is near-uniform, so the
sampler drew words uniformly at random from the whole vocabulary -- at *every*
temperature setting. The notebook's own saved output is the proof:

    "the alone court, to. pop sits bursting particularly just conversation."

That cell is presented as a demonstration of softmax temperature. It
demonstrates the opposite.

Two things prevent it recurring here:

1. Models return raw logits (``Rules.md`` C1), so no probability vector exists
   to divide by mistake.
2. Temperature scales logits and nothing else (``Rules.md`` C2):
   ``softmax(logits / T)``.

If you find yourself dividing something that sums to 1 by a temperature, you
have rewritten D2.
"""

from __future__ import annotations

import torch

from lstm_nlp.errors import DataError

#: Below this, ``softmax(logits / T)`` is numerically a one-hot vector and
#: sampling is indistinguishable from argmax. Guards against division blowing up
#: as T approaches zero.
MIN_TEMPERATURE = 1e-3


def apply_top_k(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    """Mask all but the ``k`` highest logits with ``-inf``.

    Applied before temperature. Because dividing by a positive temperature is
    monotonic, the surviving set is the same either way -- the ordering is a
    readability choice, not a behavioural one.

    Args:
        logits: ``(V,)`` or ``(B, V)`` raw logits.
        k: How many to keep. ``None`` or ``k >= V`` keeps everything.

    Returns:
        A new tensor; the input is not modified.

    Raises:
        DataError: If ``k`` is less than 1.
    """
    if k is None:
        return logits
    if k < 1:
        raise DataError(f"top_k must be >= 1, got {k}")

    vocab_size = logits.shape[-1]
    if k >= vocab_size:
        return logits

    kth_value = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def temperature_distribution(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
) -> torch.Tensor:
    """Return the sampling distribution -- ``softmax(logits / T)``.

    This is the whole of the D2 fix, and it is deliberately a separate function
    from :func:`sample_from_logits` so the frontend can chart exactly the
    distribution the sampler will draw from (PRD FR-34).

    Args:
        logits: ``(V,)`` or ``(B, V)`` raw logits, never probabilities.
        temperature: Positive. Below 1 sharpens toward greedy; above 1 flattens
            toward uniform.
        top_k: Optional restriction applied first.

    Returns:
        Probabilities summing to 1 along the last dimension.

    Raises:
        DataError: If ``temperature`` is not positive.
    """
    if temperature <= 0:
        raise DataError(
            f"temperature must be > 0, got {temperature}. For greedy decoding "
            f"use a small temperature such as 0.01, or call argmax directly."
        )
    scaled = apply_top_k(logits, top_k) / max(temperature, MIN_TEMPERATURE)
    return torch.softmax(scaled, dim=-1)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Draw one token index from ``softmax(logits / T)``.

    Args:
        logits: ``(V,)`` raw logits for a single position.
        temperature: Positive. As it approaches zero this converges to
            :func:`greedy_from_logits`.
        top_k: Optional restriction to the k most likely tokens.
        generator: Explicit RNG. Pass one for reproducible generation; library
            code never touches the global RNG (``Rules.md`` §4).

    Returns:
        The sampled token index.

    Raises:
        DataError: If ``logits`` is not 1-D or the temperature is not positive.
    """
    if logits.ndim != 1:
        raise DataError(f"expected 1-D logits for a single position, got shape {tuple(logits.shape)}")

    probabilities = temperature_distribution(logits, temperature, top_k)
    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())


def greedy_from_logits(logits: torch.Tensor) -> int:
    """Return the single most likely token index."""
    return int(logits.argmax(dim=-1).item())


def distribution_entropy(probabilities: torch.Tensor) -> float:
    """Shannon entropy in nats.

    The measurable statement of what temperature does: entropy increases
    monotonically with T, from ~0 (one certain word) toward ln(V) (uniform).
    PRD S5 asserts exactly this.
    """
    safe = probabilities.clamp_min(1e-12)
    return float(-(safe * safe.log()).sum(dim=-1).mean())


def top_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    n: int = 12,
) -> list[tuple[int, float]]:
    """Return the ``n`` most likely ``(index, probability)`` pairs at ``T``.

    Feeds the frontend's next-word chart (PRD FR-34), so that the relationship
    between temperature and uncertainty is something a reader observes rather
    than something the documentation asserts.
    """
    probabilities = temperature_distribution(logits, temperature, top_k)
    n = min(n, probabilities.shape[-1])
    values, indices = probabilities.topk(n, dim=-1)
    return [(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist(), strict=True)]
