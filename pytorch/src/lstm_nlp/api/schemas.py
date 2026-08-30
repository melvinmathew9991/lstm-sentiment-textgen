"""Pydantic request and response models: the HTTP contract, executable.

Validation lives here rather than inside the route bodies for a reason beyond
tidiness. FastAPI derives the OpenAPI document from these classes, so the
constraints published at ``/docs`` are the constraints actually enforced --
they cannot drift apart, because they are the same object (Architecture.md
section 6).

A rejected request therefore never reaches a model. That matters for more than
latency: `temperature=0` reaching the sampler would be a division guarded by a
clamp, and a caller who wrote `0` deserves to be told so rather than quietly
served greedy decoding.

These bounds do not replace the domain checks in ``inference.sampler`` and
``config``. They are the same rule stated at the edge, where a caller can be
handed a precise 422; the inner checks remain because a Python caller reaching
``TextGenerator.generate`` directly must not be able to bypass them. One rule,
enforced at both boundaries -- not two opinions about what is legal.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Upper bound on sampling temperature accepted over HTTP.
#:
#: Well past useful -- the distribution is within 0.5% of uniform by T=5 -- but
#: bounded, because an unbounded float here is an invitation to send ``inf``.
MAX_TEMPERATURE = 5.0

#: Largest batch ``POST /predict/batch`` will accept in one request.
MAX_BATCH = 256

#: Largest passage ``POST /generate`` will produce in one request.
#:
#: Generation is sequential -- word *n* needs word *n-1* -- so this is the
#: knob that bounds request latency (NFR-5).
MAX_WORDS = 200

NonEmptyText = Annotated[str, Field(min_length=1, max_length=4000)]


class _Model(BaseModel):
    """Shared configuration: reject unknown fields rather than ignoring them."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------- #


class PredictRequest(_Model):
    """One string to classify."""

    text: NonEmptyText = Field(..., description="Raw text; cleaned server-side as in training.")

    @field_validator("text")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        """Reject whitespace-only input.

        ``min_length`` alone accepts ``" "``, which cleans to nothing and would
        surface as a confusing error from deep inside preprocessing.
        """
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value


class BatchPredictRequest(_Model):
    """Several strings to classify in one call."""

    texts: list[NonEmptyText] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH,
        description=f"Between 1 and {MAX_BATCH} strings.",
    )

    @field_validator("texts")
    @classmethod
    def _none_only_whitespace(cls, value: list[str]) -> list[str]:
        """Reject a batch containing a blank item, naming its position.

        A batch is answered positionally, so a caller who cannot see *which*
        item was rejected cannot line the error up with their input.
        """
        blank = [i for i, text in enumerate(value) if not text.strip()]
        if blank:
            raise ValueError(f"texts[{blank[0]}] is blank; every item must have content")
        return value


class GenerateRequest(_Model):
    """A seed and the sampling settings to continue it with."""

    seed: NonEmptyText = Field(..., description="Starting text. Unknown words become <unk>.")
    n_words: int = Field(40, ge=1, le=MAX_WORDS, description="How many words to generate.")
    temperature: float = Field(
        0.7,
        gt=0,
        le=MAX_TEMPERATURE,
        description="Applied to logits. Lower is greedier; higher is flatter.",
    )
    top_k: int | None = Field(None, ge=1, description="Restrict each step to the k likeliest.")
    rng_seed: int | None = Field(None, description="Set for reproducible output (FR-23).")

    @field_validator("seed")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        """Reject whitespace-only seeds, for the reason given on PredictRequest."""
        if not value.strip():
            raise ValueError("seed must contain at least one non-whitespace character")
        return value


class DistributionRequest(_Model):
    """A seed and a temperature, for inspecting the next-word distribution.

    This backs the frontend's temperature chart (FR-34). It exists as its own
    endpoint because the chart must show *exactly* the distribution the sampler
    draws from -- recomputing it client-side would be a second inference path,
    which is the thing C15 forbids.
    """

    seed: NonEmptyText = Field(..., description="Text whose next word is being examined.")
    temperature: float = Field(0.7, gt=0, le=MAX_TEMPERATURE)
    top_k: int | None = Field(None, ge=1)
    n: int = Field(12, ge=1, le=50, description="How many of the likeliest words to return.")

    @field_validator("seed")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        """Reject whitespace-only seeds, for the reason given on PredictRequest."""
        if not value.strip():
            raise ValueError("seed must contain at least one non-whitespace character")
        return value


# --------------------------------------------------------------------------- #
# responses
# --------------------------------------------------------------------------- #


class PredictResponse(_Model):
    """One classification, with the evidence needed to judge it.

    ``n_unk`` and ``unk_rate`` are part of the contract, not diagnostics. A
    prediction resting on mostly-unknown tokens is uninformative, and a caller
    that cannot see that will present it as though it were not.
    """

    label: str
    label_id: int
    probabilities: dict[str, float]
    n_tokens: int
    n_unk: int
    unk_rate: float
    #: Whether a validation-fitted temperature was applied. False means the
    #: number is a score, not a calibrated probability -- the caller has to be
    #: able to tell those apart.
    calibrated: bool = False


class BatchPredictResponse(_Model):
    """Classifications in request order."""

    predictions: list[PredictResponse]


class GenerateResponse(_Model):
    """A generated passage and the settings that produced it.

    The settings are echoed because reproducing a result requires them, and a
    caller that relied on server defaults does not otherwise know what they were.
    """

    text: str
    seed_tokens: list[str]
    generated_tokens: list[str]
    temperature: float
    top_k: int | None
    n_unk_in_seed: int


class WordProbability(_Model):
    """One candidate next word and its probability at the requested temperature."""

    word: str
    probability: float


class DistributionResponse(_Model):
    """The next-word distribution the sampler would draw from.

    ``entropy`` is reported beside ``uniform_entropy`` so the number can be
    judged rather than merely read -- the same rule that makes accuracy print
    beside its baseline (Rules.md C11). Their ratio is the whole of the D2
    demonstration: the reference's sampler sat at 100% of uniform for every
    temperature it was given.
    """

    temperature: float
    top_k: int | None
    words: list[WordProbability]
    entropy: float
    uniform_entropy: float
    vocab_size: int


class ModelInfo(_Model):
    """What a loaded checkpoint is and how it scored."""

    task: str
    model_class: str
    model_cfg: dict
    vocab_size: int
    metrics: dict
    created_utc: str
    lib_versions: dict[str, str]
    checkpoint_path: str


class ModelsResponse(_Model):
    """Every checkpoint this process has loaded."""

    models: list[ModelInfo]


class HealthResponse(_Model):
    """Liveness plus which models are actually usable.

    A process that started is not a process that can answer: the checkpoints
    are loaded at startup and either arrived or did not. ``models`` reports
    that per task so a caller can tell "the service is down" from "generation
    is unavailable because nobody has trained it yet".
    """

    status: str
    models: dict[str, bool]
    device: str


class ErrorResponse(_Model):
    """The body of the 500 and 503 responses this service produces.

    Deliberately just ``detail``: a message a caller can act on, and never a
    traceback (FR-30).

    A schema violation is *not* this shape. FastAPI answers those with the same
    ``detail`` key holding a **list** of per-field errors, which is what
    Architecture.md section 6 specifies and what a form needs in order to mark
    the offending field. So a client reads ``detail`` in both cases and must
    branch on its type -- documented here because discovering it from a
    traceback in production is the alternative.
    """

    detail: str
