"""FastAPI service: the HTTP contract from ``Architecture.md`` section 6.

Both checkpoints are loaded **once**, in the ``lifespan`` handler, and held for
the process lifetime (FR-29). Loading per request would be the difference
between a 30 ms answer and a 900 ms one, and would reread a 5 MB file to
produce a result it already had.

The service starts even when a checkpoint is missing. That is deliberate: a
process that refuses to boot because text generation has not been trained yet
cannot serve sentiment either, and cannot tell anyone *why* it is down.
``GET /health`` reports what actually loaded, and a route whose model is absent
answers 503 with an instruction rather than a stack trace (FR-30).

Every route in this module is a thin adapter. The inference lives in
``inference.predictor``, which the CLI calls too -- one inference path in the
system, which is what makes the API's numbers and the CLI's numbers the same
numbers by construction rather than by coincidence.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from lstm_nlp import __version__
from lstm_nlp.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DistributionRequest,
    DistributionResponse,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    WordProbability,
)
from lstm_nlp.errors import CheckpointError, DataError, LstmNlpError
from lstm_nlp.inference.predictor import SentimentPredictor, TextGenerator
from lstm_nlp.inference.sampler import distribution_entropy
from lstm_nlp.utils.device import describe_device, resolve_device
from lstm_nlp.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

#: Environment variables naming an explicit checkpoint, one per task.
#:
#: Paths are configuration, never literals in code (Rules.md C13). Unset, the
#: newest run under ``runs/<task>/`` is used, which is what a developer who has
#: just trained a model means.
ENV_SENTIMENT_CKPT = "LSTM_NLP_SENTIMENT_CKPT"
ENV_TEXTGEN_CKPT = "LSTM_NLP_TEXTGEN_CKPT"
ENV_RUNS_DIR = "LSTM_NLP_RUNS_DIR"
ENV_LOG_LEVEL = "LSTM_NLP_LOG_LEVEL"

#: HTTP 422, spelled as a number on purpose.
#:
#: Starlette renamed ``HTTP_422_UNPROCESSABLE_ENTITY`` to ``..._CONTENT`` and
#: deprecated the old symbol, so importing either one pins us to a version
#: range for no benefit. The status code itself has not moved since RFC 4918.
HTTP_422_UNPROCESSABLE = 422

#: Package root, used to locate ``runs/`` when nothing is configured.
PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def default_runs_dir() -> Path:
    """Where to look for trained runs when no explicit path is configured."""
    configured = os.environ.get(ENV_RUNS_DIR)
    return Path(configured) if configured else PACKAGE_ROOT / "runs"


def latest_checkpoint(task: str) -> Path | None:
    """Newest ``best.pt`` for ``task``, or ``None`` if the model is untrained.

    Run directories are timestamped (``20260829T211240``), so lexical order is
    chronological order and no filesystem mtime is consulted.

    Args:
        task: ``"sentiment"`` or ``"textgen"``.

    Returns:
        The path, or ``None`` when nothing has been trained.
    """
    task_dir = default_runs_dir() / task
    if not task_dir.is_dir():
        return None
    found = sorted(task_dir.glob("*/best.pt"))
    return found[-1] if found else None


def resolve_checkpoint(task: str, env_var: str) -> Path | None:
    """Resolve a task's checkpoint from the environment, else the newest run."""
    configured = os.environ.get(env_var)
    return Path(configured) if configured else latest_checkpoint(task)


@dataclass
class ModelRegistry:
    """The process's loaded models, and the record of how they were loaded.

    ``load_events`` is not instrumentation for its own sake. FR-29 says the
    models load once, and a claim like that is worth exactly as much as the
    test behind it -- so the registry counts, and a test asserts the count
    does not move across many requests.
    """

    sentiment: SentimentPredictor | None = None
    textgen: TextGenerator | None = None
    device: str = "cpu"
    load_events: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def availability(self) -> dict[str, bool]:
        """Which tasks can actually be served."""
        return {"sentiment": self.sentiment is not None, "textgen": self.textgen is not None}

    @property
    def any_loaded(self) -> bool:
        """True when at least one model is usable."""
        return any(self.availability.values())


def load_models() -> ModelRegistry:
    """Load every available checkpoint once.

    A checkpoint that is missing or unreadable is logged and recorded, not
    raised: one absent model must not take down the task that did load. The
    reason is kept in ``errors`` so ``/health`` and the 503 bodies can say what
    went wrong instead of only that something did.

    Returns:
        The populated registry.
    """
    registry = ModelRegistry(device=describe_device(resolve_device("cpu")))

    for task, env_var, cls in (
        ("sentiment", ENV_SENTIMENT_CKPT, SentimentPredictor),
        ("textgen", ENV_TEXTGEN_CKPT, TextGenerator),
    ):
        path = resolve_checkpoint(task, env_var)
        if path is None:
            reason = f"no {task} checkpoint found under {default_runs_dir() / task}"
            registry.errors[task] = reason
            logger.warning("%s; %s endpoints will answer 503", reason, task)
            continue
        try:
            setattr(registry, task, cls(path))
        except LstmNlpError as exc:
            registry.errors[task] = str(exc)
            logger.error("could not load %s checkpoint %s: %s", task, path, exc)
            continue
        registry.load_events.append(f"{task}:{path}")
        logger.info("loaded %s checkpoint %s", task, path)

    return registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models at startup and hold them for the process lifetime (FR-29).

    Logging is configured here, because this is the entry point (Rules.md
    section 4) -- but only when nothing has configured it already. Under
    ``uvicorn`` the root logger has no handlers, so the checkpoint-load lines
    would otherwise go nowhere and the operator-visible half of FR-29 would not
    exist. Under pytest, the harness owns root, and stamping on it would
    silently disable ``caplog`` for every test in the session.
    """
    if not logging.getLogger().handlers:
        setup_logging(os.environ.get(ENV_LOG_LEVEL, "INFO"))
    app.state.models = load_models()
    logger.info(
        "api ready: %s model(s) loaded on %s",
        sum(app.state.models.availability.values()),
        app.state.models.device,
    )
    yield


app = FastAPI(
    title="lstm-nlp",
    version=__version__,
    summary="Many-to-one LSTMs for sentiment detection and text generation.",
    description=(
        "Every response is produced by `lstm_nlp.inference`, the same code path the "
        "CLI uses. Sampling temperature scales **logits** -- see `POST /distribution`, "
        "which returns the exact distribution the sampler draws from."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# errors -- FR-30: 422 for bad input, 503 for an absent model, never a trace
# --------------------------------------------------------------------------- #


def _registry(request: Request) -> ModelRegistry:
    """The loaded models for this process."""
    return request.app.state.models


def _require(request: Request, task: str) -> SentimentPredictor | TextGenerator:
    """Return the predictor for ``task``, or raise if it never loaded.

    Raises:
        CheckpointError: If the task's checkpoint is unavailable. The handler
            below turns this into a 503 carrying the recorded reason.
    """
    registry = _registry(request)
    model = getattr(registry, task)
    if model is None:
        reason = registry.errors.get(task, "checkpoint not loaded")
        raise CheckpointError(f"{task} model not loaded; train it first ({reason})")
    return model


@app.exception_handler(CheckpointError)
async def _checkpoint_unavailable(request: Request, exc: CheckpointError) -> JSONResponse:
    """503: the service is up but this model is not."""
    logger.warning("503 on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(detail=str(exc)).model_dump(),
    )


@app.exception_handler(DataError)
async def _bad_input(request: Request, exc: DataError) -> JSONResponse:
    """422: the request parsed but the content is unusable.

    Reached by input that satisfies the schema and still cannot be processed --
    a seed of pure punctuation, which tokenises to nothing. That is the
    caller's input problem, so it earns the same status as a schema violation
    rather than a 500.
    """
    logger.debug("422 on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE,
        content=ErrorResponse(detail=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    """500: a bug. Logged in full here, described in one line to the caller.

    The traceback is what a maintainer needs and what a caller must never see
    (FR-30). Splitting it this way is the only way to have both.
    """
    logger.exception(
        "unhandled %s serving %s %s", type(exc).__name__, request.method, request.url.path
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(detail="internal error; see server logs").model_dump(),
    )


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health(request: Request) -> JSONResponse:
    """Report liveness and which models are usable.

    200 when at least one model loaded, 503 when none did. A monitor needs the
    status code; a human needs the per-task detail, so both are returned.
    """
    registry = _registry(request)
    body = HealthResponse(
        status="ok" if registry.any_loaded else "unavailable",
        models=registry.availability,
        device=registry.device,
    )
    code = status.HTTP_200_OK if registry.any_loaded else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=body.model_dump())


@app.get("/models", response_model=ModelsResponse, tags=["status"])
async def models(request: Request) -> ModelsResponse:
    """Describe every loaded checkpoint: architecture, vocabulary, metrics."""
    registry = _registry(request)
    described: list[ModelInfo] = []
    for task in ("sentiment", "textgen"):
        predictor = getattr(registry, task)
        if predictor is None:
            continue
        payload = predictor.payload
        path = next(
            (e.split(":", 1)[1] for e in registry.load_events if e.startswith(f"{task}:")), ""
        )
        described.append(
            ModelInfo(
                task=payload["task"],
                model_class=payload["model_class"],
                model_cfg=payload["model_cfg"],
                vocab_size=predictor.vocab_size,
                metrics=payload.get("metrics", {}),
                created_utc=payload.get("created_utc", ""),
                lib_versions=payload.get("lib_versions", {}),
                checkpoint_path=path,
            )
        )
    return ModelsResponse(models=described)


@app.post("/predict", response_model=PredictResponse, tags=["sentiment"])
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """Classify one string."""
    model = _require(request, "sentiment")
    return PredictResponse(**model.predict(body.text).to_dict())


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["sentiment"])
async def predict_batch(request: Request, body: BatchPredictRequest) -> BatchPredictResponse:
    """Classify several strings, answered in request order."""
    model = _require(request, "sentiment")
    results = model.predict_batch(body.texts)
    return BatchPredictResponse(
        predictions=[PredictResponse(**r.to_dict()) for r in results]
    )


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    """Continue a seed for ``n_words`` words.

    Seed words outside the vocabulary become ``<unk>`` and are counted rather
    than rejected (FR-24); ``n_unk_in_seed`` tells the caller how much of their
    seed the model could actually read.
    """
    model = _require(request, "textgen")
    started = time.perf_counter()
    result = model.generate(
        body.seed,
        n_words=body.n_words,
        temperature=body.temperature,
        top_k=body.top_k,
        rng_seed=body.rng_seed,
    )
    logger.info(
        "generated %d words at T=%.2f in %.0f ms",
        body.n_words,
        body.temperature,
        (time.perf_counter() - started) * 1000,
    )
    return GenerateResponse(**result.to_dict())


@app.post("/distribution", response_model=DistributionResponse, tags=["generation"])
async def distribution(request: Request, body: DistributionRequest) -> DistributionResponse:
    """Return the next-word distribution the sampler would draw from.

    This is the endpoint behind the frontend's temperature chart (FR-34), and
    the reason it exists on the server rather than in the client: the chart has
    to show the distribution that is actually sampled, not a re-derivation of
    it. ``entropy`` beside ``uniform_entropy`` is what makes the D2 claim
    checkable by anyone with a browser -- the reference's sampler would sit at
    the uniform value for every temperature it was handed.
    """
    model = _require(request, "textgen")
    words = model.next_word_distribution(
        body.seed, temperature=body.temperature, top_k=body.top_k, n=body.n
    )
    full = model.distribution_at(body.seed, body.temperature, body.top_k)
    vocab_size = model.vocab_size
    return DistributionResponse(
        temperature=body.temperature,
        top_k=body.top_k,
        words=[WordProbability(word=w, probability=p) for w, p in words],
        entropy=round(distribution_entropy(full), 4),
        uniform_entropy=round(math.log(vocab_size), 4),
        vocab_size=vocab_size,
    )
