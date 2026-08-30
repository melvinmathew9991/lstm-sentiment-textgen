"""Phase 6: the HTTP contract, every route and every documented failure (S12).

These tests build their own **tiny untrained checkpoints** rather than reaching
for ``runs/``. That is the deliberate difference from ``test_predictor.py``:
those tests skip wherever no model has been trained, which means they skip in
CI, which means CI has never checked them. A contract test that only runs on
one laptop is not a contract test. Random weights are fine here -- nothing
below asserts what the model *says*, only what the service does with it.

The two properties worth stating plainly, because they are what the phase is
for:

* **Load once** (FR-29). The registry counts its own load events and a test
  fires many requests at it to prove the count does not move.
* **No traceback ever reaches a body** (FR-30). Asserted against a route
  deliberately made to explode, not merely against the routes that behave.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from lstm_nlp.api import app as app_module
from lstm_nlp.api.schemas import MAX_BATCH, MAX_TEMPERATURE, MAX_WORDS
from tests.conftest import TINY_WORDS as WORDS


@pytest.fixture
def checkpoints(tiny_runs: Path) -> Path:
    """The shared tiny-checkpoint tree, from ``conftest``.

    Defined there rather than here because the frontend suite needs the same
    thing, and two definitions of "a valid checkpoint tree" would eventually
    disagree about what one is.
    """
    return tiny_runs


@pytest.fixture
def client(checkpoints: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose service loaded both tiny checkpoints."""
    monkeypatch.setenv(app_module.ENV_RUNS_DIR, str(checkpoints))
    monkeypatch.delenv(app_module.ENV_SENTIMENT_CKPT, raising=False)
    monkeypatch.delenv(app_module.ENV_TEXTGEN_CKPT, raising=False)
    with TestClient(app_module.app) as running:
        yield running


@pytest.fixture
def empty_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose service found no checkpoints at all -- the 503 world."""
    monkeypatch.setenv(app_module.ENV_RUNS_DIR, str(tmp_path / "nothing-here"))
    monkeypatch.delenv(app_module.ENV_SENTIMENT_CKPT, raising=False)
    monkeypatch.delenv(app_module.ENV_TEXTGEN_CKPT, raising=False)
    with TestClient(app_module.app) as running:
        yield running


# --------------------------------------------------------------------------- #
# GET /health
# --------------------------------------------------------------------------- #


def test_health_reports_both_models(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models"] == {"sentiment": True, "textgen": True}
    assert body["device"] == "cpu"


def test_health_is_503_when_nothing_loaded(empty_client: TestClient) -> None:
    """A process that started is not a process that can answer."""
    response = empty_client.get("/health")
    assert response.status_code == 503
    assert response.json()["models"] == {"sentiment": False, "textgen": False}


def test_health_distinguishes_down_from_untrained(empty_client: TestClient) -> None:
    """The caller must be able to tell "service dead" from "model never trained"."""
    body = empty_client.get("/health").json()
    assert body["status"] == "unavailable"
    assert set(body["models"]) == {"sentiment", "textgen"}


# --------------------------------------------------------------------------- #
# GET /models
# --------------------------------------------------------------------------- #


def test_models_describes_every_loaded_checkpoint(client: TestClient) -> None:
    body = client.get("/models").json()
    assert {m["task"] for m in body["models"]} == {"sentiment", "textgen"}
    for model in body["models"]:
        assert model["vocab_size"] == len(WORDS) + 2  # + <pad>, <unk>
        assert model["lib_versions"]["torch"] == torch.__version__
        assert model["created_utc"]


def test_models_reports_metrics_with_their_baselines(client: TestClient) -> None:
    """C11 at the HTTP boundary: a metric travels with what it is measured against."""
    body = client.get("/models").json()
    sentiment = next(m for m in body["models"] if m["task"] == "sentiment")
    assert sentiment["metrics"]["macro_f1"] == 0.8485
    assert sentiment["metrics"]["baseline_macro_f1"] == 0.4430


def test_models_is_empty_not_broken_when_nothing_loaded(empty_client: TestClient) -> None:
    assert empty_client.get("/models").json() == {"models": []}


# --------------------------------------------------------------------------- #
# POST /predict
# --------------------------------------------------------------------------- #


def test_predict_returns_the_documented_shape(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "the flight was late"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "label", "label_id", "probabilities", "n_tokens", "n_unk", "unk_rate",
        "calibrated",
    }
    assert body["label"] in ("negative", "positive")
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_surfaces_the_unknown_rate(client: TestClient) -> None:
    """A prediction resting on unreadable input must say so over HTTP too."""
    known = client.post("/predict", json={"text": "flight late crew"}).json()
    unknown = client.post("/predict", json={"text": "qwertyuiop zxcvbnm flurbulate"}).json()
    assert unknown["unk_rate"] > known["unk_rate"]
    assert unknown["unk_rate"] == 1.0


def test_predict_rejects_empty_text(client: TestClient) -> None:
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_rejects_whitespace_only_text(client: TestClient) -> None:
    """``min_length`` alone would accept this and fail confusingly further in."""
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_a_missing_field(client: TestClient) -> None:
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_unknown_fields(client: TestClient) -> None:
    """A typo'd field name is a caller error, not something to silently ignore."""
    response = client.post("/predict", json={"text": "hello", "temprature": 0.7})
    assert response.status_code == 422


def test_predict_is_503_when_untrained(empty_client: TestClient) -> None:
    response = empty_client.post("/predict", json={"text": "the flight was late"})
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /predict/batch
# --------------------------------------------------------------------------- #


def test_batch_answers_in_request_order(client: TestClient) -> None:
    texts = ["flight late", "crew rude", "great thanks"]
    body = client.post("/predict/batch", json={"texts": texts}).json()
    assert len(body["predictions"]) == 3
    singles = [client.post("/predict", json={"text": t}).json() for t in texts]
    assert body["predictions"] == singles


def test_batch_rejects_an_oversized_request(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"texts": ["ok"] * (MAX_BATCH + 1)})
    assert response.status_code == 422


def test_batch_accepts_exactly_the_limit(client: TestClient) -> None:
    """The boundary is inclusive; a test that only checks past it cannot say so."""
    response = client.post("/predict/batch", json={"texts": ["flight late"] * MAX_BATCH})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == MAX_BATCH


def test_batch_rejects_an_empty_list(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_batch_names_the_blank_item(client: TestClient) -> None:
    """A positional answer needs a positional error."""
    response = client.post("/predict/batch", json={"texts": ["fine", "  ", "fine"]})
    assert response.status_code == 422
    assert "texts[1]" in response.text


def test_batch_is_503_when_untrained(empty_client: TestClient) -> None:
    assert empty_client.post("/predict/batch", json={"texts": ["hi"]}).status_code == 503


# --------------------------------------------------------------------------- #
# POST /generate
# --------------------------------------------------------------------------- #


def test_generate_returns_the_documented_shape(client: TestClient) -> None:
    response = client.post("/generate", json={"seed": "alice was", "n_words": 5})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "text", "seed_tokens", "generated_tokens", "temperature", "top_k", "n_unk_in_seed",
    }
    assert len(body["generated_tokens"]) == 5
    assert body["text"].startswith("alice was")


def test_generate_echoes_the_settings_it_used(client: TestClient) -> None:
    """A caller relying on server defaults cannot reproduce a result without them."""
    body = client.post("/generate", json={"seed": "alice was"}).json()
    assert body["temperature"] == 0.7
    assert len(body["generated_tokens"]) == 40


def test_generate_is_reproducible_with_an_rng_seed(client: TestClient) -> None:
    """FR-23. Without this, nothing the service generates can be cited."""
    payload = {"seed": "alice was", "n_words": 12, "temperature": 1.0, "rng_seed": 123}
    first = client.post("/generate", json=payload).json()
    second = client.post("/generate", json=payload).json()
    assert first["generated_tokens"] == second["generated_tokens"]


def test_generate_differs_without_an_rng_seed(client: TestClient) -> None:
    """The converse: reproducibility must be a choice, not an accident of caching."""
    payload = {"seed": "alice was", "n_words": 30, "temperature": 2.0}
    runs = {tuple(client.post("/generate", json=payload).json()["generated_tokens"])
            for _ in range(4)}
    assert len(runs) > 1


def test_generate_counts_unknown_seed_words_rather_than_failing(client: TestClient) -> None:
    """FR-24: an unknown seed word becomes <unk> and is reported, not rejected."""
    body = client.post("/generate", json={"seed": "zzzznotaword alice", "n_words": 3}).json()
    assert body["n_unk_in_seed"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"seed": "alice was", "temperature": 0},
        {"seed": "alice was", "temperature": -1},
        {"seed": "alice was", "temperature": MAX_TEMPERATURE + 0.1},
        {"seed": "alice was", "n_words": 0},
        {"seed": "alice was", "n_words": MAX_WORDS + 1},
        {"seed": "alice was", "top_k": 0},
        {"seed": "", "n_words": 5},
        {"seed": "   ", "n_words": 5},
    ],
)
def test_generate_rejects_out_of_range_settings(client: TestClient, payload: dict) -> None:
    """Every bound in Architecture.md section 6, checked at its edge."""
    assert client.post("/generate", json=payload).status_code == 422


def test_generate_accepts_the_temperature_boundary(client: TestClient) -> None:
    """``le=5.0`` means 5.0 is legal. Asserting only the rejection would not say that."""
    response = client.post(
        "/generate", json={"seed": "alice was", "n_words": 2, "temperature": MAX_TEMPERATURE}
    )
    assert response.status_code == 200


def test_generate_rejects_a_seed_that_tokenises_to_nothing(client: TestClient) -> None:
    """Schema-valid, still unusable: DataError must surface as 422, never 500."""
    response = client.post("/generate", json={"seed": "!!! ???", "n_words": 3})
    assert response.status_code == 422
    assert "traceback" not in response.text.lower()


def test_generate_is_503_when_untrained(empty_client: TestClient) -> None:
    response = empty_client.post("/generate", json={"seed": "alice was"})
    assert response.status_code == 503
    assert "train it first" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /distribution -- the D2 payload, served
# --------------------------------------------------------------------------- #


def test_distribution_returns_words_with_probabilities(client: TestClient) -> None:
    body = client.post("/distribution", json={"seed": "alice was", "n": 5}).json()
    assert len(body["words"]) == 5
    assert all(0.0 <= w["probability"] <= 1.0 for w in body["words"])
    assert body["vocab_size"] == len(WORDS) + 2


def test_distribution_reports_entropy_against_uniform(client: TestClient) -> None:
    """C11 again: a number nobody can scale is a number nobody can judge."""
    body = client.post("/distribution", json={"seed": "alice was"}).json()
    assert body["uniform_entropy"] == pytest.approx(torch.log(torch.tensor(19.0)).item(), abs=1e-3)
    assert 0.0 <= body["entropy"] <= body["uniform_entropy"] + 1e-6


def test_distribution_entropy_rises_with_temperature(client: TestClient) -> None:
    """The whole of D2, as an HTTP property.

    The reference's sampler returned the uniform entropy at every temperature.
    Any implementation that reproduces that defect fails here, on random
    weights, without needing a trained model.
    """
    entropies = [
        client.post("/distribution", json={"seed": "alice was", "temperature": t}).json()["entropy"]
        for t in (0.2, 0.7, 1.0, 2.0, 5.0)
    ]
    assert entropies == sorted(entropies)
    assert entropies[0] < entropies[-1]


def test_distribution_top_k_truncates_the_support(client: TestClient) -> None:
    body = client.post("/distribution", json={"seed": "alice was", "top_k": 3, "n": 10}).json()
    non_zero = [w for w in body["words"] if w["probability"] > 0]
    assert len(non_zero) == 3


def test_distribution_is_503_when_untrained(empty_client: TestClient) -> None:
    assert empty_client.post("/distribution", json={"seed": "alice was"}).status_code == 503


# --------------------------------------------------------------------------- #
# FR-29 -- models load once
# --------------------------------------------------------------------------- #


def test_models_load_once_not_per_request(client: TestClient) -> None:
    """FR-29, asserted rather than asserted-about.

    The reference reloaded and re-derived state constantly; the cost of that is
    invisible until something counts. Twenty requests across three routes must
    not add a single load event.
    """
    registry = app_module.app.state.models
    assert len(registry.load_events) == 2

    for _ in range(6):
        client.post("/predict", json={"text": "flight late"})
        client.post("/generate", json={"seed": "alice was", "n_words": 2})
        client.get("/models")

    assert len(registry.load_events) == 2


def test_startup_logs_each_load_exactly_once(
    checkpoints: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The log is the operator-visible half of FR-29."""
    monkeypatch.setenv(app_module.ENV_RUNS_DIR, str(checkpoints))
    with caplog.at_level("INFO", logger="lstm_nlp.api.app"), TestClient(app_module.app) as running:
        for _ in range(5):
            running.post("/predict", json={"text": "flight late"})

    loaded = [r for r in caplog.records if r.message.startswith("loaded ")]
    assert len(loaded) == 2


# --------------------------------------------------------------------------- #
# FR-30 -- no stack trace ever reaches a body
# --------------------------------------------------------------------------- #


def test_unexpected_errors_return_500_without_a_traceback(
    checkpoints: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asserted against a route made to explode, not against routes that behave.

    A handler that has never been fired is indistinguishable from one that is
    broken -- the same reasoning as the D1 checker's negative controls.
    """

    monkeypatch.setenv(app_module.ENV_RUNS_DIR, str(checkpoints))

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret internal detail at /etc/passwd line 42")

    # Patch *inside* the context manager: entering a TestClient runs lifespan,
    # which rebuilds app.state.models and would discard a patch applied first.
    with TestClient(app_module.app, raise_server_exceptions=False) as raw:
        monkeypatch.setattr(app_module.app.state.models.sentiment, "predict", boom)
        response = raw.post("/predict", json={"text": "flight late"})

    assert response.status_code == 500
    assert response.json() == {"detail": "internal error; see server logs"}
    assert "RuntimeError" not in response.text
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text


def test_validation_errors_do_not_leak_internals(client: TestClient) -> None:
    response = client.post("/generate", json={"seed": "alice was", "temperature": -5})
    assert response.status_code == 422
    assert "Traceback" not in response.text
    assert "lstm_nlp" not in response.text


# --------------------------------------------------------------------------- #
# the contract is the documentation
# --------------------------------------------------------------------------- #


def test_openapi_publishes_the_real_constraints(client: TestClient) -> None:
    """The bounds at /docs are the bounds enforced, because they are one object."""
    schema = client.get("/openapi.json").json()
    generate = schema["components"]["schemas"]["GenerateRequest"]["properties"]
    assert generate["temperature"]["exclusiveMinimum"] == 0
    assert generate["temperature"]["maximum"] == MAX_TEMPERATURE
    assert generate["n_words"]["maximum"] == MAX_WORDS


def test_every_documented_route_exists(client: TestClient) -> None:
    """Architecture.md section 6, enumerated."""
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/health", "/models", "/predict", "/predict/batch", "/generate"} <= paths


def test_docs_render(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200


# --------------------------------------------------------------------------- #
# NFR-5 -- warm latency, against the real models
# --------------------------------------------------------------------------- #

REAL_RUNS = Path(__file__).resolve().parents[1] / "runs"

#: NFR-5 budgets, in milliseconds.
PREDICT_BUDGET_MS = 100
GENERATE_BUDGET_MS = 2000


def _real_client() -> TestClient:
    """A client over the actually-trained checkpoints, or a skip."""
    if not any(REAL_RUNS.glob("*/*/best.pt")):
        pytest.skip("no trained checkpoints; run: lstm-nlp train --config configs/sentiment.yaml")
    return TestClient(app_module.app)


def _warm_median_ms(client: TestClient, path: str, payload: dict, n: int = 15) -> float:
    """Median wall time of ``n`` warm calls.

    Median, not mean: one GC pause should not decide whether a budget is met,
    and NFR-5 is about the latency a caller typically sees.
    """
    for _ in range(3):
        client.post(path, json=payload)
    samples = []
    for _ in range(n):
        started = time.perf_counter()
        response = client.post(path, json=payload)
        assert response.status_code == 200, response.text
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


@pytest.mark.slow
def test_predict_meets_its_latency_budget() -> None:
    """NFR-5: < 100 ms warm. Measured 2.3 ms median on the dev machine."""
    with _real_client() as client:
        if not client.get("/health").json()["models"]["sentiment"]:
            pytest.skip("sentiment model not loaded")
        median = _warm_median_ms(client, "/predict", {"text": "the flight was not great"})
    assert median < PREDICT_BUDGET_MS, f"/predict median {median:.1f} ms"


@pytest.mark.slow
def test_generate_meets_its_latency_budget() -> None:
    """NFR-5: < 2 s for 40 words. Measured 57.4 ms median on the dev machine.

    The budget is asserted, not the measurement. A test that pinned 57 ms would
    fail on any slower machine while telling nobody anything true.
    """
    with _real_client() as client:
        if not client.get("/health").json()["models"]["textgen"]:
            pytest.skip("textgen model not loaded")
        median = _warm_median_ms(
            client,
            "/generate",
            {"seed": "alice was beginning to", "n_words": 40, "temperature": 0.7},
            n=5,
        )
    assert median < GENERATE_BUDGET_MS, f"/generate median {median:.1f} ms"


# --------------------------------------------------------------------------- #
# a truncated run must never be served
# --------------------------------------------------------------------------- #


def test_a_smoke_run_is_not_served(
    tiny_runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --max-steps run must not become the model the API serves.

    Found by running the documented smoke command (Rules.md B7) during the
    Phase 8 review: because resolution picks the newest run, the half-trained
    model silently replaced the real one -- macro-F1 0.6997 against 0.8485,
    with nothing anywhere saying so.

    The smoke checkpoint here is *newer* than the real one, so a resolver that
    ignored the marker would pick it and this test would fail.
    """
    import shutil

    root = tmp_path / "runs"
    shutil.copytree(tiny_runs, root)

    real = root / "sentiment" / "20260101T000000" / "best.pt"
    smoke_dir = root / "sentiment" / "29991231T235959"     # sorts last
    smoke_dir.mkdir(parents=True)
    payload = torch.load(real, map_location="cpu", weights_only=False)
    payload["train"] = {**payload.get("train", {}), "max_steps": 3}
    torch.save(payload, smoke_dir / "best.pt")

    monkeypatch.setenv(app_module.ENV_RUNS_DIR, str(root))
    resolved = app_module.latest_checkpoint("sentiment")

    assert resolved is not None, "the real run must still be found"
    assert resolved.parent.name == "20260101T000000", (
        f"resolution picked {resolved.parent.name}; a --max-steps run must be skipped"
    )


def test_the_smoke_marker_is_what_makes_the_difference(
    tiny_runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: without the marker, the newest run *is* chosen.

    Otherwise the test above would pass even if resolution simply always
    returned the oldest run, which would be a different bug.
    """
    import shutil

    root = tmp_path / "runs"
    shutil.copytree(tiny_runs, root)
    newer = root / "sentiment" / "29991231T235959"
    newer.mkdir(parents=True)
    shutil.copy(root / "sentiment" / "20260101T000000" / "best.pt", newer / "best.pt")

    monkeypatch.setenv(app_module.ENV_RUNS_DIR, str(root))
    assert app_module.latest_checkpoint("sentiment").parent.name == "29991231T235959"


def test_an_explicit_checkpoint_is_still_honoured(
    tiny_runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming a file is a decision; the guard only stops an accident."""
    explicit = tiny_runs / "sentiment" / "20260101T000000" / "best.pt"
    monkeypatch.setenv(app_module.ENV_SENTIMENT_CKPT, str(explicit))
    assert app_module.resolve_checkpoint("sentiment", app_module.ENV_SENTIMENT_CKPT) == explicit
