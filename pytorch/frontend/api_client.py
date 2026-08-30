"""The only module in the frontend that speaks HTTP.

Every network call in the app goes through here, which is what makes the error
states in ``Design.md`` section 7 possible: there is one place where a
connection can fail, so there is one place that decides what the user sees. A
page that called ``httpx`` directly would eventually grow its own idea of what
a timeout looks like.

Three failure kinds are distinguished, because the UI must respond to them
differently (FR-35):

* :class:`BackendUnreachable` -- nothing answered. The whole app is unusable;
  say so once, at the top, with the command that fixes it.
* :class:`ModelUnavailable` -- 503. One task is untrained; the other page still
  works, so this is a page-level notice, not a global one.
* :class:`ValidationFailed` -- 422. The user's input was rejected; the message
  belongs beside the control that produced it.

Collapsing these into one exception would force every page to re-derive the
distinction from a status code, which is how a "backend down" banner ends up
appearing because someone typed an empty string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from frontend.settings import Settings, load_settings


class BackendError(Exception):
    """Base class for every way the backend can fail this app."""


class BackendUnreachable(BackendError):
    """Nothing answered at the configured URL.

    Carries the URL and the start command so the UI never has to reconstruct
    them from settings it may not have loaded.
    """

    def __init__(self, url: str, start_command: str, cause: str) -> None:
        self.url = url
        self.start_command = start_command
        self.cause = cause
        super().__init__(f"backend unreachable at {url}: {cause}")


class ModelUnavailable(BackendError):
    """The service is up, but the model this route needs never loaded (503)."""

    def __init__(self, task: str, detail: str) -> None:
        self.task = task
        self.detail = detail
        super().__init__(detail)

    @property
    def train_command(self) -> str:
        """The command that produces the missing model."""
        return f"lstm-nlp train --config configs/{self.task}.yaml"


class ValidationFailed(BackendError):
    """The request was rejected by the contract (422).

    ``detail`` is flattened to a sentence here because the backend answers
    schema violations with a *list* of per-field errors, and a page should not
    have to know that shape to show a message.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


@dataclass
class ApiClient:
    """Typed wrappers over the routes in ``Architecture.md`` section 6.

    ``transport`` exists so tests can inject ``httpx.MockTransport`` and
    exercise this module's real response handling -- status codes, JSON
    decoding, error translation -- rather than a hand-rolled stand-in that
    would agree with whatever the code happened to do.
    """

    settings: Settings
    transport: httpx.BaseTransport | None = None

    def _client(self, timeout: float) -> httpx.Client:
        """An httpx client honouring the injected transport, if any."""
        return httpx.Client(transport=self.transport, timeout=timeout)

    # ----------------------------------------------------------------- #
    # transport
    # ----------------------------------------------------------------- #

    def _request(self, method: str, path: str, timeout: float, **kwargs: Any) -> dict:
        """Perform one call and translate every failure into a typed error.

        Args:
            method: HTTP verb.
            path: Route path, e.g. ``"/predict"``.
            timeout: Seconds to wait.
            **kwargs: Passed to ``httpx.request`` (typically ``json=``).

        Returns:
            The decoded JSON body.

        Raises:
            BackendUnreachable: Connection refused, DNS failure, or timeout.
            ModelUnavailable: The backend answered 503.
            ValidationFailed: The backend answered 422.
            BackendError: Any other non-2xx response.
        """
        url = f"{self.settings.api_url}{path}"
        try:
            with self._client(timeout) as client:
                response = client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise BackendUnreachable(
                self.settings.api_url,
                self.settings.start_command,
                f"no response within {timeout:.0f}s",
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendUnreachable(
                self.settings.api_url, self.settings.start_command, str(exc)
            ) from exc

        if response.status_code == 503:
            detail = _detail_text(response)
            raise ModelUnavailable(_task_from_detail(detail), detail)
        if response.status_code == 422:
            raise ValidationFailed(_detail_text(response))
        if response.status_code >= 400:
            raise BackendError(f"backend returned {response.status_code}: {_detail_text(response)}")

        return response.json()

    # ----------------------------------------------------------------- #
    # routes
    # ----------------------------------------------------------------- #

    def health(self) -> dict:
        """Liveness and per-task availability.

        A 503 here means *no* model loaded, which is a whole-app condition
        rather than a page one -- so it is returned as a body, not raised.
        """
        url = f"{self.settings.api_url}/health"
        try:
            with self._client(self.settings.fast_timeout_s) as client:
                response = client.get(url)
        except httpx.HTTPError as exc:
            raise BackendUnreachable(
                self.settings.api_url, self.settings.start_command, str(exc)
            ) from exc
        return response.json()

    def models(self) -> list[dict]:
        """Describe every loaded checkpoint, for the sidebar metadata."""
        return self._request("GET", "/models", self.settings.fast_timeout_s)["models"]

    def predict(self, text: str) -> dict:
        """Classify one string."""
        return self._request(
            "POST", "/predict", self.settings.fast_timeout_s, json={"text": text}
        )

    def generate(
        self,
        seed: str,
        n_words: int,
        temperature: float,
        top_k: int | None = None,
        rng_seed: int | None = None,
    ) -> dict:
        """Continue a seed under the given sampling settings."""
        payload: dict[str, Any] = {
            "seed": seed,
            "n_words": n_words,
            "temperature": temperature,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        if rng_seed is not None:
            payload["rng_seed"] = rng_seed
        return self._request("POST", "/generate", self.settings.generate_timeout_s, json=payload)

    def distribution(
        self, seed: str, temperature: float, top_k: int | None = None, n: int = 12
    ) -> dict:
        """The next-word distribution the sampler would draw from.

        This is what the temperature chart renders (FR-34). The frontend asks
        the backend for it rather than deriving it, because a chart computed
        beside the sampler instead of *from* it is free to disagree with it --
        which is precisely the defect D2 was.
        """
        payload: dict[str, Any] = {"seed": seed, "temperature": temperature, "n": n}
        if top_k is not None:
            payload["top_k"] = top_k
        return self._request(
            "POST", "/distribution", self.settings.fast_timeout_s, json=payload
        )


def _detail_text(response: httpx.Response) -> str:
    """Flatten a backend error body into one human sentence.

    The backend uses ``detail`` for both a plain string (500, 503) and a list
    of per-field errors (422 validation). Handling that split once, here, is
    the reason no page has to.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"

    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                field = ".".join(str(p) for p in item.get("loc", [])[1:]) or "request"
                parts.append(f"{field}: {item.get('msg', 'invalid')}")
            else:
                parts.append(str(item))
        return "; ".join(parts)
    return str(detail)


def _task_from_detail(detail: str) -> str:
    """Recover which task a 503 refers to, for the retraining hint.

    The backend phrases these as ``"<task> model not loaded; train it first"``,
    so the first word is the task. Defaults to ``sentiment`` rather than
    raising: a wrong hint in an error message is a smaller failure than an
    exception raised while reporting an exception.
    """
    first = detail.split(" ", 1)[0].strip().lower()
    return first if first in ("sentiment", "textgen") else "sentiment"


def get_client() -> ApiClient:
    """Build a client from the environment."""
    return ApiClient(load_settings())
