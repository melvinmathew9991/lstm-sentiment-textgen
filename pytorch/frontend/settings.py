"""Frontend configuration, read from the environment.

The backend URL is never a literal in page code (FR-36). A hardcoded
``localhost`` is the reason so many demos work only on the machine that built
them, and it is the same class of defect as ``input_words[-28701]`` (D10): a
value that matters, buried where nobody can change it.

Nothing here imports from ``lstm_nlp``. The frontend's only contract with the
rest of the system is the HTTP surface in ``Architecture.md`` section 6
(Rules.md C15).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Environment variable naming the backend.
ENV_API_URL = "LSTM_API_URL"

#: Default backend, matching the port ``Phases.md`` tells you to start.
DEFAULT_API_URL = "http://127.0.0.1:8000"

#: Seconds to wait on a fast route (`/health`, `/models`, `/predict`).
#:
#: Warm ``/predict`` is measured at 2.3 ms, so five seconds is not a latency
#: budget -- it is the point at which "slow" becomes "broken" and the user
#: deserves to be told rather than left watching a spinner (FR-35).
FAST_TIMEOUT_S = 5.0

#: Seconds to wait on generation, which is sequential and therefore slower.
#:
#: 40 words is measured at 57 ms; 200 words is the schema's maximum. Thirty
#: seconds is generous enough that a real answer is never cut off, and short
#: enough that a dead backend is reported inside one attention span.
GENERATE_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Settings:
    """Where the backend is and how long to wait for it."""

    api_url: str = DEFAULT_API_URL
    fast_timeout_s: float = FAST_TIMEOUT_S
    generate_timeout_s: float = GENERATE_TIMEOUT_S

    @property
    def start_command(self) -> str:
        """The exact command that starts the backend this app expects.

        Shown verbatim in the unreachable state (FR-35). An error that names
        the fix is worth more than one that names the failure.
        """
        port = self.api_url.rsplit(":", 1)[-1].split("/")[0]
        return f"uvicorn lstm_nlp.api.app:app --port {port}"


def load_settings() -> Settings:
    """Read settings from the environment, falling back to the defaults."""
    return Settings(api_url=os.environ.get(ENV_API_URL, DEFAULT_API_URL).rstrip("/"))
