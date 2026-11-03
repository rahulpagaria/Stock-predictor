from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AgentApiError(RuntimeError):
    """Raised when an external agent API request or response is invalid."""


class AgentApiAdapter(Protocol):
    def send_prediction(
        self, prediction: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Send a prediction to an external agent service."""


class HttpAgentApiAdapter:
    def __init__(self, endpoint: str, timeout_seconds: float = 10.0) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Agent API endpoint must use http:// or https://.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Agent API timeout must be a finite, positive value.")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def send_prediction(
        self, prediction: Mapping[str, object]
    ) -> Mapping[str, object]:
        body = json.dumps(dict(prediction)).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read()
        except HTTPError as error:
            raise AgentApiError(
                f"Agent API returned HTTP {error.code} for {self.endpoint}."
            ) from error
        except URLError as error:
            raise AgentApiError(
                f"Could not reach agent API at {self.endpoint}: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise AgentApiError(
                f"Agent API request timed out after {self.timeout_seconds:g} seconds."
            ) from error

        try:
            decoded = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AgentApiError("Agent API response must be valid UTF-8 JSON.") from error
        if not isinstance(decoded, dict):
            raise AgentApiError("Agent API response must be a JSON object.")
        return decoded
