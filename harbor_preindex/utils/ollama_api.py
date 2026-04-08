"""Shared Ollama HTTP client."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from harbor_preindex.logging_config import get_logger

logger = get_logger(__name__)


class OllamaApiError(RuntimeError):
    """Raised when the Ollama API request fails."""


class OllamaApiClient:
    """Small wrapper over the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        api_key: str | None = None,
        max_retries: int = 0,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def get_version(self) -> str:
        payload = self._request("GET", "/api/version")
        version = payload.get("version")
        if not isinstance(version, str):
            raise OllamaApiError("missing Ollama version in response")
        return version

    def list_models(self) -> list[str]:
        payload = self._request("GET", "/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            return []
        return [str(item.get("name", "")) for item in models if isinstance(item, dict)]

    def embed(self, model: str, inputs: Sequence[str]) -> list[list[float]]:
        if not model.strip():
            raise OllamaApiError("embedding model is not configured")
        payload = self._request(
            "POST",
            "/api/embed",
            json_payload={"model": model, "input": list(inputs)},
        )
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise OllamaApiError("invalid embeddings response from Ollama")
        parsed_embeddings = [
            [float(value) for value in vector]
            for vector in embeddings
            if isinstance(vector, list)
        ]
        if len(parsed_embeddings) != len(inputs):
            raise OllamaApiError("embedding response size does not match request size")
        return parsed_embeddings

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        if not model.strip():
            raise OllamaApiError("llm model is not configured")

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_schema is not None:
            payload["format"] = json_schema

        response = self._request("POST", "/api/generate", json_payload=payload)
        content = response.get("response")
        if not isinstance(content, str) or not content.strip():
            raise OllamaApiError("empty generation response from Ollama")
        return content

    def _request(
        self,
        method: str,
        path: str,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.request(
                        method,
                        f"{self.base_url}{path}",
                        json=json_payload,
                        headers=headers,
                    )
                    response.raise_for_status()

                data = response.json()
                if not isinstance(data, dict):
                    raise OllamaApiError("unexpected Ollama response payload")
                return data
            except (httpx.HTTPError, ValueError, OllamaApiError) as exc:
                last_error = exc
                if attempt >= total_attempts:
                    break

                backoff_seconds = self.retry_backoff_seconds * attempt
                logger.warning(
                    "ollama_request_retry",
                    extra={
                        "path": path,
                        "attempt": attempt,
                        "total_attempts": total_attempts,
                        "backoff_seconds": round(backoff_seconds, 2),
                        "error": str(exc),
                    },
                )
                time.sleep(backoff_seconds)

        raise OllamaApiError(
            f"Ollama request failed after {total_attempts} attempt(s): {last_error}"
        ) from last_error
