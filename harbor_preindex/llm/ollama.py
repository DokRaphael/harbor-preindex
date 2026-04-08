"""Ollama LLM backend."""

from __future__ import annotations

from typing import Any

from harbor_preindex.llm.base import LLMBackend
from harbor_preindex.utils.ollama_api import OllamaApiClient


class OllamaLLMBackend(LLMBackend):
    """LLM backend backed by Ollama HTTP."""

    def __init__(self, client: OllamaApiClient, model: str) -> None:
        self.client = client
        self.model = model

    def generate_json(
        self,
        *,
        system_prompt: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> str:
        return self.client.generate(
            self.model,
            prompt,
            system_prompt=system_prompt,
            json_schema=schema,
        )
