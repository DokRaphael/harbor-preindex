"""Ollama embedding backend."""

from __future__ import annotations

from collections.abc import Sequence

from harbor_preindex.embedding.base import EmbeddingBackend
from harbor_preindex.utils.ollama_api import OllamaApiClient


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Embedding backend backed by Ollama HTTP."""

    def __init__(self, client: OllamaApiClient, model: str) -> None:
        self.client = client
        self.model = model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.client.embed(self.model, texts)
