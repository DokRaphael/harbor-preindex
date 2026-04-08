"""Embedding backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingBackend(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""

        return self.embed_texts([text])[0]
