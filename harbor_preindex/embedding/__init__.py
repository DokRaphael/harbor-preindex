"""Embedding backends."""

from harbor_preindex.embedding.base import EmbeddingBackend
from harbor_preindex.embedding.ollama import OllamaEmbeddingBackend

__all__ = ["EmbeddingBackend", "OllamaEmbeddingBackend"]
