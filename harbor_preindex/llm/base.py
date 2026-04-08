"""LLM backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMBackend(ABC):
    """Abstract LLM provider."""

    @abstractmethod
    def generate_json(
        self,
        *,
        system_prompt: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> str:
        """Generate a JSON response represented as text."""
