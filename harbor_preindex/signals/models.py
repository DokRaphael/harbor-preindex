"""Signal data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExtractedSignal:
    """Normalized signal extracted from a file."""

    modality: str
    text_for_embedding: str
    metadata: dict[str, Any]
    confidence: float
