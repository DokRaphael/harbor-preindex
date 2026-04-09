"""Semantic enrichment base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from harbor_preindex.semantic.models import EnrichedSignal
from harbor_preindex.signals.models import ExtractedSignal


class SemanticEnricher(ABC):
    """Base interface for enriching extracted signals."""

    @abstractmethod
    def supports(self, file_path: Path, signal: ExtractedSignal) -> bool:
        """Return whether this enricher can handle the file and signal."""

    @abstractmethod
    def enrich(self, file_path: Path, signal: ExtractedSignal) -> EnrichedSignal:
        """Return an enriched signal."""
