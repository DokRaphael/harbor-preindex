"""Semantic enricher registry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from harbor_preindex.semantic.base import SemanticEnricher
from harbor_preindex.semantic.models import EnrichedSignal
from harbor_preindex.signals.models import ExtractedSignal


class SemanticEnricherRegistry:
    """Resolve and apply the appropriate semantic enricher."""

    def __init__(self, enrichers: Sequence[SemanticEnricher]) -> None:
        self.enrichers = list(enrichers)

    def resolve(self, file_path: Path, signal: ExtractedSignal) -> SemanticEnricher:
        for enricher in self.enrichers:
            if enricher.supports(file_path, signal):
                return enricher
        raise ValueError(f"no semantic enricher found for file: {file_path}")

    def enrich(self, file_path: Path, signal: ExtractedSignal) -> EnrichedSignal:
        return self.resolve(file_path, signal).enrich(file_path, signal)
