"""Semantic enrichment data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SemanticHints:
    """Compact semantic hints extracted from a file signal."""

    kind_hints: list[str] = field(default_factory=list)
    topic_hints: list[str] = field(default_factory=list)
    entity_candidates: list[str] = field(default_factory=list)
    time_hints: list[str] = field(default_factory=list)
    structure_hints: list[str] = field(default_factory=list)
    language_hint: str | None = None
    functional_summary: str | None = None
    evidence_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind_hints": list(self.kind_hints),
            "topic_hints": list(self.topic_hints),
            "entity_candidates": list(self.entity_candidates),
            "time_hints": list(self.time_hints),
            "structure_hints": list(self.structure_hints),
            "evidence_hints": list(self.evidence_hints),
        }
        if self.language_hint:
            payload["language_hint"] = self.language_hint
        if self.functional_summary:
            payload["functional_summary"] = self.functional_summary
        return payload


@dataclass(slots=True)
class EnrichedSignal:
    """Signal enriched with compact semantic hints."""

    modality: str
    semantic_hints: SemanticHints
    metadata: dict[str, Any]
    confidence: float
