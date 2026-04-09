"""Semantic enrichment for text documents."""

from __future__ import annotations

import re
from pathlib import Path

from harbor_preindex.semantic.base import SemanticEnricher
from harbor_preindex.semantic.models import EnrichedSignal, SemanticHints
from harbor_preindex.signals.models import ExtractedSignal
from harbor_preindex.utils.text import normalize_text

_STOPWORDS = {
    "a",
    "an",
    "and",
    "aux",
    "avec",
    "dans",
    "de",
    "des",
    "du",
    "for",
    "from",
    "into",
    "la",
    "le",
    "les",
    "of",
    "ou",
    "pour",
    "sur",
    "the",
    "this",
    "une",
    "with",
}


class DocumentSemanticEnricher(SemanticEnricher):
    """Build compact semantic hints for general text documents."""

    def supports(self, file_path: Path, signal: ExtractedSignal) -> bool:
        return signal.modality == "document"

    def enrich(self, file_path: Path, signal: ExtractedSignal) -> EnrichedSignal:
        excerpt = normalize_text(str(signal.metadata.get("text_excerpt", "")))
        lines = [line.strip() for line in excerpt.splitlines() if line.strip()]

        entity_candidates = _entity_candidates(file_path=file_path, lines=lines)
        time_hints = _time_hints(excerpt)
        structure_hints = _structure_hints(lines, excerpt)
        topic_hints = _topic_hints(file_path=file_path, excerpt=excerpt)
        kind_hints = _kind_hints(
            time_hints=time_hints,
            structure_hints=structure_hints,
            topic_hints=topic_hints,
        )
        evidence_hints = _evidence_hints(file_path=file_path, lines=lines)
        functional_summary = _functional_summary(
            kind_hints=kind_hints,
            structure_hints=structure_hints,
            entity_candidates=entity_candidates,
            time_hints=time_hints,
        )

        hints = SemanticHints(
            kind_hints=kind_hints,
            topic_hints=topic_hints,
            entity_candidates=entity_candidates,
            time_hints=time_hints,
            structure_hints=structure_hints,
            functional_summary=functional_summary,
            evidence_hints=evidence_hints,
        )
        metadata = {
            **signal.metadata,
            "semantic_hints": hints.to_dict(),
        }
        return EnrichedSignal(
            modality=signal.modality,
            semantic_hints=hints,
            metadata=metadata,
            confidence=signal.confidence,
        )


def _topic_hints(file_path: Path, excerpt: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", f"{file_path.stem} {excerpt}")
    ranked: dict[str, int] = {}
    for token in tokens:
        lowered = token.lower()
        if lowered in _STOPWORDS:
            continue
        ranked[lowered] = ranked.get(lowered, 0) + 1
    sorted_tokens = sorted(ranked.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _count in sorted_tokens[:6]]


def _entity_candidates(file_path: Path, lines: list[str]) -> list[str]:
    candidates = _title_like_tokens(file_path.stem)
    for line in lines[:6]:
        candidates.extend(_title_like_tokens(line))
    return _compact_list(candidates, limit=5)


def _time_hints(excerpt: str) -> list[str]:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
        r"\b(?:19|20)\d{2}\b",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, excerpt))
    return _compact_list(matches, limit=4)


def _structure_hints(lines: list[str], excerpt: str) -> list[str]:
    hints: list[str] = []
    if re.search(
        r"(?:[$€£¥]\s?\d)|(?:\b\d+(?:[.,]\d{2})?\s?(?:EUR|USD|GBP|CHF|JPY)\b)",
        excerpt,
    ):
        hints.append("contains_monetary_values")
    if re.search(r"\b[A-Z0-9]{6,}(?:[-_/][A-Z0-9]{2,})+\b", excerpt):
        hints.append("contains_identifiers")
    if any(line.startswith(("#", "##")) or line.endswith(":") for line in lines[:8]):
        hints.append("sectioned_content")
    if any(re.match(r"^(?:[-*]|\d+\.)\s+", line) for line in lines[:8]):
        hints.append("list_structure")
    if excerpt.count("|") >= 2 or excerpt.count("\t") >= 2:
        hints.append("tabular_layout")
    if _numeric_density(excerpt) >= 0.12:
        hints.append("numeric_dense")
    return _compact_list(hints, limit=6)


def _kind_hints(
    time_hints: list[str],
    structure_hints: list[str],
    topic_hints: list[str],
) -> list[str]:
    hints: list[str] = []
    if "contains_monetary_values" in structure_hints and time_hints:
        hints.append("transactional_document")
    if "sectioned_content" in structure_hints or "tabular_layout" in structure_hints:
        hints.append("structured_document")
    if any(
        topic in {"api", "config", "module", "query", "retrieval", "storage"}
        for topic in topic_hints
    ):
        hints.append("technical_document")
    if not hints:
        hints.append("general_document")
    return _compact_list(hints, limit=4)


def _functional_summary(
    kind_hints: list[str],
    structure_hints: list[str],
    entity_candidates: list[str],
    time_hints: list[str],
) -> str:
    base = kind_hints[0].replace("_", " ").capitalize()
    details: list[str] = []
    if structure_hints:
        details.append(", ".join(hint.replace("_", " ") for hint in structure_hints[:3]))
    if entity_candidates:
        details.append("named entities such as " + ", ".join(entity_candidates[:2]))
    if time_hints:
        details.append("time hints such as " + ", ".join(time_hints[:2]))
    if details:
        return base + " with " + "; ".join(details) + "."
    return base + "."


def _evidence_hints(file_path: Path, lines: list[str]) -> list[str]:
    values = [file_path.name]
    values.extend(line for line in lines[:3] if len(line) <= 100)
    return _compact_list(values, limit=4)


def _title_like_tokens(value: str) -> list[str]:
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:[ _-][A-Z][A-Za-z0-9]+)*\b", value)
    return [token.replace("_", " ").replace("-", " ") for token in tokens]


def _numeric_density(excerpt: str) -> float:
    if not excerpt:
        return 0.0
    numeric_count = sum(1 for char in excerpt if char.isdigit())
    return numeric_count / max(len(excerpt), 1)


def _compact_list(values: list[str | None], limit: int) -> list[str]:
    seen: set[str] = set()
    compacted: list[str] = []
    for value in values:
        if value is None:
            continue
        cleaned = str(value).strip().strip(",;:")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        compacted.append(cleaned)
        if len(compacted) >= limit:
            break
    return compacted
