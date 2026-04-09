"""Folder semantic signature alignment and lightweight reranking."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import re
import unicodedata
from collections.abc import Sequence

from harbor_preindex.schemas import SearchCandidate, StructuredQueryHints

_STOPWORDS = {
    "a",
    "an",
    "and",
    "de",
    "des",
    "document",
    "documents",
    "du",
    "et",
    "file",
    "files",
    "find",
    "folder",
    "for",
    "from",
    "in",
    "invoice",
    "invoices",
    "is",
    "la",
    "le",
    "les",
    "my",
    "of",
    "path",
    "project",
    "projects",
    "query",
    "the",
    "where",
}


@dataclass(slots=True, frozen=True)
class FolderSignatureAlignment:
    """Compact alignment summary between query hints and a folder signature."""

    matched_kind_hints: list[str]
    matched_topic_hints: list[str]
    matched_entity_candidates: list[str]
    matched_time_hints: list[str]
    matched_discriminative_terms: list[str]
    matched_representative_terms: list[str]
    matched_technical_hints: list[str]
    notes: list[str]
    bonus: float


def rerank_folder_candidates(
    query_text: str,
    query_hints: StructuredQueryHints,
    candidates: Sequence[SearchCandidate],
) -> list[SearchCandidate]:
    """Apply a small semantic-signature rerank to folder candidates."""

    reranked: list[SearchCandidate] = []
    for candidate in candidates:
        alignment = folder_signature_alignment(query_text, query_hints, candidate)
        raw_score = candidate.raw_score if candidate.raw_score is not None else candidate.score
        reranked.append(
            replace(
                candidate,
                raw_score=raw_score,
                semantic_bonus=alignment.bonus,
                score=round(max(0.0, min(0.9999, raw_score + alignment.bonus)), 4),
            )
        )
    reranked.sort(key=lambda item: (item.score, item.project_id), reverse=True)
    return reranked


def folder_signature_alignment(
    query_text: str,
    query_hints: StructuredQueryHints,
    candidate: SearchCandidate,
) -> FolderSignatureAlignment:
    """Return the compact semantic alignment between a query and a folder signature."""

    signature = candidate.semantic_signature
    if signature is None:
        return FolderSignatureAlignment(
            matched_kind_hints=[],
            matched_topic_hints=[],
            matched_entity_candidates=[],
            matched_time_hints=[],
            matched_discriminative_terms=[],
            matched_representative_terms=[],
            matched_technical_hints=[],
            notes=[],
            bonus=0.0,
        )

    query_tokens = _meaningful_tokens(query_text)
    kind_matches = _matched_hint_values(query_hints.kind_hints, signature.dominant_kinds)
    topic_matches = _matched_hint_terms(
        query_hints.topic_hints,
        [
            *signature.dominant_topics,
            *signature.representative_terms,
            *signature.discriminative_terms,
        ],
    )
    technical_matches = _matched_hint_terms(
        query_hints.technical_hints,
        [
            *signature.dominant_topics,
            *signature.discriminative_terms,
            *signature.representative_terms,
        ],
    )
    entity_matches = _matched_hint_terms(
        query_hints.entity_terms,
        [
            *signature.dominant_entities,
            *signature.representative_terms,
        ],
    )
    time_matches = _matched_time_hints(
        query_hints.time_hints,
        [
            *signature.dominant_time_hints,
            *signature.sample_filenames,
        ],
    )
    discriminative_terms = _matched_query_terms(
        query_tokens,
        " ".join(signature.discriminative_terms),
    )
    representative_terms = _matched_query_terms(
        query_tokens,
        " ".join(signature.representative_terms),
    )

    semantic_family = bool(kind_matches or topic_matches)
    technical_family = bool(technical_matches)
    entity_family = bool(entity_matches)
    time_family = bool(time_matches)
    discriminative_family = bool(discriminative_terms)
    coverage = sum(
        (
            semantic_family,
            technical_family,
            entity_family,
            time_family,
            discriminative_family,
        )
    )

    bonus = 0.0
    notes: list[str] = []
    if semantic_family:
        bonus += 0.014
    if technical_family:
        bonus += 0.02
    if entity_family:
        bonus += 0.016
    if discriminative_family:
        bonus += 0.02
    if representative_terms and not discriminative_terms:
        bonus += 0.006

    if time_family:
        if coverage <= 1:
            bonus += 0.003
            notes.append("time hint matched in folder semantics with limited weight")
        else:
            bonus += 0.01

    if coverage >= 2:
        bonus += 0.012
        notes.append("folder semantic signature matched across multiple hint families")
    if coverage >= 3:
        bonus += 0.008

    if signature.folder_role == "leaf_specialized" and (
        technical_family or discriminative_family or semantic_family
    ):
        bonus += 0.01
        notes.append("specialized folder role reinforced the semantic match")

    if signature.folder_role in {"container", "time_bucket"} and coverage <= 1 and not technical_family:
        bonus -= 0.012
        notes.append("generic folder role kept the semantic bonus conservative")

    if signature.folder_role == "project_root" and technical_family:
        bonus += 0.008

    if coverage == 0:
        notes = []
        bonus = 0.0

    return FolderSignatureAlignment(
        matched_kind_hints=kind_matches,
        matched_topic_hints=topic_matches,
        matched_entity_candidates=entity_matches,
        matched_time_hints=time_matches,
        matched_discriminative_terms=discriminative_terms,
        matched_representative_terms=representative_terms,
        matched_technical_hints=technical_matches,
        notes=_compact_list(notes, limit=3),
        bonus=round(max(-0.03, min(0.06, bonus)), 4),
    )


def _meaningful_tokens(value: str) -> list[str]:
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    )
    tokens = re.findall(r"[a-z0-9]+", normalized)
    filtered = [token for token in tokens if token not in _STOPWORDS]
    return filtered or tokens


def _normalized_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _matched_query_terms(query_tokens: Sequence[str], value: str) -> list[str]:
    value_tokens = set(_meaningful_tokens(value))
    return _compact_list([token for token in query_tokens if token in value_tokens], limit=6)


def _matched_hint_values(hints: Sequence[str], values: Sequence[str]) -> list[str]:
    normalized_values = [_normalized_value(value) for value in values if str(value).strip()]
    matches: list[str] = []
    for hint in hints:
        normalized_hint = _normalized_value(hint)
        if not normalized_hint:
            continue
        if any(
            normalized_hint == value
            or normalized_hint in value
            or value in normalized_hint
            for value in normalized_values
            if value
        ):
            matches.append(hint)
    return _compact_list(matches, limit=4)


def _matched_hint_terms(hints: Sequence[str], values: Sequence[str]) -> list[str]:
    normalized_values = [_normalized_value(value) for value in values if str(value).strip()]
    matches: list[str] = []
    for hint in hints:
        normalized_hint = _normalized_value(hint)
        if not normalized_hint:
            continue
        hint_tokens = set(_meaningful_tokens(normalized_hint))
        if any(
            normalized_hint in value
            or value in normalized_hint
            or hint_tokens & set(_meaningful_tokens(value))
            for value in normalized_values
            if value
        ):
            matches.append(hint)
    return _compact_list(matches, limit=4)


def _matched_time_hints(hints: Sequence[str], values: Sequence[str]) -> list[str]:
    normalized_values = [_normalized_value(value) for value in values if str(value).strip()]
    matches: list[str] = []
    for hint in hints:
        normalized_hint = _normalized_value(hint)
        if not normalized_hint:
            continue
        comparable_hint = normalized_hint.split(":", 1)[-1] if ":" in normalized_hint else normalized_hint
        if any(comparable_hint in value or value == comparable_hint for value in normalized_values):
            matches.append(hint)
    return _compact_list(matches, limit=4)


def _compact_list(values: Sequence[str], limit: int) -> list[str]:
    compacted: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
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
