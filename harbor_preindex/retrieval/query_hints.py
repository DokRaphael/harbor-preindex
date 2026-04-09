"""Lightweight structured query hints for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
import unicodedata

from harbor_preindex.schemas import StructuredQueryHints

_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "de",
    "des",
    "document",
    "documents",
    "du",
    "est",
    "et",
    "file",
    "files",
    "find",
    "from",
    "in",
    "is",
    "la",
    "le",
    "les",
    "mes",
    "mon",
    "my",
    "of",
    "or",
    "ou",
    "pour",
    "the",
    "that",
    "to",
    "where",
}

_GENERIC_ENTITY_TERMS = {
    "backend",
    "bill",
    "cli",
    "code",
    "config",
    "contract",
    "doc",
    "docs",
    "documentation",
    "embed",
    "embedding",
    "extract",
    "extraction",
    "file",
    "folder",
    "guide",
    "invoice",
    "manual",
    "module",
    "parser",
    "pdf",
    "query",
    "readme",
    "receipt",
    "resume",
    "retrieval",
    "search",
    "spec",
    "storage",
    "test",
}

_TRANSACTIONAL_TERMS = {
    "bill",
    "billing",
    "expense",
    "expenses",
    "facture",
    "factures",
    "invoice",
    "invoices",
    "order",
    "orders",
    "payment",
    "payments",
    "receipt",
    "receipts",
}

_TECHNICAL_DOCUMENT_TERMS = {
    "doc",
    "docs",
    "documentation",
    "guide",
    "manual",
    "readme",
    "spec",
    "specs",
}

_CODE_LOOKUP_TERMS = {
    "backend",
    "cli",
    "code",
    "config",
    "configuration",
    "module",
    "parser",
    "script",
    "test",
    "tests",
}

_TECHNICAL_HINT_ALIASES = {
    "api": "api_client",
    "argparse": "cli",
    "cli": "cli",
    "client": "api_client",
    "config": "configuration",
    "configuration": "configuration",
    "embed": "embeddings",
    "embedding": "embeddings",
    "embeddings": "embeddings",
    "extract": "text_extraction",
    "extraction": "text_extraction",
    "http": "api_client",
    "ollama": "embeddings",
    "parser": "parsing",
    "parsing": "parsing",
    "pdf": "pdf_processing",
    "qdrant": "vector_storage",
    "query": "retrieval",
    "retrieval": "retrieval",
    "search": "retrieval",
    "settings": "configuration",
    "sqlite": "storage",
    "storage": "storage",
    "test": "tests",
    "tests": "tests",
    "vector": "vector_storage",
}

_MONTH_NAMES = {
    "apr": "month:april",
    "april": "month:april",
    "aug": "month:august",
    "august": "month:august",
    "dec": "month:december",
    "december": "month:december",
    "feb": "month:february",
    "february": "month:february",
    "jan": "month:january",
    "january": "month:january",
    "jul": "month:july",
    "july": "month:july",
    "jun": "month:june",
    "june": "month:june",
    "mar": "month:march",
    "march": "month:march",
    "may": "month:may",
    "nov": "month:november",
    "november": "month:november",
    "oct": "month:october",
    "october": "month:october",
    "sep": "month:september",
    "sept": "month:september",
    "september": "month:september",
}


@dataclass(slots=True)
class QueryHintExtractor:
    """Extract soft structured hints from a plain text query."""

    today: date | None = None

    def extract(self, query_text: str) -> StructuredQueryHints:
        normalized_terms = meaningful_query_terms(query_text)
        time_hints = _time_hints(query_text, normalized_terms, today=self.today or date.today())
        technical_hints = _technical_hints(normalized_terms)
        kind_hints = _kind_hints(normalized_terms, technical_hints)
        entity_terms = _entity_terms(normalized_terms, time_hints, technical_hints)
        topic_hints = _topic_hints(normalized_terms, entity_terms, time_hints)
        intent_hint = _intent_hint(kind_hints, technical_hints)

        return StructuredQueryHints(
            raw_query=query_text,
            normalized_terms=normalized_terms,
            kind_hints=kind_hints,
            entity_terms=entity_terms,
            time_hints=time_hints,
            topic_hints=topic_hints,
            technical_hints=technical_hints,
            intent_hint=intent_hint,
        )


def meaningful_query_terms(value: str) -> list[str]:
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    )
    tokens = re.findall(r"[a-z0-9]+", normalized)
    canonical = [_canonical_term(token) for token in tokens]
    filtered = [token for token in canonical if token not in _QUERY_STOPWORDS]
    return _compact_list(filtered or canonical, limit=12)


def _kind_hints(normalized_terms: list[str], technical_hints: list[str]) -> list[str]:
    hints: list[str] = []
    if any(term in _TRANSACTIONAL_TERMS for term in normalized_terms):
        hints.append("transactional_document")
    if any(term in _TECHNICAL_DOCUMENT_TERMS for term in normalized_terms):
        hints.append("technical_document")
    if technical_hints or any(term in _CODE_LOOKUP_TERMS for term in normalized_terms):
        hints.append("code_artifact")
    return _compact_list(hints, limit=4)


def _entity_terms(
    normalized_terms: list[str],
    time_hints: list[str],
    technical_hints: list[str],
) -> list[str]:
    excluded_terms = {
        *time_hints,
        *technical_hints,
        *[hint.split(":", 1)[-1] for hint in time_hints if ":" in hint],
    }
    entity_terms = [
        term
        for term in normalized_terms
        if len(term) >= 4
        and term not in _GENERIC_ENTITY_TERMS
        and term not in excluded_terms
        and term not in _TRANSACTIONAL_TERMS
        and term not in _TECHNICAL_DOCUMENT_TERMS
    ]
    return _compact_list(entity_terms, limit=4)


def _topic_hints(
    normalized_terms: list[str],
    entity_terms: list[str],
    time_hints: list[str],
) -> list[str]:
    excluded_terms = {
        *entity_terms,
        *time_hints,
        *[hint.split(":", 1)[-1] for hint in time_hints if ":" in hint],
    }
    topic_terms = [
        term
        for term in normalized_terms
        if term not in excluded_terms and len(term) >= 3
    ]
    return _compact_list(topic_terms, limit=6)


def _technical_hints(normalized_terms: list[str]) -> list[str]:
    hints: list[str] = []
    for term in normalized_terms:
        if term in _TECHNICAL_HINT_ALIASES:
            hints.append(_TECHNICAL_HINT_ALIASES[term])
    return _compact_list(hints, limit=6)


def _time_hints(query_text: str, normalized_terms: list[str], today: date) -> list[str]:
    hints: list[str] = []
    normalized_query = _normalize_value(query_text)

    explicit_years = re.findall(r"\b(?:19|20)\d{2}\b", normalized_query)
    hints.extend(explicit_years)

    if "last year" in normalized_query:
        hints.extend(["relative:last_year", str(today.year - 1)])
    if "this year" in normalized_query:
        hints.extend(["relative:this_year", str(today.year)])
    if any(term in {"recent", "latest", "new"} for term in normalized_terms):
        hints.append("recent")
    if any(term in {"old", "older", "archive", "archived"} for term in normalized_terms):
        hints.append("old")

    for term in normalized_terms:
        month_hint = _MONTH_NAMES.get(term)
        if month_hint:
            hints.append(month_hint)

    return _compact_list(hints, limit=6)


def _intent_hint(kind_hints: list[str], technical_hints: list[str]) -> str | None:
    if technical_hints or "code_artifact" in kind_hints:
        return "code_lookup"
    if kind_hints:
        return "document_lookup"
    return "semantic_lookup"


def _canonical_term(value: str) -> str:
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and len(value) > 4 and not value.endswith("ss"):
        return value[:-1]
    return value


def _normalize_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _compact_list(values: list[str | None], limit: int) -> list[str]:
    compacted: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
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
