"""Lightweight folder semantic signature construction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import unicodedata

from harbor_preindex.semantic import SemanticEnricherRegistry
from harbor_preindex.schemas import DiscoveredProject, FolderSemanticSignature
from harbor_preindex.signals.models import ExtractedSignal
from harbor_preindex.utils.text import normalize_text

_CODE_LIKE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".html",
    ".css",
    ".xml",
    ".ini",
    ".conf",
}
_GENERIC_TERMS = {
    "admin",
    "archive",
    "config",
    "configs",
    "data",
    "doc",
    "docs",
    "document",
    "documents",
    "file",
    "files",
    "folder",
    "general",
    "incoming",
    "misc",
    "note",
    "notes",
    "other",
    "pdf",
    "project",
    "projects",
    "sample",
    "samples",
    "storage",
    "structured",
    "text",
}


class FolderSemanticSignatureBuilder:
    """Build compact semantic signatures for indexed folders."""

    def __init__(
        self,
        semantic_registry: SemanticEnricherRegistry,
        max_profile_chars: int,
    ) -> None:
        self.semantic_registry = semantic_registry
        self.max_profile_chars = max_profile_chars

    def build(self, project: DiscoveredProject, excerpts_by_path: dict[Path, str]) -> FolderSemanticSignature:
        sample_filenames = [sample.name for sample in project.sample_files]
        extension_counts: Counter[str] = Counter()
        topic_counts: Counter[str] = Counter()
        entity_counts: Counter[str] = Counter()
        time_counts: Counter[str] = Counter()
        kind_counts: Counter[str] = Counter()
        representative_counts: Counter[str] = Counter()
        discriminative_counts: Counter[str] = Counter()
        child_counts: Counter[str] = Counter()

        path_terms = set(_normalized_tokens(project.relative_path.as_posix()))

        for sample_path in project.sample_files:
            extension_counts[sample_path.suffix.lower() or "[no extension]"] += 1
            relative_sample = sample_path.relative_to(project.path)
            if len(relative_sample.parts) > 1:
                child_counts[relative_sample.parts[0]] += 1

            excerpt = excerpts_by_path.get(sample_path, "")
            enriched = self.semantic_registry.enrich(sample_path, _sample_signal(sample_path, excerpt))
            hints = enriched.semantic_hints
            excerpt_terms = _excerpt_terms(excerpt)

            _add_weighted(topic_counts, hints.topic_hints, weight=3)
            _add_weighted(entity_counts, hints.entity_candidates, weight=3)
            _add_weighted(time_counts, hints.time_hints, weight=2)
            _add_weighted(kind_counts, hints.kind_hints, weight=2)

            representative_values = [
                *hints.topic_hints,
                *hints.entity_candidates,
                *hints.time_hints,
                *_semantic_filename_terms(sample_path.stem),
                *excerpt_terms,
                *_summary_terms(hints.functional_summary),
            ]
            discriminative_values = [
                *hints.topic_hints,
                *_string_list(enriched.metadata.get("imports")),
                *_string_list(enriched.metadata.get("symbols")),
                *_semantic_filename_terms(sample_path.stem),
            ]

            representative_counts.update(
                term for term in representative_values if _keep_term(term, path_terms)
            )
            discriminative_counts.update(
                term for term in discriminative_values if _keep_term(term, path_terms)
            )
            _add_weighted(
                discriminative_counts,
                [term for term in excerpt_terms if _keep_term(term, path_terms)],
                weight=2,
            )

        dominant_topics = _top_terms(topic_counts, limit=6)
        dominant_entities = _top_terms(entity_counts, limit=4)
        dominant_time_hints = _top_terms(time_counts, limit=4)
        dominant_kinds = _top_terms(kind_counts, limit=4)
        frequent_extensions = _top_terms(extension_counts, limit=4)
        representative_terms = _top_terms(representative_counts, limit=8)
        discriminative_terms = _top_terms(discriminative_counts, limit=8)
        notable_children = _top_terms(child_counts, limit=4)
        folder_role = _folder_role(
            folder_name=project.path.name,
            dominant_entities=dominant_entities,
            dominant_time_hints=dominant_time_hints,
            dominant_kinds=dominant_kinds,
            frequent_extensions=frequent_extensions,
            discriminative_terms=discriminative_terms,
            notable_children=notable_children,
        )

        return FolderSemanticSignature(
            folder_role=folder_role,
            dominant_topics=dominant_topics,
            dominant_entities=dominant_entities,
            dominant_time_hints=dominant_time_hints,
            dominant_kinds=dominant_kinds,
            frequent_extensions=frequent_extensions,
            representative_terms=representative_terms,
            discriminative_terms=discriminative_terms,
            notable_children=notable_children,
            sample_filenames=sample_filenames[:5],
        )


def signature_text_lines(signature: FolderSemanticSignature) -> list[str]:
    """Render a compact text representation for embedding and debug."""

    lines = [f"Folder role: {signature.folder_role}"]
    if signature.dominant_kinds:
        lines.append("Dominant kinds: " + ", ".join(signature.dominant_kinds))
    if signature.dominant_topics:
        lines.append("Dominant topics: " + ", ".join(signature.dominant_topics))
    if signature.dominant_entities:
        lines.append("Dominant entities: " + ", ".join(signature.dominant_entities))
    if signature.dominant_time_hints:
        lines.append("Dominant time hints: " + ", ".join(signature.dominant_time_hints))
    if signature.frequent_extensions:
        lines.append("Frequent extensions: " + ", ".join(signature.frequent_extensions))
    if signature.discriminative_terms:
        lines.append("Discriminative terms: " + ", ".join(signature.discriminative_terms))
    if signature.representative_terms:
        lines.append("Representative terms: " + ", ".join(signature.representative_terms))
    if signature.notable_children:
        lines.append("Notable children: " + ", ".join(signature.notable_children))
    return lines


def _sample_signal(file_path: Path, excerpt: str) -> ExtractedSignal:
    cleaned_excerpt = normalize_text(excerpt)
    parts = [
        f"Incoming file name: {file_path.name}",
        f"Incoming suffix: {file_path.suffix.lower() or 'unknown'}",
        f"Incoming parent folder: {file_path.parent.name}",
    ]
    if cleaned_excerpt:
        parts.append(f"Incoming text excerpt: {cleaned_excerpt}")
    return ExtractedSignal(
        modality="document",
        text_for_embedding="\n".join(parts),
        metadata={
            "input_file": str(file_path),
            "file_name": file_path.name,
            "suffix": file_path.suffix.lower(),
            "parent": file_path.parent.name,
            "text_excerpt": cleaned_excerpt,
        },
        confidence=0.95 if cleaned_excerpt else 0.6,
    )


def _folder_role(
    *,
    folder_name: str,
    dominant_entities: list[str],
    dominant_time_hints: list[str],
    dominant_kinds: list[str],
    frequent_extensions: list[str],
    discriminative_terms: list[str],
    notable_children: list[str],
) -> str:
    normalized_folder_name = _normalized_value(folder_name)
    if re.fullmatch(r"(?:19|20)\d{2}", normalized_folder_name):
        return "time_bucket"
    if dominant_time_hints and normalized_folder_name and any(
        normalized_folder_name in _normalized_value(value) for value in dominant_time_hints
    ):
        return "time_bucket"
    if dominant_entities and any(
        _normalized_value(entity) == normalized_folder_name for entity in dominant_entities
    ):
        return "entity_bucket"
    if notable_children and any(extension in _CODE_LIKE_EXTENSIONS for extension in frequent_extensions):
        return "project_root"
    if discriminative_terms and not notable_children:
        return "leaf_specialized"
    if len(notable_children) >= 2 and len(frequent_extensions) >= 2:
        return "container"
    if len(dominant_kinds) >= 2 or len(frequent_extensions) >= 3:
        return "mixed"
    if discriminative_terms:
        return "leaf_specialized"
    return "container"


def _normalized_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r"[A-Za-z0-9]{2,}", normalized.lower())
    return [token for token in tokens if token]


def _normalized_value(value: str) -> str:
    return " ".join(_normalized_tokens(value))


def _semantic_filename_terms(value: str) -> list[str]:
    prepared = re.sub(r"[_\-.]+", " ", value)
    prepared = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", prepared)
    prepared = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", prepared)
    return _normalized_tokens(prepared)


def _summary_terms(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        token
        for token in _normalized_tokens(value)
        if token not in _GENERIC_TERMS and len(token) >= 3
    ][:8]


def _excerpt_terms(value: str) -> list[str]:
    return [
        token
        for token in _normalized_tokens(value)
        if token not in _GENERIC_TERMS and len(token) >= 3
    ][:10]


def _keep_term(value: str, path_terms: set[str]) -> bool:
    normalized = _normalized_value(value)
    if not normalized:
        return False
    if normalized in _GENERIC_TERMS:
        return False
    if normalized.startswith("."):
        return True
    token_set = set(_normalized_tokens(normalized))
    if not token_set:
        return False
    return True


def _add_weighted(counter: Counter[str], values: list[str], weight: int) -> None:
    for value in values:
        normalized = _normalized_value(value)
        if not normalized:
            continue
        counter[normalized] += weight


def _top_terms(counter: Counter[str], limit: int) -> list[str]:
    return [term for term, _count in counter.most_common(limit)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
