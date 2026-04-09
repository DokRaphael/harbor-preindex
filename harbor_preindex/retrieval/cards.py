"""Builders for retrieval cards."""

from __future__ import annotations

import re
from pathlib import Path

from harbor_preindex.semantic import SemanticEnricherRegistry
from harbor_preindex.semantic.models import SemanticHints
from harbor_preindex.schemas import FileCard, FolderCard, ProjectProfile, SearchCandidate
from harbor_preindex.signals.registry import SignalExtractorRegistry
from harbor_preindex.utils.fs import relative_display
from harbor_preindex.utils.qdrant_ids import make_file_point_id
from harbor_preindex.utils.text import truncate_text


class RetrievalCardBuilder:
    """Build file and folder cards from existing Harbor structures."""

    def __init__(
        self,
        root_path: Path,
        signal_registry: SignalExtractorRegistry,
        semantic_registry: SemanticEnricherRegistry,
        max_profile_chars: int,
    ) -> None:
        self.root_path = root_path
        self.signal_registry = signal_registry
        self.semantic_registry = semantic_registry
        self.max_profile_chars = max_profile_chars

    def build_folder_card(self, profile: ProjectProfile) -> FolderCard:
        """Build a folder card from an indexed project profile."""

        return FolderCard(
            folder_id=profile.project_id,
            path=profile.path,
            relative_path=profile.relative_path,
            name=profile.name,
            parent_path=str(Path(profile.path).parent) if profile.parent else None,
            text_for_embedding=profile.text_profile,
            metadata={
                "doc_count": profile.doc_count,
                "sample_filenames": list(profile.sample_filenames),
                "semantic_signature": (
                    profile.semantic_signature.to_dict()
                    if profile.semantic_signature is not None
                    else None
                ),
            },
        )

    def build_folder_card_from_candidate(self, candidate: SearchCandidate) -> FolderCard:
        """Build a folder card from a search candidate payload."""

        path = Path(candidate.path)
        return FolderCard(
            folder_id=candidate.project_id,
            path=candidate.path,
            relative_path=relative_display(path, self.root_path),
            name=candidate.name,
            parent_path=str(path.parent) if candidate.parent else None,
            text_for_embedding=candidate.text_profile,
            metadata={
                "doc_count": candidate.doc_count,
                "sample_filenames": list(candidate.sample_filenames),
                "semantic_signature": (
                    candidate.semantic_signature.to_dict()
                    if candidate.semantic_signature is not None
                    else None
                ),
            },
        )

    def build_file_card(self, file_path: Path) -> FileCard:
        """Build a file card from a supported file path."""

        extractor = self.signal_registry.resolve(file_path)
        signal = extractor.extract(file_path)
        enriched_signal = self.semantic_registry.enrich(file_path, signal)
        relative_parent = relative_display(file_path.parent, self.root_path)
        relative_path = relative_display(file_path, self.root_path)
        excerpt = str(signal.metadata.get("text_excerpt", "")).strip()
        readable_stem = _readable_stem(file_path.stem)
        filename_terms = _semantic_filename_terms(file_path.stem)
        semantic_hints = enriched_signal.semantic_hints
        parts = _file_card_parts(
            file_path=file_path,
            readable_stem=readable_stem,
            filename_terms=filename_terms,
            relative_parent=relative_parent,
            relative_path=relative_path,
            semantic_hints=semantic_hints,
        )

        return FileCard(
            file_id=make_file_point_id(str(file_path)),
            path=str(file_path),
            filename=file_path.name,
            extension=file_path.suffix.lower(),
            parent_path=str(file_path.parent),
            modality=enriched_signal.modality,
            text_for_embedding=truncate_text("\n".join(parts), self.max_profile_chars),
            metadata={
                "relative_path": relative_path,
                "relative_parent_path": relative_parent,
                "filename_terms": filename_terms,
                "text_excerpt": excerpt,
                "signal_confidence": signal.confidence,
                "semantic_hints": semantic_hints.to_dict(),
                "functional_summary": semantic_hints.functional_summary,
                "enriched_confidence": enriched_signal.confidence,
                "enriched_modality": enriched_signal.modality,
                **{
                    key: value
                    for key, value in enriched_signal.metadata.items()
                    if key not in {"input_file", "file_name", "suffix", "parent", "text_excerpt"}
                },
            },
        )


def _readable_stem(value: str) -> str:
    cleaned = re.sub(r"[_\-\.]+", " ", value).strip()
    return cleaned or value


def _semantic_filename_terms(value: str) -> list[str]:
    raw_chunks = [chunk for chunk in re.split(r"[_\-.]+", value.strip()) if chunk]
    terms: list[str] = []
    for chunk in raw_chunks:
        for segment in _camel_case_segments(chunk):
            terms.extend(_compound_filename_segments(segment))
    return _compact_terms(terms, limit=8)


def _camel_case_segments(value: str) -> list[str]:
    prepared = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    prepared = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", prepared)
    return [segment for segment in prepared.split() if segment]


def _compound_filename_segments(value: str) -> list[str]:
    if not value:
        return []

    alnum_suffix_match = re.match(r"^([A-Za-z]+(?:\d+)+)([A-Za-z]{3,})$", value)
    if alnum_suffix_match:
        return [alnum_suffix_match.group(1), alnum_suffix_match.group(2)]

    year_suffix_match = re.match(r"^([A-Za-z]{3,})(19\d{2}|20\d{2})$", value)
    if year_suffix_match:
        return [year_suffix_match.group(1), year_suffix_match.group(2)]

    year_prefix_match = re.match(r"^(19\d{2}|20\d{2})([A-Za-z]{3,})$", value)
    if year_prefix_match:
        return [year_prefix_match.group(1), year_prefix_match.group(2)]

    return [value]


def _compact_terms(values: list[str], limit: int) -> list[str]:
    compacted: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        if len(cleaned) < 2 and not cleaned.isdigit():
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        compacted.append(cleaned)
        if len(compacted) >= limit:
            break
    return compacted


def _file_card_parts(
    file_path: Path,
    readable_stem: str,
    filename_terms: list[str],
    relative_parent: str,
    relative_path: str,
    semantic_hints: SemanticHints,
) -> list[str]:
    parts = [
        f"File name: {file_path.name}",
        f"File stem: {readable_stem}",
        f"Extension: {file_path.suffix.lower() or 'unknown'}",
        f"Parent folder: {file_path.parent.name}",
        f"Parent path: {relative_parent}",
        f"Relative file path: {relative_path}",
    ]
    if filename_terms:
        parts.append("Filename terms: " + ", ".join(filename_terms))
    if semantic_hints.language_hint:
        parts.append(f"Language hint: {semantic_hints.language_hint}")
    if semantic_hints.kind_hints:
        parts.append("Kind hints: " + ", ".join(semantic_hints.kind_hints))
    if semantic_hints.topic_hints:
        parts.append("Topic hints: " + ", ".join(semantic_hints.topic_hints))
    if semantic_hints.entity_candidates:
        parts.append("Entity candidates: " + ", ".join(semantic_hints.entity_candidates))
    if semantic_hints.time_hints:
        parts.append("Time hints: " + ", ".join(semantic_hints.time_hints))
    if semantic_hints.structure_hints:
        parts.append("Structure hints: " + ", ".join(semantic_hints.structure_hints))
    if semantic_hints.functional_summary:
        parts.append(f"Functional summary: {semantic_hints.functional_summary}")
    if semantic_hints.evidence_hints:
        parts.append("Evidence hints: " + "; ".join(semantic_hints.evidence_hints[:4]))
    return parts
