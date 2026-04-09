"""Builders for retrieval cards."""

from __future__ import annotations

import re
from pathlib import Path

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
        max_profile_chars: int,
    ) -> None:
        self.root_path = root_path
        self.signal_registry = signal_registry
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
            },
        )

    def build_file_card(self, file_path: Path) -> FileCard:
        """Build a file card from a supported file path."""

        extractor = self.signal_registry.resolve(file_path)
        signal = extractor.extract(file_path)
        relative_parent = relative_display(file_path.parent, self.root_path)
        relative_path = relative_display(file_path, self.root_path)
        excerpt = str(signal.metadata.get("text_excerpt", "")).strip()
        readable_stem = _readable_stem(file_path.stem)

        parts = [
            f"File name: {file_path.name}",
            f"File stem: {readable_stem}",
            f"Extension: {file_path.suffix.lower() or 'unknown'}",
            f"Parent folder: {file_path.parent.name}",
            f"Parent path: {relative_parent}",
            f"Relative file path: {relative_path}",
        ]
        if excerpt:
            parts.append(f"Extracted text excerpt: {excerpt}")

        return FileCard(
            file_id=make_file_point_id(str(file_path)),
            path=str(file_path),
            filename=file_path.name,
            extension=file_path.suffix.lower(),
            parent_path=str(file_path.parent),
            modality=signal.modality,
            text_for_embedding=truncate_text("\n".join(parts), self.max_profile_chars),
            metadata={
                "relative_path": relative_path,
                "relative_parent_path": relative_parent,
                "text_excerpt": excerpt,
                "signal_confidence": signal.confidence,
            },
        )


def _readable_stem(value: str) -> str:
    cleaned = re.sub(r"[_\-\.]+", " ", value).strip()
    return cleaned or value
