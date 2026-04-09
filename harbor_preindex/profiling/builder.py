"""Semantic profile construction."""

from __future__ import annotations

from pathlib import Path

from harbor_preindex.profiling.extraction import ContentExtractor
from harbor_preindex.profiling.folder_semantics import (
    FolderSemanticSignatureBuilder,
    signature_text_lines,
)
from harbor_preindex.schemas import DiscoveredProject, FileQueryContext, ProjectProfile
from harbor_preindex.signals.models import ExtractedSignal
from harbor_preindex.utils.fs import relative_display
from harbor_preindex.utils.text import slugify, truncate_text


class ProjectProfileBuilder:
    """Build semantic profiles for project directories and query files."""

    def __init__(
        self,
        root_path: Path,
        extractor: ContentExtractor,
        max_profile_chars: int,
        folder_signature_builder: FolderSemanticSignatureBuilder | None = None,
    ) -> None:
        self.root_path = root_path
        self.extractor = extractor
        self.max_profile_chars = max_profile_chars
        self.folder_signature_builder = folder_signature_builder

    def build_project_profile(self, project: DiscoveredProject) -> ProjectProfile:
        """Build a single project profile from a discovered directory."""

        relative_path = relative_display(project.path, self.root_path)
        sample_filenames = [sample.name for sample in project.sample_files]
        excerpts_by_path: dict[Path, str] = {}
        sample_excerpts: list[str] = []
        for sample_path in project.sample_files:
            excerpt = self.extractor.extract_excerpt(sample_path)
            excerpts_by_path[sample_path] = excerpt
            if excerpt:
                sample_excerpts.append(f"{sample_path.name}: {excerpt}")

        semantic_signature = None
        if self.folder_signature_builder is not None:
            semantic_signature = self.folder_signature_builder.build(project, excerpts_by_path)

        parts = [
            f"Document count: {project.doc_count}",
        ]
        if semantic_signature is not None:
            parts.extend(signature_text_lines(semantic_signature))
        parts.extend(
            [
                f"Project folder name: {project.path.name}",
                f"Relative path: {relative_path}",
                f"Absolute path: {project.path}",
                f"Parent folder: {project.path.parent.name}",
                f"Sample filenames: {', '.join(sample_filenames) if sample_filenames else 'none'}",
            ]
        )
        if sample_excerpts:
            parts.append("Sample excerpts:\n" + "\n".join(sample_excerpts))

        text_profile = truncate_text("\n".join(parts), self.max_profile_chars)
        return ProjectProfile(
            project_id=slugify(relative_path if relative_path != "." else project.path.name),
            path=str(project.path),
            relative_path=relative_path,
            name=project.path.name,
            parent=project.path.parent.name if project.path.parent != project.path else None,
            sample_filenames=sample_filenames,
            doc_count=project.doc_count,
            text_profile=text_profile,
            semantic_signature=semantic_signature,
        )

    def build_query_context_from_signal(
        self,
        file_path: Path,
        signal: ExtractedSignal,
    ) -> FileQueryContext:
        """Build the minimal query context from an extracted signal."""

        file_name = str(signal.metadata.get("file_name", file_path.name))
        suffix = str(signal.metadata.get("suffix", file_path.suffix.lower()))
        text_excerpt = str(signal.metadata.get("text_excerpt", ""))
        return FileQueryContext(
            input_file=str(signal.metadata.get("input_file", file_path)),
            file_name=file_name,
            suffix=suffix,
            text_excerpt=text_excerpt,
            text_profile=truncate_text(signal.text_for_embedding, self.max_profile_chars),
        )
