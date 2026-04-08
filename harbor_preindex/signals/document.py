"""Document signal extraction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from harbor_preindex.profiling.extraction import ContentExtractor
from harbor_preindex.signals.base import SignalExtractor
from harbor_preindex.signals.models import ExtractedSignal
from harbor_preindex.utils.text import truncate_text


class DocumentSignalExtractor(SignalExtractor):
    """Extract a text signal from document-like files."""

    def __init__(
        self,
        extractor: ContentExtractor,
        supported_extensions: Sequence[str],
        max_profile_chars: int,
    ) -> None:
        self.extractor = extractor
        self.supported_extensions = {extension.lower() for extension in supported_extensions}
        self.max_profile_chars = max_profile_chars

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions

    def extract(self, file_path: Path) -> ExtractedSignal:
        if not self.supports(file_path):
            raise ValueError(f"unsupported document signal for file: {file_path}")

        excerpt = self.extractor.extract_excerpt(file_path)
        parts = [
            f"Incoming file name: {file_path.name}",
            f"Incoming suffix: {file_path.suffix.lower() or 'unknown'}",
            f"Incoming parent folder: {file_path.parent.name}",
        ]
        if excerpt:
            parts.append(f"Incoming text excerpt: {excerpt}")

        return ExtractedSignal(
            modality="document",
            text_for_embedding=truncate_text("\n".join(parts), self.max_profile_chars),
            metadata={
                "input_file": str(file_path),
                "file_name": file_path.name,
                "suffix": file_path.suffix.lower(),
                "parent": file_path.parent.name,
                "text_excerpt": excerpt,
            },
            confidence=0.95 if excerpt else 0.6,
        )
