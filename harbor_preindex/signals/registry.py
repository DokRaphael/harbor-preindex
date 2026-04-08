"""Signal extractor registry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from harbor_preindex.signals.base import SignalExtractor


class SignalExtractorRegistry:
    """Resolve the appropriate extractor for a file."""

    def __init__(self, extractors: Sequence[SignalExtractor]) -> None:
        self.extractors = list(extractors)

    def resolve(self, file_path: Path) -> SignalExtractor:
        """Return the first extractor that supports the given file."""

        for extractor in self.extractors:
            if extractor.supports(file_path):
                return extractor

        suffix = file_path.suffix.lower() or "[no extension]"
        supported_suffixes = sorted(
            {
                supported_suffix
                for extractor in self.extractors
                for supported_suffix in extractor.supported_suffixes()
            }
        )
        supported_display = ", ".join(supported_suffixes) if supported_suffixes else "unknown"
        raise ValueError(
            f"unsupported file type {suffix!r} for file: {file_path}. "
            f"Supported extensions: {supported_display}"
        )
