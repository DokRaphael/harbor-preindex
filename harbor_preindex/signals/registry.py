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
        raise ValueError(f"no SignalExtractor available for file: {file_path}")
