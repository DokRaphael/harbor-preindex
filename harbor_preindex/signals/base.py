"""Signal extractor base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from harbor_preindex.signals.models import ExtractedSignal


class SignalExtractor(ABC):
    """Base interface for extracting a signal from a file."""

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Return whether this extractor can handle the given file."""

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedSignal:
        """Extract a signal ready to be embedded."""
