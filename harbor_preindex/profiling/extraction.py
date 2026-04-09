"""Lightweight content extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harbor_preindex.logging_config import get_logger
from harbor_preindex.utils.text import normalize_text, truncate_text

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class PdfExtractionStats:
    """Simple counters for PDF extraction attempts."""

    success_count: int
    failure_count: int


class ContentExtractor:
    """Extract small text excerpts from supported files."""

    def __init__(self, max_chars: int, max_pdf_pages: int = 2) -> None:
        self.max_chars = max_chars
        self.max_pdf_pages = max_pdf_pages
        self._pdf_success_count = 0
        self._pdf_failure_count = 0

    def reset_pdf_stats(self) -> None:
        """Reset PDF extraction counters."""

        self._pdf_success_count = 0
        self._pdf_failure_count = 0

    def pdf_stats(self) -> PdfExtractionStats:
        """Return current PDF extraction counters."""

        return PdfExtractionStats(
            success_count=self._pdf_success_count,
            failure_count=self._pdf_failure_count,
        )

    def extract_excerpt(self, file_path: Path) -> str:
        """Extract a normalized excerpt from a supported document."""

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            try:
                excerpt = self._extract_pdf(file_path)
                self._pdf_success_count += 1
                return excerpt
            except Exception as exc:
                self._pdf_failure_count += 1
                logger.warning(
                    "pdf_excerpt_failed",
                    extra={
                        "file_path": str(file_path),
                        "error": str(exc),
                        "fallback": "filename_only",
                    },
                )
                return ""
        try:
            return self._extract_text_file(file_path)
        except OSError:
            return ""

    def _extract_text_file(self, file_path: Path) -> str:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return truncate_text(normalize_text(content), self.max_chars)

    def _extract_pdf(self, file_path: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        parts: list[str] = []
        page_count = min(len(reader.pages), self.max_pdf_pages)
        for index in range(page_count):
            page = reader.pages[index]
            extracted = page.extract_text() or ""
            if extracted:
                parts.append(extracted)
            if sum(len(part) for part in parts) >= self.max_chars:
                break
        return truncate_text(normalize_text("\n".join(parts)), self.max_chars)
