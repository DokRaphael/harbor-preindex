"""Lightweight content extraction."""

from __future__ import annotations

import json
import re
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any

from harbor_preindex.logging_config import get_logger
from harbor_preindex.utils.text import normalize_text, truncate_text

logger = get_logger(__name__)
_LINE_STRUCTURED_SUFFIXES = {".yaml", ".yml", ".ini", ".conf"}
_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
_MAX_STRUCTURED_LINES = 24
_MAX_XML_NODES = 18


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
            if suffix == ".json":
                return self._extract_json_file(file_path)
            if suffix == ".toml":
                return self._extract_toml_file(file_path)
            if suffix in _LINE_STRUCTURED_SUFFIXES:
                return self._extract_line_structured_file(file_path)
            if suffix == ".xml":
                return self._extract_xml_file(file_path)
            if suffix == ".html":
                return self._extract_html_file(file_path)
            return self._extract_text_file(file_path)
        except OSError:
            return ""

    def _extract_text_file(self, file_path: Path) -> str:
        content = self._read_text_file_best_effort(file_path)
        return self._normalize_excerpt(content)

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

    def _extract_json_file(self, file_path: Path) -> str:
        content = self._read_text_file_best_effort(file_path)
        if not content:
            return ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._normalize_excerpt(content)
        return self._normalize_excerpt("\n".join(_flatten_structured_value(data)))

    def _extract_toml_file(self, file_path: Path) -> str:
        content = self._read_text_file_best_effort(file_path)
        if not content:
            return ""
        try:
            data = tomllib.loads(content)
        except tomllib.TOMLDecodeError:
            return self._normalize_excerpt(content)
        return self._normalize_excerpt("\n".join(_flatten_structured_value(data)))

    def _extract_line_structured_file(self, file_path: Path) -> str:
        content = self._read_text_file_best_effort(file_path)
        if not content:
            return ""

        section_prefix: str | None = None
        nested_prefixes: list[tuple[int, str]] = []
        lines: list[str] = []
        for raw_line in content.splitlines():
            indent = len(raw_line) - len(raw_line.lstrip(" \t"))
            line = raw_line.strip()
            if not line or line.startswith(("#", ";", "//")):
                continue
            if line.startswith("[") and line.endswith("]") and len(line) > 2:
                section_prefix = line[1:-1].strip()
                nested_prefixes.clear()
                lines.append(f"section: {section_prefix}")
                continue

            while nested_prefixes and indent <= nested_prefixes[-1][0]:
                nested_prefixes.pop()

            separator = ":" if ":" in line else "=" if "=" in line else None
            if separator is None:
                lines.append(line.lstrip("- ").strip())
            else:
                key, value = line.split(separator, 1)
                key_text = key.strip().lstrip("- ").strip()
                value_text = value.strip().strip("'\"")
                if not key_text:
                    continue
                prefix_parts = [name for _, name in nested_prefixes]
                if section_prefix:
                    prefix_parts.insert(0, section_prefix)
                label = ".".join([*prefix_parts, key_text]) if prefix_parts else key_text
                if separator == ":" and not value_text:
                    nested_prefixes.append((indent, key_text))
                    lines.append(label)
                else:
                    lines.append(f"{label}: {value_text}" if value_text else label)

            if len(lines) >= _MAX_STRUCTURED_LINES:
                break
        if not lines:
            return self._normalize_excerpt(content)
        return self._normalize_excerpt("\n".join(lines))

    def _extract_xml_file(self, file_path: Path) -> str:
        content = self._read_text_file_best_effort(file_path)
        if not content:
            return ""
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return self._normalize_excerpt(content)

        lines = [f"root: {_strip_xml_namespace(root.tag)}"]
        _flatten_xml_element(root, prefix="", lines=lines)
        return self._normalize_excerpt("\n".join(lines))

    def _extract_html_file(self, file_path: Path) -> str:
        content = self._read_text_file_best_effort(file_path)
        if not content:
            return ""

        title_match = re.search(
            r"<title[^>]*>(.*?)</title>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        body = re.sub(
            r"<(script|style)[^>]*>.*?</\1>",
            " ",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        body = re.sub(r"<[^>]+>", " ", body)
        parts: list[str] = []
        if title_match:
            parts.append("title: " + normalize_text(unescape(title_match.group(1))))
        body_text = normalize_text(unescape(body))
        if body_text:
            parts.append(body_text)
        return self._normalize_excerpt("\n".join(parts))

    def _read_text_file_best_effort(self, file_path: Path) -> str:
        payload = file_path.read_bytes()
        if _looks_binary(payload):
            return ""
        for encoding in _TEXT_ENCODINGS:
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        return payload.decode("utf-8", errors="ignore")

    def _normalize_excerpt(self, value: str) -> str:
        return truncate_text(normalize_text(value), self.max_chars)


def _flatten_structured_value(value: Any, prefix: str = "", lines: list[str] | None = None) -> list[str]:
    output = [] if lines is None else lines
    if len(output) >= _MAX_STRUCTURED_LINES:
        return output

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_structured_value(child, prefix=child_prefix, lines=output)
            if len(output) >= _MAX_STRUCTURED_LINES:
                break
        return output

    if isinstance(value, list):
        if not value or not prefix:
            return output
        scalar_values = [_structured_scalar_text(item) for item in value]
        if all(item is not None for item in scalar_values):
            joined = ", ".join(item for item in scalar_values[:4] if item is not None)
            if joined:
                output.append(f"{prefix}: {joined}")
            return output
        for index, child in enumerate(value[:4]):
            _flatten_structured_value(child, prefix=f"{prefix}[{index}]", lines=output)
            if len(output) >= _MAX_STRUCTURED_LINES:
                break
        return output

    scalar_text = _structured_scalar_text(value)
    if scalar_text is not None and prefix:
        output.append(f"{prefix}: {scalar_text}")
    return output


def _structured_scalar_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    cleaned = normalize_text(str(value))
    return truncate_text(cleaned, 120) if cleaned else None


def _flatten_xml_element(element: ET.Element, prefix: str, lines: list[str]) -> None:
    if len(lines) >= _MAX_XML_NODES:
        return

    tag = _strip_xml_namespace(element.tag)
    path = f"{prefix}.{tag}" if prefix else tag
    text = normalize_text(element.text or "")
    if text:
        lines.append(f"{path}: {truncate_text(text, 120)}")
    for attribute, value in element.attrib.items():
        lines.append(f"{path}.@{attribute}: {truncate_text(normalize_text(value), 120)}")
        if len(lines) >= _MAX_XML_NODES:
            return
    for child in list(element):
        _flatten_xml_element(child, path, lines)
        if len(lines) >= _MAX_XML_NODES:
            return


def _strip_xml_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _looks_binary(payload: bytes) -> bool:
    if not payload:
        return False
    sample = payload[:4096]
    if b"\x00" in sample:
        return True
    control_bytes = sum(1 for byte in sample if byte < 9 or 13 < byte < 32)
    return (control_bytes / max(len(sample), 1)) > 0.2
