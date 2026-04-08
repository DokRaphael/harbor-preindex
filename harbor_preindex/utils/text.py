"""Text helpers."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime


def normalize_text(value: str) -> str:
    """Collapse noisy whitespace while preserving paragraph breaks."""

    cleaned = value.replace("\x00", " ")
    lines = [" ".join(line.strip().split()) for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line)


def truncate_text(value: str, max_chars: int) -> str:
    """Truncate long text conservatively."""

    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def slugify(value: str) -> str:
    """Build a stable identifier from a path-like value."""

    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    lowered = ascii_value.lower().replace("/", "_").replace("\\", "_")
    lowered = re.sub(r"[^a-z0-9_]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "project"


def strip_json_fences(value: str) -> str:
    """Remove common markdown code fences around JSON."""

    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def utc_now_iso() -> str:
    """Return an ISO 8601 UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def utc_now_compact() -> str:
    """Return a compact UTC timestamp for filenames."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
