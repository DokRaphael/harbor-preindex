"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path


def is_hidden(path: Path) -> bool:
    """Return whether any path segment is hidden."""

    return any(part.startswith(".") for part in path.parts if part not in {".", ".."})


def relative_display(path: Path, root: Path) -> str:
    """Return a readable relative path when possible."""

    try:
        value = path.relative_to(root)
    except ValueError:
        return str(path)
    return "." if str(value) == "." else str(value)
