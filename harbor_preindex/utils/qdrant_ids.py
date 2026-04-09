"""Helpers for stable Qdrant point identifiers."""

from __future__ import annotations

import uuid

QDRANT_POINT_NAMESPACE = uuid.UUID("9b0f2a2a-c4d2-5d68-94d3-66b04f6a4204")


def make_qdrant_point_id(path: str) -> str:
    """Return a stable UUID5 point id derived from a project path."""

    normalized_path = path.strip()
    if not normalized_path:
        raise ValueError("path must not be empty when building a Qdrant point id")
    return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, normalized_path))


def make_file_point_id(path: str) -> str:
    """Return a stable UUID5 point id derived from a file path."""

    return make_qdrant_point_id(f"file:{path}")
