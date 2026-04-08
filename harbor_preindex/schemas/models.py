"""Core schemas used across the application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DiscoveredProject:
    """Directory candidate found during crawling."""

    path: Path
    relative_path: Path
    sample_files: list[Path]
    doc_count: int


@dataclass(slots=True)
class ProjectProfile:
    """Semantic profile stored in the vector index."""

    project_id: str
    path: str
    relative_path: str
    name: str
    parent: str | None
    sample_filenames: list[str]
    doc_count: int
    text_profile: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "path": self.path,
            "relative_path": self.relative_path,
            "name": self.name,
            "parent": self.parent,
            "sample_filenames": self.sample_filenames,
            "doc_count": self.doc_count,
            "text_profile": self.text_profile,
        }


@dataclass(slots=True)
class IndexedProject:
    """Profile plus embedding vector."""

    profile: ProjectProfile
    embedding: list[float]


@dataclass(slots=True)
class SearchCandidate:
    """Vector search result."""

    project_id: str
    path: str
    name: str
    parent: str | None
    score: float
    sample_filenames: list[str]
    doc_count: int
    text_profile: str

    def to_result_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "path": self.path,
            "score": round(self.score, 4),
        }


@dataclass(slots=True)
class FileQueryContext:
    """Minimal context extracted from an incoming file."""

    input_file: str
    file_name: str
    suffix: str
    text_excerpt: str
    text_profile: str


@dataclass(slots=True)
class Decision:
    """Structured final routing decision."""

    selected_project_id: str | None
    selected_path: str | None
    confidence: float
    mode: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "selected_project_id": self.selected_project_id,
            "selected_path": self.selected_path,
            "confidence": round(self.confidence, 4),
            "mode": self.mode,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(slots=True)
class QueryResult:
    """JSON-serializable query result."""

    input_file: str
    top_candidates: list[SearchCandidate]
    decision: Decision
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_file": self.input_file,
            "top_candidates": [candidate.to_result_dict() for candidate in self.top_candidates],
            "decision": self.decision.to_dict(),
            "generated_at": self.generated_at,
        }


@dataclass(slots=True)
class IndexBuildSummary:
    """JSON-serializable index build summary."""

    root_path: str
    collection: str
    indexed_projects: int
    scanned_directories: int
    recreated_collection: bool
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_path": self.root_path,
            "collection": self.collection,
            "indexed_projects": self.indexed_projects,
            "scanned_directories": self.scanned_directories,
            "recreated_collection": self.recreated_collection,
            "generated_at": self.generated_at,
        }
