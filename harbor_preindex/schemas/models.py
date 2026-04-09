"""Core schemas used across the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

TargetKind = Literal["file", "folder"]
MatchType = Literal["exact_file", "likely_file", "folder_zone", "mixed", "no_match"]

_VALID_TARGET_KINDS = set(get_args(TargetKind))
_VALID_MATCH_TYPES = set(get_args(MatchType))


@dataclass(slots=True)
class FileCard:
    """Indexable file representation for retrieval."""

    file_id: str
    path: str
    filename: str
    extension: str
    parent_path: str
    modality: str
    text_for_embedding: str
    metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "path": self.path,
            "filename": self.filename,
            "extension": self.extension,
            "parent_path": self.parent_path,
            "modality": self.modality,
            "text_for_embedding": self.text_for_embedding,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FolderCard:
    """Indexable folder representation for retrieval."""

    folder_id: str
    path: str
    relative_path: str
    name: str
    parent_path: str | None
    text_for_embedding: str
    metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "folder_id": self.folder_id,
            "path": self.path,
            "relative_path": self.relative_path,
            "name": self.name,
            "parent_path": self.parent_path,
            "text_for_embedding": self.text_for_embedding,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class StructuredQueryHints:
    """Lightweight structured hints extracted from a plain text query."""

    raw_query: str
    normalized_terms: list[str] = field(default_factory=list)
    kind_hints: list[str] = field(default_factory=list)
    entity_terms: list[str] = field(default_factory=list)
    time_hints: list[str] = field(default_factory=list)
    topic_hints: list[str] = field(default_factory=list)
    technical_hints: list[str] = field(default_factory=list)
    intent_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "raw_query": self.raw_query,
            "normalized_terms": list(self.normalized_terms),
        }
        if self.kind_hints:
            payload["kind_hints"] = list(self.kind_hints)
        if self.entity_terms:
            payload["entity_terms"] = list(self.entity_terms)
        if self.time_hints:
            payload["time_hints"] = list(self.time_hints)
        if self.topic_hints:
            payload["topic_hints"] = list(self.topic_hints)
        if self.technical_hints:
            payload["technical_hints"] = list(self.technical_hints)
        if self.intent_hint:
            payload["intent_hint"] = self.intent_hint
        return payload


@dataclass(slots=True)
class RetrievalQuery:
    """Structured retrieval request."""

    text: str
    limit: int
    structured_hints: StructuredQueryHints | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "limit": self.limit,
        }


@dataclass(slots=True)
class RetrievalEvidence:
    """Compact explanation metadata for a retrieval match."""

    matched_query_terms: list[str] = field(default_factory=list)
    matched_sources: list[str] = field(default_factory=list)
    source_terms: dict[str, list[str]] = field(default_factory=dict)
    matched_kind_hints: list[str] = field(default_factory=list)
    matched_topic_hints: list[str] = field(default_factory=list)
    matched_entity_candidates: list[str] = field(default_factory=list)
    matched_time_hints: list[str] = field(default_factory=list)
    matched_technical_hints: list[str] = field(default_factory=list)
    matched_imports: list[str] = field(default_factory=list)
    matched_symbols: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.matched_query_terms:
            payload["matched_query_terms"] = list(self.matched_query_terms)
        if self.matched_sources:
            payload["matched_sources"] = list(self.matched_sources)
        if self.source_terms:
            payload["source_terms"] = {
                source: list(terms) for source, terms in self.source_terms.items()
            }
        if self.matched_kind_hints:
            payload["matched_kind_hints"] = list(self.matched_kind_hints)
        if self.matched_topic_hints:
            payload["matched_topic_hints"] = list(self.matched_topic_hints)
        if self.matched_entity_candidates:
            payload["matched_entity_candidates"] = list(self.matched_entity_candidates)
        if self.matched_time_hints:
            payload["matched_time_hints"] = list(self.matched_time_hints)
        if self.matched_technical_hints:
            payload["matched_technical_hints"] = list(self.matched_technical_hints)
        if self.matched_imports:
            payload["matched_imports"] = list(self.matched_imports)
        if self.matched_symbols:
            payload["matched_symbols"] = list(self.matched_symbols)
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


@dataclass(slots=True)
class RetrievalMatch:
    """A file or folder candidate returned by retrieval."""

    target_kind: TargetKind
    target_id: str
    path: str
    score: float
    label: str
    why: str
    evidence: RetrievalEvidence | None = None
    raw_score: float | None = None
    decision_score: float | None = None

    def __post_init__(self) -> None:
        if self.target_kind not in _VALID_TARGET_KINDS:
            raise ValueError(
                f"unsupported target_kind={self.target_kind!r}; "
                f"expected one of {sorted(_VALID_TARGET_KINDS)}"
            )
        if self.raw_score is None:
            self.raw_score = self.score
        if self.decision_score is None:
            self.decision_score = self.score

    def to_dict(self, include_evidence: bool = False) -> dict[str, Any]:
        payload = {
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "path": self.path,
            "score": round(self.score, 4),
            "label": self.label,
            "why": self.why,
        }
        if include_evidence and self.evidence is not None:
            payload["evidence"] = self.evidence.to_dict()
        return payload


@dataclass(slots=True)
class RetrievalResponse:
    """Stable JSON response for Harbor retrieval queries."""

    query: str
    match_type: MatchType
    matches: list[RetrievalMatch]
    confidence: float
    needs_review: bool
    generated_at: str
    query_hints: StructuredQueryHints | None = None

    def __post_init__(self) -> None:
        if self.match_type not in _VALID_MATCH_TYPES:
            raise ValueError(
                f"unsupported match_type={self.match_type!r}; "
                f"expected one of {sorted(_VALID_MATCH_TYPES)}"
            )

    def to_dict(self, include_evidence: bool = False) -> dict[str, Any]:
        payload = {
            "query": self.query,
            "match_type": self.match_type,
            "confidence": round(self.confidence, 4),
            "needs_review": self.needs_review,
            "matches": [match.to_dict(include_evidence=include_evidence) for match in self.matches],
            "generated_at": self.generated_at,
        }
        if include_evidence and self.query_hints is not None:
            payload["query_hints"] = self.query_hints.to_dict()
        return payload


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
class IndexedFileCard:
    """File card plus embedding vector."""

    card: FileCard
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
class FileSearchCandidate:
    """Vector search result for a file card."""

    file_id: str
    path: str
    filename: str
    extension: str
    parent_path: str
    modality: str
    score: float
    text_for_embedding: str
    metadata: dict[str, Any]


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
class BatchSummary:
    """Compact counters for a batch placement run."""

    scanned_files: int
    supported_files: int
    classified: int
    needs_review: int
    skipped: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_files": self.scanned_files,
            "supported_files": self.supported_files,
            "classified": self.classified,
            "needs_review": self.needs_review,
            "skipped": self.skipped,
        }


@dataclass(slots=True)
class BatchPlacement:
    """Per-file placement result inside a batch plan."""

    source_path: str
    selected_path: str | None
    confidence: float
    needs_review: bool
    why: str
    decision_mode: str
    selected_project_id: str | None = None
    top_candidates: list[SearchCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_path": self.source_path,
            "selected_path": self.selected_path,
            "confidence": round(self.confidence, 4),
            "needs_review": self.needs_review,
            "why": self.why,
            "decision_mode": self.decision_mode,
        }
        if self.selected_project_id is not None:
            payload["selected_project_id"] = self.selected_project_id
        if self.top_candidates:
            payload["top_candidates"] = [
                candidate.to_result_dict() for candidate in self.top_candidates
            ]
        return payload


@dataclass(slots=True)
class BatchGroup:
    """Compact grouped view for batch placements sharing the same destination."""

    suggested_target_path: str
    file_count: int
    members: list[str]
    average_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "suggested_target_path": self.suggested_target_path,
            "file_count": self.file_count,
            "members": list(self.members),
        }
        if self.average_confidence is not None:
            payload["average_confidence"] = round(self.average_confidence, 4)
        return payload


@dataclass(slots=True)
class BatchReviewItem:
    """Review queue item extracted from a batch placement run."""

    source_path: str
    why: str
    confidence: float
    top_candidates: list[SearchCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_path": self.source_path,
            "why": self.why,
            "confidence": round(self.confidence, 4),
        }
        if self.top_candidates:
            payload["top_candidates"] = [
                candidate.to_result_dict() for candidate in self.top_candidates
            ]
        return payload


@dataclass(slots=True)
class BatchSkippedItem:
    """Skipped file inside a batch placement run."""

    source_path: str
    reason: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_path": self.source_path,
            "reason": self.reason,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(slots=True)
class BatchQueryResult:
    """JSON-serializable placement plan for a batch of incoming files."""

    input_path: str
    mode: str
    summary: BatchSummary
    placements: list[BatchPlacement]
    groups: list[BatchGroup]
    review_queue: list[BatchReviewItem]
    skipped: list[BatchSkippedItem]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "mode": self.mode,
            "summary": self.summary.to_dict(),
            "placements": [placement.to_dict() for placement in self.placements],
            "groups": [group.to_dict() for group in self.groups],
            "review_queue": [item.to_dict() for item in self.review_queue],
            "skipped": [item.to_dict() for item in self.skipped],
            "generated_at": self.generated_at,
        }


@dataclass(slots=True)
class IndexBuildSummary:
    """JSON-serializable index build summary."""

    root_path: str
    collection: str
    indexed_projects: int
    indexed_files: int
    scanned_directories: int
    recreated_collection: bool
    generated_at: str
    file_collection: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "root_path": self.root_path,
            "collection": self.collection,
            "indexed_projects": self.indexed_projects,
            "indexed_files": self.indexed_files,
            "scanned_directories": self.scanned_directories,
            "recreated_collection": self.recreated_collection,
            "generated_at": self.generated_at,
        }
        if self.file_collection:
            payload["file_collection"] = self.file_collection
        return payload
