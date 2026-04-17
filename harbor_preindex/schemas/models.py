"""Core schemas used across the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args
from uuid import uuid4

TargetKind = Literal["file", "folder"]
MatchType = Literal["exact_file", "likely_file", "folder_zone", "mixed", "no_match"]
FolderRole = Literal[
    "container",
    "project_root",
    "entity_bucket",
    "time_bucket",
    "leaf_specialized",
    "mixed",
]
QueryKind = Literal["query_file", "query_batch", "query"]
FeedbackStatus = Literal["good", "bad", "corrected"]
FeedbackReason = Literal[
    "correct_match",
    "wrong_path",
    "wrong_parent",
    "should_have_split",
    "should_not_have_split",
    "review_was_correct",
    "review_was_unnecessary",
    "bad_new_subfolder_proposal",
    "good_new_subfolder_proposal_bad_name",
    "ambiguous",
    "other",
]

_VALID_TARGET_KINDS = set(get_args(TargetKind))
_VALID_MATCH_TYPES = set(get_args(MatchType))
_VALID_FOLDER_ROLES = set(get_args(FolderRole))
_VALID_QUERY_KINDS = set(get_args(QueryKind))
_VALID_FEEDBACK_STATUSES = set(get_args(FeedbackStatus))
_VALID_FEEDBACK_REASONS = set(get_args(FeedbackReason))


def _generate_result_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


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
class FolderSemanticSignature:
    """Compact semantic signature for an indexed destination folder."""

    folder_role: FolderRole
    dominant_topics: list[str] = field(default_factory=list)
    dominant_entities: list[str] = field(default_factory=list)
    dominant_time_hints: list[str] = field(default_factory=list)
    dominant_kinds: list[str] = field(default_factory=list)
    frequent_extensions: list[str] = field(default_factory=list)
    representative_terms: list[str] = field(default_factory=list)
    discriminative_terms: list[str] = field(default_factory=list)
    notable_children: list[str] = field(default_factory=list)
    sample_filenames: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.folder_role not in _VALID_FOLDER_ROLES:
            raise ValueError(
                f"unsupported folder_role={self.folder_role!r}; "
                f"expected one of {sorted(_VALID_FOLDER_ROLES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "folder_role": self.folder_role,
        }
        if self.dominant_topics:
            payload["dominant_topics"] = list(self.dominant_topics)
        if self.dominant_entities:
            payload["dominant_entities"] = list(self.dominant_entities)
        if self.dominant_time_hints:
            payload["dominant_time_hints"] = list(self.dominant_time_hints)
        if self.dominant_kinds:
            payload["dominant_kinds"] = list(self.dominant_kinds)
        if self.frequent_extensions:
            payload["frequent_extensions"] = list(self.frequent_extensions)
        if self.representative_terms:
            payload["representative_terms"] = list(self.representative_terms)
        if self.discriminative_terms:
            payload["discriminative_terms"] = list(self.discriminative_terms)
        if self.notable_children:
            payload["notable_children"] = list(self.notable_children)
        if self.sample_filenames:
            payload["sample_filenames"] = list(self.sample_filenames)
        return payload


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
    result_id: str = field(default_factory=lambda: _generate_result_id("retrieval"))

    def __post_init__(self) -> None:
        if self.match_type not in _VALID_MATCH_TYPES:
            raise ValueError(
                f"unsupported match_type={self.match_type!r}; "
                f"expected one of {sorted(_VALID_MATCH_TYPES)}"
            )

    def to_dict(self, include_evidence: bool = False) -> dict[str, Any]:
        payload = {
            "query": self.query,
            "result_id": self.result_id,
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
    semantic_signature: FolderSemanticSignature | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "path": self.path,
            "relative_path": self.relative_path,
            "name": self.name,
            "parent": self.parent,
            "sample_filenames": self.sample_filenames,
            "doc_count": self.doc_count,
            "text_profile": self.text_profile,
        }
        if self.semantic_signature is not None:
            payload["semantic_signature"] = self.semantic_signature.to_dict()
        return payload


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
    semantic_signature: FolderSemanticSignature | None = None
    raw_score: float | None = None
    semantic_bonus: float = 0.0

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
    result_id: str = field(default_factory=lambda: _generate_result_id("query_file"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_file": self.input_file,
            "result_id": self.result_id,
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
    groups_total: int = 0
    groups_existing_path: int = 0
    groups_existing_subpath: int = 0
    groups_proposed_new_subfolder: int = 0
    groups_review_needed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_files": self.scanned_files,
            "supported_files": self.supported_files,
            "classified": self.classified,
            "needs_review": self.needs_review,
            "skipped": self.skipped,
            "groups_total": self.groups_total,
            "groups_existing_path": self.groups_existing_path,
            "groups_existing_subpath": self.groups_existing_subpath,
            "groups_proposed_new_subfolder": self.groups_proposed_new_subfolder,
            "groups_review_needed": self.groups_review_needed,
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
class BatchGroupSemantics:
    """Compact semantic summary for a planned batch subset."""

    dominant_topics: list[str] = field(default_factory=list)
    dominant_entities: list[str] = field(default_factory=list)
    dominant_time_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.dominant_topics:
            payload["dominant_topics"] = list(self.dominant_topics)
        if self.dominant_entities:
            payload["dominant_entities"] = list(self.dominant_entities)
        if self.dominant_time_hints:
            payload["dominant_time_hints"] = list(self.dominant_time_hints)
        return payload


@dataclass(slots=True)
class BatchGroupDecision:
    """Planned decision for a coherent batch subset."""

    mode: str
    selected_parent_path: str | None
    selected_path: str | None
    confidence: float
    needs_review: bool
    why: str
    proposed_subfolder_name: str | None = None
    proposed_full_path: str | None = None
    naming_basis: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "selected_parent_path": self.selected_parent_path,
            "selected_path": self.selected_path,
            "confidence": round(self.confidence, 4),
            "needs_review": self.needs_review,
            "why": self.why,
        }
        if self.proposed_subfolder_name is not None:
            payload["proposed_subfolder_name"] = self.proposed_subfolder_name
        if self.proposed_full_path is not None:
            payload["proposed_full_path"] = self.proposed_full_path
        if self.naming_basis:
            payload["naming_basis"] = dict(self.naming_basis)
        return payload


@dataclass(slots=True)
class BatchPlacementGroup:
    """Coherent planned subset for batch placement."""

    group_id: str
    member_count: int
    members: list[str]
    group_semantics: BatchGroupSemantics
    decision: BatchGroupDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "member_count": self.member_count,
            "members": list(self.members),
            "group_semantics": self.group_semantics.to_dict(),
            "decision": self.decision.to_dict(),
        }


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
    result_id: str = field(default_factory=lambda: _generate_result_id("query_batch"))
    placement_groups: list[BatchPlacementGroup] = field(default_factory=list)
    ungrouped_review_items: list[BatchReviewItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "result_id": self.result_id,
            "mode": self.mode,
            "summary": self.summary.to_dict(),
            "placements": [placement.to_dict() for placement in self.placements],
            "groups": [group.to_dict() for group in self.groups],
            "review_queue": [item.to_dict() for item in self.review_queue],
            "placement_groups": [group.to_dict() for group in self.placement_groups],
            "ungrouped_review_items": [
                item.to_dict() for item in self.ungrouped_review_items
            ],
            "skipped": [item.to_dict() for item in self.skipped],
            "generated_at": self.generated_at,
        }


@dataclass(slots=True)
class FeedbackSourceResult:
    """Minimal stored system decision context used for feedback capture."""

    result_id: str
    query_kind: QueryKind
    system_mode: str
    system_selected_path: str | None
    system_parent_path: str | None
    system_confidence: float
    system_needs_review: bool


@dataclass(slots=True)
class FeedbackRecord:
    """Structured human feedback event stored alongside local results."""

    source_result_id: str
    query_kind: QueryKind
    feedback_status: FeedbackStatus
    feedback_reason: FeedbackReason
    created_at: str
    feedback_id: str = field(default_factory=lambda: _generate_result_id("feedback"))
    corrected_path: str | None = None
    corrected_parent_path: str | None = None
    notes: str | None = None
    system_mode: str | None = None
    system_selected_path: str | None = None
    system_parent_path: str | None = None
    system_confidence: float | None = None
    system_needs_review: bool | None = None

    def __post_init__(self) -> None:
        if self.query_kind not in _VALID_QUERY_KINDS:
            raise ValueError(
                f"unsupported query_kind={self.query_kind!r}; "
                f"expected one of {sorted(_VALID_QUERY_KINDS)}"
            )
        if self.feedback_status not in _VALID_FEEDBACK_STATUSES:
            raise ValueError(
                f"unsupported feedback_status={self.feedback_status!r}; "
                f"expected one of {sorted(_VALID_FEEDBACK_STATUSES)}"
            )
        if self.feedback_reason not in _VALID_FEEDBACK_REASONS:
            raise ValueError(
                f"unsupported feedback_reason={self.feedback_reason!r}; "
                f"expected one of {sorted(_VALID_FEEDBACK_REASONS)}"
            )
        if self.feedback_status == "corrected" and not (
            self.corrected_path or self.corrected_parent_path
        ):
            raise ValueError(
                "corrected feedback requires corrected_path or corrected_parent_path"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feedback_id": self.feedback_id,
            "source_result_id": self.source_result_id,
            "query_kind": self.query_kind,
            "feedback_status": self.feedback_status,
            "feedback_reason": self.feedback_reason,
            "created_at": self.created_at,
        }
        if self.corrected_path is not None:
            payload["corrected_path"] = self.corrected_path
        if self.corrected_parent_path is not None:
            payload["corrected_parent_path"] = self.corrected_parent_path
        if self.notes:
            payload["notes"] = self.notes
        if self.system_mode is not None:
            payload["system_mode"] = self.system_mode
        if self.system_selected_path is not None:
            payload["system_selected_path"] = self.system_selected_path
        if self.system_parent_path is not None:
            payload["system_parent_path"] = self.system_parent_path
        if self.system_confidence is not None:
            payload["system_confidence"] = round(self.system_confidence, 4)
        if self.system_needs_review is not None:
            payload["system_needs_review"] = self.system_needs_review
        return payload


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
