"""Hybrid retrieval core."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

from harbor_preindex.retrieval.cards import RetrievalCardBuilder
from harbor_preindex.schemas import (
    FileSearchCandidate,
    MatchType,
    RetrievalEvidence,
    RetrievalMatch,
    RetrievalQuery,
    RetrievalResponse,
    TargetKind,
)
from harbor_preindex.schemas import SearchCandidate as FolderSearchCandidate
from harbor_preindex.utils.text import utc_now_iso

_STOPWORDS = {
    "a",
    "an",
    "and",
    "de",
    "des",
    "document",
    "documents",
    "du",
    "est",
    "et",
    "file",
    "fichier",
    "files",
    "find",
    "in",
    "is",
    "la",
    "le",
    "les",
    "mes",
    "mon",
    "my",
    "of",
    "ou",
    "où",
    "pour",
    "retrouve",
    "retrouver",
    "sont",
    "the",
    "where",
}
_INDEX_RAW_FLOORS: dict[TargetKind, float] = {
    "file": 0.55,
    "folder": 0.35,
}
_RAW_COMPONENT_WEIGHT = 0.4
_WITHIN_INDEX_COMPONENT_WEIGHT = 0.4
_RANK_COMPONENT_WEIGHT = 0.2
_MIN_WITHIN_INDEX_SPREAD = 0.08


@dataclass(slots=True, frozen=True)
class ScoreCalibration:
    """Normalized score used to compare results across retrieval indexes."""

    raw_score: float
    decision_score: float


@dataclass(slots=True, frozen=True)
class MatchExplanation:
    """Compact explanation assembled for a retrieval match."""

    why: str
    evidence: RetrievalEvidence


class HybridRetrievalCore:
    """Run a hybrid file and folder retrieval for plain text queries."""

    def __init__(
        self,
        folder_retriever: FolderRetriever,
        card_builder: RetrievalCardBuilder,
        file_retriever: FileRetriever | None = None,
    ) -> None:
        self.folder_retriever = folder_retriever
        self.file_retriever = file_retriever
        self.card_builder = card_builder

    def retrieve(self, query: RetrievalQuery, query_vector: Sequence[float]) -> RetrievalResponse:
        file_candidates = (
            self.file_retriever.retrieve(query_vector, query.limit)
            if self.file_retriever is not None
            else []
        )
        folder_candidates = self.folder_retriever.retrieve(query_vector, query.limit)

        file_matches = self._build_file_matches(query.text, file_candidates)
        folder_matches = self._build_folder_matches(query.text, folder_candidates)
        matches = self._merge_matches(file_matches, folder_matches, query.limit)
        match_type = self._select_match_type(query.text, file_matches, folder_matches)
        confidence = self._confidence(match_type, matches)
        needs_review = self._needs_review(match_type, matches, confidence)

        return RetrievalResponse(
            query=query.text,
            match_type=match_type,
            matches=matches,
            confidence=confidence,
            needs_review=needs_review,
            generated_at=utc_now_iso(),
        )

    def _build_file_matches(
        self,
        query_text: str,
        file_candidates: Sequence[FileSearchCandidate],
    ) -> list[RetrievalMatch]:
        top_raw_score, tail_raw_score = _score_bounds(file_candidates)
        return [
            self._file_candidate_to_match(
                query_text=query_text,
                candidate=candidate,
                calibration=self._calibrate_score(
                    target_kind="file",
                    raw_score=candidate.score,
                    rank=rank,
                    total=len(file_candidates),
                    top_raw_score=top_raw_score,
                    tail_raw_score=tail_raw_score,
                ),
            )
            for rank, candidate in enumerate(file_candidates)
        ]

    def _build_folder_matches(
        self,
        query_text: str,
        folder_candidates: Sequence[FolderSearchCandidate],
    ) -> list[RetrievalMatch]:
        top_raw_score, tail_raw_score = _score_bounds(folder_candidates)
        return [
            self._folder_candidate_to_match(
                query_text=query_text,
                candidate=candidate,
                calibration=self._calibrate_score(
                    target_kind="folder",
                    raw_score=candidate.score,
                    rank=rank,
                    total=len(folder_candidates),
                    top_raw_score=top_raw_score,
                    tail_raw_score=tail_raw_score,
                ),
            )
            for rank, candidate in enumerate(folder_candidates)
        ]

    def _merge_matches(
        self,
        file_matches: Sequence[RetrievalMatch],
        folder_matches: Sequence[RetrievalMatch],
        limit: int,
    ) -> list[RetrievalMatch]:
        matches = list(file_matches) + list(folder_matches)
        matches.sort(
            key=lambda item: (_decision_score(item), item.target_kind == "file"),
            reverse=True,
        )
        return matches[:limit]

    def _calibrate_score(
        self,
        target_kind: TargetKind,
        raw_score: float,
        rank: int,
        total: int,
        top_raw_score: float,
        tail_raw_score: float,
    ) -> ScoreCalibration:
        raw_floor = _INDEX_RAW_FLOORS[target_kind]
        raw_component = _clamp01((raw_score - raw_floor) / max(1.0 - raw_floor, 1e-9))

        spread = max(top_raw_score - tail_raw_score, _MIN_WITHIN_INDEX_SPREAD)
        within_index_component = 1.0 - _clamp01((top_raw_score - raw_score) / spread)

        if total <= 1:
            rank_component = 1.0
        else:
            rank_component = 1.0 - (rank / (total - 1))

        decision_score = (
            (_RAW_COMPONENT_WEIGHT * raw_component)
            + (_WITHIN_INDEX_COMPONENT_WEIGHT * within_index_component)
            + (_RANK_COMPONENT_WEIGHT * rank_component)
        )
        return ScoreCalibration(
            raw_score=raw_score,
            decision_score=round(_clamp01(decision_score), 4),
        )

    def _file_candidate_to_match(
        self,
        query_text: str,
        candidate: FileSearchCandidate,
        calibration: ScoreCalibration,
    ) -> RetrievalMatch:
        explanation = self._build_file_explanation(query_text, candidate)

        return RetrievalMatch(
            target_kind="file",
            target_id=candidate.file_id,
            path=candidate.path,
            score=calibration.decision_score,
            label=candidate.filename,
            why=explanation.why,
            evidence=explanation.evidence,
            raw_score=calibration.raw_score,
            decision_score=calibration.decision_score,
        )

    def _folder_candidate_to_match(
        self,
        query_text: str,
        candidate: FolderSearchCandidate,
        calibration: ScoreCalibration,
    ) -> RetrievalMatch:
        card = self.card_builder.build_folder_card_from_candidate(candidate)
        label = card.relative_path if card.relative_path != "." else card.name
        explanation = self._build_folder_explanation(query_text, candidate, label)

        return RetrievalMatch(
            target_kind="folder",
            target_id=card.folder_id,
            path=card.path,
            score=calibration.decision_score,
            label=label,
            why=explanation.why,
            evidence=explanation.evidence,
            raw_score=calibration.raw_score,
            decision_score=calibration.decision_score,
        )

    def _select_match_type(
        self,
        query_text: str,
        file_matches: Sequence[RetrievalMatch],
        folder_matches: Sequence[RetrievalMatch],
    ) -> MatchType:
        top_file = file_matches[0] if file_matches else None
        top_folder = folder_matches[0] if folder_matches else None

        if top_file is None and top_folder is None:
            return "no_match"
        if top_file is None:
            return "folder_zone"
        if top_folder is None:
            if self._is_exact_file_match(query_text, top_file):
                return "exact_file"
            return "likely_file"

        file_score = _decision_score(top_file)
        folder_score = _decision_score(top_folder)
        score_gap = file_score - folder_score

        if self._is_exact_file_match(query_text, top_file):
            if file_score >= 0.88 and score_gap >= -0.02:
                return "exact_file"
        if file_score >= 0.84 and score_gap >= 0.04:
            return "likely_file"
        if folder_score >= 0.84 and score_gap <= -0.06:
            return "folder_zone"
        return "mixed"

    def _confidence(self, match_type: MatchType, matches: Sequence[RetrievalMatch]) -> float:
        if not matches or match_type == "no_match":
            return 0.0

        best_score = _decision_score(matches[0])
        ambiguity_penalty = 0.0
        if len(matches) > 1:
            score_gap = _decision_score(matches[0]) - _decision_score(matches[1])
            if score_gap < 0.04:
                ambiguity_penalty += 0.12
            elif score_gap < 0.08:
                ambiguity_penalty += 0.05
        if match_type == "mixed":
            ambiguity_penalty += 0.08

        confidence = best_score - ambiguity_penalty
        return round(max(0.0, min(0.99, confidence)), 4)

    def _needs_review(
        self,
        match_type: MatchType,
        matches: Sequence[RetrievalMatch],
        confidence: float,
    ) -> bool:
        if match_type in {"mixed", "no_match"}:
            return True
        if confidence < 0.72:
            return True
        if len(matches) > 1 and (_decision_score(matches[0]) - _decision_score(matches[1])) < 0.03:
            return True
        return False

    def _is_exact_file_match(self, query_text: str, match: RetrievalMatch) -> bool:
        query_tokens = _meaningful_tokens(query_text)
        if len(query_tokens) < 2:
            return False

        file_tokens = _meaningful_tokens(match.label)
        if not file_tokens:
            return False

        overlap = len(set(query_tokens) & set(file_tokens))
        return overlap == len(query_tokens) and _decision_score(match) >= 0.88

    def _shared_query_tokens(self, query_text: str, label: str) -> bool:
        return bool(set(_meaningful_tokens(query_text)) & set(_meaningful_tokens(label)))

    def _build_file_explanation(
        self,
        query_text: str,
        candidate: FileSearchCandidate,
    ) -> MatchExplanation:
        query_tokens = _meaningful_tokens(query_text)
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        semantic_hints = metadata.get("semantic_hints")
        if not isinstance(semantic_hints, dict):
            semantic_hints = {}

        source_terms: dict[str, list[str]] = {}
        filename_terms = _matched_query_terms(query_tokens, candidate.filename)
        parent_terms = _matched_query_terms(
            query_tokens,
            str(metadata.get("relative_parent_path", candidate.parent_path)),
        )
        content_terms = _matched_query_terms(query_tokens, str(metadata.get("text_excerpt", "")))
        summary_terms = _matched_query_terms(
            query_tokens,
            str(metadata.get("functional_summary", "")),
        )
        topic_matches, topic_terms = _matched_values_and_terms(
            query_text,
            query_tokens,
            _string_list(semantic_hints.get("topic_hints")),
        )
        entity_matches, entity_terms = _matched_values_and_terms(
            query_text,
            query_tokens,
            _string_list(semantic_hints.get("entity_candidates")),
        )
        time_matches, time_terms = _matched_values_and_terms(
            query_text,
            query_tokens,
            _string_list(semantic_hints.get("time_hints")),
        )
        import_matches, import_terms = _matched_values_and_terms(
            query_text,
            query_tokens,
            _string_list(metadata.get("imports")),
        )
        symbol_matches, symbol_terms = _matched_values_and_terms(
            query_text,
            query_tokens,
            _string_list(metadata.get("symbols")),
        )

        _add_source_terms(source_terms, "filename", filename_terms)
        _add_source_terms(source_terms, "parent_path", parent_terms)
        _add_source_terms(source_terms, "content", content_terms)
        _add_source_terms(
            source_terms,
            "semantic_hints",
            [*summary_terms, *topic_terms, *entity_terms, *time_terms],
        )
        _add_source_terms(source_terms, "imports", import_terms)
        _add_source_terms(source_terms, "symbols", symbol_terms)

        notes = _compact_list(
            [
                _note_from_matches("topic hints aligned", topic_matches),
                _note_from_matches("entity candidates aligned", entity_matches),
                _note_from_matches("time hints aligned", time_matches),
                _note_from_matches("imports aligned", import_matches),
                _note_from_matches("symbols aligned", symbol_matches),
            ],
            limit=3,
        )
        evidence = RetrievalEvidence(
            matched_query_terms=_compact_list(
                [
                    *filename_terms,
                    *parent_terms,
                    *content_terms,
                    *summary_terms,
                    *topic_terms,
                    *entity_terms,
                    *time_terms,
                    *import_terms,
                    *symbol_terms,
                ],
                limit=8,
            ),
            matched_sources=list(source_terms),
            source_terms=source_terms,
            matched_topic_hints=topic_matches,
            matched_entity_candidates=entity_matches,
            matched_time_hints=time_matches,
            matched_imports=import_matches,
            matched_symbols=symbol_matches,
            notes=notes,
        )
        why = self._file_why(
            filename_terms=filename_terms,
            parent_terms=parent_terms,
            content_terms=content_terms,
            topic_matches=topic_matches,
            entity_matches=entity_matches,
            time_matches=time_matches,
            import_matches=import_matches,
            symbol_matches=symbol_matches,
            summary_terms=summary_terms,
        )
        return MatchExplanation(why=why, evidence=evidence)

    def _build_folder_explanation(
        self,
        query_text: str,
        candidate: FolderSearchCandidate,
        label: str,
    ) -> MatchExplanation:
        query_tokens = _meaningful_tokens(query_text)
        source_terms: dict[str, list[str]] = {}

        folder_path_terms = _matched_query_terms(query_tokens, f"{label} {candidate.path}")
        folder_profile_terms = _matched_query_terms(query_tokens, candidate.text_profile)
        sample_matches, sample_terms = _matched_values_and_terms(
            query_text,
            query_tokens,
            candidate.sample_filenames,
        )

        _add_source_terms(source_terms, "folder_path", folder_path_terms)
        _add_source_terms(source_terms, "folder_profile", folder_profile_terms)
        _add_source_terms(source_terms, "sample_filenames", sample_terms)

        evidence = RetrievalEvidence(
            matched_query_terms=_compact_list(
                [*folder_path_terms, *folder_profile_terms, *sample_terms],
                limit=8,
            ),
            matched_sources=list(source_terms),
            source_terms=source_terms,
            notes=_compact_list(
                [_note_from_matches("sample filenames aligned", sample_matches)],
                limit=3,
            ),
        )
        why = self._folder_why(
            folder_path_terms=folder_path_terms,
            folder_profile_terms=folder_profile_terms,
            sample_matches=sample_matches,
        )
        return MatchExplanation(why=why, evidence=evidence)

    def _file_why(
        self,
        *,
        filename_terms: list[str],
        parent_terms: list[str],
        content_terms: list[str],
        topic_matches: list[str],
        entity_matches: list[str],
        time_matches: list[str],
        import_matches: list[str],
        symbol_matches: list[str],
        summary_terms: list[str],
    ) -> str:
        semantic_match = bool(topic_matches or entity_matches or time_matches or summary_terms)
        if import_matches and semantic_match:
            return "code imports and functional summary overlap with query topics"
        if symbol_matches and semantic_match:
            return "code symbols and semantic hints overlap with query topics"
        if entity_matches and time_matches:
            return "entity candidate and time hint align with the query"
        if filename_terms and semantic_match:
            return "matched query terms in filename and semantic summary"
        if filename_terms and content_terms:
            return "matched query terms in filename and extracted text"
        if parent_terms and semantic_match:
            return "parent path and semantic hints overlap with query"
        if content_terms and semantic_match:
            return "extracted text and semantic hints overlap with query"
        if filename_terms:
            return "matched query terms in filename"
        if semantic_match:
            return "semantic hints overlap with query topics"
        if content_terms:
            return "extracted text overlaps with query terms"
        if parent_terms:
            return "parent path overlaps with query terms"
        return "retrieval score and file context suggest a match"

    def _folder_why(
        self,
        *,
        folder_path_terms: list[str],
        folder_profile_terms: list[str],
        sample_matches: list[str],
    ) -> str:
        if folder_path_terms and folder_profile_terms:
            return "folder path and profile strongly overlap with query topics"
        if sample_matches and folder_profile_terms:
            return "sample filenames and folder profile overlap with query"
        if folder_path_terms and sample_matches:
            return "folder path and sample filenames overlap with query"
        if folder_path_terms:
            return "folder path matches query terms"
        if folder_profile_terms:
            return "folder profile overlaps with query topics"
        if sample_matches:
            return "sample filenames overlap with query terms"
        return "folder profile retrieved as a likely match"


def _meaningful_tokens(value: str) -> list[str]:
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    )
    tokens = re.findall(r"[a-z0-9]+", normalized)
    filtered = [token for token in tokens if token not in _STOPWORDS]
    return filtered or tokens


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _decision_score(match: RetrievalMatch) -> float:
    if match.decision_score is not None:
        return match.decision_score
    return match.score


def _score_bounds(
    candidates: Sequence[FileSearchCandidate | FolderSearchCandidate],
) -> tuple[float, float]:
    if not candidates:
        return 0.0, 0.0
    top_raw_score = candidates[0].score
    tail_index = min(len(candidates) - 1, 2)
    tail_raw_score = candidates[tail_index].score
    return top_raw_score, tail_raw_score


def _matched_query_terms(query_tokens: Sequence[str], value: str) -> list[str]:
    value_tokens = set(_meaningful_tokens(value))
    return [token for token in query_tokens if token in value_tokens]


def _matched_values_and_terms(
    query_text: str,
    query_tokens: Sequence[str],
    values: Sequence[str],
) -> tuple[list[str], list[str]]:
    normalized_query = _normalized_value(query_text)
    matched_values: list[str] = []
    matched_terms: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        value_terms = _matched_query_terms(query_tokens, cleaned)
        normalized_value = _normalized_value(cleaned)
        if value_terms or (normalized_value and normalized_value in normalized_query):
            matched_values.append(cleaned)
            matched_terms.extend(value_terms)
    return _compact_list(matched_values, limit=4), _compact_list(matched_terms, limit=6)


def _add_source_terms(source_terms: dict[str, list[str]], source: str, terms: list[str]) -> None:
    compact_terms = _compact_list(terms, limit=6)
    if compact_terms:
        source_terms[source] = compact_terms


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _note_from_matches(prefix: str, matches: Sequence[str]) -> str | None:
    compact_matches = _compact_list(list(matches), limit=2)
    if not compact_matches:
        return None
    return f"{prefix}: {', '.join(compact_matches)}"


def _normalized_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _compact_list(values: list[str | None], limit: int) -> list[str]:
    compacted: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        cleaned = str(value).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        compacted.append(cleaned)
        if len(compacted) >= limit:
            break
    return compacted


class FolderRetriever(Protocol):
    def retrieve(self, query_vector: Sequence[float], limit: int) -> list[FolderSearchCandidate]:
        """Return folder candidates for a query vector."""


class FileRetriever(Protocol):
    def retrieve(self, query_vector: Sequence[float], limit: int) -> list[FileSearchCandidate]:
        """Return file candidates for a query vector."""
