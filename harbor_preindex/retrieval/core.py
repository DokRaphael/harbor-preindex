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
        why = "filename and folder context match query"
        if candidate.metadata.get("text_excerpt"):
            why = "filename and extracted text strongly match query"
        elif self._shared_query_tokens(query_text, candidate.filename):
            why = "filename strongly matches query"

        return RetrievalMatch(
            target_kind="file",
            target_id=candidate.file_id,
            path=candidate.path,
            score=calibration.decision_score,
            label=candidate.filename,
            why=why,
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
        why = "folder profile matches query"
        sample_filenames = card.metadata.get("sample_filenames")
        if isinstance(sample_filenames, list) and any(
            self._shared_query_tokens(query_text, str(filename)) for filename in sample_filenames
        ):
            why = "folder path and sample filenames match query"
        elif self._shared_query_tokens(query_text, label):
            why = "folder path strongly matches query"

        return RetrievalMatch(
            target_kind="folder",
            target_id=card.folder_id,
            path=card.path,
            score=calibration.decision_score,
            label=label,
            why=why,
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


class FolderRetriever(Protocol):
    def retrieve(self, query_vector: Sequence[float], limit: int) -> list[FolderSearchCandidate]:
        """Return folder candidates for a query vector."""


class FileRetriever(Protocol):
    def retrieve(self, query_vector: Sequence[float], limit: int) -> list[FileSearchCandidate]:
        """Return file candidates for a query vector."""
