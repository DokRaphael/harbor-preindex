"""Final decision engine."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from harbor_preindex.llm.base import LLMBackend
from harbor_preindex.logging_config import get_logger
from harbor_preindex.schemas import Decision, FileQueryContext, SearchCandidate
from harbor_preindex.utils.text import strip_json_fences, truncate_text

logger = get_logger(__name__)

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selected_project_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "mode": {"type": "string", "enum": ["llm_rerank", "review_needed"]},
        "reason": {"type": "string"},
    },
    "required": ["selected_project_id", "confidence", "mode", "reason"],
    "additionalProperties": False,
}


class DecisionEngine:
    """Choose the final project directory for an incoming file."""

    def __init__(
        self,
        llm_backend: LLMBackend,
        auto_accept_score: float,
        auto_accept_score_gap: float,
        llm_max_candidates: int,
    ) -> None:
        self.llm_backend = llm_backend
        self.auto_accept_score = auto_accept_score
        self.auto_accept_score_gap = auto_accept_score_gap
        self.llm_max_candidates = llm_max_candidates

    def decide(
        self,
        query: FileQueryContext,
        candidates: Sequence[SearchCandidate],
    ) -> Decision:
        """Return the final structured decision."""

        if not candidates:
            return Decision(
                selected_project_id=None,
                selected_path=None,
                confidence=0.0,
                mode="review_needed",
                reason="no_candidates_found",
            )

        automatic_decision = self._try_automatic_decision(candidates)
        if automatic_decision is not None:
            return automatic_decision

        return self._llm_rerank(query, list(candidates[: self.llm_max_candidates]))

    def _try_automatic_decision(self, candidates: Sequence[SearchCandidate]) -> Decision | None:
        top1 = candidates[0]
        top2_score = candidates[1].score if len(candidates) > 1 else 0.0
        gap = top1.score - top2_score
        if top1.score >= self.auto_accept_score and gap >= self.auto_accept_score_gap:
            return Decision(
                selected_project_id=top1.project_id,
                selected_path=top1.path,
                confidence=min(0.99, round(top1.score + 0.03, 4)),
                mode="auto_top1",
                reason="top1_score_gate_passed",
            )
        return None

    def _llm_rerank(self, query: FileQueryContext, candidates: list[SearchCandidate]) -> Decision:
        system_prompt = (
            "You route one incoming document to one existing project folder. "
            "Choose only among the provided candidate ids. "
            "If the input is ambiguous or weak, return review_needed. "
            "Respond with JSON only."
        )
        prompt = self._build_prompt(query, candidates)

        try:
            response_text = self.llm_backend.generate_json(
                system_prompt=system_prompt,
                prompt=prompt,
                schema=DECISION_SCHEMA,
            )
            return self._parse_llm_response(response_text, candidates)
        except Exception as exc:
            logger.warning(
                "llm_rerank_failed",
                extra={
                    "input_file": query.input_file,
                    "error": str(exc),
                },
            )
            return Decision(
                selected_project_id=None,
                selected_path=None,
                confidence=0.0,
                mode="review_needed",
                reason="llm_error_or_invalid_response",
            )

    def _build_prompt(self, query: FileQueryContext, candidates: Sequence[SearchCandidate]) -> str:
        lines = [
            "Incoming file:",
            f"- name: {query.file_name}",
            f"- suffix: {query.suffix or 'unknown'}",
            f"- excerpt: {truncate_text(query.text_excerpt or 'unavailable', 900)}",
            "",
            "Candidates:",
        ]
        for index, candidate in enumerate(candidates, start=1):
            lines.append(
                f"{index}. id={candidate.project_id} | "
                f"score={candidate.score:.4f} | path={candidate.path}"
            )
            lines.append(f"   profile={truncate_text(candidate.text_profile, 700)}")

        lines.extend(
            [
                "",
                "Return review_needed when ambiguity remains.",
                "Use the exact candidate id if you choose a folder.",
            ]
        )
        return "\n".join(lines)

    def _parse_llm_response(
        self,
        response_text: str,
        candidates: Sequence[SearchCandidate],
    ) -> Decision:
        data = json.loads(strip_json_fences(response_text))
        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")

        mode = str(data.get("mode", "")).strip()
        confidence = _clamp_confidence(data.get("confidence"))
        reason = str(data.get("reason", "")).strip() or None

        if mode == "review_needed":
            return Decision(
                selected_project_id=None,
                selected_path=None,
                confidence=confidence,
                mode="review_needed",
                reason=reason or "llm_requested_review",
            )

        if mode != "llm_rerank":
            raise ValueError(f"unsupported LLM decision mode: {mode}")

        selected_project_id = data.get("selected_project_id")
        candidate_map = {candidate.project_id: candidate for candidate in candidates}
        if not isinstance(selected_project_id, str) or selected_project_id not in candidate_map:
            raise ValueError("LLM selected an unknown project_id")

        selected = candidate_map[selected_project_id]
        return Decision(
            selected_project_id=selected.project_id,
            selected_path=selected.path,
            confidence=confidence,
            mode="llm_rerank",
            reason=reason,
        )


def _clamp_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
