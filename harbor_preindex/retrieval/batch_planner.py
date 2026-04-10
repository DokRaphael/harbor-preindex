"""Lightweight batch placement planning."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from harbor_preindex.schemas import (
    BatchGroupDecision,
    BatchGroupSemantics,
    BatchPlacementGroup,
    BatchReviewItem,
    FileQueryContext,
    QueryResult,
    SearchCandidate,
    StructuredQueryHints,
)
from harbor_preindex.utils.text import slugify

_GENERIC_TOPIC_TERMS = {
    "document",
    "documents",
    "file",
    "files",
    "invoice",
    "invoices",
    "note",
    "notes",
    "receipt",
    "receipts",
    "scan",
    "scans",
}
_GENERIC_NAMING_TERMS = {
    "billing",
    "document",
    "documents",
    "invoice",
    "invoices",
    "receipt",
    "receipts",
    "telecom",
    "utility",
    "utilities",
}
_NOISY_BATCH_TERMS = {
    "excerpt",
    "file",
    "incoming",
    "md",
    "name",
    "parent",
    "pdf",
    "suffix",
    "text",
    "txt",
    "unknown",
}


@dataclass(slots=True)
class BatchPlanningInput:
    """Per-file planning input reused by the batch planner."""

    result: QueryResult
    query_context: FileQueryContext
    query_hints: StructuredQueryHints


@dataclass(slots=True)
class BatchPlanningResult:
    """Planned batch-level grouping and routing suggestions."""

    placement_groups: list[BatchPlacementGroup]
    ungrouped_review_items: list[BatchReviewItem]


@dataclass(slots=True)
class _ObservationFeatures:
    planning_input: BatchPlanningInput
    dominant_parent_path: str | None
    parent_confidence: float
    parent_weights: dict[str, float]
    anchor_key: str | None
    entity_terms: list[str]
    topic_hints: list[str]
    time_hints: list[str]
    kind_hints: list[str]
    selected_path: str | None
    selected_parent_path: str | None
    selected_path_confidence: float

    @property
    def source_path(self) -> str:
        return self.planning_input.result.input_file


def plan_batch_placements(
    planning_inputs: list[BatchPlanningInput],
    today: date | None = None,
) -> BatchPlanningResult:
    """Plan coherent batch placement groups above per-file query results."""

    features = sorted(
        (_build_features(item) for item in planning_inputs),
        key=lambda item: item.source_path,
    )
    grouped_features = _group_features(features)

    placement_groups: list[BatchPlacementGroup] = []
    ungrouped_review_items: list[BatchReviewItem] = []

    for group_index, members in enumerate(grouped_features, start=1):
        decision = _plan_group_decision(members, today=today or date.today())
        member_paths = sorted(member.source_path for member in members)
        if decision.mode == "review_needed" and len(members) == 1:
            item = members[0]
            ungrouped_review_items.append(
                BatchReviewItem(
                    source_path=item.source_path,
                    why=decision.why,
                    confidence=decision.confidence,
                    top_candidates=item.planning_input.result.top_candidates,
                )
            )
            continue

        placement_groups.append(
            BatchPlacementGroup(
                group_id=f"grp_{group_index:03d}",
                member_count=len(member_paths),
                members=member_paths,
                group_semantics=_group_semantics(members),
                decision=decision,
            )
        )

    placement_groups.sort(
        key=lambda group: (
            group.decision.selected_parent_path or "",
            group.decision.selected_path or group.decision.proposed_full_path or "",
            group.members[0] if group.members else "",
        )
    )
    for index, group in enumerate(placement_groups, start=1):
        group.group_id = f"grp_{index:03d}"

    ungrouped_review_items.sort(key=lambda item: item.source_path)
    return BatchPlanningResult(
        placement_groups=placement_groups,
        ungrouped_review_items=ungrouped_review_items,
    )


def _group_features(features: list[_ObservationFeatures]) -> list[list[_ObservationFeatures]]:
    grouped: dict[tuple[str, str], list[_ObservationFeatures]] = defaultdict(list)
    singletons: list[list[_ObservationFeatures]] = []

    for feature in features:
        if feature.anchor_key and feature.dominant_parent_path:
            grouped[(feature.dominant_parent_path, feature.anchor_key)].append(feature)
        else:
            singletons.append([feature])

    combined = [grouped[key] for key in sorted(grouped)] + singletons
    return [sorted(group, key=lambda item: item.source_path) for group in combined]


def _build_features(planning_input: BatchPlanningInput) -> _ObservationFeatures:
    result = planning_input.result
    decision = result.decision
    parent_weights = _parent_weights(result.top_candidates)
    dominant_parent_path, parent_confidence = _best_weighted_path(parent_weights)
    selected_path = decision.selected_path
    selected_parent_path = _path_parent(selected_path) if selected_path else None

    entity_terms = _meaningful_terms(planning_input.query_hints.entity_terms)
    topic_hints = [
        topic
        for topic in _meaningful_terms(planning_input.query_hints.topic_hints)
        if topic not in _GENERIC_TOPIC_TERMS
    ]
    time_hints = list(planning_input.query_hints.time_hints)
    kind_hints = list(planning_input.query_hints.kind_hints)
    anchor_key = _group_anchor(
        selected_path=selected_path,
        selected_parent_path=selected_parent_path,
        decision_mode=decision.mode,
        confidence=decision.confidence,
        entity_terms=entity_terms,
        topic_hints=topic_hints,
        time_hints=time_hints,
    )

    return _ObservationFeatures(
        planning_input=planning_input,
        dominant_parent_path=dominant_parent_path,
        parent_confidence=parent_confidence,
        parent_weights=parent_weights,
        anchor_key=anchor_key,
        entity_terms=entity_terms,
        topic_hints=topic_hints,
        time_hints=time_hints,
        kind_hints=kind_hints,
        selected_path=selected_path,
        selected_parent_path=selected_parent_path,
        selected_path_confidence=decision.confidence if selected_path else 0.0,
    )


def _group_anchor(
    *,
    selected_path: str | None,
    selected_parent_path: str | None,
    decision_mode: str,
    confidence: float,
    entity_terms: list[str],
    topic_hints: list[str],
    time_hints: list[str],
) -> str | None:
    if (
        selected_path
        and selected_parent_path
        and selected_path != selected_parent_path
        and decision_mode == "auto_top1"
        and confidence >= 0.82
    ):
        return f"path:{selected_path}"

    year_hint = _explicit_year(time_hints)
    dominant_entity = entity_terms[0] if entity_terms else None
    dominant_topic = topic_hints[0] if topic_hints else None

    if dominant_entity and year_hint:
        return f"entity:{dominant_entity}|year:{year_hint}"
    if dominant_entity:
        return f"entity:{dominant_entity}"
    if dominant_topic and year_hint:
        return f"topic:{dominant_topic}|year:{year_hint}"
    if dominant_topic:
        return f"topic:{dominant_topic}"
    return None


def _plan_group_decision(
    members: list[_ObservationFeatures],
    today: date,
) -> BatchGroupDecision:
    group_semantics = _group_semantics(members)
    parent_path, parent_confidence = _best_group_parent(members)
    existing_path, existing_path_support, existing_path_confidence = _best_existing_path(members)
    coherent = _group_is_coherent(members, group_semantics)

    if existing_path and existing_path_support >= 0.75 and existing_path_confidence >= 0.82:
        selected_parent = parent_path or _path_parent(existing_path)
        mode = "existing_subpath" if selected_parent and selected_parent != existing_path else "existing_path"
        return BatchGroupDecision(
            mode=mode,
            selected_parent_path=selected_parent,
            selected_path=existing_path,
            confidence=min(0.99, round(existing_path_confidence + 0.03, 4)),
            needs_review=False,
            why=(
                "coherent subset strongly aligns with an existing specialized child folder"
                if mode == "existing_subpath"
                else "coherent subset strongly aligns with an existing destination path"
            ),
        )

    if parent_path and parent_confidence >= 0.55 and coherent:
        proposed_subfolder_name, naming_basis = _proposed_subfolder_name(group_semantics, today=today)
        if proposed_subfolder_name and _can_propose_new_subfolder(members, group_semantics):
            proposed_full_path = str(Path(parent_path) / proposed_subfolder_name)
            confidence = min(
                0.89,
                round(0.58 + (parent_confidence * 0.18) + (0.08 if len(members) > 1 else 0.04), 4),
            )
            return BatchGroupDecision(
                mode="proposed_new_subfolder",
                selected_parent_path=parent_path,
                selected_path=None,
                proposed_subfolder_name=proposed_subfolder_name,
                proposed_full_path=proposed_full_path,
                naming_basis=naming_basis,
                confidence=confidence,
                needs_review=True,
                why="parent folder is plausible but no existing specialized child path is a strong enough match",
            )

    if parent_path and coherent and len(members) > 1:
        return BatchGroupDecision(
            mode="review_needed",
            selected_parent_path=parent_path,
            selected_path=None,
            confidence=max(0.0, round(min(parent_confidence, 0.74), 4)),
            needs_review=True,
            why="coherent subset shares a plausible parent, but existing child paths remain ambiguous",
        )

    return BatchGroupDecision(
        mode="review_needed",
        selected_parent_path=parent_path,
        selected_path=None,
        confidence=round(_average(member.planning_input.result.decision.confidence for member in members), 4),
        needs_review=True,
        why="ambiguous with no reliable group-level support",
    )


def _group_semantics(members: list[_ObservationFeatures]) -> BatchGroupSemantics:
    topic_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    time_counts: Counter[str] = Counter()
    for member in members:
        topic_counts.update(member.topic_hints)
        entity_counts.update(member.entity_terms)
        time_counts.update(member.time_hints)
    return BatchGroupSemantics(
        dominant_topics=_top_terms(topic_counts, limit=4),
        dominant_entities=_top_terms(entity_counts, limit=3),
        dominant_time_hints=_top_terms(time_counts, limit=3),
    )


def _best_existing_path(members: list[_ObservationFeatures]) -> tuple[str | None, float, float]:
    path_counter: Counter[str] = Counter()
    confidence_counter: defaultdict[str, list[float]] = defaultdict(list)
    for member in members:
        if not member.selected_path:
            continue
        path_counter[member.selected_path] += 1
        confidence_counter[member.selected_path].append(member.selected_path_confidence)

    if not path_counter:
        return None, 0.0, 0.0
    existing_path, count = sorted(path_counter.items(), key=lambda item: (-item[1], item[0]))[0]
    support = count / max(len(members), 1)
    avg_confidence = _average(confidence_counter[existing_path])
    return existing_path, round(support, 4), round(avg_confidence, 4)


def _best_group_parent(members: list[_ObservationFeatures]) -> tuple[str | None, float]:
    weights: Counter[str] = Counter()
    total = 0.0
    for member in members:
        for path, weight in member.parent_weights.items():
            weights[path] += weight
            total += weight
        if member.selected_parent_path:
            weights[member.selected_parent_path] += 0.18
            total += 0.18
    if not weights:
        return None, 0.0
    parent_path, weight = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[0]
    return parent_path, round(weight / max(total, 1e-9), 4)


def _group_is_coherent(members: list[_ObservationFeatures], semantics: BatchGroupSemantics) -> bool:
    if len(members) == 1:
        return bool(
            semantics.dominant_entities
            or semantics.dominant_topics
            or semantics.dominant_time_hints
            or members[0].dominant_parent_path
        )

    selected_paths = {member.selected_path for member in members if member.selected_path}
    if len(selected_paths) == 1 and selected_paths:
        return True

    if semantics.dominant_entities and all(
        semantics.dominant_entities[0] in member.entity_terms for member in members if member.entity_terms
    ):
        return True

    if semantics.dominant_topics and all(
        semantics.dominant_topics[0] in member.topic_hints for member in members if member.topic_hints
    ):
        return True

    return False


def _proposed_subfolder_name(
    group_semantics: BatchGroupSemantics,
    today: date,
) -> tuple[str | None, dict[str, str]]:
    basis: dict[str, str] = {}
    dominant_entity = group_semantics.dominant_entities[0] if group_semantics.dominant_entities else None
    dominant_topic = group_semantics.dominant_topics[0] if group_semantics.dominant_topics else None
    dominant_year = _explicit_year(group_semantics.dominant_time_hints)

    if dominant_entity in _GENERIC_NAMING_TERMS:
        dominant_entity = None

    if dominant_entity:
        basis["dominant_entity"] = dominant_entity
    if dominant_topic:
        basis["dominant_topic"] = dominant_topic
    if dominant_year:
        basis["dominant_time_hint"] = dominant_year

    if dominant_entity and dominant_year:
        return slugify(f"{dominant_entity}_{dominant_year}"), basis
    if dominant_topic and dominant_year:
        return slugify(f"{dominant_topic}_{dominant_year}"), basis
    if dominant_entity:
        return slugify(dominant_entity), basis
    if dominant_topic:
        return slugify(dominant_topic), basis
    return None, basis


def _can_propose_new_subfolder(
    members: list[_ObservationFeatures],
    group_semantics: BatchGroupSemantics,
) -> bool:
    has_structured_signal = any(member.kind_hints for member in members)
    has_specific_anchor = bool(
        group_semantics.dominant_entities
        or _explicit_year(group_semantics.dominant_time_hints)
        or (
            group_semantics.dominant_topics
            and group_semantics.dominant_topics[0] not in _GENERIC_TOPIC_TERMS
        )
    )
    return has_structured_signal and has_specific_anchor


def _parent_weights(candidates: list[SearchCandidate]) -> dict[str, float]:
    weights: Counter[str] = Counter()
    for rank, candidate in enumerate(candidates[:3]):
        parent_path = _candidate_parent_path(candidate)
        if not parent_path:
            continue
        weight = max(candidate.score, 0.01) * (1.0 - (rank * 0.15))
        weights[parent_path] += weight
    return dict(weights)


def _candidate_parent_path(candidate: SearchCandidate) -> str | None:
    candidate_path = candidate.path.strip()
    if not candidate_path:
        return None

    signature = candidate.semantic_signature
    path = Path(candidate_path)
    if signature is not None:
        if signature.folder_role in {"leaf_specialized", "entity_bucket", "time_bucket"} and path.parent != path:
            return str(path.parent)
        if signature.folder_role in {"container", "project_root", "mixed"}:
            return candidate_path

    if candidate.parent and path.parent != path:
        return str(path.parent)
    return candidate_path


def _path_parent(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.parent == path:
        return path_value
    return str(path.parent)


def _explicit_year(time_hints: list[str]) -> str | None:
    for hint in time_hints:
        if len(hint) == 4 and hint.isdigit():
            return hint
    return None


def _best_weighted_path(weights: dict[str, float]) -> tuple[str | None, float]:
    if not weights:
        return None, 0.0
    total = sum(weights.values())
    path, weight = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[0]
    return path, round(weight / max(total, 1e-9), 4)


def _top_terms(counter: Counter[str], limit: int) -> list[str]:
    return [term for term, _count in counter.most_common(limit)]


def _average(values: list[float] | tuple[float, ...] | object) -> float:
    numeric_values = list(values) if not isinstance(values, list) else values
    if not numeric_values:
        return 0.0
    return sum(float(value) for value in numeric_values) / len(numeric_values)


def _meaningful_terms(values: list[str]) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip().lower()
        if not cleaned or cleaned in _NOISY_BATCH_TERMS or cleaned.isdigit():
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        filtered.append(cleaned)
    return filtered
