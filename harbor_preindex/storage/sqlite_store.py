"""SQLite audit and feedback storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from harbor_preindex.schemas import (
    BatchQueryResult,
    FeedbackRecord,
    FeedbackSourceResult,
    IndexBuildSummary,
    QueryResult,
    RetrievalResponse,
)
from harbor_preindex.utils.text import utc_now_iso


class SQLiteAuditStore:
    """Persist build, query, and feedback events for local audit."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def record_index_run(self, summary: IndexBuildSummary) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO index_runs (
                    created_at,
                    root_path,
                    collection_name,
                    indexed_projects,
                    scanned_directories,
                    recreated_collection,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now_iso(),
                    summary.root_path,
                    summary.collection,
                    summary.indexed_projects,
                    summary.scanned_directories,
                    int(summary.recreated_collection),
                    json.dumps(summary.to_dict(), ensure_ascii=False),
                ),
            )

    def record_query_run(self, result: QueryResult) -> None:
        decision = result.decision
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO query_runs (
                    result_id,
                    created_at,
                    input_file,
                    decision_mode,
                    selected_project_id,
                    confidence,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.result_id,
                    utc_now_iso(),
                    result.input_file,
                    decision.mode,
                    decision.selected_project_id,
                    decision.confidence,
                    json.dumps(result.to_dict(), ensure_ascii=False),
                ),
            )

    def record_batch_query_run(self, result: BatchQueryResult) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO batch_query_runs (
                    result_id,
                    created_at,
                    input_path,
                    mode,
                    scanned_files,
                    supported_files,
                    classified,
                    needs_review,
                    skipped,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.result_id,
                    utc_now_iso(),
                    result.input_path,
                    result.mode,
                    result.summary.scanned_files,
                    result.summary.supported_files,
                    result.summary.classified,
                    result.summary.needs_review,
                    result.summary.skipped,
                    json.dumps(result.to_dict(), ensure_ascii=False),
                ),
            )

    def record_retrieval_run(self, response: RetrievalResponse) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO retrieval_runs (
                    result_id,
                    created_at,
                    query_text,
                    match_type,
                    confidence,
                    needs_review,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.result_id,
                    utc_now_iso(),
                    response.query,
                    response.match_type,
                    response.confidence,
                    int(response.needs_review),
                    json.dumps(response.to_dict(), ensure_ascii=False),
                ),
            )

    def lookup_feedback_source(self, result_id: str) -> FeedbackSourceResult | None:
        for query_kind, table_name in (
            ("query_file", "query_runs"),
            ("query_batch", "batch_query_runs"),
            ("query", "retrieval_runs"),
        ):
            payload = self._lookup_payload(table_name, result_id)
            if payload is None:
                continue
            if query_kind == "query_file":
                return self._query_file_feedback_source(result_id, payload)
            if query_kind == "query_batch":
                return self._query_batch_feedback_source(result_id, payload)
            return self._retrieval_feedback_source(result_id, payload)
        return None

    def record_feedback(self, feedback: FeedbackRecord) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO feedback_events (
                    feedback_id,
                    source_result_id,
                    query_kind,
                    feedback_status,
                    feedback_reason,
                    corrected_path,
                    corrected_parent_path,
                    notes,
                    system_mode,
                    system_selected_path,
                    system_parent_path,
                    system_confidence,
                    system_needs_review,
                    created_at,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.feedback_id,
                    feedback.source_result_id,
                    feedback.query_kind,
                    feedback.feedback_status,
                    feedback.feedback_reason,
                    feedback.corrected_path,
                    feedback.corrected_parent_path,
                    feedback.notes,
                    feedback.system_mode,
                    feedback.system_selected_path,
                    feedback.system_parent_path,
                    feedback.system_confidence,
                    None
                    if feedback.system_needs_review is None
                    else int(feedback.system_needs_review),
                    feedback.created_at,
                    json.dumps(feedback.to_dict(), ensure_ascii=False),
                ),
            )

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS index_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    indexed_projects INTEGER NOT NULL,
                    scanned_directories INTEGER NOT NULL,
                    recreated_collection INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id TEXT,
                    created_at TEXT NOT NULL,
                    input_file TEXT NOT NULL,
                    decision_mode TEXT NOT NULL,
                    selected_project_id TEXT,
                    confidence REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id TEXT,
                    created_at TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    match_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    needs_review INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_query_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id TEXT,
                    created_at TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    scanned_files INTEGER NOT NULL,
                    supported_files INTEGER NOT NULL,
                    classified INTEGER NOT NULL,
                    needs_review INTEGER NOT NULL,
                    skipped INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT NOT NULL,
                    source_result_id TEXT NOT NULL,
                    query_kind TEXT NOT NULL,
                    feedback_status TEXT NOT NULL,
                    feedback_reason TEXT NOT NULL,
                    corrected_path TEXT,
                    corrected_parent_path TEXT,
                    notes TEXT,
                    system_mode TEXT,
                    system_selected_path TEXT,
                    system_parent_path TEXT,
                    system_confidence REAL,
                    system_needs_review INTEGER,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._ensure_column(connection, "query_runs", "result_id", "TEXT")
            self._ensure_column(connection, "retrieval_runs", "result_id", "TEXT")
            self._ensure_column(connection, "batch_query_runs", "result_id", "TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_runs_result_id ON query_runs(result_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_batch_query_runs_result_id ON batch_query_runs(result_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_retrieval_runs_result_id ON retrieval_runs(result_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_events_source_result_id "
                "ON feedback_events(source_result_id)"
            )

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        existing_columns = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in existing_columns:
            return
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )

    def _lookup_payload(self, table_name: str, result_id: str) -> dict[str, object] | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                f"""
                SELECT payload_json
                FROM {table_name}
                WHERE result_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (result_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(str(row[0]))

    def _query_file_feedback_source(
        self,
        result_id: str,
        payload: dict[str, object],
    ) -> FeedbackSourceResult:
        decision = payload.get("decision", {})
        if not isinstance(decision, dict):
            decision = {}
        selected_path = _optional_text(decision.get("selected_path"))
        system_mode = _optional_text(decision.get("mode")) or "review_needed"
        return FeedbackSourceResult(
            result_id=result_id,
            query_kind="query_file",
            system_mode=system_mode,
            system_selected_path=selected_path,
            system_parent_path=_path_parent(selected_path),
            system_confidence=_optional_float(decision.get("confidence")) or 0.0,
            system_needs_review=system_mode == "review_needed" or selected_path is None,
        )

    def _query_batch_feedback_source(
        self,
        result_id: str,
        payload: dict[str, object],
    ) -> FeedbackSourceResult:
        placement_groups = payload.get("placement_groups", [])
        if not isinstance(placement_groups, list):
            placement_groups = []

        decision_dicts = [
            group.get("decision", {})
            for group in placement_groups
            if isinstance(group, dict) and isinstance(group.get("decision", {}), dict)
        ]
        parent_paths = sorted(
            {
                _optional_text(decision.get("selected_parent_path"))
                for decision in decision_dicts
                if _optional_text(decision.get("selected_parent_path"))
            }
        )
        group_confidences = [
            confidence
            for decision in decision_dicts
            if (confidence := _optional_float(decision.get("confidence"))) is not None
        ]

        if len(decision_dicts) == 1:
            decision = decision_dicts[0]
            selected_path = _optional_text(decision.get("selected_path"))
            selected_parent_path = _optional_text(decision.get("selected_parent_path"))
            return FeedbackSourceResult(
                result_id=result_id,
                query_kind="query_batch",
                system_mode=_optional_text(decision.get("mode")) or "batch_plan",
                system_selected_path=selected_path,
                system_parent_path=selected_parent_path or _path_parent(selected_path),
                system_confidence=_optional_float(decision.get("confidence")) or 0.0,
                system_needs_review=bool(decision.get("needs_review", True)),
            )

        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        ungrouped_review_items = payload.get("ungrouped_review_items", [])
        if not isinstance(ungrouped_review_items, list):
            ungrouped_review_items = []
        return FeedbackSourceResult(
            result_id=result_id,
            query_kind="query_batch",
            system_mode="batch_plan" if decision_dicts else "review_needed",
            system_selected_path=None,
            system_parent_path=parent_paths[0] if len(parent_paths) == 1 else None,
            system_confidence=round(sum(group_confidences) / len(group_confidences), 4)
            if group_confidences
            else 0.0,
            system_needs_review=bool(ungrouped_review_items) or bool(summary.get("needs_review", 0)),
        )

    def _retrieval_feedback_source(
        self,
        result_id: str,
        payload: dict[str, object],
    ) -> FeedbackSourceResult:
        matches = payload.get("matches", [])
        if not isinstance(matches, list):
            matches = []
        top_match = matches[0] if matches and isinstance(matches[0], dict) else {}
        top_path = _optional_text(top_match.get("path"))
        return FeedbackSourceResult(
            result_id=result_id,
            query_kind="query",
            system_mode=_optional_text(payload.get("match_type")) or "no_match",
            system_selected_path=top_path,
            system_parent_path=_path_parent(top_path),
            system_confidence=_optional_float(payload.get("confidence")) or 0.0,
            system_needs_review=bool(payload.get("needs_review", False)),
        )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _path_parent(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.parent == path:
        return path_value
    return str(path.parent)
