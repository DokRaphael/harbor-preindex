"""SQLite audit storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from harbor_preindex.schemas import IndexBuildSummary, QueryResult, RetrievalResponse
from harbor_preindex.utils.text import utc_now_iso


class SQLiteAuditStore:
    """Persist build and query events for local audit."""

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
                    created_at,
                    input_file,
                    decision_mode,
                    selected_project_id,
                    confidence,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now_iso(),
                    result.input_file,
                    decision.mode,
                    decision.selected_project_id,
                    decision.confidence,
                    json.dumps(result.to_dict(), ensure_ascii=False),
                ),
            )

    def record_retrieval_run(self, response: RetrievalResponse) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO retrieval_runs (
                    created_at,
                    query_text,
                    match_type,
                    confidence,
                    needs_review,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now_iso(),
                    response.query,
                    response.match_type,
                    response.confidence,
                    int(response.needs_review),
                    json.dumps(response.to_dict(), ensure_ascii=False),
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
                    created_at TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    match_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    needs_review INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
