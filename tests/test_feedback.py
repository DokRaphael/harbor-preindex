from __future__ import annotations

import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def _install_dependency_stubs() -> None:
    if "dotenv" not in sys.modules:
        dotenv_module = types.ModuleType("dotenv")

        def load_dotenv(*args: object, **kwargs: object) -> bool:
            return False

        dotenv_module.load_dotenv = load_dotenv
        sys.modules["dotenv"] = dotenv_module

    if "httpx" not in sys.modules:
        httpx_module = types.ModuleType("httpx")

        class HTTPError(Exception):
            pass

        class Client:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs

            def __enter__(self) -> Client:
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                return None

        httpx_module.Client = Client
        httpx_module.HTTPError = HTTPError
        sys.modules["httpx"] = httpx_module

    if "qdrant_client" not in sys.modules:
        qdrant_module = types.ModuleType("qdrant_client")
        qdrant_models_module = types.ModuleType("qdrant_client.models")

        class QdrantClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs

        class Distance:
            COSINE = "cosine"

        class PointStruct:
            def __init__(self, **kwargs: object) -> None:
                self.payload = kwargs

        class VectorParams:
            def __init__(self, size: int, distance: object) -> None:
                self.size = size
                self.distance = distance

        qdrant_module.QdrantClient = QdrantClient
        qdrant_models_module.Distance = Distance
        qdrant_models_module.PointStruct = PointStruct
        qdrant_models_module.VectorParams = VectorParams
        sys.modules["qdrant_client"] = qdrant_module
        sys.modules["qdrant_client.models"] = qdrant_models_module


_install_dependency_stubs()

from harbor_preindex.main import HarborPreindexApp
from harbor_preindex.retrieval.query_hints import QueryHintExtractor
from harbor_preindex.schemas import (
    BatchGroup,
    BatchGroupDecision,
    BatchGroupSemantics,
    BatchPlacement,
    BatchPlacementGroup,
    BatchQueryResult,
    BatchReviewItem,
    BatchSkippedItem,
    BatchSummary,
    Decision,
    FileQueryContext,
    QueryResult,
    RetrievalMatch,
    RetrievalResponse,
    SearchCandidate,
)
from harbor_preindex.signals.models import ExtractedSignal
from harbor_preindex.storage.sqlite_store import SQLiteAuditStore


class StubSignalExtractor:
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".txt"

    def extract(self, file_path: Path) -> ExtractedSignal:
        return ExtractedSignal(
            modality="document",
            text_for_embedding=f"Incoming file name: {file_path.name}",
            metadata={
                "input_file": str(file_path),
                "file_name": file_path.name,
                "suffix": file_path.suffix.lower(),
                "text_excerpt": file_path.read_text(encoding="utf-8"),
            },
            confidence=0.9,
        )


class StubSignalRegistry:
    def __init__(self) -> None:
        self.extractors = [StubSignalExtractor()]

    def resolve(self, file_path: Path) -> StubSignalExtractor:
        extractor = self.extractors[0]
        if extractor.supports(file_path):
            return extractor
        raise ValueError("unsupported file")


class StubProfileBuilder:
    def build_query_context_from_signal(
        self,
        file_path: Path,
        signal: ExtractedSignal,
    ) -> FileQueryContext:
        return FileQueryContext(
            input_file=str(file_path),
            file_name=str(signal.metadata["file_name"]),
            suffix=str(signal.metadata["suffix"]),
            text_excerpt=str(signal.metadata["text_excerpt"]),
            text_profile=signal.text_for_embedding,
        )


class StubEmbeddingBackend:
    def embed_text(self, text: str) -> list[str]:
        return [text]


class StubRetriever:
    def retrieve(self, query_vector: list[str], limit: int) -> list[SearchCandidate]:
        return [
            SearchCandidate(
                project_id="admin_docs",
                path="/nas/admin/docs",
                name="docs",
                parent="admin",
                score=0.91,
                sample_filenames=["incoming.txt"],
                doc_count=4,
                text_profile="Administrative documents",
            )
        ][:limit]


class StubDecisionEngine:
    def decide(
        self,
        query: FileQueryContext,
        candidates: list[SearchCandidate],
    ) -> Decision:
        return Decision(
            selected_project_id=candidates[0].project_id,
            selected_path=candidates[0].path,
            confidence=0.92,
            mode="auto_top1",
            reason="top1_score_gate_passed",
        )


class StubCollectionStore:
    def collection_exists(self) -> bool:
        return True


class StubResultStore:
    def __init__(self) -> None:
        self.query_results: list[object] = []
        self.batch_results: list[object] = []
        self.retrieval_results: list[object] = []

    def save_query_result(self, result: object) -> None:
        self.query_results.append(result)

    def save_batch_query_result(self, result: object) -> None:
        self.batch_results.append(result)

    def save_retrieval_response(self, result: object) -> None:
        self.retrieval_results.append(result)


class FeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.sqlite_path = self.base_path / "runtime" / "harbor-preindex.sqlite3"
        self.audit_store = SQLiteAuditStore(self.sqlite_path)
        self.result_store = StubResultStore()
        self.app = HarborPreindexApp(
            settings=SimpleNamespace(
                top_k=5,
                qdrant_collection="projects",
                qdrant_file_collection="files",
            ),
            crawler=SimpleNamespace(),
            profile_builder=StubProfileBuilder(),
            signal_registry=StubSignalRegistry(),
            card_builder=SimpleNamespace(),
            embedding_backend=StubEmbeddingBackend(),
            llm_backend=SimpleNamespace(),
            ollama_client=SimpleNamespace(),
            vector_store=StubCollectionStore(),
            file_vector_store=StubCollectionStore(),
            retriever=StubRetriever(),
            file_retriever=SimpleNamespace(),
            retrieval_core=SimpleNamespace(),
            query_hint_extractor=QueryHintExtractor(),
            decision_engine=StubDecisionEngine(),
            result_store=self.result_store,
            audit_store=self.audit_store,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_query_file_result_includes_result_id_and_feedback_can_be_marked_good(self) -> None:
        input_file = self.base_path / "incoming.txt"
        input_file.write_text("hello world", encoding="utf-8")

        result = self.app.query_file(input_file)
        self.assertTrue(result.result_id.startswith("query_file_"))

        feedback = self.app.record_feedback(
            result.result_id,
            feedback_status="good",
            feedback_reason="correct_match",
            notes="looks good",
        )

        self.assertEqual(feedback.query_kind, "query_file")
        self.assertEqual(feedback.feedback_status, "good")
        self.assertEqual(feedback.system_selected_path, "/nas/admin/docs")

        with sqlite3.connect(self.sqlite_path) as connection:
            row = connection.execute(
                """
                SELECT feedback_status, feedback_reason, system_selected_path, notes
                FROM feedback_events
                WHERE source_result_id = ?
                """,
                (result.result_id,),
            ).fetchone()
        self.assertEqual(row, ("good", "correct_match", "/nas/admin/docs", "looks good"))

    def test_query_batch_feedback_uses_query_batch_kind(self) -> None:
        batch_result = BatchQueryResult(
            input_path="/tmp/incoming_dump",
            mode="recursive",
            summary=BatchSummary(
                scanned_files=2,
                supported_files=2,
                classified=0,
                needs_review=2,
                skipped=0,
                groups_total=2,
                groups_proposed_new_subfolder=2,
            ),
            placements=[
                BatchPlacement(
                    source_path="/tmp/incoming_dump/edf_2024.txt",
                    selected_path=None,
                    confidence=0.62,
                    needs_review=True,
                    why="review requested: ambiguous match",
                    decision_mode="review_needed",
                )
            ],
            groups=[],
            review_queue=[],
            placement_groups=[
                BatchPlacementGroup(
                    group_id="grp_001",
                    member_count=1,
                    members=["/tmp/incoming_dump/edf_2024.txt"],
                    group_semantics=BatchGroupSemantics(
                        dominant_topics=["utility"],
                        dominant_entities=["edf"],
                        dominant_time_hints=["2024"],
                    ),
                    decision=BatchGroupDecision(
                        mode="proposed_new_subfolder",
                        selected_parent_path="/nas/admin/invoices",
                        selected_path=None,
                        proposed_subfolder_name="edf_2024",
                        proposed_full_path="/nas/admin/invoices/edf_2024",
                        naming_basis={
                            "dominant_entity": "edf",
                            "dominant_time_hint": "2024",
                        },
                        confidence=0.78,
                        needs_review=True,
                        why="parent folder is plausible but no existing specialized child path is a strong enough match",
                    ),
                )
            ],
            ungrouped_review_items=[],
            skipped=[],
            generated_at="2026-04-10T10:00:00Z",
        )
        self.audit_store.record_batch_query_run(batch_result)

        feedback = self.app.record_feedback(
            batch_result.result_id,
            feedback_status="bad",
            feedback_reason="bad_new_subfolder_proposal",
        )

        self.assertEqual(feedback.query_kind, "query_batch")
        self.assertEqual(feedback.system_mode, "proposed_new_subfolder")
        self.assertEqual(feedback.system_parent_path, "/nas/admin/invoices")

    def test_corrected_feedback_stores_corrected_path_and_parent(self) -> None:
        response = RetrievalResponse(
            query="where are the EDF invoices?",
            match_type="folder_zone",
            matches=[
                RetrievalMatch(
                    target_kind="folder",
                    target_id="utility_invoices",
                    path="/nas/admin/invoices/utilities",
                    score=0.71,
                    label="utilities",
                    why="folder profile overlaps with query topics",
                )
            ],
            confidence=0.71,
            needs_review=True,
            generated_at="2026-04-10T10:00:00Z",
        )
        self.audit_store.record_retrieval_run(response)

        feedback = self.app.record_feedback(
            response.result_id,
            feedback_status="corrected",
            feedback_reason="wrong_path",
            corrected_path="/nas/admin/invoices/edf_2024",
        )

        self.assertEqual(feedback.query_kind, "query")
        self.assertEqual(feedback.corrected_path, "/nas/admin/invoices/edf_2024")
        self.assertEqual(feedback.corrected_parent_path, "/nas/admin/invoices")

        with sqlite3.connect(self.sqlite_path) as connection:
            row = connection.execute(
                """
                SELECT corrected_path, corrected_parent_path
                FROM feedback_events
                WHERE feedback_id = ?
                """,
                (feedback.feedback_id,),
            ).fetchone()
        self.assertEqual(row, ("/nas/admin/invoices/edf_2024", "/nas/admin/invoices"))

    def test_invalid_result_id_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            self.app.record_feedback(
                "query_file_missing",
                feedback_status="bad",
                feedback_reason="wrong_path",
            )

    def test_invalid_feedback_reason_is_rejected(self) -> None:
        query_result = QueryResult(
            input_file="/tmp/incoming.txt",
            top_candidates=[],
            decision=Decision(
                selected_project_id=None,
                selected_path=None,
                confidence=0.0,
                mode="review_needed",
                reason="no_candidates_found",
            ),
            generated_at="2026-04-10T10:00:00Z",
        )
        self.audit_store.record_query_run(query_result)

        with self.assertRaises(ValueError):
            self.app.record_feedback(
                query_result.result_id,
                feedback_status="bad",
                feedback_reason="not_a_reason",
            )

    def test_sqlite_lookup_and_persistence_are_stable(self) -> None:
        query_result = QueryResult(
            input_file="/tmp/incoming.txt",
            top_candidates=[],
            decision=Decision(
                selected_project_id=None,
                selected_path=None,
                confidence=0.52,
                mode="review_needed",
                reason="ambiguous_match",
            ),
            generated_at="2026-04-10T10:00:00Z",
        )
        self.audit_store.record_query_run(query_result)

        source = self.audit_store.lookup_feedback_source(query_result.result_id)
        assert source is not None
        self.assertEqual(source.query_kind, "query_file")
        self.assertEqual(source.system_mode, "review_needed")
        self.assertTrue(source.system_needs_review)

        feedback = self.app.record_feedback(
            query_result.result_id,
            feedback_status="bad",
            feedback_reason="review_was_unnecessary",
        )
        self.assertTrue(feedback.feedback_id.startswith("feedback_"))

        with sqlite3.connect(self.sqlite_path) as connection:
            row = connection.execute(
                """
                SELECT source_result_id, query_kind, feedback_status
                FROM feedback_events
                WHERE feedback_id = ?
                """,
                (feedback.feedback_id,),
            ).fetchone()
        self.assertEqual(
            row,
            (query_result.result_id, "query_file", "bad"),
        )


if __name__ == "__main__":
    unittest.main()
