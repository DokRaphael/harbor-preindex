from __future__ import annotations

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
from harbor_preindex.schemas import Decision, FileQueryContext, SearchCandidate
from harbor_preindex.signals.models import ExtractedSignal


class StubSignalExtractor:
    def __init__(self, broken_names: set[str] | None = None) -> None:
        self.broken_names = broken_names or set()

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".txt", ".md", ".pdf"}

    def extract(self, file_path: Path) -> ExtractedSignal:
        if file_path.name in self.broken_names:
            raise OSError("unable to read file")
        return ExtractedSignal(
            modality="document",
            text_for_embedding=f"Incoming file name: {file_path.name}",
            metadata={
                "input_file": str(file_path),
                "file_name": file_path.name,
                "suffix": file_path.suffix.lower(),
                "parent": file_path.parent.name,
                "text_excerpt": f"Excerpt for {file_path.stem}",
            },
            confidence=0.9,
        )


class StubSignalRegistry:
    def __init__(self, extractor: StubSignalExtractor) -> None:
        self.extractors = [extractor]
        self.extractor = extractor

    def resolve(self, file_path: Path) -> StubSignalExtractor:
        if self.extractor.supports(file_path):
            return self.extractor
        raise ValueError(f"unsupported file type for file: {file_path}")


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
        text = query_vector[0]
        if "amazon_jan_2025" in text or "amazon_feb_2025" in text:
            return [
                SearchCandidate(
                    project_id="amazon_invoices",
                    path="/nas/admin/invoices/amazon",
                    name="amazon",
                    parent="invoices",
                    score=0.91,
                    sample_filenames=["amazon_invoice_2025.txt"],
                    doc_count=8,
                    text_profile="Amazon invoices and receipts",
                ),
                SearchCandidate(
                    project_id="personal_misc",
                    path="/nas/admin/personal",
                    name="personal",
                    parent="admin",
                    score=0.52,
                    sample_filenames=["note.txt"],
                    doc_count=12,
                    text_profile="Personal notes and misc files",
                ),
            ][:limit]
        if "notes" in text:
            return [
                SearchCandidate(
                    project_id="project_docs",
                    path="/nas/projects/docs",
                    name="docs",
                    parent="projects",
                    score=0.61,
                    sample_filenames=["overview.md"],
                    doc_count=4,
                    text_profile="Project documentation and notes",
                ),
                SearchCandidate(
                    project_id="personal_misc",
                    path="/nas/admin/personal",
                    name="personal",
                    parent="admin",
                    score=0.58,
                    sample_filenames=["note.txt"],
                    doc_count=12,
                    text_profile="Personal notes and misc files",
                ),
            ][:limit]
        if "nested" in text or "single" in text:
            return [
                SearchCandidate(
                    project_id="sample_docs",
                    path="/nas/projects/sample/docs",
                    name="docs",
                    parent="sample",
                    score=0.88,
                    sample_filenames=["single.txt"],
                    doc_count=2,
                    text_profile="Sample documents",
                )
            ][:limit]
        return []


class StubDecisionEngine:
    def decide(
        self,
        query: FileQueryContext,
        candidates: list[SearchCandidate],
    ) -> Decision:
        if query.file_name in {"amazon_jan_2025.txt", "amazon_feb_2025.txt", "nested.txt", "single.txt"}:
            top_candidate = candidates[0]
            return Decision(
                selected_project_id=top_candidate.project_id,
                selected_path=top_candidate.path,
                confidence=0.93 if query.file_name != "amazon_feb_2025.txt" else 0.91,
                mode="auto_top1",
                reason="top1_score_gate_passed",
            )
        if query.file_name == "notes.txt":
            return Decision(
                selected_project_id=None,
                selected_path=None,
                confidence=0.58,
                mode="review_needed",
                reason="ambiguous_match",
            )
        return Decision(
            selected_project_id=None,
            selected_path=None,
            confidence=0.0,
            mode="review_needed",
            reason="no_candidates_found",
        )


class StubCollectionStore:
    def collection_exists(self) -> bool:
        return True


class StubResultStore:
    def save_batch_query_result(self, result: object) -> None:
        self.batch_result = result

    def save_query_result(self, result: object) -> None:
        self.query_result = result


class StubAuditStore:
    def record_batch_query_run(self, result: object) -> None:
        self.batch_result = result

    def record_query_run(self, result: object) -> None:
        self.query_result = result


class BatchQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        extractor = StubSignalExtractor(broken_names={"broken.txt"})
        self.app = HarborPreindexApp(
            settings=SimpleNamespace(
                top_k=5,
                qdrant_collection="projects",
            ),
            crawler=SimpleNamespace(),
            profile_builder=StubProfileBuilder(),
            signal_registry=StubSignalRegistry(extractor),
            card_builder=SimpleNamespace(),
            embedding_backend=StubEmbeddingBackend(),
            llm_backend=SimpleNamespace(),
            ollama_client=SimpleNamespace(),
            vector_store=StubCollectionStore(),
            file_vector_store=SimpleNamespace(),
            retriever=StubRetriever(),
            file_retriever=SimpleNamespace(),
            retrieval_core=SimpleNamespace(),
            query_hint_extractor=SimpleNamespace(),
            decision_engine=StubDecisionEngine(),
            result_store=StubResultStore(),
            audit_store=StubAuditStore(),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_query_batch_directory_groups_results_and_keeps_review_queue(self) -> None:
        batch_dir = self.base_path / "incoming"
        self._write_text(batch_dir / "amazon_jan_2025.txt", "amazon january invoice")
        self._write_text(batch_dir / "amazon_feb_2025.txt", "amazon february invoice")
        self._write_text(batch_dir / "notes.txt", "misc notes")
        self._write_text(batch_dir / "image.png", "not supported")

        result = self.app.query_batch(batch_dir)

        self.assertEqual(result.mode, "recursive")
        self.assertEqual(result.summary.to_dict(), {
            "scanned_files": 4,
            "supported_files": 3,
            "classified": 2,
            "needs_review": 1,
            "skipped": 1,
        })
        self.assertEqual(len(result.placements), 3)
        self.assertEqual(result.groups[0].suggested_target_path, "/nas/admin/invoices/amazon")
        self.assertEqual(result.groups[0].file_count, 2)
        self.assertEqual(
            result.groups[0].members,
            sorted(
                [
                    str(batch_dir / "amazon_jan_2025.txt"),
                    str(batch_dir / "amazon_feb_2025.txt"),
                ]
            ),
        )
        self.assertEqual(len(result.review_queue), 1)
        self.assertEqual(result.review_queue[0].source_path, str(batch_dir / "notes.txt"))
        self.assertEqual(result.skipped[0].reason, "unsupported_extension")

    def test_query_batch_no_recursive_only_scans_top_level_files(self) -> None:
        batch_dir = self.base_path / "incoming"
        nested_dir = batch_dir / "nested"
        self._write_text(batch_dir / "single.txt", "single file")
        self._write_text(nested_dir / "nested.txt", "nested file")

        result = self.app.query_batch(batch_dir, recursive=False)

        self.assertEqual(result.mode, "flat")
        self.assertEqual(result.summary.scanned_files, 1)
        self.assertEqual(result.summary.supported_files, 1)
        self.assertEqual(len(result.placements), 1)
        self.assertEqual(result.placements[0].source_path, str(batch_dir / "single.txt"))

    def test_query_batch_skips_unreadable_supported_files_without_failing(self) -> None:
        batch_dir = self.base_path / "incoming"
        self._write_text(batch_dir / "amazon_jan_2025.txt", "amazon january invoice")
        self._write_text(batch_dir / "broken.txt", "broken")

        result = self.app.query_batch(batch_dir)

        self.assertEqual(result.summary.supported_files, 2)
        self.assertEqual(result.summary.classified, 1)
        self.assertEqual(result.summary.skipped, 1)
        self.assertEqual(result.skipped[0].reason, "unreadable_or_malformed")
        self.assertIn("unable to read file", result.skipped[0].error or "")

    def test_query_batch_non_existent_path_raises_clear_error(self) -> None:
        missing_path = self.base_path / "missing"

        with self.assertRaises(FileNotFoundError):
            self.app.query_batch(missing_path)

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
