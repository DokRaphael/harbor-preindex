from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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
from harbor_preindex.schemas import DiscoveredProject, FileCard, ProjectProfile


class StubExtractor:
    def reset_pdf_stats(self) -> None:
        return None

    def pdf_stats(self) -> SimpleNamespace:
        return SimpleNamespace(success_count=0, failure_count=0)


class StubProfileBuilder:
    def __init__(self) -> None:
        self.extractor = StubExtractor()

    def build_project_profile(self, project: DiscoveredProject) -> ProjectProfile:
        return ProjectProfile(
            project_id="admin_docs",
            path=str(project.path),
            relative_path=str(project.relative_path),
            name=project.path.name,
            parent=project.path.parent.name,
            sample_filenames=[path.name for path in project.sample_files],
            doc_count=project.doc_count,
            text_profile="Project profile text",
        )


class StubCrawler:
    def __init__(self, project: DiscoveredProject, project_files: list[Path]) -> None:
        self.project = project
        self.project_files = project_files

    def scan_projects(self) -> tuple[list[DiscoveredProject], int]:
        return [self.project], 1

    def list_project_files(self, project: DiscoveredProject) -> list[Path]:
        self.last_project = project
        return list(self.project_files)


class StubCardBuilder:
    def build_file_card(self, file_path: Path) -> FileCard:
        if file_path.name == "broken.txt":
            raise ValueError("corrupted file")

        return FileCard(
            file_id=f"file:{file_path.name}",
            path=str(file_path),
            filename=file_path.name,
            extension=file_path.suffix,
            parent_path=str(file_path.parent),
            modality="document",
            text_for_embedding=f"File name: {file_path.name}",
            metadata={},
        )


class StubEmbeddingBackend:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.5] for index, _ in enumerate(texts)]


class StubVectorStore:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[int, bool]] = []
        self.upserted_payloads: list[object] = []
        self.clear_called = False

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        self.ensure_calls.append((vector_size, recreate))

    def upsert_projects(self, payloads: list[object]) -> None:
        self.upserted_payloads = list(payloads)

    def upsert_file_cards(self, payloads: list[object]) -> None:
        self.upserted_payloads = list(payloads)

    def clear_collection(self) -> None:
        self.clear_called = True


class StubResultStore:
    def save_index_summary(self, summary: object) -> None:
        self.summary = summary


class StubAuditStore:
    def record_index_run(self, summary: object) -> None:
        self.summary = summary


class BuildIndexBestEffortTests(unittest.TestCase):
    def test_broken_file_card_is_skipped_without_failing_build_index(self) -> None:
        project = DiscoveredProject(
            path=Path("/tmp/storage-root/admin/docs"),
            relative_path=Path("admin/docs"),
            sample_files=[Path("/tmp/storage-root/admin/docs/good.txt")],
            doc_count=2,
        )
        crawler = StubCrawler(
            project=project,
            project_files=[
                Path("/tmp/storage-root/admin/docs/good.txt"),
                Path("/tmp/storage-root/admin/docs/broken.txt"),
            ],
        )
        project_store = StubVectorStore()
        file_store = StubVectorStore()

        app = HarborPreindexApp(
            settings=SimpleNamespace(
                harbor_root=Path("/tmp/storage-root"),
                qdrant_collection="projects",
                qdrant_file_collection="files",
                embedding_batch_size=8,
                ollama_max_retries=0,
            ),
            crawler=crawler,
            profile_builder=StubProfileBuilder(),
            signal_registry=SimpleNamespace(),
            card_builder=StubCardBuilder(),
            embedding_backend=StubEmbeddingBackend(),
            llm_backend=SimpleNamespace(),
            ollama_client=SimpleNamespace(),
            vector_store=project_store,
            file_vector_store=file_store,
            retriever=SimpleNamespace(),
            file_retriever=SimpleNamespace(),
            retrieval_core=SimpleNamespace(),
            decision_engine=SimpleNamespace(),
            result_store=StubResultStore(),
            audit_store=StubAuditStore(),
        )

        with patch("harbor_preindex.main.logger.warning") as warning_logger:
            summary = app.build_index()

        self.assertEqual(summary.indexed_projects, 1)
        self.assertEqual(summary.indexed_files, 1)
        self.assertEqual(len(project_store.upserted_payloads), 1)
        self.assertEqual(len(file_store.upserted_payloads), 1)
        warning_logger.assert_called_once()
        self.assertEqual(
            warning_logger.call_args.kwargs["extra"]["file_path"],
            "/tmp/storage-root/admin/docs/broken.txt",
        )
        self.assertIn("corrupted file", warning_logger.call_args.kwargs["extra"]["error"])


if __name__ == "__main__":
    unittest.main()
