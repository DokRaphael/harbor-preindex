"""Application entrypoints and service wiring."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from harbor_preindex.crawler import ProjectCrawler
from harbor_preindex.decision import DecisionEngine
from harbor_preindex.embedding import OllamaEmbeddingBackend
from harbor_preindex.llm import OllamaLLMBackend
from harbor_preindex.logging_config import configure_logging, get_logger
from harbor_preindex.profiling import ContentExtractor, ProjectProfileBuilder
from harbor_preindex.retrieval import ProjectRetriever
from harbor_preindex.schemas import (
    FileQueryContext,
    IndexBuildSummary,
    IndexedProject,
    ProjectProfile,
    QueryResult,
)
from harbor_preindex.settings import Settings, load_settings
from harbor_preindex.signals.document import DocumentSignalExtractor
from harbor_preindex.signals.models import ExtractedSignal
from harbor_preindex.signals.registry import SignalExtractorRegistry
from harbor_preindex.storage import JsonResultStore, QdrantProjectStore, SQLiteAuditStore
from harbor_preindex.utils.iterables import chunked
from harbor_preindex.utils.ollama_api import OllamaApiClient, OllamaApiError
from harbor_preindex.utils.text import utc_now_iso

logger = get_logger(__name__)


@dataclass(slots=True)
class HarborPreindexApp:
    """Main application object used by the CLI."""

    settings: Settings
    crawler: ProjectCrawler
    profile_builder: ProjectProfileBuilder
    signal_registry: SignalExtractorRegistry
    embedding_backend: OllamaEmbeddingBackend
    llm_backend: OllamaLLMBackend
    ollama_client: OllamaApiClient
    vector_store: QdrantProjectStore
    retriever: ProjectRetriever
    decision_engine: DecisionEngine
    result_store: JsonResultStore
    audit_store: SQLiteAuditStore

    @classmethod
    def from_settings(cls, settings: Settings) -> HarborPreindexApp:
        configure_logging(settings.log_level)

        extractor = ContentExtractor(max_chars=settings.max_text_snippet_chars)
        profile_builder = ProjectProfileBuilder(
            root_path=settings.harbor_root,
            extractor=extractor,
            max_profile_chars=settings.max_profile_chars,
        )
        signal_registry = SignalExtractorRegistry(
            [
                DocumentSignalExtractor(
                    extractor=extractor,
                    supported_extensions=settings.supported_extensions,
                    max_profile_chars=settings.max_profile_chars,
                )
            ]
        )
        crawler = ProjectCrawler(
            root_path=settings.harbor_root,
            sample_files_per_directory=settings.sample_files_per_directory,
            supported_extensions=settings.supported_extensions,
            excluded_path_segments=settings.excluded_path_segments,
        )
        ollama_client = OllamaApiClient(
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
            api_key=settings.ollama_api_key,
            max_retries=settings.ollama_max_retries,
        )
        embedding_backend = OllamaEmbeddingBackend(ollama_client, settings.embedding_model)
        llm_backend = OllamaLLMBackend(ollama_client, settings.llm_model)
        vector_store = QdrantProjectStore(
            settings.qdrant_mode,
            settings.qdrant_path,
            settings.qdrant_collection,
        )
        retriever = ProjectRetriever(vector_store)
        decision_engine = DecisionEngine(
            llm_backend=llm_backend,
            auto_accept_score=settings.auto_accept_score,
            auto_accept_score_gap=settings.auto_accept_score_gap,
            llm_max_candidates=settings.llm_max_candidates,
        )
        result_store = JsonResultStore(settings.results_dir)
        audit_store = SQLiteAuditStore(settings.sqlite_path)

        return cls(
            settings=settings,
            crawler=crawler,
            profile_builder=profile_builder,
            signal_registry=signal_registry,
            embedding_backend=embedding_backend,
            llm_backend=llm_backend,
            ollama_client=ollama_client,
            vector_store=vector_store,
            retriever=retriever,
            decision_engine=decision_engine,
            result_store=result_store,
            audit_store=audit_store,
        )

    def build_index(self, recreate: bool = False) -> IndexBuildSummary:
        """Build or rebuild the vector index from the NAS."""

        logger.info(
            "build_index_started",
            extra={
                "root_path": str(self.settings.harbor_root),
                "collection": self.settings.qdrant_collection,
                "recreate": recreate,
            },
        )

        self.profile_builder.extractor.reset_pdf_stats()
        discovered_projects, visited_directories = self.crawler.scan_projects()
        profiles = [
            self.profile_builder.build_project_profile(project)
            for project in discovered_projects
        ]
        pdf_stats = self.profile_builder.extractor.pdf_stats()

        logger.info(
            "profiling_finished",
            extra={
                "project_profiles": len(profiles),
                "pdf_extraction_successes": pdf_stats.success_count,
                "pdf_extraction_failures": pdf_stats.failure_count,
            },
        )

        if profiles:
            indexed_projects = self._embed_profiles_in_batches(profiles)
            self.vector_store.ensure_collection(
                len(indexed_projects[0].embedding),
                recreate=recreate,
            )
            self.vector_store.upsert_projects(indexed_projects)
        elif recreate:
            self.vector_store.clear_collection()

        summary = IndexBuildSummary(
            root_path=str(self.settings.harbor_root),
            collection=self.settings.qdrant_collection,
            indexed_projects=len(profiles),
            scanned_directories=visited_directories,
            recreated_collection=recreate,
            generated_at=utc_now_iso(),
        )
        self.result_store.save_index_summary(summary)
        self.audit_store.record_index_run(summary)

        logger.info(
            "build_index_finished",
            extra={
                "indexed_projects": summary.indexed_projects,
                "scanned_directories": summary.scanned_directories,
                "pdf_extraction_successes": pdf_stats.success_count,
                "pdf_extraction_failures": pdf_stats.failure_count,
            },
        )
        return summary

    def _embed_profiles_in_batches(self, profiles: list[ProjectProfile]) -> list[IndexedProject]:
        """Embed project profiles in small batches to avoid Ollama timeouts."""

        total_profiles = len(profiles)
        batch_size = self.settings.embedding_batch_size
        total_batches = ceil(total_profiles / batch_size)

        logger.info(
            "embedding_batches_started",
            extra={
                "total_profiles": total_profiles,
                "batch_size": batch_size,
                "total_batches": total_batches,
            },
        )

        indexed_projects: list[IndexedProject] = []
        for batch_number, profile_batch in enumerate(chunked(profiles, batch_size), start=1):
            logger.info(
                "embedding_batch_started",
                extra={
                    "batch_number": batch_number,
                    "total_batches": total_batches,
                    "batch_size": len(profile_batch),
                },
            )
            try:
                embeddings = self.embedding_backend.embed_texts(
                    [profile.text_profile for profile in profile_batch]
                )
                batch_projects = [
                    IndexedProject(profile=profile, embedding=embedding)
                    for profile, embedding in zip(profile_batch, embeddings, strict=True)
                ]
            except (OllamaApiError, ValueError) as exc:
                logger.error(
                    "embedding_batch_failed",
                    extra={
                        "batch_number": batch_number,
                        "total_batches": total_batches,
                        "batch_size": len(profile_batch),
                        "error": str(exc),
                    },
                )
                raise RuntimeError(
                    f"embedding batch {batch_number}/{total_batches} failed after "
                    f"{self.settings.ollama_max_retries + 1} attempt(s)"
                ) from exc

            indexed_projects.extend(batch_projects)
            logger.info(
                "embedding_batch_finished",
                extra={
                    "batch_number": batch_number,
                    "total_batches": total_batches,
                    "batch_size": len(profile_batch),
                    "embedded_profiles": len(indexed_projects),
                    "total_profiles": total_profiles,
                },
            )

        return indexed_projects

    def query_file(self, file_path: Path, top_k: int | None = None) -> QueryResult:
        """Route a new file against the existing project index."""

        result, _query_context, _signal = self._run_query(file_path, top_k=top_k)
        return result

    def query_file_debug_payload(self, file_path: Path, top_k: int | None = None) -> dict[str, Any]:
        """Return query output augmented with debug profile information."""

        result, query_context, signal = self._run_query(file_path, top_k=top_k)
        payload = result.to_dict()
        payload["debug"] = {
            "signal_modality": signal.modality,
            "signal_confidence": round(signal.confidence, 4),
            "query_text_excerpt": query_context.text_excerpt,
            "query_text_profile": query_context.text_profile,
            "candidate_text_profiles": [
                {
                    "project_id": candidate.project_id,
                    "path": candidate.path,
                    "score": round(candidate.score, 4),
                    "text_profile": candidate.text_profile,
                }
                for candidate in result.top_candidates
            ],
        }
        return payload

    def _run_query(
        self,
        file_path: Path,
        top_k: int | None = None,
    ) -> tuple[QueryResult, FileQueryContext, ExtractedSignal]:
        """Execute a query and return both result and extracted query context."""

        if not file_path.exists():
            raise FileNotFoundError(f"input file does not exist: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"input path is not a file: {file_path}")
        if not self.vector_store.collection_exists():
            raise RuntimeError(
                f"Qdrant collection '{self.settings.qdrant_collection}' does not exist; "
                "run `build-index` first"
            )

        limit = top_k if top_k is not None else self.settings.top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than zero")

        signal_extractor = self.signal_registry.resolve(file_path)
        signal = signal_extractor.extract(file_path)
        query_context = self.profile_builder.build_query_context_from_signal(file_path, signal)
        embedding = self.embedding_backend.embed_text(signal.text_for_embedding)
        candidates = self.retriever.retrieve(embedding, limit)
        decision = self.decision_engine.decide(query_context, candidates)

        result = QueryResult(
            input_file=str(file_path),
            top_candidates=candidates,
            decision=decision,
            generated_at=utc_now_iso(),
        )
        self.result_store.save_query_result(result)
        self.audit_store.record_query_run(result)

        logger.info(
            "query_file_finished",
            extra={
                "input_file": str(file_path),
                "candidate_count": len(candidates),
                "decision_mode": decision.mode,
                "signal_modality": signal.modality,
                "selected_project_id": decision.selected_project_id,
            },
        )
        return result, query_context, signal

    def health_check(self) -> dict[str, Any]:
        """Return a JSON-friendly health report."""

        report: dict[str, Any] = {
            "app_name": self.settings.app_name,
            "orchestrator_node": self.settings.orchestrator_node,
            "generated_at": utc_now_iso(),
            "status": "ok",
            "checks": {},
        }

        harbor_root_ok = self.settings.harbor_root.exists() and self.settings.harbor_root.is_dir()
        report["checks"]["harbor_root"] = {
            "ok": harbor_root_ok,
            "path": str(self.settings.harbor_root),
        }

        try:
            version = self.ollama_client.get_version()
            models = self.ollama_client.list_models()
            embedding_available = self.settings.embedding_model in models
            llm_available = self.settings.llm_model in models
            report["checks"]["ollama"] = {
                "ok": embedding_available and llm_available,
                "base_url": self.settings.ollama_base_url,
                "version": version,
                "models": models,
                "embedding_model": self.settings.embedding_model,
                "embedding_model_available": embedding_available,
                "llm_model": self.settings.llm_model,
                "llm_model_available": llm_available,
            }
        except Exception as exc:
            report["checks"]["ollama"] = {
                "ok": False,
                "base_url": self.settings.ollama_base_url,
                "error": str(exc),
            }

        try:
            qdrant_info = self.vector_store.collection_info()
            report["checks"]["qdrant_local"] = {"ok": True, **qdrant_info}
        except Exception as exc:
            report["checks"]["qdrant_local"] = {
                "ok": False,
                "path": str(self.settings.qdrant_path),
                "error": str(exc),
            }

        if not all(bool(check.get("ok")) for check in report["checks"].values()):
            report["status"] = "degraded"

        return report


def create_application() -> HarborPreindexApp:
    """Create the fully wired application from environment configuration."""

    settings = load_settings()
    return HarborPreindexApp.from_settings(settings)
