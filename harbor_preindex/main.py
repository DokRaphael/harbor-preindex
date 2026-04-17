"""Application entrypoints and service wiring."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
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
from harbor_preindex.profiling.folder_semantics import FolderSemanticSignatureBuilder
from harbor_preindex.retrieval.cards import RetrievalCardBuilder
from harbor_preindex.retrieval.batch_planner import BatchPlanningInput, plan_batch_placements
from harbor_preindex.retrieval.core import HybridRetrievalCore
from harbor_preindex.retrieval.folder_semantics import rerank_folder_candidates
from harbor_preindex.retrieval.query_hints import QueryHintExtractor
from harbor_preindex.retrieval.service import FileCardRetriever, ProjectRetriever
from harbor_preindex.semantic import (
    CodeSemanticEnricher,
    DocumentSemanticEnricher,
    SemanticEnricherRegistry,
)
from harbor_preindex.schemas import (
    BatchGroup,
    BatchPlacement,
    BatchQueryResult,
    BatchReviewItem,
    BatchSkippedItem,
    BatchSummary,
    DiscoveredProject,
    FeedbackRecord,
    FileCard,
    FileQueryContext,
    IndexBuildSummary,
    IndexedFileCard,
    IndexedProject,
    ProjectProfile,
    QueryResult,
    RetrievalQuery,
    RetrievalResponse,
)
from harbor_preindex.settings import Settings, load_settings
from harbor_preindex.signals.document import DocumentSignalExtractor
from harbor_preindex.signals.models import ExtractedSignal
from harbor_preindex.signals.registry import SignalExtractorRegistry
from harbor_preindex.storage import (
    JsonResultStore,
    QdrantFileStore,
    QdrantProjectStore,
    SQLiteAuditStore,
    create_local_qdrant_client,
)
from harbor_preindex.utils.iterables import chunked
from harbor_preindex.utils.ollama_api import OllamaApiClient, OllamaApiError
from harbor_preindex.utils.text import utc_now_iso

logger = get_logger(__name__)


class BatchFileProcessingError(Exception):
    """Expected file-level failure during a batch placement run."""

    def __init__(self, source_path: Path, reason: str, error: Exception) -> None:
        self.source_path = str(source_path)
        self.reason = reason
        self.error = error
        super().__init__(f"{reason}: {error}")


@dataclass(slots=True)
class HarborPreindexApp:
    """Main application object used by the CLI."""

    settings: Settings
    crawler: ProjectCrawler
    profile_builder: ProjectProfileBuilder
    signal_registry: SignalExtractorRegistry
    card_builder: RetrievalCardBuilder
    embedding_backend: OllamaEmbeddingBackend
    llm_backend: OllamaLLMBackend
    ollama_client: OllamaApiClient
    vector_store: QdrantProjectStore
    file_vector_store: QdrantFileStore
    retriever: ProjectRetriever
    file_retriever: FileCardRetriever
    retrieval_core: HybridRetrievalCore
    query_hint_extractor: QueryHintExtractor
    decision_engine: DecisionEngine
    result_store: JsonResultStore
    audit_store: SQLiteAuditStore

    @classmethod
    def from_settings(cls, settings: Settings) -> HarborPreindexApp:
        configure_logging(settings.log_level)

        extractor = ContentExtractor(max_chars=settings.max_text_snippet_chars)
        semantic_registry = SemanticEnricherRegistry(
            [
                CodeSemanticEnricher(),
                DocumentSemanticEnricher(),
            ]
        )
        folder_signature_builder = FolderSemanticSignatureBuilder(
            semantic_registry=semantic_registry,
            max_profile_chars=settings.max_profile_chars,
        )
        profile_builder = ProjectProfileBuilder(
            root_path=settings.harbor_root,
            extractor=extractor,
            max_profile_chars=settings.max_profile_chars,
            folder_signature_builder=folder_signature_builder,
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
        card_builder = RetrievalCardBuilder(
            root_path=settings.harbor_root,
            signal_registry=signal_registry,
            semantic_registry=semantic_registry,
            max_profile_chars=settings.max_profile_chars,
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
        qdrant_client = create_local_qdrant_client(settings.qdrant_path)
        vector_store = QdrantProjectStore(
            settings.qdrant_mode,
            settings.qdrant_path,
            settings.qdrant_collection,
            client=qdrant_client,
        )
        file_vector_store = QdrantFileStore(
            settings.qdrant_mode,
            settings.qdrant_path,
            settings.qdrant_file_collection,
            client=qdrant_client,
        )
        retriever = ProjectRetriever(vector_store)
        file_retriever = FileCardRetriever(file_vector_store)
        query_hint_extractor = QueryHintExtractor()
        retrieval_core = HybridRetrievalCore(
            folder_retriever=retriever,
            file_retriever=file_retriever,
            card_builder=card_builder,
            query_hint_extractor=query_hint_extractor,
        )
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
            card_builder=card_builder,
            embedding_backend=embedding_backend,
            llm_backend=llm_backend,
            ollama_client=ollama_client,
            vector_store=vector_store,
            file_vector_store=file_vector_store,
            retriever=retriever,
            file_retriever=file_retriever,
            retrieval_core=retrieval_core,
            query_hint_extractor=query_hint_extractor,
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
        file_cards, skipped_file_cards = self._build_file_cards(discovered_projects)
        pdf_stats = self.profile_builder.extractor.pdf_stats()

        logger.info(
            "profiling_finished",
            extra={
                "project_profiles": len(profiles),
                "file_cards": len(file_cards),
                "skipped_file_cards": skipped_file_cards,
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

        if file_cards:
            indexed_file_cards = self._embed_file_cards_in_batches(file_cards)
            self.file_vector_store.ensure_collection(
                len(indexed_file_cards[0].embedding),
                recreate=recreate,
            )
            self.file_vector_store.upsert_file_cards(indexed_file_cards)
        elif recreate:
            self.file_vector_store.clear_collection()

        summary = IndexBuildSummary(
            root_path=str(self.settings.harbor_root),
            collection=self.settings.qdrant_collection,
            indexed_projects=len(profiles),
            indexed_files=len(file_cards),
            scanned_directories=visited_directories,
            recreated_collection=recreate,
            generated_at=utc_now_iso(),
            file_collection=self.settings.qdrant_file_collection,
        )
        self.result_store.save_index_summary(summary)
        self.audit_store.record_index_run(summary)

        logger.info(
            "build_index_finished",
            extra={
                "indexed_projects": summary.indexed_projects,
                "indexed_files": summary.indexed_files,
                "skipped_file_cards": skipped_file_cards,
                "scanned_directories": summary.scanned_directories,
                "pdf_extraction_successes": pdf_stats.success_count,
                "pdf_extraction_failures": pdf_stats.failure_count,
            },
        )
        return summary

    def _build_file_cards(
        self,
        discovered_projects: list[DiscoveredProject],
    ) -> tuple[list[FileCard], int]:
        """Build file cards for supported files discovered in project directories."""

        file_cards: list[FileCard] = []
        skipped_file_cards = 0
        for project in discovered_projects:
            for file_path in self.crawler.list_project_files(project):
                try:
                    file_cards.append(self.card_builder.build_file_card(file_path))
                except Exception as exc:
                    skipped_file_cards += 1
                    logger.warning(
                        "file_card_build_skipped",
                        extra={
                            "file_path": str(file_path),
                            "error": str(exc),
                        },
                    )
        return file_cards, skipped_file_cards

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

    def _embed_file_cards_in_batches(self, file_cards: list[FileCard]) -> list[IndexedFileCard]:
        """Embed file cards in small batches to avoid Ollama timeouts."""

        total_cards = len(file_cards)
        batch_size = self.settings.embedding_batch_size
        total_batches = ceil(total_cards / batch_size)

        logger.info(
            "file_embedding_batches_started",
            extra={
                "total_cards": total_cards,
                "batch_size": batch_size,
                "total_batches": total_batches,
            },
        )

        indexed_cards: list[IndexedFileCard] = []
        for batch_number, card_batch in enumerate(chunked(file_cards, batch_size), start=1):
            logger.info(
                "file_embedding_batch_started",
                extra={
                    "batch_number": batch_number,
                    "total_batches": total_batches,
                    "batch_size": len(card_batch),
                },
            )
            try:
                embeddings = self.embedding_backend.embed_texts(
                    [card.text_for_embedding for card in card_batch]
                )
                batch_indexed_cards = [
                    IndexedFileCard(card=card, embedding=embedding)
                    for card, embedding in zip(card_batch, embeddings, strict=True)
                ]
            except (OllamaApiError, ValueError) as exc:
                logger.error(
                    "file_embedding_batch_failed",
                    extra={
                        "batch_number": batch_number,
                        "total_batches": total_batches,
                        "batch_size": len(card_batch),
                        "error": str(exc),
                    },
                )
                raise RuntimeError(
                    f"file embedding batch {batch_number}/{total_batches} failed after "
                    f"{self.settings.ollama_max_retries + 1} attempt(s)"
                ) from exc

            indexed_cards.extend(batch_indexed_cards)
            logger.info(
                "file_embedding_batch_finished",
                extra={
                    "batch_number": batch_number,
                    "total_batches": total_batches,
                    "batch_size": len(card_batch),
                    "embedded_cards": len(indexed_cards),
                    "total_cards": total_cards,
                },
            )

        return indexed_cards

    def query_file(self, file_path: Path, top_k: int | None = None) -> QueryResult:
        """Route a new file against the existing project index."""

        result, _query_context, _signal = self._run_query(file_path, top_k=top_k)
        return result

    def query_file_debug_payload(self, file_path: Path, top_k: int | None = None) -> dict[str, Any]:
        """Return query output augmented with debug profile information."""

        result, query_context, signal = self._run_query(file_path, top_k=top_k)
        payload = result.to_dict()
        payload["debug"] = self._query_debug_payload_data(result, query_context, signal)
        return payload

    def query_batch(
        self,
        input_path: Path,
        top_k: int | None = None,
        recursive: bool = True,
    ) -> BatchQueryResult:
        """Route a directory or a single file as a batch placement plan."""

        result, _debug_map = self._run_batch_query(input_path, top_k=top_k, recursive=recursive)
        return result

    def query_batch_debug_payload(
        self,
        input_path: Path,
        top_k: int | None = None,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """Return batch placement output augmented with per-file debug profiles."""

        result, debug_map = self._run_batch_query(input_path, top_k=top_k, recursive=recursive)
        payload = result.to_dict()
        for placement in payload["placements"]:
            debug_payload = debug_map.get(str(placement["source_path"]))
            if debug_payload:
                placement["debug"] = debug_payload
        for review_item in payload["review_queue"]:
            debug_payload = debug_map.get(str(review_item["source_path"]))
            if debug_payload:
                review_item["debug"] = debug_payload
        return payload

    def query(self, text: str, top_k: int | None = None) -> RetrievalResponse:
        """Run the hybrid retrieval core for a plain text query."""

        query_text = text.strip()
        if not query_text:
            raise ValueError("query text must not be empty")

        limit = top_k if top_k is not None else self.settings.top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than zero")

        if not self.vector_store.collection_exists():
            raise RuntimeError(
                f"Qdrant collection '{self.settings.qdrant_collection}' does not exist; "
                "run `build-index` first"
            )

        retrieval_query = RetrievalQuery(text=query_text, limit=limit)
        query_hints = self.query_hint_extractor.extract(query_text)
        retrieval_query.structured_hints = query_hints
        query_vector = self.embedding_backend.embed_text(query_text)
        retrieval_core = self.retrieval_core
        if not self.file_vector_store.collection_exists():
            retrieval_core = HybridRetrievalCore(
                folder_retriever=self.retriever,
                file_retriever=None,
                card_builder=self.card_builder,
                query_hint_extractor=self.query_hint_extractor,
            )

        response = retrieval_core.retrieve(retrieval_query, query_vector)
        self.result_store.save_retrieval_response(response)
        self.audit_store.record_retrieval_run(response)

        logger.info(
            "query_finished",
            extra={
                "query": query_text,
                "match_type": response.match_type,
                "match_count": len(response.matches),
                "needs_review": response.needs_review,
                "top_match_score": (
                    round(response.matches[0].score, 4) if response.matches else None
                ),
                "top_match_raw_score": (
                    round(response.matches[0].raw_score or 0.0, 4) if response.matches else None
                ),
                "query_hint_count": len(query_hints.normalized_terms),
                "query_time_hints": list(query_hints.time_hints),
                "query_technical_hints": list(query_hints.technical_hints),
            },
        )
        return response

    def record_feedback(
        self,
        result_id: str,
        *,
        feedback_status: str,
        feedback_reason: str,
        corrected_path: str | None = None,
        corrected_parent_path: str | None = None,
        notes: str | None = None,
    ) -> FeedbackRecord:
        """Persist a compact human feedback event for a stored result."""

        source = self.audit_store.lookup_feedback_source(result_id)
        if source is None:
            raise ValueError(f"unknown result_id: {result_id}")

        normalized_corrected_path = _normalize_optional_path(corrected_path)
        normalized_corrected_parent = _normalize_optional_path(corrected_parent_path)
        if normalized_corrected_parent is None and normalized_corrected_path is not None:
            normalized_corrected_parent = _path_parent(normalized_corrected_path)

        feedback = FeedbackRecord(
            source_result_id=result_id,
            query_kind=source.query_kind,
            feedback_status=feedback_status,
            feedback_reason=feedback_reason,
            corrected_path=normalized_corrected_path,
            corrected_parent_path=normalized_corrected_parent,
            notes=(notes.strip() if notes and notes.strip() else None),
            system_mode=source.system_mode,
            system_selected_path=source.system_selected_path,
            system_parent_path=source.system_parent_path,
            system_confidence=source.system_confidence,
            system_needs_review=source.system_needs_review,
            created_at=utc_now_iso(),
        )
        self.audit_store.record_feedback(feedback)
        logger.info(
            "feedback_recorded",
            extra={
                "feedback_id": feedback.feedback_id,
                "source_result_id": feedback.source_result_id,
                "query_kind": feedback.query_kind,
                "feedback_status": feedback.feedback_status,
                "feedback_reason": feedback.feedback_reason,
            },
        )
        return feedback

    def _run_query(
        self,
        file_path: Path,
        top_k: int | None = None,
    ) -> tuple[QueryResult, FileQueryContext, ExtractedSignal]:
        """Execute a query and return both result and extracted query context."""

        result, query_context, signal = self._compute_query_result(file_path, top_k=top_k)
        self.result_store.save_query_result(result)
        self.audit_store.record_query_run(result)

        logger.info(
            "query_file_finished",
            extra={
                "input_file": str(file_path),
                "candidate_count": len(result.top_candidates),
                "decision_mode": result.decision.mode,
                "signal_modality": signal.modality,
                "selected_project_id": result.decision.selected_project_id,
            },
        )
        return result, query_context, signal

    def _run_batch_query(
        self,
        input_path: Path,
        top_k: int | None,
        recursive: bool,
    ) -> tuple[BatchQueryResult, dict[str, dict[str, Any]]]:
        if not input_path.exists():
            raise FileNotFoundError(f"input path does not exist: {input_path}")

        limit = self._resolve_query_limit(top_k)
        file_paths, mode = self._collect_batch_file_paths(input_path, recursive=recursive)
        scanned_files = len(file_paths)

        skipped: list[BatchSkippedItem] = []
        supported_paths: list[Path] = []
        for file_path in file_paths:
            if self._supports_query_file(file_path):
                supported_paths.append(file_path)
            else:
                skipped.append(
                    BatchSkippedItem(
                        source_path=str(file_path),
                        reason="unsupported_extension",
                    )
                )

        if supported_paths:
            self._ensure_query_collection_exists()

        placements: list[BatchPlacement] = []
        review_queue: list[BatchReviewItem] = []
        debug_map: dict[str, dict[str, Any]] = {}
        planning_inputs: list[BatchPlanningInput] = []

        for file_path in supported_paths:
            try:
                result, query_context, signal = self._compute_query_result_for_limit(file_path, limit)
            except BatchFileProcessingError as exc:
                skipped.append(
                    BatchSkippedItem(
                        source_path=exc.source_path,
                        reason=exc.reason,
                        error=str(exc.error),
                    )
                )
                logger.warning(
                    "batch_query_file_skipped",
                    extra={
                        "source_path": exc.source_path,
                        "reason": exc.reason,
                        "error": str(exc.error),
                    },
                )
                continue

            placement = self._build_batch_placement(result)
            placements.append(placement)
            planning_inputs.append(
                BatchPlanningInput(
                    result=result,
                    query_context=query_context,
                    query_hints=self.query_hint_extractor.extract(
                        " ".join(
                            part
                            for part in [query_context.file_name, query_context.text_excerpt]
                            if part
                        )
                    ),
                )
            )
            if placement.needs_review:
                review_queue.append(
                    BatchReviewItem(
                        source_path=placement.source_path,
                        why=placement.why,
                        confidence=placement.confidence,
                        top_candidates=result.top_candidates,
                    )
                )
            debug_map[placement.source_path] = self._query_debug_payload_data(
                result,
                query_context,
                signal,
            )

        groups = self._group_batch_placements(placements)
        planning_result = plan_batch_placements(planning_inputs)
        group_mode_counts = Counter(
            group.decision.mode for group in planning_result.placement_groups
        )
        summary = BatchSummary(
            scanned_files=scanned_files,
            supported_files=len(supported_paths),
            classified=sum(1 for placement in placements if not placement.needs_review),
            needs_review=sum(1 for placement in placements if placement.needs_review),
            skipped=len(skipped),
            groups_total=len(planning_result.placement_groups),
            groups_existing_path=group_mode_counts.get("existing_path", 0),
            groups_existing_subpath=group_mode_counts.get("existing_subpath", 0),
            groups_proposed_new_subfolder=group_mode_counts.get("proposed_new_subfolder", 0),
            groups_review_needed=group_mode_counts.get("review_needed", 0),
        )
        batch_result = BatchQueryResult(
            input_path=str(input_path),
            mode=mode,
            summary=summary,
            placements=placements,
            groups=groups,
            review_queue=review_queue,
            placement_groups=planning_result.placement_groups,
            ungrouped_review_items=planning_result.ungrouped_review_items,
            skipped=skipped,
            generated_at=utc_now_iso(),
        )
        self.result_store.save_batch_query_result(batch_result)
        self.audit_store.record_batch_query_run(batch_result)

        logger.info(
            "query_batch_finished",
            extra={
                "input_path": str(input_path),
                "mode": mode,
                "scanned_files": scanned_files,
                "supported_files": len(supported_paths),
                "classified": summary.classified,
                "needs_review": summary.needs_review,
                "skipped": summary.skipped,
                "group_count": len(groups),
                "placement_group_count": summary.groups_total,
                "proposed_new_subfolder_groups": summary.groups_proposed_new_subfolder,
            },
        )
        return batch_result, debug_map

    def _compute_query_result(
        self,
        file_path: Path,
        top_k: int | None = None,
    ) -> tuple[QueryResult, FileQueryContext, ExtractedSignal]:
        limit = self._resolve_query_limit(top_k)
        return self._compute_query_result_for_limit(file_path, limit)

    def _compute_query_result_for_limit(
        self,
        file_path: Path,
        limit: int,
    ) -> tuple[QueryResult, FileQueryContext, ExtractedSignal]:
        query_context, signal = self._prepare_query_file(file_path)
        embedding = self.embedding_backend.embed_text(signal.text_for_embedding)
        candidates = self.retriever.retrieve(embedding, limit)
        query_hints = self.query_hint_extractor.extract(query_context.text_profile)
        candidates = rerank_folder_candidates(query_context.text_profile, query_hints, candidates)
        decision = self.decision_engine.decide(query_context, candidates)

        return (
            QueryResult(
                input_file=str(file_path),
                top_candidates=candidates,
                decision=decision,
                generated_at=utc_now_iso(),
            ),
            query_context,
            signal,
        )

    def _prepare_query_file(
        self,
        file_path: Path,
    ) -> tuple[FileQueryContext, ExtractedSignal]:
        self._validate_query_file_input(file_path)
        self._ensure_query_collection_exists()

        try:
            signal_extractor = self.signal_registry.resolve(file_path)
            signal = signal_extractor.extract(file_path)
        except (OSError, UnicodeError) as exc:
            raise BatchFileProcessingError(
                source_path=file_path,
                reason="unreadable_or_malformed",
                error=exc,
            ) from exc

        query_context = self.profile_builder.build_query_context_from_signal(file_path, signal)
        return query_context, signal

    def _resolve_query_limit(self, top_k: int | None) -> int:
        limit = top_k if top_k is not None else self.settings.top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than zero")
        return limit

    def _validate_query_file_input(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"input file does not exist: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"input path is not a file: {file_path}")

    def _ensure_query_collection_exists(self) -> None:
        if not self.vector_store.collection_exists():
            raise RuntimeError(
                f"Qdrant collection '{self.settings.qdrant_collection}' does not exist; "
                "run `build-index` first"
            )

    def _supports_query_file(self, file_path: Path) -> bool:
        return any(extractor.supports(file_path) for extractor in self.signal_registry.extractors)

    def _collect_batch_file_paths(
        self,
        input_path: Path,
        recursive: bool,
    ) -> tuple[list[Path], str]:
        if input_path.is_file():
            return [input_path], "single_file"
        if not input_path.is_dir():
            raise ValueError(f"input path is neither a file nor a directory: {input_path}")

        if recursive:
            file_paths: list[Path] = []
            for current_root, dirnames, filenames in os.walk(input_path, followlinks=False):
                dirnames[:] = sorted(dirnames)
                current_path = Path(current_root)
                file_paths.extend(current_path / filename for filename in sorted(filenames))
            return file_paths, "recursive"

        return (
            sorted(path for path in input_path.iterdir() if path.is_file()),
            "flat",
        )

    def _build_batch_placement(self, result: QueryResult) -> BatchPlacement:
        decision = result.decision
        needs_review = decision.mode == "review_needed" or decision.selected_path is None
        return BatchPlacement(
            source_path=result.input_file,
            selected_path=decision.selected_path,
            confidence=decision.confidence,
            needs_review=needs_review,
            why=self._batch_why(result),
            decision_mode=decision.mode,
            selected_project_id=decision.selected_project_id,
            top_candidates=list(result.top_candidates) if needs_review else [],
        )

    def _group_batch_placements(self, placements: list[BatchPlacement]) -> list[BatchGroup]:
        grouped: dict[str, list[BatchPlacement]] = defaultdict(list)
        for placement in placements:
            if placement.needs_review or not placement.selected_path:
                continue
            grouped[placement.selected_path].append(placement)

        groups: list[BatchGroup] = []
        for target_path in sorted(grouped):
            members = sorted(placement.source_path for placement in grouped[target_path])
            average_confidence = sum(
                placement.confidence for placement in grouped[target_path]
            ) / max(len(grouped[target_path]), 1)
            groups.append(
                BatchGroup(
                    suggested_target_path=target_path,
                    file_count=len(grouped[target_path]),
                    members=members,
                    average_confidence=average_confidence,
                )
            )
        return groups

    def _batch_why(self, result: QueryResult) -> str:
        if not result.top_candidates:
            return "no candidate folder matched the incoming file"

        decision = result.decision
        if decision.mode == "auto_top1" and decision.selected_path:
            return "top candidate score gate passed for the selected folder"
        if decision.mode == "llm_rerank" and decision.selected_path:
            return "reranking selected the most plausible folder among close candidates"
        if decision.reason == "llm_error_or_invalid_response":
            return "automatic routing was inconclusive and reranking failed"
        if decision.reason == "no_candidates_found":
            return "no candidate folder matched the incoming file"
        if decision.reason:
            return f"review requested: {decision.reason.replace('_', ' ')}"
        return "incoming file signal remains ambiguous across candidate folders"

    def _query_debug_payload_data(
        self,
        result: QueryResult,
        query_context: FileQueryContext,
        signal: ExtractedSignal,
    ) -> dict[str, Any]:
        return {
            "signal_modality": signal.modality,
            "signal_confidence": round(signal.confidence, 4),
            "query_text_excerpt": query_context.text_excerpt,
            "query_text_profile": query_context.text_profile,
            "candidate_text_profiles": [
                {
                    "project_id": candidate.project_id,
                    "path": candidate.path,
                    "score": round(candidate.score, 4),
                    "raw_score": round(
                        candidate.raw_score if candidate.raw_score is not None else candidate.score,
                        4,
                    ),
                    "semantic_bonus": round(candidate.semantic_bonus, 4),
                    "text_profile": candidate.text_profile,
                    "semantic_signature": (
                        candidate.semantic_signature.to_dict()
                        if candidate.semantic_signature is not None
                        else None
                    ),
                }
                for candidate in result.top_candidates
            ],
        }

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

        try:
            qdrant_file_info = self.file_vector_store.collection_info()
            report["checks"]["qdrant_local_files"] = {"ok": True, **qdrant_file_info}
        except Exception as exc:
            report["checks"]["qdrant_local_files"] = {
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


def _normalize_optional_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return None
    return str(Path(text))


def _path_parent(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.parent == path:
        return path_value
    return str(path.parent)
