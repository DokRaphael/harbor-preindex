from __future__ import annotations

from datetime import date
import unittest

from harbor_preindex.retrieval.core import HybridRetrievalCore
from harbor_preindex.retrieval.query_hints import QueryHintExtractor
from harbor_preindex.schemas import (
    FileSearchCandidate,
    FolderCard,
    RetrievalEvidence,
    RetrievalMatch,
    RetrievalQuery,
    RetrievalResponse,
    SearchCandidate,
)


class StubFolderRetriever:
    def __init__(self, candidates: list[SearchCandidate]) -> None:
        self.candidates = candidates

    def retrieve(self, query_vector: list[float], limit: int) -> list[SearchCandidate]:
        return self.candidates[:limit]


class StubFileRetriever:
    def __init__(self, candidates: list[FileSearchCandidate]) -> None:
        self.candidates = candidates

    def retrieve(self, query_vector: list[float], limit: int) -> list[FileSearchCandidate]:
        return self.candidates[:limit]


class StubCardBuilder:
    def build_folder_card_from_candidate(self, candidate: SearchCandidate) -> FolderCard:
        return FolderCard(
            folder_id=candidate.project_id,
            path=candidate.path,
            relative_path=candidate.path.split("/storage-root/")[-1],
            name=candidate.name,
            parent_path=None,
            text_for_embedding=candidate.text_profile,
            metadata={
                "doc_count": candidate.doc_count,
                "sample_filenames": candidate.sample_filenames,
            },
        )


class HybridRetrievalCoreTests(unittest.TestCase):
    def test_prefers_exact_file_when_filename_tokens_align(self) -> None:
        core = HybridRetrievalCore(
            folder_retriever=StubFolderRetriever(
                [
                    SearchCandidate(
                        project_id="admin_cv",
                        path="/tmp/storage-root/admin/cv",
                        name="cv",
                        parent="admin",
                        score=0.71,
                        sample_filenames=["Raphael_Dok_CV.txt"],
                        doc_count=1,
                        text_profile="CV folder",
                    )
                ]
            ),
            file_retriever=StubFileRetriever(
                [
                    FileSearchCandidate(
                        file_id="file-1",
                        path="/tmp/storage-root/admin/cv/Raphael_Dok_CV.txt",
                        filename="Raphael_Dok_CV.txt",
                        extension=".txt",
                        parent_path="/tmp/storage-root/admin/cv",
                        modality="document",
                        score=0.93,
                        text_for_embedding="CV text",
                        metadata={"text_excerpt": "curriculum vitae"},
                    )
                ]
            ),
            card_builder=StubCardBuilder(),
        )

        response = core.retrieve(RetrievalQuery(text="Raphael Dok CV", limit=5), [0.1, 0.2])

        self.assertEqual(response.match_type, "exact_file")
        self.assertFalse(response.needs_review)
        self.assertEqual(response.matches[0].target_kind, "file")
        self.assertEqual(response.matches[0].label, "Raphael_Dok_CV.txt")
        self.assertEqual(
            response.matches[0].why,
            "matched query terms in filename and semantic summary",
        )
        self.assertAlmostEqual(response.matches[0].raw_score or 0.0, 0.93, places=4)
        self.assertNotEqual(response.matches[0].score, response.matches[0].raw_score)

    def test_returns_folder_zone_when_only_folder_index_has_candidates(self) -> None:
        core = HybridRetrievalCore(
            folder_retriever=StubFolderRetriever(
                [
                    SearchCandidate(
                        project_id="admin_factures",
                        path="/tmp/storage-root/admin/factures",
                        name="factures",
                        parent="admin",
                        score=0.84,
                        sample_filenames=["amazon_invoice_2025.txt"],
                        doc_count=4,
                        text_profile="Amazon invoices",
                    )
                ]
            ),
            file_retriever=None,
            card_builder=StubCardBuilder(),
        )

        response = core.retrieve(
            RetrievalQuery(text="where are my Amazon invoices?", limit=5),
            [0.1, 0.2],
        )

        self.assertEqual(response.match_type, "folder_zone")
        self.assertFalse(response.needs_review)
        self.assertEqual(response.matches[0].target_kind, "folder")
        self.assertEqual(
            response.matches[0].why,
            "sample filenames and folder profile overlap with query",
        )

    def test_uses_calibrated_scores_instead_of_raw_cross_index_scores(self) -> None:
        core = HybridRetrievalCore(
            folder_retriever=StubFolderRetriever(
                [
                    SearchCandidate(
                        project_id="projects_neuraloop",
                        path="/tmp/storage-root/projects/neuraloop",
                        name="neuraloop",
                        parent="projects",
                        score=0.79,
                        sample_filenames=["spec.md"],
                        doc_count=3,
                        text_profile="Neuraloop docs",
                    )
                ]
            ),
            file_retriever=StubFileRetriever(
                [
                    FileSearchCandidate(
                        file_id="file-2",
                        path="/tmp/storage-root/projects/neuraloop/spec.md",
                        filename="spec.md",
                        extension=".md",
                        parent_path="/tmp/storage-root/projects/neuraloop",
                        modality="document",
                        score=0.8,
                        text_for_embedding="Neuraloop specification",
                        metadata={"text_excerpt": "Neuraloop retrieval pipeline"},
                    )
                ]
            ),
            card_builder=StubCardBuilder(),
        )

        response = core.retrieve(RetrievalQuery(text="find the Neuraloop docs", limit=5), [0.1])

        self.assertEqual(response.match_type, "mixed")
        self.assertTrue(response.needs_review)
        self.assertEqual([match.target_kind for match in response.matches[:2]], ["folder", "file"])
        self.assertLess(response.matches[0].raw_score or 0.0, response.matches[1].raw_score or 0.0)
        self.assertGreater(response.matches[0].score, response.matches[1].score)

    def test_document_match_exposes_compact_semantic_evidence(self) -> None:
        core = HybridRetrievalCore(
            folder_retriever=StubFolderRetriever([]),
            file_retriever=StubFileRetriever(
                [
                    FileSearchCandidate(
                        file_id="file-3",
                        path="/tmp/storage-root/admin/factures/amazon_invoice_2025.txt",
                        filename="amazon_invoice_2025.txt",
                        extension=".txt",
                        parent_path="/tmp/storage-root/admin/factures",
                        modality="document",
                        score=0.89,
                        text_for_embedding="Amazon invoice 2025",
                        metadata={
                            "text_excerpt": "Amazon order invoice total EUR 23.90 2025",
                            "functional_summary": "Transactional document with named entities such as Amazon and time hints such as 2025.",
                            "semantic_hints": {
                                "topic_hints": ["amazon", "invoice"],
                                "entity_candidates": ["Amazon"],
                                "time_hints": ["2025"],
                            },
                        },
                    )
                ]
            ),
            card_builder=StubCardBuilder(),
        )

        response = core.retrieve(
            RetrievalQuery(text="where are my Amazon invoices from 2025?", limit=5),
            [0.1, 0.2],
        )

        match = response.matches[0]
        self.assertEqual(match.why, "entity candidate and time hint align with the query")
        self.assertIsNotNone(match.evidence)
        assert match.evidence is not None
        self.assertEqual(match.evidence.matched_entity_candidates, ["Amazon"])
        self.assertEqual(match.evidence.matched_time_hints, ["2025"])
        self.assertIn("filename", match.evidence.matched_sources)
        self.assertIn("semantic_hints", match.evidence.matched_sources)
        self.assertIn("amazon", match.evidence.source_terms["filename"])
        self.assertIn("2025", match.evidence.source_terms["semantic_hints"])

    def test_filename_segments_help_explain_compound_document_names(self) -> None:
        core = HybridRetrievalCore(
            folder_retriever=StubFolderRetriever([]),
            file_retriever=StubFileRetriever(
                [
                    FileSearchCandidate(
                        file_id="file-compound",
                        path="/tmp/storage-root/admin/factures/ESP8266amazon.pdf",
                        filename="ESP8266amazon.pdf",
                        extension=".pdf",
                        parent_path="/tmp/storage-root/admin/factures",
                        modality="document",
                        score=0.83,
                        text_for_embedding="Amazon invoice for ESP8266 order",
                        metadata={
                            "filename_terms": ["ESP8266", "amazon"],
                            "text_excerpt": "Amazon invoice for an ESP8266 order",
                            "functional_summary": "Transactional document mentioning Amazon.",
                            "semantic_hints": {
                                "kind_hints": ["transactional_document"],
                                "topic_hints": ["amazon", "invoice"],
                                "entity_candidates": ["Amazon"],
                                "time_hints": [],
                            },
                        },
                    )
                ]
            ),
            card_builder=StubCardBuilder(),
        )

        response = core.retrieve(RetrievalQuery(text="amazon invoice", limit=5), [0.1, 0.2])

        match = response.matches[0]
        self.assertEqual(match.why, "matched query terms in filename and semantic summary")
        assert match.evidence is not None
        self.assertIn("filename", match.evidence.matched_sources)
        self.assertIn("amazon", match.evidence.source_terms["filename"])

    def test_query_hints_lightly_rerank_results_without_hard_filtering(self) -> None:
        core = HybridRetrievalCore(
            folder_retriever=StubFolderRetriever(
                [
                    SearchCandidate(
                        project_id="projects_neuraloop",
                        path="/tmp/storage-root/projects/neuraloop",
                        name="neuraloop",
                        parent="projects",
                        score=0.8,
                        sample_filenames=["overview.txt"],
                        doc_count=4,
                        text_profile="Neuraloop docs folder",
                    )
                ]
            ),
            file_retriever=StubFileRetriever(
                [
                    FileSearchCandidate(
                        file_id="file-5",
                        path="/tmp/storage-root/projects/neuraloop/spec_2024.md",
                        filename="spec_2024.md",
                        extension=".md",
                        parent_path="/tmp/storage-root/projects/neuraloop",
                        modality="document",
                        score=0.81,
                        text_for_embedding="Neuraloop spec 2024",
                        metadata={
                            "text_excerpt": "Neuraloop specification updated in 2024",
                            "functional_summary": "Technical document with time hints such as 2024.",
                            "semantic_hints": {
                                "kind_hints": ["technical_document"],
                                "topic_hints": ["neuraloop", "spec"],
                                "entity_candidates": ["Neuraloop"],
                                "time_hints": ["2024"],
                            },
                        },
                    )
                ]
            ),
            card_builder=StubCardBuilder(),
            query_hint_extractor=QueryHintExtractor(today=date(2026, 4, 9)),
        )

        response = core.retrieve(
            RetrievalQuery(text="find the Neuraloop docs from 2024", limit=5),
            [0.1, 0.2],
        )

        self.assertEqual(response.matches[0].target_kind, "file")
        self.assertGreater(response.matches[0].score, response.matches[1].score)
        self.assertEqual(response.matches[0].why, "entity candidate and time hint align with the query")
        self.assertIsNotNone(response.matches[0].evidence)
        assert response.matches[0].evidence is not None
        self.assertIn(
            "multi-hint coverage increased ranking confidence",
            " ".join(response.matches[0].evidence.notes),
        )
        self.assertEqual(response.matches[0].evidence.matched_time_hints, ["2024"])

    def test_multi_hint_coverage_beats_time_only_alignment(self) -> None:
        core = HybridRetrievalCore(
            folder_retriever=StubFolderRetriever(
                [
                    SearchCandidate(
                        project_id="folder-time-only",
                        path="/tmp/storage-root/admin/archive/2018",
                        name="2018",
                        parent="archive",
                        score=0.84,
                        sample_filenames=["scan_2018.pdf"],
                        doc_count=8,
                        text_profile="Administrative archive 2018",
                    )
                ]
            ),
            file_retriever=StubFileRetriever(
                [
                    FileSearchCandidate(
                        file_id="file-multi",
                        path="/tmp/storage-root/admin/factures/amazon_invoice_2018.pdf",
                        filename="amazon_invoice_2018.pdf",
                        extension=".pdf",
                        parent_path="/tmp/storage-root/admin/factures",
                        modality="document",
                        score=0.8,
                        text_for_embedding="Amazon invoice 2018",
                        metadata={
                            "text_excerpt": "Amazon invoice for ESP8266 order in 2018",
                            "functional_summary": "Transactional document with named entities such as Amazon and time hints such as 2018.",
                            "semantic_hints": {
                                "kind_hints": ["transactional_document"],
                                "topic_hints": ["amazon", "invoice"],
                                "entity_candidates": ["Amazon"],
                                "time_hints": ["2018"],
                            },
                        },
                    ),
                ]
            ),
            card_builder=StubCardBuilder(),
            query_hint_extractor=QueryHintExtractor(today=date(2026, 4, 9)),
        )

        response = core.retrieve(RetrievalQuery(text="Amazon invoice from 2018", limit=5), [0.1, 0.2])

        self.assertEqual(response.matches[0].target_id, "file-multi")
        self.assertIn(
            "multi-hint coverage increased ranking confidence",
            " ".join(response.matches[0].evidence.notes if response.matches[0].evidence else []),
        )
        self.assertIn(
            "time hint matched with no other strong hint; conservative adjustment applied",
            " ".join(response.matches[1].evidence.notes if response.matches[1].evidence else []),
        )

    def test_specific_document_query_keeps_folder_time_bonus_conservative(self) -> None:
        core = HybridRetrievalCore(
            folder_retriever=StubFolderRetriever(
                [
                    SearchCandidate(
                        project_id="folder-2018",
                        path="/tmp/storage-root/admin/bulletindepaie/Snips/2018",
                        name="2018",
                        parent="Snips",
                        score=0.85,
                        sample_filenames=["scan_2018.pdf"],
                        doc_count=12,
                        text_profile="Administrative archive 2018",
                    )
                ]
            ),
            file_retriever=StubFileRetriever(
                [
                    FileSearchCandidate(
                        file_id="file-amazon-2018",
                        path="/tmp/storage-root/liloFactures/ESP8266amazon.pdf",
                        filename="ESP8266amazon.pdf",
                        extension=".pdf",
                        parent_path="/tmp/storage-root/liloFactures",
                        modality="document",
                        score=0.79,
                        text_for_embedding="Amazon invoice ESP8266 2018",
                        metadata={
                            "text_excerpt": "Amazon invoice for ESP8266 order in 2018",
                            "functional_summary": "Transactional document with named entities such as Amazon and time hints such as 2018.",
                            "semantic_hints": {
                                "kind_hints": ["transactional_document"],
                                "topic_hints": ["amazon", "invoice"],
                                "entity_candidates": ["Amazon"],
                                "time_hints": ["2018"],
                            },
                        },
                    )
                ]
            ),
            card_builder=StubCardBuilder(),
            query_hint_extractor=QueryHintExtractor(today=date(2026, 4, 9)),
        )

        response = core.retrieve(RetrievalQuery(text="facture amazon de 2018", limit=5), [0.1, 0.2])

        self.assertEqual(response.matches[0].target_kind, "file")
        self.assertEqual(response.matches[0].target_id, "file-amazon-2018")
        self.assertIn(
            "file precision preference applied for specific document-like query",
            " ".join(response.matches[0].evidence.notes if response.matches[0].evidence else []),
        )
        self.assertIn(
            "time hint matched with no other strong hint; conservative adjustment applied",
            " ".join(response.matches[1].evidence.notes if response.matches[1].evidence else []),
        )

    def test_code_match_exposes_import_and_symbol_evidence(self) -> None:
        core = HybridRetrievalCore(
            folder_retriever=StubFolderRetriever([]),
            file_retriever=StubFileRetriever(
                [
                    FileSearchCandidate(
                        file_id="file-4",
                        path="/tmp/storage-root/projects/neuraloop/qdrant_cli.py",
                        filename="qdrant_cli.py",
                        extension=".py",
                        parent_path="/tmp/storage-root/projects/neuraloop",
                        modality="code",
                        score=0.87,
                        text_for_embedding="Qdrant CLI module",
                        metadata={
                            "functional_summary": "Python code module covering vector storage, retrieval, and CLI flows.",
                            "semantic_hints": {
                                "topic_hints": ["vector_storage", "retrieval", "cli", "qdrant"],
                                "entity_candidates": [],
                                "time_hints": [],
                            },
                            "imports": ["qdrant_client", "argparse"],
                            "symbols": ["build_parser", "query_qdrant"],
                        },
                    )
                ]
            ),
            card_builder=StubCardBuilder(),
        )

        response = core.retrieve(
            RetrievalQuery(text="where is the code that talks to qdrant?", limit=5),
            [0.1, 0.2],
        )

        match = response.matches[0]
        self.assertEqual(match.why, "technical hints and code imports align with the query")
        self.assertIsNotNone(match.evidence)
        assert match.evidence is not None
        self.assertEqual(match.evidence.matched_imports, ["qdrant_client"])
        self.assertIn("semantic_hints", match.evidence.matched_sources)
        self.assertIn("imports", match.evidence.matched_sources)
        self.assertIn("qdrant", match.evidence.source_terms["imports"])
        self.assertEqual(match.evidence.matched_technical_hints, ["vector_storage"])

    def test_retrieval_contract_rejects_unknown_literals(self) -> None:
        with self.assertRaises(ValueError):
            RetrievalMatch(
                target_kind="project",
                target_id="bad",
                path="/tmp/bad",
                score=0.5,
                label="bad",
                why="bad",
            )

        valid_match = RetrievalMatch(
            target_kind="file",
            target_id="file-1",
            path="/tmp/file.txt",
            score=0.7,
            label="file.txt",
            why="ok",
            evidence=RetrievalEvidence(
                matched_query_terms=["file"],
                matched_sources=["filename"],
                source_terms={"filename": ["file"]},
            ),
            raw_score=0.8,
            decision_score=0.7,
        )

        with self.assertRaises(ValueError):
            RetrievalResponse(
                query="broken",
                match_type="project",
                matches=[valid_match],
                confidence=0.7,
                needs_review=False,
                generated_at="2026-04-08T00:00:00Z",
            )

        self.assertEqual(
            valid_match.to_dict(),
            {
                "target_kind": "file",
                "target_id": "file-1",
                "path": "/tmp/file.txt",
                "score": 0.7,
                "label": "file.txt",
                "why": "ok",
            },
        )
        self.assertEqual(
            valid_match.to_dict(include_evidence=True),
            {
                "target_kind": "file",
                "target_id": "file-1",
                "path": "/tmp/file.txt",
                "score": 0.7,
                "label": "file.txt",
                "why": "ok",
                "evidence": {
                    "matched_query_terms": ["file"],
                    "matched_sources": ["filename"],
                    "source_terms": {"filename": ["file"]},
                },
            },
        )
        response = RetrievalResponse(
            query="file",
            match_type="likely_file",
            matches=[valid_match],
            confidence=0.7,
            needs_review=False,
            generated_at="2026-04-08T00:00:00Z",
        )
        self.assertNotIn("evidence", response.to_dict()["matches"][0])
        self.assertIn("evidence", response.to_dict(include_evidence=True)["matches"][0])

    def test_debug_output_includes_query_hints_but_standard_output_stays_compact(self) -> None:
        core = HybridRetrievalCore(
            folder_retriever=StubFolderRetriever([]),
            file_retriever=StubFileRetriever(
                [
                    FileSearchCandidate(
                        file_id="file-6",
                        path="/tmp/storage-root/projects/neuraloop/qdrant_cli.py",
                        filename="qdrant_cli.py",
                        extension=".py",
                        parent_path="/tmp/storage-root/projects/neuraloop",
                        modality="code",
                        score=0.86,
                        text_for_embedding="Qdrant CLI module",
                        metadata={
                            "functional_summary": "Python code module covering vector storage and CLI flows.",
                            "semantic_hints": {
                                "kind_hints": ["code_artifact"],
                                "topic_hints": ["vector_storage", "cli"],
                                "entity_candidates": [],
                                "time_hints": [],
                            },
                            "imports": ["qdrant_client"],
                            "symbols": ["build_parser"],
                        },
                    )
                ]
            ),
            card_builder=StubCardBuilder(),
            query_hint_extractor=QueryHintExtractor(today=date(2026, 4, 9)),
        )

        response = core.retrieve(
            RetrievalQuery(text="where is the code that talks to qdrant?", limit=5),
            [0.1, 0.2],
        )

        self.assertNotIn("query_hints", response.to_dict())
        debug_payload = response.to_dict(include_evidence=True)
        self.assertIn("query_hints", debug_payload)
        self.assertIn("technical_hints", debug_payload["query_hints"])
        self.assertNotIn("evidence", response.to_dict()["matches"][0])
        self.assertIn("evidence", debug_payload["matches"][0])


if __name__ == "__main__":
    unittest.main()
