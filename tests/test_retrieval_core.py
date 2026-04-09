from __future__ import annotations

import unittest

from harbor_preindex.retrieval.core import HybridRetrievalCore
from harbor_preindex.schemas import (
    FileSearchCandidate,
    FolderCard,
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


if __name__ == "__main__":
    unittest.main()
