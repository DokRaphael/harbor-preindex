from __future__ import annotations

import unittest
from pathlib import Path

from harbor_preindex.profiling.extraction import ContentExtractor
from harbor_preindex.retrieval.cards import RetrievalCardBuilder
from harbor_preindex.semantic import (
    CodeSemanticEnricher,
    DocumentSemanticEnricher,
    SemanticEnricherRegistry,
)
from harbor_preindex.schemas import ProjectProfile
from harbor_preindex.signals.document import DocumentSignalExtractor
from harbor_preindex.signals.registry import SignalExtractorRegistry


class RetrievalCardBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root_path = Path(__file__).parent / "fixtures" / "storage-root"
        extractor = ContentExtractor(max_chars=400)
        registry = SignalExtractorRegistry(
            [
                DocumentSignalExtractor(
                    extractor=extractor,
                    supported_extensions=(".txt", ".md", ".pdf", ".py"),
                    max_profile_chars=600,
                )
            ]
        )
        semantic_registry = SemanticEnricherRegistry(
            [
                CodeSemanticEnricher(),
                DocumentSemanticEnricher(),
            ]
        )
        self.builder = RetrievalCardBuilder(
            root_path=self.root_path,
            signal_registry=registry,
            semantic_registry=semantic_registry,
            max_profile_chars=600,
        )

    def test_build_file_card_includes_core_fields_and_text(self) -> None:
        file_path = self.root_path / "admin" / "cv" / "Raphael_Dok_CV.txt"

        first = self.builder.build_file_card(file_path)
        second = self.builder.build_file_card(file_path)

        self.assertEqual(first.file_id, second.file_id)
        self.assertEqual(first.filename, "Raphael_Dok_CV.txt")
        self.assertEqual(first.extension, ".txt")
        self.assertEqual(first.parent_path, str(file_path.parent))
        self.assertEqual(first.metadata["relative_path"], "admin/cv/Raphael_Dok_CV.txt")
        self.assertIn("File name: Raphael_Dok_CV.txt", first.text_for_embedding)
        self.assertIn("Parent path: admin/cv", first.text_for_embedding)
        self.assertIn("Kind hints:", first.text_for_embedding)
        self.assertIn("Functional summary:", first.text_for_embedding)
        self.assertIn("semantic_hints", first.metadata)
        self.assertLessEqual(len(first.text_for_embedding), 600)

    def test_build_folder_card_from_profile_keeps_existing_contract(self) -> None:
        profile = ProjectProfile(
            project_id="admin_cv",
            path=str(self.root_path / "admin" / "cv"),
            relative_path="admin/cv",
            name="cv",
            parent="admin",
            sample_filenames=["Raphael_Dok_CV.txt"],
            doc_count=1,
            text_profile="Folder profile text",
        )

        card = self.builder.build_folder_card(profile)

        self.assertEqual(card.folder_id, "admin_cv")
        self.assertEqual(card.relative_path, "admin/cv")
        self.assertEqual(card.name, "cv")
        self.assertEqual(card.metadata["doc_count"], 1)
        self.assertEqual(card.metadata["sample_filenames"], ["Raphael_Dok_CV.txt"])


if __name__ == "__main__":
    unittest.main()
