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
from harbor_preindex.signals.document import DocumentSignalExtractor
from harbor_preindex.signals.registry import SignalExtractorRegistry


class SemanticEnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root_path = Path(__file__).parent / "fixtures" / "storage-root"
        self.extractor = ContentExtractor(max_chars=600)
        self.signal_registry = SignalExtractorRegistry(
            [
                DocumentSignalExtractor(
                    extractor=self.extractor,
                    supported_extensions=(".txt", ".md", ".pdf", ".py"),
                    max_profile_chars=1000,
                )
            ]
        )
        self.semantic_registry = SemanticEnricherRegistry(
            [
                CodeSemanticEnricher(),
                DocumentSemanticEnricher(),
            ]
        )

    def test_document_semantic_enricher_adds_general_transactional_hints(self) -> None:
        file_path = self.root_path / "admin" / "factures" / "amazon_invoice_2025.txt"

        signal = self.signal_registry.resolve(file_path).extract(file_path)
        enriched = self.semantic_registry.enrich(file_path, signal)

        self.assertEqual(enriched.modality, "document")
        self.assertIn("transactional_document", enriched.semantic_hints.kind_hints)
        self.assertIn("contains_monetary_values", enriched.semantic_hints.structure_hints)
        self.assertIn("2025", enriched.semantic_hints.time_hints)
        self.assertIn("amazon", enriched.semantic_hints.topic_hints)
        self.assertTrue(enriched.semantic_hints.functional_summary)

    def test_code_semantic_enricher_builds_functional_abstraction(self) -> None:
        file_path = self.root_path / "projects" / "neuraloop" / "qdrant_cli.py"

        signal = self.signal_registry.resolve(file_path).extract(file_path)
        enriched = self.semantic_registry.enrich(file_path, signal)

        self.assertEqual(enriched.modality, "code")
        self.assertEqual(enriched.semantic_hints.language_hint, "python")
        self.assertIn("vector_storage", enriched.semantic_hints.topic_hints)
        self.assertIn("cli", enriched.semantic_hints.topic_hints)
        self.assertIn("qdrant_client", enriched.metadata["imports"])
        self.assertIn("build_parser", enriched.metadata["symbols"])
        self.assertIn("defines_functions", enriched.semantic_hints.structure_hints)
        self.assertTrue(enriched.semantic_hints.functional_summary)

    def test_file_card_embedding_text_is_richer_but_compact_for_code(self) -> None:
        builder = RetrievalCardBuilder(
            root_path=self.root_path,
            signal_registry=self.signal_registry,
            semantic_registry=self.semantic_registry,
            max_profile_chars=520,
        )
        file_path = self.root_path / "projects" / "neuraloop" / "qdrant_cli.py"

        card = builder.build_file_card(file_path)

        self.assertEqual(card.modality, "code")
        self.assertIn("Language hint: python", card.text_for_embedding)
        self.assertIn("Topic hints:", card.text_for_embedding)
        self.assertIn("Functional summary:", card.text_for_embedding)
        self.assertIn("semantic_hints", card.metadata)
        self.assertLessEqual(len(card.text_for_embedding), 520)


if __name__ == "__main__":
    unittest.main()
