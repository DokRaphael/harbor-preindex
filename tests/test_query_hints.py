from __future__ import annotations

from datetime import date
import unittest

from harbor_preindex.retrieval.query_hints import QueryHintExtractor


class QueryHintExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = QueryHintExtractor(today=date(2026, 4, 9))

    def test_extracts_transactional_and_relative_time_hints(self) -> None:
        hints = self.extractor.extract("where is my plumber invoice from last year?")

        self.assertIn("plumber", hints.entity_terms)
        self.assertIn("transactional_document", hints.kind_hints)
        self.assertIn("relative:last_year", hints.time_hints)
        self.assertIn("2025", hints.time_hints)
        self.assertIn("invoice", hints.topic_hints)

    def test_extracts_explicit_year_and_named_entity_for_docs_query(self) -> None:
        hints = self.extractor.extract("find the Neuraloop docs from 2024")

        self.assertIn("2024", hints.time_hints)
        self.assertIn("neuraloop", hints.entity_terms)
        self.assertIn("technical_document", hints.kind_hints)
        self.assertEqual(hints.intent_hint, "document_lookup")

    def test_extracts_transactional_hint_for_french_invoice_query(self) -> None:
        hints = self.extractor.extract("facture amazon de 2018")

        self.assertIn("transactional_document", hints.kind_hints)
        self.assertIn("amazon", hints.entity_terms)
        self.assertIn("2018", hints.time_hints)

    def test_extracts_technical_hints_for_code_queries(self) -> None:
        hints = self.extractor.extract("where is the code that talks to qdrant?")

        self.assertIn("code_artifact", hints.kind_hints)
        self.assertIn("vector_storage", hints.technical_hints)
        self.assertEqual(hints.intent_hint, "code_lookup")
        self.assertIn("qdrant", hints.entity_terms)


if __name__ == "__main__":
    unittest.main()
