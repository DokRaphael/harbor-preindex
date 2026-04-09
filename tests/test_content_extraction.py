from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")

    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False

    dotenv_module.load_dotenv = load_dotenv
    sys.modules["dotenv"] = dotenv_module

from harbor_preindex.profiling.extraction import ContentExtractor
from harbor_preindex.retrieval.cards import RetrievalCardBuilder
from harbor_preindex.semantic import (
    CodeSemanticEnricher,
    DocumentSemanticEnricher,
    SemanticEnricherRegistry,
)
from harbor_preindex.settings import DEFAULT_SUPPORTED_EXTENSIONS
from harbor_preindex.signals.document import DocumentSignalExtractor
from harbor_preindex.signals.registry import SignalExtractorRegistry


class ContentExtractionPhaseOneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name) / "storage-root"
        self.root_path.mkdir(parents=True, exist_ok=True)

        self.extractor = ContentExtractor(max_chars=400)
        self.signal_registry = SignalExtractorRegistry(
            [
                DocumentSignalExtractor(
                    extractor=self.extractor,
                    supported_extensions=DEFAULT_SUPPORTED_EXTENSIONS,
                    max_profile_chars=600,
                )
            ]
        )
        self.semantic_registry = SemanticEnricherRegistry(
            [
                CodeSemanticEnricher(),
                DocumentSemanticEnricher(),
            ]
        )
        self.builder = RetrievalCardBuilder(
            root_path=self.root_path,
            signal_registry=self.signal_registry,
            semantic_registry=self.semantic_registry,
            max_profile_chars=600,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_supported_extensions_cover_phase_one_formats(self) -> None:
        required_extensions = {
            ".txt",
            ".md",
            ".pdf",
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".sh",
            ".bash",
            ".zsh",
            ".sql",
            ".html",
            ".css",
            ".xml",
        }

        self.assertTrue(required_extensions.issubset(set(DEFAULT_SUPPORTED_EXTENSIONS)))

    def test_python_file_builds_a_code_card(self) -> None:
        file_path = self._write_text(
            "projects/sample_project/vector_client.py",
            """import argparse
import sqlite3


def build_parser() -> None:
    \"\"\"Build the parser for the local client.\"\"\"
    return None
""",
        )

        card = self.builder.build_file_card(file_path)

        self.assertEqual(card.modality, "code")
        self.assertEqual(card.metadata["semantic_hints"]["language_hint"], "python")
        self.assertIn("Language hint: python", card.text_for_embedding)
        self.assertIn("build_parser", card.metadata["symbols"])

    def test_typescript_file_is_supported_and_builds_a_code_card(self) -> None:
        file_path = self._write_text(
            "projects/sample_project/api_client.ts",
            """import { createClient } from "qdrant";

export async function queryIndex(): Promise<void> {
  return;
}
""",
        )

        card = self.builder.build_file_card(file_path)

        self.assertEqual(card.modality, "code")
        self.assertEqual(card.metadata["semantic_hints"]["language_hint"], "typescript")
        self.assertIn("code_artifact", card.metadata["semantic_hints"]["kind_hints"])
        self.assertIn("Language hint: typescript", card.text_for_embedding)

    def test_json_and_yaml_configs_produce_useful_compact_signals(self) -> None:
        json_path = self._write_text(
            "projects/sample_project/service_config.json",
            """{
  "service": {
    "name": "harbor",
    "backends": ["local", "cache"]
  },
  "features": {
    "retrieval": true
  }
}""",
        )
        yaml_path = self._write_text(
            "projects/sample_project/service.yaml",
            """service:
  name: harbor
  mode: local
features:
  retrieval: true
""",
        )

        json_excerpt = self.extractor.extract_excerpt(json_path)
        yaml_excerpt = self.extractor.extract_excerpt(yaml_path)
        yaml_card = self.builder.build_file_card(yaml_path)

        self.assertIn("service.name: harbor", json_excerpt)
        self.assertIn("service.backends: local, cache", json_excerpt)
        self.assertIn("service.name: harbor", yaml_excerpt)
        self.assertIn("features.retrieval: true", yaml_excerpt)
        self.assertEqual(yaml_card.modality, "code")
        self.assertIn(
            "configuration_artifact",
            yaml_card.metadata["semantic_hints"]["kind_hints"],
        )

    def test_text_extraction_uses_a_reasonable_encoding_fallback(self) -> None:
        file_path = self.root_path / "projects" / "sample_project" / "legacy.conf"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes("display_name = café".encode("cp1252"))

        excerpt = self.extractor.extract_excerpt(file_path)

        self.assertIn("display_name: café", excerpt)

    def test_malformed_json_falls_back_to_plain_text_excerpt(self) -> None:
        file_path = self._write_text(
            "projects/sample_project/broken.json",
            '{"service": {"name": "harbor", "mode": "local"',
        )

        excerpt = self.extractor.extract_excerpt(file_path)

        self.assertIn('"service"', excerpt)
        self.assertIn('"name"', excerpt)

    def _write_text(self, relative_path: str, content: str) -> Path:
        file_path = self.root_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return file_path


if __name__ == "__main__":
    unittest.main()
