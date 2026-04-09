from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harbor_preindex.profiling.builder import ProjectProfileBuilder
from harbor_preindex.profiling.extraction import ContentExtractor
from harbor_preindex.profiling.folder_semantics import FolderSemanticSignatureBuilder
from harbor_preindex.semantic import (
    CodeSemanticEnricher,
    DocumentSemanticEnricher,
    SemanticEnricherRegistry,
)
from harbor_preindex.schemas import DiscoveredProject


class FolderSemanticsTests(unittest.TestCase):
    def test_builds_compact_folder_semantic_signature_from_sample_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "storage-root"
            tech_dir = root / "projects" / "max_dsp_docs"
            music_dir = root / "music" / "max_tabs"

            tech_file = tech_dir / "guide.txt"
            music_file = music_dir / "song.txt"
            self._write_text(
                tech_file,
                "MAX MSP DSP patch guide for Cycling74 signal routing and audio processing.",
            )
            self._write_text(
                music_file,
                "MAX tablature lyrics guitar chords chorus verse arrangement.",
            )

            semantic_registry = SemanticEnricherRegistry(
                [
                    CodeSemanticEnricher(),
                    DocumentSemanticEnricher(),
                ]
            )
            builder = ProjectProfileBuilder(
                root_path=root,
                extractor=ContentExtractor(max_chars=400),
                max_profile_chars=1200,
                folder_signature_builder=FolderSemanticSignatureBuilder(
                    semantic_registry=semantic_registry,
                    max_profile_chars=1200,
                ),
            )

            tech_profile = builder.build_project_profile(
                DiscoveredProject(
                    path=tech_dir,
                    relative_path=Path("projects/max_dsp_docs"),
                    sample_files=[tech_file],
                    doc_count=1,
                )
            )
            music_profile = builder.build_project_profile(
                DiscoveredProject(
                    path=music_dir,
                    relative_path=Path("music/max_tabs"),
                    sample_files=[music_file],
                    doc_count=1,
                )
            )

            assert tech_profile.semantic_signature is not None
            assert music_profile.semantic_signature is not None
            self.assertEqual(tech_profile.semantic_signature.folder_role, "leaf_specialized")
            self.assertIn("dsp", tech_profile.semantic_signature.discriminative_terms)
            self.assertIn("cycling74", tech_profile.semantic_signature.representative_terms)
            self.assertIn("tablature", music_profile.semantic_signature.discriminative_terms)
            self.assertIn("guitar", music_profile.semantic_signature.representative_terms)
            self.assertIn("Discriminative terms:", tech_profile.text_profile)
            self.assertIn("Folder role: leaf_specialized", tech_profile.text_profile)

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
