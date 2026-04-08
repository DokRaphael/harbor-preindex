"""NAS crawler."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from harbor_preindex.schemas import DiscoveredProject


class ProjectCrawler:
    """Scan the NAS root and identify simple project directories."""

    def __init__(
        self,
        root_path: Path,
        sample_files_per_directory: int,
        supported_extensions: Sequence[str],
        excluded_path_segments: Sequence[str],
    ) -> None:
        self.root_path = root_path
        self.sample_files_per_directory = sample_files_per_directory
        self.supported_extensions = tuple(ext.lower() for ext in supported_extensions)
        self.excluded_path_segments = {segment.lower() for segment in excluded_path_segments}

    def scan_projects(self) -> tuple[list[DiscoveredProject], int]:
        """Return discovered project directories and total directories visited."""

        if not self.root_path.exists():
            raise FileNotFoundError(f"NAS root does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"NAS root is not a directory: {self.root_path}")

        discovered: list[DiscoveredProject] = []
        visited_directories = 0

        for current_root, dirnames, filenames in os.walk(self.root_path, followlinks=False):
            visited_directories += 1
            current_path = Path(current_root)
            if self._is_excluded_directory(current_path):
                dirnames[:] = []
                continue

            dirnames[:] = sorted(
                name
                for name in dirnames
                if not name.startswith(".") and name.lower() not in self.excluded_path_segments
            )

            supported_files = sorted(
                current_path / filename
                for filename in filenames
                if not filename.startswith(".") and self._is_supported(filename)
            )
            if current_path == self.root_path or not supported_files:
                continue

            relative_path = current_path.relative_to(self.root_path)
            discovered.append(
                DiscoveredProject(
                    path=current_path,
                    relative_path=relative_path,
                    sample_files=supported_files[: self.sample_files_per_directory],
                    doc_count=len(supported_files),
                )
            )

        return discovered, visited_directories

    def _is_supported(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.supported_extensions

    def _is_excluded_directory(self, path: Path) -> bool:
        try:
            relative_segments = path.relative_to(self.root_path).parts
        except ValueError:
            relative_segments = path.parts
        return any(segment.lower() in self.excluded_path_segments for segment in relative_segments)
