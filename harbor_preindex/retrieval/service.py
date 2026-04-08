"""Vector retrieval service."""

from __future__ import annotations

from collections.abc import Sequence

from harbor_preindex.schemas import FileSearchCandidate, SearchCandidate
from harbor_preindex.storage.qdrant_store import QdrantFileStore, QdrantProjectStore


class ProjectRetriever:
    """Retrieve top-k candidate project directories."""

    def __init__(self, store: QdrantProjectStore) -> None:
        self.store = store

    def retrieve(self, query_vector: Sequence[float], limit: int) -> list[SearchCandidate]:
        return self.store.search(query_vector, limit)


class FileCardRetriever:
    """Retrieve top-k candidate files."""

    def __init__(self, store: QdrantFileStore) -> None:
        self.store = store

    def retrieve(self, query_vector: Sequence[float], limit: int) -> list[FileSearchCandidate]:
        return self.store.search(query_vector, limit)
