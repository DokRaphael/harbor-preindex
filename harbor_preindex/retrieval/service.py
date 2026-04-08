"""Vector retrieval service."""

from __future__ import annotations

from collections.abc import Sequence

from harbor_preindex.schemas import SearchCandidate
from harbor_preindex.storage.qdrant_store import QdrantProjectStore


class ProjectRetriever:
    """Retrieve top-k candidate project directories."""

    def __init__(self, store: QdrantProjectStore) -> None:
        self.store = store

    def retrieve(self, query_vector: Sequence[float], limit: int) -> list[SearchCandidate]:
        return self.store.search(query_vector, limit)
