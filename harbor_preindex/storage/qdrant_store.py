"""Local Qdrant storage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from harbor_preindex.schemas import (
    FileSearchCandidate,
    FolderSemanticSignature,
    IndexedFileCard,
    IndexedProject,
    SearchCandidate,
)
from harbor_preindex.utils.qdrant_ids import make_qdrant_point_id


class QdrantProjectStore:
    """Persist project folder embeddings in local Qdrant."""

    def __init__(
        self,
        mode: str,
        path: Path,
        collection_name: str,
        client: QdrantClient | None = None,
    ) -> None:
        if mode != "local":
            raise ValueError(f"unsupported qdrant mode: {mode}")
        self.mode = mode
        self.path = path
        self.collection_name = collection_name
        self._client = client

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        client = self._client_instance()
        exists = client.collection_exists(self.collection_name)

        if exists and recreate:
            client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return

        existing_size = self._existing_vector_size()
        if existing_size is not None and existing_size != vector_size:
            raise RuntimeError(
                "existing Qdrant collection vector size does not match current embedding size; "
                "run `rescan` or `build-index --recreate`"
            )

    def clear_collection(self) -> None:
        """Delete the collection if it already exists."""

        client = self._client_instance()
        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)

    def collection_exists(self) -> bool:
        """Return whether the collection already exists."""

        return self._client_instance().collection_exists(self.collection_name)

    def upsert_projects(self, projects: Sequence[IndexedProject]) -> None:
        client = self._client_instance()
        points = [
            PointStruct(
                id=make_qdrant_point_id(project.profile.path),
                vector=project.embedding,
                payload=project.profile.to_payload(),
            )
            for project in projects
        ]
        if points:
            client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: Sequence[float], limit: int) -> list[SearchCandidate]:
        client = self._client_instance()
        response = client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results = _query_response_points(response)

        candidates: list[SearchCandidate] = []
        for item in results:
            payload = item.payload or {}
            candidates.append(
                SearchCandidate(
                    project_id=str(payload.get("project_id", item.id)),
                    path=str(payload.get("path", "")),
                    name=str(payload.get("name", "")),
                    parent=_optional_str(payload.get("parent")),
                    score=float(item.score or 0.0),
                    sample_filenames=_string_list(payload.get("sample_filenames")),
                    doc_count=int(payload.get("doc_count", 0)),
                    text_profile=str(payload.get("text_profile", "")),
                    semantic_signature=_folder_semantic_signature(payload.get("semantic_signature")),
                    raw_score=float(item.score or 0.0),
                )
            )
        return candidates

    def collection_info(self) -> dict[str, Any]:
        client = self._client_instance()
        exists = client.collection_exists(self.collection_name)
        info: dict[str, Any] = {
            "mode": self.mode,
            "path": str(self.path),
            "collection": self.collection_name,
            "exists": exists,
        }
        if exists:
            count = client.count(collection_name=self.collection_name, exact=True)
            info["points"] = int(count.count)
            info["vector_size"] = self._existing_vector_size()
        return info

    def _client_instance(self) -> QdrantClient:
        if self._client is None:
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.path))
        return self._client

    def _existing_vector_size(self) -> int | None:
        collection = self._client_instance().get_collection(self.collection_name)
        vectors = collection.config.params.vectors
        if vectors is None:
            return None
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()), None)
            if first is not None and hasattr(first, "size"):
                return int(first.size)
            return None
        size = getattr(vectors, "size", None)
        if size is not None:
            return int(size)
        return None


class QdrantFileStore:
    """Persist file card embeddings in local Qdrant."""

    def __init__(
        self,
        mode: str,
        path: Path,
        collection_name: str,
        client: QdrantClient | None = None,
    ) -> None:
        if mode != "local":
            raise ValueError(f"unsupported qdrant mode: {mode}")
        self.mode = mode
        self.path = path
        self.collection_name = collection_name
        self._client = client

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        client = self._client_instance()
        exists = client.collection_exists(self.collection_name)

        if exists and recreate:
            client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return

        existing_size = self._existing_vector_size()
        if existing_size is not None and existing_size != vector_size:
            raise RuntimeError(
                "existing Qdrant file collection vector size does not match current "
                "embedding size; run `rescan` or `build-index --recreate`"
            )

    def clear_collection(self) -> None:
        client = self._client_instance()
        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)

    def collection_exists(self) -> bool:
        return self._client_instance().collection_exists(self.collection_name)

    def upsert_file_cards(self, indexed_cards: Sequence[IndexedFileCard]) -> None:
        client = self._client_instance()
        points = [
            PointStruct(
                id=indexed.card.file_id,
                vector=indexed.embedding,
                payload=indexed.card.to_payload(),
            )
            for indexed in indexed_cards
        ]
        if points:
            client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: Sequence[float], limit: int) -> list[FileSearchCandidate]:
        client = self._client_instance()
        response = client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results = _query_response_points(response)

        candidates: list[FileSearchCandidate] = []
        for item in results:
            payload = item.payload or {}
            metadata = payload.get("metadata")
            candidates.append(
                FileSearchCandidate(
                    file_id=str(payload.get("file_id", item.id)),
                    path=str(payload.get("path", "")),
                    filename=str(payload.get("filename", "")),
                    extension=str(payload.get("extension", "")),
                    parent_path=str(payload.get("parent_path", "")),
                    modality=str(payload.get("modality", "document")),
                    score=float(item.score or 0.0),
                    text_for_embedding=str(payload.get("text_for_embedding", "")),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        return candidates

    def collection_info(self) -> dict[str, Any]:
        client = self._client_instance()
        exists = client.collection_exists(self.collection_name)
        info: dict[str, Any] = {
            "mode": self.mode,
            "path": str(self.path),
            "collection": self.collection_name,
            "exists": exists,
        }
        if exists:
            count = client.count(collection_name=self.collection_name, exact=True)
            info["points"] = int(count.count)
            info["vector_size"] = self._existing_vector_size()
        return info

    def _client_instance(self) -> QdrantClient:
        if self._client is None:
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.path))
        return self._client

    def _existing_vector_size(self) -> int | None:
        collection = self._client_instance().get_collection(self.collection_name)
        vectors = collection.config.params.vectors
        if vectors is None:
            return None
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()), None)
            if first is not None and hasattr(first, "size"):
                return int(first.size)
            return None
        size = getattr(vectors, "size", None)
        if size is not None:
            return int(size)
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _folder_semantic_signature(value: Any) -> FolderSemanticSignature | None:
    if not isinstance(value, dict):
        return None
    folder_role = value.get("folder_role")
    if not isinstance(folder_role, str):
        return None
    return FolderSemanticSignature(
        folder_role=folder_role,
        dominant_topics=_string_list(value.get("dominant_topics")),
        dominant_entities=_string_list(value.get("dominant_entities")),
        dominant_time_hints=_string_list(value.get("dominant_time_hints")),
        dominant_kinds=_string_list(value.get("dominant_kinds")),
        frequent_extensions=_string_list(value.get("frequent_extensions")),
        representative_terms=_string_list(value.get("representative_terms")),
        discriminative_terms=_string_list(value.get("discriminative_terms")),
        notable_children=_string_list(value.get("notable_children")),
        sample_filenames=_string_list(value.get("sample_filenames")),
    )


def _query_response_points(response: Any) -> list[Any]:
    points = getattr(response, "points", None)
    if isinstance(points, list):
        return points
    if isinstance(response, list):
        return response
    raise RuntimeError("unexpected Qdrant query response shape")


def create_local_qdrant_client(path: Path) -> QdrantClient:
    """Create a single local Qdrant client for a storage directory."""

    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))
