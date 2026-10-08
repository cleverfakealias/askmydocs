"""Chroma-backed storage for document chunks and their embeddings."""

import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.api.types import Metadata, Where
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from palimpsest.models import IndexedSource

COLLECTION_NAME = "palimpsest"


class VectorStore:
    """Stores chunks on disk and answers similarity and MMR searches.

    Every chunk carries a `source_id` (a stable key for its file) and a `source` (the
    file name shown to people).
    """

    def __init__(self, persist_directory: Path, embeddings: Embeddings) -> None:
        """Open or create the collection in `persist_directory`.

        Args:
            persist_directory: Folder for the Chroma database.
            embeddings: The embedder used for stored chunks and for queries.
        """
        # Chroma sends anonymous usage telemetry unless it is turned off.
        self._client = chromadb.PersistentClient(
            path=str(persist_directory), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._db = Chroma(
            client=self._client, collection_name=COLLECTION_NAME, embedding_function=embeddings
        )

    def replace_source(self, source_id: str, documents: Sequence[Document]) -> int:
        """Store new chunks for one file, then remove its older chunks.

        The new chunks are written first, so a failed write leaves the older version
        searchable.

        Args:
            source_id: The file's stable key.
            documents: The file's new chunks.

        Returns:
            How many chunks were stored.

        Raises:
            Exception: Any error from the embedder or from Chroma. Partly written chunks
                are removed before the error is raised again.
        """
        old_ids = self._ids_where({"source_id": source_id})
        new_ids = [str(uuid.uuid7()) for _ in documents]
        if not new_ids:
            return 0
        try:
            self._db.add_documents(list(documents), ids=new_ids)
        except Exception:
            self._delete_ids(new_ids)
            raise
        self._delete_ids(old_ids)
        return len(new_ids)

    def similarity(self, query: str, k: int) -> list[Document]:
        """Return the `k` chunks closest to the query embedding."""
        return self._db.similarity_search(query, k=k)

    def mmr(self, query: str, k: int, fetch_k: int, lambda_mult: float) -> list[Document]:
        """Return relevant chunks that do not repeat each other (maximal marginal relevance).

        Args:
            query: The search text.
            k: Chunks to return.
            fetch_k: Candidates to consider before the diversity filter.
            lambda_mult: 0 favours variety, 1 favours relevance.

        Returns:
            Up to `k` chunks.
        """
        return self._db.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

    def all_documents(self) -> list[Document]:
        """Return every stored chunk. This reads the whole collection."""
        data = self._db.get(include=["documents", "metadatas"])
        texts: list[str | None] = data.get("documents") or []
        metadatas: list[Metadata | None] = data.get("metadatas") or []
        return [
            Document(page_content=text, metadata=dict(metadata or {}))
            for text, metadata in zip(texts, metadatas, strict=False)
            if text
        ]

    def count(self) -> int:
        """Return the number of stored chunks without reading them."""
        return self._client.get_collection(COLLECTION_NAME).count()

    def sources(self) -> list[IndexedSource]:
        """Return every indexed file with its chunk count, sorted by name."""
        metadatas: list[Metadata | None] = (
            self._db.get(include=["metadatas"]).get("metadatas") or []
        )
        counts: Counter[str] = Counter()
        names: dict[str, str] = {}
        for metadata in metadatas:
            if not metadata:
                continue
            name = str(metadata.get("source", "unknown"))
            source_id = str(metadata.get("source_id", name))
            counts[source_id] += 1
            names[source_id] = name
        return sorted(
            (IndexedSource(source_id, names[source_id], n) for source_id, n in counts.items()),
            key=lambda source: (source.name, source.source_id),
        )

    def delete_source(self, source_id: str) -> int:
        """Remove every chunk of one file. Returns how many were removed."""
        ids = self._ids_where({"source_id": source_id})
        self._delete_ids(ids)
        return len(ids)

    def clear(self) -> int:
        """Remove every chunk. Returns how many were removed."""
        ids = self._ids_where(None)
        self._delete_ids(ids)
        return len(ids)

    def _ids_where(self, where: Where | None) -> list[str]:
        ids: list[str] = self._db.get(where=where, include=[]).get("ids") or []
        return ids

    def _delete_ids(self, ids: Sequence[str]) -> None:
        if ids:
            self._db.delete(ids=list(ids))
