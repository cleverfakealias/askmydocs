import pytest
from langchain_core.documents import Document

from askmydocs.models import IndexedSource
from askmydocs.retrieval.store import VectorStore
from tests.fakes import FlakyEmbeddings


def _chunks(source_id: str, count: int, name: str | None = None) -> list[Document]:
    return [
        Document(
            page_content=f"{source_id} passage number {index} about retrieval and storage.",
            metadata={"source_id": source_id, "source": name or source_id, "chunk_index": index},
        )
        for index in range(count)
    ]


def test_replace_source_stores_and_counts(store: VectorStore) -> None:
    stored = store.replace_source("alpha", _chunks("alpha", 3))

    assert stored == 3
    assert store.count() == 3


def test_replace_source_with_no_chunks_stores_nothing(store: VectorStore) -> None:
    assert store.replace_source("alpha", []) == 0
    assert store.count() == 0


def test_replace_source_swaps_old_chunks_for_new(store: VectorStore) -> None:
    store.replace_source("alpha", _chunks("alpha", 3))
    store.replace_source("beta", _chunks("beta", 1))

    store.replace_source("alpha", _chunks("alpha", 2))

    assert store.count() == 3
    assert {source.source_id: source.chunks for source in store.sources()} == {
        "alpha": 2,
        "beta": 1,
    }


def test_failed_write_keeps_old_chunks_and_leaves_no_partial_rows(
    store: VectorStore, embeddings: FlakyEmbeddings
) -> None:
    store.replace_source("alpha", _chunks("alpha", 2))
    embeddings.fail = True

    with pytest.raises(RuntimeError, match="out of memory"):
        store.replace_source("alpha", _chunks("alpha", 5))

    assert store.count() == 2


def test_sources_keep_same_named_files_apart(store: VectorStore) -> None:
    store.replace_source("/a/README.md", _chunks("/a/README.md", 2, name="README.md"))
    store.replace_source("/b/README.md", _chunks("/b/README.md", 1, name="README.md"))

    assert store.sources() == [
        IndexedSource("/a/README.md", "README.md", 2),
        IndexedSource("/b/README.md", "README.md", 1),
    ]


def test_delete_source_removes_only_that_file(store: VectorStore) -> None:
    store.replace_source("alpha", _chunks("alpha", 2))
    store.replace_source("beta", _chunks("beta", 1))

    removed = store.delete_source("alpha")

    assert removed == 2
    assert [source.source_id for source in store.sources()] == ["beta"]


def test_clear_removes_everything(store: VectorStore) -> None:
    store.replace_source("alpha", _chunks("alpha", 2))
    store.replace_source("beta", _chunks("beta", 2))

    assert store.clear() == 4
    assert store.count() == 0


def test_similarity_returns_documents(store: VectorStore) -> None:
    store.replace_source("alpha", _chunks("alpha", 3))

    results = store.similarity("retrieval", k=2)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_mmr_returns_at_most_k(store: VectorStore) -> None:
    store.replace_source("alpha", _chunks("alpha", 5))

    results = store.mmr("storage", k=2, fetch_k=5, lambda_mult=0.5)

    assert len(results) == 2


def test_all_documents_returns_every_chunk(store: VectorStore) -> None:
    store.replace_source("alpha", _chunks("alpha", 4))

    documents = store.all_documents()

    assert len(documents) == 4
    assert {doc.metadata["source_id"] for doc in documents} == {"alpha"}
