import pytest
from langchain_core.documents import Document

from palimpsest.retrieval.ranking import BM25Index, bm25_rank, fuse_rankings, tokenize


def _doc(text: str, index: int = 0) -> Document:
    return Document(page_content=text, metadata={"source_id": "a.txt", "chunk_index": index})


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Hello, World! It's 2026.") == ["hello", "world", "it", "s", "2026"]


def test_bm25_prefers_documents_with_rare_query_terms() -> None:
    docs = [
        _doc("the cat sat on the mat", index=0),
        _doc("the dog chased the ball", index=1),
        _doc("quantum chromodynamics is about the strong force", index=2),
    ]

    ranked = bm25_rank("strong force physics", docs, top_k=3)

    assert ranked[0].metadata["chunk_index"] == 2
    assert len(ranked) == 1  # Only one document shares a query term.


@pytest.mark.parametrize(
    ("query", "documents"),
    [
        ("anything", []),
        ("   ", [_doc("some text here")]),
        ("missing words", [_doc("nothing in common")]),
    ],
)
def test_bm25_returns_nothing_when_no_term_matches(query: str, documents: list[Document]) -> None:
    assert bm25_rank(query, documents, top_k=5) == []


def test_bm25_respects_top_k() -> None:
    docs = [_doc(f"shared term {i}", index=i) for i in range(6)]

    assert len(bm25_rank("shared", docs, top_k=2)) == 2


def test_bm25_index_answers_many_queries() -> None:
    index = BM25Index([_doc("apples and pears", 0), _doc("pears and plums", 1)])

    assert len(index) == 2
    assert [doc.metadata["chunk_index"] for doc in index.rank("apples", top_k=5)] == [0]
    assert [doc.metadata["chunk_index"] for doc in index.rank("plums", top_k=5)] == [1]


def test_bm25_index_handles_documents_without_words() -> None:
    assert BM25Index([_doc("!!!"), _doc("...")]).rank("word", top_k=3) == []


def test_fusion_puts_heavily_weighted_list_first() -> None:
    semantic = [_doc("semantic best", index=0), _doc("semantic second", index=1)]
    keyword = [_doc("keyword best", index=2)]

    fused = fuse_rankings([(semantic, 0.9), (keyword, 0.1)], top_k=3)

    assert fused[0].page_content == "semantic best"


def test_fusion_merges_duplicates_across_lists() -> None:
    shared = _doc("found by both methods", index=5)
    keyword = [shared, _doc("only keyword", index=6)]

    fused = fuse_rankings([([shared], 0.5), (keyword, 0.5)], top_k=5)

    assert [doc.page_content for doc in fused].count("found by both methods") == 1
    assert fused[0].page_content == "found by both methods"


def test_fusion_handles_empty_input() -> None:
    assert fuse_rankings([], top_k=3) == []
