from langchain_core.documents import Document

from askmydocs.config import Settings
from askmydocs.documents.chunking import Chunker


def _page(text: str, file_type: str = ".txt") -> Document:
    return Document(
        page_content=text, metadata={"source": "doc" + file_type, "file_type": file_type}
    )


def test_long_text_is_split_within_chunk_size(settings: Settings) -> None:
    sentence = "Retrieval keeps answers grounded in your own files. "
    chunks = Chunker(settings).split([_page(sentence * 40)])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= settings.chunk_size for chunk in chunks)


def test_chunks_are_numbered_per_file(settings: Settings) -> None:
    chunks = Chunker(settings).split([_page("word " * 200), _page("another page " * 60)])

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["chunk_count"] == len(chunks) for chunk in chunks)


def test_short_fragments_are_dropped(settings: Settings) -> None:
    chunks = Chunker(settings).split([_page("tiny")])

    assert chunks == []


def test_source_metadata_is_kept(settings: Settings) -> None:
    page = Document(
        page_content="A page with enough text to keep as a chunk for this test.",
        metadata={"source": "report.pdf", "file_type": ".pdf", "page": 3},
    )

    chunks = Chunker(settings).split([page])

    assert chunks[0].metadata["source"] == "report.pdf"
    assert chunks[0].metadata["page"] == 3


def test_whitespace_is_normalized(settings: Settings) -> None:
    text = "First line with   extra   spaces.\n\n\n\nSecond paragraph that is long enough to keep."

    chunks = Chunker(settings).split([_page(text)])

    assert "   " not in chunks[0].page_content
    assert "\n\n\n" not in chunks[0].page_content
