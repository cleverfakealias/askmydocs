"""Split loaded pages into overlapping chunks that the embedder can index."""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from palimpsest.config import Settings

# Markdown splits on headings first, so a section stays in one chunk when it fits.
MARKDOWN_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
PLAIN_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_BLANK_LINES = re.compile(r"\n{3,}")
_INLINE_SPACE = re.compile(r"[ \t]+")


class Chunker:
    """Splits the pages of one file into chunks, choosing separators by file type."""

    def __init__(self, settings: Settings) -> None:
        """Build the splitters from the chunk size, overlap, and minimum length settings."""
        self._min_chars = settings.min_chunk_chars
        self._markdown = _splitter(settings, MARKDOWN_SEPARATORS)
        self._plain = _splitter(settings, PLAIN_SEPARATORS)

    def split(self, pages: list[Document]) -> list[Document]:
        """Split the pages of one file into chunks.

        Whitespace is normalized, and chunks shorter than `min_chunk_chars` are dropped.

        Args:
            pages: The file's pages, as returned by `load_file`.

        Returns:
            Chunks that keep the page metadata and add `chunk_index` and `chunk_count`.
        """
        chunks: list[Document] = []
        for page in pages:
            splitter = self._markdown if page.metadata.get("file_type") == ".md" else self._plain
            for piece in splitter.split_documents([page]):
                text = _normalize(piece.page_content)
                if len(text) < self._min_chars:
                    continue
                chunks.append(Document(page_content=text, metadata=dict(piece.metadata)))

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_count"] = len(chunks)
        return chunks


def _splitter(settings: Settings, separators: list[str]) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=separators,
    )


def _normalize(text: str) -> str:
    text = _INLINE_SPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()
