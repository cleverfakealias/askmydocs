"""Plain data types shared across the engine, the UI, and the CLI."""

from dataclasses import dataclass, field

from palimpsest.config import RetrievalMethod

EXCERPT_CHARS = 240


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """One retrieved passage that supports an answer."""

    source: str
    page: int | None
    text: str

    @property
    def label(self) -> str:
        """The file name, plus the page number when the file has pages."""
        return f"{self.source}, page {self.page}" if self.page is not None else self.source

    @property
    def excerpt(self) -> str:
        """The passage text, shortened for display."""
        if len(self.text) <= EXCERPT_CHARS:
            return self.text
        return self.text[:EXCERPT_CHARS].rstrip() + "..."


@dataclass(frozen=True, slots=True)
class Answer:
    """The reply to one question, with the passages it used."""

    text: str
    sources: tuple[SourceChunk, ...]
    retrieval_method: RetrievalMethod
    degraded: bool = False
    """True when the chat model failed and the text is built from raw excerpts."""


@dataclass(frozen=True, slots=True)
class IndexedSource:
    """One indexed file.

    Attributes:
        source_id: Stable key for the file: its resolved path when it was indexed.
        name: The file name shown to people.
        chunks: How many chunks are stored for the file.
    """

    source_id: str
    name: str
    chunks: int


@dataclass(frozen=True, slots=True)
class FileIngest:
    """Outcome of ingesting one file."""

    path: str
    chunks: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the file was indexed without an error."""
        return self.error is None


@dataclass(slots=True)
class IngestReport:
    """Outcome of ingesting a batch of files."""

    files: list[FileIngest] = field(default_factory=list[FileIngest])

    @property
    def total_chunks(self) -> int:
        """Chunks stored across every file that succeeded."""
        return sum(f.chunks for f in self.files if f.ok)

    @property
    def failed(self) -> list[FileIngest]:
        """The files that could not be indexed."""
        return [f for f in self.files if not f.ok]


def chunk_label(count: int) -> str:
    """Return '1 chunk' or 'N chunks'."""
    return f"{count} chunk" if count == 1 else f"{count} chunks"
