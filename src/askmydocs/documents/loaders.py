"""Read supported files into LangChain documents, one per page or section."""

from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.documents import Document
from pypdf import PdfReader

from askmydocs.config import SUPPORTED_EXTENSIONS


class UnsupportedFileError(ValueError):
    """Raised when a file type is not in SUPPORTED_EXTENSIONS."""


def load_file(path: Path) -> list[Document]:
    """Return the text of a file as documents. Empty pages are skipped.

    Args:
        path: A PDF, TXT, MD, or DOCX file.

    Returns:
        One document per PDF page, or one document for other file types. Each has
        `source` (the file name) and `file_type` metadata, and PDF pages add `page`.

    Raises:
        UnsupportedFileError: The file extension is not supported.
        OSError: The file cannot be read.
    """
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileError(
            f"{path.name}: unsupported type '{extension}'. Use {supported}."
        )

    pages: list[tuple[int | None, str]]
    if extension == ".pdf":
        pages = _load_pdf(path)
    elif extension == ".docx":
        pages = [(None, _load_docx(path))]
    else:
        pages = [(None, path.read_text(encoding="utf-8", errors="replace"))]

    documents: list[Document] = []
    for page_number, text in pages:
        if not text.strip():
            continue
        metadata: dict[str, str | int] = {"source": path.name, "file_type": extension}
        if page_number is not None:
            metadata["page"] = page_number
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def _load_pdf(path: Path) -> list[tuple[int | None, str]]:
    reader = PdfReader(path)
    return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]


def _load_docx(path: Path) -> str:
    document = DocxDocument(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(text for text in paragraphs if text.strip())
