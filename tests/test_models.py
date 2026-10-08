import pytest

from askmydocs.models import EXCERPT_CHARS, SourceChunk, chunk_label


@pytest.mark.parametrize(
    ("page", "expected"),
    [(3, "report.pdf, page 3"), (None, "report.pdf")],
)
def test_source_label_includes_page_when_known(page: int | None, expected: str) -> None:
    assert SourceChunk("report.pdf", page, "text").label == expected


def test_excerpt_shortens_long_text() -> None:
    chunk = SourceChunk("a.txt", None, "word " * 100)

    assert chunk.excerpt.endswith("...")
    assert len(chunk.excerpt) <= EXCERPT_CHARS + 3


@pytest.mark.parametrize(("count", "expected"), [(0, "0 chunks"), (1, "1 chunk"), (5, "5 chunks")])
def test_chunk_label_uses_the_right_plural(count: int, expected: str) -> None:
    assert chunk_label(count) == expected
