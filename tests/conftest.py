"""Shared fixtures. Tests never download models. They use LangChain's fake embedder."""

from pathlib import Path

import pytest

from askmydocs.config import Settings
from askmydocs.retrieval.store import VectorStore
from tests.fakes import FlakyEmbeddings


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the developer's own settings, such as TOP_K, from every test."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that point every path into the test's temporary folder."""
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        vector_db_path=tmp_path / "vector_db",
        documents_path=tmp_path / "documents",
        chunk_size=300,
        chunk_overlap=30,
        min_chunk_chars=20,
    )


@pytest.fixture
def embeddings() -> FlakyEmbeddings:
    return FlakyEmbeddings(size=16)


@pytest.fixture
def store(settings: Settings, embeddings: FlakyEmbeddings) -> VectorStore:
    return VectorStore(settings.vector_db_path, embeddings)
