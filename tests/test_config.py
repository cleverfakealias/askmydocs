from pathlib import Path

import pytest
from pydantic import ValidationError

from palimpsest.config import DEFAULT_PRESET, PRESETS, RetrievalSettings, Settings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_default_preset_is_balanced() -> None:
    settings = _settings()

    assert settings.model_preset == DEFAULT_PRESET == "balanced"
    assert settings.resolved_llm_model == PRESETS["balanced"].llm_model


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("RETRIEVAL_METHOD", "hybrid")
    monkeypatch.setenv("VECTOR_DB_PATH", "/tmp/elsewhere")  # noqa: S108

    settings = _settings()

    assert settings.top_k == 7
    assert settings.retrieval_method == "hybrid"
    assert settings.vector_db_path == Path("/tmp/elsewhere")  # noqa: S108


def test_custom_model_overrides_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PRESET", "quality")
    monkeypatch.setenv("LLM_MODEL", "example/custom-chat")

    settings = _settings()

    assert settings.resolved_llm_model == "example/custom-chat"
    assert settings.resolved_embedding_model == PRESETS["quality"].embedding_model


def test_unknown_preset_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PRESET", "nonexistent")

    with pytest.raises(ValidationError, match="Unknown MODEL_PRESET"):
        _settings()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_size": 200, "chunk_overlap": 200}, "CHUNK_OVERLAP must be smaller"),
        ({"top_k": 10, "fetch_k": 5}, r"fetch_k \(5\) must be at least top_k \(10\)"),
    ],
)
def test_inconsistent_settings_are_rejected(overrides: dict[str, int], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _settings(**overrides)


def test_retrieval_defaults_come_from_settings() -> None:
    retrieval = _settings(retrieval_method="similarity", top_k=3).retrieval_defaults()

    assert retrieval.method == "similarity"
    assert retrieval.top_k == 3


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"method": "bm25"}, "Unknown retrieval method 'bm25'"),
        ({"top_k": 0, "fetch_k": 0}, "top_k must be at least 1"),
        ({"top_k": 10, "fetch_k": 4}, r"fetch_k \(4\) must be at least top_k \(10\)"),
        ({"mmr_lambda": 1.5}, "mmr_lambda must be between 0 and 1"),
        ({"hybrid_alpha": -0.1}, "hybrid_alpha must be between 0 and 1"),
    ],
)
def test_invalid_retrieval_settings_are_rejected(changes: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "method": "mmr",
        "top_k": 4,
        "fetch_k": 20,
        "mmr_lambda": 0.5,
        "hybrid_alpha": 0.7,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        RetrievalSettings(**values)  # pyright: ignore[reportArgumentType]
