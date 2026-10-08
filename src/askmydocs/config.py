"""Typed application settings, read from environment variables or a `.env` file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, get_args

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type RetrievalMethod = Literal["similarity", "mmr", "hybrid"]
type Device = Literal["auto", "cpu", "cuda", "mps"]

RETRIEVAL_METHODS: tuple[RetrievalMethod, ...] = get_args(RetrievalMethod.__value__)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".docx"})


@dataclass(frozen=True, slots=True)
class Preset:
    """A named pair of chat model and embedding model."""

    description: str
    llm_model: str
    embedding_model: str


# Open-weight models that load without an access request or API key.
PRESETS: dict[str, Preset] = {
    "fast": Preset(
        description="Smallest chat model. Good for quick checks on a laptop.",
        llm_model="Qwen/Qwen2.5-1.5B-Instruct",
        embedding_model="BAAI/bge-small-en-v1.5",
    ),
    "balanced": Preset(
        description="Default. Solid answers on a laptop with 8 GB of RAM.",
        llm_model="Qwen/Qwen2.5-3B-Instruct",
        embedding_model="BAAI/bge-small-en-v1.5",
    ),
    "quality": Preset(
        description="Best answers. Wants a GPU or 16 GB of RAM.",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        embedding_model="BAAI/bge-base-en-v1.5",
    ),
    "code": Preset(
        description="Tuned for source code and technical documentation.",
        llm_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        embedding_model="BAAI/bge-base-en-v1.5",
    ),
}
DEFAULT_PRESET = "balanced"


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """How a conversation searches the index. Every instance is validated on creation.

    Attributes:
        method: The search strategy.
        top_k: Passages sent to the chat model.
        fetch_k: Candidates considered before MMR or hybrid fusion.
        mmr_lambda: MMR balance. 0 favours variety, 1 favours relevance.
        hybrid_alpha: Semantic weight in hybrid search. 0 uses keywords only.
    """

    method: RetrievalMethod
    top_k: int
    fetch_k: int
    mmr_lambda: float
    hybrid_alpha: float

    def __post_init__(self) -> None:
        """Reject values that the search functions cannot use.

        Raises:
            ValueError: An option is out of range or inconsistent with another.
        """
        if self.method not in RETRIEVAL_METHODS:
            known = ", ".join(RETRIEVAL_METHODS)
            raise ValueError(f"Unknown retrieval method '{self.method}'. Choose one of: {known}.")
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}.")
        if self.fetch_k < self.top_k:
            raise ValueError(f"fetch_k ({self.fetch_k}) must be at least top_k ({self.top_k}).")
        for name, value in (("mmr_lambda", self.mmr_lambda), ("hybrid_alpha", self.hybrid_alpha)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")


class Settings(BaseSettings):
    """All runtime options. Each field maps to an environment variable of the same name."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    model_preset: str = DEFAULT_PRESET
    llm_model: str | None = Field(default=None, description="Overrides the preset chat model.")
    embedding_model: str | None = Field(default=None, description="Overrides the preset embedder.")
    device: Device = "auto"

    max_new_tokens: int = Field(default=512, ge=16, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)
    min_chunk_chars: int = Field(default=50, ge=0)

    retrieval_method: RetrievalMethod = "mmr"
    top_k: int = Field(default=4, ge=1, le=50)
    fetch_k: int = Field(default=20, ge=1, le=200)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Semantic weight.")

    vector_db_path: Path = Path("vector_db")
    documents_path: Path = Path("documents")

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.model_preset not in PRESETS:
            known = ", ".join(PRESETS)
            raise ValueError(
                f"Unknown MODEL_PRESET '{self.model_preset}'. Choose one of: {known}."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        self.retrieval_defaults()  # Raises when the retrieval options are inconsistent.
        return self

    def retrieval_defaults(self) -> RetrievalSettings:
        """Return the retrieval options that each new conversation starts with."""
        return RetrievalSettings(
            method=self.retrieval_method,
            top_k=self.top_k,
            fetch_k=self.fetch_k,
            mmr_lambda=self.mmr_lambda,
            hybrid_alpha=self.hybrid_alpha,
        )

    @property
    def preset(self) -> Preset:
        """The model preset named by `model_preset`."""
        return PRESETS[self.model_preset]

    @property
    def resolved_llm_model(self) -> str:
        """The chat model ID: the explicit override, or else the preset's model."""
        return self.llm_model or self.preset.llm_model

    @property
    def resolved_embedding_model(self) -> str:
        """The embedding model ID: the explicit override, or else the preset's model."""
        return self.embedding_model or self.preset.embedding_model
