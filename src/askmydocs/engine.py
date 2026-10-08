"""The RAG engine: ingest files, retrieve passages, and answer questions with a local model."""

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import assert_never

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from askmydocs.config import RetrievalSettings, Settings
from askmydocs.documents.chunking import Chunker
from askmydocs.documents.loaders import load_file
from askmydocs.models import Answer, FileIngest, IndexedSource, IngestReport, SourceChunk
from askmydocs.models_factory import build_chat_model, build_embeddings
from askmydocs.retrieval.ranking import BM25Index, fuse_rankings
from askmydocs.retrieval.store import VectorStore

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 12  # Six question and answer pairs.

SYSTEM_PROMPT = """You answer questions using only the numbered passages from the user's documents.
- Start with a direct answer in one or two sentences. Add detail after that.
- If the passages do not contain the answer, say so. Do not guess.
- Cite passages inline by number, such as [1] or [2].
- Put code in fenced blocks and explain what it does."""

NO_MATCH_TEXT = (
    "I could not find anything relevant in your documents. "
    "Try rephrasing the question, or check that the files were processed."
)


@dataclass(slots=True)
class Conversation:
    """One person's chat: their retrieval options and recent history.

    Keep one conversation per user session. The engine itself holds no per-user state,
    so one engine can serve many sessions.
    """

    retrieval: RetrievalSettings
    history: list[BaseMessage] = field(default_factory=list[BaseMessage])

    def record(self, question: str, answer: str) -> None:
        """Add a question and its answer, keeping only the most recent messages."""
        self.history.extend([HumanMessage(question), AIMessage(answer)])
        del self.history[:-MAX_HISTORY_MESSAGES]

    def reset(self) -> None:
        """Forget the history. The retrieval options are kept."""
        self.history.clear()


class RAGEngine:
    """Owns the vector store and the chat model. Safe to share between threads."""

    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        chat_model_factory: Callable[[], BaseChatModel],
    ) -> None:
        """Create an engine. The chat model is built on the first question.

        Args:
            settings: The application settings.
            store: Where chunks are stored and searched.
            chat_model_factory: Builds the chat model. It is called at most once.
        """
        self.settings = settings
        self._store = store
        self._chunker = Chunker(settings)
        self._chat_model_factory = chat_model_factory
        self._chat_model: BaseChatModel | None = None
        self._model_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._keyword_index: BM25Index | None = None

    @classmethod
    def create(cls, settings: Settings) -> RAGEngine:
        """Load the embedder now and the chat model on the first question.

        The first run downloads the models, which can take minutes.
        """
        store = VectorStore(settings.vector_db_path, build_embeddings(settings))
        return cls(settings, store, partial(build_chat_model, settings))

    def new_conversation(self) -> Conversation:
        """Start a conversation with the configured retrieval options."""
        return Conversation(retrieval=self.settings.retrieval_defaults())

    def ingest_files(self, paths: Sequence[Path]) -> IngestReport:
        """Index files. A file that fails is recorded in the report, and the rest continue."""
        report = IngestReport()
        for path in paths:
            report.files.append(self._ingest_one(path))
        return report

    def _ingest_one(self, path: Path) -> FileIngest:
        # The resolved path keys the file, so same-named files in other folders stay apart.
        source_id = str(path.resolve())
        try:
            chunks = self._chunker.split(load_file(path))
            if not chunks:
                return FileIngest(str(path), 0, error="No readable text found.")
            for chunk in chunks:
                chunk.metadata["source_id"] = source_id
            stored = self._store.replace_source(source_id, chunks)
        except Exception as exc:
            # One bad file must not stop the batch. The error is kept for the report.
            logger.exception("Failed to ingest %s", path)
            return FileIngest(str(path), 0, error=f"{type(exc).__name__}: {exc}")
        self._invalidate_keyword_index()
        logger.info("Indexed %s into %d chunks", path.name, stored)
        return FileIngest(str(path), stored)

    def retrieve(self, query: str, retrieval: RetrievalSettings) -> list[Document]:
        """Return the passages to use for a query.

        Args:
            query: The search text.
            retrieval: The search options.

        Returns:
            Up to `retrieval.top_k` passages, best first.
        """
        match retrieval.method:
            case "similarity":
                return self._store.similarity(query, k=retrieval.top_k)
            case "mmr":
                return self._store.mmr(
                    query,
                    k=retrieval.top_k,
                    fetch_k=retrieval.fetch_k,
                    lambda_mult=retrieval.mmr_lambda,
                )
            case "hybrid":
                semantic = self._store.similarity(query, k=retrieval.fetch_k)
                keyword = self._keyword_index_snapshot().rank(query, top_k=retrieval.fetch_k)
                return fuse_rankings(
                    [(semantic, retrieval.hybrid_alpha), (keyword, 1 - retrieval.hybrid_alpha)],
                    top_k=retrieval.top_k,
                )
            case _:
                assert_never(retrieval.method)

    def ask(self, question: str, conversation: Conversation) -> Answer:
        """Answer a question from the indexed documents and record it in the conversation.

        Args:
            question: The question text.
            conversation: The asker's conversation. Its history is sent to the model.

        Returns:
            The answer with its sources. When the chat model fails, the answer lists the
            matching passages instead and `degraded` is True.

        Raises:
            ValueError: The question is empty.
        """
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")

        method = conversation.retrieval.method
        documents = self.retrieve(question, conversation.retrieval)
        sources = tuple(_to_source(doc) for doc in documents)
        if not documents:
            return Answer(NO_MATCH_TEXT, sources=(), retrieval_method=method)

        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            *conversation.history,
            HumanMessage(_build_user_turn(question, sources)),
        ]
        try:
            # One generation at a time: the model is shared, and GPU memory is limited.
            with self._model_lock:
                reply = self._get_chat_model().invoke(messages)
            text = _text_of(reply).strip()
            if not text:
                raise ValueError("The chat model returned an empty reply.")
        except Exception:
            # The model can fail to load, run out of memory, or reply with nothing.
            logger.exception("Chat model failed. Returning raw passages.")
            return Answer(_excerpt_fallback(sources), sources, method, degraded=True)

        conversation.record(question, text)
        return Answer(text, sources, method)

    def indexed_sources(self) -> list[IndexedSource]:
        """Return every indexed file with its chunk count, sorted by name."""
        return self._store.sources()

    def chunk_count(self) -> int:
        """Return the number of stored chunks."""
        return self._store.count()

    def remove_source(self, source_id: str) -> int:
        """Remove one file from the index. Returns how many chunks were removed."""
        removed = self._store.delete_source(source_id)
        self._invalidate_keyword_index()
        return removed

    def clear_documents(self) -> int:
        """Remove every file from the index. Returns how many chunks were removed."""
        removed = self._store.clear()
        self._invalidate_keyword_index()
        return removed

    def _get_chat_model(self) -> BaseChatModel:
        # Called with _model_lock held.
        if self._chat_model is None:
            self._chat_model = self._chat_model_factory()
        return self._chat_model

    def _keyword_index_snapshot(self) -> BM25Index:
        with self._index_lock:
            if self._keyword_index is None:
                self._keyword_index = BM25Index(self._store.all_documents())
            return self._keyword_index

    def _invalidate_keyword_index(self) -> None:
        with self._index_lock:
            self._keyword_index = None


def _to_source(doc: Document) -> SourceChunk:
    page = doc.metadata.get("page")
    return SourceChunk(
        source=str(doc.metadata.get("source", "unknown")),
        page=page if isinstance(page, int) else None,
        text=doc.page_content,
    )


def _build_user_turn(question: str, sources: Sequence[SourceChunk]) -> str:
    passages = "\n\n".join(
        f"[{number}] ({source.label})\n{source.text}"
        for number, source in enumerate(sources, start=1)
    )
    return f"Passages:\n{passages}\n\nQuestion: {question}"


def _excerpt_fallback(sources: Sequence[SourceChunk]) -> str:
    lines = ["The chat model was unavailable. These passages matched your question:", ""]
    lines.extend(
        f"{number}. **{source.label}**: {source.excerpt}"
        for number, source in enumerate(sources, start=1)
    )
    return "\n".join(lines)


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
