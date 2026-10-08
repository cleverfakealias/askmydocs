from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from palimpsest.config import RetrievalMethod, RetrievalSettings, Settings
from palimpsest.engine import MAX_HISTORY_MESSAGES, NO_MATCH_TEXT, Conversation, RAGEngine
from palimpsest.retrieval.store import VectorStore
from tests.fakes import BrokenChatModel, FlakyEmbeddings, RecordingChatModel

PARAGRAPH = (
    "Palimpsest stores each document as overlapping chunks. "
    "Each chunk is embedded so that questions can find the closest passages. "
)


def _engine(
    settings: Settings, embeddings: FlakyEmbeddings, chat: BaseChatModel | None = None
) -> RAGEngine:
    store = VectorStore(settings.vector_db_path, embeddings)
    return RAGEngine(settings, store, lambda: chat or BrokenChatModel())


def _write(path: Path, text: str = PARAGRAPH * 6) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    return _write(tmp_path / "notes.txt")


def test_ingest_then_ask_returns_answer_with_sources(
    settings: Settings, embeddings: FlakyEmbeddings, notes: Path
) -> None:
    engine = _engine(settings, embeddings, FakeListChatModel(responses=["Chunks [1] are small."]))
    report = engine.ingest_files([notes])

    answer = engine.ask("How are documents stored?", engine.new_conversation())

    assert report.failed == []
    assert answer.text == "Chunks [1] are small."
    assert answer.degraded is False
    assert {source.source for source in answer.sources} == {"notes.txt"}


def test_question_without_documents_skips_the_model(
    settings: Settings, embeddings: FlakyEmbeddings
) -> None:
    engine = _engine(settings, embeddings)

    answer = engine.ask("Anything at all?", engine.new_conversation())

    assert answer.text == NO_MATCH_TEXT
    assert answer.sources == ()


def test_chat_model_is_built_only_when_a_question_needs_it(
    settings: Settings, embeddings: FlakyEmbeddings, notes: Path
) -> None:
    built: list[BaseChatModel] = []

    def factory() -> BaseChatModel:
        model = FakeListChatModel(responses=["ok"])
        built.append(model)
        return model

    engine = RAGEngine(settings, VectorStore(settings.vector_db_path, embeddings), factory)
    engine.ingest_files([notes])
    engine.indexed_sources()
    assert built == []

    conversation = engine.new_conversation()
    engine.ask("What is a chunk?", conversation)
    engine.ask("And an embedding?", conversation)
    assert len(built) == 1


def test_conversations_do_not_share_history(
    settings: Settings, embeddings: FlakyEmbeddings, notes: Path
) -> None:
    chat = RecordingChatModel(responses=["first answer", "second answer"])
    engine = _engine(settings, embeddings, chat)
    engine.ingest_files([notes])
    alice, bob = engine.new_conversation(), engine.new_conversation()

    engine.ask("Alice's private question", alice)
    engine.ask("Bob's question", bob)

    bob_prompt = chat.received[-1]
    assert not any("Alice" in str(message.content) for message in bob_prompt)
    assert len(alice.history) == 2
    assert len(bob.history) == 2


def test_history_keeps_only_recent_messages() -> None:
    conversation = Conversation(
        retrieval=RetrievalSettings("mmr", top_k=4, fetch_k=20, mmr_lambda=0.5, hybrid_alpha=0.7)
    )

    for number in range(MAX_HISTORY_MESSAGES):
        conversation.record(f"question {number}", f"answer {number}")

    assert len(conversation.history) == MAX_HISTORY_MESSAGES
    assert isinstance(conversation.history[0], HumanMessage)
    assert conversation.history[-1].content == f"answer {MAX_HISTORY_MESSAGES - 1}"


def test_reingesting_a_file_does_not_duplicate_chunks(
    settings: Settings, embeddings: FlakyEmbeddings, notes: Path
) -> None:
    engine = _engine(settings, embeddings)
    engine.ingest_files([notes])
    first_count = engine.chunk_count()

    engine.ingest_files([notes])

    assert engine.chunk_count() == first_count
    assert [source.name for source in engine.indexed_sources()] == ["notes.txt"]


def test_same_file_name_in_two_folders_is_indexed_twice(
    settings: Settings, embeddings: FlakyEmbeddings, tmp_path: Path
) -> None:
    first = _write(tmp_path / "a" / "README.md")
    second = _write(tmp_path / "b" / "README.md")
    engine = _engine(settings, embeddings)

    engine.ingest_files([first, second])

    sources = engine.indexed_sources()
    assert [source.name for source in sources] == ["README.md", "README.md"]
    assert {source.source_id for source in sources} == {
        str(first.resolve()),
        str(second.resolve()),
    }


def test_failed_reingest_keeps_the_previous_chunks(
    settings: Settings, embeddings: FlakyEmbeddings, notes: Path
) -> None:
    engine = _engine(settings, embeddings)
    engine.ingest_files([notes])
    before = engine.chunk_count()

    embeddings.fail = True
    report = engine.ingest_files([notes])

    assert report.failed
    assert "CUDA out of memory" in (report.failed[0].error or "")
    assert engine.chunk_count() == before


def test_failed_file_is_reported_and_the_batch_continues(
    settings: Settings, embeddings: FlakyEmbeddings, notes: Path, tmp_path: Path
) -> None:
    bad = tmp_path / "picture.png"
    bad.write_bytes(b"\x89PNG")
    engine = _engine(settings, embeddings)

    report = engine.ingest_files([bad, notes])

    assert [item.ok for item in report.files] == [False, True]
    assert "unsupported type" in (report.failed[0].error or "")
    assert engine.chunk_count() > 0


def test_model_failure_returns_raw_passages(
    settings: Settings, embeddings: FlakyEmbeddings, notes: Path
) -> None:
    engine = _engine(settings, embeddings)
    engine.ingest_files([notes])
    conversation = engine.new_conversation()

    answer = engine.ask("What is a chunk?", conversation)

    assert answer.degraded is True
    assert "1. **notes.txt**: " in answer.text
    assert answer.text.count("notes.txt") == len(answer.sources)
    assert conversation.history == []


def test_empty_question_is_rejected(settings: Settings, embeddings: FlakyEmbeddings) -> None:
    engine = _engine(settings, embeddings)

    with pytest.raises(ValueError, match="empty"):
        engine.ask("   ", engine.new_conversation())


def test_removing_a_source_updates_keyword_search(
    settings: Settings, embeddings: FlakyEmbeddings, notes: Path, tmp_path: Path
) -> None:
    other = _write(tmp_path / "zebra.txt", "Zebras have black and white stripes. " * 5)
    engine = _engine(settings, embeddings)
    engine.ingest_files([notes, other])
    keyword_only = RetrievalSettings(
        "hybrid", top_k=2, fetch_k=4, mmr_lambda=0.5, hybrid_alpha=0.0
    )
    # The query must match a whole token ("zebras"), or BM25 returns nothing and
    # only the zero-weight semantic hits from the random fake embeddings remain.
    assert engine.retrieve("zebras", keyword_only)[0].metadata["source"] == "zebra.txt"

    engine.remove_source(str(other.resolve()))

    assert all(
        doc.metadata["source"] != "zebra.txt" for doc in engine.retrieve("zebras", keyword_only)
    )


@pytest.mark.parametrize("method", ["similarity", "mmr", "hybrid"])
def test_every_retrieval_method_returns_passages(
    settings: Settings, embeddings: FlakyEmbeddings, notes: Path, method: RetrievalMethod
) -> None:
    engine = _engine(settings, embeddings)
    engine.ingest_files([notes])
    retrieval = RetrievalSettings(method, top_k=2, fetch_k=4, mmr_lambda=0.5, hybrid_alpha=0.7)

    passages = engine.retrieve("chunks and embeddings", retrieval)

    assert 0 < len(passages) <= 2
