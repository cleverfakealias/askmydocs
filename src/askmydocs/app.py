"""Streamlit interface for AskMyDocs. Start it with `askmydocs ui`.

One engine is shared by every browser session. Each session keeps its own
`Conversation` (history and retrieval options) and chat log in `st.session_state`.
"""

import logging
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from askmydocs.config import RETRIEVAL_METHODS, SUPPORTED_EXTENSIONS, RetrievalSettings, Settings
from askmydocs.engine import Conversation, RAGEngine
from askmydocs.markdown_safety import escape_markdown, neutralize_markdown
from askmydocs.models import Answer, chunk_label

logger = logging.getLogger(__name__)

type ChatEntry = str | Answer
"""A user's question (str) or the assistant's answer."""

METHOD_HELP = {
    "mmr": "Maximal marginal relevance. Relevant passages that do not repeat each other.",
    "similarity": "Plain nearest-neighbour search on embeddings.",
    "hybrid": "Blends embedding search with BM25 keyword search.",
}


@st.cache_resource(show_spinner="Loading models. The first run downloads them.")
def get_engine() -> RAGEngine:
    """Return the engine shared by every session. It holds no per-user state."""
    return RAGEngine.create(Settings())


def session_conversation(engine: RAGEngine) -> Conversation:
    """Return this browser session's conversation, creating it on first use."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = engine.new_conversation()
    conversation: Conversation = st.session_state.conversation
    return conversation


def session_chat_log() -> list[ChatEntry]:
    """Return this browser session's displayed chat, creating it on first use."""
    if "chat_log" not in st.session_state:
        empty_log: list[ChatEntry] = []
        st.session_state.chat_log = empty_log
    chat_log: list[ChatEntry] = st.session_state.chat_log
    return chat_log


def documents_dir(engine: RAGEngine) -> Path:
    """Return the folder for uploaded files, creating it when it is missing."""
    path = engine.settings.documents_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def saved_documents(engine: RAGEngine) -> list[Path]:
    """Return the supported files in the documents folder, sorted by name."""
    return sorted(
        path
        for path in documents_dir(engine).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def render_answer(answer: Answer) -> None:
    """Show an answer, its sources, and the retrieval method.

    Model and document text is untrusted: images are removed from the answer, and
    excerpts and file names are shown literally.
    """
    st.markdown(neutralize_markdown(answer.text))
    if answer.degraded:
        st.caption("The chat model failed, so the matching passages are shown instead.")
    if answer.sources:
        with st.expander(f"Sources ({len(answer.sources)})"):
            for number, source in enumerate(answer.sources, start=1):
                st.markdown(f"**[{number}] {escape_markdown(source.label)}**")
                st.caption(escape_markdown(source.excerpt))
    st.caption(f"Retrieval: {answer.retrieval_method.upper()}")


def sidebar_retrieval(conversation: Conversation) -> None:
    """Show the retrieval controls. Changes apply to this session only."""
    st.subheader("Retrieval")
    current = conversation.retrieval
    method = (
        st.selectbox(
            "Method",
            RETRIEVAL_METHODS,
            index=RETRIEVAL_METHODS.index(current.method),
            format_func=str.upper,
            help=METHOD_HELP[current.method],
        )
        or current.method
    )
    top_k = st.slider("Passages per answer", min_value=1, max_value=10, value=current.top_k)
    mmr_lambda, hybrid_alpha = current.mmr_lambda, current.hybrid_alpha
    fetch_k = max(current.fetch_k, top_k)
    if method == "mmr":
        mmr_lambda = st.slider(
            "Diversity (0 = varied, 1 = most relevant)",
            min_value=0.0,
            max_value=1.0,
            value=current.mmr_lambda,
            step=0.1,
        )
        fetch_k = st.slider(
            "Candidates before diversity filter",
            min_value=top_k,
            max_value=max(50, fetch_k),
            value=fetch_k,
        )
    elif method == "hybrid":
        hybrid_alpha = st.slider(
            "Semantic weight (0 = keywords only, 1 = meaning only)",
            min_value=0.0,
            max_value=1.0,
            value=current.hybrid_alpha,
            step=0.1,
        )
    conversation.retrieval = RetrievalSettings(
        method=method,
        top_k=top_k,
        fetch_k=fetch_k,
        mmr_lambda=mmr_lambda,
        hybrid_alpha=hybrid_alpha,
    )


def sidebar_documents(engine: RAGEngine) -> None:
    """List saved and indexed files, with a remove button for each."""
    st.subheader("Documents")
    indexed = {source.source_id: source for source in engine.indexed_sources()}
    files = {str(path.resolve()): path for path in saved_documents(engine)}
    if not files and not indexed:
        st.info("No documents yet. Upload some above.")
        return

    total_chunks = sum(source.chunks for source in indexed.values())
    st.caption(f"{len(indexed)} indexed · {chunk_label(total_chunks)}")
    for source_id, path in files.items():
        name_col, status_col, delete_col = st.columns([5, 2, 1])
        name_col.markdown(escape_markdown(path.name))
        status_col.caption("indexed" if source_id in indexed else "not indexed")
        if delete_col.button("✕", key=f"delete-{source_id}", help="Remove the file and its index"):
            engine.remove_source(source_id)
            path.unlink(missing_ok=True)
            st.rerun()

    orphans = [source for source_id, source in indexed.items() if source_id not in files]
    if orphans:
        st.caption("Indexed, but not in the documents folder:")
    for source in orphans:
        name_col, forget_col = st.columns([6, 2])
        name_col.markdown(escape_markdown(source.name), help=source.source_id)
        if forget_col.button("Forget", key=f"forget-{source.source_id}"):
            engine.remove_source(source.source_id)
            st.rerun()


@st.dialog("Clear all documents?")
def confirm_clear_all(engine: RAGEngine) -> None:
    """Ask for confirmation, then remove every chunk and every saved file."""
    st.write("This removes every indexed chunk and every saved file. It cannot be undone.")
    if st.button("Delete everything", type="primary", width="stretch"):
        engine.clear_documents()
        for path in saved_documents(engine):
            path.unlink(missing_ok=True)
        session_conversation(engine).reset()
        session_chat_log().clear()
        st.rerun()


def sidebar_upload_and_index(engine: RAGEngine) -> None:
    """Show the uploader and the button that indexes the uploaded files."""
    st.subheader("Add documents")
    uploads = st.file_uploader(
        "Files",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Index documents", type="primary", width="stretch", disabled=not uploads):
        _index_uploads(engine, uploads or [])


def _index_uploads(engine: RAGEngine, uploads: list[UploadedFile]) -> None:
    with st.status("Indexing documents", expanded=True) as status:
        paths: list[Path] = []
        folder = documents_dir(engine)
        for upload in uploads:
            # Keep only the file name, so an upload cannot write outside the documents folder.
            target = folder / Path(upload.name).name
            target.write_bytes(upload.getbuffer())
            paths.append(target)
            st.markdown(f"Saved {escape_markdown(target.name)}")

        report = engine.ingest_files(paths)
        for item in report.files:
            name = escape_markdown(Path(item.path).name)
            if item.ok:
                st.markdown(f"✅ {name}: {chunk_label(item.chunks)}")
            else:
                st.markdown(f"❌ {name}: {escape_markdown(item.error or 'failed')}")
        status.update(
            label=f"Indexed {chunk_label(report.total_chunks)}",
            state="error" if report.failed else "complete",
        )


def render_sidebar(engine: RAGEngine, conversation: Conversation) -> None:
    """Show the whole sidebar."""
    with st.sidebar:
        st.title("📚 AskMyDocs")
        sidebar_upload_and_index(engine)
        st.divider()
        sidebar_documents(engine)
        st.divider()
        sidebar_retrieval(conversation)
        st.divider()
        if st.button("Clear chat", width="stretch"):
            conversation.reset()
            session_chat_log().clear()
            st.rerun()
        if st.button("Clear all documents", width="stretch"):
            confirm_clear_all(engine)


def render_chat(engine: RAGEngine, conversation: Conversation) -> None:
    """Show this session's chat and answer a new question."""
    chat_log = session_chat_log()
    for entry in chat_log:
        if isinstance(entry, Answer):
            with st.chat_message("assistant"):
                render_answer(entry)
        else:
            with st.chat_message("user"):
                st.markdown(escape_markdown(entry))

    question = st.chat_input("Ask about your documents")
    if not question or not question.strip():
        return

    chat_log.append(question)
    with st.chat_message("user"):
        st.markdown(escape_markdown(question))

    with st.chat_message("assistant"), st.spinner("Searching your documents"):
        try:
            answer = engine.ask(question, conversation)
        except Exception:
            # Retrieval can fail on a damaged database. Keep the chat usable.
            logger.exception("Question failed")
            st.error("Something went wrong while answering. See the terminal for details.")
            return
        render_answer(answer)
    chat_log.append(answer)


def main() -> None:
    """Render the page."""
    st.set_page_config(page_title="AskMyDocs", page_icon="📚", layout="wide")
    try:
        engine = get_engine()
    except Exception as exc:
        logger.exception("Could not start the RAG engine")
        st.error(f"The RAG engine could not start: {exc}. See the terminal for details.")
        st.stop()

    conversation = session_conversation(engine)
    render_sidebar(engine, conversation)
    st.header("Chat with your documents")
    if engine.chunk_count() == 0:
        st.info("Upload and index a document in the sidebar to start asking questions.")
    render_chat(engine, conversation)


main()
