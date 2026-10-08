# Palimpsest

Chat with your own documents. Palimpsest indexes PDF, TXT, Markdown, and Word files on your machine, then answers questions with a local open-weight model. It needs no API keys, and your documents and questions never leave your machine. The only network traffic is the model download from Hugging Face, which you can turn off after the first run. See [Privacy and network use](#privacy-and-network-use).

> A palimpsest is a manuscript written over older text that still shows through. The app layers answers over your files the same way, and cites the passages it used.

## Features

- **Local by default.** Chat and embedding models run through Hugging Face on your own hardware.
- **Cited answers.** Each answer lists the file and page of every passage it used.
- **Three retrieval methods.** MMR (default), plain similarity, and hybrid semantic plus BM25 keyword search.
- **Clear failure handling.** A bad file does not stop a batch. A failed model shows the matching passages instead.
- **Web UI and CLI.** Use Streamlit in the browser or the `palimpsest` command in a terminal.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/) for installing and running the project
- About 7 GB of free disk space for the default models, and 8 GB of RAM for the default preset
- Optional: an NVIDIA GPU. On Windows and Linux, `uv sync` installs the CUDA 13.0 build of PyTorch, which supports RTX 50-series cards. On macOS it installs the standard build, which uses Apple Silicon through `DEVICE=mps`.

## Quick start

```bash
# Install the locked dependencies into .venv
uv sync

# Open the web interface at http://localhost:8501
uv run palimpsest ui
```

The first run downloads the chat model and the embedding model. This can take several minutes.

## Command line

```bash
uv run palimpsest ingest docs/handbook.pdf notes.md   # index files
uv run palimpsest ask "What does the handbook say about refunds?"
uv run palimpsest status                              # list indexed files
uv run palimpsest -v ask "..."                        # show debug logs
```

## Configuration

Set options as environment variables, or in a `.env` file in the project root. Every option has a default.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PRESET` | `balanced` | Chooses a chat and embedding model pair. See the table below. |
| `LLM_MODEL` | from preset | Hugging Face ID of a chat model. Overrides the preset. |
| `EMBEDDING_MODEL` | from preset | Hugging Face ID of an embedding model. Overrides the preset. |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `mps` (Apple Silicon). |
| `MAX_NEW_TOKENS` | `512` | Longest answer the model writes. |
| `TEMPERATURE` | `0.7` | Sampling temperature. `0` gives greedy, repeatable answers. |
| `CHUNK_SIZE` | `1000` | Characters per chunk. Changing it applies to new uploads only. |
| `CHUNK_OVERLAP` | `200` | Characters shared by neighbouring chunks. |
| `MIN_CHUNK_CHARS` | `50` | Chunks shorter than this are dropped. |
| `RETRIEVAL_METHOD` | `mmr` | `mmr`, `similarity`, or `hybrid`. |
| `TOP_K` | `4` | Passages sent to the model for each question. |
| `FETCH_K` | `20` | Candidates considered before MMR or hybrid fusion. |
| `MMR_LAMBDA` | `0.5` | MMR balance. `0` favours variety, `1` favours relevance. |
| `HYBRID_ALPHA` | `0.7` | Semantic weight in hybrid search. `0` uses keywords only. |
| `VECTOR_DB_PATH` | `vector_db` | Folder for the Chroma database. |
| `DOCUMENTS_PATH` | `documents` | Folder where the web UI saves uploads. |

### Model presets

| Preset | Chat model | Embedding model | Suggested hardware |
| --- | --- | --- | --- |
| `fast` | Qwen2.5-1.5B-Instruct | bge-small-en-v1.5 | Any laptop |
| `balanced` (default) | Qwen2.5-3B-Instruct | bge-small-en-v1.5 | 8 GB RAM |
| `quality` | Qwen2.5-7B-Instruct | bge-base-en-v1.5 | GPU or 16 GB RAM |
| `code` | Qwen2.5-Coder-7B-Instruct | bge-base-en-v1.5 | GPU or 16 GB RAM |

## Retrieval methods

- **MMR** fetches `FETCH_K` candidates and keeps `TOP_K` that are relevant and different from each other. Good default for long documents with repeated content.
- **Similarity** returns the closest `TOP_K` chunks. Fastest, and least varied.
- **Hybrid** blends embedding similarity with BM25 keyword scores. Helps with exact names, codes, and rare terms.

## Project layout

```
src/palimpsest/
├── app.py                # Streamlit interface
├── cli.py                # `palimpsest` command
├── config.py             # Settings, RetrievalSettings, and model presets
├── engine.py             # RAGEngine and per-session Conversation
├── markdown_safety.py    # Sanitizes model and document text before display
├── models.py             # Answer, SourceChunk, IndexedSource, and report types
├── models_factory.py     # Builds the embedder and chat model
├── documents/
│   ├── loaders.py        # PDF, TXT, MD, and DOCX readers
│   └── chunking.py       # Splits pages into chunks
└── retrieval/
    ├── ranking.py        # BM25 index and reciprocal rank fusion (pure functions)
    └── store.py          # Chroma vector store
tests/                    # pytest suite. Uses fake models, so no downloads.
```

## Privacy and network use

- **Documents and questions stay local.** Indexing, search, and answers run on your machine.
- **The web UI listens on `localhost` only.** It has no login, so `palimpsest ui` and `.streamlit/config.toml` keep it off your network.
- **Telemetry is off.** Chroma's anonymous telemetry and Streamlit's usage statistics are both disabled.
- **Model downloads.** The first run downloads the models from huggingface.co, and later runs check it for updates. After the first run, set `HF_HUB_OFFLINE=1` to work fully offline, and set `HF_HUB_DISABLE_TELEMETRY=1` to turn off Hugging Face's own telemetry.
- **Untrusted content.** A document can try to steer the model with hidden instructions. The UI removes image embeds from answers and shows link targets in plain text, so a reply cannot quietly send data to another site.
- **Sessions.** Each browser tab keeps its own chat history and retrieval options. The indexed documents are shared, because they live in one database.

## Development

```bash
uv run pytest             # run the test suite
uv run ruff format        # format the code
uv run ruff check --fix   # lint: PEP 8, PEP 257 docstrings, naming, bandit security, and more
uv run pyright            # type check in strict mode
```

The tests use LangChain's fake embedder and fake chat model. They run without downloading any model.

## Troubleshooting

- **Out of memory when loading a model.** Use `MODEL_PRESET=fast`, or set `DEVICE=cpu` if the GPU runs out of memory.
- **Answers are slow.** Use a smaller preset, or install a CUDA build of PyTorch that matches your driver.
- **Answers do not match the documents.** Try `RETRIEVAL_METHOD=hybrid`, or raise `TOP_K`.
- **A file fails to index.** Check that it is not scanned images. PDFs need selectable text, because the app does not run OCR.

## Upgrading from 1.x

Version 2 is a breaking change.

- The Python floor is now 3.14. Install with `uv sync`, not `pip install -r requirements.txt`. The requirements file is gone.
- The presets `gpt2`, `llama2`, `gemma`, `mixtral`, and `mistral` were removed. They were legacy, gated, or too large for most machines. Use `balanced`, `quality`, or set `LLM_MODEL`.
- `LOAD_IN_8BIT`, `DEVICE_MAP`, `TORCH_DTYPE`, and `CHUNKING_STRATEGY` were removed. The bitsandbytes dependency was removed too. The chunker now picks separators by file type.
- The old `scripts/` launchers and `config/` helpers were replaced by the `palimpsest` command.
- Re-run `palimpsest ingest` on your files, because the chunk metadata changed.

## License

MIT
