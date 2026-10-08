# AGENTS.md

<!-- Cross-tool contract. Read natively by Cursor, Codex, Copilot, Devin, Zed, and
     most other agents; Claude Code imports it via CLAUDE.md. Keep under ~150 lines —
     every line costs context in every session. -->

## Project

- **Name**: askmydocs (formerly langchain-rag-app)
- **Purpose**: A Retrieval-Augmented Generation app. Ingest documents, embed them
  into a vector store, and answer questions with citations through a Streamlit UI or a CLI.
- **Stack**: Python 3.14+ · LangChain 1.x · Chroma · Hugging Face models (local,
  no API keys) · Streamlit · pydantic-settings. Managed with uv.

## Layout

- `src/askmydocs/app.py` — Streamlit UI. Keep it thin. Logic belongs in the engine.
- `src/askmydocs/cli.py` — the `askmydocs` command (`ingest`, `ask`, `status`, `ui`).
- `src/askmydocs/config.py` — `Settings` (pydantic-settings) and model presets.
- `src/askmydocs/engine.py` — `RAGEngine` (shared, stateless per user) and `Conversation`
  (one per user session: history and retrieval options).
- `src/askmydocs/markdown_safety.py` — sanitizes model and document text before the UI renders it.
- `src/askmydocs/models.py` — shared dataclasses (`Answer`, `SourceChunk`, reports).
- `src/askmydocs/models_factory.py` — builds the embedder and chat model.
- `src/askmydocs/documents/` — file loaders and chunking.
- `src/askmydocs/retrieval/` — Chroma store and pure ranking functions.
- `tests/` — pytest suite. Uses LangChain fakes. Never downloads a model.

## Commands

```bash
uv sync                                  # install locked dependencies (dev group included)
uv run pytest                            # run the test suite
uv run pytest tests/test_engine.py       # one file
uv run ruff format                       # format (line length 99, the PEP 8 maximum)
uv run ruff check --fix                  # lint: pycodestyle, pydocstyle (Google), bandit, ...
uv run pyright                           # type check, strict mode
uv run askmydocs ui                      # Streamlit UI on http://localhost:8501
uv run askmydocs ingest <files...>       # index documents from the command line
uv run askmydocs ask "<question>"        # ask from the command line
```

Before you call a change finished, run `uv run ruff check`, `uv run ruff format --check`,
`uv run pyright`, and `uv run pytest`.

## Conventions

- Python standards live in `.claude/skills/python-standards/SKILL.md`. They apply
  to every agent, not just Claude. Read them before you write or refactor Python.
- Use `uv add` and `uv run`. Never `pip install` into the project environment.
  Commit `uv.lock`.
- Import heavy packages (`torch`, `transformers`, `langchain_huggingface`) inside
  functions when only the runtime path needs them. Tests and CLI help must stay fast.
- Put logic in `engine.py` and the pure modules. Keep `app.py` and `cli.py` thin.
- Inject models and stores. Do not mock what the project owns. For true external
  models, use the fakes in `langchain_core`.
- Keep diffs small and focused. One logical change per commit. Use conventional
  commit messages (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`).

## Claude Code hooks

- **On every Write/Edit**, a hook formats and lints changed files when a formatter
  is available. Ruff is configured, so Python files get formatted.
- **When a turn ends**, a Stop hook runs `pytest` for Python files changed in the
  session. Fix failures before you stop.
- **Guard hooks** block writes to secret files, shell reads of secrets, env dumps,
  destructive commands, edits to policy files, and unlisted WebFetch or package runners.
