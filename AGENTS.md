# AGENTS.md

<!-- Cross-tool contract. Read natively by Cursor, Codex, Copilot, Devin, Zed, and
     most other agents; Claude Code imports it via CLAUDE.md. Keep under ~150 lines —
     every line costs context in every session. -->

## Project

- **Name**: langchain-rag-app
- **Purpose**: A Retrieval-Augmented Generation app — ingest documents, embed them
  into a vector store, and answer questions through a conversational Streamlit UI.
- **Stack**: Python 3.8+ · LangChain · ChromaDB · HuggingFace models (local,
  no API keys) · Streamlit UI. Dependencies managed with pip + `requirements.txt`.

## Layout

- `src/rag_engine/` — core RAG orchestration (retrieval + answer generation).
- `src/document_processor/` — document loading and chunking (PDF/TXT/MD/DOCX).
- `src/vector_store/` — ChromaDB vector store operations.
- `src/web_interface/` — Streamlit app (`app.py`) and UI helpers.
- `src/utils/` — shared helpers.
- `config/` — setup and configuration scripts.
- `scripts/` — launchers (`run.py`, `run_web.py`, `run.sh`, `run.bat`, `main.py`).
- `tests/` — pytest suite (`test_*.py`).

## Commands

```bash
# Create and activate a virtual environment first, then:
pip install -r requirements.txt        # install deps (includes pytest)

pytest                                 # run the test suite (config in pyproject.toml)
pytest -m "not slow"                   # skip slow/model-heavy tests
pytest tests/test_vector_store.py      # single test file

# Run the app (Streamlit UI on http://localhost:8501):
streamlit run src/web_interface/app.py
python scripts/run.py                  # launcher (also run.sh / run.bat)

# Configuration helpers:
python config/setup.py                 # automated environment setup
python config/show_config.py           # print active model/config options
```

There is no separate lint, format, or type-check tool configured in this repo —
`pytest` is the only automated check. Do not introduce ruff/black/mypy config
unless the user asks; follow the conventions below by hand instead.

## Conventions

- Python standards live in `.claude/skills/python-standards/SKILL.md` — they apply
  to every agent, not just Claude. Read it before writing or refactoring Python.
- Keep diffs small and focused. One logical change per commit; conventional commit
  messages (`feat:`, `fix:`, `refactor:`, `chore:`, ...).
- Te