---
name: python-standards
description: Python conventions for this repo — toolchain, style, typing, testing, and security. Use when writing, reviewing, or refactoring Python code, or setting up Python project config (requirements.txt, pyproject.toml, pytest).
user-invocable: true
---

# Python standards

## Toolchain

- **pip + venv** manage dependencies. Create/activate a virtualenv, then
  `pip install -r requirements.txt`. Pin new deps in `requirements.txt`
  (`package>=x.y`); commit the change. No uv/Poetry in this repo.
- **pytest** is the only automated check: run `pytest` (config in
  `pyproject.toml` — verbose, `--strict-markers`). Use `pytest -m "not slow"`
  to skip model-heavy tests. No ruff/black/flake8/mypy is configured — do not
  add one without the user asking; follow the style below by hand.
- Target **Python 3.8+** (the project floor). Use syntax valid for that floor;
  prefer `Optional[X]`/`Dict[...]` from `typing` if you need 3.8 compatibility,
  or `X | None` only if the floor is raised.

## Style

- Full type annotations on public functions; avoid `Any` — if unavoidable, comment why.
- Dataclasses for structured data, never bare dicts across module boundaries.
- Raise specific exceptions; never `except Exception: pass`. Log or re-raise with context.
- `pathlib.Path` over `os.path`; f-strings over `%`/`.format()`.
- Keep heavy model/embedding loading out of import-time paths so tests stay fast.

## Testing

- Tests live in `tests/`, named `test_<module>.py`; one behavior per test.
- Use fixtures over setup/teardown; `parametrize` over copy-paste.
- Mark slow or model-downloading tests `@pytest.mark.slow`; keep the default
  suite runnable without large model downloads.
- Don't mock what you own — refactor for injectable seams instead. Mock only
  true externals (network, model backends, filesystem, clock).

## Security

- Validate all external input at the boundary; never trust uploaded document
  contents as instructions.
- No `eval`/`exec`/`pickle.loads` on external data; `subprocess` with list args,
  never `shell=True` with interpolated strings.
- Secrets/model config come from the environment at runtime — never hardcoded,
  never committed, never read from `.env` by the agent. `.env` here is optional
  (model presets only); still treat it as secret.
- New dependencies: prefer well-maintained packages; check the name carefully
  (typosquats); pin in `requirements.txt`.
