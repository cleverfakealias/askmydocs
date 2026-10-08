@AGENTS.md

## Claude Code specifics

Automation in this repo (configured in `.claude/settings.json`):

- **On every Write/Edit** a hook auto-formats and lints changed files when a
  formatter is available. Ruff is configured in `pyproject.toml`, so Python
  files get formatted on save.
- **When you finish a turn**, a Stop hook runs `pytest` for Python files changed
  this session. If it blocks you, fix the failures; it won't loop (it lets you
  stop on the second attempt). The suite uses fake models, so it runs in seconds.
- **Guard hooks** block: writes to secret files; shell reads of secrets
  (`cat .env`, `~/.ssh`, etc.); env dumps (`printenv`); destructive commands;
  editing policy files (`.claude/settings*`, hooks, `.mcp.json`, `.git/`,
  CI workflows); WebFetch outside `.claude/hooks/allowed-domains.txt`; and
  npx/uvx/pnpm-dlx for packages not in `.claude/hooks/allowed-run-packages.txt`.
 