#!/usr/bin/env bash
# Stop hook: when the turn touched Python, TS/JS, or Lua files, run typecheck + tests once
# before Claude finishes. Exit 0 = pass or nothing to do. Exit 2 = failures; Claude
# keeps working and sees stderr.
#
# Escape hatches:
#   - stop_hook_active guard prevents infinite loops (required; do not remove)
#   - CLAUDE_SKIP_STOP_TESTS=1 disables this hook entirely
set -uo pipefail

input=$(cat)

# Loop guard: if we're already continuing because this hook blocked once, let Claude stop.
if command -v jq >/dev/null 2>&1; then
  active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false')
else
  case "$input" in
    *'"stop_hook_active":true'* | *'"stop_hook_active": true'*) active=true ;;
    *) active=false ;;
  esac
fi
[[ "${active}" == "true" ]] && exit 0
[[ "${CLAUDE_SKIP_STOP_TESTS:-0}" == "1" ]] && exit 0

root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "${root}" || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

changed=$({ git diff --name-only HEAD -- 2>/dev/null; git ls-files --others --exclude-standard; } | sort -u)
[[ -z "${changed}" ]] && exit 0

py_changed=$(grep -E '\.py$' <<<"${changed}" || true)
ts_changed=$(grep -E '\.(ts|tsx|js|jsx|mjs|cjs)$' <<<"${changed}" || true)
lua_changed=$(grep -E '\.lua$' <<<"${changed}" || true)
[[ -z "${py_changed}" && -z "${ts_changed}" && -z "${lua_changed}" ]] && exit 0

fail=""
TAIL_LINES=60
note_failure() { fail+="── $1 ──"$'\n'"$(tail -n ${TAIL_LINES} <<<"$2")"$'\n'; }

# ── Python: pytest ─────────────────────────────────────────────────────────────
if [[ -n "${py_changed}" && -f "pyproject.toml" ]]; then
  PYTEST=()
  if command -v uv >/dev/null 2>&1 && uv run --no-sync pytest --version >/dev/null 2>&1; then
    PYTEST=(uv run --no-sync pytest)
  elif command -v pytest >/dev/null 2>&1; then
    PYTEST=(pytest)
  fi
  if ((${#PYTEST[@]})); then
    if ! out=$("${PYTEST[@]}" -q -x 2>&1); then
      note_failure "pytest" "${out}"
    fi
  fi
fi

# ── TypeScript/JavaScript: tsc --noEmit, then vitest ──────────────────────────
if [[ -n "${ts_changed}" && -f "package.json" ]]; then
  RUN=()
  if [[ -f "pnpm-lock.yaml" ]] && command -v pnpm >/dev/null 2>&1; then
    RUN=(pnpm exec)
  elif command -v npx >/dev/null 2>&1; then
    RUN=(npx --no-install)
  fi
  if ((${#RUN[@]})); then
    if compgen -G "tsconfig*.json" >/dev/null; then
      if ! out=$("${RUN[@]}" tsc --noEmit 2>&1); then
        note_failure "tsc --noEmit" "${out}"
      fi
    fi
    if compgen -G "vitest.config.*" >/dev/null || grep -q '"vitest"' package.json 2>/dev/null; then
      if ! out=$("${RUN[@]}" vitest run --reporter=dot 2>&1); then
        note_failure "vitest" "${out}"
      fi
    elif command -v jq >/dev/null 2>&1 \
        && jq -e '.scripts.test and (.scripts.test | test("no test specified") | not)' package.json >/dev/null 2>&1; then
      if ! out=$(npm test --silent 2>&1); then
        note_failure "npm test" "${out}"
      fi
    fi
  fi
fi

# ── Lua: busted ────────────────────────────────────────────────────────────────
if [[ -n "${lua_changed}" ]] && command -v busted >/dev/null 2>&1; then
  if [[ -f ".busted" ]] || compgen -G "*.rockspec" >/dev/null || [[ -d "spec" ]]; then
    if ! out=$(busted -o plainTerminal 2>&1); then
      note_failure "busted" "${out}"
    fi
  fi
fi

if [[ -n "${fail}" ]]; then
  printf 'Typecheck/tests failed for code changed this session. Fix before finishing:\n%s' "${fail}" >&2
  exit 2
fi
exit 0
