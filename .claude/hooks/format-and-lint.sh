#!/usr/bin/env bash
# PostToolUse hook (Write|Edit): auto-format + lint the file Claude just touched.
# Exit 0 = clean (silent). Exit 2 = remaining problems; stderr is fed back to Claude.
#
# Python:     ruff format + ruff check --fix   (via uv when the project uses it)
# TS/JS/etc.: biome check --write              (eslint/prettier fallback if no biome config)
# Lua:        stylua, then selene or luacheck  (no Lua linter has autofix)
set -uo pipefail

input=$(cat)

if command -v jq >/dev/null 2>&1; then
  file_path=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
else
  file_path=$(printf '%s' "$input" | grep -oE '"file_path"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*:[[:space:]]*"([^"]+)"/\1/')
fi

[[ -z "${file_path}" || ! -f "${file_path}" ]] && exit 0

root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "${root}" || exit 0
errors=""

note_failure() { errors+="── $1 ──"$'\n'"$2"$'\n'; }

case "${file_path##*.}" in
  py)
    RUFF=()
    if [[ -f "pyproject.toml" || -f "uv.lock" ]] && command -v uv >/dev/null 2>&1 \
       && uv run --no-sync ruff --version >/dev/null 2>&1; then
      RUFF=(uv run --no-sync ruff)
    elif command -v ruff >/dev/null 2>&1; then
      RUFF=(ruff)
    fi
    if ((${#RUFF[@]})); then
      "${RUFF[@]}" format --quiet "${file_path}" >/dev/null 2>&1 || true
      if ! out=$("${RUFF[@]}" check --fix --output-format=concise "${file_path}" 2>&1); then
        note_failure "ruff check (after autofix)" "${out}"
      fi
    fi
    ;;
  ts|tsx|js|jsx|mjs|cjs|json|jsonc|css)
    BIOME=()
    if [[ -f "biome.json" || -f "biome.jsonc" ]]; then
      if [[ -f "pnpm-lock.yaml" ]] && command -v pnpm >/dev/null 2>&1; then
        BIOME=(pnpm exec biome)
      elif command -v npx >/dev/null 2>&1 && npx --no-install biome --version >/dev/null 2>&1; then
        BIOME=(npx --no-install biome)
      elif command -v biome >/dev/null 2>&1; then
        BIOME=(biome)
      fi
    fi
    if ((${#BIOME[@]})); then
      if ! out=$("${BIOME[@]}" check --write --reporter=summary "${file_path}" 2>&1); then
        note_failure "biome check (after autofix)" "${out}"
      fi
    elif [[ "${file_path##*.}" =~ ^(ts|tsx|js|jsx|mjs|cjs)$ ]] && ls eslint.config.* .eslintrc* >/dev/null 2>&1; then
      if [[ -f "pnpm-lock.yaml" ]] && command -v pnpm >/dev/null 2>&1; then
        ESLINT=(pnpm exec eslint)
      elif command -v npx >/dev/null 2>&1; then
        ESLINT=(npx --no-install eslint)
      else
        ESLINT=()
      fi
      if ((${#ESLINT[@]})); then
        if ! out=$("${ESLINT[@]}" --fix "${file_path}" 2>&1); then
          note_failure "eslint (after autofix)" "${out}"
        fi
      fi
    fi
    ;;
  lua)
    if command -v stylua >/dev/null 2>&1; then
      stylua "${file_path}" >/dev/null 2>&1 || true
    fi
    if [[ -f "selene.toml" ]] && command -v selene >/dev/null 2>&1; then
      if ! out=$(selene -q "${file_path}" 2>&1); then
        note_failure "selene" "${out}"
      fi
    elif command -v luacheck >/dev/null 2>&1; then
      if ! out=$(luacheck --formatter plain "${file_path}" 2>&1); then
        note_failure "luacheck" "${out}"
      fi
    fi
    ;;
esac

if [[ -n "${errors}" ]]; then
  printf 'Auto-fix left unresolved issues in %s — fix them:\n%s' "${file_path}" "${errors}" >&2
  exit 2
fi
exit 0
