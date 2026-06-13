#!/usr/bin/env bash
# PreToolUse hook (Write|Edit): blocks the agent from writing to secret/credential
# files and from modifying its own policy files (settings, hooks, MCP config, git
# internals). Mirrors the permission deny rules because hook enforcement is the
# reliable layer.
# Exit 0 = allow. Exit 2 = block; stderr explains why to Claude.
set -uo pipefail

input=$(cat)

if command -v jq >/dev/null 2>&1; then
  file_path=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
else
  file_path=$(printf '%s' "$input" | grep -oE '"file_path"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*:[[:space:]]*"([^"]+)"/\1/')
fi
[[ -z "${file_path}" ]] && exit 0

# Templates with placeholder values are fine.
case "${file_path}" in
  *.env.example | *.env.template | *.env.sample) exit 0 ;;
esac

blocked_patterns=(
  '(^|/)\.env$'
  '(^|/)\.env\.'
  '(^|/)\.envrc$'
  '(^|/)\.dev\.vars'
  '(^|/)secrets?\.'
  '\.pem$'
  '\.key$'
  '\.p12$'
  '\.pfx$'
  '(^|/)id_(rsa|ed25519)'
  '(^|/)\.npmrc$'
  '(^|/)\.pypirc$'
  '(^|/)\.terraformrc$'
  '(^|/)\.netrc$'
  'gha-creds-.*\.json$'
  '(^|/)\.claude/settings(\.local)?\.json$'
  '(^|/)\.claude/hooks/'
  '(^|/)\.mcp\.json$'
  '(^|/)\.git/'
)

for pattern in "${blocked_patterns[@]}"; do
  if [[ "${file_path}" =~ ${pattern} ]]; then
    cat >&2 <<MSG
blocked: writes to secret/credential files are not permitted (hook: block-secret-writes.sh).
file: ${file_path}  (matched: ${pattern})
If a template is needed, write .env.example with placeholder values instead.
Policy files (.claude/settings*, hooks, .mcp.json, .git internals) are changed by the user, not the agent.
MSG
    exit 2
  fi
done
exit 0
