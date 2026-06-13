#!/usr/bin/env bash
# PreToolUse hook (WebFetch): allow fetches only to domains in
# .claude/hooks/allowed-domains.txt (one per line, # comments; subdomains of a
# listed domain are allowed). Fetched web content is the primary prompt-injection
# vector for coding agents — keep this list short and boring.
# Exit 0 = allow. Exit 2 = block. Fails CLOSED if the allowlist file is missing.
set -uo pipefail

input=$(cat)

if command -v jq >/dev/null 2>&1; then
  url=$(printf '%s' "$input" | jq -r '.tool_input.url // empty')
else
  url=$(printf '%s' "$input" | grep -oE '"url"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*:[[:space:]]*"([^"]+)"/\1/')
fi
[[ -z "${url}" ]] && exit 0

host=$(printf '%s' "${url}" | sed -E 's#^[a-zA-Z][a-zA-Z0-9+.-]*://##; s#[/:?].*$##' | tr 'A-Z' 'a-z')
[[ -z "${host}" ]] && exit 0

allowlist="${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/hooks/allowed-domains.txt"
if [[ ! -f "${allowlist}" ]]; then
  printf 'blocked: WebFetch allowlist missing (%s). Create it with one domain per line.\n' "${allowlist}" >&2
  exit 2
fi

while IFS= read -r domain; do
  domain=$(printf '%s' "${domain}" | tr -d '[:space:]' | tr 'A-Z' 'a-z')
  [[ -z "${domain}" || "${domain}" == \#* ]] && continue
  if [[ "${host}" == "${domain}" || "${host}" == *".${domain}" ]]; then
    exit 0
  fi
done < "${allowlist}"

cat >&2 <<MSG
blocked: '${host}' is not in the WebFetch domain allowlist (hook: webfetch-allowlist.sh).
Fetched pages can carry prompt-injection payloads. If this domain is needed,
the user can add it to .claude/hooks/allowed-domains.txt.
MSG
exit 2
