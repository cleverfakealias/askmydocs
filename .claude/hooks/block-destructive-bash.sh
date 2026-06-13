#!/usr/bin/env bash
# PreToolUse hook (Bash): defense-in-depth guard for shell commands.
# Blocks: destructive ops, secret-file reads (the Bash bypass of Read deny rules),
# env dumps, agent self-modification of policy files, and un-allowlisted one-off
# remote package execution (npx/uvx/pnpm dlx — the npm-worm vector).
# Exit 0 = allow. Exit 2 = block; stderr explains why to Claude.
#
# NOTE: permission deny rules only constrain Claude's Read/Edit tools — Bash can
# bypass them, which is why this hook exists. The OS sandbox (see
# settings.local.json.example) is the final boundary; this hook is the middle layer.
set -uo pipefail

input=$(cat)

if command -v jq >/dev/null 2>&1; then
  cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty')
else
  cmd=$(printf '%s' "$input" | grep -oE '"command"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed -E 's/.*:[[:space:]]*"([^"]*)"/\1/')
fi
[[ -z "${cmd}" ]] && exit 0

deny() {
  printf 'blocked: %s (hook: block-destructive-bash.sh)\ncommand: %s\n' "$1" "$2" >&2
  exit 2
}

# ── destructive operations ────────────────────────────────────────────────────
re_rm_critical='rm[[:space:]]+-[a-zA-Z]*[rR][a-zA-Z]*[[:space:]]+(/|~|\$HOME|\.|\.git)([[:space:]]|$|/)'
re_force_push='git[[:space:]]+push[[:space:]].*(--force|[[:space:]]-f)([[:space:]]|$)'
re_force_lease='--force-with-lease'
re_reset_hard='git[[:space:]]+reset[[:space:]]+--hard'
re_git_clean='git[[:space:]]+clean[[:space:]]+-[a-zA-Z]*f'
re_pipe_shell='(curl|wget)[[:space:]][^|]*\|[[:space:]]*(sudo[[:space:]]+)?(ba|z|fi)?sh'
re_chmod_777='chmod[[:space:]]+(-[a-zA-Z]+[[:space:]]+)*777'
re_raw_disk='(^|[[:space:]])(mkfs|dd[[:space:]]+if=)'
re_npm_scripts='--ignore-scripts=false'

[[ "${cmd}" =~ ${re_rm_critical} ]] && deny "recursive delete of a critical path" "${cmd}"
if [[ "${cmd}" =~ ${re_force_push} ]] && [[ ! "${cmd}" =~ ${re_force_lease} ]]; then
  deny "force push (ask the user; --force-with-lease only with explicit approval)" "${cmd}"
fi
[[ "${cmd}" =~ ${re_reset_hard} ]] && deny "git reset --hard discards uncommitted work" "${cmd}"
[[ "${cmd}" =~ ${re_git_clean} ]] && deny "git clean -f deletes untracked files" "${cmd}"
[[ "${cmd}" =~ ${re_pipe_shell} ]] && deny "piping a download straight into a shell" "${cmd}"
[[ "${cmd}" =~ ${re_chmod_777} ]] && deny "world-writable permissions" "${cmd}"
[[ "${cmd}" =~ ${re_raw_disk} ]] && deny "raw disk operation" "${cmd}"
[[ "${cmd}" =~ ${re_npm_scripts} ]] && deny "explicitly re-enabling npm lifecycle scripts" "${cmd}"

# ── secret reads via shell (Read deny rules do not cover Bash) ────────────────
re_read_verb='(^|[;&|[:space:]])(cat|less|more|head|tail|grep|rg|cp|mv|scp|rsync|base64|xxd|od|strings|source|\.)[[:space:]]'
re_secret_path='(~/\.ssh/|~/\.aws/|~/\.kube/|~/\.gnupg/|~/\.docker/config\.json|(^|[/[:space:]"'"'"'])\.netrc|(^|[/[:space:]"'"'"'])\.pypirc|(^|[/[:space:]"'"'"'])\.npmrc|id_rsa|id_ed25519|\.pem([[:space:]"'"'"']|$)|(^|[/[:space:]"'"'"'])\.env(\.[A-Za-z0-9_-]+)?([[:space:]"'"'"']|$))'
re_secret_exempt='\.env\.(example|template|sample)'
if [[ "${cmd}" =~ ${re_read_verb} && "${cmd}" =~ ${re_secret_path} ]] && [[ ! "${cmd}" =~ ${re_secret_exempt} ]]; then
  deny "shell read of a secret/credential file (use .env.example for templates)" "${cmd}"
fi
re_env_dump='(^|[;&|[:space:]])(printenv|env[[:space:]]*($|[|>]))'
[[ "${cmd}" =~ ${re_env_dump} ]] && deny "dumping the process environment (secrets live there)" "${cmd}"

# ── self-modification of agent policy files ───────────────────────────────────
re_policy_path='(\.claude/(settings(\.local)?\.json|hooks)|\.mcp\.json|\.git/hooks|\.github/workflows)'
re_write_verb='(>|>>|[[:space:]]tee[[:space:]]|sed[[:space:]]+-i|mv[[:space:]]|cp[[:space:]]|rm[[:space:]]|truncate|chmod)'
if [[ "${cmd}" =~ ${re_policy_path} && "${cmd}" =~ ${re_write_verb} ]]; then
  deny "modifying agent policy / CI files via shell (settings, hooks, .mcp.json, workflows) — ask the user" "${cmd}"
fi

# ── one-off remote package execution (npx/uvx/dlx) ────────────────────────────
re_runner='(^|[;&|[:space:]])(npx|uvx|pipx[[:space:]]+run|pnpm[[:space:]]+dlx|yarn[[:space:]]+dlx|bunx)[[:space:]]'
if [[ "${cmd}" =~ ${re_runner} ]] && [[ ! "${cmd}" =~ npx[[:space:]]+--no-install ]]; then
  pkg=$(printf '%s' "${cmd}" | sed -E 's/.*(npx|uvx|pipx[[:space:]]+run|pnpm[[:space:]]+dlx|yarn[[:space:]]+dlx|bunx)[[:space:]]+//' \
        | tr ' ' '\n' | grep -v '^-' | head -1)
  pkg_base=$(printf '%s' "${pkg}" | sed -E 's/(.)@[^@]*$/\1/')   # strip @version, keep @scope
  allowlist="${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/hooks/allowed-run-packages.txt"
  allowed=false
  if [[ -f "${allowlist}" ]] && grep -qxF "${pkg_base}" <(grep -v '^[[:space:]]*#' "${allowlist}"); then
    allowed=true
  fi
  if [[ "${allowed}" != "true" ]]; then
    deny "one-off remote package execution of '${pkg_base}' — packages run code on download (npm-worm vector). If legitimate, the user can add it to .claude/hooks/allowed-run-packages.txt" "${cmd}"
  fi
fi

exit 0
