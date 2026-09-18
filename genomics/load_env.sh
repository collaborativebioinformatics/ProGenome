#!/usr/bin/env bash
# Source this from the shell scripts:  . "$(dirname "$0")/load_env.sh"
# Same rules as config.py: exported variables win, then genomics/.env, then ~/.progenome.env.
# Lines are KEY=VALUE (optional 'export '), quotes stripped, full-line comments ignored.
_progenome_load() {
  [ -f "$1" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    line="${line#export }"
    key="${line%%=*}"; val="${line#*=}"
    val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
    if [ -z "${!key:-}" ]; then export "$key=$val"; fi
  done < "$1"
}
_progenome_load "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env"
_progenome_load "$HOME/.progenome.env"
