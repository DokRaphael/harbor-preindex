#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x ./.venv/bin/harbor-preindex ]]; then
  echo "Missing ./.venv/bin/harbor-preindex. Create the local virtualenv and install the project first." >&2
  echo "Run: python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ." >&2
  exit 1
fi

exec ./.venv/bin/harbor-preindex build-index "$@"
