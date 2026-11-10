#!/usr/bin/env bash
# Sets up the bridge in this folder on a Mac: Python environment, models, .env.
#   scripts/install_mac.sh             set up and run the tests
#   scripts/install_mac.sh --launchd   also keep it running with launchd (starts at login)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

if [ ! -d .venv ]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.12 .venv
  else
    PY=""
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
      if command -v $c >/dev/null && $c -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then PY=$c; break; fi
    done
    [ -n "$PY" ] || { echo "Need Python 3.10 or newer (brew install python@3.12)"; exit 1; }
    $PY -m venv .venv
  fi
fi
if command -v uv >/dev/null; then uv pip install -q --python .venv/bin/python -r requirements.txt
else .venv/bin/python -m pip install -q -r requirements.txt; fi

scripts/fetch_models.sh

if [ ! -f .env ]; then
  cp .env.example .env
  TOKEN=$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')
  sed -i '' "s|^OMIBRIDGE_TOKEN=.*|OMIBRIDGE_TOKEN=$TOKEN|" .env
  chmod 600 .env
  echo "Created .env with a new webhook token. Add OPENROUTER_API_KEY to it."
fi

.venv/bin/python -m pytest -q tests

if [ "${1:-}" = "--launchd" ]; then
  PLIST=~/Library/LaunchAgents/com.omiomy.bridge.plist
  mkdir -p launchd logs
  sed -e "s|__ROOT__|$ROOT|g" scripts/com.omiomy.bridge.plist > launchd/com.omiomy.bridge.plist
  cp launchd/com.omiomy.bridge.plist "$PLIST"
  launchctl bootout "gui/$(id -u)/com.omiomy.bridge" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "Bridge is running. Review page: http://127.0.0.1:8378/"
fi
