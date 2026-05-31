#!/usr/bin/env bash
# Launch the EyeQ live dashboard with the Qt platform-plugin path pinned to the
# venv's bundled PySide6 plugins. This is needed with a framework Python, where
# Qt otherwise resolves the platform plugin relative to the interpreter app
# bundle and fails with: Could not find the Qt platform plugin "cocoa" in "".
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "error: $PY not found. Create the venv first:" >&2
  echo "  python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev,sim,gui]'" >&2
  exit 1
fi

PLUGINS="$("$PY" -c 'import os, PySide6; print(os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins", "platforms"))')"
export QT_QPA_PLATFORM_PLUGIN_PATH="$PLUGINS"

exec "$PY" -m eyeq.gui.dashboard "$@"
