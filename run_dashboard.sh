#!/usr/bin/env bash
# Launch the EyeQ live dashboard.
#
# The venv lives OUTSIDE the iCloud-synced project folder (~/eyeq-venv): iCloud's
# "Desktop & Documents" sync creates conflict copies (e.g. "libqcocoa 2.dylib")
# of the binary packages while pip writes them, which makes Qt see duplicate
# plugins and fail with: Could not find the Qt platform plugin "cocoa".
#
# The Qt platform-plugin path is also pinned (framework Python otherwise looks in
# the interpreter app bundle and fails with: ... plugin "cocoa" in "").
set -euo pipefail
cd "$(dirname "$0")"

PY="$HOME/eyeq-venv/bin/python"
[[ -x "$PY" ]] || PY=".venv/bin/python"   # fallback for an in-project venv
if [[ ! -x "$PY" ]]; then
  echo "error: no venv found. Create one OUTSIDE iCloud:" >&2
  echo "  python3.11 -m venv ~/eyeq-venv && ~/eyeq-venv/bin/pip install -e '.[dev,sim,gui]'" >&2
  exit 1
fi

PLUGINS="$("$PY" -c 'import os, PySide6; print(os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins", "platforms"))')"
export QT_QPA_PLATFORM_PLUGIN_PATH="$PLUGINS"

exec "$PY" -m eyeq.gui.dashboard "$@"
