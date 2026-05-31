"""GUI (Phase 4 — a thin, swappable client over the headless engine).

* :mod:`eyeq.gui.dashboard` — the dockarea dashboard: eye-density ImageItem,
  frequency cascade, SBR, amplitude histograms, slider panel, start/stop,
  config save/load.
* :mod:`eyeq.gui.binding`   — auto-generates controls from each block's ``params``
  schema and routes slider changes by ``Kind`` (LTI -> statistical recompute;
  NONLINEAR -> transient worker; STRUCTURAL -> rebuild).
* :mod:`eyeq.gui.plots`     — the pyqtgraph plot widgets.

The GUI imports PyQtGraph + PySide6 (the ``gui`` optional dependency); the engine
never imports the GUI.
"""

# Pin Qt's platform-plugin path to PySide6's own plugins. With a framework Python
# (e.g. python.org's), Qt otherwise resolves plugins relative to the interpreter
# app bundle (".../Python.app/Contents/MacOS/platforms"), which is empty, and
# fails with: Could not find the Qt platform plugin "cocoa" in "". Setting this
# before any QApplication is created makes the bundled plugins discoverable.
import os as _os


def _pin_qt_plugin_path() -> None:
    if _os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return
    try:
        import PySide6
    except ImportError:
        return
    platforms = _os.path.join(_os.path.dirname(PySide6.__file__), "Qt", "plugins", "platforms")
    if _os.path.isdir(platforms):
        _os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platforms


_pin_qt_plugin_path()
