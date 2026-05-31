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
