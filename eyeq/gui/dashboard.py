"""Live dashboard entry point (Phase 4 — placeholder).

Run as ``python -m eyeq.gui.dashboard --config examples/112g_pam4.yaml``. Wires
the statistical + transient engines to the four plots and the auto-bound slider
panel, with start/stop and config save/load. LTI sliders trigger only a cascade
recompute; the transient engine runs continuously without blocking the UI.
"""
