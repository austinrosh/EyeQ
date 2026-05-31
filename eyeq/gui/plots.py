"""pyqtgraph plot widgets (Phase 4 — placeholder).

The eye density is rendered with ``pyqtgraph.ImageItem`` using a fixed LUT and
fixed levels each frame (``autoLevels=False``) to avoid flicker; the frequency
cascade, SBR, and amplitude histograms are pyqtgraph line plots. A ``QTimer`` at
~30 FPS pulls the latest density snapshot, decoupled from the sim batch cadence.
"""
