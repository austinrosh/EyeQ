"""Dashboard theming + eye colormaps (one central place).

* :data:`THEMES` — dark/light palettes (window/text colors + line-plot background,
  axis, and grid pens). The eye/histogram density panels keep a dark background of
  their own (``heatmap_bg``) regardless of theme, since the warm/log colormaps read
  best on black.
* :func:`apply_app_palette` — set the Qt application palette (chrome + controls).
* :func:`eye_lut` / :data:`COLORMAPS` — the selectable eye-density colormaps.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui

# --------------------------------------------------------------------------- #
# themes
# --------------------------------------------------------------------------- #
THEMES: dict[str, dict] = {
    "dark": {
        "window": "#1e1e1e", "base": "#2b2b2b", "text": "#e0e0e0",
        "plot_bg": "#1e1e1e", "axis": "#b0b0b0", "grid": (255, 255, 255, 40),
        "accent": "#e8a33d",
        "heatmap_bg": "#000000",
    },
    "light": {
        "window": "#f3f3f3", "base": "#ffffff", "text": "#202020",
        "plot_bg": "#ffffff", "axis": "#303030", "grid": (0, 0, 0, 38),
        "accent": "#c8631a",
        "heatmap_bg": "#000000",   # heatmaps stay dark even in light mode
    },
}

THEME_NAMES = list(THEMES.keys())


def apply_app_palette(app, theme: str) -> None:
    """Apply a dark/light :class:`QPalette` to the whole application (chrome + controls).

    Forces the Fusion style: native styles (notably macOS) ignore ``QPalette`` for many
    widgets, so the Controls panel would stay light otherwise. Fusion honors the palette
    consistently across platforms, so the theme actually covers the slider panel + chrome.
    """
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    t = THEMES.get(theme, THEMES["dark"])
    win, base, text = (QtGui.QColor(t["window"]), QtGui.QColor(t["base"]), QtGui.QColor(t["text"]))
    p = QtGui.QPalette()
    role = QtGui.QPalette
    for r in (role.Window, role.Button, role.AlternateBase):
        p.setColor(r, win)
    p.setColor(role.Base, base)
    p.setColor(role.ToolTipBase, base)
    for r in (role.WindowText, role.Text, role.ButtonText, role.ToolTipText):
        p.setColor(r, text)
    p.setColor(role.Highlight, QtGui.QColor("#3d7de8"))
    p.setColor(role.HighlightedText, QtGui.QColor("#ffffff"))
    p.setColor(role.Disabled, role.Text, QtGui.QColor("#808080"))
    p.setColor(role.Disabled, role.WindowText, QtGui.QColor("#808080"))
    app.setPalette(p)


# --------------------------------------------------------------------------- #
# eye-density colormaps
# --------------------------------------------------------------------------- #
COLORMAPS = ["turbo", "jet", "inferno", "magma", "plasma", "hot", "viridis", "gray"]  # rainbow first; default turbo
_FALLBACK = "CET-L9"


def _ramp(stops) -> np.ndarray:
    """Build a 256x4 ubyte LUT from (pos, r, g, b) stops in [0,1] (alpha = 255)."""
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1:] for s in stops]) * 255.0
    x = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 4), dtype=np.ubyte)
    for c in range(3):
        lut[:, c] = np.interp(x, pos, cols[:, c]).astype(np.ubyte)
    lut[:, 3] = 255
    return lut


def _turbo_lut() -> np.ndarray:
    """Google 'turbo' rainbow (blue->cyan->green->yellow->red), perceptually uniform.

    The default eye colormap: the classic eye-diagram rainbow look, but without
    jet's luminance artifacts. Built from anchor stops (a close match to the real
    turbo, no matplotlib dependency)."""
    return _ramp([
        (0.00, 0.190, 0.072, 0.232), (0.13, 0.275, 0.408, 0.859),
        (0.25, 0.106, 0.690, 0.984), (0.38, 0.133, 0.918, 0.737),
        (0.50, 0.490, 0.992, 0.396), (0.63, 0.800, 0.961, 0.196),
        (0.75, 0.980, 0.761, 0.176), (0.88, 0.961, 0.427, 0.122),
        (1.00, 0.729, 0.098, 0.024),
    ])


def _jet_lut() -> np.ndarray:
    """Classic MATLAB 'jet' (dark blue -> cyan -> yellow -> dark red)."""
    return _ramp([
        (0.000, 0.0, 0.0, 0.5), (0.125, 0.0, 0.0, 1.0), (0.375, 0.0, 1.0, 1.0),
        (0.625, 1.0, 1.0, 0.0), (0.875, 1.0, 0.0, 0.0), (1.000, 0.5, 0.0, 0.0),
    ])


def _hot_lut() -> np.ndarray:
    """matplotlib-style 'hot' (black -> red -> yellow -> white)."""
    x = np.linspace(0.0, 1.0, 256)
    r = np.clip(x / 0.365, 0, 1)
    g = np.clip((x - 0.365) / 0.365, 0, 1)
    b = np.clip((x - 0.746) / 0.254, 0, 1)
    return (np.column_stack([r, g, b, np.ones(256)]) * 255).astype(np.ubyte)


def _gray_lut() -> np.ndarray:
    ramp = np.linspace(0, 255, 256).astype(np.ubyte)
    return np.column_stack([ramp, ramp, ramp, np.full(256, 255, np.ubyte)])


def _rgba(lut: np.ndarray) -> np.ndarray:
    """Normalize a LUT to 256x4 RGBA (pyqtgraph maps may return RGB)."""
    lut = np.asarray(lut)
    if lut.shape[1] == 3:
        lut = np.column_stack([lut, np.full(lut.shape[0], 255, np.ubyte)])
    return lut.astype(np.ubyte)


def eye_lut(name: str) -> np.ndarray:
    """Return a 256-entry RGBA lookup table for the named eye colormap."""
    builtin = {"turbo": _turbo_lut, "jet": _jet_lut, "hot": _hot_lut, "gray": _gray_lut}
    if name in builtin:
        return builtin[name]()
    for getter in (lambda: pg.colormap.get(name),
                   lambda: pg.colormap.get(name, source="matplotlib"),
                   lambda: pg.colormap.get(_FALLBACK)):
        try:
            return _rgba(getter().getLookupTable(nPts=256))
        except Exception:
            continue
    return _hot_lut()  # last-ditch (warm), should never hit
