"""Dashboard theming + eye colormaps (one central place).

* :data:`THEMES` — dark/light palettes: window/panel/text colors, a cyan-teal accent,
  and the line-plot background/axis/grid pens. The eye/histogram density panels keep a
  dark background of their own (``heatmap_bg``) regardless of theme, since the warm/log
  colormaps read best on black.
* :func:`apply_app_palette` — set the Qt application palette *and* the app-wide
  stylesheet (the modern, crisp chrome: rounded flat controls, accent slider/handles).
* :func:`eye_lut` / :data:`COLORMAPS` — the selectable eye-density colormaps.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui

# --------------------------------------------------------------------------- #
# themes — a modern "instrument" look: deep charcoal dark / crisp light, cyan-teal accent
# --------------------------------------------------------------------------- #
THEMES: dict[str, dict] = {
    "dark": {
        "window": "#0d1117", "base": "#161b22", "elevated": "#1c2430",
        "text": "#e6edf3", "muted": "#8b949e", "border": "#2a313c",
        "accent": "#2dd4bf", "accent2": "#22d3ee", "on_accent": "#06121a",
        "plot_bg": "#0d1117", "axis": "#8b949e", "grid": (255, 255, 255, 24),
        "heatmap_bg": "#000000",
    },
    "light": {
        "window": "#eef1f5", "base": "#ffffff", "elevated": "#f4f6f9",
        "text": "#1f2328", "muted": "#6e7781", "border": "#d0d7de",
        "accent": "#0d9488", "accent2": "#0891b2", "on_accent": "#ffffff",
        "plot_bg": "#ffffff", "axis": "#57606a", "grid": (0, 0, 0, 30),
        "heatmap_bg": "#000000",   # heatmaps stay dark even in light mode
    },
}

THEME_NAMES = list(THEMES.keys())


def app_stylesheet(theme: str) -> str:
    """Return the app-wide QSS for the modern chrome (flat rounded controls, accent)."""
    t = THEMES.get(theme, THEMES["dark"])
    c = {k: t[k] for k in ("window", "base", "elevated", "text", "muted",
                           "border", "accent", "accent2", "on_accent")}
    return f"""
    * {{ font-size: 12px; }}
    QWidget {{ background: {c['window']}; color: {c['text']}; }}
    QToolTip {{ background: {c['elevated']}; color: {c['text']};
               border: 1px solid {c['border']}; border-radius: 4px; padding: 3px 6px; }}

    QGroupBox {{ background: {c['base']}; border: 1px solid {c['border']};
                 border-radius: 9px; margin-top: 13px; padding: 8px 8px 6px 8px; }}
    QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
                        left: 11px; padding: 1px 6px; color: {c['accent']};
                        font-weight: 700; }}

    QScrollArea, QAbstractScrollArea {{ border: none; background: transparent; }}
    QLabel {{ background: transparent; }}

    QPushButton {{ background: {c['elevated']}; color: {c['text']};
                   border: 1px solid {c['border']}; border-radius: 7px;
                   padding: 5px 13px; }}
    QPushButton:hover {{ border-color: {c['accent']}; color: {c['accent']}; }}
    QPushButton:pressed {{ background: {c['accent']}; color: {c['on_accent']}; }}
    QPushButton:checked {{ background: {c['accent']}; color: {c['on_accent']};
                           border-color: {c['accent']}; font-weight: 600; }}

    QComboBox, QSpinBox, QLineEdit {{ background: {c['elevated']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 7px; padding: 4px 9px; min-height: 18px; }}
    QComboBox:hover, QSpinBox:hover {{ border-color: {c['accent']}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{ background: {c['base']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 7px; outline: none;
        selection-background-color: {c['accent']}; selection-color: {c['on_accent']}; }}

    QCheckBox {{ spacing: 7px; background: transparent; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px;
        border: 1px solid {c['border']}; border-radius: 5px; background: {c['elevated']}; }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {c['accent']}; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background: {c['accent']}; border-color: {c['accent']}; }}

    QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0;
        background: {c['accent']}; border: none; border-radius: 7px; }}
    QSlider::handle:horizontal:hover {{ background: {c['accent2']}; }}

    QToolBar {{ background: {c['window']}; border: none; spacing: 5px; padding: 5px 6px; }}
    QToolBar::separator {{ background: {c['border']}; width: 1px; margin: 4px 6px; }}
    QMenuBar {{ background: {c['window']}; }}
    QMenuBar::item:selected {{ background: {c['elevated']}; }}
    QMenu {{ background: {c['base']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 4px; }}
    QMenu::item {{ padding: 5px 22px; border-radius: 5px; }}
    QMenu::item:selected {{ background: {c['accent']}; color: {c['on_accent']}; }}

    QTableWidget {{ background: {c['base']}; gridline-color: {c['border']};
        border: 1px solid {c['border']}; border-radius: 8px; }}
    QHeaderView::section {{ background: {c['elevated']}; color: {c['muted']};
        border: none; border-bottom: 1px solid {c['border']}; padding: 5px 8px; }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 5px; min-height: 24px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['muted']}; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {c['border']}; border-radius: 5px; min-width: 24px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    """


def apply_app_palette(app, theme: str) -> None:
    """Apply a dark/light :class:`QPalette` + the modern stylesheet to the whole app.

    Forces the Fusion style: native styles (notably macOS) ignore ``QPalette``/QSS for
    many widgets, so the controls would stay un-themed otherwise. Fusion honors both
    consistently across platforms, so the theme covers every panel + the chrome.
    """
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    t = THEMES.get(theme, THEMES["dark"])
    win, base, text = (QtGui.QColor(t["window"]), QtGui.QColor(t["base"]), QtGui.QColor(t["text"]))
    p = QtGui.QPalette()
    role = QtGui.QPalette
    for r in (role.Window, role.Button):
        p.setColor(r, win)
    p.setColor(role.Base, base)
    p.setColor(role.AlternateBase, QtGui.QColor(t["elevated"]))
    p.setColor(role.ToolTipBase, QtGui.QColor(t["elevated"]))
    for r in (role.WindowText, role.Text, role.ButtonText, role.ToolTipText):
        p.setColor(r, text)
    p.setColor(role.Highlight, QtGui.QColor(t["accent"]))
    p.setColor(role.HighlightedText, QtGui.QColor(t["on_accent"]))
    p.setColor(role.Link, QtGui.QColor(t["accent2"]))
    p.setColor(role.Disabled, role.Text, QtGui.QColor(t["muted"]))
    p.setColor(role.Disabled, role.WindowText, QtGui.QColor(t["muted"]))
    p.setColor(role.Disabled, role.ButtonText, QtGui.QColor(t["muted"]))
    app.setPalette(p)
    try:
        app.setStyleSheet(app_stylesheet(theme))
    except Exception:
        pass


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
