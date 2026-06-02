"""The dashboard's pyqtgraph plot widgets.

* :class:`EyePlot`      — the RX density eye (ImageItem, fixed LUT, two UI wide),
  annotated with eye height / width / sampling point at the CDR-recovered phase.
* :class:`CascadePlot`  — Channel / TX+Channel / TX+Channel+RX magnitude vs f/fnyq.
* :class:`SbrPlot`      — single-bit response with sampled cursors.
* :class:`HistPlot`     — amplitude histogram at the decision phase.
* :class:`BathtubPlot`  — a vertical or horizontal bathtub (SER vs level / phase),
  with the eye opening marked at the target BER (used by the bathtub window).

Every plot mixes in :class:`InteractivePlot`, which restores a consistent
right-click menu (the native pyqtgraph menu is disabled project-wide): a
one-action **Reset view** / auto-range and **Export PNG / CSV** with sensible
timestamped filenames. Mouse scroll-zoom and drag-pan are pyqtgraph defaults;
double-click also resets the view.

Each ``update_*`` takes engine result objects and caches the backing arrays so a
CSV export reflects exactly what is on screen; nothing here knows how the numbers
were computed.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401 — registers the exporters used below
from PySide6 import QtCore, QtWidgets

from . import theme

_HEATMAP_BG = "#000000"   # eye/histogram density panels stay dark regardless of app theme


def _db(h: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-6))


def _theme_line_plot(plot, name: str) -> None:
    """Apply a dark/light theme to a line plot (background + axis + grid)."""
    t = theme.THEMES.get(name, theme.THEMES["dark"])
    plot.setBackground(t["plot_bg"])
    pen = pg.mkPen(t["axis"])
    for ax in ("left", "bottom"):
        plot.getAxis(ax).setPen(pen)
        plot.getAxis(ax).setTextPen(pen)


def _dark_axes(plot) -> None:
    """Light axis pens for the always-dark heatmap panels (eye / histogram)."""
    pen = pg.mkPen("#b0b0b0")
    for ax in ("left", "bottom"):
        plot.getAxis(ax).setPen(pen)
        plot.getAxis(ax).setTextPen(pen)


# --------------------------------------------------------------------------- #
# shared interaction: reset/fit + PNG/CSV export (one consistent menu)
# --------------------------------------------------------------------------- #
class InteractivePlot:
    """Mixin for ``pg.PlotWidget`` subclasses: reset-view + PNG/CSV export menu.

    Subclasses call :meth:`_install_interactive` from ``__init__`` and may
    override :meth:`_default_ranges` (the fit-to-default extents) and
    :meth:`_csv_columns` (the data behind the plot, for CSV export).
    """

    def _install_interactive(self, basename: str) -> None:
        self._export_basename = basename
        self._io_dir = ""  # seeded from the last config Load/Save dir by the dashboard
        self.setMenuEnabled(False)  # suppress the native menu; we provide our own
        self.scene().sigMouseClicked.connect(self._on_scene_click)

    def set_io_dir(self, path: str) -> None:
        """Point default export paths at the user's last config directory."""
        self._io_dir = path or ""

    # -- reset / fit ----------------------------------------------------------
    def reset_view(self) -> None:
        """Return the plot to its natural, data-driven default extents."""
        self._default_ranges()

    def _default_ranges(self) -> None:
        self.getViewBox().autoRange()

    def _on_scene_click(self, ev) -> None:
        if ev.double() and ev.button() == QtCore.Qt.LeftButton:
            self.reset_view()

    # -- context menu ---------------------------------------------------------
    def contextMenuEvent(self, ev) -> None:  # noqa: N802 (Qt signature)
        menu = QtWidgets.QMenu(self)
        menu.addAction("Reset view (fit to default)", self.reset_view)
        menu.addAction("Auto-range to data", lambda: self.getViewBox().autoRange())
        menu.addSeparator()
        menu.addAction("Export PNG…", self._export_png_dialog)
        if self._csv_columns() is not None:
            menu.addAction("Export CSV…", self._export_csv_dialog)
        menu.exec(ev.globalPos())
        ev.accept()

    # -- export ---------------------------------------------------------------
    def _default_path(self, ext: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self._io_dir or os.getcwd()
        return os.path.join(base, f"{self._export_basename}_{ts}.{ext}")

    def _export_png_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PNG", self._default_path("png"), "PNG image (*.png)"
        )
        if path:
            self.export_png(path)

    def export_png(self, path: str) -> None:
        pg.exporters.ImageExporter(self.plotItem).export(path)

    def _export_csv_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", self._default_path("csv"), "CSV (*.csv)"
        )
        if path:
            self.export_csv(path)

    def export_csv(self, path: str) -> None:
        cols = self._csv_columns()
        if cols is None:
            return
        headers, data = cols
        np.savetxt(path, np.asarray(data, float), delimiter=",",
                   header=",".join(headers), comments="")

    def _csv_columns(self):
        """Return ``(headers, 2D-array)`` for CSV export, or ``None`` if not backed."""
        return None


# --------------------------------------------------------------------------- #
# eye
# --------------------------------------------------------------------------- #
class EyePlot(InteractivePlot, pg.PlotWidget):
    """RX density eye, centered on the sampling instant (0 UI), spanning ±1 UI.

    The eye/histogram density panels keep a dark background regardless of the app
    theme. The amplitude axis is fixed (±full-scale) by default so the eye visibly
    breathes with swing/loss; ``set_amp_scale`` toggles auto-fit. The colormap and
    linear/log density scale are configurable.
    """

    def __init__(self):
        super().__init__()
        self.setBackground(_HEATMAP_BG)
        self.setLabel("left", "Amplitude", units="V")
        self.setLabel("bottom", "UI")
        self.showGrid(x=False, y=False)
        _dark_axes(self)
        self.img = pg.ImageItem()
        self._colormap = "turbo"
        self._density_scale = "log"
        self.img.setLookupTable(theme.eye_lut(self._colormap))
        self.addItem(self.img)
        self.setXRange(-1.0, 1.0, padding=0)
        # marker at the (now centered) decision sampling phase
        self.sample_line = pg.InfiniteLine(pos=0.0, angle=90,
                                           pen=pg.mkPen("#ff5050", style=QtCore.Qt.DashLine))
        self.addItem(self.sample_line)

        # Eye height / width / sampling-point annotation, pinned to the top-left
        # corner of the view (repositioned on every range change so zoom/pan
        # never hides it).
        self.annot = pg.TextItem(anchor=(0, 0), color=(255, 255, 255),
                                 fill=pg.mkBrush(0, 0, 0, 130))
        self.annot.setZValue(100)
        self.addItem(self.annot)
        self.getViewBox().sigRangeChanged.connect(self._reposition_annot)

        # Hover crosshair + live UI/mV readout — drag the mouse over the eye to
        # probe any point (the interactivity the static density panel was missing).
        cross_pen = pg.mkPen("#7fd4cf", width=1, style=QtCore.Qt.DotLine)
        self.cross_v = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self.cross_h = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self.cursor_label = pg.TextItem(anchor=(0, 1), color=(220, 240, 240),
                                        fill=pg.mkBrush(0, 0, 0, 150))
        self.cursor_label.setZValue(101)
        for it in (self.cross_v, self.cross_h, self.cursor_label):
            it.setVisible(False)
            self.addItem(it)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._last = None           # last DensitySnapshot (for CSV + annotation)
        self._last_levels = None    # (v0, v1) data extent for auto-fit
        self._amp_mode = "fixed"    # fixed (±full_scale) | auto
        self._full_scale = None     # ±full_scale [V] (set from swing by the controller)
        self._eye_width_ui = None   # set from the statistical BER (LTI cadence)
        self._target_ber = None
        self._detector_note = ""    # e.g. MLSD: eye opening is not the BER predictor
        self._install_interactive("eye")

    def set_detector_note(self, text: str) -> None:
        self._detector_note = text or ""
        self._refresh_annot()

    # -- view settings --------------------------------------------------------
    def set_colormap(self, name: str) -> None:
        self._colormap = name
        self.img.setLookupTable(theme.eye_lut(name))

    def set_density_scale(self, scale: str) -> None:
        self._density_scale = scale
        if self._last is not None:
            self.update_eye(self._last)

    def set_amp_scale(self, mode: str, full_scale_v: float | None) -> None:
        self._amp_mode = mode
        if full_scale_v is not None:
            self._full_scale = float(full_scale_v)
        self._apply_amp()

    def _apply_amp(self) -> None:
        if self._amp_mode == "fixed" and self._full_scale:
            # Frame to ±full-scale (swing-tied, so the eye breathes with loss) but
            # never below the actual data extent — RX-FFE overshoot can push the
            # received eye past the launch swing, and it must not clip top/bottom.
            fs = self._full_scale
            if self._last_levels is not None:
                fs = max(fs, abs(self._last_levels[0]), abs(self._last_levels[1]))
            self.setYRange(-fs, fs, padding=0)
        elif self._last_levels is not None:
            self.setYRange(self._last_levels[0], self._last_levels[1], padding=0.05)
        else:
            self.getViewBox().enableAutoRange(axis="y")

    def update_eye(self, snap) -> None:
        self._last = snap
        img = snap.image  # [phase, voltage]
        sps = img.shape[0]
        # Roll so the CDR-recovered sampling instant lands on the tile boundary,
        # then the two tiled UIs frame the central eye opening at 0 (crossings ±0.5,
        # adjacent openings ±1). Pure plotting — the underlying density is untouched.
        j = int(round(snap.stats.get("recovered_phase_ui", 0.0) * sps + sps // 2)) % sps
        two = np.vstack([np.roll(img, -j, axis=0)] * 2)
        if self._density_scale == "log":
            floor = max(float(two.max()) * 1e-6, 1e-12)
            disp = np.log10(np.maximum(two, floor))
            levels = (float(np.log10(floor)), float(disp.max()) or 0.0)
        else:
            disp = two
            levels = (0.0, float(two.max()) or 1.0)
        self.img.setImage(disp, autoLevels=False, levels=levels)
        v0, v1 = snap.levels
        self._last_levels = (v0, v1)
        self.img.setRect(QtCore.QRectF(-1.0, v0, 2.0, v1 - v0))
        st = snap.stats
        self.sample_line.setPos(0.0)
        self.setTitle(f"MSE SNR = {st.get('mse_snr_db', 0):.1f} dB    "
                      f"SER = {st.get('ser', 0):.1e}")
        self._apply_amp()
        self._refresh_annot()

    def set_metrics(self, eye_width_ui: float, target_ber: float) -> None:
        """Statistical eye width + target BER (changes only on an LTI update)."""
        self._eye_width_ui = eye_width_ui
        self._target_ber = target_ber
        self._refresh_annot()

    def _refresh_annot(self) -> None:
        if self._last is None:
            return
        st = self._last.stats
        ph = st.get("recovered_phase_ui", 0.0)
        lines = []
        h = st.get("eye_height_at_phase_v")
        if h is not None:
            lines.append(f"Eye height: {h * 1e3:.1f} mV  (at sample phase {ph:+.2f} UI)")
        if self._eye_width_ui is not None:
            at = f" @ BER {self._target_ber:.0e}" if self._target_ber else ""
            lines.append(f"Eye width:  {self._eye_width_ui:.3f} UI{at}")
        lines.append(f"Sample @ {ph:+.2f} UI (CDR-recovered)")
        if self._detector_note:
            lines.append(self._detector_note)
        self.annot.setText("\n".join(lines))

    def _reposition_annot(self) -> None:
        (x0, _x1), (_y0, y1) = self.getViewBox().viewRange()
        self.annot.setPos(x0, y1)  # top-left corner of the current view

    def _on_mouse_moved(self, pos) -> None:
        """Track the mouse: position the crosshair + readout, hide it off-plot."""
        vb = self.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            for it in (self.cross_v, self.cross_h, self.cursor_label):
                it.setVisible(False)
            return
        pt = vb.mapSceneToView(pos)
        x, y = float(pt.x()), float(pt.y())
        self.cross_v.setPos(x)
        self.cross_h.setPos(y)
        self.cursor_label.setText(f"{x:+.3f} UI\n{y * 1e3:+.0f} mV")
        self.cursor_label.setPos(x, y)
        for it in (self.cross_v, self.cross_h, self.cursor_label):
            it.setVisible(True)

    def _default_ranges(self) -> None:
        self.setXRange(-1.0, 1.0, padding=0)
        self._apply_amp()

    def _csv_columns(self):
        if self._last is None:
            return None
        img, t, v = self._last.image, self._last.t_ui, self._last.v
        P, V = img.shape
        data = np.column_stack([np.repeat(t, V), np.tile(v, P), img.ravel()])
        return (["phase_ui", "voltage_v", "density"], data)


# --------------------------------------------------------------------------- #
# frequency cascade
# --------------------------------------------------------------------------- #
class CascadePlot(InteractivePlot, pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setLabel("left", "Magnitude", units="dB")
        self.setLabel("bottom", "f / f_nyq")
        self.setLogMode(x=True, y=False)
        self.addLegend(offset=(-10, 10))
        self.setYRange(-50, 5)
        self.c_ch = self.plot(pen=pg.mkPen("#888", width=1), name="Channel")
        self.c_tx = self.plot(pen=pg.mkPen("#e8a33d", width=1), name="TX+Channel")
        self.c_rx = self.plot(pen=pg.mkPen("#2ca089", width=2), name="TX+Channel+RX")
        self.nyq = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#aaa", style=QtCore.Qt.DashLine))
        self.addItem(self.nyq)
        self._last = None
        self._install_interactive("cascade")

    def apply_theme(self, name: str) -> None:
        _theme_line_plot(self, name)

    def update_cascade(self, casc) -> None:
        x = casc.f_over_fnyq
        m = x > 0
        ch, tx, rx = (_db(casc.H_channel[m]), _db(casc.H_tx_chan[m]), _db(casc.H_tx_chan_rx[m]))
        self.c_ch.setData(x[m], ch)
        self.c_tx.setData(x[m], tx)
        self.c_rx.setData(x[m], rx)
        self.nyq.setPos(np.log10(1.0))
        self.setTitle(f"Nyquist loss: {casc.nyquist_loss_db:.2f} dB")
        self._last = (x[m], ch, tx, rx)

    def _default_ranges(self) -> None:
        self.getViewBox().autoRange()
        self.setYRange(-50, 5)

    def _csv_columns(self):
        if self._last is None:
            return None
        x, ch, tx, rx = self._last
        return (["f_over_fnyq", "Channel_dB", "TX+Channel_dB", "TX+Channel+RX_dB"],
                np.column_stack([x, ch, tx, rx]))


# --------------------------------------------------------------------------- #
# single-bit / pulse response
# --------------------------------------------------------------------------- #
class SbrPlot(InteractivePlot, pg.PlotWidget):
    """Single-bit (pulse) response with the cursors drawn as emphasized stems:
    red vertical stems from baseline to each sampled cursor, a distinct (larger,
    bright) main-cursor marker, and h-1/h0/h+1 labels on the central cursors."""

    _STEM = "#d6453c"
    _MAIN = "#ffd24d"

    def __init__(self):
        super().__init__()
        self.setLabel("left", "Voltage", units="V")
        self.setLabel("bottom", "Time", units="UI")
        self.curve = self.plot(pen=pg.mkPen(self._STEM, width=2))   # continuous pulse
        self.stems = pg.PlotCurveItem(pen=pg.mkPen(self._STEM, width=2),
                                      connect="pairs")             # vertical cursor stems
        self.markers = pg.ScatterPlotItem()
        self.baseline = pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen("#666", width=1))
        self.addItem(self.baseline)
        self.addItem(self.stems)
        self.addItem(self.markers)
        self._labels: list[pg.TextItem] = []
        self._fg = "#e0e0e0"
        self._last = None
        self._xr = None             # explicit data-driven ranges (see update_sbr)
        self._yr = None
        self._install_interactive("sbr")

    def apply_theme(self, name: str) -> None:
        _theme_line_plot(self, name)
        self._fg = theme.THEMES.get(name, theme.THEMES["dark"])["text"]
        for lbl in self._labels:
            lbl.setColor(self._fg)

    def update_sbr(self, sbr) -> None:
        m = (sbr.t_ui > -3) & (sbr.t_ui < sbr.cursor_k.max() + 3)
        self.curve.setData(sbr.t_ui[m], sbr.sbr[m])

        ks = sbr.cursor_k.astype(float)
        ys = np.asarray(sbr.cursors, float)
        # NaN-separated point pairs -> disconnected vertical stems (baseline -> sample)
        sx = np.empty(ks.size * 3); sy = np.empty(ks.size * 3)
        sx[0::3] = ks; sx[1::3] = ks; sx[2::3] = np.nan
        sy[0::3] = 0.0; sy[1::3] = ys; sy[2::3] = np.nan
        self.stems.setData(sx, sy)

        spots = [{"pos": (k, y),
                  "size": 13 if k == 0 else 7,
                  "brush": pg.mkBrush(self._MAIN if k == 0 else self._STEM),
                  "pen": pg.mkPen("#000000")}
                 for k, y in zip(ks, ys)]
        self.markers.setData(spots)

        # labels on the central cursors (h-1, h0, h+1)
        for lbl in self._labels:
            self.removeItem(lbl)
        self._labels = []
        for k, y in zip(ks, ys):
            if abs(k) <= 1:
                name = "h0" if k == 0 else f"h{int(k):+d}"
                lbl = pg.TextItem(name, color=self._fg,
                                  anchor=(0.5, 1.2 if y >= 0 else -0.2))
                lbl.setPos(k, y)
                self.addItem(lbl)
                self._labels.append(lbl)

        self.setTitle("Pulse response (SBR)")

        # Explicit, data-driven ranges. The SBR must never rely on pyqtgraph's
        # autorange alone: the NaN-separated stems can make an item report
        # degenerate/NaN bounds, which sticks the view at a ~1e-306 scale (the data
        # then renders off-screen). Framing to the pulse extents every update is robust.
        t = sbr.t_ui[m]
        ys = np.concatenate([sbr.sbr[m], np.asarray(sbr.cursors, float)])
        ymax = float(max(np.max(ys), 0.0))
        ymin = float(min(np.min(ys), 0.0))
        span = (ymax - ymin) or abs(ymax) or 1e-3
        self._yr = (ymin - 0.12 * span, ymax + 0.12 * span)
        self._xr = (float(t[0]), float(t[-1])) if t.size else None
        self._apply_ranges()
        self._last = (sbr.t_ui[m], sbr.sbr[m])

    def _apply_ranges(self) -> None:
        if self._yr is not None:
            self.setYRange(*self._yr, padding=0)
        if self._xr is not None:
            self.setXRange(*self._xr, padding=0.02)

    def _default_ranges(self) -> None:
        self._apply_ranges()

    def _csv_columns(self):
        if self._last is None:
            return None
        t, y = self._last
        return (["t_ui", "sbr_v"], np.column_stack([t, y]))


# --------------------------------------------------------------------------- #
# amplitude histogram (bathtub now lives in its own window)
# --------------------------------------------------------------------------- #
class HistPlot(InteractivePlot, pg.PlotWidget):
    """Amplitude histogram at the sampling instant (shares the eye's amplitude axis).

    Part of the dark "heatmap group": dark background, warm-themed curve. Slices the
    density at the CDR-recovered sampling column so it matches the centered eye.
    """

    def __init__(self):
        super().__init__()
        self.setBackground(_HEATMAP_BG)
        self.setLabel("bottom", "Probability")
        self.setLabel("left", "Amplitude", units="V")
        _dark_axes(self)
        self.curve = self.plot(pen=pg.mkPen("#e8a33d", width=1),
                               fillLevel=0, brush=pg.mkBrush(232, 163, 61, 90))
        self._last = None
        self._amp_mode = "fixed"
        self._full_scale = None
        self._v_range = None
        self._install_interactive("histogram")

    def set_amp_scale(self, mode: str, full_scale_v: float | None) -> None:
        self._amp_mode = mode
        if full_scale_v is not None:
            self._full_scale = float(full_scale_v)
        self._apply_amp()

    def _apply_amp(self) -> None:
        if self._amp_mode == "fixed" and self._full_scale:
            fs = self._full_scale  # contain the data even when EQ overshoot exceeds full-scale
            if self._v_range is not None:
                fs = max(fs, abs(self._v_range[0]), abs(self._v_range[1]))
            self.setYRange(-fs, fs, padding=0)
        elif self._v_range is not None:
            self.setYRange(self._v_range[0], self._v_range[1], padding=0.05)
        else:
            self.getViewBox().enableAutoRange(axis="y")

    def update_hist(self, snap) -> None:
        sps = snap.image.shape[0]
        ci = int(round(snap.stats.get("recovered_phase_ui", 0.0) * sps + sps // 2)) % sps
        prob = snap.image[ci]
        self.curve.setData(prob, snap.v)
        self._last = (snap.v, prob)
        self._v_range = (float(snap.v[0]), float(snap.v[-1]))
        self._apply_amp()

    def _default_ranges(self) -> None:
        self.getViewBox().enableAutoRange(axis="x")
        self._apply_amp()

    def _csv_columns(self):
        if self._last is None:
            return None
        v, prob = self._last
        return (["voltage_v", "probability"], np.column_stack([v, prob]))


# --------------------------------------------------------------------------- #
# bathtub (vertical: SER vs level; horizontal: SER vs phase)
# --------------------------------------------------------------------------- #
class BathtubPlot(InteractivePlot, pg.PlotWidget):
    """A bathtub curve with the eye opening marked at the target BER, plus an
    optional post-FEC overlay.

    ``orient='vertical'`` plots log10(error rate) (x) vs decision level (y);
    ``orient='horizontal'`` plots sampling phase (x) vs log10(error rate) (y).
    When a :class:`~eyeq.analysis.fec.FecResult` is passed and enabled, both
    curves switch to BER and a green post-FEC curve is overlaid, with the
    pre-FEC threshold and post-FEC target marked distinctly from the raw target.
    """

    def __init__(self, orient: str = "vertical"):
        super().__init__()
        self.orient = orient
        angle = 90 if orient == "vertical" else 0
        if orient == "vertical":
            self.setLabel("bottom", "log10 SER")
            self.setLabel("left", "Amplitude", units="V")
            self.setTitle("Vertical bathtub (voltage margin)")
            self._region = pg.LinearRegionItem(orientation="horizontal", movable=False,
                                               brush=pg.mkBrush(45, 160, 137, 60))
        else:
            self.setLabel("bottom", "Sampling phase", units="UI")
            self.setLabel("left", "log10 SER")
            self.setTitle("Horizontal bathtub (timing margin)")
            self._region = pg.LinearRegionItem(orientation="vertical", movable=False,
                                               brush=pg.mkBrush(45, 160, 137, 60))
        self._target = pg.InfiniteLine(angle=angle, pen=pg.mkPen("#888", style=QtCore.Qt.DashLine))
        self.curve = self.plot(pen=pg.mkPen("#d6453c", width=2))            # pre-FEC
        self.curve_post = self.plot(pen=pg.mkPen("#2ca02c", width=2))       # post-FEC
        self._pre_thresh = pg.InfiniteLine(angle=angle,
                                           pen=pg.mkPen("#e8a33d", width=1.5, style=QtCore.Qt.DashLine))
        self._post_target = pg.InfiniteLine(angle=angle,
                                            pen=pg.mkPen("#2ca02c", width=1.5, style=QtCore.Qt.DashLine))
        self.addItem(self._region)
        for it in (self._target, self._pre_thresh, self._post_target):
            self.addItem(it)
        self.curve_post.hide()
        self._pre_thresh.hide()
        self._post_target.hide()
        self._last = None
        self._install_interactive(f"bathtub_{orient}")

    def apply_theme(self, name: str) -> None:
        _theme_line_plot(self, name)

    def update_bathtub(self, ber, fec=None, detector_label=None) -> None:
        on = fec is not None and getattr(fec, "enabled", False)
        mlsd = getattr(ber, "detector", "decision") == "mlsd"
        raw_thr = float(np.log10(max(ber.target_ber, 1e-30)))
        # raw SER arrays drive the (voltage/phase) eye-opening region regardless of FEC
        raw_axis, raw_ser = ((ber.v_eye, ber.v_bathtub) if self.orient == "vertical"
                             else (ber.t_axis, ber.h_bathtub))
        if self.orient == "vertical":
            axis = ber.v_eye
            pre = fec.pre_v_ber if on else ber.v_bathtub
            self.curve.setData(pre, axis)
            if on:
                self.curve_post.setData(fec.post_v_ber, axis)
        else:
            axis = ber.t_axis
            pre = fec.pre_h_ber if on else ber.h_bathtub
            self.curve.setData(axis, pre)
            if on:
                self.curve_post.setData(axis, fec.post_h_ber)

        self._target.setPos(raw_thr)
        post = None
        if on:
            post = fec.post_v_ber if self.orient == "vertical" else fec.post_h_ber
            self._pre_thresh.setPos(float(np.log10(max(fec.pre_threshold_ber, 1e-300))))
            self._post_target.setPos(float(np.log10(max(fec.target_post_ber, 1e-300))))
            self.curve_post.show()
            self._pre_thresh.show()
            self._post_target.show()
            tag = (f"{fec.scheme_label} — pre-FEC thr {fec.pre_threshold_ber:.1e}, "
                   f"post target {fec.target_post_ber:.0e} (model-based)")
            unit = "log10 BER"
        else:
            self.curve_post.hide()
            self._pre_thresh.hide()
            self._post_target.hide()
            tag = f"@ BER {ber.target_ber:.0e}"
            unit = "log10 SER"

        if mlsd:  # the MLSD bathtub is sequence-error-derived, not an eye tail
            tag = f"{detector_label or 'MLSD'} (sequence-error model) — {tag}"

        if self.orient == "vertical":
            self.setLabel("bottom", unit)
            self.setTitle(f"Vertical bathtub — opening {ber.eye_height_v * 1e3:.1f} mV  {tag}")
        else:
            self.setLabel("left", unit)
            self.setTitle(f"Horizontal bathtub — opening {ber.eye_width_ui:.3f} UI  {tag}")

        span = _open_span(raw_axis, raw_ser, raw_thr)
        if span is not None:
            self._region.setRegion(span)
            self._region.show()
        else:
            self._region.hide()
        self._last = (axis, pre, post)

    def _csv_columns(self):
        if self._last is None:
            return None
        axis, pre, post = self._last
        base = "voltage_v" if self.orient == "vertical" else "phase_ui"
        if post is not None:
            return ([base, "log10_pre_fec_ber", "log10_post_fec_ber"],
                    np.column_stack([axis, pre, post]))
        return ([base, "log10_ser"], np.column_stack([axis, pre]))


def _open_span(axis, log_ser, log_thr):
    """Contiguous (lo, hi) of ``axis`` around the SER minimum where SER < target."""
    below = np.asarray(log_ser) < log_thr
    if not below.any():
        return None
    j = int(np.argmin(log_ser))
    if not below[j]:
        return None
    lo = j
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = j
    while hi < len(axis) - 1 and below[hi + 1]:
        hi += 1
    return float(axis[lo]), float(axis[hi])
