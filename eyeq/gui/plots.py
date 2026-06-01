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


def _lut(name: str = "viridis"):
    try:
        return pg.colormap.get(name).getLookupTable(nPts=256)
    except Exception:  # pragma: no cover - colormap name fallback
        return pg.colormap.get("CET-L9").getLookupTable(nPts=256)


def _db(h: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-6))


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
    def __init__(self):
        super().__init__()
        self.setLabel("left", "Amplitude", units="V")
        self.setLabel("bottom", "UI")
        self.showGrid(x=False, y=False)
        self.img = pg.ImageItem()
        self.img.setLookupTable(_lut())
        self.addItem(self.img)
        self.setXRange(-0.5, 1.5, padding=0)
        # marker at the (CDR-recovered or static) decision sampling phase
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

        self._last = None           # last DensitySnapshot (for CSV + annotation)
        self._last_levels = None    # (v0, v1) for the reset extents
        self._eye_width_ui = None   # set from the statistical BER (LTI cadence)
        self._target_ber = None
        self._detector_note = ""    # e.g. MLSD: eye opening is not the BER predictor
        self._install_interactive("eye")

    def set_detector_note(self, text: str) -> None:
        self._detector_note = text or ""
        self._refresh_annot()

    def update_eye(self, snap) -> None:
        self._last = snap
        img = snap.image  # [phase, voltage]
        two = np.vstack([img, img])  # tile two UI along the phase axis
        hi = float(two.max()) or 1.0
        self.img.setImage(two, autoLevels=False, levels=(0.0, hi))
        v0, v1 = snap.levels
        self._last_levels = (v0, v1)
        self.img.setRect(QtCore.QRectF(-0.5, v0, 2.0, v1 - v0))
        st = snap.stats
        self.sample_line.setPos(st.get("recovered_phase_ui", 0.0))
        self.setTitle(f"MSE SNR = {st.get('mse_snr_db', 0):.1f} dB    "
                      f"SER = {st.get('ser', 0):.1e}")
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

    def _default_ranges(self) -> None:
        self.setXRange(-0.5, 1.5, padding=0)
        if self._last_levels is not None:
            self.setYRange(self._last_levels[0], self._last_levels[1], padding=0.05)
        else:
            self.getViewBox().enableAutoRange(axis="y")

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
    def __init__(self):
        super().__init__()
        self.setLabel("left", "Voltage", units="V")
        self.setLabel("bottom", "Time", units="UI")
        self.curve = self.plot(pen=pg.mkPen("#d6453c", width=2))
        self.cursors = pg.ScatterPlotItem(size=6, brush=pg.mkBrush("#d6453c"))
        self.addItem(self.cursors)
        self._last = None
        self._install_interactive("sbr")

    def update_sbr(self, sbr) -> None:
        m = (sbr.t_ui > -3) & (sbr.t_ui < sbr.cursor_k.max() + 3)
        self.curve.setData(sbr.t_ui[m], sbr.sbr[m])
        self.cursors.setData(sbr.cursor_k.astype(float), sbr.cursors)
        self.setTitle("Pulse response (SBR)")
        self._last = (sbr.t_ui[m], sbr.sbr[m])

    def _csv_columns(self):
        if self._last is None:
            return None
        t, y = self._last
        return (["t_ui", "sbr_v"], np.column_stack([t, y]))


# --------------------------------------------------------------------------- #
# amplitude histogram (bathtub now lives in its own window)
# --------------------------------------------------------------------------- #
class HistPlot(InteractivePlot, pg.PlotWidget):
    """Amplitude histogram at the decision phase (vertical voltage axis)."""

    def __init__(self):
        super().__init__()
        self.setLabel("bottom", "Probability")
        self.setLabel("left", "Amplitude", units="V")
        self.curve = self.plot(pen=pg.mkPen("#3d7de8", width=1),
                               fillLevel=0, brush=pg.mkBrush(61, 125, 232, 90))
        self._last = None
        self._install_interactive("histogram")

    def update_hist(self, snap) -> None:
        ci = int(np.argmin(np.abs(snap.t_ui)))
        prob = snap.image[ci]
        self.curve.setData(prob, snap.v)
        self._last = (snap.v, prob)

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
