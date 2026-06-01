"""The dashboard's pyqtgraph plot widgets.

* :class:`EyePlot`      — the RX density eye (ImageItem, fixed LUT, two UI wide).
* :class:`CascadePlot`  — Channel / TX+Channel / TX+Channel+RX magnitude vs f/fnyq.
* :class:`SbrPlot`      — single-bit response with sampled cursors.
* :class:`HistPlot`     — amplitude histogram at the decision phase.

Each ``update_*`` takes engine result objects; nothing here knows how they were
computed.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore


def _lut(name: str = "viridis"):
    try:
        return pg.colormap.get(name).getLookupTable(nPts=256)
    except Exception:  # pragma: no cover - colormap name fallback
        return pg.colormap.get("CET-L9").getLookupTable(nPts=256)


class EyePlot(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setLabel("left", "Amplitude", units="V")
        self.setLabel("bottom", "UI")
        self.setMenuEnabled(False)
        self.showGrid(x=False, y=False)
        self.img = pg.ImageItem()
        self.img.setLookupTable(_lut())
        self.addItem(self.img)
        self.setXRange(-0.5, 1.5, padding=0)
        # marker at the (CDR-recovered or static) decision sampling phase
        self.sample_line = pg.InfiniteLine(pos=0.0, angle=90,
                                           pen=pg.mkPen("#ff5050", style=QtCore.Qt.DashLine))
        self.addItem(self.sample_line)

    def update_eye(self, snap) -> None:
        img = snap.image  # [phase, voltage]
        two = np.vstack([img, img])  # tile two UI along the phase axis
        hi = float(two.max()) or 1.0
        self.img.setImage(two, autoLevels=False, levels=(0.0, hi))
        v0, v1 = snap.levels
        self.img.setRect(QtCore.QRectF(-0.5, v0, 2.0, v1 - v0))
        st = snap.stats
        self.sample_line.setPos(st.get("recovered_phase_ui", 0.0))
        self.setTitle(f"MSE SNR = {st.get('mse_snr_db', 0):.1f} dB    "
                      f"SER = {st.get('ser', 0):.1e}    "
                      f"phase = {st.get('recovered_phase_ui', 0.0):+.2f} UI")


class CascadePlot(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setLabel("left", "Magnitude", units="dB")
        self.setLabel("bottom", "f / f_nyq")
        self.setLogMode(x=True, y=False)
        self.setMenuEnabled(False)
        self.addLegend(offset=(-10, 10))
        self.setYRange(-50, 5)
        self.c_ch = self.plot(pen=pg.mkPen("#888", width=1), name="Channel")
        self.c_tx = self.plot(pen=pg.mkPen("#e8a33d", width=1), name="TX+Channel")
        self.c_rx = self.plot(pen=pg.mkPen("#2ca089", width=2), name="TX+Channel+RX")
        self.nyq = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#aaa", style=QtCore.Qt.DashLine))
        self.addItem(self.nyq)

    def update_cascade(self, casc) -> None:
        x = casc.f_over_fnyq
        m = x > 0
        self.c_ch.setData(x[m], _db(casc.H_channel[m]))
        self.c_tx.setData(x[m], _db(casc.H_tx_chan[m]))
        self.c_rx.setData(x[m], _db(casc.H_tx_chan_rx[m]))
        self.nyq.setPos(np.log10(1.0))
        self.setTitle(f"Nyquist loss: {casc.nyquist_loss_db:.2f} dB")


class SbrPlot(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setLabel("left", "Voltage", units="V")
        self.setLabel("bottom", "Time", units="UI")
        self.setMenuEnabled(False)
        self.curve = self.plot(pen=pg.mkPen("#d6453c", width=2))
        self.cursors = pg.ScatterPlotItem(size=6, brush=pg.mkBrush("#d6453c"))
        self.addItem(self.cursors)

    def update_sbr(self, sbr) -> None:
        m = (sbr.t_ui > -3) & (sbr.t_ui < sbr.cursor_k.max() + 3)
        self.curve.setData(sbr.t_ui[m], sbr.sbr[m])
        self.cursors.setData(sbr.cursor_k.astype(float), sbr.cursors)
        self.setTitle("Pulse response (SBR)")


class HistPlot(pg.PlotWidget):
    """Amplitude histogram (bottom axis) + vertical bathtub (top axis), shared V."""

    def __init__(self):
        super().__init__()
        self.setLabel("bottom", "Probability")
        self.setLabel("left", "Amplitude", units="V")
        self.setMenuEnabled(False)
        self.curve = self.plot(pen=pg.mkPen("#3d7de8", width=1),
                               fillLevel=0, brush=pg.mkBrush(61, 125, 232, 90))

        # Secondary view sharing the voltage (y) axis, for the bathtub vs log10 SER.
        self._p2 = pg.ViewBox()
        self.plotItem.showAxis("top")
        self.plotItem.scene().addItem(self._p2)
        self.plotItem.getAxis("top").linkToView(self._p2)
        self.plotItem.getAxis("top").setLabel("log10 SER")
        self._p2.setYLink(self.plotItem)
        self._p2.setXRange(-15, 0, padding=0)
        self.bathtub = pg.PlotCurveItem(pen=pg.mkPen("#d6453c", width=2))
        self._p2.addItem(self.bathtub)
        self.plotItem.vb.sigResized.connect(self._sync_views)
        self._sync_views()

    def _sync_views(self) -> None:
        self._p2.setGeometry(self.plotItem.vb.sceneBoundingRect())
        self._p2.linkedViewChanged(self.plotItem.vb, self._p2.YAxis)

    def update_hist(self, snap) -> None:
        ci = int(np.argmin(np.abs(snap.t_ui)))
        self.curve.setData(snap.image[ci], snap.v)

    def update_bathtub(self, ber) -> None:
        self.bathtub.setData(ber.v_bathtub, ber.v_eye)  # log10 SER (x) vs voltage (y)
        self.setTitle(f"BER {ber.ber:.1e}   COM {ber.com_db:+.1f} dB")


def _db(h: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-6))
