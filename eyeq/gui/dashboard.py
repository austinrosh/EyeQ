"""EyeQ live dashboard — a thin client over the headless engines.

Run:  python -m eyeq.gui.dashboard [--config path.yaml]

The statistical engine runs on the GUI thread (sub-ms; updates the cascade/SBR
plots on every LTI change). The transient engine runs on a worker thread and the
density eye is pulled at ~30 FPS, so dragging a slider never blocks the UI.
Parameter changes are routed by :class:`~eyeq.core.schema.Kind`:

* LTI        -> recompute statistical + mark the worker dirty (new SBR/eye).
* NONLINEAR  -> the worker picks it up at the next batch (no recompute).
* STRUCTURAL -> rebuild the worker (and, for a scenario change, the pipeline).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets
from pyqtgraph.dockarea import Dock, DockArea

from ..analysis.optimize import optimize_link
from ..core.schema import Kind
from ..engines import StatisticalEngine, ThreadWorker
from ..io import build_pipeline, default_link_config, load, save
from .binding import build_param_panel
from .plots import CascadePlot, EyePlot, HistPlot, SbrPlot

_BATCH = 15_000


class Controller:
    """Owns the pipeline + engines + worker and routes parameter changes."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.stat = StatisticalEngine()
        self.pipe = build_pipeline(cfg)
        self.worker = ThreadWorker(self.pipe, batch_symbols=_BATCH)
        self._running = False
        self.recompute_statistical()

    # -- engine state ---------------------------------------------------------
    def recompute_statistical(self):
        self.cascade, self.sbr, self.eye = self.stat.compute(self.pipe)

    def latest(self):
        return self.worker.latest()

    def start(self):
        self._running = True
        self.worker.start()

    def stop(self):
        self._running = False
        self.worker.stop()

    def _replace_worker(self):
        self.worker.stop()
        self.worker = ThreadWorker(self.pipe, batch_symbols=_BATCH)
        if self._running:
            self.worker.start()

    # -- routing --------------------------------------------------------------
    def on_param(self, block_name: str, param: str, value) -> str:
        block = self.pipe.by_name(block_name)
        p = block._param(param)

        # Channel reach is a scenario change (it re-derives the SimContext).
        if block_name == "channel" and param == "reach":
            return self.set_scenario(reach=value)

        block.set_params(**{param: value})
        if p.kind is Kind.STRUCTURAL:
            self.recompute_statistical()
            self._replace_worker()
            return "structural"
        if p.kind is Kind.LTI or p.also_statistical:
            self.recompute_statistical()
            self.worker.mark_dirty()
            return "lti"
        return "nonlinear"  # worker reads it next batch

    def set_scenario(self, *, modulation=None, rate=None, reach=None) -> str:
        if modulation:
            self.cfg.modulation = modulation
        if rate:
            self.cfg.data_rate_gbps = float(rate)
        if reach:
            self.cfg.reach_class = reach
        self.worker.stop()
        self.pipe = build_pipeline(self.cfg)  # fresh blocks (EQ resets)
        self.recompute_statistical()
        self.worker = ThreadWorker(self.pipe, batch_symbols=_BATCH)
        if self._running:
            self.worker.start()
        return "scenario"

    def auto_eq(self):
        self.worker.stop()
        optimize_link(self.pipe)
        self.recompute_statistical()
        self.worker = ThreadWorker(self.pipe, batch_symbols=_BATCH)
        if self._running:
            self.worker.start()

    def load_config(self, path):
        self.cfg = load(path)
        return self.set_scenario()

    def save_config(self, path):
        for bc in self.cfg.blocks:
            bc.params = self.pipe.by_name(_type_to_name(bc.type)).get_params()
        save(self.cfg, path)


def _type_to_name(type_name: str) -> str:
    from ..io.config import _NAME_TO_TYPE

    return {v: k for k, v in _NAME_TO_TYPE.items()}[type_name]


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.setWindowTitle("EyeQ — live SerDes link")
        self.ctrl = Controller(cfg)

        self.eye = EyePlot()
        self.cascade = CascadePlot()
        self.sbr = SbrPlot()
        self.hist = HistPlot()
        panel, self.panels = build_param_panel(self.ctrl.pipe, self._on_param)

        area = DockArea()
        self.setCentralWidget(area)
        d_eye = Dock("Eye", size=(560, 600))
        d_hist = Dock("Histogram", size=(150, 600))
        d_casc = Dock("Frequency cascade", size=(360, 300))
        d_sbr = Dock("Pulse response", size=(360, 300))
        d_ctl = Dock("Controls", size=(330, 600))
        area.addDock(d_eye, "left")
        area.addDock(d_hist, "right", d_eye)
        area.addDock(d_casc, "right", d_hist)
        area.addDock(d_sbr, "bottom", d_casc)
        area.addDock(d_ctl, "right", d_casc)
        d_eye.addWidget(self.eye)
        d_hist.addWidget(self.hist)
        d_casc.addWidget(self.cascade)
        d_sbr.addWidget(self.sbr)
        d_ctl.addWidget(panel)

        self._build_toolbar()
        self._update_static_plots()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)  # ~30 FPS

    # -- toolbar --------------------------------------------------------------
    def _build_toolbar(self):
        tb = self.addToolBar("main")
        self.run_btn = QtWidgets.QPushButton("Start")
        self.run_btn.setCheckable(True)
        self.run_btn.toggled.connect(self._toggle_run)
        tb.addWidget(self.run_btn)

        auto = QtWidgets.QPushButton("Auto-EQ")
        auto.clicked.connect(self._auto_eq)
        tb.addWidget(auto)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(" Mod "))
        self.mod_combo = QtWidgets.QComboBox()
        self.mod_combo.addItems(["NRZ", "PAM4"])
        self.mod_combo.setCurrentText(self.ctrl.cfg.modulation)
        self.mod_combo.currentTextChanged.connect(lambda m: self._scenario(modulation=m))
        tb.addWidget(self.mod_combo)

        tb.addWidget(QtWidgets.QLabel(" Rate(Gb/s) "))
        self.rate_combo = QtWidgets.QComboBox()
        self.rate_combo.addItems(["112", "224", "448"])
        self.rate_combo.setCurrentText(str(int(self.ctrl.cfg.data_rate_gbps)))
        self.rate_combo.currentTextChanged.connect(lambda r: self._scenario(rate=float(r)))
        tb.addWidget(self.rate_combo)
        tb.addSeparator()

        for label, slot in (("Load", self._load), ("Save", self._save)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            tb.addWidget(b)

    # -- updates --------------------------------------------------------------
    def _on_param(self, block, param, value):
        kind = self.ctrl.on_param(block, param, value)
        if kind in ("lti", "structural", "scenario"):
            self._update_static_plots()
        if kind in ("structural", "scenario"):
            self._resync_panels()

    def _update_static_plots(self):
        self.cascade.update_cascade(self.ctrl.cascade)
        self.sbr.update_sbr(self.ctrl.sbr)

    def _resync_panels(self):
        for name, panel in self.panels.items():
            panel.sync(self.ctrl.pipe.by_name(name))

    def _tick(self):
        snap = self.ctrl.latest()
        if snap is not None:
            self.eye.update_eye(snap)
            self.hist.update_hist(snap)

    # -- toolbar slots --------------------------------------------------------
    def _toggle_run(self, on):
        self.run_btn.setText("Stop" if on else "Start")
        self.ctrl.start() if on else self.ctrl.stop()

    def _auto_eq(self):
        self.ctrl.auto_eq()
        self._update_static_plots()
        self._resync_panels()

    def _scenario(self, **kw):
        self.ctrl.set_scenario(**kw)
        self._update_static_plots()
        # The block/param structure is identical across rates and modulations,
        # so the controls (keyed by name) stay valid — just re-sync their values.
        self._resync_panels()

    def _load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load config", "", "Config (*.yaml *.json)")
        if path:
            self.ctrl.load_config(path)
            self.mod_combo.blockSignals(True)
            self.rate_combo.blockSignals(True)
            self.mod_combo.setCurrentText(self.ctrl.cfg.modulation)
            self.rate_combo.setCurrentText(str(int(self.ctrl.cfg.data_rate_gbps)))
            self.mod_combo.blockSignals(False)
            self.rate_combo.blockSignals(False)
            self._update_static_plots()
            self._resync_panels()

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save config", "link.yaml", "Config (*.yaml *.json)")
        if path:
            self.ctrl.save_config(path)

    def closeEvent(self, event):
        self.ctrl.stop()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description="EyeQ live dashboard")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = load(args.config) if args.config else default_link_config(modulation="PAM4", reach_class="VSR")

    app = QtWidgets.QApplication(sys.argv)
    win = Dashboard(cfg)
    win.resize(1280, 760)
    win.show()
    win.run_btn.setChecked(True)  # auto-start the transient engine
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
