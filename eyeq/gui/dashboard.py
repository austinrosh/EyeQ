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
import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets
from pyqtgraph.dockarea import Dock, DockArea

from ..analysis import ber as _ber
from ..analysis import fec as _fec
from ..analysis.optimize import optimize_link
from ..core.schema import Kind
from ..engines import StatisticalEngine, ThreadWorker
from ..io import build_pipeline, default_link_config, load, save
from ..io.config import default_detector, default_fec
from .binding import build_param_panel
from .panels import BathtubWindow, DetectorWindow, FecWindow, ReportWindow
from .plots import CascadePlot, EyePlot, HistPlot, SbrPlot

# Equalizer-stage bypass toggles surfaced in the toolbar (label, block, param).
_EQ_TOGGLES = [
    ("CTLE", "ctle", "enabled"),
    ("TX-FFE", "txffe", "ffe_enabled"),
    ("TX-drv", "txffe", "driver_enabled"),
    ("RX-FFE", "rxffe", "enabled"),
    ("DFE", "dfe", "enabled"),
]

_BATCH = 15_000


class Controller:
    """Owns the pipeline + engines + worker and routes parameter changes."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.stat = StatisticalEngine()
        self.pipe = build_pipeline(cfg)
        self.worker = ThreadWorker(self.pipe, batch_symbols=_BATCH)
        self._running = False
        self.fec_cfg = {**default_fec(), **(cfg.fec or {})}
        self.fec_result = None
        self.detector_cfg = {**default_detector(), **(cfg.detector or {})}
        self._apply_detector_arch()
        self.recompute_statistical()

    # -- engine state ---------------------------------------------------------
    def recompute_statistical(self):
        # Assess at the reach class's spec BER (per-preset) so the bathtub markers,
        # eye opening, and COM are evaluated at the BER the link is designed to.
        self.target_ber = self.pipe.ctx.reach.target_ber
        self.cascade, self.sbr, self.eye = self.stat.compute(self.pipe)
        self.ber = self._assess_ber()
        self._refresh_fec()

    def _assess_ber(self):
        """Branch the BER computation by detector: MLSD uses the sequence-error
        (minimum-distance) estimate; slicer/DFE use the eye-tail method."""
        if self.detector_cfg.get("mode") == "mlsd":
            return _ber.assess_mlsd(self.stat, self.pipe, self.sbr, target_ber=self.target_ber,
                                    mlsd_taps=int(self.detector_cfg.get("mlsd_taps", 4)))
        return _ber.assess(self.stat, self.pipe, self.sbr,
                           target_ber=self.target_ber, phase_points=33, v_bins=512)

    def _apply_detector_arch(self):
        """Selector owns architecture: 'dfe' -> DFE on; 'slicer'/'mlsd' -> DFE off."""
        mode = self.detector_cfg.get("mode", "dfe")
        try:
            self.pipe.by_name("dfe").set_params(enabled=("on" if mode == "dfe" else "off"))
        except KeyError:
            pass

    def on_detector_change(self, cfg: dict):
        """Apply a new detector config: set the DFE architecture + re-assess the BER."""
        self.detector_cfg = dict(cfg)
        self._apply_detector_arch()
        self.recompute_statistical()  # re-assess via the selected detector (+ FEC refresh)

    def _refresh_fec(self):
        """Recompute the post-FEC estimate from the current pre-FEC BER + FEC config.

        Pure analysis (no engine/worker) — orthogonal to the EQ toggles and Auto-EQ.
        """
        self.fec_result = _fec.assess_fec(self.ber, self.pipe.ctx, self.fec_cfg)

    def on_fec_change(self, cfg: dict):
        """Apply a new FEC config and recompute the post-FEC result (live, cheap)."""
        self.fec_cfg = dict(cfg)
        self._refresh_fec()

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
        self._apply_detector_arch()           # re-impose the detector's DFE state on the fresh pipe
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
        self.fec_cfg = {**default_fec(), **(self.cfg.fec or {})}
        self.detector_cfg = {**default_detector(), **(self.cfg.detector or {})}
        return self.set_scenario()  # _apply_detector_arch + recompute use the new cfgs

    def save_config(self, path):
        for bc in self.cfg.blocks:
            bc.params = self.pipe.by_name(_type_to_name(bc.type)).get_params()
        self.cfg.fec = dict(self.fec_cfg)
        self.cfg.detector = dict(self.detector_cfg)
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

        # Lazily-created side windows (bathtub curves, link report, FEC) + export dir.
        self.bathtub_win: BathtubWindow | None = None
        self.report_win: ReportWindow | None = None
        self.fec_win: FecWindow | None = None
        self.detector_win: DetectorWindow | None = None
        self._io_dir = ""

        self._build_toolbar()
        self._update_static_plots()
        self._resync_panels()  # reflect values build_pipeline set (e.g. channel.reach)

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

        # Per-stage EQ bypass toggles — isolate any stage's contribution. Each is
        # a true bypass (signal unmodified), independent of the auto-EQ solver.
        tb.addWidget(QtWidgets.QLabel(" EQ "))
        self.eq_checks: dict[tuple[str, str], QtWidgets.QCheckBox] = {}
        for label, block, param in _EQ_TOGGLES:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(self._eq_is_on(block, param))
            cb.toggled.connect(
                lambda on, b=block, p=param: self._on_param(b, p, "on" if on else "off")
            )
            tb.addWidget(cb)
            self.eq_checks[(block, param)] = cb
        tb.addSeparator()

        # Receiver detector: peer to the EQ toggles; owns the slicer/DFE/MLSD architecture.
        tb.addWidget(QtWidgets.QLabel(" Detector "))
        self.detector_combo = QtWidgets.QComboBox()
        for label, key in (("Slicer", "slicer"), ("DFE", "dfe"), ("MLSD", "mlsd")):
            self.detector_combo.addItem(label, key)
        i = self.detector_combo.findData(self.ctrl.detector_cfg.get("mode", "dfe"))
        self.detector_combo.setCurrentIndex(max(0, i))
        self.detector_combo.currentIndexChanged.connect(self._on_detector_mode)
        tb.addWidget(self.detector_combo)
        det_btn = QtWidgets.QPushButton("Detector…")
        det_btn.clicked.connect(self._toggle_detector_window)
        tb.addWidget(det_btn)
        tb.addSeparator()

        for label, slot in (("Bathtub", self._toggle_bathtub), ("Report", self._toggle_report)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            tb.addWidget(b)
        tb.addSeparator()

        # FEC: a master on/off (live) + a settings window for scheme/params.
        self.fec_check = QtWidgets.QCheckBox("FEC")
        self.fec_check.setChecked(bool(self.ctrl.fec_cfg.get("enabled", False)))
        self.fec_check.toggled.connect(self._on_fec_toggle)
        tb.addWidget(self.fec_check)
        fec_btn = QtWidgets.QPushButton("FEC…")
        fec_btn.clicked.connect(self._toggle_fec_window)
        tb.addWidget(fec_btn)
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

    def _detector_label(self):
        return {"slicer": "Slicer", "dfe": "DFE", "mlsd": "MLSD"}.get(
            self.ctrl.detector_cfg.get("mode", "dfe"), "DFE")

    def _update_static_plots(self):
        ber = self.ctrl.ber
        self.cascade.update_cascade(self.ctrl.cascade)
        self.sbr.update_sbr(self.ctrl.sbr)
        self.eye.set_metrics(ber.eye_width_ui, ber.target_ber)
        self.eye.set_detector_note(
            "MLSD active — eye opening is not the BER predictor"
            if self.ctrl.detector_cfg.get("mode") == "mlsd" else "")
        if self.bathtub_win is not None and self.bathtub_win.isVisible():
            self.bathtub_win.update_bathtub(ber, self.ctrl.fec_result, self._detector_label())

    def _refresh_report(self):
        if self.report_win is not None and self.report_win.isVisible():
            snap = self.ctrl.latest()
            self.report_win.refresh(self.ctrl.ber, snap.stats if snap else {}, self.ctrl.pipe,
                                    self.ctrl.fec_result, self.ctrl.detector_cfg)

    def _resync_panels(self):
        for name, panel in self.panels.items():
            panel.sync(self.ctrl.pipe.by_name(name))
        self._resync_eq()

    def _tick(self):
        snap = self.ctrl.latest()
        if snap is not None:
            self.eye.update_eye(snap)
            self.hist.update_hist(snap)
            self._refresh_report()

    # -- EQ bypass toggles ----------------------------------------------------
    def _eq_is_on(self, block: str, param: str) -> bool:
        try:
            return self.ctrl.pipe.by_name(block).get(param) == "on"
        except KeyError:
            return False

    def _resync_eq(self):
        """Reflect the pipeline's enable flags (after rebuild / load / auto-EQ)."""
        for (block, param), cb in self.eq_checks.items():
            cb.blockSignals(True)
            cb.setChecked(self._eq_is_on(block, param))
            cb.blockSignals(False)

    # -- side windows ---------------------------------------------------------
    def _toggle_bathtub(self):
        if self.bathtub_win is None:
            self.bathtub_win = BathtubWindow()
            self.bathtub_win.set_io_dir(self._io_dir)
        self.bathtub_win.update_bathtub(self.ctrl.ber, self.ctrl.fec_result, self._detector_label())
        self.bathtub_win.show()
        self.bathtub_win.raise_()

    def _toggle_report(self):
        if self.report_win is None:
            self.report_win = ReportWindow()
            self.report_win.set_io_dir(self._io_dir)
        self.report_win.show()
        self.report_win.raise_()
        self._refresh_report()

    # -- FEC ------------------------------------------------------------------
    def _on_fec_toggle(self, on: bool):
        self._apply_fec_cfg({**self.ctrl.fec_cfg, "enabled": on})

    def _toggle_fec_window(self):
        if self.fec_win is None:
            self.fec_win = FecWindow(self.ctrl.fec_cfg, self._apply_fec_cfg, self.ctrl.cfg.modulation)
        self.fec_win.set_modulation(self.ctrl.cfg.modulation)
        self.fec_win.show()
        self.fec_win.raise_()

    def _apply_fec_cfg(self, cfg: dict):
        """Single entry point for any FEC config change (toolbar toggle or window)."""
        self.ctrl.on_fec_change(cfg)
        self.fec_check.blockSignals(True)
        self.fec_check.setChecked(bool(cfg.get("enabled", False)))
        self.fec_check.blockSignals(False)
        if self.fec_win is not None:
            self.fec_win.sync(cfg)  # blocks its own signals
        self._update_fec_views()

    def _update_fec_views(self):
        if self.bathtub_win is not None and self.bathtub_win.isVisible():
            self.bathtub_win.update_bathtub(self.ctrl.ber, self.ctrl.fec_result, self._detector_label())
        self._refresh_report()

    # -- detector -------------------------------------------------------------
    def _on_detector_mode(self, *_a):
        self._apply_detector_cfg({**self.ctrl.detector_cfg, "mode": self.detector_combo.currentData()})

    def _toggle_detector_window(self):
        if self.detector_win is None:
            self.detector_win = DetectorWindow(self.ctrl.detector_cfg, self._apply_detector_cfg,
                                               self.ctrl.cfg.modulation)
        self.detector_win.set_modulation(self.ctrl.cfg.modulation)
        self.detector_win.show()
        self.detector_win.raise_()

    def _apply_detector_cfg(self, cfg: dict):
        """Single entry point for any detector change (toolbar combo or window)."""
        self.ctrl.on_detector_change(cfg)        # sets DFE arch + re-assesses the BER
        self.detector_combo.blockSignals(True)
        i = self.detector_combo.findData(cfg.get("mode", "dfe"))
        self.detector_combo.setCurrentIndex(max(0, i))
        self.detector_combo.blockSignals(False)
        if self.detector_win is not None:
            self.detector_win.sync(cfg)
        self._resync_eq()            # the DFE checkbox follows the selector
        self._update_static_plots()  # eye note + bathtub
        self._refresh_report()

    def _set_io_dir(self, path: str):
        self._io_dir = path
        for p in (self.eye, self.cascade, self.sbr, self.hist):
            p.set_io_dir(path)
        if self.bathtub_win is not None:
            self.bathtub_win.set_io_dir(path)
        if self.report_win is not None:
            self.report_win.set_io_dir(path)

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
        # modulation may change the FEC pairing advisory and the MLSD trellis cap
        if self.fec_win is not None:
            self.fec_win.set_modulation(self.ctrl.cfg.modulation)
        if self.detector_win is not None:
            self.detector_win.set_modulation(self.ctrl.cfg.modulation)
        self.detector_combo.blockSignals(True)  # set_scenario may have re-imposed the DFE arch
        i = self.detector_combo.findData(self.ctrl.detector_cfg.get("mode", "dfe"))
        self.detector_combo.setCurrentIndex(max(0, i))
        self.detector_combo.blockSignals(False)
        self._update_fec_views()

    def _load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load config", self._io_dir, "Config (*.yaml *.json)")
        if path:
            self._set_io_dir(os.path.dirname(path))
            self.ctrl.load_config(path)
            self.mod_combo.blockSignals(True)
            self.rate_combo.blockSignals(True)
            self.mod_combo.setCurrentText(self.ctrl.cfg.modulation)
            self.rate_combo.setCurrentText(str(int(self.ctrl.cfg.data_rate_gbps)))
            self.mod_combo.blockSignals(False)
            self.rate_combo.blockSignals(False)
            self._update_static_plots()
            self._resync_panels()
            self._apply_fec_cfg(self.ctrl.fec_cfg)        # reflect loaded FEC settings in the UI
            self._apply_detector_cfg(self.ctrl.detector_cfg)  # and the loaded detector mode

    def _save(self):
        default = os.path.join(self._io_dir, "link.yaml") if self._io_dir else "link.yaml"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save config", default, "Config (*.yaml *.json)")
        if path:
            self._set_io_dir(os.path.dirname(path))
            self.ctrl.save_config(path)

    def closeEvent(self, event):
        self.ctrl.stop()
        for win in (self.bathtub_win, self.report_win, self.fec_win, self.detector_win):
            if win is not None:
                win.close()
        super().closeEvent(event)


def main(argv=None):
    import signal

    parser = argparse.ArgumentParser(description="EyeQ live dashboard")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = load(args.config) if args.config else default_link_config(modulation="PAM4", reach_class="VSR")

    app = QtWidgets.QApplication(sys.argv)
    # Make Ctrl+C in the launching terminal quit immediately (Qt's C++ event loop
    # otherwise swallows it). The worker is a daemon thread, so the process exits
    # cleanly. Closing the window (red button / Cmd+Q) also shuts down via closeEvent.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # Wake the event loop periodically so the signal is delivered promptly.
    nudge = QtCore.QTimer()
    nudge.start(200)
    nudge.timeout.connect(lambda: None)

    win = Dashboard(cfg)
    win.resize(1280, 760)
    win.show()
    win.run_btn.setChecked(True)  # auto-start the transient engine
    print("EyeQ dashboard running. Close the window (red button / Cmd+Q) or press "
          "Ctrl+C here to quit.")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
