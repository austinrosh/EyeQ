"""Dedicated, openable side windows for the dashboard.

* :class:`BathtubWindow` — the vertical + horizontal bathtub curves promoted out
  of the voltage panel into their own window beside the eye (each marks the eye
  opening at the target BER).
* :class:`ReportWindow`  — the link performance report: an extensible metric
  table (BER, COM, eye height/width, SNR, active EQ/CDR state, plus deferred
  compliance metrics) driven by :mod:`eyeq.analysis.report`, with capture/compare
  across configurations.

Both are lazily created and toggled from the dashboard toolbar; the dashboard
pushes fresh data while the window is open. They are thin views — the bathtub
math lives in :mod:`eyeq.analysis.ber` and the metric registry in
:mod:`eyeq.analysis.report`.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis import fec as _fec
from ..analysis import mlsd as _mlsd
from ..analysis import report
from .plots import BathtubPlot

_FMT = {m.key: m.fmt for m in report.METRICS}


# --------------------------------------------------------------------------- #
class DisplayPanel(QtWidgets.QWidget):
    """Always-visible view/display settings — the in-window home for everything that
    used to hide in the (macOS-global) View menu: theme, eye colormap, density/amplitude
    scaling, swing tracking, SBR labels, and eye liveliness.

    Emits ``on_view(key, value)`` for ui-config settings and ``on_avg(n)`` for the
    eye-averaging factor; the dashboard routes both to the live plots/worker.
    """

    def __init__(self, ui_cfg: dict, avg_factor: int, colormaps, avg_factors,
                 on_view, on_avg):
        super().__init__()
        self._on_view = on_view
        self._on_avg = on_avg
        self._avg_factors = list(avg_factors)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        box = QtWidgets.QGroupBox("Display")
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(10, 10, 10, 8)
        form.setSpacing(9)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.theme = self._combo([("Dark", "dark"), ("Light", "light")],
                                 ui_cfg.get("theme", "dark"), "theme")
        self.colormap = self._combo([(c.capitalize(), c) for c in colormaps],
                                    ui_cfg.get("eye_colormap", "turbo"), "eye_colormap")
        self.density = self._combo([("Log", "log"), ("Linear", "linear")],
                                   ui_cfg.get("density_scale", "log"), "density_scale")
        self.amp = self._combo([("Fixed", "fixed"), ("Auto-fit", "auto")],
                               ui_cfg.get("amp_mode", "fixed"), "amp_mode")
        self.live = self._combo([(self._live_label(n), n) for n in self._avg_factors],
                                int(avg_factor), None)
        self.live.currentIndexChanged.connect(
            lambda *_: self._on_avg(int(self.live.currentData())))
        self.track = self._check(ui_cfg.get("track_swing", True), "track_swing")
        self.labels = self._check(ui_cfg.get("sbr_labels", True), "sbr_labels")

        for label, w in (("Theme", self.theme), ("Eye colormap", self.colormap),
                         ("Density scale", self.density), ("Amplitude axis", self.amp),
                         ("Eye liveliness", self.live)):
            form.addRow(label, w)
        form.addRow(self.track)
        form.addRow(self.labels)
        hint = QtWidgets.QLabel("Track swing off → eye/SBR sit on a fixed scale and grow "
                                "or shrink as the launch swing changes.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8b949e; font-size:11px;")
        form.addRow(hint)
        outer.addWidget(box)
        outer.addStretch(1)

    # -- construction helpers -------------------------------------------------
    @staticmethod
    def _live_label(n: int) -> str:
        tag = "live" if n <= 2 else "smooth" if n >= 10 else "balanced"
        return f"{n}  ({tag})"

    def _combo(self, items, current, key):
        cb = QtWidgets.QComboBox()
        for label, data in items:
            cb.addItem(label, data)
        i = cb.findData(current)
        cb.setCurrentIndex(i if i >= 0 else 0)
        if key is not None:
            cb.currentIndexChanged.connect(
                lambda *_a, c=cb, k=key: self._on_view(k, c.currentData()))
        return cb

    def _check(self, checked, key):
        cb = QtWidgets.QCheckBox()
        cb.setChecked(bool(checked))
        cb.setText({"track_swing": "Track swing", "sbr_labels": "SBR cursor labels"}.get(key, ""))
        cb.toggled.connect(lambda on, k=key: self._on_view(k, on))
        return cb

    # -- external sync (config load / scenario) -------------------------------
    def sync(self, ui_cfg: dict, avg_factor: int) -> None:
        pairs = [(self.theme, ui_cfg.get("theme", "dark")),
                 (self.colormap, ui_cfg.get("eye_colormap", "turbo")),
                 (self.density, ui_cfg.get("density_scale", "log")),
                 (self.amp, ui_cfg.get("amp_mode", "fixed")),
                 (self.live, int(avg_factor))]
        for cb, val in pairs:
            cb.blockSignals(True)
            i = cb.findData(val)
            cb.setCurrentIndex(i if i >= 0 else 0)
            cb.blockSignals(False)
        for chk, val in ((self.track, ui_cfg.get("track_swing", True)),
                         (self.labels, ui_cfg.get("sbr_labels", True))):
            chk.blockSignals(True)
            chk.setChecked(bool(val))
            chk.blockSignals(False)


def _fmt_val(key: str, raw) -> str:
    if raw is None:
        return "—"
    if isinstance(raw, str):
        return raw
    try:
        return _FMT.get(key, "{:.3g}").format(raw)
    except Exception:
        return str(raw)


# --------------------------------------------------------------------------- #
class BathtubWindow(QtWidgets.QWidget):
    """Vertical + horizontal bathtub curves in a window beside the eye."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeQ — Bathtub curves")
        self.resize(460, 620)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.vert = BathtubPlot("vertical")
        self.horiz = BathtubPlot("horizontal")
        lay.addWidget(self.vert, 1)
        lay.addWidget(self.horiz, 1)

    def update_bathtub(self, ber, fec=None, detector_label=None) -> None:
        if ber is None:
            return
        self.vert.update_bathtub(ber, fec, detector_label)
        self.horiz.update_bathtub(ber, fec, detector_label)

    def set_io_dir(self, path: str) -> None:
        self.vert.set_io_dir(path)
        self.horiz.set_io_dir(path)


# --------------------------------------------------------------------------- #
class ReportWindow(QtWidgets.QWidget):
    """Extensible link-performance report with capture/compare across configs."""

    _BASE_COLS = ["Metric", "Unit", "Definition", "Live"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeQ — Link report")
        self.resize(720, 460)
        self._io_dir = ""
        self._captures: list[tuple[str, dict]] = []
        self._live_raw: dict = {}
        self._meta = [(m.key, m.label, m.unit, m.definition, m.model_limited) for m in report.METRICS]

        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        for label, slot in (("Capture config", self._capture),
                            ("Clear captures", self._clear),
                            ("Export CSV…", self._export_csv),
                            ("Export PNG…", self._export_png)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table)
        note = QtWidgets.QLabel("Greyed rows are model-limited or not yet modeled — no fabricated "
                                "precision. Capture a config, change the EQ, and compare.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;")
        lay.addWidget(note)

        self._build_table()

    # -- table construction ---------------------------------------------------
    def _headers(self) -> list[str]:
        return self._BASE_COLS + [name for name, _ in self._captures]

    def _build_table(self) -> None:
        headers = self._headers()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self._meta))
        for r, (key, label, unit, definition, model_limited) in enumerate(self._meta):
            grey = model_limited
            self._set(r, 0, label, tip=definition, grey=grey)
            self._set(r, 1, unit, grey=grey)
            self._set(r, 2, definition, grey=grey)
            self._set(r, 3, "—", grey=grey)            # Live, filled by refresh()
            for c, (_name, snap) in enumerate(self._captures):
                self._set(r, 4 + c, _fmt_val(key, snap.get(key)), grey=grey)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Stretch)  # let Definition take the slack

    def _set(self, r, c, text, *, tip=None, grey=False) -> None:
        item = QtWidgets.QTableWidgetItem(str(text))
        if tip:
            item.setToolTip(tip)
        if grey:
            item.setForeground(QtGui.QColor("#888"))
        self.table.setItem(r, c, item)

    # -- live refresh ---------------------------------------------------------
    def refresh(self, ber, stats: dict, pipe, fec=None, detector=None) -> None:
        rc = report.ReportContext(ber=ber, stats=stats or {}, pipe=pipe, fec=fec, detector=detector)
        rows = report.evaluate(rc)
        self._live_raw = {row.key: row.raw for row in rows}
        for r, row in enumerate(rows):
            live = self.table.item(r, 3)
            if live is not None:
                live.setText(row.value)
                live.setForeground(QtGui.QColor("#888") if (row.model_limited or row.raw is None)
                                   else QtGui.QColor("#ddd"))
            # refresh the per-capture Δ vs the new live value
            for c, (_name, snap) in enumerate(self._captures):
                cell = self.table.item(r, 4 + c)
                if cell is not None:
                    cell.setText(self._capture_text(row.key, snap.get(row.key)))

    def _capture_text(self, key, cap_raw) -> str:
        txt = _fmt_val(key, cap_raw)
        live_raw = self._live_raw.get(key)
        if isinstance(cap_raw, (int, float)) and isinstance(live_raw, (int, float)):
            txt += f"  (Δ {cap_raw - live_raw:+.3g})"
        return txt

    # -- capture/compare ------------------------------------------------------
    def _capture(self) -> None:
        name = f"Cap {len(self._captures) + 1}"
        self._captures.append((name, dict(self._live_raw)))
        self._build_table()

    def _clear(self) -> None:
        self._captures.clear()
        self._build_table()

    # -- export ---------------------------------------------------------------
    def set_io_dir(self, path: str) -> None:
        self._io_dir = path or ""

    def _default_path(self, ext: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self._io_dir or os.getcwd(), f"report_{ts}.{ext}")

    def _export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export report CSV", self._default_path("csv"), "CSV (*.csv)")
        if not path:
            return
        headers = self._headers()
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(headers)
            for r in range(self.table.rowCount()):
                w.writerow([self.table.item(r, c).text() if self.table.item(r, c) else ""
                            for c in range(len(headers))])

    def _export_png(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export report PNG", self._default_path("png"), "PNG image (*.png)")
        if path:
            self.grab().save(path)


# --------------------------------------------------------------------------- #
class FecWindow(QtWidgets.QWidget):
    """FEC scheme + parameter settings; emits the full FEC config on any change."""

    _SCHEME_ORDER = ["none", "kp4", "kr4", "custom"]
    _TARGETS = ["1e-15", "1e-12", "1e-9"]

    def __init__(self, cfg: dict, on_change, mod_name: str = "PAM4"):
        super().__init__()
        self.setWindowTitle("EyeQ — FEC settings")
        self.resize(420, 360)
        self._cfg = dict(cfg)
        self._on_change = on_change
        self._mod = mod_name

        form = QtWidgets.QFormLayout(self)
        self.enable = QtWidgets.QCheckBox("Enable FEC (post-FEC BER estimate)")
        form.addRow(self.enable)

        self.scheme = QtWidgets.QComboBox()
        for key in self._SCHEME_ORDER:
            label = ("Custom RS(n,k)" if key == "custom" else _fec.SCHEMES[key].label)
            self.scheme.addItem(label, key)
        form.addRow("Scheme", self.scheme)

        self.note = QtWidgets.QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#e8a33d;")
        form.addRow(self.note)

        self.n_spin = self._spin(2, 1023, 544)
        self.k_spin = self._spin(1, 1022, 514)
        self.m_spin = self._spin(2, 12, 10)
        form.addRow("n (codeword symbols)", self.n_spin)
        form.addRow("k (message symbols)", self.k_spin)
        form.addRow("m (symbol bits)", self.m_spin)
        self.t_label = QtWidgets.QLabel("—")
        form.addRow("t (correctable)", self.t_label)

        self.target = QtWidgets.QComboBox()
        self.target.addItems(self._TARGETS)
        form.addRow("Post-FEC target BER", self.target)

        self.error_model = QtWidgets.QComboBox()
        self.error_model.addItems(["random", "bursty"])
        form.addRow("Error model", self.error_model)
        self.burst = self._spin(1, 100000, 1)
        self.interleave = self._spin(1, 4096, 1)
        form.addRow("Burst length (bits)", self.burst)
        form.addRow("Interleave depth", self.interleave)

        self.fidelity = QtWidgets.QLabel(
            "Post-FEC numbers are model-based: hard-decision RS assuming i.i.d. (random) symbol "
            "errors. EyeQ's noise model does not generate real bursts; the bursty option is a "
            "coarse approximation, not a measured burst response.")
        self.fidelity.setWordWrap(True)
        self.fidelity.setStyleSheet("color:#888;")
        form.addRow(self.fidelity)

        self.sync(cfg)
        # wire signals AFTER the initial sync so we don't emit during construction
        self.enable.toggled.connect(self._emit)
        self.scheme.currentIndexChanged.connect(self._emit)
        for sp in (self.n_spin, self.k_spin, self.m_spin, self.burst, self.interleave):
            sp.valueChanged.connect(self._emit)
        self.target.currentTextChanged.connect(self._emit)
        self.error_model.currentTextChanged.connect(self._emit)

    @staticmethod
    def _spin(lo, hi, val):
        s = QtWidgets.QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    # -- external API ---------------------------------------------------------
    def set_modulation(self, mod_name: str) -> None:
        self._mod = mod_name
        self._update_derived()

    def sync(self, cfg: dict) -> None:
        """Reflect an external config (load / toolbar toggle) without emitting."""
        self._cfg = {**self._cfg, **cfg}
        widgets = [self.enable, self.scheme, self.n_spin, self.k_spin, self.m_spin,
                   self.target, self.error_model, self.burst, self.interleave]
        for w in widgets:
            w.blockSignals(True)
        self.enable.setChecked(bool(self._cfg.get("enabled", False)))
        i = self.scheme.findData(self._cfg.get("scheme", "kp4"))
        self.scheme.setCurrentIndex(max(0, i))
        self.n_spin.setValue(int(self._cfg.get("custom_n", 544)))
        self.k_spin.setValue(int(self._cfg.get("custom_k", 514)))
        self.m_spin.setValue(int(self._cfg.get("custom_m", 10)))
        ti = self.target.findText(_fmt_target(self._cfg.get("target_post_ber", 1e-15)))
        self.target.setCurrentIndex(ti if ti >= 0 else 0)
        ei = self.error_model.findText(self._cfg.get("error_model", "random"))
        self.error_model.setCurrentIndex(max(0, ei))
        self.burst.setValue(int(self._cfg.get("burst_len_bits", 1)))
        self.interleave.setValue(int(self._cfg.get("interleave_depth", 1)))
        for w in widgets:
            w.blockSignals(False)
        self._update_derived()

    # -- internals ------------------------------------------------------------
    def _read(self) -> dict:
        cfg = dict(self._cfg)
        cfg["enabled"] = self.enable.isChecked()
        cfg["scheme"] = self.scheme.currentData()
        cfg["target_post_ber"] = float(self.target.currentText())
        cfg["error_model"] = self.error_model.currentText()
        cfg["burst_len_bits"] = self.burst.value()
        cfg["interleave_depth"] = self.interleave.value()
        if cfg["scheme"] == "custom":
            cfg["custom_n"] = self.n_spin.value()
            cfg["custom_k"] = self.k_spin.value()
            cfg["custom_m"] = self.m_spin.value()
        return cfg

    def _emit(self, *_a) -> None:
        self._cfg = self._read()
        self._update_derived()
        self._on_change(self._cfg)

    def _update_derived(self) -> None:
        scheme = _fec.scheme_from_cfg(self._cfg)
        is_custom = self._cfg.get("scheme") == "custom"
        # show the resolved n/k/m for named schemes (read-only); editable for custom
        for sp, val in ((self.n_spin, scheme.n), (self.k_spin, scheme.k), (self.m_spin, scheme.m)):
            sp.setEnabled(is_custom)
            if not is_custom and scheme.kind == "rs":
                sp.blockSignals(True)
                sp.setValue(int(val))
                sp.blockSignals(False)
        self.t_label.setText(str(scheme.t) if scheme.kind == "rs" else "—")
        bursty = self._cfg.get("error_model") == "bursty"
        self.burst.setEnabled(bursty)
        self.interleave.setEnabled(bursty)
        # soft pairing advisory
        applicable = (not scheme.pairing) or (self._mod in scheme.pairing)
        if scheme.kind != "none" and not applicable:
            self.note.setText(f"Note: {scheme.label.split(' ')[0]} is the standard pairing for "
                              f"{'/'.join(scheme.pairing)} — selectable on {self._mod} anyway.")
        else:
            self.note.setText("")


def _fmt_target(v) -> str:
    return f"{float(v):.0e}".replace("e-0", "e-")


# --------------------------------------------------------------------------- #
class DetectorWindow(QtWidgets.QWidget):
    """Receiver detector selector + MLSD trellis settings; emits the detector cfg."""

    _MODES = [("Slicer", "slicer"), ("DFE", "dfe"), ("MLSD (Viterbi)", "mlsd")]

    def __init__(self, cfg: dict, on_change, mod_name: str = "PAM4"):
        super().__init__()
        self.setWindowTitle("EyeQ — Detector")
        self.resize(440, 300)
        self._cfg = dict(cfg)
        self._on_change = on_change
        self._mod = mod_name

        form = QtWidgets.QFormLayout(self)
        self.mode = QtWidgets.QComboBox()
        for label, key in self._MODES:
            self.mode.addItem(label, key)
        form.addRow("Detector", self.mode)

        self.taps = QtWidgets.QSpinBox()
        self.taps.setRange(1, 8)
        form.addRow("Trellis memory L (taps)", self.taps)
        self.states = QtWidgets.QLabel("—")
        form.addRow("≈ trellis states", self.states)

        self.note = QtWidgets.QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#e8a33d;")
        form.addRow(self.note)

        self.fidelity = QtWidgets.QLabel(
            "MLSD BER is model-based: a minimum-distance union bound (optimistic at low SNR), "
            "assuming a whitened-matched-filter front end and the same noise limits as the rest of "
            "the tool. Under MLSD the eye-opening metrics no longer predict the BER.")
        self.fidelity.setWordWrap(True)
        self.fidelity.setStyleSheet("color:#888;")
        form.addRow(self.fidelity)

        self.sync(cfg)
        self.mode.currentIndexChanged.connect(self._emit)
        self.taps.valueChanged.connect(self._emit)

    # -- external API ---------------------------------------------------------
    def set_modulation(self, mod_name: str) -> None:
        self._mod = mod_name
        self._update_derived()

    def sync(self, cfg: dict) -> None:
        self._cfg = {**self._cfg, **cfg}
        for w in (self.mode, self.taps):
            w.blockSignals(True)
        i = self.mode.findData(self._cfg.get("mode", "dfe"))
        self.mode.setCurrentIndex(max(0, i))
        self.taps.setValue(int(self._cfg.get("mlsd_taps", 4)))
        for w in (self.mode, self.taps):
            w.blockSignals(False)
        self._update_derived()

    # -- internals ------------------------------------------------------------
    def _read(self) -> dict:
        c = dict(self._cfg)
        c["mode"] = self.mode.currentData()
        c["mlsd_taps"] = self.taps.value()
        return c

    def _emit(self, *_a) -> None:
        self._cfg = self._read()
        self._update_derived()
        self._on_change(self._cfg)

    def _update_derived(self) -> None:
        n_levels = 4 if self._mod == "PAM4" else 2
        cap = _mlsd.l_cap(n_levels)
        self.taps.blockSignals(True)
        self.taps.setMaximum(cap)            # guard: PAM-4 capped tighter so M^L can't blow up
        self.taps.blockSignals(False)
        L = min(self.taps.value(), cap)
        is_mlsd = self._cfg.get("mode") == "mlsd"
        self.taps.setEnabled(is_mlsd)
        self.states.setEnabled(is_mlsd)
        self.states.setText(f"{n_levels}^{L} = {n_levels ** L}" if is_mlsd else "—")
        if is_mlsd and n_levels == 4:
            self.note.setText(f"PAM-4 trellis grows as 4^L; L is capped at {cap} to bound compute.")
        else:
            self.note.setText("")
