"""GUI dashboard (Phase 4).

The Controller (routing/worker/auto-EQ logic) needs no display and is tested
everywhere. The Qt *widget* tests require a working Qt platform plugin; they run
on a real display and skip in headless sandboxes (creating a QApplication with no
platform aborts the process, so availability is probed in a subprocess first).
"""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from eyeq.gui.dashboard import Controller
from eyeq.io import default_link_config, load, save


def _gui_available() -> bool:
    probe = "from PySide6 import QtWidgets; QtWidgets.QApplication([])"
    try:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


GUI = _gui_available()
needs_display = pytest.mark.skipif(not GUI, reason="no working Qt platform plugin")


# --------------------------------------------------------------------------- #
# Controller — pure logic, no display
# --------------------------------------------------------------------------- #
def _ctrl():
    return Controller(default_link_config(modulation="PAM4", reach_class="VSR"))


def test_controller_constructs():
    c = _ctrl()
    assert c.cascade.nyquist_loss_db == pytest.approx(16.0, abs=0.2)
    assert c.sbr.main_cursor > 0


def test_routing_by_kind():
    c = _ctrl()
    assert c.on_param("ctle", "fz", 0.45) == "lti"
    assert c.on_param("dfe", "h1", 10.0) == "nonlinear"
    assert c.on_param("channel", "model", "tl") == "structural"
    assert c.on_param("channel", "reach", "MR") == "scenario"
    assert c.cfg.reach_class == "MR"


def test_scenario_change_rebuilds_context():
    c = _ctrl()
    c.set_scenario(modulation="NRZ", rate=224.0)
    assert c.pipe.ctx.mod.name == "NRZ"
    assert c.pipe.ctx.data_rate == pytest.approx(224e9)
    assert c.pipe.ctx.f_nyq == pytest.approx(112e9)


def test_auto_eq_improves_live_snr():
    c = _ctrl()
    c.start()
    try:
        time.sleep(0.5)
        before = c.latest().stats["mse_snr_db"]
        c.auto_eq()
        time.sleep(0.5)
        after = c.latest()
        assert after.stats["mse_snr_db"] > before
        assert after.stats["ser"] <= 1e-2
    finally:
        c.stop()


def test_config_round_trip(tmp_path):
    c = _ctrl()
    c.on_param("ctle", "fz", 0.42)
    path = tmp_path / "link.yaml"
    c.save_config(path)
    reloaded = load(path)
    # the ctle fz override survived the save
    ctle_cfg = next(b for b in reloaded.blocks if b.type == "CTLE")
    assert ctle_cfg.params["fz"] == pytest.approx(0.42)


# --------------------------------------------------------------------------- #
# Widgets — require a Qt platform (run on a real display)
# --------------------------------------------------------------------------- #
@needs_display
def test_dashboard_builds_and_renders(tmp_path):
    from PySide6 import QtWidgets

    from eyeq.gui.dashboard import Dashboard

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Dashboard(default_link_config(modulation="PAM4", reach_class="VSR"))
    win.resize(1100, 700)
    win.ctrl.auto_eq()
    win.ctrl.start()
    t0 = time.time()
    while time.time() - t0 < 1.0:
        app.processEvents()
        time.sleep(0.02)
    win._tick()
    snap = win.ctrl.latest()
    assert snap is not None and np.all(np.isfinite(snap.image))
    # the auto-generated panel has one group per block
    assert set(win.panels) == set(win.ctrl.pipe.names())
    win.ctrl.stop()


@needs_display
def test_param_panel_emits_changes():
    from PySide6 import QtWidgets

    from eyeq.gui.binding import build_param_panel
    from eyeq.io import build_pipeline

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pipe = build_pipeline(default_link_config())
    received = []
    _, panels = build_param_panel(pipe, lambda b, p, v: received.append((b, p, v)))
    panels["ctle"].controls["fz"].slider.setValue(500)
    assert received and received[-1][0] == "ctle" and received[-1][1] == "fz"
