"""Visual-overhaul support: ui-config persistence, colormap/theme registry, SBR
correctness (the Item-4 verification). All headless (no QApplication)."""

import numpy as np
import pytest

from eyeq.engines import StatisticalEngine
from eyeq.io import build_pipeline, default_link_config
from eyeq.io.config import default_ui, from_dict, load, save, to_dict

STAT = StatisticalEngine()


# --------------------------------------------------------------------------- #
# ui config persistence
# --------------------------------------------------------------------------- #
def test_default_ui_shape():
    ui = default_ui()
    assert set(ui) == {"theme", "eye_colormap", "density_scale", "amp_mode",
                       "track_swing", "sbr_labels"}
    assert ui["theme"] == "dark" and ui["eye_colormap"] == "turbo"
    assert ui["density_scale"] == "log" and ui["amp_mode"] == "fixed"
    assert ui["track_swing"] is True and ui["sbr_labels"] is True


def test_decay_for_monotonic_and_clamped():
    from eyeq.engines import decay_for

    factors = [1, 2, 3, 5, 10, 50]
    decays = [decay_for(n) for n in factors]
    assert decays == sorted(decays)                      # more averaging -> more decay
    assert all(0.5 <= d <= 0.97 for d in decays)         # clamped to a usable band
    assert decay_for(1) == 0.5 and decay_for(50) == 0.97  # hit both rails
    assert decay_for(3) == pytest.approx(1 - 1 / 3)      # N=3 -> ~0.667 (default, lively)


def test_ui_config_round_trip(tmp_path):
    cfg = default_link_config()
    cfg.ui.update({"theme": "light", "eye_colormap": "magma", "amp_mode": "auto",
                   "density_scale": "linear", "track_swing": False, "sbr_labels": False})
    path = tmp_path / "c.yaml"
    save(cfg, path)
    r = load(path)
    assert r.ui == {"theme": "light", "eye_colormap": "magma",
                    "amp_mode": "auto", "density_scale": "linear",
                    "track_swing": False, "sbr_labels": False}


def test_old_config_without_ui_loads():
    d = to_dict(default_link_config())
    d.pop("ui")
    cfg = from_dict(d)
    assert cfg.ui == {}  # -> controller falls back to default_ui()


# --------------------------------------------------------------------------- #
# SBR correctness (Item 4 verification, no change to the math)
# --------------------------------------------------------------------------- #
def test_sbr_main_cursor_is_peak_and_ordered():
    p = build_pipeline(default_link_config(modulation="PAM4", reach_class="VSR"))
    sbr = STAT.sbr(p)
    # main cursor is the pulse peak, at the argmax index
    assert sbr.main_idx == int(np.argmax(sbr.sbr))
    assert sbr.main_cursor == pytest.approx(float(sbr.sbr.max()))
    # cursors are strictly ordered with exactly one main (k=0), pre precede / post follow
    assert (np.diff(sbr.cursor_k) > 0).all()
    assert int((sbr.cursor_k == 0).sum()) == 1
    assert sbr.cursor_k[0] < 0 < sbr.cursor_k[-1]
    assert sbr.main_cursor > 0


def test_sbr_scales_with_swing():
    p = build_pipeline(default_link_config(modulation="PAM4", reach_class="VSR"))
    a = STAT.sbr(p).main_cursor
    p.apply_params({"txffe": {"swing": 0.4}})  # half the launch swing
    b = STAT.sbr(p).main_cursor
    assert b == pytest.approx(a * 0.5, rel=1e-6)  # SBR ∝ swing/2


# --------------------------------------------------------------------------- #
# theme + colormap registry (needs pyqtgraph/PySide6, but no display)
# --------------------------------------------------------------------------- #
def test_theme_and_colormaps():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from eyeq.gui import theme

    assert set(theme.THEME_NAMES) == {"dark", "light"}
    for t in theme.THEMES.values():
        assert {"plot_bg", "axis", "text", "heatmap_bg"} <= set(t)
    assert len(theme.COLORMAPS) >= 5 and theme.COLORMAPS[0] == "turbo"
    assert {"turbo", "jet"} <= set(theme.COLORMAPS)  # rainbow maps available + selectable
    for name in theme.COLORMAPS:
        lut = theme.eye_lut(name)
        assert lut.shape == (256, 4) and lut.dtype == np.uint8
