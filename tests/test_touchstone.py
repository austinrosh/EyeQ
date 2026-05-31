"""Synthetic .s4p generation + Touchstone import (Phase 2a)."""

import numpy as np
import pytest

pytest.importorskip("skrf")

from eyeq.core.context import REACH_PRESETS, Modulation, SimContext
from eyeq.io import build_pipeline, default_link_config
from eyeq.io.synth_channel import build_network, differential_transfer, write_reference_s4p
from eyeq.io.touchstone import load_sdd21, s4p_to_transfer, sdd21_from_se


def ctx(mod=Modulation.PAM4, reach="XSR"):
    return SimContext.from_data_rate(
        112.0, mod, reach=REACH_PRESETS[("112G", reach)], sps=32
    )


@pytest.fixture(scope="module")
def s4p_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("s4p")
    for reach in ("XSR", "VSR", "MR", "LR"):
        write_reference_s4p(d, "112G", reach)
    return d


def _path(d, reach):
    return str(d / f"112G_{reach.replace('+', 'plus')}.s4p")


# --------------------------------------------------------------------------- #
# mixed-mode transform is exactly invertible (synth <-> import)
# --------------------------------------------------------------------------- #
def test_mixed_mode_roundtrip_is_exact():
    f = np.linspace(0, 120e9, 401)
    net = build_network("112G", "VSR", n=401)
    recovered = sdd21_from_se(np.asarray(net.s))
    original = differential_transfer(f, "112G", "VSR")
    assert np.allclose(recovered, original, atol=1e-9)


# --------------------------------------------------------------------------- #
# importing a smooth channel reproduces the loss budget
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reach,budget", [("XSR", 8.0), ("VSR", 16.0)])
def test_import_smooth_matches_budget(s4p_dir, reach, budget):
    c = ctx(reach=reach)
    H = s4p_to_transfer(_path(s4p_dir, reach), c)
    i28 = int(np.argmin(np.abs(c.freq_grid() - 28e9)))
    assert -20 * np.log10(abs(H[i28])) == pytest.approx(budget, abs=0.3)
    assert abs(H[0]) == pytest.approx(1.0, abs=0.02)  # ~0 dB at DC


def test_import_matches_analytical_magnitude_in_band(s4p_dir):
    # A smooth imported channel ~ the analytical channel in magnitude.
    c = ctx(reach="VSR")
    ch = build_pipeline(default_link_config(reach_class="VSR")).by_name("channel")
    analytical = np.abs(ch.transfer(c))
    imported = np.abs(s4p_to_transfer(_path(s4p_dir, "VSR"), c))
    band = (c.freq_grid() > 1e9) & (c.freq_grid() < 50e9)
    assert np.allclose(imported[band], analytical[band], rtol=0.05, atol=0.01)


# --------------------------------------------------------------------------- #
# MR/LR carry reflection notches the analytical model cannot
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reach", ["MR", "LR"])
def test_mr_lr_have_reflection_notches(s4p_dir, reach):
    c = ctx(reach=reach)
    f = c.freq_grid()
    imported = np.abs(s4p_to_transfer(_path(s4p_dir, reach), c))
    ch = build_pipeline(default_link_config(reach_class=reach)).by_name("channel")
    analytical = np.abs(ch.transfer(c))  # smooth, monotonic
    band = (f > 5e9) & (f < 70e9)
    # analytical is monotonic; the imported channel rises again after a notch.
    assert np.all(np.diff(analytical[band]) <= 1e-9)
    assert np.any(np.diff(imported[band]) > 1e-4)


# --------------------------------------------------------------------------- #
# passivity & out-of-band behavior
# --------------------------------------------------------------------------- #
def test_import_is_passive(s4p_dir):
    c = ctx(reach="LR")
    H = s4p_to_transfer(_path(s4p_dir, "LR"), c)
    assert np.all(np.abs(H) <= 1.0 + 1e-9)


def test_transfer_tapers_to_zero_above_file_band(s4p_dir):
    c = ctx(reach="XSR")  # ctx grid extends to ~896 GHz; file stops at 120 GHz
    H = s4p_to_transfer(_path(s4p_dir, "XSR"), c)
    f = c.freq_grid()
    assert np.all(np.abs(H[f > 130e9]) < 1e-6)


# --------------------------------------------------------------------------- #
# wired into the Channel block / config
# --------------------------------------------------------------------------- #
def test_channel_touchstone_model(s4p_dir):
    c = ctx(reach="VSR")
    ch = build_pipeline(default_link_config(reach_class="VSR")).by_name("channel")
    ch.set_touchstone(_path(s4p_dir, "VSR"))
    ch.set_params(model="touchstone")
    H = ch.transfer(c)
    i28 = int(np.argmin(np.abs(c.freq_grid() - 28e9)))
    assert -20 * np.log10(abs(H[i28])) == pytest.approx(16.0, abs=0.3)


def test_touchstone_model_requires_path():
    c = ctx()
    ch = build_pipeline(default_link_config()).by_name("channel")
    ch.set_params(model="touchstone")
    with pytest.raises(ValueError):
        ch.transfer(c)


def test_build_pipeline_sets_touchstone_path(s4p_dir):
    cfg = default_link_config(reach_class="VSR")
    cfg.channel_s4p = _path(s4p_dir, "VSR")
    pipe = build_pipeline(cfg)
    assert pipe.by_name("channel").touchstone_path == _path(s4p_dir, "VSR")


def test_statistical_engine_runs_through_imported_channel(s4p_dir):
    # End-to-end: imported .s4p -> cascade -> SBR -> statistical eye.
    from eyeq.engines import StatisticalEngine

    cfg = default_link_config(reach_class="XSR")
    cfg.channel_s4p = _path(s4p_dir, "XSR")
    pipe = build_pipeline(cfg)
    pipe.apply_params({
        "channel": {"model": "touchstone"},
        "ctle": {"fz": 0.35, "fp": 1.0, "dc_gain": -2.0},
        "txffe": {"pre": -0.08, "post": -0.12},
    })
    cascade, sbr, eye = StatisticalEngine().compute(pipe)
    i28 = int(np.argmin(np.abs(cascade.f - 28e9)))
    assert -20 * np.log10(abs(cascade.H_channel[i28])) == pytest.approx(8.0, abs=0.5)
    assert sbr.main_cursor > 0 and np.isfinite(sbr.main_cursor)
    assert eye.eye_height_v > 0  # XSR opens an eye off the imported channel too
