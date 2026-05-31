"""Closed-form MMSE auto-EQ (Phase 3)."""

import numpy as np
import pytest

from eyeq.analysis.optimize import mmse_ffe, optimize_link, solve_dfe
from eyeq.engines import StatisticalEngine, TransientEngine
from eyeq.io import build_pipeline, default_link_config

STAT, TRAN = StatisticalEngine(), TransientEngine()


def pipe(mod="PAM4", reach="VSR"):
    p = build_pipeline(default_link_config(modulation=mod, reach_class=reach))
    p.apply_params({"ctle": {"fz": 0.35, "fp": 1.0, "dc_gain": -2.0}})
    return p


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #
def test_mmse_ffe_reduces_isi():
    # A dispersive pulse: one main + two trailing cursors.
    x = np.array([0.05, 1.0, 0.4, 0.2, 0.05])
    w, main_pos, mmse = mmse_ffe(x, 7, sig_var=1.0, noise_var=1e-4)
    out = np.convolve(w, x)
    out_main = int(np.argmax(np.abs(out)))
    isi = (np.sum(np.abs(out)) - abs(out[out_main])) / abs(out[out_main])
    raw_isi = (np.sum(np.abs(x)) - abs(x[np.argmax(np.abs(x))])) / abs(x[np.argmax(np.abs(x))])
    assert isi < raw_isi * 0.5  # FFE at least halves the relative ISI
    assert abs(out[out_main]) == pytest.approx(abs(x[np.argmax(np.abs(x))]), rel=0.3)


def test_solve_dfe_returns_postcursors():
    taps = solve_dfe(np.array([0.03, 0.02, 0.01]), 2)
    assert taps.size == 2
    assert np.allclose(taps, [0.03, 0.02])


# --------------------------------------------------------------------------- #
# optimize_link opens closed eyes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod,reach", [("PAM4", "VSR"), ("NRZ", "VSR"), ("PAM4", "MR"), ("PAM4", "LR")])
def test_optimize_link_opens_eye(mod, reach):
    p = pipe(mod, reach)
    _, sbr0, eye0 = STAT.compute(p)
    before = TRAN.run_batch(p, n_symbols=80_000, sbr=sbr0, v=eye0.v, rng=np.random.default_rng(0))

    optimize_link(p)
    _, sbr1, eye1 = STAT.compute(p)
    after = TRAN.run_batch(p, n_symbols=80_000, sbr=sbr1, v=eye1.v, rng=np.random.default_rng(0))

    assert eye0.eye_height_v == 0.0          # closed before EQ
    assert eye1.eye_height_v > 0.0           # open after EQ
    assert after.ser < before.ser
    assert after.mse_snr_db > before.mse_snr_db + 3.0


def test_optimize_keeps_physical_scale():
    # The equalized eye must not exceed the launch swing (no runaway FFE gain).
    p = pipe("NRZ", "VSR")
    optimize_link(p)
    _, sbr, eye = STAT.compute(p)
    swing = p.by_name("txffe").get("swing")
    assert eye.eye_height_v < swing  # comfortably below full-scale


def test_optimize_sets_block_taps():
    p = pipe("PAM4", "MR")
    res = optimize_link(p, n_rxffe=11, n_dfe=8)
    assert p.by_name("rxffe").taps().size == 11
    assert p.by_name("dfe").get("n_taps") == 8
    assert res.rxffe_taps.size == 11 and res.dfe_taps.size == 8
