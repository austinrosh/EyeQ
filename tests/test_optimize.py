"""Closed-form MMSE auto-EQ (Phase 3)."""

import numpy as np
import pytest

from eyeq.analysis.optimize import (
    _link_snr_db,
    mmse_ffe,
    optimize_link,
    solve_dfe,
    solve_tx_ffe,
)
from eyeq.engines import StatisticalEngine, TransientEngine
from eyeq.io import build_pipeline, default_link_config

STAT, TRAN = StatisticalEngine(), TransientEngine()


def _reflective_s4p(tmp_path, reach="LR"):
    from eyeq.io.synth_channel import write_reference_s4p

    return str(write_reference_s4p(tmp_path, "112G", reach))


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
    assert res.txffe_taps.size >= 1


# --------------------------------------------------------------------------- #
# TX FFE auto-EQ + the optimum TX/RX split
# --------------------------------------------------------------------------- #
def test_solve_tx_ffe_removes_precursors():
    pre_main = np.array([0.1, -0.15, 0.3, 1.0])  # [pre.. main]
    v, mp = solve_tx_ffe(pre_main, 4)
    out = np.convolve(v, pre_main)
    main = int(np.argmax(np.abs(out)))
    assert np.sum(out[:main] ** 2) < 0.1 * np.sum(pre_main[:-1] ** 2)  # precursors crushed


def _snr(p):
    return _link_snr_db(STAT, p, p.ctx.default_dfe_taps())


def test_co_opt_never_worse_than_rx_only_on_smooth_channel():
    p_rx = pipe("PAM4", "MR")
    p_rx.apply_params({"noise": {"sigma_mvrms": 2.0}})
    p_co = pipe("PAM4", "MR")
    p_co.apply_params({"noise": {"sigma_mvrms": 2.0}})
    optimize_link(p_rx, tx_ffe=False)
    optimize_link(p_co, tx_ffe=True)
    assert _snr(p_co) >= _snr(p_rx) - 0.05


def test_tx_ffe_helps_on_reflective_channel(tmp_path):
    s4p = _reflective_s4p(tmp_path, "LR")  # notches -> large pre-cursors

    def make():
        cfg = default_link_config(modulation="PAM4", reach_class="LR")
        cfg.channel_s4p = s4p
        p = build_pipeline(cfg)
        p.apply_params({"channel": {"model": "touchstone"}, "ctle": {"fz": 0.35, "fp": 1.0},
                        "noise": {"sigma_mvrms": 1.5}})
        return p

    rx, co = make(), make()
    optimize_link(rx, tx_ffe=False)
    optimize_link(co, tx_ffe=True)
    assert _snr(co) > _snr(rx) + 0.3                 # TX FFE buys real margin
    assert co.by_name("txffe")._taps is not None      # it actually engaged the TX FFE


# --------------------------------------------------------------------------- #
# front-end noise model: the RX FFE amplifies RX noise
# --------------------------------------------------------------------------- #
def test_rx_ffe_amplifies_referred_noise():
    p = pipe("PAM4", "VSR")
    p.apply_params({"noise": {"sigma_mvrms": 5.0}})
    sigma_identity = STAT._amplitude_sigma(p)        # RX FFE = identity
    optimize_link(p)                                 # equalizing RX FFE
    sigma_eq = STAT._amplitude_sigma(p)
    assert sigma_eq > sigma_identity                 # equalizer pays a noise penalty
