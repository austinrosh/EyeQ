"""EQ-stage bypass toggles + eye-height-at-phase (UX-polish task, item 1/2).

Each equalizer stage has an independent on/off toggle that is a *true* bypass
(the signal is passed unmodified), not merely zeroed coefficients, and is
independent of the auto-EQ solver (solved taps are preserved across a toggle).
"""

import numpy as np
import pytest

from eyeq.analysis.ber import assess
from eyeq.analysis.optimize import optimize_link
from eyeq.engines import StatisticalEngine, TransientEngine
from eyeq.io import build_pipeline, default_link_config

STAT, TRAN = StatisticalEngine(), TransientEngine()


def pipe(mod="PAM4", reach="VSR", **ov):
    p = build_pipeline(default_link_config(modulation=mod, reach_class=reach))
    if ov:
        p.apply_params(ov)
    return p


# --------------------------------------------------------------------------- #
# LTI stages: a disabled transfer is exactly identity (all ones)
# --------------------------------------------------------------------------- #
def test_ctle_bypass_is_identity():
    p = pipe(**{"ctle": {"dc_gain": -6.0, "fz": 0.3, "fp": 1.2}})
    ctle, ctx = p.by_name("ctle"), p.ctx
    assert not np.allclose(ctle.transfer(ctx), 1.0)  # active CTLE shapes the signal
    ctle.set_params(enabled="off")
    assert np.array_equal(ctle.transfer(ctx), np.ones(ctx.freq_grid().size, complex))


def test_rxffe_bypass_is_identity_and_preserves_taps():
    p = pipe()
    rx, ctx = p.by_name("rxffe"), p.ctx
    rx.set_taps(np.array([-0.1, 1.0, -0.2]), main_pos=1)  # auto-EQ-style override
    assert not np.allclose(rx.transfer(ctx), 1.0)
    rx.set_params(enabled="off")
    assert np.array_equal(rx.transfer(ctx), np.ones(ctx.freq_grid().size, complex))
    # toggling did not clear the solved taps
    rx.set_params(enabled="on")
    assert np.allclose(rx.taps(), [-0.1, 1.0, -0.2])


def test_txffe_two_independent_toggles():
    p = pipe(**{"txffe": {"pre": -0.1, "post": -0.15, "tr_ui": 0.4}})
    tx, ctx = p.by_name("txffe"), p.ctx
    ffe, drv = tx.ffe_transfer(ctx), tx.driver_transfer(ctx)
    # both on -> product
    assert np.allclose(tx.transfer(ctx), ffe * drv)
    # FFE off -> just the driver bandwidth
    tx.set_params(ffe_enabled="off", driver_enabled="on")
    assert np.allclose(tx.transfer(ctx), drv)
    # driver off -> just the FFE de-emphasis
    tx.set_params(ffe_enabled="on", driver_enabled="off")
    assert np.allclose(tx.transfer(ctx), ffe)
    # both off -> identity launch
    tx.set_params(ffe_enabled="off", driver_enabled="off")
    assert np.allclose(tx.transfer(ctx), 1.0)


def test_bypass_propagates_through_the_cascade():
    """Disabling CTLE in the pipeline equals removing its transfer from the cascade."""
    p = pipe(**{"ctle": {"dc_gain": -3.0, "fz": 0.35}})
    casc_on = STAT.cascade(p)
    p.by_name("ctle").set_params(enabled="off")
    casc_off = STAT.cascade(p)
    # the channel-only trace is unchanged; the RX trace drops the CTLE peaking
    assert np.allclose(casc_on.H_channel, casc_off.H_channel)
    assert not np.allclose(casc_on.H_tx_chan_rx, casc_off.H_tx_chan_rx)


# --------------------------------------------------------------------------- #
# DFE: a true bypass contributes no feedback even when the kernel runs for CDR
# --------------------------------------------------------------------------- #
def test_dfe_bypass_is_lti_static_clock():
    p = pipe()
    d = p.by_name("dfe")
    d.set_taps(np.array([0.06, 0.03]))   # non-trivial feedback...
    d.set_params(enabled="off")          # ...but bypassed
    r_off = TRAN.run_batch(p, n_symbols=20_000, rng=np.random.default_rng(1))
    # reference: the same link with the DFE genuinely absent (no taps)
    d.set_taps(np.zeros(0))
    d.set_params(enabled="on")
    r_ref = TRAN.run_batch(p, n_symbols=20_000, rng=np.random.default_rng(1))
    assert np.allclose(r_off.density, r_ref.density)
    assert r_off.ser == pytest.approx(r_ref.ser, abs=1e-12)


def test_dfe_bypass_zeroes_feedback_with_cdr_running():
    p = pipe(**{"cdr_slicer": {"cdr_mode": "mueller-muller"}})
    d = p.by_name("dfe")
    d.set_taps(np.array([0.06, 0.03]))
    d.set_params(enabled="off")
    r_off = TRAN.run_batch(p, n_symbols=20_000, rng=np.random.default_rng(2))
    # reference: kernel still runs for the CDR, but the DFE contributes nothing
    d.set_taps(np.zeros(0))
    d.set_params(enabled="on")
    r_ref = TRAN.run_batch(p, n_symbols=20_000, rng=np.random.default_rng(2))
    assert np.allclose(r_off.density, r_ref.density)


def test_dfe_toggle_preserves_solved_taps():
    p = pipe()
    d = p.by_name("dfe")
    d.set_taps(np.array([0.05, 0.02, 0.01]))
    d.set_params(enabled="off")
    assert np.allclose(d.taps(), [0.05, 0.02, 0.01])  # solver output not cleared


# --------------------------------------------------------------------------- #
# eye height at a specific phase column (item 2)
# --------------------------------------------------------------------------- #
def test_eye_height_at_col_matches_best_column():
    p = pipe()
    optimize_link(p)  # open the eye so the opening is non-zero
    _, sbr, eye = STAT.compute(p)
    h_best, best_pi = STAT._eye_height(eye.pdf, eye.v, sbr.main_cursor, p.ctx.levels)
    h_at = STAT._eye_height_at_col(eye.pdf, eye.v, sbr.main_cursor, p.ctx.levels, best_pi)
    assert h_best > 0
    assert h_at == pytest.approx(h_best)


def test_eye_height_at_col_clamps_out_of_range():
    p = pipe()
    _, sbr, eye = STAT.compute(p)
    # an out-of-range column index is clamped, not an IndexError
    val = STAT._eye_height_at_col(eye.pdf, eye.v, sbr.main_cursor, p.ctx.levels, 10_000)
    assert val >= 0.0


# --------------------------------------------------------------------------- #
# assessment target BER is honoured (item 3 marker / item 4 report row)
# --------------------------------------------------------------------------- #
def test_assess_respects_target_ber():
    p = pipe()
    assert assess(STAT, p, target_ber=1e-9).target_ber == 1e-9
    assert assess(STAT, p, target_ber=1e-12).target_ber == 1e-12
