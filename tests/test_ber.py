"""BER, bathtub curves, and COM from the statistical eye (Phase 5a)."""

import numpy as np
import pytest

from eyeq.analysis.ber import assess
from eyeq.analysis.optimize import optimize_link
from eyeq.engines import StatisticalEngine, TransientEngine
from eyeq.io import build_pipeline, default_link_config

STAT, TRAN = StatisticalEngine(), TransientEngine()


def pipe(mod="PAM4", reach="VSR", noise=3.0, **ov):
    p = build_pipeline(default_link_config(modulation=mod, reach_class=reach))
    p.apply_params({
        "ctle": {"fz": 0.35, "fp": 1.0, "dc_gain": -2.0},
        "txffe": {"pre": -0.08, "post": -0.12},
        "noise": {"sigma_mvrms": noise},
    })
    if ov:
        p.apply_params(ov)
    return p


# --------------------------------------------------------------------------- #
# BER vs the transient SER (the cross-engine check)
# --------------------------------------------------------------------------- #
def test_ber_matches_transient_ser_on_marginal_eye():
    # A closed VSR eye produces a measurable error rate both ways.
    p = pipe("PAM4", "VSR")
    _, sbr, eye = STAT.compute(p)
    r = assess(STAT, p, target_ber=1e-6)
    ts = TRAN.run_batch(p, n_symbols=300_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0)).ser
    assert r.ser == pytest.approx(ts, rel=0.5)  # same SER to within Monte Carlo


def test_ber_drops_after_auto_eq():
    p = pipe("PAM4", "VSR")
    before = assess(STAT, p).ser
    optimize_link(p)
    after = assess(STAT, p).ser
    assert before > 1e-3            # closed before EQ
    assert after < before / 1e3     # auto-EQ drops it by orders of magnitude


# --------------------------------------------------------------------------- #
# COM
# --------------------------------------------------------------------------- #
def test_com_sign_tracks_eye():
    closed = assess(STAT, pipe("PAM4", "VSR"))               # closed
    p = pipe("PAM4", "VSR"); optimize_link(p)
    open_eye = assess(STAT, p)                               # opened by auto-EQ
    assert closed.com_db < 0 < open_eye.com_db


def test_com_improves_with_auto_eq():
    p = pipe("PAM4", "LR")
    before = assess(STAT, p).com_db
    optimize_link(p)
    assert assess(STAT, p).com_db > before


# --------------------------------------------------------------------------- #
# eye margins / bathtubs
# --------------------------------------------------------------------------- #
def test_open_eye_has_positive_margins():
    p = pipe("NRZ", "XSR")
    r = assess(STAT, p, target_ber=1e-6)
    assert r.eye_height_v > 0 and r.eye_width_ui > 0
    assert r.ser < 1e-6


def test_closed_eye_has_zero_margins():
    r = assess(STAT, pipe("PAM4", "VSR"))
    assert r.eye_height_v == 0.0 and r.eye_width_ui == 0.0


def test_tighter_target_shrinks_the_eye():
    p = pipe("NRZ", "XSR")
    loose = assess(STAT, p, target_ber=1e-3)
    tight = assess(STAT, p, target_ber=1e-12)
    assert tight.eye_height_v <= loose.eye_height_v
    assert tight.eye_width_ui <= loose.eye_width_ui


def test_bathtub_shapes_and_minimum():
    p = pipe("NRZ", "XSR")
    r = assess(STAT, p, phase_points=65)
    assert r.h_bathtub.shape == r.t_axis.shape == (65,)
    assert r.v_bathtub.shape == r.v_eye.shape
    # the horizontal bathtub bottoms out at the best sampling phase
    assert r.t_axis[int(np.argmin(r.h_bathtub))] == pytest.approx(r.best_phase_ui, abs=1e-6)
